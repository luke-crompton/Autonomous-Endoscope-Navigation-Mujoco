"""
Probe a real camera and report what it ACTUALLY delivers.

Cameras lie: `cap.set(CAP_PROP_FRAME_WIDTH, ...)` silently succeeds and then
hands back a different resolution, and the reported FPS is often the mode's
nominal rate rather than what the link sustains. Everything printed here is
measured from delivered frames, not from what the driver claims.

Also MEASURES the lens field of view (--measure_fov), and converts a datasheet
FOV between diagonal / horizontal / vertical. Endoscope and borescope specs
almost always quote DIAGONAL, while the DA3 fine-tune was rendered at 100
degrees HORIZONTAL (blender_depth_dataset.py:50, Blender lens_unit='FOV' on a
16:9 sensor fit). Confusing the two is a systematic depth-scale error, not
visible noise -- and the 100 degrees itself is an assumption nobody has checked
against the real lens, which is what --measure_fov exists to settle.

Usage -- just Run it; no arguments needed, and it relaunches itself under
Python 3.11 if the IDE is pointed at a different interpreter.
    python probe_camera.py                          # measure the scope (index 1)
    python probe_camera.py --enumerate              # list every camera present
    python probe_camera.py --index 0 --seconds 5    # measure a different camera
    python probe_camera.py --no-modes               # skip the mode sweep
    python probe_camera.py --fov_diag 120           # spec conversion only

    python probe_camera.py --measure_fov target         # ruler at known distance
    python probe_camera.py --measure_fov checkerboard   # full calibration
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _py311  # noqa: E402,F401  -- relaunches under Python 3.11 if needed

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# What the depth model was fine-tuned on -- the target to match.
TRAIN_W, TRAIN_H = 320, 180
TRAIN_FOV_H_DEG = 100.0

# The scope. Index 0 is the laptop webcam and index 2 is a broken NVIDIA
# Broadcast virtual camera -- confirmed by --enumerate, see camera_probe.json.
DEFAULT_INDEX = 1

# Modes worth trying on a scope camera, widest-relevant first.
CANDIDATE_MODES = [
    (1920, 1080), (1280, 720), (1024, 576), (800, 600),
    (640, 480), (640, 360), (320, 240), (320, 180),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int, default=None,
                   help=f'Camera index to measure. Default: {DEFAULT_INDEX} '
                        f'(the scope).')
    p.add_argument('--enumerate', action='store_true',
                   help='List every camera present instead of measuring one.')
    p.add_argument('--backend', default='auto',
                   choices=['auto', 'dshow', 'msmf', 'any'],
                   help='Windows: DirectShow and Media Foundation disagree about '
                        'available modes. auto = try dshow then msmf.')
    p.add_argument('--max_index', type=int, default=4,
                   help='Highest index to probe when enumerating.')
    p.add_argument('--seconds', type=float, default=3.0,
                   help='Sustained-capture measurement window.')
    p.add_argument('--width', type=int, default=None)
    p.add_argument('--height', type=int, default=None)
    p.add_argument('--modes', action=argparse.BooleanOptionalAction, default=True,
                   help='Try every candidate mode and report what is delivered.')
    p.add_argument('--save_frame', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Write one frame to sample_frame.png for eyeballing FOV.')
    p.add_argument('--fov_diag', type=float, default=None,
                   help='Datasheet DIAGONAL FOV in degrees -> convert and compare.')

    g = p.add_argument_group(
        'FOV measurement',
        'Measure the real lens instead of trusting a datasheet. '
        'Writes camera_fov.json.')
    g.add_argument('--measure_fov', choices=['target', 'checkerboard'],
                   default=None,
                   help='target = one shot against a ruler at a known distance. '
                        'checkerboard = full pinhole+fisheye calibration.')
    g.add_argument('--target_mm', type=float, default=None,
                   help='target: real separation of the two clicked points, mm. '
                        'Prompted for if omitted.')
    g.add_argument('--distance_mm', type=float, default=None,
                   help='target: lens-to-target distance, mm. Prompted for if '
                        'omitted.')
    g.add_argument('--board_cols', type=int, default=9,
                   help='checkerboard: INNER corners across (10 squares -> 9).')
    g.add_argument('--board_rows', type=int, default=6,
                   help='checkerboard: INNER corners down (7 squares -> 6).')
    g.add_argument('--square_mm', type=float, default=25.0,
                   help='checkerboard: square size. Does not affect FOV.')
    g.add_argument('--shots', type=int, default=12,
                   help='checkerboard: views to collect.')
    g.add_argument('--fov_out', default=os.path.join(HERE, 'camera_fov.json'))

    p.add_argument('--out', default=os.path.join(HERE, 'camera_probe.json'))
    return p.parse_args()


def require_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        sys.exit(
            'opencv-python is not installed in this interpreter.\n'
            '  Use the Python 3.11 env (see README interpreter matrix) and:\n'
            '  python -m pip install opencv-python'
        )


def backends(cv2, choice):
    table = {'dshow': cv2.CAP_DSHOW, 'msmf': cv2.CAP_MSMF, 'any': cv2.CAP_ANY}
    if choice == 'auto':
        return [('dshow', cv2.CAP_DSHOW), ('msmf', cv2.CAP_MSMF)]
    return [(choice, table[choice])]


def fourcc_str(v):
    v = int(v)
    if v <= 0:
        return '?'
    return ''.join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


# ------------------------- FOV maths --------------------------------------

def fov_convert(fov_deg, kind, aspect_w, aspect_h):
    """Convert between diagonal / horizontal / vertical FOV for a rectilinear lens.

    Rectilinear (pinhole) assumption. A wide endoscope lens is usually
    fisheye-ish, where this UNDERSTATES the true angular coverage -- so treat
    the output as a first check, not a calibration.
    """
    diag = math.hypot(aspect_w, aspect_h)
    dims = {'d': diag, 'h': aspect_w, 'v': aspect_h}
    ref = dims[kind]
    f = ref / (2.0 * math.tan(math.radians(fov_deg) / 2.0))   # focal, same units
    return {k: math.degrees(2.0 * math.atan(d / (2.0 * f))) for k, d in dims.items()}


def report_fov(fov_diag, aspect_w=16, aspect_h=9):
    conv = fov_convert(fov_diag, 'd', aspect_w, aspect_h)
    print(f'\nFOV conversion for a {aspect_w}:{aspect_h} sensor '
          f'(rectilinear assumption):')
    print(f"  diagonal   {conv['d']:6.1f} deg   <- datasheet value")
    print(f"  horizontal {conv['h']:6.1f} deg   <- compare to training's "
          f'{TRAIN_FOV_H_DEG:.0f} deg')
    print(f"  vertical   {conv['v']:6.1f} deg")
    delta = conv['h'] - TRAIN_FOV_H_DEG
    if abs(delta) < 3:
        print(f'  -> within {abs(delta):.1f} deg of the fine-tune. Good match.')
    else:
        print(f'  -> {abs(delta):.1f} deg {"wider" if delta > 0 else "narrower"} '
              f'than the fine-tune. Depth will carry a systematic scale error; '
              f'either re-render the dataset at this FOV or crop/pad to match.')
    return conv


def compare_to_training(fov_h, label='measured'):
    """One place that decides whether a horizontal FOV matches the fine-tune."""
    delta = fov_h - TRAIN_FOV_H_DEG
    print(f'\n  {label} horizontal {fov_h:.1f} deg vs fine-tune '
          f'{TRAIN_FOV_H_DEG:.0f} deg -> {abs(delta):.1f} deg '
          f'{"wider" if delta > 0 else "narrower"}')
    if abs(delta) < 3:
        print('  Good match. No dataset re-render needed on the FOV account.')
    else:
        print('  Mismatch. The DA3 fine-tune was rendered at '
              f'{TRAIN_FOV_H_DEG:.0f} deg horizontal, so depth carries a '
              'systematic scale error until the dataset is re-rendered at this '
              'FOV or the frame is cropped to match.')
    print('  Feed the VERTICAL number into the sim camera: MuJoCo fovy is '
          'vertical (build_collision_scene.py, currently fovy="85").')
    return delta


# ------------------------- FOV measurement --------------------------------
#
# Two methods, because they answer different questions:
#
#   target        one shot against a ruler at a known distance. Fast, needs no
#                 printed pattern, and if the span you click reaches the frame
#                 edges it measures the true edge-to-edge coverage with NO lens
#                 model at all. That is the number the sim camera needs.
#
#   checkerboard  full cv2 calibration, pinhole AND fisheye, ~12 views. Slower,
#                 but it is the only one that reports distortion, and for a
#                 100-degree-plus endoscope lens the fisheye fit is usually the
#                 honest one. Also gives you K, which nothing else here does.
#
# For a strongly distorting lens the two will disagree, and that disagreement is
# information: it is the size of the rectilinear assumption's error.

def _edge_angles(cv2, K, D, size, fisheye):
    """Angular coverage of the frame edges, distortion included.

    Undistorts the edge-midpoint and corner pixels back to normalised camera
    coordinates; atan of the radius there IS the incidence angle of the ray
    that lands on that pixel. Deriving FOV this way rather than from fx/fy
    matters here: for a fisheye lens the 2*atan(w/2f) formula understates true
    coverage badly, which is exactly the 100-vs-110 question we are trying to
    settle.
    """
    w, h = size
    pts = np.array([
        [0.0, h / 2.0], [w - 1.0, h / 2.0],        # left, right
        [w / 2.0, 0.0], [w / 2.0, h - 1.0],        # top, bottom
        [0.0, 0.0], [w - 1.0, h - 1.0],            # corners -> diagonal
    ], dtype=np.float64).reshape(-1, 1, 2)
    und = (cv2.fisheye.undistortPoints(pts, K, D) if fisheye
           else cv2.undistortPoints(pts, K, D))
    a = np.degrees(np.arctan(np.hypot(*und.reshape(-1, 2).T)))
    return {'h': float(a[0] + a[1]),
            'v': float(a[2] + a[3]),
            'd': float(a[4] + a[5])}


def _grab_still(cv2, cap, title):
    """Live view until SPACE freezes a frame. Returns None if aborted."""
    print(f'  {title}')
    print('  SPACE = freeze this frame,  q / ESC = abort')
    win = 'probe_camera  [SPACE] freeze  [q] abort'
    frozen = None
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        view = frame.copy()
        # Guides: centre cross plus the extreme columns/rows, so you can line a
        # ruler up with the actual frame edge rather than guessing.
        cv2.line(view, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
        cv2.line(view, (0, h // 2), (w, h // 2), (0, 255, 0), 1)
        for x in (1, w - 2):
            cv2.line(view, (x, 0), (x, h), (0, 0, 255), 2)
        for y in (1, h - 2):
            cv2.line(view, (0, y), (w, y), (0, 0, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            frozen = frame
            break
        if key in (ord('q'), 27):
            break
    cv2.destroyWindow(win)
    return frozen


def _click_two_points(cv2, frame):
    """Click the two points whose real-world separation you measured."""
    pts = []
    win = 'click the two target points  [u] undo  [ENTER] accept  [q] abort'

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print('  Click the LEFT point, then the RIGHT point. ENTER accepts, '
          'u undoes, q aborts.')
    while True:
        view = frame.copy()
        for i, (x, y) in enumerate(pts):
            cv2.drawMarker(view, (x, y), (0, 255, 255),
                           cv2.MARKER_CROSS, 18, 2)
            cv2.putText(view, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        if len(pts) == 2:
            cv2.line(view, pts[0], pts[1], (0, 255, 255), 1)
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(pts) == 2:
            break
        if key == ord('u') and pts:
            pts.pop()
        if key in (ord('q'), 27):
            pts = []
            break
    cv2.destroyWindow(win)
    return pts


def _ask_float(prompt, preset):
    if preset is not None:
        print(f'  {prompt} {preset}')
        return float(preset)
    while True:
        try:
            return float(input(f'  {prompt} ').strip())
        except ValueError:
            print('  Enter a number.')


def measure_fov_target(cv2, args, index):
    """Ruler at a known distance -> horizontal/vertical/diagonal FOV.

    Setup: tape a ruler or printed scale to a wall, perpendicular to the
    optical axis, centred, and measure lens-to-wall distance. Then click two
    marks whose separation you know.

    Push the two clicks as close to the left and right frame EDGES as you can.
    A span clicked near the centre and extrapolated outward assumes the lens is
    rectilinear, and a wide endoscope lens is not -- it will read low.
    """
    cap, backend = open_camera(cv2, index, args.backend, args.width, args.height)
    if cap is None:
        sys.exit(f'Could not open camera index {index}.')
    frame = _grab_still(
        cv2, cap,
        'Aim at the ruler, square-on and centred. Red lines are the frame edges.')
    cap.release()
    if frame is None:
        sys.exit('Aborted.')

    h, w = frame.shape[:2]
    pts = _click_two_points(cv2, frame)
    cv2.destroyAllWindows()
    if len(pts) != 2:
        sys.exit('Aborted.')

    px = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
    if px < 2:
        sys.exit('The two points are the same. Click them further apart.')
    span_mm = _ask_float('Real distance between the two points, in mm:',
                         args.target_mm)
    dist_mm = _ask_float('Lens-to-target distance, in mm:', args.distance_mm)
    if span_mm <= 0 or dist_mm <= 0:
        sys.exit('Both measurements must be positive.')

    shot = os.path.join(HERE, 'fov_target_shot.png')
    cv2.imwrite(shot, frame)

    # Pinhole reading: pixels-per-mm at that distance gives fx directly.
    fx = px * dist_mm / span_mm
    out = {
        'method': 'target',
        'camera_index': index,
        'backend': backend,
        'frame': [w, h],
        'points': [list(map(int, pts[0])), list(map(int, pts[1]))],
        'span_px': round(px, 2),
        'span_mm': span_mm,
        'distance_mm': dist_mm,
        'focal_px': round(fx, 2),
        'shot': shot,
        'pinhole': {
            'h': math.degrees(2 * math.atan(w / (2 * fx))),
            'v': math.degrees(2 * math.atan(h / (2 * fx))),
            'd': math.degrees(2 * math.atan(math.hypot(w, h) / (2 * fx))),
        },
    }

    print(f'\n  span {px:.1f} px = {span_mm:.1f} mm at {dist_mm:.1f} mm '
          f'-> focal {fx:.1f} px')
    print('\n  Pinhole (rectilinear) reading:')
    for k, name in (('h', 'horizontal'), ('v', 'vertical'), ('d', 'diagonal')):
        print(f"    {name:<10} {out['pinhole'][k]:6.1f} deg")

    # If the clicks actually reached the frame edges, the horizontal FOV is
    # direct trigonometry on the measured span -- no lens model, no assumption.
    x0, x1 = sorted((pts[0][0], pts[1][0]))
    edge_tol = 0.02 * w
    if x0 <= edge_tol and x1 >= (w - 1 - edge_tol):
        direct = math.degrees(2 * math.atan(span_mm / (2 * dist_mm)))
        out['direct_edge_to_edge'] = {'h': direct}
        print(f'\n  Clicks reached the frame edges, so horizontal FOV can be '
              f'read directly:\n    horizontal {direct:6.1f} deg  '
              f'<- MODEL-FREE, trust this over the pinhole row')
        if abs(direct - out['pinhole']['h']) > 5:
            print(f"    ({abs(direct - out['pinhole']['h']):.1f} deg apart -- "
                  f"that gap is the lens's barrel distortion. Use "
                  f"--measure_fov checkerboard for the vertical number too.)")
        compare_to_training(direct, 'measured (edge-to-edge)')
    else:
        print('\n  Clicks did NOT reach the frame edges, so these numbers '
              'extrapolate outward\n  assuming a rectilinear lens. For a wide '
              'endoscope lens they read LOW.\n  Re-measure edge-to-edge, or '
              'use --measure_fov checkerboard.')
        compare_to_training(out['pinhole']['h'], 'measured (pinhole)')
    return out


def measure_fov_checkerboard(cv2, args, index):
    """Full calibration from ~12 checkerboard views: pinhole and fisheye.

    --board_cols/--board_rows are INNER corner counts, not squares: a board
    with 10x7 squares is 9x6 inner corners. --square_mm does not affect FOV
    (it only sets the scale of the extrinsics), but calibrateCamera needs it.
    """
    cols, rows = args.board_cols, args.board_rows
    need = args.shots
    print(f'\nCheckerboard calibration: {cols}x{rows} inner corners, '
          f'{args.square_mm} mm squares, {need} views.')
    print('Tilt the board between shots and push it into the CORNERS of the '
          'frame -- \ndistortion is only observable where the board reaches the '
          'edges, and that\nis precisely what the FOV number depends on.')

    cap, backend = open_camera(cv2, index, args.backend, args.width, args.height)
    if cap is None:
        sys.exit(f'Could not open camera index {index}.')

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * args.square_mm

    obj_pts, img_pts = [], []
    shots_dir = os.path.join(HERE, 'fov_checkerboard')
    os.makedirs(shots_dir, exist_ok=True)
    win = 'checkerboard  [SPACE] keep  [u] drop last  [ENTER] calibrate  [q] abort'
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    size = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = (frame.shape[1], frame.shape[0])
            found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
            view = frame.copy()
            if found:
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
                cv2.drawChessboardCorners(view, (cols, rows), corners, found)
            cv2.putText(view,
                        f'{len(img_pts)}/{need} kept   '
                        f'{"BOARD FOUND - SPACE to keep" if found else "no board"}',
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if found else (0, 0, 255), 2)
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key == 32 and found:
                obj_pts.append(objp.copy())
                img_pts.append(corners)
                cv2.imwrite(os.path.join(shots_dir,
                                         f'shot_{len(img_pts):02d}.png'), frame)
                print(f'  kept {len(img_pts)}/{need}')
            elif key == ord('u') and img_pts:
                obj_pts.pop(), img_pts.pop()
                print(f'  dropped -> {len(img_pts)}')
            elif key in (13, 10) or (len(img_pts) >= need and key == 32):
                if len(img_pts) >= 4:
                    break
                print('  need at least 4 views')
            elif key in (ord('q'), 27):
                sys.exit('Aborted.')
            if len(img_pts) >= need:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f'\nCalibrating on {len(img_pts)} views ({size[0]}x{size[1]})...')
    out = {'method': 'checkerboard', 'camera_index': index, 'backend': backend,
           'frame': list(size), 'views': len(img_pts),
           'board': [cols, rows], 'square_mm': args.square_mm,
           'shots_dir': shots_dir}

    rms, K, D, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    out['pinhole'] = {'rms_px': round(float(rms), 4),
                      'K': K.tolist(), 'dist': D.ravel().tolist(),
                      **_edge_angles(cv2, K, D, size, fisheye=False)}
    print(f"  pinhole  RMS {rms:.3f} px   "
          f"h {out['pinhole']['h']:.1f}  v {out['pinhole']['v']:.1f}  "
          f"d {out['pinhole']['d']:.1f} deg")

    # The fisheye model needs its own array layout and often refuses to
    # converge on marginal data; a failure here is informative, not fatal.
    try:
        fo = [o.reshape(-1, 1, 3).astype(np.float64) for o in obj_pts]
        fi = [p.reshape(-1, 1, 2).astype(np.float64) for p in img_pts]
        Kf = np.zeros((3, 3)); Df = np.zeros((4, 1))
        f_rms, Kf, Df, _, _ = cv2.fisheye.calibrate(
            fo, fi, size, Kf, Df,
            flags=(cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                   | cv2.fisheye.CALIB_FIX_SKEW),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                      100, 1e-6))
        out['fisheye'] = {'rms_px': round(float(f_rms), 4),
                          'K': Kf.tolist(), 'dist': Df.ravel().tolist(),
                          **_edge_angles(cv2, Kf, Df, size, fisheye=True)}
        print(f"  fisheye  RMS {f_rms:.3f} px   "
              f"h {out['fisheye']['h']:.1f}  v {out['fisheye']['v']:.1f}  "
              f"d {out['fisheye']['d']:.1f} deg")
    except cv2.error as e:
        out['fisheye'] = {'error': str(e).strip().splitlines()[-1]}
        print(f"  fisheye  FAILED: {out['fisheye']['error']}")
        print('  (usually too few views, or the board never reached the '
              'frame corners)')

    # Lower reprojection RMS = the model that actually describes this lens.
    best = 'pinhole'
    if 'rms_px' in out['fisheye'] and out['fisheye']['rms_px'] < out['pinhole']['rms_px']:
        best = 'fisheye'
    out['best_model'] = best
    b = out[best]
    print(f'\n  Lower reprojection error: {best.upper()} '
          f'({b["rms_px"]:.3f} px) -- use its numbers.')
    print(f"    horizontal {b['h']:6.1f} deg")
    print(f"    vertical   {b['v']:6.1f} deg   <- this is what sim fovy needs")
    print(f"    diagonal   {b['d']:6.1f} deg   <- compare to the datasheet")
    compare_to_training(b['h'], f'measured ({best})')
    return out


# ------------------------- Probing ----------------------------------------

def enumerate_cameras(cv2, max_index, backend_choice):
    print('Enumerating cameras...')
    found = []
    for name, api in backends(cv2, backend_choice):
        for idx in range(max_index + 1):
            cap = cv2.VideoCapture(idx, api)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                entry = {
                    'index': idx,
                    'backend': name,
                    'delivered_w': w,
                    'delivered_h': h,
                    'reported_fps': round(cap.get(cv2.CAP_PROP_FPS), 2),
                    'fourcc': fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)),
                }
                found.append(entry)
                print(f"  index {idx} [{name}]  {w}x{h}  "
                      f"fourcc={entry['fourcc']}  "
                      f"reported {entry['reported_fps']} fps")
            cap.release()
    if not found:
        print('  none found. Check the camera is connected and not held by '
              'another application.')
    return found


def open_camera(cv2, index, backend_choice, width, height):
    for name, api in backends(cv2, backend_choice):
        cap = cv2.VideoCapture(index, api)
        if not cap.isOpened():
            cap.release()
            continue
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, _ = cap.read()
        if ok:
            return cap, name
        cap.release()
    return None, None


def try_modes(cv2, index, backend_choice):
    """Request each candidate mode, report what actually comes back."""
    print('\nMode probe (requested -> delivered):')
    rows = []
    for (w, h) in CANDIDATE_MODES:
        cap, name = open_camera(cv2, index, backend_choice, w, h)
        if cap is None:
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            dh, dw = frame.shape[:2]
            match = 'ok' if (dw, dh) == (w, h) else 'SUBSTITUTED'
            rows.append({'requested': [w, h], 'delivered': [dw, dh],
                         'backend': name, 'status': match})
            print(f'  {w:>5}x{h:<5} -> {dw:>5}x{dh:<5} [{name}]  {match}')
        cap.release()
    return rows


def measure_capture(cv2, cap, seconds):
    """Sustained capture rate and per-frame read latency, from delivered frames."""
    print(f'\nMeasuring sustained capture for {seconds:.1f}s...')
    for _ in range(10):          # let auto-exposure/gain settle
        cap.read()

    lat, n_bad = [], 0
    t_end = time.perf_counter() + seconds
    t_start = time.perf_counter()
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok or frame is None:
            n_bad += 1
            continue
        lat.append((t1 - t0) * 1e3)
    wall = time.perf_counter() - t_start

    if not lat:
        return {'error': 'no frames captured'}
    a = np.asarray(lat)
    return {
        'frames': len(lat),
        'dropped_reads': n_bad,
        'sustained_fps': round(len(lat) / wall, 2),
        'read_median_ms': round(float(np.median(a)), 3),
        'read_p95_ms': round(float(np.percentile(a, 95)), 3),
        'read_max_ms': round(float(a.max()), 3),
    }


def main():
    args = parse_args()
    results = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}

    if args.fov_diag is not None:
        results['fov'] = report_fov(args.fov_diag)
        # --fov_diag on its own is a desk-side conversion -- don't make it
        # depend on the scope being plugged in.
        if args.index is None and not args.enumerate:
            with open(args.out, 'w') as f:
                json.dump(results, f, indent=2)
            print(f'\nWritten -> {args.out}')
            return

    cv2 = require_cv2()

    if args.enumerate:
        results['enumerated'] = enumerate_cameras(cv2, args.max_index, args.backend)
        print('\nRe-run with --index N to measure one of these.')
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Written -> {args.out}')
        return

    index = DEFAULT_INDEX if args.index is None else args.index

    # Measurement is its own job and writes its own file, so it does not
    # clobber camera_probe.json with a partial result.
    if args.measure_fov:
        fn = (measure_fov_target if args.measure_fov == 'target'
              else measure_fov_checkerboard)
        measured = {'timestamp': results['timestamp'], **fn(cv2, args, index)}
        with open(args.fov_out, 'w') as f:
            json.dump(measured, f, indent=2)
        print(f'\nWritten -> {args.fov_out}')
        return

    if args.modes:
        results['modes'] = try_modes(cv2, index, args.backend)

    cap, name = open_camera(cv2, index, args.backend, args.width, args.height)
    if cap is None:
        sys.exit(f'Could not open camera index {index}. '
                 f'Is another application holding it? '
                 f'Run with --enumerate to see what is present.')

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        sys.exit('Camera opened but delivered no frame.')
    h, w = frame.shape[:2]
    info = {
        'index': index,
        'backend': name,
        'delivered_w': w,
        'delivered_h': h,
        'aspect': round(w / h, 4),
        'reported_fps': round(cap.get(cv2.CAP_PROP_FPS), 2),
        'fourcc': fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    print(f"\nCamera {index} [{name}]: {w}x{h} "
          f"(aspect {info['aspect']}), fourcc={info['fourcc']}, "
          f"driver reports {info['reported_fps']} fps")

    train_aspect = TRAIN_W / TRAIN_H
    if abs(info['aspect'] - train_aspect) > 0.02:
        print(f'  ⚠ aspect {info["aspect"]:.3f} != training {train_aspect:.3f} '
              f'({TRAIN_W}x{TRAIN_H}). Resizing to 504x280 will distort '
              f'differently than it did in training -- crop to 16:9 first.')
    else:
        print(f'  aspect matches the {TRAIN_W}x{TRAIN_H} training renders.')

    results['camera'] = info
    results['capture'] = measure_capture(cv2, cap, args.seconds)
    c = results['capture']
    if 'error' not in c:
        print(f"  sustained {c['sustained_fps']} fps  "
              f"(read median {c['read_median_ms']:.2f} ms, "
              f"p95 {c['read_p95_ms']:.2f} ms, "
              f"{c['dropped_reads']} failed reads)")
        print(f"  capture alone leaves "
              f"{1000.0 / c['sustained_fps']:.1f} ms per frame before any "
              f"depth inference.")

    if args.save_frame:
        ok, frame = cap.read()
        if ok and frame is not None:
            path = os.path.join(HERE, 'sample_frame.png')
            cv2.imwrite(path, frame)
            print(f'  sample frame -> {path}')
            results['sample_frame'] = path

    cap.release()
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nWritten -> {args.out}')


if __name__ == '__main__':
    main()
