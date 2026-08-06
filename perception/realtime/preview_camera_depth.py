"""
Live camera -> DA3 depth preview. Eyeball the output, read off the frame rate.

Three panels, left to right:
  1. MODEL INPUT   the exact 504x280 image the network is fed (not the raw
                   camera frame -- if --match_train_blur is on you are looking
                   at the 320x180 bottleneck too)
  2. DEPTH         colourised prediction, per-frame normalised
  3. POLICY OBS    the 64x64 the nav policy would actually receive, shown
                   nearest-neighbour so you see the real pixelation

⚠️ The depth panel is RELATIVE. The model was fine-tuned with metric_weight=0
(scale-invariant loss only), so brightness means "far IN THIS FRAME" and the
scale drifts between frames. Do not read distances off it.

⚠️ The FPS shown here is LOWER than the true pipeline rate -- it includes
capture, colourisation and window drawing. Quote bench_da3.py for speed; this
number is "what the preview manages".

What to look for, given there is no ground truth (see ITERATION_HISTORY.md §3
-- the zero-shot model's specific failure was putting the deep region on a
brightly lit wall instead of the off-centre lumen hole):
  - aim down a tube with a bend: does the deepest region track the HOLE, or
    does it follow whatever is brightest?
  - move toward a wall: does near/far ordering stay correct?
  - specular glare: does a highlight punch a fake deep hole in the depth?

Usage -- just Run it; no arguments needed, and it relaunches itself under
Python 3.11 if the IDE is pointed at a different interpreter.
    python preview_camera_depth.py                       # live, scope on index 1
    python preview_camera_depth.py --cam_w 1280 --cam_h 720
    python preview_camera_depth.py --still sample_frame.png
    python preview_camera_depth.py --no_display --frames 200
Keys: q or ESC quit, s save the current panel.

Three measurements bracket the pipeline -- run all three to see where the
time goes:
    bench_da3.py --source camera   depth only (frames replayed from RAM)
    this, --no_display             capture + depth
    this, with the window          capture + depth + display
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _py311  # noqa: E402,F401  -- relaunches under Python 3.11 if needed

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from da3_runtime import Da3Depth, DEFAULT_CHECKPOINT, TARGET_H, sync  # noqa: E402
from probe_camera import DEFAULT_INDEX, open_camera, require_cv2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_ASPECT = 320 / 180


def _obs_res_arg(s):
    """--obs_res accepts "HxW" (e.g. 54x96) or a single int meaning square.

    HEIGHT x WIDTH, matching numpy's .shape and the env's
    (DEFAULT_DEPTH_RES_H, DEFAULT_DEPTH_RES_W) -- not the WxH images use.
    """
    s = s.strip().lower()
    if 'x' in s:
        h, w = s.split('x', 1)
        return (int(h), int(w))
    n = int(s)
    return (n, n)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int, default=DEFAULT_INDEX,
                   help=f'Camera index. Default: {DEFAULT_INDEX} (the scope).')
    p.add_argument('--backend', default='auto',
                   choices=['auto', 'dshow', 'msmf', 'any'])
    p.add_argument('--cam_w', type=int, default=None, help='Requested mode width.')
    p.add_argument('--cam_h', type=int, default=None)
    p.add_argument('--still', default=None,
                   help='Run on one image file instead of the camera.')
    p.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                   help="Fine-tuned .pth. Pass '' for zero-shot comparison.")
    p.add_argument('--precision', default='bf16', choices=['fp32', 'bf16', 'fp16'])
    p.add_argument('--device', default='auto')
    p.add_argument('--no_train_blur', action='store_true',
                   help="Skip the 320x180 bottleneck that matches the fine-tune "
                        "data's sharpness. Turn on to see how much it matters.")
    p.add_argument('--no_crop', action='store_true',
                   help='Do not centre-crop to 16:9 first. Only sensible if the '
                        'camera already delivers 16:9.')
    p.add_argument('--obs_res', type=_obs_res_arg, default=(54, 96),
                   help='policy depth-obs size as HxW, or one int for square '
                        '(default: 54x96, 16:9 -- matches navigation/v8_p1 '
                        'DEFAULT_DEPTH_RES since 2026-08-04)')
    p.add_argument('--scale', type=float, default=1.0, help='Display scale.')
    p.add_argument('--no_display', action='store_true',
                   help='No window: per-frame timings to the terminal instead. '
                        'Measures capture + depth with zero display cost -- the '
                        'configuration a deployed loop would actually run.')
    p.add_argument('--frames', type=int, default=0,
                   help='Stop after N frames (0 = until q / Ctrl-C). Useful '
                        'with --no_display, where there is no key to press.')
    p.add_argument('--print_every', type=int, default=1,
                   help='--no_display: print every Nth frame. Console I/O sits '
                        'inside the timed loop, so thin it out if it starts to '
                        'cost at high frame rates.')
    p.add_argument('--save_dir', default=os.path.join(HERE, 'preview_captures'))
    return p.parse_args()


def centre_crop_16_9(frame):
    """Crop to 16:9 about the centre. Training renders were 320x180; feeding a
    4:3 frame straight to a 504x280 resize distorts differently than training
    did, which the model reads as geometry."""
    h, w = frame.shape[:2]
    if abs((w / h) - TRAIN_ASPECT) < 0.02:
        return frame
    if (w / h) > TRAIN_ASPECT:          # too wide -> trim sides
        new_w = int(round(h * TRAIN_ASPECT))
        x0 = (w - new_w) // 2
        return frame[:, x0:x0 + new_w]
    new_h = int(round(w / TRAIN_ASPECT))  # too tall -> trim top/bottom
    y0 = (h - new_h) // 2
    return frame[y0:y0 + new_h, :]


def colourise(cv2, depth):
    """Per-frame normalised INFERNO. Percentile clip so one hot pixel does not
    flatten the rest of the map."""
    d = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(d)
    if not finite.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo = float(np.percentile(d[finite], 2))
    hi = float(np.percentile(d[finite], 98))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    norm[~finite] = 0.0
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def label(cv2, img, text, y=18):
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def build_panel(cv2, model_pil, depth, obs, stats_text):
    """MODEL INPUT | DEPTH | POLICY OBS, with a stats strip underneath."""
    rgb = np.asarray(model_pil, dtype=np.uint8)[:, :, ::-1]     # RGB -> BGR
    rgb = np.ascontiguousarray(rgb)
    dep = colourise(cv2, depth)

    o = (np.clip(obs[0], 0, 1) * 255).astype(np.uint8)
    # Preserve the obs's own aspect: it is 16:9 (54x96) since 2026-08-04, and
    # blowing it up to a square here would make the preview lie about the very
    # geometry this change exists to fix. Height is pinned to TARGET_H so the
    # three panes still concatenate; only the width follows the obs.
    _o_w = max(1, int(round(TARGET_H * obs.shape[2] / obs.shape[1])))
    o = cv2.resize(o, (_o_w, TARGET_H), interpolation=cv2.INTER_NEAREST)
    o = cv2.applyColorMap(o, cv2.COLORMAP_INFERNO)

    label(cv2, rgb, 'MODEL INPUT 504x280')
    label(cv2, dep, 'DEPTH (relative, per-frame norm)')
    label(cv2, o, f'POLICY OBS {obs.shape[1]}x{obs.shape[2]}')

    panel = np.concatenate([rgb, dep, o], axis=1)
    strip = np.zeros((26, panel.shape[1], 3), dtype=np.uint8)
    label(cv2, strip, stats_text)
    return np.concatenate([panel, strip], axis=0)


def median_ms(buf):
    return float(np.median(buf)) if buf else 0.0


def main():
    args = parse_args()
    cv2 = require_cv2()

    engine = Da3Depth(
        checkpoint=args.checkpoint or None,
        device=args.device,
        precision=args.precision,
        match_train_blur=not args.no_train_blur,
    )
    info = engine.describe()
    print(f"{info['gpu']}  {info['precision']}  "
          f"train_blur={info['match_train_blur']}  "
          f"loaded in {info['load_seconds']}s")
    if not args.checkpoint:
        print('⚠ running ZERO-SHOT (no fine-tuned head) -- expect the flexure '
              'failure described in ITERATION_HISTORY.md §3')

    os.makedirs(args.save_dir, exist_ok=True)
    win = 'DA3 preview  [q]uit  [s]ave'

    # ---- still mode ----
    if args.still:
        if not os.path.exists(args.still):
            sys.exit(f'No such file: {args.still}')
        frame = np.asarray(Image.open(args.still).convert('RGB'))
        if not args.no_crop:
            frame = centre_crop_16_9(frame)
        t0 = time.perf_counter()
        pil = engine.to_model_pil(frame)
        depth = engine.postprocess(engine.forward(engine.preprocess(frame)))
        sync(engine.device)
        dt = (time.perf_counter() - t0) * 1e3
        obs = engine.to_obs_64(depth, res=args.obs_res)
        panel = build_panel(cv2, pil, depth, obs,
                            f'still: {os.path.basename(args.still)}   {dt:.1f} ms')
        out = os.path.join(args.save_dir, 'still_panel.png')
        cv2.imwrite(out, panel)
        print(f'{dt:.1f} ms   panel -> {out}')
        cv2.imshow(win, panel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ---- live mode ----
    cap, backend = open_camera(cv2, args.index, args.backend, args.cam_w, args.cam_h)
    if cap is None:
        sys.exit(f'Could not open camera index {args.index}. '
                 f'Run probe_camera.py to see what is available.')
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        sys.exit('Camera opened but delivered no frame.')
    h, w = frame.shape[:2]
    print(f'camera {args.index} [{backend}]  {w}x{h}  aspect {w/h:.3f}')
    if not args.no_crop and abs((w / h) - TRAIN_ASPECT) >= 0.02:
        print(f'  centre-cropping to 16:9 to match the {TRAIN_ASPECT:.3f} '
              f'training aspect')

    cap_ms, inf_ms, tot_ms = [], [], []
    n_saved = 0
    n = 0
    if args.no_display:
        print(f'headless -- capture + depth, no window'
              + (f', {args.frames} frames' if args.frames else ', Ctrl-C to stop'))
    else:
        print('running -- q or ESC to quit, s to save')
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                print('dropped frame')
                continue
            t1 = time.perf_counter()

            if not args.no_crop:
                frame = centre_crop_16_9(frame)

            # Build the model input once. Displaying it and feeding it are the
            # same PIL, so the resize chain is never paid twice.
            pil = engine.to_model_pil(frame, bgr=True)
            depth = engine.postprocess(engine.forward(engine.preprocess_pil(pil)))
            sync(engine.device)
            t2 = time.perf_counter()
            obs = engine.to_obs_64(depth, res=args.obs_res)

            cap_ms.append((t1 - t0) * 1e3)
            inf_ms.append((t2 - t1) * 1e3)

            if args.no_display:
                if args.print_every <= 1 or n % args.print_every == 0:
                    loop = median_ms(tot_ms)
                    print(f'frame {n:6d}  capture {cap_ms[-1]:6.1f} ms | '
                          f'depth {inf_ms[-1]:6.1f} ms | '
                          f'total {tot_ms[-1] if tot_ms else 0:6.1f} ms | '
                          f'{1000.0/max(loop, 1e-6):5.1f} fps',
                          flush=True)
            else:
                stats = (f'capture {median_ms(cap_ms):5.1f} ms | '
                         f'depth {median_ms(inf_ms):5.1f} ms | '
                         f'loop {median_ms(tot_ms):5.1f} ms | '
                         f'{1000.0/max(median_ms(tot_ms), 1e-6):4.1f} fps '
                         f'(incl. display)')
                panel = build_panel(cv2, pil, depth, obs, stats)
                if args.scale != 1.0:
                    panel = cv2.resize(panel, None, fx=args.scale, fy=args.scale,
                                       interpolation=cv2.INTER_AREA)
                cv2.imshow(win, panel)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('s'):
                    p = os.path.join(args.save_dir, f'panel_{n_saved:03d}.png')
                    cv2.imwrite(p, panel)
                    print(f'  saved {p}')
                    n_saved += 1

            tot_ms.append((time.perf_counter() - t0) * 1e3)
            n += 1
            if args.frames and n >= args.frames:
                break
            for buf in (cap_ms, inf_ms, tot_ms):   # rolling window
                if len(buf) > 120:
                    del buf[:-120]
    except KeyboardInterrupt:
        print()          # so the summary below still prints on Ctrl-C
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if tot_ms:
        print(f'\nmedians over last {len(tot_ms)} frames:')
        print(f'  capture      {median_ms(cap_ms):6.2f} ms')
        print(f'  depth        {median_ms(inf_ms):6.2f} ms')
        print(f'  loop total   {median_ms(tot_ms):6.2f} ms  '
              f'-> {1000.0/max(median_ms(tot_ms), 1e-6):.1f} fps')
        if args.no_display:
            print('  (capture + depth, no display. bench_da3.py --source camera '
                  'gives depth alone.)')
        else:
            print('  (loop includes colourisation + window draw; '
                  'rerun with --no_display to drop it)')


if __name__ == '__main__':
    main()
