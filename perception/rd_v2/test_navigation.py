"""
Closed-loop nav test using the fine-tuned DA3 model as perception.

Architecture (Option D, hybrid):
  - MuJoCo: scope physics + base mocap advancement along the Blender colon's
    centreline. No colon mesh in MuJoCo (we don't render here).
  - Blender (persistent subprocess): renders the camera view at every step,
    matching the dataset distribution DA3 was trained on.
  - Main process: orchestrates, runs DA3, computes aim point from the depth
    field (centroid of top-5%-deepest pixels), applies proportional tendon
    control, advances base, steps physics.

Outputs:
  nav_test_output/frames/frame_XXXXXX.png      annotated RGB (crosshair on aim point)
  nav_test_output/control_log.csv              per-step state
  nav_test_output/nav_scene.xml                generated MuJoCo scene

Self-contained transfer version: generate_videoscope_one_section.py and
new8.stl are bundled in the same directory as this file. Set BLENDER_EXE
to the Blender executable path on the target machine.

First-run notes:
  - SIGN_X / SIGN_Y default to Stage-4's values. If on first run the scope
    moves the wrong way for an axis, flip the relevant sign and rerun.
  - The Blender colon convention (det = -1 in the cam-matrix) means the
    rendered image may be horizontally mirrored relative to MuJoCo's renderer;
    SIGN_X absorbs that.

Usage:
    python test_navigation.py
    python test_navigation.py --max_steps 500 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw


# ----- defaults (mirror Stage 4 controller_loop.py) ---------------------

DEFAULT_SEED = 42
DEFAULT_MAX_STEPS = 1500
IMG_W, IMG_H = 320, 180

DEADZONE = 0.08
K_P = 0.0008
K_D = 0.0008  # derivative gain; brakes when error is shrinking
RATE_MAX = 0.0003
MAX_PULL = 0.024
PHYSICS_STEPS_PER_CONTROL = 5
ADVANCE_RATE_M_PER_STEP = 0.0005   # full rate when aim is inside the deadzone
ADVANCE_RATE_OFF_TARGET = 0.0001   # reduced rate (20%) when aim is outside it
REVERSE_RATE = 0.0003              # backing up rate when stuck against a wall
SIGN_X = +1.0
SIGN_Y = +1.0

# "Stuck against a wall" detector. The fine-tuned DA3 outputs metric depth in
# metres. We detect stuck by checking the FRACTION of pixels with depth above
# a "this is a lumen-direction" threshold. A healthy lumen view has lots of
# deep pixels; a wall-jam has very few. The previous centre-window detector
# fired even when the lumen was visible but off-centre.
STUCK_DEEP_PIXEL_M = 0.050        # pixels deeper than this count as "lumen-direction"
STUCK_DEEP_FRACTION = 0.05        # if fewer than 5% qualify, scope is wall-jammed
STUCK_RELAX_FACTOR = 0.95         # mild multiplicative decay of cmd while stuck

# Hybrid advance: scope always moves forward, just slower when off-target.
# Pure conditional dwell caused dwell-oscillation (controller integrated while
# scope didn't move, repeatedly overshooting). Slowing-but-not-stopping
# preserves the "pause to look" feel without freezing the loop.

AIM_PERCENTILE = 95.0  # "centroid of top (100-AIM_PERCENTILE)% deepest pixels"

HERE = Path(__file__).parent
# Set BLENDER_EXE to the Blender executable on the target machine.
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
WORKER_SCRIPT = HERE / "blender_render_worker.py"
CHECKPOINT_PATH = HERE / "checkpoints" / "da3small_finetune_head.pth"

PYTHON_FOR_REFERENCE = sys.executable


# ----- centreline (same as worker; deterministic from seed) -------------

def cubic_interp(t_c, vals, t_o):
    n = len(t_c); r = np.zeros(len(t_o))
    for i in range(n - 1):
        m = (t_o >= t_c[i]) & (t_o <= t_c[min(i + 1, n - 1)])
        if i == n - 2:
            m = t_o >= t_c[i]
        if m.any():
            f = (t_o[m] - t_c[i]) / (t_c[i + 1] - t_c[i] + 1e-10)
            p0, p1 = vals[max(i - 1, 0)], vals[i]
            p2, p3 = vals[min(i + 1, n - 1)], vals[min(i + 2, n - 1)]
            r[m] = 0.5 * ((2 * p1) + (-p0 + p2) * f
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f**2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * f**3)
    return r


def random_centerline(seed):
    np.random.seed(seed)
    n = np.random.randint(5, 10)
    L = np.random.uniform(0.20, 0.35)
    ctrl = np.zeros((n, 3))
    ctrl[:, 2] = np.linspace(0, L, n)
    amp = np.random.uniform(0.01, 0.06)
    for i in range(1, n):
        ctrl[i, 0] = amp * np.sin(np.random.uniform(0, 2 * np.pi))
        ctrl[i, 1] = amp * 0.6 * np.sin(np.random.uniform(0, 2 * np.pi))
    t_c = np.linspace(0, 1, n); t_f = np.linspace(0, 1, 500)
    return np.column_stack([cubic_interp(t_c, ctrl[:, k], t_f) for k in range(3)])


def compute_frames(cl):
    T = np.gradient(cl, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-10
    N = np.zeros_like(T)
    N[0] = np.cross(T[0], [1, 0, 0])
    N[0] /= np.linalg.norm(N[0]) + 1e-10
    for i in range(1, len(cl)):
        proj = N[i - 1] - np.dot(N[i - 1], T[i]) * T[i]
        pn = np.linalg.norm(proj)
        N[i] = proj / pn if pn > 1e-8 else N[i - 1]
    B = np.cross(T, N)
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-10
    return T, N, B


def arc_lengths(cl):
    diffs = np.diff(cl, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def interp_tangent_at_s(T_arr, arc, s):
    """Linear interpolation of the centreline tangent at arc length s."""
    s_c = float(np.clip(s, arc[0], arc[-1]))
    i_f = float(np.interp(s_c, arc, np.arange(len(arc), dtype=np.float64)))
    i_lo = int(np.floor(i_f))
    i_hi = min(len(T_arr) - 1, i_lo + 1)
    fr = i_f - i_lo
    t = (1.0 - fr) * T_arr[i_lo] + fr * T_arr[i_hi]
    return t / (np.linalg.norm(t) + 1e-12)


def interp_centerline_at_s(cl, arc, s):
    """Linear interpolation of the centreline position at arc length s."""
    s_c = float(np.clip(s, arc[0], arc[-1]))
    i_f = float(np.interp(s_c, arc, np.arange(len(arc), dtype=np.float64)))
    i_lo = int(np.floor(i_f))
    i_hi = min(len(cl) - 1, i_lo + 1)
    fr = i_f - i_lo
    return (1.0 - fr) * cl[i_lo] + fr * cl[i_hi]


def find_good_s_start(cl, T_arr, arc, chain_length, min_s=0.005, search_max_offset=0.04):
    """Find an s_start where the colon ahead is straight enough that the rigid
    scope chain (chain_length forward of base) ends up close to the centreline
    at s + chain_length.

    Returns (best_s, best_offset_metres). Caller should accept if offset is
    less than the tube radius (~17-32 mm for seed 42).
    """
    s_max = arc[-1] - chain_length - 0.07  # leave 70 mm headroom at far end
    candidates = np.linspace(min_s, max(min_s + 0.001, s_max), 200)
    best_s, best_dist = float(candidates[0]), float('inf')
    for s in candidates:
        pos_s = interp_centerline_at_s(cl, arc, s)
        t_s = interp_tangent_at_s(T_arr, arc, s)
        tip_world = pos_s + chain_length * t_s
        pos_tip_target = interp_centerline_at_s(cl, arc, s + chain_length)
        dist = float(np.linalg.norm(tip_world - pos_tip_target))
        if dist < best_dist:
            best_dist = dist
            best_s = float(s)
    return best_s, best_dist


def precurve_scope_joints(model, data, T_arr, arc, s_base, anchor_quat_wxyz,
                          n_disks, disk_spacing, joint_range=0.55):
    """Set joint_1..joint_{n_disks-1} qpos so the scope chain follows the
    centreline shape starting from the base. Idempotent; call once before
    the control loop.

    Hinge axis pattern (per generate_videoscope_one_section.py):
        joint_i has axis X (= local (1,0,0)) if i is odd, Z (= local (0,0,1)) if i is even.
    Joint range is +/-0.55 rad; we clamp.
    """
    R_base_flat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(R_base_flat, np.asarray(anchor_quat_wxyz, dtype=np.float64))
    R_world_prev = R_base_flat.reshape(3, 3).copy()

    set_count = 0
    max_abs = 0.0
    for i in range(1, n_disks):
        s_i = s_base + i * disk_spacing
        t_world = interp_tangent_at_s(T_arr, arc, s_i)
        # Express target tangent in the previous disk's local frame.
        t_local = R_world_prev.T @ t_world

        if i % 2 == 1:  # X-axis hinge: rotates +Y in the Y-Z plane.
            theta = float(np.arctan2(t_local[2], t_local[1]))
            axis_x = True
        else:  # Z-axis hinge: rotates +Y in the X-Y plane.
            theta = float(np.arctan2(-t_local[0], t_local[1]))
            axis_x = False

        max_abs = max(max_abs, abs(theta))
        theta = float(np.clip(theta, -joint_range, joint_range))

        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i}")
        if jid >= 0:
            qadr = int(model.jnt_qposadr[jid])
            data.qpos[qadr] = theta
            set_count += 1

        # Update R_world_prev for the next hinge.
        c, s = np.cos(theta), np.sin(theta)
        if axis_x:
            R_joint = np.array([[1, 0, 0],
                                [0, c, -s],
                                [0, s, c]], dtype=np.float64)
        else:
            R_joint = np.array([[c, -s, 0],
                                [s, c, 0],
                                [0, 0, 1]], dtype=np.float64)
        R_world_prev = R_world_prev @ R_joint

    mujoco.mj_forward(model, data)
    return set_count, max_abs


# ----- pose at arc length -----------------------------------------------

def pose_at_s(cl, T, N, B, arc, s):
    """Return (pos_world, quat_wxyz_for_mujoco) for the mocap base.

    The MuJoCo scope's tip_cam has xyaxes="-1 0 0 0 0 -1", so to match the
    Blender dataset's image-up = -N convention, we set the base body so that:
        body_X_in_world = B   (right)
        body_Y_in_world = T   (forward)
        body_Z_in_world = N   (up)
    The tip_cam then has image-up = -body_Z = -N. (Image-right comes out as
    -body_X = -B, opposite to the dataset's +B convention -- SIGN_X absorbs
    the flip.)
    """
    s_clamped = float(np.clip(s, arc[0], arc[-1]))
    i_float = np.interp(s_clamped, arc, np.arange(len(arc), dtype=np.float64))
    i_low = int(np.floor(i_float))
    i_high = min(len(cl) - 1, i_low + 1)
    frac = i_float - i_low

    pos = (1.0 - frac) * cl[i_low] + frac * cl[i_high]
    t = (1.0 - frac) * T[i_low] + frac * T[i_high]; t /= np.linalg.norm(t) + 1e-10
    n = (1.0 - frac) * N[i_low] + frac * N[i_high]
    n = n - np.dot(n, t) * t; n /= np.linalg.norm(n) + 1e-10
    b = np.cross(t, n)

    mat = np.column_stack([b, t, n]).astype(np.float64).ravel()
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, mat)
    return pos, quat


# ----- Blender worker handle --------------------------------------------

class BlenderWorker:
    def __init__(self, seed, width, height, samples, render_path):
        self.render_path = Path(render_path)
        cmd = [
            BLENDER_EXE, "--background", "--python", str(WORKER_SCRIPT), "--",
            "--seed", str(seed),
            "--width", str(width),
            "--height", str(height),
            "--samples", str(samples),
            "--render_path", str(render_path),
        ]
        print(f"[main] spawning Blender worker ...")
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Read stdout until "READY" appears. Blender prints its own startup chatter
        # before our worker code runs; we skip non-protocol lines.
        t0 = time.time()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                err = self.proc.stderr.read()
                raise RuntimeError(f"Blender worker died during setup. stderr:\n{err}")
            if line.strip() == "READY":
                print(f"[main] worker READY ({time.time()-t0:.1f}s)")
                break

    def render(self, pos, right, up, fwd):
        msg = json.dumps({
            "pos": [float(x) for x in pos],
            "right": [float(x) for x in right],
            "up": [float(x) for x in up],
            "fwd": [float(x) for x in fwd],
        })
        self.proc.stdin.write(msg + "\n")
        self.proc.stdin.flush()
        # Wait for DONE (skip Blender's per-render progress chatter).
        while True:
            line = self.proc.stdout.readline()
            if not line:
                err = self.proc.stderr.read()
                raise RuntimeError(f"Worker died mid-render. stderr:\n{err}")
            s = line.strip()
            if s == "DONE":
                return
            if s.startswith("ERR:"):
                raise RuntimeError(f"Worker render error: {s}")

    def close(self):
        try:
            self.proc.stdin.write("EXIT\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


# ----- aim-point computation --------------------------------------------

def compute_aim_point(depth, percentile=AIM_PERCENTILE):
    """Centroid of pixels above the given depth percentile.

    Returns (x_pix, y_pix) in image coords (origin top-left, x right, y down).
    Returns image centre if no valid pixels.
    """
    H, W = depth.shape
    valid = (depth > 0) & np.isfinite(depth)
    if valid.sum() < 10:
        return W * 0.5, H * 0.5
    thresh = np.percentile(depth[valid], percentile)
    deep = valid & (depth >= thresh)
    if deep.sum() < 1:
        return W * 0.5, H * 0.5
    ys, xs = np.nonzero(deep)
    return float(xs.mean()), float(ys.mean())


# ----- annotation -------------------------------------------------------

def annotate(rgb_arr, aim_x, aim_y):
    """Draw a red crosshair on the RGB image at the aim point."""
    img = Image.fromarray(rgb_arr)
    draw = ImageDraw.Draw(img)
    x, y = int(round(aim_x)), int(round(aim_y))
    r = 6
    draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0), width=2)
    draw.line((x - 12, y, x + 12, y), fill=(255, 0, 0), width=1)
    draw.line((x, y - 12, x, y + 12), fill=(255, 0, 0), width=1)
    return img


# ----- main -------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=DEFAULT_SEED)
    p.add_argument('--max_steps', type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument('--samples', type=int, default=8, help='Blender Cycles samples per render (default 8 for nav-test speed; dataset gen used 32)')
    p.add_argument('--checkpoint', default=str(CHECKPOINT_PATH))
    p.add_argument('--out_dir', default=str(HERE / 'nav_test_output'))
    return p.parse_args()


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'control_log.csv'
    render_path = out_dir / '_render_tmp.png'
    xml_path = out_dir / 'nav_scene.xml'

    # --- centreline -----------------------------------------------------
    cl = random_centerline(args.seed)
    T, N, B = compute_frames(cl)
    arc = arc_lengths(cl)
    total_length = arc[-1]
    print(f"[main] centreline seed={args.seed} length={total_length*1000:.0f}mm n={len(cl)}")

    # Auto-find an s_start where the rigid scope chain ends up close to the
    # centreline (i.e. the colon ahead is straight enough for the 65 mm chain).
    import sys as _sys_pre
    _sys_pre.path.insert(0, str(HERE))
    import generate_videoscope_one_section as _scope_pre
    _chain_length = _scope_pre.N_JOINTS * _scope_pre.DISK_SPACING
    s_start, _tip_offset = find_good_s_start(cl, T, arc, _chain_length)
    print(f"[main] auto-chose s_start={s_start*1000:.0f}mm "
          f"(rigid-tip offset from centreline at s+65mm = {_tip_offset*1000:.1f}mm)")
    s_max = max(s_start, total_length - 0.07)  # leave 70mm headroom
    anchor_pos, anchor_quat = pose_at_s(cl, T, N, B, arc, s_start)
    print(f"[main] will stop at s<={s_max*1000:.0f}mm")

    # --- MuJoCo scene ---------------------------------------------------
    from build_nav_scene import build_nav_scene
    build_nav_scene(anchor_pos, anchor_quat, xml_path)
    print(f"[main] wrote {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # No pre-curve: with auto-chosen s_start the rigid scope already fits inside
    # the tube. (Pre-curving was tried but the tendons' high-gain spring
    # immediately undoes any qpos set without matching baseline ctrl values.)

    def aid(n):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if i < 0:
            raise RuntimeError(f"actuator '{n}' not found")
        return i

    act_px = aid("pull_px"); act_nx = aid("pull_nx")
    act_pz = aid("pull_pz"); act_nz = aid("pull_nz")
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "tip_cam")
    if cam_id < 0:
        raise RuntimeError("tip_cam not found in nav scene")
    anchor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "scope_anchor")
    mocap_id = int(model.body_mocapid[anchor_body_id])

    # --- DA3 model ------------------------------------------------------
    from depth_anything_3.api import DepthAnything3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[main] loading DA3-SMALL + checkpoint on {device}")
    da3 = DepthAnything3.from_pretrained('depth-anything/DA3-SMALL').to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    da3.load_state_dict(state_dict)
    da3.eval()
    if isinstance(ckpt, dict) and 'val_scale_aligned_mae_m' in ckpt:
        print(f"[main]   checkpoint val MAE = {ckpt['val_scale_aligned_mae_m']*1000:.2f} mm")

    # --- Blender worker -------------------------------------------------
    worker = BlenderWorker(args.seed, IMG_W, IMG_H, args.samples, render_path)

    # --- log ------------------------------------------------------------
    log_f = log_path.open('w', newline='', encoding='utf-8')
    logw = csv.writer(log_f)
    logw.writerow(['step', 's_current_m', 'aim_x_pix', 'aim_y_pix',
                   'x_norm', 'y_norm', 'mag', 'cmd_x_pair', 'cmd_y_pair',
                   'tip_x', 'tip_y', 'tip_z',
                   'pred_depth_min_m', 'pred_depth_max_m',
                   'deep_frac', 'is_stuck'])

    # --- main loop ------------------------------------------------------
    s_current = float(s_start)
    cmd_x_pair = 0.0
    cmd_y_pair = 0.0
    prev_x_norm = 0.0  # for derivative term
    prev_y_norm = 0.0
    t_loop_start = time.time()
    try:
        for step in range(args.max_steps):
            # Get tip-cam world pose (after physics from previous step, or initial).
            mujoco.mj_forward(model, data)
            cam_pos = data.cam_xpos[cam_id].copy()
            cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()

            # Convert MuJoCo cam-to-world rotation into Blender's
            # dataset-script camera-setup args:
            #   right = cam_X_in_world          (Blender col 0)
            #   up    = -cam_Y_in_world         (Blender col 1 = -up)
            #   fwd   = -cam_Z_in_world         (Blender col 2 = -fwd; cam looks in +fwd)
            right_v = cam_mat[:, 0]
            up_v = -cam_mat[:, 1]
            fwd_v = -cam_mat[:, 2]

            # Render in Blender.
            worker.render(cam_pos, right_v, up_v, fwd_v)
            rgb_arr = np.asarray(Image.open(render_path).convert('RGB'))

            # Run DA3.
            with torch.no_grad():
                pred = da3.inference([str(render_path)])
            depth_pred = np.asarray(pred.depth[0], dtype=np.float32)
            # DA3 resizes internally; ensure we work at the input image's size.
            if depth_pred.shape != (IMG_H, IMG_W):
                depth_pred = np.asarray(
                    Image.fromarray(depth_pred).resize((IMG_W, IMG_H), Image.BILINEAR),
                    dtype=np.float32,
                )

            # Aim point: centroid of deepest 5% of pixels.
            aim_x, aim_y = compute_aim_point(depth_pred, AIM_PERCENTILE)
            x_norm = (aim_x - IMG_W * 0.5) / (IMG_W * 0.5)
            y_norm = (aim_y - IMG_H * 0.5) / (IMG_H * 0.5)
            mag = float(np.sqrt(x_norm * x_norm + y_norm * y_norm))

            # PD controller, rate-limited update to antagonistic tendon pairs.
            # Derivative term brakes when error is shrinking, reducing overshoot.
            if mag >= DEADZONE:
                d_x_norm = x_norm - prev_x_norm
                d_y_norm = y_norm - prev_y_norm
                dx = float(np.clip(SIGN_X * (K_P * x_norm + K_D * d_x_norm),
                                   -RATE_MAX, RATE_MAX))
                dy = float(np.clip(SIGN_Y * (K_P * y_norm + K_D * d_y_norm),
                                   -RATE_MAX, RATE_MAX))
                cmd_x_pair = float(np.clip(cmd_x_pair + dx, -MAX_PULL, MAX_PULL))
                cmd_y_pair = float(np.clip(cmd_y_pair + dy, -MAX_PULL, MAX_PULL))
            prev_x_norm = x_norm
            prev_y_norm = y_norm

            data.ctrl[act_px] = cmd_x_pair
            data.ctrl[act_nx] = -cmd_x_pair
            data.ctrl[act_pz] = cmd_y_pair
            data.ctrl[act_nz] = -cmd_y_pair

            # Stuck detector: fraction of pixels showing "lumen-direction" depth.
            # If almost no pixels are deeper than STUCK_DEEP_PIXEL_M, the camera
            # is nose-to-wall and we should back off + relax the tendons rather
            # than advance and keep bending.
            valid_mask = (depth_pred > 0) & np.isfinite(depth_pred)
            if valid_mask.any():
                deep_frac = float(((depth_pred > STUCK_DEEP_PIXEL_M) & valid_mask).sum()) \
                            / float(valid_mask.sum())
            else:
                deep_frac = 0.0
            is_stuck = deep_frac < STUCK_DEEP_FRACTION

            if is_stuck:
                # Reverse but never go below the auto-found s_start (that was the
                # best straight section for the rigid scope; further back is worse).
                s_current = max(s_current - REVERSE_RATE, s_start)
                # Aggressively relax tendons so the bent scope straightens out.
                cmd_x_pair *= STUCK_RELAX_FACTOR
                cmd_y_pair *= STUCK_RELAX_FACTOR
            else:
                # Hybrid advance: full rate when on-target, reduced when off.
                rate = ADVANCE_RATE_M_PER_STEP if mag < DEADZONE else ADVANCE_RATE_OFF_TARGET
                s_current = min(s_current + rate, s_max)
            mp, mq = pose_at_s(cl, T, N, B, arc, s_current)
            data.mocap_pos[mocap_id] = mp
            data.mocap_quat[mocap_id] = mq

            for _ in range(PHYSICS_STEPS_PER_CONTROL):
                mujoco.mj_step(model, data)

            # Annotate and save frame.
            annotate(rgb_arr, aim_x, aim_y).save(frames_dir / f'frame_{step:06d}.png')

            tip_pos = data.cam_xpos[cam_id]
            logw.writerow([
                step, f"{s_current:.6f}",
                f"{aim_x:.2f}", f"{aim_y:.2f}",
                f"{x_norm:+.4f}", f"{y_norm:+.4f}", f"{mag:.4f}",
                f"{cmd_x_pair:+.6f}", f"{cmd_y_pair:+.6f}",
                f"{tip_pos[0]:.6f}", f"{tip_pos[1]:.6f}", f"{tip_pos[2]:.6f}",
                f"{float(depth_pred[depth_pred>0].min()) if (depth_pred>0).any() else 0.0:.4f}",
                f"{float(depth_pred.max()):.4f}",
                f"{deep_frac:.3f}",
                int(is_stuck),
            ])

            if (step + 1) % 10 == 0:
                el = time.time() - t_loop_start
                fps = (step + 1) / el
                eta = (args.max_steps - step - 1) / fps if fps > 0 else 0
                print(f"[main] step {step+1}/{args.max_steps}  s={s_current*1000:.0f}mm  "
                      f"aim=({aim_x:.0f},{aim_y:.0f})  cmd=({cmd_x_pair:+.4f},{cmd_y_pair:+.4f})  "
                      f"{fps:.2f} fps  ETA {eta/60:.1f}min")

            if s_current >= s_max - 1e-9:
                print(f"[main] reached s_max={s_max:.4f}m; stopping.")
                break
    except KeyboardInterrupt:
        print("\n[main] interrupted by user")
    finally:
        log_f.close()
        worker.close()
        try:
            os.remove(render_path)
        except Exception:
            pass

    print(f"\n[main] frames -> {frames_dir}")
    print(f"[main] log    -> {log_path}")


if __name__ == '__main__':
    main()
