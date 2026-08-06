
"""
Interactive keyboard test for v7_shaft flexible shaft dynamics.

Drives the MuJoCo model directly (no policy, no obs rendering).
Starts with shaft tip at the colon entrance — ready to push in immediately.

Controls (focus the MuJoCo viewer window):
  UP / DOWN        bend tip +X / -X  (1 mm per key event, springs back on release)
  LEFT / RIGHT     bend tip -Z / +Z  (1 mm per key event, springs back on release)
  Left Shift       feed shaft in (spins entrance rollers to insert)
  Left Ctrl        retract shaft (spins entrance rollers to withdraw)
  Backspace        reset
  ESC              quit

Usage:
    python manual_test.py --seed 0
    python manual_test.py --seed 3 --show_collision
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import mujoco           # noqa: E402
import mujoco.viewer    # noqa: E402

from build_collision_scene import (  # noqa: E402
    build_scene,
    tangent_at_s,
)
import generate_videoscope_one_section as scope_gen  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# MuJoCo's XML compiler parses/compiles the body tree via recursive descent —
# one recursive call per nested <body>. v7_shaft nests 60 shaft links + 25 tip
# disks = 85 levels deep, which overflows Windows' default 1 MB thread stack
# (STATUS_STACK_OVERFLOW). Building the model on a thread with a bigger stack
# avoids this without changing the scene itself.
MODEL_BUILD_STACK_SIZE = 64 * 1024 * 1024   # 64 MB

PHYSICS_PER_STEP = 20           # MuJoCo substeps per frame (0.5 ms each = 10 ms/frame)
MAX_PULL_X       = scope_gen.MAX_PULL_X        # ~6.80 mm (120deg X limit)
MAX_PULL_Z       = scope_gen.MAX_PULL_Z        # ~7.36 mm (130deg Z limit)

# Per key-event steps — each press / GLFW repeat fires exactly one delta.
# GLFW repeats at ~30 ms when held, so holding gives ~33 events/s.
TENDON_STEP  = 0.001            # 1 mm per event (v8_p1: ~7 taps for full curl, was 24)

# Roller feed actuators are POSITION-controlled (2026-07-01): ctrl is a
# target angle (rad), not a speed. A pure velocity actuator only resisted
# CHANGES in speed, not a fixed angle, so residual bending stress could
# creep the shaft for ~1s after releasing keys (measured ~2mm) -- real
# motorized feed rollers hold position when not commanded. We now track a
# target angle that advances at ROLLER_OMEGA_CMD (rad/s) while a key is
# held, and stays fixed otherwise -- the position actuator then genuinely
# holds the shaft still when idle. Validated sign convention
# (build_collision_scene.py): negative angle = insertion, positive = retraction.
ROLLER_OMEGA_CMD = 3.0          # rad/s equivalent while Shift/Ctrl held

# Decay factor per frame when no tendon key is held (exponential spring return).
# 0.82 per frame at ~60 fps ≈ 90% gone in ~12 frames (~0.2 s).
TENDON_DECAY = 0.82

# Key is considered "held" if its callback fired within this window.
# GLFW fires: 1 press event, then ~500 ms silence, then repeats at ~30 ms.
# Setting this above 500 ms means the decay is suppressed across the entire
# initial-delay gap, so holding feels continuous from the first tap.
_HOLD_MS = 0.60

# ---------------------------------------------------------------------------
# GLFW key codes
# ---------------------------------------------------------------------------
KEY_UP   = 265
KEY_DN   = 264
KEY_LF   = 263
KEY_RT   = 262
KEY_SHF  = 340   # Left Shift  — advance
KEY_CTL  = 341   # Left Ctrl   — retract
KEY_RSFT = 344   # Right Shift (also accepted for advance)
KEY_RCTL = 345   # Right Ctrl  (also accepted for retract)
KEY_BS   = 259   # Backspace   — reset
KEY_ESC  = 256   # quit

# ---------------------------------------------------------------------------
# Shared key state (callback thread → main thread)
# ---------------------------------------------------------------------------
_lock   = threading.Lock()
_pending = {'x': 0.0, 'z': 0.0}
_key_last: dict[int, float] = defaultdict(float)
_reset_flag = False
_quit_flag  = False


def _key_cb(keycode: int) -> None:
    global _reset_flag, _quit_flag
    now = time.monotonic()
    _key_last[keycode] = now
    with _lock:
        if   keycode == KEY_UP:                    _pending['x']     += TENDON_STEP
        elif keycode == KEY_DN:                    _pending['x']     -= TENDON_STEP
        elif keycode == KEY_LF:                    _pending['z']     -= TENDON_STEP
        elif keycode == KEY_RT:                    _pending['z']     += TENDON_STEP
    if keycode == KEY_BS:  _reset_flag = True
    if keycode == KEY_ESC: _quit_flag  = True


def _held(k: int) -> bool:
    return (time.monotonic() - _key_last.get(k, 0.0)) < _HOLD_MS


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def _do_reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset physics. shaft_link_0's free joint returns to its XML-specified
    initial pose (already threaded partway through the entrance rollers --
    see SCOPE_TIP_AT_S in build_collision_scene.py); no manual repositioning
    needed since there's no mocap to drive anymore."""
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def _build_model_with_bigger_stack(seed: int):
    """Build the scene and compile the MjModel on a thread with a larger
    stack, so the 85-level nested body tree doesn't overflow the default
    1 MB Windows thread stack. Returns (xml_path, centreline, model, data)."""
    result: dict = {}

    def _worker() -> None:
        try:
            xml_path, centreline = build_scene(seed)
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            data = mujoco.MjData(model)
            result["ok"] = (xml_path, centreline, model, data)
        except BaseException as exc:  # noqa: BLE001 - re-raised on caller's thread
            result["err"] = exc

    old_size = threading.stack_size()
    threading.stack_size(MODEL_BUILD_STACK_SIZE)
    try:
        t = threading.Thread(target=_worker)
        t.start()
        t.join()
    finally:
        threading.stack_size(old_size)

    if "err" in result:
        raise result["err"]
    return result["ok"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _reset_flag, _quit_flag

    parser = argparse.ArgumentParser(description="Manual keyboard test — v7_shaft")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show_collision", action="store_true",
                        help="show collision prism geoms (group 3)")
    args = parser.parse_args()

    print(f"Building scene for seed {args.seed}…")
    xml_path, centreline, model, data = _build_model_with_bigger_stack(args.seed)

    # Actuator order: 4 tendon-pull actuators, then 2 roller rings x 4 rollers
    # each (see build_collision_scene.py's <actuator> block). Both rings drive
    # together at the same commanded rate.
    CTRL_PX, CTRL_PZ, CTRL_NX, CTRL_NZ = 0, 1, 2, 3
    ROLLER_CTRL_SLICE = slice(4, 12)

    disk0_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "disk_0")
    entrance_tangent = tangent_at_s(centreline, 0.0)
    FRAME_DT = PHYSICS_PER_STEP * model.opt.timestep   # sim time advanced per rendered frame

    cmd_x = 0.0
    cmd_z = 0.0
    roller_target = 0.0   # accumulated target angle (rad) for the position-controlled roller actuators
    _do_reset(model, data)
    start_pos = data.xpos[disk0_bid].copy()

    print(
        "\n  Controls (focus the MuJoCo viewer window)\n"
        "  -----------------------------------------------------------\n"
        "  UP / DOWN        bend tip +X / -X   (1 mm per tap, springs back)\n"
        "  LEFT / RIGHT     bend tip -Z / +Z   (1 mm per tap, springs back)\n"
        "  Left Shift       feed shaft in       (entrance rollers spin to insert)\n"
        "  Left Ctrl        retract shaft       (entrance rollers spin to withdraw)\n"
        "  Backspace        reset\n"
        "  ESC              quit\n"
        f"\n  Shaft starts pre-threaded through the entrance rollers.\n"
        f"  MAX_PULL_X = {MAX_PULL_X*1000:.2f} mm -> full curl in {int(MAX_PULL_X/TENDON_STEP)} taps.\n"
        f"  MAX_PULL_Z = {MAX_PULL_Z*1000:.2f} mm -> full curl in {int(MAX_PULL_Z/TENDON_STEP)} taps.\n"
    )

    frame   = 0
    t0      = time.monotonic()
    _cbuf   = np.zeros(6)

    with mujoco.viewer.launch_passive(model, data, key_callback=_key_cb) as viewer:
        viewer.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = model.stat.center
        viewer.cam.distance  = max(0.25, model.stat.extent * 1.5)
        if args.show_collision:
            viewer.opt.geomgroup[3] = 1

        while viewer.is_running() and not _quit_flag:

            # ---- reset ----
            if _reset_flag:
                _reset_flag = False
                cmd_x = cmd_z = 0.0
                roller_target = 0.0
                with _lock:
                    _pending['x'] = _pending['z'] = 0.0
                _do_reset(model, data)
                data.ctrl[ROLLER_CTRL_SLICE] = 0.0
                start_pos = data.xpos[disk0_bid].copy()
                frame = 0
                t0    = time.monotonic()
                print("--- RESET ---")

            # ---- consume key events ----
            with _lock:
                dx = _pending['x']
                dz = _pending['z']
                _pending['x'] = _pending['z'] = 0.0

            # Apply per-event tendon deltas then clamp
            cmd_x = float(np.clip(cmd_x + dx, -MAX_PULL_X, MAX_PULL_X))
            cmd_z = float(np.clip(cmd_z + dz, -MAX_PULL_Z, MAX_PULL_Z))

            # Decay toward zero when key not recently held
            if not (_held(KEY_UP) or _held(KEY_DN)):
                cmd_x *= TENDON_DECAY
            if not (_held(KEY_LF) or _held(KEY_RT)):
                cmd_z *= TENDON_DECAY
            if abs(cmd_x) < 5e-5: cmd_x = 0.0
            if abs(cmd_z) < 5e-5: cmd_z = 0.0

            # ---- apply tendons ----
            data.ctrl[CTRL_PX] =  cmd_x
            data.ctrl[CTRL_NX] = -cmd_x
            data.ctrl[CTRL_PZ] =  cmd_z
            data.ctrl[CTRL_NZ] = -cmd_z

            # ---- feed / retract via entrance rollers ----
            # Position-controlled: roller_target is a target ANGLE, only
            # advanced while a key is held. Left untouched otherwise, so the
            # position actuator holds the shaft still instead of the old
            # velocity actuator's ~2mm creep after releasing keys. Sign
            # convention validated in build_collision_scene.py: negative
            # angle = insertion, positive = retraction.
            if _held(KEY_SHF) or _held(KEY_RSFT):
                roller_target -= ROLLER_OMEGA_CMD * FRAME_DT
            elif _held(KEY_CTL) or _held(KEY_RCTL):
                roller_target += ROLLER_OMEGA_CMD * FRAME_DT
            data.ctrl[ROLLER_CTRL_SLICE] = roller_target

            # ---- physics ----
            for _ in range(PHYSICS_PER_STEP):
                mujoco.mj_step(model, data)

            frame += 1
            viewer.sync()

            # ---- status every 60 frames ≈ 1 s ----
            if frame % 60 == 0:
                ncon    = int(data.ncon)
                raw_f   = 0.0
                for i in range(ncon):
                    mujoco.mj_contactForce(model, data, i, _cbuf)
                    raw_f += float(np.linalg.norm(_cbuf[:3]))
                pct_x = cmd_x / MAX_PULL_X * 100
                pct_z = cmd_z / MAX_PULL_Z * 100
                pushed = float(np.dot(data.xpos[disk0_bid] - start_pos, entrance_tangent))
                roller_omega = float(np.mean(data.ctrl[ROLLER_CTRL_SLICE]))
                roller_force = data.actuator_force[ROLLER_CTRL_SLICE]
                print(
                    f"t={time.monotonic()-t0:5.1f}s  "
                    f"pushed={pushed*1000:6.1f}mm  "
                    f"ncon={ncon:3d}  force={raw_f:6.1f}N  "
                    f"roller_ctrl={roller_omega:+4.1f}rad/s  "
                    f"roller_force=[mean={np.mean(np.abs(roller_force)):.2f}N max={np.max(np.abs(roller_force)):.2f}N]  "
                    f"X={pct_x:+5.0f}%  Z={pct_z:+5.0f}%"
                )

    print("Done.")


if __name__ == "__main__":
    main()
