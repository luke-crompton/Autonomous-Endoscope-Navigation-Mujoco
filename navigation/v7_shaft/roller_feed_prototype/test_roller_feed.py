"""
Interactive keyboard test for the roller/capstan feed prototype.

Confirms, by feel, what the headless benchmark already showed: 4 friction
rollers gripping a straight length of shaft can drive it forward/backward,
and hold steady under resistance instead of buckling.

Controls (focus the MuJoCo viewer window):
  Up / Down    spin rollers forward / backward (feed in / retract)
  Backspace    reset
  ESC          quit

Usage:
    python test_roller_feed.py
    python test_roller_feed.py --obstacle
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np

import mujoco
import mujoco.viewer

import build_roller_scene as brs

HERE = Path(__file__).parent

ROLLER_OMEGA_CMD = 3.0   # rad/s while a key is held
_HOLD_MS = 0.60

_lock = threading.Lock()
_key_last: dict[int, float] = {}
_quit_flag = False
_reset_flag = False

KEY_UP, KEY_DN, KEY_BS, KEY_ESC = 265, 264, 259, 256


def _key_cb(keycode: int) -> None:
    global _quit_flag, _reset_flag
    with _lock:
        _key_last[keycode] = time.monotonic()
    if keycode == KEY_BS:
        _reset_flag = True
    if keycode == KEY_ESC:
        _quit_flag = True


def _held(k: int) -> bool:
    with _lock:
        return (time.monotonic() - _key_last.get(k, 0.0)) < _HOLD_MS


def main() -> None:
    global _reset_flag, _quit_flag
    parser = argparse.ArgumentParser(description="Manual test -- roller feed prototype")
    parser.add_argument("--obstacle", action="store_true")
    args = parser.parse_args()

    xml_path = brs.build_scene(with_obstacle=args.obstacle)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    tip_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"shaft_link_{brs.N_LINKS - 1}")
    roller_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"roller_spin_{i}") for i in range(4)]
    roller_dofadr = [model.jnt_dofadr[j] for j in roller_jids]
    shaft_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"shaft_joint_{k}") for k in range(1, brs.N_LINKS)]
    shaft_qpos_addr = [model.jnt_qposadr[j] for j in shaft_jids]

    print(
        "\n  Controls (focus the MuJoCo viewer window)\n"
        "  -----------------------------------------------------------\n"
        "  UP     spin rollers to feed the shaft IN\n"
        "  DOWN   spin rollers to RETRACT the shaft\n"
        "  Backspace   reset\n"
        "  ESC         quit\n"
    )

    frame = 0
    with mujoco.viewer.launch_passive(model, data, key_callback=_key_cb) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.05, 0.0]
        viewer.cam.distance = 0.3

        while viewer.is_running() and not _quit_flag:
            if _reset_flag:
                _reset_flag = False
                mujoco.mj_resetData(model, data)
                mujoco.mj_forward(model, data)
                print("--- RESET ---")

            if _held(KEY_UP):
                data.ctrl[:] = -ROLLER_OMEGA_CMD
            elif _held(KEY_DN):
                data.ctrl[:] = ROLLER_OMEGA_CMD
            else:
                data.ctrl[:] = 0.0

            for _ in range(20):
                mujoco.mj_step(model, data)
            frame += 1
            viewer.sync()

            if frame % 60 == 0:
                y = data.xpos[tip_bid][1]
                omega = float(np.mean([data.qvel[a] for a in roller_dofadr]))
                max_angle = float(np.degrees(np.max(np.abs([data.qpos[a] for a in shaft_qpos_addr]))))
                print(f"tip_y={y*1000:7.1f}mm  roller_omega={omega:+5.2f}rad/s  max_joint_angle={max_angle:5.1f}deg  ncon={data.ncon}")

    print("Done.")


if __name__ == "__main__":
    main()
