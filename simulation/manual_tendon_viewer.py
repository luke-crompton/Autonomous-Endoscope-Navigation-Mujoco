"""
Manual MuJoCo viewer for the one-section videoscope model.

Run:
  python manual_tendon_viewer.py

Use the MuJoCo viewer's Controls panel to drag either tendon in a pair:
  pull_px, pull_nx, pull_pz, pull_nz

The script enforces antagonistic pairing:
  pull_px = +0.003 sets pull_nx = -0.003
  pull_nx = +0.003 sets pull_px = -0.003

Slider units are metres. For example:
  0.003 = 3 mm tendon pull
 -0.003 = 3 mm tendon release
"""

from __future__ import annotations

import os
import time

import mujoco
import mujoco.viewer


XML_FILE = "videoscope_one_section.xml"
TIP_SITE = "tip_site"
MAX_PAIR_PULL = 0.024
PAIR_SPECS = (
    ("X", "pull_px", "pull_nx", "+X", "-X"),
    ("Z", "pull_pz", "pull_nz", "+Z", "-Z"),
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def actuator_id(model: mujoco.MjModel, name: str) -> int:
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator < 0:
        raise ValueError(f"Missing actuator: {name}")
    return actuator


def enforce_antagonistic_pairs(data: mujoco.MjData, pairs: tuple, previous_ctrl) -> dict[str, float]:
    commands: dict[str, float] = {}
    for axis, pull_a, pull_b, _label_a, _label_b, idx_a, idx_b in pairs:
        delta_a = data.ctrl[idx_a] - previous_ctrl[idx_a]
        delta_b = data.ctrl[idx_b] - previous_ctrl[idx_b]

        if abs(delta_b) > abs(delta_a):
            command = -float(data.ctrl[idx_b])
        else:
            command = float(data.ctrl[idx_a])

        command = clamp(command, -MAX_PAIR_PULL, MAX_PAIR_PULL)
        data.ctrl[idx_a] = command
        data.ctrl[idx_b] = -command
        commands[axis] = command

    previous_ctrl[:] = data.ctrl
    return commands


def pair_status(axis: str, command: float, plus_label: str, minus_label: str) -> str:
    value = abs(command) * 1000.0
    if command >= 0:
        return f"{axis}:{value:5.2f}mm {plus_label} pull/{minus_label} release"
    return f"{axis}:{value:5.2f}mm {minus_label} pull/{plus_label} release"


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if not os.path.exists(XML_FILE):
        raise FileNotFoundError(
            f"{XML_FILE} is missing. Run generate_videoscope_one_section.py first."
        )

    model = mujoco.MjModel.from_xml_path(XML_FILE)
    data = mujoco.MjData(model)
    tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TIP_SITE)
    if tip_site_id < 0:
        raise ValueError(f"Missing site: {TIP_SITE}")
    pairs = tuple(
        (*spec, actuator_id(model, spec[1]), actuator_id(model, spec[2]))
        for spec in PAIR_SPECS
    )

    mujoco.mj_forward(model, data)

    print("=" * 70)
    print("Videoscope one-section manual tendon viewer")
    print("=" * 70)
    print(f"Bodies: {model.nbody}  Joints: {model.njnt}  Tendons: {model.ntendon}  Controls: {model.nu}")
    print("")
    print("Open the viewer Controls panel and drag either side of a tendon pair.")
    print("The opposite side is released by the same amount automatically.")
    print("Slider units are metres: +0.003 means 3 mm pull, -0.003 means 3 mm release.")
    print("Pair range: +/-0.024 m, so each side can pull or release 24 mm.")
    print("Controls: pull_px/pull_nx for X, pull_pz/pull_nz for Z")
    print("")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18
        viewer.cam.distance = 0.16
        viewer.cam.lookat[:] = [0.0, 0.032, 0.0]

        last_print = 0.0
        previous_ctrl = data.ctrl.copy()
        while viewer.is_running():
            commands = enforce_antagonistic_pairs(data, pairs, previous_ctrl)
            mujoco.mj_step(model, data)
            viewer.sync()

            now = time.time()
            if now - last_print > 0.25:
                tip_mm = data.site_xpos[tip_site_id] * 1000.0
                pull_text = " | ".join(
                    pair_status(axis, commands[axis], plus_label, minus_label)
                    for axis, _pull_a, _pull_b, plus_label, minus_label, _idx_a, _idx_b in pairs
                )
                print(
                    f"\r{pull_text} | Tip "
                    f"({tip_mm[0]:+6.1f}, {tip_mm[1]:+6.1f}, {tip_mm[2]:+6.1f}) mm",
                    end="",
                    flush=True,
                )
                last_print = now

            time.sleep(0.002)

    print("\nViewer closed.")


if __name__ == "__main__":
    main()
