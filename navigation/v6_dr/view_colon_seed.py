"""
Open a MuJoCo viewer for one generated colon seed.

This does not load a policy or run the environment loop. It only regenerates
the requested seed scene XML, loads it into MuJoCo, and opens the viewer.

Usage:
    python3 -B view_colon_seed.py --seed 16
    python3 -B view_colon_seed.py --seed 16 --show_collision
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from build_collision_scene import build_scene  # noqa: E402


def _colon_length_m(centreline: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(centreline, axis=0), axis=1)))


def main() -> None:
    parser = argparse.ArgumentParser(description="View one generated colon seed in MuJoCo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--show_collision",
        action="store_true",
        help="show hidden collision prisms/geoms in group 3",
    )
    parser.add_argument(
        "--no_scope",
        action="store_true",
        help="hide the scope geoms so only the colon is visible",
    )
    args = parser.parse_args()

    xml_path, centreline = build_scene(args.seed)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if args.no_scope:
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if (
                name.startswith("disk_")
                or name.startswith("joint_")
                or name.startswith("hole_")
                or name == "base_block"
            ):
                model.geom_rgba[geom_id, 3] = 0.0

    print(f"Seed        : {args.seed}")
    print(f"Scene       : {xml_path}")
    print(f"Length      : {_colon_length_m(centreline):.3f} m")
    print(f"Collision   : {'shown' if args.show_collision else 'hidden'}")
    print("Close the MuJoCo viewer window to quit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[3] = 1 if args.show_collision else 0
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = model.stat.center
        viewer.cam.distance = max(0.25, model.stat.extent * 1.4)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
