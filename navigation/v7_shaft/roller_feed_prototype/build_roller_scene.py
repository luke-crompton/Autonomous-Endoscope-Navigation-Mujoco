"""
Standalone prototype: 4-roller capstan feeder gripping a straight length of
flexible shaft, no colon, no bending tip.

Purpose: validate that spinning, friction-gripping rollers can actually
drive a floppy multi-link chain (and slip gracefully under resistance)
*before* replacing v7_shaft's rigid mocap-teleport drive with this
mechanism. See the plan this came from:
C:\\Users\\lukec\\.claude\\plans\\eager-munching-ladybug.md

Layout: shaft lies along world +Y. 20 links, same per-link physical
parameters as the production shaft in build_collision_scene.py, alternating
X/Z hinges, free joint at the root (nothing rigidly drives it -- only
roller friction, and later an optional fixed obstacle, act on it). 4 rollers
sit at a fixed point (y=0, "the entrance") at 0/90/180/270 degrees around
the shaft axis, each spinnable via its own velocity actuator.

CLI:
    python build_roller_scene.py [--obstacle] [--out path.xml]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Shaft parameters (matches Stage4/v7_shaft/build_collision_scene.py SHAFT_*)
# ---------------------------------------------------------------------------
N_LINKS       = 20
LINK_SPACING  = 0.010     # 10 mm
LINK_RADIUS   = 0.0045    # 4.5 mm -> 9 mm OD
JOINT_RANGE   = 0.349     # +/-20 deg
STIFFNESS     = 2.0
DAMPING       = 0.3
LINK_MASS     = 0.005     # 5 g/link
LINK_FRICTION = "0.08 0.005 0.001"   # same low shaft-vs-wall friction as production

# Root starts at y = -0.100 so the initial chain (length ~190mm) straddles
# the rollers at y=0 -- there's immediate roller/shaft contact to test grip
# from frame 1, rather than needing to first drive an untouched chain into
# the rollers.
ROOT_START_Y = -0.100

# ---------------------------------------------------------------------------
# Roller parameters
# ---------------------------------------------------------------------------
ROLLER_RADIUS   = 0.008    # 8 mm
ROLLER_HALF_LEN = 0.006    # 6 mm half-height (12 mm wide roller)
PRELOAD_M       = 0.0002   # 1 mm geometric interference -> normal (grip) force
MOUNT_RADIUS    = LINK_RADIUS + ROLLER_RADIUS - PRELOAD_M
ROLLER_FRICTION = "1.2 0.02 0.01"   # high friction -- opposite of the cage's near-zero friction
ROLLER_DAMPING  = 0.02              # spin-axis damping (bearing friction), separate from grip friction
ROLLER_ARMATURE = 0.0005            # added rotor inertia -- the roller's true rigid-body inertia
                                     # (tiny, ~1e-6) let contact impulses cause huge angular
                                     # accelerations each substep (chattering); armature adds
                                     # effective inertia to the joint itself to stabilize spin
ACTUATOR_KV     = 0.05              # velocity actuator gain (torque = kv * (target - actual))
ACTUATOR_FORCE_LIMIT = 1.0          # N*m; generous so grip friction is the real limiting factor, not the motor

# Optional fixed obstacle further down +Y, to test graceful slip under load.
OBSTACLE_Y = 0.30
OBSTACLE_HALF_SIZE = "0.03 0.002 0.03"


def _roller_spin_axis(theta: float) -> np.ndarray:
    """Unit spin axis for a roller sitting at radial angle theta (X-Z plane),
    computed as radial_direction x shaft_axis(+Y) -- purely in the X-Z plane,
    so the roller's rim surface velocity at the contact point is parallel
    to +/-Y (the capstan/feed direction) when spun."""
    radial = np.array([np.cos(theta), 0.0, np.sin(theta)])
    y_axis = np.array([0.0, 1.0, 0.0])
    axis = np.cross(radial, y_axis)
    return axis / np.linalg.norm(axis)


def build_scene(out_path: Path | None = None, with_obstacle: bool = False) -> Path:
    if out_path is None:
        out_path = HERE / "roller_scene.xml"

    lines: list[str] = []
    w = lines.append

    w('<?xml version="1.0" ?>')
    w('<mujoco model="roller_feed_prototype">')
    w('  <compiler angle="radian" autolimits="true"/>')
    w('  <option gravity="0 0 0" timestep="0.0005" integrator="implicitfast">')
    w('    <flag contact="enable"/>')
    w('  </option>')
    w('  <default>')
    w('    <geom condim="3" margin="0.0002"/>')
    w('  </default>')
    w('')
    w('  <asset>')
    w('    <material name="shaft_mat" rgba="0.16 0.18 0.20 1"/>')
    w('    <material name="roller_mat" rgba="0.75 0.25 0.20 1"/>')
    w('    <material name="obstacle_mat" rgba="0.3 0.3 0.3 0.5"/>')
    w('  </asset>')
    w('')
    w('  <worldbody>')

    # ---- 4 rollers, fixed mounts at y=0 ----
    for gi, theta in enumerate([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]):
        mount_pos = MOUNT_RADIUS * np.array([np.cos(theta), 0.0, np.sin(theta)])
        axis = _roller_spin_axis(theta)
        mx, my, mz = mount_pos
        ax, ay, az = axis
        w(f'    <body name="roller_{gi}" pos="{mx:.6f} {my:.6f} {mz:.6f}">')
        w(
            f'      <joint name="roller_spin_{gi}" type="hinge" '
            f'axis="{ax:.6f} {ay:.6f} {az:.6f}" damping="{ROLLER_DAMPING}" '
            f'armature="{ROLLER_ARMATURE}"/>'
        )
        w(
            f'      <inertial pos="0 0 0" mass="0.02" '
            f'diaginertia="1e-6 1e-6 1e-6"/>'
        )
        # Cylinder's own axis is local +Z by default in MuJoCo; orient it
        # along `axis` via a quaternion computed from the axis directly using
        # the "zaxis" shorthand (rotates local +Z to point along `axis`).
        w(
            f'      <geom name="roller_{gi}_geom" type="cylinder" '
            f'size="{ROLLER_RADIUS:.6f} {ROLLER_HALF_LEN:.6f}" '
            f'zaxis="{ax:.6f} {ay:.6f} {az:.6f}" '
            f'friction="{ROLLER_FRICTION}" material="roller_mat"/>'
        )
        w('    </body>')
    w('')

    # ---- optional fixed obstacle ----
    if with_obstacle:
        w(
            f'    <geom name="obstacle" type="box" '
            f'pos="0 {OBSTACLE_Y:.6f} 0" size="{OBSTACLE_HALF_SIZE}" '
            f'material="obstacle_mat"/>'
        )
        w('')

    # ---- shaft chain, free joint at the root ----
    sp = f"{LINK_SPACING:.6f}"
    sr = f"{LINK_RADIUS:.6f}"
    sm = f"{LINK_MASS:.6f}"

    w(f'    <body name="shaft_link_0" pos="0 {ROOT_START_Y:.6f} 0">')
    w('      <joint name="shaft_root_free" type="free"/>')
    w(f'      <inertial pos="0 {LINK_SPACING/2:.6f} 0" mass="{sm}" diaginertia="1e-6 1e-6 1e-6"/>')
    w(
        f'      <geom name="shaft_link_0_geom" type="capsule" fromto="0 0 0 0 {sp} 0" '
        f'size="{sr}" material="shaft_mat" friction="{LINK_FRICTION}"/>'
    )

    indent = "      "
    for k in range(1, N_LINKS):
        indent += "  "
        axis = "1 0 0" if k % 2 == 1 else "0 0 1"
        w(f'{indent}<body name="shaft_link_{k}" pos="0 {sp} 0">')
        w(
            f'{indent}  <joint name="shaft_joint_{k}" type="hinge" axis="{axis}" '
            f'range="-{JOINT_RANGE:.6f} {JOINT_RANGE:.6f}" '
            f'stiffness="{STIFFNESS}" damping="{DAMPING}"/>'
        )
        w(f'{indent}  <inertial pos="0 {LINK_SPACING/2:.6f} 0" mass="{sm}" diaginertia="1e-6 1e-6 1e-6"/>')
        w(
            f'{indent}  <geom name="shaft_link_{k}_geom" type="capsule" '
            f'fromto="0 0 0 0 {sp} 0" size="{sr}" material="shaft_mat" '
            f'friction="{LINK_FRICTION}"/>'
        )

    for k in range(N_LINKS - 1, 0, -1):
        indent = "      " + "  " * k
        w(f'{indent}</body>')
    w('    </body>')  # close shaft_link_0

    w('  </worldbody>')
    w('')

    # ---- contact exclusions ----
    w('  <contact>')
    for k in range(N_LINKS - 1):
        w(f'    <exclude body1="shaft_link_{k}" body2="shaft_link_{k + 1}"/>')
    # Rollers should only ever touch the shaft, never each other. At 90-degree
    # spacing their curved rims (8mm radius, 12mm wide) overlap near the shaft
    # axis -- confirmed by contact dump showing ~1.7mm interpenetration on
    # every adjacent pair, which was swamping the actual grip/drive signal.
    for i in range(4):
        for j in range(i + 1, 4):
            w(f'    <exclude body1="roller_{i}" body2="roller_{j}"/>')
    w('  </contact>')
    w('')

    # ---- roller spin actuators ----
    w('  <actuator>')
    for gi in range(4):
        w(
            f'    <velocity name="roller_act_{gi}" joint="roller_spin_{gi}" '
            f'kv="{ACTUATOR_KV}" forcelimited="true" '
            f'forcerange="-{ACTUATOR_FORCE_LIMIT} {ACTUATOR_FORCE_LIMIT}"/>'
        )
    w('  </actuator>')
    w('</mujoco>')

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--obstacle", action="store_true", help="add a fixed obstacle further down +Y")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out_path = build_scene(args.out, with_obstacle=args.obstacle)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
