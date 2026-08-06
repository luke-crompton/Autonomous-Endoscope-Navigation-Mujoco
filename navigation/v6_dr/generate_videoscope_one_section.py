"""
Generate a one-section videoscope-style MuJoCo robot.

Geometry:
  - 25 hinge joints, 26 disks
  - 9.0 mm outer diameter disks
  - 4 tendon holes at 0, 90, 180, and 270 degrees
  - Tendon hole radius: 3.2 mm from the centerline
  - 2.0 mm diameter ball/socket rotation marker at each joint
  - 2.5 mm between adjacent disk centers
  - 2.5 mm from each joint center to the top of its disk

The output XML has no automated trajectory. It exposes four tendon controls:
  pull_px, pull_nx, pull_pz, pull_nz

Use manual_tendon_viewer.py for antagonistic pairing. Positive pull on one
side sets the opposite side to the same release distance.
"""

from __future__ import annotations

import argparse
import math
import os


N_JOINTS = 25
N_DISKS = N_JOINTS + 1

DISK_OD = 0.009
DISK_RADIUS = DISK_OD / 2.0
DISK_HEIGHT = 0.00167
DISK_HALF_HEIGHT = DISK_HEIGHT / 2.0
DISK_SPACING = 0.0025
JOINT_TO_DISK_TOP = 0.0025
STL_CENTER_Y = 0.0019
STL_TOP_Y = 0.0035
DISK_CENTER_FROM_JOINT = JOINT_TO_DISK_TOP - (STL_TOP_Y - STL_CENTER_Y)
MESH_Y_FROM_JOINT = JOINT_TO_DISK_TOP - STL_TOP_Y
ROTATION_BALL_DIAMETER = 0.0020
ROTATION_BALL_RADIUS = ROTATION_BALL_DIAMETER / 2.0

HOLE_RADIUS = 0.0032
HOLE_SITE_SIZE = 0.00033
HOLE_DOT_SIZE = 0.00022
TENDON_WIDTH = 0.00032

MAX_PULL = 0.024
REST_LENGTH = N_JOINTS * DISK_SPACING
ACTUATOR_KP = 6000.0
ACTUATOR_DAMPING = 0.08
ACTUATOR_PULL_FORCE_LIMIT = 80.0

TENDONS = [
    ("px", 0, "0.95 0.20 0.20 1"),
    ("pz", 90, "0.15 0.45 1.00 1"),
    ("nx", 180, "0.20 0.75 0.35 1"),
    ("nz", 270, "1.00 0.72 0.18 1"),
]


def fmt(value: float) -> str:
    return f"{value:.6f}"


def tendon_site_pos(angle_deg: float) -> tuple[float, float, float]:
    angle = math.radians(angle_deg)
    return (
        HOLE_RADIUS * math.cos(angle),
        0.0,
        HOLE_RADIUS * math.sin(angle),
    )


def add_disk(lines: list[str], disk_idx: int, indent: str) -> None:
    w = lines.append
    center_y = DISK_CENTER_FROM_JOINT

    if disk_idx == 0:
        w(f'{indent}<body name="disk_{disk_idx}" pos="0 0 0">')
    else:
        w(f'{indent}<body name="disk_{disk_idx}" pos="0 {fmt(DISK_SPACING)} 0">')
        axis = "1 0 0" if disk_idx % 2 == 1 else "0 0 1"
        w(
            f'{indent}  <joint name="joint_{disk_idx}" type="hinge" '
            f'axis="{axis}" range="-0.1745 0.1745" damping="0.035" '
            f'stiffness="0.50" armature="0.00001"/>'
        )
        w(
            f'{indent}  <geom name="joint_{disk_idx}_ball" type="sphere" '
            f'size="{fmt(ROTATION_BALL_RADIUS)}" material="joint_ball_mat" '
            f'contype="0" conaffinity="0"/>'
        )

    mesh_euler = "0 0 0" if disk_idx % 2 == 0 else "0 1.570796 0"
    w(
        f'{indent}  <geom name="disk_{disk_idx}_mesh" type="mesh" '
        f'mesh="new8_mesh" pos="0 {fmt(MESH_Y_FROM_JOINT)} 0" euler="{mesh_euler}" '
        f'material="new8_mat" contype="0" conaffinity="0"/>'
    )
    w(
        f'{indent}  <geom name="disk_{disk_idx}_body" type="cylinder" '
        f'pos="0 {fmt(center_y)} 0" size="{fmt(DISK_RADIUS)} {fmt(DISK_HALF_HEIGHT)}" '
        f'euler="1.570796 0 0" material="disk_mat" group="3"/>'
    )

    if disk_idx == 0:
        w(
            f'{indent}  <site name="base_site" pos="0 {fmt(center_y)} 0" '
            f'size="0.00075" rgba="1 1 1 1"/>'
        )
    if disk_idx == N_DISKS - 1:
        w(
            f'{indent}  <site name="tip_site" pos="0 {fmt(JOINT_TO_DISK_TOP)} 0" '
            f'size="0.0009" rgba="1 1 0 1"/>'
        )

    for name, angle_deg, rgba in TENDONS:
        x, y, z = tendon_site_pos(angle_deg)
        y += center_y
        w(
            f'{indent}  <site name="hole_{name}_{disk_idx}" '
            f'pos="{fmt(x)} {fmt(y)} {fmt(z)}" size="{fmt(HOLE_SITE_SIZE)}" '
            f'rgba="{rgba}"/>'
        )
        w(
            f'{indent}  <geom name="hole_{name}_{disk_idx}_dot" type="sphere" '
            f'pos="{fmt(x)} {fmt(y)} {fmt(z)}" size="{fmt(HOLE_DOT_SIZE)}" '
            f'material="hole_{name}_mat" contype="0" conaffinity="0"/>'
        )


def generate_xml() -> str:
    lines: list[str] = []
    w = lines.append

    w('<?xml version="1.0" ?>')
    w('<mujoco model="videoscope_one_section">')
    w('  <compiler angle="radian" autolimits="true"/>')
    w('')
    w('  <visual>')
    w('    <global offwidth="1280" offheight="960"/>')
    w('    <quality shadowsize="2048"/>')
    w(
        '    <headlight ambient="0.45 0.45 0.45" diffuse="0.85 0.85 0.85" '
        'specular="0.35 0.35 0.35"/>'
    )
    w('  </visual>')
    w('')
    w('  <option gravity="0 0 0" timestep="0.0005" integrator="implicitfast">')
    w('    <flag contact="enable"/>')
    w('  </option>')
    w('')
    w('  <default>')
    w('    <geom friction="0.8 0.02 0.01" condim="3" margin="0.0001"/>')
    w(f'    <tendon width="{fmt(TENDON_WIDTH)}"/>')
    w('  </default>')
    w('')
    w('  <asset>')
    w('    <mesh name="new8_mesh" file="new8.stl" scale="0.001 0.001 0.001"/>')
    w('    <material name="new8_mat" rgba="0.72 0.78 0.82 1" specular="0.75" shininess="0.9" reflectance="0.25"/>')
    w('    <material name="disk_mat" rgba="0.70 0.78 0.82 0.10" specular="0.2" shininess="0.2"/>')
    w('    <material name="base_mat" rgba="0.16 0.18 0.20 1"/>')
    w('    <material name="joint_ball_mat" rgba="0.05 0.05 0.05 1" specular="0.7" shininess="0.8"/>')
    for name, _angle, rgba in TENDONS:
        w(f'    <material name="hole_{name}_mat" rgba="{rgba}"/>')
    w('  </asset>')
    w('')
    w('  <worldbody>')
    w('    <light pos="0.06 -0.08 0.10" dir="-0.5 0.6 -0.7" diffuse="0.9 0.9 0.9"/>')
    w('    <camera name="overview" pos="0.075 -0.115 0.075" xyaxes="0.837 0.547 0 -0.308 0.472 0.826"/>')
    w('    <body name="base_mount" pos="0 0 0">')
    w('      <geom name="base_block" type="cylinder" pos="0 -0.0022 0" size="0.0045 0.0015" euler="1.570796 0 0" material="base_mat"/>')

    for disk_idx in range(N_DISKS):
        add_disk(lines, disk_idx, "      " + "  " * disk_idx)

    for disk_idx in range(N_DISKS - 1, -1, -1):
        w(f'{"      " + "  " * disk_idx}</body>')

    w('    </body>')
    w('  </worldbody>')
    w('')
    w('  <tendon>')
    for name, _angle, rgba in TENDONS:
        w(f'    <spatial name="tendon_{name}" width="{fmt(TENDON_WIDTH)}" rgba="{rgba}">')
        for disk_idx in range(N_DISKS):
            w(f'      <site site="hole_{name}_{disk_idx}"/>')
        w('    </spatial>')
    w('  </tendon>')
    w('')
    w('  <actuator>')
    b0 = ACTUATOR_KP * REST_LENGTH
    for name, _angle, _rgba in TENDONS:
        w(
            f'    <general name="pull_{name}" tendon="tendon_{name}" '
            f'gaintype="fixed" biastype="affine" '
            f'ctrllimited="true" ctrlrange="-{fmt(MAX_PULL)} {fmt(MAX_PULL)}" '
            f'forcelimited="true" forcerange="-{fmt(ACTUATOR_PULL_FORCE_LIMIT)} 0" '
            f'gainprm="-{fmt(ACTUATOR_KP)} 0 0" '
            f'biasprm="{fmt(b0)} -{fmt(ACTUATOR_KP)} -{fmt(ACTUATOR_DAMPING)}"/>'
        )
    w('  </actuator>')
    w('')
    w('  <contact>')
    for disk_idx in range(N_DISKS - 1):
        w(f'    <exclude body1="disk_{disk_idx}" body2="disk_{disk_idx + 1}"/>')
    w('  </contact>')
    w('</mujoco>')

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one-section videoscope MuJoCo XML")
    parser.add_argument(
        "--output",
        default="videoscope_one_section.xml",
        help="Output XML filename",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, args.output)
    xml = generate_xml()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml)

    print("Generated one-section videoscope model")
    print(f"  XML:              {output_path}")
    print(f"  Joints:           {N_JOINTS}")
    print(f"  Disks:            {N_DISKS}")
    print(f"  Disk OD:          {DISK_OD * 1000:.1f} mm")
    print(f"  Tendon holes:     4 at r = {HOLE_RADIUS * 1000:.1f} mm")
    print(f"  Disk spacing:     {DISK_SPACING * 1000:.1f} mm")
    print(f"  Rotation ball:    {ROTATION_BALL_DIAMETER * 1000:.1f} mm diameter")
    print(f"  Joint to top:     {JOINT_TO_DISK_TOP * 1000:.1f} mm")
    print(f"  Hole plane:       {DISK_CENTER_FROM_JOINT * 1000:.1f} mm from joint")
    print(f"  Rest tendon len:  {REST_LENGTH * 1000:.1f} mm")
    print(f"  Pair pull/release: +/-{MAX_PULL * 1000:.1f} mm")


if __name__ == "__main__":
    main()
