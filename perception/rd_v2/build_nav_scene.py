"""
Generate a MuJoCo scene with ONLY the videoscope (no colon mesh).

The colon is rendered separately in Blender by `blender_render_worker.py` so
the fine-tuned DA3 model sees images that match its training distribution.
MuJoCo handles only the scope physics + the mocap base advancement; the
camera in this scene is used only to extract tip-cam world pose, never to
render.

Mirrors the structure of `Stage4/build_combined_scene.py`, minus the colon
mesh asset and the colon visual geom in the worldbody.

Inputs:
    anchor_pos: (3,) float ndarray, world coords of scope base at start.
    anchor_quat_wxyz: (4,) float, MuJoCo (w, x, y, z) quat for scope base.
    out_path: where to write the XML.

Outputs:
    Writes the XML to disk and returns the Path.

Self-contained transfer version: generate_videoscope_one_section.py and
new8.stl are bundled in the same directory as this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import generate_videoscope_one_section as scope_gen  # noqa: E402


def build_nav_scene(
    anchor_pos: np.ndarray,
    anchor_quat_wxyz: np.ndarray,
    out_path: Path,
) -> Path:
    # Absolute path to the bundled STL so the XML works regardless of CWD.
    scope_mesh_rel = (HERE / "new8.stl").resolve().as_posix()
    qw, qx, qy, qz = (float(v) for v in anchor_quat_wxyz)
    quat_str = f"{qw:.7f} {qx:.7f} {qy:.7f} {qz:.7f}"

    lines: list[str] = []
    w = lines.append

    w('<?xml version="1.0" ?>')
    w('<mujoco model="nav_scene">')
    w('  <compiler angle="radian" autolimits="true"/>')
    w('')
    w('  <visual>')
    w('    <map znear="0.001" zfar="5.0"/>')
    w('    <quality shadowsize="2048"/>')
    w('    <global offwidth="1280" offheight="960"/>')
    w('    <headlight active="0" ambient="0.3 0.3 0.3" diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3"/>')
    w('  </visual>')
    w('')
    w('  <option gravity="0 0 0" timestep="0.0005" integrator="implicitfast">')
    w('    <flag contact="enable"/>')
    w('  </option>')
    w('')
    w('  <default>')
    w('    <geom friction="0.8 0.02 0.01" condim="3" margin="0.0001"/>')
    w(f'    <tendon width="{scope_gen.fmt(scope_gen.TENDON_WIDTH)}"/>')
    w('  </default>')
    w('')

    w('  <asset>')
    w(f'    <mesh name="new8_mesh" file="{scope_mesh_rel}" scale="0.001 0.001 0.001"/>')
    w('    <material name="new8_mat" rgba="0.72 0.78 0.82 1" specular="0.75" shininess="0.9" reflectance="0.25"/>')
    w('    <material name="disk_mat" rgba="0.70 0.78 0.82 0.10" specular="0.2" shininess="0.2"/>')
    w('    <material name="base_mat" rgba="0.16 0.18 0.20 1"/>')
    w('    <material name="joint_ball_mat" rgba="0.05 0.05 0.05 1" specular="0.7" shininess="0.8"/>')
    for name, _angle, rgba in scope_gen.TENDONS:
        w(f'    <material name="hole_{name}_mat" rgba="{rgba}"/>')
    w('  </asset>')
    w('')

    w('  <worldbody>')
    w(
        f'    <body name="scope_anchor" mocap="true" '
        f'pos="{scope_gen.fmt(anchor_pos[0])} '
        f'{scope_gen.fmt(anchor_pos[1])} {scope_gen.fmt(anchor_pos[2])}" '
        f'quat="{quat_str}">'
    )
    w('      <body name="base_mount" pos="0 0 0">')
    w(
        '        <geom name="base_block" type="box" pos="0 -0.0022 0" '
        'size="0.008 0.0015 0.008" material="base_mat"/>'
    )

    for disk_idx in range(scope_gen.N_DISKS):
        scope_gen.add_disk(lines, disk_idx, "        " + "  " * disk_idx)

        if disk_idx == scope_gen.N_DISKS - 1:
            tip_indent = "        " + "  " * disk_idx + "  "
            tip_y = scope_gen.fmt(scope_gen.JOINT_TO_DISK_TOP)
            w(
                f'{tip_indent}<camera name="tip_cam" pos="0 {tip_y} 0" '
                f'xyaxes="-1 0 0 0 0 -1" fovy="140"/>'
            )
            w(
                f'{tip_indent}<light name="tip_light" pos="0 {tip_y} 0" '
                f'dir="0 1 0" directional="false" castshadow="false" '
                f'diffuse="1.0 0.90 0.82" specular="0.045 0.035 0.03" '
                f'attenuation="1 18 140" cutoff="70" exponent="1"/>'
            )

    for disk_idx in range(scope_gen.N_DISKS - 1, -1, -1):
        w(f'{"        " + "  " * disk_idx}</body>')

    w('      </body>')
    w('    </body>')
    w('  </worldbody>')
    w('')

    w('  <tendon>')
    for name, _angle, rgba in scope_gen.TENDONS:
        w(
            f'    <spatial name="tendon_{name}" '
            f'width="{scope_gen.fmt(scope_gen.TENDON_WIDTH)}" rgba="{rgba}">'
        )
        for disk_idx in range(scope_gen.N_DISKS):
            w(f'      <site site="hole_{name}_{disk_idx}"/>')
        w('    </spatial>')
    w('  </tendon>')
    w('')

    w('  <actuator>')
    b0 = scope_gen.ACTUATOR_KP * scope_gen.REST_LENGTH
    for name, _angle, _rgba in scope_gen.TENDONS:
        w(
            f'    <general name="pull_{name}" tendon="tendon_{name}" '
            f'gaintype="fixed" biastype="affine" '
            f'ctrllimited="true" '
            f'ctrlrange="-{scope_gen.fmt(scope_gen.MAX_PULL)} {scope_gen.fmt(scope_gen.MAX_PULL)}" '
            f'forcelimited="true" '
            f'forcerange="-{scope_gen.fmt(scope_gen.ACTUATOR_PULL_FORCE_LIMIT)} 0" '
            f'gainprm="-{scope_gen.fmt(scope_gen.ACTUATOR_KP)} 0 0" '
            f'biasprm="{scope_gen.fmt(b0)} '
            f'-{scope_gen.fmt(scope_gen.ACTUATOR_KP)} '
            f'-{scope_gen.fmt(scope_gen.ACTUATOR_DAMPING)}"/>'
        )
    w('  </actuator>')
    w('')

    w('  <contact>')
    for disk_idx in range(scope_gen.N_DISKS - 1):
        w(f'    <exclude body1="disk_{disk_idx}" body2="disk_{disk_idx + 1}"/>')
    w('  </contact>')
    w('</mujoco>')

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":
    # Smoke test: build with identity pose at origin.
    out = Path(__file__).parent / "nav_scene.xml"
    build_nav_scene(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        out,
    )
    print(f"Wrote {out}")
