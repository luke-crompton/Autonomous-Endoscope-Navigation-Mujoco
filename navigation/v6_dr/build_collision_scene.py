"""
Build the Task-1 collision scene XML for a given random-colon seed.

Combines:
  - Visual mesh: high-res random colon (no collision)
  - Collision:   per-triangle thin-prism geoms from the low-res random colon
                 (hidden by default via group="3")
  - Scope:       25-disk Stage 4 videoscope, mocap-anchored
  - Anchor pose: tip at SCOPE_TIP_AT_S (default 5 mm in from entrance),
                 base 62.5 mm back along -tangent so body extends outside
  - Tip cam + tip light at the last disk (lighting matches v1)

Returns (xml_path, centreline_array) so a caller can step the mocap base
along the centreline.

CLI:
    python build_collision_scene.py --seed 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import generate_videoscope_one_section as scope_gen  # noqa: E402

from colon_generator import generate_colon, write_binary_stl  # noqa: E402
from mesh_to_thin_prisms import (  # noqa: E402
    emit_triangle_prism_meshes,
    rotation_matrix_to_quat,
)


SCENE_DIR = HERE / "scenes"
SCENE_DIR.mkdir(exist_ok=True)

SCOPE_TIP_AT_S = 0.010     # tip is 10 mm into the colon (was 5; deeper spawn lets slide floor prevent escape without immediately popping tip)
WALL_THICKNESS_M = 0.003   # 3 mm prism thickness (was 0.5 mm; 6x thicker to
                           # stop tunnelling now that the advance block is
                           # removed and the scope can push forward in contact)


# ---------------------------------------------------------------------------
# Centreline / anchor pose helpers
# ---------------------------------------------------------------------------

def arc_length(centreline: np.ndarray) -> np.ndarray:
    """Cumulative arc length array, same length as centreline."""
    ds = np.linalg.norm(np.diff(centreline, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(ds)])


def position_at_s(centreline: np.ndarray, s: float) -> np.ndarray:
    """Linearly interpolated centreline position at arc length s."""
    s_arr = arc_length(centreline)
    s = float(np.clip(s, 0.0, s_arr[-1]))
    idx = int(np.searchsorted(s_arr, s, side="right") - 1)
    idx = max(0, min(idx, len(centreline) - 2))
    frac = (s - s_arr[idx]) / max(s_arr[idx + 1] - s_arr[idx], 1e-12)
    return centreline[idx] * (1 - frac) + centreline[idx + 1] * frac


def tangent_at_s(centreline: np.ndarray, s: float) -> np.ndarray:
    """Unit tangent at arc length s (centred difference, then normalised)."""
    s_arr = arc_length(centreline)
    s = float(np.clip(s, 0.0, s_arr[-1]))
    idx = int(np.argmin(np.abs(s_arr - s)))
    tangents = np.gradient(centreline, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12
    return tangents[idx]


def pose_for_base_s(
    centreline: np.ndarray, s_base: float,
) -> tuple[tuple[float, float, float], str]:
    """Return (base_pos, quat_str) for the scope base mocap at centreline arc
    length s_base, with scope local +Y aligned to the local tangent.

    For s_base >= 0 the base sits exactly on the centreline at that arc
    length, so the scope body follows the colon curve. For s_base < 0
    (e.g. just after reset, when most of the scope body lies outside the
    entrance), the pose is a linear extrapolation backward from centreline[0]
    along the tangent at s=0 -- so the body just extends straight back out
    of the cap.
    """
    if s_base >= 0.0:
        pos = position_at_s(centreline, s_base)
        tangent = tangent_at_s(centreline, s_base)
    else:
        tangent = tangent_at_s(centreline, 0.0)
        pos = centreline[0] + float(s_base) * tangent

    # Same frame construction as anchor_pose_for_tip_at_s.
    y_w = tangent
    world_x = np.array([1.0, 0.0, 0.0])
    x_w = world_x - np.dot(world_x, y_w) * y_w
    if np.linalg.norm(x_w) < 1e-6:
        world_x = np.array([0.0, 1.0, 0.0])
        x_w = world_x - np.dot(world_x, y_w) * y_w
    x_w /= np.linalg.norm(x_w)
    z_w = np.cross(x_w, y_w)
    R = np.column_stack([x_w, y_w, z_w])
    q = rotation_matrix_to_quat(R)
    quat_str = f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"
    return tuple(float(v) for v in pos), quat_str


def anchor_pose_for_tip_at_s(
    centreline: np.ndarray, s_tip: float,
) -> tuple[tuple[float, float, float], str]:
    """Return (base_pos, quat_str) so the scope's tip lands at arc length
    s_tip along the centreline with local +Y aligned to the local tangent.
    Base sits scope_gen.REST_LENGTH back along -tangent."""
    tip_pos = position_at_s(centreline, s_tip)
    tangent = tangent_at_s(centreline, s_tip)
    base_pos = tip_pos - scope_gen.REST_LENGTH * tangent

    # Orthonormal frame: local +Y = tangent. Local +X = world +X projected
    # perpendicular to tangent (fallback +Y if degenerate). Local +Z = X x Y.
    y_w = tangent
    world_x = np.array([1.0, 0.0, 0.0])
    x_w = world_x - np.dot(world_x, y_w) * y_w
    if np.linalg.norm(x_w) < 1e-6:
        world_x = np.array([0.0, 1.0, 0.0])
        x_w = world_x - np.dot(world_x, y_w) * y_w
    x_w /= np.linalg.norm(x_w)
    z_w = np.cross(x_w, y_w)
    R = np.column_stack([x_w, y_w, z_w])
    q = rotation_matrix_to_quat(R)
    quat_str = f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"
    return tuple(float(v) for v in base_pos), quat_str


# ---------------------------------------------------------------------------
# XML builder
# ---------------------------------------------------------------------------

def build_scene(
    seed: int,
    out_path: Path | None = None,
    s_tip: float = SCOPE_TIP_AT_S,
    wall_thickness_m: float = WALL_THICKNESS_M,
) -> tuple[Path, np.ndarray]:
    """Generate the random colon for `seed`, write the scene XML, return
    (xml_path, centreline).  Always regenerates so training always uses the
    current colon-generator code."""
    colon = generate_colon(seed)
    actual_seed = colon["seed"]
    if out_path is None:
        out_path = SCENE_DIR / f"collision_scene_seed{actual_seed}.xml"

    # Write the visual and collision STLs into the scenes/ folder so the
    # XML can reference them by basename.
    visual_stl = SCENE_DIR / f"colon_visual_seed{actual_seed}.stl"
    coll_stl   = SCENE_DIR / f"colon_collision_seed{actual_seed}.stl"
    write_binary_stl(colon["visual_vertices"], colon["visual_triangles"], visual_stl)
    write_binary_stl(colon["collision_vertices"], colon["collision_triangles"], coll_stl)

    asset_xml, geom_xml, n_prisms = emit_triangle_prism_meshes(
        coll_stl,
        thickness_m=wall_thickness_m,
        asset_indent="    ",
        geom_indent="      ",
    )
    # Hide collision prism geoms behind group=3 in the default visualisation.
    # solref="0.002 1": 2 ms response time constant (vs default 20 ms — 10x
    # stiffer). Prevents scope body tunnelling through 3 mm wall prisms.
    # solimp="0.99 0.9999 0.001 0.5 2": near-rigid impedance (default is
    # "0.9 0.95 0.001 0.5 2" which allows significant penetration).
    geom_xml = geom_xml.replace(
        'mass="0"/>',
        'mass="0" group="3" solref="0.002 1" solimp="0.99 0.9999 0.001 0.5 2"/>',
    )

    centreline = colon["centreline"]
    anchor_pos, anchor_quat = anchor_pose_for_tip_at_s(centreline, s_tip)

    scope_mesh_rel = (Path("..") / "new8.stl").as_posix()

    lines: list[str] = []
    w = lines.append

    w('<?xml version="1.0" ?>')
    w('<mujoco model="collision_scene">')
    w('  <compiler angle="radian" autolimits="true"/>')
    w('')
    w('  <visual>')
    w('    <map znear="0.001" zfar="5.0"/>')
    w('    <quality shadowsize="2048"/>')
    w('    <global offwidth="1280" offheight="960"/>')
    w(
        '    <headlight active="0" ambient="0.3 0.3 0.3" '
        'diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3"/>'
    )
    w('  </visual>')
    w('')
    w('  <option gravity="0 0 0" timestep="0.0005" integrator="implicitfast">')
    w('    <flag contact="enable"/>')
    w('  </option>')
    w('')
    w('  <default>')
    w('    <geom friction="0.8 0.02 0.01" condim="3" margin="0.0005"/>')
    w(f'    <tendon width="{scope_gen.fmt(scope_gen.TENDON_WIDTH)}"/>')
    w('  </default>')
    w('')

    # ----- assets -----
    w('  <asset>')
    w(f'    <mesh name="colon_visual_mesh" file="{visual_stl.name}"/>')
    w('    <material name="colon_mat"')
    w('              rgba="0.92 0.58 0.55 1"')
    w('              emission="0.005"')
    w('              specular="0.15"')
    w('              shininess="0.30"/>')
    w(f'    <mesh name="new8_mesh" file="{scope_mesh_rel}" scale="0.001 0.001 0.001"/>')
    w(
        '    <material name="new8_mat" rgba="0.72 0.78 0.82 1" '
        'specular="0.75" shininess="0.9" reflectance="0.25"/>'
    )
    w(
        '    <material name="disk_mat" rgba="0.70 0.78 0.82 0.10" '
        'specular="0.2" shininess="0.2"/>'
    )
    w('    <material name="base_mat" rgba="0.16 0.18 0.20 1"/>')
    w(
        '    <material name="joint_ball_mat" rgba="0.05 0.05 0.05 1" '
        'specular="0.7" shininess="0.8"/>'
    )
    for name, _angle, rgba in scope_gen.TENDONS:
        w(f'    <material name="hole_{name}_mat" rgba="{rgba}"/>')
    # Per-triangle prism <mesh> assets
    w(asset_xml)
    w('  </asset>')
    w('')

    # ----- worldbody -----
    w('  <worldbody>')
    # Visual mesh: no collision, visible everywhere
    w(
        '    <geom name="colon_visual" type="mesh" mesh="colon_visual_mesh" '
        'material="colon_mat" contype="0" conaffinity="0"/>'
    )
    # Collision wall: hidden by group=3
    w('    <body name="colon_wall">')
    w(geom_xml)
    w('    </body>')
    w('')

    # Target body: kinematic mocap goal pose for the scope base. The env
    # advances this each step along the centreline. scope_anchor is now a
    # CHILD of target (instead of a separate body connected by weld), so the
    # base's allowed motion is one small slide DOF along target's local +Y
    # (= centreline tangent). Env code caps lag so this local tangent motion
    # cannot become a large shortcut through a curved wall.
    ax, ay, az = anchor_pos
    w(
        f'    <body name="target" mocap="true" '
        f'pos="{ax:.6f} {ay:.6f} {az:.6f}" quat="{anchor_quat}">'
    )
    # scope_anchor: dynamic child with a 1-D slide joint. Pos (0,0,0) means
    # at slide_qpos=0 the scope_anchor is exactly at target's pose. Walls
    # pushing back on the chain transmit force up to this slide DOF; if the
    # force exceeds what the joint damping resists, slide_qpos goes negative
    # and the whole chain can lag slightly instead of being dragged through a
    # contact. Range allows up to 20 mm lag backward, 5 mm overshoot forward.
    w('      <body name="scope_anchor" pos="0 0 0">')
    w(
        '        <joint name="anchor_slide" type="slide" axis="0 1 0" '
        'damping="0.3" range="-0.02 0.005" limited="true"/>'
    )
    w('        <inertial pos="0 0 0" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>')
    w('        <body name="base_mount" pos="0 0 0">')
    w(
        '          <geom name="base_block" type="cylinder" pos="0 -0.0022 0" '
        'size="0.0045 0.0015" euler="1.570796 0 0" material="base_mat"/>'
    )
    for disk_idx in range(scope_gen.N_DISKS):
        scope_gen.add_disk(lines, disk_idx, "          " + "  " * disk_idx)
        if disk_idx == scope_gen.N_DISKS - 1:
            tip_indent = "          " + "  " * disk_idx + "  "
            tip_y = scope_gen.fmt(scope_gen.JOINT_TO_DISK_TOP)
            w(
                f'{tip_indent}<camera name="tip_cam" pos="0 {tip_y} 0" '
                f'xyaxes="-1 0 0 0 0 -1" fovy="85"/>'
            )
            w(
                f'{tip_indent}<light name="tip_light" pos="0 {tip_y} 0" '
                f'dir="0 1 0" directional="false" castshadow="false" '
                f'diffuse="1.0 0.90 0.82" specular="0.045 0.035 0.03" '
                f'attenuation="1 18 140" cutoff="70" exponent="1"/>'
            )
    for disk_idx in range(scope_gen.N_DISKS - 1, -1, -1):
        w(f'{"          " + "  " * disk_idx}</body>')
    w('        </body>')   # close base_mount
    w('      </body>')     # close scope_anchor
    w('    </body>')       # close target
    w('  </worldbody>')
    w('')

    # The previous equality<weld> between scope_anchor and target is gone --
    # the slide joint above provides the constraint directly.

    # ----- tendons / actuators / contact -----
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
            f'ctrlrange="-{scope_gen.fmt(scope_gen.MAX_PULL)} '
            f'{scope_gen.fmt(scope_gen.MAX_PULL)}" '
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
    return out_path, centreline


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--s_tip", type=float, default=SCOPE_TIP_AT_S,
        help="Arc length (m) at which the scope tip starts (default 0.005 = 5 mm in)",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    xml_path, centreline = build_scene(args.seed, args.out, s_tip=args.s_tip)
    t1 = time.perf_counter()
    print(f"Wrote: {xml_path}")
    print(f"  XML size:    {xml_path.stat().st_size / 1024:.0f} KB")
    print(f"  Centreline:  {len(centreline)} points "
          f"({arc_length(centreline)[-1] * 1000:.0f} mm long)")
    print(f"  Build time:  {t1 - t0:.2f} s")


if __name__ == "__main__":
    main()
