"""
Build the Task-1 collision scene XML for a given random-colon seed.

Combines:
  - Visual mesh: high-res random colon (no collision)
  - Collision:   per-triangle thin-prism geoms from the low-res random colon
                 (hidden by default via group="3")
  - Scope:       25-disk Stage 4 videoscope on a 60-link flexible shaft
  - Feed:        4 friction-gripping rollers at the colon entrance drive the
                 shaft in/out (see ROLLER_* constants); shaft_link_0 is a
                 free body held only by roller grip + colon wall contact
  - Anchor pose: tip at SCOPE_TIP_AT_S (default 5 mm in from entrance),
                 base ~600mm back along -tangent so body extends outside
  - Tip cam + tip light at the last disk (lighting matches v1)

Returns (xml_path, centreline_array) so a caller can drive the roller feed
actuators and read back the shaft's true physical position.

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

SCOPE_TIP_AT_S = 0.095     # tip 95mm into the colon at XML build time. Must clear
                           # REST_LENGTH (62.5mm, disk chain) + one shaft link spacing
                           # (10mm) = 72.5mm minimum, or the entire 600mm shaft sits
                           # behind the entrance rollers with a gap -- nothing to grip,
                           # so spinning the rollers does nothing. 95mm gives ~1-2 links
                           # of solid initial engagement past the minimum.
WALL_THICKNESS_M = 0.003   # 3 mm prism thickness (was 0.5 mm; 6x thicker to
                           # stop tunnelling now that the advance block is
                           # removed and the scope can push forward in contact)

# Flexible shaft parameters (v7_shaft)
SHAFT_N_LINKS        = 50       # 50 × 10 mm = 500 mm total shaft length
SHAFT_LINK_SPACING_M = 0.010    # 10 mm centre-to-centre → min bend radius ~29 mm at ±20°
SHAFT_LINK_RADIUS_M  = 0.0045   # 4.5 mm radius → 9 mm OD, matching bending tip
SHAFT_JOINT_RANGE    = 0.349    # ±20° per link (radians)
SHAFT_STIFFNESS      = 0.25     # History: 2.0 -> 1.0 (2026-07-02, real RL-policy viewer logs
                                 # showed sustained max-forward commands with actual_base_s
                                 # completely flat and roller_force staying far below its torque
                                 # ceiling -- kinetic slip, not actuator/friction-limited, both ruled
                                 # out empirically); 1.0 gave a modest, partial improvement under the
                                 # OLD plain-cylinder roller (stall moved further but still occurred).
                                 # 1.0 -> 0.5 after the V-groove roller swap (see ROLLER_* above):
                                 # tested on Linux, got further before hitting the same stall.
                                 # 0.5 -> 0.25 (2026-07-02): at 0.5 the roller was pushing the whole
                                 # time (not idle/slipping-then-stopping) but progress was still too
                                 # slow, AND the shaft did not visibly buckle mid-span -- i.e. still
                                 # stiff enough to resist yielding into a real bend/buckle shape under
                                 # load, which is the behaviour lower stiffness is meant to unlock.
                                 # Lower stiffness means less reaction force needs to be fed back
                                 # through the roller grip to bend the shaft around the same corner,
                                 # which may keep the contact under the static-friction budget
                                 # instead of breaking into slip. NOTE: the only earlier stiffness
                                 # sweep (0.5/1.0/2.0/4.0) was for a DIFFERENT metric (post-release
                                 # creep, not active-drive slip) and found 2.0 optimal there -- values
                                 # below that have not been re-verified against creep under this
                                 # failure mode.
SHAFT_DAMPING        = 0.3      # angular damping per shaft hinge joint
SHAFT_LINK_MASS      = 0.005    # 5 g per link (~150 g total shaft mass)
SHAFT_TIP_BALL_STIFFNESS = 0.25 # stiffness of the single ball joint at the shaft/bending-tip
                                 # junction (shaft_tip_ball below). Was hardcoded at 2.0 and never
                                 # swept before; moved in lockstep with SHAFT_STIFFNESS's 2.0->1.0->
                                 # 0.5->0.25 history (2026-07-02) on the theory that it's the same
                                 # active-drive-slip mechanism -- still untested in isolation.
# Entrance roller feeder — replaces both the old mocap-teleport drive AND the
# frictionless guide cage. 4 friction-gripping rollers at 90-degree intervals
# spin to push/pull the shaft via contact friction, applied right at the
# entrance instead of at the far proximal handle end. shaft_link_0 is now a
# free body (see build_scene()) held in place only by roller grip + colon
# wall contact -- nothing rigidly drives it anymore. Validated in
# Stage4/v7_shaft/roller_feed_prototype/ before this integration: force
# applied at a fixed point near the entrance means the only span that can
# ever be under compressive/buckling load is entrance->tip, not the full
# proximal tail, and the friction grip slips gracefully under resistance
# instead of rigidly overdriving the chain.
# contype=2/conaffinity=2 so rollers only collide with shaft links
# (contype=3/conaffinity=3), not with the colon wall or bending tip disks
# (contype=1/conaffinity=1) -- same bit scheme the old guide cage used.
ROLLER_RADIUS_M   = 0.008    # 8 mm -- radius at the roller's WAIST (z=0, the plane through the
                              # mount point/shaft centreline). Unchanged from the old plain-cylinder
                              # radius, so the centre contact point keeps the exact interference
                              # behaviour the old preload/mount-radius calibration was tuned against.
ROLLER_HALF_LEN_M = 0.006    # 6 mm half-height (12 mm wide roller)
GROOVE_FLARE_M    = 0.0035   # radius GAINED between the waist (z=0, radius=ROLLER_RADIUS_M) and
                              # each shoulder (z=+-half_len, radius=ROLLER_RADIUS_M+GROOVE_FLARE_M).
                              # The groove must flare OUTWARD away from the mount plane, not narrow
                              # inward -- the roller's spin axis is the AZIMUTHAL direction around
                              # the shaft's cross-section (perpendicular to both "toward the shaft"
                              # and "along the shaft"), so a plain constant-radius drum (old design)
                              # only ever touches the shaft at the single z=0 point; the shaft's own
                              # surface recedes in X as |z| grows (sqrt(shaft_r^2 - z^2)), so to keep
                              # touching off-centre the roller surface has to grow to chase it, i.e.
                              # flare outward, cradling the shaft's curvature instead of just kissing
                              # it at one point. (An inward-narrowing V, tried first, was checked with
                              # a standalone smoke test and produces a ~2.4mm gap at the centre --
                              # zero contact anywhere -- because it cuts material away from exactly
                              # the point the old preload calibration relies on.)
                              #
                              # 1.5mm -> 3.5mm (2026-07-02). An isolated 8-roller numeric harness
                              # (full ring geometry, resistive-force-vs-slip sweep, no colon needed)
                              # showed PRELOAD has essentially no effect on transferable force in
                              # this solver regime (see ROLLER_PRELOAD_M below) but FLARE does, and
                              # monotonically: interpolated zero-slip-crossing force was ~5.9N at
                              # 1.5mm, ~6.1N at 2.5mm, ~6.3N at 3.5mm, ~6.6N at 4.5mm -- real but
                              # diminishing (~+0.25N per +1mm), consistent with approaching the
                              # theoretical fully-conforming-cradle limit (~4.5mm flare for this
                              # 4.5mm-radius shaft, see the physics note on GROOVE_FLARE_M's effect
                              # in the numeric-test writeup). 3.5mm chosen as a middle point (+0.4N /
                              # ~7% over 1.5mm) rather than pushing to the ~4.5mm ceiling, partly for
                              # buildability (this also grows the roller's real outer/shoulder radius
                              # from 9.5mm to 11.5mm). NOT yet validated on Linux with the real
                              # bending shaft + colon wall + RL policy -- the numeric test isolates
                              # only the straight-line entrance-grip mechanism.
ROLLER_SHOULDER_RADIUS_M = ROLLER_RADIUS_M + GROOVE_FLARE_M
GROOVE_N_AROUND   = 16        # circumferential resolution of each groove-wall frustum's convex
                              # hull (see _groove_half_vertices). Smooth enough to roll against the
                              # shaft without faceting artifacts; small enough to keep the added
                              # vertex/contact count cheap (2 hulls x 16 pts x 8 rollers).
ROLLER_PRELOAD_M  = 0.0005   # geometric interference -> grip (normal) force. 0.5mm (2026-07-02,
                              # V-groove introduced) -> 0.7mm same day (hypothesis: more preload ->
                              # more normal force -> more friction budget) -> REVERTED back to 0.5mm
                              # (2026-07-02, same day) after the isolated 8-roller numeric harness
                              # (see GROOVE_FLARE_M above) swept preload 0.5/0.7/1.0/1.5/2.0mm -- a
                              # 4x range -- and found ESSENTIALLY NO EFFECT on transferable force
                              # before slip (zero-crossing force flat, if anything trending very
                              # slightly worse at higher preload). Likely mechanism: ROLLER_SOLREF/
                              # ROLLER_SOLIMP below are set very stiff (near-rigid contact), so once
                              # geometric interference is enough to guarantee contact exists, the
                              # converged normal force is set by the load being resisted, not by how
                              # much extra overlap was nominally requested -- preload matters for
                              # "does contact happen" not "how much force transfers once it does".
                              # GROOVE_FLARE_M (contact wrap/area, not squeeze) is the lever that
                              # actually moved the numeric test's result. No reason to carry the
                              # unvalidated 0.7mm bump forward.
                              #
                              # History at 0.5mm under the V-groove (this same geometry): ported
                              # from a number tuned for a DIFFERENT metric (post-release creep)
                              # under the OLD plain-cylinder geometry, never swept for active-drive
                              # slip under the groove specifically.
                              #
                              # History under the OLD plain-cylinder geometry (kept for reference):
                              # 0.0002 (prototype) -> 0.0004 (2026-07-01, cut tail-swing 34.6mm ->
                              # 0.3mm) -> 0.0006 (2026-07-01, fixed post-release creep: swept
                              # 0.4/0.5/0.6/0.75mm, slip was -2.30/-1.10/+0.15/-3.48mm, 0.6mm was
                              # the sweet spot) -> 0.0008 (2026-07-02, attempted fix for active-
                              # drive slip into a bend -- measured NO EFFECT: roller_force and
                              # stall point were identical to the 0.6mm case, ruling out preload
                              # under a plain cylinder as the lever for that specific failure).
ROLLER_MOUNT_RADIUS_M = SHAFT_LINK_RADIUS_M + ROLLER_RADIUS_M - ROLLER_PRELOAD_M
ROLLER_FRICTION   = "1.2 0.02 0.01"   # high friction -- opposite of the old cage's near-zero friction.
                              # Re-tested at 2.0 (2026-07-02) under a real RL-policy stall (roller
                              # spinning at near-max forward, actual_base_s flat, tip not wall-jammed):
                              # made net advance slower with no fix to the stall -- confirms the earlier
                              # post-release-creep finding (1.2->2.0->3.0 harmful) generalises to this
                              # active-driving scenario too. Reverted to 1.2.
# Tighter contact stiffness than MuJoCo's default (~"0.02 1" / "0.9 0.95 0.001 0.5 2") --
# this was the dominant lever for grip tightness, far more than preload alone: at the
# SAME 0.0002 preload, tightening solref/solimp alone cut tail-swing from 34.6mm to 1.6mm.
# Matches the same stiffening pattern already used for the colon wall prisms.
ROLLER_SOLREF     = "0.001 1"
ROLLER_SOLIMP     = "0.99 0.9999 0.0005 0.5 2"
ROLLER_DAMPING    = 0.02             # spin-axis bearing damping
ROLLER_ARMATURE   = 0.0005           # added rotor inertia -- without this the roller's true inertia
                                      # (~1e-6) lets contact impulses cause huge angular accelerations
                                      # each substep (numerical chattering), confirmed in the prototype
# Position-controlled (not velocity-controlled) roller actuators (2026-07-01).
# A pure <velocity> actuator only resists CHANGES in speed (torque = kv*(0 -
# qvel) at rest) -- it has no notion of holding a fixed angle, so any
# external torque (e.g. residual bending stress relaxing right after an
# active drive+steer phase releases) can slowly creep the roller and hence
# the shaft, even at ctrl=0. Measured ~2mm of "reversing" over ~1.25s after
# releasing commands. Real motorized feed rollers don't back-drive like this
# -- IRL they hold position when not commanded. A <position> actuator
# (torque = kp*(target_angle - qpos) - kv*qvel) gives that genuinely: the
# caller now tracks a target ANGLE that only advances while a key/action is
# held, and stays fixed (hence the roller stays put) otherwise. See
# manual_test.py's target-angle accumulator.
ROLLER_ACTUATOR_KP = 60.0             # position gain. Tested 6 -> 30 -> 60: post-release creep after
                                      # a drive+steer phase went 5.80mm -> 4.76mm -> 0.56mm -- needed to
                                      # be much stiffer than the first guess (6.0) to actually hold, not
                                      # just slow the creep. Paired with the force limit below.
ROLLER_ACTUATOR_KV = 0.15            # damping term, same value the old velocity actuator used
ROLLER_ACTUATOR_FORCE_LIMIT = 5.0    # N*m; raised alongside kp (was 1.0) -- at kp=60 with the old 1.0
                                      # force cap the actuator saturated before it could hold position.
                                      # Doubled 5.0->10.0 then reverted (2026-07-02): real RL-policy
                                      # viewer logs during a genuine stall showed roller_force staying
                                      # at 0.03-0.59N regardless of the ceiling -- the motor was never
                                      # close to saturating, so the higher cap changed nothing. Confirms
                                      # this is a grip/friction slip problem, not an actuator-torque one.

# Second roller ring: a single ring can only constrain lateral position at one
# point -- it's mechanically a hinge, not a clamp, so it can't resist rotation.
# Any reaction moment from the tip contacting a real bend pivots the entire
# unsupported proximal tail (up to ~500mm) around that one point, which shows
# up as the tail visibly swinging even though individual joint angles stay
# small (lever-arm amplification: swing ~= tail_length * sin(angle)). A second
# ring, axially spaced from the first, constrains rotation too -- same
# principle as a two-point clamp/chuck. Offset EXTERNALLY (further out along
# -entrance_tangent, away from the colon) so the two rings clamp a short
# straight span straddling the entrance boundary itself.
ROLLER_RING_SPACING_M = 0.020   # 20 mm external offset for the second ring;
                                 # rollers are 12mm wide (+/-6mm half-length),
                                 # leaving an 8mm gap between rings -- no overlap


def _groove_half_vertices(
    r_waist: float, r_shoulder: float, half_len: float, n_around: int,
) -> str:
    """Point cloud (as a MuJoCo <mesh vertex="..."> string) for one wall of a
    V-groove roller: a frustum from the waist (radius r_waist at local z=0,
    the plane through the shaft centreline) flaring out to the shoulder
    (radius r_shoulder at local z=+half_len).

    No face indices are given -- MuJoCo auto-computes the convex hull of the
    point set, same technique mesh_to_thin_prisms.py uses for its per-
    triangle prisms. A frustum's own convex hull IS the frustum, so this is
    lossless. Two instances of this same mesh (placed with zaxis flipped for
    the second) form the two walls of the groove, straddling the roller's
    mount point. They MUST stay as two separate geoms/mesh assets rather
    than one mesh spanning shoulder->waist->shoulder: a single such mesh
    would get convex-hulled straight back into a plain cylinder (the waist
    point sits inside the hull of the two shoulder rings), silently erasing
    the groove -- exactly the failure mode that motivated the colon wall's
    thin-prism decomposition.
    """
    theta = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    pts = []
    for z, r in ((0.0, r_waist), (half_len, r_shoulder)):
        for t in theta:
            pts.append((r * np.cos(t), r * np.sin(t), z))
    return "  ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in pts)


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


def tangent_at_s(centreline: np.ndarray, s: float, window_m: float = 0.02) -> np.ndarray:
    """Unit tangent at arc length s, averaged over a +/-window_m arc-length
    window rather than a single 2-point difference -- a lone sample can be
    skewed by an upcoming bend's curvature bleeding into the Catmull-Rom fit
    (most consequential at s=0, which anchors both the shaft spawn pose and
    the roller frame in build_scene)."""
    s_arr = arc_length(centreline)
    s = float(np.clip(s, 0.0, s_arr[-1]))
    tangents = np.gradient(centreline, axis=0)
    mask = (s_arr >= s - window_m) & (s_arr <= s + window_m)
    if not np.any(mask):
        mask = np.zeros_like(s_arr, dtype=bool)
        mask[int(np.argmin(np.abs(s_arr - s)))] = True
    avg = tangents[mask].mean(axis=0)
    norm = np.linalg.norm(avg)
    if norm < 1e-12:
        avg = tangents[int(np.argmin(np.abs(s_arr - s)))]
        norm = np.linalg.norm(avg) + 1e-12
    return avg / norm


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

    scope_mesh_rel = (Path("..") / "new8.stl").as_posix()

    lines: list[str] = []
    w = lines.append

    w('<?xml version="1.0" ?>')
    w('<mujoco model="collision_scene">')
    w('  <compiler angle="radian" autolimits="true"/>')
    w('  <size memory="32M"/>')
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
    # Scene default friction lowered 0.8/0.02/0.01 -> 0.08/0.005/0.001 (2026-07-02) to
    # match the shaft's own explicit "lubricated real scope" friction (see its geom
    # emission below). Colon wall prisms and bending-tip disk geoms both have no
    # explicit friction override, so they inherit this default -- wall, disks, and
    # shaft now all combine to the SAME low friction via MuJoCo's element-wise-max
    # rule (both sides low -> max is low), no `priority` attribute needed anywhere.
    # Confirmed safe for roller grip two ways: (1) magnitude -- ROLLER_FRICTION=1.2
    # stays higher than this default no matter how low it goes, so max(1.2, low)
    # always wins for roller; (2) rollers never even contact the wall or disks in
    # the first place (contype=2/conaffinity=2 vs wall/disk's contype=1/conaffinity=1
    # -- rollers only ever collide with the shaft's contype=3/conaffinity=3). Verified
    # empirically by reading d.contact[i].friction directly for both pairings before
    # applying this. This sidesteps the earlier `priority`-based attempt entirely --
    # that approach broke because MuJoCo's priority override also silently swaps
    # solref/solimp, not just friction; lowering the shared default needs no priority
    # at all, so that failure mode doesn't apply here.
    w('    <geom friction="0.08 0.005 0.001" condim="3" margin="0.0005"/>')
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
    # V-groove roller wall mesh: one shared frustum reused by both walls of
    # every roller (zaxis flipped for the mirrored wall) -- see
    # _groove_half_vertices for why this is one mesh, not two.
    _groove_verts = _groove_half_vertices(
        ROLLER_RADIUS_M, ROLLER_SHOULDER_RADIUS_M, ROLLER_HALF_LEN_M, GROOVE_N_AROUND,
    )
    w(f'    <mesh name="roller_groove_half" vertex="{_groove_verts}"/>')
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

    # Compute target initial position: shaft proximal end, outside the colon.
    # The target mocap moves only along the fixed entrance tangent (1-D push/pull).
    _entrance_tangent = tangent_at_s(centreline, 0.0)
    _tip_world = position_at_s(centreline, s_tip)
    _disk0_world = _tip_world - scope_gen.REST_LENGTH * _entrance_tangent
    _shaft_length = SHAFT_N_LINKS * SHAFT_LINK_SPACING_M
    _target_world = _disk0_world - _shaft_length * _entrance_tangent

    # Quaternion: local +Y aligned to entrance tangent (fixed throughout episode).
    _y_w = _entrance_tangent
    _wx = np.array([1.0, 0.0, 0.0])
    _x_w = _wx - np.dot(_wx, _y_w) * _y_w
    if np.linalg.norm(_x_w) < 1e-6:
        _wx = np.array([0.0, 1.0, 0.0])
        _x_w = _wx - np.dot(_wx, _y_w) * _y_w
    _x_w /= np.linalg.norm(_x_w)
    _z_w = np.cross(_x_w, _y_w)
    _R = np.column_stack([_x_w, _y_w, _z_w])
    _q = rotation_matrix_to_quat(_R)
    _target_quat = f"{_q[0]:.6f} {_q[1]:.6f} {_q[2]:.6f} {_q[3]:.6f}"
    _tx, _ty, _tz = (float(v) for v in _target_world)

    # Entrance roller feeder: 4 friction-gripping rollers at the colon entrance
    # (s=0), replacing the old frictionless guide cage. Each roller's spin
    # axis is computed as radial_direction x entrance_tangent -- perpendicular
    # to both the direction toward the shaft axis and the shaft's own length,
    # so the roller's rim surface velocity at the contact point is parallel
    # to the shaft (the capstan/feed direction) when spun. Validated sign
    # convention (from the prototype): negative ctrl = insertion, positive
    # ctrl = retraction.
    # Two rings: ring 0 at the entrance (s=0), ring 1 offset EXTERNALLY
    # (further out along -entrance_tangent, away from the colon) by
    # ROLLER_RING_SPACING_M. Together they clamp a short straight span
    # straddling the entrance boundary, constraining both lateral position
    # AND rotation there -- ring 0 alone is mechanically a hinge, not a clamp.
    _e_pos = position_at_s(centreline, 0.0)
    for _ring in range(2):
        _ring_center = _e_pos - _ring * ROLLER_RING_SPACING_M * _y_w
        for _gi, _ga in enumerate([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]):
            _radial = np.cos(_ga) * _x_w + np.sin(_ga) * _z_w
            _spin_axis = np.cross(_radial, _y_w)
            _spin_axis /= np.linalg.norm(_spin_axis)
            _mount_pos = _ring_center + ROLLER_MOUNT_RADIUS_M * _radial
            _mx, _my, _mz = (float(v) for v in _mount_pos)
            _ax, _ay, _az = (float(v) for v in _spin_axis)
            _rname = f"roller_r{_ring}_{_gi}"
            w(f'    <body name="{_rname}" pos="{_mx:.6f} {_my:.6f} {_mz:.6f}">')
            w(
                f'      <joint name="{_rname}_spin" type="hinge" '
                f'axis="{_ax:.6f} {_ay:.6f} {_az:.6f}" damping="{ROLLER_DAMPING}" '
                f'armature="{ROLLER_ARMATURE}"/>'
            )
            w('      <inertial pos="0 0 0" mass="0.02" diaginertia="1e-6 1e-6 1e-6"/>')
            # V-groove: two mirrored frustum walls sharing this body/joint,
            # straddling the mount point (zaxis flipped -> the mesh's local
            # z=0..+half_len range gets mirrored to the other side). See
            # _groove_half_vertices for why these must be two geoms, not one.
            for _side, _tag in ((1.0, "p"), (-1.0, "n")):
                _sax, _say, _saz = (_side * _ax, _side * _ay, _side * _az)
                w(
                    f'      <geom name="{_rname}_geom_{_tag}" type="mesh" '
                    f'mesh="roller_groove_half" '
                    f'zaxis="{_sax:.6f} {_say:.6f} {_saz:.6f}" '
                    f'contype="2" conaffinity="2" '
                    f'friction="{ROLLER_FRICTION}" '
                    f'solref="{ROLLER_SOLREF}" solimp="{ROLLER_SOLIMP}" '
                    f'mass="0" material="base_mat"/>'
                )
            w('    </body>')
    w('')

    # Flexible shaft chain: SHAFT_N_LINKS rigid capsule bodies. shaft_link_0
    # has a FREE joint (no more mocap teleport / rigid weld) -- it's held in
    # place only by roller grip and colon wall contact, so resistance at the
    # tip makes the drive slip instead of rigidly buckling the chain.
    #
    # Bend points are true 2-DOF universal joints (X-hinge then Z-hinge
    # co-located via a zero-length dummy body) at every OTHER junction
    # (k=1,3,5,...,59 -- 30 total), with the junctions in between (k=2,4,...)
    # rigidly fused (no joint). This replaces the old scheme of a single-axis
    # hinge alternating X/Z at every junction (59 single-DOF hinges): that
    # design needed two separate, spatially-offset hinges to approximate one
    # arbitrary bend direction, and each hinge's axis is defined in its own
    # local frame -- which itself gets rotated by every joint before it in
    # the chain, so the two axes ended up contributing very unevenly (one
    # axis doing nearly all the work for long stretches, looking like "half
    # the joints aren't flexing"). A true 2-DOF joint bends in the exact
    # combined direction needed at a single point, with no such asymmetry.
    # Total shaft DOF is unchanged (30 joints x 2 DOF = 60, vs 59 before) --
    # this trades joint COUNT/spacing (30 bend points over 20mm segments
    # instead of 59 over 10mm) for correctness, not more compute. Still zero
    # axial twist DOF either way (only X/Z axes ever used), matching a real
    # scope shaft's torsional rigidity.
    # Optimisation (2026-07-01): the "rigid fusion" link of each pair (even k)
    # used to get its own separate 10mm capsule geom even though it has zero
    # relative motion from the universal-joint link right before it -- pure
    # collision-detection overhead for no physics benefit. Merged into ONE
    # 20mm capsule per pair (mass combined, same total), halving shaft geom
    # count (60->31) with zero change to DOF, bending resolution, or
    # behaviour. Only the very last segment (u=59) has no fusion partner
    # (nothing follows it but the disk chain) and stays a standalone 10mm
    # capsule.
    _sp   = f"{SHAFT_LINK_SPACING_M:.6f}"
    _sr   = f"{SHAFT_LINK_RADIUS_M:.6f}"
    _sm   = f"{SHAFT_LINK_MASS:.6f}"
    _sh   = f"{SHAFT_LINK_SPACING_M / 2.0:.6f}"
    _sp2  = f"{2 * SHAFT_LINK_SPACING_M:.6f}"
    _sh2  = f"{SHAFT_LINK_SPACING_M:.6f}"
    _sm2  = f"{2 * SHAFT_LINK_MASS:.6f}"
    # diaginertia for merged (20mm/10g) segments scaled up ~6x from the single-link
    # placeholder ("1e-6", itself ~15x the true ~6.7e-8 kg*m^2 for a 10mm/5g capsule,
    # apparently used as a stability buffer). Doubling both mass and length roughly
    # scales true transverse inertia by ~6x (m*(3r^2+L^2)/12); keeping the OLD tiny
    # value on a heavier, longer body left it reacting to contact torque like a
    # point mass -- caused a multi-metre "explosion" under roller drive, confirmed
    # by testing. Standalone (unmerged, still 10mm/5g) segments keep the original.
    _uni_diaginertia = "6e-6 6e-6 6e-6"
    _uni_mass = "0.000500"   # negligible dummy-body mass, just needs to be nonzero

    def _sind() -> str:
        return "      " + "  " * _depth

    _depth = 0
    w(f'{_sind()}<body name="shaft_link_0" pos="{_tx:.6f} {_ty:.6f} {_tz:.6f}" quat="{_target_quat}">')
    w(f'{_sind()}  <joint name="shaft_root_free" type="free"/>')
    w(f'{_sind()}  <inertial pos="0 {_sh} 0" mass="{_sm}" diaginertia="1e-6 1e-6 1e-6"/>')
    w(f'{_sind()}  <geom type="capsule" fromto="0 0 0 0 {_sp} 0" size="{_sr}" material="base_mat" contype="3" conaffinity="3" friction="0.08 0.005 0.001"/>')
    _depth += 1

    _real_seg_names: list[int] = [0]
    _prev_seg_len = _sp   # shaft_link_0 is always a plain 10mm segment

    for u in range(1, SHAFT_N_LINKS, 2):
        _has_partner = (u + 1) < SHAFT_N_LINKS
        _seg_len  = _sp2 if _has_partner else _sp
        _seg_half = _sh2 if _has_partner else _sh
        _seg_mass = _sm2 if _has_partner else _sm
        _seg_diag = _uni_diaginertia if _has_partner else "1e-6 1e-6 1e-6"

        # Universal joint junction: zero-length dummy body hosts the X hinge;
        # shaft_link_u sits at the SAME point (zero further offset) and hosts
        # the Z hinge -- together an omnidirectional bend with no twist DOF.
        # Offset by the PREVIOUS segment's actual length (10mm or 20mm,
        # merged segments aren't uniform) -- not a hardcoded SPACING, or this
        # dummy lands mid-way through the previous capsule instead of at its
        # true end, stacking every segment after it on top of the one before.
        w(f'{_sind()}<body name="shaft_uni_{u}" pos="0 {_prev_seg_len} 0">')
        w(f'{_sind()}  <joint name="shaft_uni_{u}_x" type="hinge" axis="1 0 0" '
          f'range="-{SHAFT_JOINT_RANGE:.6f} {SHAFT_JOINT_RANGE:.6f}" '
          f'stiffness="{SHAFT_STIFFNESS}" damping="{SHAFT_DAMPING}"/>')
        w(f'{_sind()}  <inertial pos="0 0 0" mass="{_uni_mass}" diaginertia="1e-8 1e-8 1e-8"/>')
        _depth += 1
        w(f'{_sind()}<body name="shaft_link_{u}" pos="0 0 0">')
        w(f'{_sind()}  <joint name="shaft_uni_{u}_z" type="hinge" axis="0 0 1" '
          f'range="-{SHAFT_JOINT_RANGE:.6f} {SHAFT_JOINT_RANGE:.6f}" '
          f'stiffness="{SHAFT_STIFFNESS}" damping="{SHAFT_DAMPING}"/>')
        w(f'{_sind()}  <inertial pos="0 {_seg_half} 0" mass="{_seg_mass}" diaginertia="{_seg_diag}"/>')
        w(f'{_sind()}  <geom type="capsule" fromto="0 0 0 0 {_seg_len} 0" size="{_sr}" material="base_mat" contype="3" conaffinity="3" friction="0.08 0.005 0.001"/>')
        _depth += 1
        _real_seg_names.append(u)
        _prev_seg_len = _seg_len

    _shaft_chain_depth = _depth   # total <body> opens in the shaft chain, for closing later

    # Bending tip: attached to the distal end of shaft_link_{N-1}.
    # disk_0 is written manually with pos="0 SPACING 0" (at the distal tip of
    # the last shaft link). disk_1..25 use the standard add_disk() helper.
    _dbi = _sind()    # disk base indent continues from wherever the shaft chain ended

    w(f'{_dbi}<body name="disk_0" pos="0 {_sp} 0">')
    w(f'{_dbi}  <joint name="shaft_tip_ball" type="ball" stiffness="{SHAFT_TIP_BALL_STIFFNESS}" damping="0.3"/>')
    _d0_tmp: list[str] = []
    scope_gen.add_disk(_d0_tmp, 0, _dbi)
    for _line in _d0_tmp[1:]:    # skip the <body ...> opening line (already written)
        lines.append(_line)

    for disk_idx in range(1, scope_gen.N_DISKS):
        scope_gen.add_disk(lines, disk_idx, _dbi + "  " * disk_idx)
        if disk_idx == scope_gen.N_DISKS - 1:
            tip_indent = _dbi + "  " * disk_idx + "  "
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

    # Close disk bodies (disk_25 down to disk_0)
    for disk_idx in range(scope_gen.N_DISKS - 1, -1, -1):
        w(f'{_dbi + "  " * disk_idx}</body>')

    # Close shaft chain bodies (shaft_link_59 / shaft_uni_59 ... down to shaft_link_0)
    for _d in range(_shaft_chain_depth - 1, -1, -1):
        w(f'{"      " + "  " * _d}</body>')

    w('  </worldbody>')
    w('')

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
    # Roller feed actuators, added after the 4 tendon-pull actuators (indices
    # 0-3): ring 0 lands at ctrl indices 4-7, ring 1 at indices 8-11.
    for _ring in range(2):
        for _gi in range(4):
            _rname = f"roller_r{_ring}_{_gi}"
            w(
                f'    <position name="{_rname}_act" joint="{_rname}_spin" '
                f'kp="{ROLLER_ACTUATOR_KP}" kv="{ROLLER_ACTUATOR_KV}" forcelimited="true" '
                f'forcerange="-{ROLLER_ACTUATOR_FORCE_LIMIT} {ROLLER_ACTUATOR_FORCE_LIMIT}"/>'
            )
    w('  </actuator>')
    w('')
    w('  <contact>')
    # Shaft segment pairs: consecutive real (geom-bearing) shaft segments and
    # the shaft-to-disk junction need explicit exclusion (shaft_link_0's free
    # joint means it's no longer auto-excluded from anything via a
    # parent-mocap hierarchy; the shaft_uni_* dummy bodies between segments
    # carry no geom, so they can never contact anything and need no excludes
    # of their own). _real_seg_names lists the merged/standalone segment
    # names in order, e.g. [0, 1, 3, 5, ..., 59].
    for _i in range(len(_real_seg_names) - 1):
        w(f'    <exclude body1="shaft_link_{_real_seg_names[_i]}" body2="shaft_link_{_real_seg_names[_i + 1]}"/>')
    w(f'    <exclude body1="shaft_link_{_real_seg_names[-1]}" body2="disk_0"/>')
    for disk_idx in range(scope_gen.N_DISKS - 1):
        w(f'    <exclude body1="disk_{disk_idx}" body2="disk_{disk_idx + 1}"/>')
    # Rollers must only ever touch the shaft, never each other -- at 90-degree
    # spacing their curved rims overlap near the shaft axis (confirmed via
    # contact-point inspection in the prototype: ~1.7mm interpenetration on
    # every adjacent pair, which swamped the actual grip/drive signal). Applies
    # within each ring, and across the two rings for safety even though the
    # 20mm spacing should leave an 8mm gap.
    _all_rollers = [f"roller_r{_ring}_{_gi}" for _ring in range(2) for _gi in range(4)]
    for _i in range(len(_all_rollers)):
        for _j in range(_i + 1, len(_all_rollers)):
            w(f'    <exclude body1="{_all_rollers[_i]}" body2="{_all_rollers[_j]}"/>')
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
