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
SHAFT_N_LINKS        = 44       # 2026-07-13 (second change, AFTER the 76->50 revert below): SHORTENED
                                 # 50 -> 44. This is a LENGTH change, not a density change -- spacing
                                 # stays 10mm, so effective stiffness/damping are untouched and
                                 # SHAFT_STIFFNESS=0.05 stays valid. Keep N EVEN: the body loop
                                 # (range(1, N, 2)) only emits the 2.5mm end stub when the last index
                                 # has no partner, and disk_0 is hard-placed at pos="0 {stub_len} 0" --
                                 # an odd N puts disk_0 *inside* the final 20mm capsule.
                                 # WHY: the scope was LONGER than the colon. 50 links built a 492.5mm
                                 # shaft (10mm link_0 + 24 fused 20mm pairs + 2.5mm stub) which, plus
                                 # the 62.5mm flexible tip, is 555mm of scope in a 500mm colon. Since
                                 # `shaft_exhausted` fires when the shaft's BASE reaches the roller ring
                                 # (20mm outside the entrance), the episode ran until the TIP was at
                                 # ~535mm -- about 35mm OUT the far end of the colon, in free space with
                                 # no wall contact, for the last ~117 steps. Worse, new_territory reward
                                 # is not capped at s_goal, so the policy was being PAID to drive the tip
                                 # out of the colon into open air where no tip penalty could reach it.
                                 # 44 links -> 432.5mm shaft, 495mm scope in a 500mm colon; at
                                 # termination the tip sits ~475mm, i.e. ~25mm INSIDE the colon, still in
                                 # contact. Measured, not derived (seed 0).
                                 # COST, accepted deliberately: progress headroom -- how far disk_0
                                 # overshoots s_goal (= colon_len - END_CLEARANCE(100mm)) at termination
                                 # -- falls from 72.5mm to 12.5mm. That headroom is the buffer absorbing
                                 # shaft BUCKLING (a buckled shaft consumes more length than the
                                 # centreline arc, so disk_0's arc-s falls short of ring+shaft_length).
                                 # With 12.5mm, heavy buckling could cap completing episodes at ~95-99%
                                 # progress rather than 100%. If that bites, the fix is END_CLEARANCE
                                 # 0.10 -> 0.13 (scope_colon_env.py), NOT a further shortening: 42 links
                                 # puts headroom at -7.5mm, making 100% progress outright unreachable.
                                 # NOT a throughput change -- do not repeat this expecting fps. Measured
                                 # 50->44 gains only ~6% (1027 -> 1088 mj_step/s) and contact count is
                                 # FLAT at 52.0, because the removed links are the free-floating tail
                                 # BEHIND the rollers, which touches nothing. Physics cost here is
                                 # dominated by the colon-wall contact solve, not shaft body count. (The
                                 # 76->50 revert below was different: it thinned links INSIDE the colon,
                                 # cutting contacts 60 -> 52, and that is where its 1.58x came from.)
                                 # NOTE this REDEFINES the completion metric: `shaft_exhausted` now fires
                                 # ~60mm of shaft feed (~200 steps at max advance) earlier, so completion
                                 # % rises for reasons unrelated to policy skill. v8_p1_v1's 25.7%
                                 # completion is NOT comparable to anything measured after this change.
                                 #
                                 # --- prior change, same day: 76 -> 50 (density revert) ---
                                 # 2026-07-13: REVERTED 76 -> 50 (38 -> 25 bend-joint pairs), back to
                                 # the original density. Two reasons, both pointing the same way:
                                 #   1. Throughput: 1.5x joints cost ~40% of rig fps (~500-550 ->
                                 #      ~300-350), i.e. ~1.5x the wall-clock for the same step budget.
                                 #   2. The 76-link shaft buckled hard at the entrance (live-viewer
                                 #      observation) -- too flexible in practice.
                                 # Note this is a LITERAL revert: SHAFT_STIFFNESS/SHAFT_DAMPING are
                                 # deliberately left where they are, NOT density-compensated back.
                                 # Effective beam stiffness and damping both scale as
                                 # (per-joint value x segment length), so restoring the 20mm fused-pair
                                 # segment raises both by 20/13.16 ~= 1.52x versus the 76-link model.
                                 # That stiffening is the POINT here (see reason 2), not a side effect
                                 # to be corrected. It also lands SHAFT_STIFFNESS=0.05 on the 50-link
                                 # geometry it was originally specified against (see below).
                                 # Total shaft mass returns to 50 x 5g = 250g (the 76-link model was
                                 # accidentally 380g -- SHAFT_LINK_MASS was never density-compensated;
                                 # 250g is also the more plausible figure for 500mm of 9mm-OD tube).
                                 # Prior value (76, 2026-07-12): finer joint resolution, to test whether
                                 # the shaft would form a smoother bend curve instead of concentrating
                                 # load at sparse joints. Trained 10M->15M steps as part of a 5-change
                                 # bundle (with the tendon integral control, stiffness 0.05, stub shrink,
                                 # tip stiffness) that took mean progress 37% -> 60%; density's own
                                 # contribution within that bundle was never isolated.
                                 # NOTE for anyone raising this again: 100 links (2x, 50 pairs) crashes
                                 # MuJoCo's XML compiler on load (no Python traceback, hard process
                                 # failure) -- <body> nesting-depth blowing the 1MB thread stack
                                 # (~127 levels at 2x vs ~101 at 1.5x). At 50 links we are back to
                                 # ~77 levels, well clear of it.
SHAFT_LINK_SPACING_M = 0.010    # 10 mm centre-to-centre. Was `0.5 / SHAFT_N_LINKS`, which pinned the
                                 # nominal shaft to 500mm and silently RE-DERIVED the segment length
                                 # from the link count -- so shortening the shaft would have thinned the
                                 # segments and softened the shaft as a side effect. Now stated
                                 # directly: segment length is a physics constant (it sets effective
                                 # stiffness/damping alongside SHAFT_STIFFNESS/SHAFT_DAMPING), and shaft
                                 # LENGTH is SHAFT_N_LINKS' job alone. The two are now independent.
                                 # Built shaft length = SHAFT_LINK_SPACING_M * (N - 1) + 2.5mm stub.
SHAFT_LINK_RADIUS_M  = 0.0045   # 4.5 mm radius → 9 mm OD, matching bending tip
SHAFT_JOINT_RANGE    = 0.349    # ±20° per link (radians)
SHAFT_STIFFNESS      = 0.05     # Per-joint. NOTE (2026-07-13): the density-compensation arithmetic in
                                 # the history below is now HISTORICAL -- SHAFT_N_LINKS came back off 76
                                 # to 50 and then to 44 (see above), so
                                 # this 0.05 again sits on the 20mm fused-pair segment it was originally
                                 # specified against. Effective beam stiffness is therefore 1.52x the
                                 # 76-link model's, which is intended (that shaft buckled at the
                                 # entrance). Do not "restore" 0.0329 unless N_LINKS goes back to 76.
                                 # 2026-07-12: lowered again from 0.076 -- still visibly not soft
                                 # enough relative to the tip. User's direct target this round.
                                 # Prior value (0.076): live-viewer observation showed the
                                 # tip buckling much more easily than the shaft, backwards from real
                                 # endoscope behaviour (shaft should be the part that yields/buckles;
                                 # the tip should stay comparatively rigid). User's target: 0.125 (the
                                 # flat value at the ORIGINAL 50-link/10mm-spacing model) -> 0.05, a
                                 # 0.4x reduction. Applied as the same 0.4x to this model's
                                 # density-compensated 0.19 (see note below) -> 0.076, so the just-
                                 # requested softening and the earlier density-compensation scaling
                                 # both hold rather than one silently overriding the other. Untested.
                                 # Prior value (0.19): scaled up from 0.125 -- effective bending stiffness
                                 # scales roughly as (per-joint stiffness x segment length), so
                                 # shrinking segment length ~34% (25->38 pairs) while leaving this
                                 # unchanged would have silently softened the shaft further, stacking
                                 # an unintended change on top of the one already validated below.
                                 # 0.125 * (20mm / 13.16mm fused-pair length) ~= 0.19 holds effective
                                 # stiffness constant at the just-confirmed value, isolating joint
                                 # density as its own variable.
                                 # Prior value (0.125): halved again from 0.25 -- live-viewer observation on
                                 # the v8_p1_v1 best-yet checkpoint (37% mean, reward 0.440) showed
                                 # the shaft not buckling at all at tight bends (segments still too
                                 # short/stiff to yield), forcing all the bend load onto the much
                                 # more compliant flexible tip instead -- visibly over-deforming it.
                                 # Untested at this new value; revert to 0.25 if the shaft becomes
                                 # too floppy (loses pushability / column strength) instead.
                                 # History: 2.0 -> 1.0 (2026-07-02, real RL-policy viewer logs
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
SHAFT_DAMPING        = 0.05     # angular damping per shaft hinge joint.
                                 # 2026-08-02: 0.3 -> 0.05. This is a TIMESCALE fix, not a
                                 # compliance one -- SHAFT_STIFFNESS is untouched, so the
                                 # shaft's static shape under load is unchanged.
                                 # The problem: a full traverse takes ~10.2 s of sim time
                                 # (305 mm of task travel at the 30 mm/s full-command advance),
                                 # but the shaft's own relaxation time was tau = c/k = 0.3/0.05
                                 # = 6.0 s. Ratio 1.7, meaning the shaft NEVER reached
                                 # equilibrium during an episode -- it was dragged through the
                                 # colon permanently lagging and stressed. A real colonoscope
                                 # advancing at ~3-5 mm/s covers the same 305 mm in ~61 s, a
                                 # ratio of ~10, i.e. near-quasi-static at every instant.
                                 # Why fix it here instead of slowing the advance: every hinge
                                 # in this model is massively overdamped (shaft zeta ~ 2592, tip
                                 # zeta ~ 24), so the dynamics are viscous-elastic, not inertial,
                                 # and depend only on the RATIO of driving rate to relaxation
                                 # rate -- not on absolute wall-clock time. The ratio can
                                 # therefore be corrected from either end. c = 0.05 gives
                                 # tau = 1.0 s and a ratio of 10.2, matching real colonoscopy,
                                 # at zero cost. Slowing the advance 3x instead would cost 3x
                                 # the compute AND 3x fewer episodes per fixed Sample Factory
                                 # step budget, losing colon-seed diversity.
                                 # Advance rate is deliberately UNCHANGED (2026-08-02 decision).
SHAFT_LINK_MASS      = 0.005    # 5 g per link (44 links -> ~0.22 kg total shaft mass)
SHAFT_TIP_BALL_STIFFNESS = 0.125 # 2026-07-12: halved in lockstep with SHAFT_STIFFNESS above, same
                                 # reasoning (tip over-deforming at tight bends) -- still not swept
                                 # in isolation. stiffness of the single ball joint at the shaft/bending-tip
                                 # junction (shaft_tip_ball below). Was hardcoded at 2.0 and never
                                 # swept before; moved in lockstep with SHAFT_STIFFNESS's 2.0->1.0->
                                 # 0.5->0.25 history (2026-07-02) on the theory that it's the same
                                 # active-drive-slip mechanism -- still untested in isolation.
SHAFT_TIP_BALL_DAMPING = 0.125   # 2026-08-02. This was previously HARDCODED as damping="0.3" in the
                                 # shaft_tip_ball emission below, so it silently did NOT track
                                 # SHAFT_DAMPING despite looking like it should -- the 0.3 was simply
                                 # a copy of the old SHAFT_DAMPING value. Caught when SHAFT_DAMPING
                                 # went 0.3 -> 0.05 and this joint was left behind at tau = c/k =
                                 # 0.3/0.125 = 2.4 s, which would have made this single junction the
                                 # slowest element in the whole scope and dominated the very
                                 # traverse/relaxation ratio that change exists to fix.
                                 # 0.125 gives tau = 1.0 s, matching the shaft exactly. Named rather
                                 # than inlined so it cannot drift out of lockstep again -- same
                                 # lockstep reasoning as SHAFT_TIP_BALL_STIFFNESS above.
TIP_CAM_FOVY = 67.7              # 2026-08-04: 85 -> 67.7. MuJoCo's fovy is the VERTICAL field of
                                 # view, and a square 64x64 render forced horizontal == vertical,
                                 # so the policy trained on 85x85 -- matching neither the depth
                                 # model nor any real lens.
                                 # The real scope was measured at 140 degrees DIAGONAL on a 16:9
                                 # frame. A 140-degree lens is necessarily barrel-distorted (a
                                 # rectilinear lens that wide does not exist at this scale), so
                                 # the equidistant mapping r = f*theta applies, NOT r = f*tan(theta):
                                 #   equidistant: H = 122.0 deg, V = 68.6 deg   <- the real lens
                                 #   rectilinear: H = 134.7 deg, V = 106.8 deg  <- wrong model
                                 # 67.7 is chosen over the measured 68.6 (a 0.9 deg difference)
                                 # because it makes this camera EXACTLY reproduce the DA3 depth
                                 # model's training camera: Blender rendered the fine-tune dataset
                                 # at 100 deg HORIZONTAL rectilinear on 16:9
                                 # (perception/.../blender_depth_dataset.py, --fov_deg 100.0),
                                 # which is V = 67.67. MuJoCo is rectilinear too, so fovy=67.7 on a
                                 # 16:9 render gives back H = 100.0 deg. Sim renders and DA3
                                 # training renders are then the same camera, and the vertical
                                 # field also matches the real scope to within a degree.
                                 # RESIDUAL, deliberately not fixed here: horizontally the real
                                 # frame covers 122 deg where sim covers 100 deg -- that gap IS the
                                 # lens's barrel distortion. It is a PERCEPTION-side fix (undistort
                                 # the live frame to 100 deg H before DA3, which needs a fisheye
                                 # calibration for the coefficients), not a sim-side one, and it
                                 # does not block training. See docs/CURRENT_PLAN.md section 8.
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
    # 2026-07-12: the last shaft segment (u=SHAFT_N_LINKS-1, standalone/unfused) used to
    # stay a full 10mm rigid capsule between the shaft's last flex joint (shaft_uni_u) and
    # the tip's first joint (shaft_tip_ball on disk_0) -- live-viewer debugging found this
    # acted as a stiff moment arm right at the shaft/tip boundary, dumping bend load onto
    # the much more finely-jointed tip instead of the shaft. Shrunk to match the tip's own
    # disk spacing (2.5mm) instead of the shaft's 10mm unit -- can't go to zero (degenerate
    # capsule, colocated joints), but this matches scale on both sides of the junction.
    # Mass scaled proportionally (was flat SHAFT_LINK_MASS per 10mm unit).
    _stub_len_m = scope_gen.DISK_SPACING
    _stub_len = f"{_stub_len_m:.6f}"
    _stub_half = f"{_stub_len_m / 2.0:.6f}"
    _stub_mass = f"{SHAFT_LINK_MASS * (_stub_len_m / SHAFT_LINK_SPACING_M):.6f}"

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
        _seg_len  = _sp2 if _has_partner else _stub_len
        _seg_half = _sh2 if _has_partner else _stub_half
        _seg_mass = _sm2 if _has_partner else _stub_mass
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
    # disk_0 is written manually with pos="0 STUB_LEN 0" (at the distal tip of
    # the last shaft link, now the shrunk 2.5mm stub, not the full 10mm shaft
    # spacing -- see _stub_len above). disk_1..25 use the standard add_disk() helper.
    _dbi = _sind()    # disk base indent continues from wherever the shaft chain ended

    w(f'{_dbi}<body name="disk_0" pos="0 {_stub_len} 0">')
    w(f'{_dbi}  <joint name="shaft_tip_ball" type="ball" stiffness="{SHAFT_TIP_BALL_STIFFNESS}" '
      f'damping="{SHAFT_TIP_BALL_DAMPING}"/>')
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
                f'xyaxes="-1 0 0 0 0 -1" fovy="{TIP_CAM_FOVY}"/>'
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

    # Close shaft chain bodies (the last shaft_link_u / shaft_uni_u ... down to shaft_link_0)
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
        # px/nx bend the X axis (12 joints, 120deg); pz/nz bend Z (13 joints, 130deg).
        _max_pull = scope_gen.MAX_PULL_X if "x" in name else scope_gen.MAX_PULL_Z
        w(
            f'    <general name="pull_{name}" tendon="tendon_{name}" '
            f'gaintype="fixed" biastype="affine" '
            f'ctrllimited="true" '
            f'ctrlrange="-{scope_gen.fmt(_max_pull)} '
            f'{scope_gen.fmt(_max_pull)}" '
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
    # Tendon integral actuators (2026-07-12), appended AFTER the roller actuators
    # (indices 12-15) so ctrl[0:4] (tendon P+D) and ROLLER_CTRL_SLICE=slice(4,12)
    # in scope_colon_env.py stay exactly as they are -- these are new, additive
    # actuators on the SAME 4 tendons, not a replacement for pull_{name} above.
    # See ACTUATOR_KI note in generate_videoscope_one_section.py for why this
    # exists and how it's meant to be driven (scope_colon_env.py's step()).
    for name, _angle, _rgba in scope_gen.TENDONS:
        w(
            f'    <general name="pull_{name}_i" tendon="tendon_{name}" '
            f'dyntype="integrator" gaintype="fixed" biastype="none" '
            f'ctrllimited="false" '
            f'forcelimited="true" '
            f'forcerange="-{scope_gen.fmt(scope_gen.ACTUATOR_PULL_FORCE_LIMIT)} 0" '
            f'gainprm="-{scope_gen.fmt(scope_gen.ACTUATOR_KI)} 0 0"/>'
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
