"""
Gymnasium environment for the scope-in-colon scene.

  - action_space      = Box(-1, 1, shape=(3,))
                          a[0]  -> tendon px/nx setpoint
                          a[1]  -> tendon pz/nz setpoint
                          a[2]  -> entrance roller feed rate (signed). Drives a
                                   target angle for the position-controlled
                                   entrance rollers (see build_collision_scene.py);
                                   the roller can slip under resistance, so
                                   progress is measured from disk_0's true
                                   physical position, not the commanded angle.
  - observation_space = Dict{
        depth: Box(0,1, (1,H,W))   metric depth from tip_cam, normalised
        state: Box(-inf,inf, (8,))  [cmd_x_n, cmd_y_n, ten_x_n, ten_z_n,
                                     last_action[0..2], tip_contact]
    }
    (v8_p1, V8 plan Layer 1: force_norm and net_fx/net_fz -- both privileged
    sim-only signals with no real-hardware equivalent -- replaced by
    ten_x_n/ten_z_n, normalised cable-pair extension read from
    data.ten_length, matching a real linear encoder differential.)
  - reward            = W_PROGRESS * new_territory * force multiplier
                      - W_TIP_PENALTY while the tip disk contacts the wall
  - terminated        = shaft exhausted (physical base reached the entrance
                        rollers, no more shaft to feed) | ejected
  - truncated         = stuck | tip_stuck | step >= max_steps

  Reward stays based on the TIP's real physical progress (actual_base_s,
  proxied by disk_0) regardless of how the episode ends -- shaft exhaustion
  is a natural physical boundary (2026-07-02, replaces the old distance-
  target "goal" concept), not a success condition. A shaft-exhausted episode
  can still show low progress if a lot of it was consumed by slip.

Usage:
    env = ScopeColonEnv(seed=0)
    obs, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 0.5]))
"""
from __future__ import annotations

import os
import sys
import tracemalloc
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Optional memory-leak diagnostic — zero overhead unless SF_MEM_PROFILE=1
# ---------------------------------------------------------------------------
_MEM_PROFILE = os.environ.get("SF_MEM_PROFILE", "0") == "1"
_RENDERER_RECYCLE_EVERY = 50  # episodes between renderer close+recreate (releases EGL mmap regions)


def _rss_mb() -> float:
    """Read process RSS from /proc/self/status (Linux only). Returns 0 elsewhere."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
from gymnasium import spaces  # noqa: E402

import generate_videoscope_one_section as scope_gen  # noqa: E402
from build_collision_scene import (  # noqa: E402
    arc_length,
    build_scene,
    ROLLER_RADIUS_M,
    ROLLER_RING_SPACING_M,
    SCOPE_TIP_AT_S,
)

# Roller feed actuator ctrl indices: 4 tendon-pull actuators (0-3), then 2
# roller rings x 4 rollers each (4-11), all position-controlled and driven
# together at the same target angle. See build_collision_scene.py.
ROLLER_CTRL_SLICE = slice(4, 12)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_STEPS = 10000
DEFAULT_ADVANCE_RATE_MAX = 0.0003      # 0.3 mm per env step (~30 mm/s; was 1 mm/step = 100 mm/s)
DEFAULT_PHYSICS_PER_STEP = 20          # MuJoCo timestep is 0.5 ms; 20 -> 10 ms per env step
DEFAULT_TIP_S_INIT = SCOPE_TIP_AT_S    # Must match build_collision_scene.py's SCOPE_TIP_AT_S -- that's
                                        # what the XML is actually built with (shaft_link_0's free-joint
                                        # reference pose), so this env's own bookkeeping (base_s_init,
                                        # actual_base_s) has to agree or actual_base_s jumps the instant
                                        # it's re-projected from disk_0's true (XML-driven) position.
                                        # Also >=~72.5mm is required for the shaft to reach the entrance
                                        # rollers at all -- see SCOPE_TIP_AT_S's own comment.
END_CLEARANCE = 0.10                   # stop base 10 cm before physical centreline end; with the 20 cm visual-only tail this is 30 cm before the rendered distal end.
                                        # s_goal (below) is kept only as a fixed reference distance for the
                                        # "progress" reporting percentage -- it is NOT a termination target
                                        # (2026-07-02: removed the old distance-target "goal" termination in
                                        # favour of shaft exhaustion, see step()'s termination_reason logic).
PD_KP = 0.06                           # proportional gain: ~1.44mm max first step, matches v3 rate
PD_KD = 0.30                           # derivative gain: strongly over-damped, ±1.1mm steady-state jitter
ANTIWINDUP_FORCE_MARGIN = 2.0          # N below ACTUATOR_PULL_FORCE_LIMIT before freezing each tendon's
                                        # integral actuator (2026-07-12, see build_collision_scene.py's
                                        # pull_{name}_i actuators / generate_videoscope_one_section.py's
                                        # ACTUATOR_KI note). Verified via standalone diagnostic (single
                                        # tendon and all 4 together, real antagonistic ctrl mapping).
DEFAULT_DEPTH_RES = 64                 # tip-cam depth obs resolution (HxW)
DEFAULT_DEPTH_CLIP_M = 0.5             # depth values clipped & normalised by this many metres
TIP_CAM_NAME = "tip_cam"
# State obs: [cmd_x_n, cmd_y_n, ten_x_n, ten_z_n, last_action[0..2], tip_contact] = 8D
# Removed from v17: prev_action[3] (redundant with RNN) and hinge_angles[25] (67% of
# obs vector, mostly reflects colon geometry not policy choices; depth image covers this).
# tip_contact added: explicit 0/1 flag for tip disk touching colon wall (UFO posture detection).
# Removed 2026-07-01: progress placeholder slot (was always 0.0, never carried real
# colon-progress info -- kept previously only for checkpoint/architecture compatibility
# with older runs, which is no longer needed for the v7_shaft training line).
# v8_p1 (V8 plan Layer 1, 2026-07-10): force_norm and net_fx/net_fz removed,
# ten_x_n/ten_z_n added. Both removed fields are privileged sim-only signals:
# force_norm sums raw MuJoCo contact force (no real contact-force sensor on
# the rig), and net_fx/net_fz derive from data.actuator_force, which in sim
# is a clean readout of commanded-tendon resistance but on real hardware
# would be motor current -- dominated by unmodelled Bowden/sheath/capstan
# friction (only actuator gain is domain-randomised, not friction). Both
# stay fully wired into reward shaping and info/ diagnostics -- only removed
# from what the policy observes. ten_x_n/ten_z_n are the antagonistic
# cable-pair length differential from data.ten_length, matching a real linear
# encoder reading.
STATE_OBS_DIM = 8

# ten_x_n/ten_z_n normalisation: the antagonistic tendon-pair length
# differential (data.ten_length[nx]-ten_length[px], etc.) at full joint-limit
# bend, measured directly via mj_forward with all joints on one axis driven
# to their hinge range limit (scratchpad/measure_max_pull.py, 2026-07-10).
# Not simply 2*MAX_PULL_{X,Z}: the pulled tendon's shortening and the slack
# tendon's lengthening are close but not exactly equal (kinematic asymmetry
# of the hole geometry under bend), so the true differential is measured
# directly rather than derived, giving an exact +-1 at the physical limit.
TEN_X_RANGE_M = 0.013380   # nx-px differential at the 120deg X limit
TEN_Z_RANGE_M = 0.014495   # nz-pz differential at the 130deg Z limit

# ----- reward shaping ------------------------------------------------------
W_PROGRESS = 4.0          # multiplied by metres of *new-territory* advance. Was 2.0 (the
                          # value this reward structure was designed and validated under
                          # for v6_dr, which had no accumulating per-step wall-contact
                          # penalty on this scale). Doubled 2026-07-02 for v7_shaft_v2:
                          # live training logs showed reward improving while mean progress
                          # and shaft_exhausted rate both DECLINED and episodes got shorter
                          # (901k->1212k steps) -- the signature of a policy learning to
                          # truncate early via `stuck` to cap accumulating r_tip cost,
                          # since W_TIP_PENALTY(0.001/step) is unbounded by distance while
                          # r_progress is capped by the fixed travel budget. Total
                          # r_progress per episode was landing ~0.3-0.4 against tip-contact
                          # penalty routinely reaching -0.6 to -1.0 over now-long
                          # (1600-2300 step) episodes -- forward progress was structurally
                          # outmatched by the time-accumulating penalty.
                          # Doubled again 2026-07-10 for v8_p1_v1 (4.0->8.0): the predicted
                          # recurrence happened. At ~300k steps, ~640-step episodes, mean
                          # progress ~16.5%, reward had settled at ~-0.18/episode -- worse
                          # than the ~-0.05 a policy gets by doing nothing at all (idles to
                          # a `stuck` truncation at exactly STUCK_WINDOW=300 steps, earning
                          # 0 progress + 0 tip contact - the same flat W_STUCK_PENALTY=0.05
                          # terminal cost every episode currently pays regardless of
                          # strategy, since 100% of episodes truncate via stuck/tip_stuck
                          # -- no episode has reached shaft_exhausted yet to skip it).
                          # Confirmed via the actual reward formula, not just log-watching:
                          # per-step return for the active policy (~-0.00028/step) was
                          # already below the idle baseline (~-0.00017/step). This is the
                          # SECOND time this project has hit this exact failure mode --
                          # if it recurs a third time, stop re-doubling this and implement
                          # the capping/decoupling fix instead (see above).
                          # Reverted 8.0->4.0 (2026-07-11, v8_p1_v2): this doubling was
                          # calibrated while W_FORCE_MULT's force-contact discount was still
                          # taxing r_progress (see W_FORCE_MULT's own comment below). Zeroing
                          # that discount already restores a chunk of the progress reward this
                          # doubling existed to compensate for -- stacking both changes risked
                          # overshooting r_progress relative to W_TIP_PENALTY the other way.
                          # The original 2.0->4.0 doubling (tip-penalty-vs-capped-progress-
                          # budget reasoning above) is independent of the force discount and
                          # still holds, so only the second increment was undone.
W_SHAPING  = 0.0          # dense shaping: any forward step earns reward, not just new territory.
                          # Additive on top of W_PROGRESS * new_territory so first-pass advance
                          # earns (W_PROGRESS + W_SHAPING) * ds; re-advance after reversal earns
                          # W_SHAPING * ds. Eliminates zero-gradient on re-traversed ground.
W_FORCE = 0.0             # per-step force penalty — disabled (accumulated too fast).
W_LUMEN_BOOST = 0.0       # lumen bonus multiplied INTO the progress reward:
                          #   r_progress = W_PROGRESS * new_territory
                          #              * (1 - force_norm)
                          #              * (1 + W_LUMEN_BOOST * lumen_score)
                          # Earns nothing when not advancing (new_territory=0),
                          # so sitting and staring at the lumen cannot be farmed.
                          # Perfect lumen view gives +50% on every metre advanced.
W_STUCK_PENALTY = 0.05    # terminal penalty for no-progress endings. This is
                          # not a finish bonus; it teaches that stopping for a
                          # long window is bad anywhere in the colon.

# ----- insertion-force diagnostics ----------------------------------------
# force_norm = sum(||mj_contactForce_i||) / MAX_INSERTION_FORCE, clipped [0,1].
# Force no longer terminates the episode; it reduces progress reward through
# the force multiplier, and force_steps remains in info for diagnostics.
# Calibrated from _calibrate_force.py smoke test (seed 0):
#   straight advance (normal):  peak ~0.17 N  -> force_norm ~0.006
#   max curl + advance (cheat): peak ~215 N   -> force_norm >> 1 (immediate fail)
# 30 N puts light wall grazing at ~0.2, hard jam at ~0.5+, curl cheat at >> 1.
MAX_INSERTION_FORCE = 30.0
FORCE_LIMIT_STEPS  = 10   # diagnostic legacy threshold, not an episode terminator
ADVANCE_BLOCK_THRESHOLD = 0.3  # force_norm above which forward advance is hard-blocked
                           # transient wall contact during navigation is fine;
                           # sustained crushing force (curl cheat) terminates.

# ----- early-termination knobs --------------------------------------------
STUCK_WINDOW     = 300    # rolling steps over which Δmax_actual_base_s is summed
STUCK_THRESHOLD  = 0.002  # m — no new territory in STUCK_WINDOW steps → truncate
TIP_STUCK_WINDOW = 150    # consecutive steps tip disk contacts colon wall → truncate (UFO check)
                          # 2026-07-13: raised 50 -> 150. At ADVANCE_RATE_MAX=0.3mm/step, 50 steps
                          # capped in-contact travel at 15mm -- shorter than the in-contact arc of a
                          # 32-48mm-radius bend, so a sharp bend had NO surviving strategy: push
                          # through and it truncated mid-bend, hold still and it truncated anyway.
                          # The manoeuvre was simply absent from the MDP, which is the leading
                          # explanation for v8_p1_v1's 60% plateau (73.3% of episodes ended here,
                          # 0% on stuck/ejected/max_steps) and for the bimodal seed split in the
                          # v7_shaft_v4 eval (gentle bends clear inside 50 contact-steps; sharp
                          # ones cannot, at any skill level). 150 buys ~45mm of in-contact travel,
                          # enough for a typical bend arc, while staying 2x tighter than STUCK_WINDOW.
                          # NOTE this is a proxy check: tip-in-contact is NOT the same as tip-not-
                          # progressing. The genuine UFO posture is caught independently by `stuck`
                          # (300 steps / 2mm of new territory), and progress is measured at disk_0
                          # (the shaft/tip junction), so a jammed tip cannot farm reward by buckling
                          # shaft in behind it. This truncation is therefore a fast backstop, not
                          # the load-bearing anti-UFO guard.
W_TIP_PENALTY    = 0.001  # per-step reward penalty while tip disk is in wall contact.
                          # KEPT DELIBERATELY (2026-07-13) alongside the window increase above: the
                          # easier colons are still learnable under it, so it remains the graded
                          # signal teaching the tip not to park on the wall. Be aware of the balance
                          # it sets, though: r_progress at full advance on new territory is
                          # W_PROGRESS * ADVANCE_RATE_MAX = 4.0 * 0.0003 = +0.0012/step, so advancing
                          # WHILE the tip touches the wall nets only +0.0002/step, and any bend-
                          # clearing slower than 0.25mm/step (83% of max) is net-NEGATIVE. If the
                          # window increase alone doesn't lift progress, this coupling is the next
                          # thing to fix -- gate it on non-advancement
                          # (r_tip = -W_TIP_PENALTY if _tip_in_contact and ds_actual <= 0) rather
                          # than re-raising W_PROGRESS, which W_PROGRESS's own comment warns against.
W_FORCE_MULT     = 0.0    # weight of the whole-shaft force_norm discount on progress/shaping
                          # reward (was a flat 0.3-1.0 multiplier, i.e. W_FORCE_MULT=0.7 with a
                          # 0.3 floor). Zeroed 2026-07-11: per-shaft contact is not just a
                          # pathological "curl and force-advance" cheat signal -- the shaft
                          # physically cannot clear a colon bend without pressing against the
                          # inside wall, so even normal bend-clearing progress was being taxed
                          # (calibration comment above: light grazing alone reads ~0.2, costing
                          # ~14% of every metre of progress). This on top of the separate
                          # tip-only W_TIP_PENALTY may have been reproducing a milder version of
                          # the v11/v12 collapse (additive force penalty -> policy refused to
                          # move), which lines up with v8_p1_v1's stuck-dominant plateau at the
                          # bend-1 progress band. force_norm itself is untouched -- still
                          # computed and reported in info/diagnostics, just no longer taxes
                          # reward. Reversible by setting this back above 0.
EJECTED_MARGIN = 0.05     # m -- terminate "ejected" if scope reverses past base_s_init - this

# ---------------------------------------------------------------------------
# Domain randomisation — sim-to-real transfer
# ---------------------------------------------------------------------------
# All ranges are per-episode: a fresh sample is drawn in reset() each time.
# Ranges are engineering estimates; calibrate against real hardware measurements
# (motor current logs, free-air tip-position mapping) before deploying.
#
# Tendon gains (DR_GAIN_*_RANGE):
#   Tendon efficiency drops as the shaft bends — friction at each bend reduces
#   tip deflection for the same motor command. 1.0 = sim free-air; 0.45 ≈ half
#   efficiency at a tight S-bend. Axes are independent because shaft curvature
#   is asymmetric during insertion.
#   Calibrate: command a known tendon displacement in free air and at maximum
#   insertion depth; the ratio of measured tip angle gives the gain range.
#
# Dead zones (DR_DEAD_*_RANGE):
#   Fraction of max_pull (v8_p1: per-axis, MAX_PULL_X ~6.80mm / MAX_PULL_Z
#   ~7.36mm, was a shared 24mm in v7_shaft) that must be exceeded before the
#   tip moves. Caused by tendon slack that varies with insertion depth and
#   shaft geometry. Keep this small for training: 0.03 -> ~0.2mm of slack at
#   the new (smaller) pull range. Larger values made fine steering vanish and
#   encouraged jittery bang-bang corrections.
#   Calibrate: measure the command threshold below which tip angle is zero at
#   each insertion depth.
#
# Command lag (DR_LAG_RANGE, env steps):
#   Delay between issuing a tendon command and it reaching the tip (mechanical
#   inertia + tendon stretch). Each env step = 10 ms, so lag 2 = 20 ms.
#   Longer shafts and tighter bends increase lag.
#   Calibrate: step-response test — measure time from command to tip movement.
#
# Advance gain (DR_ADV_GAIN_RANGE) -- REMOVED (2026-07-01). This synthesised
# "friction on the shaft body when pushing forward" as an artificial per-episode
# multiplier, because v6's kinematic advance (a mocap dragging the base along
# the centreline) had no real mechanism for insertion resistance to vary at
# all. v7_shaft replaced that with an actual physical shaft + entrance roller
# feeder: insertion speed now varies for real, per-step, from genuine contact
# friction against the colon wall and the roller's own grip/slip dynamics
# (see build_collision_scene.py, ROLLER_* constants). A synthetic multiplier
# on top of already-real variable resistance would double-count the effect
# this DR parameter existed to approximate.

DR_GAIN_X_RANGE   = (0.45, 1.0)   # x-axis tendon efficiency [unitless]
DR_GAIN_Y_RANGE   = (0.45, 1.0)   # y-axis tendon efficiency [unitless]
DR_DEAD_X_RANGE   = (0.0,  0.03)  # x-axis dead zone as fraction of max_pull [unitless]
DR_DEAD_Y_RANGE   = (0.0,  0.03)  # y-axis dead zone as fraction of max_pull [unitless]
DR_LAG_RANGE      = (0,    2)     # command lag in env steps, inclusive [steps]


class ScopeColonEnv(gym.Env):
    """One procedural colon per env instance (seed). RL drives the scope's
    tip target + base advance; the colon is fixed for the lifetime of the
    instance."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int = 0,
        max_steps: int = DEFAULT_MAX_STEPS,
        advance_rate_max: float = DEFAULT_ADVANCE_RATE_MAX,
        physics_per_step: int = DEFAULT_PHYSICS_PER_STEP,
        tip_s_init: float = DEFAULT_TIP_S_INIT,
        depth_res: int = DEFAULT_DEPTH_RES,
        depth_clip_m: float = DEFAULT_DEPTH_CLIP_M,
    ):
        super().__init__()
        self.seed_value = int(seed)
        self.max_steps = int(max_steps)
        self.advance_rate_max = float(advance_rate_max)
        self.physics_per_step = int(physics_per_step)
        self.tip_s_init = float(tip_s_init)
        self.max_pull_x = float(scope_gen.MAX_PULL_X)   # px/nx (env's cmd_x)
        self.max_pull_y = float(scope_gen.MAX_PULL_Z)   # pz/nz (env's cmd_y)
        self.depth_res = int(depth_res)
        self.depth_clip_m = float(depth_clip_m)

        # Generate scene + load MuJoCo model
        self.xml_path, self.centreline = build_scene(self.seed_value, s_tip=self.tip_s_init)
        self.arc_s = arc_length(self.centreline)
        self.total_length = float(self.arc_s[-1])
        self.s_goal = self.total_length - END_CLEARANCE

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        # Pristine tendon actuator coefficients (before any DR gain scaling),
        # ctrl order px/pz/nx/nz -- reset() rescales FROM this baseline each
        # episode so repeated resets don't compound the scale-down.
        self._tendon_gainprm0 = self.model.actuator_gainprm[:4].copy()
        self._tendon_biasprm0 = self.model.actuator_biasprm[:4].copy()

        # ---- tendon integral actuators (2026-07-12) ----
        # Separate dyntype="integrator" actuators (see build_collision_scene.py),
        # one per tendon, appended after the roller actuators. Resolve their ids
        # + the tendon ids (to read ten_length each substep) once here; driven in
        # step()'s physics loop. ctrl order matches scope_gen.TENDONS: px,pz,nx,nz.
        self._tendon_names = [name for name, _a, _r in scope_gen.TENDONS]
        self._tid = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, f"tendon_{name}")
            for name in self._tendon_names
        }
        self._aid_p = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"pull_{name}")
            for name in self._tendon_names
        }
        self._aid_i = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"pull_{name}_i")
            for name in self._tendon_names
        }
        for _name in self._tendon_names:
            if self._tid[_name] < 0 or self._aid_p[_name] < 0 or self._aid_i[_name] < 0:
                raise RuntimeError(f"tendon/actuator ids for '{_name}' not found in compiled model")
        # Pristine integral-actuator gainprm, same DR-gain rescaling as the P+D
        # actuators above (a lossy/friction-heavy cable should attenuate the
        # integral's holding force too, or DR gain would stop meaning anything --
        # the policy could just lean on the integral term regardless of DR).
        self._tendon_i_gainprm0 = {
            name: self.model.actuator_gainprm[self._aid_i[name]].copy()
            for name in self._tendon_names
        }

        # ---- ids: disk_0 (shaft distal end / tip root) ----
        # No more mocap "target" body -- shaft_link_0 has a free joint and is
        # held in place only by the entrance roller grip + colon wall contact
        # (see build_collision_scene.py). Its initial pose is baked into the
        # compiled model's free-joint reference position; mj_resetData()
        # restores it automatically, no manual positioning needed here.
        self.disk0_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "disk_0",
        )
        if self.disk0_body_id < 0:
            raise RuntimeError("disk_0 body not found in compiled model")

        # shaft_link_0 (proximal/base end of the physical 60-link flexible
        # shaft, held by roller grip -- see build_collision_scene.py). Used
        # to detect shaft exhaustion: once this end has been physically fed
        # all the way up to the outer roller ring, there is no more shaft
        # trailing behind for the rollers to grip -- the episode's real
        # physical boundary, replacing the old distance-target "goal".
        self.shaft0_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "shaft_link_0",
        )
        if self.shaft0_body_id < 0:
            raise RuntimeError("shaft_link_0 body not found in compiled model")

        # Entrance tangent for extrapolated-backward projection (s < 0).
        entrance_tangent = self.centreline[1] - self.centreline[0]
        self.entrance_tangent = entrance_tangent / (
            np.linalg.norm(entrance_tangent) + 1e-12
        )
        # Outer roller ring's fixed world position, projected onto the entrance
        # tangent. Ring 1 (not ring 0) because it's mounted EXTERNALLY -- further
        # out along -entrance_tangent, away from the colon (see
        # ROLLER_RING_SPACING_M in build_collision_scene.py) -- so it's the true
        # outer boundary of the grip mechanism: the whole shaft has been fed
        # through once shaft_link_0 reaches it, not just the inner ring.
        _roller_ring1_pos = self.centreline[0] - ROLLER_RING_SPACING_M * self.entrance_tangent
        self._roller_ring1_proj = float(np.dot(_roller_ring1_pos, self.entrance_tangent))
        self.target_insertion_depth = 0.0   # metres "commanded" along entrance_tangent (see step());
                                             # actual physical progress is tracked separately via
                                             # actual_base_s, since the roller can slip under resistance

        # Renderer for tip-cam depth (created in __init__, reused every step).
        # Note: not picklable -- SubprocVecEnv users need lazy init via a
        # subprocess-side wrapper, or re-create after un-pickling.
        self.renderer = mujoco.Renderer(
            self.model, height=self.depth_res, width=self.depth_res,
        )
        self.renderer.enable_depth_rendering()
        self.tip_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, TIP_CAM_NAME,
        )
        if self.tip_cam_id < 0:
            raise RuntimeError(f"Camera {TIP_CAM_NAME!r} not found in compiled model")

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32,
        )
        # Depth has explicit channel dim (1, H, W) so SB3's NatureCNN treats it
        # as a single-channel image without needing an obs wrapper.
        self.observation_space = spaces.Dict({
            "depth": spaces.Box(
                low=0.0, high=1.0,
                shape=(1, self.depth_res, self.depth_res), dtype=np.float32,
            ),
            "state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(STATE_OBS_DIM,), dtype=np.float32,
            ),
        })

        # State
        # target_insertion_depth : metres the handle has been pushed in along
        #                          the entrance tangent. Updated each step by
        #                          the advance action (a[2]). Drives target mocap.
        # actual_base_s : where disk_0 PHYSICALLY sits, projected onto the
        #                 centreline. Reward uses Δactual_base_s so the policy
        #                 only gets credit for motion that actually happened —
        #                 a wall-stuck commanded advance earns 0.
        self.base_s_init = float(self.tip_s_init - scope_gen.REST_LENGTH)
        self.actual_base_s = self.base_s_init
        # High-water mark for furthest s reached so far. Used so that
        # going forward, then back, then forward over the same ground earns
        # no second reward -- only NEW territory counts.
        self.max_actual_base_s = self.base_s_init
        # Rolling window of per-step Δactual_base_s; sum < threshold = stuck.
        self.recent_deltas: deque[float] = deque(maxlen=STUCK_WINDOW)
        self.ejected_threshold = self.base_s_init - EJECTED_MARGIN
        self.s_span = max(self.s_goal - self.base_s_init, 1e-6)
        self.step_count = 0
        self.last_action = np.zeros(3, dtype=np.float32)
        self.prev_action = np.zeros(3, dtype=np.float32)
        # PD-controlled tendon-pair commands (metres). Positive = pulled in +X/+Z
        # tendon, negative = pulled in opposite. ctrl[px]=+cmd, ctrl[nx]=-cmd, etc.
        self.cmd_x_pair = 0.0
        self.cmd_y_pair = 0.0
        self.prev_cmd_x = 0.0   # previous cmd values for PD derivative term
        self.prev_cmd_y = 0.0
        self.force_norm = 0.0
        self._force_steps = 0   # consecutive steps with force_norm >= 1.0
        self._ep_count = 0      # episode counter for periodic renderer recycle
        # Geom IDs belonging to the tip (last) disk, and separately the colon
        # wall body -- used to detect tip-WALL contact specifically. If the tip
        # presses against the wall at 90 deg (UFO posture), a contact between
        # these two sets increments _tip_contact_steps -> truncate after
        # TIP_STUCK_WINDOW. Requiring BOTH sides (not just tip-involvement)
        # excludes tip-vs-shaft self-collision (e.g. the tip curling back onto
        # an earlier, non-adjacent disk -- <contact><exclude> only excludes
        # ADJACENT disk pairs) from counting as a wall jam: self-contact was
        # previously indistinguishable from a real wall obstruction, incurring
        # the same r_tip penalty and tip_stuck truncation either way (found
        # 2026-07-10, fixed same day -- see [[v8-plan-status]] in memory).
        _tip_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, f"disk_{scope_gen.N_DISKS - 1}",
        )
        if _tip_body_id >= 0:
            _ga = int(self.model.body_geomadr[_tip_body_id])
            _gn = int(self.model.body_geomnum[_tip_body_id])
            self._tip_geom_ids: frozenset[int] = frozenset(range(_ga, _ga + _gn))
        else:
            self._tip_geom_ids = frozenset()
        _wall_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "colon_wall",
        )
        if _wall_body_id >= 0:
            _wa = int(self.model.body_geomadr[_wall_body_id])
            _wn = int(self.model.body_geomnum[_wall_body_id])
            self._wall_geom_ids: frozenset[int] = frozenset(range(_wa, _wa + _wn))
        else:
            self._wall_geom_ids = frozenset()
        self._tip_contact_steps = 0
        # Cached depth from the last _observation() call; used by step() to
        # compute the lumen reward without a second render pass.
        self._last_depth = np.zeros(
            (1, self.depth_res, self.depth_res), dtype=np.float32
        )

        # CNN error map for obs noise injection (Option A).
        # Loaded from error_map_64.npy in the same directory as this file.
        # If not found, noise injection is skipped with a one-time warning.
        # Generate the file by running perception/rd_v2/compute_error_map.py
        # and copying error_map_64.npy here.
        _error_map_path = HERE / "error_map_64.npy"
        if _error_map_path.exists():
            _err = np.load(str(_error_map_path)).astype(np.float32)
            if _err.shape != (self.depth_res, self.depth_res):
                from PIL import Image as _PIL_Image
                _err = np.asarray(
                    _PIL_Image.fromarray(_err).resize(
                        (self.depth_res, self.depth_res), _PIL_Image.BILINEAR,
                    ),
                    dtype=np.float32,
                )
            self._cnn_error_map: np.ndarray | None = _err
        else:
            print(
                f"[ScopeColonEnv] error_map_64.npy not found at {_error_map_path}"
                " — CNN obs noise injection disabled.",
                file=sys.stderr, flush=True,
            )
            self._cnn_error_map = None
        # Separate RNG for per-step obs noise so it varies every step independently
        # of the per-episode DR parameter sampling.
        self._obs_rng = np.random.default_rng(self.seed_value + 999983)

        # DR state — resampled each reset(); initialised to nominal (no perturbation)
        self._dr_gain_x   = 1.0
        self._dr_gain_y   = 1.0
        self._dr_dead_x   = 0.0
        self._dr_dead_y   = 0.0
        self._dr_lag      = 0
        # Ring buffer for command lag: stores (eff_x, eff_y) after dead-zone+gain.
        # maxlen=3 covers lags 0, 1, 2 steps.
        self._cmd_buf: deque[tuple[float, float]] = deque(
            [(0.0, 0.0)] * 3, maxlen=3
        )

        # ---- memory-leak diagnostic (SF_MEM_PROFILE=1 only) ----
        if _MEM_PROFILE:
            self._prof_ep      = 0
            self._prof_snap    = None
            self._prof_log     = open(
                f"/tmp/sf_mem_seed{self.seed_value}_pid{os.getpid()}.log", "a",
            )
            self._prof_log.write(
                f"[init] seed={self.seed_value} pid={os.getpid()} "
                f"rss={_rss_mb():.0f}MB\n"
            )
            self._prof_log.flush()
            if not tracemalloc.is_tracing():
                tracemalloc.start(5)

    # ----------------------------------------------------------------- API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self._ep_count += 1
        if self._ep_count % _RENDERER_RECYCLE_EVERY == 0 and self._ep_count > 0:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = mujoco.Renderer(
                self.model, height=self.depth_res, width=self.depth_res,
            )
            self.renderer.enable_depth_rendering()

        if _MEM_PROFILE:
            self._prof_ep += 1
            if self._prof_ep % 5 == 0:
                rss = _rss_mb()
                snap = tracemalloc.take_snapshot()
                if self._prof_snap is not None:
                    diffs = snap.compare_to(self._prof_snap, "lineno")
                    growing = [d for d in diffs if d.size_diff > 0][:12]
                    self._prof_log.write(
                        f"\n[ep={self._prof_ep}] rss={rss:.0f}MB\n"
                    )
                    for d in growing:
                        self._prof_log.write(
                            f"  +{d.size_diff/1024:6.1f}KB  {d}\n"
                        )
                else:
                    self._prof_log.write(
                        f"\n[ep=0] rss={rss:.0f}MB  (baseline)\n"
                    )
                self._prof_log.flush()
                self._prof_snap = snap

        mujoco.mj_resetData(self.model, self.data)
        self.target_insertion_depth = 0.0
        self.actual_base_s = self.base_s_init
        self.max_actual_base_s = self.base_s_init
        self.recent_deltas.clear()
        self.step_count = 0
        self.last_action[:] = 0.0
        self.prev_action[:] = 0.0
        self.cmd_x_pair = 0.0
        self.cmd_y_pair = 0.0
        self.prev_cmd_x = 0.0
        self.prev_cmd_y = 0.0
        self.force_norm = 0.0
        self._force_steps = 0
        self._tip_contact_steps = 0

        # Resample DR parameters for this episode.
        # Seed is deterministic (reproducible) but varies per episode.
        _dr_rng = np.random.default_rng(self.seed_value * 10007 + self._ep_count)
        self._dr_gain_x   = float(_dr_rng.uniform(*DR_GAIN_X_RANGE))
        self._dr_gain_y   = float(_dr_rng.uniform(*DR_GAIN_Y_RANGE))
        # Apply gain to the actuator's FORCE output (friction/efficiency loss),
        # not the commanded target -- the endpoint stays reachable, only the
        # force/speed getting there is attenuated. ctrl order px/pz/nx/nz.
        for _i, _g in ((0, self._dr_gain_x), (2, self._dr_gain_x),
                       (1, self._dr_gain_y), (3, self._dr_gain_y)):
            self.model.actuator_gainprm[_i] = self._tendon_gainprm0[_i] * _g
            self.model.actuator_biasprm[_i] = self._tendon_biasprm0[_i] * _g
        # Same DR gain also attenuates the integral actuators' force -- otherwise
        # the policy could lean on an always-fully-effective integral term to
        # bypass DR entirely regardless of how lossy the sampled cable gain is.
        for _name, _g in (("px", self._dr_gain_x), ("nx", self._dr_gain_x),
                          ("pz", self._dr_gain_y), ("nz", self._dr_gain_y)):
            self.model.actuator_gainprm[self._aid_i[_name]] = (
                self._tendon_i_gainprm0[_name] * _g
            )
        self._dr_dead_x   = float(_dr_rng.uniform(*DR_DEAD_X_RANGE))
        self._dr_dead_y   = float(_dr_rng.uniform(*DR_DEAD_Y_RANGE))
        self._dr_lag      = int(_dr_rng.integers(DR_LAG_RANGE[0], DR_LAG_RANGE[1] + 1))
        self._cmd_buf.clear()
        for _ in range(3):
            self._cmd_buf.append((0.0, 0.0))

        # shaft_link_0's free joint returns to its XML-specified initial pose
        # (already threaded partway through the entrance rollers) via
        # mj_resetData() above -- no manual repositioning needed since there's
        # no mocap to drive anymore. Just zero the roller target so it starts
        # matching the reset (unloaded) roller qpos.
        self.data.ctrl[ROLLER_CTRL_SLICE] = 0.0
        mujoco.mj_forward(self.model, self.data)
        try:
            obs = self._observation()
        except Exception as _e:
            print(
                f"[ScopeColonEnv seed={self.seed_value}] reset() _observation() error: {_e}",
                file=sys.stderr, flush=True,
            )
            obs = {
                "depth": np.zeros((1, self.depth_res, self.depth_res), dtype=np.float32),
                "state": np.zeros(STATE_OBS_DIM, dtype=np.float32),
            }
        return obs, {}

    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        a = np.clip(a, -1.0, 1.0)
        x, y, z = float(a[0]), float(a[1]), float(a[2])

        # ---- action -> PD-controlled tendon-pair commands ----
        # PD tracks the RL's desired position (a[0/1]*max_pull_{x,y}).
        # D term damps cmd velocity — resists rapid reversals that rate-limiting
        # only clipped rather than actively opposing.
        target_x = x * self.max_pull_x
        target_y = y * self.max_pull_y
        vel_x = self.cmd_x_pair - self.prev_cmd_x
        vel_y = self.cmd_y_pair - self.prev_cmd_y
        self.prev_cmd_x = self.cmd_x_pair
        self.prev_cmd_y = self.cmd_y_pair
        self.cmd_x_pair = float(np.clip(
            self.cmd_x_pair + PD_KP * (target_x - self.cmd_x_pair) - PD_KD * vel_x,
            -self.max_pull_x, self.max_pull_x,
        ))
        self.cmd_y_pair = float(np.clip(
            self.cmd_y_pair + PD_KP * (target_y - self.cmd_y_pair) - PD_KD * vel_y,
            -self.max_pull_y, self.max_pull_y,
        ))
        # ---- apply domain randomisation: dead zone → lag ----
        # Dead zone: commands below the threshold produce no tip movement (tendon slack).
        # Gain (friction/efficiency loss) is applied to the actuator's force output
        # instead (see reset()) -- the commanded target itself is unaffected, so the
        # endpoint stays reachable at any gain, only the force/speed getting there varies.
        _dead_x_abs = self._dr_dead_x * self.max_pull_x
        _dead_y_abs = self._dr_dead_y * self.max_pull_y
        _eff_x = 0.0 if abs(self.cmd_x_pair) < _dead_x_abs else self.cmd_x_pair
        _eff_y = 0.0 if abs(self.cmd_y_pair) < _dead_y_abs else self.cmd_y_pair
        # Lag: push this step's effective command into the ring buffer, then
        # pull the delayed command from lag steps ago.
        self._cmd_buf.append((_eff_x, _eff_y))
        _delayed_x, _delayed_y = self._cmd_buf[-(self._dr_lag + 1)]
        # ctrl order from scope_gen.TENDONS: px, pz, nx, nz.
        self.data.ctrl[0] = _delayed_x       # pull_px
        self.data.ctrl[2] = -_delayed_x      # pull_nx (antagonistic)
        self.data.ctrl[1] = _delayed_y       # pull_pz
        self.data.ctrl[3] = -_delayed_y      # pull_nz

        # ---- action -> entrance roller target angle ----
        # Drive the entrance feed rollers (position-controlled, see
        # build_collision_scene.py) toward a target angle. The rollers grip
        # and push the shaft via friction; wall contact at bends converts the
        # straight-line push into curved-path advance (real colonoscopy
        # physics), and the roller can slip under resistance rather than
        # rigidly forcing the chain -- actual progress is measured separately
        # from disk_0's true physical position (actual_base_s below), not
        # this commanded value.
        ds_command = z * self.advance_rate_max
        # Lower bound only (can't command negative insertion). The old upper
        # bound (shaft length + a 200mm margin, an arbitrary safety ceiling)
        # was removed 2026-07-02: the real physical limit is now enforced by
        # the shaft_exhausted termination below, which fires before this
        # accumulator could run away to an unreasonable value -- that old
        # ceiling being hit mid-episode was what silently froze the roller's
        # commanded target (zero actuator error, looked like "the roller
        # stopped moving" with no diagnostic explaining why).
        self.target_insertion_depth = float(max(0.0, self.target_insertion_depth + ds_command))
        # Sign convention validated in build_collision_scene.py / manual_test.py:
        # negative roller angle = insertion, positive = retraction.
        self.data.ctrl[ROLLER_CTRL_SLICE] = -self.target_insertion_depth / ROLLER_RADIUS_M

        # ---- physics ----
        # Drive the 4 tendon integral actuators each substep (not just once per
        # RL step): each is fed the current length-tracking error, or 0 (freezing
        # its accumulated activation state -- anti-windup) whenever this tendon's
        # combined P+I force is already within ANTIWINDUP_FORCE_MARGIN of the real
        # 80N cable limit. Target length is re-derived from ctrl[0:4] (set just
        # above) each substep since that's the actual position-actuator target;
        # ctrl itself doesn't change during substeps, only ten_length does.
        prev_actual_s = self.actual_base_s
        _rest_length = scope_gen.REST_LENGTH
        _force_rail = -scope_gen.ACTUATOR_PULL_FORCE_LIMIT
        for _ in range(self.physics_per_step):
            for _name in self._tendon_names:
                _ap, _ai, _tid = self._aid_p[_name], self._aid_i[_name], self._tid[_name]
                _target_len = _rest_length - self.data.ctrl[_ap]
                _length_error = self.data.ten_length[_tid] - _target_len
                _combined_force = self.data.actuator_force[_ap] + self.data.actuator_force[_ai]
                _saturated = _combined_force <= (_force_rail + ANTIWINDUP_FORCE_MARGIN)
                self.data.ctrl[_ai] = 0.0 if _saturated else _length_error
            mujoco.mj_step(self.model, self.data)

        # NaN guard: extreme collision forces can corrupt qpos/qvel. Catching
        # this here prevents the worker from dying on a subsequent bad render
        # or contact-force calculation.
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            print(
                f"[ScopeColonEnv seed={self.seed_value}] NaN physics at step "
                f"{self.step_count} — resetting data and terminating episode",
                file=sys.stderr, flush=True,
            )
            mujoco.mj_resetData(self.model, self.data)
            _safe_obs = {
                "depth": np.zeros((1, self.depth_res, self.depth_res), dtype=np.float32),
                "state": np.zeros(STATE_OBS_DIM, dtype=np.float32),
            }
            return _safe_obs, 0.0, True, False, {
                "termination_reason": "nan_physics",
                "episode_extra_stats": {
                    "progress": 0.0, "shaft_exhausted": 0.0,
                    "stuck": 0.0, "tip_stuck": 0.0, "ejected": 0.0, "truncated": 0.0,
                },
            }

        # actual_base_s = where disk_0 physically ended up, projected onto the
        # centreline. Wall contact holds the shaft back so the projection lags
        # behind the commanded insertion depth — exactly the physics we want.
        disk0_world_pos = self.data.xpos[self.disk0_body_id].copy()
        self.actual_base_s = self._project_onto_centreline(disk0_world_pos)
        ds_actual = self.actual_base_s - prev_actual_s

        # shaft_exhausted = the shaft's proximal/base end has been physically
        # fed all the way up to the outer roller ring -- no shaft left
        # trailing behind to grip and feed further. Real physical boundary,
        # not a distance target; see shaft0_body_id's own comment in __init__.
        shaft0_world_pos = self.data.xpos[self.shaft0_body_id]
        shaft0_proj = float(np.dot(shaft0_world_pos, self.entrance_tangent))
        shaft_exhausted = shaft0_proj >= self._roller_ring1_proj

        # ---- insertion force ----
        # Sum of all contact force magnitudes across the scope/colon interface.
        # Captures both wall grazing and the curl-into-a-ball cheat (which
        # produces ~215 N vs ~0.2 N for gentle advance — 1000x discriminative).
        _cbuf = np.zeros(6)
        raw_force = 0.0
        _tip_in_contact = False
        for _i in range(self.data.ncon):
            mujoco.mj_contactForce(self.model, self.data, _i, _cbuf)
            raw_force += float(np.linalg.norm(_cbuf[:3]))
            if not _tip_in_contact:
                _c = self.data.contact[_i]
                _tip_side = _c.geom1 in self._tip_geom_ids or _c.geom2 in self._tip_geom_ids
                _wall_side = _c.geom1 in self._wall_geom_ids or _c.geom2 in self._wall_geom_ids
                if _tip_side and _wall_side:
                    _tip_in_contact = True
        if _tip_in_contact:
            self._tip_contact_steps += 1
        else:
            self._tip_contact_steps = 0
        self.force_norm = float(np.clip(raw_force / MAX_INSERTION_FORCE, 0.0, 1.0))

        # ---- reward ----
        # r = W_PROGRESS * new_territory * (1 - W_FORCE_MULT * force_norm)
        #
        # Reward trial history — removed components:
        #   W_LUMEN_BOOST * lumen_s multiplier (was 0.5, v16): boosted progress
        #     when lumen centred in depth image. Removed v17: noisy _lumen_score
        #     at early training added noise to an already weak gradient; force
        #     multiplier already implicitly teaches lumen-centering.
        #   r_force = -W_FORCE * force_norm (v11, W_FORCE=0.01): per-step force
        #     penalty. Removed v12: penalty accumulated faster than progress
        #     reward — policy collapsed and refused to move; advantage ≈ 0.
        #   r_lumen standalone (v14, W_LUMEN=0.001): per-step lumen bonus.
        #     Removed v15: caused farming — scope stared at lumen without
        #     advancing; per-step accumulation >> sparse progress reward.
        #   (1 - force_norm) hard gate (v14-v17, sole multiplier): zero gradient
        #     at high force → chicken-and-egg: no reward to learn steering, no
        #     steering to earn reward. Softened to (0.3 + 0.7*(1-f)) (v18-v8_p1_v1)
        #     to keep a gradient floor while still rewarding low-contact advances
        #     more. Zeroed via W_FORCE_MULT (2026-07-11, v8_p1_v2): shaft-wall
        #     contact is a physical necessity when clearing a bend, not just the
        #     cheat case this was meant to catch -- see W_FORCE_MULT's own
        #     comment above.
        prev_max_s = self.max_actual_base_s
        new_territory = max(0.0, self.actual_base_s - self.max_actual_base_s)
        self.max_actual_base_s = max(self.max_actual_base_s, self.actual_base_s)
        self.recent_deltas.append(self.max_actual_base_s - prev_max_s)
        _force_mult = 1.0 - W_FORCE_MULT * self.force_norm
        r_progress = W_PROGRESS * new_territory * _force_mult
        # Dense shaping: reward any forward step, not just new-territory records.
        # ds_actual is the physical step taken this tick (positive = forward).
        # Re-advancing over previously visited ground earns W_SHAPING * ds (vs
        # W_PROGRESS + W_SHAPING on first-pass), so territory stays the primary signal.
        r_shaping = W_SHAPING * max(0.0, ds_actual) * _force_mult
        r_tip = -W_TIP_PENALTY if _tip_in_contact else 0.0

        # Stuck: high-water mark hasn't advanced STUCK_THRESHOLD in STUCK_WINDOW
        # steps. Farming (oscillating without new territory) is also caught
        # since the deque tracks d_max_s, not raw ds_actual.
        _stuck = (
            len(self.recent_deltas) >= STUCK_WINDOW
            and sum(self.recent_deltas) < STUCK_THRESHOLD
        )
        _tip_stuck = self._tip_contact_steps >= TIP_STUCK_WINDOW

        # _force_steps kept for info-dict diagnostics only — force_limit
        # termination was removed (v17) to prevent the policy exploiting it
        # as a fast episode-reset mechanism.
        if self.force_norm >= 1.0:
            self._force_steps += 1
        else:
            self._force_steps = 0

        termination_reason: str | None = None
        if shaft_exhausted:
            termination_reason = "shaft_exhausted"
        # Only shaft_exhausted/ejected are true terminations. Stuck, tip-stuck,
        # and max steps are truncations. Policy must learn to minimise wall
        # contact via the (1-force_norm) reward multiplier and tip-contact
        # penalty. shaft_exhausted is a natural physical boundary, NOT a
        # success condition -- it earns no reward bonus, and progress below
        # is reported as whatever was actually achieved (not forced to 100%),
        # since running out of shaft says nothing about how far the tip got.
        elif self.actual_base_s < self.ejected_threshold:
            termination_reason = "ejected"

        r_stuck = -W_STUCK_PENALTY if (
            termination_reason is None and (_stuck or _tip_stuck)
        ) else 0.0
        reward = float(r_progress + r_shaping + r_tip + r_stuck)
        terminated = termination_reason is not None
        self.step_count += 1
        _max_steps = self.step_count >= self.max_steps
        truncated = (not terminated) and (bool(_max_steps) or _stuck or _tip_stuck)
        truncation_reason: str | None = None
        if truncated and not terminated:
            if _tip_stuck:
                truncation_reason = "tip_stuck"
            elif _stuck:
                truncation_reason = "stuck"
            elif _max_steps:
                truncation_reason = "max_steps"

        self.prev_action[:] = self.last_action
        self.last_action[:] = a
        # goal_gap_m is always the RAW measured value now -- no more forcing to
        # 100%/0 on termination, since shaft_exhausted (unlike the old "goal")
        # doesn't imply the tip actually got anywhere near the end.
        # s_goal/s_span are kept purely as a fixed reference distance for this
        # percentage, not as a termination target.
        #
        # progress is CLIPPED to [0, 1]. s_goal sits END_CLEARANCE (100mm) short
        # of the physical centreline end, but nothing stops disk_0 travelling
        # into that clearance -- so the raw ratio tops out at ~127% on the 500mm
        # colon, and goes negative when the scope backs out. v6_dr's goal
        # termination capped this implicitly; v7_shaft's shaft_exhausted does
        # not. Reward never reads progress (r_progress uses new_territory in
        # metres), so this is a reporting fix only. progress_raw keeps the
        # unclipped value for diagnostics.
        progress_raw = (self.actual_base_s - self.base_s_init) / self.s_span
        progress = min(1.0, max(0.0, progress_raw))
        goal_gap_m = max(0.0, self.s_goal - self.actual_base_s)
        max_seen_goal_gap_m = max(0.0, self.s_goal - self.max_actual_base_s)
        info = {
            "target_insertion_depth": self.target_insertion_depth,
            "actual_base_s": self.actual_base_s,
            "max_seen_s": self.max_actual_base_s,
            "goal_s": self.s_goal,
            "goal_gap_m": goal_gap_m,
            "max_seen_goal_gap_m": max_seen_goal_gap_m,
            "progress": progress,
            "progress_raw": progress_raw,
            "ncon": int(self.data.ncon),
            "force_norm": self.force_norm,
            "force_steps": self._force_steps,
            "raw_force_n": raw_force,
            "tip_contact_steps": self._tip_contact_steps,
            "stuck_window_delta_m": float(sum(self.recent_deltas)),
            "r_progress": float(r_progress),
            "r_shaping": float(r_shaping),
            "r_tip": float(r_tip),
            "r_stuck": float(r_stuck),
            "termination_reason": termination_reason,
            "truncation_reason": truncation_reason,
        }
        if terminated or truncated:
            # SF reads episode_extra_stats at episode end and merges the keys
            # into policy_avg_stats, making them visible to training observers.
            info["episode_extra_stats"] = {
                "progress":        float(progress),
                "shaft_exhausted": 1.0 if termination_reason == "shaft_exhausted" else 0.0,
                "ejected":         1.0 if termination_reason == "ejected"         else 0.0,
                "stuck":           1.0 if _stuck                                  else 0.0,
                "tip_stuck":       1.0 if _tip_stuck                              else 0.0,
                "truncated":       1.0 if truncated                               else 0.0,
            }
        try:
            obs = self._observation()
        except Exception as _e:
            print(
                f"[ScopeColonEnv seed={self.seed_value}] _observation() error: {_e}",
                file=sys.stderr, flush=True,
            )
            obs = {
                "depth": np.zeros((1, self.depth_res, self.depth_res), dtype=np.float32),
                "state": np.zeros(STATE_OBS_DIM, dtype=np.float32),
            }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if getattr(self, "renderer", None) is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None

    # ------------------------------------------------------------ internals

    def _observation(self) -> dict[str, np.ndarray]:
        """Dict obs:

            depth : (1, H, W) float32, metric depth from tip_cam,
                    clipped & normalised by depth_clip_m to lie in [0, 1].
            state : (8,) float32:
                    [cmd_x_n, cmd_y_n, ten_x_n, ten_z_n,
                     last_action[0..2], tip_contact]
        """
        # Depth render — group=3 collision prisms are hidden so we see the
        # smooth visual mesh, matching what DA3 would see at deploy.
        self.renderer.update_scene(self.data, camera=self.tip_cam_id)
        depth_raw = self.renderer.render()
        # Normalise by the maximum depth in this frame so the farthest visible
        # point is always 1.0. This preserves relative depth structure regardless
        # of scale — the policy can always read the lumen geometry even when
        # pressed against a wall (fixed /0.5m clipping gave near-black images there).
        max_d = max(float(depth_raw.max()), 1e-6)
        depth = np.clip(depth_raw / max_d, 0.0, 1.0).astype(np.float32)
        depth = depth[None, :, :]   # (1, H, W) for SB3 NatureCNN

        # Inject spatially-structured CNN obs noise (Option A).
        # Simulates DA3 depth estimation errors at deployment so the RL policy
        # learns to be robust to the actual spatial noise distribution rather
        # than assuming perfect depth observations.
        # Noise is multiplicative: std scales with depth value so close walls
        # get small absolute noise and open lumen gets larger absolute noise —
        # matching how monocular depth error grows with distance.
        if self._cnn_error_map is not None:
            _noise = self._obs_rng.standard_normal(
                (self.depth_res, self.depth_res)
            ).astype(np.float32)
            _noise *= self._cnn_error_map * depth[0]
            depth = np.clip(depth + _noise[None, :, :], 0.0, 1.0)

        cmd_x_n = self.cmd_x_pair / self.max_pull_x
        cmd_y_n = self.cmd_y_pair / self.max_pull_y
        # Net per-axis cable extension: hardware-realistic analogue of a
        # linear encoder differential on each antagonistic cable pair.
        # tendon order (scope_gen.TENDONS): px(0), pz(1), nx(2), nz(3). Sign
        # convention matches cmd_x_n/cmd_y_n (positive = net pull toward
        # +X/+Z): pulling +X shortens px and lengthens nx, so
        # (ten_length[nx] - ten_length[px]) grows positive. Directly
        # measurable on real hardware via motor/encoder position on each cable.
        tl = self.data.ten_length
        ten_x_n = float(np.clip((tl[2] - tl[0]) / TEN_X_RANGE_M, -1.0, 1.0))
        ten_z_n = float(np.clip((tl[3] - tl[1]) / TEN_Z_RANGE_M, -1.0, 1.0))
        # 8D state. True colon progress is privileged sim information and is not
        # available at deployment, so it is excluded here; progress is still
        # reported in info/reward diagnostics.
        state = np.array([
            cmd_x_n, cmd_y_n, ten_x_n, ten_z_n,
            self.last_action[0], self.last_action[1], self.last_action[2],
            float(self._tip_contact_steps > 0),
        ], dtype=np.float32)
        self._last_depth = depth   # cache for lumen reward in next step()
        return {"depth": depth, "state": state}

    def _lumen_score(self, depth: np.ndarray) -> float:
        """Mean depth in the central 50%×50% patch of the tip camera image.

        Returns a value in [0, 1]: near 1.0 when the scope is looking down
        open lumen (far space ahead); near 0.0 when facing a wall. The policy
        learns the optimal viewing angle to maximise this through exploration —
        the reward provides direction without specifying lumen geometry.
        """
        _, H, W = depth.shape
        cy, cx = H // 2, W // 2
        h4, w4 = max(1, H // 4), max(1, W // 4)
        return float(depth[0, cy - h4 : cy + h4, cx - w4 : cx + w4].mean())

    def _project_onto_centreline(self, world_pos: np.ndarray) -> float:
        """Return the centreline arc length corresponding to world_pos.
        For positions backward of the entrance, returns a negative s using
        the tangent at s=0 (so the extrapolation region is continuous with
        the in-tube parameterisation)."""
        dists = np.linalg.norm(self.centreline - world_pos[None, :], axis=1)
        idx = int(np.argmin(dists))
        if idx == 0:
            # Might be in the negative-s extrapolation region.
            offset = world_pos - self.centreline[0]
            projection = float(np.dot(offset, self.entrance_tangent))
            if projection < 0.0:
                return projection
        return float(self.arc_s[idx])
