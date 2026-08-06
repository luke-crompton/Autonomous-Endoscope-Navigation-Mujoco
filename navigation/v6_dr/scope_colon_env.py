"""
Gymnasium environment for the scope-in-colon scene.

  - action_space      = Box(-1, 1, shape=(3,))
                          a[0]  -> tendon px/nx setpoint
                          a[1]  -> tendon pz/nz setpoint
                          a[2]  -> mocap base advance rate (signed)
  - observation_space = Dict{
        depth: Box(0,1, (1,H,W))   metric depth from tip_cam, normalised
        state: Box(-inf,inf, (10,)) [cmd_x_n, cmd_y_n, progress_placeholder,
                                     force_norm, net_fx, net_fz,
                                     last_action[0..2], tip_contact]
    }
  - reward            = W_PROGRESS * new_territory * force multiplier
                      - W_TIP_PENALTY while the tip disk contacts the wall
  - terminated        = goal reached | ejected
  - truncated         = stuck | tip_stuck | step >= max_steps

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
    pose_for_base_s,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_STEPS = 10000
DEFAULT_ADVANCE_RATE_MAX = 0.0003      # 0.3 mm per env step (~30 mm/s; was 1 mm/step = 100 mm/s)
DEFAULT_PHYSICS_PER_STEP = 20          # MuJoCo timestep is 0.5 ms; 20 -> 10 ms per env step
DEFAULT_TIP_S_INIT = 0.020             # 20 mm into the colon — deeper start so more of the scope body is inside and constrained by the walls, reducing curl-out behaviour
END_CLEARANCE = 0.10                   # stop base 10 cm before physical centreline end; with the 20 cm visual-only tail this is 30 cm before the rendered distal end
GOAL_TOLERANCE_M = 0.010               # 10 mm finish band; near-end stalls count as goal instead of stuck/truncated
GOAL_PULL_THROUGH_M = 0.050            # allow the target to pull past the admin goal so lagged physical base can still cross it
MAX_BASE_LAG = 0.020                   # max target-vs-physical base lag; prevents tangent-slide shortcuts through bends
PD_KP = 0.06                           # proportional gain: ~1.44mm max first step, matches v3 rate
PD_KD = 0.30                           # derivative gain: strongly over-damped, ±1.1mm steady-state jitter
DEFAULT_DEPTH_RES = 64                 # tip-cam depth obs resolution (HxW)
DEFAULT_DEPTH_CLIP_M = 0.5             # depth values clipped & normalised by this many metres
TIP_CAM_NAME = "tip_cam"
N_HINGE_JOINTS = 25                    # joint_1..joint_25, one per scope hinge
HINGE_RANGE = 0.1745                   # rad; joint XML range is ±0.1745 — normalises angles to [-1, 1]
# State obs: [cmd_x_n, cmd_y_n, progress_placeholder, force_norm, net_fx, net_fz, last_action[0..2], tip_contact] = 10D
# The progress slot is kept only for checkpoint/architecture compatibility and is
# always 0.0, so the policy does not receive privileged colon-progress info.
# Removed from v17: prev_action[3] (redundant with RNN) and hinge_angles[25] (67% of
# obs vector, mostly reflects colon geometry not policy choices; depth image covers this).
# tip_contact added: explicit 0/1 flag for tip disk touching colon wall (UFO posture detection).
STATE_OBS_DIM = 10
OBS_PROGRESS_PLACEHOLDER = 0.0

# ----- reward shaping ------------------------------------------------------
W_PROGRESS = 2.0          # multiplied by metres of *new-territory* advance
W_SHAPING  = 0.0          # dense shaping: any forward step earns reward, not just new territory.
                          # Additive on top of W_PROGRESS * new_territory so first-pass advance
                          # earns (W_PROGRESS + W_SHAPING) * ds; re-advance after reversal earns
                          # W_SHAPING * ds. Eliminates zero-gradient on re-traversed ground.
W_GOAL = 0.0              # no terminal jackpot; high-water progress already makes finishing optimal
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
TIP_STUCK_WINDOW = 50     # consecutive steps tip disk contacts colon wall → truncate (UFO check)
W_TIP_PENALTY    = 0.001  # per-step reward penalty while tip disk is in wall contact
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
#   Fraction of max_pull (24 mm) that must be exceeded before the tip moves.
#   Caused by tendon slack that varies with insertion depth and shaft geometry.
#   Keep this small for training: 0.03 → ~0.7 mm of slack. Larger values
#   made fine steering vanish and encouraged jittery bang-bang corrections.
#   Calibrate: measure the command threshold below which tip angle is zero at
#   each insertion depth.
#
# Command lag (DR_LAG_RANGE, env steps):
#   Delay between issuing a tendon command and it reaching the tip (mechanical
#   inertia + tendon stretch). Each env step = 10 ms, so lag 2 = 20 ms.
#   Longer shafts and tighter bends increase lag.
#   Calibrate: step-response test — measure time from command to tip movement.
#
# Advance gain (DR_ADV_GAIN_RANGE):
#   Friction on the shaft body when pushing forward. Wider range than steering
#   gains because the whole rigid shaft is being pushed, not just a wire.
#   Keep the floor high enough that clean straight-lumen insertion does not
#   crawl for tens of seconds; 0.65 with the default advance rate is 19.5 mm/s.
#   Calibrate: measure actual insertion speed vs commanded advance rate at
#   different insertion depths and colon curvatures.

DR_GAIN_X_RANGE   = (0.45, 1.0)   # x-axis tendon efficiency [unitless]
DR_GAIN_Y_RANGE   = (0.45, 1.0)   # y-axis tendon efficiency [unitless]
DR_DEAD_X_RANGE   = (0.0,  0.03)  # x-axis dead zone as fraction of max_pull [unitless]
DR_DEAD_Y_RANGE   = (0.0,  0.03)  # y-axis dead zone as fraction of max_pull [unitless]
DR_LAG_RANGE      = (0,    2)     # command lag in env steps, inclusive [steps]
DR_ADV_GAIN_RANGE = (0.65, 1.0)   # advance efficiency [unitless]


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
        self.max_pull = float(scope_gen.MAX_PULL)
        self.depth_res = int(depth_res)
        self.depth_clip_m = float(depth_clip_m)

        # Generate scene + load MuJoCo model
        self.xml_path, self.centreline = build_scene(self.seed_value)
        self.arc_s = arc_length(self.centreline)
        self.total_length = float(self.arc_s[-1])
        self.s_goal = self.total_length - END_CLEARANCE
        self.target_s_max = min(self.total_length, self.s_goal + GOAL_PULL_THROUGH_M)

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        # ---- ids: target (mocap goal pose), scope_anchor (dynamic body) ----
        target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target",
        )
        if target_body_id < 0:
            raise RuntimeError("target body not found in compiled model")
        self.target_mocap_id = int(self.model.body_mocapid[target_body_id])
        if self.target_mocap_id < 0:
            raise RuntimeError("target body is not mocap")

        self.anchor_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "scope_anchor",
        )
        if self.anchor_body_id < 0:
            raise RuntimeError("scope_anchor body not found in compiled model")

        self.slide_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "anchor_slide",
        )
        if self.slide_joint_id < 0:
            raise RuntimeError("anchor_slide joint not found")
        self.slide_qpos_addr = int(self.model.jnt_qposadr[self.slide_joint_id])
        self.slide_qvel_addr = int(self.model.jnt_dofadr[self.slide_joint_id])
        # Static upper bound on slide (matches XML); the lower bound gets
        # rewritten each step so actual_base_s >= base_s_init at all times.
        self.slide_qpos_max = float(self.model.jnt_range[self.slide_joint_id, 1])
        self.slide_qpos_floor_hard = float(self.model.jnt_range[self.slide_joint_id, 0])

        # Entrance tangent for extrapolated-backward projection (s < 0).
        entrance_tangent = self.centreline[1] - self.centreline[0]
        self.entrance_tangent = entrance_tangent / (
            np.linalg.norm(entrance_tangent) + 1e-12
        )

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

        # Collect qpos addresses for the 25 scope hinge joints (joint_1..joint_25).
        # Provides proprioception: the policy can observe its own shape.
        self.hinge_qpos_addrs: list[int] = [
            int(self.model.jnt_qposadr[j])
            for j in range(self.model.njnt)
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        if len(self.hinge_qpos_addrs) != N_HINGE_JOINTS:
            raise RuntimeError(
                f"Expected {N_HINGE_JOINTS} hinge joints, "
                f"found {len(self.hinge_qpos_addrs)}"
            )

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
        # target_base_s : where the env *wants* the base, advanced each step
        #                 by the action's z component. Drives the target mocap.
        # actual_base_s : where the scope base PHYSICALLY ended up after the
        #                 weld + collisions resolved. Projection of the live
        #                 scope_anchor world position onto the centreline.
        # Reward uses Δactual_base_s so the policy only gets credit for motion
        # that actually happened (a wall-stuck commanded advance earns 0).
        self.base_s_init = float(self.tip_s_init - scope_gen.REST_LENGTH)
        self.target_base_s = self.base_s_init
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
        # Geom IDs belonging to the tip (last) disk — used to detect tip-wall contact.
        # If the tip presses against the ring at 90° (UFO posture), any contact
        # involving these geoms increments _tip_contact_steps → truncate after TIP_STUCK_WINDOW.
        _tip_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, f"disk_{scope_gen.N_DISKS - 1}",
        )
        if _tip_body_id >= 0:
            _ga = int(self.model.body_geomadr[_tip_body_id])
            _gn = int(self.model.body_geomnum[_tip_body_id])
            self._tip_geom_ids: frozenset[int] = frozenset(range(_ga, _ga + _gn))
        else:
            self._tip_geom_ids = frozenset()
        self._tip_contact_steps = 0
        # Cached depth from the last _observation() call; used by step() to
        # compute the lumen reward without a second render pass.
        self._last_depth = np.zeros(
            (1, self.depth_res, self.depth_res), dtype=np.float32
        )

        # CNN error map for obs noise injection (Option A).
        # Loaded from error_map_64.npy in the same directory as this file.
        # If not found, noise injection is skipped with a one-time warning.
        # Generate the file by running Perception/RD_V2/compute_error_map.py
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
        self._dr_adv_gain = 1.0
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
        self.target_base_s = self.base_s_init
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
        self._dr_dead_x   = float(_dr_rng.uniform(*DR_DEAD_X_RANGE))
        self._dr_dead_y   = float(_dr_rng.uniform(*DR_DEAD_Y_RANGE))
        self._dr_lag      = int(_dr_rng.integers(DR_LAG_RANGE[0], DR_LAG_RANGE[1] + 1))
        self._dr_adv_gain = float(_dr_rng.uniform(*DR_ADV_GAIN_RANGE))
        self._cmd_buf.clear()
        for _ in range(3):
            self._cmd_buf.append((0.0, 0.0))

        # Place the target mocap at the initial base pose. scope_anchor's
        # slide-joint qpos defaults to 0 (at rest = at target's pose), which
        # is what we want.
        init_pos, init_quat = self._pose_for_base_s_np(self.base_s_init)
        self._set_target_mocap(init_pos, init_quat)
        self.data.qpos[self.slide_qpos_addr] = 0.0
        self.data.qvel[self.slide_qvel_addr] = 0.0
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
        # PD tracks the RL's desired position (a[0/1]*max_pull).
        # D term damps cmd velocity — resists rapid reversals that rate-limiting
        # only clipped rather than actively opposing.
        target_x = x * self.max_pull
        target_y = y * self.max_pull
        vel_x = self.cmd_x_pair - self.prev_cmd_x
        vel_y = self.cmd_y_pair - self.prev_cmd_y
        self.prev_cmd_x = self.cmd_x_pair
        self.prev_cmd_y = self.cmd_y_pair
        self.cmd_x_pair = float(np.clip(
            self.cmd_x_pair + PD_KP * (target_x - self.cmd_x_pair) - PD_KD * vel_x,
            -self.max_pull, self.max_pull,
        ))
        self.cmd_y_pair = float(np.clip(
            self.cmd_y_pair + PD_KP * (target_y - self.cmd_y_pair) - PD_KD * vel_y,
            -self.max_pull, self.max_pull,
        ))
        # ---- apply domain randomisation: dead zone → gain → lag ----
        # Dead zone: commands below the threshold produce no tip movement (tendon slack).
        _dead_x_abs = self._dr_dead_x * self.max_pull
        _dead_y_abs = self._dr_dead_y * self.max_pull
        _eff_x = 0.0 if abs(self.cmd_x_pair) < _dead_x_abs else self.cmd_x_pair * self._dr_gain_x
        _eff_y = 0.0 if abs(self.cmd_y_pair) < _dead_y_abs else self.cmd_y_pair * self._dr_gain_y
        # Lag: push this step's effective command into the ring buffer, then
        # pull the delayed command from lag steps ago.
        self._cmd_buf.append((_eff_x, _eff_y))
        _delayed_x, _delayed_y = self._cmd_buf[-(self._dr_lag + 1)]
        # ctrl order from scope_gen.TENDONS: px, pz, nx, nz.
        self.data.ctrl[0] = _delayed_x       # pull_px
        self.data.ctrl[2] = -_delayed_x      # pull_nx (antagonistic)
        self.data.ctrl[1] = _delayed_y       # pull_pz
        self.data.ctrl[3] = -_delayed_y      # pull_nz

        # ---- action -> target advance ----
        ds_command = z * self.advance_rate_max * self._dr_adv_gain
        desired_target_base_s = float(np.clip(
            self.target_base_s + ds_command, self.base_s_init, self.target_s_max,
        ))
        # Do not let the mocap target run far ahead of the physical base.
        # With the slide joint parented under a curved centreline target, large
        # lag becomes a straight tangent offset through the bend; contacts cannot
        # resolve that lateral mocap constraint. Keep lag local instead.
        self.target_base_s = min(
            desired_target_base_s,
            self.actual_base_s + MAX_BASE_LAG,
        )
        target_pos, target_quat = self._pose_for_base_s_np(self.target_base_s)
        self._set_target_mocap(target_pos, target_quat)
        # Dynamic slide-joint floor: keeps actual_base_s >= base_s_init, while
        # also capping lag so the anchor cannot shortcut through a curved wall.
        slide_floor = max(
            self.base_s_init - self.target_base_s,
            -MAX_BASE_LAG,
        )
        self.model.jnt_range[self.slide_joint_id, 0] = slide_floor

        # ---- physics ----
        prev_actual_s = self.actual_base_s
        for _ in range(self.physics_per_step):
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
                    "progress": 0.0, "goal": 0.0, "force_limit": 0.0,
                    "stuck": 0.0, "ejected": 0.0, "truncated": 0.0,
                },
            }

        # Hard slide clamp: MuJoCo's joint-range constraint is soft and can be
        # violated under large collision forces. Clamp the actual qpos to the
        # same hard floor used above, then resync geom positions for rendering.
        slide_qpos = float(self.data.qpos[self.slide_qpos_addr])
        if slide_qpos < slide_floor:
            slide_qpos = slide_floor
            self.data.qpos[self.slide_qpos_addr] = slide_qpos
            if float(self.data.qvel[self.slide_qvel_addr]) < 0.0:
                self.data.qvel[self.slide_qvel_addr] = 0.0
            mujoco.mj_forward(self.model, self.data)

        # actual_base_s = where the scope physically ended up along the
        # centreline. With the slide joint, slide_qpos is the displacement
        # of scope_anchor along target's local +Y (centreline tangent) in
        # metres. So:
        #     actual_base_s = target_base_s + slide_qpos
        # slide_qpos is 0 when chain follows target, negative when walls
        # hold the chain back, slightly positive when chain overshoots.
        self.actual_base_s = self.target_base_s + slide_qpos
        ds_actual = self.actual_base_s - prev_actual_s

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
                if _c.geom1 in self._tip_geom_ids or _c.geom2 in self._tip_geom_ids:
                    _tip_in_contact = True
        if _tip_in_contact:
            self._tip_contact_steps += 1
        else:
            self._tip_contact_steps = 0
        self.force_norm = float(np.clip(raw_force / MAX_INSERTION_FORCE, 0.0, 1.0))

        # ---- reward ----
        # r = W_PROGRESS * new_territory * (0.3 + 0.7*(1 - force_norm))
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
        #     steering to earn reward. Softened to (0.3 + 0.7*(1-f)) to keep a
        #     gradient floor while still rewarding low-contact advances more.
        prev_max_s = self.max_actual_base_s
        new_territory = max(0.0, self.actual_base_s - self.max_actual_base_s)
        self.max_actual_base_s = max(self.max_actual_base_s, self.actual_base_s)
        self.recent_deltas.append(self.max_actual_base_s - prev_max_s)
        _force_mult = 0.3 + 0.7 * (1.0 - self.force_norm)
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
        if self.actual_base_s >= self.s_goal - GOAL_TOLERANCE_M:
            termination_reason = "goal"
        # Only goal/ejected are true terminations. Stuck, tip-stuck, and max
        # steps are truncations. Policy must learn to minimise wall contact via
        # the (1-force_norm) reward multiplier and tip-contact penalty.
        elif self.actual_base_s < self.ejected_threshold:
            termination_reason = "ejected"

        r_goal = W_GOAL if termination_reason == "goal" else 0.0
        r_stuck = -W_STUCK_PENALTY if (
            termination_reason is None and (_stuck or _tip_stuck)
        ) else 0.0
        reward = float(r_progress + r_shaping + r_tip + r_goal + r_stuck)
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
        weld_lag = self.target_base_s - self.actual_base_s
        raw_progress = (self.actual_base_s - self.base_s_init) / self.s_span
        raw_goal_gap_m = max(0.0, self.s_goal - self.actual_base_s)
        raw_max_seen_goal_gap_m = max(0.0, self.s_goal - self.max_actual_base_s)
        # The success threshold is intentionally before the exact admin goal
        # line. For reporting, a goal termination is a completed task, so show
        # it as 100% complete even if the raw base position is inside the
        # tolerance band rather than exactly at s_goal.
        progress = 1.0 if termination_reason == "goal" else raw_progress
        goal_gap_m = 0.0 if termination_reason == "goal" else raw_goal_gap_m
        max_seen_goal_gap_m = (
            0.0 if termination_reason == "goal" else raw_max_seen_goal_gap_m
        )
        info = {
            "target_base_s": self.target_base_s,
            "actual_base_s": self.actual_base_s,
            "max_seen_s": self.max_actual_base_s,
            "goal_s": self.s_goal,
            "target_s_max": self.target_s_max,
            "goal_tolerance_m": GOAL_TOLERANCE_M,
            "raw_progress": raw_progress,
            "raw_goal_gap_m": raw_goal_gap_m,
            "raw_max_seen_goal_gap_m": raw_max_seen_goal_gap_m,
            "goal_gap_m": goal_gap_m,
            "max_seen_goal_gap_m": max_seen_goal_gap_m,
            "weld_lag": float(weld_lag),
            "progress": progress,
            "ncon": int(self.data.ncon),
            "force_norm": self.force_norm,
            "force_steps": self._force_steps,
            "raw_force_n": raw_force,
            "tip_contact_steps": self._tip_contact_steps,
            "stuck_window_delta_m": float(sum(self.recent_deltas)),
            "r_progress": float(r_progress),
            "r_shaping": float(r_shaping),
            "r_tip": float(r_tip),
            "r_goal": float(r_goal),
            "r_stuck": float(r_stuck),
            "termination_reason": termination_reason,
            "truncation_reason": truncation_reason,
        }
        if terminated or truncated:
            # SF reads episode_extra_stats at episode end and merges the keys
            # into policy_avg_stats, making them visible to training observers.
            info["episode_extra_stats"] = {
                "progress":  float(progress),
                "goal":      1.0 if termination_reason == "goal"    else 0.0,
                "ejected":   1.0 if termination_reason == "ejected" else 0.0,
                "stuck":     1.0 if _stuck                          else 0.0,
                "tip_stuck": 1.0 if _tip_stuck                      else 0.0,
                "truncated": 1.0 if truncated                       else 0.0,
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
            state : (10,) float32:
                    [cmd_x_n, cmd_y_n, progress_placeholder, force_norm,
                     net_fx, net_fz, last_action[0..2], tip_contact]
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

        cmd_x_n = self.cmd_x_pair / self.max_pull
        cmd_y_n = self.cmd_y_pair / self.max_pull
        # Tendon force differentials: net force in scope X and Z bending axes.
        # actuator order: px(0), pz(1), nx(2), nz(3) — matches scope_gen.TENDONS.
        # Positive net_fx means resistance from the +X wall; steer -X to escape.
        # Directly observable from motor currents on real hardware.
        f = self.data.actuator_force
        net_fx = float(np.clip(
            (f[0] - f[2]) / scope_gen.ACTUATOR_PULL_FORCE_LIMIT, -1.0, 1.0
        ))
        net_fz = float(np.clip(
            (f[1] - f[3]) / scope_gen.ACTUATOR_PULL_FORCE_LIMIT, -1.0, 1.0
        ))
        # 10D state. The third slot is deliberately a constant placeholder:
        # true colon progress is privileged sim information and is not available
        # at deployment. Progress is still reported in info/reward diagnostics.
        state = np.array([
            cmd_x_n, cmd_y_n, OBS_PROGRESS_PLACEHOLDER, self.force_norm,
            net_fx, net_fz,
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

    def _pose_for_base_s_np(
        self, s_base: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """pose_for_base_s wrapper that returns numpy arrays (pos, quat)."""
        pos_tup, quat_str = pose_for_base_s(self.centreline, s_base)
        pos = np.array(pos_tup, dtype=np.float64)
        quat = np.array([float(v) for v in quat_str.split()], dtype=np.float64)
        return pos, quat

    def _set_target_mocap(self, pos: np.ndarray, quat: np.ndarray) -> None:
        """Write a target pose to the target mocap body."""
        self.data.mocap_pos[self.target_mocap_id, 0] = pos[0]
        self.data.mocap_pos[self.target_mocap_id, 1] = pos[1]
        self.data.mocap_pos[self.target_mocap_id, 2] = pos[2]
        self.data.mocap_quat[self.target_mocap_id, 0] = quat[0]
        self.data.mocap_quat[self.target_mocap_id, 1] = quat[1]
        self.data.mocap_quat[self.target_mocap_id, 2] = quat[2]
        self.data.mocap_quat[self.target_mocap_id, 3] = quat[3]

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
