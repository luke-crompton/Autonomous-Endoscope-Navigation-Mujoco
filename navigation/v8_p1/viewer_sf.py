"""
Live viewer for a trained v8_p1 checkpoint.

Loads the best checkpoint from runs_sf/v8_p1_v1/checkpoint_p0/, opens the MuJoCo
passive viewer + cv2 depth windows, and runs the policy on the chosen seed.

Actions are DETERMINISTIC by default (the policy Gaussian's mean), i.e. what you
would deploy. Pass --stochastic for the sampled actions PPO trained under.
Domain randomisation and the depth error map stay on either way -- both are real
on the rig -- so the plain `viewer_sf.py --seed N` is the deployment preview.

Usage:
    python viewer_sf.py
    python viewer_sf.py --seed 5
    python viewer_sf.py --experiment v8_p1_v1
    python viewer_sf.py --checkpoint runs_sf/v8_p1_v1/checkpoint_p0
    python viewer_sf.py --seed 17 --speed 4 --no_depth

    # Hand-feed rehearsal: insertion at 3 mm/s (1/10th the trained 30 mm/s).
    # --feed_scale changes what the POLICY experiences; --speed changes only how
    # fast YOU watch. They are independent.
    #   A full traverse is now ~25,000 steps, i.e. ~17 min at real-time pacing.
    #   --speed 10 drops the sleep target to 4 ms/step, which this loop cannot
    #   hit, so it simply runs at max render throughput -- the real wall-clock is
    #   then set by rendering, not by --speed. Add --no_depth to roughly halve it:
    #   the cv2 path renders the tip camera a SECOND time for the clean/error
    #   panels (_render_clean_depth), on top of the env's own render.
    python viewer_sf.py --seed 0 --feed_scale 0.1 --speed 10 --no_depth
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")  # WSL2 GPU/OpenGL passthrough is
                                                      # flaky; force Mesa's CPU software
                                                      # rasterizer (llvmpipe) instead

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import cv2
import mujoco
import mujoco.viewer

if sys.platform == "win32":
    from unittest.mock import MagicMock
    _pwd = MagicMock()
    _pwd.getpwuid.return_value = MagicMock(pw_name="user")
    sys.modules["pwd"] = _pwd

import sample_factory.cfg.arguments as sf_arguments
import sample_factory.algo.runners.runner as sf_runner_module
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.model.actor_critic import create_actor_critic

from sf_encoder import register_scope_encoder
# The module itself, not just names off it: --feed_scale rebinds three of its
# module-level constants before the env is constructed (see _apply_feed_scale).
import scope_colon_env as _env_mod
from scope_colon_env import (
    DEFAULT_ADVANCE_RATE_MAX,
    DEFAULT_DEPTH_RES,
    DEFAULT_MAX_STEPS,
    STUCK_THRESHOLD,
    STUCK_WINDOW,
    TIP_STUCK_WINDOW,
    W_TIP_PENALTY,
    ScopeColonEnv,
    ROLLER_CTRL_SLICE,
)


SEED               = 0
SHOW_DEPTH         = True
DEPTH_DISPLAY_SIZE = 256
DEPTH_RES          = DEFAULT_DEPTH_RES   # (H, W) = (54, 96), 16:9 -- tracks scope_colon_env.py
EXPERIMENT         = "v8_p1_v1"
TRAIN_DIR          = str(HERE / "runs_sf")


def _checkpoint_dir(experiment: str) -> Path:
    return Path(TRAIN_DIR) / experiment / "checkpoint_p0"


def _sf_defaults(experiment: str) -> list[str]:
    return [
        "--algo=APPO", "--env=scope_colon",
        f"--experiment={experiment}", f"--train_dir={TRAIN_DIR}",
        "--num_workers=1", "--num_envs_per_worker=1",
        "--rollout=32", "--batch_size=2048",
        "--normalize_input=False", "--use_rnn=True",
        "--policy_workers_per_policy=1", "--worker_num_splits=1",
        "--device=cpu",  # WSL2/CUDA was flaky for viewing; CPU-only is plenty for one env
    ]


_SEED = SEED


def _make_env(full_env_name, cfg=None, env_config=None, render_mode=None):
    return ScopeColonEnv(seed=_SEED, depth_res=DEPTH_RES)


def _find_best_checkpoint(checkpoint_dir: Path) -> Path:
    best = sorted(checkpoint_dir.glob("best_*.pth"), key=lambda p: p.stat().st_mtime)
    if not best:
        raise FileNotFoundError(f"No best_*.pth checkpoint in {checkpoint_dir}")
    return best[-1]


def _resolve_best_checkpoint(path_arg: str | None, experiment: str) -> Path:
    path = Path(path_arg) if path_arg else _checkpoint_dir(experiment)
    if path.is_dir():
        return _find_best_checkpoint(path)
    if path.is_file() and path.name.startswith("best_"):
        return path
    if path.is_file():
        raise ValueError(f"Viewer only loads best_*.pth checkpoints, got: {path}")
    raise FileNotFoundError(f"Checkpoint folder does not exist: {path}")


def _obs_to_tensors(obs: dict) -> dict:
    return {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in obs.items()}


def _depth_to_u8(depth: np.ndarray) -> np.ndarray:
    """Convert a normalised 2D depth map in [0, 1] to uint8 for display."""
    return (depth * 255.0).clip(0, 255).astype(np.uint8)


def _render_clean_depth(env: ScopeColonEnv) -> np.ndarray:
    """Render noiseless tip-cam depth and normalise exactly like env._observation()."""
    env.renderer.update_scene(env.data, camera=env.tip_cam_id)
    depth_raw = env.renderer.render()
    max_d = max(float(depth_raw.max()), 1e-6)
    return np.clip(depth_raw / max_d, 0.0, 1.0).astype(np.float32)


def _show_depth_window(name: str, depth: np.ndarray) -> None:
    depth_img = _depth_to_u8(depth)
    # Preserve the obs's aspect: it is 16:9 (54x96) since 2026-08-04, and
    # forcing it square here would misrepresent the camera geometry in the one
    # tool used to eyeball it. Height pinned, width follows.
    _h, _w = depth_img.shape[:2]
    depth_big = cv2.resize(
        depth_img,
        (max(1, int(round(DEPTH_DISPLAY_SIZE * _w / _h))), DEPTH_DISPLAY_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imshow(name, depth_big)


def _classify_geoms(model: mujoco.MjModel) -> dict[int, str]:
    """Map geom_id -> category for the contact-force breakdown below.

    Categories: 'roller', 'shaft', 'disk_N' (per-disk, not just the last
    one), 'wall' (the ~12k thin-prism collision geoms, all children of the
    single 'colon_wall' body), or 'other:<bodyname>' as a catch-all.

    Classifies by BODY name, not geom name -- mj_id2name on individual geoms
    returns None once a scene has this many geoms (a known MuJoCo
    naming-table limit, not specific to this classifier), but body names
    stay reliable since there are only ~100 bodies.
    """
    cats: dict[int, str] = {}
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if bname == "colon_wall":
            cats[gid] = "wall"
        elif bname.startswith("roller_"):
            cats[gid] = "roller"
        elif bname.startswith("shaft_link_") or bname.startswith("shaft_uni_"):
            cats[gid] = "shaft"
        elif bname.startswith("disk_"):
            cats[gid] = bname
        else:
            cats[gid] = f"other:{bname}"
    return cats


def _contact_force_breakdown(env: ScopeColonEnv, geom_cat: dict[int, str]) -> dict:
    """Sum contact-force magnitude per (category_a, category_b) pair for the
    CURRENT physics state, plus a per-disk-vs-wall breakdown so a single
    jammed disk mid-chain shows up instead of being averaged away.

    Returns {'roller_shaft': N, 'shaft_wall': N, 'disk_wall_total': N,
             'disk_wall_by_disk': {disk_name: N, ...}, 'other_total': N}.
    """
    cbuf = np.zeros(6)
    roller_shaft = 0.0
    shaft_wall = 0.0
    disk_wall_by_disk: dict[str, float] = {}
    other_total = 0.0
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        cat1 = geom_cat.get(c.geom1, "other:?")
        cat2 = geom_cat.get(c.geom2, "other:?")
        mujoco.mj_contactForce(env.model, env.data, i, cbuf)
        mag = float(np.linalg.norm(cbuf[:3]))
        pair = {cat1, cat2}
        if pair == {"roller", "shaft"}:
            roller_shaft += mag
        elif pair == {"shaft", "wall"}:
            shaft_wall += mag
        elif "wall" in pair and any(c.startswith("disk_") for c in pair):
            disk_name = cat1 if cat1.startswith("disk_") else cat2
            disk_wall_by_disk[disk_name] = disk_wall_by_disk.get(disk_name, 0.0) + mag
        else:
            other_total += mag
    return {
        "roller_shaft": roller_shaft,
        "shaft_wall": shaft_wall,
        "disk_wall_total": float(sum(disk_wall_by_disk.values())),
        "disk_wall_by_disk": disk_wall_by_disk,
        "other_total": other_total,
    }


def _apply_feed_scale(feed_scale: float) -> dict:
    """Slow (or speed) the INSERTION FEED by `feed_scale`, holding every guard's
    physical DISTANCE meaning fixed. Returns the resulting values for printing.

    WHY THE GUARDS MUST MOVE TOO. Three of the env's truncation guards are
    counted in STEPS, not millimetres, so cutting the per-step advance without
    touching them shrinks the physical distance each one permits by the same
    factor. At feed_scale=0.1, left alone:

      TIP_STUCK_WINDOW  38 steps x 1.2 mm = 45.6 mm  ->  4.6 mm of in-contact
                        travel, SHORTER THAN A BEND ARC. This is the 2026-07-13
                        bug exactly (window = 50 allowed 15 mm, less than the
                        in-contact arc of a bend); it made a healthy policy look
                        broken then and would do so again here.
      STUCK_WINDOW      75 steps x 1.2 mm = 90 mm expected against a 2 mm
                        threshold  ->  9 mm against 2 mm, margin effectively gone.
      max_steps         2500 steps = 100 s ~ the 305 mm traverse  ->  truncates
                        after ~30 mm, so no episode could ever finish.

    So the invariant held fixed here is DISTANCE, deliberately not wall-clock
    time. That is the opposite of Change F (2026-08-04, 100 Hz -> 25 Hz), where
    advance-per-SECOND was unchanged and time was the right thing to preserve.
    Here advance-per-second is the very thing being changed.

    STUCK_THRESHOLD is left alone -- it is already stated in metres.
    W_TIP_PENALTY is scaled the other way (x feed_scale) purely so the reward
    printed in the status line stays comparable across scales: it is a per-step
    penalty, so the same physical path in contact would otherwise accumulate
    1/feed_scale times more of it. Nothing here learns, so this is cosmetic.

    Patching module globals is enough because the env reads STUCK_WINDOW,
    TIP_STUCK_WINDOW and STUCK_THRESHOLD off the module at step time. But
    `recent_deltas` is built as deque(maxlen=STUCK_WINDOW) in __init__ and
    reset() only .clear()s it, so this MUST run before the env is constructed.
    """
    if feed_scale <= 0.0:
        raise ValueError(f"--feed_scale must be > 0, got {feed_scale}")

    scaled = {
        "advance_rate_max": DEFAULT_ADVANCE_RATE_MAX * feed_scale,
        "max_steps":        math.ceil(DEFAULT_MAX_STEPS / feed_scale),
        "stuck_window":     max(1, round(STUCK_WINDOW / feed_scale)),
        "tip_stuck_window": max(1, round(TIP_STUCK_WINDOW / feed_scale)),
        "w_tip_penalty":    W_TIP_PENALTY * feed_scale,
    }

    # feed_scale == 1.0 must be bit-identical to not passing the flag at all.
    if feed_scale == 1.0:
        assert scaled["advance_rate_max"] == DEFAULT_ADVANCE_RATE_MAX
        assert scaled["max_steps"]        == DEFAULT_MAX_STEPS
        assert scaled["stuck_window"]     == STUCK_WINDOW
        assert scaled["tip_stuck_window"] == TIP_STUCK_WINDOW
        assert scaled["w_tip_penalty"]    == W_TIP_PENALTY

    _env_mod.STUCK_WINDOW     = scaled["stuck_window"]
    _env_mod.TIP_STUCK_WINDOW = scaled["tip_stuck_window"]
    _env_mod.W_TIP_PENALTY    = scaled["w_tip_penalty"]
    return scaled


def _print_feed_scale(feed_scale: float, scaled: dict) -> None:
    """Always print what the guards actually became -- the scaling is a
    judgement call about intent, not a proven equivalence, so it must never be
    ambiguous which regime an observed run was watched under."""
    _mm_s = scaled["advance_rate_max"] * 1000.0 / (
        _env_mod.DEFAULT_PHYSICS_PER_STEP * 0.0005
    )
    print(f"Feed scale : {feed_scale:g}x  ->  advance "
          f"{scaled['advance_rate_max']*1000:.3f} mm/step = {_mm_s:.1f} mm/s")
    if feed_scale != 1.0:
        print(
            f"             guards rescaled to hold DISTANCE fixed:\n"
            f"               max_steps        {DEFAULT_MAX_STEPS:6d} -> {scaled['max_steps']:6d}"
            f"   (~305 mm traverse)\n"
            f"               STUCK_WINDOW     {STUCK_WINDOW:6d} -> {scaled['stuck_window']:6d}"
            f"   ({scaled['stuck_window']*scaled['advance_rate_max']*1000:.0f} mm vs "
            f"{STUCK_THRESHOLD*1000:.0f} mm threshold)\n"
            f"               TIP_STUCK_WINDOW {TIP_STUCK_WINDOW:6d} -> {scaled['tip_stuck_window']:6d}"
            f"   ({scaled['tip_stuck_window']*scaled['advance_rate_max']*1000:.1f} mm in-contact)\n"
            f"             STUCK_THRESHOLD unchanged ({STUCK_THRESHOLD*1000:.0f} mm -- already a distance).\n"
            f"             NOTE: this models a clean {1/feed_scale:g}x slower feed, NOT hand-feed jitter."
        )


def _neutralize_dr(env: ScopeColonEnv) -> None:
    """Viewer-only diagnostic: force nominal command response for this episode."""
    env._dr_gain_x = 1.0
    env._dr_gain_y = 1.0
    env._dr_dead_x = 0.0
    env._dr_dead_y = 0.0
    env._dr_lag = 0
    env._cmd_buf.clear()
    for _ in range(3):
        env._cmd_buf.append((0.0, 0.0))
    # Gain now lives on the model's actuator coefficients (set in reset()),
    # not just the Python-side attribute -- reset those too or this flag
    # would silently leave whatever gain reset() last sampled in place.
    for i in range(4):
        env.model.actuator_gainprm[i] = env._tendon_gainprm0[i]
        env.model.actuator_biasprm[i] = env._tendon_biasprm0[i]


def main() -> None:
    global _SEED

    parser = argparse.ArgumentParser(description="v8_p1 policy viewer")
    parser.add_argument("--seed",       type=int, default=SEED)
    parser.add_argument(
        "--experiment",
        type=str,
        default=EXPERIMENT,
        help=f"run folder under runs_sf/ (default: {EXPERIMENT})",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint_p0 folder; viewer loads the newest best_*.pth inside it. "
             "Defaults to the --experiment run's checkpoint_p0/",
    )
    parser.add_argument(
        "--disable_dr",
        action="store_true",
        help="viewer diagnostic only: force nominal command response after each reset",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="viewer playback speed multiplier; use 4 or 8 to inspect slow seeds faster",
    )
    parser.add_argument(
        "--no_realtime",
        action="store_true",
        help="do not sleep to maintain real time; run as fast as rendering allows",
    )
    parser.add_argument(
        "--no_depth",
        action="store_true",
        help="hide cv2 depth windows to reduce viewer overhead",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample actions from the policy's Gaussian instead of taking its mean. "
             "This is TRAINING mode -- PPO's exploration noise, redrawn every step. "
             "Default is deterministic (the mean), which is what deployment reads. "
             "Use this only to see the noise the policy was trained under.",
    )
    parser.add_argument(
        "--feed_scale",
        type=float,
        default=1.0,
        help="scale the INSERTION FEED rate (1.0 = trained 30 mm/s; 0.1 = 3 mm/s, "
             "roughly hand-fed / real-colonoscopy pace). Rescales max_steps and the "
             "stuck/tip-stuck windows alongside it so their physical DISTANCE meaning "
             "is preserved. Orthogonal to --speed, which only changes how fast you "
             "watch: try --feed_scale 0.1 --speed 10.",
    )
    args = parser.parse_args()
    _SEED = args.seed

    # Must happen before ANY ScopeColonEnv is constructed -- recent_deltas takes
    # its maxlen from STUCK_WINDOW in __init__ and reset() only clears it.
    _feed = _apply_feed_scale(args.feed_scale)

    torch.backends.cudnn.enabled = False
    sf_arguments.get_git_commit_hash = lambda: ("unknown", "not a git repository")
    sf_runner_module.save_git_diff = lambda _: None

    register_env("scope_colon", _make_env)
    register_scope_encoder()

    _sf_argv     = _sf_defaults(args.experiment)
    sf_parser, _ = parse_sf_args(argv=_sf_argv)
    cfg           = parse_full_cfg(sf_parser, argv=_sf_argv)

    ckpt_path = _resolve_best_checkpoint(args.checkpoint, args.experiment)
    print(f"Experiment : {args.experiment}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Seed       : {_SEED}")

    _tmp         = ScopeColonEnv(seed=0, depth_res=DEPTH_RES)
    obs_space    = _tmp.observation_space
    action_space = _tmp.action_space
    _tmp.close()

    actor_critic = create_actor_critic(cfg, obs_space, action_space)
    actor_critic.eval()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    actor_critic.load_state_dict(ckpt["model"])
    env_steps = ckpt.get("env_steps", "?")
    print(f"Loaded weights  (env_steps={env_steps:,})\n" if isinstance(env_steps, int)
          else f"Loaded weights  (env_steps={env_steps})\n")

    # max_steps tracks the env default (2500 = 100 s at 25 Hz) so the viewer runs
    # the policy under the same episode cap it was TRAINED under. This was a
    # hardcoded 20_000, which meant 2x the old 100 Hz default -- after Change F
    # (2026-08-04, 100 Hz -> 25 Hz) the same literal became 800 s, 8x the cap, so a
    # wedged episode would have run eight times longer here than in training before
    # truncating.
    env = ScopeColonEnv(
        seed=_SEED,
        max_steps=_feed["max_steps"],
        advance_rate_max=_feed["advance_rate_max"],
        depth_res=DEPTH_RES,
    )
    _print_feed_scale(args.feed_scale, _feed)
    print(
        "Actions    : "
        + ("STOCHASTIC — sampled from the policy Gaussian (TRAINING mode, "
           "exploration noise redrawn every step)"
           if args.stochastic else
           "deterministic (distribution mean) — deployment mode")
    )
    # Kept separate on purpose: --disable_dr neutralises the per-episode
    # gain/dead-zone/lag draw ONLY. The depth error map is injected in
    # _observation() and is not affected by any flag -- DA3 has real per-pixel
    # error on the rig, so hiding it would flatter the policy.
    print(
        f"             DR (gain/dead-zone/lag, drawn per episode): "
        f"{'DISABLED — nominal scope, NOT a deployment preview' if args.disable_dr else 'ACTIVE'}"
    )
    print("             depth error-map noise: ACTIVE (always — DA3 error is real)")
    print(
        f"Total length : {env.total_length * 1000:.0f} mm  "
        f"Progress-% reference : {env.s_goal * 1000:.0f} mm "
        f"(NOT a termination target -- episode ends when the shaft runs out, "
        f"i.e. shaft_exhausted; s_goal only sets the 'progress' percentage's denominator)"
    )
    geom_cat = _classify_geoms(env.model)

    _rnn_size  = getattr(cfg, "rnn_size", 512) if cfg.use_rnn else 1
    rnn_states = torch.zeros([1, _rnn_size])

    show_depth = SHOW_DEPTH and not args.no_depth
    speed = max(1e-6, float(args.speed))

    if show_depth:
        # Titles carry the live obs shape rather than a hardcoded "64x64", which
        # went stale when the obs became 16:9 on 2026-08-04.
        _dh, _dw = DEPTH_RES
        _dw_px = max(1, int(round(DEPTH_DISPLAY_SIZE * _dw / _dh)))
        WIN_POLICY = f"policy depth {_dh}x{_dw} (noisy obs)"
        WIN_CLEAN  = f"clean tip depth {_dh}x{_dw} (no noise)"
        WIN_ERR    = "abs depth error x4"
        for _win in (WIN_POLICY, WIN_CLEAN, WIN_ERR):
            cv2.namedWindow(_win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(_win, _dw_px, DEPTH_DISPLAY_SIZE)

    obs, _ = env.reset()
    if args.disable_dr:
        _neutralize_dr(env)
        obs = env._observation()
        print("Viewer diagnostic: domain randomisation disabled for this run.")
    ep = ep_steps = 0
    ep_reward  = 0.0
    target_dt  = env.physics_per_step * env.model.opt.timestep
    last_print = time.time()

    print(
        "MuJoCo viewer open. Close the window"
        + (" (or press q in depth window)" if show_depth else "")
        + " to quit.\n"
    )

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()

            obs_t = _obs_to_tensors(obs)
            with torch.no_grad():
                policy_out = actor_critic(obs_t, rnn_states)
                # For a continuous action space the network emits Gaussian
                # PARAMETERS, not an action: action_parameterization() returns
                # (means, log_std). SF's forward() defaults to
                # forward_tail(sample_actions=True), so policy_out["actions"] is a
                # fresh draw mu + sigma*eps -- PPO's exploration, redrawn EVERY step,
                # and it needs that sample's log_prob to form its gradient.
                # Nothing is learning here, so the default is the mean: deployment
                # reads mu, and the same observation must give the same command.
                # Measured on best_000000940 (2026-08-04): sigma ~ 0.627 on a +-1
                # action space, so the sampled action is dominated by noise -- mean
                # |a_xy| 0.034 deterministic vs 0.468 sampled, and per-step tip
                # command jitter drops 95%. DR (gain/dead-zone/lag, drawn once per
                # episode in reset()) and the depth error map are NOT affected by
                # this flag and stay on -- both are real on the rig.
                _act_t = (
                    policy_out["actions"] if args.stochastic
                    else actor_critic.action_distribution().means
                )
            rnn_states = policy_out.get("new_rnn_states", rnn_states)
            action     = _act_t.squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_steps  += 1
            ep_reward += float(reward)

            if show_depth:
                policy_depth = obs["depth"][0]
                clean_depth = _render_clean_depth(env)
                err_depth = np.clip(np.abs(policy_depth - clean_depth) * 4.0, 0.0, 1.0)
                _show_depth_window(WIN_POLICY, policy_depth)
                _show_depth_window(WIN_CLEAN, clean_depth)
                _show_depth_window(WIN_ERR, err_depth)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            viewer.sync()

            if time.time() - last_print > 1.0:
                dead_x_mm = getattr(env, "_dr_dead_x", 0.0) * env.max_pull_x * 1000.0
                dead_y_mm = getattr(env, "_dr_dead_y", 0.0) * env.max_pull_y * 1000.0
                cmd_x_mm = env.cmd_x_pair * 1000.0
                cmd_y_mm = env.cmd_y_pair * 1000.0
                ctrl_x_mm = float(env.data.ctrl[0]) * 1000.0
                ctrl_y_mm = float(env.data.ctrl[1]) * 1000.0
                goal_gap_mm = info.get("goal_gap_m", 0.0) * 1000.0
                max_seen_gap_mm = info.get("max_seen_goal_gap_m", 0.0) * 1000.0
                stuck_delta_mm = info.get("stuck_window_delta_m", 0.0) * 1000.0
                roller_force = np.abs(env.data.actuator_force[ROLLER_CTRL_SLICE])
                cforce = _contact_force_breakdown(env, geom_cat)
                actual_s_mm = info.get('actual_base_s', 0) * 1000.0
                target_ins_mm = info.get("target_insertion_depth", 0.0) * 1000.0
                slip_mm = target_ins_mm - actual_s_mm
                print(
                    f"ep {ep}  step {ep_steps:4d}  "
                    f"s={actual_s_mm:6.1f}mm  "
                    f"target={target_ins_mm:6.1f}mm  "
                    f"slip={slip_mm:6.1f}mm  "
                    f"progress={info.get('progress',0):.3f}  "
                    f"gap={goal_gap_mm:5.1f}mm  "
                    f"maxgap={max_seen_gap_mm:5.1f}mm  "
                    f"force={info.get('force_norm',0):.2f}  "
                    f"ncon={info.get('ncon',0):3d}  "
                    f"tip={info.get('tip_contact_steps',0):2d}  "
                    f"stuckΔ={stuck_delta_mm:4.1f}mm  "
                    f"a=({action[0]:+.2f},{action[1]:+.2f},{action[2]:+.2f})  "
                    f"cmd=({cmd_x_mm:+.1f},{cmd_y_mm:+.1f})mm  "
                    f"ctrl=({ctrl_x_mm:+.1f},{ctrl_y_mm:+.1f})mm  "
                    f"dead=({dead_x_mm:.1f},{dead_y_mm:.1f})mm  "
                    f"lag={getattr(env, '_dr_lag', 0)}  "
                    f"roller_force=[mean={np.mean(roller_force):.2f}N max={np.max(roller_force):.2f}N]  "
                    f"reward={ep_reward:+.4f}"
                )
                _top_disk = max(
                    cforce["disk_wall_by_disk"].items(), key=lambda kv: kv[1], default=(None, 0.0)
                )
                print(
                    f"    contact_force[N]: roller_shaft={cforce['roller_shaft']:.2f}  "
                    f"shaft_wall={cforce['shaft_wall']:.2f}  "
                    f"disk_wall_total={cforce['disk_wall_total']:.2f}  "
                    f"top_disk={_top_disk[0]}({_top_disk[1]:.2f}N)  "
                    f"other={cforce['other_total']:.2f}"
                )
                last_print = time.time()

            if terminated or truncated:
                reason = info.get("termination_reason") or info.get("truncation_reason")
                print(
                    f"── ep {ep}: reason={reason!r}  "
                    f"progress={info.get('progress',0)*100:.1f}%  "
                    f"gap={info.get('goal_gap_m',0)*1000:.1f}mm  "
                    f"maxgap={info.get('max_seen_goal_gap_m',0)*1000:.1f}mm  "
                    f"steps={ep_steps}  reward={ep_reward:+.4f}"
                )
                ep        += 1
                ep_steps   = 0
                ep_reward  = 0.0
                rnn_states = torch.zeros([1, _rnn_size])
                obs, _ = env.reset()
                if args.disable_dr:
                    _neutralize_dr(env)
                    obs = env._observation()

            if not args.no_realtime:
                elapsed = time.perf_counter() - t0
                target_sleep_dt = target_dt / speed
                if elapsed < target_sleep_dt:
                    time.sleep(target_sleep_dt - elapsed)

    env.close()
    if show_depth:
        cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
