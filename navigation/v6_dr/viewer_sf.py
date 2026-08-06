"""
Live viewer for a trained v6 domain-randomised checkpoint.

Loads the best checkpoint from runs_sf/v6_dr_700V3/checkpoint_p0/, opens the MuJoCo
passive viewer + cv2 depth windows, and runs the policy on the chosen seed.

Usage:
    python viewer_sf.py
    python viewer_sf.py --seed 5
    python viewer_sf.py --checkpoint runs_sf/v6_dr_700V3/checkpoint_p0
    python viewer_sf.py --seed 17 --speed 4 --no_depth
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "glfw")

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
from scope_colon_env import ScopeColonEnv


SEED               = 0
CHECKPOINT_PATH    = None
SHOW_DEPTH         = True
DEPTH_DISPLAY_SIZE = 256
DEPTH_RES          = 64
EXPERIMENT         = "v6_dr_700V3"
TRAIN_DIR          = str(HERE / "runs_sf")
DEFAULT_CHECKPOINT_DIR = Path(TRAIN_DIR) / EXPERIMENT / "checkpoint_p0"

_SF_DEFAULTS = [
    "--algo=APPO", "--env=scope_colon",
    f"--experiment={EXPERIMENT}", f"--train_dir={TRAIN_DIR}",
    "--num_workers=1", "--num_envs_per_worker=1",
    "--rollout=32", "--batch_size=2048",
    "--normalize_input=False", "--use_rnn=True",
    "--policy_workers_per_policy=1", "--worker_num_splits=1",
]

_SEED = SEED


def _make_env(full_env_name, cfg=None, env_config=None, render_mode=None):
    return ScopeColonEnv(seed=_SEED, depth_res=DEPTH_RES)


def _find_best_checkpoint(checkpoint_dir: Path) -> Path:
    best = sorted(checkpoint_dir.glob("best_*.pth"), key=lambda p: p.stat().st_mtime)
    if not best:
        raise FileNotFoundError(f"No best_*.pth checkpoint in {checkpoint_dir}")
    return best[-1]


def _resolve_best_checkpoint(path_arg: str | None) -> Path:
    path = Path(path_arg) if path_arg else DEFAULT_CHECKPOINT_DIR
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
    depth_big = cv2.resize(
        depth_img,
        (DEPTH_DISPLAY_SIZE, DEPTH_DISPLAY_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imshow(name, depth_big)


def _neutralize_dr(env: ScopeColonEnv) -> None:
    """Viewer-only diagnostic: force nominal command response for this episode."""
    env._dr_gain_x = 1.0
    env._dr_gain_y = 1.0
    env._dr_dead_x = 0.0
    env._dr_dead_y = 0.0
    env._dr_lag = 0
    env._dr_adv_gain = 1.0
    env._cmd_buf.clear()
    for _ in range(3):
        env._cmd_buf.append((0.0, 0.0))


def main() -> None:
    global _SEED, CHECKPOINT_PATH

    parser = argparse.ArgumentParser(description="v6 domain-randomised viewer")
    parser.add_argument("--seed",       type=int, default=SEED)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="checkpoint_p0 folder; viewer loads the newest best_*.pth inside it",
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
    args = parser.parse_args()
    _SEED = args.seed
    if args.checkpoint:
        CHECKPOINT_PATH = args.checkpoint

    torch.backends.cudnn.enabled = False
    sf_arguments.get_git_commit_hash = lambda: ("unknown", "not a git repository")
    sf_runner_module.save_git_diff = lambda _: None

    register_env("scope_colon", _make_env)
    register_scope_encoder()

    sf_parser, _ = parse_sf_args(argv=_SF_DEFAULTS)
    cfg           = parse_full_cfg(sf_parser, argv=_SF_DEFAULTS)

    ckpt_path = _resolve_best_checkpoint(CHECKPOINT_PATH)
    print(f"Checkpoint : {ckpt_path}")
    print(f"Seed       : {_SEED}")

    _tmp         = ScopeColonEnv(seed=0, depth_res=DEPTH_RES)
    obs_space    = _tmp.observation_space
    action_space = _tmp.action_space
    _tmp.close()

    actor_critic = create_actor_critic(cfg, obs_space, action_space)
    actor_critic.eval()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    actor_critic.load_state_dict(ckpt["model"])
    env_steps = ckpt.get("env_steps", "?")
    print(f"Loaded weights  (env_steps={env_steps:,})\n" if isinstance(env_steps, int)
          else f"Loaded weights  (env_steps={env_steps})\n")

    env = ScopeColonEnv(seed=_SEED, max_steps=20_000, depth_res=DEPTH_RES)
    print(f"Total length : {env.total_length * 1000:.0f} mm  Goal at : {env.s_goal * 1000:.0f} mm")

    _rnn_size  = getattr(cfg, "rnn_size", 512) if cfg.use_rnn else 1
    rnn_states = torch.zeros([1, _rnn_size])

    show_depth = SHOW_DEPTH and not args.no_depth
    speed = max(1e-6, float(args.speed))

    if show_depth:
        cv2.namedWindow("policy depth 64x64 (noisy obs)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("policy depth 64x64 (noisy obs)", DEPTH_DISPLAY_SIZE, DEPTH_DISPLAY_SIZE)
        cv2.namedWindow("clean tip depth 64x64 (no noise)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("clean tip depth 64x64 (no noise)", DEPTH_DISPLAY_SIZE, DEPTH_DISPLAY_SIZE)
        cv2.namedWindow("abs depth error x4", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("abs depth error x4", DEPTH_DISPLAY_SIZE, DEPTH_DISPLAY_SIZE)

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
            rnn_states = policy_out.get("new_rnn_states", rnn_states)
            action     = policy_out["actions"].squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_steps  += 1
            ep_reward += float(reward)

            if show_depth:
                policy_depth = obs["depth"][0]
                clean_depth = _render_clean_depth(env)
                err_depth = np.clip(np.abs(policy_depth - clean_depth) * 4.0, 0.0, 1.0)
                _show_depth_window("policy depth 64x64 (noisy obs)", policy_depth)
                _show_depth_window("clean tip depth 64x64 (no noise)", clean_depth)
                _show_depth_window("abs depth error x4", err_depth)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            viewer.sync()

            if time.time() - last_print > 1.0:
                dead_x_mm = getattr(env, "_dr_dead_x", 0.0) * env.max_pull * 1000.0
                dead_y_mm = getattr(env, "_dr_dead_y", 0.0) * env.max_pull * 1000.0
                cmd_x_mm = env.cmd_x_pair * 1000.0
                cmd_y_mm = env.cmd_y_pair * 1000.0
                ctrl_x_mm = float(env.data.ctrl[0]) * 1000.0
                ctrl_y_mm = float(env.data.ctrl[1]) * 1000.0
                goal_gap_mm = info.get("goal_gap_m", 0.0) * 1000.0
                max_seen_gap_mm = info.get("max_seen_goal_gap_m", 0.0) * 1000.0
                stuck_delta_mm = info.get("stuck_window_delta_m", 0.0) * 1000.0
                print(
                    f"ep {ep}  step {ep_steps:4d}  "
                    f"s={info.get('actual_base_s', 0)*1000:6.1f}mm  "
                    f"progress={info.get('progress',0):.3f}  "
                    f"gap={goal_gap_mm:5.1f}mm  "
                    f"maxgap={max_seen_gap_mm:5.1f}mm  "
                    f"force={info.get('force_norm',0):.2f}  "
                    f"weld={info.get('weld_lag',0)*1000:5.1f}mm  "
                    f"ncon={info.get('ncon',0):3d}  "
                    f"tip={info.get('tip_contact_steps',0):2d}  "
                    f"stuckΔ={stuck_delta_mm:4.1f}mm  "
                    f"a=({action[0]:+.2f},{action[1]:+.2f},{action[2]:+.2f})  "
                    f"cmd=({cmd_x_mm:+.1f},{cmd_y_mm:+.1f})mm  "
                    f"ctrl=({ctrl_x_mm:+.1f},{ctrl_y_mm:+.1f})mm  "
                    f"dead=({dead_x_mm:.1f},{dead_y_mm:.1f})mm  "
                    f"lag={getattr(env, '_dr_lag', 0)}  "
                    f"adv={getattr(env, '_dr_adv_gain', 1.0):.2f}  "
                    f"reward={ep_reward:+.4f}"
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
