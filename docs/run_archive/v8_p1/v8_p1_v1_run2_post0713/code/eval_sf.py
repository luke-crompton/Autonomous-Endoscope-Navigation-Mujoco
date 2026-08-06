"""
Evaluate a Sample Factory v8_p1 checkpoint on each training seed.

Defaults to the v8_p1_v1 run's best checkpoint.

Usage:
    python eval_sf.py
    python eval_sf.py --experiment v8_p1_v1
    python eval_sf.py --checkpoint runs_sf/v8_p1_v1/checkpoint_p0
    python eval_sf.py --checkpoint runs_sf/v8_p1_v1/checkpoint_p0/checkpoint_XXX.pth
    python eval_sf.py --episodes 5 --seeds 80
    python eval_sf.py --episodes 5 --seeds 80 --workers 32

    # WSL2: CPU-only rendering (egl is unstable AND ~7x slower there)
    python eval_sf.py --gl osmesa --episodes 3 --seeds 80 --workers 24
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Match the working shell prefix:
# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

GL_BACKENDS = ("egl", "osmesa", "glfw")
DEFAULT_GL  = "egl"


def _select_gl_backend() -> str:
    """Bind MUJOCO_GL before anything imports mujoco.

    MuJoCo picks its render backend at import time, so this has to run before
    the `scope_colon_env` import below -- setting MUJOCO_GL afterwards has no
    effect. Precedence is CLI flag > existing env var > DEFAULT_GL. Worker
    processes inherit os.environ, so the choice propagates to the pool without
    being threaded through initargs.

    osmesa is CPU-only. Under WSL2 it is both the *stable* and the *faster*
    choice (~24 ms/env-step vs ~179 ms under egl, whose GPU paravirtualization
    also throws EGLError on context teardown). On the native rig, egl wins.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--gl", choices=GL_BACKENDS, default=None)
    chosen = pre.parse_known_args()[0].gl or os.environ.get("MUJOCO_GL") or DEFAULT_GL
    os.environ["MUJOCO_GL"] = chosen
    return chosen


GL_BACKEND = _select_gl_backend()

import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from scope_colon_env import DEFAULT_DEPTH_RES, DEFAULT_MAX_STEPS, ScopeColonEnv


N_EPISODES = 5
N_SEEDS    = 80
DEFAULT_WORKERS = 24
MAX_STEPS  = DEFAULT_MAX_STEPS
DEPTH_RES  = DEFAULT_DEPTH_RES
EXPERIMENT = "v8_p1_v1"
TRAIN_DIR  = str(HERE / "runs_sf")


def _checkpoint_dir(experiment: str) -> Path:
    return Path(TRAIN_DIR) / experiment / "checkpoint_p0"


def _find_latest_checkpoint(train_dir: str, experiment: str) -> Path:
    ckpt_dir = Path(train_dir) / experiment / "checkpoint_p0"
    checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pth"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    return checkpoints[-1]


def _resolve_checkpoint(path_arg: str | None, experiment: str) -> Path:
    """Return a concrete .pth checkpoint path.

    If the user passes a checkpoint directory, prefer the newest best_*.pth.
    That matches the common eval case while still falling back to regular
    checkpoint_*.pth files when no best checkpoint exists.
    """
    path = Path(path_arg) if path_arg else _checkpoint_dir(experiment)
    if path.is_file():
        return path
    if path.is_dir():
        best = sorted(path.glob("best_*.pth"), key=lambda p: p.stat().st_mtime)
        if best:
            return best[-1]
        checkpoints = sorted(path.glob("checkpoint_*.pth"), key=lambda p: p.stat().st_mtime)
        if checkpoints:
            return checkpoints[-1]
        raise FileNotFoundError(f"No .pth checkpoints in {path}")
    raise FileNotFoundError(f"Checkpoint path does not exist: {path}")


def _checkpoint_run_dir(ckpt_path: Path) -> Path:
    """Return runs_sf/<experiment>/ for a checkpoint inside checkpoint_p0/."""
    if ckpt_path.parent.name == "checkpoint_p0":
        return ckpt_path.parent.parent
    return ckpt_path.parent


def _default_csv_path(ckpt_path: Path, n_episodes: int, n_seeds: int) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return (
        _checkpoint_run_dir(ckpt_path)
        / f"eval_{ckpt_path.stem}_eps{n_episodes}_seeds{n_seeds}_{stamp}.csv"
    )


def _list_checkpoints(experiment: str) -> list[Path]:
    ckpt_dir = _checkpoint_dir(experiment)
    return sorted(
        [*ckpt_dir.glob("best_*.pth"), *ckpt_dir.glob("checkpoint_*.pth")],
        key=lambda p: p.stat().st_mtime,
    )


def _obs_to_tensors(obs: dict) -> dict:
    return {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in obs.items()}


def run_episode(env: ScopeColonEnv, actor_critic, cfg) -> dict:
    _rnn_size  = getattr(cfg, "rnn_size", 512) if cfg.use_rnn else 1
    rnn_states = torch.zeros([1, _rnn_size])
    obs, _     = env.reset()
    sum_reward = 0.0
    sum_ncon   = 0
    info: dict = {}
    max_steps = int(getattr(env, "max_steps", MAX_STEPS))
    for step in range(max_steps):
        obs_t = _obs_to_tensors(obs)
        with torch.no_grad():
            policy_out = actor_critic(obs_t, rnn_states)
        rnn_states = policy_out.get("new_rnn_states", rnn_states)
        action     = policy_out["actions"].squeeze(0).cpu().numpy()
        obs, r, term, trunc, info = env.step(action)
        sum_reward += r
        sum_ncon   += info.get("ncon", 0)
        if term or trunc:
            break
    steps = step + 1
    extra = info.get("episode_extra_stats", {}) if isinstance(info, dict) else {}
    if term:
        reason = info.get("termination_reason") or "terminated"
    elif trunc:
        if info.get("truncation_reason"):
            reason = str(info["truncation_reason"])
        elif extra.get("stuck", 0.0) > 0.5:
            reason = "stuck"
        elif extra.get("tip_stuck", 0.0) > 0.5:
            reason = "tip_stuck"
        elif steps >= max_steps:
            reason = "max_steps"
        else:
            reason = "truncated"
    else:
        reason = "unknown"

    actual_s = float(info.get("actual_base_s", 0.0))
    max_seen_s = float(info.get("max_seen_s", actual_s))
    goal_s = float(getattr(env, "s_goal", actual_s))
    # "shaft_exhausted" replaces the old distance-target "goal" (2026-07-02):
    # the episode ran the shaft's full physical length rather than getting cut
    # short by stuck/tip_stuck/ejected/max_steps. It is NOT a success flag --
    # a lot of shaft can be consumed by slip with the tip barely advancing, so
    # final_progress is always the RAW measured value now, never forced to
    # 1.0/0 on this condition (that forcing was specific to the old "goal"
    # concept, where reaching the target position really did mean done).
    completed = info.get("termination_reason") == "shaft_exhausted"
    final_progress = float(info.get("progress", 0.0))
    goal_gap_m = float(info.get("goal_gap_m", max(0.0, goal_s - actual_s)))
    max_seen_gap_m = float(info.get("max_seen_goal_gap_m", max(0.0, goal_s - max_seen_s)))
    return {
        "steps":              steps,
        "sum_reward":         sum_reward,
        "mean_ncon":          sum_ncon / max(1, steps),
        "final_progress":     final_progress,
        "termination_reason": reason,
        "goal_gap_m":         goal_gap_m,
        "max_seen_gap_m":     max_seen_gap_m,
        "force_norm":         float(info.get("force_norm", 0.0)),
        "tip_contact_steps":  int(info.get("tip_contact_steps", 0)),
        "completed": completed,
    }


def _seed_to_summary(seed: int, results: list[dict], n_episodes: int) -> dict:
    progs   = np.array([r["final_progress"] for r in results])
    ncons   = np.array([r["mean_ncon"]       for r in results])
    rews    = np.array([r["sum_reward"]       for r in results])
    steps   = np.array([r["steps"]            for r in results])
    gaps_mm = np.array([r["goal_gap_m"] * 1000.0 for r in results])
    max_seen_gaps_mm = np.array([r["max_seen_gap_m"] * 1000.0 for r in results])
    forces  = np.array([r["force_norm"] for r in results])
    near_miss_99 = sum(
        1 for r in results
        if (not r["completed"]) and r["final_progress"] >= 0.99
    )
    near_miss_995 = sum(
        1 for r in results
        if (not r["completed"]) and r["final_progress"] >= 0.995
    )
    compl   = sum(1 for r in results if r["completed"])
    reasons = Counter(r["termination_reason"] for r in results)
    return {
        "seed": seed, "n_episodes": n_episodes,
        "completion_rate": compl / len(results),
        "progress_mean":   float(progs.mean()),
        "progress_min":    float(progs.min()),
        "progress_max":    float(progs.max()),
        "reward_mean":     float(rews.mean()),
        "steps_mean":      float(steps.mean()),
        "ncon_mean":       float(ncons.mean()),
        "goal_gap_mm_mean": float(gaps_mm.mean()),
        "goal_gap_mm_min":  float(gaps_mm.min()),
        "max_seen_gap_mm_mean": float(max_seen_gaps_mm.mean()),
        "force_mean":      float(forces.mean()),
        "near_miss_99":    int(near_miss_99),
        "near_miss_995":   int(near_miss_995),
        "reasons":         dict(reasons),
    }


def _print_summary(s: dict) -> None:
    compl = int(round(s["completion_rate"] * s["n_episodes"]))
    n     = s["n_episodes"]
    print(
        f"  seed {s['seed']:2d}: compl={compl}/{n}  "
        f"prog mean={s['progress_mean']*100:5.1f}%  "
        f"min={s['progress_min']*100:5.1f}%  max={s['progress_max']*100:5.1f}%  "
        f"gap={s['goal_gap_mm_mean']:5.1f}mm  reasons={s['reasons']}"
    )


def _write_summary_csv(
    csv_path: Path,
    summaries: list[dict],
    ckpt_path: Path,
    args: argparse.Namespace,
    seed_list: list[int],
    total_dt: float,
) -> None:
    reason_keys = sorted({k for s in summaries for k in s["reasons"]})
    fieldnames = [
        "rank",
        "seed",
        "checkpoint",
        "checkpoint_name",
        "run_dir",
        "episodes_per_seed",
        "n_eval_seeds",
        "workers",
        "gl_backend",
        "depth_res",
        "max_steps",
        "total_eval_time_s",
        "n_episodes",
        "completed_episodes",
        "completion_rate",
        "completion_pct",
        "progress_mean",
        "progress_mean_pct",
        "progress_min",
        "progress_min_pct",
        "progress_max",
        "progress_max_pct",
        "goal_gap_mm_mean",
        "goal_gap_mm_min",
        "max_seen_gap_mm_mean",
        "near_miss_99",
        "near_miss_995",
        "reward_mean",
        "steps_mean",
        "ncon_mean",
        "force_mean",
    ] + [f"reason_{k}" for k in reason_keys]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, s in enumerate(summaries, start=1):
            completed = int(round(s["completion_rate"] * s["n_episodes"]))
            row = {
                "rank": rank,
                "seed": s["seed"],
                "checkpoint": str(ckpt_path),
                "checkpoint_name": ckpt_path.name,
                "run_dir": str(_checkpoint_run_dir(ckpt_path)),
                "episodes_per_seed": args.episodes,
                "n_eval_seeds": len(seed_list),
                "workers": args.workers,
                "gl_backend": GL_BACKEND,
                "depth_res": args.depth_res,
                "max_steps": args.max_steps,
                "total_eval_time_s": f"{total_dt:.3f}",
                "n_episodes": s["n_episodes"],
                "completed_episodes": completed,
                "completion_rate": f"{s['completion_rate']:.6f}",
                "completion_pct": f"{s['completion_rate'] * 100.0:.3f}",
                "progress_mean": f"{s['progress_mean']:.6f}",
                "progress_mean_pct": f"{s['progress_mean'] * 100.0:.3f}",
                "progress_min": f"{s['progress_min']:.6f}",
                "progress_min_pct": f"{s['progress_min'] * 100.0:.3f}",
                "progress_max": f"{s['progress_max']:.6f}",
                "progress_max_pct": f"{s['progress_max'] * 100.0:.3f}",
                "goal_gap_mm_mean": f"{s['goal_gap_mm_mean']:.3f}",
                "goal_gap_mm_min": f"{s['goal_gap_mm_min']:.3f}",
                "max_seen_gap_mm_mean": f"{s['max_seen_gap_mm_mean']:.3f}",
                "near_miss_99": s["near_miss_99"],
                "near_miss_995": s["near_miss_995"],
                "reward_mean": f"{s['reward_mean']:.6f}",
                "steps_mean": f"{s['steps_mean']:.3f}",
                "ncon_mean": f"{s['ncon_mean']:.6f}",
                "force_mean": f"{s['force_mean']:.6f}",
            }
            for reason in reason_keys:
                row[f"reason_{reason}"] = s["reasons"].get(reason, 0)
            writer.writerow(row)


# Module-level worker globals
_W_actor_critic = None
_W_cfg          = None
_W_depth_res: int = DEPTH_RES
_W_max_steps: int = MAX_STEPS


def _worker_init(
    ckpt_path_str: str, depth_res: int, max_steps: int, experiment: str,
) -> None:
    # experiment arrives via initargs, not the module global: workers are spawned
    # processes that re-import this module fresh, so a --experiment override in
    # main() would not reach them.
    global _W_actor_critic, _W_cfg, _W_depth_res, _W_max_steps
    import sys
    from pathlib import Path as _Path
    _HERE = _Path(__file__).parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    if sys.platform == "win32":
        from unittest.mock import MagicMock
        _pwd = MagicMock()
        _pwd.getpwuid.return_value = MagicMock(pw_name="user")
        sys.modules["pwd"] = _pwd
    import torch as _torch
    _torch.backends.cudnn.enabled = False
    _torch.set_num_threads(1)
    try:
        _torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    import sample_factory.cfg.arguments as _sf_args
    import sample_factory.algo.runners.runner as _sf_runner
    from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
    from sample_factory.envs.env_utils import register_env
    from sample_factory.model.actor_critic import create_actor_critic
    from sf_encoder import register_scope_encoder
    import numpy as _np
    from gymnasium import spaces as _spaces
    from scope_colon_env import ScopeColonEnv as _Env, STATE_OBS_DIM
    _sf_args.get_git_commit_hash = lambda: ("unknown", "not a git repository")
    _sf_runner.save_git_diff = lambda _: None
    _train_dir = str(_HERE / "runs_sf")
    _defaults  = [
        "--algo=APPO", "--env=scope_colon",
        f"--experiment={experiment}", f"--train_dir={_train_dir}",
        "--num_workers=1", "--num_envs_per_worker=1",
        "--rollout=32", "--batch_size=2048",
        "--normalize_input=False", "--use_rnn=True",
        "--policy_workers_per_policy=1", "--worker_num_splits=1",
    ]
    register_env("scope_colon", lambda *a, **k: _Env(seed=0, depth_res=depth_res))
    register_scope_encoder()
    _sf_parser, _ = parse_sf_args(argv=_defaults)
    cfg           = parse_full_cfg(_sf_parser, argv=_defaults)
    obs_space     = _spaces.Dict({
        "depth": _spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, depth_res, depth_res),
            dtype=_np.float32,
        ),
        "state": _spaces.Box(
            low=-_np.inf,
            high=_np.inf,
            shape=(STATE_OBS_DIM,),
            dtype=_np.float32,
        ),
    })
    action_space  = _spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=_np.float32)
    actor_critic = create_actor_critic(cfg, obs_space, action_space)
    actor_critic.eval()
    ckpt = _torch.load(ckpt_path_str, map_location="cpu", weights_only=False)
    actor_critic.load_state_dict(ckpt["model"])
    _W_actor_critic = actor_critic
    _W_cfg          = cfg
    _W_depth_res    = depth_res
    _W_max_steps    = max_steps


def _worker_eval_seed(task: tuple) -> dict:
    from scope_colon_env import ScopeColonEnv as _Env
    seed, n_episodes = task
    env     = _Env(seed=seed, max_steps=_W_max_steps, depth_res=_W_depth_res)
    results = [run_episode(env, _W_actor_critic, _W_cfg) for _ in range(n_episodes)]
    env.close()
    return _seed_to_summary(seed, results, n_episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v7_shaft checkpoint")
    parser.add_argument("--experiment", type=str, default=EXPERIMENT,
                        help=f"run folder under runs_sf/ (default: {EXPERIMENT})")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="explicit .pth or checkpoint_p0 folder; "
                             "defaults to the --experiment run's checkpoint_p0/")
    parser.add_argument("--episodes",   type=int, default=N_EPISODES)
    parser.add_argument("--seeds",      type=int, default=N_SEEDS)
    parser.add_argument("--seed",       type=int, action="append", default=None,
                        help="specific seed to evaluate; can be repeated")
    parser.add_argument("--workers",    type=int, default=DEFAULT_WORKERS,
                        help=f"parallel workers (default: {DEFAULT_WORKERS}; 0 = min(seeds, cpu_count))")
    # Already consumed by _select_gl_backend() at import time; declared here so
    # it shows up in --help and lands in args for the CSV.
    parser.add_argument("--gl", choices=GL_BACKENDS, default=GL_BACKEND,
                        help=f"MuJoCo render backend (default: {DEFAULT_GL}; "
                             f"use osmesa for CPU-only rendering, e.g. under WSL2)")
    parser.add_argument("--depth_res",  type=int, default=DEPTH_RES)
    parser.add_argument("--max_steps",  type=int, default=MAX_STEPS,
                        help=f"per-episode step cap (default: {MAX_STEPS})")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV output path; default writes beside the model folder")
    parser.add_argument("--no_csv", action="store_true",
                        help="disable CSV summary output")
    parser.add_argument("--list_checkpoints", action="store_true")
    args = parser.parse_args()

    if args.list_checkpoints:
        ckpts = _list_checkpoints(args.experiment)
        if not ckpts:
            print("No checkpoints found.")
        else:
            print(f"Checkpoints in runs_sf/{args.experiment}/checkpoint_p0/:")
            for p in ckpts:
                print(f"  {p.name}")
        return

    ckpt_path  = _resolve_checkpoint(args.checkpoint, args.experiment)
    seed_list = args.seed if args.seed is not None else list(range(args.seeds))
    n_workers  = args.workers if args.workers > 0 else min(len(seed_list), os.cpu_count() or 8)
    n_workers  = max(1, min(n_workers, len(seed_list)))
    total_t0   = time.perf_counter()
    summaries: list[dict] = []

    print(f"Experiment : {args.experiment}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Render     : MUJOCO_GL={GL_BACKEND}"
          f"{' (CPU)' if GL_BACKEND == 'osmesa' else ''}")
    tasks = [(seed, args.episodes) for seed in seed_list]
    print(f"Running {args.episodes} eps × {len(seed_list)} seeds across {n_workers} workers ...\n")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(str(ckpt_path), args.depth_res, args.max_steps, args.experiment),
    ) as ex:
        futures = {ex.submit(_worker_eval_seed, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futures):
            summary = fut.result()
            summaries.append(summary)
            done += 1
            _print_summary(summary)
            print(f"    ({done}/{len(seed_list)} seeds done)", flush=True)

    total_dt = time.perf_counter() - total_t0
    print(f"\nTotal eval time: {total_dt:.1f}s")
    summaries.sort(key=lambda s: (-s["completion_rate"], -s["progress_mean"]))
    print("\n" + "=" * 92)
    print("PER-SEED SUMMARY  (best → worst)")
    print("=" * 92)
    print(
        f"{'seed':>4}  {'compl%':>6}  {'prog mean':>9}  {'prog min':>9}  "
        f"{'prog max':>9}  {'gap mm':>7}  {'near99':>6}  {'reward':>8}  "
        f"{'steps':>5}  {'ncon':>5}  reasons"
    )
    print("-" * 112)
    for s in summaries:
        reasons_str = "  ".join(f"{k}:{v}" for k, v in s["reasons"].items())
        print(
            f"{s['seed']:>4}  {s['completion_rate']*100:>5.0f}%  "
            f"{s['progress_mean']*100:>8.1f}%  {s['progress_min']*100:>8.1f}%  "
            f"{s['progress_max']*100:>8.1f}%  {s['goal_gap_mm_mean']:>7.1f}  "
            f"{s['near_miss_99']:>6d}  {s['reward_mean']:>+8.3f}  "
            f"{s['steps_mean']:>5.0f}  {s['ncon_mean']:>5.1f}  {reasons_str}"
        )
    overall_compl = np.mean([s["completion_rate"] for s in summaries])
    overall_prog  = np.mean([s["progress_mean"]   for s in summaries])
    overall_gap = np.mean([s["goal_gap_mm_mean"] for s in summaries])
    overall_near99 = sum(s["near_miss_99"] for s in summaries)
    overall_near995 = sum(s["near_miss_995"] for s in summaries)
    overall_reasons = Counter()
    for s in summaries:
        overall_reasons.update(s["reasons"])
    print(
        f"\nOverall: completion {overall_compl*100:.1f}%,  "
        f"mean progress {overall_prog*100:.1f}%,  "
        f"mean goal gap {overall_gap:.1f} mm"
    )
    print(
        f"Near misses: >=99% {overall_near99}, >=99.5% {overall_near995} "
        f"out of {args.episodes * len(seed_list)} episodes"
    )
    print(f"Reasons: {dict(overall_reasons)}")
    print(f"Best:    seed {summaries[0]['seed']}  ({summaries[0]['completion_rate']*100:.0f}% completion)")
    print(f"Worst:   seed {summaries[-1]['seed']}  ({summaries[-1]['completion_rate']*100:.0f}% completion)")
    if not args.no_csv:
        csv_path = Path(args.csv) if args.csv else _default_csv_path(
            ckpt_path, args.episodes, len(seed_list),
        )
        _write_summary_csv(csv_path, summaries, ckpt_path, args, seed_list, total_dt)
        print(f"CSV:     {csv_path}")


if __name__ == "__main__":
    main()
