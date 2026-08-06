"""
Sample Factory APPO training for ScopeColonEnv — v8_p1 (V8 Phase 1).

Inherits v7_shaft's physical shaft (universal-joint chain fed by an entrance
roller/capstan mechanism, replacing v6_dr's kinematic mocap-driven base
advance) and adds the hardware-matching observation: no force-derived signals,
a 6-D state vector, and a 16:9 54×96 depth image on a camera that matches the
DA3 depth model's training geometry. Domain randomisation on tendon gains,
dead-zones and lag is retained; the synthetic base advance-gain DR is gone
since real contact friction now provides that variability physically.
All dependencies are bundled in this directory — no external path additions needed.

Training runs on the remote native-Linux rig, never from Windows (Sample
Factory cannot even be imported there — sample_factory/utils/utils.py imports
the POSIX-only `pwd`).

USAGE
-----
  python train_sf.py --smoke          # 10k step smoke test
  python train_sf.py                  # full training run
  python train_sf.py --resume         # continue from last checkpoint

⚠️ Without --resume this runs with restart_behavior=overwrite. Sample Factory
otherwise RESUMES from any checkpoint already in the experiment dir, so a
genuinely from-scratch run means deleting runs_sf/<EXPERIMENT>/ — renaming
EXPERIMENT is not enough.

OUTPUT
------
  runs_sf/v8_p1_v1/checkpoint_p0/
"""
from __future__ import annotations

import logging
import numbers
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import torch

_orig_torch_load = torch.load


def _torch_load_default_weights_only_false(*args, **kwargs):
    """Sample Factory's own checkpoint loader (site-packages, learner.py)
    calls torch.load without weights_only, so PyTorch 2.6+'s new default
    (True) rejects the numpy globals pickled in our checkpoints. All
    checkpoints here are self-generated and trusted."""
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_default_weights_only_false

from sample_factory.algo.runners.runner import AlgoObserver
import sample_factory.algo.runners.runner as sf_runner_module
from sample_factory.algo.utils.misc import ExperimentStatus
import sample_factory.cfg.arguments as sf_arguments
from sample_factory.train import make_runner
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.utils.utils import log as sf_log

from sf_encoder import register_scope_encoder
from scope_colon_env import DEFAULT_DEPTH_RES, ScopeColonEnv


# ---------------------------------------------------------------------------
# Training configuration — v17 best known config
# ---------------------------------------------------------------------------

N_WORKERS     = 40
ROLLOUT       = 64
BATCH_SIZE    = 4096
NUM_EPOCHS    = 2
LEARNING_RATE = 1e-4
ENTROPY_COEFF = 0.003
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_RANGE    = 0.2
TOTAL_STEPS   = 5_000_000    # 2026-08-04: 10M -> 5M, chosen alongside the 100 Hz -> 25 Hz move
                              # (Change F). An env step now covers 40 ms of task instead of 10 ms
                              # and costs ~4x more wall-clock -- physics per sim-second is identical
                              # (2000 mj_steps/s either way), so the step count had to be rebased or
                              # the run would have taken 4x longer for no reason.
                              # Reference points, all at 25 Hz unless stated:
                              #   v8_p1_v1   10M @ 100 Hz = 100,000 sim-s, ~9,800 eps  -> 93.5%
                              #   2.5M       = the EXACT equivalent (same sim-time and wall-clock)
                              #   5M (here)  = 200,000 sim-s, ~19,600 eps, ~2x the last run
                              # 2x rather than 1x because this bundle changes the physics regime
                              # (A, D), the obs (C, E) and the decision rate (F) all at once, so the
                              # policy is relearning from scratch rather than refining -- headroom is
                              # worth more than usual. SF checkpoints continuously, so stop it on the
                              # high-water mark rather than waiting for the budget if it plateaus.
                              # Judge that mark over a LONG window, not 3-4 printouts.
SMOKE_STEPS   = 10_000
DEPTH_RES     = DEFAULT_DEPTH_RES   # (H, W) = (54, 96), 16:9 -- see scope_colon_env.py.
                                     # Was a hardcoded 64 (square); now tracks the env so the
                                     # two cannot silently disagree about obs shape.
EXPERIMENT    = "v8_p1_v1"
TRAIN_DIR     = str(HERE / "runs_sf")


# ---------------------------------------------------------------------------
# Stats observer
# ---------------------------------------------------------------------------

class ScopeStatsObserver(AlgoObserver):
    PRINT_INTERVAL_S = 30.0

    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self._t_start    = time.time()
        self._last_print = self._t_start
        self._last_steps = 0
        self._runner     = None

    def on_init(self, runner) -> None:
        self._runner = runner

    def process_infos(self, *args, **kwargs) -> None:
        self._trigger_print()

    def on_training_step(self, *args, **kwargs) -> None:
        self._trigger_print()

    def after_training_step(self, *args, **kwargs) -> None:
        self._trigger_print()

    def _trigger_print(self) -> None:
        runner = getattr(self, "_runner", None)
        if runner is None:
            return
        raw       = getattr(runner, "env_steps", 0)
        env_steps = int(sum(raw.values())) if isinstance(raw, dict) else int(raw)
        self._maybe_print(env_steps, runner=runner)

    @staticmethod
    def _runner_stat_values(runner, key: str) -> list[float]:
        policy_stats = getattr(runner, "policy_avg_stats", {})
        if key not in policy_stats:
            return []
        values: list[float] = []
        raw = policy_stats[key]
        if isinstance(raw, numbers.Real):
            return [float(raw)]
        try:
            for item in raw:
                if isinstance(item, numbers.Real):
                    values.append(float(item))
                else:
                    try:
                        for v in item:
                            if isinstance(v, numbers.Real):
                                values.append(float(v))
                    except TypeError:
                        pass
        except TypeError:
            pass
        return values

    def _maybe_print(self, env_steps: int, runner=None) -> None:
        now = time.time()
        if now - self._last_print < self.PRINT_INTERVAL_S:
            return

        elapsed = now - self._t_start
        fps     = (env_steps - self._last_steps) / max(now - self._last_print, 1e-6)
        pct     = env_steps / max(1, self.total_steps) * 100

        h, rem = divmod(int(elapsed), 3600)
        m, s   = divmod(rem, 60)
        elapsed_str = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

        if fps > 1:
            eta_s    = (self.total_steps - env_steps) / fps
            eh, erem = divmod(int(eta_s), 3600)
            em, es   = divmod(erem, 60)
            eta_str  = f"{eh}h{em:02d}m" if eh else f"{em}m{es:02d}s"
        else:
            eta_str = "???"

        prog_vals = self._runner_stat_values(runner, "progress") if runner else []
        rew_vals  = self._runner_stat_values(runner, "reward")   if runner else []
        len_vals  = self._runner_stat_values(runner, "len")      if runner else []

        if prog_vals or rew_vals:
            parts: list[str] = []
            if prog_vals:
                mean_p = sum(prog_vals) / len(prog_vals) * 100
                max_p  = max(prog_vals) * 100
                parts.append(f"mean={mean_p:.1f}%  max={max_p:.1f}%")
            if rew_vals:
                parts.append(f"rew={sum(rew_vals)/len(rew_vals):+.3f}")
            if len_vals:
                parts.append(f"len={sum(len_vals)/len(len_vals):.0f}")
            if runner is not None:
                reason_fracs = {}
                for k in ("shaft_exhausted", "stuck", "tip_stuck", "ejected", "truncated"):
                    vals = self._runner_stat_values(runner, k)
                    if vals:
                        frac = sum(vals) / len(vals)
                        if frac > 0.01:
                            reason_fracs[k] = frac
                if reason_fracs:
                    r_str = "  ".join(
                        f"{k}:{frac*100:.0f}%"
                        for k, frac in sorted(reason_fracs.items(), key=lambda x: -x[1])[:4]
                    )
                    parts.append(f"| {r_str}")
            ep_line = "  ".join(parts)
        else:
            ep_line = "(starting up — episode stats appear after first episode ends)"

        print(
            f"\n── [SF v8_p1]  {env_steps:>10,} / {self.total_steps:,}  ({pct:.1f}%)"
            f"  elapsed={elapsed_str}  ETA={eta_str}  fps={fps:.0f}"
            f"\n           {ep_line}\n",
            flush=True,
        )
        self._last_print = now
        self._last_steps = env_steps


# ---------------------------------------------------------------------------
# Parallel scene pre-generation (module-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _pregen_init(here_str: str) -> None:
    if here_str not in sys.path:
        sys.path.insert(0, here_str)


def _pregen_scene(seed: int) -> int:
    from build_collision_scene import build_scene
    build_scene(seed)
    return seed


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env(full_env_name: str, cfg=None, env_config=None, render_mode=None):
    seed = getattr(env_config, "env_id", 0) if env_config is not None else 0
    return ScopeColonEnv(seed=int(seed), depth_res=DEPTH_RES)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.getLogger("sample_factory").setLevel(logging.WARNING)
    sf_log.setLevel(logging.WARNING)
    for handler in sf_log.handlers:
        handler.setLevel(logging.WARNING)
    sf_arguments.get_git_commit_hash = lambda: ("unknown", "not a git repository")
    sf_runner_module.save_git_diff = lambda _directory: None

    smoke  = "--smoke"  in sys.argv
    resume = "--resume" in sys.argv
    if smoke:
        sys.argv.remove("--smoke")
    if resume:
        sys.argv.remove("--resume")

    _n_scenes = N_WORKERS * 2
    _n_cores  = min(_n_scenes, os.cpu_count() or 8)
    print(f"Pre-generating {_n_scenes} scenes across {_n_cores} cores ...", flush=True)
    with ProcessPoolExecutor(
        max_workers=_n_cores,
        initializer=_pregen_init,
        initargs=(str(HERE),),
    ) as _ex:
        _done = 0
        for _ in as_completed([_ex.submit(_pregen_scene, s) for s in range(_n_scenes)]):
            _done += 1
            print(f"  {_done}/{_n_scenes} scenes ready", end="\r", flush=True)
    print(f"  {_n_scenes}/{_n_scenes} scenes ready          ", flush=True)

    register_env("scope_colon", make_env)
    register_scope_encoder()

    defaults = [
        "--algo=APPO",
        "--env=scope_colon",
        f"--experiment={EXPERIMENT}",
        f"--train_dir={TRAIN_DIR}",
        f"--num_workers={N_WORKERS}",
        "--num_envs_per_worker=2",
        f"--rollout={ROLLOUT}",
        f"--batch_size={BATCH_SIZE}",
        f"--num_batches_per_epoch={NUM_EPOCHS}",
        f"--learning_rate={LEARNING_RATE}",
        f"--exploration_loss_coeff={ENTROPY_COEFF}",
        f"--gamma={GAMMA}",
        f"--gae_lambda={GAE_LAMBDA}",
        f"--ppo_clip_ratio={CLIP_RANGE}",
        f"--train_for_env_steps={SMOKE_STEPS if smoke else TOTAL_STEPS}",
        "--normalize_input=False",
        "--max_grad_norm=0.5",
        "--use_rnn=True",
        "--policy_workers_per_policy=1",
        "--worker_num_splits=2",
        "--decorrelate_experience_max_seconds=10",
        f"--restart_behavior={'resume' if resume else 'overwrite'}",
    ]

    argv    = defaults + sys.argv[1:]
    parser, _partial_cfg = parse_sf_args(argv=argv)
    cfg     = parse_full_cfg(parser, argv=argv)

    if smoke:
        print(f"[smoke] {cfg.train_for_env_steps:,} steps  {cfg.num_workers} workers  depth_res={DEPTH_RES}")

    observer    = ScopeStatsObserver(total_steps=int(cfg.train_for_env_steps))
    cfg, runner = make_runner(cfg)
    runner.register_observer(observer)

    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()
    return status


if __name__ == "__main__":
    main()
