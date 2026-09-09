"""
PolicyRunner -- APPO-checkpoint inference, wrapped for the ROS control loop.

WHAT THIS IS
------------
`policy_node` needs one thing from the trained model: given a depth image and a
6-D state vector, return the 3-D action. This class is that, and nothing else:

    runner = PolicyRunner(experiment="v8_p1_v1", checkpoint_root="~/mujoco_v3/navigation/v8_p1")
    runner.reset()                       # zero the GRU hidden state (start of a run)
    action = runner.act(depth, state)    # (3,) float32, the distribution MEAN

WHY IT REUSES SAMPLE FACTORY
---------------------------
The checkpoint was trained with Sample Factory (APPO). SF *does* import on Linux
(the `pwd`-import problem is Windows-only), so the lowest-risk way to get bit-exact
behaviour is to let SF build the network exactly as it did at training time
(`create_actor_critic`) and then load the weights. This is the same approach the
already-verified `navigation/v8_p1/policy_server_wsl.py` takes -- see the memory
note "Closed-loop nav test (Blender+DA3+policy)".

Note this does NOT start an SF runner, learner, or any worker process. It only
uses SF's *model construction* code. The heavy import is contained to __init__.

A future option (flagged in the deployment-architecture memory as "the gate") is
to reimplement encoder+GRU+head as a standalone torch module so this node has no
SF dependency at all. That is a drop-in replacement for this class -- keep the
`reset()` / `act()` API stable and `policy_node` never needs to change. Verify any
such reimplementation numerically against `viewer_sf.py --deterministic` to 1e-6
before trusting it.

DETERMINISM
-----------
`act()` returns the Gaussian's MEAN, never a sample. Sampling is PPO exploration
noise (measured sigma ~= 0.63 on a +/-1 action space) and dominates the command.
docs/architecture.md and the viewer memory both say: deployment takes the mean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


class PolicyRunnerError(RuntimeError):
    pass


class MockPolicyRunner:
    """Zero-action stand-in with the same API.

    Lets the whole graph (depth_bridge -> policy_node -> mock_scope_link) be
    brought up and rate-checked on the bench before torch / the checkpoint /
    Sample Factory are in place. Selected by the `mock_policy` parameter.
    """

    def __init__(self, *, depth_hw=(54, 96), state_dim=6, **_ignored):
        self.depth_hw = tuple(depth_hw)
        self.state_dim = int(state_dim)
        self.is_mock = True

    def reset(self):
        pass

    def act(self, depth, state):  # noqa: D401 - matches PolicyRunner
        _ = np.asarray(depth), np.asarray(state)
        return np.zeros(3, dtype=np.float32)


class PolicyRunner:
    def __init__(
        self,
        *,
        experiment: str,
        nav_code_dir: str,
        checkpoint_dir: str = "",
        checkpoint_root: str = "",
        depth_hw=(54, 96),
        state_dim: int = 6,
        device: str = "cpu",
    ):
        """
        experiment       : SF experiment name, e.g. "v8_p1_v1".
        nav_code_dir      : directory containing `sf_encoder.py` (the custom
                            encoder the checkpoint was trained with). On the WSL
                            mirror this is ~/mujoco_v3/navigation/v8_p1 (or v9).
        checkpoint_dir    : absolute path to the `checkpoint_p0` folder, or to a
                            specific `best_*.pth`. Takes priority if set.
        checkpoint_root   : absolute path that *contains* `runs_sf/`. The
                            checkpoint is then
                            <root>/runs_sf/<experiment>/checkpoint_p0/best_*.pth.
        depth_hw          : (H, W) of the depth obs. MUST match the trained env
                            (54, 96 for the v8_p1 / v9 line).
        state_dim         : length of the state obs (6 for the v8_p1 / v9 line).
        device            : "cpu" (default -- the net is tiny; see
                            policy_server_wsl.py's reasoning) or "cuda".
        """
        self.experiment = experiment
        self.depth_hw = (int(depth_hw[0]), int(depth_hw[1]))
        self.state_dim = int(state_dim)
        self.device = device
        self.is_mock = False

        # --- make sf_encoder importable -----------------------------------
        if not nav_code_dir:
            raise PolicyRunnerError("nav_code_dir must be set (dir containing sf_encoder.py)")
        nav_dir = Path(nav_code_dir).expanduser().resolve()
        if not (nav_dir / "sf_encoder.py").is_file():
            raise PolicyRunnerError(f"sf_encoder.py not found in nav_code_dir: {nav_dir}")
        if str(nav_dir) not in sys.path:
            sys.path.insert(0, str(nav_dir))

        # --- resolve the checkpoint file --------------------------------
        ckpt_path, resolved_ckpt_dir = self._resolve_checkpoint(
            checkpoint_dir, checkpoint_root, experiment
        )
        self.checkpoint_path = str(ckpt_path)

        # --- heavy imports (contained here) ----------------------------
        import torch
        from gymnasium import spaces
        import sample_factory.cfg.arguments as sf_arguments
        import sample_factory.algo.runners.runner as sf_runner_module
        from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
        from sample_factory.envs.env_utils import register_env
        from sample_factory.model.actor_critic import create_actor_critic

        from sf_encoder import register_scope_encoder

        self._torch = torch

        # cuDNN can fail to init in odd process contexts; the net is tiny so
        # this costs nothing. Same line as sf_encoder.py / policy_server_wsl.py.
        torch.backends.cudnn.enabled = False

        # SF tries to read git metadata from the cwd on cfg parse; stub it so a
        # non-repo working dir (or a ROS install space) does not error.
        sf_arguments.get_git_commit_hash = lambda: ("unknown", "not a git repository")
        sf_runner_module.save_git_diff = lambda _: None

        def _no_env(*_a, **_k):
            raise RuntimeError("PolicyRunner never constructs an env instance")

        register_env("scope_colon", _no_env)
        register_scope_encoder()

        # These flags are the ARCHITECTURE the checkpoint was built under, not
        # run-mode options -- copied from policy_server_wsl.py / viewer_sf.py so
        # `create_actor_critic` produces a network the state_dict fits exactly.
        sf_argv = [
            "--algo=APPO", "--env=scope_colon",
            f"--experiment={experiment}", f"--train_dir={resolved_ckpt_dir}",
            "--num_workers=1", "--num_envs_per_worker=1",
            "--rollout=32", "--batch_size=2048",
            "--normalize_input=False", "--use_rnn=True",
            "--policy_workers_per_policy=1", "--worker_num_splits=1",
            f"--device={device}",
        ]
        sf_parser, _ = parse_sf_args(argv=sf_argv)
        cfg = parse_full_cfg(sf_parser, argv=sf_argv)

        # Obs / action spaces, hand-built to match ScopeColonEnv without
        # importing it (which would pull in MuJoCo + a GL context).
        h, w = self.depth_hw
        obs_space = spaces.Dict({
            "depth": spaces.Box(low=0.0, high=1.0, shape=(1, h, w), dtype=np.float32),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32),
        })
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self._ac = create_actor_critic(cfg, obs_space, action_space)
        self._ac.eval()

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self._ac.load_state_dict(ckpt["model"])
        self.env_steps = ckpt.get("env_steps", None)

        self._rnn_size = int(getattr(cfg, "rnn_size", 512)) if cfg.use_rnn else 1
        self._rnn = torch.zeros([1, self._rnn_size])

        # Sanity: the depth CNN's first Linear tells you the obs geometry the
        # checkpoint was trained on (13440 = 54x96, 10816 = old 64x64).
        try:
            w_in = self._ac.encoder.depth_cnn.net[1].weight.shape[1]
            expected = self._depth_flatten_size(h, w)
            if w_in != expected:
                raise PolicyRunnerError(
                    f"checkpoint depth CNN expects flatten={w_in} but depth_hw={self.depth_hw} "
                    f"gives {expected}. Wrong depth resolution for this checkpoint."
                )
        except AttributeError:
            pass  # different encoder layout; skip the check rather than crash

    # -------------------------------------------------------------- API

    def reset(self):
        """Zero the GRU hidden state. Call at the start of every run -- the
        recurrent state is load-bearing (partial observability at haustral
        folds), and carrying it across runs is not the trained policy."""
        self._rnn = self._torch.zeros([1, self._rnn_size])

    def act(self, depth, state):
        """
        depth : (H, W) or (1, H, W) float32, already normalised to [0, 1]
                (per-frame-max, done upstream by Da3Depth.to_obs).
        state : (state_dim,) float32 -> [cmd_x_n, cmd_y_n, last_a0, last_a1, last_a2, tip_contact]
        returns : (3,) float32 -- the action distribution MEAN.
        """
        torch = self._torch

        d = np.asarray(depth, dtype=np.float32)
        if d.ndim == 2:
            d = d[None, :, :]
        if d.shape != (1, *self.depth_hw):
            raise PolicyRunnerError(f"depth shape {d.shape}, expected {(1, *self.depth_hw)}")

        s = np.asarray(state, dtype=np.float32).reshape(-1)
        if s.shape[0] != self.state_dim:
            raise PolicyRunnerError(f"state length {s.shape[0]}, expected {self.state_dim}")

        obs_t = {
            "depth": torch.from_numpy(np.ascontiguousarray(d)).unsqueeze(0),   # (1,1,H,W)
            "state": torch.from_numpy(np.ascontiguousarray(s)).unsqueeze(0),   # (1,state_dim)
        }
        with torch.no_grad():
            out = self._ac(obs_t, self._rnn)
            action_t = self._ac.action_distribution().means   # deterministic, not a sample
        self._rnn = out.get("new_rnn_states", self._rnn)
        return action_t.squeeze(0).cpu().numpy().astype(np.float32)

    # ---------------------------------------------------------- helpers

    @staticmethod
    def _depth_flatten_size(h: int, w: int) -> int:
        """Replicates _DepthCNN's conv stack (k3s2, k3s2, k3s1) to predict the
        flatten width, so a resolution mismatch is caught at load time."""
        def s2(x):
            return (x - 3) // 2 + 1

        def s1(x):
            return (x - 3) + 1

        hh, ww = s1(s2(s2(h))), s1(s2(s2(w)))
        return hh * ww * 64

    @staticmethod
    def _resolve_checkpoint(checkpoint_dir: str, checkpoint_root: str, experiment: str):
        if checkpoint_dir:
            p = Path(checkpoint_dir).expanduser().resolve()
        elif checkpoint_root:
            p = (Path(checkpoint_root).expanduser().resolve()
                 / "runs_sf" / experiment / "checkpoint_p0")
        else:
            raise PolicyRunnerError("set either checkpoint_dir or checkpoint_root")

        if p.is_file():
            if not p.name.startswith("best_"):
                raise PolicyRunnerError(f"expected a best_*.pth checkpoint, got {p}")
            return p, str(p.parent)
        if not p.is_dir():
            raise PolicyRunnerError(f"checkpoint path does not exist: {p}")

        best = sorted(p.glob("best_*.pth"), key=lambda q: q.stat().st_mtime)
        if not best:
            raise PolicyRunnerError(f"no best_*.pth in {p}")
        return best[-1], str(p)
