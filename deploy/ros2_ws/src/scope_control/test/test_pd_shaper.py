"""
Verify pd_shaper_step() reproduces the simulator's PD command shaper exactly.

Runs standalone on any machine:
    python -m pytest deploy/ros2_ws/src/scope_control/test/test_pd_shaper.py

The reference below is a literal transcription of scope_colon_env.py step()
(v9). If that filter ever changes in the env, this test must be updated in
lockstep -- and so must policy_node, because the shaper output is a policy
*input*.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scope_control"))

from shaper import pd_shaper_step  # noqa: E402

# The gains and (sim) max-pull values that ship with the v8_p1 / v9 line.
PD_KP = 0.1803
PD_KD = 0.0475
MAX_PULL_X = 0.006795   # metres, as in sim
MAX_PULL_Y = 0.007361


def _sim_reference(actions, max_pull):
    """scope_colon_env.py step(), the two lines per axis, unrolled."""
    cmd = 0.0
    prev_cmd = 0.0
    out = []
    for a in actions:
        target = a * max_pull
        vel = cmd - prev_cmd
        prev_cmd = cmd
        cmd = float(np.clip(
            cmd + PD_KP * (target - cmd) - PD_KD * vel,
            -max_pull, max_pull,
        ))
        out.append(cmd)
    return out


def _node_impl(actions, max_pull):
    cmd = 0.0
    prev_cmd = 0.0
    out = []
    for a in actions:
        target = a * max_pull
        cmd, prev_cmd = pd_shaper_step(cmd, prev_cmd, target, PD_KP, PD_KD, max_pull)
        out.append(cmd)
    return out


def test_matches_sim_random_sequence():
    rng = np.random.default_rng(0)
    for max_pull in (MAX_PULL_X, MAX_PULL_Y):
        actions = rng.uniform(-1.0, 1.0, size=400)
        ref = _sim_reference(actions, max_pull)
        got = _node_impl(actions, max_pull)
        assert np.allclose(ref, got, atol=0.0, rtol=0.0)


def test_scale_invariance_mm_vs_m():
    """policy_node works in mm; sim works in m. The normalised obs value
    cmd/max_pull must come out identical either way."""
    rng = np.random.default_rng(1)
    actions = rng.uniform(-1.0, 1.0, size=200)
    in_m = np.array(_node_impl(actions, MAX_PULL_X)) / MAX_PULL_X
    in_mm = np.array(_node_impl(actions, MAX_PULL_X * 1000.0)) / (MAX_PULL_X * 1000.0)
    assert np.allclose(in_m, in_mm, rtol=1e-12, atol=1e-12)


def test_clip_holds_at_limit():
    cmd, prev = 0.0, 0.0
    for _ in range(100):
        cmd, prev = pd_shaper_step(cmd, prev, 10.0, PD_KP, PD_KD, 1.5)  # target far past limit
    assert cmd == 1.5
    for _ in range(200):
        cmd, prev = pd_shaper_step(cmd, prev, -10.0, PD_KP, PD_KD, 1.5)
    assert cmd == -1.5


def test_zero_action_stays_zero():
    cmd, prev = 0.0, 0.0
    for _ in range(50):
        cmd, prev = pd_shaper_step(cmd, prev, 0.0, PD_KP, PD_KD, MAX_PULL_X)
    assert cmd == 0.0
