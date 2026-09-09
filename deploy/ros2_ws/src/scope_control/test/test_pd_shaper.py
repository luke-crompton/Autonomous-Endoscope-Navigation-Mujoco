"""
Verify pd_shaper_step() reproduces the simulator's PD command shaper.

    python -m pytest deploy/ros2_ws/src/scope_control/test/test_pd_shaper.py

policy_node runs the shaper NORMALISED (limit = 1.0, target = action). The sim
runs it with limit = MAX_PULL, target = action * MAX_PULL, then divides the
output by MAX_PULL for the observation. These are algebraically identical -- the
first two tests check that against sim's actual constants.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scope_control"))

from shaper import pd_shaper_step  # noqa: E402

PD_KP = 0.1803
PD_KD = 0.0475
# sim's (symmetric) constants, just to prove equivalence
SIM_MAX_PULL_X = 0.006795
SIM_MAX_PULL_Z = 0.007361


def _sim_reference(actions, max_pull):
    """scope_colon_env.py step(): the two lines per axis, then normalise."""
    cmd = 0.0
    prev_cmd = 0.0
    out = []
    for a in actions:
        target = a * max_pull
        vel = cmd - prev_cmd
        prev_cmd = cmd
        cmd = float(np.clip(cmd + PD_KP * (target - cmd) - PD_KD * vel, -max_pull, max_pull))
        out.append(cmd / max_pull)          # cmd_x_n
    return out


def _node_impl(actions):
    """policy_node: shaper normalised, limit = 1.0, target = action."""
    cmd = 0.0
    prev_cmd = 0.0
    out = []
    for a in actions:
        cmd, prev_cmd = pd_shaper_step(cmd, prev_cmd, float(a), PD_KP, PD_KD, 1.0)
        out.append(cmd)
    return out


def test_normalised_matches_sim_x():
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1.0, 1.0, size=400)
    assert np.allclose(_node_impl(actions), _sim_reference(actions, SIM_MAX_PULL_X),
                       rtol=1e-9, atol=1e-12)


def test_normalised_matches_sim_z():
    rng = np.random.default_rng(1)
    actions = rng.uniform(-1.0, 1.0, size=400)
    assert np.allclose(_node_impl(actions), _sim_reference(actions, SIM_MAX_PULL_Z),
                       rtol=1e-9, atol=1e-12)


def test_output_stays_in_unit_range():
    rng = np.random.default_rng(2)
    actions = rng.uniform(-1.0, 1.0, size=500)
    out = np.array(_node_impl(actions))
    assert out.max() <= 1.0 and out.min() >= -1.0


def test_clip_holds_at_limit():
    cmd, prev = 0.0, 0.0
    for _ in range(200):
        cmd, prev = pd_shaper_step(cmd, prev, 5.0, PD_KP, PD_KD, 1.0)   # target past +1
    assert cmd == 1.0
    for _ in range(400):
        cmd, prev = pd_shaper_step(cmd, prev, -5.0, PD_KP, PD_KD, 1.0)
    assert cmd == -1.0


def test_zero_action_stays_zero():
    cmd, prev = 0.0, 0.0
    for _ in range(50):
        cmd, prev = pd_shaper_step(cmd, prev, 0.0, PD_KP, PD_KD, 1.0)
    assert cmd == 0.0
