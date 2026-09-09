"""
The PD command shaper -- isolated, dependency-free, so it can be unit-tested
against the simulator on any machine (no ROS, no torch, no MuJoCo).

This is a verbatim port of scope_colon_env.py step() (v9, ~lines 1004-1021):

    target   = action * max_pull
    vel      = cmd_pair - prev_cmd
    prev_cmd = cmd_pair
    cmd_pair = clip(cmd_pair + KP*(target - cmd_pair) - KD*vel, -max_pull, +max_pull)

Its output (`cmd_pair`, normalised by `max_pull`) is fed straight back into the
next policy observation, so it MUST match the trained env exactly. `PD_KP` /
`PD_KD` were fitted at 25 Hz and are NOT arithmetically rescalable to another
rate -- a x4 rescale puts the discrete poles at |z| = 1.337 (diverging).
"""

from __future__ import annotations


def pd_shaper_step(cmd: float, prev_cmd: float, target: float,
                   kp: float, kd: float, limit: float) -> tuple[float, float]:
    """One update. Returns (new_cmd, new_prev_cmd).

    `cmd`      : current shaper output
    `prev_cmd` : shaper output one step ago -- for the derivative term
    `target`   : action * limit for this axis
    `limit`    : output is clipped to +/- this. policy_node runs the shaper
                 normalised (limit = 1.0); the sim runs it with limit = MAX_PULL.
                 The two are identical up to the MAX_PULL scale, which cancels
                 out of the observation.
    """
    vel = cmd - prev_cmd
    new_cmd = cmd + kp * (target - cmd) - kd * vel
    if new_cmd > limit:
        new_cmd = limit
    elif new_cmd < -limit:
        new_cmd = -limit
    return new_cmd, cmd
