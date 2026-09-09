# Steering setpoint for the endoscope tip. Produced by policy_node once per
# incoming depth frame (~25 Hz), consumed by scope_link -> ESP32-S3 firmware.
#
# NORMALISED command per antagonistic tendon pair: -1.0 .. +1.0, where +/-1 is
# "full commanded bend" for that axis. This is the output of the PD command
# shaper (which lives in policy_node because its own output is fed back as two
# of the six policy observations -- it must sit inside the inference loop).
#
# The shaper works in normalised units, exactly matching the simulator's
# cmd_pair / MAX_PULL: the MAX_PULL scale factor cancels out of the observation,
# so nothing on the ROS side needs to know it. The real per-axis, per-direction
# encoder limits (which are ASYMMETRIC on the rig -- see
# hardware/bringup/measured_constants.md) live only in the firmware, which maps
# this normalised command to encoder ticks. Recalibrating is a CONFIG-frame /
# params-file change, no reflash.
#
# Because every message carries the full absolute (normalised) setpoint, a
# dropped message just means firmware holds the previous one for a cycle --
# nothing to resynchronise. Transport QoS is BEST_EFFORT / depth 1.
#
# Sign convention (matches scope_colon_env.py ctrl[px]=+cmd, ctrl[nx]=-cmd, and
# PULL_SIGN in the firmware):
#   cmd_x_n > 0  ->  pull servo 1's cable (+X), pay servo 3's out
#   cmd_x_n < 0  ->  pull servo 3's cable (-X)
#   cmd_y_n > 0  ->  pull servo 2's cable (+Z);  < 0 -> pull servo 4's (-Z)
# policy_node clips to [-1, 1] before publishing.
#
# Insertion feed is deliberately ABSENT: the rig is hand-fed, so the policy's
# third action is used only as observation echo inside policy_node.

std_msgs/Header header

uint32 seq          # monotonic. Firmware echoes the last applied seq in ScopeTelemetry.seq_echo.
float32 cmd_x_n     # {1,3} pair, -1..+1
float32 cmd_y_n     # {2,4} pair, -1..+1
