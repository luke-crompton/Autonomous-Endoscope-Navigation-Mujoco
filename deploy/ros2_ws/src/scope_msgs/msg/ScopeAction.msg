# Steering setpoint for the endoscope tip. Produced by policy_node once per
# incoming depth frame (~25 Hz), consumed by scope_link -> ESP32-S3 firmware.
#
# ABSOLUTE pull setpoints in MILLIMETRES, one per antagonistic tendon pair.
# Not rates. Not normalised. These are the output of the PD command shaper,
# which lives in policy_node (not firmware) because its own output is fed back
# as two of the six policy observations -- so it must sit inside the inference
# loop, on one side of the wire, stepped once per decision.
#
# Because every message carries the full absolute setpoint (not a delta), a
# dropped message just means firmware holds the previous setpoint for one
# cycle. There is nothing to resynchronise -- this is why the transport QoS is
# BEST_EFFORT / depth 1 (newest wins, no stale queue).
#
# Sign convention -- matches scope_colon_env.py step():  ctrl[px] = +cmd_x,
# ctrl[nx] = -cmd_x  (and the same for the Z pair):
#   cmd_x_pair_mm > 0  ->  pull the +X cable in by this many mm AND pay the
#                          -X cable out by the same amount, synchronously
#                          (one Feetech SYNC WRITE, identical speed/accel).
#   cmd_y_pair_mm > 0  ->  same for the +Z / -Z pair.
# Magnitude is clipped to +/- MAX_PULL for that axis by policy_node before
# publishing (MAX_PULL_X and MAX_PULL_Z differ -- 12 vs 13 driven joints).
#
# Insertion feed is deliberately ABSENT: the rig is hand-fed, so the policy's
# third action is used only as observation echo inside policy_node and never
# reaches the motors.

std_msgs/Header header

uint32 seq              # monotonic. Firmware echoes the last applied seq in ScopeTelemetry.seq_echo.
float32 cmd_x_pair_mm   # {1,3} pair, signed
float32 cmd_y_pair_mm   # {2,4} pair, signed
