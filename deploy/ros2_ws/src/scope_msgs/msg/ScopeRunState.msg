# Operator command to policy_node. Publish it LATCHED
# (RELIABLE + TRANSIENT_LOCAL, depth 1) so a policy_node that starts after the
# operator does still receives the current desired state.

std_msgs/Header header

uint8 STOP  = 0   # stop acting. policy_node stops publishing; firmware watchdog then SAFE_HOLDs.
uint8 RUN   = 1   # act on every incoming depth frame.
uint8 RESET = 2   # zero the PD shaper, the GRU hidden state and the last-action echo, then RUN.
                  # Use at the start of every run -- it is the equivalent of the env's reset().

uint8 command
