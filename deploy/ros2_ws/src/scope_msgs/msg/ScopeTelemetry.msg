# Everything the ESP32-S3 reports back each cycle (~50 Hz), via scope_link.
#
# Only ONE field is consumed by the policy: tip_contact. Everything else is for
# the operator, the safety monitor, and rosbag logging. This matches the design
# rule in docs/architecture.md -- five of the six policy state values are
# software echo reproduced in policy_node, not measured on the rig.

std_msgs/Header header

# --- safety_state values ---------------------------------------------------
uint8 SAFETY_OK    = 0   # normal operation
uint8 SAFETY_CLAMP = 1   # firmware is holding one axis back below its tension/current limit
uint8 SAFETY_HOLD  = 2   # SAFE_HOLD: all servo goals frozen at present position, awaiting operator reset
uint8 SAFETY_ESTOP = 3   # hardware e-stop asserted (servo supply cut by the MOSFET)

# --- fields --------------------------------------------------------------
# All 4-arrays are SERVO-ID INDEXED: [0]=servo 1, [1]=servo 2, [2]=servo 3, [3]=servo 4.
# The HX711 tension channels are NOT wired in servo order (servo 1 -> GPIO10 ...
# servo 4 -> GPIO13, i.e. reversed) -- de-scrambling that is scope_link's /
# firmware's job so everything on the ROS side is plain servo order.
uint32 seq_echo               # last ScopeAction.seq the firmware actually applied (staleness / desync check)

bool tip_contact              # binary flex-sensor flag -- the only exogenous signal in the policy state

float32[4] tension_g          # cable tension per servo; no-load zero already subtracted
int32[4]   servo_pos_ticks    # present encoder position per servo, raw counts 0..4095
int32[4]   servo_current_ma   # present current per servo, milliamps (fast safety signal)

uint8 safety_state            # one of SAFETY_* above
