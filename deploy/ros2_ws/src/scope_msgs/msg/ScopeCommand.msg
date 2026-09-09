# Supervisor command to scope_link -> the ESP32-S3 runtime firmware.
# Distinct from ScopeRunState, which controls policy_node. Publish it LATCHED
# (RELIABLE + TRANSIENT_LOCAL, depth 1).

std_msgs/Header header

uint8 DISABLE      = 0   # firmware SAFE_HOLD: freeze servo goals at present position, ignore setpoints
uint8 ENABLE       = 1   # follow /scope/action setpoints
uint8 CLEAR_SAFETY = 2   # leave SAFE_HOLD / CLAMP and resume -- only takes effect if the cause has cleared
uint8 ESTOP        = 3   # assert the hardware e-stop (cut the servo supply via the MOSFET); latching in firmware

uint8 command
