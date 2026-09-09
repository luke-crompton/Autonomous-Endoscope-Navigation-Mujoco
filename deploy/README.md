# deploy/ — the physical-rig control loop

The runtime stack that drives the real scope: **image → DA3 depth → APPO policy →
tendon setpoints → ESP32-S3**. Training / simulation lives in `navigation/`; this
directory is deployment only.

See `docs/architecture.md` ("The control loop") and `hardware/README.md`
("Constants the deployment loop must reproduce") for the contract, and
`docs/CURRENT_PLAN.md` §7 for where bring-up stands.

## Node graph

```
WINDOWS (no ROS, py3.11)          |  WSL2 UBUNTU (ROS 2 Humble)
camera idx 1 → DA3 → to_obs       |
  (1,54,96) f32, [0,1]            |
        │  TCP (TCP_NODELAY)      |
        └───────────────────────► depth_bridge ──/scope/depth───┐   sensor_msgs/Image 32FC1
                                                                 │
                                   /scope/telemetry ─────────────┤   scope_msgs/ScopeTelemetry
                                   /scope/estop (latched) ───────┤   std_msgs/Bool
                                   /scope/run_state (latched) ───┤   scope_msgs/ScopeRunState
                                                                 ▼
                                                          [ policy_node ]
                                                            APPO + GRU + PD command shaper
                                                                 │
                                                    /scope/action ▼   ScopeAction (normalised -1..1 per pair)
                                                          [ scope_link ] ──serial(COBS)──► ESP32-S3
                                                                 ▲                          runtime firmware
                                                                 └── /scope/telemetry, /scope/estop
```

No fisheye undistort — the 122°/100° lens gap is a known, accepted transfer risk
(`CURRENT_PLAN.md` §8). Insertion is hand-fed, so the policy's feed action never
reaches the motors; `policy_node` consumes it only as observation echo.

The steering command on the wire is **normalised** (±1 = full commanded bend).
The PD shaper runs normalised too — exactly matching the sim's `cmd / MAX_PULL`,
so nothing ROS-side needs `MAX_PULL`. The real per-axis-per-direction encoder
limits are **asymmetric** (measured: x +724/−681, z +840/−603 ticks —
`hardware/bringup/measured_constants.md`) and live only in the firmware, pushed
from `scope_params.yaml` via the `CONFIG` frame.

## Layout

| Path | State |
|---|---|
| `ros2_ws/src/scope_msgs/` | `ScopeAction`, `ScopeTelemetry`, `ScopeRunState`, `ScopeCommand` — **done** |
| `ros2_ws/src/scope_control/scope_control/policy_node.py` | inference + PD shaper + obs assembly — **done** |
| `ros2_ws/src/scope_control/scope_control/policy_runner.py` | SF-checkpoint wrapper (`reset()` / `act()`) — **done** |
| `ros2_ws/src/scope_control/scope_control/depth_bridge.py` | Windows TCP depth → `/scope/depth` — **done** |
| `ros2_ws/src/scope_control/scope_control/scope_link.py` | serial (COBS) ↔ ESP32-S3, derives `/scope/estop` — **done** |
| `ros2_ws/src/scope_control/scope_control/scope_link_proto.py` | canonical serial protocol (COBS + CRC-16) — **done** |
| `ros2_ws/src/scope_control/scope_control/mock_scope_link.py` | bench MCU stand-in for `policy_node` tests — **done** |
| `ros2_ws/src/scope_control/scope_control/fake_depth_pub.py` | synthetic `/scope/depth` for bench tests — **done** |
| `ros2_ws/src/scope_control/launch/bench.launch.py` | mock graph (mock policy + mock MCU) — **done** |
| `ros2_ws/src/scope_control/launch/rig.launch.py` | real graph (depth_bridge + policy_node + scope_link) — **done** |
| `ros2_ws/src/scope_control/config/scope_params.yaml` | the loop's constants — **done** |
| `win_vision/win_vision_server.py` | camera → DA3 → TCP (Windows, py3.11) — **done** |
| `tools/mock_firmware.py` | fake ESP32-S3 over a serial pty, for `scope_link` tests — **done** |
| `../hardware/firmware/scope_control_s3/` | ESP32-S3 runtime firmware (normalised→ticks, SYNC-WRITE pairs, safety governor, e-stop) — **done, not compile-tested here** |

**Whole loop is now drafted.** Remaining before a rig trial is calibration +
verification, not new code: verify `AXIS_SERVO` / `AXIS_INVERT` (which pair is X
vs Z, steering sign) by driving the tip; wire the e-stop MOSFET + tip sensor;
bench-measure `CURRENT_LSB_MA`; verify `PolicyRunner.act()` against
`viewer_sf.py --deterministic` to 1e-6. `MAX_PULL` is measured
(`scope_params.yaml`).

### Depth wire protocol (win_vision → depth_bridge)

TCP, `TCP_NODELAY`. One frame = 20-byte header + payload:

```
struct "<4sHHIQ" :  magic b"SDP1" | uint16 height | uint16 width | uint32 seq | uint64 sender_t_ns
payload          :  height*width*4 bytes, float32 little-endian, row-major, already [0,1]
```

`depth_bridge` validates `height×width` against `expected_*` params (drops with a
loud error on mismatch — `win_vision` owns the resize) and republishes as
`sensor_msgs/Image` `32FC1`.

### Serial protocol (scope_link ↔ ESP32-S3)

Canonical definition + a COBS/CRC-16 codec in `scope_link_proto.py` (`PROTO_VERSION 2`);
the firmware mirrors it in C++. Frames are `COBS(type | payload | crc16)` + `0x00`.
Host→MCU: `SETPOINT` (cmd_x_n, cmd_y_n ∈ [-1,1] + seq), `CONFIG` (safety limits +
rates + the 4 asymmetric MAX_PULL tick limits + headroom),
`COMMAND` (ENABLE/DISABLE/CLEAR_SAFETY/ESTOP), `PING`.
MCU→Host: `TELEMETRY` (seq echo, flags, tension[4], pos[4], current[4]),
`LOG`, `PONG`, `HELLO` (applied MAX_PULL + zeros).
`/scope/estop` is asserted by `scope_link` whenever the link is down, the
firmware reports ESTOP, or an operator ESTOP is outstanding.

## Build & run (WSL2)

```bash
cd deploy/ros2_ws
colcon build --packages-select scope_msgs scope_control
source install/setup.bash

# --- bench: whole graph, no rig, no checkpoint ---
ros2 launch scope_control bench.launch.py
ros2 run scope_control fake_depth_pub --ros-args -r __ns:=/scope
ros2 topic pub -1 /scope/run_state scope_msgs/msg/ScopeRunState '{command: 2}'   # RESET+RUN
ros2 topic hz /scope/action                                                     # expect ~25 Hz

# --- bench: exercise the real scope_link against a fake firmware ---
socat -d -d pty,raw,echo=0 pty,raw,echo=0          # note the two /dev/pts/N it prints
python ../tools/mock_firmware.py --port /dev/pts/B --spike-after 10
ros2 run scope_control scope_link --ros-args -r __ns:=/scope \
  --params-file src/scope_control/config/scope_params.yaml -p port:=/dev/pts/A
ros2 topic pub -1 /scope/command scope_msgs/msg/ScopeCommand '{command: 1}'      # ENABLE
#   -> /scope/telemetry flows; after 10 s the mock trips -> /scope/estop true

# --- rig: real inference + real serial ---
#   edit config/scope_params.yaml (nav_code_dir, checkpoint_root, port) first
ros2 launch scope_control rig.launch.py port:=/dev/ttyUSB0
#   then on Windows:  py -3.11 deploy/win_vision/win_vision_server.py --host <WSL_IP>
ros2 topic pub -1 /scope/command   scope_msgs/msg/ScopeCommand  '{command: 1}'   # ENABLE firmware
ros2 topic pub -1 /scope/run_state scope_msgs/msg/ScopeRunState '{command: 2}'   # RESET+RUN policy
```

## The one rule

`policy_node` reproduces the simulator's PD command shaper and previous-action
echo **exactly** — they are policy *inputs*, not just outputs. `pd_kp` / `pd_kd`
were fitted at 25 Hz and are not arithmetically rescalable. The GRU hidden state
is load-bearing and is zeroed only on `RESET`.
