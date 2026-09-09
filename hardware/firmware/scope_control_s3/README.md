# scope_control_s3 — runtime control firmware

The **deployment** firmware for the rig ESP32-S3. Not interactive, not a
calibration tool — it is one half of the control loop:

```
scope_link (WSL ROS node)  <—UART 921600, COBS+CRC16—>  scope_control_s3
   /scope/action  → SETPOINT → [ mm → encoder ticks, SYNC WRITE the pairs ]
   /scope/command → ENABLE / DISABLE / CLEAR_SAFETY / ESTOP
                  ← TELEMETRY @ 50 Hz (seq echo, tip_contact, tension, pos, current, safety)
```

The protocol is defined once in
`deploy/ros2_ws/src/scope_control/scope_control/scope_link_proto.py` — this
sketch mirrors it in C++. Bump `PROTO_VERSION` in both on any change.

## What it does

| Concern | Behaviour |
|---|---|
| Setpoint → motion | normalised `cmd_x/y_n` ∈ [−1,1] → per-servo goal ticks via the **asymmetric per-direction MAX_PULL** limits, `PULL_SIGN`, `ZERO_TICK=2048`. Each antagonistic pair moved **equal-and-opposite in one SYNC WRITE** so the two servos start together. |
| Rate | servo goals streamed at `motor_hz` (200); firmware-side slew clamp `MAX_TICKS_PER_CYCLE` bounds tip speed regardless of what's commanded. |
| Safety governor | per-servo current (fast, from the bus) and per-cable tension (HX711) — over a configurable limit for `spike_debounce_ms` → **SAFE_HOLD** (freeze goals, keep torque). |
| E-stop | `ESTOP` command or a critical fault → `ESTOP_MOSFET_PIN` cuts the servo supply. Latching; `CLEAR_SAFETY` re-energises and returns to SAFE_HOLD. |
| Comms watchdog | no SETPOINT for `comms_timeout_ms` → SAFE_HOLD (auto-recovers when frames resume, if still ENABLEd). A safety trip needs an explicit `CLEAR_SAFETY`. |
| tip_contact | ADC threshold on the flex sensor; **forced to 0** until the sensor is wired (`TIP_CONTACT_FORCE_ZERO`). |

Boot state is **SAFE_HOLD** — servos hold their present position, setpoints are
ignored until `/scope/command` `ENABLE`.

## Board / upload

- Board **"ESP32S3 Dev Module"**, **USB CDC On Boot = Disabled** (so `Serial` is
  the CP2102 UART jack = the scope_link link).
- **Flash via the NATIVE-USB jack**; leave the CP2102 jack forwarded to WSL
  (`usbipd`) for the protocol.
- No serial monitor during runtime — output is binary framed. Human-readable
  events come out as `T_LOG` frames and surface in the `scope_link` ROS log.

## Before it can drive the real scope — set these (all marked TODO in the file)

| Constant | Source |
|---|---|
| `AXIS_SERVO`, `AXIS_INVERT` | which pair is cmd_x vs cmd_y, and steering sign — watch the camera image and flip |
| `ESTOP_MOSFET_PIN`, `ESTOP_ENABLE_LEVEL` | the actual e-stop switch wiring |
| `CURRENT_LSB_MA` | STS3032 REG 69 scale (bench-measure against a known load) |
| `TIP_CONTACT_PIN` + threshold, `TIP_CONTACT_FORCE_ZERO` | once the flex sensor is fitted |
| `HX_RATE_PIN` | set if you wire HX711 RATE high for 80 SPS (else tension is ~10 Hz) |

`MAXPULL_DEFAULT` is the measured `bend` result (x +724/−681, z +840/−603 ticks,
2026-09-09) — `scope_link` overrides it at runtime from `scope_params.yaml`
(`maxpull_*_ticks`, `maxpull_headroom`), so it need not be exact.

## Testing without hardware

`deploy/tools/mock_firmware.py` is the inverse of this sketch — it lets you test
`scope_link` with no ESP32. To test *this* sketch, flash it and drive it from
`scope_link` (or a scratch Python client using `scope_link_proto`).
