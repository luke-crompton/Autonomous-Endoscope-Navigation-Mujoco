"""
mock_firmware.py -- pretends to be the ESP32-S3 runtime firmware, over a serial
port, so `scope_link` can be tested end to end with no hardware.

Speaks scope_control/scope_link_proto: sends HELLO once, streams TELEMETRY at
telem_hz, echoes the last SETPOINT's seq, replies PONG to PING, and honours
COMMAND (ENABLE / DISABLE / CLEAR_SAFETY / ESTOP). Optionally simulates a
current spike after N seconds so you can watch scope_link assert /scope/estop.

It does NOT model tip physics -- servo positions are just the mm setpoint
scaled by a mock mm_per_tick, so the numbers move.

Bench setup on WSL (make a linked pair of PTYs):
    socat -d -d pty,raw,echo=0 pty,raw,echo=0
    # -> prints e.g. /dev/pts/3  and  /dev/pts/4
    python deploy/tools/mock_firmware.py --port /dev/pts/4
    ros2 run scope_control scope_link --ros-args -r __ns:=/scope -p port:=/dev/pts/3

Requires: pip install pyserial
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "ros2_ws", "src", "scope_control", "scope_control"),
)

import scope_link_proto as p  # noqa: E402

try:
    import serial
except ImportError:
    raise SystemExit("pyserial not installed: pip install pyserial")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="serial port to open (the firmware end)")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--telem-hz", type=float, default=50.0)
    ap.add_argument("--mm-per-tick", type=float, default=0.02)
    ap.add_argument("--tip-contact", action="store_true")
    ap.add_argument("--spike-after", type=float, default=0.0,
                    help=">0: report a current spike -> SAFE_HOLD after this many seconds")
    ap.add_argument("--estop-after", type=float, default=0.0,
                    help=">0: assert ESTOP after this many seconds")
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=0.05)
    print(f"[mock_fw] open {args.port} @ {args.baud}")

    reader = p.FrameReader()
    ser.write(p.pack_frame(p.T_HELLO, p.S_HELLO.pack(p.PROTO_VERSION, 4, args.mm_per_tick,
                                                     2048, 2048, 2048, 2048)))

    enabled = False
    safety = 0                       # 0 OK, 2 HOLD, 3 ESTOP
    last_seq = 0
    cmd_x = cmd_y = 0.0
    t_start = time.monotonic()
    next_telem = t_start
    telem_period = 1.0 / max(args.telem_hz, 1.0)

    try:
        while True:
            data = ser.read(256)
            if data:
                for item in reader.feed(data):
                    if item[0] == "error":
                        print(f"[mock_fw] frame error: {item[1]}")
                        continue
                    mtype, payload = item
                    if mtype == p.T_SETPOINT:
                        cmd_x, cmd_y, last_seq = p.S_SETPOINT.unpack(payload)
                    elif mtype == p.T_COMMAND:
                        c = p.S_COMMAND.unpack(payload)[0]
                        if c == 1:      # ENABLE
                            enabled, safety = True, 0 if safety != 3 else 3
                        elif c == 0:    # DISABLE
                            enabled, safety = False, 2
                        elif c == 2:    # CLEAR_SAFETY
                            safety = 0
                            print("[mock_fw] safety cleared")
                        elif c == 3:    # ESTOP
                            safety, enabled = 3, False
                            print("[mock_fw] ESTOP asserted")
                        print(f"[mock_fw] command {c} -> enabled={enabled} safety={safety}")
                    elif mtype == p.T_CONFIG:
                        vals = p.S_CONFIG.unpack(payload)
                        print(f"[mock_fw] CONFIG {vals}")
                    elif mtype == p.T_PING:
                        ser.write(p.pack_frame(p.T_PONG, payload))

            now = time.monotonic()
            el = now - t_start
            if args.spike_after and safety == 0 and el >= args.spike_after:
                safety = 2
                ser.write(p.pack_frame(p.T_LOG, bytes([3]) + b"SAFE_HOLD: mock current spike"))
                print("[mock_fw] simulated current spike -> SAFE_HOLD")
            if args.estop_after and safety != 3 and el >= args.estop_after:
                safety, enabled = 3, False
                print("[mock_fw] simulated ESTOP")

            if now >= next_telem:
                next_telem += telem_period
                dx = int(round(cmd_x / args.mm_per_tick)) if enabled and safety == 0 else 0
                dy = int(round(cmd_y / args.mm_per_tick)) if enabled and safety == 0 else 0
                flags = 0
                if args.tip_contact and safety != 3:
                    flags |= p.FLAG_TIP_CONTACT
                flags |= (safety & p.FLAG_SAFETY_MASK) << p.FLAG_SAFETY_SHIFT
                if enabled:
                    flags |= p.FLAG_ENABLED
                payload = p.S_TELEMETRY.pack(
                    int(last_seq) & 0xFFFFFFFF, flags,
                    abs(cmd_x) * 8.0, abs(cmd_y) * 8.0, abs(cmd_x) * 8.0, abs(cmd_y) * 8.0,
                    2048 + dx, 2048 + dy, 2048 - dx, 2048 - dy,
                    abs(dx) * 3, abs(dy) * 3, abs(dx) * 3, abs(dy) * 3,
                    int(el * 1000) & 0xFFFFFFFF,
                )
                ser.write(p.pack_frame(p.T_TELEMETRY, payload))
            time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print("[mock_fw] closed")


if __name__ == "__main__":
    main()
