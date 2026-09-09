"""
Serial wire protocol between scope_link (WSL, this package) and the ESP32-S3
runtime firmware (hardware/firmware/scope_control_s3/, not written yet).

This file is the CANONICAL definition. The firmware mirrors it in C++ -- keep
the two in lockstep and bump PROTO_VERSION on any change.

FRAMING
-------
UART, 921600 8N1. Frames are COBS-encoded and delimited by a single 0x00 byte:

    <COBS( msg_type[1] | payload[...] | crc16[2] LE )>  0x00

    crc16 = CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over msg_type+payload.

COBS guarantees the payload contains no 0x00, so the delimiter is unambiguous
and a receiver can always resync on the next 0x00. The CRC catches the rare
corrupted frame -- this link carries the safety loop, so a garbled setpoint
must be dropped, not applied.

MESSAGE TYPES
-------------
Host -> MCU:
  0x01 SETPOINT  <ffI>        cmd_x_pair_mm, cmd_y_pair_mm, seq
  0x02 CONFIG    <ffHHHH>     current_limit_ma, tension_limit_g,
                              spike_debounce_ms, comms_timeout_ms,
                              motor_hz, telem_hz
  0x03 COMMAND   <B>          command (see scope_msgs/ScopeCommand)
  0x04 PING      <I>          token

MCU -> Host:
  0x81 TELEMETRY <IB4f4h4hI>  seq_echo, flags, tension_g[4], servo_pos_ticks[4],
                              servo_current_ma[4], mcu_uptime_ms
                              flags: bit0 tip_contact, bits1-2 safety_state (0..3),
                                     bit3 enabled
                              all 4-arrays are SERVO-ID order [servo1..servo4];
                              firmware de-scrambles the reversed HX711 wiring.
  0x82 LOG       <B> + utf8   level (0 debug,1 info,2 warn,3 error), message text
  0x84 PONG      <I>          token echo
  0x85 HELLO     <BBf4h>      proto_version, num_servos, mm_per_tick, zero_ticks[4]
                              sent once on boot
"""

from __future__ import annotations

import struct

PROTO_VERSION = 1
DELIM = 0x00

# ---- message types -----------------------------------------------------
T_SETPOINT = 0x01
T_CONFIG = 0x02
T_COMMAND = 0x03
T_PING = 0x04

T_TELEMETRY = 0x81
T_LOG = 0x82
T_PONG = 0x84
T_HELLO = 0x85

# ---- payload layouts --------------------------------------------------
S_SETPOINT = struct.Struct("<ffI")
S_CONFIG = struct.Struct("<ffHHHH")
S_COMMAND = struct.Struct("<B")
S_PING = struct.Struct("<I")
S_TELEMETRY = struct.Struct("<IB4f4h4hI")
S_PONG = struct.Struct("<I")
S_HELLO = struct.Struct("<BBf4h")

# telemetry flag bits
FLAG_TIP_CONTACT = 0x01
FLAG_SAFETY_SHIFT = 1
FLAG_SAFETY_MASK = 0x03
FLAG_ENABLED = 0x08


# ---- CRC-16/CCITT-FALSE ---------------------------------------------
def crc16(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


# ---- COBS ----------------------------------------------------------
def cobs_encode(data: bytes) -> bytes:
    out = bytearray([0])          # placeholder for the first code byte
    code_i = 0
    code = 1
    for byte in data:
        if byte != 0:
            out.append(byte)
            code += 1
            if code == 0xFF:
                out[code_i] = code
                code_i = len(out)
                out.append(0)
                code = 1
        else:
            out[code_i] = code
            code_i = len(out)
            out.append(0)
            code = 1
    out[code_i] = code
    return bytes(out)


def cobs_decode(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        code = data[i]
        i += 1
        if code == 0:
            raise ValueError("COBS: unexpected 0x00 code byte")
        for _ in range(code - 1):
            if i >= n:
                raise ValueError("COBS: truncated frame")
            out.append(data[i])
            i += 1
        if code != 0xFF and i < n:
            out.append(0)
    return bytes(out)


# ---- frame pack / unpack -------------------------------------------
def pack_frame(msg_type: int, payload: bytes = b"") -> bytes:
    """type + payload -> a full on-wire frame including the trailing 0x00."""
    body = bytes([msg_type]) + payload
    body += struct.pack("<H", crc16(body))
    return cobs_encode(body) + bytes([DELIM])


def unpack_frame(segment: bytes) -> tuple[int, bytes]:
    """One COBS segment (no delimiter) -> (msg_type, payload).

    Raises ValueError on a CRC failure or a malformed frame -- the caller
    drops it and waits for the next.
    """
    body = cobs_decode(segment)
    if len(body) < 3:
        raise ValueError("frame too short")
    got = struct.unpack("<H", body[-2:])[0]
    want = crc16(body[:-2])
    if got != want:
        raise ValueError(f"CRC mismatch: got {got:#06x} want {want:#06x}")
    return body[0], body[1:-2]


class FrameReader:
    """Feed it raw serial bytes; iterate decoded (msg_type, payload) frames.
    Silently skips corrupt frames (logs are the caller's job via the return
    of `feed`)."""

    def __init__(self, max_buffer: int = 8192):
        self._buf = bytearray()
        self._max = max_buffer

    def feed(self, data: bytes):
        """Yields (msg_type, payload) for each complete, CRC-valid frame.
        Also yields ('error', reason) tuples for frames that failed to decode."""
        self._buf.extend(data)
        if len(self._buf) > self._max:
            self._buf.clear()
            yield ("error", "buffer overflow -- resync")
            return
        while True:
            idx = self._buf.find(DELIM)
            if idx < 0:
                return
            seg = bytes(self._buf[:idx])
            del self._buf[: idx + 1]
            if not seg:
                continue
            try:
                yield unpack_frame(seg)
            except ValueError as e:
                yield ("error", str(e))
