"""
Wire-protocol round-trip and framing-robustness checks for scope_link_proto.

    python -m pytest deploy/ros2_ws/src/scope_control/test/test_scope_link_proto.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scope_control"))

import scope_link_proto as p  # noqa: E402


def test_cobs_roundtrip_edge_cases():
    for data in [b"", b"\x00", b"\x00\x00\x00", bytes(range(256)),
                 b"\xff" * 300, b"\x01" * 254, b"\x00" * 254 + b"\x01"]:
        enc = p.cobs_encode(data)
        assert 0 not in enc  # COBS output is zero-free
        assert p.cobs_decode(enc) == data


def test_frame_roundtrip_all_types():
    cases = [
        (p.T_SETPOINT, p.S_SETPOINT.pack(0.42, -0.71, 12345)),
        (p.T_CONFIG, p.S_CONFIG.pack(800.0, 250.0, 20, 120, 200, 50,
                                     724, 681, 840, 603, 0.9)),
        (p.T_COMMAND, p.S_COMMAND.pack(3)),
        (p.T_PING, p.S_PING.pack(0xDEADBEEF)),
        (p.T_TELEMETRY, p.S_TELEMETRY.pack(
            42, 0b0000_1011,
            1.0, 2.0, 3.0, 4.0,
            2048, 2050, 2046, 2049,
            10, 20, 15, 12,
            999999)),
        (p.T_HELLO, p.S_HELLO.pack(p.PROTO_VERSION, 4, 652, 613, 756, 543,
                                   2048, 2048, 2048, 2048)),
    ]
    for mtype, payload in cases:
        frame = p.pack_frame(mtype, payload)
        assert frame[-1] == p.DELIM
        assert frame.count(0) == 1  # only the trailing delimiter
        t, pl = p.unpack_frame(frame[:-1])
        assert t == mtype
        assert pl == payload


def test_crc_rejects_bit_flip():
    frame = p.pack_frame(p.T_SETPOINT, p.S_SETPOINT.pack(1.0, 1.0, 1))
    body = bytearray(p.cobs_decode(frame[:-1]))
    body[1] ^= 0x01  # corrupt a payload byte
    corrupted = p.cobs_encode(bytes(body))
    try:
        p.unpack_frame(corrupted)
        assert False, "CRC should have rejected the corrupted frame"
    except ValueError:
        pass


def test_framereader_fragmented_and_corrupt():
    a = p.pack_frame(p.T_PING, p.S_PING.pack(1))
    b = p.pack_frame(p.T_PONG, p.S_PONG.pack(1))
    garbage = b"\x03\x11\x22\x00"  # decodes to 2 bytes -> "frame too short"
    stream = a + garbage + b

    reader = p.FrameReader()
    out = []
    # feed one byte at a time to exercise buffering
    for i in range(len(stream)):
        out.extend(reader.feed(stream[i:i + 1]))

    good = [x for x in out if x[0] != "error"]
    errs = [x for x in out if x[0] == "error"]
    assert [g[0] for g in good] == [p.T_PING, p.T_PONG]
    assert len(errs) == 1


def test_telemetry_flag_helpers():
    flags = p.FLAG_TIP_CONTACT | (2 << p.FLAG_SAFETY_SHIFT) | p.FLAG_ENABLED
    assert flags & p.FLAG_TIP_CONTACT
    assert (flags >> p.FLAG_SAFETY_SHIFT) & p.FLAG_SAFETY_MASK == 2
    assert flags & p.FLAG_ENABLED
