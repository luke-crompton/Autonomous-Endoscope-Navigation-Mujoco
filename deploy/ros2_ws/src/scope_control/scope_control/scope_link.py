"""
scope_link -- serial bridge between the ROS graph and the ESP32-S3 firmware.

    /scope/action   (ScopeAction)   --> SETPOINT frames -----> ESP32-S3
    /scope/command  (ScopeCommand)  --> COMMAND / CONFIG ----> ESP32-S3
                                    <-- TELEMETRY frames ----- ESP32-S3
    /scope/telemetry (ScopeTelemetry)  published from those
    /scope/estop     (std_msgs/Bool, latched)  the single "is it safe" signal

This is the plain-serial alternative to micro-ROS (see the deployment memory /
the ROS 2 curriculum): a COBS/CRC binary protocol over pyserial. Everything
downstream still lands on named topics, and rosbag / `ros2 topic echo` work as
normal -- the payoff of the graph without the micro-ROS toolchain.

WHY /scope/estop IS DERIVED HERE
-------------------------------
policy_node freezes on /scope/estop. scope_link owns that signal and asserts it
(true) whenever ANY of these hold:
  - the serial link is down (firmware unreachable)
  - the firmware reports safety_state == ESTOP
  - an operator ESTOP command has not been cleared
It only publishes false when connected AND the firmware is not in ESTOP AND no
operator ESTOP is outstanding. Fail-safe: before the first connection it is true.

WHAT LIVES IN FIRMWARE, NOT HERE
-------------------------------
The mm -> encoder-tick conversion, PULL_SIGN, the synchronous equal-and-opposite
SYNC WRITE of each antagonistic pair, the current/tension safety governor, and
the e-stop MOSFET. scope_link just ships the mm setpoint and the config numbers.
"""

from __future__ import annotations

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Bool

from scope_msgs.msg import ScopeAction, ScopeCommand, ScopeTelemetry

from scope_control import scope_link_proto as proto

try:
    import serial  # pyserial
except ImportError:  # pragma: no cover
    serial = None


SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)
LATCHED_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

_LOG_FN = {0: "debug", 1: "info", 2: "warn", 3: "error"}


class ScopeLink(Node):

    def __init__(self):
        super().__init__("scope_link")

        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baud", 921600)
        self.declare_parameter("reconnect_period_s", 1.0)
        self.declare_parameter("enable_on_connect", False)

        # Runtime-tunable safety config pushed to the firmware on every connect
        # (the firmware also has compiled-in defaults). These are the "variable
        # that can be changed" the design called for.
        self.declare_parameter("current_limit_ma", 800.0)
        self.declare_parameter("tension_limit_g", 250.0)
        self.declare_parameter("spike_debounce_ms", 20)
        self.declare_parameter("comms_timeout_ms", 120)
        self.declare_parameter("motor_hz", 200)
        self.declare_parameter("telem_hz", 50)

        # Asymmetric per-axis-per-direction encoder limits (ticks), measured with
        # `bend <x|z>` -- hardware/bringup/measured_constants.md. Firmware maps the
        # normalised command to ticks with these. maxpull_headroom scales them
        # down (bend was measured tip-free; friction in the housing reduces it).
        self.declare_parameter("maxpull_x_pos_ticks", 724)
        self.declare_parameter("maxpull_x_neg_ticks", 681)
        self.declare_parameter("maxpull_z_pos_ticks", 840)
        self.declare_parameter("maxpull_z_neg_ticks", 603)
        self.declare_parameter("maxpull_headroom", 0.9)

        gp = self.get_parameter
        self._port = str(gp("port").value)
        self._baud = int(gp("baud").value)
        self._reconnect = float(gp("reconnect_period_s").value)
        self._enable_on_connect = bool(gp("enable_on_connect").value)
        self._comms_timeout_s = int(gp("comms_timeout_ms").value) / 1000.0

        if serial is None:
            self.get_logger().fatal("pyserial not installed (apt: python3-serial / pip: pyserial)")
            raise RuntimeError("pyserial missing")

        # --- I/O ---
        self._tele_pub = self.create_publisher(ScopeTelemetry, "telemetry", SENSOR_QOS)
        self._estop_pub = self.create_publisher(Bool, "estop", LATCHED_QOS)
        self.create_subscription(ScopeAction, "action", self._on_action, SENSOR_QOS)
        self.create_subscription(ScopeCommand, "command", self._on_command, LATCHED_QOS)

        # --- state ---
        self._ser = None
        self._wlock = threading.Lock()
        self._stop = threading.Event()
        self._connected = False
        self._mcu_estop = False
        self._operator_estop = False
        self._estop_val = None            # last published value (None => not yet)
        self._last_rx = 0.0
        self._ping_token = 0
        self._n_setpoints = 0
        self._n_telem = 0
        self._hello_seen = False

        self._publish_estop()             # assert true (not connected yet)

        self._reader = proto.FrameReader()
        self._thread = threading.Thread(target=self._serial_loop, name="scope_link_serial", daemon=True)
        self._thread.start()

        self.create_timer(0.5, self._ping_tick)
        self.create_timer(2.0, self._log_stats)

        self.get_logger().info(f"scope_link: {self._port} @ {self._baud}, proto v{proto.PROTO_VERSION}")

    # ------------------------------------------------------------ shutdown

    def destroy_node(self):
        self._stop.set()
        try:
            self._thread.join(timeout=2.0)
        except RuntimeError:
            pass
        return super().destroy_node()

    # -------------------------------------------------------- ROS -> serial

    def _on_action(self, msg: ScopeAction):
        payload = proto.S_SETPOINT.pack(
            float(msg.cmd_x_n), float(msg.cmd_y_n), int(msg.seq) & 0xFFFFFFFF
        )
        if self._write(proto.T_SETPOINT, payload):
            self._n_setpoints += 1

    def _on_command(self, msg: ScopeCommand):
        c = int(msg.command)
        if c == ScopeCommand.ESTOP:
            self._operator_estop = True
        elif c == ScopeCommand.CLEAR_SAFETY:
            self._operator_estop = False
        self._write(proto.T_COMMAND, proto.S_COMMAND.pack(c & 0xFF))
        self.get_logger().info(f"command -> firmware: {c}")
        self._publish_estop()

    def _send_config(self):
        gp = self.get_parameter
        payload = proto.S_CONFIG.pack(
            float(gp("current_limit_ma").value),
            float(gp("tension_limit_g").value),
            int(gp("spike_debounce_ms").value) & 0xFFFF,
            int(gp("comms_timeout_ms").value) & 0xFFFF,
            int(gp("motor_hz").value) & 0xFFFF,
            int(gp("telem_hz").value) & 0xFFFF,
            int(gp("maxpull_x_pos_ticks").value) & 0xFFFF,
            int(gp("maxpull_x_neg_ticks").value) & 0xFFFF,
            int(gp("maxpull_z_pos_ticks").value) & 0xFFFF,
            int(gp("maxpull_z_neg_ticks").value) & 0xFFFF,
            float(gp("maxpull_headroom").value),
        )
        self._write(proto.T_CONFIG, payload)

    def _ping_tick(self):
        if not self._connected:
            return
        self._ping_token = (self._ping_token + 1) & 0xFFFFFFFF
        self._write(proto.T_PING, proto.S_PING.pack(self._ping_token))

    def _write(self, msg_type: int, payload: bytes) -> bool:
        ser = self._ser
        if ser is None:
            return False
        frame = proto.pack_frame(msg_type, payload)
        with self._wlock:
            try:
                ser.write(frame)
                return True
            except Exception as e:  # noqa: BLE001 - serial errors vary by platform
                self.get_logger().error(f"serial write failed: {e}")
                return False

    # -------------------------------------------------------- serial -> ROS

    def _serial_loop(self):
        while not self._stop.is_set():
            try:
                ser = serial.Serial(self._port, self._baud, timeout=0.1)
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"open {self._port} failed: {e}; retry in {self._reconnect:.1f}s")
                self._set_connected(False)
                self._stop.wait(self._reconnect)
                continue

            self._ser = ser
            self._reader = proto.FrameReader()
            self._hello_seen = False
            self._last_rx = time.monotonic()
            self._set_connected(True)
            self.get_logger().info(f"serial open: {self._port}")
            self._send_config()
            if self._enable_on_connect:
                self._write(proto.T_COMMAND, proto.S_COMMAND.pack(ScopeCommand.ENABLE))

            try:
                while not self._stop.is_set():
                    data = ser.read(256)
                    now = time.monotonic()
                    if data:
                        for item in self._reader.feed(data):
                            if item[0] == "error":
                                self.get_logger().warn(f"frame error: {item[1]}")
                                continue
                            self._last_rx = now
                            self._dispatch(item[0], item[1])
                    if now - self._last_rx > self._comms_timeout_s * 3.0:
                        self.get_logger().error("no telemetry from firmware -- reconnecting")
                        break
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"serial read error: {e}")
            finally:
                try:
                    ser.close()
                except Exception:
                    pass
                self._ser = None
                self._set_connected(False)

    def _dispatch(self, msg_type: int, payload: bytes):
        if msg_type == proto.T_TELEMETRY:
            self._on_telemetry_frame(payload)
        elif msg_type == proto.T_LOG:
            level = payload[0] if payload else 1
            text = payload[1:].decode("utf-8", "replace")
            getattr(self.get_logger(), _LOG_FN.get(level, "info"))(f"[fw] {text}")
        elif msg_type == proto.T_PONG:
            pass  # liveness only; _last_rx already bumped
        elif msg_type == proto.T_HELLO:
            ver, nsrv, xp, xn, zp, zn, z0, z1, z2, z3 = proto.S_HELLO.unpack(payload)
            self._hello_seen = True
            self.get_logger().info(
                f"firmware HELLO: proto v{ver}, {nsrv} servos, "
                f"applied MAX_PULL ticks x+{xp} x-{xn} z+{zp} z-{zn}, "
                f"zeros=[{z0},{z1},{z2},{z3}]"
            )
            if ver != proto.PROTO_VERSION:
                self.get_logger().error(
                    f"PROTOCOL MISMATCH: firmware v{ver}, scope_link v{proto.PROTO_VERSION}"
                )
        else:
            self.get_logger().warn(f"unknown frame type {msg_type:#04x}")

    def _on_telemetry_frame(self, payload: bytes):
        try:
            vals = proto.S_TELEMETRY.unpack(payload)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"bad TELEMETRY payload: {e}")
            return
        seq_echo = vals[0]
        flags = vals[1]
        tension = vals[2:6]
        pos = vals[6:10]
        cur = vals[10:14]
        # vals[14] = mcu_uptime_ms  (unused for now)

        safety = (flags >> proto.FLAG_SAFETY_SHIFT) & proto.FLAG_SAFETY_MASK

        msg = ScopeTelemetry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "scope_base"
        msg.seq_echo = int(seq_echo)
        msg.tip_contact = bool(flags & proto.FLAG_TIP_CONTACT)
        msg.tension_g = [float(x) for x in tension]
        msg.servo_pos_ticks = [int(x) for x in pos]
        msg.servo_current_ma = [int(x) for x in cur]
        msg.safety_state = int(safety)
        self._tele_pub.publish(msg)
        self._n_telem += 1

        mcu_estop = (safety == ScopeTelemetry.SAFETY_ESTOP)
        if mcu_estop != self._mcu_estop:
            self._mcu_estop = mcu_estop
            self._publish_estop()

    # ------------------------------------------------------------ estop

    def _set_connected(self, value: bool):
        if value == self._connected:
            return
        self._connected = value
        self.get_logger().info(f"serial connected={value}")
        self._publish_estop()

    def _publish_estop(self):
        est = (not self._connected) or self._mcu_estop or self._operator_estop
        if est == self._estop_val:
            return
        self._estop_val = est
        m = Bool()
        m.data = est
        self._estop_pub.publish(m)
        (self.get_logger().error if est else self.get_logger().info)(
            f"/scope/estop -> {est}"
        )

    # ------------------------------------------------------------ stats

    def _log_stats(self):
        self.get_logger().info(
            f"setpoints tx {self._n_setpoints/2:.0f}/s  telemetry rx {self._n_telem/2:.0f}/s  "
            f"connected={self._connected} estop={self._estop_val}"
        )
        self._n_setpoints = 0
        self._n_telem = 0


def main(args=None):
    rclpy.init(args=args)
    node = ScopeLink()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
