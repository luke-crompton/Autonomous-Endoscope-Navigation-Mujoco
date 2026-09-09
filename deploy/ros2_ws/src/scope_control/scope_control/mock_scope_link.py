"""
mock_scope_link -- bench stand-in for scope_link + the ESP32-S3 firmware.

Lets the full graph run with no rig:

    depth_bridge --/scope/depth--> policy_node --/scope/action--> mock_scope_link
                                        ^                              |
                                        |    /scope/telemetry, /scope/estop
                                        +------------------------------+

It does NOT simulate any tip physics. It:
  - subscribes /scope/action, remembers the last one
  - publishes /scope/telemetry at a fixed rate, echoing the last action's seq
    and turning the pull setpoints into plausible-looking servo positions so
    you can eyeball that the loop is alive and the numbers move
  - publishes a latched /scope/estop (Bool), default False; can trip it after
    N seconds to check that policy_node freezes
  - can be told to report tip_contact so you can see the observation change

The sign / tick math here is deliberately simplistic -- the REAL mapping
(PULL_SIGN, synchronous SYNC WRITE of equal-and-opposite goals, MM_PER_TICK
from the drum radius) belongs in the firmware, not here.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Bool

from scope_msgs.msg import ScopeAction, ScopeTelemetry


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

_ZERO_TICK = 2048   # STS3032 centre, i.e. the homed zero after `setzero`


class MockScopeLink(Node):

    def __init__(self):
        super().__init__("mock_scope_link")

        self.declare_parameter("telemetry_rate_hz", 50.0)
        # mock per-direction tick limits (x+, x-, z+, z-), like the firmware's
        # applied MAX_PULL -- so servo_pos_ticks move asymmetrically
        self.declare_parameter("mock_maxpull_ticks", [652, 613, 756, 543])
        self.declare_parameter("tip_contact", False)        # report tip in contact
        self.declare_parameter("trip_estop_at_s", 0.0)      # >0 -> assert e-stop after this long
        self.declare_parameter("action_timeout_s", 0.5)     # no action for this long -> SAFETY_HOLD

        gp = self.get_parameter
        self._maxpull = [int(x) for x in gp("mock_maxpull_ticks").value]
        self._tip_contact = bool(gp("tip_contact").value)
        self._trip_at = float(gp("trip_estop_at_s").value)
        self._action_timeout = float(gp("action_timeout_s").value)

        self._last_action: ScopeAction | None = None
        self._last_action_stamp = None
        self._n_actions = 0
        self._estopped = False

        self._tele_pub = self.create_publisher(ScopeTelemetry, "telemetry", SENSOR_QOS)
        self._estop_pub = self.create_publisher(Bool, "estop", LATCHED_QOS)
        self.create_subscription(ScopeAction, "action", self._on_action, SENSOR_QOS)

        # publish the initial (clear) e-stop state once, latched
        self._publish_estop(False)

        rate = max(float(gp("telemetry_rate_hz").value), 1.0)
        self.create_timer(1.0 / rate, self._tick)
        self.create_timer(2.0, self._log_stats)
        self._start = self.get_clock().now()

        self.get_logger().info(
            f"mock_scope_link up: telemetry {rate:.0f} Hz, maxpull_ticks={self._maxpull}, "
            f"tip_contact={self._tip_contact}, trip_estop_at_s={self._trip_at}"
        )

    # --------------------------------------------------------- callbacks

    def _on_action(self, msg: ScopeAction):
        self._last_action = msg
        self._last_action_stamp = self.get_clock().now()
        self._n_actions += 1

    def _publish_estop(self, value: bool):
        self._estopped = value
        m = Bool()
        m.data = value
        self._estop_pub.publish(m)

    # ------------------------------------------------------------- loop

    def _tick(self):
        now = self.get_clock().now()

        if self._trip_at > 0.0 and not self._estopped:
            if (now - self._start).nanoseconds * 1e-9 >= self._trip_at:
                self.get_logger().error("mock: tripping E-STOP now")
                self._publish_estop(True)

        safety = ScopeTelemetry.SAFETY_OK
        if self._estopped:
            safety = ScopeTelemetry.SAFETY_ESTOP
        elif self._last_action_stamp is None:
            safety = ScopeTelemetry.SAFETY_HOLD
        elif (now - self._last_action_stamp).nanoseconds * 1e-9 > self._action_timeout:
            safety = ScopeTelemetry.SAFETY_HOLD

        cx_n = self._last_action.cmd_x_n if self._last_action else 0.0
        cy_n = self._last_action.cmd_y_n if self._last_action else 0.0
        seq_echo = self._last_action.seq if self._last_action else 0

        live = safety == ScopeTelemetry.SAFETY_OK
        dx = int(round(cx_n * (self._maxpull[0] if cx_n >= 0 else self._maxpull[1]))) if live else 0
        dy = int(round(cy_n * (self._maxpull[2] if cy_n >= 0 else self._maxpull[3]))) if live else 0

        msg = ScopeTelemetry()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "scope_base"
        msg.seq_echo = int(seq_echo)
        msg.tip_contact = self._tip_contact and not self._estopped
        # Pairs {1,3}=X, {2,4}=Z (flagged assumption).
        # One cable of each pair pulls (+delta), the other pays out (-delta).
        msg.servo_pos_ticks = [
            _ZERO_TICK + dx,   # servo 1  (X pair)
            _ZERO_TICK + dy,   # servo 2  (Z pair)
            _ZERO_TICK - dx,   # servo 3  (X pair, antagonist)
            _ZERO_TICK - dy,   # servo 4  (Z pair, antagonist)
        ]
        # rough "tension proportional to pull" so the numbers move; not physical
        msg.tension_g = [abs(cx_n) * 120.0, abs(cy_n) * 120.0,
                         abs(cx_n) * 120.0, abs(cy_n) * 120.0]
        msg.servo_current_ma = [abs(dx) // 4, abs(dy) // 4, abs(dx) // 4, abs(dy) // 4]
        msg.safety_state = safety
        self._tele_pub.publish(msg)

    def _log_stats(self):
        if self._last_action is not None:
            last = (f"last=({self._last_action.cmd_x_n:.3f}, "
                    f"{self._last_action.cmd_y_n:.3f})")
        else:
            last = "(no action yet)"
        self.get_logger().info(f"actions rx last 2s: {self._n_actions}  {last}")
        self._n_actions = 0


def main(args=None):
    rclpy.init(args=args)
    node = MockScopeLink()
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
