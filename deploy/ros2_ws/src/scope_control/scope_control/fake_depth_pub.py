"""
fake_depth_pub -- synthetic /scope/depth frames for bench testing.

Stands in for win_vision + depth_bridge when you have no camera / no Windows
side. Publishes a normalised (0..1) 32FC1 image at a fixed rate: a slowly
drifting radial gradient, so it looks vaguely lumen-shaped and changes over
time (which matters -- the GRU integrates across frames).

    ros2 run scope_control fake_depth_pub
    ros2 run scope_control fake_depth_pub --ros-args -p rate_hz:=25.0 -p height:=54 -p width:=96

Not for any accuracy claim -- it is not real depth. Use it to check the graph
spins, stays at rate, and that policy_node/mock_scope_link react.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image


SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class FakeDepthPub(Node):

    def __init__(self):
        super().__init__("fake_depth_pub")
        self.declare_parameter("rate_hz", 25.0)
        self.declare_parameter("height", 54)
        self.declare_parameter("width", 96)

        gp = self.get_parameter
        self._h = int(gp("height").value)
        self._w = int(gp("width").value)
        rate = max(float(gp("rate_hz").value), 1.0)

        self._yy, self._xx = np.mgrid[0:self._h, 0:self._w].astype(np.float32)

        self._pub = self.create_publisher(Image, "depth", SENSOR_QOS)
        self._k = 0
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f"fake_depth_pub: {self._h}x{self._w} at {rate:.0f} Hz")

    def _tick(self):
        self._k += 1
        # a "lumen" bright spot that drifts in a small circle so successive
        # frames differ (the GRU integrates across them)
        phase = self._k * 0.05
        cx = self._w / 2.0 + 12.0 * math.cos(phase)
        cy = self._h / 2.0 + 6.0 * math.sin(phase)
        r = np.sqrt(((self._xx - cx) / self._w) ** 2 + ((self._yy - cy) / self._h) ** 2)
        d = np.clip(1.0 - 2.0 * r, 0.0, 1.0).astype(np.float32)
        d /= max(float(d.max()), 1e-6)   # per-frame-max normalise, like Da3Depth.to_obs

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "scope_tip"
        msg.height = self._h
        msg.width = self._w
        msg.encoding = "32FC1"
        msg.is_bigendian = 0
        msg.step = self._w * 4
        msg.data = np.ascontiguousarray(d).tobytes()
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeDepthPub()
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
