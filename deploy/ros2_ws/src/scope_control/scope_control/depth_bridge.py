"""
depth_bridge -- Windows DA3 depth stream  ->  /scope/depth

Windows runs the camera and DA3 (no ROS -- see the deployment-architecture
memory / docs). It ships finished depth observations over a plain TCP socket.
This node is the WSL-side receiver: it turns each frame into a
`sensor_msgs/Image` on `/scope/depth` so everything downstream
(`policy_node`, `ros2 topic echo`, `ros2 bag record`) sees a normal topic.

It does NOT process the image. The obs contract -- 54x96, per-frame-max
normalised to [0, 1] -- is owned upstream by `win_vision` (`Da3Depth.to_obs`).
This node only validates the shape at the boundary and forwards the bytes.

WIRE PROTOCOL  (must match deploy/win_vision/win_vision_server.py)
----------------------------------------------------------------
TCP, TCP_NODELAY, one frame = fixed 20-byte header + payload:

    struct  "<4sHHIQ"
      4s   magic   b"SDP1"
      H    height  (rows)
      H    width   (cols)
      I    seq     uint32, monotonic from the sender
      Q    t_ns    uint64, sender's wall clock in ns (informational)
    payload  height*width*4 bytes, float32 little-endian, row-major

Rationale for TCP not UDP: the payload (20 KB) fragments across any MTU and one
lost fragment kills the datagram; loopback / mirrored-net loss is ~0 so TCP
head-of-line blocking never fires. Absolute frames that supersede => the ROS
side still publishes BEST_EFFORT depth 1 so a slow consumer cannot build a
backlog.
"""

from __future__ import annotations

import socket
import struct
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image


_HEADER = struct.Struct("<4sHHIQ")
_MAGIC = b"SDP1"

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


def _recv_exact(conn: socket.socket, n: int) -> bytes | None:
    """Read exactly n bytes, or None if the peer closed / errored partway."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        try:
            k = conn.recv_into(view[got:], n - got)
        except (BlockingIOError, InterruptedError):
            continue
        except OSError:
            return None
        if k == 0:
            return None
        got += k
    return bytes(buf)


class DepthBridge(Node):

    def __init__(self):
        super().__init__("depth_bridge")

        self.declare_parameter("tcp_host", "0.0.0.0")
        self.declare_parameter("tcp_port", 5599)
        self.declare_parameter("expected_height", 54)
        self.declare_parameter("expected_width", 96)
        self.declare_parameter("frame_id", "scope_tip")
        # strict_shape=true  -> drop + error on any frame that is not exactly
        #   expected_height x expected_width (a wrong shape means win_vision is
        #   misconfigured; fail loud at the boundary).
        # strict_shape=false -> forward whatever arrives (policy_node will still
        #   reject it, but this lets you debug the transport in isolation).
        self.declare_parameter("strict_shape", True)

        gp = self.get_parameter
        self._host = str(gp("tcp_host").value)
        self._port = int(gp("tcp_port").value)
        self._eh = int(gp("expected_height").value)
        self._ew = int(gp("expected_width").value)
        self._frame_id = str(gp("frame_id").value)
        self._strict = bool(gp("strict_shape").value)

        self._pub = self.create_publisher(Image, "depth", SENSOR_QOS)

        self._stop = threading.Event()
        self._n_frames = 0
        self._n_dropped = 0
        self._last_seq = None
        self._connected = False

        self._thread = threading.Thread(target=self._serve, name="depth_bridge_tcp", daemon=True)
        self._thread.start()

        self.create_timer(5.0, self._log_stats)
        self.get_logger().info(
            f"depth_bridge listening on {self._host}:{self._port}, "
            f"expecting {self._eh}x{self._ew} 32FC1 (strict_shape={self._strict})"
        )

    # ---------------------------------------------------------- shutdown

    def destroy_node(self):
        self._stop.set()
        try:
            # nudge accept() out of its timeout wait quickly
            self._thread.join(timeout=2.0)
        except RuntimeError:
            pass
        return super().destroy_node()

    # -------------------------------------------------------- tcp thread

    def _serve(self):
        while not self._stop.is_set():
            srv = None
            try:
                srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.settimeout(1.0)
                srv.bind((self._host, self._port))
                srv.listen(1)
                while not self._stop.is_set():
                    try:
                        conn, addr = srv.accept()
                    except socket.timeout:
                        continue
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    conn.settimeout(2.0)
                    self._connected = True
                    self.get_logger().info(f"win_vision connected from {addr}")
                    try:
                        self._pump(conn)
                    finally:
                        self._connected = False
                        try:
                            conn.close()
                        except OSError:
                            pass
                        self.get_logger().warn("win_vision disconnected -- waiting for reconnect")
            except OSError as e:
                if not self._stop.is_set():
                    self.get_logger().error(f"listen socket error: {e} -- retrying in 1s")
                    time.sleep(1.0)
            finally:
                if srv is not None:
                    try:
                        srv.close()
                    except OSError:
                        pass

    def _pump(self, conn: socket.socket):
        while not self._stop.is_set():
            hdr = _recv_exact(conn, _HEADER.size)
            if hdr is None:
                return
            magic, h, w, seq, t_ns = _HEADER.unpack(hdr)
            if magic != _MAGIC:
                self.get_logger().error(f"bad frame magic {magic!r} -- dropping connection")
                return
            payload = _recv_exact(conn, h * w * 4)
            if payload is None:
                return

            if (h, w) != (self._eh, self._ew):
                self._n_dropped += 1
                if self._strict:
                    if self._n_dropped <= 3 or self._n_dropped % 100 == 0:
                        self.get_logger().error(
                            f"frame is {h}x{w}, expected {self._eh}x{self._ew} "
                            "-- fix win_vision (owns the resize). Dropped."
                        )
                    continue
                self.get_logger().warn(f"forwarding non-standard {h}x{w} frame (strict_shape=false)")

            if self._last_seq is not None and seq != self._last_seq + 1:
                # not fatal -- absolute frames supersede -- but worth seeing
                self.get_logger().warn(f"seq jump {self._last_seq} -> {seq}")
            self._last_seq = seq

            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self._frame_id
            msg.height = h
            msg.width = w
            msg.encoding = "32FC1"
            msg.is_bigendian = 0
            msg.step = w * 4
            msg.data = payload
            self._pub.publish(msg)
            self._n_frames += 1
            _ = t_ns  # reserved for a sender-vs-receiver latency check later

    # ------------------------------------------------------------ stats

    def _log_stats(self):
        if not self._connected and self._n_frames == 0:
            return
        fps = self._n_frames / 5.0
        self.get_logger().info(
            f"depth {fps:.1f} fps  (total {self._n_frames}, dropped {self._n_dropped}, "
            f"connected={self._connected})"
        )
        self._n_frames = 0


def main(args=None):
    rclpy.init(args=args)
    node = DepthBridge()
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
