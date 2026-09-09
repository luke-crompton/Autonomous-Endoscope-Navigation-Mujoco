"""
policy_node -- the brain of the deployment control loop.

    /scope/depth  (sensor_msgs/Image, 32FC1, 1x54x96, already [0,1])
    /scope/telemetry (scope_msgs/ScopeTelemetry)      -> tip_contact, safety
    /scope/estop  (std_msgs/Bool, latched)            -> freeze
    /scope/run_state (scope_msgs/ScopeRunState, latched) -> RUN / STOP / RESET
                              |
                              v
                        [ policy_node ]
                              |
                              v
    /scope/action (scope_msgs/ScopeAction)  -> absolute tendon pull, mm

The node is FRAME-DRIVEN: one decision per incoming depth frame, not on a timer.
The depth model is the slowest link (~32 ms of the 40 ms budget), so its frame
arrival IS the loop clock. A timer-driven design would either starve or double up
whenever the two rates drift -- exactly the phase-offset trap.

WHAT THIS NODE REPRODUCES FROM THE SIMULATOR (and why here)
----------------------------------------------------------
Five of the six policy state values are software, not sensors, and MUST be
reproduced bit-for-bit or the network sees an observation it was never trained
on (docs/architecture.md, "What is deliberately not observed"):

  state[0:2]  cmd_x_n, cmd_y_n   -- the PD command shaper's own output, normalised
  state[2:5]  last_action[0:3]   -- the previous action the network emitted
  state[5]    tip_contact        -- the ONE real sensor value (from telemetry)

The PD shaper (scope_colon_env.py step(), ~lines 1004-1021) is a 4-scalar
recursive filter whose output is fed straight back into the next observation. It
therefore lives *inside* this loop. Its gains were fitted at 25 Hz and do NOT
rescale arithmetically -- a x4 rescale puts the poles at |z|=1.337 (unstable).
Do not touch PD_KP / PD_KD without re-fitting against the wall-clock step
response (navigation/v9/diagnostics/pd_rate_rescale.py).

The order of operations per frame mirrors the sim exactly:
  env.reset() returns obs with cmd=0, last_action=0            <-> RESET / first frame
  env.step(a) : shaper updates from a, THEN obs is rebuilt     <-> on_depth()
so the observation built at the end of step t carries the shaper state *after*
applying action t, and that is what feeds inference for action t+1.
"""

from __future__ import annotations

import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Bool
from sensor_msgs.msg import Image

from scope_msgs.msg import ScopeAction, ScopeRunState, ScopeTelemetry

from scope_control.policy_runner import MockPolicyRunner, PolicyRunner
from scope_control.shaper import pd_shaper_step


# --- QoS profiles --------------------------------------------------------
# Absolute setpoints / sensor streams that SUPERSEDE: keep only the newest, and
# never let a slow consumer build a stale backlog. This is the fix for the
# "works, then progressively lags" symptom.
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)
# Latched state/events: a late subscriber must still get the last value.
LATCHED_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class PolicyNode(Node):

    def __init__(self):
        super().__init__("policy_node")

        # ---------------- parameters -----------------------------------
        # Contract with the trained checkpoint. Defaults are the v8_p1 / v9 line.
        self.declare_parameter("experiment", "v8_p1_v1")
        self.declare_parameter("nav_code_dir", "")           # dir with sf_encoder.py
        self.declare_parameter("checkpoint_dir", "")         # checkpoint_p0 folder or a best_*.pth
        self.declare_parameter("checkpoint_root", "")        # dir containing runs_sf/
        self.declare_parameter("device", "cpu")
        self.declare_parameter("mock_policy", False)

        self.declare_parameter("depth_height", 54)
        self.declare_parameter("depth_width", 96)
        self.declare_parameter("state_dim", 6)

        # PD command shaper -- fitted at 25 Hz, DO NOT arithmetically rescale.
        self.declare_parameter("pd_kp", 0.1803)
        self.declare_parameter("pd_kd", 0.0475)

        # Absolute pull (mm) at full commanded bend, per axis. The two axes
        # genuinely differ (X pair spans 12 joints, Z pair 13). These defaults
        # are SIM GEOMETRY -- measure on the real scope (CURRENT_PLAN.md
        # section 7, step 1b) and set them in the params file.
        self.declare_parameter("max_pull_x_mm", 6.795)
        self.declare_parameter("max_pull_y_mm", 7.361)

        self.declare_parameter("control_rate_hz", 25.0)   # only sizes the stale-frame warning
        self.declare_parameter("tip_contact_default", False)
        self.declare_parameter("start_running", False)

        gp = self.get_parameter
        self._h = int(gp("depth_height").value)
        self._w = int(gp("depth_width").value)
        self._state_dim = int(gp("state_dim").value)
        self._kp = float(gp("pd_kp").value)
        self._kd = float(gp("pd_kd").value)
        self._max_pull_x = float(gp("max_pull_x_mm").value)
        self._max_pull_y = float(gp("max_pull_y_mm").value)
        self._ctrl_dt = 1.0 / max(float(gp("control_rate_hz").value), 1e-3)
        self._tip_default = bool(gp("tip_contact_default").value)
        self._running = bool(gp("start_running").value)

        if self._state_dim != 6:
            self.get_logger().warn(
                f"state_dim={self._state_dim} (expected 6). The state vector layout "
                "below assumes the 6-D [cmd_x_n, cmd_y_n, last_action[0:3], tip_contact]."
            )
        if abs(self._max_pull_x - 6.795) < 1e-6 and abs(self._max_pull_y - 7.361) < 1e-6:
            self.get_logger().warn(
                "MAX_PULL_X/Y are still the SIM defaults (6.795 / 7.361 mm). These "
                "normalise two live policy inputs -- measure them on the real scope."
            )

        # ---------------- policy runner -------------------------------
        mock = bool(gp("mock_policy").value)
        if mock:
            self.get_logger().warn("mock_policy=true -- emitting ZERO actions, no inference.")
            self._runner = MockPolicyRunner(depth_hw=(self._h, self._w), state_dim=self._state_dim)
        else:
            try:
                self._runner = PolicyRunner(
                    experiment=str(gp("experiment").value),
                    nav_code_dir=str(gp("nav_code_dir").value),
                    checkpoint_dir=str(gp("checkpoint_dir").value),
                    checkpoint_root=str(gp("checkpoint_root").value),
                    depth_hw=(self._h, self._w),
                    state_dim=self._state_dim,
                    device=str(gp("device").value),
                )
                self.get_logger().info(
                    f"loaded checkpoint {self._runner.checkpoint_path} "
                    f"(env_steps={self._runner.env_steps})"
                )
            except Exception as e:  # noqa: BLE001 - fail loud, do not run blind
                self.get_logger().fatal(f"PolicyRunner init failed: {e}")
                raise

        # ---------------- loop state ---------------------------------
        self._reset_loop_state()
        self._estop = False
        self._safety_state = ScopeTelemetry.SAFETY_OK
        self._last_depth_stamp = None
        self._last_telemetry_stamp = None
        self._infer_warned = False
        self._norm_warned = False
        self._shape_warned = False

        # ---------------- I/O ---------------------------------------
        self._action_pub = self.create_publisher(ScopeAction, "action", SENSOR_QOS)

        self.create_subscription(Image, "depth", self._on_depth, SENSOR_QOS)
        self.create_subscription(ScopeTelemetry, "telemetry", self._on_telemetry, SENSOR_QOS)
        self.create_subscription(Bool, "estop", self._on_estop, LATCHED_QOS)
        self.create_subscription(ScopeRunState, "run_state", self._on_run_state, LATCHED_QOS)

        # health tick: warn if depth stalls
        self.create_timer(0.5, self._on_health)

        self.get_logger().info(
            f"policy_node up. depth={self._h}x{self._w} state_dim={self._state_dim} "
            f"KP={self._kp} KD={self._kd} MAX_PULL=({self._max_pull_x},{self._max_pull_y}) mm "
            f"running={self._running}"
        )

    # ------------------------------------------------------------ state

    def _reset_loop_state(self):
        """Equivalent of env.reset(): zero the shaper, the action echo and the
        recurrent state. tip_contact falls back to its configured default until
        the first telemetry message arrives."""
        self._cmd_x = 0.0
        self._cmd_y = 0.0
        self._prev_cmd_x = 0.0
        self._prev_cmd_y = 0.0
        self._last_action = np.zeros(3, dtype=np.float32)
        self._tip_contact = self._tip_default
        self._seq = 0
        self._runner.reset()

    # --------------------------------------------------------- callbacks

    def _on_run_state(self, msg: ScopeRunState):
        if msg.command == ScopeRunState.RESET:
            self._reset_loop_state()
            self._running = True
            self.get_logger().info("RESET -> shaper/GRU/action zeroed, running.")
        elif msg.command == ScopeRunState.RUN:
            self._running = True
            self.get_logger().info("RUN.")
        elif msg.command == ScopeRunState.STOP:
            self._running = False
            self.get_logger().info("STOP -> not acting (firmware watchdog will SAFE_HOLD).")
        else:
            self.get_logger().warn(f"unknown ScopeRunState.command={msg.command}")

    def _on_estop(self, msg: Bool):
        if msg.data and not self._estop:
            self.get_logger().error("E-STOP asserted -- freezing; publish RESET to resume.")
        elif not msg.data and self._estop:
            self.get_logger().warn("E-STOP cleared (still not acting until RUN/RESET).")
        self._estop = bool(msg.data)

    def _on_telemetry(self, msg: ScopeTelemetry):
        self._tip_contact = bool(msg.tip_contact)
        self._last_telemetry_stamp = self.get_clock().now()
        if msg.safety_state != self._safety_state:
            self.get_logger().warn(
                f"firmware safety_state {self._safety_state} -> {msg.safety_state}"
            )
            self._safety_state = msg.safety_state
        # Desync check: firmware should be echoing a recent seq. A large gap
        # means /scope/action is not getting through.
        if self._seq > 0 and (self._seq - msg.seq_echo) > 5:
            self.get_logger().warn(
                f"firmware seq_echo={msg.seq_echo} lags publisher seq={self._seq} "
                "-- /scope/action may be dropping."
            )

    def _on_health(self):
        if not self._running or self._last_depth_stamp is None:
            return
        age = (self.get_clock().now() - self._last_depth_stamp).nanoseconds * 1e-9
        if age > 5 * self._ctrl_dt:
            self.get_logger().warn(f"no depth frame for {age:.2f}s -- loop is starved.")

    # ------------------------------------------------------------- loop

    def _on_depth(self, msg: Image):
        # Gate: only act when explicitly running and not e-stopped. When not
        # running we publish nothing -- firmware holds the last setpoint and its
        # own comms watchdog takes it to SAFE_HOLD.
        if not self._running or self._estop:
            return
        if self._safety_state in (ScopeTelemetry.SAFETY_HOLD, ScopeTelemetry.SAFETY_ESTOP):
            return

        now = self.get_clock().now()
        self._last_depth_stamp = now

        depth = self._decode_depth(msg)
        if depth is None:
            return

        # ---- build the observation (state BEFORE this step's action) ----
        # This is exactly what scope_colon_env._observation() returns at the end
        # of the previous step: shaper state + previous action + latest contact.
        state = np.array([
            self._cmd_x / self._max_pull_x,
            self._cmd_y / self._max_pull_y,
            self._last_action[0],
            self._last_action[1],
            self._last_action[2],
            1.0 if self._tip_contact else 0.0,
        ], dtype=np.float32)

        # ---- inference (deterministic mean) ----
        try:
            action = self._runner.act(depth, state)
        except Exception as e:  # noqa: BLE001
            if not self._infer_warned:
                self.get_logger().error(f"inference failed: {e}")
                self._infer_warned = True
            return
        self._infer_warned = False
        action = np.asarray(action, dtype=np.float32).reshape(-1)[:3]

        # ---- PD command shaper: the "step" (mirrors scope_colon_env.step) ----
        # Work in mm throughout. The filter is linear and scale-free, so the
        # normalised obs value cmd/MAX_PULL is identical whether we carry mm or m.
        target_x = float(action[0]) * self._max_pull_x
        target_y = float(action[1]) * self._max_pull_y
        self._cmd_x, self._prev_cmd_x = pd_shaper_step(
            self._cmd_x, self._prev_cmd_x, target_x, self._kp, self._kd, self._max_pull_x
        )
        self._cmd_y, self._prev_cmd_y = pd_shaper_step(
            self._cmd_y, self._prev_cmd_y, target_y, self._kp, self._kd, self._max_pull_y
        )

        # last_action stores the COMMANDED action (pre-shaper), same as
        # scope_colon_env.py:1229 `self.last_action[:] = a`.
        self._last_action = action.copy()
        self._seq += 1

        # ---- publish absolute pull setpoints (mm) ----
        out = ScopeAction()
        out.header.stamp = now.to_msg()
        out.header.frame_id = "scope_tip"
        out.seq = self._seq
        out.cmd_x_pair_mm = float(self._cmd_x)
        out.cmd_y_pair_mm = float(self._cmd_y)
        self._action_pub.publish(out)

    # --------------------------------------------------------- helpers

    def _decode_depth(self, msg: Image):
        """sensor_msgs/Image (32FC1) -> (H, W) float32, or None on a bad frame.

        This node does NOT resize or normalise: the depth obs contract
        (54x96, per-frame-max normalised to [0,1]) is owned upstream by
        win_vision (Da3Depth.to_obs). A mismatch here is a wiring bug, not
        something to paper over.
        """
        if msg.encoding != "32FC1":
            if not self._shape_warned:
                self.get_logger().error(f"depth encoding '{msg.encoding}', expected 32FC1")
                self._shape_warned = True
            return None
        if (msg.height, msg.width) != (self._h, self._w):
            if not self._shape_warned:
                self.get_logger().error(
                    f"depth is {msg.height}x{msg.width}, expected {self._h}x{self._w}. "
                    "win_vision owns the resize -- fix it there, not here."
                )
                self._shape_warned = True
            return None
        self._shape_warned = False

        dt = np.dtype(np.float32).newbyteorder(">" if msg.is_bigendian else "<")
        arr = np.frombuffer(bytes(msg.data), dtype=dt)
        if arr.size != self._h * self._w:
            self.get_logger().error(
                f"depth payload {arr.size} floats, expected {self._h * self._w}"
            )
            return None
        arr = np.ascontiguousarray(arr.reshape(self._h, self._w), dtype=np.float32)

        if not np.isfinite(arr).all():
            self.get_logger().warn("depth frame has non-finite values -- dropped.")
            return None
        if (arr.max() > 1.5 or arr.min() < -0.5) and not self._norm_warned:
            self.get_logger().warn(
                f"depth range [{arr.min():.2f}, {arr.max():.2f}] does not look "
                "per-frame-max normalised -- check win_vision calls Da3Depth.to_obs()."
            )
            self._norm_warned = True
        return arr


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
