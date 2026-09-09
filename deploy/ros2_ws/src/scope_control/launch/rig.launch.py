"""
Rig run: depth_bridge + policy_node + scope_link, all under /scope.
Real inference, real serial. win_vision runs separately on Windows.

    ros2 launch scope_control rig.launch.py port:=/dev/ttyUSB0

Start-up order for a trial:
  1. this launch file
  2. win_vision_server.py on Windows  -> depth starts flowing on /scope/depth
  3. ros2 topic pub -1 /scope/command scope_msgs/msg/ScopeCommand '{command: 1}'   # ENABLE firmware
  4. ros2 topic pub -1 /scope/run_state scope_msgs/msg/ScopeRunState '{command: 2}' # RESET+RUN policy
Stop: run_state command 0 (policy), command 0 (firmware SAFE_HOLD), or command 3 (E-STOP).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory("scope_control"), "config", "scope_params.yaml"
    )
    port = LaunchConfiguration("port")
    tcp_port = LaunchConfiguration("tcp_port")
    mock_policy = LaunchConfiguration("mock_policy")

    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("tcp_port", default_value="5599"),
        DeclareLaunchArgument("mock_policy", default_value="false"),

        Node(
            package="scope_control", executable="depth_bridge", name="depth_bridge",
            namespace="scope", output="screen",
            parameters=[{"tcp_port": tcp_port}],
        ),
        Node(
            package="scope_control", executable="policy_node", name="policy_node",
            namespace="scope", output="screen",
            parameters=[params, {"mock_policy": mock_policy}],
        ),
        Node(
            package="scope_control", executable="scope_link", name="scope_link",
            namespace="scope", output="screen",
            parameters=[params, {"port": port}],
        ),
    ])
