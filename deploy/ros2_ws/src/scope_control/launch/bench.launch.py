"""
Bench bring-up: depth_bridge + policy_node + mock_scope_link, all under /scope,
with no rig and no checkpoint (mock policy). This is the "does the graph spin
and stay at rate" test.

    ros2 launch scope_control bench.launch.py
    ros2 launch scope_control bench.launch.py mock_policy:=false   # real inference

Then feed frames on /scope/depth from win_vision (or a test publisher) and:
    ros2 topic pub -1 /scope/run_state scope_msgs/msg/ScopeRunState '{command: 2}'
    ros2 topic hz /scope/action
    ros2 topic echo /scope/telemetry --once
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

    mock_policy = LaunchConfiguration("mock_policy")
    tcp_port = LaunchConfiguration("tcp_port")
    trip_estop_at_s = LaunchConfiguration("trip_estop_at_s")

    return LaunchDescription([
        DeclareLaunchArgument("mock_policy", default_value="true"),
        DeclareLaunchArgument("tcp_port", default_value="5599"),
        DeclareLaunchArgument("trip_estop_at_s", default_value="0.0"),

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
            package="scope_control", executable="mock_scope_link", name="mock_scope_link",
            namespace="scope", output="screen",
            parameters=[{"trip_estop_at_s": trip_estop_at_s}],
        ),
    ])
