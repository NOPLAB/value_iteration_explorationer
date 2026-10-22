# SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
# SPDX-License-Identifier: BSD-3-Clause

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("value_iteration_explorationer")
    params = LaunchConfiguration("params_file")
    clock = {"use_sim_time": ParameterValue(
        LaunchConfiguration("use_sim_time"), value_type=bool)}
    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file", default_value=os.path.join(share, "config", "exploration.yaml")),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("map_topic", default_value="/map"),
        DeclareLaunchArgument("scan_topic", default_value="/scan"),
        DeclareLaunchArgument("action_topic", default_value="/exploration/action"),
        Node(
            package="value_iteration_explorationer",
            executable="exploration_map_node",
            parameters=[params, clock, {"map_topic": LaunchConfiguration("map_topic")}],
            output="screen",
        ),
        Node(
            package="value_iteration_explorationer",
            executable="exploration_node",
            parameters=[params, clock, {
                "config_file": os.path.join(share, "config", "params.yaml"),
                "action_topic": LaunchConfiguration("action_topic"),
            }],
            remappings=[("/scan", LaunchConfiguration("scan_topic"))],
            output="screen",
        ),
    ])
