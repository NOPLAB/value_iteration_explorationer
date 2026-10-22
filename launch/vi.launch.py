import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("value_iteration_explorationer"), "config", "params.yaml"
    )
    with open(config, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    parameters = dict(raw["/**"]["ros__parameters"])
    # action_list is a list of maps, which rclcpp cannot store. The node reads it.
    parameters["config_file"] = config
    return LaunchDescription(
        [
            Node(
                package="value_iteration_explorationer",
                executable="vi_node",
                name="vi_node",
                parameters=[parameters],
                output="screen",
            )
        ]
    )
