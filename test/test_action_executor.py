# SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
# SPDX-License-Identifier: BSD-3-Clause

import math
from collections.abc import Iterator

import pytest
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from discrete_action_executor import ActionExecutor, Pose2, step_command


@pytest.fixture
def moving_executor(
        monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[ActionExecutor, list[Twist]]]:
    rclpy.init(args=["--ros-args", "-r", "/cmd_vel:=/test_executor/cmd_vel"])
    node = ActionExecutor()
    monkeypatch.setattr(node, "seconds", lambda: 100.0)
    published: list[Twist] = []
    monkeypatch.setattr(node.output, "publish", published.append)
    odom = Odometry()
    odom.header.stamp.sec = 100
    odom.pose.pose.orientation.w = 1.0
    node.on_odom(odom)
    scan = LaserScan()
    scan.header.stamp.sec = 100
    scan.range_min, scan.range_max, scan.ranges = .01, 10.0, [1.0]
    node.on_scan(scan)
    try:
        yield node, published
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_changed_translation_waits_for_current_primitive(moving_executor):
    node, published = moving_executor
    forward, reverse = Twist(), Twist()
    forward.linear.x, reverse.linear.x = .3, -.2
    node.on_action(forward)
    original_start = node.start
    node.pose = Pose2(.05, 0, 0)
    node.on_action(reverse)
    node.control()
    assert node.start == original_start
    assert published[-1].linear.x > 0
    node.pose = Pose2(.3, 0, 0)
    node.control()
    assert published[-1].linear.x < 0
    assert node.start == node.pose


def test_changed_rotation_waits_for_current_primitive(moving_executor):
    node, published = moving_executor
    left, right = Twist(), Twist()
    left.angular.z, right.angular.z = math.radians(20), math.radians(-20)
    node.on_action(left)
    original_start = node.start
    node.pose = Pose2(0, 0, math.radians(5))
    node.on_action(right)
    node.control()
    assert node.start == original_start
    assert published[-1].angular.z > 0
    node.pose = Pose2(0, 0, math.radians(20))
    node.control()
    assert published[-1].angular.z < 0


def test_repeated_primitive_has_no_boundary_stop(moving_executor):
    node, published = moving_executor
    action = Twist()
    action.linear.x = .3
    node.on_action(action)
    node.pose = Pose2(.3, 0, 0)
    node.control()
    assert published[-1].linear.x > 0
    assert node.start == node.pose


def test_zero_interrupts_active_and_queued_primitive(moving_executor):
    node, published = moving_executor
    forward, reverse = Twist(), Twist()
    forward.linear.x, reverse.linear.x = .3, -.2
    node.on_action(forward)
    node.on_action(reverse)
    node.on_action(Twist())
    node.control()
    assert published[-1] == Twist()
    assert node.motion is None
    assert node.pending_motion is None


def test_stale_input_discards_queued_primitive(moving_executor):
    node, published = moving_executor
    action = Twist()
    action.linear.x = .3
    node.on_action(action)
    node.action_stamp = 99
    node.control()
    assert published[-1] == Twist()
    assert node.motion is None
    assert node.pending_motion is None


@pytest.mark.parametrize("forward, expected", [(0.3, 0.12), (-0.3, -0.12)])
def test_translation_has_direction_and_velocity_limit(forward, expected):
    command = step_command(Pose2(0, 0, 0), Pose2(0, 0, 0), forward, 0, False)
    assert command.linear == pytest.approx(expected)
    assert command.angular == 0
    assert not command.complete and not command.turning


@pytest.mark.parametrize("pose, turning, linear, angular, complete", [
    (Pose2(0, 0, 0), False, 0.12, 0.0, False),
    (Pose2(0.3, 0, 0), False, 0.0, 0.5, False),
    (Pose2(0.3, 0, math.radians(20)), True, 0.0, 0.0, True),
])
def test_combined_step_translates_before_rotating(pose, turning, linear, angular, complete):
    command = step_command(Pose2(0, 0, 0), pose, 0.3, math.radians(20), turning)
    assert command.linear == pytest.approx(linear)
    assert command.angular == pytest.approx(angular)
    assert command.complete == complete


def test_rotation_completion_wraps_at_pi():
    command = step_command(
        Pose2(0, 0, math.radians(179)), Pose2(0, 0, math.radians(-179)),
        0, math.radians(2), True)
    assert command.complete
    assert command.linear == command.angular == 0


@pytest.mark.parametrize("use_sim_time", [False, True])
@pytest.mark.parametrize("condition", [
    "fresh", "stale_action", "stale_odom", "stale_scan", "blocked", "stop",
])
def test_executor_with_both_clocks_preserves_stops(use_sim_time, condition, monkeypatch):
    # Given: real ROS messages and an executor with a deterministic clock.
    rclpy.init(args=[
        "--ros-args", "-p", f"use_sim_time:={str(use_sim_time).lower()}",
        "-r", "/cmd_vel:=/test_executor/cmd_vel",
    ])
    node = None
    try:
        node = ActionExecutor()
        monkeypatch.setattr(node, "seconds", lambda: 100.0)
        published: list[Twist] = []
        monkeypatch.setattr(node.output, "publish", published.append)
        odom = Odometry()
        odom.header.stamp.sec = 100
        odom.pose.pose.orientation.w = 1.0
        scan = LaserScan()
        scan.header.stamp.sec = 100
        scan.range_min, scan.range_max = 0.01, 10.0
        scan.ranges = [0.1 if condition == "blocked" else 1.0]
        node.on_odom(odom)
        node.on_scan(scan)
        action = Twist()
        action.linear.x = 0.3
        node.on_action(action)
        if condition == "stale_action":
            node.action_stamp = 99.0
        if condition == "stale_odom":
            node.odom_stamp = 99.0
        if condition == "stale_scan":
            node.scan_stamp = 99.0
        if condition == "stop":
            node.on_action(Twist())

        # When: the controller runs without publishing to a robot.
        node.control()

        # Then: only fresh, clear, nonzero input permits motion.
        assert published[-1].linear.x == pytest.approx(0.12 if condition == "fresh" else 0.0)
        assert published[-1].angular.z == 0.0
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
