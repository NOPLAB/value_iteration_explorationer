#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
# SPDX-License-Identifier: BSD-3-Clause

"""Continuous execution of the planner's discrete actions using odometry feedback."""

import math
import signal
import threading
from typing import NamedTuple

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan


class Pose2(NamedTuple):
    x: float
    y: float
    yaw: float


class StepCommand(NamedTuple):
    linear: float
    angular: float
    turning: bool
    complete: bool


def step_command(
        start: Pose2, pose: Pose2, forward: float, rotation: float,
        turning: bool, max_linear: float = 0.12, max_angular: float = 0.5) -> StepCommand:
    """Translate along the initial heading, then rotate, as in the VI motion model."""
    if not turning:
        progress = ((pose.x - start.x) * math.cos(start.yaw)
                    + (pose.y - start.y) * math.sin(start.yaw))
        remaining = forward - progress
        if abs(remaining) > 0.02:
            yaw_error = math.remainder(start.yaw - pose.yaw, 2 * math.pi)
            return StepCommand(
                max(-max_linear, min(max_linear, remaining * 1.2)),
                max(-max_angular, min(max_angular, yaw_error * 2)),
                False, False)
        turning = True
    error = math.remainder(start.yaw + rotation - pose.yaw, 2 * math.pi)
    if abs(error) <= 0.04:
        return StepCommand(0.0, 0.0, True, True)
    return StepCommand(
        0.0, max(-max_angular, min(max_angular, error * 2)), turning, False)


class ActionExecutor(Node):
    """Keep accepting steps; only zero/stale/unsafe input or shutdown stops motion."""

    def __init__(self) -> None:
        super().__init__("example_action_executor")
        self.max_linear = self.declare_parameter("max_linear", 0.12).value
        self.max_angular = self.declare_parameter("max_angular", 0.5).value
        self.action_timeout = self.declare_parameter("action_timeout", 0.35).value
        self.input_timeout = self.declare_parameter("input_timeout", 0.5).value
        self.stop_distance = self.declare_parameter("stop_distance", 0.13).value
        for value in (self.max_linear, self.max_angular, self.action_timeout,
                      self.input_timeout, self.stop_distance):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Executor limits and timeouts must be finite and positive")
        self.pose: Pose2 | None = None
        self.start: Pose2 | None = None
        self.motion: tuple[float, float] | None = None
        self.pending_motion: tuple[float, float] | None = None
        self.turning = False
        self.action_stamp = float("-inf")
        self.odom_stamp = float("-inf")
        self.scan_stamp = float("-inf")
        self.scan_distance = float("inf")
        self.output = self.create_publisher(
            Twist, self.declare_parameter("cmd_vel_topic", "/cmd_vel").value, 1)
        self.subscriptions_owned = [
            self.create_subscription(
                Twist, self.declare_parameter("action_topic", "/exploration/action").value,
                self.on_action, 10),
            self.create_subscription(
                Odometry, self.declare_parameter("odom_topic", "/odom").value,
                self.on_odom, 10),
            self.create_subscription(
                LaserScan, self.declare_parameter("scan_topic", "/scan").value,
                self.on_scan, qos_profile_sensor_data),
        ]
        self.create_timer(0.05, self.control)
        self.get_logger().info("Continuous action executor ready; no measurement auto-stop")

    def seconds(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def on_action(self, message: Twist) -> None:
        desired = (message.linear.x, message.angular.z)
        self.action_stamp = self.seconds()
        if desired == (0.0, 0.0) or not all(math.isfinite(value) for value in desired):
            self.motion = None
            self.pending_motion = None
            self.output.publish(Twist())
        else:
            self.pending_motion = desired
            if self.pose is not None and self.motion is None:
                self.begin_step(desired, self.pose)

    def begin_step(self, motion: tuple[float, float], pose: Pose2) -> None:
        self.motion = motion
        self.start = pose
        self.turning = abs(motion[0]) < 0.001

    def on_odom(self, message: Odometry) -> None:
        pose = message.pose.pose
        q = pose.orientation
        current = Pose2(
            pose.position.x, pose.position.y,
            math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
        if all(math.isfinite(value) for value in current):
            self.pose = current
            self.odom_stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9

    def on_scan(self, message: LaserScan) -> None:
        values = [value for value in message.ranges
                  if math.isfinite(value) and message.range_min <= value <= message.range_max]
        if values or any(math.isinf(value) and value > 0 for value in message.ranges):
            self.scan_distance = min(values, default=float("inf"))
            self.scan_stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9

    def control(self) -> None:
        output = Twist()
        now = self.seconds()
        if (self.pose is not None and self.start is not None and self.motion is not None
                and self.pending_motion is not None
                and 0 <= now - self.action_stamp <= self.action_timeout
                and 0 <= now - self.odom_stamp <= self.input_timeout
                and 0 <= now - self.scan_stamp <= self.input_timeout
                and self.scan_distance >= self.stop_distance):
            command = step_command(
                self.start, self.pose, *self.motion, self.turning,
                self.max_linear, self.max_angular)
            if command.complete:
                self.begin_step(self.pending_motion, self.pose)
                command = step_command(
                    self.start, self.pose, *self.motion, self.turning,
                    self.max_linear, self.max_angular)
            self.turning = command.turning
            output.linear.x = command.linear
            output.angular.z = command.angular
        else:
            self.motion = None
            self.pending_motion = None
        self.output.publish(output)


def main() -> None:
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = ActionExecutor()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    stopped = threading.Event()

    def stop(signum, frame) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    worker = threading.Thread(target=executor.spin)
    worker.start()
    try:
        stopped.wait()
    finally:
        executor.shutdown()
        worker.join()
        node.output.publish(Twist())
        acknowledged = node.output.wait_for_all_acked(Duration(seconds=1))
        node.destroy_node()
        rclpy.shutdown()
        if not acknowledged:
            raise RuntimeError("Final stop command was not acknowledged")


if __name__ == "__main__":
    main()
