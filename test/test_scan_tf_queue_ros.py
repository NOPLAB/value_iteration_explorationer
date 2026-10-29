# SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
# SPDX-License-Identifier: BSD-3-Clause

import json
from math import pi

from geometry_msgs.msg import TransformStamped, Twist
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from test_exploration_ros import RosScenario, make_map


def test_scan_is_retained_until_its_transform_arrives():
    ros = RosScenario()
    try:
        ros.start('exploration_map_node', {
            'use_sim_time': True, 'robot_radius': 0.01,
            'safety_margin': 0.0, 'inflation_radius': 0.0,
        })
        ros.start('exploration_node', {
            'use_sim_time': True, 'theta_cell_num': 36, 'global_thread_num': 1,
            'sensor_range': 0.6, 'candidate_spacing': 0.2,
            'observation_radius': 0.2, 'min_gain': 0.001,
            'robot_radius': 0.01, 'safety_margin': 0.0,
            'replan_interval': 0.1, 'input_timeout': 5.0,
            'map_timeout': 60.0, 'progress_timeout': 60.0,
            'base_frame': ros.tag + '_base',
        })
        waiting_scan = ros.expect(
            String, '/exploration/status', lambda message: message.data == 'waiting_scan')
        waiting_tf = ros.expect(
            String, '/exploration/status', lambda message: message.data == 'waiting_transform')
        ready = ros.expect(
            String, '/exploration/diagnostics',
            lambda message: json.loads(message.data)['plan_status'] == 'ready')
        action = ros.expect(
            Twist, '/exploration/action',
            lambda message: ready.done() and message != Twist(), qos=10)

        ros.broadcaster = StaticTransformBroadcaster(ros.node)
        base = TransformStamped()
        base.header.frame_id = ros.tag + '_map'
        base.child_frame_id = ros.tag + '_base'
        base.transform.translation.x = base.transform.translation.y = 0.05
        base.transform.rotation.w = 1.0
        ros.broadcaster.sendTransform(base)
        clock = Clock()
        clock.clock.sec = 1
        ros.publish_once('/clock', 'rosgraph_msgs/msg/Clock', clock, matches=2)
        grid = make_map(ros, 1)
        grid.data = [35] * len(grid.data)
        ros.publish_once('/map', 'nav_msgs/msg/OccupancyGrid', grid)
        ros.result(waiting_scan, 'accepted map awaiting initial scan')

        scan = LaserScan()
        scan.header.frame_id = ros.tag + '_sensor'
        scan.header.stamp.sec = 1
        scan.angle_min, scan.angle_increment = -pi, pi / 360
        scan.range_min, scan.range_max = 0.05, 5.0
        # Real finite coverage consumes the initial view; an all-infinite scan
        # is freshness input, not evidence of newly observed map regions.
        scan.ranges = [5.0] * 720
        ros.publish_once('/scan', 'sensor_msgs/msg/LaserScan', scan)
        ros.result(waiting_tf, 'retained scan awaiting its missing transform')

        sensor = TransformStamped()
        sensor.header.frame_id = ros.tag + '_base'
        sensor.child_frame_id = ros.tag + '_sensor'
        sensor.transform.rotation.w = 1.0
        ros.broadcaster.sendTransform(sensor)
        # No second scan: releasing the exact queued input must start planning.
        diagnostic = json.loads(ros.result(
            ready, 'queued scan released by transform availability').data)
        assert diagnostic['accepted_scan_stamp_ns'] == 10**9
        assert diagnostic['observed_region_count'] > 0
        ros.result(action, 'nonzero policy from the released scan')
    finally:
        ros.close()
