# SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import subprocess
import uuid

from ament_index_python.packages import get_package_prefix
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap
import pytest
import rclpy
from rclpy.task import Future


@pytest.mark.parametrize('mode', ['occupancy', 'cost_add', 'cost_replace'])
def test_map_service(mode):
    """Exercise the installed node against both actual ROS 2 GetMap services."""
    rclpy.init()
    node = rclpy.create_node('map_service_test')
    calls = []
    received = Future()
    prefix = '/map_service_' + uuid.uuid4().hex

    def respond(name):
        def callback(request, response):
            calls.append(name)
            grid = response.map
            grid.info.width = 8
            grid.info.height = 8
            grid.info.resolution = 0.1
            grid.info.origin.position.x = 2.0
            grid.info.origin.position.y = 3.0
            grid.info.origin.orientation.w = 1.0
            grid.data = [1 if name == '/cost_map' else 0] * 64
            grid.data[63] = -1
            grid.data[36] = -2  # unsigned 254: free only in cost mode
            return response
        return callback

    services = [node.create_service(GetMap, prefix + name, respond(name))
                for name in ['/map_server/map', '/cost_map']]

    def on_grid(grid):
        if not received.done():
            received.set_result(grid)

    subscription = node.create_subscription(
        OccupancyGrid, prefix + '/value_function', on_grid, 10)
    package = 'value_iteration_explorationer'
    executable = Path(get_package_prefix(package)) / 'lib' / package / 'vi_node'
    process = subprocess.Popen([
        str(executable), '--ros-args', '-p', 'online:=true',
        '-p', f'map_type:={mode}', '-p', 'theta_cell_num:=60',
        '-p', 'goal_margin_theta:=30', '-p', 'goal_margin_radius:=0.3',
        '-p', 'global_thread_num:=1', '-p', 'cost_drawing_threshold:=10000',
        '-r', f'/map_server/map:={prefix}/map_server/map',
        '-r', f'/cost_map:={prefix}/cost_map',
        '-r', f'/goal_pose:={prefix}/goal_pose',
        '-r', f'/value_function:={prefix}/value_function',
        '-r', f'/cmd_vel:={prefix}/cmd_vel',
        '-r', f'/scan:={prefix}/scan',
    ])
    goal_process = subprocess.Popen([
        'ros2', 'topic', 'pub', '--once', '--wait-matching-subscriptions', '1',
        prefix + '/goal_pose', 'geometry_msgs/msg/PoseStamped',
        '{header: {frame_id: map}, pose: {position: {x: 2.15, y: 3.15}, '
        'orientation: {w: 1.0}}}',
    ])
    try:
        rclpy.spin_until_future_complete(node, received, timeout_sec=30)
        assert process.poll() is None, 'vi_node exited before publishing'
        cost_mode = mode != 'occupancy'
        assert calls == ['/cost_map' if cost_mode else '/map_server/map']
        assert received.done(), 'no solved value function received'
        grid = received.result()
        assert grid.info.width == 8 and grid.info.height == 8
        assert grid.info.origin.position.x == 2.0
        assert grid.info.origin.position.y == 3.0
        assert grid.data[63] == 100, '255 must stay blocked'
        if cost_mode:
            assert grid.data[36] < 100, '254 must be traversable'
        else:
            assert grid.data[36] == 100, 'nonzero occupancy must stay blocked'
    finally:
        goal_process.terminate()
        goal_process.wait(timeout=5)
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        node.destroy_subscription(subscription)
        for service in services:
            node.destroy_service(service)
        node.destroy_node()
        rclpy.shutdown()
