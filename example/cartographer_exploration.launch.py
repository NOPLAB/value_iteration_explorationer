"""Launch Cartographer and exploration using an existing robot Lua configuration."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare('value_iteration_explorationer')
    use_sim_time = LaunchConfiguration('use_sim_time')
    scan_topic = LaunchConfiguration('scan_topic')

    return LaunchDescription([
        DeclareLaunchArgument(
            'configuration_directory',
            description='Absolute directory containing the robot Cartographer Lua configuration'),
        DeclareLaunchArgument(
            'configuration_basename',
            description='Robot Cartographer Lua configuration filename'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        DeclareLaunchArgument('map_resolution', default_value='0.05'),
        DeclareLaunchArgument('map_publish_period_sec', default_value='1.0'),
        DeclareLaunchArgument('action_topic', default_value='/exploration/action'),
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution([
                package_share, 'config', 'exploration.yaml',
            ])),
        Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=[
                '-configuration_directory', LaunchConfiguration('configuration_directory'),
                '-configuration_basename', LaunchConfiguration('configuration_basename'),
            ],
            remappings=[('scan', scan_topic)],
        ),
        Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='cartographer_occupancy_grid_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=[
                '-resolution', LaunchConfiguration('map_resolution'),
                '-publish_period_sec', LaunchConfiguration('map_publish_period_sec'),
            ],
            remappings=[('map', '/map')],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                package_share, 'launch', 'exploration.launch.py',
            ])),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'scan_topic': scan_topic,
                'map_topic': '/map',
                'action_topic': LaunchConfiguration('action_topic'),
                'params_file': LaunchConfiguration('params_file'),
            }.items(),
        ),
    ])
