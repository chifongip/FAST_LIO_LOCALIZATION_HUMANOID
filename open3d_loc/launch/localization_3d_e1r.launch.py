from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    fast_lio_share = FindPackageShare('fast_lio')
    open3d_loc_share = FindPackageShare('open3d_loc')

    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value='/home/ubuntu/Humanoid-Nav/2025-10-02-Lab.pcd',
        description='PCD map used for global localization'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation time'
    )
    start_rviz_arg = DeclareLaunchArgument(
        'start_rviz', default_value='true',
        description='Start RViz'
    )

    fast_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            fast_lio_share, 'launch', 'mapping.launch.py'
        ])),
        launch_arguments={
            'config_file': 'e1r.yaml',
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'rviz': 'false',
        }.items()
    )

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            open3d_loc_share, 'launch', 'open3d_loc_g1.launch.py'
        ])),
        launch_arguments={
            'map_file': LaunchConfiguration('map_file'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'imu_frame': 'lidar_imu_chest_front',
            'body_frame': 'base_link',
            'output_frame': 'torso_link',
            'publish_robot_root_tf': 'true',
            'publish_output_tf': 'false',
            'tf_lookup_max_age_ms': '100.0',
            'publish_legacy_static_frames': 'false',
        }.items()
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_map_cur',
        arguments=['-d', PathJoinSubstitution([
            open3d_loc_share, 'rviz_cfg', 'fastlio.rviz'
        ])],
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_rviz'))
    )

    return LaunchDescription([
        map_file_arg,
        use_sim_time_arg,
        start_rviz_arg,
        fast_lio_launch,
        localization_launch,
        rviz_node,
    ])
