from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
import os


def generate_launch_description():
    # 获取包路径
    fast_lio_share = FindPackageShare('fast_lio')
    open3d_loc_share = FindPackageShare('open3d_loc')

    # 包含 fast_lio 的 launch 文件
    fast_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                fast_lio_share,
                'launch',
                'mapping.launch.py'
            ])
        ])
    )

    # 包含 open3d_loc 的 launch 文件
    open3d_loc_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                open3d_loc_share,
                'launch',
                'open3d_loc_g1.launch.py'
            ])
        ]),
        # This legacy profile starts FAST-LIO with mid360.yaml (Livox).
        launch_arguments={
            'runtime_recovery_enabled': 'false',
            'recovery_imu_topic': '/livox/imu',
            'recovery_imu_time_offset_sec': '0.0',
        }.items(),
    )

    # RViz 节点配置
    rviz_config_path = PathJoinSubstitution([
        open3d_loc_share,
        'rviz_cfg',
        'fastlio.rviz'
    ])

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_map_cur',
        arguments=['-d', rviz_config_path],
        output='screen',
        prefix='nice'  # 对应 launch-prefix="nice"
    )

    return LaunchDescription([
        fast_lio_launch,
        open3d_loc_launch,
        rviz_node
    ])
