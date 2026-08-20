from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # 获取包路径
    open3d_loc_share = FindPackageShare('open3d_loc')

    # 声明 use_sim_time 参数
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value='/home/ubuntu/Humanoid-Nav/2025-10-02-Lab.pcd',
        description='PCD map used for global localization'
    )
    imu_frame_arg = DeclareLaunchArgument(
        'imu_frame', default_value='imu_link',
        description='Frame represented by FAST-LIO odometry'
    )
    body_frame_arg = DeclareLaunchArgument(
        'body_frame', default_value='base_link',
        description='Robot body frame used for scan matching'
    )
    output_frame_arg = DeclareLaunchArgument(
        'output_frame', default_value='motion_link',
        description='Frame represented by /localization_3d'
    )
    publish_robot_root_tf_arg = DeclareLaunchArgument(
        'publish_robot_root_tf', default_value='false',
        description='Publish the local odometry frame to robot root transform'
    )
    publish_output_tf_arg = DeclareLaunchArgument(
        'publish_output_tf', default_value='true',
        description='Publish a direct map transform for the output frame'
    )
    tf_lookup_max_age_arg = DeclareLaunchArgument(
        'tf_lookup_max_age_ms', default_value='100.0',
        description='Maximum age for a fallback robot transform'
    )
    legacy_static_frames_arg = DeclareLaunchArgument(
        'publish_legacy_static_frames', default_value='true',
        description='Publish the G1 identity imu/base/motion transforms'
    )

    # 配置文件路径
    config_file = PathJoinSubstitution([
        open3d_loc_share,
        'config',
        'loc_param_g1.yaml'
    ])

    map_file = LaunchConfiguration('map_file')

    # 静态TF发布节点 - imu_link to base_link
    # 修正：父frame是imu_link，子frame是base_link
    static_tf_imulink2baselink = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='imulink2baselink',
        arguments=['0', '0', '0', '0', '0', '0', '1', 'imu_link', 'base_link'],
        condition=IfCondition(LaunchConfiguration('publish_legacy_static_frames'))
    )

    # 静态TF发布节点 - base_link to motion_link
    # 修正：base_link是父frame，motion_link是子frame
    static_tf_base_center = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_center_broadcaster',
        arguments=['0', '0', '0', '0', '0', '0',
                   '1', 'base_link', 'motion_link'],
        condition=IfCondition(LaunchConfiguration('publish_legacy_static_frames'))
    )

    # 全局定位节点
    global_localization_node = Node(
        package='open3d_loc',
        executable='global_localization_node',
        name='global_localization_node',
        output='screen',
        parameters=[
            config_file,
            {
                'path_map': map_file,
                'pcd_queue_maxsize': 10,
                'voxelsize_coarse': 0.01,
                'voxelsize_fine': 0.2,
                'threshold_fitness': 0.5,
                'threshold_fitness_init': 0.5,
                'loc_frequence': 2.5,
                'save_scan': False,
                'hidden_removal': False,
                'maxpoints_source': 80000,
                'maxpoints_target': 400000,
                'filter_odom2map': False,
                'kalman_processVar2': 0.001,
                'kalman_estimatedMeasVar2': 0.02,
                'confidence_loc_th': 0.7,
                'dis_updatemap': 3.5,
                'imu_frame': LaunchConfiguration('imu_frame'),
                'body_frame': LaunchConfiguration('body_frame'),
                'output_frame': LaunchConfiguration('output_frame'),
                'publish_robot_root_tf': LaunchConfiguration('publish_robot_root_tf'),
                'publish_output_tf': LaunchConfiguration('publish_output_tf'),
                'tf_lookup_max_age_ms': LaunchConfiguration('tf_lookup_max_age_ms'),
                'use_sim_time': LaunchConfiguration('use_sim_time')
            }
        ]
    )

    # 点云转换节点
    pointcloud_transformer_node = Node(
        package='open3d_loc',
        executable='pointcloud_transformer_node',
        name='pointcloud_transformer_node',
        output='screen',
        parameters=[{
            'input_topic': '/cloud_registered_body_1',
            'output_topic': '/cloud_registered_map',
            'global_map_topic': '/global_map',
            'source_frame': LaunchConfiguration('body_frame'),
            'target_frame': 'map',
            'voxel_leaf_size': 0.1,
            'map_voxel_leaf_size': 0.2,
            'max_global_points': 1000000,
            'map_publish_frequency': 1.0,
            'enable_global_map': True,
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )

    return LaunchDescription([
        use_sim_time_arg,
        map_file_arg,
        imu_frame_arg,
        body_frame_arg,
        output_frame_arg,
        publish_robot_root_tf_arg,
        publish_output_tf_arg,
        tf_lookup_max_age_arg,
        legacy_static_frames_arg,
        static_tf_imulink2baselink,
        static_tf_base_center,
        global_localization_node,
        # pointcloud_transformer_node
    ])
