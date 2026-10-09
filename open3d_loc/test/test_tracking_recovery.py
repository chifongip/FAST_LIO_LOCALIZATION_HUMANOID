"""Exercise both real estimators with synthetic stationary E1R/IMU inputs."""

import math
import os
import signal
import subprocess
import time

import pytest
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from fast_lio.msg import TrackingStatus
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud
from std_msgs.msg import Header
from std_srvs.srv import Trigger
import yaml


def test_obstruction_loss_and_initialpose_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", str(170 + os.getpid() % 30))
    monkeypatch.delenv("FASTRTPS_DEFAULT_PROFILES_FILE", raising=False)
    environment = os.environ.copy()
    environment["ROS_LOG_DIR"] = str(tmp_path / "ros_logs")
    environment["OMP_NUM_THREADS"] = "1"
    points = []
    for i in range(21):
        for j in range(21):
            a, b = -2 + i * 0.2, -2 + j * 0.2
            points.extend([(a, b, -0.8), (2.5, a, b), (a, 2.5, b)])
    map_path = tmp_path / "map.pcd"
    map_path.write_text(
        "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
        f"COUNT 1 1 1\nWIDTH {len(points)}\nHEIGHT 1\n"
        f"POINTS {len(points)}\nDATA ascii\n"
        + "".join(f"{x} {y} {z}\n" for x, y, z in points)
    )
    fast_parameters = {
        "tracking.startup_warning_timeout": 0.2,
        "common.lid_topic": "/test/lidar",
        "common.imu_topic": "/test/imu",
        "preprocess.lidar_type": 5,
        "preprocess.scan_line": 96,
        "preprocess.timestamp_unit": 3,
        "preprocess.scan_rate": 10,
        "preprocess.blind": 0.1,
        "point_filter_num": 1,
        "filter_size_surf": 0.1,
        "filter_size_map": 0.1,
        "cube_side_length": 100.0,
        "mapping.det_range": 50.0,
        "mapping.fov_degree": 360.0,
        "mapping.extrinsic_est_en": False,
        "mapping.extrinsic_T": [0.0] * 3,
        "mapping.extrinsic_R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "publish.body_frame": "base_link",
        "publish.tf_en": False,
        "publish.scan_publish_en": True,
        "publish.dense_publish_en": False,
        "publish.scan_bodyframe_pub_en": True,
        "publish.map_en": False,
    }
    global_parameters = {
        "path_map": str(map_path),
        "initialpose": [0.0] * 6,
        "imu_frame": "base_link",
        "output_frame": "base_link",
        "publish_output_tf": False,
        "publish_robot_root_tf": True,
        "fusion.enabled": True,
        "fusion.update_mask": [True] * 6,
        "pcd_queue_maxsize": 1,
        "loc_frequence": 0.1,
        "voxelsize_fine": 0.1,
        "threshold_fitness": 0.8,
        "threshold_fitness_init": 0.8,
        "kf_baselink2map/x": [0.001, 0.002],
        "kf_baselink2map/y": [0.001, 0.002],
        "kf_baselink2map/z": [0.001, 0.002],
        "open3d_verbosity": "error",
    }
    rclpy.init()
    node = rclpy.create_node("tracking_recovery_test")
    imu_pub = node.create_publisher(
        Imu, "/test/imu",
        QoSProfile(depth=1000, reliability=ReliabilityPolicy.BEST_EFFORT),
    )
    cloud_pub = node.create_publisher(PointCloud2, "/test/lidar", 10)
    pose_pub = node.create_publisher(
        PoseWithCovarianceStamped, "/initialpose", 10
    )
    statuses, odometry, localization = [], [], []
    subscriptions = [
        node.create_subscription(
            TrackingStatus,
            "/fast_lio/tracking_status",
            statuses.append,
            QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        ),
        node.create_subscription(
            Odometry, "/Odometry_loc", odometry.append, 100
        ),
        node.create_subscription(
            Odometry, "/localization_3d_odom", localization.append, 100
        ),
    ]
    fields = [
        PointField(name=name, offset=offset, datatype=datatype, count=1)
        for name, offset, datatype in [
            ("x", 0, PointField.FLOAT32),
            ("y", 4, PointField.FLOAT32),
            ("z", 8, PointField.FLOAT32),
            ("intensity", 12, PointField.FLOAT32),
            ("ring", 16, PointField.UINT16),
            ("timestamp", 18, PointField.FLOAT64),
        ]
    ]
    processes, logs = [], []
    last_imu_ns = 0

    diagnostics = []
    node.create_subscription(
        DiagnosticArray, "/fast_lio/diagnostics", diagnostics.append, 20
    )

    def stamp(ns):
        return rclpy.time.Time(nanoseconds=ns).to_msg()

    def pump(cloud_points=points, acceleration=(0.0, 0.0, 9.81), publish_imu=True):
        nonlocal last_imu_ns
        cycle_started = time.monotonic()
        begin = node.get_clock().now().nanoseconds - 110_000_000
        end = begin + 90_000_000
        if not publish_imu:
            last_imu_ns = 0
        if not last_imu_ns and publish_imu:
            last_imu_ns = begin - 5_000_000
        while publish_imu and last_imu_ns < end + 5_000_000:
            last_imu_ns += 5_000_000
            imu = Imu(
                header=Header(stamp=stamp(last_imu_ns), frame_id="base_link")
            )
            (
                imu.linear_acceleration.x,
                imu.linear_acceleration.y,
                imu.linear_acceleration.z,
            ) = acceleration
            imu_pub.publish(imu)
            time.sleep(0.001)
        cloud = create_cloud(
            Header(stamp=stamp(begin), frame_id="lidar"),
            fields,
            [
                (
                    x,
                    y,
                    z,
                    1.0,
                    i % 96,
                    90_000_000.0 * i / max(1, len(cloud_points) - 1),
                )
                for i, (x, y, z) in enumerate(cloud_points)
            ],
        )
        cloud_pub.publish(cloud)
        deadline = cycle_started + 0.1
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.005)
        for process in processes:
            assert process.poll() is None, "\n".join(
                p.read_text() for p in tmp_path.glob("*.log")
            )

    def until(predicate, timeout=15.0, **kwargs):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pump(**kwargs)
            if predicate():
                return
        pytest.fail(
            "statuses="
            + repr(
                [
                    (
                        s.state,
                        s.reason,
                        s.effective_features,
                        s.information_ratio,
                    )
                    for s in statuses[-10:]
                ]
            )
            + "\n"
            + "\n".join(p.read_text()[-12000:] for p in tmp_path.glob("*.log"))
        )

    def request_pose(x=0.0, yaw=0.0):
        request = PoseWithCovarianceStamped()
        request.header = Header(
            stamp=node.get_clock().now().to_msg(), frame_id="map"
        )
        request.pose.pose.position.x = x
        request.pose.pose.orientation.z = math.sin(yaw / 2)
        request.pose.pose.orientation.w = math.cos(yaw / 2)
        pose_pub.publish(request)

    try:
        for name, executable, parameters in [
            ("fast", os.environ["FAST_LIO_EXECUTABLE"], fast_parameters),
            (
                "global",
                os.environ["GLOBAL_LOCALIZATION_EXECUTABLE"],
                global_parameters,
            ),
        ]:
            parameter_path = tmp_path / f"{name}.yaml"
            parameter_path.write_text(
                yaml.safe_dump({"/**": {"ros__parameters": parameters}})
            )
            log = (tmp_path / f"{name}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        executable,
                        "--ros-args",
                        "--params-file",
                        str(parameter_path),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
            )
        # Cold start with LiDAR before any IMU: retry fresh windows without
        # requiring the operator to reset or changing the external generation.
        deadline = time.monotonic() + 5
        while not (cloud_pub.get_subscription_count() and imu_pub.get_subscription_count()):
            assert time.monotonic() < deadline
            rclpy.spin_once(node, timeout_sec=0.01)
        for _ in range(5):
            pump(publish_imu=False)
        # Volatile diagnostics discovery can lag the sensor subscriptions.
        until(
            lambda: any(
                entry.level == DiagnosticStatus.ERROR
                and entry.message == "startup_wait_timeout"
                for message in diagnostics for entry in message.status
            ),
            timeout=5,
            publish_imu=False,
        )
        assert not odometry
        until(
            lambda: localization
            and statuses[-1].state == TrackingStatus.TRACKING
        )
        assert statuses[-1].generation == 0
        assert any(
            float(value.value) >= 1
            for message in diagnostics for entry in message.status
            for value in entry.values if value.key == "startup_retries"
        )
        good_generation = statuses[-1].generation
        good_position = odometry[-1].pose.pose.position
        assert abs(good_position.z) < 0.05

        # Short loss recovers against the preserved local map without reset.
        pump(cloud_points=points[::300])
        assert statuses[-1].state == TrackingStatus.DEGRADED
        until(lambda: statuses[-1].state == TrackingStatus.TRACKING, timeout=2)
        assert statuses[-1].generation == good_generation

        # Many points on one plane still cannot constrain all six pose axes.
        plane_interior = [
            point
            for point in points[::3]
            if abs(point[0]) < 1.3 and abs(point[1]) < 1.3
        ]
        pump(cloud_points=plane_interior)
        assert statuses[-1].state == TrackingStatus.DEGRADED
        assert statuses[-1].reason == "degenerate_geometry"
        until(lambda: statuses[-1].state == TrackingStatus.TRACKING, timeout=2)

        # An actual post-start IMU outage stays latched despite fresh data.
        for _ in range(3):
            pump(publish_imu=False)
        until(lambda: statuses[-1].state == TrackingStatus.LOST)
        assert statuses[-1].reason in ("missing_imu_coverage", "imu_time_gap")
        lost_values = diagnostics[-1].status[0].values
        for _ in range(3):
            pump()
        assert statuses[-1].state == TrackingStatus.LOST
        assert diagnostics[-1].status[0].values == lost_values
        generation = statuses[-1].generation
        request_pose()
        until(
            lambda: statuses[-1].generation > generation
            and statuses[-1].state == TrackingStatus.TRACKING
        )
        good_generation = statuses[-1].generation

        # Prolonged loss holds the last accepted output, including timestamp.
        until(
            lambda: statuses[-1].state == TrackingStatus.LOST,
            cloud_points=points[::300],
            acceleration=(0.1, 0.0, 9.81),
        )
        held = odometry[-1]
        for _ in range(5):
            pump()
        assert statuses[-1].state == TrackingStatus.LOST
        assert odometry[-1] == held
        assert (
            max(abs(getattr(held.pose.pose.position, axis)) for axis in "xyz")
            < 1.0
        )

        # Reset the real local estimator, then re-localize.
        localization.clear()
        reset_time = node.get_clock().now().nanoseconds
        request_pose()
        until(
            lambda: statuses[-1].generation > good_generation
            and statuses[-1].state == TrackingStatus.TRACKING
            and localization
            and rclpy.time.Time.from_msg(
                localization[-1].header.stamp
            ).nanoseconds
            > reset_time
        )
        assert abs(localization[-1].pose.pose.position.x) < 0.1

        # Healthy tracking also resets, including a nonzero requested yaw.
        generation = statuses[-1].generation
        localization.clear()
        reset_time = node.get_clock().now().nanoseconds
        request_pose(yaw=0.1)
        until(
            lambda: statuses[-1].generation > generation
            and localization
            and rclpy.time.Time.from_msg(
                localization[-1].header.stamp
            ).nanoseconds
            > reset_time
        )

        # Invalid requests leave the healthy generation intact.
        generation = statuses[-1].generation
        invalid = PoseWithCovarianceStamped(header=Header(frame_id="map"))
        invalid.pose.pose.orientation.w = 0.0
        pose_pub.publish(invalid)
        for _ in range(3):
            pump()
        assert statuses[-1].generation == generation

        # Timestamp regression cannot be integrated into an extreme state.
        stale = Imu(header=Header(stamp=stamp(last_imu_ns - 1_000_000_000)))
        stale.linear_acceleration.z = 9.81
        imu_pub.publish(stale)
        until(lambda: statuses[-1].state == TrackingStatus.LOST)
        assert statuses[-1].reason == "timestamp_regression"
        reset_client = node.create_client(
            Trigger, "/localization/reset_tracking"
        )
        assert reset_client.wait_for_service(timeout_sec=2)
        future = reset_client.call_async(Trigger.Request())
        until(future.done)
        assert future.result().success
        until(
            lambda: statuses[-1].generation > generation
            and statuses[-1].state == TrackingStatus.TRACKING
        )
        # An extreme IMU prediction must never replace the held trusted pose.
        held = odometry[-1]
        until(
            lambda: statuses[-1].state == TrackingStatus.LOST,
            acceleration=(0.0, 0.0, -1000.0),
        )
        assert statuses[-1].reason == "implausible_prediction"
        assert odometry[-1] == held

        # A backend restart has a new instance and generation zero.
        instance = statuses[-1].instance_id
        processes[0].send_signal(signal.SIGINT)
        processes[0].wait(timeout=5)
        restart_time = node.get_clock().now().nanoseconds
        localization.clear()
        last_imu_ns = 0
        processes[0] = subprocess.Popen(
            [
                os.environ["FAST_LIO_EXECUTABLE"],
                "--ros-args",
                "--params-file",
                str(tmp_path / "fast.yaml"),
            ],
            stdout=logs[0],
            stderr=subprocess.STDOUT,
            env=environment,
        )
        until(
            lambda: statuses[-1].instance_id != instance
            and statuses[-1].state == TrackingStatus.TRACKING
            and localization
            and rclpy.time.Time.from_msg(
                localization[-1].header.stamp
            ).nanoseconds
            > restart_time
        )
    finally:
        for process in processes:
            process.send_signal(signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for log in logs:
            log.close()
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()
