"""Exercise map loss and manual reset with synthetic ROS inputs only."""

import os
import signal
import subprocess
import time

import pytest
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Float32, Header
from std_srvs.srv import Trigger
import yaml
from tracking_fixture import TrackingFixture


@pytest.mark.parametrize("fusion_enabled", [False, True])
def test_empty_map_crop_survives_and_manual_reset_recovers(
    tmp_path, monkeypatch, fusion_enabled
):
    monkeypatch.setenv("ROS_DOMAIN_ID", str(100 + os.getpid() % 100))
    executable = os.environ["GLOBAL_LOCALIZATION_EXECUTABLE"]
    points = []
    for i in range(15):
        for j in range(15):
            a, b = i * 0.2, j * 0.2
            points.extend([(a, b, 0.0), (a, 0.0, b), (0.0, a, b)])
    map_path = tmp_path / "synthetic.pcd"
    map_path.write_text(
        "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
        f"COUNT 1 1 1\nWIDTH {len(points)}\nHEIGHT 1\n"
        f"POINTS {len(points)}\nDATA ascii\n"
        + "".join(f"{x} {y} {z}\n" for x, y, z in points)
    )
    params = tmp_path / "params.yaml"
    params.write_text(
        yaml.safe_dump(
            {
                "/**": {
                    "ros__parameters": {
                        "path_map": str(map_path),
                        "initialpose": [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "imu_frame": "base_link",
                        "output_frame": "base_link",
                        "publish_output_tf": False,
                        "fusion.enabled": fusion_enabled,
                        "pcd_queue_maxsize": 1,
                        "loc_frequence": 0.1,
                        "voxelsize_fine": 0.1,
                        "kf_baselink2map/x": [0.001, 0.002],
                        "kf_baselink2map/y": [0.001, 0.002],
                        "kf_baselink2map/z": [0.001, 0.002],
                    }
                }
            }
        )
    )
    environment = os.environ.copy()
    environment.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)
    environment["ROS_LOG_DIR"] = str(tmp_path / "ros_logs")
    environment["OMP_NUM_THREADS"] = "1"
    rclpy.init()
    node = rclpy.create_node("registration_reset_test")
    tracking = TrackingFixture(node)
    odom_pub = node.create_publisher(Odometry, "/Odometry_loc", 10)
    scan_pub = node.create_publisher(PointCloud2, "/cloud_registered_1", 10)
    reset_pub = node.create_publisher(
        PoseWithCovarianceStamped, "/initialpose", 10
    )
    confidence = []
    diagnostics = []
    poses = []
    pose_stamps = []
    stamp_offset_ns = 0
    lag_inputs = False
    delayed_odometry = None

    def receive_pose(message):
        poses.append(message.pose.pose.position.x)
        pose_stamps.append(
            rclpy.time.Time.from_msg(message.header.stamp).nanoseconds
        )

    subscriptions = [
        node.create_subscription(
            Float32,
            "/localization_3d_confidence",
            lambda msg: confidence.append(msg.data),
            10,
        ),
        node.create_subscription(
            DiagnosticArray,
            "/localization_3d_diagnostics",
            lambda msg: diagnostics.extend(msg.status),
            10,
        ),
        node.create_subscription(
            Odometry,
            "/localization_3d_odom",
            receive_pose,
            10,
        ),
    ]
    log_path = tmp_path / "node.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [executable, "--ros-args", "--params-file", str(params)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        try:

            def pump_until(
                predicate,
                reset_x=None,
                require_fresh_pose=False,
                cloud_points=points,
                timeout=15.0,
            ):
                nonlocal delayed_odometry
                deadline = time.monotonic() + timeout
                reset_sent = False
                reset_stamp = 0
                while time.monotonic() < deadline:
                    assert process.poll() is None, log_path.read_text()
                    stamp = rclpy.time.Time(
                        nanoseconds=node.get_clock().now().nanoseconds
                        + stamp_offset_ns
                    ).to_msg()
                    odom = Odometry()
                    odom.header = Header(stamp=stamp, frame_id="odom")
                    odom.child_frame_id = "base_link"
                    odom.pose.pose.orientation.w = 1.0
                    tracking.publish(stamp)
                    if lag_inputs:
                        # Let the new status arrive before data accepted by
                        # the previous status, as with differing topic delays.
                        time.sleep(0.03)
                        delayed_odometry, odom = odom, delayed_odometry
                    if odom is not None:
                        odom_pub.publish(odom)
                        scan_pub.publish(
                            create_cloud_xyz32(odom.header, cloud_points)
                        )
                    if (
                        reset_x is not None
                        and not reset_sent
                        and reset_pub.get_subscription_count()
                    ):
                        reset_sent = True
                        reset_stamp = rclpy.time.Time.from_msg(
                            stamp
                        ).nanoseconds
                        reset = PoseWithCovarianceStamped()
                        reset.header = Header(stamp=stamp, frame_id="map")
                        reset.pose.pose.position.x = reset_x
                        reset.pose.pose.orientation.w = 1.0
                        reset_pub.publish(reset)
                    rclpy.spin_once(node, timeout_sec=0.05)
                    if predicate() and (
                        not require_fresh_pose
                        or (pose_stamps and pose_stamps[-1] > reset_stamp)
                    ):
                        return reset_stamp
                pytest.fail(log_path.read_text())

            def saw_failure():
                return any(
                    status.name == "localization_registration"
                    for status in diagnostics
                )

            # Startup outside the map stays in initialization until reset.
            pump_until(saw_failure)

            # Sparse scans must not initialize, even with perfect overlap.
            pump_until(
                lambda: tracking.resets >= 1,
                reset_x=0.0,
                cloud_points=points[:6],
            )
            for _ in range(10):
                pump_until(lambda: True, cloud_points=points[:6])
            assert not poses

            # Repeating one good scan cannot count as three confirmations.
            single_stamp = node.get_clock().now().to_msg()
            single_odom = Odometry(
                header=Header(stamp=single_stamp, frame_id="odom")
            )
            single_odom.child_frame_id = "base_link"
            single_odom.pose.pose.orientation.w = 1.0
            for _ in range(10):
                tracking.publish(single_stamp)
                odom_pub.publish(single_odom)
                scan_pub.publish(
                    create_cloud_xyz32(single_odom.header, points)
                )
                rclpy.spin_once(node, timeout_sec=0.05)
            assert not poses

            # Every pose request resets the local estimator before global ICP.
            # A reset repopulates the crop and valid ICP resumes.
            confidence.clear()
            poses.clear()
            # A reset permits sensor time to restart before the scan that was
            # already used for an initialization confirmation.
            stamp_offset_ns = -60_000_000_000
            pump_until(
                lambda: poses and abs(poses[-1]) < 0.1,
                reset_x=0.0,
                require_fresh_pose=True,
                timeout=5.0,
            )
            stamp_offset_ns = 0
            recent_stamp = node.get_clock().now().nanoseconds - 1_000_000_000
            pump_until(
                lambda: confidence and confidence[-1] > 0.9
                and pose_stamps[-1] > recent_stamp
            )

            # An independently restarted backend must hold its old global
            # output until three fresh global registrations succeed.
            restart_stamp = node.get_clock().now().nanoseconds
            tracking.instance_id += 1
            tracking.generation = 0
            diagnostics.clear()
            pump_until(
                lambda: any(
                    status.name == "open3d_loc/reset_tracking"
                    and status.message == "waiting_for_global_registration"
                    for status in diagnostics
                ),
                cloud_points=points[:6],
            )
            for _ in range(5):
                pump_until(lambda: True, cloud_points=points[:6])
            assert pose_stamps[-1] <= restart_stamp
            pump_until(lambda: pose_stamps[-1] > restart_stamp)

            # A sustained one-scan delay between status and sensor topics
            # must still permit coordinated reset and global initialization.
            lag_inputs = True
            pump_until(
                lambda: poses and abs(poses[-1]) < 0.1,
                reset_x=0.0,
                require_fresh_pose=True,
                timeout=3.0,
            )
            lag_inputs = False

            # Reproduce loss after successful registration, then recover again.
            diagnostics.clear()
            confidence.clear()
            previous_resets = tracking.resets
            pump_until(
                lambda: saw_failure() and tracking.resets > previous_resets,
                reset_x=1000.0,
            )
            assert process.poll() is None
            pump_until(
                lambda: poses and abs(poses[-1]) < 0.1,
                reset_x=0.0,
                require_fresh_pose=True,
            )
            confidence.clear()
            pump_until(lambda: confidence and confidence[-1] > 0.9)
            assert tracking.resets >= 4

            # A backend reset failure cannot resume publishing fresh poses.
            tracking.succeed = False
            diagnostics.clear()
            failure_stamp = pump_until(
                lambda: any(
                    any(
                        value.key == "reason"
                        and "FAST-LIO reset failed" in value.value
                        for value in status.values
                    )
                    for status in diagnostics
                ),
                reset_x=0.0,
            )
            for _ in range(3):
                rclpy.spin_once(node, timeout_sec=0.05)
            for _ in range(3):
                pump_until(lambda: True)
            assert pose_stamps[-1] <= failure_stamp
            tracking.succeed = True
            pump_until(
                lambda: poses and abs(poses[-1]) < 0.1,
                reset_x=0.0,
                require_fresh_pose=True,
            )
            node.destroy_service(tracking.service)
            diagnostics.clear()
            pump_until(
                lambda: any(
                    any(
                        value.key == "reason"
                        and "service unavailable" in value.value
                        for value in status.values
                    )
                    for status in diagnostics
                ),
                reset_x=0.0,
            )
            tracking.service = node.create_service(
                Trigger, "/fast_lio/reset_tracking", tracking.reset
            )
            pump_until(
                lambda: poses and abs(poses[-1]) < 0.1,
                reset_x=0.0,
                require_fresh_pose=True,
            )
            # Two queued requests must not let the older far-away pose win.
            previous_resets = tracking.resets
            request_stamp = node.get_clock().now().nanoseconds
            for reset_x in (1000.0, 0.0):
                reset = PoseWithCovarianceStamped()
                reset.header.frame_id = "map"
                reset.pose.pose.position.x = reset_x
                reset.pose.pose.orientation.w = 1.0
                reset_pub.publish(reset)
            pump_until(
                lambda: tracking.resets >= previous_resets + 2
                and pose_stamps[-1] > request_stamp
                and abs(poses[-1]) < 0.1
            )
            assert any(
                status.name == "open3d_loc/reset_tracking"
                and status.message == "relocalization_complete"
                for status in diagnostics
            )
        finally:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            for subscription in subscriptions:
                node.destroy_subscription(subscription)
            node.destroy_node()
            rclpy.shutdown()
