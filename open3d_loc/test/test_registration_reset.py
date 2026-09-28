"""Exercise map loss and manual reset with synthetic ROS inputs only."""

import os
import signal
import subprocess
import time

import pytest
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Float32, Header
import yaml


def test_empty_map_crop_survives_and_manual_reset_recovers(
    tmp_path, monkeypatch
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
    params.write_text(yaml.safe_dump({"/**": {
        "ros__parameters": {
            "path_map": str(map_path),
            "initialpose": [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "imu_frame": "base_link",
            "output_frame": "base_link",
            "publish_output_tf": False,
            "pcd_queue_maxsize": 1,
            "loc_frequence": 0.1,
            "voxelsize_fine": 0.1,
            "kf_baselink2map/x": [0.001, 0.002],
            "kf_baselink2map/y": [0.001, 0.002],
            "kf_baselink2map/z": [0.001, 0.002],
        }
    }}))
    environment = os.environ.copy()
    environment.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)
    environment["ROS_LOG_DIR"] = str(tmp_path / "ros_logs")
    environment["OMP_NUM_THREADS"] = "1"
    rclpy.init()
    node = rclpy.create_node("registration_reset_test")
    odom_pub = node.create_publisher(Odometry, "/Odometry_loc", 10)
    scan_pub = node.create_publisher(PointCloud2, "/cloud_registered_1", 10)
    reset_pub = node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
    confidence = []
    diagnostics = []
    poses = []
    subscriptions = [
        node.create_subscription(Float32, "/localization_3d_confidence",
                                 lambda msg: confidence.append(msg.data), 10),
        node.create_subscription(DiagnosticArray, "/localization_3d_diagnostics",
                                 lambda msg: diagnostics.extend(msg.status), 10),
        node.create_subscription(PoseStamped, "/localization_3d",
                                 lambda msg: poses.append(msg.pose.position.x), 10),
    ]
    log_path = tmp_path / "node.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [executable, "--ros-args", "--params-file", str(params)],
            stdout=log, stderr=subprocess.STDOUT, env=environment,
        )
        try:
            def pump_until(predicate, reset_x=None):
                deadline = time.monotonic() + 15.0
                while time.monotonic() < deadline:
                    assert process.poll() is None, log_path.read_text()
                    stamp = node.get_clock().now().to_msg()
                    odom = Odometry()
                    odom.header = Header(stamp=stamp, frame_id="odom")
                    odom.child_frame_id = "base_link"
                    odom.pose.pose.orientation.w = 1.0
                    odom_pub.publish(odom)
                    scan_pub.publish(create_cloud_xyz32(odom.header, points))
                    if reset_x is not None:
                        reset = PoseWithCovarianceStamped()
                        reset.header = Header(stamp=stamp, frame_id="map")
                        reset.pose.pose.position.x = reset_x
                        reset.pose.pose.orientation.w = 1.0
                        reset_pub.publish(reset)
                    rclpy.spin_once(node, timeout_sec=0.05)
                    time.sleep(0.02)
                    if predicate():
                        return
                pytest.fail(log_path.read_text())

            def saw_failure():
                return any(status.name == "localization_registration"
                           for status in diagnostics)

            # Startup outside the map stays in initialization until reset.
            pump_until(saw_failure)

            # A reset repopulates the crop and valid ICP resumes.
            confidence.clear()
            poses.clear()
            pump_until(lambda: poses and abs(poses[-1]) < 0.1, reset_x=0.0)
            pump_until(lambda: confidence and confidence[-1] > 0.9)

            # Reproduce loss after successful registration, then recover again.
            diagnostics.clear()
            confidence.clear()
            pump_until(lambda: saw_failure() and confidence
                       and confidence[-1] == 0.0, reset_x=1000.0)
            assert process.poll() is None
            pump_until(lambda: poses and abs(poses[-1]) < 0.1, reset_x=0.0)
            confidence.clear()
            pump_until(lambda: confidence and confidence[-1] > 0.9)
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
