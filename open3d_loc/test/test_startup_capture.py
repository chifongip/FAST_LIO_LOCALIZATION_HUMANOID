"""Check the read-only capture against an isolated synthetic IMU publisher."""

import json
import os
import subprocess
import sys
import time

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu


def test_capture_compares_qos_and_preserves_existing_output(tmp_path, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", str(170 + os.getpid() % 30))
    monkeypatch.delenv("FASTRTPS_DEFAULT_PROFILES_FILE", raising=False)
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "logs"))
    rclpy.init()
    node = rclpy.create_node("startup_capture_test")
    publisher = node.create_publisher(
        Imu, "/test/capture_imu",
        QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE),
    )
    capture = tmp_path / "capture.jsonl"
    command = [
        sys.executable, os.environ["STARTUP_CAPTURE_SCRIPT"],
        "--output", str(capture), "--duration", "2",
        "--imu-topic", "/test/capture_imu",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 8
        while process.poll() is None and time.monotonic() < deadline:
            message = Imu()
            message.header.stamp = node.get_clock().now().to_msg()
            message.linear_acceleration.z = 9.81
            publisher.publish(message)
            rclpy.spin_once(node, timeout_sec=0.005)
        stdout, stderr = process.communicate(timeout=3)
        assert process.returncode == 0, stderr.decode()
        summary = json.loads(stdout)
        assert summary["counts"]["imu_best_effort"] > 10
        assert summary["counts"]["imu_reliable"] > 10
        rows = [json.loads(line) for line in capture.read_text().splitlines()]
        assert rows[0]["kind"] == "metadata"
        assert rows[-1]["kind"] == "summary"
        assert all(row["arrival_monotonic_ns"] > 0 for row in rows)
        original = capture.read_bytes()
        duplicate = subprocess.run(command, capture_output=True, timeout=3)
        assert duplicate.returncode != 0
        assert capture.read_bytes() == original
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        node.destroy_node()
        rclpy.shutdown()
