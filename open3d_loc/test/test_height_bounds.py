"""Check bounded map height against TF while preserving local odometry."""

import math
import os
import signal
import subprocess
import time

import pytest
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Float32, Header
from tf2_msgs.msg import TFMessage
from tf2_ros import StaticTransformBroadcaster
import yaml


def xyz(vector):
    return (vector.x, vector.y, vector.z)


def quaternion(rotation):
    return (rotation.x, rotation.y, rotation.z, rotation.w)


def quaternion_product(left, right):
    x1, y1, z1, w1 = left
    x2, y2, z2, w2 = right
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def rotate(rotation, vector):
    q = quaternion(rotation)
    product = quaternion_product(
        quaternion_product(q, (*vector, 0.0)), (-q[0], -q[1], -q[2], q[3])
    )
    return product[:3]


def assert_quaternion(actual, expected):
    actual = quaternion(actual)
    sign = 1 if sum(a * b for a, b in zip(actual, expected)) >= 0 else -1
    assert actual == pytest.approx(tuple(sign * v for v in expected), abs=1e-6)


@pytest.mark.parametrize(
    "bounds_enabled,floor_z,all_axes",
    [
        (False, -0.5, False),
        (True, -0.5, False),
        (True, 0.0, False),
        (True, -0.5, True),
    ],
)
def test_height_bounds_tf_and_pose_agree(
    tmp_path, monkeypatch, bounds_enabled, floor_z, all_axes
):
    monkeypatch.setenv("ROS_DOMAIN_ID", str(150 + os.getpid() % 50))
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
                        "initialpose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "imu_frame": "base_link",
                        "output_frame": (
                            "test_torso" if all_axes else "base_link"
                        ),
                        "publish_output_tf": False,
                        "fusion.enabled": True,
                        "fusion.update_mask": (
                            [True] * 6
                            if all_axes
                            else [True, True, False, False, False, True]
                        ),
                        "height_bounds.enabled": bounds_enabled,
                        "height_bounds.floor_z": floor_z,
                        "height_bounds.min_height": 0.3,
                        "height_bounds.max_height": 0.7,
                        "publish_robot_root_tf": True,
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
    node = rclpy.create_node("height_bounds_test")
    odom_pub = node.create_publisher(Odometry, "/Odometry_loc", 10)
    scan_pub = node.create_publisher(PointCloud2, "/cloud_registered_1", 10)
    reset_pub = node.create_publisher(
        PoseWithCovarianceStamped, "/initialpose", 10
    )
    output_offset = (0.0, 0.0, 0.2) if all_axes else (0.0, 0.0, 0.0)
    static_broadcaster = StaticTransformBroadcaster(node)
    if all_axes:
        torso = TransformStamped()
        torso.header.frame_id = "base_link"
        torso.child_frame_id = "test_torso"
        torso.transform.translation.z = output_offset[2]
        torso.transform.rotation.w = 1.0
        static_broadcaster.sendTransform(torso)
    confidence = []
    diagnostics = []
    poses = {}
    output_poses = {}
    transforms = {}
    last_correction = []

    def stamp_key(stamp):
        return stamp.sec, stamp.nanosec

    def receive_tf(message):
        for transform in message.transforms:
            key = stamp_key(transform.header.stamp)
            transforms.setdefault(key, {})[
                transform.child_frame_id
            ] = transform

    def receive_base(message):
        if bounds_enabled:
            height = message.pose.pose.position.z - floor_z
            assert 0.3 - 1e-9 <= height <= 0.7 + 1e-9
        poses[stamp_key(message.header.stamp)] = message.pose.pose

    def consistent(
        z,
        height,
        expected_x=0.0,
        xy=(0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        expected_yaw=None,
    ):
        for key, pose in poses.items():
            tf = transforms.get(key, {})
            if (
                "odom" not in tf
                or "base_link" not in tf
                or key not in output_poses
            ):
                continue
            local = tf["base_link"].transform
            if xyz(local.translation) != pytest.approx((*xy, z), abs=1e-9):
                continue
            if quaternion(local.rotation) != pytest.approx(
                orientation, abs=1e-9
            ):
                continue
            correction = tf["odom"].transform
            rotated = rotate(correction.rotation, xyz(local.translation))
            composed = tuple(
                a + b for a, b in zip(rotated, xyz(correction.translation))
            )
            assert xyz(pose.position) == pytest.approx(composed, abs=1e-6)
            expected_orientation = quaternion_product(
                quaternion(correction.rotation), quaternion(local.rotation)
            )
            assert_quaternion(pose.orientation, expected_orientation)
            output = output_poses[key]
            offset = rotate(pose.orientation, output_offset)
            assert xyz(output.position) == pytest.approx(
                tuple(a + b for a, b in zip(composed, offset)), abs=1e-6
            )
            assert_quaternion(output.orientation, expected_orientation)
            if expected_yaw is not None:
                desired = (
                    0.0,
                    0.0,
                    math.sin(expected_yaw / 2),
                    math.cos(expected_yaw / 2),
                )
                dot = sum(
                    a * b
                    for a, b in zip(quaternion(pose.orientation), desired)
                )
                if abs(dot) < 1.0 - 1e-10:
                    continue
            # Startup ICP can introduce small horizontal corrections.
            if (
                expected_x is not None
                and abs(pose.position.x - expected_x) > 0.01
            ):
                continue
            if composed[2] - floor_z == pytest.approx(height, abs=1e-4):
                last_correction[:] = [correction]
                return True
        return False

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
            lambda msg: output_poses.__setitem__(
                stamp_key(msg.header.stamp), msg.pose.pose
            ),
            10,
        ),
        node.create_subscription(Odometry, "/baselink2map", receive_base, 10),
        node.create_subscription(TFMessage, "/tf", receive_tf, 100),
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
                z=0.0,
                scans=True,
                xy=(0.0, 0.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
                reset_yaw=0.0,
            ):
                deadline = time.monotonic() + 15.0
                while time.monotonic() < deadline:
                    assert process.poll() is None, log_path.read_text()
                    stamp = node.get_clock().now().to_msg()
                    odom = Odometry()
                    odom.header = Header(stamp=stamp, frame_id="odom")
                    odom.child_frame_id = "base_link"
                    odom.pose.pose.position.x, odom.pose.pose.position.y = xy
                    odom.pose.pose.position.z = z
                    (
                        odom.pose.pose.orientation.x,
                        odom.pose.pose.orientation.y,
                        odom.pose.pose.orientation.z,
                        odom.pose.pose.orientation.w,
                    ) = orientation
                    odom_pub.publish(odom)
                    if scans:
                        scan_pub.publish(
                            create_cloud_xyz32(odom.header, points)
                        )
                    if reset_x is not None:
                        reset = PoseWithCovarianceStamped()
                        reset.header = Header(stamp=stamp, frame_id="map")
                        reset.pose.pose.position.x = reset_x
                        reset.pose.pose.orientation.z = math.sin(reset_yaw / 2)
                        reset.pose.pose.orientation.w = math.cos(reset_yaw / 2)
                        reset_pub.publish(reset)
                    spin_deadline = time.monotonic() + 0.05
                    while time.monotonic() < spin_deadline:
                        rclpy.spin_once(node, timeout_sec=0.005)
                    if predicate():
                        return
                pytest.fail(
                    f"poses={list(poses.items())[-2:]} "
                    f"tf={list(transforms.items())[-2:]}\n"
                    + log_path.read_text()
                )

            # Real synthetic ICP initialization, then freeze scans to isolate
            # odometry-driven projection from asynchronous registration.
            pump_until(lambda: confidence and confidence[-1] > 0.9)
            # Startup-only parameters must reject changes instead of accepting
            # values that differ from the node's active height constraint.
            parameter_client = node.create_client(
                SetParameters, "/global_loc_node/set_parameters"
            )
            assert parameter_client.wait_for_service(timeout_sec=2.0)
            request = SetParameters.Request(
                parameters=[
                    Parameter(
                        "height_bounds.enabled", value=not bounds_enabled
                    ).to_parameter_msg(),
                    Parameter(
                        "height_bounds.floor_z", value=floor_z + 1.0
                    ).to_parameter_msg(),
                    Parameter(
                        "height_bounds.min_height", value=0.4
                    ).to_parameter_msg(),
                    Parameter(
                        "height_bounds.max_height", value=0.8
                    ).to_parameter_msg(),
                ]
            )
            future = parameter_client.call_async(request)
            pump_until(future.done, scans=False)
            assert all(
                not result.successful for result in future.result().results
            )
            node.destroy_client(parameter_client)
            initial_height = max(0.3, -floor_z) if bounds_enabled else -floor_z
            pump_until(lambda: consistent(0.0, initial_height), scans=False)
            sequence = [(0.4, 0.7), (0.45, 0.7), (-0.6, 0.3), (-0.55, 0.35)]
            for z, bounded_height in sequence:
                poses.clear()
                transforms.clear()
                expected = bounded_height if bounds_enabled else z - floor_z
                pump_until(lambda: consistent(z, expected), z=z, scans=False)
            # A manual pose reset must preserve the current valid height.
            poses.clear()
            transforms.clear()
            expected = 0.35 if bounds_enabled else -0.55 - floor_z
            pump_until(
                lambda: consistent(-0.55, expected, expected_x=0.1),
                reset_x=0.1,
                z=-0.55,
                scans=False,
            )
            # A reset with articulated body attitude creates a genuinely
            # rotated map/odom correction. Horizontal motion then changes Z.
            attitude = quaternion_product(
                quaternion_product(
                    (0, 0, math.sin(0.05), math.cos(0.05)),
                    (0, math.sin(-0.075), 0, math.cos(-0.075)),
                ),
                (math.sin(0.1), 0, 0, math.cos(0.1)),
            )
            poses.clear()
            transforms.clear()
            pump_until(
                lambda: consistent(
                    -0.55,
                    expected,
                    expected_x=0.1,
                    orientation=attitude,
                    expected_yaw=0.4,
                ),
                reset_x=0.1,
                reset_yaw=0.4,
                z=-0.55,
                orientation=attitude,
                scans=False,
            )
            for xy in ((3.0, 3.0), (-3.0, -3.0)):
                before = last_correction[0]
                raw_height = (
                    rotate(before.rotation, (*xy, -0.55))[2]
                    + before.translation.z
                    - floor_z
                )
                expected = (
                    min(0.7, max(0.3, raw_height))
                    if bounds_enabled
                    else raw_height
                )
                poses.clear()
                transforms.clear()
                pump_until(
                    lambda: consistent(
                        -0.55,
                        expected,
                        expected_x=None,
                        xy=xy,
                        orientation=attitude,
                    ),
                    z=-0.55,
                    xy=xy,
                    orientation=attitude,
                    scans=False,
                )
                after = last_correction[0]
                assert xyz(after.translation)[:2] == pytest.approx(
                    xyz(before.translation)[:2], abs=1e-9
                )
                assert_quaternion(after.rotation, quaternion(before.rotation))
                # Check persistence on a subsequent filter prediction.
                poses.clear()
                transforms.clear()
                pump_until(
                    lambda: consistent(
                        -0.55,
                        expected,
                        expected_x=None,
                        xy=xy,
                        orientation=attitude,
                    ),
                    z=-0.55,
                    xy=xy,
                    orientation=attitude,
                    scans=False,
                )
            if bounds_enabled:
                assert any(
                    status.name == "open3d_loc/height_bounds"
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
