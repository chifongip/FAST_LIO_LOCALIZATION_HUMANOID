#!/usr/bin/env python3
"""Read-only, bounded JSONL capture of IMU transport and localization startup."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from diagnostic_msgs.msg import DiagnosticArray
from fast_lio.msg import TrackingStatus
from nav_msgs.msg import Odometry
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from tf2_msgs.msg import TFMessage


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument(
        "--imu-topic", default="/aima/hal/sensor/lidar_chest_front/imu"
    )
    args, ros_args = parser.parse_known_args()
    if not 0 < args.duration <= 600:
        parser.error("--duration must be finite, positive, and at most 600 seconds")
    # Exclusive creation prevents accidentally overwriting an earlier capture.
    with args.output.open("x", encoding="utf-8") as output:
        rclpy.init(args=ros_args)
        node = rclpy.create_node("localization_startup_capture")
        counts = Counter()
        stamps = {"best_effort": set(), "reliable": set()}
        previous = {}
        maximum_gap_ns = Counter()

        def record(kind, payload):
            counts[kind] += 1
            output.write(json.dumps({
                "kind": kind,
                "arrival_monotonic_ns": time.monotonic_ns(),
                "arrival_ros_ns": node.get_clock().now().nanoseconds,
                **payload,
            }) + "\n")

        def imu(message, mode):
            timestamp = stamp_ns(message.header.stamp)
            if mode in previous:
                maximum_gap_ns[mode] = max(
                    maximum_gap_ns[mode], timestamp - previous[mode]
                )
            previous[mode] = timestamp
            stamps[mode].add(timestamp)
            record("imu_" + mode, {
                "stamp_ns": timestamp,
                "acceleration": [getattr(message.linear_acceleration, a) for a in "xyz"],
                "angular_velocity": [getattr(message.angular_velocity, a) for a in "xyz"],
            })

        def diagnostics(message):
            record("diagnostics", {
                "stamp_ns": stamp_ns(message.header.stamp),
                "status": [
                    {"name": entry.name, "message": entry.message,
                     "level": int.from_bytes(entry.level, "little"),
                     "values": {v.key: v.value for v in entry.values}}
                    for entry in message.status
                ],
            })

        def tracking(message):
            record("tracking", {
                "stamp_ns": stamp_ns(message.header.stamp),
                "accepted_ns": stamp_ns(message.last_accepted_stamp),
                "instance": message.instance_id, "generation": message.generation,
                "state": message.state, "reason": message.reason,
            })

        subscriptions = []
        for mode, reliability in [
            ("best_effort", ReliabilityPolicy.BEST_EFFORT),
            ("reliable", ReliabilityPolicy.RELIABLE),
        ]:
            subscriptions.append(node.create_subscription(
                Imu, args.imu_topic, lambda m, label=mode: imu(m, label),
                QoSProfile(depth=1000, reliability=reliability),
            ))
        subscriptions.append(node.create_subscription(
            TrackingStatus, "/fast_lio/tracking_status", tracking,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        ))
        for topic in ["/fast_lio/diagnostics", "/localization_3d_diagnostics"]:
            subscriptions.append(node.create_subscription(
                DiagnosticArray, topic, diagnostics, 100
            ))
        for topic in ["/Odometry_loc", "/localization_3d_odom", "/odom"]:
            subscriptions.append(node.create_subscription(
                Odometry, topic,
                lambda m, label=topic: record(label, {
                    "stamp_ns": stamp_ns(m.header.stamp),
                    "position": [getattr(m.pose.pose.position, a) for a in "xyz"],
                }), 100,
            ))
        subscriptions.append(node.create_subscription(
            TFMessage, "/tf", lambda m: record("tf", {
                "transforms": [
                    [t.header.frame_id, t.child_frame_id, stamp_ns(t.header.stamp)]
                    for t in m.transforms
                ],
            }), 100,
        ))
        record("metadata", {
            "imu_topic": args.imu_topic, "duration": args.duration,
            "note": "Observer reception cannot reconstruct earlier startup loss.",
        })
        deadline = time.monotonic() + args.duration
        try:
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=min(0.1, max(0, deadline - time.monotonic())))
        except KeyboardInterrupt:
            pass
        finally:
            summary = {
                "counts": dict(counts),
                "maximum_imu_header_gap_ns": dict(maximum_gap_ns),
                "reliable_only_stamps": len(stamps["reliable"] - stamps["best_effort"]),
                "best_effort_only_stamps": len(stamps["best_effort"] - stamps["reliable"]),
                "publishers": [
                    {"node": endpoint.node_name,
                     "reliability": str(endpoint.qos_profile.reliability)}
                    for endpoint in node.get_publishers_info_by_topic(args.imu_topic)
                ],
                "note": "Discovery and capture boundaries can also cause unmatched stamps.",
            }
            record("summary", summary)
            print(json.dumps(summary, indent=2))
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
