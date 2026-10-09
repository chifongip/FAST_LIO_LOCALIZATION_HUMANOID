"""Independent FAST-LIO status/reset stub for global-localization tests."""

from fast_lio.msg import TrackingStatus
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_srvs.srv import Trigger


class TrackingFixture:
    def __init__(self, node):
        self.node = node
        self.generation = 0
        self.instance_id = 1
        self.resets = 0
        self.succeed = True
        self.publisher = node.create_publisher(
            TrackingStatus,
            "/fast_lio/tracking_status",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.service = node.create_service(
            Trigger, "/fast_lio/reset_tracking", self.reset
        )

    def reset(self, request, response):
        self.resets += 1
        response.success = self.succeed
        response.message = (
            f"reset_generation={self.generation + 1}; "
            f"instance_id={self.instance_id}; synthetic reset"
        )
        if self.succeed:
            self.generation += 1
            self.publish(
                self.node.get_clock().now().to_msg(),
                TrackingStatus.INITIALIZING,
            )
        return response

    def publish(self, stamp, state=TrackingStatus.TRACKING):
        message = TrackingStatus()
        message.header.stamp = stamp
        message.state = state
        message.instance_id = self.instance_id
        message.generation = self.generation
        if state == TrackingStatus.TRACKING:
            message.last_accepted_stamp = stamp
        self.publisher.publish(message)
