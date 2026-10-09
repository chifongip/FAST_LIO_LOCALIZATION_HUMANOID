#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <thread>
#include "imu_receiver.hpp"

using namespace std::chrono_literals;

class ImuReceiverTest : public ::testing::Test
{
protected:
  void SetUp() override {rclcpp::init(0, nullptr);}
  void TearDown() override {rclcpp::shutdown();}
};

TEST_F(ImuReceiverTest, ReceivesAt200HzWhileEstimatorBlocksWithSmallAndLargeQueues)
{
  for (int depth : {5, 200}) {
    auto node = std::make_shared<rclcpp::Node>("imu_receiver_test");
    auto source = std::make_shared<rclcpp::Node>("imu_source_test");
    auto publisher = source->create_publisher<sensor_msgs::msg::Imu>(
      "test_imu", rclcpp::SensorDataQoS().keep_last(200));
    std::atomic<int> received{0}, maximum_gap{0};
    int previous = -1;
    fast_lio::ImuReceiver receiver(*node, "test_imu", depth,
      [&](sensor_msgs::msg::Imu::UniquePtr msg) {
        const int sequence = msg->header.stamp.sec;
        if (previous >= 0) maximum_gap.store(std::max(maximum_gap.load(), sequence - previous));
        previous = sequence;
        ++received;
      });
    const auto discovery_deadline = std::chrono::steady_clock::now() + 3s;
    while (publisher->get_subscription_count() == 0 &&
      std::chrono::steady_clock::now() < discovery_deadline) std::this_thread::sleep_for(10ms);
    ASSERT_GT(publisher->get_subscription_count(), 0u);
    std::atomic<bool> stopped{false};
    std::thread producer([&]() {
      auto next = std::chrono::steady_clock::now();
      int sequence = 0;
      while (!stopped.load()) {
        sensor_msgs::msg::Imu msg;
        msg.header.stamp.sec = ++sequence;
        publisher->publish(msg);
        next += 5ms;
        std::this_thread::sleep_until(next);
      }
    });
    // Runs in the default group, just like LiDAR preprocessing and estimation.
    int blocking_calls = 0;
    auto timer = node->create_wall_timer(10ms, [&]() {
      ++blocking_calls;
      std::this_thread::sleep_for(150ms);
    });
    rclcpp::executors::SingleThreadedExecutor estimator;
    estimator.add_node(node);
    const auto deadline = std::chrono::steady_clock::now() + 1500ms;
    while (std::chrono::steady_clock::now() < deadline) estimator.spin_once(20ms);
    stopped.store(true);
    producer.join();
    receiver.stop();
    EXPECT_GE(blocking_calls, 8);
    EXPECT_GE(received.load(), 250);
    EXPECT_LE(maximum_gap.load(), 2) << "IMU reception blocked at queue depth " << depth;
  }
}

TEST_F(ImuReceiverTest, StopsSafelyBeforeExecutorStartsAndStopsTwice)
{
  auto node = std::make_shared<rclcpp::Node>("imu_shutdown_test");
  for (int i = 0; i < 40; ++i) {
    fast_lio::ImuReceiver receiver(*node, "test_imu", 5,
      [](sensor_msgs::msg::Imu::UniquePtr) {});
    receiver.stop();
    receiver.stop();
  }
}
