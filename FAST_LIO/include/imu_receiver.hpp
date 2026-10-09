#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

namespace fast_lio
{
// This executor never runs estimator work, even when the main executor is busy.
class ImuReceiver
{
public:
  ImuReceiver(rclcpp::Node & node, const std::string & topic, int depth,
    std::function<void(sensor_msgs::msg::Imu::UniquePtr)> callback, bool reliable = false)
  {
    group_ = node.create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive, false);
    rclcpp::SubscriptionOptions options;
    options.callback_group = group_;
    auto qos = rclcpp::SensorDataQoS().keep_last(depth);
    if (reliable) qos.reliable();
    subscription_ = node.create_subscription<sensor_msgs::msg::Imu>(
      topic, qos, std::move(callback), options);
    executor_.add_callback_group(group_, node.get_node_base_interface());
    thread_ = std::thread([this]() {
      // Bounded spin_once also makes stop safe before this thread starts spinning.
      while (!stopping_.load() && rclcpp::ok())
        executor_.spin_once(std::chrono::milliseconds(10));
    });
  }
  ~ImuReceiver() {stop();}
  void stop()
  {
    stopping_.store(true);
    executor_.cancel();
    if (thread_.joinable()) thread_.join();
  }
private:
  rclcpp::CallbackGroup::SharedPtr group_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subscription_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::atomic<bool> stopping_{false};
  std::thread thread_;
};
}  // namespace fast_lio
