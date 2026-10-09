#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace open3d_loc
{
// Owned by the ROS executor. Registration workers never mutate this policy.
class StartupRecovery
{
public:
  bool enabled = true;
  double retry_backoff = 1.0;
  double warning_timeout = 10.0;
  std::string phase = "waiting_for_local_tracking";
  std::string last_failure;
  std::uint64_t retries = 0;
  bool acquiring = true;
  bool ready = false;

  void validate() const
  {
    if (!std::isfinite(retry_backoff) || retry_backoff <= 0.0 ||
      !std::isfinite(warning_timeout) || warning_timeout <= 0.0)
      throw std::invalid_argument("startup recovery times must be finite and positive");
  }
  void begin(double now)
  {
    acquiring = true; ready = false; blocked_ = false; pending_ = false;
    retries = 0; started_ = now; have_failure_ = false; last_failure.clear();
    phase = "waiting_for_local_tracking";
  }
  static bool retryable(const std::string & reason)
  {
    return reason == "imu_time_gap" || reason == "missing_imu_coverage" ||
      reason == "scan_time_gap" || reason == "initialization_timeout" ||
      reason == "prediction_timeout";
  }
  void lost(std::uint64_t instance, std::uint64_t generation,
    const std::string & reason, double now)
  {
    ready = false;
    if (!acquiring) {last_failure = reason; phase = "tracking_unavailable"; return;}
    if (blocked_) return;
    if (!retryable(reason)) {fault(reason); return;}
    if (!enabled) {fault(reason); return;}
    if (have_failure_ && instance == failed_instance_ && generation <= failed_generation_) return;
    have_failure_ = true; failed_instance_ = instance; failed_generation_ = generation;
    last_failure = reason; pending_ = true; due_ = now + retry_backoff;
    phase = "retry_backoff";
  }
  bool take_retry(double now, bool reset_busy)
  {
    if (!acquiring || blocked_ || !pending_ || reset_busy || now < due_) return false;
    pending_ = false; ++retries; phase = "reset_in_progress";
    return true;
  }
  void waiting(const std::string & next)
  {
    if (acquiring && !blocked_ && !pending_) phase = next;
  }
  void complete()
  {
    if (blocked_ || pending_) return;
    acquiring = false; ready = true; phase = "ready";
  }
  void unavailable()
  {
    ready = false;
    if (!acquiring) phase = "tracking_unavailable";
  }
  void fault(const std::string & reason)
  {
    ready = false; blocked_ = true; pending_ = false;
    last_failure = reason; phase = "fault";
  }
  bool error(double now) const
  {
    return blocked_ || (!ready && (!acquiring || now - started_ >= warning_timeout));
  }
private:
  bool pending_ = false, blocked_ = false;
  double started_ = 0.0, due_ = 0.0;
  bool have_failure_ = false;
  std::uint64_t failed_instance_ = 0, failed_generation_ = 0;
};
}  // namespace open3d_loc
