#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>

namespace open3d_loc
{
// Executor-owned policy. Never uses estimator velocity to infer standstill.
class RuntimeRecovery
{
public:
  bool enabled = false, active = false, blocked = false, ever_ready = false;
  int max_attempts = 3, attempts = 0;
  double settle = 1.0, command_age = 0.20, imu_age = 0.05;
  double linear_limit = 0.01, angular_limit = 0.01, gyro_limit = 0.05;
  double gravity_tolerance = 0.5, acceleration_rms = 0.20;
  double anchor_age_limit = 3.0, retry_backoff = 2.0, attempt_timeout = 15.0;
  double budget_rearm = 30.0;
  std::string phase = "disabled", blocker = "", reason = "";
  double anchor_age = 0.0;

  void validate() const
  {
    const double values[] = {settle, command_age, imu_age, linear_limit, angular_limit,
      gyro_limit, gravity_tolerance, acceleration_rms, anchor_age_limit, retry_backoff,
      attempt_timeout, budget_rearm};
    for (auto v : values) if (!std::isfinite(v) || v <= 0.0)
      throw std::invalid_argument("runtime recovery limits must be finite and positive");
    if (max_attempts < 1 || max_attempts > 100)
      throw std::invalid_argument("runtime recovery max_attempts must be in [1,100]");
  }
  static bool eligible(const std::string & fault)
  {
    return fault == "prediction_timeout" || fault == "imu_time_gap" ||
      fault == "missing_imu_coverage" || fault == "scan_time_gap";
  }
  void command(double now, std::int64_t stamp, double linear, double angular, bool single)
  {
    const bool valid = single && stamp > 0 && stamp > command_stamp_ &&
      std::isfinite(linear) && std::isfinite(angular) &&
      linear <= linear_limit && angular <= angular_limit;
    if (!command_ok_ || !valid || now - command_received_ > command_age) command_since_ = now;
    command_ok_ = valid; command_received_ = now;
    command_stamp_ = stamp;
  }
  void imu(double now, std::int64_t stamp, const std::array<double, 3> & a, double gyro)
  {
    double squared = 0.0;
    for (auto v : a) squared += v * v;
    const bool valid = stamp > 0 && stamp > imu_stamp_ && std::isfinite(squared) &&
      std::isfinite(gyro) && gyro <= gyro_limit &&
      std::abs(std::sqrt(squared) - 9.81) <= gravity_tolerance;
    if (!valid || now - imu_received_ > imu_age ||
      (imu_stamp_ > 0 && (stamp - imu_stamp_) * 1e-9 > imu_age)) samples_.clear();
    imu_ok_ = valid; imu_received_ = now; imu_stamp_ = stamp;
    if (valid) samples_.push_back({now, a});
    while (samples_.size() > 1 && now - samples_[1].time >= settle + 1e-9) samples_.pop_front();
    // Bound memory even if the sensor publishes excessive duplicate arrival times.
    if (samples_.size() > 20000) {samples_.clear(); imu_ok_ = false;}
  }
  bool inputs_valid(double now)
  {
    if (!command_ok_ || now - command_received_ > command_age) {
      blocker = "command_unavailable_or_nonzero"; return false;
    }
    if (!imu_ok_ || now - imu_received_ > imu_age || samples_.empty()) {
      blocker = "imu_unavailable_or_moving"; return false;
    }
    std::array<double, 3> mean{};
    for (const auto & s : samples_) for (int i = 0; i < 3; ++i) mean[i] += s.a[i];
    for (auto & v : mean) v /= samples_.size();
    double variance = 0.0;
    for (const auto & s : samples_) for (int i = 0; i < 3; ++i)
      variance += (s.a[i] - mean[i]) * (s.a[i] - mean[i]);
    if (std::sqrt(variance / samples_.size()) > acceleration_rms) {
      samples_.clear(); blocker = "imu_variation"; return false;
    }
    blocker.clear(); return true;
  }
  bool quiet(double now)
  {
    if (!inputs_valid(now)) return false;
    if (now - command_since_ + 1e-9 < settle || now - samples_.front().time + 1e-9 < settle) {
      blocker = "settling"; return false;
    }
    return true;
  }
  bool lost(const std::string & fault, double age, double now)
  {
    stable_since_ = -1.0;
    if (!enabled || !ever_ready || active || blocked) return false;
    reason = fault;
    if (!eligible(fault)) {faulted("ineligible_fault"); return false;}
    active = true; phase = "waiting_for_stop"; anchor_age = age;
    loss_time_ = now; due_ = now; in_attempt_ = false; episode_started_ = false;
    return true;
  }
  bool take_attempt(double now)
  {
    if (!active || blocked) return false;
    if (episode_started_ && !quiet(now)) {faulted(blocker); return false;}
    if (in_attempt_) {
      if (now - attempt_started_ < attempt_timeout) return false;
      in_attempt_ = false; due_ = now + retry_backoff;
      phase = "waiting_for_stop";
    }
    if (!episode_started_ && anchor_age + now - loss_time_ > anchor_age_limit) {
      faulted("anchor_too_old"); return false;
    }
    if (attempts >= max_attempts) {faulted("attempts_exhausted"); return false;}
    if (now < due_ || !quiet(now)) return false;
    ++attempts; episode_started_ = true; in_attempt_ = true; attempt_started_ = now; phase = "resetting";
    return true;
  }
  void attempt_lost(const std::string & fault, double now)
  {
    if (!active || !in_attempt_) return;
    if (!eligible(fault) && fault != "initialization_timeout") {faulted(fault); return;}
    reason = fault; in_attempt_ = false; due_ = now + retry_backoff;
    phase = "waiting_for_stop";
  }
  void faulted(const std::string & why)
  {
    active = false; blocked = true; in_attempt_ = false;
    phase = "operator_required"; blocker = why; stable_since_ = -1.0;
  }
  void ready(double now)
  {
    if (blocked) return;
    ever_ready = true;
    if (active && !in_attempt_) return;
    if (active) {active = false; in_attempt_ = false; phase = "recovered";}
    if (stable_since_ < 0.0) stable_since_ = now;
    if (now - stable_since_ >= budget_rearm) attempts = 0;
    if (phase != "recovered") phase = enabled ? "idle" : "disabled";
  }
  void unavailable() {stable_since_ = -1.0;}
  void operator_reset()
  {
    active = false; blocked = false; in_attempt_ = false; episode_started_ = false; attempts = 0;
    reason.clear(); blocker.clear(); stable_since_ = -1.0;
    phase = enabled ? "idle" : "disabled";
  }
  bool started() const {return episode_started_;}
  bool in_attempt() const {return in_attempt_;}
  double command_elapsed(double now) const {return now - command_received_;}
  double imu_elapsed(double now) const {return now - imu_received_;}
private:
  struct Sample {double time; std::array<double, 3> a;};
  std::deque<Sample> samples_;
  bool command_ok_ = false, imu_ok_ = false, in_attempt_ = false, episode_started_ = false;
  std::int64_t command_stamp_ = 0, imu_stamp_ = 0;
  double command_received_ = -1e30, imu_received_ = -1e30, command_since_ = 0.0;
  double loss_time_ = 0.0, due_ = 0.0, attempt_started_ = 0.0, stable_since_ = -1.0;
};
}  // namespace open3d_loc
