#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace fast_lio
{
enum class ImuSyncAction {READY, WAITING, SKIP_SCAN, INVALID};
struct ImuCoverage
{
  ImuSyncAction action = ImuSyncAction::WAITING;
  std::string reason = "waiting_for_imu_coverage";
  std::size_t count = 0;
  double first = -1.0, last = -1.0, following = -1.0, end_gap = 0.0, gap = 0.0;
};

inline ImuCoverage imuCoverage(const std::vector<double> & stamps, double scan_begin,
  double scan_end, double previous_sample, double max_gap, double waited,
  double wait_timeout, bool warmup)
{
  ImuCoverage result;
  auto invalid = [&result](const std::string & reason) {
    result.action = ImuSyncAction::INVALID;
    result.reason = reason;
    return result;
  };
  if (!std::isfinite(scan_begin) || !std::isfinite(scan_end) || scan_end < scan_begin)
    return invalid("invalid_scan_time");
  result.last = previous_sample;
  result.end_gap = previous_sample >= 0.0 ? scan_end - previous_sample : 0.0;
  if (stamps.empty()) {
    if (waited >= wait_timeout) return invalid("imu_arrival_timeout");
    return result;
  }
  for (std::size_t i = 0; i < stamps.size(); ++i) {
    if (!std::isfinite(stamps[i]) || (i && stamps[i] <= stamps[i - 1]))
      return invalid("timestamp_regression");
  }
  const auto boundary = std::upper_bound(stamps.begin(), stamps.end(), scan_end);
  result.count = std::distance(stamps.begin(), boundary);
  result.first = result.count ? stamps.front() : -1.0;
  result.last = result.count ? *(boundary - 1) : previous_sample;
  result.following = boundary == stamps.end() ? -1.0 : *boundary;
  result.end_gap = result.last >= 0.0 ? scan_end - result.last : 0.0;
  if (stamps.back() < scan_end) {
    if (waited >= wait_timeout) return invalid("imu_arrival_timeout");
    return result;
  }
  if (warmup) {
    // Only the most recent continuous 200 ms are needed to initialize.
    auto first = boundary;
    if (first != stamps.begin()) --first;
    while (first != stamps.begin() && *first - *(first - 1) <= max_gap) --first;
    if (result.last < 0.0 || result.last - *first < 0.2 || *first > scan_begin) {
      result.action = ImuSyncAction::SKIP_SCAN;
      result.reason = "startup_imu_window";
      return result;
    }
    result.first = *first;
    // The caller trims older discontinuous samples before extracting the batch.
  }
  double previous = warmup ? -1.0 : previous_sample;
  for (auto it = stamps.begin(); it != stamps.end(); ++it) {
    if (warmup && *it < result.first) continue;
    if (!std::isfinite(*it) || (previous >= 0.0 &&
      *it <= previous))
      return invalid("timestamp_regression");
    if (previous >= 0.0) {
      result.gap = std::max(result.gap, *it - previous);
      if (*it - previous > max_gap) return invalid("imu_time_gap");
    }
    previous = *it;
    if (*it >= scan_end) break;
  }
  if (result.last < 0.0 || result.end_gap > max_gap)
    return invalid("imu_scan_end_gap");
  if (!result.count) {
    result.action = ImuSyncAction::SKIP_SCAN;
    result.reason = "covered_empty_imu_interval";
    return result;
  }
  result.action = ImuSyncAction::READY;
  result.reason = "covered";
  return result;
}
}  // namespace fast_lio
