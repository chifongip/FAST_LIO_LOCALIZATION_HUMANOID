#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace fast_lio
{
struct TrackingLimits
{
  int min_features = 100;
  double min_feature_ratio = 0.2;
  double max_residual = 0.15;
  double min_information_ratio = 1e-4;
  double max_imu_gap = 0.05;
  double prediction_timeout = 0.5;
  double max_speed = 3.0;
  double max_angular_speed = 3.0;

  void validate() const
  {
    const double positive[] = {min_feature_ratio, max_residual, min_information_ratio,
      max_imu_gap, prediction_timeout, max_speed, max_angular_speed};
    for (double value : positive) {
      if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument("tracking limits must be finite and positive");
      }
    }
    if (min_features < 6 || min_feature_ratio > 1.0 || min_information_ratio > 1.0) {
      throw std::invalid_argument("invalid tracking feature/observability limits");
    }
  }
};

inline double informationRatio(const Eigen::MatrixXd & jacobian, double radius)
{
  if (jacobian.rows() < 6 || jacobian.cols() < 6 || !jacobian.allFinite() ||
    !std::isfinite(radius) || radius <= 0.0) return 0.0;
  Eigen::MatrixXd normalized = jacobian.leftCols(6);
  normalized.rightCols(3) /= std::max(radius, 0.1);
  Eigen::Matrix<double, 6, 6> information = normalized.transpose() * normalized;
  if (!information.allFinite()) return 0.0;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(information);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite() ||
    solver.eigenvalues().maxCoeff() <= 0.0) return 0.0;
  return std::max(0.0, solver.eigenvalues().minCoeff()) / solver.eigenvalues().maxCoeff();
}

inline std::string measurementRejection(const TrackingLimits & limits, int effective,
  int total, double residual, double information_ratio)
{
  if (effective < limits.min_features || total <= 0 ||
    double(effective) / total < limits.min_feature_ratio) return "insufficient_features";
  if (!std::isfinite(residual) || residual > limits.max_residual) return "residual_gate";
  if (!std::isfinite(information_ratio) || information_ratio < limits.min_information_ratio)
    return "degenerate_geometry";
  return "";
}

inline bool plausibleMotion(const TrackingLimits & limits, const Eigen::Vector3d & previous,
  const Eigen::Matrix3d & previous_rotation, const Eigen::Vector3d & position,
  const Eigen::Matrix3d & rotation, const Eigen::Vector3d & velocity, double dt)
{
  if (!previous.allFinite() || !previous_rotation.allFinite() || !position.allFinite() ||
    !rotation.allFinite() || !velocity.allFinite() || !std::isfinite(dt) || dt <= 0.0)
    return false;
  const double angle = std::acos(std::clamp(
    ((previous_rotation.transpose() * rotation).trace() - 1.0) / 2.0, -1.0, 1.0));
  return velocity.norm() <= limits.max_speed &&
    (position - previous).norm() <= limits.max_speed * dt + 0.1 &&
    angle <= limits.max_angular_speed * dt + 0.1;
}

class TrackingGuard
{
public:
  enum State {INITIALIZING, TRACKING, DEGRADED, LOST};
  TrackingLimits limits;
  State state = INITIALIZING;
  double accepted_stamp = -1.0;
  double prediction_started = -1.0;
  int confirmations = 0;
  std::string reason = "initializing";

  void reset()
  {
    state = INITIALIZING;
    accepted_stamp = -1.0;
    prediction_started = -1.0;
    confirmations = 0;
    reason = "initializing";
  }
  bool allowPrediction(double stamp)
  {
    const double reference = accepted_stamp >= 0.0 ? accepted_stamp : prediction_started;
    if (reference >= 0.0 && stamp - reference >= limits.prediction_timeout) {
      lose(accepted_stamp >= 0.0 ? "prediction_timeout" : "initialization_timeout");
    }
    return state != LOST;
  }
  void lose(const std::string & why)
  {
    if (state == LOST) return;
    state = LOST;
    confirmations = 0;
    reason = why;
  }
  void reject(double stamp, const std::string & why)
  {
    if (state == LOST) return;
    confirmations = 0;
    reason = why;
    if (accepted_stamp >= 0.0) {
      state = DEGRADED;
      if (stamp - accepted_stamp >= limits.prediction_timeout) lose("prediction_timeout");
    }
  }
  bool accept(double stamp)
  {
    if (state == LOST) return false;
    if (state != TRACKING && ++confirmations < 3) {
      reason = "confirming_tracking";
      return false;
    }
    state = TRACKING;
    accepted_stamp = stamp;
    confirmations = 0;
    reason = "accepted";
    return true;
  }
};
}  // namespace fast_lio
