#ifndef FAST_LIO_ODOMETRY_UTILS_H
#define FAST_LIO_ODOMETRY_UTILS_H

#include <cmath>

#include <Eigen/Core>

namespace fast_lio
{
constexpr int kOdometryStateDof = 23;
constexpr int kPositionStateIndex = 0;
constexpr int kOrientationStateIndex = 3;
constexpr int kVelocityStateIndex = 12;
constexpr int kGyroBiasStateIndex = 15;

using StateCovariance = Eigen::Matrix<double, kOdometryStateDof, kOdometryStateDof>;
using PoseCovariance = Eigen::Matrix<double, 6, 6>;
using TwistCovariance = Eigen::Matrix<double, 6, 6>;

struct BodyTwist
{
  Eigen::Vector3d linear;
  Eigen::Vector3d angular;
};

inline bool interpolateAngularVelocity(
    const double earlier_time,
    const Eigen::Vector3d &earlier_measurement,
    const double later_time,
    const Eigen::Vector3d &later_measurement,
    const double target_time,
    Eigen::Vector3d &interpolated_measurement)
{
  if (!std::isfinite(earlier_time) || !std::isfinite(later_time) ||
      !std::isfinite(target_time) || !earlier_measurement.allFinite() ||
      !later_measurement.allFinite() || later_time < earlier_time ||
      target_time < earlier_time || target_time > later_time)
    return false;

  if (later_time == earlier_time)
  {
    interpolated_measurement = earlier_measurement;
    return true;
  }

  const double ratio = (target_time - earlier_time) / (later_time - earlier_time);
  interpolated_measurement =
      earlier_measurement + ratio * (later_measurement - earlier_measurement);
  return true;
}

inline BodyTwist bodyTwistFromState(
    const Eigen::Matrix3d &world_from_body,
    const Eigen::Vector3d &linear_velocity_world,
    const Eigen::Vector3d &gyro_measurement_body,
    const Eigen::Vector3d &gyro_bias_body)
{
  BodyTwist twist;
  twist.linear = world_from_body.transpose() * linear_velocity_world;
  twist.angular = gyro_measurement_body - gyro_bias_body;
  return twist;
}

template <typename Derived>
inline PoseCovariance poseCovarianceFromState(
    const Eigen::MatrixBase<Derived> &state_covariance)
{
  return state_covariance.template block<6, 6>(
      kPositionStateIndex, kPositionStateIndex);
}

template <typename Derived>
inline TwistCovariance bodyTwistCovarianceFromState(
    const Eigen::MatrixBase<Derived> &state_covariance,
    const Eigen::Matrix3d &world_from_body,
    const Eigen::Matrix3d &gyro_measurement_covariance)
{
  Eigen::Matrix<double, 6, kOdometryStateDof> jacobian =
      Eigen::Matrix<double, 6, kOdometryStateDof>::Zero();
  jacobian.template block<3, 3>(0, kVelocityStateIndex) =
      world_from_body.transpose();
  jacobian.template block<3, 3>(3, kGyroBiasStateIndex) =
      -Eigen::Matrix3d::Identity();

  TwistCovariance covariance = jacobian * state_covariance * jacobian.transpose();
  covariance.template block<3, 3>(3, 3) += gyro_measurement_covariance;
  return 0.5 * (covariance + covariance.transpose());
}
}  // namespace fast_lio

#endif
