#ifndef FAST_LIO_GRAVITY_ALIGNMENT_H
#define FAST_LIO_GRAVITY_ALIGNMENT_H

#include <Eigen/Geometry>

namespace fast_lio
{
// Builds R_world_imu so the stationary accelerometer direction becomes +Z in
// odom. Yaw remains intentionally unconstrained by gravity.
inline bool gravityAlignedWorldFromImu(
    const Eigen::Vector3d &specific_force, Eigen::Matrix3d &world_from_imu)
{
  world_from_imu.setIdentity();
  if (!specific_force.allFinite() || specific_force.squaredNorm() < 1e-12)
  {
    return false;
  }

  const Eigen::Quaterniond alignment = Eigen::Quaterniond::FromTwoVectors(
      specific_force.normalized(), Eigen::Vector3d::UnitZ());
  if (!alignment.coeffs().allFinite())
  {
    return false;
  }

  world_from_imu = alignment.normalized().toRotationMatrix();
  return world_from_imu.allFinite() && world_from_imu.determinant() > 0.0;
}
}  // namespace fast_lio

#endif
