#ifndef OPEN3D_LOC_TRANSFORM_UTILS_HPP
#define OPEN3D_LOC_TRANSFORM_UTILS_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>

namespace open3d_loc
{
inline Eigen::Matrix4d deriveBodyPose(
    const Eigen::Matrix4d &tracking_pose, const Eigen::Matrix4d &body_to_tracking)
{
  return tracking_pose * body_to_tracking.inverse();
}

inline Eigen::Matrix4d deriveOutputPose(
    const Eigen::Matrix4d &body_pose, const Eigen::Matrix4d &body_to_output)
{
  return body_pose * body_to_output;
}

inline Eigen::Matrix3d levelOrientationFromYaw(const double yaw)
{
  return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}
}  // namespace open3d_loc

#endif
