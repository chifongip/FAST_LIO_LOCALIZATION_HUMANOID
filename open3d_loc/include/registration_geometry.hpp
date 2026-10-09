#pragma once

#include <fast_lio/tracking_guard.hpp>
#include <open3d/geometry/PointCloud.h>
#include <open3d/pipelines/registration/Registration.h>

namespace open3d_loc
{
inline double pointToPlaneInformationRatio(
  const open3d::geometry::PointCloud & source,
  const open3d::geometry::PointCloud & target,
  const open3d::pipelines::registration::RegistrationResult & result,
  const Eigen::Matrix4d & transform)
{
  if (!target.HasNormals() || result.correspondence_set_.size() < 6 ||
    !transform.allFinite()) return 0.0;
  // Evaluate rotations about the matched cloud center. Using the map or odom
  // origin makes the conditioning depend on where that origin was placed.
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (const auto & pair : result.correspondence_set_) {
    if (pair(0) < 0 || pair(1) < 0 ||
      static_cast<std::size_t>(pair(0)) >= source.points_.size() ||
      static_cast<std::size_t>(pair(1)) >= target.points_.size()) return 0.0;
    center += source.points_[pair(0)];
  }
  center /= result.correspondence_set_.size();
  Eigen::MatrixXd jacobian(result.correspondence_set_.size(), 6);
  double squared_radius = 0.0;
  for (std::size_t i = 0; i < result.correspondence_set_.size(); ++i) {
    const auto & pair = result.correspondence_set_[i];
    const Eigen::Vector3d relative =
      transform.block<3, 3>(0, 0) * (source.points_[pair(0)] - center);
    const Eigen::Vector3d & normal = target.normals_[pair(1)];
    jacobian.block<1, 3>(i, 0) = normal.transpose();
    jacobian.block<1, 3>(i, 3) = relative.cross(normal).transpose();
    squared_radius += relative.squaredNorm();
  }
  return fast_lio::informationRatio(jacobian,
    std::sqrt(squared_radius / result.correspondence_set_.size()));
}
}  // namespace open3d_loc
