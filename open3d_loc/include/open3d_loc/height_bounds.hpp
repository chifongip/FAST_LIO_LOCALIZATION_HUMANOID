// Copyright 2026 Open3D Loc Contributors
// SPDX-License-Identifier: BSD-3-Clause
#ifndef OPEN3D_LOC__HEIGHT_BOUNDS_HPP_
#define OPEN3D_LOC__HEIGHT_BOUNDS_HPP_

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace open3d_loc
{
struct HeightBounds
{
  bool enabled = false;
  double floor_z = 0.0;
  double min_height = 0.3;
  double max_height = 0.7;

  void validate(bool fusion_enabled) const
  {
    if (!std::isfinite(floor_z) || !std::isfinite(min_height) ||
      !std::isfinite(max_height) || min_height > max_height)
    {
      throw std::invalid_argument("height_bounds requires finite, ordered limits and floor_z");
    }
    if (enabled && !fusion_enabled) {
      throw std::invalid_argument(
        "height_bounds requires fusion.enabled");
    }
  }

  // Project the composed map-frame body height, including map/odom rotation.
  double adjustment(
    const Eigen::Matrix4d & map_to_odom, const Eigen::Matrix4d & odom_to_body) const
  {
    if (!enabled) {
      return 0.0;
    }
    const double height = (map_to_odom * odom_to_body)(2, 3) - floor_z;
    if (!std::isfinite(height)) {
      throw std::invalid_argument("Cannot bound a nonfinite body height");
    }
    return std::clamp(height, min_height, max_height) - height;
  }
};
}  // namespace open3d_loc
#endif  // OPEN3D_LOC__HEIGHT_BOUNDS_HPP_
