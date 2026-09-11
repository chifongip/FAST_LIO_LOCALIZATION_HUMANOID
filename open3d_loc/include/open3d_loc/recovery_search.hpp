#pragma once

#include <Eigen/Core>
#include <open3d/geometry/PointCloud.h>

#include <chrono>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

namespace open3d_loc
{

struct RecoverySearchConfig
{
  double xy_radius = 6.0;
  double xy_step = 2.0;
  double z_radius = 1.0;
  double z_step = 1.0;
  double yaw_step_degrees = 30.0;
  double coarse_voxel = 0.8;
  double middle_voxel = 0.4;
  double fine_voxel = 0.2;
  double coarse_distance = 2.4;
  double middle_distance = 1.2;
  double fine_distance = 0.4;
  int coarse_iterations = 20;
  int middle_iterations = 30;
  int fine_iterations = 40;
  // Huber delta is half each point-to-plane stage's correspondence distance.
  std::size_t refine_count = 16;
  std::size_t candidate_count = 4;
  double dedup_translation = 0.3;
  double dedup_angle_degrees = 5.0;
  double min_fitness = 0.5;
  double max_rmse = 0.4;
  std::size_t min_correspondences = 30;
  double min_information_ratio = 1e-4;
  double covariance_scale = 4.0;
  std::array<bool, 6> update_mask{{true, true, true, true, true, true}};
};

struct RecoveryCandidate
{
  Eigen::Matrix4d map_odom = Eigen::Matrix4d::Identity();
  double score = 0.0;  // Mean min(NN squared distance, evaluation radius squared).
  double fitness = 0.0;
  double rmse = 0.0;
  std::size_t correspondences = 0;
  double information_ratio = 0.0;
  // Robot-centered MAP-axis tangent [x,y,z,roll,pitch,yaw], centered at
  // translation(map_odom * odom_base). Information uses
  // rotational lever arms divided by RMS source radius; covariance is restored
  // to meters/radians, multiplied by covariance_scale, and is only meaningful
  // when accepted is true. Only active axes enter the eigenratio/inverse;
  // inactive covariance axes retain identity placeholders, NOT measurements.
  Eigen::Matrix<double, 6, 6> normalized_information =
    Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 6> covariance =
    Eigen::Matrix<double, 6, 6>::Zero();
  bool accepted = false;
};

// Synchronous, single-owner search. Inputs are copied; clouds are expressed in
// map and odom respectively, odom_base maps base into odom. Lattice perturbations
// rotate about the snapshot robot position, around initial_map_odom * odom_base.
// No generation, worker, timeout, or ROS ownership is hidden in this library.
class RecoverySearch
{
public:
  using Cloud = open3d::geometry::PointCloud;
  // Prepare once per map/voxel configuration. Separate search/evaluator objects
  // share the immutable pyramid and own their own KD trees.
  struct Map;
  static std::shared_ptr<const Map> prepareMap(
    const Cloud & cloud,
    const RecoverySearchConfig & config);
  RecoverySearch(
    std::shared_ptr<const Map> map, const Cloud & odom_cloud,
    const Eigen::Matrix4d & odom_base, const Eigen::Matrix4d & initial_map_odom,
    const RecoverySearchConfig & config = RecoverySearchConfig{});
  std::shared_ptr<const Map> map() const;
  RecoverySearch(
    const Cloud & map_cloud, const Cloud & odom_cloud,
    const Eigen::Matrix4d & odom_base, const Eigen::Matrix4d & initial_map_odom,
    const RecoverySearchConfig & config = RecoverySearchConfig{});
  ~RecoverySearch();
  RecoverySearch(RecoverySearch &&) noexcept;
  RecoverySearch & operator=(RecoverySearch &&) noexcept;
  RecoverySearch(const RecoverySearch &) = delete;
  RecoverySearch & operator=(const RecoverySearch &) = delete;

  // Returns done(). Zero budget/work does nothing. One work unit is one coarse
  // seed score or one ICP level. Budget may overrun by one indivisible work unit
  // (including final sorting/evaluation). All seeds rank before refinement starts.
  bool tick(std::chrono::milliseconds budget, std::size_t max_work = 1);
  bool step(std::chrono::milliseconds budget)
  {
    return tick(budget, std::numeric_limits<std::size_t>::max());
  }
  bool done() const;
  std::size_t rankedSeedCount() const;
  std::size_t totalSeedCount() const;
  const std::vector<RecoveryCandidate> & candidates() const;

  // Every transform compared in a confirmation must use the same fresh cloud.
  // Both paths evaluate on fine-downsampled source with radius fine_voxel * 4.
  RecoveryCandidate evaluateFresh(
    const Cloud & odom_cloud, const Eigen::Matrix4d & odom_base,
    const Eigen::Matrix4d & map_odom) const;
  RecoveryCandidate refineFresh(
    const Cloud & odom_cloud, const Eigen::Matrix4d & odom_base,
    const Eigen::Matrix4d & seed_map_odom) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace open3d_loc
