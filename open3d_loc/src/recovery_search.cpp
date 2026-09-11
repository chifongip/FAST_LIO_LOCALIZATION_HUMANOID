// Copyright 2026 Open3D Loc Contributors
#include "open3d_loc/recovery_search.hpp"
#include <open3d/geometry/BoundingVolume.h>
#include <open3d/geometry/KDTreeFlann.h>
#include <open3d/pipelines/registration/Registration.h>
#include <open3d/pipelines/registration/RobustKernel.h>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace open3d_loc
{
namespace reg = open3d::pipelines::registration;
namespace
{
constexpr double pi = 3.14159265358979323846;
using Cloud = open3d::geometry::PointCloud;
using Matrix6 = Eigen::Matrix<double, 6, 6>;
bool distinct(
  const Eigen::Matrix4d & a, const Eigen::Matrix4d & b,
  const Eigen::Matrix4d & odom, const RecoverySearchConfig & config)
{
  const Eigen::Matrix4d ar = a * odom, br = b * odom;
  return (ar.block<3, 1>(0, 3) - br.block<3, 1>(0, 3)).norm() > config.dedup_translation ||
         Eigen::AngleAxisd(ar.block<3, 3>(0, 0) * br.block<3, 3>(0, 0).transpose()).angle() >
         config.dedup_angle_degrees * pi / 180.0;
}
void validate(const RecoverySearchConfig & c)
{
  for (double x : {c.xy_step, c.z_step, c.yaw_step_degrees, c.coarse_voxel, c.middle_voxel,
      c.fine_voxel, c.coarse_distance, c.middle_distance, c.fine_distance, c.covariance_scale,
      c.dedup_translation, c.dedup_angle_degrees})
  {
    if (!std::isfinite(x) || x <= 0.0) {
      throw std::invalid_argument("Invalid recovery search scale");
    }
  }
  if (!std::isfinite(c.xy_radius) || !std::isfinite(c.z_radius) || c.xy_radius < 0 ||
    c.z_radius < 0 ||
    c.xy_radius / c.xy_step > 50 || c.z_radius / c.z_step > 10 || c.yaw_step_degrees < 1.0 ||
    c.refine_count < 1 || c.candidate_count < 1 || c.candidate_count > 4 ||
    c.coarse_iterations < 1 ||
    c.middle_iterations < 1 || c.fine_iterations < 1 || !std::isfinite(c.min_fitness) ||
    c.min_fitness < 0 || c.min_fitness > 1 || c.min_correspondences == 0 ||
    !std::isfinite(c.max_rmse) || c.max_rmse <= 0 || !std::isfinite(c.min_information_ratio) ||
    c.min_information_ratio <= 0 || c.min_information_ratio > 1 ||
    std::none_of(c.update_mask.begin(), c.update_mask.end(), [](bool active) {return active;}))
  {
    throw std::invalid_argument("Invalid recovery search bounds");
  }
}
void validateCloud(const Cloud & cloud)
{
  if (cloud.IsEmpty() || std::any_of(
      cloud.points_.begin(), cloud.points_.end(),
      [](const Eigen::Vector3d & point) {return !point.allFinite();}))
  {
    throw std::invalid_argument("Recovery cloud must be nonempty with finite points");
  }
}
void validatePose(const Eigen::Matrix4d & pose)
{
  const Eigen::Matrix3d rotation = pose.block<3, 3>(0, 0);
  if (!pose.allFinite() || !pose.row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1), 1e-8) ||
    !(rotation * rotation.transpose()).isApprox(Eigen::Matrix3d::Identity(), 1e-6) ||
    std::abs(rotation.determinant() - 1.0) > 1e-6)
  {
    throw std::invalid_argument("Recovery pose must be a finite rigid transform");
  }
}
}
struct RecoverySearch::Map
{
  std::array<std::shared_ptr<Cloud>, 3> levels;
};
std::shared_ptr<const RecoverySearch::Map> RecoverySearch::prepareMap(
  const Cloud & cloud, const RecoverySearchConfig & config)
{
  validate(config);
  validateCloud(cloud);
  auto map = std::make_shared<Map>();
  const double voxels[] = {config.coarse_voxel, config.middle_voxel, config.fine_voxel};
  for (int i = 0; i < 3; ++i) {
    map->levels[i] = cloud.VoxelDownSample(voxels[i]);
    map->levels[i]->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxels[i] * 2.0, 30));
  }
  return map;
}
struct RecoverySearch::Impl
{
  std::shared_ptr<const Map> map;
  RecoverySearchConfig config;
  Eigen::Matrix4d odom;
  std::array<std::shared_ptr<Cloud>, 3> source;
  std::unique_ptr<open3d::geometry::KDTreeFlann> coarse_tree, fine_tree;
  std::vector<std::pair<double, Eigen::Matrix4d>> seeds;
  std::vector<Eigen::Matrix4d> selected;
  std::vector<RecoveryCandidate> results;
  std::size_t ranked = 0, refined = 0;
  int level = 0;
  Eigen::Matrix4d working = Eigen::Matrix4d::Identity();
  bool finished = false;

  double score(
    const Cloud & cloud, const Eigen::Matrix4d & transform,
    const open3d::geometry::KDTreeFlann & tree, double radius) const
  {
    if (cloud.IsEmpty() || !transform.allFinite()) {return std::numeric_limits<double>::infinity();}
    double sum = 0;
    std::vector<int> indices(1); std::vector<double> distances(1);
    for (const auto & point : cloud.points_) {
      const Eigen::Vector3d mapped = transform.block<3, 3>(0, 0) * point + transform.block<3, 1>(
        0,
        3);
      if (!mapped.allFinite()) {return std::numeric_limits<double>::infinity();}
      sum += tree.SearchKNN(mapped, 1, indices, distances) ?
        std::min(distances[0], radius * radius) : radius * radius;
    }
    return sum / cloud.points_.size();
  }
  Eigen::Matrix4d refineLevel(const Cloud & cloud, const Eigen::Matrix4d & seed, int stage) const
  {
    const double distance[] =
    {config.coarse_distance, config.middle_distance, config.fine_distance};
    const int iterations[] =
    {config.coarse_iterations, config.middle_iterations, config.fine_iterations};
    // Crop to source extent transformed under this hypothesis, with a generous coarse margin.
    Cloud mapped = cloud; mapped.Transform(seed);
    open3d::geometry::AxisAlignedBoundingBox box(
      mapped.GetMinBound() - Eigen::Vector3d::Constant(3.0 * config.coarse_distance),
      mapped.GetMaxBound() + Eigen::Vector3d::Constant(3.0 * config.coarse_distance));
    const auto target = map->levels[stage]->Crop(box);
    if (target->IsEmpty() || cloud.IsEmpty()) {return seed;}
    const reg::ICPConvergenceCriteria criteria(1e-6, 1e-6, iterations[stage]);
    if (stage == 0) {
      return reg::RegistrationICP(
        cloud, *target, distance[stage], seed,
        reg::TransformationEstimationPointToPoint(), criteria).transformation_;
    }
    return reg::RegistrationICP(
      cloud, *target, distance[stage], seed,
      reg::TransformationEstimationPointToPlane(
        std::make_shared<reg::HuberLoss>(
          distance[stage] /
          2.0)),
      criteria).transformation_;
  }
  RecoveryCandidate evaluate(
    const Cloud & cloud, const Eigen::Matrix4d & base,
    const Eigen::Matrix4d & transform) const
  {
    RecoveryCandidate result; result.map_odom = transform;
    result.score = std::numeric_limits<double>::infinity();
    result.covariance = Matrix6::Identity() * 1e6;
    if (cloud.IsEmpty() || !transform.allFinite()) {return result;}
    const Eigen::Matrix3d rotation = transform.block<3, 3>(0, 0);
    if (!(rotation * rotation.transpose()).isApprox(Eigen::Matrix3d::Identity(), 1e-5) ||
      std::abs(rotation.determinant() - 1.0) > 1e-5) {return result;}
    const double radius = config.fine_voxel * 4.0;
    result.score = score(cloud, transform, *fine_tree, radius);
    const Eigen::Vector3d center = (transform * base).block<3, 1>(0, 3);
    double lever_squared = 0.0, squared_error = 0.0;
    for (const auto & p : cloud.points_) {
      lever_squared +=
        (p - base.block<3, 1>(0, 3)).squaredNorm();
    }
    const double lever = std::max(0.1, std::sqrt(lever_squared / cloud.points_.size()));
    std::vector<int> indices(1); std::vector<double> distances(1);
    for (const auto & p : cloud.points_) {
      const Eigen::Vector3d q = rotation * p + transform.block<3, 1>(0, 3);
      if (!fine_tree->SearchKNN(q, 1, indices, distances) || distances[0] > radius * radius) {
        continue;
      }
      ++result.correspondences; squared_error += distances[0];
      const Eigen::Vector3d normal = map->levels[2]->normals_[indices[0]].normalized();
      if (!normal.allFinite()) {continue;}
      Eigen::Matrix<double, 6, 1> j;
      j.head<3>() = normal; j.tail<3>() = (q - center).cross(normal) / lever;
      const double error = std::abs(normal.dot(q - map->levels[2]->points_[indices[0]]));
      const double weight = std::min(1.0, (config.fine_distance / 2.0) / std::max(error, 1e-12));
      result.normalized_information += weight * j * j.transpose();
    }
    result.fitness = static_cast<double>(result.correspondences) / cloud.points_.size();
    result.rmse = result.correspondences ? std::sqrt(squared_error / result.correspondences) :
      std::numeric_limits<double>::infinity();
    std::vector<int> active;
    for (int i = 0; i < 6; ++i) {if (config.update_mask[i]) {active.push_back(i);}}
    if (active.empty()) {return result;}
    Eigen::MatrixXd information(active.size(), active.size());
    for (std::size_t i = 0; i < active.size(); ++i) {
      for (std::size_t j = 0; j < active.size(); ++j) {
        information(i, j) = result.normalized_information(active[i], active[j]);
      }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(information);
    if (spectrum.info() != Eigen::Success) {return result;}
    result.information_ratio = spectrum.eigenvalues().minCoeff() /
      std::max(1e-12, spectrum.eigenvalues().maxCoeff());
    result.accepted = result.fitness > config.min_fitness && result.rmse <= config.max_rmse &&
      result.correspondences >= config.min_correspondences &&
      result.information_ratio >= config.min_information_ratio;
    if (result.accepted) {
      const Eigen::MatrixXd covariance = spectrum.eigenvectors() *
        spectrum.eigenvalues().cwiseMax(1e-12).cwiseInverse().asDiagonal() *
        spectrum.eigenvectors().transpose() *
        std::max(1e-6, result.rmse * result.rmse) * config.covariance_scale;
      result.covariance.setIdentity();
      for (std::size_t i = 0; i < active.size(); ++i) {
        for (std::size_t j = 0; j < active.size(); ++j) {
          result.covariance(active[i], active[j]) = covariance(i, j) /
            ((active[i] >= 3 ? lever : 1.0) * (active[j] >= 3 ? lever : 1.0));
        }
      }
    }
    return result;
  }
};
RecoverySearch::RecoverySearch(
  const Cloud & map_cloud, const Cloud & cloud,
  const Eigen::Matrix4d & base, const Eigen::Matrix4d & initial,
  const RecoverySearchConfig & config)
: RecoverySearch(prepareMap(map_cloud, config), cloud, base, initial, config) {}
RecoverySearch::RecoverySearch(
  std::shared_ptr<const Map> map, const Cloud & cloud,
  const Eigen::Matrix4d & base, const Eigen::Matrix4d & initial,
  const RecoverySearchConfig & config)
: impl_(std::make_unique<Impl>())
{
  validate(config);
  validateCloud(cloud);
  validatePose(base);
  validatePose(initial);
  if (!map || cloud.IsEmpty()) {throw std::invalid_argument("Empty recovery source or map");}
  impl_->map = std::move(map); impl_->config = config; impl_->odom = base;
  const double voxels[] = {config.coarse_voxel, config.middle_voxel, config.fine_voxel};
  for (int i = 0; i < 3; ++i) {
    impl_->source[i] = cloud.VoxelDownSample(voxels[i]);
  }
  impl_->coarse_tree = std::make_unique<open3d::geometry::KDTreeFlann>(*impl_->map->levels[0]);
  impl_->fine_tree = std::make_unique<open3d::geometry::KDTreeFlann>(*impl_->map->levels[2]);
  const Eigen::Matrix4d robot = initial * base;
  for (int x = -static_cast<int>(config.xy_radius / config.xy_step);
    x <= config.xy_radius / config.xy_step; ++x)
  {
    for (int y = -static_cast<int>(config.xy_radius / config.xy_step);
      y <= config.xy_radius / config.xy_step; ++y)
    {
      if (std::hypot(x * config.xy_step, y * config.xy_step) > config.xy_radius + 1e-9) {continue;}
      for (int z = -static_cast<int>(config.z_radius / config.z_step);
        z <= config.z_radius / config.z_step; ++z)
      {
        for (double yaw = 0; yaw < 360.0 - 1e-9; yaw += config.yaw_step_degrees) {
          Eigen::Matrix4d seed = robot;
          seed.block<3, 1>(0, 3) += Eigen::Vector3d(
            x * config.xy_step, y * config.xy_step,
            z * config.z_step);
          seed.block<3, 3>(
            0,
            0) =
            Eigen::AngleAxisd(yaw * pi / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
            robot.block<3, 3>(0, 0);
          impl_->seeds.emplace_back(0.0, seed * base.inverse());
        }
      }
    }
  }
}
RecoverySearch::~RecoverySearch() = default;
RecoverySearch::RecoverySearch(RecoverySearch &&) noexcept = default;
RecoverySearch & RecoverySearch::operator=(RecoverySearch &&) noexcept = default;
std::shared_ptr<const RecoverySearch::Map> RecoverySearch::map() const {return impl_->map;}
bool RecoverySearch::done() const {return impl_->finished;}
std::size_t RecoverySearch::rankedSeedCount() const {return impl_->ranked;}
std::size_t RecoverySearch::totalSeedCount() const {return impl_->seeds.size();}
const std::vector<RecoveryCandidate> & RecoverySearch::candidates() const {return impl_->results;}
bool RecoverySearch::tick(std::chrono::milliseconds budget, std::size_t max_work)
{
  const auto deadline = std::chrono::steady_clock::now() + budget;
  for (std::size_t work = 0;
    work < max_work && !done() && std::chrono::steady_clock::now() < deadline; ++work)
  {
    if (impl_->ranked < impl_->seeds.size()) {
      auto & seed = impl_->seeds[impl_->ranked++];
      seed.first = impl_->score(
        *impl_->source[0], seed.second, *impl_->coarse_tree,
        impl_->config.coarse_distance);
      continue;
    }
    if (impl_->selected.empty()) {
      std::stable_sort(
        impl_->seeds.begin(), impl_->seeds.end(), [](const auto & a,
        const auto & b) {
          return a.first < b.first;
        });
      for (const auto & seed : impl_->seeds) {
        bool unique = true;
        for (const auto & selected : impl_->selected) {
          unique = unique && distinct(selected, seed.second, impl_->odom, impl_->config);
        }
        if (unique) {impl_->selected.push_back(seed.second);}
        if (impl_->selected.size() >= impl_->config.refine_count) {break;}
      }
      if (impl_->selected.empty()) {impl_->finished = true; break;}
    }
    if (impl_->level == 0) {impl_->working = impl_->selected[impl_->refined];}
    impl_->working = impl_->refineLevel(*impl_->source[impl_->level], impl_->working, impl_->level);
    if (++impl_->level == 3) {
      auto candidate = impl_->evaluate(*impl_->source[2], impl_->odom, impl_->working);
      if (candidate.accepted) {
        bool unique = true;
        for (auto & previous : impl_->results) {
          if (!distinct(previous.map_odom, candidate.map_odom, impl_->odom, impl_->config)) {
            if (candidate.score < previous.score) {previous = candidate;}
            unique = false; break;
          }
        }
        if (unique) {impl_->results.push_back(candidate);}
        std::stable_sort(
          impl_->results.begin(), impl_->results.end(),
          [](const auto & a, const auto & b) {return a.score < b.score;});
        if (impl_->results.size() > impl_->config.candidate_count) {
          impl_->results.resize(impl_->config.candidate_count);
        }
      }
      impl_->level = 0;
      if (++impl_->refined >= impl_->selected.size()) {impl_->finished = true;}
    }
  }
  return done();
}
RecoveryCandidate RecoverySearch::evaluateFresh(
  const Cloud & cloud, const Eigen::Matrix4d & base,
  const Eigen::Matrix4d & transform) const
{
  validateCloud(cloud);
  validatePose(base);
  validatePose(transform);
  return impl_->evaluate(*cloud.VoxelDownSample(impl_->config.fine_voxel), base, transform);
}
RecoveryCandidate RecoverySearch::refineFresh(
  const Cloud & cloud, const Eigen::Matrix4d & base,
  const Eigen::Matrix4d & seed) const
{
  validateCloud(cloud);
  validatePose(base);
  validatePose(seed);
  Eigen::Matrix4d transform = seed;
  const double voxels[] =
  {impl_->config.coarse_voxel, impl_->config.middle_voxel, impl_->config.fine_voxel};
  for (int i = 0; i < 3; ++i) {
    transform = impl_->refineLevel(*cloud.VoxelDownSample(voxels[i]), transform, i);
  }
  return evaluateFresh(cloud, base, transform);
}
}  // namespace open3d_loc
