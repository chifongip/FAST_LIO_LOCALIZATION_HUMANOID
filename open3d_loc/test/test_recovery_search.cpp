// Copyright 2026 Open3D Loc Contributors
#include <gtest/gtest.h>
#include "open3d_loc/recovery_search.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
using Search = open3d_loc::RecoverySearch;
using Config = open3d_loc::RecoverySearchConfig;
Search::Cloud room()
{
  Search::Cloud cloud;
  for (int i = -12; i <= 12; ++i) {
    for (int j = -10; j <= 10; ++j) {
      const double u = i * 0.23, v = j * 0.23;
      cloud.points_.emplace_back(u, v, -1.7);
      cloud.points_.emplace_back(3.2, u, v);
      cloud.points_.emplace_back(u, -3.5, v + 0.4);
    }
  }
  return cloud;
}
Config smallConfig()
{
  Config c;
  c.xy_radius = 0;
  c.z_radius = 0;
  c.yaw_step_degrees = 360;
  c.refine_count = 1;
  return c;
}
Eigen::Matrix4d pose(const Eigen::Vector3d & p, double yaw = 0)
{
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t.topLeftCorner<3, 3>() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  t.topRightCorner<3, 1>() = p;
  return t;
}
}
TEST(RecoverySearch, RejectsInvalidSearchConfiguration)
{
  open3d_loc::RecoverySearchConfig config; config.xy_step = 0;
  open3d::geometry::PointCloud cloud; cloud.points_.emplace_back(0, 0, 0);
  EXPECT_THROW(open3d_loc::RecoverySearch::prepareMap(cloud, config), std::invalid_argument);
}
TEST(RecoverySearch, BoundedWorkAndFullYawSeeds)
{
  open3d::geometry::PointCloud cloud;
  for (int x = 0; x < 10; ++x) {
    for (int y = 0; y < 10; ++y) {
      cloud.points_.emplace_back(x * 0.2, y * 0.2, 0);
    }
  }
  open3d_loc::RecoverySearchConfig config; config.xy_radius = 0; config.z_radius = 0;
  open3d_loc::RecoverySearch search(cloud, cloud, Eigen::Matrix4d::Identity(),
    Eigen::Matrix4d::Identity(), config);
  EXPECT_EQ(search.totalSeedCount(), 12u);
  search.tick(std::chrono::milliseconds(100), 1);
  EXPECT_EQ(search.rankedSeedCount(), 1u);
  EXPECT_FALSE(search.done());
}

TEST(RecoverySearch, RanksEntireDiskBeforeRefinement)
{
  const auto cloud = room();
  Search search(cloud, cloud, Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity());
  EXPECT_EQ(search.totalSeedCount(), 29u * 3u * 12u);
  EXPECT_FALSE(search.tick(std::chrono::milliseconds(0)));
  EXPECT_FALSE(search.tick(std::chrono::seconds(1), 0));
  EXPECT_EQ(search.rankedSeedCount(), 0u);
  for (std::size_t i = 0; i < search.totalSeedCount(); ++i) {
    search.tick(std::chrono::seconds(1));
    ASSERT_EQ(search.rankedSeedCount(), i + 1);
    ASSERT_TRUE(search.candidates().empty());
  }
  EXPECT_FALSE(search.done());
  search.tick(std::chrono::seconds(1), 2);
  EXPECT_TRUE(search.candidates().empty());
  search.tick(std::chrono::seconds(1));
  ASSERT_FALSE(search.candidates().empty());
  EXPECT_TRUE(search.candidates().front().accepted);
}

TEST(RecoverySearch, RecoversAroundDistantRobotAndDeduplicates)
{
  auto source = room();
  const auto base = pose({80, -40, 8});
  source.Transform(base);
  const auto expected_base = pose({84, -38, 9}, 1.5707963267948966);
  const Eigen::Matrix4d expected = expected_base * base.inverse();
  auto target = source;
  target.Transform(expected);
  Search search(target, source, base, Eigen::Matrix4d::Identity());
  for (int i = 0; i < 1200 && !search.done(); ++i) {
    search.tick(std::chrono::seconds(1));
  }
  ASSERT_TRUE(search.done());
  ASSERT_FALSE(search.candidates().empty());
  EXPECT_LE(search.candidates().size(), 4u);
  EXPECT_TRUE((search.candidates().front().map_odom * base).isApprox(expected_base, 1e-4));
  for (std::size_t i = 1; i < search.candidates().size(); ++i) {
    EXPECT_LE(search.candidates()[i - 1].score, search.candidates()[i].score);
    for (std::size_t j = 0; j < i; ++j) {
      const Eigen::Matrix4d a = search.candidates()[i].map_odom * base;
      const Eigen::Matrix4d b = search.candidates()[j].map_odom * base;
      const double distance = (a.topRightCorner<3, 1>() - b.topRightCorner<3, 1>()).norm();
      const double angle = Eigen::AngleAxisd(
        a.topLeftCorner<3, 3>() * b.topLeftCorner<3, 3>().transpose()).angle();
      EXPECT_TRUE(distance > 0.3 || angle > 5.0 * 3.141592653589793 / 180.0);
    }
  }
}

TEST(RecoverySearch, CappedScoreUsesWholeSourceAndCommonRadius)
{
  const auto target = room();
  auto source = target;
  for (int i = 0; i < 100; ++i) {
    source.points_.emplace_back(30.0 + i, 20, 10);
  }
  const auto c = smallConfig();
  Search search(target, source, Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity(), c);
  const auto result = search.evaluateFresh(
    source,
    Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity());
  const auto s = source.VoxelDownSample(c.fine_voxel);
  const auto t = target.VoxelDownSample(c.fine_voxel);
  double score = 0;
  std::size_t inliers = 0;
  for (const auto & p : s->points_) {
    double nearest = std::numeric_limits<double>::infinity();
    for (const auto & q : t->points_) {
      nearest = std::min(nearest, (p - q).squaredNorm());
    }
    score += std::min(0.64, nearest);
    inliers += nearest <= 0.64;
  }
  EXPECT_NEAR(result.score, score / s->points_.size(), 1e-12);
  EXPECT_EQ(result.correspondences, inliers);
  EXPECT_NEAR(result.fitness, static_cast<double>(inliers) / s->points_.size(), 1e-12);
  const auto far = search.evaluateFresh(source, Eigen::Matrix4d::Identity(), pose({1000, 0, 0}));
  EXPECT_NEAR(far.score, 0.64, 1e-12);
  EXPECT_EQ(far.correspondences, 0u);
  EXPECT_FALSE(far.accepted);
}

TEST(RecoverySearch, PlaneRejectedExceptForObservableAxes)
{
  Search::Cloud cloud;
  for (int i = -15; i <= 15; ++i) {
    for (int j = -15; j <= 15; ++j) {
      cloud.points_.emplace_back(0.25 * i, 0.25 * j, 0);
    }
  }
  auto c = smallConfig();
  auto map = Search::prepareMap(cloud, c);
  Search full(map, cloud, Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity(), c);
  const auto rejected = full.evaluateFresh(
    cloud,
    Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity());
  EXPECT_FALSE(rejected.accepted);
  EXPECT_LT(rejected.information_ratio, 1e-4);
  c.update_mask = {{false, false, true, true, true, false}};
  Search masked(map, cloud, Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity(), c);
  const auto accepted = masked.evaluateFresh(
    cloud,
    Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity());
  EXPECT_TRUE(accepted.accepted);
  EXPECT_GE(accepted.information_ratio, 1e-4);
  EXPECT_GT(accepted.covariance(2, 2), 0);
  EXPECT_TRUE(accepted.covariance.allFinite());
}

TEST(RecoverySearch, CovarianceScalingAndRobotCenteredMapAxes)
{
  const auto cloud = room();
  auto c = smallConfig();
  auto map = Search::prepareMap(cloud, c);
  Search first(map, cloud, Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity(), c);
  const auto a = first.evaluateFresh(cloud, Eigen::Matrix4d::Identity(), pose({0.02, 0.03, 0.01}));
  ASSERT_TRUE(a.accepted);
  c.covariance_scale = 1;
  Search second(map, cloud, Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity(), c);
  const auto b = second.evaluateFresh(cloud, Eigen::Matrix4d::Identity(), pose({0.02, 0.03, 0.01}));
  EXPECT_TRUE(a.covariance.isApprox(4.0 * b.covariance, 1e-9));
  const auto rotated_base = first.evaluateFresh(
    cloud, pose({0, 0, 0}, 0.7), pose(
      {0.02, 0.03,
        0.01}));
  EXPECT_TRUE(rotated_base.covariance.isApprox(a.covariance, 1e-9));
  auto shifted = cloud;
  const auto offset = pose({800, -400, 80});
  shifted.Transform(offset);
  const auto origin = first.evaluateFresh(
    cloud,
    Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity());
  const auto distant = first.evaluateFresh(shifted, offset, offset.inverse());
  EXPECT_NEAR(origin.information_ratio, distant.information_ratio, 1e-10);
  EXPECT_TRUE(origin.normalized_information.isApprox(distant.normalized_information, 1e-9));
  EXPECT_TRUE(origin.covariance.isApprox(distant.covariance, 1e-8));
}

TEST(RecoverySearch, SharedMapFreshRefinementAndEvaluation)
{
  auto cloud = room();
  const auto original = cloud;
  const auto c = smallConfig();
  auto map = Search::prepareMap(cloud, c);
  Search search(map, cloud, Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity(), c);
  Search evaluator(map, cloud, Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity(), c);
  EXPECT_EQ(search.map().get(), map.get());
  EXPECT_EQ(evaluator.map().get(), map.get());
  cloud.points_.clear();
  const auto seed = pose({0.15, -0.12, 0.08}, 0.03);
  const auto before = evaluator.evaluateFresh(original, Eigen::Matrix4d::Identity(), seed);
  const auto after = evaluator.refineFresh(original, Eigen::Matrix4d::Identity(), seed);
  EXPECT_TRUE(before.map_odom.isApprox(seed));
  EXPECT_TRUE(after.accepted);
  EXPECT_TRUE(after.map_odom.isApprox(Eigen::Matrix4d::Identity(), 1e-4));
  EXPECT_LT(after.score, before.score);
  for (int i = 0; i < 10 && !search.done(); ++i) {
    search.step(std::chrono::milliseconds(500));
  }
  EXPECT_TRUE(search.done());
  EXPECT_FALSE(search.candidates().empty());
}

TEST(RecoverySearch, FitnessRmseAndCorrespondenceGates)
{
  const auto cloud = room();
  auto c = smallConfig();
  c.min_correspondences = 100000;
  auto map = Search::prepareMap(cloud, c);
  Search few(map, cloud, Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity(), c);
  EXPECT_FALSE(
    few.evaluateFresh(
      cloud, Eigen::Matrix4d::Identity(),
      Eigen::Matrix4d::Identity()).accepted);
  c.min_correspondences = 30;
  c.max_rmse = 0.001;
  Search error(map, cloud, Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity(), c);
  EXPECT_FALSE(
    error.evaluateFresh(
      cloud, Eigen::Matrix4d::Identity(), pose(
        {0.04, 0.03,
          0.02})).accepted);
  c.max_rmse = 0.4;
  c.min_fitness = 0.9999;
  auto outlier = cloud;
  outlier.points_.emplace_back(100, 100, 100);
  Search partial(map, cloud, Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity(), c);
  EXPECT_FALSE(
    partial.evaluateFresh(
      outlier, Eigen::Matrix4d::Identity(),
      Eigen::Matrix4d::Identity()).accepted);
}

TEST(RecoverySearch, RejectsNonfiniteThresholdsAndEmptyMask)
{
  const auto cloud = room();
  for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(), -1.0})
  {
    auto c = smallConfig();
    c.min_fitness = invalid;
    EXPECT_THROW(Search::prepareMap(cloud, c), std::invalid_argument);
    c = smallConfig();
    c.dedup_translation = invalid;
    EXPECT_THROW(Search::prepareMap(cloud, c), std::invalid_argument);
    c = smallConfig();
    c.dedup_angle_degrees = invalid;
    EXPECT_THROW(Search::prepareMap(cloud, c), std::invalid_argument);
  }
  auto c = smallConfig();
  c.update_mask.fill(false);
  EXPECT_THROW(Search::prepareMap(cloud, c), std::invalid_argument);
  c = smallConfig();
  c.dedup_translation = 0;
  EXPECT_THROW(Search::prepareMap(cloud, c), std::invalid_argument);
  c = smallConfig();
  c.candidate_count = 5;
  EXPECT_THROW(Search::prepareMap(cloud, c), std::invalid_argument);
}

TEST(RecoverySearch, RejectsCloudAndPoseNaNsBeforeOpen3DCalls)
{
  const auto cloud = room();
  const auto c = smallConfig();
  const Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
  auto map = Search::prepareMap(cloud, c);
  Search search(map, cloud, identity, identity, c);
  for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity()})
  {
    auto bad_cloud = cloud;
    bad_cloud.points_[0].x() = invalid;
    EXPECT_THROW(Search::prepareMap(bad_cloud, c), std::invalid_argument);
    EXPECT_THROW(Search(map, bad_cloud, identity, identity, c), std::invalid_argument);
    EXPECT_THROW(search.evaluateFresh(bad_cloud, identity, identity), std::invalid_argument);
    EXPECT_THROW(search.refineFresh(bad_cloud, identity, identity), std::invalid_argument);
    auto bad_pose = identity;
    bad_pose(0, 3) = invalid;
    EXPECT_THROW(Search(map, cloud, bad_pose, identity, c), std::invalid_argument);
    EXPECT_THROW(Search(map, cloud, identity, bad_pose, c), std::invalid_argument);
    EXPECT_THROW(search.evaluateFresh(cloud, bad_pose, identity), std::invalid_argument);
    EXPECT_THROW(search.evaluateFresh(cloud, identity, bad_pose), std::invalid_argument);
    EXPECT_THROW(search.refineFresh(cloud, bad_pose, identity), std::invalid_argument);
    EXPECT_THROW(search.refineFresh(cloud, identity, bad_pose), std::invalid_argument);
  }
  auto nonrigid = identity;
  nonrigid(0, 0) = 2;
  EXPECT_THROW(search.evaluateFresh(cloud, identity, nonrigid), std::invalid_argument);
  Search::Cloud empty;
  EXPECT_THROW(Search::prepareMap(empty, c), std::invalid_argument);
  EXPECT_THROW(search.refineFresh(empty, identity, identity), std::invalid_argument);
}
