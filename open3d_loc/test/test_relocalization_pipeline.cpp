// Copyright 2026 Open3D Loc Contributors
// Exercises the public recovery namespace with real Open3D measurements.
#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <vector>

#include "open3d_loc/global_correction_filter.hpp"
#include "open3d_loc/recovery_coordinator.hpp"
#include "open3d_loc/recovery_search.hpp"

namespace
{
using Cloud = open3d::geometry::PointCloud;
constexpr double kPi = 3.14159265358979323846;

Eigen::Matrix4d pose(double x, double y, double z, double yaw = 0.0)
{
  Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
  result.block<3, 3>(0, 0) =
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  result.block<3, 1>(0, 3) = Eigen::Vector3d(x, y, z);
  return result;
}

// Deliberately unequal walls, an offset alcove, and a tilted panel break room
// translation/yaw symmetries. No random sampling or external assets are used.
Cloud room()
{
  Cloud cloud;
  for (int i = 0; i <= 32; ++i) {
    for (int j = 0; j <= 24; ++j) {
      const double u = i * 0.125;
      const double v = j * 0.125;
      cloud.points_.emplace_back(u - 2.0, v - 1.5, -1.0);
      cloud.points_.emplace_back(-2.0, v - 1.5, u * 0.65 - 1.0);
      cloud.points_.emplace_back(u - 2.0, 1.5, v * 0.8 - 1.0);
    }
  }
  for (int i = 0; i <= 12; ++i) {
    for (int j = 0; j <= 16; ++j) {
      cloud.points_.emplace_back(0.4 + 0.08 * i, -0.7, -0.8 + 0.1 * j);
      cloud.points_.emplace_back(1.4, -1.2 + 0.08 * i, -0.8 + 0.1 * j);
      cloud.points_.emplace_back(-0.7 + 0.08 * i, -0.9 + 0.08 * j, 0.6 + 0.025 * i);
    }
  }
  return cloud;
}

open3d_loc::RecoverySearchConfig search_config()
{
  open3d_loc::RecoverySearchConfig config;
  config.xy_radius = 1.0;
  config.xy_step = 1.0;
  config.z_radius = 0.0;
  config.yaw_step_degrees = 90.0;
  config.coarse_voxel = 0.35;
  config.middle_voxel = 0.2;
  config.fine_voxel = 0.1;
  config.coarse_distance = 1.2;
  config.middle_distance = 0.6;
  config.fine_distance = 0.3;
  config.refine_count = 12;
  config.min_fitness = 0.8;
  config.max_rmse = 0.15;
  return config;
}

void finish(open3d_loc::RecoverySearch & search)
{
  // Deterministic work bound, independent of machine wall-clock speed.
  for (std::size_t i = 0; i < 10000 && !search.done(); ++i) {
    search.tick(std::chrono::milliseconds(100), 1);
  }
  ASSERT_TRUE(search.done());
}

void expect_covariance(const open3d_loc::RecoveryCandidate & candidate)
{
  ASSERT_TRUE(candidate.covariance.allFinite());
  EXPECT_TRUE(candidate.covariance.isApprox(candidate.covariance.transpose(), 1e-9));
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(candidate.covariance);
  ASSERT_EQ(solver.info(), Eigen::Success);
  EXPECT_GT(solver.eigenvalues().minCoeff(), 0.0);
}

open3d_loc::recovery::Candidate observation(
  const open3d_loc::RecoveryCandidate & measured, const Eigen::Matrix4d & odom_base,
  std::uint64_t scan_id, std::uint64_t generation, bool ambiguity_ok = true)
{
  open3d_loc::recovery::Candidate result;
  result.generation = generation;
  result.window = {scan_id, scan_id, static_cast<double>(scan_id), static_cast<double>(scan_id)};
  result.map_to_odom = measured.map_odom;
  result.odom_to_robot = odom_base;
  open3d_loc::Matrix6d jacobian = open3d_loc::Matrix6d::Identity();
  jacobian.block<3, 3>(3, 3) = (measured.map_odom * odom_base).block<3, 3>(0, 0).transpose();
  result.covariance = jacobian * measured.covariance * jacobian.transpose();
  result.quality = {measured.accepted, ambiguity_ok, measured.accepted};
  return result;
}

open3d_loc::Matrix6d reset_covariance(
  const open3d_loc::RecoveryCandidate & measured, const Eigen::Matrix4d & odom_base)
{
  const Eigen::Vector3d lever =
    measured.map_odom.block<3, 3>(0, 0) * odom_base.block<3, 1>(0, 3);
  Eigen::Matrix3d skew;
  skew << 0.0, -lever.z(), lever.y(), lever.z(), 0.0, -lever.x(), -lever.y(), lever.x(), 0.0;
  open3d_loc::Matrix6d jacobian = open3d_loc::Matrix6d::Identity();
  jacobian.block<3, 3>(0, 3) = skew;
  return jacobian * measured.covariance * jacobian.transpose();
}
}  // namespace

TEST(RelocalizationPipeline, AsymmetricRoomSearchRecoversInjectedCorrection)
{
  const Cloud map = room();
  const Eigen::Matrix4d truth = pose(0.4, -0.3, 0.2, 0.2);
  Cloud scan = map;
  scan.Transform(truth.inverse());
  const Cloud unchanged = scan;
  const Eigen::Matrix4d odom_base = pose(0.2, 0.1, 0.0, 0.1);
  open3d_loc::RecoverySearch search(
    map, scan, odom_base, pose(1.0, 0.0, 0.0, kPi / 2.0) * truth, search_config());
  EXPECT_FALSE(search.tick(std::chrono::milliseconds(0), 0));
  finish(search);
  ASSERT_FALSE(search.candidates().empty());
  const auto & best = search.candidates().front();
  ASSERT_TRUE(best.accepted);
  EXPECT_LT((best.map_odom.block<3, 1>(0, 3) - truth.block<3, 1>(0, 3)).norm(), 0.1);
  EXPECT_LT(
    Eigen::AngleAxisd(
      best.map_odom.block<3, 3>(0, 0) * truth.block<3, 3>(0, 0).transpose()).angle(), 0.05);
  expect_covariance(best);
  ASSERT_EQ(scan.points_.size(), unchanged.points_.size());
  for (std::size_t i = 0; i < scan.points_.size(); ++i) {
    EXPECT_TRUE(scan.points_[i].isApprox(unchanged.points_[i], 0.0));
  }
}

TEST(RelocalizationPipeline, FloorOnlyDoesNotProvideObservableRecovery)
{
  Cloud floor;
  for (int x = -20; x <= 20; ++x) {
    for (int y = -20; y <= 20; ++y) {
      floor.points_.emplace_back(x * 0.15, y * 0.15, -1.0);
    }
  }
  open3d_loc::RecoverySearch search(
    floor, floor, Eigen::Matrix4d::Identity(), pose(0.5, 0.0, 0.0), search_config());
  finish(search);
  for (const auto & candidate : search.candidates()) {
    EXPECT_FALSE(candidate.accepted);
  }
  const auto measured = search.evaluateFresh(
    floor, Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity());
  EXPECT_FALSE(measured.accepted);
  open3d_loc::recovery::Config config;
  config.application_mode = open3d_loc::recovery::ApplicationMode::RESET;
  open3d_loc::recovery::RecoveryCoordinator coordinator(config);
  for (std::uint64_t id = 1; id <= 3; ++id) {
    const auto result = coordinator.observe(
      observation(measured, Eigen::Matrix4d::Identity(), id, coordinator.generation()),
      pose(0.5, 0.0, 0.0), static_cast<double>(id));
    EXPECT_FALSE(result.candidate_recorded);
    EXPECT_FALSE(result.should_apply);
  }
}

TEST(RelocalizationPipeline, CovarianceIsIndependentOfArbitraryMapOrigin)
{
  Cloud map = room();
  // Binary-exact geometry and voxel sizes avoid changing voxel membership at
  // decimal rounding boundaries when translating by a large map offset.
  for (auto & point : map.points_) {
    point = (point.array() * 32.0).round().matrix() / 32.0;
  }
  auto config = search_config();
  config.coarse_voxel = 0.5;
  config.middle_voxel = 0.25;
  config.fine_voxel = 0.125;
  const Eigen::Matrix4d odom_base = pose(0.3, -0.2, 0.0);
  open3d_loc::RecoverySearch local(
    map, map, odom_base, Eigen::Matrix4d::Identity(), config);
  const auto near = local.evaluateFresh(map, odom_base, Eigen::Matrix4d::Identity());
  ASSERT_TRUE(near.accepted);
  expect_covariance(near);

  // Integer multiples of voxel sizes preserve the same voxel memberships.
  const Eigen::Matrix4d shift = pose(1000.0, -2000.0, 300.0);
  Cloud distant_map = map;
  distant_map.Transform(shift);
  open3d_loc::RecoverySearch distant(
    distant_map, map, odom_base, shift, config);
  const auto far = distant.evaluateFresh(map, odom_base, shift);
  ASSERT_TRUE(far.accepted);
  expect_covariance(far);
  EXPECT_NEAR(far.information_ratio, near.information_ratio, 1e-5);
  EXPECT_LT((far.covariance - near.covariance).norm(), near.covariance.norm() * 0.02);
}

TEST(RelocalizationPipeline, FreshPartialScanRanksTruthAboveInjectedPose)
{
  const Cloud map = room();
  const Eigen::Matrix4d truth = pose(0.5, -0.4, 0.1, 0.15);
  Cloud initial_scan = map;
  initial_scan.Transform(truth.inverse());
  open3d_loc::RecoverySearch search(
    map, initial_scan, Eigen::Matrix4d::Identity(), pose(1.0, 0.0, 0.0) * truth,
    search_config());
  finish(search);
  ASSERT_FALSE(search.candidates().empty());
  ASSERT_TRUE(search.candidates().front().accepted);

  Cloud fresh;
  for (const auto & point : map.points_) {
    if (point.y() > -1.1) {
      fresh.points_.push_back(point);
    }
  }
  fresh.Transform(truth.inverse());
  const Eigen::Matrix4d moved = pose(0.45, 0.15, 0.0, 0.08);
  const auto confirmed = search.refineFresh(fresh, moved, search.candidates().front().map_odom);
  const auto incumbent = search.evaluateFresh(fresh, moved, pose(1.0, 0.0, 0.0) * truth);
  ASSERT_TRUE(confirmed.accepted);
  EXPECT_LT(confirmed.score, incumbent.score);
  EXPECT_LT((confirmed.map_odom - truth).norm(), 0.15);
  expect_covariance(confirmed);
}

TEST(RelocalizationPipeline, MovingFreshWindowsConfirmResetAndVerifyActualSearch)
{
  using namespace open3d_loc;
  const Cloud map = room();
  const Eigen::Matrix4d truth = pose(0.4, -0.3, 0.2, 0.2);
  const Eigen::Matrix4d injected = pose(1.0, 0.0, 0.0, kPi / 2.0) * truth;
  Cloud scan = map;
  scan.Transform(truth.inverse());
  RecoverySearch search(map, scan, Eigen::Matrix4d::Identity(), injected, search_config());
  finish(search);
  ASSERT_FALSE(search.candidates().empty());
  ASSERT_TRUE(search.candidates().front().accepted);
  recovery::Config config;
  config.application_mode = recovery::ApplicationMode::RESET;
  config.update_mask = {{true, true, true, true, true, true}};
  config.confirmation_windows = 3;
  config.verification_windows = 3;
  recovery::RecoveryCoordinator coordinator(config);
  GlobalCorrectionFilter filter;
  filter.reset(injected);
  int resets = 0;
  for (std::uint64_t id = 1; id <= 3; ++id) {
    const Eigen::Matrix4d moving_odom = pose(0.3 * id, 0.1 * id, 0.0, 0.04 * id);
    const Eigen::Matrix4d saved_odom = moving_odom;
    Cloud fresh;
    for (std::size_t i = 0; i < scan.points_.size(); ++i) {
      if ((i + id) % 7 != 0) {
        fresh.points_.push_back(scan.points_[i]);
      }
    }
    const Cloud saved_cloud = fresh;
    const auto measured = search.refineFresh(
      fresh, moving_odom, search.candidates().front().map_odom);
    ASSERT_TRUE(measured.accepted);
    const auto incumbent = search.evaluateFresh(fresh, moving_odom, filter.pose());
    ASSERT_LT(measured.score, 0.85 * incumbent.score);
    const auto candidate = observation(measured, moving_odom, id, coordinator.generation());
    const auto result = coordinator.observe(candidate, filter.pose(), static_cast<double>(id));
    if (id < 3) {
      EXPECT_FALSE(result.should_apply);
      EXPECT_TRUE(filter.pose().isApprox(injected));
    } else {
      ASSERT_TRUE(result.proposal_available);
      ASSERT_TRUE(result.should_apply);
      EXPECT_FALSE(result.verified);
      const auto requested = search.evaluateFresh(fresh, moving_odom, result.requested_pose);
      ASSERT_TRUE(requested.accepted);
      ASSERT_LT(requested.score, 0.85 * incumbent.score);
      const auto covariance = reset_covariance(requested, moving_odom);
      filter.reset(result.requested_pose, covariance);
      EXPECT_TRUE(filter.covariance().isApprox(covariance, 1e-6));
      ASSERT_TRUE(coordinator.acknowledgeApplied(result.generation, filter.pose()));
      ++resets;
      EXPECT_EQ(coordinator.state(), recovery::State::VERIFYING);
    }
    EXPECT_TRUE(moving_odom.isApprox(saved_odom, 0.0));
    ASSERT_EQ(fresh.points_.size(), saved_cloud.points_.size());
    for (std::size_t i = 0; i < fresh.points_.size(); ++i) {
      EXPECT_TRUE(fresh.points_[i].isApprox(saved_cloud.points_[i], 0.0));
    }
  }
  EXPECT_EQ(resets, 1);
  EXPECT_LT((filter.pose() - truth).norm(), 0.15);
  for (std::uint64_t id = 4; id <= 6; ++id) {
    const Eigen::Matrix4d moving_odom = pose(0.3 * id, 0.1 * id, 0.0, 0.04 * id);
    const auto measured = search.evaluateFresh(scan, moving_odom, filter.pose());
    ASSERT_TRUE(measured.accepted);
    const auto result = coordinator.observeVerification(
      observation(measured, moving_odom, id, coordinator.generation()), static_cast<double>(id));
    EXPECT_EQ(result.verified, id == 6);
  }
  EXPECT_EQ(coordinator.state(), recovery::State::VERIFIED);
  Eigen::SelfAdjointEigenSolver<Matrix6d> solver(filter.covariance());
  ASSERT_EQ(solver.info(), Eigen::Success);
  EXPECT_GT(solver.eigenvalues().minCoeff(), 0.0);
}

TEST(RelocalizationPipeline, ReusedScanCannotConfirmAndInvalidationRejectsOldSearch)
{
  using namespace open3d_loc;
  const Cloud map = room();
  RecoverySearch search(
    map, map, Eigen::Matrix4d::Identity(), pose(1.0, 0.0, 0.0), search_config());
  finish(search);
  ASSERT_FALSE(search.candidates().empty());
  const auto measured = search.refineFresh(
    map, Eigen::Matrix4d::Identity(), search.candidates().front().map_odom);
  ASSERT_TRUE(measured.accepted);
  recovery::Config config;
  config.application_mode = recovery::ApplicationMode::RESET;
  recovery::RecoveryCoordinator coordinator(config);
  const auto candidate = observation(
    measured,
    Eigen::Matrix4d::Identity(), 1, coordinator.generation());
  EXPECT_TRUE(coordinator.observe(candidate, pose(1.0, 0.0, 0.0), 1.0).candidate_recorded);
  const auto duplicate = coordinator.observe(candidate, pose(1.0, 0.0, 0.0), 1.1);
  EXPECT_FALSE(duplicate.candidate_recorded);
  EXPECT_FALSE(duplicate.should_apply);
  coordinator.invalidate();
  auto stale = candidate;
  stale.window = {2, 2, 2.0, 2.0};
  const auto result = coordinator.observe(stale, pose(1.0, 0.0, 0.0), 2.0);
  EXPECT_FALSE(result.candidate_recorded);
  EXPECT_FALSE(result.should_apply);
  EXPECT_EQ(coordinator.state(), recovery::State::IDLE);
}

TEST(RelocalizationPipeline, RepeatedRoomsProduceAmbiguityAndNoReset)
{
  using namespace open3d_loc;
  const Cloud scan = room();
  Cloud map = scan;
  Cloud second_room = scan;
  second_room.Transform(pose(6.0, 0.0, 0.0));
  map += second_room;
  auto config = search_config();
  config.xy_radius = 3.0;
  config.xy_step = 3.0;
  config.refine_count = 24;
  config.candidate_count = 4;
  RecoverySearch search(map, scan, Eigen::Matrix4d::Identity(), pose(3.0, 0.0, 0.0), config);
  finish(search);
  // Confirm that BOTH physical aliases were discovered by actual search.
  bool found_first = false, found_second = false;
  for (const auto & candidate : search.candidates()) {
    if (candidate.accepted) {
      found_first |= (candidate.map_odom - Eigen::Matrix4d::Identity()).norm() < 0.15;
      found_second |= (candidate.map_odom - pose(6.0, 0.0, 0.0)).norm() < 0.15;
    }
  }
  ASSERT_TRUE(found_first);
  ASSERT_TRUE(found_second);
  const std::vector<RecoveryCandidate> scores{
    search.evaluateFresh(scan, Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity()),
    search.evaluateFresh(scan, Eigen::Matrix4d::Identity(), pose(6.0, 0.0, 0.0))};
  ASSERT_EQ(scores.size(), 2u);
  ASSERT_TRUE(scores[0].accepted);
  ASSERT_TRUE(scores[1].accepted);
  const double best_score = std::min(scores[0].score, scores[1].score);
  const double other_score = std::max(scores[0].score, scores[1].score);
  const bool ambiguity_ok = other_score - best_score > 1e-9 && best_score <= 0.85 * other_score;
  EXPECT_FALSE(ambiguity_ok);
  recovery::Config coordinator_config;
  coordinator_config.application_mode = recovery::ApplicationMode::RESET;
  recovery::RecoveryCoordinator coordinator(coordinator_config);
  for (std::uint64_t id = 1; id <= 4; ++id) {
    const auto result = coordinator.observe(
      observation(
        scores[0], Eigen::Matrix4d::Identity(), id,
        coordinator.generation(), ambiguity_ok),
      pose(3.0, 0.0, 0.0), static_cast<double>(id));
    EXPECT_FALSE(result.candidate_recorded);
    EXPECT_FALSE(result.should_apply);
  }
  EXPECT_EQ(coordinator.state(), recovery::State::IDLE);
}

TEST(RelocalizationPipeline, LowOverlapCannotConfirmRecovery)
{
  using namespace open3d_loc;
  const Cloud map = room();
  Cloud remote_scan = map;
  remote_scan.Transform(pose(50.0, 20.0, 10.0));
  RecoverySearch search(
    map, remote_scan, Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity(), search_config());
  finish(search);
  EXPECT_TRUE(search.candidates().empty());
  const auto measured = search.evaluateFresh(
    remote_scan, Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity());
  EXPECT_FALSE(measured.accepted);
  EXPECT_LT(measured.fitness, search_config().min_fitness);
  recovery::Config config;
  config.application_mode = recovery::ApplicationMode::RESET;
  recovery::RecoveryCoordinator coordinator(config);
  for (std::uint64_t id = 1; id <= 3; ++id) {
    const auto result = coordinator.observe(
      observation(measured, Eigen::Matrix4d::Identity(), id, coordinator.generation()),
      Eigen::Matrix4d::Identity(), static_cast<double>(id));
    EXPECT_FALSE(result.should_apply);
  }
  EXPECT_EQ(coordinator.state(), recovery::State::IDLE);
}

TEST(RelocalizationPipeline, ShadowProposalNeverChangesFilterOrAcceptsAcknowledgement)
{
  using namespace open3d_loc;
  const Cloud map = room();
  const Eigen::Matrix4d injected = pose(1.0, 0.0, 0.0);
  RecoverySearch search(map, map, Eigen::Matrix4d::Identity(), injected, search_config());
  finish(search);
  ASSERT_FALSE(search.candidates().empty());
  recovery::RecoveryCoordinator coordinator;
  GlobalCorrectionFilter filter;
  filter.reset(injected);
  for (std::uint64_t id = 1; id <= 3; ++id) {
    const auto measured = search.refineFresh(
      map, Eigen::Matrix4d::Identity(), search.candidates().front().map_odom);
    ASSERT_TRUE(measured.accepted);
    const auto result = coordinator.observe(
      observation(measured, Eigen::Matrix4d::Identity(), id, coordinator.generation()),
      filter.pose(), static_cast<double>(id));
    EXPECT_FALSE(result.should_apply);
    EXPECT_EQ(result.proposal_available, id == 3);
    EXPECT_FALSE(coordinator.acknowledgeApplied(result.generation, result.requested_pose));
    EXPECT_TRUE(filter.pose().isApprox(injected, 0.0));
  }
}
