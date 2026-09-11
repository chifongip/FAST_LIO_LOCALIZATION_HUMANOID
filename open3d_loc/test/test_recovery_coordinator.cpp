// Copyright 2026 Open3D Loc Contributors
#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include <limits>
#include <stdexcept>
#include "open3d_loc/recovery_coordinator.hpp"
namespace
{
open3d_loc::RecoveryObservation sample(int i)
{
  open3d_loc::RecoveryObservation o;
  o.stamp = i; o.oldest_stamp = i - 0.5; o.scan_ids = {static_cast<std::uint64_t>(i)};
  o.quality_valid = o.unambiguous = o.observable = true;
  o.map_base(0, 3) = 5;
  o.map_base.block<3, 3>(
    0,
    0) = Eigen::AngleAxisd(1.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  return o;
}
}
TEST(RecoveryCoordinator, ThreeFreshWindowsThenResetAndVerification)
{
  open3d_loc::RecoveryCoordinator c;
  EXPECT_FALSE(c.observe(sample(1), Eigen::Matrix4d::Identity()).proposal);
  EXPECT_FALSE(c.observe(sample(2), Eigen::Matrix4d::Identity()).proposal);
  auto decision = c.observe(sample(3), Eigen::Matrix4d::Identity());
  ASSERT_TRUE(decision.proposal);
  EXPECT_NEAR(decision.map_odom(0, 3), 5.0, 1e-12);
  c.applied(1, sample(3));
  for (int i = 4; i <= 6; ++i) {
    auto o = sample(i); o.generation = 1;
    EXPECT_FALSE(c.observe(o, decision.map_odom).proposal);
  }
  EXPECT_EQ(c.state(), open3d_loc::RelocalizationState::TRACKING);
}
TEST(RecoveryCoordinator, MotionAndDistantOdometryOrigin)
{
  open3d_loc::RecoveryCoordinator c;
  Eigen::Matrix4d correction = sample(1).map_base;
  for (int i = 1; i <= 3; ++i) {
    auto o = sample(i);
    o.odom_base(0, 3) = 1000.0 + i;
    o.odom_base.block<3, 3>(
      0,
      0) = Eigen::AngleAxisd(i * 0.1, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    o.map_base = correction * o.odom_base;
    auto d = c.observe(o, Eigen::Matrix4d::Identity());
    EXPECT_EQ(d.proposal, i == 3);
    EXPECT_LT(d.translation_residual, 1e-9);
  }
}
TEST(RecoveryCoordinator, DuplicateAmbiguousAndStaleEvidenceCannotReset)
{
  open3d_loc::RecoveryCoordinator c;
  c.observe(sample(1), Eigen::Matrix4d::Identity());
  auto repeated = sample(2); repeated.scan_ids = {1};
  EXPECT_EQ(c.observe(repeated, Eigen::Matrix4d::Identity()).reason, "overlapping_scan_window");
  auto ambiguous = sample(3); ambiguous.unambiguous = false;
  EXPECT_FALSE(c.observe(ambiguous, Eigen::Matrix4d::Identity()).proposal);
  EXPECT_EQ(c.count(), 0);
  c.observe(sample(4), Eigen::Matrix4d::Identity());
  EXPECT_FALSE(c.observe(sample(8), Eigen::Matrix4d::Identity()).proposal);
  c.invalidate(2);
  EXPECT_EQ(c.observe(sample(9), Eigen::Matrix4d::Identity()).reason, "generation_mismatch");
}
TEST(RecoveryCoordinator, FirstAnchorPreventsCreepingCandidate)
{
  open3d_loc::RecoveryCoordinator c;
  for (int i = 1; i <= 3; ++i) {
    auto o = sample(i); o.map_base(0, 3) += 0.3 * (i - 1);
    EXPECT_FALSE(c.observe(o, Eigen::Matrix4d::Identity()).proposal);
  }
  EXPECT_EQ(c.count(), 1);
}
TEST(RecoveryCoordinator, ShadowAndMask)
{
  open3d_loc::RecoveryCoordinatorConfig config; config.application_mode = "shadow";
  config.update_mask = {{true, false, false, false, false, true}};
  open3d_loc::RecoveryCoordinator c(config);
  open3d_loc::RecoveryDecision d;
  for (int i = 1; i <= 3; ++i) {
    auto o = sample(i); o.map_base(1, 3) = 3; d = c.observe(o, Eigen::Matrix4d::Identity());
  }
  EXPECT_TRUE(d.proposal); EXPECT_TRUE(d.shadow); EXPECT_DOUBLE_EQ(d.map_odom(1, 3), 0);
}
TEST(RecoveryCoordinator, CorrectionCovarianceMatchesNumericalJacobian)
{
  Eigen::Matrix4d odom = Eigen::Matrix4d::Identity(); odom(0, 3) = 100;
  const Eigen::Matrix4d correction = sample(1).map_base;
  const Eigen::Matrix4d robot = correction * odom;
  open3d_loc::Matrix6d j;
  for (int i = 0; i < 6; ++i) {
    auto perturbed = robot;
    if (i < 3) {perturbed(i, 3) += 1e-6;} else {
      perturbed.block<3, 3>(
        0,
        0) = Eigen::AngleAxisd(
        1e-6,
        Eigen::Vector3d::Unit(i - 3)).toRotationMatrix() * robot.block<3, 3>(0, 0);
    }
    const Eigen::Matrix4d next = perturbed * odom.inverse();
    j.block<3, 1>(0, i) = (next.block<3, 1>(0, 3) - correction.block<3, 1>(0, 3)) / 1e-6;
    const Eigen::AngleAxisd r(next.block<3, 3>(0, 0) * correction.block<3, 3>(0, 0).transpose());
    j.block<3, 1>(3, i) = r.angle() * r.axis() / 1e-6;
  }
  const auto analytic = open3d_loc::RecoveryCoordinator::correctionCovariance(
    open3d_loc::Matrix6d::Identity(), correction, odom);
  EXPECT_LT((analytic - j * j.transpose()).norm() / analytic.norm(), 1e-5);
}

TEST(RecoveryCoordinator, BoundedStepsRetainTargetAndVerifyOnlyAfterArrival)
{
  open3d_loc::RecoveryCoordinatorConfig config;
  config.application_mode = "bounded_step";
  config.max_step_translation = 1.0;
  config.max_step_rotation = 0.5;
  open3d_loc::RecoveryCoordinator coordinator(config);
  Eigen::Matrix4d current = Eigen::Matrix4d::Identity();
  open3d_loc::RecoveryDecision decision;
  for (int i = 1; i <= 3; ++i) {
    decision = coordinator.observe(sample(i), current);
  }
  ASSERT_TRUE(decision.proposal);
  std::uint64_t generation = 0;
  for (int step = 0; step < 5; ++step) {
    ASSERT_TRUE(decision.proposal);
    EXPECT_NEAR(decision.map_odom(0, 3), step + 1, 1e-10);
    auto applied = sample(step + 3);
    applied.generation = generation;
    // Passing the target instead of the applied step must not acknowledge it.
    if (step == 0) {
      coordinator.applied(generation + 1, applied);
      EXPECT_EQ(coordinator.state(), open3d_loc::RelocalizationState::CONFIRMING);
      EXPECT_EQ(coordinator.count(), 3);
    }
    current = decision.map_odom;
    applied.map_base = current * applied.odom_base;
    coordinator.applied(++generation, applied);
    if (step < 4) {
      EXPECT_EQ(coordinator.state(), open3d_loc::RelocalizationState::CONFIRMING);
      EXPECT_EQ(coordinator.count(), 2);
      auto next = sample(step + 4);
      next.generation = generation;
      // Fresh measurements may jitter, but the confirmed bounded target stays fixed.
      next.map_base(0, 3) += 0.05;
      decision = coordinator.observe(next, current);
    } else {
      EXPECT_EQ(coordinator.state(), open3d_loc::RelocalizationState::VERIFYING);
    }
  }
  for (int i = 8; i <= 10; ++i) {
    auto incumbent = sample(i);
    incumbent.generation = generation;
    coordinator.observe(incumbent, current);
  }
  EXPECT_EQ(coordinator.state(), open3d_loc::RelocalizationState::TRACKING);
}

TEST(RecoveryCoordinator, FailedVerificationQualityIsConsumedBeforeMinimumInterval)
{
  open3d_loc::RecoveryCoordinator coordinator;
  open3d_loc::RecoveryDecision decision;
  for (int i = 1; i <= 3; ++i) {
    decision = coordinator.observe(sample(i), Eigen::Matrix4d::Identity());
  }
  coordinator.applied(1, sample(3));
  auto failure = sample(4);
  failure.generation = 1;
  failure.stamp = 3.2;
  failure.oldest_stamp = 3.1;
  failure.quality_valid = false;
  failure.map_base.setConstant(std::numeric_limits<double>::quiet_NaN());
  EXPECT_EQ(coordinator.observe(failure, decision.map_odom).reason, "verification_failed");
  EXPECT_EQ(coordinator.state(), open3d_loc::RelocalizationState::DEGRADED);
}

TEST(RecoveryCoordinator, WallTimeExpiryAllowsNormalAcquisitionLatency)
{
  open3d_loc::RecoveryCoordinator coordinator;
  for (int i = 1; i <= 3; ++i) {
    coordinator.expire(i + 0.1);
    const auto decision = coordinator.observe(sample(i), Eigen::Matrix4d::Identity());
    EXPECT_EQ(decision.proposal, i == 3);
  }
  coordinator.expire(6.01);
  EXPECT_FALSE(coordinator.acknowledgeApplied(0, sample(3).map_base));
  EXPECT_EQ(coordinator.state(), open3d_loc::RelocalizationState::DEGRADED);
}

TEST(RecoveryCoordinator, SearchWorkerLaunchPreservesConfirmationAndVerification)
{
  open3d_loc::RecoveryCoordinator coordinator;
  open3d_loc::RecoveryDecision decision;
  for (int i = 1; i <= 3; ++i) {
    coordinator.searching();
    decision = coordinator.observe(sample(i), Eigen::Matrix4d::Identity());
    EXPECT_EQ(decision.count, i);
    EXPECT_EQ(decision.proposal, i == 3);
  }
  coordinator.searching();
  coordinator.applied(1, sample(3));
  ASSERT_EQ(coordinator.state(), open3d_loc::RelocalizationState::VERIFYING);
  for (int i = 4; i <= 6; ++i) {
    coordinator.searching();
    EXPECT_EQ(coordinator.state(), open3d_loc::RelocalizationState::VERIFYING);
    auto incumbent = sample(i);
    incumbent.generation = 1;
    decision = coordinator.observe(incumbent, decision.map_odom);
    EXPECT_EQ(decision.count, i - 3);
  }
  EXPECT_EQ(coordinator.state(), open3d_loc::RelocalizationState::TRACKING);
}

TEST(RecoveryCoordinator, LegacyTwoObservationConfigurationStillConstructs)
{
  open3d_loc::RecoveryCoordinatorConfig config;
  config.required_observations = 2;
  open3d_loc::RecoveryCoordinator coordinator(config);
  EXPECT_FALSE(coordinator.observe(sample(1), Eigen::Matrix4d::Identity()).proposal);
  EXPECT_TRUE(coordinator.observe(sample(2), Eigen::Matrix4d::Identity()).proposal);
  config.required_observations = 1;
  EXPECT_THROW(open3d_loc::RecoveryCoordinator{config}, std::invalid_argument);
  config.required_observations = 3;
  config.application_mode = "bounded";
  EXPECT_NO_THROW(open3d_loc::RecoveryCoordinator{config});
}

namespace
{
open3d_loc::recovery::Candidate core_sample(int i, double x = 0, double yaw = 0)
{
  open3d_loc::recovery::Candidate input;
  input.window = {static_cast<std::uint64_t>(i), static_cast<std::uint64_t>(i),
    i - 0.5, static_cast<double>(i)};
  input.quality = {true, true, true};
  input.map_to_odom(0, 3) = x;
  input.map_to_odom.block<3, 3>(0, 0) =
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  return input;
}
}

TEST(RecoveryCore, OdometryHistoryInterpolatesFullPoseAndPrunes)
{
  open3d_loc::OdometryHistory history(3, 2);
  Eigen::Matrix4d start = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d end = start;
  end.block<3, 1>(0, 3) = Eigen::Vector3d(2, 4, 6);
  end.block<3, 3>(0, 0) =
    Eigen::AngleAxisd(1.0, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
  ASSERT_TRUE(history.add(1, start));
  ASSERT_TRUE(history.add(3, end));
  ASSERT_TRUE(history.interpolate(2));
  const auto middle = *history.interpolate(2);
  EXPECT_TRUE((middle.block<3, 1>(0, 3).isApprox(Eigen::Vector3d(1, 2, 3))));
  EXPECT_NEAR(Eigen::AngleAxisd(middle.block<3, 3>(0, 0)).angle(), 0.5, 1e-12);
  EXPECT_FALSE(history.interpolate(0));
  EXPECT_FALSE(history.interpolate(4));
  EXPECT_FALSE(history.add(3, end));
  EXPECT_FALSE(history.add(2, end));
  EXPECT_FALSE(history.add(4, Eigen::Matrix4d::Zero()));
  EXPECT_TRUE(history.add(5, start));
  EXPECT_EQ(history.size(), 2u);
  EXPECT_FALSE(history.interpolate(1));
  history.clear();
  EXPECT_EQ(history.size(), 0u);
}

TEST(RecoveryCore, ThreeWindowsNoiseFloorAndExplicitAcknowledgement)
{
  using namespace open3d_loc::recovery;
  Config config;
  config.application_mode = ApplicationMode::RESET;
  RecoveryCoordinator coordinator(config);
  Result proposal;
  for (int i = 1; i <= 3; ++i) {
    proposal = coordinator.observe(core_sample(i, 2), Eigen::Matrix4d::Identity(), i);
    EXPECT_EQ(proposal.should_apply, i == 3);
  }
  EXPECT_EQ(coordinator.state(), State::PROPOSED);
  EXPECT_FALSE(coordinator.acknowledgeApplied(1, proposal.requested_pose));
  EXPECT_FALSE(coordinator.acknowledgeApplied(0, Eigen::Matrix4d::Identity()));
  ASSERT_TRUE(coordinator.acknowledgeApplied(0, proposal.requested_pose));
  for (int i = 4; i <= 6; ++i) {
    EXPECT_EQ(coordinator.observeVerification(core_sample(i, 2.01), i).verified, i == 6);
  }
}

TEST(RecoveryCore, BothAnchorsRejectDriftAndOscillation)
{
  using namespace open3d_loc::recovery;
  for (double last : {0.6, -0.3}) {
    RecoveryCoordinator coordinator;
    coordinator.observe(core_sample(1), Eigen::Matrix4d::Identity(), 1);
    coordinator.observe(core_sample(2, 0.3), Eigen::Matrix4d::Identity(), 2);
    EXPECT_EQ(
      coordinator.observe(core_sample(3, last), Eigen::Matrix4d::Identity(), 3).reason,
      "candidate_inconsistent");
  }
}

TEST(RecoveryCore, MotionCompensationAndRobotLeverArm)
{
  using namespace open3d_loc::recovery;
  RecoveryCoordinator moving;
  for (int i = 1; i <= 3; ++i) {
    auto input = core_sample(i, 3, 0.4);
    input.odom_to_robot(0, 3) = 100 * i;
    input.odom_to_robot(2, 3) = 2 * i;
    input.odom_to_robot.block<3, 3>(0, 0) =
      Eigen::AngleAxisd(i * 0.4, Eigen::Vector3d::UnitX()).toRotationMatrix();
    EXPECT_EQ(moving.observe(input, Eigen::Matrix4d::Identity(), i).proposal_available, i == 3);
  }
  RecoveryCoordinator lever;
  auto input = core_sample(1);
  input.odom_to_robot(0, 3) = 100;
  lever.observe(input, Eigen::Matrix4d::Identity(), 1);
  input = core_sample(2, 0, 0.01);
  input.odom_to_robot(0, 3) = 100;
  const auto out = lever.observe(input, Eigen::Matrix4d::Identity(), 2);
  EXPECT_EQ(out.reason, "candidate_inconsistent");
  EXPECT_GT(out.translation_delta, 0.99);
}

TEST(RecoveryCore, NisAndHardGates)
{
  using namespace open3d_loc::recovery;
  Config config;
  config.nis_threshold = 1;
  RecoveryCoordinator strict(config);
  strict.observe(core_sample(1), Eigen::Matrix4d::Identity(), 1);
  const auto out = strict.observe(core_sample(2, 0.3), Eigen::Matrix4d::Identity(), 2);
  EXPECT_EQ(out.reason, "candidate_inconsistent");
  EXPECT_NEAR(out.nis, 2.0, 1e-12);
  for (bool rotate : {false, true}) {
    RecoveryCoordinator hard;
    auto first = core_sample(1);
    first.covariance *= 0;
    first.covariance.diagonal().setConstant(1000);
    hard.observe(first, Eigen::Matrix4d::Identity(), 1);
    auto next = core_sample(2, rotate ? 0 : 0.501, rotate ? 0.201 : 0);
    next.covariance = first.covariance;
    EXPECT_EQ(hard.observe(next, Eigen::Matrix4d::Identity(), 2).reason, "candidate_inconsistent");
  }
}

TEST(RecoveryCore, TimeoutInvalidationAndShadowCannotCommit)
{
  using namespace open3d_loc::recovery;
  RecoveryCoordinator coordinator;
  Result out;
  for (int i = 1; i <= 3; ++i) {
    out = coordinator.observe(core_sample(i), Eigen::Matrix4d::Identity(), i);
  }
  ASSERT_TRUE(out.proposal_available);
  EXPECT_FALSE(out.should_apply);
  EXPECT_FALSE(coordinator.acknowledgeApplied(0, out.requested_pose));
  EXPECT_EQ(coordinator.tick(6).state, State::PROPOSED);
  EXPECT_EQ(coordinator.tick(6.01).state, State::IDLE);
  EXPECT_EQ(coordinator.generation(), 1u);
  EXPECT_EQ(
    coordinator.observe(core_sample(7), Eigen::Matrix4d::Identity(), 7).reason,
    "stale_generation");
  coordinator.tick(1);
  EXPECT_EQ(coordinator.generation(), 2u);
}

TEST(RecoveryCore, QualityCovarianceAndWindowValidation)
{
  using namespace open3d_loc::recovery;
  for (int gate = 0; gate < 3; ++gate) {
    RecoveryCoordinator coordinator;
    auto input = core_sample(1);
    if (gate == 0) {input.quality.quality_ok = false;}
    if (gate == 1) {input.quality.ambiguity_ok = false;}
    if (gate == 2) {input.quality.observability_ok = false;}
    EXPECT_FALSE(coordinator.observe(input, Eigen::Matrix4d::Identity(), 1).candidate_recorded);
  }
  RecoveryCoordinator coordinator;
  auto input = core_sample(1);
  input.covariance(0, 0) = -1;
  EXPECT_FALSE(coordinator.observe(input, Eigen::Matrix4d::Identity(), 1).candidate_recorded);
  input = core_sample(2);
  EXPECT_TRUE(coordinator.observe(input, Eigen::Matrix4d::Identity(), 2).candidate_recorded);
  input = core_sample(3);
  input.window.first_scan_id = 2;
  EXPECT_FALSE(coordinator.observe(input, Eigen::Matrix4d::Identity(), 3).candidate_recorded);
  input = core_sample(4);
  input.window.start_stamp = 2;
  EXPECT_FALSE(coordinator.observe(input, Eigen::Matrix4d::Identity(), 4).candidate_recorded);
}

TEST(RecoveryCore, InvalidRegistrationBreaksAnExistingConfirmationStreak)
{
  using namespace open3d_loc::recovery;
  RecoveryCoordinator coordinator;
  coordinator.observe(core_sample(1), Eigen::Matrix4d::Identity(), 1);
  auto failure = core_sample(2);
  failure.covariance.setConstant(std::numeric_limits<double>::quiet_NaN());
  failure.quality.quality_ok = false;
  EXPECT_FALSE(coordinator.observe(failure, Eigen::Matrix4d::Identity(), 2).candidate_recorded);
  EXPECT_EQ(coordinator.state(), State::IDLE);
  auto next = coordinator.observe(core_sample(3), Eigen::Matrix4d::Identity(), 3);
  EXPECT_EQ(next.consistent_windows, 1);
  EXPECT_FALSE(next.proposal_available);
}
