// Copyright 2026 Open3D Loc Contributors
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd
#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <limits>

#include "open3d_loc/global_correction_filter.hpp"
#include "open3d_loc/height_bounds.hpp"

namespace
{
Eigen::Matrix4d pose(double x, double y, double z, double yaw)
{
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.block<3, 3>(0, 0) =
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  transform.block<3, 1>(0, 3) = Eigen::Vector3d(x, y, z);
  return transform;
}
}  // namespace

TEST(GlobalCorrectionFilter, BlendsAcceptedMeasurement)
{
  open3d_loc::GlobalCorrectionFilter filter;
  filter.reset(Eigen::Matrix4d::Identity());
  const auto result = filter.update(
    pose(0.2, 0.0, 0.0, 0.1),
    open3d_loc::Matrix6d::Identity() * 0.01);

  EXPECT_TRUE(result.accepted);
  EXPECT_GT(filter.pose()(0, 3), 0.0);
  EXPECT_LT(filter.pose()(0, 3), 0.2);
}

TEST(GlobalCorrectionFilter, DisabledAxesRemainUnchanged)
{
  open3d_loc::GlobalCorrectionFilterConfig config;
  config.update_mask = {{true, true, true, false, false, true}};
  open3d_loc::GlobalCorrectionFilter filter(config);
  filter.reset(Eigen::Matrix4d::Identity());
  Eigen::Matrix4d measurement = pose(0.1, 0.1, 0.1, 0.1);
  measurement.block<3, 3>(0, 0) =
    Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX()).toRotationMatrix() *
    Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  EXPECT_TRUE(filter.update(measurement, open3d_loc::Matrix6d::Identity() * 0.01).accepted);
  const Eigen::Vector3d euler = filter.pose().block<3, 3>(0, 0).eulerAngles(0, 1, 2);
  EXPECT_NEAR(euler.x(), 0.0, 1e-12);
}

TEST(GlobalCorrectionFilter, DisabledAxesIgnoreCorrelatedCovarianceUpdates)
{
  open3d_loc::GlobalCorrectionFilterConfig config;
  config.update_mask = {{true, false, false, false, false, false}};
  open3d_loc::GlobalCorrectionFilter filter(config);
  open3d_loc::Matrix6d covariance = open3d_loc::Matrix6d::Identity() * 0.1;
  covariance(0, 3) = 0.02;
  covariance(3, 0) = 0.02;
  filter.reset(Eigen::Matrix4d::Identity(), covariance);
  const double roll_variance = filter.covariance()(3, 3);

  EXPECT_TRUE(
    filter.update(
      pose(0.1, 0.0, 0.0, 0.0), open3d_loc::Matrix6d::Identity() * 0.01).accepted);
  const Eigen::Vector3d euler = filter.pose().block<3, 3>(0, 0).eulerAngles(0, 1, 2);
  EXPECT_NEAR(euler.x(), 0.0, 1e-12);
  EXPECT_NEAR(filter.covariance()(3, 3), roll_variance, 1e-12);
}

TEST(GlobalCorrectionFilter, RejectsLargeJump)
{
  open3d_loc::GlobalCorrectionFilter filter;
  filter.reset(Eigen::Matrix4d::Identity());
  const auto result = filter.update(
    pose(5.0, 0.0, 0.0, 0.0),
    open3d_loc::Matrix6d::Identity() * 0.01);

  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.reason, "translation_gate");
  EXPECT_TRUE(filter.pose().isApprox(Eigen::Matrix4d::Identity()));
}

TEST(GlobalCorrectionFilter, PredictionGrowsCovariance)
{
  open3d_loc::GlobalCorrectionFilter filter;
  filter.reset(Eigen::Matrix4d::Identity());
  const auto before = filter.covariance();
  filter.predict(1.0, 2.0, 0.5);

  EXPECT_GT(filter.covariance()(0, 0), before(0, 0));
  EXPECT_GT(filter.covariance()(5, 5), before(5, 5));
}

TEST(GlobalCorrectionFilter, CovarianceRemainsPositiveDefinite)
{
  open3d_loc::GlobalCorrectionFilter filter;
  filter.reset(Eigen::Matrix4d::Identity());
  EXPECT_TRUE(
    filter.update(
      pose(0.1, 0.0, 0.0, 0.05),
      open3d_loc::Matrix6d::Identity() * 0.001).accepted);

  Eigen::SelfAdjointEigenSolver<open3d_loc::Matrix6d> solver(filter.covariance());
  EXPECT_EQ(solver.info(), Eigen::Success);
  EXPECT_GT(solver.eigenvalues().minCoeff(), 0.0);
}

TEST(GlobalCorrectionFilter, MeasurementCovarianceControlsCorrectionStrength)
{
  open3d_loc::GlobalCorrectionFilter low_noise_filter;
  open3d_loc::GlobalCorrectionFilter high_noise_filter;
  low_noise_filter.reset(Eigen::Matrix4d::Identity());
  high_noise_filter.reset(Eigen::Matrix4d::Identity());

  EXPECT_TRUE(
    low_noise_filter.update(
      pose(0.2, 0.0, 0.0, 0.0), open3d_loc::Matrix6d::Identity() * 0.001).accepted);
  EXPECT_TRUE(
    high_noise_filter.update(
      pose(0.2, 0.0, 0.0, 0.0), open3d_loc::Matrix6d::Identity() * 0.5).accepted);
  EXPECT_GT(low_noise_filter.pose()(0, 3), high_noise_filter.pose()(0, 3));
}

TEST(GlobalCorrectionFilter, HandlesYawWraparound)
{
  open3d_loc::GlobalCorrectionFilter filter;
  filter.reset(pose(0.0, 0.0, 0.0, M_PI - 0.05));
  const auto result = filter.update(
    pose(0.0, 0.0, 0.0, -M_PI + 0.05), open3d_loc::Matrix6d::Identity() * 0.01);

  EXPECT_TRUE(result.accepted);
  EXPECT_NEAR(result.rotation_innovation, 0.1, 1e-6);
}

TEST(GlobalCorrectionFilter, MahalanobisGateRejectsOverconfidentMeasurement)
{
  open3d_loc::GlobalCorrectionFilterConfig config;
  config.initial_stddev = open3d_loc::Vector6d::Constant(0.01);
  config.max_innovation_translation = 10.0;
  open3d_loc::GlobalCorrectionFilter filter(config);
  filter.reset(Eigen::Matrix4d::Identity());

  const auto result = filter.update(
    pose(0.5, 0.0, 0.0, 0.0), open3d_loc::Matrix6d::Identity() * 0.0001);
  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.reason, "mahalanobis_gate");
}

TEST(GlobalCorrectionFilter, ResetReplacesStateAndCovariance)
{
  open3d_loc::GlobalCorrectionFilter filter;
  filter.reset(Eigen::Matrix4d::Identity());
  filter.predict(2.0, 1.0, 0.5);
  const Eigen::Matrix4d reset_pose = pose(1.0, 2.0, 3.0, 0.4);
  filter.reset(reset_pose);

  EXPECT_TRUE(filter.pose().isApprox(reset_pose, 1e-12));
  EXPECT_NEAR(filter.covariance()(0, 0), 0.25 * 0.25, 1e-12);
}

TEST(GlobalCorrectionFilter, NonFinitePredictionDoesNotContaminateCovariance)
{
  open3d_loc::GlobalCorrectionFilter filter;
  filter.reset(Eigen::Matrix4d::Identity());
  const auto before = filter.covariance();
  filter.predict(std::numeric_limits<double>::quiet_NaN(), 1.0, 0.1);

  EXPECT_TRUE(filter.covariance().isApprox(before, 1e-12));
}

TEST(GlobalCorrectionFilter, RecoveryRequiresTwoConsistentCandidates)
{
  open3d_loc::GlobalCorrectionFilterConfig config;
  config.recovery.enabled = true;
  open3d_loc::GlobalCorrectionFilter filter(config);
  filter.reset(Eigen::Matrix4d::Identity());
  const auto covariance = open3d_loc::Matrix6d::Identity() * 0.01;

  EXPECT_EQ(
    filter.update(pose(2.0, 0.0, 0.0, 0.0), covariance).reason,
    "translation_gate");
  const auto first = filter.observeRecoveryCandidate(
    pose(2.0, 0.0, 0.0, 0.0), covariance, 1.0, 0.0, 0.0);
  EXPECT_FALSE(first.correction_applied);
  EXPECT_EQ(first.mode, open3d_loc::RecoveryMode::CONFIRMING);
  EXPECT_TRUE(filter.pose().isApprox(Eigen::Matrix4d::Identity()));

  const auto second = filter.observeRecoveryCandidate(
    pose(2.05, 0.0, 0.0, 0.0), covariance, 2.0, 0.5, 0.1);
  EXPECT_TRUE(second.correction_applied);
  EXPECT_EQ(second.mode, open3d_loc::RecoveryMode::RECOVERING);
  EXPECT_NEAR(second.applied_translation, 0.5, 1e-12);
  EXPECT_NEAR(filter.pose()(0, 3), 0.5, 1e-12);
}

TEST(GlobalCorrectionFilter, RecoveryRejectsRepeatedScanCandidate)
{
  open3d_loc::GlobalCorrectionFilterConfig config;
  config.recovery.enabled = true;
  open3d_loc::GlobalCorrectionFilter filter(config);
  filter.reset(Eigen::Matrix4d::Identity());
  const auto covariance = open3d_loc::Matrix6d::Identity() * 0.01;

  filter.observeRecoveryCandidate(pose(2.0, 0.0, 0.0, 0.0), covariance, 1.0, 0.0, 0.0);
  const auto repeated = filter.observeRecoveryCandidate(
    pose(2.0, 0.0, 0.0, 0.0), covariance, 1.5, 0.1, 0.0);

  EXPECT_FALSE(repeated.correction_applied);
  EXPECT_EQ(repeated.reason, "recovery_candidate_too_soon");
  EXPECT_EQ(filter.recoveryCandidateCount(), 1);
}

TEST(GlobalCorrectionFilter, RecoveryUsesMotionDependentConsistencyCovariance)
{
  open3d_loc::GlobalCorrectionFilterConfig config;
  config.recovery.enabled = true;
  config.process_stddev_distance = open3d_loc::Vector6d::Zero();
  config.process_stddev_distance(0) = 0.2;
  config.recovery.max_candidate_translation_delta = 0.5;
  open3d_loc::GlobalCorrectionFilter filter(config);
  filter.reset(Eigen::Matrix4d::Identity());
  const auto covariance = open3d_loc::Matrix6d::Identity() * 0.0001;

  filter.observeRecoveryCandidate(pose(2.0, 0.0, 0.0, 0.0), covariance, 1.0, 0.0, 0.0);
  const auto result = filter.observeRecoveryCandidate(
    pose(2.2, 0.0, 0.0, 0.0), covariance, 2.0, 2.0, 0.0);

  EXPECT_TRUE(result.correction_applied);
  EXPECT_GT(result.motion_distance, 1.9);
  EXPECT_LT(result.consistency_mahalanobis_distance, 13.277);
}

TEST(GlobalCorrectionFilter, InconsistentCandidateRestartsConfirmation)
{
  open3d_loc::GlobalCorrectionFilterConfig config;
  config.recovery.enabled = true;
  open3d_loc::GlobalCorrectionFilter filter(config);
  filter.reset(Eigen::Matrix4d::Identity());
  const auto covariance = open3d_loc::Matrix6d::Identity() * 0.01;

  filter.observeRecoveryCandidate(pose(2.0, 0.0, 0.0, 0.0), covariance, 1.0, 0.0, 0.0);
  const auto inconsistent = filter.observeRecoveryCandidate(
    pose(3.0, 0.0, 0.0, 0.0), covariance, 2.0, 0.0, 0.0);

  EXPECT_FALSE(inconsistent.correction_applied);
  EXPECT_EQ(inconsistent.reason, "recovery_translation_inconsistent");
  EXPECT_EQ(filter.recoveryCandidateCount(), 1);
  EXPECT_EQ(filter.recoveryMode(), open3d_loc::RecoveryMode::CONFIRMING);
}

TEST(GlobalCorrectionFilter, RecoveryCandidateExpires)
{
  open3d_loc::GlobalCorrectionFilterConfig config;
  config.recovery.enabled = true;
  open3d_loc::GlobalCorrectionFilter filter(config);
  filter.reset(Eigen::Matrix4d::Identity());
  const auto covariance = open3d_loc::Matrix6d::Identity() * 0.01;

  filter.observeRecoveryCandidate(pose(2.0, 0.0, 0.0, 0.0), covariance, 1.0, 0.0, 0.0);
  const auto expired = filter.observeRecoveryCandidate(
    pose(2.0, 0.0, 0.0, 0.0), covariance, 5.0, 0.0, 0.0);

  EXPECT_FALSE(expired.correction_applied);
  EXPECT_EQ(expired.reason, "recovery_candidate_timeout");
  EXPECT_EQ(filter.recoveryCandidateCount(), 1);
}

TEST(GlobalCorrectionFilter, RecoveryConvergesAndReturnsToTracking)
{
  open3d_loc::GlobalCorrectionFilterConfig config;
  config.recovery.enabled = true;
  open3d_loc::GlobalCorrectionFilter filter(config);
  filter.reset(Eigen::Matrix4d::Identity());
  const auto covariance = open3d_loc::Matrix6d::Identity() * 0.01;
  const auto measurement = pose(2.0, 0.0, 0.0, 0.5);

  EXPECT_FALSE(filter.update(measurement, covariance).accepted);
  filter.observeRecoveryCandidate(measurement, covariance, 1.0, 0.0, 0.0);
  EXPECT_TRUE(
    filter.observeRecoveryCandidate(measurement, covariance, 2.0, 0.5, 0.1)
    .correction_applied);
  for (int i = 0; i < 5 && filter.recoveryMode() != open3d_loc::RecoveryMode::TRACKING; ++i) {
    const auto update = filter.update(measurement, covariance);
    if (!update.accepted) {
      const auto recovery = filter.observeRecoveryCandidate(
        measurement, covariance, 3.0 + i, 1.0 + i * 0.5, 0.2 + i * 0.1);
      EXPECT_LE(recovery.applied_translation, 0.5 + 1e-12);
      EXPECT_LE(recovery.applied_rotation, 0.15 + 1e-12);
    }
  }

  EXPECT_EQ(filter.recoveryMode(), open3d_loc::RecoveryMode::TRACKING);
  EXPECT_TRUE(filter.update(measurement, covariance).accepted);
  EXPECT_NEAR(filter.pose()(0, 3), 2.0, 0.1);
}

TEST(GlobalCorrectionFilter, RecoveryPreservesDisabledRollAndPitch)
{
  open3d_loc::GlobalCorrectionFilterConfig config;
  config.recovery.enabled = true;
  open3d_loc::GlobalCorrectionFilter filter(config);
  filter.reset(Eigen::Matrix4d::Identity());
  Eigen::Matrix4d measurement = pose(2.0, 0.0, 0.0, 0.4);
  measurement.block<3, 3>(0, 0) =
    Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX()).toRotationMatrix() *
    measurement.block<3, 3>(0, 0);
  const auto covariance = open3d_loc::Matrix6d::Identity() * 0.01;

  filter.observeRecoveryCandidate(measurement, covariance, 1.0, 0.0, 0.0);
  EXPECT_TRUE(
    filter.observeRecoveryCandidate(measurement, covariance, 2.0, 0.0, 0.0)
    .correction_applied);

  const Eigen::Vector3d euler = filter.pose().block<3, 3>(0, 0).eulerAngles(0, 1, 2);
  EXPECT_NEAR(euler.x(), 0.0, 1e-12);
  EXPECT_NEAR(euler.y(), 0.0, 1e-12);
}

TEST(HeightBounds, ProjectsComposedHeightAndPreservesOtherCoordinates)
{
  open3d_loc::HeightBounds bounds;
  bounds.enabled = true;
  bounds.floor_z = 1.0;
  bounds.validate(true);
  for (double height : {0.1, 0.3, 0.5, 0.7, 0.9}) {
    Eigen::Matrix4d correction = pose(0.4, -0.2, 1.0, 0.2);
    correction.block<3, 3>(0, 0) =
      Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::Matrix4d body = pose(0.2, 0.1, 0.0, 0.1);
    body(2, 3) = (height - correction(2, 0) * body(0, 3)) / correction(2, 2);
    const Eigen::Matrix4d original = correction;
    correction(2, 3) += bounds.adjustment(correction, body);
    EXPECT_NEAR((correction * body)(2, 3) - bounds.floor_z,
      std::clamp(height, 0.3, 0.7), 1e-12);
    correction(2, 3) = original(2, 3);
    EXPECT_TRUE(correction.isApprox(original));
  }
  bounds.enabled = false;
  EXPECT_DOUBLE_EQ(bounds.adjustment(pose(0, 0, 100, 0), Eigen::Matrix4d::Identity()), 0.0);
}

TEST(HeightBounds, RejectsInvalidConfiguration)
{
  open3d_loc::HeightBounds bounds;
  bounds.enabled = true;
  EXPECT_THROW(bounds.validate(false), std::invalid_argument);
  EXPECT_NO_THROW(bounds.validate(true));
  bounds.min_height = 0.8;
  EXPECT_THROW(bounds.validate(true), std::invalid_argument);
  bounds.min_height = 0.3;
  bounds.floor_z = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(bounds.validate(true), std::invalid_argument);
  bounds.floor_z = 0.0;
  bounds.max_height = std::numeric_limits<double>::infinity();
  EXPECT_THROW(bounds.validate(true), std::invalid_argument);
}

TEST(GlobalCorrectionFilter, HeightProjectionPersistsAcrossUpdatesAndRecovery)
{
  open3d_loc::GlobalCorrectionFilterConfig config;
  config.recovery.enabled = true;
  config.update_mask = {{true, true, false, false, false, true}};
  open3d_loc::GlobalCorrectionFilter filter(config);
  open3d_loc::HeightBounds bounds;
  bounds.enabled = true;
  filter.reset(pose(0, 0, 0.5, 0));
  const auto covariance = filter.covariance();
  const auto candidate = pose(2.0, 0, 0, 0);
  filter.observeRecoveryCandidate(candidate, open3d_loc::Matrix6d::Identity() * 0.01, 1, 0, 0);
  const int candidate_count = filter.recoveryCandidateCount();
  const auto mode = filter.recoveryMode();
  filter.adjustHeight(-0.2);
  EXPECT_TRUE(filter.covariance().isApprox(covariance));
  EXPECT_EQ(filter.recoveryCandidateCount(), candidate_count);
  EXPECT_EQ(filter.recoveryMode(), mode);
  EXPECT_NEAR(filter.pose()(2, 3), 0.3, 1e-12);
  for (int i = 0; i < 3; ++i) {
    filter.predict(0.1, 0.0, 0.0);
    EXPECT_TRUE(filter.update(pose(0, 0, 2, 0),
      open3d_loc::Matrix6d::Identity() * 0.01).accepted);
    EXPECT_NEAR(filter.pose()(2, 3), 0.3, 1e-12);
  }
  filter.observeRecoveryCandidate(candidate, open3d_loc::Matrix6d::Identity() * 0.01, 3, 0, 0);
  const auto recovery = filter.observeRecoveryCandidate(
    candidate, open3d_loc::Matrix6d::Identity() * 0.01, 4, 0, 0);
  EXPECT_TRUE(recovery.correction_applied);
  filter.adjustHeight(bounds.adjustment(filter.pose(), Eigen::Matrix4d::Identity()));
  EXPECT_NEAR(filter.pose()(2, 3), 0.3, 1e-12);
}

TEST(GlobalCorrectionFilter, AllAxesUpdatesAndRecoveryRespectHeightBounds)
{
  open3d_loc::GlobalCorrectionFilterConfig config;
  config.update_mask.fill(true);
  config.recovery.enabled = true;
  open3d_loc::GlobalCorrectionFilter filter(config);
  open3d_loc::HeightBounds bounds;
  bounds.enabled = true;
  bounds.validate(true);
  filter.reset(pose(0, 0, 0.5, 0));
  const Eigen::Matrix4d body = pose(0.2, -0.1, 0.0, 0.0);
  Eigen::Matrix4d measurement = pose(0.1, 0.1, 0.6, 0.1);
  measurement.block<3, 3>(0, 0) = (
    Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(0.04, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(0.03, Eigen::Vector3d::UnitX())).toRotationMatrix();
  EXPECT_TRUE(filter.update(measurement, open3d_loc::Matrix6d::Identity() * 0.01).accepted);
  EXPECT_GT(filter.pose()(2, 3), 0.5);
  EXPECT_GT(filter.pose()(2, 1), 0.0);  // Roll updated.
  EXPECT_LT(filter.pose()(2, 0), 0.0);  // Pitch updated.
  filter.adjustHeight(bounds.adjustment(filter.pose(), body));
  EXPECT_GT((filter.pose() * body)(2, 3), 0.3);
  EXPECT_LT((filter.pose() * body)(2, 3), 0.7);

  for (double z : {3.0, -3.0}) {
    measurement(2, 3) = z;
    filter.clearRecovery();
    filter.observeRecoveryCandidate(measurement,
      open3d_loc::Matrix6d::Identity() * 0.01, 1.0, 0.0, 0.0);
    const auto recovery = filter.observeRecoveryCandidate(measurement,
      open3d_loc::Matrix6d::Identity() * 0.01, 2.0, 0.0, 0.0);
    ASSERT_TRUE(recovery.correction_applied);
    const Eigen::Matrix4d before = filter.pose();
    const auto covariance = filter.covariance();
    filter.adjustHeight(bounds.adjustment(filter.pose(), body));
    EXPECT_NEAR((filter.pose() * body)(2, 3), z > 0 ? 0.7 : 0.3, 1e-12);
    EXPECT_TRUE((filter.pose().block<3, 3>(0, 0).isApprox(before.block<3, 3>(0, 0))));
    EXPECT_DOUBLE_EQ(filter.pose()(0, 3), before(0, 3));
    EXPECT_DOUBLE_EQ(filter.pose()(1, 3), before(1, 3));
    EXPECT_TRUE(filter.covariance().isApprox(covariance));
    filter.predict(0.1, 0.0, 0.0);
    EXPECT_NEAR((filter.pose() * body)(2, 3), z > 0 ? 0.7 : 0.3, 1e-12);
  }
}

TEST(GlobalCorrectionFilter, NormalAllAxesCorrectionsAreProjectedAtBothHeightLimits)
{
  open3d_loc::GlobalCorrectionFilterConfig config;
  config.update_mask.fill(true);
  config.initial_stddev.setOnes();
  open3d_loc::GlobalCorrectionFilter filter(config);
  open3d_loc::HeightBounds bounds;
  bounds.enabled = true;
  const Eigen::Matrix4d body = pose(0.2, -0.1, 0.0, 0.0);
  for (double height : {1.1, -0.1}) {
    filter.reset(pose(0.0, 0.0, 0.5, 0.0));
    Eigen::Matrix4d measurement = pose(0.0, 0.0, height, 0.05);
    measurement.block<3, 3>(0, 0) =
      Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitY()).toRotationMatrix();
    ASSERT_TRUE(filter.update(
      measurement, open3d_loc::Matrix6d::Identity() * 0.001).accepted);
    const Eigen::Matrix4d unbounded = filter.pose();
    const auto covariance = filter.covariance();
    const double delta = bounds.adjustment(unbounded, body);
    ASSERT_GT(std::abs(delta), 0.0);
    filter.adjustHeight(delta);
    EXPECT_NEAR((filter.pose() * body)(2, 3), height > 0 ? 0.7 : 0.3, 1e-12);
    EXPECT_TRUE((filter.pose().block<3, 3>(0, 0).isApprox(unbounded.block<3, 3>(0, 0))));
    EXPECT_DOUBLE_EQ(filter.pose()(0, 3), unbounded(0, 3));
    EXPECT_DOUBLE_EQ(filter.pose()(1, 3), unbounded(1, 3));
    EXPECT_TRUE(filter.covariance().isApprox(covariance));
    filter.predict(0.1, 0.0, 0.0);
    EXPECT_NEAR((filter.pose() * body)(2, 3), height > 0 ? 0.7 : 0.3, 1e-12);
  }
}
