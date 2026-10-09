#include <gtest/gtest.h>
#include "tracking_guard.hpp"

TEST(TrackingGuard, RequiresThreeScansAndLatchesLoss)
{
  fast_lio::TrackingGuard guard;
  EXPECT_FALSE(guard.accept(1.0));
  EXPECT_FALSE(guard.accept(1.1));
  EXPECT_TRUE(guard.accept(1.2));
  guard.reject(1.3, "insufficient_features");
  EXPECT_EQ(guard.state, fast_lio::TrackingGuard::DEGRADED);
  EXPECT_FALSE(guard.accept(1.4));
  EXPECT_FALSE(guard.accept(1.5));
  EXPECT_TRUE(guard.accept(1.6));
  guard.reject(2.2, "insufficient_features");
  EXPECT_EQ(guard.state, fast_lio::TrackingGuard::LOST);
  EXPECT_FALSE(guard.accept(2.3));
  guard.reset();
  EXPECT_EQ(guard.state, fast_lio::TrackingGuard::INITIALIZING);
  EXPECT_DOUBLE_EQ(guard.accepted_stamp, -1.0);
}

TEST(TrackingGuard, RejectsWeakMeasurements)
{
  fast_lio::TrackingLimits limits;
  EXPECT_EQ(fast_lio::measurementRejection(limits, 6, 100, 0, 1), "insufficient_features");
  EXPECT_EQ(fast_lio::measurementRejection(limits, 100, 100, 0.2, 1), "residual_gate");
  EXPECT_EQ(fast_lio::measurementRejection(limits, 100, 100, 0, 0), "degenerate_geometry");
  EXPECT_TRUE(fast_lio::measurementRejection(limits, 100, 100, 0.01, 0.1).empty());
}

TEST(TrackingGuard, DetectsNullDirectionsAndNormalizesRotation)
{
  Eigen::MatrixXd jacobian = Eigen::Matrix<double, 6, 6>::Identity();
  jacobian.rightCols(3) *= 10;
  EXPECT_NEAR(fast_lio::informationRatio(jacobian, 10), 1, 1e-12);
  jacobian.col(0).setZero();
  EXPECT_DOUBLE_EQ(fast_lio::informationRatio(jacobian, 10), 0);
  jacobian = Eigen::Matrix<double, 6, 6>::Identity() * 1e200;
  EXPECT_DOUBLE_EQ(fast_lio::informationRatio(jacobian, 1), 0);
}

TEST(TrackingGuard, RejectsExtremeAndNonfiniteMotion)
{
  fast_lio::TrackingLimits limits;
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
  EXPECT_TRUE(fast_lio::plausibleMotion(limits, zero, identity, zero, identity, zero, .1));
  EXPECT_FALSE(fast_lio::plausibleMotion(limits, zero, identity,
    Eigen::Vector3d(-10000, -20000, -100000), identity, zero, .1));
  EXPECT_FALSE(fast_lio::plausibleMotion(limits, zero, identity, zero, identity,
    Eigen::Vector3d(0, 0, -1800), .1));
  EXPECT_FALSE(fast_lio::plausibleMotion(limits, zero, identity, zero, identity, zero, 0));
  limits.max_speed = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(limits.validate(), std::invalid_argument);
}

TEST(TrackingGuard, BoundsPredictionBeforeFirstAcceptedUpdate)
{
  fast_lio::TrackingGuard guard;
  guard.prediction_started = 1.0;
  EXPECT_TRUE(guard.allowPrediction(1.4));
  EXPECT_FALSE(guard.allowPrediction(1.5));
  EXPECT_EQ(guard.reason, "initialization_timeout");
  guard.reset();
  EXPECT_DOUBLE_EQ(guard.prediction_started, -1.0);
  EXPECT_FALSE(guard.accept(2.0));
  EXPECT_FALSE(guard.accept(2.1));
  EXPECT_TRUE(guard.accept(2.2));
  EXPECT_TRUE(guard.allowPrediction(2.4));
  EXPECT_FALSE(guard.allowPrediction(2.8));
  EXPECT_EQ(guard.reason, "prediction_timeout");
}

TEST(TrackingGuard, PreservesFirstFailureUntilReset)
{
  fast_lio::TrackingGuard guard;
  guard.lose("imu_time_gap");
  guard.lose("prediction_timeout");
  EXPECT_EQ(guard.reason, "imu_time_gap");
  guard.reset();
  guard.lose("timestamp_regression");
  EXPECT_EQ(guard.reason, "timestamp_regression");
}
