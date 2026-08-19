#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Geometry>

#include "odometry_utils.hpp"

TEST(OdometryUtils, InterpolatesGyroAtTheLidarEndTime)
{
  Eigen::Vector3d gyro_at_lidar_end;
  EXPECT_TRUE(fast_lio::interpolateAngularVelocity(
      1.0, Eigen::Vector3d(0.0, 2.0, 4.0),
      3.0, Eigen::Vector3d(2.0, 4.0, 6.0),
      2.0, gyro_at_lidar_end));
  EXPECT_TRUE(gyro_at_lidar_end.isApprox(Eigen::Vector3d(1.0, 3.0, 5.0), 1e-12));

  EXPECT_FALSE(fast_lio::interpolateAngularVelocity(
      3.0, Eigen::Vector3d::Zero(),
      1.0, Eigen::Vector3d::Zero(),
      2.0, gyro_at_lidar_end));
}

TEST(OdometryUtils, ConvertsWorldVelocityAndCorrectsGyroBias)
{
  const Eigen::Matrix3d world_from_body =
      Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const fast_lio::BodyTwist twist = fast_lio::bodyTwistFromState(
      world_from_body,
      Eigen::Vector3d(0.0, 2.0, 0.0),
      Eigen::Vector3d(0.1, -0.2, 0.3),
      Eigen::Vector3d(0.01, 0.02, -0.03));

  EXPECT_TRUE(twist.linear.isApprox(Eigen::Vector3d(2.0, 0.0, 0.0), 1e-12));
  EXPECT_TRUE(twist.angular.isApprox(Eigen::Vector3d(0.09, -0.22, 0.33), 1e-12));
}

TEST(OdometryUtils, KeepsPoseAndTwistCovariancesInTheirStateBlocks)
{
  fast_lio::StateCovariance state_covariance =
      fast_lio::StateCovariance::Zero();
  state_covariance(0, 0) = 1.0;
  state_covariance(3, 3) = 4.0;
  state_covariance(12, 12) = 9.0;
  state_covariance(15, 15) = 0.25;
  state_covariance(12, 15) = 0.5;
  state_covariance(15, 12) = 0.5;

  const fast_lio::PoseCovariance pose_covariance =
      fast_lio::poseCovarianceFromState(state_covariance);
  const fast_lio::TwistCovariance twist_covariance =
      fast_lio::bodyTwistCovarianceFromState(
          state_covariance,
          Eigen::Matrix3d::Identity(),
          0.01 * Eigen::Matrix3d::Identity());

  EXPECT_DOUBLE_EQ(pose_covariance(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(pose_covariance(3, 3), 4.0);
  EXPECT_DOUBLE_EQ(twist_covariance(0, 0), 9.0);
  EXPECT_DOUBLE_EQ(twist_covariance(3, 3), 0.26);
  EXPECT_DOUBLE_EQ(twist_covariance(0, 3), -0.5);
  EXPECT_DOUBLE_EQ(twist_covariance(3, 0), -0.5);
}

TEST(OdometryUtils, RotatesLinearVelocityCovarianceIntoTheBodyFrame)
{
  fast_lio::StateCovariance state_covariance =
      fast_lio::StateCovariance::Zero();
  state_covariance.block<3, 3>(12, 12) =
      (Eigen::Vector3d(1.0, 4.0, 9.0)).asDiagonal();
  const Eigen::Matrix3d world_from_body =
      Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  const fast_lio::TwistCovariance twist_covariance =
      fast_lio::bodyTwistCovarianceFromState(
          state_covariance, world_from_body, Eigen::Matrix3d::Zero());

  EXPECT_DOUBLE_EQ(twist_covariance(0, 0), 4.0);
  EXPECT_DOUBLE_EQ(twist_covariance(1, 1), 1.0);
  EXPECT_DOUBLE_EQ(twist_covariance(2, 2), 9.0);
}
