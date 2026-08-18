#include <gtest/gtest.h>

#include "gravity_alignment.h"

TEST(GravityAlignment, MapsImuUpAxisToWorldUp)
{
  Eigen::Matrix3d world_from_imu;
  ASSERT_TRUE(fast_lio::gravityAlignedWorldFromImu(
      Eigen::Vector3d::UnitX(), world_from_imu));

  EXPECT_TRUE((world_from_imu * Eigen::Vector3d::UnitX()).isApprox(
      Eigen::Vector3d::UnitZ(), 1e-12));
  EXPECT_TRUE((world_from_imu.transpose() * world_from_imu).isApprox(
      Eigen::Matrix3d::Identity(), 1e-12));
  EXPECT_NEAR(world_from_imu.determinant(), 1.0, 1e-12);
}

TEST(GravityAlignment, HandlesTiltedStationaryImu)
{
  Eigen::Matrix3d world_from_imu;
  const Eigen::Vector3d measured_up(8.49, 0.11, 5.09);
  ASSERT_TRUE(fast_lio::gravityAlignedWorldFromImu(measured_up, world_from_imu));

  EXPECT_TRUE((world_from_imu * measured_up.normalized()).isApprox(
      Eigen::Vector3d::UnitZ(), 1e-12));
}

TEST(GravityAlignment, RejectsInvalidAcceleration)
{
  Eigen::Matrix3d world_from_imu;
  EXPECT_FALSE(fast_lio::gravityAlignedWorldFromImu(
      Eigen::Vector3d::Zero(), world_from_imu));
  EXPECT_TRUE(world_from_imu.isApprox(Eigen::Matrix3d::Identity(), 1e-12));
}
