#include <gtest/gtest.h>

#include "open3d_loc/transform_utils.hpp"

TEST(TransformUtils, ArticulationIsRemovedFromDerivedBodyPose)
{
  Eigen::Matrix4d body_to_tracking = Eigen::Matrix4d::Identity();
  body_to_tracking(0, 3) = 1.0;
  Eigen::Matrix4d tracking_to_odom = Eigen::Matrix4d::Identity();
  tracking_to_odom(0, 3) = 2.0;

  const Eigen::Matrix4d first_body_pose =
      open3d_loc::deriveBodyPose(tracking_to_odom, body_to_tracking);

  body_to_tracking(0, 3) = 1.5;
  tracking_to_odom(0, 3) = 2.5;
  const Eigen::Matrix4d second_body_pose =
      open3d_loc::deriveBodyPose(tracking_to_odom, body_to_tracking);

  EXPECT_TRUE(first_body_pose.isApprox(second_body_pose, 1e-12));
  EXPECT_DOUBLE_EQ(first_body_pose(0, 3), 1.0);
}

TEST(TransformUtils, OutputPoseUsesCurrentBodyToOutputTransform)
{
  Eigen::Matrix4d body_pose = Eigen::Matrix4d::Identity();
  body_pose(0, 3) = 3.0;
  Eigen::Matrix4d body_to_torso = Eigen::Matrix4d::Identity();
  body_to_torso(2, 3) = 0.8;

  const Eigen::Matrix4d torso_pose =
      open3d_loc::deriveOutputPose(body_pose, body_to_torso);

  EXPECT_DOUBLE_EQ(torso_pose(0, 3), 3.0);
  EXPECT_DOUBLE_EQ(torso_pose(2, 3), 0.8);
}

TEST(TransformUtils, PlanarInitialPoseIsLevel)
{
  const Eigen::Matrix3d orientation = open3d_loc::levelOrientationFromYaw(0.8);
  const Eigen::Vector3d euler = orientation.eulerAngles(0, 1, 2);

  EXPECT_NEAR(euler.x(), 0.0, 1e-12);
  EXPECT_NEAR(euler.y(), 0.0, 1e-12);
  EXPECT_NEAR(euler.z(), 0.8, 1e-12);
}
