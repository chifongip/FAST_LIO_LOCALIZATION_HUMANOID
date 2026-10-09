#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include "registration_geometry.hpp"

TEST(RegistrationGeometry, ConditioningDoesNotDependOnMapOrOdometryOrigin)
{
  open3d::geometry::PointCloud source, target;
  open3d::pipelines::registration::RegistrationResult result;
  for (int axis = 0; axis < 3; ++axis) {
    for (int a = -2; a <= 2; ++a) {
      for (int b = -2; b <= 2; ++b) {
        Eigen::Vector3d point = Eigen::Vector3d::Zero();
        point(axis) = 3.0;
        point((axis + 1) % 3) = a;
        point((axis + 2) % 3) = b;
        result.correspondence_set_.emplace_back(source.points_.size(), target.points_.size());
        source.points_.push_back(point);
        target.points_.push_back(point);
        target.normals_.push_back(Eigen::Vector3d::Unit(axis));
      }
    }
  }
  const auto identity = Eigen::Matrix4d::Identity().eval();
  const double expected = open3d_loc::pointToPlaneInformationRatio(source, target, result, identity);
  ASSERT_GT(expected, 1e-4);
  Eigen::Matrix4d map_pose = identity;
  map_pose.block<3, 3>(0, 0) = Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  map_pose.block<3, 1>(0, 3) = Eigen::Vector3d(10000, -20000, 30000);
  target.Transform(map_pose);
  EXPECT_NEAR(open3d_loc::pointToPlaneInformationRatio(source, target, result, map_pose), expected, 1e-12);
  source.Transform(map_pose);
  EXPECT_NEAR(open3d_loc::pointToPlaneInformationRatio(source, target, result, identity), expected, 1e-10);
}

TEST(RegistrationGeometry, RejectsPlanarAndInvalidCorrespondences)
{
  open3d::geometry::PointCloud source, target;
  open3d::pipelines::registration::RegistrationResult result;
  for (int i = 0; i < 20; ++i) {
    source.points_.emplace_back(i % 5, i / 5, 0);
    target.points_.push_back(source.points_.back());
    target.normals_.push_back(Eigen::Vector3d::UnitZ());
    result.correspondence_set_.emplace_back(i, i);
  }
  EXPECT_DOUBLE_EQ(open3d_loc::pointToPlaneInformationRatio(
    source, target, result, Eigen::Matrix4d::Identity()), 0.0);
  result.correspondence_set_[0](0) = 100;
  EXPECT_DOUBLE_EQ(open3d_loc::pointToPlaneInformationRatio(
    source, target, result, Eigen::Matrix4d::Identity()), 0.0);
}
