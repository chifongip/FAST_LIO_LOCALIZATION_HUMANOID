#include <gtest/gtest.h>

#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "preprocess.h"

namespace
{
struct InputPoint
{
  float x;
  float y;
  float z;
  float intensity;
  std::uint16_t ring;
  double timestamp;
};

sensor_msgs::msg::PointCloud2::UniquePtr make_cloud(const std::vector<InputPoint> &points)
{
  auto message = std::make_unique<sensor_msgs::msg::PointCloud2>();
  message->height = 1;
  message->width = points.size();
  message->is_bigendian = false;
  message->is_dense = true;
  message->point_step = 32;
  message->row_step = message->point_step * message->width;
  message->fields = {
      sensor_msgs::msg::PointField().set__name("x").set__offset(0).set__datatype(sensor_msgs::msg::PointField::FLOAT32).set__count(1),
      sensor_msgs::msg::PointField().set__name("y").set__offset(4).set__datatype(sensor_msgs::msg::PointField::FLOAT32).set__count(1),
      sensor_msgs::msg::PointField().set__name("z").set__offset(8).set__datatype(sensor_msgs::msg::PointField::FLOAT32).set__count(1),
      sensor_msgs::msg::PointField().set__name("intensity").set__offset(16).set__datatype(sensor_msgs::msg::PointField::FLOAT32).set__count(1),
      sensor_msgs::msg::PointField().set__name("ring").set__offset(20).set__datatype(sensor_msgs::msg::PointField::UINT16).set__count(1),
      sensor_msgs::msg::PointField().set__name("timestamp").set__offset(24).set__datatype(sensor_msgs::msg::PointField::FLOAT64).set__count(1),
  };
  message->data.resize(message->row_step);

  for (std::size_t index = 0; index < points.size(); ++index)
  {
    std::uint8_t *data = message->data.data() + index * message->point_step;
    std::memcpy(data + 0, &points[index].x, sizeof(float));
    std::memcpy(data + 4, &points[index].y, sizeof(float));
    std::memcpy(data + 8, &points[index].z, sizeof(float));
    std::memcpy(data + 16, &points[index].intensity, sizeof(float));
    std::memcpy(data + 20, &points[index].ring, sizeof(std::uint16_t));
    std::memcpy(data + 24, &points[index].timestamp, sizeof(double));
  }
  return message;
}
} // namespace

TEST(E1RPreprocess, DecodesAndOrdersNanosecondOffsets)
{
  Preprocess preprocess;
  preprocess.set(false, ROBOSENSE_E1R, 0.5, 1);
  preprocess.N_SCANS = 96;
  preprocess.time_unit = NS;

  auto message = make_cloud({
      {2.0f, 0.0f, 0.0f, 10.0f, 1, 95000000.0},
      {0.1f, 0.0f, 0.0f, 20.0f, 2, 5000000.0},
      {3.0f, 0.0f, 0.0f, 30.0f, 95, 1000000.0},
      {4.0f, 0.0f, 0.0f, 40.0f, 96, 25000000.0},
      {5.0f, 0.0f, 0.0f, 50.0f, 2, 50000000.0},
      {6.0f, 0.0f, 0.0f, 60.0f, 3, std::numeric_limits<double>::quiet_NaN()},
      {7.0f, 0.0f, 0.0f, 70.0f, 4, -1000000.0},
      {8.0f, 0.0f, 0.0f, 80.0f, 5, std::numeric_limits<double>::infinity()},
      {9.0f, 0.0f, 0.0f, 90.0f, 6, 250000000.0},
  });
  auto output = std::make_shared<PointCloudXYZI>();

  EXPECT_TRUE(preprocess.process(message, output));

  ASSERT_EQ(output->size(), 3U);
  EXPECT_FLOAT_EQ(output->points[0].intensity, 30.0f);
  EXPECT_NEAR(output->points[0].curvature, 1.0, 1e-5);
  EXPECT_NEAR(output->points[1].curvature, 50.0, 1e-5);
  EXPECT_NEAR(output->points[2].curvature, 95.0, 1e-5);
}

TEST(E1RPreprocess, AppliesPointFilterAfterValidityChecks)
{
  Preprocess preprocess;
  preprocess.set(false, ROBOSENSE_E1R, 0.5, 2);
  preprocess.N_SCANS = 96;
  preprocess.time_unit = NS;

  auto message = make_cloud({
      {2.0f, 0.0f, 0.0f, 10.0f, 0, 1000000.0},
      {0.1f, 0.0f, 0.0f, 20.0f, 0, 2000000.0},
      {3.0f, 0.0f, 0.0f, 30.0f, 1, 3000000.0},
      {4.0f, 0.0f, 0.0f, 40.0f, 2, 4000000.0},
  });
  auto output = std::make_shared<PointCloudXYZI>();

  EXPECT_TRUE(preprocess.process(message, output));

  ASSERT_EQ(output->size(), 2U);
  EXPECT_FLOAT_EQ(output->points[0].intensity, 10.0f);
  EXPECT_FLOAT_EQ(output->points[1].intensity, 40.0f);
}

TEST(E1RPreprocess, RejectsCloudWhenAllPointTimesExceedScanLimit)
{
  Preprocess preprocess;
  preprocess.set(false, ROBOSENSE_E1R, 0.5, 1);
  preprocess.N_SCANS = 96;
  preprocess.SCAN_RATE = 10;
  preprocess.time_unit = NS;
  preprocess.max_scan_duration_ms = 200.0;

  auto message = make_cloud({{2.0f, 0.0f, 0.0f, 10.0f, 0, 250000000.0}});
  auto output = std::make_shared<PointCloudXYZI>();

  EXPECT_FALSE(preprocess.process(message, output));
  EXPECT_TRUE(output->empty());
}

TEST(E1RPreprocess, RejectsIncompatibleTimestampSchema)
{
  Preprocess preprocess;
  preprocess.set(false, ROBOSENSE_E1R, 0.5, 1);
  preprocess.N_SCANS = 96;
  preprocess.time_unit = NS;

  auto message = make_cloud({{2.0f, 0.0f, 0.0f, 10.0f, 0, 1000000.0}});
  for (auto &field : message->fields)
  {
    if (field.name == "timestamp")
      field.datatype = sensor_msgs::msg::PointField::UINT32;
  }
  auto output = std::make_shared<PointCloudXYZI>();

  EXPECT_FALSE(preprocess.process(message, output));
  EXPECT_TRUE(output->empty());
}
