#include <gtest/gtest.h>

#include <limits>

#include "time_utils.h"

TEST(TimeUtils, ConvertsAndNormalizesValidTimestamps)
{
  builtin_interfaces::msg::Time time;

  ASSERT_TRUE(try_get_ros_time(1786956062.010, time));
  EXPECT_EQ(time.sec, 1786956062);
  EXPECT_NEAR(time.nanosec, 10000000U, 128U);

  ASSERT_TRUE(try_get_ros_time(1.9999999996, time));
  EXPECT_EQ(time.sec, 2);
  EXPECT_EQ(time.nanosec, 0U);

  ASSERT_TRUE(try_get_ros_time(static_cast<double>(std::numeric_limits<int32_t>::max()) + 0.5, time));
  EXPECT_EQ(time.sec, std::numeric_limits<int32_t>::max());
  EXPECT_EQ(time.nanosec, 500000000U);
}

TEST(TimeUtils, RejectsInvalidOrOutOfRangeTimestamps)
{
  builtin_interfaces::msg::Time time;
  EXPECT_FALSE(try_get_ros_time(-1.0, time));
  EXPECT_FALSE(try_get_ros_time(std::numeric_limits<double>::quiet_NaN(), time));
  EXPECT_FALSE(try_get_ros_time(std::numeric_limits<double>::infinity(), time));
  EXPECT_FALSE(try_get_ros_time(static_cast<double>(std::numeric_limits<int32_t>::max()) + 1.0, time));
}

TEST(TimeUtils, AppliesFiniteNonnegativeOffsets)
{
  double adjusted_timestamp = 0.0;

  ASSERT_TRUE(try_apply_time_offset(1787123492.0, -0.35, adjusted_timestamp));
  EXPECT_NEAR(adjusted_timestamp, 1787123491.65, 1e-6);

  EXPECT_FALSE(try_apply_time_offset(0.1, -0.2, adjusted_timestamp));
  EXPECT_FALSE(try_apply_time_offset(1.0, std::numeric_limits<double>::quiet_NaN(), adjusted_timestamp));
}
