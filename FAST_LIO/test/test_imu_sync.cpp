#include <gtest/gtest.h>
#include <limits>
#include "imu_sync.hpp"

using fast_lio::ImuSyncAction;
using fast_lio::imuCoverage;

TEST(ImuSync, WaitsForFollowingSampleWithoutTreatingArrivalOrderAsGap)
{
  auto result = imuCoverage({1.0, 1.005}, 1.0, 1.009, -1, .05, .01, .1, false);
  EXPECT_EQ(result.action, ImuSyncAction::WAITING);
  result = imuCoverage({1.0, 1.005, 1.01}, 1.0, 1.009, -1, .05, .02, .1, false);
  EXPECT_EQ(result.action, ImuSyncAction::READY);
  EXPECT_EQ(result.count, 2u);
  EXPECT_DOUBLE_EQ(result.following, 1.01);
}

TEST(ImuSync, ArrivalTimeoutIncludesEmptyQueue)
{
  for (const auto & stamps : {std::vector<double>{}, std::vector<double>{1.0}}) {
    const auto result = imuCoverage(stamps, 1.0, 1.01, -1, .05, .11, .1, false);
    EXPECT_EQ(result.action, ImuSyncAction::INVALID);
    EXPECT_EQ(result.reason, "imu_arrival_timeout");
  }
}

TEST(ImuSync, ChecksRetainedBoundaryAndInternalGap)
{
  EXPECT_EQ(imuCoverage({1.10, 1.105}, 1.09, 1.102, 1.0, .05, 0, .1, false).reason,
    "imu_time_gap");
  EXPECT_EQ(imuCoverage({1.0, 1.10}, 1.0, 1.09, -1, .05, 0, .1, false).reason,
    "imu_scan_end_gap");
}

TEST(ImuSync, CoveredEmptyIntervalSkipsWithoutAdvancingEstimator)
{
  EXPECT_EQ(imuCoverage({1.01}, 1.001, 1.004, 1.0, .05, 0, .1, false).action,
    ImuSyncAction::SKIP_SCAN);
  EXPECT_EQ(imuCoverage({1.10}, 1.001, 1.004, 1.0, .05, 0, .1, false).action,
    ImuSyncAction::SKIP_SCAN);
  EXPECT_EQ(imuCoverage({1.01}, 1.001, 1.004, -1, .05, 0, .1, false).action,
    ImuSyncAction::INVALID);
}

TEST(ImuSync, WarmupRequiresRecentContinuousWindow)
{
  std::vector<double> stamps{0.0, 0.1};
  for (int i = 0; i <= 50; ++i) stamps.push_back(1.0 + i * .005);
  EXPECT_EQ(imuCoverage(stamps, 1.0, 1.1, -1, .05, 0, .1, true).action,
    ImuSyncAction::SKIP_SCAN);
  const auto result = imuCoverage(stamps, 1.15, 1.24, -1, .05, 0, .1, true);
  EXPECT_EQ(result.action, ImuSyncAction::READY);
  EXPECT_DOUBLE_EQ(result.first, 1.0);
}

TEST(ImuSync, RejectsRegressionDuplicatesAndNonfiniteStamps)
{
  for (const auto & stamps : {std::vector<double>{1, 1}, std::vector<double>{1, .9},
    std::vector<double>{1, std::numeric_limits<double>::quiet_NaN()}}) {
    EXPECT_EQ(imuCoverage(stamps, 1, 1, -1, .05, 0, .1, true).action,
      ImuSyncAction::INVALID);
  }
  EXPECT_EQ(imuCoverage({1, 1.005}, 1, 1.005, 1, .05, 0, .1, false).action,
    ImuSyncAction::INVALID);
}

TEST(ImuSync, ExactEndDoesNotRequireFutureCoverage)
{
  EXPECT_EQ(imuCoverage({1, 1.005, 2}, 1, 1.005, -1, .05, 0, .1, false).action,
    ImuSyncAction::READY);
}

TEST(ImuSync, DoesNotRejectCoveredScanForFutureGap)
{
  const auto result = imuCoverage({1.005, 1.01, 1.1}, 1, 1.012, 1, .05, 0, .1, false);
  EXPECT_EQ(result.action, ImuSyncAction::READY);
  EXPECT_DOUBLE_EQ(result.following, 1.1);
  EXPECT_EQ(imuCoverage({1.1, 1.105}, 1.012, 1.102, 1.01, .05, 0, .1, false).reason,
    "imu_time_gap");
}

TEST(ImuSync, RecordsExactOffendingPairAndBoundary)
{
  const auto result = imuCoverage({1.005, 1.059287, 1.065, 1.11},
    1, 1.1, 1, .05, 0, .1, false);
  EXPECT_EQ(result.reason, "imu_time_gap");
  EXPECT_DOUBLE_EQ(result.gap_before, 1.005);
  EXPECT_DOUBLE_EQ(result.gap_after, 1.059287);
  EXPECT_NEAR(result.gap, .054287, 1e-12);
  EXPECT_DOUBLE_EQ(result.integration_boundary, 1);
}

TEST(ImuSync, HandlesUnixEpochRoundingAtGapLimit)
{
  const double start = 1791511234.05;
  EXPECT_FALSE(fast_lio::imuGapExceeded((start + .05) - start, .05));
  EXPECT_TRUE(fast_lio::imuGapExceeded((start + .05001) - start, .05));
  EXPECT_EQ(imuCoverage({start, start + .05}, start, start + .05,
    -1, .05, 0, .1, false).action, ImuSyncAction::READY);
}
