#include <gtest/gtest.h>
#include <limits>
#include "open3d_loc/startup_recovery.hpp"

using open3d_loc::StartupRecovery;

TEST(StartupRecovery, RetriesOnlyAfterBackoffAndOncePerEpoch)
{
  StartupRecovery policy;
  policy.begin(0);
  policy.lost(10, 0, "imu_time_gap", 6);
  EXPECT_FALSE(policy.take_retry(6.99, false));
  EXPECT_FALSE(policy.take_retry(7, true));
  EXPECT_TRUE(policy.take_retry(7, false));
  EXPECT_EQ(policy.retries, 1u);
  policy.lost(10, 0, "imu_time_gap", 8);
  EXPECT_FALSE(policy.take_retry(10, false));
  policy.lost(10, 1, "missing_imu_coverage", 10);
  EXPECT_TRUE(policy.take_retry(11, false));
  EXPECT_EQ(policy.retries, 2u);
  policy.lost(10, 0, "imu_time_gap", 20);
  EXPECT_FALSE(policy.take_retry(100, false));
}

TEST(StartupRecovery, LocalTrackingDoesNotCloseAcquisition)
{
  StartupRecovery policy;
  policy.begin(0);
  policy.waiting("waiting_for_global_registration");
  EXPECT_TRUE(policy.acquiring);
  policy.waiting("waiting_for_timestamped_tf");
  EXPECT_FALSE(policy.ready);
  policy.lost(1, 0, "imu_time_gap", 2);
  policy.complete();
  EXPECT_FALSE(policy.ready);
  EXPECT_TRUE(policy.take_retry(3, false));
}

TEST(StartupRecovery, LossAfterReadinessNeverReopensAcquisition)
{
  StartupRecovery policy;
  policy.begin(0);
  policy.complete();
  EXPECT_TRUE(policy.ready);
  policy.lost(1, 0, "imu_time_gap", 2);
  EXPECT_FALSE(policy.ready);
  EXPECT_FALSE(policy.acquiring);
  EXPECT_EQ(policy.phase, "tracking_unavailable");
  EXPECT_FALSE(policy.take_retry(100, false));
  policy.begin(101);  // Explicit operator reset or backend restart.
  policy.lost(1, 1, "imu_time_gap", 102);
  EXPECT_TRUE(policy.take_retry(103, false));
}

TEST(StartupRecovery, HardFailuresAndResetErrorsRequireOperatorReset)
{
  for (const auto & reason : {"nonfinite_imu", "timestamp_regression", "imu_buffer_overflow",
    "backend_reset_timeout", "backend_reset_failed", "invalid_reset_acknowledgment"}) {
    StartupRecovery policy;
    policy.begin(0);
    policy.lost(1, 0, reason, 1);
    EXPECT_EQ(policy.phase, "fault");
    EXPECT_TRUE(policy.error(1));
    policy.lost(1, 1, "imu_time_gap", 2);
    EXPECT_FALSE(policy.take_retry(100, false));
    policy.complete();
    EXPECT_FALSE(policy.ready);
    policy.begin(101);
    policy.complete();
    EXPECT_TRUE(policy.ready);
  }
}

TEST(StartupRecovery, NewOperatorAcquisitionSupersedesPendingAutomaticRetry)
{
  StartupRecovery policy;
  policy.begin(0);
  policy.lost(1, 0, "imu_time_gap", 2);
  policy.begin(2.5);
  EXPECT_FALSE(policy.take_retry(4, false));
  EXPECT_EQ(policy.retries, 0u);
  EXPECT_TRUE(policy.last_failure.empty());
}

TEST(StartupRecovery, WarningDeadlineDoesNotStopRetries)
{
  StartupRecovery policy;
  policy.begin(5);
  EXPECT_FALSE(policy.error(14.99));
  EXPECT_TRUE(policy.error(15));
  policy.lost(1, 0, "imu_time_gap", 16);
  EXPECT_TRUE(policy.take_retry(17, false));
  policy.complete();
  EXPECT_FALSE(policy.error(100));
}

TEST(StartupRecovery, ValidatesTimesAndSupportsDisabledRecovery)
{
  StartupRecovery policy;
  EXPECT_NO_THROW(policy.validate());
  policy.retry_backoff = 0;
  EXPECT_THROW(policy.validate(), std::invalid_argument);
  policy.retry_backoff = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(policy.validate(), std::invalid_argument);
  policy.retry_backoff = 1;
  policy.warning_timeout = -1;
  EXPECT_THROW(policy.validate(), std::invalid_argument);
  policy.enabled = false;
  policy.begin(0);
  policy.lost(1, 0, "imu_time_gap", 1);
  EXPECT_FALSE(policy.take_retry(10, false));
}
