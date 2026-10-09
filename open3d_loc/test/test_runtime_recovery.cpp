#include <gtest/gtest.h>
#include "open3d_loc/runtime_recovery.hpp"

namespace
{
using open3d_loc::RuntimeRecovery;
void feed(RuntimeRecovery & p, double start, double end, double linear = 0.0)
{
  for (double t = start; t <= end + 1e-8; t += 0.01) {
    auto stamp = static_cast<std::int64_t>((t + 100.0) * 1e9);
    p.command(t, stamp, linear, 0.0, true);
    p.imu(t, stamp, {0.0, 0.0, 9.81}, 0.0);
  }
}
RuntimeRecovery armed()
{
  RuntimeRecovery p;
  p.enabled = true; p.operator_reset(); p.ready(0.0);
  return p;
}
TEST(RuntimeRecovery, RequiresReadinessAndEligibleFault)
{
  RuntimeRecovery p; p.enabled = true;
  EXPECT_FALSE(p.lost("prediction_timeout", 0.1, 1.0));
  p.ready(1.0);
  EXPECT_FALSE(p.lost("implausible_state", 0.1, 2.0));
  EXPECT_TRUE(p.blocked);
}
TEST(RuntimeRecovery, MissingOrMovingInputsCannotReset)
{
  auto p = armed(); EXPECT_TRUE(p.lost("prediction_timeout", 0.1, 0.1));
  EXPECT_FALSE(p.take_attempt(0.1));
  feed(p, 0.2, 1.3, 0.1);
  EXPECT_FALSE(p.take_attempt(1.3));
  feed(p, 1.31, 2.4);
  EXPECT_TRUE(p.take_attempt(2.4)); EXPECT_EQ(p.attempts, 1);
}
TEST(RuntimeRecovery, SettlingIncludesBothInputs)
{
  auto p = armed(); p.lost("prediction_timeout", 0.0, 0.1);
  feed(p, 0.1, 0.6); EXPECT_FALSE(p.take_attempt(0.6));
  feed(p, 0.61, 1.2); EXPECT_TRUE(p.take_attempt(1.2));
  p.ready(1.21); EXPECT_FALSE(p.active); EXPECT_EQ(p.phase, "recovered");
}
TEST(RuntimeRecovery, StaleAnchorNeedsOperator)
{
  auto p = armed(); p.lost("prediction_timeout", 3.1, 0.1);
  EXPECT_FALSE(p.take_attempt(0.1)); EXPECT_EQ(p.blocker, "anchor_too_old");
}
TEST(RuntimeRecovery, StaleInputsAbortAttempt)
{
  auto p = armed(); feed(p, 0.1, 1.2);
  p.lost("prediction_timeout", 0.0, 1.2); ASSERT_TRUE(p.take_attempt(1.2));
  EXPECT_FALSE(p.take_attempt(1.5)); EXPECT_TRUE(p.blocked);
  p.ready(1.6); EXPECT_TRUE(p.blocked); // Late registration cannot clear failure.
}
TEST(RuntimeRecovery, RetryBudgetSurvivesReadinessUntilCooldown)
{
  auto p = armed(); p.max_attempts = 1; feed(p, 0.1, 1.2);
  p.lost("prediction_timeout", 0.0, 1.2); ASSERT_TRUE(p.take_attempt(1.2));
  p.ready(1.2); p.ready(5.0); EXPECT_EQ(p.attempts, 1);
  p.unavailable(); p.ready(6.0); p.ready(36.1); EXPECT_EQ(p.attempts, 0);
}
TEST(RuntimeRecovery, PersistentObstructionExhaustsAttempts)
{
  auto p = armed(); p.max_attempts = 2; p.retry_backoff = 0.1;
  feed(p, 0.1, 1.2); p.lost("prediction_timeout", 0.0, 1.2);
  ASSERT_TRUE(p.take_attempt(1.2)); p.attempt_lost("initialization_timeout", 1.21);
  feed(p, 1.21, 1.4); ASSERT_TRUE(p.take_attempt(1.4));
  p.attempt_lost("initialization_timeout", 1.41);
  EXPECT_FALSE(p.take_attempt(1.42)); EXPECT_TRUE(p.blocked); EXPECT_EQ(p.attempts, 2);
  p.operator_reset(); EXPECT_FALSE(p.blocked); EXPECT_EQ(p.attempts, 0);
}
TEST(RuntimeRecovery, AttemptDeadlineRetriesWithBackoff)
{
  auto p = armed(); p.attempt_timeout = 1.0; p.retry_backoff = 0.2;
  feed(p, 0.1, 1.2); p.lost("prediction_timeout", 0.0, 1.2);
  ASSERT_TRUE(p.take_attempt(1.2)); feed(p, 1.21, 2.3);
  EXPECT_FALSE(p.take_attempt(2.3)); feed(p, 2.31, 2.6);
  EXPECT_TRUE(p.take_attempt(2.6)); EXPECT_EQ(p.attempts, 2);
}
TEST(RuntimeRecovery, DuplicateStampAndMultiplePublishersBreakSettling)
{
  auto p = armed(); feed(p, 0.1, 1.2); ASSERT_TRUE(p.quiet(1.2));
  p.command(1.21, 1, 0.0, 0.0, false); EXPECT_FALSE(p.quiet(1.21));
  feed(p, 1.22, 1.5); EXPECT_FALSE(p.quiet(1.5));
}
TEST(RuntimeRecovery, ImuRotationGravityVariationAndGapsBlock)
{
  auto p = armed(); feed(p, 0.1, 1.2);
  p.imu(1.21, 102000000000LL, {0.0, 0.0, 9.81}, 1.0);
  EXPECT_FALSE(p.quiet(1.21));
  feed(p, 1.22, 2.4); ASSERT_TRUE(p.quiet(2.4));
  p.imu(2.41, 103000000000LL, {0.0, 0.0, 0.0}, 0.0);
  EXPECT_FALSE(p.quiet(2.41));
  feed(p, 2.42, 3.6); ASSERT_TRUE(p.quiet(3.6));
  p.imu(3.61, 104000000000LL, {0.0, 0.0, 9.81}, 0.0);
  EXPECT_FALSE(p.quiet(3.61));
}
TEST(RuntimeRecovery, AccelerationVariationMustSettleAgain)
{
  auto p = armed(); p.acceleration_rms = 0.01; feed(p, 0.1, 1.2);
  p.imu(1.21, 101210000000LL, {0.4, 0.0, 9.81}, 0.0);
  EXPECT_FALSE(p.quiet(1.21));
  feed(p, 1.22, 2.4); EXPECT_TRUE(p.quiet(2.4));
}
TEST(RuntimeRecovery, NewEpisodeRequiresFreshAnchorAndWaitsForStop)
{
  auto p = armed(); feed(p, 0.1, 1.2);
  p.lost("prediction_timeout", 0.0, 1.2); ASSERT_TRUE(p.take_attempt(1.2));
  p.ready(1.21);
  ASSERT_TRUE(p.lost("prediction_timeout", 0.0, 1.3));
  feed(p, 1.3, 1.5, 0.1);
  EXPECT_FALSE(p.take_attempt(1.5)); EXPECT_FALSE(p.blocked);
  EXPECT_FALSE(p.take_attempt(5.0)); EXPECT_EQ(p.blocker, "anchor_too_old");
}
TEST(RuntimeRecovery, MotionDuringBackoffRequiresOperator)
{
  auto p = armed(); feed(p, 0.1, 1.2);
  p.lost("prediction_timeout", 0.0, 1.2); ASSERT_TRUE(p.take_attempt(1.2));
  p.attempt_lost("initialization_timeout", 1.21);
  feed(p, 1.21, 1.3, 0.1);
  EXPECT_FALSE(p.take_attempt(1.3)); EXPECT_TRUE(p.blocked);
}
TEST(RuntimeRecovery, ExactDuplicateImuStampClearsWindow)
{
  auto p = armed(); feed(p, 0.1, 1.2);
  p.imu(1.21, 101210000000LL, {0.0, 0.0, 9.81}, 0.0);
  p.imu(1.22, 101210000000LL, {0.0, 0.0, 9.81}, 0.0);
  EXPECT_FALSE(p.quiet(1.22));
}
TEST(RuntimeRecovery, StopWindowBeginsWithFirstValidZeroCommand)
{
  auto p = armed(); p.command(0.1, 100100000000LL, 0.1, 0.0, true);
  feed(p, 0.29, 1.1);
  EXPECT_FALSE(p.quiet(1.1)); // 0.81 s of zero commands, not 1 s.
  feed(p, 1.11, 1.4); EXPECT_TRUE(p.quiet(1.4));
}
TEST(RuntimeRecovery, CommandGapCannotHideBetweenTimerTicks)
{
  auto p = armed(); feed(p, 0.1, 1.2);
  p.lost("prediction_timeout", 0.0, 1.2); ASSERT_TRUE(p.take_attempt(1.2));
  for (double t = 1.21; t < 1.5; t += 0.01)
    p.imu(t, static_cast<std::int64_t>((t+100)*1e9), {0.0, 0.0, 9.81}, 0.0);
  p.command(1.5, 101500000000LL, 0.0, 0.0, true);
  EXPECT_FALSE(p.take_attempt(1.5)); EXPECT_TRUE(p.blocked);
}
TEST(RuntimeRecovery, ImuGapCannotHideBehindFreshResumedSample)
{
  auto p = armed(); feed(p, 0.1, 1.2);
  p.lost("prediction_timeout", 0.0, 1.2); ASSERT_TRUE(p.take_attempt(1.2));
  p.command(1.26, 101260000000LL, 0.0, 0.0, true);
  p.imu(1.26, 101260000000LL, {0.0, 0.0, 9.81}, 0.0);
  EXPECT_FALSE(p.take_attempt(1.26)); EXPECT_TRUE(p.blocked);
}
TEST(RuntimeRecovery, ValidatesParameters)
{
  RuntimeRecovery p; p.max_attempts = 0; EXPECT_THROW(p.validate(), std::invalid_argument);
  p.max_attempts = 3; p.settle = NAN; EXPECT_THROW(p.validate(), std::invalid_argument);
}
}  // namespace
