#include <gtest/gtest.h>
#include <limits>

#include "IMU_Processing.hpp"

namespace
{
using Filter = esekfom::esekf<state_ikfom, 12, input_ikfom>;
int iteration = 0;
int invalid_iteration = -99;

void measurement(state_ikfom &, esekfom::dyn_share_datastruct<double> &data)
{
  data.valid = iteration++ != invalid_iteration;
  if (!data.valid) return;
  data.h_x = Eigen::MatrixXd::Identity(12, 12);
  data.h = Eigen::VectorXd::Constant(12, 0.01);
}

void initialize(Filter &filter)
{
  double limits[23];
  std::fill(limits, limits + 23, 0.001);
  filter.init_dyn_share(get_f, df_dx, df_dw, measurement, 1, limits);
  iteration = 0;
}

MeasureGroup scan(double begin)
{
  MeasureGroup data;
  data.lidar_beg_time = begin;
  data.lidar_end_time = begin + 0.1;
  data.lidar.reset(new PointCloudXYZI());
  PointType point;
  point.x = 1.0;
  point.y = point.z = 0.0;
  point.curvature = 100.0;
  data.lidar->push_back(point);
  for (int i = 0; i <= 20; ++i)
  {
    auto imu = std::make_shared<sensor_msgs::msg::Imu>();
    imu->header.stamp = rclcpp::Time(static_cast<int64_t>((begin + i * 0.005) * 1e9));
    imu->linear_acceleration.z = G_m_s2;
    data.imu.push_back(imu);
  }
  return data;
}
}  // namespace

TEST(ScanUpdate, InvalidFinalIterationRestoresPrediction)
{
  Filter filter;
  initialize(filter);
  auto state = filter.get_x();
  auto covariance = filter.get_P();
  invalid_iteration = 1;
  double solve_time = 0.0;
  EXPECT_FALSE(filter.update_iterated_dyn_share_modified(0.001, solve_time));
  EXPECT_TRUE(filter.get_x().pos.isApprox(state.pos));
  EXPECT_TRUE(filter.get_x().rot.toRotationMatrix().isApprox(state.rot.toRotationMatrix()));
  EXPECT_TRUE(filter.get_P().isApprox(covariance));
}

TEST(ScanUpdate, InvalidIterationCanRecover)
{
  Filter filter;
  initialize(filter);
  invalid_iteration = 0;
  double solve_time = 0.0;
  EXPECT_TRUE(filter.update_iterated_dyn_share_modified(0.001, solve_time));
  EXPECT_TRUE(filter.get_P().allFinite());
}

TEST(ScanUpdate, MissingImuClearsCandidateCloud)
{
  Filter filter;
  ImuProcess processor;
  auto data = scan(1.0);
  data.imu.clear();
  PointCloudXYZI::Ptr cloud(new PointCloudXYZI(*data.lidar));
  const auto result = processor.Process(data, filter, cloud);
  EXPECT_EQ(result.status, ImuProcess::ProcessStatus::Rejected);
  EXPECT_TRUE(cloud->empty());
}

TEST(ScanUpdate, EmptyLidarIsRejected)
{
  Filter filter;
  ImuProcess processor;
  auto data = scan(1.0);
  data.lidar->clear();
  PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
  EXPECT_EQ(processor.Process(data, filter, cloud).status,
    ImuProcess::ProcessStatus::Rejected);
}

TEST(ScanUpdate, BackwardImuIsRejected)
{
  Filter filter;
  ImuProcess processor;
  auto data = scan(1.0);
  std::swap(data.imu[1], data.imu[2]);
  PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
  EXPECT_EQ(processor.Process(data, filter, cloud).status,
    ImuProcess::ProcessStatus::Rejected);
}

TEST(ScanUpdate, InvalidImuRejectedBeforePrediction)
{
  Filter filter;
  ImuProcess processor;
  auto data = scan(1.0);
  auto bad = std::make_shared<sensor_msgs::msg::Imu>(*data.imu.front());
  bad->linear_acceleration.x = std::numeric_limits<double>::quiet_NaN();
  data.imu.front() = bad;
  const auto covariance = filter.get_P();
  PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
  EXPECT_EQ(processor.Process(data, filter, cloud).status,
    ImuProcess::ProcessStatus::Rejected);
  EXPECT_TRUE(filter.get_P().isApprox(covariance));
}

TEST(ScanUpdate, RejectedIntervalsAreNotReplayed)
{
  Filter filter;
  initialize(filter);
  ImuProcess processor;
  processor.fout_imu.open("/tmp/fast-lio-scan-update-test-imu.txt");
  PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
  auto data = scan(1.0);
  ASSERT_EQ(processor.Process(data, filter, cloud).status,
    ImuProcess::ProcessStatus::Initializing);
  auto retained_state = filter.get_x();
  auto retained_covariance = filter.get_P();
  for (int i = 0; i < 4; ++i)
  {
    data = scan(2.0 + i);
    EXPECT_EQ(processor.Process(data, filter, cloud).status,
      ImuProcess::ProcessStatus::Rejected);
    filter.change_x(retained_state);
    filter.change_P(retained_covariance);
    processor.Skip(data, retained_state);
  }
  data = scan(5.1);
  ASSERT_EQ(processor.Process(data, filter, cloud).status,
    ImuProcess::ProcessStatus::Ready);
  EXPECT_LT(filter.get_x().pos.norm(), 1e-5);
  EXPECT_TRUE(filter.get_P().allFinite());
  EXPECT_FALSE(cloud->empty());
}
