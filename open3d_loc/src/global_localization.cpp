#include <rclcpp/rclcpp.hpp>
#include <fast_lio/msg/tracking_status.hpp>
#include <fast_lio/tracking_guard.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <atomic>
#include <unordered_set>
#include <rclcpp/wait_for_message.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/static_transform_broadcaster.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float32.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>

#include <tf2_eigen/tf2_eigen.hpp>
#include <deque>
#include <queue>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <optional>
#include <stdexcept>
#include <string>
#include <array>
#include <cstdint>
// #include <pcl/common/transforms.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <open3d/Open3D.h>

#include "open3d_registration/open3d_registration.h"
#include "open3d_conversions/open3d_conversions.h"
#include "open3d_loc/transform_utils.hpp"
#include "open3d_loc/global_correction_filter.hpp"
#include "open3d_loc/height_bounds.hpp"
#include "registration_geometry.hpp"

#define PI 3.1415926

namespace
{
std::optional<open3d::utility::VerbosityLevel> ParseOpen3dVerbosity(
    std::string verbosity)
{
    std::transform(
        verbosity.begin(), verbosity.end(), verbosity.begin(),
        [](unsigned char character) {return static_cast<char>(std::tolower(character));});

    if (verbosity == "debug") {
        return open3d::utility::VerbosityLevel::Debug;
    }
    if (verbosity == "info") {
        return open3d::utility::VerbosityLevel::Info;
    }
    if (verbosity == "warning" || verbosity == "warn") {
        return open3d::utility::VerbosityLevel::Warning;
    }
    if (verbosity == "error") {
        return open3d::utility::VerbosityLevel::Error;
    }
    return std::nullopt;
}

open3d_loc::Vector6d ToVector6(
    const std::vector<double> & values, const open3d_loc::Vector6d & fallback)
{
    if (values.size() != 6)
        return fallback;
    open3d_loc::Vector6d result;
    for (int i = 0; i < 6; ++i)
        result(i) = values[static_cast<std::size_t>(i)];
    return result;
}

open3d_loc::Matrix6d RosPoseCovariance(const nav_msgs::msg::Odometry & odometry)
{
    open3d_loc::Matrix6d covariance;
    for (int row = 0; row < 6; ++row)
        for (int column = 0; column < 6; ++column)
            covariance(row, column) = odometry.pose.covariance[static_cast<std::size_t>(row * 6 + column)];
    return covariance;
}

void SetRosPoseCovariance(
    const open3d_loc::Matrix6d & covariance, nav_msgs::msg::Odometry & odometry)
{
    for (int row = 0; row < 6; ++row)
        for (int column = 0; column < 6; ++column)
            odometry.pose.covariance[static_cast<std::size_t>(row * 6 + column)] = covariance(row, column);
}

Eigen::Matrix3d Skew(const Eigen::Vector3d & vector)
{
    Eigen::Matrix3d matrix;
    matrix << 0.0, -vector.z(), vector.y(),
              vector.z(), 0.0, -vector.x(),
              -vector.y(), vector.x(), 0.0;
    return matrix;
}

open3d_loc::Matrix6d ComposePoseCovariance(
    const Eigen::Matrix4d & map_to_odom,
    const open3d_loc::Matrix6d & correction_covariance,
    const open3d_loc::Matrix6d & odometry_covariance)
{
    open3d_loc::Matrix6d adjoint = open3d_loc::Matrix6d::Zero();
    const Eigen::Matrix3d rotation = map_to_odom.block<3, 3>(0, 0);
    adjoint.block<3, 3>(0, 0) = rotation;
    adjoint.block<3, 3>(0, 3) = Skew(map_to_odom.block<3, 1>(0, 3)) * rotation;
    adjoint.block<3, 3>(3, 3) = rotation;
    return open3d_loc::GlobalCorrectionFilter::regularizeCovariance(
        correction_covariance + adjoint * odometry_covariance * adjoint.transpose(),
        open3d_loc::Vector6d::Constant(1e-6));
}

open3d_loc::Matrix6d IcpMeasurementCovariance(
    const open3d::geometry::PointCloud & source,
    const open3d::geometry::PointCloud & target,
    const double correspondence_distance,
    const Eigen::Matrix4d & map_to_odom,
    const double rmse,
    const double covariance_scale)
{
    const open3d_loc::Matrix6d information =
        open3d::pipelines::registration::GetInformationMatrixFromPointClouds(
            source, target, correspondence_distance, map_to_odom);
    Eigen::SelfAdjointEigenSolver<open3d_loc::Matrix6d> solver(
        0.5 * (information + information.transpose()));
    open3d_loc::Matrix6d covariance_open3d = open3d_loc::Matrix6d::Identity() * 1e6;
    if (solver.info() == Eigen::Success)
    {
        Eigen::Matrix<double, 6, 1> inverse_eigenvalues;
        const double maximum = std::max(solver.eigenvalues().maxCoeff(), 1.0);
        for (int i = 0; i < 6; ++i)
            inverse_eigenvalues(i) = 1.0 / std::max(solver.eigenvalues()(i), maximum * 1e-9);
        covariance_open3d = solver.eigenvectors() * inverse_eigenvalues.asDiagonal() *
            solver.eigenvectors().transpose();
    }
    covariance_open3d *= std::max(rmse * rmse, 1e-6) * std::max(covariance_scale, 1.0);

    // Open3D orders the tangent as rotation then translation; ROS uses translation then rotation.
    open3d_loc::Matrix6d covariance_ros;
    for (int row = 0; row < 6; ++row)
    {
        const int open3d_row = row < 3 ? row + 3 : row - 3;
        for (int column = 0; column < 6; ++column)
        {
            const int open3d_column = column < 3 ? column + 3 : column - 3;
            covariance_ros(row, column) = covariance_open3d(open3d_row, open3d_column);
        }
    }
    return covariance_ros;
}

diagnostic_msgs::msg::KeyValue DiagnosticValue(
    const std::string & key, const std::string & value)
{
    diagnostic_msgs::msg::KeyValue item;
    item.key = key;
    item.value = value;
    return item;
}

bool IsRecoverableFilterRejection(const std::string & reason)
{
    return reason == "translation_gate" || reason == "rotation_gate" ||
           reason == "mahalanobis_gate";
}
}  // namespace

class KalmanFilter
{
public:
    KalmanFilter() : processVar_(0.0), estimatedMeasVar_(0.0),
                     posteriEstimate_(0.0), posteriErrorEstimate_(1.0)
    {
    }

    void KalmanFilterInit(double processVar, double estimatedMeasVar, double posteriEstimate = 0.0, double posteriErrorEstimate = 1.0)
    {
        processVar_ = processVar;
        estimatedMeasVar_ = estimatedMeasVar;
        posteriEstimate_ = posteriEstimate;
        posteriErrorEstimate_ = posteriErrorEstimate;
    }
    void inputLatestNoisyMeasurement(double measurement)
    {
        double prioriEstimate = posteriEstimate_;
        double prioriErrorEstimate = posteriErrorEstimate_ + processVar_;

        double denominator = prioriErrorEstimate + estimatedMeasVar_;

        // 防止除零导致 NaN
        if (std::abs(denominator) < 1e-10)
        {
            // 如果分母接近零，直接使用测量值
            posteriEstimate_ = measurement;
            posteriErrorEstimate_ = 1.0;
            return;
        }

        double blendingFactor = prioriErrorEstimate / denominator;
        posteriEstimate_ = prioriEstimate + blendingFactor * (measurement - prioriEstimate);
        posteriErrorEstimate_ = (1 - blendingFactor) * prioriErrorEstimate;
    }

    double getLatestEstimatedMeasurement()
    {
        return posteriEstimate_;
    }

private:
    double processVar_;
    double estimatedMeasVar_;
    double posteriEstimate_;
    double posteriErrorEstimate_;
};

class GloabalLocalization : public rclcpp::Node
{
private:
    /* data */
public:
    GloabalLocalization();
    ~GloabalLocalization();

    /// @brief 初始化定位
    void LocalizationInitialize();

    /// @brief 订阅fast_lio里程计信息
    void CallbackBaselink2Odom(const nav_msgs::msg::Odometry::SharedPtr baselink2odom);
    /// @brief 订阅在baselink下的点云
    void CallbackScan(const sensor_msgs::msg::PointCloud2::SharedPtr scan_in_baselink);

    /// @brief 订阅在初始位姿
    void CallbackInitialPose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr initialpose);

    void StartLoc();
    void ReportRegistrationFailure(const std::string & reason);

    void Localization();

    /// @brief 欧拉角转mat3x3
    /// @param euler
    /// @return
    Eigen::Matrix3d Euler2Matrix3d(const Eigen::Vector3d euler);

    /// @brief 获取tf关系到矩阵
    /// @param frame_id
    /// @param child_frame_id
    /// @param matrix
    /// @return
    bool GetTfTransformToMatrix(const std::string &frame_id, const std::string &child_frame_id,
                                const builtin_interfaces::msg::Time &stamp, Eigen::Matrix4d &matrix);

    /// @brief compute 3d distance between two points
    /// @param a
    /// @param b
    /// @return 距离值
    double ComputeMotionDis(const Eigen::Vector3d &a, const Eigen::Vector3d &b);

    struct MotionHistorySample
    {
        rclcpp::Time stamp;
        double cumulative_distance = 0.0;
        double cumulative_rotation = 0.0;
    };

    bool LookupMotionSampleLocked(
        const rclcpp::Time & stamp, MotionHistorySample & sample) const;

private:
    /// @brief 订阅baselink2odom,即fast_lio的里程计信息
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_baselink2odom_;

    /// @brief 订阅当前帧点云
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_scan_cur_;

    /// @brief 订阅初始位姿
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_initialpose_;

    /// @brief baselink到odom的pose表达
    nav_msgs::msg::Odometry pose_baselink2odom_;

    /// @brief bselink到odom的变换矩阵表达
    Eigen::Matrix4d mat_baselink2odom_;
    /// @brief odom到map的矩阵
    Eigen::Matrix4d mat_odom2map_;
    Eigen::Matrix4d mat_odom2map_kalman_;
    /// @brief baselink到map = mat_odom2map * mat_baselink2odom
    Eigen::Matrix4d mat_baselink2map_;
    /// @brief initialpose初始位姿
    Eigen::Matrix4d mat_initialpose_;

    std::mutex lock_state_;
    rclcpp::Subscription<fast_lio::msg::TrackingStatus>::SharedPtr tracking_sub_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr reset_client_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
    rclcpp::TimerBase::SharedPtr reset_timer_;
    std::atomic<bool> tracking_ready_{false};
    std::atomic<bool> reset_pending_{false};
    std::atomic<bool> reinitialize_requested_{false};
    std::uint64_t tracking_generation_ = 0, reset_serial_ = 0;
    std::uint64_t tracking_instance_ = 0, expected_reset_instance_ = 0;
    std::unordered_set<std::uint64_t> retired_instances_;
    bool tracking_seen_ = false, reset_inflight_ = false, reset_acknowledged_ = false, reset_failed_ = false;
    std::uint64_t reset_started_generation_ = 0, expected_reset_generation_ = 0;
    std::int64_t reset_request_id_ = 0;
    rclcpp::Time accepted_tracking_stamp_{0, 0, RCL_ROS_TIME};
    std::deque<rclcpp::Time> accepted_tracking_stamps_;
    std::chrono::steady_clock::time_point reset_sent_;
    Eigen::Matrix4d pending_body_pose_ = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d trusted_body_pose_ = Eigen::Matrix4d::Identity();
    bool have_trusted_body_pose_ = false;
    double min_information_ratio_ = 1e-4;

    nav_msgs::msg::Odometry::SharedPtr pending_odometry_;
    sensor_msgs::msg::PointCloud2::SharedPtr pending_scan_;
    std::uint64_t latest_scan_generation_ = 0;
    bool IsAcceptedTrackingStamp(const builtin_interfaces::msg::Time & stamp) const
    {
        const rclcpp::Time time(stamp);
        return std::find(accepted_tracking_stamps_.begin(), accepted_tracking_stamps_.end(),
            time) != accepted_tracking_stamps_.end();
    }
    void DrainTrackingInputs()
    {
        if (!tracking_ready_ || (reset_pending_ && (!reset_acknowledged_ || reset_failed_))) return;
        if (pending_odometry_ && IsAcceptedTrackingStamp(pending_odometry_->header.stamp)) {
            auto message = pending_odometry_;
            pending_odometry_.reset();
            CallbackBaselink2Odom(message);
        }
        if (!reset_pending_ && pending_scan_ &&
            IsAcceptedTrackingStamp(pending_scan_->header.stamp)) {
            auto message = pending_scan_;
            pending_scan_.reset();
            CallbackScan(message);
        }
    }
    std::vector<std::function<void()>> held_publications_;
    template<typename Message>
    void PublishRemember(const typename rclcpp::Publisher<Message>::SharedPtr & publisher,
        const Message & message)
    {
        publisher->publish(message);
        held_publications_.push_back([publisher, message]() {publisher->publish(message);});
    }
    void RememberTransform(const geometry_msgs::msg::TransformStamped & transform)
    {
        br_odom2map_->sendTransform(transform);
        held_publications_.push_back([this, transform]() {br_odom2map_->sendTransform(transform);});
    }

    void ClearScanHistory()
    {
        std::lock_guard<std::mutex> lock(lock_scan_);
        que_pcd_scan_ = {};
        pcd_scan_cur_->Clear();
        latest_scan_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
        last_processed_scan_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    }

    void ReportResetStatus(std::uint8_t level, const std::string & reason)
    {
        diagnostic_msgs::msg::DiagnosticArray diagnostics;
        diagnostics.header.stamp = now();
        diagnostic_msgs::msg::DiagnosticStatus status;
        status.name = "open3d_loc/reset_tracking";
        status.hardware_id = "open3d_loc";
        status.level = level;
        status.message = reason;
        status.values.push_back(DiagnosticValue("reason", reason));
        diagnostics.status.push_back(std::move(status));
        pub_fusion_diagnostics_->publish(diagnostics);
    }

    void DispatchReset()
    {
        if (reset_inflight_ || !reset_pending_ || reset_acknowledged_ || reset_failed_) return;
        if (!reset_client_->service_is_ready()) return;
        reset_inflight_ = true;
        ReportResetStatus(1, "reset_in_progress");
        reset_sent_ = std::chrono::steady_clock::now();
        const auto serial = reset_serial_;
        reset_started_generation_ = tracking_generation_;
        auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
        auto pending = reset_client_->async_send_request(request,
            [this, serial](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
                reset_inflight_ = false;
                if (serial != reset_serial_) {DispatchReset(); return;}
                auto response = future.get();
                if (!response->success) {
                    reset_acknowledged_ = true;
                    reset_failed_ = true;
                    ReportRegistrationFailure("FAST-LIO reset failed: " + response->message);
                    ReportResetStatus(2, "backend_reset_failed");
                    return;
                }
                try {
                    const std::string prefix = "reset_generation=";
                    if (response->message.rfind(prefix, 0) != 0)
                        throw std::invalid_argument("missing reset generation");
                    std::size_t consumed = 0;
                    expected_reset_generation_ = std::stoull(response->message.substr(prefix.size()), &consumed);
                    const std::string instance_prefix = "; instance_id=";
                    const auto instance_offset = response->message.find(instance_prefix);
                    if (instance_offset == std::string::npos)
                        throw std::invalid_argument("missing estimator instance");
                    expected_reset_instance_ = std::stoull(
                        response->message.substr(instance_offset + instance_prefix.size()));
                    if (expected_reset_generation_ == 0)
                        throw std::invalid_argument("invalid reset generation");
                } catch (const std::exception & error) {
                    reset_failed_ = true;
                    ReportRegistrationFailure(std::string("Invalid FAST-LIO reset acknowledgment: ") + error.what());
                    ReportResetStatus(2, "invalid_reset_acknowledgment");
                }
                reset_acknowledged_ = true;
                if (!reset_failed_) ReportResetStatus(1, "waiting_for_local_tracking");
                RCLCPP_INFO(get_logger(), "FAST-LIO reset acknowledged; waiting for new tracking generation");
            });
        reset_request_id_ = pending.request_id;
    }

    void BeginReset(const Eigen::Matrix4d & desired)
    {
        pending_body_pose_ = desired;
        ++reset_serial_;
        reset_pending_ = true;
        reset_acknowledged_ = false;
        reset_failed_ = false;
        reset_wait_started_ = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(lock_state_);
            ++fusion_generation_;
            loc_initialized_ = false;
            loc_fitness_ = 0.0;
            have_odom_ = false;
            last_fusion_prediction_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
            accepted_tracking_stamps_.clear();
            motion_history_.clear();
            cumulative_odom_distance_ = 0.0;
            cumulative_odom_rotation_ = 0.0;
            force_submap_refresh_ = true;
            correction_filter_.reset(mat_odom2map_);
        }
        ClearScanHistory();
        pending_odometry_.reset(); pending_scan_.reset();
        ReportRegistrationFailure("reset_requested; holding last trusted localization");
        ReportResetStatus(1, "reset_requested");
        DispatchReset();
    }

    std::chrono::steady_clock::time_point reset_wait_started_;


    /// @brief baselink和运动中心
    Eigen::Matrix4d mat_baselink2motionlink_;

    /// @brief imulink到baselink
    Eigen::Matrix4d mat_imulink2baselink_;

    std::string imu_frame_ = "imu_link";
    std::string body_frame_ = "base_link";
    std::string output_frame_ = "motion_link";
    bool publish_robot_root_tf_ = false;
    bool publish_output_tf_ = true;
    double tf_lookup_max_age_ms_ = 100.0;
    bool have_odom_ = false;
    open3d_loc::HeightBounds height_bounds_;
    double height_before_bounds_ = 0.0;
    double bounded_height_ = 0.0;
    double height_adjustment_ = 0.0;
    std::uint64_t height_clamp_count_ = 0;

    // Caller holds lock_state_. Keep filter state and all derived poses consistent.
    void ApplyHeightBoundsLocked()
    {
        if (!have_odom_) return;
        height_before_bounds_ = (mat_odom2map_ * mat_baselink2odom_)(2, 3) - height_bounds_.floor_z;
        height_adjustment_ = height_bounds_.adjustment(mat_odom2map_, mat_baselink2odom_);
        if (height_adjustment_ != 0.0)
        {
            mat_odom2map_(2, 3) += height_adjustment_;
            if (correction_filter_.initialized())
                correction_filter_.adjustHeight(mat_odom2map_(2, 3) - correction_filter_.pose()(2, 3));
            ++height_clamp_count_;
            force_submap_refresh_ = true;
        }
        bounded_height_ = (mat_odom2map_ * mat_baselink2odom_)(2, 3) - height_bounds_.floor_z;
        mat_baselink2map_ = mat_odom2map_ * mat_baselink2odom_;
    }

    void AppendHeightDiagnosticsLocked(diagnostic_msgs::msg::DiagnosticStatus & status) const
    {
        status.values.push_back(DiagnosticValue("height_bounds_enabled", height_bounds_.enabled ? "true" : "false"));
        status.values.push_back(DiagnosticValue("height_floor_z", std::to_string(height_bounds_.floor_z)));
        status.values.push_back(DiagnosticValue("height_min", std::to_string(height_bounds_.min_height)));
        status.values.push_back(DiagnosticValue("height_max", std::to_string(height_bounds_.max_height)));
        status.values.push_back(DiagnosticValue("height_before_bounds", std::to_string(height_before_bounds_)));
        status.values.push_back(DiagnosticValue("height_published", std::to_string(bounded_height_)));
        status.values.push_back(DiagnosticValue("height_adjustment", std::to_string(height_adjustment_)));
        status.values.push_back(DiagnosticValue("height_clamp_count", std::to_string(height_clamp_count_)));
    }


    /// @brief 初始位姿, x, y, z, roll, pitch, yaw (单位:度degrees)
    std::vector<double> initialpose_;

    /// @brief 原始地图点云
    std::shared_ptr<open3d::geometry::PointCloud> pcd_map_ori_;
    std::shared_ptr<open3d::geometry::PointCloud> pcd_map_coarse_;
    std::shared_ptr<open3d::geometry::PointCloud> pcd_map_fine_;
    std::shared_ptr<open3d::geometry::PointCloud> pcd_map_cur_;
    std::shared_ptr<open3d::geometry::PointCloud> pcd_scan_cur_;

    std::queue<open3d::geometry::PointCloud> que_pcd_scan_;
    int queue_maxsize_;
    double voxelsize_coarse_;
    double voxelsize_fine_;

    /// @brief 定位配准fitness(overlap)阈值
    double threshold_fitness_;
    /// @brief 配准fitness(overlap)阈值
    double threshold_fitness_init_;

    std::thread thread_loc_;
    std::mutex lock_scan_;
    std::mutex lock_exit_;
    bool flag_exit_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_baselink2map_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_baselink2map_kalman_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_motionlink2map_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom2map_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom2map_kalman_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom2map_icp_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_localization_3d_odom_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr pub_fusion_diagnostics_;
    rclcpp::Time timestamp_odom_{0, 0, RCL_ROS_TIME};
    std::mutex lock_timestamp_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_scan_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_scan2map_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_submap_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_localization_3d_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_localization_3d_confidence_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_localization_3d_delay_ms_;

    geometry_msgs::msg::PoseStamped localization_3d_;
    std_msgs::msg::Float32 localization_3d_confidence_;
    std_msgs::msg::Float32 localization_3d_delay_ms_;

    std::shared_ptr<tf2_ros::TransformBroadcaster> br_odom2map_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;

    bool save_scan_;

    /// @brief 定位频率(定位间隔时间，多少秒1次)
    double loc_frequence_;

    /// @brief source点云最大点数量
    int maxpoints_source_ = 50000;
    /// @brief target点云最大点数量
    int maxpoints_target_ = 200000;

    /// @brief 初始化成功标志
    bool loc_initialized_ = false;

    /// @brief 当前定位overlap，confidence
    double loc_fitness_;

    /// @brief 定位置信度阈值
    double confidence_loc_th_;

    /// 卡尔曼滤波器
    KalmanFilter kf_baselink_x_;
    KalmanFilter kf_baselink_y_;
    KalmanFilter kf_baselink_z_;
    KalmanFilter kalman_filter_odom2map_;

    // 0:kf_processVar 1:kf_estimatedMeasVar
    std::vector<double> kf_param_x_;
    std::vector<double> kf_param_y_;
    std::vector<double> kf_param_z_;

    /// @brief 对odom2map进行kalman滤波
    bool filter_odom2map_ = false;
    double kalman_processVar2_ = 0.0;
    double kalman_estimatedMeasVar2_ = 0.0;

    bool fusion_enabled_ = false;
    open3d_loc::GlobalCorrectionFilterConfig fusion_config_;
    open3d_loc::GlobalCorrectionFilter correction_filter_;
    double fusion_shared_lidar_covariance_scale_ = 4.0;
    double fusion_max_measurement_age_ = 1.0;
    double fusion_max_icp_rmse_ = 0.3;
    int fusion_min_correspondences_ = 100;
    int fusion_max_consecutive_rejections_ = 5;
    int fusion_rejection_count_ = 0;
    std::uint64_t fusion_generation_ = 0;
    rclcpp::Time latest_scan_stamp_{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_processed_scan_stamp_{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_fusion_prediction_stamp_{0, 0, RCL_ROS_TIME};
    std::deque<MotionHistorySample> motion_history_;
    double cumulative_odom_distance_ = 0.0;
    double cumulative_odom_rotation_ = 0.0;
    bool force_submap_refresh_ = false;
    double last_localization_duration_ms_ = 0.0;
    std::uint64_t localization_overrun_count_ = 0;

    /// 1202
    /// @brief 上次更新定位时的定位值
    Eigen::Vector3d last_loc_;
    // Eigen::Vector3d cur_loc_;
    /// @brief 更新地图子图的距离,超过则更新地图子图
    double dis_updatemap_;

    tf2_ros::Buffer tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

GloabalLocalization::GloabalLocalization() : Node("global_loc_node"),
                                             tf_buffer_(this->get_clock()),
                                             tf_listener_(std::make_shared<tf2_ros::TransformListener>(tf_buffer_))
{
    const std::string open3d_verbosity =
        this->declare_parameter<std::string>("open3d_verbosity", "warning");
    const auto verbosity_level = ParseOpen3dVerbosity(open3d_verbosity);
    if (verbosity_level) {
        open3d::utility::SetVerbosityLevel(*verbosity_level);
    } else {
        RCLCPP_WARN(
            this->get_logger(),
            "Unsupported open3d_verbosity '%s'; using 'warning'",
            open3d_verbosity.c_str());
        open3d::utility::SetVerbosityLevel(open3d::utility::VerbosityLevel::Warning);
    }

    flag_exit_ = false;
    loc_initialized_ = false;
    mat_baselink2odom_ = Eigen::Matrix4d::Identity();
    mat_odom2map_ = Eigen::Matrix4d::Identity();
    mat_odom2map_kalman_ = Eigen::Matrix4d::Identity();
    mat_baselink2map_ = Eigen::Matrix4d::Identity();
    mat_initialpose_ = Eigen::Matrix4d::Identity();
    last_loc_ = Eigen::Vector3d(0, 0, -5000);

    pcd_map_ori_.reset(new open3d::geometry::PointCloud);
    pcd_map_coarse_.reset(new open3d::geometry::PointCloud);
    pcd_map_cur_.reset(new open3d::geometry::PointCloud);
    pcd_scan_cur_.reset(new open3d::geometry::PointCloud);
    pcd_map_fine_.reset(new open3d::geometry::PointCloud);
    queue_maxsize_ = 5;

    pub_baselink2map_ = this->create_publisher<nav_msgs::msg::Odometry>("/baselink2map", 100000);
    pub_baselink2map_kalman_ = this->create_publisher<nav_msgs::msg::Odometry>("/baselink2map_kalman", 100000);
    pub_motionlink2map_ = this->create_publisher<nav_msgs::msg::Odometry>("/motionlink2map", 100000);
    pub_odom2map_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom2map", 100000);
    pub_odom2map_kalman_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom2map_kalman", 100000);
    pub_odom2map_icp_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom2map_icp", 10);
    pub_localization_3d_odom_ = this->create_publisher<nav_msgs::msg::Odometry>("/localization_3d_odom", 10);
    pub_fusion_diagnostics_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
        "/localization_3d_diagnostics", 10);

    pub_map_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/global_map_3d", rclcpp::QoS(1).reliable().transient_local());
    pub_submap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/submap", 1);
    pub_scan2map_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/scan2map", 1);
    pub_scan_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/scan", 1);
    pub_localization_3d_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/localization_3d", 1);
    pub_localization_3d_confidence_ = this->create_publisher<std_msgs::msg::Float32>("/localization_3d_confidence", 1);
    pub_localization_3d_delay_ms_ = this->create_publisher<std_msgs::msg::Float32>("/localization_3d_delay_ms", 1);

    loc_frequence_ = 2.0; //
    loc_fitness_ = 0.0;
    // 注册回调函数
    sub_baselink2odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/Odometry_loc", 50, std::bind(&GloabalLocalization::CallbackBaselink2Odom, this, std::placeholders::_1));
    sub_scan_cur_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/cloud_registered_1", 50, std::bind(&GloabalLocalization::CallbackScan, this, std::placeholders::_1));
    sub_initialpose_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", 50, std::bind(&GloabalLocalization::CallbackInitialPose, this, std::placeholders::_1));

    rcl_interfaces::msg::ParameterDescriptor information_descriptor;
    information_descriptor.read_only = true;
    information_descriptor.description = "Startup-only point-to-plane observability gate";
    min_information_ratio_ = declare_parameter<double>("fusion.min_information_ratio", 1e-4, information_descriptor);
    if (!std::isfinite(min_information_ratio_) || min_information_ratio_ <= 0.0 ||
        min_information_ratio_ > 1.0) throw std::invalid_argument("invalid fusion.min_information_ratio");
    reset_client_ = create_client<std_srvs::srv::Trigger>("/fast_lio/reset_tracking");
    tracking_sub_ = create_subscription<fast_lio::msg::TrackingStatus>(
        "/fast_lio/tracking_status", rclcpp::QoS(1).reliable().transient_local(),
        [this](fast_lio::msg::TrackingStatus::ConstSharedPtr status) {
            bool invalidate = false;
            {
                std::lock_guard<std::mutex> lock(lock_state_);
                const bool new_instance = tracking_seen_ && status->instance_id != tracking_instance_;
                if (new_instance && retired_instances_.count(status->instance_id)) return;
                if (!new_instance && tracking_seen_ && status->generation < tracking_generation_) return;
                if (new_instance) retired_instances_.insert(tracking_instance_);
                const bool ready = status->state == fast_lio::msg::TrackingStatus::TRACKING;
                invalidate = new_instance || tracking_generation_ != status->generation || (tracking_ready_ && !ready);
                if (invalidate) {
                    ++fusion_generation_;
                    accepted_tracking_stamps_.clear();
                    force_submap_refresh_ = true;
                    motion_history_.clear();
                    if (new_instance || tracking_generation_ != status->generation) {
                        have_odom_ = false;
                        loc_initialized_ = false;
                        loc_fitness_ = 0.0;
                        cumulative_odom_distance_ = 0.0;
                        cumulative_odom_rotation_ = 0.0;
                        last_fusion_prediction_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
                        if (!reset_pending_) {
                            pending_body_pose_ = have_trusted_body_pose_ ? trusted_body_pose_ : mat_baselink2map_;
                            reset_pending_ = true;
                            reset_acknowledged_ = true;
                            reset_started_generation_ = tracking_generation_;
                            expected_reset_generation_ = status->generation;
                            expected_reset_instance_ = status->instance_id;
                        }
                    }
                }
                if (!tracking_seen_ || tracking_generation_ != status->generation || tracking_ready_ != ready)
                    RCLCPP_INFO(get_logger(), "FAST-LIO tracking state=%u generation=%lu reset_pending=%d",
                        status->state, static_cast<unsigned long>(status->generation), bool(reset_pending_));
                tracking_instance_ = status->instance_id;
                tracking_seen_ = true;
                tracking_generation_ = status->generation;
                accepted_tracking_stamp_ = status->last_accepted_stamp;
                tracking_ready_ = ready;
                if (ready && accepted_tracking_stamp_.nanoseconds() > 0 &&
                    (accepted_tracking_stamps_.empty() ||
                    accepted_tracking_stamp_ > accepted_tracking_stamps_.back())) {
                    accepted_tracking_stamps_.push_back(accepted_tracking_stamp_);
                    if (accepted_tracking_stamps_.size() > 32) accepted_tracking_stamps_.pop_front();
                }
            }
            if (invalidate) {
                ClearScanHistory();
                pending_odometry_.reset(); pending_scan_.reset();
            }
            if (invalidate && !tracking_ready_)
                ReportRegistrationFailure("upstream tracking unavailable: " + status->reason);
            DrainTrackingInputs();
        });
    reset_service_ = create_service<std_srvs::srv::Trigger>("/localization/reset_tracking",
        [this](const std_srvs::srv::Trigger::Request::SharedPtr,
            std_srvs::srv::Trigger::Response::SharedPtr response) {
            Eigen::Matrix4d desired;
            {
                std::lock_guard<std::mutex> lock(lock_state_);
                desired = have_trusted_body_pose_ ? trusted_body_pose_ : mat_initialpose_;
                if (!have_trusted_body_pose_ && height_bounds_.enabled)
                    desired(2, 3) = height_bounds_.floor_z + std::clamp(
                    mat_initialpose_(2, 3) - height_bounds_.floor_z,
                    height_bounds_.min_height, height_bounds_.max_height);
            }
            BeginReset(desired);
            response->success = true;
            response->message = "Reset accepted; relocalization completion is reported through diagnostics";
        });
    reset_timer_ = create_wall_timer(std::chrono::milliseconds(100), [this]() {
        if (reset_inflight_ &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - reset_sent_).count() > 5.0) {
            reset_client_->remove_pending_request(reset_request_id_);
            reset_inflight_ = false;
            ++reset_serial_;
            reset_acknowledged_ = true;
            reset_failed_ = true;
            ReportRegistrationFailure("FAST-LIO reset service timeout; reset remains unavailable");
            ReportResetStatus(2, "backend_reset_timeout");
        }
        if (reset_pending_ && !reset_acknowledged_ &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - reset_wait_started_).count() > 5.0) {
            reset_acknowledged_ = true;
            reset_failed_ = true;
            ReportRegistrationFailure("FAST-LIO reset service unavailable; retry initialpose after restoring service");
            ReportResetStatus(2, "backend_reset_unavailable");
        }
        DispatchReset();
        DrainTrackingInputs();
        bool hold;
        {
            std::lock_guard<std::mutex> lock(lock_state_);
            hold = !tracking_ready_ || reset_pending_ || !loc_initialized_;
        }
        if (hold) for (const auto & publish : held_publications_) publish();
    });

    pose_baselink2odom_ = nav_msgs::msg::Odometry();
    pose_baselink2odom_.header.frame_id = "odom";
    pose_baselink2odom_.child_frame_id = "base_link";
    // geometry_msgs的Quaternion会被初始化为0,0,0,0,而不是正确的0,0,0,1
    pose_baselink2odom_.pose.pose.orientation.w = 1;
    RCLCPP_INFO(this->get_logger(), "pose baselink2odom:\nx: %f, y: %f, z: %f, qx: %f, \
                            qy: %f, qz: %f, qw: %f",
                pose_baselink2odom_.pose.pose.position.x,
                pose_baselink2odom_.pose.pose.position.y,
                pose_baselink2odom_.pose.pose.position.z,
                pose_baselink2odom_.pose.pose.orientation.x,
                pose_baselink2odom_.pose.pose.orientation.y,
                pose_baselink2odom_.pose.pose.orientation.z,
                pose_baselink2odom_.pose.pose.orientation.w);

    // 队列最大数量
    this->declare_parameter<int>("pcd_queue_maxsize", 5);
    this->declare_parameter<bool>("save_scan", false);
    /// 最大点数量限制
    this->declare_parameter<int>("maxpoints_source", 50000);
    this->declare_parameter<int>("maxpoints_target", 200000);

    // 定位间隔时间
    this->declare_parameter<double>("loc_frequence", 2.0);

    /// 定位阈值
    this->declare_parameter<double>("confidence_loc_th", 0.6);

    /// 卡尔曼参数
    this->declare_parameter<std::vector<double>>("kf_baselink2map/x", std::vector<double>(2));
    this->declare_parameter<std::vector<double>>("kf_baselink2map/y", std::vector<double>(2));
    this->declare_parameter<std::vector<double>>("kf_baselink2map/z", std::vector<double>(2));

    this->declare_parameter<bool>("filter_odom2map", false);
    this->declare_parameter<double>("kalman_processVar2", 0.02);
    this->declare_parameter<double>("kalman_estimatedMeasVar2", 0.04);
    this->declare_parameter<bool>("fusion.enabled", false);
    this->declare_parameter<std::vector<bool>>(
        "fusion.update_mask", std::vector<bool>{true, true, true, false, false, true});
    this->declare_parameter<std::vector<double>>(
        "fusion.initial_stddev", std::vector<double>{0.25, 0.25, 0.15, 0.05, 0.05, 0.15});
    this->declare_parameter<std::vector<double>>(
        "fusion.process_stddev_time", std::vector<double>{0.02, 0.02, 0.01, 0.005, 0.005, 0.01});
    this->declare_parameter<std::vector<double>>(
        "fusion.process_stddev_distance", std::vector<double>{0.01, 0.01, 0.005, 0.002, 0.002, 0.005});
    this->declare_parameter<std::vector<double>>(
        "fusion.process_stddev_rotation", std::vector<double>{0.01, 0.01, 0.005, 0.005, 0.005, 0.01});
    this->declare_parameter<std::vector<double>>(
        "fusion.measurement_stddev_floor", std::vector<double>{0.03, 0.03, 0.03, 0.01, 0.01, 0.02});
    this->declare_parameter<double>("fusion.shared_lidar_covariance_scale", 4.0);
    this->declare_parameter<double>("fusion.max_measurement_age", 1.0);
    this->declare_parameter<double>("fusion.max_icp_rmse", 0.3);
    this->declare_parameter<int>("fusion.min_correspondences", 100);
    this->declare_parameter<int>("fusion.max_consecutive_rejections", 5);
    this->declare_parameter<double>("fusion.max_innovation_translation", 1.0);
    this->declare_parameter<double>("fusion.max_innovation_rotation", 0.35);
    this->declare_parameter<double>("fusion.mahalanobis_threshold", 13.277);
    this->declare_parameter<bool>("fusion.recovery.enabled", false);
    this->declare_parameter<int>("fusion.recovery.required_consistent_measurements", 2);
    this->declare_parameter<double>("fusion.recovery.minimum_candidate_interval", 0.75);
    this->declare_parameter<double>("fusion.recovery.candidate_timeout", 3.0);
    this->declare_parameter<double>("fusion.recovery.consistency_mahalanobis_threshold", 13.277);
    this->declare_parameter<double>("fusion.recovery.max_candidate_translation_delta", 0.50);
    this->declare_parameter<double>("fusion.recovery.max_candidate_rotation_delta", 0.20);
    this->declare_parameter<double>("fusion.recovery.max_step_translation", 0.50);
    this->declare_parameter<double>("fusion.recovery.max_step_rotation", 0.15);
    // voxelsize
    this->declare_parameter<double>("voxelsize_coarse", 0.2);
    this->declare_parameter<double>("voxelsize_fine", 0.05);
    this->declare_parameter<double>("threshold_fitness_init", 0.9);
    this->declare_parameter<double>("threshold_fitness", 0.9);
    this->declare_parameter<std::vector<double>>("initialpose", std::vector<double>());
    this->declare_parameter<double>("dis_updatemap", 1);
    this->declare_parameter<std::string>("imu_frame", "imu_link");
    this->declare_parameter<std::string>("body_frame", "base_link");
    this->declare_parameter<std::string>("output_frame", "motion_link");
    this->declare_parameter<bool>("publish_robot_root_tf", false);
    this->declare_parameter<bool>("publish_output_tf", true);
    this->declare_parameter<double>("tf_lookup_max_age_ms", 100.0);

    this->get_parameter("pcd_queue_maxsize", queue_maxsize_);
    this->get_parameter("save_scan", save_scan_);
    this->get_parameter("maxpoints_source", maxpoints_source_);
    this->get_parameter("maxpoints_target", maxpoints_target_);
    this->get_parameter("loc_frequence", loc_frequence_);
    loc_frequence_ = std::max(loc_frequence_, 0.1);
    this->get_parameter("confidence_loc_th", confidence_loc_th_);
    this->get_parameter("kf_baselink2map/x", kf_param_x_);
    this->get_parameter("kf_baselink2map/y", kf_param_y_);
    this->get_parameter("kf_baselink2map/z", kf_param_z_);
    this->get_parameter("filter_odom2map", filter_odom2map_);
    this->get_parameter("kalman_processVar2", kalman_processVar2_);
    this->get_parameter("kalman_estimatedMeasVar2", kalman_estimatedMeasVar2_);

    this->get_parameter("fusion.enabled", fusion_enabled_);
    std::vector<bool> fusion_update_mask;
    std::vector<double> fusion_initial_stddev;
    std::vector<double> fusion_process_stddev_time;
    std::vector<double> fusion_process_stddev_distance;
    std::vector<double> fusion_process_stddev_rotation;
    std::vector<double> fusion_measurement_stddev_floor;
    this->get_parameter("fusion.update_mask", fusion_update_mask);
    this->get_parameter("fusion.initial_stddev", fusion_initial_stddev);
    this->get_parameter("fusion.process_stddev_time", fusion_process_stddev_time);
    this->get_parameter("fusion.process_stddev_distance", fusion_process_stddev_distance);
    this->get_parameter("fusion.process_stddev_rotation", fusion_process_stddev_rotation);
    this->get_parameter("fusion.measurement_stddev_floor", fusion_measurement_stddev_floor);
    this->get_parameter("fusion.shared_lidar_covariance_scale", fusion_shared_lidar_covariance_scale_);
    this->get_parameter("fusion.max_measurement_age", fusion_max_measurement_age_);
    this->get_parameter("fusion.max_icp_rmse", fusion_max_icp_rmse_);
    this->get_parameter("fusion.min_correspondences", fusion_min_correspondences_);
    this->get_parameter(
        "fusion.max_consecutive_rejections", fusion_max_consecutive_rejections_);
    this->get_parameter("fusion.max_innovation_translation", fusion_config_.max_innovation_translation);
    this->get_parameter("fusion.max_innovation_rotation", fusion_config_.max_innovation_rotation);
    this->get_parameter("fusion.mahalanobis_threshold", fusion_config_.mahalanobis_threshold);
    this->get_parameter("fusion.recovery.enabled", fusion_config_.recovery.enabled);
    this->get_parameter(
        "fusion.recovery.required_consistent_measurements",
        fusion_config_.recovery.required_consistent_measurements);
    this->get_parameter(
        "fusion.recovery.minimum_candidate_interval",
        fusion_config_.recovery.minimum_candidate_interval);
    this->get_parameter(
        "fusion.recovery.candidate_timeout", fusion_config_.recovery.candidate_timeout);
    this->get_parameter(
        "fusion.recovery.consistency_mahalanobis_threshold",
        fusion_config_.recovery.consistency_mahalanobis_threshold);
    this->get_parameter(
        "fusion.recovery.max_candidate_translation_delta",
        fusion_config_.recovery.max_candidate_translation_delta);
    this->get_parameter(
        "fusion.recovery.max_candidate_rotation_delta",
        fusion_config_.recovery.max_candidate_rotation_delta);
    this->get_parameter(
        "fusion.recovery.max_step_translation", fusion_config_.recovery.max_step_translation);
    this->get_parameter(
        "fusion.recovery.max_step_rotation", fusion_config_.recovery.max_step_rotation);
    fusion_shared_lidar_covariance_scale_ =
        std::max(fusion_shared_lidar_covariance_scale_, 1.0);
    fusion_max_measurement_age_ = std::max(fusion_max_measurement_age_, 0.0);
    fusion_max_icp_rmse_ = std::max(fusion_max_icp_rmse_, 0.0);
    fusion_min_correspondences_ = std::max(fusion_min_correspondences_, 1);
    fusion_max_consecutive_rejections_ = std::max(fusion_max_consecutive_rejections_, 1);
    fusion_config_.max_innovation_translation =
        std::max(fusion_config_.max_innovation_translation, 0.0);
    fusion_config_.max_innovation_rotation =
        std::max(fusion_config_.max_innovation_rotation, 0.0);
    fusion_config_.mahalanobis_threshold =
        std::max(fusion_config_.mahalanobis_threshold, 0.0);
    if (fusion_update_mask.size() == 6)
    {
        for (int i = 0; i < 6; ++i)
            fusion_config_.update_mask[static_cast<std::size_t>(i)] =
                fusion_update_mask[static_cast<std::size_t>(i)];
    }
    else
    {
        RCLCPP_WARN(this->get_logger(), "fusion.update_mask must contain six values; using XYZ+yaw");
    }
    fusion_config_.initial_stddev = ToVector6(fusion_initial_stddev, fusion_config_.initial_stddev);
    fusion_config_.process_stddev_time =
        ToVector6(fusion_process_stddev_time, fusion_config_.process_stddev_time);
    fusion_config_.process_stddev_distance =
        ToVector6(fusion_process_stddev_distance, fusion_config_.process_stddev_distance);
    fusion_config_.process_stddev_rotation =
        ToVector6(fusion_process_stddev_rotation, fusion_config_.process_stddev_rotation);
    fusion_config_.measurement_stddev_floor =
        ToVector6(fusion_measurement_stddev_floor, fusion_config_.measurement_stddev_floor);
    rcl_interfaces::msg::ParameterDescriptor height_descriptor;
    height_descriptor.read_only = true;
    height_descriptor.description = "Startup-only height constraint; restart to change";
    height_bounds_.enabled = declare_parameter<bool>(
        "height_bounds.enabled", false, height_descriptor);
    height_bounds_.floor_z = declare_parameter<double>(
        "height_bounds.floor_z", 0.0, height_descriptor);
    height_bounds_.min_height = declare_parameter<double>(
        "height_bounds.min_height", 0.3, height_descriptor);
    height_bounds_.max_height = declare_parameter<double>(
        "height_bounds.max_height", 0.7, height_descriptor);
    height_bounds_.validate(fusion_enabled_);
    correction_filter_.configure(fusion_config_);
    if (fusion_enabled_ && filter_odom2map_)
        RCLCPP_WARN(this->get_logger(), "fusion.enabled overrides the legacy filter_odom2map path");
    RCLCPP_INFO(this->get_logger(), "Global correction fusion: %s", fusion_enabled_ ? "enabled" : "disabled");

    RCLCPP_INFO(this->get_logger(), "Kalman filter parameters:");
    RCLCPP_INFO(this->get_logger(), "  kf_x: [%.6f, %.6f], size: %zu",
                kf_param_x_.size() >= 1 ? kf_param_x_[0] : 0.0,
                kf_param_x_.size() >= 2 ? kf_param_x_[1] : 0.0,
                kf_param_x_.size());
    RCLCPP_INFO(this->get_logger(), "  kf_y: [%.6f, %.6f], size: %zu",
                kf_param_y_.size() >= 1 ? kf_param_y_[0] : 0.0,
                kf_param_y_.size() >= 2 ? kf_param_y_[1] : 0.0,
                kf_param_y_.size());
    RCLCPP_INFO(this->get_logger(), "  kf_z: [%.6f, %.6f], size: %zu",
                kf_param_z_.size() >= 1 ? kf_param_z_[0] : 0.0,
                kf_param_z_.size() >= 2 ? kf_param_z_[1] : 0.0,
                kf_param_z_.size());
    RCLCPP_INFO(this->get_logger(), "  filter_odom2map: %s", filter_odom2map_ ? "true" : "false");
    this->get_parameter("voxelsize_coarse", voxelsize_coarse_);
    this->get_parameter("voxelsize_fine", voxelsize_fine_);
    this->get_parameter("threshold_fitness_init", threshold_fitness_init_);
    this->get_parameter("threshold_fitness", threshold_fitness_);
    this->get_parameter("initialpose", initialpose_);
    this->get_parameter("dis_updatemap", dis_updatemap_);
    this->get_parameter("imu_frame", imu_frame_);
    this->get_parameter("body_frame", body_frame_);
    this->get_parameter("output_frame", output_frame_);
    this->get_parameter("publish_robot_root_tf", publish_robot_root_tf_);
    this->get_parameter("publish_output_tf", publish_output_tf_);
    this->get_parameter("tf_lookup_max_age_ms", tf_lookup_max_age_ms_);
    pose_baselink2odom_.child_frame_id = body_frame_;

    for (auto i : initialpose_)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    mat_initialpose_.block<3, 3>(0, 0) = Euler2Matrix3d(Eigen::Vector3d(initialpose_[3], initialpose_[4], initialpose_[5]));
    mat_initialpose_.block<3, 1>(0, 3) = Eigen::Vector3d(initialpose_[0], initialpose_[1], initialpose_[2]);

    // 读取地图
    std::string path_map = "";
    this->declare_parameter<std::string>("path_map", "");
    this->get_parameter("path_map", path_map);
    open3d::io::ReadPointCloud(path_map, *pcd_map_ori_);
    if (pcd_map_ori_ == nullptr || pcd_map_ori_->IsEmpty())
    {
        RCLCPP_ERROR(this->get_logger(), "read map from path: %s failed", path_map.c_str());
        rclcpp::shutdown();
    }

    if (!pcd_map_ori_->HasColors())
    {
        pcd_map_ori_->PaintUniformColor({1, 0, 0});
    }
    // pcd_map_ori_->PaintUniformColor({1, 0, 0});

    pcd_map_coarse_ = pcd_map_ori_->VoxelDownSample(voxelsize_coarse_);
    pcd_map_coarse_->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxelsize_coarse_ * 2, 30));

    /// publish map, 用粗地图可视化，减少资源占用
    sensor_msgs::msg::PointCloud2 pc2_map;
    open3d_conversions::open3dToRos(*pcd_map_coarse_, pc2_map);
    pc2_map.header.frame_id = "map";
    pc2_map.header.stamp = this->now();
    pub_map_->publish(pc2_map);

    pcd_map_fine_ = pcd_map_ori_->VoxelDownSample(voxelsize_fine_);
    pcd_map_fine_->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxelsize_fine_ * 2, 30));

    mat_imulink2baselink_ = Eigen::Matrix4d::Identity();
    mat_baselink2motionlink_ = Eigen::Matrix4d::Identity();

    RCLCPP_WARN(this->get_logger(), "initialize finished");

    br_odom2map_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

    StartLoc();
}

GloabalLocalization::~GloabalLocalization()
{
    lock_exit_.lock();
    flag_exit_ = true;
    lock_exit_.unlock();
    if (thread_loc_.joinable())
        thread_loc_.join();
}

Eigen::Matrix3d GloabalLocalization::Euler2Matrix3d(const Eigen::Vector3d euler)
{
    Eigen::Matrix3d mat3d;
    // convert degrees to radians
    auto eulerAngle = euler / 180 * M_PI;
    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(eulerAngle[0], Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(eulerAngle[1], Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(eulerAngle[2], Eigen::Vector3d::UnitZ()));
    mat3d = rollAngle * pitchAngle * yawAngle;
    return mat3d;
}
bool GloabalLocalization::GetTfTransformToMatrix(
    const std::string &frame_id, const std::string &child_frame_id,
    const builtin_interfaces::msg::Time &stamp, Eigen::Matrix4d &matrix)
{
    geometry_msgs::msg::TransformStamped pose;
    try
    {
        pose = tf_buffer_.lookupTransform(
            frame_id, child_frame_id, rclcpp::Time(stamp), rclcpp::Duration::from_seconds(0.02));
    }
    catch (const tf2::TransformException &)
    {
        try
        {
            pose = tf_buffer_.lookupTransform(frame_id, child_frame_id, rclcpp::Time(0));
            const bool is_static = pose.header.stamp.sec == 0 && pose.header.stamp.nanosec == 0;
            if (!is_static)
            {
                const double age_ms = std::abs((rclcpp::Time(stamp) - rclcpp::Time(pose.header.stamp)).seconds()) * 1000.0;
                if (age_ms > tf_lookup_max_age_ms_)
                    return false;
            }
        }
        catch (const tf2::TransformException &)
        {
            return false;
        }
    }

    Eigen::Vector3d translation = Eigen::Vector3d(pose.transform.translation.x, pose.transform.translation.y, pose.transform.translation.z);
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();

    quat = Eigen::Quaterniond(pose.transform.rotation.w,
                              pose.transform.rotation.x,
                              pose.transform.rotation.y,
                              pose.transform.rotation.z);
    Eigen::Matrix3d rotation = quat.matrix();

    matrix = Eigen::Matrix4d::Identity();
    matrix.block<3, 3>(0, 0) = rotation;
    matrix.matrix().block<3, 1>(0, 3) = translation;
    return true;
}

void GloabalLocalization::CallbackBaselink2Odom(const nav_msgs::msg::Odometry::SharedPtr baselink2odom)
{
    {
        std::lock_guard<std::mutex> lock(lock_state_);
        if (!tracking_ready_ || !IsAcceptedTrackingStamp(baselink2odom->header.stamp)) {
            if (!pending_odometry_ || rclcpp::Time(baselink2odom->header.stamp) >
                rclcpp::Time(pending_odometry_->header.stamp)) pending_odometry_ = baselink2odom;
            return;
        }
        // Historical retransmissions and late older messages must not roll
        // back the body pose or advance the scalar filters repeatedly.
        if (have_odom_ && rclcpp::Time(baselink2odom->header.stamp) <=
            last_fusion_prediction_stamp_) return;
        if (reset_pending_ && (reset_failed_ || !reset_acknowledged_ || (tracking_instance_ != expected_reset_instance_ || tracking_generation_ < expected_reset_generation_))) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Waiting for reset: generation=%lu expected=%lu acknowledged=%d failed=%d",
                static_cast<unsigned long>(tracking_generation_),
                static_cast<unsigned long>(expected_reset_generation_), reset_acknowledged_, reset_failed_);
            return;
        }
    }
    Eigen::Matrix4d mat_body_to_imu = Eigen::Matrix4d::Identity();
    if (!GetTfTransformToMatrix(body_frame_, imu_frame_, baselink2odom->header.stamp, mat_body_to_imu))
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "Waiting for a recent %s -> %s transform", body_frame_.c_str(), imu_frame_.c_str());
        return;
    }

    Eigen::Matrix4d mat_body_to_output = Eigen::Matrix4d::Identity();
    bool output_transform_valid = output_frame_ == body_frame_;
    if (!output_transform_valid)
    {
        output_transform_valid = GetTfTransformToMatrix(
            body_frame_, output_frame_, baselink2odom->header.stamp, mat_body_to_output);
        if (!output_transform_valid)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Waiting for a recent %s -> %s transform; torso pose output is paused",
                                 body_frame_.c_str(), output_frame_.c_str());
        }
    }

    Eigen::Isometry3d mat_current = Eigen::Isometry3d::Identity();
    tf2::fromMsg(baselink2odom->pose.pose, mat_current);
    const Eigen::Matrix4d mat_imu_to_odom = mat_current.matrix();
    const Eigen::Matrix4d mat_body_to_odom =
        open3d_loc::deriveBodyPose(mat_imu_to_odom, mat_body_to_imu);

    if (!mat_body_to_odom.allFinite())
    {
        RCLCPP_WARN(this->get_logger(), "Ignoring nonfinite body odometry");
        return;
    }
    Eigen::Matrix4d mat_odom_to_map = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d mat_body_to_map = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d mat_body_to_map_filtered = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d mat_odom_to_map_kalman = Eigen::Matrix4d::Identity();
    bool localization_initialized = false;
    double localization_fitness = 0.0;
    open3d_loc::Matrix6d correction_covariance = open3d_loc::Matrix6d::Zero();
    open3d_loc::Matrix6d fused_pose_covariance = RosPoseCovariance(*baselink2odom);

    {
        std::lock_guard<std::mutex> lock(lock_state_);
        const rclcpp::Time odom_stamp(baselink2odom->header.stamp);
        double dt = 0.0;
        double distance = 0.0;
        double rotation = 0.0;
        if (have_odom_)
        {
            if (last_fusion_prediction_stamp_.nanoseconds() != 0 &&
                odom_stamp.get_clock_type() == last_fusion_prediction_stamp_.get_clock_type())
            {
                dt = (odom_stamp - last_fusion_prediction_stamp_).seconds();
            }
            const Eigen::Matrix4d relative_motion =
                mat_baselink2odom_.inverse() * mat_body_to_odom;
            distance = relative_motion.block<3, 1>(0, 3).norm();
            rotation = std::abs(
                Eigen::AngleAxisd(relative_motion.block<3, 3>(0, 0)).angle());
        }

        const bool timestamp_rollback =
            !motion_history_.empty() && odom_stamp < motion_history_.back().stamp;
        if (timestamp_rollback)
        {
            motion_history_.clear();
            cumulative_odom_distance_ = 0.0;
            cumulative_odom_rotation_ = 0.0;
            correction_filter_.clearRecovery();
        }
        else if (have_odom_ &&
            (motion_history_.empty() || odom_stamp > motion_history_.back().stamp))
        {
            cumulative_odom_distance_ += distance;
            cumulative_odom_rotation_ += rotation;
        }
        if (motion_history_.empty() || odom_stamp > motion_history_.back().stamp)
        {
            motion_history_.push_back(
                MotionHistorySample{odom_stamp, cumulative_odom_distance_, cumulative_odom_rotation_});
            const double history_duration = std::max(
                10.0, fusion_config_.recovery.candidate_timeout + 2.0);
            while (motion_history_.size() > 1 &&
                (odom_stamp - motion_history_.front().stamp).seconds() > history_duration)
            {
                motion_history_.pop_front();
            }
        }

        if (loc_initialized_ && fusion_enabled_ && correction_filter_.initialized())
        {
            correction_filter_.predict(std::max(dt, 0.0), distance, rotation);
            mat_odom2map_ = correction_filter_.pose();
            correction_covariance = correction_filter_.covariance();
        }
        last_fusion_prediction_stamp_ = odom_stamp;
        mat_imulink2baselink_ = mat_body_to_imu;
        mat_baselink2motionlink_ = mat_body_to_output;
        mat_baselink2odom_ = mat_body_to_odom;
        have_odom_ = true;
        if (reset_pending_) {
            // Keep fresh gravity roll/pitch and apply only the requested map yaw.
            const double desired_yaw = std::atan2(pending_body_pose_(1, 0), pending_body_pose_(0, 0));
            const double local_yaw = std::atan2(mat_body_to_odom(1, 0), mat_body_to_odom(0, 0));
            Eigen::Matrix4d desired = pending_body_pose_;
            desired.block<3, 3>(0, 0) =
                open3d_loc::levelOrientationFromYaw(desired_yaw - local_yaw) *
                mat_body_to_odom.block<3, 3>(0, 0);
            mat_odom2map_ = desired * mat_body_to_odom.inverse();
            mat_initialpose_ = mat_odom2map_;
            mat_odom2map_kalman_ = mat_odom2map_;
            correction_filter_.reset(mat_odom2map_);
            ++fusion_generation_;
            last_fusion_prediction_stamp_ = odom_stamp;
            fusion_rejection_count_ = 0;
            const auto position = desired.block<3, 1>(0, 3);
            if (kf_param_x_.size() >= 2 && kf_param_y_.size() >= 2 && kf_param_z_.size() >= 2) {
                kf_baselink_x_.KalmanFilterInit(kf_param_x_[0], kf_param_x_[1], position.x(), 1);
                kf_baselink_y_.KalmanFilterInit(kf_param_y_[0], kf_param_y_[1], position.y(), 1);
                kf_baselink_z_.KalmanFilterInit(kf_param_z_[0], kf_param_z_[1], position.z(), 1);
            }
            kalman_filter_odom2map_.KalmanFilterInit(kalman_processVar2_, kalman_estimatedMeasVar2_, mat_odom2map_(2, 3), 1);
            RCLCPP_INFO(get_logger(), "Fresh post-reset odometry received; starting global relocalization");
            ReportResetStatus(1, "waiting_for_global_registration");
            reset_pending_ = false;
            reinitialize_requested_ = true;
            force_submap_refresh_ = true;
        }
        ApplyHeightBoundsLocked();

        mat_odom_to_map = mat_odom2map_;
        mat_body_to_map = mat_baselink2map_;
        mat_odom_to_map_kalman = mat_odom2map_kalman_;
        mat_body_to_map_filtered = mat_body_to_map;
        if (height_bounds_.enabled)
        {
            diagnostic_msgs::msg::DiagnosticArray diagnostics;
            diagnostics.header.stamp = baselink2odom->header.stamp;
            diagnostic_msgs::msg::DiagnosticStatus status;
            status.name = "open3d_loc/height_bounds";
            status.hardware_id = "open3d_loc";
            status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
            status.message = height_adjustment_ == 0.0 ? "Height within bounds" : "Height bounded";
            AppendHeightDiagnosticsLocked(status);
            diagnostics.status.push_back(std::move(status));
            pub_fusion_diagnostics_->publish(diagnostics);
        }
        localization_initialized = loc_initialized_;
        if (localization_initialized) {
            trusted_body_pose_ = mat_baselink2map_;
            have_trusted_body_pose_ = true;
        }
        localization_fitness = loc_fitness_;
        if (fusion_enabled_ && correction_filter_.initialized())
        {
            fused_pose_covariance = ComposePoseCovariance(
                mat_odom2map_, correction_covariance, RosPoseCovariance(*baselink2odom));
        }

        if (localization_initialized && !fusion_enabled_)
        {
            if (filter_odom2map_)
            {
                kf_baselink_z_.inputLatestNoisyMeasurement((mat_odom2map_kalman_ * mat_baselink2odom_)(2, 3));
                mat_body_to_map_filtered = mat_odom2map_kalman_ * mat_baselink2odom_;
            }
            else
            {
                kf_baselink_x_.inputLatestNoisyMeasurement(mat_baselink2map_(0, 3));
                kf_baselink_y_.inputLatestNoisyMeasurement(mat_baselink2map_(1, 3));
                kf_baselink_z_.inputLatestNoisyMeasurement(mat_baselink2map_(2, 3));
            }

            const double filtered_z = kf_baselink_z_.getLatestEstimatedMeasurement();
            mat_body_to_map_filtered(2, 3) = std::isfinite(filtered_z) ? filtered_z : mat_baselink2map_(2, 3);
        }
    }

    {
        std::lock_guard<std::mutex> lock(lock_timestamp_);
        timestamp_odom_ = baselink2odom->header.stamp;
    }

    if (!localization_initialized) return;
    held_publications_.clear();
    // ICP updates map -> odom asynchronously, so differentiating this map-frame
    // pose would create correction spikes rather than a physical body twist.
    Eigen::Isometry3d Isometry3d_baselink2map;
    Isometry3d_baselink2map.matrix() = mat_body_to_map;
    nav_msgs::msg::Odometry baselink2map;
    baselink2map.pose.pose = tf2::toMsg(Isometry3d_baselink2map);
    baselink2map.header.frame_id = "map";
    baselink2map.child_frame_id = body_frame_;
    baselink2map.header.stamp = baselink2odom->header.stamp;
    SetRosPoseCovariance(fused_pose_covariance, baselink2map);
    PublishRemember(pub_baselink2map_, baselink2map);

    Eigen::Isometry3d Isometry3d_odom2map;
    Isometry3d_odom2map.matrix() = mat_odom_to_map;
    nav_msgs::msg::Odometry odom2map;
    odom2map.pose.pose = tf2::toMsg(Isometry3d_odom2map);
    odom2map.header.frame_id = "map";
    odom2map.child_frame_id = "odom";
    odom2map.header.stamp = baselink2odom->header.stamp;
    SetRosPoseCovariance(correction_covariance, odom2map);
    PublishRemember(pub_odom2map_, odom2map);

    /// 发布tf关系
    geometry_msgs::msg::TransformStamped transform_odom2map;
    transform_odom2map.header.frame_id = "map";
    transform_odom2map.child_frame_id = "odom";
    transform_odom2map.header.stamp = baselink2odom->header.stamp;
    transform_odom2map.transform.translation.x = odom2map.pose.pose.position.x;
    transform_odom2map.transform.translation.y = odom2map.pose.pose.position.y;
    transform_odom2map.transform.translation.z = odom2map.pose.pose.position.z;
    transform_odom2map.transform.rotation = odom2map.pose.pose.orientation;
    RememberTransform(transform_odom2map);

    if (publish_robot_root_tf_)
    {
        Eigen::Isometry3d Isometry3d_body2odom;
        Isometry3d_body2odom.matrix() = mat_body_to_odom;
        const auto body_pose = tf2::toMsg(Isometry3d_body2odom);
        geometry_msgs::msg::TransformStamped transform_body2odom;
        transform_body2odom.header.frame_id = baselink2odom->header.frame_id;
        transform_body2odom.child_frame_id = body_frame_;
        transform_body2odom.header.stamp = baselink2odom->header.stamp;
        transform_body2odom.transform.translation.x = body_pose.position.x;
        transform_body2odom.transform.translation.y = body_pose.position.y;
        transform_body2odom.transform.translation.z = body_pose.position.z;
        transform_body2odom.transform.rotation = body_pose.orientation;
        RememberTransform(transform_body2odom);
    }

    /// 卡尔曼滤波 - 只在定位初始化完成后执行
    if (localization_initialized)
    {
        if (filter_odom2map_ && !fusion_enabled_)
        {
            Eigen::Isometry3d Isometry3d_odom2map_kalman;
            Isometry3d_odom2map_kalman.matrix() = mat_odom_to_map_kalman;
            nav_msgs::msg::Odometry odom2map_kalman;
            odom2map_kalman.pose.pose = tf2::toMsg(Isometry3d_odom2map_kalman);
            odom2map_kalman.header.frame_id = "map";
            odom2map_kalman.child_frame_id = "odom_kalman";
            odom2map_kalman.header.stamp = baselink2odom->header.stamp;
            PublishRemember(pub_odom2map_kalman_, odom2map_kalman);
        }
        Eigen::Isometry3d Isometry3d_baselink2map_kalman;
        Isometry3d_baselink2map_kalman.matrix() = mat_body_to_map_filtered;
        nav_msgs::msg::Odometry baselink2map_kalman;
        baselink2map_kalman.pose.pose = tf2::toMsg(Isometry3d_baselink2map_kalman);
        baselink2map_kalman.header.frame_id = "map";
        baselink2map_kalman.child_frame_id = body_frame_;
        // baselink2map_kalman.child_frame_id = "base_link_kalman";
        baselink2map_kalman.header.stamp = baselink2odom->header.stamp;
        PublishRemember(pub_baselink2map_kalman_, baselink2map_kalman);

        if (!output_transform_valid)
            return;

        Eigen::Matrix4d mat_motionlink2map =
            open3d_loc::deriveOutputPose(mat_body_to_map_filtered, mat_body_to_output);
        Eigen::Isometry3d Isometry3d_motionlink2map;
        Isometry3d_motionlink2map.matrix() = mat_motionlink2map;
        nav_msgs::msg::Odometry motionlink2map;
        motionlink2map.pose.pose = tf2::toMsg(Isometry3d_motionlink2map);
        motionlink2map.header.frame_id = "map";
        motionlink2map.child_frame_id = output_frame_;
        // baselink2map_kalman.child_frame_id = "base_link_kalman";
        motionlink2map.header.stamp = baselink2odom->header.stamp;
        SetRosPoseCovariance(fused_pose_covariance, motionlink2map);
        PublishRemember(pub_motionlink2map_, motionlink2map);

        /// 发布tf关系
        geometry_msgs::msg::TransformStamped transform;
        transform.header.frame_id = "map";
        transform.child_frame_id = output_frame_;
        transform.header.stamp = baselink2odom->header.stamp;
        transform.transform.translation.x = motionlink2map.pose.pose.position.x;
        transform.transform.translation.y = motionlink2map.pose.pose.position.y;
        transform.transform.translation.z = motionlink2map.pose.pose.position.z;
        transform.transform.rotation = motionlink2map.pose.pose.orientation;
        if (publish_output_tf_)
        {
            RememberTransform(transform);
        }

        localization_3d_confidence_.data = localization_fitness;
        PublishRemember(pub_localization_3d_confidence_, localization_3d_confidence_);
        localization_3d_delay_ms_.data = (this->now() - baselink2odom->header.stamp).seconds() * 1000.0;
        PublishRemember(pub_localization_3d_delay_ms_, localization_3d_delay_ms_);
        localization_3d_.header.frame_id = "map";
        localization_3d_.header.stamp = baselink2odom->header.stamp;
        localization_3d_.pose = motionlink2map.pose.pose;
        PublishRemember(pub_localization_3d_, localization_3d_);
        nav_msgs::msg::Odometry localization_3d_odom = motionlink2map;
        PublishRemember(pub_localization_3d_odom_, localization_3d_odom);
    }
}
void GloabalLocalization::CallbackScan(
    const sensor_msgs::msg::PointCloud2::SharedPtr scan_in_baselink)
{
    {
        std::lock_guard<std::mutex> lock(lock_state_);
        if (!tracking_ready_ || reset_pending_ ||
            !IsAcceptedTrackingStamp(scan_in_baselink->header.stamp)) {
            if (!pending_scan_ || rclcpp::Time(scan_in_baselink->header.stamp) >
                rclcpp::Time(pending_scan_->header.stamp)) pending_scan_ = scan_in_baselink;
            return;
        }
    }
    open3d::geometry::PointCloud pcd_recieved;
    sensor_msgs::msg::PointCloud2::ConstSharedPtr const_scan_ptr = scan_in_baselink;
    open3d_conversions::rosToOpen3d(const_scan_ptr, pcd_recieved);
    std::scoped_lock lock(lock_state_, lock_scan_);
    if (rclcpp::Time(scan_in_baselink->header.stamp) <= latest_scan_stamp_) return;
    if (que_pcd_scan_.size() >= static_cast<std::size_t>(queue_maxsize_))
        que_pcd_scan_.pop();
    que_pcd_scan_.push(std::move(pcd_recieved));
    latest_scan_stamp_ = scan_in_baselink->header.stamp;
    latest_scan_generation_ = fusion_generation_;

    if (que_pcd_scan_.size() == static_cast<std::size_t>(queue_maxsize_))
    {
        pcd_scan_cur_->Clear();
        std::queue<open3d::geometry::PointCloud> snapshot = que_pcd_scan_;
        while (!snapshot.empty())
        {
            *pcd_scan_cur_ += snapshot.front();
            snapshot.pop();
        }
    }
}

void GloabalLocalization::LocalizationInitialize()
{
    /// 裁剪后的地图
    std::shared_ptr<open3d::geometry::PointCloud> map_coarse_crop(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> map_fine_crop(new open3d::geometry::PointCloud);

    /// 当前环境感知子图点云
    std::shared_ptr<open3d::geometry::PointCloud> pcd_scan(new open3d::geometry::PointCloud);
    /// 环境感知子图转换到地图坐标系
    std::shared_ptr<open3d::geometry::PointCloud> pcd_scan2map(new open3d::geometry::PointCloud);

    /// 用于配准的source target
    std::shared_ptr<open3d::geometry::PointCloud> source(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> target(new open3d::geometry::PointCloud);

    /// cropbox,用于裁剪地图和当前环境感知子图
    std::shared_ptr<open3d::geometry::OrientedBoundingBox> OBB_map(new open3d::geometry::OrientedBoundingBox);
    std::shared_ptr<open3d::geometry::OrientedBoundingBox> OBB_scan(new open3d::geometry::OrientedBoundingBox);

    /// 当前baselink到odom和map坐标系的关系
    Eigen::Matrix4d mat_baselink2odom_cur = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d mat_baselink2map_cur = Eigen::Matrix4d::Identity();

    /// 固定感知子图/历史地图子图大小
    OBB_map->extent_ = Eigen::Vector3d(60, 60, 40);
    OBB_map->color_ = Eigen::Vector3d(1, 0.5, 0);
    OBB_scan->extent_ = Eigen::Vector3d(60, 60, 40);
    OBB_scan->color_ = Eigen::Vector3d(0, 1, 0);

    double fitness_initial; /// overlap
    double loc_cost = 0;    /// 定位耗时(ms)
    int count_success = 0;
    rclcpp::Time last_confirmation(0, 0, RCL_ROS_TIME);
    std::uint64_t confirmation_generation = 0;
    while (rclcpp::ok())
    {
        if (!tracking_ready_ || reset_pending_) {
            count_success = 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        auto loc_s = std::chrono::high_resolution_clock::now(); /// 开始定位计时
        Eigen::Matrix4d reg_matrix = Eigen::Matrix4d::Identity();
        bool have_fresh_scan = false;
        {
            std::scoped_lock lock(lock_state_, lock_scan_);
            if (confirmation_generation != fusion_generation_) {
                count_success = 0;
                last_confirmation = rclcpp::Time(0, 0, RCL_ROS_TIME);
                confirmation_generation = fusion_generation_;
            }
            have_fresh_scan = tracking_ready_ && !reset_pending_ &&
                latest_scan_generation_ == confirmation_generation &&
                !pcd_scan_cur_->IsEmpty() && latest_scan_stamp_ > last_confirmation;
            if (have_fresh_scan) {
                *pcd_scan = *pcd_scan_cur_;
                last_confirmation = latest_scan_stamp_;
                mat_baselink2odom_cur = mat_baselink2odom_;
                mat_baselink2map_cur = mat_baselink2map_;
                reg_matrix = mat_odom2map_;
            }
        }
        if (!have_fresh_scan) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        else
        {
            try
            {
                /// 将cropbox转换到对应位置进行裁剪点云
                OBB_map->center_ = mat_baselink2map_cur.block<3, 1>(0, 3);
                OBB_map->R_ = mat_baselink2map_cur.block<3, 3>(0, 0);
                OBB_scan->center_ = mat_baselink2odom_cur.block<3, 1>(0, 3);
                OBB_scan->R_ = mat_baselink2odom_cur.block<3, 3>(0, 0);
                *map_fine_crop = *pcd_map_fine_->Crop(*OBB_map);

                /// 配准计时
                *target = *map_fine_crop;
                open3d::utility::LogInfo("before sample, target size: {}, has normal: {}", target->points_.size(), target->HasNormals() ? "true" : "false");
                if (target->points_.size() > static_cast<size_t>(maxpoints_target_))
                {
                    target = target->RandomDownSample(double(maxpoints_target_) / target->points_.size());
                }
                open3d::utility::LogInfo("after sample, target size: {}, has normal: {}", target->points_.size(), target->HasNormals() ? "true" : "false");

                source = pcd_scan->Crop(*OBB_scan);
                open3d::utility::LogInfo("source size: {}, has normal: {}", source->points_.size(), source->HasNormals() ? "true" : "false");
                if (source->points_.size() > static_cast<size_t>(maxpoints_source_))
                {
                    source = source->RandomDownSample(double(maxpoints_source_) / source->points_.size());
                }
                open3d::utility::LogInfo("source size: {}, has normal: {}", source->points_.size(), source->HasNormals() ? "true" : "false");

                if (source->IsEmpty() || target->IsEmpty())
                    throw std::runtime_error("empty scan or map crop");

                source->Transform(reg_matrix);
                *pcd_scan2map = *source;

                // auto multiScale_reg_matrix = pcd_tools::RegistrationMultiScaleIcp(source, target, voxelsize_fine_, 1, {1, 2, 4});
                auto multiScale_reg_matrix = pcd_tools::RegistrationMultiScaleIcp(source, target, voxelsize_fine_, 1, {1, 2, 3});
                reg_matrix = multiScale_reg_matrix * reg_matrix;
                source->Transform(multiScale_reg_matrix);
                auto eva_result_coarse = open3d::pipelines::registration::EvaluateRegistration(*source, *target, voxelsize_fine_ * 3);
                open3d::utility::LogInfo("eva fitness: {}", eva_result_coarse.fitness_);
                fitness_initial = eva_result_coarse.fitness_;
                if (eva_result_coarse.correspondence_set_.size() < static_cast<std::size_t>(fusion_min_correspondences_) ||
                    !std::isfinite(eva_result_coarse.inlier_rmse_) ||
                    eva_result_coarse.inlier_rmse_ > fusion_max_icp_rmse_ ||
                    open3d_loc::pointToPlaneInformationRatio(*source, *target, eva_result_coarse,
                        Eigen::Matrix4d::Identity()) < min_information_ratio_)
                    fitness_initial = 0.0;
                *pcd_scan2map = *source;

            }
            catch (const std::exception & error)
            {
                count_success = 0;
                ReportRegistrationFailure(error.what());
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }

            auto loc_e = std::chrono::high_resolution_clock::now(); /// 结束定位计时
            loc_cost = std::chrono::duration_cast<std::chrono::microseconds>(loc_e - loc_s).count() / 1000.0;
            RCLCPP_INFO(this->get_logger(), "localization cost: %f ms", loc_cost);

            if (fitness_initial > threshold_fitness_init_)
            {
                {
                    std::lock_guard<std::mutex> lock(lock_state_);
                    if (confirmation_generation != fusion_generation_ || !tracking_ready_ || reset_pending_) {
                        count_success = 0;
                        continue;
                    }
                    mat_odom2map_ = reg_matrix;
                    ApplyHeightBoundsLocked();
                }
                count_success += 1;
                // Require three distinct scans from the same estimator epoch.
                if (count_success >= 3)
                {
                    std::lock_guard<std::mutex> lock(lock_state_);
                    if (confirmation_generation != fusion_generation_ || !tracking_ready_ || reset_pending_) {
                        count_success = 0;
                        continue;
                    }
                    const auto position = mat_baselink2map_.block<3, 1>(0, 3);
                    if (kf_param_x_.size() >= 2 && kf_param_y_.size() >= 2 && kf_param_z_.size() >= 2) {
                        kf_baselink_x_.KalmanFilterInit(kf_param_x_[0], kf_param_x_[1], position.x(), 1);
                        kf_baselink_y_.KalmanFilterInit(kf_param_y_[0], kf_param_y_[1], position.y(), 1);
                        kf_baselink_z_.KalmanFilterInit(kf_param_z_[0], kf_param_z_[1], position.z(), 1);
                    }
                    kalman_filter_odom2map_.KalmanFilterInit(
                        kalman_processVar2_, kalman_estimatedMeasVar2_, mat_odom2map_(2, 3), 1);
                    correction_filter_.reset(mat_odom2map_);
                    ++fusion_generation_;
                    loc_initialized_ = true;
                    reinitialize_requested_ = false;
                    ReportResetStatus(0, "relocalization_complete");
                    RCLCPP_INFO(get_logger(), "Localization initialization complete, Kalman filters ready");
                    break;
                }
            }
            else
            {
                count_success = 0;
            }
        }
    }

    open3d::utility::LogInfo("\n\n\nlocalization initialize success!!!!\n\n\n");
}
void GloabalLocalization::Localization()
{
    RCLCPP_INFO(this->get_logger(), "wait for Odometry_loc");
    // 等待接收到第一条里程计消息（通过检查timestamp是否有效）
    while (rclcpp::ok())
    {
        bool have_odom = false;
        {
            std::lock_guard<std::mutex> lock(lock_state_);
            have_odom = have_odom_;
        }
        if (have_odom)
            break;
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for Odometry_loc...");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    RCLCPP_INFO(this->get_logger(), "Received Odometry_loc");

    RCLCPP_INFO(this->get_logger(), "wait for cloud_registered_1");
    // 等待接收到第一条点云消息（通过检查pcd_scan_cur_是否为空）
    while (rclcpp::ok())
    {
        lock_scan_.lock();
        bool has_scan = !pcd_scan_cur_->IsEmpty();
        lock_scan_.unlock();
        if (has_scan)
        {
            break;
        }
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for cloud_registered_1...");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    RCLCPP_INFO(this->get_logger(), "Received cloud_registered_1");

    // initialize
    /****初始化定位****/
    {
        std::lock_guard<std::mutex> lock(lock_state_);
        mat_odom2map_ = mat_initialpose_;
        ApplyHeightBoundsLocked();
    }
    LocalizationInitialize();

    auto coordinate_ori = open3d::geometry::TriangleMesh::CreateCoordinateFrame(2.0);
    auto coordinate_loc = open3d::geometry::TriangleMesh::CreateCoordinateFrame(2.0);
    auto coordinate_OBB_scan = open3d::geometry::TriangleMesh::CreateCoordinateFrame(2.0);
    std::shared_ptr<open3d::geometry::PointCloud> pcd_scan(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> pcd_scancrop(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> pcd_scan2map(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> source(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> target(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> map_coarse_crop(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> map_fine_crop(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> pcd_submap(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::OrientedBoundingBox> OBB_map(new open3d::geometry::OrientedBoundingBox);
    std::shared_ptr<open3d::geometry::OrientedBoundingBox> OBB_scan(new open3d::geometry::OrientedBoundingBox);
    OBB_map->color_ = Eigen::Vector3d(1, 0.5, 0);
    OBB_map->extent_ = Eigen::Vector3d(60, 60, 40);

    OBB_scan->extent_ = Eigen::Vector3d(60, 60, 40);
    OBB_scan->color_ = Eigen::Vector3d(0, 1, 0);
    rclcpp::Time time_current;
    {
        std::lock_guard<std::mutex> lock(lock_timestamp_);
        time_current = timestamp_odom_;
    }
    rclcpp::Time time_last = time_current - rclcpp::Duration(3, 0);

    RCLCPP_INFO(this->get_logger(), "time_last: %f", time_last.seconds());
    RCLCPP_INFO(this->get_logger(), "time_current: %f", time_current.seconds());
    int scan_count = 0;

    std::string save_path = "/home/carlos/mount/E/lixin/data/yq_bag/scan_submap/";

    double time_diff_loc = 5;                                     /// 前后两次定位的时间差(s)
    std::chrono::high_resolution_clock::time_point time_last_loc; /// 上次定位的完成时间点
    std::chrono::high_resolution_clock::time_point time_this_loc; /// 当前定位的开始时间点
    double loc_cost = 0;                                          /// 定位耗时(ms)
    while (rclcpp::ok())
    {

        if (!tracking_ready_ || reset_pending_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        if (reinitialize_requested_.exchange(false)) {
            LocalizationInitialize();
            RCLCPP_INFO(get_logger(), "Coordinated reset relocalization complete");
            continue;
        }
        lock_timestamp_.lock();
        time_current = timestamp_odom_;
        lock_timestamp_.unlock();
        auto time_diff_frame = time_current.seconds() - time_last.seconds();
        time_last = time_current;
        if (std::fabs(time_diff_frame) < 1e-6)
        {
            loc_cost = 0.0;
            continue;
        }

        time_this_loc = std::chrono::high_resolution_clock::now();
        time_diff_loc = std::chrono::duration_cast<std::chrono::microseconds>(time_this_loc - time_last_loc).count() / 1000000.0 + loc_cost / 1000.0;

        if (time_diff_loc < loc_frequence_)
        {
            int wait_time = int((loc_frequence_ - time_diff_loc) * 1000);
            open3d::utility::LogInfo("\n\ntime_this_loc: {}, time_last: {},\ntime_diff: {} s, sleep {} ms",
                                     std::chrono::duration_cast<std::chrono::milliseconds>(time_this_loc.time_since_epoch()).count() / 1000.0,
                                     std::chrono::duration_cast<std::chrono::milliseconds>(time_last_loc.time_since_epoch()).count() / 1000.0, time_diff_loc, wait_time);
            std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));
        }
        else
        {
            open3d::utility::LogInfo("\n\ntime_diff:{} s, localization right now", time_diff_loc);
        }
        auto loc_s = std::chrono::high_resolution_clock::now(); /// 开始定位计时

        lock_scan_.lock();
        if (pcd_scan_cur_->IsEmpty())
        {
            lock_scan_.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        else
        {
            Eigen::Matrix4d mat_baselink2odom_cur = Eigen::Matrix4d::Identity();
            Eigen::Matrix4d mat_baselink2map_cur = Eigen::Matrix4d::Identity();
            rclcpp::Time scan_stamp;
            std::uint64_t fusion_generation = 0;
            bool force_submap_refresh = false;

            scan_stamp = latest_scan_stamp_;
            const auto scan_generation = latest_scan_generation_;
            if (scan_stamp.nanoseconds() <= last_processed_scan_stamp_.nanoseconds())
            {
                lock_scan_.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            *pcd_scan = *pcd_scan_cur_;
            last_processed_scan_stamp_ = scan_stamp;
            lock_scan_.unlock();

            Eigen::Matrix4d reg_matrix = Eigen::Matrix4d::Identity();
            {
                std::lock_guard<std::mutex> lock(lock_state_);
                // A reset can occur between the scan copy and this snapshot.
                if (!loc_initialized_ || !tracking_ready_ || reset_pending_ ||
                    scan_generation != fusion_generation_) continue;
                if (!fusion_enabled_ && filter_odom2map_)
                {
                    kalman_filter_odom2map_.inputLatestNoisyMeasurement(mat_odom2map_(2, 3));
                    kalman_filter_odom2map_.inputLatestNoisyMeasurement(mat_odom2map_(2, 3));
                    mat_odom2map_kalman_ = mat_odom2map_;
                    mat_odom2map_kalman_(2, 3) = kalman_filter_odom2map_.getLatestEstimatedMeasurement();
                }
                mat_baselink2odom_cur = mat_baselink2odom_;
                mat_baselink2map_cur = mat_baselink2map_;
                reg_matrix = mat_odom2map_;
                fusion_generation = fusion_generation_;
                force_submap_refresh = force_submap_refresh_;
                force_submap_refresh_ = false;
            }
            open3d::pipelines::registration::RegistrationResult reg_result2;
            open3d::pipelines::registration::RegistrationResult eva_result2;
            try
            {
                Eigen::Vector3d cur_loc(mat_baselink2map_cur(0, 3), mat_baselink2map_cur(1, 3), mat_baselink2map_cur(2, 3));
                auto dis_motion = ComputeMotionDis(last_loc_, cur_loc);
                if (force_submap_refresh || dis_motion > dis_updatemap_)
                {
                    auto submap_s = std::chrono::high_resolution_clock::now();

                    open3d::utility::LogInfo("\n***\n****\n***\n\n\nlast map update loc: x: {}, y: {}, z{},\n\
                    now loc: x: {}, y: {}, z{}, 3d distance: {}, now needpdate submap",
                                             last_loc_.x(), last_loc_.y(), last_loc_.z(), cur_loc.x(), cur_loc.y(), cur_loc.z(), dis_motion);
                    last_loc_ = cur_loc;
                    OBB_map->center_ = mat_baselink2map_cur.block<3, 1>(0, 3);
                    OBB_map->R_ = mat_baselink2map_cur.block<3, 3>(0, 0);

                    /// 粗地图和精地图
                    *map_fine_crop = *pcd_map_fine_->Crop(*OBB_map);

                    auto submap_e = std::chrono::high_resolution_clock::now();
                    auto submap_cost = std::chrono::duration_cast<std::chrono::microseconds>(submap_e - submap_s).count() / 1000.0;
                    RCLCPP_INFO(this->get_logger(), "submap_cost: %f ms", submap_cost);
                }

                OBB_scan->center_ = mat_baselink2odom_cur.block<3, 1>(0, 3);
                OBB_scan->R_ = mat_baselink2odom_cur.block<3, 3>(0, 0);

                *target = *map_fine_crop;
                open3d::utility::LogInfo("before sample, target size: {}, has normal: {}", target->points_.size(), target->HasNormals() ? "true" : "false");
                if (target->points_.size() > static_cast<size_t>(maxpoints_target_))
                {
                    target = target->RandomDownSample(double(maxpoints_target_) / target->points_.size());
                }
                open3d::utility::LogInfo("after sample, target size: {}, has normal: {}", target->points_.size(), target->HasNormals() ? "true" : "false");

                source = pcd_scan->Crop(*OBB_scan);
                open3d::utility::LogInfo("source size: {}, maxpoints_source_: {}", source->points_.size(), maxpoints_source_);
                source = source->VoxelDownSample(voxelsize_fine_);
                open3d::utility::LogInfo("source size after voxel downsample: {}", source->points_.size());
                if (source->points_.size() > static_cast<size_t>(maxpoints_source_))
                {
                    source = source->RandomDownSample(double(maxpoints_source_) / source->points_.size());
                }
                open3d::utility::LogInfo("after prerpocess: {}", source->points_.size());

                if (source->IsEmpty() || target->IsEmpty())
                    throw std::runtime_error("empty scan or map crop");
                if (!target->HasNormals())
                    throw std::runtime_error("map crop has no normals");

                reg_result2 = pcd_tools::RegistrationIcp(source, target, voxelsize_fine_ * 2, reg_matrix, 1);
                reg_matrix = reg_result2.transformation_ * reg_matrix;
                eva_result2 = open3d::pipelines::registration::EvaluateRegistration(*source, *target, voxelsize_fine_ * 4, reg_matrix);
            }
            catch (const std::exception & error)
            {
                ReportRegistrationFailure(error.what());
                loc_cost = 0.0;
                continue;
            }
            /// 给发布的置信度赋值
            const double localization_fitness = eva_result2.fitness_;
            open3d::utility::LogInfo("reg_result.fitness: {}, eva fitness: {}", reg_result2.fitness_, eva_result2.fitness_);
            const std::size_t correspondence_count = eva_result2.correspondence_set_.size();
            const double measurement_age = (this->now() - scan_stamp).seconds();
            open3d_loc::Matrix6d measurement_covariance = open3d_loc::Matrix6d::Identity() * 1e6;
            if (fusion_enabled_)
            {
                measurement_covariance = IcpMeasurementCovariance(
                    *source, *target, voxelsize_fine_ * 4, reg_matrix,
                    eva_result2.inlier_rmse_, fusion_shared_lidar_covariance_scale_);

                Eigen::Isometry3d icp_isometry = Eigen::Isometry3d::Identity();
                icp_isometry.matrix() = reg_matrix;
                nav_msgs::msg::Odometry icp_odometry;
                icp_odometry.header.frame_id = "map";
                icp_odometry.child_frame_id = "odom";
                icp_odometry.header.stamp = scan_stamp;
                icp_odometry.pose.pose = tf2::toMsg(icp_isometry);
                SetRosPoseCovariance(measurement_covariance, icp_odometry);
                pub_odom2map_icp_->publish(icp_odometry);
            }

            {
                std::lock_guard<std::mutex> lock(lock_state_);
                loc_fitness_ = localization_fitness;
                bool correction_accepted = false;
                bool recovery_applied = false;
                std::string diagnostic_reason = fusion_enabled_ ? "not_evaluated" : "fusion_disabled";
                std::string normal_rejection_reason;
                open3d_loc::CorrectionUpdateResult update_result;
                open3d_loc::RecoveryUpdateResult recovery_result;
                if (fusion_enabled_)
                {
                    std::string rejection_reason;
                    if (!tracking_ready_ || reset_pending_ || fusion_generation != fusion_generation_)
                        rejection_reason = "reset_during_registration";
                    else if (localization_fitness <= threshold_fitness_)
                        rejection_reason = "fitness_gate";
                    else if (!std::isfinite(eva_result2.inlier_rmse_) ||
                        eva_result2.inlier_rmse_ > fusion_max_icp_rmse_)
                        rejection_reason = "rmse_gate";
                    else if (correspondence_count < static_cast<std::size_t>(fusion_min_correspondences_))
                        rejection_reason = "correspondence_gate";
                    else if (open3d_loc::pointToPlaneInformationRatio(*source, *target, eva_result2,
                        reg_matrix) < min_information_ratio_)
                        rejection_reason = "degenerate_geometry";
                    else if (!std::isfinite(measurement_age) || measurement_age < 0.0 ||
                        measurement_age > fusion_max_measurement_age_)
                        rejection_reason = "stale_measurement";

                    if (rejection_reason.empty())
                    {
                        update_result = correction_filter_.update(reg_matrix, measurement_covariance);
                        if (update_result.accepted)
                        {
                            mat_odom2map_ = correction_filter_.pose();
                            ApplyHeightBoundsLocked();
                            fusion_rejection_count_ = 0;
                            correction_accepted = true;
                            diagnostic_reason = update_result.reason;
                            RCLCPP_INFO(this->get_logger(),
                                "Accepted ICP correction: fitness=%.3f rmse=%.3f innovation=%.3fm/%.3frad mahalanobis=%.3f",
                                localization_fitness, eva_result2.inlier_rmse_,
                                update_result.translation_innovation, update_result.rotation_innovation,
                                update_result.mahalanobis_distance);
                        }
                        else
                        {
                            rejection_reason = update_result.reason;
                            normal_rejection_reason = update_result.reason;
                            if (fusion_config_.recovery.enabled &&
                                IsRecoverableFilterRejection(update_result.reason))
                            {
                                MotionHistorySample motion_sample;
                                if (LookupMotionSampleLocked(scan_stamp, motion_sample))
                                {
                                    recovery_result = correction_filter_.observeRecoveryCandidate(
                                        reg_matrix, measurement_covariance, scan_stamp.seconds(),
                                        motion_sample.cumulative_distance,
                                        motion_sample.cumulative_rotation);
                                    diagnostic_reason = recovery_result.reason;
                                    if (recovery_result.correction_applied)
                                    {
                                        mat_odom2map_ = correction_filter_.pose();
                                        ApplyHeightBoundsLocked();
                                        force_submap_refresh_ = true;
                                        fusion_rejection_count_ = 0;
                                        recovery_applied = true;
                                        RCLCPP_WARN(this->get_logger(),
                                            "Applied bounded recovery correction: translation=%.3fm rotation=%.3frad",
                                            recovery_result.applied_translation,
                                            recovery_result.applied_rotation);
                                    }
                                }
                                else
                                {
                                    correction_filter_.clearRecovery();
                                    diagnostic_reason = "recovery_motion_unavailable";
                                }
                            }
                        }
                    }
                    if (!rejection_reason.empty() && !recovery_applied)
                    {
                        ++fusion_rejection_count_;
                        if (diagnostic_reason == "not_evaluated")
                            diagnostic_reason = rejection_reason;
                        RCLCPP_WARN(this->get_logger(),
                            "Rejected ICP correction (%s): fitness=%.3f rmse=%.3f correspondences=%zu age=%.3fs",
                            rejection_reason.c_str(), localization_fitness, eva_result2.inlier_rmse_,
                            correspondence_count, measurement_age);
                    }
                }
                else if (tracking_ready_ && !reset_pending_ && fusion_generation == fusion_generation_ &&
                    localization_fitness > threshold_fitness_ &&
                    correspondence_count >= static_cast<std::size_t>(fusion_min_correspondences_) &&
                    eva_result2.inlier_rmse_ <= fusion_max_icp_rmse_ &&
                    open3d_loc::pointToPlaneInformationRatio(*source, *target, eva_result2, reg_matrix) >= min_information_ratio_)
                {
                    mat_odom2map_ = reg_matrix;
                    ApplyHeightBoundsLocked();
                }

                if (fusion_enabled_)
                {
                    diagnostic_msgs::msg::DiagnosticArray diagnostics;
                    diagnostics.header.stamp = this->now();
                    diagnostic_msgs::msg::DiagnosticStatus status;
                    status.name = "open3d_loc/global_correction_fusion";
                    status.hardware_id = "open3d_loc";
                    if (correction_accepted)
                    {
                        status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
                        status.message = "ICP correction accepted";
                    }
                    else if (recovery_applied)
                    {
                        status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
                        status.message = "Bounded recovery correction applied";
                    }
                    else if (fusion_rejection_count_ >= fusion_max_consecutive_rejections_)
                    {
                        status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
                        status.message = "Global correction degraded";
                    }
                    else
                    {
                        status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
                        status.message = "ICP correction rejected";
                    }
                    status.values.push_back(DiagnosticValue("reason", diagnostic_reason));
                    status.values.push_back(DiagnosticValue(
                        "fusion_mode",
                        open3d_loc::GlobalCorrectionFilter::recoveryModeName(
                            correction_filter_.recoveryMode())));
                    status.values.push_back(DiagnosticValue(
                        "normal_rejection_reason", normal_rejection_reason));
                    status.values.push_back(DiagnosticValue("fitness", std::to_string(localization_fitness)));
                    status.values.push_back(DiagnosticValue("rmse", std::to_string(eva_result2.inlier_rmse_)));
                    status.values.push_back(DiagnosticValue(
                        "correspondences", std::to_string(correspondence_count)));
                    status.values.push_back(DiagnosticValue("measurement_age", std::to_string(measurement_age)));
                    status.values.push_back(DiagnosticValue(
                        "consecutive_rejections", std::to_string(fusion_rejection_count_)));
                    status.values.push_back(DiagnosticValue(
                        "recovery_candidate_count",
                        std::to_string(correction_filter_.recoveryCandidateCount())));
                    status.values.push_back(DiagnosticValue(
                        "recovery_candidate_interval",
                        std::to_string(recovery_result.candidate_interval)));
                    status.values.push_back(DiagnosticValue(
                        "recovery_motion_distance",
                        std::to_string(recovery_result.motion_distance)));
                    status.values.push_back(DiagnosticValue(
                        "recovery_motion_rotation",
                        std::to_string(recovery_result.motion_rotation)));
                    status.values.push_back(DiagnosticValue(
                        "recovery_candidate_translation_delta",
                        std::to_string(recovery_result.candidate_translation_delta)));
                    status.values.push_back(DiagnosticValue(
                        "recovery_candidate_rotation_delta",
                        std::to_string(recovery_result.candidate_rotation_delta)));
                    status.values.push_back(DiagnosticValue(
                        "recovery_consistency_mahalanobis",
                        std::to_string(recovery_result.consistency_mahalanobis_distance)));
                    status.values.push_back(DiagnosticValue(
                        "recovery_applied_translation",
                        std::to_string(recovery_result.applied_translation)));
                    status.values.push_back(DiagnosticValue(
                        "recovery_applied_rotation",
                        std::to_string(recovery_result.applied_rotation)));
                    status.values.push_back(DiagnosticValue(
                        "previous_localization_duration_ms",
                        std::to_string(last_localization_duration_ms_)));
                    status.values.push_back(DiagnosticValue(
                        "localization_overrun_count",
                        std::to_string(localization_overrun_count_)));
                    AppendHeightDiagnosticsLocked(status);
                    diagnostics.status.push_back(std::move(status));
                    pub_fusion_diagnostics_->publish(diagnostics);
                }
            }

            // save_path
            if (save_scan_)
            {
                pcd_scan->Transform(mat_baselink2odom_cur.inverse());
                pcd_scan2map->Transform(mat_baselink2map_cur.inverse());
                open3d::io::WritePointCloud(save_path + std::to_string(scan_count) + "_ori.ply", *pcd_scan);
                open3d::io::WritePointCloud(save_path + std::to_string(scan_count) + "_crop.ply", *pcd_scan2map);
                scan_count += 1;
            }

            auto loc_e = std::chrono::high_resolution_clock::now(); /// 结束定位计时
            time_last_loc = loc_e;
            loc_cost = std::chrono::duration_cast<std::chrono::microseconds>(loc_e - loc_s).count() / 1000.0;
            last_localization_duration_ms_ = loc_cost;
            if (loc_cost > loc_frequence_ * 1000.0)
            {
                ++localization_overrun_count_;
                RCLCPP_WARN(this->get_logger(),
                    "Localization overrun: %.1fms exceeds %.1fms interval",
                    loc_cost, loc_frequence_ * 1000.0);
            }
            RCLCPP_INFO(this->get_logger(), "localization cost: %f ms", loc_cost);
        }
    }
}

void GloabalLocalization::ReportRegistrationFailure(const std::string & reason)
{
    {
        std::lock_guard<std::mutex> lock(lock_state_);
        loc_fitness_ = 0.0;
        if (fusion_enabled_)
        {
            ++fusion_rejection_count_;
            correction_filter_.clearRecovery();
        }
    }
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "Skipping localization registration: %s. Manual reset via /initialpose remains available.",
        reason.c_str());
    diagnostic_msgs::msg::DiagnosticArray diagnostics;
    diagnostics.header.stamp = this->now();
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "localization_registration";
    status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
    status.message = "Registration unavailable; manual localization reset remains available";
    status.values.push_back(DiagnosticValue("reason", reason));
    {
        std::lock_guard<std::mutex> lock(lock_state_);
        AppendHeightDiagnosticsLocked(status);
    }
    diagnostics.status.push_back(std::move(status));
    pub_fusion_diagnostics_->publish(diagnostics);
}

void GloabalLocalization::StartLoc()
{
    thread_loc_ = std::thread(&GloabalLocalization::Localization, this);
}

void GloabalLocalization::CallbackInitialPose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr initialpose)
{
    const auto & pose = initialpose->pose.pose;
    Eigen::Quaterniond quaternion(pose.orientation.w, pose.orientation.x,
        pose.orientation.y, pose.orientation.z);
    if (initialpose->header.frame_id != "map" || !std::isfinite(pose.position.x) ||
        !std::isfinite(pose.position.y) || !quaternion.coeffs().allFinite() ||
        !std::isfinite(quaternion.norm()) || quaternion.norm() < 1e-9) {
        RCLCPP_ERROR(get_logger(), "Invalid initialpose: expected finite XY, nonzero quaternion and map frame");
        return;
    }
    quaternion.normalize();
    const Eigen::Matrix3d rotation = quaternion.toRotationMatrix();
    const double yaw = std::atan2(rotation(1, 0), rotation(0, 0));
    Eigen::Matrix4d desired;
    {
        std::lock_guard<std::mutex> lock(lock_state_);
        desired = have_trusted_body_pose_ ? trusted_body_pose_ : mat_initialpose_;
        if (!have_trusted_body_pose_ && height_bounds_.enabled)
            desired(2, 3) = height_bounds_.floor_z + std::clamp(
                    mat_initialpose_(2, 3) - height_bounds_.floor_z,
                    height_bounds_.min_height, height_bounds_.max_height);
        if (height_bounds_.enabled)
            desired(2, 3) = height_bounds_.floor_z + std::clamp(
                desired(2, 3) - height_bounds_.floor_z, height_bounds_.min_height, height_bounds_.max_height);
    }
    desired(0, 3) = pose.position.x;
    desired(1, 3) = pose.position.y;
    desired.block<3, 3>(0, 0) = open3d_loc::levelOrientationFromYaw(yaw);
    BeginReset(desired);
    RCLCPP_INFO(get_logger(), "Initialpose accepted: resetting FAST-LIO before relocalization");
}

bool GloabalLocalization::LookupMotionSampleLocked(
    const rclcpp::Time & stamp, MotionHistorySample & sample) const
{
    for (auto iterator = motion_history_.rbegin(); iterator != motion_history_.rend(); ++iterator)
    {
        if (iterator->stamp <= stamp)
        {
            sample = *iterator;
            return true;
        }
    }
    return false;
}

double GloabalLocalization::ComputeMotionDis(const Eigen::Vector3d &a, const Eigen::Vector3d &b)
{
    return std::sqrt(std::pow(a.x() - b.x(), 2) + std::pow(a.y() - b.y(), 2) + std::pow(a.z() - b.z(), 2));
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GloabalLocalization>();

    // 使用多线程执行器，可以指定线程数
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
