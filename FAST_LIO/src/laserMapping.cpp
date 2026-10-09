// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <cmath>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <chrono>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
// #include <livox_interfaces/msg/custom_msg.hpp>
#include "odometry_utils.hpp"
#include "tracking_guard.hpp"
#include "imu_receiver.hpp"
#include "imu_sync.hpp"
#include <atomic>
#include <fast_lio/msg/tracking_status.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <new>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
bool publish_tf_en = true;
string odom_frame_id = "odom", body_frame_id = "body";
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;
double sensor_time_offset_to_ros_sec = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0;
std::atomic<int> publish_count{0};
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool point_selected_surf[100000] = {0};
bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool is_first_lidar = true;

vector<vector<int>> pointSearchInd_surf;
vector<BoxPointType> cub_needrm;
vector<PointVector> Nearest_Points;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
deque<double> time_buffer;
deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;
fast_lio::TrackingGuard tracking;
bool measurement_valid = true;
std::string measurement_reason;
std::string timing_fault;
std::uint64_t sensor_epoch = 0;
std::size_t imu_queue_high_water = 0;
double received_imu_gap = 0.0, received_arrival_gap = 0.0;
std::chrono::steady_clock::time_point last_imu_arrival;
fast_lio::ImuCoverage sync_coverage;
double sync_previous_imu = -1.0, sync_wait_timeout = 0.1, sync_wait_elapsed = 0.0;
bool startup_imu_ready = false;
std::chrono::steady_clock::time_point sync_wait_started;
double measurement_rms = 0.0, measurement_information_ratio = 0.0;
sensor_msgs::msg::PointCloud2 held_world_cloud, held_body_cloud;
geometry_msgs::msg::TransformStamped held_transform;
nav_msgs::msg::Odometry held_odometry;
bool have_held_odometry = false;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());

rclcpp::Clock &diagnostic_clock()
{
    static rclcpp::Clock clock(RCL_STEADY_TIME);
    return clock;
}

bool current_lidar_stamp(builtin_interfaces::msg::Time &stamp)
{
    if (try_get_ros_time(lidar_end_time, stamp))
        return true;
    RCLCPP_ERROR_THROTTLE(rclcpp::get_logger("fastlio_mapping"), diagnostic_clock(), 2000,
                          "Refusing to publish invalid LiDAR timestamp: %.9f", lidar_end_time);
    return false;
}
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig)
{
    flg_exit = true;
    std::cout << "catch sig %d" << sig << std::endl;
    sig_buffer.notify_all();
    rclcpp::shutdown();
}

inline void dump_lio_state_to_log(FILE *fp)
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                            // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2));    // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // omega
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2));    // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // Acc
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));       // Bias_g
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));       // Bias_a
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a
    fprintf(fp, "\r\n");
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const *const pi, PointType *const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized)
    {
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }
    if (!need_move)
        return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if (cub_needrm.size() > 0)
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg)
{
    if (msg->header.stamp.sec < 0)
    {
        RCLCPP_WARN_THROTTLE(rclcpp::get_logger("fastlio_mapping"), diagnostic_clock(), 2000,
                             "Dropping LiDAR message with a negative header timestamp");
        return;
    }
    const double source_timestamp = get_time_sec(msg->header.stamp);
    double cur_time = 0.0;
    if (!try_apply_time_offset(source_timestamp, sensor_time_offset_to_ros_sec, cur_time))
    {
        RCLCPP_WARN_THROTTLE(rclcpp::get_logger("fastlio_mapping"), diagnostic_clock(), 2000,
                             "Dropping LiDAR message with invalid corrected timestamp: %.9f",
                             source_timestamp + sensor_time_offset_to_ros_sec);
        return;
    }
    const double preprocess_start_time = omp_get_wtime();
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    if (!p_pre->process(msg, ptr))
    {
        RCLCPP_WARN_THROTTLE(rclcpp::get_logger("fastlio_mapping"), diagnostic_clock(), 2000,
                             "Dropping LiDAR message with an invalid schema, timing, or no usable points");
        return;
    }
    mtx_buffer.lock();
    if (scan_count < MAXN - 1) ++scan_count;
    if (!is_first_lidar && cur_time <= last_timestamp_lidar)
    {
        timing_fault = "timestamp_regression";
        std::cerr << "sensor timestamp regression" << std::endl;
        lidar_buffer.clear(); time_buffer.clear(); lidar_pushed = false;
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }

    if (lidar_buffer.size() >= 50) {
        timing_fault = "lidar_buffer_overflow";
        lidar_buffer.clear(); time_buffer.clear(); lidar_pushed = false;
    }
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(cur_time);
    last_timestamp_lidar = cur_time;
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg)
// void livox_pcl_cbk(const livox_interfaces::msg::CustomMsg::UniquePtr msg)
{
    const double source_timestamp = get_time_sec(msg->header.stamp);
    double cur_time = 0.0;
    if (!try_apply_time_offset(source_timestamp, sensor_time_offset_to_ros_sec, cur_time))
    {
        RCLCPP_WARN_THROTTLE(rclcpp::get_logger("fastlio_mapping"), diagnostic_clock(), 2000,
                             "Dropping LiDAR message with invalid corrected timestamp: %.9f",
                             source_timestamp + sensor_time_offset_to_ros_sec);
        return;
    }
    const double preprocess_start_time = omp_get_wtime();
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    mtx_buffer.lock();
    if (scan_count < MAXN - 1) ++scan_count;
    if (!is_first_lidar && cur_time <= last_timestamp_lidar)
    {
        timing_fault = "timestamp_regression";
        std::cerr << "sensor timestamp regression" << std::endl;
        lidar_buffer.clear(); time_buffer.clear(); lidar_pushed = false;
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }
    last_timestamp_lidar = cur_time;

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    if (lidar_buffer.size() >= 50) {
        timing_fault = "lidar_buffer_overflow";
        lidar_buffer.clear(); time_buffer.clear(); lidar_pushed = false;
    }
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in)
{
    std::uint64_t epoch;
    double alignment;
    {
        std::lock_guard<std::mutex> lock(mtx_buffer);
        epoch = sensor_epoch;
        alignment = timediff_lidar_wrt_imu;
    }
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));

    if (msg_in->header.stamp.sec < 0)
    {
        RCLCPP_WARN_THROTTLE(rclcpp::get_logger("fastlio_mapping"), diagnostic_clock(), 2000,
                             "Dropping IMU message with a negative header timestamp");
        return;
    }
    const double source_timestamp = get_time_sec(msg_in->header.stamp);
    double corrected_timestamp = source_timestamp - time_diff_lidar_to_imu;
    if (abs(alignment) > 0.1 && time_sync_en)
    {
        corrected_timestamp = alignment + source_timestamp;
    }
    double ros_timestamp = 0.0;
    if (!try_apply_time_offset(
            corrected_timestamp, sensor_time_offset_to_ros_sec, ros_timestamp))
    {
        RCLCPP_WARN_THROTTLE(rclcpp::get_logger("fastlio_mapping"), diagnostic_clock(), 2000,
                             "Dropping IMU message with invalid corrected timestamp: %.9f", corrected_timestamp);
        return;
    }
    if (!try_get_ros_time(ros_timestamp, msg->header.stamp))
    {
        RCLCPP_WARN_THROTTLE(rclcpp::get_logger("fastlio_mapping"), diagnostic_clock(), 2000,
                             "Dropping IMU message with an out-of-range timestamp: %.9f", corrected_timestamp);
        return;
    }

    if (!V3D(msg->linear_acceleration.x, msg->linear_acceleration.y,
            msg->linear_acceleration.z).allFinite() ||
        !V3D(msg->angular_velocity.x, msg->angular_velocity.y,
            msg->angular_velocity.z).allFinite())
    {
        std::lock_guard<std::mutex> lock(mtx_buffer);
        if (epoch == sensor_epoch) timing_fault = "nonfinite_imu";
        return;
    }
    double timestamp = get_time_sec(msg->header.stamp);

    std::lock_guard<std::mutex> lock(mtx_buffer);
    if (epoch != sensor_epoch) return;
    ++publish_count;
    const auto arrival = std::chrono::steady_clock::now();
    received_arrival_gap = last_timestamp_imu >= 0.0 ?
        std::chrono::duration<double>(arrival - last_imu_arrival).count() : 0.0;
    received_imu_gap = last_timestamp_imu >= 0.0 ? timestamp - last_timestamp_imu : 0.0;
    last_imu_arrival = arrival;

    if (last_timestamp_imu >= 0.0 && timestamp <= last_timestamp_imu)
    {
        timing_fault = "timestamp_regression";
        std::cerr << "sensor timestamp regression" << std::endl;
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    if (imu_buffer.size() >= 2000) {
        timing_fault = "imu_buffer_overflow";
        imu_buffer.clear();
    }
    imu_buffer.push_back(msg);
    imu_queue_high_water = std::max(imu_queue_high_water, imu_buffer.size());
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    std::lock_guard<std::mutex> lock(mtx_buffer);
    if (lidar_buffer.empty())
    {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (!std::isfinite(meas.lidar_beg_time) || meas.lidar_beg_time < 0.0)
        {
            RCLCPP_WARN_THROTTLE(rclcpp::get_logger("fastlio_mapping"), diagnostic_clock(), 2000,
                                 "Dropping LiDAR scan with invalid header timestamp: %.9f", meas.lidar_beg_time);
            lidar_buffer.pop_front();
            time_buffer.pop_front();
            return false;
        }
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            std::cerr << "Too few input point cloud!\n";
        }
        else
        {
            const double scan_duration = meas.lidar->points.back().curvature / double(1000);
            const double max_scan_duration = p_pre->max_scan_duration_ms > 0.0
                                                 ? p_pre->max_scan_duration_ms / 1000.0
                                                 : 2.0 / std::max(1, p_pre->SCAN_RATE);
            if (!std::isfinite(scan_duration) || scan_duration < 0.0 ||
                (scan_duration > max_scan_duration))
            {
                RCLCPP_WARN_THROTTLE(rclcpp::get_logger("fastlio_mapping"), diagnostic_clock(), 2000,
                                     "Dropping LiDAR scan with invalid point time offset: %.9f", scan_duration);
                lidar_buffer.pop_front();
                time_buffer.pop_front();
                return false;
            }
            if (scan_duration < 0.5 * lidar_mean_scantime)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            }
            else
            {
                scan_num++;
                lidar_end_time = meas.lidar_beg_time + scan_duration;
                lidar_mean_scantime += (scan_duration - lidar_mean_scantime) / scan_num;
            }
        }

        if (!std::isfinite(lidar_end_time) || lidar_end_time < 0.0)
        {
            RCLCPP_WARN_THROTTLE(rclcpp::get_logger("fastlio_mapping"), diagnostic_clock(), 2000,
                                 "Dropping LiDAR scan with invalid end timestamp: %.9f", lidar_end_time);
            lidar_buffer.pop_front();
            time_buffer.pop_front();
            return false;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
        sync_wait_started = std::chrono::steady_clock::now();
    }

    sync_wait_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - sync_wait_started).count();
    std::vector<double> stamps;
    stamps.reserve(imu_buffer.size());
    for (const auto & imu : imu_buffer) stamps.push_back(get_time_sec(imu->header.stamp));
    sync_coverage = fast_lio::imuCoverage(stamps, meas.lidar_beg_time, lidar_end_time,
        sync_previous_imu, tracking.limits.max_imu_gap, sync_wait_elapsed,
        sync_wait_timeout, !startup_imu_ready);
    if (sync_coverage.action == fast_lio::ImuSyncAction::WAITING) return false;
    if (sync_coverage.action == fast_lio::ImuSyncAction::INVALID) {
        // Keep the established public reason for end-of-scan/arrival coverage failures.
        tracking.lose(sync_coverage.reason == "imu_arrival_timeout" ||
            sync_coverage.reason == "imu_scan_end_gap" ? "missing_imu_coverage" : sync_coverage.reason);
        return false;
    }
    if (sync_coverage.action == fast_lio::ImuSyncAction::SKIP_SCAN) {
        lidar_buffer.pop_front(); time_buffer.pop_front(); lidar_pushed = false;
        return false;
    }
    if (!startup_imu_ready) {
        while (!imu_buffer.empty() &&
            get_time_sec(imu_buffer.front()->header.stamp) < sync_coverage.first)
            imu_buffer.pop_front();
        startup_imu_ready = true;
    }

    /*** push imu data, and pop from imu buffer ***/
    meas.imu.clear();
    meas.gyro_at_lidar_end_valid = false;
    while (!imu_buffer.empty())
    {
        const double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
        if (imu_time > lidar_end_time)
            break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    if (!meas.imu.empty())
    {
        const auto &earlier_imu = meas.imu.back();
        const double earlier_time = get_time_sec(earlier_imu->header.stamp);
        const auto &earlier_gyro = earlier_imu->angular_velocity;
        const V3D earlier_measurement(
            earlier_gyro.x, earlier_gyro.y, earlier_gyro.z);

        if (std::abs(earlier_time - lidar_end_time) <= 1e-9)
        {
            meas.gyro_at_lidar_end = earlier_measurement;
            meas.gyro_at_lidar_end_valid = earlier_measurement.allFinite();
        }
        else if (!imu_buffer.empty())
        {
            const auto &later_imu = imu_buffer.front();
            const auto &later_gyro = later_imu->angular_velocity;
            const V3D later_measurement(later_gyro.x, later_gyro.y, later_gyro.z);
            const double later_time = get_time_sec(later_imu->header.stamp);
            if (fast_lio::imuGapExceeded(later_time - earlier_time, tracking.limits.max_imu_gap)) {
                meas.gyro_at_lidar_end = earlier_measurement;
                meas.gyro_at_lidar_end_valid = earlier_measurement.allFinite();
            } else meas.gyro_at_lidar_end_valid = fast_lio::interpolateAngularVelocity(
                earlier_time, earlier_measurement,
                get_time_sec(later_imu->header.stamp), later_measurement,
                lidar_end_time, meas.gyro_at_lidar_end);
        }
    }

    sync_previous_imu = get_time_sec(meas.imu.back()->header.stamp);
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(feats_down_world->points[i], mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                if (points_near.size() < NUM_MATCH_POINTS)
                    break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull)
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        if (!current_lidar_stamp(laserCloudmsg.header.stamp))
            return;
        laserCloudmsg.header.frame_id = odom_frame_id;
        held_world_cloud = laserCloudmsg;
        pubLaserCloudFull->publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    /*
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
    */
}

void publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i],
                               &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    if (!current_lidar_stamp(laserCloudmsg.header.stamp))
        return;
    laserCloudmsg.header.frame_id = body_frame_id;
    held_body_cloud = laserCloudmsg;
    pubLaserCloudFull_body->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld(
        new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i],
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    if (!current_lidar_stamp(laserCloudFullRes3.header.stamp))
        return;
    laserCloudFullRes3.header.frame_id = odom_frame_id;
    pubLaserCloudEffect->publish(laserCloudFullRes3);
}

void publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap)
{
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(
        new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                            &laserCloudWorld->points[i]);
    }
    *pcl_wait_pub += *laserCloudWorld;

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
    if (!current_lidar_stamp(laserCloudmsg.header.stamp))
        return;
    laserCloudmsg.header.frame_id = odom_frame_id;
    pubLaserCloudMap->publish(laserCloudmsg);

    // sensor_msgs::msg::PointCloud2 laserCloudMap;
    // pcl::toROSMsg(*featsFromMap, laserCloudMap);
    // laserCloudMap.header.stamp = get_ros_time(lidar_end_time);
    // laserCloudMap.header.frame_id = "odom";
    // pubLaserCloudMap->publish(laserCloudMap);
}

void save_to_pcd()
{
    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(map_file_path, *pcl_wait_pub);
}

template <typename T>
void set_posestamp(T &out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
}

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, std::unique_ptr<tf2_ros::TransformBroadcaster> &tf_br)
{
    odomAftMapped.header.frame_id = odom_frame_id;
    odomAftMapped.child_frame_id = body_frame_id;
    if (!current_lidar_stamp(odomAftMapped.header.stamp))
        return;

    V3D gyro_measurement_body;
    if (!p_imu->last_gyro_measurement(gyro_measurement_body))
    {
        RCLCPP_WARN_THROTTLE(rclcpp::get_logger("fastlio_mapping"), diagnostic_clock(), 2000,
                             "Refusing to publish odometry without a time-aligned IMU angular velocity");
        return;
    }

    const M3D world_from_body = state_point.rot.toRotationMatrix();
    const Eigen::Vector3d linear_velocity_world = state_point.vel;
    const Eigen::Vector3d gyro_bias_body = state_point.bg;
    const fast_lio::BodyTwist body_twist = fast_lio::bodyTwistFromState(
        world_from_body, linear_velocity_world, gyro_measurement_body, gyro_bias_body);
    const auto state_covariance = kf.get_P();
    const fast_lio::PoseCovariance pose_covariance =
        fast_lio::poseCovarianceFromState(state_covariance);
    const fast_lio::TwistCovariance twist_covariance =
        fast_lio::bodyTwistCovarianceFromState(
            state_covariance, world_from_body, gyr_cov * Eigen::Matrix3d::Identity());

    set_posestamp(odomAftMapped.pose);
    odomAftMapped.twist.twist.linear.x = body_twist.linear.x();
    odomAftMapped.twist.twist.linear.y = body_twist.linear.y();
    odomAftMapped.twist.twist.linear.z = body_twist.linear.z();
    odomAftMapped.twist.twist.angular.x = body_twist.angular.x();
    odomAftMapped.twist.twist.angular.y = body_twist.angular.y();
    odomAftMapped.twist.twist.angular.z = body_twist.angular.z();
    for (int row = 0; row < 6; ++row)
    {
        for (int column = 0; column < 6; ++column)
        {
            const std::size_t index = static_cast<std::size_t>(row * 6 + column);
            odomAftMapped.pose.covariance[index] = pose_covariance(row, column);
            odomAftMapped.twist.covariance[index] = twist_covariance(row, column);
        }
    }
    held_odometry = odomAftMapped;
    have_held_odometry = true;
    pubOdomAftMapped->publish(odomAftMapped);

    geometry_msgs::msg::TransformStamped trans;
    trans.header.frame_id = odom_frame_id;
    trans.header.stamp = odomAftMapped.header.stamp;
    trans.child_frame_id = body_frame_id;
    trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
    trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
    trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
    trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
    held_transform = trans;
    if (publish_tf_en)
        tf_br->sendTransform(trans);
}

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
    set_posestamp(msg_body_pose);
    if (!current_lidar_stamp(msg_body_pose.header.stamp))
        return;
    msg_body_pose.header.frame_id = odom_frame_id;

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0)
    {
        path.poses.push_back(msg_body_pose);
        pubPath->publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    if (!measurement_valid) {
        ekfom_data.valid = false;
        return;
    }
    double match_start = omp_get_wtime();
    laserCloudOri->clear();
    corr_normvect->clear();
    total_residual = 0.0;

/** closest surface search and residual computation **/
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body = feats_down_body->points[i];
        PointType &point_world = feats_down_world->points[i];

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                                                                                                : true;
        }

        if (!point_selected_surf[i])
            continue;

        VF(4)
        pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }

    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->push_back(feats_down_body->points[i]);
            corr_normvect->push_back(normvec->points[i]);
            total_residual += res_last[i];
            effct_feat_num++;
        }
    }

    if (effct_feat_num < tracking.limits.min_features)
    {
        measurement_valid = false;
        measurement_reason = "insufficient_features";
        ekfom_data.valid = false;
        std::cerr << "No Effective Points!" << std::endl;
        // ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time += omp_get_wtime() - match_start;
    double solve_start_ = omp_get_wtime();

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); // 23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() * norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); // s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    double radius_squared = 0.0;
    for (int i = 0; i < effct_feat_num; ++i)
        radius_squared += laserCloudOri->points[i].getVector3fMap().squaredNorm();
    measurement_rms = std::sqrt(ekfom_data.h.squaredNorm() / effct_feat_num);
    measurement_information_ratio = fast_lio::informationRatio(
        ekfom_data.h_x, std::sqrt(radius_squared / effct_feat_num));
    measurement_reason = fast_lio::measurementRejection(tracking.limits, effct_feat_num,
        feats_down_size, measurement_rms, measurement_information_ratio);
    if (!measurement_reason.empty())
    {
        measurement_valid = false;
        ekfom_data.valid = false;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

class LaserMappingNode : public rclcpp::Node
{
public:
    LaserMappingNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions()) : Node("laser_mapping", options)
    {
        rcl_interfaces::msg::ParameterDescriptor tracking_descriptor;
        tracking_descriptor.read_only = true;
        tracking_descriptor.description = "Startup-only tracking protection limit";
        const auto imu_reliability = declare_parameter<std::string>(
            "tracking.imu_reliability", "best_effort", tracking_descriptor);
        if (imu_reliability != "best_effort" && imu_reliability != "reliable")
            throw std::invalid_argument("tracking.imu_reliability must be best_effort or reliable");
        const int imu_queue_depth = declare_parameter<int>("tracking.imu_queue_depth", 200, tracking_descriptor);
        sync_wait_timeout = declare_parameter<double>("tracking.sync_wait_timeout", 0.1, tracking_descriptor);
        startup_warning_timeout_ = declare_parameter<double>("tracking.startup_warning_timeout", 10.0, tracking_descriptor);
        if (imu_queue_depth <= 0 || imu_queue_depth > 2000 ||
            !std::isfinite(sync_wait_timeout) || sync_wait_timeout <= 0.0 ||
            !std::isfinite(startup_warning_timeout_) || startup_warning_timeout_ <= 0.0)
            throw std::invalid_argument("invalid IMU queue or startup synchronization limits");
        tracking.limits.min_features = declare_parameter<int>("tracking.min_features", 100, tracking_descriptor);
        tracking.limits.min_feature_ratio = declare_parameter<double>("tracking.min_feature_ratio", 0.2, tracking_descriptor);
        tracking.limits.max_residual = declare_parameter<double>("tracking.max_residual_rms", 0.15, tracking_descriptor);
        tracking.limits.min_information_ratio = declare_parameter<double>("tracking.min_information_ratio", 1e-4, tracking_descriptor);
        tracking.limits.max_imu_gap = declare_parameter<double>("tracking.max_imu_gap", 0.05, tracking_descriptor);
        tracking.limits.prediction_timeout = declare_parameter<double>("tracking.prediction_timeout", 0.5, tracking_descriptor);
        tracking.limits.max_speed = declare_parameter<double>("tracking.max_speed", 3.0, tracking_descriptor);
        tracking.limits.max_angular_speed = declare_parameter<double>("tracking.max_angular_speed", 3.0, tracking_descriptor);
        tracking.limits.validate();
        if (sync_wait_timeout > tracking.limits.prediction_timeout)
            throw std::invalid_argument("sync_wait_timeout must not exceed prediction_timeout");
        this->declare_parameter<bool>("publish.path_en", true);
        this->declare_parameter<bool>("publish.effect_map_en", false);
        this->declare_parameter<bool>("publish.map_en", false);
        this->declare_parameter<bool>("publish.scan_publish_en", true);
        this->declare_parameter<bool>("publish.dense_publish_en", true);
        this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
        this->declare_parameter<bool>("publish.tf_en", true);
        this->declare_parameter<string>("publish.odom_frame", "odom");
        this->declare_parameter<string>("publish.body_frame", "body");
        this->declare_parameter<int>("max_iteration", 4);
        this->declare_parameter<string>("map_file_path", "");
        this->declare_parameter<string>("common.lid_topic", "/livox/lidar");
        this->declare_parameter<string>("common.imu_topic", "/livox/imu");
        this->declare_parameter<bool>("common.time_sync_en", false);
        this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
        this->declare_parameter<double>("common.sensor_time_offset_to_ros_sec", 0.0);
        this->declare_parameter<double>("filter_size_corner", 0.5);
        this->declare_parameter<double>("filter_size_surf", 0.5);
        this->declare_parameter<double>("filter_size_map", 0.5);
        this->declare_parameter<double>("cube_side_length", 200.);
        this->declare_parameter<float>("mapping.det_range", 300.);
        this->declare_parameter<double>("mapping.fov_degree", 180.);
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<double>("preprocess.blind", 0.01);
        this->declare_parameter<int>("preprocess.lidar_type", AVIA);
        this->declare_parameter<int>("preprocess.scan_line", 16);
        this->declare_parameter<int>("preprocess.timestamp_unit", US);
        this->declare_parameter<int>("preprocess.scan_rate", 10);
        this->declare_parameter<double>("preprocess.max_scan_duration_ms", 0.0);
        this->declare_parameter<int>("point_filter_num", 2);
        this->declare_parameter<bool>("feature_extract_enable", false);
        this->declare_parameter<bool>("runtime_pos_log_enable", false);
        this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
        this->declare_parameter<bool>("pcd_save.pcd_save_en", false);
        this->declare_parameter<int>("pcd_save.interval", -1);
        this->declare_parameter<vector<double>>("mapping.extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_R", vector<double>());

        this->get_parameter_or<bool>("publish.path_en", path_en, true);
        this->get_parameter_or<bool>("publish.effect_map_en", effect_pub_en, false);
        this->get_parameter_or<bool>("publish.map_en", map_pub_en, false);
        this->get_parameter_or<bool>("publish.scan_publish_en", scan_pub_en, true);
        this->get_parameter_or<bool>("publish.dense_publish_en", dense_pub_en, true);
        this->get_parameter_or<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        this->get_parameter_or<bool>("publish.tf_en", publish_tf_en, true);
        this->get_parameter_or<string>("publish.odom_frame", odom_frame_id, "odom");
        this->get_parameter_or<string>("publish.body_frame", body_frame_id, "body");
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        this->get_parameter_or<string>("map_file_path", map_file_path, "");
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/livox/lidar");
        this->get_parameter_or<string>("common.imu_topic", imu_topic, "/livox/imu");
        this->get_parameter_or<bool>("common.time_sync_en", time_sync_en, false);
        this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
        this->get_parameter_or<double>(
            "common.sensor_time_offset_to_ros_sec", sensor_time_offset_to_ros_sec, 0.0);
        if (!std::isfinite(sensor_time_offset_to_ros_sec))
        {
            RCLCPP_WARN(this->get_logger(),
                        "Ignoring non-finite common.sensor_time_offset_to_ros_sec");
            sensor_time_offset_to_ros_sec = 0.0;
        }
        this->get_parameter_or<double>("filter_size_corner", filter_size_corner_min, 0.5);
        this->get_parameter_or<double>("filter_size_surf", filter_size_surf_min, 0.5);
        this->get_parameter_or<double>("filter_size_map", filter_size_map_min, 0.5);
        this->get_parameter_or<double>("cube_side_length", cube_len, 200.f);
        this->get_parameter_or<float>("mapping.det_range", DET_RANGE, 300.f);
        this->get_parameter_or<double>("mapping.fov_degree", fov_deg, 180.f);
        this->get_parameter_or<double>("mapping.gyr_cov", gyr_cov, 0.1);
        this->get_parameter_or<double>("mapping.acc_cov", acc_cov, 0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov", b_gyr_cov, 0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov", b_acc_cov, 0.0001);
        this->get_parameter_or<double>("preprocess.blind", p_pre->blind, 0.01);
        this->get_parameter_or<int>("preprocess.lidar_type", p_pre->lidar_type, AVIA);
        this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
        this->get_parameter_or<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
        this->get_parameter_or<double>("preprocess.max_scan_duration_ms", p_pre->max_scan_duration_ms, 0.0);
        this->get_parameter_or<int>("point_filter_num", p_pre->point_filter_num, 2);
        this->get_parameter_or<bool>("feature_extract_enable", p_pre->feature_enabled, false);
        this->get_parameter_or<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
        this->get_parameter_or<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
        this->get_parameter_or<bool>("pcd_save.pcd_save_en", pcd_save_en, false);
        this->get_parameter_or<int>("pcd_save.interval", pcd_save_interval, -1);
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());

        if (p_pre->N_SCANS < 1 || p_pre->N_SCANS > 128)
            throw std::invalid_argument("preprocess.scan_line must be between 1 and 128");
        if (p_pre->SCAN_RATE <= 0)
            throw std::invalid_argument("preprocess.scan_rate must be positive");
        if (p_pre->point_filter_num <= 0)
            throw std::invalid_argument("point_filter_num must be positive");
        if (p_pre->lidar_type == ROBOSENSE_E1R && p_pre->max_scan_duration_ms < 0.0)
            throw std::invalid_argument("preprocess.max_scan_duration_ms cannot be negative");

        RCLCPP_INFO(this->get_logger(), "p_pre->lidar_type %d", p_pre->lidar_type);

        path.header.stamp = this->get_clock()->now();
        path.header.frame_id = odom_frame_id;

        // /*** variables definition ***/
        // int effect_feat_num = 0, frame_num = 0;
        // double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
        // bool flg_EKF_converged, EKF_stop_flg = 0;

        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

        _featsArray.reset(new PointCloudXYZI());

        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));

        Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

        fill(epsi, epsi + 23, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        if (runtime_pos_log) {
            string pos_log_dir = root_dir + "/Log/pos_log.txt";
            fp = fopen(pos_log_dir.c_str(), "w");
            fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), ios::out);
            fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
            fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"), ios::out);
        }

        /*** ROS subscribe initialization ***/
        if (p_pre->lidar_type == AVIA)
        {
            sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, 20, livox_pcl_cbk);
            // sub_pcl_livox_ = this->create_subscription<livox_interfaces::msg::CustomMsg>(lid_topic, 20, livox_pcl_cbk);
        }
        else
        {
            sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, rclcpp::SensorDataQoS(), standard_pcl_cbk);
        }
        imu_receiver_ = std::make_unique<fast_lio::ImuReceiver>(*this, imu_topic, imu_queue_depth, imu_cbk, imu_reliability == "reliable");
        pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_1", 20);
        pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body_1", 20);
        pubLaserCloudEffect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected_1", 20);
        pubLaserCloudMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map_1", 20);
        pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry_loc", 20);
        pubPath_ = this->create_publisher<nav_msgs::msg::Path>("/path_1", 20);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        //------------------------------------------------------------------------------------------------------
        auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
        timer_ = rclcpp::create_timer(this, this->get_clock(), period_ms, std::bind(&LaserMappingNode::timer_callback, this));

        auto map_period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0));
        map_pub_timer_ = rclcpp::create_timer(this, this->get_clock(), map_period_ms, std::bind(&LaserMappingNode::map_publish_callback, this));

        map_save_srv_ = this->create_service<std_srvs::srv::Trigger>("map_save", std::bind(&LaserMappingNode::map_save_callback, this, std::placeholders::_1, std::placeholders::_2));

        tracking_pub_ = create_publisher<fast_lio::msg::TrackingStatus>(
            "/fast_lio/tracking_status", rclcpp::QoS(1).reliable().transient_local());
        diagnostics_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/fast_lio/diagnostics", 10);
        reset_srv_ = create_service<std_srvs::srv::Trigger>("/fast_lio/reset_tracking",
            [this](const std_srvs::srv::Trigger::Request::SharedPtr,
                std_srvs::srv::Trigger::Response::SharedPtr response) {
                reset_tracking();
                response->success = true;
                response->message = "reset_generation=" + std::to_string(generation_) +
                    "; instance_id=" + std::to_string(instance_id_) +
                    "; waiting for fresh IMU and LiDAR initialization";
            });
        health_timer_ = create_wall_timer(std::chrono::milliseconds(100), [this]() {
            if (tracking.state != fast_lio::TrackingGuard::LOST &&
                (tracking.accepted_stamp >= 0.0 || tracking.prediction_started >= 0.0) &&
                std::chrono::duration<double>(std::chrono::steady_clock::now() - last_accepted_wall_).count()
                    >= tracking.limits.prediction_timeout)
                tracking.lose(tracking.accepted_stamp >= 0.0 ? "prediction_timeout" : "initialization_timeout");
            retry_startup();
            publish_tracking();
            if (tracking.state != fast_lio::TrackingGuard::TRACKING) publish_held();
        });
        publish_tracking();
        RCLCPP_INFO(this->get_logger(), "Node init finished.");
    }

    ~LaserMappingNode()
    {
        if (imu_receiver_) imu_receiver_->stop();
        fout_out.close();
        fout_pre.close();
        if (fp) fclose(fp);
    }

private:
    bool scan_geometry_valid()
    {
        if (feats_down_body->empty()) return false;
        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        for (const auto & point : *feats_down_body) mean += point.getVector3fMap().cast<double>();
        mean /= feats_down_body->size();
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (const auto & point : *feats_down_body) {
            Eigen::Vector3d centered = point.getVector3fMap().cast<double>() - mean;
            if (!centered.allFinite()) return false;
            covariance += centered * centered.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        return solver.info() == Eigen::Success && solver.eigenvalues().maxCoeff() > 0.0 &&
            solver.eigenvalues().minCoeff() / solver.eigenvalues().maxCoeff() >= tracking.limits.min_information_ratio;
    }

    void publish_tracking()
    {
        retry_startup();
        if (processing_scan_) processing_duration_ = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - processing_started_).count();
        fast_lio::msg::TrackingStatus status;
        status.header.stamp = now();
        status.state = tracking.state;
        status.instance_id = instance_id_;
        status.generation = generation_;
        if (tracking.accepted_stamp >= 0.0)
            try_get_ros_time(tracking.accepted_stamp, status.last_accepted_stamp);
        status.reason = tracking.reason;
        status.effective_features = std::max(effct_feat_num, 0);
        status.residual_rms = measurement_rms;
        status.information_ratio = measurement_information_ratio;
        status.prediction_interval = tracking.accepted_stamp < 0.0 ? 0.0 :
            std::max(0.0, lidar_end_time - tracking.accepted_stamp);
        tracking_pub_->publish(status);
        diagnostic_msgs::msg::DiagnosticArray diagnostics;
        diagnostics.header = status.header;
        diagnostic_msgs::msg::DiagnosticStatus entry;
        entry.name = "fast_lio/tracking";
        entry.hardware_id = "fast_lio";
        entry.level = tracking.state == fast_lio::TrackingGuard::TRACKING ? 0 :
            tracking.state == fast_lio::TrackingGuard::LOST ? 2 : 1;
        entry.message = tracking.reason;
        auto add = [&entry](const std::string & key, double value) {
            diagnostic_msgs::msg::KeyValue pair;
            pair.key = key;
            pair.value = std::to_string(value);
            entry.values.push_back(pair);
        };
        add("generation", generation_);
        add("effective_features", effct_feat_num);
        add("residual_rms", measurement_rms);
        add("information_ratio", measurement_information_ratio);
        add("prediction_interval", status.prediction_interval);
        add("map_points", ikdtree.Root_Node ? ikdtree.validnum() : 0);
        for (int axis = 0; axis < 3; ++axis) {
            add("velocity_" + std::to_string(axis), state_point.vel(axis));
            add("gyro_bias_" + std::to_string(axis), state_point.bg(axis));
            add("accel_bias_" + std::to_string(axis), state_point.ba(axis));
            add("gravity_" + std::to_string(axis), state_point.grav[axis]);
        }
        add("last_imu_gap", last_imu_gap_);
        const double startup_elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - startup_started_).count();
        if (tracking.accepted_stamp < 0.0 && startup_elapsed >= startup_warning_timeout_) {
            entry.level = 2;
            if (tracking.state != fast_lio::TrackingGuard::LOST) entry.message = "startup_wait_timeout";
        }
        add("startup_elapsed", startup_elapsed);
        add("startup_retries", startup_retries_);
        add("processing_duration", processing_duration_);
        add("sync_wait", sync_wait_elapsed);
        add("scan_begin", Measures.lidar_beg_time);
        add("scan_end", lidar_end_time);
        add("imu_batch_size", sync_coverage.count);
        add("imu_first", sync_coverage.first);
        add("imu_last", sync_coverage.last);
        add("imu_following", sync_coverage.following);
        add("imu_end_gap", sync_coverage.end_gap);
        add("imu_coverage_gap", sync_coverage.gap);
        add("imu_gap_before", sync_coverage.gap_before);
        add("imu_gap_after", sync_coverage.gap_after);
        add("imu_integration_boundary", sync_coverage.integration_boundary);
        add("sensor_epoch", sensor_epoch);
        diagnostic_msgs::msg::KeyValue sync_reason;
        sync_reason.key = "sync_reason"; sync_reason.value = sync_coverage.reason;
        entry.values.push_back(sync_reason);
        {
            std::lock_guard<std::mutex> lock(mtx_buffer);
            add("imu_queue_size", imu_buffer.size());
            add("imu_queue_high_water", imu_queue_high_water);
            add("received_imu_gap", received_imu_gap);
            add("received_arrival_gap", received_arrival_gap);
        }
        if (tracking.state == fast_lio::TrackingGuard::LOST) {
            if (!lost_reported_) {
                fault_snapshot_ = entry.values;
                RCLCPP_ERROR(get_logger(), "Tracking LOST: %s; sync=%s scan_end=%.9f last=%.9f following=%.9f wait=%.3f processing=%.3f",
                    tracking.reason.c_str(), sync_coverage.reason.c_str(), lidar_end_time,
                    sync_coverage.last, sync_coverage.following, sync_wait_elapsed, processing_duration_);
                lost_reported_ = true;
            }
            entry.values = fault_snapshot_;
        }
        diagnostics.status.push_back(entry);
        diagnostics_pub_->publish(diagnostics);
    }

    void publish_held()
    {
        if (!have_held_odometry) return;
        pubOdomAftMapped_->publish(held_odometry);
        if (publish_tf_en) tf_broadcaster_->sendTransform(held_transform);
        if (scan_pub_en && !held_world_cloud.data.empty()) pubLaserCloudFull_->publish(held_world_cloud);
        if (scan_pub_en && scan_body_pub_en && !held_body_cloud.data.empty())
            pubLaserCloudFull_body_->publish(held_body_cloud);
    }

    void reset_tracking(bool external = true)
    {
        {
            std::lock_guard<std::mutex> lock(mtx_buffer);
            ++sensor_epoch;
            lidar_buffer.clear(); time_buffer.clear(); imu_buffer.clear();
            lidar_pushed = false;
            last_timestamp_lidar = 0.0; last_timestamp_imu = -1.0;
            is_first_lidar = true; flg_first_scan = true; flg_EKF_inited = false;
            lidar_end_time = 0.0; first_lidar_time = 0.0;
            lidar_mean_scantime = 0.0; scan_num = 0; scan_count = 0; publish_count = 0;
            time_log_counter = 0; frame_num = 0; position_last = Zero3d;
            timediff_set_flg = false; timediff_lidar_wrt_imu = 0.0;
            timing_fault.clear();
            imu_queue_high_water = 0; received_imu_gap = 0.0; received_arrival_gap = 0.0;
            startup_imu_ready = false; sync_previous_imu = -1.0;
            sync_coverage = fast_lio::ImuCoverage(); sync_wait_elapsed = 0.0;
        }
        last_batch_end_ = -1.0; last_batch_imu_ = -1.0;
        tracking.reset(); have_accepted_state_ = false;
        if (external) {
            ++generation_; startup_retries_ = 0;
            startup_started_ = std::chrono::steady_clock::now();
        }
        last_accepted_wall_ = std::chrono::steady_clock::now();
        fault_snapshot_.clear(); lost_reported_ = false;
        p_imu->Reset();
        state_ikfom fresh;
        fresh.offset_T_L_I = Lidar_T_wrt_IMU;
        fresh.offset_R_L_I = Lidar_R_wrt_IMU;
        kf.change_x(fresh);
        auto covariance = kf.get_P().eval(); covariance.setIdentity(); kf.change_P(covariance);
        state_point = fresh;
        // Destroying the tree joins its rebuild thread before constructing a fresh tree.
        ikdtree.~KD_TREE<PointType>();
        new (&ikdtree) KD_TREE<PointType>();
        Localmap_Initialized = false; cub_needrm.clear(); Nearest_Points.clear();
        pointSearchInd_surf.clear(); Measures = MeasureGroup();
        feats_undistort->clear(); feats_down_body->clear(); feats_down_world->clear();
        featsFromMap->clear(); pcl_wait_pub->clear(); pcl_wait_save->clear(); path.poses.clear();
        std::fill(std::begin(point_selected_surf), std::end(point_selected_surf), false);
        effct_feat_num = 0; measurement_rms = 0.0; measurement_information_ratio = 0.0;
        measurement_valid = true; measurement_reason.clear();
        publish_tracking();
    }

    bool validate_batch()
    {
        {
            std::lock_guard<std::mutex> lock(mtx_buffer);
            if (!timing_fault.empty()) {tracking.lose(timing_fault); return false;}
        }
        if (Measures.imu.empty()) {tracking.lose("missing_imu_coverage"); return false;}
        if (last_batch_end_ >= 0.0 && (Measures.lidar_end_time <= last_batch_end_ ||
            Measures.lidar_end_time - last_batch_end_ > tracking.limits.prediction_timeout)) {
            tracking.lose("scan_time_gap"); return false;
        }
        double previous = last_batch_imu_;
        for (const auto & imu : Measures.imu) {
            const double stamp = get_time_sec(imu->header.stamp);
            if (previous >= 0.0) {
                last_imu_gap_ = stamp - previous;
                if (last_imu_gap_ <= 0.0 || fast_lio::imuGapExceeded(last_imu_gap_, tracking.limits.max_imu_gap)) {
                    tracking.lose("imu_time_gap"); return false;
                }
            }
            previous = stamp;
        }
        if (fast_lio::imuGapExceeded(Measures.lidar_end_time - previous, tracking.limits.max_imu_gap)) {
            tracking.lose("missing_imu_coverage"); return false;
        }
        last_batch_imu_ = previous;
        last_batch_end_ = Measures.lidar_end_time;
        return tracking.allowPrediction(Measures.lidar_end_time);
    }

    void retry_startup()
    {
        if (tracking.accepted_stamp >= 0.0 || tracking.state != fast_lio::TrackingGuard::LOST) return;
        const auto reason = tracking.reason;
        if (reason != "missing_imu_coverage" && reason != "imu_time_gap" &&
            reason != "scan_time_gap" && reason != "initialization_timeout") return;
        ++startup_retries_;
        RCLCPP_WARN(get_logger(), "Retrying unpublished startup window (%lu): %s; sync=%s",
            static_cast<unsigned long>(startup_retries_), reason.c_str(), sync_coverage.reason.c_str());
        reset_tracking(false);
    }

    void timer_callback()
    {
        processing_scan_ = false;
        process_scan();
        if (processing_scan_) processing_duration_ = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - processing_started_).count();
        processing_scan_ = false;
        retry_startup();
        if (tracking.state == fast_lio::TrackingGuard::LOST) publish_tracking();
    }

    void process_scan()
    {
        {
            std::lock_guard<std::mutex> lock(mtx_buffer);
            if (!timing_fault.empty()) tracking.lose(timing_fault);
        }
        if (tracking.state == fast_lio::TrackingGuard::LOST) {
            if (tracking.accepted_stamp >= 0.0) publish_tracking();
            // Bound buffers while keeping the lost state latched.
            std::lock_guard<std::mutex> lock(mtx_buffer);
            lidar_buffer.clear(); time_buffer.clear(); imu_buffer.clear(); lidar_pushed = false;
            return;
        }
        if (sync_packages(Measures))
        {
            processing_scan_ = true;
            processing_started_ = std::chrono::steady_clock::now();
            if (!validate_batch()) {publish_tracking(); return;}
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                return;
            }

            double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time = 0;
            t0 = omp_get_wtime();

            feats_undistort->clear();
            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            if (!state_point.pos.allFinite() || !state_point.vel.allFinite() ||
                !state_point.rot.toRotationMatrix().allFinite() || !state_point.bg.allFinite() ||
                !state_point.ba.allFinite() || !state_point.offset_T_L_I.allFinite() ||
                !state_point.offset_R_L_I.toRotationMatrix().allFinite() ||
                !std::isfinite(state_point.grav[0]) || !std::isfinite(state_point.grav[1]) ||
                !std::isfinite(state_point.grav[2]) || !kf.get_P().allFinite() ||
                state_point.vel.norm() > tracking.limits.max_speed ||
                (have_accepted_state_ && !fast_lio::plausibleMotion(tracking.limits,
                    accepted_state_.pos, accepted_state_.rot.toRotationMatrix(), state_point.pos,
                    state_point.rot.toRotationMatrix(), state_point.vel,
                    Measures.lidar_end_time - accepted_state_stamp_))) {
                tracking.lose("implausible_prediction"); publish_tracking(); return;
            }

            if (!feats_undistort || feats_undistort->empty())
            {
                effct_feat_num = 0; measurement_information_ratio = 0.0;
                tracking.reject(Measures.lidar_end_time, "insufficient_features");
                publish_tracking();
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;
            }

            if (tracking.prediction_started < 0.0) {
                tracking.prediction_started = Measures.lidar_end_time;
                last_accepted_wall_ = std::chrono::steady_clock::now();
            }
            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
            /*** Segment the map in lidar FOV ***/
            // Local map pruning is deferred until the update is accepted.

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if (ikdtree.Root_Node == nullptr)
            {
                RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
                if (feats_down_size >= tracking.limits.min_features && scan_geometry_valid())
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for (int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                return;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();

            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < tracking.limits.min_features || feats_down_size > 100000)
            {
                effct_feat_num = 0; measurement_information_ratio = 0.0;
                tracking.reject(Measures.lidar_end_time, "insufficient_features");
                publish_tracking();
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;
            }

            // Neighbor plane fitting can invent varied normals for collinear
            // neighborhoods. A planar scan still cannot constrain six pose axes.
            if (!scan_geometry_valid()) {
                effct_feat_num = 0; measurement_information_ratio = 0.0;
                tracking.reject(Measures.lidar_end_time, "degenerate_geometry");
                publish_tracking();
                return;
            }

            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
                     << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << endl;

            if (0) // If you need to see map point, change to "if(1)"
            {
                PointVector().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();

            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            auto predicted_state = kf.get_x();
            auto predicted_covariance = kf.get_P().eval();
            measurement_valid = true;
            measurement_reason.clear();
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            if (!measurement_valid) {
                kf.change_x(predicted_state); kf.change_P(predicted_covariance);
                state_point = predicted_state;
                tracking.reject(Measures.lidar_end_time, measurement_reason);
                publish_tracking();
                return;
            }
            state_point = kf.get_x();
            const bool finite_state = state_point.pos.allFinite() && state_point.vel.allFinite() &&
                state_point.rot.toRotationMatrix().allFinite() && state_point.bg.allFinite() &&
                state_point.ba.allFinite() && state_point.offset_T_L_I.allFinite() &&
                state_point.offset_R_L_I.toRotationMatrix().allFinite() &&
                std::isfinite(state_point.grav[0]) && std::isfinite(state_point.grav[1]) &&
                std::isfinite(state_point.grav[2]) && kf.get_P().allFinite();
            const bool plausible = finite_state && state_point.vel.norm() <= tracking.limits.max_speed &&
                (!have_accepted_state_ || fast_lio::plausibleMotion(tracking.limits,
                    accepted_state_.pos, accepted_state_.rot.toRotationMatrix(), state_point.pos,
                    state_point.rot.toRotationMatrix(), state_point.vel,
                    Measures.lidar_end_time - accepted_state_stamp_));
            if (!plausible) {
                kf.change_x(predicted_state); kf.change_P(predicted_covariance);
                state_point = predicted_state; tracking.lose("implausible_state"); publish_tracking(); return;
            }
            {
                std::lock_guard<std::mutex> lock(mtx_buffer);
                if (!timing_fault.empty()) tracking.lose(timing_fault);
            }
            if (!tracking.accept(Measures.lidar_end_time)) {publish_tracking(); return;}
            accepted_state_ = state_point; accepted_state_stamp_ = Measures.lidar_end_time;
            have_accepted_state_ = true; last_accepted_wall_ = std::chrono::steady_clock::now();
            publish_tracking();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped_, tf_broadcaster_);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            lasermap_fov_segment();
            map_incremental();
            t5 = omp_get_wtime();

            /******* Publish points *******/
            if (path_en)
                publish_path(pubPath_);
            if (scan_pub_en)
                publish_frame_world(pubLaserCloudFull_);
            if (scan_pub_en && scan_body_pub_en)
                publish_frame_body(pubLaserCloudFull_body_);
            if (effect_pub_en)
                publish_effect_world(pubLaserCloudEffect_);
            // if (map_pub_en) publish_map(pubLaserCloudMap_);

            /*** Debug variables ***/
            if (runtime_pos_log && time_log_counter < MAXN)
            {
                frame_num++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1) / frame_num + (kdtree_incremental_time) / frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time + solve_H_time) / frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n", t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu, aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
                         << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << " " << feats_undistort->points.size() << endl;
                if (fp) dump_lio_state_to_log(fp);
            }
        }
    }

    void map_publish_callback()
    {
        if (map_pub_en && tracking.state == fast_lio::TrackingGuard::TRACKING)
            publish_map(pubLaserCloudMap_);
    }

    void map_save_callback(std_srvs::srv::Trigger::Request::ConstSharedPtr req, std_srvs::srv::Trigger::Response::SharedPtr res)
    {
        RCLCPP_INFO(this->get_logger(), "Saving map to %s...", map_file_path.c_str());
        if (pcd_save_en)
        {
            save_to_pcd();
            res->success = true;
            res->message = "Map saved.";
        }
        else
        {
            res->success = false;
            res->message = "Map save disabled.";
        }
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    std::unique_ptr<fast_lio::ImuReceiver> imu_receiver_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;
    // rclcpp::Subscription<livox_interfaces::msg::CustomMsg>::SharedPtr sub_pcl_livox_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr map_pub_timer_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;

    rclcpp::Publisher<fast_lio::msg::TrackingStatus>::SharedPtr tracking_pub_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_srv_;
    rclcpp::TimerBase::SharedPtr health_timer_;
    const std::uint64_t instance_id_ = static_cast<std::uint64_t>(
        std::chrono::system_clock::now().time_since_epoch().count());
    std::uint64_t generation_ = 0;
    double last_batch_end_ = -1.0, last_batch_imu_ = -1.0, last_imu_gap_ = 0.0;
    state_ikfom accepted_state_;
    bool have_accepted_state_ = false;
    bool processing_scan_ = false;
    std::chrono::steady_clock::time_point processing_started_;
    double startup_warning_timeout_ = 10.0, processing_duration_ = 0.0;
    std::uint64_t startup_retries_ = 0;
    std::chrono::steady_clock::time_point startup_started_ = std::chrono::steady_clock::now();
    bool lost_reported_ = false;
    std::vector<diagnostic_msgs::msg::KeyValue> fault_snapshot_;
    double accepted_state_stamp_ = -1.0;
    std::chrono::steady_clock::time_point last_accepted_wall_;
    bool effect_pub_en = false, map_pub_en = false;
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    double epsi[23] = {0.001};

    FILE *fp = nullptr;
    ofstream fout_pre, fout_out, fout_dbg;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    signal(SIGINT, SigHandle);

    rclcpp::spin(std::make_shared<LaserMappingNode>());

    if (rclcpp::ok())
        rclcpp::shutdown();
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(), "w");
        fprintf(fp2, "time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0; i < time_log_counter; i++)
        {
            fprintf(fp2, "%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n", T1[i], s_plot[i], int(s_plot2[i]), s_plot3[i], s_plot4[i], int(s_plot5[i]), s_plot6[i], int(s_plot7[i]), int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
