// Copyright 2026 Open3D Loc Contributors
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd
#ifndef OPEN3D_LOC__GLOBAL_CORRECTION_FILTER_HPP_
#define OPEN3D_LOC__GLOBAL_CORRECTION_FILTER_HPP_

#include <Eigen/Core>

#include <array>
#include <string>

namespace open3d_loc
{

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

struct GlobalCorrectionFilterConfig
{
  std::array<bool, 6> update_mask{{true, true, true, false, false, true}};
  Vector6d initial_stddev =
    (Vector6d() << 0.25, 0.25, 0.15, 0.05, 0.05, 0.15).finished();
  Vector6d process_stddev_time =
    (Vector6d() << 0.02, 0.02, 0.01, 0.005, 0.005, 0.01).finished();
  Vector6d process_stddev_distance =
    (Vector6d() << 0.01, 0.01, 0.005, 0.002, 0.002, 0.005).finished();
  Vector6d process_stddev_rotation =
    (Vector6d() << 0.01, 0.01, 0.005, 0.005, 0.005, 0.01).finished();
  Vector6d measurement_stddev_floor =
    (Vector6d() << 0.03, 0.03, 0.03, 0.01, 0.01, 0.02).finished();
  double max_innovation_translation = 1.0;
  double max_innovation_rotation = 0.35;
  double mahalanobis_threshold = 16.812;

  struct RecoveryConfig
  {
    bool enabled = false;
    int required_consistent_measurements = 2;
    double minimum_candidate_interval = 0.75;
    double candidate_timeout = 3.0;
    double consistency_mahalanobis_threshold = 13.277;
    double max_candidate_translation_delta = 0.50;
    double max_candidate_rotation_delta = 0.20;
    double max_step_translation = 0.50;
    double max_step_rotation = 0.15;
  } recovery;
};

struct CorrectionUpdateResult
{
  bool accepted = false;
  double translation_innovation = 0.0;
  double rotation_innovation = 0.0;
  double mahalanobis_distance = 0.0;
  std::string reason;
};

enum class RecoveryMode
{
  TRACKING,
  CONFIRMING,
  RECOVERING
};

struct RecoveryUpdateResult
{
  bool candidate_recorded = false;
  bool correction_applied = false;
  RecoveryMode mode = RecoveryMode::TRACKING;
  int consistent_measurements = 0;
  double candidate_interval = 0.0;
  double motion_distance = 0.0;
  double motion_rotation = 0.0;
  double candidate_translation_delta = 0.0;
  double candidate_rotation_delta = 0.0;
  double consistency_mahalanobis_distance = 0.0;
  double applied_translation = 0.0;
  double applied_rotation = 0.0;
  std::string reason;
};

class GlobalCorrectionFilter
{
public:
  explicit GlobalCorrectionFilter(
    const GlobalCorrectionFilterConfig & config = GlobalCorrectionFilterConfig());

  void configure(const GlobalCorrectionFilterConfig & config);
  void reset(const Eigen::Matrix4d & map_to_odom);
  void reset(const Eigen::Matrix4d & map_to_odom, const Matrix6d & covariance);
  void predict(double dt, double distance, double rotation);
  CorrectionUpdateResult update(
    const Eigen::Matrix4d & measurement, const Matrix6d & measurement_covariance);
  RecoveryUpdateResult observeRecoveryCandidate(
    const Eigen::Matrix4d & measurement,
    const Matrix6d & measurement_covariance,
    double stamp_seconds,
    double cumulative_distance,
    double cumulative_rotation);
  void clearRecovery();
  // Deterministic projection; preserve covariance and recovery history.
  void adjustHeight(double delta_z);

  bool initialized() const;
  const Eigen::Matrix4d & pose() const;
  const Matrix6d & covariance() const;
  RecoveryMode recoveryMode() const;
  int recoveryCandidateCount() const;

  static const char * recoveryModeName(RecoveryMode mode);

  static Matrix6d regularizeCovariance(
    const Matrix6d & covariance, const Vector6d & stddev_floor);

private:
  static Vector6d poseResidual(
    const Eigen::Matrix4d & estimate, const Eigen::Matrix4d & measurement);
  static bool validTransform(const Eigen::Matrix4d & transform);
  void applyError(const Vector6d & error);
  RecoveryUpdateResult recordFirstRecoveryCandidate(
    const Eigen::Matrix4d & measurement,
    const Matrix6d & measurement_covariance,
    double stamp_seconds,
    double cumulative_distance,
    double cumulative_rotation,
    const std::string & reason);
  void storeRecoveryCandidate(
    const Eigen::Matrix4d & measurement,
    const Matrix6d & measurement_covariance,
    double stamp_seconds,
    double cumulative_distance,
    double cumulative_rotation);
  RecoveryUpdateResult applyRecoveryStep(
    const Eigen::Matrix4d & measurement,
    const Matrix6d & measurement_covariance,
    RecoveryUpdateResult result);

  GlobalCorrectionFilterConfig config_;
  Eigen::Matrix4d pose_ = Eigen::Matrix4d::Identity();
  Matrix6d covariance_ = Matrix6d::Identity();
  bool initialized_ = false;
  RecoveryMode recovery_mode_ = RecoveryMode::TRACKING;
  int recovery_candidate_count_ = 0;
  bool have_recovery_candidate_ = false;
  Eigen::Matrix4d recovery_candidate_pose_ = Eigen::Matrix4d::Identity();
  Matrix6d recovery_candidate_covariance_ = Matrix6d::Identity();
  double recovery_candidate_stamp_ = 0.0;
  double recovery_candidate_cumulative_distance_ = 0.0;
  double recovery_candidate_cumulative_rotation_ = 0.0;
};

}  // namespace open3d_loc

#endif  // OPEN3D_LOC__GLOBAL_CORRECTION_FILTER_HPP_
