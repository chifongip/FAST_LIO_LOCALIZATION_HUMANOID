// Copyright 2026 Open3D Loc Contributors
#ifndef OPEN3D_LOC__RECOVERY_COORDINATOR_HPP_
#define OPEN3D_LOC__RECOVERY_COORDINATOR_HPP_
#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <string>
#include <cstdint>
#include <optional>
#include <vector>
#include <deque>

namespace open3d_loc
{
using Matrix6d = Eigen::Matrix<double, 6, 6>;
namespace recovery
{
using Matrix6d = Eigen::Matrix<double, 6, 6>;
enum class ApplicationMode { RESET, SHADOW, BOUNDED };
enum class State { IDLE, CONFIRMING, PROPOSED, VERIFYING, VERIFIED };
struct Config
{
  ApplicationMode application_mode = ApplicationMode::SHADOW;
  std::array<bool, 6> update_mask{{true, true, true, false, false, true}};
  int confirmation_windows = 3;
  int verification_windows = 3;
  double translation_noise_floor = 0.15;
  double rotation_noise_floor = 0.05;
  double nis_threshold = 16.812;
  double max_translation_delta = 0.5;
  double max_rotation_delta = 0.2;
  double candidate_timeout = 3.0;
  double max_step_translation = 0.5;
  double max_step_rotation = 0.2;
};
// Inclusive IDs; accepted windows must be disjoint in IDs and time.
struct ScanWindow
{
  std::uint64_t first_scan_id = 0;
  std::uint64_t last_scan_id = 0;
  double start_stamp = 0.0;
  double end_stamp = 0.0;
};
struct Quality
{
  bool quality_ok = false;
  bool ambiguity_ok = false;
  bool observability_ok = false;
  bool accepted() const {return quality_ok && ambiguity_ok && observability_ok;}
};
struct Candidate
{
  std::uint64_t generation = 0;
  ScanWindow window;
  Eigen::Matrix4d map_to_odom = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d odom_to_robot = Eigen::Matrix4d::Identity();
  // Robot covariance at end_stamp: map translation, local rotation.
  Matrix6d covariance = Matrix6d::Zero();
  Quality quality;
};
struct Result
{
  State state = State::IDLE;
  std::uint64_t generation = 0;
  bool candidate_recorded = false;
  bool proposal_available = false;
  bool should_apply = false;
  bool verified = false;
  int consistent_windows = 0;
  int verified_windows = 0;
  Eigen::Matrix4d requested_pose = Eigen::Matrix4d::Identity();
  double translation_delta = 0.0;
  double rotation_delta = 0.0;
  double nis = 0.0;
  std::string reason;
};
class OdometryHistory
{
public:
  explicit OdometryHistory(double duration = 10.0, std::size_t capacity = 2000);
  bool add(double stamp, const Eigen::Matrix4d & odom_to_robot);
  std::optional<Eigen::Matrix4d> interpolate(double stamp) const;
  void clear();
  std::size_t size() const;

private:
  struct Sample {double stamp; Eigen::Matrix4d pose;};
  double duration_;
  std::size_t capacity_;
  std::deque<Sample> samples_;
};
class RecoveryCoordinator
{
public:
  // Single-threaded, proposal-only core. Call tick(now) before acknowledging
  // asynchronous work, and invalidate on external map/seed/odometry resets.
  explicit RecoveryCoordinator(const Config & config = Config());
  Result observe(
    const Candidate & candidate,
    const Eigen::Matrix4d & current_map_to_odom, double now);
  bool acknowledgeApplied(std::uint64_t generation, const Eigen::Matrix4d & applied_pose);
  Result observeVerification(const Candidate & incumbent, double now);
  Result tick(double now);
  std::uint64_t invalidate();
  std::uint64_t generation() const;
  State state() const;
  const Config & config() const;

private:
  Result result(const std::string & reason) const;
  void clear();
  bool expire(double now);
  bool consistent(const Candidate & anchor, const Candidate & candidate, Result & out) const;
  Config config_;
  std::uint64_t generation_ = 0;
  State state_ = State::IDLE;
  std::optional<Candidate> first_;
  std::optional<Candidate> previous_;
  std::optional<ScanWindow> verification_window_;
  Eigen::Matrix4d requested_pose_ = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d target_pose_ = Eigen::Matrix4d::Identity();
  bool bounded_active_ = false;
  int count_ = 0;
  int verification_count_ = 0;
  double last_stamp_ = 0.0;
  double last_now_ = 0.0;
  bool have_now_ = false;
};
}  // namespace recovery

using OdometryHistory = recovery::OdometryHistory;
enum class RelocalizationState {TRACKING, SEARCHING, CONFIRMING, VERIFYING, DEGRADED};
struct RecoveryCoordinatorConfig
{
  std::string application_mode = "reset";
  int required_observations = 3;
  int verification_observations = 3;
  double minimum_interval = 0.75;
  double timeout = 3.0;
  double translation_limit = 0.5;
  double rotation_limit = 0.2;
  double translation_stddev = 0.15;
  double rotation_stddev = 0.05;
  double mahalanobis_threshold = 16.812;
  double max_step_translation = 0.5;
  double max_step_rotation = 0.15;
  std::array<bool, 6> update_mask{{true, true, true, true, true, true}};
};
struct RecoveryObservation
{
  std::uint64_t generation = 0;
  double stamp = 0.0;
  double oldest_stamp = 0.0;
  std::vector<std::uint64_t> scan_ids;
  Eigen::Matrix4d odom_base = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d map_base = Eigen::Matrix4d::Identity();
  // Translation and rotation perturbations expressed in map coordinates.
  Matrix6d covariance = Matrix6d::Identity() * 0.01;
  // Translation and rotation perturbations expressed in odometry coordinates.
  Matrix6d odom_covariance = Matrix6d::Zero();
  bool quality_valid = false;
  bool unambiguous = false;
  bool observable = false;
};
struct RecoveryDecision
{
  bool proposal = false;
  bool shadow = false;
  int count = 0;
  double translation_residual = 0.0;
  double rotation_residual = 0.0;
  double mahalanobis = 0.0;
  std::string reason;
  Eigen::Matrix4d map_odom = Eigen::Matrix4d::Identity();
  Matrix6d covariance = Matrix6d::Identity();
};
class RecoveryCoordinator
{
public:
  explicit RecoveryCoordinator(const RecoveryCoordinatorConfig & config = {});
  void invalidate(std::uint64_t generation);
  // Search-worker lifecycle notification; preserves active recovery evidence.
  void searching();
  RecoveryDecision observe(
    const RecoveryObservation & observation,
    const Eigen::Matrix4d & current_map_odom);
  // observation retains the proposal generation and contains the actual
  // applied map_base. generation is that generation or its immediate successor.
  // Bounded steps preserve the confirmed target and return to CONFIRMING until
  // it is reached; only then do three incumbent observations verify recovery.
  void applied(std::uint64_t generation, const RecoveryObservation & observation);
  // Same-generation acknowledgement; returns false for stale/mismatched poses.
  bool acknowledgeApplied(std::uint64_t generation, const Eigen::Matrix4d & applied_pose);
  void rejectedProposal();
  void expire(double now);
  RelocalizationState state() const {return state_;}
  int count() const {return count_;}
  static const char * stateName(RelocalizationState state);
  static Matrix6d correctionCovariance(
    const Matrix6d & robot_covariance,
    const Eigen::Matrix4d & map_odom, const Eigen::Matrix4d & odom_base);
  static Eigen::Matrix4d maskedPose(
    const Eigen::Matrix4d & current,
    const Eigen::Matrix4d & target, const std::array<bool, 6> & mask);

private:
  RecoveryCoordinatorConfig config_;
  RelocalizationState state_ = RelocalizationState::TRACKING;
  std::uint64_t generation_ = 0;
  int count_ = 0;
  std::optional<RecoveryObservation> first_, previous_;
  void anchor(const RecoveryObservation & observation);
  recovery::RecoveryCoordinator core_;
  Eigen::Matrix4d pending_pose_ = Eigen::Matrix4d::Identity();
  bool pending_ = false;
  std::optional<double> wall_now_;
};
}  // namespace open3d_loc
#endif
