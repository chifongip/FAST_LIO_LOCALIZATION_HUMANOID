#include "open3d_loc/recovery_coordinator.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace open3d_loc
{
namespace recovery
{
namespace
{
using Vector6d = Eigen::Matrix<double, 6, 1>;

bool valid_pose(const Eigen::Matrix4d & pose)
{
  const Eigen::Matrix3d rotation = pose.topLeftCorner<3, 3>();
  return pose.allFinite() &&
         pose.row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1), 1e-8) &&
         (rotation.transpose() * rotation).isApprox(Eigen::Matrix3d::Identity(), 1e-6) &&
         std::abs(rotation.determinant() - 1.0) < 1e-6;
}

bool valid_window(const ScanWindow & w)
{
  return std::isfinite(w.start_stamp) && std::isfinite(w.end_stamp) &&
         w.start_stamp <= w.end_stamp && w.first_scan_id <= w.last_scan_id;
}

bool disjoint(const ScanWindow & before, const ScanWindow & after)
{
  return after.first_scan_id > before.last_scan_id && after.start_stamp > before.end_stamp;
}

bool valid_covariance(const Matrix6d & covariance)
{
  if (!covariance.allFinite() || !covariance.isApprox(covariance.transpose(), 1e-8)) {
    return false;
  }
  const Eigen::SelfAdjointEigenSolver<Matrix6d> solver(covariance);
  return solver.info() == Eigen::Success && solver.eigenvalues().minCoeff() >= -1e-10;
}

bool valid_candidate(const Candidate & candidate, double now)
{
  return valid_window(candidate.window) && candidate.window.end_stamp <= now &&
         valid_pose(candidate.map_to_odom) && valid_pose(candidate.odom_to_robot) &&
         valid_covariance(candidate.covariance);
}

Eigen::Vector3d rotation_vector(const Eigen::Matrix3d & rotation)
{
  const Eigen::AngleAxisd angle_axis(rotation);
  return angle_axis.angle() * angle_axis.axis();
}

Eigen::Matrix3d rotation_matrix(const Eigen::Vector3d & vector)
{
  const double angle = vector.norm();
  if (angle < 1e-15) {
    return Eigen::Matrix3d::Identity();
  }
  return Eigen::AngleAxisd(angle, vector / angle).toRotationMatrix();
}

Eigen::Matrix3d skew(const Eigen::Vector3d & v)
{
  Eigen::Matrix3d out;
  out << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return out;
}

Vector6d residual(const Eigen::Matrix4d & expected, const Eigen::Matrix4d & measured)
{
  Vector6d out;
  out.head<3>() = measured.topRightCorner<3, 1>() - expected.topRightCorner<3, 1>();
  out.tail<3>() = rotation_vector(
    expected.topLeftCorner<3, 3>().transpose() * measured.topLeftCorner<3, 3>());
  return out;
}

Eigen::Matrix4d project(
  const Eigen::Matrix4d & current, const Eigen::Matrix4d & target, const Config & config)
{
  Vector6d delta = residual(current, target);
  for (int i = 0; i < 6; ++i) {
    if (!config.update_mask[static_cast<std::size_t>(i)]) {
      delta[i] = 0.0;
    }
  }
  if (config.application_mode == ApplicationMode::BOUNDED) {
    const double translation = delta.head<3>().norm();
    const double rotation = delta.tail<3>().norm();
    if (translation > config.max_step_translation) {
      delta.head<3>() *= config.max_step_translation / translation;
    }
    if (rotation > config.max_step_rotation) {
      delta.tail<3>() *= config.max_step_rotation / rotation;
    }
  }
  Eigen::Matrix4d out = current;
  out.topRightCorner<3, 1>() += delta.head<3>();
  out.topLeftCorner<3, 3>() = current.topLeftCorner<3, 3>() * rotation_matrix(delta.tail<3>());
  return out;
}
}  // namespace

OdometryHistory::OdometryHistory(double duration, std::size_t capacity)
: duration_(duration), capacity_(capacity)
{
  if (!std::isfinite(duration) || duration <= 0.0 || capacity < 2) {
    throw std::invalid_argument("Odometry history requires positive duration and capacity >= 2");
  }
}

bool OdometryHistory::add(double stamp, const Eigen::Matrix4d & pose)
{
  if (!std::isfinite(stamp) || !valid_pose(pose) ||
    (!samples_.empty() && stamp <= samples_.back().stamp))
  {
    return false;
  }
  samples_.push_back({stamp, pose});
  while (samples_.size() > capacity_ ||
    (samples_.size() > 1 && stamp - samples_.front().stamp > duration_))
  {
    samples_.pop_front();
  }
  return true;
}

std::optional<Eigen::Matrix4d> OdometryHistory::interpolate(double stamp) const
{
  if (!std::isfinite(stamp) || samples_.empty() ||
    stamp < samples_.front().stamp || stamp > samples_.back().stamp)
  {
    return std::nullopt;
  }
  const auto upper = std::lower_bound(
    samples_.begin(), samples_.end(), stamp,
    [](const Sample & sample, double time) {return sample.stamp < time;});
  if (upper->stamp == stamp) {
    return upper->pose;
  }
  const auto lower = std::prev(upper);
  const double alpha = (stamp - lower->stamp) / (upper->stamp - lower->stamp);
  Eigen::Matrix4d out = Eigen::Matrix4d::Identity();
  out.topRightCorner<3, 1>() = (1.0 - alpha) * lower->pose.topRightCorner<3, 1>() +
    alpha * upper->pose.topRightCorner<3, 1>();
  out.topLeftCorner<3, 3>() = Eigen::Quaterniond(lower->pose.topLeftCorner<3, 3>()).slerp(
    alpha, Eigen::Quaterniond(upper->pose.topLeftCorner<3, 3>())).toRotationMatrix();
  return out;
}

void OdometryHistory::clear() {samples_.clear();}
std::size_t OdometryHistory::size() const {return samples_.size();}

RecoveryCoordinator::RecoveryCoordinator(const Config & config)
: config_(config)
{
  const double positive[] = {config.translation_noise_floor, config.rotation_noise_floor,
    config.nis_threshold, config.max_translation_delta, config.max_rotation_delta,
    config.candidate_timeout, config.max_step_translation, config.max_step_rotation};
  for (double value : positive) {
    if (!std::isfinite(value) || value <= 0.0) {
      throw std::invalid_argument("Recovery thresholds must be finite and positive");
    }
  }
  if (config.confirmation_windows < 2 || config.verification_windows < 2) {
    throw std::invalid_argument("Recovery requires at least two independent windows");
  }
  if (config.application_mode != ApplicationMode::RESET &&
    config.application_mode != ApplicationMode::SHADOW &&
    config.application_mode != ApplicationMode::BOUNDED)
  {
    throw std::invalid_argument("Invalid recovery application mode");
  }
}

Result RecoveryCoordinator::result(const std::string & reason) const
{
  Result out;
  out.state = state_;
  out.generation = generation_;
  out.consistent_windows = count_;
  out.verified_windows = verification_count_;
  out.requested_pose = requested_pose_;
  out.verified = state_ == State::VERIFIED;
  out.reason = reason;
  return out;
}

void RecoveryCoordinator::clear()
{
  state_ = State::IDLE;
  first_.reset();
  previous_.reset();
  verification_window_.reset();
  count_ = 0;
  verification_count_ = 0;
  requested_pose_.setIdentity();
  target_pose_.setIdentity();
  bounded_active_ = false;
}

std::uint64_t RecoveryCoordinator::invalidate()
{
  clear();
  have_now_ = false;
  return ++generation_;
}

bool RecoveryCoordinator::expire(double now)
{
  if (!std::isfinite(now) || (have_now_ && now < last_now_)) {
    invalidate();
    return true;
  }
  last_now_ = now;
  have_now_ = true;
  if (state_ != State::IDLE && now - last_stamp_ > config_.candidate_timeout) {
    invalidate();
    last_now_ = now;
    have_now_ = true;
    return true;
  }
  return false;
}

Result RecoveryCoordinator::tick(double now)
{
  return result(expire(now) ? "expired_or_clock_reset" : "unchanged");
}

bool RecoveryCoordinator::consistent(
  const Candidate & anchor, const Candidate & candidate, Result & out) const
{
  // Transport the full anchor robot pose to the new acquisition time. Checking
  // correction translations alone misses rotational lever-arm errors far from
  // the odometry origin; checking raw robot poses rejects legitimate motion.
  const Eigen::Matrix4d motion = anchor.odom_to_robot.inverse() * candidate.odom_to_robot;
  const Eigen::Matrix4d anchor_robot = anchor.map_to_odom * anchor.odom_to_robot;
  const Eigen::Matrix4d expected = anchor_robot * motion;
  const Eigen::Matrix4d measured = candidate.map_to_odom * candidate.odom_to_robot;
  const Vector6d error = residual(expected, measured);
  out.translation_delta = std::max(out.translation_delta, error.head<3>().norm());
  out.rotation_delta = std::max(out.rotation_delta, error.tail<3>().norm());
  Matrix6d jacobian = Matrix6d::Identity();
  jacobian.topRightCorner<3, 3>() =
    -anchor_robot.topLeftCorner<3, 3>() * skew(motion.topRightCorner<3, 1>());
  jacobian.bottomRightCorner<3, 3>() = motion.topLeftCorner<3, 3>().transpose();
  Matrix6d floor = Matrix6d::Zero();
  floor.diagonal().head<3>().setConstant(
    config_.translation_noise_floor * config_.translation_noise_floor);
  floor.diagonal().tail<3>().setConstant(
    config_.rotation_noise_floor * config_.rotation_noise_floor);
  Matrix6d measured_frame = Matrix6d::Identity();
  measured_frame.bottomRightCorner<3, 3>() =
    expected.topLeftCorner<3, 3>().transpose() * measured.topLeftCorner<3, 3>();
  // Transport each independent robot-pose uncertainty, including its noise
  // floor, to the comparison point and rotational tangent frame.
  const Matrix6d covariance = jacobian * (anchor.covariance + floor) * jacobian.transpose() +
    measured_frame * (candidate.covariance + floor) * measured_frame.transpose();
  const Eigen::LDLT<Matrix6d> solver(covariance);
  if (solver.info() != Eigen::Success || !solver.isPositive()) {
    return false;
  }
  const double nis = error.dot(solver.solve(error));
  if (!std::isfinite(nis)) {
    return false;
  }
  out.nis = std::max(out.nis, nis);
  return error.head<3>().norm() <= config_.max_translation_delta &&
         error.tail<3>().norm() <= config_.max_rotation_delta && nis <= config_.nis_threshold;
}

Result RecoveryCoordinator::observe(
  const Candidate & candidate, const Eigen::Matrix4d & current_map_to_odom, double now)
{
  if (expire(now)) {
    return result("expired_or_clock_reset");
  }
  if (candidate.generation != generation_) {
    return result("stale_generation");
  }
  if (state_ == State::VERIFYING) {
    return observeVerification(candidate, now);
  }
  if (!valid_candidate(candidate, now) || !valid_pose(current_map_to_odom) ||
    now - candidate.window.end_stamp > config_.candidate_timeout)
  {
    if (state_ == State::CONFIRMING) {
      clear();
    }
    return result("invalid_or_stale_candidate");
  }
  if (state_ == State::PROPOSED || state_ == State::VERIFIED) {
    return result("awaiting_application_or_invalidation");
  }
  if (previous_ && !disjoint(previous_->window, candidate.window)) {
    return result("overlapping_or_out_of_order_window");
  }
  if (!candidate.quality.accepted()) {
    clear();
    return result("candidate_quality_rejected");
  }
  Result diagnostics;
  if (first_ && (!consistent(*first_, candidate, diagnostics) ||
    !consistent(*previous_, candidate, diagnostics)))
  {
    clear();
    first_ = candidate;
    previous_ = candidate;
    count_ = 1;
    last_stamp_ = candidate.window.end_stamp;
    state_ = State::CONFIRMING;
    Result out = result("candidate_inconsistent");
    out.candidate_recorded = true;
    out.translation_delta = diagnostics.translation_delta;
    out.rotation_delta = diagnostics.rotation_delta;
    out.nis = diagnostics.nis;
    return out;
  }
  if (!first_) {
    first_ = candidate;
  }
  previous_ = candidate;
  last_stamp_ = candidate.window.end_stamp;
  ++count_;
  state_ = State::CONFIRMING;
  if (count_ >= config_.confirmation_windows) {
    if (!bounded_active_) {
      Config full_config = config_;
      full_config.application_mode = ApplicationMode::RESET;
      target_pose_ = project(current_map_to_odom, candidate.map_to_odom, full_config);
    }
    requested_pose_ = project(current_map_to_odom, target_pose_, config_);
    state_ = State::PROPOSED;
  }
  Result out = result(state_ == State::PROPOSED ? "proposal_ready" : "confirming");
  out.candidate_recorded = true;
  out.proposal_available = state_ == State::PROPOSED;
  out.should_apply = out.proposal_available && config_.application_mode != ApplicationMode::SHADOW;
  out.translation_delta = diagnostics.translation_delta;
  out.rotation_delta = diagnostics.rotation_delta;
  out.nis = diagnostics.nis;
  return out;
}

bool RecoveryCoordinator::acknowledgeApplied(
  std::uint64_t generation, const Eigen::Matrix4d & applied_pose)
{
  if (generation != generation_ || state_ != State::PROPOSED ||
    config_.application_mode == ApplicationMode::SHADOW || !valid_pose(applied_pose) ||
    !applied_pose.isApprox(requested_pose_, 1e-9))
  {
    return false;
  }
  if (config_.application_mode == ApplicationMode::BOUNDED &&
    !applied_pose.isApprox(target_pose_, 1e-9))
  {
    // Keep both target anchors: a fresh independent window must reconfirm the
    // hypothesis before every bounded step. Never verify an intermediate pose.
    state_ = State::CONFIRMING;
    bounded_active_ = true;
    count_ = config_.confirmation_windows - 1;
    return true;
  }
  state_ = State::VERIFYING;
  verification_count_ = 0;
  verification_window_ = previous_->window;
  // Verification is against the actual applied (possibly bounded/masked)
  // transform, not the unprojected recovery candidate.
  first_->map_to_odom = applied_pose;
  previous_->map_to_odom = applied_pose;
  return true;
}

Result RecoveryCoordinator::observeVerification(const Candidate & incumbent, double now)
{
  if (expire(now)) {
    return result("expired_or_clock_reset");
  }
  if (incumbent.generation != generation_) {
    return result("stale_generation");
  }
  if (state_ != State::VERIFYING) {
    return result("not_verifying");
  }
  if (!valid_window(incumbent.window) || incumbent.window.end_stamp > now ||
    now - incumbent.window.end_stamp > config_.candidate_timeout ||
    !disjoint(*verification_window_, incumbent.window))
  {
    return result("invalid_or_overlapping_verification_window");
  }
  Result diagnostics;
  if (!incumbent.quality.accepted() || !valid_candidate(incumbent, now) ||
    !consistent(*first_, incumbent, diagnostics) ||
    !consistent(*previous_, incumbent, diagnostics))
  {
    invalidate();
    return result("verification_failed");
  }
  previous_ = incumbent;
  verification_window_ = incumbent.window;
  last_stamp_ = incumbent.window.end_stamp;
  ++verification_count_;
  if (verification_count_ >= config_.verification_windows) {
    state_ = State::VERIFIED;
  }
  Result out = result(state_ == State::VERIFIED ? "verified" : "verifying");
  out.candidate_recorded = true;
  out.translation_delta = diagnostics.translation_delta;
  out.rotation_delta = diagnostics.rotation_delta;
  out.nis = diagnostics.nis;
  return out;
}

std::uint64_t RecoveryCoordinator::generation() const {return generation_;}
State RecoveryCoordinator::state() const {return state_;}
const Config & RecoveryCoordinator::config() const {return config_;}
}  // namespace recovery

namespace
{
recovery::Config core_config(const RecoveryCoordinatorConfig & input)
{
  recovery::Config out;
  if (input.application_mode == "reset") {
    out.application_mode = recovery::ApplicationMode::RESET;
  } else if (input.application_mode == "shadow") {
    out.application_mode = recovery::ApplicationMode::SHADOW;
  } else if (input.application_mode == "bounded" || input.application_mode == "bounded_step") {
    out.application_mode = recovery::ApplicationMode::BOUNDED;
  } else {
    throw std::invalid_argument("application_mode must be reset, shadow, bounded or bounded_step");
  }
  if (!std::isfinite(input.minimum_interval) || input.minimum_interval < 0 ||
    input.minimum_interval > input.timeout)
  {
    throw std::invalid_argument("minimum_interval must be finite and within timeout");
  }
  out.update_mask = input.update_mask;
  out.confirmation_windows = input.required_observations;
  out.verification_windows = input.verification_observations;
  out.translation_noise_floor = input.translation_stddev;
  out.rotation_noise_floor = input.rotation_stddev;
  out.nis_threshold = input.mahalanobis_threshold;
  out.max_translation_delta = input.translation_limit;
  out.max_rotation_delta = input.rotation_limit;
  out.candidate_timeout = input.timeout;
  out.max_step_translation = input.max_step_translation;
  out.max_step_rotation = input.max_step_rotation;
  return out;
}
}  // namespace

RecoveryCoordinator::RecoveryCoordinator(const RecoveryCoordinatorConfig & config)
: config_(config), core_(core_config(config))
{
}

void RecoveryCoordinator::invalidate(std::uint64_t generation)
{
  generation_ = generation;
  core_.invalidate();
  first_.reset();
  previous_.reset();
  count_ = 0;
  pending_ = false;
  state_ = RelocalizationState::TRACKING;
  wall_now_.reset();
}

void RecoveryCoordinator::searching()
{
  if (state_ == RelocalizationState::CONFIRMING || state_ == RelocalizationState::VERIFYING) {
    return;
  }
  core_.invalidate();
  first_.reset();
  previous_.reset();
  pending_ = false;
  count_ = 0;
  state_ = RelocalizationState::SEARCHING;
}

void RecoveryCoordinator::anchor(const RecoveryObservation & observation)
{
  if (!first_) {
    first_ = observation;
  }
  previous_ = observation;
}

RecoveryDecision RecoveryCoordinator::observe(
  const RecoveryObservation & observation, const Eigen::Matrix4d & current_map_odom)
{
  RecoveryDecision decision;
  decision.count = count_;
  if (observation.generation != generation_) {
    decision.reason = "generation_mismatch";
    return decision;
  }
  if (observation.scan_ids.empty() ||
    !std::is_sorted(observation.scan_ids.begin(), observation.scan_ids.end()) ||
    std::adjacent_find(observation.scan_ids.begin(), observation.scan_ids.end()) !=
    observation.scan_ids.end())
  {
    decision.reason = "invalid_scan_ids";
    return decision;
  }
  // Quality failures in verification must reach the state machine even when
  // registration did not produce a usable pose or covariance.
  const bool failed_verification = state_ == RelocalizationState::VERIFYING &&
    !(observation.quality_valid && observation.unambiguous && observation.observable);
  if (!failed_verification && previous_ &&
    observation.stamp >= previous_->stamp &&
    observation.stamp - previous_->stamp < config_.minimum_interval)
  {
    decision.reason = "minimum_interval";
    return decision;
  }
  recovery::Candidate input;
  input.generation = core_.generation();
  input.window = {observation.scan_ids.front(), observation.scan_ids.back(),
    observation.oldest_stamp, observation.stamp};
  input.odom_to_robot = observation.odom_base;
  input.map_to_odom = observation.map_base * observation.odom_base.inverse();
  input.covariance = observation.covariance;
  if (!recovery::valid_covariance(observation.odom_covariance)) {
    input.covariance.setConstant(std::numeric_limits<double>::quiet_NaN());
  } else {
    Matrix6d transport = Matrix6d::Identity();
    transport.topLeftCorner<3, 3>() = input.map_to_odom.topLeftCorner<3, 3>();
    transport.bottomRightCorner<3, 3>() = input.map_to_odom.topLeftCorner<3, 3>();
    input.covariance += transport * observation.odom_covariance * transport.transpose();
  }
  Matrix6d local_rotation = Matrix6d::Identity();
  local_rotation.bottomRightCorner<3, 3>() = observation.map_base.topLeftCorner<3, 3>().transpose();
  input.covariance = local_rotation * input.covariance * local_rotation.transpose();
  input.quality = {observation.quality_valid, observation.unambiguous, observation.observable};
  const double now = wall_now_ ? std::max(*wall_now_, observation.stamp) : observation.stamp;
  const recovery::Result out = core_.observe(input, current_map_odom, now);
  decision.proposal = out.proposal_available;
  decision.shadow = out.proposal_available && !out.should_apply;
  decision.map_odom = out.requested_pose;
  decision.translation_residual = out.translation_delta;
  decision.rotation_residual = out.rotation_delta;
  decision.mahalanobis = out.nis;
  decision.reason = out.reason;
  if (out.reason == "overlapping_or_out_of_order_window") {
    decision.reason = "overlapping_scan_window";
  }
  decision.covariance = observation.covariance;
  switch (out.state) {
    case recovery::State::IDLE:
      state_ = RelocalizationState::DEGRADED;
      first_.reset();
      previous_.reset();
      pending_ = false;
      break;
    case recovery::State::CONFIRMING:
    case recovery::State::PROPOSED:
      state_ = RelocalizationState::CONFIRMING;
      break;
    case recovery::State::VERIFYING:
      state_ = RelocalizationState::VERIFYING;
      break;
    case recovery::State::VERIFIED:
      state_ = RelocalizationState::TRACKING;
      break;
  }
  count_ = out.state == recovery::State::VERIFYING || out.state == recovery::State::VERIFIED ?
    out.verified_windows : out.consistent_windows;
  decision.count = count_;
  if (out.candidate_recorded) {
    anchor(observation);
  }
  if (out.proposal_available) {
    pending_ = true;
    pending_pose_ = out.requested_pose;
  }
  if (out.verified) {
    core_.invalidate();
    first_.reset();
    previous_.reset();
  }
  return decision;
}

bool RecoveryCoordinator::acknowledgeApplied(
  std::uint64_t generation, const Eigen::Matrix4d & applied_pose)
{
  if (generation != generation_ || !pending_ ||
    !core_.acknowledgeApplied(core_.generation(), applied_pose))
  {
    return false;
  }
  pending_ = false;
  const bool continuing = core_.state() == recovery::State::CONFIRMING;
  state_ = continuing ? RelocalizationState::CONFIRMING : RelocalizationState::VERIFYING;
  count_ = continuing ? config_.required_observations - 1 : 0;
  return true;
}

void RecoveryCoordinator::applied(
  std::uint64_t generation, const RecoveryObservation & observation)
{
  if (observation.generation != generation_ || generation < generation_ ||
    generation - generation_ > 1 || !recovery::valid_pose(observation.odom_base))
  {
    return;
  }
  if (acknowledgeApplied(generation_, observation.map_base * observation.odom_base.inverse())) {
    generation_ = generation;
  }
}

void RecoveryCoordinator::rejectedProposal()
{
  core_.invalidate();
  first_.reset();
  previous_.reset();
  pending_ = false;
  count_ = 0;
  state_ = RelocalizationState::DEGRADED;
}

void RecoveryCoordinator::expire(double now)
{
  wall_now_ = now;
  const recovery::Result out = core_.tick(now);
  if (out.reason == "expired_or_clock_reset") {
    first_.reset();
    previous_.reset();
    pending_ = false;
    count_ = 0;
    state_ = RelocalizationState::DEGRADED;
  }
}

const char * RecoveryCoordinator::stateName(RelocalizationState state)
{
  switch (state) {
    case RelocalizationState::TRACKING: return "tracking";
    case RelocalizationState::SEARCHING: return "searching";
    case RelocalizationState::CONFIRMING: return "confirming";
    case RelocalizationState::VERIFYING: return "verifying";
    case RelocalizationState::DEGRADED: return "degraded";
  }
  return "unknown";
}

Matrix6d RecoveryCoordinator::correctionCovariance(
  const Matrix6d & robot_covariance, const Eigen::Matrix4d & map_odom,
  const Eigen::Matrix4d & odom_base)
{
  if (!recovery::valid_pose(map_odom) || !recovery::valid_pose(odom_base) ||
    !recovery::valid_covariance(robot_covariance))
  {
    throw std::invalid_argument("Invalid pose or covariance");
  }
  Matrix6d jacobian = Matrix6d::Identity();
  jacobian.topRightCorner<3, 3>() = recovery::skew(
    map_odom.topLeftCorner<3, 3>() * odom_base.topRightCorner<3, 1>());
  return jacobian * robot_covariance * jacobian.transpose();
}

Eigen::Matrix4d RecoveryCoordinator::maskedPose(
  const Eigen::Matrix4d & current, const Eigen::Matrix4d & target,
  const std::array<bool, 6> & mask)
{
  if (!recovery::valid_pose(current) || !recovery::valid_pose(target)) {
    throw std::invalid_argument("Invalid pose");
  }
  recovery::Config config;
  config.application_mode = recovery::ApplicationMode::RESET;
  config.update_mask = mask;
  return recovery::project(current, target, config);
}
}  // namespace open3d_loc
