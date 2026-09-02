// Copyright 2026 Open3D Loc Contributors
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd
#include "open3d_loc/global_correction_filter.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <vector>

namespace open3d_loc
{

namespace
{
constexpr double kMinimumVariance = 1e-12;
}

GlobalCorrectionFilter::GlobalCorrectionFilter(const GlobalCorrectionFilterConfig & config)
{
  configure(config);
}

void GlobalCorrectionFilter::configure(const GlobalCorrectionFilterConfig & config)
{
  config_ = config;
  for (int i = 0; i < 6; ++i) {
    config_.initial_stddev(i) = std::max(config_.initial_stddev(i), 0.0);
    config_.process_stddev_time(i) = std::max(config_.process_stddev_time(i), 0.0);
    config_.process_stddev_distance(i) =
      std::max(config_.process_stddev_distance(i), 0.0);
    config_.process_stddev_rotation(i) =
      std::max(config_.process_stddev_rotation(i), 0.0);
    config_.measurement_stddev_floor(i) =
      std::max(config_.measurement_stddev_floor(i), 0.0);
  }
  config_.recovery.required_consistent_measurements =
    std::max(config_.recovery.required_consistent_measurements, 1);
  config_.recovery.minimum_candidate_interval =
    std::max(config_.recovery.minimum_candidate_interval, 0.0);
  config_.recovery.candidate_timeout = std::max(
    config_.recovery.candidate_timeout,
    config_.recovery.minimum_candidate_interval);
  config_.recovery.consistency_mahalanobis_threshold =
    std::max(config_.recovery.consistency_mahalanobis_threshold, 0.0);
  config_.recovery.max_candidate_translation_delta =
    std::max(config_.recovery.max_candidate_translation_delta, 0.0);
  config_.recovery.max_candidate_rotation_delta =
    std::max(config_.recovery.max_candidate_rotation_delta, 0.0);
  config_.recovery.max_step_translation =
    std::max(config_.recovery.max_step_translation, 0.0);
  config_.recovery.max_step_rotation =
    std::max(config_.recovery.max_step_rotation, 0.0);
  clearRecovery();
}

void GlobalCorrectionFilter::reset(const Eigen::Matrix4d & map_to_odom)
{
  Matrix6d covariance = config_.initial_stddev.array().square().matrix().asDiagonal();
  reset(map_to_odom, covariance);
}

void GlobalCorrectionFilter::reset(
  const Eigen::Matrix4d & map_to_odom, const Matrix6d & covariance)
{
  pose_ = validTransform(map_to_odom) ? map_to_odom : Eigen::Matrix4d::Identity();
  covariance_ = regularizeCovariance(covariance, Vector6d::Constant(1e-6));
  initialized_ = true;
  clearRecovery();
}

void GlobalCorrectionFilter::predict(double dt, double distance, double rotation)
{
  if (!initialized_) {
    return;
  }
  if (!std::isfinite(dt) || !std::isfinite(distance) || !std::isfinite(rotation)) {
    return;
  }
  dt = std::max(dt, 0.0);
  distance = std::max(distance, 0.0);
  rotation = std::max(rotation, 0.0);
  const Vector6d variance =
    config_.process_stddev_time.array().square() * dt +
    config_.process_stddev_distance.array().square() * distance +
    config_.process_stddev_rotation.array().square() * rotation;
  covariance_ += variance.matrix().asDiagonal();
  covariance_ = 0.5 * (covariance_ + covariance_.transpose());
}

CorrectionUpdateResult GlobalCorrectionFilter::update(
  const Eigen::Matrix4d & measurement, const Matrix6d & measurement_covariance)
{
  CorrectionUpdateResult result;
  if (!initialized_) {
    result.reason = "not_initialized";
    return result;
  }
  if (!validTransform(measurement) || !measurement_covariance.allFinite()) {
    result.reason = "invalid_measurement";
    return result;
  }

  const Vector6d residual = poseResidual(pose_, measurement);
  Vector6d gated_residual = residual;
  for (int i = 0; i < 6; ++i) {
    if (!config_.update_mask[static_cast<std::size_t>(i)]) {
      gated_residual(i) = 0.0;
    }
  }
  result.translation_innovation = gated_residual.head<3>().norm();
  result.rotation_innovation = gated_residual.tail<3>().norm();
  if (result.translation_innovation > config_.max_innovation_translation) {
    result.reason = "translation_gate";
    return result;
  }
  if (result.rotation_innovation > config_.max_innovation_rotation) {
    result.reason = "rotation_gate";
    return result;
  }

  std::vector<int> indices;
  for (int i = 0; i < 6; ++i) {
    if (config_.update_mask[static_cast<std::size_t>(i)]) {
      indices.push_back(i);
    }
  }
  if (indices.empty()) {
    result.reason = "empty_update_mask";
    return result;
  }

  const Matrix6d measurement_covariance_regularized =
    regularizeCovariance(measurement_covariance, config_.measurement_stddev_floor);
  const int dimension = static_cast<int>(indices.size());
  Eigen::MatrixXd observation = Eigen::MatrixXd::Zero(dimension, 6);
  Eigen::VectorXd innovation(dimension);
  Eigen::MatrixXd measurement_noise(dimension, dimension);
  for (int row = 0; row < dimension; ++row) {
    observation(row, indices[static_cast<std::size_t>(row)]) = 1.0;
    innovation(row) = residual(indices[static_cast<std::size_t>(row)]);
    for (int column = 0; column < dimension; ++column) {
      measurement_noise(row, column) = measurement_covariance_regularized(
        indices[static_cast<std::size_t>(row)], indices[static_cast<std::size_t>(column)]);
    }
  }

  const Eigen::MatrixXd innovation_covariance = observation * covariance_ *
    observation.transpose() + measurement_noise;
  const Eigen::LDLT<Eigen::MatrixXd> solver(innovation_covariance);
  if (solver.info() != Eigen::Success || !solver.isPositive()) {
    result.reason = "singular_innovation_covariance";
    return result;
  }
  result.mahalanobis_distance = innovation.dot(solver.solve(innovation));
  if (!std::isfinite(result.mahalanobis_distance) ||
    result.mahalanobis_distance > config_.mahalanobis_threshold)
  {
    result.reason = "mahalanobis_gate";
    return result;
  }

  Eigen::MatrixXd gain =
    covariance_ * observation.transpose() *
    solver.solve(Eigen::MatrixXd::Identity(dimension, dimension));
  for (int i = 0; i < 6; ++i) {
    if (!config_.update_mask[static_cast<std::size_t>(i)]) {
      gain.row(i).setZero();
    }
  }
  const Vector6d error = gain * innovation;
  applyError(error);

  const Matrix6d identity = Matrix6d::Identity();
  const Matrix6d correction = identity - gain * observation;
  covariance_ = correction * covariance_ * correction.transpose() +
    gain * measurement_noise * gain.transpose();
  covariance_ = regularizeCovariance(covariance_, Vector6d::Constant(1e-6));

  result.accepted = true;
  result.reason = "accepted";
  clearRecovery();
  return result;
}

RecoveryUpdateResult GlobalCorrectionFilter::observeRecoveryCandidate(
  const Eigen::Matrix4d & measurement,
  const Matrix6d & measurement_covariance,
  const double stamp_seconds,
  const double cumulative_distance,
  const double cumulative_rotation)
{
  RecoveryUpdateResult result;
  result.mode = recovery_mode_;
  result.consistent_measurements = recovery_candidate_count_;
  if (!config_.recovery.enabled) {
    result.reason = "recovery_disabled";
    return result;
  }
  if (!initialized_) {
    result.reason = "not_initialized";
    return result;
  }
  if (!validTransform(measurement) || !measurement_covariance.allFinite() ||
    !std::isfinite(stamp_seconds) || !std::isfinite(cumulative_distance) ||
    !std::isfinite(cumulative_rotation))
  {
    clearRecovery();
    result.mode = recovery_mode_;
    result.reason = "invalid_recovery_candidate";
    return result;
  }

  if (!have_recovery_candidate_) {
    return recordFirstRecoveryCandidate(
      measurement, measurement_covariance, stamp_seconds,
      cumulative_distance, cumulative_rotation, "recovery_confirmation_started");
  }

  result.candidate_interval = stamp_seconds - recovery_candidate_stamp_;
  if (result.candidate_interval <= 0.0 ||
    cumulative_distance < recovery_candidate_cumulative_distance_ ||
    cumulative_rotation < recovery_candidate_cumulative_rotation_)
  {
    clearRecovery();
    return recordFirstRecoveryCandidate(
      measurement, measurement_covariance, stamp_seconds,
      cumulative_distance, cumulative_rotation, "recovery_history_reset");
  }
  if (result.candidate_interval < config_.recovery.minimum_candidate_interval) {
    result.reason = "recovery_candidate_too_soon";
    return result;
  }
  if (result.candidate_interval > config_.recovery.candidate_timeout) {
    clearRecovery();
    return recordFirstRecoveryCandidate(
      measurement, measurement_covariance, stamp_seconds,
      cumulative_distance, cumulative_rotation, "recovery_candidate_timeout");
  }

  result.motion_distance =
    cumulative_distance - recovery_candidate_cumulative_distance_;
  result.motion_rotation =
    cumulative_rotation - recovery_candidate_cumulative_rotation_;
  Vector6d residual = poseResidual(recovery_candidate_pose_, measurement);
  for (int i = 0; i < 6; ++i) {
    if (!config_.update_mask[static_cast<std::size_t>(i)]) {
      residual(i) = 0.0;
    }
  }
  result.candidate_translation_delta = residual.head<3>().norm();
  result.candidate_rotation_delta = residual.tail<3>().norm();

  const Vector6d process_variance =
    config_.process_stddev_time.array().square() * result.candidate_interval +
    config_.process_stddev_distance.array().square() * result.motion_distance +
    config_.process_stddev_rotation.array().square() * result.motion_rotation;
  const Matrix6d previous_covariance = regularizeCovariance(
    recovery_candidate_covariance_, config_.measurement_stddev_floor);
  const Matrix6d current_covariance = regularizeCovariance(
    measurement_covariance, config_.measurement_stddev_floor);
  Matrix6d pair_covariance = previous_covariance + current_covariance;
  pair_covariance.diagonal() += process_variance;

  std::vector<int> indices;
  for (int i = 0; i < 6; ++i) {
    if (config_.update_mask[static_cast<std::size_t>(i)]) {
      indices.push_back(i);
    }
  }
  if (indices.empty()) {
    clearRecovery();
    result.mode = recovery_mode_;
    result.reason = "empty_update_mask";
    return result;
  }

  const int dimension = static_cast<int>(indices.size());
  Eigen::VectorXd active_residual(dimension);
  Eigen::MatrixXd active_covariance(dimension, dimension);
  for (int row = 0; row < dimension; ++row) {
    active_residual(row) = residual(indices[static_cast<std::size_t>(row)]);
    for (int column = 0; column < dimension; ++column) {
      active_covariance(row, column) = pair_covariance(
        indices[static_cast<std::size_t>(row)],
        indices[static_cast<std::size_t>(column)]);
    }
  }
  const Eigen::LDLT<Eigen::MatrixXd> solver(active_covariance);
  if (solver.info() != Eigen::Success || !solver.isPositive()) {
    clearRecovery();
    result.mode = recovery_mode_;
    result.reason = "singular_recovery_covariance";
    return result;
  }
  result.consistency_mahalanobis_distance =
    active_residual.dot(solver.solve(active_residual));

  std::string inconsistency_reason;
  if (result.candidate_translation_delta >
    config_.recovery.max_candidate_translation_delta)
  {
    inconsistency_reason = "recovery_translation_inconsistent";
  }
  if (inconsistency_reason.empty() &&
    result.candidate_rotation_delta > config_.recovery.max_candidate_rotation_delta)
  {
    inconsistency_reason = "recovery_rotation_inconsistent";
  }
  if (inconsistency_reason.empty() &&
    (!std::isfinite(result.consistency_mahalanobis_distance) ||
    result.consistency_mahalanobis_distance >
    config_.recovery.consistency_mahalanobis_threshold))
  {
    inconsistency_reason = "recovery_mahalanobis_inconsistent";
  }
  if (!inconsistency_reason.empty()) {
    clearRecovery();
    return recordFirstRecoveryCandidate(
      measurement, measurement_covariance, stamp_seconds,
      cumulative_distance, cumulative_rotation, inconsistency_reason);
  }

  storeRecoveryCandidate(
    measurement, measurement_covariance, stamp_seconds,
    cumulative_distance, cumulative_rotation);
  recovery_candidate_count_ += 1;
  result.candidate_recorded = true;
  result.consistent_measurements = recovery_candidate_count_;
  if (recovery_candidate_count_ < config_.recovery.required_consistent_measurements) {
    recovery_mode_ = RecoveryMode::CONFIRMING;
    result.mode = recovery_mode_;
    result.reason = "recovery_candidate_confirmed";
    return result;
  }

  recovery_mode_ = RecoveryMode::RECOVERING;
  result.mode = recovery_mode_;
  return applyRecoveryStep(measurement, measurement_covariance, result);
}

void GlobalCorrectionFilter::clearRecovery()
{
  recovery_mode_ = RecoveryMode::TRACKING;
  recovery_candidate_count_ = 0;
  have_recovery_candidate_ = false;
  recovery_candidate_pose_ = Eigen::Matrix4d::Identity();
  recovery_candidate_covariance_ = Matrix6d::Identity();
  recovery_candidate_stamp_ = 0.0;
  recovery_candidate_cumulative_distance_ = 0.0;
  recovery_candidate_cumulative_rotation_ = 0.0;
}

bool GlobalCorrectionFilter::initialized() const
{
  return initialized_;
}

const Eigen::Matrix4d & GlobalCorrectionFilter::pose() const
{
  return pose_;
}

const Matrix6d & GlobalCorrectionFilter::covariance() const
{
  return covariance_;
}

RecoveryMode GlobalCorrectionFilter::recoveryMode() const
{
  return recovery_mode_;
}

int GlobalCorrectionFilter::recoveryCandidateCount() const
{
  return recovery_candidate_count_;
}

const char * GlobalCorrectionFilter::recoveryModeName(const RecoveryMode mode)
{
  switch (mode) {
    case RecoveryMode::TRACKING:
      return "tracking";
    case RecoveryMode::CONFIRMING:
      return "confirming";
    case RecoveryMode::RECOVERING:
      return "recovering";
  }
  return "unknown";
}

Matrix6d GlobalCorrectionFilter::regularizeCovariance(
  const Matrix6d & covariance, const Vector6d & stddev_floor)
{
  Matrix6d symmetric = 0.5 * (covariance + covariance.transpose());
  if (!symmetric.allFinite()) {
    return stddev_floor.array().square().max(kMinimumVariance).matrix().asDiagonal();
  }
  Eigen::SelfAdjointEigenSolver<Matrix6d> solver(symmetric);
  if (solver.info() != Eigen::Success) {
    return stddev_floor.array().square().max(kMinimumVariance).matrix().asDiagonal();
  }
  Eigen::Matrix<double, 6, 1> eigenvalues = solver.eigenvalues();
  for (int i = 0; i < 6; ++i) {
    eigenvalues(i) = std::max(eigenvalues(i), kMinimumVariance);
  }
  symmetric = solver.eigenvectors() * eigenvalues.asDiagonal() * solver.eigenvectors().transpose();
  for (int i = 0; i < 6; ++i) {
    symmetric(i, i) = std::max(symmetric(i, i), stddev_floor(i) * stddev_floor(i));
  }
  return 0.5 * (symmetric + symmetric.transpose());
}

Vector6d GlobalCorrectionFilter::poseResidual(
  const Eigen::Matrix4d & estimate, const Eigen::Matrix4d & measurement)
{
  Vector6d residual = Vector6d::Zero();
  residual.head<3>() = measurement.block<3, 1>(0, 3) - estimate.block<3, 1>(0, 3);
  const Eigen::Matrix3d rotation_error =
    measurement.block<3, 3>(0, 0) * estimate.block<3, 3>(0, 0).transpose();
  const Eigen::AngleAxisd angle_axis(rotation_error);
  if (std::isfinite(angle_axis.angle()) && angle_axis.axis().allFinite()) {
    residual.tail<3>() = angle_axis.angle() * angle_axis.axis();
  }
  return residual;
}

bool GlobalCorrectionFilter::validTransform(const Eigen::Matrix4d & transform)
{
  if (!transform.allFinite() ||
    !transform.row(3).isApprox(Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), 1e-6))
  {
    return false;
  }
  const Eigen::Matrix3d rotation = transform.block<3, 3>(0, 0);
  return rotation.transpose().isApprox(rotation.inverse(), 1e-5) &&
         std::abs(rotation.determinant() - 1.0) < 1e-5;
}

void GlobalCorrectionFilter::applyError(const Vector6d & error)
{
  pose_.block<3, 1>(0, 3) += error.head<3>();
  const Eigen::Vector3d rotation_error = error.tail<3>();
  const double angle = rotation_error.norm();
  if (angle > 1e-12) {
    pose_.block<3, 3>(0, 0) =
      Eigen::AngleAxisd(angle, rotation_error / angle).toRotationMatrix() *
      pose_.block<3, 3>(0, 0);
  }
}

RecoveryUpdateResult GlobalCorrectionFilter::recordFirstRecoveryCandidate(
  const Eigen::Matrix4d & measurement,
  const Matrix6d & measurement_covariance,
  const double stamp_seconds,
  const double cumulative_distance,
  const double cumulative_rotation,
  const std::string & reason)
{
  storeRecoveryCandidate(
    measurement, measurement_covariance, stamp_seconds,
    cumulative_distance, cumulative_rotation);
  recovery_candidate_count_ = 1;
  recovery_mode_ = config_.recovery.required_consistent_measurements == 1 ?
    RecoveryMode::RECOVERING : RecoveryMode::CONFIRMING;

  RecoveryUpdateResult result;
  result.candidate_recorded = true;
  result.mode = recovery_mode_;
  result.consistent_measurements = recovery_candidate_count_;
  result.reason = reason;
  if (recovery_mode_ == RecoveryMode::RECOVERING) {
    return applyRecoveryStep(measurement, measurement_covariance, result);
  }
  return result;
}

void GlobalCorrectionFilter::storeRecoveryCandidate(
  const Eigen::Matrix4d & measurement,
  const Matrix6d & measurement_covariance,
  const double stamp_seconds,
  const double cumulative_distance,
  const double cumulative_rotation)
{
  recovery_candidate_pose_ = measurement;
  recovery_candidate_covariance_ = measurement_covariance;
  recovery_candidate_stamp_ = stamp_seconds;
  recovery_candidate_cumulative_distance_ = cumulative_distance;
  recovery_candidate_cumulative_rotation_ = cumulative_rotation;
  have_recovery_candidate_ = true;
}

RecoveryUpdateResult GlobalCorrectionFilter::applyRecoveryStep(
  const Eigen::Matrix4d & measurement,
  const Matrix6d & measurement_covariance,
  RecoveryUpdateResult result)
{
  Vector6d error = poseResidual(pose_, measurement);
  for (int i = 0; i < 6; ++i) {
    if (!config_.update_mask[static_cast<std::size_t>(i)]) {
      error(i) = 0.0;
    }
  }

  const double translation_norm = error.head<3>().norm();
  if (translation_norm > config_.recovery.max_step_translation && translation_norm > 0.0) {
    error.head<3>() *= config_.recovery.max_step_translation / translation_norm;
  }
  const double rotation_norm = error.tail<3>().norm();
  if (rotation_norm > config_.recovery.max_step_rotation && rotation_norm > 0.0) {
    error.tail<3>() *= config_.recovery.max_step_rotation / rotation_norm;
  }

  applyError(error);
  covariance_ = regularizeCovariance(
    measurement_covariance, config_.initial_stddev);
  result.correction_applied = true;
  result.applied_translation = error.head<3>().norm();
  result.applied_rotation = error.tail<3>().norm();
  result.mode = recovery_mode_;
  result.reason = "recovery_step_applied";
  return result;
}

}  // namespace open3d_loc
