// Copyright 2026 Open3D Loc Contributors
// Offline evaluation only: scan PCD coordinates must already be in odom.
// Usage: relocalization_replay MAP.pcd MANIFEST.csv OUTPUT.csv
//          [--inject DX DY DZ YAW_DEGREES]
// Manifest: stamp,scan_path,odom_x,odom_y,odom_z,odom_qx,odom_qy,odom_qz,odom_qw,
//           reference_x,reference_y,reference_z,reference_qx,reference_qy,
//           reference_qz,reference_qw
// Paths are relative to the manifest directory. Optional header starts "stamp".
// Reference map_base initializes the baseline on the FIRST row only; thereafter
// reference poses are used exclusively for reporting errors, never for search.
#include <Eigen/Geometry>
#include <open3d/io/PointCloudIO.h>
#include <open3d/utility/Logging.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "open3d_loc/global_correction_filter.hpp"
#include "open3d_loc/recovery_coordinator.hpp"
#include "open3d_loc/recovery_search.hpp"

namespace
{
constexpr double kPi = 3.14159265358979323846;
using Clock = std::chrono::steady_clock;
using Cloud = open3d::geometry::PointCloud;

struct Frame
{
  double stamp;
  std::filesystem::path scan_path;
  Eigen::Matrix4d odom_base;
  Eigen::Matrix4d reference_map_base;
};

std::string trim(const std::string & value)
{
  const auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return {};
  }
  return value.substr(begin, value.find_last_not_of(" \t\r\n") - begin + 1);
}

std::vector<std::string> csv_fields(const std::string & line)
{
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  bool closed = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (quoted) {
      if (c == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') {
          field += '"';
          ++i;
        } else {
          quoted = false;
          closed = true;
        }
      } else {
        field += c;
      }
    } else if (c == ',') {
      fields.push_back(trim(field));
      field.clear();
      closed = false;
    } else if (c == '"' && trim(field).empty() && !closed) {
      field.clear();
      quoted = true;
    } else {
      if (c == '"' || (closed && c != ' ' && c != '\t' && c != '\r')) {
        throw std::runtime_error("invalid CSV quoting");
      }
      field += c;
    }
  }
  if (quoted) {
    throw std::runtime_error("unterminated CSV quote");
  }
  fields.push_back(trim(field));
  return fields;
}

double number(const std::string & text)
{
  std::size_t consumed = 0;
  const double value = std::stod(text, &consumed);
  if (consumed != text.size() || !std::isfinite(value)) {
    throw std::runtime_error("expected finite number: " + text);
  }
  return value;
}

Eigen::Matrix4d read_pose(const std::vector<std::string> & fields, std::size_t begin)
{
  Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
  result.block<3, 1>(0, 3) = Eigen::Vector3d(
    number(fields.at(begin)), number(fields.at(begin + 1)), number(fields.at(begin + 2)));
  Eigen::Quaterniond rotation(
    number(fields.at(begin + 6)), number(fields.at(begin + 3)),
    number(fields.at(begin + 4)), number(fields.at(begin + 5)));
  if (!std::isfinite(rotation.norm()) || rotation.norm() < 1e-12) {
    throw std::runtime_error("invalid quaternion");
  }
  result.block<3, 3>(0, 0) = rotation.normalized().toRotationMatrix();
  return result;
}

std::vector<Frame> read_manifest(const std::filesystem::path & path)
{
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open manifest: " + path.string());
  }
  std::vector<Frame> frames;
  std::string line;
  std::size_t line_number = 0;
  bool header_seen = false;
  while (std::getline(input, line)) {
    ++line_number;
    if (trim(line).empty() || trim(line).front() == '#') {
      continue;
    }
    try {
      const auto fields = csv_fields(line);
      if (fields.size() != 16) {
        throw std::runtime_error("expected exactly 16 CSV fields");
      }
      if (fields.front() == "stamp" && frames.empty() && !header_seen) {
        header_seen = true;
        continue;
      }
      Frame frame;
      frame.stamp = number(fields[0]);
      if (!frames.empty() && frame.stamp <= frames.back().stamp) {
        throw std::runtime_error("timestamps must strictly increase");
      }
      if (fields[1].empty()) {
        throw std::runtime_error("empty scan path");
      }
      frame.scan_path = path.parent_path() / fields[1];
      frame.odom_base = read_pose(fields, 2);
      frame.reference_map_base = read_pose(fields, 9);
      frames.push_back(frame);
    } catch (const std::exception & error) {
      throw std::runtime_error(
              path.string() + ":" + std::to_string(line_number) + ": " + error.what());
    }
  }
  if (frames.empty()) {
    throw std::runtime_error("manifest contains no frames");
  }
  return frames;
}

Cloud read_cloud(const std::filesystem::path & path)
{
  Cloud cloud;
  if (!open3d::io::ReadPointCloud(path.string(), cloud) || cloud.points_.empty()) {
    throw std::runtime_error("cannot read nonempty cloud: " + path.string());
  }
  for (const auto & point : cloud.points_) {
    if (!point.allFinite()) {
      throw std::runtime_error("cloud contains nonfinite points: " + path.string());
    }
  }
  return cloud;
}

double rotation_error(const Eigen::Matrix4d & actual, const Eigen::Matrix4d & reference)
{
  return Eigen::AngleAxisd(
    actual.block<3, 3>(0, 0) * reference.block<3, 3>(0, 0).transpose()).angle() * 180.0 / kPi;
}

double translation_error(const Eigen::Matrix4d & actual, const Eigen::Matrix4d & reference)
{
  return (actual.block<3, 1>(0, 3) - reference.block<3, 1>(0, 3)).norm();
}

double milliseconds(Clock::time_point begin)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

Eigen::Matrix3d skew(const Eigen::Vector3d & v)
{
  Eigen::Matrix3d result;
  result << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return result;
}

open3d_loc::Matrix6d filter_covariance(
  const open3d_loc::RecoveryCandidate & candidate, const Eigen::Matrix4d & odom_base)
{
  const Eigen::Vector3d lever =
    candidate.map_odom.block<3, 3>(0, 0) * odom_base.block<3, 1>(0, 3);
  // Robot-centered map tangent -> additive map_odom translation/rotation.
  open3d_loc::Matrix6d jacobian = open3d_loc::Matrix6d::Identity();
  jacobian.block<3, 3>(0, 3) = skew(lever);
  return jacobian * candidate.covariance * jacobian.transpose();
}

open3d_loc::recovery::Candidate observation(
  const open3d_loc::RecoveryCandidate & candidate, const Frame & frame,
  std::size_t index, std::uint64_t generation, bool unambiguous)
{
  open3d_loc::recovery::Candidate result;
  result.generation = generation;
  result.window = {index + 1, index + 1, frame.stamp, frame.stamp};
  result.map_to_odom = candidate.map_odom;
  result.odom_to_robot = frame.odom_base;
  open3d_loc::Matrix6d jacobian = open3d_loc::Matrix6d::Identity();
  jacobian.block<3, 3>(3, 3) =
    (candidate.map_odom * frame.odom_base).block<3, 3>(0, 0).transpose();
  result.covariance = jacobian * candidate.covariance * jacobian.transpose();
  result.quality = {candidate.accepted, unambiguous, candidate.accepted};
  return result;
}

bool distinct(
  const Eigen::Matrix4d & a, const Eigen::Matrix4d & b, const Eigen::Matrix4d & odom_base)
{
  return translation_error(a * odom_base, b * odom_base) > 0.3 || rotation_error(a, b) > 5.0;
}

// All scores passed here must have been evaluated on the SAME fresh cloud.
bool has_advantage(double score, double competitor)
{
  return std::isfinite(score) && std::isfinite(competitor) &&
         competitor - score > 1e-9 && score <= 0.85 * competitor;
}

const char * state_name(open3d_loc::recovery::State state)
{
  using State = open3d_loc::recovery::State;
  switch (state) {
    case State::IDLE: return "idle";
    case State::CONFIRMING: return "confirming";
    case State::PROPOSED: return "proposed";
    case State::VERIFYING: return "verifying";
    case State::VERIFIED: return "verified";
  }
  return "unknown";
}

std::vector<open3d_loc::RecoveryCandidate> refine_seeds(
  const open3d_loc::RecoverySearch & search, const Cloud & scan,
  const Eigen::Matrix4d & odom_base, const std::vector<Eigen::Matrix4d> & seeds)
{
  std::vector<open3d_loc::RecoveryCandidate> results;
  results.reserve(seeds.size());
  for (const auto & seed : seeds) {
    results.push_back(search.refineFresh(scan, odom_base, seed));
  }
  return results;
}

void replay(
  const Cloud & map, const std::vector<Frame> & frames,
  const Eigen::Matrix4d & injection, std::ostream & output)
{
  using namespace open3d_loc;
  const auto preparation_begin = Clock::now();
  recovery::Config config;
  config.application_mode = recovery::ApplicationMode::RESET;
  RecoverySearchConfig search_config;
  // Estimate all six covariance axes. The coordinator separately masks the
  // applied correction; inactive search placeholders are not measurements.
  const auto shared_map = RecoverySearch::prepareMap(map, search_config);
  const double map_prepare_ms = milliseconds(preparation_begin);
  recovery::RecoveryCoordinator coordinator(config);
  GlobalCorrectionFilter filter;
  Eigen::Matrix4d initial_base = frames.front().reference_map_base;
  initial_base.block<3, 3>(0, 0) =
    injection.block<3, 3>(0, 0) * initial_base.block<3, 3>(0, 0).eval();
  initial_base.block<3, 1>(0, 3) += injection.block<3, 1>(0, 3);
  filter.reset(initial_base * frames.front().odom_base.inverse());
  std::unique_ptr<RecoverySearch> search;
  std::vector<Eigen::Matrix4d> seeds;
  std::size_t searches = 0, proposals = 0, resets = 0, verified = 0;
  std::size_t rejected = 0, ambiguous = 0, confirmations = 0, no_advantage = 0;
  double last_search_stamp = -std::numeric_limits<double>::infinity();
  output << "stamp,scan_id,state,reason,map_prepare_ms,load_ms,search_ms,confirmation_ms,"
         << "runtime_ms,ranked_seeds,total_seeds,candidates,fitness,rmse,information_ratio,"
         << "incumbent_score,candidate_score,translation_error_m,rotation_error_deg,"
         << "reset,reset_translation_error_m,reset_rotation_error_deg,covariance_trace,"
         << "consistent_windows,verified_windows,searches,proposals,confirmations,resets,"
         << "verified_recoveries,rejected_windows,ambiguous_windows,no_advantage_windows\n";
  for (std::size_t index = 0; index < frames.size(); ++index) {
    const Frame & frame = frames[index];
    const auto begin = Clock::now();
    const Cloud scan = read_cloud(frame.scan_path);
    const double load_ms = milliseconds(begin);
    auto result = coordinator.tick(frame.stamp);
    double search_ms = 0.0;
    bool did_reset = false;
    const double missing = std::numeric_limits<double>::quiet_NaN();
    double reset_translation = missing, reset_rotation = missing;
    // Offline scheduling: finish a bounded-work search on a snapshot, then
    // confirm its hypotheses on subsequent manifest rows. Retry after 10 s if
    // the coordinator is idle. Recorded sensor time never advances with CPU time.
    if (!search || (coordinator.state() == recovery::State::IDLE &&
      frame.stamp - last_search_stamp >= 10.0))
    {
      const auto search_begin = Clock::now();
      search = std::make_unique<RecoverySearch>(
        shared_map, scan, frame.odom_base, filter.pose(), search_config);
      while (!search->done()) {
        search->tick(std::chrono::milliseconds(50), 64);
      }
      seeds.clear();
      for (const auto & candidate : search->candidates()) {
        seeds.push_back(candidate.map_odom);
      }
      last_search_stamp = frame.stamp;
      ++searches;
      search_ms = milliseconds(search_begin);
    }
    const auto confirmation_begin = Clock::now();
    const auto incumbent = search->evaluateFresh(scan, frame.odom_base, filter.pose());
    RecoveryCandidate best = incumbent;
    bool unambiguous = false;
    if (coordinator.state() == recovery::State::VERIFYING) {
      bool incumbent_unambiguous = true;
      const auto alternatives = refine_seeds(*search, scan, frame.odom_base, seeds);
      for (const auto & alternative : alternatives) {
        if (distinct(incumbent.map_odom, alternative.map_odom, frame.odom_base) &&
          !has_advantage(incumbent.score, alternative.score))
        {
          incumbent_unambiguous = false;
        }
      }
      result = coordinator.observeVerification(
        observation(incumbent, frame, index, coordinator.generation(), incumbent_unambiguous),
        frame.stamp);
      if (!result.candidate_recorded) {
        ++rejected;
      }
      if (!incumbent_unambiguous) {
        ++ambiguous;
      }
      if (result.verified) {
        ++verified;
      }
    } else if (coordinator.state() != recovery::State::VERIFIED) {
      auto fresh = refine_seeds(*search, scan, frame.odom_base, seeds);
      std::stable_sort(
        fresh.begin(), fresh.end(), [](const auto & a, const auto & b) {
          return a.score < b.score;
        });
      if (!fresh.empty()) {
        best = fresh.front();
        unambiguous = has_advantage(best.score, incumbent.score);
        if (best.accepted && !unambiguous) {
          ++no_advantage;
        }
        bool competing_alias = false;
        for (std::size_t i = 1; i < fresh.size(); ++i) {
          if (distinct(best.map_odom, fresh[i].map_odom, frame.odom_base) &&
            !has_advantage(best.score, fresh[i].score))
          {
            unambiguous = false;
            competing_alias = true;
          }
        }
        if (competing_alias) {
          ++ambiguous;
        }
      }
      result = coordinator.observe(
        observation(best, frame, index, coordinator.generation(), unambiguous),
        filter.pose(), frame.stamp);
      if (result.candidate_recorded) {
        ++confirmations;
      } else {
        ++rejected;
      }
      if (result.proposal_available) {
        ++proposals;
      }
      if (result.should_apply) {
        const auto requested = search->evaluateFresh(
          scan, frame.odom_base, result.requested_pose);
        bool requested_advantage = has_advantage(requested.score, incumbent.score);
        for (const auto & competitor : fresh) {
          if (distinct(requested.map_odom, competitor.map_odom, frame.odom_base) &&
            !has_advantage(requested.score, competitor.score))
          {
            requested_advantage = false;
          }
        }
        if (requested.accepted && requested_advantage) {
          filter.reset(result.requested_pose, filter_covariance(requested, frame.odom_base));
          if (!coordinator.acknowledgeApplied(result.generation, filter.pose())) {
            throw std::runtime_error("coordinator rejected its applied reset");
          }
          ++resets;
          did_reset = true;
          const Eigen::Matrix4d applied_base = filter.pose() * frame.odom_base;
          reset_translation = translation_error(applied_base, frame.reference_map_base);
          reset_rotation = rotation_error(applied_base, frame.reference_map_base);
        } else {
          ++rejected;
          coordinator.invalidate();
          result.reason = "requested_pose_revalidation_failed";
        }
      }
    }
    const double confirmation_ms = milliseconds(confirmation_begin);
    const Eigen::Matrix4d current_base = filter.pose() * frame.odom_base;
    output << frame.stamp << ',' << index + 1 << ',' << state_name(coordinator.state())
           << ',' << result.reason << ',' << (index == 0 ? map_prepare_ms : 0.0)
           << ',' << load_ms << ',' << search_ms << ',' << confirmation_ms << ','
           << milliseconds(begin) << ',' << search->rankedSeedCount() << ','
           << search->totalSeedCount() << ',' << seeds.size() << ',' << best.fitness
           << ',' << best.rmse << ',' << best.information_ratio << ',' << incumbent.score
           << ',' << best.score << ',' << translation_error(current_base, frame.reference_map_base)
           << ',' << rotation_error(current_base, frame.reference_map_base) << ',' << did_reset
           << ',' << reset_translation << ',' << reset_rotation << ',' <<
      filter.covariance().trace()
           << ',' << result.consistent_windows << ',' << result.verified_windows << ',' << searches
           << ',' << proposals << ',' << confirmations << ',' << resets << ',' << verified
           << ',' << rejected << ',' << ambiguous << ',' << no_advantage << '\n';
  }
  std::cerr << "frames=" << frames.size() << " searches=" << searches
            << " proposals=" << proposals << " resets=" << resets
            << " verified=" << verified << " rejected_windows=" << rejected
            << " ambiguous_windows=" << ambiguous
            << " no_advantage_windows=" << no_advantage << '\n';
}
}  // namespace

int main(int argc, char ** argv)
{
  const auto usage = []() {
      std::cerr << "Usage: relocalization_replay MAP.pcd MANIFEST.csv OUTPUT.csv "
                << "[--inject DX DY DZ YAW_DEGREES]\n"
                << "Scans must be expressed in odom. Manifest has 16 columns:\n"
                << "stamp,scan_path,odom_x,odom_y,odom_z,odom_qx,odom_qy,odom_qz,odom_qw,"
                << "reference_x,reference_y,reference_z,reference_qx,reference_qy,"
                << "reference_qz,reference_qw\n"
                << "First reference initializes map<-odom; subsequent references only score.\n"
                << "Injection perturbs initial map<-base about its position in map axes; "
                << "odom poses and clouds are unchanged.\n";
    };
  if (argc == 2 && std::string(argv[1]) == "--help") {
    usage();
    return 0;
  }
  if (argc != 4 && argc != 9) {
    usage();
    return 2;
  }
  try {
    Eigen::Matrix4d injection = Eigen::Matrix4d::Identity();
    if (argc == 9) {
      if (std::string(argv[4]) != "--inject") {
        throw std::runtime_error("expected --inject before four injection values");
      }
      injection.block<3, 1>(0, 3) =
        Eigen::Vector3d(number(argv[5]), number(argv[6]), number(argv[7]));
      injection.block<3, 3>(0, 0) = Eigen::AngleAxisd(
        std::remainder(number(argv[8]), 360.0) * kPi / 180.0,
        Eigen::Vector3d::UnitZ()).toRotationMatrix();
    }
    // Open3D messages must not be mixed into the machine-readable report.
    open3d::utility::SetVerbosityLevel(open3d::utility::VerbosityLevel::Error);
    const auto frames = read_manifest(argv[2]);
    const Cloud map = read_cloud(argv[1]);
    const std::filesystem::path output_path(argv[3]);
    if (std::filesystem::exists(output_path)) {
      throw std::runtime_error("output already exists; choose a new report path");
    }
    std::ofstream output(output_path);
    if (!output) {
      throw std::runtime_error("cannot create report: " + output_path.string());
    }
    output << std::setprecision(17);
    replay(map, frames, injection, output);
    output.flush();
    if (!output) {
      throw std::runtime_error("failed writing report: " + output_path.string());
    }
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "relocalization_replay: " << error.what() << '\n';
    return 1;
  }
}
