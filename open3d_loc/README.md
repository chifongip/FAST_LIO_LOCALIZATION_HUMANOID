# Open3D global-localization parameters

This guide covers the `global_localization_node` parameters used by the E1R
launch flow. The node combines continuous FAST-LIO odometry (`odom ->
base_link`) with ICP measurements of the global correction (`map -> odom`).
The `fusion` filter and its recovery mode only change `map -> odom`; they do
not change FAST-LIO's local odometry.

The configured values live in `config/loc_param_g1.yaml`. Parameters supplied
directly by `launch/open3d_loc_g1.launch.py` override a value with the same
name in that YAML file. Tune a copy of the YAML file or a launch-specific
parameter overlay, then validate it against a recorded bag before using it on
the robot.

## Units and axes

- Distances are metres.
- Angles in `initialpose` are degrees; every other angle parameter is radians.
- Time parameters are seconds unless their name ends in `_ms`.
- Six-element vectors use `[x, y, z, roll, pitch, yaw]`.
- Fitness is a unitless overlap score in `[0, 1]`; larger is better.

## Input, map, and scheduling parameters

| Parameter | Current launch/config value | Meaning and tuning effect |
| --- | --- | --- |
| `path_map` / launch argument `map_file` | launch argument | Path to the `.pcd`/`.ply` global map. The map must be in the `map` frame and use the same scale and orientation as the expected localization output. |
| `initialpose` | `[x, y, z, roll, pitch, yaw]` | Initial `map -> odom` pose. A sufficiently close initial pose is required because the normal ICP registration is local. Use RViz `/initialpose` to reset it at runtime. |
| `pcd_queue_maxsize` | `10` | Number of registered scans accumulated into one ICP source cloud. Larger values improve density but add latency and can blur moving motion; lower values reduce latency but may make ICP sparse. |
| `loc_frequence` | `1.0` | Minimum interval between ICP attempts. Despite the historical name, it is a period, not Hz. The node ignores already processed scan stamps. Do not set below the measured localization runtime. |
| `maxpoints_source` | `80000` | Maximum source-cloud points after crop/downsampling. Lower it to meet the cadence; raise it only when source geometry is too sparse. |
| `maxpoints_target` | `400000` | Maximum cropped map points. Lower it first if ICP overruns; raise it only for a sparse map with enough RAM/CPU. |
| `voxelsize_coarse` | `0.01` | Voxel size used when publishing/preparing the coarse global map. Larger values reduce memory and visualization cost. |
| `voxelsize_fine` | `0.2` | Voxel size of the ICP source and fine map. Larger values are faster and less sensitive to point density; smaller values preserve detail but increase cost and can overfit noise. |
| `threshold_fitness_init` | `0.5` | Fitness required by the initialization loop. Lowering it starts sooner but makes a bad initial correction more likely. |
| `threshold_fitness` | `0.5` | Fitness required by regular ICP before either normal fusion or recovery is considered. This is a quality gate, not a correction gain. |
| `dis_updatemap` | `3.5` | Map-frame motion before the cropped submap is refreshed. Lower it after large recovery steps or in dense, changing geometry; raise it to reduce crop work. Recovery already forces a refresh after a bounded step. |
| `save_scan` | `false` | Debug-only point-cloud saving. Do not enable on the robot without first updating the legacy hard-coded output path in the node; it can create large files and increase localization latency. |
| `open3d_verbosity` | `warning` | Open3D log level: `debug`, `info`, `warning`, or `error`. Use `info` temporarily while diagnosing registration, then return to `warning` or `error`. |
| `confidence_loc_th` | `0.7` | Legacy compatibility parameter. It is currently read but is not used as a fusion or ICP gate; use `threshold_fitness` instead. |
| `hidden_removal` | `false` launch override | Legacy launch value that is not consumed by `global_localization_node`; changing it has no current localization effect. |

## Frames, timing, and TF parameters

| Parameter | Current E1R value | Meaning and tuning effect |
| --- | --- | --- |
| `imu_frame` | `lidar_imu_chest_front` | Frame represented by the FAST-LIO pose. It must agree with the FAST-LIO configuration and a valid TF chain to `body_frame`. |
| `body_frame` | `base_link` | Robot body frame used for global localization and `map -> base_link`. |
| `output_frame` | `torso_link` | Frame carried by `/localization_3d`. It is derived from the current `body_frame -> output_frame` transform. |
| `publish_robot_root_tf` | `true` | Publishes the local odometry transform to the robot root. Enable only when this node is the intended authority for that transform. |
| `publish_output_tf` | `false` | Publishes direct `map -> output_frame` TF. Keep disabled if another component owns the same transform. |
| `tf_lookup_max_age_ms` | `100.0` | Maximum age accepted for a fallback TF lookup. Increase only for delayed replay data; a large value can apply stale articulation transforms. |
| `use_sim_time` | `false` | Use bag `/clock` time when `true`. It must match the setting used by FAST-LIO and the bag replay. |
| `sensor_time_offset_to_ros_sec` (E1R launch) | `0.0` | Offset added to E1R LiDAR/IMU header times. Use only to correct a known sensor-to-ROS clock offset. Incorrect values cause stale measurements and TF lookup failures. |
| `start_rviz` (E1R launch) | `true` | Starts RViz. It has no effect on localization calculations. |
| `publish_legacy_static_frames` (Open3D launch) | `false` for E1R | Publishes identity `imu_link -> base_link -> motion_link` transforms for legacy G1 setups. Keep it disabled on E1R, which has real robot TF. |

## Legacy scalar smoothing parameters

These apply only when `fusion.enabled: false`. When fusion is enabled it
overrides this path.

| Parameter | Meaning |
| --- | --- |
| `filter_odom2map` | Enables the legacy scalar smoothing path for `map -> odom` when fusion is disabled. |
| `kalman_processVar2` | Legacy scalar process variance. Larger values follow new measurements more quickly. |
| `kalman_estimatedMeasVar2` | Legacy scalar measurement variance. Larger values smooth more strongly. |
| `kf_baselink2map/x`, `/y`, `/z` | Two-element legacy Kalman settings for the map-frame body position. They are not used by the error-state global-correction filter. |

## Global-correction filter (`fusion`) parameters

Set `fusion.enabled: true` to use the six-dimensional error-state Kalman
filter. It fuses quality-valid ICP estimates of `map -> odom` rather than
replacing the correction directly.

| Parameter | Current value | Meaning and tuning effect |
| --- | --- | --- |
| `fusion.enabled` | `true` | Enables the error-state global-correction filter and recovery logic. |
| `fusion.update_mask` | `[true, true, true, true, true, true]` | Axes ICP is allowed to correct. The current configuration corrects all six axes. For a gravity-aligned map with reliable IMU attitude, `[true, true, true, false, false, true]` is often safer because it keeps roll/pitch with FAST-LIO. |
| `fusion.initial_stddev` | `[0.25, 0.25, 0.15, 0.05, 0.05, 0.15]` | Initial correction uncertainty. Larger values let the first valid ICP measurement move the filter more; smaller values make startup more conservative. |
| `fusion.process_stddev_time` | `[0.02, 0.02, 0.01, 0.005, 0.005, 0.01]` | Uncertainty growth per second. Increase an axis when its global correction legitimately changes over time; decrease it to resist jitter. |
| `fusion.process_stddev_distance` | `[0.01, 0.01, 0.005, 0.002, 0.002, 0.005]` | Additional uncertainty growth per metre of FAST-LIO motion. It also makes recovery confirmation more tolerant while the robot moves. |
| `fusion.process_stddev_rotation` | `[0.01, 0.01, 0.005, 0.005, 0.005, 0.01]` | Additional uncertainty growth per radian of FAST-LIO motion. Increase yaw terms only if turn-induced map-correction variation is genuine. |
| `fusion.measurement_stddev_floor` | `[0.03, 0.03, 0.03, 0.01, 0.01, 0.02]` | Minimum ICP measurement noise. Increase it to reduce jitter or prevent overconfident ICP from dominating; do not reduce it below demonstrated repeatability. |
| `fusion.shared_lidar_covariance_scale` | `4.0` | Multiplies ICP covariance to account for correlated LiDAR points. Increase it when corrections are too aggressive or noisy; reduce it cautiously when valid ICP is underweighted. |
| `fusion.max_measurement_age` | `1.0` | Maximum age of the scan used for ICP. A value below the actual compute/transport delay rejects all measurements; a large value applies stale global corrections. |
| `fusion.max_icp_rmse` | `0.3` | Maximum ICP inlier RMSE in metres. Lower it for precise, static maps; raise it only for coarser or noisier maps after checking fitness and correspondence counts. |
| `fusion.min_correspondences` | `100` | Minimum evaluated ICP correspondences. Raise it in repetitive or sparse maps to reduce false matches; lower it only when valid cropped scans contain fewer stable points. |
| `fusion.max_consecutive_rejections` | `5` | Rejection count at which diagnostics report an error/degraded global correction. It changes alerting, not acceptance. |
| `fusion.max_innovation_translation` | `1.0` | Maximum normal-filter translation innovation. Larger errors are routed to recovery, not accepted normally. Keep this conservative. |
| `fusion.max_innovation_rotation` | `0.35` | Maximum normal-filter rotation innovation (~20 degrees). Larger errors are routed to recovery when ICP quality is valid. |
| `fusion.mahalanobis_threshold` | `13.277` | Statistical normal-update gate. Increase only if the filter covariance/noise model is demonstrably too confident; otherwise it prevents overconfident ICP jumps. |

## Recovery parameters (`fusion.recovery`)

Recovery is used only when ICP passes fitness, RMSE, correspondence, and age
gates but fails a normal translation, rotation, or Mahalanobis gate. It does
not create a global ICP hypothesis; it validates and safely applies one that
the existing ICP has already found.

| Parameter | Current value | Meaning and tuning effect |
| --- | --- | --- |
| `enabled` | `true` | Enables confirmation and bounded recovery steps. |
| `required_consistent_measurements` | `2` | Number of mutually consistent ICP candidates required before the first recovery step. Keep at least `2` on hardware. |
| `minimum_candidate_interval` | `0.75` | Minimum time between candidates. It prevents the same scan or nearly identical registration from counting twice. It must be lower than `loc_frequence`. |
| `candidate_timeout` | `3.0` | Maximum gap before confirmation restarts. Increase modestly if ICP occasionally misses a cycle; a long timeout allows stale evidence to combine. |
| `consistency_mahalanobis_threshold` | `13.277` | Statistical agreement gate between consecutive candidates, using ICP covariance plus motion-dependent process noise. Increase only after confirming that covariance is too optimistic. |
| `max_candidate_translation_delta` | `0.50` | Maximum translation difference between successive candidates. It limits candidate-to-candidate disagreement, not the total correction magnitude. |
| `max_candidate_rotation_delta` | `0.20` | Maximum rotation difference between successive candidates (~11.5 degrees). It is also a candidate consistency limit, not an absolute recovery limit. |
| `max_step_translation` | `0.50` | Largest translation applied by each confirmed recovery cycle. Larger values recover faster but increase the impact of a false alignment. |
| `max_step_rotation` | `0.15` | Largest rotation applied by each confirmed recovery cycle (~8.6 degrees). A stable 180-degree yaw candidate needs about 21 steps at this limit. |

## Tuning workflow

Change one group at a time, replay a representative bag, and record the
diagnostics before changing another group. The diagnostics topic is
`/localization_3d_diagnostics`; its `reason`, `fusion_mode`,
`normal_rejection_reason`, recovery candidate fields, and localization overrun
count identify which stage is rejecting work.

1. Start with map and time correctness. Confirm map scale/orientation, TF
   frames, LiDAR/IMU timestamps, and `use_sim_time` before relaxing any gate.
   Incorrect timestamps commonly appear as `stale_measurement` or TF failures.
2. Establish normal ICP quality. Inspect `fitness`, `rmse`, and
   `correspondences`. If normal matches fail quality gates, adjust scan/map
   voxel sizes, point limits, crop refresh distance, or map quality first.
   Do not loosen recovery parameters for a poor ICP match.
3. Remove small-correction jitter by increasing
   `measurement_stddev_floor` or `shared_lidar_covariance_scale`. If jitter
   accumulates during motion, reduce the relevant process standard deviations.
4. Handle normal-filter over-rejection by comparing diagnostics. A
   `translation_gate`, `rotation_gate`, or `mahalanobis_gate` is expected for
   a genuine large correction: recovery should enter `confirming`. Do not
   simply raise normal innovation limits, which bypasses two-candidate
   confirmation.
5. Tune recovery confirmation only when candidates fail to reach
   `recovering`. If valid candidates fluctuate slightly, increase
   `candidate_timeout`, `max_candidate_translation_delta`, or
   `max_candidate_rotation_delta` in small increments. Keep
   `required_consistent_measurements: 2` and a positive candidate interval.
6. Tune recovery speed only after confirmation is reliable. If diagnostics
   show `recovery_step_applied` but drift closes too slowly, increase
   `max_step_translation` and `max_step_rotation` gradually. A cautious
   large-recovery starting point is `0.75 m` and `0.25 rad` per second.
7. Address 180-degree yaw errors at the ICP-hypothesis stage. Parameters can
   let recovery apply a stable 180-degree candidate gradually, but local ICP
   initialized at the wrong pose often cannot discover that candidate. If
   diagnostics never enter `confirming`, use multi-start yaw ICP or another
   global-hypothesis method; do not compensate by allowing an immediate
   pi-radian recovery step.
8. Decide the attitude policy deliberately. The current YAML permits ICP roll
   and pitch correction. If the map is level and IMU attitude is reliable,
   disable those axes to avoid tilting `map -> odom`. Keep them enabled only
   when recorded-bag testing shows that global map alignment corrects a real,
   repeatable roll/pitch bias.
9. Keep cadence sustainable. If `localization_overrun_count` rises, first
   lower `maxpoints_target`/`maxpoints_source` or increase `voxelsize_fine`.
   Then increase `loc_frequence` only if needed. A busy node must not process
   stale scans merely to preserve a nominal update rate.

### Conservative large-recovery example

Use this only after a bag replay shows repeatable, quality-valid large ICP
candidates. It speeds bounded recovery without accepting a large ICP jump in a
single normal-filter update:

```yaml
fusion:
  recovery:
    required_consistent_measurements: 2
    minimum_candidate_interval: 0.75
    candidate_timeout: 4.0
    max_candidate_translation_delta: 0.75
    max_candidate_rotation_delta: 0.30
    max_step_translation: 0.75
    max_step_rotation: 0.25
```

For a 180-degree yaw correction, this still requires a consistent ICP result
and several bounded steps. It is intentionally not an immediate flip.

## Topic reference

| Topic | Purpose |
| --- | --- |
| `/odom2map` and TF `map -> odom` | Authoritative filtered global correction. Do not differentiate it to estimate velocity. |
| `/odom2map_icp` | Raw ICP `map -> odom` candidate and covariance. |
| `/localization_3d_odom` | Fused map-frame output pose with covariance. |
| `/localization_3d_diagnostics` | Fusion/recovery state, gates, candidate consistency, applied steps, and overrun count. |
| `/localization_3d_confidence` | Published registration fitness. |
