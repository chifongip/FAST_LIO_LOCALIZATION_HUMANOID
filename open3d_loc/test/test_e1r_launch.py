from pathlib import Path

import yaml


LAUNCH_FILE = (
    Path(__file__).parents[1] / "launch" / "localization_3d_e1r.launch.py"
)
OPEN3D_LAUNCH_FILE = LAUNCH_FILE.parent / "open3d_loc_g1.launch.py"
CONFIG_FILE = Path(__file__).parents[1] / "config" / "loc_param_g1.yaml"


def test_e1r_localization_is_the_odom_to_base_link_tf_authority():
    launch_source = LAUNCH_FILE.read_text(encoding="utf-8")

    assert "'publish_robot_root_tf': 'true'" in launch_source
    assert "'publish_output_tf': 'false'" in launch_source


def test_global_correction_fusion_matches_current_axis_policy():
    configuration = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    parameters = configuration["global_localization_node"]["ros__parameters"]
    fusion = parameters["fusion"]

    assert fusion["enabled"] is True
    assert fusion["update_mask"] == [True, True, True, True, True, True]
    assert len(fusion["initial_stddev"]) == 6
    assert len(fusion["measurement_stddev_floor"]) == 6
    assert fusion["recovery"]["enabled"] is True
    assert fusion["recovery"]["required_consistent_measurements"] == 2
    assert fusion["recovery"]["minimum_candidate_interval"] == 0.75
    assert fusion["recovery"]["max_odometry_gap"] == 0.15
    launch_source = OPEN3D_LAUNCH_FILE.read_text(encoding="utf-8")
    assert "'fusion.enabled':" not in launch_source
    assert "'fusion.recovery.enabled':" not in launch_source


def test_recovery_overlay_is_forwarded_and_loaded_after_base():
    e1r_source = LAUNCH_FILE.read_text(encoding="utf-8")
    launch_source = OPEN3D_LAUNCH_FILE.read_text(encoding="utf-8")
    assert "'recovery_config': LaunchConfiguration('recovery_config')" in e1r_source
    assert "config_file,\n            LaunchConfiguration('recovery_config')," in launch_source
    overlay = yaml.safe_load(
        (CONFIG_FILE.parent / "recovery_param.yaml").read_text(encoding="utf-8")
    )["global_localization_node"]["ros__parameters"]["fusion"]
    assert overlay["enabled"] is True
    recovery = overlay["recovery"]
    assert recovery["application_mode"] == "reset"
    assert recovery["required_consistent_measurements"] == 3
    assert recovery["verification_observations"] == 3
    assert recovery["search_radius"] == 6.0
    assert recovery["vertical_range"] == 1.0
    assert recovery["translation_step"] == 2.0
    assert recovery["yaw_step_degrees"] == 30.0
    assert recovery["search_interval"] == 10.0
    assert recovery["failure_trigger"] == 3
    assert recovery["search_budget_ms"] == 500.0
    assert recovery["timeout"] == 20.0
    assert recovery["max_refined"] == 16
    assert recovery["max_candidates"] == 4
    assert recovery["score_advantage"] == 0.15
    assert recovery["consistency_stddev_translation"] == 0.15
    assert recovery["consistency_stddev_rotation"] == 0.05
    assert recovery["consistency_mahalanobis_threshold"] == 16.812
    assert recovery["observability_ratio"] == 1.0e-4


def test_global_localization_uses_one_second_cadence():
    launch_source = OPEN3D_LAUNCH_FILE.read_text(encoding="utf-8")

    assert "'loc_frequence': 1.0" in launch_source


def test_fusion_prediction_timestamp_uses_ros_time_and_is_guarded():
    source_file = Path(__file__).parents[1] / "src" / "global_localization.cpp"
    source = source_file.read_text(encoding="utf-8")

    assert "last_fusion_prediction_stamp_{0, 0, RCL_ROS_TIME}" in source
    assert "last_fusion_prediction_stamp_.nanoseconds() != 0" in source
    assert "odom_stamp.get_clock_type() ==" in source


def test_open3d_verbosity_is_launch_configurable():
    launch_source = (
        LAUNCH_FILE.parent / "open3d_loc_g1.launch.py"
    ).read_text(encoding="utf-8")

    assert "'open3d_verbosity', default_value='warning'" in launch_source
    assert "'open3d_verbosity': LaunchConfiguration('open3d_verbosity')" in launch_source


def test_global_localization_ros_logger_is_error_only():
    launch_source = (
        LAUNCH_FILE.parent / "open3d_loc_g1.launch.py"
    ).read_text(encoding="utf-8")

    assert "'--log-level', 'global_localization_node:=error'" in launch_source
