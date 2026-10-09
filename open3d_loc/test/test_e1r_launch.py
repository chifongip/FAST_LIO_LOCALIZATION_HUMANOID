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
    assert "'fusion.enabled': True" in OPEN3D_LAUNCH_FILE.read_text(encoding="utf-8")
    assert "'fusion.recovery.enabled': True" in OPEN3D_LAUNCH_FILE.read_text(encoding="utf-8")


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


def test_flat_floor_height_bounds_configuration():
    configuration = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    parameters = configuration["global_localization_node"]["ros__parameters"]
    assert parameters["height_bounds"] == {
        "enabled": True, "floor_z": 0.0, "min_height": 0.3, "max_height": 0.7,
    }
    assert parameters["fusion"]["enabled"] is True
    assert all(parameters["fusion"]["update_mask"])


def _generic_launch():
    import importlib.util
    spec = importlib.util.spec_from_file_location("generic_localization_launch", OPEN3D_LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recovery_yaml_values_are_not_overridden_by_unspecified_launch_arguments():
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument, OpaqueFunction
    from launch_ros.utilities import evaluate_parameters

    context = LaunchContext()
    description = _generic_launch().generate_launch_description()
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    opaque = next(a for a in description.entities if isinstance(a, OpaqueFunction))
    node = opaque.execute(context)[0]
    parameters = evaluate_parameters(context, node._Node__parameters)
    # Check actual launch parameters: the later dictionaries must not mask YAML.
    overrides = {k: v for entry in parameters if isinstance(entry, dict) for k, v in entry.items()}
    assert not any(k.startswith("runtime_recovery.") for k in overrides)
    yaml_parameters = yaml.safe_load(CONFIG_FILE.read_text())["global_localization_node"]["ros__parameters"]
    assert yaml_parameters["runtime_recovery"]["imu_topic"] == "/aima/hal/sensor/lidar_chest_front/imu"
    assert yaml_parameters["runtime_recovery"]["anchor_max_age"] == 10.0


def test_explicit_recovery_launch_arguments_have_typed_precedence():
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument, OpaqueFunction
    from launch_ros.utilities import evaluate_parameters

    context = LaunchContext()
    context.launch_configurations.update({
        "runtime_recovery_enabled": "true", "recovery_imu_topic": "/test/imu",
        "recovery_imu_time_offset_sec": "-0.25",
    })
    description = _generic_launch().generate_launch_description()
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    opaque = next(a for a in description.entities if isinstance(a, OpaqueFunction))
    node = opaque.execute(context)[0]
    parameters = evaluate_parameters(context, node._Node__parameters)
    assert parameters[-1] == {
        "runtime_recovery.enabled": True, "runtime_recovery.imu_topic": "/test/imu",
        "runtime_recovery.imu_time_offset_sec": -0.25,
    }


def test_invalid_recovery_enablement_is_rejected():
    import pytest
    from launch import LaunchContext

    context = LaunchContext()
    context.launch_configurations.update({
        "runtime_recovery_enabled": "yes", "recovery_imu_topic": "",
        "recovery_imu_time_offset_sec": "",
    })
    with pytest.raises(ValueError):
        _generic_launch().recovery_overrides(context)
