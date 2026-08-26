from pathlib import Path


LAUNCH_FILE = (
    Path(__file__).parents[1] / "launch" / "localization_3d_e1r.launch.py"
)


def test_e1r_localization_is_the_odom_to_base_link_tf_authority():
    launch_source = LAUNCH_FILE.read_text(encoding="utf-8")

    assert "'publish_robot_root_tf': 'true'" in launch_source
    assert "'publish_output_tf': 'false'" in launch_source


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
