from pathlib import Path


LAUNCH_FILE = (
    Path(__file__).parents[1] / "launch" / "localization_3d_e1r.launch.py"
)


def test_e1r_localization_is_the_odom_to_base_link_tf_authority():
    launch_source = LAUNCH_FILE.read_text(encoding="utf-8")

    assert "'publish_robot_root_tf': 'true'" in launch_source
    assert "'publish_output_tf': 'false'" in launch_source
