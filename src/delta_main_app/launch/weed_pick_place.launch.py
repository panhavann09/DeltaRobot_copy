"""One-shot launch for weed picking: plant_perception's direct-camera
pipeline (merge_code_test.py — opens the UVC global-shutter camera device
itself, no separate camera-driver node) + weed_bridge_node (/delta/target_xyz
etc.) + pick_place (the actual pick/place FSM driving the motors).

ENABLE_MOTORS is True in delta_common/config.py by default — pick_place
will command the real motors/gripper the moment a valid weed target arrives.
Make sure the workspace is clear and you're watching the arm before running
this.

weed_bridge_node's own "Delta Camera" window shows everything — plant_perception's
per-object weed boxes/track_id/roots/FPS plus the workspace/conveyor zones and
EE marker — so rqt_image_view is off by default (enable_viewer:=true to also
launch it).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    perception_params_path = os.path.join(
        get_package_share_directory("plant_perception"), "config", "perception_direct.yaml"
    )

    plant_perception_node = Node(
        package="plant_perception",
        executable="merge_code_test.py",
        name="merge_code_test_node",
        output="screen",
        parameters=[LaunchConfiguration("perception_params_file")],
    )

    weed_bridge_node = Node(
        package="delta_weed_bridge",
        executable="weed_bridge_node",
        output="screen",
    )

    pick_place = Node(
        package="delta_main_app",
        executable="pick_place",
        output="screen",
    )

    viewer = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="plant_perception_viewer",
        arguments=["/plant_perception/visualization"],
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_viewer")),
    )

    return LaunchDescription([
        DeclareLaunchArgument("perception_params_file", default_value=perception_params_path),
        DeclareLaunchArgument(
            "enable_viewer", default_value="false",
            description="Launch rqt_image_view too (redundant now — weed_bridge_node's own \"Delta Camera\" window already shows plant_perception's frame plus workspace zones/EE marker)",
        ),
        plant_perception_node,
        weed_bridge_node,
        pick_place,
        viewer,
    ])
