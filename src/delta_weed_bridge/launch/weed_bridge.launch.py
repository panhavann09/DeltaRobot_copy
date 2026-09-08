"""Launches plant_perception's direct-camera pipeline (merge_code_test.py
— opens the UVC camera device itself, no separate camera-driver node) +
weed_bridge_node (publishes /delta/target_xyz etc.).

Do not run this alongside delta_camera_system's camera_system.launch.py or
delta_main_app's weed_pick_place.launch.py — only one process can hold the
camera device open at a time, and both would publish the same /delta/target_xyz
topics.
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
        viewer,
    ])
