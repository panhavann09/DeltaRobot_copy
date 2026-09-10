"""RealSense variant of weed_pick_place.launch.py: official realsense2_camera_node
driver + plant_perception_node.py (the ROS-topic-subscribing variant, not
merge_code_test.py's direct-capture one) + weed_bridge_node (/delta/target_xyz
etc.) + pick_place (the actual pick/place FSM driving the motors).

Unlike the UVC global-shutter path, this gets real aligned depth from the
camera (perception.yaml has use_depth: true) instead of relying on
FAKE_DEPTH_M — plant_perception_node.py's own MAD-filtered root-depth
estimate becomes real Z instead of a flat assumed distance.

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
    color_profile = LaunchConfiguration("color_profile")
    depth_profile = LaunchConfiguration("depth_profile")
    align_depth_enable = LaunchConfiguration("align_depth_enable")
    enable_accel = LaunchConfiguration("enable_accel")
    enable_gyro = LaunchConfiguration("enable_gyro")
    unite_imu_method = LaunchConfiguration("unite_imu_method")

    realsense = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        namespace="camera",
        parameters=[{
            "align_depth.enable": align_depth_enable,
            "rgb_camera.color_profile": color_profile,
            "depth_module.depth_profile": depth_profile,
            "enable_accel": enable_accel,
            "enable_gyro": enable_gyro,
            "unite_imu_method": unite_imu_method,
        }],
        output="screen",
    )

    perception_params_path = os.path.join(
        get_package_share_directory("plant_perception"), "config", "perception.yaml"
    )

    plant_perception_node = Node(
        package="plant_perception",
        executable="plant_perception_node.py",
        name="plant_perception_node",
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
        DeclareLaunchArgument("align_depth_enable", default_value="true"),
        # 2026-09-08: raised to 1920x1080 to feed tiled inference (see
        # plant_perception/tiled_inference.py + perception.yaml's
        # tile_size/tile_overlap). depth_profile intentionally NOT matched to
        # 1920x1080 — the D455's depth sensor doesn't natively run there, and
        # align_depth.enable reprojects into color space regardless of the
        # depth stream's native resolution. VERIFY this color_profile is
        # actually supported by the attached camera before relying on it
        # (`rs-enumerate-devices -c` on-device).
        DeclareLaunchArgument("color_profile", default_value="1920x1080x15"),
        DeclareLaunchArgument("depth_profile", default_value="640x480x15"),
        DeclareLaunchArgument("enable_accel", default_value="false"),
        DeclareLaunchArgument("enable_gyro", default_value="false"),
        DeclareLaunchArgument("unite_imu_method", default_value="0"),
        DeclareLaunchArgument("perception_params_file", default_value=perception_params_path),
        DeclareLaunchArgument(
            "enable_viewer", default_value="false",
            description="Launch rqt_image_view too (redundant now — weed_bridge_node's own \"Delta Camera\" window already shows plant_perception's frame plus workspace zones/EE marker)",
        ),
        realsense,
        plant_perception_node,
        weed_bridge_node,
        pick_place,
        viewer,
    ])
