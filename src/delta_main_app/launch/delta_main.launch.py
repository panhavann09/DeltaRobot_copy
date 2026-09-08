# CHANGES: [global shutter] replaced realsense2_camera_node with uvc_camera_publisher (no depth)
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    camera_device = LaunchConfiguration("camera_device")
    camera_width = LaunchConfiguration("camera_width")
    camera_height = LaunchConfiguration("camera_height")
    camera_fps = LaunchConfiguration("camera_fps")
    enable_rviz = LaunchConfiguration("enable_rviz")

    pkg_share = get_package_share_directory("delta_description")
    xacro_path = os.path.join(pkg_share, "urdf", "delta_robot.urdf.xacro")
    rviz_config = os.path.join(pkg_share, "rviz", "delta_description.rviz")
    robot_description = ParameterValue(Command(["xacro ", xacro_path]), value_type=str)

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        condition=IfCondition(enable_rviz),
        parameters=[{"robot_description": robot_description}],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(enable_rviz),
        arguments=["-d", rviz_config],
    )

    uvc_camera = Node(
        package="delta_camera_system",
        executable="uvc_camera_publisher",
        namespace="camera",
        parameters=[{
            "camera_device": camera_device,
            "camera_width": camera_width,
            "camera_height": camera_height,
            "camera_fps": camera_fps,
        }],
        output="screen",
    )

    camera_node = Node(
        package="delta_camera_system",
        executable="camera_node",
        output="screen",
    )

    pick_place = Node(
        package="delta_main_app",
        executable="pick_place",
        output="screen",
    )

    test1 = Node(
        package="delta_main_app",
        executable="test1",
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument("camera_device", default_value="/dev/video0"),
        DeclareLaunchArgument("camera_width", default_value="640"),
        DeclareLaunchArgument("camera_height", default_value="480"),
        DeclareLaunchArgument("camera_fps", default_value="30"),
        DeclareLaunchArgument("enable_rviz", default_value="false"),
        robot_state_publisher,
        rviz,
        uvc_camera,
        camera_node,
        pick_place,
        test1,
    ])
