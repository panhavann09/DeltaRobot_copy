# CHANGES: [global shutter] replaced realsense2_camera_node with uvc_camera_publisher (no depth)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera_device = LaunchConfiguration("camera_device")
    camera_width = LaunchConfiguration("camera_width")
    camera_height = LaunchConfiguration("camera_height")
    camera_fps = LaunchConfiguration("camera_fps")

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

    return LaunchDescription([
        DeclareLaunchArgument("camera_device", default_value="/dev/video0"),
        DeclareLaunchArgument("camera_width", default_value="640"),
        DeclareLaunchArgument("camera_height", default_value="480"),
        DeclareLaunchArgument("camera_fps", default_value="30"),
        uvc_camera,
        camera_node,
    ])
