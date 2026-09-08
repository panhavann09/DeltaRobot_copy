"""
Launch robot_state_publisher + joint_state_bridge + RViz2 for the delta robot.

Drive the arms with, e.g.:
    ros2 param set /joint_state_bridge theta1_deg 30.0
or with sliders via:
    ros2 run rqt_reconfigure rqt_reconfigure

Also starts a standalone uvc_camera_publisher (namespace "camera") for the
RViz Image display. This is separate from
delta_camera_system/launch/camera_system.launch.py (the production perception
pipeline) — don't run both at once, only one process can hold the camera
device open. The global-shutter camera has no depth stream, so there is no
PointCloud2 display anymore.
Disable with `enable_camera:=false` if you just want the arm model.

Pass `drive_real_motors:=true` to also mirror theta1/2/3_deg onto the real
robot over CAN (joint_state_bridge then doubles as a standalone manual jog
tool). Do NOT combine with delta_main_app/pick_place_node running at the
same time — they'd both fight over can1.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('delta_description')
    xacro_path = os.path.join(pkg_share, 'urdf', 'delta_robot.urdf.xacro')
    rviz_config = os.path.join(pkg_share, 'rviz', 'delta_description.rviz')

    robot_description = ParameterValue(Command(['xacro ', xacro_path]), value_type=str)
    enable_camera = LaunchConfiguration('enable_camera')
    drive_real_motors = LaunchConfiguration('drive_real_motors')

    return LaunchDescription([
        DeclareLaunchArgument('enable_camera', default_value='true'),
        DeclareLaunchArgument('drive_real_motors', default_value='false'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='delta_description',
            executable='joint_state_bridge',
            name='joint_state_bridge',
            output='screen',
            parameters=[{'drive_real_motors': drive_real_motors}],
        ),
        Node(
            package='delta_camera_system',
            executable='uvc_camera_publisher',
            namespace='camera',
            output='screen',
            condition=IfCondition(enable_camera),
            parameters=[{
                'camera_device': '/dev/video0',
                'camera_width': 640,
                'camera_height': 480,
                'camera_fps': 30,
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
        ),
    ])
