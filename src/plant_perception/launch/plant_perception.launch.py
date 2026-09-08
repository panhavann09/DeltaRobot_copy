import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """Generate launch description for the plant perception and tracking pipeline."""
    # Find package share directory
    pkg_share = get_package_share_directory('plant_perception')
    
    # Path to default YAML parameters configuration
    default_params_path = os.path.join(pkg_share, 'config', 'perception.yaml')
    
    # Launch arguments
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_path,
        description='Absolute path to the YAML configuration parameters file'
    )
    
    # Node definition
    plant_perception_node = Node(
        package='plant_perception',
        executable='plant_perception_node.py',
        name='plant_perception_node',
        output='screen',
        parameters=[LaunchConfiguration('params_file')]
    )
    
    # Construct LaunchDescription
    ld = LaunchDescription()
    ld.add_action(params_file_arg)
    ld.add_action(plant_perception_node)
    
    return ld
