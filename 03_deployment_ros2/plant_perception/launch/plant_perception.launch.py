import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("plant_perception")
    params_file = os.path.join(pkg_dir, "config", "params.yaml")

    return LaunchDescription([
        Node(
            package="plant_perception",
            executable="plant_perception_node.py",
            name="plant_perception_node",
            output="screen",
            parameters=[params_file],
        )
    ])
