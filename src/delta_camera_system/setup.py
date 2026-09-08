from setuptools import find_packages, setup

package_name = 'delta_camera_system'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/launch', ['launch/camera_system.launch.py']),
    ('share/' + package_name + '/examples', ['examples/camera_base_pairs.example.json']),
    ('share/' + package_name + '/examples', ['examples/plane_homography_pairs.example.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='s4mb4th',
    maintainer_email='s4mb4th@todo.todo',
    description='Delta camera system',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'camera_node = delta_camera_system.camera_system:main',
            'uvc_camera_publisher = delta_camera_system.uvc_camera_publisher:main',
            'calibrate_camera_transform = delta_camera_system.calibrate_transform:main',
            'calibrate_plane_homography = delta_camera_system.calibrate_plane_homography:main',
            'calibrate_camera_intrinsics = delta_camera_system.calibrate_intrinsics:main',
        ],
    },
)
