from setuptools import find_packages, setup

package_name = 'delta_main_app'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/delta_main.launch.py',
            'launch/blind_pick_place.launch.py',
            'launch/matlab_bridge.launch.py',
            'launch/weed_pick_place.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='s4mb4th',
    maintainer_email='s4mb4th@todo.todo',
    description='Delta robot main integration app',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'main_app             = delta_main_app.main_app:main',
            'blind_pick_place     = delta_main_app.blind_pick_place:main',
            'pick_place_ui        = delta_main_app.pick_place_ui:main',
            'laser_accuracy_experiment = delta_main_app.laser_accuracy_experiment:main',
            'analyze_repeatability     = delta_main_app.analyze_repeatability:main',
            'laser_preview             = delta_main_app.laser_preview:main',
            'repeatability_test        = delta_main_app.repeatability_test:main',
            'workspace_accuracy_sweep  = delta_main_app.workspace_accuracy_sweep:main',
            'matlab_bridge            = delta_main_app.matlab_bridge_node:main',
            'test1                    = delta_main_app.test1:main',
            'check_cube_detection_accuracy = delta_main_app.check_cube_detection_accuracy:main',
            'camera_detection_monitor      = delta_main_app.camera_detection_monitor:main',
            'dynamic_tracking_log          = delta_main_app.dynamic_tracking_log:main',
        ],
    },
)