from setuptools import find_packages, setup

package_name = 'delta_weed_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/weed_bridge.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='s4mb4th',
    maintainer_email='s4mb4th@todo.todo',
    description='Delta weed detection to pick-and-place bridge',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'weed_bridge_node = delta_weed_bridge.weed_bridge_node:main',
        ],
    },
)
