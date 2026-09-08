from setuptools import find_packages, setup

package_name = 'rob_and_ros_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='s4mb4th',
    maintainer_email='s4mb4th@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': ['rob_n_ros_node = rob_and_ros_pkg.rob_n_ros:main',
                            'motor_pub_node = rob_and_ros_pkg.motor_pub_node:main',
                            'simple_pub_node = rob_and_ros_pkg.simple_pub_node:main',
                            'simple_sub_node = rob_and_ros_pkg.simple_sub_node:main',
                            'mit_motor_node = rob_and_ros_pkg.mit_motor_node:main',
        ],
    },
)
