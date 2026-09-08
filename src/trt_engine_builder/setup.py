from setuptools import find_packages, setup

package_name = 'trt_engine_builder'

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
    description='trtexec wrapper for exporting device-local TensorRT engines',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'build_trt_engine = trt_engine_builder.build_engine:main',
        ],
    },
)
