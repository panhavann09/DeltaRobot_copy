cmake_minimum_required(VERSION 3.8)
project(plant_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrackedPlant.msg"
  "msg/TrackedPlantArray.msg"
  DEPENDENCIES std_msgs sensor_msgs
)

# Install Python node and module scripts
install(
  PROGRAMS
  plant_perception_node.py
  merge_code_test.py
  model_test_node.py
  trt_inferencer.py
  postprocess.py
  tiled_inference.py
  tracker.py
  track_state.py
  depth_utils.py
  visualization.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch, config, and model weight files
install(
  DIRECTORY launch config ../weight
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
