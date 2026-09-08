# Delta Robot — ROS 2 Workspace

A ROS 2 (Humble) workspace for a **3-axis delta robot** with pneumatic gripper, vision-guided and blind pick-and-place capability, and a CAN-bus hardware abstraction layer.

---

## Table of Contents

- [Hardware Overview](#hardware-overview)
- [Package Structure](#package-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modes of Operation](#modes-of-operation)
- [CAN Bus Setup](#can-bus-setup)
- [Configuration](#configuration)
- [Topics & Services Reference](#topics--services-reference)
- [Developer Guide](#developer-guide)

---

## Hardware Overview

| Component | Details |
|---|---|
| Arm | 3-axis parallel delta robot |
| Motors | 3× RobStride RS-00 (CAN bus, IDs 1–3) |
| Gripper | Pneumatic solenoid valve (CAN ID 4, solenoid1) |
| Camera | Intel RealSense D455 (color + aligned depth) |
| CAN Interface | `can0` at 1 Mbit/s (SocketCAN) |

**Kinematic parameters** (see `delta_common/config.py`):

| Symbol | Value | Description |
|---|---|---|
| `e` | 35 mm | End-effector triangle side |
| `f` | 157 mm | Base triangle side |
| `re` | 400 mm | Forearm length |
| `rf` | 200 mm | Upper arm length |

**Workspace limits:**

| Axis | Min | Max |
|---|---|---|
| X / Y | −151.6 mm | +151.6 mm |
| Z | −500 mm | −196.9 mm |

---

## Package Structure

```
delta_ws/
├── src/
│   ├── can_driver/               # CAN bus driver + custom ROS messages
│   │   ├── can_driver/           # Python package
│   │   │   ├── can_driver.py     # SocketCAN driver node (brings up can0)
│   │   │   └── can_driver_smarttek.py
│   │   └── custom_messages/      # ROS 2 message definitions (ament_cmake)
│   │
│   ├── delta_common/             # Shared library (no ROS node)
│   │   └── delta_common/
│   │       ├── config.py         # All tunable parameters
│   │       ├── fk_ik.py          # Forward / Inverse kinematics
│   │       └── trajectory.py     # Trapezoidal trajectory planner
│   │
│   ├── delta_motor_controller/   # Motor + gripper control
│   │   └── delta_motor_controller/
│   │       ├── motor_controller.py   # DeltaMotorController class
│   │       ├── pneumatic_gripper.py  # PneumaticGripper (via can_driver)
│   │       └── valve_can_node.py     # Gamepad-driven solenoid node
│   │
│   ├── delta_camera_system/      # Intel RealSense + object detection
│   │   └── delta_camera_system/
│   │       ├── camera_system.py          # Main camera ROS node
│   │       ├── calibrate_transform.py    # Camera-to-robot calibration
│   │       └── calibrate_plane_homography.py
│   │
│   ├── delta_main_app/           # Top-level application nodes
│   │   ├── delta_main_app/
│   │   │   ├── main_app.py           # Vision-guided pick-and-place
│   │   │   └── blind_pick_place.py   # Pre-programmed pick-and-place
│   │   └── launch/
│   │       ├── delta_main.launch.py          # Camera + main_app
│   │       └── blind_pick_place.launch.py    # can_driver + blind_pick_place
│   │
│   ├── testing_can/              # Hardware test utilities
│   └── rob_and_ros_pkg/          # RobStride motor experiments
│
└── README.md
```

---

## Prerequisites

- **OS**: Ubuntu 22.04
- **ROS 2**: Humble Hawksbill (`ros-humble-desktop`)
- **Python**: 3.10+
- **CAN tools**: `can-utils`, `iproute2`
- **Python packages**: `python-can`, `numpy`, `opencv-python`, `ultralytics` (YOLO)
- **RealSense SDK**: `librealsense2` + `ros-humble-realsense2-camera`

```bash
# System dependencies
sudo apt install can-utils python3-can ros-humble-realsense2-camera

# Python dependencies
pip install python-can numpy opencv-python ultralytics tqdm
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url> ~/delta_ws
cd ~/delta_ws

# 2. Source ROS 2
source /opt/ros/humble/setup.bash

# 3. Install ROS dependencies
rosdep install --from-paths src --ignore-src -r -y

# 4. Build
colcon build --symlink-install

# 5. Source the workspace
source install/setup.bash
```

---

## Quick Start

### Blind Pick-and-Place (no camera)

```bash
ros2 launch delta_main_app blind_pick_place.launch.py
```

This starts:
1. `can_driver_node` — brings up `can0`, owns all CAN communication
2. `blind_pick_place` node (after 3 s delay) — runs the FSM

Trigger a pick cycle:
```bash
# Via service
ros2 service call /delta/trigger_pick std_srvs/srv/Empty

# Via topic (custom pick position in mm)
ros2 topic pub /delta/blind_target geometry_msgs/msg/PointStamped \
  "{header: {frame_id: 'base'}, point: {x: 0.0, y: 50.0, z: -350.0}}"
```

Override pick/drop positions at launch:
```bash
ros2 launch delta_main_app blind_pick_place.launch.py \
  --ros-args -p pick_x:=10.0 -p pick_y:=60.0 -p pick_z:=-360.0 \
             -p drop_x:=0.0  -p drop_y:=-80.0 -p drop_z:=-300.0
```

### Vision-Guided Pick-and-Place

```bash
ros2 launch delta_main_app delta_main.launch.py
```

This starts the RealSense camera node + the vision-guided `main_app`.

---

## Modes of Operation

### 1. Blind Pick-and-Place (`blind_pick_place.py`)

No camera. The pick position is either set via ROS parameters or sent over the `/delta/blind_target` topic.

**FSM states:**

```
IDLE → APPROACHING → PICKING → GRASPING → LIFTING → TRANSPORTING → DROPPING → RESETTING → IDLE
```

| Parameter | Default | Description |
|---|---|---|
| `pick_x/y/z` | 0, 50, -350 mm | Pick position |
| `drop_x/y/z` | 0, -80, -300 mm | Drop position |
| `approach_z_offset` | 50 mm | Z above pick for safe approach |
| `lift_z_offset` | 70 mm | Z above pick after grasp |
| `traj_v_max_mm_s` | 80 mm/s | Transport speed |
| `pick_v_max_mm_s` | 40 mm/s | Pick/lift speed |
| `pneumatic_can_id` | 4 | CAN ID of solenoid board |

### 2. Vision-Guided Pick-and-Place (`main_app.py`)

RealSense camera detects objects (white rectangle or YOLO). Sends `PointStamped` to pick location automatically.

**Detection mode** is set in `config.py`:
- `"white_rectangle"` — HSV + contour detection for light-coloured boxes
- `"yolo"` — YOLO11 model inference

### 3. Gamepad-Controlled Gripper (`valve_can_node.py`)

Manual gripper control via gamepad (dc_gamepad_msgs):
- **Button A**: toggle gripper open/close
- **Button LB**: start dribbling/pick sequence

---

## CAN Bus Setup

The `can_driver_node` handles this automatically. If you need manual control:

```bash
# Bring up can0 at 1 Mbit/s
sudo ip link set can0 up type can bitrate 1000000

# Monitor live traffic
candump can0

# Take down
sudo ip link set can0 down
```

### CAN ID Map

| CAN ID Range | Direction | Content |
|---|---|---|
| 1–3 | TX | RobStride motor commands (extended ID, Robstride protocol) |
| 4 | TX | Solenoid/digital commands (`DigitalAndSolenoidCommand`) |
| 100–199 | RX | Encoder feedback (`EncoderFeedback`) |
| 500–510 | RX | Digital/analog sensor feedback (`DigitalAndAnalogFeedback`) |

> **Note:** RobStride motors use **extended CAN IDs** (29-bit). The `can_driver_node` only processes standard 11-bit frames for IDs 100–199 and 500–510 — it transparently ignores Robstride frames.

---

## Configuration

All tunable parameters live in **`src/delta_common/delta_common/config.py`**.

**Enabling/disabling hardware:**

```python
ENABLE_MOTORS = True   # False → dry-run (logs only, no CAN output)
AUTO_MOVE     = True   # False → vision detects but does not move
```

**Motor limits** (degrees):

```python
THETA1_MIN, THETA1_MAX = 0.0, 82.11
THETA2_MIN, THETA2_MAX = 0.0, 85.49
THETA3_MIN, THETA3_MAX = 0.0, 85.49
```

**Camera transform** (camera frame → robot base frame):

```python
CAMERA_DIRECT_MATRIX = ((1, 0, 0), (0, -1, 0), (0, 0, -1))
CAM_TX_MM, CAM_TY_MM, CAM_TZ_MM = 97.59, 11.93, 432.00
```

---

## Topics & Services Reference

### Published

| Topic | Type | Publisher | Description |
|---|---|---|---|
| `/delta/robot_state` | `std_msgs/String` | `main_app`, `blind_pick_place` | Current FSM state |
| `/encoder_feedback` | `EncoderFeedback` | `can_driver_node` | Motor encoder (pos, speed) |
| `/digital_analog_feedback` | `DigitalAndAnalogFeedback` | `can_driver_node` | Sensor inputs |

### Subscribed

| Topic | Type | Subscriber | Description |
|---|---|---|---|
| `/publish_motor` | `MotorCommand` | `can_driver_node` | Motor position/speed/voltage command |
| `/publish_servo` | `ServoCommand` | `can_driver_node` | Servo value (0–1 mapped to 14-bit) |
| `/publish_pwm` | `PwmCommand` | `can_driver_node` | PWM value (0–1 mapped to 14-bit) |
| `/publish_digital_solenoid` | `DigitalAndSolenoidCommand` | `can_driver_node` | Solenoid + digital outputs |
| `/delta/blind_target` | `PointStamped` | `blind_pick_place` | Pick position override (mm) |
| `/delta/target_xyz` | `PointStamped` | `main_app` | Vision-detected pick position |

### Services

| Service | Type | Server | Description |
|---|---|---|---|
| `/delta/trigger_pick` | `std_srvs/Empty` | `blind_pick_place` | Start pick at configured position |

---

## Developer Guide

See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for:
- Architecture deep-dive
- Adding new pick modes
- Calibrating the camera transform
- Writing new CAN message handlers
- Testing without hardware
# DeltaRobot_copy
