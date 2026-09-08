#!/usr/bin/env python3
"""
check_cube_detection_accuracy.py — Compare a known cube position against
what the vision pipeline reports on /delta/all_targets.

Isolates the camera→robot calibration (delta_camera_system/calibrate_transform.py,
calibrate_plane_homography.py) from robot/IK accuracy: this script never moves
the arm, it only listens.

Physical setup before running
------------------------------
    1. Place a single cube in the workspace, stationary.
    2. Measure its true position in the robot base frame (mm) with calipers/
       tape measure from the robot's origin — the same frame_id="robot_base"
       that delta_camera_system publishes in.
    3. z_mm is EE-tip Z (see camera_system.py: z = cube surface + gripper
       offset), matching what /delta/all_targets publishes — NOT platform Z.

Usage
-----
    ros2 run delta_main_app check_cube_detection_accuracy \\
        --ros-args -p known_x_mm:=50.0 -p known_y_mm:=-20.0 -p known_z_mm:=-410.0

Samples are averaged over SAMPLE_WINDOW_S seconds (camera noise), matching
the detection nearest the known position on each frame (in case of multiple
cubes), then prints per-axis and total error plus the sample spread (std
dev) so you can tell calibration bias apart from detection jitter.
"""

import math
import statistics
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray

SAMPLE_WINDOW_S = 3.0


class AccuracyCheckNode(Node):
    def __init__(self):
        super().__init__("check_cube_detection_accuracy")

        self.declare_parameter("known_x_mm", 0.0)
        self.declare_parameter("known_y_mm", 0.0)
        self.declare_parameter("known_z_mm", 0.0)
        self._known = (
            self.get_parameter("known_x_mm").value,
            self.get_parameter("known_y_mm").value,
            self.get_parameter("known_z_mm").value,
        )

        self._samples = []   # list of (x_mm, y_mm, z_mm)

        self.create_subscription(PoseArray, "/delta/all_targets", self._on_targets, 10)

        self.get_logger().info(
            f"Known cube position (robot_base, EE-tip mm): {self._known}\n"
            f"Listening on /delta/all_targets for {SAMPLE_WINDOW_S:.0f}s..."
        )

    def _on_targets(self, msg: PoseArray) -> None:
        if not msg.poses:
            return
        kx, ky, kz = self._known
        nearest = min(
            msg.poses,
            key=lambda p: math.dist(
                (p.position.x * 1000.0, p.position.y * 1000.0, p.position.z * 1000.0),
                (kx, ky, kz),
            ),
        )
        self._samples.append((
            nearest.position.x * 1000.0,
            nearest.position.y * 1000.0,
            nearest.position.z * 1000.0,
        ))


def main(args=None):
    rclpy.init(args=args)
    node = AccuracyCheckNode()

    deadline = time.time() + SAMPLE_WINDOW_S
    while rclpy.ok() and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    if not node._samples:
        node.get_logger().error(
            "No detections received on /delta/all_targets — is delta_camera_system "
            "running and is the cube inside its workspace filter?"
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    xs = [s[0] for s in node._samples]
    ys = [s[1] for s in node._samples]
    zs = [s[2] for s in node._samples]
    mean = (statistics.mean(xs), statistics.mean(ys), statistics.mean(zs))
    std = (
        statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        statistics.pstdev(ys) if len(ys) > 1 else 0.0,
        statistics.pstdev(zs) if len(zs) > 1 else 0.0,
    )
    kx, ky, kz = node._known
    err = (mean[0] - kx, mean[1] - ky, mean[2] - kz)
    err_mag = math.sqrt(sum(e * e for e in err))

    print(f"\n{'':20s} {'X':>10s} {'Y':>10s} {'Z':>10s}")
    print(f"{'known (measured)':20s} {kx:10.2f} {ky:10.2f} {kz:10.2f}")
    print(f"{'detected (mean)':20s} {mean[0]:10.2f} {mean[1]:10.2f} {mean[2]:10.2f}")
    print(f"{'detected (stddev)':20s} {std[0]:10.2f} {std[1]:10.2f} {std[2]:10.2f}   (n={len(xs)} samples)")
    print(f"{'error (detected-known)':20s} {err[0]:+10.2f} {err[1]:+10.2f} {err[2]:+10.2f}")
    print(f"\n|error| = {err_mag:.2f} mm")
    if err_mag > 5.0:
        print(
            "Bias exceeds 5mm and is well above the stddev jitter above — "
            "check calibrate_transform.py / calibrate_plane_homography.py."
        )
    print()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
