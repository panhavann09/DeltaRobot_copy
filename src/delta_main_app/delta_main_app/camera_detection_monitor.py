#!/usr/bin/env python3
"""
camera_detection_monitor.py — Live readout of what the camera pipeline
detects, camera-only, for manually comparing against a tape measure.

Subscribes to /delta/all_targets exactly the way test1.py does (same
topic, same metres→mm conversion, same EE-tip Z convention) and just
prints what comes in — no MATLAB, no motors, no calibration math beyond
what delta_camera_system already applied.

How to use
----------
    1. Start delta_camera_system (the actual detection pipeline).
    2. Run this node.
    3. Place a cube in the workspace and hold a tape measure to it from
       the robot's base-frame origin (X=0, Y=0). Compare the real X/Y/Z
       to the printed values.
    4. Move the cube around / add more cubes and watch the numbers track.

Usage
-----
    ros2 run delta_main_app camera_detection_monitor
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray

from delta_common import config

PRINT_PERIOD_S = 0.5   # throttle prints so the terminal stays readable


class CameraDetectionMonitor(Node):
    def __init__(self):
        super().__init__("camera_detection_monitor")
        self._last_print = 0.0
        self.create_subscription(PoseArray, "/delta/all_targets", self._on_targets, 10)
        self.get_logger().info(
            "Listening on /delta/all_targets — place a cube and compare "
            "the printed X/Y/Z (mm, robot_base, EE-tip Z) against a tape "
            "measure from the robot origin (0,0)."
        )

    def _on_targets(self, msg: PoseArray) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_print < PRINT_PERIOD_S:
            return
        self._last_print = now

        if not msg.poses:
            print("(no detections)")
            return

        print(f"\n{len(msg.poses)} cube(s) detected  —  {'X':>8s} {'Y':>8s} {'Z(tip)':>8s}  {'dist_from_origin':>18s}")
        for i, p in enumerate(msg.poses):
            x_mm = p.position.x * 1000.0
            y_mm = p.position.y * 1000.0
            z_mm = p.position.z * 1000.0   # EE-tip Z, same convention as test1.py
            dist = math.hypot(x_mm - config.HOME_X, y_mm - config.HOME_Y)
            print(f"  [{i}]{'':16s}{x_mm:8.1f} {y_mm:8.1f} {z_mm:8.1f}  {dist:18.1f}")


def main(args=None):
    rclpy.init(args=args)
    node = CameraDetectionMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
