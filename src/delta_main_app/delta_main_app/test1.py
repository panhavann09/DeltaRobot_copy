#!/usr/bin/env python3
"""
test1.py — Republish camera_node's cube targets as base_link poses for RViz.

camera_node (delta_camera_system) already runs the real YOLO-based
detection pipeline and publishes /delta/all_targets: a PoseArray of
detected objects in x_base/y_base/z_base (meters), computed via its own
calibrated camera_to_base_mm() — the same numbers the robot actually
picks with. This node just relabels that frame as base_link for RViz;
it does no detection of its own.

(Earlier versions of this file ran an independent HSV-blob detector and
a separate tf2 camera→base_link transform. That duplicated, weaker path
produced scattered false-positive poses and could disagree with the
calibration actually used for picking, so it was replaced with a
straight republish.)

Subscribe:
  /delta/all_targets            geometry_msgs/PoseArray, frame_id=robot_base

Publish:
  /delta/cube_poses_base_link   geometry_msgs/PoseArray, frame_id=base_link
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray

BASE_FRAME = "base_link"


class CubePoseNode(Node):

    def __init__(self):
        super().__init__("test1_cube_pose_node")

        self.create_subscription(PoseArray, "/delta/all_targets", self._on_targets, 10)
        self._pub_poses = self.create_publisher(PoseArray, "/delta/cube_poses_base_link", 10)

        self.get_logger().info(
            "test1 ready — republishing /delta/all_targets as "
            f"/delta/cube_poses_base_link (frame_id={BASE_FRAME})"
        )

    def _on_targets(self, msg: PoseArray) -> None:
        arr = PoseArray()
        arr.header.stamp = msg.header.stamp
        arr.header.frame_id = BASE_FRAME
        arr.poses = msg.poses
        self._pub_poses.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = CubePoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
