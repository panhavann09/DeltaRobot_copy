#!/usr/bin/env python3
"""
kinematics_test_node.py — drive the motors from a known ground-truth XYZ pose
and report the full IK -> command -> FK-feedback round trip.

Publish a known pose (robot base frame, mm) to /delta/test/target_xyz using
the existing custom_messages/DeltaTarget message, e.g.:

    ros2 topic pub --once /delta/test/target_xyz custom_messages/msg/DeltaTarget \
        "{x_mm: 100.0, y_mm: 0.0, z_mm: -450.0}"

The node solves IK for that exact pose (no EE offset applied), commands the
three motors, waits for the CAN position feedback to settle, runs FK on the
feedback angles, and logs:

    commanded XYZ -> IK angles -> settled feedback angles -> FK(feedback) XYZ -> error (mm)

This isolates the kinematics math + motor tracking accuracy from the rest of
the pick pipeline (no camera, no EE offset, no auto-move logic).
"""

import threading
import time

import rclpy
from rclpy.node import Node

from custom_messages.msg import DeltaTarget
from delta_common import config
from delta_motor_controller.motor_controller import DeltaMotorController


class KinematicsTestNode(Node):
    """Commands motors to an exact XYZ pose via IK, verifies via FK feedback."""

    def __init__(self):
        super().__init__("kinematics_test_node")

        self._ctrl = DeltaMotorController(
            can_port="can1",
            vel_max=config.MOTOR_VEL_MAX,
            acc_set=config.MOTOR_ACC_SET,
        )
        self._explicit_shutdown = False

        if config.ENABLE_MOTORS:
            self._ctrl.connect()
            self.get_logger().info("Motors connected (can1).")
        else:
            self.get_logger().warn("ENABLE_MOTORS=False — dry-run mode, no CAN writes")

        self._lock = threading.Lock()
        self._busy = False

        self._sub = self.create_subscription(
            DeltaTarget,
            "/delta/test/target_xyz",
            self._on_target,
            10,
        )

        self.get_logger().info(
            "KinematicsTestNode ready — publish known poses to /delta/test/target_xyz "
            "(custom_messages/DeltaTarget: x_mm, y_mm, z_mm)"
        )

    def _on_target(self, msg: DeltaTarget) -> None:
        x, y, z = msg.x_mm, msg.y_mm, msg.z_mm

        with self._lock:
            if self._busy:
                self.get_logger().warn(
                    f"Still executing previous move — dropping ({x:.1f},{y:.1f},{z:.1f})"
                )
                return
            self._busy = True

        threading.Thread(target=self._execute_move, args=(x, y, z), daemon=True).start()

    def _execute_move(self, x: float, y: float, z: float) -> None:
        try:
            if not config.ENABLE_MOTORS:
                self.get_logger().info(f"[dry-run] would move to XYZ=({x:.2f},{y:.2f},{z:.2f})")
                return

            ok, ik_deg, fb_deg, fk_xyz, err = self._ctrl.move_xyz(x, y, z, raw=True)

            if ik_deg is None:
                self.get_logger().warn(
                    f"Target ({x:.1f},{y:.1f},{z:.1f}) rejected before IK "
                    "(outside workspace)"
                )
                return
            if fb_deg is None:
                self.get_logger().warn(
                    f"IK=({ik_deg[0]:.2f},{ik_deg[1]:.2f},{ik_deg[2]:.2f})° "
                    "rejected — outside joint limits"
                )
                return
            if fk_xyz is None:
                self.get_logger().warn(
                    f"IK=({ik_deg[0]:.2f},{ik_deg[1]:.2f},{ik_deg[2]:.2f})° commanded, "
                    f"fb=({fb_deg[0]:.2f},{fb_deg[1]:.2f},{fb_deg[2]:.2f})° but FK on "
                    "feedback failed (non-existing position)"
                )
                return

            self.get_logger().info(
                f"cmd XYZ=({x:.2f},{y:.2f},{z:.2f}) -> "
                f"IK=({ik_deg[0]:.2f},{ik_deg[1]:.2f},{ik_deg[2]:.2f})° -> "
                f"fb=({fb_deg[0]:.2f},{fb_deg[1]:.2f},{fb_deg[2]:.2f})° -> "
                f"FK=({fk_xyz[0]:.2f},{fk_xyz[1]:.2f},{fk_xyz[2]:.2f}) "
                f"| err={err:.2f}mm | {'OK' if ok else 'OUT OF TOLERANCE'}"
            )
        except Exception as exc:
            self.get_logger().error(f"Move to ({x:.1f},{y:.1f},{z:.1f}) failed: {exc}")
        finally:
            with self._lock:
                self._busy = False

    def destroy_node(self) -> None:
        if config.ENABLE_MOTORS and self._explicit_shutdown:
            try:
                self._ctrl.move_thetas(0.0, 0.0, 0.0)   # home before disabling
                time.sleep(0.5)
            except Exception:
                pass
            self._ctrl.shutdown()
        super().destroy_node()


def main(args=None):
    import signal
    rclpy.init(args=args)
    node = KinematicsTestNode()

    def handle_sigint(*_):
        node.get_logger().info("Ctrl+C — shutting down cleanly")
        node._explicit_shutdown = True
        node.destroy_node()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, handle_sigint)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
