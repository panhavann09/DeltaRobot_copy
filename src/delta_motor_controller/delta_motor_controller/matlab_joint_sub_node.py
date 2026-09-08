#!/usr/bin/env python3
"""
matlab_joint_sub_node.py — Direct MATLAB joint-angle → CAN1 motor bridge.

Subscribes to /delta/matlab/joint_thetas (custom_messages/DeltaJointAngles),
published by MATLAB with the IK solution (theta1_deg, theta2_deg, theta3_deg,
ik_valid), and drives the three delta-robot motors over CAN1 to those angles.

No camera pipeline, no FSM — this is the minimal bridge for driving the
motors straight from MATLAB-computed joint angles.

MATLAB side (Robotics System Toolbox ≥ R2022b):
    node = ros2node("/matlab_ik");
    pub  = ros2publisher(node, "/delta/matlab/joint_thetas",
                         "custom_messages/DeltaJointAngles");
    msg  = ros2message("custom_messages/DeltaJointAngles");
    msg.theta1_deg = t1;  msg.theta2_deg = t2;  msg.theta3_deg = t3;
    msg.ik_valid   = true;
    send(pub, msg);
"""

import threading
import time

import rclpy
from rclpy.node import Node

from custom_messages.msg import DeltaJointAngles
from delta_common import config
from delta_motor_controller.motor_controller import DeltaMotorController


class MatlabJointSubNode(Node):
    """Drives CAN1 motors directly from MATLAB joint-angle messages."""

    def __init__(self):
        super().__init__("matlab_joint_sub_node")

        self._ctrl = DeltaMotorController(
            can_port="can1",
            vel_max=config.MOTOR_VEL_MAX,
            acc_set=config.MOTOR_ACC_SET,
        )
        self._explicit_shutdown = False   # True only on intentional Ctrl+C

        if config.ENABLE_MOTORS:
            self._ctrl.connect()
            self.get_logger().info("Motors connected (can1).")
        else:
            self.get_logger().warn("ENABLE_MOTORS=False — dry-run mode, no CAN writes")

        self._lock = threading.Lock()
        self._busy = False

        self._sub = self.create_subscription(
            DeltaJointAngles,
            "/delta/matlab/joint_thetas",
            self._on_joint_angles,
            10,
        )

        self.get_logger().info(
            "MatlabJointSubNode ready — listening on /delta/matlab/joint_thetas"
        )

    def _on_joint_angles(self, msg: DeltaJointAngles) -> None:
        if not msg.ik_valid:
            self.get_logger().warn("ik_valid=False — skipping move")
            return

        t1, t2, t3 = msg.theta1_deg, msg.theta2_deg, msg.theta3_deg

        with self._lock:
            if self._busy:
                self.get_logger().warn(
                    f"Still executing previous move — dropping "
                    f"({t1:.1f},{t2:.1f},{t3:.1f})°"
                )
                return
            self._busy = True

        threading.Thread(target=self._execute_move, args=(t1, t2, t3), daemon=True).start()

    def _execute_move(self, t1: float, t2: float, t3: float) -> None:
        try:
            if not config.ENABLE_MOTORS:
                self.get_logger().info(
                    f"[dry-run] would move to θ=({t1:.2f},{t2:.2f},{t3:.2f})°"
                )
                return

            ok, fk_xyz, err, fb_deg = self._ctrl.move_thetas(t1, t2, t3)
            if ok:
                self.get_logger().info(
                    f"Move OK  FK=({fk_xyz[0]:.1f},{fk_xyz[1]:.1f},{fk_xyz[2]:.1f}) "
                    f"err={err:.2f} mm"
                )
            else:
                self.get_logger().warn(
                    f"Move settled with err={err:.2f} mm (above tolerance) "
                    f"fb_deg={fb_deg}"
                )
        except Exception as exc:
            self.get_logger().error(f"Move to ({t1:.1f},{t2:.1f},{t3:.1f})° failed: {exc}")
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
    node = MatlabJointSubNode()

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
