#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from custom_messages.msg import DeltaJointAngles
from rob_and_ros_pkg.robstride_controller import PositionController

MOTOR_IDS = [1, 2, 3]   # adjust to your CAN IDs

class DeltaMotorSubscriber(Node):
    def __init__(self):
        super().__init__('delta_motor_sub_node')

        self.controllers = {}
        for mid in MOTOR_IDS:
            ctrl = PositionController(mid)
            if not ctrl.connect():
                # disconnect any that already connected, then bail
                for c in self.controllers.values():
                    c.stop_and_exit()
                raise RuntimeError(f"Motor {mid} connection failed")
            self.controllers[mid] = ctrl

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,          # always act on the freshest target
        )
        self.subscription = self.create_subscription(
            DeltaJointAngles,
            '/delta/matlab/joint_thetas',
            self.angle_callback,
            qos,
        )
        self.get_logger().info("✅ Delta Motor Subscriber Ready (3 motors)")

    def angle_callback(self, msg: DeltaJointAngles):
        # Assumes Simulink sends ABSOLUTE angles in RADIANS
        targets = [msg.theta1, msg.theta2, msg.theta3]
        for mid, target_rad in zip(MOTOR_IDS, targets):
            ctrl = self.controllers[mid]
            delta_deg = math.degrees(target_rad) - math.degrees(ctrl.position)
            ctrl.set_angle_relative(delta_deg)

        self.get_logger().info(
            "Target [deg]: " + ", ".join(f"{math.degrees(t):.2f}" for t in targets)
        )

    def destroy_node(self):
        for ctrl in self.controllers.values():
            ctrl.stop_and_exit()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DeltaMotorSubscriber()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"Startup failed: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from custom_message.msg import DeltaJointAngles
from rob_and_ros_pkg.robstride_controller import PositionController

MOTOR_IDS = [1, 2, 3]   # adjust to your CAN IDs

class DeltaMotorSubscriber(Node):
    def __init__(self):
        super().__init__('delta_motor_sub_node')

        self.controllers = {}
        for mid in MOTOR_IDS:
            ctrl = PositionController(mid)
            if not ctrl.connect():
                # disconnect any that already connected, then bail
                for c in self.controllers.values():
                    c.stop_and_exit()
                raise RuntimeError(f"Motor {mid} connection failed")
            self.controllers[mid] = ctrl

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,          # always act on the freshest target
        )
        self.subscription = self.create_subscription(
            DeltaJointAngles,
            '/delta/matlab/joint_thetas',
            self.angle_callback,
            qos,
        )
        self.get_logger().info("✅ Delta Motor Subscriber Ready (3 motors)")

    def angle_callback(self, msg: DeltaJointAngles):
        # Assumes Simulink sends ABSOLUTE angles in RADIANS
        targets = [msg.theta1, msg.theta2, msg.theta3]
        for mid, target_rad in zip(MOTOR_IDS, targets):
            ctrl = self.controllers[mid]
            delta_deg = math.degrees(target_rad) - math.degrees(ctrl.position)
            ctrl.set_angle_relative(delta_deg)

        self.get_logger().info(
            "Target [deg]: " + ", ".join(f"{math.degrees(t):.2f}" for t in targets)
        )

    def destroy_node(self):
        for ctrl in self.controllers.values():
            ctrl.stop_and_exit()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DeltaMotorSubscriber()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"Startup failed: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
