#!/usr/bin/env python3
"""ROS 2 solenoid control node driven by gamepad input and feedback."""

import rclpy
from rclpy.node import Node

from custom_messages.msg import (
    DigitalAndAnalogFeedback,
    DigitalAndSolenoidCommand,
)


class SolenoidControlNode(Node):
    """Drive the hand/push solenoids from gamepad and analog feedback."""

    PUBLISH_CAN_ID = 4
    DETECTION_CAN_ID = 500
    DETECTION_THRESHOLD = 0.5
    PUSH_TICKS = 20
    HOLD_TICKS = 25
    DETECT_TICKS = 35

    def __init__(self):
        super().__init__('valve_can_node')
        self._init_state()
        self._init_interfaces()
        self.publish_solenoid()

    def _init_state(self):
        self.sensor_value = 0.0
        self.dribbling = False
        self.start_detect = False
        self.close_hand = False

        self.solenoid_push = False
        self.solenoid_hand = False

        self.counter_loop = 0

    def _init_interfaces(self):
        self.solenoid_publisher = self.create_publisher(
            DigitalAndSolenoidCommand,
            '/publish_digital_solenoid',
            10,
        )


    def _cancel_cycle(self):
        self.dribbling = False
        self.start_detect = False
        self.close_hand = False
        self.counter_loop = 0
        self.solenoid_push = False

    def _start_cycle(self):
        self.dribbling = True
        self.start_detect = False
        self.close_hand = False
        self.counter_loop = 0
        self.control_loop()

    def gamepad_callback(self, msg):
        if msg.button_a and not msg.previous_button_a:
            self._cancel_cycle()
            self.solenoid_hand = not self.solenoid_hand
            self.get_logger().info(
                f"Manual hand toggle: {'closed' if self.solenoid_hand else 'open'}"
            )
            self.publish_solenoid()

        if msg.button_lb and not msg.previous_button_lb:
            self._start_cycle()
            self.get_logger().info('Started dribbling sequence')
            self.publish_solenoid()

    def digital_and_analog_callback(self, msg):
        if msg.can_id == self.DETECTION_CAN_ID:
            self.sensor_value = msg.analog2_value
            if self.start_detect and self.sensor_value >= self.DETECTION_THRESHOLD:
                self.close_hand = True

        if self.dribbling:
            self.counter_loop += 1
            self.control_loop()

        self.publish_solenoid()

    def DigitalAndAnalog_callback(self, msg):
        """Backward-compatible alias for older callback naming."""
        self.digital_and_analog_callback(msg)

    def control_loop(self):
        if self.close_hand:
            self.solenoid_hand = False
            self.solenoid_push = False
            self.dribbling = False
            self.start_detect = False
            self.close_hand = False
            self.counter_loop = 0
            self.get_logger().info('Object detected, closing hand')
            return

        if not self.dribbling:
            return

        if self.counter_loop <= self.PUSH_TICKS:
            self.solenoid_push = True
            self.solenoid_hand = True
        elif self.counter_loop <= self.HOLD_TICKS:
            self.solenoid_push = False
            self.solenoid_hand = True
        elif self.counter_loop <= self.DETECT_TICKS:
            self.solenoid_push = False
            self.solenoid_hand = True
            self.start_detect = True
        else:
            self.solenoid_push = False
            self.solenoid_hand = True
            self.start_detect = True

    def publish_solenoid(self):
        # Only solenoid1 (hand) is actually wired on this board — every other
        # solenoid bit must always stay False, regardless of the push/dribble
        # state machine above.
        solenoid_msg = DigitalAndSolenoidCommand()
        solenoid_msg.can_id = self.PUBLISH_CAN_ID
        solenoid_msg.solenoid1_value = self.solenoid_hand
        solenoid_msg.solenoid2_value = False
        solenoid_msg.solenoid3_value = False
        solenoid_msg.solenoid4_value = False
        solenoid_msg.solenoid5_value = False
        solenoid_msg.solenoid6_value = False
        self.solenoid_publisher.publish(solenoid_msg)


Solenoid_control = SolenoidControlNode
ValveCANNode = SolenoidControlNode


def main(args=None):
    rclpy.init(args=args)
    main_node = SolenoidControlNode()
    try:
        rclpy.spin(main_node)
    finally:
        main_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

