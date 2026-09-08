"""
Publishes /joint_states for the delta_description URDF.

The URDF can't express the robot's real closed kinematic loop, so it's split
into two disconnected branches (3 arm chains + a platform chain, see
delta_robot.urdf.xacro). This node is what keeps them geometrically
consistent: it takes the 3 commanded shoulder angles, runs them through the
same DeltaGeometry FK used on the real robot (delta_common.fk_ik), and
publishes the resulting elbow + platform joint values every tick.

Drive it with `ros2 param set` or rqt_reconfigure, e.g.:
    ros2 param set /joint_state_bridge theta1_deg 30.0

By default this only drives the RViz model. Pass the `drive_real_motors`
parameter to also mirror theta1/2/3_deg onto the real robot over CAN — this
turns the node into a standalone manual jog tool, isolated from
delta_main_app (no camera, no MATLAB, no pick/place logic):
    ros2 launch delta_description view_delta.launch.py \
        enable_camera:=false drive_real_motors:=true

Do NOT run with drive_real_motors:=true at the same time as main_app or
matlab_bridge_node — they'd all open can1 and fight over the same motors.
Connecting homes the real robot to theta=(0,0,0) immediately (motor_controller's
init_zero()), matching this node's own default RViz pose.

With drive_real_motors:=true, /joint_states reflects real CAN encoder
feedback, not the commanded setpoint: while idle it polls live position at
30 Hz, and while a jog is settling it holds the last published pose (polling
concurrently would race the jog's own CAN reads) then snaps to the real
settled position once the move completes. Without drive_real_motors, there's
no real robot to read from, so it publishes the commanded thetas directly.
"""
import math
import threading
import time

import rclpy
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from delta_common import config
from delta_common.fk_ik import JOINT_NAMES, DeltaGeometry, joint_state
from delta_motor_controller.motor_controller import DeltaMotorController

PUBLISH_RATE_HZ = 30.0
THETA_MOVE_EPS_DEG = 0.3   # ignore changes smaller than this (float/slider noise)


class JointStateBridge(Node):
    def __init__(self):
        super().__init__('joint_state_bridge')
        self._geom = DeltaGeometry()

        limits = {
            'theta1_deg': (config.THETA1_MIN, config.THETA1_MAX),
            'theta2_deg': (config.THETA2_MIN, config.THETA2_MAX),
            'theta3_deg': (config.THETA3_MIN, config.THETA3_MAX),
        }
        for name, (lo, hi) in limits.items():
            angle_range = FloatingPointRange(from_value=float(lo), to_value=float(hi), step=0.1)
            descriptor = ParameterDescriptor(floating_point_range=[angle_range])
            self.declare_parameter(name, 0.0, descriptor)

        self.declare_parameter('drive_real_motors', False)
        self._drive_real_motors = (
            self.get_parameter('drive_real_motors').value and config.ENABLE_MOTORS
        )

        self._pub = self.create_publisher(JointState, 'joint_states', 10)

        self._ctrl = None
        self._explicit_shutdown = False
        self._move_lock = threading.Lock()
        self._move_busy = False
        self._last_sent_deg = (0.0, 0.0, 0.0)
        if self._drive_real_motors:
            self._ctrl = DeltaMotorController(
                can_port="can1",
                vel_max=config.MOTOR_VEL_MAX,
                acc_set=config.MOTOR_ACC_SET,
            )
            self._ctrl.connect()   # also homes to theta=(0,0,0) via init_zero()
            self.get_logger().warn(
                "drive_real_motors=True — the real robot will physically follow "
                "theta1/2/3_deg. Make sure nothing else (main_app, matlab_bridge_node) "
                "is holding can1.")
        else:
            self.get_logger().info("Visualization only — no CAN connection opened.")

        self._timer = self.create_timer(1.0 / PUBLISH_RATE_HZ, self._on_timer)
        self.get_logger().info(
            'joint_state_bridge up — set theta1_deg/theta2_deg/theta3_deg via '
            'ros2 param set or rqt_reconfigure')

    def _publish_joint_states(self, thetas_deg) -> None:
        thetas_rad = [math.radians(t) for t in thetas_deg]
        values = joint_state(self._geom, thetas_rad)
        if values is None:
            self.get_logger().warn(
                f'thetas {thetas_deg} deg unreachable (IK spheres do not intersect) '
                '— holding last pose', throttle_duration_sec=2.0)
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = [values[name] for name in JOINT_NAMES]
        self._pub.publish(msg)

    def _on_timer(self):
        thetas_deg = [
            self.get_parameter(name).value
            for name in ('theta1_deg', 'theta2_deg', 'theta3_deg')
        ]

        if not self._drive_real_motors:
            self._publish_joint_states(thetas_deg)
            return

        self._maybe_command_motors(tuple(thetas_deg))
        with self._move_lock:
            busy = self._move_busy
        if busy:
            # A jog is mid-settle and already polling CAN in its own thread —
            # reading feedback here too would steal its response frame (same
            # hazard matlab_bridge_node.py guards against). It republishes
            # fresh real feedback itself the moment it settles, so just hold
            # the last published pose until then.
            return
        try:
            fb_deg = self._ctrl.get_current_thetas_deg()
            self._publish_joint_states(fb_deg)
        except Exception as exc:
            self.get_logger().debug(f"Encoder feedback read failed: {exc}")

    def _maybe_command_motors(self, thetas_deg) -> None:
        changed = any(
            abs(a - b) > THETA_MOVE_EPS_DEG
            for a, b in zip(thetas_deg, self._last_sent_deg)
        )
        if not changed:
            return
        with self._move_lock:
            if self._move_busy:
                return   # previous jog still settling — next tick will pick up the latest value
            self._move_busy = True
        self._last_sent_deg = thetas_deg
        threading.Thread(target=self._run_move, args=(thetas_deg,), daemon=True).start()

    def _run_move(self, thetas_deg) -> None:
        try:
            ok, fk_xyz, err, fb_deg = self._ctrl.move_thetas(*thetas_deg)
            if fb_deg is not None:
                self._publish_joint_states(fb_deg)   # snap RViz to the real settled encoder pose
            if not ok:
                self.get_logger().warn(f"Jog to {thetas_deg} deg did not settle cleanly (err={err})")
        except Exception as exc:
            self.get_logger().error(f"Jog move to {thetas_deg} deg failed: {exc}")
        finally:
            with self._move_lock:
                self._move_busy = False

    def destroy_node(self) -> None:
        if self._drive_real_motors and self._explicit_shutdown and self._ctrl is not None:
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
    node = JointStateBridge()

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


if __name__ == '__main__':
    main()
