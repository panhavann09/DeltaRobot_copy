#!/usr/bin/env python3
"""
main_app.py  -  Delta Robot Main Application
Replaces the stub in delta_main_app/delta_main_app/main_app.py

Subscribes:
    /delta/target_xyz       PointStamped  - object position in base frame (mm)
    /delta/detection_status String        - camera detection state

Publishes:
    /delta/gripper_cmd      Float32       - gripper position [0=open, 1=closed]
    /delta/robot_state      String        - current FSM state name

IMPORTANT - Theta constraint:
    All three motor joint angles must stay within [0 deg, ~85 deg].
    Negative thetas are physically unreachable in this robot's workspace.
    Z_MAX = -196.875 mm is the upper (shallowest) reachable position.
    Approach and lift Z values are clamped to Z_MAX before every move
    so they never generate negative theta solutions.

State Machine:
    IDLE -> CONFIRMING -> APPROACHING -> PICKING ->
    GRASPING -> LIFTING -> TRANSPORTING -> DROPPING -> RESETTING -> IDLE
"""

import math
import time
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseArray
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, String

from delta_common import config
from delta_common.fk_ik import JOINT_NAMES, DeltaGeometry, check_workspace, joint_state, solve_fk_mm, solve_ik_mm
from delta_motor_controller.motor_controller import DeltaMotorController
from delta_motor_controller.pneumatic_gripper import PneumaticGripper
from delta_main_app.belt_predictor import BeltPredictor

# -- Pick-and-place geometry (mm, robot base frame) --------------------
# All Z values must stay within [config.Z_MIN, config.Z_MAX]
# i.e. between -500.0 and -196.875 mm.
# The helpers _approach_z() and _lift_z() clamp automatically.

APPROACH_Z_OFFSET = 0.0
LIFT_Z_OFFSET = 30.0
HOME_Z = config.HOME_Z

GRASP_WAIT_SEC = 0.4
DROP_WAIT_SEC = 0.4
MOVE_SETTLE_SEC = 0.05

LASER_ALIGN_TIMEOUT_S = 1.0   # max time to wait for laser-to-object alignment before grip
LASER_ERROR_MAX_AGE_S = 1.0   # discard /delta/laser_error_mm readings older than this

CONFIRM_FRAMES = 2
STABLE_THRESH_MM = 10.0


def _clamp_z(z: float) -> float:
    """
    Clamp Z to the valid workspace range [Z_MIN, Z_MAX].
    This prevents moves that would require negative theta angles.

    Z is negative in this frame (below the base plane):
        Z_MAX = -196.875 mm  (shallowest / closest to base - theta approaches 0 deg)
        Z_MIN = -500.0   mm  (deepest - theta approaches max)

    If an approach/lift offset would push Z above Z_MAX, it is silently
    clamped to Z_MAX so the arm stays reachable.
    """
    return max(config.Z_MIN, min(config.Z_MAX, z))


def _approach_z(z_obj: float) -> float:
    """Z position above the object, clamped to workspace."""
    return _clamp_z(z_obj + APPROACH_Z_OFFSET)


def _lift_z(z_obj: float) -> float:
    """Z position to lift to after grasping, clamped to workspace."""
    return _clamp_z(z_obj + LIFT_Z_OFFSET)


class PickAndPlaceStateMachine:
    """
    Pure state machine - no ROS dependency.
    Receives events from the ROS node and calls motor_controller + gripper.
    All blocking arm moves run in a background thread.
    """

    def __init__(
        self,
        controller: DeltaMotorController,
        gripper_pub,
        state_pub,
        logger,
        get_next_pick_xy=None,
        get_laser_error_fn=None,
    ):
        self._ctrl = controller
        self._gripper_pub = gripper_pub
        self._state_pub = state_pub
        self._log = logger
        self._state = "IDLE"
        self._get_next_pick_xy = get_next_pick_xy
        self._get_laser_error = get_laser_error_fn   # fn() -> (dx, dy, tol_mm, age_s) | None

        self._recent_xyz = []
        self._target_xyz = None
        self._busy = False
        self._lock = threading.Lock()
        self._ee_error_x = None
        self._ee_error_y = None

    @property
    def current_pick_y(self):
        """Y (lateral, stable) of the object currently being picked, or None when idle."""
        with self._lock:
            return self._target_xyz[1] if self._target_xyz else None

    def detection_update(self, x: float, y: float, z: float) -> None:
        with self._lock:
            if self._busy:
                return

            if self._state == "IDLE":
                self._state = "CONFIRMING"
                self._recent_xyz = []
                self._publish_state()

            if self._state == "CONFIRMING":
                self._recent_xyz.append((x, y, z))
                if len(self._recent_xyz) > CONFIRM_FRAMES:
                    self._recent_xyz.pop(0)

                if len(self._recent_xyz) < CONFIRM_FRAMES:
                    return

                if self._is_stable(self._recent_xyz):
                    avg_x = sum(p[0] for p in self._recent_xyz) / CONFIRM_FRAMES
                    avg_y = sum(p[1] for p in self._recent_xyz) / CONFIRM_FRAMES
                    avg_z = sum(p[2] for p in self._recent_xyz) / CONFIRM_FRAMES
                    self._target_xyz = (avg_x, avg_y, avg_z)
                    self._log.info(
                        f"Target confirmed: ({avg_x:.1f}, {avg_y:.1f}, {avg_z:.1f}) mm"
                        f"  approach_z={_approach_z(avg_z):.1f}"
                        f"  lift_z={_lift_z(avg_z):.1f}"
                    )
                    self._start_sequence()

    def force_target(self, x: float, y: float, z: float) -> bool:
        """Start a pick sequence immediately, bypassing confirmation.
        Camera already confirmed the target — no need to buffer frames.
        Returns False if the robot is currently busy."""
        az = _approach_z(z)
        lz = _lift_z(z)
        ee_off = config.EE_OFFSET_Z_MM
        for cx, cy, cz_ee, label in (
            (x,   y,   az,    "approach"),
            (x,   y,   z,     "pick"),
            (x,   y,   lz,    "lift"),
            (0.0, 0.0, -350.0, "home"),
        ):
            # home uses raw platform Z; others are EE-tip Z that need offset conversion
            cz = cz_ee if label == "home" else cz_ee + ee_off
            ok, t1, t2, t3 = solve_ik_mm(cx, cy, cz)
            if ok:
                self._log.info(
                    f"IK pre-check {label}: ({cx:.1f},{cy:.1f},{cz:.1f})"
                    f" → OK t=({t1:.1f},{t2:.1f},{t3:.1f})"
                )
            else:
                self._log.info(
                    f"IK pre-check {label}: ({cx:.1f},{cy:.1f},{cz:.1f}) → FAIL"
                )
        with self._lock:
            if self._busy:
                return False
            self._target_xyz = (x, y, z)
            self._log.info(f"Queue target: ({x:.1f}, {y:.1f}, {z:.1f}) mm")
            self._start_sequence()
            return True

    def no_detection(self) -> None:
        with self._lock:
            if self._busy:
                return
            if self._state == "CONFIRMING":
                self._recent_xyz = []
                self._state = "IDLE"
                self._publish_state()

    @property
    def state(self) -> str:
        return self._state

    @staticmethod
    def _is_stable(positions) -> bool:
        if len(positions) < 2:
            return False
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        spread = math.sqrt(
            (max(xs) - min(xs)) ** 2
            + (max(ys) - min(ys)) ** 2
            + (max(zs) - min(zs)) ** 2
        )
        return spread < STABLE_THRESH_MM

    def update_ee_error(self, err_x: float, err_y: float) -> None:
        with self._lock:
            if self._state not in ("APPROACHING", "PICKING"):
                return
            self._ee_error_x = err_x
            self._ee_error_y = err_y

    def _get_ee_error(self):
        """Return latest EE error (mm) or (None, None) if not available."""
        with self._lock:
            if self._ee_error_x is None:
                return None, None
            return self._ee_error_x, self._ee_error_y

    def _wait_fresh_ee_error(self, timeout=1.0):
        """Collect 5 fresh error samples and return their median.

        One bad frame from a shiny reflection can't drive a correction by itself.
        """
        deadline = time.time() + timeout
        samples_x, samples_y = [], []
        while time.time() < deadline and len(samples_x) < 2:
            with self._lock:
                self._ee_error_x = None
                self._ee_error_y = None
            t_s = time.time()
            while time.time() - t_s < 0.15:
                time.sleep(0.02)
                with self._lock:
                    if self._ee_error_x is not None:
                        samples_x.append(self._ee_error_x)
                        samples_y.append(self._ee_error_y)
                        break
        if not samples_x:
            return None, None
        samples_x.sort()
        samples_y.sort()
        n = len(samples_x)
        return samples_x[n // 2], samples_y[n // 2]

    def _wait_for_laser_alignment(self, timeout=LASER_ALIGN_TIMEOUT_S):
        """Poll /delta/laser_error_mm until the laser dot lands within its
        tolerance radius of the object, or timeout elapses.

        Returns (aligned, dist_mm).  If no laser feedback is wired up
        (get_laser_error_fn=None), alignment is skipped and treated as OK so
        deployments without the laser configured keep picking unconditionally.
        """
        if self._get_laser_error is None:
            return True, None
        deadline = time.time() + timeout
        while time.time() < deadline:
            reading = self._get_laser_error()
            if reading is not None:
                dx, dy, tol_mm, age_s = reading
                if age_s <= LASER_ERROR_MAX_AGE_S:
                    dist = math.hypot(dx, dy)
                    if dist <= tol_mm:
                        return True, dist
            time.sleep(0.02)
        return False, None

    def _start_sequence(self) -> None:
        self._busy = True
        threading.Thread(target=self._run_sequence, daemon=True).start()

    def _run_sequence(self) -> None:
        from delta_common.fk_ik import solve_ik_mm

        x, y, z = self._target_xyz
        z_platform = z + config.EE_OFFSET_Z_MM   # EE-tip → platform Z for all moves
        az = _approach_z(z_platform)
        lz = _lift_z(z_platform)

        ok_az, *_ = solve_ik_mm(x, y, az)
        use_approach = ok_az
        if not use_approach:
            self._log.warn(
                f"Approach Z={az:.1f} unreachable at XY=({x:.1f},{y:.1f}) "
                f"— skipping approach, descending directly to Z={z_platform:.1f} (ee_tip={z:.1f})"
            )

        if lz != z_platform + LIFT_Z_OFFSET:
            self._log.warn(
                f"Lift Z clamped: {z_platform + LIFT_Z_OFFSET:.1f} -> {lz:.1f} mm"
            )

        try:
            self._set_state("APPROACHING")
            if use_approach:
                if not self._move(x, y, az):
                    raise RuntimeError("APPROACH failed")
                time.sleep(MOVE_SETTLE_SEC)

                # ── Dynamic arrival gate ─────────────────────────────────────
                # Wait at approach Z until the object is under the EE in X
                # (|err_x| < threshold).  This replaces the fixed approach wait
                # and adapts automatically to belt speed variation.
                if config.CONVEYOR_MODE and config.EE_CORRECTION_ENABLE:
                    deadline_arr = time.time() + config.CONVEYOR_ARRIVAL_TIMEOUT_S
                    no_signal = 0
                    while time.time() < deadline_arr:
                        ex, _ = self._wait_fresh_ee_error(timeout=0.1)
                        if ex is None:
                            no_signal += 1
                            if no_signal >= 5:
                                self._log.warn("Arrival gate: no EE signal — proceeding")
                                break
                            continue
                        no_signal = 0
                        self._log.info(f"Arrival wait: err_x={ex:+.1f}mm")
                        if abs(ex) < config.CONVEYOR_ARRIVAL_X_THRESH_MM:
                            self._log.info("Object arrived under EE")
                            break
                elif config.CONVEYOR_APPROACH_WAIT_SEC > 0:
                    time.sleep(config.CONVEYOR_APPROACH_WAIT_SEC)

                if config.EE_CORRECTION_ENABLE:
                    t_start = time.time()
                    self._log.info(
                        f"EE correction start  "
                        f"max={config.EE_CORRECTION_MAX_ITERS}  "
                        f"thresh={config.EE_CORRECTION_THRESH_MM}mm"
                    )
                    for i in range(config.EE_CORRECTION_MAX_ITERS):

                        err_x, err_y = self._wait_fresh_ee_error(
                            timeout=config.EE_CORRECTION_WAIT_S
                        )

                        if err_x is None:
                            self._log.warn(f"Iter {i}: no EE error — stop")
                            break

                        dist_2d = math.sqrt(err_x**2 + err_y**2)
                        # On conveyor, belt predictor owns X timing — only correct Y.
                        check_err = abs(err_y) if config.CONVEYOR_MODE else dist_2d
                        self._log.info(
                            f"Correction iter {i}: "
                            f"err_x={err_x:+.1f}mm err_y={err_y:+.1f}mm "
                            f"check={check_err:.1f}mm"
                        )

                        if check_err < config.EE_CORRECTION_MIN_MM:
                            self._log.info("Below MIN — done")
                            break

                        if check_err < config.EE_CORRECTION_THRESH_MM:
                            self._log.info("Converged")
                            break

                        if time.time() - t_start > config.EE_CORRECTION_TIMEOUT_S:
                            self._log.warn("Correction timeout — proceeding")
                            break

                        # X correction disabled in conveyor mode — belt predictor handles timing
                        x_new = x if config.CONVEYOR_MODE else x - err_x * config.EE_CORRECTION_GAIN
                        y_new = y - err_y * config.EE_CORRECTION_GAIN

                        x_new = max(-config.X_LIMIT + 5.0,
                                min( config.X_LIMIT - 5.0, x_new))
                        y_new = max(-config.Y_LIMIT + 5.0,
                                min( config.Y_LIMIT - 5.0, y_new))

                        x = x_new
                        y = y_new
                        self._log.info(
                            f"  → move X={x:.1f}mm Y={y:.1f}mm"
                        )
                        self._move(x, y, az)

                    time.sleep(0.1)

            self._set_state("PICKING")
            pick_z = _clamp_z(config.PICK_Z)
            if not self._move(x, y, pick_z):
                raise RuntimeError("PICK descend failed")
            time.sleep(MOVE_SETTLE_SEC)

            aligned, dist = self._wait_for_laser_alignment()
            if not aligned:
                raise RuntimeError(
                    f"Laser not aligned with object after "
                    f"{LASER_ALIGN_TIMEOUT_S:.1f}s — aborting grip"
                )
            if dist is not None:
                self._log.info(f"Laser aligned (dist={dist:.1f}mm) — gripping")

            self._set_state("GRASPING")
            self._gripper(1.0)
            time.sleep(GRASP_WAIT_SEC)
            with self._lock:
                self._ee_error_x = None
                self._ee_error_y = None

            self._set_state("LIFTING")
            if not self._move(x, y, lz):
                raise RuntimeError("LIFT failed")
            time.sleep(MOVE_SETTLE_SEC)

            self._set_state("TRANSPORTING")
            place_z = _clamp_z(config.PLACE_Z)
            if not self._move(config.PLACE_X, config.PLACE_Y, place_z):
                raise RuntimeError("TRANSPORT failed")
            time.sleep(MOVE_SETTLE_SEC)

            self._set_state("DROPPING")
            self._gripper(0.0)
            time.sleep(DROP_WAIT_SEC)

        except RuntimeError as exc:
            self._log.error(f"Sequence aborted: {exc}")
            self._gripper(0.0)
        except Exception as exc:
            # Non-RuntimeError (e.g. CAN bus exception from bus.read/write) — log
            # and fall through to finally so _busy is always cleared.
            self._log.error(f"Unexpected error in sequence: {exc}", exc_info=True)
            try:
                self._gripper(0.0)
            except Exception:
                pass

        finally:
            self._set_state("RESETTING")
            # Pre-position toward the next queued pick instead of going to
            # (0, 0).  This turns two sequential moves (home then
            # approach) into a single diagonal move, saving ~0.3–0.4 s.
            try:
                next_xy = self._get_next_pick_xy() if self._get_next_pick_xy else None
                if next_xy:
                    nx, ny = next_xy
                    self._log.info(f"Reset: pre-positioning at ({nx:.1f}, {ny:.1f}, {HOME_Z:.0f})")
                    self._move(nx, ny, HOME_Z, raw=True)
                else:
                    ok, fk_xyz, _, _ = self._ctrl.move_thetas(0.0, 0.0, 0.0)   # exact θ=0,0,0 home
                    if ok and fk_xyz is not None:
                        self._current_ee_xyz = (fk_xyz[0], fk_xyz[1], fk_xyz[2] - config.EE_OFFSET_Z_MM)
            except Exception as exc:
                self._log.error(f"Reset move failed: {exc}")
            finally:
                with self._lock:
                    self._state = "IDLE"
                    self._busy = False
                    self._target_xyz = None
                    self._recent_xyz = []
                self._publish_state()
                self._log.info("Sequence complete - IDLE")

    def _move(self, x: float, y: float, z: float, raw: bool = False) -> bool:
        from delta_common.fk_ik import solve_ik_mm

        ok, t1, t2, t3 = solve_ik_mm(x, y, z)
        if not ok:
            self._log.error(f"IK no solution for ({x:.1f}, {y:.1f}, {z:.1f})")
            return False
        if t1 < 0 or t2 < 0 or t3 < 0:
            self._log.error(
                f"Negative theta rejected: theta=({t1:.1f}, {t2:.1f}, {t3:.1f}) "
                f"for ({x:.1f}, {y:.1f}, {z:.1f}) - Z too shallow?"
            )
            return False

        self._log.info(
            f"  Move ({x:.1f}, {y:.1f}, {z:.1f})  "
            f"theta=({t1:.1f}, {t2:.1f}, {t3:.1f})"
        )
        ok, _, _, _, err = self._ctrl.move_xyz(x, y, z, raw=raw)
        if ok:
            self._log.info(f"  FK err={err:.2f} mm  OK")
            # raw=True: (x,y,z) is already the platform target -> tip = z - offset.
            # raw=False: move_xyz added EE_OFFSET_Z_MM internally to convert this
            # tip-frame (x,y,z) into the platform target, so z itself IS the tip.
            if raw:
                self._current_ee_xyz = (x, y, z - config.EE_OFFSET_Z_MM)
            else:
                self._current_ee_xyz = (x, y, z)
        else:
            self._log.warn("  Move FAILED")
        return ok

    def _gripper(self, pos: float) -> None:
        self._log.info(f'  Gripper -> {"CLOSE" if pos > 0.5 else "OPEN"}')
        self._gripper_pub(pos)

    def _set_state(self, state: str) -> None:
        self._state = state
        self._publish_state()
        self._log.info(f"[FSM] -> {state}")

    def _publish_state(self) -> None:
        self._state_pub(self._state)


class DeltaMainApp(Node):
    def __init__(self):
        super().__init__("delta_main_app")

        self._explicit_shutdown = False   # True only on intentional Ctrl+C
        self._ctrl = DeltaMotorController(
            can_port="can1",
            vel_max=config.MOTOR_VEL_MAX,
            acc_set=config.MOTOR_ACC_SET,
        )
        if config.ENABLE_MOTORS:
            self._ctrl.connect()
            self.get_logger().info("Motors connected.")
        else:
            self.get_logger().warn("ENABLE_MOTORS=False - dry run mode")

        self._gripper_pub = self.create_publisher(Float32, "/delta/gripper_cmd", 10)
        self._state_pub = self.create_publisher(String, "/delta/robot_state", 10)

        self._gripper = PneumaticGripper(can_channel='can2', can_id=600)
        if config.ENABLE_MOTORS:
            self._gripper.connect()
            self.get_logger().info("Gripper connected (direct CAN).")

        self._fsm = PickAndPlaceStateMachine(
            controller=self._ctrl,
            gripper_pub=self._send_gripper,
            state_pub=self._send_state,
            logger=self.get_logger(),
            get_next_pick_xy=self._peek_next_pick_xy,
            get_laser_error_fn=self._get_laser_error,
        )

        self._target_queue: list = []
        self._belt_vx_mm_s: float = 0.0
        self._queue_lock = threading.Lock()
        self._predictor = BeltPredictor()
        self._verify_deadline: float = 0.0

        self._sub_all_targets = self.create_subscription(
            PoseArray, "/delta/all_targets", self._all_targets_callback, 10
        )
        self._sub_velocity = self.create_subscription(
            PointStamped, "/delta/object_velocity_mm_s", self._velocity_callback, 10
        )
        self._sub_status = self.create_subscription(
            String, "/delta/detection_status", self._status_callback, 10
        )
        self._ee_error_x = 0.0
        self._ee_error_y = 0.0
        self._ee_error_count = 0
        self._sub_ee_error = self.create_subscription(
            PointStamped, "/delta/ee_error_mm", self._ee_error_callback, 10
        )
        # Latest reading from camera_system's true laser-vs-object error
        # (dx_mm, dy_mm, tol_mm, receipt_time) — gates the grip in PICKING.
        self._latest_laser_error = None
        self._sub_laser_error = self.create_subscription(
            PointStamped, "/delta/laser_error_mm", self._on_laser_error, 10
        )
        self._pub_ee_fk = self.create_publisher(PointStamped, "/delta/ee_fk_xyz", 10)
        self._pub_ff_target = self.create_publisher(
            PointStamped, "/delta/matlab/target_xyz_mm", 10
        )
        self._pub_ff_ee = self.create_publisher(
            PointStamped, "/delta/matlab/ee_position_mm", 10
        )
        # Real motor-encoder joint state — lets robot_state_publisher/RViz
        # animate the URDF with the real robot's live pose.
        self._joint_geom = DeltaGeometry()
        self._pub_joint_states = self.create_publisher(JointState, "joint_states", 10)
        self._fsm._current_ee_xyz = None
        self.create_timer(0.1, self._queue_timer)
        self.create_timer(0.05, self._publish_ee_fk_timer)
        self.create_timer(0.05, self._publish_feedforward_timer)

        self.get_logger().info(
            f"DeltaMainApp ready  "
            f"Z_workspace=[{config.Z_MIN:.0f}, {config.Z_MAX:.0f}] mm  "
            f"place=({config.PLACE_X:.0f}, {config.PLACE_Y:.0f}, {config.PLACE_Z:.0f}) mm"
        )
        self._homing = True  # blocks pick queue until startup home completes
        self.get_logger().info("main_app node started, waiting for detections...")
        threading.Thread(target=self._startup_home, daemon=True).start()

    def _publish_ee_fk_timer(self) -> None:
        xyz = self._fsm._current_ee_xyz
        if xyz is None:
            return
        pt = PointStamped()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = "robot_base"
        pt.point.x = float(xyz[0])
        pt.point.y = float(xyz[1])
        pt.point.z = float(xyz[2])
        self._pub_ee_fk.publish(pt)

    def _publish_joint_states(self, fb_deg) -> None:
        """Publish full 12-joint /joint_states from the 3 live shoulder thetas."""
        thetas_rad = [math.radians(t) for t in fb_deg]
        values = joint_state(self._joint_geom, thetas_rad)
        if values is None:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = [values[name] for name in JOINT_NAMES]
        self._pub_joint_states.publish(msg)

    def _publish_feedforward_timer(self) -> None:
        now = self.get_clock().now().to_msg()
        tgt = self._fsm._target_xyz
        if tgt is not None:
            pt = PointStamped()
            pt.header.stamp = now
            pt.header.frame_id = "robot_base"
            pt.point.x = float(tgt[0])
            pt.point.y = float(tgt[1])
            pt.point.z = float(tgt[2])   # EE-tip Z in mm
            self._pub_ff_target.publish(pt)
        if config.ENABLE_MOTORS and self._ctrl.connected:
            try:
                fb_deg = self._ctrl.get_current_thetas_deg()
                self._publish_joint_states(fb_deg)
                ok, x, y, z = solve_fk_mm(*fb_deg)
                if ok:
                    pt = PointStamped()
                    pt.header.stamp = now
                    pt.header.frame_id = "robot_base"
                    pt.point.x = float(x)
                    pt.point.y = float(y)
                    pt.point.z = float(z) - config.EE_OFFSET_Z_MM  # platform → EE-tip
                    self._pub_ff_ee.publish(pt)
            except Exception as exc:
                self.get_logger().debug(f"FF EE publish error: {exc}")

    def _ee_error_callback(self, msg: PointStamped) -> None:
        alpha = config.EE_CORRECTION_ALPHA
        self._ee_error_x = alpha * msg.point.x + (1.0 - alpha) * self._ee_error_x
        self._ee_error_y = alpha * msg.point.y + (1.0 - alpha) * self._ee_error_y
        self._ee_error_count += 1
        self._fsm.update_ee_error(msg.point.x, msg.point.y)
        self.get_logger().info(
            f"EE error: dx={self._ee_error_x:+.1f}mm "
            f"dy={self._ee_error_y:+.1f}mm "
            f"(n={self._ee_error_count})"
        )

    def _on_laser_error(self, msg: PointStamped) -> None:
        # point.z carries the alignment tolerance (mm), not a Z coordinate —
        # see camera_system.py's laser_error_mm publisher.
        self._latest_laser_error = (msg.point.x, msg.point.y, msg.point.z, time.time())

    def _get_laser_error(self):
        """Return (dx_mm, dy_mm, tol_mm, age_s) from the latest
        /delta/laser_error_mm reading, or None if camera_system hasn't
        published one yet."""
        if self._latest_laser_error is None:
            return None
        dx, dy, tol_mm, stamp = self._latest_laser_error
        return dx, dy, tol_mm, time.time() - stamp

    def _all_targets_callback(self, msg: PoseArray) -> None:
        with self._queue_lock:
            candidates = [
                (p.position.x * 1000.0, p.position.y * 1000.0, p.position.z * 1000.0)
                for p in msg.poses
            ]
            if not candidates:
                return
            # Ascending X: most advanced on belt (closest to exit) first.
            candidates.sort(key=lambda p: p[0])
            now = time.time()

            # Always feed the predictor — regression window must stay populated.
            self._predictor.update_velocity(candidates[0][0], now)

            # Refresh stale queue positions with fresh detections.
            # Objects move in X only; match existing queue items by Y proximity (±25 mm)
            # and overwrite their X + timestamp so extrapolation stays accurate.
            queued_ys: list = []
            for i, (_, qy, _, _) in enumerate(self._target_queue):
                queued_ys.append(qy)
                for cx, cy, cz in candidates:
                    if abs(cy - qy) < 25.0:
                        self._target_queue[i] = (cx, cy, cz, now)
                        break

            if not self._target_queue:
                # Queue empty — build from fresh candidates only when IDLE.
                # While busy the current pick target may still be visible;
                # rebuilding the queue here would re-dispatch it.
                if self._fsm.state == "IDLE":
                    self._target_queue = [(x, y, z, now) for x, y, z in candidates]
                    self.get_logger().info(f"Queued {len(candidates)} object(s)")
                return

            # Queue has items; add any objects newly detected that aren't in it yet,
            # excluding the one currently being picked.
            current_y = self._fsm.current_pick_y
            for cx, cy, cz in candidates:
                if len(self._target_queue) >= 5:
                    break
                if current_y is not None and abs(cy - current_y) < 25.0:
                    continue  # this is the object currently being picked
                if all(abs(cy - qy) >= 25.0 for qy in queued_ys):
                    self._target_queue.append((cx, cy, cz, now))
                    queued_ys.append(cy)
                    self.get_logger().info(
                        f"New object added to queue: ({cx:.1f}, {cy:.1f})"
                    )

    def _velocity_callback(self, msg: PointStamped) -> None:
        self._belt_vx_mm_s = msg.point.x

    def _queue_timer(self) -> None:
        if self._homing:
            return
        with self._queue_lock:
            if self._fsm.state != "IDLE" or not self._target_queue:
                return

            now = time.time()
            vx = self._predictor.measured_vx
            if vx is None:
                vx = -config.CONVEYOR_BELT_SPEED_MM_S

            # Extrapolate every queued item's X to "now" using the belt velocity
            # estimate, then drop any that have already passed through the workspace.
            fresh = []
            for qx, qy, qz, qt in self._target_queue:
                x_now = qx + vx * (now - qt)
                if x_now < -(config.X_LIMIT + 20.0):
                    self.get_logger().warn(
                        f"Object Y={qy:.1f}mm exited workspace "
                        f"(x_extrap={x_now:.1f}mm) — dropped"
                    )
                    continue
                fresh.append((x_now, qy, qz, now))

            # Re-sort: most advanced (lowest X) first.
            fresh.sort(key=lambda p: p[0])
            self._target_queue = fresh

            if not self._target_queue:
                return

            x, y, z, _ = self._target_queue.pop(0)

            self.get_logger().info(
                f"QUEUE_TIMER: dispatching ({x:.1f}, {y:.1f}, {z:.1f})  "
                f"queue_remaining={len(self._target_queue)}"
            )

            result = self._predictor.predict(y, x)
            y_pick = y
            x_pick = result.x_pick
            if not result.valid:
                # Object is in approach zone — predicted pick still outside workspace.
                # Pre-position at workspace boundary now; arrival gate holds until object arrives.
                if x > config.X_LIMIT and abs(y) <= config.Y_LIMIT:
                    x_pick = config.X_LIMIT - 5.0
                    self.get_logger().info(
                        f"Approach zone: pre-positioning at X={x_pick:.1f}mm "
                        f"(object at X={x:.1f}, predicted X={result.x_pick:.1f})"
                    )
                else:
                    self.get_logger().warn(
                        f"Predicted pick ({result.x_pick:.1f}, {y:.1f}) outside workspace — skipped"
                    )
                    return
            elif not check_workspace(x_pick, y, z + config.EE_OFFSET_Z_MM):
                self.get_logger().warn(
                    f"Predicted pick ({x_pick:.1f}, {y:.1f}, {z:.1f}) outside workspace — skipped"
                )
                return
            self.get_logger().info(
                f"Pre-position target: ({x_pick:.1f}, {y_pick:.1f}, {z:.1f}) mm  "
                f"vx={result.vx_mm_s:.1f} mm/s  offset={result.belt_offset:.1f} mm  "
                f"t={result.t_total:.2f} s"
            )
            self._check_belt_verify()
            if not self._fsm.force_target(x_pick, y_pick, z):
                self._target_queue.insert(0, (x, y, z, now))
                self.get_logger().warn("force_target busy — item re-queued")

    def _status_callback(self, msg: String) -> None:
        if "NO_DETECTION" in msg.data or "LOST" in msg.data:
            self._fsm.no_detection()

    def _send_gripper(self, pos: float) -> None:
        m = Float32()
        m.data = float(pos)
        self._gripper_pub.publish(m)
        if config.ENABLE_MOTORS:
            if pos > 0.3:
                self._gripper.grip(wait=False)
            else:
                self._gripper.release(wait=False)

    def _send_state(self, state: str) -> None:
        m = String()
        m.data = state
        self._state_pub.publish(m)

    def verify_belt_speed(self) -> None:
        """
        Start a 10-second belt speed verification window.
        Ensure objects are on the belt before calling.
        After 10 s the measured velocity is compared to CONVEYOR_BELT_SPEED_MM_S
        and a warning is printed if the difference exceeds 1.0 mm/s.
        """
        self._predictor.clear()
        self._verify_deadline = time.time() + 10.0
        self.get_logger().info(
            "Belt speed verification started — keep belt running with objects. "
            "Result in 10 s..."
        )

    def _check_belt_verify(self) -> None:
        if self._verify_deadline == 0.0 or time.time() < self._verify_deadline:
            return
        self._verify_deadline = 0.0
        vx = self._predictor.measured_vx
        if vx is None:
            self.get_logger().warn(
                "Belt verify: not enough samples — no objects detected during window?"
            )
            return
        measured = abs(vx)
        expected = config.CONVEYOR_BELT_SPEED_MM_S
        diff = abs(measured - expected)
        status = "WARNING" if diff > 1.0 else "OK"
        msg = (
            f"\n===== BELT SPEED VERIFICATION =====\n"
            f"  Measured : {measured:.2f} mm/s\n"
            f"  Config   : {expected:.2f} mm/s\n"
            f"  Diff     : {diff:.2f} mm/s  [{status}]\n"
        )
        if diff > 1.0:
            msg += f"  → Update config.py: CONVEYOR_BELT_SPEED_MM_S = {measured:.2f}\n"
        msg += "==================================="
        if diff > 1.0:
            self.get_logger().warn(msg)
        else:
            self.get_logger().info(msg)

    def _peek_next_pick_xy(self):
        """Return (x_predicted, y) of the next queued object, or None.
        Called by the FSM during RESETTING so it can pre-position toward the
        next pick instead of returning to the centre home position."""
        with self._queue_lock:
            if not self._target_queue:
                return None
            qx, qy, _, qt = self._target_queue[0]
            vx = self._predictor.measured_vx or -config.CONVEYOR_BELT_SPEED_MM_S
            x_now = qx + vx * (time.time() - qt)
            result = self._predictor.predict(qy, x_now)
            # Clamp to workspace so the IK always succeeds for the pre-position move.
            y_pre = max(-config.Y_LIMIT + 5.0, min(config.Y_LIMIT - 5.0, qy))
            x_pre = max(-config.X_LIMIT + 5.0, min(config.X_LIMIT - 5.0, result.x_pick))
            return (x_pre, y_pre)

    def _startup_home(self) -> None:
        time.sleep(0.5)
        self.get_logger().info("Startup: opening gripper and homing to workspace center")
        self._send_gripper(0.0)          # open gripper
        try:
            if config.ENABLE_MOTORS:
                self._ctrl.move_thetas(0.0, 0.0, 0.0)   # exact θ=0,0,0 home
        except Exception as exc:
            self.get_logger().error(f"Startup home failed: {exc}")
        finally:
            self._homing = False         # always allow picks — better to try than freeze

    def destroy_node(self) -> None:
        self._homing = True              # stop new picks immediately
        if self._explicit_shutdown:
            self._send_gripper(0.0)      # open gripper (ROS + direct CAN)
            if config.ENABLE_MOTORS:
                try:
                    self._ctrl.move_thetas(0.0, 0.0, 0.0)   # exact θ=0,0,0 home
                    time.sleep(0.5)
                except Exception:
                    pass
                self._ctrl.shutdown()
                self._gripper.disconnect()
        super().destroy_node()


def main(args=None):
    import signal
    rclpy.init(args=args)
    node = DeltaMainApp()

    def handle_sigint(*_):
        node.get_logger().info("Ctrl+C received — shutting down cleanly")
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
