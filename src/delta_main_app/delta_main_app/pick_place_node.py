#!/usr/bin/env python3
"""
pick_place_node.py — Delta robot vision-guided pick-and-place with local IK.

Receives a confirmed target XYZ from the camera pipeline and solves inverse
kinematics locally (delta_common.fk_ik.solve_ik_mm) — no external solver
round-trip. The node executes the move, grips, then drives straight home (no
lift, no place stop) and releases there — either chaining straight into the
next queued pick or landing in IDLE.

━━━━ Topic map ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Camera → Node    (subscribe):
      /delta/target_xyz            geometry_msgs/PointStamped
          x/y/z in mm, robot base frame, EE-tip Z — a CONFIRMED, reachable
          target; triggers the real pick
      /delta/approaching_target    custom_messages/DeltaTarget
          a weed tracked but still outside the workspace, within the
          approach-zone band (delta_weed_bridge) — triggers a WAITING
          pre-position, not a pick

  Node → All       (publish):
      /delta/target_committed      custom_messages/DeltaTarget
          x_mm, y_mm, z_mm, confidence, track_id, detection_mode — telemetry
          for the target this node has committed to and is solving/executing
      /delta/bridge_state          std_msgs/String   — current FSM state
      /delta/fk_result             geometry_msgs/PointStamped — FK after move
      /delta/motor_thetas          custom_messages/DeltaJointAngles — live
          motor-encoder feedback, published as telemetry after every move

━━━━ State machine ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  IDLE → WAITING (pre-position) ─┬─→ MOVING(+grip) → LIFTING(+release) ─┬─→ MOVING (next pick, chained)
       ↘ MOVING(+grip) ──────────┘        ↘ ERROR ──────────────────────┴─→ WAITING (no next pick queued —
       (WAITING also times out back to IDLE if no weed arrives)              parks & waits for the next one)
  No place/home step: LIFTING only rises LIFT_AFTER_PICK_MM straight up from
  the pick point and releases there, then goes back to WAITING immediately.
"""

import collections
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSPresetProfiles, QoSProfile
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float32, String

from custom_messages.msg import DeltaTarget, DeltaJointAngles
from delta_common import config
from delta_common.fk_ik import JOINT_NAMES, DeltaGeometry, joint_state, solve_fk_mm, solve_ik_mm
from delta_main_app.belt_predictor import BeltPredictor
from delta_motor_controller.motor_controller import DeltaMotorController
from delta_motor_controller.pneumatic_gripper import PneumaticGripper

# Was -15.0 (forced extra descent below detected surface). Removed 2026-07-09 —
# with the Z-guard/floor check gone, that overrun went uncaught and picks were
# landing too deep. Target Z now matches the detected surface exactly.
Z_DROP_EXTRA_MM      = -25.0    # extra descent commanded on pick (deeper = more negative)

# No place step: after gripping, the FSM lifts straight up (same X/Y) by this
# much and releases there instead of returning all the way home. TUNE this to
# whatever clearance is needed to clear the belt/plants before the arm parks
# back at WAITING.
LIFT_AFTER_PICK_MM  = 50.0

# Safety: maximum depth the local IK solver's result may command beyond the
# detected target Z. If the FK of the solved thetas puts the EE-tip more than
# this below the detected surface, the move is rejected before any motors move.
# Includes headroom for the intentional Z_DROP_EXTRA_MM offset above, plus a
# 20mm anomaly margin for the solver's own correction.
IK_Z_GUARD_MM        = 20.0 + abs(Z_DROP_EXTRA_MM)
# Hard absolute floor for EE-tip Z (mm, robot base frame, negative = below base).
# Crash confirmed at EE-tip < -640 mm.  Set 5 mm above that as the kill limit.
# Checked on BOTH the incoming detected target AND the IK-commanded position.
EETIP_Z_FLOOR_MM     = -638.0

# ── Rigid-object Z estimation ─────────────────────────────────────────────────
# Pick targets here are rigid delta-robot objects (blocks/cubes on a belt), not
# vegetation: the object's own top surface facing the camera IS the reliable
# reading, unlike a plant top which is noisy/holed. So sample a small disk
# centred ON the object (not a ring around it) and take the median of its own
# returns. RIGID_Z_SAMPLE_R_PX must stay smaller than the smallest object's
# on-screen half-extent — a radius reaching past the object's edge starts
# pulling in the surrounding belt, which is deeper than the object's real top
# and commands the gripper straight through the object on descent.
RIGID_Z_SAMPLE_R_PX  = 8     # disk radius around centroid (px) — stay inside the object
RIGID_Z_PERCENTILE   = 50.0  # median of the object's own surface returns
Z_BG_REJECT_MM        = 50.0  # reject disk returns > this depth below the disk minimum
Z_MIN_VALID_PX        = 5     # need at least this many valid depth pixels
# Temporal: keep a per-target rolling Z buffer to suppress frame-to-frame noise.
# Buffer resets when the target XY jumps >50 mm (different plant).
Z_HISTORY_MAXLEN      = 6
Z_HISTORY_RESET_MM    = 50.0  # XY distance that triggers a buffer reset (mm)


# ── State machine ──────────────────────────────────────────────────────────────

class PickPlaceFSM:
    """
    Pure FSM — no ROS dependency; driven by the ROS node via public methods.

    States
    ------
    IDLE            : home pose (theta=0,0,0), gripper open, waiting for a target
    WAITING         : a weed is approaching but not yet reachable — parked at
                      the fixed pre-position pose (config.WAIT_X/Y/Z_MM) to
                      shorten the final travel once it actually arrives.
                      A confirmed on_target() while WAITING interrupts the
                      wait and goes straight to MOVING. Times out back to
                      IDLE after config.CONVEYOR_ARRIVAL_TIMEOUT_S with no
                      arrival (bg thread)
    MOVING          : executing the locally-solved IK thetas on the motors; grips
                      once settled (bg thread)
    LIFTING         : no place stop — rises LIFT_AFTER_PICK_MM straight up
                      from the pick point (same X/Y) with the object gripped,
                      releases once settled there, then either chains
                      straight into the next queued target (MOVING, no trip
                      through WAITING) or returns to the WAIT pose and
                      re-enters WAITING for the next weed (bg thread)
    ERROR           : any failure (invalid IK solution, unreachable thetas, or
                      a move that never settled) — forces the gripper open,
                      returns home, recovers to IDLE
    """

    def __init__(self, controller: DeltaMotorController, publish_state_fn, publish_fk_fn,
                 publish_motor_thetas_fn, publish_target_fn, logger,
                 gripper=None, publish_gripper_fn=None):
        self._ctrl              = controller
        self._pub_state         = publish_state_fn         # fn(state: str)
        self._pub_fk             = publish_fk_fn             # fn(x, y, z)
        self._pub_motor_thetas   = publish_motor_thetas_fn   # fn(fb_deg, ok)
        self._pub_target         = publish_target_fn         # fn(x_m, y_m, z_m) -> None, publishes committed-target telemetry
        self._gripper            = gripper                   # PneumaticGripper | None
        self._pub_gripper        = publish_gripper_fn        # fn(pos: float) -> None, pos: 1.0=grip 0.0=release
        self._log                = logger

        self._state  = "IDLE"
        self._busy   = False
        self._lock   = threading.Lock()

        self._target_xyz      = None   # (x, y, z) metres — target currently being worked
        self._target_detect_t = None   # time.time() when this target was received
        self._ik_thetas       = None   # (t1, t2, t3) deg
        self._pick_x_mm        = None   # commanded pick pose (platform frame), reused by
        self._pick_y_mm        = None   # LIFTING to rise straight up from the same X/Y
        self._pick_z_platform_mm = None
        self._pending_target  = None   # (x, y, z, detect_time) received while busy — PLACING
                                        # picks this up so the next pick chains immediately,
                                        # with no return-home in between.
        self._waiting_since   = None   # time.time() when WAITING was entered, for the timeout

    # ── public API ─────────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    @property
    def target_xyz(self):
        """(x, y, z) metres of the target currently committed to, or None."""
        with self._lock:
            return self._target_xyz

    def on_target(self, x: float, y: float, z: float, detect_time=None) -> bool:
        """Accept a new, confirmed-reachable target. If a pick is already in
        progress (MOVING/LIFTING), it's buffered as the next pick (NOT
        dropped) and picked up automatically once the current pick+lift
        cycle finishes. If WAITING (pre-positioned for an approaching weed,
        including the WAIT-pose park right after a LIFTING release),
        this interrupts the wait and starts the real pick immediately — that
        wait was for exactly this moment. Returns True only if this target
        was started immediately (from IDLE or WAITING)."""
        with self._lock:
            if self._busy and self._state != "WAITING":
                self._pending_target = (
                    x, y, z, detect_time if detect_time is not None else time.time()
                )
                return False
            self._commit_target(x, y, z, detect_time)
        self._pub_target(x, y, z)
        self._solve_ik_and_proceed(x, y, z)
        return True

    def on_approaching(self, x: float, y: float, z: float) -> None:
        """A weed is tracked but still outside the workspace, within the
        approach-zone band (see delta_weed_bridge). Pre-position at the fixed
        WAIT pose to shorten the final travel once it actually arrives.
        Only acts from IDLE — never interrupts an in-progress pick or an
        already-active wait. x/y/z are the approaching weed's position, used
        only for logging (the wait pose itself is fixed)."""
        with self._lock:
            if self._state != "IDLE":
                return
            self._busy = True
            self._waiting_since = time.time()
            self._set_state("WAITING")
        self._log.info(
            f"Weed approaching at ({x * 1000.0:.1f},{y * 1000.0:.1f},"
            f"{z * 1000.0:.1f}) mm — pre-positioning to WAIT pose"
        )
        threading.Thread(target=self._run_wait_move, daemon=True).start()

    def _solve_ik_and_proceed(self, x: float, y: float, z: float) -> None:
        """Compute joint angles locally via delta_common's solve_ik_mm and
        start the move — no external solver round-trip."""
        x_mm = x * 1000.0
        y_mm = y * 1000.0
        z_platform_mm = z * 1000.0 + config.EE_OFFSET_Z_MM + Z_DROP_EXTRA_MM
        self._pick_x_mm          = x_mm
        self._pick_y_mm          = y_mm
        self._pick_z_platform_mm = z_platform_mm
        ok_ik, t1, t2, t3 = solve_ik_mm(x_mm, y_mm, z_platform_mm)
        self._begin_move(t1, t2, t3, ok_ik)

    def _commit_target(self, x: float, y: float, z: float, detect_time=None) -> None:
        """Must be called with self._lock held. Commits (x,y,z) as the active
        target. Caller is responsible for publishing telemetry and solving IK
        (outside the lock)."""
        self._target_xyz      = (x, y, z)
        self._target_detect_t = detect_time if detect_time is not None else time.time()
        self._ik_thetas       = None
        self._busy            = True

    def _begin_move(self, t1: float, t2: float, t3: float, ik_valid: bool) -> None:
        """Start MOVING with the solved joint angles, or abort to ERROR if the
        IK solver found no valid solution."""
        if not ik_valid:
            self._log.error(
                f"IK solver reported no valid solution for target "
                f"({self._target_xyz[0]:.1f},{self._target_xyz[1]:.1f},"
                f"{self._target_xyz[2]:.1f}) — aborting"
            )
            self._go_error()
            return
        with self._lock:
            self._ik_thetas = (t1, t2, t3)
            self._set_state("MOVING")
        threading.Thread(target=self._run_move, daemon=True).start()

    # ── background threads ─────────────────────────────────────────────────────

    def _run_wait_move(self) -> None:
        """State WAITING: drive to the fixed pre-position pose and hold.
        config.WAIT_X/Y/Z_MM are already platform-frame (mirrors HOME_Z's
        convention), so no EE-tip-to-platform offset is needed before
        solve_ik_mm, unlike a camera-detected EE-tip target."""
        ok_ik, t1, t2, t3 = solve_ik_mm(config.WAIT_X_MM, config.WAIT_Y_MM, config.WAIT_Z_MM)
        if not ok_ik or not self._ctrl.within_joint_limits(t1, t2, t3):
            self._log.error(
                f"WAIT pose ({config.WAIT_X_MM:.1f},{config.WAIT_Y_MM:.1f},"
                f"{config.WAIT_Z_MM:.1f}) unreachable (ik_valid={ok_ik}) — aborting"
            )
            self._go_error()
            return

        ok, fk_xyz, err, fb_deg = self._ctrl.move_thetas(t1, t2, t3)
        if fk_xyz is not None:
            self._pub_fk(fk_xyz[0], fk_xyz[1], fk_xyz[2])
        self._pub_motor_thetas(fb_deg, ok)

        if not ok:
            self._log.warn(
                f"Move to WAIT pose did NOT settle (err={err:.2f} mm, above "
                f"{self._ctrl.POS_TOL_MM}mm tolerance) — aborting"
            )
            self._go_error()
            return

        self._log.info(
            f"At WAIT pose  FK=({fk_xyz[0]:.1f},{fk_xyz[1]:.1f},{fk_xyz[2]:.1f})"
            f"  err={err:.2f} mm — holding for a confirmed target"
        )

    def check_waiting_timeout(self) -> None:
        """Call periodically (node timer). If WAITING has held longer than
        config.CONVEYOR_ARRIVAL_TIMEOUT_S with no confirmed pick, give up and
        return home — this is an expected outcome (the weed never arrived
        in range, e.g. lost track), not a fault, so it does NOT go through
        ERROR."""
        timed_out = False
        with self._lock:
            if self._state != "WAITING":
                return
            if time.time() - self._waiting_since > config.CONVEYOR_ARRIVAL_TIMEOUT_S:
                timed_out = True
        if timed_out:
            self._log.info(
                f"WAITING timeout ({config.CONVEYOR_ARRIVAL_TIMEOUT_S}s) — "
                "no confirmed target arrived, returning home"
            )
            threading.Thread(target=self._return_home_from_waiting, daemon=True).start()

    def _return_home_from_waiting(self) -> None:
        """Background: drive home and land IDLE after a WAITING timeout.
        Re-checks state==WAITING both before and after the move — a real
        target may have arrived and taken over in the meantime (on_target()
        interrupts WAITING directly), in which case this must not stomp on
        the now-in-progress pick or issue a conflicting CAN command."""
        with self._lock:
            if self._state != "WAITING":
                return
        try:
            self._ctrl.move_thetas(0.0, 0.0, 0.0)
        except Exception as exc:
            self._log.error(f"Home move after WAITING timeout failed: {exc}")
        with self._lock:
            if self._state != "WAITING":
                return
            self._state          = "IDLE"
            self._busy            = False
            self._target_xyz      = None
            self._ik_thetas       = None
        self._pub_state("IDLE")
        self._log.info("[FSM] → IDLE (WAITING timeout, no weed arrived)")

    def _run_move(self) -> None:
        """State MOVING: drive to the IK-solved thetas, verify settle, then grip."""
        t1, t2, t3 = self._ik_thetas

        # Hard clamp: reject before any physical move if the detected target
        # itself is already beyond the crash floor.
        z_eetip_target_mm = self._target_xyz[2] * 1000.0
        if z_eetip_target_mm < EETIP_Z_FLOOR_MM:
            self._log.error(
                f"Target EE-tip={z_eetip_target_mm:.1f}mm < floor={EETIP_Z_FLOOR_MM}mm "
                "— aborting before move"
            )
            self._go_error()
            return

        self._log.info(
            f"Executing IK thetas: θ1={t1:.2f}° θ2={t2:.2f}° θ3={t3:.2f}°"
        )

        if not self._ctrl.within_joint_limits(t1, t2, t3):
            self._log.error(
                f"IK thetas outside joint limits "
                f"(limits θ_max={config.THETA1_MAX}°): "
                f"({t1:.1f},{t2:.1f},{t3:.1f}) — aborting"
            )
            self._go_error()
            return

        # ── Z guard ───────────────────────────────────────────────────────────
        # Compute FK of the solved thetas BEFORE sending to motors. If the
        # resulting EE-tip Z is below the absolute floor, or more than
        # IK_Z_GUARD_MM below the detected target surface, reject the move —
        # the solver's own correction must not drive the gripper through the
        # belt/terrain.
        ok_fk, _, _, z_platform_commanded = solve_fk_mm(t1, t2, t3)
        if not ok_fk:
            self._log.error(
                f"Z guard: FK of IK thetas ({t1:.1f},{t2:.1f},{t3:.1f})° "
                "returned no solution — aborting"
            )
            self._go_error()
            return
        z_eetip_commanded = z_platform_commanded - config.EE_OFFSET_Z_MM
        overrun = z_eetip_target_mm - z_eetip_commanded   # positive = commanded deeper
        self._log.info(
            f"Z guard: EE-tip commanded={z_eetip_commanded:.1f}mm  "
            f"target={z_eetip_target_mm:.1f}mm  overrun={overrun:+.1f}mm  "
            f"floor={EETIP_Z_FLOOR_MM}mm"
        )
        if z_eetip_commanded < EETIP_Z_FLOOR_MM:
            self._log.error(
                f"Z floor TRIP: IK commands EE-tip={z_eetip_commanded:.1f}mm "
                f"< floor={EETIP_Z_FLOOR_MM}mm — aborting"
            )
            self._go_error()
            return
        if overrun > IK_Z_GUARD_MM:
            self._log.error(
                f"Z guard TRIP: IK commands EE-tip {overrun:.1f}mm below "
                f"detected surface (guard={IK_Z_GUARD_MM}mm) — aborting"
            )
            self._go_error()
            return

        ok, fk_xyz, err, fb_deg = self._ctrl.move_thetas(t1, t2, t3)
        if fk_xyz is not None:
            self._pub_fk(fk_xyz[0], fk_xyz[1], fk_xyz[2])
        self._pub_motor_thetas(fb_deg, ok)

        if not ok:
            self._log.warn(
                f"Move did NOT settle (err={err:.2f} mm, above "
                f"{self._ctrl.POS_TOL_MM}mm tolerance) — aborting"
            )
            self._go_error()
            return

        self._log.info(
            f"Move OK  FK=({fk_xyz[0]:.1f},{fk_xyz[1]:.1f},{fk_xyz[2]:.1f})"
            f"  err={err:.2f} mm"
        )
        if self._target_detect_t is not None:
            latency_s = time.time() - self._target_detect_t
            self._log.info(
                f"Detection→arrival latency: {latency_s:.2f}s "
                f"(predict_target's t_robot_move should match this)"
            )

        if self._gripper is not None:
            self._log.info("Gripping")
            self._gripper.grip()
            if self._pub_gripper is not None:
                self._pub_gripper(1.0)

        with self._lock:
            self._set_state("LIFTING")
        self._run_lift_and_release()

    def _run_lift_and_release(self) -> None:
        """State LIFTING: no place stop — rise LIFT_AFTER_PICK_MM straight up
        from the pick point (same X/Y, platform frame) with the object
        gripped, release once settled there, then chain straight into the
        next queued target or return to the WAIT pose and re-enter WAITING
        for the next weed (never lands in plain IDLE on a normal cycle —
        only a WAITING timeout or ERROR recovery does that)."""
        lift_z_platform_mm = self._pick_z_platform_mm + LIFT_AFTER_PICK_MM
        ok_ik, t1, t2, t3 = solve_ik_mm(self._pick_x_mm, self._pick_y_mm, lift_z_platform_mm)
        if not ok_ik or not self._ctrl.within_joint_limits(t1, t2, t3):
            self._log.error(
                f"Lift pose ({self._pick_x_mm:.1f},{self._pick_y_mm:.1f},"
                f"{lift_z_platform_mm:.1f}) unreachable (ik_valid={ok_ik}) — aborting"
            )
            self._go_error()
            return

        ok, fk_xyz, err, fb_deg = self._ctrl.move_thetas(t1, t2, t3)
        if fk_xyz is not None:
            self._pub_fk(fk_xyz[0], fk_xyz[1], fk_xyz[2])
        self._pub_motor_thetas(fb_deg, ok)

        if not ok:
            self._log.warn(
                f"Lift move did NOT settle (err={err:.2f} mm, above "
                f"{self._ctrl.POS_TOL_MM}mm tolerance) — releasing anyway"
            )
        else:
            self._log.info(
                f"Lifted  FK=({fk_xyz[0]:.1f},{fk_xyz[1]:.1f},{fk_xyz[2]:.1f})"
                f"  err={err:.2f} mm"
            )

        if self._gripper is not None:
            self._log.info("Releasing")
            self._gripper.release()
            if self._pub_gripper is not None:
                self._pub_gripper(0.0)

        with self._lock:
            pending = self._pending_target
            self._pending_target = None
            if pending is not None:
                x, y, z, detect_time = pending
                self._commit_target(x, y, z, detect_time)

        if pending is not None:
            x, y, z, _ = pending
            self._pub_target(x, y, z)
            self._solve_ik_and_proceed(x, y, z)
            return   # already moving on the next target

        # No next pick queued — go straight back to the WAIT pose instead of
        # a plain home/IDLE, since the belt keeps feeding weeds. Re-uses the
        # same WAITING state/timeout as the approach-zone pre-position.
        with self._lock:
            self._waiting_since = time.time()
            self._set_state("WAITING")
        self._run_wait_move()

    def _go_error(self) -> None:
        with self._lock:
            self._set_state("ERROR")
        threading.Thread(target=self._recover, daemon=True).start()

    def _recover(self) -> None:
        """Force the gripper open, return home, and land in IDLE — the single
        shared recovery path for every failure mode (invalid/unreachable IK
        solution, or a move that never settled)."""
        if self._gripper is not None:
            self._log.info("ERROR recovery — forcing gripper open")
            try:
                self._gripper.release()
                if self._pub_gripper is not None:
                    self._pub_gripper(0.0)
            except Exception as exc:
                self._log.error(f"Gripper release during recovery failed: {exc}")
        try:
            self._ctrl.move_thetas(0.0, 0.0, 0.0)   # exact θ=0,0,0 home
        except Exception as exc:
            self._log.error(f"Home move during recovery failed: {exc}")
        with self._lock:
            self._state           = "IDLE"
            self._busy             = False
            self._target_xyz       = None
            self._ik_thetas         = None
            self._pending_target   = None
        self._pub_state("IDLE")
        self._log.info("[FSM] → IDLE (recovered from ERROR)")

    def _set_state(self, state: str) -> None:
        self._state = state
        self._pub_state(state)
        self._log.info(f"[FSM] → {state}")


# ── ROS2 node ──────────────────────────────────────────────────────────────────

class PickPlaceNode(Node):
    """
    Vision-guided pick-and-place FSM for the delta robot — commands the
    physical motors/gripper directly from locally-solved IK.

    Run alongside the camera node. Should be the only node commanding the
    motors at a time (they share can1).
    """

    def __init__(self):
        super().__init__("pick_place_node")

        # ── motor controller ──────────────────────────────────────────────────
        self._ctrl = DeltaMotorController(
            can_port="can1",
            vel_max=config.MOTOR_VEL_MAX,
            acc_set=config.MOTOR_ACC_SET,
        )
        self._explicit_shutdown = False   # True only on intentional Ctrl+C
        if config.ENABLE_MOTORS:
            self._ctrl.connect()
            self.get_logger().info("Motors connected.")
        else:
            self.get_logger().warn("ENABLE_MOTORS=False — dry-run mode")

        # ── gripper (own bus, can2 — motors are on can1, same as main_app.py) ────
        self._gripper = PneumaticGripper(can_channel='can1', can_id=4)
        if config.ENABLE_MOTORS:
            self._gripper.connect()
            self.get_logger().info("Gripper connected (direct CAN).")

        # ── publishers ────────────────────────────────────────────────────────
        self._pub_target_committed = self.create_publisher(
            DeltaTarget, "/delta/target_committed", 100
        )
        self._pub_state = self.create_publisher(
            String, "/delta/bridge_state", 10
        )
        # Gripper state — [0=open, 1=closed], mirrors main_app.py's topic so
        # `ros2 topic echo /delta/gripper_cmd` works regardless of which node
        # is running.  TRANSIENT_LOCAL (latched): a subscriber that starts
        # after this node — e.g. `ros2 topic echo` run mid-session — still
        # immediately gets the last known state instead of waiting for the
        # next grip/release event.
        self._pub_gripper_cmd = self.create_publisher(
            Float32, "/delta/gripper_cmd",
            QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._send_gripper(0.0)   # connect() above already released — reflect it immediately
        self._pub_fk = self.create_publisher(
            PointStamped, "/delta/fk_result", 10
        )
        # Actual motor-encoder thetas (CAN feedback), read back after every
        # move and published as telemetry.
        self._pub_motor_thetas = self.create_publisher(
            DeltaJointAngles, "/delta/motor_thetas", 10
        )
        # Shared EE-tip FK — read by camera_system
        self._pub_ee_fk = self.create_publisher(
            PointStamped, "/delta/ee_fk_xyz", 10
        )
        # EE position telemetry — belt-prediction/logging consumers subscribe here.
        self._pub_ee_pos_mm = self.create_publisher(
            PointStamped, "/delta/ee_position_mm", 10
        )
        # Real motor-encoder joint state — lets robot_state_publisher/RViz
        # animate the URDF with the real robot's live pose.
        self._joint_geom = DeltaGeometry()
        self._pub_joint_states = self.create_publisher(JointState, "joint_states", 10)

        # ── FSM ───────────────────────────────────────────────────────────────
        self._fsm = PickPlaceFSM(
            controller=self._ctrl,
            publish_state_fn=self._send_state,
            publish_fk_fn=self._send_fk,
            publish_motor_thetas_fn=self._send_motor_thetas,
            publish_target_fn=self._publish_target_committed,
            logger=self.get_logger(),
            gripper=self._gripper,
            publish_gripper_fn=self._send_gripper,
        )

        # ── depth camera (unused: no depth stream, FAKE_DEPTH_ENABLE=True) ──────
        # Camera intrinsics (filled on first CameraInfo message)
        self._fx = self._fy = self._cx = self._cy = None

        # Pre-build camera transforms from config so we can reverse-project
        # base-frame coords back to pixel and then re-deproject with real depth.
        R = np.array(config.CAMERA_DIRECT_MATRIX, dtype=np.float64)
        t = np.array([config.CAM_TX_MM, config.CAM_TY_MM, config.CAM_TZ_MM],
                     dtype=np.float64)
        self._T_cam_to_base = np.eye(4, dtype=np.float64)
        self._T_cam_to_base[:3, :3] = R
        self._T_cam_to_base[:3, 3] = t
        self._T_base_to_cam = np.eye(4, dtype=np.float64)
        self._T_base_to_cam[:3, :3] = R.T
        self._T_base_to_cam[:3, 3] = -R.T @ t

        self._depth_lock = threading.Lock()
        self._depth_image = None   # H×W uint16, each count = 1 mm

        # Disk mask geometry is fixed (only depends on the constant radius), so
        # build it once here instead of rebuilding the rows/cols/sqrt distance
        # grid on every _reproject_with_real_depth call.
        _r = RIGID_Z_SAMPLE_R_PX
        _rows = np.arange(-_r, _r + 1)[:, None]
        _cols = np.arange(-_r, _r + 1)[None, :]
        _dist = np.sqrt(_rows ** 2 + _cols ** 2)
        self._disk_mask_full = _dist <= RIGID_Z_SAMPLE_R_PX

        # ── Terrain Z temporal smoother ───────────────────────────────────────
        # Accumulates per-target Z readings and outputs a rolling median to
        # suppress frame-to-frame depth noise over vegetation.  Resets when the
        # detected XY jumps > Z_HISTORY_RESET_MM (different plant).
        self._z_history: collections.deque = collections.deque(maxlen=Z_HISTORY_MAXLEN)
        self._z_history_xy: tuple | None = None   # (x_m, y_m) of buffered readings

        # ── belt velocity / pick-point prediction ───────────────────────────────
        self._predictor = BeltPredictor()
        self._last_robot_x_mm = config.HOME_X   # updated on every FK publish

        # ── subscribers ───────────────────────────────────────────────────────
        self._sub_target = self.create_subscription(
            PointStamped,
            "/delta/target_xyz",
            self._on_target_xyz,
            10,
        )
        self._sub_approaching = self.create_subscription(
            DeltaTarget,
            "/delta/approaching_target",
            self._on_approaching_target,
            10,
        )
        self._sub_depth = self.create_subscription(
            Image,
            config.DEPTH_TOPIC,
            self._on_depth_image,
            QoSPresetProfiles.SENSOR_DATA.value,
        )
        self._sub_cam_info = self.create_subscription(
            CameraInfo,
            config.CAMERA_INFO_TOPIC,
            self._on_camera_info,
            1,
        )

        # ── 5 Hz FK heartbeat — keeps camera_system's ee_fk_xyz fresh ─────────
        self.create_timer(0.2, self._publish_fk_heartbeat)

        self.get_logger().info(
            "PickPlaceNode ready\n"
            "  Listening : /delta/target_xyz, /delta/approaching_target\n"
            "  State     : /delta/bridge_state\n"
            "  IK solved locally — no external solver round-trip"
        )

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_target_xyz(self, msg: PointStamped) -> None:
        detect_time = time.time()
        x = msg.point.x   # metres, robot base frame
        y = msg.point.y
        z = msg.point.z

        # Replace z with real depth-camera depth when available — except in
        # weed_seg mode, where /delta/target_xyz's z already comes from
        # plant_perception's MAD-filtered root-depth estimate (robust for a
        # noisy/holed weed root, unlike this rigid-disk method which assumes
        # a flat cube top). Re-sampling at a re-projected pixel here would be
        # redundant and can disagree with the better upstream estimate.
        # Also skipped entirely when FAKE_DEPTH_ENABLE is on — re-sampling
        # real depth here would overwrite the fake z upstream already set.
        skip_real_depth = config.FAKE_DEPTH_ENABLE or config.DETECTION_MODE == "weed_seg"
        real = None if skip_real_depth else self._reproject_with_real_depth(x, y, z)
        if real is not None:
            x_real, y_real, z_real = real
            self.get_logger().info(
                f"Depth camera: z_fake={z * 1000.0:.1f} mm → "
                f"z_real={z_real * 1000.0:.1f} mm  "
                f"(Δ={abs(z_real - z) * 1000.0:.1f} mm)"
            )
            x, y, z = x_real, y_real, z_real
        elif not skip_real_depth:
            self.get_logger().warn(
                "Depth camera unavailable — using z from /delta/target_xyz "
                f"(z={z * 1000.0:.1f} mm)"
            )

        # Terrain temporal smoothing — median over recent Z readings for this
        # plant location.  Suppresses frame-to-frame depth flicker over vegetation.
        z_raw_mm = z * 1000.0
        z = self._smooth_z(x, y, z)
        z_smooth_mm = z * 1000.0
        if abs(z_smooth_mm - z_raw_mm) > 1.0:
            self.get_logger().info(
                f"Z smoothed: raw={z_raw_mm:.1f} mm → "
                f"smooth={z_smooth_mm:.1f} mm  "
                f"(buf={len(self._z_history)})"
            )

        # Belt-velocity prediction: replace raw detected X with the predicted
        # pick-time X, accounting for belt motion during the robot's
        # travel+descend time. Feed the regression on every detection so the
        # window stays populated even while busy (main_app.py's same pattern).
        if config.BELT_PREDICTION_ENABLE:
            x_mm_raw = x * 1000.0
            self._predictor.update_velocity(x_mm_raw, detect_time)
            pred = self._predictor.predict(y * 1000.0, x_mm_raw, self._last_robot_x_mm)
            if pred.valid:
                self.get_logger().info(
                    f"Belt predict: x_detected={x_mm_raw:.1f}mm → x_pick={pred.x_pick:.1f}mm "
                    f"(vx={pred.vx_mm_s:.1f}mm/s, t_total={pred.t_total:.2f}s, "
                    f"belt_offset={pred.belt_offset:.1f}mm)"
                )
                x = pred.x_pick / 1000.0
            else:
                self.get_logger().debug(
                    f"Belt predict: invalid (x_pick={pred.x_pick:.1f}mm outside workspace, "
                    f"vx={pred.vx_mm_s:.1f}mm/s, t_total={pred.t_total:.2f}s, "
                    f"belt_offset={pred.belt_offset:.1f}mm) — using raw detected X"
                )

        if not self._fsm.on_target(x, y, z, detect_time=detect_time):
            self.get_logger().info(
                f"Target ({x * 1000.0:.1f},{y * 1000.0:.1f},{z * 1000.0:.1f}) mm "
                f"queued as next pick (state={self._fsm.state})"
            )

    def _on_approaching_target(self, msg: DeltaTarget) -> None:
        """A weed is tracked but still outside the workspace, within the
        approach-zone band (delta_weed_bridge). Positions in mm already —
        DeltaTarget doesn't need the depth-reprojection/smoothing/belt-predict
        pipeline _on_target_xyz uses, since this isn't a pick commitment."""
        self._fsm.on_approaching(msg.x_mm / 1000.0, msg.y_mm / 1000.0, msg.z_mm / 1000.0)

    def _publish_target_committed(self, x_m: float, y_m: float, z_m: float) -> None:
        """Convert EE-tip metres to platform-frame mm and publish DeltaTarget
        telemetry for the target this node has committed to and is solving/
        executing. Called by the FSM both for a freshly-accepted target and
        for one chained straight from PLACING into the next pick."""
        x_mm = float(x_m) * 1000.0
        y_mm = float(y_m) * 1000.0
        z_eetip_mm = float(z_m) * 1000.0
        z_platform_mm = (
            z_eetip_mm + config.EE_OFFSET_Z_MM + Z_DROP_EXTRA_MM
        )  # e.g. -670 + 150 - 15 = -535

        self.get_logger().info(
            f"z_eetip={z_eetip_mm:.1f} mm → z_platform={z_platform_mm:.1f} mm "
            f"(incl. {Z_DROP_EXTRA_MM:+.1f}mm extra drop)"
        )

        out = DeltaTarget()
        out.header.stamp    = self.get_clock().now().to_msg()
        out.header.frame_id = "robot_base"
        out.x_mm            = x_mm
        out.y_mm            = y_mm
        out.z_mm            = z_platform_mm   # platform Z — what the IK solver used
        out.confidence      = -1.0
        out.track_id        = -1
        out.detection_mode  = config.DETECTION_MODE
        self._pub_target_committed.publish(out)

    # ── depth camera callbacks (unused — no depth stream) ───────────────────

    def _on_depth_image(self, msg: Image) -> None:
        # msg.data (array.array) already supports the buffer protocol —
        # wrapping it in bytes() forced a full HxW copy on every depth frame.
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        with self._depth_lock:
            self._depth_image = arr

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self._fx is None:
            self._fx = msg.k[0]
            self._fy = msg.k[4]
            self._cx = msg.k[2]
            self._cy = msg.k[5]
            self.get_logger().info(
                f"Camera intrinsics: fx={self._fx:.1f} fy={self._fy:.1f} "
                f"cx={self._cx:.1f} cy={self._cy:.1f}"
            )

    def _smooth_z(self, x_m: float, y_m: float, z_m: float) -> float:
        """Rolling median filter on Z — resets when XY position jumps to a new plant."""
        reset = False
        if self._z_history_xy is not None:
            dx = (x_m - self._z_history_xy[0]) * 1000.0
            dy = (y_m - self._z_history_xy[1]) * 1000.0
            if math.sqrt(dx ** 2 + dy ** 2) > Z_HISTORY_RESET_MM:
                self._z_history.clear()
                reset = True
        self._z_history_xy = (x_m, y_m)
        self._z_history.append(z_m)
        if reset:
            self.get_logger().debug("Z history reset — new plant location")
        return float(np.median(self._z_history))

    def _reproject_with_real_depth(self, x_m: float, y_m: float, z_m: float):
        """Replace z using real depth-camera depth.

        Projects base-frame XYZ to an image pixel, samples depth in a small disk
        centred on the object (its own top surface, not the surrounding belt),
        then re-deprojects to base frame.

        Returns (x_m, y_m, z_real_m) or None if depth is unavailable.
        """
        if self._fx is None:
            return None

        with self._depth_lock:
            depth_img = self._depth_image
        if depth_img is None:
            self.get_logger().warn("Depth image not yet received")
            return None

        # metres → mm, then base frame → camera frame
        p_b = np.array([x_m * 1000.0, y_m * 1000.0, z_m * 1000.0, 1.0])
        p_c = self._T_base_to_cam @ p_b
        xc, yc, zc = p_c[:3]
        if zc <= 1.0:
            return None

        u = int(round(self._fx * xc / zc + self._cx))
        v = int(round(self._fy * yc / zc + self._cy))

        h, w = depth_img.shape
        if not (0 <= u < w and 0 <= v < h):
            self.get_logger().warn(
                f"Projected pixel ({u},{v}) outside depth image ({w}×{h})"
            )
            return None

        # ── Rigid-object depth sampling ──────────────────────────────────────
        # The object's own top surface is the reliable reading here (unlike
        # vegetation, there's no depth hole over a rigid block). Strategy:
        #   1. Sample a small disk centred on the centroid, radius
        #      RIGID_Z_SAMPLE_R_PX — must stay inside the object's own
        #      footprint so it doesn't pick up the surrounding belt.
        #   2. Take the median (RIGID_Z_PERCENTILE) of that disk's returns.
        #   3. Fall back to full-patch median only if the disk is empty
        #      (e.g. object too close to image border).
        z_cam_mm = None
        r = RIGID_Z_SAMPLE_R_PX

        y0 = max(0, v - r);  y1 = min(h, v + r + 1)
        x0 = max(0, u - r);  x1 = min(w, u + r + 1)
        patch = depth_img[y0:y1, x0:x1].astype(np.float32)

        # Slice the precomputed disk mask by the same offsets used to clip
        # the patch — avoids rebuilding the rows/cols/sqrt distance grid here
        # on every call (the disk geometry never changes, only where it gets
        # cropped near image borders).
        disk_mask = self._disk_mask_full[
            y0 - v + r : y1 - v + r,
            x0 - u + r : x1 - u + r,
        ]

        # Boolean-mask indexing already returns a flat 1-D copy, so no extra
        # .flatten() needed.
        disk_flat  = patch[disk_mask]
        valid_disk = disk_flat[(disk_flat > 1.0) & (disk_flat < 10_000.0)]

        if len(valid_disk) >= Z_MIN_VALID_PX:
            # Reject background: values more than Z_BG_REJECT_MM beyond the
            # shallowest disk return are likely through-surface or room clutter.
            fg_disk = valid_disk[valid_disk <= valid_disk.min() + Z_BG_REJECT_MM]
            use = fg_disk if len(fg_disk) >= Z_MIN_VALID_PX else valid_disk
            z_cam_mm = float(np.percentile(use, RIGID_Z_PERCENTILE))
            self.get_logger().debug(
                f"Object Z (disk, n={len(use)}/{len(valid_disk)}): "
                f"{z_cam_mm:.1f} mm"
            )
        else:
            # Disk is empty or too sparse — fall back to full-patch median
            # (may happen near image borders).
            full_flat  = patch.flatten()
            valid_full = full_flat[(full_flat > 1.0) & (full_flat < 10_000.0)]
            if len(valid_full) >= Z_MIN_VALID_PX:
                fg_full = valid_full[valid_full <= valid_full.min() + Z_BG_REJECT_MM]
                use = fg_full if len(fg_full) >= Z_MIN_VALID_PX else valid_full
                z_cam_mm = float(np.percentile(use, RIGID_Z_PERCENTILE))
                self.get_logger().debug(
                    f"Object Z (full patch fallback, n={len(use)}): "
                    f"{z_cam_mm:.1f} mm"
                )
            else:
                self.get_logger().warn(
                    f"No valid depth at pixel ({u},{v}) — "
                    f"disk={len(valid_disk)} px, full={len(valid_full)} px"
                )
                return None

        x_cam_mm = (u - self._cx) * z_cam_mm / self._fx
        y_cam_mm = (v - self._cy) * z_cam_mm / self._fy

        p_real = self._T_cam_to_base @ np.array([x_cam_mm, y_cam_mm, z_cam_mm, 1.0])
        return p_real[0] / 1000.0, p_real[1] / 1000.0, p_real[2] / 1000.0

    # ── publish helpers ───────────────────────────────────────────────────────

    def _send_state(self, state: str) -> None:
        m = String()
        m.data = state
        self._pub_state.publish(m)

    def _send_gripper(self, pos: float) -> None:
        m = Float32()
        m.data = float(pos)
        self._pub_gripper_cmd.publish(m)

    def _send_motor_thetas(self, fb_deg, ok: bool) -> None:
        """Publish actual motor-encoder thetas (CAN feedback) as telemetry."""
        if fb_deg is None:
            return
        m = DeltaJointAngles()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = "robot_base"
        m.theta1_deg = float(fb_deg[0])
        m.theta2_deg = float(fb_deg[1])
        m.theta3_deg = float(fb_deg[2])
        m.ik_valid   = bool(ok)
        self._pub_motor_thetas.publish(m)
        self._publish_joint_states(fb_deg)

    def _send_fk(self, x: float, y: float, z: float) -> None:
        """x, y, z are PLATFORM coordinates from solve_fk_mm."""
        self._last_robot_x_mm = float(x)
        pt = PointStamped()
        pt.header.stamp    = self.get_clock().now().to_msg()
        pt.header.frame_id = "robot_base"
        pt.point.x = float(x)
        pt.point.y = float(y)
        pt.point.z = float(z)
        self._pub_fk.publish(pt)
        # Camera system expects EE-TIP Z on /delta/ee_fk_xyz (tip = platform - offset)
        pt_ee = PointStamped()
        pt_ee.header.stamp    = pt.header.stamp
        pt_ee.header.frame_id = "robot_base"
        pt_ee.point.x = float(x)
        pt_ee.point.y = float(y)
        pt_ee.point.z = float(z) - config.EE_OFFSET_Z_MM
        self._pub_ee_fk.publish(pt_ee)

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

    def _publish_fk_heartbeat(self) -> None:
        """Refresh /delta/ee_fk_xyz, /delta/ee_position_mm and /joint_states from live CAN feedback.
        Also the WAITING-timeout tick — checked unconditionally (even in
        dry-run/not-connected) since it's just an FSM state transition, not
        a CAN read."""
        self._fsm.check_waiting_timeout()
        if not config.ENABLE_MOTORS or not self._ctrl.connected:
            return
        # _run_move/_run_lift_and_release drive the same CAN bus from a background thread
        # while busy; polling here concurrently can steal its response frame
        # and leave that thread blocked forever on a CAN read with no timeout
        # (bus.read() -> receive() has none). Those paths already publish
        # fresh FK after every move, so skip while busy.
        if self._fsm.state != "IDLE":
            return
        try:
            fb_deg = self._ctrl.get_current_thetas_deg()
            self._publish_joint_states(fb_deg)
            ok, x, y, z = solve_fk_mm(*fb_deg)   # platform mm
            if ok:
                self._send_fk(x, y, z)
                # /delta/ee_position_mm — EE-tip coordinates in mm, telemetry.
                pt = PointStamped()
                pt.header.stamp    = self.get_clock().now().to_msg()
                pt.header.frame_id = "robot_base"
                pt.point.x = float(x)
                pt.point.y = float(y)
                pt.point.z = float(z) - config.EE_OFFSET_Z_MM   # platform → EE-tip
                self._pub_ee_pos_mm.publish(pt)
        except Exception as exc:
            self.get_logger().debug(f"FK heartbeat CAN error: {exc}")

    def destroy_node(self) -> None:
        if config.ENABLE_MOTORS and self._explicit_shutdown:
            self._gripper.release(wait=False)
            try:
                self._ctrl.move_thetas(0.0, 0.0, 0.0)   # exact θ=0,0,0 home
                time.sleep(0.5)
            except Exception:
                pass
            self._ctrl.shutdown()
            self._gripper.disconnect()
        super().destroy_node()


# ── entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    import signal
    rclpy.init(args=args)
    node = PickPlaceNode()

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
