#!/usr/bin/env python3
"""
laser_teleop_node.py — point a handheld laser pointer at a spot in the
workspace; the robot's end-effector moves there.

Detects the laser dot in the color feed on every frame (stateless — no
ROI-lock like delta_common.ee_marker.EEMarkerDetector, which can get stuck
tracking a false positive once its search window drifts off the real dot;
here we always score the whole frame fresh, which is what following a
freely-moving handheld pointer needs). Deprojects the dot to base-frame XYZ
using the calibrated camera transform (fixed FAKE_DEPTH_M, same convention as
the rest of this depth-less-camera pipeline — see delta_common/config.py).

A move only fires once the dot has held steady (within STABLE_RADIUS_PX) for
STABLE_HOLD_S seconds — this "point and hold" gesture is what triggers
motion, so a hand sweeping the pointer across the frame doesn't make the
robot chase it.

Run this ALONGSIDE a node that publishes the color/camera_info topics
(delta_camera_system's camera_system.launch.py, or delta_weed_bridge's
weed_bridge.launch.py — either works, this node only subscribes). Do NOT run
this at the same time as kinematics_test_node or pick_place_node (or
anything else that opens its own DeltaMotorController) — only one process
may hold the CAN connection to the motors at a time.
"""

import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from delta_common import config, camera_geometry
from delta_motor_controller.motor_controller import DeltaMotorController

STABLE_RADIUS_PX = 6.0      # dot must stay within this radius to count as "held"
STABLE_HOLD_S = 0.6         # how long it must hold before a move fires
MOVE_COOLDOWN_S = 1.5       # minimum time between moves
MOVE_REPEAT_MIN_MM = 5.0    # ignore a new stable point this close to the last commanded one


class LaserTeleopNode(Node):
    def __init__(self):
        super().__init__("laser_teleop_node")

        self._ctrl = DeltaMotorController(
            can_port="can1",
            vel_max=config.MOTOR_VEL_MAX,
            acc_set=config.MOTOR_ACC_SET,
        )
        self._explicit_shutdown = False

        if config.ENABLE_MOTORS:
            self._ctrl.connect()
            self.get_logger().info("Motors connected (can1) — homed to (0,0,0).")
        else:
            self.get_logger().warn("ENABLE_MOTORS=False — dry-run mode, no CAN writes")

        self.bridge = CvBridge()
        self.T_cam_to_base, _ = camera_geometry.build_T_cam_to_base()
        self._fx = self._fy = self._cx = self._cy = None

        self._history = deque()  # (t, u, v)
        self._last_move_time = 0.0
        self._last_target_mm = None  # (x, y, z) of last commanded move
        self._move_lock = threading.Lock()
        self._busy = False

        best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(CameraInfo, config.CAMERA_INFO_TOPIC, self._on_camera_info, 1)
        self.create_subscription(Image, config.COLOR_TOPIC, self._on_color, best_effort)

        self.get_logger().info(
            "laser_teleop_node ready — point the laser and hold it steady "
            f"for {STABLE_HOLD_S:.1f}s to move the robot there."
        )

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

    def _detect_laser_dot(self, frame):
        """Stateless single-frame laser-dot detector — full-frame search every
        call, best circularity*brightness candidate wins. See module docstring
        for why this doesn't reuse EEMarkerDetector's stateful ROI-lock."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat_min = int(config.EE_LASER_SAT_MIN)
        val_min = int(config.EE_LASER_VAL_MIN)
        max_area = int(config.EE_LASER_MAX_AREA)

        mask1 = cv2.inRange(hsv,
            (config.EE_LASER_HUE_LOW1, sat_min, val_min),
            (config.EE_LASER_HUE_HIGH1, 255, 255))
        mask2 = cv2.inRange(hsv,
            (config.EE_LASER_HUE_LOW2, sat_min, val_min),
            (config.EE_LASER_HUE_HIGH2, 255, 255))
        mask_core = cv2.inRange(hsv,
            (0, 0, int(config.EE_LASER_CORE_VAL_MIN)),
            (180, int(config.EE_LASER_CORE_SAT_MAX), 255))
        mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask_core)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        v_chan = hsv[:, :, 2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_c, best_score = None, -1.0
        for c in contours:
            area = float(cv2.contourArea(c))
            if not (config.EE_LASER_MIN_AREA <= area <= max_area):
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter < 1.0:
                continue
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            if circularity < 0.35:
                continue
            (_, _), (bw, bh), _ = cv2.minAreaRect(c)
            if bw > 0 and bh > 0 and max(bw, bh) / min(bw, bh) > 2.5:
                continue

            tmp_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(tmp_mask, [c], -1, 255, cv2.FILLED)
            mean_v = float(cv2.mean(v_chan, mask=tmp_mask)[0])
            score = mean_v * circularity
            if score > best_score:
                best_score = score
                best_c = c

        if best_c is None:
            return None
        M = cv2.moments(best_c)
        if M["m00"] == 0:
            return None
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    def _on_color(self, msg: Image) -> None:
        if self._fx is None:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Color decode failed: {exc}")
            return

        uv = self._detect_laser_dot(frame)
        now = time.time()
        if uv is not None:
            self._history.append((now, uv[0], uv[1]))
        while self._history and now - self._history[0][0] > STABLE_HOLD_S:
            self._history.popleft()

        stable_uv = self._check_stable(now)
        if stable_uv is None:
            return
        if self._busy:
            return
        if now - self._last_move_time < MOVE_COOLDOWN_S:
            return

        u, v = stable_uv
        target = self._deproject(u, v)
        if target is None:
            return
        x_base, y_base, z_base = target

        if self._last_target_mm is not None:
            lx, ly, lz = self._last_target_mm
            if ((x_base - lx) ** 2 + (y_base - ly) ** 2) ** 0.5 < MOVE_REPEAT_MIN_MM:
                return

        self._last_move_time = now
        self._last_target_mm = (x_base, y_base, z_base)
        self.get_logger().info(
            f"Laser held at uv=({u},{v}) -> base=({x_base:.1f},{y_base:.1f},{z_base:.1f}) mm — moving"
        )
        with self._move_lock:
            self._busy = True
        threading.Thread(target=self._execute_move, args=(x_base, y_base, z_base), daemon=True).start()

    def _check_stable(self, now: float):
        """Return the mean (u,v) if every point in the trailing STABLE_HOLD_S
        window is within STABLE_RADIUS_PX of that mean, else None."""
        if not self._history:
            return None
        span = now - self._history[0][0]
        if span < STABLE_HOLD_S or len(self._history) < 3:
            return None
        us = [p[1] for p in self._history]
        vs = [p[2] for p in self._history]
        mu, mv = sum(us) / len(us), sum(vs) / len(vs)
        for u, v in zip(us, vs):
            if ((u - mu) ** 2 + (v - mv) ** 2) ** 0.5 > STABLE_RADIUS_PX:
                return None
        return mu, mv

    def _deproject(self, u: float, v: float):
        z_m = config.FAKE_DEPTH_M
        x_cam, y_cam, z_cam = camera_geometry.pixel_to_camera_xyz_mm(
            self._fx, self._fy, self._cx, self._cy, u, v, z_m
        )
        return camera_geometry.camera_to_base_mm(self.T_cam_to_base, x_cam, y_cam, z_cam)

    def _execute_move(self, x: float, y: float, z: float) -> None:
        try:
            if not config.ENABLE_MOTORS:
                self.get_logger().info(f"[dry-run] would move to XYZ=({x:.2f},{y:.2f},{z:.2f})")
                return
            ok, ik_deg, fb_deg, fk_xyz, err = self._ctrl.move_xyz(x, y, z, raw=False)
            if ik_deg is None:
                self.get_logger().warn(f"Target ({x:.1f},{y:.1f},{z:.1f}) rejected — outside workspace")
            elif fb_deg is None:
                self.get_logger().warn(f"IK={ik_deg} rejected — outside joint limits")
            elif fk_xyz is None:
                self.get_logger().warn("Commanded, but FK on feedback failed")
            else:
                self.get_logger().info(
                    f"Moved -> FK=({fk_xyz[0]:.2f},{fk_xyz[1]:.2f},{fk_xyz[2]:.2f}) "
                    f"err={err:.2f}mm {'OK' if ok else 'OUT OF TOLERANCE'}"
                )
        except Exception as exc:
            self.get_logger().error(f"Move to ({x:.1f},{y:.1f},{z:.1f}) failed: {exc}")
        finally:
            with self._move_lock:
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
    node = LaserTeleopNode()

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
