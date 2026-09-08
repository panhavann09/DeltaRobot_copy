#!/usr/bin/env python3
"""
weed_bridge_node.py — Bridges plant_perception's TrackedPlantArray to the
existing DeltaRobot pick-and-place interface, replacing
delta_camera_system's orange-cube detector for weed picking.

Publishes the exact same topics/types camera_system.py already publishes
(/delta/target_xyz, /delta/all_targets, /delta/detection_status,
/delta/object_velocity_mm_s) so pick_place_node.py / blind_pick_place.py
need no changes. Reuses delta_common.camera_geometry
(the camera->base transform and IK-feasibility gate extracted from
camera_system.py) as the single source of truth for that math.

Also reproduces delta_camera_system's own "Delta Camera" live window —
workspace/conveyor zones + EE laser/FK marker, via the shared
delta_common.workspace_overlay / delta_common.ee_marker modules extracted
from camera_system.py — drawn on top of plant_perception's own
/plant_perception/visualization frame (which already has the per-object weed
boxes/track_id/roots and its own FPS/latency HUD burned in), so everything
shows in one window. No separate rqt_image_view needed.

EE-marker laser detection runs on the raw color feed (config.COLOR_TOPIC) —
plant_perception's frame has boxes/text drawn over it, which would confuse
the HSV laser-dot search — but the marker itself is drawn onto the
already-annotated visualization frame for display.

Run in place of delta_camera_system's camera_node — do not run both at once,
they'd both publish to the same topics.
"""

import time
from collections import deque

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PointStamped, Pose, PoseArray
from std_msgs.msg import String
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

from delta_common import config, camera_geometry, workspace_overlay
from delta_common.ee_marker import EEMarkerDetector

try:
    from plant_perception.msg import TrackedPlantArray
except ImportError:
    TrackedPlantArray = None

# Mirrors plant_perception/track_state.py's CROP_CLASSES/WEED_CLASSES.
# Duplicated rather than imported: plant_perception installs its Python
# modules as standalone executables (lib/plant_perception/), not an
# importable package, so another package can't `from plant_perception import
# track_state`. Keep this in sync with track_state.py's CLASS_NAMES.
CROP_CLASSES = {0, 1}
WEED_CLASSES = {2, 3}


class WeedBridgeNode(Node):
    def __init__(self):
        super().__init__("weed_bridge_node")

        if TrackedPlantArray is None:
            self.get_logger().error(
                "plant_perception.msg.TrackedPlantArray unavailable — "
                "build/source the plant_perception package first."
            )

        self.fx = self.fy = self.cx = self.cy = None
        self.T_cam_to_base, self.T_base_to_cam = camera_geometry.build_T_cam_to_base()

        self._last_target_publish_time = 0.0
        self._timed_x_bufs: dict = {}   # track_id -> deque[(t_sec, x_base_mm)]
        self._cal_active = False
        self._cal_samples: list = []

        self._ws_poly = None
        self._ee_marker = EEMarkerDetector()
        self._ee_fk_pixel = None
        self._last_ee_uv = None

        self._pub_target = self.create_publisher(PointStamped, "/delta/target_xyz", 10)
        self._pub_velocity = self.create_publisher(PointStamped, "/delta/object_velocity_mm_s", 10)
        self._pub_status = self.create_publisher(String, "/delta/detection_status", 10)
        self._pub_all_targets = self.create_publisher(PoseArray, "/delta/all_targets", 10)

        self.create_subscription(CameraInfo, config.CAMERA_INFO_TOPIC, self._on_camera_info, 1)
        if TrackedPlantArray is not None:
            self.create_subscription(
                TrackedPlantArray, config.TRACKED_PLANTS_TOPIC, self._on_tracked_plants, 10
            )
        self.create_service(Trigger, "/delta/calibrate_cam_offset", self._calibrate_offset_srv)
        # Published by pick_place_node.py (mm, robot base frame) — FK fallback
        # for the EE marker when the laser dot isn't visible, same as camera_system.py.
        self.create_subscription(PointStamped, "/delta/ee_fk_xyz", self._on_ee_fk, 10)

        self.view_image = bool(config.VIEW_IMAGE)
        if self.view_image:
            self.bridge = CvBridge()
            # merge_code_test.py publishes both of these BEST_EFFORT
            # (see _best_effort_qos()) — match it here or the subscriptions
            # silently receive nothing.
            best_effort_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.create_subscription(Image, config.COLOR_TOPIC, self._on_color_image, best_effort_qos)
            self.create_subscription(
                Image, config.PLANT_PERCEPTION_VISUALIZATION_TOPIC, self._on_visualization, best_effort_qos
            )
            cv2.namedWindow("Delta Camera", cv2.WINDOW_NORMAL)

        self.get_logger().info(
            "weed_bridge_node ready\n"
            f"  Listening : {config.TRACKED_PLANTS_TOPIC} (TrackedPlantArray)\n"
            "  Publishing: /delta/target_xyz, /delta/all_targets, "
            "/delta/detection_status, /delta/object_velocity_mm_s"
        )

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info(
                f"Camera intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} "
                f"cx={self.cx:.1f} cy={self.cy:.1f}"
            )

    def _on_ee_fk(self, msg: PointStamped) -> None:
        if self.fx is None:
            return
        self._ee_fk_pixel = camera_geometry.project_base_to_pixel(
            self.fx, self.fy, self.cx, self.cy, self.T_base_to_cam,
            msg.point.x, msg.point.y, msg.point.z,
        )

    def _on_color_image(self, msg: Image) -> None:
        """Raw feed — EE laser-marker detection only (needs clean pixel colors,
        not plant_perception's already-annotated frame). Not displayed itself;
        _on_visualization draws the result onto the display frame."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Color image decode failed: {e}")
            return
        self._last_ee_uv = self._ee_marker.detect(frame, ws_corners=self._ws_poly)

        # Cam-offset calibration: sample the laser/EE-marker pixel directly
        # (not the weed detector) so /delta/calibrate_cam_offset works with
        # the laser fixed at true center, no weed needed. Only valid while
        # the marker sits at belt depth (config.FAKE_DEPTH_M) — see
        # _on_tracked_plants' identical root_d assumption for weeds.
        if self._cal_active and self._last_ee_uv is not None and self.fx is not None:
            eu, ev = self._last_ee_uv
            x_cam, y_cam, z_cam = camera_geometry.pixel_to_camera_xyz_mm(
                self.fx, self.fy, self.cx, self.cy, eu, ev, config.FAKE_DEPTH_M
            )
            x_base, y_base, _ = camera_geometry.camera_to_base_mm(
                self.T_cam_to_base, x_cam, y_cam, z_cam
            )
            self._collect_cal_sample({"x_base": x_base, "y_base": y_base})

    def _on_visualization(self, msg: Image) -> None:
        """Delta Camera window — workspace/conveyor zones + EE marker drawn on
        top of plant_perception's own annotated frame (weed boxes/track_id/
        roots + its own FPS/latency HUD already burned in)."""
        try:
            annotated = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Visualization decode failed: {e}")
            return

        frame_h, frame_w = annotated.shape[:2]

        if self.fx is not None:
            def project_fn(x_b, y_b, z_b):
                return camera_geometry.project_base_to_pixel(
                    self.fx, self.fy, self.cx, self.cy, self.T_base_to_cam, x_b, y_b, z_b
                )
            ws_poly = workspace_overlay.draw_workspace_overlay(annotated, frame_w, frame_h, project_fn)
            if ws_poly is not None:
                self._ws_poly = ws_poly

        if self._last_ee_uv is not None:
            if getattr(config, "DRAW_EE_MARKER", True):
                eu, ev = self._last_ee_uv
                cv2.circle(annotated, (eu, ev), 6, (255, 255, 0), 2)
                cv2.putText(annotated, "EE", (eu + 8, ev),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0), 1)
        elif self._ee_fk_pixel is not None and getattr(config, "DRAW_EE_MARKER", True):
            fu, fv = self._ee_fk_pixel
            cv2.drawMarker(annotated, (fu, fv), (0, 200, 0), cv2.MARKER_CROSS, 10, 1)
            cv2.putText(annotated, "FK", (fu + 8, fv),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 200, 0), 1)

        cv2.imshow("Delta Camera", annotated)
        cv2.waitKey(1)

    def _on_tracked_plants(self, msg) -> None:
        if self.fx is None:
            return

        crops = [p for p in msg.plants if p.class_id in CROP_CLASSES]
        weeds = [p for p in msg.plants if p.class_id in WEED_CLASSES and p.actionable]

        # TEMP DEBUG: dump every weed-class track's gating state, actionable
        # or not, so we can see exactly which condition (confirm-hit count vs
        # depth validity vs missed frames) is delaying a track from becoming
        # actionable while it's still entering frame. Run with
        # --ros-args --log-level weed_bridge_node:=debug to see this.
        for p in msg.plants:
            if p.class_id not in WEED_CLASSES:
                continue
            self.get_logger().debug(
                f"[track_dbg] id={p.track_id} actionable={p.actionable} "
                f"bbox=({p.x1:.0f},{p.y1:.0f})-({p.x2:.0f},{p.y2:.0f}) "
                f"root=({p.root_x:.0f},{p.root_y:.0f}) "
                f"confirmed={p.is_confirmed} detected={p.is_currently_detected} "
                f"depth_valid={p.current_depth_valid} "
                f"hits={p.consecutive_hits} missed={p.missed_frames}"
            )

        candidates = []
        for w in weeds:
            if self._overlaps_crop(w, crops):
                continue

            root_d = config.FAKE_DEPTH_M if config.FAKE_DEPTH_ENABLE else w.root_d
            x_cam, y_cam, z_cam = camera_geometry.pixel_to_camera_xyz_mm(
                self.fx, self.fy, self.cx, self.cy, w.root_x, w.root_y, root_d
            )
            x_base, y_base, z_base = camera_geometry.camera_to_base_mm(
                self.T_cam_to_base, x_cam, y_cam, z_cam
            )

            _, _, _, allowed, reason = camera_geometry.validate_target(x_base, y_base, z_base)
            vx = self._update_velocity(w.track_id, x_base)

            # TEMP DEBUG: see part 2 above — compare this against the bbox
            # printed there to check whether check_workspace (the robot's
            # reachable X_LIMIT/Y_LIMIT/Z envelope) is the actual gate,
            # independent of the track's actionable/confirm-hit state.
            self.get_logger().debug(
                f"[target_dbg] id={w.track_id} "
                f"x_base={x_base:.1f} y_base={y_base:.1f} z_base={z_base:.1f} "
                f"allowed={allowed} reason={reason}"
            )

            candidates.append({
                "track_id": w.track_id,
                "x_base": x_base,
                "y_base": y_base,
                "z_base": z_base,
                "allowed": allowed,
                "reason": reason,
                "vx": vx,
            })

        self._publish(candidates)

    def _calibrate_offset_srv(self, request, response):
        self._cal_samples = []
        self._cal_active = True
        self.get_logger().info(
            "Cam offset calibration started — EE laser must be at robot centre "
            "(X=0, Y=0) AND at belt height "
            f"({config.FAKE_DEPTH_M * 1000:.0f}mm from camera). "
            "Collecting 30 detections..."
        )
        response.success = True
        response.message = "Calibration started — results printed after 30 detections"
        return response

    def _collect_cal_sample(self, best: dict) -> None:
        if not self._cal_active:
            return
        self._cal_samples.append((best["x_base"], best["y_base"]))
        n = len(self._cal_samples)
        if n < 30:
            if n % 5 == 0:
                self.get_logger().info(f"Calibration: {n}/30 samples...")
            return
        self._cal_active = False
        mean_x = sum(s[0] for s in self._cal_samples) / n
        mean_y = sum(s[1] for s in self._cal_samples) / n
        new_tx = config.CAM_TX_MM - mean_x
        new_ty = config.CAM_TY_MM - mean_y
        self.get_logger().info(
            f"\n====== CAM OFFSET CALIBRATION RESULT ======\n"
            f"  Samples     : {n}\n"
            f"  mean x_base : {mean_x:+.2f} mm\n"
            f"  mean y_base : {mean_y:+.2f} mm\n"
            f"  Corrections needed:\n"
            f"    CAM_TX_MM  {config.CAM_TX_MM:.2f} + ({-mean_x:+.2f}) = {new_tx:.2f}\n"
            f"    CAM_TY_MM  {config.CAM_TY_MM:.2f} + ({-mean_y:+.2f}) = {new_ty:.2f}\n"
            f"  → Update config.py:\n"
            f"    CAM_TX_MM = {new_tx:.2f}\n"
            f"    CAM_TY_MM = {new_ty:.2f}\n"
            f"==========================================="
        )

    @staticmethod
    def _overlaps_crop(weed, crops) -> bool:
        wx1, wy1, wx2, wy2 = weed.x1, weed.y1, weed.x2, weed.y2
        w_area = max(0.0, wx2 - wx1) * max(0.0, wy2 - wy1)
        if w_area <= 0.0:
            return False
        for c in crops:
            ix1, iy1 = max(wx1, c.x1), max(wy1, c.y1)
            ix2, iy2 = min(wx2, c.x2), min(wy2, c.y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / w_area > config.WEED_CROP_OVERLAP_MAX_IOU:
                return True
        return False

    def _update_velocity(self, track_id: int, x_base_mm: float) -> float:
        buf = self._timed_x_bufs.setdefault(track_id, deque(maxlen=config.AVG_FRAME_COUNT))
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        buf.append((now_sec, x_base_mm))
        return camera_geometry.estimate_conveyor_vx_mm_s(buf)

    def _publish(self, candidates) -> None:
        allowed = [c for c in candidates if c["allowed"]]

        if candidates:
            best_any = min(candidates, key=lambda c: c["z_base"])
            status_msg = String()
            status_msg.data = best_any["reason"]
            self._pub_status.publish(status_msg)

        if not allowed:
            return

        best = min(allowed, key=lambda c: c["z_base"])
        now_t = time.time()
        if now_t - self._last_target_publish_time >= config.TARGET_PUBLISH_COOLDOWN_SEC:
            self._last_target_publish_time = now_t
            stamp = self.get_clock().now().to_msg()

            pt = PointStamped()
            pt.header.stamp = stamp
            pt.header.frame_id = "robot_base"
            pt.point.x = best["x_base"] / 1000.0
            pt.point.y = best["y_base"] / 1000.0
            pt.point.z = best["z_base"] / 1000.0
            self._pub_target.publish(pt)

            vel = PointStamped()
            vel.header.stamp = stamp
            vel.header.frame_id = "robot_base"
            vel.point.x = best["vx"]
            self._pub_velocity.publish(vel)

        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = "robot_base"
        for c in sorted(allowed, key=lambda c: c["x_base"]):
            p = Pose()
            p.position.x = c["x_base"] / 1000.0
            p.position.y = c["y_base"] / 1000.0
            p.position.z = c["z_base"] / 1000.0
            pa.poses.append(p)
        self._pub_all_targets.publish(pa)

    def destroy_node(self):
        if self.view_image:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WeedBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
