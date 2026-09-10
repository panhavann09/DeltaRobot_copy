#!/usr/bin/env python3
"""
merge_code_test.py — Direct-camera-pipeline variant of plant_perception_node.py.

Grabs color frames straight from a UVC/V4L2 global-shutter camera via a
GStreamer appsink pipeline instead of subscribing to a camera-driver ROS
node's topics. This removes the camera-driver -> DDS serialization ->
subscriber hop entirely. The camera has no depth stream, so depth_img is
always None here — downstream code already falls back to an invalid
DepthEstimate in that case, and pick_place_node uses config.FAKE_DEPTH_M
for Z.

2026-09-08: capture used to go through cv2.VideoCapture(..., CAP_V4L2),
which decodes this camera's MJPG stream in software (libjpeg, on the CPU) —
measured at 133ms/frame at 1920x1080 (7.5 FPS flat), a hard ceiling
independent of how fast inference ran. Jetson has a dedicated hardware JPEG
decoder exposed via GStreamer's nvv4l2decoder; routing capture through it
(v4l2src -> nvv4l2decoder mjpeg=1 -> nvvidconv -> appsink) measured
27.6ms/frame (~36 FPS) with the CPU almost entirely idle instead. Requires
camera_fourcc == "MJPG" (this camera's only mode — see
`v4l2-ctl --list-formats-ext`); there's no software-decode fallback path
since one was never needed here.

Do not run this alongside plant_perception_node.py — only one process can
hold the camera device open at a time.
"""

import queue
import threading
import time

import numpy as np
import cv2
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge

from delta_common import config as delta_config

Gst.init(None)

try:
    from plant_perception.msg import TrackedPlant, TrackedPlantArray
except ImportError:
    print("[merge_code_test] ERROR: Custom messages not compiled or not in path.")
    TrackedPlant = None
    TrackedPlantArray = None

try:
    from trt_inferencer import TRTInferencer
    from tiled_inference import run_tiled_inference
    from tracker import Tracker
    from track_state import TrackStateManager
    from depth_utils import estimate_root_depth, DepthEstimate
    from visualization import draw_visualizations
except ImportError:
    from .trt_inferencer import TRTInferencer
    from .tiled_inference import run_tiled_inference
    from .tracker import Tracker
    from .track_state import TrackStateManager
    from .depth_utils import estimate_root_depth, DepthEstimate
    from .visualization import draw_visualizations

CLASS_NAMES = {
    0: "crop_small_leaf",
    1: "crop_large_leaf",
    2: "weed_small_leaf",
    3: "weed_large_leaf",
}


class MergeCodeTestNode(Node):
    """Direct-pipeline perception node: pulls color frames straight from a
    UVC/V4L2 global-shutter camera via cv2.VideoCapture instead of
    subscribing to a camera-driver ROS node's topics."""

    def __init__(self):
        super().__init__("merge_code_test_node")
        self.get_logger().info("Initializing merge_code_test_node (direct UVC camera pipeline)...")

        # Model parameters
        # best_fp16.engine's content is actually the INT8 engine — the real
        # fp16 weights are best_fp16__copy.engine, see perception_direct.yaml.
        self.declare_parameter("model_path", "src/weight/best_fp16__copy.engine")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf_thres", 0.25)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("device", "tensorrt")
        self.declare_parameter("target_fps", 30.0)
        self.declare_parameter("trt_fp16_enable", True)

        # Tiled inference (see tiled_inference.py) — splits a high-res frame
        # into overlapping tiles run through the model individually instead
        # of one full-frame downscale, so small/distant weeds keep more
        # detail. 1088/96 -> 2 tiles at 1920x1080 instead of 6 (see
        # tiled_inference.py / perception_direct.yaml for the full story).
        self.declare_parameter("tile_size", 1088)
        self.declare_parameter("tile_overlap", 96)

        # Vertical center-crop applied to every captured frame BEFORE
        # anything else touches it (inference, visualization, publish) — see
        # frame_crop_height below and ~/yolo_bench_ws/yolo_test_fp16_high.py,
        # which this mirrors: 1280x720 capture, center-cropped to 640 tall so
        # tile_size=640/tile_overlap=0 splits the frame into exactly two
        # 640x640 tiles with zero letterbox resize on either axis. 0 disables
        # cropping (frame used as captured).
        self.declare_parameter("frame_crop_height", 0)

        # Tracker parameters
        self.declare_parameter("tracker_type", "bytetrack")
        self.declare_parameter("track_high_thresh", 0.35)
        self.declare_parameter("track_low_thresh", 0.10)
        self.declare_parameter("new_track_thresh", 0.40)
        self.declare_parameter("track_buffer", 30)
        self.declare_parameter("match_thresh", 0.75)
        self.declare_parameter("frame_rate", 30)

        # Track stability parameters
        self.declare_parameter("minimum_confirm_hits", 3)
        self.declare_parameter("maximum_publish_misses", 2)
        self.declare_parameter("maximum_track_misses", 15)
        self.declare_parameter("bbox_ema_alpha", 0.50)
        self.declare_parameter("root_ema_alpha", 0.35)
        self.declare_parameter("class_history_size", 5)
        self.declare_parameter("depth_history_size", 5)

        # Segmentation parameters
        self.declare_parameter("publish_masks", False)

        # Depth parameters (no depth stream on the UVC global-shutter camera —
        # kept false by default; depth_img is always None regardless)
        self.declare_parameter("use_depth", False)
        self.declare_parameter("depth_window_size", 7)
        self.declare_parameter("minimum_depth_valid_ratio", 0.40)
        self.declare_parameter("minimum_depth_m", 0.10)
        self.declare_parameter("maximum_depth_m", 3.00)
        self.declare_parameter("depth_mad_multiplier", 2.5)

        # Visualization parameters
        self.declare_parameter("enable_visualization", True)
        self.declare_parameter("visualization_show_masks", False)
        self.declare_parameter("visualization_jpeg_quality", 80)
        # 2026-09-08: a subscribed debug viewer used to draw+encode+publish
        # inline in process_frame, on the same thread as inference/tracking
        # -> tracked_plants (the topic pick_place_node actually acts on)
        # dropped to whatever rate the viewer could keep up with (measured
        # ~5.8-9.7 FPS with a viewer attached vs ~22 FPS with none). Moved
        # to its own thread below (_visualization_loop) so a debug viewer
        # can never throttle robot-control-relevant output again.
        # visualization_max_width also downscales the annotated frame before
        # encode/publish — cv2.imencode of a full 1920x1080 frame measured
        # 28.3ms, vs 6.0ms at 960x540 — 0 disables downscaling.
        self.declare_parameter("visualization_max_width", 960)

        # Direct camera parameters (replaces color/depth topic parameters)
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("camera_fps", 30)
        self.declare_parameter("camera_device", "/dev/video0")
        self.declare_parameter("camera_fourcc", "MJPG")
        # Global-shutter UVC camera has no hardware intrinsics API like
        # librealsense2 did — these MUST be replaced with real values from a
        # standard OpenCV chessboard calibration of this camera/lens before
        # relying on pixel_to_camera_xyz_mm() for accurate picking.
        self.declare_parameter("camera_fx", 600.0)
        self.declare_parameter("camera_fy", 600.0)
        self.declare_parameter("camera_cx", 320.0)
        self.declare_parameter("camera_cy", 240.0)
        self.declare_parameter("camera_dist_k1", 0.0)
        self.declare_parameter("camera_dist_k2", 0.0)
        self.declare_parameter("camera_dist_p1", 0.0)
        self.declare_parameter("camera_dist_p2", 0.0)
        self.declare_parameter("camera_dist_k3", 0.0)
        self.declare_parameter("tracked_plants_topic", "/plant_perception/tracked_plants")
        self.declare_parameter("visualization_topic", "/plant_perception/visualization")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("color_image_topic", "/camera/camera/color/image_raw")

        # Retrieve and cache parameters
        self.model_path = self._resolve_model_path(self.get_parameter("model_path").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf_thres = float(self.get_parameter("conf_thres").value)
        self.iou_thres = float(self.get_parameter("iou_thres").value)
        self.device = str(self.get_parameter("device").value)
        self.target_fps = float(self.get_parameter("target_fps").value)
        self.trt_fp16_enable = bool(self.get_parameter("trt_fp16_enable").value)
        self.tile_size = int(self.get_parameter("tile_size").value)
        self.tile_overlap = int(self.get_parameter("tile_overlap").value)
        self.frame_crop_height = int(self.get_parameter("frame_crop_height").value)

        self.tracker_type = str(self.get_parameter("tracker_type").value)
        self.track_high_thresh = float(self.get_parameter("track_high_thresh").value)
        self.track_low_thresh = float(self.get_parameter("track_low_thresh").value)
        self.new_track_thresh = float(self.get_parameter("new_track_thresh").value)
        self.track_buffer = int(self.get_parameter("track_buffer").value)
        self.match_thresh = float(self.get_parameter("match_thresh").value)
        self.frame_rate = int(self.get_parameter("frame_rate").value)

        self.minimum_confirm_hits = int(self.get_parameter("minimum_confirm_hits").value)
        self.maximum_publish_misses = int(self.get_parameter("maximum_publish_misses").value)
        self.maximum_track_misses = int(self.get_parameter("maximum_track_misses").value)
        self.bbox_ema_alpha = float(self.get_parameter("bbox_ema_alpha").value)
        self.root_ema_alpha = float(self.get_parameter("root_ema_alpha").value)
        self.class_history_size = int(self.get_parameter("class_history_size").value)
        self.depth_history_size = int(self.get_parameter("depth_history_size").value)

        self.publish_masks = bool(self.get_parameter("publish_masks").value)
        self.use_depth = bool(self.get_parameter("use_depth").value)
        self.depth_window_size = int(self.get_parameter("depth_window_size").value)
        if self.depth_window_size % 2 == 0:
            self.depth_window_size += 1
            self.get_logger().warn(f"Even depth_window_size converted to odd: {self.depth_window_size}")

        self.minimum_depth_valid_ratio = float(self.get_parameter("minimum_depth_valid_ratio").value)
        self.minimum_depth_m = float(self.get_parameter("minimum_depth_m").value)
        self.maximum_depth_m = float(self.get_parameter("maximum_depth_m").value)
        self.depth_mad_multiplier = float(self.get_parameter("depth_mad_multiplier").value)

        self.enable_visualization = bool(self.get_parameter("enable_visualization").value)
        self.visualization_show_masks = bool(self.get_parameter("visualization_show_masks").value)
        self.visualization_jpeg_quality = int(self.get_parameter("visualization_jpeg_quality").value)
        self.visualization_max_width = int(self.get_parameter("visualization_max_width").value)

        self.camera_width = int(self.get_parameter("camera_width").value)
        self.camera_height = int(self.get_parameter("camera_height").value)
        self.camera_fps = int(self.get_parameter("camera_fps").value)
        if self.frame_crop_height > 0 and self.frame_crop_height < self.camera_height:
            self.crop_top = (self.camera_height - self.frame_crop_height) // 2
            self.cropped_height = self.frame_crop_height
        else:
            self.crop_top = 0
            self.cropped_height = self.camera_height
        self.camera_device = str(self.get_parameter("camera_device").value)
        self.camera_fourcc = str(self.get_parameter("camera_fourcc").value)
        self.camera_fx = float(self.get_parameter("camera_fx").value)
        self.camera_fy = float(self.get_parameter("camera_fy").value)
        self.camera_cx = float(self.get_parameter("camera_cx").value)
        self.camera_cy = float(self.get_parameter("camera_cy").value)
        self.camera_dist = [
            float(self.get_parameter("camera_dist_k1").value),
            float(self.get_parameter("camera_dist_k2").value),
            float(self.get_parameter("camera_dist_p1").value),
            float(self.get_parameter("camera_dist_p2").value),
            float(self.get_parameter("camera_dist_k3").value),
        ]
        self.tracked_plants_topic = str(self.get_parameter("tracked_plants_topic").value)
        self.visualization_topic = str(self.get_parameter("visualization_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.color_image_topic = str(self.get_parameter("color_image_topic").value)

        # Core modules
        self.bridge = CvBridge()
        self.inferencer = TRTInferencer(
            model_path=self.model_path,
            device=self.device,
            imgsz=self.imgsz,
            trt_fp16_enable=self.trt_fp16_enable,
        )
        self.tracker = Tracker(
            tracker_type=self.tracker_type,
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            frame_rate=self.frame_rate,
        )
        self.state_manager = TrackStateManager(
            minimum_confirm_hits=self.minimum_confirm_hits,
            maximum_publish_misses=self.maximum_publish_misses,
            maximum_track_misses=self.maximum_track_misses,
            bbox_ema_alpha=self.bbox_ema_alpha,
            root_ema_alpha=self.root_ema_alpha,
            class_history_size=self.class_history_size,
            depth_history_size=self.depth_history_size,
            require_depth=not delta_config.FAKE_DEPTH_ENABLE,
        )

        # Publishers (same topics/types as plant_perception_node.py, so existing
        # tools like rqt_image_view work against this node without reconfiguring)
        if TrackedPlantArray is not None:
            self.pub_tracked_plants = self.create_publisher(
                TrackedPlantArray, self.tracked_plants_topic, self._reliable_qos()
            )
        else:
            self.pub_tracked_plants = None
            self.get_logger().warn("TrackedPlantArray message class unavailable — publisher not created.")

        self.pub_visualization = self.create_publisher(
            Image, self.visualization_topic, self._best_effort_qos()
        )
        self.pub_visualization_compressed = self.create_publisher(
            CompressedImage, self.visualization_topic + "/compressed", self._best_effort_qos()
        )
        # No camera-driver ROS node running in this pipeline, so publish
        # CameraInfo ourselves — other nodes (e.g. weed_bridge_node) that need
        # intrinsics subscribe here exactly like they would to a ROS driver.
        self.pub_camera_info = self.create_publisher(CameraInfo, self.camera_info_topic, 1)
        self._camera_info_msg = None
        # Raw (unannotated) color frame, same topic name/content a ROS
        # camera-driver node would publish — lets other nodes (e.g.
        # weed_bridge_node's own "Delta Camera" window: workspace zones,
        # EE laser-marker detection) work directly off the camera feed.
        self.pub_color = self.create_publisher(Image, self.color_image_topic, self._best_effort_qos())

        # Frame timing / heartbeat
        self.frame_times = []
        self.last_infer_time = 0.0
        self.last_proc_time = time.time()
        self.last_capture_time = time.time()

        # Direct UVC/V4L2 camera capture — no camera-driver ROS node involved.
        # This camera has no depth stream, so depth_img is always None from
        # here on; use_depth just controls whether a warning is logged.
        if self.use_depth:
            self.get_logger().warn(
                "use_depth=true but this pipeline reads a global-shutter UVC "
                "camera with no depth stream — depth estimates will always be "
                "invalid. Set use_depth:=false to silence this."
            )
        self.depth_scale = None

        if self.camera_fourcc != "MJPG":
            raise RuntimeError(
                f"camera_fourcc={self.camera_fourcc!r} not supported — the hardware-decode "
                "GStreamer capture path only handles this camera's MJPG mode. Add a "
                "software-decode (cv2.VideoCapture) fallback here if a non-MJPG camera "
                "is ever used."
            )

        self._gst_pipeline, self._gst_sink = self._open_camera_gst()
        actual_w, actual_h = self.camera_width, self.cropped_height
        crop_note = (
            f", center-cropped {self.camera_height}->{self.cropped_height} (rows "
            f"{self.crop_top}:{self.crop_top + self.cropped_height})"
            if self.crop_top > 0 else ""
        )
        self.get_logger().info(
            f"UVC camera opened via GStreamer hardware JPEG decode: {self.camera_device} "
            f"{self.camera_width}x{self.camera_height}@{self.camera_fps}fps ({self.camera_fourcc})"
            f"{crop_note}"
        )
        self._camera_info_msg = self._build_camera_info_msg(actual_w, actual_h)

        # Capture (camera I/O), processing (inference/tracking/publish), and
        # visualization (draw/encode/publish for a debug viewer) each run on
        # their own thread so a slow stage can never delay the others — in
        # particular, a debug viewer subscribed to the visualization topic
        # must never throttle tracked_plants, the topic pick_place_node
        # actually acts on.
        self._frame_queue = queue.Queue(maxsize=1)
        self._vis_queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._vis_thread = threading.Thread(target=self._visualization_loop, daemon=True)
        self._capture_thread.start()
        self._worker_thread.start()
        self._vis_thread.start()

        self.timer_diag = self.create_timer(5.0, self._check_heartbeat)
        self.get_logger().info("merge_code_test_node startup complete.")

    # ------------------------------------------------------------------
    # Direct camera capture
    # ------------------------------------------------------------------
    def _open_camera_gst(self):
        """Build and start a GStreamer pipeline that decodes this camera's
        MJPG stream on Jetson's dedicated hardware JPEG decoder
        (nvv4l2decoder) instead of the CPU, then converts to a plain BGR
        appsink so the rest of this file keeps working with numpy arrays.
        See the module docstring for the measured 7.5 FPS -> ~36 FPS why."""
        pipeline_str = (
            f"v4l2src device={self.camera_device} ! "
            f"image/jpeg,width={self.camera_width},height={self.camera_height},"
            f"framerate={self.camera_fps}/1 ! "
            "nvv4l2decoder mjpeg=1 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
        )
        pipeline = Gst.parse_launch(pipeline_str)
        sink = pipeline.get_by_name("sink")
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(
                f"Could not start GStreamer capture pipeline for {self.camera_device!r} "
                f"at {self.camera_width}x{self.camera_height}@{self.camera_fps}fps"
            )

        # set_state(PLAYING) above only requests the transition — for a live
        # source it returns ASYNC immediately, well before nvv4l2decoder's
        # hardware block actually finishes init. Block here until the
        # pipeline confirms PLAYING (or the bus reports why it didn't), so
        # _capture_loop never starts pulling during that window — appsink's
        # pull-sample returns None with no error while the sink is still in
        # READY, which otherwise shows up as a burst of spurious "frame grab
        # failed" warnings on every startup.
        state_ret, state, _ = pipeline.get_state(10 * Gst.SECOND)
        if state_ret != Gst.StateChangeReturn.SUCCESS or state != Gst.State.PLAYING:
            bus = pipeline.get_bus()
            msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.WARNING)
            detail = []
            while msg:
                if msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                else:
                    err, debug = msg.parse_warning()
                detail.append(f"{err} ({debug})")
                msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.WARNING)
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(
                f"GStreamer capture pipeline for {self.camera_device!r} at "
                f"{self.camera_width}x{self.camera_height}@{self.camera_fps}fps never "
                f"reached PLAYING (state={state.value_nick}): " + ("; ".join(detail) or "no bus error reported")
            )
        return pipeline, sink

    def _capture_loop(self):
        while not self._stop_event.is_set() and rclpy.ok():
            sample = self._gst_sink.emit("pull-sample")
            if sample is None:
                self.get_logger().warn("UVC camera frame grab failed, retrying...")
                time.sleep(0.01)
                continue

            buf = sample.get_buffer()
            caps = sample.get_caps().get_structure(0)
            w = caps.get_value("width")
            h = caps.get_value("height")
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                self.get_logger().warn("UVC camera frame buffer map failed, retrying...")
                continue
            try:
                cv_img = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(h, w, 3).copy()
            finally:
                buf.unmap(mapinfo)

            if self.crop_top > 0:
                cv_img = cv_img[self.crop_top:self.crop_top + self.cropped_height, :]

            self.last_capture_time = time.time()
            self._enqueue_frame(cv_img, None)

    def _enqueue_frame(self, cv_img, depth_img):
        try:
            self._frame_queue.put_nowait((cv_img, depth_img))
        except queue.Full:
            try:
                self._frame_queue.get_nowait()  # drop the stale pending frame
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait((cv_img, depth_img))
            except queue.Full:
                pass

    def _worker_loop(self):
        while not self._stop_event.is_set() and rclpy.ok():
            try:
                cv_img, depth_img = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self.process_frame(cv_img, depth_img)
            except Exception as e:
                self.get_logger().error(f"Worker frame processing failed: {e}")

    def _visualization_loop(self):
        """Draw/encode/publish the debug visualization AND the raw color
        feed on their own thread — see the visualization_max_width and
        color_needed comments in process_frame for why both are split out
        of the critical inference/tracking/tracked_plants path."""
        while not self._stop_event.is_set() and rclpy.ok():
            try:
                cv_img, active_states, fps, inference_ms, header, viz_needed, color_needed = (
                    self._vis_queue.get(timeout=1.0)
                )
            except queue.Empty:
                continue

            if color_needed and self.pub_color.get_subscription_count() > 0:
                try:
                    # passthrough + explicit encoding sidesteps a cv_bridge/opencv-python
                    # ABI mismatch in this environment (see pub_visualization below).
                    color_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="passthrough")
                    color_msg.encoding = "bgr8"
                    color_msg.header = header
                    self.pub_color.publish(color_msg)
                except Exception as e:
                    self.get_logger().error(f"Color image publishing failed: {e}")

            if not viz_needed:
                continue

            try:
                annotated_img = draw_visualizations(
                    img_bgr=cv_img,
                    detections=active_states,
                    show_boxes=True,
                    show_masks=(self.publish_masks and self.visualization_show_masks),
                    show_roots=True,
                    fps=fps,
                    latency_ms=inference_ms,
                    device=self.device,
                )

                if self.visualization_max_width > 0 and annotated_img.shape[1] > self.visualization_max_width:
                    vh, vw = annotated_img.shape[:2]
                    new_w = self.visualization_max_width
                    new_h = int(round(vh * (new_w / vw)))
                    annotated_img = cv2.resize(annotated_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                if self.pub_visualization.get_subscription_count() > 0:
                    # passthrough + explicit encoding sidesteps a cv_bridge/opencv-python
                    # ABI mismatch in this environment (see plant_perception_node.py).
                    vis_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding="passthrough")
                    vis_msg.encoding = "bgr8"
                    vis_msg.header = header
                    self.pub_visualization.publish(vis_msg)

                if self.pub_visualization_compressed.get_subscription_count() > 0:
                    compressed_msg = CompressedImage()
                    compressed_msg.header = header
                    compressed_msg.format = "jpeg"
                    quality = [int(cv2.IMWRITE_JPEG_QUALITY), self.visualization_jpeg_quality]
                    ret, jpeg_data = cv2.imencode(".jpg", annotated_img, quality)
                    if ret:
                        compressed_msg.data = jpeg_data.tobytes()
                        self.pub_visualization_compressed.publish(compressed_msg)
            except Exception as e:
                self.get_logger().error(f"Visualization publishing failed: {e}")

    # ------------------------------------------------------------------
    # Processing pipeline (mirrors PlantPerceptionNode.sync_callback, adapted
    # for numpy frames straight from cv2.VideoCapture instead of ROS Image msgs)
    # ------------------------------------------------------------------
    def process_frame(self, cv_img, depth_img):
        now = time.time()
        self.last_proc_time = now

        # Target FPS throttling
        if self.target_fps > 0 and (now - self.last_infer_time) < (1.0 / self.target_fps):
            return
        self.last_infer_time = now

        self.frame_times.append(now)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        fps = ((len(self.frame_times) - 1) / (self.frame_times[-1] - self.frame_times[0])
               if len(self.frame_times) > 1 else 0.0)

        header = self._make_header()

        if self._camera_info_msg is not None:
            self._camera_info_msg.header = header
            self.pub_camera_info.publish(self._camera_info_msg)

        # pub_color's publish used to happen inline right here — cheap when nothing
        # subscribes to color_image_topic, but weed_bridge_node's own "Delta Camera"
        # window does (config.VIEW_IMAGE=True by default), and at that point this was
        # a full 1920x1080 bgr8 cv2_to_imgmsg + publish on every accepted frame, on
        # the same thread as inference/tracking/tracked_plants — the exact bug already
        # fixed for the annotated visualization output below, just missed here because
        # nothing subscribed to color_image_topic in isolated testing. Now handed to
        # _visualization_loop along with the rest of the per-frame publish work; see
        # its "colour" fields below.
        color_needed = self.pub_color.get_subscription_count() > 0

        # 1+2. Tiled inference + per-tile post-processing, merged (tiled_inference.py)
        try:
            detections, inference_ms, n_tiles = run_tiled_inference(
                self.inferencer,
                cv_img,
                imgsz=self.imgsz,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                num_classes=4,
                class_names=CLASS_NAMES,
                decode_segmentation=self.publish_masks,
                tile_size=self.tile_size,
                tile_overlap=self.tile_overlap,
            )
        except Exception as e:
            self.get_logger().error(f"Tiled inference failed: {e}")
            return

        # 3. ByteTrack update (runs even if detections is empty)
        try:
            detections = self.tracker.update(detections)
        except Exception as e:
            self.get_logger().error(f"Tracking update failed: {e}")
            detections = []

        # 4. Depth estimation for current detections (at candidate filtered root location)
        depth_estimates = {}
        for det in detections:
            t_id = getattr(det, "track_id", None)
            if t_id is None or t_id < 0:
                continue

            cand_root = self.state_manager.predict_filtered_root(t_id, det.root_point)
            det._cand_root = cand_root

            rx, ry = cand_root[0], cand_root[1]
            if depth_img is not None:
                d_est = estimate_root_depth(
                    depth_img, rx, ry,
                    window_size=self.depth_window_size,
                    minimum_depth_valid_ratio=self.minimum_depth_valid_ratio,
                    minimum_depth_m=self.minimum_depth_m,
                    maximum_depth_m=self.maximum_depth_m,
                    depth_mad_multiplier=self.depth_mad_multiplier,
                    encoding="16UC1",
                )
            else:
                d_est = DepthEstimate(depth_m=-1.0, valid=False, valid_ratio=0.0, std_m=0.0, sample_count=0)

            depth_estimates[t_id] = d_est
            det.root_d = d_est.depth_m if d_est.valid else -1.0

        # 5. TrackStateManager update (maintains lifecycle across misses)
        active_states = self.state_manager.update(detections, depth_estimates)

        # 6. Sort active states by filtered root_x in ascending order
        active_states.sort(key=lambda s: s.root_point[0])

        # 7. Build and publish TrackedPlantArray ROS message
        if self.pub_tracked_plants is not None and TrackedPlantArray is not None:
            try:
                plant_msgs = []
                for st in active_states:
                    if st.track_id < 0:
                        continue  # Never publish invalid tracking IDs

                    p_msg = TrackedPlant()
                    p_msg.header = header
                    p_msg.track_id = int(st.track_id)
                    p_msg.class_id = int(st.class_id)
                    p_msg.class_name = str(st.class_name)
                    p_msg.confidence = float(st.confidence)
                    p_msg.x1 = float(st.bbox_xyxy[0])
                    p_msg.y1 = float(st.bbox_xyxy[1])
                    p_msg.x2 = float(st.bbox_xyxy[2])
                    p_msg.y2 = float(st.bbox_xyxy[3])
                    p_msg.root_x = float(st.root_point[0])
                    p_msg.root_y = float(st.root_point[1])
                    p_msg.root_d = float(st.depth_m)

                    p_msg.track_state = int(st.track_state)
                    p_msg.is_confirmed = bool(st.is_confirmed)
                    p_msg.is_currently_detected = bool(st.is_currently_detected)
                    p_msg.actionable = bool(st.actionable)

                    p_msg.age_frames = int(st.age_frames)
                    p_msg.hit_count = int(st.hit_count)
                    p_msg.consecutive_hits = int(st.consecutive_hits)
                    p_msg.missed_frames = int(st.missed_frames)

                    p_msg.depth_valid = bool(st.depth_valid)
                    p_msg.current_depth_valid = bool(st.current_depth_valid)
                    p_msg.depth_valid_ratio = float(st.depth_valid_ratio)
                    p_msg.depth_std_m = float(st.depth_std_m)

                    if self.publish_masks and st.mask is not None and st.is_currently_detected:
                        try:
                            mask_msg = self.bridge.cv2_to_imgmsg(st.mask, encoding="mono8")
                            mask_msg.header = header
                            p_msg.mask = mask_msg
                        except Exception:
                            p_msg.mask = Image()
                    else:
                        p_msg.mask = Image()

                    plant_msgs.append(p_msg)

                array_msg = TrackedPlantArray()
                array_msg.header = header
                array_msg.plants = plant_msgs
                self.pub_tracked_plants.publish(array_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish TrackedPlantArray: {e}")

        # 8. Optional visualization + raw color publish — both handed off to
        # _visualization_loop (its own thread) so neither a debug viewer nor
        # another node subscribed to the raw color feed (e.g. weed_bridge_node's
        # EE-marker detector) can throttle this method's own rate.
        viz_needed = False
        if self.enable_visualization:
            sub_raw_count = self.pub_visualization.get_subscription_count()
            sub_comp_count = self.pub_visualization_compressed.get_subscription_count()
            viz_needed = sub_raw_count > 0 or sub_comp_count > 0

        if viz_needed or color_needed:
            try:
                self._vis_queue.put_nowait(
                    (cv_img, active_states, fps, inference_ms, header, viz_needed, color_needed)
                )
            except queue.Full:
                pass  # visualization thread is behind — drop this frame's viz/color, not a real one

    def _make_header(self):
        from std_msgs.msg import Header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "camera_color_optical_frame"
        return header

    def _build_camera_info_msg(self, width, height):
        """Build a static CameraInfo template from the camera_fx/fy/cx/cy
        parameters (only K/width/height are populated — plenty for
        downstream pixel<->camera-frame projection; header is filled in
        per-publish in process_frame).

        Unlike librealsense2, a plain UVC camera exposes no hardware
        intrinsics — camera_fx/fy/cx/cy default to a rough placeholder and
        MUST be overridden with values from a real chessboard calibration
        of this camera/lens for accurate pixel_to_camera_xyz_mm() results.

        width/height here are the POST-crop dimensions actually published
        (see frame_crop_height) — camera_cy is shifted by crop_top to match,
        since the vertical crop moves the frame's own origin down by that
        many rows. camera_fx/fy/cx/cy themselves are still whatever
        resolution they were calibrated at (see camera_fx's own parameter
        comment) — this shift does NOT re-derive them for a different
        capture resolution.
        """
        msg = CameraInfo()
        msg.width = width
        msg.height = height
        msg.distortion_model = "plumb_bob"
        msg.d = list(self.camera_dist)
        cy = self.camera_cy - self.crop_top
        msg.k = [
            self.camera_fx, 0.0, self.camera_cx,
            0.0, self.camera_fy, cy,
            0.0, 0.0, 1.0,
        ]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [
            self.camera_fx, 0.0, self.camera_cx, 0.0,
            0.0, self.camera_fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        return msg

    def _check_heartbeat(self):
        if time.time() - self.last_capture_time > 5.0:
            self.get_logger().warn("No frames captured from the UVC camera in 5s.")

    def _resolve_model_path(self, path_str):
        import os
        if not path_str:
            return path_str
        if os.path.isabs(path_str) and os.path.exists(path_str):
            return path_str
        model_filename = os.path.basename(path_str)

        try:
            from ament_index_python.packages import get_package_share_directory
            share_model = os.path.join(get_package_share_directory("plant_perception"), "weight", model_filename)
            if os.path.exists(share_model):
                return os.path.abspath(share_model)
        except Exception:
            pass

        curr = os.path.dirname(os.path.abspath(__file__))
        for _ in range(5):
            c1 = os.path.join(curr, "src", "weight", model_filename)
            c2 = os.path.join(curr, "weight", model_filename)
            if os.path.exists(c1):
                return os.path.abspath(c1)
            if os.path.exists(c2):
                return os.path.abspath(c2)
            parent = os.path.dirname(curr)
            if parent == curr:
                break
            curr = parent

        return path_str

    @staticmethod
    def _best_effort_qos():
        return QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

    @staticmethod
    def _reliable_qos():
        return QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

    def destroy_node(self):
        self._stop_event.set()
        try:
            self._gst_pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MergeCodeTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
