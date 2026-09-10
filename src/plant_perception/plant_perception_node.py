#!/usr/bin/env python3
"""
plant_perception_node.py — ROS 2 node for YOLO-Seg-Root ONNX inference,
object tracking, persistent track state management, depth estimation, and visualization.
"""

import os
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import message_filters
import cv2

from delta_common import config as delta_config

try:
    from plant_perception.msg import TrackedPlant, TrackedPlantArray
except ImportError:
    print("[plant_perception_node] ERROR: Custom messages not compiled or not in path.")
    TrackedPlant = None
    TrackedPlantArray = None

try:
    from trt_inferencer import TRTInferencer
    from tiled_inference import run_tiled_inference
    from tracker import Tracker
    from track_state import TrackStateManager, PlantTrackState
    from depth_utils import estimate_root_depth, DepthEstimate
    from visualization import draw_visualizations
except ImportError:
    try:
        from .trt_inferencer import TRTInferencer
        from .tiled_inference import run_tiled_inference
        from .tracker import Tracker
        from .track_state import TrackStateManager, PlantTrackState
        from .depth_utils import estimate_root_depth, DepthEstimate
        from .visualization import draw_visualizations
    except ImportError:
        from .onnx_inferencer import ONNXInferencer as TRTInferencer
        from .tiled_inference import run_tiled_inference
        from .tracker import Tracker
        from .track_state import TrackStateManager, PlantTrackState
        from .depth_utils import estimate_root_depth, DepthEstimate
        from .visualization import draw_visualizations

CLASS_NAMES = {
    0: "crop_small_leaf",
    1: "crop_large_leaf",
    2: "weed_small_leaf",
    3: "weed_large_leaf",
}


class PlantPerceptionNode(Node):
    """ROS 2 node for real-time plant perception with persistent track state lifecycle."""

    def __init__(self):
        super().__init__("plant_perception_node")
        self.get_logger().info("Initializing plant_perception_node (TensorRT)...")

        # Model parameters
        self.declare_parameter("model_path", "src/weight/best_fp16.engine")
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
        # tiled_inference.py / perception.yaml for the full story).
        self.declare_parameter("tile_size", 1088)
        self.declare_parameter("tile_overlap", 96)

        # Tracker parameters
        self.declare_parameter("tracker_type", "bytetrack")
        self.declare_parameter("track_high_thresh", 0.35)
        self.declare_parameter("track_low_thresh", 0.10)
        self.declare_parameter("new_track_thresh", 0.40)
        self.declare_parameter("track_buffer", 15)
        self.declare_parameter("match_thresh", 0.75)
        self.declare_parameter("frame_rate", 15)

        # Track stability parameters
        self.declare_parameter("minimum_confirm_hits", 3)
        self.declare_parameter("maximum_publish_misses", 2)
        self.declare_parameter("maximum_track_misses", 15)
        self.declare_parameter("bbox_ema_alpha", 0.50)
        self.declare_parameter("root_ema_alpha", 0.35)
        self.declare_parameter("class_history_size", 5)
        self.declare_parameter("depth_history_size", 5)

        # Topic parameters
        self.declare_parameter("color_image_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_image_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("tracked_plants_topic", "/plant_perception/tracked_plants")
        self.declare_parameter("visualization_topic", "/plant_perception/visualization")

        # Segmentation parameters
        self.declare_parameter("publish_masks", False)

        # Depth parameters
        self.declare_parameter("use_depth", True)
        self.declare_parameter("depth_window_size", 7)
        self.declare_parameter("minimum_depth_valid_ratio", 0.40)
        self.declare_parameter("minimum_depth_m", 0.10)
        self.declare_parameter("maximum_depth_m", 3.00)
        self.declare_parameter("depth_mad_multiplier", 2.5)

        # Synchronization parameters
        self.declare_parameter("sync_slop_seconds", 0.05)
        self.declare_parameter("sync_queue_size", 10)

        # Visualization parameters
        self.declare_parameter("enable_visualization", True)
        self.declare_parameter("visualization_show_masks", False)
        self.declare_parameter("visualization_jpeg_quality", 80)
        # Downscales the annotated frame before encode/publish — see
        # merge_code_test.py's visualization_max_width comment: cv2.imencode
        # of a full 1920x1080 frame measured 28.3ms vs 6.0ms at 960x540.
        # 0 disables downscaling. Unlike merge_code_test.py's dedicated
        # visualization thread, this stays inline in sync_callback — this
        # node is driven by rclpy's subscriber callback, not a capture loop,
        # so decoupling it needs a bigger restructure than this pass covers.
        self.declare_parameter("visualization_max_width", 960)

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

        self.color_topic = str(self.get_parameter("color_image_topic").value)
        self.depth_topic = str(self.get_parameter("depth_image_topic").value)
        self.tracked_plants_topic = str(self.get_parameter("tracked_plants_topic").value)
        self.visualization_topic = str(self.get_parameter("visualization_topic").value)

        self.publish_masks = bool(self.get_parameter("publish_masks").value)
        self.use_depth = bool(self.get_parameter("use_depth").value)
        self.depth_window_size = int(self.get_parameter("depth_window_size").value)
        if self.depth_window_size <= 0:
            raise ValueError("depth_window_size must be positive")
        if self.depth_window_size % 2 == 0:
            self.depth_window_size += 1
            self.get_logger().warn(f"Even depth_window_size converted to odd: {self.depth_window_size}")

        self.minimum_depth_valid_ratio = float(self.get_parameter("minimum_depth_valid_ratio").value)
        self.minimum_depth_m = float(self.get_parameter("minimum_depth_m").value)
        self.maximum_depth_m = float(self.get_parameter("maximum_depth_m").value)
        self.depth_mad_multiplier = float(self.get_parameter("depth_mad_multiplier").value)

        self.sync_slop_seconds = float(self.get_parameter("sync_slop_seconds").value)
        self.sync_queue_size = int(self.get_parameter("sync_queue_size").value)

        self.enable_visualization = bool(self.get_parameter("enable_visualization").value)
        self.visualization_show_masks = bool(self.get_parameter("visualization_show_masks").value)
        self.visualization_jpeg_quality = int(self.get_parameter("visualization_jpeg_quality").value)
        self.visualization_max_width = int(self.get_parameter("visualization_max_width").value)

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

        # Publishers
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

        # Frame timing
        self.frame_times = []
        self.last_infer_time = 0.0
        self.last_proc_time = time.time()

        # Subscribers
        if self.use_depth:
            self.get_logger().info(f"Subscribing to synchronized RGB-D topics: {self.color_topic} & {self.depth_topic}")
            self.sub_color = message_filters.Subscriber(
                self, Image, self.color_topic, qos_profile=self._best_effort_qos()
            )
            self.sub_depth = message_filters.Subscriber(
                self, Image, self.depth_topic, qos_profile=self._best_effort_qos()
            )
            self.sync = message_filters.ApproximateTimeSynchronizer(
                [self.sub_color, self.sub_depth],
                queue_size=self.sync_queue_size,
                slop=self.sync_slop_seconds
            )
            self.sync.registerCallback(self.sync_callback)
        else:
            self.get_logger().info(f"Subscribing to color-only topic: {self.color_topic}")
            self.sub_color = self.create_subscription(
                Image, self.color_topic, self.color_callback, self._best_effort_qos()
            )

        self.timer_diag = self.create_timer(5.0, self._check_heartbeat)
        self.get_logger().info("plant_perception_node startup complete.")

    def color_callback(self, color_msg):
        self.sync_callback(color_msg, None)

    def sync_callback(self, color_msg, depth_msg):
        """
        Processing pipeline:
        RGB & depth sync → ONNX inference → optional segmentation → ByteTrack →
        depth estimation → TrackStateManager update → sort by root_x → publish → optional visualization
        """
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

        # 1. Image conversion
        try:
            cv_img = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth_img = (self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                         if depth_msg is not None else None)
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")
            return

        # 2+3. Tiled inference + per-tile post-processing, merged (tiled_inference.py)
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

        # 4. ByteTrack update (Runs even if detections is empty)
        try:
            detections = self.tracker.update(detections)
        except Exception as e:
            self.get_logger().error(f"Tracking update failed: {e}")
            detections = []

        # 5. Depth estimation for current detections (at candidate filtered root location)
        depth_estimates = {}
        encoding = depth_msg.encoding if depth_msg is not None else "16UC1"

        for det in detections:
            t_id = getattr(det, "track_id", None)
            if t_id is None or t_id < 0:
                continue

            # Candidate filtered root location prior to depth ROI extraction
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
                    encoding=encoding
                )
            else:
                d_est = DepthEstimate(depth_m=-1.0, valid=False, valid_ratio=0.0, std_m=0.0, sample_count=0)

            depth_estimates[t_id] = d_est
            det.root_d = d_est.depth_m if d_est.valid else -1.0

        # 6. TrackStateManager update (Maintains lifecycle across misses)
        active_states = self.state_manager.update(detections, depth_estimates)

        # 7. Sort active states by filtered root_x in ascending order
        active_states.sort(key=lambda s: s.root_point[0])

        # 8. Build and publish TrackedPlantArray ROS message
        if self.pub_tracked_plants is not None and TrackedPlantArray is not None:
            try:
                plant_msgs = []
                for st in active_states:
                    if st.track_id < 0:
                        continue  # Never publish invalid tracking IDs

                    p_msg = TrackedPlant()
                    p_msg.header = color_msg.header
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
                            mask_msg.header = color_msg.header
                            p_msg.mask = mask_msg
                        except Exception:
                            p_msg.mask = Image()
                    else:
                        p_msg.mask = Image()

                    plant_msgs.append(p_msg)

                array_msg = TrackedPlantArray()
                array_msg.header = color_msg.header
                array_msg.plants = plant_msgs
                self.pub_tracked_plants.publish(array_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish TrackedPlantArray: {e}")

        # 9. Optional Visualization
        if self.enable_visualization:
            sub_raw_count = self.pub_visualization.get_subscription_count()
            sub_comp_count = self.pub_visualization_compressed.get_subscription_count()

            if sub_raw_count > 0 or sub_comp_count > 0:
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

                    if sub_raw_count > 0:
                        vis_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding="bgr8")
                        vis_msg.header = color_msg.header
                        self.pub_visualization.publish(vis_msg)

                    if sub_comp_count > 0:
                        compressed_msg = CompressedImage()
                        compressed_msg.header = color_msg.header
                        compressed_msg.format = "jpeg"
                        quality = [int(cv2.IMWRITE_JPEG_QUALITY), self.visualization_jpeg_quality]
                        ret, jpeg_data = cv2.imencode(".jpg", annotated_img, quality)
                        if ret:
                            compressed_msg.data = jpeg_data.tobytes()
                            self.pub_visualization_compressed.publish(compressed_msg)
                except Exception as e:
                    self.get_logger().error(f"Visualization publishing failed: {e}")

    def _check_heartbeat(self):
        if time.time() - self.last_proc_time > 5.0:
            if self.use_depth:
                self.get_logger().warn(
                    f"No synchronized RGB-D frames received on {self.color_topic} / {self.depth_topic} in 5s."
                )
            else:
                self.get_logger().warn(f"No color frames received on {self.color_topic} in 5s.")

    def _resolve_model_path(self, path_str):
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


def main(args=None):
    rclpy.init(args=args)
    node = PlantPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
