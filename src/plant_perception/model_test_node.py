#!/usr/bin/env python3
"""
model_test_node.py — Run a TensorRT engine against a recorded video file
instead of a live camera, to eyeball a candidate model before wiring it into
merge_code_test.py / plant_perception_node.py.

Reuses the same TRTInferencer + tiled_inference.run_tiled_inference +
visualization pipeline as merge_code_test.py, unchanged — only the frame
source differs (cv2.VideoCapture on a file instead of a live camera feed).
No tracker, no depth, no CAN/motor involvement — detection only, so a
video's raw per-frame detections are what gets shown (no track smoothing to
mask a bad model).

Default video/model match the 2026-09-10 check: 1.mp4 is 1280x720, same as
the weed_pick_place.launch.py global-shutter camera pipeline, so the default
tile_size=640/tile_overlap=0/frame_crop_height=640 reproduces that pipeline's
tiling exactly (see perception_direct.yaml's tile_size comment).
"""
import os
import time
from collections import defaultdict

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

try:
    from trt_inferencer import TRTInferencer
    from tiled_inference import run_tiled_inference
    from visualization import draw_visualizations
except ImportError:
    from .trt_inferencer import TRTInferencer
    from .tiled_inference import run_tiled_inference
    from .visualization import draw_visualizations

CLASS_NAMES = {
    0: "crop_small_leaf",
    1: "crop_large_leaf",
    2: "weed_small_leaf",
    3: "weed_large_leaf",
}


class ModelTestNode(Node):
    """One-shot test harness: decode a video file, run tiled TensorRT
    inference on every frame, draw + publish/show the result, and log
    per-class detection counts so a candidate engine can be compared against
    the one currently wired into the live pipeline."""

    def __init__(self):
        super().__init__("model_test_node")

        self.declare_parameter("video_path", "/home/panha/DeltaRobot_copy/1.mp4")
        self.declare_parameter("model_path", "src/weight/best_fp16__copy.engine")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf_thres", 0.25)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("num_classes", 4)
        self.declare_parameter("device", "tensorrt")

        # Same tiling scheme as weed_pick_place.launch.py's live pipeline —
        # see perception_direct.yaml's tile_size/frame_crop_height comments.
        self.declare_parameter("tile_size", 640)
        self.declare_parameter("tile_overlap", 0)
        self.declare_parameter("frame_crop_height", 640)

        self.declare_parameter("loop_video", True)
        self.declare_parameter("max_frames", 0)  # 0 = no limit
        # 0 = run flat-out (best for benchmarking raw FPS); >0 throttles
        # playback to that rate for a more natural-looking preview.
        self.declare_parameter("playback_fps_limit", 0.0)
        self.declare_parameter("show_window", True)
        self.declare_parameter("publish_visualization", True)
        self.declare_parameter("visualization_topic", "/plant_perception/model_test/visualization")
        self.declare_parameter("log_every_n_frames", 30)

        self.video_path = str(self.get_parameter("video_path").value)
        self.model_path = self._resolve_model_path(self.get_parameter("model_path").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf_thres = float(self.get_parameter("conf_thres").value)
        self.iou_thres = float(self.get_parameter("iou_thres").value)
        self.num_classes = int(self.get_parameter("num_classes").value)
        self.device = str(self.get_parameter("device").value)

        self.tile_size = int(self.get_parameter("tile_size").value)
        self.tile_overlap = int(self.get_parameter("tile_overlap").value)
        self.frame_crop_height = int(self.get_parameter("frame_crop_height").value)

        self.loop_video = bool(self.get_parameter("loop_video").value)
        self.max_frames = int(self.get_parameter("max_frames").value)
        self.playback_fps_limit = float(self.get_parameter("playback_fps_limit").value)
        self.show_window = bool(self.get_parameter("show_window").value) and bool(os.environ.get("DISPLAY"))
        if bool(self.get_parameter("show_window").value) and not os.environ.get("DISPLAY"):
            self.get_logger().warn("show_window requested but $DISPLAY is unset — running headless (publish/log only).")
        self.publish_visualization = bool(self.get_parameter("publish_visualization").value)
        self.visualization_topic = str(self.get_parameter("visualization_topic").value)
        self.log_every_n_frames = max(1, int(self.get_parameter("log_every_n_frames").value))

        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.get_logger().info(f"Loading engine: {self.model_path}")
        self.inferencer = TRTInferencer(
            model_path=self.model_path,
            device=self.device,
            imgsz=self.imgsz,
            trt_fp16_enable=True,
        )

        self.bridge = CvBridge()
        self.vis_pub = None
        if self.publish_visualization:
            self.vis_pub = self.create_publisher(Image, self.visualization_topic, 10)

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.get_logger().info(f"Video: {self.video_path}  {src_w}x{src_h}")

        if self.frame_crop_height > 0 and self.frame_crop_height < src_h:
            self.crop_top = (src_h - self.frame_crop_height) // 2
            self.cropped_height = self.frame_crop_height
        else:
            self.crop_top = 0
            self.cropped_height = src_h

        self.class_counts = defaultdict(int)
        self.frame_idx = 0
        self.total_inference_ms = 0.0
        self.run_start = time.time()

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

    def run(self):
        frame_delay = 1.0 / self.playback_fps_limit if self.playback_fps_limit > 0 else 0.0

        while rclpy.ok():
            loop_t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                if self.loop_video:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            if self.crop_top > 0:
                frame = frame[self.crop_top:self.crop_top + self.cropped_height, :]

            detections, inference_ms, n_tiles = run_tiled_inference(
                self.inferencer,
                frame,
                imgsz=self.imgsz,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                num_classes=self.num_classes,
                class_names=CLASS_NAMES,
                decode_segmentation=False,
                tile_size=self.tile_size,
                tile_overlap=self.tile_overlap,
            )

            self.frame_idx += 1
            self.total_inference_ms += inference_ms
            for det in detections:
                self.class_counts[det.class_name] += 1

            fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
            vis = draw_visualizations(
                frame, detections,
                fps=fps, latency_ms=inference_ms, device=self.device,
            )

            if self.frame_idx % self.log_every_n_frames == 0:
                counts_str = ", ".join(f"{k}={v}" for k, v in sorted(self.class_counts.items())) or "none yet"
                self.get_logger().info(
                    f"frame {self.frame_idx}: {len(detections)} dets this frame "
                    f"({n_tiles} tiles, {inference_ms:.1f}ms) | cumulative: {counts_str}"
                )

            if self.vis_pub is not None:
                # "passthrough" (not "bgr8") — matches merge_code_test.py;
                # some cv_bridge/numpy combos raise KeyError inside
                # cv2_to_imgmsg's dtype check when an explicit encoding is
                # given even though the array genuinely is BGR8.
                msg = self.bridge.cv2_to_imgmsg(vis, encoding="passthrough")
                msg.header.stamp = self.get_clock().now().to_msg()
                self.vis_pub.publish(msg)

            if self.show_window:
                cv2.imshow("model_test_node — press 'q' to quit", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if self.max_frames > 0 and self.frame_idx >= self.max_frames:
                break

            if frame_delay > 0:
                elapsed = time.time() - loop_t0
                remaining = frame_delay - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        self._log_summary()

    def _log_summary(self):
        elapsed = time.time() - self.run_start
        avg_ms = self.total_inference_ms / self.frame_idx if self.frame_idx else 0.0
        counts_str = ", ".join(f"{k}={v}" for k, v in sorted(self.class_counts.items())) or "none"
        self.get_logger().info(
            f"=== Done: {self.frame_idx} frames in {elapsed:.1f}s "
            f"(avg inference {avg_ms:.1f}ms/frame = {1000.0/avg_ms if avg_ms else 0:.1f} FPS) ==="
        )
        self.get_logger().info(f"=== Total detections by class: {counts_str} ===")

    def destroy_node(self):
        self.cap.release()
        if self.show_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ModelTestNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
