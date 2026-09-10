"""Standalone replacement for realsense2_camera_node — publishes color +
camera_info from a UVC/V4L2 global-shutter camera on the same topics the
RealSense ROS driver used, so downstream nodes (camera_system.py,
pick_place_node.py, weed_bridge_node.py, RViz) need no changes.

This camera has no depth stream, so /camera/camera/aligned_depth_to_color/
image_raw is simply never published — consumers already handle a missing
depth image (FAKE_DEPTH_ENABLE is on by default in delta_common/config.py).
"""
import array
import threading

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class UvcCameraPublisher(Node):
    def __init__(self):
        super().__init__('uvc_camera_publisher')

        self.declare_parameter('camera_device', '/dev/video1')
        self.declare_parameter('camera_fourcc', 'MJPG')
        self.declare_parameter('camera_width', 640)
        self.declare_parameter('camera_height', 480)
        self.declare_parameter('camera_fps', 30)
        # No hardware intrinsics API on a plain UVC camera — these MUST be
        # replaced with real values from a chessboard calibration of this
        # camera/lens before relying on downstream pixel<->mm projection.
        self.declare_parameter('camera_fx', 600.0)
        self.declare_parameter('camera_fy', 600.0)
        self.declare_parameter('camera_cx', 320.0)
        self.declare_parameter('camera_cy', 240.0)
        self.declare_parameter('camera_dist_k1', 0.0)
        self.declare_parameter('camera_dist_k2', 0.0)
        self.declare_parameter('camera_dist_p1', 0.0)
        self.declare_parameter('camera_dist_p2', 0.0)
        self.declare_parameter('camera_dist_k3', 0.0)

        camera_device = str(self.get_parameter('camera_device').value)
        camera_fourcc = str(self.get_parameter('camera_fourcc').value)
        self._width = int(self.get_parameter('camera_width').value)
        self._height = int(self.get_parameter('camera_height').value)
        self._fps = int(self.get_parameter('camera_fps').value)

        self.pub_color = self.create_publisher(Image, '/camera/camera/color/image_raw', 1)
        self.pub_info = self.create_publisher(CameraInfo, '/camera/camera/color/camera_info', 1)

        self._cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f'Could not open camera device {camera_device!r}')
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera_fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        # NOTE: do NOT set CAP_PROP_BUFFERSIZE=1 here — on this camera/OpenCV
        # combo it halves real capture throughput (measured 30fps -> 15fps)
        # because the V4L2 backend has to wait for a buffer to be dequeued
        # before it can requeue it. It bought no latency benefit anyway:
        # _capture_loop already reads continuously in its own thread and
        # the publisher itself uses queue depth 1, so downstream never sees
        # a stale backlog.

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self._width
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self._height
        self._camera_info = self._build_camera_info(actual_w, actual_h)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f'UVC camera publisher started: {camera_device} requested '
            f'{self._width}x{self._height}@{self._fps}fps ({camera_fourcc}), '
            f'got {actual_w}x{actual_h}'
        )

    def _capture_loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self.get_logger().warn('UVC camera frame grab failed, retrying...')
                continue

            stamp = self.get_clock().now().to_msg()
            self.pub_color.publish(self._to_image_msg(frame, 'bgr8', stamp))
            self._camera_info.header.stamp = stamp
            self.pub_info.publish(self._camera_info)

    def _to_image_msg(self, arr: np.ndarray, encoding: str, stamp: Time) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = 'camera_color_optical_frame'
        msg.height, msg.width = arr.shape[:2]
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = arr.strides[0]
        # rosidl's data.setter only takes the fast path for array.array —
        # assigning bytes/ndarray falls through to a pure-Python per-element
        # isinstance/range check over every byte (measured ~170ms for a
        # 640x480x3 frame vs ~0.1ms via array.array).
        msg.data = array.array('B', arr.tobytes())
        return msg

    def _build_camera_info(self, width: int, height: int) -> CameraInfo:
        fx = float(self.get_parameter('camera_fx').value)
        fy = float(self.get_parameter('camera_fy').value)
        cx = float(self.get_parameter('camera_cx').value)
        cy = float(self.get_parameter('camera_cy').value)
        dist = [
            float(self.get_parameter('camera_dist_k1').value),
            float(self.get_parameter('camera_dist_k2').value),
            float(self.get_parameter('camera_dist_p1').value),
            float(self.get_parameter('camera_dist_p2').value),
            float(self.get_parameter('camera_dist_k3').value),
        ]

        msg = CameraInfo()
        msg.header.frame_id = 'camera_color_optical_frame'
        msg.width = width
        msg.height = height
        msg.distortion_model = 'plumb_bob'
        msg.d = dist
        msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    def destroy_node(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self._cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UvcCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
