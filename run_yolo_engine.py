#!/usr/bin/env python3
"""
Fair benchmark script for YOLOv11-Seg-Root TensorRT engines.
Designed to be run on both Jetson AGX Xavier and Jetson Orin Nano
with the exact same .engine file.
"""

import argparse
import time
import os
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

# ---------------------------------------------------------------------------
# Config defaults (same on both platforms)
# ---------------------------------------------------------------------------
DEFAULT_ENGINE = "best.engine"
DEFAULT_VIDEO  = "1.mp4"
IMGSZ          = 640
CONF_THRES     = 0.25
IOU_THRES      = 0.45
NUM_CLASSES    = 4
NUM_MASK_COEFF = 32          # skip these

CLASS_NAMES = {
    0: "crop_small_leaf",
    1: "crop_large_leaf",
    2: "weed_small_leaf",
    3: "weed_large_leaf",
}
CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 150, 0),
    2: (0, 0, 255),
    3: (0, 0, 150),
}
ROOT_COLOR = (255, 0, 255)


# ---------------------------------------------------------------------------
# TensorRT Inferencer
# ---------------------------------------------------------------------------
class TRTInferencer:
    def __init__(self, engine_path, imgsz=640):
        self.imgsz = imgsz
        logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_name = None
        self.output_names = []
        self.host_inputs = []
        self.host_outputs = []
        self.device_inputs = []
        self.device_outputs = []
        self.bindings = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.engine.get_tensor_shape(name))
            size = int(np.prod(shape))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(device_mem)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
                print(f"  INPUT  {name}: {shape}")
            else:
                self.output_names.append(name)
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)
                print(f"  OUTPUT {name}: {shape}")

        self._rgb_buf = np.empty((imgsz, imgsz, 3), dtype=np.uint8)
        self._input_buf = np.empty((1, 3, imgsz, imgsz), dtype=np.float32)
        print(f"[TRT] Ready on CUDA")

    def letterbox(self, img, color=(114, 114, 114)):
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw = (self.imgsz - new_unpad[0]) / 2
        dh = (self.imgsz - new_unpad[1]) / 2
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        return cv2.copyMakeBorder(img, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=color), r, (dw, dh)

    def preprocess(self, img_bgr):
        img_lb, ratio, pad = self.letterbox(img_bgr)
        cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB, dst=self._rgb_buf)
        self._input_buf[0] = self._rgb_buf.transpose(2, 0, 1)
        self._input_buf /= 255.0
        return self._input_buf, ratio, pad

    def run(self, img_bgr):
        blob, ratio, pad = self.preprocess(img_bgr)
        np.copyto(self.host_inputs[0], blob.ravel())
        cuda.memcpy_htod_async(self.device_inputs[0], self.host_inputs[0], self.stream)

        for i, name in enumerate([self.input_name] + self.output_names):
            self.context.set_tensor_address(name, int(self.bindings[i]))

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        for h, d in zip(self.host_outputs, self.device_outputs):
            cuda.memcpy_dtoh_async(h, d, self.stream)
        self.stream.synchronize()

        outputs = []
        for i, name in enumerate(self.output_names):
            shape = tuple(self.engine.get_tensor_shape(name))
            outputs.append(self.host_outputs[i].reshape(shape))
        return outputs, ratio, pad


# ---------------------------------------------------------------------------
# Post-process (identical logic on both platforms)
# ---------------------------------------------------------------------------
def postprocess(outputs, orig_shape, input_shape,
                conf_thres=CONF_THRES, iou_thres=IOU_THRES, num_classes=NUM_CLASSES):
    # Find the detection tensor (shape ending with 8400)
    det = None
    for o in outputs:
        if o.ndim == 3 and o.shape[-1] == 8400:
            det = o
            break
        if o.ndim == 3 and o.shape[1] == 8400:
            det = o.transpose(0, 2, 1)
            break
    if det is None:
        raise RuntimeError(f"No detection output found. Shapes: {[o.shape for o in outputs]}")

    preds = np.squeeze(det).T                         # (8400, 42)
    boxes_cxcywh = preds[:, :4]
    class_scores = preds[:, 4:4 + num_classes]
    root_xy = preds[:, 4 + num_classes + NUM_MASK_COEFF :
                       4 + num_classes + NUM_MASK_COEFF + 2]

    class_ids = np.argmax(class_scores, axis=1)
    confs = class_scores[np.arange(len(class_scores)), class_ids]

    keep = confs > conf_thres
    if not np.any(keep):
        return [], [], [], []

    boxes_cxcywh = boxes_cxcywh[keep]
    confs = confs[keep]
    class_ids = class_ids[keep]
    root_xy = root_xy[keep]

    cx, cy, w, h = boxes_cxcywh.T
    x1 = cx - w / 2
    y1 = cy - h / 2
    boxes_xyxy = np.stack([x1, y1, x1 + w, y1 + h], axis=1)

    # NMS
    nms_boxes = np.stack([x1, y1, w, h], axis=1).tolist()
    idxs = cv2.dnn.NMSBoxes(nms_boxes, confs.tolist(), conf_thres, iou_thres)
    idxs = np.array(idxs).flatten() if len(idxs) else np.array([], dtype=int)
    if len(idxs) == 0:
        return [], [], [], []

    boxes_xyxy = boxes_xyxy[idxs]
    confs = confs[idxs]
    class_ids = class_ids[idxs]
    root_xy = root_xy[idxs]

    # Scale back to original image
    h0, w0 = orig_shape
    ih, iw = input_shape
    gain = min(iw / w0, ih / h0)
    pad_w = (iw - w0 * gain) / 2
    pad_h = (ih - h0 * gain) / 2

    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / gain
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)

    roots = root_xy.copy()
    roots[:, 0] = ((roots[:, 0] - pad_w) / gain).clip(0, w0)
    roots[:, 1] = ((roots[:, 1] - pad_h) / gain).clip(0, h0)

    return boxes.astype(int), confs, class_ids.astype(int), roots.astype(int)


def draw(frame, boxes, confs, cls_ids, roots):
    out = frame.copy()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        color = CLASS_COLORS.get(int(cls_ids[i]), (255, 255, 255))
        label = f"{CLASS_NAMES.get(int(cls_ids[i]), cls_ids[i])} {confs[i]:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    h, w = out.shape[:2]
    for rx, ry in roots:
        if 0 <= rx < w and 0 <= ry < h:
            cv2.circle(out, (rx, ry), 5, ROOT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(out, (rx, ry), 7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_ENGINE)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--conf", type=float, default=CONF_THRES)
    parser.add_argument("--iou", type=float, default=IOU_THRES)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--print-every", type=int, default=30)
    args = parser.parse_args()

    print(f"Loading engine: {args.model}")
    inferencer = TRTInferencer(args.model, imgsz=IMGSZ)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {src_fps:.1f} fps, {total} frames")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, src_fps, (w, h))

    display = not args.no_display
    if display:
        try:
            cv2.namedWindow("YOLO-Seg-Root", cv2.WINDOW_NORMAL)
        except cv2.error:
            display = False
            print("No display available – running headless")

    # Warm-up
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        _ = inferencer.run(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    t_start = time.time()
    pure_infer_ms = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            t0 = time.time()
            outputs, _, _ = inferencer.run(frame)
            t1 = time.time()
            pure_infer_ms.append((t1 - t0) * 1000)

            boxes, confs, cls_ids, roots = postprocess(
                outputs, frame.shape[:2], (IMGSZ, IMGSZ),
                conf_thres=args.conf, iou_thres=args.iou
            )

            annotated = draw(frame, boxes, confs, cls_ids, roots) if len(boxes) else frame

            # FPS overlay
            elapsed = time.time() - t_start
            fps = frame_idx / elapsed if elapsed > 0 else 0
            cv2.putText(annotated, f"{fps:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            if writer:
                writer.write(annotated)

            if display:
                cv2.imshow("YOLO-Seg-Root", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.print_every and frame_idx % args.print_every == 0:
                avg_infer = np.mean(pure_infer_ms[-args.print_every:])
                print(f"frame {frame_idx}/{total} | "
                      f"end-to-end {fps:.1f} FPS | "
                      f"pure infer {avg_infer:.1f} ms ({1000/avg_infer:.1f} FPS)")
    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    total_time = time.time() - t_start
    avg_fps = frame_idx / total_time if total_time > 0 else 0
    avg_infer_ms = np.mean(pure_infer_ms) if pure_infer_ms else 0

    print("\n========== BENCHMARK SUMMARY ==========")
    print(f"Platform          : (fill in: AGX Xavier / Orin Nano)")
    print(f"Engine            : {args.model}")
    print(f"Frames processed  : {frame_idx}")
    print(f"Total time        : {total_time:.2f} s")
    print(f"End-to-end FPS    : {avg_fps:.1f}")
    print(f"Pure inference    : {avg_infer_ms:.1f} ms  →  {1000/avg_infer_ms:.1f} FPS")
    print("=======================================")


if __name__ == "__main__":
    main()