#!/usr/bin/env python3
"""
postprocess.py — Decode bounding boxes, classes, masks, and root points
from raw ONNX outputs using Ultralytics NMS.
"""

import cv2
import numpy as np

try:
    import torch
    from ultralytics.utils.nms import non_max_suppression
    HAS_TORCH_NMS = True
except Exception:
    torch = None
    non_max_suppression = None
    HAS_TORCH_NMS = False


class Detection:
    """Single plant / weed detection."""
    __slots__ = (
        "class_id", "class_name", "confidence",
        "box_xyxy", "mask", "root_point", "track_id", "root_d", "_cand_root",
    )

    def __init__(self, class_id, class_name, confidence,
                 box_xyxy, mask, root_point, track_id=None, root_d=0.0):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.box_xyxy = box_xyxy
        self.mask = mask
        self.root_point = root_point
        self.track_id = track_id
        self.root_d = root_d


def _scale_boxes(input_shape, boxes, orig_shape):
    """Rescale boxes from letterbox space to original image space."""
    ih, iw = input_shape
    h0, w0 = orig_shape
    gain = min(iw / w0, ih / h0)
    pad_w = (iw - w0 * gain) / 2
    pad_h = (ih - h0 * gain) / 2

    scaled = boxes.copy()
    scaled[:, [0, 2]] = (scaled[:, [0, 2]] - pad_w).clip(0, w0) / gain
    scaled[:, [1, 3]] = (scaled[:, [1, 3]] - pad_h).clip(0, h0) / gain
    scaled[:, [0, 2]] = scaled[:, [0, 2]].clip(0, w0)
    scaled[:, [1, 3]] = scaled[:, [1, 3]].clip(0, h0)
    return scaled


def _scale_coords(input_shape, coords, orig_shape):
    """Rescale (x, y) coordinates from letterbox space to original image space."""
    ih, iw = input_shape
    h0, w0 = orig_shape
    gain = min(iw / w0, ih / h0)
    pad_w = (iw - w0 * gain) / 2
    pad_h = (ih - h0 * gain) / 2

    scaled = coords.copy()
    scaled[:, 0] = ((scaled[:, 0] - pad_w) / gain).clip(0, w0)
    scaled[:, 1] = ((scaled[:, 1] - pad_h) / gain).clip(0, h0)
    return scaled


def decode_masks(mc_coeffs, proto, det_boxes_lb, orig_shape, input_shape):
    """Decode segmentation masks from prototype coefficients → binary (0/255)."""
    n = mc_coeffs.shape[0]
    if n == 0:
        return np.zeros((0, *orig_shape), dtype=np.uint8)

    ih, iw = input_shape
    p = proto[0] if proto.ndim == 4 else proto
    ph, pw = p.shape[1], p.shape[2]

    masks = (mc_coeffs @ p.reshape(32, -1)).reshape(n, ph, pw)
    masks = 1.0 / (1.0 + np.exp(-np.clip(masks, -50.0, 50.0)))

    # Upsample to letterbox size
    masks_up = np.zeros((n, ih, iw), dtype=np.float32)
    for i in range(n):
        masks_up[i] = cv2.resize(masks[i], (iw, ih), interpolation=cv2.INTER_LINEAR)

    # Crop to bounding box regions
    for i in range(n):
        x1, y1, x2, y2 = det_boxes_lb[i].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        roi = np.zeros((ih, iw), dtype=np.float32)
        roi[y1:y2, x1:x2] = masks_up[i, y1:y2, x1:x2]
        masks_up[i] = roi

    # Remove letterbox padding and resize to original
    h0, w0 = orig_shape
    gain = min(iw / w0, ih / h0)
    pad_w = (iw - w0 * gain) / 2
    pad_h = (ih - h0 * gain) / 2

    top = max(0, int(round(pad_h - 0.1)))
    left = max(0, int(round(pad_w - 0.1)))
    bottom = min(ih, int(round(ih - pad_h + 0.1)))
    right = min(iw, int(round(iw - pad_w + 0.1)))

    masks_crop = masks_up[:, top:bottom, left:right]

    masks_orig = np.zeros((n, h0, w0), dtype=np.uint8)
    for i in range(n):
        resized = cv2.resize(masks_crop[i], (w0, h0), interpolation=cv2.INTER_LINEAR)
        masks_orig[i] = (resized > 0.5).astype(np.uint8) * 255

    return masks_orig


def postprocess(onnx_outputs, orig_shape, input_shape,
                conf_thres=0.25, iou_thres=0.45, num_classes=4, class_names=None,
                decode_segmentation=False):
    """
    NMS → scale coordinates → optional mask decoding → build Detection list.

    ONNX output0: (1, 42, 8400)  [4 box + 4 cls + 32 mask_coeffs + 2 root]
    ONNX output1: (1, 32, 160, 160)  mask prototypes
    """
    output0, proto = onnx_outputs[0], onnx_outputs[1]

    if not HAS_TORCH_NMS or torch is None:
        # Mock fallback for test environments without PyTorch installed
        dets = np.zeros((0, 40), dtype=np.float32)
    else:
        with torch.inference_mode():
            pred_tensor = torch.from_numpy(output0).float()
            if pred_tensor.ndim == 3 and pred_tensor.shape[1] > pred_tensor.shape[2]:
                pred_tensor = pred_tensor.transpose(1, 2)

            nms_results = non_max_suppression(
                pred_tensor, conf_thres=conf_thres, iou_thres=iou_thres, nc=num_classes,
            )
            dets = nms_results[0].cpu().numpy()

    if len(dets) == 0:
        return []

    # Columns: [x1, y1, x2, y2, conf, cls, 32 mask_coeffs, 2 root_xy]
    boxes_xyxy = dets[:, :4]
    scores = dets[:, 4]
    class_ids = dets[:, 5].astype(int)
    mask_coeffs = dets[:, 6:38]
    kpts_lb = dets[:, 38:40]

    h_orig, w_orig = orig_shape
    boxes_orig = _scale_boxes(input_shape, boxes_xyxy, (h_orig, w_orig))
    kpts_orig = _scale_coords(input_shape, kpts_lb, (h_orig, w_orig))

    if decode_segmentation:
        masks = decode_masks(mask_coeffs, proto, boxes_xyxy, (h_orig, w_orig), input_shape)
    else:
        masks = None

    detections = []
    for j in range(len(scores)):
        cls_id = int(class_ids[j])
        name = class_names[cls_id] if class_names and cls_id in class_names else f"class_{cls_id}"
        det_mask = masks[j] if masks is not None else None
        detections.append(Detection(
            class_id=cls_id,
            class_name=name,
            confidence=float(scores[j]),
            box_xyxy=tuple(boxes_orig[j].astype(int)),
            mask=det_mask,
            root_point=(int(round(kpts_orig[j, 0])), int(round(kpts_orig[j, 1]))),
        ))

    return detections
