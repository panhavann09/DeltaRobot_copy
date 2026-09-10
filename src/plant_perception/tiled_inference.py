#!/usr/bin/env python3
"""
tiled_inference.py — Tiled high-resolution inference wrapper around the
existing TRTInferencer + postprocess() pipeline.

Ported from ~/yolo_bench_ws/yolo_test_fp16_high.py's tiling approach
(generate_tiles / global NMS merge), adapted to call this project's own
TRTInferencer.run() and postprocess() UNCHANGED per tile — both already
operate on an arbitrary-size single image via their own letterboxing /
orig_shape parameter, so no changes were needed to either.

Why: at native 640x480 (or 1920x1080 downscaled to 640x640 in one pass),
small/distant weeds get heavily shrunk before the model ever sees them.
Splitting a high-res frame into overlapping tiles and running each tile at
a mild ~1.5x downscale (960px tile -> 640px model input, instead of a
~3x downscale for a 1920px-wide frame) preserves far more detail.

Known limitation: mask compositing across tile boundaries is NOT
implemented. Both perception.yaml and perception_direct.yaml currently ship
with publish_masks: false, so this doesn't regress anything today — but if
decode_segmentation=True is requested, run_tiled_inference logs one warning
and skips mask decoding entirely rather than returning wrong/tile-sized
mask arrays.
"""
import logging
from typing import List, Tuple

import cv2
import numpy as np

try:
    from postprocess import Detection, postprocess
except ImportError:
    from .postprocess import Detection, postprocess

_warned_masks_unsupported = False


def _axis_starts(dim: int, tile_size: int, overlap: int) -> List[int]:
    """Evenly-spaced tile start offsets covering [0, dim) with >= `overlap`
    px of overlap between neighbors — never less than requested, but not
    wildly more either.

    A fixed-step walk (previous approach) clamps its last tile to the frame
    edge whenever `dim` isn't an exact multiple of the step, which silently
    turns the requested ~`overlap`px overlap into a near-duplicate of the
    previous tile (e.g. 864px of overlap instead of 96px) — extra full
    inference passes for zero coverage benefit. Spacing tiles evenly instead
    keeps the overlap close to what was asked for on every tile, including
    the last one.
    """
    if dim <= tile_size:
        return [0]

    step = tile_size - overlap
    n_tiles = -(-(dim - tile_size) // step) + 1  # ceil division
    span = dim - tile_size
    return [round(i * span / (n_tiles - 1)) for i in range(n_tiles)]


def generate_tiles(h: int, w: int, tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """Generate overlapping (x1, y1, x2, y2) tiles covering an h x w frame.
    For 1920x1080 with tile_size=1088/overlap=96 (perception.yaml /
    perception_direct.yaml default) this produces a 1x2 (2-tile) layout —
    height fits in one tile, width splits in two. A dimension smaller than
    tile_size never gets split (see _axis_starts)."""
    xs = _axis_starts(w, tile_size, overlap)
    ys = _axis_starts(h, tile_size, overlap)

    tiles = [(x, y, min(x + tile_size, w), min(y + tile_size, h)) for y in ys for x in xs]
    return list(dict.fromkeys(tiles))   # de-dup (small frames can repeat the same tile)


def _offset_detection(det: Detection, offset_x: int, offset_y: int) -> Detection:
    """Detection uses __slots__ with immutable tuple fields — rebuild rather
    than mutate."""
    x1, y1, x2, y2 = det.box_xyxy
    rx, ry = det.root_point
    return Detection(
        class_id=det.class_id,
        class_name=det.class_name,
        confidence=det.confidence,
        box_xyxy=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
        mask=det.mask,   # None in the tiled path — see module docstring
        root_point=(rx + offset_x, ry + offset_y),
    )


def _global_merge_nms(detections: List[Detection], conf_thres: float, merge_iou_thres: float) -> List[Detection]:
    """Drop duplicate detections in tile-overlap regions — same primitive
    (cv2.dnn.NMSBoxes) the bench script's global_nms uses."""
    if not detections:
        return []

    boxes = []
    scores = []
    for d in detections:
        x1, y1, x2, y2 = d.box_xyxy
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(d.confidence)

    idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, merge_iou_thres)
    idxs = np.array(idxs).flatten() if len(idxs) > 0 else np.array([], dtype=int)
    return [detections[i] for i in idxs]


def run_tiled_inference(
    inferencer,
    frame_bgr: np.ndarray,
    *,
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    num_classes: int,
    class_names: dict,
    decode_segmentation: bool,
    tile_size: int,
    tile_overlap: int,
    merge_iou_thres: float = 0.5,
) -> Tuple[List[Detection], float, int]:
    """Run inference over overlapping tiles of frame_bgr and merge results.

    inferencer must be a TRTInferencer (trt_inferencer.py) — its .run() is
    called once per tile, unchanged. Returns (merged_detections,
    total_inference_ms, n_tiles).
    """
    global _warned_masks_unsupported
    if decode_segmentation and not _warned_masks_unsupported:
        logging.getLogger("tiled_inference").warning(
            "decode_segmentation=True requested but mask compositing across "
            "tile boundaries isn't implemented — masks will be skipped in "
            "tiled mode (boxes/roots are unaffected)."
        )
        _warned_masks_unsupported = True

    h, w = frame_bgr.shape[:2]
    tiles = generate_tiles(h, w, tile_size, tile_overlap)

    all_dets: List[Detection] = []
    total_inference_ms = 0.0

    for x1, y1, x2, y2 in tiles:
        tile_img = frame_bgr[y1:y2, x1:x2]
        onnx_outputs, _ratio, _pad, inference_ms = inferencer.run(tile_img)
        total_inference_ms += inference_ms

        tile_dets = postprocess(
            onnx_outputs=onnx_outputs,
            orig_shape=tile_img.shape[:2],
            input_shape=(imgsz, imgsz),
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            num_classes=num_classes,
            class_names=class_names,
            decode_segmentation=False,   # per-tile masks aren't composited — see module docstring
        )
        all_dets.extend(_offset_detection(d, x1, y1) for d in tile_dets)

    merged = _global_merge_nms(all_dets, conf_thres, merge_iou_thres)
    return merged, total_inference_ms, len(tiles)
