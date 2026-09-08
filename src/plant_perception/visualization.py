#!/usr/bin/env python3
"""
visualization.py — Draw segmentation masks, bounding boxes, root markers,
track states, and HUD overlay on BGR camera frames.
"""

import cv2
import numpy as np

CLASS_COLORS_BGR = {
    0: (0, 255, 0),    # crop_small_leaf  — bright green
    1: (0, 180, 0),    # crop_large_leaf  — dark green
    2: (0, 0, 255),    # weed_small_leaf  — bright red
    3: (0, 0, 180),    # weed_large_leaf  — dark red
}
COASTING_COLOR = (128, 128, 128)  # Gray for coasting state
ROOT_COLOR = (255, 0, 255)       # Magenta for active root point
COASTING_ROOT_COLOR = (180, 180, 180)


def draw_visualizations(img_bgr, detections, show_boxes=True, show_masks=False,
                        show_roots=True, fps=None, latency_ms=None, device=None):
    """Return an annotated copy of img_bgr for detections or PlantTrackStates."""
    out = img_bgr.copy()
    h, w = out.shape[:2]

    # Resolution-adaptive scaling
    scale = max(h, w) / 1200.0
    lw = max(1, int(round(2 * scale)))
    r_inner = max(3, int(round(5 * scale)))
    r_outer = max(5, int(round(8 * scale)))
    font_scl = max(0.4, 0.55 * scale)
    fthick = max(1, int(round(scale)))

    # Masks (only drawn if show_masks is enabled AND det.mask is not None)
    if show_masks:
        for det in detections:
            mask = getattr(det, "mask", None)
            if mask is None:
                continue
            
            is_coasting = getattr(det, "track_state", None) == 2 or not getattr(det, "is_currently_detected", True)
            if is_coasting:
                continue  # Never display stale masks on coasting objects

            color = CLASS_COLORS_BGR.get(det.class_id, (255, 255, 255))
            mask_bool = mask.astype(bool)
            if mask_bool.any():
                overlay = np.zeros_like(out)
                overlay[:] = color
                out[mask_bool] = cv2.addWeighted(out, 0.55, overlay, 0.45, 0)[mask_bool]
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, contours, -1, color, max(1, lw // 2), cv2.LINE_AA)

    # Bounding boxes & labels
    if show_boxes:
        for det in detections:
            bbox = getattr(det, "bbox_xyxy", getattr(det, "box_xyxy", None))
            if bbox is None:
                continue

            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            is_coasting = getattr(det, "track_state", None) == 2 or not getattr(det, "is_currently_detected", True)

            color = COASTING_COLOR if is_coasting else CLASS_COLORS_BGR.get(det.class_id, (255, 255, 255))

            # Header info
            t_id = getattr(det, "track_id", None)
            track_prefix = f"#{t_id} " if t_id is not None and t_id >= 0 else ""
            
            # State label
            state_val = getattr(det, "track_state", None)
            state_str = ""
            if state_val == 0:
                state_str = "[TENTATIVE]"
            elif state_val == 1:
                state_str = "[CONF]"
            elif state_val == 2:
                state_str = "[COASTING]"

            # Depth label
            depth_val = getattr(det, "depth_m", getattr(det, "root_d", -1.0))
            depth_valid = getattr(det, "depth_valid", depth_val > 0)
            depth_str = f" d:{depth_val:.2f}m" if depth_valid and depth_val > 0 else " d:N/A"

            cls_name = getattr(det, "class_name", f"class_{det.class_id}")
            conf = getattr(det, "confidence", 0.0)

            label = f"{track_prefix}{cls_name} {conf:.2f}{depth_str} {state_str}".strip()

            if is_coasting:
                # Draw dashed bounding box for coasting state
                _draw_dashed_rectangle(out, (x1, y1), (x2, y2), color, lw)
            else:
                cv2.rectangle(out, (x1, y1), (x2, y2), color, lw, cv2.LINE_AA)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scl, fthick)
            bg_y1 = max(0, y1 - th - 8)
            bg_y2 = max(th + 8, y1)
            text_y = max(th + 4, y1 - 4)
            cv2.rectangle(out, (x1, bg_y1), (x1 + tw + 4, bg_y2), color, -1, cv2.LINE_AA)
            cv2.putText(out, label, (x1 + 2, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scl, (255, 255, 255), fthick, cv2.LINE_AA)

    # Root markers
    if show_roots:
        for det in detections:
            root = getattr(det, "root_point", None)
            if root is None:
                continue
            rx, ry = int(round(root[0])), int(round(root[1]))
            is_coasting = getattr(det, "track_state", None) == 2 or not getattr(det, "is_currently_detected", True)
            root_color = COASTING_ROOT_COLOR if is_coasting else ROOT_COLOR

            if 0 <= rx < w and 0 <= ry < h:
                cv2.circle(out, (rx, ry), r_outer, (255, 255, 255), max(1, lw // 2), cv2.LINE_AA)
                cv2.circle(out, (rx, ry), r_inner, root_color, -1, cv2.LINE_AA)

    # HUD overlay
    if fps is not None or latency_ms is not None or device is not None:
        lines = []
        if fps is not None:
            lines.append(f"FPS: {fps:.1f}")
        if latency_ms is not None:
            lines.append(f"Latency: {latency_ms:.1f} ms")
        if device is not None:
            lines.append(f"Device: {device.upper()}")
        lines.append(f"Active Objects: {len(detections)}")

        hud_scl = max(0.45, 0.55 * scale)
        hud_thick = max(1, int(round(scale)))

        box_w, box_h = 0, 0
        line_heights = []
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, hud_scl, hud_thick)
            box_w = max(box_w, tw)
            box_h += th + 10
            line_heights.append(th)
        box_w += 30
        box_h += 15

        hx, hy = 15, 15
        box_w = min(box_w, w - hx)
        box_h = min(box_h, h - hy)
        sub = out[hy:hy + box_h, hx:hx + box_w]
        if sub.size > 0:
            out[hy:hy + box_h, hx:hx + box_w] = cv2.addWeighted(
                sub, 0.45, np.zeros_like(sub), 0.55, 0
            )

        cv2.circle(out, (hx + 12, hy + 16), int(4 * scale), (0, 255, 0), -1, cv2.LINE_AA)

        cy = hy + 20
        for i, line in enumerate(lines):
            x_off = 24 if i == 0 else 10
            cv2.putText(out, line, (hx + x_off, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, hud_scl, (255, 255, 255), hud_thick, cv2.LINE_AA)
            cy += line_heights[i] + 10

    return out


def _draw_dashed_rectangle(img, pt1, pt2, color, thickness=1, dash_len=8):
    """Draw dashed rectangle for coasting track state visualization."""
    x1, y1 = pt1
    x2, y2 = pt2

    for start_x in range(x1, x2, dash_len * 2):
        end_x = min(start_x + dash_len, x2)
        cv2.line(img, (start_x, y1), (end_x, y1), color, thickness)
        cv2.line(img, (start_x, y2), (end_x, y2), color, thickness)

    for start_y in range(y1, y2, dash_len * 2):
        end_y = min(start_y + dash_len, y2)
        cv2.line(img, (x1, start_y), (x1, end_y), color, thickness)
        cv2.line(img, (x2, start_y), (x2, end_y), color, thickness)
