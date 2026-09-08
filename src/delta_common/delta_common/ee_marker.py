#!/usr/bin/env python3
"""
ee_marker.py — 650nm laser end-effector marker detection, extracted verbatim
from delta_camera_system.camera_system.DeltaCamera.detect_ee_marker so any
other detector-side node (e.g. the weed bridge) can find the same laser dot.
"""

import math
from collections import deque

import cv2
import numpy as np

from delta_common import config


class EEMarkerDetector:
    """Stateful laser-dot detector — call detect() once per color frame.
    Keeps a short position history internally (EE_LASER_SMOOTH_FRAMES) for
    median smoothing and jump rejection, same as camera_system.py's own
    self._ee_history."""

    def __init__(self):
        self._history = deque(maxlen=config.EE_LASER_SMOOTH_FRAMES)

    def detect(self, frame, ws_corners=None):
        """Detect 650nm laser dot. Returns (u, v) centroid or None.

        Uses ROI-based search: once the dot is found, only searches within
        EE_LASER_ROI_PX pixels of the last known position. This prevents the
        laser reflection off the object surface (20-50px away due to parallax)
        from being mistaken for the EE tip.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fh, fw = hsv.shape[:2]

        sat_min  = int(config.EE_LASER_SAT_MIN)
        val_min  = int(config.EE_LASER_VAL_MIN)
        max_area = int(config.EE_LASER_MAX_AREA)
        roi_r    = int(getattr(config, 'EE_LASER_ROI_PX', 35))

        mask1 = cv2.inRange(hsv,
            (config.EE_LASER_HUE_LOW1, sat_min, val_min),
            (config.EE_LASER_HUE_HIGH1, 255, 255))
        mask2 = cv2.inRange(hsv,
            (config.EE_LASER_HUE_LOW2, sat_min, val_min),
            (config.EE_LASER_HUE_HIGH2, 255, 255))
        mask_red  = cv2.bitwise_or(mask1, mask2)
        mask_core = cv2.inRange(hsv,
            (0, 0, int(config.EE_LASER_CORE_VAL_MIN)),
            (180, int(config.EE_LASER_CORE_SAT_MAX), 255))

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_red  = cv2.morphologyEx(mask_red,  cv2.MORPH_CLOSE, k)
        mask_core = cv2.morphologyEx(mask_core, cv2.MORPH_CLOSE, k)

        last_pos = self._history[-1] if self._history else None
        if last_pos is not None:
            lx, ly = last_pos
            roi_mask = np.zeros((fh, fw), dtype=np.uint8)
            x1r = max(0, lx - roi_r); x2r = min(fw, lx + roi_r)
            y1r = max(0, ly - roi_r); y2r = min(fh, ly + roi_r)
            roi_mask[y1r:y2r, x1r:x2r] = 255
            search_red  = cv2.bitwise_and(mask_red,  roi_mask)
            search_core = cv2.bitwise_and(mask_core, roi_mask)
        else:
            search_red  = mask_red
            search_core = mask_core

        ws_xs = ws_ys = None
        if ws_corners is not None:
            ws_xs = [p[0] for p in ws_corners]
            ws_ys = [p[1] for p in ws_corners]

        v_chan   = hsv[:, :, 2]
        tmp_mask = np.zeros((fh, fw), dtype=np.uint8)

        def _pick(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_c, best_score = None, -1.0
            for c in contours:
                area = float(cv2.contourArea(c))
                if not (config.EE_LASER_MIN_AREA <= area <= max_area):
                    continue

                # Shape classification: laser dot is circular and compact
                perimeter = cv2.arcLength(c, True)
                if perimeter < 1.0:
                    continue
                circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
                if circularity < 0.35:   # elongated or irregular — not a laser dot
                    continue
                (_, _), (bw, bh), _ = cv2.minAreaRect(c)
                if bw > 0 and bh > 0:
                    aspect = max(bw, bh) / min(bw, bh)
                    if aspect > 2.5:     # too elongated — reflection artifact
                        continue

                M_c = cv2.moments(c)
                if M_c["m00"] == 0:
                    continue
                cx = int(M_c["m10"] / M_c["m00"])
                cy = int(M_c["m01"] / M_c["m00"])
                if ws_xs is not None:
                    if not (min(ws_xs) <= cx <= max(ws_xs) and
                            min(ws_ys) <= cy <= max(ws_ys)):
                        continue

                tmp_mask[:] = 0
                cv2.drawContours(tmp_mask, [c], -1, 255, cv2.FILLED)
                mean_v = float(cv2.mean(v_chan, mask=tmp_mask)[0])

                # Score = brightness × circularity — rewards compact bright dots
                score = mean_v * circularity
                if score > best_score:
                    best_score = score
                    best_c = c
            return best_c

        best = _pick(search_red)
        if best is None:
            best = _pick(search_core)
        if best is None and last_pos is not None:
            best = _pick(mask_red)
        if best is None and last_pos is not None:
            best = _pick(mask_core)

        raw = None
        if best is not None:
            M = cv2.moments(best)
            if M["m00"] != 0:
                raw = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        result = None
        if raw is not None:
            if self._history:
                lx, ly = self._history[-1]
                jump = ((raw[0] - lx) ** 2 + (raw[1] - ly) ** 2) ** 0.5
                if jump > config.EE_LASER_MAX_JUMP_PX:
                    raw = None

        if raw is not None:
            self._history.append(raw)

        if self._history:
            xs = sorted(p[0] for p in self._history)
            ys = sorted(p[1] for p in self._history)
            n  = len(xs)
            result = (xs[n // 2], ys[n // 2])

        return result
