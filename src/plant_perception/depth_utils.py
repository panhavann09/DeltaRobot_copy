#!/usr/bin/env python3
"""
depth_utils.py — Robust depth estimation at root-point coordinates.
Supports 16UC1 (mm integers) and 32FC1 (metres floats) encodings with MAD outlier removal.
"""

from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DepthEstimate:
    depth_m: float
    valid: bool
    valid_ratio: float
    std_m: float
    sample_count: int


def estimate_root_depth(depth_img, root_x, root_y,
                        window_size=7,
                        minimum_depth_valid_ratio=0.40,
                        minimum_depth_m=0.10,
                        maximum_depth_m=3.00,
                        depth_mad_multiplier=2.5,
                        encoding="16UC1") -> DepthEstimate:
    """
    Estimate robust depth at (root_x, root_y) using ROI median + MAD outlier removal.
    Returns a DepthEstimate instance containing depth in metres, validity, valid_ratio, std_m, sample_count.
    """
    if depth_img is None:
        return DepthEstimate(depth_m=-1.0, valid=False, valid_ratio=0.0, std_m=0.0, sample_count=0)

    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1

    half_w = window_size // 2
    h, w = depth_img.shape[:2]
    cx, cy = int(round(root_x)), int(round(root_y))

    # Boundary check & ROI extraction
    x_min = max(0, cx - half_w)
    x_max = min(w, cx + half_w + 1)
    y_min = max(0, cy - half_w)
    y_max = min(h, cy + half_w + 1)

    total_pixels = (x_max - x_min) * (y_max - y_min)
    if total_pixels <= 0:
        return DepthEstimate(depth_m=-1.0, valid=False, valid_ratio=0.0, std_m=0.0, sample_count=0)

    roi = depth_img[y_min:y_max, x_min:x_max]

    # Filter invalid values (zero, negative, NaN, Inf)
    valid_mask = (roi > 0) & np.isfinite(roi)
    valid_raw = roi[valid_mask].astype(np.float32)

    if len(valid_raw) == 0:
        return DepthEstimate(depth_m=-1.0, valid=False, valid_ratio=0.0, std_m=0.0, sample_count=0)

    # Convert to metres
    if encoding in ("16UC1", "mono16") or depth_img.dtype == np.uint16:
        valid_m = valid_raw / 1000.0
    else:
        valid_m = valid_raw

    # Median Absolute Deviation (MAD) filtering
    med_raw = float(np.median(valid_m))
    mad = float(np.median(np.abs(valid_m - med_raw)))

    epsilon = 1e-6
    if mad > epsilon:
        threshold = depth_mad_multiplier * mad
        filtered_m = valid_m[np.abs(valid_m - med_raw) <= threshold]
    else:
        tolerance_m = max(0.01, 0.02 * med_raw)
        filtered_m = valid_m[np.abs(valid_m - med_raw) <= tolerance_m]

    sample_count = len(filtered_m)
    if sample_count == 0:
        return DepthEstimate(depth_m=-1.0, valid=False, valid_ratio=0.0, std_m=0.0, sample_count=0)

    final_depth_m = float(np.median(filtered_m))
    std_m = float(np.std(filtered_m)) if sample_count > 1 else 0.0
    valid_ratio = float(sample_count) / float(total_pixels)

    valid = (
        sample_count > 0
        and valid_ratio >= minimum_depth_valid_ratio
        and minimum_depth_m <= final_depth_m <= maximum_depth_m
    )

    out_depth_m = final_depth_m if valid else -1.0
    return DepthEstimate(
        depth_m=out_depth_m,
        valid=valid,
        valid_ratio=valid_ratio,
        std_m=std_m,
        sample_count=sample_count
    )


def sample_depth_median(depth_img, root_x, root_y, window_size=7, encoding="16UC1"):
    """Backward-compatible helper returning single float depth in metres."""
    est = estimate_root_depth(depth_img, root_x, root_y, window_size=window_size, encoding=encoding)
    return est.depth_m if est.valid else -1.0
