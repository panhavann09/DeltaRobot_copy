#!/usr/bin/env python3
"""
workspace_overlay.py — Conveyor/workspace zone overlay drawing, extracted
verbatim from delta_camera_system.camera_system.DeltaCamera so any other
detector-side node (e.g. the weed bridge) can draw the identical overlay.

Only the currently-enabled-by-default subset is ported here
(DRAW_WORKSPACE_ZONES / draw_conveyor_zones) — the WORKSPACE_ROI_ENABLE and
ROBOT_EXCLUDE_ENABLE blocks in camera_system.py's draw_workspace_overlay are
both False in delta_common.config today and were not ported; port them the
same way (project_fn-parameterized) if those flags are ever turned on.
"""

import math

import numpy as np
import cv2

from delta_common import config


def draw_conveyor_zones(annotated, frame_w: int, frame_h: int, project_fn):
    """Overlay the exit zone projected from robot base frame onto the camera image.

    Zone layout (robot front = +X of robot base; conveyor moves objects
    in the -X direction of robot base):
      APPROACH (amber) — object visible but not yet in robot reach (x > +X_LIMIT) — not drawn
      WORKSPACE (green) — robot can pick here (|x|,|y| within ±X_LIMIT/Y_LIMIT) — not drawn
      EXIT (red) — object has passed workspace (x < -X_LIMIT)

    Only the EXIT zone is drawn; APPROACH and WORKSPACE fills/borders were
    removed from the overlay, but the workspace polygon is still computed
    and returned since callers use it as an EE-marker search-bounds filter.

    project_fn(x_b, y_b, z_b) -> (u, v) or None — e.g.
    delta_common.camera_geometry.project_base_to_pixel bound to fx/fy/cx/cy/T_base_to_cam.

    Returns the workspace polygon (Nx2 int32 array) for callers that want it
    cached (camera_system.py uses this as an EE-marker search-bounds filter),
    or None if nothing was drawn.
    """
    L = config.X_LIMIT          # half-width of workspace square
    z = config.WORKSPACE_PICK_Z_MM  # belt surface in base frame
    ox = getattr(config, "WORKSPACE_OVERLAY_X_OFFSET_MM", 0.0)
    oy = getattr(config, "WORKSPACE_OVERLAY_Y_OFFSET_MM", 0.0)

    # Project 4 corners of the reachable square at pick Z.
    # Split into entry edge (x=+L, where belt objects arrive — robot front)
    # and exit edge (x=-L, where objects leave robot reach).
    entry_px, exit_px = [], []
    for sy in (-1.0, 1.0):
        for sx, bucket in ((+1.0, entry_px), (-1.0, exit_px)):
            pt = project_fn(sx * L + ox, sy * L + oy, z)
            if pt is None:
                return None   # camera not ready or point behind camera
            bucket.append(pt)

    # Sort each edge by pixel u so polygon vertices wind consistently
    entry_px.sort(key=lambda p: p[0])
    exit_px.sort(key=lambda p: p[0])
    el, er = entry_px   # left & right pixel of entry edge (x_base = +L)
    xl, xr = exit_px    # left & right pixel of exit edge  (x_base = -L)

    # Skip drawing if the projection is wildly outside the frame
    all_v = [el[1], er[1], xl[1], xr[1]]
    if min(all_v) > 2 * frame_h or max(all_v) < -frame_h:
        return None

    overlay = annotated.copy()

    # ── fill zones ────────────────────────────────────────────────────────
    # ws_poly is still computed (and returned) for the EE-marker search-bounds
    # filter — only the approach (amber) and workspace (green) fill/border are
    # no longer drawn.
    ws_poly = np.array([list(el), list(er), list(xr), list(xl)], dtype=np.int32)
    centroid = (
        (el[0] + er[0] + xl[0] + xr[0]) / 4.0,
        (el[1] + er[1] + xl[1] + xr[1]) / 4.0,
    )

    # Exit edge is extended outward (away from the workspace center) until
    # well past the frame — cv2.fillPoly clips to the canvas automatically.
    # This makes the fill direction follow whatever way exit actually
    # projects (top/bottom, left/right, or diagonal), instead of assuming
    # a fixed vertical camera mount.
    ex_poly = _extend_zone_poly(xl, xr, centroid, frame_w, frame_h)
    cv2.fillPoly(overlay, [ex_poly], (60, 60, 200))          # red

    cv2.addWeighted(overlay, 0.20, annotated, 0.80, 0.0, annotated)

    # ── border lines ──────────────────────────────────────────────────────
    cv2.line(annotated, tuple(xl), tuple(xr), (60, 60, 200), 2)  # exit

    # ── workspace center crosshair (X=0, Y=0) ────────────────────────────
    ctr = project_fn(ox, oy, z)
    if ctr is not None:
        cx, cy = int(ctr[0]), int(ctr[1])
        cv2.line(annotated, (cx - 4, cy), (cx + 4, cy), (255, 255, 255), 1)
        cv2.line(annotated, (cx, cy - 4), (cx, cy + 4), (255, 255, 255), 1)

    return ws_poly


def _extend_zone_poly(edge_a, edge_b, centroid, frame_w: int, frame_h: int):
    """Quad from edge (edge_a→edge_b) extended outward past the frame.

    Direction is away from the workspace centroid, so the zone fill
    follows wherever the edge actually projects (top/bottom, left/right,
    or diagonal) instead of assuming a fixed camera mount orientation.
    cv2.fillPoly clips to the canvas, so overshooting is safe.
    """
    mid_x = (edge_a[0] + edge_b[0]) / 2.0
    mid_y = (edge_a[1] + edge_b[1]) / 2.0
    dx, dy = mid_x - centroid[0], mid_y - centroid[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        dx, dy, norm = 0.0, 1.0, 1.0
    dx, dy = dx / norm, dy / norm

    far = math.hypot(frame_w, frame_h) * 2.0
    far_a = (edge_a[0] + dx * far, edge_a[1] + dy * far)
    far_b = (edge_b[0] + dx * far, edge_b[1] + dy * far)
    return np.array([list(edge_a), list(edge_b), far_b, far_a], dtype=np.int32)


def draw_workspace_overlay(annotated, frame_w: int, frame_h: int, project_fn):
    """Draws the conveyor zones (if enabled). Returns the workspace polygon
    (for EE-marker bounds filtering) or None."""
    if config.DRAW_WORKSPACE_ZONES:
        return draw_conveyor_zones(annotated, frame_w, frame_h, project_fn)
    return None
