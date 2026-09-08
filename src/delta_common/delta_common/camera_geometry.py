#!/usr/bin/env python3
"""
camera_geometry.py — Shared camera<->robot-base geometry and target-feasibility
helpers.  Extracted from delta_camera_system.camera_system.DeltaCamera so both
the orange-cube pipeline and any other detector-side node (e.g. the weed
bridge) build the exact same camera->base transform and IK feasibility check
from a single source of truth.
"""

import math

import numpy as np

from delta_common import config
from delta_common.fk_ik import check_workspace, solve_fk_mm, solve_ik_mm


def legacy_rotation_matrix(mode: str):
    mode = mode.upper()
    if mode == "A":
        return np.array(
            ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
            dtype=np.float64,
        )
    if mode == "B":
        return np.array(
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
            dtype=np.float64,
        )
    if mode == "C":
        return np.array(
            ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
            dtype=np.float64,
        )
    return np.array(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
        dtype=np.float64,
    )


def rpy_rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float):
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)

    rx = np.array(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)), dtype=np.float64)
    ry = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)), dtype=np.float64)
    rz = np.array(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)
    return rz @ ry @ rx


def build_camera_rotation_matrix():
    legacy = legacy_rotation_matrix(config.CAMERA_TRANSFORM_MODE)

    if config.CAMERA_USE_DIRECT_MATRIX:
        try:
            matrix = np.array(config.CAMERA_DIRECT_MATRIX, dtype=np.float64)
            if matrix.shape != (3, 3):
                raise ValueError(f"expected 3x3 matrix, got {matrix.shape}")
        except Exception:
            matrix = legacy
    else:
        matrix = legacy

    fine = rpy_rotation_matrix(
        config.CAM_FINE_ROLL_DEG,
        config.CAM_FINE_PITCH_DEG,
        config.CAM_FINE_YAW_DEG,
    )
    return fine @ matrix


def build_T_cam_to_base():
    """Build the 4x4 homogeneous transform T_cam_to_base.

    p_base = T_cam_to_base @ [p_cam; 1]

    Returns (T_cam_to_base, T_base_to_cam) both as (4,4) float64 arrays.
    The inverse is computed analytically: T_inv = [[R.T, -R.T @ t], [0,0,0,1]]
    which avoids numerical error from np.linalg.inv on a rotation matrix.
    """
    R = build_camera_rotation_matrix()
    t = np.array([config.CAM_TX_MM, config.CAM_TY_MM, config.CAM_TZ_MM], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T, T_inv


def camera_to_base_mm(T_cam_to_base, x_cam: float, y_cam: float, z_cam: float):
    p_cam = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float64)
    p_base = T_cam_to_base @ p_cam
    return float(p_base[0]), float(p_base[1]), float(p_base[2])


def pixel_to_camera_xyz_mm(fx: float, fy: float, cx: float, cy: float, u: int, v: int, z_m: float):
    z_mm = z_m * 1000.0
    x_mm = ((u - cx) * z_mm) / fx
    y_mm = ((v - cy) * z_mm) / fy
    return x_mm, y_mm, z_mm


def project_base_to_pixel(fx: float, fy: float, cx: float, cy: float, T_base_to_cam,
                           x_b: float, y_b: float, z_b: float):
    """Project a base-frame point (mm) to image pixel (u, v) via T_base_to_cam
    and pinhole projection. Returns None if the point is behind the camera
    (z_cam <= 0)."""
    p_cam = T_base_to_cam @ np.array([x_b, y_b, z_b, 1.0], dtype=np.float64)
    x_c, y_c, z_c = p_cam[:3]
    if z_c <= 1.0:
        return None
    u = int(round(fx * x_c / z_c + cx))
    v = int(round(fy * y_c / z_c + cy))
    return u, v


def validate_target(x_base: float, y_base: float, z_base: float):
    """Workspace + IK feasibility gate for a candidate target.

    z_base is EE-tip Z; converted here to platform Z for IK, mirroring what
    motor_controller does downstream.  Returns
    (ik_deg, fk_xyz, fk_err, allowed, reason).
    """
    z_platform = z_base + config.EE_OFFSET_Z_MM
    if not check_workspace(x_base, y_base, z_platform):
        return None, None, None, False, "OUTSIDE_WORKSPACE"

    ok_ik, t1, t2, t3 = solve_ik_mm(x_base, y_base, z_platform)
    if not ok_ik:
        return None, None, None, False, "IK_FAILED"

    if not (
        config.THETA1_MIN <= t1 <= config.THETA1_MAX
        and config.THETA2_MIN <= t2 <= config.THETA2_MAX
        and config.THETA3_MIN <= t3 <= config.THETA3_MAX
    ):
        return (t1, t2, t3), None, None, False, "JOINT_LIMIT"

    ok_fk, x_fk, y_fk, z_fk = solve_fk_mm(t1, t2, t3)
    if not ok_fk:
        return (t1, t2, t3), None, None, False, "FK_FAILED"

    fk_xyz = (x_fk, y_fk, z_fk)
    fk_err = math.sqrt(
        (x_base - x_fk) ** 2
        + (y_base - y_fk) ** 2
        + (z_platform - z_fk) ** 2
    )

    if fk_err > config.FK_VERIFY_TOL_MM:
        return (t1, t2, t3), fk_xyz, fk_err, False, "FK_MISMATCH"

    return (t1, t2, t3), fk_xyz, fk_err, True, "OK"


def estimate_conveyor_vx_mm_s(timed_x_buf) -> float:
    """Linear-regression conveyor velocity (mm/s) from a buffer of
    (t_sec, x_mm) samples for one track, sanity-checked against
    config.CONVEYOR_VX_MIN/MAX_MM_S.

    Fewer than 3 samples or |vx| < MIN -> belt is stopped, return 0.
    |vx| > MAX -> regression outlier, fall back to design speed.
    """
    if len(timed_x_buf) < 3:
        return 0.0

    buf = np.array(timed_x_buf, dtype=np.float64)
    t = buf[:, 0] - buf[0, 0]
    x = buf[:, 1]
    A = np.vstack([t, np.ones(len(t))]).T
    vx, _ = np.linalg.lstsq(A, x, rcond=None)[0]

    if not config.CONVEYOR_MODE:
        return float(vx)

    vx_abs = abs(vx)
    if vx_abs < config.CONVEYOR_VX_MIN_MM_S:
        return 0.0
    if vx_abs > config.CONVEYOR_VX_MAX_MM_S:
        return -config.CONVEYOR_BELT_SPEED_MM_S
    return float(vx)
