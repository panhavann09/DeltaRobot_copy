#!/usr/bin/env python3
"""calibrate_intrinsics_offline.py — Run cv2.calibrateCamera() on chessboard
images already captured to disk (e.g. by calibrate_intrinsics.py's 'c' key),
without needing to reopen the camera device.

Usage:
    python3 calibrate_intrinsics_offline.py \\
        --images-dir ~/delta_ws/camera_calibration \\
        --cols 9 --rows 6 --square-mm 25
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir", default=str(Path.home() / "delta_ws" / "camera_calibration"),
        help="directory containing capture_*.png (or --glob) images",
    )
    parser.add_argument("--glob", default="capture_*.png")
    parser.add_argument("--cols", type=int, default=9, help="inner corners per row")
    parser.add_argument("--rows", type=int, default=6, help="inner corners per column")
    parser.add_argument(
        "--square-mm", type=float, default=25.0,
        help="printed square size in mm (only affects reported units, not fx/fy/cx/cy)",
    )
    parser.add_argument("--min-captures", type=int, default=10)
    return parser.parse_args()


def main():
    args = _parse_args()
    board_size = (args.cols, args.rows)
    images_dir = Path(args.images_dir).expanduser()

    image_paths = sorted(images_dir.glob(args.glob))
    if not image_paths:
        print(f"ERROR: no images matching {args.glob!r} in {images_dir}")
        sys.exit(1)

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= args.square_mm

    objpoints = []
    imgpoints = []
    used_names = []
    frame_size = None

    print(f"Looking for a {args.cols}x{args.rows}-inner-corner chessboard "
          f"({args.square_mm:.1f} mm squares) in {len(image_paths)} image(s)...")

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  {path.name}: could not read, skipping")
            continue

        if frame_size is None:
            frame_size = (img.shape[1], img.shape[0])
        elif (img.shape[1], img.shape[0]) != frame_size:
            print(f"  {path.name}: size {img.shape[1]}x{img.shape[0]} != "
                  f"{frame_size[0]}x{frame_size[1]}, skipping")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, board_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            print(f"  {path.name}: board NOT found, skipping")
            continue

        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), subpix_criteria
        )
        objpoints.append(objp.copy())
        imgpoints.append(corners_refined)
        used_names.append(path.name)
        print(f"  {path.name}: board found")

    n_captured = len(objpoints)
    if n_captured < args.min_captures:
        print(
            f"\nOnly {n_captured} usable images (< --min-captures {args.min_captures}) "
            "— calibration skipped. Capture more poses or check board size/lighting."
        )
        sys.exit(1)

    print(f"\nRunning cv2.calibrateCamera() on {n_captured} usable image(s)...")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, frame_size, None, None
    )

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    k1, k2, p1, p2, k3 = (dist_coeffs.reshape(-1).tolist() + [0.0] * 5)[:5]

    per_view_errors = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        diff = imgpoints[i].reshape(-1, 2).astype(np.float64) - proj.reshape(-1, 2).astype(np.float64)
        err = float(np.sqrt((diff ** 2).sum(axis=1)).mean())
        per_view_errors.append(err)

    result_lines = [
        "=" * 60,
        "CAMERA INTRINSIC CALIBRATION RESULT (offline)",
        "=" * 60,
        f"images used     : {n_captured} / {len(image_paths)} found in {images_dir}",
        f"image size      : {frame_size[0]}x{frame_size[1]}",
        f"RMS reproj error: {rms:.4f} px  (want < ~0.5 px; > 1.0 px is suspect)",
        "",
        "fx = %.4f" % fx,
        "fy = %.4f" % fy,
        "cx = %.4f" % cx,
        "cy = %.4f" % cy,
        "k1 = %.6f" % k1,
        "k2 = %.6f" % k2,
        "p1 = %.6f" % p1,
        "p2 = %.6f" % p2,
        "k3 = %.6f" % k3,
        "",
        "Paste into src/plant_perception/config/perception_direct.yaml:",
        f"    camera_fx: {fx:.4f}",
        f"    camera_fy: {fy:.4f}",
        f"    camera_cx: {cx:.4f}",
        f"    camera_cy: {cy:.4f}",
        f"    camera_dist_k1: {k1:.6f}",
        f"    camera_dist_k2: {k2:.6f}",
        f"    camera_dist_p1: {p1:.6f}",
        f"    camera_dist_p2: {p2:.6f}",
        f"    camera_dist_k3: {k3:.6f}",
        "",
        "Per-view reprojection error (px), flag anything way above the rest:",
    ]
    for name, err in zip(used_names, per_view_errors):
        result_lines.append(f"    {name}: {err:.4f}")
    result_lines.append("=" * 60)

    result_text = "\n".join(result_lines)
    print("\n" + result_text)

    result_path = images_dir / "calibration_result.txt"
    result_path.write_text(result_text + "\n")
    print(f"\nSaved -> {result_path}")
    print(
        "\nAfter updating perception_direct.yaml, re-run the camera-offset "
        "calibration too (ros2 service call /delta/calibrate_cam_offset "
        "std_srvs/srv/Trigger) — CAM_TX_MM/CAM_TY_MM were fit against the "
        "old intrinsics and are now stale."
    )


if __name__ == "__main__":
    main()
