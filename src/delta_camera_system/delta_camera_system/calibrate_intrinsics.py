#!/usr/bin/env python3
"""calibrate_intrinsics.py — Chessboard intrinsic calibration for the
global-shutter UVC camera (no hardware intrinsics API, unlike the RealSense
it replaced).

Opens the camera device directly (same V4L2/MJPG settings as
uvc_camera_publisher.py / merge_code_test.py), shows a live preview with
chessboard-corner overlay, and lets you capture poses interactively. Once
enough poses are captured, runs cv2.calibrateCamera() and prints fx/fy/cx/cy
+ distortion + reprojection error, ready to paste into
plant_perception/config/perception_direct.yaml (camera_fx/fy/cx/cy/
camera_dist_k1/k2/p1/p2/k3) and delta_camera_system/uvc_camera_publisher.py's
declared parameter defaults.

Needs the camera device free — stop any running pipeline
(weed_pick_place.launch.py / merge_code_test.py) first, they hold
/dev/video0 exclusively.

Controls (preview window):
    c        capture the current frame (only works when a board is found)
    q / ESC  finish capturing and run calibration (needs >= --min-captures)

Usage:
    python3 calibrate_intrinsics.py --device /dev/video0 \\
        --cols 9 --rows 6 --square-mm 25
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--cols", type=int, default=9, help="inner corners per row")
    parser.add_argument("--rows", type=int, default=6, help="inner corners per column")
    parser.add_argument(
        "--square-mm", type=float, default=25.0,
        help="printed square size in mm (only affects reported units, not fx/fy/cx/cy)",
    )
    parser.add_argument("--min-captures", type=int, default=15)
    parser.add_argument(
        "--out-dir", default=str(Path.home() / "delta_ws" / "camera_calibration"),
        help="where to save captured frames + calibration_result.txt",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    board_size = (args.cols, args.rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(
            f"ERROR: could not open {args.device} — is another node "
            "(merge_code_test.py / uvc_camera_publisher) already holding it? "
            "Stop the pipeline first."
        )
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    print(f"Camera opened: {args.device} @ {actual_w}x{actual_h}")
    print(
        f"Looking for a {args.cols}x{args.rows}-inner-corner chessboard "
        f"({args.square_mm:.1f} mm squares)."
    )
    print("Move the board around: corners, edges, tilted — not just flat-on-center.")
    print(f"Press 'c' to capture when the board is detected (green), 'q' to finish "
          f"(need >= {args.min_captures} captures).")

    # Termination criteria for cv2.cornerSubPix
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= args.square_mm

    objpoints = []   # 3D points in board space, one array per capture
    imgpoints = []   # 2D corner points in image space, one array per capture
    n_captured = 0

    win = "Chessboard calibration — c=capture, q=finish"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_found_corners = None
    frame_size = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame grab failed, retrying...")
                time.sleep(0.05)
                continue

            frame_size = (frame.shape[1], frame.shape[0])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, board_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )

            display = frame.copy()
            last_found_corners = None
            if found:
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), subpix_criteria
                )
                last_found_corners = corners_refined
                cv2.drawChessboardCorners(display, board_size, corners_refined, found)
                status_color = (0, 220, 0)
                status_text = "BOARD FOUND — press c to capture"
            else:
                status_color = (0, 0, 220)
                status_text = "no board"

            cv2.putText(display, status_text, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(display, f"captured: {n_captured} / {args.min_captures} min",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow(win, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("c"), ord("C")) and last_found_corners is not None:
                objpoints.append(objp.copy())
                imgpoints.append(last_found_corners)
                n_captured += 1
                frame_path = out_dir / f"capture_{n_captured:02d}.png"
                cv2.imwrite(str(frame_path), frame)
                print(f"  captured #{n_captured} -> {frame_path}")
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if n_captured < args.min_captures:
        print(
            f"\nOnly {n_captured} captures (< --min-captures {args.min_captures}) "
            "— calibration skipped. Re-run and capture more poses."
        )
        sys.exit(1)

    print(f"\nRunning cv2.calibrateCamera() on {n_captured} captures...")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, frame_size, None, None
    )

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    k1, k2, p1, p2, k3 = (dist_coeffs.reshape(-1).tolist() + [0.0] * 5)[:5]

    # Per-view reprojection error, to spot any bad captures.
    per_view_errors = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        per_view_errors.append(err)

    result_lines = [
        "=" * 60,
        "CAMERA INTRINSIC CALIBRATION RESULT",
        "=" * 60,
        f"captures        : {n_captured}",
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
    for i, err in enumerate(per_view_errors, start=1):
        result_lines.append(f"    capture_{i:02d}: {err:.4f}")
    result_lines.append("=" * 60)

    result_text = "\n".join(result_lines)
    print("\n" + result_text)

    result_path = out_dir / "calibration_result.txt"
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
