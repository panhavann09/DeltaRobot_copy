#!/usr/bin/env python3
"""
build_trt_engine — trtexec wrapper for exporting a device-local TensorRT
.engine from an ONNX model.

Why this exists: a .engine file is compiled for the exact GPU architecture
and TensorRT version that built it — src/weight/best.engine in this
workspace was built for a Jetson Orin (Ampere, SM 8.7) and will not
deserialize on a Jetson Xavier (Volta, SM 7.2). Porting to Xavier means
re-exporting the engine ON the Xavier itself; this script is that export
step. The rest of the pipeline (plant_perception's trt_inferencer.py) is
unchanged — it still loads whatever .engine file you point model_path at
via `import tensorrt as trt`, same as on Orin.

Prerequisite: an ONNX export of the model (src/weight/yolov11-seg-root.onnx
in this workspace already is one; if only a .pt exists, export it first
with ultralytics: `model.export(format="onnx", imgsz=640)`). trtexec builds
engines from ONNX, not directly from .pt.

Usage
-----
    ros2 run trt_engine_builder build_trt_engine \\
        --onnx ~/DeltaRobot_copy/src/weight/yolov11-seg-root.onnx \\
        --output ~/DeltaRobot_copy/src/weight/best_xavier.engine \\
        --precision fp16

Then point plant_perception's model_path (perception.yaml /
perception_direct.yaml) at the new engine file.
"""
from __future__ import annotations  # keeps type hints below safe on Python 3.8 (Foxy/Xavier)

import argparse
import os
import shutil
import subprocess
import sys

# Common install locations across JetPack versions when trtexec isn't on PATH.
_FALLBACK_TRTEXEC_PATHS = [
    "/usr/src/tensorrt/bin/trtexec",
    "/usr/src/tensorrt/samples/trtexec/trtexec",
]


def _find_trtexec(explicit_path: str | None) -> str:
    if explicit_path:
        if not os.path.isfile(explicit_path) or not os.access(explicit_path, os.X_OK):
            sys.exit(f"error: --trtexec path {explicit_path!r} is not an executable file")
        return explicit_path

    env_path = os.environ.get("TRTEXEC")
    if env_path:
        if not os.path.isfile(env_path) or not os.access(env_path, os.X_OK):
            sys.exit(f"error: TRTEXEC env var {env_path!r} is not an executable file")
        return env_path

    on_path = shutil.which("trtexec")
    if on_path:
        return on_path

    for candidate in _FALLBACK_TRTEXEC_PATHS:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    sys.exit(
        "error: could not find trtexec (not on PATH, not at a known JetPack "
        f"install location: {_FALLBACK_TRTEXEC_PATHS}). Pass --trtexec "
        "explicitly or set the TRTEXEC env var."
    )


def _build_command(args, trtexec_path: str) -> list:
    cmd = [
        trtexec_path,
        f"--onnx={args.onnx}",
        f"--saveEngine={args.output}",
        f"--memPoolSize=workspace:{args.workspace_mb}",
    ]

    if args.precision == "fp16":
        cmd.append("--fp16")
    elif args.precision == "int8":
        cmd.append("--int8")
        if args.calib:
            cmd.append(f"--calib={args.calib}")
    # fp32 needs no extra flag — trtexec's default builder precision.

    if not args.dynamic:
        shape = f"1x3x{args.imgsz}x{args.imgsz}"
        cmd.append(f"--shapes=images:{shape}")

    if args.extra_args:
        cmd.extend(args.extra_args.split())

    return cmd


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Export a device-local TensorRT .engine from an ONNX model via trtexec.",
    )
    parser.add_argument("--onnx", required=True, help="Path to the source .onnx model")
    parser.add_argument("--output", "-o", required=True, help="Path to write the .engine file")
    parser.add_argument(
        "--precision", choices=["fp32", "fp16", "int8"], default="fp16",
        help="Builder precision (default: fp16, matching best.engine on Orin)",
    )
    parser.add_argument(
        "--calib", default=None,
        help="INT8 calibration cache path (only used with --precision int8)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Fixed square input size, e.g. 640 -> 1x3x640x640 (default: 640, matches perception.yaml)",
    )
    parser.add_argument(
        "--dynamic", action="store_true",
        help="Skip the fixed --shapes flag for a dynamic-shape ONNX export",
    )
    parser.add_argument(
        "--workspace-mb", type=int, default=4096,
        help="Builder workspace memory pool size in MB (default: 4096 — lower this on "
             "memory-constrained devices like Xavier if the build OOMs)",
    )
    parser.add_argument(
        "--trtexec", default=None,
        help="Explicit path to the trtexec binary (default: search PATH, then known "
             "JetPack install locations, then $TRTEXEC)",
    )
    parser.add_argument(
        "--extra-args", default=None,
        help="Extra raw flags appended verbatim to the trtexec command, space-separated "
             '(e.g. --extra-args="--noTF32 --avgRuns=100")',
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.onnx):
        sys.exit(f"error: --onnx file not found: {args.onnx}")
    if args.precision != "int8" and args.calib:
        sys.exit("error: --calib only makes sense with --precision int8")

    trtexec_path = _find_trtexec(args.trtexec)
    cmd = _build_command(args, trtexec_path)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    if os.path.exists(args.output):
        print(f"warning: {args.output} already exists and will be overwritten", file=sys.stderr)

    print(f"Running: {' '.join(cmd)}\n", flush=True)
    result = subprocess.run(cmd)

    if result.returncode != 0:
        sys.exit(f"trtexec failed (exit code {result.returncode}) — see output above")

    if not os.path.isfile(args.output):
        sys.exit("error: trtexec reported success but no engine file was written")

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nBuilt {args.output} ({size_mb:.1f} MB, precision={args.precision})")


if __name__ == "__main__":
    main()
