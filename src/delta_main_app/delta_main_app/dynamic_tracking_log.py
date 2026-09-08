#!/usr/bin/env python3
"""
dynamic_tracking_log.py — FK-based tracking-error logger, no camera/laser.

Compares the commanded target against the robot's *actual* position, where
"actual" comes from forward kinematics on live motor-encoder feedback
(matlab_bridge_node.py's /delta/matlab/fk_result — published right after
every settled move, see solve_fk_mm() over the CAN feedback thetas). Purely
mechanical/encoder ground truth, independent of vision.

Targets matlab_bridge_node.py specifically — that's the node both
delta_main.launch.py and matlab_bridge.launch.py actually start (executable
"matlab_bridge"; main_app.py has a console-script entry but is NOT wired
into any launch file, so it isn't the live pick-place brain). Both target
and FK-actual are read in the same platform-frame mm the bridge already
uses, so no unit conversion is needed here.

Meant to compare the 2x2 matrix of conditions:
    belt stationary / belt moving   ×   ADRC_BIAS_ENABLE False / True
config.ADRC_BIAS_ENABLE is read at startup and stamped into the filename/
summary automatically, so a run is self-documenting either way. --run-label
just tags belt state for the filename (e.g. "static" / "moving") — set
config.ADRC_BIAS_ENABLE and restart matlab_bridge_node + this node between
runs to cover all four combinations (bias_observer's d_hat is in-memory and
resets on restart; it lives inside DeltaMotorController, shared by every
frontend node, so it doesn't matter which frontend calls it).

Per-pick accuracy/repeatability: this node also listens to
/delta/matlab/bridge_state and, on each transition into LIFT (the bridge FSM
state right after PneumaticGripper.grip() fires — see matlab_bridge_node.py
_run_move()), snapshots the current tracking error as that pick's "accuracy"
sample — the moment the gripper actually closes, not the noisier in-flight
approach samples. After N_PICKS such events (default 10, override with
-p n_picks:=) the node stops itself and prints accuracy (mean) +
repeatability (std) across the picks, alongside the existing raw
per-sample CSV/summary.

Requires matlab_bridge_node.py already running the normal pick cycle (e.g.
via delta_main.launch.py) — this node only listens, it does not move the
robot.

Usage
-----
    ros2 run delta_main_app dynamic_tracking_log --ros-args -p run_label:=static
    ros2 run delta_main_app dynamic_tracking_log --ros-args -p run_label:=moving -p n_picks:=10
    # Stops automatically after n_picks LIFT-transition (grip) events, or Ctrl+C early.
"""

import csv
import math
import os
import statistics
import time
from datetime import datetime
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import String

from custom_messages.msg import DeltaTarget
from delta_common import config

OUTPUT_DIR = os.path.expanduser("~/delta_ws/experiment_results")
PRINT_EVERY = 20   # console status every N logged samples


class DynamicTrackingLog(Node):
    def __init__(self):
        super().__init__("dynamic_tracking_log")

        self.declare_parameter("run_label", "run")
        self.declare_parameter("n_picks", 10)
        self._run_label = self.get_parameter("run_label").value
        self._n_picks = int(self.get_parameter("n_picks").value)
        self._bias_on = bool(config.ADRC_BIAS_ENABLE)

        self._latest_target: Optional[Tuple[float, float, float]] = None
        self._latest_belt_vx: float = 0.0
        self._latest_row: Optional[dict] = None
        self._rows = []
        self._picks = []
        self._prev_state: Optional[str] = None
        self._start_t = time.time()
        self.request_stop = False

        self.create_subscription(
            DeltaTarget, "/delta/matlab/target_xyz", self._on_target, 10
        )
        self.create_subscription(
            PointStamped, "/delta/object_velocity_mm_s", self._on_velocity, 10
        )
        self.create_subscription(
            PointStamped, "/delta/matlab/fk_result", self._on_ee, 10
        )
        self.create_subscription(
            String, "/delta/matlab/bridge_state", self._on_state, 10
        )

        self.get_logger().info(
            f"dynamic_tracking_log: run_label={self._run_label!r} "
            f"ADRC_BIAS_ENABLE={self._bias_on} n_picks={self._n_picks} "
            f"— listening, Ctrl+C to finish early."
        )

    def _on_target(self, msg: DeltaTarget) -> None:
        self._latest_target = (msg.x_mm, msg.y_mm, msg.z_mm)

    def _on_velocity(self, msg: PointStamped) -> None:
        self._latest_belt_vx = msg.point.x

    def _on_ee(self, msg: PointStamped) -> None:
        if self._latest_target is None:
            return   # not actively tracking a target yet
        tx, ty, tz = self._latest_target
        ex, ey, ez = msg.point.x, msg.point.y, msg.point.z
        dx, dy, dz = ex - tx, ey - ty, ez - tz
        err_xy = math.hypot(dx, dy)
        row = {
            "t_s": time.time() - self._start_t,
            "belt_vx_mm_s": self._latest_belt_vx,
            "target_x_mm": tx, "target_y_mm": ty, "target_z_mm": tz,
            "ee_x_mm": ex, "ee_y_mm": ey, "ee_z_mm": ez,
            "err_x_mm": dx, "err_y_mm": dy, "err_z_mm": dz,
            "err_xy_mm": err_xy,
        }
        self._rows.append(row)
        self._latest_row = row

        if len(self._rows) % PRINT_EVERY == 0:
            recent = self._rows[-PRINT_EVERY:]
            mean_err = statistics.mean(r["err_xy_mm"] for r in recent)
            mean_vx = statistics.mean(r["belt_vx_mm_s"] for r in recent)
            self.get_logger().info(
                f"n={len(self._rows)}  err_xy(mean of last {PRINT_EVERY})="
                f"{mean_err:.2f}mm  belt_vx={mean_vx:+.1f}mm/s"
            )

    def _on_state(self, msg: String) -> None:
        state = msg.data
        entering_grasp = state == "LIFT" and self._prev_state != "LIFT"
        self._prev_state = state
        if not entering_grasp or self._latest_row is None:
            return

        pick = dict(self._latest_row)
        pick["pick_idx"] = len(self._picks) + 1
        self._picks.append(pick)
        self.get_logger().info(
            f"PICK {pick['pick_idx']}/{self._n_picks}: "
            f"err_xy={pick['err_xy_mm']:.2f}mm "
            f"(err_x={pick['err_x_mm']:+.2f} err_y={pick['err_y_mm']:+.2f}) "
            f"belt_vx={pick['belt_vx_mm_s']:+.1f}mm/s"
        )
        if len(self._picks) >= self._n_picks:
            self.get_logger().info(f"n_picks={self._n_picks} reached — stopping.")
            self.request_stop = True

    def finish(self) -> None:
        if not self._rows:
            self.get_logger().warning("No samples logged — nothing to save.")
            return

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bias_tag = "bias_on" if self._bias_on else "bias_off"
        path = os.path.join(
            OUTPUT_DIR, f"dynamic_tracking_{self._run_label}_{bias_tag}_{ts}.csv"
        )
        fields = list(self._rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self._rows)

        moving = [r for r in self._rows if abs(r["belt_vx_mm_s"]) > config.CONVEYOR_VX_MIN_MM_S]
        static = [r for r in self._rows if abs(r["belt_vx_mm_s"]) <= config.CONVEYOR_VX_MIN_MM_S]

        def _summ(rows, name):
            if not rows:
                return
            errs = [r["err_xy_mm"] for r in rows]
            print(
                f"{name:16s} n={len(rows):5d}  "
                f"err_xy mean={statistics.mean(errs):6.2f}mm  "
                f"std={statistics.pstdev(errs) if len(errs) > 1 else 0.0:6.2f}mm  "
                f"max={max(errs):6.2f}mm"
            )

        print(f"\nrun_label={self._run_label}  ADRC_BIAS_ENABLE={self._bias_on}")
        _summ(self._rows, "all")
        _summ(static, "belt static")
        _summ(moving, "belt moving")
        print(f"\nSaved: {path}\n")

        if self._picks:
            picks_path = os.path.join(
                OUTPUT_DIR, f"dynamic_picks_{self._run_label}_{bias_tag}_{ts}.csv"
            )
            with open(picks_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self._picks[0].keys()))
                w.writeheader()
                w.writerows(self._picks)

            xs = [p["err_x_mm"] for p in self._picks]
            ys = [p["err_y_mm"] for p in self._picks]
            ds = [p["err_xy_mm"] for p in self._picks]
            n = len(self._picks)
            print(f"── Per-pick accuracy/repeatability (n={n} picks) "
                  f"[run_label={self._run_label} bias={'on' if self._bias_on else 'off'}] ──")
            print(f"  accuracy (mean err_xy)     = {statistics.mean(ds):.2f} mm")
            print(f"  repeatability (std err_xy) = "
                  f"{statistics.pstdev(ds) if n > 1 else 0.0:.2f} mm")
            print(f"  mean err_x={statistics.mean(xs):+.2f}mm  "
                  f"std err_x={statistics.pstdev(xs) if n > 1 else 0.0:.2f}mm")
            print(f"  mean err_y={statistics.mean(ys):+.2f}mm  "
                  f"std err_y={statistics.pstdev(ys) if n > 1 else 0.0:.2f}mm")
            print(f"  max err_xy = {max(ds):.2f} mm")
            print(f"Saved: {picks_path}\n")
        else:
            print("No GRASPING events observed — no per-pick summary.\n")


def main(args=None):
    import signal
    rclpy.init(args=args)
    node = DynamicTrackingLog()

    def handle_sigint(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)
    try:
        while rclpy.ok() and not node.request_stop:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
