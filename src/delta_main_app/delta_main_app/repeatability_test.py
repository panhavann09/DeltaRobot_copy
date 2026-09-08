#!/usr/bin/env python3
"""
repeatability_test.py — Point-to-Point Tracking Repeatability Test

Exact methodology from: Yamtuan et al. JRC 2023, Section III-B
    Figs 14, 15, 16 — Tables IV, V

No camera. Laser dot on paper is a visual reference only.

Paper's experiment
------------------
    Motion : HOME(0,0,Z) → pos_i (record θ) → HOME → pos_i (record θ) … × 6
    Speed  : ~214 mm/s  (set VEL_MAX / ACC_SET to match)
    Records: encoder counts per motor at every target stop
    Outputs:
        Fig 14 — encoder count scatter at HOME and TARGET, ±1σ, per motor
        Fig 15 — end-effector XY scatter + 2σ ellipse, per position
        Fig 16 — θ vs time (experiment=blue vs IK=red) for whole sequence

Run directly — no ROS launch needed, no camera/main_app/matlab_bridge_node
required (this script drives the arm itself via DeltaMotorController):
    python3 repeatability_test.py --run-label static
    python3 repeatability_test.py --run-label moving
config.ADRC_BIAS_ENABLE is read at startup and stamped into the output
filename/console header automatically (bias_observer's d_hat lives inside
DeltaMotorController, so it applies here too) — set it in config.py and
restart between runs to cover bias on/off.  --run-label just tags the belt
condition for the filename; since this script never looks at the belt (no
camera), "moving" means physically running the conveyor idle in the
background while the arm does the same point-to-point moves, to see
whether belt vibration disturbs positioning — the script does not start/
stop it for you.
"""

import argparse
import csv
import math
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from delta_common import config
from delta_common.fk_ik import check_workspace, solve_ik_mm
from delta_motor_controller.motor_controller import DeltaMotorController

# ── Parameters — match these to the paper ─────────────────────────────────────

EXPERIMENT_Z_MM = -450.0   # tip-frame Z; move_xyz adds EE_OFFSET_Z_MM(150) -> platform
                            # Z=-300 (config.py current value — re-verify this comment
                            # if EE_OFFSET_Z_MM ever changes, it has drifted out of sync
                            # with config.py before)

HOME_X, HOME_Y = 0.0, 0.0

# THETA1/2/3_MIN=-5deg (config.py) bounds the reachable envelope at
# EXPERIMENT_Z_MM — re-verified 2026-08-22 by calling solve_ik_mm()/
# check_workspace() directly at the real platform Z=-300 for every point
# below; each keeps >=3.99deg margin on every joint against both the -5deg
# and 90deg limits.
TARGET_POSITIONS: List[Tuple[float, float]] = [
    (   0.0,   50.0),   # pos1  +Y                    thetas=(18.3,29.8, 5.6) margin 10.57deg
    ( -50.0,    0.0),   # pos2  -X                     thetas=( 3.5,25.1,25.1) margin 8.51deg
    ( -50.0,  -50.0),   # pos3  -X-Y                   thetas=( 4.7,14.2,36.8) margin 9.68deg
    (  50.0,    0.0),   # pos4  +X                     thetas=(31.4,11.1,11.1) margin 16.07deg
    ( -50.0,   100.0),  # pos5  edge -X+Y (120deg spoke, r=30) thetas=( 8.2,49.0, 4.6) margin 9.64deg
    (   0.0,  -50.0),   # pos6  -Y                     thetas=(18.3, 5.6,29.8) margin 10.57deg
    (  50.0,   50.0),   # pos7  +X+Y                   thetas=(32.5,24.4,-1.0) margin 3.99deg
    (  50.0,  -50.0),   # pos8  +X-Y                   thetas=(32.5,-1.0,24.4) margin 3.99deg
    ( -50.0,   50.0),   # pos9  -X+Y                   thetas=( 4.7,36.8,14.2) margin 9.68deg
    ( -50.0,  -100.0),  # pos10 edge -X-Y (240deg spoke, r=30) thetas=( 8.2, 4.6,49.0) margin 9.64deg
]

N_REPEATS   = 6      # paper uses 6
SETTLE_TIME = 0.8    # seconds after move before reading encoder

# Operational speed — paper runs at ~214 mm/s end-effector speed.
# Tune VEL_MAX / ACC_SET to approximate that on your robot.
VEL_MAX = 2.0    # rad/s
ACC_SET = 5.0    # rad/s²

OUTPUT_DIR = os.path.expanduser("~/delta_ws/experiment_results")


# ── IK helper (for the Fig.16 "simulation" red line) ──────────────────────────

def _ik_angles(x: float, y: float, z: float) -> Optional[Tuple[float, float, float]]:
    """Return (θ1, θ2, θ3) in degrees from IK, or None on failure."""
    ok, t1, t2, t3 = solve_ik_mm(x, y, z)
    return (t1, t2, t3) if ok else None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-label", default="static",
                         help="belt condition tag for the output filename, "
                              "e.g. static / moving (default: static)")
    parser.add_argument("--control-mode", default="pp", choices=["pp", "mit"],
                         help="motor control mode: pp = Profile Position (default, "
                              "unchanged behavior), mit = Operation Control mode "
                              "(thesis Table 4.2 run_mode=0), for the PP-vs-MIT "
                              "comparison. See config.MIT_* for gains/profile.")
    args = parser.parse_args()

    bias_on = bool(config.ADRC_BIAS_ENABLE)
    bias_tag = "bias_on" if bias_on else "bias_off"

    motor = DeltaMotorController(vel_max=VEL_MAX, acc_set=ACC_SET,
                                  control_mode=args.control_mode)
    motor.connect()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(
        OUTPUT_DIR,
        f"repeatability_{args.run_label}_{args.control_mode}_{bias_tag}_{ts}.csv"
    )

    # stopped_log  : one row per target stop (for Table IV/V and Figs 14, 15)
    # timeline_log : one row per motion event (for Fig 16)
    stopped_log:  List[Dict] = []
    timeline_log: List[Dict] = []

    print(f"\n{'='*62}")
    print(f"  Point-to-Point Tracking Repeatability (paper Sec. III-B)")
    print(f"  control_mode={args.control_mode}  run_label={args.run_label}  "
          f"ADRC_BIAS_ENABLE={bias_on}")
    print(f"  {len(TARGET_POSITIONS)} targets × {N_REPEATS} repeats")
    if args.control_mode == "mit":
        print(f"  Z = {EXPERIMENT_Z_MM} mm    "
              f"MIT_V_MAX={config.MIT_V_MAX_MMPS} mm/s  MIT_A_MAX={config.MIT_A_MAX_MMPS2} mm/s²  "
              f"Kp={config.MIT_KP}  Kd={config.MIT_KD}")
    else:
        print(f"  Z = {EXPERIMENT_Z_MM} mm    VEL={VEL_MAX} rad/s  ACC={ACC_SET} rad/s²")
    print(f"{'='*62}")

    t_start = time.monotonic()

    def _record_timeline(label: str, x: float, y: float,
                         ik_deg, fb_deg, fk_xyz):
        """Append one time-stamped row for Fig 16."""
        t = time.monotonic() - t_start
        timeline_log.append({
            "t_s":    round(t, 3),
            "label":  label,
            "x_ref":  x, "y_ref": y,
            "t1_cmd": round(ik_deg[0], 4) if ik_deg else float("nan"),
            "t2_cmd": round(ik_deg[1], 4) if ik_deg else float("nan"),
            "t3_cmd": round(ik_deg[2], 4) if ik_deg else float("nan"),
            "t1_fb":  round(fb_deg[0], 4) if fb_deg else float("nan"),
            "t2_fb":  round(fb_deg[1], 4) if fb_deg else float("nan"),
            "t3_fb":  round(fb_deg[2], 4) if fb_deg else float("nan"),
        })

    # ── Home ──────────────────────────────────────────────────────────────────
    ok, ik0, fb0, fk0, _ = motor.move(HOME_X, HOME_Y, EXPERIMENT_Z_MM)
    time.sleep(1.5)
    _record_timeline("home_init", HOME_X, HOME_Y, ik0, fb0, fk0)

    for pt_idx, (x_tgt, y_tgt) in enumerate(TARGET_POSITIONS):
        if not check_workspace(x_tgt, y_tgt, EXPERIMENT_Z_MM):
            print(f"  SKIP ({x_tgt},{y_tgt}) — outside workspace")
            continue

        print(f"\n── Target {pt_idx+1}/{len(TARGET_POSITIONS)}"
              f"  ({x_tgt:.1f}, {y_tgt:.1f}) ──")
        print(f"  {'rep':>3}  "
              f"{'θ1_cmd':>8} {'θ1_fb':>7}  "
              f"{'θ2_cmd':>8} {'θ2_fb':>7}  "
              f"{'θ3_cmd':>8} {'θ3_fb':>7}  "
              f"{'x_meas':>8} {'y_meas':>7}  {'dist':>6}")
        print("  " + "─" * 87)

        for rep in range(N_REPEATS):
            # ── HOME → TARGET ─────────────────────────────────────────────
            ok, ik_t, fb_t, fk_t, _ = motor.move(
                x_tgt, y_tgt, EXPERIMENT_Z_MM
            )
            time.sleep(SETTLE_TIME)
            _record_timeline(f"target_p{pt_idx+1}_r{rep+1}",
                             x_tgt, y_tgt, ik_t, fb_t, fk_t)

            x_meas = fk_t[0] if fk_t else float("nan")
            y_meas = fk_t[1] if fk_t else float("nan")
            dx     = x_tgt - x_meas
            dy     = y_tgt - y_meas
            dist   = math.hypot(dx, dy) if not math.isnan(dx) else float("nan")

            t1c = ik_t[0] if ik_t else float("nan")
            t2c = ik_t[1] if ik_t else float("nan")
            t3c = ik_t[2] if ik_t else float("nan")
            t1f = fb_t[0] if fb_t else float("nan")
            t2f = fb_t[1] if fb_t else float("nan")
            t3f = fb_t[2] if fb_t else float("nan")

            print(f"  {rep+1:>3}  "
                  f"{t1c:>8.3f} {t1f:>7.3f}  "
                  f"{t2c:>8.3f} {t2f:>7.3f}  "
                  f"{t3c:>8.3f} {t3f:>7.3f}  "
                  f"{x_meas:>8.3f} {y_meas:>7.3f}  {dist:>6.3f}")

            stopped_log.append({
                "target_idx": pt_idx,
                "x_tgt":   x_tgt,  "y_tgt":  y_tgt,
                "repeat":  rep + 1,
                "t1_cmd":  round(t1c, 4), "t2_cmd": round(t2c, 4),
                "t3_cmd":  round(t3c, 4),
                "t1_fb":   round(t1f, 4), "t2_fb":  round(t2f, 4),
                "t3_fb":   round(t3f, 4),
                "x_meas":  round(x_meas, 3), "y_meas": round(y_meas, 3),
                "dx":      round(dx,     3), "dy":     round(dy,    3),
                "dist":    round(dist,   3),
            })

            # ── TARGET → HOME ─────────────────────────────────────────────
            ok, ik_h, fb_h, fk_h, _ = motor.move(
                HOME_X, HOME_Y, EXPERIMENT_Z_MM
            )
            time.sleep(SETTLE_TIME)
            _record_timeline(f"home_p{pt_idx+1}_r{rep+1}",
                             HOME_X, HOME_Y, ik_h, fb_h, fk_h)

    motor.shutdown()

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    _save_csv(stopped_log,
              ["target_idx","x_tgt","y_tgt","repeat",
               "t1_cmd","t2_cmd","t3_cmd",
               "t1_fb","t2_fb","t3_fb",
               "x_meas","y_meas","dx","dy","dist"],
              csv_path)
    print(f"\nCSV (stopped) → {csv_path}")

    tl_path = csv_path.replace(".csv", "_timeline.csv")
    _save_csv(timeline_log,
              ["t_s","label","x_ref","y_ref",
               "t1_cmd","t2_cmd","t3_cmd",
               "t1_fb","t2_fb","t3_fb"],
              tl_path)
    print(f"CSV (timeline) → {tl_path}")

    _print_tables(stopped_log)

    try:
        _plot_all(stopped_log, timeline_log, csv_path.replace(".csv", ""))
    except Exception as exc:
        print(f"Plot skipped: {exc}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save_csv(rows: List[Dict], fields: List[str], path: str):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def _mean(v): return sum(v) / len(v)
def _std(v):
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1))

def _by_target(log: List[Dict]) -> Dict:
    d: Dict = {}
    for r in log:
        d.setdefault((r["x_tgt"], r["y_tgt"]), []).append(r)
    return d


# ── Console tables ─────────────────────────────────────────────────────────────

def _print_tables(log: List[Dict]):
    valid  = [r for r in log if not math.isnan(r["x_meas"])]
    by_pt  = _by_target(valid)
    sorted_pts = sorted(by_pt.keys())

    # Table IV: μ and σ of θ per motor per position
    print(f"\n{'='*72}")
    print(f"  TABLE IV — θ feedback per motor  (deg, paper uses encoder pulse)")
    print(f"  n = {len(valid[0:1] and [r for r in valid[:1]])  or 0}  "
          f"per position" if valid else "")
    print(f"{'='*72}")
    print(f"  {'Pos':>3}  {'target':>12}  "
          f"{'μ θ1':>7} {'σ θ1':>6}  "
          f"{'μ θ2':>7} {'σ θ2':>6}  "
          f"{'μ θ3':>7} {'σ θ3':>6}")
    print("  " + "─" * 65)
    for pi, (xc, yc) in enumerate(sorted_pts):
        recs = by_pt[(xc, yc)]
        t1s = [r["t1_fb"] for r in recs]
        t2s = [r["t2_fb"] for r in recs]
        t3s = [r["t3_fb"] for r in recs]
        print(f"  {pi+1:>3}  ({xc:>5.0f},{yc:>5.0f})  "
              f"{_mean(t1s):>7.3f} {_std(t1s):>6.3f}  "
              f"{_mean(t2s):>7.3f} {_std(t2s):>6.3f}  "
              f"{_mean(t3s):>7.3f} {_std(t3s):>6.3f}")

    # Table V: μ and σ of XY per position
    print(f"\n  TABLE V — end-effector XY position (mm)")
    print(f"  {'Pos':>3}  {'target':>12}  "
          f"{'μx':>8} {'σx':>6}  {'μy':>8} {'σy':>6}  "
          f"{'2σx':>7} {'2σy':>7}  {'max':>7}")
    print("  " + "─" * 72)
    for pi, (xc, yc) in enumerate(sorted_pts):
        recs = by_pt[(xc, yc)]
        xs = [r["x_meas"] for r in recs]
        ys = [r["y_meas"] for r in recs]
        sx = _std(xs);  sy = _std(ys)
        md = max(r["dist"] for r in recs)
        print(f"  {pi+1:>3}  ({xc:>5.0f},{yc:>5.0f})  "
              f"{_mean(xs):>+8.3f} {sx:>6.3f}  "
              f"{_mean(ys):>+8.3f} {sy:>6.3f}  "
              f"{2*sx:>7.3f} {2*sy:>7.3f}  {md:>7.3f}")
    print("=" * 72)


# ── Plots ──────────────────────────────────────────────────────────────────────

def _plot_all(stopped: List[Dict], timeline: List[Dict], prefix: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    valid     = [r for r in stopped if not math.isnan(r["x_meas"])]
    by_pt     = _by_target(valid)
    positions = sorted(by_pt.keys())
    n_pos     = len(positions)
    colors    = plt.cm.tab10.colors
    rep_markers = ["x", "o", "^", "s", "D", "v"]
    rep_colors  = [colors[i] for i in range(N_REPEATS)]

    # ── Fig 15: XY scatter + 2σ ellipse per position ──────────────────────────
    fig15, axes15 = plt.subplots(1, n_pos, figsize=(4.5 * n_pos, 5))
    if n_pos == 1:
        axes15 = [axes15]
    fig15.suptitle("Fig 15 — End-effector position distribution (2σ ellipse)", fontsize=12)

    for ax, (xc, yc), ci in zip(axes15, positions, range(n_pos)):
        recs = by_pt[(xc, yc)]
        xs = [r["x_meas"] for r in recs]
        ys = [r["y_meas"] for r in recs]
        mx = _mean(xs);  my = _mean(ys)
        sx = _std(xs);   sy = _std(ys)

        ax.set_title(f"Pos {ci+1}: ({xc:.0f},{yc:.0f}) mm", fontsize=9)
        ax.set_xlabel("x (mm)");  ax.set_ylabel("y (mm)")
        ax.set_aspect("equal");   ax.grid(True, linestyle="--", alpha=0.4)

        for ri, (x_, y_) in enumerate(zip(xs, ys)):
            ax.plot(x_, y_, marker=rep_markers[ri % len(rep_markers)],
                    color=rep_colors[ri], markersize=8,
                    label=f"round{ri+1}", linestyle="none")

        ax.plot(mx, my, "o", color="black", markersize=8, zorder=5, label="mean (○)")
        if len(xs) > 1:
            ax.add_patch(mpatches.Ellipse(
                (mx, my), 2 * sx, 2 * sy,
                edgecolor="gray", facecolor="lightgray",
                linewidth=1.5, linestyle="-", alpha=0.4, label="2σ",
            ))
        ax.annotate(f"σx={sx:.3f}\nσy={sy:.3f}",
                    xy=(0.05, 0.05), xycoords="axes fraction", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")

    fig15.tight_layout()
    p = prefix + "_fig15_xy_scatter.png"
    fig15.savefig(p, dpi=150); plt.close(fig15)
    print(f"Fig 15 → {p}")

    # ── Fig 14: encoder scatter at pos0 (home) and pos1 (target), ±1σ ─────────
    # Paper shows this only for pos1, with position 0 = home on x-axis.
    # We generalise: one column per target, 3 rows (one per motor).
    fig14, axes14 = plt.subplots(3, n_pos, figsize=(4.5 * n_pos, 11), squeeze=False)
    fig14.suptitle("Fig 14 — Motor θ feedback distribution at HOME and TARGET", fontsize=12)

    motor_keys   = ["t1_fb", "t2_fb", "t3_fb"]
    motor_labels = ["Motor 1 (θ₁)", "Motor 2 (θ₂)", "Motor 3 (θ₃)"]

    # Collect home θ values from the timeline log
    home_rows = [r for r in timeline if r["label"].startswith("home")]
    home_t1 = [r["t1_fb"] for r in home_rows if not math.isnan(r["t1_fb"])]
    home_t2 = [r["t2_fb"] for r in home_rows if not math.isnan(r["t2_fb"])]
    home_t3 = [r["t3_fb"] for r in home_rows if not math.isnan(r["t3_fb"])]
    home_vals = {"t1_fb": home_t1, "t2_fb": home_t2, "t3_fb": home_t3}

    for ci, (xc, yc) in enumerate(positions):
        recs = by_pt[(xc, yc)]

        for mi, (key, mlabel) in enumerate(zip(motor_keys, motor_labels)):
            ax = axes14[mi][ci]
            ax.set_title(f"{mlabel}\nPos {ci+1}: ({xc:.0f},{yc:.0f})", fontsize=8)
            ax.set_xlabel("Position  [0=HOME  1=TARGET]")
            ax.set_ylabel("θ (deg)")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["0\n(home)", "1\n(target)"])

            tgt_vals = [r[key] for r in recs]

            # Scatter individual rounds at each position
            for ri, v in enumerate(home_vals[key][:N_REPEATS]):
                ax.plot(0, v,
                        marker=rep_markers[ri % len(rep_markers)],
                        color=rep_colors[ri], markersize=8, linestyle="none",
                        label=f"round{ri+1}" if (mi == 0 and ci == 0) else "")
            for ri, v in enumerate(tgt_vals):
                ax.plot(1, v,
                        marker=rep_markers[ri % len(rep_markers)],
                        color=rep_colors[ri], markersize=8, linestyle="none")

            # Mean ± 1σ error bar (paper's solid dot)
            for x_pos, vals in [(0, home_vals[key][:N_REPEATS]), (1, tgt_vals)]:
                if vals:
                    mu = _mean(vals); sigma = _std(vals)
                    ax.errorbar(x_pos, mu, yerr=sigma,
                                fmt="k.", markersize=12, capsize=6,
                                elinewidth=1.5, zorder=5)

    # Single shared legend for round colours
    handles = [
        mpatches.Patch(color=rep_colors[i], label=f"round{i+1}")
        for i in range(N_REPEATS)
    ]
    fig14.legend(handles=handles, loc="lower center",
                 ncol=N_REPEATS, fontsize=8, title="Round")

    fig14.tight_layout(rect=[0, 0.04, 1, 1])
    p = prefix + "_fig14_motor_scatter.png"
    fig14.savefig(p, dpi=150); plt.close(fig14)
    print(f"Fig 14 → {p}")

    # ── Fig 16: θ vs time (experiment=blue, IK simulation=red) ───────────────
    # One row per motor, showing the entire sequence for pos1 (first target).
    if not timeline:
        return

    # Filter to first target's events
    tgt_label = "target_p1"
    home_label = "home_p1"
    seq = [r for r in timeline
           if tgt_label in r["label"] or home_label in r["label"]
           or r["label"] == "home_init"]
    if not seq:
        return

    fig16, axes16 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig16.suptitle(
        "Fig 16 — θ vs time for repeats at Pos 1\n"
        "Blue = experiment (encoder)   Red = IK commanded (simulation)",
        fontsize=11,
    )

    ts_arr = np.array([r["t_s"] for r in seq])

    for mi, (fb_key, cmd_key, mlabel) in enumerate(zip(
        ["t1_fb", "t2_fb", "t3_fb"],
        ["t1_cmd", "t2_cmd", "t3_cmd"],
        ["θ₁₁", "θ₂₁", "θ₃₁"],
    )):
        ax = axes16[mi]
        fb_arr  = np.array([r[fb_key]  for r in seq])
        cmd_arr = np.array([r[cmd_key] for r in seq])

        ax.plot(ts_arr, fb_arr,  "b.-", linewidth=1.5, markersize=8,
                label="Experiment (encoder)")
        ax.plot(ts_arr, cmd_arr, "r--", linewidth=1.5, markersize=6,
                label="Simulation (IK)")
        ax.set_ylabel(f"{mlabel} (deg)")
        ax.grid(True, linestyle="--", alpha=0.4)
        if mi == 0:
            ax.legend(fontsize=9)

    axes16[-1].set_xlabel("Time (s)")
    fig16.tight_layout()
    p = prefix + "_fig16_theta_time.png"
    fig16.savefig(p, dpi=150); plt.close(fig16)
    print(f"Fig 16 → {p}")


if __name__ == "__main__":
    main()
