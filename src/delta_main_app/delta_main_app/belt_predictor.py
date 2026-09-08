#!/usr/bin/env python3
"""
belt_predictor.py — Belt velocity estimator and pick-point predictor.

Maintains a rolling window of (timestamp, x_mm) samples from the camera,
fits a linear regression to estimate belt velocity, then predicts where a
detected object will be when the robot end-effector arrives.

Pick-point timing model (medium preset, triangular profile):
    t_travel  = 2 * sqrt(dist / TRAJ_A_MAX_MM_S2)
    t_descend = 2 * sqrt(DESCEND_DIST_MM / TRAJ_A_MAX_MM_S2)
    t_total   = LATENCY_S + t_travel + t_descend
    x_pick    = x_detected + vx * t_total
"""

import math
from dataclasses import dataclass
from typing import Optional

from delta_common import config

_LATENCY_S       = 0.04    # camera capture + ROS comms latency
                            # was 0.083 (unmeasured guess) — cut ~half 2026-07-22:
                            # belt speed is now measured (26.45mm/s) but picks were
                            # still landing ahead of the object, meaning total lead
                            # (belt_offset = vx * t_total) was still too large.
                            # Re-check dx sign at grip after this change: if still
                            # ahead, cut further; if now behind, split the difference.
_DESCEND_DIST_MM = 60.0    # approach Z to belt surface (Z=-350 → Z=-410)
_WINDOW_S        = 2.0     # rolling regression window
_MIN_SAMPLES     = 5       # minimum samples before velocity is trusted


@dataclass
class PredictResult:
    x_pick:      float
    valid:       bool
    t_total:     float
    belt_offset: float
    vx_mm_s:     float


class BeltPredictor:
    """Estimates belt velocity and predicts object X at pick time."""

    def __init__(self):
        self._samples: list = []   # list of (timestamp_s, x_mm)

    # ── data ingestion ────────────────────────────────────────────────────────

    def update_velocity(self, x_mm: float, timestamp: float) -> None:
        """Record a new X observation. Call on every detection frame."""
        # If X jumped beyond what belt physics allow, the tracked object changed
        # (previous one was picked). Stale samples from the old object would corrupt
        # the velocity regression — discard them before adding the new reading.
        if self._samples:
            last_x = self._samples[-1][1]
            dt = max(timestamp - self._samples[-1][0], 0.0)
            # Belt only moves in -X.  Clear window if X jumped in the wrong
            # direction (+X) or moved more than physically possible in -X.
            # 5 mm tolerance absorbs camera noise on the same object.
            if (x_mm > last_x + 5.0 or
                    x_mm < last_x - config.CONVEYOR_VX_MAX_MM_S * dt - 5.0):
                self._samples.clear()
        self._samples.append((timestamp, x_mm))
        cutoff = timestamp - _WINDOW_S
        self._samples = [(t, x) for t, x in self._samples if t >= cutoff]

    def clear(self) -> None:
        self._samples.clear()

    # ── velocity estimate ─────────────────────────────────────────────────────

    @property
    def measured_vx(self) -> Optional[float]:
        """Current belt velocity estimate in mm/s (negative = toward EXIT).
        Returns None if insufficient data."""
        return self._estimate_vx()

    def _estimate_vx(self) -> Optional[float]:
        if len(self._samples) < _MIN_SAMPLES:
            return None

        times = [s[0] for s in self._samples]
        xs    = [s[1] for s in self._samples]
        t_mean = sum(times) / len(times)
        x_mean = sum(xs)    / len(xs)

        num   = sum((t - t_mean) * (x - x_mean) for t, x in zip(times, xs))
        denom = sum((t - t_mean) ** 2            for t in times)

        if denom < 1e-9:
            return None

        vx = num / denom

        # Sanity bounds from config
        if abs(vx) < config.CONVEYOR_VX_MIN_MM_S:
            return 0.0
        if abs(vx) > config.CONVEYOR_VX_MAX_MM_S:
            vx = math.copysign(config.CONVEYOR_VX_MAX_MM_S, vx)

        return vx

    # ── pick-point prediction ─────────────────────────────────────────────────

    def predict(self, y_mm: float, x_mm: float, current_robot_x: float = 0.0) -> PredictResult:
        """
        Predict where the object will be when the robot EEF arrives.

        Iterates 4 times: x_pick depends on t_travel, which depends on
        dist to x_pick. Converges in 3-4 iterations.

        Parameters
        ----------
        y_mm : lateral (across-belt) coordinate — passed through unchanged,
            only used for the workspace bound check.
        x_mm : belt-axis coordinate — the one being predicted.
        current_robot_x : robot's current position along the belt axis.

        Returns PredictResult with valid=False if velocity is unknown or the
        predicted pick point is outside the workspace.
        """
        vx = self._estimate_vx()
        if vx is None:
            vx = -config.CONVEYOR_BELT_SPEED_MM_S

        a = max(config.TRAJ_A_MAX_MM_S2, 1.0)
        t_descend = 2.0 * math.sqrt(_DESCEND_DIST_MM / a)

        x_pred = x_mm
        t_travel = 0.0
        for _ in range(4):
            dist_mm  = abs(x_pred - current_robot_x)
            t_travel = 2.0 * math.sqrt(dist_mm / a)
            t_total  = _LATENCY_S + t_travel + t_descend
            x_pred   = x_mm + vx * t_total

        belt_offset = x_pred - x_mm
        valid = (
            abs(y_mm)  <= config.Y_LIMIT
            and abs(x_pred) <= config.X_LIMIT
        )

        return PredictResult(
            x_pick=x_pred,
            valid=valid,
            t_total=t_total,
            belt_offset=belt_offset,
            vx_mm_s=vx,
        )
