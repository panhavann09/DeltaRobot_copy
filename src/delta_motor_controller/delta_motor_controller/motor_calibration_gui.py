#!/usr/bin/env python3
"""
motor_calibration_gui.py — Motor Calibration GUI for the Delta Robot.

Tkinter graphical tool for homing and configuring the three RobStride RS-00
motor drivers via CAN bus (SocketCAN, 1 Mbit/s).

Features:
    - Connect / Disconnect to CAN bus
    - Enable / Disable all motors
    - Home all motors to 0 degrees
    - Set individual motor joint-angle targets with joint-limit safety check
    - Configure PP-mode velocity and acceleration
    - Live position readback from motor encoders at 10 Hz

Usage:
    python3 motor_calibration_gui.py
"""

import math
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

try:
    from robstride_dynamics import Motor, ParameterType, RobstrideBus
except ImportError:
    from bus import RobstrideBus, Motor
    from protocol import ParameterType

from delta_common import config

# ── Hardware constants ────────────────────────────────────────────────────────

CAN_PORT    = "can1"
MOTOR_IDS   = [1, 2, 3]
MOTOR_NAMES = [f"motor_{i}" for i in MOTOR_IDS]
THETA_MIN   = config.THETA1_MIN   # degrees  (shared for all three arms)
THETA_MAX   = config.THETA1_MAX
POLL_HZ     = 10                  # live readback rate


class MotorCalibrationGUI:
    """
    Main application window.

    All CAN I/O runs in daemon threads; GUI updates are marshalled back to
    the Tk event loop via root.after() so the UI stays responsive.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Delta Robot — Motor Calibration")
        self.root.resizable(False, False)

        self.bus: RobstrideBus | None = None
        self.connected = False
        self._lock = threading.Lock()
        self._poll_running = False

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        PAD = {"padx": 8, "pady": 4}

        # ── CAN connection ────────────────────────────────────────────────────
        conn = ttk.LabelFrame(self.root, text="CAN Connection", padding=6)
        conn.grid(row=0, column=0, columnspan=3, sticky="ew", **PAD)

        ttk.Label(conn, text="Channel:").grid(row=0, column=0, sticky="w")
        self._channel_var = tk.StringVar(value=CAN_PORT)
        ttk.Entry(conn, textvariable=self._channel_var, width=10).grid(
            row=0, column=1, padx=4
        )
        self._btn_connect = ttk.Button(conn, text="Connect", command=self._connect)
        self._btn_connect.grid(row=0, column=2, padx=4)
        self._btn_disconnect = ttk.Button(
            conn, text="Disconnect", command=self._disconnect, state="disabled"
        )
        self._btn_disconnect.grid(row=0, column=3, padx=4)

        # ── PP-mode settings ──────────────────────────────────────────────────
        pp = ttk.LabelFrame(self.root, text="PP Mode Settings", padding=6)
        pp.grid(row=1, column=0, columnspan=3, sticky="ew", **PAD)

        ttk.Label(pp, text="Velocity max (rad/s):").grid(row=0, column=0, sticky="w")
        self._vel_var = tk.StringVar(value=str(config.MOTOR_VEL_MAX))
        ttk.Entry(pp, textvariable=self._vel_var, width=8).grid(row=0, column=1)

        ttk.Label(pp, text="Acceleration (rad/s²):").grid(
            row=1, column=0, sticky="w", pady=2
        )
        self._acc_var = tk.StringVar(value=str(config.MOTOR_ACC_SET))
        ttk.Entry(pp, textvariable=self._acc_var, width=8).grid(row=1, column=1)

        ttk.Button(pp, text="Apply", command=self._apply_pp_settings).grid(
            row=0, column=2, rowspan=2, padx=8
        )

        # ── Per-motor controls ────────────────────────────────────────────────
        motors = ttk.LabelFrame(self.root, text="Motor Control", padding=6)
        motors.grid(row=2, column=0, columnspan=3, sticky="ew", **PAD)

        for col, header in enumerate(["Motor", "Target (°)", "Set", "Position (°)"]):
            ttk.Label(motors, text=header, font=("", 9, "bold")).grid(
                row=0, column=col, padx=6, pady=2
            )

        self._target_vars: list[tk.StringVar] = []
        self._pos_labels: list[ttk.Label] = []

        for i, (mid, name) in enumerate(zip(MOTOR_IDS, MOTOR_NAMES)):
            row = i + 1
            ttk.Label(motors, text=f"Motor {mid}").grid(row=row, column=0, padx=6, pady=3)

            tv = tk.StringVar(value="0.0")
            self._target_vars.append(tv)
            ttk.Entry(motors, textvariable=tv, width=8).grid(row=row, column=1, padx=4)

            ttk.Button(
                motors,
                text="Set",
                command=lambda n=name, v=tv: self._set_angle(n, v),
            ).grid(row=row, column=2, padx=4)

            lbl = ttk.Label(motors, text="—", width=10)
            lbl.grid(row=row, column=3, padx=6)
            self._pos_labels.append(lbl)

        # ── Global action buttons ─────────────────────────────────────────────
        act = ttk.Frame(self.root, padding=6)
        act.grid(row=3, column=0, columnspan=3, **PAD)

        for col, (label, cmd) in enumerate([
            ("Home All",    self._home_all),
            ("Enable All",  self._enable_all),
            ("Disable All", self._disable_all),
        ]):
            ttk.Button(act, text=label, command=cmd, width=14).grid(
                row=0, column=col, padx=6
            )

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Disconnected")
        ttk.Label(
            self.root,
            textvariable=self._status_var,
            relief="sunken",
            anchor="w",
            padding=(4, 2),
        ).grid(
            row=4, column=0, columnspan=3, padx=8, pady=(0, 6), sticky="ew"
        )

        self.root.columnconfigure(0, weight=1)

    # ── CAN lifecycle ─────────────────────────────────────────────────────────

    def _connect(self) -> None:
        channel = self._channel_var.get().strip()
        try:
            vel = float(self._vel_var.get())
            acc = float(self._acc_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid velocity or acceleration value.")
            return

        self._status_var.set(f"Connecting to {channel}…")

        def _worker() -> None:
            try:
                motors = {
                    name: Motor(id=mid, model="rs-00")
                    for name, mid in zip(MOTOR_NAMES, MOTOR_IDS)
                }
                calib = {
                    name: {"direction": 1, "homing_offset": 0.0}
                    for name in MOTOR_NAMES
                }
                bus = RobstrideBus(channel, motors, calib)
                bus.connect(handshake=False)

                for name in MOTOR_NAMES:
                    bus.write(name, ParameterType.MODE, 1)
                    bus.write(name, ParameterType.PP_VELOCITY_MAX, float(vel))
                    bus.write(name, ParameterType.PP_ACCELERATION_TARGET, float(acc))
                    bus.enable(name)
                    time.sleep(0.05)

                with self._lock:
                    self.bus = bus
                    self.connected = True

                self.root.after(0, self._on_connected)
                self._start_poll()

            except Exception as exc:
                self.root.after(
                    0, lambda: self._status_var.set(f"Connection failed: {exc}")
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _disconnect(self) -> None:
        self._poll_running = False

        def _worker() -> None:
            with self._lock:
                if self.bus:
                    try:
                        for name in MOTOR_NAMES:
                            self.bus.write(name, ParameterType.POSITION_TARGET, 0.0)
                        time.sleep(0.3)
                        for name in MOTOR_NAMES:
                            self.bus.disable(name)
                        self.bus.disconnect(disable_torque=True)
                    except Exception:
                        pass
                    self.bus = None
                    self.connected = False
            self.root.after(0, self._on_disconnected)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_connected(self) -> None:
        self._btn_connect.config(state="disabled")
        self._btn_disconnect.config(state="normal")
        self._status_var.set(
            f"Connected — {self._channel_var.get()} — 3 motors active"
        )

    def _on_disconnected(self) -> None:
        self._btn_connect.config(state="normal")
        self._btn_disconnect.config(state="disabled")
        for lbl in self._pos_labels:
            lbl.config(text="—")
        self._status_var.set("Disconnected")

    # ── Motor commands ────────────────────────────────────────────────────────

    def _apply_pp_settings(self) -> None:
        try:
            vel = float(self._vel_var.get())
            acc = float(self._acc_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid velocity or acceleration value.")
            return

        def _worker() -> None:
            with self._lock:
                if not self.connected:
                    return
                for name in MOTOR_NAMES:
                    self.bus.write(name, ParameterType.PP_VELOCITY_MAX, float(vel))
                    self.bus.write(name, ParameterType.PP_ACCELERATION_TARGET, float(acc))
            self.root.after(
                0,
                lambda: self._status_var.set(
                    f"PP settings applied: vel={vel} rad/s  acc={acc} rad/s²"
                ),
            )

        threading.Thread(target=_worker, daemon=True).start()

    def _set_angle(self, motor_name: str, var: tk.StringVar) -> None:
        try:
            angle_deg = float(var.get())
        except ValueError:
            messagebox.showerror("Error", "Enter a numeric angle in degrees.")
            return

        if not (THETA_MIN <= angle_deg <= THETA_MAX):
            messagebox.showerror(
                "Joint Limit",
                f"Angle {angle_deg:.1f}° is outside the safe range "
                f"[{THETA_MIN}°, {THETA_MAX}°].",
            )
            return

        rad = math.radians(angle_deg)

        def _worker() -> None:
            with self._lock:
                if not self.connected:
                    return
                self.bus.write(motor_name, ParameterType.POSITION_TARGET, float(rad))
            self.root.after(
                0,
                lambda: self._status_var.set(
                    f"{motor_name}: target → {angle_deg:.1f}°"
                ),
            )

        threading.Thread(target=_worker, daemon=True).start()

    def _home_all(self) -> None:
        def _worker() -> None:
            with self._lock:
                if not self.connected:
                    return
                for name in MOTOR_NAMES:
                    self.bus.write(name, ParameterType.POSITION_TARGET, 0.0)
                time.sleep(1.0)
            self.root.after(
                0, lambda: self._status_var.set("All motors homed to 0°")
            )

        threading.Thread(target=_worker, daemon=True).start()

    def _enable_all(self) -> None:
        def _worker() -> None:
            with self._lock:
                if not self.connected:
                    return
                for name in MOTOR_NAMES:
                    self.bus.enable(name)
                    time.sleep(0.05)
            self.root.after(
                0, lambda: self._status_var.set("All motors enabled")
            )

        threading.Thread(target=_worker, daemon=True).start()

    def _disable_all(self) -> None:
        def _worker() -> None:
            with self._lock:
                if not self.connected:
                    return
                for name in MOTOR_NAMES:
                    self.bus.disable(name)
                    time.sleep(0.05)
            self.root.after(
                0, lambda: self._status_var.set("All motors disabled")
            )

        threading.Thread(target=_worker, daemon=True).start()

    # ── Live position readback ────────────────────────────────────────────────

    def _start_poll(self) -> None:
        self._poll_running = True
        threading.Thread(target=self._poll_loop, daemon=True).start()

    def _poll_loop(self) -> None:
        interval = 1.0 / POLL_HZ
        while self._poll_running:
            positions: list[float | None] = []
            with self._lock:
                if not self.connected:
                    break
                for name in MOTOR_NAMES:
                    try:
                        rad = float(
                            self.bus.read(name, ParameterType.MECHANICAL_POSITION)
                        )
                        positions.append(math.degrees(rad))
                    except Exception:
                        positions.append(None)

            for lbl, pos in zip(self._pos_labels, positions):
                text = f"{pos:.2f}°" if pos is not None else "err"
                self.root.after(0, lambda lb=lbl, t=text: lb.config(text=t))

            time.sleep(interval)


def main() -> None:
    root = tk.Tk()
    MotorCalibrationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
