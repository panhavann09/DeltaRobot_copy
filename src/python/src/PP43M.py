#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import signal

from fk_ik import delta_calcInverse, delta_calcForward, e, f, re, rf

# --- Import your SDK (same style you used before) ---
try:
    from robstride_dynamics import RobstrideBus, Motor, ParameterType
except ImportError:
    from bus import RobstrideBus, Motor
    from protocol import ParameterType

# ---------------- CONFIG ----------------
CAN_PORT = "can1"

MOTOR_IDS = [1, 2, 3]
MOTOR_NAMES = [f"motor_{mid}" for mid in MOTOR_IDS]

# PP settings (RS00 manual + your protocol list)
VEL_MAX = 1.0        # rad/s
ACC_SET = 2.0       # rad/s^2

VERIFY_DELAY = 0.30  # seconds after command
POS_TOL_MM = 2.0     # FK error tolerance

# Workspace safety limits (your envelope)
X_LIMIT = 151.563
Y_LIMIT = 151.563
Z_MIN = -500.0
Z_MAX = -196.875

running = True


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def check_workspace(x, y, z):
    return (abs(x) <= X_LIMIT) and (abs(y) <= Y_LIMIT) and (Z_MIN <= z <= Z_MAX)


def setup_pp_mode(bus: RobstrideBus):
    # PP mode uses:
    # MODE = run_mode = 1
    # vel_max (0x7024)
    # acc_set (0x7025)
    for name in MOTOR_NAMES:
        print(f"⚙️ Config PP: {name}")
        bus.write(name, ParameterType.MODE, 1)  # run_mode = 1
        bus.write(name, ParameterType.PP_VELOCITY_MAX, float(VEL_MAX))
        bus.write(name, ParameterType.PP_ACCELERATION_TARGET, float(ACC_SET))


def enable_all(bus: RobstrideBus):
    for name in MOTOR_NAMES:
        print(f"⚡ Enable: {name}")
        bus.enable(name)
        time.sleep(0.05)


def disable_all(bus: RobstrideBus):
    for name in MOTOR_NAMES:
        try:
            bus.disable(name)
        except Exception as e:
            print(f"⚠ disable {name} failed: {e}")


def init_zero(bus: RobstrideBus):
    # Your request: start with all motors at 0°
    # In PP mode, we command lof_ref = 0 rad.
    print("\n🏠 Init -> command all joints to 0 rad")
    for name in MOTOR_NAMES:
        bus.write(name, ParameterType.POSITION_TARGET, 0.0)  # lof_ref = 0 rad

    time.sleep(1.2)

    # Verify
    for name in MOTOR_NAMES:
        pos = float(bus.read(name, ParameterType.MECHANICAL_POSITION))  # mechPos rad
        print(f"{name}: mechPos = {math.degrees(pos):.2f}°")


def move_xyz(bus: RobstrideBus, x, y, z):
    print("\n--------------------------------")
    print(f"XYZ cmd: ({x:.2f}, {y:.2f}, {z:.2f}) mm")

    if not check_workspace(x, y, z):
        print("❌ Outside workspace limits")
        return

    # IK (degrees)
    st, t1, t2, t3 = delta_calcInverse(x, y, z, e, f, re, rf)
    if st != 0:
        print("❌ IK failed (no solution)")
        return

    print(f"IK θ(deg) = [{t1:.2f}, {t2:.2f}, {t3:.2f}]")

    # Convert to radians for lof_ref
    targets = [math.radians(t1), math.radians(t2), math.radians(t3)]

    # Write lof_ref for each motor
    for name, rad in zip(MOTOR_NAMES, targets):
        bus.write(name, ParameterType.POSITION_TARGET, float(rad))

    time.sleep(VERIFY_DELAY)

    # Read mechPos (rad) feedback
    fb_rad = []
    for name in MOTOR_NAMES:
        fb_rad.append(float(bus.read(name, ParameterType.MECHANICAL_POSITION)))

    fb_deg = [math.degrees(r) for r in fb_rad]
    print(f"FB θ(deg) = [{fb_deg[0]:.2f}, {fb_deg[1]:.2f}, {fb_deg[2]:.2f}]")

    # FK expects degrees
    st_fk, x_fk, y_fk, z_fk = delta_calcForward(fb_deg[0], fb_deg[1], fb_deg[2], e, f, re, rf)
    if st_fk != 0:
        print("❌ FK failed")
        return

    print(f"FK xyz = ({x_fk:.2f}, {y_fk:.2f}, {z_fk:.2f}) mm")

    err = math.sqrt((x - x_fk)**2 + (y - y_fk)**2 + (z - z_fk)**2)
    print(f"Error = {err:.2f} mm")

    if err > POS_TOL_MM:
        print("⚠ WARNING: error above tolerance")
    else:
        print("✅ Verified")


def on_sigint(sig, frame):
    global running
    running = False


def main():
    global running
    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)

    motors = {name: Motor(id=mid, model="rs-00") for name, mid in zip(MOTOR_NAMES, MOTOR_IDS)}
    calibration = {name: {"direction": 1, "homing_offset": 0.0} for name in MOTOR_NAMES}

    bus = RobstrideBus(CAN_PORT, motors, calibration)
    bus.connect(handshake=False)

    try:
        setup_pp_mode(bus)
        enable_all(bus)
        init_zero(bus)

        print("\n✅ PP mode ready.")
        print("Commands:")
        print("  xyz x y z        (mm)")
        print("  zero             (all joints to 0)")
        print("  fb               (print mechPos)")
        print("  q                quit\n")

        while running:
            line = input(">> ").strip().lower()
            if not line:
                continue
            if line in ("q", "quit", "exit"):
                break

            if line == "zero":
                init_zero(bus)
                continue

            if line == "fb":
                for name in MOTOR_NAMES:
                    pos = float(bus.read(name, ParameterType.MECHANICAL_POSITION))
                    print(f"{name}: {math.degrees(pos):.2f}°")
                continue

            if line.startswith("xyz "):
                try:
                    _, xs, ys, zs = line.split()
                    move_xyz(bus, float(xs), float(ys), float(zs))
                except Exception:
                    print("❌ Usage: xyz x y z")
                continue

            print("❌ Unknown command")

    finally:
        print("\n🛑 Stopping -> command 0 rad then disable")
        try:
            for name in MOTOR_NAMES:
                bus.write(name, ParameterType.POSITION_TARGET, 0.0)
            time.sleep(0.8)
        except Exception as e:
            print(f"⚠ failed to command zero: {e}")

        disable_all(bus)
        bus.disconnect(disable_torque=False)
        print("👋 bye")


if __name__ == "__main__":
    main()
