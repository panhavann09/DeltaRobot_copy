#!/usr/bin/env python3

# CHANGES: [2.4] centralized workspace check via delta_common.fk_ik.check_workspace

import math
import time

from delta_common import config
from delta_common.bias_observer import BiasObserver
from delta_common.fk_ik import check_workspace, solve_fk_mm, solve_ik_mm
from delta_common.trajectory import linear_waypoints

try:
    from robstride_dynamics import Motor, ParameterType, RobstrideBus
except ImportError:
    from bus import RobstrideBus, Motor
    from protocol import ParameterType


class DeltaMotorController:
    def __init__(self, can_port="can1", vel_max=1.0, acc_set=2.0, verify_delay=1.50,
                 control_mode="pp"):
        self.CAN_PORT = can_port
        self.connected = False
        self._shutdown_done = False

        assert control_mode in ("pp", "mit"), f"unknown control_mode: {control_mode!r}"
        self.control_mode = control_mode

        self.MOTOR_IDS = [1, 2, 3]
        self.MOTOR_NAMES = [f"motor_{mid}" for mid in self.MOTOR_IDS]

        self.VEL_MAX = vel_max
        self.ACC_SET = acc_set

        self.VERIFY_DELAY = verify_delay
        self.POS_TOL_MM = 0.5

        motors = {
            name: Motor(id=mid, model="rs-00")
            for name, mid in zip(self.MOTOR_NAMES, self.MOTOR_IDS)
        }
        calibration = {
            name: {"direction": 1, "homing_offset": 0.0}
            for name in self.MOTOR_NAMES
        }

        self.bus = RobstrideBus(self.CAN_PORT, motors, calibration)
        self._bias = BiasObserver()

    def within_joint_limits(self, t1, t2, t3):
        return (
            config.THETA1_MIN <= t1 <= config.THETA1_MAX
            and config.THETA2_MIN <= t2 <= config.THETA2_MAX
            and config.THETA3_MIN <= t3 <= config.THETA3_MAX
        )

    def connect(self):
        self.bus.connect(handshake=False)
        if self.control_mode == "mit":
            self.setup_mit_mode()
        else:
            self.setup_pp_mode()
        self.enable_all()
        if self.control_mode != "mit":
            # init_zero() writes POSITION_TARGET (loc_ref), the PP/CSP register --
            # meaningless in Operation Control mode, which reads pset only from the
            # operation-control frame (write_operation_frame), never from loc_ref.
            self.init_zero()
        self.connected = True
        self._shutdown_done = False

    def setup_pp_mode(self):
        for name in self.MOTOR_NAMES:
            self.bus.write(name, ParameterType.MODE, 1)
            self.bus.write(name, ParameterType.PP_VELOCITY_MAX, float(self.VEL_MAX))
            self.bus.write(name, ParameterType.PP_ACCELERATION_TARGET, float(self.ACC_SET))

    def setup_mit_mode(self):
        """Operation Control mode (run_mode=0, thesis Table 4.2) -- host streams
        pset/vset/Kp/Kd/tau_ff every frame via write_operation_frame(); no firmware
        profiling or motion limits, both are done host-side (see move_xyz_mit)."""
        for name in self.MOTOR_NAMES:
            self.bus.write(name, ParameterType.MODE, 0)

    def enable_all(self):
        for name in self.MOTOR_NAMES:
            self.bus.enable(name)
            time.sleep(0.05)

    def init_zero(self):
        for name in self.MOTOR_NAMES:
            self.bus.write(name, ParameterType.POSITION_TARGET, 0.0)
        time.sleep(1.0)

    def get_bias(self):
        """Current ADRC bias estimate (dx, dy, dz) in mm."""
        return self._bias.correction()

    def reset_bias(self):
        """Zero the ADRC bias estimate (e.g. between bench-test runs)."""
        self._bias.reset()

    def move_xyz(self, x, y, z, raw=False):
        print(f"\nXYZ -> ({x:.2f}, {y:.2f}, {z:.2f})")
        if not raw:
            if config.EE_OFFSET_X_MM or config.EE_OFFSET_Y_MM or config.EE_OFFSET_Z_MM:
                x += config.EE_OFFSET_X_MM
                y += config.EE_OFFSET_Y_MM
                z += config.EE_OFFSET_Z_MM
                print(f"EE offset applied -> ({x:.2f}, {y:.2f}, {z:.2f})")

        if not check_workspace(x, y, z):
            print(f"Workspace reject: XYZ=({x:.1f},{y:.1f},{z:.1f}) outside limits")
            return False, None, None, None, None

        # ADRC-flavored bias correction: bias the commanded target only,
        # never the true desired (x, y, z) -- those stay as-is so the
        # tolerance/err computation below and the caller's `ok` semantics
        # keep meaning "distance from what was actually wanted."
        x_cmd, y_cmd, z_cmd = x, y, z
        if config.ADRC_BIAS_ENABLE:
            dx, dy, dz = self._bias.correction()
            if dx or dy or dz:
                cx, cy, cz = x - dx, y - dy, z - dz
                if check_workspace(cx, cy, cz):
                    x_cmd, y_cmd, z_cmd = cx, cy, cz
                    print(f"ADRC bias applied -> ({cx:.2f},{cy:.2f},{cz:.2f}) "
                          f"(d_hat=({dx:.2f},{dy:.2f},{dz:.2f}))")
                else:
                    print(f"ADRC bias: corrected target infeasible "
                          f"(d_hat=({dx:.2f},{dy:.2f},{dz:.2f})) -- using uncorrected target")

        ok_ik, t1, t2, t3 = solve_ik_mm(x_cmd, y_cmd, z_cmd)
        if not ok_ik:
            print("IK failed")
            return False, None, None, None, None

        ik_deg = (t1, t2, t3)
        if not self.within_joint_limits(t1, t2, t3):
            print(f"Joint limit reject: {ik_deg}")
            return False, ik_deg, None, None, None

        target_radians = [math.radians(theta) for theta in ik_deg]

        for name, rad in zip(self.MOTOR_NAMES, target_radians):
            self.bus.write(name, ParameterType.POSITION_TARGET, float(rad))

        # Poll until the motor settles within tolerance or the timeout expires.
        # VERIFY_DELAY is the maximum wait; we exit early once the FK error is
        # within POS_TOL_MM.  This handles the variation in settle time across
        # speed presets without either timing out too early (max speed) or
        # waiting too long (low speed).
        POLL_INTERVAL = 0.020   # 50 Hz poll rate
        deadline = time.time() + self.VERIFY_DELAY
        fb_deg = None
        fk_xyz = None
        err    = float('inf')

        while True:
            time.sleep(POLL_INTERVAL)

            feedback_radians = [
                float(self.bus.read(name, ParameterType.MECHANICAL_POSITION))
                for name in self.MOTOR_NAMES
            ]
            fb_deg = tuple(math.degrees(v) for v in feedback_radians)

            ok_fk, x_fk, y_fk, z_fk = solve_fk_mm(fb_deg[0], fb_deg[1], fb_deg[2])
            if not ok_fk:
                if time.time() >= deadline:
                    print("FK failed")
                    return False, ik_deg, fb_deg, None, None
                continue

            fk_xyz = (x_fk, y_fk, z_fk)
            err = math.sqrt((x - x_fk) ** 2 + (y - y_fk) ** 2 + (z - z_fk) ** 2)

            if err < self.POS_TOL_MM:
                break                    # settled — exit early

            if time.time() >= deadline:
                break                    # timeout — report whatever error we have

        print(f"FK = ({x_fk:.2f}, {y_fk:.2f}, {z_fk:.2f}) | err={err:.2f} mm")

        if config.ADRC_BIAS_ENABLE:
            self._bias.update(x_fk - x, y_fk - y, z_fk - z)

        return err < self.POS_TOL_MM, ik_deg, fb_deg, fk_xyz, err

    def move_thetas(self, t1: float, t2: float, t3: float, verify_delay: float = None):
        """Command motors to already-solved joint angles (e.g. from delta_common.fk_ik.solve_ik_mm).

        Bypasses the internal IK solver. Joint limits are still enforced.

        Parameters
        ----------
        t1, t2, t3   : degrees
        verify_delay : seconds to wait for settle; defaults to self.VERIFY_DELAY
                       when not given. Callers that need a bigger settle budget
                       for a specific move (e.g. a lift) can override it here
                       without changing the default for every other move.

        Returns
        -------
        (ok, fk_xyz, err, fb_deg)
            ok      : True if FK settled within POS_TOL_MM
            fk_xyz  : (x, y, z) mm from FK at final feedback position
            err     : residual FK error in mm
            fb_deg  : (t1, t2, t3) actual motor angles read from CAN feedback, degrees
        """
        if not self.within_joint_limits(t1, t2, t3):
            print(f"Joint limit reject: ({t1:.1f},{t2:.1f},{t3:.1f})")
            return False, None, float('inf'), None

        ok_ref, x_ref, y_ref, z_ref = solve_fk_mm(t1, t2, t3)
        if not ok_ref:
            print(f"FK failed for commanded thetas ({t1:.1f},{t2:.1f},{t3:.1f})")
            return False, None, float('inf'), None

        print(
            f"\nThetas -> ({t1:.2f}°, {t2:.2f}°, {t3:.2f}°)  "
            f"FK target=({x_ref:.2f},{y_ref:.2f},{z_ref:.2f}) mm"
        )

        target_radians = [math.radians(t) for t in (t1, t2, t3)]
        for name, rad in zip(self.MOTOR_NAMES, target_radians):
            self.bus.write(name, ParameterType.POSITION_TARGET, float(rad))

        POLL_INTERVAL = 0.020
        deadline = time.time() + (verify_delay if verify_delay is not None else self.VERIFY_DELAY)
        fk_xyz = None
        fb_deg = None
        err = float('inf')

        while True:
            time.sleep(POLL_INTERVAL)
            feedback_radians = [
                float(self.bus.read(name, ParameterType.MECHANICAL_POSITION))
                for name in self.MOTOR_NAMES
            ]
            fb_deg = tuple(math.degrees(v) for v in feedback_radians)
            ok_fk, x_fk, y_fk, z_fk = solve_fk_mm(fb_deg[0], fb_deg[1], fb_deg[2])
            if not ok_fk:
                if time.time() >= deadline:
                    return False, None, float('inf'), fb_deg
                continue

            fk_xyz = (x_fk, y_fk, z_fk)
            err = math.sqrt(
                (x_ref - x_fk) ** 2 + (y_ref - y_fk) ** 2 + (z_ref - z_fk) ** 2
            )
            if err < self.POS_TOL_MM:
                break
            if time.time() >= deadline:
                break

        print(f"FK actual=({x_fk:.2f},{y_fk:.2f},{z_fk:.2f}) | err={err:.2f}mm")
        return err < self.POS_TOL_MM, fk_xyz, err, fb_deg

    def get_current_thetas_deg(self):
        """Read the 3 shoulder angles via CAN feedback, in degrees."""
        feedback_radians = [
            float(self.bus.read(name, ParameterType.MECHANICAL_POSITION))
            for name in self.MOTOR_NAMES
        ]
        return tuple(math.degrees(v) for v in feedback_radians)

    def get_current_xyz(self):
        """Read joint positions via CAN feedback and return the FK result.

        Returns
        -------
        (ok, x, y, z) where ok is True if FK succeeded and x/y/z are in mm.
        """
        fb_deg = self.get_current_thetas_deg()
        ok, x, y, z = solve_fk_mm(fb_deg[0], fb_deg[1], fb_deg[2])
        if not ok:
            return False, 0.0, 0.0, 0.0
        return True, x, y, z

    def execute_trajectory(self, waypoints, dt=0.05):
        """Move through a list of (x, y, z) Cartesian waypoints.

        Intermediate waypoints are written directly to the motors (IK solved,
        joints commanded) at intervals of *dt* seconds without waiting for the
        full verification delay.  The **final** waypoint goes through the
        standard ``move_xyz()`` with FK verification and tolerance check.

        Parameters
        ----------
        waypoints : list of (x, y, z) tuples in mm (robot base frame)
        dt        : time step between intermediate waypoints in seconds

        Returns
        -------
        bool : True if the final position is within ``POS_TOL_MM``.

        Notes
        -----
        Waypoints that fail the workspace check or have no IK solution are
        silently skipped (the motors keep tracking the previous setpoint).
        Only the final waypoint failure is returned to the caller.
        """
        if not waypoints:
            return True

        # Intermediate waypoints — IK + write only, no verify delay
        for wp in waypoints[:-1]:
            x, y, z = wp
            if not check_workspace(x, y, z):
                continue
            ok, t1, t2, t3 = solve_ik_mm(x, y, z)
            if not ok or not self.within_joint_limits(t1, t2, t3):
                continue
            radians = [math.radians(t) for t in (t1, t2, t3)]
            for name, rad in zip(self.MOTOR_NAMES, radians):
                self.bus.write(name, ParameterType.POSITION_TARGET, float(rad))
            time.sleep(dt)

        # Final waypoint — full IK + write + FK verification
        ok, *_ = self.move_xyz(*waypoints[-1])
        return ok

    def move(self, x, y, z):
        """Mode-agnostic move: dispatches to move_xyz (PP) or move_xyz_mit (Operation
        Control) based on self.control_mode. Both return the same (ok, ik_deg, fb_deg,
        fk_xyz, err) tuple shape, so callers (e.g. repeatability_test.py) don't need to
        branch on mode themselves."""
        if self.control_mode == "mit":
            return self.move_xyz_mit(x, y, z)
        return self.move_xyz(x, y, z)

    def move_xyz_mit(self, x, y, z):
        """Move to (x, y, z) in Operation Control ("MIT") mode: host generates a
        Cartesian trapezoidal profile (linear_waypoints) and streams a position/Kp/Kd
        setpoint every frame via write_operation_frame(), since run_mode=0 has no
        firmware profiling or motion limits of its own -- unlike move_xyz()'s PP path,
        both are the controller's responsibility here.

        Returns the same (ok, ik_deg, fb_deg, fk_xyz, err) shape as move_xyz(), where
        ik_deg/fk_xyz/err describe the final waypoint.
        """
        if not check_workspace(x, y, z):
            print(f"Workspace reject: XYZ=({x:.1f},{y:.1f},{z:.1f}) outside limits")
            return False, None, None, None, None

        ok0, x0, y0, z0 = self.get_current_xyz()
        if not ok0:
            print("MIT move aborted: could not read current FK position")
            return False, None, None, None, None

        dt = 1.0 / config.MIT_CONTROL_HZ
        waypoints = linear_waypoints(
            (x0, y0, z0), (x, y, z),
            v_max=config.MIT_V_MAX_MMPS, a_max=config.MIT_A_MAX_MMPS2, dt=dt,
        )

        ik_deg = None
        fault_count = 0

        def _stream(wx, wy, wz):
            nonlocal ik_deg
            ok_ik, t1, t2, t3 = solve_ik_mm(wx, wy, wz)
            if not ok_ik or not self.within_joint_limits(t1, t2, t3):
                return False  # hold previous setpoint, same as execute_trajectory()
            ik_deg = (t1, t2, t3)
            for name, deg in zip(self.MOTOR_NAMES, ik_deg):
                self.bus.write_operation_frame(
                    name, math.radians(deg), config.MIT_KP, config.MIT_KD
                )
            return True

        for wx, wy, wz in waypoints:
            _stream(wx, wy, wz)
            time.sleep(dt)

        # Hold the final setpoint while polling FK, same tolerance move_xyz() uses.
        fb_deg = None
        fk_xyz = None
        err = float("inf")
        deadline = time.time() + config.MIT_HOLD_TIME_S

        while True:
            _stream(x, y, z)
            time.sleep(dt)

            feedback_radians = [
                float(self.bus.read(name, ParameterType.MECHANICAL_POSITION))
                for name in self.MOTOR_NAMES
            ]
            fb_deg = tuple(math.degrees(v) for v in feedback_radians)
            ok_fk, x_fk, y_fk, z_fk = solve_fk_mm(fb_deg[0], fb_deg[1], fb_deg[2])
            if not ok_fk:
                if time.time() >= deadline:
                    print("FK failed")
                    return False, ik_deg, fb_deg, None, None
                continue

            fk_xyz = (x_fk, y_fk, z_fk)
            err = math.sqrt((x - x_fk) ** 2 + (y - y_fk) ** 2 + (z - z_fk) ** 2)

            if err > config.MIT_FK_FAULT_MM:
                fault_count += 1
                if fault_count >= config.MIT_FK_FAULT_N:
                    print(f"MIT fault: FK error {err:.1f}mm exceeded "
                          f"{config.MIT_FK_FAULT_MM}mm for {fault_count} cycles -- aborting move")
                    return False, ik_deg, fb_deg, fk_xyz, err
            else:
                fault_count = 0

            if err < self.POS_TOL_MM:
                break
            if time.time() >= deadline:
                break

        print(f"MIT FK = ({fk_xyz[0]:.2f}, {fk_xyz[1]:.2f}, {fk_xyz[2]:.2f}) | err={err:.2f} mm")
        return err < self.POS_TOL_MM, ik_deg, fb_deg, fk_xyz, err

    def shutdown(self):
        if self._shutdown_done:
            return

        self._shutdown_done = True

        if not self.connected:
            return

        try:
            if self.control_mode != "mit":
                # PP/CSP-only return-to-zero; loc_ref is not read in Operation
                # Control mode, so this would be a no-op there -- just disable instead.
                for name in self.MOTOR_NAMES:
                    self.bus.write(name, ParameterType.POSITION_TARGET, 0.0)
                time.sleep(0.5)

            for name in self.MOTOR_NAMES:
                try:
                    self.bus.disable(name)
                except Exception as ex:
                    print(f"disable failed for {name}: {ex}")
        finally:
            try:
                self.bus.disconnect(disable_torque=True)
            except Exception as ex:
                print(f"bus disconnect failed: {ex}")
            self.connected = False

    def stop(self):
        self.shutdown()
