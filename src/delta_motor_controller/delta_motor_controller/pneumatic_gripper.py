#!/usr/bin/env python3
"""
pneumatic_gripper.py — Pneumatic solenoid gripper for delta pick-and-place.

Sends CAN frames directly via python-can (socketcan) — no can_driver_node needed.

CAN frame format — matches can_driver.py's digital_and_solenoid_command_callback
exactly (the original, authoritative driver for this board):
    arbitration_id = can_id (4)
    data[0] = 0x40
    data[1] = digital bitmask (unused here, always 0)
    data[2] = solenoid bitmask, bit i = solenoid(i+1)_value, direct (no inversion,
              unused solenoid bits left at 0 — can_driver.py never forces them high)
    is_extended_id = False

Hardware convention (verified directly on this rig 2026-07-21 via gripper_cli.py —
supersedes the 2026-07-20 testing_solenoid.py-derived mapping below, which no
longer matches this board/wiring):
    solenoid1_value=False (bit0=0, all other bits 0)  →  OPEN  (RELEASE)
    solenoid1_value=True  (bit0=1, all other bits 0)  →  CLOSE (GRIP)
(Prior reference, now stale: testing_solenoid.py logged the opposite polarity
on 2026-07-20. Earlier revisions before that guessed an "active-low, force
bits 2-5 high" convention that was never verified — polarity has flip-flopped
more than once on this project, so trust the most recent hardware test.)
"""

import threading
import time
import can

# testing_solenoid.py (the proven-working reference for this board) never sends
# a one-shot command — it re-publishes the desired state every 1s via a ROS
# timer, continuously, for the node's whole lifetime.  A single frame from
# grip()/release() alone was observed settling back to a gripped state shortly
# after being sent, which points to the board having a command watchdog that
# reverts outputs if it stops hearing from us.  This period is comfortably
# under the 1s that was proven to work.
HEARTBEAT_PERIOD_S = 0.3


class PneumaticGripper:
    """
    Gripper controller for delta robot pick-and-place.
    Sends CAN frames directly — no ROS topic or can_driver_node required.

    Continuously re-sends the last commanded solenoid state in the background
    (see HEARTBEAT_PERIOD_S) — the board appears to need a refreshed signal to
    hold a state, not just a single frame.

    Parameters
    ----------
    can_channel   : str   — SocketCAN channel (default 'can0')
    can_id        : int   — CAN ID of solenoid board (default 4)
    grip_settle_s : float — wait after grip command   (default 0.5 s)
    open_settle_s : float — wait after release command (default 0.3 s)
    """

    def __init__(self, can_channel: str = 'can1', can_id: int = 4,
                 grip_settle_s: float = 0.5,
                 open_settle_s: float = 0.3):
        self._can_channel = can_channel
        self._can_id = can_id
        self._grip_settle_s = grip_settle_s
        self._open_settle_s = open_settle_s
        self._bus = None
        self._last_solenoid1 = False
        self._last_solenoid2 = False
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = None

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open CAN bus, send initial release command (safe state), and start
        the heartbeat thread that keeps re-sending the current state."""
        self._bus = can.interface.Bus(
            interface='socketcan',
            channel=self._can_channel,
            bitrate=1000000,
        )
        self.release(wait=False)
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

    def disconnect(self) -> None:
        """Stop the heartbeat, release gripper, and close CAN bus."""
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)
            self._heartbeat_thread = None
        if self._bus is not None:
            self.release(wait=False)
            self._bus.shutdown()
            self._bus = None

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(HEARTBEAT_PERIOD_S):
            self._send(solenoid1=self._last_solenoid1, solenoid2=self._last_solenoid2)

    # ── control ────────────────────────────────────────────────────────────────

    def grip(self, wait: bool = True) -> None:
        """Grip = CLOSE = solenoid1_value=True (verified on hardware 2026-07-21)."""
        self._send(solenoid1=True)
        if wait:
            time.sleep(self._grip_settle_s)

    def release(self, wait: bool = True) -> None:
        """Release = OPEN = solenoid1_value=False (verified on hardware 2026-07-21)."""
        self._send(solenoid1=False)
        if wait:
            time.sleep(self._open_settle_s)

    # ── backward-compatible aliases ────────────────────────────────────────────

    def close(self, wait: bool = True) -> None:
        self.grip(wait=wait)

    def open(self, wait: bool = True) -> None:
        self.release(wait=wait)

    # ── internal ───────────────────────────────────────────────────────────────

    def _send(self, solenoid1: bool = False, solenoid2: bool = False) -> None:
        # Direct bit assignment, matching can_driver.py's
        # digital_and_solenoid_command_callback exactly — no inversion, unused
        # solenoid bits (2-6) left at 0.
        self._last_solenoid1 = solenoid1
        self._last_solenoid2 = solenoid2
        if self._bus is None:
            print('PneumaticGripper: send called before connect()')
            return
        data = [0] * 8
        data[0] = 0x40
        data[1] = 0
        data[2] = (int(solenoid1)       |   # bit 0
                   int(solenoid2) << 1)      # bit 1
        msg = can.Message(
            arbitration_id=self._can_id,
            data=data,
            is_extended_id=False,
        )
        try:
            self._bus.send(msg)
        except can.CanError as e:
            print(f'PneumaticGripper: CAN send failed: {e}')
