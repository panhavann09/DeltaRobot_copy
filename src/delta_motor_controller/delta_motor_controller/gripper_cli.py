#!/usr/bin/env python3
"""gripper_cli.py — simple open/close test for the pneumatic gripper (CAN ID 600).

No ROS involved — direct CAN via PneumaticGripper, same code path the main
pick-and-place pipeline uses.

Usage:
    python3 gripper_cli.py open   [can_channel]
    python3 gripper_cli.py close  [can_channel]

can_channel defaults to 'can1' (matches pick_place_node.py).
Use 'can0' if you're testing against the blind_pick_place / can_driver_node setup.
"""

import sys

try:
    from delta_motor_controller.pneumatic_gripper import PneumaticGripper
except ImportError:
    # Running directly (python3 gripper_cli.py) from inside this source dir —
    # the package isn't on sys.path, but the sibling module is.
    from pneumatic_gripper import PneumaticGripper


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("open", "close"):
        print(__doc__)
        sys.exit(1)

    action = sys.argv[1]
    can_channel = "can1"

    gripper = PneumaticGripper(can_channel=can_channel, can_id=4)
    print(f"Connecting on {can_channel} (CAN ID 4)...")
    gripper.connect()   # connect() itself sends an initial release

    try:
        if action == "open":
            gripper.release()
            print("Gripper OPEN (released).")
        else:
            gripper.grip()
            print("Gripper CLOSE (gripped).")
    finally:
        gripper.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
