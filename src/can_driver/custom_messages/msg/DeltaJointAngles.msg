# DeltaJointAngles — IK solution / motor-encoder feedback, used by
# pick_place_node (published as telemetry on /delta/motor_thetas after every
# move) and kinematics_test_node.
#
# ik_valid: True = valid IK solution; False = solver could not find one.

std_msgs/Header header

float64 theta1_deg      # motor 1 joint angle (degrees, must be ≥ 0)
float64 theta2_deg      # motor 2 joint angle (degrees, must be ≥ 0)
float64 theta3_deg      # motor 3 joint angle (degrees, must be ≥ 0)

bool    ik_valid        # True = valid IK solution; False = solver could not solve
