# DeltaTarget — camera detection result committed to by the pick-place node.
# Published on /delta/target_committed by pick_place_node (telemetry) and
# used as a convenient test-injection format by kinematics_test_node.
#
# All positions are in the robot base frame (mm).
# Z is EE-tip depth (negative = below base plane).

std_msgs/Header header

float64 x_mm            # target X  (robot base frame, mm)
float64 y_mm            # target Y  (robot base frame, mm)
float64 z_mm            # target Z  (EE-tip, mm — negative below base plane)

float32 confidence      # detection confidence [0.0 – 1.0]; -1 if unavailable
int32   track_id        # camera tracking ID; -1 if unavailable
string  detection_mode  # "orange_blob", "yolo", "fake", etc.
