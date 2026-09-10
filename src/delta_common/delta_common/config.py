# CHANGES: [2.2] added AUTO_MOVE, ENABLE_MOTORS, MOVE_COOLDOWN_SEC, MOVE_THRESHOLD_MM
COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"
TRACKED_PLANTS_TOPIC = "/plant_perception/tracked_plants"   # weed_bridge_node input, must match plant_perception's config/perception.yaml
PLANT_PERCEPTION_VISUALIZATION_TOPIC = "/plant_perception/visualization"   # base image for weed_bridge_node's "Delta Camera" window

DETECTION_MODE = "orange_blob"
YOLO_MODEL = "yolo11n.pt"
YOLO_IMGSZ = 1280
YOLO_IOU = 0.45
YOLO_MAX_DET = 20
YOLO_DEVICE = "cuda:0"
CONF_THRES = 0.25
TARGET_CLASS = None
VIEW_IMAGE = True
# 2026-09-08: weed_bridge_node's "Delta Camera" window used to run EE-marker
# laser detection + cv2.imshow/waitKey on every incoming frame (~20+ FPS from
# plant_perception) — measured cost in isolation: imshow/waitKey ~38ms/frame,
# EE-marker detection ~18ms/frame, stacking on top of plant_perception's own
# ~49ms and dragging tracked_plants down to ~10.3 FPS system-wide (CPU
# contention between the two processes' own work, confirmed via tegrastats:
# all 6 cores pegged 75-92%, GPU bursty/idle waiting on CPU). Neither needs
# to run faster than a human can watch a window, and _on_tracked_plants (the
# actual weed-targeting math) never reads _last_ee_uv/_ee_fk_pixel, so both
# are throttled to this rate instead of running inline on every frame.
DISPLAY_MAX_FPS = 10.0
DRAW_CAMERA_AXIS_LEGEND = False
DRAW_BASE_AXIS_OVERLAY = False
DRAW_DETECTION_CONTOUR = False
DRAW_DETECTION_CORNERS = False
DRAW_OBJECT_SIZE_LABEL = False
REQUIRE_FULL_BBOX_IN_FRAME = True
FRAME_MARGIN_PX = 12
DETECT_ONLY_IN_WORKSPACE = True
DRAW_REJECTED_DETECTIONS = False
BBOX_LABEL = "box"
BBOX_FRAME_MIN_AREA = 2500.0
BBOX_FRAME_MAX_AREA_RATIO = 0.70
BBOX_FRAME_MIN_WIDTH_PX = 60
BBOX_FRAME_MIN_HEIGHT_PX = 40
BBOX_NMS_IOU = 0.35
CANNY_LOW = 60
CANNY_HIGH = 180

ORANGE_SQ_HUE_LOW  = 5        # OpenCV H (0-180): start of orange band
ORANGE_SQ_HUE_HIGH = 30       # end of orange band
ORANGE_SQ_SAT_MIN  = 120      # reject washed-out / gray
ORANGE_SQ_VAL_MIN  = 80       # reject near-black shadows
ORANGE_SQ_MIN_SAT_MEAN        = 100.0

# Orange blob detection (simpler — centroid only, no shape check)
ORANGE_BLOB_HUE_LOW   = 5     # OpenCV H (0-180)
ORANGE_BLOB_HUE_HIGH  = 30
ORANGE_BLOB_SAT_MIN   = 100
ORANGE_BLOB_VAL_MIN   = 80
ORANGE_BLOB_MIN_AREA_PX      = 80.0    # 30mm @ Z=-410 → ~230 px², well below
ORANGE_BLOB_MAX_AREA_RATIO   = 0.05
ORANGE_BLOB_MASK_OPEN_PX     = 2
ORANGE_BLOB_MASK_CLOSE_PX    = 5
ORANGE_BLOB_BORDER_REJECT_PX = 4
ORANGE_SQ_MIN_AREA_PX         = 100.0   # 30mm @ Z=-410 → ~230 px², set well below
ORANGE_SQ_MAX_AREA_RATIO      = 0.05
ORANGE_SQ_MIN_WIDTH_PX        = 8       # 30mm → ~15 px wide
ORANGE_SQ_MIN_HEIGHT_PX       = 8
ORANGE_SQ_BORDER_REJECT_PX    = 5       # reduced — small object near edge should not be rejected too aggressively
ORANGE_SQ_MASK_OPEN_PX        = 3
ORANGE_SQ_MASK_CLOSE_PX       = 7
ORANGE_SQ_POLY_EPSILON_SCALE  = 0.05
ORANGE_SQ_MIN_RECTANGULARITY  = 0.76
ORANGE_SQ_MIN_ASPECT_RATIO    = 0.75  # square: allow slight perspective
ORANGE_SQ_MAX_ASPECT_RATIO    = 1.33

PLANE_HOMOGRAPHY_ENABLE = False
PLANE_HOMOGRAPHY_MATRIX = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

RECT_POSE_ENABLE = False
RECT_REAL_WIDTH_MM = 0.0
RECT_REAL_HEIGHT_MM = 0.0

WORKSPACE_ROI_ENABLE = False
WORKSPACE_MARGIN_X_PX = 80
WORKSPACE_MARGIN_Y_PX = 80
ROBOT_EXCLUDE_ENABLE = False
ROBOT_EXCLUDE_POLYGONS_NORM = (
    ((0.02, 0.00), (0.12, 0.00), (0.56, 0.28), (0.50, 0.40), (0.40, 0.34)),
    ((0.88, 0.00), (0.98, 0.00), (0.60, 0.34), (0.50, 0.40), (0.44, 0.28)),
    ((0.42, 0.30), (0.58, 0.30), (0.63, 0.58), (0.58, 0.82), (0.42, 0.82), (0.37, 0.58)),
)
ROBOT_EXCLUDE_MAX_BBOX_OVERLAP = 0.12
ROBOT_EXCLUDE_MAX_CONTOUR_OVERLAP = 0.08
DRAW_ROBOT_EXCLUDE = False

# weed_bridge_node: skip a weed candidate whose bbox overlaps a currently
# tracked crop's bbox above this IoU-style ratio (avoid picking near a crop).
WEED_CROP_OVERLAP_MAX_IOU = 0.10
FOREGROUND_FLOOR_DEPTH_M = 0.870
FOREGROUND_FLOOR_PERCENTILE = 80.0
FOREGROUND_BG_CLOSE_PX = 41
FOREGROUND_MASK_OPEN_PX = 5
FOREGROUND_MASK_CLOSE_PX = 11
FOREGROUND_MIN_HEIGHT_M = 0.008
FOREGROUND_MAX_HEIGHT_M = 0.200
FOREGROUND_MIN_AREA_PX = 3000.0
FOREGROUND_MAX_AREA_RATIO = 0.50
FOREGROUND_MIN_WIDTH_PX = 60
FOREGROUND_MIN_HEIGHT_PX = 40
FOREGROUND_REQUIRE_BOX_LIKE = True
FOREGROUND_BORDER_REJECT_PX = 30
FOREGROUND_MAX_ASPECT_RATIO = 4.0


BOX_ONLY_ENABLE = True
BOX_ROI_SHRINK = 0.20
BOX_MIN_CONTOUR_AREA = 80.0
BOX_POLY_EPSILON_SCALE = 0.04
BOX_MIN_RECTANGULARITY = 0.72
BOX_MAX_ASPECT_RATIO = 3.0

AVG_FRAME_COUNT = 5              # 5 frames: less lag on conveyor, still smooth
CENTER_WINDOW = 5
TRACK_GRID_PX = 80
# Larger grid for conveyor mode: object moves ~2-4 px/frame so 160 px gives
# ~40-80 stable frames per cell before the track ID resets at a boundary.
CONVEYOR_TRACK_GRID_PX = 160
DETECTION_CONFIRM_FRAMES = 2     # 2 frames: faster trigger, still avoids false positives
DETECTION_LOST_RESET_FRAMES = 8  # hold on longer before resetting — avoids flicker

STABLE_THRESH_X_MM = 1.5
STABLE_THRESH_Y_MM = 1.5
STABLE_THRESH_Z_MM = 2.0

DEPTH_MIN_M = 0.001
DEPTH_MAX_M = 1.00
DEPTH_BOX_SCALE = 0.35
DEPTH_MIN_VALID_PIXELS = 20
DEPTH_USE_CONTOUR_MASK = True
DEPTH_CONTOUR_ERODE_PX = 5
DEPTH_TRIM_LOW_PERCENT = 25.0
DEPTH_TRIM_HIGH_PERCENT = 75.0
DEPTH_MAD_SCALE = 2.0
DEPTH_HISTORY_SIZE = 7
DEPTH_MAX_JUMP_M = 0.015

# Home position — θ1=θ2=θ3=0° exactly (arms horizontal). Actual homing goes
# through move_thetas(0,0,0), bypassing IK; this Z is FK(0,0,0) for reference
# only (e.g. pre-position height, logging). solve_fk_mm(0,0,0) = -228.644mm
# (recomputed for the corrected DeltaGeometry kinematics; was -323.531mm under
# the old e/f/re/rf model).
HOME_X =   0.0
HOME_Y =   0.0
HOME_Z = -228.644

# Drop near belt exit — pulled in from the (-150,150) edge point (only 4.1deg
# joint-limit margin, theta2=85.9deg there at Z=-390) toward workspace centre
# for a safer 15.3deg margin (theta2=74.7deg). Verify this still lands in the
# collection bin/chute before running unattended — it's ~42mm inward of the
# old edge-of-belt drop point.
PLACE_X = -120.2
PLACE_Y =  120.2

PLACE_Z = -330.0   # wrist Z for place — was -483.5 (theta2=110.4deg, unreachable,
                    # aborted every place). -390 is reachable with margin at PLACE_X/Y above.
PICK_Z  = -450   # wrist Z for pick   (EE tip = PICK_Z  - 150 = -633.5mm, 30mm above belt@-663.5)
# Tip target -633.5mm is only 4.5mm above EETIP_Z_FLOOR_MM(-638.0, confirmed crash) — verify
# with a slow manual descent before running full-speed automated cycles.

# Acceleration used for triangular travel-time prediction (medium preset).
# Within ±150mm workspace all moves are triangular: t = 2*sqrt(dist / TRAJ_A_MAX_MM_S2)
TRAJ_A_MAX_MM_S2 = 5000.0

# Extra wait at approach Z (mm above object) after robot pre-positions.
# Gives the object time to arrive if Y prediction is still slightly off.
# Increase in small steps (0.1 s) if robot still arrives before object.
CONVEYOR_APPROACH_WAIT_SEC = 0.3   # short motor-settle; arrival now handled dynamically
CONVEYOR_ARRIVAL_X_THRESH_MM = 15.0  # pick when |err_x| < this (object within 15mm in X)
CONVEYOR_ARRIVAL_TIMEOUT_S   = 5.0   # give up and pick anyway after 5s

# Static EE correction offsets (mm).  Measure the real error at your target
# position and set each offset to negate it: if the EE lands 10 mm too far in
# +X, set EE_OFFSET_X_MM = -10.0.
EE_OFFSET_X_MM = 0.0
EE_OFFSET_Y_MM = 0.0
EE_OFFSET_Z_MM   = 150.0   # gripper length below platform (mm); measured gripper length

FAKE_DEPTH_ENABLE = True   # always use FAKE_DEPTH_M — no depth stream on the global-shutter camera
FAKE_DEPTH_M = 0.58  # camera-to-belt/ground distance (m), measured directly 2026-09-02

ERROR_MAP_ENABLE = False

MOTOR_VEL_MAX    = 40.0  # PP mode max velocity — increase for faster moves (was 3.0; motor spec max 50)
MOTOR_ACC_SET    = 40.0  # PP mode acceleration — increase for snappier starts (was 5.0)
FK_VERIFY_TOL_MM = 3.0
PRINT_COOLDOWN_SEC = 1.0
FAKE_MOVE_COOLDOWN_SEC = 1.0
AUTO_MOVE = True
ENABLE_MOTORS = True
MOVE_COOLDOWN_SEC = 1.5
MOVE_THRESHOLD_MM = 4.0

# Measured 2026-07-01 at true θ=0,0,0 home: base_to_plate=320mm (FK predicts 323.5mm,
# model validated), plate_to_tip=150mm (fixed), tip_to_belt=190mm.
# home tip Z (FK) = -473.531mm
# z_base = home_tip_Z - tip_to_belt(190) = -663.531mm = belt surface (tip frame)
# z_platform = z_base + EE_OFFSET_Z_MM(150) = -513.531mm =  
#  surface (wrist/platform frame)

# Fixed fake object position for testing EE correction
# Set FAKE_OBJ_ENABLE = True to use fixed position
# instead of real camera detection


VISION_ONLY_ENABLE = False
SIMPLE_RESULT_PRINT = True
PURE_CAMERA_TEST_ENABLE = False

# Conveyor belt mode: skip stability check (object is always moving).
# Camera publishes a target once track is confirmed + averaged over AVG_FRAME_COUNT frames.
CONVEYOR_MODE = True

# ── Conveyor belt hardware parameters ────────────────────────────────────────
# Stepper motor: MIN_DELAY=70µs, STEPS_PER_REV=1600, GEAR_RATIO=36, PULLEY_DIA=49mm
#   Motor RPM  = 60_000_000 / (2 × 70 × 1600) = 267.86 RPM
#   Output RPM = 267.86 / 36                   = 7.44  RPM
#   Belt mm/s  = (7.44 × π × 49) / 60         = 19.09 mm/s  ← old design-math estimate
#   Measured physical speed: 26.0 mm/s (2026-07-09 remeasurement; exceeds the
#   stepper-math estimate above — re-derive pulley/gearing if that gap matters)
# 2026-08-17: swapped to the field tractor rig — 0.5 m/s ground speed.
CONVEYOR_BELT_SPEED_MM_S = 500.0   # mm/s — 0.5 m/s tractor ground speed

# Velocity sanity bounds: camera estimate is rejected if it falls outside this range.
# Lower bound covers belt not yet at speed; upper bound catches outlier regression.
CONVEYOR_VX_MIN_MM_S =  1.0    # below this → belt probably stopped, use 0
CONVEYOR_VX_MAX_MM_S = 600.0   # above this → regression outlier, clamp to design speed

# False: pick_place_node.py grips the raw detected/depth-smoothed object
# pose directly — no belt-velocity lead compensation, no travel/descend timing
# offset. For bench-testing pure Cartesian pick accuracy without conveyor
# motion in the loop. True restores the BeltPredictor lead correction.
BELT_PREDICTION_ENABLE = False

# Minimum seconds between consecutive target publishes to avoid flooding the robot.
TARGET_PUBLISH_COOLDOWN_SEC = 1.0

# Camera-frame workspace zone overlay.
# Projects the robot reachable square (±X_LIMIT, ±Y_LIMIT) at WORKSPACE_PICK_Z_MM
# and draws three coloured zones: approach (amber), workspace (green), exit (red).
DRAW_WORKSPACE_ZONES = True
# platform Z at pick height (EE tip 150mm below = belt). Re-derived for the
# corrected DeltaGeometry kinematics by applying the same shift as HOME_Z
# (new HOME_Z -228.644 vs old-model HOME_Z -323.531 = +94.887mm): the old
# value (-513.5) was computed against the old (wrong) FK and is now
# unreachable at all 4 box corners. Like X_LIMIT/Y_LIMIT, this square is a
# rectangular over-approximation — corners are drawn for reference even
# where IK doesn't quite hold at this exact Z.
WORKSPACE_PICK_Z_MM  = -418.6

# ADRC-flavored persistent bias corrector for Cartesian tracking error (PP
# mode scoped -- only the commanded position reference can be biased, no
# torque/gain access). Learns a per-axis additive bias from move_xyz()'s
# existing FK feedback; no new sensing. In-memory only, resets on restart.
# Default OFF -- opt-in, zero behavior change until explicitly enabled.
ADRC_BIAS_ENABLE     = False   # on for repeatability_test.py bias validation (static/moving belt)
ADRC_BIAS_BETA       = 0.25   # leaky-integrator gain; ~8 cycles to ~90% of true bias
ADRC_BIAS_MAX_MM     = 8.0    # clamp on |d_hat| per axis
ADRC_BIAS_MIN_ERR_MM = 1.0    # deadband (2x POS_TOL_MM); ignore residuals below this

# Operation Control ("MIT") mode -- run_mode=0, host streams pset/vset/Kp/Kd/tau_ff
# every frame (see thesis Table 4.2). Used by DeltaMotorController.move_xyz_mit() /
# move() dispatch for the PP-vs-MIT repeatability comparison. Kp/Kd seeded from the
# rob_and_ros_pkg prototype -- MUST be re-verified on hardware (small jog, watch for
# oscillation/overshoot) before running a full repeatability sweep at these gains.
MIT_KP           = 15.0    # N*m/rad, per-joint proportional gain
MIT_KD           = 5.0     # N*m/(rad/s), per-joint derivative gain
MIT_CONTROL_HZ   = 100.0   # setpoint stream rate; thesis notes torque modes want ~1kHz --
                            # treat the achievable rate on this ROS2/USB-CAN stack as a finding
MIT_V_MAX_MMPS   = 214.0   # Cartesian transport speed, matches PP run's paper-referenced speed
MIT_A_MAX_MMPS2  = 800.0   # Cartesian accel for linear_waypoints(); placeholder, tune during bring-up
MIT_HOLD_TIME_S  = 0.3     # extra time streaming the final setpoint before the move ends
MIT_FK_FAULT_MM  = 15.0    # host-side fault threshold (PP mode gets this from firmware for free)
MIT_FK_FAULT_N   = 5       # consecutive over-threshold cycles before aborting the move

# EE marker detection (white laser dot on end-effector tip)
EE_CORRECTION_ENABLE    = True
EE_CORRECTION_MAX_MM    = 30.0
EE_CORRECTION_ALPHA     = 0.5
EE_CORRECTION_GAIN      = 0.5    # halves X error each iter: 13→6.5→3.25→1.6mm
EE_CORRECTION_THRESH_MM = 3.0    # descend only when X ≤ 3mm from centroid
EE_CORRECTION_MIN_MM    = 2.0
EE_CORRECTION_MAX_ITERS = 6      # up to 6 iters; ~3 needed for 13mm→<2mm
EE_CORRECTION_TIMEOUT_S = 2.5
EE_CORRECTION_WAIT_S    = 0.05
EE_LASER_HUE_LOW1  = 0      # red lower range
EE_LASER_HUE_HIGH1 = 10
EE_LASER_HUE_LOW2  = 130    # magenta/pink: 650nm laser on D455 appears H≈150-165
EE_LASER_HUE_HIGH2 = 180
EE_LASER_SAT_MIN        = 10     # laser pixels have S=14-48 (near-white pink core)
EE_LASER_VAL_MIN        = 230    # only near-saturated pixels: cuts ambient surfaces
EE_LASER_CORE_VAL_MIN   = 180    # overexposed white core: any hue, very bright
EE_LASER_CORE_SAT_MAX   = 80     # white core has near-zero saturation
EE_LASER_MIN_AREA  = 1
EE_LASER_MAX_AREA  = 100
EE_LASER_MAX_JUMP_PX   = 999   # disabled — ROI search handles rejection instead
EE_LASER_SMOOTH_FRAMES = 5     # 5-frame median
EE_LASER_ROI_PX        = 35    # search radius around last position (px)
DRAW_EE_MARKER         = True
# Shift the overlay box in the camera stream without affecting 3D detection.
# Positive X_OFFSET_MM moves box toward the approach side (belt entry direction).
# Positive Y_OFFSET_MM moves box right in robot base frame (→ left in image).
WORKSPACE_OVERLAY_X_OFFSET_MM = 0.0
WORKSPACE_OVERLAY_Y_OFFSET_MM = 0.0

CAMERA_TRANSFORM_MODE = "A"
CAMERA_USE_DIRECT_MATRIX = True

# Rotation: x_base = y_cam,  y_base = x_cam,  z_base = -z_cam
# Measured mount: Y = 300 mm in front of base, Z = 80 mm above base origin
# Working distance to home (z=-350): 80 + 350 = 430 mm
CAMERA_DIRECT_MATRIX = (
    ( 0.0,  1.0,  0.0),
    ( 1.0,  0.0,  0.0),
    ( 0.0,  0.0, -1.0),
)
CAM_FINE_ROLL_DEG  = 0.0
CAM_FINE_PITCH_DEG = 0.0
CAM_FINE_YAW_DEG   = 0.0
CAM_TX_MM =220.0   # calibrated via fixed EE laser at true center (30-sample avg, reproduced across 2 runs)
CAM_TY_MM =0.0    # calibrated via fixed EE laser at true center (30-sample avg, reproduced across 2 runs)
CAM_TZ_MM =0   # camera Z == base-frame origin Z (measured directly 2026-09-02, no offset)

# Full 4×4 homogeneous T_cam_to_base  (p_base = T @ [p_cam; 1])
# Built from the R and t above — use camera_system._build_T_cam_to_base() at runtime
CAMERA_T_BASE = (
    ( 0.0,  1.0,  0.0,  CAM_TX_MM),
    ( 1.0,  0.0,  0.0, CAM_TY_MM),
    ( 0.0,  0.0, -1.0,  CAM_TZ_MM),
    ( 0.0,  0.0,  0.0,    1.0),
)

# Rectangular pre-filter — real per-point gating is the solve_ik_mm() call in
# check_workspace, so this box only needs to be a loose over-approximation.
# 2026-08-19: at WORKSPACE_PICK_Z_MM the true IK-feasible region is a lobed
# shape (3 arms at 180/300/60 deg), not a square — max reach measured by
# azimuth: ~207mm toward each arm (az 0/120/240), ~218mm at az 90/270,
# ~252mm between arms (az 60/180/300). The old 200mm box under-used reach in
# every direction (e.g. az=180 real max ~252mm) while its corners (283mm out)
# were always past 90 deg on some arm and rejected downstream anyway. Raised
# to 255mm — covers the full lobe with small margin; check_workspace's IK
# call still correctly rejects the still-unreachable corner regions.
X_LIMIT = 200.0
Y_LIMIT = 200.0
Z_MIN = -650.0   # NOTE: not re-derived for the new DeltaGeometry — solve_ik_mm(0,0,z)
                 # already fails by z=-600 at centre, so this floor is now looser than
                 # actually reachable (safe: the IK feasibility check in check_workspace
                 # still rejects unreachable points, just later than an exact bound would).
# IK ceiling at centre (theta=0) under the corrected DeltaGeometry is
# solve_fk_mm(0,0,0) = -228.644mm (see HOME_Z). Z_MAX kept ~0.6mm shallower than
# that, mirroring the old model's margin, so the box pre-filter stays permissive
# and the real feasibility check in check_workspace does the rejecting.
Z_MAX = -228.0

# Approach-zone band for pick_place_node.py's WAITING pre-position: a weed
# still outside the workspace (validate_target reason="OUTSIDE_WORKSPACE")
# counts as "approaching" once its X is within this margin of X_LIMIT on the
# entry side (belt moves in -X — see belt_predictor.py — so weeds arrive
# from +X). Weeds further out than this are ignored to avoid pre-positioning
# for something still far up the belt that may never arrive.
APPROACH_ZONE_MARGIN_MM = 80.0

# Pre-position pose for the WAITING state: once a weed is approaching (see
# APPROACH_ZONE_MARGIN_MM above) but still outside the workspace, the arm
# parks here instead of sitting at HOME — closer to the entry edge, so the
# final travel once the weed actually arrives is short.
# WAIT_Z_MM is NOT HOME_Z: HOME_Z (-228.644) is the shallow ceiling height,
# only reachable near X=0 — at WAIT_X_MM (near the +X edge) it's outside
# joint limits (confirmed: solve_ik_mm(210,0,-228.6) gives theta2/3=-16deg,
# below THETA_MIN=-5). The workspace's true reachable region is a lobed
# shape, not the rectangular X/Y/Z_LIMIT box (see the box's own comments
# above) — swept solve_ik_mm at (WAIT_X_MM, WAIT_Y_MM) to find the in-limits
# Z band (roughly -280 to -400mm there) and picked -330 for margin
# (theta1=78.8deg, ~11deg below the 90deg limit).
WAIT_X_MM = X_LIMIT - 10.0   # just inside the reachable boundary, entry side (+X)
WAIT_Y_MM = 0.0
WAIT_Z_MM = -330.0   # platform-frame, like HOME_Z/PLACE_Z/PICK_Z

THETA1_MIN = -5
THETA1_MAX = 90   # verified on hardware 2026-07-21: 90 deg is safe
THETA2_MIN = -5
THETA2_MAX = 90.0
THETA3_MIN = -5
THETA3_MAX = 90.0
