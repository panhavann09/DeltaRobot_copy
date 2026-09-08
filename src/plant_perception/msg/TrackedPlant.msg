std_msgs/Header header
int32 track_id
int32 class_id
string class_name
float32 confidence
float32 x1
float32 y1
float32 x2
float32 y2
float32 root_x
float32 root_y
float32 root_d
sensor_msgs/Image mask

uint8 TENTATIVE=0
uint8 CONFIRMED=1
uint8 COASTING=2

uint8 track_state
bool is_confirmed
bool is_currently_detected
bool actionable

uint32 age_frames
uint32 hit_count
uint32 consecutive_hits
uint32 missed_frames

bool depth_valid
bool current_depth_valid
float32 depth_valid_ratio
float32 depth_std_m
