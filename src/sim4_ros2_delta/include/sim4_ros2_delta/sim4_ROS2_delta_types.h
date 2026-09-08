//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// File: sim4_ROS2_delta_types.h
//
// Code generated for Simulink model 'sim4_ROS2_delta'.
//
// Model version                  : 1.367
// Simulink Coder version         : 25.2 (R2025b) 28-Jul-2025
// C/C++ source code generated on : Sat Aug 22 08:20:52 2026
//
// Target selection: ert.tlc
// Embedded hardware selection: ARM Compatible->ARM Cortex-A (64-bit)
// Code generation objectives: Unspecified
// Validation result: Not run
//
#ifndef sim4_ROS2_delta_types_h_
#define sim4_ROS2_delta_types_h_
#include "rtwtypes.h"
#ifndef DEFINED_TYPEDEF_FOR_SL_Bus_builtin_interfaces_Time_
#define DEFINED_TYPEDEF_FOR_SL_Bus_builtin_interfaces_Time_

// MsgType=builtin_interfaces/Time
struct SL_Bus_builtin_interfaces_Time
{
  int32_T sec;
  uint32_T nanosec;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_SL_Bus_ROSVariableLengthArrayInfo_
#define DEFINED_TYPEDEF_FOR_SL_Bus_ROSVariableLengthArrayInfo_

struct SL_Bus_ROSVariableLengthArrayInfo
{
  uint32_T CurrentLength;
  uint32_T ReceivedLength;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_SL_Bus_std_msgs_Header_
#define DEFINED_TYPEDEF_FOR_SL_Bus_std_msgs_Header_

// MsgType=std_msgs/Header
struct SL_Bus_std_msgs_Header
{
  // MsgType=builtin_interfaces/Time
  SL_Bus_builtin_interfaces_Time stamp;

  // PrimitiveROSType=string:IsVarLen=1:VarLenCategory=data:VarLenElem=frame_id_SL_Info:TruncateAction=warn 
  uint8_T frame_id[128];

  // IsVarLen=1:VarLenCategory=length:VarLenElem=frame_id
  SL_Bus_ROSVariableLengthArrayInfo frame_id_SL_Info;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_SL_Bus_custom_messages_DeltaJointAngles_
#define DEFINED_TYPEDEF_FOR_SL_Bus_custom_messages_DeltaJointAngles_

// MsgType=custom_messages/DeltaJointAngles
struct SL_Bus_custom_messages_DeltaJointAngles
{
  // MsgType=std_msgs/Header
  SL_Bus_std_msgs_Header header;
  real_T theta1_deg;
  real_T theta2_deg;
  real_T theta3_deg;
  boolean_T ik_valid;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_SL_Bus_custom_messages_DeltaTarget_
#define DEFINED_TYPEDEF_FOR_SL_Bus_custom_messages_DeltaTarget_

// MsgType=custom_messages/DeltaTarget
struct SL_Bus_custom_messages_DeltaTarget
{
  // MsgType=std_msgs/Header
  SL_Bus_std_msgs_Header header;
  real_T x_mm;
  real_T y_mm;
  real_T z_mm;
  real32_T confidence;
  int32_T track_id;

  // PrimitiveROSType=string:IsVarLen=1:VarLenCategory=data:VarLenElem=detection_mode_SL_Info:TruncateAction=warn 
  uint8_T detection_mode[128];

  // IsVarLen=1:VarLenCategory=length:VarLenElem=detection_mode
  SL_Bus_ROSVariableLengthArrayInfo detection_mode_SL_Info;
};

#endif

#ifndef struct_sJ4ih70VmKcvCeguWN0mNVF
#define struct_sJ4ih70VmKcvCeguWN0mNVF

struct sJ4ih70VmKcvCeguWN0mNVF
{
  real_T sec;
  real_T nsec;
};

#endif                                 // struct_sJ4ih70VmKcvCeguWN0mNVF

#ifndef struct_ros_slros2_internal_block_Pub_T
#define struct_ros_slros2_internal_block_Pub_T

struct ros_slros2_internal_block_Pub_T
{
  boolean_T matlabCodegenIsDeleted;
  int32_T isInitialized;
  boolean_T isSetupComplete;
  boolean_T QOSAvoidROSNamespaceConventions;
};

#endif                                // struct_ros_slros2_internal_block_Pub_T

#ifndef struct_ros_slros2_internal_block_Sub_T
#define struct_ros_slros2_internal_block_Sub_T

struct ros_slros2_internal_block_Sub_T
{
  boolean_T matlabCodegenIsDeleted;
  int32_T isInitialized;
  boolean_T isSetupComplete;
  boolean_T QOSAvoidROSNamespaceConventions;
};

#endif                                // struct_ros_slros2_internal_block_Sub_T

// Forward declaration for rtModel
typedef struct tag_RTM_sim4_ROS2_delta_T RT_MODEL_sim4_ROS2_delta_T;

#endif                                 // sim4_ROS2_delta_types_h_

//
// File trailer for generated code.
//
// [EOF]
//
