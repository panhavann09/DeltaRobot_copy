//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// File: sim4_ROS2_delta.h
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
#ifndef sim4_ROS2_delta_h_
#define sim4_ROS2_delta_h_
#include "rtwtypes.h"
#include "slros2_initialize.h"
#include "sim4_ROS2_delta_types.h"

extern "C"
{

#include "rtGetNaN.h"

}

extern "C"
{

#include "rt_nonfinite.h"

}

#include <stddef.h>

// Block signals (default storage)
struct B_sim4_ROS2_delta_T {
  SL_Bus_custom_messages_DeltaTarget In1;// '<S6>/In1'
  SL_Bus_custom_messages_DeltaTarget rtb_SourceBlock_o2_m;
  SL_Bus_custom_messages_DeltaJointAngles BusAssignment1;// '<Root>/Bus Assignment1' 
  char_T b_zeroDelimTopic[27];
  char_T b_zeroDelimTopic_c[25];
  real_T thetas[3];
  real_T c;
  real_T s;
  real_T lp2;
  real_T d;
};

// Block states (default storage) for system '<Root>'
struct DW_sim4_ROS2_delta_T {
  ros_slros2_internal_block_Pub_T obj; // '<S4>/SinkBlock'
  ros_slros2_internal_block_Sub_T obj_l;// '<S5>/SourceBlock'
};

// Real-time Model Data Structure
struct tag_RTM_sim4_ROS2_delta_T {
  const char_T * volatile errorStatus;
  const char_T* getErrorStatus() const;
  void setErrorStatus(const char_T* const volatile aErrorStatus);
};

// Class declaration for model sim4_ROS2_delta
class sim4_ROS2_delta
{
  // public data and function members
 public:
  // Real-Time Model get method
  RT_MODEL_sim4_ROS2_delta_T * getRTM();

  // model initialize function
  void initialize();

  // model step function
  void step();

  // model terminate function
  void terminate();

  // Constructor
  sim4_ROS2_delta();

  // Destructor
  ~sim4_ROS2_delta();

  // private data and function members
 private:
  // Block signals
  B_sim4_ROS2_delta_T sim4_ROS2_delta_B;

  // Block states
  DW_sim4_ROS2_delta_T sim4_ROS2_delta_DW;

  // private member function(s) for subsystem '<Root>'
  void sim4_ROS2__Subscriber_setupImpl(const ros_slros2_internal_block_Sub_T
    *obj);
  void sim4_ROS2_d_Publisher_setupImpl(const ros_slros2_internal_block_Pub_T
    *obj);

  // Real-Time Model
  RT_MODEL_sim4_ROS2_delta_T sim4_ROS2_delta_M;
};

extern volatile boolean_T stopRequested;
extern volatile boolean_T runModel;

//-
//  These blocks were eliminated from the model due to optimizations:
//
//  Block '<Root>/Display13' : Unused code path elimination
//  Block '<Root>/Display7' : Unused code path elimination
//  Block '<Root>/Display8' : Unused code path elimination
//  Block '<Root>/Display9' : Unused code path elimination
//  Block '<Root>/Scope1' : Unused code path elimination
//  Block '<Root>/Gain1' : Eliminated nontunable gain of 1


//-
//  The generated code includes comments that allow you to trace directly
//  back to the appropriate location in the model.  The basic format
//  is <system>/block_name, where system is the system number (uniquely
//  assigned by Simulink) and block_name is the name of the block.
//
//  Use the MATLAB hilite_system command to trace the generated code back
//  to the model.  For example,
//
//  hilite_system('<S3>')    - opens system 3
//  hilite_system('<S3>/Kp') - opens and selects block Kp which resides in S3
//
//  Here is the system hierarchy for this model
//
//  '<Root>' : 'sim4_ROS2_delta'
//  '<S1>'   : 'sim4_ROS2_delta/Blank Message1'
//  '<S2>'   : 'sim4_ROS2_delta/MATLAB Function1'
//  '<S3>'   : 'sim4_ROS2_delta/MATLAB Function4'
//  '<S4>'   : 'sim4_ROS2_delta/Publish1'
//  '<S5>'   : 'sim4_ROS2_delta/Subscribe2'
//  '<S6>'   : 'sim4_ROS2_delta/Subscribe2/Enabled Subsystem'

#endif                                 // sim4_ROS2_delta_h_

//
// File trailer for generated code.
//
// [EOF]
//
