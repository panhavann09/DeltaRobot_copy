//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// File: sim4_ROS2_delta.cpp
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
#include "sim4_ROS2_delta.h"
#include "sim4_ROS2_delta_types.h"
#include <string.h>
#include <math.h>
#include "sim4_ROS2_delta_private.h"
#include "rtwtypes.h"
#include "rmw/qos_profiles.h"
#include <stddef.h>

extern "C"
{

#include "rt_nonfinite.h"

}

#include "rt_defines.h"

real_T rt_hypotd_snf(real_T u0, real_T u1)
{
  real_T a;
  real_T b;
  real_T y;
  a = fabs(u0);
  b = fabs(u1);
  if (a < b) {
    a /= b;
    y = sqrt(a * a + 1.0) * b;
  } else if (a > b) {
    b /= a;
    y = sqrt(b * b + 1.0) * a;
  } else if (rtIsNaN(b)) {
    y = (rtNaN);
  } else {
    y = a * 1.4142135623730951;
  }

  return y;
}

real_T rt_atan2d_snf(real_T u0, real_T u1)
{
  real_T y;
  if (rtIsNaN(u0) || rtIsNaN(u1)) {
    y = (rtNaN);
  } else if (rtIsInf(u0) && rtIsInf(u1)) {
    int32_T tmp;
    int32_T tmp_0;
    if (u0 > 0.0) {
      tmp = 1;
    } else {
      tmp = -1;
    }

    if (u1 > 0.0) {
      tmp_0 = 1;
    } else {
      tmp_0 = -1;
    }

    y = atan2(static_cast<real_T>(tmp), static_cast<real_T>(tmp_0));
  } else if (u1 == 0.0) {
    if (u0 > 0.0) {
      y = RT_PI / 2.0;
    } else if (u0 < 0.0) {
      y = -(RT_PI / 2.0);
    } else {
      y = 0.0;
    }
  } else {
    y = atan2(u0, u1);
  }

  return y;
}

void sim4_ROS2_delta::sim4_ROS2__Subscriber_setupImpl(const
  ros_slros2_internal_block_Sub_T *obj)
{
  rmw_qos_profile_t qos_profile;
  sJ4ih70VmKcvCeguWN0mNVF deadline;
  sJ4ih70VmKcvCeguWN0mNVF lifespan;
  sJ4ih70VmKcvCeguWN0mNVF liveliness_lease_duration;
  static const char_T b_zeroDelimTopic[25] = "/delta/matlab/target_xyz";
  qos_profile = rmw_qos_profile_default;

  // Start for MATLABSystem: '<S5>/SourceBlock'
  deadline.sec = 0.0;
  deadline.nsec = 0.0;
  lifespan.sec = 0.0;
  lifespan.nsec = 0.0;
  liveliness_lease_duration.sec = 0.0;
  liveliness_lease_duration.nsec = 0.0;
  SET_QOS_VALUES(qos_profile, RMW_QOS_POLICY_HISTORY_KEEP_LAST, (size_t)10.0,
                 RMW_QOS_POLICY_DURABILITY_VOLATILE,
                 RMW_QOS_POLICY_RELIABILITY_RELIABLE, deadline, lifespan,
                 RMW_QOS_POLICY_LIVELINESS_AUTOMATIC, liveliness_lease_duration,
                 (bool)obj->QOSAvoidROSNamespaceConventions);
  for (int32_T i = 0; i < 25; i++) {
    // Start for MATLABSystem: '<S5>/SourceBlock'
    sim4_ROS2_delta_B.b_zeroDelimTopic_c[i] = b_zeroDelimTopic[i];
  }

  Sub_sim4_ROS2_delta_75.createSubscriber(&sim4_ROS2_delta_B.b_zeroDelimTopic_c
    [0], qos_profile);
}

void sim4_ROS2_delta::sim4_ROS2_d_Publisher_setupImpl(const
  ros_slros2_internal_block_Pub_T *obj)
{
  rmw_qos_profile_t qos_profile;
  sJ4ih70VmKcvCeguWN0mNVF deadline;
  sJ4ih70VmKcvCeguWN0mNVF lifespan;
  sJ4ih70VmKcvCeguWN0mNVF liveliness_lease_duration;
  static const char_T b_zeroDelimTopic[27] = "/delta/matlab/joint_thetas";
  qos_profile = rmw_qos_profile_default;

  // Start for MATLABSystem: '<S4>/SinkBlock'
  deadline.sec = 0.0;
  deadline.nsec = 0.0;
  lifespan.sec = 0.0;
  lifespan.nsec = 0.0;
  liveliness_lease_duration.sec = 0.0;
  liveliness_lease_duration.nsec = 0.0;
  SET_QOS_VALUES(qos_profile, RMW_QOS_POLICY_HISTORY_KEEP_LAST, (size_t)1.0,
                 RMW_QOS_POLICY_DURABILITY_VOLATILE,
                 RMW_QOS_POLICY_RELIABILITY_RELIABLE, deadline, lifespan,
                 RMW_QOS_POLICY_LIVELINESS_AUTOMATIC, liveliness_lease_duration,
                 (bool)obj->QOSAvoidROSNamespaceConventions);
  for (int32_T i = 0; i < 27; i++) {
    // Start for MATLABSystem: '<S4>/SinkBlock'
    sim4_ROS2_delta_B.b_zeroDelimTopic[i] = b_zeroDelimTopic[i];
  }

  Pub_sim4_ROS2_delta_72.createPublisher(&sim4_ROS2_delta_B.b_zeroDelimTopic[0],
    qos_profile);
}

// Model step function
void sim4_ROS2_delta::step()
{
  real_T e1_idx_0_tmp;
  real_T mx;
  real_T rtb_Sum3_idx_0;
  real_T rtb_Sum3_idx_1;
  real_T rtb_Sum3_idx_2;
  int32_T i;
  boolean_T b_varargout_1;
  static const int16_T f[3] = { 180, 300, 60 };

  int32_T exitg1;
  boolean_T exitg2;

  // BusAssignment: '<Root>/Bus Assignment1'
  memset(&sim4_ROS2_delta_B.BusAssignment1, 0, sizeof
         (SL_Bus_custom_messages_DeltaJointAngles));

  // MATLABSystem: '<S5>/SourceBlock'
  b_varargout_1 = Sub_sim4_ROS2_delta_75.getLatestMessage
    (&sim4_ROS2_delta_B.rtb_SourceBlock_o2_m);

  // Outputs for Enabled SubSystem: '<S5>/Enabled Subsystem' incorporates:
  //   EnablePort: '<S6>/Enable'

  // Start for MATLABSystem: '<S5>/SourceBlock'
  if (b_varargout_1) {
    // SignalConversion generated from: '<S6>/In1'
    sim4_ROS2_delta_B.In1 = sim4_ROS2_delta_B.rtb_SourceBlock_o2_m;
  }

  // End of Start for MATLABSystem: '<S5>/SourceBlock'
  // End of Outputs for SubSystem: '<S5>/Enabled Subsystem'

  // MATLAB Function: '<Root>/MATLAB Function1' incorporates:
  //   SignalConversion generated from: '<Root>/Bus Selector2'

  if (sim4_ROS2_delta_B.In1.x_mm >= 0.0) {
    // Sum: '<Root>/Sum3'
    rtb_Sum3_idx_0 = sim4_ROS2_delta_B.In1.x_mm * 1.8;
  } else {
    // Sum: '<Root>/Sum3'
    rtb_Sum3_idx_0 = sim4_ROS2_delta_B.In1.x_mm * 0.5;
  }

  // End of MATLAB Function: '<Root>/MATLAB Function1'

  // Sum: '<Root>/Sum3' incorporates:
  //   SignalConversion generated from: '<Root>/Bus Selector2'
  //
  rtb_Sum3_idx_1 = sim4_ROS2_delta_B.In1.y_mm;
  rtb_Sum3_idx_2 = sim4_ROS2_delta_B.In1.z_mm;

  // BusAssignment: '<Root>/Bus Assignment1' incorporates:
  //   MATLAB Function: '<Root>/MATLAB Function4'

  sim4_ROS2_delta_B.BusAssignment1.theta1_deg = 0.0;
  sim4_ROS2_delta_B.BusAssignment1.theta2_deg = 0.0;
  sim4_ROS2_delta_B.BusAssignment1.theta3_deg = 0.0;
  sim4_ROS2_delta_B.BusAssignment1.ik_valid = false;

  // MATLAB Function: '<Root>/MATLAB Function4' incorporates:
  //   SignalConversion generated from: '<Root>/Bus Selector2'
  //
  if ((!(fabs(rtb_Sum3_idx_0) > 220.0)) && (!(fabs(sim4_ROS2_delta_B.In1.y_mm) >
        220.0)) && ((!(sim4_ROS2_delta_B.In1.z_mm < -700.0)) &&
                    (!(sim4_ROS2_delta_B.In1.z_mm > -228.0)))) {
    sim4_ROS2_delta_B.thetas[0] = 0.0;
    sim4_ROS2_delta_B.thetas[1] = 0.0;
    sim4_ROS2_delta_B.thetas[2] = 0.0;
    i = 0;
    do {
      exitg1 = 0;
      if (i < 3) {
        sim4_ROS2_delta_B.s = 0.017453292519943295 * static_cast<real_T>(f[i]);
        sim4_ROS2_delta_B.c = cos(sim4_ROS2_delta_B.s);
        sim4_ROS2_delta_B.s = sin(sim4_ROS2_delta_B.s);
        sim4_ROS2_delta_B.lp2 = -rtb_Sum3_idx_0 * sim4_ROS2_delta_B.s +
          rtb_Sum3_idx_1 * sim4_ROS2_delta_B.c;
        sim4_ROS2_delta_B.lp2 = 160000.0 - sim4_ROS2_delta_B.lp2 *
          sim4_ROS2_delta_B.lp2;
        if (sim4_ROS2_delta_B.lp2 < 0.0) {
          exitg1 = 1;
        } else {
          sim4_ROS2_delta_B.lp2 = sqrt(sim4_ROS2_delta_B.lp2);
          sim4_ROS2_delta_B.s = ((rtb_Sum3_idx_0 * sim4_ROS2_delta_B.c +
            rtb_Sum3_idx_1 * sim4_ROS2_delta_B.s) + 35.0) - 163.21;
          sim4_ROS2_delta_B.d = rt_hypotd_snf(sim4_ROS2_delta_B.s,
            rtb_Sum3_idx_2);
          if (sim4_ROS2_delta_B.d > sim4_ROS2_delta_B.lp2 + 200.0) {
            exitg1 = 1;
          } else {
            sim4_ROS2_delta_B.lp2 = ((40000.0 - sim4_ROS2_delta_B.lp2 *
              sim4_ROS2_delta_B.lp2) + sim4_ROS2_delta_B.d * sim4_ROS2_delta_B.d)
              / (2.0 * sim4_ROS2_delta_B.d);
            sim4_ROS2_delta_B.c = 40000.0 - sim4_ROS2_delta_B.lp2 *
              sim4_ROS2_delta_B.lp2;
            if (sim4_ROS2_delta_B.c < 0.0) {
              exitg1 = 1;
            } else {
              sim4_ROS2_delta_B.c = sqrt(sim4_ROS2_delta_B.c);
              sim4_ROS2_delta_B.s /= sim4_ROS2_delta_B.d;
              sim4_ROS2_delta_B.d = rtb_Sum3_idx_2 / sim4_ROS2_delta_B.d;
              mx = sim4_ROS2_delta_B.lp2 * sim4_ROS2_delta_B.s + 163.21;
              sim4_ROS2_delta_B.lp2 *= sim4_ROS2_delta_B.d;
              e1_idx_0_tmp = sim4_ROS2_delta_B.c * -sim4_ROS2_delta_B.d;
              sim4_ROS2_delta_B.d = e1_idx_0_tmp + mx;
              mx -= e1_idx_0_tmp;
              sim4_ROS2_delta_B.c *= sim4_ROS2_delta_B.s;
              sim4_ROS2_delta_B.s = sim4_ROS2_delta_B.lp2 - sim4_ROS2_delta_B.c;
              if (sim4_ROS2_delta_B.d >= mx) {
                mx = sim4_ROS2_delta_B.d;
                sim4_ROS2_delta_B.s = sim4_ROS2_delta_B.c +
                  sim4_ROS2_delta_B.lp2;
              }

              sim4_ROS2_delta_B.thetas[i] = rt_atan2d_snf(-sim4_ROS2_delta_B.s,
                mx - 163.21) * 57.295779513082323;
              i++;
            }
          }
        }
      } else {
        b_varargout_1 = false;
        i = 0;
        exitg2 = false;
        while ((!exitg2) && (i < 3)) {
          if (sim4_ROS2_delta_B.thetas[i] < -5.0) {
            b_varargout_1 = true;
            exitg2 = true;
          } else {
            i++;
          }
        }

        if (!b_varargout_1) {
          b_varargout_1 = false;
          i = 0;
          exitg2 = false;
          while ((!exitg2) && (i < 3)) {
            if (sim4_ROS2_delta_B.thetas[i] > 90.0) {
              b_varargout_1 = true;
              exitg2 = true;
            } else {
              i++;
            }
          }

          if (!b_varargout_1) {
            sim4_ROS2_delta_B.BusAssignment1.theta1_deg =
              sim4_ROS2_delta_B.thetas[0];
            sim4_ROS2_delta_B.BusAssignment1.theta2_deg =
              sim4_ROS2_delta_B.thetas[1];
            sim4_ROS2_delta_B.BusAssignment1.theta3_deg =
              sim4_ROS2_delta_B.thetas[2];
            sim4_ROS2_delta_B.BusAssignment1.ik_valid = true;
          }
        }

        exitg1 = 1;
      }
    } while (exitg1 == 0);
  }

  // MATLABSystem: '<S4>/SinkBlock'
  Pub_sim4_ROS2_delta_72.publish(&sim4_ROS2_delta_B.BusAssignment1);
}

// Model initialize function
void sim4_ROS2_delta::initialize()
{
  // Registration code

  // initialize non-finites
  rt_InitInfAndNaN(sizeof(real_T));

  // Start for MATLABSystem: '<S5>/SourceBlock'
  sim4_ROS2_delta_DW.obj_l.QOSAvoidROSNamespaceConventions = false;
  sim4_ROS2_delta_DW.obj_l.matlabCodegenIsDeleted = false;
  sim4_ROS2_delta_DW.obj_l.isSetupComplete = false;
  sim4_ROS2_delta_DW.obj_l.isInitialized = 1;
  sim4_ROS2__Subscriber_setupImpl(&sim4_ROS2_delta_DW.obj_l);
  sim4_ROS2_delta_DW.obj_l.isSetupComplete = true;

  // Start for MATLABSystem: '<S4>/SinkBlock'
  sim4_ROS2_delta_DW.obj.QOSAvoidROSNamespaceConventions = false;
  sim4_ROS2_delta_DW.obj.matlabCodegenIsDeleted = false;
  sim4_ROS2_delta_DW.obj.isSetupComplete = false;
  sim4_ROS2_delta_DW.obj.isInitialized = 1;
  sim4_ROS2_d_Publisher_setupImpl(&sim4_ROS2_delta_DW.obj);
  sim4_ROS2_delta_DW.obj.isSetupComplete = true;
}

// Model terminate function
void sim4_ROS2_delta::terminate()
{
  // Terminate for MATLABSystem: '<S5>/SourceBlock'
  if (!sim4_ROS2_delta_DW.obj_l.matlabCodegenIsDeleted) {
    sim4_ROS2_delta_DW.obj_l.matlabCodegenIsDeleted = true;
    if ((sim4_ROS2_delta_DW.obj_l.isInitialized == 1) &&
        sim4_ROS2_delta_DW.obj_l.isSetupComplete) {
      Sub_sim4_ROS2_delta_75.resetSubscriberPtr();//();
    }
  }

  // End of Terminate for MATLABSystem: '<S5>/SourceBlock'

  // Terminate for MATLABSystem: '<S4>/SinkBlock'
  if (!sim4_ROS2_delta_DW.obj.matlabCodegenIsDeleted) {
    sim4_ROS2_delta_DW.obj.matlabCodegenIsDeleted = true;
    if ((sim4_ROS2_delta_DW.obj.isInitialized == 1) &&
        sim4_ROS2_delta_DW.obj.isSetupComplete) {
      Pub_sim4_ROS2_delta_72.resetPublisherPtr();//();
    }
  }

  // End of Terminate for MATLABSystem: '<S4>/SinkBlock'
}

// Constructor
sim4_ROS2_delta::sim4_ROS2_delta() :
  sim4_ROS2_delta_B(),
  sim4_ROS2_delta_DW(),
  sim4_ROS2_delta_M()
{
  // Currently there is no constructor body generated.
}

// Destructor
sim4_ROS2_delta::~sim4_ROS2_delta()
{
  // Currently there is no destructor body generated.
}

// Real-Time Model get method
RT_MODEL_sim4_ROS2_delta_T * sim4_ROS2_delta::getRTM()
{
  return (&sim4_ROS2_delta_M);
}

const char_T* RT_MODEL_sim4_ROS2_delta_T::getErrorStatus() const
{
  return (errorStatus);
}

void RT_MODEL_sim4_ROS2_delta_T::setErrorStatus(const char_T* const volatile
  aErrorStatus)
{
  (errorStatus = aErrorStatus);
}

//
// File trailer for generated code.
//
// [EOF]
//
