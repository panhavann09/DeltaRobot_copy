#ifndef _SLROS_BUSMSG_CONVERSION_H_
#define _SLROS_BUSMSG_CONVERSION_H_

#include "rclcpp/rclcpp.hpp"
#include <builtin_interfaces/msg/time.hpp>
#include <custom_messages/msg/delta_joint_angles.hpp>
#include <custom_messages/msg/delta_target.hpp>
#include <std_msgs/msg/header.hpp>
#include "sim4_ROS2_delta_types.h"
#include "slros_msgconvert_utils.h"


[[maybe_unused]] static void convertFromBus(builtin_interfaces::msg::Time& msgPtr, SL_Bus_builtin_interfaces_Time const* busPtr);
[[maybe_unused]] static void convertToBus(SL_Bus_builtin_interfaces_Time* busPtr, const builtin_interfaces::msg::Time& msgPtr);

[[maybe_unused]] static void convertFromBus(custom_messages::msg::DeltaJointAngles& msgPtr, SL_Bus_custom_messages_DeltaJointAngles const* busPtr);
[[maybe_unused]] static void convertToBus(SL_Bus_custom_messages_DeltaJointAngles* busPtr, const custom_messages::msg::DeltaJointAngles& msgPtr);

[[maybe_unused]] static void convertFromBus(custom_messages::msg::DeltaTarget& msgPtr, SL_Bus_custom_messages_DeltaTarget const* busPtr);
[[maybe_unused]] static void convertToBus(SL_Bus_custom_messages_DeltaTarget* busPtr, const custom_messages::msg::DeltaTarget& msgPtr);

[[maybe_unused]] static void convertFromBus(std_msgs::msg::Header& msgPtr, SL_Bus_std_msgs_Header const* busPtr);
[[maybe_unused]] static void convertToBus(SL_Bus_std_msgs_Header* busPtr, const std_msgs::msg::Header& msgPtr);



// Conversions between SL_Bus_builtin_interfaces_Time and builtin_interfaces::msg::Time

[[maybe_unused]] static void convertFromBus(builtin_interfaces::msg::Time& msgPtr, SL_Bus_builtin_interfaces_Time const* busPtr)
{
  const std::string rosMessageType("builtin_interfaces/Time");

  msgPtr.nanosec =  busPtr->nanosec;
  msgPtr.sec =  busPtr->sec;
}

[[maybe_unused]] static void convertToBus(SL_Bus_builtin_interfaces_Time* busPtr, const builtin_interfaces::msg::Time& msgPtr)
{
  const std::string rosMessageType("builtin_interfaces/Time");

  busPtr->nanosec =  msgPtr.nanosec;
  busPtr->sec =  msgPtr.sec;
}


// Conversions between SL_Bus_custom_messages_DeltaJointAngles and custom_messages::msg::DeltaJointAngles

[[maybe_unused]] static void convertFromBus(custom_messages::msg::DeltaJointAngles& msgPtr, SL_Bus_custom_messages_DeltaJointAngles const* busPtr)
{
  const std::string rosMessageType("custom_messages/DeltaJointAngles");

  convertFromBus(msgPtr.header, &busPtr->header);
  msgPtr.ik_valid =  busPtr->ik_valid;
  msgPtr.theta1_deg =  busPtr->theta1_deg;
  msgPtr.theta2_deg =  busPtr->theta2_deg;
  msgPtr.theta3_deg =  busPtr->theta3_deg;
}

[[maybe_unused]] static void convertToBus(SL_Bus_custom_messages_DeltaJointAngles* busPtr, const custom_messages::msg::DeltaJointAngles& msgPtr)
{
  const std::string rosMessageType("custom_messages/DeltaJointAngles");

  convertToBus(&busPtr->header, msgPtr.header);
  busPtr->ik_valid =  msgPtr.ik_valid;
  busPtr->theta1_deg =  msgPtr.theta1_deg;
  busPtr->theta2_deg =  msgPtr.theta2_deg;
  busPtr->theta3_deg =  msgPtr.theta3_deg;
}


// Conversions between SL_Bus_custom_messages_DeltaTarget and custom_messages::msg::DeltaTarget

[[maybe_unused]] static void convertFromBus(custom_messages::msg::DeltaTarget& msgPtr, SL_Bus_custom_messages_DeltaTarget const* busPtr)
{
  const std::string rosMessageType("custom_messages/DeltaTarget");

  msgPtr.confidence =  busPtr->confidence;
  convertFromBusVariablePrimitiveArray(msgPtr.detection_mode, busPtr->detection_mode, busPtr->detection_mode_SL_Info);
  convertFromBus(msgPtr.header, &busPtr->header);
  msgPtr.track_id =  busPtr->track_id;
  msgPtr.x_mm =  busPtr->x_mm;
  msgPtr.y_mm =  busPtr->y_mm;
  msgPtr.z_mm =  busPtr->z_mm;
}

[[maybe_unused]] static void convertToBus(SL_Bus_custom_messages_DeltaTarget* busPtr, const custom_messages::msg::DeltaTarget& msgPtr)
{
  const std::string rosMessageType("custom_messages/DeltaTarget");

  busPtr->confidence =  msgPtr.confidence;
  convertToBusVariablePrimitiveArray(busPtr->detection_mode, busPtr->detection_mode_SL_Info, msgPtr.detection_mode, slros::EnabledWarning(rosMessageType, "detection_mode"));
  convertToBus(&busPtr->header, msgPtr.header);
  busPtr->track_id =  msgPtr.track_id;
  busPtr->x_mm =  msgPtr.x_mm;
  busPtr->y_mm =  msgPtr.y_mm;
  busPtr->z_mm =  msgPtr.z_mm;
}


// Conversions between SL_Bus_std_msgs_Header and std_msgs::msg::Header

[[maybe_unused]] static void convertFromBus(std_msgs::msg::Header& msgPtr, SL_Bus_std_msgs_Header const* busPtr)
{
  const std::string rosMessageType("std_msgs/Header");

  convertFromBusVariablePrimitiveArray(msgPtr.frame_id, busPtr->frame_id, busPtr->frame_id_SL_Info);
  convertFromBus(msgPtr.stamp, &busPtr->stamp);
}

[[maybe_unused]] static void convertToBus(SL_Bus_std_msgs_Header* busPtr, const std_msgs::msg::Header& msgPtr)
{
  const std::string rosMessageType("std_msgs/Header");

  convertToBusVariablePrimitiveArray(busPtr->frame_id, busPtr->frame_id_SL_Info, msgPtr.frame_id, slros::EnabledWarning(rosMessageType, "frame_id"));
  convertToBus(&busPtr->stamp, msgPtr.stamp);
}



#endif
