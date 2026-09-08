#ifndef SIM4_ROS2_DELTA__VISIBILITY_CONTROL_H_
#define SIM4_ROS2_DELTA__VISIBILITY_CONTROL_H_
#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define SIM4_ROS2_DELTA_EXPORT __attribute__ ((dllexport))
    #define SIM4_ROS2_DELTA_IMPORT __attribute__ ((dllimport))
  #else
    #define SIM4_ROS2_DELTA_EXPORT __declspec(dllexport)
    #define SIM4_ROS2_DELTA_IMPORT __declspec(dllimport)
  #endif
  #ifdef SIM4_ROS2_DELTA_BUILDING_LIBRARY
    #define SIM4_ROS2_DELTA_PUBLIC SIM4_ROS2_DELTA_EXPORT
  #else
    #define SIM4_ROS2_DELTA_PUBLIC SIM4_ROS2_DELTA_IMPORT
  #endif
  #define SIM4_ROS2_DELTA_PUBLIC_TYPE SIM4_ROS2_DELTA_PUBLIC
  #define SIM4_ROS2_DELTA_LOCAL
#else
  #define SIM4_ROS2_DELTA_EXPORT __attribute__ ((visibility("default")))
  #define SIM4_ROS2_DELTA_IMPORT
  #if __GNUC__ >= 4
    #define SIM4_ROS2_DELTA_PUBLIC __attribute__ ((visibility("default")))
    #define SIM4_ROS2_DELTA_LOCAL  __attribute__ ((visibility("hidden")))
  #else
    #define SIM4_ROS2_DELTA_PUBLIC
    #define SIM4_ROS2_DELTA_LOCAL
  #endif
  #define SIM4_ROS2_DELTA_PUBLIC_TYPE
#endif
#endif  // SIM4_ROS2_DELTA__VISIBILITY_CONTROL_H_
// Generated 22-Aug-2026 08:21:16
 