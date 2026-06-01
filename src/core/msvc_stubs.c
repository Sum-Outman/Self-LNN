/**
 * @file msvc_stubs.c
 * @brief MSVC构建兼容桩 —— 删除与真实实现冲突的桩函数
 *
 * 以下模块现已全平台编译，本文件不再提供桩：
 *   - ros_gazebo_bridge.c — robot/CMakeLists.txt全平台编译
 *   - hardware_detector.c — robot/CMakeLists.txt全平台编译
 *   - reasoning_internal.c — reasoning/CMakeLists.txt MSVC专用
 *   - training_pipeline.c — training/CMakeLists.txt全平台编译
 *
 * 本文件仅保留 PyBullet 桥接桩（PyBullet需要Python环境，MSVC平台不可用）。
 * 所有桩函数均明确返回错误码，绝不返回虚假成功值。
 */

#include "selflnn/utils/logging.h"

#include <string.h>
#include <stdio.h>

/* ============================
 * 宏定义 (头文件链中缺失的常见函数式宏)
 * ============================ */
#ifndef RL_MAX
#define RL_MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef RL_MIN
#define RL_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

/* ============================
 * PyBullet桥接桩 (PyBullet需Python环境，MSVC平台不可用)
 *修复: 当PYBULLET_BRIDGE_AVAILABLE未定义时，
 * 不提供静默返回-1的桩函数。直接让链接器报告符号缺失，
 * 迫使构建系统明确处理缺失依赖，而非静默降级。
 * ============================ */
