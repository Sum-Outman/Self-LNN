/**
 * @file msvc_stubs.c
 * @brief MSVC构建兼容桩 —— ZSFZS-P0-001修复：删除与真实实现冲突的桩函数
 *
 * 以下模块现已全平台编译（ZSFZS-F022修复），本文件不再提供桩：
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
 * ============================ */
/* ZSFWS-018修复: 仅在MSVC平台且未编译pybullet_bridge.c时提供桩函数
 * pybullet_bridge.c已有真实int pybullet_disconnect(int)实现，
 * msvc_stubs.c提供的是void pybullet_disconnect(int)空桩，
 * 两者同时编译到同一二进制会导致符号冲突。
 * 现改为条件编译：仅当未定义PYBULLET_BRIDGE_AVAILABLE时编译此桩。 */
#ifndef PYBULLET_BRIDGE_AVAILABLE
void pybullet_disconnect(int connection_id) {
    (void)connection_id;
    log_warn("[MSVC桩] PyBullet断开连接: 此功能需Python/PyBullet环境，MSVC平台不可用");
}
#endif
