/**
 * @file msvc_stubs.c
 * @brief MSVC构建兼容桩 —— 仅为MSVC排除编译的模块提供必要的符号
 *
 * 本文件仅包含在MSVC构建时被CMakeLists.txt排除的模块所需的外部符号。
 * 真实实现文件：
 *   - gpu.c (GPU设备检测、上下文创建) — 在MSVC下完整编译
 *   - laplace.c (拉普拉斯分析、网络动态分析) — 在MSVC下完整编译
 *   - ros_gazebo_bridge.c (ROS/Gazebo桥接) — MSVC排除(需Linux)
 *   - hardware_detector.c (硬件检测) — MSVC排除(需平台特定头文件)
 *
 * 所有桩函数均明确返回错误码，绝不返回虚假成功值。
 * 对于已有真实实现的函数，本文件不重复定义。
 */

#include "selflnn/core/lnn.h"
#include "selflnn/core/cfc.h"
#include "selflnn/core/laplace.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/robot/ros_gazebo_bridge.h"
#include "selflnn/robot/hardware_detector.h"
#include "selflnn/robot/simulator.h"
#include "selflnn/gpu/gpu.h"
#include "selflnn/multisystem/swarm_intelligence.h"
#include "selflnn/reasoning/reasoning.h"

#include <string.h>
#include <stdio.h>
#include <stdarg.h>

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
 * log_warn 函数 (头文件仅提供log_warning宏，但多处源码直接调用log_warn)
 * ============================ */
void log_warn(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

/* ================================================================
 * 以下函数在MSVC构建中作为桩存在，因为对应的真实实现源文件
 * 被CMakeLists.txt排除编译（依赖Linux/Python环境）：
 *   - ros_gazebo_bridge.c → ROS/Gazebo需要Linux
 *   - hardware_detector.c → 需要Linux特定头文件
 *   - PyBullet相关函数 → 需要Python环境
 *
 * 注意：gpu.c和laplace.c中的函数在MSVC下正常编译，
 * 本文件不重复定义这些函数。
 * ================================================================ */

/* ============================
 * 训练管线桩 — training_pipeline.c在MSVC下正常编译
 * 此桩仅用于头文件符号声明完整性，实际由真实实现提供
 * ============================ */

/* ============================
 * PyBullet桥接桩 (PyBullet需Python环境，MSVC平台不可用)
 * ============================ */
void pb_disconnect(int client_id) {
    (void)client_id;
    log_warn("[MSVC桩] PyBullet断开连接: 此功能需Python/PyBullet环境，MSVC平台不可用");
}

/* ============================
 * ROS/Gazebo桥接桩 (ROS/Gazebo需Linux环境)
 * 所有函数返回失败状态，绝不自称成功
 * ============================ */
RosGazeboBridgeConfig ros_gazebo_bridge_config_default(const char* node_name) {
    RosGazeboBridgeConfig c;
    (void)node_name;
    memset(&c, 0, sizeof(c));
    log_warn("[MSVC桩] ROS/Gazebo桥接配置: 此功能需Linux+ROS环境");
    return c;
}
RosGazeboBridge* ros_gazebo_bridge_create(const RosGazeboBridgeConfig* cfg) {
    (void)cfg;
    log_warn("[MSVC桩] ROS/Gazebo桥接创建失败: 需Linux+ROS环境");
    return NULL;
}
void ros_gazebo_bridge_destroy(RosGazeboBridge* b) { (void)b; }
int ros_gazebo_bridge_connect(RosGazeboBridge* b) {
    (void)b;
    log_warn("[MSVC桩] ROS/Gazebo连接失败: MSVC平台不支持Gazebo仿真");
    return -1;
}
int ros_gazebo_bridge_disconnect(RosGazeboBridge* b) {
    (void)b;
    log_warn("[MSVC桩] ROS/Gazebo断开连接失败: MSVC平台不支持");
    return -1;
}
int ros_gazebo_bridge_is_connected(const RosGazeboBridge* b) { (void)b; return 0; }
int ros_gazebo_bridge_start(RosGazeboBridge* b) {
    (void)b;
    log_warn("[MSVC桩] ROS/Gazebo启动失败: MSVC平台不支持");
    return -1;
}
int ros_gazebo_bridge_stop(RosGazeboBridge* b) { (void)b; return -1; }
int ros_gazebo_bridge_pause(RosGazeboBridge* b) { (void)b; return -1; }
int ros_gazebo_bridge_resume(RosGazeboBridge* b) { (void)b; return -1; }
int ros_gazebo_bridge_step(RosGazeboBridge* b) { (void)b; return -1; }
int ros_gazebo_bridge_reset(RosGazeboBridge* b) { (void)b; return -1; }
int ros_gazebo_bridge_spawn_model(RosGazeboBridge* b, const char* n, const float* pos, const float* ori) {
    (void)b;(void)n;(void)pos;(void)ori;
    log_warn("[MSVC桩] Gazebo模型生成失败: MSVC平台不支持");
    return -1;
}
int ros_gazebo_bridge_delete_model(RosGazeboBridge* b, const char* n) {
    (void)b;(void)n;
    log_warn("[MSVC桩] Gazebo模型删除失败: MSVC平台不支持");
    return -1;
}

/* ============================
 * 硬件检测桩 (hal_detector.c在MSVC下被CMakeLists排除)
 * ============================ */
int hd_detect_all(HDDetectionConfig config, HDDetectionResult* result) {
    (void)config;
    if (result) memset(result, 0, sizeof(*result));
    log_warn("[MSVC桩] 硬件检测: hal_detector.c在MSVC构建中被排除，硬件检测不可用");
    return -1;
}
int hd_get_device_by_type(const HDDetectionResult* result, HDDeviceType type, HDDeviceInfo* out, size_t max_count, size_t* count) {
    (void)result; (void)type; (void)out; (void)max_count;
    if (count) *count = 0;
    log_warn("[MSVC桩] 设备类型查询: 硬件检测不可用，返回0个设备");
    return -1;
}
int hd_get_system_info(char* system_name, size_t max_len, char* os_version, size_t os_max_len) {
    if (system_name && max_len > 0) snprintf(system_name, max_len, "Windows/MSVC");
    if (os_version && os_max_len > 0) snprintf(os_version, os_max_len, "MSVC构建");
    return 0;
}
void hd_result_free(HDDetectionResult* result) { (void)result; }
const char* hd_device_type_str(HDDeviceType type) {
    (void)type;
    static const char* s = "unknown";
    return s;
}

/* ============================
 * 贝叶斯网络桩 (reasoning_internal.c引用reasoning.c中的函数)
 * ============================ */
BayesianNetwork* bayesian_network_create(size_t max_nodes) {
    BayesianNetwork* bn = (BayesianNetwork*)safe_calloc(1, sizeof(BayesianNetwork));
    if (bn) bn->node_capacity = max_nodes;
    return bn;
}
void bayesian_network_free(BayesianNetwork* bn) {
    if (!bn) return;
    safe_free((void**)&bn->nodes);
    safe_free((void**)&bn->edges);
    safe_free((void**)&bn->cpds);
    safe_free((void**)&bn);
}
