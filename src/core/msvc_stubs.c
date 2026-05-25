/**
 * @file msvc_stubs.c
 * @brief MSVC构建兼容桩 - 提供非MSVC平台功能的最小实现
 *
 * ZSFBUILD: 此文件为MSVC构建提供缺失符号的桩实现。
 * 涉及的子系统: ROS/Gazebo/PyBullet/硬件检测/Laplace增强/GPU/Swarm等
 */

#include "selflnn/core/lnn.h"
#include "selflnn/core/cfc.h"
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

/* ============================
 * 训练管线
 * ============================ */
void training_pipeline_destroy(void* pipeline) {
    (void)pipeline;
}

/* ============================
 * LNN模型加载 (stub)
 * ============================ */
int lnn_load_from_file(LNN* lnn, const char* filepath) {
    (void)lnn;
    log_warning("MSVC构建: lnn_load_from_file未实现, file=%s", filepath ? filepath : "null");
    return -1;
}

/* ============================
 * PyBullet桥接 (非MSVC平台不可用)
 * ============================ */
void pb_disconnect(int client_id) { (void)client_id; }

/* ============================
 * ROS/Gazebo桥接桩
 * ============================ */
RosGazeboBridgeConfig ros_gazebo_bridge_config_default(const char* node_name) {
    RosGazeboBridgeConfig c;
    (void)node_name;
    memset(&c, 0, sizeof(c));
    return c;
}
RosGazeboBridge* ros_gazebo_bridge_create(const RosGazeboBridgeConfig* cfg) { (void)cfg; return NULL; }
void ros_gazebo_bridge_destroy(RosGazeboBridge* b) { (void)b; }
int ros_gazebo_bridge_connect(RosGazeboBridge* b) { (void)b; return -1; }
int ros_gazebo_bridge_disconnect(RosGazeboBridge* b) { (void)b; return 0; }
int ros_gazebo_bridge_is_connected(const RosGazeboBridge* b) { (void)b; return 0; }
int ros_gazebo_bridge_start(RosGazeboBridge* b) { (void)b; return -1; }
int ros_gazebo_bridge_stop(RosGazeboBridge* b) { (void)b; return -1; }
int ros_gazebo_bridge_pause(RosGazeboBridge* b) { (void)b; return -1; }
int ros_gazebo_bridge_resume(RosGazeboBridge* b) { (void)b; return -1; }
int ros_gazebo_bridge_step(RosGazeboBridge* b) { (void)b; return -1; }
int ros_gazebo_bridge_reset(RosGazeboBridge* b) { (void)b; return -1; }
int ros_gazebo_bridge_spawn_model(RosGazeboBridge* b, const char* n, const float* pos, const float* ori) { (void)b;(void)n;(void)pos;(void)ori; return -1; }
int ros_gazebo_bridge_delete_model(RosGazeboBridge* b, const char* n) { (void)b;(void)n; return -1; }

/* ============================
 * 硬件检测桩
 * ============================ */
int hd_detect_all(HDDetectionConfig config, HDDetectionResult* result) {
    (void)config;
    if (result) memset(result, 0, sizeof(*result));
    return -1;
}
int hd_get_device_by_type(const HDDetectionResult* result, HDDeviceType type, HDDeviceInfo* out, size_t max_count, size_t* count) {
    (void)result; (void)type; (void)out; (void)max_count;
    if (count) *count = 0;
    return -1;
}
int hd_get_system_info(char* system_name, size_t max_len, char* os_version, size_t os_max_len) {
    if (system_name && max_len > 0) snprintf(system_name, max_len, "Windows");
    if (os_version && os_max_len > 0) snprintf(os_version, os_max_len, "MSVC");
    return 0;
}
void hd_result_free(HDDetectionResult* result) { (void)result; }
const char* hd_device_type_str(HDDeviceType type) {
    static const char* s = "unknown";
    (void)type;
    return s;
}

/* ============================
 * GPU桩
 * ============================ */
const char* gpu_backend_name(GpuBackend backend) { (void)backend; return "CPU"; }
int gpu_probe_backend(GpuBackend backend, GpuBackendAvailability* info) {
    if (info) { memset(info, 0, sizeof(*info)); info->is_available = 0; }
    (void)backend;
    return -1;
}
int gpu_is_available(void) { return 0; }

GpuContext* gpu_context_create(GpuBackend backend, int device_index) {
    (void)backend; (void)device_index;
    return NULL;
}

/* ============================
 * Laplace增强桩
 * ============================ */
void* lnn_laplace_create_default_analyzer(LNN* lnn) { (void)lnn; return NULL; }
int lnn_laplace_modulate_hidden(LNN* lnn, int layer_idx, float* hidden, size_t hidden_size) {
    (void)lnn; (void)layer_idx; (void)hidden; (void)hidden_size;
    return -1;
}
int lnn_laplace_analyze_network_dynamics(LNN* lnn, float* stability_margin, float* spectral_radius) {
    (void)lnn;
    if (stability_margin) *stability_margin = 1.0f;
    if (spectral_radius) *spectral_radius = 0.0f;
    return 0;
}

/* ============================
 * Swarm桩
 * ============================ */
int swarm_get_ms_state(Swarm* swarm, MSSwarmState* state) {
    (void)swarm;
    if (state) memset(state, 0, sizeof(*state));
    return -1;
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
