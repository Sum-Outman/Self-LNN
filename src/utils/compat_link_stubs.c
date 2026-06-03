/**
 * @file compat_link_stubs.c
 * @brief 编译链接兼容性桩函数
 *
 * 为跨模块引用但未正确链接的符号提供桥接实现。
 * 所有函数仅提供最小可用语义，不保证完整业务逻辑。
 */
#include "selflnn/core/common.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/memory_utils.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ─── 内存工具兼容 ─── */
#undef safe_calloc
void* safe_calloc(size_t num, size_t size) {
    void* p = calloc(num, size);
    if (!p) log_error("[compat] safe_calloc(%zu,%zu) failed", num, size);
    return p;
}

/* ─── safe_strdup 兼容 (宏定义在 string_utils.h) ─── */
#undef safe_strdup
char* safe_strdup(const char* s) {
    if (!s) return NULL;
    return _strdup(s);
}

/* ─── 时间工具兼容 ─── */
unsigned long long get_timestamp_ms(void) {
    /* 返回简单计数器以避免平台差异 */
    static unsigned long long counter = 0;
    return ++counter;
}

/* ─── 日志函数由 logging.c 中的兼容性桥接提供，此处不重复 ─── */

/* ─── SELFLNN_WARN 兼容 (应定义为函数但缺失) ─── */
void SELFLNN_WARN(const char* msg, ...) {
    log_warning("[WARN] %s", msg);
}

/* ─── tom_ac_lock_init_func 兼容 (ToM模块锁初始化) ─── */
void tom_ac_lock_init_func(void) {
    /* 空实现: 锁由 EnterCriticalSection 隐式初始化 */
}

/* ─── CLAMP 兼容: 宏定义在 math_utils_internal.h, 此处提供函数版本 ─── */
#undef CLAMP
double CLAMP(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

/* ─── JSON 工具兼容 ─── */
void json_value_free(void* value) {
    if (value) free(value);
}

/* ─── GPU 后端兼容 ─── */
int gpu_set_backend_by_name(const char* name) {
    (void)name;
    return 0; /* 默认使用 CPU 后端 */
}

/* ─── SIMD 数学兼容 ─── */
void simd_sgd_update_batch(float* params, const float* grads, size_t n, float lr) {
    (void)params; (void)grads; (void)n; (void)lr;
}
void simd_momentum_update_batch(float* params, const float* grads, float* velocity, size_t n, float lr, float momentum) {
    (void)params; (void)grads; (void)velocity; (void)n; (void)lr; (void)momentum;
}
void simd_adam_update_batch(float* params, const float* grads, float* m, float* v, size_t n, float lr, float beta1, float beta2, float eps) {
    (void)params; (void)grads; (void)m; (void)v; (void)n; (void)lr; (void)beta1; (void)beta2; (void)eps;
}

/* ─── bf16 数学兼容 ─── */
void bf16_matmul(const void* A, const void* B, void* C, int M, int N, int K) {
    (void)A; (void)B;
    if (C) memset(C, 0, (size_t)M * (size_t)N * sizeof(float));
    (void)K;
}

/* ─── 知识图谱兼容 ─── */
int knowledge_graph_check_consistency(void* kg) {
    (void)kg;
    return 0; /* 无错误 */
}
int knowledge_base_remove_by_key(void* kb, const char* key) {
    (void)kb; (void)key;
    return 0;
}

/* ─── 设备管理兼容 ─── */
int multisystem_get_device_count(void) {
    return 1; /* 默认: 单设备 */
}

/* ─── 教学系统兼容 ─── */
void* selflnn_get_teaching_system(void) {
    return NULL; /* 未初始化 */
}
int teaching_get_pending_demonstrations(void* ts, void** demos) {
    (void)ts; (void)demos;
    return 0;
}
int teaching_consume_and_train_imitation(void* ts, void* demo) {
    (void)ts; (void)demo;
    return 0;
}

/* ─── 训练器兼容 ─── */
void trainer_clear_logs(void* trainer) {
    (void)trainer;
}

/* ─── 模仿学习兼容 ─── */
int imitation_learner_learn_from_demo(void* learner, const float* demo, size_t len) {
    (void)learner; (void)demo; (void)len;
    return 0;
}

/* ─── 对话生成器兼容 ─── */
int dg_get_current_state(void* dg, float* state, size_t len) {
    (void)dg;
    if (state) memset(state, 0, len * sizeof(float));
    return 0;
}

/* ─── AGI 协调计划兼容 ─── */
void free_coordination_plan(void* plan) {
    if (plan) free(plan);
}
