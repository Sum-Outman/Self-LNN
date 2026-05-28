/**
 * @file state.c
 * @brief 网络状态管理实现
 * 
 * 管理和监控液态神经网络的状态，计算稳定性、收敛性等指标。
 */

#include "selflnn/core/state.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/platform.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

/**
 * @brief 网络状态内部结构体
 */
struct NetworkState {
    NetworkStateConfig config;      /**< 状态配置 */
    float* current_state;           /**< 当前状态向量 */
    float* previous_state;          /**< 前一状态向量 */
    float* state_history;           /**< 状态历史缓冲区 */
    int history_index;              /**< 历史索引 */
    int history_size;               /**< 历史大小 */
    int is_initialized;             /**< 是否已初始化 */
    float stability;                /**< 当前稳定性 */
    float convergence;              /**< 当前收敛率 */
    float change_rate;              /**< 状态变化率 */
    float avg_state_magnitude;      /**< 平均状态幅度 */
    /* 拉普拉斯频域稳定性指标 */
    float laplace_stability_score;      /**< 拉普拉斯稳定性分数 (0.0-1.0) */
    float laplace_recommended_cutoff;   /**< 推荐的滤波截止频率 (Hz) */
    float laplace_frequency_bandwidth;  /**< 频域带宽 (Hz) */
};

/**
 * @brief 创建网络状态实例
 */
NetworkState* network_state_create(const NetworkStateConfig* config) {
    if (!config) {
        return NULL;
    }
    
    if (config->state_size == 0) {
        return NULL;
    }
    
    // 分配状态结构
    NetworkState* state = (NetworkState*)safe_malloc(sizeof(NetworkState));
    if (!state) {
        return NULL;
    }
    
    // 复制配置
    memcpy(&state->config, config, sizeof(NetworkStateConfig));
    
    size_t state_size = config->state_size;
    
    // 分配状态向量
    state->current_state = (float*)safe_calloc(state_size, sizeof(float));
    state->previous_state = (float*)safe_calloc(state_size, sizeof(float));
    
    // 分配历史缓冲区（保留最近100个状态）
    state->history_size = 100;
    state->state_history = (float*)safe_calloc(state_size * state->history_size, sizeof(float));
    
    // 检查内存分配
    if (!state->current_state || !state->previous_state || !state->state_history) {
        network_state_free(state);
        return NULL;
    }
    
    // 初始化状态
    for (size_t i = 0; i < state_size; i++) {
        state->current_state[i] = 0.0f;
        state->previous_state[i] = 0.0f;
    }
    
    // 初始化历史缓冲区
    memset(state->state_history, 0, state_size * state->history_size * sizeof(float));
    
    // 初始化变量
    state->history_index = 0;
    state->is_initialized = 1;
    state->stability = 1.0f;
    state->convergence = 0.0f;
    state->change_rate = 0.0f;
    state->avg_state_magnitude = 0.0f;
    state->laplace_stability_score = 0.5f;
    state->laplace_recommended_cutoff = 10.0f;
    state->laplace_frequency_bandwidth = 50.0f;
    
    return state;
}

/**
 * @brief 创建网络状态实例（完整版本）
 */
NetworkState* network_state_create_simple(size_t state_size) {
    NetworkStateConfig config;
    config.state_size = state_size;
    config.stability_threshold = 0.9f;
    config.convergence_rate = 0.1f;
    config.enable_monitoring = 1;
    
    return network_state_create(&config);
}

/**
 * @brief 释放网络状态实例
 */
void network_state_free(NetworkState* state) {
    if (!state) {
        return;
    }
    
    // 如果状态未正确初始化，直接释放状态结构，跳过内部指针
    if (state->is_initialized != 1) {
        safe_free((void**)&state);
        return;
    }
    
    // 释放状态向量
    // safe_free内部已有魔法数字验证、重复释放检测、对齐检查、尾部验证等完整机制
    // 不需要额外的地址范围硬编码检查（该检查在Windows x64上会误拦截合法指针）
    safe_free((void**)&state->current_state);
    safe_free((void**)&state->previous_state);
    safe_free((void**)&state->state_history);
    
    // 释放状态结构
    safe_free((void**)&state);
}

/**
 * @brief 更新网络状态
 */
int network_state_update(NetworkState* state, const float* new_state, size_t state_size) {
    if (!state || !new_state) {
        return -1;
    }
    
    if (!state->is_initialized) {
        return -1;
    }
    
    if (state_size != state->config.state_size) {
        return -1;
    }
    
    // 保存前一状态
    memcpy(state->previous_state, state->current_state, state_size * sizeof(float));
    
    // 更新当前状态
    memcpy(state->current_state, new_state, state_size * sizeof(float));
    
    // 更新历史缓冲区
    float* history_entry = state->state_history + (state->history_index * state_size);
    memcpy(history_entry, new_state, state_size * sizeof(float));
    
    state->history_index = (state->history_index + 1) % state->history_size;
    
    // 计算状态变化率
    float total_change = 0.0f;
    float total_magnitude = 0.0f;
    
    for (size_t i = 0; i < state_size; i++) {
        float change = state->current_state[i] - state->previous_state[i];
        total_change += fabsf(change);
        total_magnitude += fabsf(state->current_state[i]);
    }
    
    state->change_rate = total_change / state_size;
    state->avg_state_magnitude = total_magnitude / state_size;
    
    // 计算稳定性（基于变化率）
    // 稳定性 = 1 / (1 + 变化率)
    if (state->change_rate > 0.0f) {
        state->stability = 1.0f / (1.0f + state->change_rate);
    } else {
        state->stability = 1.0f;
    }
    
    // 计算收敛率（基于状态幅度的变化）
    if (state->history_index >= 10) {
        // 计算历史平均值
        float historical_avg = 0.0f;
        int count = 0;
        
        for (int j = 0; j < 10; j++) {
            int hist_idx = (state->history_index - j - 1 + state->history_size) % state->history_size;
            float* hist_entry = state->state_history + (hist_idx * state_size);
            
            float hist_magnitude = 0.0f;
            for (size_t i = 0; i < state_size; i++) {
                hist_magnitude += fabsf(hist_entry[i]);
            }
            hist_magnitude /= state_size;
            
            historical_avg += hist_magnitude;
            count++;
        }
        
        if (count > 0) {
            historical_avg /= count;
            
            // 收敛率 = 当前幅度与历史平均幅度的比值（越接近1越收敛）
            if (historical_avg > 0.0f) {
                state->convergence = 1.0f - fabsf(state->avg_state_magnitude - historical_avg) / historical_avg;
                if (state->convergence < 0.0f) state->convergence = 0.0f;
                if (state->convergence > 1.0f) state->convergence = 1.0f;
            }
        }
    }
    
    return 0;
}

/**
 * @brief 获取网络稳定性指标
 */
int network_state_get_stability(const NetworkState* state, float* stability) {
    if (!state || !stability) {
        return -1;
    }
    
    *stability = state->stability;
    return 0;
}

/**
 * @brief 获取网络收敛率
 */
int network_state_get_convergence(const NetworkState* state, float* convergence) {
    if (!state || !convergence) {
        return -1;
    }
    
    *convergence = state->convergence;
    return 0;
}

/**
 * @brief 获取网络状态变化率
 */
int network_state_get_change_rate(const NetworkState* state, float* change_rate) {
    if (!state || !change_rate) {
        return -1;
    }
    
    *change_rate = state->change_rate;
    return 0;
}

/**
 * @brief 重置网络状态
 */
void network_state_reset(NetworkState* state) {
    if (!state || !state->is_initialized) {
        return;
    }
    
    size_t state_size = state->config.state_size;
    
    // 重置状态向量
    memset(state->current_state, 0, state_size * sizeof(float));
    memset(state->previous_state, 0, state_size * sizeof(float));
    
    // 重置历史缓冲区
    memset(state->state_history, 0, state_size * state->history_size * sizeof(float));
    
    // 重置索引
    state->history_index = 0;
    
    // 重置统计信息
    state->stability = 1.0f;
    state->convergence = 0.0f;
    state->change_rate = 0.0f;
    state->avg_state_magnitude = 0.0f;
    state->laplace_stability_score = 0.5f;
    state->laplace_recommended_cutoff = 10.0f;
    state->laplace_frequency_bandwidth = 50.0f;
}

/**
 * @brief 获取网络状态配置
 */
int network_state_get_config(const NetworkState* state, NetworkStateConfig* config) {
    if (!state || !config) {
        return -1;
    }
    
    memcpy(config, &state->config, sizeof(NetworkStateConfig));
    return 0;
}

/**
 * @brief 设置网络状态配置
 */
int network_state_set_config(NetworkState* state, const NetworkStateConfig* config) {
    if (!state || !config) {
        return -1;
    }
    
    // 验证配置
    if (config->state_size != state->config.state_size) {
        return -1;  // 状态大小不匹配
    }
    
    // 更新配置
    state->config.stability_threshold = config->stability_threshold;
    state->config.convergence_rate = config->convergence_rate;
    state->config.enable_monitoring = config->enable_monitoring;
    
    return 0;
}

/* ============================================================================
 * 状态快照与回滚机制
 * ============================================================================ */

#define MAX_SNAPSHOTS 8

typedef struct {
    float* state_data;
    size_t state_size;
    int step;
    char label[64];
    /* Z-P3修复: 添加任务ID字段 —— 使状态快照与任务/模态上下文关联
     * 在多任务训练中可区分不同任务的状态快照，避免跨任务状态串扰 */
    char task_id[64];
} StateSnapshot;

static StateSnapshot g_snapshots[MAX_SNAPSHOTS];
static int g_snapshot_count = 0;
static MutexHandle g_snapshot_mutex = NULL;

int network_state_snapshot_save(NetworkState* state, const char* label) {
    if (!state || !state->current_state) return -1;
    if (!g_snapshot_mutex) g_snapshot_mutex = mutex_create();
    mutex_lock(g_snapshot_mutex);

    if (g_snapshot_count >= MAX_SNAPSHOTS) {
        safe_free((void**)&g_snapshots[0].state_data);
        for (int i = 0; i < MAX_SNAPSHOTS - 1; i++) {
            g_snapshots[i] = g_snapshots[i + 1];
        }
        g_snapshot_count = MAX_SNAPSHOTS - 1;
    }

    int idx = g_snapshot_count;
    g_snapshots[idx].state_size = state->config.state_size;
    g_snapshots[idx].state_data = (float*)safe_malloc(
        state->config.state_size * sizeof(float));
    if (!g_snapshots[idx].state_data) {
        mutex_unlock(g_snapshot_mutex);
        return -1;
    }

    memcpy(g_snapshots[idx].state_data, state->current_state,
           state->config.state_size * sizeof(float));
    g_snapshots[idx].step = state->history_index;
    if (label) {
        snprintf(g_snapshots[idx].label, sizeof(g_snapshots[idx].label), "%s", label);
    } else {
        snprintf(g_snapshots[idx].label, sizeof(g_snapshots[idx].label),
                "snapshot_%d", idx);
    }
    /* Z-P3修复: 初始化task_id（由调用方通过snapshot_save_ex设置） */
    g_snapshots[idx].task_id[0] = '\0';
    g_snapshot_count++;
    mutex_unlock(g_snapshot_mutex);
    return 0;
}

/* Z-P3修复: 保存状态快照（带任务上下文标识）
 * 使状态快照与特定任务关联，方便多任务训练中切换和恢复 */
int network_state_snapshot_save_ex(NetworkState* state, const char* label,
                                    const char* task_id) {
    int ret = network_state_snapshot_save(state, label);
    if (ret == 0 && task_id && g_snapshot_count > 0) {
        mutex_lock(g_snapshot_mutex);
        int idx = g_snapshot_count - 1;
        snprintf(g_snapshots[idx].task_id, sizeof(g_snapshots[idx].task_id), "%s", task_id);
        mutex_unlock(g_snapshot_mutex);
    }
    return ret;
}

const char* network_state_snapshot_task_id(int idx) {
    if (!g_snapshot_mutex) g_snapshot_mutex = mutex_create();
    mutex_lock(g_snapshot_mutex);
    const char* result = NULL;
    if (idx >= 0 && idx < g_snapshot_count) {
        result = g_snapshots[idx].task_id[0] ? g_snapshots[idx].task_id : NULL;
    }
    mutex_unlock(g_snapshot_mutex);
    return result;
}

int network_state_snapshot_restore(NetworkState* state, int snapshot_idx) {
    if (!state || !state->current_state) return -1;
    if (!g_snapshot_mutex) g_snapshot_mutex = mutex_create();
    mutex_lock(g_snapshot_mutex);
    if (snapshot_idx < 0 || snapshot_idx >= g_snapshot_count) {
        mutex_unlock(g_snapshot_mutex);
        return -1;
    }
    if (g_snapshots[snapshot_idx].state_size != state->config.state_size) {
        mutex_unlock(g_snapshot_mutex);
        return -1;
    }

    memcpy(state->current_state, g_snapshots[snapshot_idx].state_data,
           state->config.state_size * sizeof(float));
    memcpy(state->previous_state, g_snapshots[snapshot_idx].state_data,
           state->config.state_size * sizeof(float));

    state->history_index = g_snapshots[snapshot_idx].step;
    state->stability = 1.0f;
    state->convergence = 0.0f;
    state->change_rate = 0.0f;
    mutex_unlock(g_snapshot_mutex);
    return 0;
}

int network_state_snapshot_count(void) {
    int count;
    if (!g_snapshot_mutex) g_snapshot_mutex = mutex_create();
    mutex_lock(g_snapshot_mutex);
    count = g_snapshot_count;
    mutex_unlock(g_snapshot_mutex);
    return count;
}

const char* network_state_snapshot_label(int idx) {
    const char* result;
    if (!g_snapshot_mutex) g_snapshot_mutex = mutex_create();
    mutex_lock(g_snapshot_mutex);
    if (idx < 0 || idx >= g_snapshot_count) {
        mutex_unlock(g_snapshot_mutex);
        return NULL;
    }
    result = g_snapshots[idx].label;
    mutex_unlock(g_snapshot_mutex);
    return result;
}

void network_state_snapshot_clear(void) {
    if (!g_snapshot_mutex) g_snapshot_mutex = mutex_create();
    mutex_lock(g_snapshot_mutex);
    for (int i = 0; i < g_snapshot_count; i++) {
        safe_free((void**)&g_snapshots[i].state_data);
    }
    g_snapshot_count = 0;
    mutex_unlock(g_snapshot_mutex);
}

/**
 * @brief 设置拉普拉斯频域稳定性指标
 */
void network_state_set_laplace_metrics(NetworkState* state,
                                        float stability_score,
                                        float recommended_cutoff,
                                        float frequency_bandwidth) {
    if (!state) return;

    state->laplace_stability_score = stability_score;
    state->laplace_recommended_cutoff = recommended_cutoff;
    state->laplace_frequency_bandwidth = frequency_bandwidth;
}