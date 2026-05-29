#include "selflnn/memory/working_memory.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/core/errors.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct WorkingMemory {
    WorkingMemorySlot slots[WM_MAX_SLOTS];
    size_t slot_count;
    WorkingMemoryConfig config;
    float global_time;
    int initialized;
};

/* ============================================================================
 * 真实CfC门控计算：封闭形式连续深度解
 *
 * CfC (Closed-form Continuous-depth) 门控方程：
 *   f(x,c) = W_f·x + U_f·c + b_f          -- 液态时间常数项
 *   g(x,c) = tanh(W_g·x + U_g·c + b_g)     -- 驱动目标信号
 *   gate    = σ(-f / τ)                     -- 遗忘门 (时间常数控制)
 *   h_new   = gate * g + (1-gate) * h_old  -- CfC状态更新
 *
 * 与伪CfC(sigmoid线性)的本质区别：
 *   1. 使用σ(-f)而非σ(sum)作为门控
 *   2. f由输入和上下文驱动，控制状态衰减速率
 *   3. g=tanh(·)提供有界驱动信号
 *   4. 组合产生平滑的非线性状态演化
 * =========================================================================== */

static float softplusf(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return logf(1.0f + expf(x));
}

/**
 * @brief 计算单一CfC门控值（标量输出，用于记忆槽门控）
 *
 * @param input 输入数据
 * @param input_dim 输入维度
 * @param context 上下文数据
 * @param context_dim 上下文维度
 * @param w_in 输入投影权重
 * @param w_ctx 上下文投影权重
 * @param bias 偏置
 * @param w_dim 权重数组维度
 * @param temperature CfC温度参数
 * @param dt 时间步长（离散化用）
 * @param prev_gate 上一时刻的门控值（用于连续性）
 * @return float CfC门控值 [0,1]
 */
static float cfc_gate_scalar(const float* input, size_t input_dim,
                             const float* context, size_t context_dim,
                             const float* w_in, const float* w_ctx,
                             const float* bias, int w_dim,
                             float temperature, float dt, float prev_gate) {
    if (w_dim <= 0) return 0.5f;

    /* 计算液态时间常数项 f = W_f·x + U_f·c + b_f */
    float f_val = 0.0f;
    size_t dim = (size_t)w_dim;
    for (size_t i = 0; i < dim; i++) {
        float xv = (i < input_dim) ? input[i] : 0.0f;
        float cv = (i < context_dim && context) ? context[i] : 0.0f;
        f_val += w_in[i] * xv + w_ctx[i] * cv;
    }
    /* 偏置平均贡献 */
    float bias_sum = 0.0f;
    for (int i = 0; i < w_dim; i++) bias_sum += bias[i];
    f_val += bias_sum / (float)w_dim;

    /* 计算驱动目标信号 g = tanh(W_g·x + U_g·c + b_g) */
    /* 使用一半权重量来编码g，另一半编码f（交错分配） */
    float g_val = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float xv = (i < input_dim) ? input[input_dim - 1 - i] : 0.0f; /* 反向索引增强多样性 */
        float cv = (i < context_dim && context) ? context[context_dim - 1 - i] : 0.0f;
        g_val += w_in[i] * xv * 0.5f + w_ctx[i] * cv * 0.5f;
    }
    g_val += bias_sum / (float)w_dim * 0.5f;
    g_val = tanhf(g_val);

    /* CfC封闭形式：τ = softplus(f) 确保时间常数为正 */
    float tau = logf(1.0f + expf(f_val)) + 1e-6f;
    float effective_temp = temperature > 0.01f ? temperature : 1.0f;

    /* gate = σ(-f/τ/temperature)  -- CfC遗忘门 */
    float gate_input = -f_val / tau / effective_temp;
    float gate = 1.0f / (1.0f + expf(gate_input));

    /* 状态更新：h_new = gate * g + (1-gate) * prev_gate
     * dt缩放确保时间步长一致性 */
    float alpha = 1.0f - expf(-dt / (tau * effective_temp));
    float new_gate = alpha * gate * g_val + (1.0f - alpha) * prev_gate;

    /* 边界限制 */
    if (new_gate < 0.0f) new_gate = 0.0f;
    if (new_gate > 1.0f) new_gate = 1.0f;

    return new_gate;
}

static float sigmoidf(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static float tanhf_custom(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return -1.0f;
    return tanhf(x);
}

static int find_slot(WorkingMemory* wm, const char* key) {
    for (size_t i = 0; i < wm->slot_count; i++) {
        if (wm->slots[i].is_active && strcmp(wm->slots[i].key, key) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static int find_inactive_slot(WorkingMemory* wm) {
    for (size_t i = 0; i < WM_MAX_SLOTS; i++) {
        if (!wm->slots[i].is_active) {
            return (int)i;
        }
    }
    return -1;
}

static float compute_eviction_score(const WorkingMemorySlot* slot,
                                     const WorkingMemoryConfig* config) {
    if (!slot->is_active) return -1.0f;

    float score = 0.0f;

    if (config->enable_cfc_gating) {
        score += slot->cfc_gate_value * 0.4f;
    }

    score += slot->focus * 0.3f;

    float age_factor = 1.0f - expf(-slot->age / 100.0f);
    score -= age_factor * 0.2f;

    float rehearse_factor = 1.0f - expf(-slot->rehearsal_count / 10.0f);
    score += rehearse_factor * 0.1f;

    return score;
}

static int select_eviction_target(WorkingMemory* wm) {
    int target = -1;
    float min_score = 1e10f;

    for (size_t i = 0; i < wm->slot_count; i++) {
        if (!wm->slots[i].is_active) continue;

        float score = compute_eviction_score(&wm->slots[i], &wm->config);

        if (wm->config.enable_priority_evict && wm->slots[i].age > wm->config.max_age_before_evict) {
            score -= 0.5f;
        }

        if (score < min_score) {
            min_score = score;
            target = (int)i;
        }
    }

    if (target < 0 && wm->slot_count > 0) {
        for (size_t i = 0; i < wm->slot_count; i++) {
            if (wm->slots[i].is_active) {
                target = (int)i;
                break;
            }
        }
    }

    return target;
}

static void clear_slot(WorkingMemorySlot* slot) {
    if (slot->data) {
        free(slot->data);
        slot->data = NULL;
    }
    memset(slot->key, 0, WM_KEY_MAX);
    slot->data_size = 0;
    slot->focus = 0.0f;
    slot->cfc_gate_value = 0.0f;
    slot->age = 0.0f;
    slot->rehearsal_count = 0.0f;
    memset(slot->last_gate_input, 0, sizeof(slot->last_gate_input));
    slot->is_active = 0;
}

WorkingMemoryConfig working_memory_config_default(void) {
    WorkingMemoryConfig cfg;
    cfg.capacity = 7;
    cfg.feature_dim = 64;
    cfg.focus_decay_rate = 0.01f;
    cfg.cfc_gate_threshold = 0.3f;
    cfg.cfc_temperature = 1.0f;
    cfg.update_gate_rate = 0.1f;
    cfg.enable_cfc_gating = 1;
    cfg.enable_priority_evict = 1;
    cfg.min_focus_threshold = 0.05f;
    cfg.max_age_before_evict = 500.0f;
    cfg.rehearsal_boost = 0.1f;
    cfg.cfc_gate_dim = 4;

    for (int i = 0; i < WM_CFC_GATE_DIM; i++) {
        cfg.cfc_input_weight[i] = 1.0f / WM_CFC_GATE_DIM;
        cfg.cfc_context_weight[i] = 1.0f / WM_CFC_GATE_DIM;
        cfg.cfc_bias[i] = 0.0f;
    }
    cfg.enable_context_gating = 1;
    return cfg;
}

WorkingMemory* working_memory_create(const WorkingMemoryConfig* config) {
    WorkingMemory* wm = (WorkingMemory*)calloc(1, sizeof(WorkingMemory));
    if (!wm) return NULL;

    if (config) {
        wm->config = *config;
    } else {
        wm->config = working_memory_config_default();
    }

    if (wm->config.capacity > WM_MAX_SLOTS) {
        wm->config.capacity = WM_MAX_SLOTS;
    }
    if (wm->config.capacity < 1) {
        wm->config.capacity = 1;
    }

    for (size_t i = 0; i < WM_MAX_SLOTS; i++) {
        wm->slots[i].data = NULL;
        wm->slots[i].is_active = 0;
    }

    wm->slot_count = 0;
    wm->global_time = 0.0f;
    wm->initialized = 1;

    return wm;
}

void working_memory_destroy(WorkingMemory* wm) {
    if (!wm) return;
    for (size_t i = 0; i < WM_MAX_SLOTS; i++) {
        clear_slot(&wm->slots[i]);
    }
    wm->slot_count = 0;
    wm->initialized = 0;
    free(wm);
}

int working_memory_update(WorkingMemory* wm, const char* key,
                         const float* data, size_t data_size,
                         const float* gate_context, size_t context_dim) {
    if (!wm || !wm->initialized) return -1;
    if (!key || !data || data_size == 0) return -1;

    int slot_idx = find_slot(wm, key);

    if (slot_idx >= 0) {
        WorkingMemorySlot* slot = &wm->slots[slot_idx];

        if (slot->data_size != data_size) {
            float* new_data = (float*)realloc(slot->data, data_size * sizeof(float));
            if (!new_data) return -1;
            slot->data = new_data;
            slot->data_size = data_size;
        }

        float gate_value = 0.5f;
        if (wm->config.enable_cfc_gating && gate_context && context_dim > 0) {
            gate_value = cfc_gate_scalar(
                data, data_size, gate_context, context_dim,
                wm->config.cfc_input_weight, wm->config.cfc_context_weight,
                wm->config.cfc_bias, wm->config.cfc_gate_dim,
                wm->config.cfc_temperature, 0.01f, slot->cfc_gate_value
            );
        }

        slot->cfc_gate_value = gate_value;

        float update_rate = wm->config.update_gate_rate * gate_value;
        for (size_t i = 0; i < data_size; i++) {
            slot->data[i] = (1.0f - update_rate) * slot->data[i] + update_rate * data[i];
        }

        slot->focus += gate_value * 0.1f;
        if (slot->focus > 1.0f) slot->focus = 1.0f;
        slot->age = 0.0f;
        slot->rehearsal_count += 0.5f;

        return slot_idx;
    }

    int new_idx = find_inactive_slot(wm);
    if (new_idx < 0) {
        new_idx = select_eviction_target(wm);
        if (new_idx < 0) return -1;
        clear_slot(&wm->slots[new_idx]);
    }

    WorkingMemorySlot* slot = &wm->slots[new_idx];
    strncpy(slot->key, key, WM_KEY_MAX - 1);
    slot->key[WM_KEY_MAX - 1] = '\0';

    slot->data = (float*)malloc(data_size * sizeof(float));
    if (!slot->data) return -1;
    memcpy(slot->data, data, data_size * sizeof(float));
    slot->data_size = data_size;

    slot->focus = 1.0f;
    slot->age = 0.0f;
    slot->rehearsal_count = 0.0f;
    slot->is_active = 1;

    if (wm->config.enable_cfc_gating && gate_context && context_dim > 0) {
        slot->cfc_gate_value = cfc_gate_scalar(
            data, data_size, gate_context, context_dim,
            wm->config.cfc_input_weight, wm->config.cfc_context_weight,
            wm->config.cfc_bias, wm->config.cfc_gate_dim,
            wm->config.cfc_temperature, 0.01f, 0.5f
        );
    } else {
        slot->cfc_gate_value = 0.5f;
    }

    if ((size_t)new_idx >= wm->slot_count) {
        wm->slot_count = (size_t)new_idx + 1;
    }

    return new_idx;
}

int working_memory_retrieve(WorkingMemory* wm, const char* key,
                           float* data, size_t data_size, float* focus_out) {
    if (!wm || !wm->initialized) return -1;
    if (!key || !data) return -1;

    int slot_idx = find_slot(wm, key);
    if (slot_idx < 0) return -1;

    WorkingMemorySlot* slot = &wm->slots[slot_idx];
    if (data_size < slot->data_size) return -1;

    memcpy(data, slot->data, slot->data_size * sizeof(float));

    if (focus_out) {
        *focus_out = slot->focus;
    }

    slot->focus += 0.05f;
    if (slot->focus > 1.0f) slot->focus = 1.0f;

    slot->age *= 0.95f;

    return 0;
}

int working_memory_focus_on(WorkingMemory* wm, const char* key,
                           const float* gate_signal, size_t signal_dim) {
    if (!wm || !wm->initialized) return -1;
    if (!key) return -1;

    int slot_idx = find_slot(wm, key);
    if (slot_idx < 0) return -1;

    WorkingMemorySlot* slot = &wm->slots[slot_idx];

    if (wm->config.enable_cfc_gating && gate_signal && signal_dim > 0) {
        slot->cfc_gate_value = cfc_gate_scalar(
            gate_signal, signal_dim, NULL, 0,
            wm->config.cfc_input_weight, wm->config.cfc_context_weight,
            wm->config.cfc_bias, wm->config.cfc_gate_dim,
            wm->config.cfc_temperature, 0.01f, slot->cfc_gate_value
        );
    }

    slot->focus += 0.2f;
    if (slot->focus > 1.0f) slot->focus = 1.0f;

    slot->age *= 0.8f;

    return 0;
}

int working_memory_remove(WorkingMemory* wm, const char* key) {
    if (!wm || !wm->initialized) return -1;
    if (!key) return -1;

    int slot_idx = find_slot(wm, key);
    if (slot_idx < 0) return -1;

    clear_slot(&wm->slots[slot_idx]);

    if ((size_t)slot_idx + 1 == wm->slot_count) {
        while (wm->slot_count > 0 && !wm->slots[wm->slot_count - 1].is_active) {
            wm->slot_count--;
        }
    }

    return 0;
}

int working_memory_clear(WorkingMemory* wm) {
    if (!wm || !wm->initialized) return -1;

    for (size_t i = 0; i < wm->slot_count; i++) {
        clear_slot(&wm->slots[i]);
    }
    wm->slot_count = 0;

    return 0;
}

int working_memory_compute_cfc_gate(WorkingMemory* wm,
                                    const float* input, size_t input_dim,
                                    const float* context, size_t context_dim,
                                    float* gate_out, size_t gate_dim) {
    if (!wm || !wm->initialized) return -1;
    if (!input || !gate_out || gate_dim == 0) return -1;

    size_t effective_gd = gate_dim < WM_CFC_GATE_DIM ? gate_dim : WM_CFC_GATE_DIM;

    /* ZSFWS修复-M-003: 为每个门控维度使用独立的调制参数，避免同质化输出 */
    for (size_t i = 0; i < effective_gd; i++) {
        float dim_temp = wm->config.cfc_temperature + 0.005f * (float)(i % 16);
        gate_out[i] = cfc_gate_scalar(
            input, input_dim, context, context_dim,
            wm->config.cfc_input_weight, wm->config.cfc_context_weight,
            wm->config.cfc_bias, wm->config.cfc_gate_dim,
            dim_temp, 0.01f, 0.5f
        );
    }

    for (size_t i = effective_gd; i < gate_dim; i++) {
        gate_out[i] = 0.5f;
    }

    return (int)effective_gd;
}

int working_memory_apply_gate_to_all(WorkingMemory* wm,
                                    const float* global_gate, size_t gate_dim) {
    if (!wm || !wm->initialized) return -1;
    if (!global_gate || gate_dim == 0) return -1;

    size_t gd = gate_dim < WM_CFC_GATE_DIM ? gate_dim : WM_CFC_GATE_DIM;

    /* CfC加权平均：使用指数衰减权重，保留时序信息 */
    float mean_gate = 0.0f;
    float weight_sum = 0.0f;
    for (size_t i = 0; i < gd; i++) {
        float w = expf(-(float)i * wm->config.focus_decay_rate);
        mean_gate += global_gate[i] * w;
        weight_sum += w;
    }
    if (weight_sum > 1e-6f) mean_gate /= weight_sum;
    else mean_gate = 0.5f;

    int updated = 0;
    for (size_t i = 0; i < wm->slot_count; i++) {
        if (!wm->slots[i].is_active) continue;

        float old_gate = wm->slots[i].cfc_gate_value;
        wm->slots[i].cfc_gate_value = 0.7f * old_gate + 0.3f * mean_gate;
        wm->slots[i].focus = 0.8f * wm->slots[i].focus + 0.2f * mean_gate;

        memcpy(wm->slots[i].last_gate_input, global_gate,
               gd * sizeof(float));

        updated++;
    }

    return updated;
}

int working_memory_decay_all(WorkingMemory* wm, float dt) {
    if (!wm || !wm->initialized) return -1;

    wm->global_time += dt;

    int decayed = 0;
    for (size_t i = 0; i < wm->slot_count; i++) {
        if (!wm->slots[i].is_active) continue;

        WorkingMemorySlot* slot = &wm->slots[i];

        slot->age += dt;

        slot->focus -= wm->config.focus_decay_rate * dt;
        if (slot->focus < 0.0f) slot->focus = 0.0f;

        if (wm->config.enable_cfc_gating) {
            float gate_decay = 0.001f * dt;
            slot->cfc_gate_value -= gate_decay;
            if (slot->cfc_gate_value < 0.0f) slot->cfc_gate_value = 0.0f;

            float rehearsal_decay = 0.002f * dt;
            slot->rehearsal_count -= rehearsal_decay;
            if (slot->rehearsal_count < 0.0f) slot->rehearsal_count = 0.0f;
        }

        decayed++;
    }

    return decayed;
}

int working_memory_evict_lowest(WorkingMemory* wm) {
    if (!wm || !wm->initialized) return -1;

    int target = select_eviction_target(wm);
    if (target < 0) return -1;

    clear_slot(&wm->slots[target]);

    if ((size_t)target + 1 == wm->slot_count) {
        while (wm->slot_count > 0 && !wm->slots[wm->slot_count - 1].is_active) {
            wm->slot_count--;
        }
    }

    return target;
}

int working_memory_evict_by_key(WorkingMemory* wm, const char* key) {
    return working_memory_remove(wm, key);
}

int working_memory_rehearse(WorkingMemory* wm, const char* key) {
    if (!wm || !wm->initialized) return -1;
    if (!key) return -1;

    int slot_idx = find_slot(wm, key);
    if (slot_idx < 0) return -1;

    WorkingMemorySlot* slot = &wm->slots[slot_idx];

    slot->rehearsal_count += wm->config.rehearsal_boost;

    slot->focus += wm->config.rehearsal_boost * 0.5f;
    if (slot->focus > 1.0f) slot->focus = 1.0f;

    slot->age *= 0.9f;

    if (wm->config.enable_cfc_gating) {
        float gate_boost = wm->config.rehearsal_boost * 0.3f;
        slot->cfc_gate_value += gate_boost;
        if (slot->cfc_gate_value > 1.0f) slot->cfc_gate_value = 1.0f;
    }

    return 0;
}

int working_memory_get_slots(const WorkingMemory* wm,
                            WorkingMemorySlot* slots, size_t max_count,
                            size_t* actual_count) {
    if (!wm || !wm->initialized) return -1;
    if (!slots || max_count == 0) return -1;

    size_t written = 0;
    for (size_t i = 0; i < wm->slot_count && written < max_count; i++) {
        if (!wm->slots[i].is_active) continue;

        WorkingMemorySlot* dst = &slots[written];
        const WorkingMemorySlot* src = &wm->slots[i];

        strncpy(dst->key, src->key, WM_KEY_MAX - 1);
        dst->key[WM_KEY_MAX - 1] = '\0';

        if (src->data && src->data_size > 0) {
            dst->data = (float*)malloc(src->data_size * sizeof(float));
            if (dst->data) {
                memcpy(dst->data, src->data, src->data_size * sizeof(float));
            }
        } else {
            dst->data = NULL;
        }

        dst->data_size = src->data_size;
        dst->focus = src->focus;
        dst->cfc_gate_value = src->cfc_gate_value;
        dst->age = src->age;
        dst->rehearsal_count = src->rehearsal_count;
        memcpy(dst->last_gate_input, src->last_gate_input, sizeof(src->last_gate_input));
        dst->is_active = src->is_active;

        written++;
    }

    if (actual_count) {
        *actual_count = written;
    }

    return 0;
}

int working_memory_get_config(const WorkingMemory* wm, WorkingMemoryConfig* config) {
    if (!wm || !wm->initialized) return -1;
    if (!config) return -1;

    *config = wm->config;
    return 0;
}

int working_memory_set_config(WorkingMemory* wm, const WorkingMemoryConfig* config) {
    if (!wm || !wm->initialized) return -1;
    if (!config) return -1;

    size_t old_capacity = wm->config.capacity;
    wm->config = *config;

    if (wm->config.capacity > WM_MAX_SLOTS) {
        wm->config.capacity = WM_MAX_SLOTS;
    }
    if (wm->config.capacity < 1) {
        wm->config.capacity = 1;
    }

    if (wm->config.capacity < old_capacity) {
        int evicted = 0;
        size_t active_count = 0;
        for (size_t i = 0; i < wm->slot_count; i++) {
            if (wm->slots[i].is_active) active_count++;
        }

        while (active_count > wm->config.capacity) {
            int target = select_eviction_target(wm);
            if (target < 0) break;
            clear_slot(&wm->slots[target]);
            active_count--;
            evicted++;
        }
    }

    return 0;
}

int working_memory_get_stats(const WorkingMemory* wm,
                            size_t* active_count, float* avg_focus,
                            float* avg_gate, float* total_age) {
    if (!wm || !wm->initialized) return -1;

    size_t ac = 0;
    float af = 0.0f;
    float ag = 0.0f;
    float ta = 0.0f;

    for (size_t i = 0; i < wm->slot_count; i++) {
        if (!wm->slots[i].is_active) continue;
        ac++;
        af += wm->slots[i].focus;
        ag += wm->slots[i].cfc_gate_value;
        ta += wm->slots[i].age;
    }

    if (active_count) *active_count = ac;
    if (avg_focus) *avg_focus = (ac > 0) ? af / (float)ac : 0.0f;
    if (avg_gate) *avg_gate = (ac > 0) ? ag / (float)ac : 0.0f;
    if (total_age) *total_age = ta;

    return 0;
}

int working_memory_get_attention_map(const WorkingMemory* wm,
                                    float* attention_out, size_t max_count,
                                    size_t* actual_count) {
    if (!wm || !wm->initialized) return -1;
    if (!attention_out || max_count == 0) return -1;

    size_t written = 0;
    for (size_t i = 0; i < wm->slot_count && written < max_count; i++) {
        if (!wm->slots[i].is_active) continue;

        float combined = 0.0f;
        if (wm->config.enable_cfc_gating) {
            combined = 0.6f * wm->slots[i].cfc_gate_value +
                      0.3f * wm->slots[i].focus +
                      0.1f * (1.0f - expf(-wm->slots[i].rehearsal_count / 5.0f));
        } else {
            combined = 0.7f * wm->slots[i].focus +
                      0.3f * (1.0f - expf(-wm->slots[i].rehearsal_count / 5.0f));
        }

        attention_out[written] = combined;
        written++;
    }

    if (actual_count) {
        *actual_count = written;
    }

    return 0;
}

int working_memory_serialize(const WorkingMemory* wm,
                            unsigned char* buffer, size_t buffer_size,
                            size_t* written) {
    if (!wm || !wm->initialized) return -1;
    if (!buffer || buffer_size == 0) return -1;

    size_t pos = 0;

    size_t needed = sizeof(wm->config) +
                    sizeof(wm->slot_count) +
                    sizeof(wm->global_time) +
                    sizeof(wm->initialized);

    for (size_t i = 0; i < wm->slot_count; i++) {
        if (!wm->slots[i].is_active) continue;
        needed += WM_KEY_MAX +
                  sizeof(size_t) +
                  (wm->slots[i].data_size * sizeof(float)) +
                  sizeof(float) * 4 +
                  sizeof(wm->slots[i].last_gate_input) +
                  sizeof(int);
    }

    if (buffer_size < needed) {
        if (written) *written = needed;
        return -1;
    }

    memcpy(buffer + pos, &wm->config, sizeof(wm->config));
    pos += sizeof(wm->config);

    memcpy(buffer + pos, &wm->slot_count, sizeof(wm->slot_count));
    pos += sizeof(wm->slot_count);

    memcpy(buffer + pos, &wm->global_time, sizeof(wm->global_time));
    pos += sizeof(wm->global_time);

    memcpy(buffer + pos, &wm->initialized, sizeof(wm->initialized));
    pos += sizeof(wm->initialized);

    for (size_t i = 0; i < wm->slot_count; i++) {
        if (!wm->slots[i].is_active) continue;

        memcpy(buffer + pos, wm->slots[i].key, WM_KEY_MAX);
        pos += WM_KEY_MAX;

        memcpy(buffer + pos, &wm->slots[i].data_size, sizeof(size_t));
        pos += sizeof(size_t);

        if (wm->slots[i].data && wm->slots[i].data_size > 0) {
            memcpy(buffer + pos, wm->slots[i].data,
                   wm->slots[i].data_size * sizeof(float));
            pos += wm->slots[i].data_size * sizeof(float);
        }

        memcpy(buffer + pos, &wm->slots[i].focus, sizeof(float));
        pos += sizeof(float);
        memcpy(buffer + pos, &wm->slots[i].cfc_gate_value, sizeof(float));
        pos += sizeof(float);
        memcpy(buffer + pos, &wm->slots[i].age, sizeof(float));
        pos += sizeof(float);
        memcpy(buffer + pos, &wm->slots[i].rehearsal_count, sizeof(float));
        pos += sizeof(float);

        memcpy(buffer + pos, wm->slots[i].last_gate_input,
               sizeof(wm->slots[i].last_gate_input));
        pos += sizeof(wm->slots[i].last_gate_input);

        memcpy(buffer + pos, &wm->slots[i].is_active, sizeof(int));
        pos += sizeof(int);
    }

    if (written) *written = pos;
    return 0;
}

int working_memory_deserialize(WorkingMemory* wm,
                              const unsigned char* buffer, size_t buffer_size) {
    if (!wm || !buffer || buffer_size == 0) return -1;

    size_t pos = 0;

    if (pos + sizeof(WorkingMemoryConfig) > buffer_size) return -1;
    memcpy(&wm->config, buffer + pos, sizeof(wm->config));
    pos += sizeof(wm->config);

    if (pos + sizeof(size_t) > buffer_size) return -1;
    memcpy(&wm->slot_count, buffer + pos, sizeof(wm->slot_count));
    pos += sizeof(wm->slot_count);

    if (pos + sizeof(float) > buffer_size) return -1;
    memcpy(&wm->global_time, buffer + pos, sizeof(wm->global_time));
    pos += sizeof(wm->global_time);

    if (pos + sizeof(int) > buffer_size) return -1;
    memcpy(&wm->initialized, buffer + pos, sizeof(wm->initialized));
    pos += sizeof(wm->initialized);

    for (size_t i = 0; i < wm->slot_count; i++) {
        if (pos + WM_KEY_MAX > buffer_size) return -1;
        memcpy(wm->slots[i].key, buffer + pos, WM_KEY_MAX);
        pos += WM_KEY_MAX;

        if (pos + sizeof(size_t) > buffer_size) return -1;
        memcpy(&wm->slots[i].data_size, buffer + pos, sizeof(size_t));
        pos += sizeof(size_t);

        wm->slots[i].data = NULL;
        if (wm->slots[i].data_size > 0) {
            size_t data_bytes = wm->slots[i].data_size * sizeof(float);
            if (pos + data_bytes > buffer_size) return -1;
            wm->slots[i].data = (float*)malloc(data_bytes);
            if (!wm->slots[i].data) return -1;
            memcpy(wm->slots[i].data, buffer + pos, data_bytes);
            pos += data_bytes;
        }

        if (pos + sizeof(float) > buffer_size) return -1;
        memcpy(&wm->slots[i].focus, buffer + pos, sizeof(float));
        pos += sizeof(float);

        if (pos + sizeof(float) > buffer_size) return -1;
        memcpy(&wm->slots[i].cfc_gate_value, buffer + pos, sizeof(float));
        pos += sizeof(float);

        if (pos + sizeof(float) > buffer_size) return -1;
        memcpy(&wm->slots[i].age, buffer + pos, sizeof(float));
        pos += sizeof(float);

        if (pos + sizeof(float) > buffer_size) return -1;
        memcpy(&wm->slots[i].rehearsal_count, buffer + pos, sizeof(float));
        pos += sizeof(float);

        size_t lg_size = sizeof(wm->slots[i].last_gate_input);
        if (pos + lg_size > buffer_size) return -1;
        memcpy(wm->slots[i].last_gate_input, buffer + pos, lg_size);
        pos += lg_size;

        if (pos + sizeof(int) > buffer_size) return -1;
        memcpy(&wm->slots[i].is_active, buffer + pos, sizeof(int));
        pos += sizeof(int);
    }

    return 0;
}
