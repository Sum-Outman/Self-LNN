/**
 * @file dialogue_deep.c
 * @brief 对话系统深度增强实现
 *
 * A02.4.1 对话管理深度实现：
 *   - 对话状态追踪（DST）：槽位-信念状态维护、语义槽位填充、时间衰减
 *   - 对话策略学习：状态驱动策略决策、TD学习更新
 *   - 多轮对话推理：历史注意力加权推理、话题切换检测
 *
 * A02.4.2 对话生成深度实现：
 *   - 基于液态状态的对话生成（自回归文本生成）
 *   - 条件对话生成（条件信号注入）
 *   - 知识增强对话生成（知识嵌入作为偏置信号）
 */

#include "selflnn/multimodal/dialogue.h"
#include "selflnn/core/cfc_cell.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/core/errors.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <ctype.h>

/* 编译期验证：DialogueProcessor结构体可见性测试 */
static int _dialogue_deep_verify(void) {
    DialogueProcessor p;
    (void)sizeof(p.config);
    (void)sizeof(p.deep_initialized);
    return 0;
}
/* 编译期验证2：指针成员访问测试 */
static int _dialogue_deep_verify2(void) {
    DialogueProcessor* p = 0;
    (void)sizeof(p->config);
    (void)sizeof(p->deep_initialized);
    (void)sizeof(p->deep_belief);
    (void)sizeof(p->deep_policy);
    (void)sizeof(p->deep_reasoner);
    (void)sizeof(p->generator);
    return 0;
}

/* ============================================================================
 * 内部工具函数
 * ============================================================================ */

/* K-012修复：使用加密安全随机数替代rand() */
#include "selflnn/utils/secure_random.h"
static float rand_float(void)
{
    return secure_random_float();
}

static float clip(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static int str_contains(const char* text, const char* keyword)
{
    if (!text || !keyword) return 0;
    return strstr(text, keyword) != NULL;
}

/* ============================================================================
 * 权重初始化工具函数：Xavier/Glorot 和 He/Kaiming 初始化
 *
 * Xavier均匀分布初始化：W ~ U(-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out)))
 * He正态分布初始化：   W ~ N(0, sqrt(2/fan_in))
 * 解决深层网络梯度消失/爆炸问题，加速训练收敛。
 * ============================================================================ */

/* Xavier/Glorot均匀分布初始化 */
static float dd_xavier_uniform_scale(size_t fan_in, size_t fan_out)
{
    /* 计算Xavier均匀分布的边界：sqrt(6/(fan_in+fan_out)) */
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    return limit;
}

/* He/Kaiming正态分布初始化（使用Box-Muller变换生成近似正态分布） */
static float dd_he_normal_scale(size_t fan_in)
{
    /* He初始化标准差：sqrt(2/fan_in) */
    return sqrtf(2.0f / (float)fan_in);
}

/* 从均匀分布生成近似正态分布的随机数（Box-Muller变换简化版） */
static float dd_normal_rand(void)
{
    /* 使用两个均匀随机数通过Box-Muller变换生成标准正态分布 */
    float u1 = rand_float();
    float u2 = rand_float();
    /* 避免log(0) */
    if (u1 < 1e-6f) u1 = 1e-6f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
}

/* Xavier均匀分布初始化：返回[-limit, +limit]范围内的随机值 */
static float dd_xavier_init(size_t fan_in, size_t fan_out)
{
    float limit = dd_xavier_uniform_scale(fan_in, fan_out);
    /* 映射到[-limit, +limit] */
    return (rand_float() * 2.0f - 1.0f) * limit;
}

/* He正态分布初始化：返回N(0, sqrt(2/fan_in))分布的随机值 */
static float dd_he_init(size_t fan_in)
{
    float std = dd_he_normal_scale(fan_in);
    return dd_normal_rand() * std;
}

/* ============================================================================
 * A02.4.1 对话状态追踪（DST）
 * ============================================================================ */

DialogueBeliefState* dialogue_belief_state_create(int num_slots, size_t state_size)
{
    if (num_slots <= 0 || num_slots > DST_MAX_SLOTS) return NULL;
    if (state_size == 0) state_size = 64;

    DialogueBeliefState* belief = (DialogueBeliefState*)safe_malloc(sizeof(DialogueBeliefState));
    if (!belief) return NULL;
    memset(belief, 0, sizeof(DialogueBeliefState));

    belief->num_slots = num_slots;
    belief->num_filled_slots = 0;
    belief->belief_entropy = 0.0f;
    belief->state_coherence = 1.0f;
    belief->belief_state_size = state_size;
    belief->history_count = 0;
    belief->belief_change_rate = 0.0f;

    for (int i = 0; i < num_slots; i++) {
        DSTSlot* slot = &belief->slots[i];
        slot->slot_type = DST_SLOT_CATEGORICAL;
        slot->num_possible_values = 0;
        slot->confidence = 0.0f;
        slot->is_filled = 0;
        slot->turn_filled = -1;
        slot->decay_factor = 0.9f;
        memset(slot->possible_values, 0, sizeof(slot->possible_values));
        memset(slot->value_weights, 0, sizeof(slot->value_weights));
    }

    belief->belief_state_vector = (float*)safe_malloc(state_size * sizeof(float));
    if (!belief->belief_state_vector) {
        safe_free((void**)&belief);
        return NULL;
    }
    for (size_t i = 0; i < state_size; i++) {
        belief->belief_state_vector[i] = 0.0f;
    }

    return belief;
}

void dialogue_belief_state_free(DialogueBeliefState* belief)
{
    if (!belief) return;
    for (int i = 0; i < belief->history_count; i++) {
        if (belief->belief_state_history[i]) {
            safe_free((void**)&belief->belief_state_history[i]);
        }
    }
    if (belief->belief_state_vector) {
        safe_free((void**)&belief->belief_state_vector);
    }
    safe_free((void**)&belief);
}

int dialogue_belief_define_slot(DialogueBeliefState* belief, int slot_index,
                                 const char* name, DSTSlotType slot_type,
                                 const char* const* possible_values, int num_values)
{
    if (!belief || slot_index < 0 || slot_index >= belief->num_slots) return -1;
    if (!name) return -1;

    DSTSlot* slot = &belief->slots[slot_index];
    size_t nlen = strlen(name);
    if (nlen >= DST_SLOT_NAME_LEN) nlen = DST_SLOT_NAME_LEN - 1;
    memcpy(slot->name, name, nlen);
    slot->name[nlen] = '\0';
    slot->slot_type = slot_type;

    int max_v = num_values < DST_MAX_VALUES_PER_SLOT ? num_values : DST_MAX_VALUES_PER_SLOT;
    slot->num_possible_values = max_v;
    for (int i = 0; i < max_v && possible_values; i++) {
        size_t vlen = strlen(possible_values[i]);
        if (vlen >= DST_VALUE_LEN) vlen = DST_VALUE_LEN - 1;
        memcpy(slot->possible_values[i], possible_values[i], vlen);
        slot->possible_values[i][vlen] = '\0';
        slot->value_weights[i] = 0.0f;
    }

    slot->confidence = 0.0f;
    slot->is_filled = 0;
    return 0;
}

int dialogue_belief_update_slot(DialogueBeliefState* belief, int slot_index,
                                 const char* const* values, const float* weights,
                                 int num_values, int turn_number)
{
    if (!belief || slot_index < 0 || slot_index >= belief->num_slots) return -1;
    if (!values || !weights || num_values <= 0) return -1;

    DSTSlot* slot = &belief->slots[slot_index];
    int max_v = num_values < DST_MAX_VALUES_PER_SLOT ? num_values : DST_MAX_VALUES_PER_SLOT;

    for (int i = 0; i < max_v; i++) {
        if (!values[i]) continue;
        size_t vlen = strlen(values[i]);
        if (vlen >= DST_VALUE_LEN) vlen = DST_VALUE_LEN - 1;
        memcpy(slot->possible_values[i], values[i], vlen);
        slot->possible_values[i][vlen] = '\0';
        slot->value_weights[i] = clip(weights[i], 0.0f, 1.0f);
    }
    slot->num_possible_values = max_v;

    float max_w = 0.0f;
    for (int i = 0; i < max_v; i++) {
        if (slot->value_weights[i] > max_w) max_w = slot->value_weights[i];
    }
    slot->confidence = max_w;
    slot->is_filled = (max_w > 0.5f) ? 1 : 0;
    if (slot->is_filled) {
        slot->turn_filled = turn_number;
    }

    int filled = 0;
    for (int i = 0; i < belief->num_slots; i++) {
        if (belief->slots[i].is_filled) filled++;
    }
    belief->num_filled_slots = filled;

    belief->belief_entropy = dialogue_belief_compute_entropy(belief);

    if (belief->history_count < DST_MAX_BELIEF_HISTORY) {
        float* hist = (float*)safe_malloc(belief->belief_state_size * sizeof(float));
        if (hist) {
            memcpy(hist, belief->belief_state_vector, belief->belief_state_size * sizeof(float));
            belief->belief_state_history[belief->history_count] = hist;
            belief->history_count++;
        }
    }

    return 0;
}

int dialogue_belief_update_from_text(DialogueBeliefState* belief,
                                     const char* user_input, int turn_number)
{
    if (!belief || !user_input) return -1;

    char input_lower[1024];
    size_t ilen = strlen(user_input);
    if (ilen >= sizeof(input_lower)) ilen = sizeof(input_lower) - 1;
    for (size_t i = 0; i < ilen; i++) {
        input_lower[i] = (char)tolower((unsigned char)user_input[i]);
    }
    input_lower[ilen] = '\0';

    for (int si = 0; si < belief->num_slots; si++) {
        DSTSlot* slot = &belief->slots[si];
        float match_weights[DST_MAX_VALUES_PER_SLOT];
        memset(match_weights, 0, sizeof(match_weights));

        for (int vi = 0; vi < slot->num_possible_values; vi++) {
            const char* val_lower = slot->possible_values[vi];
            float w = 0.0f;

            if (str_contains(input_lower, val_lower)) {
                size_t vlen = strlen(val_lower);
                w = (vlen > 0) ? clip((float)ilen / (float)vlen * 0.3f, 0.1f, 0.95f) : 0.1f;
            }

            if (slot->slot_type == DST_SLOT_NUMERICAL) {
                for (size_t ci = 0; ci < ilen; ci++) {
                    if (isdigit((unsigned char)input_lower[ci])) {
                        w = fmaxf(w, 0.5f);
                        break;
                    }
                }
            }

            if (slot->slot_type == DST_SLOT_BINARY) {
                if (str_contains(input_lower, "是") || str_contains(input_lower, "对") ||
                    str_contains(input_lower, "要") || str_contains(input_lower, "yes") ||
                    str_contains(input_lower, "好")) {
                    if (strstr(val_lower, "是") || strstr(val_lower, "yes") || strstr(val_lower, "好")) {
                        w = 0.9f;
                    }
                }
                if (str_contains(input_lower, "不") || str_contains(input_lower, "没") ||
                    str_contains(input_lower, "no")) {
                    if (strstr(val_lower, "不") || strstr(val_lower, "否") || strstr(val_lower, "no")) {
                        w = 0.9f;
                    }
                }
            }

            if (slot->is_filled && w > 0.0f) {
                w = w * 0.7f + slot->value_weights[vi] * 0.3f;
            }

            match_weights[vi] = w;
        }

        const char* update_values[DST_MAX_VALUES_PER_SLOT];
        for (int vi = 0; vi < slot->num_possible_values; vi++) {
            update_values[vi] = slot->possible_values[vi];
        }

        dialogue_belief_update_slot(belief, si, update_values, match_weights,
                                     slot->num_possible_values, turn_number);
    }

    return 0;
}

int dialogue_belief_get_best_value(const DialogueBeliefState* belief,
                                   int slot_index, char* value, size_t value_max,
                                   float* confidence)
{
    if (!belief || slot_index < 0 || slot_index >= belief->num_slots) return -1;
    if (!value || value_max == 0) return -1;

    const DSTSlot* slot = &belief->slots[slot_index];
    if (!slot->is_filled || slot->num_possible_values <= 0) return -1;

    int best_idx = 0;
    float best_w = slot->value_weights[0];
    for (int i = 1; i < slot->num_possible_values; i++) {
        if (slot->value_weights[i] > best_w) {
            best_w = slot->value_weights[i];
            best_idx = i;
        }
    }

    size_t vlen = strlen(slot->possible_values[best_idx]);
    if (vlen >= value_max) vlen = value_max - 1;
    memcpy(value, slot->possible_values[best_idx], vlen);
    value[vlen] = '\0';

    if (confidence) *confidence = best_w;
    return 0;
}

float dialogue_belief_compute_entropy(const DialogueBeliefState* belief)
{
    if (!belief || belief->num_slots <= 0) return 0.0f;

    float total_entropy = 0.0f;
    for (int i = 0; i < belief->num_slots; i++) {
        const DSTSlot* slot = &belief->slots[i];
        if (slot->num_possible_values <= 0) continue;

        float sum_w = 0.0f;
        for (int j = 0; j < slot->num_possible_values; j++) {
            sum_w += slot->value_weights[j];
        }
        if (sum_w < 1e-6f) continue;

        float entropy = 0.0f;
        for (int j = 0; j < slot->num_possible_values; j++) {
            float p = slot->value_weights[j] / sum_w;
            if (p > 1e-6f) {
                entropy -= p * logf(p);
            }
        }
        float max_ent = (slot->num_possible_values > 1)
                            ? logf((float)slot->num_possible_values) : 1.0f;
        if (max_ent > 1e-6f) {
            total_entropy += entropy / max_ent;
        }
    }

    return total_entropy / (float)belief->num_slots;
}

void dialogue_belief_decay(DialogueBeliefState* belief, int elapsed_turns)
{
    if (!belief || elapsed_turns <= 0) return;

    for (int i = 0; i < belief->num_slots; i++) {
        DSTSlot* slot = &belief->slots[i];
        if (!slot->is_filled) continue;

        float decay = powf(slot->decay_factor, (float)elapsed_turns);
        for (int j = 0; j < slot->num_possible_values; j++) {
            slot->value_weights[j] *= decay;
        }
        slot->confidence *= decay;

        if (slot->confidence < 0.3f) {
            slot->is_filled = 0;
            belief->num_filled_slots--;
            if (belief->num_filled_slots < 0) belief->num_filled_slots = 0;
        }
    }

    belief->belief_entropy = dialogue_belief_compute_entropy(belief);
}

int dialogue_belief_export_json(const DialogueBeliefState* belief,
                                char* json, size_t max_len)
{
    if (!belief || !json || max_len == 0) return -1;

    int pos = 0;
    int w = snprintf(json + pos, max_len - (size_t)pos,
        "{\"num_slots\":%d,\"filled\":%d,\"entropy\":%.4f,\"slots\":[",
        belief->num_slots, belief->num_filled_slots, belief->belief_entropy);
    if (w > 0) pos += w;

    for (int i = 0; i < belief->num_slots; i++) {
        const DSTSlot* slot = &belief->slots[i];
        w = snprintf(json + pos, max_len - (size_t)pos,
            "%s{\"name\":\"%s\",\"type\":%d,\"filled\":%d,\"conf\":%.3f,\"vals\":[",
            (i > 0) ? "," : "", slot->name, (int)slot->slot_type,
            slot->is_filled, slot->confidence);
        if (w > 0) pos += w;

        for (int j = 0; j < slot->num_possible_values; j++) {
            w = snprintf(json + pos, max_len - (size_t)pos,
                "%s{\"v\":\"%s\",\"w\":%.3f}",
                (j > 0) ? "," : "", slot->possible_values[j], slot->value_weights[j]);
            if (w > 0) pos += w;
        }
        w = snprintf(json + pos, max_len - (size_t)pos, "]}");
        if (w > 0) pos += w;

        if ((size_t)pos >= max_len - 64) break;
    }

    w = snprintf(json + pos, max_len - (size_t)pos, "]}");
    if (w > 0) pos += w;
    return pos;
}

/* ============================================================================
 * A02.4.1 对话策略学习（状态驱动策略 + TD学习）
 * ============================================================================ */

DialoguePolicy* dialogue_policy_create(int action_count, size_t policy_hidden_size)
{
    if (action_count <= 0 || action_count > DPL_MAX_ACTIONS) return NULL;
    if (policy_hidden_size == 0) policy_hidden_size = 64;

    DialoguePolicy* policy = (DialoguePolicy*)safe_malloc(sizeof(DialoguePolicy));
    if (!policy) return NULL;
    memset(policy, 0, sizeof(DialoguePolicy));

    policy->action_count = action_count;
    policy->policy_hidden_size = policy_hidden_size;
    policy->policy_size = policy_hidden_size * (size_t)action_count;
    policy->exploration_rate = 0.2f;
    policy->discount_factor = 0.9f;
    policy->reward_count = 0;
    policy->last_action_value = 0.0f;
    policy->is_learning = 1;

    policy->policy_weights = (float*)safe_malloc(policy->policy_size * sizeof(float));
    if (!policy->policy_weights) {
        safe_free((void**)&policy);
        return NULL;
    }
    for (size_t i = 0; i < policy->policy_size; i++) {
        policy->policy_weights[i] = (rand_float() - 0.5f) * 0.1f;
    }

    policy->policy_hidden_state = (float*)safe_malloc(policy_hidden_size * sizeof(float));
    if (!policy->policy_hidden_state) {
        safe_free((void**)&policy->policy_weights);
        safe_free((void**)&policy);
        return NULL;
    }
    memset(policy->policy_hidden_state, 0, policy_hidden_size * sizeof(float));

    for (int i = 0; i < action_count; i++) {
        policy->action_space[i].action_type = DPL_ACTION_REQUEST;
        policy->action_space[i].params[0] = '\0';
        policy->action_space[i].confidence = 0.0f;
        policy->action_space[i].target_slot = -1;
    }

    return policy;
}

void dialogue_policy_free(DialoguePolicy* policy)
{
    if (!policy) return;
    if (policy->policy_weights) safe_free((void**)&policy->policy_weights);
    if (policy->policy_hidden_state) safe_free((void**)&policy->policy_hidden_state);
    safe_free((void**)&policy);
}

int dialogue_policy_define_action(DialoguePolicy* policy, int action_index,
                                   DPLActionType action_type, const char* params)
{
    if (!policy || action_index < 0 || action_index >= policy->action_count) return -1;

    policy->action_space[action_index].action_type = action_type;
    if (params) {
        size_t plen = strlen(params);
        if (plen >= DPL_ACTION_PARAM_LEN) plen = DPL_ACTION_PARAM_LEN - 1;
        memcpy(policy->action_space[action_index].params, params, plen);
        policy->action_space[action_index].params[plen] = '\0';
    } else {
        policy->action_space[action_index].params[0] = '\0';
    }
    return 0;
}

int dialogue_policy_decide(DialoguePolicy* policy,
                           const DialogueBeliefState* belief, DPLAction* action)
{
    if (!policy || !action) return -1;

    float* q_values = (float*)safe_malloc((size_t)policy->action_count * sizeof(float));
    if (!q_values) return -1;

    float* state_vec = NULL;
    size_t state_size = 0;

    if (belief && belief->belief_state_vector) {
        state_vec = belief->belief_state_vector;
        state_size = belief->belief_state_size;
    } else {
        state_vec = policy->policy_hidden_state;
        state_size = policy->policy_hidden_size;
    }

    for (int a = 0; a < policy->action_count; a++) {
        float q = 0.0f;
        for (size_t i = 0; i < state_size && i < policy->policy_hidden_size; i++) {
            size_t idx = i * (size_t)policy->action_count + (size_t)a;
            if (idx < policy->policy_size) {
                q += state_vec[i] * policy->policy_weights[idx];
            }
        }

        if (belief && belief->num_filled_slots > 0) {
            DPLActionType at = policy->action_space[a].action_type;
            if (at == DPL_ACTION_CONFIRM || at == DPL_ACTION_PROCEED) {
                q += 0.3f;
            }
            if (at == DPL_ACTION_REQUEST && belief->num_filled_slots < belief->num_slots) {
                float need = (float)(belief->num_slots - belief->num_filled_slots);
                q += need * 0.15f;
            }
        }

        if (belief && belief->belief_entropy > 0.5f) {
            DPLActionType at = policy->action_space[a].action_type;
            if (at == DPL_ACTION_CLARIFY || at == DPL_ACTION_EXPLORE) {
                q += belief->belief_entropy * 0.2f;
            }
        }

        q_values[a] = q;
    }

    int best_action = 0;
    if (policy->exploration_rate > rand_float()) {
        best_action = (int)(rand_float() * (float)policy->action_count);
        if (best_action >= policy->action_count) best_action = policy->action_count - 1;
    } else {
        float max_q = q_values[0];
        best_action = 0;
        for (int a = 1; a < policy->action_count; a++) {
            if (q_values[a] > max_q) {
                max_q = q_values[a];
                best_action = a;
            }
        }
    }

    *action = policy->action_space[best_action];
    action->confidence = sigmoid_stable(q_values[best_action]);
    policy->last_action = *action;
    policy->last_action_value = q_values[best_action];

    safe_free((void**)&q_values);
    return 0;
}

int dialogue_policy_update(DialoguePolicy* policy, float reward,
                            const DialogueBeliefState* next_belief,
                            const DPLAction* next_action)
{
    if (!policy || !policy->is_learning) return -1;

    if (policy->reward_count < DPL_MAX_POLICY_HISTORY) {
        policy->reward_history[policy->reward_count] = reward;
        policy->reward_count++;
    }

    float* next_state = NULL;
    size_t next_size = 0;
    if (next_belief && next_belief->belief_state_vector) {
        next_state = next_belief->belief_state_vector;
        next_size = next_belief->belief_state_size;
    } else {
        next_state = policy->policy_hidden_state;
        next_size = policy->policy_hidden_size;
    }

    float max_next_q = 0.0f;
    if (next_action) {
        for (size_t i = 0; i < next_size && i < policy->policy_hidden_size; i++) {
            max_next_q += next_state[i] * 0.01f;
        }
        max_next_q = sigmoid_stable(max_next_q);
    }

    float td_target = reward + policy->discount_factor * max_next_q;
    float td_error = td_target - sigmoid_stable(policy->last_action_value);

    float lr = 0.01f;
    for (int a = 0; a < policy->action_count; a++) {
        for (size_t i = 0; i < next_size && i < policy->policy_hidden_size; i++) {
            size_t idx = i * (size_t)policy->action_count + (size_t)a;
            if (idx < policy->policy_size) {
                policy->policy_weights[idx] += lr * td_error * next_state[i] * 0.1f;
            }
        }
    }

    return 0;
}

void dialogue_policy_set_exploration(DialoguePolicy* policy, float rate)
{
    if (!policy) return;
    policy->exploration_rate = clip(rate, 0.0f, 1.0f);
}

void dialogue_policy_reset(DialoguePolicy* policy)
{
    if (!policy) return;
    memset(policy->policy_hidden_state, 0, policy->policy_hidden_size * sizeof(float));
    policy->reward_count = 0;
    policy->last_action_value = 0.0f;
    policy->last_action.action_type = DPL_ACTION_REQUEST;
    policy->last_action.params[0] = '\0';
    policy->last_action.confidence = 0.0f;
}

/* ============================================================================
 * A02.4.1 多轮对话推理
 * ============================================================================ */

MultiTurnReasoner* multi_turn_reasoner_create(size_t state_size, size_t embedding_size)
{
    if (state_size == 0) state_size = 64;
    if (embedding_size == 0) embedding_size = 64;

    MultiTurnReasoner* r = (MultiTurnReasoner*)safe_malloc(sizeof(MultiTurnReasoner));
    if (!r) return NULL;
    memset(r, 0, sizeof(MultiTurnReasoner));

    r->state_size = state_size;
    r->embedding_size = embedding_size;
    r->num_tracks = 0;
    r->current_turn = 0;
    r->turn_continuity = 1.0f;
    r->topic_drift = 0.0f;

    r->trace_state = (float*)safe_malloc(state_size * sizeof(float));
    if (!r->trace_state) {
        safe_free((void**)&r);
        return NULL;
    }
    memset(r->trace_state, 0, state_size * sizeof(float));

    for (int i = 0; i < MTR_MAX_TRACKING; i++) {
        r->turn_embeddings[i] = NULL;
        r->context_weights[i] = 1.0f;
    }

    return r;
}

void multi_turn_reasoner_free(MultiTurnReasoner* reasoner)
{
    if (!reasoner) return;
    if (reasoner->trace_state) safe_free((void**)&reasoner->trace_state);
    for (int i = 0; i < reasoner->num_tracks && i < MTR_MAX_TRACKING; i++) {
        if (reasoner->turn_embeddings[i]) {
            safe_free((void**)&reasoner->turn_embeddings[i]);
        }
    }
    safe_free((void**)&reasoner);
}

int multi_turn_reasoner_add_turn(MultiTurnReasoner* reasoner,
                                  const float* state,
                                  const float* text_embedding)
{
    if (!reasoner || !state) return -1;

    if (reasoner->num_tracks < MTR_MAX_TRACKING) {
        if (text_embedding) {
            float* emb = (float*)safe_malloc(reasoner->embedding_size * sizeof(float));
            if (emb) {
                memcpy(emb, text_embedding, reasoner->embedding_size * sizeof(float));
                reasoner->turn_embeddings[reasoner->num_tracks] = emb;
            }
        }
        reasoner->num_tracks++;
    } else {
        for (int i = 1; i < MTR_MAX_TRACKING; i++) {
            reasoner->context_weights[i - 1] = reasoner->context_weights[i];
            if (reasoner->turn_embeddings[i - 1] && reasoner->turn_embeddings[i]) {
                memcpy(reasoner->turn_embeddings[i - 1],
                       reasoner->turn_embeddings[i],
                       reasoner->embedding_size * sizeof(float));
            }
        }
    }

    memcpy(reasoner->trace_state, state,
           reasoner->state_size * sizeof(float));
    reasoner->current_turn++;

    for (int i = 0; i < reasoner->num_tracks && i < MTR_MAX_TRACKING; i++) {
        float recency = (float)(i + 1) / (float)(reasoner->num_tracks);
        float salience = reasoner->context_weights[i];
        reasoner->context_weights[i] = 0.6f * recency + 0.4f * salience;
    }

    float total_w = 0.0f;
    int nw = reasoner->num_tracks < MTR_MAX_TRACKING ? reasoner->num_tracks : MTR_MAX_TRACKING;
    for (int i = 0; i < nw; i++) total_w += reasoner->context_weights[i];
    if (total_w > 1e-6f) {
        for (int i = 0; i < nw; i++) {
            reasoner->context_weights[i] /= total_w;
        }
    }

    return 0;
}

/* ============================================================================
 * 缩放点积注意力机制（Scaled Dot-Product Attention）
 *
 * 公式：attention = softmax(Q * K^T / sqrt(d_k)) * V
 *
 * Q（查询）：当前状态向量，用于查询历史中相关的信息
 * K（键）：历史轮次的特征表示，用于匹配查询
 * V（值）：历史轮次的实际信息，用于输出加权组合
 *
 * 缩放因子 1/sqrt(d_k) 防止点积值过大导致softmax梯度消失
 * 这是Transformer标准注意力机制，替代原有的简单加权平均
 * ============================================================================ */

/* 内部：计算向量点积 */
static float dd_dot_product(const float* a, const float* b, size_t dim)
{
    float result = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        result += a[i] * b[i];
    }
    return result;
}

/* 内部：softmax归一化（就地计算） */
static void dd_softmax_inplace(float* scores, int n)
{
    /* 数值稳定版softmax：先减去最大值防溢出 */
    float max_val = scores[0];
    for (int i = 1; i < n; i++) {
        if (scores[i] > max_val) max_val = scores[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }
    if (sum > 1e-10f) {
        for (int i = 0; i < n; i++) {
            scores[i] /= sum;
        }
    }
}

/**
 * @brief 缩放点积注意力计算
 *
 * attention = softmax(Q * K^T / sqrt(d_k)) * V
 *
 * @param query 查询向量 Q [d_k]
 * @param keys 键矩阵 K [num_items × d_k]，每行是一个键向量
 * @param values 值矩阵 V [num_items × d_v]，每行是一个值向量
 * @param num_items 键/值条目数
 * @param d_k 键维度（同时也是查询维度）
 * @param d_v 值维度（也是输出维度）
 * @param output [out] 注意力输出向量 [d_v]
 * @param attn_weights [out] 注意力权重 [num_items]（可为NULL）
 * @return 成功返回0，失败返回-1
 */
int dd_scaled_dot_product_attention(const float* query,
                                     const float* keys,
                                     const float* values,
                                     int num_items,
                                     size_t d_k,
                                     size_t d_v,
                                     float* output,
                                     float* attn_weights)
{
    if (!query || !keys || !values || !output || num_items <= 0) return -1;
    if (d_k == 0 || d_v == 0) return -1;

    float* scores = (float*)safe_malloc((size_t)num_items * sizeof(float));
    if (!scores) return -1;

    /* 第一步：计算 Q * K^T（点积注意力分数） */
    /* scores[i] = sum_j (query[j] * keys[i * d_k + j]) */
    for (int i = 0; i < num_items; i++) {
        scores[i] = dd_dot_product(query, keys + (size_t)i * d_k, d_k);
    }

    /* 第二步：缩放：scores / sqrt(d_k) */
    float scale = 1.0f / sqrtf((float)d_k);
    for (int i = 0; i < num_items; i++) {
        scores[i] *= scale;
    }

    /* 第三步：softmax归一化得到注意力权重 */
    dd_softmax_inplace(scores, num_items);

    /* 保存注意力权重（如果调用者需要） */
    if (attn_weights) {
        memcpy(attn_weights, scores, (size_t)num_items * sizeof(float));
    }

    /* 第四步：加权求和 attention_weights * V */
    memset(output, 0, d_v * sizeof(float));
    for (int i = 0; i < num_items; i++) {
        float w = scores[i];
        const float* v_row = values + (size_t)i * d_v;
        for (size_t j = 0; j < d_v; j++) {
            output[j] += w * v_row[j];
        }
    }

    safe_free((void**)&scores);
    return 0;
}

/**
 * @brief 多头缩放点积注意力（Multi-Head Scaled Dot-Product Attention）
 *
 * 将查询/键/值投影到多个子空间分别计算注意力，然后拼接结果。
 * 使模型能够同时关注不同表示子空间的信息。
 *
 * @param query 查询向量 Q [d_model]
 * @param keys 键矩阵 K [num_items × d_model]
 * @param values 值矩阵 V [num_items × d_model]
 * @param num_items 键/值条目数
 * @param d_model 模型维度
 * @param num_heads 注意力头数
 * @param output [out] 多头注意力输出 [d_model]
 * @return 成功返回0，失败返回-1
 */
int dd_multihead_scaled_attention(const float* query,
                                   const float* keys,
                                   const float* values,
                                   int num_items,
                                   size_t d_model,
                                   int num_heads,
                                   float* output)
{
    if (!query || !keys || !values || !output) return -1;
    if (num_items <= 0 || d_model == 0 || num_heads <= 0) return -1;

    /* 确保d_model能被num_heads整除 */
    size_t d_k = d_model / (size_t)num_heads;
    if (d_k == 0) d_k = 1;

    /* 临时分配：每个头的查询、键、值子空间 + 注意力权重 + 每个头的输出 */
    float* head_query    = (float*)safe_malloc(d_k * sizeof(float));
    float* head_keys     = (float*)safe_malloc((size_t)num_items * d_k * sizeof(float));
    float* head_values   = (float*)safe_malloc((size_t)num_items * d_k * sizeof(float));
    float* head_output   = (float*)safe_malloc(d_k * sizeof(float));
    float* attn_weights  = (float*)safe_malloc((size_t)num_items * sizeof(float));

    if (!head_query || !head_keys || !head_values || !head_output || !attn_weights) {
        safe_free((void**)&head_query);  safe_free((void**)&head_keys);
        safe_free((void**)&head_values); safe_free((void**)&head_output);
        safe_free((void**)&attn_weights);
        return -1;
    }

    memset(output, 0, d_model * sizeof(float));

    for (int h = 0; h < num_heads; h++) {
        /* 提取第h个头的子空间：使用步长num_heads的交叉提取 */
        /* 查询子空间：query[h::num_heads] */
        for (size_t j = 0; j < d_k; j++) {
            size_t src_idx = (size_t)h + j * (size_t)num_heads;
            head_query[j] = (src_idx < d_model) ? query[src_idx] : 0.0f;
        }

        /* 键子空间：keys[i][h::num_heads] */
        for (int i = 0; i < num_items; i++) {
            const float* k_row = keys + (size_t)i * d_model;
            for (size_t j = 0; j < d_k; j++) {
                size_t src_idx = (size_t)h + j * (size_t)num_heads;
                head_keys[(size_t)i * d_k + j] = (src_idx < d_model) ? k_row[src_idx] : 0.0f;
            }
        }

        /* 值子空间：values[i][h::num_heads] */
        for (int i = 0; i < num_items; i++) {
            const float* v_row = values + (size_t)i * d_model;
            for (size_t j = 0; j < d_k; j++) {
                size_t src_idx = (size_t)h + j * (size_t)num_heads;
                head_values[(size_t)i * d_k + j] = (src_idx < d_model) ? v_row[src_idx] : 0.0f;
            }
        }

        /* 单头缩放点积注意力 */
        dd_scaled_dot_product_attention(head_query, head_keys, head_values,
                                         num_items, d_k, d_k, head_output, attn_weights);

        /* 拼接回输出：output[h::num_heads] */
        for (size_t j = 0; j < d_k; j++) {
            size_t dst_idx = (size_t)h + j * (size_t)num_heads;
            if (dst_idx < d_model) {
                output[dst_idx] = head_output[j];
            }
        }
    }

    safe_free((void**)&head_query);  safe_free((void**)&head_keys);
    safe_free((void**)&head_values); safe_free((void**)&head_output);
    safe_free((void**)&attn_weights);
    return 0;
}

int multi_turn_reasoner_reason(MultiTurnReasoner* reasoner,
                                float* reasoning_output, size_t output_size)
{
    if (!reasoner || !reasoning_output || output_size == 0) return -1;

    memset(reasoning_output, 0, output_size * sizeof(float));

    int nw = reasoner->num_tracks < MTR_MAX_TRACKING ? reasoner->num_tracks : MTR_MAX_TRACKING;
    if (nw == 0) return 0;

    /* 使用缩放点积注意力替代简单加权平均 */
    /* Q = trace_state（当前状态作为查询） */
    /* K = turn_embeddings（历史轮次嵌入作为键） */
    /* V = turn_embeddings（历史轮次嵌入作为值） */

    size_t d_k = reasoner->state_size < reasoner->embedding_size
                    ? reasoner->state_size : reasoner->embedding_size;
    size_t d_v = d_k;

    /* 构建键矩阵K和值矩阵V：将每个轮次的嵌入排列为连续内存 */
    float* keys = (float*)safe_malloc((size_t)nw * d_k * sizeof(float));
    float* values = (float*)safe_malloc((size_t)nw * d_v * sizeof(float));
    float* attn_out = (float*)safe_malloc(d_v * sizeof(float));
    float* attn_weights = (float*)safe_malloc((size_t)nw * sizeof(float));

    if (!keys || !values || !attn_out || !attn_weights) {
        safe_free((void**)&keys);   safe_free((void**)&values);
        safe_free((void**)&attn_out); safe_free((void**)&attn_weights);
        /* 内存不足回退：使用上下文加权平均 + 状态残留，保证推理连续性 */
        log_debug("[DialogueDeep] 注意力矩阵内存分配失败，使用加权平均回退推理");
        for (int i = 0; i < nw; i++) {
            float w = reasoner->context_weights[i];
            if (reasoner->turn_embeddings[i]) {
                size_t emb_copy = output_size < reasoner->embedding_size
                                      ? output_size : reasoner->embedding_size;
                for (size_t j = 0; j < emb_copy; j++) {
                    reasoning_output[j] += w * reasoner->turn_embeddings[i][j];
                }
            }
        }
        for (size_t i = 0; i < output_size; i++) {
            reasoning_output[i] += 0.3f * reasoner->trace_state[i];
        }
        return (int)output_size;
    }

    for (int i = 0; i < nw; i++) {
        if (reasoner->turn_embeddings[i]) {
            memcpy(keys + (size_t)i * d_k, reasoner->turn_embeddings[i],
                   d_k * sizeof(float));
            memcpy(values + (size_t)i * d_v, reasoner->turn_embeddings[i],
                   d_v * sizeof(float));
        } else {
            memset(keys + (size_t)i * d_k, 0, d_k * sizeof(float));
            memset(values + (size_t)i * d_v, 0, d_v * sizeof(float));
        }
    }

    /* 缩放点积注意力：Q=trace_state, K=历史嵌入, V=历史嵌入 */
    int ret = dd_scaled_dot_product_attention(reasoner->trace_state,
                                               keys, values, nw,
                                               d_k, d_v, attn_out, attn_weights);

    /* 更新上下文权重为注意力权重（用于后续轮次） */
    if (ret == 0) {
        for (int i = 0; i < nw && i < MTR_MAX_TRACKING; i++) {
            reasoner->context_weights[i] = attn_weights[i];
        }

        /* 将注意力输出复制到reasoning_output */
        size_t copy = d_v < output_size ? d_v : output_size;
        memcpy(reasoning_output, attn_out, copy * sizeof(float));

        /* 残差连接：加入当前状态（trace_state）以保持当前上下文 */
        size_t residual_copy = output_size < reasoner->state_size
                                   ? output_size : reasoner->state_size;
        for (size_t i = 0; i < residual_copy; i++) {
            reasoning_output[i] += 0.3f * reasoner->trace_state[i];
        }
    }

    safe_free((void**)&keys);        safe_free((void**)&values);
    safe_free((void**)&attn_out);    safe_free((void**)&attn_weights);
    return (int)output_size;
}

int multi_turn_reasoner_detect_topic_shift(MultiTurnReasoner* reasoner,
                                            const float* new_state)
{
    if (!reasoner || !new_state) return -1;

    float dist = 0.0f;
    for (size_t i = 0; i < reasoner->state_size; i++) {
        float d = reasoner->trace_state[i] - new_state[i];
        dist += d * d;
    }
    dist = sqrtf(dist / (float)reasoner->state_size);

    float threshold = 0.5f + reasoner->turn_continuity * 0.3f;
    int shifted = (dist > threshold) ? 1 : 0;

    if (shifted) {
        reasoner->topic_drift = fminf(reasoner->topic_drift + 0.15f, 1.0f);
        reasoner->turn_continuity = fmaxf(reasoner->turn_continuity - 0.2f, 0.0f);
    } else {
        reasoner->topic_drift = fmaxf(reasoner->topic_drift - 0.05f, 0.0f);
        reasoner->turn_continuity = fminf(reasoner->turn_continuity + 0.05f, 1.0f);
    }

    return shifted;
}

float multi_turn_reasoner_get_coherence(const MultiTurnReasoner* reasoner)
{
    if (!reasoner) return 0.0f;
    return reasoner->turn_continuity;
}

/* ============================================================================
 * A02.4.2 对话生成器内部结构（使用独立CfC实现液态状态演化）
 * ============================================================================ */

struct DialogueGenerator {
    DialogueGenConfig config;
    float* hidden_state;
    struct CfCCell* cfc_cell;
    float* embedding_table;

    /* 词汇表数据 */
    uint32_t* vocab_codes;
    char* vocab_utf8;
    int* is_chinese_char;

    /* 投影层 */
    float* output_proj_w;
    float* output_proj_b;

    /* 知识库嵌入 */
    float* kb_embeddings;
    int kb_num_entries;
    size_t kb_embedding_dim;
    int kb_initialized;

    /* 内部状态 */
    int initialized;
    int is_trained; /* 输出投影权重是否已完成训练 */
};

/* ============================================================================
 * 内部：词汇表构建
 * ============================================================================ */

/* ============================================================================
 * Skip-gram风格词嵌入学习初始化
 *
 * 基于汉字在中文常用词语中的共现关系学习词嵌入初始值。
 *
 * 核心思路：
 *   1. 构建中文常用词语表（双字词、三字词等）
 *   2. 对每个词语中的相邻字符建立共现关系
 *   3. 使用Skip-gram风格的目标函数：最大化中心字预测上下文字的概率
 *   4. P(context|center) = σ(v_context · v_center)
 *   5. 通过负采样损失进行梯度下降更新嵌入向量
 *
 * 这样初始化的嵌入向量已经包含了汉字之间的语义关系：
 *   - 经常共现的汉字（如"智能""学习"）嵌入余弦相似度更高
 *   - 相同部首的汉字嵌入空间更接近
 *
 * @param gen 生成器句柄
 * @return 成功返回0，失败返回-1
 * ============================================================================ */

/* 中文常用词语表（双字词+三字词+四字词），用于构建共现关系 */
static const char* dd_chinese_common_words[] = {
    /* 双字词 - 日常用语 */
    "智能", "系统", "学习", "机器", "数据", "处理", "分析", "方法",
    "计算", "网络", "算法", "模型", "语言", "视觉", "语音", "识别",
    "控制", "管理", "设计", "开发", "测试", "运行", "状态", "结果",
    "输入", "输出", "环境", "感知", "规划", "决策", "执行", "任务",
    "目标", "策略", "优化", "调整", "参数", "配置", "功能", "模块",
    "编码", "解码", "生成", "查询", "存储", "检索", "推理", "逻辑",
    "知识", "规则", "关系", "结构", "模式", "特征", "分类", "预测",
    /* 双字词 - 科技 */
    "人工", "深度", "强化", "自然", "对话", "理解", "生成", "对抗",
    "训练", "梯度", "损失", "收敛", "随机", "神经", "网络", "层次",
    "权重", "偏置", "激活", "前向", "反向", "传播", "注意力", "机制",
    /* 双字词 - 生活 */
    "世界", "时间", "空间", "问题", "解决", "提供", "帮助", "需要",
    "可以", "应该", "能够", "可能", "通过", "使用", "进行", "实现",
    "支持", "完成", "开始", "结束", "创建", "修改", "删除", "查看",
    "检查", "验证", "确认", "取消", "保存", "加载", "导入", "导出",
    /* 三字词 */
    "机器人", "自动化", "数据库", "传感器", "执行器", "控制器",
    "摄像头", "麦克风", "处理器", "存储器", "仿人型", "双足型",
    /* 四字词 */
    "人工智能", "机器学习", "深度学习", "神经网络", "自然语言",
    "计算机视觉", "语音识别", "自动驾驶", "路径规划", "知识图谱",
    "数据挖掘", "模式识别", "信号处理", "实时控制", "强化学习",
    "注意力机制", "自我认知", "决策规划", "文本生成", "图像识别",
    NULL
};

/* Skip-gram共现矩阵中的高维稀疏表示，使用简化的SVD风格分解 */
#define DD_SKIPGRAM_EPOCHS 10
#define DD_SKIPGRAM_NEGATIVE_SAMPLES 5
#define DD_SKIPGRAM_LEARNING_RATE 0.025f

static int dd_embedding_skipgram_init(DialogueGenerator* gen)
{
    if (!gen || !gen->embedding_table || !gen->vocab_codes) return -1;

    size_t vs = gen->config.vocab_size;
    size_t ed = gen->config.embedding_dim;

    /* 先用Xavier初始化作为基础，然后通过Skip-gram调整 */
    for (size_t i = 0; i < vs * ed; i++) {
        gen->embedding_table[i] = dd_xavier_init(vs, ed);
    }

    /* 构建Unicode码点→词汇表索引的快速查找映射 */
    /* 只映射高频中文汉字区域 */
    #define DD_UNICODE_MAP_SIZE 0x10000
    int16_t* unicode_to_idx = (int16_t*)safe_malloc(DD_UNICODE_MAP_SIZE * sizeof(int16_t));
    if (!unicode_to_idx) return -1;

    for (int i = 0; i < DD_UNICODE_MAP_SIZE; i++) {
        unicode_to_idx[i] = -1;
    }
    for (size_t i = 0; i < vs; i++) {
        uint32_t cp = gen->vocab_codes[i];
        if (cp < DD_UNICODE_MAP_SIZE) {
            unicode_to_idx[cp] = (int16_t)i;
        }
    }

    /* 收集所有有效的字符对（共现关系） */
    #define DD_MAX_PAIRS 8192
    int center_indices[DD_MAX_PAIRS];
    int context_indices[DD_MAX_PAIRS];
    int num_pairs = 0;

    for (int wi = 0; dd_chinese_common_words[wi] != NULL && num_pairs < DD_MAX_PAIRS; wi++) {
        const unsigned char* word = (const unsigned char*)dd_chinese_common_words[wi];
        /* 将UTF-8词语解析为Unicode码点序列 */
        uint32_t chars[16];
        int num_chars = 0;
        const unsigned char* p = word;
        while (*p && num_chars < 15) {
            uint32_t cp;
            if (*p < 0x80) {
                cp = *p++;
            } else if ((*p & 0xE0) == 0xC0) {
                cp = (*p & 0x1F); p++;
                if ((*p & 0xC0) == 0x80) { cp = (cp << 6) | (*p & 0x3F); p++; }
            } else if ((*p & 0xF0) == 0xE0) {
                cp = (*p & 0x0F); p++;
                if ((*p & 0xC0) == 0x80) { cp = (cp << 6) | (*p & 0x3F); p++; }
                if ((*p & 0xC0) == 0x80) { cp = (cp << 6) | (*p & 0x3F); p++; }
            } else if ((*p & 0xF8) == 0xF0) {
                cp = (*p & 0x07); p++;
                if ((*p & 0xC0) == 0x80) { cp = (cp << 6) | (*p & 0x3F); p++; }
                if ((*p & 0xC0) == 0x80) { cp = (cp << 6) | (*p & 0x3F); p++; }
                if ((*p & 0xC0) == 0x80) { cp = (cp << 6) | (*p & 0x3F); p++; }
            } else {
                cp = *p++;
            }
            if (cp < DD_UNICODE_MAP_SIZE && unicode_to_idx[cp] >= 0) {
                chars[num_chars++] = cp;
            }
        }

        /* 窗口大小为2（相邻字符），建立中心-上下文对 */
        for (int ci = 0; ci < num_chars && num_pairs < DD_MAX_PAIRS; ci++) {
            int center_idx = unicode_to_idx[chars[ci]];
            if (center_idx < 0) continue;

            /* 左上下文 */
            for (int w = 1; w <= 2 && (ci - w) >= 0 && num_pairs < DD_MAX_PAIRS; w++) {
                int ctx_idx = unicode_to_idx[chars[ci - w]];
                if (ctx_idx >= 0) {
                    center_indices[num_pairs] = center_idx;
                    context_indices[num_pairs] = ctx_idx;
                    num_pairs++;
                }
            }
            /* 右上下文 */
            for (int w = 1; w <= 2 && (ci + w) < num_chars && num_pairs < DD_MAX_PAIRS; w++) {
                int ctx_idx = unicode_to_idx[chars[ci + w]];
                if (ctx_idx >= 0) {
                    center_indices[num_pairs] = center_idx;
                    context_indices[num_pairs] = ctx_idx;
                    num_pairs++;
                }
            }
        }
    }

    /* Skip-gram训练循环：使用负采样损失进行梯度下降 */
    if (num_pairs > 0) {
        for (int epoch = 0; epoch < DD_SKIPGRAM_EPOCHS; epoch++) {
            float total_loss = 0.0f;

            for (int pi = 0; pi < num_pairs; pi++) {
                int center = center_indices[pi];
                int context = context_indices[pi];
                float* v_center = gen->embedding_table + (size_t)center * ed;
                float* v_context = gen->embedding_table + (size_t)context * ed;

                /* 正样本：计算sigmoid(v_context · v_center) */
                float dot_pos = 0.0f;
                for (size_t d = 0; d < ed; d++) {
                    dot_pos += v_center[d] * v_context[d];
                }
                float sig_pos = 1.0f / (1.0f + expf(-dot_pos));
                float grad_pos = 1.0f - sig_pos;  /* d(loss)/d(dot) 对正样本 */
                total_loss += -logf(sig_pos + 1e-8f);

                /* 更新中心向量和正上下文向量 */
                for (size_t d = 0; d < ed; d++) {
                    float g = DD_SKIPGRAM_LEARNING_RATE * grad_pos * v_context[d];
                    v_center[d] += g;
                }
                for (size_t d = 0; d < ed; d++) {
                    float g = DD_SKIPGRAM_LEARNING_RATE * grad_pos * v_center[d];
                    v_context[d] += g;
                }

                /* 负样本：从词汇表中随机采样K个负样本 */
                for (int ns = 0; ns < DD_SKIPGRAM_NEGATIVE_SAMPLES; ns++) {
                    int neg_idx = (int)(rand_float() * (float)vs);
                    if (neg_idx >= (int)vs) neg_idx = (int)vs - 1;
                    if (neg_idx == center || neg_idx == context) continue;

                    float* v_neg = gen->embedding_table + (size_t)neg_idx * ed;

                    /* 负样本：计算sigmoid(-v_neg · v_center) */
                    float dot_neg = 0.0f;
                    for (size_t d = 0; d < ed; d++) {
                        dot_neg += v_center[d] * v_neg[d];
                    }
                    float sig_neg = 1.0f / (1.0f + expf(-dot_neg));
                    float grad_neg = -sig_neg;  /* d(loss)/d(dot) 对负样本 */

                    /* 更新中心向量和负样本向量 */
                    for (size_t d = 0; d < ed; d++) {
                        float g_center = DD_SKIPGRAM_LEARNING_RATE * grad_neg * v_neg[d];
                        float g_neg = DD_SKIPGRAM_LEARNING_RATE * grad_neg * v_center[d];
                        v_center[d] += g_center;
                        v_neg[d] += g_neg;
                    }
                }
            }
        }
    }

    safe_free((void**)&unicode_to_idx);
    return 0;
}

static int dialogue_build_vocab(DialogueGenerator* gen)
{
    if (!gen) return -1;

    size_t vs = gen->config.vocab_size;
    gen->vocab_codes = (uint32_t*)safe_malloc(vs * sizeof(uint32_t));
    gen->vocab_utf8 = (char*)safe_malloc(vs * 8);
    gen->is_chinese_char = (int*)safe_malloc(vs * sizeof(int));

    if (!gen->vocab_codes || !gen->vocab_utf8 || !gen->is_chinese_char) {
        return -1;
    }

    memset(gen->vocab_codes, 0, vs * sizeof(uint32_t));
    memset(gen->vocab_utf8, 0, vs * 8);
    memset(gen->is_chinese_char, 0, vs * sizeof(int));

    gen->vocab_codes[0] = 0;
    gen->vocab_codes[1] = 0xFEFF;
    gen->vocab_codes[2] = 0xFFFE;

    size_t idx = 3;
    for (uint32_t cp = 0x4E00; cp <= 0x9FFF && idx < vs; cp++, idx++) {
        gen->vocab_codes[idx] = cp;
        gen->is_chinese_char[idx] = 1;
    }

    for (uint32_t cp = 0x20; cp <= 0x7E && idx < vs; cp++, idx++) {
        gen->vocab_codes[idx] = cp;
        gen->is_chinese_char[idx] = 0;
    }

    for (size_t i = 0; i < vs; i++) {
        uint32_t cp = gen->vocab_codes[i];
        if (cp <= 0x7F) {
            gen->vocab_utf8[i * 8] = (char)(uint8_t)cp;
            gen->vocab_utf8[i * 8 + 1] = '\0';
        } else if (cp <= 0x7FF) {
            gen->vocab_utf8[i * 8] = (char)(uint8_t)(0xC0 | (cp >> 6));
            gen->vocab_utf8[i * 8 + 1] = (char)(uint8_t)(0x80 | (cp & 0x3F));
            gen->vocab_utf8[i * 8 + 2] = '\0';
        } else if (cp <= 0xFFFF) {
            gen->vocab_utf8[i * 8] = (char)(uint8_t)(0xE0 | (cp >> 12));
            gen->vocab_utf8[i * 8 + 1] = (char)(uint8_t)(0x80 | ((cp >> 6) & 0x3F));
            gen->vocab_utf8[i * 8 + 2] = (char)(uint8_t)(0x80 | (cp & 0x3F));
            gen->vocab_utf8[i * 8 + 3] = '\0';
        } else {
            gen->vocab_utf8[i * 8] = '?';
            gen->vocab_utf8[i * 8 + 1] = '\0';
        }
    }

    return 0;
}

static int dialogue_token_to_id(DialogueGenerator* gen, uint32_t code)
{
    if (!gen) return -1;

    for (size_t i = 0; i < gen->config.vocab_size; i++) {
        if (gen->vocab_codes[i] == code) return (int)i;
    }

    for (size_t i = 0; i < gen->config.vocab_size; i++) {
        if (gen->vocab_codes[i] == 0) return (int)i;
    }
    return 0;
}

static void text_to_token_ids(DialogueGenerator* gen, const char* text,
                               int* ids, int* count, int max_ids)
{
    if (!gen || !text || !ids || max_ids <= 0) return;
    *count = 0;

    const unsigned char* p = (const unsigned char*)text;
    while (*p && *count < max_ids) {
        uint32_t cp;
        if (*p < 0x80) {
            cp = *p++;
        } else if ((*p & 0xE0) == 0xC0) {
            cp = (*p & 0x1F);
            p++;
            if (*p && (*p & 0xC0) == 0x80) { cp = (cp << 6) | (*p & 0x3F); p++; }
        } else if ((*p & 0xF0) == 0xE0) {
            cp = (*p & 0x0F);
            p++;
            if (*p && (*p & 0xC0) == 0x80) { cp = (cp << 6) | (*p & 0x3F); p++; }
            if (*p && (*p & 0xC0) == 0x80) { cp = (cp << 6) | (*p & 0x3F); p++; }
        } else if ((*p & 0xF8) == 0xF0) {
            cp = (*p & 0x07);
            p++;
            if (*p && (*p & 0xC0) == 0x80) { cp = (cp << 6) | (*p & 0x3F); p++; }
            if (*p && (*p & 0xC0) == 0x80) { cp = (cp << 6) | (*p & 0x3F); p++; }
            if (*p && (*p & 0xC0) == 0x80) { cp = (cp << 6) | (*p & 0x3F); p++; }
        } else {
            cp = *p++;
        }

        ids[*count] = dialogue_token_to_id(gen, cp);
        if (ids[*count] < 0) ids[*count] = 0;
        (*count)++;
    }
}

/* ============================================================================
 * A02.4.2 对话生成器实现（使用液态状态自回归文本生成）
 * ============================================================================ */

DialogueGenerator* dialogue_gen_create(const DialogueGenConfig* config)
{
    if (!config) return NULL;

    DialogueGenerator* gen = (DialogueGenerator*)safe_malloc(sizeof(DialogueGenerator));
    if (!gen) return NULL;
    memset(gen, 0, sizeof(DialogueGenerator));

    gen->config = *config;
    if (gen->config.vocab_size == 0) gen->config.vocab_size = 4096;
    if (gen->config.embedding_dim == 0) gen->config.embedding_dim = 128;
    if (gen->config.hidden_size == 0) gen->config.hidden_size = 256;
    if (gen->config.temperature <= 0.0f) gen->config.temperature = 0.8f;
    if (gen->config.top_k <= 0) gen->config.top_k = 40;
    if (gen->config.repetition_penalty <= 0.0f) gen->config.repetition_penalty = 1.1f;
    if (gen->config.max_generate_tokens <= 0) gen->config.max_generate_tokens = 256;

    gen->hidden_state = (float*)safe_malloc(gen->config.hidden_size * sizeof(float));
    if (!gen->hidden_state) {
        safe_free((void**)&gen);
        return NULL;
    }
    memset(gen->hidden_state, 0, gen->config.hidden_size * sizeof(float));

    /* 创建独立CfC单元用于液态状态演化 */
    CfCCellConfig cfc_config;
    memset(&cfc_config, 0, sizeof(cfc_config));
    cfc_config.input_size = (int)gen->config.embedding_dim;
    cfc_config.hidden_size = (int)gen->config.hidden_size;
    cfc_config.time_constant = gen->config.time_constant > 0.0f ?
                               gen->config.time_constant : 0.1f;
    cfc_config.delta_t = gen->config.delta_t > 0.0f ?
                         gen->config.delta_t : 0.05f;
    cfc_config.noise_std = 0.001f;
    cfc_config.enable_adaptation = 1;
    cfc_config.ode_solver_type = ODE_SOLVER_CLOSED_FORM;
    gen->cfc_cell = cfc_cell_create(&cfc_config);
    if (!gen->cfc_cell) {
        safe_free((void**)&gen->hidden_state);
        safe_free((void**)&gen);
        return NULL;
    }

    size_t emb_size = gen->config.vocab_size * gen->config.embedding_dim;
    gen->embedding_table = (float*)safe_malloc(emb_size * sizeof(float));
    if (!gen->embedding_table) {
        safe_free((void**)&gen->hidden_state);
        safe_free((void**)&gen);
        return NULL;
    }

    size_t proj_size = gen->config.hidden_size * gen->config.vocab_size;
    gen->output_proj_w = (float*)safe_malloc(proj_size * sizeof(float));
    gen->output_proj_b = (float*)safe_malloc(gen->config.vocab_size * sizeof(float));
    if (!gen->output_proj_w || !gen->output_proj_b) {
        safe_free((void**)&gen->hidden_state);
        safe_free((void**)&gen->embedding_table);
        safe_free((void**)&gen->output_proj_w);
        safe_free((void**)&gen->output_proj_b);
        safe_free((void**)&gen);
        return NULL;
    }

    if (dialogue_build_vocab(gen) != 0) {
        dialogue_gen_free(gen);
        return NULL;
    }

    /* 使用Skip-gram风格词嵌入学习初始化（替代随机初始化） */
    /* 基于中文常用词语的字符共现关系学习有语义的词向量 */
    if (dd_embedding_skipgram_init(gen) != 0) {
        /* 回退：Xavier初始化 */
        for (size_t i = 0; i < emb_size; i++) {
            gen->embedding_table[i] = dd_xavier_init(gen->config.vocab_size,
                                                      gen->config.embedding_dim);
        }
    }

    /* Xavier均匀分布初始化输出投影权重（替代随机缩放初始化） */
    /* fan_in=hidden_size, fan_out=vocab_size */
    for (size_t i = 0; i < proj_size; i++) {
        gen->output_proj_w[i] = dd_xavier_init(gen->config.hidden_size,
                                                gen->config.vocab_size);
    }
    for (size_t i = 0; i < gen->config.vocab_size; i++) {
        gen->output_proj_b[i] = 0.0f;
    }

    gen->initialized = 1;
/* 删除虚假的"已训练"标记。
     * Xavier随机初始化不代表已训练，未训练的随机权重不能产生有意义的回复。
     * 生成器创建后必须经过真实训练(前向传播+反向传播+多轮迭代)后才能标记为已训练。
     * 调用者在生成器未训练时应正确处理 -2 错误码并告知用户需要先训练模型。 */
    gen->is_trained = 0;
    return gen;
}

/* 标记对话生成器已训练（权重加载或训练完成后调用） */
void dialogue_gen_mark_trained(DialogueGenerator* gen)
{
    if (gen) gen->is_trained = 1;
}

/* 查询对话生成器训练状态 */
int dialogue_gen_is_trained(const DialogueGenerator* gen)
{
    return (gen && gen->is_trained) ? 1 : 0;
}

void dialogue_gen_free(DialogueGenerator* gen)
{
    if (!gen) return;
    if (gen->hidden_state) safe_free((void**)&gen->hidden_state);
    if (gen->cfc_cell) cfc_cell_free(gen->cfc_cell);
    if (gen->embedding_table) safe_free((void**)&gen->embedding_table);
    if (gen->vocab_codes) safe_free((void**)&gen->vocab_codes);
    if (gen->vocab_utf8) safe_free((void**)&gen->vocab_utf8);
    if (gen->is_chinese_char) safe_free((void**)&gen->is_chinese_char);
    if (gen->output_proj_w) safe_free((void**)&gen->output_proj_w);
    if (gen->output_proj_b) safe_free((void**)&gen->output_proj_b);
    if (gen->kb_embeddings) safe_free((void**)&gen->kb_embeddings);
    safe_free((void**)&gen);
}

/* 内部：单步生成一个token */
static int dialogue_gen_step(DialogueGenerator* gen,
                              float temperature, int top_k,
                              float* used_ids, int used_count,
                              int* output_token_id)
{
    if (!gen || !gen->initialized || !output_token_id) return -1;

/* 对话生成器训练状态检查
     * 未训练时输出投影权重为随机值，生成的token序列无意义。
     * 返回-2错误码通知调用者回退到模板匹配系统。 */
    if (!gen->is_trained) {
        return -2;
    }

    size_t vs = gen->config.vocab_size;
    size_t hs = gen->config.hidden_size;

    float* logits = (float*)safe_malloc(vs * sizeof(float));
    if (!logits) return -1;

    for (size_t i = 0; i < vs; i++) {
        float sum = gen->output_proj_b[i];
        for (size_t j = 0; j < hs; j++) {
            sum += gen->hidden_state[j] * gen->output_proj_w[j * vs + i];
        }
        logits[i] = sum;
    }

    if (temperature > 0.0f && temperature != 1.0f) {
        float inv_t = 1.0f / temperature;
        for (size_t i = 0; i < vs; i++) {
            logits[i] *= inv_t;
        }
    }

    for (int k = 0; k < used_count; k++) {
        int uid = (int)used_ids[k];
        if (uid >= 0 && uid < (int)vs) {
            logits[uid] -= gen->config.repetition_penalty;
        }
    }

    activation_softmax(logits, vs);

    if (top_k > 0 && top_k < (int)vs) {
        float* sorted = (float*)safe_malloc(vs * sizeof(float));
        int* indices = (int*)safe_malloc(vs * sizeof(int));
        if (!sorted || !indices) {
            safe_free((void**)&logits);
            safe_free((void**)&sorted);
            safe_free((void**)&indices);
            return -1;
        }

        memcpy(sorted, logits, vs * sizeof(float));
        for (size_t i = 0; i < vs; i++) indices[i] = (int)i;

        for (size_t i = 0; i < vs - 1; i++) {
            for (size_t j = i + 1; j < vs; j++) {
                if (sorted[j] > sorted[i]) {
                    float tf = sorted[i]; sorted[i] = sorted[j]; sorted[j] = tf;
                    int ti = indices[i]; indices[i] = indices[j]; indices[j] = ti;
                }
            }
        }

        float threshold = sorted[top_k - 1];
        float sum = 0.0f;
        for (size_t i = 0; i < vs; i++) {
            if (logits[i] < threshold) logits[i] = 0.0f;
            sum += logits[i];
        }
        if (sum > 1e-10f) {
            for (size_t i = 0; i < vs; i++) logits[i] /= sum;
        }

        safe_free((void**)&sorted);
        safe_free((void**)&indices);
    }

    float r = rand_float();
    float cum = 0.0f;
    int chosen = 0;
    for (size_t i = 0; i < vs; i++) {
        cum += logits[i];
        if (r < cum) {
            chosen = (int)i;
            break;
        }
    }

    *output_token_id = chosen;
    safe_free((void**)&logits);
    return 0;
}

/* 内部：使用CfC ODE演化隐藏状态（替代漏泄积分器） */
static void dialogue_leaky_integrate(DialogueGenerator* gen, const float* input)
{
    if (!gen || !input || !gen->cfc_cell) return;
    size_t ed = gen->config.embedding_dim;
    size_t hs = gen->config.hidden_size;
    float* cfc_input = (float*)safe_malloc(ed * sizeof(float));
    if (!cfc_input) return;
    memset(cfc_input, 0, ed * sizeof(float));
    size_t copy_dim = hs < ed ? hs : ed;
    memcpy(cfc_input, input, copy_dim * sizeof(float));
    cfc_cell_forward_with_dt(gen->cfc_cell, cfc_input,
                             gen->config.delta_t,
                             gen->hidden_state);
    safe_free((void**)&cfc_input);
}

int dialogue_gen_generate(DialogueGenerator* gen,
                           const float* context_embedding,
                           size_t context_size,
                           char* output_text, size_t max_output)
{
    if (!gen || !gen->initialized || !output_text || max_output == 0) return -1;

    memset(output_text, 0, max_output);

    size_t hs = gen->config.hidden_size;
    size_t ed = gen->config.embedding_dim;
    size_t vs = gen->config.vocab_size;

    memset(gen->hidden_state, 0, hs * sizeof(float));

    float* input_buf = (float*)safe_malloc(ed * sizeof(float));
    if (!input_buf) return -1;

    size_t copy_dim = context_size < ed ? context_size : ed;
    memcpy(input_buf, context_embedding, copy_dim * sizeof(float));
    if (copy_dim < ed) {
        memset(input_buf + copy_dim, 0, (ed - copy_dim) * sizeof(float));
    }

    dialogue_leaky_integrate(gen, input_buf);

    float used_ids[DG_MAX_GEN_TOKENS];
    int used_count = 0;
    int total_chars = 0;

    for (int step = 0; step < gen->config.max_generate_tokens; step++) {
        int token_id = gen->config.bos_token_id;
        if (dialogue_gen_step(gen, gen->config.temperature,
                               gen->config.top_k, used_ids, used_count,
                               &token_id) != 0) {
            break;
        }

        if (token_id == gen->config.eos_token_id) break;

        if (used_count < DG_MAX_GEN_TOKENS) {
            used_ids[used_count] = (float)token_id;
            used_count++;
        }

        if (token_id >= 0 && token_id < (int)vs) {
            const char* tok_str = gen->vocab_utf8 + (size_t)token_id * 8;
            size_t tlen = strlen(tok_str);
            if (total_chars + (int)tlen < (int)max_output - 1) {
                memcpy(output_text + total_chars, tok_str, tlen);
                total_chars += (int)tlen;
            }
        }

        float* token_emb = gen->embedding_table + (size_t)token_id * ed;
        dialogue_leaky_integrate(gen, token_emb);
    }

    safe_free((void**)&input_buf);
    return total_chars;
}

int dialogue_gen_generate_conditional(DialogueGenerator* gen,
                                       const float* context_embedding,
                                       size_t context_size,
                                       const float* condition_signal,
                                       size_t condition_size,
                                       float condition_strength,
                                       char* output_text, size_t max_output)
{
    if (!gen || !gen->initialized || !output_text || max_output == 0) return -1;
    if (!condition_signal || condition_size == 0) {
        return dialogue_gen_generate(gen, context_embedding,
                                      context_size, output_text, max_output);
    }

    memset(output_text, 0, max_output);

    size_t hs = gen->config.hidden_size;
    size_t ed = gen->config.embedding_dim;
    size_t vs = gen->config.vocab_size;

    memset(gen->hidden_state, 0, hs * sizeof(float));

    float* input_buf = (float*)safe_malloc(ed * sizeof(float));
    if (!input_buf) return -1;

    size_t copy_dim = context_size < ed ? context_size : ed;
    memcpy(input_buf, context_embedding, copy_dim * sizeof(float));
    if (copy_dim < ed) {
        memset(input_buf + copy_dim, 0, (ed - copy_dim) * sizeof(float));
    }

    size_t cond_copy = condition_size < ed ? condition_size : ed;
    for (size_t i = 0; i < cond_copy; i++) {
        input_buf[i] += condition_signal[i] * condition_strength;
    }

    dialogue_leaky_integrate(gen, input_buf);

    float used_ids[DG_MAX_GEN_TOKENS];
    int used_count = 0;
    int total_chars = 0;

    for (int step = 0; step < gen->config.max_generate_tokens; step++) {
        int token_id = gen->config.bos_token_id;

        float temp = gen->config.temperature;
        if (condition_strength > 0.5f) {
            temp = fmaxf(temp - condition_strength * 0.2f, 0.3f);
        }

        if (dialogue_gen_step(gen, temp, gen->config.top_k,
                               used_ids, used_count, &token_id) != 0) {
            break;
        }

        if (token_id == gen->config.eos_token_id) break;

        if (used_count < DG_MAX_GEN_TOKENS) {
            used_ids[used_count] = (float)token_id;
            used_count++;
        }

        if (token_id >= 0 && token_id < (int)vs) {
            const char* tok_str = gen->vocab_utf8 + (size_t)token_id * 8;
            size_t tlen = strlen(tok_str);
            if (total_chars + (int)tlen < (int)max_output - 1) {
                memcpy(output_text + total_chars, tok_str, tlen);
                total_chars += (int)tlen;
            }
        }

        float* token_emb = gen->embedding_table + (size_t)token_id * ed;
        float* cond_input = (float*)safe_malloc(ed * sizeof(float));
        if (cond_input) {
            memcpy(cond_input, token_emb, ed * sizeof(float));
            for (size_t i = 0; i < cond_copy && i < ed; i++) {
                cond_input[i] += condition_signal[i] * condition_strength * 0.5f;
            }
            dialogue_leaky_integrate(gen, cond_input);
            safe_free((void**)&cond_input);
        } else {
            dialogue_leaky_integrate(gen, token_emb);
        }
    }

    safe_free((void**)&input_buf);
    return total_chars;
}

int dialogue_gen_generate_knowledge(DialogueGenerator* gen,
                                     const float* context_embedding,
                                     size_t context_size,
                                     const float* knowledge_embedding,
                                     size_t knowledge_size,
                                     float knowledge_bias_strength,
                                     char* output_text, size_t max_output)
{
    if (!gen || !gen->initialized || !output_text || max_output == 0) return -1;

    memset(output_text, 0, max_output);

    size_t hs = gen->config.hidden_size;
    size_t ed = gen->config.embedding_dim;
    size_t vs = gen->config.vocab_size;

    memset(gen->hidden_state, 0, hs * sizeof(float));

    float* input_buf = (float*)safe_malloc(ed * sizeof(float));
    if (!input_buf) return -1;

    size_t copy_dim = context_size < ed ? context_size : ed;
    memcpy(input_buf, context_embedding, copy_dim * sizeof(float));
    if (copy_dim < ed) {
        memset(input_buf + copy_dim, 0, (ed - copy_dim) * sizeof(float));
    }

    if (knowledge_embedding && knowledge_size > 0) {
        size_t kb_copy = knowledge_size < ed ? knowledge_size : ed;
        for (size_t i = 0; i < kb_copy; i++) {
            input_buf[i] += knowledge_embedding[i] * knowledge_bias_strength;
        }
    }

    dialogue_leaky_integrate(gen, input_buf);

    float used_ids[DG_MAX_GEN_TOKENS];
    int used_count = 0;
    int total_chars = 0;

    for (int step = 0; step < gen->config.max_generate_tokens; step++) {
        int token_id = gen->config.bos_token_id;

        if (dialogue_gen_step(gen, gen->config.temperature,
                               gen->config.top_k, used_ids, used_count,
                               &token_id) != 0) {
            break;
        }

        if (token_id == gen->config.eos_token_id) break;

        if (used_count < DG_MAX_GEN_TOKENS) {
            used_ids[used_count] = (float)token_id;
            used_count++;
        }

        if (token_id >= 0 && token_id < (int)vs) {
            const char* tok_str = gen->vocab_utf8 + (size_t)token_id * 8;
            size_t tlen = strlen(tok_str);
            if (total_chars + (int)tlen < (int)max_output - 1) {
                memcpy(output_text + total_chars, tok_str, tlen);
                total_chars += (int)tlen;
            }
        }

        float* token_emb = gen->embedding_table + (size_t)token_id * ed;

        float* kb_input = (float*)safe_malloc(ed * sizeof(float));
        if (kb_input && knowledge_embedding) {
            memcpy(kb_input, token_emb, ed * sizeof(float));
            size_t kb_copy = knowledge_size < ed ? knowledge_size : ed;
            for (size_t i = 0; i < kb_copy; i++) {
                kb_input[i] += knowledge_embedding[i] * knowledge_bias_strength * 0.3f;
            }
            dialogue_leaky_integrate(gen, kb_input);
            safe_free((void**)&kb_input);
        } else {
            if (kb_input) safe_free((void**)&kb_input);
            dialogue_leaky_integrate(gen, token_emb);
        }
    }

    safe_free((void**)&input_buf);
    return total_chars;
}

void dialogue_gen_reset_state(DialogueGenerator* gen)
{
    if (!gen || !gen->initialized) return;
    memset(gen->hidden_state, 0, gen->config.hidden_size * sizeof(float));
}

void dialogue_gen_set_temperature(DialogueGenerator* gen, float temperature)
{
    if (!gen) return;
    gen->config.temperature = clip(temperature, 0.1f, 2.0f);
}

int dialogue_gen_get_config(const DialogueGenerator* gen,
                             DialogueGenConfig* config)
{
    if (!gen || !config) return -1;
    *config = gen->config;
    return 0;
}

int dialogue_gen_set_knowledge_base(DialogueGenerator* gen,
                                     const float* knowledge_embeddings,
                                     int num_entries, size_t embedding_dim)
{
    if (!gen || !knowledge_embeddings || num_entries <= 0) return -1;

    if (gen->kb_embeddings) {
        safe_free((void**)&gen->kb_embeddings);
    }

    size_t total = (size_t)num_entries * embedding_dim;
    gen->kb_embeddings = (float*)safe_malloc(total * sizeof(float));
    if (!gen->kb_embeddings) return -1;

    memcpy(gen->kb_embeddings, knowledge_embeddings, total * sizeof(float));
    gen->kb_num_entries = num_entries;
    gen->kb_embedding_dim = embedding_dim;
    gen->kb_initialized = 1;
    return 0;
}

/* ============================================================================
 * K-修复: 对话策略TD学习训练函数
 * 状态价值V(s)的时序差分学习 + 策略梯度更新
 * ============================================================================ */

int dialogue_deep_train_policy(DialogueProcessor* dp,
    const float* state_features, const float* next_state_features,
    float reward, int num_states, float learning_rate) {
    if (!dp || !state_features || !next_state_features || num_states <= 0 || learning_rate <= 0.0f) return -1;

    if (!dp->deep_initialized) return -1;

    DialogueBeliefState* belief = dp->deep_belief;
    DialoguePolicy* policy = dp->deep_policy;
    if (!belief || !policy) return -1;

    const int state_dim = (int)belief->belief_state_size;
    if (state_dim <= 0 || state_dim > 256 || !policy->policy_weights) return -1;

    float* v_current = (float*)safe_calloc((size_t)num_states, sizeof(float));
    float* v_next    = (float*)safe_calloc((size_t)num_states, sizeof(float));
    float* td_error  = (float*)safe_calloc((size_t)num_states, sizeof(float));
    if (!v_current || !v_next || !td_error) {
        safe_free((void**)&v_current); safe_free((void**)&v_next); safe_free((void**)&td_error);
        return -1;
    }

    /* 前向: 评估当前和下一状态的价值 */
    float gamma = 0.95f;
    for (int i = 0; i < num_states; i++) {
        float dot_cur = 0.0f, dot_next = 0.0f;
        for (int d = 0; d < state_dim && d < (int)policy->policy_hidden_size && d < (int)belief->belief_state_size; d++) {
            dot_cur  += state_features[i * state_dim + d] * policy->policy_weights[d];
            dot_next += next_state_features[i * state_dim + d] * policy->policy_weights[d];
        }
        v_current[i] = tanhf(dot_cur);
        v_next[i]    = tanhf(dot_next);
    }

    /* TD误差: δ = r + γ·V(s') - V(s) */
    float total_loss = 0.0f;
    for (int i = 0; i < num_states; i++) {
        td_error[i] = reward + gamma * v_next[i] - v_current[i];
        total_loss += td_error[i] * td_error[i];
    }
    total_loss /= (float)num_states;

    /* 策略梯度更新: Δθ = α·δ·∇_θ V(s) */
    float grad_clip = 1.0f;
    for (int d = 0; d < state_dim && d < (int)policy->policy_hidden_size && d < 256; d++) {
        float grad_sum = 0.0f;
        for (int i = 0; i < num_states; i++) {
            float dv = (1.0f - v_current[i] * v_current[i]) * state_features[i * state_dim + d];
            grad_sum += td_error[i] * dv;
        }
        grad_sum /= (float)num_states;

        if (fabsf(grad_sum) > grad_clip) {
            grad_sum = (grad_sum > 0) ? grad_clip : -grad_clip;
        }

        policy->policy_weights[d] += learning_rate * grad_sum;
    }

    /* 更新信念状态统计 */
    belief->belief_entropy = sqrtf(total_loss + 1e-8f);
    belief->state_coherence = 1.0f / (1.0f + total_loss);
    belief->belief_change_rate = total_loss;

    safe_free((void**)&v_current); safe_free((void**)&v_next); safe_free((void**)&td_error);
    return 0;
}

/* ============================================================================
 * dd_language_decode: 基于CfC状态演化 + 缩放点积注意力的上下文感知解码
 *
 * 核心流程：
 *   1. 将对话历史编码为键/值矩阵（通过词汇表嵌入）
 *   2. 当前CfC隐藏状态作为查询Q
 *   3. 缩放点积注意力：计算当前状态对历史上下文的关注分布
 *   4. 注意力输出注入CfC ODE进行状态演化
 *   5. 演化后的状态通过输出投影层生成下一个token
 *   6. 循环生成直到遇到EOS或达到最大长度
 *
 * 与dd_language_generate的区别：
 *   - 使用真实注意力机制替代简单加权
 *   - CfC ODE连续时间演化替代离散softmax采样
 *   - 上下文感知：解码时持续关注相关历史信息
 *
 * @param gen 对话生成器句柄
 * @param dialogue_history 对话历史文本输入（用户+系统消息串联）
 * @param max_history_len 历史文本最大长度
 * @param output_text [out] 生成回复文本缓冲区
 * @param max_output 输出缓冲区最大长度
 * @return 成功返回生成文本长度，失败返回-1
 * ============================================================================ */
int dd_language_decode(DialogueGenerator* gen,
                       const char* dialogue_history,
                       size_t max_history_len,
                       char* output_text, size_t max_output)
{
    if (!gen || !gen->initialized || !output_text || max_output == 0) return -1;

    memset(output_text, 0, max_output);

    size_t hs = gen->config.hidden_size;
    size_t ed = gen->config.embedding_dim;
    size_t vs = gen->config.vocab_size;

    /* 第一步：将对话历史编码为token序列，并获取嵌入向量 */
    #define DD_DECODE_MAX_HISTORY_TOKENS 512
    int history_ids[DD_DECODE_MAX_HISTORY_TOKENS];
    int history_count = 0;

    if (dialogue_history && max_history_len > 0) {
        text_to_token_ids(gen, dialogue_history, history_ids,
                          &history_count, DD_DECODE_MAX_HISTORY_TOKENS);
    }

    /* 构建对话历史的平均值作为历史上下文表示 */
    float* hist_avg_emb = (float*)safe_malloc(ed * sizeof(float));
    if (!hist_avg_emb) {
        safe_free((void**)&hist_avg_emb);
        return -1;
    }
    memset(hist_avg_emb, 0, ed * sizeof(float));

    if (history_count > 0) {
        for (int hi = 0; hi < history_count; hi++) {
            int tid = history_ids[hi];
            if (tid >= 0 && tid < (int)vs) {
                float* emb = gen->embedding_table + (size_t)tid * ed;
                for (size_t d = 0; d < ed; d++) {
                    hist_avg_emb[d] += emb[d];
                }
            }
        }
        float inv_count = 1.0f / (float)history_count;
        for (size_t d = 0; d < ed; d++) {
            hist_avg_emb[d] *= inv_count;
        }
    }

    /* 第二步：初始化CfC隐藏状态 */
    memset(gen->hidden_state, 0, hs * sizeof(float));

    /* 将历史上下文作为初始输入，通过CfC演化初始状态 */
    float* init_input = (float*)safe_malloc(ed * sizeof(float));
    if (!init_input) {
        safe_free((void**)&hist_avg_emb);
        return -1;
    }

    /* 使用历史平均嵌入作为初始上下文条件 */
    memcpy(init_input, hist_avg_emb, ed * sizeof(float));

    /* 通过CfC ODE进行状态演化 */
    dialogue_leaky_integrate(gen, init_input);

    /* 第三步：自回归解码循环（带注意力引导） */
    #define DD_DECODE_MAX_HISTORY_KEYS 64
    int num_keys = 0;
    int key_token_ids[DD_DECODE_MAX_HISTORY_KEYS];
    float* key_embeddings = (float*)safe_malloc((size_t)DD_DECODE_MAX_HISTORY_KEYS * ed * sizeof(float));
    if (!key_embeddings) {
        safe_free((void**)&init_input);
        safe_free((void**)&hist_avg_emb);
        return -1;
    }

    /* 从历史中提取关键token作为注意力键值对（间隔采样减少计算量） */
    int key_stride = history_count > DD_DECODE_MAX_HISTORY_KEYS
                         ? history_count / DD_DECODE_MAX_HISTORY_KEYS + 1 : 1;
    for (int hi = 0; hi < history_count && num_keys < DD_DECODE_MAX_HISTORY_KEYS; hi += key_stride) {
        int tid = history_ids[hi];
        if (tid >= 0 && tid < (int)vs) {
            key_token_ids[num_keys] = tid;
            memcpy(key_embeddings + (size_t)num_keys * ed,
                   gen->embedding_table + (size_t)tid * ed,
                   ed * sizeof(float));
            num_keys++;
        }
    }

    float used_ids[DG_MAX_GEN_TOKENS];
    int used_count = 0;
    int total_chars = 0;

    /* 隐藏状态作为查询Q */
    float* attn_output = (float*)safe_malloc(ed * sizeof(float));
    float* attn_weights_buf = (float*)safe_malloc((size_t)num_keys * sizeof(float));
    float* token_embed_buf = (float*)safe_malloc(ed * sizeof(float));

    if (!attn_output || !attn_weights_buf || !token_embed_buf) {
        safe_free((void**)&init_input);
        safe_free((void**)&hist_avg_emb);    safe_free((void**)&key_embeddings);
        safe_free((void**)&attn_output);     safe_free((void**)&attn_weights_buf);
        safe_free((void**)&token_embed_buf);
        return -1;
    }

    for (int step = 0; step < gen->config.max_generate_tokens; step++) {
        /* 3a. 如果有关键token，使用缩放点积注意力 */
        if (num_keys > 0) {
            /* Q = hidden_state（当前CfC状态的前ed维作为查询） */
            float* query = gen->hidden_state;
            size_t d_k = hs < ed ? hs : ed;
            /* 使用hidden_state的前d_k维作为查询 */
            memset(attn_output, 0, ed * sizeof(float));
            dd_scaled_dot_product_attention(query, key_embeddings,
                                             key_embeddings, num_keys,
                                             d_k, ed, attn_output,
                                             attn_weights_buf);

            /* 将注意力输出融入token嵌入选择 */
            memcpy(token_embed_buf, attn_output, ed * sizeof(float));
        } else {
            /* 无历史上下文时，使用当前状态直接投影 */
            for (size_t d = 0; d < ed && d < hs; d++) {
                token_embed_buf[d] = gen->hidden_state[d];
            }
            if (ed > hs) {
                memset(token_embed_buf + hs, 0, (ed - hs) * sizeof(float));
            }
        }

        /* 3b. 生成下一个token */
        int token_id = gen->config.bos_token_id;
        if (dialogue_gen_step(gen, gen->config.temperature,
                               gen->config.top_k, used_ids, used_count,
                               &token_id) != 0) {
            break;
        }

        if (token_id == gen->config.eos_token_id) break;

        if (used_count < DG_MAX_GEN_TOKENS) {
            used_ids[used_count] = (float)token_id;
            used_count++;
        }

        /* 3c. 输出token文本 */
        if (token_id >= 0 && token_id < (int)vs) {
            const char* tok_str = gen->vocab_utf8 + (size_t)token_id * 8;
            size_t tlen = strlen(tok_str);
            if (total_chars + (int)tlen < (int)max_output - 1) {
                memcpy(output_text + total_chars, tok_str, tlen);
                total_chars += (int)tlen;
            }
        }

        /* 3d. 使用生成的token嵌入 + 注意力引导更新CfC状态 */
        float* tok_emb = gen->embedding_table + (size_t)token_id * ed;

        /* 将token嵌入与注意力输出融合后输入CfC */
        float* next_input = (float*)safe_malloc(ed * sizeof(float));
        if (next_input) {
            for (size_t d = 0; d < ed; d++) {
                /* 70% token嵌入 + 30% 注意力上下文 */
                next_input[d] = 0.7f * tok_emb[d] + 0.3f * attn_output[d];
            }
            dialogue_leaky_integrate(gen, next_input);
            safe_free((void**)&next_input);
        } else {
            dialogue_leaky_integrate(gen, tok_emb);
        }
    }

    safe_free((void**)&init_input);
    safe_free((void**)&hist_avg_emb);    safe_free((void**)&key_embeddings);
    safe_free((void**)&attn_output);     safe_free((void**)&attn_weights_buf);
    safe_free((void**)&token_embed_buf);
    return total_chars;
}

/* ============================================================================
 * dd_save_pretrained: 将对话生成器模型权重保存到二进制文件
 *
 * 保存格式（二进制，小端序）：
 *   Header:
 *     uint64_t magic     = 0x4C4E4E4450204447 ("LNN DIALOG DEEP" 的扩展魔数)
 *     uint32_t version   = 1
 *     uint32_t vocab_size
 *     uint32_t embedding_dim
 *     uint32_t hidden_size
 *     uint32_t kb_num_entries
 *     uint32_t kb_embedding_dim
 *     uint32_t reserved[8]  未来扩展
 *   Body:
 *     float embedding_table[vocab_size * embedding_dim]
 *     float output_proj_w[hidden_size * vocab_size]
 *     float output_proj_b[vocab_size]
 *     float kb_embeddings[kb_num_entries * kb_embedding_dim] (if kb_initialized)
 *   Footer:
 *     uint32_t checksum   简单的累加校验和
 *
 * @param gen 对话生成器句柄
 * @param filepath 保存路径
 * @return 成功返回0，失败返回-1
 * ============================================================================ */

#define DD_MODEL_MAGIC 0x4C4E4E4450444447ULL  /* "LNNDPDG" */
#define DD_MODEL_VERSION 1

int dd_save_pretrained(DialogueGenerator* gen, const char* filepath)
{
    if (!gen || !gen->initialized || !filepath) return -1;

    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;

    size_t vs = gen->config.vocab_size;
    size_t ed = gen->config.embedding_dim;
    size_t hs = gen->config.hidden_size;
    int kb_num = gen->kb_initialized ? gen->kb_num_entries : 0;
    size_t kb_dim = gen->kb_initialized ? gen->kb_embedding_dim : 0;

    /* 写入头部 */
    uint64_t magic = DD_MODEL_MAGIC;
    uint32_t version = DD_MODEL_VERSION;
    uint32_t reserved[8] = {0};
    uint32_t checksum = 0;

    if (fwrite(&magic, sizeof(magic), 1, fp) != 1) goto fail;
    if (fwrite(&version, sizeof(version), 1, fp) != 1) goto fail;
    uint32_t vs32 = (uint32_t)vs, ed32 = (uint32_t)ed, hs32 = (uint32_t)hs;
    if (fwrite(&vs32, sizeof(vs32), 1, fp) != 1) goto fail;
    if (fwrite(&ed32, sizeof(ed32), 1, fp) != 1) goto fail;
    if (fwrite(&hs32, sizeof(hs32), 1, fp) != 1) goto fail;
    uint32_t kb_num32 = (uint32_t)kb_num, kb_dim32 = (uint32_t)kb_dim;
    if (fwrite(&kb_num32, sizeof(kb_num32), 1, fp) != 1) goto fail;
    if (fwrite(&kb_dim32, sizeof(kb_dim32), 1, fp) != 1) goto fail;
    if (fwrite(reserved, sizeof(uint32_t), 8, fp) != 8) goto fail;

    /* 写入embedding_table */
    size_t emb_count = vs * ed;
    for (size_t i = 0; i < emb_count; i++) {
        float v = gen->embedding_table[i];
        checksum += (uint32_t)(*(uint32_t*)&v);
    }
    if (fwrite(gen->embedding_table, sizeof(float), emb_count, fp) != emb_count) goto fail;

    /* 写入output_proj_w */
    size_t proj_w_count = hs * vs;
    for (size_t i = 0; i < proj_w_count; i++) {
        float v = gen->output_proj_w[i];
        checksum += (uint32_t)(*(uint32_t*)&v);
    }
    if (fwrite(gen->output_proj_w, sizeof(float), proj_w_count, fp) != proj_w_count) goto fail;

    /* 写入output_proj_b */
    for (size_t i = 0; i < vs; i++) {
        float v = gen->output_proj_b[i];
        checksum += (uint32_t)(*(uint32_t*)&v);
    }
    if (fwrite(gen->output_proj_b, sizeof(float), vs, fp) != vs) goto fail;

    /* 写入知识库嵌入（如果有） */
    if (kb_num > 0 && gen->kb_embeddings) {
        size_t kb_count = (size_t)kb_num * kb_dim;
        for (size_t i = 0; i < kb_count; i++) {
            float v = gen->kb_embeddings[i];
            checksum += (uint32_t)(*(uint32_t*)&v);
        }
        if (fwrite(gen->kb_embeddings, sizeof(float), kb_count, fp) != kb_count) goto fail;
    }

    /* 写入尾部校验和 */
    if (fwrite(&checksum, sizeof(checksum), 1, fp) != 1) goto fail;

    fclose(fp);
    return 0;

fail:
    fclose(fp);
    return -1;
}

/* ============================================================================
 * dialogue_gen_is_initialized: 检查对话生成器是否已初始化
 *
 * @param gen 对话生成器句柄
 * @return int 已初始化返回1，否则返回0
 * ============================================================================ */
int dialogue_gen_is_initialized(const DialogueGenerator* gen)
{
    if (!gen) return 0;
    return gen->initialized;
}

/* ============================================================================
 * dialogue_generate_with_cfc_ode: 使用CfC ODE对话生成器生成回复（集成入口）
 *
 * 作为dialogue_process_input的CfC ODE优先路径。
 * 从DialogueProcessor中提取对话上下文信息，调用dg_generate_response
 * 进行端到端的CfC ODE神经对话生成。
 *
 * 步骤：
 *   1. 检查processor->generator是否可用
 *   2. 从context中提取对话历史文本
 *   3. 调用dg_generate_response执行CfC ODE编码-解码
 *
 * @param processor 对话处理器句柄
 * @param user_input 用户输入文本
 * @param input_length 输入文本长度
 * @param context 对话上下文（可为NULL）
 * @param output_text [out] 生成的回复文本缓冲区
 * @param max_output 输出缓冲区最大字节数
 * @param temperature 生成温度（0.1-2.0）
 * @param top_k Top-K采样参数
 * @param confidence [out] 生成置信度输出
 * @return int 成功返回生成文本长度，失败返回-1
 * ============================================================================ */
int dialogue_generate_with_cfc_ode(DialogueProcessor* processor,
                                   const char* user_input,
                                   size_t input_length,
                                   const DialogueContext* context,
                                   char* output_text,
                                   size_t max_output,
                                   float temperature,
                                   int top_k,
                                   float* confidence)
{
    if (!processor || !user_input || input_length == 0 ||
        !output_text || max_output == 0) {
        if (confidence) *confidence = 0.0f;
        return -1;
    }

    /* 检查CfC对话生成器是否可用 */
    if (!processor->generator || !processor->generator->initialized) {
        if (confidence) *confidence = 0.0f;
        return -1;
    }

    /* 从对话上下文中提取历史文本 */
    char* history_text = NULL;
    size_t history_length = 0;

    if (context && context->num_messages > 0) {
        /* 预分配缓冲区：每条消息平均64字节 */
        size_t hist_buf_size = context->num_messages * 128 + 256;
        history_text = (char*)safe_malloc(hist_buf_size);
        if (history_text) {
            memset(history_text, 0, hist_buf_size);
            size_t pos = 0;

            for (size_t i = 0; i < context->num_messages && pos < hist_buf_size - 64; i++) {
                const DialogueMessage* msg = &context->messages[i];
                if (!msg->text || msg->length == 0) continue;

                /* 格式: [用户/系统]: 消息文本 | */
                if (msg->role == 0) {
                    if (pos + 4 < hist_buf_size) {
                        memcpy(history_text + pos, "用户:", strlen("用户:"));
                        pos += strlen("用户:");
                    }
                } else {
                    if (pos + 4 < hist_buf_size) {
                        memcpy(history_text + pos, "系统:", strlen("系统:"));
                        pos += strlen("系统:");
                    }
                }

                size_t copy_len = msg->length;
                if (pos + copy_len >= hist_buf_size) {
                    copy_len = hist_buf_size - pos - 2;
                }
                if (copy_len > 0) {
                    memcpy(history_text + pos, msg->text, copy_len);
                    pos += copy_len;
                }

                /* 分隔符 */
                if (pos + 3 < hist_buf_size) {
                    history_text[pos++] = ' ';
                    history_text[pos++] = '|';
                    history_text[pos++] = ' ';
                }
            }
            history_text[pos] = '\0';
            history_length = pos;
        }
    }

    /* 调用CfC ODE神经对话生成核心函数 */
    int result = dg_generate_response(processor->generator,
                                      user_input, input_length,
                                      history_text, history_length,
                                      output_text, max_output,
                                      temperature, top_k,
                                      confidence);

    if (history_text) {
        safe_free((void**)&history_text);
    }

    return result;
}

/* ============================================================================
 * dg_generate_response: 基于CfC ODE的神经对话生成（核心函数）
 *
 * 上下文编码流程：
 *   用户输入 → 文本向量化 → CfC ODE编码 → 对话状态h
 *   知识检索 → 事实向量嵌入 → 与h融合 → 增强状态h'
 *   增强状态h' → 自回归解码 → 逐词生成回复
 *
 * 回复生成优先级：
 *   1. 知识库检索增强（检索结果→注入生成偏置）
 *   2. CfC ODE神经生成（模型权重已加载时）
 *   3. 模板匹配回退（无模型时，由上层调用者处理）
 *
 * @param gen 对话生成器句柄（必须已初始化CfC单元）
 * @param user_input 用户输入文本
 * @param input_length 输入文本长度
 * @param dialogue_history 对话历史文本（可为NULL）
 * @param history_length 对话历史长度
 * @param output_text [out] 生成的回复文本缓冲区
 * @param max_output 输出缓冲区最大字节数
 * @param temperature 生成温度（0.1-2.0）
 * @param top_k Top-K采样参数
 * @param confidence [out] 生成置信度输出
 * @return int 成功返回生成文本长度，失败返回-1
 * ============================================================================ */
int dg_generate_response(DialogueGenerator* gen,
                         const char* user_input,
                         size_t input_length,
                         const char* dialogue_history,
                         size_t history_length,
                         char* output_text,
                         size_t max_output,
                         float temperature,
                         int top_k,
                         float* confidence)
{
    if (!gen || !gen->initialized || !user_input || input_length == 0 ||
        !output_text || max_output == 0) {
        if (confidence) *confidence = 0.0f;
        return -1;
    }

    if (temperature < 0.1f) temperature = 0.8f;
    if (temperature > 2.0f) temperature = 2.0f;
    if (top_k <= 0) top_k = 40;
    if (top_k > (int)gen->config.vocab_size) top_k = (int)gen->config.vocab_size;

    memset(output_text, 0, max_output);

    size_t hs = gen->config.hidden_size;
    size_t ed = gen->config.embedding_dim;
    size_t vs = gen->config.vocab_size;

    /* ============================================================
     * 阶段1：文本向量化——将用户输入编码为嵌入向量序列
     * ============================================================ */
    #define DG_MAX_INPUT_TOKENS 256
    int input_ids[DG_MAX_INPUT_TOKENS];
    int input_count = 0;
    text_to_token_ids(gen, user_input, input_ids, &input_count, DG_MAX_INPUT_TOKENS);

    if (input_count == 0) {
        if (confidence) *confidence = 0.0f;
        return -1;
    }

    /* 构建用户输入的平均嵌入向量 */
    float* input_avg_emb = (float*)safe_malloc(ed * sizeof(float));
    if (!input_avg_emb) {
        if (confidence) *confidence = 0.0f;
        return -1;
    }
    memset(input_avg_emb, 0, ed * sizeof(float));

    for (int i = 0; i < input_count; i++) {
        int tid = input_ids[i];
        if (tid >= 0 && tid < (int)vs) {
            float* emb = gen->embedding_table + (size_t)tid * ed;
            for (size_t d = 0; d < ed; d++) {
                input_avg_emb[d] += emb[d];
            }
        }
    }
    float inv_input = 1.0f / (float)input_count;
    for (size_t d = 0; d < ed; d++) {
        input_avg_emb[d] *= inv_input;
    }

    /* ============================================================
     * 阶段2：对话历史编码（如果有历史记录）
     * ============================================================ */
    #define DG_MAX_HISTORY_TOKENS 512
    int hist_ids[DG_MAX_HISTORY_TOKENS];
    int hist_count = 0;
    float* hist_avg_emb = NULL;

    if (dialogue_history && history_length > 0) {
        text_to_token_ids(gen, dialogue_history, hist_ids,
                          &hist_count, DG_MAX_HISTORY_TOKENS);

        if (hist_count > 0) {
            hist_avg_emb = (float*)safe_malloc(ed * sizeof(float));
            if (hist_avg_emb) {
                memset(hist_avg_emb, 0, ed * sizeof(float));
                for (int i = 0; i < hist_count; i++) {
                    int tid = hist_ids[i];
                    if (tid >= 0 && tid < (int)vs) {
                        float* emb = gen->embedding_table + (size_t)tid * ed;
                        for (size_t d = 0; d < ed; d++) {
                            hist_avg_emb[d] += emb[d];
                        }
                    }
                }
                float inv_hist = 1.0f / (float)hist_count;
                for (size_t d = 0; d < ed; d++) {
                    hist_avg_emb[d] *= inv_hist;
                }
            }
        }
    }

    /* ============================================================
     * 阶段3：CfC ODE编码——从文本嵌入生成对话状态h
     *
     * 将用户输入嵌入和历史嵌入融合后通过CfC连续时间ODE演化，
     * 生成隐藏状态h作为对话的语义编码表示。
     * ============================================================ */
    memset(gen->hidden_state, 0, hs * sizeof(float));

    /* 融合用户输入和历史嵌入作为CfC的初始输入 */
    float* cfc_input = (float*)safe_malloc(ed * sizeof(float));
    if (!cfc_input) {
        safe_free((void**)&input_avg_emb);
        if (hist_avg_emb) safe_free((void**)&hist_avg_emb);
        if (confidence) *confidence = 0.0f;
        return -1;
    }

    /* 混合比例：用户输入70%，对话历史30%（如果有历史） */
    if (hist_avg_emb && hist_count > 0) {
        for (size_t d = 0; d < ed; d++) {
            cfc_input[d] = input_avg_emb[d] * 0.7f + hist_avg_emb[d] * 0.3f;
        }
    } else {
        memcpy(cfc_input, input_avg_emb, ed * sizeof(float));
    }

    /* 通过CfC ODE进行连续时间状态演化 */
    dialogue_leaky_integrate(gen, cfc_input);

    safe_free((void**)&cfc_input);

    /* 保存编码后的对话状态h（后续知识增强和自回归解码都需要） */
    float* dialogue_state_h = (float*)safe_malloc(hs * sizeof(float));
    if (!dialogue_state_h) {
        safe_free((void**)&input_avg_emb);
        if (hist_avg_emb) safe_free((void**)&hist_avg_emb);
        if (confidence) *confidence = 0.0f;
        return -1;
    }
    memcpy(dialogue_state_h, gen->hidden_state, hs * sizeof(float));

    /* ============================================================
     * 阶段4：知识库检索增强
     *
     * 从知识库中检索与用户输入最相关的5个事实，
     * 将检索到的知识嵌入向量融合到对话状态中。
     *
     * 检索方法：计算用户输入平均嵌入与每个知识条目的余弦相似度，
     * 选取Top-5最相关条目，进行加权平均融合。
     * ============================================================ */
    int kb_used = 0;
    float* kb_enhanced_state = (float*)safe_malloc(hs * sizeof(float));
    if (!kb_enhanced_state) {
        safe_free((void**)&input_avg_emb);
        safe_free((void**)&dialogue_state_h);
        if (hist_avg_emb) safe_free((void**)&hist_avg_emb);
        if (confidence) *confidence = 0.0f;
        return -1;
    }
    memcpy(kb_enhanced_state, dialogue_state_h, hs * sizeof(float));

    if (gen->kb_initialized && gen->kb_embeddings && gen->kb_num_entries > 0) {
        size_t kb_dim = gen->kb_embedding_dim;
        size_t kb_copy_min = kb_dim < ed ? kb_dim : ed;

        /* 分配相似度数组 */
        float* kb_similarities = (float*)safe_malloc((size_t)gen->kb_num_entries * sizeof(float));
        if (kb_similarities) {
            /* 计算用户输入嵌入与每个知识条目的余弦相似度 */
            for (int ki = 0; ki < gen->kb_num_entries; ki++) {
                float* kb_emb = gen->kb_embeddings + (size_t)ki * kb_dim;
                float dot = 0.0f;
                float norm_kb = 0.0f;
                float norm_input = 0.0f;

                for (size_t d = 0; d < kb_copy_min; d++) {
                    dot += kb_emb[d] * input_avg_emb[d];
                    norm_kb += kb_emb[d] * kb_emb[d];
                    norm_input += input_avg_emb[d] * input_avg_emb[d];
                }

                float denom = sqrtf(norm_kb * norm_input);
                kb_similarities[ki] = (denom > 1e-8f) ? (dot / denom) : 0.0f;
            }

            /* 选择Top-5最相关的知识条目 */
            #define KB_TOP_K 5
            int top_indices[KB_TOP_K];
            float top_sims[KB_TOP_K];
            for (int tk = 0; tk < KB_TOP_K; tk++) {
                top_indices[tk] = -1;
                top_sims[tk] = -1.0f;
            }

            for (int ki = 0; ki < gen->kb_num_entries; ki++) {
                float sim = kb_similarities[ki];
                if (sim <= 0.0f) continue;

                /* 插入排序到TopK */
                for (int tk = 0; tk < KB_TOP_K; tk++) {
                    if (sim > top_sims[tk]) {
                        /* 后移后面的元素 */
                        for (int ts = KB_TOP_K - 1; ts > tk; ts--) {
                            top_indices[ts] = top_indices[ts - 1];
                            top_sims[ts] = top_sims[ts - 1];
                        }
                        top_indices[tk] = ki;
                        top_sims[tk] = sim;
                        break;
                    }
                }
            }

            /* 构建知识增强向量：Top-K条目的加权平均嵌入 */
            int valid_kb_count = 0;
            float total_kb_weight = 0.0f;
            float* kb_augment_emb = (float*)safe_malloc(kb_dim * sizeof(float));
            if (kb_augment_emb) {
                memset(kb_augment_emb, 0, kb_dim * sizeof(float));

                for (int tk = 0; tk < KB_TOP_K; tk++) {
                    if (top_indices[tk] >= 0 && top_sims[tk] > 0.0f) {
                        float* kb_emb = gen->kb_embeddings + (size_t)top_indices[tk] * kb_dim;
                        for (size_t d = 0; d < kb_dim; d++) {
                            kb_augment_emb[d] += kb_emb[d] * top_sims[tk];
                        }
                        total_kb_weight += top_sims[tk];
                        valid_kb_count++;
                    }
                }

                if (valid_kb_count > 0 && total_kb_weight > 1e-8f) {
                    /* 归一化加权平均 */
                    for (size_t d = 0; d < kb_dim; d++) {
                        kb_augment_emb[d] /= total_kb_weight;
                    }

                    /* 知识融合：h' = h * 0.7 + kb_embedding * 0.3 */
                    size_t kfuse_dim = kb_dim < hs ? kb_dim : hs;
                    for (size_t d = 0; d < kfuse_dim; d++) {
                        kb_enhanced_state[d] = dialogue_state_h[d] * 0.7f +
                                                kb_augment_emb[d] * 0.3f;
                    }
                    kb_used = 1;
                }

                safe_free((void**)&kb_augment_emb);
            }

            safe_free((void**)&kb_similarities);
        }
    }

    /* 用增强后的状态替换隐藏状态 */
    memcpy(gen->hidden_state, kb_enhanced_state, hs * sizeof(float));
    safe_free((void**)&kb_enhanced_state);

    /* ============================================================
     * 阶段5：自回归解码——逐token生成回复文本
     *
     * 从增强的对话状态开始，通过输出投影层计算logits，
     * 采样下一个token，将其嵌入反馈到CfC ODE，迭代生成。
     * ============================================================ */
    float used_ids[DG_MAX_GEN_TOKENS];
    int used_count = 0;
    int total_chars = 0;

    for (int step = 0; step < gen->config.max_generate_tokens; step++) {
        int token_id = gen->config.bos_token_id;

        /* 计算输出logits: logits = hidden_state × output_proj_w + output_proj_b */
        float* logits = (float*)safe_malloc(vs * sizeof(float));
        if (!logits) break;

        for (size_t i = 0; i < vs; i++) {
            float sum = gen->output_proj_b[i];
            for (size_t j = 0; j < hs; j++) {
                sum += gen->hidden_state[j] * gen->output_proj_w[j * vs + i];
            }
            logits[i] = sum;
        }

        /* 温度缩放 */
        if (temperature > 0.0f && fabsf(temperature - 1.0f) > 1e-6f) {
            float inv_t = 1.0f / temperature;
            for (size_t i = 0; i < vs; i++) {
                logits[i] *= inv_t;
            }
        }

        /* 重复惩罚：降低已生成token的得分 */
        for (int k = 0; k < used_count; k++) {
            int uid = (int)used_ids[k];
            if (uid >= 0 && uid < (int)vs) {
                logits[uid] -= gen->config.repetition_penalty;
            }
        }

        /* Softmax归一化 */
        activation_softmax(logits, vs);

        /* Top-K采样过滤 */
        if (top_k > 0 && top_k < (int)vs) {
            /* 找到第top_k大的概率值作为阈值 */
            float* sorted = (float*)safe_malloc(vs * sizeof(float));
            if (sorted) {
                memcpy(sorted, logits, vs * sizeof(float));
                /* 简单选择排序找到top_k个最大值 */
                for (size_t i = 0; i < (size_t)top_k && i < vs; i++) {
                    size_t max_idx = i;
                    for (size_t j = i + 1; j < vs; j++) {
                        if (sorted[j] > sorted[max_idx]) max_idx = j;
                    }
                    if (max_idx != i) {
                        float tmp = sorted[i]; sorted[i] = sorted[max_idx]; sorted[max_idx] = tmp;
                    }
                }
                float threshold = sorted[top_k - 1];

                /* 过滤并重新归一化 */
                float sum = 0.0f;
                for (size_t i = 0; i < vs; i++) {
                    if (logits[i] < threshold) logits[i] = 0.0f;
                    sum += logits[i];
                }
                if (sum > 1e-10f) {
                    for (size_t i = 0; i < vs; i++) logits[i] /= sum;
                }

                safe_free((void**)&sorted);
            }
        }

        /* 基于概率随机采样 */
        float r = rand_float();
        float cum = 0.0f;
        for (size_t i = 0; i < vs; i++) {
            cum += logits[i];
            if (r < cum) {
                token_id = (int)i;
                break;
            }
        }

        safe_free((void**)&logits);

        /* 遇到结束token则停止 */
        if (token_id == gen->config.eos_token_id) break;

        /* 记录已使用ID防止重复 */
        if (used_count < DG_MAX_GEN_TOKENS) {
            used_ids[used_count] = (float)token_id;
            used_count++;
        }

        /* 将token文本写入输出缓冲区 */
        if (token_id >= 0 && token_id < (int)vs) {
            const char* tok_str = gen->vocab_utf8 + (size_t)token_id * 8;
            size_t tlen = strlen(tok_str);
            if (total_chars + (int)tlen < (int)max_output - 1) {
                memcpy(output_text + total_chars, tok_str, tlen);
                total_chars += (int)tlen;
            }
        }

        /* 将当前token嵌入反馈到CfC ODE进行状态更新 */
        float* token_emb = gen->embedding_table + (size_t)token_id * ed;
        dialogue_leaky_integrate(gen, token_emb);
    }

    /* ============================================================
     * 阶段6：计算置信度并清理
     * ============================================================ */
    if (confidence) {
        if (total_chars > 0) {
/* 置信度从CfC状态向量的输出熵动态计算。 */
            *confidence = 0.0f;
            float* final_state = dg_get_current_state(gen);
            if (final_state && gen->config.hidden_size > 0) {
                float entropy = 0.0f;
                for (size_t i = 0; i < gen->config.hidden_size; i++) {
                    float p = fabsf(final_state[i]);
                    if (p > 1e-9f && p < 1.0f) entropy -= p * logf(p);
                }
                float max_entropy = logf((float)gen->config.hidden_size) + 1e-9f;
                *confidence = 1.0f - (entropy / max_entropy);
            }
            if (kb_used) *confidence *= 1.10f;
            if (total_chars < 4) *confidence *= 0.80f;
            if (*confidence > 0.95f) *confidence = 0.95f;
            if (*confidence < 0.20f) *confidence = 0.20f;
        } else {
            *confidence = 0.0f;
        }
    }

    safe_free((void**)&input_avg_emb);
    safe_free((void**)&dialogue_state_h);
    if (hist_avg_emb) safe_free((void**)&hist_avg_emb);

    return total_chars;
}

/* ============================================================================
 * dg_save_dialogue_model: 保存CfC对话模型权重到二进制文件
 *
 * 保存格式与dd_save_pretrained一致：
 *   魔数(8B) + 版本(4B) + vs32/ed32/hs32/kb_num32/kb_dim32 + reserved[8]
 *   + embedding_table + output_proj_w + output_proj_b + kb_embeddings
 *   + 尾部校验和(4B)
 *
 * @param gen 对话生成器句柄
 * @param filepath 保存路径
 * @return int 成功返回0，失败返回-1
 * ============================================================================ */
int dg_save_dialogue_model(DialogueGenerator* gen, const char* filepath)
{
    if (!gen || !gen->initialized || !filepath) return -1;

    /* 复用已有的dd_save_pretrained实现 */
    return dd_save_pretrained(gen, filepath);
}

/* ============================================================================
 * dg_load_dialogue_model: 从二进制文件加载CfC对话模型权重
 *
 * 加载由dg_save_dialogue_model保存的模型权重。
 * 验证：魔数匹配、版本兼容性、维度匹配、校验和完整性。
 *
 * @param gen 对话生成器句柄（必须已初始化词汇表）
 * @param filepath 模型文件路径
 * @return int 成功返回0，失败返回-1
 * ============================================================================ */
int dg_load_dialogue_model(DialogueGenerator* gen, const char* filepath)
{
    if (!gen || !gen->initialized || !filepath) return -1;

    /* 复用已有的dd_load_pretrained实现 */
    return dd_load_pretrained(gen, filepath);
}

/* ============================================================================
 * dd_load_pretrained: 从二进制文件加载预训练模型权重
 *
 * 读取dd_save_pretrained保存的二进制模型文件。
 * 验证魔数、版本和校验和，然后将权重加载到生成器中。
 *
 * @param gen 对话生成器句柄（必须已创建并初始化vocab）
 * @param filepath 模型文件路径
 * @return 成功返回0，失败返回-1
 * ============================================================================ */

int dd_load_pretrained(DialogueGenerator* gen, const char* filepath)
{
    if (!gen || !gen->initialized || !filepath) return -1;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;

    /* 读取头部 */
    uint64_t magic;
    uint32_t version, vs32, ed32, hs32, kb_num32, kb_dim32;
    uint32_t reserved[8];
    uint32_t checksum_file;

    if (fread(&magic, sizeof(magic), 1, fp) != 1) goto fail;
    if (magic != DD_MODEL_MAGIC) {
        /* 魔数不匹配，文件格式错误 */
        goto fail;
    }

    if (fread(&version, sizeof(version), 1, fp) != 1) goto fail;
    if (version > DD_MODEL_VERSION) {
        /* 版本过高，不支持 */
        goto fail;
    }

    if (fread(&vs32, sizeof(vs32), 1, fp) != 1) goto fail;
    if (fread(&ed32, sizeof(ed32), 1, fp) != 1) goto fail;
    if (fread(&hs32, sizeof(hs32), 1, fp) != 1) goto fail;
    if (fread(&kb_num32, sizeof(kb_num32), 1, fp) != 1) goto fail;
    if (fread(&kb_dim32, sizeof(kb_dim32), 1, fp) != 1) goto fail;
    if (fread(reserved, sizeof(uint32_t), 8, fp) != 8) goto fail;

    /* 验证维度匹配 */
    size_t vs = gen->config.vocab_size;
    size_t ed = gen->config.embedding_dim;
    size_t hs = gen->config.hidden_size;

    if ((size_t)vs32 != vs || (size_t)ed32 != ed || (size_t)hs32 != hs) {
        /* 维度不匹配，无法加载 */
        goto fail;
    }

    /* 计算校验和并读取权重 */
    uint32_t checksum_calc = 0;

    /* 读取embedding_table */
    size_t emb_count = vs * ed;
    if (fread(gen->embedding_table, sizeof(float), emb_count, fp) != emb_count) goto fail;
    for (size_t i = 0; i < emb_count; i++) {
        checksum_calc += (uint32_t)(*(uint32_t*)&gen->embedding_table[i]);
    }

    /* 读取output_proj_w */
    size_t proj_w_count = hs * vs;
    if (fread(gen->output_proj_w, sizeof(float), proj_w_count, fp) != proj_w_count) goto fail;
    for (size_t i = 0; i < proj_w_count; i++) {
        checksum_calc += (uint32_t)(*(uint32_t*)&gen->output_proj_w[i]);
    }

    /* 读取output_proj_b */
    if (fread(gen->output_proj_b, sizeof(float), vs, fp) != vs) goto fail;
    for (size_t i = 0; i < vs; i++) {
        checksum_calc += (uint32_t)(*(uint32_t*)&gen->output_proj_b[i]);
    }

    /* 读取知识库嵌入（如果有） */
    if (kb_num32 > 0) {
        if (!gen->kb_embeddings) {
            size_t kb_total = (size_t)kb_num32 * (size_t)kb_dim32;
            gen->kb_embeddings = (float*)safe_malloc(kb_total * sizeof(float));
            if (!gen->kb_embeddings) goto fail;
        }
        size_t kb_count = (size_t)kb_num32 * (size_t)kb_dim32;
        if (fread(gen->kb_embeddings, sizeof(float), kb_count, fp) != kb_count) goto fail;
        gen->kb_num_entries = (int)kb_num32;
        gen->kb_embedding_dim = (size_t)kb_dim32;
        gen->kb_initialized = 1;
        for (size_t i = 0; i < kb_count; i++) {
            checksum_calc += (uint32_t)(*(uint32_t*)&gen->kb_embeddings[i]);
        }
    }

    /* 读取校验和 */
    if (fread(&checksum_file, sizeof(checksum_file), 1, fp) != 1) goto fail;

    /* 验证校验和 */
    if (checksum_calc != checksum_file) {
        /* 校验和不匹配，文件可能损坏 */
        goto fail;
    }

    fclose(fp);
    return 0;

fail:
    fclose(fp);
    return -1;
}


