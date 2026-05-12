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

int multi_turn_reasoner_reason(MultiTurnReasoner* reasoner,
                                float* reasoning_output, size_t output_size)
{
    if (!reasoner || !reasoning_output || output_size == 0) return -1;

    memset(reasoning_output, 0, output_size * sizeof(float));

    int nw = reasoner->num_tracks < MTR_MAX_TRACKING ? reasoner->num_tracks : MTR_MAX_TRACKING;
    if (nw == 0) return 0;

    size_t copy_size = output_size < reasoner->state_size
                           ? output_size : reasoner->state_size;

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

    for (size_t i = 0; i < copy_size && i < output_size; i++) {
        reasoning_output[i] += 0.3f * reasoner->trace_state[i];
    }

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
};

/* ============================================================================
 * 内部：词汇表构建
 * ============================================================================ */

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

    for (size_t i = 0; i < emb_size; i++) {
        gen->embedding_table[i] = (rand_float() - 0.5f) * 0.1f;
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

    for (size_t i = 0; i < proj_size; i++) {
        gen->output_proj_w[i] = (rand_float() - 0.5f) * 0.02f;
    }
    for (size_t i = 0; i < gen->config.vocab_size; i++) {
        gen->output_proj_b[i] = 0.0f;
    }

    if (dialogue_build_vocab(gen) != 0) {
        dialogue_gen_free(gen);
        return NULL;
    }

    gen->initialized = 1;
    return gen;
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


