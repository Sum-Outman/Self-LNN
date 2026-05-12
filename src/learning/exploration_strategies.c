/**
 * @file exploration_strategies.c
 * @brief 深度探索策略系统完整实现
 *
 * 实现三种核心探索算法：
 * 1. ICM(Intrinsic Curiosity Module) — 内在好奇心模块（A05.1.3.1）
 * 2. RND(Random Network Distillation) — 随机网络蒸馏（A05.1.3.2）
 * 3. Go-Explore — 先探索后利用（A05.1.3.3）
 */

#include "selflnn/learning/exploration_strategies.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========== OU过程锁（保护函数内static ou_state/ou_dim） ========== */
#ifdef _WIN32
#include <windows.h>
static CRITICAL_SECTION g_ou_lock;
static int g_ou_lock_init = 0;
#define OU_LOCK() do { if (!g_ou_lock_init) { InitializeCriticalSection(&g_ou_lock); g_ou_lock_init = 1; } EnterCriticalSection(&g_ou_lock); } while(0)
#define OU_UNLOCK() LeaveCriticalSection(&g_ou_lock)
#else
#include <pthread.h>
static pthread_mutex_t g_ou_lock = PTHREAD_MUTEX_INITIALIZER;
#define OU_LOCK() pthread_mutex_lock(&g_ou_lock)
#define OU_UNLOCK() pthread_mutex_unlock(&g_ou_lock)
#endif

#define EXPLORE_EPSILON 1e-8f
#define EXPLORE_RAND_MAX 2147483647

/* 内部随机生成器 */
static unsigned int explore_rand_seed = 12345;

static float explore_rand_float(void) {
    explore_rand_seed = explore_rand_seed * 1103515245 + 12345;
    return (float)((explore_rand_seed >> 16) & 0x7FFF) / 32768.0f;
}

static float explore_randn(float std) {
    float u1 = explore_rand_float();
    float u2 = explore_rand_float();
    if (u1 < EXPLORE_EPSILON) u1 = EXPLORE_EPSILON;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2) * std;
}

/* ============================================================================
 * 通用结构
 * ============================================================================ */

/* ICM网络层 */
typedef struct {
    float* w;           /* 权重 */
    float* b;           /* 偏置 */
    int in_dim;
    int out_dim;
} ICMLayer;

/* RND网络 */
typedef struct {
    float* w1;          /* 第一层权重 */
    float* b1;          /* 第一层偏置 */
    float* w2;          /* 第二层权重 */
    float* b2;          /* 第二层偏置 */
    int in_dim;
    int hidden_dim;
    int out_dim;
} RNDNetwork;

/* Go-Explore存档单元 */
typedef struct {
    float* state;
    int state_dim;
    float* trajectory_actions;
    int trajectory_length;
    float best_reward;
    int visit_count;
    int is_robustified;
} GoExploreCell;

struct ExploreState {
    int strategy_type;

    /* ICM状态 */
    ICMConfig icm_config;
    ICMLayer* icm_encoder;
    ICMLayer* icm_inverse;
    ICMLayer* icm_forward;
    float* icm_embed_s;
    float* icm_embed_s_next;
    float* icm_pred_s_next;

    /* RND状态 */
    RNDConfig rnd_config;
    RNDNetwork* rnd_target;
    RNDNetwork* rnd_predictor;
    float* rnd_target_out;
    float* rnd_predictor_out;

    /* Go-Explore状态 */
    GoExploreConfig go_config;
    GoExploreCell* go_cells;
    int go_cell_count;
    int go_cell_capacity;
    int go_phase;

    /* NoisyNet状态 */
    NoisyNetConfig noisynet_config;
    float* noisynet_w_mu;
    float* noisynet_w_sigma;
    float* noisynet_b_mu;
    float* noisynet_b_sigma;
    float* noisynet_noise_in;
    float* noisynet_noise_out;

    /* 计数表 */
    float* count_table;
    int count_table_size;
    int count_dim;

    /* 通用 */
    float learning_rate;
    int is_initialized;
};

/* ============================================================================
 * 网络前向传播
 * ============================================================================ */

static void fc_forward(const float* input, const float* w, const float* b,
                        float* output, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        float sum = b ? b[o] : 0.0f;
        for (int i = 0; i < in_dim; i++) {
            sum += input[i] * w[o * in_dim + i];
        }
        output[o] = tanhf(sum);
    }
}

static void fc_forward_linear(const float* input, const float* w, const float* b,
                               float* output, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        float sum = b ? b[o] : 0.0f;
        for (int i = 0; i < in_dim; i++) {
            sum += input[i] * w[o * in_dim + i];
        }
        output[o] = sum;
    }
}

static void fc_backward(const float* input, const float* grad_output,
                         float* w, float* b, int in_dim, int out_dim, float lr) {
    for (int o = 0; o < out_dim; o++) {
        float go = grad_output[o];
        if (b) b[o] -= lr * go;
        for (int i = 0; i < in_dim; i++) {
            w[o * in_dim + i] -= lr * go * input[i];
        }
    }
}

/* ============================================================================
 * ICM实现（A05.1.3.1）
 * ============================================================================ */

ExploreState* explore_icm_create(const ICMConfig* config) {
    if (!config) return NULL;
    ExploreState* state = (ExploreState*)safe_calloc(1, sizeof(ExploreState));
    if (!state) return NULL;

    state->strategy_type = EXPLORE_ICM;
    state->icm_config = *config;
    state->learning_rate = config->learning_rate;

    int state_dim = config->state_dim;
    int action_dim = config->action_dim;
    int emb_dim = config->embedding_dim;

    /* 编码器: state -> embedding */
    state->icm_encoder = (ICMLayer*)safe_calloc(1, sizeof(ICMLayer));
    if (state->icm_encoder) {
        state->icm_encoder->in_dim = state_dim;
        state->icm_encoder->out_dim = emb_dim;
        state->icm_encoder->w = (float*)safe_calloc(emb_dim * state_dim, sizeof(float));
        state->icm_encoder->b = (float*)safe_calloc(emb_dim, sizeof(float));
        if (state->icm_encoder->w) {
            for (int i = 0; i < emb_dim * state_dim; i++)
                state->icm_encoder->w[i] = explore_randn(0.01f);
        }
    }

    /* 逆动力学: [embed_s, embed_s_next] -> action */
    state->icm_inverse = (ICMLayer*)safe_calloc(1, sizeof(ICMLayer));
    if (state->icm_inverse) {
        state->icm_inverse->in_dim = emb_dim * 2;
        state->icm_inverse->out_dim = action_dim;
        state->icm_inverse->w = (float*)safe_calloc(action_dim * emb_dim * 2, sizeof(float));
        state->icm_inverse->b = (float*)safe_calloc(action_dim, sizeof(float));
        if (state->icm_inverse->w) {
            for (int i = 0; i < action_dim * emb_dim * 2; i++)
                state->icm_inverse->w[i] = explore_randn(0.01f);
        }
    }

    /* 正动力学: [embed_s, action] -> embed_s_next */
    state->icm_forward = (ICMLayer*)safe_calloc(1, sizeof(ICMLayer));
    if (state->icm_forward) {
        state->icm_forward->in_dim = emb_dim + action_dim;
        state->icm_forward->out_dim = emb_dim;
        state->icm_forward->w = (float*)safe_calloc(emb_dim * (emb_dim + action_dim), sizeof(float));
        state->icm_forward->b = (float*)safe_calloc(emb_dim, sizeof(float));
        if (state->icm_forward->w) {
            for (int i = 0; i < emb_dim * (emb_dim + action_dim); i++)
                state->icm_forward->w[i] = explore_randn(0.01f);
        }
    }

    state->icm_embed_s = (float*)safe_calloc(emb_dim, sizeof(float));
    state->icm_embed_s_next = (float*)safe_calloc(emb_dim, sizeof(float));
    state->icm_pred_s_next = (float*)safe_calloc(emb_dim, sizeof(float));

    state->is_initialized = 1;
    return state;
}

int explore_icm_compute_reward(ExploreState* state,
                                const float* current_state,
                                const float* next_state,
                                const float* action,
                                float* intrinsic_reward) {
    if (!state || !current_state || !next_state || !action || !intrinsic_reward)
        return -1;

    ICMConfig* cfg = &state->icm_config;
    int emb_dim = cfg->embedding_dim;
    int state_dim = cfg->state_dim;
    int action_dim = cfg->action_dim;

    /* 编码当前状态和下一状态 */
    fc_forward(current_state, state->icm_encoder->w, state->icm_encoder->b,
               state->icm_embed_s, state_dim, emb_dim);
    fc_forward(next_state, state->icm_encoder->w, state->icm_encoder->b,
               state->icm_embed_s_next, state_dim, emb_dim);

    /* 前向预测: [embed_s, action] -> pred_embed_s_next */
    float* fwd_input = (float*)safe_calloc(emb_dim + action_dim, sizeof(float));
    if (!fwd_input) return -1;
    memcpy(fwd_input, state->icm_embed_s, emb_dim * sizeof(float));
    memcpy(fwd_input + emb_dim, action, action_dim * sizeof(float));
    fc_forward_linear(fwd_input, state->icm_forward->w, state->icm_forward->b,
                       state->icm_pred_s_next, emb_dim + action_dim, emb_dim);
    safe_free((void**)&fwd_input);

    /* 内在奖励 = 前向预测误差的L2范数 */
    float error = 0.0f;
    for (int i = 0; i < emb_dim; i++) {
        float diff = state->icm_pred_s_next[i] - state->icm_embed_s_next[i];
        error += diff * diff;
    }
    *intrinsic_reward = sqrtf(error + EXPLORE_EPSILON);

    return 0;
}

float explore_icm_train_batch(ExploreState* state,
                               const float* states, const float* next_states,
                               const float* actions, int batch_size) {
    if (!state || !states || !next_states || !actions || batch_size < 1)
        return -1.0f;

    float total_loss = 0.0f;
    ICMConfig* cfg = &state->icm_config;
    int emb_dim = cfg->embedding_dim;
    int state_dim = cfg->state_dim;
    int action_dim = cfg->action_dim;
    float lr = state->learning_rate;

    for (int b = 0; b < batch_size; b++) {
        const float* s = states + b * state_dim;
        const float* s_next = next_states + b * state_dim;
        const float* a = actions + b * action_dim;

        /* 编码 */
        fc_forward(s, state->icm_encoder->w, state->icm_encoder->b,
                   state->icm_embed_s, state_dim, emb_dim);
        fc_forward(s_next, state->icm_encoder->w, state->icm_encoder->b,
                   state->icm_embed_s_next, state_dim, emb_dim);

        /* 逆动力学损失 */
        float* inv_input = (float*)safe_calloc(emb_dim * 2, sizeof(float));
        if (!inv_input) continue;
        memcpy(inv_input, state->icm_embed_s, emb_dim * sizeof(float));
        memcpy(inv_input + emb_dim, state->icm_embed_s_next, emb_dim * sizeof(float));

        float* inv_pred = (float*)safe_calloc(action_dim, sizeof(float));
        if (!inv_pred) { safe_free((void**)&inv_input); continue; }
        fc_forward_linear(inv_input, state->icm_inverse->w, state->icm_inverse->b,
                          inv_pred, emb_dim * 2, action_dim);

        float* inv_grad = (float*)safe_calloc(action_dim, sizeof(float));
        if (inv_grad) {
            for (int i = 0; i < action_dim; i++) {
                inv_grad[i] = 2.0f * (inv_pred[i] - a[i]);
            }
            fc_backward(inv_input, inv_grad, state->icm_inverse->w,
                        state->icm_inverse->b, emb_dim * 2, action_dim,
                        lr * cfg->inverse_loss_weight);
            safe_free((void**)&inv_grad);
        }

        /* 正动力学损失 */
        float* fwd_input = (float*)safe_calloc(emb_dim + action_dim, sizeof(float));
        if (!fwd_input) { safe_free((void**)&inv_pred); safe_free((void**)&inv_input); continue; }
        memcpy(fwd_input, state->icm_embed_s, emb_dim * sizeof(float));
        memcpy(fwd_input + emb_dim, a, action_dim * sizeof(float));

        fc_forward_linear(fwd_input, state->icm_forward->w, state->icm_forward->b,
                          state->icm_pred_s_next, emb_dim + action_dim, emb_dim);

        float fwd_loss = 0.0f;
        float* fwd_grad = (float*)safe_calloc(emb_dim, sizeof(float));
        if (fwd_grad) {
            for (int i = 0; i < emb_dim; i++) {
                float diff = state->icm_pred_s_next[i] - state->icm_embed_s_next[i];
                fwd_grad[i] = 2.0f * diff;
                fwd_loss += diff * diff;
            }
            fc_backward(fwd_input, fwd_grad, state->icm_forward->w,
                        state->icm_forward->b, emb_dim + action_dim, emb_dim,
                        lr * cfg->forward_loss_weight);
            safe_free((void**)&fwd_grad);
        }

        total_loss += fwd_loss;

        safe_free((void**)&fwd_input);
        safe_free((void**)&inv_pred);
        safe_free((void**)&inv_input);
    }

    return total_loss / batch_size;
}

int explore_icm_get_embedding(ExploreState* state,
                               const float* raw_state, float* embedding) {
    if (!state || !raw_state || !embedding) return -1;
    fc_forward(raw_state, state->icm_encoder->w, state->icm_encoder->b,
               embedding, state->icm_config.state_dim,
               state->icm_config.embedding_dim);
    return 0;
}

/* ============================================================================
 * RND实现（A05.1.3.2）
 * ============================================================================ */

static RNDNetwork* rnd_network_create(int in_dim, int hidden_dim, int out_dim, float noise_std) {
    RNDNetwork* net = (RNDNetwork*)safe_calloc(1, sizeof(RNDNetwork));
    if (!net) return NULL;
    net->in_dim = in_dim;
    net->hidden_dim = hidden_dim;
    net->out_dim = out_dim;
    net->w1 = (float*)safe_calloc(hidden_dim * in_dim, sizeof(float));
    net->b1 = (float*)safe_calloc(hidden_dim, sizeof(float));
    net->w2 = (float*)safe_calloc(out_dim * hidden_dim, sizeof(float));
    net->b2 = (float*)safe_calloc(out_dim, sizeof(float));
    if (!net->w1 || !net->b1 || !net->w2 || !net->b2) {
        safe_free((void**)&net->w1); safe_free((void**)&net->b1);
        safe_free((void**)&net->w2); safe_free((void**)&net->b2);
        safe_free((void**)&net); return NULL;
    }
    for (int i = 0; i < hidden_dim * in_dim; i++)
        net->w1[i] = explore_randn(noise_std);
    for (int i = 0; i < out_dim * hidden_dim; i++)
        net->w2[i] = explore_randn(noise_std);
    return net;
}

static void rnd_network_forward(RNDNetwork* net, const float* input,
                                 float* hidden_out, float* output) {
    /* 隐藏层 */
    fc_forward(input, net->w1, net->b1, hidden_out, net->in_dim, net->hidden_dim);
    /* 输出层 */
    fc_forward_linear(hidden_out, net->w2, net->b2, output, net->hidden_dim, net->out_dim);
}

static void rnd_network_destroy(RNDNetwork* net) {
    if (!net) return;
    safe_free((void**)&net->w1); safe_free((void**)&net->b1);
    safe_free((void**)&net->w2); safe_free((void**)&net->b2);
    safe_free((void**)&net);
}

ExploreState* explore_rnd_create(const RNDConfig* config) {
    if (!config) return NULL;
    ExploreState* state = (ExploreState*)safe_calloc(1, sizeof(ExploreState));
    if (!state) return NULL;

    state->strategy_type = EXPLORE_RND;
    state->rnd_config = *config;
    state->learning_rate = config->learning_rate;

    /* 目标网络（固定） */
    state->rnd_target = rnd_network_create(config->state_dim, config->hidden_dim,
                                            config->embedding_dim, config->target_noise_std);
    /* 预测网络（可训练） */
    state->rnd_predictor = rnd_network_create(config->state_dim, config->hidden_dim,
                                              config->embedding_dim, config->predictor_noise_std);

    state->rnd_target_out = (float*)safe_calloc(config->embedding_dim, sizeof(float));
    state->rnd_predictor_out = (float*)safe_calloc(config->embedding_dim, sizeof(float));

    state->count_table = (float*)safe_calloc(1024, sizeof(float));
    state->count_table_size = 1024;
    state->count_dim = config->state_dim;

    state->is_initialized = 1;
    return state;
}

int explore_rnd_compute_reward(ExploreState* state,
                                const float* observation, float* reward) {
    if (!state || !observation || !reward) return -1;

    RNDConfig* cfg = &state->rnd_config;
    int emb_dim = cfg->embedding_dim;
    int hidden_dim = cfg->hidden_dim;

    float* hidden_buf = (float*)safe_calloc(hidden_dim, sizeof(float));
    if (!hidden_buf) return -1;

    /* 预测网络输出 */
    rnd_network_forward(state->rnd_predictor, observation,
                         hidden_buf, state->rnd_predictor_out);
    /* 目标网络输出 */
    rnd_network_forward(state->rnd_target, observation,
                         hidden_buf, state->rnd_target_out);

    /* 蒸馏误差 = 探索奖励 */
    float error = 0.0f;
    for (int i = 0; i < emb_dim; i++) {
        float diff = state->rnd_predictor_out[i] - state->rnd_target_out[i];
        error += diff * diff;
    }
    *reward = sqrtf(error + EXPLORE_EPSILON);

    safe_free((void**)&hidden_buf);
    return 0;
}

float explore_rnd_train_batch(ExploreState* state,
                               const float* observations, int batch_size) {
    if (!state || !observations || batch_size < 1) return -1.0f;

    RNDConfig* cfg = &state->rnd_config;
    int state_dim = cfg->state_dim;
    int emb_dim = cfg->embedding_dim;
    int hidden_dim = cfg->hidden_dim;
    float lr = state->learning_rate;

    float total_error = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        const float* obs = observations + b * state_dim;

        float* hidden_buf = (float*)safe_calloc(hidden_dim, sizeof(float));
        float* pred_out = (float*)safe_calloc(emb_dim, sizeof(float));
        float* target_out = (float*)safe_calloc(emb_dim, sizeof(float));
        if (!hidden_buf || !pred_out || !target_out) {
            safe_free((void**)&hidden_buf);
            safe_free((void**)&pred_out);
            safe_free((void**)&target_out);
            continue;
        }

        rnd_network_forward(state->rnd_predictor, obs, hidden_buf, pred_out);
        rnd_network_forward(state->rnd_target, obs, hidden_buf, target_out);

        /* 反向传播预测网络 */
        float* grad_out = (float*)safe_calloc(emb_dim, sizeof(float));
        if (grad_out) {
            for (int i = 0; i < emb_dim; i++) {
                grad_out[i] = 2.0f * (pred_out[i] - target_out[i]);
                total_error += (pred_out[i] - target_out[i]) * (pred_out[i] - target_out[i]);
            }

            /* 输出层梯度 */
            for (int o = 0; o < emb_dim; o++) {
                for (int h = 0; h < hidden_dim; h++) {
                    state->rnd_predictor->w2[o * hidden_dim + h] -= lr * grad_out[o] * hidden_buf[h];
                }
                state->rnd_predictor->b2[o] -= lr * grad_out[o];
            }

            /* 隐藏层梯度（tanh导数）*/
            float* grad_hidden = (float*)safe_calloc(hidden_dim, sizeof(float));
            if (grad_hidden) {
                for (int h = 0; h < hidden_dim; h++) {
                    float tanh_h = tanhf(hidden_buf[h]);
                    float dh = 1.0f - tanh_h * tanh_h;
                    float gh = 0.0f;
                    for (int o = 0; o < emb_dim; o++) {
                        gh += grad_out[o] * state->rnd_predictor->w2[o * hidden_dim + h];
                    }
                    grad_hidden[h] = gh * dh;
                }
                for (int h = 0; h < hidden_dim; h++) {
                    for (int i = 0; i < state_dim; i++) {
                        state->rnd_predictor->w1[h * state_dim + i] -= lr * grad_hidden[h] * obs[i];
                    }
                    state->rnd_predictor->b1[h] -= lr * grad_hidden[h];
                }
                safe_free((void**)&grad_hidden);
            }
            safe_free((void**)&grad_out);
        }

        safe_free((void**)&hidden_buf);
        safe_free((void**)&pred_out);
        safe_free((void**)&target_out);
    }

    return total_error / batch_size;
}

float explore_rnd_get_novelty(ExploreState* state, const float* observation) {
    float reward;
    if (explore_rnd_compute_reward(state, observation, &reward) != 0)
        return 0.0f;
    return reward;
}

/* ============================================================================
 * Go-Explore实现（A05.1.3.3）
 * ============================================================================ */

ExploreState* explore_go_create(const GoExploreConfig* config) {
    if (!config) return NULL;
    ExploreState* state = (ExploreState*)safe_calloc(1, sizeof(ExploreState));
    if (!state) return NULL;

    state->strategy_type = EXPLORE_GO_EXPLORE;
    state->go_config = *config;
    state->go_phase = GO_EXPLORE_PHASE_EXPLORE;

    state->go_cell_capacity = 1024;
    state->go_cells = (GoExploreCell*)safe_calloc(
        state->go_cell_capacity, sizeof(GoExploreCell));

    state->count_table = (float*)safe_calloc(1024, sizeof(float));
    state->count_table_size = 1024;

    state->is_initialized = 1;
    return state;
}

int explore_go_add_cell(ExploreState* state,
                         const float* cell_state, float episode_reward) {
    if (!state || !cell_state) return -1;

    GoExploreConfig* cfg = &state->go_config;
    int state_dim = cfg->state_dim;

    /* 检查是否与现有单元相似 */
    for (int c = 0; c < state->go_cell_count; c++) {
        GoExploreCell* cell = &state->go_cells[c];
        float dist = 0.0f;
        for (int i = 0; i < state_dim && i < cell->state_dim; i++) {
            float diff = cell_state[i] - cell->state[i];
            dist += diff * diff;
        }
        if (sqrtf(dist) < cfg->cell_threshold) {
            /* 更新现有单元 */
            cell->visit_count++;
            if (episode_reward > cell->best_reward) {
                cell->best_reward = episode_reward;
            }
            return c;
        }
    }

    /* 添加新单元 */
    if (state->go_cell_count >= state->go_cell_capacity) {
        int new_cap = state->go_cell_capacity * 2;
        GoExploreCell* new_cells = (GoExploreCell*)
            safe_realloc(state->go_cells, new_cap * sizeof(GoExploreCell));
        if (!new_cells) return -1;
        memset(new_cells + state->go_cell_capacity, 0,
               (new_cap - state->go_cell_capacity) * sizeof(GoExploreCell));
        state->go_cells = new_cells;
        state->go_cell_capacity = new_cap;
    }

    int idx = state->go_cell_count;
    GoExploreCell* cell = &state->go_cells[idx];
    cell->state = (float*)safe_calloc(state_dim, sizeof(float));
    if (!cell->state) return -1;
    memcpy(cell->state, cell_state, state_dim * sizeof(float));
    cell->state_dim = state_dim;
    cell->best_reward = episode_reward;
    cell->visit_count = 1;
    cell->is_robustified = 0;
    cell->trajectory_length = 0;
    cell->trajectory_actions = NULL;
    state->go_cell_count++;

    return idx;
}

int explore_go_select_cell(ExploreState* state,
                            float* selected_cell, int* cell_id) {
    if (!state || !selected_cell || !cell_id) return -1;

    if (state->go_cell_count < 1) {
        memset(selected_cell, 0, state->go_config.state_dim * sizeof(float));
        *cell_id = 0;
        return 0;
    }

    GoExploreConfig* cfg = &state->go_config;
    int selection = cfg->selection_strategy;

    if (selection == 0) {
        /* 随机选择 */
        *cell_id = (int)(explore_rand_float() * state->go_cell_count);
        if (*cell_id >= state->go_cell_count) *cell_id = state->go_cell_count - 1;
    } else if (selection == 1) {
        /* 加权采样 P ∝ reward/visit */
        float* weights = (float*)safe_calloc(state->go_cell_count, sizeof(float));
        if (!weights) return -1;
        float total = 0.0f;
        for (int c = 0; c < state->go_cell_count; c++) {
            weights[c] = state->go_cells[c].best_reward /
                (state->go_cells[c].visit_count + 1.0f);
            total += weights[c];
        }
        if (total < EXPLORE_EPSILON) {
            for (int c = 0; c < state->go_cell_count; c++) weights[c] = 1.0f;
            total = (float)state->go_cell_count;
        }
        float r = explore_rand_float() * total;
        float cum = 0.0f;
        *cell_id = 0;
        for (int c = 0; c < state->go_cell_count; c++) {
            cum += weights[c];
            if (r <= cum) { *cell_id = c; break; }
        }
        safe_free((void**)&weights);
    } else {
        /* 新颖优先（最少访问） */
        int min_visits = state->go_cells[0].visit_count;
        *cell_id = 0;
        for (int c = 1; c < state->go_cell_count; c++) {
            if (state->go_cells[c].visit_count < min_visits) {
                min_visits = state->go_cells[c].visit_count;
                *cell_id = c;
            }
        }
    }

    GoExploreCell* cell = &state->go_cells[*cell_id];
    memcpy(selected_cell, cell->state, cfg->state_dim * sizeof(float));
    return 0;
}

int explore_go_explore_from_cell(ExploreState* state,
                                  int cell_id,
                                  float* trajectory_states,
                                  float* trajectory_actions,
                                  int max_steps) {
    if (!state || !trajectory_states || !trajectory_actions) return -1;
    if (cell_id < 0 || cell_id >= state->go_cell_count) return -1;

    GoExploreCell* cell = &state->go_cells[cell_id];
    int state_dim = state->go_config.state_dim;
    int action_dim = state->go_config.action_dim;

    /* 从指定cell出发执行探索轨迹生成 */
    memcpy(trajectory_states, cell->state, state_dim * sizeof(float));

    int steps = max_steps < state->go_config.max_episode_steps ?
        max_steps : state->go_config.max_episode_steps;

    /* Ornstein-Uhlenbeck过程参数：生成时间相关探索噪声 */
    float ou_theta = 0.15f;
    float ou_sigma = 0.20f;
    float ou_mu = 0.0f;
    OU_LOCK();
    static float* ou_state = NULL;
    static int ou_dim = 0;
    if (ou_dim != action_dim) {
        safe_free((void**)&ou_state);
        ou_state = (float*)safe_calloc(action_dim, sizeof(float));
        ou_dim = action_dim;
    }

    for (int t = 0; t < steps; t++) {
        float* s = trajectory_states + t * state_dim;
        float* a = trajectory_actions + t * action_dim;

        /* Ornstein-Uhlenbeck噪声：dX = θ(μ-X)dt + σdW */
        for (int i = 0; i < action_dim; i++) {
            float dW = explore_randn(1.0f);
            ou_state[i] += ou_theta * (ou_mu - ou_state[i]) + ou_sigma * dW;
            a[i] = ou_state[i];
        }

        /* 状态转移：基于当前状态+动作的物理一致性更新 */
        if (t + 1 < steps) {
            float* s_next = trajectory_states + (t + 1) * state_dim;
            float decay = 1.0f / (1.0f + 0.05f * (float)t);
            for (int i = 0; i < state_dim; i++) {
                float action_proj = a[i % action_dim] * 0.1f * decay;
                s_next[i] = s[i] + action_proj;
            }
            /* 状态边界约束：确保状态在合理范围内 */
            for (int i = 0; i < state_dim; i++) {
                if (s_next[i] > 1e3f) s_next[i] = 1e3f;
                if (s_next[i] < -1e3f) s_next[i] = -1e3f;
            }
        }
    }

    cell->visit_count++;
    OU_UNLOCK();
    return steps;
}

float explore_go_robustify(ExploreState* state,
                            int target_cell_id, int max_attempts) {
    if (!state || target_cell_id < 0 || target_cell_id >= state->go_cell_count)
        return -1.0f;

    GoExploreCell* cell = &state->go_cells[target_cell_id];
    int state_dim = state->go_config.state_dim;
    int successes = 0;

    for (int a = 0; a < max_attempts; a++) {
        /* 从存档中随机选择其他cell作为起点 */
        int src_id = (int)(explore_rand_float() * state->go_cell_count);
        if (src_id == target_cell_id) {
            src_id = (src_id + 1) % state->go_cell_count;
        }
        GoExploreCell* src = &state->go_cells[src_id];

        /* 计算起点到目标单元的欧氏距离 */
        float total_dist = 0.0f;
        int min_dim = state_dim < src->state_dim ? state_dim : src->state_dim;
        min_dim = min_dim < cell->state_dim ? min_dim : cell->state_dim;
        for (int i = 0; i < min_dim; i++) {
            float diff = cell->state[i] - src->state[i];
            total_dist += diff * diff;
        }
        float euclidean_dist = sqrtf(total_dist);

        /* 如果距离小于阈值，认为可达 */
        if (euclidean_dist < state->go_config.cell_threshold * 5.0f) {
            successes++;
        }
    }

    float rate = (float)successes / (float)(max_attempts > 0 ? max_attempts : 1);
    if (rate >= state->go_config.robustify_success_rate) {
        cell->is_robustified = 1;
    }
    return rate;
}

int explore_go_get_stats(ExploreState* state,
                          int* num_cells, int* total_visits, float* best_reward) {
    if (!state || !num_cells || !total_visits || !best_reward) return -1;
    *num_cells = state->go_cell_count;
    *total_visits = 0;
    *best_reward = -FLT_MAX;
    for (int c = 0; c < state->go_cell_count; c++) {
        *total_visits += state->go_cells[c].visit_count;
        if (state->go_cells[c].best_reward > *best_reward)
            *best_reward = state->go_cells[c].best_reward;
    }
    return 0;
}

/* ============================================================================
 * 通用探索工具API
 * ============================================================================ */

ExploreState* explore_noisynet_create(const NoisyNetConfig* config) {
    if (!config) return NULL;
    ExploreState* state = (ExploreState*)safe_calloc(1, sizeof(ExploreState));
    if (!state) return NULL;

    state->strategy_type = EXPLORE_NOISY_NET;
    state->noisynet_config = *config;
    state->learning_rate = config->learning_rate;

    int in_dim = config->input_dim;
    int out_dim = config->output_dim;
    int total = in_dim * out_dim;

    state->noisynet_w_mu = (float*)safe_calloc(total, sizeof(float));
    state->noisynet_w_sigma = (float*)safe_calloc(total, sizeof(float));
    state->noisynet_b_mu = (float*)safe_calloc(out_dim, sizeof(float));
    state->noisynet_b_sigma = (float*)safe_calloc(out_dim, sizeof(float));
    state->noisynet_noise_in = (float*)safe_calloc(in_dim, sizeof(float));
    state->noisynet_noise_out = (float*)safe_calloc(out_dim, sizeof(float));

    if (state->noisynet_w_mu) {
        for (int i = 0; i < total; i++)
            state->noisynet_w_mu[i] = explore_randn(0.1f);
    }
    if (state->noisynet_w_sigma) {
        for (int i = 0; i < total; i++)
            state->noisynet_w_sigma[i] = config->sigma_init;
    }

    state->is_initialized = 1;
    return state;
}

int explore_noisynet_forward(ExploreState* state,
                              const float* input, float* output,
                              int sample_noise) {
    if (!state || !input || !output) return -1;

    NoisyNetConfig* cfg = &state->noisynet_config;
    int in_dim = cfg->input_dim;
    int out_dim = cfg->output_dim;

    if (sample_noise) {
        for (int i = 0; i < in_dim; i++)
            state->noisynet_noise_in[i] = explore_randn(1.0f);
        for (int i = 0; i < out_dim; i++)
            state->noisynet_noise_out[i] = explore_randn(1.0f);
    }

    for (int o = 0; o < out_dim; o++) {
        float sum = state->noisynet_b_mu[o] +
            state->noisynet_b_sigma[o] * state->noisynet_noise_out[o];
        for (int i = 0; i < in_dim; i++) {
            float w = state->noisynet_w_mu[o * in_dim + i] +
                state->noisynet_w_sigma[o * in_dim + i] * state->noisynet_noise_in[i];
            sum += input[i] * w;
        }
        output[o] = tanhf(sum);
    }
    return 0;
}

int explore_parameter_noise(ExploreState* state,
                             const float* params, float noise_std,
                             float* noisy_params) {
    if (!state || !params || !noisy_params) return -1;
    int n = state->icm_config.embedding_dim * state->icm_config.state_dim;
    if (n < 1) n = 1024;
    for (int i = 0; i < n; i++) {
        noisy_params[i] = params[i] + explore_randn(noise_std);
    }
    return 0;
}

float explore_count_based_reward(ExploreState* state,
                                  const float* state_vector, int state_dim) {
    if (!state || !state_vector) return 0.0f;

    /* 简单哈希计数 */
    unsigned int hash = 0;
    for (int i = 0; i < state_dim && i < 4; i++) {
        hash = hash * 31 + (unsigned int)(state_vector[i] * 1000);
    }
    int idx = hash % state->count_table_size;
    state->count_table[idx] += 1.0f;
    return 1.0f / sqrtf(state->count_table[idx] + 1.0f);
}

float explore_uncertainty_based_reward(ExploreState* state,
                                        const float* predictions,
                                        int num_predictions) {
    if (!state || !predictions || num_predictions < 2) return 0.0f;

    float mean = 0.0f;
    for (int i = 0; i < num_predictions; i++) mean += predictions[i];
    mean /= num_predictions;

    float variance = 0.0f;
    for (int i = 0; i < num_predictions; i++) {
        float diff = predictions[i] - mean;
        variance += diff * diff;
    }
    return sqrtf(variance / (num_predictions - 1));
}

float explore_hybrid_reward(ExploreState* state,
                             const float* rewards,
                             const ExploreStrategyType* strategy_types,
                             int num_strategies) {
    if (!state || !rewards || !strategy_types || num_strategies < 1) return 0.0f;

    float total = 0.0f;
    float weights[8] = {0.3f, 0.3f, 0.2f, 0.1f, 0.05f, 0.025f, 0.025f, 0.0f};

    for (int i = 0; i < num_strategies && i < 8; i++) {
        total += weights[strategy_types[i]] * rewards[i];
    }
    return total;
}

/* ============================================================================
 * 销毁与模型管理
 * ============================================================================ */

void explore_destroy(ExploreState* state) {
    if (!state) return;

    /* ICM释放 */
    if (state->icm_encoder) {
        safe_free((void**)&state->icm_encoder->w);
        safe_free((void**)&state->icm_encoder->b);
        safe_free((void**)&state->icm_encoder);
    }
    if (state->icm_inverse) {
        safe_free((void**)&state->icm_inverse->w);
        safe_free((void**)&state->icm_inverse->b);
        safe_free((void**)&state->icm_inverse);
    }
    if (state->icm_forward) {
        safe_free((void**)&state->icm_forward->w);
        safe_free((void**)&state->icm_forward->b);
        safe_free((void**)&state->icm_forward);
    }
    safe_free((void**)&state->icm_embed_s);
    safe_free((void**)&state->icm_embed_s_next);
    safe_free((void**)&state->icm_pred_s_next);

    /* RND释放 */
    if (state->rnd_target) rnd_network_destroy(state->rnd_target);
    if (state->rnd_predictor) rnd_network_destroy(state->rnd_predictor);
    safe_free((void**)&state->rnd_target_out);
    safe_free((void**)&state->rnd_predictor_out);

    /* Go-Explore释放 */
    for (int c = 0; c < state->go_cell_count; c++) {
        safe_free((void**)&state->go_cells[c].state);
        safe_free((void**)&state->go_cells[c].trajectory_actions);
    }
    safe_free((void**)&state->go_cells);

    /* NoisyNet释放 */
    safe_free((void**)&state->noisynet_w_mu);
    safe_free((void**)&state->noisynet_w_sigma);
    safe_free((void**)&state->noisynet_b_mu);
    safe_free((void**)&state->noisynet_b_sigma);
    safe_free((void**)&state->noisynet_noise_in);
    safe_free((void**)&state->noisynet_noise_out);

    safe_free((void**)&state->count_table);
    safe_free((void**)&state);
}

int explore_save(const ExploreState* state, ExploreStrategyType strategy_type,
                  const char* filepath) {
    if (!state || !filepath) return -1;
    FILE* f = fopen(filepath, "wb");
    if (!f) return -1;

    int type = strategy_type;
    fwrite(&type, sizeof(int), 1, f);

    if (strategy_type == EXPLORE_ICM) {
        ICMConfig* cfg = (ICMConfig*)&state->icm_config;
        fwrite(cfg, sizeof(ICMConfig), 1, f);
        int total_w = cfg->embedding_dim * cfg->state_dim;
        fwrite(state->icm_encoder->w, sizeof(float), total_w, f);
    } else if (strategy_type == EXPLORE_RND) {
        RNDConfig* cfg = (RNDConfig*)&state->rnd_config;
        fwrite(cfg, sizeof(RNDConfig), 1, f);
    }

    fclose(f);
    return 0;
}

ExploreState* explore_load(ExploreStrategyType strategy_type, const char* filepath) {
    (void)strategy_type;
    if (!filepath) return NULL;
    FILE* f = fopen(filepath, "rb");
    if (!f) return NULL;

    int type;
    if (fread(&type, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }

    ExploreState* state = NULL;
    if (type == EXPLORE_ICM) {
        ICMConfig cfg;
        if (fread(&cfg, sizeof(ICMConfig), 1, f) == 1)
            state = explore_icm_create(&cfg);
    } else if (type == EXPLORE_RND) {
        RNDConfig cfg;
        if (fread(&cfg, sizeof(RNDConfig), 1, f) == 1)
            state = explore_rnd_create(&cfg);
    }

    fclose(f);
    return state;
}
