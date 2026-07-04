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
#include "selflnn/selflnn.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/optimizer.h" /* 统一优化器替代独立SGD */
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/secure_random.h"
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

/* P2-001修复: 使用密码学安全随机数替代简单LCG确定性伪随机 */
static float explore_rand_float(void) {
    return secure_random_float();
}

static float explore_randn(float std) {
    float u1 = secure_random_float();
    float u2 = secure_random_float();
    if (u1 < EXPLORE_EPSILON) u1 = EXPLORE_EPSILON;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2) * std;
}

/* DEADCODE-FIX: 高级探索策略条件编译（P2-6标注：已通过宏控制，此非问题而是设计选择）
 * ICM / RND / Go-Explore / NoisyNet / count-based / uncertainty / hybrid
 * 共26个函数约900行——目前仅UCB的3个函数被robot_agent.c使用，
 * 其余均未集成。通过 SELFLNN_SKIP_ADVANCED_EXPLORATION 宏控制编译。
 * 如需减小二进制体积，编译时定义此宏即可跳过所有高级探索策略。 */
#ifndef SELFLNN_SKIP_ADVANCED_EXPLORATION

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
    Optimizer* optimizer; /**< 统一优化器实例 */
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

/* 创建统一SGD优化器（ICM使用fc_backward中的SGD更新） */
    {
        OptimizerConfig opt_cfg = {0};
        opt_cfg.type = OPTIMIZER_SGD;
        opt_cfg.learning_rate = config->learning_rate;
        state->optimizer = optimizer_create(&opt_cfg);
    }

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

/* 创建统一SGD优化器（RND使用统一optimizer_step） */
    {
        OptimizerConfig opt_cfg = {0};
        opt_cfg.type = OPTIMIZER_SGD;
        opt_cfg.learning_rate = config->learning_rate;
        state->optimizer = optimizer_create(&opt_cfg);
    }

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

/* 使用统一优化器替代手动SGD权重更新。
         * 先计算所有梯度到临时数组，再通过optimizer_step统一更新。 */
        float* grad_out = (float*)safe_calloc(emb_dim, sizeof(float));
        if (grad_out) {
            for (int i = 0; i < emb_dim; i++) {
                grad_out[i] = 2.0f * (pred_out[i] - target_out[i]);
                total_error += (pred_out[i] - target_out[i]) * (pred_out[i] - target_out[i]);
            }

            /* 输出层梯度收集 */
            size_t w2_total = (size_t)emb_dim * (size_t)hidden_dim;
            float* w2_grads = (float*)safe_calloc(w2_total, sizeof(float));
            float* b2_grads = (float*)safe_calloc((size_t)emb_dim, sizeof(float));
            if (w2_grads && b2_grads) {
                for (int o = 0; o < emb_dim; o++) {
                    for (int h = 0; h < hidden_dim; h++) {
                        w2_grads[(size_t)o * (size_t)hidden_dim + h] = grad_out[o] * hidden_buf[h];
                    }
                    b2_grads[o] = grad_out[o];
                }
                optimizer_step(state->optimizer, state->rnd_predictor->w2, w2_grads, w2_total, 0);
                optimizer_step(state->optimizer, state->rnd_predictor->b2, b2_grads, (size_t)emb_dim, 0);
            }
            safe_free((void**)&w2_grads); safe_free((void**)&b2_grads);

            /* 隐藏层梯度收集 */
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
                size_t w1_total = (size_t)hidden_dim * (size_t)state_dim;
                float* w1_grads = (float*)safe_calloc(w1_total, sizeof(float));
                float* b1_grads = (float*)safe_calloc((size_t)hidden_dim, sizeof(float));
                if (w1_grads && b1_grads) {
                    for (int h = 0; h < hidden_dim; h++) {
                        for (int i = 0; i < state_dim; i++) {
                            w1_grads[(size_t)h * (size_t)state_dim + i] = grad_hidden[h] * obs[i];
                        }
                        b1_grads[h] = grad_hidden[h];
                    }
                    optimizer_step(state->optimizer, state->rnd_predictor->w1, w1_grads, w1_total, 0);
                    optimizer_step(state->optimizer, state->rnd_predictor->b1, b1_grads, (size_t)hidden_dim, 0);
                }
                safe_free((void**)&w1_grads); safe_free((void**)&b1_grads);
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

    /* H-009修复: 通过selflnn_get_shared_lnn()获取共享LNN实例 */
    LNN* shared_lnn = (LNN*)selflnn_get_shared_lnn();
    if (!shared_lnn) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "Go-Explore探索失败：共享LNN未初始化");
        return -1;
    }

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

        if (t + 1 < steps) {
            float* s_next = trajectory_states + (t + 1) * state_dim;
            /* 尝试使用LNN的CfC动力学进行状态转移 */
            if (lnn_forward_safe(shared_lnn, s, (size_t)state_dim, s_next, (size_t)state_dim) > 0) {
                /* CfC动力学成功，使用LNN预测的状态 */
                float noise_scale = 0.05f / (1.0f + 0.1f * (float)t);
                for (int i = 0; i < state_dim; i++) {
                    s_next[i] += explore_randn(noise_scale);
                }
            } else {
                /* LNN不可用时：使用随机扰动+OU噪声的采样器，
                 * 不做线性简化推断，而是基于当前状态的带噪声采样 */
                float decay = 1.0f / (1.0f + 0.05f * (float)t);
                for (int i = 0; i < state_dim; i++) {
                    float noise = explore_randn(0.02f * decay);
                    s_next[i] = s[i] + noise;
                }
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

/* 释放统一优化器 */
    if (state->optimizer) optimizer_free(state->optimizer);

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

#endif /* SELFLNN_SKIP_ADVANCED_EXPLORATION */

/* ============================================================================
 * D-013: UCB1 (Upper Confidence Bound) 完整实现
 * 公式: a_t = argmax [ Q(a) + c * sqrt(ln(t) / N(a)) ]
 * 收敛条件: 当最优臂选择频率>90%且连续100步稳定时，降低探索系数
 * ============================================================================ */

typedef struct {
    float* Q;              /* 动作价值估计 [num_actions] */
    int* N;                /* 动作选择次数 [num_actions] */
    float* U;              /* 动作上界 [num_actions] */
    int total_steps;       /* 总步数 */
    int num_actions;       /* 动作数 */
    float c;               /* 探索系数 */
    int best_action;       /* 最近最优动作 */
    int best_streak;       /* 最优动作连续选中次数 */
    int early_stop_thresh; /* 早停阈值 */
    int converged;         /* 是否已收敛 */
} UCBState;

void* exploration_ucb_create(int num_actions, float c) {
    UCBState* s = (UCBState*)safe_calloc(1, sizeof(UCBState));
    if (!s || num_actions <= 0) { safe_free((void**)&s); return NULL; }
    s->num_actions = num_actions;
    s->c = (c > 0.0f) ? c : 1.414f;
    s->Q = (float*)safe_calloc((size_t)num_actions, sizeof(float));
    s->N = (int*)safe_calloc((size_t)num_actions, sizeof(int));
    s->U = (float*)safe_calloc((size_t)num_actions, sizeof(float));
    if (!s->Q || !s->N || !s->U) {
        safe_free((void**)&s->U); safe_free((void**)&s->Q);
        safe_free((void**)&s->N); safe_free((void**)&s);
        return NULL;
    }
    s->total_steps = 0;
    s->best_action = -1;
    s->best_streak = 0;
    s->early_stop_thresh = 100;
    s->converged = 0;
    return (void*)s;
}

void exploration_ucb_free(void* state) {
    if (!state) return;
    UCBState* s = (UCBState*)state;
    safe_free((void**)&s->Q);
    safe_free((void**)&s->N);
    safe_free((void**)&s->U);
    safe_free((void**)&state);
}

int exploration_ucb_select(void* state, float* action_values, int num_actions) {
    UCBState* s = (UCBState*)state;
    if (!s || !action_values || num_actions <= 0) return -1;

    s->total_steps++;
    int best_a = 0;
    float best_val = -1e20f;
    float ln_t = logf((float)(s->total_steps > 0 ? s->total_steps : 1));
    float effective_c = s->c;

    /* 收敛检测: 当探索系数很低时进一步降低 */
    if (s->converged) effective_c *= 0.1f;

    for (int a = 0; a < num_actions && a < s->num_actions; a++) {
        /* 每个动作至少选一次 (保证探索) */
        if (s->N[a] == 0) { best_a = a; best_val = 1e19f; break; }
        /* UCB1公式: 经验均值 + 置信上界 */
        float confidence = effective_c * sqrtf(ln_t / (float)s->N[a]);
        float ucb = s->Q[a] + confidence;
        s->U[a] = ucb;
        if (ucb > best_val) { best_val = ucb; best_a = a; }
    }

    /* 早停检测: 连续选中同一最优动作 */
    if (best_a == s->best_action) {
        s->best_streak++;
        if (s->best_streak >= s->early_stop_thresh) s->converged = 1;
    } else {
        s->best_action = best_a;
        s->best_streak = 0;
        s->converged = 0;
    }

    for (int a = 0; a < num_actions && a < s->num_actions; a++) {
        action_values[a] = s->Q[a];
    }
    return best_a;
}

void exploration_ucb_update(void* state, int action, float reward) {
    UCBState* s = (UCBState*)state;
    if (!s || action < 0 || action >= s->num_actions) return;
    /* 增量平均更新: Q_new = Q_old + (reward - Q_old) / N */
    s->N[action]++;
    s->Q[action] += (reward - s->Q[action]) / (float)s->N[action];
}

float exploration_ucb_get_best(void* state) {
    UCBState* s = (UCBState*)state;
    if (!s || s->num_actions <= 0) return 0.0f;
    float best = s->Q[0];
    for (int a = 1; a < s->num_actions; a++) {
        if (s->Q[a] > best) best = s->Q[a];
    }
    return best;
}

int exploration_ucb_is_converged(void* state) {
    UCBState* s = (UCBState*)state;
    if (!s) return 0;
    return s->converged;
}

/* DEADCODE-FIX: Boltzmann / Thompson 探索策略同样受条件编译控制 */
#ifndef SELFLNN_SKIP_ADVANCED_EXPLORATION

/* ============================================================================
 * D-013: Boltzmann（Softmax）探索 - 温度退火完整实现
 * 公式: P(a) = exp(Q(a) / T) / Σ exp(Q(j) / T)
 * 温度退火: T_t = T_min + (T_max - T_min) * exp(-λ * t)
 * 温度范围: T_max=10.0, T_min=0.1, 衰减率λ可在初始化时设定
 * ============================================================================ */

typedef struct {
    float* Q;                  /* 动作价值估计 [num_actions] */
    int* N;                    /* 动作选择次数 [num_actions] */
    float* P;                  /* 动作选择概率 [num_actions] */
    int total_steps;           /* 总步数 */
    int num_actions;           /* 动作数 */
    float T_max;               /* 初始温度（高探索） */
    float T_min;               /* 最终温度（高利用） */
    float T_current;           /* 当前温度 */
    float lambda;              /* 温度衰减率 */
    int best_action;           /* 最近最优动作 */
    int best_streak;           /* 最优动作连续选中计数 */
    int early_stop_thresh;     /* 早停阈值 */
    int converged;             /* 是否已收敛 */
} BoltzmannState;

void* exploration_boltzmann_create(float temperature, float decay) {
    /* 匹配头文件签名；内部使用默认action数和温度范围 */
    int num_actions = 128;
    float T_max = temperature;
    float T_min = 0.1f;
    float lambda = decay;
    BoltzmannState* s = (BoltzmannState*)safe_calloc(1, sizeof(BoltzmannState));
    if (!s || num_actions <= 0) { safe_free((void**)&s); return NULL; }
    s->num_actions = num_actions;
    s->T_max = (T_max > 0.0f) ? T_max : 10.0f;
    s->T_min = (T_min > 0.0f) ? T_min : 0.1f;
    s->lambda = (lambda > 0.0f) ? lambda : 0.001f;
    s->T_current = s->T_max;
    s->Q = (float*)safe_calloc((size_t)num_actions, sizeof(float));
    s->N = (int*)safe_calloc((size_t)num_actions, sizeof(int));
    s->P = (float*)safe_calloc((size_t)num_actions, sizeof(float));
    if (!s->Q || !s->N || !s->P) {
        safe_free((void**)&s->P); safe_free((void**)&s->Q);
        safe_free((void**)&s->N); safe_free((void**)&s);
        return NULL;
    }
    /* 初始均匀概率 */
    for (int i = 0; i < num_actions; i++) {
        s->P[i] = 1.0f / (float)num_actions;
    }
    s->total_steps = 0;
    s->best_action = -1;
    s->best_streak = 0;
    s->early_stop_thresh = 200;
    s->converged = 0;
    return (void*)s;
}

void exploration_boltzmann_free(void* state) {
    if (!state) return;
    BoltzmannState* s = (BoltzmannState*)state;
    safe_free((void**)&s->Q);
    safe_free((void**)&s->N);
    safe_free((void**)&s->P);
    safe_free((void**)&state);
}

int exploration_boltzmann_select(void* state, float* action_values, int num_actions) {
    BoltzmannState* s = (BoltzmannState*)state;
    if (!s || !action_values || num_actions <= 0) return -1;

    s->total_steps++;

    /* 温度指数退火: T_t = T_min + (T_max - T_min) * exp(-λ * t) */
    s->T_current = s->T_min + (s->T_max - s->T_min) *
        expf(-s->lambda * (float)s->total_steps);
    if (s->T_current < s->T_min) s->T_current = s->T_min;

    float T = s->T_current;
    if (T < EXPLORE_EPSILON) T = EXPLORE_EPSILON;

    /* 计算每个动作的Softmax概率 */
    float max_q = -1e20f;
    for (int a = 0; a < num_actions && a < s->num_actions; a++) {
        if (s->Q[a] > max_q) max_q = s->Q[a];
    }

    float sum = 0.0f;
    for (int a = 0; a < num_actions && a < s->num_actions; a++) {
        /* 数值稳定: 减去max_q后再exp */
        s->P[a] = expf((s->Q[a] - max_q) / T);
        sum += s->P[a];
    }

    /* 归一化概率 */
    if (sum > EXPLORE_EPSILON) {
        for (int a = 0; a < num_actions && a < s->num_actions; a++) {
            s->P[a] /= sum;
        }
    } else {
        for (int a = 0; a < num_actions && a < s->num_actions; a++) {
            s->P[a] = 1.0f / (float)s->num_actions;
        }
    }

    /* 轮盘赌选择 */
    float r = explore_rand_float();
    float cum = 0.0f;
    int selected = 0;
    for (int a = 0; a < num_actions && a < s->num_actions; a++) {
        cum += s->P[a];
        if (r <= cum) { selected = a; break; }
    }

    /* 早停检测 */
    int best_a = 0;
    float best_q = s->Q[0];
    for (int a = 1; a < s->num_actions; a++) {
        if (s->Q[a] > best_q) { best_q = s->Q[a]; best_a = a; }
    }
    if (selected == s->best_action) {
        s->best_streak++;
        if (s->best_streak >= s->early_stop_thresh) s->converged = 1;
    } else {
        s->best_action = selected;
        s->best_streak = 0;
    }

    for (int a = 0; a < num_actions && a < s->num_actions; a++) {
        action_values[a] = s->Q[a];
    }
    return selected;
}

void exploration_boltzmann_update(void* state, int action, float reward) {
    BoltzmannState* s = (BoltzmannState*)state;
    if (!s || action < 0 || action >= s->num_actions) return;
    s->N[action]++;
    s->Q[action] += (reward - s->Q[action]) / (float)s->N[action];
}

float exploration_boltzmann_get_temperature(void* state) {
    BoltzmannState* s = (BoltzmannState*)state;
    if (!s) return 0.0f;
    return s->T_current;
}

float exploration_boltzmann_get_best(void* state) {
    BoltzmannState* s = (BoltzmannState*)state;
    if (!s || s->num_actions <= 0) return 0.0f;
    float best = s->Q[0];
    for (int a = 1; a < s->num_actions; a++) {
        if (s->Q[a] > best) best = s->Q[a];
    }
    return best;
}

void exploration_boltzmann_get_probabilities(void* state, float* probs, int num_actions) {
    BoltzmannState* s = (BoltzmannState*)state;
    if (!s || !probs || num_actions <= 0) return;
    int n = num_actions < s->num_actions ? num_actions : s->num_actions;
    memcpy(probs, s->P, (size_t)n * sizeof(float));
}

/* ============================================================================
 * D-013: Thompson Sampling (贝叶斯) - Beta分布完整实现
 * 使用Beta-Bernoulli共轭先验: Beta(α=success+1, β=failure+1)
 * 
 * Beta采样采用Marsaglia-Tsang Gamma分布生成法:
 *   1. Gamma(α, 1) 采样: 使用Marsaglia-Tsang拒绝采样法
 *   2. Beta(α, β) ~ Gamma(α) / (Gamma(α) + Gamma(β))
 * 
 * 后验更新:
 *   α += reward (clipped to [0, 1])
 *   β += (1 - reward)
 * 
 * 优点:
 *   - 自动平衡探索与利用
 *   - 无需手动调节温度或探索率
 *   - 不确定性高的动作自动获得更多尝试
 * ============================================================================ */

typedef struct {
    float* alpha;           /* Beta分布α参数 (成功计数) [num_actions] */
    float* beta_param;      /* Beta分布β参数 (失败计数) [num_actions] */
    float* Q_mean;          /* 动作后验均值 [num_actions] */
    int* N;                 /* 动作选择次数 [num_actions] */
    int num_actions;        /* 动作数 */
    int total_steps;        /* 总步数 */
    int best_action;        /* 最近最优动作 */
    int best_streak;        /* 最优动作连续选中次数 */
    int early_stop_thresh;  /* 早停阈值 */
    int converged;          /* 是否已收敛 */
} ThompsonState;

/* Marsaglia-Tsang Gamma分布采样: X ~ Gamma(shape, 1)
 * 适用于 shape >= 1 的情况; shape < 1 时使用 Ahrens-Dieter 法 */
static float gamma_sample_mt(float shape) {
    float d, c, x, v, u;
    if (shape < 0.1f) shape = 0.1f;

    if (shape < 1.0f) {
        /* Ahrens-Dieter方法: Gamma(α, 1), 0 < α < 1 */
        float e = 2.718281828459045f;
        float alpha = shape;
        float p;
        while (1) {
            p = e / (alpha + e);
            float u1 = explore_rand_float();
            if (u1 < p) {
                x = powf(u1 / p, 1.0f / alpha);
                float u2 = explore_rand_float();
                if (u2 <= expf(-x)) return x;
            } else {
                x = -logf((p - u1) / p + EXPLORE_EPSILON);
                float u2 = explore_rand_float();
                if (u2 <= powf(x, alpha - 1.0f)) return x;
            }
        }
    }

    /* Marsaglia-Tsang方法: shape >= 1 */
    d = shape - 1.0f / 3.0f;
    c = 1.0f / sqrtf(9.0f * d);

    while (1) {
        x = explore_randn(1.0f);
        v = 1.0f + c * x;
        if (v <= 0.0f) continue;
        v = v * v * v;
        u = explore_rand_float();
        if (u < 1.0f - 0.0331f * (x * x) * (x * x)) return d * v;
        if (logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v))) return d * v;
    }
}

/* Beta分布采样: X ~ Beta(α, β) = Gamma(α) / (Gamma(α) + Gamma(β)) */
static float beta_sample(float alpha, float beta_val) {
    if (alpha <= 0.0f) alpha = 0.001f;
    if (beta_val <= 0.0f) beta_val = 0.001f;
    float g1 = gamma_sample_mt(alpha);
    float g2 = gamma_sample_mt(beta_val);
    float denom = g1 + g2;
    if (denom < EXPLORE_EPSILON) return 0.5f;
    return g1 / denom;
}

void* exploration_thompson_create(int num_actions) {
    ThompsonState* s = (ThompsonState*)safe_calloc(1, sizeof(ThompsonState));
    if (!s || num_actions <= 0) { safe_free((void**)&s); return NULL; }
    s->num_actions = num_actions;
    s->alpha = (float*)safe_calloc((size_t)num_actions, sizeof(float));
    s->beta_param = (float*)safe_calloc((size_t)num_actions, sizeof(float));
    s->Q_mean = (float*)safe_calloc((size_t)num_actions, sizeof(float));
    s->N = (int*)safe_calloc((size_t)num_actions, sizeof(int));
    if (!s->alpha || !s->beta_param || !s->Q_mean || !s->N) {
        safe_free((void**)&s->N); safe_free((void**)&s->Q_mean);
        safe_free((void**)&s->beta_param); safe_free((void**)&s->alpha);
        safe_free((void**)&s);
        return NULL;
    }
    /* 无信息先验 Beta(1, 1) — 均匀分布 */
    for (int i = 0; i < num_actions; i++) {
        s->alpha[i] = 1.0f;
        s->beta_param[i] = 1.0f;
        s->Q_mean[i] = 0.5f;
    }
    s->total_steps = 0;
    s->best_action = -1;
    s->best_streak = 0;
    s->early_stop_thresh = 150;
    s->converged = 0;
    return (void*)s;
}

void exploration_thompson_free(void* state) {
    if (!state) return;
    ThompsonState* s = (ThompsonState*)state;
    safe_free((void**)&s->alpha);
    safe_free((void**)&s->beta_param);
    safe_free((void**)&s->Q_mean);
    safe_free((void**)&s->N);
    safe_free((void**)&state);
}

int exploration_thompson_select(void* state, float* action_values, int num_actions) {
    ThompsonState* s = (ThompsonState*)state;
    if (!s || !action_values || num_actions <= 0) return -1;

    s->total_steps++;
    int best_a = 0;
    float best_sample = -1.0f;

    for (int a = 0; a < num_actions && a < s->num_actions; a++) {
        /* 从Beta(α, β)后验分布采样 */
        float sample = beta_sample(s->alpha[a], s->beta_param[a]);

        /* 后验均值用于显示 */
        s->Q_mean[a] = s->alpha[a] / (s->alpha[a] + s->beta_param[a]);
        action_values[a] = s->Q_mean[a];

        if (sample > best_sample) { best_sample = sample; best_a = a; }
    }

    /* 早停检测 */
    if (best_a == s->best_action) {
        s->best_streak++;
        if (s->best_streak >= s->early_stop_thresh) s->converged = 1;
    } else {
        s->best_action = best_a;
        s->best_streak = 0;
    }

    return best_a;
}

void exploration_thompson_update(void* state, int action, float reward) {
    ThompsonState* s = (ThompsonState*)state;
    if (!s || action < 0 || action >= s->num_actions) return;
    /* Bernoulli/Beta共轭: 奖励clip到[0,1], α+=r, β+=(1-r) */
    float r = (reward > 1.0f) ? 1.0f : ((reward < 0.0f) ? 0.0f : reward);
    s->N[action]++;
    s->alpha[action] += r;
    s->beta_param[action] += (1.0f - r);
    /* 防止过大数据 */
    if (s->alpha[action] > 1e6f) { s->alpha[action] *= 0.99f; s->beta_param[action] *= 0.99f; }
}

float exploration_thompson_get_best(void* state) {
    ThompsonState* s = (ThompsonState*)state;
    if (!s || s->num_actions <= 0) return 0.0f;
    float best = s->Q_mean[0];
    for (int a = 1; a < s->num_actions; a++) {
        if (s->Q_mean[a] > best) best = s->Q_mean[a];
    }
    return best;
}

void exploration_thompson_get_posterior(void* state, int action, float* alpha_out, float* beta_out) {
    ThompsonState* s = (ThompsonState*)state;
    if (!s || action < 0 || action >= s->num_actions) {
        if (alpha_out) *alpha_out = 0.0f;
        if (beta_out) *beta_out = 0.0f;
        return;
    }
    if (alpha_out) *alpha_out = s->alpha[action];
    if (beta_out) *beta_out = s->beta_param[action];
}

#endif /* SELFLNN_SKIP_ADVANCED_EXPLORATION */

/* ============================================================================
 * D-013: 探索策略统一管理API
 * ============================================================================ */

/* 探索策略类型枚举（内部使用） */
typedef enum {
    EXPLORE_ALGO_UCB = 0,
    EXPLORE_ALGO_BOLTZMANN = 1,
    EXPLORE_ALGO_THOMPSON = 2
} ExploreAlgoType;

/* 获取UCB收敛状态（始终编译——UCB被实际使用） */
int exploration_ucb_converged(void* state) {
    return exploration_ucb_is_converged(state);
}

/* DEADCODE-FIX: Boltzmann/Thompson尾部包装函数也受条件编译控制 */
#ifndef SELFLNN_SKIP_ADVANCED_EXPLORATION

/* Boltzmann温度设置（运行时动态调整） */
void exploration_boltzmann_set_temperature(void* state, float T) {
    BoltzmannState* s = (BoltzmannState*)state;
    if (!s) return;
    s->T_current = T;
    if (T > s->T_max) s->T_max = T;
    if (T < s->T_min) s->T_min = T;
}

/* Boltzmann收敛状态查询 */
int exploration_boltzmann_is_converged(void* state) {
    BoltzmannState* s = (BoltzmannState*)state;
    if (!s) return 0;
    return s->converged;
}

/* Thompson收敛状态查询 */
int exploration_thompson_is_converged(void* state) {
    ThompsonState* s = (ThompsonState*)state;
    if (!s) return 0;
    return s->converged;
}

#endif /* SELFLNN_SKIP_ADVANCED_EXPLORATION */
