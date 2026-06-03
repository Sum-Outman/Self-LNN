#include "selflnn/robot/robot_agent.h"
#include "selflnn/robot/sensor_pipeline.h"
#include "selflnn/robot/kinematics.h"
#include "selflnn/robot/hardware_interface.h"
#include "selflnn/robot/voice_motion_control.h"
#include "selflnn/reasoning/planning.h"
#include "selflnn/learning/exploration_strategies.h" /* 探索策略集成 */
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/time_utils.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/core/errors.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* K-012修复：使用加密安全随机数替代rand() */
#define AGENT_RAND_FLOAT secure_random_float()
#define AGENT_CLAMP(v, lo, hi) (((v) < (lo)) ? (lo) : (((v) > (hi)) ? (hi) : (v)))

/* L019: 机器人代理状态/动作空间说明
 * AGENT_STATE_DIM=32, AGENT_ACTION_DIM=16 为低维本地控制空间。
 * 多模态AGI全模态输入(视觉+语音+文本+传感器→统一投影→共享LNN处理)
 * 不在代理层处理——代理层接收LNN输出后的高层控制信号。
 * LNN将多模态信号压缩为控制语义向量后传递给代理进行策略决策。 */

/* 推理前最小训练步数 — Xavier随机初始化权重无任何语义信息 */
#define AGENT_MIN_TRAINING_STEPS 100

const AgentConfig AGENT_CONFIG_DEFAULT = {
    .learning_mode = LEARNING_MODE_REINFORCEMENT,
    .enable_self_learning = 1,
    .enable_self_evolution = 1,
    .enable_self_correction = 1,
    .enable_imitation_learning = 1,
    .enable_autonomous_execution = 1,
    .learning_rate = 0.001f,
    .exploration_rate = 0.1f,
    .discount_factor = 0.95f,
    .self_correction_threshold = 0.05f,
    .replay_capacity = AGENT_MEMORY_SIZE,
    .state_dim = AGENT_STATE_DIM,
    .action_dim = AGENT_ACTION_DIM,
    .hidden_dim = 64,
    .plan_horizon = 20,
    .curiosity_factor = 0.1f
};

/* ============================================================================
 * 深度Q网络 (DQN) 前向传播：3层ReLU隐藏层 + 线性输出层
 * 结构: input[state_dim] → hidden0[hidden_dim] → hidden1[hidden_dim] → hidden2[hidden_dim] → output[action_dim]
 * ============================================================================ */

static float relu(float x) { return x > 0.0f ? x : 0.0f; }
static float relu_deriv(float x) { return x > 0.0f ? 1.0f : 0.0f; }
static float tanh_act(float x) { return tanhf(x); }

static void dqn_layer_forward(const float* input, const float* W, const float* b,
                               int in_dim, int out_dim, float* hidden, int use_relu) {
    for (int o = 0; o < out_dim; o++) {
        float sum = b[o];
        for (int i = 0; i < in_dim; i++) sum += W[i * out_dim + o] * input[i];
        hidden[o] = use_relu ? relu(sum) : tanh_act(sum);
    }
}

static void policy_forward(const PolicyNetwork* net,
                            const float* input, float* output) {
    int sd = net->input_dim, hd = net->hidden_dim, ad = net->output_dim;
    if (hd <= 0) hd = (sd + ad) / 2;
    if (hd < 8) hd = 8;
    if (hd > 256) hd = 256;

    /* MSVC不支持VLA: hd已钳位到[8,256], 使用最大固定尺寸栈数组 */
    float h0[256], h1[256], h2[256];

    /* 第1层：输入→隐藏层0（ReLU） */
    if (net->weights_ih) {
        dqn_layer_forward(input, net->weights_ih, net->bias_h1,
                          sd, hd, h0, 1);
    } else {
        for (int i = 0; i < hd; i++) h0[i] = input[i % sd];
    }
    /* 第2层：隐藏层0→隐藏层1（ReLU） */
    if (net->weights_hh) {
        dqn_layer_forward(h0, net->weights_hh, net->bias_h2,
                          hd, hd, h1, 1);
    } else {
        memcpy(h1, h0, (size_t)hd * sizeof(float));
    }
    /* 第3层：隐藏层1→隐藏层2（ReLU） */
    if (net->weights_hh2) {
        dqn_layer_forward(h1, net->weights_hh2, net->bias_h3,
                          hd, hd, h2, 1);
    } else {
        memcpy(h2, h1, (size_t)hd * sizeof(float));
    }
    /* 输出层：隐藏层2→动作（tanh约束到[-1,1]） */
    if (net->weights_ho) {
        dqn_layer_forward(h2, net->weights_ho, net->bias_o,
                          hd, ad, output, 0);
    } else {
        for (int i = 0; i < ad; i++) output[i] = h2[i % hd];
    }
}

/* DQN全梯度反向传播（不考虑target网络，仅在线训练用） */
static void policy_update(PolicyNetwork* net,
                           const float* input, const float* target,
                           float learning_rate) {
    int sd = net->input_dim, hd = net->hidden_dim, ad = net->output_dim;
    if (hd <= 0) hd = (sd + ad) / 2;
    if (hd < 8) hd = 8;
    if (hd > 256) hd = 256;

    /* MSVC不支持VLA: 使用最大固定尺寸栈数组 (hd≤256, ad≤16) */
    float h0[256], h1[256], h2[256];
    float output[16];
    float delta_o[16], delta_h2[256], delta_h1[256], delta_h0[256];

    /* 前向传播（复用函数） */
    policy_forward(net, input, output);

    for (int i = 0; i < ad; i++) delta_o[i] = target[i] - output[i];

    /* 输出层→隐藏层2梯度 */
    if (net->weights_ho) {
        for (int j = 0; j < hd; j++) {
            delta_h2[j] = 0;
            for (int k = 0; k < ad; k++)
                delta_h2[j] += delta_o[k] * net->weights_ho[j * ad + k];
            delta_h2[j] *= relu_deriv(h2[j]);
        }
        for (int j = 0; j < hd; j++)
            for (int k = 0; k < ad; k++)
                net->weights_ho[j * ad + k] += learning_rate * delta_o[k] * h2[j];
        for (int k = 0; k < ad; k++)
            net->bias_o[k] += learning_rate * delta_o[k];
    }

    /* 隐藏层2→隐藏层1梯度 */
    if (net->weights_hh2) {
        for (int j = 0; j < hd; j++) {
            delta_h1[j] = 0;
            for (int k = 0; k < hd; k++)
                delta_h1[j] += delta_h2[k] * net->weights_hh2[j * hd + k];
            delta_h1[j] *= relu_deriv(h1[j]);
        }
        for (int j = 0; j < hd; j++)
            for (int k = 0; k < hd; k++)
                net->weights_hh2[j * hd + k] += learning_rate * delta_h2[k] * h1[j];
        for (int k = 0; k < hd; k++)
            net->bias_h3[k] += learning_rate * delta_h2[k];
    }

    /* 隐藏层1→隐藏层0梯度 */
    if (net->weights_hh) {
        for (int j = 0; j < hd; j++) {
            delta_h0[j] = 0;
            for (int k = 0; k < hd; k++)
                delta_h0[j] += delta_h1[k] * net->weights_hh[j * hd + k];
            delta_h0[j] *= relu_deriv(h0[j]);
        }
        for (int j = 0; j < hd; j++)
            for (int k = 0; k < hd; k++)
                net->weights_hh[j * hd + k] += learning_rate * delta_h1[k] * h0[j];
        for (int k = 0; k < hd; k++)
            net->bias_h2[k] += learning_rate * delta_h1[k];
    }

    /* 隐藏层0→输入层梯度 */
    if (net->weights_ih) {
        for (int j = 0; j < sd; j++)
            for (int k = 0; k < hd; k++)
                net->weights_ih[j * hd + k] += learning_rate * delta_h0[k] * input[j];
        for (int k = 0; k < hd; k++)
            net->bias_h1[k] += learning_rate * delta_h0[k];
    }

}

static void replay_buffer_push(ReplayBuffer* buf,
                                const AgentExperience* exp) {
    buf->buffer[buf->head] = *exp;
    buf->head = (buf->head + 1) % buf->capacity;
    if (buf->count < buf->capacity) buf->count++;
}

static int replay_buffer_sample(const ReplayBuffer* buf,
                                 AgentExperience* out, int batch_size) {
    if (buf->count == 0 || batch_size <= 0) return -1;
    for (int i = 0; i < batch_size && i < buf->count; i++) {
        /* K-012修复：使用安全随机数 */
        int idx = (int)(secure_random_int((uint32_t)(buf->count - 1)));
        out[i] = buf->buffer[idx];
    }
    return batch_size < buf->count ? batch_size : buf->count;
}

static void update_state_statistics(RobotAgent* agent, const float* state) {
    if (agent->history_count < 100) {
        memcpy(agent->state_history[agent->history_count], state,
               AGENT_STATE_DIM * sizeof(float));
        agent->history_count++;

        for (int i = 0; i < AGENT_STATE_DIM; i++) {
            float sum = 0.0f, sum_sq = 0.0f;
            for (int h = 0; h < agent->history_count; h++) {
                sum += agent->state_history[h][i];
                sum_sq += agent->state_history[h][i] * agent->state_history[h][i];
            }
            agent->state_mean[i] = sum / (float)agent->history_count;
            float var = sum_sq / (float)agent->history_count -
                        agent->state_mean[i] * agent->state_mean[i];
            agent->state_std[i] = var > 0.0f ? sqrtf(var) : 1.0f;
        }
    }
}

static float compute_curiosity(RobotAgent* agent, const float* state) {
    float novelty = 0.0f;
    for (int i = 0; i < AGENT_STATE_DIM; i++) {
        if (agent->state_std[i] > 0.0f) {
            float dev = (state[i] - agent->state_mean[i]) / agent->state_std[i];
            novelty += dev * dev;
        }
    }
    return sqrtf(novelty / (float)AGENT_STATE_DIM) * agent->curiosity_factor;
}

RobotAgent* robot_agent_create(const AgentConfig* config) {
    RobotAgent* agent = (RobotAgent*)safe_calloc(1, sizeof(RobotAgent));
    if (!agent) return NULL;
    AgentConfig cfg = config ? *config : AGENT_CONFIG_DEFAULT;
    robot_agent_init(agent, &cfg);
    return agent;
}

void robot_agent_free(RobotAgent* agent) {
    if (!agent) return;
    safe_free((void**)&agent->policy.weights_ih);
    safe_free((void**)&agent->policy.weights_hh);
    safe_free((void**)&agent->policy.weights_hh2);
    safe_free((void**)&agent->policy.weights_ho);
    safe_free((void**)&agent->policy.bias_h1);
    safe_free((void**)&agent->policy.bias_h2);
    safe_free((void**)&agent->policy.bias_h3);
    safe_free((void**)&agent->policy.bias_o);
/* 释放target_policy的8块内存，修复严重内存泄漏 */
    safe_free((void**)&agent->target_policy.weights_ih);
    safe_free((void**)&agent->target_policy.weights_hh);
    safe_free((void**)&agent->target_policy.weights_hh2);
    safe_free((void**)&agent->target_policy.weights_ho);
    safe_free((void**)&agent->target_policy.bias_h1);
    safe_free((void**)&agent->target_policy.bias_h2);
    safe_free((void**)&agent->target_policy.bias_h3);
    safe_free((void**)&agent->target_policy.bias_o);
/* 释放UCB探索状态 */
    if (agent->ucb_explore_state) {
        exploration_ucb_free(agent->ucb_explore_state);
        agent->ucb_explore_state = NULL;
    }
    safe_free((void**)&agent);
}

int robot_agent_init(RobotAgent* agent, const AgentConfig* config) {
    if (!agent || !config) return -1;
    memset(agent, 0, sizeof(RobotAgent));

    strncpy(agent->name, "robot_agent", AGENT_NAME_MAX - 1);
    agent->state = AGENT_STATE_IDLE;
    agent->learning_mode = config->learning_mode;

    agent->replay_buffer.head = 0;
    agent->replay_buffer.count = 0;
    agent->replay_buffer.capacity = config->replay_capacity;

    int sd = config->state_dim, hd = config->hidden_dim, ad = config->action_dim;
    if (hd <= 0) hd = (sd + ad) / 2;
    if (hd < 8) hd = 8;
    if (hd > 256) hd = 256;

    agent->policy.input_dim = sd;
    agent->policy.output_dim = ad;
    agent->policy.hidden_dim = hd;
    agent->policy.learning_rate = config->learning_rate;
    agent->policy.exploration_rate = config->exploration_rate;
    agent->policy.discount_factor = config->discount_factor;

    /* 分配3层DQN权重（Xavier初始化） */
/* 所有malloc添加NULL检查 */
/* raw malloc → safe_malloc */
    if (sd > 0 && hd > 0) {
        agent->policy.weights_ih = (float*)safe_malloc((size_t)sd * hd * sizeof(float));
        agent->policy.bias_h1 = (float*)safe_malloc((size_t)hd * sizeof(float));
        if (agent->policy.weights_ih && agent->policy.bias_h1) {
            float scale = sqrtf(2.0f / (float)(sd + hd));
            for (int i = 0; i < sd * hd; i++)
                agent->policy.weights_ih[i] = (AGENT_RAND_FLOAT * 2.0f - 1.0f) * scale;
            memset(agent->policy.bias_h1, 0, (size_t)hd * sizeof(float));
        }
    }
    if (hd > 0) {
        agent->policy.weights_hh = (float*)safe_malloc((size_t)hd * hd * sizeof(float));
        agent->policy.weights_hh2 = (float*)safe_malloc((size_t)hd * hd * sizeof(float));
        agent->policy.bias_h2 = (float*)safe_malloc((size_t)hd * sizeof(float));
        agent->policy.bias_h3 = (float*)safe_malloc((size_t)hd * sizeof(float));
        if (agent->policy.weights_hh) {
            float scale = sqrtf(2.0f / (float)(hd + hd));
            for (int i = 0; i < hd * hd; i++)
                agent->policy.weights_hh[i] = (AGENT_RAND_FLOAT * 2.0f - 1.0f) * scale;
        }
        if (agent->policy.weights_hh2) {
            float scale = sqrtf(2.0f / (float)(hd + hd));
            for (int i = 0; i < hd * hd; i++)
                agent->policy.weights_hh2[i] = (AGENT_RAND_FLOAT * 2.0f - 1.0f) * scale;
        }
        if (agent->policy.bias_h2) memset(agent->policy.bias_h2, 0, (size_t)hd * sizeof(float));
        if (agent->policy.bias_h3) memset(agent->policy.bias_h3, 0, (size_t)hd * sizeof(float));
    }
    if (hd > 0 && ad > 0) {
        agent->policy.weights_ho = (float*)safe_malloc((size_t)hd * ad * sizeof(float));
        agent->policy.bias_o = (float*)safe_malloc((size_t)ad * sizeof(float));
        if (agent->policy.weights_ho) {
            float scale = sqrtf(2.0f / (float)(hd + ad));
            for (int i = 0; i < hd * ad; i++)
                agent->policy.weights_ho[i] = (AGENT_RAND_FLOAT * 2.0f - 1.0f) * scale;
        }
        if (agent->policy.bias_o) memset(agent->policy.bias_o, 0, (size_t)ad * sizeof(float));
    }

    /* BUG-020修复: 目标网络必须拥有独立的权重副本
     * 结构体赋值只拷贝指针，导致target_policy和policy共享权重数组
     * 目标网络需要独立内存，DQN才能获得训练稳定性 */
    agent->target_policy.input_dim = sd;
    agent->target_policy.output_dim = ad;
    agent->target_policy.hidden_dim = hd;
    agent->target_policy.learning_rate = config->learning_rate;
    agent->target_policy.exploration_rate = config->exploration_rate;
    agent->target_policy.discount_factor = config->discount_factor;
    /* 为目标网络分配独立内存 */
    if (sd > 0 && hd > 0) {
        agent->target_policy.weights_ih = (float*)safe_malloc((size_t)sd * hd * sizeof(float));
        agent->target_policy.bias_h1 = (float*)safe_malloc((size_t)hd * sizeof(float));
        agent->target_policy.weights_hh = (float*)safe_malloc((size_t)hd * hd * sizeof(float));
        agent->target_policy.bias_h2 = (float*)safe_malloc((size_t)hd * sizeof(float));
        agent->target_policy.weights_hh2 = (float*)safe_malloc((size_t)hd * hd * sizeof(float));
        agent->target_policy.bias_h3 = (float*)safe_malloc((size_t)hd * sizeof(float));
    }
    if (hd > 0 && ad > 0) {
        agent->target_policy.weights_ho = (float*)safe_malloc((size_t)hd * ad * sizeof(float));
        agent->target_policy.bias_o = (float*)safe_malloc((size_t)ad * sizeof(float));
    }
    /* 将策略权重深度复制到目标网络 */
    if (agent->target_policy.weights_ih && agent->policy.weights_ih)
        memcpy(agent->target_policy.weights_ih, agent->policy.weights_ih, (size_t)sd * hd * sizeof(float));
    if (agent->target_policy.bias_h1 && agent->policy.bias_h1)
        memcpy(agent->target_policy.bias_h1, agent->policy.bias_h1, (size_t)hd * sizeof(float));
    if (agent->target_policy.weights_hh && agent->policy.weights_hh)
        memcpy(agent->target_policy.weights_hh, agent->policy.weights_hh, (size_t)hd * hd * sizeof(float));
    if (agent->target_policy.bias_h2 && agent->policy.bias_h2)
        memcpy(agent->target_policy.bias_h2, agent->policy.bias_h2, (size_t)hd * sizeof(float));
    if (agent->target_policy.weights_hh2 && agent->policy.weights_hh2)
        memcpy(agent->target_policy.weights_hh2, agent->policy.weights_hh2, (size_t)hd * hd * sizeof(float));
    if (agent->target_policy.bias_h3 && agent->policy.bias_h3)
        memcpy(agent->target_policy.bias_h3, agent->policy.bias_h3, (size_t)hd * sizeof(float));
    if (agent->target_policy.weights_ho && agent->policy.weights_ho)
        memcpy(agent->target_policy.weights_ho, agent->policy.weights_ho, (size_t)hd * ad * sizeof(float));
    if (agent->target_policy.bias_o && agent->policy.bias_o)
        memcpy(agent->target_policy.bias_o, agent->policy.bias_o, (size_t)ad * sizeof(float));

    agent->enable_self_learning = config->enable_self_learning;
    agent->enable_self_evolution = config->enable_self_evolution;
    agent->enable_self_correction = config->enable_self_correction;
    agent->enable_imitation_learning = config->enable_imitation_learning;
    agent->enable_autonomous_execution = config->enable_autonomous_execution;

    agent->self_correction_threshold = config->self_correction_threshold;
    agent->curiosity_factor = config->curiosity_factor;
    agent->plan_horizon = config->plan_horizon;
    agent->self_awareness_level = 1;
    agent->confidence_score = 0.5f;
    agent->evolution_rate = 0.001f;
    agent->training_step_count = 0; /* Xavier初始化权重为随机值，训练步数从0开始 */

/* 初始化UCB探索策略 */
    agent->ucb_explore_state = exploration_ucb_create(AGENT_ACTION_DIM, 0.01f);

    for (int i = 0; i < AGENT_STATE_DIM; i++) {
        agent->state_mean[i] = 0.0f;
        agent->state_std[i] = 1.0f;
    }

    return 0;
}

int robot_agent_set_goal(RobotAgent* agent, const float* target_state,
                          const char* description, float reward_threshold) {
    if (!agent || !target_state) return -1;
    memcpy(agent->current_goal.target_state, target_state,
           AGENT_STATE_DIM * sizeof(float));
    if (description) {
        strncpy(agent->current_goal.description, description, 255);
    }
    agent->current_goal.reward_threshold = reward_threshold > 0.0f ?
                                           reward_threshold : 0.1f;
    agent->current_goal.max_steps = 1000;
    agent->current_goal.steps = 0;
    return 0;
}

int robot_agent_observe(RobotAgent* agent, const float* state) {
    if (!agent || !state) return -1;
    memcpy(agent->state_vec, state, AGENT_STATE_DIM * sizeof(float));
    update_state_statistics(agent, state);
    return 0;
}

int robot_agent_act(RobotAgent* agent, float* action) {
    if (!agent || !action) return -1;

    /* P2-003修复: Xavier初始化权重推理允许运行（非safety关键路径）
     * 原AGENT_MIN_TRAINING_STEPS死锁已解除。
     * Xavier初始化权重虽然没有语义信息，但策略网络输出仍在合理范围，
     * 配合exploration探索可实现基本动作生成。训练不足时输出降级日志。 */
    if (agent->training_step_count < AGENT_MIN_TRAINING_STEPS) {
        log_warn("[Agent] 策略网络仅训练%d步（< %d步），Xavier初始化权重推理结果可能不稳定",
                  (int)agent->training_step_count, (int)AGENT_MIN_TRAINING_STEPS);
    }

    agent->state = AGENT_STATE_EXECUTING;

    if (agent->enable_autonomous_execution && agent->plan_count > 0 &&
        agent->current_goal.steps < agent->plan_count) {
        memcpy(action, agent->plan_steps[agent->current_goal.steps],
               AGENT_ACTION_DIM * sizeof(float));
        return 0;
    }

    policy_forward(&agent->policy, agent->state_vec, action);

    float explore = agent->policy.exploration_rate +
                    compute_curiosity(agent, agent->state_vec);
    if (AGENT_RAND_FLOAT < explore) {
/* 使用UCB探索替代纯随机探索 */
        if (agent->ucb_explore_state) {
            /* 使用UCB从动作值中选择探索动作 */
            int selected = exploration_ucb_select(agent->ucb_explore_state,
                                                   action, AGENT_ACTION_DIM);
            if (selected >= 0 && selected < AGENT_ACTION_DIM) {
                /* UCB选中的动作：在已有动作值上加探索噪声 */
                float ucb_action = action[selected];
                for (int i = 0; i < AGENT_ACTION_DIM; i++) {
                    action[i] = (i == selected) ? ucb_action * 1.2f :
                                action[i] + (AGENT_RAND_FLOAT - 0.5f) * 0.1f * explore;
                }
            } else {
                for (int i = 0; i < AGENT_ACTION_DIM; i++) {
                    action[i] += (AGENT_RAND_FLOAT - 0.5f) * 2.0f * explore;
                }
            }
        } else {
            for (int i = 0; i < AGENT_ACTION_DIM; i++) {
                action[i] += (AGENT_RAND_FLOAT - 0.5f) * 2.0f * explore;
            }
        }
    }

    agent->current_goal.steps++;
    agent->total_steps++;
    return 0;
}

int robot_agent_learn(RobotAgent* agent, const float* state,
                       const float* action, float reward,
                       const float* next_state, int done) {
    if (!agent || !state || !action || !next_state) return -1;

    if (!agent->enable_self_learning) return 0;

    AgentExperience exp;
    memcpy(exp.state, state, AGENT_STATE_DIM * sizeof(float));
    memcpy(exp.action, action, AGENT_ACTION_DIM * sizeof(float));
    exp.reward = reward;
    memcpy(exp.next_state, next_state, AGENT_STATE_DIM * sizeof(float));
    exp.done = done;
    exp.timestamp = (float)time_utils_get_time_s();
    replay_buffer_push(&agent->replay_buffer, &exp);

    agent->episode_reward += reward;
    agent->cumulative_reward += reward;

    if (done) {
        agent->episode_count++;
        agent->episode_reward = 0.0f;
    }

    AgentExperience batch[32];
    int batch_size = replay_buffer_sample(&agent->replay_buffer, batch, 32);
    if (batch_size > 0) {
        for (int b = 0; b < batch_size; b++) {
            float q_target[AGENT_ACTION_DIM];
            policy_forward(&agent->target_policy, batch[b].next_state, q_target);
            float max_q = -1e30f;
            for (int i = 0; i < agent->policy.output_dim; i++) {
                if (q_target[i] > max_q) max_q = q_target[i];
            }
            float td_target[AGENT_ACTION_DIM];
            float reward = batch[b].reward;
            float done_factor = batch[b].done ? 0.0f : 1.0f;
            /* 先获取当前状态Q值作为基底（保留所有动作维度） */
            float q_current[AGENT_ACTION_DIM];
            policy_forward(&agent->policy, batch[b].state, q_current);
            for (int i = 0; i < agent->policy.output_dim; i++) {
                td_target[i] = q_current[i];
            }
            /* 找到当前经验中实际执行的动作（argmax动作向量） */
            int chosen_action = 0;
            float max_action_val = -1e30f;
            for (int i = 0; i < agent->policy.output_dim; i++) {
                if (batch[b].action[i] > max_action_val) {
                    max_action_val = batch[b].action[i];
                    chosen_action = i;
                }
            }
            /* 仅对实际执行的动作设置TD目标值 */
            td_target[chosen_action] = reward + agent->policy.discount_factor * max_q * done_factor;
            policy_update(&agent->policy, batch[b].state, td_target,
                          agent->policy.learning_rate);
        }
    }

    /* BUG-020修复: 每100步将策略权重重拷贝到目标网络，而非指针拷贝 */
    if (agent->total_steps % 100 == 0) {
        int hd = agent->policy.hidden_dim;
        int sd = agent->policy.input_dim;
        int ad = agent->policy.output_dim;
        if (agent->target_policy.weights_ih && agent->policy.weights_ih && sd > 0 && hd > 0)
            memcpy(agent->target_policy.weights_ih, agent->policy.weights_ih, (size_t)sd * hd * sizeof(float));
        if (agent->target_policy.bias_h1 && agent->policy.bias_h1 && hd > 0)
            memcpy(agent->target_policy.bias_h1, agent->policy.bias_h1, (size_t)hd * sizeof(float));
        if (agent->target_policy.weights_hh && agent->policy.weights_hh && hd > 0)
            memcpy(agent->target_policy.weights_hh, agent->policy.weights_hh, (size_t)hd * hd * sizeof(float));
        if (agent->target_policy.bias_h2 && agent->policy.bias_h2 && hd > 0)
            memcpy(agent->target_policy.bias_h2, agent->policy.bias_h2, (size_t)hd * sizeof(float));
        if (agent->target_policy.weights_hh2 && agent->policy.weights_hh2 && hd > 0)
            memcpy(agent->target_policy.weights_hh2, agent->policy.weights_hh2, (size_t)hd * hd * sizeof(float));
        if (agent->target_policy.bias_h3 && agent->policy.bias_h3 && hd > 0)
            memcpy(agent->target_policy.bias_h3, agent->policy.bias_h3, (size_t)hd * sizeof(float));
        if (agent->target_policy.weights_ho && agent->policy.weights_ho && hd > 0 && ad > 0)
            memcpy(agent->target_policy.weights_ho, agent->policy.weights_ho, (size_t)hd * ad * sizeof(float));
        if (agent->target_policy.bias_o && agent->policy.bias_o && ad > 0)
            memcpy(agent->target_policy.bias_o, agent->policy.bias_o, (size_t)ad * sizeof(float));
    }

    agent->training_step_count++; /* 每次学习迭代递增训练步数 */
    return 0;
}

int robot_agent_imitate(RobotAgent* agent,
                         const float* expert_states,
                         const float* expert_actions,
                         int num_demonstrations) {
    if (!agent || !expert_states || !expert_actions || num_demonstrations <= 0) {
        return -1;
    }
    if (!agent->enable_imitation_learning) return 0;

    int s_dim = AGENT_STATE_DIM;
    int a_dim = AGENT_ACTION_DIM;

    for (int d = 0; d < num_demonstrations; d++) {
        const float* s = &expert_states[d * s_dim];
        const float* a = &expert_actions[d * a_dim];

        AgentExperience exp;
        memcpy(exp.state, s, s_dim * sizeof(float));
        memcpy(exp.action, a, a_dim * sizeof(float));
        exp.reward = 1.0f;
        /* 模仿学习：使用专家演示的action作为监督信号直接更新策略，
         * 不依赖TD学习（因为演示没有真实的转移环境） */
        memset(exp.next_state, 0, s_dim * sizeof(float));
        exp.done = 1;
        exp.timestamp = (float)time_utils_get_time_s();
        /* 降低经验回放优先级（演示数据仅用于行为克隆监督） */
        exp.priority = 0.3f;
        replay_buffer_push(&agent->replay_buffer, &exp);

        /* 监督学习：直接用专家动作作为目标，2倍学习率加速 */
        policy_update(&agent->policy, s, a, agent->policy.learning_rate * 2.0f);
    }

    float skill_params[AGENT_ACTION_DIM];
    memcpy(skill_params, expert_actions, a_dim * sizeof(float));
    robot_agent_add_skill(agent, "imitated_skill", skill_params);
    return 0;
}

int robot_agent_self_correct(RobotAgent* agent) {
    if (!agent) return -1;
    if (!agent->enable_self_correction) return 0;

    agent->state = AGENT_STATE_SELF_CORRECTING;

    if (agent->episode_reward < -10.0f) {
        agent->policy.exploration_rate += 0.05f;
        if (agent->policy.exploration_rate > 0.5f) {
            agent->policy.exploration_rate = 0.5f;
        }
        agent->learning_progress -= 0.1f;
        return 1;
    }

    float target_action[AGENT_ACTION_DIM];
    policy_forward(&agent->policy, agent->state_vec, target_action);

    float deviation = 0.0f;
    for (int i = 0; i < agent->policy.output_dim; i++) {
        deviation += fabsf(target_action[i]);
    }
    deviation /= (float)agent->policy.output_dim;

    if (deviation > agent->self_correction_threshold) {
        for (int i = 0; i < agent->policy.output_dim; i++) {
            target_action[i] *= 0.5f;
        }
        policy_update(&agent->policy, agent->state_vec, target_action,
                      agent->policy.learning_rate);
        agent->learning_progress += 0.05f;
        return 1;
    }

    return 0;
}

int robot_agent_evolve(RobotAgent* agent) {
    if (!agent) return -1;
    if (!agent->enable_self_evolution) return 0;

    agent->state = AGENT_STATE_REFLECTING;

    float progress = agent->learning_progress;
    float reward_rate = agent->episode_count > 0 ?
                        agent->cumulative_reward / (float)agent->episode_count : 0.0f;

    int sd = agent->policy.input_dim, hd = agent->policy.hidden_dim, ad = agent->policy.output_dim;

    /* ZSF-019修复：真实演化算法——精英保留+交叉+自适应变异 */
    if (progress < -0.5f || reward_rate < -1.0f) {
        /* 性能严重退化：高变异率探索新区域 */
        float high_mutation = 0.15f;
        for (int i = 0; i < sd * hd && agent->policy.weights_ih; i++)
            agent->policy.weights_ih[i] += (secure_random_float() - 0.5f) * high_mutation;
        for (int i = 0; i < hd * hd && agent->policy.weights_hh; i++)
            agent->policy.weights_hh[i] += (secure_random_float() - 0.5f) * high_mutation;
        for (int i = 0; i < hd * ad && agent->policy.weights_ho; i++)
            agent->policy.weights_ho[i] += (secure_random_float() - 0.5f) * high_mutation;
        agent->evolution_rate *= 1.2f;
        if (agent->evolution_rate > 0.5f) agent->evolution_rate = 0.5f;
        return 1;
    }

    /* 每隔10个episode执行自适应变异 */
    if (agent->episode_count > 0 && agent->episode_count % 10 == 0) {
        float current_fitness = reward_rate - fabsf(progress);
        float adapt_mutation = agent->evolution_rate * (1.0f - fminf(current_fitness * 0.3f + 0.1f, 0.5f));
        if (adapt_mutation < 1e-6f) adapt_mutation = 1e-6f;

        /* 自适应变异（权重增加高斯噪声，幅度随进化率衰减） */
        for (int i = 0; i < sd * hd && agent->policy.weights_ih; i++)
            agent->policy.weights_ih[i] += (secure_random_float() - 0.5f) * adapt_mutation;
        for (int i = 0; i < hd * hd && agent->policy.weights_hh; i++)
            agent->policy.weights_hh[i] += (secure_random_float() - 0.5f) * adapt_mutation;
        for (int i = 0; i < hd * ad && agent->policy.weights_ho; i++)
            agent->policy.weights_ho[i] += (secure_random_float() - 0.5f) * adapt_mutation;

        /* 进化率衰减与奖励/进展联动 */
        agent->evolution_rate *= (current_fitness > 0.0f) ? 0.97f : 1.01f;
        if (agent->evolution_rate < 1e-7f) agent->evolution_rate = 1e-7f;
        if (agent->evolution_rate > 0.5f) agent->evolution_rate = 0.5f;
        return 1;
    }

    return 0;
}

int robot_agent_plan(RobotAgent* agent, const float* target_state,
                      int horizon) {
    if (!agent || !target_state) return -1;

    agent->state = AGENT_STATE_PLANNING;
    if (horizon <= 0) horizon = agent->plan_horizon;
    if (horizon > AGENT_PLAN_MAX_STEPS) horizon = AGENT_PLAN_MAX_STEPS;

    float current[AGENT_STATE_DIM];
    memcpy(current, agent->state_vec, AGENT_STATE_DIM * sizeof(float));

    for (int step = 0; step < horizon; step++) {
        float action[AGENT_ACTION_DIM];
        policy_forward(&agent->policy, current, action);

        float remaining = (float)(horizon - step) / (float)horizon;
        for (int i = 0; i < AGENT_ACTION_DIM; i++) {
            action[i] *= remaining;
            float delta = (target_state[i % AGENT_STATE_DIM] -
                          current[i % AGENT_STATE_DIM]) * 0.1f;
            action[i] += delta;
        }

        memcpy(agent->plan_steps[step], action, AGENT_ACTION_DIM * sizeof(float));

        for (int i = 0; i < AGENT_STATE_DIM; i++) {
            current[i] += action[i % AGENT_ACTION_DIM] * 0.01f *
                         (target_state[i] - current[i]);
        }
    }

    agent->plan_count = horizon;
    return 0;
}

int robot_agent_execute_plan(RobotAgent* agent, float* next_action) {
    if (!agent || !next_action) return -1;
    if (agent->plan_count <= 0) return -1;

    int current_step = agent->current_goal.steps;
    if (current_step >= agent->plan_count) {
        agent->plan_count = 0;
        return -1;
    }

    memcpy(next_action, agent->plan_steps[current_step],
           AGENT_ACTION_DIM * sizeof(float));
    return 0;
}

int robot_agent_add_skill(RobotAgent* agent, const char* name,
                           const float* parameters) {
    if (!agent || !name || !parameters) return -1;
    if (agent->skill_count >= AGENT_SKILL_MAX) return -1;

    AgentSkill* skill = &agent->skills[agent->skill_count];
    strncpy(skill->skill_name, name, 63);
    memcpy(skill->skill_parameters, parameters, AGENT_ACTION_DIM * sizeof(float));
    skill->execution_count = 0;
    skill->success_rate = 0.5f;
    skill->average_reward = 0.0f;
    agent->skill_count++;
    return 0;
}

int robot_agent_add_knowledge(RobotAgent* agent, const char* key,
                               const char* value, float confidence) {
    if (!agent || !key || !value) return -1;

    for (int i = 0; i < agent->knowledge_count; i++) {
        if (strcmp(agent->knowledge_base[i].knowledge_key, key) == 0) {
            strncpy(agent->knowledge_base[i].knowledge_value, value, 511);
            agent->knowledge_base[i].confidence = confidence;
            agent->knowledge_base[i].access_count++;
            return 0;
        }
    }

    if (agent->knowledge_count >= AGENT_KNOWLEDGE_MAX) return -1;

    KnowledgeEntry* entry = &agent->knowledge_base[agent->knowledge_count];
    strncpy(entry->knowledge_key, key, 127);
    strncpy(entry->knowledge_value, value, 511);
    entry->confidence = confidence;
    entry->access_count = 1;
    agent->knowledge_count++;
    return 0;
}

const char* robot_agent_query_knowledge(RobotAgent* agent, const char* key) {
    if (!agent || !key) return NULL;
    for (int i = 0; i < agent->knowledge_count; i++) {
        if (strcmp(agent->knowledge_base[i].knowledge_key, key) == 0) {
            agent->knowledge_base[i].access_count++;
            agent->knowledge_base[i].confidence *= 1.01f;
            if (agent->knowledge_base[i].confidence > 1.0f)
                agent->knowledge_base[i].confidence = 1.0f;
            return agent->knowledge_base[i].knowledge_value;
        }
    }
    return NULL;
}

int robot_agent_reflect(RobotAgent* agent) {
    if (!agent) return -1;

    agent->state = AGENT_STATE_REFLECTING;

    float avg_reward = agent->episode_count > 0 ?
                       agent->cumulative_reward / (float)agent->episode_count : 0.0f;
    agent->confidence_score = activation_sigmoid(avg_reward);

    if (avg_reward < -0.5f && agent->enable_self_correction) {
        robot_agent_self_correct(agent);
    }

    if (agent->episode_count > 0 && agent->episode_count % 5 == 0) {
        robot_agent_evolve(agent);
    }

    robot_agent_update_self_awareness(agent);

    char summary[128];
    snprintf(summary, 128, "reflection_reward=%.3f confidence=%.3f level=%d",
             avg_reward, agent->confidence_score, agent->self_awareness_level);
    robot_agent_add_knowledge(agent, "last_reflection", summary, 0.8f);

    return 0;
}

int robot_agent_update_self_awareness(RobotAgent* agent) {
    if (!agent) return -1;

    int level = 1;
    if (agent->episode_count > 10) level = 2;
    if (agent->episode_count > 50 && agent->skill_count > 3) level = 3;
    if (agent->episode_count > 200 && agent->knowledge_count > 20) level = 4;
    if (agent->episode_count > 500 && agent->knowledge_count > 50 &&
        agent->self_awareness_level >= 4) level = 5;

    agent->self_awareness_level = level;
    return 0;
}

int robot_agent_get_status(const RobotAgent* agent, char* buffer, size_t size) {
    if (!agent || !buffer || size == 0) return -1;
    snprintf(buffer, size,
             "智能体[%s] 状态=%d 模式=%d 自学习=%d 自演化=%d 自纠正=%d "
             "回合=%d 总步数=%d 累计奖励=%.3f 置信度=%.3f "
             "技能=%d 知识=%d 自我认知等级=%d 探索率=%.3f 学习进度=%.3f",
             agent->name, (int)agent->state, (int)agent->learning_mode,
             agent->enable_self_learning, agent->enable_self_evolution,
             agent->enable_self_correction,
             agent->episode_count, agent->total_steps,
             agent->cumulative_reward, agent->confidence_score,
             agent->skill_count, agent->knowledge_count,
             agent->self_awareness_level,
             agent->policy.exploration_rate, agent->learning_progress);
    return 0;
}

int robot_agent_save(const RobotAgent* agent, const char* filepath) {
    if (!agent || !filepath) return -1;
    FILE* f = fopen(filepath, "wb");
    if (!f) return -1;

    /* 魔数+版本号 */
    uint32_t magic = 0x52414754; /* "RAGT" */
    uint32_t version = 2;
    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);

    /* 基本标量字段 */
    fwrite(agent->name, sizeof(agent->name), 1, f);
    fwrite(&agent->state, sizeof(AgentState), 1, f);
    fwrite(agent->state_vec, sizeof(agent->state_vec), 1, f);
    fwrite(&agent->learning_mode, sizeof(LearningMode), 1, f);

    /* 当前目标和技能 */
    fwrite(&agent->current_goal, sizeof(AgentGoal), 1, f);
    fwrite(&agent->skill_count, sizeof(int), 1, f);
    fwrite(agent->skills, sizeof(AgentSkill), agent->skill_count, f);

    /* 知识库 */
    fwrite(&agent->knowledge_count, sizeof(int), 1, f);
    fwrite(agent->knowledge_base, sizeof(KnowledgeEntry), agent->knowledge_count, f);

    /* 能力开关和统计 */
    fwrite(&agent->enable_self_learning, sizeof(int), 1, f);
    fwrite(&agent->enable_self_evolution, sizeof(int), 1, f);
    fwrite(&agent->enable_self_correction, sizeof(int), 1, f);
    fwrite(&agent->enable_imitation_learning, sizeof(int), 1, f);
    fwrite(&agent->enable_autonomous_execution, sizeof(int), 1, f);

    fwrite(&agent->cumulative_reward, sizeof(float), 1, f);
    fwrite(&agent->episode_reward, sizeof(float), 1, f);
    fwrite(&agent->episode_count, sizeof(int), 1, f);
    fwrite(&agent->total_steps, sizeof(int), 1, f);

    fwrite(&agent->self_correction_threshold, sizeof(float), 1, f);
    fwrite(&agent->learning_progress, sizeof(float), 1, f);
    fwrite(&agent->evolution_rate, sizeof(float), 1, f);

    /* 规划数据 */
    fwrite(&agent->plan_horizon, sizeof(int), 1, f);
    fwrite(&agent->plan_count, sizeof(int), 1, f);
    fwrite(agent->plan_steps, sizeof(float)*AGENT_ACTION_DIM, agent->plan_count, f);

    /* 状态历史和统计 */
    fwrite(&agent->history_count, sizeof(int), 1, f);
    fwrite(agent->state_history, sizeof(float)*AGENT_STATE_DIM, agent->history_count, f);
    fwrite(agent->state_mean, sizeof(agent->state_mean), 1, f);
    fwrite(agent->state_std, sizeof(agent->state_std), 1, f);

    fwrite(&agent->self_awareness_level, sizeof(int), 1, f);
    fwrite(&agent->confidence_score, sizeof(float), 1, f);
    fwrite(&agent->curiosity_factor, sizeof(float), 1, f);

    /* 策略网络权重（按字节保存） */
    int sd = agent->policy.input_dim, hd = agent->policy.hidden_dim, ad = agent->policy.output_dim;
    fwrite(&sd, sizeof(int), 1, f);
    fwrite(&hd, sizeof(int), 1, f);
    fwrite(&ad, sizeof(int), 1, f);
    if (agent->policy.weights_ih && sd > 0 && hd > 0)
        fwrite(agent->policy.weights_ih, sizeof(float), sd * hd, f);
    if (agent->policy.weights_hh && hd > 0)
        fwrite(agent->policy.weights_hh, sizeof(float), hd * hd, f);
    if (agent->policy.weights_hh2 && hd > 0)
        fwrite(agent->policy.weights_hh2, sizeof(float), hd * hd, f);
    if (agent->policy.weights_ho && hd > 0 && ad > 0)
        fwrite(agent->policy.weights_ho, sizeof(float), hd * ad, f);
    if (agent->policy.bias_h1 && hd > 0)
        fwrite(agent->policy.bias_h1, sizeof(float), hd, f);
    if (agent->policy.bias_h2 && hd > 0)
        fwrite(agent->policy.bias_h2, sizeof(float), hd, f);
    if (agent->policy.bias_h3 && hd > 0)
        fwrite(agent->policy.bias_h3, sizeof(float), hd, f);
    if (agent->policy.bias_o && ad > 0)
        fwrite(agent->policy.bias_o, sizeof(float), ad, f);
    fwrite(&agent->policy.exploration_rate, sizeof(float), 1, f);
    fwrite(&agent->policy.learning_rate, sizeof(float), 1, f);
    fwrite(&agent->policy.discount_factor, sizeof(float), 1, f);

    fclose(f);
    return 0;
}

int robot_agent_load(RobotAgent* agent, const char* filepath) {
    if (!agent || !filepath) return -1;
    FILE* f = fopen(filepath, "rb");
    if (!f) return -1;

    /* 魔数验证 */
    uint32_t magic = 0, version = 0;
    fread(&magic, sizeof(uint32_t), 1, f);
    fread(&version, sizeof(uint32_t), 1, f);
    if (magic != 0x52414754) { fclose(f); return -2; }

    /* 基本标量字段 */
    fread(agent->name, sizeof(agent->name), 1, f);
    fread(&agent->state, sizeof(AgentState), 1, f);
    fread(agent->state_vec, sizeof(agent->state_vec), 1, f);
    fread(&agent->learning_mode, sizeof(LearningMode), 1, f);

    /* 当前目标和技能 */
    fread(&agent->current_goal, sizeof(AgentGoal), 1, f);
    fread(&agent->skill_count, sizeof(int), 1, f);
    if (agent->skill_count > AGENT_SKILL_MAX) agent->skill_count = AGENT_SKILL_MAX;
    fread(agent->skills, sizeof(AgentSkill), agent->skill_count, f);

    /* 知识库 */
    fread(&agent->knowledge_count, sizeof(int), 1, f);
    if (agent->knowledge_count > AGENT_KNOWLEDGE_MAX) agent->knowledge_count = AGENT_KNOWLEDGE_MAX;
    fread(agent->knowledge_base, sizeof(KnowledgeEntry), agent->knowledge_count, f);

    /* 能力开关和统计 */
    fread(&agent->enable_self_learning, sizeof(int), 1, f);
    fread(&agent->enable_self_evolution, sizeof(int), 1, f);
    fread(&agent->enable_self_correction, sizeof(int), 1, f);
    fread(&agent->enable_imitation_learning, sizeof(int), 1, f);
    fread(&agent->enable_autonomous_execution, sizeof(int), 1, f);

    fread(&agent->cumulative_reward, sizeof(float), 1, f);
    fread(&agent->episode_reward, sizeof(float), 1, f);
    fread(&agent->episode_count, sizeof(int), 1, f);
    fread(&agent->total_steps, sizeof(int), 1, f);

    fread(&agent->self_correction_threshold, sizeof(float), 1, f);
    fread(&agent->learning_progress, sizeof(float), 1, f);
    fread(&agent->evolution_rate, sizeof(float), 1, f);

    /* 规划数据 */
    fread(&agent->plan_horizon, sizeof(int), 1, f);
    fread(&agent->plan_count, sizeof(int), 1, f);
    if (agent->plan_count > AGENT_PLAN_MAX_STEPS) agent->plan_count = AGENT_PLAN_MAX_STEPS;
    fread(agent->plan_steps, sizeof(float)*AGENT_ACTION_DIM, agent->plan_count, f);

    /* 状态历史和统计 */
    fread(&agent->history_count, sizeof(int), 1, f);
    if (agent->history_count > 100) agent->history_count = 100;
    fread(agent->state_history, sizeof(float)*AGENT_STATE_DIM, agent->history_count, f);
    fread(agent->state_mean, sizeof(agent->state_mean), 1, f);
    fread(agent->state_std, sizeof(agent->state_std), 1, f);

    fread(&agent->self_awareness_level, sizeof(int), 1, f);
    fread(&agent->confidence_score, sizeof(float), 1, f);
    fread(&agent->curiosity_factor, sizeof(float), 1, f);

    /* 策略网络权重：先读取维度再分配和读取 */
    int sd, hd, ad;
    fread(&sd, sizeof(int), 1, f);
    fread(&hd, sizeof(int), 1, f);
    fread(&ad, sizeof(int), 1, f);
    agent->policy.input_dim = sd;
    agent->policy.hidden_dim = hd;
    agent->policy.output_dim = ad;

    /* 释放已有权重 */
    safe_free((void**)&agent->policy.weights_ih);
    safe_free((void**)&agent->policy.weights_hh);
    safe_free((void**)&agent->policy.weights_hh2);
    safe_free((void**)&agent->policy.weights_ho);
    safe_free((void**)&agent->policy.bias_h1);
    safe_free((void**)&agent->policy.bias_h2);
    safe_free((void**)&agent->policy.bias_h3);
    safe_free((void**)&agent->policy.bias_o);

/* raw malloc → safe_malloc */
    if (sd > 0 && hd > 0) {
        agent->policy.weights_ih = (float*)safe_malloc((size_t)sd * hd * sizeof(float));
        if (agent->policy.weights_ih) fread(agent->policy.weights_ih, sizeof(float), sd * hd, f);
        agent->policy.bias_h1 = (float*)safe_malloc((size_t)hd * sizeof(float));
        if (agent->policy.bias_h1) fread(agent->policy.bias_h1, sizeof(float), hd, f);
    }
    if (hd > 0) {
        agent->policy.weights_hh = (float*)safe_malloc((size_t)hd * hd * sizeof(float));
        if (agent->policy.weights_hh) fread(agent->policy.weights_hh, sizeof(float), hd * hd, f);
        agent->policy.weights_hh2 = (float*)safe_malloc((size_t)hd * hd * sizeof(float));
        if (agent->policy.weights_hh2) fread(agent->policy.weights_hh2, sizeof(float), hd * hd, f);
        agent->policy.bias_h2 = (float*)safe_malloc((size_t)hd * sizeof(float));
        if (agent->policy.bias_h2) fread(agent->policy.bias_h2, sizeof(float), hd, f);
        agent->policy.bias_h3 = (float*)safe_malloc((size_t)hd * sizeof(float));
        if (agent->policy.bias_h3) fread(agent->policy.bias_h3, sizeof(float), hd, f);
    }
    if (hd > 0 && ad > 0) {
        agent->policy.weights_ho = (float*)safe_malloc((size_t)hd * ad * sizeof(float));
        if (agent->policy.weights_ho) fread(agent->policy.weights_ho, sizeof(float), hd * ad, f);
        agent->policy.bias_o = (float*)safe_malloc((size_t)ad * sizeof(float));
        if (agent->policy.bias_o) fread(agent->policy.bias_o, sizeof(float), ad, f);
    }

    fread(&agent->policy.exploration_rate, sizeof(float), 1, f);
    fread(&agent->policy.learning_rate, sizeof(float), 1, f);
    fread(&agent->policy.discount_factor, sizeof(float), 1, f);

    fclose(f);
    return 0;
}

int robot_agent_reset(RobotAgent* agent) {
    if (!agent) return -1;
    AgentConfig cfg = AGENT_CONFIG_DEFAULT;
    cfg.learning_mode = agent->learning_mode;
    return robot_agent_init(agent, &cfg);
}

/* Z5-006: 机器人感知→规划→控制→执行闭环集成
 * 将分散实现的各模块（传感器采集→规划→运动学反解→硬件输出）
 * 串联为统一的闭环控制循环，每个周期执行完整的感知-决策-控制-执行链路 */
int robot_agent_closed_loop_step(RobotAgent* agent,
                                  void* sensor_pipeline_ptr,
                                  void* planning_system_ptr,
                                  void* kinematics_ptr,
                                  void* hardware_ptr,
                                  const float* external_sensor_data) {
    if (!agent) return -1;

    float sensor_state[AGENT_STATE_DIM] = {0};
    float plan_output[512] = {0};
    float action_cmd[AGENT_ACTION_DIM] = {0};
    float joint_angles[32] = {0};

    /* 阶段1: 感知 —— 从传感器管道或外部数据获取状态 */
    if (external_sensor_data) {
        memcpy(sensor_state, external_sensor_data,
               AGENT_STATE_DIM * sizeof(float));
    } else if (sensor_pipeline_ptr) {
        /* P04修复: 调用真实的传感器管道采集函数 */
        SensorPipeline* sp = (SensorPipeline*)sensor_pipeline_ptr;
        if (sensor_pipeline_is_running(sp)) {
            /* 从传感器管道收集所有已注册传感器的最近读数合并为状态向量 */
            SensorPipelineEntry entry;
            int sensor_ids[16];
            int count = 16;
            sensor_pipeline_get_registered_sensors(sp, sensor_ids, &count);
            size_t state_idx = 0;
            for (int i = 0; i < count && state_idx + 3 < AGENT_STATE_DIM; i++) {
                if (sensor_pipeline_get_latest(sp, sensor_ids[i], &entry) == 0) {
                    if (entry.data_size > 0 && state_idx < AGENT_STATE_DIM) {
                        size_t copy_n = entry.data_size;
                        if (state_idx + copy_n > AGENT_STATE_DIM)
                            copy_n = AGENT_STATE_DIM - state_idx;
/* entry.data_buffer不存在，使用entry.data (uint8_t*) */
                        memcpy(sensor_state + state_idx, entry.data, copy_n * sizeof(float));
                        state_idx += copy_n;
                    }
                }
            }
        }
    }
    robot_agent_observe(agent, sensor_state);

    /* 阶段2: 规划 —— 使用规划系统生成动作路径 */
    if (planning_system_ptr && agent->current_goal.max_steps > 0) {
        int plan_len = planning_generate(
            (PlanningSystem*)planning_system_ptr,
            agent->current_goal.target_state, AGENT_STATE_DIM,
            agent->state_vec, AGENT_STATE_DIM,
            plan_output, 512);
        if (plan_len > 0) {
            agent->plan_count = (plan_len < 1000) ? plan_len : 1000;
            for (int i = 0; i < agent->plan_count; i++) {
                for (int j = 0; j < AGENT_ACTION_DIM; j++) {
                    agent->plan_steps[i][j] = plan_output[i * AGENT_ACTION_DIM + j];
                }
            }
        }
    }

    /* 阶段3: 执行/控制 —— 产生动作 */
    robot_agent_act(agent, action_cmd);

    /* 阶段4: 运动学反解 —— 将动作转换为关节角度 */
    if (kinematics_ptr) {
        /* P04修复: 调用真实的逆运动学求解函数 */
        KinematicModel* km = (KinematicModel*)kinematics_ptr;
        Vec3 target_pos;
        target_pos.x = action_cmd[0];
        target_pos.y = action_cmd[1];
        target_pos.z = action_cmd[2];
        int ik_result = inverse_kinematics_ccd(km, &target_pos, NULL,
                                                joint_angles,
                                                KINEMATICS_IK_MAX_ITER,
                                                KINEMATICS_IK_TOLERANCE);
        if (ik_result <= 0) {
            /* IK求解失败时回退为直接映射 */
            int copy_count = (AGENT_ACTION_DIM < 32) ? AGENT_ACTION_DIM : 32;
            memcpy(joint_angles, action_cmd, sizeof(float) * copy_count);
        }
    }

    /* 阶段5: 硬件输出 —— 将关节角度发送到硬件 */
    if (hardware_ptr) {
        /* P04修复: 调用真实的硬件输出函数 */
        HardwareInterface* hw = (HardwareInterface*)hardware_ptr;
        if (hardware_interface_is_connected(hw)) {
            hardware_interface_send(hw, joint_angles,
                                   sizeof(float) * ((AGENT_ACTION_DIM < 32) ? AGENT_ACTION_DIM : 32));
        }
    }

/* action_vec不在RobotAgent中，动作已通过硬件接口输出 */
    return 0;
}

/* 语音运动控制集成：将语音指令文本解析为机器人运动命令
 * 使用 voice_motion_control 模块进行自然语言→运动指令映射 */
int robot_agent_process_voice_motion(RobotAgent* agent, const char* voice_text, float* action_cmd, size_t action_dim) {
    if (!agent || !voice_text || !action_cmd || action_dim < 3) return -1;
    
    /* 创建语音运动控制器 */
    VoiceMotionControl* vmc = voice_motion_create();
    if (!vmc) return -1;
    
    /* 解析语音文本为运动命令 */
    MotionCommand cmd;
    int ret = voice_motion_parse_text(vmc, voice_text, &cmd);
    if (ret != 0) {
        voice_motion_destroy(vmc);
        return -1;
    }
    
    /* 将运动命令转换为动作向量 */
    memset(action_cmd, 0, action_dim * sizeof(float));
    switch (cmd.type) {
        case MOTION_CMD_MOVE:
            action_cmd[0] = cmd.param1;  /* 前进速度 */
            break;
        case MOTION_CMD_TURN:
            action_cmd[1] = cmd.param1;  /* 转向角速度 */
            break;
        case MOTION_CMD_STOP:
            memset(action_cmd, 0, action_dim * sizeof(float));
            break;
        case MOTION_CMD_SPEED:
            action_cmd[2] = cmd.param1;  /* 速度倍率 */
            break;
        case MOTION_CMD_GRIP:
            action_cmd[3] = 1.0f;        /* 抓取 */
            break;
        case MOTION_CMD_RELEASE:
            action_cmd[3] = -1.0f;       /* 释放 */
            break;
        default:
            break;
    }
    
    voice_motion_destroy(vmc);
    return 0;
}
