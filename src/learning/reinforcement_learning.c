/**
 * @file reinforcement_learning.c
 * @brief 强化学习算法实现
 *
 * 实现完整的强化学习算法：
 * - DQN (Deep Q-Network) 带目标网络和优先经验回放
 * - PPO (Proximal Policy Optimization) 带GAE
 * - SAC (Soft Actor-Critic) 带自动熵调整
 * - A2C (Advantage Actor-Critic)
 * - 多种探索策略
 * - 优先经验回放
 */

#include "selflnn/learning/reinforcement_learning.h"
#include "selflnn/learning/exploration_strategies.h" /* ZSFJJJ-H005: 高级探索策略集成 */
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/utils/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

#define RL_MIN(X,Y) (((X)<(Y))?(X):(Y))
#define RL_MAX(X,Y) (((X)>(Y))?(X):(Y))
#ifndef RL_CLAMP
#define RL_CLAMP(X,LO,HI) (((X)<(LO))?(LO):(((X)>(HI))?(HI):(X)))
#endif
#define RL_SQ(X) ((X)*(X))


/* I-016修复：使用secure_random_float替代简单LCG伪随机 */
static float rl_randf(void)
{
    return secure_random_float();
}

static float rl_randn(float mu, float sigma)
{
    float u1 = rl_randf();
    float u2 = rl_randf();
    if (u1 < 1e-7f) u1 = 1e-7f;
    return mu + sigma * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

typedef struct {
    float* noise_state;
    int dim;
    int theta;
    float mu;
    float sigma;
    float dt;
} OUNoiseState;

typedef enum {
    RL_NETWORK_Q = 0,
    RL_NETWORK_Q_TARGET = 1,
    RL_NETWORK_ACTOR = 2,
    RL_NETWORK_CRITIC = 3,
    RL_NETWORK_VALUE = 4,
    RL_NETWORK_VALUE_TARGET = 5,
    RL_NETWORK_CRITIC2 = 6,
    RL_NETWORK_Q2 = 7,
    RL_NETWORK_Q_TARGET2 = 8,
    /* CfC-TD3 额外网络 */
    RL_NETWORK_ACTOR_TARGET = 9,
    RL_NETWORK_CRITIC1_TARGET = 10,
    /* CfC-Rainbow DQN 分布网络 */
    RL_NETWORK_DISTRIBUTION = 11,
    RL_NETWORK_DISTRIBUTION_TARGET = 12,
    RL_NETWORK_VALUE_STREAM = 13,
    RL_NETWORK_ADVANTAGE_STREAM = 14,
    /* CfC-IMPALA 额外网络 */
    RL_NETWORK_BASELINE = 15,
    /* CfC-R2D2 循环网络 */
    RL_NETWORK_RECURRENT_Q = 16,
    RL_NETWORK_RECURRENT_Q_TARGET = 17,
/* SAC目标Critic2网络 */
    RL_NETWORK_CRITIC2_TARGET = 18,
    RL_NETWORK_COUNT = 19
} RLNetworkType;

struct RLAgent {
    RLConfig config;
    int is_initialized;
    LNN* networks[RL_NETWORK_COUNT];
    int network_active[RL_NETWORK_COUNT];
    float epsilon;
    int total_steps;
    int total_episodes;
    float episode_return_sum;
    float episode_return_count;
    float best_return;
    float current_episode_return;
    OUNoiseState* ou_noise;
    /* ZSFJJJ-H005: 高级探索策略状态 */
    void* icm_state;                    /**< ICM好奇心模块状态 */
    void* rnd_state;                    /**< RND随机网络蒸馏状态 */
    void* go_explore_state;            /**< Go-Explore状态 */
    float* action_buffer;
    int action_buffer_size;
    RLReplayBuffer* external_buffer;
    float alpha;
    float* log_alpha;
    float target_entropy;
    float* last_action_probs;
    float* last_log_probs;
    float* last_value;
    int last_state_dim;

    /* CfC-TD3 状态 */
    int td3_critic_updates;            /**< 评论家更新计数 */
    float td3_avg_critic_loss;         /**< 平均评论家损失 */
    float td3_avg_actor_loss;          /**< 平均演员损失 */
    float td3_avg_q_value;             /**< 平均Q值 */

    /* CfC-Rainbow 分布RL状态 */
    float* rainbow_atom_support;       /**< 原子支撑向量 [num_atoms] */
    int rainbow_num_atoms;             /**< 原子数 */
    float rainbow_v_min;               /**< 值范围下限 */
    float rainbow_v_max;               /**< 值范围上限 */
    float* rainbow_noisy_params;       /**< Noisy Nets参数 */

    /* CfC-IMPALA 状态 */
    float* impala_trajectory_buffer;   /**< 轨迹缓冲区 */
    int impala_trajectory_length;      /**< 轨迹长度 */
    float* impala_vtrace_values;       /**< V-trace值缓冲区 */
    float impala_avg_entropy;          /**< 平均熵 */

    /* CfC-R2D2 循环状态 */
    float* r2d2_recurrent_state;       /**< CfC循环隐藏状态 */
    int r2d2_recurrent_state_dim;      /**< 循环状态维度 */
    float* r2d2_sequence_buffer;       /**< 序列缓冲区 */
    int* r2d2_sequence_dones;          /**< 序列终止标志 */
    int r2d2_buffer_pos;               /**< 缓冲区写入位置 */
};

static float rl_sigmoid(float x)
{
    if (x > 0.0f) return 1.0f / (1.0f + expf(-x));
    float ex = expf(x);
    return ex / (1.0f + ex);
}

static float rl_tanh(float x)
{
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return -1.0f;
    float ep = expf(2.0f * x);
    return (ep - 1.0f) / (ep + 1.0f);
}

static float rl_gaussian_log_prob(float x, float mean, float log_std)
{
    float std = expf(log_std);
    float var = std * std;
    float diff = x - mean;
    return -0.5f * (diff * diff / var + logf(2.0f * (float)M_PI * var));
}

static int rl_lnn_create_if(RLAgent* agent, RLNetworkType type, const LNNConfig* config)
{
    if (agent->network_active[type]) return 0;
    agent->networks[type] = lnn_create(config);
    if (!agent->networks[type]) return -1;
    agent->network_active[type] = 1;
    return 0;
}

/* ============================================================================
 * 4.2 修复：强化学习→物理引擎闭环训练桥接
 *
 * rl_physics_rollout — 将RL策略部署到物理引擎中，
 * 执行完整的状态→动作→物理步进→奖励的闭环rollout。
 * 支持PyBullet和内部simulator两种物理后端。
 * ============================================================================ */

#include "selflnn/robot/pybullet_bridge.h"
#include "selflnn/robot/simulator.h"
#include "selflnn/core/ode_solvers.h"

/* 内部物理模拟器前置声明 */
typedef struct InternalPhysicsEnv InternalPhysicsEnv;

/* 物理引擎状态定义 */
typedef struct {
    int connection_id;          /* PyBullet连接ID (0=内部sim) */
    int robot_id;               /* 机器人ID */
    int num_joints;             /* 关节数 */
    int state_dim;              /* 状态维度 (joint_pos*2 + joint_vel*2 + base(7)) */
    int action_dim;             /* 动作维度 (joint_targets) */
    float dt;                   /* 仿真步长 */
    int use_pybullet;           /* 使用PyBullet还是内部sim */
    /* L-019修复：Simulator* sim 字段已废除，使用 internal_env 替代
    Simulator* sim; */         /* 内部simulator实例（已废弃） */
    InternalPhysicsEnv* internal_env; /* 内部ODE物理模拟器（无PyBullet时使用） */
    unsigned int seed;          /* 随机种子 */
    int is_valid;               /**< M-009修复: 环境是否有效（有真实物理后端或内部模拟器） */
} RLPhysicsEnv;

/* ============================================================================
 * 内部物理模拟器：基于ODE求解器的简化刚体动力学
 * 当PyBullet不可用时作为回退物理后端。
 * 状态格式：二阶系统 [pos_0...pos_{n-1}, vel_0...vel_{n-1}]
 * 使用项目内置DP54/RK4求解器进行真实的刚性ODE步进。
 * ============================================================================ */
struct InternalPhysicsEnv {
    int state_dim;              /* 总状态维度（= half_dim * 2） */
    int half_dim;               /* 一半维度（位置或速度的维度数） */
    int action_dim;             /* 动作维度 */
    float dt;                   /* 仿真步长（秒） */
    int max_steps;              /* 最大回合步数 */
    int current_step;           /* 当前步数 */
    float* state;               /* 当前状态 [state_dim] */
    float* init_state;          /* 初始状态 [state_dim] */
    float* pending_action;      /* 当前步待执行动作 [action_dim] */
    float* workspace;           /* DP54/RK4 ODE求解器工作空间 */
    size_t workspace_size;      /* 工作空间大小（字节数） */
    DP54Config ode_config;      /* ODE求解器配置 */
    unsigned int seed;          /* 随机种子 */
    int is_active;              /* 是否已激活 */
};

/* 内部物理模拟器RHS：二阶动力学系统
 * d(pos)/dt = vel,  d(vel)/dt = action（控制加速度） - 阻尼*vel + 重力
 */
static int internal_physics_rhs(float t, const float* y, float* dydt, void* ctx) {
    InternalPhysicsEnv* ipenv = (InternalPhysicsEnv*)ctx;
    if (!ipenv || !ipenv->pending_action) return -1;
    (void)t;

    int half = ipenv->half_dim;

    /* 位置导数 = 速度 */
    for (int i = 0; i < half; i++) {
        dydt[i] = y[half + i];
    }

    /* 速度导数 = 动作加速度 - 阻尼系数 * 速度 + 重力 */
    for (int i = 0; i < half; i++) {
        float acc = (i < ipenv->action_dim) ? ipenv->pending_action[i] : 0.0f;
        float damping = 0.15f;
        float gravity_term = 0.0f;
        /* Z轴（索引2）施加重力 */
        if (i == 2 && half >= 3) {
            gravity_term = -9.81f;
        }
        dydt[half + i] = acc - damping * y[half + i] + gravity_term;
    }

    return 0;
}

/**
 * @brief 初始化内部物理模拟器
 * @param half_dim 位置/速度的维度数（总状态=half_dim*2）
 * @param action_dim 动作维度（控制输入维度）
 * @param dt 仿真步长
 * @param max_steps 最大步数
 * @return 模拟器指针，失败返回NULL
 */
static void ipenv_free(InternalPhysicsEnv* ipenv);

static InternalPhysicsEnv* internal_physics_env_init(int half_dim, int action_dim,
                                                      float dt, int max_steps) {
    if (half_dim <= 0 || action_dim <= 0 || half_dim > 128 || action_dim > 64)
        return NULL;

    InternalPhysicsEnv* ipenv = (InternalPhysicsEnv*)safe_calloc(1, sizeof(InternalPhysicsEnv));
    if (!ipenv) return NULL;

    ipenv->half_dim = half_dim;
    ipenv->state_dim = half_dim * 2;
    ipenv->action_dim = action_dim;
    ipenv->dt = dt > 0.0f ? dt : 0.004f;
    ipenv->max_steps = max_steps > 0 ? max_steps : 1000;
    ipenv->current_step = 0;
    ipenv->seed = (unsigned int)time(NULL);

    ipenv->state = (float*)safe_calloc((size_t)ipenv->state_dim, sizeof(float));
    ipenv->init_state = (float*)safe_calloc((size_t)ipenv->state_dim, sizeof(float));
    ipenv->pending_action = (float*)safe_calloc((size_t)ipenv->action_dim, sizeof(float));

    if (!ipenv->state || !ipenv->init_state || !ipenv->pending_action) {
        ipenv_free(ipenv);
        return NULL;
    }

    /* 配置DP54 ODE求解器 */
    ipenv->ode_config = ode_dp54_default_config();
    ipenv->ode_config.rel_tolerance = 1e-5f;
    ipenv->ode_config.abs_tolerance = 1e-7f;
    ipenv->ode_config.min_step_size = 1e-6f;
    ipenv->ode_config.max_step_size = ipenv->dt * 0.5f;

    ipenv->workspace_size = ode_dp54_workspace_size((size_t)ipenv->state_dim);
    ipenv->workspace = (float*)safe_malloc(ipenv->workspace_size);
    if (!ipenv->workspace) {
        ipenv_free(ipenv);
        return NULL;
    }

    /* 设置初始状态：位置随机在[-1,1]，速度随机在[-0.5,0.5] */
    for (int i = 0; i < half_dim; i++) {
        ipenv->init_state[i] = rl_randf() * 2.0f - 1.0f;
        ipenv->init_state[half_dim + i] = rl_randf() * 1.0f - 0.5f;
    }

    ipenv->is_active = 1;
    return ipenv;
}

/**
 * @brief 释放内部物理模拟器资源
 */
static void ipenv_free(InternalPhysicsEnv* ipenv) {
    if (!ipenv) return;
    safe_free((void**)&ipenv->state);
    safe_free((void**)&ipenv->init_state);
    safe_free((void**)&ipenv->pending_action);
    safe_free((void**)&ipenv->workspace);
    safe_free((void**)&ipenv);
}

/**
 * @brief 重置内部物理模拟器到初始状态
 */
static int internal_physics_env_reset(InternalPhysicsEnv* ipenv, float* state_out, int* state_dim_out) {
    if (!ipenv || !state_out || !state_dim_out) return -1;
    if (!ipenv->is_active) return -1;

    ipenv->current_step = 0;
    *state_dim_out = ipenv->state_dim;

    /* 复制初始状态（带微小随机扰动避免确定性陷阱） */
    for (int i = 0; i < ipenv->state_dim; i++) {
        ipenv->state[i] = ipenv->init_state[i] + rl_randf() * 0.01f - 0.005f;
        state_out[i] = ipenv->state[i];
    }

    return 0;
}

/**
 * @brief 内部物理模拟器步进：使用DP54求解器积分
 * @param ipenv 模拟器
 * @param action 动作向量（控制加速度）
 * @param action_dim 动作维度
 * @param next_state 输出下一状态
 * @param next_state_dim 输出状态维度
 * @param reward 输出奖励
 * @param done 输出终止标志
 * @return 0成功，-1失败
 */
static int internal_physics_env_step(InternalPhysicsEnv* ipenv,
                                      const float* action, int action_dim,
                                      float* next_state, int* next_state_dim,
                                      float* reward, int* done) {
    if (!ipenv || !action || !next_state || !next_state_dim || !reward || !done)
        return -1;
    if (!ipenv->is_active) return -1;

    *next_state_dim = ipenv->state_dim;
    *done = 0;

    /* 存储当前动作供RHS函数使用 */
    int act_count = (action_dim < ipenv->action_dim) ? action_dim : ipenv->action_dim;
    memset(ipenv->pending_action, 0, (size_t)ipenv->action_dim * sizeof(float));
    for (int i = 0; i < act_count; i++) {
        ipenv->pending_action[i] = action[i];
    }

    /* 使用DP54求解器进行ODE积分步进 */
    float h_actual = 0.0f;
    int steps_used = 0;
    int solve_ret = ode_dp54_solve(
        ipenv->state,
        0.0f,                    /* 从t=0开始 */
        ipenv->dt,
        internal_physics_rhs,
        (void*)ipenv,
        (size_t)ipenv->state_dim,
        &ipenv->ode_config,
        ipenv->workspace,
        &h_actual,
        &steps_used
    );

    /* 即使求解器返回非零（如步长限制），也输出当前状态 */
    for (int i = 0; i < ipenv->state_dim; i++) {
        next_state[i] = ipenv->state[i];
    }

    if (solve_ret != 0) {
        /* ODE求解异常，标记终止 */
        *done = 1;
        *reward = -1.0f;
        ipenv->current_step++;
        return 0;
    }

    /* 计算奖励：基于位置跟踪误差 + 能耗惩罚 + 存活奖励 */
    float tracking_reward = 0.0f;
    int half = ipenv->half_dim;
    for (int i = 0; i < half; i++) {
        /* 目标位置为0，越接近0奖励越高 */
        float pos_error = fabsf(ipenv->state[i]);
        tracking_reward -= pos_error * 0.5f;
    }

    float energy_penalty = 0.0f;
    for (int i = 0; i < act_count; i++) {
        energy_penalty += fabsf(action[i]) * 0.02f;
    }

    float alive_bonus = 1.0f;
    *reward = alive_bonus + tracking_reward - energy_penalty;

    /* 终止条件：超时或位置发散过大 */
    ipenv->current_step++;
    if (ipenv->current_step >= ipenv->max_steps) {
        *done = 1;
    }

    /* 检查状态是否发散 */
    for (int i = 0; i < ipenv->state_dim; i++) {
        if (fabsf(ipenv->state[i]) > 100.0f) {
            *done = 1;
            *reward -= 10.0f;
            break;
        }
    }

    return 0;
}

/**
 * @brief 创建强化学习物理环境
 *
 * @param connection_id PyBullet连接ID (0=使用内部simulator, >0=PyBullet)
 * @param robot_id 机器人ID (PyBullet模式下)
 * @param num_joints 关节数
 * @param dt 仿真步长 (秒)
 * @return RLPhysicsEnv* 环境句柄, NULL失败
 */
static RLPhysicsEnv* rl_physics_env_create(int connection_id, int robot_id,
                                            int num_joints, float dt) {
    RLPhysicsEnv* env = (RLPhysicsEnv*)safe_calloc(1, sizeof(RLPhysicsEnv));
    if (!env) return NULL;
    env->connection_id = connection_id;
    env->robot_id = robot_id;
    env->num_joints = num_joints;
    env->state_dim = num_joints * 2 + 7;
    env->action_dim = num_joints;
    env->dt = dt > 0 ? dt : 0.004f;
    env->use_pybullet = (connection_id > 0) ? 1 : 0;
    env->seed = (unsigned int)time(NULL);
    env->internal_env = NULL;
    /* 当PyBullet可用时使用PyBullet，否则初始化内部ODE物理模拟器作为回退 */
    if (env->use_pybullet) {
        env->is_valid = 1;
    } else {
        /* 使用内部ODE模拟器：half_dim=num_joints（位置+速度各num_joints维） */
        env->internal_env = internal_physics_env_init(num_joints, num_joints, env->dt, 1000);
        env->is_valid = (env->internal_env != NULL) ? 1 : 0;
        if (!env->is_valid) {
            fprintf(stderr, "[RL-物理环境] 内部物理模拟器初始化失败\n");
        } else {
            printf("[RL-物理环境] 使用内部ODE物理模拟器（DP54），状态维度=%d，动作维度=%d\n",
                   env->internal_env->state_dim, env->internal_env->action_dim);
        }
    }
    return env;
}

static void rl_physics_env_free(RLPhysicsEnv* env) {
    if (!env) return;
    if (env->internal_env) {
        ipenv_free(env->internal_env);
        env->internal_env = NULL;
    }
    safe_free((void**)&env);
}

/**
 * @brief 重置物理环境到初始状态
 *
 * M-009修复: 添加 is_valid 检查。当无真实物理后端时返回明确错误状态
 * 而非静默填充零状态。
 *
 * @param env 环境
 * @param state 输出初始状态 [state_dim]
 * @param state_dim 输出实际状态维度
 * @return 0成功, -1失败
 */
static int rl_physics_env_reset(RLPhysicsEnv* env, float* state, int* state_dim) {
    if (!env || !state || !state_dim) return -1;

    if (!env->is_valid) {
        *state_dim = 0;
        memset(state, 0, (size_t)env->state_dim * sizeof(float));
        fprintf(stderr, "[RL-物理环境] 环境重置失败: 无物理后端\n");
        return -1;
    }

    if (env->use_pybullet) {
        *state_dim = env->state_dim;
        memset(state, 0, (size_t)(*state_dim) * sizeof(float));
        /* PyBullet: 获取初始关节位置 */
        for (int j = 0; j < env->num_joints; j++) {
            PyBulletJointState js;
            memset(&js, 0, sizeof(js));
            if (pybullet_get_joint_state(env->connection_id, env->robot_id, j, &js) == 0) {
                state[j] = js.position;
                state[env->num_joints + j] = js.velocity;
            }
        }
        PyBulletBaseState bs;
        memset(&bs, 0, sizeof(bs));
        if (pybullet_get_base_state(env->connection_id, env->robot_id, &bs) == 0) {
            memcpy(state + env->num_joints * 2, bs.position, 3 * sizeof(float));
            memcpy(state + env->num_joints * 2 + 3, bs.orientation, 4 * sizeof(float));
        }
        pybullet_step_simulation(env->connection_id);
        return 0;
    } else {
        /* 使用内部ODE物理模拟器重置 */
        return internal_physics_env_reset(env->internal_env, state, state_dim);
    }
}

/**
 * @brief 执行一步物理环境交互
 *
 * @param env 环境
 * @param action 动作向量 (关节目标位置) [action_dim]
 * @param action_dim 动作维度
 * @param next_state 输出下一状态 [state_dim]
 * @param next_state_dim 输出状态维度
 * @param reward 输出奖励
 * @param done 输出终止标志
 * @return 0成功, -1失败
 */
static int rl_physics_env_step(RLPhysicsEnv* env,
                                const float* action, int action_dim,
                                float* next_state, int* next_state_dim,
                                float* reward, int* done) {
    if (!env || !action || !next_state || !next_state_dim || !reward || !done)
        return -1;

    *next_state_dim = env->state_dim;
    *done = 0;

    if (env->use_pybullet) {
        int* joints = (int*)safe_malloc((size_t)env->num_joints * sizeof(int));
        float* targets = (float*)safe_malloc((size_t)env->num_joints * sizeof(float));
        if (joints && targets) {
            int n = (action_dim < env->num_joints) ? action_dim : env->num_joints;
            for (int j = 0; j < n; j++) {
                joints[j] = j;
                targets[j] = (j < action_dim) ? action[j] : 0.0f;
            }
            pybullet_set_joint_control(env->connection_id, env->robot_id, joints, targets, n);
            safe_free((void**)&joints);
            safe_free((void**)&targets);
        }
        for (int s = 0; s < 10; s++) pybullet_step_simulation(env->connection_id);

        /* 读取新状态 */
        memset(next_state, 0, (size_t)(*next_state_dim) * sizeof(float));
        for (int j = 0; j < env->num_joints; j++) {
            PyBulletJointState js;
            memset(&js, 0, sizeof(js));
            pybullet_get_joint_state(env->connection_id, env->robot_id, j, &js);
            next_state[j] = js.position;
            next_state[env->num_joints + j] = js.velocity;
        }

        /* 计算奖励：移动奖励 + 能耗惩罚 + 存活奖励 */
        float move_reward = 0.0f;
        for (int j = 0; j < env->num_joints && j < action_dim; j++) {
            float delta = fabsf(next_state[j] - action[j]);
            move_reward -= delta * 0.1f;
        }
        float energy_penalty = 0.0f;
        for (int j = 0; j < env->num_joints; j++) {
            float vel = next_state[env->num_joints + j];
            energy_penalty += fabsf(vel) * 0.01f;
        }
        *reward = 1.0f + move_reward - energy_penalty;
        return 0;
    } else {
        /* 使用内部ODE物理模拟器步进（自带奖励计算） */
        return internal_physics_env_step(env->internal_env,
                                          action, action_dim,
                                          next_state, next_state_dim,
                                          reward, done);
    }
}

/**
 * @brief RL策略在物理引擎中执行完整rollout
 *
 * 这是RL与物理引擎的完整闭环：
 *   observe → agent.act → physics.step → observe → reward → train
 *
 * 封装了一整条轨迹的采样过程，可直接用于RL训练循环。
 *
 * @param agent RL智能体
 * @param connection_id PyBullet连接ID (0=内部sim)
 * @param robot_id 机器人ID
 * @param num_joints 关节数
 * @param dt 仿真步长
 * @param max_steps 最大步数
 * @param total_reward 输出累积奖励
 * @return 实际步数, -1失败
 */
static int rl_physics_rollout(RLAgent* agent, int connection_id, int robot_id,
                        int num_joints, float dt, int max_steps, float* total_reward) {
    if (!agent || !agent->is_initialized || !total_reward) return -1;
    if (num_joints <= 0) return -1;

    *total_reward = 0.0f;
    RLPhysicsEnv* env = rl_physics_env_create(connection_id, robot_id, num_joints, dt);
    if (!env) return -1;

    float obs[256];
    int obs_dim = 0;
    if (rl_physics_env_reset(env, obs, &obs_dim) != 0) {
        rl_physics_env_free(env);
        return -1;
    }

    int steps = 0;
    for (int t = 0; t < max_steps; t++) {
        float action[64];
        int act_dim = env->action_dim < 64 ? env->action_dim : 64;
        memset(action, 0, sizeof(action));

        /* 使用RL智能体选择动作（训练中带探索噪声） */
        if (steps % 3 == 0) {
            rl_select_action(agent, obs, obs_dim, action, act_dim);
        } else {
            for (int i = 0; i < act_dim; i++)
                action[i] = obs[i] * 0.5f + rl_randf() * 0.2f - 0.1f;
        }

        float next_obs[256];
        int next_obs_dim = 0;
        float reward = 0.0f;
        int done = 0;

        if (rl_physics_env_step(env, action, act_dim,
                                next_obs, &next_obs_dim, &reward, &done) != 0)
            break;

        /* 存储经验到回放缓冲区 */
        RLExperience exp;
        memset(&exp, 0, sizeof(exp));
        memcpy(exp.state, obs, (size_t)obs_dim * sizeof(float));
        exp.state_dim = obs_dim;
        memcpy(exp.action, action, (size_t)act_dim * sizeof(float));
        exp.action_dim = act_dim;
        exp.reward = reward;
        memcpy(exp.next_state, next_obs, (size_t)next_obs_dim * sizeof(float));
        exp.next_state_dim = next_obs_dim;
        exp.done = done;
        rl_replay_buffer_add(&agent->config.replay_buffer, &exp);

        *total_reward += reward;
        steps = t + 1;

        if (done) break;

        memcpy(obs, next_obs, (size_t)next_obs_dim * sizeof(float));
        obs_dim = next_obs_dim;

        /* 每16步执行一次训练 */
        if (steps % 16 == 0) {
            rl_train(agent, 32);
        }
    }

    rl_physics_env_free(env);
    return steps;
}

/* ============ 探索策略 ============ */

static float rl_explore_epsilon_greedy(RLExploreConfig* cfg, int total_steps)
{
    float rate = cfg->epsilon_start * expf(-(float)total_steps * cfg->epsilon_decay);
    return RL_MAX(cfg->epsilon_end, rate);
}

static float rl_explore_ucb(float q_value, int visit_count, int total_visits, float constant)
{
    if (visit_count <= 0) return 1e9f;
    return q_value + constant * sqrtf(logf((float)total_visits + 1.0f) / (float)visit_count);
}

static float rl_explore_boltzmann(const float* q_values, int num_actions, float temperature, int* selected)
{
    float max_q = q_values[0];
    for (int i = 1; i < num_actions; i++)
        if (q_values[i] > max_q) max_q = q_values[i];

    float sum = 0.0f;
    float probs[RL_MAX_ACTION_DIM];
    for (int i = 0; i < num_actions && i < RL_MAX_ACTION_DIM; i++)
    {
        probs[i] = expf((q_values[i] - max_q) / RL_MAX(temperature, 0.001f));
        sum += probs[i];
    }
    if (sum < 1e-7f) sum = 1e-7f;
    float r = rl_randf();
    float cum = 0.0f;
    for (int i = 0; i < num_actions; i++)
    {
        cum += probs[i] / sum;
        if (r <= cum) { *selected = i; return probs[i] / sum; }
    }
    *selected = num_actions - 1;
    return probs[num_actions - 1] / sum;
}

static OUNoiseState* rl_ou_noise_create(int dim, int theta, float mu, float sigma, float dt)
{
    OUNoiseState* noise = (OUNoiseState*)safe_malloc(sizeof(OUNoiseState));
    if (!noise) return NULL;
    noise->noise_state = (float*)safe_calloc(dim, sizeof(float));
    if (!noise->noise_state) { safe_free((void**)&noise); return NULL; }
    noise->dim = dim;
    noise->theta = theta;
    noise->mu = mu;
    noise->sigma = sigma;
    noise->dt = dt;
    return noise;
}

static void rl_ou_noise_free(OUNoiseState* noise)
{
    if (!noise) return;
    safe_free((void**)&noise->noise_state);
    safe_free((void**)&noise);
}

static void rl_ou_noise_reset(OUNoiseState* noise)
{
    if (!noise || !noise->noise_state) return;
    memset(noise->noise_state, 0, noise->dim * sizeof(float));
}

static void rl_ou_noise_sample(OUNoiseState* noise, float* out, int dim)
{
    if (!noise || !out) return;
    int d = RL_MIN(dim, noise->dim);
    for (int i = 0; i < d; i++)
    {
        float dx = (float)noise->theta * (noise->mu - noise->noise_state[i]) * noise->dt
                   + noise->sigma * sqrtf(noise->dt) * rl_randn(0.0f, 1.0f);
        noise->noise_state[i] += dx;
        out[i] = noise->noise_state[i];
    }
}

static float rl_epsilon_greedy_select(RLAgent* agent, const float* q_values, int num_actions)
{
    if (rl_randf() < agent->epsilon)
        return (float)(int)(rl_randf() * num_actions);
    int best = 0;
    for (int i = 1; i < num_actions; i++)
        if (q_values[i] > q_values[best]) best = i;
    return (float)best;
}

/* ============ 经验回放缓冲区 ============ */

RLReplayBuffer rl_replay_buffer_create(int capacity, int state_dim, int action_dim, RLReplayType replay_type)
{
    RLReplayBuffer buf;
    memset(&buf, 0, sizeof(RLReplayBuffer));
    if (capacity <= 0 || state_dim <= 0) return buf;

    buf.buffer = (RLExperience*)safe_calloc(capacity, sizeof(RLExperience));
    if (!buf.buffer) return buf;
    buf.capacity = capacity;
    buf.state_dim = state_dim;
    buf.action_dim = action_dim;
    buf.replay_type = replay_type;
    buf.alpha = 0.6f;
    buf.beta = 0.4f;
    buf.beta_increment = 0.001f;
    buf.max_priority = 1.0f;

    if (replay_type == RL_REPLAY_PRIORITIZED)
    {
        buf.priorities = (float*)safe_calloc(capacity, sizeof(float));
        if (!buf.priorities)
        {
            safe_free((void**)&buf.buffer);
            memset(&buf, 0, sizeof(RLReplayBuffer));
            return buf;
        }
    }
    return buf;
}

void rl_replay_buffer_destroy(RLReplayBuffer* buffer)
{
    if (!buffer) return;
    safe_free((void**)&buffer->buffer);
    safe_free((void**)&buffer->priorities);
    memset(buffer, 0, sizeof(RLReplayBuffer));
}

int rl_replay_buffer_add(RLReplayBuffer* buffer, const RLExperience* experience)
{
    if (!buffer || !experience) return -1;
    int idx = buffer->head;
    buffer->buffer[idx] = *experience;
    if (buffer->replay_type == RL_REPLAY_PRIORITIZED && buffer->priorities)
        buffer->priorities[idx] = buffer->max_priority;
    buffer->head = (buffer->head + 1) % buffer->capacity;
    if (buffer->size < buffer->capacity) buffer->size++;
    return 0;
}

int rl_replay_buffer_sample(const RLReplayBuffer* buffer, int batch_size,
                            RLExperience* samples, int* indices, float* weights)
{
    if (!buffer || !samples || buffer->size <= 0 || batch_size <= 0) return -1;
    int actual = RL_MIN(batch_size, buffer->size);
    actual = RL_MIN(actual, RL_MAX_BATCH_SIZE);

    if (buffer->replay_type == RL_REPLAY_PRIORITIZED && buffer->priorities)
    {
        float sum = 0.0f;
        for (int i = 0; i < buffer->size; i++)
            sum += powf(buffer->priorities[i], buffer->alpha);
        if (sum < 1e-7f) sum = 1e-7f;
        float inv_sum = 1.0f / sum;

        float beta = buffer->beta;
        float max_weight = -1e9f;

        for (int i = 0; i < actual; i++)
        {
            float r = rl_randf();
            float cum = 0.0f;
            int sel = 0;
            for (int j = 0; j < buffer->size; j++)
            {
                cum += powf(buffer->priorities[j], buffer->alpha) * inv_sum;
                if (r <= cum) { sel = j; break; }
            }
            samples[i] = buffer->buffer[sel];
            if (indices) indices[i] = sel;
            if (weights)
            {
                float w = powf(1.0f / ((float)buffer->size * powf(buffer->priorities[sel], buffer->alpha) * inv_sum + 1e-7f), beta);
                weights[i] = w;
                if (w > max_weight) max_weight = w;
            }
        }
        if (weights && max_weight > 0.0f)
        {
            float inv = 1.0f / max_weight;
            for (int i = 0; i < actual; i++) weights[i] *= inv;
        }
    }
    else
    {
        for (int i = 0; i < actual; i++)
        {
            int idx = (int)(rl_randf() * buffer->size);
            if (idx >= buffer->size) idx = buffer->size - 1;
            samples[i] = buffer->buffer[idx];
            if (indices) indices[i] = idx;
            if (weights) weights[i] = 1.0f;
        }
    }
    return actual;
}

void rl_replay_buffer_update_priorities(RLReplayBuffer* buffer, const int* indices,
                                        const float* priorities, int count)
{
    if (!buffer || !indices || !priorities || !buffer->priorities) return;
    for (int i = 0; i < count; i++)
    {
        if (indices[i] >= 0 && indices[i] < buffer->capacity)
        {
            buffer->priorities[indices[i]] = RL_MAX(priorities[i] + 1e-6f, buffer->priorities[indices[i]]);
            if (buffer->priorities[indices[i]] > buffer->max_priority)
                buffer->max_priority = buffer->priorities[indices[i]];
        }
    }
    buffer->beta = RL_MIN(1.0f, buffer->beta + buffer->beta_increment);
}

/* ============ 探索配置 ============ */

RLExploreConfig rl_explore_config_default(RLExploreStrategy strategy)
{
    RLExploreConfig cfg;
    memset(&cfg, 0, sizeof(RLExploreConfig));
    cfg.strategy = strategy;
    cfg.epsilon_start = 1.0f;
    cfg.epsilon_end = 0.01f;
    cfg.epsilon_decay = 0.001f;
    cfg.noise_std = 0.1f;
    cfg.noise_decay = 0.999f;
    cfg.temperature = 1.0f;
    cfg.ucb_constant = 1.0f;
    cfg.ou_theta = 1;
    cfg.ou_mu = 0.0f;
    cfg.ou_sigma = 0.2f;
    cfg.ou_dt = 0.01f;
    return cfg;
}

/* ============ DQN 实现 ============ */

static int rl_dqn_init(RLAgent* agent)
{
    RLConfigDQN* dc = &agent->config.algo_config.dqn;
    if (dc->hidden_size <= 0) dc->hidden_size = 64;
    if (dc->num_layers <= 0) dc->num_layers = 2;
    if (dc->learning_rate <= 0.0f) dc->learning_rate = 0.001f;
    if (dc->gamma <= 0.0f) dc->gamma = 0.99f;
    if (dc->tau <= 0.0f) dc->tau = 0.005f;
    if (dc->target_update_freq <= 0) dc->target_update_freq = 100;

    int output_size = agent->config.discrete_actions ?
                      agent->config.num_actions : agent->config.action_dim;

    dc->lnn_config.input_size = (size_t)agent->config.state_dim;
    dc->lnn_config.hidden_size = (size_t)dc->hidden_size;
    dc->lnn_config.output_size = (size_t)output_size;
    dc->lnn_config.learning_rate = dc->learning_rate;
    dc->lnn_config.num_layers = dc->num_layers;
    dc->lnn_config.enable_training = 1;

    int ret = rl_lnn_create_if(agent, RL_NETWORK_Q, &dc->lnn_config);
    if (ret != 0) return -1;

    dc->lnn_config.learning_rate = dc->learning_rate * 0.5f;
    ret = rl_lnn_create_if(agent, RL_NETWORK_Q_TARGET, &dc->lnn_config);
    if (ret != 0) return -1;

    return 0;
}

static int rl_dqn_select_action(RLAgent* agent, const float* state, int state_dim,
                                float* action, int action_dim)
{
    (void)state_dim;
    int num_actions = agent->config.discrete_actions ?
                      agent->config.num_actions : action_dim;

    float* q_values = (float*)safe_malloc(num_actions * sizeof(float));
    if (!q_values) return -1;

    LNN* q_net = agent->networks[RL_NETWORK_Q];
    if (!q_net) { safe_free((void**)&q_values); return -1; }

    int ret = lnn_forward(q_net, state, q_values);
    if (ret != 0) { safe_free((void**)&q_values); return -1; }

    if (agent->config.discrete_actions)
    {
        action[0] = rl_epsilon_greedy_select(agent, q_values, num_actions);
    }
    else
    {
        int best = 0;
        for (int i = 1; i < num_actions; i++)
            if (q_values[i] > q_values[best]) best = i;
        action[best] = 1.0f;

        if (rl_randf() < agent->epsilon)
        {
            memset(action, 0, action_dim * sizeof(float));
            action[(int)(rl_randf() * num_actions)] = 1.0f;
        }
    }
    safe_free((void**)&q_values);
    return 0;
}

static int rl_dqn_train(RLAgent* agent, int batch_size)
{
    RLConfigDQN* dc = &agent->config.algo_config.dqn;
    RLReplayBuffer* buf = &agent->config.replay_buffer;
    if (buf->size < batch_size) return 0;

    int alloc_batch = (batch_size < RL_MAX_BATCH_SIZE) ? batch_size : RL_MAX_BATCH_SIZE;
    RLExperience* samples = (RLExperience*)safe_calloc((size_t)alloc_batch, sizeof(RLExperience));
    int* indices = (int*)safe_calloc((size_t)alloc_batch, sizeof(int));
    float* weights = (float*)safe_calloc((size_t)alloc_batch, sizeof(float));
    if (!samples || !indices || !weights) {
        safe_free((void**)&samples);
        safe_free((void**)&indices);
        safe_free((void**)&weights);
        return -1;
    }

    int actual = rl_replay_buffer_sample(buf, batch_size, samples, indices, weights);
    if (actual <= 0) {
        safe_free((void**)&samples);
        safe_free((void**)&indices);
        safe_free((void**)&weights);
        return 0;
    }

    LNN* q_net = agent->networks[RL_NETWORK_Q];
    LNN* target_net = agent->networks[RL_NETWORK_Q_TARGET];
    if (!q_net || !target_net) {
        safe_free((void**)&samples);
        safe_free((void**)&indices);
        safe_free((void**)&weights);
        return -1;
    }

    float* td_errors = (float*)safe_calloc((size_t)alloc_batch, sizeof(float));
    if (!td_errors) {
        safe_free((void**)&samples);
        safe_free((void**)&indices);
        safe_free((void**)&weights);
        return -1;
    }
    float total_loss = 0.0f;

    for (int i = 0; i < actual; i++)
    {
        float current_q[RL_MAX_ACTION_DIM];
        memset(current_q, 0, sizeof(current_q));
        int ret = lnn_forward(q_net, samples[i].state, current_q);
        if (ret != 0) continue;

        int act_idx = 0;
        if (agent->config.discrete_actions)
            act_idx = (int)samples[i].action[0];
        else
        {
            float max_v = samples[i].action[0];
            for (int a = 1; a < samples[i].action_dim; a++)
                if (samples[i].action[a] > max_v) { max_v = samples[i].action[a]; act_idx = a; }
        }
        if (act_idx < 0) act_idx = 0;
        if (act_idx >= RL_MAX_ACTION_DIM) act_idx = RL_MAX_ACTION_DIM - 1;

        float target_q;
        if (samples[i].done)
        {
            target_q = samples[i].reward;
        }
        else
        {
            float next_q[RL_MAX_ACTION_DIM];
            memset(next_q, 0, sizeof(next_q));

            if (dc->use_double_dqn)
            {
                float next_q_online[RL_MAX_ACTION_DIM];
                memset(next_q_online, 0, sizeof(next_q_online));
                lnn_forward(q_net, samples[i].next_state, next_q_online);
                int best_a = 0;
                for (int a = 1; a < (agent->config.discrete_actions ? agent->config.num_actions : samples[i].action_dim); a++)
                    if (next_q_online[a] > next_q_online[best_a]) best_a = a;
                lnn_forward(target_net, samples[i].next_state, next_q);
                target_q = samples[i].reward + dc->gamma * next_q[best_a];
            }
            else
            {
                lnn_forward(target_net, samples[i].next_state, next_q);
                float max_q = next_q[0];
                int nq_dim = agent->config.discrete_actions ? agent->config.num_actions : samples[i].action_dim;
                for (int a = 1; a < nq_dim; a++)
                    if (next_q[a] > max_q) max_q = next_q[a];
                target_q = samples[i].reward + dc->gamma * max_q;
            }
        }

        td_errors[i] = target_q - current_q[act_idx];
        float loss = td_errors[i] * td_errors[i] * weights[i];
        total_loss += loss;

        float* q_grad = (float*)safe_calloc(RL_MAX_ACTION_DIM, sizeof(float));
        if (q_grad)
        {
            q_grad[act_idx] = -2.0f * td_errors[i] * weights[i];
            lnn_backward(q_net, q_grad, &loss);
            safe_free((void**)&q_grad);
        }
    }

    float avg_td = 0.0f;
    for (int i = 0; i < actual; i++) avg_td += fabsf(td_errors[i]);
    avg_td /= (float)actual + 1e-7f;

    if (buf->replay_type == RL_REPLAY_PRIORITIZED)
        rl_replay_buffer_update_priorities(buf, indices, (const float*)td_errors, actual);

    agent->total_steps++;

    if (agent->total_steps % dc->target_update_freq == 0)
    {
        float tau = dc->tau;
        float* q_params = lnn_get_parameters(q_net);
        float* t_params = lnn_get_parameters(target_net);
        size_t nparams = lnn_get_parameter_count(q_net);
        if (q_params && t_params && nparams > 0)
        {
            for (size_t p = 0; p < nparams; p++)
                t_params[p] = (1.0f - tau) * t_params[p] + tau * q_params[p];
        }
    }
    safe_free((void**)&td_errors);
    safe_free((void**)&samples);
    safe_free((void**)&indices);
    safe_free((void**)&weights);
    return 0;
}

static int rl_ppo_init(RLAgent* agent)
{
    RLConfigPPO* pc = &agent->config.algo_config.ppo;
    if (pc->actor_hidden_size <= 0) pc->actor_hidden_size = 64;
    if (pc->critic_hidden_size <= 0) pc->critic_hidden_size = 64;
    if (pc->actor_lr <= 0.0f) pc->actor_lr = 0.0003f;
    if (pc->critic_lr <= 0.0f) pc->critic_lr = 0.001f;
    if (pc->gamma <= 0.0f) pc->gamma = 0.99f;
    if (pc->gae_lambda <= 0.0f) pc->gae_lambda = 0.95f;
    if (pc->clip_epsilon <= 0.0f) pc->clip_epsilon = 0.2f;
    if (pc->entropy_coef <= 0.0f) pc->entropy_coef = 0.01f;
    if (pc->value_coef <= 0.0f) pc->value_coef = 0.5f;
    if (pc->update_epochs <= 0) pc->update_epochs = 10;
    if (pc->mini_batch_size <= 0) pc->mini_batch_size = 32;

    int act_out = agent->config.discrete_actions ?
                  agent->config.num_actions : agent->config.action_dim * 2;

    pc->actor_lnn_config.input_size = (size_t)agent->config.state_dim;
    pc->actor_lnn_config.hidden_size = (size_t)pc->actor_hidden_size;
    pc->actor_lnn_config.output_size = (size_t)act_out;
    pc->actor_lnn_config.learning_rate = pc->actor_lr;
    pc->actor_lnn_config.num_layers = pc->actor_num_layers > 0 ? pc->actor_num_layers : 2;
    pc->actor_lnn_config.enable_training = 1;
    int ret = rl_lnn_create_if(agent, RL_NETWORK_ACTOR, &pc->actor_lnn_config);
    if (ret != 0) return -1;

    pc->critic_lnn_config.input_size = (size_t)agent->config.state_dim;
    pc->critic_lnn_config.hidden_size = (size_t)pc->critic_hidden_size;
    pc->critic_lnn_config.output_size = 1;
    pc->critic_lnn_config.learning_rate = pc->critic_lr;
    pc->critic_lnn_config.num_layers = pc->critic_num_layers > 0 ? pc->critic_num_layers : 2;
    pc->critic_lnn_config.enable_training = 1;
    ret = rl_lnn_create_if(agent, RL_NETWORK_CRITIC, &pc->critic_lnn_config);
    if (ret != 0) return -1;
    return 0;
}

static float rl_ppo_get_action_probs(RLAgent* agent, const float* state, int state_dim,
                                     float* probs_out, int* selected_action)
{
    (void)state_dim;
    LNN* actor = agent->networks[RL_NETWORK_ACTOR];
    if (!actor) return 0.0f;

/* 直接使用lnn_get_config获取output_size
     * 原代码先调用lnn_get_parameter_count（返回全部参数数量，可能数千），
     * 后被output_size覆盖，但parameter_count可能极大导致不必要的大内存分配风险 */
    LNNConfig acfg;
    memset(&acfg, 0, sizeof(LNNConfig));
    lnn_get_config(actor, &acfg);
    int output_size = (int)acfg.output_size;

    float* raw = (float*)safe_malloc((size_t)output_size * sizeof(float));
    if (!raw) return 0.0f;

    int ret = lnn_forward(actor, state, raw);
    if (ret != 0) { safe_free((void**)&raw); return 0.0f; }

    if (agent->config.discrete_actions)
    {
        float max_v = raw[0];
        for (int i = 1; i < output_size; i++)
            if (raw[i] > max_v) max_v = raw[i];
        float sum = 0.0f;
        for (int i = 0; i < output_size; i++)
        {
            probs_out[i] = expf(raw[i] - max_v);
            sum += probs_out[i];
        }
        if (sum < 1e-7f) sum = 1e-7f;
        for (int i = 0; i < output_size; i++) probs_out[i] /= sum;
        float r = rl_randf();
        float cum = 0.0f;
        for (int i = 0; i < output_size; i++)
        {
            cum += probs_out[i];
            if (r <= cum) { *selected_action = i; return logf(probs_out[i] + 1e-7f); }
        }
        *selected_action = output_size - 1;
        return logf(probs_out[output_size - 1] + 1e-7f);
    }
    else
    {
        int dim = output_size / 2;
        float total_log_prob = 0.0f;
        for (int i = 0; i < dim; i++)
        {
            float mean = raw[i];
            float log_std = RL_CLAMP(raw[i + dim], -5.0f, 2.0f);
            float std = expf(log_std);
            /* Box-Muller变换生成标准正态噪声 */
            float u1 = rl_randf() + 1e-7f;
            float u2 = rl_randf();
            float noise = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
            float action = mean + noise * std;
            probs_out[i] = RL_CLAMP(action, agent->config.action_low[i], agent->config.action_high[i]);
            /* 高斯对数概率: -0.5*(log(2πσ²) + (a-μ)²/σ²) */
            total_log_prob += -0.5f * (1.8378770664f + 2.0f * log_std
                                    + (action - mean) * (action - mean) / (std * std + 1e-7f));
        }
        *selected_action = 0; /* 连续动作无离散索引 */
        safe_free((void**)&raw);
        return total_log_prob;
    }
}

static int rl_ppo_train(RLAgent* agent, int batch_size)
{
    RLConfigPPO* pc = &agent->config.algo_config.ppo;
    RLReplayBuffer* buf = &agent->config.replay_buffer;
    if (buf->size < batch_size) return 0;

    int alloc_batch = (batch_size < RL_MAX_BATCH_SIZE) ? batch_size : RL_MAX_BATCH_SIZE;
    RLExperience* samples = (RLExperience*)safe_calloc((size_t)alloc_batch, sizeof(RLExperience));
    if (!samples) return -1;
    int actual = rl_replay_buffer_sample(buf, batch_size, samples, NULL, NULL);
    if (actual <= 0) { safe_free((void**)&samples); return 0; }

    LNN* actor = agent->networks[RL_NETWORK_ACTOR];
    LNN* critic = agent->networks[RL_NETWORK_CRITIC];
    if (!actor || !critic) { safe_free((void**)&samples); return -1; }

    float* old_log_probs = (float*)safe_malloc((size_t)actual * sizeof(float));
    float* advantages = (float*)safe_malloc((size_t)actual * sizeof(float));
    float* returns = (float*)safe_malloc((size_t)actual * sizeof(float));
    float* values = (float*)safe_malloc((size_t)actual * sizeof(float));
    if (!old_log_probs || !advantages || !returns || !values)
    {
        safe_free((void**)&old_log_probs);
        safe_free((void**)&advantages);
        safe_free((void**)&returns);
        safe_free((void**)&values);
        return -1;
    }

    /*
     * S-007修复: 完整GAE (Generalized Advantage Estimation) 实现
     * Schulman et al. 2016
     *
     * 步骤0：计算所有样本的旧策略对数概率（用于PPO重要性采样比例）
     * 步骤1：前向计算所有状态值V(s_t)和V(s_{t+1})
     * 步骤2：计算TD残差 δ_t = r_t + γ·V(s_{t+1})·(1-done_t) - V(s_t)
     * 步骤3：反向递推 GAE_t = δ_t + γ·λ·(1-done_t)·GAE_{t+1}
     * 步骤4：标准化优势函数（均值0，标准差1）
     */

    /* 步骤0: 计算所有样本的旧策略对数概率（PPO重要性采样所需） */
    for (int i = 0; i < actual; i++) {
        float probs[RL_MAX_ACTION_DIM];
        int sel_a = 0;
        old_log_probs[i] = rl_ppo_get_action_probs(agent, samples[i].state,
            samples[i].state_dim, probs, &sel_a);
    }

    /* 步骤1: 前向计算所有状态值V(s_t) */
    for (int i = 0; i < actual; i++) {
        float v = 0.0f;
        lnn_forward(critic, samples[i].state, &v);
        values[i] = v;
    }

    /* 步骤2-3: 反向递推GAE */
    float last_gae = 0.0f;
    for (int i = actual - 1; i >= 0; i--) {
        /* TD残差: δ_t = r_t + γ·V(s_{t+1})·(1-done_t) - V(s_t) */
        float delta = samples[i].reward - values[i];
        if (!samples[i].done) {
            float next_v = 0.0f;
            lnn_forward(critic, samples[i].next_state, &next_v);
            delta += pc->gamma * next_v;
        }
        /* GAE递推: A_t^{GAE(γ,λ)} = δ_t + γ·λ·(1-done_t)·A_{t+1}^{GAE(γ,λ)} */
        float done_mask = samples[i].done ? 0.0f : 1.0f;
        last_gae = delta + pc->gamma * pc->gae_lambda * done_mask * last_gae;
        advantages[i] = last_gae;
        returns[i] = advantages[i] + values[i];
    }

    /* 步骤4: 优势函数标准化（零均值、单位方差，提升训练稳定性） */
    float adv_mean = 0.0f, adv_std = 0.0f;
    for (int i = 0; i < actual; i++) adv_mean += advantages[i];
    adv_mean /= (float)actual;
    for (int i = 0; i < actual; i++) adv_std += RL_SQ(advantages[i] - adv_mean);
    adv_std = sqrtf(adv_std / (float)actual + 1e-7f);
    if (adv_std < 1e-7f) adv_std = 1.0f;
    for (int i = 0; i < actual; i++) advantages[i] = (advantages[i] - adv_mean) / adv_std;

    for (int e = 0; e < pc->update_epochs; e++)
    {
        int perm[RL_MAX_BATCH_SIZE];
        for (int i = 0; i < actual; i++) perm[i] = i;
        for (int i = actual - 1; i > 0; i--)
        {
            int j = (int)(rl_randf() * (float)(i + 1));
            int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
        }

        for (int start = 0; start < actual; start += pc->mini_batch_size)
        {
            int end = RL_MIN(start + pc->mini_batch_size, actual);
            for (int m = start; m < end; m++)
            {
                int idx = perm[m];

                float new_v = 0.0f;
                lnn_forward(critic, samples[idx].state, &new_v);
                float v_loss = RL_SQ(returns[idx] - new_v);
                float v_target[1] = {returns[idx]};
                lnn_backward(critic, v_target, &v_loss);

                float probs[RL_MAX_ACTION_DIM];
                int sel_a = 0;
                float new_log_prob = rl_ppo_get_action_probs(agent, samples[idx].state,
                    samples[idx].state_dim, probs, &sel_a);

                float ratio = expf(RL_CLAMP(new_log_prob - old_log_probs[idx], -10.0f, 10.0f));
                float surr1 = ratio * advantages[idx];
                float surr2 = RL_CLAMP(ratio, 1.0f - pc->clip_epsilon, 1.0f + pc->clip_epsilon) * advantages[idx];
                float p_loss = -RL_MIN(surr1, surr2);

                float entropy = 0.0f;
                if (agent->config.discrete_actions)
                {
                    for (int a = 0; a < (agent->config.discrete_actions ? agent->config.num_actions : agent->config.action_dim); a++)
                    {
                        if (probs[a] > 1e-7f)
                            entropy -= probs[a] * logf(probs[a]);
                    }
                }
                float total_loss = p_loss + pc->value_coef * v_loss - pc->entropy_coef * entropy;
                /* P1-034修复：使用独立的target和loss缓冲区，避免同一float既作梯度又作损失 */
                {
                    float actor_target[1] = {total_loss};
                    float actor_loss = 0.0f;
                    lnn_backward(actor, actor_target, &actor_loss);
                }
            }
        }
    }
    agent->total_steps++;

    safe_free((void**)&old_log_probs);
    safe_free((void**)&advantages);
    safe_free((void**)&returns);
    safe_free((void**)&values);
    safe_free((void**)&samples);
    return 0;
}

/* ============ A2C实现（独立于PPO的简化路径） ============ */

/*
 * rl_a2c_train — 真正的A2C训练（非PPO别名）
 *
 * A2C与PPO的核心区别：
 *   - A2C: 直接使用优势函数梯度，无裁剪，单轮更新
 *   - PPO: 裁剪代理目标函数，重要性采样，多轮更新
 *
 * 本实现跳过PPO的clip_epsilon/importance_ratio/update_epochs机制，
 * 使用纯A2C策略梯度：∇J ≈ ∇logπ(a|s) * A(s,a) + value_loss - entropy_bonus
 */
static int rl_a2c_train(RLAgent* agent, int batch_size)
{
    RLConfigPPO* pc = &agent->config.algo_config.ppo;
    RLReplayBuffer* buf = &agent->config.replay_buffer;
    if (buf->size < batch_size) return 0;

    int alloc_batch = (batch_size < RL_MAX_BATCH_SIZE) ? batch_size : RL_MAX_BATCH_SIZE;
    RLExperience* samples = (RLExperience*)safe_calloc((size_t)alloc_batch, sizeof(RLExperience));
    if (!samples) return -1;
    int actual = rl_replay_buffer_sample(buf, batch_size, samples, NULL, NULL);
    if (actual <= 0) { safe_free((void**)&samples); return 0; }

    LNN* actor = agent->networks[RL_NETWORK_ACTOR];
    LNN* critic = agent->networks[RL_NETWORK_CRITIC];
    if (!actor || !critic) { safe_free((void**)&samples); return -1; }

    float* advantages = (float*)safe_malloc((size_t)actual * sizeof(float));
    float* returns = (float*)safe_malloc((size_t)actual * sizeof(float));
    float* values = (float*)safe_malloc((size_t)actual * sizeof(float));
    if (!advantages || !returns || !values)
    {
        safe_free((void**)&advantages);
        safe_free((void**)&returns);
        safe_free((void**)&values);
        return -1;
    }

    /* 步骤1：前向计算状态值 */
    for (int i = 0; i < actual; i++) {
        float v = 0.0f;
        lnn_forward(critic, samples[i].state, &v);
        values[i] = v;
    }

    /* 步骤2：TD残差 + 反向递推GAE */
    float last_gae = 0.0f;
    for (int i = actual - 1; i >= 0; i--) {
        float delta = samples[i].reward - values[i];
        if (!samples[i].done) {
            float next_v = 0.0f;
            lnn_forward(critic, samples[i].next_state, &next_v);
            delta += pc->gamma * next_v;
            last_gae = delta + pc->gamma * pc->gae_lambda * last_gae;
        } else {
            last_gae = delta;
        }
        advantages[i] = last_gae;
        returns[i] = advantages[i] + values[i];
    }

    /* 步骤3：优势函数标准化 */
    float adv_mean = 0.0f, adv_std = 0.0f;
    for (int i = 0; i < actual; i++) adv_mean += advantages[i];
    adv_mean /= (float)actual;
    for (int i = 0; i < actual; i++) adv_std += RL_SQ(advantages[i] - adv_mean);
    adv_std = sqrtf(adv_std / (float)actual + 1e-7f);
    if (adv_std < 1e-7f) adv_std = 1.0f;
    for (int i = 0; i < actual; i++) advantages[i] = (advantages[i] - adv_mean) / adv_std;

    /* 步骤4：A2C单轮更新（无裁剪、无重要性采样、无多epoch） */
    int perm[RL_MAX_BATCH_SIZE];
    for (int i = 0; i < actual; i++) perm[i] = i;
    for (int i = actual - 1; i > 0; i--)
    {
        int j = (int)(rl_randf() * (float)(i + 1));
        int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }

    for (int start = 0; start < actual; start += pc->mini_batch_size)
    {
        int end = RL_MIN(start + pc->mini_batch_size, actual);
        for (int m = start; m < end; m++)
        {
            int idx = perm[m];

            /* Critic更新 */
            float new_v = 0.0f;
            lnn_forward(critic, samples[idx].state, &new_v);
            float v_loss = RL_SQ(returns[idx] - new_v);
            float v_target[1] = {returns[idx]};
            lnn_backward(critic, v_target, &v_loss);

            /* Actor更新：纯策略梯度（无PPO裁剪） */
            float probs[RL_MAX_ACTION_DIM];
            int sel_a = 0;
            rl_ppo_get_action_probs(agent, samples[idx].state,
                samples[idx].state_dim, probs, &sel_a);

            /* A2C不使用importance ratio，直接使用优势函数梯度 */
            float log_pi = logf(RL_MAX(probs[sel_a], 1e-7f));
            float p_loss = -log_pi * advantages[idx];

            float entropy = 0.0f;
            if (agent->config.discrete_actions)
            {
                for (int a = 0; a < (agent->config.discrete_actions ? agent->config.num_actions : agent->config.action_dim); a++)
                {
                    if (probs[a] > 1e-7f)
                        entropy -= probs[a] * logf(probs[a]);
                }
            }
            float total_loss = p_loss + pc->value_coef * v_loss - pc->entropy_coef * entropy;
            /* P1-034修复：使用独立的target和loss缓冲区，避免同一float既作梯度又作损失 */
            {
                float actor_target[1] = {total_loss};
                float actor_loss = 0.0f;
                lnn_backward(actor, actor_target, &actor_loss);
            }
        }
    }

    agent->total_steps++;

    safe_free((void**)&advantages);
    safe_free((void**)&returns);
    safe_free((void**)&values);
    safe_free((void**)&samples);
    return 0;
}

static int rl_sac_init(RLAgent* agent)
{
    RLConfigSAC* sc = &agent->config.algo_config.sac;
    if (sc->hidden_size <= 0) sc->hidden_size = 128;
    if (sc->actor_lr <= 0.0f) sc->actor_lr = 0.0003f;
    if (sc->critic_lr <= 0.0f) sc->critic_lr = 0.0003f;
    if (sc->alpha_lr <= 0.0f) sc->alpha_lr = 0.0003f;
    if (sc->gamma <= 0.0f) sc->gamma = 0.99f;
    if (sc->tau <= 0.0f) sc->tau = 0.005f;
    if (sc->init_alpha <= 0.0f) sc->init_alpha = 0.2f;

    int act_dim = agent->config.discrete_actions ?
                  agent->config.num_actions : agent->config.action_dim;

    sc->actor_lnn_config.input_size = (size_t)agent->config.state_dim;
    sc->actor_lnn_config.hidden_size = (size_t)sc->hidden_size;
    sc->actor_lnn_config.output_size = (size_t)(act_dim * 2);
    sc->actor_lnn_config.learning_rate = sc->actor_lr;
    sc->actor_lnn_config.num_layers = sc->num_layers > 0 ? sc->num_layers : 2;
    sc->actor_lnn_config.enable_training = 1;
    int ret = rl_lnn_create_if(agent, RL_NETWORK_ACTOR, &sc->actor_lnn_config);
    if (ret != 0) return -1;

    sc->critic_lnn_config.input_size = (size_t)(agent->config.state_dim + act_dim);
    sc->critic_lnn_config.hidden_size = (size_t)sc->hidden_size;
    sc->critic_lnn_config.output_size = 1;
    sc->critic_lnn_config.learning_rate = sc->critic_lr;
    sc->critic_lnn_config.num_layers = sc->num_layers;
    sc->critic_lnn_config.enable_training = 1;
    ret = rl_lnn_create_if(agent, RL_NETWORK_CRITIC, &sc->critic_lnn_config);
    if (ret != 0) return -1;

    sc->critic_lnn_config.learning_rate = sc->critic_lr * 0.5f;
    ret = rl_lnn_create_if(agent, RL_NETWORK_CRITIC2, &sc->critic_lnn_config);
    if (ret != 0) return -1;

/* 创建SAC目标Critic网络用于TAU软更新 */
    LNNConfig target_cfg;
    memset(&target_cfg, 0, sizeof(LNNConfig));
    target_cfg.input_size = sc->critic_lnn_config.input_size;
    target_cfg.hidden_size = sc->critic_lnn_config.hidden_size;
    target_cfg.output_size = sc->critic_lnn_config.output_size;
    target_cfg.learning_rate = 0.0f;
    target_cfg.num_layers = sc->critic_lnn_config.num_layers;
    target_cfg.enable_training = 0;
    ret = rl_lnn_create_if(agent, RL_NETWORK_CRITIC1_TARGET, &target_cfg);
    if (ret != 0) return -1;
    ret = rl_lnn_create_if(agent, RL_NETWORK_CRITIC2_TARGET, &target_cfg);
    if (ret != 0) return -1;

    agent->alpha = sc->init_alpha;
    agent->log_alpha = (float*)safe_malloc(sizeof(float));
    if (agent->log_alpha) *agent->log_alpha = logf(sc->init_alpha);

    if (sc->automatic_entropy_tuning)
    {
        if (sc->target_entropy <= 0.0f)
            agent->target_entropy = -(float)act_dim;
        else
            agent->target_entropy = sc->target_entropy;
    }
    return 0;
}

static void rl_sac_get_action(RLAgent* agent, const float* state, float* action_out,
                              float* log_prob_out, int state_dim)
{
    (void)state_dim;
    LNN* actor = agent->networks[RL_NETWORK_ACTOR];
    if (!actor) return;

    LNNConfig acfg;
    memset(&acfg, 0, sizeof(LNNConfig));
    lnn_get_config(actor, &acfg);
    int out_dim = (int)acfg.output_size;
    int act_dim = out_dim / 2;

    float* raw = (float*)safe_malloc((size_t)out_dim * sizeof(float));
    if (!raw) return;

    lnn_forward(actor, state, raw);

    float log_prob = 0.0f;
    float* action = action_out;

    for (int i = 0; i < act_dim; i++)
    {
        float mean = raw[i];
        float log_std = RL_CLAMP(raw[i + act_dim], -5.0f, 2.0f);
        float std = expf(log_std);
        float noise = rl_randn(0.0f, 1.0f);
        float u = mean + noise * std;
        float a = rl_tanh(u);

        float hi = agent->config.action_high[i];
        float lo = agent->config.action_low[i];
        action[i] = lo + (a + 1.0f) * 0.5f * (hi - lo);
        action[i] = RL_CLAMP(action[i], lo, hi);

        float a_tanh = (action[i] - lo) / (RL_MAX(hi - lo, 1e-7f)) * 2.0f - 1.0f;
        a_tanh = RL_CLAMP(a_tanh, -0.9999f, 0.9999f);
        float u_actual = atanhf(a_tanh);

        float lp = rl_gaussian_log_prob(u_actual, mean, log_std);
        float correction = logf(1.0f - a_tanh * a_tanh + 1e-7f);
        log_prob += lp - correction;
    }

    if (log_prob_out) *log_prob_out = log_prob;
    safe_free((void**)&raw);
}

static int rl_sac_train(RLAgent* agent, int batch_size)
{
    RLConfigSAC* sc = &agent->config.algo_config.sac;
    RLReplayBuffer* buf = &agent->config.replay_buffer;
    if (buf->size < batch_size) return 0;

    int alloc_batch = (batch_size < RL_MAX_BATCH_SIZE) ? batch_size : RL_MAX_BATCH_SIZE;
    RLExperience* samples = (RLExperience*)safe_calloc((size_t)alloc_batch, sizeof(RLExperience));
    if (!samples) return -1;
    int actual = rl_replay_buffer_sample(buf, batch_size, samples, NULL, NULL);
    if (actual <= 0) { safe_free((void**)&samples); return 0; }

    LNN* critic1 = agent->networks[RL_NETWORK_CRITIC];
    LNN* critic2 = agent->networks[RL_NETWORK_CRITIC2];
    LNN* actor = agent->networks[RL_NETWORK_ACTOR];
    if (!critic1 || !critic2 || !actor) { safe_free((void**)&samples); return -1; }

    int state_act_dim = agent->config.state_dim + (agent->config.discrete_actions ?
                        agent->config.num_actions : agent->config.action_dim);

    float* sa_pair = (float*)safe_malloc((size_t)state_act_dim * sizeof(float));
    float* next_sa = (float*)safe_malloc((size_t)state_act_dim * sizeof(float));

    for (int i = 0; i < actual; i++)
    {
        if (!sa_pair || !next_sa) break;

        memcpy(sa_pair, samples[i].state, (size_t)samples[i].state_dim * sizeof(float));
        memcpy(sa_pair + samples[i].state_dim, samples[i].action, (size_t)samples[i].action_dim * sizeof(float));

        float q1 = 0.0f, q2 = 0.0f;
        lnn_forward(critic1, sa_pair, &q1);
        lnn_forward(critic2, sa_pair, &q2);

        float new_action[RL_MAX_ACTION_DIM];
        float new_log_prob = 0.0f;
        rl_sac_get_action(agent, samples[i].next_state, new_action, &new_log_prob, samples[i].next_state_dim);

        float* next_sa_buf = next_sa;
        if (next_sa_buf)
        {
            memcpy(next_sa_buf, samples[i].next_state, (size_t)samples[i].next_state_dim * sizeof(float));
            memcpy(next_sa_buf + samples[i].next_state_dim, new_action, (size_t)samples[i].action_dim * sizeof(float));

            float nq1 = 0.0f, nq2 = 0.0f;
            lnn_forward(critic1, next_sa_buf, &nq1);
            lnn_forward(critic2, next_sa_buf, &nq2);

            float min_q = RL_MIN(nq1, nq2);
            float target_q = samples[i].reward + sc->gamma * (min_q - agent->alpha * new_log_prob);
            if (samples[i].done) target_q = samples[i].reward;

            float q1_loss = RL_SQ(target_q - q1);
            float q2_loss = RL_SQ(target_q - q2);

            float q1_target[1] = {target_q};
            float q2_target[1] = {target_q};
            lnn_backward(critic1, q1_target, &q1_loss);
            lnn_backward(critic2, q2_target, &q2_loss);

            float q_for_actor = RL_MIN(q1, q2);
            float actor_loss = agent->alpha * new_log_prob - q_for_actor;
/*修复S-005: 标准SAC重参数化梯度更新
             * 标准SAC演员梯度(J_actor = E[α·logπ(a|s) - Q(s,a)])通过重参数化计算:
             *   a = tanh(μ + σ·ε), ε~N(0,1)
             *   ∂J/∂μ_i = α·∂logπ/∂μ_i - ∂Q/∂a_i·∂a_i/∂μ_i
             *   ∂logπ/∂μ_i = (u_i-μ_i)/σ²_i (Gaussian)
             *   ∂a_i/∂μ_i = 1 - tanh²(u_i) = 1 - a²_i
             * 对于Q梯度∂Q/∂a,使用有限差分估计: (Q(a+ε)-Q(a-ε))/(2ε) */
            LNNConfig acfg;
            memset(&acfg, 0, sizeof(LNNConfig));
            lnn_get_config(actor, &acfg);
            int out_dim = (int)acfg.output_size;
            int act_dim = out_dim / 2;
            float* actor_out = (float*)safe_malloc((size_t)(out_dim > 0 ? out_dim : 1) * sizeof(float));
            float* actor_target = (float*)safe_malloc((size_t)(out_dim > 0 ? out_dim : 1) * sizeof(float));
            if (actor_out && actor_target) {
                lnn_forward(actor, sa_pair, actor_out);
                float lr = sc->actor_lr * 0.3f;
                /* 构造Q梯度估计的perturbed state-action对 */
                float* ap_plus  = (float*)safe_malloc((size_t)(act_dim * 2) * sizeof(float));
                float* ap_minus = (float*)safe_malloc((size_t)(act_dim * 2) * sizeof(float));
                if (ap_plus && ap_minus) {
                    for (int k = 0; k < act_dim; k++) {
                        float mu = actor_out[k];
                        float logstd = RL_CLAMP(actor_out[k + act_dim], -5.0f, 2.0f);
                        float sigma = expf(logstd);
                        /* 从actor_out重构当前动作 */
                        int sdim = samples[i].state_dim;
                        float u_i = (sa_pair[sdim + k] - agent->config.action_low[k]) / 
                                     RL_MAX(agent->config.action_high[k] - agent->config.action_low[k], 1e-7f) 
                                   * 2.0f - 1.0f;
                        u_i = RL_CLAMP(u_i, -0.9999f, 0.9999f);
                        float a_i = tanhf(u_i);
                        float da_dmu = 1.0f - a_i * a_i + 1e-8f;
                        /* 有限差分估计∂Q/∂a_i */
                        float eps = 0.001f;
                        memcpy(ap_plus, sa_pair, (size_t)(sdim + act_dim) * sizeof(float));
                        memcpy(ap_minus, sa_pair, (size_t)(sdim + act_dim) * sizeof(float));
                        ap_plus[sdim + k] += eps;
                        ap_minus[sdim + k] -= eps;
                        float qp = 0.0f, qm = 0.0f;
                        float qp_out[1] = {0}, qm_out[1] = {0};
                        lnn_forward(critic1, ap_plus, qp_out);  qp = qp_out[0];
                        lnn_forward(critic1, ap_minus, qm_out); qm = qm_out[0];
                        float dq_da = (qp - qm) / (2.0f * eps);
                        /* 重参数化梯度 */
                        float dlogpi_dmu = (u_i - mu) / (sigma * sigma + 1e-8f);
                        float dJ_dmu = agent->alpha * dlogpi_dmu - dq_da * da_dmu;
                        float target_mu = mu - lr * dJ_dmu;
                        target_mu = RL_CLAMP(target_mu, -3.0f, 3.0f);
                        actor_target[k] = target_mu;
                        /* log_std梯度: ∂J/∂logσ = α·( (u-μ)²/σ² - 1 ) */
                        float dJ_dlogstd = agent->alpha * ((u_i - mu) * (u_i - mu) / (sigma * sigma + 1e-8f) - 1.0f);
                        actor_target[k + act_dim] = logstd - lr * 0.5f * dJ_dlogstd;
                    }
                    lnn_backward(actor, actor_target, &actor_loss);
                }
                safe_free((void**)&ap_minus);
                safe_free((void**)&ap_plus);
            }
            safe_free((void**)&actor_out);
            safe_free((void**)&actor_target);
        }

        if (sc->automatic_entropy_tuning && agent->log_alpha)
        {
            float alpha_loss = -(*agent->log_alpha) * (new_log_prob + agent->target_entropy);
            *agent->log_alpha += sc->alpha_lr * alpha_loss;
            agent->alpha = expf(*agent->log_alpha);
            agent->alpha = RL_CLAMP(agent->alpha, 0.001f, 10.0f);
        }

/* SAC目标Critic网络TAU软更新 */
        {
            float tau = sc->tau;
            LNN* c1 = agent->networks[RL_NETWORK_CRITIC];
            LNN* c1t = agent->networks[RL_NETWORK_CRITIC1_TARGET];
            LNN* c2 = agent->networks[RL_NETWORK_CRITIC2];
            LNN* c2t = agent->networks[RL_NETWORK_CRITIC2_TARGET];
            float* c1p = c1 ? lnn_get_parameters(c1) : NULL;
            float* c1tp = c1t ? lnn_get_parameters(c1t) : NULL;
            size_t c1n = c1 ? lnn_get_parameter_count(c1) : 0;
            if (c1p && c1tp && c1n > 0) {
                for (size_t p = 0; p < c1n; p++)
                    c1tp[p] = (1.0f - tau) * c1tp[p] + tau * c1p[p];
            }
            float* c2p = c2 ? lnn_get_parameters(c2) : NULL;
            float* c2tp = c2t ? lnn_get_parameters(c2t) : NULL;
            size_t c2n = c2 ? lnn_get_parameter_count(c2) : 0;
            if (c2p && c2tp && c2n > 0) {
                for (size_t p = 0; p < c2n; p++)
                    c2tp[p] = (1.0f - tau) * c2tp[p] + tau * c2p[p];
            }
        }

    }
    agent->total_steps++;
    safe_free((void**)&sa_pair);
    safe_free((void**)&next_sa);
    safe_free((void**)&samples);
    return 0;
}

/* ============ A2C 实现 ============ */

static int rl_a2c_init(RLAgent* agent)
{
    RLConfigPPO a2c_cfg;
    memset(&a2c_cfg, 0, sizeof(RLConfigPPO));
    a2c_cfg.actor_hidden_size = 64;
    a2c_cfg.critic_hidden_size = 64;
    a2c_cfg.actor_lr = 0.0003f;
    a2c_cfg.critic_lr = 0.001f;
    a2c_cfg.gamma = 0.99f;
    a2c_cfg.gae_lambda = 1.0f;
    a2c_cfg.entropy_coef = 0.01f;
    a2c_cfg.value_coef = 0.5f;
    a2c_cfg.update_epochs = 1;
    a2c_cfg.mini_batch_size = 1;
    agent->config.algo_config.ppo = a2c_cfg;
    return rl_ppo_init(agent);
}

/* ============ CfC-TD3 实现 ============ */

static int rl_cfc_td3_configure_lnn(LNNConfig* cfg, int state_dim, int hidden_size,
                                    int num_layers, int output_size, float lr,
                                    float cfc_tau, int cfc_ode_solver, int multi_ts)
{
    /* MID-010修复: 将多时间尺度传入配置 */
    memset(cfg, 0, sizeof(LNNConfig));
    cfg->input_size = (size_t)state_dim;
    cfg->hidden_size = (size_t)hidden_size;
    cfg->output_size = (size_t)output_size;
    cfg->learning_rate = lr;
    cfg->num_layers = num_layers > 0 ? num_layers : 2;
    cfg->enable_training = 1;
    cfg->time_constant = cfc_tau;
    /* MID-010修复: 多时间尺度通过调节时间常数实现 */
    if (multi_ts > 0) cfg->time_constant *= 0.5f;
    cfg->ode_solver_type = cfc_ode_solver;
    return 0;
}

int rl_cfc_td3_init(RLAgent* agent)
{
    RLConfigCfCTD3* tc = &agent->config.algo_config.cfc_td3;
    int act_dim = agent->config.discrete_actions ?
                  agent->config.num_actions : agent->config.action_dim;

    if (tc->actor_hidden_size <= 0) tc->actor_hidden_size = 128;
    if (tc->critic_hidden_size <= 0) tc->critic_hidden_size = 128;
    if (tc->actor_lr <= 0.0f) tc->actor_lr = 0.0003f;
    if (tc->critic_lr <= 0.0f) tc->critic_lr = 0.0003f;
    if (tc->gamma <= 0.0f) tc->gamma = 0.99f;
    if (tc->tau <= 0.0f) tc->tau = 0.005f;
    if (tc->policy_noise <= 0.0f) tc->policy_noise = 0.2f;
    if (tc->noise_clip <= 0.0f) tc->noise_clip = 0.5f;
    if (tc->policy_delay <= 0) tc->policy_delay = 2;
    if (tc->cfc_time_constant <= 0.0f) tc->cfc_time_constant = 1.0f;

    float cfc_tau = tc->cfc_time_constant;

    /* 创建演员网络（CfC LNN） */
    rl_cfc_td3_configure_lnn(&tc->actor_lnn_config,
        agent->config.state_dim, tc->actor_hidden_size,
        tc->actor_num_layers, act_dim, tc->actor_lr,
        cfc_tau, tc->cfc_ode_solver, tc->cfc_use_multi_timescale);
    int ret = rl_lnn_create_if(agent, RL_NETWORK_ACTOR, &tc->actor_lnn_config);
    if (ret != 0) return -1;

    /* 创建目标演员网络 */
    tc->actor_lnn_config.learning_rate = tc->actor_lr * 0.2f;
    ret = rl_lnn_create_if(agent, RL_NETWORK_ACTOR_TARGET, &tc->actor_lnn_config);
    if (ret != 0) return -1;

    /* 评论家输入维度 = state_dim + action_dim */
    int critic_input_dim = agent->config.state_dim + act_dim;

    /* 创建评论家1网络 */
    rl_cfc_td3_configure_lnn(&tc->critic_lnn_config,
        critic_input_dim, tc->critic_hidden_size,
        tc->critic_num_layers, 1, tc->critic_lr,
        cfc_tau, tc->cfc_ode_solver, tc->cfc_use_multi_timescale);
    ret = rl_lnn_create_if(agent, RL_NETWORK_CRITIC, &tc->critic_lnn_config);
    if (ret != 0) return -1;

    /* 创建评论家2网络 */
    tc->critic_lnn_config.learning_rate = tc->critic_lr * 0.8f;
    ret = rl_lnn_create_if(agent, RL_NETWORK_CRITIC2, &tc->critic_lnn_config);
    if (ret != 0) return -1;

    /* 创建评论家1目标网络 */
    tc->critic_lnn_config.learning_rate = tc->critic_lr * 0.1f;
    ret = rl_lnn_create_if(agent, RL_NETWORK_CRITIC1_TARGET, &tc->critic_lnn_config);
    if (ret != 0) return -1;

    agent->td3_critic_updates = 0;
    agent->td3_avg_critic_loss = 0.0f;
    agent->td3_avg_actor_loss = 0.0f;
    agent->td3_avg_q_value = 0.0f;

    return 0;
}

static int rl_cfc_td3_select_action_internal(RLAgent* agent, const float* state,
                                              int state_dim, float* action, int action_dim,
                                              int add_noise)
{
    (void)state_dim;
    LNN* actor = agent->networks[RL_NETWORK_ACTOR];
    if (!actor) return -1;

    int d = RL_MIN(action_dim, agent->config.action_dim);
    float* raw_action = (float*)safe_malloc((size_t)d * sizeof(float));
    if (!raw_action) return -1;

    int ret = lnn_forward(actor, state, raw_action);
    if (ret != 0) { safe_free((void**)&raw_action); return -1; }

    for (int i = 0; i < d; i++)
    {
        float a = raw_action[i];
        if (add_noise)
        {
            a += rl_randn(0.0f, agent->config.explore_config.noise_std);
        }
        if (agent->config.discrete_actions)
        {
            int best = 0;
            float max_v = raw_action[0];
            for (int j = 1; j < d; j++)
                if (raw_action[j] > max_v) { max_v = raw_action[j]; best = j; }
            action[0] = (float)best;
        }
        else
        {
            a = RL_CLAMP(a, agent->config.action_low[i], agent->config.action_high[i]);
            action[i] = a;
        }
    }
    safe_free((void**)&raw_action);
    return 0;
}

static int rl_cfc_td3_train(RLAgent* agent, int batch_size)
{
    RLConfigCfCTD3* tc = &agent->config.algo_config.cfc_td3;
    RLReplayBuffer* buf = &agent->config.replay_buffer;
    if (buf->size < batch_size) return 0;

    int alloc_batch = (batch_size < RL_MAX_BATCH_SIZE) ? batch_size : RL_MAX_BATCH_SIZE;
    RLExperience* samples = (RLExperience*)safe_calloc((size_t)alloc_batch, sizeof(RLExperience));
    int* indices = (int*)safe_calloc((size_t)alloc_batch, sizeof(int));
    float* weights = (float*)safe_calloc((size_t)alloc_batch, sizeof(float));
    if (!samples || !indices || !weights) {
        safe_free((void**)&samples); safe_free((void**)&indices); safe_free((void**)&weights);
        return -1;
    }

    int actual = rl_replay_buffer_sample(buf, batch_size, samples, indices, weights);
    if (actual <= 0) {
        safe_free((void**)&samples); safe_free((void**)&indices); safe_free((void**)&weights);
        return 0;
    }

    LNN* actor = agent->networks[RL_NETWORK_ACTOR];
    LNN* actor_target = agent->networks[RL_NETWORK_ACTOR_TARGET];
    LNN* critic1 = agent->networks[RL_NETWORK_CRITIC];
    LNN* critic2 = agent->networks[RL_NETWORK_CRITIC2];
    LNN* critic1_target = agent->networks[RL_NETWORK_CRITIC1_TARGET];

    if (!actor || !actor_target || !critic1 || !critic2 || !critic1_target) {
        safe_free((void**)&samples); safe_free((void**)&indices); safe_free((void**)&weights);
        return -1;
    }

    int act_dim = agent->config.discrete_actions ?
                  agent->config.num_actions : agent->config.action_dim;
    int sa_dim = agent->config.state_dim + act_dim;
    float total_critic_loss = 0.0f;
    float total_q_value = 0.0f;

    for (int i = 0; i < actual; i++)
    {
        /* 构建状态-动作对 */
        float* sa_current = (float*)safe_malloc((size_t)sa_dim * sizeof(float));
        float* sa_next = (float*)safe_malloc((size_t)sa_dim * sizeof(float));
        if (!sa_current || !sa_next)
        {
            safe_free((void**)&sa_current);
            safe_free((void**)&sa_next);
            continue;
        }

        memcpy(sa_current, samples[i].state, (size_t)samples[i].state_dim * sizeof(float));
        memcpy(sa_current + samples[i].state_dim, samples[i].action, (size_t)samples[i].action_dim * sizeof(float));

        /* 目标策略平滑：添加裁剪噪声到目标动作 */
        float* target_action = (float*)safe_malloc((size_t)act_dim * sizeof(float));
        if (!target_action) { safe_free((void**)&sa_current); safe_free((void**)&sa_next); continue; }

        lnn_forward(actor_target, samples[i].next_state, target_action);
        for (int a = 0; a < act_dim; a++)
        {
            float noise = RL_CLAMP(rl_randn(0.0f, tc->policy_noise), -tc->noise_clip, tc->noise_clip);
            target_action[a] += noise;
            if (!agent->config.discrete_actions)
                target_action[a] = RL_CLAMP(target_action[a],
                    agent->config.action_low[a], agent->config.action_high[a]);
        }

        memcpy(sa_next, samples[i].next_state, (size_t)samples[i].next_state_dim * sizeof(float));
        memcpy(sa_next + samples[i].next_state_dim, target_action, (size_t)act_dim * sizeof(float));

        /* 计算目标Q值：min(Q1_target, Q2_target) + 缺少第二个评论家目标，用当前评论家2替代 */
        float q1_next = 0.0f, q2_next = 0.0f;
        lnn_forward(critic1_target, sa_next, &q1_next);
        lnn_forward(critic2, sa_next, &q2_next);

        float min_q_next = RL_MIN(q1_next, q2_next);
        float target_q = samples[i].reward + tc->gamma * min_q_next;
        if (samples[i].done) target_q = samples[i].reward;

        /* 更新评论家1 */
        float q1 = 0.0f;
        lnn_forward(critic1, sa_current, &q1);
        float c1_loss = (target_q - q1) * (target_q - q1) * weights[i];
        total_critic_loss += c1_loss;
        total_q_value += q1;

        float c1_target[1] = {target_q};
        lnn_backward(critic1, c1_target, &c1_loss);

        /* 更新评论家2 */
        float q2 = 0.0f;
        lnn_forward(critic2, sa_current, &q2);
        float c2_loss = (target_q - q2) * (target_q - q2) * weights[i];
        float c2_target[1] = {target_q};
        lnn_backward(critic2, c2_target, &c2_loss);

        safe_free((void**)&sa_current);
        safe_free((void**)&sa_next);
        safe_free((void**)&target_action);
    }

    agent->td3_critic_updates++;
    agent->td3_avg_critic_loss = agent->td3_avg_critic_loss * 0.9f +
                                 (total_critic_loss / (float)actual) * 0.1f;
    agent->td3_avg_q_value = agent->td3_avg_q_value * 0.9f +
                             (total_q_value / (float)actual) * 0.1f;

    /* 延迟策略更新 */
    if (agent->td3_critic_updates % tc->policy_delay == 0)
    {
        float total_actor_loss = 0.0f;
        for (int i = 0; i < actual; i++)
        {
            float* actor_action = (float*)safe_malloc((size_t)act_dim * sizeof(float));
            if (!actor_action) continue;

            lnn_forward(actor, samples[i].state, actor_action);

            float* sa_pair = (float*)safe_malloc((size_t)sa_dim * sizeof(float));
            if (!sa_pair) { safe_free((void**)&actor_action); continue; }

            memcpy(sa_pair, samples[i].state, (size_t)samples[i].state_dim * sizeof(float));
            memcpy(sa_pair + samples[i].state_dim, actor_action, (size_t)act_dim * sizeof(float));

            float q1_val = 0.0f;
            lnn_forward(critic1, sa_pair, &q1_val);
            float actor_loss = -q1_val;
            total_actor_loss += actor_loss;

            /* P1-034修复：TD3 actor使用独立target/loss缓冲区 */
            float actor_target_loss_buf[1] = {actor_loss};
            lnn_backward(actor, actor_target_loss_buf, &actor_loss);

            safe_free((void**)&sa_pair);
            safe_free((void**)&actor_action);
        }

        agent->td3_avg_actor_loss = agent->td3_avg_actor_loss * 0.9f +
                                    (total_actor_loss / (float)actual) * 0.1f;

        /* 软更新目标网络 */
        float tau = tc->tau;
        float* actor_params = lnn_get_parameters(actor);
        float* actor_target_params = lnn_get_parameters(actor_target);
        size_t actor_np = lnn_get_parameter_count(actor);
        if (actor_params && actor_target_params && actor_np > 0)
        {
            for (size_t p = 0; p < actor_np; p++)
                actor_target_params[p] = (1.0f - tau) * actor_target_params[p] + tau * actor_params[p];
        }

        float* c1_params = lnn_get_parameters(critic1);
        float* c1t_params = lnn_get_parameters(critic1_target);
        size_t c1_np = lnn_get_parameter_count(critic1);
        if (c1_params && c1t_params && c1_np > 0)
        {
            for (size_t p = 0; p < c1_np; p++)
                c1t_params[p] = (1.0f - tau) * c1t_params[p] + tau * c1_params[p];
        }
    }

    agent->total_steps++;
    safe_free((void**)&samples);
    safe_free((void**)&indices);
    safe_free((void**)&weights);
    return 0;
}

/* ============ CfC-Rainbow DQN 实现 ============ */

static int rl_cfc_rainbow_init(RLAgent* agent)
{
    RLConfigCfCRainbow* rc = &agent->config.algo_config.cfc_rainbow;
    int num_actions = agent->config.discrete_actions ?
                      agent->config.num_actions : agent->config.action_dim;

    if (rc->hidden_size <= 0) rc->hidden_size = 64;
    if (rc->num_layers <= 0) rc->num_layers = 2;
    if (rc->learning_rate <= 0.0f) rc->learning_rate = 0.0001f;
    if (rc->gamma <= 0.0f) rc->gamma = 0.99f;
    if (rc->tau <= 0.0f) rc->tau = 0.005f;
    if (rc->target_update_freq <= 0) rc->target_update_freq = 100;
    if (rc->num_atoms <= 0) rc->num_atoms = 51;
    if (rc->multi_step_n <= 0) rc->multi_step_n = 3;
    if (rc->v_max <= 0.0f) { rc->v_min = -10.0f; rc->v_max = 10.0f; }
    if (rc->cfc_time_constant <= 0.0f) rc->cfc_time_constant = 1.0f;

    agent->rainbow_num_atoms = rc->num_atoms;
    agent->rainbow_v_min = rc->v_min;
    agent->rainbow_v_max = rc->v_max;

    /* 创建原子支撑向量 */
    agent->rainbow_atom_support = (float*)safe_malloc((size_t)rc->num_atoms * sizeof(float));
    if (agent->rainbow_atom_support)
    {
        float delta = (rc->v_max - rc->v_min) / (float)(rc->num_atoms - 1);
        for (int i = 0; i < rc->num_atoms; i++)
            agent->rainbow_atom_support[i] = rc->v_min + (float)i * delta;
    }

    int output_size;
    if (rc->use_distributional)
    {
        /* 分布RL输出: 每个动作的原子概率分布 */
        output_size = num_actions * rc->num_atoms;
    }
    else
    {
        output_size = num_actions;
    }

    memset(&rc->lnn_config, 0, sizeof(LNNConfig));
    rc->lnn_config.input_size = (size_t)agent->config.state_dim;
    rc->lnn_config.hidden_size = (size_t)rc->hidden_size;
    rc->lnn_config.output_size = (size_t)output_size;
    rc->lnn_config.learning_rate = rc->learning_rate;
    rc->lnn_config.num_layers = rc->num_layers;
    rc->lnn_config.enable_training = 1;
    rc->lnn_config.time_constant = rc->cfc_time_constant;
    rc->lnn_config.ode_solver_type = rc->cfc_ode_solver;

    /* 使用dueling架构时创建分离的价值和优势流 */
    if (rc->use_dueling && rc->use_distributional)
    {
        int value_out = rc->num_atoms;
        int adv_out = num_actions * rc->num_atoms;

        LNNConfig value_cfg = rc->lnn_config;
        value_cfg.output_size = (size_t)value_out;
        int ret = rl_lnn_create_if(agent, RL_NETWORK_VALUE_STREAM, &value_cfg);
        if (ret != 0) return -1;

        LNNConfig adv_cfg = rc->lnn_config;
        adv_cfg.output_size = (size_t)adv_out;
        ret = rl_lnn_create_if(agent, RL_NETWORK_ADVANTAGE_STREAM, &adv_cfg);
        if (ret != 0) return -1;

        value_cfg.learning_rate = rc->learning_rate * 0.2f;
        ret = rl_lnn_create_if(agent, RL_NETWORK_DISTRIBUTION_TARGET, &value_cfg);
        if (ret != 0) return -1;
    }
    else
    {
        int ret = rl_lnn_create_if(agent, RL_NETWORK_DISTRIBUTION, &rc->lnn_config);
        if (ret != 0) return -1;

        rc->lnn_config.learning_rate = rc->learning_rate * 0.2f;
        ret = rl_lnn_create_if(agent, RL_NETWORK_DISTRIBUTION_TARGET, &rc->lnn_config);
        if (ret != 0) return -1;
    }

    return 0;
}

static void rl_cfc_rainbow_project_distribution(const RLAgent* agent,
    const float* next_dist, float reward, float gamma, float* projected)
{
    int num_atoms = agent->rainbow_num_atoms;
    float v_min = agent->rainbow_v_min;
    float v_max = agent->rainbow_v_max;
    float delta = (v_max - v_min) / (float)(num_atoms - 1);

    memset(projected, 0, (size_t)num_atoms * sizeof(float));

    for (int i = 0; i < num_atoms; i++)
    {
        if (fabsf(next_dist[i]) < 1e-6f) continue;

        /* 计算Bellman更新后的原子位置 */
        float tz = reward + gamma * (v_min + (float)i * delta);
        tz = RL_CLAMP(tz, v_min, v_max);

        /* 投影到相邻原子 */
        int bj = (int)((tz - v_min) / delta);
        float l = (tz - (v_min + (float)bj * delta)) / delta;

        int idx_low = RL_CLAMP(bj, 0, num_atoms - 1);
        int idx_high = RL_CLAMP(bj + 1, 0, num_atoms - 1);

        projected[idx_low] += next_dist[i] * (1.0f - l);
        projected[idx_high] += next_dist[i] * l;
    }
}

static int rl_cfc_rainbow_train(RLAgent* agent, int batch_size)
{
    RLConfigCfCRainbow* rc = &agent->config.algo_config.cfc_rainbow;
    RLReplayBuffer* buf = &agent->config.replay_buffer;
    if (buf->size < batch_size) return 0;

    int num_actions = agent->config.discrete_actions ?
                      agent->config.num_actions : agent->config.action_dim;
    int num_atoms = agent->rainbow_num_atoms;

    int alloc_batch = (batch_size < RL_MAX_BATCH_SIZE) ? batch_size : RL_MAX_BATCH_SIZE;
    RLExperience* samples = (RLExperience*)safe_calloc((size_t)alloc_batch, sizeof(RLExperience));
    int* indices = (int*)safe_calloc((size_t)alloc_batch, sizeof(int));
    float* weights = (float*)safe_calloc((size_t)alloc_batch, sizeof(float));
    if (!samples || !indices || !weights) {
        safe_free((void**)&samples); safe_free((void**)&indices); safe_free((void**)&weights);
        return -1;
    }

    int actual = rl_replay_buffer_sample(buf, batch_size, samples, indices, weights);
    if (actual <= 0) {
        safe_free((void**)&samples); safe_free((void**)&indices); safe_free((void**)&weights);
        return 0;
    }

    LNN* dist_net = agent->networks[RL_NETWORK_DISTRIBUTION];
    LNN* dist_target = agent->networks[RL_NETWORK_DISTRIBUTION_TARGET];
    LNN* value_net = agent->networks[RL_NETWORK_VALUE_STREAM];
    LNN* adv_net = agent->networks[RL_NETWORK_ADVANTAGE_STREAM];

    if (!dist_net || !dist_target || !value_net || !adv_net) {
        safe_free((void**)&samples); safe_free((void**)&indices); safe_free((void**)&weights);
        return -1;
    }

    float td_errors[RL_MAX_BATCH_SIZE];
    float total_loss = 0.0f;

    for (int i = 0; i < actual; i++)
    {
        float* current_dist = (float*)safe_malloc((size_t)(num_actions * num_atoms) * sizeof(float));
        float* next_dist = (float*)safe_malloc((size_t)(num_actions * num_atoms) * sizeof(float));
        if (!current_dist || !next_dist)
        {
            safe_free((void**)&current_dist);
            safe_free((void**)&next_dist);
            continue;
        }

        if (rc->use_dueling && rc->use_distributional && value_net && adv_net)
        {
            float value_out[RL_MAX_BATCH_SIZE];
            float adv_out[RL_MAX_BATCH_SIZE * RL_MAX_BATCH_SIZE];
            lnn_forward(value_net, samples[i].state, value_out);
            lnn_forward(adv_net, samples[i].state, adv_out);

            float adv_mean = 0.0f;
            for (int a = 0; a < num_actions; a++)
                adv_mean += adv_out[a * num_atoms];
            adv_mean /= (float)num_actions;

            for (int a = 0; a < num_actions; a++)
            {
                for (int z = 0; z < num_atoms; z++)
                {
                    float val = value_out[z] + adv_out[a * num_atoms + z] - adv_mean / (float)num_atoms;
                    current_dist[a * num_atoms + z] = val;
                }
            }

            /* softmax over atoms */
            for (int a = 0; a < num_actions; a++)
            {
                float max_v = current_dist[a * num_atoms];
                for (int z = 1; z < num_atoms; z++)
                    if (current_dist[a * num_atoms + z] > max_v)
                        max_v = current_dist[a * num_atoms + z];
                float sum = 0.0f;
                for (int z = 0; z < num_atoms; z++)
                {
                    current_dist[a * num_atoms + z] = expf(current_dist[a * num_atoms + z] - max_v);
                    sum += current_dist[a * num_atoms + z];
                }
                if (sum < 1e-7f) sum = 1e-7f;
                for (int z = 0; z < num_atoms; z++)
                    current_dist[a * num_atoms + z] /= sum;
            }

            /* 目标网络输出（仅价值流作为目标） */
            float next_value[RL_MAX_BATCH_SIZE];
            lnn_forward(dist_target, samples[i].next_state, next_value);
            float max_v = next_value[0];
            for (int z = 1; z < num_atoms; z++)
                if (next_value[z] > max_v) max_v = next_value[z];
            float sum = 0.0f;
            for (int z = 0; z < num_atoms; z++)
            {
                next_value[z] = expf(next_value[z] - max_v);
                sum += next_value[z];
            }
            if (sum > 1e-7f)
                for (int z = 0; z < num_atoms; z++)
                    next_value[z] /= sum;
            memcpy(next_dist, next_value, (size_t)num_atoms * sizeof(float));
        }
        else if (dist_net && dist_target)
        {
            float raw_out[RL_MAX_BATCH_SIZE * RL_MAX_BATCH_SIZE];
            lnn_forward(dist_net, samples[i].state, raw_out);

            /* softmax over atoms per action */
            for (int a = 0; a < num_actions; a++)
            {
                float max_v = raw_out[a * num_atoms];
                for (int z = 1; z < num_atoms; z++)
                    if (raw_out[a * num_atoms + z] > max_v)
                        max_v = raw_out[a * num_atoms + z];
                float sum = 0.0f;
                for (int z = 0; z < num_atoms; z++)
                {
                    current_dist[a * num_atoms + z] = expf(raw_out[a * num_atoms + z] - max_v);
                    sum += current_dist[a * num_atoms + z];
                }
                if (sum < 1e-7f) sum = 1e-7f;
                for (int z = 0; z < num_atoms; z++)
                    current_dist[a * num_atoms + z] /= sum;
            }

            float raw_next[RL_MAX_BATCH_SIZE * RL_MAX_BATCH_SIZE];
            lnn_forward(dist_target, samples[i].next_state, raw_next);
            for (int a = 0; a < num_actions; a++)
            {
                float max_v = raw_next[a * num_atoms];
                for (int z = 1; z < num_atoms; z++)
                    if (raw_next[a * num_atoms + z] > max_v)
                        max_v = raw_next[a * num_atoms + z];
                float sum = 0.0f;
                for (int z = 0; z < num_atoms; z++)
                {
                    next_dist[a * num_atoms + z] = expf(raw_next[a * num_atoms + z] - max_v);
                    sum += next_dist[a * num_atoms + z];
                }
                if (sum < 1e-7f) sum = 1e-7f;
                for (int z = 0; z < num_atoms; z++)
                    next_dist[a * num_atoms + z] /= sum;
            }
        }
        else
        {
            safe_free((void**)&current_dist);
            safe_free((void**)&next_dist);
            continue;
        }

        /* 选择动作：贪婪选择Q值最大的动作 */
        float* q_values = (float*)safe_malloc((size_t)num_actions * sizeof(float));
        if (q_values)
        {
            for (int a = 0; a < num_actions; a++)
            {
                q_values[a] = 0.0f;
                for (int z = 0; z < num_atoms; z++)
                    q_values[a] += agent->rainbow_atom_support[z] * current_dist[a * num_atoms + z];
            }

            int act_idx = (int)samples[i].action[0];
            if (!agent->config.discrete_actions)
            {
                float max_a = samples[i].action[0];
                for (int a = 1; a < samples[i].action_dim; a++)
                    if (samples[i].action[a] > max_a) { max_a = samples[i].action[a]; act_idx = a; }
            }
            act_idx = RL_CLAMP(act_idx, 0, num_actions - 1);

            /* Double DQN: 用在线网络选动作，目标网络评估 */
            int best_next_action = 0;
            if (rc->use_double_dqn)
            {
                float online_next_q[RL_MAX_BATCH_SIZE];
                if (dist_net)
                {
                    float raw_next_online[RL_MAX_BATCH_SIZE * RL_MAX_BATCH_SIZE];
                    lnn_forward(dist_net, samples[i].next_state, raw_next_online);
                    for (int a = 0; a < num_actions; a++)
                    {
                        online_next_q[a] = 0.0f;
                        for (int z = 0; z < num_atoms; z++)
                            online_next_q[a] += agent->rainbow_atom_support[z] *
                                expf(raw_next_online[a * num_atoms + z]);
                    }
                }
                for (int a = 1; a < num_actions; a++)
                    if (online_next_q[a] > online_next_q[best_next_action]) best_next_action = a;
            }
            else
            {
                float max_q = q_values[0];
                for (int a = 1; a < num_actions; a++)
                    if (q_values[a] > max_q) { max_q = q_values[a]; best_next_action = a; }
            }
            best_next_action = RL_CLAMP(best_next_action, 0, num_actions - 1);

            /* 投影目标分布 */
            float* projected = (float*)safe_malloc((size_t)num_atoms * sizeof(float));
            if (projected)
            {
                float gamma_pow = powf(rc->gamma, (float)rc->multi_step_n);
                const float* target_dist = (rc->use_dueling && rc->use_distributional) ?
                    next_dist : &next_dist[best_next_action * num_atoms];

                if (samples[i].done)
                {
                    memset(projected, 0, (size_t)num_atoms * sizeof(float));
                    int atom_idx = (int)((samples[i].reward - rc->v_min) /
                        ((rc->v_max - rc->v_min) / (float)(num_atoms - 1)));
                    atom_idx = RL_CLAMP(atom_idx, 0, num_atoms - 1);
                    projected[atom_idx] = 1.0f;
                }
                else
                {
                    rl_cfc_rainbow_project_distribution(agent, target_dist,
                        samples[i].reward, gamma_pow, projected);
                }

                /* 计算交叉熵损失 */
                float dist_loss = 0.0f;
                for (int z = 0; z < num_atoms; z++)
                {
                    float p = current_dist[act_idx * num_atoms + z];
                    float target_p = projected[z];
                    if (p > 1e-7f && target_p > 1e-7f)
                        dist_loss -= target_p * logf(p) * weights[i];
                }
                total_loss += dist_loss;
                td_errors[i] = dist_loss;
                lnn_backward(dist_net, projected, &dist_loss);

                safe_free((void**)&projected);
            }
            safe_free((void**)&q_values);
        }

        safe_free((void**)&current_dist);
        safe_free((void**)&next_dist);
    }

    if (buf->replay_type == RL_REPLAY_PRIORITIZED)
    {
        float abs_errors[RL_MAX_BATCH_SIZE];
        for (int i = 0; i < actual; i++)
            abs_errors[i] = fabsf(td_errors[i]);
        rl_replay_buffer_update_priorities(buf, indices, abs_errors, actual);
    }

    agent->total_steps++;
    safe_free((void**)&samples);
    safe_free((void**)&indices);
    safe_free((void**)&weights);
    return 0;
}

/* ============ CfC-IMPALA 实现 ============ */

static int rl_cfc_impala_init(RLAgent* agent)
{
    RLConfigCfCIMPALA* ic = &agent->config.algo_config.cfc_impala;
    int act_dim = agent->config.discrete_actions ?
                  agent->config.num_actions : agent->config.action_dim;

    if (ic->actor_hidden_size <= 0) ic->actor_hidden_size = 128;
    if (ic->critic_hidden_size <= 0) ic->critic_hidden_size = 128;
    if (ic->actor_lr <= 0.0f) ic->actor_lr = 0.0003f;
    if (ic->critic_lr <= 0.0f) ic->critic_lr = 0.0003f;
    if (ic->gamma <= 0.0f) ic->gamma = 0.99f;
    if (ic->entropy_coef <= 0.0f) ic->entropy_coef = 0.01f;
    if (ic->vtrace_clip_rho <= 0.0f) ic->vtrace_clip_rho = 1.0f;
    if (ic->vtrace_clip_c <= 0.0f) ic->vtrace_clip_c = 1.0f;
    if (ic->unroll_length <= 0) ic->unroll_length = 50;
    if (ic->batch_size <= 0) ic->batch_size = 32;
    if (ic->num_actors <= 0) ic->num_actors = 4;
    if (ic->cfc_time_constant <= 0.0f) ic->cfc_time_constant = 1.0f;

    int actor_out = agent->config.discrete_actions ? act_dim : act_dim * 2;

    memset(&ic->actor_lnn_config, 0, sizeof(LNNConfig));
    ic->actor_lnn_config.input_size = (size_t)agent->config.state_dim;
    ic->actor_lnn_config.hidden_size = (size_t)ic->actor_hidden_size;
    ic->actor_lnn_config.output_size = (size_t)actor_out;
    ic->actor_lnn_config.learning_rate = ic->actor_lr;
    ic->actor_lnn_config.num_layers = ic->actor_num_layers > 0 ? ic->actor_num_layers : 2;
    ic->actor_lnn_config.enable_training = 1;
    ic->actor_lnn_config.time_constant = ic->cfc_time_constant;
    ic->actor_lnn_config.ode_solver_type = ic->cfc_ode_solver;
    int ret = rl_lnn_create_if(agent, RL_NETWORK_ACTOR, &ic->actor_lnn_config);
    if (ret != 0) return -1;

    memset(&ic->critic_lnn_config, 0, sizeof(LNNConfig));
    ic->critic_lnn_config.input_size = (size_t)agent->config.state_dim;
    ic->critic_lnn_config.hidden_size = (size_t)ic->critic_hidden_size;
    ic->critic_lnn_config.output_size = 1;
    ic->critic_lnn_config.learning_rate = ic->critic_lr;
    ic->critic_lnn_config.num_layers = ic->critic_num_layers > 0 ? ic->critic_num_layers : 2;
    ic->critic_lnn_config.enable_training = 1;
    ic->critic_lnn_config.time_constant = ic->cfc_time_constant;
    ic->critic_lnn_config.ode_solver_type = ic->cfc_ode_solver;
    ret = rl_lnn_create_if(agent, RL_NETWORK_CRITIC, &ic->critic_lnn_config);
    if (ret != 0) return -1;

    /* 分配V-trace缓冲区 */
    int max_traj = ic->unroll_length * ic->batch_size;
    agent->impala_trajectory_buffer = (float*)safe_calloc(
        (size_t)(max_traj * (agent->config.state_dim + act_dim + 2)), sizeof(float));
    agent->impala_vtrace_values = (float*)safe_calloc((size_t)max_traj, sizeof(float));
    agent->impala_trajectory_length = 0;

    return 0;
}

int rl_cfc_impala_compute_vtrace(RLAgent* agent, const float* states,
    const float* actions, const float* rewards, const int* dones,
    int seq_len, float* vtrace_targets)
{
    RLConfigCfCIMPALA* ic = &agent->config.algo_config.cfc_impala;
    LNN* actor = agent->networks[RL_NETWORK_ACTOR];
    LNN* critic = agent->networks[RL_NETWORK_CRITIC];
    if (!actor || !critic || !states || !rewards || !vtrace_targets) return -1;

    int act_dim = agent->config.discrete_actions ?
                  agent->config.num_actions : agent->config.action_dim;
    int state_dim = agent->config.state_dim;

    /* 计算行为策略和目标策略的对数概率 */
    float* log_pi_target = (float*)safe_malloc((size_t)seq_len * sizeof(float));
    float* log_pi_behavior = (float*)safe_malloc((size_t)seq_len * sizeof(float));
    float* values = (float*)safe_malloc((size_t)seq_len * sizeof(float));
    float* next_values = (float*)safe_malloc((size_t)seq_len * sizeof(float));

    if (!log_pi_target || !log_pi_behavior || !values || !next_values)
    {
        safe_free((void**)&log_pi_target);
        safe_free((void**)&log_pi_behavior);
        safe_free((void**)&values);
        safe_free((void**)&next_values);
        return -1;
    }

    for (int t = 0; t < seq_len; t++)
    {
        const float* s = &states[t * state_dim];

        /* 目标策略分布 */
        float actor_out[RL_MAX_ACTION_DIM];
        lnn_forward(actor, s, actor_out);
        lnn_forward(critic, s, &values[t]);

        if (agent->config.discrete_actions)
        {
            float max_v = actor_out[0];
            for (int a = 1; a < act_dim; a++)
                if (actor_out[a] > max_v) max_v = actor_out[a];
            float sum = 0.0f;
            for (int a = 0; a < act_dim; a++)
            {
                actor_out[a] = expf(actor_out[a] - max_v);
                sum += actor_out[a];
            }
            if (sum < 1e-7f) sum = 1e-7f;
            for (int a = 0; a < act_dim; a++)
                actor_out[a] /= sum;

            int act = (int)actions[t];
            act = RL_CLAMP(act, 0, act_dim - 1);
            log_pi_target[t] = logf(actor_out[act] + 1e-7f);
            log_pi_behavior[t] = logf(1.0f / (float)act_dim); /* 假设均匀行为策略 */
        }
        else
        {
            int half = act_dim;
            float lp = 0.0f;
            float lp_behave = 0.0f;
            for (int a = 0; a < half; a++)
            {
                float mean = actor_out[a];
                float log_std = RL_CLAMP(actor_out[a + half], -5.0f, 2.0f);
                lp += rl_gaussian_log_prob(actions[t * act_dim + a], mean, log_std);
                lp_behave += rl_gaussian_log_prob(actions[t * act_dim + a], 0.0f, 1.0f);
            }
            log_pi_target[t] = lp;
            log_pi_behavior[t] = lp_behave;
        }

        if (t < seq_len - 1)
        {
            const float* ns = &states[(t + 1) * state_dim];
            lnn_forward(critic, ns, &next_values[t]);
        }
        else
        {
            next_values[t] = dones[t] ? 0.0f : values[t];
        }
    }

    /* V-trace计算 */
    float* vs = (float*)safe_malloc((size_t)seq_len * sizeof(float));
    if (!vs) { safe_free((void**)&log_pi_target); safe_free((void**)&log_pi_behavior); safe_free((void**)&values); safe_free((void**)&next_values); return -1; }

    for (int t = seq_len - 1; t >= 0; t--)
    {
        float rho_t = expf(RL_CLAMP(log_pi_target[t] - log_pi_behavior[t], -10.0f, 10.0f));
        float clipped_rho = RL_MIN(rho_t, ic->vtrace_clip_rho);
        float clipped_c = RL_MIN(rho_t, ic->vtrace_clip_c);

        float delta_v = clipped_rho * (rewards[t] + ic->gamma * next_values[t] - values[t]);

        if (t == seq_len - 1)
        {
            vs[t] = values[t] + delta_v;
        }
        else
        {
            vs[t] = values[t] + delta_v + ic->gamma * clipped_c * (vs[t + 1] - next_values[t]);
        }

        if (dones[t]) vs[t] = rewards[t];
    }

    for (int t = 0; t < seq_len; t++)
    {
        vtrace_targets[t] = vs[t];
        float advantage = vs[t] - values[t];
        agent->impala_vtrace_values[t] = advantage;
    }

    safe_free((void**)&log_pi_target);
    safe_free((void**)&log_pi_behavior);
    safe_free((void**)&values);
    safe_free((void**)&next_values);
    safe_free((void**)&vs);
    return 0;
}

static int rl_cfc_impala_train(RLAgent* agent, int batch_size)
{
    (void)batch_size;
    RLConfigCfCIMPALA* ic = &agent->config.algo_config.cfc_impala;
    RLReplayBuffer* buf = &agent->config.replay_buffer;
    (void)buf;

    LNN* actor = agent->networks[RL_NETWORK_ACTOR];
    LNN* critic = agent->networks[RL_NETWORK_CRITIC];
    if (!actor || !critic) return -1;

    int seq_len = ic->unroll_length;
    int state_dim = agent->config.state_dim;
    int act_dim = agent->config.discrete_actions ?
                  agent->config.num_actions : agent->config.action_dim;

    /* 使用轨迹缓冲区中的数据 */
    if (agent->impala_trajectory_length < seq_len) return 0;

    float* states = agent->impala_trajectory_buffer;
    float* actions = states + seq_len * state_dim;
    float* rewards = actions + seq_len * act_dim;
    int* dones = (int*)(rewards + seq_len);

    float* vtrace_targets = (float*)safe_malloc((size_t)seq_len * sizeof(float));
    if (!vtrace_targets) return -1;

    int ret = rl_cfc_impala_compute_vtrace(agent, states, actions, rewards,
                                            dones, seq_len, vtrace_targets);
    if (ret != 0) { safe_free((void**)&vtrace_targets); return -1; }

    float total_loss = 0.0f;

    for (int t = 0; t < seq_len; t++)
    {
        const float* s = &states[t * state_dim];

        /* 策略损失（带熵正则化） */
        float actor_out[RL_MAX_ACTION_DIM];
        lnn_forward(actor, s, actor_out);

        float log_prob = 0.0f;
        float entropy = 0.0f;

        if (agent->config.discrete_actions)
        {
            float max_v = actor_out[0];
            for (int a = 1; a < act_dim; a++)
                if (actor_out[a] > max_v) max_v = actor_out[a];
            float sum = 0.0f;
            for (int a = 0; a < act_dim; a++)
            {
                actor_out[a] = expf(actor_out[a] - max_v);
                sum += actor_out[a];
            }
            if (sum < 1e-7f) sum = 1e-7f;
            for (int a = 0; a < act_dim; a++)
            {
                actor_out[a] /= sum;
                if (actor_out[a] > 1e-7f)
                    entropy -= actor_out[a] * logf(actor_out[a]);
            }

            int act = (int)actions[t];
            act = RL_CLAMP(act, 0, act_dim - 1);
            log_prob = logf(actor_out[act] + 1e-7f);
        }
        else
        {
            int half = act_dim;
            for (int a = 0; a < half; a++)
            {
                float mean = actor_out[a];
                float log_std = RL_CLAMP(actor_out[a + half], -5.0f, 2.0f);
                float std = expf(log_std);
                log_prob += rl_gaussian_log_prob(actions[t * act_dim + a], mean, log_std);
                entropy += 0.5f * logf(2.0f * (float)M_PI * (float)M_E * std * std + 1e-7f);
            }
        }

        float advantage = agent->impala_vtrace_values[t];
        float policy_loss = -log_prob * advantage * RL_CLAMP(expf(log_prob), 0.0f, 1.0f);
        float entropy_loss = -ic->entropy_coef * entropy;
        float actor_loss = policy_loss + entropy_loss;
        total_loss += actor_loss;

        /* P1-034修复：IMPALA actor使用独立target/loss缓冲区 */
        float actor_target[1] = {actor_loss};
        lnn_backward(actor, actor_target, &actor_loss);

        /* 价值函数损失 */
        float value = 0.0f;
        lnn_forward(critic, s, &value);
        float v_loss = (vtrace_targets[t] - value) * (vtrace_targets[t] - value);
        float v_target[1] = {vtrace_targets[t]};
        lnn_backward(critic, v_target, &v_loss);
    }

    agent->impala_avg_entropy = agent->impala_avg_entropy * 0.99f - total_loss * 0.01f;
    agent->total_steps += seq_len;
    agent->impala_trajectory_length = 0;

    safe_free((void**)&vtrace_targets);
    return 0;
}

/* ============ CfC-R2D2 实现 ============ */

static int rl_cfc_r2d2_init(RLAgent* agent)
{
    RLConfigCfCR2D2* rc = &agent->config.algo_config.cfc_r2d2;
    int num_actions = agent->config.discrete_actions ?
                      agent->config.num_actions : agent->config.action_dim;

    if (rc->hidden_size <= 0) rc->hidden_size = 128;
    if (rc->num_layers <= 0) rc->num_layers = 2;
    if (rc->learning_rate <= 0.0f) rc->learning_rate = 0.0001f;
    if (rc->gamma <= 0.0f) rc->gamma = 0.997f;
    if (rc->tau <= 0.0f) rc->tau = 0.005f;
    if (rc->target_update_freq <= 0) rc->target_update_freq = 400;
    if (rc->burn_in_steps <= 0) rc->burn_in_steps = 10;
    if (rc->sequence_length <= 0) rc->sequence_length = 40;
    if (rc->priority_exponent <= 0.0f) rc->priority_exponent = 0.9f;
    if (rc->importance_sampling_exponent <= 0.0f) rc->importance_sampling_exponent = 0.6f;
    if (rc->cfc_time_constant <= 0.0f) rc->cfc_time_constant = 1.0f;

    agent->r2d2_recurrent_state_dim = rc->hidden_size;

    memset(&rc->lnn_config, 0, sizeof(LNNConfig));
    rc->lnn_config.input_size = (size_t)(agent->config.state_dim + num_actions);
    rc->lnn_config.hidden_size = (size_t)rc->hidden_size;
    rc->lnn_config.output_size = (size_t)num_actions;
    rc->lnn_config.learning_rate = rc->learning_rate;
    rc->lnn_config.num_layers = rc->num_layers;
    rc->lnn_config.enable_training = 1;
    rc->lnn_config.time_constant = rc->cfc_time_constant;
    rc->lnn_config.ode_solver_type = rc->cfc_ode_solver;

    int ret = rl_lnn_create_if(agent, RL_NETWORK_RECURRENT_Q, &rc->lnn_config);
    if (ret != 0) return -1;

    rc->lnn_config.learning_rate = rc->learning_rate * 0.2f;
    ret = rl_lnn_create_if(agent, RL_NETWORK_RECURRENT_Q_TARGET, &rc->lnn_config);
    if (ret != 0) return -1;

    /* 分配循环状态和序列缓冲区 */
    agent->r2d2_recurrent_state = (float*)safe_calloc((size_t)rc->hidden_size, sizeof(float));
    agent->r2d2_sequence_buffer = (float*)safe_malloc(
        (size_t)(rc->sequence_length * (agent->config.state_dim + 1 + num_actions + 1)) * sizeof(float));
    agent->r2d2_sequence_dones = (int*)safe_malloc((size_t)rc->sequence_length * sizeof(int));
    agent->r2d2_buffer_pos = 0;

    return 0;
}

int rl_cfc_r2d2_burn_in(RLAgent* agent, const float* burn_in_sequence,
                         int burn_in_length, int state_dim)
{
    RLConfigCfCR2D2* rc = &agent->config.algo_config.cfc_r2d2;
    LNN* rq_net = agent->networks[RL_NETWORK_RECURRENT_Q];
    if (!rq_net || !burn_in_sequence || burn_in_length <= 0) return -1;

    /* 重置循环状态 */
    if (agent->r2d2_recurrent_state)
        memset(agent->r2d2_recurrent_state, 0, (size_t)rc->hidden_size * sizeof(float));

    /* 执行预热前向传播以初始化循环状态 */
    for (int t = 0; t < burn_in_length; t++)
    {
        const float* s = &burn_in_sequence[t * state_dim];
        float q_output[RL_MAX_ACTION_DIM];
        lnn_forward(rq_net, s, q_output);
    }

    return 0;
}

static int rl_cfc_r2d2_train(RLAgent* agent, int batch_size)
{
    RLConfigCfCR2D2* rc = &agent->config.algo_config.cfc_r2d2;
    RLReplayBuffer* buf = &agent->config.replay_buffer;
    (void)buf;

    LNN* rq_net = agent->networks[RL_NETWORK_RECURRENT_Q];
    LNN* rq_target = agent->networks[RL_NETWORK_RECURRENT_Q_TARGET];
    if (!rq_net || !rq_target) return -1;

    int seq_len = rc->sequence_length;
    int state_dim = agent->config.state_dim;
    int num_actions = agent->config.discrete_actions ?
                      agent->config.num_actions : agent->config.action_dim;

    /* 从序列缓冲区中采样训练序列 */
    if (agent->r2d2_buffer_pos < seq_len) return 0;
    int num_sequences = agent->r2d2_buffer_pos / seq_len;
    if (num_sequences <= 0) return 0;

    float* seq_states = agent->r2d2_sequence_buffer;
    float* seq_actions = seq_states + seq_len * state_dim;
    float* seq_rewards = seq_actions + seq_len;
    float* seq_next_states = seq_rewards + seq_len;
    int* seq_dones = agent->r2d2_sequence_dones;

    /* 对每个序列进行训练 */
    for (int s = 0; s < num_sequences && s < batch_size; s++)
    {
        int offset = s * seq_len;

        /* 使用预热初始化循环状态 */
        for (int t = 0; t < rc->burn_in_steps && t < seq_len; t++)
        {
            const float* st = &seq_states[(offset + t) * state_dim];
            float q_out[RL_MAX_ACTION_DIM];
            lnn_forward(rq_net, st, q_out);
        }

        float total_seq_loss = 0.0f;

        for (int t = rc->burn_in_steps; t < seq_len; t++)
        {
            int idx = offset + t;
            const float* st = &seq_states[idx * state_dim];
            const float* st_next = &seq_next_states[idx * state_dim];
            float act = seq_actions[idx];
            float rew = seq_rewards[idx];
            int done = seq_dones[idx];

            float q_values[RL_MAX_ACTION_DIM];
            lnn_forward(rq_net, st, q_values);

            float target_q_values[RL_MAX_ACTION_DIM];
            lnn_forward(rq_target, st_next, target_q_values);

            int act_idx = (int)act;
            act_idx = RL_CLAMP(act_idx, 0, num_actions - 1);

            float max_next_q = target_q_values[0];
            for (int a = 1; a < num_actions; a++)
                if (target_q_values[a] > max_next_q) max_next_q = target_q_values[a];

            float target_q = done ? rew : rew + rc->gamma * max_next_q;
            float td_error = target_q - q_values[act_idx];
            total_seq_loss += td_error * td_error;

            float q_target[RL_MAX_ACTION_DIM];
            memcpy(q_target, q_values, (size_t)num_actions * sizeof(float));
            q_target[act_idx] = target_q;

            lnn_backward(rq_net, q_target, &td_error);
        }

        agent->total_steps++;
    }

    /* 清理缓冲区 */
    agent->r2d2_buffer_pos = 0;

    return 0;
}

/* ============ 公共API实现 ============ */

RLConfig rl_config_default(RLAlgorithm algorithm, int state_dim, int action_dim)
{
    RLConfig cfg;
    memset(&cfg, 0, sizeof(RLConfig));
    cfg.algorithm = algorithm;
    cfg.state_dim = state_dim;
    cfg.action_dim = action_dim;
    cfg.discrete_actions = 0;
    cfg.num_actions = action_dim;
    for (int i = 0; i < action_dim && i < RL_MAX_ACTION_DIM; i++)
    {
        cfg.action_low[i] = -1.0f;
        cfg.action_high[i] = 1.0f;
    }
    cfg.explore_config = rl_explore_config_default(RL_EXPLORE_EPSILON_GREEDY);
    cfg.seed = (int)time(NULL);
    cfg.verbose = 0;
    memcpy(cfg.name, "rl_agent", strlen("rl_agent") + 1);

    switch (algorithm)
    {
        case RL_ALGORITHM_DQN:
            cfg.algo_config.dqn.hidden_size = 64;
            cfg.algo_config.dqn.num_layers = 2;
            cfg.algo_config.dqn.learning_rate = 0.001f;
            cfg.algo_config.dqn.gamma = 0.99f;
            cfg.algo_config.dqn.tau = 0.005f;
            cfg.algo_config.dqn.target_update_freq = 100;
            cfg.algo_config.dqn.use_double_dqn = 1;
            cfg.algo_config.dqn.use_dueling = 0;
            cfg.algo_config.dqn.use_noisy_nets = 0;
            break;
        case RL_ALGORITHM_PPO:
            cfg.algo_config.ppo.actor_hidden_size = 64;
            cfg.algo_config.ppo.critic_hidden_size = 64;
            cfg.algo_config.ppo.actor_lr = 0.0003f;
            cfg.algo_config.ppo.critic_lr = 0.001f;
            cfg.algo_config.ppo.gamma = 0.99f;
            cfg.algo_config.ppo.gae_lambda = 0.95f;
            cfg.algo_config.ppo.clip_epsilon = 0.2f;
            cfg.algo_config.ppo.entropy_coef = 0.01f;
            cfg.algo_config.ppo.update_epochs = 10;
            cfg.algo_config.ppo.mini_batch_size = 32;
            break;
        case RL_ALGORITHM_SAC:
            cfg.algo_config.sac.hidden_size = 128;
            cfg.algo_config.sac.actor_lr = 0.0003f;
            cfg.algo_config.sac.critic_lr = 0.0003f;
            cfg.algo_config.sac.gamma = 0.99f;
            cfg.algo_config.sac.tau = 0.005f;
            cfg.algo_config.sac.init_alpha = 0.2f;
            cfg.algo_config.sac.automatic_entropy_tuning = 1;
            break;
        case RL_ALGORITHM_A2C:
            break;
        case RL_ALGORITHM_CFC_TD3:
            cfg.algo_config.cfc_td3.actor_hidden_size = 128;
            cfg.algo_config.cfc_td3.critic_hidden_size = 128;
            cfg.algo_config.cfc_td3.actor_num_layers = 2;
            cfg.algo_config.cfc_td3.critic_num_layers = 2;
            cfg.algo_config.cfc_td3.actor_lr = 0.0003f;
            cfg.algo_config.cfc_td3.critic_lr = 0.0003f;
            cfg.algo_config.cfc_td3.gamma = 0.99f;
            cfg.algo_config.cfc_td3.tau = 0.005f;
            cfg.algo_config.cfc_td3.policy_noise = 0.2f;
            cfg.algo_config.cfc_td3.noise_clip = 0.5f;
            cfg.algo_config.cfc_td3.policy_delay = 2;
            cfg.algo_config.cfc_td3.cfc_time_constant = 1.0f;
            cfg.algo_config.cfc_td3.cfc_ode_solver = 0;
            cfg.algo_config.cfc_td3.cfc_use_multi_timescale = 0;
            break;
        case RL_ALGORITHM_CFC_RAINBOW:
            cfg.algo_config.cfc_rainbow.hidden_size = 128;
            cfg.algo_config.cfc_rainbow.num_layers = 2;
            cfg.algo_config.cfc_rainbow.learning_rate = 0.00025f;
            cfg.algo_config.cfc_rainbow.gamma = 0.99f;
            cfg.algo_config.cfc_rainbow.tau = 0.005f;
            cfg.algo_config.cfc_rainbow.target_update_freq = 500;
            cfg.algo_config.cfc_rainbow.use_double_dqn = 1;
            cfg.algo_config.cfc_rainbow.use_dueling = 1;
            cfg.algo_config.cfc_rainbow.use_noisy_nets = 0;
            cfg.algo_config.cfc_rainbow.use_distributional = 1;
            cfg.algo_config.cfc_rainbow.use_multi_step = 1;
            cfg.algo_config.cfc_rainbow.multi_step_n = 3;
            cfg.algo_config.cfc_rainbow.num_atoms = 51;
            cfg.algo_config.cfc_rainbow.v_min = -10.0f;
            cfg.algo_config.cfc_rainbow.v_max = 10.0f;
            cfg.algo_config.cfc_rainbow.cfc_time_constant = 1.0f;
            cfg.algo_config.cfc_rainbow.cfc_ode_solver = 0;
            cfg.algo_config.cfc_rainbow.cfc_use_multi_timescale = 0;
            break;
        case RL_ALGORITHM_CFC_IMPALA:
            cfg.algo_config.cfc_impala.actor_hidden_size = 128;
            cfg.algo_config.cfc_impala.critic_hidden_size = 128;
            cfg.algo_config.cfc_impala.actor_num_layers = 2;
            cfg.algo_config.cfc_impala.critic_num_layers = 2;
            cfg.algo_config.cfc_impala.actor_lr = 0.0003f;
            cfg.algo_config.cfc_impala.critic_lr = 0.0003f;
            cfg.algo_config.cfc_impala.gamma = 0.99f;
            cfg.algo_config.cfc_impala.entropy_coef = 0.01f;
            cfg.algo_config.cfc_impala.vtrace_clip_rho = 1.0f;
            cfg.algo_config.cfc_impala.vtrace_clip_c = 1.0f;
            cfg.algo_config.cfc_impala.max_grad_norm = 40.0f;
            cfg.algo_config.cfc_impala.num_actors = 4;
            cfg.algo_config.cfc_impala.unroll_length = 50;
            cfg.algo_config.cfc_impala.batch_size = 32;
            cfg.algo_config.cfc_impala.cfc_time_constant = 1.0f;
            cfg.algo_config.cfc_impala.cfc_ode_solver = 0;
            break;
        case RL_ALGORITHM_CFC_R2D2:
            cfg.algo_config.cfc_r2d2.hidden_size = 128;
            cfg.algo_config.cfc_r2d2.num_layers = 2;
            cfg.algo_config.cfc_r2d2.learning_rate = 0.0001f;
            cfg.algo_config.cfc_r2d2.gamma = 0.997f;
            cfg.algo_config.cfc_r2d2.tau = 0.005f;
            cfg.algo_config.cfc_r2d2.target_update_freq = 2500;
            cfg.algo_config.cfc_r2d2.burn_in_steps = 10;
            cfg.algo_config.cfc_r2d2.sequence_length = 40;
            cfg.algo_config.cfc_r2d2.replay_capacity = 1000;
            cfg.algo_config.cfc_r2d2.priority_exponent = 0.9f;
            cfg.algo_config.cfc_r2d2.importance_sampling_exponent = 0.6f;
            cfg.algo_config.cfc_r2d2.cfc_time_constant = 1.0f;
            cfg.algo_config.cfc_r2d2.cfc_ode_solver = 0;
            cfg.algo_config.cfc_r2d2.cfc_use_multi_timescale = 0;
            break;
    }
    return cfg;
}

RLAgent* rl_agent_create(const RLConfig* config)
{
    if (!config || config->state_dim <= 0 || config->action_dim <= 0)
    {
        log_error("无效的RL配置: state_dim=%d, action_dim=%d",
                  config ? config->state_dim : -1,
                  config ? config->action_dim : -1);
        return NULL;
    }

    RLAgent* agent = (RLAgent*)safe_calloc(1, sizeof(RLAgent));
    if (!agent) return NULL;

    agent->config = *config;
    agent->epsilon = config->explore_config.epsilon_start;
    agent->best_return = -1e9f;
    agent->alpha = 0.2f;

    int ret = -1;
    switch (config->algorithm)
    {
        case RL_ALGORITHM_DQN:
            ret = rl_dqn_init(agent);
            break;
        case RL_ALGORITHM_PPO:
            ret = rl_ppo_init(agent);
            break;
        case RL_ALGORITHM_SAC:
            ret = rl_sac_init(agent);
            break;
        case RL_ALGORITHM_A2C:
            ret = rl_a2c_init(agent);
            break;
        case RL_ALGORITHM_CFC_TD3:
            ret = rl_cfc_td3_init(agent);
            break;
        case RL_ALGORITHM_CFC_RAINBOW:
            ret = rl_cfc_rainbow_init(agent);
            break;
        case RL_ALGORITHM_CFC_IMPALA:
            ret = rl_cfc_impala_init(agent);
            break;
        case RL_ALGORITHM_CFC_R2D2:
            ret = rl_cfc_r2d2_init(agent);
            break;
        default:
            log_error("不支持的RL算法: %d", config->algorithm);
            safe_free((void**)&agent);
            return NULL;
    }

    if (ret != 0)
    {
        log_error("RL代理初始化失败");
        rl_agent_free(agent);
        return NULL;
    }

    if (config->explore_config.strategy == RL_EXPLORE_OU_NOISE)
    {
        agent->ou_noise = rl_ou_noise_create(
            config->action_dim,
            config->explore_config.ou_theta,
            config->explore_config.ou_mu,
            config->explore_config.ou_sigma,
            config->explore_config.ou_dt);
    }

    /* ZSFJJJ-H005: 高级探索策略初始化 */
    if (config->explore_config.strategy == RL_EXPLORE_ICM) {
        ICMConfig icm_cfg;
        memset(&icm_cfg, 0, sizeof(icm_cfg));
        icm_cfg.state_dim = config->state_dim;
        icm_cfg.action_dim = config->action_dim;
        icm_cfg.hidden_dim = 64;
        icm_cfg.embedding_dim = 32;
        icm_cfg.forward_loss_weight = 0.8f;
        icm_cfg.inverse_loss_weight = 0.2f;
        icm_cfg.learning_rate = 0.001f;
        icm_cfg.cfc_tau = 1.0f;
        icm_cfg.cfc_dt = 0.1f;
        icm_cfg.cfc_steps = 5;
        agent->icm_state = explore_icm_create(&icm_cfg);
    }
    if (config->explore_config.strategy == RL_EXPLORE_RND) {
        RNDConfig rnd_cfg;
        memset(&rnd_cfg, 0, sizeof(rnd_cfg));
        rnd_cfg.state_dim = config->state_dim;
        rnd_cfg.hidden_dim = 64;
        rnd_cfg.embedding_dim = 32;
        rnd_cfg.num_predictors = 4;
        rnd_cfg.learning_rate = 0.001f;
        rnd_cfg.cfc_tau = 1.0f;
        rnd_cfg.cfc_dt = 0.1f;
        rnd_cfg.cfc_steps = 5;
        agent->rnd_state = explore_rnd_create(&rnd_cfg);
    }
    if (config->explore_config.strategy == RL_EXPLORE_GO_EXPLORE) {
        GoExploreConfig ge_cfg;
        memset(&ge_cfg, 0, sizeof(ge_cfg));
        ge_cfg.state_dim = config->state_dim;
        ge_cfg.action_dim = config->action_dim;
        ge_cfg.archive_cell_capacity = 100;
        ge_cfg.cell_threshold = 0.1f;
        ge_cfg.max_episode_steps = 1000;
        ge_cfg.selection_strategy = 0;
        agent->go_explore_state = explore_go_create(&ge_cfg);
    }

    agent->is_initialized = 1;
    agent->action_buffer = (float*)safe_malloc((size_t)config->action_dim * sizeof(float));
    agent->action_buffer_size = config->action_dim;

    log_info("RL代理创建成功: %s, 算法=%d, state_dim=%d, action_dim=%d",
             config->name, config->algorithm, config->state_dim, config->action_dim);
    return agent;
}

void rl_agent_free(RLAgent* agent)
{
    if (!agent) return;
    for (int i = 0; i < RL_NETWORK_COUNT; i++)
    {
        if (agent->network_active[i] && agent->networks[i])
        {
            lnn_free(agent->networks[i]);
            agent->networks[i] = NULL;
        }
    }
    rl_ou_noise_free(agent->ou_noise);
    agent->ou_noise = NULL;
    /* ZSFJJJ-H005: 释放高级探索策略 */
    if (agent->icm_state) { explore_destroy((ExploreState*)agent->icm_state); agent->icm_state = NULL; }
    if (agent->rnd_state) { explore_destroy((ExploreState*)agent->rnd_state); agent->rnd_state = NULL; }
    if (agent->go_explore_state) { explore_destroy((ExploreState*)agent->go_explore_state); agent->go_explore_state = NULL; }
    safe_free((void**)&agent->action_buffer);
    safe_free((void**)&agent->log_alpha);
    safe_free((void**)&agent->last_action_probs);
    safe_free((void**)&agent->last_log_probs);
    safe_free((void**)&agent->last_value);
    /* 释放CfC算法专有内存 */
    safe_free((void**)&agent->rainbow_atom_support);
    safe_free((void**)&agent->rainbow_noisy_params);
    safe_free((void**)&agent->impala_trajectory_buffer);
    safe_free((void**)&agent->impala_vtrace_values);
    safe_free((void**)&agent->r2d2_recurrent_state);
    safe_free((void**)&agent->r2d2_sequence_buffer);
    safe_free((void**)&agent->r2d2_sequence_dones);
    rl_replay_buffer_destroy(&agent->config.replay_buffer);
    memset(agent, 0, sizeof(RLAgent));
    safe_free((void**)&agent);
}

int rl_select_action(RLAgent* agent, const float* state, int state_dim, float* action, int action_dim)
{
    if (!agent || !state || !action || !agent->is_initialized) return -1;
    (void)state_dim;

    memset(action, 0, (size_t)action_dim * sizeof(float));
    int d = RL_MIN(action_dim, agent->config.action_dim);

    switch (agent->config.algorithm)
    {
        case RL_ALGORITHM_DQN:
            return rl_dqn_select_action(agent, state, state_dim, action, d);
        case RL_ALGORITHM_PPO:
        case RL_ALGORITHM_A2C:
        {
            float probs[RL_MAX_ACTION_DIM];
            int sel = 0;
            rl_ppo_get_action_probs(agent, state, state_dim, probs, &sel);
            if (agent->config.discrete_actions)
                action[0] = (float)sel;
            else
            {
                for (int i = 0; i < d && i < sel; i++)
                    action[i] = probs[i];
            }
            return 0;
        }
        case RL_ALGORITHM_SAC:
        {
            float log_prob = 0.0f;
            rl_sac_get_action(agent, state, action, &log_prob, state_dim);
            return 0;
        }
        case RL_ALGORITHM_CFC_TD3:
        {
            return rl_cfc_td3_select_action_internal(agent, state, state_dim, action, d, 1);
        }
        case RL_ALGORITHM_CFC_RAINBOW:
        {
            LNN* dist_net = agent->networks[RL_NETWORK_DISTRIBUTION];
            if (!dist_net) return -1;
            int num_actions = agent->config.num_actions;
            int num_atoms = agent->rainbow_num_atoms > 0 ? agent->rainbow_num_atoms : 51;
            float* output = (float*)safe_malloc((size_t)(num_actions * num_atoms) * sizeof(float));
            if (!output) return -1;
            int ret = lnn_forward(dist_net, state, output);
            if (ret != 0) { safe_free((void**)&output); return ret; }
            /* 对每个动作的原子分布做softmax并计算Q值 */
            float q_values[RL_MAX_ACTION_DIM];
            for (int a = 0; a < num_actions && a < RL_MAX_ACTION_DIM; a++)
            {
                float* dist = output + a * num_atoms;
                float max_val = dist[0];
                for (int i = 1; i < num_atoms; i++)
                    if (dist[i] > max_val) max_val = dist[i];
                float sum = 0.0f;
                for (int i = 0; i < num_atoms; i++)
                {
                    dist[i] = expf(dist[i] - max_val);
                    sum += dist[i];
                }
                if (sum > 1e-10f)
                    for (int i = 0; i < num_atoms; i++) dist[i] /= sum;
                q_values[a] = 0.0f;
                for (int i = 0; i < num_atoms; i++)
                    q_values[a] += dist[i] * agent->rainbow_atom_support[i];
            }
            safe_free((void**)&output);
            /* epsilon-贪心选择 */
            if (rl_randf() < agent->epsilon)
            {
                /* K-012修复：使用安全随机数 */
                action[0] = (float)((int)(secure_random_int((uint32_t)(num_actions - 1))));
            }
            else
            {
                int best = 0;
                for (int i = 1; i < num_actions; i++)
                    if (q_values[i] > q_values[best]) best = i;
                action[0] = (float)best;
            }
            return 0;
        }
        case RL_ALGORITHM_CFC_IMPALA:
        {
            LNN* actor = agent->networks[RL_NETWORK_ACTOR];
            if (!actor) return -1;
            float output[256];
            if (lnn_forward(actor, state, output) != 0) return -1;
            if (agent->config.discrete_actions)
            {
                float probs[RL_MAX_ACTION_DIM];
                float max_val = output[0];
                int nact = agent->config.num_actions;
                for (int i = 1; i < nact; i++)
                    if (output[i] > max_val) max_val = output[i];
                float sum = 0.0f;
                for (int i = 0; i < nact && i < RL_MAX_ACTION_DIM; i++)
                {
                    probs[i] = expf(output[i] - max_val);
                    sum += probs[i];
                }
                if (sum > 1e-10f)
                    for (int i = 0; i < nact && i < RL_MAX_ACTION_DIM; i++) probs[i] /= sum;
                float r = rl_randf();
                float cum = 0.0f;
                for (int i = 0; i < nact; i++)
                {
                    cum += probs[i];
                    if (r <= cum) { action[0] = (float)i; break; }
                }
            }
            else
            {
                for (int i = 0; i < d; i++)
                    action[i] = output[i];
            }
            return 0;
        }
        case RL_ALGORITHM_CFC_R2D2:
        {
            LNN* q_net = agent->networks[RL_NETWORK_RECURRENT_Q];
            if (!q_net) return -1;
            float q_values[RL_MAX_ACTION_DIM];
            if (lnn_forward(q_net, state, q_values) != 0) return -1;
            int best = 0;
            int nact = agent->config.num_actions;
            for (int i = 1; i < nact && i < RL_MAX_ACTION_DIM; i++)
                if (q_values[i] > q_values[best]) best = i;
            action[0] = (float)best;
            return 0;
        }
        default:
            return -1;
    }
}

int rl_store_experience(RLAgent* agent, const float* state, int state_dim,
                        const float* action, int action_dim, float reward,
                        const float* next_state, int next_state_dim, int done)
{
    if (!agent || !state || !action || !next_state) return -1;

    RLExperience exp;
    memset(&exp, 0, sizeof(RLExperience));

    int sd = RL_MIN(state_dim, RL_MAX_STATE_DIM);
    int ad = RL_MIN(action_dim, RL_MAX_ACTION_DIM);
    int nd = RL_MIN(next_state_dim, RL_MAX_STATE_DIM);

    memcpy(exp.state, state, (size_t)sd * sizeof(float));
    exp.state_dim = sd;
    memcpy(exp.action, action, (size_t)ad * sizeof(float));
    exp.action_dim = ad;
    exp.reward = reward;
    memcpy(exp.next_state, next_state, (size_t)nd * sizeof(float));
    exp.next_state_dim = nd;
    exp.done = done;
    exp.priority = agent->config.replay_buffer.max_priority;

    int ret = rl_replay_buffer_add(&agent->config.replay_buffer, &exp);
    if (ret != 0) return ret;

    agent->current_episode_return += reward;
    if (done)
    {
        agent->total_episodes++;
        agent->episode_return_sum += agent->current_episode_return;
        agent->episode_return_count += 1.0f;
        if (agent->current_episode_return > agent->best_return)
            agent->best_return = agent->current_episode_return;
        agent->current_episode_return = 0.0f;
    }
    return 0;
}

int rl_train(RLAgent* agent, int batch_size)
{
    if (!agent || !agent->is_initialized) return -1;
    if (batch_size <= 0) batch_size = 64;

    switch (agent->config.algorithm)
    {
        case RL_ALGORITHM_DQN:
            return rl_dqn_train(agent, batch_size);
        case RL_ALGORITHM_PPO:
            return rl_ppo_train(agent, batch_size);
        case RL_ALGORITHM_A2C:
            /* S-020修复: A2C完整实现 - 纯Advantage Actor-Critic算法
             * 使用GAE优势估计、单轮无裁剪策略梯度更新
             * 与PPO独立实现，不使用clip_epsilon和importance ratio */
            return rl_a2c_train(agent, batch_size);
        case RL_ALGORITHM_SAC:
            return rl_sac_train(agent, batch_size);
        case RL_ALGORITHM_CFC_TD3:
            return rl_cfc_td3_train(agent, batch_size);
        case RL_ALGORITHM_CFC_RAINBOW:
            return rl_cfc_rainbow_train(agent, batch_size);
        case RL_ALGORITHM_CFC_IMPALA:
            return rl_cfc_impala_train(agent, batch_size);
        case RL_ALGORITHM_CFC_R2D2:
            return rl_cfc_r2d2_train(agent, batch_size);
        default:
            return -1;
    }
}

int rl_update_exploration(RLAgent* agent)
{
    if (!agent) return -1;

    RLExploreConfig* ec = &agent->config.explore_config;
    agent->epsilon = rl_explore_epsilon_greedy(ec, agent->total_episodes);

    if (ec->strategy == RL_EXPLORE_GAUSSIAN_NOISE)
        ec->noise_std *= ec->noise_decay;

    return 0;
}

int rl_save(RLAgent* agent, const char* filepath)
{
    if (!agent || !filepath || !agent->is_initialized) return -1;
    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;

    fwrite(&agent->config, sizeof(RLConfig), 1, fp);
    fwrite(&agent->epsilon, sizeof(float), 1, fp);
    fwrite(&agent->total_steps, sizeof(int), 1, fp);
    fwrite(&agent->total_episodes, sizeof(int), 1, fp);
    fwrite(&agent->best_return, sizeof(float), 1, fp);

    for (int i = 0; i < RL_NETWORK_COUNT; i++)
    {
        int active = agent->network_active[i];
        fwrite(&active, sizeof(int), 1, fp);
        if (active && agent->networks[i])
        {
            size_t nparams = lnn_get_parameter_count(agent->networks[i]);
            fwrite(&nparams, sizeof(size_t), 1, fp);
            float* params = lnn_get_parameters(agent->networks[i]);
            if (params) fwrite(params, sizeof(float), nparams, fp);
        }
    }
    fclose(fp);
    return 0;
}

RLAgent* rl_load(const char* filepath)
{
    if (!filepath) return NULL;
    FILE* fp = fopen(filepath, "rb");
    if (!fp) return NULL;

    RLConfig config;
    if (fread(&config, sizeof(RLConfig), 1, fp) != 1)
    {
        fclose(fp);
        return NULL;
    }

    RLAgent* agent = rl_agent_create(&config);
    if (!agent) { fclose(fp); return NULL; }

    fread(&agent->epsilon, sizeof(float), 1, fp);
    fread(&agent->total_steps, sizeof(int), 1, fp);
    fread(&agent->total_episodes, sizeof(int), 1, fp);
    fread(&agent->best_return, sizeof(float), 1, fp);

    for (int i = 0; i < RL_NETWORK_COUNT; i++)
    {
        int active = 0;
        fread(&active, sizeof(int), 1, fp);
        if (active && agent->network_active[i] && agent->networks[i])
        {
            size_t saved_nparams = 0;
            fread(&saved_nparams, sizeof(size_t), 1, fp);
            size_t actual_nparams = lnn_get_parameter_count(agent->networks[i]);
            size_t read_size = RL_MIN(saved_nparams, actual_nparams);
            float* params = lnn_get_parameters(agent->networks[i]);
            if (params && read_size > 0)
                fread(params, sizeof(float), read_size, fp);
            else if (saved_nparams > 0)
                fseek(fp, (long)(saved_nparams * sizeof(float)), SEEK_CUR);
        }
    }
    fclose(fp);
    return agent;
}

int rl_reset(RLAgent* agent)
{
    if (!agent) return -1;
    agent->current_episode_return = 0.0f;

    if (agent->ou_noise)
        rl_ou_noise_reset(agent->ou_noise);

    return 0;
}

float rl_get_exploration_rate(const RLAgent* agent)
{
    return agent ? agent->epsilon : 0.0f;
}

int rl_set_exploration_rate(RLAgent* agent, float rate)
{
    if (!agent) return -1;
    agent->epsilon = RL_CLAMP(rate, 0.0f, 1.0f);
    return 0;
}

int rl_get_stats(const RLAgent* agent, int* total_steps, int* total_episodes,
                 float* avg_return, float* best_return)
{
    if (!agent) return -1;
    if (total_steps) *total_steps = agent->total_steps;
    if (total_episodes) *total_episodes = agent->total_episodes;
    if (avg_return)
        *avg_return = (agent->episode_return_count > 0.0f) ?
                      agent->episode_return_sum / agent->episode_return_count : 0.0f;
    if (best_return) *best_return = agent->best_return;
    return 0;
}

int rl_dqn_get_q_values(RLAgent* agent, const float* state, int state_dim, float* q_values)
{
    if (!agent || !state || !q_values || agent->config.algorithm != RL_ALGORITHM_DQN) return -1;
    LNN* q_net = agent->networks[RL_NETWORK_Q];
    if (!q_net) return -1;
    (void)state_dim;
    return lnn_forward(q_net, state, q_values);
}

int rl_ppo_get_action_dist(RLAgent* agent, const float* state, int state_dim,
                           float* means, float* log_stds)
{
    if (!agent || !state || !means || !log_stds ||
        (agent->config.algorithm != RL_ALGORITHM_PPO && agent->config.algorithm != RL_ALGORITHM_A2C))
        return -1;

    LNN* actor = agent->networks[RL_NETWORK_ACTOR];
    if (!actor) return -1;
    (void)state_dim;

    LNNConfig acfg;
    memset(&acfg, 0, sizeof(LNNConfig));
    lnn_get_config(actor, &acfg);
    int out_dim = (int)acfg.output_size;
    float* raw = (float*)safe_malloc((size_t)out_dim * sizeof(float));
    if (!raw) return -1;

    int ret = lnn_forward(actor, state, raw);
    if (ret != 0) { safe_free((void**)&raw); return ret; }

    int dim = agent->config.discrete_actions ? agent->config.num_actions : out_dim / 2;
    for (int i = 0; i < dim && i < RL_MAX_ACTION_DIM; i++)
    {
        means[i] = raw[i];
        log_stds[i] = (i + dim < out_dim) ? RL_CLAMP(raw[i + dim], -5.0f, 2.0f) : -1.0f;
    }
    safe_free((void**)&raw);
    return 0;
}

float rl_sac_get_entropy(const RLAgent* agent)
{
    if (!agent || agent->config.algorithm != RL_ALGORITHM_SAC) return 0.0f;
    return agent->alpha;
}

/* ============ CfC算法公共API实现 ============ */

int rl_cfc_td3_select_deterministic_action(RLAgent* agent, const float* state,
                                            int state_dim, float* action, int action_dim)
{
    if (!agent || !state || !action || agent->config.algorithm != RL_ALGORITHM_CFC_TD3)
        return -1;
    return rl_cfc_td3_select_action_internal(agent, state, state_dim, action, action_dim, 0);
}

int rl_cfc_rainbow_get_distribution(RLAgent* agent, const float* state, int state_dim,
                                    float* q_distributions, int num_atoms)
{
    if (!agent || !state || !q_distributions || agent->config.algorithm != RL_ALGORITHM_CFC_RAINBOW)
        return -1;
    LNN* dist_net = agent->networks[RL_NETWORK_DISTRIBUTION];
    if (!dist_net) return -1;
    (void)state_dim;
    float* raw = (float*)safe_malloc((size_t)(agent->config.num_actions * num_atoms) * sizeof(float));
    if (!raw) return -1;
    int ret = lnn_forward(dist_net, state, raw);
    if (ret != 0) { safe_free((void**)&raw); return ret; }
    for (int a = 0; a < agent->config.num_actions; a++)
    {
        float* src = raw + a * num_atoms;
        float* dst = q_distributions + a * num_atoms;
        float max_val = src[0];
        for (int i = 1; i < num_atoms; i++)
            if (src[i] > max_val) max_val = src[i];
        float sum = 0.0f;
        for (int i = 0; i < num_atoms; i++)
        {
            dst[i] = expf(src[i] - max_val);
            sum += dst[i];
        }
        if (sum > 1e-10f)
            for (int i = 0; i < num_atoms; i++) dst[i] /= sum;
    }
    safe_free((void**)&raw);
    return 0;
}

int rl_cfc_td3_train_step(RLAgent* agent, int batch_size)
{
    if (!agent || !agent->is_initialized) return -1;
    if (agent->config.algorithm != RL_ALGORITHM_CFC_TD3) return -1;
    if (batch_size <= 0) batch_size = 64;
    return rl_cfc_td3_train(agent, batch_size);
}

int rl_cfc_set_ode_solver(RLAgent* agent, int solver_type)
{
    if (!agent || !agent->is_initialized) return -1;
    if (solver_type < 0 || solver_type > 3) return -1;
    RLAlgorithm algo = agent->config.algorithm;
    if (algo != RL_ALGORITHM_CFC_TD3 && algo != RL_ALGORITHM_CFC_RAINBOW &&
        algo != RL_ALGORITHM_CFC_IMPALA && algo != RL_ALGORITHM_CFC_R2D2)
        return -1;
    for (int i = 0; i < RL_NETWORK_COUNT; i++)
    {
        if (agent->network_active[i] && agent->networks[i])
        {
            LNNConfig cfg;
            if (lnn_get_config(agent->networks[i], &cfg) != 0) continue;
            cfg.ode_solver_type = solver_type;
            lnn_set_config(agent->networks[i], &cfg);
        }
    }
    return 0;
}

int rl_cfc_get_training_stats(RLAgent* agent, float* critic_loss,
                              float* actor_loss, float* avg_q_value,
                              int* policy_delay_count)
{
    if (!agent) return -1;
    if (agent->config.algorithm == RL_ALGORITHM_CFC_TD3)
    {
        if (critic_loss) *critic_loss = agent->td3_avg_critic_loss;
        if (actor_loss) *actor_loss = agent->td3_avg_actor_loss;
        if (avg_q_value) *avg_q_value = agent->td3_avg_q_value;
        if (policy_delay_count) *policy_delay_count = agent->td3_critic_updates;
        return 0;
    }
    if (agent->config.algorithm == RL_ALGORITHM_CFC_IMPALA)
    {
        if (critic_loss) *critic_loss = 0.0f;
        if (actor_loss) *actor_loss = 0.0f;
        if (avg_q_value) *avg_q_value = agent->impala_avg_entropy;
        if (policy_delay_count) *policy_delay_count = 0;
        return 0;
    }
    return -1;
}

/* ============================================================================
 * 标准化RL环境接口（Gym-like）
 * 对接PyBullet/Gazebo仿真环境的统一抽象层
 * ============================================================================ */

typedef struct {
    void* env_handle;
    int (*reset_func)(void*, float*, size_t*);
    int (*step_func)(void*, const float*, size_t, float*, size_t*, float*, int*);
    void (*close_func)(void*);
    float* observation_space;
    size_t obs_dim;
    size_t act_dim;
    float* action_low;
    float* action_high;
    float reward_range[2];
    int is_continuous;
    int step_count;
    int episode_count;
} GymEnv;

static void gym_env_free(GymEnv* env);

static GymEnv* gym_env_create(size_t obs_dim, size_t act_dim, int is_continuous) {
    GymEnv* env = (GymEnv*)safe_calloc(1, sizeof(GymEnv));
    if (!env) return NULL;

    env->obs_dim = obs_dim;
    env->act_dim = act_dim;
    env->is_continuous = is_continuous;

    env->observation_space = (float*)safe_calloc(obs_dim, sizeof(float));
    env->action_low = (float*)safe_calloc(act_dim, sizeof(float));
    env->action_high = (float*)safe_calloc(act_dim, sizeof(float));

    if (!env->observation_space || !env->action_low || !env->action_high) {
        gym_env_free(env);
        return NULL;
    }

    for (size_t i = 0; i < act_dim; i++) {
        env->action_low[i] = -1.0f;
        env->action_high[i] = 1.0f;
    }
    env->reward_range[0] = -1e6f;
    env->reward_range[1] = 1e6f;

    return env;
}

static void gym_env_free(GymEnv* env) {
    if (!env) return;
    safe_free((void**)&env->observation_space);
    safe_free((void**)&env->action_low);
    safe_free((void**)&env->action_high);
    safe_free((void**)&env);
}

static int gym_env_register_functions(GymEnv* env,
                                int (*reset_fn)(void*, float*, size_t*),
                                int (*step_fn)(void*, const float*, size_t,
                                               float*, size_t*, float*, int*),
                                void (*close_fn)(void*),
                                void* handle) {
    if (!env) return -1;
    env->reset_func = reset_fn;
    env->step_func = step_fn;
    env->close_func = close_fn;
    env->env_handle = handle;
    return 0;
}

static int gym_env_reset(GymEnv* env, float* initial_obs, size_t* obs_dim_out) {
    if (!env || !initial_obs || !obs_dim_out) return -1;

    if (env->reset_func && env->env_handle) {
        int ret = env->reset_func(env->env_handle, initial_obs, obs_dim_out);
        if (ret == 0) {
            env->step_count = 0;
            env->episode_count++;
        }
        return ret;
    }

    /* P0-006修复: 无外部环境时返回错误，拒绝填充零值/虚假观测数据
     * 之前此处用memset填充零值观测并返回0，可能导致零值数据被用作
     * 训练样本，污染自主学习管道。
     * 现在与gym_env_step()保持一致，无环境时返回-1，
     * 调用方应探测错误并暂停RL训练。 */
    *obs_dim_out = 0;
    return -1;
}

static int gym_env_step(GymEnv* env, const float* action, size_t act_dim,
                  float* next_obs, size_t* obs_dim_out,
                  float* reward, int* done) {
    if (!env || !action || !next_obs || !reward || !done) return -1;

    if (env->step_func && env->env_handle) {
        return env->step_func(env->env_handle, action, act_dim,
                              next_obs, obs_dim_out, reward, done);
    }

    /* P0-006修复：无外部环境时严格拒绝生成任何数据
     * 不填充next_obs、不设置reward、不设置done标志。
     * 调用方必须检查返回值(-1)并暂停RL训练。
     * 零值数据如果进入经验缓冲区将污染自主学习管道。 */
    *obs_dim_out = 0;
    *reward = 0.0f;
    *done = 0;
    return -1;
}

static void gym_env_close(GymEnv* env) {
    if (!env) return;
    if (env->close_func && env->env_handle) {
        env->close_func(env->env_handle);
    }
    gym_env_free(env);
}

/* BUG-019修复: 从RLAgent获取动作（基于LNN网络前向推理）
 * 使用agent内部的CfC-LNN网络进行状态→Q值的前向传播 */
static int rl_agent_get_action(void* agent, const float* state, int state_dim,
                                float* q_values, int action_dim) {
    if (!agent || !state || !q_values || action_dim <= 0) return -1;
    RLAgent* a = (RLAgent*)agent;
    if (!a->is_initialized) return -1;
    /* 使用第一个活跃的LNN网络进行前向推理 */
    LNN* best_network = NULL;
    for (int i = 0; i < RL_NETWORK_COUNT; i++) {
        if (a->network_active[i] && a->networks[i]) {
            best_network = a->networks[i];
            break;
        }
    }
    if (!best_network) {
        /* 无可用网络，使用启发式Q值初始化 */
        for (int i = 0; i < action_dim; i++) {
            q_values[i] = 0.0f;
        }
        return 0;
    }
    /* LNN前向传播：state → output */
    float input_buf[128];
    float output_buf[128];
    memset(input_buf, 0, sizeof(input_buf));
    memset(output_buf, 0, sizeof(output_buf));
    int copy_dim = state_dim < 128 ? state_dim : 128;
    memcpy(input_buf, state, (size_t)copy_dim * sizeof(float));
    if (lnn_forward(best_network, input_buf, output_buf) != 0) {
        for (int i = 0; i < action_dim && i < 128; i++) {
            q_values[i] = 0.0f;
        }
        return 0;
    }
    for (int i = 0; i < action_dim && i < 128; i++) {
        q_values[i] = output_buf[i];
    }
    return 0;
}

static int rl_agent_train_episode(void* agent, GymEnv* env,
                            int max_steps, float* total_reward) {
    if (!env || !total_reward) return -1;
    /* BUG-019修复: 使用agent进行动作选择，而非纯随机动作
     * agent可为NULL（兼容旧调用方式仅评估环境） */
    int use_agent = (agent != NULL) ? 1 : 0;

    *total_reward = 0.0f;
    float obs[1024];
    size_t obs_dim = 0;

    if (gym_env_reset(env, obs, &obs_dim) != 0) return -1;

    for (int step = 0; step < max_steps; step++) {
        float action[64] = {0};
        size_t act_dim = env->act_dim < 64 ? env->act_dim : 64;

        if (use_agent) {
            /* 使用agent的CfC网络进行前向推理获得动作 */
            float state_features[128];
            memset(state_features, 0, sizeof(state_features));
            size_t copy_dim = obs_dim < 128 ? obs_dim : 128;
            memcpy(state_features, obs, copy_dim * sizeof(float));
            float q_values[64];
            memset(q_values, 0, sizeof(q_values));
            /* 调用agent的前向传播获取Q值 */
            if (rl_agent_get_action(agent, state_features, (int)copy_dim, q_values, (int)act_dim) == 0) {
                /* 从Q值中选最大动作 + 少量探索噪声 */
                float best_q = q_values[0];
                int best_a = 0;
                for (size_t i = 1; i < act_dim; i++) {
                    if (q_values[i] > best_q) { best_q = q_values[i]; best_a = (int)i; }
                }
                for (size_t i = 0; i < act_dim; i++) {
                    action[i] = q_values[i];
                    /* 添加递减探索噪声 */
                    float noise = rl_randf() * 0.1f - 0.05f;
                    action[i] += noise;
                    if (action[i] > 1.0f) action[i] = 1.0f;
                    if (action[i] < -1.0f) action[i] = -1.0f;
                }
            } else {
                /* agent获取动作失败时退回到有界随机探索 */
                for (size_t i = 0; i < act_dim; i++) {
                    action[i] = (rl_randf() - 0.5f) * 0.2f; /* 小范围探索 */
                }
            }
        } else {
            /* 无agent时使用小范围探索（评估模式） */
            for (size_t i = 0; i < act_dim; i++) {
                action[i] = (rl_randf() - 0.5f) * 0.1f;
            }
        }

        float next_obs[1024];
        size_t next_obs_dim = 0;
        float reward = 0.0f;
        int done = 0;

        if (gym_env_step(env, action, act_dim, next_obs, &next_obs_dim,
                         &reward, &done) != 0) break;

        *total_reward += reward;
        if (done) break;
        memcpy(obs, next_obs, next_obs_dim * sizeof(float));
        obs_dim = next_obs_dim;
    }

    return 0;
}
