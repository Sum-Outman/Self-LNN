/**
 * @file reinforcement_learning.h
 * @brief 强化学习算法接口
 *
 * 提供完整的强化学习算法实现，包括：
 * - DQN (Deep Q-Network) 深度Q网络
 * - PPO (Proximal Policy Optimization) 近端策略优化
 * - SAC (Soft Actor-Critic) 软演员-评论家
 * - A2C (Advantage Actor-Critic) 优势演员-评论家
 * - 优先经验回放 (Prioritized Experience Replay)
 * - 多种探索策略
 */

#ifndef SELFLNN_REINFORCEMENT_LEARNING_H
#define SELFLNN_REINFORCEMENT_LEARNING_H

#include <stddef.h>
#include "selflnn/core/lnn.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RL_MAX_ACTION_DIM 64
#define RL_MAX_STATE_DIM 512
#define RL_MAX_BATCH_SIZE 4096  /* 支持大规模多机器人并行训练，不足时动态分配 */
#define RL_NAME_MAX 128

/**
 * @brief 强化学习算法类型
 */
typedef enum {
    RL_ALGORITHM_DQN = 0,
    RL_ALGORITHM_PPO = 1,
    RL_ALGORITHM_SAC = 2,
    RL_ALGORITHM_A2C = 3,
    /* 基于CfC的强化学习算法 */
    RL_ALGORITHM_CFC_TD3 = 10,
    RL_ALGORITHM_CFC_RAINBOW = 11,
    RL_ALGORITHM_CFC_IMPALA = 12,
    RL_ALGORITHM_CFC_R2D2 = 13
} RLAlgorithm;

/**
 * @brief 探索策略类型
 */
typedef enum {
    RL_EXPLORE_EPSILON_GREEDY = 0,
    RL_EXPLORE_UCB = 1,
    RL_EXPLORE_GAUSSIAN_NOISE = 2,
    RL_EXPLORE_OU_NOISE = 3,
    RL_EXPLORE_BOLTZMANN = 4
} RLExploreStrategy;

/**
 * @brief 经验回放类型
 */
typedef enum {
    RL_REPLAY_UNIFORM = 0,
    RL_REPLAY_PRIORITIZED = 1
} RLReplayType;

/**
 * @brief 单步经验结构体
 */
typedef struct {
    float state[RL_MAX_STATE_DIM];
    int state_dim;
    float action[RL_MAX_ACTION_DIM];
    int action_dim;
    float reward;
    float next_state[RL_MAX_STATE_DIM];
    int next_state_dim;
    int done;
    float priority;
} RLExperience;

/**
 * @brief 经验回放缓冲区
 */
typedef struct {
    RLExperience* buffer;
    int capacity;
    int size;
    int head;
    RLReplayType replay_type;
    float alpha;
    float beta;
    float beta_increment;
    float* priorities;
    float max_priority;
    int state_dim;
    int action_dim;
} RLReplayBuffer;

/**
 * @brief 探索配置
 */
typedef struct {
    RLExploreStrategy strategy;
    float epsilon_start;
    float epsilon_end;
    float epsilon_decay;
    float noise_std;
    float noise_decay;
    float temperature;
    float ucb_constant;
    int ou_theta;
    float ou_mu;
    float ou_sigma;
    float ou_dt;
} RLExploreConfig;

/**
 * @brief DQN 配置
 */
typedef struct {
    int hidden_size;
    int num_layers;
    float learning_rate;
    float gamma;
    float tau;
    int target_update_freq;
    int use_double_dqn;
    int use_dueling;
    int use_noisy_nets;
    LNNConfig lnn_config;
} RLConfigDQN;

/**
 * @brief PPO 配置
 */
typedef struct {
    int actor_hidden_size;
    int critic_hidden_size;
    int actor_num_layers;
    int critic_num_layers;
    float actor_lr;
    float critic_lr;
    float gamma;
    float gae_lambda;
    float clip_epsilon;
    float entropy_coef;
    float value_coef;
    float max_grad_norm;
    int update_epochs;
    int mini_batch_size;
    LNNConfig actor_lnn_config;
    LNNConfig critic_lnn_config;
} RLConfigPPO;

/**
 * @brief SAC 配置
 */
typedef struct {
    int hidden_size;
    int num_layers;
    float actor_lr;
    float critic_lr;
    float alpha_lr;
    float gamma;
    float tau;
    float init_alpha;
    int automatic_entropy_tuning;
    float target_entropy;
    LNNConfig actor_lnn_config;
    LNNConfig critic_lnn_config;
    LNNConfig value_lnn_config;
} RLConfigSAC;

/**
 * @brief CfC-TD3 配置
 * 使用CfC网络的双延迟深度确定性策略梯度
 */
typedef struct {
    int actor_hidden_size;             /**< 演员网络隐藏层大小 */
    int critic_hidden_size;            /**< 评论家网络隐藏层大小 */
    int actor_num_layers;              /**< 演员网络层数 */
    int critic_num_layers;             /**< 评论家网络层数 */
    float actor_lr;                    /**< 演员学习率 */
    float critic_lr;                   /**< 评论家学习率 */
    float gamma;                       /**< 折扣因子 */
    float tau;                         /**< 目标网络软更新系数 */
    float policy_noise;                /**< 目标策略平滑噪声标准差 */
    float noise_clip;                  /**< 噪声裁剪范围 */
    int policy_delay;                  /**< 策略更新延迟步数（默认2） */
    float cfc_time_constant;           /**< CfC时间常数 */
    int cfc_ode_solver;                /**< CfC ODE求解器类型 */
    int cfc_use_multi_timescale;       /**< 是否启用多时间尺度 */
    LNNConfig actor_lnn_config;        /**< 演员网络LNN配置 */
    LNNConfig critic_lnn_config;       /**< 评论家网络LNN配置 */
} RLConfigCfCTD3;

/**
 * @brief CfC-Rainbow DQN 配置
 * 结合Rainbow DQN七项改进的CfC深度Q网络
 */
typedef struct {
    int hidden_size;                   /**< 隐藏层大小 */
    int num_layers;                    /**< 网络层数 */
    float learning_rate;               /**< 学习率 */
    float gamma;                       /**< 折扣因子 */
    float tau;                         /**< 目标网络软更新系数 */
    int target_update_freq;            /**< 目标网络硬更新频率 */
    int use_double_dqn;                /**< 使用Double DQN */
    int use_dueling;                   /**< 使用Dueling架构 */
    int use_noisy_nets;                /**< 使用Noisy Nets */
    int use_distributional;            /**< 使用分布强化学习（C51） */
    int use_multi_step;                /**< 使用多步学习 */
    int multi_step_n;                  /**< 多步步数 */
    int num_atoms;                     /**< 分布原子数（C51默认51） */
    float v_min;                       /**< 分布值范围最小值 */
    float v_max;                       /**< 分布值范围最大值 */
    float cfc_time_constant;           /**< CfC时间常数 */
    int cfc_ode_solver;                /**< CfC ODE求解器类型 */
    int cfc_use_multi_timescale;       /**< 是否启用多时间尺度 */
    LNNConfig lnn_config;              /**< LNN配置 */
} RLConfigCfCRainbow;

/**
 * @brief CfC-IMPALA 配置
 * 基于CfC的重要性加权演员-学习器架构
 */
typedef struct {
    int actor_hidden_size;             /**< 演员网络隐藏层大小 */
    int critic_hidden_size;            /**< 评论家网络隐藏层大小 */
    int actor_num_layers;              /**< 演员网络层数 */
    int critic_num_layers;             /**< 评论家网络层数 */
    float actor_lr;                    /**< 演员学习率 */
    float critic_lr;                   /**< 评论家学习率 */
    float gamma;                       /**< 折扣因子 */
    float entropy_coef;                /**< 熵正则化系数 */
    float vtrace_clip_rho;             /**< V-trace重要性采样裁剪上限（默认1.0） */
    float vtrace_clip_c;               /**< V-trace重要性采样裁剪上限c（默认1.0） */
    float max_grad_norm;               /**< 最大梯度范数 */
    int num_actors;                    /**< 并行演员数 */
    int unroll_length;                 /**< 展开长度 */
    int batch_size;                    /**< 批大小 */
    float cfc_time_constant;           /**< CfC时间常数 */
    int cfc_ode_solver;                /**< CfC ODE求解器类型 */
    LNNConfig actor_lnn_config;        /**< 演员网络LNN配置 */
    LNNConfig critic_lnn_config;       /**< 评论家网络LNN配置 */
} RLConfigCfCIMPALA;

/**
 * @brief CfC-R2D2 配置
 * 基于CfC的循环重放分布式DQN
 */
typedef struct {
    int hidden_size;                   /**< 隐藏层大小 */
    int num_layers;                    /**< 网络层数 */
    float learning_rate;               /**< 学习率 */
    float gamma;                       /**< 折扣因子 */
    float tau;                         /**< 目标网络软更新系数 */
    int target_update_freq;            /**< 目标网络更新频率 */
    int burn_in_steps;                 /**< 预热步数（用于初始化循环状态） */
    int sequence_length;               /**< 训练序列长度 */
    int replay_capacity;               /**< 回放缓冲区容量（序列数） */
    float priority_exponent;           /**< 优先级指数 */
    float importance_sampling_exponent;/**< 重要性采样指数 */
    float cfc_time_constant;           /**< CfC时间常数 */
    int cfc_ode_solver;                /**< CfC ODE求解器类型 */
    int cfc_use_multi_timescale;       /**< 是否启用多时间尺度 */
    LNNConfig lnn_config;              /**< LNN配置 */
} RLConfigCfCR2D2;

/**
 * @brief 强化学习代理配置
 */
typedef struct {
    RLAlgorithm algorithm;
    int state_dim;
    int action_dim;
    int discrete_actions;
    int num_actions;
    float action_low[RL_MAX_ACTION_DIM];
    float action_high[RL_MAX_ACTION_DIM];
    RLReplayBuffer replay_buffer;
    RLExploreConfig explore_config;
    union {
        RLConfigDQN dqn;
        RLConfigPPO ppo;
        RLConfigSAC sac;
        RLConfigCfCTD3 cfc_td3;
        RLConfigCfCRainbow cfc_rainbow;
        RLConfigCfCIMPALA cfc_impala;
        RLConfigCfCR2D2 cfc_r2d2;
    } algo_config;
    char name[RL_NAME_MAX];
    int use_gpu;
    int seed;
    int verbose;
} RLConfig;

/**
 * @brief 强化学习代理句柄
 */
typedef struct RLAgent RLAgent;

/**
 * @brief 创建强化学习代理
 *
 * @param config 强化学习配置
 * @return RLAgent* 代理句柄，失败返回NULL
 */
RLAgent* rl_agent_create(const RLConfig* config);

/**
 * @brief 释放强化学习代理
 *
 * @param agent 代理句柄
 */
void rl_agent_free(RLAgent* agent);

/**
 * @brief 根据当前状态选择动作
 *
 * @param agent 代理句柄
 * @param state 当前状态向量
 * @param state_dim 状态维度
 * @param action 动作输出缓冲区
 * @param action_dim 动作维度
 * @return int 成功返回0，失败返回-1
 */
int rl_select_action(RLAgent* agent, const float* state, int state_dim, float* action, int action_dim);

/**
 * @brief 存储经验到回放缓冲区
 *
 * @param agent 代理句柄
 * @param state 状态
 * @param state_dim 状态维度
 * @param action 动作
 * @param action_dim 动作维度
 * @param reward 奖励
 * @param next_state 下一状态
 * @param next_state_dim 下一状态维度
 * @param done 是否终止
 * @return int 成功返回0，失败返回-1
 */
int rl_store_experience(RLAgent* agent, const float* state, int state_dim,
                        const float* action, int action_dim, float reward,
                        const float* next_state, int next_state_dim, int done);

/**
 * @brief 训练代理（从回放缓冲区采样并更新）
 *
 * @param agent 代理句柄
 * @param batch_size 批大小
 * @return int 成功返回0，失败返回-1
 */
int rl_train(RLAgent* agent, int batch_size);

/**
 * @brief 更新探索参数（通常每个episode调用一次）
 *
 * @param agent 代理句柄
 * @return int 成功返回0，失败返回-1
 */
int rl_update_exploration(RLAgent* agent);

/**
 * @brief 保存代理模型到文件
 *
 * @param agent 代理句柄
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int rl_save(RLAgent* agent, const char* filepath);

/**
 * @brief 从文件加载代理模型
 *
 * @param filepath 文件路径
 * @return RLAgent* 代理句柄，失败返回NULL
 */
RLAgent* rl_load(const char* filepath);

/**
 * @brief 重置代理内部状态（用于新episode开始）
 *
 * @param agent 代理句柄
 * @return int 成功返回0，失败返回-1
 */
int rl_reset(RLAgent* agent);

/**
 * @brief 获取代理当前探索率
 *
 * @param agent 代理句柄
 * @return float 当前探索率
 */
float rl_get_exploration_rate(const RLAgent* agent);

/**
 * @brief 设置代理探索率
 *
 * @param agent 代理句柄
 * @param rate 探索率
 * @return int 成功返回0，失败返回-1
 */
int rl_set_exploration_rate(RLAgent* agent, float rate);

/**
 * @brief 获取代理统计信息
 *
 * @param agent 代理句柄
 * @param total_steps 总步数（可选）
 * @param total_episodes 总回合数（可选）
 * @param avg_return 平均回报（可选）
 * @param best_return 最佳回报（可选）
 * @return int 成功返回0，失败返回-1
 */
int rl_get_stats(const RLAgent* agent, int* total_steps, int* total_episodes,
                 float* avg_return, float* best_return);

/**
 * @brief 创建默认RL配置
 *
 * @param algorithm 算法类型
 * @param state_dim 状态维度
 * @param action_dim 动作维度
 * @return RLConfig 默认配置
 */
RLConfig rl_config_default(RLAlgorithm algorithm, int state_dim, int action_dim);

/**
 * @brief 创建经验回放缓冲区
 *
 * @param capacity 容量
 * @param state_dim 状态维度
 * @param action_dim 动作维度
 * @param replay_type 回放类型（均匀/优先）
 * @return RLReplayBuffer 缓冲区（需要调用rl_replay_buffer_destroy释放内存）
 */
RLReplayBuffer rl_replay_buffer_create(int capacity, int state_dim, int action_dim, RLReplayType replay_type);

/**
 * @brief 销毁经验回放缓冲区
 *
 * @param buffer 缓冲区指针
 */
void rl_replay_buffer_destroy(RLReplayBuffer* buffer);

/**
 * @brief 向回放缓冲区添加经验
 *
 * @param buffer 缓冲区指针
 * @param experience 经验数据
 * @return int 成功返回0，失败返回-1
 */
int rl_replay_buffer_add(RLReplayBuffer* buffer, const RLExperience* experience);

/**
 * @brief 从回放缓冲区采样
 *
 * @param buffer 缓冲区指针
 * @param batch_size 批大小
 * @param samples 采样输出缓冲区
 * @param indices 索引输出缓冲区（用于优先回放更新，可为NULL）
 * @param weights 重要性采样权重（用于优先回放，可为NULL）
 * @return int 成功返回采样数，失败返回-1
 */
int rl_replay_buffer_sample(const RLReplayBuffer* buffer, int batch_size,
                            RLExperience* samples, int* indices, float* weights);

/**
 * @brief 更新优先回放中的优先级
 *
 * @param buffer 缓冲区指针
 * @param indices 索引数组
 * @param priorities 优先级数组
 * @param count 数量
 */
void rl_replay_buffer_update_priorities(RLReplayBuffer* buffer, const int* indices,
                                        const float* priorities, int count);

/**
 * @brief 创建默认探索配置
 *
 * @param strategy 探索策略
 * @return RLExploreConfig 默认配置
 */
RLExploreConfig rl_explore_config_default(RLExploreStrategy strategy);

/**
 * @brief DQN专用：获取当前Q值估计
 *
 * @param agent 代理句柄
 * @param state 状态
 * @param state_dim 状态维度
 * @param q_values Q值输出缓冲区（大小为动作数）
 * @return int 成功返回动作数，失败返回-1
 */
int rl_dqn_get_q_values(RLAgent* agent, const float* state, int state_dim, float* q_values);

/**
 * @brief PPO专用：获取动作概率分布
 *
 * @param agent 代理句柄
 * @param state 状态
 * @param state_dim 状态维度
 * @param means 均值输出缓冲区
 * @param log_stds 对数标准差输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int rl_ppo_get_action_dist(RLAgent* agent, const float* state, int state_dim,
                           float* means, float* log_stds);

/**
 * @brief SAC专用：获取策略熵
 *
 * @param agent 代理句柄
 * @return float 当前熵值
 */
float rl_sac_get_entropy(const RLAgent* agent);

/* ============ CfC强化学习算法API ============ */

/**
 * @brief CfC-TD3专用：初始化TD3代理
 * 使用CfC网络的演员-评论家架构，双Q网络延迟更新
 *
 * @param agent 代理句柄
 * @return int 成功返回0，失败返回-1
 */
int rl_cfc_td3_init(RLAgent* agent);

/**
 * @brief CfC-TD3专用：选择确定性动作（无探索噪声）
 *
 * @param agent 代理句柄
 * @param state 状态向量
 * @param state_dim 状态维度
 * @param action 动作输出缓冲区
 * @param action_dim 动作维度
 * @return int 成功返回0，失败返回-1
 */
int rl_cfc_td3_select_deterministic_action(RLAgent* agent, const float* state,
                                            int state_dim, float* action, int action_dim);

/**
 * @brief CfC-Rainbow DQN专用：获取分布Q值
 *
 * @param agent 代理句柄
 * @param state 状态向量
 * @param state_dim 状态维度
 * @param q_distributions 分布输出缓冲区[actions][atoms]
 * @param num_atoms 原子数
 * @return int 成功返回0，失败返回-1
 */
int rl_cfc_rainbow_get_distribution(RLAgent* agent, const float* state, int state_dim,
                                    float* q_distributions, int num_atoms);

/**
 * @brief CfC-IMPALA专用：V-trace目标计算
 *
 * @param agent 代理句柄
 * @param states 状态序列
 * @param actions 动作序列
 * @param rewards 奖励序列
 * @param dones 终止标志序列
 * @param sequence_length 序列长度
 * @param vtrace_targets V-trace目标值输出
 * @return int 成功返回0，失败返回-1
 */
int rl_cfc_impala_compute_vtrace(RLAgent* agent, const float* states,
                                 const float* actions, const float* rewards,
                                 const int* dones, int sequence_length,
                                 float* vtrace_targets);

/**
 * @brief CfC-R2D2专用：使用预热步初始化循环状态
 *
 * @param agent 代理句柄
 * @param burn_in_sequence 预热状态序列
 * @param burn_in_length 预热序列长度
 * @param state_dim 状态维度
 * @return int 成功返回0，失败返回-1
 */
int rl_cfc_r2d2_burn_in(RLAgent* agent, const float* burn_in_sequence,
                         int burn_in_length, int state_dim);

/**
 * @brief CfC-TD3专用：执行TD3训练步（含延迟策略更新）
 * 执行完整的TD3训练，包括双Q学习、延迟策略更新、目标平滑
 *
 * @param agent 代理句柄
 * @param batch_size 批大小
 * @return int 成功返回0，失败返回-1
 */
int rl_cfc_td3_train_step(RLAgent* agent, int batch_size);

/**
 * @brief 设置CfC强化学习代理的ODE求解器类型
 *
 * @param agent 代理句柄
 * @param solver_type ODE求解器类型（0=封闭形式解, 1=RK4, 2=RK45, 3=CTBP）
 * @return int 成功返回0，失败返回-1
 */
int rl_cfc_set_ode_solver(RLAgent* agent, int solver_type);

/**
 * @brief 获取CfC强化学习代理的内部状态（用于调试和监控）
 *
 * @param agent 代理句柄
 * @param critic_loss 评论家损失输出（可选）
 * @param actor_loss 演员损失输出（可选）
 * @param avg_q_value 平均Q值输出（可选）
 * @param policy_delay_count 策略延迟计数输出（可选）
 * @return int 成功返回0，失败返回-1
 */
int rl_cfc_get_training_stats(RLAgent* agent, float* critic_loss,
                              float* actor_loss, float* avg_q_value,
                              int* policy_delay_count);

#ifdef __cplusplus
}
#endif

#endif
