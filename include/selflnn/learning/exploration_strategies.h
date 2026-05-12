/**
 * @file exploration_strategies.h
 * @brief 深度探索策略系统接口
 *
 * 基于CfC液态神经网络的深度探索策略系统，提供三种核心探索算法：
 * 1. A05.1.3.1 ICM(Intrinsic Curiosity Module) — 内在好奇心模块
 * 2. A05.1.3.2 RND(Random Network Distillation) — 随机网络蒸馏
 * 3. A05.1.3.3 Go-Explore — 先探索后利用
 *
 * 以及多种辅助探索策略：
 * - 噪声网络探索 (NoisyNet)
 * - 参数空间噪声探索
 * - 计数基础探索
 * - 不确定性基础探索
 *
 * 100% 纯C实现，无第三方库依赖。
 */

#ifndef SELFLNN_EXPLORATION_STRATEGIES_H
#define SELFLNN_EXPLORATION_STRATEGIES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 配置常量
 * ============================================================================ */
#define EXPLORE_MAX_STATES          16777216 /**< 最大状态数 */
#define EXPLORE_MAX_ACTIONS         65536    /**< 最大动作数 */
#define EXPLORE_MAX_HIDDEN_DIM      4096     /**< 最大隐藏维度 */
#define EXPLORE_MAX_BONUS_HISTORY   1048576  /**< 最大激励历史 */
#define EXPLORE_ARCHIVE_MAX_CELLS   1048576  /**< Go-Explore存档最大单元数 */

/* ============================================================================
 * 类型定义
 * ============================================================================ */

/**
 * @brief 探索策略类型
 */
typedef enum {
    EXPLORE_ICM = 0,                         /**< 内在好奇心模块 */
    EXPLORE_RND,                             /**< 随机网络蒸馏 */
    EXPLORE_GO_EXPLORE,                      /**< Go-Explore */
    EXPLORE_NOISY_NET,                       /**< 噪声网络 */
    EXPLORE_PARAMETER_NOISE,                 /**< 参数空间噪声 */
    EXPLORE_COUNT_BASED,                     /**< 计数基础 */
    EXPLORE_UNCERTAINTY_BASED,               /**< 不确定基础 */
    EXPLORE_HYBRID                           /**< 混合策略 */
} ExploreStrategyType;

/**
 * @brief ICM预测头类型
 */
typedef enum {
    ICM_INVERSE_DYNAMICS = 0,                /**< 逆动力学(状态→动作) */
    ICM_FORWARD_DYNAMICS,                    /**< 正动力学(状态+动作→下一状态) */
    ICM_DUAL                                 /**< 双头(逆+正动力学) */
} ICMHeadType;

/**
 * @brief Go-Explore阶段
 */
typedef enum {
    GO_EXPLORE_PHASE_EXPLORE = 0,           /**< 探索阶段 */
    GO_EXPLORE_PHASE_ROBUSTIFY              /**< 鲁棒化阶段 */
} GoExplorePhase;

/**
 * @brief 探索激励配置
 */
typedef struct {
    float intrinsic_reward_coeff;            /**< 内在奖励系数(η) */
    float extrinsic_reward_coeff;            /**< 外在奖励系数 */
    float exploration_bonus_decay;           /**< 探索奖励衰减 */
    float novelty_threshold;                 /**< 新颖性阈值 */
    int use_adaptive_coeff;                  /**< 自适应系数 */
} ExploreRewardConfig;

/**
 * @brief ICM配置（A05.1.3.1）
 */
typedef struct {
    int state_dim;                           /**< 状态维度 */
    int action_dim;                          /**< 动作维度 */
    int embedding_dim;                       /**< 嵌入维度 */
    int hidden_dim;                          /**< 隐藏层维度 */
    ICMHeadType head_type;                   /**< 预测头类型 */
    float forward_loss_weight;               /**< 正动力学损失权重(β) */
    float inverse_loss_weight;               /**< 逆动力学损失权重(1-β) */
    float cfc_tau;                           /**< CfC时间常数 */
    float cfc_dt;                            /**< CfC步长 */
    int cfc_steps;                           /**< CfC积分步数 */
    float learning_rate;                     /**< 学习率 */
} ICMConfig;

/**
 * @brief RND配置（A05.1.3.2）
 */
typedef struct {
    int state_dim;                           /**< 状态维度 */
    int embedding_dim;                       /**< 嵌入维度 */
    int hidden_dim;                          /**< 隐藏层维度 */
    int num_predictors;                      /**< 预测器数量 */
    float predictor_noise_std;               /**< 预测器噪声标准差 */
    float target_noise_std;                  /**< 目标网络噪声标准差 */
    float learning_rate;                     /**< 学习率 */
    float cfc_tau;                           /**< CfC时间常数 */
    float cfc_dt;                            /**< CfC步长 */
    int cfc_steps;                           /**< CfC积分步数 */
} RNDConfig;

/**
 * @brief Go-Explore配置（A05.1.3.3）
 */
typedef struct {
    int state_dim;                           /**< 状态维度 */
    int action_dim;                          /**< 动作维度 */
    int archive_cell_capacity;               /**< 每个存档单元的容量 */
    int max_episode_steps;                   /**< 最大回合步数 */
    float cell_threshold;                    /**< 单元相似度阈值 */
    int selection_strategy;                  /**< 选择策略(0=随机,1=加权,2=新奇优先) */
    float robustify_success_rate;            /**< 鲁棒化成功率阈值 */
    float cfc_tau;                           /**< CfC时间常数 */
    float cfc_dt;                            /**< CfC步长 */
    int cfc_steps;                           /**< CfC积分步数 */
} GoExploreConfig;

/**
 * @brief 噪声网络配置
 */
typedef struct {
    int input_dim;                           /**< 输入维度 */
    int output_dim;                          /**< 输出维度 */
    int hidden_dim;                          /**< 隐藏层维度 */
    float sigma_init;                        /**< 噪声标准差初始值 */
    float learning_rate;                     /**< 学习率 */
} NoisyNetConfig;

/**
 * @brief 探索状态
 */
typedef struct ExploreState ExploreState;

/* ============================================================================
 * A05.1.3.1 — ICM内在好奇心模块API
 * ============================================================================ */

/**
 * @brief 创建ICM好奇心模块
 *
 * @param config ICM配置
 * @return ExploreState* 成功返回状态指针，失败返回NULL
 */
ExploreState* explore_icm_create(const ICMConfig* config);

/**
 * @brief 计算ICM内在奖励
 *
 * 内在奖励 = 前向动力学预测误差：
 *   r_intrinsic = ||φ(s_{t+1}) - f(φ(s_t), a_t)||₂
 *
 * 逆动力学损失（辅助训练）：
 *   L_inv = CrossEntropy(g(φ(s_t), φ(s_{t+1})), a_t)
 *
 * 总损失 = (1-β) * L_inv + β * L_forward
 *
 * @param state 探索状态
 * @param current_state 当前状态 [state_dim]
 * @param next_state 下一状态 [state_dim]
 * @param action 执行动作 [action_dim]
 * @param intrinsic_reward 输出内在奖励值
 * @return int 成功返回0，失败返回-1
 */
int explore_icm_compute_reward(ExploreState* state,
                                const float* current_state,
                                const float* next_state,
                                const float* action,
                                float* intrinsic_reward);

/**
 * @brief 训练ICM模块（一个batch）
 *
 * @param state 探索状态
 * @param states 状态序列 [batch_size][state_dim]
 * @param next_states 下一状态序列 [batch_size][state_dim]
 * @param actions 动作序列 [batch_size][action_dim]
 * @param batch_size 批量大小
 * @return float 平均损失值
 */
float explore_icm_train_batch(ExploreState* state,
                               const float* states, const float* next_states,
                               const float* actions, int batch_size);

/**
 * @brief 获取ICM嵌入特征
 *
 * @param state 探索状态
 * @param raw_state 原始状态 [state_dim]
 * @param embedding 输出嵌入特征 [embedding_dim]
 * @return int 成功返回0，失败返回-1
 */
int explore_icm_get_embedding(ExploreState* state,
                               const float* raw_state, float* embedding);

/* ============================================================================
 * A05.1.3.2 — RND随机网络蒸馏API
 * ============================================================================ */

/**
 * @brief 创建RND模块
 *
 * RND使用两个网络：
 *   - 目标网络f*：随机初始化且固定不变
 *   - 预测网络f_θ：训练使得f_θ(x) ≈ f*(x)
 * 探索奖励 = ||f_θ(x) - f*(x)||₂ （预测误差=新奇度）
 *
 * @param config RND配置
 * @return ExploreState* 成功返回状态指针，失败返回NULL
 */
ExploreState* explore_rnd_create(const RNDConfig* config);

/**
 * @brief 计算RND探索奖励
 *
 * r_rnd = ||f_θ(s) - f*(s)||₂
 * 用CfC ODE平滑：dr/dt = (r_raw - r)/τ
 *
 * @param state 探索状态
 * @param observation 观测状态 [state_dim]
 * @param reward 输出探索奖励
 * @return int 成功返回0，失败返回-1
 */
int explore_rnd_compute_reward(ExploreState* state,
                                const float* observation, float* reward);

/**
 * @brief 训练RND预测网络
 *
 * 最小化预测误差：min_θ ||f_θ(s) - f*(s)||₂²
 *
 * @param state 探索状态
 * @param observations 观测序列 [batch_size][state_dim]
 * @param batch_size 批量大小
 * @return float 平均蒸馏误差
 */
float explore_rnd_train_batch(ExploreState* state,
                               const float* observations, int batch_size);

/**
 * @brief 获取RND新颖性分数
 *
 * @param state 探索状态
 * @param observation 观测
 * @return float 新颖性分数(0~∞，越大越新奇)
 */
float explore_rnd_get_novelty(ExploreState* state, const float* observation);

/* ============================================================================
 * A05.1.3.3 — Go-Explore先探索后利用API
 * ============================================================================ */

/**
 * @brief 创建Go-Explore探索引擎
 *
 * @param config Go-Explore配置
 * @return ExploreState* 成功返回状态指针，失败返回NULL
 */
ExploreState* explore_go_create(const GoExploreConfig* config);

/**
 * @brief 向存档添加新状态单元
 *
 * 对新状态进行离散化（降维/量化）：
 *   1. 使用CfC编码：h = CFC_encoder(s)
 *   2. 离散化：cell_id = quantize(h)
 *   3. 如果cell_id不在存档中，添加新单元
 *   4. 更新单元统计（访问次数、最佳奖励等）
 *
 * @param state 探索状态
 * @param cell_state 单元状态 [state_dim]
 * @param episode_reward 该回合累积奖励
 * @return int 成功返回cell_id，失败返回-1
 */
int explore_go_add_cell(ExploreState* state,
                         const float* cell_state, float episode_reward);

/**
 * @brief 从存档中选择探索单元
 *
 * 选择策略：
 *   0=随机均匀采样
 *   1=加权采样（P(cell) ∝ reward / visit_count）
 *   2=新奇优先（最少访问单元）
 *   CfC选择: P(cell_i) = softmax(W_sel · h_cell_i)
 *
 * @param state 探索状态
 * @param selected_cell 输出选中单元状态 [state_dim]
 * @param cell_id 输出单元ID
 * @return int 成功返回0，失败返回-1
 */
int explore_go_select_cell(ExploreState* state,
                            float* selected_cell, int* cell_id);

/**
 * @brief 从选中单元开始探索轨迹
 *
 * 从存档单元出发执行探索：
 *   1. 加载选中单元的状态
 *   2. 用记录的动作序列回到该状态（如果可用）
 *   3. 从该状态开始随机/策略探索
 *   4. 记录新发现的单元
 *
 * @param state 探索状态
 * @param cell_id 起始单元ID
 * @param trajectory_states 轨迹状态输出 [max_steps][state_dim]
 * @param trajectory_actions 轨迹动作输出 [max_steps][action_dim]
 * @param max_steps 最大步数
 * @return int 实际探索步数，失败返回-1
 */
int explore_go_explore_from_cell(ExploreState* state,
                                  int cell_id,
                                  float* trajectory_states,
                                  float* trajectory_actions,
                                  int max_steps);

/**
 * @brief 执行鲁棒化阶段
 *
 * 对高奖励单元，学习可靠的策略回到该单元：
 *   使用CfC策略网络学习：
 *     π_θ(a|s)：从任何状态回到目标单元的策略
 *   训练目标：最大化回到目标单元的成功率
 *
 * @param state 探索状态
 * @param target_cell_id 目标单元ID
 * @param max_attempts 最大尝试次数
 * @return float 鲁棒化成功率(0~1)
 */
float explore_go_robustify(ExploreState* state,
                            int target_cell_id, int max_attempts);

/**
 * @brief 获取存档统计
 *
 * @param state 探索状态
 * @param num_cells 输出存档单元数
 * @param total_visits 输出总访问次数
 * @param best_reward 输出最佳奖励
 * @return int 成功返回0，失败返回-1
 */
int explore_go_get_stats(ExploreState* state,
                          int* num_cells, int* total_visits, float* best_reward);

/* ============================================================================
 * 通用探索工具API
 * ============================================================================ */

/**
 * @brief 创建噪声网络（NoisyNet）
 *
 * 参数：W = μ_W + σ_W ⊙ ε_W (ε_W ~ N(0,1))
 * 噪声在每个前向传播前采样，反向传播通过μ和σ
 *
 * @param config 噪声网络配置
 * @return ExploreState* 成功返回状态指针，失败返回NULL
 */
ExploreState* explore_noisynet_create(const NoisyNetConfig* config);

/**
 * @brief 噪声网络前向传播
 *
 * @param state 探索状态
 * @param input 输入向量
 * @param output 输出向量
 * @param sample_noise 是否重新采样噪声(1=是,0=复用上次噪声)
 * @return int 成功返回0，失败返回-1
 */
int explore_noisynet_forward(ExploreState* state,
                              const float* input, float* output,
                              int sample_noise);

/**
 * @brief 参数空间噪声探索
 *
 * 在策略网络参数上添加高斯噪声：
 *   θ' = θ + σ * ε  (ε ~ N(0, I))
 * 使用自适应σ调整：
 *   σ_new = σ * exp(α * (δ_actual - δ_expected))
 *
 * @param state 探索状态
 * @param params 原始参数量
 * @param noise_std 噪声标准差
 * @param noisy_params 输出带噪声的参数
 * @return int 成功返回0，失败返回-1
 */
int explore_parameter_noise(ExploreState* state,
                             const float* params, float noise_std,
                             float* noisy_params);

/**
 * @brief 计数基础探索奖励
 *
 * r_count = β / sqrt(N(s) + 1)
 * 其中N(s)是状态s的访问计数
 * CfC增强：N_cfc(s) = N(s) + f_cfc(h_s) （液态计数泛化）
 *
 * @param state 探索状态
 * @param state_vector 状态向量
 * @param state_dim 状态维度
 * @return float 计数基础探索奖励
 */
float explore_count_based_reward(ExploreState* state,
                                  const float* state_vector, int state_dim);

/**
 * @brief 不确定性基础探索奖励
 *
 * r_uncertain = H(P(·|s, θ))  (预测熵)
 * 或 = Var[Q(s, a)]  (Q值方差)
 *
 * @param state 探索状态
 * @param predictions 预测分布 [num_predictions]
 * @param num_predictions 预测数
 * @return float 不确定性基础探索奖励
 */
float explore_uncertainty_based_reward(ExploreState* state,
                                        const float* predictions,
                                        int num_predictions);

/**
 * @brief 计算混合探索奖励
 *
 * r_total = w_icm * r_icm + w_rnd * r_rnd + w_count * r_count + w_uncertain * r_uncertain
 * 权重自适应调节
 *
 * @param state 探索状态
 * @param rewards 各策略奖励数组 [num_strategies]
 * @param strategy_types 对应策略类型数组
 * @param num_strategies 策略数
 * @return float 混合探索奖励
 */
float explore_hybrid_reward(ExploreState* state,
                             const float* rewards,
                             const ExploreStrategyType* strategy_types,
                             int num_strategies);

/**
 * @brief 销毁探索状态
 *
 * @param state 探索状态指针
 */
void explore_destroy(ExploreState* state);

/**
 * @brief 保存探索模型
 *
 * @param state 探索状态
 * @param strategy_type 策略类型
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int explore_save(const ExploreState* state, ExploreStrategyType strategy_type,
                  const char* filepath);

/**
 * @brief 加载探索模型
 *
 * @param strategy_type 策略类型
 * @param filepath 文件路径
 * @return ExploreState* 成功返回状态指针，失败返回NULL
 */
ExploreState* explore_load(ExploreStrategyType strategy_type, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_EXPLORATION_STRATEGIES_H */
