/**
 * @file imitation_learning.h
 * @brief 模仿学习算法实现
 * 
 * 提供机器人模仿学习的算法，包括行为克隆（Behavioral Cloning）、
 * 逆强化学习（Inverse Reinforcement Learning）等。
 * 支持从专家演示中学习策略，实现机器人对人类行为的模仿。
 */

#ifndef SELFLNN_IMITATION_LEARNING_H
#define SELFLNN_IMITATION_LEARNING_H

#include <stddef.h>
#include "selflnn/core/lnn.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 模仿学习算法类型
 */
typedef enum {
    IMITATION_LEARNING_BEHAVIORAL_CLONING = 0,    /**< 行为克隆（直接监督学习） */
    IMITATION_LEARNING_DAgger = 1,                /**< DAgger算法（数据集聚合） */
    IMITATION_LEARNING_INVERSE_RL = 2,            /**< 逆强化学习（推断奖励函数） */
    IMITATION_LEARNING_GAIL = 3,                  /**< 生成对抗模仿学习（GAN式） */
    IMITATION_LEARNING_BEHAVIORAL_CLONING_PLUS = 4, /**< 增强型行为克隆（带正则化） */
    IMITATION_LEARNING_IQ_LEARN = 5,                 /**< 逆Q学习（隐式奖励函数） */
    IMITATION_LEARNING_BAYESIAN_IRL = 6              /**< 贝叶斯逆强化学习 */
} ImitationLearningType;

/**
 * @brief 专家演示数据结构
 */
typedef struct {
    float* state_sequence;        /**< 状态序列（连续状态） */
    float* action_sequence;       /**< 动作序列（专家动作） */
    size_t sequence_length;       /**< 序列长度（时间步） */
    size_t state_dim;             /**< 状态维度 */
    size_t action_dim;            /**< 动作维度 */
    float* reward_sequence;       /**< 奖励序列（可选） */
    int has_rewards;              /**< 是否包含奖励信息 */
    char* description;            /**< 演示描述（可选） */
    long timestamp;               /**< 时间戳（毫秒） */
} ExpertDemonstration;

/**
 * @brief 模仿学习配置
 */
typedef struct {
    ImitationLearningType algorithm_type; /**< 模仿学习算法类型 */
    float learning_rate;                  /**< 学习率 */
    size_t batch_size;                    /**< 批次大小 */
    size_t epochs;                        /**< 训练轮数 */
    float regularization_strength;        /**< 正则化强度（L2） */
    float dropout_rate;                   /**< Dropout率（防止过拟合） */
    int use_mixed_precision;              /**< 是否使用混合精度训练 */
    int enable_gpu_acceleration;          /**< 是否启用GPU加速 */
    
    // DAgger算法特定配置
    float dagger_beta;                    /**< DAgger beta参数（专家查询概率） */
    size_t dagger_iterations;             /**< DAgger迭代次数 */
    
    // 逆强化学习特定配置
    size_t irl_reward_dim;                /**< 奖励函数维度 */
    size_t irl_max_iterations;            /**< 最大迭代次数 */
    float irl_convergence_threshold;      /**< 收敛阈值 */
    
    // 行为克隆增强配置
    float bc_plus_margin;                 /**< 边界损失函数边界值 */
    int bc_use_consistency_loss;          /**< 是否使用一致性损失 */

    // IQ-Learn 特定配置
    float iq_learn_temperature;           /**< IQ-Learn softmax温度参数 */
    float iq_learn_soft_update_tau;       /**< IQ-Learn目标网络软更新系数 */
    int iq_learn_target_update_interval;  /**< IQ-Learn目标网络更新间隔 */
    float iq_learn_bc_coefficient;        /**< IQ-Learn行为克隆正则化系数 */
    int iq_learn_num_action_samples;      /**< IQ-Learn动作采样数量（计算soft V） */
    int iq_learn_hidden_dim;              /**< IQ-Learn Q网络隐层维度 */
    int iq_learn_hidden_layers;           /**< IQ-Learn Q网络隐层数 */
    int iq_learn_replay_buffer_size;      /**< IQ-Learn经验回放缓冲区大小 */
    int iq_learn_gradient_steps;          /**< IQ-Learn每步梯度更新次数 */

    // 贝叶斯逆强化学习特定配置
    int bayesian_irl_mcmc_samples;        /**< 贝叶斯IRL MCMC采样数 */
    int bayesian_irl_burn_in;             /**< 贝叶斯IRL烧入步数 */
    float bayesian_irl_mcmc_step_size;    /**< 贝叶斯IRL MCMC步长 */
    float bayesian_irl_prior_mean;        /**< 贝叶斯IRL先验均值 */
    float bayesian_irl_prior_std;         /**< 贝叶斯IRL先验标准差 */
    float bayesian_irl_likelihood_temp;   /**< 贝叶斯IRL似然温度 */
    int bayesian_irl_thin_interval;       /**< 贝叶斯IRL采样稀疏间隔 */
    int bayesian_irl_reward_hidden_dim;   /**< 贝叶斯IRL奖励网络隐层维度 */
    int bayesian_irl_reward_hidden_layers;/**< 贝叶斯IRL奖励网络隐层数 */
    size_t bayesian_irl_max_iterations;   /**< 贝叶斯IRL最大迭代次数 */
    
    // 通用训练配置
    int verbose;                          /**< 详细输出 */
    int save_checkpoints;                 /**< 是否保存检查点 */
    char checkpoint_dir[256];             /**< 检查点保存目录 */
    
    /* DEEP-005修复: preferred_gpu_backend（imitation_learning.c交叉引用） */
    int preferred_gpu_backend;            /**< 偏好的GPU后端（GpuBackend枚举值，-1=自动检测） */
} ImitationLearningConfig;

/**
 * @brief 模仿学习策略
 */
typedef struct {
    LNN* policy_network;                  /**< 策略网络（状态→动作映射） */
    ImitationLearningType algorithm_type; /**< 使用的算法类型 */
    float* performance_metrics;           /**< 性能指标数组 */
    size_t metrics_count;                 /**< 指标数量 */
    char* algorithm_description;          /**< 算法描述 */
    void* algorithm_specific_data;        /**< 算法特定数据（私有） */
} ImitationPolicy;

/**
 * @brief 模仿学习器
 */
typedef struct {
    ImitationLearningConfig config;       /**< 配置参数 */
    ImitationPolicy* policy;              /**< 学习到的策略 */
    ExpertDemonstration* demonstrations;  /**< 专家演示数组 */
    size_t num_demonstrations;            /**< 演示数量 */
    size_t total_timesteps;               /**< 总时间步数（所有演示） */
    
    // 训练状态
    size_t current_epoch;                 /**< 当前训练轮数 */
    size_t current_iteration;             /**< 当前迭代次数（DAgger/IRL） */
    float current_loss;                   /**< 当前损失值 */
    float best_loss;                      /**< 最佳损失值 */
    
    // 训练缓冲区
    float* state_buffer;                  /**< 状态缓冲区 */
    float* action_buffer;                 /**< 动作缓冲区 */
    float* loss_buffer;                   /**< 损失缓冲区（历史记录） */
    size_t buffer_capacity;               /**< 缓冲区容量 */
    size_t buffer_size;                   /**< 缓冲区当前大小 */
    
    // GPU相关
    void* gpu_context;                    /**< GPU上下文（如果启用） */
    int gpu_initialized;                  /**< GPU是否已初始化 */
    int enabled;                          /**< 能力开关（P3.3） */
} ImitationLearner;

/**
 * @brief 模仿学习结果
 */
typedef struct {
    ImitationPolicy* learned_policy;      /**< 学习到的策略 */
    float final_loss;                     /**< 最终损失值 */
    float policy_accuracy;                /**< 策略准确率（与专家动作匹配度） */
    size_t training_time_ms;              /**< 训练时间（毫秒） */
    size_t total_iterations;              /**< 总迭代次数 */
    char* training_summary;               /**< 训练摘要（文本） */
    float* loss_history;                  /**< 损失历史记录 */
    size_t loss_history_size;             /**< 损失历史记录大小 */
} ImitationLearningResult;

/**
 * @brief 创建模仿学习器
 * 
 * @param config 模仿学习配置
 * @return 模仿学习器指针，失败返回NULL
 */
ImitationLearner* imitation_learner_create(const ImitationLearningConfig* config);

/**
 * @brief 添加专家演示到学习器
 * 
 * @param learner 模仿学习器
 * @param demonstration 专家演示数据
 * @return 成功返回0，失败返回-1
 */
int imitation_learner_add_demonstration(ImitationLearner* learner,
                                       const ExpertDemonstration* demonstration);

/**
 * @brief 执行模仿学习训练
 * 
 * @param learner 模仿学习器
 * @return 模仿学习结果指针，失败返回NULL
 */
ImitationLearningResult* imitation_learner_train(ImitationLearner* learner);

/**
 * @brief 使用学习到的策略进行预测（状态→动作映射）
 * 
 * @param policy 模仿学习策略
 * @param state 当前状态
 * @param action 输出动作（预分配缓冲区）
 * @param state_dim 状态维度
 * @param action_dim 动作维度
 * @return 成功返回0，失败返回-1
 */
int imitation_policy_predict(const ImitationPolicy* policy,
                            const float* state, float* action,
                            size_t state_dim, size_t action_dim);

/**
 * @brief 评估模仿学习策略
 * 
 * @param policy 模仿学习策略
 * @param demonstrations 专家演示数组
 * @param num_demonstrations 演示数量
 * @param metrics 输出评估指标数组（预分配）
 * @param num_metrics 评估指标数量
 * @return 成功返回0，失败返回-1
 */
int imitation_policy_evaluate(const ImitationPolicy* policy,
                             const ExpertDemonstration* demonstrations,
                             size_t num_demonstrations,
                             float* metrics, size_t num_metrics);

/**
 * @brief 保存模仿学习策略到文件
 * 
 * @param policy 模仿学习策略
 * @param filepath 文件路径
 * @return 成功返回0，失败返回-1
 */
int imitation_policy_save(const ImitationPolicy* policy, const char* filepath);

/**
 * @brief 从文件加载模仿学习策略
 * 
 * @param filepath 文件路径
 * @return 模仿学习策略指针，失败返回NULL
 */
ImitationPolicy* imitation_policy_load(const char* filepath);

/**
 * @brief 释放模仿学习器资源
 * 
 * @param learner 模仿学习器
 */
void imitation_learner_free(ImitationLearner* learner);

/**
 * @brief 释放模仿学习策略资源
 * 
 * @param policy 模仿学习策略
 */
void imitation_policy_free(ImitationPolicy* policy);

/**
 * @brief 释放模仿学习结果资源
 * 
 * @param result 模仿学习结果
 */
void imitation_learning_result_free(ImitationLearningResult* result);

/**
 * @brief 创建专家演示数据
 * 
 * @param state_sequence 状态序列
 * @param action_sequence 动作序列
 * @param sequence_length 序列长度
 * @param state_dim 状态维度
 * @param action_dim 动作维度
 * @param description 演示描述（可选）
 * @return 专家演示数据指针，失败返回NULL
 */
ExpertDemonstration* expert_demonstration_create(const float* state_sequence,
                                               const float* action_sequence,
                                               size_t sequence_length,
                                               size_t state_dim, size_t action_dim,
                                               const char* description);

/**
 * @brief 释放专家演示数据资源
 * 
 * @param demonstration 专家演示数据
 */
void expert_demonstration_free(ExpertDemonstration* demonstration);

// ============================================================================
// 视觉模仿学习接口
// 从视觉输入（图像/视频帧）中学习策略，使用深度视觉特征提取作为状态表示
// ============================================================================

/**
 * @brief 视觉模仿学习配置
 */
typedef struct {
    int image_width;                      /**< 输入图像宽度 */
    int image_height;                     /**< 输入图像高度 */
    int image_channels;                   /**< 输入图像通道数（1=灰度，3=RGB） */
    int feature_dimension;                /**< 视觉特征提取维度 */
    float learning_rate;                  /**< 学习率 */
    size_t batch_size;                    /**< 批次大小 */
    size_t epochs;                        /**< 训练轮数 */
    int dagger_iterations;                /**< DAgger迭代次数 */
    float dagger_beta;                    /**< DAgger专家查询概率（初始值） */
    int use_multi_scale_features;         /**< 是否使用多尺度特征 */
    int enable_gpu;                       /**< 是否启用GPU加速 */
} ImitationVisualConfig;

/**
 * @brief 视觉专家演示（图像序列+动作序列）
 */
typedef struct {
    float* image_sequence;                /**< 图像序列 [seq_len][ch][h][w] */
    size_t sequence_length;               /**< 序列长度 */
    int width;                            /**< 图像宽度 */
    int height;                           /**< 图像高度 */
    int channels;                         /**< 图像通道数 */
    float* action_sequence;               /**< 动作序列 [seq_len][action_dim] */
    size_t action_dim;                    /**< 动作维度 */
    char* description;                    /**< 演示描述（可选） */
} VisualExpertDemonstration;

/**
 * @brief 视觉模仿学习器
 */
typedef struct VisualImitationLearner VisualImitationLearner;

/**
 * @brief 创建视觉模仿学习器
 *
 * @param config 视觉模仿学习配置
 * @return 视觉模仿学习器指针，失败返回NULL
 */
VisualImitationLearner* visual_imitation_learner_create(const ImitationVisualConfig* config);

/**
 * @brief 释放视觉模仿学习器
 *
 * @param learner 视觉模仿学习器
 */
void visual_imitation_learner_free(VisualImitationLearner* learner);

/**
 * @brief 添加视觉专家演示
 *
 * @param learner 视觉模仿学习器
 * @param demonstration 视觉专家演示数据
 * @return 成功返回0，失败返回-1
 */
int visual_imitation_learner_add_demonstration(VisualImitationLearner* learner,
                                                const VisualExpertDemonstration* demonstration);

/**
 * @brief 执行视觉行为克隆（BC）训练
 *
 * 将图像通过深度视觉特征提取转换为状态特征向量，然后执行标准行为克隆。
 *
 * @param learner 视觉模仿学习器
 * @return 模仿学习结果指针，失败返回NULL
 */
ImitationLearningResult* visual_imitation_train_bc(VisualImitationLearner* learner);

/**
 * @brief 执行视觉DAgger训练
 *
 * 交互式模仿学习，在策略执行过程中用专家纠正，图像→特征→策略循环。
 *
 * @param learner 视觉模仿学习器
 * @param oracle_policy 专家策略函数指针（接收图像输出动作，可为NULL使用内置演示）
 * @param oracle_context 专家策略上下文指针
 * @return 模仿学习结果指针，失败返回NULL
 */
ImitationLearningResult* visual_imitation_train_dagger(VisualImitationLearner* learner,
                                                       int (*oracle_policy)(const float* image,
                                                                            int w, int h, int ch,
                                                                            float* action,
                                                                            size_t action_dim,
                                                                            void* context),
                                                       void* oracle_context);

/**
 * @brief 使用视觉模仿策略预测动作
 *
 * @param learner 视觉模仿学习器
 * @param image 输入图像数据
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 图像通道数
 * @param action 输出动作缓冲区
 * @param action_dim 动作维度
 * @return 成功返回0，失败返回-1
 */
int visual_imitation_predict(VisualImitationLearner* learner,
                             const float* image, int width, int height, int channels,
                             float* action, size_t action_dim);

/**
 * @brief 保存视觉模仿学习器到文件
 *
 * @param learner 视觉模仿学习器
 * @param filepath 文件路径
 * @return 成功返回0，失败返回-1
 */
int visual_imitation_learner_save(const VisualImitationLearner* learner, const char* filepath);

/**
 * @brief 从文件加载视觉模仿学习器
 *
 * @param filepath 文件路径
 * @return 视觉模仿学习器指针，失败返回NULL
 */
VisualImitationLearner* visual_imitation_learner_load(const char* filepath);

/* ============================
 * 能力开关控制（P3.3）
 * ============================ */

/**
 * @brief 启用模仿学习器
 * 
 * @param learner 模仿学习器句柄
 */
void imitation_learner_enable(ImitationLearner* learner);

/**
 * @brief 禁用模仿学习器
 * 
 * @param learner 模仿学习器句柄
 */
void imitation_learner_disable(ImitationLearner* learner);

/**
 * @brief 检查模仿学习器是否已启用
 * 
 * @param learner 模仿学习器句柄
 * @return 1表示启用，0表示禁用
 */
int imitation_learner_is_enabled(const ImitationLearner* learner);

/**
 * @brief 自动采集演示数据：当视觉/动作数据可用时自动填充演示缓冲区
 * 
 * 封装了expert_demonstration_create和imitation_learner_add_demonstration，
 * 为自动数据采集管道提供便捷接口。
 * 
 * @param learner 模仿学习器
 * @param state_sequence 状态序列（连续状态观测）
 * @param action_sequence 动作序列（专家动作）
 * @param sequence_length 序列长度（时间步）
 * @param state_dim 状态维度
 * @param action_dim 动作维度
 * @param description 演示描述（可选，可为NULL）
 * @return 成功返回0，失败返回-1
 */
int imitation_learner_auto_collect_demonstrations(ImitationLearner* learner,
    const float* state_sequence, const float* action_sequence,
    size_t sequence_length, size_t state_dim, size_t action_dim,
    const char* description);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_IMITATION_LEARNING_H */