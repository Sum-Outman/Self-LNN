/**
 * @file training.h
 * @brief 训练模块
 * 
 * 提供神经网络训练算法，包括梯度下降、反向传播、正则化和优化器。
 */

#ifndef SELFLNN_TRAINING_H
#define SELFLNN_TRAINING_H

#include "selflnn/core/lnn.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/learning/learning.h"
#include "selflnn/training/regularization.h"
#include "selflnn/training/model_version.h"
#include "selflnn/memory/memory.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 前向声明 */
typedef struct MixedPrecisionContext MixedPrecisionContext;
typedef struct MemoryManager MemoryManager;

/**
 * @brief 训练模式
 */
typedef enum {
    TRAIN_MODE_BATCH = 0,      /**< 批量训练 */
    TRAIN_MODE_MINI_BATCH = 1, /**< 小批量训练 */
    TRAIN_MODE_ONLINE = 2,     /**< 在线训练 */
    TRAIN_MODE_ADAPTIVE = 3    /**< 自适应训练 */
} TrainMode;

/**
 * @brief 正则化类型
 */
typedef enum {
    REGULARIZATION_NONE = 0,   /**< 无正则化 */
    REGULARIZATION_L1 = 1,     /**< L1正则化 */
    REGULARIZATION_L2 = 2,     /**< L2正则化 */
    REGULARIZATION_DROPOUT = 3,/**< Dropout正则化 */
    REGULARIZATION_EARLY_STOP = 4 /**< 早停正则化 */
} RegularizationType;

/**
 * @brief 梯度裁剪类型
 */
typedef enum {
    GRADIENT_CLIP_NONE = 0,    /**< 无梯度裁剪 */
    GRADIENT_CLIP_VALUE = 1,   /**< 值裁剪 */
    GRADIENT_CLIP_NORM = 2     /**< 范数裁剪 */
} GradientClipType;

/**
 * @brief 训练配置
 */
typedef struct {
    TrainMode mode;                   /**< 训练模式 */
    OptimizerType optimizer;          /**< 优化器类型 */
    LossType loss_function;           /**< 损失函数类型 */
    RegularizationType regularization;/**< 正则化类型 */
    GradientClipType gradient_clip;   /**< 梯度裁剪类型 */
    
    float learning_rate;              /**< 学习率 */
    float learning_rate_decay;        /**< 学习率衰减 */
    float momentum;                   /**< 动量 */
    float beta1;                      /**< Adam beta1 */
    float beta2;                      /**< Adam beta2 */
    float epsilon;                    /**< 数值稳定性epsilon */
    
    float regularization_lambda;      /**< 正则化强度 */
    float dropout_rate;               /**< Dropout率 */
    float gradient_clip_value;        /**< 梯度裁剪值 */
    float gradient_clip_norm;         /**< 梯度裁剪范数 */
    
    size_t input_size;                 /**< 输入特征维度 */
    size_t output_size;                /**< 输出特征维度 */
    
    // 梯度压缩配置
    int use_gradient_compression;     /**< 是否启用梯度压缩（Top-k稀疏化） */
    float gradient_compression_ratio; /**< 梯度压缩率 (0.01~1.0, 1.0=无压缩) */
    
    size_t batch_size;                /**< 批量大小 */
    size_t epochs;                    /**< 训练轮数 */
    size_t patience;                  /**< 早停耐心值 */
    size_t validation_split;          /**< 验证集分割比例（百分比） */
    
    float target_accuracy;            /**< 目标准确率 */
    int enable_validation;            /**< 是否启用验证 */
    
    int shuffle_data;                 /**< 是否打乱数据 */
    int verbose;                      /**< 是否显示训练信息 */
    int save_best_model;              /**< 是否保存最佳模型 */
    int use_gpu;                      /**< 是否使用GPU */
    
    // 拉普拉斯优化配置
    int use_laplace_optimization;     /**< 是否使用拉普拉斯优化 */
    float laplace_filter_cutoff;      /**< 拉普拉斯滤波器截止频率（Hz） */
    float laplace_stability_margin;   /**< 稳定性裕度阈值（dB/度） */
    int laplace_adaptive_filtering;   /**< 是否使用自适应滤波 */
    int laplace_monitor_stability;    /**< 是否监控训练稳定性 */
    
    // 混合精度训练配置
    int use_mixed_precision;          /**< 是否使用混合精度训练 */
    int mixed_precision_mode;         /**< 混合精度模式 (0=禁用, 1=FP16, 2=BFLOAT16, 3=自动, 4=动态) */
    float mixed_precision_initial_scale; /**< 初始梯度缩放因子 */
    float mixed_precision_max_scale;  /**< 最大梯度缩放因子 */
    float mixed_precision_min_scale;  /**< 最小梯度缩放因子 */
    int mixed_precision_enable_fp16_arithmetic; /**< 启用FP16算术运算 */
    int mixed_precision_enable_fp16_storage; /**< 启用FP16存储 */
    int mixed_precision_check_nan_inf; /**< 检查NaN/Inf */
    
    // 自适应学习率配置
    int enable_adaptive_lr;          /**< 是否启用自适应学习率 (ReduceLROnPlateau) */
    float lr_min_delta;              /**< 最小改善delta（用于判断是否改善） */
    size_t lr_patience;              /**< 耐心值（多少个epoch无改善后降低学习率） */
    float lr_factor;                 /**< 学习率调整因子（乘以当前学习率） */
    float lr_min_factor;             /**< 最小学习率因子（相对于初始学习率） */
    
    // 分布式训练配置
    int use_distributed_training;           /**< 是否使用分布式训练 */
    int distributed_node_id;                /**< 分布式节点ID (0-based) */
    int distributed_num_nodes;              /**< 分布式总节点数 */
    int distributed_communication_backend;  /**< 分布式通信后端 (0=MPI, 1=NCCL, 2=自定义) */
    int distributed_sync_frequency;         /**< 梯度同步频率 (每多少个批次同步一次) */
    int distributed_allreduce_algorithm;    /**< 全归约算法 (0=环, 1=树, 2=混合) */
    int distributed_enable_checkpointing;   /**< 是否启用检查点 */
    int distributed_checkpoint_frequency;   /**< 检查点保存频率 (每多少个批次保存一次) */
    int distributed_enable_fault_tolerance; /**< 是否启用容错机制 */
    int distributed_max_retries;            /**< 分布式操作最大重试次数 */
    float distributed_learning_rate_scale;  /**< 分布式学习率缩放因子 */
    int distributed_failure_recovery_enabled; /**< 是否启用分布式节点故障恢复 */
    int distributed_failure_recovery_max_attempts; /**< 分布式节点故障恢复最大尝试次数 */
    int distributed_heartbeat_timeout_ms;   /**< 心跳超时时间（毫秒） */
    int distributed_heartbeat_interval_ms;  /**< 心跳间隔（毫秒） */
    int gradient_accumulation_steps;        /**< 梯度累积步数（>1启用梯度累积） */
    
    // 训练流程配置
    int training_phase;                     /**< 训练阶段：0=从零训练，1=预训练，2=微调 */
    char* pretrained_weights_path;          /**< 预训练权重文件路径（如果提供） */
    int freeze_base_layers;                 /**< 是否冻结基础层（微调时） */
    float fine_tune_learning_rate;          /**< 微调学习率（如果为0，则使用原学习率） */
    int enable_transfer_learning;           /**< 是否启用迁移学习 */
    int enable_continual_learning;          /**< 是否启用持续学习 */
    float knowledge_retention_factor;       /**< 知识保留因子（0-1） */
    
    // 记忆系统集成配置
    int enable_memory_integration;         /**< 是否启用记忆系统集成 */
    size_t memory_short_term_capacity;     /**< 记忆系统短期记忆容量 */
    size_t memory_long_term_capacity;      /**< 记忆系统长期记忆容量 */
    float memory_context_strength;         /**< 记忆上下文影响强度 (0.0~1.0) */
    size_t memory_consolidation_interval;  /**< 记忆巩固间隔（epoch数） */
    
    // P3.6 演化算法集成配置
    int enable_evolution;                   /**< 是否启用演化算法集成到训练循环 */
    size_t evolution_interval;              /**< 演化触发间隔（epoch数），0表示禁用 */
    size_t evolution_population_size;       /**< 演化种群大小（默认10-30） */
    float evolution_mutation_rate;          /**< 演化突变率（默认0.15） */
    float evolution_crossover_rate;         /**< 演化交叉率（默认0.8） */
    float evolution_elitism_rate;           /**< 演化精英保留率（默认0.1） */
    int evolution_use_validation;           /**< 演化是否使用验证集评估（默认1） */
} TrainingConfig;

/**
 * @brief 训练状态
 */
typedef struct {
    float current_loss;               /**< 当前损失 */
    float current_accuracy;           /**< 当前准确率 */
    float validation_loss;            /**< 验证损失 */
    float validation_accuracy;        /**< 验证准确率 */
    float best_loss;                  /**< 最佳损失 */
    float best_accuracy;              /**< 最佳准确率 */
    
    size_t current_epoch;             /**< 当前轮数 */
    size_t current_batch;             /**< 当前批次 */
    size_t total_batches;             /**< 总批次数 */
    size_t steps_without_improvement; /**< 没有改进的步数 */
    
    float learning_rate;              /**< 当前学习率 */
    float gradient_norm;              /**< 梯度范数 */
    float weight_norm;                /**< 权重范数 */
    
    uint64_t training_time_ms;        /**< 训练时间（毫秒） */
    uint64_t start_time;              /**< 开始时间 */
    uint64_t samples_processed;       /**< 已处理的样本数 */
    size_t total_iterations;          /**< 总迭代次数 */
    uint64_t communication_overhead_ms; /**< 通信开销（毫秒） */
    int failed_sync_attempts;         /**< 失败的同步尝试次数 */
} TrainingState;

/**
 * @brief 训练历史记录
 */
typedef struct {
    float* train_losses;              /**< 训练损失历史 */
    float* train_accuracies;          /**< 训练准确率历史 */
    float* val_losses;                /**< 验证损失历史 */
    float* val_accuracies;            /**< 验证准确率历史 */
    float* learning_rates;            /**< 学习率历史 */
    char** train_modes;               /**< 训练模式名称历史（如"预训练"、"微调"、"持续训练"等） */
    
    size_t size;                      /**< 历史记录大小 */
    size_t capacity;                  /**< 历史记录容量 */
} TrainingHistory;

/**
 * @brief 稳定性指标
 */
typedef struct {
    float stability_score;            /**< 稳定性分数（0.0-1.0） */
    float phase_margin;               /**< 相位裕度（度） */
    float gain_margin;                /**< 增益裕度（dB） */
    float damping_ratio;              /**< 阻尼比 */
    float natural_frequency;          /**< 自然频率（Hz） */
    float gradient_variance;          /**< 梯度方差 */
    float gradient_autocorrelation;   /**< 梯度自相关系数 */
    int is_stable;                    /**< 是否稳定（1=稳定，0=不稳定） */
    int warning_type;                 /**< 警告类型（0=无警告，1=梯度爆炸，2=梯度消失） */
} StabilityMetrics;

/**
 * @brief 训练回调函数类型
 */
typedef void (*TrainingCallback)(TrainingState* state, void* user_data);

/**
 * @brief 训练事件（用于预训练/渐进式训练回调）
 */
typedef struct {
    size_t epoch;                    /**< 当前轮数 */
    size_t total_epochs;             /**< 总轮数 */
    float loss;                      /**< 当前损失 */
    float best_loss;                 /**< 最佳损失 */
    int phase;                       /**< 训练阶段(0=从零,1=单模态/预训练,2=多模态/微调,3=联合微调/对齐) */
} TrainingEvent;

/**
 * @brief 训练事件回调类型（返回非0停止训练）
 */
typedef int (*TrainingEventCallback)(TrainingEvent* event, void* user_data);

/**
 * @brief 训练器结构体
 */
typedef struct Trainer Trainer;

/**
 * @brief 创建训练器
 * 
 * @param config 训练配置
 * @param network 要训练的神经网络
 * @return Trainer* 训练器指针，失败返回NULL
 */
Trainer* trainer_create(const TrainingConfig* config, LNN* network);

/**
 * @brief 销毁训练器
 * 
 * @param trainer 训练器指针
 */
void trainer_free(Trainer* trainer);

/**
 * @brief 训练完成后清理检查点临时文件
 * 
 * @param checkpoint_path 检查点文件路径
 * @return int 成功返回0，失败返回-1
 */
int training_cleanup_checkpoint_temp(const char* checkpoint_path);

/**
 * @brief 训练神经网络
 * 
 * @param trainer 训练器
 * @param inputs 输入数据（样本数 x 输入维度）
 * @param targets 目标数据（样本数 x 输出维度）
 * @param num_samples 样本数
 * @param callback 训练回调函数（可选）
 * @param user_data 回调函数用户数据（可选）
 * @return int 成功返回0，失败返回-1
 */
int trainer_train(Trainer* trainer, const float* inputs, const float* targets,
                  size_t num_samples,
                  TrainingCallback callback, void* user_data);

/**
 * @brief 在线训练神经网络（流式处理）
 *
 * 处理流式到达的样本，每次处理一个样本或小批次，支持：
 * - 单样本流式更新
 * - 经验回放缓冲区（稳定训练）
 * - 弹性权重巩固（EWC）知识保留
 * - 运行统计归一化
 * - 概念漂移检测
 *
 * @param trainer 训练器
 * @param inputs 输入数据（样本数 x 输入维度）
 * @param targets 目标数据（样本数 x 输出维度）
 * @param num_samples 样本数
 * @param callback 训练回调函数（可选）
 * @param user_data 回调函数用户数据（可选）
 * @return int 成功返回0，失败返回-1
 */
int trainer_train_online(Trainer* trainer, const float* inputs, const float* targets,
                         size_t num_samples, TrainingCallback callback, void* user_data);

/**
 * @brief 重置在线学习状态
 *
 * 清空经验回放缓冲区、重置运行统计和EWC状态，
 * 使训练器可以开始新的在线学习会话。
 *
 * @param trainer 训练器
 * @return int 成功返回0，失败返回-1
 */
int trainer_online_reset(Trainer* trainer);

/**
 * @brief 获取概念漂移检测状态
 *
 * @param trainer 训练器
 * @return int 1=检测到概念漂移，0=未检测到，-1=参数无效
 */
int trainer_online_drift_detected(Trainer* trainer);

/**
 * @brief 获取经验回放缓冲区统计
 *
 * @param trainer 训练器
 * @param size 输出缓冲区当前大小
 * @param capacity 输出缓冲区容量
 * @return int 成功返回0，失败返回-1
 */
int trainer_online_replay_stats(Trainer* trainer, size_t* size, size_t* capacity);

/**
 * @brief 设置概念漂移检测阈值
 *
 * @param trainer 训练器
 * @param threshold 漂移检测阈值（建议0.1~0.5）
 * @return int 成功返回0，失败返回-1
 */
int trainer_online_set_drift_threshold(Trainer* trainer, float threshold);

/**
 * @brief 在线学习单步更新（实时AGI专用）
 *
 * 针对CfC液态神经网络的实时单样本在线更新。
 * 结合经验回放、EWC持续学习和概念漂移检测。
 * 专为实时AGI系统的逐个样本学习场景设计。
 *
 * @param trainer 训练器
 * @param input 单个输入样本
 * @param target 单个目标样本
 * @param input_dim 输入维度
 * @param output_dim 输出维度
 * @param loss 输出当前样本损失
 * @return int 成功返回0，失败返回-1
 */
int trainer_online_step(Trainer* trainer, const float* input, const float* target,
                        size_t input_dim, size_t output_dim, float* loss);

/**
 * @brief 验证神经网络
 * 
 * @param trainer 训练器
 * @param inputs 输入数据
 * @param targets 目标数据
 * @param num_samples 样本数
 * @param loss 输出损失值
 * @param accuracy 输出准确率
 * @return int 成功返回0，失败返回-1
 */
int trainer_validate(Trainer* trainer, const float* inputs, const float* targets,
                     size_t num_samples, float* loss, float* accuracy);

/**
 * @brief 预测
 * 
 * @param trainer 训练器
 * @param inputs 输入数据
 * @param outputs 输出数据缓冲区
 * @param num_samples 样本数
 * @return int 成功返回0，失败返回-1
 */
int trainer_predict(Trainer* trainer, const float* inputs, float* outputs,
                    size_t num_samples);

/**
 * @brief 获取训练历史
 * 
 * @param trainer 训练器
 * @return TrainingHistory* 训练历史指针
 */
TrainingHistory* trainer_get_history(Trainer* trainer);

/**
 * @brief 设置训练器当前训练阶段名称
 * 
 * 在每次训练前调用，指定当前训练阶段（如"预训练"、"微调"、"持续训练"等），
 * 该阶段名称会被记录到TrainingHistory的train_modes字段中。
 * 
 * @param trainer 训练器
 * @param phase_name 阶段名称（长度不超过31字符）
 */
void trainer_set_current_training_phase(Trainer* trainer, const char* phase_name);

/**
 * @brief 保存训练历史到文件
 * 
 * @param history 训练历史
 * @param filename 文件名
 * @return int 成功返回0，失败返回-1
 */
int training_history_save(const TrainingHistory* history, const char* filename);

/**
 * @brief 从文件加载训练历史
 * 
 * @param filename 文件名
 * @return TrainingHistory* 训练历史指针，失败返回NULL
 */
TrainingHistory* training_history_load(const char* filename);

/**
 * @brief 释放训练历史
 * 
 * @param history 训练历史
 */
void training_history_free(TrainingHistory* history);

/**
 * @brief 获取默认训练配置
 * 
 * @return TrainingConfig 默认训练配置
 */
TrainingConfig training_config_default(void);

/**
 * @brief 打印训练配置
 * 
 * @param config 训练配置
 */
void training_config_print(const TrainingConfig* config);

/**
 * @brief 打印训练状态
 * 
 * @param state 训练状态
 */
void training_state_print(const TrainingState* state);

/**
 * @brief 学习率调度器类型
 */
typedef enum {
    SCHEDULER_CONSTANT = 0,     /**< 恒定学习率 */
    SCHEDULER_STEP = 1,         /**< 阶梯衰减 */
    SCHEDULER_EXPONENTIAL = 2,  /**< 指数衰减 */
    SCHEDULER_COSINE = 3,       /**< 余弦衰减 */
    SCHEDULER_CYCLIC = 4,        /**< 循环学习率 */
    SCHEDULER_PLATEAU = 5       /**< 指标停滞衰减（ReduceLROnPlateau） */
} LearningRateSchedulerType;

/**
 * @brief 学习率调度器配置
 */
typedef struct {
    LearningRateSchedulerType type; /**< 调度器类型 */
    float base_learning_rate;       /**< 基础学习率 */
    float max_learning_rate;        /**< 最大学习率（用于循环调度器） */
    float min_learning_rate;        /**< 最小学习率 */
    float decay_rate;               /**< 衰减率 */
    size_t decay_steps;             /**< 衰减步数 */
    size_t step_size;               /**< 步长（用于阶梯衰减） */
    size_t cycle_length;            /**< 循环长度（用于循环调度器） */
    float plateau_factor;           /**< 停滞衰减因子（用于Plateau调度器，默认0.1） */
    size_t plateau_patience;        /**< 停滞耐心轮数（默认10） */
    float plateau_threshold;        /**< 改进阈值（默认1e-4） */
    size_t plateau_cooldown;        /**< 衰减后冷却步数（默认0） */
    float min_learning_rate_abs;    /**< 绝对最小学习率下限 */
} LearningRateSchedulerConfig;

/**
 * @brief 创建学习率调度器
 * 
 * @param config 调度器配置
 * @param total_steps 总训练步数
 * @return void* 调度器句柄，失败返回NULL
 */
void* learning_rate_scheduler_create(const LearningRateSchedulerConfig* config,
                                     size_t total_steps);

/**
 * @brief 获取当前学习率
 * 
 * @param scheduler 调度器句柄
 * @param step 当前步数
 * @return float 当前学习率
 */
float learning_rate_scheduler_get(void* scheduler, size_t step);

/**
 * @brief ReduceLROnPlateau：基于指标评估调整学习率
 * 
 * @param scheduler 调度器句柄
 * @param metric 当前评估指标值（如验证损失）
 * @param step 当前步数
 * @return float 当前学习率
 */
float learning_rate_scheduler_on_plateau(void* scheduler, float metric, size_t step);

/**
 * @brief 释放学习率调度器
 * 
 * @param scheduler 调度器句柄
 */
void learning_rate_scheduler_free(void* scheduler);

/**
 * @brief 梯度裁剪
 * 
 * @param gradients 梯度数组
 * @param num_gradients 梯度数量
 * @param clip_type 裁剪类型
 * @param clip_value 裁剪值
 * @param clip_norm 裁剪范数
 */
void gradient_clip(float* gradients, size_t num_gradients,
                   GradientClipType clip_type, float clip_value, float clip_norm);

/**
 * @brief 计算梯度范数
 * 
 * @param gradients 梯度数组
 * @param num_gradients 梯度数量
 * @return float 梯度范数
 */
float gradient_norm(const float* gradients, size_t num_gradients);

/**
 * @brief 权重衰减
 * 
 * @param weights 权重数组
 * @param num_weights 权重数量
 * @param lambda 衰减系数
 * @param regularization_type 正则化类型
 */
void weight_decay(float* weights, size_t num_weights,
                  float lambda, RegularizationType regularization_type);

/**
 * @brief Dropout正则化
 * 
 * @param activations 激活值数组
 * @param num_activations 激活值数量
 * @param dropout_rate Dropout率
 * @param is_training 是否在训练模式
 * @param mask 输出掩码（可选）
 */
void dropout(float* activations, size_t num_activations,
             float dropout_rate, int is_training, float* mask);

/* ========================================================================
 * P1-002: 梯度验证 (Gradient Checking)
 * 通过有限差分数值梯度验证解析梯度的正确性
 * ======================================================================== */

/**
 * @brief 梯度验证结果
 */
typedef struct {
    float max_relative_error;        /**< 最大相对误差 */
    float avg_relative_error;        /**< 平均相对误差 */
    float max_absolute_error;        /**< 最大绝对误差 */
    float* per_parameter_error;      /**< 每个参数的相对误差（需调用方释放） */
    size_t num_parameters;           /**< 参数数量 */
    int num_mismatches;              /**< 高误差参数数量（相对误差 > threshold） */
    int passed;                      /**< 验证是否通过（1=通过, 0=失败） */
    float threshold;                 /**< 使用的相对误差阈值 */
} GradientCheckResult;

/**
 * @brief 执行梯度验证
 * 对指定批次数据，用有限差分计算数值梯度，与解析梯度比较
 *
 * @param trainer 训练器句柄
 * @param inputs 输入数据 (batch_size × input_dim)
 * @param targets 目标数据 (batch_size × output_dim)
 * @param num_samples 样本数
 * @param threshold 相对误差阈值（建议1e-4到1e-3），超过此值标记为不匹配
 * @param epsilon 有限差分步长（建议1e-6到1e-5）
 * @param result 输出验证结果（调用方需调用gradient_check_result_free释放内部缓冲区）
 * @return int 成功返回0，失败返回-1
 */
int trainer_gradient_check(Trainer* trainer, const float* inputs, const float* targets,
                           size_t num_samples, float threshold, float epsilon,
                           GradientCheckResult* result);

/**
 * @brief 释放梯度验证结果中的动态缓冲区
 */
void gradient_check_result_free(GradientCheckResult* result);

/* ========================================================================
 * P2-001: 梯度流健康度监控
 * 监控各层梯度范数、检测梯度消失/爆炸、辅助训练诊断
 * ======================================================================== */

/**
 * @brief 梯度流健康度报告
 */
typedef struct {
    float output_grad_norm;          /**< 输出层梯度范数 */
    float hidden_grad_norm;          /**< 隐藏层梯度范数 */
    float weight_grad_norm;          /**< 权重梯度范数 */
    float bias_grad_norm;            /**< 偏置梯度范数 */
    float gate_weight_grad_norm;     /**< 门控权重梯度范数 */
    float hidden_weight_grad_norm;   /**< 隐藏到隐藏权重梯度范数 */
    float grad_norm_ratio;           /**< 梯度范数比（weight_grad / output_grad），衡量梯度衰减 */
    int is_vanishing;                /**< 是否检测到梯度消失（各层梯度 < 1e-7） */
    int is_exploding;                /**< 是否检测到梯度爆炸（各层梯度 > 1e3） */
    int is_healthy;                  /**< 梯度流是否健康 */
    float vanishing_threshold;       /**< 梯度消失阈值 */
    float exploding_threshold;       /**< 梯度爆炸阈值 */
    float recommended_clip_norm;     /**< 建议的梯度裁剪范数 */
} GradientHealthReport;

/**
 * @brief 获取当前训练步骤的梯度流健康度报告
 * 从LNN网络内部提取各层梯度统计并分析健康状态
 *
 * @param trainer 训练器句柄
 * @param report 输出健康报告
 * @return int 成功返回0，失败返回-1
 */
int trainer_check_gradient_health(Trainer* trainer, GradientHealthReport* report);

/**
 * @brief 打印梯度流健康度报告到控制台
 */
void gradient_health_report_print(const GradientHealthReport* report);

/**
 * @brief 早停检查
 * 
 * @param current_loss 当前损失
 * @param best_loss 最佳损失
 * @param patience 耐心值
 * @param steps_without_improvement 没有改进的步数
 * @return int 应该停止返回1，否则返回0
 */
int early_stopping_check(float current_loss, float best_loss,
                         size_t patience, size_t* steps_without_improvement);

/**
 * @brief 模型检查点
 */
typedef struct {
    char filename[256];              /**< 文件名 */
    float loss;                      /**< 损失值 */
    float accuracy;                  /**< 准确率 */
    size_t epoch;                    /**< 轮数 */
    uint64_t timestamp;              /**< 时间戳 */
    size_t total_iterations;         /**< 总迭代次数 */
    size_t current_batch;            /**< 当前批次 */
    float learning_rate;             /**< 学习率 */
} ModelCheckpoint;

/**
 * @brief 训练统计信息
 */
typedef struct {
    float avg_train_loss;            /**< 平均训练损失 */
    float avg_val_loss;              /**< 平均验证损失 */
    float avg_train_accuracy;        /**< 平均训练准确率 */
    float avg_val_accuracy;          /**< 平均验证准确率 */
    float avg_gradient_norm;         /**< 平均梯度范数 */
    float avg_weight_norm;           /**< 平均权重范数 */
    uint64_t total_training_time_ms; /**< 总训练时间（毫秒） */
    size_t total_epochs_trained;     /**< 总训练轮数 */
    size_t total_batches_trained;    /**< 总训练批次数 */
} TrainingStats;

/**
 * @brief 保存模型检查点到文件
 * 
 * 使用原子写入机制确保数据完整性。
 * 
 * @param trainer 训练器
 * @param filename 检查点文件名
 * @return int 成功返回0，失败返回-1
 */
int save_model_checkpoint(const Trainer* trainer, const char* filename);

/**
 * @brief 从文件加载模型检查点
 * 
 * 自动恢复网络权重、训练状态和配置。
 * 执行CRC校验和验证以确保数据完整性。
 * 
 * @param trainer 训练器
 * @param filename 检查点文件名
 * @return int 成功返回0，失败返回-1
 */
int load_model_checkpoint(Trainer* trainer, const char* filename);

/**
 * @brief 设置自动检查点
 * 
 * 设置后，训练过程中每隔指定epoch数自动保存检查点。
 * 
 * @param trainer 训练器
 * @param checkpoint_path 检查点保存路径（为NULL时禁用）
 * @param interval_epochs 保存间隔（epoch数）
 * @return int 成功返回0，失败返回-1
 */
int trainer_set_auto_checkpoint(Trainer* trainer, const char* checkpoint_path, size_t interval_epochs);

/**
 * @brief 从检查点恢复训练
 * 
 * 自动加载检查点并设置自动保存路径用于后续保存。
 * 
 * @param trainer 训练器
 * @param checkpoint_path 检查点文件路径
 * @return int 成功返回0，失败返回-1
 */
int trainer_resume_from_checkpoint(Trainer* trainer, const char* checkpoint_path);

/**
 * @brief 创建训练数据加载器
 * 
 * @param inputs 输入数据（形状为num_samples × input_dim）
 * @param targets 目标数据（形状为num_samples × output_dim）
 * @param num_samples 样本数
 * @param input_dim 输入维度
 * @param output_dim 输出维度
 * @param batch_size 批量大小
 * @param shuffle 是否打乱数据
 * @return void* 数据加载器句柄，失败返回NULL
 */
void* data_loader_create(const float* inputs, const float* targets,
                         size_t num_samples, size_t input_dim, size_t output_dim,
                         size_t batch_size, int shuffle);

/**
 * @brief 获取下一个批次
 * 
 * @param loader 数据加载器句柄
 * @param batch_inputs 输出批次输入数据
 * @param batch_targets 输出批次目标数据
 * @param batch_size 输出批次大小
 * @return int 成功返回1，没有更多数据返回0，失败返回-1
 */
int data_loader_next_batch(void* loader, float** batch_inputs,
                           float** batch_targets, size_t* batch_size);

/**
 * @brief 重置数据加载器
 * 
 * @param loader 数据加载器句柄
 */
void data_loader_reset(void* loader);

/**
 * @brief 释放数据加载器
 * 
 * @param loader 数据加载器句柄
 */
void data_loader_free(void* loader);

/**
 * @brief 交叉验证
 * 
 * @param trainer 训练器
 * @param inputs 输入数据
 * @param targets 目标数据
 * @param num_samples 样本数
 * @param k_folds 折数
 * @param results 输出结果数组（大小为k_folds）
 * @return int 成功返回0，失败返回-1
 */
int cross_validation(Trainer* trainer, const float* inputs, const float* targets,
                     size_t num_samples, size_t k_folds, float* results);

/**
 * @brief 超参数搜索模式
 */
typedef enum {
    HYPERPARAM_SEARCH_GRID = 0,     /**< 网格搜索（全部组合穷举） */
    HYPERPARAM_SEARCH_RANDOM = 1,   /**< 随机搜索（随机采样N组） */
    HYPERPARAM_SEARCH_BAYESIAN = 2  /**< 贝叶斯优化（高斯过程代理模型） */
} HyperparamSearchMode;

/**
 * @brief 超参数搜索配置
 */
typedef struct {
    float learning_rate_min;          /**< 学习率最小值 */
    float learning_rate_max;          /**< 学习率最大值 */
    size_t learning_rate_steps;       /**< 学习率搜索步数（网格模式） */
    
    float regularization_min;         /**< 正则化强度最小值 */
    float regularization_max;         /**< 正则化强度最大值 */
    size_t regularization_steps;      /**< 正则化强度搜索步数（网格模式） */
    
    size_t batch_size_min;            /**< 批量大小最小值 */
    size_t batch_size_max;            /**< 批量大小最大值 */
    size_t batch_size_steps;          /**< 批量大小搜索步数（网格模式） */
    
    size_t max_iterations;            /**< 最大迭代次数 */
    size_t cv_folds;                  /**< 交叉验证折数 */
    
    // ---- 需求20.2: 增强搜索配置 ----
    HyperparamSearchMode search_mode; /**< 搜索模式 */
    size_t num_random_trials;         /**< 随机搜索试验次数（随机/贝叶斯模式） */
    int enable_early_stopping;        /**< 是否启用早停（每个试验内） */
    size_t early_stop_patience;       /**< 早停耐心值（试验内验证轮数） */
    float early_stop_min_delta;       /**< 早停最小改善阈值 */
    int verbose;                      /**< 是否输出搜索过程日志 */
} HyperparameterSearchConfig;

/**
 * @brief 超参数搜索结果（增强版）
 */
typedef struct {
    float best_learning_rate;         /**< 最佳学习率 */
    float best_regularization;        /**< 最佳正则化强度 */
    size_t best_batch_size;           /**< 最佳批量大小 */
    float best_score;                 /**< 最佳得分 */
    float* all_scores;                /**< 所有得分 */
    size_t num_scores;                /**< 得分数量 */
    size_t num_combinations;          /**< 总组合数 */
    
    // ---- 需求20.2: 增强搜索结果 ----
    size_t trials_evaluated;          /**< 实际评估的试验次数 */
    size_t trials_early_stopped;      /**< 早停跳过的试验次数 */
    float search_duration_sec;        /**< 搜索耗时（秒） */
    float* learning_rate_history;     /**< 各次试验的学习率历史 */
    float* regularization_history;    /**< 各次试验的正则化历史 */
    size_t* batch_size_history;       /**< 各次试验的批量大小历史 */
    size_t history_size;              /**< 历史记录条数 */
} HyperparameterSearchResult;

/**
 * @brief 超参数搜索
 * 
 * @param trainer 训练器
 * @param inputs 输入数据
 * @param targets 目标数据
 * @param num_samples 样本数
 * @param config 搜索配置
 * @param result 输出搜索结果
 * @return int 成功返回0，失败返回-1
 */
int hyperparameter_search(Trainer* trainer, const float* inputs, const float* targets,
                          size_t num_samples, const HyperparameterSearchConfig* config,
                          HyperparameterSearchResult* result);

/**
 * @brief 释放超参数搜索结果
 * 
 * @param result 搜索结果
 */
void hyperparameter_search_result_free(HyperparameterSearchResult* result);

/**
 * @brief 训练停滞检测结果
 */
typedef struct {
    int is_stagnated;                    /**< 是否停滞 */
    float loss_improvement_rate;         /**< 损失改善率（正数=仍在下降, 负数=上升, 接近0=停滞） */
    float accuracy_improvement_rate;     /**< 准确率改善率 */
    float gradient_stability;            /**< 梯度稳定性（0-1, 1=最稳定） */
    size_t epochs_since_best_loss;       /**< 自最佳损失以来的轮数 */
    size_t epochs_since_best_accuracy;   /**< 自最佳准确率以来的轮数 */
    int recommended_action;              /**< 建议动作: 0=继续, 1=降低学习率, 2=增加正则化, 3=全面搜索 */
    char description[256];               /**< 停滞检测描述 */
} StagnationDetectionResult;

/**
 * @brief 检测训练是否停滞
 * 
 * 分析训练历史中的损失和准确率趋势，检测训练是否陷入停滞或发散。
 * 使用滑动窗口线性回归和指数平滑计算改善率，结合梯度稳定性做综合判断。
 * 
 * @param trainer 训练器
 * @param window_size 滑动窗口大小（至少2）
 * @param min_improvement 最小改善阈值（如0.001表示千分之一）
 * @param result 输出检测结果
 * @return int 成功返回0，失败返回-1
 */
int trainer_detect_stagnation(Trainer* trainer, size_t window_size, float min_improvement,
                               StagnationDetectionResult* result);

/**
 * @brief 自动超参数调整（完整自我修正版）
 * 
 * 检测训练停滞时自动执行超参数搜索，应用最佳超参数配置。
 * 支持三种模式：
 *   0 = 仅降低学习率（轻微停滞）
 *   1 = 学习率+正则化联合搜索（中度停滞）
 *   2 = 全面搜索学习率+正则化+批量大小（严重停滞）
 * 
 * @param trainer 训练器
 * @param inputs 输入数据
 * @param targets 目标数据
 * @param num_samples 样本数
 * @param auto_mode 自动模式（0=轻, 1=中, 2=全面）
 * @param improvement 输出改善幅度（最佳分数-原分数, 可选）
 * @return int 成功返回0（无需调整也返回0）, 失败返回-1
 */
int trainer_auto_tune_hyperparameters(Trainer* trainer,
                                       const float* inputs, const float* targets,
                                       size_t num_samples, int auto_mode,
                                       float* improvement);

/**
 * @brief 拉普拉斯优化训练函数
 * 
 * 使用拉普拉斯变换优化训练过程，提供频域梯度滤波和稳定性分析。
 * 
 * @param trainer 训练器
 * @param inputs 输入数据
 * @param targets 目标数据
 * @param num_samples 样本数
 * @param callback 训练回调函数（可选）
 * @param user_data 回调函数用户数据（可选）
 * @return int 成功返回0，失败返回-1
 */
int trainer_train_with_laplace(Trainer* trainer, const float* inputs, const float* targets,
                               size_t num_samples, TrainingCallback callback, void* user_data);

/**
 * @brief 获取拉普拉斯优化状态
 * 
 * @param trainer 训练器
 * @param stability_score 输出稳定性分数（0.0-1.0）
 * @param current_cutoff 输出当前滤波截止频率
 * @param stability_warning 输出稳定性警告标志
 * @return int 成功返回0，失败返回-1
 */
int trainer_get_laplace_status(Trainer* trainer, float* stability_score,
                               float* current_cutoff, int* stability_warning);

/**
 * @brief 启用或禁用拉普拉斯优化
 * 
 * @param trainer 训练器
 * @param enable 是否启用
 * @return int 成功返回0，失败返回-1
 */
int trainer_set_laplace_enabled(Trainer* trainer, int enable);

/**
 * @brief 设置拉普拉斯滤波器参数
 * 
 * @param trainer 训练器
 * @param cutoff_freq 截止频率（Hz）
 * @param stability_margin 稳定性裕度阈值
 * @return int 成功返回0，失败返回-1
 */
int trainer_set_laplace_parameters(Trainer* trainer, float cutoff_freq, float stability_margin);

/**
 * @brief 分析训练稳定性
 * 
 * 使用拉普拉斯变换分析训练过程的稳定性，返回稳定性指标。
 * 
 * @param trainer 训练器
 * @param stability_metrics 输出稳定性指标
 * @return int 成功返回0，失败返回-1
 */
int trainer_analyze_stability(Trainer* trainer, StabilityMetrics* stability_metrics);

/**
 * @brief 获取训练器的神经网络
 */
LNN* trainer_get_network(Trainer* trainer);

/**
 * @brief 获取训练状态
 *
 * @param trainer 训练器句柄
 * @return TrainingState* 训练状态指针，失败返回NULL
 */
TrainingState* trainer_get_state(Trainer* trainer);

/**
 * @brief 设置训练状态
 *
 * @param trainer 训练器句柄
 * @param state 训练状态指针
 * @return int 成功返回0，失败返回-1
 */
int trainer_set_state(Trainer* trainer, const TrainingState* state);

/**
 * @brief 暂停训练
 *
 * @param trainer 训练器句柄
 * @return int 成功返回0，失败返回-1
 */
int trainer_pause(Trainer* trainer);

/**
 * @brief 恢复训练
 *
 * @param trainer 训练器句柄
 * @return int 成功返回0，失败返回-1
 */
int trainer_resume(Trainer* trainer);

/**
 * @brief 停止训练
 *
 * @param trainer 训练器句柄
 * @return int 成功返回0，失败返回-1
 */
int trainer_stop(Trainer* trainer);

/**
 * @brief 获取训练配置
 *
 * @param trainer 训练器句柄
 * @return TrainingConfig* 训练配置指针，失败返回NULL
 */
TrainingConfig* trainer_get_config(Trainer* trainer);

/**
 * @brief 获取训练器的混合精度上下文
 */
MixedPrecisionContext* trainer_get_mixed_precision_context(Trainer* trainer);

/**
 * @brief 设置训练器的混合精度上下文
 */
int trainer_set_mixed_precision_context(Trainer* trainer, MixedPrecisionContext* context);

/**
 * @brief 加载预训练权重
 * 
 * @param trainer 训练器
 * @param weights_path 权重文件路径
 * @return int 成功返回0，失败返回-1
 */
int trainer_load_pretrained_weights(Trainer* trainer, const char* weights_path);

/**
 * @brief 保存模型权重
 * 
 * @param trainer 训练器
 * @param weights_path 权重文件路径
 * @return int 成功返回0，失败返回-1
 */
int trainer_save_model_weights(Trainer* trainer, const char* weights_path);

/**
 * @brief 配置微调训练
 * 
 * @param trainer 训练器
 * @param freeze_base 是否冻结基础层
 * @param fine_tune_lr 微调学习率（如果为0，则使用原学习率）
 * @return int 成功返回0，失败返回-1
 */
int trainer_configure_fine_tuning(Trainer* trainer, int freeze_base, float fine_tune_lr);

/**
 * @brief 执行迁移学习
 * 
 * @param trainer 训练器
 * @param source_network 源网络（预训练网络）
 * @param transfer_layers 要迁移的层索引数组
 * @param num_layers 层数
 * @return int 成功返回0，失败返回-1
 */
int trainer_perform_transfer_learning(Trainer* trainer, LNN* source_network, 
                                      const int* transfer_layers, size_t num_layers);

/**
 * @brief 应用高级正则化（CutMix/MixUp/DropPath等）
 * 
 * @param trainer 训练器
 * @param inputs 输入数据
 * @param targets 目标数据
 * @param batch_size 批量大小
 * @param input_dim 输入维度
 * @param output_dim 输出维度
 * @param augmented_inputs 增强后输入缓冲区
 * @param augmented_targets 增强后目标缓冲区
 * @param epoch 当前训练轮数
 * @param total_epochs 总训练轮数
 * @return int 成功返回0，失败返回-1
 */
int trainer_apply_regularization(Trainer* trainer,
                                 const float* inputs, const float* targets,
                                 size_t batch_size, size_t input_dim, size_t output_dim,
                                 float* augmented_inputs, float* augmented_targets,
                                 size_t epoch, size_t total_epochs);

/**
 * @brief 生成对抗样本进行对抗训练
 * 
 * @param trainer 训练器
 * @param inputs 原始输入
 * @param targets 目标标签
 * @param batch_size 批量大小
 * @param input_dim 输入维度
 * @param adversarial_inputs 对抗样本输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int trainer_generate_adversarial(Trainer* trainer,
                                  const float* inputs, const float* targets,
                                  size_t batch_size, size_t input_dim,
                                  float* adversarial_inputs);

/**
 * @brief 获取高级正则化器句柄
 * 
 * @param trainer 训练器
 * @return AdvancedRegularizer* 正则化器句柄，未初始化返回NULL
 */
AdvancedRegularizer* trainer_get_regularizer(Trainer* trainer);

/**
 * @brief 配置高级正则化
 * 
 * @param trainer 训练器
 * @param config 新的高级正则化配置
 * @return int 成功返回0，失败返回-1
 */
int trainer_configure_regularization(Trainer* trainer,
                                     const AdvancedRegularizationConfig* config);

/**
 * @brief 应用网络级别正则化（DropConnect + 空间Dropout + 可切换归一化）
 *
 * 在训练过程中对网络权重和中间激活应用高级正则化技术。
 * 在每次前向传播前对权重应用DropConnect，在前向传播后对
 * 中间激活应用正则化。
 *
 * @param trainer 训练器
 * @param batch_inputs 当前批次输入
 * @param batch_size 批次大小
 * @param training 是否训练模式
 * @return int 成功返回0，失败返回-1
 */
int trainer_apply_network_regularization(Trainer* trainer,
                                         const float* batch_inputs,
                                         size_t batch_size,
                                         int training);

/**
 * @brief 应用域自适应训练
 *
 * 使用配置的域自适应方法（DANN、MMD、CORAL、ADDA）对源域和目标域
 * 数据进行域自适应训练。
 *
 * @param trainer 训练器
 * @param source_inputs 源域输入（带标签的训练数据）
 * @param target_inputs 目标域输入（无标签的部署数据）
 * @param batch_size 批量大小
 * @param input_dim 输入维度
 * @param domain_labels 输出域标签缓冲区（0=源域，1=目标域）
 * @param domain_loss 输出域损失值
 * @return int 成功返回0，失败返回-1
 */
int trainer_apply_domain_adaptation(Trainer* trainer,
                                    const float* source_inputs,
                                    const float* target_inputs,
                                    size_t batch_size, size_t input_dim,
                                    float* domain_labels,
                                    float* domain_loss);

/* ============================
 * 增强T7：GPU训练完整集成API
 * ============================ */

/**
 * @brief GPU训练配置结构
 */
typedef struct {
    int enable_gpu;                    /**< 启用GPU训练 */
    int device_id;                     /**< GPU设备ID */
    int use_mixed_precision;           /**< 启用混合精度 */
    int mixed_precision_mode;          /**< 混合精度模式 */
    size_t gpu_batch_size;             /**< GPU专用批量大小（0=使用CPU批量大小） */
    int enable_tensor_cores;           /**< 启用Tensor Core */
    int enable_gradient_checkpointing; /**< 启用梯度检查点 */
    int enable_distributed;            /**< 启用分布式训练 */
    size_t num_gpu_devices;            /**< GPU设备数量（分布式中使用） */
    int sync_batch_norm;               /**< 同步批归一化 */
} GpuTrainingConfig;

#define GPU_TRAINING_CONFIG_DEFAULT { \
    .enable_gpu = 0, \
    .device_id = 0, \
    .use_mixed_precision = 0, \
    .mixed_precision_mode = 0, \
    .gpu_batch_size = 0, \
    .enable_tensor_cores = 1, \
    .enable_gradient_checkpointing = 0, \
    .enable_distributed = 0, \
    .num_gpu_devices = 1, \
    .sync_batch_norm = 0 \
}

/**
 * @brief 启用/禁用GPU训练
 * 
 * 动态切换训练器的GPU训练状态。启用时，后续trainer_train调用将使用GPU。
 * 此函数会同步训练器内部的use_gpu和use_mixed_precision配置。
 *
 * @param trainer 训练器句柄
 * @param enable 启用标志（1=启用GPU，0=禁用GPU）
 * @return int 成功返回0，失败返回-1
 */
int trainer_enable_gpu(Trainer* trainer, int enable);

/**
 * @brief 获取当前GPU训练配置
 * 
 * 返回训练器的当前GPU训练配置副本，包括设备ID、混合精度状态等。
 *
 * @param trainer 训练器句柄
 * @param config 输出GPU训练配置结构体
 * @return int 成功返回0，失败返回-1
 */
int trainer_get_gpu_config(Trainer* trainer, GpuTrainingConfig* config);

/**
 * @brief 设置GPU训练配置
 * 
 * 批量设置GPU训练参数。传入NULL的字段将被忽略。
 * 设置后会同步到训练器的内部配置。
 *
 * @param trainer 训练器句柄
 * @param config GPU训练配置
 * @return int 成功返回0，失败返回-1
 */
int trainer_set_gpu_config(Trainer* trainer, const GpuTrainingConfig* config);

/**
 * @brief 获取GPU上下文指针
 * 
 * 返回训练器的内部GPU上下文指针，供底层GPU操作使用。
 * 返回的指针类型为void*，实际为GpuContext*。
 *
 * @param trainer 训练器句柄
 * @return void* GPU上下文指针，失败返回NULL
 */
void* trainer_get_gpu_context(Trainer* trainer);

/* ============================
 * 增强C03：记忆系统集成API
 * ============================ */

/**
 * @brief 连接外部记忆管理器到训练器
 *
 * 将已创建的MemoryManager实例连接到训练器，使训练过程能够
 * 利用记忆系统的经验存储、检索和巩固功能。
 *
 * @param trainer 训练器句柄
 * @param manager 外部记忆管理器句柄
 * @return int 成功返回0，失败返回-1
 */
int trainer_connect_memory(Trainer* trainer, MemoryManager* manager);

/**
 * @brief 为训练器创建内部记忆管理器
 *
 * 在训练器内部创建并初始化记忆管理器，自动配置短期/长期记忆容量。
 * 训练器销毁时自动释放内部记忆管理器资源。
 *
 * @param trainer 训练器句柄
 * @param short_term_capacity 短期记忆容量
 * @param long_term_capacity 长期记忆容量
 * @return int 成功返回0，失败返回-1
 */
int trainer_create_memory(Trainer* trainer, size_t short_term_capacity, size_t long_term_capacity);

/**
 * @brief 保存训练经验到记忆系统
 *
 * 将当前训练样本的输入、目标、损失和准确率作为经验保存到记忆系统。
 * 自动生成唯一键值，支持短期和长期记忆存储。
 *
 * @param trainer 训练器句柄
 * @param input 输入数据
 * @param target 目标数据
 * @param input_dim 输入维度
 * @param output_dim 输出维度
 * @param loss 损失值
 * @param accuracy 准确率
 * @return int 成功返回0，失败返回-1
 */
int trainer_memory_save_experience(Trainer* trainer, const float* input, const float* target,
                                    size_t input_dim, size_t output_dim, float loss, float accuracy);

/**
 * @brief 从记忆系统召回相似经验
 *
 * 基于当前输入从记忆系统中召回相似的经验，生成记忆上下文向量。
 * 召回的上下文通过记忆上下文缓冲区传递给训练过程，用于调制
 * 液态神经网络的前向传播。
 *
 * @param trainer 训练器句柄
 * @param context 当前输入上下文（用于相似度匹配）
 * @param context_dim 上下文维度
 * @param recalled_data 召回数据输出缓冲区
 * @param num_recalled 输出实际召回数量
 * @param max_recalled 最大召回数量
 * @return int 成功返回0，失败返回-1
 */
int trainer_memory_recall_similar(Trainer* trainer, const float* context, size_t context_dim,
                                   float* recalled_data, size_t* num_recalled, size_t max_recalled);

/**
 * @brief 触发记忆巩固
 *
 * 将短期记忆中的经验巩固到长期记忆，整合相关记忆条目，
 * 压缩语义记忆以提高存储效率。
 *
 * @param trainer 训练器句柄
 * @return int 成功返回0，失败返回-1
 */
int trainer_memory_consolidate(Trainer* trainer);

/**
 * @brief 获取记忆管理器句柄
 *
 * @param trainer 训练器句柄
 * @return MemoryManager* 记忆管理器句柄，未配置返回NULL
 */
MemoryManager* trainer_get_memory_manager(Trainer* trainer);

/**
 * @brief 从记忆系统采样数据进行训练（完整经验回放闭环）
 *
 * 直接从记忆系统中采样经验数据对LNN进行训练，形成
 * "经验积累→记忆存储→采样回放→LNN训练"的完整AGI学习闭环。
 * 每个批次从指定类型的记忆中随机采样，使用memory_sample_training_batch()
 * 获取输入/目标对（记忆项存储格式：[input_vector | target_vector]），
 * 然后执行完整的前向传播→损失计算→反向传播→参数更新循环。
 * 训练完成时会自动触发记忆巩固，将新学到的知识整合回记忆系统。
 * 
 * 此函数与trainer_train()的区别：
 * - trainer_train()：使用外部提供的输入/目标数组进行训练
 * - trainer_train_from_memory()：直接从内部记忆系统采样进行训练，
 *   适合持续学习/经验回放场景，无需外部数据源
 *
 * @param trainer 训练器句柄
 * @param mem_system 记忆系统句柄（通过memory_manager_get_system()获取）
 * @param memory_type 采样使用的记忆类型（MEMORY_TYPE_SHORT_TERM等）
 * @param batch_size 每个批次的采样大小
 * @param data_dim 数据特征维度（需与记忆项存储维度匹配）
 * @param num_batches 训练的批次数（0表示自动根据可用记忆量决定）
 * @param callback 训练回调函数（可选）
 * @param user_data 回调函数用户数据（可选）
 * @return int 成功返回0，失败返回-1
 */
int trainer_train_from_memory(Trainer* trainer, MemorySystem* mem_system,
                              MemoryType memory_type, size_t batch_size,
                              size_t data_dim, size_t num_batches,
                              TrainingCallback callback, void* user_data);

/**
 * @brief 初始化并连接模型版本管理器到训练器
 *
 * 创建模型版本管理器实例并与训练器关联。
 * 启用后可自动创建训练过程中的版本快照。
 *
 * @param trainer 训练器句柄
 * @param versions_dir 版本存储目录（NULL使用默认"model_versions"）
 * @param max_versions 最大保留版本数（0表示不限制）
 * @return int 成功返回0，失败返回-1
 */
int trainer_init_version_manager(Trainer* trainer, const char* versions_dir, size_t max_versions);

/**
 * @brief 设置自动快照
 *
 * 启用或禁用训练过程中的自动版本快照功能。
 *
 * @param trainer 训练器句柄
 * @param enabled 是否启用
 * @param interval_epochs 快照间隔（epoch数）
 * @return int 成功返回0，失败返回-1
 */
int trainer_set_auto_version_snapshot(Trainer* trainer, int enabled, size_t interval_epochs);

/**
 * @brief 手动创建版本快照
 *
 * 手动触发一次模型版本快照。
 *
 * @param trainer 训练器句柄
 * @param tag 版本标签
 * @param description 版本描述
 * @return ModelVersionID 成功返回版本ID，失败返回0
 */
ModelVersionID trainer_create_version_snapshot(Trainer* trainer, const char* tag, const char* description);

/**
 * @brief 回滚到指定版本
 *
 * 将模型恢复到指定版本的权重和训练状态。
 *
 * @param trainer 训练器句柄
 * @param version_id 目标版本ID
 * @return int 成功返回0，失败返回-1
 */
int trainer_rollback_to_version(Trainer* trainer, ModelVersionID version_id);

/**
 * @brief 获取版本管理器
 *
 * 获取训练器关联的模型版本管理器。
 *
 * @param trainer 训练器句柄
 * @return ModelVersionManager* 成功返回管理器指针，失败返回NULL
 */
ModelVersionManager* trainer_get_version_manager(const Trainer* trainer);

/* ============================
 * P1-6: 分布式训练容错和恢复增强API
 * ============================ */

/**
 * @brief 启用或禁用弹性训练（动态加减节点）
 *
 * 弹性训练允许在分布式训练过程中动态添加或移除训练节点，
 * 无需停止整个训练过程。启用后，系统会自动在心跳检测期间
 * 处理待添加/移除节点队列。
 *
 * @param trainer 训练器句柄
 * @param enable 是否启用弹性训练
 * @return int 成功返回0，失败返回-1
 */
int trainer_distributed_set_elastic_enabled(Trainer* trainer, int enable);

/**
 * @brief 动态添加训练节点到弹性训练集群
 *
 * 将新节点ID加入待添加队列，在下一个心跳检测周期中
 * 自动完成节点添加、心跳数组扩容和工作负载重平衡。
 *
 * @param trainer 训练器句柄
 * @param new_node_id 新节点ID（必须大于当前最大节点ID）
 * @return int 成功返回0，失败返回-1
 */
int trainer_distributed_elastic_add_node(Trainer* trainer, int new_node_id);

/**
 * @brief 动态从弹性训练集群移除节点
 *
 * 将节点ID加入待移除队列，在下一个心跳检测周期中
 * 自动完成节点移除、拓扑重建和工作负载重平衡。
 *
 * @param trainer 训练器句柄
 * @param remove_node_id 要移除的节点ID
 * @return int 成功返回0，失败返回-1
 */
int trainer_distributed_elastic_remove_node(Trainer* trainer, int remove_node_id);

/**
 * @brief 获取当前领导者节点ID
 *
 * 在启用领导者选举的情况下，获取当前被选举为领导者的节点ID。
 * 领导者负责协调分布式训练中的检查点保存和工作负载分配。
 *
 * @param trainer 训练器句柄
 * @return int 成功返回领导者节点ID，失败返回-1
 */
int trainer_distributed_get_leader_id(Trainer* trainer);

/**
 * @brief 设置自动恢复检查点路径
 *
 * 配置训练器在分布式节点故障且恢复尝试耗尽时，
 * 自动从指定的检查点文件加载参数恢复训练。
 * 路径为NULL时禁用自动恢复功能。
 *
 * @param trainer 训练器句柄
 * @param checkpoint_path 检查点文件路径（为NULL时禁用自动恢复）
 * @return int 成功返回0，失败返回-1
 */
int trainer_distributed_set_auto_resume(Trainer* trainer, const char* checkpoint_path);

/**
 * @brief 启用或禁用过时梯度处理
 *
 * 过时梯度处理在梯度同步前检查各节点的梯度是否过时，
 * 对过时梯度应用衰减系数（系数 = 1/(1 + 过时步数*0.5)，最小0.1），
 * 减少陈旧梯度对参数更新的负面影响。
 *
 * @param trainer 训练器句柄
 * @param enable 是否启用过时梯度处理
 * @return int 成功返回0，失败返回-1
 */
int trainer_distributed_set_stale_gradient_enabled(Trainer* trainer, int enable);

/* ============================
 * 需求20.1: 分布式训练容错增强 API
 * ============================ */

/**
 * @brief 启用或禁用法定人数共识
 *
 * 法定人数共识要求梯度同步前至少 majority (默认51%) 节点存活。
 * 不满足法定人数时梯度同步不会执行，防止使用不完整的梯度。
 *
 * @param trainer 训练器句柄
 * @param enable 是否启用
 * @return int 成功返回0，失败返回-1
 */
int trainer_distributed_set_quorum_enabled(Trainer* trainer, int enable);

/**
 * @brief 设置法定人数百分比阈值
 *
 * 设置需要多少百分比的节点存活才能执行梯度同步。
 * 例如 51 表示至少 51% 节点存活。
 *
 * @param trainer 训练器句柄
 * @param threshold_percent 百分比阈值 (1-100)
 * @return int 成功返回0，失败返回-1
 */
int trainer_distributed_set_quorum_threshold(Trainer* trainer, int threshold_percent);

/**
 * @brief 检查当前是否满足法定人数
 *
 * @param trainer 训练器句柄
 * @param quorum_met 输出：是否满足法定人数
 * @return int 成功返回0，失败返回-1
 */
int trainer_distributed_check_quorum(Trainer* trainer, int* quorum_met);

/**
 * @brief 启用或禁用梯度版本追踪
 *
 * 梯度版本追踪为每次梯度同步分配全局单调递增版本号，
 * 检测并标记过时梯度（版本落后超过1个的节点）。
 * 过时梯度在参数更新中会被衰减权重。
 *
 * @param trainer 训练器句柄
 * @param enable 是否启用
 * @return int 成功返回0，失败返回-1
 */
int trainer_distributed_set_gradient_versioning(Trainer* trainer, int enable);

/**
 * @brief 检查指定节点的梯度是否过时
 *
 * @param trainer 训练器句柄
 * @param node_id 节点ID
 * @return int 过时返回1，正常返回0，失败返回-1
 */
int trainer_distributed_is_node_stale(Trainer* trainer, int node_id);

/**
 * @brief 启用或禁用拓扑自动重平衡
 *
 * 自动检测分布式训练拓扑变化（节点加入/离开），
 * 并自动重平衡工作负载分配。
 *
 * @param trainer 训练器句柄
 * @param enable 是否启用
 * @return int 成功返回0，失败返回-1
 */
int trainer_distributed_set_auto_rebalance(Trainer* trainer, int enable);

/* ============================
 * 需求20.4: 检查点自动保存和恢复增强
 * ============================ */

/**
 * @brief 启用紧急检查点（信号处理）
 *
 * 注册 SIGINT/SIGTERM/信号处理函数，在收到中断信号时自动保存紧急检查点。
 * 防止训练过程中因意外中断导致进度丢失。
 *
 * @param trainer 训练器句柄
 * @param path 紧急检查点保存路径（为NULL时禁用）
 * @return int 成功返回0，失败返回-1
 */
int trainer_enable_emergency_checkpoint(Trainer* trainer, const char* path);

/**
 * @brief 启动时检查崩溃恢复
 *
 * 扫描指定目录下的紧急检查点文件，检测上次训练是否异常终止。
 * 如果发现紧急检查点且上次训练未正常完成，返回可恢复的信息。
 *
 * @param trainer 训练器句柄
 * @param checkpoint_dir 紧急检查点目录路径
 * @param recovered_epoch 输出上次训练到的轮数
 * @param recovered_loss 输出上次训练的损失值
 * @return int 找到可恢复的检查点返回1，未找到返回0，失败返回-1
 */
int trainer_check_crash_recovery(Trainer* trainer, const char* checkpoint_dir,
                                 size_t* recovered_epoch, float* recovered_loss);

/**
 * @brief 设置检查点保留策略
 *
 * 设置最多保留多少个历史检查点文件。超过数量时自动删除最旧的检查点。
 * 设置为0表示不限制保留数量。
 *
 * @param trainer 训练器句柄
 * @param max_checkpoints 最大保留检查点数量（0=不限制）
 * @return int 成功返回0，失败返回-1
 */
int trainer_set_checkpoint_retention(Trainer* trainer, size_t max_checkpoints);

/**
 * @brief 启用后台定时保存检查点
 *
 * 启动一个后台线程，每隔指定秒数自动保存检查点。
 * 后台保存不会阻塞训练循环。
 *
 * @param trainer 训练器句柄
 * @param interval_seconds 保存间隔（秒），必须大于0
 * @return int 成功返回0，失败返回-1
 */
int trainer_enable_background_checkpoint(Trainer* trainer, int interval_seconds);

/**
 * @brief 禁用后台定时保存检查点
 *
 * 停止后台保存线程。在 trainer_free 中会自动调用此函数。
 *
 * @param trainer 训练器句柄
 * @return int 成功返回0，失败返回-1
 */
int trainer_disable_background_checkpoint(Trainer* trainer);

/* ============================
 * 训练系统增强：权重初始化、课程学习、预训练、数据生成器
 * ============================ */

/**
 * @brief 权重初始化类型
 */
typedef enum {
    WEIGHT_INIT_RANDOM_UNIFORM = 0,   /**< 均匀随机初始化 */
    WEIGHT_INIT_XAVIER_UNIFORM = 1,   /**< Xavier均匀初始化 (Glorot) */
    WEIGHT_INIT_XAVIER_NORMAL = 2,    /**< Xavier正态初始化 */
    WEIGHT_INIT_HE_UNIFORM = 3,       /**< He均匀初始化 (Kaiming) */
    WEIGHT_INIT_HE_NORMAL = 4,        /**< He正态初始化 */
    WEIGHT_INIT_ORTHOGONAL = 5        /**< 正交初始化 */
} WeightInitType;

/**
 * @brief 初始化LNN网络所有权重
 *
 * 根据指定的初始化类型对网络中的所有权重参数进行初始化。
 * Xavier初始化适用于tanh/sigmoid激活，He初始化适用于ReLU族激活。
 * 正交初始化适用于深层网络和循环连接。
 *
 * @param network LNN网络句柄
 * @param type 初始化类型
 * @param seed 随机种子（-1=使用当前时间）
 * @return int 成功返回0，失败返回-1
 */
int lnn_weight_init(LNN* network, WeightInitType type, int seed);

/**
 * @brief 课程学习类型
 */
typedef enum {
    CURRICULUM_NONE = 0,               /**< 无课程学习 */
    CURRICULUM_DIFFICULTY_LEVEL = 1,   /**< 按难度级别（样本分组） */
    CURRICULUM_PROGRESSIVE_DIM = 2,    /**< 渐进维度扩展 */
    CURRICULUM_ADAPTIVE = 3            /**< 自适应难度（基于模型性能） */
} CurriculumType;

/**
 * @brief 课程学习配置
 */
typedef struct {
    CurriculumType type;                 /**< 课程学习类型 */
    size_t num_difficulty_levels;        /**< 难度级别数 */
    float* difficulty_thresholds;        /**< 各级别难度阈值数组 */
    float progress_rate;                 /**< 进度速率(0.0~1.0, 每epoch) */
    float completion_threshold;          /**< 当前级别完成阈值(准确率/损失) */
    int enable_adaptive;                 /**< 启用自适应进度 */
    size_t min_epochs_per_level;         /**< 每级别最小时长(epoch) */
    size_t max_epochs_per_level;         /**< 每级别最大时长(epoch) */
} TrainingCurriculumConfig;

/**
 * @brief 课程学习状态
 */
typedef struct CurriculumState CurriculumState;

/**
 * @brief 创建课程学习状态
 *
 * @param config 课程学习配置
 * @param input_dim 输入维度
 * @param output_dim 输出维度
 * @return CurriculumState* 课程状态指针，失败返回NULL
 */
CurriculumState* curriculum_create(const TrainingCurriculumConfig* config,
                                    size_t input_dim, size_t output_dim);

/**
 * @brief 添加一个难度级别的训练数据
 *
 * @param state 课程状态
 * @param level_idx 级别索引(0 ~ num_difficulty_levels-1)
 * @param data 训练数据
 * @param targets 目标数据
 * @param num_samples 样本数
 * @return int 成功返回0，失败返回-1
 */
int curriculum_add_level_data(CurriculumState* state, size_t level_idx,
                               const float* data, const float* targets,
                               size_t num_samples);

/**
 * @brief 获取当前级别的训练批次
 *
 * 根据当前训练进度自动选择合适难度的数据。
 * 简单AutoEncoder预训练→单模态简单任务→单模态复杂任务→多模态简单→多模态复杂。
 *
 * @param state 课程状态
 * @param epoch 当前训练轮数
 * @param model_accuracy 当前模型准确率（自适应模式使用）
 * @param batch_data 输出批次数据指针
 * @param batch_targets 输出批次目标指针
 * @param batch_size 输出批次大小
 * @param current_level 输出当前级别
 * @return int 成功返回0，失败返回-1
 */
int curriculum_get_batch(CurriculumState* state, size_t epoch,
                          float model_accuracy,
                          float** batch_data, float** batch_targets,
                          size_t* batch_size, size_t* current_level);

/**
 * @brief 获取课程总进度
 *
 * @param state 课程状态
 * @return float 进度值(0.0~1.0)
 */
float curriculum_get_progress(CurriculumState* state);

/**
 * @brief 释放课程学习状态
 *
 * @param state 课程状态
 */
void curriculum_free(CurriculumState* state);

/**
 * @brief 预训练类型
 */
typedef enum {
    PRETRAIN_AUTOENCODER = 0,     /**< 自动编码器预训练 */
    PRETRAIN_DENOISING_AE = 1,    /**< 去噪自动编码器 */
    PRETRAIN_CONTRASTIVE = 2,     /**< 对比预训练(CLIP式) */
    PRETRAIN_MASKED = 3           /**< 掩码自编码器(MAE式) */
} PretrainType;

/**
 * @brief 预训练配置
 */
typedef struct {
    PretrainType type;                /**< 预训练类型 */
    size_t encoding_dim;              /**< 编码维度（自动编码器） */
    float noise_level;                /**< 噪声水平（去噪AE） */
    float mask_ratio;                 /**< 掩码比例（掩码AE, 0.0~1.0） */
    float contrastive_temperature;    /**< 对比学习温度 */
    size_t num_epochs;                /**< 预训练轮数 */
    size_t batch_size;                /**< 批量大小 */
    float learning_rate;              /**< 学习率 */
    int freeze_encoder;               /**< 预训练后冻结编码器 */
    int enable_progressive_unfreeze;  /**< 渐进解冻编码器层 */
} PretrainConfig;

#define PRETRAIN_CONFIG_DEFAULT { \
    .type = PRETRAIN_AUTOENCODER, \
    .encoding_dim = 64, \
    .noise_level = 0.1f, \
    .mask_ratio = 0.25f, \
    .contrastive_temperature = 0.1f, \
    .num_epochs = 50, \
    .batch_size = 32, \
    .learning_rate = 0.001f, \
    .freeze_encoder = 0, \
    .enable_progressive_unfreeze = 0 \
}

/**
 * @brief 执行自动编码器预训练
 *
 * 使用无监督数据训练LNN作为自动编码器（输入→编码→解码→重建输入）。
 * 预训练完成后编码器部分可用于下游任务。
 *
 * @param trainer 训练器
 * @param data 无监督训练数据
 * @param num_samples 样本数
 * @param input_dim 输入维度
 * @param config 预训练配置
 * @param callback 训练回调函数（可选）
 * @param user_data 回调函数用户数据（可选）
 * @return int 成功返回0，失败返回-1
 */
int trainer_pretrain_autoencoder(Trainer* trainer, const float* data,
                                  size_t num_samples, size_t input_dim,
                                  const PretrainConfig* config,
                                  TrainingEventCallback callback, void* user_data);

/**
 * @brief 执行多模态对比预训练(CLIP式)
 *
 * 使用配对的视觉-文本数据训练多模态对齐。
 * 对比损失最大化匹配对的余弦相似度，最小化不匹配对的相似度。
 *
 * @param trainer 训练器
 * @param vision_data 视觉模态数据 (num_samples × vision_dim)
 * @param text_data 文本模态数据 (num_samples × text_dim)
 * @param num_samples 样本数
 * @param vision_dim 视觉维度
 * @param text_dim 文本维度
 * @param config 预训练配置
 * @param callback 训练回调函数
 * @param user_data 回调用户数据
 * @return int 成功返回0，失败返回-1
 */
int trainer_pretrain_contrastive(Trainer* trainer,
                                  const float* vision_data, const float* text_data,
                                  size_t num_samples, size_t vision_dim, size_t text_dim,
                                  const PretrainConfig* config,
                                  TrainingEventCallback callback, void* user_data);

/**
 * @brief 执行渐进式训练（先单模态后多模态）
 *
 * 分三阶段训练：阶段1=单模态训练，阶段2=多模态扩展，阶段3=联合微调。
 * 每个阶段自动切换输入维度和训练数据。
 *
 * @param trainer 训练器
 * @param single_inputs 单模态输入数据
 * @param multi_inputs 多模态输入数据
 * @param targets 目标数据
 * @param num_samples_single 单模态样本数
 * @param num_samples_multi 多模态样本数
 * @param single_dim 单模态输入维度
 * @param multi_dim 多模态输入维度
 * @param output_dim 输出维度
 * @param single_epochs 阶段1轮数
 * @param multi_epochs 阶段2轮数
 * @param joint_epochs 阶段3轮数
 * @param callback 训练回调函数
 * @param user_data 回调用户数据
 * @return int 成功返回0，失败返回-1
 */
int trainer_progressive_train(Trainer* trainer,
                               const float* single_inputs, const float* multi_inputs,
                               const float* targets,
                               size_t num_samples_single, size_t num_samples_multi,
                               size_t single_dim, size_t multi_dim, size_t output_dim,
                               size_t single_epochs, size_t multi_epochs, size_t joint_epochs,
                               TrainingEventCallback callback, void* user_data);

/**
 * @brief 训练数据生成器配置
 */
typedef struct {
    size_t input_dim;                    /**< 输入维度 */
    size_t output_dim;                   /**< 输出维度 */
    int signal_type;                     /**< 信号类型: 0=正弦, 1=方波, 2=三角波, 3=高斯噪声, 4=混合, 5=随机多项式 */
    float noise_level;                   /**< 噪声水平(0.0~1.0) */
    int num_frequencies;                 /**< 频率数量(混合信号) */
    float amplitude_range[2];            /**< 幅度范围[min, max] */
    float frequency_range[2];            /**< 频率范围[min, max] */
    int seed;                            /**< 随机种子 */
    int label_type;                      /**< 标签类型: 0=回归, 1=分类 */
    size_t num_classes;                  /**< 分类类别数(label_type=1) */
} DataGeneratorConfig;

#define DATA_GENERATOR_CONFIG_DEFAULT { \
    .input_dim = 10, \
    .output_dim = 1, \
    .signal_type = 4, \
    .noise_level = 0.05f, \
    .num_frequencies = 3, \
    .amplitude_range = {0.5f, 2.0f}, \
    .frequency_range = {0.5f, 5.0f}, \
    .seed = 42, \
    .label_type = 0, \
    .num_classes = 2 \
}

/**
 * @brief 创建数据生成器
 *
 * @param config 生成器配置
 * @return void* 生成器句柄，失败返回NULL
 */
void* data_generator_create(const DataGeneratorConfig* config);

/**
 * @brief 生成训练数据
 *
 * 根据配置生成指定数量的合成训练样本。
 * 支持正弦波、方波、三角波、高斯噪声、混合信号和随机多项式。
 *
 * @param generator 生成器句柄
 * @param inputs 输出输入数据缓冲区(num_samples × input_dim)
 * @param targets 输出目标数据缓冲区(num_samples × output_dim)
 * @param num_samples 要生成的样本数
 * @return int 成功返回0，失败返回-1
 */
int data_generator_generate(void* generator, float* inputs, float* targets,
                             size_t num_samples);

/**
 * @brief 释放数据生成器
 *
 * @param generator 生成器句柄
 */
void data_generator_free(void* generator);

/**
 * @brief 学习率预热调度器
 *
 * 在训练初期线性增加学习率，减少早期不稳定性。
 * 预热结束后使用余弦退火调度。
 */
typedef struct {
    size_t warmup_steps;                 /**< 预热步数 */
    float initial_lr;                    /**< 初始学习率(预热起点) */
    float peak_lr;                       /**< 峰值学习率(预热终点) */
    float min_lr;                        /**< 最小学习率(余弦退火终点) */
    size_t total_steps;                  /**< 总步数(预热+余弦) */
    int enable_cosine_annealing;         /**< 预热后启用余弦退火 */
} LRWarmupConfig;

/**
 * @brief 创建学习率预热调度器
 *
 * @param config 预热配置
 * @return void* 调度器句柄，失败返回NULL
 */
void* lr_warmup_scheduler_create(const LRWarmupConfig* config);

/**
 * @brief 获取当前预热调度后的学习率
 *
 * @param scheduler 调度器句柄
 * @param step 当前训练步数
 * @return float 当前学习率
 */
float lr_warmup_scheduler_get(void* scheduler, size_t step);

/**
 * @brief 释放学习率预热调度器
 *
 * @param scheduler 调度器句柄
 */
void lr_warmup_scheduler_free(void* scheduler);

/**
 * @brief 训练阶段控制——设置当前训练阶段
 *
 * 用于分阶段训练控制，自动切换学习率、数据加载策略等。
 * phase=0: 从零训练, phase=1: 预训练, phase=2: 微调, phase=3: 多模态对齐
 *
 * @param trainer 训练器
 * @param phase 训练阶段
 * @return int 成功返回0，失败返回-1
 */
int trainer_set_training_phase(Trainer* trainer, int phase);

/**
 * @brief 获取当前训练阶段
 *
 * @param trainer 训练器
 * @return int 当前阶段编号，失败返回-1
 */
int trainer_get_training_phase(Trainer* trainer);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_TRAINING_H */