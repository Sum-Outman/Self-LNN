/**
 * @file meta_learning.h
 * @brief 元学习（Meta-learning）系统增强
 * 
 * 元学习系统，支持few-shot learning、meta-optimization、任务适应等。
 * 提供模型无关的元学习（MAML）、原型网络、关系网络等算法。
 */

#ifndef SELFLNN_LEARNING_META_LEARNING_H
#define SELFLNN_LEARNING_META_LEARNING_H

#include "selflnn/core/cfc_network.h"
#include "selflnn/learning/learning.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 兼容性定义：NeuralNetwork 是 LNN 的别名（统一液态神经网络） */
#ifndef SELFLNN_NEURAL_NETWORK_DEFINED
#define SELFLNN_NEURAL_NETWORK_DEFINED
typedef struct LNN NeuralNetwork;
#endif

/**
 * @brief 元学习算法类型
 */
typedef enum {
    META_LEARNING_MAML = 0,         /**< 模型无关的元学习（MAML） */
    META_LEARNING_PROTOTYPICAL,     /**< 原型网络 */
    META_LEARNING_RELATION,         /**< 关系网络 */
    META_LEARNING_MATCHING,         /**< 匹配网络 */
    META_LEARNING_META_SGD,         /**< Meta-SGD */
    META_LEARNING_REPTILE,          /**< Reptile算法 */
    META_LEARNING_FOMAML,           /**< 一阶MAML */
    META_LEARNING_ANIL,             /**< 几乎无内循环（ANIL） */
    META_LEARNING_CUSTOM            /**< 自定义算法 */
} MetaLearningAlgorithm;

/**
 * @brief 元学习任务类型
 */
typedef enum {
    META_TASK_CLASSIFICATION = 0,   /**< 分类任务 */
    META_TASK_REGRESSION,           /**< 回归任务 */
    META_TASK_REINFORCEMENT,        /**< 强化学习任务 */
    META_TASK_GENERATION,           /**< 生成任务 */
    META_TASK_TRANSFER              /**< 迁移学习任务 */
} MetaTaskType;

/**
 * @brief 少样本学习设置
 */
typedef struct {
    int n_way;                      /**< N-way分类 */
    int k_shot;                     /**< K-shot学习 */
    int q_query;                    /**< 查询样本数 */
    int support_samples;            /**< 支持集样本总数 */
    int query_samples;              /**< 查询集样本总数 */
} FewShotSetting;

/**
 * @brief 元学习任务
 */
typedef struct {
    char* task_id;                  /**< 任务ID */
    MetaTaskType task_type;         /**< 任务类型 */
    FewShotSetting setting;         /**< 少样本设置 */
    
    // 数据
    void* support_data;             /**< 支持集数据 */
    void* support_labels;           /**< 支持集标签 */
    void* query_data;               /**< 查询集数据 */
    void* query_labels;             /**< 查询集标签 */
    
    // 任务参数
    void* task_params;              /**< 任务特定参数 */
    size_t params_size;             /**< 参数大小 */
    
    // 性能指标
    float accuracy;                 /**< 准确率 */
    float loss;                     /**< 损失值 */
    float adaptation_time;          /**< 适应时间（毫秒） */
} MetaTask;

/**
 * @brief 元学习配置
 */
typedef struct {
    MetaLearningAlgorithm algorithm; /**< 元学习算法 */
    
    // 训练参数
    int meta_batch_size;            /**< 元批次大小 */
    int inner_batch_size;           /**< 内循环批次大小 */
    int inner_steps;                /**< 内循环步数 */
    int outer_steps;                /**< 外循环步数 */
    
    // 学习率
    float meta_learning_rate;       /**< 元学习率 */
    float inner_learning_rate;      /**< 内循环学习率 */
    float fast_learning_rate;       /**< 快速学习率（MAML） */
    
    // 优化器设置
    int use_second_order;           /**< 使用二阶梯度 */
    int use_learned_lr;             /**< 使用学习的学习率（Meta-SGD） */
    int use_batch_norm;             /**< 使用批量归一化 */
    
    // 任务设置
    int task_count;                 /**< 任务数量 */
    FewShotSetting default_setting; /**< 默认少样本设置 */
    
    // 适应策略
    int enable_gradient_adaptation; /**< 启用梯度适应 */
    int enable_parameter_adaptation; /**< 启用参数适应 */
    int enable_architecture_adaptation; /**< 启用架构适应 */
    
    // 正则化
    float weight_decay;             /**< 权重衰减 */
    float dropout_rate;             /**< Dropout率 */
    float gradient_clip;            /**< 梯度裁剪阈值 */
    
    // 评估
    int eval_frequency;             /**< 评估频率 */
    int save_checkpoints;           /**< 保存检查点 */
    char* checkpoint_dir;           /**< 检查点目录 */
} MetaLearningConfig;

/**
 * @brief 元学习器状态
 */
typedef struct {
    NeuralNetwork* meta_model;      /**< 元模型 */
    NeuralNetwork* adapted_model;   /**< 适应后模型 */
    
    // 优化器状态
    void* meta_optimizer;           /**< 元优化器 */
    void* inner_optimizer;          /**< 内循环优化器 */
    
    // 学习率状态
    float* learned_lr;              /**< 学习的学习率（Meta-SGD） */
    size_t lr_count;                /**< 学习率数量 */
    
    // 任务状态
    MetaTask* current_task;         /**< 当前任务 */
    int current_step;               /**< 当前步数 */
    
    // 性能统计
    float total_loss;               /**< 总损失 */
    float total_accuracy;           /**< 总准确率 */
    int completed_tasks;            /**< 完成任务数 */
} MetaLearnerState;

/**
 * @brief 元学习器
 */
typedef struct MetaLearner MetaLearner;

/**
 * @brief 创建元学习器
 * 
 * @param model 基础模型
 * @param config 元学习配置
 * @return MetaLearner* 元学习器指针，失败返回NULL
 */
MetaLearner* meta_learner_create(NeuralNetwork* model, 
                                 const MetaLearningConfig* config);

/**
 * @brief 销毁元学习器
 * 
 * @param learner 元学习器
 */
void meta_learner_destroy(MetaLearner* learner);

/**
 * @brief 元训练（外循环）
 * 
 * @param learner 元学习器
 * @param tasks 任务数组
 * @param task_count 任务数量
 * @return int 成功返回0，失败返回-1
 */
int meta_learner_train(MetaLearner* learner, MetaTask* tasks, int task_count);

/**
 * @brief 任务适应（内循环）
 * 
 * @param learner 元学习器
 * @param task 元学习任务
 * @return NeuralNetwork* 适应后模型，失败返回NULL
 */
NeuralNetwork* meta_learner_adapt(MetaLearner* learner, const MetaTask* task);

/**
 * @brief 元测试
 * 
 * @param learner 元学习器
 * @param task 测试任务
 * @return float 测试准确率
 */
float meta_learner_test(MetaLearner* learner, const MetaTask* task);

/**
 * @brief 执行模型无关的元学习（MAML）步骤
 * 
 * @param learner 元学习器
 * @param task 元学习任务
 * @return float 元损失
 */
float meta_learner_maml_step(MetaLearner* learner, const MetaTask* task);

/**
 * @brief 执行Reptile算法步骤
 * 
 * @param learner 元学习器
 * @param task 元学习任务
 * @return float 元损失
 */
float meta_learner_reptile_step(MetaLearner* learner, const MetaTask* task);

/**
 * @brief 执行原型网络步骤
 * 
 * @param learner 元学习器
 * @param task 元学习任务
 * @return float 元损失
 */
float meta_learner_prototypical_step(MetaLearner* learner, const MetaTask* task);

/**
 * @brief 获取元学习器状态
 * 
 * @param learner 元学习器
 * @param state 状态输出
 * @return int 成功返回0，失败返回-1
 */
int meta_learner_get_state(MetaLearner* learner, MetaLearnerState* state);

/**
 * @brief 保存元学习器检查点
 * 
 * @param learner 元学习器
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int meta_learner_save_checkpoint(MetaLearner* learner, const char* filepath);

/**
 * @brief 加载元学习器检查点
 * 
 * @param learner 元学习器
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int meta_learner_load_checkpoint(MetaLearner* learner, const char* filepath);

/**
 * @brief 创建少样本学习任务
 * 
 * @param task_id 任务ID
 * @param task_type 任务类型
 * @param setting 少样本设置
 * @return MetaTask* 任务指针，失败返回NULL
 */
MetaTask* meta_task_create(const char* task_id, MetaTaskType task_type, 
                          const FewShotSetting* setting);

/**
 * @brief 销毁元学习任务
 * 
 * @param task 元学习任务
 */
void meta_task_destroy(MetaTask* task);

/**
 * @brief 获取默认元学习配置
 * 
 * @param config 配置输出
 */
void meta_learning_default_config(MetaLearningConfig* config);

/**
 * @brief 获取默认少样本设置
 * 
 * @param setting 设置输出
 */
void few_shot_default_setting(FewShotSetting* setting);

/**
 * @brief 评估元学习性能
 * 
 * @param learner 元学习器
 * @param test_tasks 测试任务数组
 * @param task_count 任务数量
 * @param results 结果输出数组
 * @return int 成功返回0，失败返回-1
 */
int meta_learning_evaluate(MetaLearner* learner, MetaTask* test_tasks,
                          int task_count, float* results);

/**
 * @brief 计算任务相似度
 * 
 * @param task1 任务1
 * @param task2 任务2
 * @return float 相似度分数（0-1）
 */
float meta_task_similarity(const MetaTask* task1, const MetaTask* task2);

/**
 * @brief 生成元学习任务分布
 * 
 * @param base_task 基础任务
 * @param count 生成数量
 * @param variance 变化程度
 * @return MetaTask** 任务数组，失败返回NULL
 */
MetaTask** meta_task_generate_distribution(const MetaTask* base_task,
                                          int count, float variance);

// ==================== CfC元优化器（A05.2.2） ====================

/**
 * @brief CfC元优化器不透明类型
 * 
 * CfC元优化器使用CfC ODE连续时间动态系统学习优化轨迹，
 * 替代传统SGD/Adam优化器，通过液态神经网络建模参数更新动力学。
 */
typedef struct CfcMetaOptimizer CfcMetaOptimizer;

/**
 * @brief CfC元优化器配置
 */
typedef struct {
    size_t input_dim;               /**< 输入维度（梯度+参数状态+动量，默认3） */
    size_t hidden_dim;              /**< 隐藏层维度（默认32） */
    size_t output_dim;              /**< 输出维度（默认1，即参数更新量） */
    float time_constant;            /**< CfC ODE时间常数（默认0.01） */
    float learning_rate;            /**< 元优化器学习率（默认0.001） */
    float gradient_clip;            /**< 梯度裁剪阈值（默认1.0） */
    float momentum_decay;           /**< 动量衰减系数（默认0.9） */
    int use_adaptive_time;          /**< 是否使用自适应时间常数（默认1） */
    int num_layers;                 /**< CfC网络层数（默认2） */
    int use_batch_norm;             /**< 是否使用批归一化（默认0） */
    float weight_decay;             /**< 权重衰减系数（默认0.0001） */
} CfcMetaOptimizerConfig;

/**
 * @brief 获取默认CfC元优化器配置
 * 
 * @param config 配置输出
 */
void cfc_meta_optimizer_default_config(CfcMetaOptimizerConfig* config);

/**
 * @brief 创建CfC元优化器
 * 
 * CfC元优化器使用液态神经网络建模参数优化轨迹，
 * 接收(梯度, 参数状态, 动量)作为输入，输出参数更新量。
 * 
 * @param config 优化器配置
 * @return CfcMetaOptimizer* 优化器指针，失败返回NULL
 */
CfcMetaOptimizer* cfc_meta_optimizer_create(const CfcMetaOptimizerConfig* config);

/**
 * @brief 销毁CfC元优化器
 * 
 * @param optimizer 优化器指针
 */
void cfc_meta_optimizer_destroy(CfcMetaOptimizer* optimizer);

/**
 * @brief 执行一步CfC元优化
 * 
 * 使用CfC ODE动态系统计算参数更新：
 *   更新量 = CfC(梯度, 参数状态, 动量)
 *   新参数 = 旧参数 + 更新量
 * 
 * @param optimizer CfC元优化器
 * @param gradient 梯度数组
 * @param param_state 当前参数状态数组
 * @param num_params 参数数量
 * @param updated_params 更新后的参数输出（可与param_state相同实现原地更新）
 * @return int 成功返回0，失败返回-1
 */
int cfc_meta_optimizer_step(CfcMetaOptimizer* optimizer,
                           const float* gradient,
                           const float* param_state,
                           size_t num_params,
                           float* updated_params);

/**
 * @brief 重置CfC元优化器内部状态
 * 
 * 清除动量缓冲区和CfC隐藏状态，用于新任务序列开始前。
 * 
 * @param optimizer CfC元优化器
 */
void cfc_meta_optimizer_reset_state(CfcMetaOptimizer* optimizer);

/**
 * @brief 获取CfC元优化器的CfC网络参数（用于元训练）
 * 
 * @param optimizer CfC元优化器
 * @param params 参数输出缓冲区
 * @param num_params 输出参数数量
 * @return int 成功返回0，失败返回-1
 */
int cfc_meta_optimizer_get_params(CfcMetaOptimizer* optimizer,
                                 float* params, size_t* num_params);

/**
 * @brief 设置CfC元优化器的CfC网络参数（用于元训练更新）
 * 
 * @param optimizer CfC元优化器
 * @param params 参数数组
 * @param num_params 参数数量
 * @return int 成功返回0，失败返回-1
 */
int cfc_meta_optimizer_set_params(CfcMetaOptimizer* optimizer,
                                 const float* params, size_t num_params);

/**
 * @brief 将CfC元优化器集成到元学习器
 * 
 * 替换元学习器的默认优化器为CfC元优化器。
 * 调用后，元学习器的内循环优化将使用CfC ODE动态系统。
 * 
 * @param learner 元学习器
 * @param optimizer CfC元优化器
 * @return int 成功返回0，失败返回-1
 */
int meta_learner_use_cfc_optimizer(MetaLearner* learner,
                                  CfcMetaOptimizer* optimizer);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_LEARNING_META_LEARNING_H */