/**
 * @file learning.h
 * @brief 学习与演化模块接口
 * 
 * 自我学习、自我演化、模仿学习等功能的接口。
 */

#ifndef SELFLNN_LEARNING_H
#define SELFLNN_LEARNING_H

#include <stddef.h>
#include "selflnn/learning/online_learning.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 学习类型枚举
 */
typedef enum {
    LEARNING_REINFORCEMENT = 0,   /**< 强化学习 */
    LEARNING_EVOLUTIONARY = 1,    /**< 演化学习 */
    LEARNING_IMITATION = 2,       /**< 模仿学习 */
    LEARNING_SELF_SUPERVISED = 3, /**< 自监督学习 */
    LEARNING_TRANSFER = 4         /**< 迁移学习 */
} LearningType;

/**
 * @brief 学习配置
 */
typedef struct {
    LearningType learning_type;   /**< 学习类型 */
    float learning_rate;          /**< 学习率 */
    float exploration_rate;       /**< 探索率（强化学习） */
    float discount_factor;        /**< 折扣因子（强化学习） */
    int population_size;          /**< 种群大小（演化学习） */
    int max_generations;          /**< 最大代数（演化学习） */
    float mutation_rate;          /**< 突变率（演化学习） */
    int enable_self_evolution;    /**< 是否启用自我演化 */
    int enable_self_correction;   /**< 是否启用自我修正 */
    float risk_tolerance;         /**< 风险容忍度 */
    float goal_tolerance;         /**< 目标容差 */
    int max_plan_length;          /**< 最大规划长度 */
    int enable_adaptation;        /**< 是否启用自适应 */
} LearningConfig;

/**
 * @brief 学习引擎句柄
 */
typedef struct LearningEngine LearningEngine;

/**
 * @brief 创建学习引擎
 * 
 * @param config 学习配置
 * @return LearningEngine* 学习引擎句柄，失败返回NULL
 */
LearningEngine* learning_engine_create(const LearningConfig* config);

/**
 * @brief 释放学习引擎
 * 
 * @param engine 学习引擎句柄
 */
void learning_engine_free(LearningEngine* engine);

/**
 * @brief 强化学习：根据奖励更新策略
 * 
 * @param engine 学习引擎句柄
 * @param state 当前状态
 * @param action 采取的动作
 * @param reward 获得的奖励
 * @param next_state 下一状态
 * @return int 成功返回0，失败返回-1
 */
int learning_reinforcement_update(LearningEngine* engine,
                                 const float* state, size_t state_size,
                                 const float* action, size_t action_size,
                                 float reward,
                                 const float* next_state, size_t next_state_size);

/**
 * @brief 强化学习：选择动作
 * 
 * @param engine 学习引擎句柄
 * @param state 当前状态
 * @param state_size 状态大小
 * @param action 动作输出缓冲区
 * @param max_action_size 最大动作大小
 * @return int 成功返回动作大小，失败返回-1
 */
int learning_reinforcement_select_action(LearningEngine* engine,
                                        const float* state, size_t state_size,
                                        float* action, size_t max_action_size);

/**
 * @brief 演化学习：演化种群
 * 
 * @param engine 学习引擎句柄
 * @param fitness_scores 适应度分数数组
 * @param num_scores 分数数量
 * @return int 成功返回0，失败返回-1
 */
int learning_evolutionary_evolve(LearningEngine* engine,
                                const float* fitness_scores, size_t num_scores);

/**
 * @brief 演化学习：获取当前最佳个体
 * 
 * @param engine 学习引擎句柄
 * @param individual 个体输出缓冲区
 * @param max_individual_size 最大个体大小
 * @return int 成功返回个体大小，失败返回-1
 */
int learning_evolutionary_get_best(LearningEngine* engine,
                                  float* individual, size_t max_individual_size);

#include "selflnn/evolution/pareto_optimization.h"

int learning_evolutionary_evolve_multi(LearningEngine* engine,
                                        ParetoFront* front,
                                        float* objectives_buffer,
                                        int num_objectives);

int learning_evolutionary_get_pareto_json(const LearningEngine* engine,
                                           const ParetoFront* front,
                                           char* buffer, size_t buffer_size);

/**
 * @brief 模仿学习：从示范中学习
 * 
 * @param engine 学习引擎句柄
 * @param demonstration 示范数据
 * @param demo_size 示范数据大小
 * @param context 上下文数据
 * @param context_size 上下文数据大小
 * @return int 成功返回0，失败返回-1
 */
int learning_imitation_learn(LearningEngine* engine,
                            const float* demonstration, size_t demo_size,
                            const float* context, size_t context_size);

/**
 * @brief 模仿学习：生成模仿行为
 * 
 * @param engine 学习引擎句柄
 * @param context 上下文数据
 * @param context_size 上下文数据大小
 * @param behavior 行为输出缓冲区
 * @param max_behavior_size 最大行为大小
 * @return int 成功返回行为大小，失败返回-1
 */
int learning_imitation_generate(LearningEngine* engine,
                               const float* context, size_t context_size,
                               float* behavior, size_t max_behavior_size);

/**
 * @brief 自我演化：自适应调整参数
 * 
 * @param engine 学习引擎句柄
 * @param performance_metrics 性能指标数组
 * @param num_metrics 指标数量
 * @return int 成功返回0，失败返回-1
 */
int learning_self_evolve(LearningEngine* engine,
                        const float* performance_metrics, size_t num_metrics);

/**
 * @brief 自我修正：检测并修正错误
 * 
 * @param engine 学习引擎句柄
 * @param error_signals 错误信号数组
 * @param num_errors 错误数量
 * @param correction 修正输出缓冲区
 * @param max_correction_size 最大修正大小
 * @return int 成功返回修正大小，失败返回-1
 */
int learning_self_correct(LearningEngine* engine,
                         const float* error_signals, size_t num_errors,
                         float* correction, size_t max_correction_size);

/**
 * @brief 迁移学习：迁移知识到新任务
 * 
 * @param engine 学习引擎句柄
 * @param source_task 源任务数据
 * @param source_size 源任务数据大小
 * @param target_task 目标任务数据
 * @param target_size 目标任务数据大小
 * @return int 成功返回0，失败返回-1
 */
int learning_transfer_knowledge(LearningEngine* engine,
                               const float* source_task, size_t source_size,
                               const float* target_task, size_t target_size);

/**
 * @brief 获取学习引擎配置
 * 
 * @param engine 学习引擎句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int learning_engine_get_config(const LearningEngine* engine, LearningConfig* config);

/**
 * @brief 设置学习引擎配置
 * 
 * @param engine 学习引擎句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int learning_engine_set_config(LearningEngine* engine, const LearningConfig* config);

/**
 * @brief 重置学习引擎
 * 
 * @param engine 学习引擎句柄
 */
void learning_engine_reset(LearningEngine* engine);

/**
 * @brief 在线学习更新（单样本）
 * 
 * @param engine Online learning update (single sample)
 * @param input 输入特征向量
 * @param input_size 输入特征维度
 * @param target 目标值
 * @param target_size 目标维度
 * @param loss 输出损失值（可选）
 * @return int 成功返回0，失败返回-1
 */
int learning_engine_online_update(LearningEngine* engine,
                                  const float* input, size_t input_size,
                                  const float* target, size_t target_size,
                                  float* loss);

/**
 * @brief 在线学习更新（批量）
 * 
 * @param engine 学习引擎句柄
 * @param inputs 输入特征矩阵（样本×特征）
 * @param num_samples 样本数量
 * @param input_size 输入特征维度
 * @param targets 目标矩阵（样本×目标）
 * @param target_size 目标维度
 * @param average_loss 输出平均损失（可选）
 * @return int 成功返回0，失败返回-1
 */
int learning_engine_online_update_batch(LearningEngine* engine,
                                        const float* inputs, size_t num_samples,
                                        size_t input_size,
                                        const float* targets, size_t target_size,
                                        float* average_loss);

/**
 * @brief 检查概念漂移
 * 
 * @param engine 学习引擎句柄
 * @param data 当前输入数据
 * @param data_size 数据维度
 * @param drift_detected 输出是否检测到漂移（可选）
 * @param confidence 输出检测置信度（可选）
 * @return int 成功返回0，失败返回-1
 */
int learning_engine_check_concept_drift(LearningEngine* engine,
                                        const float* data, size_t data_size,
                                        int* drift_detected,
                                        float* confidence);

/**
 * @brief 获取在线学习状态
 * 
 * @param engine 学习引擎句柄
 * @param status 在线学习状态输出
 * @return int 成功返回0，失败返回-1
 */
int learning_engine_get_online_status(LearningEngine* engine,
                                      OnlineLearningStatus* status);

/**
 * @brief 配置在线学习器
 * 
 * @param engine 学习引擎句柄
 * @param config 新的在线学习配置
 * @return int 成功返回0，失败返回-1
 */
int learning_engine_configure_online_learning(LearningEngine* engine,
                                               const OnlineLearningConfig* config);

/**
 * @brief 自动超参数调整（自我修正集成版）
 * 
 * 通过自我修正机制检测训练停滞，自动执行超参数搜索并应用最佳配置。
 * 此函数需要在外部传入Trainer句柄以访问训练状态和配置。
 * 当学习引擎检测到系统性错误模式时，可配合self_correct使用。
 * 
 * @param engine 学习引擎句柄
 * @param trainer 训练器句柄（用于超参数搜索和配置应用）
 * @param inputs 输入数据
 * @param targets 目标数据
 * @param num_samples 样本数
 * @return int 成功返回0，失败返回-1
 */
struct Trainer;
int learning_auto_tune_hyperparameters(LearningEngine* engine,
                                        struct Trainer* trainer,
                                        const float* inputs, const float* targets,
                                        size_t num_samples);

/**
 * @brief 设置学习引擎关联的LNN网络
 * 
 * 将学习引擎与一个LNN网络实例关联。当设置了网络后，
 * 自我修正验证(self_correct_verify)将使用真实LNN前向传播计算预测，
 * 而不是简化的线性近似。自我演化(self_evolve)可直接操作网络参数。
 *
 * @param engine 学习引擎句柄
 * @param network LNN网络句柄
 * @return int 成功返回0，失败返回-1
 */
struct LNN;
int learning_engine_set_network(LearningEngine* engine, struct LNN* network);

/* ============================
 * 记忆系统与训练集成API（经验回放闭环）
 * ============================ */

/**
 * @brief 设置学习引擎关联的记忆管理器
 * 
 * 将学习引擎与记忆管理器关联，实现经验回放(experience replay)闭环。
 * 强化学习更新时会自动存储经验到记忆系统，并从中采样训练。
 * 形成"经验→记忆→采样→训练→新经验"的完整闭环。
 *
 * @param engine 学习引擎句柄
 * @param manager 记忆管理器句柄
 * @return int 成功返回0，失败返回-1
 */
struct MemoryManager;
int learning_engine_set_memory_manager(LearningEngine* engine, struct MemoryManager* manager);

/**
 * @brief 经验回放：从记忆中采样并训练在线学习器
 * 
 * 从记忆系统中采样一批经验数据，使用在线学习进行训练。
 * 通过从记忆中随机采样打破经验之间的相关性，稳定训练过程。
 * 这是深度强化学习中经验回放(Experience Replay)的完整实现。
 * 
 * @param engine 学习引擎句柄
 * @param batch_size 批次大小
 * @return int 成功返回实际训练的样本数，失败返回-1
 */
int learning_engine_experience_replay(LearningEngine* engine, size_t batch_size);

/**
 * @brief 经验回放直接训练LNN：从记忆中采样并直接训练关联的LNN
 * 
 * 与learning_engine_experience_replay不同，此函数直接使用LNN网络进行训练，
 * 而非训练在线学习器。经验从记忆系统采样后，计算LNN前向传播的误差，
 * 通过lnn_backward_batch直接更新LNN网络参数。这实现了真正的
 * 经验驱动神经网络训练闭环。
 * 
 * 采样策略：从短期记忆、长期记忆和情景记忆中按比例采样，
 * 优先使用高强度的记忆项。
 * 
 * @param engine 学习引擎句柄
 * @param batch_size 批次大小
 * @return int 成功返回实际训练的样本数，失败返回-1
 */
int learning_engine_experience_replay_train_lnn(LearningEngine* engine, size_t batch_size);

/**
 * @brief 存储经验到记忆系统
 * 
 * 将强化学习经验（状态、动作、奖励、下一状态）存储到记忆系统。
 * 经验的键名按照"exp_<timestamp>"格式自动生成，确保唯一性。
 * 经验存储为短期记忆，便于后续采样和巩固。
 * 
 * @param engine 学习引擎句柄
 * @param state 当前状态
 * @param state_size 状态维度
 * @param action 采取的动作
 * @param action_size 动作维度
 * @param reward 获得的奖励
 * @param next_state 下一状态
 * @param next_state_size 下一状态维度
 * @return int 成功返回0，失败返回-1
 */
int learning_engine_store_experience(LearningEngine* engine,
                                    const float* state, size_t state_size,
                                    const float* action, size_t action_size,
                                    float reward,
                                    const float* next_state, size_t next_state_size);

/* ============================
 * 增强T4：自我修正验证闭环API
 * ============================ */

/**
 * @brief 自我修正验证状态枚举
 */
typedef enum {
    CORRECTION_PENDING = 0,             /**< 待验证 */
    CORRECTION_IN_PROGRESS = 1,         /**< 验证中 */
    CORRECTION_VERIFIED = 2,            /**< 验证通过 */
    CORRECTION_FAILED_VALIDATION = 3,   /**< 验证失败 */
    CORRECTION_COMPLETE = 4             /**< 验证完成 */
} CorrectionVerificationStatus;

/**
 * @brief 修正验证指标结构
 */
typedef struct {
    double before_error_mean;            /**< 修正前平均误差 */
    double before_error_std;             /**< 修正前误差标准差 */
    double after_error_mean;             /**< 修正后平均误差 */
    double after_error_std;              /**< 修正后误差标准差 */
    double improvement_ratio;            /**< 改进比例 (0~1) */
    double validation_accuracy;          /**< 验证准确率 */
    size_t verification_samples;         /**< 验证样本数 */
    int is_significant;                  /**< 是否达到显著改进阈值 */
    CorrectionVerificationStatus status; /**< 验证状态 */
} CorrectionVerificationResult;

/**
 * @brief 自我修正验证闭环
 * 
 * 对已执行的自我修正进行完整验证，对比修正前后的误差分布，
 * 计算改进比例，判定修正是否真正有效。
 * 必须在learning_self_correct之后调用。
 *
 * @param engine 学习引擎句柄
 * @param inputs 验证输入数据
 * @param targets 验证目标数据
 * @param num_samples 验证样本数
 * @param result 输出验证结果结构体
 * @return int 成功返回0，失败返回-1
 */
int learning_self_correct_verify(LearningEngine* engine,
                                  const float* inputs, const float* targets,
                                  size_t num_samples,
                                  CorrectionVerificationResult* result);

/* ============================
 * 示教学习集成API（teach_by_showing + manual_learning）
 * ============================ */

/* 前向声明 */
struct TeachDemoSet;
struct MLDocument;

/**
 * @brief 从演示数据集学习
 * 
 * 将TeachDemoSet中的演示数据提取为模仿学习的训练样本，
 * 自动处理不同类型任务（抓取、放置、装配、行走等）的演示数据，
 * 提取共性模式并更新模仿学习模型。
 * 
 * @param engine 学习引擎句柄
 * @param demo_set 演示数据集
 * @return int 成功返回学习的样本数，失败返回-1
 */
int learning_imitation_from_demo_set(LearningEngine* engine,
                                     const struct TeachDemoSet* demo_set);

/**
 * @brief 从文档学习
 * 
 * 提取MLDocument中的图文内容，解析为模仿学习的训练信号。
 * 支持含代码块的文档，代码作为演示、文本作为语义上下文。
 * 将文档中的步骤式说明转化为行为序列模板。
 * 
 * @param engine 学习引擎句柄
 * @param document 文档对象
 * @return int 成功返回0，失败返回-1
 */
int learning_imitation_from_document(LearningEngine* engine,
                                     const struct MLDocument* document);

/* ============================
 * 内部经验缓冲区访问API（供外部模块查询经验数据）
 * ============================ */

/**
 * @brief 获取内部经验缓冲区中的经验数量
 * 
 * @param engine 学习引擎句柄
 * @return size_t 经验数量
 */
size_t learning_engine_get_experience_count(const LearningEngine* engine);

/**
 * @brief 获取指定索引的经验数据
 * 
 * @param engine 学习引擎句柄
 * @param index 经验索引
 * @param state 状态输出缓冲区
 * @param max_state_dim 状态缓冲区最大维度
 * @param action 动作输出缓冲区
 * @param max_action_dim 动作缓冲区最大维度
 * @param reward 奖励输出
 * @param next_state 下一状态输出缓冲区
 * @param max_next_state_dim 下一状态缓冲区最大维度
 * @return int 成功返回0，失败返回-1
 */
int learning_engine_get_experience(const LearningEngine* engine, int index,
    float* state, size_t max_state_dim,
    float* action, size_t max_action_dim,
    float* reward,
    float* next_state, size_t max_next_state_dim);

/* ============================
 * 能力开关控制（P3.3）
 * ============================ */

/**
 * @brief 启用学习引擎
 * 
 * @param engine 学习引擎句柄
 */
void learning_engine_enable(LearningEngine* engine);

/**
 * @brief 禁用学习引擎
 * 
 * @param engine 学习引擎句柄
 */
void learning_engine_disable(LearningEngine* engine);

/**
 * @brief 检查学习引擎是否已启用
 * 
 * @param engine 学习引擎句柄
 * @return 1表示启用，0表示禁用
 */
int learning_engine_is_enabled(const LearningEngine* engine);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_LEARNING_H