#ifndef SELFLNN_TRAINING_ENHANCED_H
#define SELFLNN_TRAINING_ENHANCED_H

/**
 * @file training_enhanced.h
 * @brief 增强训练模块 — 【高级增强功能（与基础版互补，非替代）】
 *
 * P3-004 功能边界说明:
 *   ✅ 本文件: EMA权重管理器（指数移动平均）、Warmup+余弦退火学习率调度器
 *   ✅ 本文件: 知识蒸馏训练器（教师-学生网络）、混合精度训练(MixedPrecision)
 *   ✅ 本文件: CfC ODE特化训练、拉普拉斯增强训练、梯度累积
 *   ✅ 本文件: 与 training.h 基础版协同工作，依赖基础版的训练循环和优化器
 *   ❌ 非本文件: 核心训练循环、SGD/Adam/RMSprop优化器、L1/L2正则化、模型版本管理
 *                → 请使用 training.h（基础功能实现）
 *
 *   基础版 (training.h)         → 核心训练循环 + 基础优化器 + 基础正则化
 *   增强版 (training_enhanced.h) → EMA + 余弦退火 + 知识蒸馏 + 混合精度 + CfC特化训练
 */

#include "selflnn/core/lnn.h"
#include "selflnn/training/training.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief EMA（指数移动平均）权重管理器
 *
 * 维护权重的滑动平均值，在推理时使用EMA权重可获得更稳定的输出。
 * EMA更新公式：ema = decay * ema + (1 - decay) * current
 */
typedef struct {
    float* ema_weights;        /**< EMA权重影子变量 */
    float* ema_biases;         /**< EMA偏置影子变量 */
    float decay;               /**< EMA衰减率（推荐0.999~0.9999） */
    size_t num_weights;        /**< 权重数量 */
    size_t num_biases;         /**< 偏置数量 */
    int initialized;           /**< 是否已初始化（首次更新直接赋值） */
} EMAWeightManager;

/**
 * @brief Warmup+余弦退火学习率调度器
 *
 * 前warmup_steps步线性预热到base_lr，
 * 之后按余弦退火从base_lr下降到min_lr。
 */
typedef struct {
    float base_lr;             /**< 基础学习率（峰值） */
    float min_lr;              /**< 最小学习率（退火终点） */
    size_t warmup_steps;       /**< 预热步数 */
    size_t total_steps;        /**< 总步数（含预热） */
    size_t current_step;       /**< 当前步数 */
    int warmup_done;           /**< 是否完成预热 */
} WarmupCosineScheduler;

/**
 * @brief 知识蒸馏训练器
 *
 * 使用教师网络（冻结）指导学生网络训练。
 * 损失 = alpha * KL散度(教师软标签 || 学生软标签) + (1-alpha) * 硬标签损失
 */
typedef struct {
    LNN* teacher_network;          /**< 教师网络（冻结，不更新） */
    LNN* student_network;          /**< 学生网络（训练更新） */
    float temperature;             /**< 蒸馏温度（越高软标签越平滑） */
    float alpha;                   /**< 软标签损失权重（0~1） */
    int enable_hard_labels;        /**< 是否使用硬标签监督 */
    float learning_rate;           /**< 学生网络学习率 */
    size_t current_step;           /**< 当前训练步数 */
    float* teacher_output;         /**< 教师输出缓冲区 */
    float* student_output;         /**< 学生输出缓冲区 */
    size_t buffer_size;            /**< 输出缓冲区大小 */
} DistillationTrainer;

/**
 * @brief 训练器增强状态（通过注册表关联到Trainer）
 *
 * 在training_enhanced内部维护一个Trainer→EnhancedState的映射，
 * 无需修改Trainer结构体即可扩展功能。
 */
typedef struct {
    EMAWeightManager* ema;                 /**< EMA权重管理器 */
    float label_smoothing_factor;          /**< 标签平滑因子（0=禁用） */
    int use_label_smoothing;               /**< 是否启用标签平滑 */
    float focal_loss_gamma;                /**< Focal Loss聚焦参数 */
    float focal_loss_alpha;                /**< Focal Loss平衡参数 */
    int use_focal_loss;                    /**< 是否启用Focal Loss */
    WarmupCosineScheduler* warmup_cosine;  /**< Warmup+余弦调度器 */
    int use_warmup_cosine;                 /**< 是否启用Warmup+余弦调度 */
    int enable_ema;                        /**< 是否启用EMA */
    int ema_update_interval;               /**< EMA更新间隔（步数） */
    int ema_step_counter;                  /**< EMA步数计数器 */
} EnhancedTrainerState;

/*
 ============================================================================
 EMA权重管理器 API
 ============================================================================
*/

/**
 * @brief 创建EMA权重管理器
 *
 * @param num_weights 权重参数数量
 * @param num_biases 偏置参数数量
 * @param decay EMA衰减率（推荐0.999）
 * @return EMAWeightManager* 成功返回管理器指针，失败返回NULL
 */
EMAWeightManager* ema_weight_manager_create(size_t num_weights, size_t num_biases, float decay);

/**
 * @brief 更新EMA影子变量
 *
 * ema = decay * ema + (1 - decay) * current
 *
 * @param mgr EMA管理器
 * @param weights 当前权重参数数组
 * @param biases 当前偏置参数数组
 */
void ema_weight_manager_update(EMAWeightManager* mgr, const float* weights, const float* biases);

/**
 * @brief 将EMA影子变量应用到目标数组
 *
 * @param mgr EMA管理器
 * @param weights 目标权重数组（将被覆盖为EMA值）
 * @param biases 目标偏置数组（将被覆盖为EMA值）
 */
void ema_weight_manager_apply(EMAWeightManager* mgr, float* weights, float* biases);

/**
 * @brief 从LNN网络更新EMA影子变量
 *
 * 自动从LNN获取当前权重和偏置，执行EMA更新。
 *
 * @param mgr EMA管理器
 * @param network LNN网络实例
 */
void ema_weight_manager_update_from_lnn(EMAWeightManager* mgr, LNN* network);

/**
 * @brief 将EMA影子变量应用到LNN网络
 *
 * 将EMA加权值写入LNN网络的权重和偏置。
 *
 * @param mgr EMA管理器
 * @param network LNN网络实例
 */
void ema_weight_manager_apply_to_lnn(EMAWeightManager* mgr, LNN* network);

/**
 * @brief 重置EMA管理器（清空影子变量状态）
 *
 * @param mgr EMA管理器
 */
void ema_weight_manager_reset(EMAWeightManager* mgr);

/**
 * @brief 释放EMA权重管理器
 *
 * @param mgr EMA管理器
 */
void ema_weight_manager_free(EMAWeightManager* mgr);

/*
 ============================================================================
 标签平滑交叉熵损失函数 API
 ============================================================================
*/

/**
 * @brief 计算标签平滑交叉熵损失
 *
 * 标签平滑：target_smooth = (1 - factor) * target + factor / num_classes
 * 交叉熵：loss = -sum(target_smooth * log(predictions))
 *
 * @param predictions 模型预测值（softmax后，长度为num_classes）
 * @param targets 真实标签（one-hot编码，长度为num_classes）
 * @param num_classes 类别数
 * @param smoothing_factor 平滑因子（0~1，推荐0.1）
 * @return float 损失值
 */
float label_smoothing_cross_entropy(const float* predictions, const float* targets,
                                    size_t num_classes, float smoothing_factor);

/**
 * @brief 计算标签平滑交叉熵损失的梯度
 *
 * @param predictions 模型预测值（softmax后）
 * @param targets 真实标签（one-hot编码）
 * @param num_classes 类别数
 * @param smoothing_factor 平滑因子
 * @param gradient 输出梯度缓冲区（长度num_classes）
 */
void label_smoothing_cross_entropy_gradient(const float* predictions, const float* targets,
                                            size_t num_classes, float smoothing_factor,
                                            float* gradient);

/*
 ============================================================================
 Focal Loss API
 ============================================================================
*/

/**
 * @brief 计算Focal Loss
 *
 * FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
 * 其中 p_t = p 当 target=1，否则 p_t = 1-p
 *
 * @param predictions 模型预测值（sigmoid后，长度num_classes）
 * @param targets 真实标签（one-hot编码，长度num_classes）
 * @param num_classes 类别数
 * @param gamma 聚焦参数（>=0，推荐2.0，越大越关注难样本）
 * @param alpha 平衡参数（0~1，推荐0.25，正样本权重）
 * @return float 损失值
 */
float focal_loss(const float* predictions, const float* targets,
                 size_t num_classes, float gamma, float alpha);

/**
 * @brief 计算Focal Loss的梯度
 *
 * @param predictions 模型预测值（sigmoid后）
 * @param targets 真实标签（one-hot编码）
 * @param num_classes 类别数
 * @param gamma 聚焦参数
 * @param alpha 平衡参数
 * @param gradient 输出梯度缓冲区（长度num_classes）
 */
void focal_loss_gradient(const float* predictions, const float* targets,
                         size_t num_classes, float gamma, float alpha,
                         float* gradient);

/*
 ============================================================================
 Warmup+余弦退火学习率调度器 API
 ============================================================================
*/

/**
 * @brief 创建Warmup+余弦退火学习率调度器
 *
 * @param base_lr 基础学习率（预热后的峰值学习率）
 * @param min_lr 最小学习率（退火终点）
 * @param warmup_steps 预热步数（从0线性上升到base_lr）
 * @param total_steps 总步数（含预热）
 * @return WarmupCosineScheduler* 成功返回调度器指针，失败返回NULL
 */
WarmupCosineScheduler* warmup_cosine_scheduler_create(float base_lr, float min_lr,
                                                      size_t warmup_steps, size_t total_steps);

/**
 * @brief 执行一步调度，返回当前学习率
 *
 * @param scheduler 调度器指针
 * @return float 当前步数的学习率
 */
float warmup_cosine_scheduler_step(WarmupCosineScheduler* scheduler);

/**
 * @brief 获取当前学习率（不推进步数）
 *
 * @param scheduler 调度器指针
 * @return float 当前学习率
 */
float warmup_cosine_scheduler_get_lr(const WarmupCosineScheduler* scheduler);

/**
 * @brief 重置调度器状态
 *
 * @param scheduler 调度器指针
 */
void warmup_cosine_scheduler_reset(WarmupCosineScheduler* scheduler);

/**
 * @brief 释放Warmup+余弦调度器
 *
 * @param scheduler 调度器指针
 */
void warmup_cosine_scheduler_free(WarmupCosineScheduler* scheduler);

/*
 ============================================================================
 知识蒸馏训练器 API
 ============================================================================
*/

/**
 * @brief 创建知识蒸馏训练器
 *
 * 教师网络在蒸馏过程中冻结（不会更新权重）。
 * 学生网络通过蒸馏损失进行训练。
 *
 * @param teacher 教师网络（必须已初始化）
 * @param student 学生网络（必须已初始化）
 * @param temperature 蒸馏温度（>=1.0，推荐2.0~8.0）
 * @param alpha 软标签损失权重（0~1，推荐0.7）
 * @param learning_rate 学生网络学习率
 * @return DistillationTrainer* 成功返回训练器指针，失败返回NULL
 */
DistillationTrainer* distillation_trainer_create(LNN* teacher, LNN* student,
                                                  float temperature, float alpha,
                                                  float learning_rate);

/**
 * @brief 执行一步知识蒸馏训练
 *
 * 1. 教师网络前向传播（无梯度）
 * 2. 学生网络前向传播
 * 3. 计算蒸馏损失（软标签KL散度 + 可选的硬标签损失）
 * 4. 反向传播更新学生网络
 *
 * @param trainer 蒸馏训练器
 * @param input 输入样本
 * @param hard_targets 硬标签目标（可选，为NULL则不使用硬标签）
 * @param input_dim 输入维度
 * @param output_dim 输出维度
 * @param loss_output 输出总损失值（可选，为NULL则忽略）
 * @return int 成功返回0，失败返回-1
 */
int distillation_trainer_step(DistillationTrainer* trainer,
                              const float* input, const float* hard_targets,
                              size_t input_dim, size_t output_dim,
                              float* loss_output);

/**
 * @brief 执行批量知识蒸馏训练
 *
 * @param trainer 蒸馏训练器
 * @param inputs 批量输入数据
 * @param hard_targets 批量硬标签（可选）
 * @param num_samples 样本数
 * @param input_dim 输入维度
 * @param output_dim 输出维度
 * @param avg_loss 输出平均损失值（可选）
 * @return int 成功返回0，失败返回-1
 */
int distillation_trainer_train_batch(DistillationTrainer* trainer,
                                     const float* inputs, const float* hard_targets,
                                     size_t num_samples,
                                     size_t input_dim, size_t output_dim,
                                     float* avg_loss);

/**
 * @brief 释放知识蒸馏训练器
 *
 * @param trainer 蒸馏训练器
 */
void distillation_trainer_free(DistillationTrainer* trainer);

/*
 ============================================================================
 Trainer集成辅助 API
 ============================================================================
*/

/**
 * @brief 为Trainer注册增强状态（启用所有增强功能）
 *
 * 在training_enhanced内部维护Trainer→EnhancedTrainerState的映射。
 *
 * @param trainer 训练器指针
 * @return int 成功返回0，失败返回-1
 */
int trainer_enhanced_register(Trainer* trainer);

/**
 * @brief 注销Trainer的增强状态
 *
 * @param trainer 训练器指针
 * @return int 成功返回0，失败返回-1
 */
int trainer_enhanced_unregister(Trainer* trainer);

/**
 * @brief 获取Trainer的增强状态
 *
 * @param trainer 训练器指针
 * @return EnhancedTrainerState* 成功返回状态指针，失败返回NULL
 */
EnhancedTrainerState* trainer_enhanced_get_state(Trainer* trainer);

/**
 * @brief 启用EMA并设置衰减率
 *
 * @param trainer 训练器指针
 * @param decay EMA衰减率（推荐0.999）
 * @param update_interval EMA更新间隔（步数，默认1）
 * @return int 成功返回0，失败返回-1
 */
int trainer_enhanced_enable_ema(Trainer* trainer, float decay, int update_interval);

/**
 * @brief 禁用EMA
 *
 * @param trainer 训练器指针
 * @return int 成功返回0，失败返回-1
 */
int trainer_enhanced_disable_ema(Trainer* trainer);

/**
 * @brief 将EMA权重应用到网络（推理前调用）
 *
 * @param trainer 训练器指针
 * @return int 成功返回0，失败返回-1
 */
int trainer_enhanced_apply_ema(Trainer* trainer);

/**
 * @brief 启用标签平滑
 *
 * @param trainer 训练器指针
 * @param factor 平滑因子（推荐0.1）
 * @return int 成功返回0，失败返回-1
 */
int trainer_enhanced_enable_label_smoothing(Trainer* trainer, float factor);

/**
 * @brief 禁用标签平滑
 *
 * @param trainer 训练器指针
 * @return int 成功返回0，失败返回-1
 */
int trainer_enhanced_disable_label_smoothing(Trainer* trainer);

/**
 * @brief 启用Focal Loss
 *
 * @param trainer 训练器指针
 * @param gamma 聚焦参数（推荐2.0）
 * @param alpha 平衡参数（推荐0.25）
 * @return int 成功返回0，失败返回-1
 */
int trainer_enhanced_enable_focal_loss(Trainer* trainer, float gamma, float alpha);

/**
 * @brief 禁用Focal Loss
 *
 * @param trainer 训练器指针
 * @return int 成功返回0，失败返回-1
 */
int trainer_enhanced_disable_focal_loss(Trainer* trainer);

/**
 * @brief 启用Warmup+余弦退火学习率调度
 *
 * 会覆盖Trainer原有的学习率调度器行为。
 *
 * @param trainer 训练器指针
 * @param base_lr 基础学习率
 * @param min_lr 最小学习率
 * @param warmup_steps 预热步数
 * @param total_steps 总步数
 * @return int 成功返回0，失败返回-1
 */
int trainer_enhanced_enable_warmup_cosine(Trainer* trainer, float base_lr, float min_lr,
                                          size_t warmup_steps, size_t total_steps);

/**
 * @brief 禁用Warmup+余弦调度
 *
 * @param trainer 训练器指针
 * @return int 成功返回0，失败返回-1
 */
int trainer_enhanced_disable_warmup_cosine(Trainer* trainer);

/**
 * @brief 在训练步结束后调用（更新EMA、调度器等）
 *
 * 应在trainer_train的每个batch/epoch结束后调用。
 *
 * @param trainer 训练器指针
 * @return int 成功返回0，失败返回-1
 */
int trainer_enhanced_on_step_end(Trainer* trainer);

/**
 * @brief 在训练epoch结束后调用（记录状态等）
 *
 * @param trainer 训练器指针
 * @return int 成功返回0，失败返回-1
 */
int trainer_enhanced_on_epoch_end(Trainer* trainer);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_TRAINING_ENHANCED_H */
