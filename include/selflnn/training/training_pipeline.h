/**
 * @file training_pipeline.h
 * @brief 完整训练流水线接口
 */

#ifndef SELFLNN_TRAINING_PIPELINE_H
#define SELFLNN_TRAINING_PIPELINE_H

#include <stddef.h>
#include <time.h>

#include "selflnn/training/training_monitor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TRAIN_STAGE_PRETRAIN = 0,
    TRAIN_STAGE_DEEP_TRAIN = 1,
    TRAIN_STAGE_MULTIMODAL = 2,
    TRAIN_STAGE_FINE_TUNE = 3,
    TRAIN_STAGE_LOCAL = 4,
    TRAIN_STAGE_API = 5,      /**< ZSF-016: 外部API训练阶段 */
    TRAIN_STAGE_EVALUATION = 6,
    TRAIN_STAGE_IDLE = 7
} TrainingStage;

typedef struct {
    TrainingStage stage;
    int current_epoch;
    int total_epochs;
    float current_loss;
    float best_loss;
    float train_accuracy;
    float val_accuracy;
    float learning_rate;
    size_t samples_processed;
    time_t started_at;
    time_t last_checkpoint;
    int checkpoint_interval;
    int is_running;
    int is_paused;
    char dataset_path[512];
    char checkpoint_path[512];
    float gpu_utilization;
    float estimated_time_remaining;
    TrainingStage current_stage;   /**< 当前训练阶段 */
    float convergence_rate; /**< 收敛速率（暴露给前端） */
} TrainingPipelineState;

typedef struct {
    size_t pretrain_epochs;
    size_t deep_train_epochs;
    size_t multimodal_epochs;
    size_t fine_tune_epochs;
    size_t local_epochs;
    size_t api_train_epochs;           /**< ZSF-016: API训练轮数 */
    size_t speech_epochs; /**< 第6阶段语音训练轮数 */
    float pretrain_lr;
    float deep_train_lr;
    float multimodal_lr;
    float fine_tune_lr;
    float local_lr;
    float api_train_lr;                /**< ZSF-016: API训练学习率 */
    float speech_lr; /**< 第6阶段语音训练学习率 */
    float base_lr;                   /**< 基础学习率 */
    size_t batch_size;
    int use_gpu;
    int use_mixed_precision;
    int use_laplace_enhancement;
    float laplace_filter_cutoff;         /**< 拉普拉斯梯度滤波截止频率 (Hz) */
    int laplace_spectral_monitor;        /**< 拉普拉斯频谱监控 0=关闭 1=开启 */
    float laplace_stability_margin;      /**< 拉普拉斯稳定裕度阈值 (0.0-1.0) */
    int use_early_stopping;
    int early_stopping_patience;
    float validation_split;
    float convergence_threshold; /**< 绝对收敛阈值（损失低于此值立即停止管线） */
    int optimizer_type;              /**< 优化器类型 0=SGD 1=Momentum 2=AdaGrad 3=RMSProp 4=Adam 5=AdamW 6=AdaDelta 7=LAMB 8=LARS 9=Ranger 10=NovoGrad */
    int loss_function;               /**< 损失函数（LossType枚举值, loss.h）: 0=MSE 1=MAE 2=Huber 3=CategoricalCrossEntropy 4=BinaryCrossEntropy 5=KLD 6=Cosine 7=Contrastive 8=Focal 9=Dice 10=Triplet 11=Quantile */
/* 分布式梯度同步配置 */
    int use_distributed_training;    /**< 是否启用分布式训练 0=禁用 1=启用 */
    int distributed_sync_frequency;  /**< 分布式梯度同步频率（每N个批次同步一次，默认1） */
/* 双阶段检查点保存配置 */
    int checkpoint_frequency;        /**< 检查点保存频率（每N个epoch保存一次，默认10，0=仅在阶段结束时保存） */
    char data_directory[512];
    char output_directory[512];
    size_t input_size;               /**< 输入维度（LNN网络input_size，默认512） */
    size_t hidden_size;              /**< 隐藏层维度（LNN网络hidden_size，默认1024） */
    size_t output_size;              /**< 输出维度（LNN网络output_size，默认256） */
} TrainingPipelineConfig;

typedef struct TrainingPipeline TrainingPipeline;

TrainingPipeline* training_pipeline_create(const TrainingPipelineConfig* config);
void training_pipeline_free(TrainingPipeline* pipeline);
int training_pipeline_start(TrainingPipeline* pipeline);
int training_pipeline_pause(TrainingPipeline* pipeline);
int training_pipeline_resume(TrainingPipeline* pipeline);
int training_pipeline_stop(TrainingPipeline* pipeline);
int training_pipeline_load_data(TrainingPipeline* pipeline, const char* data_path);
int training_pipeline_step(TrainingPipeline* pipeline);
int training_pipeline_get_state(const TrainingPipeline* pipeline, TrainingPipelineState* state);

/* 检查是否有真实数据可用 */
int training_pipeline_has_real_data(const TrainingPipeline* pipeline);

/* ============================================================================
 * 异步数据加载与预取 (A08.3.1)
 * ============================================================================ */

typedef enum {
    ASYNC_LOADER_IDLE = 0,
    ASYNC_LOADER_LOADING,
    ASYNC_LOADER_READY,
    ASYNC_LOADER_ERROR
} AsyncLoaderState;

typedef enum {
    AUG_NONE        = 0,
    AUG_GAUSSIAN_NOISE = 1 << 0,
    AUG_UNIFORM_NOISE  = 1 << 1,
    AUG_RANDOM_SCALE   = 1 << 2,
    AUG_RANDOM_MASK    = 1 << 3,
    AUG_RANDOM_CROP    = 1 << 4,
    AUG_MIXUP          = 1 << 5,
    AUG_ALL            = 0x3F
} DataAugmentationType;

typedef struct {
    float* data;                    /**< 预取数据缓冲区 */
    float* targets;                 /**< 预取目标缓冲区 */
    size_t batch_size;              /**< 批次大小 */
    size_t feature_dim;             /**< 特征维度 */
    size_t target_dim;              /**< 目标维度 */
    int valid;                      /**< 数据是否有效 */
    volatile AsyncLoaderState state; /**< 加载器状态 */
    int batch_id;                   /**< 批次ID */
} PrefetchBuffer;

typedef struct {
    DataAugmentationType types;     /**< 启用的增强类型位掩码 */
    float noise_std;                /**< 高斯噪声标准差 */
    float noise_uniform_range;      /**< 均匀噪声范围 ±range */
    float scale_min;                /**< 随机缩放最小值 */
    float scale_max;                /**< 随机缩放最大值 */
    float mask_ratio;               /**< 随机遮盖比例 */
    float mixup_alpha;              /**< MixUp Alpha参数 */
    int apply_to_target;            /**< 增强是否应用于目标 */
    int enabled;                    /**< 是否启用增强 */
} DataAugmentationConfig;

#define DEFAULT_AUG_CONFIG { AUG_NONE, 0.01f, 0.02f, 0.9f, 1.1f, 0.1f, 0.2f, 0, 0 }

int pipeline_enable_async_loading(TrainingPipeline* pipeline, int buffer_count,
                                   int num_loader_threads);
int pipeline_disable_async_loading(TrainingPipeline* pipeline);
int pipeline_prefetch_batch(TrainingPipeline* pipeline, int batch_id);
const PrefetchBuffer* pipeline_get_prefetched_batch(TrainingPipeline* pipeline,
                                                      int batch_id);
int pipeline_set_augmentation(TrainingPipeline* pipeline,
                               const DataAugmentationConfig* aug_config);
int pipeline_augment_batch(float* data, float* targets, size_t batch_size,
                            size_t feature_dim, size_t target_dim,
                            const DataAugmentationConfig* aug_config);
int pipeline_augment_single(float* data, float* target,
                             size_t feature_dim, size_t target_dim,
                             DataAugmentationType aug_type, float strength);

/* 多模态联合训练 */
int pipeline_multimodal_train_step(TrainingPipeline* pipeline,
    const float* vision_features, size_t vision_size,
    const float* audio_features, size_t audio_size,
    const float* text_features, size_t text_size,
    const float* target, size_t target_size, float* loss);

/* 局部功能训练 */
int pipeline_local_train(TrainingPipeline* pipeline, const char* module_name,
    const float* data, size_t data_size, int epochs, float* final_loss);

/* 自动超参数搜索 */
int pipeline_auto_tune(TrainingPipeline* pipeline,
    const float* param_ranges, int param_count, int trials, float* best_params);

/* 分阶段训练执行 */
int pipeline_run_pretrain_phase(TrainingPipeline* pipeline, int epochs, float* final_loss);
int pipeline_run_deep_train_phase(TrainingPipeline* pipeline, int epochs, float* final_loss);
int pipeline_run_multimodal_phase(TrainingPipeline* pipeline, int epochs, float* final_loss);
int pipeline_run_fine_tune_phase(TrainingPipeline* pipeline, int epochs, float* final_loss);
int pipeline_run_local_phase(TrainingPipeline* pipeline, int epochs, float* final_loss);
int pipeline_run_speech_phase(TrainingPipeline* pipeline, int epochs, float* final_loss);
int pipeline_run_full_training(TrainingPipeline* pipeline, float* final_loss);

/* 训练监控器接口 */
TrainingMonitor* training_pipeline_get_monitor(const TrainingPipeline* pipeline);

/* ============================================================================
 * P0-跨模块集成: 认知系统修正信号注入训练管线
 * ============================================================================
 * 认知系统（self_cognition、deep_correction、deep_reflection）在运行过程中
 * 产生的修正信号（参数修正、反思结果、置信度等）通过此接口注入训练管线，
 * 作为训练数据的补充，在下一次训练迭代中应用。
 *
 * 设计原则：
 *   - 向后兼容：训练管线不可用时优雅降级（返回-1但不崩溃）
 *   - 非阻塞：修正信号仅追加到队列，不阻塞认知系统运行
 *   - 真实数据优先：修正信号作为辅助，不替代真实训练数据
 * ============================================================================ */

/** 认知修正信号来源 */
typedef enum {
    COG_CORRECTION_SOURCE_SELF_TRAINING = 0,   /**< 自我模型训练产生 */
    COG_CORRECTION_SOURCE_DEEP_CORRECTION = 1, /**< 深度修正引擎产生 */
    COG_CORRECTION_SOURCE_DEEP_REFLECTION = 2, /**< 深度反思引擎产生 */
    COG_CORRECTION_SOURCE_METACOGNITION = 3    /**< 元认知系统产生 */
} CognitionCorrectionSource;

/** 认知修正信号 —— 从认知系统传递到训练管线的修正数据 */
typedef struct {
    CognitionCorrectionSource source;          /**< 修正信号来源 */
    float* corrected_params;                   /**< 修正后的LNN参数（NULL表示无参数修正） */
    size_t param_count;                        /**< 参数数量 */
    float* param_gradients;                    /**< 修正梯度方向（可选，NULL表示无） */
    size_t gradient_count;                     /**< 梯度数量 */
    char correction_reason[512];               /**< 修正原因描述 */
    float confidence;                          /**< 修正置信度 [0.0, 1.0] */
    float effectiveness;                       /**< 修正效果评估 [0.0, 1.0] */
    time_t timestamp;                          /**< 修正信号产生时间 */
    int applied;                               /**< 是否已在训练管线中应用 */
    int error_id;                              /**< 关联的错误ID（-1表示无关联错误） */
    int hypothesis_id;                         /**< 关联的假设ID（-1表示无关联假设） */
    float reflection_depth;                    /**< 反思深度评分（来自deep_reflection） */
    float coherence_score;                     /**< 认知一致性评分（来自deep_reflection） */
} CognitionCorrectionSignal;

/** 认知修正信号队列最大容量 */
#define COG_CORRECTION_QUEUE_MAX 64

/**
 * @brief 将认知系统的修正信号注入训练管线
 *
 * 认知系统在完成自我修正/反思后调用此函数，将修正结果推送到训练管线。
 * 训练管线在下一个训练步骤中会检查并应用这些修正信号。
 *
 * @param pipeline 训练管线实例
 * @param signal 修正信号（函数内部会复制数据，调用者保留所有权）
 * @return 0成功，-1参数无效，-2队列已满，-3训练管线未初始化
 */
int training_pipeline_apply_cognition_correction(TrainingPipeline* pipeline,
                                                  const CognitionCorrectionSignal* signal);

/**
 * @brief 获取训练管线中待处理的认知修正信号数量
 *
 * @param pipeline 训练管线实例
 * @return 待处理信号数量，-1参数无效
 */
int training_pipeline_get_pending_corrections(const TrainingPipeline* pipeline);

/**
 * @brief 清除所有待处理的认知修正信号
 *
 * @param pipeline 训练管线实例
 */
void training_pipeline_clear_corrections(TrainingPipeline* pipeline);

#ifdef __cplusplus
}
#endif
#endif
