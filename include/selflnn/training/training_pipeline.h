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
    TRAIN_STAGE_EVALUATION = 5,
    TRAIN_STAGE_IDLE = 6
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
} TrainingPipelineState;

typedef struct {
    size_t pretrain_epochs;
    size_t deep_train_epochs;
    size_t multimodal_epochs;
    size_t fine_tune_epochs;
    size_t local_epochs;
    float pretrain_lr;
    float deep_train_lr;
    float multimodal_lr;
    float fine_tune_lr;
    float local_lr;
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
    int optimizer_type;              /**< 优化器类型 0=SGD 1=Momentum 2=AdaGrad 3=RMSProp 4=Adam 5=AdamW 6=AdaDelta 7=LAMB 8=LARS 9=Ranger 10=NovoGrad */
    int loss_function;               /**< 损失函数 0=MSE 1=MAE 2=CrossEntropy 3=Huber 4=LogCosh */
    char data_directory[512];
    char output_directory[512];
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

#ifdef __cplusplus
}
#endif
#endif
