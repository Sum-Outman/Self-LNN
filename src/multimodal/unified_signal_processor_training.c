/**
 * @file unified_signal_processor_training.c
 * @brief 多模态数据混合策略实现
 *
 * 实现需求20.3: 多模态数据混合策略
 * - 课程学习(Curriculum Learning): 难度渐进式增长
 * - 动态比例(Dynamic Ratio): 根据质量指标动态调整采样比例
 * - 模态丢弃(Modality Dropout): 随机丢弃模态增强鲁棒性
 */

#include "selflnn/multimodal/unified_signal_processor.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "selflnn/utils/secure_random.h"

/* ================================================================
 * 内部辅助
 * ================================================================ */

static float rng_float(void) {
    return secure_random_float();
}

static float clampf(float val, float min, float max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

/* ================================================================
 * 课程学习实现
 * ================================================================ */

float unified_signal_processor_get_curriculum_threshold(
    size_t current_iteration,
    size_t total_iterations,
    const CurriculumConfig* config)
{
    if (!config || !config->enable_curriculum || total_iterations == 0) {
        return 1.0f;
    }

    float progress = (float)current_iteration / (float)total_iterations;
    progress = clampf(progress, 0.0f, 1.0f);

    float threshold = config->difficulty_threshold_start +
                      progress * (config->difficulty_threshold_end -
                                  config->difficulty_threshold_start);
    return clampf(threshold, 0.0f, 1.0f);
}

static int curriculum_filter_sample(
    float sample_difficulty,
    float current_threshold)
{
    if (current_threshold >= 0.999f) return 1;
    return (sample_difficulty <= current_threshold) ? 1 : 0;
}

/* ================================================================
 * 动态比例实现
 * ================================================================ */

int unified_signal_processor_compute_dynamic_ratios(
    const float* modality_quality, size_t quality_count,
    const DynamicRatioConfig* config,
    float* ratios, size_t ratio_count)
{
    if (!modality_quality || quality_count < 4 ||
        !config || !ratios || ratio_count < 4) {
        return -1;
    }

    if (!config->enable_dynamic_ratio) {
        for (size_t i = 0; i < 4; i++) {
            ratios[i] = 0.25f;
        }
        return 0;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < 4; i++) {
        float q = modality_quality[i];
        if (q < 0.001f) q = 0.001f;
        ratios[i] = q;
        sum += q;
    }

    if (sum < 1e-10f) {
        for (size_t i = 0; i < 4; i++) {
            ratios[i] = 0.25f;
        }
        return 0;
    }

    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < 4; i++) {
        ratios[i] *= inv_sum;
        ratios[i] = clampf(ratios[i], config->min_ratio, config->max_ratio);
    }

    float re_sum = 0.0f;
    for (size_t i = 0; i < 4; i++) re_sum += ratios[i];
    if (re_sum > 0.0f) {
        inv_sum = 1.0f / re_sum;
        for (size_t i = 0; i < 4; i++) ratios[i] *= inv_sum;
    }

    return 0;
}

static int select_modality_by_ratio(const float* ratios, size_t ratio_count) {
    if (!ratios || ratio_count < 4) return 0;
    float r = rng_float();
    float cum = 0.0f;
    for (size_t i = 0; i < ratio_count; i++) {
        cum += ratios[i];
        if (r < cum) return (int)i;
    }
    return (int)(ratio_count - 1);
}

/* ================================================================
 * 模态丢弃实现
 * ================================================================ */

int unified_signal_processor_apply_modality_dropout(
    UnifiedSignalProcessor* processor,
    const ModalityDropoutConfig* config,
    size_t current_iteration,
    int* modality_active, size_t active_count)
{
    if (!processor || !config || !modality_active || active_count < 4) {
        return -1;
    }

    /* 从处理器配置获取维度信息 */
    UnifiedSignalProcessorConfig proc_cfg;
    if (unified_signal_processor_get_config(processor, &proc_cfg) == 0) {
        size_t proc_dim = proc_cfg.unified_dimension > 0 ? proc_cfg.unified_dimension : 256;
        (void)proc_dim;
    }
    
    if (!config->enable_modality_dropout) {
        for (size_t i = 0; i < 4; i++) {
            modality_active[i] = 1;
        }
        return 4;
    }

    float decay = powf(1.0f - config->drop_schedule_decay, (float)current_iteration);
    if (decay < 0.0f) decay = 0.0f;

    int active_count_result = 0;

    for (size_t i = 0; i < 4; i++) {
        float effective_rate = config->dropout_rate[i] * decay;
        if (effective_rate < config->min_dropout_rate) {
            effective_rate = config->min_dropout_rate;
        }
        modality_active[i] = (rng_float() >= effective_rate) ? 1 : 0;
        if (modality_active[i]) active_count_result++;
    }

    if (active_count_result < config->ensure_min_modalities) {
        int rescued = config->ensure_min_modalities - active_count_result;
        for (size_t i = 0; i < 4 && rescued > 0; i++) {
            if (!modality_active[i]) {
                modality_active[i] = 1;
                rescued--;
                active_count_result++;
            }
        }
    }

    return active_count_result;
}

/* ================================================================
 * 默认配置
 * ================================================================ */

DataMixingConfig unified_signal_processor_get_default_mixing_config(void) {
    DataMixingConfig config;
    memset(&config, 0, sizeof(DataMixingConfig));

    config.curriculum.enable_curriculum = 1;
    config.curriculum.curriculum_stages = 5;
    config.curriculum.difficulty_threshold_start = 0.3f;
    config.curriculum.difficulty_threshold_end = 1.0f;
    config.curriculum.min_modalities_per_stage = 1;

    config.dynamic_ratio.enable_dynamic_ratio = 1;
    config.dynamic_ratio.ratio_update_rate = 0.1f;
    config.dynamic_ratio.min_ratio = 0.05f;
    config.dynamic_ratio.max_ratio = 0.60f;
    config.dynamic_ratio.update_interval = 100;

    for (int i = 0; i < 4; i++) {
        config.modality_dropout.dropout_rate[i] = 0.3f;
    }
    config.modality_dropout.enable_modality_dropout = 1;
    config.modality_dropout.drop_schedule_decay = 5e-6f;
    config.modality_dropout.min_dropout_rate = 0.05f;
    config.modality_dropout.ensure_min_modalities = 1;

    config.verbose = 0;

    return config;
}

/* ================================================================
 * 批量混合训练主函数
 * ================================================================ */

int unified_signal_processor_train_batch_mixed(
    UnifiedSignalProcessor* processor,
    const VisionInput* vision, size_t vision_count,
    const AudioInput* audio, size_t audio_count,
    const TextInput* text, size_t text_count,
    const SensorInput* sensor, size_t sensor_count,
    const float* targets, size_t target_count,
    const DataMixingConfig* mixing_config,
    const float* modality_quality,
    size_t current_iteration,
    size_t total_iterations,
    float* losses, size_t loss_count)
{
    if (!processor || !mixing_config) return -1;

    size_t total_samples = vision_count + audio_count + text_count + sensor_count;
    if (total_samples == 0) return 0;
    if (!targets || target_count < total_samples) return -1;

    int verbose = mixing_config->verbose;

    float curriculum_threshold = 1.0f;
    if (mixing_config->curriculum.enable_curriculum) {
        curriculum_threshold = unified_signal_processor_get_curriculum_threshold(
            current_iteration, total_iterations, &mixing_config->curriculum);
        if (verbose) {
            printf("unified_signal_processor_train_batch_mixed: 课程阈值=%.4f (迭代%zu/%zu)\n",
                   curriculum_threshold, current_iteration, total_iterations);
        }
    }

    float dynamic_ratios[4] = {0.25f, 0.25f, 0.25f, 0.25f};
    if (mixing_config->dynamic_ratio.enable_dynamic_ratio && modality_quality) {
        int dt = (int)mixing_config->dynamic_ratio.update_interval;
        if (dt <= 0) dt = 1;
        if (current_iteration % (size_t)dt == 0) {
            DynamicRatioConfig adj_config = mixing_config->dynamic_ratio;
            adj_config.ratio_update_rate = 1.0f;
            unified_signal_processor_compute_dynamic_ratios(
                modality_quality, 4, &adj_config,
                dynamic_ratios, 4);
        }
        if (verbose) {
            printf("unified_signal_processor_train_batch_mixed: 动态比例=[%.3f,%.3f,%.3f,%.3f]\n",
                   dynamic_ratios[0], dynamic_ratios[1],
                   dynamic_ratios[2], dynamic_ratios[3]);
        }
    }

    int modality_active[4] = {1, 1, 1, 1};
    if (mixing_config->modality_dropout.enable_modality_dropout) {
        unified_signal_processor_apply_modality_dropout(
            processor, &mixing_config->modality_dropout,
            current_iteration, modality_active, 4);
        if (verbose) {
            printf("unified_signal_processor_train_batch_mixed: 模态活跃=[%d,%d,%d,%d]\n",
                   modality_active[0], modality_active[1],
                   modality_active[2], modality_active[3]);
        }
    }

    size_t trained_count = 0;
    const size_t max_samples = total_samples;
    size_t sample_queue[4] = {0, 0, 0, 0};

    size_t modality_order[4] = {0, 1, 2, 3};
    if (mixing_config->dynamic_ratio.enable_dynamic_ratio) {
        for (int pos = 0; pos < 4; pos++) {
            int chosen = select_modality_by_ratio(dynamic_ratios, 4);
            modality_order[pos] = (size_t)chosen;
            dynamic_ratios[chosen] = 0.0f;
            float re_sum = 0.0f;
            for (int j = 0; j < 4; j++) re_sum += dynamic_ratios[j];
            if (re_sum > 0.0f) {
                float inv = 1.0f / re_sum;
                for (int j = 0; j < 4; j++) dynamic_ratios[j] *= inv;
            }
        }
    }

    size_t sample_counts[4] = {vision_count, audio_count, text_count, sensor_count};

    for (size_t pass = 0; pass < max_samples && trained_count < loss_count; pass++) {
        size_t m = modality_order[pass % 4];
        if (!modality_active[m]) continue;
        if (sample_queue[m] >= sample_counts[m]) continue;

        size_t idx = sample_queue[m];
        sample_queue[m]++;

        float sample_difficulty = 0.5f;
        if (modality_quality && modality_quality[m] > 0.001f) {
            sample_difficulty = 1.0f - modality_quality[m];
            sample_difficulty = clampf(sample_difficulty, 0.0f, 1.0f);
        }

        if (mixing_config->curriculum.enable_curriculum) {
            if (!curriculum_filter_sample(sample_difficulty, curriculum_threshold)) {
                continue;
            }
        }

        const VisionInput single_vision = (m == 0 && vision) ?
            vision[idx] : (VisionInput){NULL, 0, 0, 0, 0, 0.0f};
        const AudioInput single_audio = (m == 1 && audio) ?
            audio[idx] : (AudioInput){NULL, 0, 0.0f, NULL, 0, 0.0f};
        const TextInput single_text = (m == 2 && text) ?
            text[idx] : (TextInput){NULL, 0, NULL, 0, 0.0f};
        const SensorInput single_sensor = (m == 3 && sensor) ?
            sensor[idx] : (SensorInput){NULL, 0, NULL, 0.0f};

        const float* target_ptr = (targets) ? &targets[idx] : NULL;
        float sample_loss = 0.0f;

        int ret = unified_signal_processor_train(processor,
                                                  (m == 0 && vision) ? &single_vision : NULL,
                                                  (m == 1 && audio) ? &single_audio : NULL,
                                                  (m == 2 && text) ? &single_text : NULL,
                                                  (m == 3 && sensor) ? &single_sensor : NULL,
                                                  target_ptr,
                                                  &sample_loss);

        if (ret == 0) {
            if (losses && trained_count < loss_count) {
                losses[trained_count] = sample_loss;
            }
            trained_count++;
        }
    }

    return (int)trained_count;
}
