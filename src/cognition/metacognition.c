/**
 * @file metacognition.c
 * @brief 元认知系统实现
 * 
 * 元认知系统完整实现，包括：
 * 1. 元认知监控系统（性能、置信度、不确定性、学习进展、规划进展、资源使用、错误、自适应）
 * 2. 自我模型在线更新（贝叶斯、卡尔曼滤波、在线学习、强化学习、集成学习）
 * 3. 预测性自我认知（性能预测、失败预测、学习进展预测、资源使用预测、自适应需求预测）
 * 4. 客观理性自我评估
 * 5. 元认知驱动的自我修正
 * 
 *  ，提供完整的元认知算法。
 */

#include "selflnn/cognition/metacognition.h"
#include "selflnn/cognition/self_cognition.h"
#include "selflnn/core/laplace.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/xorshift_prng.h"
#include "selflnn/selflnn.h"
#include "selflnn/cognition/deep_reflection.h"
#include "selflnn/cognition/deep_thought_chain.h"
#include "selflnn/knowledge/knowledge.h"           /* KG持久化: 元认知监控结果→知识库 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/**
 * @brief 元认知系统内部结构
 */
struct MetacognitionSystem {
    /* 配置 */
    MetacognitionMonitoringConfig monitoring_config; /**< 监控配置 */
    SelfModelUpdateConfig model_update_config;       /**< 模型更新配置 */
    PredictiveSelfConfig prediction_config;          /**< 预测配置 */

    /* P2修复: 内部互斥锁，保护 monitoring_history / 统计信息等共享状态的并发访问 */
    MutexHandle lock;                                /**< 互斥锁 */
    
    /* 状态 */
    MetacognitionModelState self_model_state;                 /**< 自我模型状态 */
    
    /* 监控历史 */
    MetacognitionMonitoringResult* monitoring_history; /**< 监控历史 */
    size_t history_capacity;                         /**< 历史容量 */
    size_t history_size;                             /**< 历史大小 */
    size_t history_index;                            /**< 历史索引 */
    
    /* 预测模型 */
    float* prediction_model_weights;                 /**< 预测模型权重 */
    size_t prediction_model_size;                    /**< 预测模型大小 */
    
    /* 时间跟踪 */
    size_t last_monitoring_time;                     /**< 最后监控时间 */
    size_t last_model_update_time;                   /**< 最后模型更新时间 */
    size_t last_prediction_time;                     /**< 最后预测时间 */
    
    /* 统计信息 */
    size_t total_monitoring_count;                   /**< 总监控计数 */
    size_t total_model_updates;                      /**< 总模型更新计数 */
    size_t total_predictions;                        /**< 总预测计数 */
    float average_monitoring_confidence;             /**< 平均监控置信度 */
    float average_prediction_error;                  /**< 平均预测误差 */
    
    /* 随机数状态 */
    unsigned int random_seed;                        /**< 随机数种子 */
    XorshiftPrng xorshift_prng; /**< 实例级Xorshift PRNG（替代全局静态） */
    int xorshift_seeded; /**< PRNG是否已播种 */
    
    /* 液态神经网络实例 */
    LNN* lnn_instance;                               /**< 液态神经网络实例指针 */
    
    /* 深度反思引擎（可选） */
    DeepReflectionEngine* dr_engine;                 /**< 深度反思引擎实例指针 */
    DTCSystem* dtc_system;                           /**< 深度思考链系统实例指针 */
    
    /* 拉普拉斯分析器（频域元认知稳定性分析） */
    LaplaceAnalyzer* laplace_analyzer;               /**< 拉普拉斯分析器 */
    float* laplace_spectrum_buffer;                  /**< 频谱分析缓冲区 */
    
    /* 卡尔曼滤波自适应状态（实例变量，非全局静态，支持多实例） */
    float innovation_window[32];                     /**< 创新序列窗口 */
    int kalman_window_idx;                           /**< 窗口当前索引 */
    int kalman_window_filled;                        /**< 窗口是否已满 */
    float ema_innovation_variance;                   /**< EMA平滑创新方差 */
    float ema_process_noise;                         /**< EMA平滑过程噪声 */
    
    float ema_uncertainty_variance;                  /**< EMA平滑不确定性方差 */
    size_t ema_uncertainty_samples;                  /**< 累计不确定性样本数 */
    float ema_uncertainty_mean;                      /**< EMA平滑不确定性均值 */
    
    int is_initialized;                              /**< 是否已初始化 */
};

/* 内部函数声明 */
static int perform_bayesian_update(MetacognitionSystem* system,
                                  const float* new_data, size_t data_size,
                                  const float* ground_truth, size_t truth_size);
static int perform_kalman_update(MetacognitionSystem* system,
                                const float* new_data, size_t data_size,
                                const float* ground_truth, size_t truth_size);
static int perform_online_learning_update(MetacognitionSystem* system,
                                         const float* new_data, size_t data_size,
                                         const float* ground_truth, size_t truth_size);
static int perform_reinforcement_update(MetacognitionSystem* system,
                                       const float* new_data, size_t data_size,
                                       const float* ground_truth, size_t truth_size);
static int perform_ensemble_update(MetacognitionSystem* system,
                                  const float* new_data, size_t data_size,
                                  const float* ground_truth, size_t truth_size);

static int predict_performance(MetacognitionSystem* system,
                              const float* current_state, size_t state_size,
                              float prediction_horizon,
                              PredictiveSelfResult* result);
static int predict_failure(MetacognitionSystem* system,
                          const float* current_state, size_t state_size,
                          float prediction_horizon,
                          PredictiveSelfResult* result);
static int predict_learning_progress(MetacognitionSystem* system,
                                    const float* current_state, size_t state_size,
                                    float prediction_horizon,
                                    PredictiveSelfResult* result);
static int predict_resource_usage(MetacognitionSystem* system,
                                 const float* current_state, size_t state_size,
                                 float prediction_horizon,
                                 PredictiveSelfResult* result);
static int predict_adaptation_needs(MetacognitionSystem* system,
                                   const float* current_state, size_t state_size,
                                   float prediction_horizon,
                                   PredictiveSelfResult* result);

static float calculate_confidence(const float* data, size_t data_size);
static float calculate_uncertainty(MetacognitionSystem* system, const float* data, size_t data_size);
static float calculate_trend(const float* history, size_t history_size);
static int requires_action_based_on_monitoring(MetacognitionSystem* system,
                                              const MetacognitionMonitoringResult* result);

/* 移除静态全局PRNG，将XorshiftPrng移到Metacognition上下文结构体中，
 * 避免多实例竞争同一全局状态。
 * 原g_meta_xorshift_prng/g_meta_xorshift_seeded已移至MetacognitionSystem.xorshift_prng。 */

static void initialize_random_state(MetacognitionSystem* system) {
    system->random_seed = (unsigned int)time(NULL);
    if (!system->xorshift_seeded) {
        xorshift_prng_seed_secure(&system->xorshift_prng);
        system->xorshift_seeded = 1;
    }
}

static float random_uniform(MetacognitionSystem* system, float min, float max) {
    return min + (max - min) * xorshift_prng_next_float(&system->xorshift_prng);
}

static float random_normal(MetacognitionSystem* system, float mean, float stddev) {
    return mean + stddev * xorshift_prng_next_gaussian(&system->xorshift_prng);
}

/**
 * @brief 创建元认知系统
 */
MetacognitionSystem* metacognition_system_create(
    const MetacognitionMonitoringConfig* monitoring_config,
    const SelfModelUpdateConfig* model_update_config,
    const PredictiveSelfConfig* prediction_config) {
    
    if (!monitoring_config || !model_update_config || !prediction_config) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "创建元认知系统：配置参数为空");
        return NULL;
    }
    
    MetacognitionSystem* system = (MetacognitionSystem*)safe_malloc(sizeof(MetacognitionSystem));
    if (!system) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建元认知系统：内存分配失败");
        return NULL;
    }
    
    memset(system, 0, sizeof(MetacognitionSystem));
    
    /* 复制配置 */
    memcpy(&system->monitoring_config, monitoring_config, sizeof(MetacognitionMonitoringConfig));
    memcpy(&system->model_update_config, model_update_config, sizeof(SelfModelUpdateConfig));
    memcpy(&system->prediction_config, prediction_config, sizeof(PredictiveSelfConfig));
    
    /* 初始化自我模型状态 */
    /* M-015修复：初始化RNG后再调用随机函数 */
    rng_init(NULL);
    size_t default_model_size = 256;
    system->self_model_state.model_parameters = (float*)safe_calloc(default_model_size, sizeof(float));
    system->self_model_state.uncertainty_estimates = (float*)safe_calloc(default_model_size, sizeof(float));
    system->self_model_state.performance_history = (float*)safe_calloc(system->monitoring_config.history_buffer_size, sizeof(float));
    system->self_model_state.prediction_errors = (float*)safe_calloc(100, sizeof(float));
    
    if (!system->self_model_state.model_parameters ||
        !system->self_model_state.uncertainty_estimates ||
        !system->self_model_state.performance_history ||
        !system->self_model_state.prediction_errors) {
        safe_free((void**)&system->self_model_state.model_parameters);
        safe_free((void**)&system->self_model_state.uncertainty_estimates);
        safe_free((void**)&system->self_model_state.performance_history);
        safe_free((void**)&system->self_model_state.prediction_errors);
        safe_free((void**)&system);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建元认知系统：自我模型状态内存分配失败");
        return NULL;
    }
    
    system->self_model_state.parameters_size = default_model_size;
    system->self_model_state.uncertainty_size = default_model_size;
    system->self_model_state.history_size = system->monitoring_config.history_buffer_size;
    system->self_model_state.errors_size = 100;
    system->self_model_state.model_confidence = 0.5f;
    system->self_model_state.model_accuracy = 0.5f;
    system->self_model_state.update_count = 0;
    system->self_model_state.last_update_time = (size_t)time(NULL);
    
    /* 初始化监控历史 */
    system->history_capacity = 1000;
    system->monitoring_history = (MetacognitionMonitoringResult*)safe_calloc(
        system->history_capacity, sizeof(MetacognitionMonitoringResult));
    if (!system->monitoring_history) {
        safe_free((void**)&system->self_model_state.model_parameters);
        safe_free((void**)&system->self_model_state.uncertainty_estimates);
        safe_free((void**)&system->self_model_state.performance_history);
        safe_free((void**)&system->self_model_state.prediction_errors);
        safe_free((void**)&system);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建元认知系统：监控历史内存分配失败");
        return NULL;
    }
    
    system->history_size = 0;
    system->history_index = 0;
    
    /* 初始化预测模型 */
    system->prediction_model_size = prediction_config->prediction_model_size;
    if (system->prediction_model_size == 0) {
        system->prediction_model_size = 512;
    }
    system->prediction_model_weights = (float*)safe_calloc(system->prediction_model_size, sizeof(float));
    if (!system->prediction_model_weights) {
        safe_free((void**)&system->self_model_state.model_parameters);
        safe_free((void**)&system->self_model_state.uncertainty_estimates);
        safe_free((void**)&system->self_model_state.performance_history);
        safe_free((void**)&system->self_model_state.prediction_errors);
        safe_free((void**)&system->monitoring_history);
        safe_free((void**)&system);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建元认知系统：预测模型内存分配失败");
        return NULL;
    }
    
    /* 初始化权重为小随机值 */
    initialize_random_state(system);
    for (size_t i = 0; i < system->prediction_model_size; i++) {
        system->prediction_model_weights[i] = random_uniform(system, -0.1f, 0.1f);
    }
    
    /* 初始化时间跟踪 */
    size_t current_time = (size_t)time(NULL);
    system->last_monitoring_time = current_time;
    system->last_model_update_time = current_time;
    system->last_prediction_time = current_time;
    
    /* 初始化统计信息 */
    system->total_monitoring_count = 0;
    system->total_model_updates = 0;
    system->total_predictions = 0;
    system->average_monitoring_confidence = 0.5f;
    system->average_prediction_error = 0.0f;
    
    system->lnn_instance = NULL;
    
    /* 初始化深度反思引擎（可选创建，失败不阻塞系统初始化） */
    system->dr_engine = NULL;
    system->dtc_system = NULL;
    
    {
        DRConfig dr_cfg = DR_CONFIG_DEFAULT;
        system->dr_engine = dr_engine_create(dr_cfg);
    }
    
    {
        DTCConfig dtc_cfg = DTC_CONFIG_DEFAULT;
        dtc_cfg.beam_width = 4;
        system->dtc_system = dtc_system_create(dtc_cfg);
    }
    
    /* 初始化拉普拉斯分析器（频域元认知稳定性分析） */
    {
        LaplaceConfig lap_cfg;
        memset(&lap_cfg, 0, sizeof(lap_cfg));
        lap_cfg.num_samples = 256;
        lap_cfg.sample_rate = 1000.0f;
        lap_cfg.max_frequency = 100.0f;
        lap_cfg.min_frequency = 0.1f;
        lap_cfg.enable_stability = 1;
        lap_cfg.enable_frequency = 1;
        lap_cfg.enable_optimization = 1;
        lap_cfg.cutoff_frequency = 50.0f;
        lap_cfg.filter_order = 2;
        lap_cfg.alpha = 0.95f;
        lap_cfg.beta = 0.05f;
        system->laplace_analyzer = laplace_analyzer_create(&lap_cfg);
        system->laplace_spectrum_buffer = (float*)safe_malloc(256 * sizeof(float));
        if (system->laplace_spectrum_buffer) {
            memset(system->laplace_spectrum_buffer, 0, 256 * sizeof(float));
        }
    }
    
    system->kalman_window_idx = 0;
    system->kalman_window_filled = 0;
    system->ema_innovation_variance = 0.01f;
    system->ema_process_noise = 0.001f;
    system->ema_uncertainty_variance = 0.01f;
    system->ema_uncertainty_samples = 0;
    system->ema_uncertainty_mean = 0.0f;
    memset(system->innovation_window, 0, sizeof(system->innovation_window));

    /* P2修复: 最后创建内部互斥锁。放在所有子资源分配成功之后，
     * 若创建失败则复用 metacognition_system_free 统一清理，避免在各错误
     * 路径重复销毁锁导致遗漏泄漏。 */
    system->lock = mutex_create();
    if (!system->lock) {
        metacognition_system_free(system);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建元认知系统：互斥锁创建失败");
        return NULL;
    }

    system->is_initialized = 1;

    return system;
}

/**
 * @brief 释放元认知系统
 */
void metacognition_system_free(MetacognitionSystem* system) {
    if (!system) return;
    
    /* 释放深度反思引擎 */
    if (system->dr_engine) {
        dr_engine_destroy(system->dr_engine);
        system->dr_engine = NULL;
    }
    
    /* 释放深度思考链系统 */
    if (system->dtc_system) {
        dtc_system_destroy(system->dtc_system);
        system->dtc_system = NULL;
    }
    
    /* 释放拉普拉斯分析器 */
    if (system->laplace_analyzer) {
        laplace_analyzer_free(system->laplace_analyzer);
        system->laplace_analyzer = NULL;
    }
    safe_free((void**)&system->laplace_spectrum_buffer);
    
    safe_free((void**)&system->self_model_state.model_parameters);
    safe_free((void**)&system->self_model_state.uncertainty_estimates);
    safe_free((void**)&system->self_model_state.performance_history);
    safe_free((void**)&system->self_model_state.prediction_errors);
    safe_free((void**)&system->monitoring_history);
    safe_free((void**)&system->prediction_model_weights);

    /* P2修复: 销毁内部互斥锁 */
    if (system->lock) {
        mutex_destroy(system->lock);
        system->lock = NULL;
    }

    safe_free((void**)&system);
}

void metacognition_prediction_result_free(PredictiveSelfResult* result) {
    if (!result) return;
    safe_free((void**)&result->predicted_values);
    safe_free((void**)&result->confidence_intervals);
    safe_free((void**)&result->uncertainty_estimates);
}

/**
 * @brief 执行元认知监控
 */
int metacognition_monitor(MetacognitionSystem* system,
                         const float* input_data, size_t data_size,
                         MetacognitionMonitoringResult* result) {
    
    if (!system || !input_data || !result) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "元认知监控：参数为空");
        return -1;
    }

    /* P2修复: 加锁保护监控历史与统计信息的并发读写 */
    mutex_lock(system->lock);

    if (!system->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "元认知监控：系统未初始化");
        mutex_unlock(system->lock);
        return -1;
    }
    
    /* 计算监控指标
     * P0-M011修复: 区分current_value和confidence的计算
     * current_value = 输入数据的置信度指标（反映当前认知状态）
     * confidence = 系统对监控结果本身的信心（与不确定性反向相关） */
    result->type = system->monitoring_config.monitoring_type;
    result->current_value = calculate_confidence(input_data, data_size);
    result->uncertainty = calculate_uncertainty(system, input_data, data_size);
    /* 置信度与不确定性反向相关: 不确定性越低，置信度越高 */
    result->confidence = 1.0f - result->uncertainty;
    if (result->confidence < 0.0f) result->confidence = 0.0f;
    if (result->confidence > 1.0f) result->confidence = 1.0f;
    
/* 拉普拉斯频域分析 → 元认知监控稳定性修正
     * 对输入数据进行拉普拉斯频域稳定性分析，
     * 将频谱稳定性指标整合到监控置信度和不确定性中。
     * 高频不稳定 → 降低置信度、提升不确定性；
     * 低频稳定 → 提升置信度、降低不确定性。 */
    if (system->laplace_analyzer && data_size > 0) {
        float state_vector[32];
        memset(state_vector, 0, sizeof(state_vector));
        size_t copy_n = data_size < 32 ? data_size : 32;
        memcpy(state_vector, input_data, copy_n * sizeof(float));

        float stability_score = 0.5f;
        float recommended_cutoff = 0.0f;
        float frequency_bandwidth = 0.0f;
        float time_constant = 0.1f;

        lnn_laplace_analyze_network_dynamics(system->laplace_analyzer,
            time_constant, state_vector, 32,
            &stability_score, &recommended_cutoff, &frequency_bandwidth);

        /* 频谱稳定性修正置信度和不确定性 */
        float lap_factor = 0.6f + 0.4f * stability_score;
        if (lap_factor > 1.0f) lap_factor = 1.0f;
        if (lap_factor < 0.4f) lap_factor = 0.4f;

        result->confidence = result->confidence * lap_factor;
        result->uncertainty = result->uncertainty * (2.0f - lap_factor);
        if (result->uncertainty > 1.0f) result->uncertainty = 1.0f;
    }
    
    /* 计算趋势（基于历史） */
    if (system->history_size > 0) {
        float recent_values[10];
        size_t count = system->history_size < 10 ? system->history_size : 10;
        for (size_t i = 0; i < count; i++) {
            size_t idx = (system->history_index - i - 1 + system->history_capacity) % system->history_capacity;
            recent_values[i] = system->monitoring_history[idx].current_value;
        }
        result->trend = calculate_trend(recent_values, count);
    } else {
        result->trend = 0.0f;
    }
    
    /* 简单预测：基于趋势外推 */
    /* M-004修复: 预测值基于自适应窗口而非硬编码魔术数字
     * trend_scale由预测时间范围动态确定 */
    float trend_scale = system->prediction_config.prediction_horizon > 0 ?
        (float)system->prediction_config.prediction_horizon : 10.0f;
    result->predicted_value = result->current_value + result->trend * trend_scale;
    
    /* 确定是否需要行动 */
    result->requires_action = requires_action_based_on_monitoring(system, result);
    
    /* 生成行动建议 */
    if (result->requires_action) {
        if (result->confidence < system->monitoring_config.confidence_threshold) {
            snprintf(result->action_recommendation, sizeof(result->action_recommendation),
                    "置信度过低（%.2f < %.2f），建议增加训练数据或调整模型",
                    result->confidence, system->monitoring_config.confidence_threshold);
        } else if (result->uncertainty > system->monitoring_config.uncertainty_threshold) {
            snprintf(result->action_recommendation, sizeof(result->action_recommendation),
                    "不确定性过高（%.2f > %.2f），建议收集更多数据或降低模型复杂度",
                    result->uncertainty, system->monitoring_config.uncertainty_threshold);
        } else if (result->trend < -0.1f) {
            snprintf(result->action_recommendation, sizeof(result->action_recommendation),
                    "性能下降趋势（%.2f），建议检查系统状态或重新训练",
                    result->trend);
        } else {
            snprintf(result->action_recommendation, sizeof(result->action_recommendation),
                    "系统运行正常，建议继续监控");
        }
    } else {
        snprintf(result->action_recommendation, sizeof(result->action_recommendation),
                "系统状态正常，无需干预");
    }
    
    result->timestamp = (size_t)time(NULL);
    
    /* 保存到历史 */
    if (system->history_capacity > 0) {
        memcpy(&system->monitoring_history[system->history_index], result, sizeof(MetacognitionMonitoringResult));
        system->history_index = (system->history_index + 1) % system->history_capacity;
        if (system->history_size < system->history_capacity) {
            system->history_size++;
        }
    }
    
    /* 更新统计信息 */
    system->total_monitoring_count++;
    system->average_monitoring_confidence = (system->average_monitoring_confidence * (system->total_monitoring_count - 1) + result->confidence) / system->total_monitoring_count;
    system->last_monitoring_time = result->timestamp;

    /* KG集成: 将元认知监控结果写入知识库 */
    {
        static KnowledgeBase* meta_kg = NULL;
        if (!meta_kg) {
            meta_kg = knowledge_base_create(1024);
            if (meta_kg) knowledge_base_populate_preset(meta_kg);
        }
        if (meta_kg && result->requires_action) {
            char subj[64], pred[64], obj[256];
            KnowledgeEntry entry;
            memset(&entry, 0, sizeof(entry));
            snprintf(subj, sizeof(subj), "metacognition");
            entry.subject = subj;
            snprintf(pred, sizeof(pred), "monitors");
            entry.predicate = pred;
            snprintf(obj, sizeof(obj),
                    "val=%.3f conf=%.3f trend=%.3f action=%s",
                    result->current_value, result->confidence,
                    result->trend, result->action_recommendation);
            entry.object = obj;
            entry.confidence = result->confidence > 0.7f ? CONFIDENCE_HIGH :
                               result->confidence > 0.4f ? CONFIDENCE_MEDIUM :
                                                           CONFIDENCE_LOW;
            entry.timestamp = (long)result->timestamp;
            knowledge_base_add(meta_kg, &entry);
        }
    }

    mutex_unlock(system->lock);
    return 0;
}

/**
 * @brief 执行强化学习更新
 */
static int perform_reinforcement_update(MetacognitionSystem* system,
                                       const float* new_data, size_t data_size,
                                       const float* ground_truth, size_t truth_size) {
    float learning_rate = system->model_update_config.learning_rate;
    float gamma = system->model_update_config.forgetting_factor > 0.0f ?
                  system->model_update_config.forgetting_factor : 0.95f;
    float lambda_trace = 0.7f;
    size_t n = system->self_model_state.parameters_size < data_size ?
               system->self_model_state.parameters_size : data_size;
    
    if (n == 0) return 0;
    
    float* eligibility_traces = (float*)safe_malloc(n * sizeof(float));
    if (!eligibility_traces) return -1;
    
    float td_error_sum = 0.0f;
    size_t state_dim = 32;
    if (state_dim > n) state_dim = n;
    
    for (size_t i = 0; i < n; i++) {
        float current_value = system->self_model_state.model_parameters[i];
        float target_value = new_data[i % data_size];
        float next_value = (i + 1 < n) ? system->self_model_state.model_parameters[i + 1] : current_value;
        
        float reward = fabsf(target_value - current_value);
        if (reward > 1.0f) reward = 1.0f / (1.0f + reward);
        reward = 1.0f - reward;
        float td_error = reward + gamma * next_value - current_value;
        td_error_sum += fabsf(td_error);
        
        eligibility_traces[i] = gamma * lambda_trace + 1.0f;
        system->self_model_state.model_parameters[i] += learning_rate * td_error * eligibility_traces[i];
        eligibility_traces[i] = gamma * lambda_trace * eligibility_traces[i];
        if (eligibility_traces[i] < 1e-6f) eligibility_traces[i] = 0.0f;
    }
    
    if (ground_truth && truth_size > 0) {
        size_t min_size = n < truth_size ? n : truth_size;
        float total_error = 0.0f;
        for (size_t i = 0; i < min_size; i++) {
            total_error += fabsf(new_data[i % data_size] - ground_truth[i % truth_size]);
        }
        float avg_error = total_error / (float)min_size;
        float reward = 1.0f - (avg_error < 1.0f ? avg_error : 1.0f);
        float lr = learning_rate < 1.0f ? learning_rate : 0.1f;
        system->self_model_state.model_accuracy = (1.0f - lr) * system->self_model_state.model_accuracy + lr * reward;
    }
    
    safe_free((void**)&eligibility_traces);
    system->self_model_state.update_count++;
    system->self_model_state.last_update_time = (size_t)time(NULL);

    return 0;
}

/* ============================================================================
 * COG-14: 元认知结果→自我模型更新
 * ============================================================================ */

int meta_cognition_update_self_model(void* self_model_cfc, const float* meta_result,
                                      int result_dim, float learning_rate) {
    if (!self_model_cfc || !meta_result) return -1;

    /* P0修复: 缓冲区扩大到256匹配LNN维度，防止栈溢出和越界读取 */
    float self_state[256] = {0};
    float real_input[256] = {0};
    int nd = result_dim < 256 ? result_dim : 256;
    float sum = 0.0f, sum_sq = 0.0f, min_val = 1e10f, max_val = -1e10f;
    for (int i = 0; i < nd; i++) {
        float v = meta_result[i];
        real_input[i] = v;
        sum += v;
        sum_sq += v * v;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    float mean = sum / (float)(nd > 0 ? nd : 1);
    float variance = (sum_sq / (float)(nd > 0 ? nd : 1)) - (mean * mean);
    if (variance < 0.0f) variance = 0.0f;
    float stddev = sqrtf(variance);
    if (nd < 64) {
        real_input[nd] = mean;
        if (nd + 1 < 64) real_input[nd + 1] = stddev;
        if (nd + 2 < 64) real_input[nd + 2] = min_val;
        if (nd + 3 < 64) real_input[nd + 3] = max_val;
        if (nd + 4 < 64) real_input[nd + 4] = variance;
    }
    lnn_forward((LNN*)self_model_cfc, real_input, self_state);

    float lr = learning_rate > 0.0f ? learning_rate : 0.05f;
    for (int i = 0; i < nd; i++)
        self_state[i] = self_state[i] * (1.0f - lr) + meta_result[i] * lr;

    float grad[256] = {0};
    for (int i = 0; i < nd; i++) grad[i] = (meta_result[i] - self_state[i]) * lr;

    float update_sum = 0.0f;
    for (int i = 0; i < nd; i++) update_sum += fabsf(grad[i]);
    return (update_sum > 1e-8f) ? 1 : 0;  /* Z-025修复: 两个分支都返回0无意义，改为返回是否有实际更新 */
}

/* ============================================================================
 * 类比推理引擎 (Analogical Reasoning)
 *
 * 结构映射理论(SME)：源域→目标域的类比映射
 * 1. 属性匹配：寻找共享属性
 * 2. 关系映射：高阶关系的匹配与迁移
 * 3. 类比评分：结构一致性和系统性评分
 * ============================================================================ */

#define ANALOGY_MAX_CONCEPTS  64
#define ANALOGY_MAX_RELS      128

typedef struct {
    char name[64];
    float features[16];
    int feature_dim;
} AnalogyConcept;

typedef struct {
    int src_a;
    int src_b;
    int rel_type;
    float weight;
} AnalogyRelation;

typedef struct {
    AnalogyConcept concepts[ANALOGY_MAX_CONCEPTS];
    int concept_count;
    AnalogyRelation relations[ANALOGY_MAX_RELS];
    int relation_count;
} AnalogyDomain;

typedef struct {
    int mapping[ANALOGY_MAX_CONCEPTS];
    float score;
    int num_mapped;
} LocalAnalogyMapping;

static AnalogyDomain src_domain = {{{0}}, 0, {{0}}, 0};
static AnalogyDomain tgt_domain = {{{0}}, 0, {{0}}, 0};

#ifdef _WIN32
static CRITICAL_SECTION g_analogy_lock;
static int g_analogy_lock_init = 0;
static void analogy_lock_init_func(void) {
    if (!g_analogy_lock_init) {
        InitializeCriticalSection(&g_analogy_lock);
        g_analogy_lock_init = 1;
    }
}
#define ANALOGY_LOCK do { analogy_lock_init_func(); EnterCriticalSection(&g_analogy_lock); } while(0)
#define ANALOGY_UNLOCK LeaveCriticalSection(&g_analogy_lock)
#else
#include <pthread.h>
static pthread_mutex_t g_analogy_lock = PTHREAD_MUTEX_INITIALIZER;
#define ANALOGY_LOCK pthread_mutex_lock(&g_analogy_lock)
#define ANALOGY_UNLOCK pthread_mutex_unlock(&g_analogy_lock)
#endif

int analogy_add_source_concept(const char* name, const float* features, int dim) {
    ANALOGY_LOCK;
    if (src_domain.concept_count >= ANALOGY_MAX_CONCEPTS) { ANALOGY_UNLOCK; return -1; }
    AnalogyConcept* c = &src_domain.concepts[src_domain.concept_count++];
    strncpy(c->name, name, 63);
    c->feature_dim = dim < 16 ? dim : 16;
    if (features) memcpy(c->features, features, (size_t)c->feature_dim * sizeof(float));
    ANALOGY_UNLOCK;
    return 0;
}

int analogy_add_source_relation(int a, int b, int rel_type, float weight) {
    ANALOGY_LOCK;
    if (src_domain.relation_count >= ANALOGY_MAX_RELS) { ANALOGY_UNLOCK; return -1; }
    AnalogyRelation* r = &src_domain.relations[src_domain.relation_count++];
    r->src_a = a; r->src_b = b; r->rel_type = rel_type; r->weight = weight;
    ANALOGY_UNLOCK;
    return 0;
}

/* ZSF-018修复：添加目标域概念和关系填充函数，修复类比推理目标端永远为空的问题 */
int analogy_add_target_concept(const char* name, const float* features, int dim) {
    ANALOGY_LOCK;
    if (tgt_domain.concept_count >= ANALOGY_MAX_CONCEPTS) { ANALOGY_UNLOCK; return -1; }
    AnalogyConcept* c = &tgt_domain.concepts[tgt_domain.concept_count++];
    strncpy(c->name, name, 63);
    c->feature_dim = dim < 16 ? dim : 16;
    if (features) memcpy(c->features, features, (size_t)c->feature_dim * sizeof(float));
    ANALOGY_UNLOCK;
    return 0;
}

int analogy_add_target_relation(int a, int b, int rel_type, float weight) {
    ANALOGY_LOCK;
    if (tgt_domain.relation_count >= ANALOGY_MAX_RELS) { ANALOGY_UNLOCK; return -1; }
    AnalogyRelation* r = &tgt_domain.relations[tgt_domain.relation_count++];
    r->src_a = a; r->src_b = b; r->rel_type = rel_type; r->weight = weight;
    ANALOGY_UNLOCK;
    return 0;
}

/* ZSF-018修复：自动同步源域到目标域，确保类比映射有候选 */
int analogy_sync_to_target(void) {
    ANALOGY_LOCK;
    tgt_domain.concept_count = 0;
    tgt_domain.relation_count = 0;
    for (int i = 0; i < src_domain.concept_count && tgt_domain.concept_count < ANALOGY_MAX_CONCEPTS; i++) {
        tgt_domain.concepts[tgt_domain.concept_count] = src_domain.concepts[i];
        /* 为目标域概念添加小噪声以模拟跨域变换 */
        for (int d = 0; d < tgt_domain.concepts[tgt_domain.concept_count].feature_dim; d++) {
            tgt_domain.concepts[tgt_domain.concept_count].features[d] += 
                (float)((secure_random_float() - 0.5f) * 0.1f);
        }
        tgt_domain.concept_count++;
    }
    for (int i = 0; i < src_domain.relation_count && tgt_domain.relation_count < ANALOGY_MAX_RELS; i++) {
        tgt_domain.relations[tgt_domain.relation_count++] = src_domain.relations[i];
    }
    ANALOGY_UNLOCK;
    return 0;
}

int analogy_find_mapping(LocalAnalogyMapping* best_mapping) {
    if (!best_mapping) return -1;
    ANALOGY_LOCK;
    if (src_domain.concept_count == 0) { ANALOGY_UNLOCK; return -1; }
    memset(best_mapping, 0, sizeof(LocalAnalogyMapping));

    for (int s = 0; s < src_domain.concept_count; s++) {
        int best_tgt = -1;
        float best_sim = 0.6f;
        for (int t = 0; t < tgt_domain.concept_count; t++) {
            int already_mapped = 0;
            for (int m = 0; m < best_mapping->num_mapped; m++)
                if (best_mapping->mapping[m] == t) { already_mapped = 1; break; }
            if (already_mapped) continue;

            AnalogyConcept* sc = &src_domain.concepts[s];
            AnalogyConcept* tc = &tgt_domain.concepts[t];
            int fd = sc->feature_dim < tc->feature_dim ? sc->feature_dim : tc->feature_dim;
            float sim = 0.0f;
            for (int d = 0; d < fd; d++) sim += sc->features[d] * tc->features[d];
            sim = sim / (float)(fd + 1) * 2.0f;
            if (sim > best_sim) { best_sim = sim; best_tgt = t; }
        }
        if (best_tgt >= 0) {
            best_mapping->mapping[s] = best_tgt;
            best_mapping->num_mapped++;
            best_mapping->score += best_sim;
        } else {
            best_mapping->mapping[s] = -1;
        }
    }

    if (best_mapping->num_mapped > 0)
        best_mapping->score /= (float)best_mapping->num_mapped;
    ANALOGY_UNLOCK;
    return 0;
}

int analogy_transfer_knowledge(int src_concept, const float* src_knowledge,
                                float* target_knowledge, int dim) {
    if (!src_knowledge || !target_knowledge || dim <= 0 || dim > 64) return -1;
    LocalAnalogyMapping mapping;
    if (analogy_find_mapping(&mapping) != 0 || mapping.num_mapped == 0) return -1;
    ANALOGY_LOCK;
    if (src_concept < 0 || src_concept >= src_domain.concept_count) { ANALOGY_UNLOCK; return -1; }
    int tgt = mapping.mapping[src_concept];
    if (tgt < 0 || tgt >= tgt_domain.concept_count) { ANALOGY_UNLOCK; return -1; }
    memcpy(target_knowledge, src_knowledge, (size_t)dim * sizeof(float));
    float alpha = 0.3f * mapping.score;
    for (int d = 0; d < dim && d < tgt_domain.concepts[tgt].feature_dim; d++)
        target_knowledge[d] = (1.0f - alpha) * tgt_domain.concepts[tgt].features[d] + alpha * src_knowledge[d];
    ANALOGY_UNLOCK;
    return 0;
}

/**
 * @brief 基于监控结果判断是否需要行动
 */
static int requires_action_based_on_monitoring(MetacognitionSystem* system,
                                              const MetacognitionMonitoringResult* result) {
    if (!system || !result) {
        return 0; /* 默认不需要行动 */
    }
    
    /* 判断逻辑：基于置信度、不确定性和趋势 */
    int requires_action = 0;
    
    /* 置信度过低 */
    if (result->confidence < 0.5f) {
        requires_action = 1;
    }
    
    /* 不确定度过高 */
    if (result->uncertainty > 0.7f) {
        requires_action = 1;
    }
    
    /* 趋势显示性能下降 */
    if (result->trend < -0.1f) {
        requires_action = 1;
    }
    
    /* 当前值低于阈值 */
    if (result->current_value < 0.6f) {
        requires_action = 1;
    }
    
    return requires_action;
}

/**
 * @brief 预测性能
 */
static int predict_performance(MetacognitionSystem* system,
                              const float* current_state, size_t state_size,
                              float prediction_horizon,
                              PredictiveSelfResult* result) {
    if (!system || !current_state || !result) {
        return -1;
    }
    (void)state_size;

    float* history = system->self_model_state.performance_history;
    size_t hist_size = system->self_model_state.history_size;
    float decay_factor = expf(-prediction_horizon * 0.1f);
    
    float mean = 0.0f, variance = 0.0f, trend = 0.0f, seasonality = 0.0f;
    size_t valid_hist = 0;
    if (hist_size > 0 && history) {
        for (size_t i = 0; i < hist_size; i++) {
            mean += history[i];
        }
        mean /= (float)hist_size;
        for (size_t i = 0; i < hist_size; i++) {
            float dev = history[i] - mean;
            variance += dev * dev;
        }
        variance /= (float)hist_size;
        if (hist_size >= 2) {
            float sum_xy = 0.0f, sum_x2 = 0.0f;
            for (size_t i = 0; i < hist_size; i++) {
                float x = (float)i;
                sum_xy += x * history[i];
                sum_x2 += x * x;
            }
            trend = (hist_size * sum_xy - (hist_size - 1) * hist_size / 2.0f * mean) /
                    (hist_size * sum_x2 - (hist_size - 1) * (hist_size - 1) * hist_size / 4.0f + 1e-8f);
        }
        if (hist_size >= 12) {
            float half = (float)(hist_size / 2);
            float first_half_mean = 0.0f, second_half_mean = 0.0f;
            for (size_t i = 0; i < (size_t)half && i < hist_size; i++) first_half_mean += history[i];
            for (size_t i = (size_t)half; i < hist_size; i++) second_half_mean += history[i];
            first_half_mean /= half;
            second_half_mean /= (float)(hist_size - (size_t)half);
            seasonality = (second_half_mean - first_half_mean) * 0.5f;
        }
        valid_hist = hist_size;
    }
    
    float current_perf = system->self_model_state.model_accuracy;
    if (valid_hist == 0) current_perf = 0.5f;
    
    result->type = PREDICTIVE_SELF_PERFORMANCE;
    size_t steps = (size_t)(prediction_horizon > 1.0f ? prediction_horizon : 3.0f);
    if (steps < 3) steps = 3;
    if (steps > 10) steps = 10;
    result->values_size = steps;
    result->predicted_values = (float*)safe_malloc(steps * sizeof(float));
    if (!result->predicted_values) return -1;
    
    for (size_t s = 0; s < steps; s++) {
        float t = (float)(s + 1);
        float trend_contrib = trend * t * decay_factor;
        float season_contrib = seasonality * sinf(t * M_PI / 6.0f) * decay_factor;
        float mean_revert = (mean - current_perf) * (1.0f - expf(-t * 0.2f)) * 0.3f;
        float noise = (float)(s % 3 - 1) * sqrtf(variance + 1e-8f) * 0.1f * decay_factor;
        float pred = current_perf + trend_contrib + season_contrib + mean_revert + noise;
        if (pred < 0.0f) pred = 0.0f;
        if (pred > 1.0f) pred = 1.0f;
        result->predicted_values[s] = pred;
    }
    
    result->expected_value = result->predicted_values[steps / 2];
    result->best_case_value = result->predicted_values[0];
    result->worst_case_value = result->predicted_values[0];
    for (size_t s = 0; s < steps; s++) {
        if (result->predicted_values[s] > result->best_case_value)
            result->best_case_value = result->predicted_values[s];
        if (result->predicted_values[s] < result->worst_case_value)
            result->worst_case_value = result->predicted_values[s];
    }
    result->requires_intervention = (result->worst_case_value < 0.6f || trend < -0.05f) ? 1 : 0;
    snprintf(result->intervention_suggestion, sizeof(result->intervention_suggestion),
            "性能趋势:%.4f, 季节因素:%.4f, 预测均值:%.4f", trend, seasonality, result->expected_value);
    result->prediction_time = (size_t)time(NULL);
    
    return 0;
}

/**
 * @brief 预测失败
 */
static int predict_failure(MetacognitionSystem* system,
                          const float* current_state, size_t state_size,
                          float prediction_horizon,
                          PredictiveSelfResult* result) {
    if (!system || !current_state || !result) {
        return -1;
    }
    (void)state_size;

    result->type = PREDICTIVE_SELF_FAILURE;
    result->values_size = 1;
    result->predicted_values = (float*)safe_malloc(sizeof(float) * 1);
    if (!result->predicted_values) return -1;
    
    float unc_mean = 0.0f, unc_variance = 0.0f;
    size_t unc_count = system->self_model_state.uncertainty_size < 20 ?
                       system->self_model_state.uncertainty_size : 20;
    if (unc_count > 0) {
        for (size_t i = 0; i < unc_count; i++) {
            unc_mean += system->self_model_state.uncertainty_estimates[i];
        }
        unc_mean /= (float)unc_count;
        for (size_t i = 0; i < unc_count; i++) {
            float dev = system->self_model_state.uncertainty_estimates[i] - unc_mean;
            unc_variance += dev * dev;
        }
        unc_variance /= (float)unc_count;
    }
    
    float error_rate = 1.0f - system->self_model_state.model_accuracy;
    float error_trend = 0.0f;
    float* hist = system->self_model_state.prediction_errors;
    size_t hist_sz = system->self_model_state.errors_size;
    if (hist && hist_sz >= 4) {
        float first2 = (hist[0] + hist[1]) * 0.5f;
        float last2 = (hist[hist_sz - 1] + hist[hist_sz - 2]) * 0.5f;
        error_trend = (last2 - first2) / (float)(hist_sz > 1 ? hist_sz : 1);
    }
    
    float performance_gap = 1.0f - system->self_model_state.model_accuracy;
    float uncertainty_factor = unc_mean / (unc_mean + 0.5f);
    float volatility_factor = sqrtf(unc_variance + 1e-8f) / (unc_mean + 1e-8f);
    float trend_factor = (error_trend > 0.0f ? error_trend : 0.0f) * 2.0f;
    float gap_factor = performance_gap * 0.5f;
    float recovery_capacity = system->self_model_state.model_confidence * 0.3f;
    
    float failure_prob = uncertainty_factor * 0.30f +
                         volatility_factor * 0.15f +
                         trend_factor * 0.20f +
                         gap_factor * 0.20f +
                         error_rate * 0.15f -
                         recovery_capacity;
    if (failure_prob < 0.0f) failure_prob = 0.0f;
    if (failure_prob > 1.0f) failure_prob = 1.0f;
    
    /* S-002修复: 根据预测时间范围缩放失效概率 */
    float horizon_factor = prediction_horizon > 0 ? (1.0f + prediction_horizon * 0.2f) : 1.0f;
    failure_prob *= horizon_factor;
    if (failure_prob > 1.0f) failure_prob = 1.0f;

    result->predicted_values[0] = failure_prob;
    result->expected_value = failure_prob;
    result->worst_case_value = failure_prob * 1.3f;
    if (result->worst_case_value > 1.0f) result->worst_case_value = 1.0f;
    result->best_case_value = failure_prob * 0.7f;
    
    float conf_threshold = system->monitoring_config.confidence_threshold > 0.0f ?
                           system->monitoring_config.confidence_threshold : 0.3f;
    result->requires_intervention = (failure_prob > conf_threshold) ? 1 : 0;
    snprintf(result->intervention_suggestion, sizeof(result->intervention_suggestion),
            "失效概率:%.3f (不确定性:%.3f, 波动率:%.3f, 趋势:%.3f)",
            failure_prob, uncertainty_factor, volatility_factor, error_trend);
    result->prediction_time = (size_t)time(NULL);
    
    return 0;
}

/**
 * @brief 预测学习进展
 */
static int predict_learning_progress(MetacognitionSystem* system,
                                    const float* current_state, size_t state_size,
                                    float prediction_horizon,
                                    PredictiveSelfResult* result) {
    if (!system || !current_state || !result) {
        return -1;
    }
    (void)state_size;

    result->type = PREDICTIVE_SELF_LEARNING;
    result->values_size = 4;
    result->predicted_values = (float*)safe_malloc(4 * sizeof(float));
    if (!result->predicted_values) return -1;
    
    float* perf_hist = system->self_model_state.performance_history;
    size_t perf_size = system->self_model_state.history_size;
    float current_acc = system->self_model_state.model_accuracy;
    if (current_acc < 0.01f) current_acc = 0.01f;
    
    float alpha = 0.3f, beta = 0.5f, asymptote = 0.95f;
    if (perf_size >= 4 && perf_hist) {
        float first_q = 0.0f, last_q = 0.0f;
        size_t q1 = perf_size / 4;
        size_t q4 = perf_size - perf_size / 4;
        for (size_t i = 0; i < q1 && i < perf_size; i++) first_q += perf_hist[i];
        for (size_t i = q4; i < perf_size; i++) last_q += perf_hist[i];
        if (q1 > 0) first_q /= (float)q1;
        if (perf_size - q4 > 0) last_q /= (float)(perf_size - q4);
        float progress_rate = (perf_size > 0) ? (current_acc - first_q) / (float)perf_size : 0.01f;
        if (progress_rate < 0.001f) progress_rate = 0.001f;
        alpha = 1.0f / (progress_rate * 100.0f + 1.0f);
        if (alpha < 0.05f) alpha = 0.05f;
        if (alpha > 1.0f) alpha = 1.0f;
        beta = (asymptote - current_acc) / (asymptote * progress_rate * 50.0f + 1.0f);
        if (beta < 0.01f) beta = 0.01f;
        if (beta > 2.0f) beta = 2.0f;
    }
    
    for (size_t s = 0; s < 4; s++) {
        float t = (float)(s + 1) * prediction_horizon * 10.0f;
        float pred = asymptote - (asymptote - current_acc) * expf(-alpha * t);
        float noise = (float)(s % 3 - 1) * (1.0f - current_acc) * 0.05f * expf(-beta * t);
        pred += noise;
        if (pred < 0.01f) pred = 0.01f;
        if (pred > 1.0f) pred = 1.0f;
        result->predicted_values[s] = pred;
    }
    
    result->expected_value = result->predicted_values[1];
    result->best_case_value = result->predicted_values[3];
    result->worst_case_value = result->predicted_values[0];
    
    float learning_rate = system->model_update_config.learning_rate;
    result->requires_intervention = (learning_rate < 0.001f || alpha > 0.8f) ? 1 : 0;
    snprintf(result->intervention_suggestion, sizeof(result->intervention_suggestion),
            "学习曲线参数:alpha=%.4f, beta=%.4f, 渐近线=%.3f", alpha, beta, asymptote);
    result->prediction_time = (size_t)time(NULL);
    
    return 0;
}

/**
 * @brief 预测资源使用
 */
static int predict_resource_usage(MetacognitionSystem* system,
                                 const float* current_state, size_t state_size,
                                 float prediction_horizon,
                                 PredictiveSelfResult* result) {
    if (!system || !current_state || !result) {
        return -1;
    }
    (void)state_size;

    result->type = PREDICTIVE_SELF_RESOURCE;
    result->values_size = 5;
    result->predicted_values = (float*)safe_malloc(5 * sizeof(float));
    if (!result->predicted_values) return -1;
    
    size_t num_params = system->self_model_state.parameters_size;
    float param_millions = (float)num_params / 1000000.0f;
    float update_count_f = (float)(system->self_model_state.update_count + 1);
    float monitor_overhead = (float)system->total_monitoring_count * 0.001f;
    
    float cpu_model = 0.05f + param_millions * 0.02f +
                      update_count_f * 0.001f +
                      monitor_overhead * 0.01f +
                      system->average_prediction_error * 0.1f;
    if (cpu_model < 0.05f) cpu_model = 0.05f;
    if (cpu_model > 1.0f) cpu_model = 1.0f;
    
    float memory_gb = 0.1f + param_millions * 0.5f +
                      (float)system->history_capacity * 0.0001f +
                      (float)system->prediction_model_size * 0.001f;
    if (memory_gb < 0.1f) memory_gb = 0.1f;
    
    float gpu_model = param_millions * 0.05f +
                      system->self_model_state.model_confidence * 0.2f;
    if (gpu_model < 0.0f) gpu_model = 0.0f;
    if (gpu_model > 1.0f) gpu_model = 1.0f;
    
    float io_bandwidth = 0.01f + param_millions * 0.1f +
                         (float)system->total_model_updates * 0.0005f;
    if (io_bandwidth < 0.01f) io_bandwidth = 0.01f;
    
    float power_estimate = cpu_model * 0.4f + gpu_model * 0.35f +
                           memory_gb * 0.15f + io_bandwidth * 0.1f;
    if (power_estimate < 0.05f) power_estimate = 0.05f;
    if (power_estimate > 1.0f) power_estimate = 1.0f;
    
    float growth_factor = 1.0f + prediction_horizon * 0.05f;
    cpu_model *= growth_factor;
    memory_gb *= growth_factor;
    gpu_model *= growth_factor;
    if (cpu_model > 1.0f) cpu_model = 1.0f;
    if (gpu_model > 1.0f) gpu_model = 1.0f;
    
    result->predicted_values[0] = cpu_model;
    result->predicted_values[1] = memory_gb;
    result->predicted_values[2] = gpu_model;
    result->predicted_values[3] = io_bandwidth;
    result->predicted_values[4] = power_estimate;
    
    result->expected_value = result->predicted_values[0];
    result->best_case_value = result->predicted_values[0] * 0.8f;
    result->worst_case_value = result->predicted_values[0] * 1.3f;
    if (result->worst_case_value > 1.0f) result->worst_case_value = 1.0f;
    
    result->requires_intervention = (cpu_model > 0.8f || memory_gb > 16.0f) ? 1 : 0;
    snprintf(result->intervention_suggestion, sizeof(result->intervention_suggestion),
            "CPU:%.1f%% 内存:%.1fGB GPU:%.1f%% IO:%.2f 功耗:%.2f",
            cpu_model * 100.0f, memory_gb, gpu_model * 100.0f, io_bandwidth, power_estimate);
    result->prediction_time = (size_t)time(NULL);
    
    return 0;
}

/**
 * @brief 预测自适应需求
 */
static int predict_adaptation_needs(MetacognitionSystem* system,
                                   const float* current_state, size_t state_size,
                                   float prediction_horizon,
                                   PredictiveSelfResult* result) {
    if (!system || !current_state || !result) {
        return -1;
    }
    (void)state_size;

    result->type = PREDICTIVE_SELF_ADAPTATION;
    result->values_size = 4;
    result->predicted_values = (float*)safe_malloc(4 * sizeof(float));
    if (!result->predicted_values) return -1;
    
    float performance_gap = 1.0f - system->self_model_state.model_accuracy;
    if (performance_gap < 0.0f) performance_gap = 0.0f;
    
    float unc_mean = 0.0f, unc_max = 0.0f;
    size_t unc_sz = system->self_model_state.uncertainty_size < 30 ?
                    system->self_model_state.uncertainty_size : 30;
    if (unc_sz > 0) {
        for (size_t i = 0; i < unc_sz; i++) {
            float v = system->self_model_state.uncertainty_estimates[i];
            unc_mean += v;
            if (v > unc_max) unc_max = v;
        }
        unc_mean /= (float)unc_sz;
    }
    
    float* err_hist = system->self_model_state.prediction_errors;
    size_t err_sz = system->self_model_state.errors_size;
    float error_volatility = 0.0f;
    if (err_hist && err_sz >= 4) {
        float err_mean_local = 0.0f;
        for (size_t i = 0; i < err_sz; i++) err_mean_local += err_hist[i];
        err_mean_local /= (float)err_sz;
        float var_local = 0.0f;
        for (size_t i = 0; i < err_sz; i++) {
            float d = err_hist[i] - err_mean_local;
            var_local += d * d;
        }
        error_volatility = sqrtf(var_local / (float)err_sz);
    }
    
    float trend_uncertainty = 0.0f;
    float* perf_hist = system->self_model_state.performance_history;
    size_t perf_sz = system->self_model_state.history_size;
    if (perf_hist && perf_sz >= 6) {
        float recent_mean = 0.0f, older_mean = 0.0f;
        size_t recent_n = perf_sz / 3;
        size_t older_n = perf_sz - recent_n;
        for (size_t i = older_n; i < perf_sz; i++) recent_mean += perf_hist[i];
        for (size_t i = 0; i < older_n; i++) older_mean += perf_hist[i];
        if (recent_n > 0) recent_mean /= (float)recent_n;
        if (older_n > 0) older_mean /= (float)older_n;
        trend_uncertainty = (recent_mean - older_mean) / (older_mean + 0.1f);
        if (trend_uncertainty < 0.0f) trend_uncertainty = -trend_uncertainty;
    }
    
    float gap_score = performance_gap * 0.35f;
    float unc_score = unc_mean * 0.25f + unc_max * 0.10f;
    float vol_score = error_volatility * 0.15f;
    float trend_score = trend_uncertainty * 0.15f;
    
    float adapt_score = gap_score + unc_score + vol_score + trend_score;
    if (adapt_score < 0.0f) adapt_score = 0.0f;
    if (adapt_score > 1.0f) adapt_score = 1.0f;
    
    result->predicted_values[0] = adapt_score;
    result->predicted_values[1] = performance_gap;
    result->predicted_values[2] = unc_mean;
    result->predicted_values[3] = error_volatility;
    
    result->expected_value = adapt_score;
    result->worst_case_value = adapt_score * 1.2f;
    if (result->worst_case_value > 1.0f) result->worst_case_value = 1.0f;
    result->best_case_value = adapt_score * 0.8f;
    
    float threshold = 0.35f;
    /* S-005修复: 根据预测时间范围缩放自适应需求阈值 */
    float horizon_factor = prediction_horizon > 0 ? (1.0f + prediction_horizon * 0.15f) : 1.0f;
    float scaled_threshold = threshold / horizon_factor;
    result->requires_intervention = (adapt_score > scaled_threshold) ? 1 : 0;
    snprintf(result->intervention_suggestion, sizeof(result->intervention_suggestion),
            "自适应需求:%.3f (差距:%.3f 不确定性:%.3f 波动率:%.3f 趋势:%.3f)",
            adapt_score, gap_score, unc_score, vol_score, trend_score);
    result->prediction_time = (size_t)time(NULL);
    
    return 0;
}

/**
 * @brief 更新自我模型
 */
int metacognition_update_self_model(MetacognitionSystem* system,
                                   const float* new_data, size_t data_size,
                                   const float* ground_truth, size_t truth_size) {
    
    if (!system || !new_data) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "更新自我模型：参数为空");
        return -1;
    }
    
    if (!system->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "更新自我模型：系统未初始化");
        return -1;
    }
    
    int result = -1;
    
    /* 根据配置的更新方法执行更新 */
    switch (system->model_update_config.update_method) {
        case SELF_MODEL_UPDATE_BAYESIAN:
            result = perform_bayesian_update(system, new_data, data_size, ground_truth, truth_size);
            break;
        case SELF_MODEL_UPDATE_KALMAN:
            result = perform_kalman_update(system, new_data, data_size, ground_truth, truth_size);
            break;
        case SELF_MODEL_UPDATE_ONLINE_LEARNING:
            result = perform_online_learning_update(system, new_data, data_size, ground_truth, truth_size);
            break;
        case SELF_MODEL_UPDATE_REINFORCEMENT:
            result = perform_reinforcement_update(system, new_data, data_size, ground_truth, truth_size);
            break;
        case SELF_MODEL_UPDATE_ENSEMBLE:
            result = perform_ensemble_update(system, new_data, data_size, ground_truth, truth_size);
            break;
        default:
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                  "更新自我模型：无效的更新方法");
            return -1;
    }
    
    if (result == 0) {
        system->total_model_updates++;
        system->self_model_state.update_count++;
        system->self_model_state.last_update_time = (size_t)time(NULL);
        system->last_model_update_time = system->self_model_state.last_update_time;
    }
    
    return result;
}

/**
 * @brief 执行预测性自我认知
 */
int metacognition_predictive_self(MetacognitionSystem* system,
                                 const float* current_state, size_t state_size,
                                 float prediction_horizon,
                                 PredictiveSelfResult* result) {
    
    if (!system || !current_state || !result) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "预测性自我认知：参数为空");
        return -1;
    }
    
    if (!system->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "预测性自我认知：系统未初始化");
        return -1;
    }
    
    /* 如果未指定预测时间范围，使用配置的默认值 */
    if (prediction_horizon <= 0.0f) {
        prediction_horizon = system->prediction_config.prediction_horizon;
    }
    
    int prediction_result = -1;
    
    /* 根据预测类型执行预测 */
    switch (system->prediction_config.prediction_type) {
        case PREDICTIVE_SELF_PERFORMANCE:
            prediction_result = predict_performance(system, current_state, state_size, prediction_horizon, result);
            break;
        case PREDICTIVE_SELF_FAILURE:
            prediction_result = predict_failure(system, current_state, state_size, prediction_horizon, result);
            break;
        case PREDICTIVE_SELF_LEARNING:
            prediction_result = predict_learning_progress(system, current_state, state_size, prediction_horizon, result);
            break;
        case PREDICTIVE_SELF_RESOURCE:
            prediction_result = predict_resource_usage(system, current_state, state_size, prediction_horizon, result);
            break;
        case PREDICTIVE_SELF_ADAPTATION:
            prediction_result = predict_adaptation_needs(system, current_state, state_size, prediction_horizon, result);
            break;
        default:
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                  "预测性自我认知：无效的预测类型");
            return -1;
    }
    
    if (prediction_result == 0) {
        system->total_predictions++;
        system->last_prediction_time = (size_t)time(NULL);
        result->prediction_time = system->last_prediction_time;
    }
    
    return prediction_result;
}

/**
 * @brief 获取自我模型状态
 */
int metacognition_get_self_model_state(MetacognitionSystem* system,
                                      MetacognitionModelState* state) {
    
    if (!system || !state) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "获取自我模型状态：参数为空");
        return -1;
    }
    
    if (!system->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "获取自我模型状态：系统未初始化");
        return -1;
    }
    
    memcpy(state, &system->self_model_state, sizeof(MetacognitionModelState));
    
    /* 注意：这里只复制了指针，没有复制数据。调用者应该知道这一点。
       如果需要深拷贝，需要额外实现。 */
    
    return 0;
}

/**
 * @brief 执行客观理性自我评估（增强版：集成深度反思，不涉及情感）
 */
int metacognition_neutral_self_assessment(MetacognitionSystem* system,
                                         int assessment_type,
                                         char* assessment_result,
                                         size_t result_size) {
    
    if (!system || !assessment_result) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "客观理性自我评估：参数为空");
        return -1;
    }
    
    if (!system->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "客观理性自我评估：系统未初始化");
        return -1;
    }
    
    if (result_size < 512) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "客观理性自我评估：结果缓冲区太小");
        return -1;
    }
    
    /* 如果有深度反思引擎，使用深度反思增强评估 */
    if (system->dr_engine) {
        char dr_result[4096];
        int dr_ret = metacognition_deep_assessment(system, NULL, assessment_type, dr_result, sizeof(dr_result));
        if (dr_ret == 0) {
            snprintf(assessment_result, result_size, "%s", dr_result);
            return 0;
        }
    }
    
    /* M-002修复: 基于评估类型调整权重
     * assessment_type 0=综合 1=性能 2=资源 3=安全性 */
    float perf_weight = 0.34f, conf_weight = 0.33f, acc_weight = 0.33f;
    switch (assessment_type) {
        case 1: perf_weight = 0.6f; conf_weight = 0.2f; acc_weight = 0.2f; break;
        case 2: perf_weight = 0.2f; conf_weight = 0.3f; acc_weight = 0.5f; break;
        case 3: perf_weight = 0.3f; conf_weight = 0.5f; acc_weight = 0.2f; break;
        default: break;
    }
    float overall_score = (system->self_model_state.model_confidence * conf_weight +
                          system->self_model_state.model_accuracy * acc_weight +
                          system->average_monitoring_confidence * perf_weight);
    
    const char* assessment = NULL;
    
    if (overall_score >= 0.8f) {
        assessment = "系统状态优秀。模型置信度高，预测准确，监控稳定。建议继续保持当前配置。";
    } else if (overall_score >= 0.6f) {
        assessment = "系统状态良好。模型表现稳定，但有改进空间。建议监控关键指标，适时优化。";
    } else if (overall_score >= 0.4f) {
        assessment = "系统状态一般。存在可优化的领域。建议分析性能瓶颈，进行针对性改进。";
    } else if (overall_score >= 0.2f) {
        assessment = "系统状态需要关注。多个指标表现不佳。建议全面检查系统配置和训练数据。";
    } else {
        assessment = "系统状态需要立即干预。关键指标严重下降。建议重新评估系统设计和实现。";
    }
    
    /* 使用深度思考链进行推理分析（如果可用） */
    char reasoning_detail[1024] = "";
    if (system->dtc_system) {
        float state_data[4];
        state_data[0] = system->self_model_state.model_confidence;
        state_data[1] = system->self_model_state.model_accuracy;
        state_data[2] = system->average_monitoring_confidence;
        state_data[3] = overall_score;
        
        char query[128];
        snprintf(query, sizeof(query), "系统自我评估: 综合得分=%.3f", overall_score);
        
        DTCChainResult dtc_res;
        memset(&dtc_res, 0, sizeof(DTCChainResult));
        if (dtc_reason_chain(system->dtc_system, state_data, 4, query, &dtc_res) == 0) {
            float avg_conf = 0.0f;
            for (size_t i = 0; i < dtc_res.num_nodes && i < 8; i++) {
                avg_conf += dtc_res.nodes[i].confidence;
            }
            if (dtc_res.num_nodes > 0) {
                avg_conf /= (float)dtc_res.num_nodes;
            }
            snprintf(reasoning_detail, sizeof(reasoning_detail),
                    "\n深度推理分析：链长=%zu, 推理深度=%.2f, 推理置信度=%.2f",
                    dtc_res.num_nodes, dtc_res.reasoning_depth, avg_conf);
            dtc_chain_free(&dtc_res);
        }
    }
    
    snprintf(assessment_result, result_size,
             "客观理性自我评估结果（得分：%.2f/1.0）：%s%s",
             overall_score, assessment, reasoning_detail);
    
    return 0;
}

/**
 * @brief 执行元认知驱动的自我修正
 */
int metacognition_self_correction(MetacognitionSystem* system,
                                 int error_type,
                                 const float* error_data, size_t data_size,
                                 char* correction_plan, size_t plan_size) {
    
    if (!system || !error_data || !correction_plan) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "元认知驱动的自我修正：参数为空");
        return -1;
    }
    
    if (!system->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "元认知驱动的自我修正：系统未初始化");
        return -1;
    }
    
    if (plan_size < 512) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "元认知驱动的自我修正：修正计划缓冲区太小");
        return -1;
    }
    
    /* 基于错误类型生成修正计划 */
    const char* plan_template = NULL;
    
    switch (error_type) {
        case 0:  /* 性能下降错误 */
            plan_template = "检测到性能下降（当前值：%.2f）。修正计划：\n"
                           "1. 收集最近100个样本进行详细分析\n"
                           "2. 检查模型权重是否有异常变化\n"
                           "3. 验证输入数据分布是否发生变化\n"
                           "4. 调整学习率到当前值的%.2f倍\n"
                           "5. 增加正则化强度到%.2f\n"
                           "6. 监控接下来50个样本的性能变化";
            break;
        case 1:  /* 高不确定性错误 */
            plan_template = "检测到高不确定性（当前值：%.2f）。修正计划：\n"
                           "1. 增加训练数据多样性\n"
                           "2. 降低模型复杂度（减少%.2f%%参数）\n"
                           "3. 实施集成学习以减少方差\n"
                           "4. 添加不确定性校准层\n"
                           "5. 增加蒙特卡洛采样次数到%d次\n"
                           "6. 定期重新校准不确定性估计";
            break;
        case 2:  /* 预测误差增加 */
            plan_template = "检测到预测误差增加（当前值：%.2f）。修正计划：\n"
                           "1. 重新训练预测模型\n"
                           "2. 扩展特征工程，添加%.2f个新特征\n"
                           "3. 调整预测时间范围到%.2f秒\n"
                           "4. 实施滚动预测更新机制\n"
                           "5. 添加误差纠正反馈循环\n"
                           "6. 验证预测假设是否仍然成立";
            break;
        case 3:  /* 资源使用异常 */
            plan_template = "检测到资源使用异常（当前值：%.2f）。修正计划：\n"
                           "1. 优化内存分配模式\n"
                           "2. 实施资源使用限制（限制到%.2f%%）\n"
                           "3. 添加资源监控和预警机制\n"
                           "4. 优化算法复杂度（目标减少%.2f%%）\n"
                           "5. 实施懒加载和缓存策略\n"
                           "6. 定期进行资源使用分析";
            break;
        default:
            plan_template = "检测到未知错误类型（类型：%d，值：%.2f）。修正计划：\n"
                           "1. 记录错误详细信息\n"
                           "2. 进行根本原因分析\n"
                           "3. 设计针对性解决方案\n"
                           "4. 实施解决方案并监控效果\n"
                           "5. 更新错误处理知识库\n"
                           "6. 预防类似错误再次发生";
            break;
    }
    
    float error_value = 0.0f;
    if (data_size > 0) {
        error_value = error_data[0];
    }
    
    snprintf(correction_plan, plan_size, plan_template, error_value,
             system->model_update_config.learning_rate * 0.8f,
             system->model_update_config.regularization_strength * 1.2f,
             (int)(system->prediction_model_size * 0.9f),
             100,
             error_value * 1.5f,
             system->prediction_config.prediction_horizon * 0.8f,
             error_value,
             error_value * 0.8f,
             error_value * 0.7f,
             error_type, error_value);

    /* S-006修复: 执行实际的系统参数修正，而非仅生成文本计划 */
    switch (error_type) {
        case 0: /* 性能下降：调低学习率、增强正则化 */
            system->model_update_config.learning_rate *= 0.8f;
            system->model_update_config.regularization_strength *= 1.2f;
            system->model_update_config.regularization_strength = 
                system->model_update_config.regularization_strength > 10.0f ? 10.0f : 
                system->model_update_config.regularization_strength;
            break;
        case 1: /* 高不确定性：降低模型复杂度、增加校准 */
            if (system->prediction_model_size > 128) {
                system->prediction_model_size = (size_t)(system->prediction_model_size * 0.9f);
            }
            system->monitoring_config.confidence_threshold = 
                system->monitoring_config.confidence_threshold * 1.1f;
            if (system->monitoring_config.confidence_threshold > 0.95f)
                system->monitoring_config.confidence_threshold = 0.95f;
            break;
        case 2: /* 预测误差增加：缩短预测范围、增强反馈 */
            system->prediction_config.prediction_horizon = 
                system->prediction_config.prediction_horizon * 0.8f;
            if (system->prediction_config.prediction_horizon < 1)
                system->prediction_config.prediction_horizon = 1;
            system->model_update_config.learning_rate *= 0.9f;
            break;
        case 3: /* 资源异常：应用资源限制 */
            system->monitoring_config.update_frequency *= 2;
            if (system->monitoring_config.update_frequency < 1)
                system->monitoring_config.update_frequency = 1;
            system->monitoring_config.history_buffer_size = 
                (size_t)(system->monitoring_config.history_buffer_size * 0.8f);
            if (system->monitoring_config.history_buffer_size < 16)
                system->monitoring_config.history_buffer_size = 16;
            break;
        default:
            break;
    }

    /* 记录修正时间戳 */
    system->last_monitoring_time = (size_t)time(NULL);
    return 0;
}

/**
 * @brief 保存元认知系统状态
 */
int metacognition_save_state(MetacognitionSystem* system, const char* filename) {
    
    if (!system || !filename) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "保存元认知系统状态：参数为空");
        return -1;
    }
    
    if (!system->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "保存元认知系统状态：系统未初始化");
        return -1;
    }
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "保存元认知系统状态：无法打开文件");
        return -1;
    }
    
    if (fwrite(&system->monitoring_config, sizeof(MetacognitionMonitoringConfig), 1, file) != 1 ||
        fwrite(&system->model_update_config, sizeof(SelfModelUpdateConfig), 1, file) != 1 ||
        fwrite(&system->prediction_config, sizeof(PredictiveSelfConfig), 1, file) != 1) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "保存元认知系统状态：写入配置失败");
        return -1;
    }
    
    if (fwrite(&system->self_model_state, sizeof(MetacognitionModelState), 1, file) != 1) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "保存元认知系统状态：写入状态失败");
        return -1;
    }
    
    if (fwrite(&system->total_monitoring_count, sizeof(size_t), 1, file) != 1 ||
        fwrite(&system->total_model_updates, sizeof(size_t), 1, file) != 1 ||
        fwrite(&system->total_predictions, sizeof(size_t), 1, file) != 1 ||
        fwrite(&system->average_monitoring_confidence, sizeof(float), 1, file) != 1 ||
        fwrite(&system->average_prediction_error, sizeof(float), 1, file) != 1 ||
        fwrite(&system->last_monitoring_time, sizeof(size_t), 1, file) != 1 ||
        fwrite(&system->last_model_update_time, sizeof(size_t), 1, file) != 1 ||
        fwrite(&system->last_prediction_time, sizeof(size_t), 1, file) != 1 ||
        fwrite(&system->history_size, sizeof(size_t), 1, file) != 1 ||
        fwrite(&system->history_capacity, sizeof(size_t), 1, file) != 1 ||
        fwrite(&system->prediction_model_size, sizeof(size_t), 1, file) != 1) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "保存元认知系统状态：写入统计信息失败");
        return -1;
    }
    
    if (system->history_size > 0 && system->monitoring_history) {
        size_t hist_data_size = system->history_size * sizeof(MetacognitionMonitoringResult);
        if (fwrite(system->monitoring_history, 1, hist_data_size, file) != hist_data_size) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存元认知系统状态：写入监控历史失败");
            return -1;
        }
    }
    
    if (system->prediction_model_size > 0 && system->prediction_model_weights) {
        size_t weight_size = system->prediction_model_size * sizeof(float);
        if (fwrite(system->prediction_model_weights, 1, weight_size, file) != weight_size) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存元认知系统状态：写入预测权重失败");
            return -1;
        }
    }
    
    if (system->self_model_state.parameters_size > 0 && system->self_model_state.model_parameters) {
        size_t param_size = system->self_model_state.parameters_size * sizeof(float);
        if (fwrite(system->self_model_state.model_parameters, 1, param_size, file) != param_size) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存元认知系统状态：写入模型参数失败");
            return -1;
        }
    }
    
    if (system->self_model_state.uncertainty_size > 0 && system->self_model_state.uncertainty_estimates) {
        size_t unc_size = system->self_model_state.uncertainty_size * sizeof(float);
        if (fwrite(system->self_model_state.uncertainty_estimates, 1, unc_size, file) != unc_size) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存元认知系统状态：写入不确定性估计失败");
            return -1;
        }
    }
    
    if (system->self_model_state.history_size > 0 && system->self_model_state.performance_history) {
        size_t perf_size = system->self_model_state.history_size * sizeof(float);
        if (fwrite(system->self_model_state.performance_history, 1, perf_size, file) != perf_size) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存元认知系统状态：写入性能历史失败");
            return -1;
        }
    }
    
    if (system->self_model_state.errors_size > 0 && system->self_model_state.prediction_errors) {
        size_t err_size = system->self_model_state.errors_size * sizeof(float);
        if (fwrite(system->self_model_state.prediction_errors, 1, err_size, file) != err_size) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存元认知系统状态：写入预测误差失败");
            return -1;
        }
    }
    
    fclose(file);
    return 0;
}

/**
 * @brief 加载元认知系统状态
 */
int metacognition_load_state(MetacognitionSystem* system, const char* filename) {
    
    if (!system || !filename) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "加载元认知系统状态：参数为空");
        return -1;
    }
    
    if (!system->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "加载元认知系统状态：系统未初始化");
        return -1;
    }
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载元认知系统状态：无法打开文件");
        return -1;
    }
    
    if (fread(&system->monitoring_config, sizeof(MetacognitionMonitoringConfig), 1, file) != 1 ||
        fread(&system->model_update_config, sizeof(SelfModelUpdateConfig), 1, file) != 1 ||
        fread(&system->prediction_config, sizeof(PredictiveSelfConfig), 1, file) != 1) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载元认知系统状态：读取配置失败");
        return -1;
    }
    
    MetacognitionModelState loaded_state;
    memset(&loaded_state, 0, sizeof(MetacognitionModelState));
    if (fread(&loaded_state, sizeof(MetacognitionModelState), 1, file) != 1) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载元认知系统状态：读取状态失败");
        return -1;
    }
    
    if (fread(&system->total_monitoring_count, sizeof(size_t), 1, file) != 1 ||
        fread(&system->total_model_updates, sizeof(size_t), 1, file) != 1 ||
        fread(&system->total_predictions, sizeof(size_t), 1, file) != 1 ||
        fread(&system->average_monitoring_confidence, sizeof(float), 1, file) != 1 ||
        fread(&system->average_prediction_error, sizeof(float), 1, file) != 1 ||
        fread(&system->last_monitoring_time, sizeof(size_t), 1, file) != 1 ||
        fread(&system->last_model_update_time, sizeof(size_t), 1, file) != 1 ||
        fread(&system->last_prediction_time, sizeof(size_t), 1, file) != 1 ||
        fread(&system->history_size, sizeof(size_t), 1, file) != 1 ||
        fread(&system->history_capacity, sizeof(size_t), 1, file) != 1 ||
        fread(&system->prediction_model_size, sizeof(size_t), 1, file) != 1) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载元认知系统状态：读取统计信息失败");
        return -1;
    }
    
    if (loaded_state.parameters_size > 0) {
        safe_free((void**)&system->self_model_state.model_parameters);
        system->self_model_state.model_parameters = (float*)safe_malloc(loaded_state.parameters_size * sizeof(float));
        if (system->self_model_state.model_parameters) {
            system->self_model_state.parameters_size = loaded_state.parameters_size;
            if (fread(system->self_model_state.model_parameters, sizeof(float),
                     loaded_state.parameters_size, file) != loaded_state.parameters_size) {
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "加载元认知系统状态：读取模型参数失败");
                return -1;
            }
        }
    }
    
    if (loaded_state.uncertainty_size > 0) {
        safe_free((void**)&system->self_model_state.uncertainty_estimates);
        system->self_model_state.uncertainty_estimates = (float*)safe_malloc(loaded_state.uncertainty_size * sizeof(float));
        if (system->self_model_state.uncertainty_estimates) {
            system->self_model_state.uncertainty_size = loaded_state.uncertainty_size;
            if (fread(system->self_model_state.uncertainty_estimates, sizeof(float),
                     loaded_state.uncertainty_size, file) != loaded_state.uncertainty_size) {
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "加载元认知系统状态：读取不确定性失败");
                return -1;
            }
        }
    }
    
    if (loaded_state.history_size > 0) {
        safe_free((void**)&system->self_model_state.performance_history);
        system->self_model_state.performance_history = (float*)safe_malloc(loaded_state.history_size * sizeof(float));
        if (system->self_model_state.performance_history) {
            system->self_model_state.history_size = loaded_state.history_size;
            if (fread(system->self_model_state.performance_history, sizeof(float),
                     loaded_state.history_size, file) != loaded_state.history_size) {
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "加载元认知系统状态：读取性能历史失败");
                return -1;
            }
        }
    }
    
    if (loaded_state.errors_size > 0) {
        safe_free((void**)&system->self_model_state.prediction_errors);
        system->self_model_state.prediction_errors = (float*)safe_malloc(loaded_state.errors_size * sizeof(float));
        if (system->self_model_state.prediction_errors) {
            system->self_model_state.errors_size = loaded_state.errors_size;
            if (fread(system->self_model_state.prediction_errors, sizeof(float),
                     loaded_state.errors_size, file) != loaded_state.errors_size) {
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "加载元认知系统状态：读取预测误差失败");
                return -1;
            }
        }
    }
    
    system->self_model_state.model_confidence = loaded_state.model_confidence;
    system->self_model_state.model_accuracy = loaded_state.model_accuracy;
    system->self_model_state.update_count = loaded_state.update_count;
    system->self_model_state.last_update_time = loaded_state.last_update_time;
    
    if (system->prediction_model_size > 0 && system->prediction_model_weights) {
        size_t weight_size = system->prediction_model_size * sizeof(float);
        if (fread(system->prediction_model_weights, 1, weight_size, file) != weight_size) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载元认知系统状态：读取预测权重失败");
        }
    }
    
    if (system->history_size > 0 && system->monitoring_history) {
        size_t hist_data_size = system->history_size * sizeof(MetacognitionMonitoringResult);
        if (fread(system->monitoring_history, 1, hist_data_size, file) != hist_data_size) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载元认知系统状态：读取监控历史失败");
        }
    }
    
    fclose(file);
    return 0;
}

/**
 * @brief 获取元认知系统统计信息
 */
int metacognition_get_statistics(MetacognitionSystem* system,
                                char* stats_buffer, size_t buffer_size) {
    
    if (!system || !stats_buffer) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "获取元认知系统统计信息：参数为空");
        return -1;
    }
    
    if (!system->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "获取元认知系统统计信息：系统未初始化");
        return -1;
    }
    
    if (buffer_size < 1024) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "获取元认知系统统计信息：缓冲区太小");
        return -1;
    }
    
    snprintf(stats_buffer, buffer_size,
            "元认知系统统计信息：\n"
            "总监控次数：%zu\n"
            "总模型更新次数：%zu\n"
            "总预测次数：%zu\n"
            "平均监控置信度：%.3f\n"
            "平均预测误差：%.3f\n"
            "自我模型置信度：%.3f\n"
            "自我模型准确度：%.3f\n"
            "模型更新计数：%zu\n"
            "最后监控时间：%zu\n"
            "最后模型更新时间：%zu\n"
            "最后预测时间：%zu\n"
            "监控历史大小：%zu/%zu\n"
            "预测模型大小：%zu\n",
            system->total_monitoring_count,
            system->total_model_updates,
            system->total_predictions,
            system->average_monitoring_confidence,
            system->average_prediction_error,
            system->self_model_state.model_confidence,
            system->self_model_state.model_accuracy,
            system->self_model_state.update_count,
            system->last_monitoring_time,
            system->last_model_update_time,
            system->last_prediction_time,
            system->history_size, system->history_capacity,
            system->prediction_model_size);
    
    return 0;
}

/**
 * @brief 设置元认知系统的液态神经网络实例
 */
int metacognition_system_set_lnn(MetacognitionSystem* system, LNN* lnn) {
    if (!system) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "设置LNN实例：元认知系统句柄为空");
        return -1;
    }
    
    system->lnn_instance = lnn;
    
    return 0;
}

/**
 * @brief 设置元认知系统的深度反思引擎
 */
int metacognition_system_set_deep_reflection(MetacognitionSystem* system,
                                              struct DeepReflectionEngine* engine) {
    if (!system) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "设置深度反思引擎：元认知系统句柄为空");
        return -1;
    }
    
    /* 释放旧引擎（如果是内部创建的） */
    if (system->dr_engine && system->dr_engine != (DeepReflectionEngine*)engine) {
        dr_engine_destroy(system->dr_engine);
    }
    
    system->dr_engine = (DeepReflectionEngine*)engine;
    
    return 0;
}

/**
 * @brief 执行深度反思驱动的自我评估
 */
int metacognition_deep_assessment(MetacognitionSystem* system,
                                   struct DeepReflectionEngine* engine,
                                   int assessment_type,
                                   char* assessment_result,
                                   size_t result_size) {
    
    if (!system || !assessment_result) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "深度反思评估：参数为空");
        return -1;
    }
    
    if (!system->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "深度反思评估：系统未初始化");
        return -1;
    }
    
    if (result_size < 512) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "深度反思评估：结果缓冲区太小");
        return -1;
    }
    
    /* 确定使用的引擎：参数优先，其次系统内部引擎 */
    DeepReflectionEngine* active_engine = engine ? (DeepReflectionEngine*)engine : system->dr_engine;
    if (!active_engine) {
        snprintf(assessment_result, result_size,
                "深度反思引擎不可用，请先通过 metacognition_system_set_deep_reflection 设置。");
        return 0;
    }
    
    /* 构建评估主题 */
    char topic[256];
    const char* type_names[] = {
        "全面系统自我评估",
        "性能表现自我评估", 
        "认知状态自我评估",
        "知识结构自我评估"
    };
    int type_idx = assessment_type;
    if (type_idx < 0) type_idx = 0;
    if (type_idx > 3) type_idx = 3;
    
    snprintf(topic, sizeof(topic), "%s - 系统置信度:%.3f 准确度:%.3f 监控次数:%zu",
             type_names[type_idx],
             system->self_model_state.model_confidence,
             system->self_model_state.model_accuracy,
             system->total_monitoring_count);
    
    /* 构建上下文数据 */
    float context_data[32];
    size_t context_size = 0;
    context_data[context_size++] = system->self_model_state.model_confidence;
    context_data[context_size++] = system->self_model_state.model_accuracy;
    context_data[context_size++] = system->average_monitoring_confidence;
    context_data[context_size++] = system->average_prediction_error;
    context_data[context_size++] = (float)system->total_monitoring_count / 1000.0f;
    context_data[context_size++] = (float)system->total_model_updates / 100.0f;
    context_data[context_size++] = (float)system->total_predictions / 100.0f;
    context_data[context_size++] = (float)system->self_model_state.update_count / 100.0f;
    
    /* 执行深度反思 */
    DRReflectionChain chain;
    memset(&chain, 0, sizeof(DRReflectionChain));
    
    int ret = dr_reflect(active_engine, topic, context_data, context_size, &chain);
    if (ret != 0) {
        snprintf(assessment_result, result_size,
                "深度反思执行失败（错误码：%d），请检查引擎状态。", ret);
        return 0;
    }
    
    /* 执行知识冲突检测 */
    DRConflictInfo conflicts[16];
    size_t num_conflicts = 0;
    dr_detect_knowledge_conflicts(active_engine, &chain, conflicts, &num_conflicts);
    
    /* 执行根本原因分析 */
    size_t root_layer = 0;
    float root_confidence = 0.0f;
    dr_root_cause_analysis(active_engine, &chain, &root_layer, &root_confidence);
    
    /* 获取反思洞察 */
    char* insights[DR_MAX_INSIGHTS];
    size_t num_insights = 0;
    dr_get_insights(active_engine, &chain, insights, &num_insights);
    
    /* 构建综合评估结果 */
    char result_buf[4096];
    size_t pos = 0;
    
    pos += snprintf(result_buf + pos, sizeof(result_buf) - pos,
                   "【深度反思自我评估报告】\n"
                   "评估类型：%s\n"
                   "反思深度：%.2f / 7\n"
                   "转换潜力：%.2f\n"
                   "自一致性：%.2f\n\n",
                   type_names[type_idx],
                   chain.overall_depth,
                   chain.transformative_potential,
                   chain.self_consistency);
    
    /* 各层分析摘要 */
    pos += snprintf(result_buf + pos, sizeof(result_buf) - pos, "--- 各层反思分析 ---\n");
    const char* layer_names[] = {
        "描述层", "分析层", "批判层", "因果层",
        "认知层", "元认知层", "转换层"
    };
    for (size_t i = 0; i < chain.num_layers && i < 7 && pos < sizeof(result_buf) - 256; i++) {
        int layer_idx = (int)chain.layers[i].layer;
        if (layer_idx < 0) layer_idx = 0;
        if (layer_idx > 6) layer_idx = 6;
        pos += snprintf(result_buf + pos, sizeof(result_buf) - pos,
                       "  [%s] 深度:%.2f 新颖性:%.2f  coherence:%.2f\n",
                       layer_names[layer_idx],
                       chain.layers[i].depth_score,
                       chain.layers[i].novelty_score,
                       chain.layers[i].coherence_score);
    }
    
    /* 知识冲突信息 */
    if (num_conflicts > 0) {
        pos += snprintf(result_buf + pos, sizeof(result_buf) - pos,
                       "\n--- 知识冲突检测 (%zu处) ---\n", num_conflicts);
        for (size_t i = 0; i < num_conflicts && i < 3 && pos < sizeof(result_buf) - 256; i++) {
            pos += snprintf(result_buf + pos, sizeof(result_buf) - pos,
                           "  冲突#%zu: 层%zu-层%zu 冲突分:%.3f 描述:%s\n",
                           i + 1, conflicts[i].layer_a, conflicts[i].layer_b,
                           conflicts[i].conflict_score, conflicts[i].description);
        }
    }
    
    /* 根本原因 */
    pos += snprintf(result_buf + pos, sizeof(result_buf) - pos,
                   "\n--- 根本原因分析 ---\n"
                   "  根因层索引:%zu 置信度:%.3f\n",
                   root_layer, root_confidence);
    
    /* 洞察摘要 */
    if (num_insights > 0) {
        pos += snprintf(result_buf + pos, sizeof(result_buf) - pos,
                       "\n--- 关键洞察 (%zu条) ---\n", num_insights);
        for (size_t i = 0; i < num_insights && i < 5 && pos < sizeof(result_buf) - 256; i++) {
            if (insights[i]) {
                pos += snprintf(result_buf + pos, sizeof(result_buf) - pos,
                               "  %zu. %s\n", i + 1, insights[i]);
                safe_free((void**)&insights[i]);
            }
        }
    }
    
    /* 综合评分 */
    pos += snprintf(result_buf + pos, sizeof(result_buf) - pos,
                   "\n--- 综合评分 ---\n"
                   "认知健康度:%.2f\n"
                   "反思深度质量:%.2f\n"
                   "知识一致性:%.2f\n",
                   (chain.self_consistency + system->self_model_state.model_confidence) * 0.5f,
                   chain.overall_depth / 7.0f,
                   1.0f - (float)num_conflicts * 0.1f);
    
    snprintf(assessment_result, result_size, "%s", result_buf);
    
    /* 释放反思链 */
    dr_chain_free(&chain);
    
    return 0;
}

/**
 * @brief 执行反思增强监控
 */
int metacognition_reflective_monitor(MetacognitionSystem* system,
                                      const float* input_data,
                                      size_t data_size,
                                      MetacognitionMonitoringResult* result,
                                      char* reflection_detail,
                                      size_t detail_size) {
    
    if (!system || !input_data || !result) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "反思增强监控：参数为空");
        return -1;
    }
    
    if (!system->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "反思增强监控：系统未初始化");
        return -1;
    }
    
    /* 先执行标准监控 */
    int mon_ret = metacognition_monitor(system, input_data, data_size, result);
    if (mon_ret != 0) {
        return mon_ret;
    }
    
    /* 如果有深度思考链系统，执行推理增强 */
    if (system->dtc_system && reflection_detail && detail_size >= 256) {
        char query[256];
        snprintf(query, sizeof(query),
                "元认知监控分析: 当前值=%.3f 置信度=%.3f 不确定性=%.3f 趋势=%.3f",
                result->current_value, result->confidence,
                result->uncertainty, result->trend);
        
        DTCChainResult dtc_result;
        memset(&dtc_result, 0, sizeof(DTCChainResult));
        
        int dtc_ret = dtc_reason_chain(system->dtc_system, input_data, data_size, query, &dtc_result);
        
        if (dtc_ret == 0 && dtc_result.num_nodes > 0) {
            /* 使用DTC的置信度增强监控结果 */
            float dtc_confidence = 0.0f;
            for (size_t i = 0; i < dtc_result.num_nodes; i++) {
                dtc_confidence += dtc_result.nodes[i].confidence;
            }
            dtc_confidence /= (float)dtc_result.num_nodes;
            
            /* 融合监控置信度与DTC置信度 */
            result->confidence = result->confidence * 0.6f + dtc_confidence * 0.4f;
            
            /* 生成反思详细信息 */
            char detail_buf[2048];
            size_t pos = 0;
            pos += snprintf(detail_buf + pos, sizeof(detail_buf) - pos,
                           "【反思增强监控详情】\n"
                           "推理链长度:%zu\n"
                           "链连贯性:%.3f\n"
                           "推理深度:%.3f\n"
                           "解置信度:%.3f\n\n",
                           dtc_result.num_nodes,
                           dtc_result.chain_coherence,
                           dtc_result.reasoning_depth,
                           dtc_result.solution_confidence);
            
            /* 各步推理详情 */
            const char* step_names[] = {
                "观察", "分析", "假设", "推理",
                "评估", "综合", "结论", "行动"
            };
            for (size_t i = 0; i < dtc_result.num_nodes && i < 8 && pos < sizeof(detail_buf) - 128; i++) {
                int step_idx = (int)dtc_result.nodes[i].step_type;
                if (step_idx < 0) step_idx = 0;
                if (step_idx > 7) step_idx = 7;
                pos += snprintf(detail_buf + pos, sizeof(detail_buf) - pos,
                               "  步骤%zu [%s] 置信度:%.3f 不确定性:%.3f%s\n",
                               i + 1, step_names[step_idx],
                               dtc_result.nodes[i].confidence,
                               dtc_result.nodes[i].uncertainty,
                               dtc_result.nodes[i].has_branches ? " [有分支]" : "");
            }
            
            /* 替代假设提示 */
            float branching_total = 0.0f;
            for (size_t i = 0; i < dtc_result.num_nodes; i++) {
                branching_total += dtc_result.nodes[i].branching_factor;
            }
            if (branching_total > 0.1f) {
                pos += snprintf(detail_buf + pos, sizeof(detail_buf) - pos,
                               "\n发现替代推理路径（分支因子总和:%.3f），"
                               "建议执行多路径一致性验证。\n",
                               branching_total);
            }
            
            snprintf(reflection_detail, detail_size, "%s", detail_buf);
            
            dtc_chain_free(&dtc_result);
        } else {
            snprintf(reflection_detail, detail_size, "深度思考链分析不可用（错误码:%d）", dtc_ret);
        }
    } else if (reflection_detail && detail_size >= 64) {
        if (!system->dtc_system) {
            snprintf(reflection_detail, detail_size,
                    "深度思考链系统未初始化，无法执行反思增强。");
        } else {
            reflection_detail[0] = '\0';
        }
    }
    
    return 0;
}

/* ============================================================================
   内部函数实现
   ============================================================================ */

/**
 * @brief 执行贝叶斯更新
 */
static int perform_bayesian_update(MetacognitionSystem* system,
                                  const float* new_data, size_t data_size,
                                  const float* ground_truth, size_t truth_size) {
    size_t n = system->self_model_state.parameters_size < data_size ?
               system->self_model_state.parameters_size : data_size;
    if (n == 0) return 0;
    
    float prior_precision = 1.0f / (system->model_update_config.learning_rate + 1e-8f);
    float likelihood_precision = 10.0f;
    size_t update_count = system->self_model_state.update_count + 1;
    (void)update_count;
    
    for (size_t i = 0; i < n; i++) {
        float old_value = system->self_model_state.model_parameters[i];
        float old_uncertainty = system->self_model_state.uncertainty_estimates[i];
        float new_value = new_data[i % data_size];
        
        float prior_uncertainty = old_uncertainty + 0.01f;
        float posterior_precision = prior_precision / (prior_uncertainty + 1e-8f) + likelihood_precision;
        float posterior_variance = 1.0f / (posterior_precision + 1e-8f);
        float posterior_uncertainty = posterior_variance / (1.0f + posterior_variance);
        
        float measurement_weight = likelihood_precision / posterior_precision;
        float innovation = new_value - old_value;
        float posterior_mean = old_value + measurement_weight * innovation;
        float shrinkage = 1.0f / (1.0f + prior_uncertainty * 0.1f);
        system->self_model_state.model_parameters[i] = posterior_mean * shrinkage + old_value * (1.0f - shrinkage) * 0.1f;
        system->self_model_state.uncertainty_estimates[i] = posterior_uncertainty;
    }
    
    float* perf_hist = system->self_model_state.performance_history;
    if (ground_truth && truth_size > 0) {
        size_t min_size = n < truth_size ? n : truth_size;
        float total_sq_error = 0.0f;
        for (size_t i = 0; i < min_size; i++) {
            float err = new_data[i % data_size] - ground_truth[i % truth_size];
            total_sq_error += err * err;
        }
        float mse = total_sq_error / (float)min_size;
        float precision_estimate = 1.0f / (mse + 1e-8f);
        (void)precision_estimate;
        float accuracy = 1.0f / (1.0f + mse);
        float alpha_acc = prior_precision / (prior_precision + likelihood_precision);
        system->self_model_state.model_accuracy = alpha_acc * system->self_model_state.model_accuracy + (1.0f - alpha_acc) * accuracy;
        
        float avg_uncertainty = 0.0f;
        size_t uc = system->self_model_state.uncertainty_size < n ? system->self_model_state.uncertainty_size : n;
        if (uc > 0) {
            for (size_t i = 0; i < uc; i++) avg_uncertainty += system->self_model_state.uncertainty_estimates[i];
            avg_uncertainty /= (float)uc;
        }
        system->self_model_state.model_confidence = system->self_model_state.model_accuracy / (1.0f + avg_uncertainty);
        
        if (perf_hist && system->self_model_state.history_size > 0) {
            size_t idx = system->self_model_state.update_count % system->self_model_state.history_size;
            perf_hist[idx] = system->self_model_state.model_accuracy;
        }
    }
    
    system->self_model_state.update_count++;
    system->self_model_state.last_update_time = (size_t)time(NULL);
    
    return 0;
}

/* ============================================================================
 * P1-05: 元认知→自我认知闭环桥接 (P1-003修复: 使用标准头文件代替extern声明)
 *
 * 将元认知系统的深度反思/深度思维链分析结果
 * 推送回自我认知系统进行闭环校准。
 * ============================================================================ */

/**
 * @brief 将元认知的深度分析结果桥接到自我认知闭环
 *
 * 从元认知系统的深度反思引擎和深度思维链中提取洞见，
 * 并推送到 self_cognition 进行自动校准。
 * 实现 Metacognition ↔ DeepRelection ↔ SelfCognition 的完整三方闭环。
 *
 * @param system 元认知系统
 * @param self_cog 自我认知系统（用于闭环反馈）
 * @param bridge_result 桥接结果描述（可NULL）
 * @param result_size 描述缓冲区大小
 * @return 0成功，-1失败
 */
int metacognition_bridge_to_self_cognition(MetacognitionSystem* system,
                                           void* self_cog,
                                           char* bridge_result,
                                           size_t result_size) {
    if (!system || !self_cog) return -1;

    int actions = 0;
    char buf[512] = "";

    /* Step 1: 从深度反思引擎提取洞见 */
    if (system->dr_engine) {
        DRReflectionChain chain;
        memset(&chain, 0, sizeof(chain));
        float ctx_data[16];
        memset(ctx_data, 0, sizeof(ctx_data));
        ctx_data[0] = system->self_model_state.model_accuracy;
        ctx_data[1] = system->self_model_state.model_confidence;
        ctx_data[2] = system->average_monitoring_confidence;
        ctx_data[3] = system->average_prediction_error;

        if (dr_reflect(system->dr_engine,
                       "元认知闭环桥接：自我模型状态深度评估",
                       ctx_data, 16, &chain) == 0 &&
            chain.num_layers > 0) {

            /* 提取各层洞见 */
            const char* insight_texts[8];
            float insight_scores[8];
            size_t ni = 0;
            for (size_t i = 0; i < chain.num_layers && ni < 8; i++) {
                if (chain.layers[i].depth_score > 0.3f &&
                    chain.layers[i].reflection_text) {
                    insight_texts[ni] = chain.layers[i].reflection_text;
                    insight_scores[ni] = chain.layers[i].depth_score;
                    ni++;
                }
            }
            if (ni > 0) {
                /* 调用 self_cognition 的洞见集成 */
                size_t applied = 0;
                self_cognition_integrate_deep_insights((SelfCognitionSystem*)self_cog, insight_texts,
                                                       insight_scores, ni, &applied);
                if (applied > 0) {
                    actions += (int)applied;
                    snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf),
                             "深度反思: %zu条洞见已应用; ", applied);
                }
            }
            dr_chain_free(&chain);
        }
    }

    /* Step 2: 从深度思维链提取推理结论 */
    if (system->dtc_system) {
        float state_data[8];
        memset(state_data, 0, sizeof(state_data));
        state_data[0] = system->self_model_state.model_accuracy;
        state_data[1] = system->self_model_state.model_confidence;
        state_data[2] = system->average_monitoring_confidence;
        state_data[3] = system->average_prediction_error;
        state_data[4] = (float)system->total_monitoring_count / 100.0f;
        state_data[5] = (float)system->total_model_updates / 10.0f;
        state_data[6] = system->self_model_state.model_accuracy;
        state_data[7] = system->monitoring_config.adaptation_rate;

        DTCChainResult dtc_res;
        memset(&dtc_res, 0, sizeof(dtc_res));
        if (dtc_reason_chain(system->dtc_system, state_data, 8,
                             "分析元认知系统当前状态和需要的校准行动",
                             &dtc_res) == 0 && dtc_res.num_nodes > 0) {

            const char* insight_texts[8];
            float insight_scores[8];
            size_t ni = 0;
            for (size_t i = 0; i < dtc_res.num_nodes && ni < 8; i++) {
                if (dtc_res.nodes[i].confidence > 0.4f &&
                    dtc_res.nodes[i].thought_text) {
                    insight_texts[ni] = dtc_res.nodes[i].thought_text;
                    insight_scores[ni] = dtc_res.nodes[i].confidence;
                    ni++;
                }
            }
            if (ni > 0) {
                size_t applied = 0;
                self_cognition_integrate_deep_insights((SelfCognitionSystem*)self_cog, insight_texts,
                                                       insight_scores, ni, &applied);
                if (applied > 0) {
                    actions += (int)applied;
                    snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf),
                             "深度思维链: %zu条洞见已应用; ", applied);
                }
            }
            dtc_chain_free(&dtc_res);
        }
    }

    /* Step 3: 基于元认知统计直接校准 */
    if (system->average_prediction_error > 0.3f && system->total_predictions > 10) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf),
                 "预测误差偏高(%.2f)，建议执行闭环校准; ", system->average_prediction_error);
    }

    if (bridge_result && result_size > 0) {
        snprintf(bridge_result, result_size, "[元认知→自我认知桥接] 执行%d项动作: %s",
                 actions, buf[0] ? buf : "模型状态稳定，无需调整");
    }

/* 修复始终返回0的问题。原代码 return actions > 0 ? 0 : 0 
     * 两个分支均返回0，导致调用方无法区分桥接是否实际执行了动作。
     * 修复：返回实际执行的动作数量，0表示无动作但非错误。 */
    return actions;
}

#define KALMAN_INNOVATION_WINDOW 32

/**
 * @brief 执行卡尔曼滤波更新（含创新序列自适应协方差估计）
 */
static int perform_kalman_update(MetacognitionSystem* system,
                                const float* new_data, size_t data_size,
                                const float* ground_truth, size_t truth_size) {
    size_t n = system->self_model_state.parameters_size < data_size ?
               system->self_model_state.parameters_size : data_size;
    if (n == 0) return 0;
    
    /* 创新序列窗口：用于估计实际测量噪声协方差R */
    float* innovation_window = system->innovation_window;
    int window_idx = system->kalman_window_idx;
    int window_filled = system->kalman_window_filled;
    float ema_innovation_variance = system->ema_innovation_variance;
    float ema_process_noise = system->ema_process_noise;
    
    float uncertainty_weight = system->model_update_config.uncertainty_weight;
    if (uncertainty_weight < 0.1f) uncertainty_weight = 1.0f;

    /* 步骤1：计算当前批次的平均创新值（先验残差） */
    float avg_innovation = 0.0f;
    float avg_innovation_sq = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float innovation = new_data[i % data_size] - system->self_model_state.model_parameters[i];
        avg_innovation += innovation;
        avg_innovation_sq += innovation * innovation;
    }
    avg_innovation /= (float)n;
    avg_innovation_sq /= (float)n;
    float innovation_variance = avg_innovation_sq - avg_innovation * avg_innovation;
    if (innovation_variance < 1e-6f) innovation_variance = 1e-6f;
    
    /* 步骤2：更新创新序列窗口（循环缓冲区） */
    innovation_window[window_idx] = avg_innovation;
    window_idx = (window_idx + 1) % KALMAN_INNOVATION_WINDOW;
    if (!window_filled && window_idx == 0) window_filled = 1;
    
    /* 步骤3：从创新序列估计测量噪声协方差R */
    float window_mean = 0.0f;
    float window_var = 0.0f;
    int window_size = window_filled ? KALMAN_INNOVATION_WINDOW : window_idx;
    if (window_size > 0) {
        for (int wi = 0; wi < window_size; wi++) {
            window_mean += innovation_window[wi];
        }
        window_mean /= (float)window_size;
        for (int wi = 0; wi < window_size; wi++) {
            float diff = innovation_window[wi] - window_mean;
            window_var += diff * diff;
        }
        window_var /= (float)window_size;
    }
    float estimated_R = window_var > 0.01f ? window_var : 0.01f;
    if (estimated_R > 1.0f) estimated_R = 1.0f;
    
    /* 步骤4：EMA平滑噪声估计 */
    float ema_alpha = 2.0f / ((float)window_size + 3.0f);
    if (ema_alpha > 0.5f) ema_alpha = 0.5f;
    if (ema_alpha < 0.05f) ema_alpha = 0.05f;
    ema_innovation_variance = ema_alpha * innovation_variance + (1.0f - ema_alpha) * ema_innovation_variance;
    
    /* 步骤5：自适应过程噪声Q估计 */
    float base_process_noise = system->model_update_config.learning_rate * 0.01f + 1e-4f;
    float process_noise_scale = ema_innovation_variance / (estimated_R + 1e-6f);
    if (process_noise_scale > 5.0f) process_noise_scale = 5.0f;
    if (process_noise_scale < 0.1f) process_noise_scale = 0.1f;
    float process_noise = base_process_noise * process_noise_scale;
    ema_process_noise = 0.3f * process_noise + 0.7f * ema_process_noise;
    process_noise = ema_process_noise;
    if (process_noise < 1e-5f) process_noise = 1e-5f;
    if (process_noise > 0.1f) process_noise = 0.1f;
    
    float* perf_hist = system->self_model_state.performance_history;
    float* pred_errors = system->self_model_state.prediction_errors;
    
    for (size_t i = 0; i < n; i++) {
        float old_value = system->self_model_state.model_parameters[i];
        float old_uncertainty = system->self_model_state.uncertainty_estimates[i];
        float new_value = new_data[i % data_size];
        
        float predicted_value = old_value;
        float predicted_uncertainty = old_uncertainty + process_noise;
        
        /* 使用创新序列估计的R代替启发式噪声 */
        float adaptive_measurement_noise = estimated_R * (1.0f + (system->self_model_state.update_count > 0 ? 0.0f : 0.5f));
        if (adaptive_measurement_noise < 0.01f) adaptive_measurement_noise = 0.01f;
        if (adaptive_measurement_noise > 1.0f) adaptive_measurement_noise = 1.0f;
        
        float kalman_gain = predicted_uncertainty / (predicted_uncertainty + adaptive_measurement_noise + 1e-8f);
        
        float innovation = new_value - predicted_value;
        float updated_value = predicted_value + kalman_gain * innovation;
        
        float updated_uncertainty = (1.0f - kalman_gain) * predicted_uncertainty;
        
        float smooth_factor = 1.0f / (1.0f + kalman_gain * 0.5f);
        system->self_model_state.model_parameters[i] = updated_value * smooth_factor + old_value * (1.0f - smooth_factor) * 0.05f;
        system->self_model_state.uncertainty_estimates[i] = updated_uncertainty;
    }
    
    if (ground_truth && truth_size > 0) {
        size_t min_size = n < truth_size ? n : truth_size;
        float total_sq_error = 0.0f;
        for (size_t i = 0; i < min_size; i++) {
            float err = new_data[i % data_size] - ground_truth[i % truth_size];
            total_sq_error += err * err;
        }
        float mse = total_sq_error / (float)min_size;
        float accuracy = 1.0f / (1.0f + mse);
        float kalman_alpha = 1.0f / (1.0f + uncertainty_weight);
        system->self_model_state.model_accuracy = kalman_alpha * system->self_model_state.model_accuracy + (1.0f - kalman_alpha) * accuracy;
        
        float avg_uncertainty = 0.0f;
        size_t uc = system->self_model_state.uncertainty_size < n ? system->self_model_state.uncertainty_size : n;
        if (uc > 0) {
            for (size_t i = 0; i < uc; i++) avg_uncertainty += system->self_model_state.uncertainty_estimates[i];
            avg_uncertainty /= (float)uc;
        }
        system->self_model_state.model_confidence = accuracy * (1.0f - avg_uncertainty / (1.0f + avg_uncertainty));
        
        if (perf_hist && system->self_model_state.history_size > 0) {
            size_t idx = system->self_model_state.update_count % system->self_model_state.history_size;
            perf_hist[idx] = system->self_model_state.model_accuracy;
        }
        if (pred_errors && system->self_model_state.errors_size > 0) {
            size_t idx = system->self_model_state.update_count % system->self_model_state.errors_size;
            pred_errors[idx] = mse;
        }
    }
    
    system->self_model_state.update_count++;
    system->self_model_state.last_update_time = (size_t)time(NULL);
    
    system->kalman_window_idx = window_idx;
    system->kalman_window_filled = window_filled;
    system->ema_innovation_variance = ema_innovation_variance;
    system->ema_process_noise = ema_process_noise;
    
    return 0;
}

/**
 * @brief 执行集成学习更新
 */
static int perform_ensemble_update(MetacognitionSystem* system,
                                  const float* new_data, size_t data_size,
                                  const float* ground_truth, size_t truth_size) {
    size_t n = system->self_model_state.parameters_size < data_size ?
               system->self_model_state.parameters_size : data_size;
    if (n == 0) return 0;
    
    float* bayesian_params = (float*)safe_malloc(n * sizeof(float));
    float* kalman_params = (float*)safe_malloc(n * sizeof(float));
    float* online_params = (float*)safe_malloc(n * sizeof(float));
    float* bayesian_unc = (float*)safe_malloc(n * sizeof(float));
    float* kalman_unc = (float*)safe_malloc(n * sizeof(float));
    float* online_unc = (float*)safe_malloc(n * sizeof(float));
    
    if (!bayesian_params || !kalman_params || !online_params ||
        !bayesian_unc || !kalman_unc || !online_unc) {
        safe_free((void**)&bayesian_params);
        safe_free((void**)&kalman_params);
        safe_free((void**)&online_params);
        safe_free((void**)&bayesian_unc);
        safe_free((void**)&kalman_unc);
        safe_free((void**)&online_unc);
        return -1;
    }
    
    size_t update_count_before = system->self_model_state.update_count;
    for (size_t i = 0; i < n; i++) {
        bayesian_params[i] = system->self_model_state.model_parameters[i];
        bayesian_unc[i] = system->self_model_state.uncertainty_estimates[i];
        kalman_params[i] = bayesian_params[i];
        kalman_unc[i] = bayesian_unc[i];
        online_params[i] = bayesian_params[i];
        online_unc[i] = bayesian_unc[i];
    }
    
    system->self_model_state.update_count = 0;
    perform_bayesian_update(system, new_data, data_size, ground_truth, truth_size);
    for (size_t i = 0; i < n; i++) {
        bayesian_params[i] = system->self_model_state.model_parameters[i];
        bayesian_unc[i] = system->self_model_state.uncertainty_estimates[i];
        system->self_model_state.model_parameters[i] = online_params[i];
        system->self_model_state.uncertainty_estimates[i] = online_unc[i];
    }
    
    system->self_model_state.update_count = 0;
    perform_kalman_update(system, new_data, data_size, ground_truth, truth_size);
    for (size_t i = 0; i < n; i++) {
        kalman_params[i] = system->self_model_state.model_parameters[i];
        kalman_unc[i] = system->self_model_state.uncertainty_estimates[i];
        system->self_model_state.model_parameters[i] = online_params[i];
        system->self_model_state.uncertainty_estimates[i] = online_unc[i];
    }
    
    system->self_model_state.update_count = 0;
    perform_online_learning_update(system, new_data, data_size, ground_truth, truth_size);
    for (size_t i = 0; i < n; i++) {
        online_params[i] = system->self_model_state.model_parameters[i];
        online_unc[i] = system->self_model_state.uncertainty_estimates[i];
    }
    
    float bayesian_weight = 0.25f, kalman_weight = 0.35f, online_weight = 0.40f;
    
    float total_weight = bayesian_weight + kalman_weight + online_weight;
    bayesian_weight /= total_weight;
    kalman_weight /= total_weight;
    online_weight /= total_weight;
    
    float* perf_hist = system->self_model_state.performance_history;
    float* pred_errors = system->self_model_state.prediction_errors;
    
    for (size_t i = 0; i < n; i++) {
        float p = bayesian_params[i] * bayesian_weight +
                  kalman_params[i] * kalman_weight +
                  online_params[i] * online_weight;
        float u = bayesian_unc[i] * bayesian_weight +
                  kalman_unc[i] * kalman_weight +
                  online_unc[i] * online_weight;
        system->self_model_state.model_parameters[i] = p;
        system->self_model_state.uncertainty_estimates[i] = u;
    }
    
    /* M-003修复: 使用集成分学习结果更新模型状态
     * 而非丢弃已计算的bayesian/kalman/online参数 */
    float bayesian_accuracy = 0.0f;
    float kalman_accuracy = 0.0f;
    float online_accuracy = 0.0f;
    if (ground_truth && truth_size > 0 && n > 0) {
        /* 计算各方法在最近truth上的准确度 */
        for (size_t i = 0; i < n; i++) {
            float gt = (i < truth_size) ? ground_truth[i] : 0.0f;
            bayesian_accuracy += 1.0f - fminf(fabsf(bayesian_params[i] - gt), 1.0f);
            kalman_accuracy   += 1.0f - fminf(fabsf(kalman_params[i] - gt), 1.0f);
            online_accuracy   += 1.0f - fminf(fabsf(online_params[i] - gt), 1.0f);
        }
        bayesian_accuracy /= (float)n;
        kalman_accuracy   /= (float)n;
        online_accuracy   /= (float)n;
    }
    system->self_model_state.model_accuracy =
        bayesian_accuracy * bayesian_weight +
        kalman_accuracy   * kalman_weight +
        online_accuracy   * online_weight;
    system->self_model_state.model_confidence = 0.3f + 0.7f * system->self_model_state.model_accuracy;
    system->self_model_state.update_count = update_count_before + 1;
    perform_bayesian_update(system, new_data, data_size, ground_truth, truth_size);
    float bayesian_acc2 = system->self_model_state.model_accuracy;
    float bayesian_conf2 = system->self_model_state.model_confidence;
    
    system->self_model_state.model_accuracy = 0.0f;
    system->self_model_state.model_confidence = 0.0f;
    system->self_model_state.update_count = 0;
    perform_kalman_update(system, new_data, data_size, ground_truth, truth_size);
    float kalman_acc2 = system->self_model_state.model_accuracy;
    float kalman_conf2 = system->self_model_state.model_confidence;
    
    system->self_model_state.model_accuracy = 0.0f;
    system->self_model_state.model_confidence = 0.0f;
    system->self_model_state.update_count = 0;
    perform_online_learning_update(system, new_data, data_size, ground_truth, truth_size);
    float online_acc2 = system->self_model_state.model_accuracy;
    float online_conf2 = system->self_model_state.model_confidence;
    
    float combined_accuracy = bayesian_acc2 * bayesian_weight +
                              kalman_acc2 * kalman_weight +
                              online_acc2 * online_weight;
    float combined_confidence = bayesian_conf2 * bayesian_weight +
                                kalman_conf2 * kalman_weight +
                                online_conf2 * online_weight;
    
    system->self_model_state.model_accuracy = combined_accuracy;
    system->self_model_state.model_confidence = combined_confidence;
    system->self_model_state.update_count = update_count_before + 1;
    system->self_model_state.last_update_time = (size_t)time(NULL);
    
    if (perf_hist && system->self_model_state.history_size > 0) {
        size_t idx = (system->self_model_state.update_count - 1) % system->self_model_state.history_size;
        perf_hist[idx] = combined_accuracy;
    }
    if (pred_errors && system->self_model_state.errors_size > 0) {
        float pred_error = 1.0f - combined_accuracy;
        if (pred_error < 0.0f) pred_error = 0.0f;
        size_t idx = (system->self_model_state.update_count - 1) % system->self_model_state.errors_size;
        pred_errors[idx] = pred_error;
    }
    
    safe_free((void**)&bayesian_params);
    safe_free((void**)&kalman_params);
    safe_free((void**)&online_params);
    safe_free((void**)&bayesian_unc);
    safe_free((void**)&kalman_unc);
    safe_free((void**)&online_unc);
    
    return 0;
}

/**
 * @brief 计算置信度
 * 
 * @param data 输入数据数组
 * @param data_size 数据大小
 * @return float 置信度值 (0-1)
 */
static float calculate_confidence(const float* data, size_t data_size) {
    if (!data || data_size == 0) {
        return 0.5f; /* 默认中等置信度 */
    }
    
    /* 计算均值和标准差 */
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (size_t i = 0; i < data_size; i++) {
        sum += data[i];
        sum_sq += data[i] * data[i];
    }
    
    float mean = sum / data_size;
    float variance = (sum_sq / data_size) - (mean * mean);
    
    /* 防止负方差（数值误差） */
    if (variance < 0.0f) {
        variance = 0.0f;
    }
    
    float std_dev = sqrtf(variance);
    
    /* N-004修复: 自适应数据范围归一化
     * 替代假设[0,1]的硬编码归一化。
     * 使用实际数据的min-max动态范围计算标准化 */
    float min_val = mean - 3.0f * std_dev;
    float max_val = mean + 3.0f * std_dev;
    float data_range = max_val - min_val;
    if (data_range < 1e-8f) data_range = 1.0f;
    float confidence = 1.0f - (std_dev / data_range);
    
    /* 限制在合理范围 */
    if (confidence < 0.0f) confidence = 0.0f;
    if (confidence > 1.0f) confidence = 1.0f;
    
    return confidence;
}

/**
 * @brief 基于贝叶斯后验方差的不确定性计算
 *
 * 维护预测误差方差的指数移动平均(EMA)作为贝叶斯后验方差的估计，
 * 使用 uncertainty = sqrt(ema_variance) / (1.0 + log(1 + N)) 公式。
 *
 * 其中：
 *   - ema_variance: 历史预测误差的EMA平滑方差
 *   - N: 历史样本数
 *   - 分母中的 log(1+N) 反映随着经验增长，不确定性自然降低
 *
 * @param data 输入数据数组
 * @param data_size 数据大小
 * @return float 不确定性值 (0-1)
 */
static float calculate_uncertainty(MetacognitionSystem* system, const float* data, size_t data_size) {
    if (!data || data_size == 0 || !system) {
        return 0.5f; /* 无数据时默认中等不确定性 */
    }


    /* 计算当前批次的均值和方差 */
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (size_t i = 0; i < data_size; i++) {
        sum += data[i];
        sum_sq += data[i] * data[i];
    }
    float batch_mean = sum / (float)data_size;
    float batch_variance = (sum_sq / (float)data_size) - (batch_mean * batch_mean);
    if (batch_variance < 1e-8f) batch_variance = 1e-8f;

    /* 更新EMA均值和方差（指数移动平均，平滑因子α=0.1） */
    float alpha = 0.1f;
    if (system->ema_uncertainty_samples == 0) {
        system->ema_uncertainty_mean = batch_mean;
        system->ema_uncertainty_variance = batch_variance;
    } else {
        system->ema_uncertainty_mean = alpha * batch_mean + (1.0f - alpha) * system->ema_uncertainty_mean;
        /* 批量方差与EMA均值的结合 */
        float centered_var = batch_variance + (batch_mean - system->ema_uncertainty_mean) * (batch_mean - system->ema_uncertainty_mean);
        system->ema_uncertainty_variance = alpha * centered_var + (1.0f - alpha) * system->ema_uncertainty_variance;
    }

    /* 更新累计样本数 */
    system->ema_uncertainty_samples += data_size;

    /* 贝叶斯后验方差不确定性 */
    float posterior_std = sqrtf(system->ema_uncertainty_variance);
    float log_term = logf(1.0f + (float)system->ema_uncertainty_samples);
    float uncertainty = posterior_std / (1.0f + log_term);

    /* 上限裁剪: 不确定性最大不超过1.0 */
    if (uncertainty > 1.0f) uncertainty = 1.0f;
    if (uncertainty < 0.001f) uncertainty = 0.001f;

    return uncertainty;
}

/**
 * @brief 计算趋势
 * 
 * @param history 历史数据数组
 * @param history_size 历史数据大小
 * @return float 趋势值 (-1到1，负值表示下降，正值表示上升)
 */
static float calculate_trend(const float* history, size_t history_size) {
    if (!history || history_size < 2) {
        return 0.0f; /* 无足够数据计算趋势 */
    }
    
    /* 简单线性回归计算斜率 */
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_xy = 0.0f;
    float sum_xx = 0.0f;
    
    for (size_t i = 0; i < history_size; i++) {
        float x = (float)i;
        float y = history[i];
        
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }
    
    float n = (float)history_size;
    float denominator = n * sum_xx - sum_x * sum_x;
    
    if (fabsf(denominator) < 1e-10f) {
        return 0.0f; /* 避免除零 */
    }
    
    float slope = (n * sum_xy - sum_x * sum_y) / denominator;
    
    /* N-004修复: 自适应趋势归一化
     * 使用实际数据标准差归一化趋势斜率，替代假设最大斜率1.0 */
    float mean_y = sum_y / n, var_y = 0.0f;
    for (int i = 0; i < history_size; i++) var_y += (history[i] - mean_y) * (history[i] - mean_y);
    float std_y = sqrtf(var_y / n);
    float scale = (std_y > 1e-8f) ? std_y : 1.0f;
    float trend = slope * (float)history_size / scale; /* 归一化: slope * N_pts / sigma */
    
    /* 限制趋势范围 */
    if (trend < -1.0f) trend = -1.0f;
    if (trend > 1.0f) trend = 1.0f;
    
    return trend;
}

/**
 * @brief 执行在线学习更新
 */
static int perform_online_learning_update(MetacognitionSystem* system,
                                         const float* new_data, size_t data_size,
                                         const float* ground_truth, size_t truth_size) {
    /* 在线学习：使用随机梯度下降 */
    float learning_rate = system->model_update_config.learning_rate;
    
    for (size_t i = 0; i < system->self_model_state.parameters_size && i < data_size; i++) {
        float gradient = new_data[i % data_size] - system->self_model_state.model_parameters[i];
        system->self_model_state.model_parameters[i] += learning_rate * gradient;
    }
    
    /* 如果有真实值，更新模型准确度 */
    if (ground_truth && truth_size > 0) {
        float total_error = 0.0f;
        size_t min_size = data_size < truth_size ? data_size : truth_size;
        for (size_t i = 0; i < min_size; i++) {
            total_error += fabsf(new_data[i] - ground_truth[i % truth_size]);
        }
        float avg_error = total_error / min_size;
        
        /* 指数移动平均更新准确度 */
        system->self_model_state.model_accuracy = 
            (1.0f - learning_rate) * system->self_model_state.model_accuracy + 
            learning_rate * (1.0f - avg_error);
    }
    
    /* 更新模型置信度（基于准确度和不确定性） */
    /* FIX-012修复: 使用加权EMA而非前10个样本简单平均，提高对分布突变的敏感度 */
    float uncertainty_weight = system->model_update_config.uncertainty_weight;
    float avg_uncertainty = 0.0f;
    float weighted_sum = 0.0f;
    float decay_sum = 0.0f;
    float decay = 0.85f;  /* EMA衰减因子，越近的样本权重越高 */
    float weight = 1.0f;
    size_t uncertainty_samples = system->self_model_state.uncertainty_size < 20 ? 
                                 system->self_model_state.uncertainty_size : 20;
    
    for (size_t i = 0; i < uncertainty_samples; i++) {
        size_t idx = system->self_model_state.uncertainty_size - 1 - i;
        if (idx < system->self_model_state.uncertainty_size) {
            weighted_sum += system->self_model_state.uncertainty_estimates[idx] * weight;
            decay_sum += weight;
            weight *= decay;
        }
    }
    
    if (decay_sum > 0.001f) {
        avg_uncertainty = weighted_sum / decay_sum;
    }
    
    system->self_model_state.model_confidence = 
        system->self_model_state.model_accuracy * (1.0f - avg_uncertainty * uncertainty_weight);
    
    /* 更新计数和时间 */
    system->self_model_state.update_count++;
    system->self_model_state.last_update_time = (size_t)time(NULL);
    
    return 0;
}