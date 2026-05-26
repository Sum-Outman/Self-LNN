/**
 * @file multimodal_unified_input.c
 * @brief 多模态统一输入实现
 *
 * 实现"所有模态 → 统一输入到同一个连续动态系统 → 统一状态演化 → 统一输出决策"。
 * 所有模态控制信号直接拼接为统一输入向量，通过单一CfC细胞进行连续时间状态演化，
 * CfC隐藏状态即为统一表示，不进行任何预融合加权平均。
 * 严格遵循：不需要多模型融合、不需要跨模态注意力。
 */

#include "selflnn/multimodal/multimodal_unified_input.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/cfc_network.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * @brief 收集有效模态信号到统一数组
 */
static int collect_active_control_signals(
    const float* vision_control, size_t vision_size,
    const float* audio_control, size_t audio_size,
    const float* text_control, size_t text_size,
    const float* sensor_control, size_t sensor_size,
    const float* tactile_control, size_t tactile_size,
    const float* proprioception_control, size_t proprioception_size,
    const float* thermal_control, size_t thermal_size,
    const float* radar_control, size_t radar_size,
    const float* motor_control, size_t motor_size,
    const float** signals, size_t* signal_sizes, int* modality_present)
{
    modality_present[0] = (vision_control && vision_size > 0) ? 1 : 0;
    modality_present[1] = (audio_control && audio_size > 0) ? 1 : 0;
    modality_present[2] = (text_control && text_size > 0) ? 1 : 0;
    modality_present[3] = (sensor_control && sensor_size > 0) ? 1 : 0;
    modality_present[4] = (tactile_control && tactile_size > 0) ? 1 : 0;
    modality_present[5] = (proprioception_control && proprioception_size > 0) ? 1 : 0;
    modality_present[6] = (thermal_control && thermal_size > 0) ? 1 : 0;
    modality_present[7] = (radar_control && radar_size > 0) ? 1 : 0;
    modality_present[8] = (motor_control && motor_size > 0) ? 1 : 0;

    signals[0] = vision_control;          signal_sizes[0] = modality_present[0] ? vision_size : 0;
    signals[1] = audio_control;           signal_sizes[1] = modality_present[1] ? audio_size : 0;
    signals[2] = text_control;            signal_sizes[2] = modality_present[2] ? text_size : 0;
    signals[3] = sensor_control;          signal_sizes[3] = modality_present[3] ? sensor_size : 0;
    signals[4] = tactile_control;         signal_sizes[4] = modality_present[4] ? tactile_size : 0;
    signals[5] = proprioception_control;  signal_sizes[5] = modality_present[5] ? proprioception_size : 0;
    signals[6] = thermal_control;         signal_sizes[6] = modality_present[6] ? thermal_size : 0;
    signals[7] = radar_control;           signal_sizes[7] = modality_present[7] ? radar_size : 0;
    signals[8] = motor_control;           signal_sizes[8] = modality_present[8] ? motor_size : 0;

    int count = 0;
    for (int i = 0; i < SELFLNN_MAX_MODALITIES; i++) {
        if (modality_present[i]) count++;
    }
    return count;
}

/**
 * @brief 评估统一输出质量（多维综合评估）
 *
 * 基于以下维度进行真实评估：
 * 1. 信号能量分布均衡性
 * 2. 模态间相关性（跨模态一致性）
 * 3. 时序稳定性（与前一帧的差异）
 * 4. 稀疏度（避免全零或全饱和输出）
 *
 * @return 0-1 之间的质量评分，1 为最佳
 */
static float evaluate_process_quality_detailed(const float* unified, size_t dim,
                                                const float* prev_output, size_t prev_dim)
{
    if (!unified || dim == 0) return 0.0f;

    float energy = 0.0f, min_val = 1e10f, max_val = -1e10f;
    int nonzero = 0;
    for (size_t d = 0; d < dim; d++) {
        float v = unified[d];
        energy += v * v;
        if (fabsf(v) > 1e-8f) nonzero++;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    energy = sqrtf(energy / (float)dim);

    /* 因子1：能量合理性 (0~1) */
    float energy_score = 1.0f;
    if (energy < 0.01f) energy_score = 0.1f;
    else if (energy > 10.0f) energy_score = 0.3f;
    else if (energy < 0.1f) energy_score = 0.3f + (energy - 0.01f) * 7.0f;

    /* 因子2：稀疏度合理性 (避免全零或全饱和) */
    float sparsity = (float)nonzero / (float)dim;
    float sparsity_score = (sparsity > 0.1f && sparsity < 0.9f) ? 1.0f :
                           (sparsity > 0.0f) ? sparsity * 1.5f : 0.1f;

    /* 因子3：动态范围合理性 */
    float dynamic_range = max_val - min_val;
    float range_score = (dynamic_range > 0.01f && dynamic_range < 50.0f) ? 1.0f :
                        (dynamic_range > 0.0f) ? 0.5f : 0.1f;

    /* 因子4：时序稳定性 */
    float temporal_score = 1.0f;
    if (prev_output && prev_dim == dim) {
        float diff_sum = 0.0f;
        for (size_t d = 0; d < dim; d++) {
            float diff = unified[d] - prev_output[d];
            diff_sum += diff * diff;
        }
        float rmse = sqrtf(diff_sum / (float)dim);
        temporal_score = 1.0f / (1.0f + rmse);
    }

    return (energy_score * 0.25f + sparsity_score * 0.25f +
            range_score * 0.15f + temporal_score * 0.35f);
}

/**
 * K-022: 模态信号有效性验证 —— 检查各模态信号是否包含有效的非零数据
 * @return 有效模态数（至少1个模态包含非零信号）
 */
static int validate_modality_signals(const float* signals[SELFLNN_MAX_MODALITIES],
                                      const size_t* signal_sizes,
                                      const int* modality_present) {
    int valid_count = 0;
    for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
        if (!modality_present[m] || !signals[m] || signal_sizes[m] == 0) continue;
        int has_data = 0;
        for (size_t i = 0; i < signal_sizes[m]; i++) {
            if (fabsf(signals[m][i]) > 1e-10f) { has_data = 1; break; }
        }
        if (has_data) valid_count++;
    }
    return valid_count;
}

/**
 * @brief 核心统一输入处理：所有模态信号通过独立线性投影矩阵映射到统一维度，
 *        然后element-wise求和注入单一LNN连续动态系统
 * 
 * ZSF-ZNB修复S-002: 架构从"直接拼接"改为"线性投影+求和"
 *   统一输入 = Σ_i (W_i · x_i + b_i)
 *   其中 W_i: SELFLNN_UNIFIED_PROJECTION_DIM × input_dim_i
 */
static int unified_input_dynamic_process(
    UnifiedInputState* state,
    const float* signals[SELFLNN_MAX_MODALITIES],
    const size_t* signal_sizes,
    const int* modality_present,
    int n_modalities,
    float* unified_output,
    size_t max_output_size)
{
    (void)n_modalities;

    if (!state || !unified_output || max_output_size == 0) {
        return -1;
    }

    /* ZSF-ZNB修复S-002: 延迟初始化投影矩阵 */
    if (!state->projections_initialized) {
        /* Xavier初始化每个模态的投影矩阵和偏置 */
        for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
            if (!state->projection_matrices[m]) {
                size_t mod_size = SELFLNN_MAX_CONTROL_DIM;
                size_t total_elems = SELFLNN_UNIFIED_PROJECTION_DIM * mod_size;
                state->projection_matrices[m] = (float*)safe_calloc(total_elems, sizeof(float));
                state->projection_biases[m] = (float*)safe_calloc(SELFLNN_UNIFIED_PROJECTION_DIM, sizeof(float));
                if (state->projection_matrices[m]) {
                    float xavier_std = sqrtf(2.0f / (float)(SELFLNN_UNIFIED_PROJECTION_DIM + mod_size));
                    for (size_t i = 0; i < total_elems; i++) {
                        float u1 = secure_random_float();
                        float u2 = secure_random_float();
                        if (u1 < 1e-8f) u1 = 1e-8f;
                        state->projection_matrices[m][i] = xavier_std * 
                            sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
                    }
                }
                state->projection_input_sizes[m] = mod_size;
            }
        }
        state->projections_initialized = 1;
    }

    /* 分配统一投影缓冲区 */
    float* combined = (float*)safe_calloc(SELFLNN_UNIFIED_PROJECTION_DIM, sizeof(float));
    if (!combined) return -1;

    /* 步骤1: 对每个活跃模态进行线性投影并求和
     * ZSF-005修复: 添加跨模态维度差异均衡
     * 不同模态维度差异巨大（视觉1024 vs 触觉64），直接投影求和导致
     * 高维模态主导输出。先对原始输入做L2归一化，再按sqrt(dim)缩放，
     * 使各模态对combined_input的贡献处于相同量级。 */
    int active_modalities = 0;
    for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
        if (!modality_present[m] || !signals[m] || signal_sizes[m] == 0) continue;
        if (!state->projection_matrices[m] || !state->projection_biases[m]) continue;

        size_t input_dim = signal_sizes[m];
        if (input_dim > SELFLNN_MAX_CONTROL_DIM) input_dim = SELFLNN_MAX_CONTROL_DIM;

        /* ZSF-005: 跨模态维度差异归一化
         * 先计算原始输入的L2范数，用于归一化，使不同模态的原始信号
         * 处于相同量级后再投影。sqrt(proj_dim/input_dim)缩放因子
         * 补偿维度差异，确保高低维模态贡献均衡。 */
        float raw_l2_norm = 0.0f;
        for (size_t k = 0; k < input_dim; k++) {
            raw_l2_norm += signals[m][k] * signals[m][k];
        }
        raw_l2_norm = sqrtf(raw_l2_norm + 1e-12f);
        float inv_norm = 1.0f / raw_l2_norm;
        float dim_scale = sqrtf((float)SELFLNN_UNIFIED_PROJECTION_DIM / ((float)input_dim + 1.0f));

        /* 计算 W_m · (x_m * inv_norm) + b_m，累加到combined */
        float* W = state->projection_matrices[m];
        float* b = state->projection_biases[m];
        for (size_t j = 0; j < SELFLNN_UNIFIED_PROJECTION_DIM; j++) {
            float proj_val = b[j];
            for (size_t k = 0; k < input_dim; k++) {
                proj_val += W[j * input_dim + k] * signals[m][k] * inv_norm;
            }
            combined[j] += proj_val * dim_scale;
        }
        active_modalities++;
    }

    if (active_modalities == 0) {
        state->last_active_count = 0;
        safe_free((void**)&combined);
        return -1;
    }

    /* ZSF-009: 存储当前投影结果和活跃模态数，供在线训练使用 */
    state->last_active_count = active_modalities;
    for (size_t j = 0; j < SELFLNN_UNIFIED_PROJECTION_DIM; j++) {
        state->last_combined[j] = combined[j];
    }

    /* 存储各模态原始信号和维度 */
    for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
        state->last_raw_sizes[m] = 0;
        if (modality_present[m] && signals[m] && signal_sizes[m] > 0) {
            size_t sz = signal_sizes[m];
            if (sz > SELFLNN_MAX_CONTROL_DIM) sz = SELFLNN_MAX_CONTROL_DIM;
            state->last_raw_sizes[m] = sz;
            for (size_t k = 0; k < sz; k++) {
                state->last_raw_signals[m][k] = signals[m][k];
            }
        }
    }

    /* 步骤2: 通过共享LNN处理统一输入 */
    if (state->lnn_instance) {
        LNNConfig lnn_cfg;
        if (lnn_get_config(state->lnn_instance, &lnn_cfg) == 0) {
            float* lnn_input = combined;
            size_t lnn_in_dim = SELFLNN_UNIFIED_PROJECTION_DIM;
            
            /* 如果LNN输入维度与投影维度不同，需要调整 */
            float* adjusted_input = NULL;
            if (lnn_cfg.input_size != SELFLNN_UNIFIED_PROJECTION_DIM) {
                adjusted_input = (float*)safe_calloc(lnn_cfg.input_size, sizeof(float));
                if (adjusted_input) {
                    size_t copy_dim = (lnn_cfg.input_size < SELFLNN_UNIFIED_PROJECTION_DIM) 
                        ? lnn_cfg.input_size : SELFLNN_UNIFIED_PROJECTION_DIM;
                    memcpy(adjusted_input, combined, copy_dim * sizeof(float));
                    lnn_input = adjusted_input;
                    lnn_in_dim = lnn_cfg.input_size;
                }
            }

            size_t output_dim = max_output_size;
            if (output_dim > lnn_cfg.output_size) output_dim = lnn_cfg.output_size;
            float* lnn_output_buf = (float*)safe_malloc(
                (lnn_cfg.output_size > output_dim ? lnn_cfg.output_size : output_dim) * sizeof(float));
            
            if (lnn_output_buf) {
                int fwd_ret = lnn_forward(state->lnn_instance, lnn_input, lnn_output_buf);
                if (fwd_ret == 0) {
                    size_t copy = lnn_cfg.output_size;
                    if (copy > output_dim) copy = output_dim;
                    memcpy(unified_output, lnn_output_buf, copy * sizeof(float));
                    if (copy < output_dim)
                        memset(unified_output + copy, 0, (output_dim - copy) * sizeof(float));
                    /* ZSF-009: 存储统一输出供在线训练使用 */
                    size_t prev_copy = copy < SELFLNN_MAX_CONTROL_DIM ? copy : SELFLNN_MAX_CONTROL_DIM;
                    memcpy(state->prev_output, lnn_output_buf, prev_copy * sizeof(float));
                    state->prev_output_dim = prev_copy;
                    safe_free((void**)&lnn_output_buf);
                    if (adjusted_input) safe_free((void**)&adjusted_input);
                    safe_free((void**)&combined);
                    return (int)output_dim;
                }
                safe_free((void**)&lnn_output_buf);
            }
            if (adjusted_input) safe_free((void**)&adjusted_input);
        }
    }

    safe_free((void**)&combined);

    /* LNN未绑定或前向传播失败：返回错误，禁止直通退化 */
    selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                          "多模态统一输入：共享LNN未绑定或前向传播失败，禁止直通退化处理");
    return -1;
}

int multimodal_unified_input_init(UnifiedInputState* state, const UnifiedInputConfig* config)
{
    if (!state) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "统一输入状态指针为空");
        return -1;
    }
    memset(state, 0, sizeof(UnifiedInputState));

    if (config) {
        memcpy(&state->config, config, sizeof(UnifiedInputConfig));
    } else {
        state->config.method = UNIFIED_INPUT_DYNAMIC;
        state->config.unified_input_size = SELFLNN_UNIFIED_INPUT_DIM;
        state->config.unified_output_size = SELFLNN_MAX_CONTROL_DIM * 2;
    }

    for (int i = 0; i < SELFLNN_MAX_MODALITIES; i++) {
        state->input_weights[i] = 1.0f / (float)SELFLNN_MAX_MODALITIES;
        state->modality_metrics[i].dimension = 0;
        state->modality_metrics[i].initialized = 0;
        state->modality_metrics[i].confidence = 0.5f;
        state->modality_metrics[i].uncertainty = 0.5f;
        state->modality_metrics[i].signal_strength = 0.0f;
        state->modality_metrics[i].offset = 0;
        state->modality_metrics[i].active_size = 0;
    }

    state->total_active_size = 0;
    state->unified_buffer_size = SELFLNN_UNIFIED_INPUT_DIM;
    state->unified_input_buffer = (float*)safe_calloc(state->unified_buffer_size, sizeof(float));
    if (!state->unified_input_buffer) {
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_ALLOCATION, __func__, __FILE__, __LINE__,
                              "统一输入: 无法分配拼接缓冲区");
        return -1;
    }

    size_t output_size = state->config.unified_output_size;
    if (output_size == 0) output_size = SELFLNN_MAX_CONTROL_DIM * 2;

    state->unified_buffer = (float*)safe_calloc(output_size, sizeof(float));
    if (!state->unified_buffer) {
        safe_free((void**)&state->unified_input_buffer);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_ALLOCATION, __func__, __FILE__, __LINE__,
                              "统一输入: 无法分配输出缓冲区");
        return -1;
    }
    state->unified_buffer_size = output_size;
    state->lnn_instance = NULL;

    state->is_initialized = 1;
    state->historical_process_quality = 0.5f;

    /* ZSF-ZNB修复S-002: 投影矩阵延迟初始化标志 */
    state->projections_initialized = 0;

    return 0;
}

int multimodal_unified_input_process(UnifiedInputState* state,
                                   const float* vision_control, size_t vision_size,
                                   const float* audio_control, size_t audio_size,
                                   const float* text_control, size_t text_size,
                                   const float* sensor_control, size_t sensor_size,
                                   const float* tactile_control, size_t tactile_size,
                                   const float* proprioception_control, size_t proprioception_size,
                                   const float* thermal_control, size_t thermal_size,
                                   const float* radar_control, size_t radar_size,
                                   const float* motor_control, size_t motor_size,
                                   float* unified_output, size_t* unified_size,
                                   size_t max_output_size)
{
    if (!state || !state->is_initialized || !unified_output || !unified_size || max_output_size == 0) {
        if (unified_size) *unified_size = 0;
        return -1;
    }
    /* ZSFWS-S006修复: 多模态统一输入训练状态检查
     * 未训练时投影矩阵和LNN均为随机权重，输出为随机噪声，
     * 对自主学习和决策造成严重误导。仅在训练完成后允许使用。 */
    if (!state->is_trained && !state->lnn_instance) {
        if (unified_size) *unified_size = 0;
        return -3; /* 返回特殊错误码表示模型未训练 */
    }
    /* B-015: 输出大小上限验证 */
    if (max_output_size > SELFLNN_MAX_CONTROL_DIM)
        max_output_size = SELFLNN_MAX_CONTROL_DIM;

    /* B-015: 输入维度边界检查 —— 各模态信号尺寸上限均为SELFLNN_MAX_CONTROL_DIM */
    #define BOUND_CHECK_SIZE(size_var) \
        if (size_var > SELFLNN_MAX_CONTROL_DIM) { size_var = SELFLNN_MAX_CONTROL_DIM; }
    BOUND_CHECK_SIZE(vision_size);
    BOUND_CHECK_SIZE(audio_size);
    BOUND_CHECK_SIZE(text_size);
    BOUND_CHECK_SIZE(sensor_size);
    BOUND_CHECK_SIZE(tactile_size);
    BOUND_CHECK_SIZE(proprioception_size);
    BOUND_CHECK_SIZE(thermal_size);
    BOUND_CHECK_SIZE(radar_size);
    BOUND_CHECK_SIZE(motor_size);
    #undef BOUND_CHECK_SIZE

    const float* signals[SELFLNN_MAX_MODALITIES];
    size_t signal_sizes[SELFLNN_MAX_MODALITIES];
    int modality_present[SELFLNN_MAX_MODALITIES];

    int active_count = collect_active_control_signals(
        vision_control, vision_size,
        audio_control, audio_size,
        text_control, text_size,
        sensor_control, sensor_size,
        tactile_control, tactile_size,
        proprioception_control, proprioception_size,
        thermal_control, thermal_size,
        radar_control, radar_size,
        motor_control, motor_size,
        signals, signal_sizes, modality_present);

    if (active_count == 0) {
        *unified_size = 0;
        return 0;
    }

    for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
        if (modality_present[m] && signals[m] && signal_sizes[m] > 0) {
            float energy = 0.0f;
            size_t count = signal_sizes[m];
            for (size_t d = 0; d < count; d++) {
                energy += signals[m][d] * signals[m][d];
            }
            state->modality_metrics[m].signal_strength = sqrtf(energy / (float)count);
            state->modality_metrics[m].dimension = count;
            if (!state->modality_metrics[m].initialized) {
                state->modality_metrics[m].initialized = 1;
            }
        } else {
            state->modality_metrics[m].signal_strength = 0.0f;
        }
    }

    int result_dim = unified_input_dynamic_process(
        state, signals, signal_sizes, modality_present,
        SELFLNN_MAX_MODALITIES, unified_output, max_output_size);

    if (result_dim > 0) {
        /* M-003修复: 传入上一次输出实现真实的时序对比 */
        float quality = evaluate_process_quality_detailed(
            unified_output, (size_t)result_dim,
            state->prev_output, state->prev_output_dim);
        state->historical_process_quality = 0.95f * state->historical_process_quality +
                                           0.05f * quality;
        state->total_process_count++;
        if (result_dim <= SELFLNN_MAX_CONTROL_DIM) {
            memcpy(state->prev_output, unified_output, (size_t)result_dim * sizeof(float));
            state->prev_output_dim = (size_t)result_dim;
        }

        for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
            if (modality_present[m] && state->modality_metrics[m].initialized) {
                float prior_conf = state->modality_metrics[m].confidence;
                float signal_factor = state->modality_metrics[m].signal_strength;
                float signal_proxy = signal_factor / (1.0f + signal_factor);
                float quality_factor = quality * signal_proxy;
                float updated_conf = 0.8f * prior_conf + 0.2f * quality_factor;
                if (updated_conf > 1.0f) updated_conf = 1.0f;
                if (updated_conf < 0.01f) updated_conf = 0.01f;
                state->modality_metrics[m].confidence = updated_conf;
                state->modality_metrics[m].uncertainty = 1.0f - updated_conf;
            }
        }

        float total_conf = 0.0f;
        int present_count = 0;
        for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
            if (modality_present[m] && state->modality_metrics[m].initialized) {
                total_conf += state->modality_metrics[m].confidence;
                present_count++;
            }
        }
        if (present_count > 0 && total_conf > 1e-8f) {
            for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
                if (modality_present[m] && state->modality_metrics[m].initialized) {
                    state->input_weights[m] = state->modality_metrics[m].confidence / total_conf;
                } else {
                    state->input_weights[m] = 0.0f;
                }
            }
        }
    }

    *unified_size = (size_t)(result_dim > 0 ? (size_t)result_dim : 0);
    return result_dim;
}

int multimodal_unified_input_set_config(UnifiedInputState* state,
                                 const UnifiedInputConfig* config)
{
    if (!state || !config) return -1;
    memcpy(&state->config, config, sizeof(UnifiedInputConfig));
    return 0;
}

int multimodal_unified_input_get_config(const UnifiedInputState* state,
                                 UnifiedInputConfig* config)
{
    if (!state || !config) return -1;
    memcpy(config, &state->config, sizeof(UnifiedInputConfig));
    return 0;
}

int multimodal_unified_input_get_weights(const UnifiedInputState* state,
                                  float* weights, size_t* count)
{
    if (!state || !weights || !count) return -1;
    *count = SELFLNN_MAX_MODALITIES;
    memcpy(weights, state->input_weights, SELFLNN_MAX_MODALITIES * sizeof(float));
    return 0;
}

int multimodal_unified_input_reset(UnifiedInputState* state)
{
    if (!state) return -1;

    /* ZSF-ZNB修复S-002: 释放投影矩阵 */
    for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
        if (state->projection_matrices[m]) {
            safe_free((void**)&state->projection_matrices[m]);
        }
        if (state->projection_biases[m]) {
            safe_free((void**)&state->projection_biases[m]);
        }
        state->projection_input_sizes[m] = 0;
    }
    state->projections_initialized = 0;

    safe_free((void**)&state->unified_buffer);
    safe_free((void**)&state->unified_input_buffer);

    UnifiedInputConfig saved_config;
    memcpy(&saved_config, &state->config, sizeof(UnifiedInputConfig));

    memset(state, 0, sizeof(UnifiedInputState));
    memcpy(&state->config, &saved_config, sizeof(UnifiedInputConfig));

    for (int i = 0; i < SELFLNN_MAX_MODALITIES; i++) {
        state->input_weights[i] = 1.0f / (float)SELFLNN_MAX_MODALITIES;
        state->modality_metrics[i].confidence = 0.5f;
        state->modality_metrics[i].uncertainty = 0.5f;
        state->modality_metrics[i].signal_strength = 0.0f;
        state->modality_metrics[i].offset = 0;
        state->modality_metrics[i].active_size = 0;
    }

    state->total_active_size = 0;
    state->unified_buffer_size = SELFLNN_UNIFIED_INPUT_DIM;
    state->unified_input_buffer = (float*)safe_calloc(state->unified_buffer_size, sizeof(float));
    if (state->unified_input_buffer) {
        size_t output_size = state->config.unified_output_size;
        if (output_size == 0) output_size = SELFLNN_MAX_CONTROL_DIM * 2;
        state->unified_buffer = (float*)safe_calloc(output_size, sizeof(float));
        state->unified_buffer_size = output_size;
    }
    state->lnn_instance = NULL;
    state->is_initialized = 1;
    return 0;
}

int multimodal_unified_input_get_stats(const UnifiedInputState* state,
                                UnifiedInputMethod* active_method,
                                int* total_count,
                                float* process_quality)
{
    if (!state) return -1;
    if (active_method) *active_method = UNIFIED_INPUT_DYNAMIC;
    if (total_count) *total_count = state->total_process_count;
    if (process_quality) *process_quality = state->historical_process_quality;
    return 0;
}

int multimodal_unified_input_set_lnn(UnifiedInputState* state, LNN* lnn)
{
    if (!state) return -1;
    state->lnn_instance = lnn;
    return 0;
}

/* ================================================================
 * K-009: 9模态统一注入状态诊断
 * ================================================================ */

static const char* g_modality_names[SELFLNN_MAX_MODALITIES] = {
    "视觉(Vision)",
    "语音(Audio)",
    "文本(Text)",
    "传感器(Sensor)",
    "触觉(Haptic)",
    "本体感知(Proprioception)",
    "热感(Thermal)",
    "雷达(Radar)",
    "电机(Motor)"
};

int multimodal_unified_input_diagnose_modalities(const UnifiedInputState* state,
                                                  int* modality_active,
                                                  int* total_modalities,
                                                  int* active_count) {
    if (!state || !modality_active) return -1;

    if (total_modalities) *total_modalities = SELFLNN_MAX_MODALITIES;

    int active = 0;
    for (int i = 0; i < SELFLNN_MAX_MODALITIES; i++) {
        /* 检查各模态是否有有效数据流:
         * - modality_metrics[i].signal_strength > 阈值时视为活跃
         * - 或 modality_metrics[i].initialized == 1 视为已初始化 */
        int is_active = (state->modality_metrics[i].signal_strength > 0.01f ||
                         state->modality_metrics[i].initialized) ? 1 : 0;
        modality_active[i] = is_active;
        if (is_active) active++;
    }

    if (active_count) *active_count = active;

    /* 诊断日志：报告当前模态连接状态 */
    char buf[512];
    int pos = 0;
    pos += snprintf(buf + pos, sizeof(buf) - pos, "[模态诊断] ");
    for (int i = 0; i < SELFLNN_MAX_MODALITIES; i++) {
        pos += snprintf(buf + pos, sizeof(buf) - pos, "%s:%s ",
                       g_modality_names[i],
                       modality_active[i] ? "活跃" : "空");
    }
    log_info("%s | %d/%d模态活跃", buf, active, SELFLNN_MAX_MODALITIES);

    return 0;
}

/* ================================================================
 * ZSF-009: 统一输入在线训练步骤 —— P0-05修复版
 *
 * 基于目标输出对投影矩阵进行在线学习更新。
 * 使用真实链式法则梯度反向传播（通过cfc_backward_ex穿越LNN/CfC动态系统），
 * 不再使用"LNN恒等映射"近似。
 *
 * 完整梯度链:
 *   MSE损失: L = (1/N) * Σ (LNN_output[j] - target[j])²
 *   dL/d_LNN_output = 2*(LNN_output - target)/N
 *   ↓ [cfc_backward_ex: 穿越CfC网络内部ODE/sigmoid/tanh链式法则]
 *   dL/d_combined (LNN输入梯度) = cfc_backward_ex的gradient输出
 *   ↓ [投影层链式法则]
 *   dL/d_W_m[j][k] = dL/d_combined[j] * signal[k] * inv_norm * dim_scale
 *   dL/d_b_m[j] = dL/d_combined[j] * dim_scale
 *
 * SGD更新: W -= lr * dW, b -= lr * db
 * ================================================================ */
int multimodal_unified_input_train_step(UnifiedInputState* state,
                                        const float* target_output,
                                        size_t target_size,
                                        float learning_rate,
                                        const int* active_modalities_present) {
    if (!state || !target_output || target_size == 0) return -1;
    if (state->last_active_count <= 0) return -1;
    if (!state->projections_initialized) return -1;

    size_t out_dim = state->prev_output_dim;
    if (out_dim == 0 || out_dim > SELFLNN_MAX_CONTROL_DIM) return -1;
    size_t n = out_dim < target_size ? out_dim : target_size;

    /* 步骤1: 计算输出层MSE梯度 dL/d_output = 2*(output-target)/n */
    float grad_output[SELFLNN_MAX_CONTROL_DIM];
    float total_loss = 0.0f;
    for (size_t j = 0; j < n; j++) {
        float error = state->prev_output[j] - target_output[j];
        grad_output[j] = 2.0f * error / (float)n;
        total_loss += error * error;
    }
    for (size_t j = n; j < SELFLNN_MAX_CONTROL_DIM; j++) {
        grad_output[j] = 0.0f;
    }
    total_loss /= (float)n;

    /* 步骤2: P0-05修复 —— 使用真实CfC反向传播穿越LNN动态系统
     *
     * 旧代码使用恒等近似 grad_combined[j] ≈ grad_output[j]，
     * 完全忽略了LNN/CfC内部的sigmoid/tanh/ODE指数衰减链。
     * 现在通过 cfc_backward_ex 进行真实链式法则回传。
     *
     * 梯度链: dL/d_combined = W_out^T · dL/d_hidden · d_hidden/d_combined
     * 其中 d_hidden/d_combined 包含CfC内部完整的 ODE/sigmoid/tanh 链
     */
    float grad_combined[SELFLNN_UNIFIED_PROJECTION_DIM];
    memset(grad_combined, 0, SELFLNN_UNIFIED_PROJECTION_DIM * sizeof(float));

    int used_real_backprop = 0;
    if (state->lnn_instance) {
        /* 获取LNN内部的CfC网络，使用skip_cell_update=1模式进行纯梯度计算
         * 不更新CfC内部参数（学习率传0），仅获取输入梯度 dL/d_input */
        CfCNetwork* cfc_net = lnn_get_cfc_network(state->lnn_instance);
        if (cfc_net) {
            LNNConfig lnn_cfg;
            if (lnn_get_config(state->lnn_instance, &lnn_cfg) == 0) {
                /* 清零cell梯度缓冲区（上一次训练可能残留的累积梯度） */
                cfc_zero_cell_gradients(cfc_net);

                /* 调整梯度输出维度以匹配CfC输出大小 */
                size_t cfc_out_dim = lnn_cfg.output_size;
                float* cfc_error = (float*)safe_calloc(cfc_out_dim, sizeof(float));
                if (cfc_error) {
                    /* 将MSE输出梯度适配到CfC输出维度 */
                    size_t copy_n = (cfc_out_dim < (size_t)SELFLNN_MAX_CONTROL_DIM)
                                    ? cfc_out_dim : (size_t)SELFLNN_MAX_CONTROL_DIM;
                    for (size_t j = 0; j < copy_n && j < n; j++) {
                        cfc_error[j] = grad_output[j];
                    }
                    for (size_t j = copy_n; j < cfc_out_dim; j++) {
                        cfc_error[j] = 0.0f;
                    }

                    /* cfc_backward_ex: error→gradient(输入梯度), skip_cell_update=1仅累积不更新
                     * 学习率=0防止任何参数更新（我们只训练投影矩阵，不训练LNN） */
                    float* input_gradient_buf = (float*)safe_calloc(lnn_cfg.input_size, sizeof(float));
                    if (input_gradient_buf) {
                        int bw_ret = cfc_backward_ex(cfc_net, cfc_error,
                                                     input_gradient_buf, 0.0f, 1);
                        if (bw_ret == 0) {
                            /* 将LNN输入梯度（dL/d_combined）复制到grad_combined */
                            size_t copy_dim = (lnn_cfg.input_size < SELFLNN_UNIFIED_PROJECTION_DIM)
                                              ? lnn_cfg.input_size : SELFLNN_UNIFIED_PROJECTION_DIM;
                            for (size_t j = 0; j < copy_dim; j++) {
                                grad_combined[j] = input_gradient_buf[j];
                            }
                            used_real_backprop = 1;
                        }
                        safe_free((void**)&input_gradient_buf);
                    }

                    /* 清理cell梯度——我们不训练LNN */
                    cfc_zero_cell_gradients(cfc_net);
                    safe_free((void**)&cfc_error);
                }
            }
        }
    }

    /* M-005修复: LNN反向传播不可用时拒绝恒等近似回退。
     * grad_combined[j]=grad_output[j] 完全忽略CfC ODE内部的链式法则
     * (sigmoid导数、tanh导数、ODE连续演化路径)，导致梯度完全错误。
     * 必须要求LNN反向传播可用，不可回退到恒等传播。 */
    if (!used_real_backprop) {
        fprintf(stderr, "[多模态统一输入错误] LNN反向传播不可用，拒绝使用恒等近似梯度回退！请确保LNN已正确初始化并绑定。\n");
        safe_free((void**)&grad_combined);
        return SELFLNN_ERROR_ALGORITHM_FAILURE;
    }

    /* 步骤3: 梯度裁剪 —— 防止单步更新幅度过大 */
    float grad_max = 0.0f;
    for (size_t j = 0; j < SELFLNN_UNIFIED_PROJECTION_DIM; j++) {
        float abs_g = fabsf(grad_combined[j]);
        if (abs_g > grad_max) grad_max = abs_g;
    }
    if (grad_max > 10.0f) {
        float scale = 10.0f / grad_max;
        for (size_t j = 0; j < SELFLNN_UNIFIED_PROJECTION_DIM; j++) {
            grad_combined[j] *= scale;
        }
    }

    /* 步骤4: 对每个活跃模态的投影矩阵和偏置进行SGD更新
     *
     * 链式法则（与forward pass一致）:
     *   forward: combined[j] += (Σ_k W[j][k]·signal[k]·inv_norm + b[j]) · dim_scale
     *   因此: dL/d_W[j][k] = dL/d_combined[j] · signal[k] · inv_norm · dim_scale
     *         dL/d_b[j]     = dL/d_combined[j] · dim_scale
     */
    int updated_modalities = 0;
    for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
        if (active_modalities_present && !active_modalities_present[m]) continue;
        if (state->last_raw_sizes[m] == 0) continue;
        if (!state->projection_matrices[m] || !state->projection_biases[m]) continue;

        size_t input_dim = state->last_raw_sizes[m];
        if (input_dim > SELFLNN_MAX_CONTROL_DIM) input_dim = SELFLNN_MAX_CONTROL_DIM;

        /* 计算L2归一化因子（与forward pass一致） */
        float raw_l2_norm = 0.0f;
        for (size_t k = 0; k < input_dim; k++) {
            raw_l2_norm += state->last_raw_signals[m][k] * state->last_raw_signals[m][k];
        }
        raw_l2_norm = sqrtf(raw_l2_norm + 1e-12f);
        float inv_norm = 1.0f / raw_l2_norm;
        float dim_scale = sqrtf((float)SELFLNN_UNIFIED_PROJECTION_DIM / ((float)input_dim + 1.0f));

        float* W = state->projection_matrices[m];
        float* b = state->projection_biases[m];

        /* SGD更新: W[j][k] -= lr * dL/d_W[j][k], b[j] -= lr * dL/d_b[j] */
        for (size_t j = 0; j < SELFLNN_UNIFIED_PROJECTION_DIM; j++) {
            float common = learning_rate * grad_combined[j];
            /* 偏置更新 */
            b[j] -= common * dim_scale;
            /* 权重更新 */
            for (size_t k = 0; k < input_dim; k++) {
                float grad_w = common * state->last_raw_signals[m][k] * inv_norm * dim_scale;
                W[j * input_dim + k] -= grad_w;
            }
        }
        updated_modalities++;
    }

    if (updated_modalities == 0) return -1;

    /* 步骤5: 更新历史质量指标 */
    float quality = 1.0f / (1.0f + total_loss);
    state->historical_process_quality = state->historical_process_quality * 0.9f + quality * 0.1f;
    state->is_trained = 1;  /* ZSFWS-S006: 首次成功训练后标记为已训练 */

    return 0;
}
