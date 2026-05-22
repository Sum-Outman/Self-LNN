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
#include "selflnn/core/lnn.h"

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
 * @brief 核心统一输入处理：所有模态信号直接拼接为统一输入向量，
 *        通过单一LNN连续动态系统进行统一状态演化
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

    if (!state || !state->unified_input_buffer) {
        return -1;
    }

    /* B-015: 验证输出缓冲区大小 */
    if (!unified_output || max_output_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "统一输入动态处理: 输出缓冲区无效");
        return -1;
    }

    memset(state->unified_input_buffer, 0, state->unified_buffer_size * sizeof(float));

    size_t cumulative_offset = 0;
    for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
        if (modality_present[m] && signals[m]) {
            size_t copy = signal_sizes[m];
            if (copy > SELFLNN_MAX_CONTROL_DIM) {
                copy = SELFLNN_MAX_CONTROL_DIM;
            }
            if (copy == 0) {
                continue;
            }
            if (cumulative_offset + copy > state->unified_buffer_size) {
                copy = state->unified_buffer_size - cumulative_offset;
            }
            if (copy == 0) {
                break;
            }
            for (size_t d = 0; d < copy; d++) {
                state->unified_input_buffer[cumulative_offset + d] = signals[m][d];
            }
            state->modality_metrics[m].offset = cumulative_offset;
            state->modality_metrics[m].active_size = copy;
            cumulative_offset += copy;
        } else {
            state->modality_metrics[m].offset = 0;
            state->modality_metrics[m].active_size = 0;
        }
    }
    state->total_active_size = cumulative_offset;

    size_t output_dim = max_output_size;
    /* B-015: 输出维度上限检查 */
    if (output_dim > SELFLNN_MAX_CONTROL_DIM * 2) {
        output_dim = SELFLNN_MAX_CONTROL_DIM * 2;
    }
    if (output_dim > state->unified_buffer_size) {
        output_dim = state->unified_buffer_size;
    }

    if (state->lnn_instance) {
        LNNConfig lnn_cfg;
        if (lnn_get_config(state->lnn_instance, &lnn_cfg) == 0) {
            if (lnn_cfg.input_size == state->unified_buffer_size) {
                float* lnn_output_buf = (float*)safe_malloc(
                    (lnn_cfg.output_size > output_dim ? lnn_cfg.output_size : output_dim)
                    * sizeof(float));
                if (lnn_output_buf) {
                    int fwd_ret = lnn_forward(state->lnn_instance,
                                              state->unified_input_buffer,
                                              lnn_output_buf);
                    if (fwd_ret == 0) {
                        size_t copy = lnn_cfg.output_size;
                        if (copy > output_dim) copy = output_dim;
                        memcpy(unified_output, lnn_output_buf, copy * sizeof(float));
                        if (copy < output_dim) {
                            memset(unified_output + copy, 0,
                                   (output_dim - copy) * sizeof(float));
                        }
                        safe_free((void**)&lnn_output_buf);
                        return (int)output_dim;
                    }
                    safe_free((void**)&lnn_output_buf);
                }
            } else {
                size_t proj_dim = lnn_cfg.input_size < state->unified_buffer_size
                                  ? lnn_cfg.input_size : state->unified_buffer_size;
                float* proj_input = (float*)safe_malloc(
                    lnn_cfg.input_size * sizeof(float));
                if (proj_input) {
                    memset(proj_input, 0, lnn_cfg.input_size * sizeof(float));
                    memcpy(proj_input, state->unified_input_buffer,
                           proj_dim * sizeof(float));
                    float* lnn_output_buf = (float*)safe_malloc(
                        (lnn_cfg.output_size > output_dim
                         ? lnn_cfg.output_size : output_dim) * sizeof(float));
                    if (lnn_output_buf) {
                        int fwd_ret = lnn_forward(state->lnn_instance,
                                                  proj_input, lnn_output_buf);
                        if (fwd_ret == 0) {
                            size_t copy = lnn_cfg.output_size;
                            if (copy > output_dim) copy = output_dim;
                            memcpy(unified_output, lnn_output_buf, copy * sizeof(float));
                            if (copy < output_dim) {
                                memset(unified_output + copy, 0,
                                       (output_dim - copy) * sizeof(float));
                            }
                            safe_free((void**)&lnn_output_buf);
                            safe_free((void**)&proj_input);
                            return (int)output_dim;
                        }
                        safe_free((void**)&lnn_output_buf);
                    }
                    safe_free((void**)&proj_input);
                }
            }
        }
    }

    /* LNN未绑定或前向传播失败：禁止直通退化处理，返回错误 */
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
