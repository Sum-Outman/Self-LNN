#include "selflnn/core/unified_lnn_state.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/lnn.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/secure_random.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <float.h>

struct UnifiedLNNState {
    UnifiedLNNStateConfig config;

    LNN* shared_lnn;

    float* modality_projections[UNIFIED_LNN_MAX_MODALITIES];
    float* modality_biases[UNIFIED_LNN_MAX_MODALITIES];
    size_t modality_proj_sizes[UNIFIED_LNN_MAX_MODALITIES];

    float* output_weight;
    float* output_bias;

    float* combined_input_buffer;
    float* hidden_state_buffer;

    /* Z-P0-01修复: 逐模态在线统计归一化 —— 消除跨模态量级污染
     * 每个模态独立维护EMA运行均值和方差，实现真正的Z-score归一化。
     * 视觉(0-255量级)、音频(-80~20dB)、传感器(0-5V)、文本嵌入(0-1)
     * 在未经独立归一化的情况下直接累加会造成高量级模态主导梯度。 */
    float* modality_running_mean[UNIFIED_LNN_MAX_MODALITIES];
    float* modality_running_var[UNIFIED_LNN_MAX_MODALITIES];
    size_t modality_sample_count[UNIFIED_LNN_MAX_MODALITIES];
    int    modality_stats_ready[UNIFIED_LNN_MAX_MODALITIES];
    #define MODALITY_STATS_MIN_SAMPLES 32

    int is_initialized;
    size_t step_count;
    float average_activation;
    /* ZSFWS-P3: 模态切换追踪——防止跨模态隐藏状态污染 */
    int    last_active_modality_bits;     /* 上一帧活跃模态位掩码 */
    #define MODALITY_CHANGE_THRESHOLD 0.5f  /* 模态集合变化超过50%触发重置 */

    /* 架构优化: 稀疏门控跨模态连接层
     *
     * 设计原理:
     *   1. 隐藏状态分区: 9×45 dim 模态私有区 + 107 dim 跨模态融合区 = 512 dim
     *   2. 模态隔离: 每个模态的投影输出仅写入自己的私有区 (MODALITY_STATE_OFFSET)
     *   3. 稀疏门控: G[m_in][m_out] = 可学习的跨模态信息流权重
     *      - 初始化为自连接=1 + 近邻=0.1 + 其余=0 (稀疏)
     *      - 训练时受L1正则约束保持稀疏性
     *   4. 融合区: 接收所有活跃模态的门控贡献
     *
     * 数据流（隔离模式）:
     *   视觉输入 → 视觉私有区[0:44]
     *   音频输入 → 音频私有区[45:89]
     *   ...
     *   融合区[405:511] = Σ G[m][fusion] * 私有区[m]
     *
     * 数据流（共享模式/向后兼容）:
     *   所有模态 → combined_input_buffer[0:511] → LNN（同原来） */
    float  cross_modal_gates[UNIFIED_LNN_MAX_MODALITIES][UNIFIED_LNN_MAX_MODALITIES + 1];
    int    modality_isolation_enabled;     /* 是否启用模态隔离 */
    float  gate_regularization_strength;   /* L1正则强度（默认0.001） */
};

static float uniform_random(void) {
    /* 使用加密安全随机数生成器替代rand()，确保权重初始化安全性 */
    return secure_random_float() * 2.0f - 1.0f;
}

static float xavier_scale(size_t fan_in, size_t fan_out) {
    return sqrtf(2.0f / (float)(fan_in + fan_out));
}

/* Z-P0-01修复: 逐模态在线Z-score归一化
 * 使用EMA维护每个模态的运行均值和方差，实现真正的独立统计归一化。
 * 参数: state(状态句柄), modality(模态索引), raw(原始特征向量), size(特征维度)
 * 返回值: 0成功, -1失败
 * 算法:
 *   - 前MODALITY_STATS_MIN_SAMPLES个样本用于收集初始统计量
 *   - 之后每个样本用EMA更新: mean = 0.99*mean + 0.01*sample_mean
 *   - Z-score归一化: norm = (raw - mean) / sqrt(var + 1e-8)
 *   - 截断异常值: |norm| > 5.0 → norm = 5.0 * sign(norm)
 */
static int modality_online_zscore_normalize(UnifiedLNNState* state, int modality,
                                             float* raw, size_t size) {
    if (!state || !raw || size == 0) return -1;
    if (!state->modality_running_mean[modality]) {
        state->modality_running_mean[modality] = (float*)safe_calloc(size, sizeof(float));
        state->modality_running_var[modality]  = (float*)safe_calloc(size, sizeof(float));
        if (!state->modality_running_mean[modality] || !state->modality_running_var[modality])
            return -1;
    }

    float* mean = state->modality_running_mean[modality];
    float* var  = state->modality_running_var[modality];
    size_t n    = state->modality_sample_count[modality];

    if (n < MODALITY_STATS_MIN_SAMPLES) {
        /* 前32个样本：累积Welford在线统计算法（数值稳定） */
        for (size_t i = 0; i < size; i++) {
            float delta = raw[i] - mean[i];
            mean[i] += delta / (float)(n + 1);
            float delta2 = raw[i] - mean[i];
            var[i] += delta * delta2;
        }
        state->modality_sample_count[modality]++;
        if (n + 1 >= MODALITY_STATS_MIN_SAMPLES) {
            for (size_t i = 0; i < size; i++) var[i] /= (float)(MODALITY_STATS_MIN_SAMPLES - 1);
            state->modality_stats_ready[modality] = 1;
        }
        /* 统计量未就绪时使用L2归一化（P1-031已有方案） */
        float l2 = 1e-12f;
        for (size_t i = 0; i < size; i++) l2 += raw[i] * raw[i];
        l2 = sqrtf(l2);
        float inv = 1.0f / l2;
        for (size_t i = 0; i < size; i++) raw[i] *= inv;
        return 0;
    }

    /* 统计量已就绪：EMA更新 + Z-score归一化 + 截断 */
    float ema_decay = 0.99f;
    float ema_new   = 0.01f;
    for (size_t i = 0; i < size; i++) {
        /* EMA更新运行统计 */
        float delta = raw[i] - mean[i];
        mean[i] = ema_decay * mean[i] + ema_new * raw[i];
        var[i]  = ema_decay * var[i]  + ema_new * delta * delta;

        /* Z-score归一化 */
        float std = sqrtf(var[i] + 1e-8f);
        float z = (raw[i] - mean[i]) / std;

        /* 截断极端异常值（|z|>5）防止单一样本污染状态 */
        if (z > 5.0f) z = 5.0f;
        else if (z < -5.0f) z = -5.0f;

        raw[i] = z;
    }
    state->modality_sample_count[modality]++;
    return 0;
}

UnifiedLNNStateConfig unified_lnn_state_get_default_config(void) {
    UnifiedLNNStateConfig config;
    memset(&config, 0, sizeof(UnifiedLNNStateConfig));
    for (int m = 0; m < UNIFIED_LNN_MAX_MODALITIES; m++) {
        switch (m) {
            case UNIFIED_MODALITY_VISION: config.raw_dimensions[m] = 1024; break;
            case UNIFIED_MODALITY_AUDIO:  config.raw_dimensions[m] = 256;  break;
            case UNIFIED_MODALITY_TEXT:   config.raw_dimensions[m] = 512;  break;
            case UNIFIED_MODALITY_SENSOR: config.raw_dimensions[m] = 128;  break;
            default:                      config.raw_dimensions[m] = 64;   break;
        }
    }
    config.state_dimension = UNIFIED_LNN_DEFAULT_STATE_DIM;
    config.output_dimension = UNIFIED_LNN_DEFAULT_OUTPUT_DIM;
    config.evolution_delta_t = 0.1f;
    config.learning_rate = 0.001f;
    config.ode_solver_type = 0;
    config.enable_online_learning = 1;
    config.enable_noise_injection = 0;
    config.noise_std = 0.001f;
    config.enable_modality_isolation = 1; /* 架构优化: 默认启用模态隔离 */
    return config;
}

static int allocate_modality_projection(UnifiedLNNState* state, int modality) {
    size_t raw_dim = state->config.raw_dimensions[modality];
    size_t state_dim = state->config.state_dimension;
    if (raw_dim == 0) {
        state->modality_projections[modality] = NULL;
        state->modality_biases[modality] = NULL;
        state->modality_proj_sizes[modality] = 0;
        return 0;
    }
    size_t proj_size = state_dim * raw_dim;
    state->modality_projections[modality] = (float*)safe_calloc(proj_size, sizeof(float));
    state->modality_biases[modality] = (float*)safe_calloc(state_dim, sizeof(float));
    if (!state->modality_projections[modality] || !state->modality_biases[modality]) {
        safe_free((void**)&state->modality_projections[modality]);
        safe_free((void**)&state->modality_biases[modality]);
        return -1;
    }
    state->modality_proj_sizes[modality] = proj_size;
    float scale = xavier_scale(raw_dim, state_dim);
    for (size_t i = 0; i < proj_size; i++) {
        state->modality_projections[modality][i] = uniform_random() * scale;
    }
    return 0;
}

/* 架构优化: 初始化稀疏门控跨模态连接矩阵
 *
 * 初始化策略（保证初始稀疏性）:
 *   G[m][m]  = 1.0   (自连接: 每个模态处理自己的数据)
 *   G[m][m±1]= 0.1   (近邻: 相关模态间的基础信息交换)
 *   G[m][9]  = 0.5   (融合通道: 每个模态向融合区输出)
 *   其余     = 0.0   (无连接: 无关联的模态间不交换信息)
 *
 * 训练时通过L1正则约束保持稀疏性:
 *   loss += gate_regularization_strength * Σ|G[m][n]|  (m ≠ n)
 */
static void init_cross_modal_gates(UnifiedLNNState* state) {
    int N = UNIFIED_LNN_MAX_MODALITIES; /* 9 */
    int F = N; /* fusion channel index = 9 */
    for (int m = 0; m < N; m++) {
        for (int n = 0; n <= N; n++) {
            if (n == m) {
                state->cross_modal_gates[m][n] = 1.0f; /* 自连接 */
            } else if (n == F) {
                state->cross_modal_gates[m][n] = 0.5f; /* 融合通道 */
            } else if (n == m + 1 || n == m - 1) {
                state->cross_modal_gates[m][n] = 0.1f; /* 近邻 */
            } else {
                state->cross_modal_gates[m][n] = 0.0f; /* 无连接 */
            }
        }
    }
}

/* 架构优化: 计算跨模态门控贡献
 *
 * 对目标列n, 计算所有源模态m对它的门控贡献:
 *   contribution[n] = Σ G[m][n] * private_state[m]
 *
 * 这确保:
 *   - 视觉私有区仅包含视觉数据 (G[vision][vision]=1.0, 其余G[?][vision]=0)
 *   - 融合区包含所有模态的加权贡献
 *   - 训练时可通过梯度更新G矩阵来学习哪些模态需要共享信息
 */
static void apply_cross_modal_gates(UnifiedLNNState* state,
                                     const float* private_states[UNIFIED_LNN_MAX_MODALITIES],
                                     float* combined_buffer,
                                     const int* modality_present) {
    int N = UNIFIED_LNN_MAX_MODALITIES;
    int P = MODALITY_PRIVATE_DIM;
    int F = N; /* fusion column index */

    /* 步骤1: 模态私有区 —— 仅写入自连接 (零污染) */
    for (int m = 0; m < N; m++) {
        if (!modality_present || !modality_present[m]) continue;
        if (!private_states[m]) continue;
        float gate_self = state->cross_modal_gates[m][m]; /* ~1.0 */
        size_t offset = MODALITY_STATE_OFFSET(m);
        for (int i = 0; i < P; i++) {
            combined_buffer[offset + i] = private_states[m][i] * gate_self;
        }
    }

    /* 步骤2: 近邻跨模态交换 —— 仅稀疏门控的非零连接 */
    for (int m = 0; m < N; m++) {
        if (!modality_present || !modality_present[m]) continue;
        if (!private_states[m]) continue;
        for (int n = 0; n < N; n++) {
            if (n == m) continue; /* 已处理 */
            if (!modality_present || !modality_present[n]) continue;
            float gate = state->cross_modal_gates[m][n];
            if (gate == 0.0f) continue; /* 稀疏跳过 */
            size_t dst_off = MODALITY_STATE_OFFSET(n);
            for (int i = 0; i < P; i++) {
                combined_buffer[dst_off + i] += private_states[m][i] * gate;
            }
        }
    }

    /* 步骤3: 融合区 —— 所有活跃模态的门控贡献 */
    size_t fusion_off = FUSION_STATE_OFFSET;
    int fusion_dim = CROSS_MODAL_FUSION_DIM;
    for (int i = 0; i < fusion_dim; i++) combined_buffer[fusion_off + i] = 0.0f;

    for (int m = 0; m < N; m++) {
        if (!modality_present || !modality_present[m]) continue;
        if (!private_states[m]) continue;
        float gate_fusion = state->cross_modal_gates[m][F]; /* ~0.5 */
        for (int i = 0; i < fusion_dim && i < P; i++) {
            combined_buffer[fusion_off + i] += private_states[m][i] * gate_fusion;
        }
    }

    /* 步骤4: L1稀疏正则 —— 累加非自连接的|G|到状态中供训练步骤使用 */
    float l1_sum = 0.0f;
    for (int m = 0; m < N; m++) {
        for (int n = 0; n < N; n++) {
            if (n == m) continue;
            l1_sum += fabsf(state->cross_modal_gates[m][n]);
        }
    }
    /* L1正则项写入combined_buffer尾部（训练步骤读取） */
    if (fusion_dim > 1) {
        combined_buffer[fusion_off + fusion_dim - 1] = l1_sum;
    }
}

UnifiedLNNState* unified_lnn_state_create(const UnifiedLNNStateConfig* config) {
    if (!config) return NULL;
    if (config->state_dimension == 0 || config->output_dimension == 0) return NULL;

    UnifiedLNNState* state = (UnifiedLNNState*)safe_calloc(1, sizeof(UnifiedLNNState));
    if (!state) return NULL;

    state->config = *config;
    size_t state_dim = config->state_dimension;

    for (int m = 0; m < UNIFIED_LNN_MAX_MODALITIES; m++) {
        if (allocate_modality_projection(state, m) != 0) {
            for (int n = 0; n <= m; n++) {
                safe_free((void**)&state->modality_projections[n]);
                safe_free((void**)&state->modality_biases[n]);
            }
            safe_free((void**)&state);
            return NULL;
        }
    }

    size_t out_size = state_dim * config->output_dimension;
    state->output_weight = (float*)safe_calloc(out_size, sizeof(float));
    state->output_bias = (float*)safe_calloc(config->output_dimension, sizeof(float));
    if (!state->output_weight || !state->output_bias) {
        safe_free((void**)&state->output_weight);
        safe_free((void**)&state->output_bias);
        for (int m = 0; m < UNIFIED_LNN_MAX_MODALITIES; m++) {
            safe_free((void**)&state->modality_projections[m]);
            safe_free((void**)&state->modality_biases[m]);
        }
        safe_free((void**)&state);
        return NULL;
    }
    float out_scale = xavier_scale(state_dim, config->output_dimension);
    for (size_t i = 0; i < out_size; i++) {
        state->output_weight[i] = uniform_random() * out_scale;
    }

    state->shared_lnn = NULL;

    state->combined_input_buffer = (float*)safe_calloc(state_dim, sizeof(float));
    state->hidden_state_buffer = (float*)safe_calloc(state_dim, sizeof(float));
    if (!state->combined_input_buffer || !state->hidden_state_buffer) {
        safe_free((void**)&state->combined_input_buffer);
        safe_free((void**)&state->hidden_state_buffer);
        for (int m = 0; m < UNIFIED_LNN_MAX_MODALITIES; m++) {
            safe_free((void**)&state->modality_projections[m]);
            safe_free((void**)&state->modality_biases[m]);
        }
        safe_free((void**)&state->output_weight);
        safe_free((void**)&state->output_bias);
        safe_free((void**)&state);
        return NULL;
    }

    state->is_initialized = 1;
    state->step_count = 0;
    state->average_activation = 0.0f;

    /* 架构优化: 初始化模态隔离 */
    state->modality_isolation_enabled = config->enable_modality_isolation ? 1 : 0;
    state->gate_regularization_strength = 0.001f;
    init_cross_modal_gates(state);

    return state;
}

void unified_lnn_state_free(UnifiedLNNState* state) {
    if (!state) return;
    for (int m = 0; m < UNIFIED_LNN_MAX_MODALITIES; m++) {
        safe_free((void**)&state->modality_projections[m]);
        safe_free((void**)&state->modality_biases[m]);
        /* Z-P0-01修复: 释放逐模态运行统计数据 */
        safe_free((void**)&state->modality_running_mean[m]);
        safe_free((void**)&state->modality_running_var[m]);
    }
    safe_free((void**)&state->output_weight);
    safe_free((void**)&state->output_bias);
    safe_free((void**)&state->combined_input_buffer);
    safe_free((void**)&state->hidden_state_buffer);
    safe_free((void**)&state);
}

int unified_lnn_state_step(UnifiedLNNState* state,
                          const float* raw_inputs[UNIFIED_LNN_MAX_MODALITIES],
                          const size_t raw_sizes[UNIFIED_LNN_MAX_MODALITIES],
                          const int modality_present[UNIFIED_LNN_MAX_MODALITIES],
                          float* output, size_t max_output_size) {
    if (!state || !state->is_initialized) return -1;
    if (!output || max_output_size == 0) return -1;

    size_t state_dim = state->config.state_dimension;

    memset(state->combined_input_buffer, 0, state_dim * sizeof(float));

    int any_present = 0;
    for (int m = 0; m < UNIFIED_LNN_MAX_MODALITIES; m++) {
        if (!modality_present || !modality_present[m]) continue;
        if (!raw_inputs || !raw_inputs[m]) continue;
        if (!state->modality_projections[m]) continue;

        size_t raw_dim = state->config.raw_dimensions[m];
        size_t actual_raw = (raw_sizes && raw_sizes[m] > 0) ? raw_sizes[m] : raw_dim;
        if (actual_raw > raw_dim) actual_raw = raw_dim;
        if (actual_raw == 0) continue;

        any_present = 1;

        /* Z-P0-01修复: 逐模态独立Z-score归一化，消除跨模态量级污染 */
        float* normalized_raw = (float*)safe_malloc(actual_raw * sizeof(float));
        if (normalized_raw) {
            memcpy(normalized_raw, raw_inputs[m], actual_raw * sizeof(float));
            modality_online_zscore_normalize(state, m, normalized_raw, actual_raw);
        } else {
            normalized_raw = (float*)raw_inputs[m];
        }

        const float* proj = state->modality_projections[m];
        const float* bias = state->modality_biases[m];

        /* 架构优化: 隔离模式—仅投影到模态私有区（45维），11倍加速
         * 非隔离模式—全512维投影（向后兼容） */
        if (state->modality_isolation_enabled) {
            size_t proj_dim = MODALITY_PRIVATE_DIM;
            size_t dst_off = MODALITY_STATE_OFFSET(m);
            for (size_t i = 0; i < proj_dim; i++) {
                float dot = bias ? bias[i] : 0.0f;
                for (size_t j = 0; j < actual_raw; j++) {
                    dot += proj[i * raw_dim + j] * normalized_raw[j];
                }
                state->combined_input_buffer[dst_off + i] = dot;
            }
        } else {
            for (size_t i = 0; i < state_dim; i++) {
                float dot = bias ? bias[i] : 0.0f;
                for (size_t j = 0; j < actual_raw; j++) {
                    dot += proj[i * raw_dim + j] * normalized_raw[j];
                }
                state->combined_input_buffer[i] += dot;
            }
        }

        if (normalized_raw != raw_inputs[m]) {
            safe_free((void**)&normalized_raw);
        }
    }

    if (!any_present) {
        memset(output, 0, max_output_size * sizeof(float));
        return 0;
    }

    /* 架构优化: 模态隔离后处理
     *
     * 隔离模式下投影已直接写入私有区[0:404]，仅需门控交叉贡献。
     * 步骤: 提取私有切片→清零buffer→apply_cross_modal_gates重建。 */
    if (state->modality_isolation_enabled) {
        float* private_states[UNIFIED_LNN_MAX_MODALITIES] = {NULL};
        int P = MODALITY_PRIVATE_DIM;

        /* 步骤A: 从combined_input_buffer提取每个模态的私有切片（已是最终值） */
        for (int m = 0; m < UNIFIED_LNN_MAX_MODALITIES; m++) {
            if (!modality_present || !modality_present[m]) continue;
            private_states[m] = (float*)safe_calloc(P, sizeof(float));
            if (private_states[m]) {
                size_t offset = MODALITY_STATE_OFFSET(m);
                memcpy(private_states[m], &state->combined_input_buffer[offset],
                       P * sizeof(float));
            }
        }

        /* 步骤B: 清零+门控重建（自连接=1保持私有区值，近邻门控添加交叉贡献） */
        memset(state->combined_input_buffer, 0, state_dim * sizeof(float));
        apply_cross_modal_gates(state, (const float**)private_states,
                                 state->combined_input_buffer, modality_present);

        /* 步骤C: 释放临时缓冲区 */
        for (int m = 0; m < UNIFIED_LNN_MAX_MODALITIES; m++) {
            safe_free((void**)&private_states[m]);
        }
    }

    /* ZSFWS-P3: 模态切换检测——防止跨模态隐藏状态污染
     * 当活跃模态集合变化超过50%时，重置隐藏状态以避免
     * 前一模态的残留状态污染新模态的推理结果。
     * 例如：从纯视觉(只看图像)切换到纯音频(只听声音)时，
     * 视觉特征的隐藏状态残留会导致音频处理偏差。 */
    {
        int current_bits = 0;
        for (int m = 0; m < UNIFIED_LNN_MAX_MODALITIES; m++) {
            if (modality_present && modality_present[m]) {
                current_bits |= (1 << m);
            }
        }
        if (state->step_count > 0 && state->last_active_modality_bits != 0) {
            int changed = current_bits ^ state->last_active_modality_bits;
            int prev_count = 0, changed_count = 0;
            for (int m = 0; m < UNIFIED_LNN_MAX_MODALITIES; m++) {
                if (state->last_active_modality_bits & (1 << m)) prev_count++;
                if (changed & (1 << m)) changed_count++;
            }
            if (prev_count > 0 && (float)changed_count / (float)prev_count > MODALITY_CHANGE_THRESHOLD) {
                memset(state->hidden_state_buffer, 0, state_dim * sizeof(float));
            }
        }
        state->last_active_modality_bits = current_bits;
    }

    if (state->config.enable_noise_injection && state->config.noise_std > 0.0f) {
        for (size_t i = 0; i < state_dim; i++) {
            state->combined_input_buffer[i] += uniform_random() * state->config.noise_std;
        }
    }

    /* 使用共享主LNN进行状态演化（单一模型原则） */
    if (!state->shared_lnn) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "统一LNN状态处理器：共享LNN未绑定，禁止直通退化处理");
        memset(output, 0, max_output_size * sizeof(float));
        return -1;
    }
    int lnn_result = lnn_forward(state->shared_lnn,
                                 state->combined_input_buffer,
                                 state->hidden_state_buffer);
    if (lnn_result != 0) {
        memset(output, 0, max_output_size * sizeof(float));
        return -1;
    }

    size_t out_dim = state->config.output_dimension;
    size_t copy_out = out_dim < max_output_size ? out_dim : max_output_size;
    for (size_t i = 0; i < copy_out; i++) {
        float val = state->output_bias ? state->output_bias[i] : 0.0f;
        for (size_t j = 0; j < state_dim; j++) {
            val += state->output_weight[i * state_dim + j] * state->hidden_state_buffer[j];
        }
        output[i] = val;
    }
    if (copy_out < max_output_size) {
        memset(output + copy_out, 0, (max_output_size - copy_out) * sizeof(float));
    }

    float act = 0.0f;
    for (size_t i = 0; i < state_dim; i++) {
        act += fabsf(state->hidden_state_buffer[i]);
    }
    act /= (float)state_dim;
    state->average_activation = state->average_activation * 0.9f + act * 0.1f;
    state->step_count++;

    return (int)copy_out;
}

int unified_lnn_state_forward(UnifiedLNNState* state,
                            const float* combined_input,
                            float* hidden_state_out) {
    if (!state || !state->is_initialized || !combined_input || !hidden_state_out) {
        return -1;
    }
    if (!state->shared_lnn) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "统一LNN状态处理器：共享LNN未绑定，禁止直通退化处理");
        return -1;
    }
    return lnn_forward(state->shared_lnn, combined_input, hidden_state_out);
}

int unified_lnn_state_reset(UnifiedLNNState* state) {
    if (!state || !state->is_initialized) return -1;
    if (state->hidden_state_buffer) {
        memset(state->hidden_state_buffer, 0, state->config.state_dimension * sizeof(float));
    }
    state->step_count = 0;
    state->average_activation = 0.0f;
    return 0;
}

int unified_lnn_state_get_hidden_state(const UnifiedLNNState* state,
                                      float* hidden_out, size_t max_size) {
    if (!state || !state->is_initialized || !hidden_out) return -1;
    size_t state_dim = state->config.state_dimension;
    size_t copy = state_dim < max_size ? state_dim : max_size;
    memcpy(hidden_out, state->hidden_state_buffer, copy * sizeof(float));
    return (int)copy;
}

int unified_lnn_state_set_learning_rate(UnifiedLNNState* state, float lr) {
    if (!state || !state->is_initialized) return -1;
    state->config.learning_rate = lr;
    return 0;
}

int unified_lnn_state_set_shared_lnn(UnifiedLNNState* state, void* lnn_instance) {
    if (!state || !state->is_initialized) return -1;
    state->shared_lnn = (LNN*)lnn_instance;
    return 0;
}

int unified_lnn_state_get_config(const UnifiedLNNState* state,
                                UnifiedLNNStateConfig* config) {
    if (!state || !state->is_initialized || !config) return -1;
    memcpy(config, &state->config, sizeof(UnifiedLNNStateConfig));
    return 0;
}

int unified_lnn_state_train_step(UnifiedLNNState* state,
                               const float* raw_inputs[UNIFIED_LNN_MAX_MODALITIES],
                               const size_t raw_sizes[UNIFIED_LNN_MAX_MODALITIES],
                               const int modality_present[UNIFIED_LNN_MAX_MODALITIES],
                               const float* target_output, size_t target_size,
                               float* loss_out) {
    if (!state || !state->is_initialized) return -1;
    if (!state->config.enable_online_learning) {
        if (loss_out) *loss_out = 0.0f;
        return 0;
    }

    float temp_output[UNIFIED_LNN_DEFAULT_OUTPUT_DIM];
    size_t out_dim = state->config.output_dimension;
    if (out_dim > UNIFIED_LNN_DEFAULT_OUTPUT_DIM) out_dim = UNIFIED_LNN_DEFAULT_OUTPUT_DIM;

    int step_result = unified_lnn_state_step(state, raw_inputs, raw_sizes,
                                            modality_present,
                                            temp_output, out_dim);
    if (step_result <= 0) {
        if (loss_out) *loss_out = 1e10f;
        return -1;
    }

    float loss = 0.0f;
    size_t tgt_size = target_size < out_dim ? target_size : out_dim;
    if (target_output) {
        for (size_t i = 0; i < tgt_size; i++) {
            float err = temp_output[i] - target_output[i];
            loss += err * err;
        }
        loss = sqrtf(loss / (float)tgt_size);
    }
    if (loss_out) *loss_out = loss;

    /* P0-010 + ZSFWS-FIX: 使用MSE梯度(dL_MSE/dout=2*err/N)替代RMSE梯度
     * 原RMSE梯度公式  dL_RMSE/dout = err / (loss*N) 在 loss→0 时
     * 分母趋零导致梯度爆炸，是训练收敛后NaN崩溃的根本原因。
     * MSE梯度无除以loss，全程数值稳定。loss报告值仍为RMSE供监控。 */
    LNNConfig lnn_cfg;
    if (state->shared_lnn && lnn_get_config(state->shared_lnn, &lnn_cfg) == 0
        && lnn_cfg.enable_training
        && target_output && loss > 1e-8f && tgt_size > 0) {

        size_t state_dim = state->config.state_dimension;
        float lr = state->config.learning_rate;
        float* h = state->hidden_state_buffer;
        float inv_n = 1.0f / (float)tgt_size;

        /* 步骤1: MSE梯度 (数值稳定, 无除以loss)
         * d(L_MSE)/d(out[i]) = 2*(out[i] - target[i]) / N */
        float dL_dout[UNIFIED_LNN_DEFAULT_OUTPUT_DIM];
        for (size_t i = 0; i < tgt_size; i++) {
            dL_dout[i] = 2.0f * (temp_output[i] - target_output[i]) * inv_n;
        }

        /* 步骤2: 更新输出投影权重（线性层 SGD）
         * W_out[i,j] -= lr * dL/dout[i] * h[j]
         * b_out[i] -= lr * dL/dout[i] */
        for (size_t i = 0; i < tgt_size; i++) {
            float lr_dout = lr * dL_dout[i];
            for (size_t j = 0; j < state_dim; j++) {
                state->output_weight[i * state_dim + j] -= lr_dout * h[j];
            }
            if (state->output_bias) {
                state->output_bias[i] -= lr_dout;
            }
        }

        /* 步骤3: 链式法则传播误差到LNN隐藏状态
         * dL/dh[j] = Σ_i W_out[i,j] * dL/dout[i] */
        float* dL_dh = (float*)safe_calloc(state_dim, sizeof(float));
        if (dL_dh) {
            for (size_t j = 0; j < state_dim; j++) {
                for (size_t i = 0; i < tgt_size; i++) {
                    dL_dh[j] += state->output_weight[i * state_dim + j] * dL_dout[i];
                }
            }

            /* 步骤4: dL_dh已计算完成。不再调用lnn_backward传播误差到CfC内部。
             * ZSFWS-FATAL5: 在线学习仅负责输出投影(W_out/b_out)和门控矩阵(G)的更新。
             * CfC内部权重(W_cfc, τ, gate_weights等)的更新由批次训练管道的Adam统一负责。
             * 分离职责: 在线学习=输出层微调, 批次训练=全参数优化。 */

            /* 架构优化: 门控矩阵L1正则梯度更新（移入dL_dh作用域内）
             * ZSFWS-FIX: 原代码在free(dL_dh)后访问dL_dh → use-after-free。 */
            if (state->modality_isolation_enabled) {
                float lr_gate = lr * 0.1f;
                float gate_lambda = state->gate_regularization_strength;
                int N = UNIFIED_LNN_MAX_MODALITIES;

                for (int m = 0; m < N; m++) {
                    for (int n = 0; n < N; n++) {
                        if (n == m) continue;
                        float g = state->cross_modal_gates[m][n];

                        float grad_l1 = gate_lambda * ((g > 0.0f) ? 1.0f : -1.0f);

                        float grad_hidden = 0.0f;
                        if (h && modality_present && modality_present[m] && modality_present[n]) {
                            size_t dst_off = MODALITY_STATE_OFFSET(n);
                            int P = MODALITY_PRIVATE_DIM;
                            for (int i = 0; i < P && (dst_off + i) < state_dim; i++) {
                                size_t src_off = MODALITY_STATE_OFFSET(m);
                                float priv_val = state->combined_input_buffer[src_off + i];
                                float gate_self = state->cross_modal_gates[m][m];
                                if (gate_self > 0.01f) priv_val /= gate_self;
                                grad_hidden += dL_dh[dst_off + i] * priv_val;
                            }
                        }

                        float grad_total = grad_l1 + grad_hidden;
                        if (grad_total > 0.1f) grad_total = 0.1f;
                        if (grad_total < -0.1f) grad_total = -0.1f;

                        state->cross_modal_gates[m][n] -= lr_gate * grad_total;
                        if (state->cross_modal_gates[m][n] > 1.0f)
                            state->cross_modal_gates[m][n] = 1.0f;
                        if (state->cross_modal_gates[m][n] < -0.5f)
                            state->cross_modal_gates[m][n] = -0.5f;
                    }
                }

                /* 融合通道门控 */
                int F = N;
                size_t fusion_off = FUSION_STATE_OFFSET;
                int fusion_dim = CROSS_MODAL_FUSION_DIM;
                for (int m = 0; m < N; m++) {
                    if (!modality_present || !modality_present[m]) continue;
                    float g = state->cross_modal_gates[m][F];
                    float grad_hidden = 0.0f;
                    size_t src_off = MODALITY_STATE_OFFSET(m);
                    int P = MODALITY_PRIVATE_DIM;
                    for (int i = 0; i < fusion_dim && i < P; i++) {
                        float priv_val = state->combined_input_buffer[src_off + i];
                        float gate_self = state->cross_modal_gates[m][m];
                        if (gate_self > 0.01f) priv_val /= gate_self;
                        grad_hidden += dL_dh[fusion_off + i] * priv_val;
                    }
                    if (grad_hidden > 0.1f) grad_hidden = 0.1f;
                    if (grad_hidden < -0.1f) grad_hidden = -0.1f;
                    state->cross_modal_gates[m][F] -= lr_gate * grad_hidden;
                    if (state->cross_modal_gates[m][F] > 1.0f)
                        state->cross_modal_gates[m][F] = 1.0f;
                    if (state->cross_modal_gates[m][F] < -0.5f)
                        state->cross_modal_gates[m][F] = -0.5f;
                }
            }

            safe_free((void**)&dL_dh);
        }
    }

    return 0;
}

int unified_lnn_state_save(const UnifiedLNNState* state, const char* filepath) {
    if (!state || !state->is_initialized || !filepath) return -1;
    FILE* f = fopen(filepath, "wb");
    if (!f) return -1;

    /* 架构优化: ULNNSTATE2 包含门控矩阵 */
    const char* header = "ULNNSTATE2";
    fwrite(header, 1, 11, f);
    fwrite(&state->config, sizeof(UnifiedLNNStateConfig), 1, f);
    fwrite(&state->step_count, sizeof(size_t), 1, f);

    size_t state_dim = state->config.state_dimension;
    for (int m = 0; m < UNIFIED_LNN_MAX_MODALITIES; m++) {
        size_t raw_dim = state->config.raw_dimensions[m];
        if (raw_dim > 0 && state->modality_projections[m]) {
            fwrite(state->modality_projections[m], sizeof(float), state_dim * raw_dim, f);
            fwrite(state->modality_biases[m], sizeof(float), state_dim, f);
        }
    }
    fwrite(state->output_weight, sizeof(float), state_dim * state->config.output_dimension, f);
    fwrite(state->output_bias, sizeof(float), state->config.output_dimension, f);

    /* 保存门控矩阵 */
    int N = UNIFIED_LNN_MAX_MODALITIES;
    fwrite(state->cross_modal_gates, sizeof(float), N * (N + 1), f);
    fwrite(&state->modality_isolation_enabled, sizeof(int), 1, f);

    fclose(f);
    return 0;
}

UnifiedLNNState* unified_lnn_state_load(const char* filepath) {
    if (!filepath) return NULL;
    FILE* f = fopen(filepath, "rb");
    if (!f) return NULL;

    char header[12] = {0};
    if (fread(header, 1, 11, f) != 11) {
        fclose(f);
        return NULL;
    }

    int is_v2 = (strncmp(header, "ULNNSTATE2", 11) == 0);
    if (strncmp(header, "ULNNSTATE1", 11) != 0 && !is_v2) {
        fclose(f);
        return NULL;
    }

    UnifiedLNNStateConfig config;
    if (fread(&config, sizeof(UnifiedLNNStateConfig), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    UnifiedLNNState* state = unified_lnn_state_create(&config);
    if (!state) { fclose(f); return NULL; }

    if (fread(&state->step_count, sizeof(size_t), 1, f) != 1) {
        unified_lnn_state_free(state);
        fclose(f);
        return NULL;
    }

    size_t state_dim = config.state_dimension;
    for (int m = 0; m < UNIFIED_LNN_MAX_MODALITIES; m++) {
        size_t raw_dim = config.raw_dimensions[m];
        if (raw_dim > 0 && state->modality_projections[m]) {
            if (fread(state->modality_projections[m], sizeof(float), state_dim * raw_dim, f) != state_dim * raw_dim) {
                unified_lnn_state_free(state);
                fclose(f);
                return NULL;
            }
            if (fread(state->modality_biases[m], sizeof(float), state_dim, f) != state_dim) {
                unified_lnn_state_free(state);
                fclose(f);
                return NULL;
            }
        }
    }
    fread(state->output_weight, sizeof(float), state_dim * config.output_dimension, f);
    fread(state->output_bias, sizeof(float), config.output_dimension, f);

    /* 架构优化: v2格式额外读取门控矩阵 */
    if (is_v2) {
        int N = UNIFIED_LNN_MAX_MODALITIES;
        fread(state->cross_modal_gates, sizeof(float), N * (N + 1), f);
        fread(&state->modality_isolation_enabled, sizeof(int), 1, f);
    }

    fclose(f);
    return state;
}

int unified_lnn_state_set_modality_raw_dim(UnifiedLNNState* state,
                                          UnifiedModalityType modality,
                                          size_t raw_dim) {
    if (!state || !state->is_initialized) return -1;
    if (modality < 0 || modality >= UNIFIED_LNN_MAX_MODALITIES) return -1;
    if (raw_dim == 0 || raw_dim > UNIFIED_LNN_MAX_RAW_DIM) return -1;

    if (state->modality_projections[modality]) {
        safe_free((void**)&state->modality_projections[modality]);
        safe_free((void**)&state->modality_biases[modality]);
        state->modality_proj_sizes[modality] = 0;
    }

    state->config.raw_dimensions[modality] = raw_dim;
    if (allocate_modality_projection(state, (int)modality) != 0) {
        state->config.raw_dimensions[modality] = 0;
        return -1;
    }
    return 0;
}

/* ================================================================
 * 架构优化: 模态隔离API实现
 * ================================================================ */

int unified_lnn_state_get_modality_state(const UnifiedLNNState* state,
                                         UnifiedModalityType modality,
                                         float* private_state_out, size_t max_size) {
    if (!state || !state->is_initialized || !private_state_out) return -1;
    if (modality < 0 || modality >= UNIFIED_LNN_MAX_MODALITIES) return -1;

    size_t copy = MODALITY_PRIVATE_DIM < max_size ? MODALITY_PRIVATE_DIM : max_size;
    size_t offset = MODALITY_STATE_OFFSET((int)modality);
    memcpy(private_state_out, &state->hidden_state_buffer[offset], copy * sizeof(float));
    return (int)copy;
}

int unified_lnn_state_get_fusion_state(const UnifiedLNNState* state,
                                       float* fusion_out, size_t max_size) {
    if (!state || !state->is_initialized || !fusion_out) return -1;

    size_t copy = CROSS_MODAL_FUSION_DIM < max_size ? CROSS_MODAL_FUSION_DIM : max_size;
    size_t offset = FUSION_STATE_OFFSET;
    memcpy(fusion_out, &state->hidden_state_buffer[offset], copy * sizeof(float));
    return (int)copy;
}

void unified_lnn_state_set_isolation_mode(UnifiedLNNState* state, int enable) {
    if (!state || !state->is_initialized) return;
    state->modality_isolation_enabled = enable ? 1 : 0;
    if (enable && state->config.state_dimension >= (size_t)FUSION_STATE_OFFSET + CROSS_MODAL_FUSION_DIM) {
        init_cross_modal_gates(state);
    }
}

int unified_lnn_state_get_isolation_enabled(const UnifiedLNNState* state) {
    if (!state || !state->is_initialized) return 0;
    return state->modality_isolation_enabled;
}

/* 架构优化: 模态轮换调度器
 *
 * 隔离模式下推荐训练管道按轮换顺序训练不同模态:
 *   step % 9 = 0 → 视觉   step % 9 = 1 → 音频   ...
 *
 * 非隔离模式下返回 -1（所有模态联合训练）。
 * 轮换调度防止单一模态主导梯度更新。 */
int unified_lnn_state_suggest_next_modality(UnifiedLNNState* state) {
    if (!state || !state->is_initialized) return -1;
    if (!state->modality_isolation_enabled) return -1;
    return (int)(state->step_count % UNIFIED_LNN_MAX_MODALITIES);
}
