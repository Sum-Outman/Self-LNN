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

    int is_initialized;
    size_t step_count;
    float average_activation;
};

static float uniform_random(void) {
    /* 使用加密安全随机数生成器替代rand()，确保权重初始化安全性 */
    return secure_random_float() * 2.0f - 1.0f;
}

static float xavier_scale(size_t fan_in, size_t fan_out) {
    return sqrtf(2.0f / (float)(fan_in + fan_out));
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
    return state;
}

void unified_lnn_state_free(UnifiedLNNState* state) {
    if (!state) return;
    for (int m = 0; m < UNIFIED_LNN_MAX_MODALITIES; m++) {
        safe_free((void**)&state->modality_projections[m]);
        safe_free((void**)&state->modality_biases[m]);
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
        const float* proj = state->modality_projections[m];
        const float* bias = state->modality_biases[m];
        const float* raw = raw_inputs[m];

        /* P1-031修复: 跨模态维度差异归一化
         * 不同模态维度差异巨大（视觉1024 vs 触觉64），直接投影求和
         * 导致高维模态主导输出。先对原始输入做L2归一化，再按sqrt(dim)缩放
         * 使各模态对combined_input的贡献处于相同量级。 */
        float raw_l2_norm = 0.0f;
        for (size_t j = 0; j < actual_raw; j++) {
            raw_l2_norm += raw[j] * raw[j];
        }
        raw_l2_norm = sqrtf(raw_l2_norm + 1e-12f);
        float inv_norm = 1.0f / raw_l2_norm;
        /* 缩放因子: sqrt(state_dim/raw_dim)使不同维度模态贡献均衡 */
        float dim_scale = sqrtf((float)state_dim / (float)(actual_raw + 1));

        for (size_t i = 0; i < state_dim; i++) {
            float dot = bias ? bias[i] : 0.0f;
            for (size_t j = 0; j < actual_raw; j++) {
                dot += proj[i * raw_dim + j] * raw[j] * inv_norm;
            }
            state->combined_input_buffer[i] += dot * dim_scale;
        }
    }

    if (!any_present) {
        memset(output, 0, max_output_size * sizeof(float));
        return 0;
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

    /* P0-010修复: 在线学习梯度反向传播到共享LNN内部
     * 原代码仅计算损失值但从不反向传播，学习信号在输出投影处截断。
     * 修复流程：
     *   1. 计算输出投影层的损失梯度 dL/d(output)
     *   2. 更新输出投影权重 W_out 和偏置 b_out (SGD)
     *   3. 通过链式法则将误差反向传播到LNN隐藏状态 dL/dh
     *   4. 构造虚拟目标并调用 lnn_backward 传播误差到CfC内部
     * 确保梯度流经: 损失→输出投影→LNN→CfC门控/ODE/时间常数
     * P0-010修复: 完整梯度链（使用LNN API避免不完整类型访问） */
    LNNConfig lnn_cfg;
    if (state->shared_lnn && lnn_get_config(state->shared_lnn, &lnn_cfg) == 0
        && lnn_cfg.enable_training
        && target_output && loss > 1e-8f && tgt_size > 0) {

        size_t state_dim = state->config.state_dimension;
        float lr = state->config.learning_rate;
        float* h = state->hidden_state_buffer;
        float inv_n = 1.0f / (float)tgt_size;

        /* 步骤1: 输出投影损失梯度
         * MSE: L = sqrt(mean((out-target)^2))
         * dL/d(out[i]) = (out[i] - target[i]) / (loss * N)
         * 当loss很小时使用 safe_inv_loss */
        float safe_inv_loss = 1.0f / (loss * (float)tgt_size + 1e-10f);
        float dL_dout[UNIFIED_LNN_DEFAULT_OUTPUT_DIM];
        for (size_t i = 0; i < tgt_size; i++) {
            dL_dout[i] = (temp_output[i] - target_output[i]) * safe_inv_loss;
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
        float* dL_dh = (float*)calloc(state_dim, sizeof(float));
        if (dL_dh) {
            for (size_t j = 0; j < state_dim; j++) {
                for (size_t i = 0; i < tgt_size; i++) {
                    dL_dh[j] += state->output_weight[i * state_dim + j] * dL_dout[i];
                }
            }

            /* 步骤4: 构造虚拟目标并调用lnn_backward进行完整反向传播
             * virtual_target[j] = h[j] - dL_dh[j]
             * lnn_backward内部: error = 2*(output - target)/N
             * = 2*(h - (h - dL_dh))/N = 2*dL_dh/N → 梯度方向正确 */
            float* virtual_target = (float*)calloc(state_dim, sizeof(float));
            if (virtual_target) {
                for (size_t j = 0; j < state_dim; j++) {
                    virtual_target[j] = h[j] - dL_dh[j];
                }

                float backward_loss = 0.0f;
                lnn_backward(state->shared_lnn, virtual_target, &backward_loss);
                free(virtual_target);
            }
            free(dL_dh);
        }
    }

    return 0;
}

int unified_lnn_state_save(const UnifiedLNNState* state, const char* filepath) {
    if (!state || !state->is_initialized || !filepath) return -1;
    FILE* f = fopen(filepath, "wb");
    if (!f) return -1;

    const char* header = "ULNNSTATE1";
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

    fclose(f);
    return 0;
}

UnifiedLNNState* unified_lnn_state_load(const char* filepath) {
    if (!filepath) return NULL;
    FILE* f = fopen(filepath, "rb");
    if (!f) return NULL;

    char header[12] = {0};
    if (fread(header, 1, 11, f) != 11 || strncmp(header, "ULNNSTATE1", 11) != 0) {
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
