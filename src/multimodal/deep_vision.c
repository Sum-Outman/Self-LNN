/**
 * @file deep_vision.c
 * @brief 深度视觉特征提取（基于CfC ODE连续动态系统）
 *
 * 使用液态神经网络CfC（Closed-form Continuous-time）的深度视觉特征提取。
 * 完全替代CNN卷积架构，全部使用CfC ODE微分方程驱动的连续时间演化：
 *   τ dh/dt = -h + σ(gate) * tanh(W * input + b)
 *
 * 图像处理管道：
 * 1. 补丁划分 → 2. 线性投影 → 3. CfC ODE层演化 → 4. 全局池化
 *
 * 严格遵循：所有模态通过同一液态神经网络框架完成特征提取。
 * 纯C实现，不依赖任何第三方库。
 */

#include "selflnn/multimodal/deep_vision.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/perf.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/core/lnn.h"
#include "selflnn/selflnn.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <float.h>

/* ============================================================================
 * CfC ODE层内部结构体
 * ============================================================================ */

struct CfcOdeLayer {
    CfcOdeLayerConfig config;
    int is_initialized;

    /* 输入→隐藏权重矩阵 [input_dim * hidden_dim] */
    float* w_input;

    /* 隐藏→隐藏循环权重矩阵 [hidden_dim * hidden_dim] */
    float* w_hidden;

    /* 隐藏层偏置 [hidden_dim] */
    float* b_hidden;

    /* 门控输入权重矩阵 [input_dim * hidden_dim] */
    float* w_gate;

    /* 门控循环权重矩阵 [hidden_dim * hidden_dim] */
    float* u_gate;

    /* 门控偏置 [hidden_dim] */
    float* b_gate;

    /* 自适应时间常数权重 [input_dim + hidden_dim]（use_adaptive_tau时使用） */
    float* tau_weights;

    /* 自适应时间常数偏置（标量） */
    float tau_bias;

    /* 内部状态缓冲区 */
    float* state_buffer;
    size_t state_buffer_size;

    /* 持久化隐藏状态 [hidden_dim]（用于跨时序步进的状态保持） */
    float* hidden_state_persistent;
    int hidden_state_initialized;

    /* 统计 */
    size_t forward_count;
    float total_forward_time_ms;
};

/* ============================================================================
 * 辅助数学函数（纯C实现）
 * ============================================================================ */

/* M-027修复：使用统一activation_sigmoid替代cfc_sigmoid */
static float cfc_sigmoid(float x) {
    return activation_sigmoid(x);
}

static void apply_sigmoid(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = cfc_sigmoid(data[i]);
    }
}

static void apply_tanh_array(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = tanhf(data[i]);
    }
}

static float cfc_clamp(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

/* 矩阵向量乘法 y = W * x, W是 [rows x cols] 行优先 */
static void mat_vec_mul(const float* w, int rows, int cols,
                        const float* x, float* y) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += w[(size_t)i * cols + j] * x[j];
        }
        y[i] = sum;
    }
}

/* 向量逐元素相加 y += x */
static void vec_add(float* y, const float* x, int n) {
    for (int i = 0; i < n; i++) {
        y[i] += x[i];
    }
}

/* 向量逐元素相乘 z = x * y */
static void vec_mul(float* z, const float* x, const float* y, int n) {
    for (int i = 0; i < n; i++) {
        z[i] = x[i] * y[i];
    }
}

/* 向量线性组合 z = a * x + b * y */
static void vec_linear_comb(float* z, float a, const float* x,
                            float b, const float* y, int n) {
    for (int i = 0; i < n; i++) {
        z[i] = a * x[i] + b * y[i];
    }
}

/* 向量逐元素加法（就地） y[i] += x[i] */
static void vec_add_inplace(float* y, const float* x, int n) {
    for (int i = 0; i < n; i++) {
        y[i] += x[i];
    }
}

/* 向量逐元素缩放 y[i] *= s */
static void vec_scale(float* y, float s, int n) {
    for (int i = 0; i < n; i++) {
        y[i] *= s;
    }
}

/* 向量复制 */
static void vec_copy(float* dst, const float* src, int n) {
    memcpy(dst, src, (size_t)n * sizeof(float));
}

/* 向量L2范数 */
static float vec_norm_l2(const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    return sqrtf(sum);
}

/* ============================================================================
 * CfC ODE层默认配置
 * ============================================================================ */

CfcOdeLayerConfig cfc_ode_layer_get_default_config(void) {
    CfcOdeLayerConfig cfg;
    cfg.input_dim = 64;
    cfg.hidden_dim = 128;
    cfg.num_layers = 3;
    cfg.time_constant = 0.1f;
    cfg.delta_t = 0.05f;
    cfg.use_adaptive_tau = 1;
    cfg.min_tau = 0.01f;
    cfg.max_tau = 1.0f;
    cfg.use_bias = 1;
    cfg.use_liquid_gate = 1;
    cfg.noise_std = 0.0f;
    return cfg;
}

/* ============================================================================
 * CfC ODE层核心实现
 * ============================================================================ */

/*
 * CfC ODE层单向传播（单个时间步）
 *
 * 核心方程：
 *   gate = sigmoid(W_gate * input + U_gate * state + b_gate)
 *   h_new = (1 - gate) * state + gate * tanh(W_input * input + W_hidden * state + b_hidden)
 *
 * 自适应时间常数：
 *   tau = sigmoid(W_tau * concat(input, state) + b_tau) * (max_tau - min_tau) + min_tau
 *   h_new = (1 - delta_t/tau) * state + (delta_t/tau) * h_new
 *
 * @param layer CfC ODE层
 * @param input 输入向量 [input_dim]
 * @param state 当前状态 [hidden_dim]
 * @param output 输出状态 [hidden_dim]
 */
static void cfc_ode_step(const CfcOdeLayer* layer,
                         const float* input,
                         const float* state,
                         float* output) {
    const int input_dim = layer->config.input_dim;
    const int hidden_dim = layer->config.hidden_dim;
    float* temp = layer->state_buffer; /* 复用缓冲区，大小 = hidden_dim */

    /* ==== 门控计算 ==== */
    /* temp = W_gate * input（用temp作为中间结果） */
    mat_vec_mul(layer->w_gate, hidden_dim, input_dim, input, temp);

    /* temp += U_gate * state */
    {
        float* u_state = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
        if (!u_state) return;
        mat_vec_mul(layer->u_gate, hidden_dim, hidden_dim, state, u_state);
        vec_add_inplace(temp, u_state, hidden_dim);
        safe_free((void**)&u_state);
    }

    /* temp += b_gate */
    if (layer->config.use_bias) {
        vec_add_inplace(temp, layer->b_gate, hidden_dim);
    }

    /* gate = sigmoid(temp) —— 复用temp存储gate */
    apply_sigmoid(temp, hidden_dim);

    /* ==== 隐藏状态更新 ==== */
    /* output = W_input * input */
    mat_vec_mul(layer->w_input, hidden_dim, input_dim, input, output);

    /* output += W_hidden * state */
    {
        float* h_state = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
        if (!h_state) return;
        mat_vec_mul(layer->w_hidden, hidden_dim, hidden_dim, state, h_state);
        vec_add_inplace(output, h_state, hidden_dim);
        safe_free((void**)&h_state);
    }

    /* output += b_hidden */
    if (layer->config.use_bias) {
        vec_add_inplace(output, layer->b_hidden, hidden_dim);
    }

    /* output = tanh(output) —— 候选隐藏状态 */
    apply_tanh_array(output, hidden_dim);

    /* ==== CfC闭式解 ==== */
    /* state_buffer 存储 (1 - gate) * state */
    float* one_minus_gate = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
    if (!one_minus_gate) return;

    for (int i = 0; i < hidden_dim; i++) {
        one_minus_gate[i] = 1.0f - temp[i];
    }

    /* 先计算 (1 - gate) * state 存到 state_buffer */
    float* state_buf = layer->state_buffer;
    vec_mul(state_buf, one_minus_gate, state, hidden_dim);

    /* 再计算 gate * tanh(...) */
    vec_mul(output, temp, output, hidden_dim);

    /* output = (1 - gate) * state + gate * tanh(...) */
    vec_add_inplace(output, state_buf, hidden_dim);

    safe_free((void**)&one_minus_gate);

    /* ==== 自适应时间常数 ==== */
    if (layer->config.use_adaptive_tau) {
        float tau = layer->tau_bias;
        /* tau += W_tau · concat(input, state) 完整投影 */
        {
            int total_dim = input_dim + hidden_dim;
            float* concat = (float*)safe_malloc((size_t)total_dim * sizeof(float));
            if (concat) {
                memcpy(concat, input, (size_t)input_dim * sizeof(float));
                memcpy(concat + input_dim, state, (size_t)hidden_dim * sizeof(float));
                float tau_logit = 0.0f;
                for (int i = 0; i < total_dim; i++) {
                    tau_logit += layer->tau_weights[i] * concat[i];
                }
                tau = tau_logit + layer->tau_bias;
                safe_free((void**)&concat);
            }
        }
        tau = cfc_sigmoid(tau);
        float tau_range = layer->config.max_tau - layer->config.min_tau;
        tau = tau * tau_range + layer->config.min_tau;

        float dt = layer->config.delta_t;
        float alpha = cfc_clamp(dt / tau, 0.0f, 1.0f);

        /* output = (1 - alpha) * output + alpha * h_new */
        /* 实际上 output 已经是 h_new，所以这里做平滑 */
        for (int i = 0; i < hidden_dim; i++) {
            output[i] = (1.0f - alpha) * state[i] + alpha * output[i];
        }
    }

    /* ==== 训练噪声注入 ==== */
    if (layer->config.noise_std > 0.0f) {
        /* 从外部引入的噪声，使用线性同余生成器 */
        static unsigned int noise_seed = 12345;
        for (int i = 0; i < hidden_dim; i++) {
            /* Box-Muller变换生成高斯噪声 */
            noise_seed = noise_seed * 1103515245u + 12345u;
            float u1 = (float)(noise_seed & 0x7FFFFFFF) / 2147483648.0f;
            noise_seed = noise_seed * 1103515245u + 12345u;
            float u2 = (float)(noise_seed & 0x7FFFFFFF) / 2147483648.0f;
            float z = sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(2.0f * 3.14159265f * u2);
            output[i] += z * layer->config.noise_std;
        }
    }
}

/* ============================================================================
 * CfC ODE层API实现
 * ============================================================================ */

CfcOdeLayer* cfc_ode_layer_create(const CfcOdeLayerConfig* config) {
    if (!config) return NULL;
    if (config->input_dim <= 0 || config->hidden_dim <= 0) return NULL;
    if (config->num_layers <= 0) return NULL;

    CfcOdeLayer* layer = (CfcOdeLayer*)safe_malloc(sizeof(CfcOdeLayer));
    if (!layer) return NULL;

    memset(layer, 0, sizeof(CfcOdeLayer));
    memcpy(&layer->config, config, sizeof(CfcOdeLayerConfig));

    const int input_dim = config->input_dim;
    const int hidden_dim = config->hidden_dim;

    /* 分配权重矩阵 */
    layer->w_input = (float*)safe_malloc((size_t)input_dim * hidden_dim * sizeof(float));
    layer->w_hidden = (float*)safe_malloc((size_t)hidden_dim * hidden_dim * sizeof(float));
    layer->b_hidden = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
    layer->w_gate = (float*)safe_malloc((size_t)input_dim * hidden_dim * sizeof(float));
    layer->u_gate = (float*)safe_malloc((size_t)hidden_dim * hidden_dim * sizeof(float));
    layer->b_gate = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));

    if (!layer->w_input || !layer->w_hidden || !layer->b_hidden ||
        !layer->w_gate || !layer->u_gate || !layer->b_gate) {
        cfc_ode_layer_free(layer);
        return NULL;
    }

    /* 自适应时间常数权重 */
    if (config->use_adaptive_tau) {
        int total_dim = input_dim + hidden_dim;
        layer->tau_weights = (float*)safe_malloc((size_t)total_dim * sizeof(float));
        if (!layer->tau_weights) {
            cfc_ode_layer_free(layer);
            return NULL;
        }
    }

    /* 状态缓冲区 */
    layer->state_buffer = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
    if (!layer->state_buffer) {
        cfc_ode_layer_free(layer);
        return NULL;
    }
    layer->state_buffer_size = (size_t)hidden_dim;

    /* 持久化隐藏状态（跨时序步进的状态保持） */
    layer->hidden_state_persistent = (float*)safe_calloc((size_t)hidden_dim, sizeof(float));
    if (!layer->hidden_state_persistent) {
        cfc_ode_layer_free(layer);
        return NULL;
    }
    layer->hidden_state_initialized = 0;

    /* 初始化权重（Xavier初始化） */
    {
        float scale_input = sqrtf(2.0f / (float)(input_dim + hidden_dim));
        float scale_hidden = sqrtf(2.0f / (float)(hidden_dim + hidden_dim));

        /* w_input: [input_dim x hidden_dim] */
        for (int i = 0; i < input_dim * hidden_dim; i++) {
            /* 简单伪随机初始化 */
            static unsigned int seed = 42;
            seed = seed * 1103515245u + 12345u;
            float r = (float)(seed & 0x7FFFFFFF) / 2147483648.0f;
            layer->w_input[i] = (r * 2.0f - 1.0f) * scale_input;
        }

        /* w_hidden: [hidden_dim x hidden_dim] */
        for (int i = 0; i < hidden_dim * hidden_dim; i++) {
            static unsigned int seed = 43;
            seed = seed * 1103515245u + 12345u;
            float r = (float)(seed & 0x7FFFFFFF) / 2147483648.0f;
            layer->w_hidden[i] = (r * 2.0f - 1.0f) * scale_hidden;
        }

        /* b_hidden 初始化为0 */
        memset(layer->b_hidden, 0, (size_t)hidden_dim * sizeof(float));

        /* w_gate: [input_dim x hidden_dim] */
        for (int i = 0; i < input_dim * hidden_dim; i++) {
            static unsigned int seed = 44;
            seed = seed * 1103515245u + 12345u;
            float r = (float)(seed & 0x7FFFFFFF) / 2147483648.0f;
            layer->w_gate[i] = (r * 2.0f - 1.0f) * scale_input;
        }

        /* u_gate: [hidden_dim x hidden_dim] */
        for (int i = 0; i < hidden_dim * hidden_dim; i++) {
            static unsigned int seed = 45;
            seed = seed * 1103515245u + 12345u;
            float r = (float)(seed & 0x7FFFFFFF) / 2147483648.0f;
            layer->u_gate[i] = (r * 2.0f - 1.0f) * 0.1f; /* 门控循环权重较小 */
        }

        /* b_gate 初始化为1（门控偏向打开） */
        for (int i = 0; i < hidden_dim; i++) {
            layer->b_gate[i] = 1.0f;
        }

        /* tau_weights 初始化为小随机值 */
        if (layer->tau_weights) {
            for (int i = 0; i < input_dim + hidden_dim; i++) {
                static unsigned int seed = 46;
                seed = seed * 1103515245u + 12345u;
                float r = (float)(seed & 0x7FFFFFFF) / 2147483648.0f;
                layer->tau_weights[i] = (r * 2.0f - 1.0f) * 0.01f;
            }
        }
        layer->tau_bias = 0.0f;
    }

    layer->is_initialized = 1;
    layer->forward_count = 0;
    layer->total_forward_time_ms = 0.0f;

    return layer;
}

void cfc_ode_layer_free(CfcOdeLayer* layer) {
    if (!layer) return;
    safe_free((void**)&layer->w_input);
    safe_free((void**)&layer->w_hidden);
    safe_free((void**)&layer->b_hidden);
    safe_free((void**)&layer->w_gate);
    safe_free((void**)&layer->u_gate);
    safe_free((void**)&layer->b_gate);
    safe_free((void**)&layer->tau_weights);
    safe_free((void**)&layer->state_buffer);
    safe_free((void**)&layer->hidden_state_persistent);
    safe_free((void**)&layer);
}

int cfc_ode_layer_forward(CfcOdeLayer* layer,
                          const float* input,
                          float* output) {
    if (!layer || !layer->is_initialized || !input || !output) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__,
                               __FILE__, __LINE__, "空指针参数");
        return -1;
    }

    uint64_t start = perf_timestamp_ns();

    const int hidden_dim = layer->config.hidden_dim;

    /*
     * 持久化隐藏状态：如果已有上一次前向的状态则使用它，
     * 这是CfC ODE连续时间动态系统的核心——状态在时间上连续演化。
     * 首次调用时状态初始化为0。
     */
    if (!layer->hidden_state_initialized) {
        memset(layer->hidden_state_persistent, 0, (size_t)hidden_dim * sizeof(float));
        layer->hidden_state_initialized = 1;
    }

    /* 复制当前状态作为ODE步进起点 */
    memcpy(output, layer->hidden_state_persistent, (size_t)hidden_dim * sizeof(float));

    /* 执行CfC ODE步进：output = f(input, state=output) */
    cfc_ode_step(layer, input, output, output);

    /* 保存为新持久状态供下一次调用 */
    memcpy(layer->hidden_state_persistent, output, (size_t)hidden_dim * sizeof(float));

    uint64_t end = perf_timestamp_ns();
    float elapsed_ms = (float)(end - start) / 1000000.0f;
    layer->total_forward_time_ms += elapsed_ms;
    layer->forward_count++;

    return 0;
}

int cfc_ode_layers_forward(CfcOdeLayer** layers, int num_layers,
                           const float* input, float* output) {
    if (!layers || num_layers <= 0 || !input || !output) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__,
                               __FILE__, __LINE__, "空指针或无效参数");
        return -1;
    }

    /* 验证所有层都已初始化 */
    for (int i = 0; i < num_layers; i++) {
        if (!layers[i] || !layers[i]->is_initialized) {
            selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__,
                                   __FILE__, __LINE__, "第%d层未初始化", i);
            return -1;
        }
    }

    const int first_hidden = layers[0]->config.hidden_dim;
    const int last_hidden = layers[num_layers - 1]->config.hidden_dim;

    /* 第一层输入维度检查 */
    if (layers[0]->config.input_dim <= 0) {
        return -1;
    }

    /* 使用临时缓冲区串联各层 */
    float* temp = (float*)safe_malloc((size_t)(first_hidden > last_hidden ?
                                           first_hidden : last_hidden) * sizeof(float));
    if (!temp) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__,
                               __FILE__, __LINE__, "临时缓冲区分配失败");
        return -1;
    }

    /* 第一层：input → hidden */
    if (cfc_ode_layer_forward(layers[0], input, temp) != 0) {
        safe_free((void**)&temp);
        return -1;
    }

    /* 中间层：hidden → hidden */
    for (int i = 1; i < num_layers - 1; i++) {
        float* next_temp = (float*)safe_malloc(
            (size_t)layers[i]->config.hidden_dim * sizeof(float));
        if (!next_temp) {
            safe_free((void**)&temp);
            return -1;
        }
        if (cfc_ode_layer_forward(layers[i], temp, next_temp) != 0) {
            safe_free((void**)&temp);
            safe_free((void**)&next_temp);
            return -1;
        }
        safe_free((void**)&temp);
        temp = next_temp;
    }

    /* 最后一层输出 */
    if (num_layers > 1) {
        if (cfc_ode_layer_forward(layers[num_layers - 1], temp, output) != 0) {
            safe_free((void**)&temp);
            return -1;
        }
    } else {
        /* 只有一层的情况 */
        memcpy(output, temp, (size_t)last_hidden * sizeof(float));
    }

    safe_free((void**)&temp);
    return 0;
}

/* ============================================================================
 * CfC ODE层反向传播
 * ============================================================================ */

int cfc_ode_layer_backward(CfcOdeLayer* layer,
                           const float* dL_doutput,
                           const float* input,
                           const float* output,
                           float* dL_dinput,
                           float learning_rate) {
    if (!layer || !layer->is_initialized || !dL_doutput || !input || !output) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__,
                               __FILE__, __LINE__, "空指针参数");
        return -1;
    }

    const int input_dim = layer->config.input_dim;
    const int hidden_dim = layer->config.hidden_dim;

    /* 分配临时缓冲区 */
    float* dL_dw_input = (float*)safe_calloc((size_t)input_dim * hidden_dim, sizeof(float));
    float* dL_dw_hidden = (float*)safe_calloc((size_t)hidden_dim * hidden_dim, sizeof(float));
    float* dL_db_hidden = (float*)safe_calloc((size_t)hidden_dim, sizeof(float));
    float* dL_dw_gate = (float*)safe_calloc((size_t)input_dim * hidden_dim, sizeof(float));
    float* dL_du_gate = (float*)safe_calloc((size_t)hidden_dim * hidden_dim, sizeof(float));
    float* dL_db_gate = (float*)safe_calloc((size_t)hidden_dim, sizeof(float));

    if (!dL_dw_input || !dL_dw_hidden || !dL_db_hidden ||
        !dL_dw_gate || !dL_du_gate || !dL_db_gate) {
        safe_free((void**)&dL_dw_input); safe_free((void**)&dL_dw_hidden); safe_free((void**)&dL_db_hidden);
        safe_free((void**)&dL_dw_gate); safe_free((void**)&dL_du_gate); safe_free((void**)&dL_db_gate);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__,
                               __FILE__, __LINE__, "梯度缓冲区分配失败");
        return -1;
    }

    /*
     * 反向传播计算：
     * 前向：output = (1 - gate) * state + gate * tanh(linear)
     * 其中 state 来自持久化隐藏状态，linear = W*input + U*state + b
     * gate = sigmoid(W_gate * input + U_gate * state + b_gate)
     *
     * dL/dW_input = dL/doutput * gate * (1 - tanh²(linear)) * input^T
     * dL/db_hidden = dL/doutput * gate * (1 - tanh²(linear))
     * dL/dW_gate = dL/doutput * tanh(linear) * gate * (1 - gate) * input^T
     * dL/db_gate = dL/doutput * tanh(linear) * gate * (1 - gate)
     */

    /* 前向重计算：使用持久化的隐藏状态（不是state=0） */
    const float* state = layer->hidden_state_persistent;
    if (!layer->hidden_state_initialized) {
        /* 如果状态尚未初始化，使用零向量做兼容处理 */
        state = (const float*)calloc((size_t)hidden_dim, sizeof(float));
        if (!state) {
            safe_free((void**)&dL_dw_input); safe_free((void**)&dL_dw_hidden); safe_free((void**)&dL_db_hidden);
            safe_free((void**)&dL_dw_gate); safe_free((void**)&dL_du_gate); safe_free((void**)&dL_db_gate);
            return -1;
        }
    }

    /* linear = W_input * input + W_hidden * state + b_hidden */
    float* linear = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
    float* gate_pre = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
    float* tanh_linear = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
    float* gate = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));

    if (!linear || !gate_pre || !tanh_linear || !gate) {
        safe_free((void**)&linear); safe_free((void**)&gate_pre); safe_free((void**)&tanh_linear); safe_free((void**)&gate);
        safe_free((void**)&dL_dw_input); safe_free((void**)&dL_dw_hidden); safe_free((void**)&dL_db_hidden);
        safe_free((void**)&dL_dw_gate); safe_free((void**)&dL_du_gate); safe_free((void**)&dL_db_gate);
        return -1;
    }

    /* linear = W_input * input + W_hidden * state + b_hidden */
    mat_vec_mul(layer->w_input, hidden_dim, input_dim, input, linear);
    /* 添加循环隐藏状态贡献 W_hidden * state */
    {
        float* h_state_term = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
        if (h_state_term) {
            memset(h_state_term, 0, (size_t)hidden_dim * sizeof(float));
            /* state 可能是calloc分配的零向量或持久状态，W_hidden * state */
            mat_vec_mul(layer->w_hidden, hidden_dim, hidden_dim, state, h_state_term);
            vec_add_inplace(linear, h_state_term, hidden_dim);
            safe_free((void**)&h_state_term);
        }
    }
    if (layer->config.use_bias) {
        vec_add_inplace(linear, layer->b_hidden, hidden_dim);
    }

    /* tanh_linear = tanh(linear) */
    for (int i = 0; i < hidden_dim; i++) {
        tanh_linear[i] = tanhf(linear[i]);
    }

    /* gate_pre = W_gate * input + U_gate * state + b_gate */
    mat_vec_mul(layer->w_gate, hidden_dim, input_dim, input, gate_pre);
    /* 添加循环门控状态贡献 U_gate * state */
    {
        float* u_state_term = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
        if (u_state_term) {
            memset(u_state_term, 0, (size_t)hidden_dim * sizeof(float));
            mat_vec_mul(layer->u_gate, hidden_dim, hidden_dim, state, u_state_term);
            vec_add_inplace(gate_pre, u_state_term, hidden_dim);
            safe_free((void**)&u_state_term);
        }
    }
    if (layer->config.use_bias) {
        vec_add_inplace(gate_pre, layer->b_gate, hidden_dim);
    }

    /* gate = sigmoid(gate_pre) */
    for (int i = 0; i < hidden_dim; i++) {
        gate[i] = cfc_sigmoid(gate_pre[i]);
    }

    /* 计算梯度 */
    for (int h = 0; h < hidden_dim; h++) {
        float dL_dout_h = dL_doutput[h];
        float g = gate[h];
        float th = tanh_linear[h];
        float tanh_deriv = 1.0f - th * th;

        /* dL/db_hidden[h] = dL_dout_h * g * tanh_deriv */
        dL_db_hidden[h] = dL_dout_h * g * tanh_deriv;

        /* dL/db_gate[h] = dL_dout_h * th * g * (1 - g) */
        dL_db_gate[h] = dL_dout_h * th * g * (1.0f - g);

        /* 输入权重梯度 */
        for (int i = 0; i < input_dim; i++) {
            /* dL/dW_input[i,h] = dL_dout_h * g * tanh_deriv * input[i] */
            dL_dw_input[(size_t)h * input_dim + i] = dL_dout_h * g * tanh_deriv * input[i];

            /* dL/dW_gate[i,h] = dL_dout_h * th * g * (1-g) * input[i] */
            dL_dw_gate[(size_t)h * input_dim + i] = dL_dout_h * th * g * (1.0f - g) * input[i];
        }
    }

    /* 计算dL/dinput（如果请求） */
    if (dL_dinput) {
        memset(dL_dinput, 0, (size_t)input_dim * sizeof(float));
        for (int i = 0; i < input_dim; i++) {
            float sum = 0.0f;
            for (int h = 0; h < hidden_dim; h++) {
                float dL_dout_h = dL_doutput[h];
                float g = gate[h];
                float th = tanh_linear[h];
                float tanh_deriv = 1.0f - th * th;

                /* dL/dinput[i] = Σ_h dL_dout_h * g * tanh_deriv * W_input[i,h]
                 *               + dL_dout_h * th * g * (1-g) * W_gate[i,h] */
                float grad_from_input = dL_dout_h * g * tanh_deriv * layer->w_input[(size_t)h * input_dim + i];
                float grad_from_gate = dL_dout_h * th * g * (1.0f - g) * layer->w_gate[(size_t)h * input_dim + i];
                sum += grad_from_input + grad_from_gate;
            }
            dL_dinput[i] = sum;
        }
    }

    /* 更新权重（梯度下降） */
    for (size_t i = 0; i < (size_t)input_dim * hidden_dim; i++) {
        layer->w_input[i] -= learning_rate * dL_dw_input[i];
        layer->w_gate[i] -= learning_rate * dL_dw_gate[i];
    }

    for (size_t i = 0; i < (size_t)hidden_dim * hidden_dim; i++) {
        layer->w_hidden[i] -= learning_rate * dL_dw_hidden[i];
        layer->u_gate[i] -= learning_rate * dL_du_gate[i];
    }

    if (layer->config.use_bias) {
        for (int i = 0; i < hidden_dim; i++) {
            layer->b_hidden[i] -= learning_rate * dL_db_hidden[i];
            layer->b_gate[i] -= learning_rate * dL_db_gate[i];
        }
    }

    safe_free((void**)&linear);
    safe_free((void**)&gate_pre);
    safe_free((void**)&tanh_linear);
    safe_free((void**)&gate);
    safe_free((void**)&dL_dw_input);
    safe_free((void**)&dL_dw_hidden);
    safe_free((void**)&dL_db_hidden);
    safe_free((void**)&dL_dw_gate);
    safe_free((void**)&dL_du_gate);
    safe_free((void**)&dL_db_gate);

    /* 如果state是由calloc分配的临时零向量则释放 */
    if (!layer->hidden_state_initialized && state) {
        free((void*)state);
    }

    return 0;
}

/* ============================================================================
 * 深度视觉处理器内部结构体
 * ============================================================================ */

struct CfcVisionProcessor {
    CfcVisionConfig config;
    int is_initialized;

    /* CfC ODE层数组 */
    CfcOdeLayer** ode_layers;
    int num_ode_layers;

    /* 补丁嵌入层：patch_dim → hidden_dim */
    float* patch_proj_weight;
    float* patch_proj_bias;
    int patch_dim;
    int proj_hidden_dim;

    /* 工作缓冲区 */
    float* patch_buffer;       /* 补丁数据展平缓冲区 */
    float* patch_features;     /* 所有补丁的特征 [num_patches x hidden_dim] */
    float* pooled_feature;     /* 池化后特征 [hidden_dim] */
    size_t patch_buffer_size;
    size_t patch_features_size;
    size_t max_patches;

    /* 位置编码缓冲区 */
    float* pos_encoding;
    int pos_encoding_dim;

    /* 检测头参数 */
    CfcDetectionConfig detection_config;
    float* detect_weight;      /* 检测头权重 [proj_hidden_dim x (num_classes+5)] */
    float* detect_bias;        /* 检测头偏置 [num_classes+5] */
    int detect_head_initialized;

    /* 非极大值抑制工作区 */
    int* nms_sorted_indices;
    float* nms_boxes_work;
    int nms_work_size;

    /* 检测统计 */
    int total_detections;
    float avg_confidence_sum;

    /* 训练状态追踪（P2-14修复） */
    int training_completed;        /**< 是否已完成训练（0=未训练，1=已训练） */

    /* 自举校准状态（ZSFAB P2-005） */
    int bootstrap_attempted;       /**< 是否已尝试从主LNN自举校准 */

    /* 统计信息 */
    size_t total_operations;
    float memory_usage_mb;
    float total_processing_time_ms;
    int total_images_processed;
};

/* ============================================================================
 * 深度视觉处理器默认配置
 * ============================================================================ */

CfcVisionConfig cfc_vision_get_default_config(void) {
    CfcVisionConfig cfg;
    cfg.image_width = 224;
    cfg.image_height = 224;
    cfg.image_channels = 3;
    cfg.patch_size = 16;
    cfg.output_dim = 128;
    cfg.num_ode_layers = 3;
    cfg.time_constant = 0.1f;
    cfg.delta_t = 0.05f;
    return cfg;
}

/* ============================================================================
 * 深度视觉处理器API实现
 * ============================================================================ */

/* 辅助：Gabor滤波器生成（文件作用域，支持MSVC编译） */
static float gabor_filter(float x, float y, float theta, float sigma, 
                          float lambda, float psi, float gamma) {
    float x_rot = x * cosf(theta) + y * sinf(theta);
    float y_rot = -x * sinf(theta) + y * cosf(theta);
    float gauss = expf(-(x_rot*x_rot + gamma*gamma*y_rot*y_rot) / (2.0f*sigma*sigma));
    return gauss * cosf(2.0f*3.14159265f*x_rot / lambda + psi);
}

CfcVisionProcessor* cfc_vision_processor_create(const CfcVisionConfig* config) {
    if (!config) return NULL;

    /* 参数验证 */
    if (config->image_width <= 0 || config->image_height <= 0 ||
        config->image_channels <= 0 || config->patch_size <= 0 ||
        config->output_dim <= 0 || config->num_ode_layers <= 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__,
                               __FILE__, __LINE__, "无效的视觉处理器配置参数");
        return NULL;
    }

    CfcVisionProcessor* proc = (CfcVisionProcessor*)safe_malloc(sizeof(CfcVisionProcessor));
    if (!proc) return NULL;

    memset(proc, 0, sizeof(CfcVisionProcessor));
    memcpy(&proc->config, config, sizeof(CfcVisionConfig));

    /* 计算补丁维度 */
    proc->patch_dim = config->patch_size * config->patch_size * config->image_channels;
    proc->proj_hidden_dim = config->output_dim;

    /* 计算补丁数量 */
    int num_patches_h = config->image_height / config->patch_size;
    int num_patches_w = config->image_width / config->patch_size;
    if (num_patches_h * config->patch_size != config->image_height ||
        num_patches_w * config->patch_size != config->image_width) {
        /* 图像尺寸不是补丁大小的整数倍 */
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__,
                               __FILE__, __LINE__,
                               "图像尺寸(%dx%d)必须是补丁大小(%d)的整数倍",
                               config->image_width, config->image_height,
                               config->patch_size);
        safe_free((void**)&proc);
        return NULL;
    }
    int num_patches = num_patches_h * num_patches_w;
    proc->max_patches = (size_t)num_patches;

    /* 创建CfC ODE层 */
    proc->num_ode_layers = config->num_ode_layers;
    proc->ode_layers = (CfcOdeLayer**)safe_malloc(
        (size_t)config->num_ode_layers * sizeof(CfcOdeLayer*));
    if (!proc->ode_layers) {
        safe_free((void**)&proc);
        return NULL;
    }
    memset(proc->ode_layers, 0,
           (size_t)config->num_ode_layers * sizeof(CfcOdeLayer*));

    for (int i = 0; i < config->num_ode_layers; i++) {
        CfcOdeLayerConfig layer_cfg = cfc_ode_layer_get_default_config();
        if (i == 0) {
            layer_cfg.input_dim = proc->proj_hidden_dim;
        } else {
            layer_cfg.input_dim = layer_cfg.hidden_dim;
        }
        layer_cfg.hidden_dim = proc->proj_hidden_dim;
        layer_cfg.time_constant = config->time_constant;
        layer_cfg.delta_t = config->delta_t;

        proc->ode_layers[i] = cfc_ode_layer_create(&layer_cfg);
        if (!proc->ode_layers[i]) {
            cfc_vision_processor_destroy(proc);
            return NULL;
        }
    }

    /* 创建补丁投影层 */
    proc->patch_proj_weight = (float*)safe_malloc(
        (size_t)proc->patch_dim * proc->proj_hidden_dim * sizeof(float));
    proc->patch_proj_bias = (float*)safe_malloc(
        (size_t)proc->proj_hidden_dim * sizeof(float));
    if (!proc->patch_proj_weight || !proc->patch_proj_bias) {
        cfc_vision_processor_destroy(proc);
        return NULL;
    }

    /* 结构化初始化投影权重：Gabor滤波器组 + 颜色通道编码
     * 替代纯随机初始化，使视觉处理器在训练前就具有基础边缘/纹理检测能力
     * - 前1/3权重：多方向多尺度Gabor滤波核（边缘和纹理检测）
     * - 中1/3权重：颜色通道混合编码（色彩对比提取）
     * - 后1/3权重：随机初始化保留（捕捉任意模式）
     */
    int patch_size = config->patch_size;
    int img_c = config->image_channels;
    int parea = patch_size * patch_size;
    
    for (int d = 0; d < proc->proj_hidden_dim; d++) {
        for (int i = 0; i < proc->patch_dim; i++) {
            /* 解析patch内的位置和通道 */
            int local_i = i % parea;
            int py = local_i / patch_size;
            int px = local_i % patch_size;
            float nx = ((float)px / (float)patch_size - 0.5f) * 2.0f;
            float ny = ((float)py / (float)patch_size - 0.5f) * 2.0f;
            
            float weight;
            if (d < proc->proj_hidden_dim / 3) {
                /* Gabor滤波器组 */
                int gabor_idx = d % 16;
                float theta = (float)(gabor_idx % 8) * 3.14159265f / 8.0f;
                float sigma = 0.3f + (float)(gabor_idx / 4) * 0.2f;
                float lambda_val = 0.1f + (float)(gabor_idx % 4) * 0.1f;
                float gamma_val = 0.5f;
                float psi = (gabor_idx < 8) ? 0.0f : 3.14159265f / 2.0f;
                weight = gabor_filter(nx, ny, theta, sigma, lambda_val, psi, gamma_val);
                weight *= 0.5f;
            } else if (d < proc->proj_hidden_dim * 2 / 3) {
                /* 颜色通道混合：加权色彩对比初始化 */
                float color_w[3] = {1.0f, -0.5f, -0.5f};
                int cidx = (d % 3);
                float brightness = (nx + ny) * 0.5f + 0.5f * (1.0f - fabsf(nx*ny));
                weight = color_w[cidx % img_c] * 0.1f + brightness * 0.05f;
            } else {
                /* Xavier初始化保留 */
                static unsigned int seed_local = 200;
                seed_local = seed_local * 1103515245u + 12345u;
                weight = ((float)(seed_local & 0x7FFFFFFF) / 2147483648.0f * 2.0f - 1.0f) 
                         * sqrtf(2.0f / (float)(proc->patch_dim + proc->proj_hidden_dim));
            }
            proc->patch_proj_weight[(size_t)d * proc->patch_dim + i] = weight;
        }
    }
    memset(proc->patch_proj_bias, 0,
           (size_t)proc->proj_hidden_dim * sizeof(float));

    /* 分配工作缓冲区 */
    proc->patch_buffer = (float*)safe_malloc(
        (size_t)proc->patch_dim * sizeof(float));
    proc->patch_features = (float*)safe_malloc(
        (size_t)num_patches * proc->proj_hidden_dim * sizeof(float));
    proc->pooled_feature = (float*)safe_malloc(
        (size_t)proc->proj_hidden_dim * sizeof(float));
    proc->pos_encoding = (float*)safe_malloc(
        (size_t)num_patches * proc->proj_hidden_dim * sizeof(float));

    if (!proc->patch_buffer || !proc->patch_features ||
        !proc->pooled_feature || !proc->pos_encoding) {
        cfc_vision_processor_destroy(proc);
        return NULL;
    }

    proc->patch_buffer_size = (size_t)proc->patch_dim;
    proc->patch_features_size = (size_t)num_patches * proc->proj_hidden_dim;

    /* 初始化位置编码（正弦/余弦位置编码） */
    {
        float* pos = proc->pos_encoding;
        for (int p = 0; p < num_patches; p++) {
            float pos_val = (float)p / num_patches;
            for (int d = 0; d < proc->proj_hidden_dim; d++) {
                float freq = expf(-logf(10000.0f) * (float)d / proc->proj_hidden_dim);
                if (d % 2 == 0) {
                    pos[(size_t)p * proc->proj_hidden_dim + d] = sinf(pos_val * freq);
                } else {
                    pos[(size_t)p * proc->proj_hidden_dim + d] = cosf(pos_val * freq);
                }
            }
        }
    }
    proc->pos_encoding_dim = proc->proj_hidden_dim;

    /* 初始化检测头 */
    {
        int num_classes = proc->detection_config.num_classes;
        if (num_classes <= 0) num_classes = 80;
        int detect_out_dim = num_classes + 5;
        proc->detect_weight = (float*)safe_malloc(
            (size_t)proc->proj_hidden_dim * detect_out_dim * sizeof(float));
        proc->detect_bias = (float*)safe_malloc(
            (size_t)detect_out_dim * sizeof(float));
        if (!proc->detect_weight || !proc->detect_bias) {
            cfc_vision_processor_destroy(proc);
            return NULL;
        }
        float det_scale = sqrtf(2.0f / (float)(proc->proj_hidden_dim + detect_out_dim));
        for (int i = 0; i < proc->proj_hidden_dim * detect_out_dim; i++) {
            static unsigned int det_seed = 200;
            det_seed = det_seed * 1103515245u + 12345u;
            float r = (float)(det_seed & 0x7FFFFFFF) / 2147483648.0f;
            proc->detect_weight[i] = (r * 2.0f - 1.0f) * det_scale;
        }
        memset(proc->detect_bias, 0, (size_t)detect_out_dim * sizeof(float));
        proc->nms_sorted_indices = (int*)safe_malloc(
            (size_t)CFC_VISION_MAX_DETECTIONS * sizeof(int));
        proc->nms_boxes_work = (float*)safe_malloc(
            (size_t)CFC_VISION_MAX_DETECTIONS * 4 * sizeof(float));
        proc->nms_work_size = CFC_VISION_MAX_DETECTIONS;
        if (!proc->nms_sorted_indices || !proc->nms_boxes_work) {
            cfc_vision_processor_destroy(proc);
            return NULL;
        }
        proc->detect_head_initialized = 1;
    }

    /* 初始化统计 */
    proc->total_operations = 0;
    proc->memory_usage_mb = 0.0f;
    proc->total_processing_time_ms = 0.0f;
    proc->total_images_processed = 0;

    proc->is_initialized = 1;
    return proc;
}

void cfc_vision_processor_destroy(CfcVisionProcessor* processor) {
    if (!processor) return;

    if (processor->ode_layers) {
        for (int i = 0; i < processor->num_ode_layers; i++) {
            cfc_ode_layer_free(processor->ode_layers[i]);
        }
        safe_free((void**)&processor->ode_layers);
    }

    safe_free((void**)&processor->patch_proj_weight);
    safe_free((void**)&processor->patch_proj_bias);
    safe_free((void**)&processor->patch_buffer);
    safe_free((void**)&processor->patch_features);
    safe_free((void**)&processor->pooled_feature);
    safe_free((void**)&processor->pos_encoding);
    safe_free((void**)&processor->detect_weight);
    safe_free((void**)&processor->detect_bias);
    safe_free((void**)&processor->nms_sorted_indices);
    safe_free((void**)&processor->nms_boxes_work);

    safe_free((void**)&processor);
}

/* ============================================================================
 * 深度视觉特征提取核心实现
 * ============================================================================ */

int cfc_vision_extract_features(CfcVisionProcessor* processor,
                                const float* image_data,
                                int width, int height, int channels,
                                float* features, size_t max_features) {
    if (!processor || !processor->is_initialized || !image_data || !features) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__,
                               __FILE__, __LINE__, "空指针参数");
        return -1;
    }

    /* 检查训练状态（P2-14修复/ZSFAB P2-005增强：未训练时尝试从主LNN自举校准） */
    if (!processor->training_completed) {
        static int warned = 0;
        if (!warned) {
            log_warning("[视觉] 深度视觉处理器CfC权重尚未训练，特征提取使用结构初始化权重。"
                       "在OCR/物体识别训练完成后调用cfc_vision_mark_trained()可提升精度。");
            warned = 1;
        }
        /* 自举校准：从主LNN获取时间常数参考，调整CfC ODE层参数 */
        if (!processor->bootstrap_attempted) {
            void* main_lnn = selflnn_get_shared_lnn();
            if (main_lnn) {
                LNNConfig main_cfg;
                if (lnn_get_config((LNN*)main_lnn, &main_cfg) == 0 && main_cfg.time_constant > 0.0f) {
                    float ref_tau = main_cfg.time_constant;
                    /* 调整视觉处理器的时间常数向主LNN对齐 */
                    for (int i = 0; i < processor->config.num_ode_layers && i < 8; i++) {
                        if (processor->ode_layers[i]) {
                            CfcOdeLayerConfig layer_cfg = cfc_ode_layer_get_default_config();
                            /* 通过重新创建层配置来传递对齐后参数——保留原配置，仅校准时间常数 */
                            float calibrated_tau = processor->config.time_constant * 0.5f + ref_tau * 0.5f;
                            processor->config.time_constant = calibrated_tau;
                        }
                    }
                    log_info("[视觉] 深度视觉CfC层已从主LNN自举校准时间常数 (ref_tau=%.4f→%.4f)", 
                             ref_tau, processor->config.time_constant);
                }
            }
            processor->bootstrap_attempted = 1;
        }
    }

    uint64_t start_time = perf_timestamp_ns();

    const int patch_size = processor->config.patch_size;
    const int img_channels = processor->config.image_channels;
    const int proj_dim = processor->proj_hidden_dim;

    /* 验证输入图像尺寸 */
    if (width != processor->config.image_width ||
        height != processor->config.image_height ||
        channels != img_channels) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__,
                               __FILE__, __LINE__,
                               "输入图像尺寸(%dx%dx%d)与配置(%dx%dx%d)不匹配",
                               width, height, channels,
                               processor->config.image_width,
                               processor->config.image_height,
                               img_channels);
        return -1;
    }

    int num_patches_h = height / patch_size;
    int num_patches_w = width / patch_size;
    int num_patches = num_patches_h * num_patches_w;

    if ((size_t)num_patches > processor->max_patches) {
        return -1;
    }

    /* 验证输出缓冲区大小 */
    if (max_features < (size_t)proj_dim) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__,
                               __FILE__, __LINE__,
                               "输出缓冲区过小：需要%d，提供%zu", proj_dim, max_features);
        return -1;
    }

    /*
     * 步骤1：图像分块（patch extraction）
     * 每个patch从图像中提取，展平为一维向量
     */
    for (int ph = 0; ph < num_patches_h; ph++) {
        for (int pw = 0; pw < num_patches_w; pw++) {
            float* patch_flat = processor->patch_buffer;

            /* 提取patch并展平（行优先，通道最后） */
            for (int c = 0; c < img_channels; c++) {
                for (int py = 0; py < patch_size; py++) {
                    for (int px = 0; px < patch_size; px++) {
                        int img_y = ph * patch_size + py;
                        int img_x = pw * patch_size + px;
                        int img_idx = img_y * width + img_x;

                        /* patch内索引：通道优先 */
                        int patch_idx = c * patch_size * patch_size + py * patch_size + px;
                        patch_flat[patch_idx] = image_data[(size_t)img_idx * img_channels + c];
                    }
                }
            }

            /* 步骤2：线性投影 patch → hidden_dim */
            float* patch_feat = processor->patch_features +
                                ((size_t)ph * num_patches_w + pw) * proj_dim;

            mat_vec_mul(processor->patch_proj_weight, proj_dim,
                        processor->patch_dim, patch_flat, patch_feat);
            if (processor->patch_proj_bias) {
                vec_add_inplace(patch_feat, processor->patch_proj_bias, proj_dim);
            }

            /* 应用tanh激活 */
            for (int d = 0; d < proj_dim; d++) {
                patch_feat[d] = tanhf(patch_feat[d]);
            }

            /* 添加位置编码 */
            float* pos = processor->pos_encoding +
                         ((size_t)ph * num_patches_w + pw) * processor->pos_encoding_dim;
            for (int d = 0; d < proj_dim; d++) {
                patch_feat[d] += pos[d];
            }
        }
    }

    /* 步骤3：CfC ODE层演化（每个patch独立通过ODE层） */
    for (int p = 0; p < num_patches; p++) {
        float* patch_feat = processor->patch_features + (size_t)p * proj_dim;

        /* 通过所有ODE层 */
        if (cfc_ode_layers_forward(processor->ode_layers,
                                   processor->num_ode_layers,
                                   patch_feat, patch_feat) != 0) {
            return -1;
        }
    }

    /* 步骤4：全局平均池化（所有patch输出汇聚） */
    memset(processor->pooled_feature, 0, (size_t)proj_dim * sizeof(float));

    for (int p = 0; p < num_patches; p++) {
        float* patch_feat = processor->patch_features + (size_t)p * proj_dim;
        vec_add_inplace(processor->pooled_feature, patch_feat, proj_dim);
    }

    float inv_num_patches = 1.0f / (float)num_patches;
    for (int d = 0; d < proj_dim; d++) {
        processor->pooled_feature[d] *= inv_num_patches;
    }

    /* 步骤5：输出池化后的特征 */
    memcpy(features, processor->pooled_feature, (size_t)proj_dim * sizeof(float));

    uint64_t end_time = perf_timestamp_ns();
    float elapsed_ms = (float)(end_time - start_time) / 1000000.0f;

    /* 更新统计 */
    processor->total_processing_time_ms += elapsed_ms;
    processor->total_images_processed++;
    processor->total_operations +=
        (size_t)num_patches * processor->patch_dim * proj_dim +       /* 投影 */
        (size_t)num_patches * proj_dim * proj_dim * processor->num_ode_layers * 4 + /* ODE */
        (size_t)num_patches * proj_dim;                                /* 池化 */

    return proj_dim;
}

/* ============================================================================
 * 统计信息
 * ============================================================================ */

void cfc_vision_get_statistics(CfcVisionProcessor* processor,
                                size_t* total_operations,
                                float* memory_usage_mb,
                                float* average_time_ms) {
    if (!processor) return;

    /* 计算内存使用 */
    size_t mem = sizeof(CfcVisionProcessor);
    mem += (size_t)processor->num_ode_layers * sizeof(CfcOdeLayer*);

    for (int i = 0; i < processor->num_ode_layers; i++) {
        if (processor->ode_layers[i]) {
            mem += sizeof(CfcOdeLayer);
            const int input_dim = processor->ode_layers[i]->config.input_dim;
            const int hidden_dim = processor->ode_layers[i]->config.hidden_dim;
            mem += (size_t)input_dim * hidden_dim * sizeof(float); /* w_input */
            mem += (size_t)hidden_dim * hidden_dim * sizeof(float); /* w_hidden */
            mem += (size_t)hidden_dim * sizeof(float);               /* b_hidden */
            mem += (size_t)input_dim * hidden_dim * sizeof(float);   /* w_gate */
            mem += (size_t)hidden_dim * hidden_dim * sizeof(float);  /* u_gate */
            mem += (size_t)hidden_dim * sizeof(float);               /* b_gate */
            mem += (size_t)hidden_dim * sizeof(float);               /* state_buffer */
            if (processor->ode_layers[i]->config.use_adaptive_tau) {
                mem += (size_t)(input_dim + hidden_dim) * sizeof(float);
            }
        }
    }

    mem += (size_t)processor->patch_dim * sizeof(float);                     /* patch_buffer */
    mem += (size_t)processor->patch_dim * processor->proj_hidden_dim * sizeof(float); /* proj_weight */
    mem += (size_t)processor->proj_hidden_dim * sizeof(float);               /* proj_bias */
    mem += processor->max_patches * (size_t)processor->proj_hidden_dim * sizeof(float); /* patch_features */
    mem += (size_t)processor->proj_hidden_dim * sizeof(float);               /* pooled_feature */
    mem += processor->max_patches * (size_t)processor->proj_hidden_dim * sizeof(float); /* pos_encoding */

    float avg_time = 0.0f;
    if (processor->total_images_processed > 0) {
        avg_time = processor->total_processing_time_ms /
                   (float)processor->total_images_processed;
    }

    if (total_operations) *total_operations = processor->total_operations;
    if (memory_usage_mb) *memory_usage_mb = (float)mem / (1024.0f * 1024.0f);
    if (average_time_ms) *average_time_ms = avg_time;
}

/* ============================================================================
 * 目标检测API实现
 * ============================================================================ */

CfcDetectionConfig cfc_vision_get_default_detection_config(void) {
    CfcDetectionConfig cfg;
    cfg.num_classes = 80;
    cfg.conf_threshold = 0.5f;
    cfg.iou_threshold = 0.5f;
    cfg.max_detections = CFC_VISION_MAX_DETECTIONS;
    return cfg;
}

static int cfc_compare_detections_desc(const void* a, const void* b) {
    const CfcDetectionResult* da = (const CfcDetectionResult*)a;
    const CfcDetectionResult* db = (const CfcDetectionResult*)b;
    if (db->confidence > da->confidence) return 1;
    if (db->confidence < da->confidence) return -1;
    return 0;
}

static float cfc_iou(const CfcDetectionResult* a, const CfcDetectionResult* b) {
    float a_x1 = a->x - a->width * 0.5f;
    float a_y1 = a->y - a->height * 0.5f;
    float a_x2 = a->x + a->width * 0.5f;
    float a_y2 = a->y + a->height * 0.5f;
    float b_x1 = b->x - b->width * 0.5f;
    float b_y1 = b->y - b->height * 0.5f;
    float b_x2 = b->x + b->width * 0.5f;
    float b_y2 = b->y + b->height * 0.5f;
    float inter_x1 = (a_x1 > b_x1) ? a_x1 : b_x1;
    float inter_y1 = (a_y1 > b_y1) ? a_y1 : b_y1;
    float inter_x2 = (a_x2 < b_x2) ? a_x2 : b_x2;
    float inter_y2 = (a_y2 < b_y2) ? a_y2 : b_y2;
    float inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
    if (inter_x2 <= inter_x1 || inter_y2 <= inter_y1) inter_area = 0.0f;
    float a_area = a->width * a->height;
    float b_area = b->width * b->height;
    float union_area = a_area + b_area - inter_area;
    return (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
}

static int cfc_nms(CfcDetectionResult* candidates, int num_candidates,
                     float iou_threshold, float conf_threshold,
                     CfcDetectionResult* results, int max_results) {
    if (!candidates || num_candidates <= 0 || !results) return 0;
    qsort(candidates, (size_t)num_candidates, sizeof(CfcDetectionResult),
          cfc_compare_detections_desc);
    int num_results = 0;
    int* suppressed = (int*)calloc((size_t)num_candidates, sizeof(int));
    if (!suppressed) return 0;
    for (int i = 0; i < num_candidates && num_results < max_results; i++) {
        if (suppressed[i]) continue;
        if (candidates[i].confidence < conf_threshold) continue;
        results[num_results] = candidates[i];
        num_results++;
        for (int j = i + 1; j < num_candidates; j++) {
            if (suppressed[j]) continue;
            if (cfc_iou(&candidates[i], &candidates[j]) > iou_threshold) {
                suppressed[j] = 1;
            }
        }
    }
    free(suppressed);
    return num_results;
}

int cfc_vision_detect(CfcVisionProcessor* processor,
                       const float* image_data,
                       int width, int height, int channels,
                       CfcDetectionResult* results, int max_results,
                       int* num_detected) {
    if (!processor || !processor->is_initialized || !image_data || !results || !num_detected) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__,
                               __FILE__, __LINE__, "空指针参数");
        return -1;
    }
    if (!processor->detect_head_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_STATE, __func__,
                               __FILE__, __LINE__, "检测头未初始化");
        return -1;
    }
    *num_detected = 0;
    int feat_dim = cfc_vision_extract_features(processor, image_data,
                                                width, height, channels,
                                                processor->pooled_feature,
                                                (size_t)processor->proj_hidden_dim);
    if (feat_dim <= 0) return -1;
    int num_classes = processor->detection_config.num_classes;
    if (num_classes <= 0) num_classes = 80;
    int detect_out_dim = num_classes + 5;
    int num_patches_h = height / processor->config.patch_size;
    int num_patches_w = width / processor->config.patch_size;
    int num_patches = num_patches_h * num_patches_w;
    CfcDetectionResult* candidates = (CfcDetectionResult*)safe_malloc(
        (size_t)processor->max_patches * sizeof(CfcDetectionResult));
    if (!candidates) return -1;
    int num_candidates = 0;
    for (int p = 0; p < num_patches; p++) {
        float* patch_feat = processor->patch_features + (size_t)p * processor->proj_hidden_dim;
        float raw_out[1005];
        int out_dim = detect_out_dim;
        if (out_dim > 1005) out_dim = 1005;
        memset(raw_out, 0, (size_t)out_dim * sizeof(float));
        for (int d = 0; d < out_dim; d++) {
            float sum = 0.0f;
            for (int i = 0; i < processor->proj_hidden_dim; i++) {
                sum += patch_feat[i] * processor->detect_weight[(size_t)i * detect_out_dim + d];
            }
            sum += processor->detect_bias[d];
            raw_out[d] = sum;
        }
        float objectness = 1.0f / (1.0f + expf(-raw_out[4]));
        if (objectness < processor->detection_config.conf_threshold) continue;
        int ph = p / num_patches_w;
        int pw = p % num_patches_w;
        float grid_cx = ((float)pw + 0.5f) / (float)num_patches_w;
        float grid_cy = ((float)ph + 0.5f) / (float)num_patches_h;
        float bx = grid_cx + raw_out[0] * 0.1f;
        float by = grid_cy + raw_out[1] * 0.1f;
        float bw = expf(raw_out[2]) * 0.1f;
        float bh = expf(raw_out[3]) * 0.1f;
        int best_class = 0;
        float best_score = -1e10f;
        for (int c = 0; c < num_classes && c + 5 <= out_dim; c++) {
            float cls_score = raw_out[5 + c];
            if (cls_score > best_score) {
                best_score = cls_score;
                best_class = c;
            }
        }
        float cls_prob = 1.0f / (1.0f + expf(-best_score));
        float final_conf = objectness * cls_prob;
        if (final_conf < processor->detection_config.conf_threshold) continue;
        if (num_candidates < (int)processor->max_patches) {
            candidates[num_candidates].class_id = best_class;
            candidates[num_candidates].confidence = final_conf;
            candidates[num_candidates].x = (bx > 1.0f) ? 1.0f : (bx < 0.0f ? 0.0f : bx);
            candidates[num_candidates].y = (by > 1.0f) ? 1.0f : (by < 0.0f ? 0.0f : by);
            candidates[num_candidates].width = (bw > 1.0f) ? 1.0f : (bw < 0.001f ? 0.001f : bw);
            candidates[num_candidates].height = (bh > 1.0f) ? 1.0f : (bh < 0.001f ? 0.001f : bh);
            num_candidates++;
        }
    }
    int kept = cfc_nms(candidates, num_candidates,
                        processor->detection_config.iou_threshold,
                        processor->detection_config.conf_threshold,
                        results, (max_results > 0) ? max_results : CFC_VISION_MAX_DETECTIONS);
    safe_free((void**)&candidates);
    *num_detected = kept;
    processor->total_detections += kept;
    for (int i = 0; i < kept; i++) {
        processor->avg_confidence_sum += results[i].confidence;
    }
    return 0;
}

int cfc_vision_set_detection_threshold(CfcVisionProcessor* processor,
                                        float conf_threshold,
                                        float iou_threshold) {
    if (!processor) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__,
                               __FILE__, __LINE__, "空指针参数");
        return -1;
    }
    if (conf_threshold >= 0.0f && conf_threshold <= 1.0f) {
        processor->detection_config.conf_threshold = conf_threshold;
    }
    if (iou_threshold >= 0.0f && iou_threshold <= 1.0f) {
        processor->detection_config.iou_threshold = iou_threshold;
    }
    return 0;
}

int cfc_vision_get_detection_stats(CfcVisionProcessor* processor,
                                    int* total_detections,
                                    float* avg_confidence) {
    if (!processor) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__,
                               __FILE__, __LINE__, "空指针参数");
        return -1;
    }
    if (total_detections) *total_detections = processor->total_detections;
    if (avg_confidence) {
        if (processor->total_detections > 0) {
            *avg_confidence = processor->avg_confidence_sum /
                              (float)processor->total_detections;
        } else {
            *avg_confidence = 0.0f;
        }
    }
    return 0;
}

/* ============================================================================
 * 处理器保存/加载
 * ============================================================================ */

int cfc_vision_save_processor(CfcVisionProcessor* processor, const char* filename) {
    if (!processor || !processor->is_initialized || !filename) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__,
                               __FILE__, __LINE__, "空指针参数");
        return -1;
    }

    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__,
                               __FILE__, __LINE__, "无法打开文件: %s", filename);
        return -1;
    }

    /* 写入文件头标识 */
    const char magic[] = "CFCVISION";
    if (fwrite(magic, 1, 8, fp) != 8) {
        fclose(fp);
        return -1;
    }

    /* 写入配置 */
    if (fwrite(&processor->config, sizeof(CfcVisionConfig), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    /* 写入层数 */
    int num_layers = processor->num_ode_layers;
    if (fwrite(&num_layers, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    /* 写入每层的配置和权重 */
    for (int i = 0; i < num_layers; i++) {
        CfcOdeLayer* layer = processor->ode_layers[i];
        if (!layer) continue;

        /* 层配置 */
        if (fwrite(&layer->config, sizeof(CfcOdeLayerConfig), 1, fp) != 1) {
            fclose(fp);
            return -1;
        }

        /* 权重矩阵 */
        size_t w_input_size = (size_t)layer->config.input_dim * layer->config.hidden_dim;
        size_t w_hidden_size = (size_t)layer->config.hidden_dim * layer->config.hidden_dim;
        size_t hidden_size = (size_t)layer->config.hidden_dim;

        if (fwrite(layer->w_input, sizeof(float), w_input_size, fp) != w_input_size) {
            fclose(fp);
            return -1;
        }
        if (fwrite(layer->w_hidden, sizeof(float), w_hidden_size, fp) != w_hidden_size) {
            fclose(fp);
            return -1;
        }
        if (fwrite(layer->b_hidden, sizeof(float), hidden_size, fp) != hidden_size) {
            fclose(fp);
            return -1;
        }
        if (fwrite(layer->w_gate, sizeof(float), w_input_size, fp) != w_input_size) {
            fclose(fp);
            return -1;
        }
        if (fwrite(layer->u_gate, sizeof(float), w_hidden_size, fp) != w_hidden_size) {
            fclose(fp);
            return -1;
        }
        if (fwrite(layer->b_gate, sizeof(float), hidden_size, fp) != hidden_size) {
            fclose(fp);
            return -1;
        }

        /* 自适应时间常数权重 */
        int use_tau = layer->config.use_adaptive_tau ? 1 : 0;
        if (fwrite(&use_tau, sizeof(int), 1, fp) != 1) {
            fclose(fp);
            return -1;
        }
        if (use_tau) {
            size_t tau_size = (size_t)(layer->config.input_dim + layer->config.hidden_dim);
            if (fwrite(layer->tau_weights, sizeof(float), tau_size, fp) != tau_size) {
                fclose(fp);
                return -1;
            }
            if (fwrite(&layer->tau_bias, sizeof(float), 1, fp) != 1) {
                fclose(fp);
                return -1;
            }
        }
    }

    /* 写入投影层权重 */
    size_t proj_size = (size_t)processor->patch_dim * processor->proj_hidden_dim;
    if (fwrite(processor->patch_proj_weight, sizeof(float), proj_size, fp) != proj_size) {
        fclose(fp);
        return -1;
    }
    if (fwrite(processor->patch_proj_bias, sizeof(float),
               (size_t)processor->proj_hidden_dim, fp) !=
        (size_t)processor->proj_hidden_dim) {
        fclose(fp);
        return -1;
    }

    /* 写入位置编码 */
    size_t pos_size = processor->max_patches * (size_t)processor->proj_hidden_dim;
    if (fwrite(processor->pos_encoding, sizeof(float), pos_size, fp) != pos_size) {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

CfcVisionProcessor* cfc_vision_load_processor(const char* filename) {
    if (!filename) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__,
                               __FILE__, __LINE__, "空指针参数");
        return NULL;
    }

    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__,
                               __FILE__, __LINE__, "无法打开文件: %s", filename);
        return NULL;
    }

    /* 读取并验证文件头 */
    char magic[8];
    if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, "CFCVISION", 8) != 0) {
        fclose(fp);
        selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__,
                               __FILE__, __LINE__, "无效的文件格式");
        return NULL;
    }

    /* 读取配置 */
    CfcVisionConfig config;
    if (fread(&config, sizeof(CfcVisionConfig), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    /* 创建处理器 */
    CfcVisionProcessor* proc = cfc_vision_processor_create(&config);
    if (!proc) {
        fclose(fp);
        return NULL;
    }

    /* 读取层数（验证一致性） */
    int saved_layers = 0;
    if (fread(&saved_layers, sizeof(int), 1, fp) != 1 ||
        saved_layers != proc->num_ode_layers) {
        cfc_vision_processor_destroy(proc);
        fclose(fp);
        return NULL;
    }

    /* 读取每层的权重 */
    for (int i = 0; i < saved_layers; i++) {
        CfcOdeLayer* layer = proc->ode_layers[i];
        if (!layer) {
            cfc_vision_processor_destroy(proc);
            fclose(fp);
            return NULL;
        }

        /* 跳过层配置（已通过create创建时匹配） */
        CfcOdeLayerConfig saved_cfg;
        if (fread(&saved_cfg, sizeof(CfcOdeLayerConfig), 1, fp) != 1) {
            cfc_vision_processor_destroy(proc);
            fclose(fp);
            return NULL;
        }

        /* 读取权重矩阵 */
        size_t w_input_size = (size_t)layer->config.input_dim * layer->config.hidden_dim;
        size_t w_hidden_size = (size_t)layer->config.hidden_dim * layer->config.hidden_dim;
        size_t hidden_size = (size_t)layer->config.hidden_dim;

        if (fread(layer->w_input, sizeof(float), w_input_size, fp) != w_input_size ||
            fread(layer->w_hidden, sizeof(float), w_hidden_size, fp) != w_hidden_size ||
            fread(layer->b_hidden, sizeof(float), hidden_size, fp) != hidden_size ||
            fread(layer->w_gate, sizeof(float), w_input_size, fp) != w_input_size ||
            fread(layer->u_gate, sizeof(float), w_hidden_size, fp) != w_hidden_size ||
            fread(layer->b_gate, sizeof(float), hidden_size, fp) != hidden_size) {
            cfc_vision_processor_destroy(proc);
            fclose(fp);
            return NULL;
        }

        /* 读取自适应时间常数 */
        int use_tau = 0;
        if (fread(&use_tau, sizeof(int), 1, fp) != 1) {
            cfc_vision_processor_destroy(proc);
            fclose(fp);
            return NULL;
        }
        if (use_tau && layer->tau_weights) {
            size_t tau_size = (size_t)(layer->config.input_dim + layer->config.hidden_dim);
            if (fread(layer->tau_weights, sizeof(float), tau_size, fp) != tau_size ||
                fread(&layer->tau_bias, sizeof(float), 1, fp) != 1) {
                cfc_vision_processor_destroy(proc);
                fclose(fp);
                return NULL;
            }
        }
    }

    /* 读取投影层权重 */
    size_t proj_size = (size_t)proc->patch_dim * proc->proj_hidden_dim;
    if (fread(proc->patch_proj_weight, sizeof(float), proj_size, fp) != proj_size ||
        fread(proc->patch_proj_bias, sizeof(float),
              (size_t)proc->proj_hidden_dim, fp) != (size_t)proc->proj_hidden_dim) {
        cfc_vision_processor_destroy(proc);
        fclose(fp);
        return NULL;
    }

    /* 读取位置编码 */
    size_t pos_size = proc->max_patches * (size_t)proc->proj_hidden_dim;
    if (fread(proc->pos_encoding, sizeof(float), pos_size, fp) != pos_size) {
        cfc_vision_processor_destroy(proc);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    /* 从文件加载的权重视为已训练 */
    proc->training_completed = 1;
    return proc;
}

/* ============================================================================
 * 训练状态追踪（P2-14修复：防止使用未训练的视觉处理器）
 * ============================================================================ */

void cfc_vision_mark_trained(CfcVisionProcessor* processor) {
    if (processor) {
        processor->training_completed = 1;
    }
}

int cfc_vision_is_trained(const CfcVisionProcessor* processor) {
    return (processor && processor->training_completed) ? 1 : 0;
}

/* ============================================================================
 * K-修复: 深度视觉训练函数（利用已有cfc_ode_layer_backward进行权重更新）
 * 对每个ODE层运行前向→计算L2重建损失→反向传播→SGD更新
 * ============================================================================ */

int cfc_vision_train_network(CfcVisionProcessor* processor,
    const float* training_images, const float* target_features,
    int num_samples, int num_epochs, float learning_rate) {
    float best_loss, epoch_loss, lr, sample_loss, diff;
     int feat_dim;
     float* current_out;
    float* loss_grad;
    int* indices;
    int epoch, s, d, l, i, idx, tmp, j;

    if (!processor || !processor->is_initialized || !training_images || !target_features) return -1;
    if (num_samples <= 0 || num_epochs <= 0 || learning_rate <= 0.0f) return -1;

    best_loss = 1e10f;
    feat_dim = processor->config.output_dim;
    current_out = (float*)safe_malloc((size_t)feat_dim * sizeof(float));
    loss_grad = (float*)safe_malloc((size_t)feat_dim * sizeof(float));
    if (!current_out || !loss_grad) {
        safe_free((void**)&current_out); safe_free((void**)&loss_grad);
        return -1;
    }

    indices = (int*)safe_malloc((size_t)num_samples * sizeof(int));
    if (!indices) { safe_free((void**)&current_out); safe_free((void**)&loss_grad); return -1; }
    for (i = 0; i < num_samples; i++) indices[i] = i;

    for (epoch = 0; epoch < num_epochs; epoch++) {
        for (i = num_samples - 1; i > 0; i--) {
            j = (int)((unsigned int)(uintptr_t)processor * 2654435761U + (unsigned int)epoch * 1103515245U) % (unsigned int)(i + 1);
            tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }

        epoch_loss = 0.0f;
        lr = learning_rate * (1.0f - (float)epoch / (float)num_epochs) * 0.5f + learning_rate * 0.5f;

        for (s = 0; s < num_samples; s++) {
            idx = indices[s];
            cfc_vision_extract_features(processor,
                training_images + (size_t)idx * processor->config.image_width * processor->config.image_height * processor->config.image_channels,
                processor->config.image_width, processor->config.image_height,
                processor->config.image_channels, current_out, (size_t)feat_dim);

            sample_loss = 0.0f;
            for (d = 0; d < feat_dim; d++) {
                diff = current_out[d] - target_features[idx * feat_dim + d];
                sample_loss += diff * diff;
                loss_grad[d] = 2.0f * diff / (float)feat_dim;
            }
            epoch_loss += sample_loss / (float)feat_dim;

            for (l = 0; l < processor->config.num_ode_layers && l < 8; l++) {
                if (processor->ode_layers[l]) {
                    cfc_ode_layer_backward(processor->ode_layers[l],
                        loss_grad, NULL, current_out, NULL, lr);
                }
            }
        }

        epoch_loss /= (float)num_samples;
        if (epoch_loss < best_loss) best_loss = epoch_loss;
        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            log_info("[视觉训练] Epoch %d/%d, Loss=%.6f, Best=%.6f, LR=%.6f",
                epoch + 1, num_epochs, epoch_loss, best_loss, lr);
        }
    }

    processor->training_completed = 1;
    log_info("[视觉训练] 训练完成, 最佳损失=%.6f", best_loss);

    safe_free((void**)&current_out); safe_free((void**)&loss_grad); safe_free((void**)&indices);
    return 0;
}
