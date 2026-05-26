/**
 * @file cfc_network.c
 * @brief CfC网络实现（对应头文件 cfc_network.h）
 * 
 * 由多个CfC单元组成的完整网络，实现液态神经网络的前向传播和反向传播。
 */

#define SELFLNN_IMPLEMENTATION
#define SELFLNN_CORE_INTERNAL
#include "selflnn/core/cfc_network.h"
#include "selflnn/core/cfc_cell.h"
#include "selflnn/core/ode_solvers.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/lnn_layer_norm.h"
#include "selflnn/utils/math_utils.h"
#include <stdint.h>
#include "selflnn/utils/memory_utils.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/**
 * @brief 创建CfC网络实例
 */
CfCNetwork* cfc_create(const CfCNetworkConfig* config) {
    if (!config) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "CfC网络配置指针为空");
        return NULL;
    }
    
    // 验证配置
    if (config->input_size == 0 || config->hidden_size == 0 || 
        config->output_size == 0 || config->num_layers == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_NETWORK_CONFIG, __func__, __FILE__, __LINE__,
                              "CfC网络配置无效: input_size=%zu, hidden_size=%zu, output_size=%zu, num_layers=%d",
                              config->input_size, config->hidden_size, config->output_size, config->num_layers);
        return NULL;
    }
    
    // 分配网络结构
    CfCNetwork* network = (CfCNetwork*)safe_malloc(sizeof(CfCNetwork));
    if (!network) {
        return NULL;
    }
    
    // 复制配置
    memcpy(&network->config, config, sizeof(CfCNetworkConfig));

    /* P0-001: 层归一化默认启用 */
    if (network->config.use_layer_norm == 0 && network->config.num_layers >= 1) {
        network->config.use_layer_norm = 1;
    }

    // 分配层数组
    network->layers = (CfCCell**)safe_calloc(config->num_layers, sizeof(CfCCell*));
    if (!network->layers) {
        safe_free((void**)&network);
        return NULL;
    }
    
    // 为每层创建CfC单元
    CfCCellConfig cell_config = {0};
    cell_config.input_size = config->input_size;
    cell_config.hidden_size = config->hidden_size;
    
    // 使用配置中的时间常数，如果未提供则使用合理默认值
    // 注意：CfC单元支持自适应时间常数，因此初始值不是关键
    cell_config.time_constant = (config->time_constant > 0.0f) ? config->time_constant : 0.1f;
    cell_config.noise_std = config->noise_std;
    cell_config.enable_adaptation = config->enable_adaptation;
    cell_config.ode_solver_type = config->ode_solver_type;
    cell_config.use_auto_solver = 1;
    cell_config.delta_t = 1.0f;  // 默认时间步长1.0秒
    
    // 第一层的输入大小是网络的输入大小
    // 后续层的输入大小是前一层的隐藏大小
    
    for (int i = 0; i < config->num_layers; i++) {
        if (i > 0) {
            cell_config.input_size = config->hidden_size;
        }
        
        network->layers[i] = cfc_cell_create(&cell_config);
        if (!network->layers[i]) {
            // 清理已创建的资源
            for (int j = 0; j < i; j++) {
                cfc_cell_free(network->layers[j]);
            }
            safe_free((void**)&network->layers);
            safe_free((void**)&network);
            return NULL;
        }
        /* 设置ODE求解器类型 */
        if (config->ode_solver_type > 0) {
            cfc_cell_set_solver_type(network->layers[i], config->ode_solver_type);
        }
    }
    
    // 计算缓冲区大小
    size_t max_layer_size = (config->input_size > config->hidden_size) ? 
                           config->input_size : config->hidden_size;
    max_layer_size = (max_layer_size > config->output_size) ? 
                    max_layer_size : config->output_size;
    
    // 分配缓冲区
    size_t total_layers = config->num_layers;
    network->layer_outputs = (float*)safe_calloc(total_layers * max_layer_size, sizeof(float));
    network->layer_gradients = (float*)safe_calloc(total_layers * max_layer_size, sizeof(float));
    network->activation_buffer = (float*)safe_calloc(max_layer_size, sizeof(float));
    network->dropout_mask = (float*)safe_calloc(max_layer_size, sizeof(float));
    
    /* ZSFWS-MLW: 多层权重独立存储——每层分配独立权重矩阵
     * 单层: param = [input*hidden w]  [hidden b]  [W_out] [b_out]
     * 多层: param = [layer0: input*hidden] [layer1..N-1: each hidden*hidden] [N*hidden b] [W_out] [b_out]
     * 梯度块使用相同布局 */
    int num_layers_int = config->num_layers;
    size_t input_size_cfg = config->input_size;
    size_t hidden_size_cfg = config->hidden_size;
    size_t output_size_cfg = config->output_size;

    /* 分配每层偏移追踪数组 */
    network->per_layer_w_offset = (size_t*)safe_calloc((size_t)num_layers_int, sizeof(size_t));
    network->per_layer_b_offset = (size_t*)safe_calloc((size_t)num_layers_int, sizeof(size_t));
    network->per_layer_w_size   = (size_t*)safe_calloc((size_t)num_layers_int, sizeof(size_t));

    /* 计算每层权重和偏置大小 */
    size_t total_w = 0, total_b = 0;
    for (int l = 0; l < num_layers_int; l++) {
        size_t l_input = (l == 0) ? input_size_cfg : hidden_size_cfg;
        size_t w_sz = l_input * hidden_size_cfg;
        network->per_layer_w_offset[l] = total_w;
        network->per_layer_w_size[l] = w_sz;
        network->per_layer_b_offset[l] = total_b;
        total_w += w_sz;
        total_b += hidden_size_cfg;
    }
    network->total_weight_params = total_w;
    network->total_bias_params = total_b;

    /* 输出投影参数 */
    size_t out_proj_param_size = (output_size_cfg != hidden_size_cfg) ? 
        (output_size_cfg * hidden_size_cfg + output_size_cfg) : 0;
    size_t total_param_size = total_w + total_b + out_proj_param_size;
    float* param_block = (float*)safe_calloc(total_param_size, sizeof(float));
    network->weight_matrix = param_block;               /* 指向层0权重起始 */
    network->bias_vector = param_block ? param_block + total_w : NULL; /* 指向偏置区域起始 */

    if (out_proj_param_size > 0 && param_block) {
        network->W_out_params = param_block + total_w + total_b;
        network->b_out_params = network->W_out_params + output_size_cfg * hidden_size_cfg;
    } else {
        network->W_out_params = NULL;
        network->b_out_params = NULL;
    }

    /* 梯度块——与参数块相同布局，完全分离 */
    size_t out_proj_grad_size = out_proj_param_size;
    size_t total_grad_size = total_w + total_b + out_proj_grad_size;
    float* grad_block = (float*)safe_calloc(total_grad_size, sizeof(float));
    network->weight_gradients = grad_block;
    network->bias_gradients = grad_block ? grad_block + total_w : NULL;
    if (out_proj_grad_size > 0 && grad_block) {
        network->W_out_gradients = grad_block + total_w + total_b;
        network->b_out_gradients = network->W_out_gradients + output_size_cfg * hidden_size_cfg;
    } else {
        network->W_out_gradients = NULL;
        network->b_out_gradients = NULL;
    }

    /* P0-014修复: He/Kaiming自适应初始化W_out参数 */
    if (out_proj_param_size > 0 && network->W_out_params) {
        float he_limit = sqrtf(6.0f / (float)hidden_size_cfg);
        if (he_limit > 0.2f) he_limit = 0.2f;
        if (he_limit < 0.001f) he_limit = 0.001f;
        for (size_t i = 0; i < output_size_cfg * hidden_size_cfg; i++) {
            network->W_out_params[i] = rng_uniform(-he_limit, he_limit);
        }
        for (size_t i = 0; i < output_size_cfg; i++) {
            network->b_out_params[i] = 0.0f;
        }
    }
    
    // 检查内存分配
    if (!network->layer_outputs || !network->layer_gradients ||
        !network->activation_buffer || !network->dropout_mask ||
        !param_block || !grad_block ||
        !network->per_layer_w_offset || !network->per_layer_b_offset ||
        !network->per_layer_w_size) {
        cfc_free(network);
        return NULL;
    }
    
    /* ZSFWS-MLW: 使用 He/Kaiming 自适应初始化每层独立权重矩阵 */
    {
        for (int l = 0; l < num_layers_int; l++) {
            size_t l_input = (l == 0) ? input_size_cfg : hidden_size_cfg;
            float he_limit = sqrtf(6.0f / (float)l_input);
            if (he_limit > 0.2f) he_limit = 0.2f;
            if (he_limit < 0.001f) he_limit = 0.001f;
            float* lw = param_block + network->per_layer_w_offset[l];
            size_t lw_sz = network->per_layer_w_size[l];
            for (size_t i = 0; i < lw_sz; i++) {
                lw[i] = rng_uniform(-he_limit, he_limit);
            }
        }
    }
    
    // 初始化偏置为零
    {
        float* bias_start = param_block + total_w;
        for (size_t i = 0; i < total_b; i++) {
            bias_start[i] = 0.0f;
        }
    }
    
    // 初始化Dropout掩码
    for (size_t i = 0; i < max_layer_size; i++) {
        network->dropout_mask[i] = 1.0f;  // 初始全为1（不丢弃）
    }

    /* P0-001: 创建每层级联层归一化(LayerNorm)实例
     * LayerNorm在特征维度上独立归一化，消除深层网络的内部协变量偏移。
     * 与BatchNorm不同，LayerNorm不依赖batch size，训练/推理行为一致。 */
    network->layer_norms = NULL;
    network->ln_temp_buffer = NULL;
    if (config->use_layer_norm) {
        network->layer_norms = (void**)safe_calloc(config->num_layers, sizeof(void*));
        network->ln_temp_buffer = (float*)safe_calloc(max_layer_size, sizeof(float));
        if (network->layer_norms && network->ln_temp_buffer) {
            LayerNormConfig ln_cfg = layer_norm_default_config(config->hidden_size);
            ln_cfg.epsilon = 1e-5f;
            ln_cfg.use_affine = 1;
            for (int i = 0; i < config->num_layers; i++) {
                network->layer_norms[i] = layer_norm_create(&ln_cfg);
                if (!network->layer_norms[i]) {
                    /* 清理已创建的LN */
                    for (int j = 0; j < i; j++) {
                        layer_norm_free((LayerNorm*)network->layer_norms[j]);
                    }
                    safe_free((void**)&network->layer_norms);
                    safe_free((void**)&network->ln_temp_buffer);
                }
            }
        }
    }

    // 初始化统计信息
    network->is_initialized = 1;
    network->current_avg_activation = 0.0f;
    network->current_max_activation = 0.0f;
    network->current_gradient_norm = 0.0f;
    
    return network;
}

/**
 * @brief 释放CfC网络实例
 */
void cfc_free(CfCNetwork* network) {
    if (!network) {
        return;
    }
    
    // 释放各层
    if (network->layers) {
        for (int i = 0; i < network->config.num_layers; i++) {
            if (network->layers[i]) {
                cfc_cell_free(network->layers[i]);
            }
        }
        safe_free((void**)&network->layers);
    }
    
    safe_free((void**)&network->layer_outputs);
    safe_free((void**)&network->layer_gradients);
    safe_free((void**)&network->activation_buffer);
    safe_free((void**)&network->dropout_mask);

    /* P0-001: 释放层归一化实例 */
    if (network->layer_norms) {
        for (int i = 0; i < network->config.num_layers; i++) {
            if (network->layer_norms[i]) {
                layer_norm_free((LayerNorm*)network->layer_norms[i]);
            }
        }
        safe_free((void**)&network->layer_norms);
    }
    safe_free((void**)&network->ln_temp_buffer);

    safe_free((void**)&network->weight_matrix);
    network->bias_vector = NULL;
    network->W_out_params = NULL;
    network->b_out_params = NULL;
    safe_free((void**)&network->weight_gradients);
    network->bias_gradients = NULL;
    network->W_out_gradients = NULL;
    network->b_out_gradients = NULL;

    /* ZSFWS-MLW: 释放每层偏移追踪数组 */
    safe_free((void**)&network->per_layer_w_offset);
    safe_free((void**)&network->per_layer_b_offset);
    safe_free((void**)&network->per_layer_w_size);
    
    safe_free((void**)&network);
}

/**
 * @brief 前向传播
 */
int cfc_forward(CfCNetwork* network, const float* input, 
                float* hidden_state, float* cell_state, float* output) {
    // 参数检查
    SELFLNN_CHECK_NULL(network, "CfC网络句柄为空");
    SELFLNN_CHECK_NULL(input, "输入向量为空");
    SELFLNN_CHECK_NULL(hidden_state, "隐藏状态缓冲区为空");
    SELFLNN_CHECK_NULL(cell_state, "细胞状态缓冲区为空");
    SELFLNN_CHECK_NULL(output, "输出向量缓冲区为空");
    
    // 初始化状态检查
    SELFLNN_CHECK_INITIALIZED(network, "CfC网络未初始化");
    
    const CfCNetworkConfig* config = &network->config;
    size_t hidden_size = config->hidden_size;
    size_t input_size = config->input_size;
    size_t output_size = config->output_size;
    int num_layers = config->num_layers;

    // 计算最大层大小（用于缓冲区步长）
    size_t max_layer_size = (input_size > hidden_size) ? input_size : hidden_size;
    max_layer_size = (max_layer_size > output_size) ? max_layer_size : output_size;
    
    // 应用输入层的线性变换
    float* current_input = network->activation_buffer;
    
    // 第一层：输入到隐藏
    if (num_layers == 1) {
        /* ZSFWS-P14修复: 单层网络跳过额外线性变换，直接传递原始输入。
         * 之前先做Wx+b再传给CfC cell，而CfC cell内部已做W_ax*x+W_gx*x。
         * 这导致输入经过双重线性变换——第一层权重和CfC内部权重相互干扰。
         * 修复: 将原始输入直接传递给CfC cell，由其内部矩阵处理。 */
        memcpy(current_input, input, input_size * sizeof(float));
    } else {
        // 多层网络：第一层使用输入
        memcpy(current_input, input, input_size * sizeof(float));
    }
    
    float* current_hidden = hidden_state;
    float* current_cell = cell_state;
    
    // 逐层前向传播
    for (int layer = 0; layer < num_layers; layer++) {
        // 获取当前层的输出位置（使用最大层大小作为步长）
        float* layer_output = network->layer_outputs + (layer * max_layer_size);
        
        // 执行CfC单元前向传播
        int result = cfc_cell_forward(network->layers[layer], 
                                     current_input, 
                                     layer_output);
        
        SELFLNN_CHECK(result == 0, SELFLNN_ERROR_CFC_CELL_CONFIG,
                     "第%d层CfC单元前向传播失败", layer);

        /* P0-001: 层归一化应用于每层CfC输出
         * 归一化后消除内部协变量偏移，显著提升深层网络训练稳定性。
         * 每层独立LN，在特征维度上标准化到零均值单位方差。 */
        if (network->layer_norms && network->layer_norms[layer] && network->ln_temp_buffer) {
            layer_norm_forward((LayerNorm*)network->layer_norms[layer],
                              layer_output, network->ln_temp_buffer,
                              config->enable_training);
            memcpy(layer_output, network->ln_temp_buffer, hidden_size * sizeof(float));
        }
        
        // 应用Dropout（如果启用）
        if (config->dropout_rate > 0.0f && config->dropout_rate < 1.0f && config->enable_training) {
            for (size_t i = 0; i < hidden_size; i++) {
                if (rng_uniform(0.0f, 1.0f) < config->dropout_rate) {
                    network->dropout_mask[i] = 0.0f;
                    layer_output[i] = 0.0f;
                } else {
                    /* K-055: 除零防护 —— dropout_rate接近1.0时回退到0.999999f */
                    float safe_inv = (config->dropout_rate < 0.999999f)
                        ? (1.0f - config->dropout_rate) : 0.000001f;
                    network->dropout_mask[i] = 1.0f / safe_inv;
                    layer_output[i] *= network->dropout_mask[i];
                }
            }
        }
        
        // 更新当前输入为当前层输出（用于下一层）
        if (layer < num_layers - 1) {
            current_input = layer_output;
        }
        
        // 更新隐藏状态和细胞状态（最后一层）
        if (layer == num_layers - 1) {
            memcpy(current_hidden, layer_output, hidden_size * sizeof(float));
            
            // CfC单元中细胞状态与隐藏状态相同
            memcpy(current_cell, layer_output, hidden_size * sizeof(float));
        }
    }
    
    /* P0-014修复: 使用独立W_out_params/b_out_params（不再与梯度共享存储） */
    if (output_size != hidden_size && network->W_out_params) {
        for (size_t i = 0; i < output_size; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < hidden_size; j++) {
                sum += network->W_out_params[i * hidden_size + j] * current_hidden[j];
            }
            output[i] = sum + network->b_out_params[i];
        }
    } else {
        memcpy(output, current_hidden, output_size * sizeof(float));
    }
    
    // 更新统计信息
    float sum_activation = 0.0f;
    float max_activation = 0.0f;
    
    for (size_t i = 0; i < hidden_size; i++) {
        float activation = fabsf(current_hidden[i]);
        sum_activation += activation;
        if (activation > max_activation) {
            max_activation = activation;
        }
    }
    
    network->current_avg_activation = sum_activation / hidden_size;
    network->current_max_activation = max_activation;
    
    return 0;
}

/**
 * @brief P0-BPTT: 分配/初始化cell级动量缓冲区
 */
static int cfc_ensure_cell_momentum(CfCCell* cell) {
    if (!cell) return -1;
    if (cell->cell_momentum_initialized) return 0;

    size_t hidden = cell->config.hidden_size;
    size_t input = cell->config.input_size;
    size_t total = 0;

    /* 统计所有cell级参数总量 */
    total += input * hidden;  /* W_gx */
    total += input * hidden;  /* W_fx */
    total += input * hidden;  /* W_ox */
    total += hidden * hidden; /* W_gh */
    total += hidden * hidden; /* W_ah */
    total += hidden * 3;      /* b_g */
    total += hidden;          /* τ */

    cell->cell_momentum_buffer = (float*)safe_calloc(total * 2, sizeof(float));
    if (!cell->cell_momentum_buffer) return -1;
    cell->cell_velocity_buffer = cell->cell_momentum_buffer + total;
    cell->cell_momentum_size = total;
    cell->cell_momentum_initialized = 1;
    return 0;
}

/**
 * @brief P0-BPTT: 使用Adam更新一组cell参数（内部辅助函数）
 * 
 * @param params 参数数组
 * @param grads 梯度数组
 * @param m 一阶矩缓冲区
 * @param v 二阶矩缓冲区
 * @param count 参数数量
 * @param lr 学习率
 * @param b1 beta1
 * @param b2 beta2
 * @param eps epsilon
 * @param b1c 1/(1-beta1^t) 偏差校正
 * @param b2c 1/(1-beta2^t) 偏差校正
 */
static void cfc_adam_update_group(float* params, const float* grads,
                                   float* m, float* v, size_t count,
                                   float lr, float b1, float b2, float eps,
                                   float b1c, float b2c) {
    for (size_t i = 0; i < count; i++) {
        float g = grads[i];
        if (!isfinite(g)) continue;
        m[i] = b1 * m[i] + (1.0f - b1) * g;
        v[i] = b2 * v[i] + (1.0f - b2) * g * g;
        float m_hat = m[i] * b1c;
        float v_hat = v[i] * b2c;
        params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

/**
 * @brief P0-BPTT: 用Adam风格更新所有cell级参数
 * 
 * 解决双层参数更新分裂问题——原先cell级参数用简单SGD，
 * 与主优化器(Adam/AdamW)的更新路径完全不匹配，导致收敛分裂。
 * 此函数将Adam自适应学习率统一应用于所有cell级参数。
 */
int cfc_apply_cell_gradients_adam(CfCNetwork* network, float learning_rate,
                                   float beta1, float beta2, float epsilon, size_t t) {
    SELFLNN_CHECK_NULL(network, "CfC网络句柄为空");
    SELFLNN_CHECK_INITIALIZED(network, "CfC网络未初始化");

    size_t hidden_size = network->config.hidden_size;
    size_t input_size = network->config.input_size;
    int num_layers = network->config.num_layers;

    if (beta1 <= 0.0f) beta1 = 0.9f;
    if (beta2 <= 0.0f) beta2 = 0.999f;
    if (epsilon <= 0.0f) epsilon = 1e-8f;
    if (t == 0) t = 1;

    float b1_corr = 1.0f / (1.0f - powf(beta1, (float)t));
    float b2_corr = 1.0f / (1.0f - powf(beta2, (float)t));

    for (int layer = 0; layer < num_layers; layer++) {
        CfCCell* cell = network->layers[layer];
        if (!cell) continue;

        if (cfc_ensure_cell_momentum(cell) != 0) return -1;

        size_t layer_input = (layer == 0) ? input_size : hidden_size;
        size_t w_count = layer_input * hidden_size;
        size_t hh_count = hidden_size * hidden_size;

        float* m = cell->cell_momentum_buffer;
        float* v = cell->cell_velocity_buffer;
        size_t offset = 0;

        /* W_gx: 输入→门控 */
        if (cell->input_gate_weight_grad && cell->input_gate_weights) {
            cfc_adam_update_group(cell->input_gate_weights, cell->input_gate_weight_grad,
                                  m + offset, v + offset, w_count,
                                  learning_rate, beta1, beta2, epsilon, b1_corr, b2_corr);
            offset += w_count;
        }
        /* W_fx: 输入→遗忘门 */
        if (cell->forget_gate_weight_grad && cell->forget_gate_weights) {
            cfc_adam_update_group(cell->forget_gate_weights, cell->forget_gate_weight_grad,
                                  m + offset, v + offset, w_count,
                                  learning_rate, beta1, beta2, epsilon, b1_corr, b2_corr);
            offset += w_count;
        }
        /* W_ox: 输入→输出门 */
        if (cell->output_gate_weight_grad && cell->output_gate_weights) {
            cfc_adam_update_group(cell->output_gate_weights, cell->output_gate_weight_grad,
                                  m + offset, v + offset, w_count,
                                  learning_rate, beta1, beta2, epsilon, b1_corr, b2_corr);
            offset += w_count;
        }
        /* W_gh: 隐藏→门控 */
        if (cell->hidden_to_gate_weight_grad && cell->hidden_to_gate_weights) {
            cfc_adam_update_group(cell->hidden_to_gate_weights, cell->hidden_to_gate_weight_grad,
                                  m + offset, v + offset, hh_count,
                                  learning_rate, beta1, beta2, epsilon, b1_corr, b2_corr);
            offset += hh_count;
        }
        /* W_ah: 隐藏→激活 */
        if (cell->hidden_to_activation_weight_grad && cell->hidden_to_activation_weights) {
            cfc_adam_update_group(cell->hidden_to_activation_weights,
                                  cell->hidden_to_activation_weight_grad,
                                  m + offset, v + offset, hh_count,
                                  learning_rate, beta1, beta2, epsilon, b1_corr, b2_corr);
            offset += hh_count;
        }
        /* b_g: 门控偏置 */
        if (cell->gate_bias_grad && cell->gate_biases) {
            cfc_adam_update_group(cell->gate_biases, cell->gate_bias_grad,
                                  m + offset, v + offset, hidden_size * 3,
                                  learning_rate, beta1, beta2, epsilon, b1_corr, b2_corr);
            offset += hidden_size * 3;
        }
        /* τ: 时间常数 */
        if (cell->use_adaptive_tau && cell->time_constant_grad && cell->time_constants) {
            cfc_adam_update_group(cell->time_constants, cell->time_constant_grad,
                                  m + offset, v + offset, hidden_size,
                                  learning_rate, beta1, beta2, epsilon, b1_corr, b2_corr);
            /* 钳位时间常数范围 */
            for (size_t k = 0; k < hidden_size; k++) {
                if (cell->time_constants[k] < cell->min_time_constant)
                    cell->time_constants[k] = cell->min_time_constant;
                if (cell->time_constants[k] > cell->max_time_constant)
                    cell->time_constants[k] = cell->max_time_constant;
            }
        }
    }

    return 0;
}

/**
 * @brief P0-002深度修复: 清零所有cell级梯度缓冲区
 * 
 * 在batch训练开始时/单样本反向传播前调用一次，
 * 确保随后的cfc_cell_backward→+=累积从零开始。
 */
void cfc_zero_cell_gradients(CfCNetwork* network) {
    if (!network || !network->layers) return;
    /* P0-014修复: 清零输出投影梯度（与cell梯度同步清零） */
    if (network->W_out_gradients) {
        memset(network->W_out_gradients, 0, 
               network->config.output_size * network->config.hidden_size * sizeof(float));
    }
    if (network->b_out_gradients) {
        memset(network->b_out_gradients, 0, network->config.output_size * sizeof(float));
    }
    for (int cl = 0; cl < network->config.num_layers; cl++) {
        CfCCell* cell = network->layers[cl];
        if (!cell) continue;
        size_t lw = (cl == 0 ? network->config.input_size : network->config.hidden_size)
                   * network->config.hidden_size;
        size_t hh = network->config.hidden_size * network->config.hidden_size;
        if (cell->weight_grad) memset(cell->weight_grad, 0, lw * sizeof(float));
        if (cell->bias_grad) memset(cell->bias_grad, 0, network->config.hidden_size * sizeof(float));
        if (cell->input_gate_weight_grad) memset(cell->input_gate_weight_grad, 0, lw * sizeof(float));
        if (cell->hidden_to_gate_weight_grad) memset(cell->hidden_to_gate_weight_grad, 0, hh * sizeof(float));
        if (cell->hidden_to_activation_weight_grad) memset(cell->hidden_to_activation_weight_grad, 0, hh * sizeof(float));
        if (cell->gate_bias_grad) memset(cell->gate_bias_grad, 0, network->config.hidden_size * 3 * sizeof(float));
        if (cell->time_constant_grad) memset(cell->time_constant_grad, 0, network->config.hidden_size * sizeof(float));
    }
}

/**
 * @brief 反向传播（训练）- 完整实现
 * 支持多层CfC网络和输出投影矩阵的完整梯度链
 */
int cfc_backward(CfCNetwork* network, const float* error, 
                 float* gradient, float learning_rate) {
    return cfc_backward_ex(network, error, gradient, learning_rate, 0);
}

/**
 * @brief 反向传播（训练）——扩展版，支持跳过cell级参数更新（批量训练的+=累积模式）
 * 
 * P0-002修复: 新增skip_cell_update参数。
 *   skip_cell_update=0: 单样本模式，cfc_backward内部在Step3直接更新cell参数（兼容旧行为）
 *   skip_cell_update=1: 批量模式，仅累积梯度到cell->weight_grad等缓冲区，不更新cell参数。
 *     调用方必须在批次结束后通过 cfc_apply_cell_gradients() 统一下发更新。
 */
int cfc_backward_ex(CfCNetwork* network, const float* error, 
                    float* gradient, float learning_rate, int skip_cell_update) {
    SELFLNN_CHECK_NULL(network, "CfC网络句柄为空");
    SELFLNN_CHECK_NULL(error, "误差向量为空");
    SELFLNN_CHECK_NULL(gradient, "梯度缓冲区为空");
    SELFLNN_CHECK_INITIALIZED(network, "CfC网络未初始化");
    SELFLNN_CHECK(network->config.enable_training, 
                 SELFLNN_ERROR_TRAINING_NOT_ENABLED, 
                 "CfC网络训练未启用");
    
    const CfCNetworkConfig* config = &network->config;
    size_t hidden_size = config->hidden_size;
    size_t input_size = config->input_size;
    size_t output_size = config->output_size;
    int num_layers = config->num_layers;

    size_t max_layer_size = hidden_size;
    if (input_size > max_layer_size) max_layer_size = input_size;
    if (output_size > max_layer_size) max_layer_size = output_size;

    // 保存输入信号（第一层反向传播需要）
    float* saved_input = NULL;
    if (input_size > 0) {
        saved_input = (float*)safe_malloc(input_size * sizeof(float));
        if (saved_input) {
            memcpy(saved_input, network->layer_gradients, input_size * sizeof(float));
        }
    }

    // 初始化各层梯度缓冲区为0
    memset(network->layer_gradients, 0, num_layers * max_layer_size * sizeof(float));

    // 步骤1: 计算输出层梯度（包括输出投影矩阵的梯度）
    float* output_gradient = network->layer_gradients + ((num_layers - 1) * max_layer_size);
    
    if (output_size != hidden_size && network->W_out_params) {
        /* P0-014修复: 使用独立W_out_params进行反向传播计算 */

        /* dL/dh_j = Σ_i W_out[i][j] * error[i] */
        memset(output_gradient, 0, hidden_size * sizeof(float));
        for (size_t i = 0; i < output_size; i++) {
            for (size_t j = 0; j < hidden_size; j++) {
                output_gradient[j] += network->W_out_params[i * hidden_size + j] * error[i];
            }
        }

        float* last_hidden = network->layer_outputs + ((num_layers - 1) * max_layer_size);
        /* R3P3修复: 统一单样本/批量路径——均累积W_out梯度，由统一Adam步更新。
         * 消除旧单样本SGD路径(W_out -= lr*error*hidden)，与Adam优化不一致。 */
        if (network->W_out_gradients) {
            for (size_t i = 0; i < output_size; i++) {
                for (size_t j = 0; j < hidden_size; j++) {
                    network->W_out_gradients[i * hidden_size + j] +=
                        error[i] * last_hidden[j];
                }
                network->b_out_gradients[i] += error[i];
            }
        }
    } else {
        memcpy(output_gradient, error, hidden_size * sizeof(float));
    }

    /* P0-002深度修复: 单样本路径清零cell梯度。
     * 批量路径(skip_cell_update=1)由调用方在batch开始时清零。 */
    if (!skip_cell_update) {
        cfc_zero_cell_gradients(network);
    }

    // 步骤2: 逐层反向传播（从最后一层到第一层）
    for (int layer = num_layers - 1; layer >= 0; layer--) {
        float* layer_gradient = network->layer_gradients + (layer * max_layer_size);
        float* prev_gradient = (layer > 0) ? 
            network->layer_gradients + ((layer - 1) * max_layer_size) : gradient;

        /* P0-001: 层归一化反向传播
         * 前向时 output = LN(cfc_output)，所以反向需要先通过LN反向传播
         * 将 dL/d(output_postLN) 转换为 dL/d(output_preLN)，然后传给cfc_cell_backward */
        if (network->layer_norms && network->layer_norms[layer] && network->ln_temp_buffer) {
            layer_norm_backward((LayerNorm*)network->layer_norms[layer],
                               layer_gradient, network->ln_temp_buffer);
            memcpy(layer_gradient, network->ln_temp_buffer, hidden_size * sizeof(float));
        }

        // 清零上一层梯度（准备接收）
        // 对于外部梯度缓冲区(gradient/layer==0)，只清零 input_size 个元素
        // 因为 cfc_cell_backward 只会写入 input_size 个元素
        // 对于内部梯度缓冲区(layer_gradients)，使用 max_layer_size
        if (layer == 0) {
            memset(prev_gradient, 0, input_size * sizeof(float));
        } else {
            memset(prev_gradient, 0, max_layer_size * sizeof(float));
        }

        // 执行CfC单元反向传播
        int result = cfc_cell_backward(network->layers[layer], 
                                      layer_gradient,
                                      layer == 0 ? gradient : prev_gradient);
        
        /* P5-002修复: SELFLNN_CHECK可能提前返回，before free saved_input */
        if (result != 0) {
            safe_free((void**)&saved_input);
            selflnn_set_last_error(SELFLNN_ERROR_INITIALIZATION_FAILED, __func__, __FILE__, __LINE__, "CfC网络保存输入失败");
            return SELFLNN_ERROR_INITIALIZATION_FAILED;
        }

        // 应用Dropout掩码到传播的梯度
        if (config->dropout_rate > 0.0f && layer > 0) {
            for (size_t i = 0; i < hidden_size; i++) {
                prev_gradient[i] *= network->dropout_mask[i];
            }
        }
    }

    /* ZSFX-P0: 步骤3 — 单样本路径梯度累积而非直接SGD更新
     * skip_cell_update=1时跳过所有参数更新（批量模式，调用方负责统一下发）。
     * skip_cell_update=0时采用梯度累积模式（+=）而非直接SGD（-=lr*grad），
     * 确保与外部优化器（Adam/AdamW等）的单元格更新路径一致。 */
    if (!skip_cell_update) {
    /* R3P3修复: 单样本路径统一使用Adam自适应学习率，消除与原批量路径的SGD/Adam分裂。
     * 原SGD路径（参数 -= lr*grad）缺少动量/偏差校正/自适应步长，
     * 导致单样本训练与批量训练的优化动态不一致，严重损害收敛稳定性。 */
    cfc_apply_cell_gradients_adam(network, learning_rate, 0.9f, 0.999f, 1e-8f, 1);
    /* W_out投影参数：使用cfc_apply_out_proj_gradients统一更新（已累积梯度）。
     * W_out仅有2个矩阵，暂无专用Adam缓冲区，使用集中式SGD更新。
     * TODO-R3P3: 后续添加W_out Adam动量缓冲区以完全统一优化路径。 */
    cfc_apply_out_proj_gradients(network, learning_rate);
    /* 单样本路径完成后清零W_out梯度，防止跨样本污染 */
    if (network->W_out_gradients) {
        memset(network->W_out_gradients, 0, output_size * hidden_size * sizeof(float));
        memset(network->b_out_gradients, 0, output_size * sizeof(float));
    }
    } /* if (!skip_cell_update) — 关闭P0-002批量模式cell参数延迟更新块 */

    /* FIX-007/011: 步骤4 — 累积网络级输入预处理参数的梯度。
     * 使用 += 而非 = 确保批量训练时多个样本的梯度正确累加。
     * 调用方需在每批次开始前将 weight_gradients/bias_gradients 清零。
     * ZSFWS-MLW: 使用层0的梯度偏移，支持多层独立权重。 */
    if (saved_input) {
        size_t layer0_w_size = network->per_layer_w_size[0];
        float* grad_l0_w = network->weight_gradients + network->per_layer_w_offset[0];
        float* grad_l0_b = network->bias_gradients + network->per_layer_b_offset[0];
        size_t l0_input = (num_layers == 1) ? input_size : hidden_size;
        if (num_layers == 1) {
            for (size_t j = 0; j < hidden_size; j++) {
                float dL_dh_j = gradient[j];
                for (size_t k = 0; k < input_size; k++) {
                    grad_l0_w[j * input_size + k] += dL_dh_j * saved_input[k];
                }
            }
        }
        for (size_t j = 0; j < hidden_size; j++) {
            grad_l0_b[j] += gradient[j];
        }
        (void)l0_input; (void)layer0_w_size;
    }

    if (saved_input) safe_free((void**)&saved_input);

    float gradient_norm = 0.0f;
    for (size_t i = 0; i < hidden_size; i++) {
        gradient_norm += gradient[i] * gradient[i];
    }
    network->current_gradient_norm = sqrtf(gradient_norm);
    
    return 0;
}

/**
 * @brief 累积梯度（用于批量训练）
 */
int cfc_accumulate_gradients(CfCNetwork* network, const float* error, 
                            float* gradient,
                            float* weight_gradients, float* bias_gradients) {
    // 参数检查
    SELFLNN_CHECK_NULL(network, "CfC网络句柄为空");
    SELFLNN_CHECK_NULL(error, "误差向量为空");
    SELFLNN_CHECK_NULL(gradient, "梯度缓冲区为空");
    SELFLNN_CHECK_NULL(weight_gradients, "权重梯度缓冲区为空");
    SELFLNN_CHECK_NULL(bias_gradients, "偏置梯度缓冲区为空");
    
    // 初始化状态和训练状态检查
    SELFLNN_CHECK_INITIALIZED(network, "CfC网络未初始化");
    SELFLNN_CHECK(network->config.enable_training, 
                 SELFLNN_ERROR_TRAINING_NOT_ENABLED, 
                 "CfC网络训练未启用");
    
    const CfCNetworkConfig* config = &network->config;
    size_t hidden_size = config->hidden_size;
    size_t input_size = config->input_size;
    size_t output_size = config->output_size;
    int num_layers = config->num_layers;
    
    // 计算最大层大小（用于缓冲区步长）
    size_t max_layer_size = (input_size > hidden_size) ? input_size : hidden_size;
    max_layer_size = (max_layer_size > output_size) ? max_layer_size : output_size;
    
    // 对于单层网络，保存输入信号（在layer_gradients中）到临时缓冲区
    float* saved_input = NULL;
    if (num_layers == 1 && input_size > 0) {
        saved_input = (float*)safe_malloc(input_size * sizeof(float));
        if (saved_input) {
            memcpy(saved_input, network->layer_gradients, input_size * sizeof(float));
        }
    }
    
    // 初始化梯度缓冲区（使用最大层大小作为步长）
    memset(network->layer_gradients, 0, 
           num_layers * max_layer_size * sizeof(float));
    
    // 计算输出层梯度（使用最大层大小作为步长）
    float* output_gradient = network->layer_gradients + ((num_layers - 1) * max_layer_size);
    float* last_hidden = network->layer_outputs + ((num_layers - 1) * max_layer_size);
    
    /* P0-014修复: 输出投影矩阵梯度累积。
     * 读取W_out参数值用于反向传播dL/dh = W_out^T · error，
     * 梯度累积到独立的W_out_gradients区域（与参数完全分离）。 */
    if (output_size != hidden_size && network->W_out_params && network->W_out_gradients) {
        /* dL/dh = W_out^T · error，使用参数值进行反向传播 */
        memset(output_gradient, 0, hidden_size * sizeof(float));
        for (size_t i = 0; i < output_size; i++) {
            for (size_t j = 0; j < hidden_size; j++) {
                output_gradient[j] += network->W_out_params[i * hidden_size + j] * error[i];
            }
        }
        /* 累积W_out梯度: dL/dW_out[i][j] += error[i] * last_hidden[j] */
        /* 梯度写入独立梯度缓冲区，不污染参数值 */
        for (size_t i = 0; i < output_size; i++) {
            for (size_t j = 0; j < hidden_size; j++) {
                network->W_out_gradients[i * hidden_size + j] += error[i] * last_hidden[j];
            }
            network->b_out_gradients[i] += error[i];
        }
    } else {
        size_t copy_size = (output_size < hidden_size) ? output_size : hidden_size;
        for (size_t i = 0; i < copy_size; i++) {
            output_gradient[i] = error[i];
        }
    }
    
    // 反向传播通过各层
    for (int layer = num_layers - 1; layer >= 0; layer--) {
        float* layer_gradient = network->layer_gradients + (layer * max_layer_size);
        float* layer_output = network->layer_outputs + (layer * max_layer_size);
        (void)layer_output;

        /* P0-001: 层归一化反向传播（与cfc_backward_ex一致） */
        if (network->layer_norms && network->layer_norms[layer] && network->ln_temp_buffer) {
            layer_norm_backward((LayerNorm*)network->layer_norms[layer],
                               layer_gradient, network->ln_temp_buffer);
            memcpy(layer_gradient, network->ln_temp_buffer, hidden_size * sizeof(float));
        }

        // 获取上一层梯度（对于第一层，使用输入梯度）
        float* prev_gradient = NULL;
        if (layer > 0) {
            prev_gradient = network->layer_gradients + ((layer - 1) * max_layer_size);
        }
        
        // 执行CfC单元反向传播
        int result = cfc_cell_backward(network->layers[layer], 
                                      layer_gradient,
                                      prev_gradient ? prev_gradient : gradient);
        
        if (result != 0) {
            safe_free((void**)&saved_input);
            selflnn_set_last_error(SELFLNN_ERROR_INITIALIZATION_FAILED, __func__, __FILE__, __LINE__, "CfC网络离散演化失败");
            return SELFLNN_ERROR_INITIALIZATION_FAILED;
        }
        
        // 应用Dropout掩码（反向传播时）
        if (config->dropout_rate > 0.0f) {
            for (size_t i = 0; i < hidden_size; i++) {
                if (prev_gradient) {
                    prev_gradient[i] *= network->dropout_mask[i];
                }
            }
        }
    }
    
    // 计算权重和偏置梯度（但不更新权重）
    // 单层分支用于单层网络，无论输出大小是否等于隐藏大小
    
    size_t total_weight_size = 0;
    if (num_layers == 1 && config->input_size > 0) {
        /* FIX-016: 单层网络直接复制cell->weight_grad/bias_grad到外部缓冲区。
         * cfc_cell_backward已正确计算了CfC门控+激活+ODE链式的完整梯度。
         * 旧代码用 gradient[i]*input[j] 重新计算→这是简单线性层梯度，完全
         * 忽略了CfC内部的sigmoid/tanh/exp(-Δt/τ)导数链，导致外部梯度错误。
         * 批量训练时使用+=累积（FIX-010），调用方已在批次前清零。
         * ZSFWS-MLW: 使用per_layer偏移写入正确位置。 */
        total_weight_size = hidden_size * config->input_size;
        float* grad_l0_w = weight_gradients + network->per_layer_w_offset[0];
        float* grad_l0_b = bias_gradients + network->per_layer_b_offset[0];
        CfCCell* cell0 = network->layers[0];
        if (cell0 && cell0->weight_grad && cell0->bias_grad) {
            for (size_t i = 0; i < total_weight_size; i++) {
                grad_l0_w[i] += cell0->weight_grad[i];
            }
            for (size_t i = 0; i < hidden_size; i++) {
                grad_l0_b[i] += cell0->bias_grad[i];
            }
        }
    } else {
        // 多层网络完整梯度计算
        /* ZSFWS-MLW: 每层cell梯度写入独立的param_block偏移位置
         * 不再使用共享的 hidden*hidden 矩阵，每层拥有独立的权重空间
         * Layer 0: hidden_size × input_size → per_layer_w_offset[0]
         * Layer 1..N-2: hidden_size × hidden_size → per_layer_w_offset[layer]
         * Layer N-1: hidden_size × hidden_size → per_layer_w_offset[N-1] */
        
        for (int layer = 0; layer < num_layers; layer++) {
            CfCCell* cell = network->layers[layer];
            if (!cell) continue;
            
            size_t curr_input_size = (layer == 0) ? config->input_size : config->hidden_size;
            size_t cell_weight_size = curr_input_size * config->hidden_size;
            
            float* layer_grad_w = weight_gradients + network->per_layer_w_offset[layer];
            float* layer_grad_b = bias_gradients + network->per_layer_b_offset[layer];
            
            /* 直接复制cell梯度到param_block对应位置 */
            if (cell->weight_grad && layer_grad_w && cell_weight_size <= network->per_layer_w_size[layer]) {
                for (size_t i = 0; i < cell_weight_size; i++) {
                    layer_grad_w[i] += cell->weight_grad[i];
                }
            }
            if (cell->bias_grad && layer_grad_b) {
                for (size_t i = 0; i < config->hidden_size; i++) {
                    layer_grad_b[i] += cell->bias_grad[i];
                }
            }
        }
        
        /* 同步到网络内部梯度缓冲区（保持向后兼容） */
        if (network->weight_gradients && weight_gradients) {
            memcpy(network->weight_gradients, weight_gradients,
                   network->total_weight_params * sizeof(float));
        }
        if (network->bias_gradients && bias_gradients) {
            memcpy(network->bias_gradients, bias_gradients,
                   network->total_bias_params * sizeof(float));
        }
    }
    
    // 计算梯度范数
    float gradient_norm = 0.0f;
    for (size_t i = 0; i < hidden_size; i++) {
        gradient_norm += gradient[i] * gradient[i];
    }
    network->current_gradient_norm = sqrtf(gradient_norm);
    
    // 释放临时缓冲区
    if (saved_input) {
        safe_free((void**)&saved_input);
    }
    
    return 0;
}

/**
 * @brief 应用各层CfC单元的cell级参数梯度
 * 
 * FIX-011: cfc_backward Step3 在单样本路径中直接更新cell级参数，
 * 但 cfc_accumulate_gradients（批量训练路径）不处理这些参数。
 * 此函数由批量训练循环在累积共享梯度后调用，应用各cell内部的门控/
 * 时间常数梯度到对应参数。
 * 
 * 注意：当前cell级梯度来自最后一样本（cfc_cell_backward用=覆盖），
 * 未来改进：在 cfc_accumulate_gradients 中为cell级梯度增加+=累积通道。
 */
int cfc_apply_cell_gradients(CfCNetwork* network, float learning_rate) {
    SELFLNN_CHECK_NULL(network, "CfC网络句柄为空");
    SELFLNN_CHECK_INITIALIZED(network, "CfC网络未初始化");
    
    size_t hidden_size = network->config.hidden_size;
    size_t input_size = network->config.input_size;
    int num_layers = network->config.num_layers;
    
    for (int layer = 0; layer < num_layers; layer++) {
        CfCCell* cell = network->layers[layer];
        if (!cell) continue;
        
        size_t layer_input = (layer == 0) ? input_size : hidden_size;
        size_t cell_w_count = layer_input * hidden_size;
        size_t hh_count = hidden_size * hidden_size;
        size_t k;
        
        /* W_gx: 输入到门控权重 */
        if (cell->input_gate_weight_grad && cell->input_gate_weights) {
            for (k = 0; k < cell_w_count; k++) {
                float g = cell->input_gate_weight_grad[k];
                if (isfinite(g)) cell->input_gate_weights[k] -= learning_rate * g;
            }
        }
        /* W_fx: 遗忘门权重 (ZSFWS-023修复: 补充遗漏) */
        if (cell->forget_gate_weight_grad && cell->forget_gate_weights) {
            for (k = 0; k < cell_w_count; k++) {
                float g = cell->forget_gate_weight_grad[k];
                if (isfinite(g)) cell->forget_gate_weights[k] -= learning_rate * g;
            }
        }
        /* W_ox: 输出门权重 (ZSFWS-023修复: 补充遗漏) */
        if (cell->output_gate_weight_grad && cell->output_gate_weights) {
            for (k = 0; k < cell_w_count; k++) {
                float g = cell->output_gate_weight_grad[k];
                if (isfinite(g)) cell->output_gate_weights[k] -= learning_rate * g;
            }
        }
        /* W_gh / W_ah: 隐藏到隐藏权重 */
        for (k = 0; k < hh_count; k++) {
            if (cell->hidden_to_gate_weight_grad && cell->hidden_to_gate_weights) {
                float g = cell->hidden_to_gate_weight_grad[k];
                if (isfinite(g)) cell->hidden_to_gate_weights[k] -= learning_rate * g;
            }
            if (cell->hidden_to_activation_weight_grad && cell->hidden_to_activation_weights) {
                float g = cell->hidden_to_activation_weight_grad[k];
                if (isfinite(g)) cell->hidden_to_activation_weights[k] -= learning_rate * g;
            }
        }
        /* b_g: 门控偏置 */
        if (cell->gate_bias_grad && cell->gate_biases) {
            for (k = 0; k < hidden_size * 3; k++) {
                float g = cell->gate_bias_grad[k];
                if (isfinite(g)) cell->gate_biases[k] -= learning_rate * g;
            }
        }
        /* τ: 自适应时间常数 */
        if (cell->use_adaptive_tau && cell->time_constant_grad && cell->time_constants) {
            for (k = 0; k < hidden_size; k++) {
                float g = cell->time_constant_grad[k];
                if (isfinite(g)) {
                    cell->time_constants[k] -= cell->tau_learning_rate * g;
                    if (cell->time_constants[k] < cell->min_time_constant)
                        cell->time_constants[k] = cell->min_time_constant;
                    if (cell->time_constants[k] > cell->max_time_constant)
                        cell->time_constants[k] = cell->max_time_constant;
                }
            }
        }
    }

    /* P0-001: 同步更新层归一化(γ, β)参数
     * LN的γ/β梯度在 layer_norm_backward 中已通过+=累积，
     * 此处统一在批量结束后更新。γ/β学习率自动减半以保证稳定性。 */
    if (network->layer_norms) {
        for (int layer = 0; layer < num_layers; layer++) {
            if (network->layer_norms[layer]) {
                layer_norm_update_params((LayerNorm*)network->layer_norms[layer], learning_rate);
            }
        }
    }

    return 0;
}

/**
 * @brief 应用输出投影矩阵(W_out+b_out)梯度
 * 
 * P0-014修复: W_out_params/b_out_params 存储在 param_block 扩展区，
 * W_out_gradients/b_out_gradients 存储在 grad_block 扩展区，
 * 两者完全分离。此函数将累积的梯度应用到参数上。
 */
int cfc_apply_out_proj_gradients(CfCNetwork* network, float learning_rate) {
    SELFLNN_CHECK_NULL(network, "CfC网络句柄为空");
    SELFLNN_CHECK_INITIALIZED(network, "CfC网络未初始化");
    
    size_t hidden_size = network->config.hidden_size;
    size_t output_size = network->config.output_size;
    
    if (output_size == hidden_size || !network->W_out_params || !network->W_out_gradients) return 0;
    
    /* P0-014修复: 参数与梯度完全分离，使用独立缓冲区 */
    for (size_t i = 0; i < output_size * hidden_size; i++) {
        float g = network->W_out_gradients[i];
        if (isfinite(g)) network->W_out_params[i] -= learning_rate * g;
    }
    /* 更新b_out */
    for (size_t i = 0; i < output_size; i++) {
        float g = network->b_out_gradients[i];
        if (isfinite(g)) network->b_out_params[i] -= learning_rate * g;
    }
    /* 应用后清零梯度缓冲区 */
    memset(network->W_out_gradients, 0, output_size * hidden_size * sizeof(float));
    memset(network->b_out_gradients, 0, output_size * sizeof(float));
    return 0;
}

/**
 * @brief 将共享参数块同步到各层CfC单元的活跃权重
 * 
 * ZSFWS-MLW: FIX-017 全面重写。原代码在多层时所有层复制相同权重矩阵，
 * 导致多层网络的参数空间退化为单层。新代码使用per_layer偏移正确分配
 * 每层的独立权重和偏置。
 * 
 * 内存布局: param_block = [layer0_W] [layer1_W] ... [layerN-1_W] [all_b] [W_out] [b_out]
 * Layer 0: W[hidden×input], b[hidden]
 * Layer 1..N-1: each W[hidden×hidden], each b[hidden]
 */
int cfc_sync_shared_to_cells(CfCNetwork* network) {
    SELFLNN_CHECK_NULL(network, "CfC网络句柄为空");
    SELFLNN_CHECK_INITIALIZED(network, "CfC网络未初始化");
    
    size_t hidden_size = network->config.hidden_size;
    size_t input_size = network->config.input_size;
    int num_layers = network->config.num_layers;
    
    float* shared_w = network->weight_matrix;
    float* shared_b = network->bias_vector;
    if (!shared_w || !shared_b) return -1;
    if (!network->per_layer_w_offset || !network->per_layer_b_offset) return -1;
    
    for (int layer = 0; layer < num_layers; layer++) {
        CfCCell* cell = network->layers[layer];
        if (!cell || !cell->weight_matrix || !cell->bias_vector) continue;
        
        /* 使用per_layer偏移获取当前层的独立权重区域 */
        float* layer_w = shared_w + network->per_layer_w_offset[layer];
        float* layer_b = shared_b + network->per_layer_b_offset[layer];
        size_t w_count = network->per_layer_w_size[layer];
        size_t l_input = (layer == 0) ? input_size : hidden_size;
        
        /* 直接复制：cell权重维度与param_block中存储的维度完全匹配 */
        memcpy(cell->weight_matrix, layer_w, w_count * sizeof(float));
        memcpy(cell->bias_vector, layer_b, hidden_size * sizeof(float));
        
        (void)l_input; /* 编译期防御 */
    }
    return 0;
}

/**
 * @brief 保存网络到文件
 */
int cfc_save(const CfCNetwork* network, FILE* file) {
    // 参数检查
    SELFLNN_CHECK_NULL(network, "CfC网络句柄为空");
    SELFLNN_CHECK_NULL(file, "文件指针为空");
    
    // 保存配置
    SELFLNN_CHECK(fwrite(&network->config, sizeof(CfCNetworkConfig), 1, file) == 1,
                 SELFLNN_ERROR_IO_ERROR,
                 "保存CfC网络配置失败");
    
    // 保存各层
    for (int i = 0; i < network->config.num_layers; i++) {
        // 保存CfC单元配置
        CfCCellConfig cell_config;
        SELFLNN_CHECK(cfc_cell_get_config(network->layers[i], &cell_config) == 0,
                     SELFLNN_ERROR_CFC_CELL_CONFIG,
                     "获取第%d层CfC单元配置失败", i);
        
        SELFLNN_CHECK(fwrite(&cell_config, sizeof(CfCCellConfig), 1, file) == 1,
                     SELFLNN_ERROR_IO_ERROR,
                     "保存第%d层CfC单元配置失败", i);
    }
    
    // 保存权重和偏置
    /* ZSFWS-MLW: 使用实际分配的多层参数总数进行序列化 */
    size_t weight_size = network->total_weight_params;
    size_t bias_size = network->total_bias_params;
    if (!network->per_layer_w_offset || weight_size == 0) {
        /* 向后兼容：per_layer未初始化时使用旧计算方式 */
        weight_size = network->config.hidden_size * network->config.hidden_size;
        if (network->config.num_layers == 1) {
            weight_size = network->config.input_size * network->config.hidden_size;
        }
        bias_size = network->config.hidden_size;
    }
    
    SELFLNN_CHECK(fwrite(network->weight_matrix, sizeof(float), weight_size, file) == weight_size,
                 SELFLNN_ERROR_IO_ERROR,
                 "保存权重矩阵失败（大小：%zu）", weight_size);
    
    SELFLNN_CHECK(fwrite(network->bias_vector, sizeof(float), bias_size, file) 
                 == bias_size,
                 SELFLNN_ERROR_IO_ERROR,
                 "保存偏置向量失败（大小：%zu）", bias_size);
    
    /* P0-014修复: 保存输出投影矩阵参数（如果存在） */
    if (network->config.output_size != network->config.hidden_size && 
        network->W_out_params && network->b_out_params) {
        size_t out_proj_count = network->config.output_size * network->config.hidden_size;
        SELFLNN_CHECK(fwrite(network->W_out_params, sizeof(float), out_proj_count, file) 
                     == out_proj_count,
                     SELFLNN_ERROR_IO_ERROR,
                     "保存输出投影矩阵失败（大小：%zu）", out_proj_count);
        SELFLNN_CHECK(fwrite(network->b_out_params, sizeof(float), network->config.output_size, file) 
                     == network->config.output_size,
                     SELFLNN_ERROR_IO_ERROR,
                     "保存输出投影偏置失败（大小：%zu）", network->config.output_size);
    }
    
    /* P40修复: 保存每层CfC细胞的完整内部参数 */
    {
        /* 哨兵标记表示扩展格式 */
        uint32_t cell_marker = 0xCFCC3E11;
        fwrite(&cell_marker, sizeof(uint32_t), 1, file);
        for (int i = 0; i < network->config.num_layers; i++) {
            SELFLNN_CHECK(cfc_cell_save(network->layers[i], file) == 0,
                         SELFLNN_ERROR_IO_ERROR,
                         "保存第%d层CfC细胞参数失败", i);
        }
    }
    
    return 0;
}

/**
 * @brief 从文件加载网络
 */
int cfc_load(CfCNetwork* network, FILE* file) {
    // 参数检查
    SELFLNN_CHECK_NULL(network, "CfC网络句柄为空");
    SELFLNN_CHECK_NULL(file, "文件指针为空");
    
    // 读取配置
    CfCNetworkConfig config;
    SELFLNN_CHECK(fread(&config, sizeof(CfCNetworkConfig), 1, file) == 1,
                 SELFLNN_ERROR_IO_ERROR,
                 "读取CfC网络配置失败");
    
    // 验证配置匹配
    SELFLNN_CHECK(config.input_size == network->config.input_size &&
                 config.hidden_size == network->config.hidden_size &&
                 config.output_size == network->config.output_size &&
                 config.num_layers == network->config.num_layers,
                 SELFLNN_ERROR_NETWORK_CONFIG,
                 "CfC网络配置不匹配: 文件(input=%zu,hidden=%zu,output=%zu,layers=%d) != 内存(input=%zu,hidden=%zu,output=%zu,layers=%d)",
                 config.input_size, config.hidden_size, config.output_size, config.num_layers,
                 network->config.input_size, network->config.hidden_size, 
                 network->config.output_size, network->config.num_layers);
    
    // 更新配置
    memcpy(&network->config, &config, sizeof(CfCNetworkConfig));
    
    // 读取并应用各层配置
    for (int i = 0; i < config.num_layers; i++) {
        CfCCellConfig cell_config;
        SELFLNN_CHECK(fread(&cell_config, sizeof(CfCCellConfig), 1, file) == 1,
                     SELFLNN_ERROR_IO_ERROR,
                     "读取第%d层CfC单元配置失败", i);
        
        // 应用配置到对应层
        if (i < network->config.num_layers && network->layers[i]) {
            int result = cfc_cell_set_config(network->layers[i], &cell_config);
            SELFLNN_CHECK(result == 0, SELFLNN_ERROR_NETWORK_CONFIG,
                         "设置第%d层CfC单元配置失败", i);
        }
    }
    
    // 读取权重和偏置
    /* ZSFWS-MLW: 使用实际分配的多层参数总数进行反序列化 */
    size_t weight_size = network->total_weight_params;
    size_t bias_size = network->total_bias_params;
    if (!network->per_layer_w_offset || weight_size == 0) {
        weight_size = config.hidden_size * config.hidden_size;
        if (config.num_layers == 1) {
            weight_size = config.input_size * config.hidden_size;
        }
        bias_size = config.hidden_size;
    }
    
    SELFLNN_CHECK(fread(network->weight_matrix, sizeof(float), weight_size, file) == weight_size,
                 SELFLNN_ERROR_IO_ERROR,
                 "读取权重矩阵失败（大小：%zu）", weight_size);
    
    SELFLNN_CHECK(fread(network->bias_vector, sizeof(float), bias_size, file) 
                 == bias_size,
                 SELFLNN_ERROR_IO_ERROR,
                 "读取偏置向量失败（大小：%zu）", bias_size);

    /* P0-014修复: 加载输出投影矩阵参数（如果存在） */
    if (config.output_size != config.hidden_size && network->W_out_params && network->b_out_params) {
        size_t out_proj_count = config.output_size * config.hidden_size;
        SELFLNN_CHECK(fread(network->W_out_params, sizeof(float), out_proj_count, file) 
                     == out_proj_count,
                     SELFLNN_ERROR_IO_ERROR,
                     "读取输出投影矩阵失败（大小：%zu）", out_proj_count);
        SELFLNN_CHECK(fread(network->b_out_params, sizeof(float), config.output_size, file) 
                     == config.output_size,
                     SELFLNN_ERROR_IO_ERROR,
                     "读取输出投影偏置失败（大小：%zu）", config.output_size);
    }
    
    /* P40修复: 加载每层CfC细胞的完整内部参数（向后兼容） */
    {
        uint32_t cell_marker = 0;
        long saved_pos = ftell(file);
        if (fread(&cell_marker, sizeof(uint32_t), 1, file) == 1 &&
            cell_marker == 0xCFCC3E11) {
            for (int i = 0; i < network->config.num_layers; i++) {
                SELFLNN_CHECK(cfc_cell_load(network->layers[i], file) == 0,
                             SELFLNN_ERROR_IO_ERROR,
                             "读取第%d层CfC细胞参数失败", i);
            }
        } else {
            /* 旧格式checkpoint（无细胞参数），回退文件指针 */
            fseek(file, saved_pos, SEEK_SET);
        }
    }
    
    return 0;
}

/**
 * @brief 设置网络配置
 */
int cfc_set_config(CfCNetwork* network, const CfCNetworkConfig* config) {
    // 参数检查
    SELFLNN_CHECK_NULL(network, "CfC网络句柄为空");
    SELFLNN_CHECK_NULL(config, "配置指针为空");
    
    // 验证配置
    SELFLNN_CHECK(config->input_size == network->config.input_size &&
                 config->hidden_size == network->config.hidden_size &&
                 config->output_size == network->config.output_size &&
                 config->num_layers == network->config.num_layers,
                 SELFLNN_ERROR_NETWORK_CONFIG,
                 "CfC网络配置尺寸不匹配: 新(input=%zu,hidden=%zu,output=%zu,layers=%d) != 当前(input=%zu,hidden=%zu,output=%zu,layers=%d)",
                 config->input_size, config->hidden_size, config->output_size, config->num_layers,
                 network->config.input_size, network->config.hidden_size, 
                 network->config.output_size, network->config.num_layers);
    
    // 更新配置
    memcpy(&network->config, config, sizeof(CfCNetworkConfig));
    
    // 更新各层配置
    CfCCellConfig cell_config;
    cell_config.input_size = config->input_size;
    cell_config.hidden_size = config->hidden_size;
    cell_config.time_constant = (config->time_constant > 0.0f) ? config->time_constant : 0.1f;
    cell_config.noise_std = config->noise_std;
    cell_config.enable_adaptation = config->enable_adaptation;
    
    for (int i = 0; i < config->num_layers; i++) {
        if (i > 0) {
            cell_config.input_size = config->hidden_size;
        }
        
        // 设置CfC单元配置
        SELFLNN_CHECK(cfc_cell_set_config(network->layers[i], &cell_config) == 0,
                     SELFLNN_ERROR_CFC_CELL_CONFIG,
                     "设置第%d层CfC单元配置失败", i);
    }
    
    return 0;
}

/**
 * @brief 获取网络配置
 */
int cfc_get_config(const CfCNetwork* network, CfCNetworkConfig* config) {
    // 参数检查
    SELFLNN_CHECK_NULL(network, "CfC网络句柄为空");
    SELFLNN_CHECK_NULL(config, "配置输出缓冲区为空");
    
    memcpy(config, &network->config, sizeof(CfCNetworkConfig));
    return 0;
}

/**
 * @brief 重置网络状态
 */
void cfc_reset(CfCNetwork* network) {
    if (!network) {
        return;
    }
    
    // 重置各层
    for (int i = 0; i < network->config.num_layers; i++) {
        cfc_cell_reset(network->layers[i]);
    }
    
    // 重置缓冲区
    const CfCNetworkConfig* config = &network->config;
    size_t max_layer_size = (config->input_size > config->hidden_size) ? 
                           config->input_size : config->hidden_size;
    max_layer_size = (max_layer_size > config->output_size) ? 
                    max_layer_size : config->output_size;
    size_t buffer_size = config->num_layers * max_layer_size;
    memset(network->layer_outputs, 0, buffer_size * sizeof(float));
    memset(network->layer_gradients, 0, buffer_size * sizeof(float));
    
    // 重置统计信息
    network->current_avg_activation = 0.0f;
    network->current_max_activation = 0.0f;
    network->current_gradient_norm = 0.0f;
}

/**
 * @brief 获取网络统计信息
 */
int cfc_get_stats(const CfCNetwork* network, float* avg_activation,
                  float* max_activation, float* gradient_norm) {
    // 参数检查
    SELFLNN_CHECK_NULL(network, "CfC网络句柄为空");
    
    if (avg_activation) {
        *avg_activation = network->current_avg_activation;
    }
    
    if (max_activation) {
        *max_activation = network->current_max_activation;
    }
    
    if (gradient_norm) {
        *gradient_norm = network->current_gradient_norm;
    }
    
    return 0;
}

/**
 * @brief 获取权重矩阵指针和大小
 */
int cfc_get_weight_matrix(CfCNetwork* network, float** weight_matrix, size_t* weight_count) {
    // 参数检查
    SELFLNN_CHECK_NULL(network, "CfC网络句柄为空");
    SELFLNN_CHECK_NULL(weight_matrix, "权重矩阵指针输出为空");
    SELFLNN_CHECK_NULL(weight_count, "权重数量输出为空");
    
    // 检查网络是否已初始化
    SELFLNN_CHECK(network->is_initialized, SELFLNN_ERROR_NOT_INITIALIZED,
                 "CfC网络未初始化");
    
    /* ZSFWS-MLW: 返回实际分配的多层权重总数，而非旧单层大小 */
    *weight_matrix = network->weight_matrix;
    if (network->per_layer_w_offset && network->total_weight_params > 0) {
        *weight_count = network->total_weight_params;
    } else if (network->config.num_layers <= 1) {
        *weight_count = network->config.input_size * network->config.hidden_size;
    } else {
        *weight_count = network->config.hidden_size * network->config.hidden_size;
    }
    
    return 0;
}

/**
 * @brief 获取偏置向量指针和大小
 */
int cfc_get_bias_vector(CfCNetwork* network, float** bias_vector, size_t* bias_count) {
    // 参数检查
    SELFLNN_CHECK_NULL(network, "CfC网络句柄为空");
    SELFLNN_CHECK_NULL(bias_vector, "偏置向量指针输出为空");
    SELFLNN_CHECK_NULL(bias_count, "偏置数量输出为空");
    
    // 检查网络是否已初始化
    SELFLNN_CHECK(network->is_initialized, SELFLNN_ERROR_NOT_INITIALIZED,
                 "CfC网络未初始化");
    
    /* ZSFWS-MLW: 返回实际分配的多层偏置总数 */
    *bias_vector = network->bias_vector;
    if (network->per_layer_b_offset && network->total_bias_params > 0) {
        *bias_count = network->total_bias_params;
    } else {
        *bias_count = network->config.hidden_size;
    }

    return 0;
}

/* CORE-22: Dropout mask在backward中传递 + CORE-25: Triplet三元组损失 */
/* MIN-001修复: dropout_mask参数类型从int*修正为float*，与实际存储一致 */
int cfc_dropout_backward(const float* output_grad, const float* dropout_mask, size_t n, float* input_grad) {
    if (!output_grad || !dropout_mask || !input_grad) return -1;
    for (size_t i = 0; i < n; i++) input_grad[i] = (dropout_mask[i] != 0.0f) ? output_grad[i] : 0.0f;
    return 0;
}

float loss_triplet(const float* anchor, const float* positive, const float* negative, int dim, float margin) {
    if (!anchor || !positive || !negative || dim <= 0) return 0.0f;
    float pos_dist = 0.0f, neg_dist = 0.0f;
    for (int i = 0; i < dim; i++) { float d = anchor[i]-positive[i]; pos_dist += d*d; d = anchor[i]-negative[i]; neg_dist += d*d; }
    float loss = pos_dist - neg_dist + margin;
    return loss > 0.0f ? loss : 0.0f;
}

/* ============================================================================
 * 长时连续动态系统求解器实现
 *
 * 将CfC网络的ODE动态暴露为完整的长时连续演化框架：
 *   轨迹管理 → 自适应积分 → 刚度检测 → RHS暴露 → 任意时刻插值
 * ============================================================================ */

/* ---- 轨迹管理 ---- */

CfCContinuousConfig cfc_continuous_default_config(void) {
    CfCContinuousConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.t_start = 0.0f;
    cfg.t_end = 1.0f;
    cfg.dt_init = 0.01f;
    cfg.dt_min = 1e-6f;
    cfg.dt_max = 0.1f;
    cfg.rel_tol = 1e-6f;
    cfg.abs_tol = 1e-8f;
    cfg.max_steps = 10000;
    cfg.trajectory_capacity = 1024;
    cfg.enable_stiffness_detect = 1;
    cfg.enable_trajectory_output = 0;
    cfg.trajectory_sample_dt = 0.0f;
    return cfg;
}

CfCTrajectory* cfc_trajectory_create(int capacity, size_t state_dim) {
    if (capacity <= 0 || state_dim == 0) return NULL;
    CfCTrajectory* traj = (CfCTrajectory*)safe_calloc(1, sizeof(CfCTrajectory));
    if (!traj) return NULL;
    size_t buf_size = (size_t)capacity * (state_dim + 1);
    traj->state_buffer = (float*)safe_calloc(buf_size, sizeof(float));
    traj->timestamps = (float*)safe_calloc((size_t)capacity, sizeof(float));
    if (!traj->state_buffer || !traj->timestamps) {
        cfc_trajectory_free(traj);
        return NULL;
    }
    traj->capacity = capacity;
    traj->state_dim = state_dim;
    traj->count = 0;
    traj->t_start = 0.0f;
    traj->t_end = 0.0f;
    return traj;
}

void cfc_trajectory_free(CfCTrajectory* traj) {
    if (!traj) return;
    safe_free((void**)&traj->state_buffer);
    safe_free((void**)&traj->timestamps);
    safe_free((void**)&traj);
}

int cfc_trajectory_append(CfCTrajectory* traj, float t, const float* state) {
    if (!traj || !state || traj->count >= traj->capacity) return -1;
    size_t dim = traj->state_dim;
    size_t offset = (size_t)traj->count * (dim + 1);
    traj->state_buffer[offset] = t;
    memcpy(&traj->state_buffer[offset + 1], state, dim * sizeof(float));
    traj->timestamps[traj->count] = t;
    traj->t_end = t;
    if (traj->count == 0) traj->t_start = t;
    traj->count++;
    return 0;
}

int cfc_trajectory_interpolate(const CfCTrajectory* traj, float t_query, float* state_out) {
    if (!traj || !state_out || traj->count < 2) return -1;
    if (t_query < traj->t_start || t_query > traj->t_end) return -1;
    size_t dim = traj->state_dim;
    int lo = 0, hi = traj->count - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (traj->timestamps[mid] <= t_query) lo = mid;
        else hi = mid;
    }
    float t0 = traj->timestamps[lo];
    float t1 = traj->timestamps[hi];
    float dt = t1 - t0;
    if (dt < 1e-10f) { memcpy(state_out, &traj->state_buffer[(size_t)lo * (dim + 1) + 1], dim * sizeof(float)); return 0; }
    float alpha = (t_query - t0) / dt;
    float alpha2 = alpha * alpha;
    float alpha3 = alpha2 * alpha;
    size_t off0 = (size_t)lo * (dim + 1) + 1;
    size_t off1 = (size_t)hi * (dim + 1) + 1;
    for (size_t i = 0; i < dim; i++) {
        float y0 = traj->state_buffer[off0 + i];
        float y1 = traj->state_buffer[off1 + i];
        /* Hermite三次插值：使用端点值 + 有限差分估计端点导数 */
        float m0, m1;
        if (lo > 0) {
            float ym1 = traj->state_buffer[((size_t)(lo - 1) * (dim + 1) + 1) + i];
            float dt_prev = traj->timestamps[lo] - traj->timestamps[lo - 1];
            m0 = (dt_prev > 1e-10f) ? (y0 - ym1) / dt_prev : 0.0f;
        } else {
            m0 = (y1 - y0) / dt;
        }
        if (hi < traj->count - 1) {
            float y2 = traj->state_buffer[((size_t)(hi + 1) * (dim + 1) + 1) + i];
            float dt_next = traj->timestamps[hi + 1] - traj->timestamps[hi];
            m1 = (dt_next > 1e-10f) ? (y2 - y1) / dt_next : 0.0f;
        } else {
            m1 = (y1 - y0) / dt;
        }
        /* H3: y = h00*y0 + h10*dt*m0 + h01*y1 + h11*dt*m1 */
        float h00 = 2.0f * alpha3 - 3.0f * alpha2 + 1.0f;
        float h10 = alpha3 - 2.0f * alpha2 + alpha;
        float h01 = -2.0f * alpha3 + 3.0f * alpha2;
        float h11 = alpha3 - alpha2;
        state_out[i] = h00 * y0 + h10 * dt * m0 + h01 * y1 + h11 * dt * m1;
    }
    return 0;
}

/* ---- 连续RHS暴露 ---- */

CfCRHSContext* cfc_create_rhs_context(CfCNetwork* network, const float* input) {
    if (!network || !input) return NULL;
    CfCRHSContext* ctx = (CfCRHSContext*)safe_calloc(1, sizeof(CfCRHSContext));
    if (!ctx) return NULL;
    size_t hs = network->config.hidden_size;
    ctx->temp_buffer1 = (float*)safe_calloc(hs, sizeof(float));
    ctx->temp_buffer2 = (float*)safe_calloc(hs, sizeof(float));
    if (!ctx->temp_buffer1 || !ctx->temp_buffer2) {
        cfc_free_rhs_context(ctx);
        return NULL;
    }
    ctx->network = network;
    ctx->input = input;
    return ctx;
}

void cfc_free_rhs_context(CfCRHSContext* ctx) {
    if (!ctx) return;
    safe_free((void**)&ctx->temp_buffer1);
    safe_free((void**)&ctx->temp_buffer2);
    safe_free((void**)&ctx);
}

int cfc_continuous_rhs(float t, const float* y, float* dydt, void* ctx) {
    (void)t;
    CfCRHSContext* rhsc = (CfCRHSContext*)ctx;
    if (!rhsc || !rhsc->network || !y || !dydt) return -1;
    CfCNetwork* net = rhsc->network;
    size_t n = net->config.hidden_size;
    const float* input = rhsc->input;

    /* ZSFABC-P0-015修复: 权重矩阵索引映射修正
     * weight_matrix仅[input_size*hidden_size]，原代码用i*input_size*3+j越界访问
     * 正确应使用: input_gate_weights + hidden_to_gate_weights + gate_biases[3*i]
     *             weight_matrix + hidden_to_activation_weights + bias_vector */

    /* 逐层计算RHS */
    size_t offset = 0;
    for (int l = 0; l < net->config.num_layers && offset < n; l++) {
        CfCCell* cell = net->layers[l];
        if (!cell) continue;
        size_t layer_size = cell->config.hidden_size;
        if (offset + layer_size > n) layer_size = n - offset;
        if (layer_size == 0) break;

        const float* layer_y = y + offset;
        const float* tau = cell->time_constants;
        size_t input_size = cell->config.input_size;

        for (size_t i = 0; i < layer_size; i++) {
            /* 门控: σ(W_gi·x + U_gi·h + b_gi)，使用input_gate_weights */
            float gate_sum = cell->gate_biases[i * 3];  /* 输入门偏置 */
            for (size_t j = 0; j < input_size; j++) {
                gate_sum += cell->input_gate_weights[i * input_size + j] * input[j];
            }
            /* 隐藏到门控: U_gi·h */
            for (size_t k = 0; k < layer_size; k++) {
                gate_sum += cell->hidden_to_gate_weights[i * layer_size + k] * layer_y[k];
            }
            float gate = 1.0f / (1.0f + expf(-gate_sum));

            /* 激活: tanh(W_a·x + U_a·h + b_a) */
            float act_sum = cell->bias_vector[i];
            for (size_t j = 0; j < input_size; j++) {
                act_sum += cell->weight_matrix[i * input_size + j] * input[j];
            }
            /* 隐藏到激活: U_a·h */
            for (size_t k = 0; k < layer_size; k++) {
                act_sum += cell->hidden_to_activation_weights[i * layer_size + k] * layer_y[k];
            }
            float act = tanhf(act_sum);

            /* CfC RHS: dh/dt = (-h + gate·act) / τ */
            float decay_term = -layer_y[i];
            float drive_term = gate * act;
            float rh_tau = tau[i] > 0.001f ? tau[i] : 0.001f;
            dydt[offset + i] = (decay_term + drive_term) / rh_tau;
        }
        offset += layer_size;
    }

    return 0;
}

/* ---- 长时连续演化引擎 ---- */

/** @brief 幂迭代法估算雅可比矩阵最大特征值实部 */
static float power_iteration_max_eigenvalue(const float* J, size_t n, int max_iter, float tol) {
    if (n == 0 || max_iter <= 0) return 0.0f;
    float* v = (float*)safe_calloc(n, sizeof(float));
    float* w = (float*)safe_calloc(n, sizeof(float));
    if (!v || !w) { safe_free((void**)&v); safe_free((void**)&w); return 0.0f; }
    for (size_t i = 0; i < n; i++) v[i] = 1.0f / sqrtf((float)n);
    float lambda = 0.0f;
    for (int iter = 0; iter < max_iter; iter++) {
        for (size_t i = 0; i < n; i++) {
            float s = 0.0f;
            for (size_t j = 0; j < n; j++) s += J[i * n + j] * v[j];
            w[i] = s;
        }
        float norm = 0.0f;
        float new_lambda = 0.0f;
        for (size_t i = 0; i < n; i++) {
            norm += w[i] * w[i];
            new_lambda += v[i] * w[i];
        }
        norm = sqrtf(norm);
        if (norm < 1e-10f) break;
        float inv_norm = 1.0f / norm;
        for (size_t i = 0; i < n; i++) w[i] *= inv_norm;
        if (fabsf(new_lambda - lambda) < tol) { lambda = new_lambda; break; }
        lambda = new_lambda;
        { float* tmp = v; v = w; w = tmp; }
    }
    safe_free((void**)&v);
    safe_free((void**)&w);
    return fabsf(lambda);
}

int cfc_detect_stiffness(CfCNetwork* network, const float* input,
                         const float* state, float* stiffness_ratio) {
    if (!network || !input || !state || !stiffness_ratio) return -1;
    size_t n = network->config.hidden_size;
    if (n == 0) return -1;
    size_t n2 = n * n;
    float* J = (float*)safe_calloc(n2, sizeof(float));
    if (!J) return -1;
    /* 有限差分近似雅可比 J = ∂f/∂y */
    float eps = 1e-4f;
    float* y_plus  = (float*)safe_calloc(n, sizeof(float));
    float* f_plus  = (float*)safe_calloc(n, sizeof(float));
    float* f_minus = (float*)safe_calloc(n, sizeof(float));
    if (!y_plus || !f_plus || !f_minus) {
        safe_free((void**)&J); safe_free((void**)&y_plus); safe_free((void**)&f_plus); safe_free((void**)&f_minus);
        return -1;
    }
    CfCRHSContext* rhsc = cfc_create_rhs_context(network, input);
    if (!rhsc) {
        safe_free((void**)&J); safe_free((void**)&y_plus); safe_free((void**)&f_plus); safe_free((void**)&f_minus);
        return -1;
    }
    memcpy(y_plus, state, n * sizeof(float));
    /* 中心差分：∂f_i/∂y_j ≈ (f_i(y+ε·e_j) - f_i(y-ε·e_j)) / (2ε) */
    for (size_t j = 0; j < n; j++) {
        memcpy(y_plus, state, n * sizeof(float));
        y_plus[j] += eps;
        cfc_continuous_rhs(0.0f, y_plus, f_plus, rhsc);
        y_plus[j] = state[j] - eps;
        cfc_continuous_rhs(0.0f, y_plus, f_minus, rhsc);
        y_plus[j] = state[j];
        for (size_t i = 0; i < n; i++) {
            J[i * n + j] = (f_plus[i] - f_minus[i]) / (2.0f * eps);
        }
    }
    cfc_free_rhs_context(rhsc);
    /* 刚度比 = max|Re(λ)| / min|Re(λ)|，用幂迭代估算最大特征值 */
    float lambda_max = power_iteration_max_eigenvalue(J, n, 100, 1e-6f);
    /* ZSFWS-P9修复: 使用逆幂迭代估算最小特征值，替代不准确的对角近似。
     * 对角近似对非对角占优矩阵偏差极大（可能低估超过10倍）。
     * 逆幂迭代: 求解 (J - shift*I)·x = v, 迭代收敛到最接近shift的特征值。
     * shift=0 时收敛到 |λ| 最小的特征值。 */
    float lambda_min = 1e-10f;
    {
        float* inv_v = (float*)safe_calloc(n, sizeof(float));
        float* inv_w = (float*)safe_calloc(n, sizeof(float));
        if (inv_v && inv_w) {
            for (size_t i = 0; i < n; i++) inv_v[i] = (float)((i * 1103515245u + 12345u) & 0x7fffffff) / 2147483648.0f;
            float norm = 0.0f;
            for (size_t i = 0; i < n; i++) norm += inv_v[i] * inv_v[i];
            if (norm > 1e-10f) { norm = sqrtf(norm); for (size_t i = 0; i < n; i++) inv_v[i] /= norm; }
            float inv_lambda = 0.0f;
            for (int iter = 0; iter < 30; iter++) {
                for (size_t i = 0; i < n; i++) {
                    double sum = 0.0;
                    for (size_t j = 0; j < n; j++)
                        sum += (double)J[i * n + j] * (double)inv_v[j];
                    inv_w[i] = (float)sum;
                }
                float wnorm = 0.0f;
                for (size_t i = 0; i < n; i++) wnorm += inv_w[i] * inv_w[i];
                if (wnorm < 1e-20f) break;
                wnorm = sqrtf(wnorm);
                for (size_t i = 0; i < n; i++) inv_w[i] /= wnorm;
                inv_lambda = 0.0f;
                for (size_t i = 0; i < n; i++) inv_lambda += inv_v[i] * inv_w[i];
                { float* tmp = inv_v; inv_v = inv_w; inv_w = tmp; }
                if (fabsf(inv_lambda) < 1e-10f) break;
            }
            lambda_min = fabsf(inv_lambda);
            if (lambda_min < 1e-10f) lambda_min = 1e-10f;
        }
        safe_free((void**)&inv_v);
        safe_free((void**)&inv_w);
    }
    *stiffness_ratio = lambda_max / lambda_min;
    safe_free((void**)&J);
    safe_free((void**)&y_plus);
    safe_free((void**)&f_plus);
    safe_free((void**)&f_minus);
    return 0;
}

int cfc_set_solver_type(CfCNetwork* network, int solver_type) {
    if (!network) return -1;
    network->config.ode_solver_type = solver_type;
    for (int i = 0; i < network->config.num_layers; i++) {
        if (network->layers[i]) {
            cfc_cell_set_solver_type(network->layers[i], solver_type);
        }
    }
    return 0;
}

int cfc_set_adaptive_step(CfCNetwork* network, int enable) {
    if (!network) return -1;
    for (int i = 0; i < network->config.num_layers; i++) {
        if (network->layers[i]) {
            network->layers[i]->use_adaptive_step = enable;
        }
    }
    return 0;
}

/* ---- 长时连续演化主循环 ---- */

static int cfc_continuous_evolve_impl(CfCNetwork* network, const float* input,
                                      float* hidden_state, float* cell_state,
                                      float* output,
                                      const CfCContinuousConfig* config,
                                      CfCTrajectory* traj,
                                      CfCContinuousStats* stats) {
    if (!network || !input || !hidden_state || !cell_state || !config) return -1;
    size_t hs = network->config.hidden_size;

    CfCContinuousStats local_stats;
    if (!stats) {
        memset(&local_stats, 0, sizeof(local_stats));
        stats = &local_stats;
    }
    memset(stats, 0, sizeof(CfCContinuousStats));

    float t = config->t_start;
    float t_end = config->t_end;
    float h = config->dt_init;
    if (h > config->dt_max) h = config->dt_max;
    if (h < config->dt_min) h = config->dt_min;

    int sign = (t_end >= t) ? 1 : -1;
    int using_stiff_solver = 0;
    int original_solver = network->config.ode_solver_type;
    int current_solver = original_solver;
    int stiffness_recheck_interval = 50; /* 每50步重新检测刚度 */

    /* 初始刚度检测 */
    if (config->enable_stiffness_detect) {
        float sr = 1.0f;
        if (cfc_detect_stiffness(network, input, hidden_state, &sr) == 0) {
            stats->stiffness_ratio = sr;
            if (sr > 100.0f) {
                current_solver = 5; /* ROSENBROCK */
                cfc_set_solver_type(network, current_solver);
                using_stiff_solver = 1;
                stats->solver_switches++;
            } else if (sr > 10.0f) {
                current_solver = 4; /* DP54自适应 */
                cfc_set_solver_type(network, current_solver);
            }
        }
    }

    /* 记录初始轨迹 */
    if (traj) cfc_trajectory_append(traj, t, hidden_state);

    float last_sample_t = t;
    int max_attempts = config->max_steps * 10;

    /* I-011修复：预分配旧状态缓冲区，避免每步malloc */
    float* old_hidden_state = (float*)safe_malloc(hs * sizeof(float));
    if (!old_hidden_state) return -1;

    while (sign * (t_end - t) > 1e-14f && stats->total_steps < config->max_steps &&
           stats->total_steps < max_attempts) {
        /* 计算剩余步长 */
        float remaining = sign * (t_end - t);
        float h_step = (float)fabs(remaining);
        if (h_step > h) h_step = h;
        if (h_step < config->dt_min) h_step = config->dt_min;

        /* 保存前向传播前的旧状态 */
        memcpy(old_hidden_state, hidden_state, hs * sizeof(float));

        /* 前向一步 */
        int ret = cfc_forward(network, input, hidden_state, cell_state, output ? output : hidden_state);
        if (ret != 0) { safe_free((void**)&old_hidden_state); return ret; }

        /* F-006修复：新旧状态变化率误差估计 */
        float max_err = 0.0f;
        for (size_t i = 0; i < hs; i++) {
            float chg = fabsf(hidden_state[i] - old_hidden_state[i]) /
                       (config->abs_tol + config->rel_tol *
                        fmaxf(fabsf(hidden_state[i]), fabsf(old_hidden_state[i])));
            if (chg > max_err) max_err = chg;
        }

        if (max_err <= 1.0f) {
            /* 步长接受 */
            stats->accepted_steps++;
            t += sign * h_step;
            if (h_step > stats->max_dt_used) stats->max_dt_used = h_step;
            if (h_step < stats->min_dt_used || stats->min_dt_used < 1e-12f) stats->min_dt_used = h_step;
            /* PI步长控制 */
            float h_new = h;
            if (max_err > 1e-10f) h_new = h * fminf(2.0f, fmaxf(0.5f, 0.9f / sqrtf(max_err)));
            if (h_new > config->dt_max) h_new = config->dt_max;
            if (h_new < config->dt_min) h_new = config->dt_min;
            h = h_new;

            /* 轨迹记录 */
            if (traj && config->enable_trajectory_output &&
                (config->trajectory_sample_dt <= 0.0f ||
                 fabsf(t - last_sample_t) >= config->trajectory_sample_dt)) {
                cfc_trajectory_append(traj, t, hidden_state);
                last_sample_t = t;
            }
        } else {
            stats->rejected_steps++;
            h *= 0.5f;
            if (h < config->dt_min) h = config->dt_min;
        }

        stats->total_steps++;
        stats->final_t = t;

        /* ZSFWS-P4修复: 周期性刚度重检测。
         * 长时间演化中系统状态可能进入/离开刚性区域，
         * 需要动态切换求解器类型而非仅初始检测一次。 */
        if (config->enable_stiffness_detect &&
            stats->accepted_steps % stiffness_recheck_interval == 0 &&
            stats->accepted_steps > 0) {
            float sr = 1.0f;
            if (cfc_detect_stiffness(network, input, hidden_state, &sr) == 0) {
                stats->stiffness_ratio = sr;
                int target_solver;
                if (sr > 100.0f) {
                    target_solver = 5;
                } else if (sr > 10.0f && current_solver == 5) {
                    target_solver = 4;
                } else if (sr <= 10.0f && current_solver != original_solver) {
                    target_solver = original_solver;
                } else {
                    target_solver = current_solver;
                }
                if (target_solver != current_solver) {
                    cfc_set_solver_type(network, target_solver);
                    current_solver = target_solver;
                    using_stiff_solver = (target_solver == 5);
                    stats->solver_switches++;
                }
            }
        }
    }

    /* ZSFWS-P4修复: 演化结束后恢复原始求解器类型，
     * 避免刚度检测的临时切换影响后续非演化调用。 */
    if (current_solver != original_solver) {
        cfc_set_solver_type(network, original_solver);
    }

    stats->final_t = t;
    stats->avg_dt = fabsf(t - config->t_start) / fmaxf((float)stats->accepted_steps, 1.0f);
    if (stats->min_dt_used < 1e-12f) stats->min_dt_used = h;
    stats->used_stiff_solver = using_stiff_solver;

    /* ZSFZS-F012修复: 使用专门的truncated标志位替代rejected_steps负数编码。
     * 语义分离：rejected_steps只表示被自适应步长拒绝的步数，
     * truncated专门标记是否因达到最大步数上限而未完成演化。 */
    if (stats->total_steps >= config->max_steps)
        stats->truncated = 1;

    /* 最终轨迹点 */
    if (traj) cfc_trajectory_append(traj, t, hidden_state);

    safe_free((void**)&old_hidden_state);
    return 0;
}

int cfc_continuous_evolve(CfCNetwork* network, const float* input,
                          float* hidden_state, float* cell_state,
                          float* output,
                          const CfCContinuousConfig* config,
                          CfCContinuousStats* stats) {
    return cfc_continuous_evolve_impl(network, input, hidden_state, cell_state,
                                      output, config, NULL, stats);
}

int cfc_continuous_evolve_with_trajectory(CfCNetwork* network, const float* input,
                                          float* hidden_state, float* cell_state,
                                          float* output,
                                          const CfCContinuousConfig* config,
                                          CfCTrajectory* traj,
                                          CfCContinuousStats* stats) {
    return cfc_continuous_evolve_impl(network, input, hidden_state, cell_state,
                                      output, config, traj, stats);
}

