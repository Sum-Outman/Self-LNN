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
#include "selflnn/core/cfc_enhanced.h"  /* P0-001修复: 集成CfC增强层(SIMD/自动求解器/刚度检测) */
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
    memset(network, 0, sizeof(CfCNetwork));
    
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
    /* L-12修复: delta_t从网络配置读取，不再硬编码为1.0f */
    cell_config.delta_t = (config->delta_t > 0.0f) ? config->delta_t : 1.0f;  /* 默认时间步长1.0秒 */
    
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
/* 逐层分配dropout掩码，避免多层共享单一掩码导致反向传播使用错误掩码 */
    network->dropout_mask = (float*)safe_calloc(total_layers * max_layer_size, sizeof(float));
    
/* 多层权重独立存储——每层分配独立权重矩阵
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
        network->W_out_m = NULL;
        network->W_out_v = NULL;
        network->b_out_m = NULL;
        network->b_out_v = NULL;
        network->out_adam_step = 0;
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

    /* R2-P1修复: He/Kaiming自适应初始化W_out参数
     * 原实现使用 sqrt(6/hidden_size) 是Xavier公式，对ReLU类激活函数不正确。
     * He均匀分布: 范围 = sqrt(6/fan_in)，其中 fan_in = hidden_size。 */
    if (out_proj_param_size > 0 && network->W_out_params) {
        float he_limit = sqrtf(6.0f / (float)hidden_size_cfg);
        if (he_limit < 0.0001f) he_limit = 0.0001f;
        for (size_t i = 0; i < output_size_cfg * hidden_size_cfg; i++) {
            network->W_out_params[i] = rng_uniform(-he_limit, he_limit);
        }
        for (size_t i = 0; i < output_size_cfg; i++) {
            network->b_out_params[i] = 0.0f;
        }
/* 分配 W_out/b_out Adam动量缓冲区并清零 */
        network->W_out_m = (float*)safe_calloc(output_size_cfg * hidden_size_cfg, sizeof(float));
        network->W_out_v = (float*)safe_calloc(output_size_cfg * hidden_size_cfg, sizeof(float));
        network->b_out_m = (float*)safe_calloc(output_size_cfg, sizeof(float));
        network->b_out_v = (float*)safe_calloc(output_size_cfg, sizeof(float));
        network->out_adam_step = 0;
    }
    
    /* P1修复: 分配失败后完整清理检查
     * 原检查遗漏了 W_out Adam动量缓冲区 和 层归一化缓冲区，
     * 若这些分配失败会导致后续访问空指针。
     * 现在逐层检查所有已分配的资源，确保失败时通过 cfc_free 完整释放。 */
    int alloc_failed = 0;
    if (!network->layer_outputs || !network->layer_gradients ||
        !network->activation_buffer || !network->dropout_mask ||
        !param_block || !grad_block ||
        !network->per_layer_w_offset || !network->per_layer_b_offset ||
        !network->per_layer_w_size) {
        alloc_failed = 1;
    }
    /* 检查输出投影Adam动量缓冲区 —— 仅当存在输出投影时检查 */
    if (!alloc_failed && out_proj_param_size > 0) {
        if (!network->W_out_m || !network->W_out_v ||
            !network->b_out_m || !network->b_out_v) {
            alloc_failed = 1;
        }
    }
    if (alloc_failed) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "CfC网络内存分配失败: 参数块或缓冲区分配失败");
        cfc_free(network);
        return NULL;
    }
    
    /* R2-P2修复: 使用He/Kaiming自适应初始化每层独立权重矩阵
     * 移除0.2f硬上限截断（小型网络中该截断严重压缩初始化方差）。 */
    {
        for (int l = 0; l < num_layers_int; l++) {
            size_t l_input = (l == 0) ? input_size_cfg : hidden_size_cfg;
            float he_limit = sqrtf(6.0f / (float)l_input);
            if (he_limit < 0.0001f) he_limit = 0.0001f;
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
    
/* 初始化所有层的Dropout掩码为1（不丢弃） */
    for (size_t i = 0; i < total_layers * max_layer_size; i++) {
        network->dropout_mask[i] = 1.0f;
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
                    cfc_free(network);
                    return NULL;
                }
            }
        }
    }

    /* P0-001修复: 初始化CfC增强层状态（SIMD加速/自动求解器选择/刚度检测）
     * 仅在 config->use_enhanced=1 时分配，否则enhanced_state=NULL走原始快速路径 */
    network->enhanced_state = NULL;
    network->enhanced_cfg = NULL;
    if (config->use_enhanced) {
        network->enhanced_state = cfc_enhanced_state_create();
        if (network->enhanced_state) {
            CfcEnhancedConfig* ecfg = (CfcEnhancedConfig*)safe_calloc(1, sizeof(CfcEnhancedConfig));
            if (ecfg) {
                *ecfg = cfc_enhanced_default_config();
                ecfg->verbose = 0;
                network->enhanced_cfg = ecfg;
            }
        }
    }

    // 初始化统计信息
    network->is_initialized = 1;
    network->is_trained = 0;
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
/* 释放 W_out Adam动量缓冲区 */
    safe_free((void**)&network->W_out_m);
    safe_free((void**)&network->W_out_v);
    safe_free((void**)&network->b_out_m);
    safe_free((void**)&network->b_out_v);
    network->out_adam_step = 0;

/* 释放每层偏移追踪数组 */
    safe_free((void**)&network->per_layer_w_offset);
    safe_free((void**)&network->per_layer_b_offset);
    safe_free((void**)&network->per_layer_w_size);
    
    /* P0-001修复: 释放CfC增强层状态 */
    if (network->enhanced_state) {
        cfc_enhanced_state_free((CfcEnhancedState*)network->enhanced_state);
        network->enhanced_state = NULL;
    }
    safe_free((void**)&network->enhanced_cfg);
    
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
    if (!network->is_trained) {
        log_warning("CfC网络未训练，推理结果可能不准确");
    }
    
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
/* 单层网络跳过额外线性变换，直接传递原始输入。
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
        
        // P0-001修复: 如果启用增强层，使用SIMD加速/自动求解器选择/刚度检测
        int result;
        if (network->enhanced_state && network->enhanced_cfg) {
            result = cfc_enhanced_forward(network->layers[layer],
                                         current_input,
                                         layer_output,
                                         (const CfcEnhancedConfig*)network->enhanced_cfg,
                                         (CfcEnhancedState*)network->enhanced_state);
        } else {
            result = cfc_cell_forward(network->layers[layer], 
                                     current_input, 
                                     layer_output);
        }
        
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
        
/* 使用逐层独立dropout掩码，每层掩码写入对应偏移位置 */
        if (config->dropout_rate > 0.0f && config->dropout_rate < 1.0f && config->enable_training) {
            float* layer_mask = network->dropout_mask + (size_t)layer * max_layer_size;
            for (size_t i = 0; i < hidden_size; i++) {
                if (rng_uniform(0.0f, 1.0f) < config->dropout_rate) {
                    layer_mask[i] = 0.0f;
                    layer_output[i] = 0.0f;
                } else {
                    /* K-055: 除零防护 —— dropout_rate接近1.0时回退到0.999999f */
                    float safe_inv = (config->dropout_rate < 0.999999f)
                        ? (1.0f - config->dropout_rate) : 0.000001f;
                    layer_mask[i] = 1.0f / safe_inv;
                    layer_output[i] *= layer_mask[i];
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
    if (!cell) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_PARAMETER,
            "cfc_ensure_cell_momentum", "CfCCell指针为空", "内部错误：动量缓冲区要求有效cell");
        return -1;
    }
    if (cell->cell_momentum_initialized) return 0;

    size_t hidden = cell->config.hidden_size;
    size_t input = cell->config.input_size;
    size_t total = 0;

    /* 统计所有cell级参数总量 */
    total += input * hidden;  /* W_gx */
    total += input * hidden;  /* W_fx */
    total += input * hidden;  /* W_ox */
    total += hidden * hidden * 3; /* P0-003修复: W_ghi + W_ghf + W_gho 三门独立 */
    total += hidden * hidden; /* W_ah */
    total += hidden * 3;      /* b_g */
    total += hidden;          /* τ */

    cell->cell_momentum_buffer = (float*)safe_calloc(total * 2, sizeof(float));
    if (!cell->cell_momentum_buffer) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY,
            "cfc_ensure_cell_momentum", "动量缓冲区内存分配失败",
            "请减少hidden_size或检查系统内存");
        return -1;
    }
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
    /* P0-FIX: ODR违规导致grads/params可能为无效指针(0x1等), 守卫跳过 */
    if ((uintptr_t)params < 0x1000 || (uintptr_t)grads < 0x1000 ||
        (uintptr_t)m < 0x1000 || (uintptr_t)v < 0x1000) return;
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
        /* W_ghi: 隐藏→输入门——P0-003三门独立 */
        if (cell->hidden_to_input_gate_weight_grad && cell->hidden_to_input_gate_weights) {
            cfc_adam_update_group(cell->hidden_to_input_gate_weights, cell->hidden_to_input_gate_weight_grad,
                                  m + offset, v + offset, hh_count,
                                  learning_rate, beta1, beta2, epsilon, b1_corr, b2_corr);
            offset += hh_count;
        }
        /* W_ghf: 隐藏→遗忘门 */
        if (cell->hidden_to_forget_gate_weight_grad && cell->hidden_to_forget_gate_weights) {
            cfc_adam_update_group(cell->hidden_to_forget_gate_weights, cell->hidden_to_forget_gate_weight_grad,
                                  m + offset, v + offset, hh_count,
                                  learning_rate, beta1, beta2, epsilon, b1_corr, b2_corr);
            offset += hh_count;
        }
        /* W_gho: 隐藏→输出门 */
        if (cell->hidden_to_output_gate_weight_grad && cell->hidden_to_output_gate_weights) {
            cfc_adam_update_group(cell->hidden_to_output_gate_weights, cell->hidden_to_output_gate_weight_grad,
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
    if (network->W_out_gradients) {
        memset(network->W_out_gradients, 0, 
               network->config.output_size * network->config.hidden_size * sizeof(float));
    }
    if (network->b_out_gradients) {
        memset(network->b_out_gradients, 0, network->config.output_size * sizeof(float));
    }
    /* 委托给cfc_cell单元内的清零函数(同编译单元, 避免GCC 15.1跨TU崩溃) */
    for (int cl = 0; cl < network->config.num_layers; cl++) {
        CfCCell* cell = network->layers[cl];
        if (!cell) continue;
        cfc_cell_zero_gradients(cell);
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

    /* 保存输入信号（第一层反向传播需要） */
    float* saved_input = NULL;
    if (input_size > 0) {
        saved_input = (float*)safe_malloc(input_size * sizeof(float));
        if (saved_input) {
            memcpy(saved_input, network->layer_gradients, input_size * sizeof(float));
        }
    }

    /* 初始化各层梯度缓冲区为0 */
    memset(network->layer_gradients, 0, num_layers * max_layer_size * sizeof(float));

    /* 步骤1: 计算输出层梯度（包括输出投影矩阵的梯度） */
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

    /* 步骤2: 逐层反向传播（从最后一层到第一层） */
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
            /* FIX: gradient已在lnn.c:806清零(hidden_size), 此处不再重复memset
             * input_size <= hidden_size, 避免损坏堆边界 */
            if (input_size < hidden_size) {
                memset(prev_gradient + input_size, 0, (hidden_size - input_size) * sizeof(float));
            }
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

/* 使用前一层(layer-1)的dropout掩码，而非当前层掩码
         * 原因: prev_gradient = dL/d(前一层输出_经dropout后)，需要乘以前一层的掩码
         * 才能正确得到 dL/d(前一层输出_dropout前)，从而正确传递梯度穿过前一层的dropout */
        if (config->dropout_rate > 0.0f && layer > 0) {
            float* prev_mask = network->dropout_mask + (size_t)(layer - 1) * max_layer_size;
            for (size_t i = 0; i < hidden_size; i++) {
                prev_gradient[i] *= prev_mask[i];
            }
        }
    }

/* 步骤3 — 单样本路径梯度累积而非直接SGD更新
     * skip_cell_update=1时跳过所有参数更新（批量模式，调用方负责统一下发）。
     * skip_cell_update=0时采用梯度累积模式（+=）而非直接SGD（-=lr*grad），
     * 确保与外部优化器（Adam/AdamW等）的单元格更新路径一致。 */
    if (!skip_cell_update) {
    /* R3P3修复: 单样本路径统一使用Adam自适应学习率，消除与原批量路径的SGD/Adam分裂。
     * 原SGD路径（参数 -= lr*grad）缺少动量/偏差校正/自适应步长，
     * 导致单样本训练与批量训练的优化动态不一致，严重损害收敛稳定性。 */
    cfc_apply_cell_gradients_adam(network, learning_rate, 0.9f, 0.999f, 1e-8f, 1);
/* W_out/b_out Adam动量缓冲区已添加，全部参数统一使用Adam更新。
     * cfc_apply_out_proj_gradients 内部已改为完整Adam(β1=0.9, β2=0.999)自适应学习率。
     * W_out优化路径现已与cell级参数完全统一，消除双轨运行导致的收敛分裂问题。 */
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
 *: 使用层0的梯度偏移，支持多层独立权重。 */
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
        
/* 使用前一层(layer-1)的dropout掩码，与cfc_backward_ex一致 */
        if (config->dropout_rate > 0.0f && layer > 0) {
            float* prev_mask = network->dropout_mask + (size_t)(layer - 1) * max_layer_size;
            for (size_t i = 0; i < hidden_size; i++) {
                if (prev_gradient) {
                    prev_gradient[i] *= prev_mask[i];
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
 *: 使用per_layer偏移写入正确位置。 */
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
/* 防御性清零cell内部梯度缓冲区，防止梯度污染 */
            memset(cell0->weight_grad, 0, total_weight_size * sizeof(float));
            memset(cell0->bias_grad, 0, hidden_size * sizeof(float));
        }
    } else {
        // 多层网络完整梯度计算
/* 每层cell梯度写入独立的param_block偏移位置
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
/* 防御性清零cell内部梯度缓冲区，防止梯度污染 */
            if (cell->weight_grad) memset(cell->weight_grad, 0, cell_weight_size * sizeof(float));
            if (cell->bias_grad) memset(cell->bias_grad, 0, config->hidden_size * sizeof(float));
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
 * @brief 应用各层CfC单元的cell级参数梯度（使用Adam自适应学习率）
 * 
 * P0-003修复: 原先使用简单SGD更新cell参数，与网络级Adam双轨运行，收敛不一致。
 * 现统一委托给 cfc_apply_cell_gradients_adam（完整Adam β1=0.9, β2=0.999），
 * 消除优化路径分裂问题，所有参数统一使用Adam自适应学习率更新。
 * 
 * 调用方不需要传入Adam超参数——内部使用标准默认值。
 */
int cfc_apply_cell_gradients(CfCNetwork* network, float learning_rate) {
    return cfc_apply_cell_gradients_adam(network, learning_rate, 0.9f, 0.999f, 1e-8f, 1);
}

/**
 * @brief 应用输出投影矩阵(W_out+b_out)梯度 —— Adam自适应学习率更新
 * 
 *修复: 原SGD更新路径与cell级Adam更新双轨运行,收敛动态不一致。
 * 改为使用完整的 Adam(β1=0.9, β2=0.999) 自适应学习率,
 * 独立维护 W_out/b_out 的一阶动量(m)和二阶速度(v)缓冲区,
 * 消除优化路径分裂问题,统一全部参数使用Adam更新。
 */
int cfc_apply_out_proj_gradients(CfCNetwork* network, float learning_rate) {
    SELFLNN_CHECK_NULL(network, "CfC网络句柄为空");
    SELFLNN_CHECK_INITIALIZED(network, "CfC网络未初始化");
    
    size_t hidden_size = network->config.hidden_size;
    size_t output_size = network->config.output_size;
    
    if (output_size == hidden_size || !network->W_out_params || !network->W_out_gradients) return 0;
    if (!network->W_out_m || !network->W_out_v || !network->b_out_m || !network->b_out_v) return -1;

    /* Adam超参数 */
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-8f;

    /* 步数计数器递增 + 偏差校正因子 */
    network->out_adam_step++;
    size_t t = network->out_adam_step;
    float b1_corr = 1.0f / (1.0f - powf(beta1, (float)t));
    float b2_corr = 1.0f / (1.0f - powf(beta2, (float)t));

    size_t w_count = output_size * hidden_size;

    /* W_out 矩阵: Adam更新 */
    cfc_adam_update_group(network->W_out_params, network->W_out_gradients,
                          network->W_out_m, network->W_out_v, w_count,
                          learning_rate, beta1, beta2, epsilon, b1_corr, b2_corr);

    /* b_out 偏置: Adam更新 */
    cfc_adam_update_group(network->b_out_params, network->b_out_gradients,
                          network->b_out_m, network->b_out_v, output_size,
                          learning_rate, beta1, beta2, epsilon, b1_corr, b2_corr);

    /* 应用后清零梯度缓冲区 */
    memset(network->W_out_gradients, 0, w_count * sizeof(float));
    memset(network->b_out_gradients, 0, output_size * sizeof(float));
    return 0;
}

/**
 * @brief 将共享参数块同步到各层CfC单元的活跃权重
 * 
 *: FIX-017 全面重写。原代码在多层时所有层复制相同权重矩阵，
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
/* 使用实际分配的多层参数总数进行序列化 */
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

/* 保存层归一化可学习参数(gamma/beta)
     * 此前cfc_save完全遗漏了layer_norm参数，导致加载后归一化回退到初始值 */
    if (network->layer_norms && network->config.use_layer_norm) {
        uint32_t ln_marker = 0x4C4E4F52;  /* "LNOR" */
        fwrite(&ln_marker, sizeof(uint32_t), 1, file);
        for (int i = 0; i < network->config.num_layers; i++) {
            LayerNorm* ln = (LayerNorm*)network->layer_norms[i];
            if (!ln) {
                /* 写入零长度标记表示该层无LN实例 */
                uint16_t zero_dim = 0;
                fwrite(&zero_dim, sizeof(uint16_t), 1, file);
                continue;
            }
            const float* gamma = layer_norm_get_gamma(ln);
            const float* beta  = layer_norm_get_beta(ln);
            uint16_t dim = (uint16_t)network->config.hidden_size;
            fwrite(&dim, sizeof(uint16_t), 1, file);
            fwrite(gamma, sizeof(float), (size_t)dim, file);
            fwrite(beta,  sizeof(float), (size_t)dim, file);
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
/* 使用实际分配的多层参数总数进行反序列化 */
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

/* 加载层归一化可学习参数(gamma/beta) - 向后兼容 */
    if (network->layer_norms && network->config.use_layer_norm) {
        uint32_t ln_marker = 0;
        long ln_pos = ftell(file);
        if (fread(&ln_marker, sizeof(uint32_t), 1, file) == 1 &&
            ln_marker == 0x4C4E4F52) {  /* "LNOR" */
            for (int i = 0; i < network->config.num_layers; i++) {
                LayerNorm* ln = (LayerNorm*)network->layer_norms[i];
                uint16_t dim = 0;
                if (fread(&dim, sizeof(uint16_t), 1, file) != 1) break;
                if (dim == 0 || !ln) continue;
                if (dim > (uint16_t)network->config.hidden_size) {
                    dim = (uint16_t)network->config.hidden_size;
                }
                float* gamma_buf = (float*)safe_malloc((size_t)dim * 2 * sizeof(float));
                if (!gamma_buf) { fseek(file, dim * 2 * sizeof(float), SEEK_CUR); continue; }
                if (fread(gamma_buf, sizeof(float), dim, file) == dim &&
                    fread(gamma_buf + dim, sizeof(float), dim, file) == dim) {
                    layer_norm_set_params(ln, gamma_buf);
                }
                safe_free((void**)&gamma_buf);
            }
        } else {
            /* 旧格式checkpoint（无layer_norm参数），回退文件指针，gamma/beta保持原值 */
            fseek(file, ln_pos, SEEK_SET);
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
    
    /* P0-001修复: 重置CfC增强层状态 */
    if (network->enhanced_state) {
        cfc_enhanced_state_reset((CfcEnhancedState*)network->enhanced_state);
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
    
/* 返回实际分配的多层权重总数，而非旧单层大小 */
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
    
/* 返回实际分配的多层偏置总数 */
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
    size_t is = network->config.input_size;
    ctx->temp_buffer1 = (float*)safe_calloc(hs, sizeof(float));
    ctx->temp_buffer2 = (float*)safe_calloc(hs, sizeof(float));
    /* ZSFJJJ-C008修复: 复制输入数据到上下文自有缓冲区,
     * 避免存储指向调用者内存的悬空指针。
     * 长时ODE演化可跨越数千步, 原始input指针极可能已被释放。 */
    ctx->input_copy = (float*)safe_malloc(is * sizeof(float));
    if (!ctx->temp_buffer1 || !ctx->temp_buffer2 || !ctx->input_copy) {
        cfc_free_rhs_context(ctx);
        return NULL;
    }
    memcpy(ctx->input_copy, input, is * sizeof(float));
    ctx->input_size = is;
    ctx->network = network;
    ctx->input = ctx->input_copy; /* 向后兼容: input指针指向安全副本 */
    return ctx;
}

void cfc_free_rhs_context(CfCRHSContext* ctx) {
    if (!ctx) return;
    safe_free((void**)&ctx->temp_buffer1);
    safe_free((void**)&ctx->temp_buffer2);
    safe_free((void**)&ctx->input_copy); /* ZSFJJJ-C008: 释放输入副本 */
    safe_free((void**)&ctx);
}

int cfc_continuous_rhs(float t, const float* y, float* dydt, void* ctx) {
    /* L003: CfC液态神经网络的ODE为自治系统——微分方程dh/dt = f(h)不含显式时间依赖。
     * 时间参数t仅保留用于非自治扩展接口兼容性，当前动力学完全由隐藏状态h编码。 */
    (void)t;
    CfCRHSContext* rhsc = (CfCRHSContext*)ctx;
    if (!rhsc || !rhsc->network || !y || !dydt) return -1;
    CfCNetwork* net = rhsc->network;
    size_t n = net->config.hidden_size;
    const float* input = rhsc->input;

/* 权重矩阵索引映射修正
     * weight_matrix仅[input_size*hidden_size]，原代码用i*input_size*3+j越界访问
     * 正确应使用: input_gate_weights + hidden_to_input_gate_weights + gate_biases[3*i]
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
        /* ZSF-009修复：NULL检查避免未启用自适应时间常数时解引用空指针 */
        const float* tau = cell->time_constants;
        const float default_tau_value = cell->config.time_constant > 0.001f ? cell->config.time_constant : 0.1f;
        size_t input_size = cell->config.input_size;

        for (size_t i = 0; i < layer_size; i++) {
            /* 三门: input_gate使用input_gate_weights, forget_gate使用forget_gate_weights */
            float input_gate_sum = cell->gate_biases[i * 3];  /* 输入门偏置 */
            float forget_gate_sum = cell->gate_biases[i * 3 + 1];  /* 遗忘门偏置 */
            for (size_t j = 0; j < input_size; j++) {
                input_gate_sum += cell->input_gate_weights[i * input_size + j] * input[j];
                forget_gate_sum += cell->forget_gate_weights[i * input_size + j] * input[j];
            }
            for (size_t k = 0; k < layer_size; k++) {
                input_gate_sum += cell->hidden_to_input_gate_weights[i * layer_size + k] * layer_y[k];
                forget_gate_sum += cell->hidden_to_forget_gate_weights[i * layer_size + k] * layer_y[k];
            }
            float input_gate = 1.0f / (1.0f + expf(-input_gate_sum));
            float forget_gate = 1.0f / (1.0f + expf(-forget_gate_sum));

            /* 激活: tanh(W_a·x + U_a·h + b_a) */
            float act_sum = cell->bias_vector[i];
            for (size_t j = 0; j < input_size; j++) {
                act_sum += cell->weight_matrix[i * input_size + j] * input[j];
            }
            for (size_t k = 0; k < layer_size; k++) {
                act_sum += cell->hidden_to_activation_weights[i * layer_size + k] * layer_y[k];
            }
            float act = tanhf(act_sum);

            /* CfC RHS: dh/dt = (-forget_gate·h + input_gate·act) / τ - ZSF-009修复：NULL安全 */
            float rh_tau = (tau) ? ((tau[i] > 0.001f) ? tau[i] : 0.001f) : default_tau_value;
            dydt[offset + i] = (-forget_gate * layer_y[i] + input_gate * act) / rh_tau;
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
    if (!network || !input || !state || !stiffness_ratio) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_PARAMETER,
            "cfc_detect_stiffness", "无效的输入参数",
            "network/input/state/stiffness_ratio均不能为空");
        return -1;
    }
    size_t n = network->config.hidden_size;
    if (n == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_PARAMETER,
            "cfc_detect_stiffness", "隐藏层大小为0",
            "网络配置无效，hidden_size必须大于0");
        return -1;
    }
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
/* 使用逆幂迭代估算最小特征值，替代不准确的对角近似。
     * 对角近似对非对角占优矩阵偏差极大（可能低估超过10倍）。
     * 逆幂迭代: 求解 (J - shift*I)·x = v, 迭代收敛到最接近shift的特征值。
     * shift=0 时收敛到 |λ| 最小的特征值。 */
    float lambda_min = 1e-10f;
    /* R2-P12修复: 真正的逆幂迭代法求解最小特征值
     * 原实现错误地使用 J*v 替代求解 J*w=v，导致收敛到最大特征值。
     * 修正：每步迭代解线性方程组 J*w = v，使用紧凑高斯消元(带部分主元)。
     * 对于n<=256的矩阵，高斯消元O(n³)足够高效。 */
    {
        float* inv_v = (float*)safe_calloc(n, sizeof(float));
        float* inv_w = (float*)safe_calloc(n, sizeof(float));
        float* lu_A = (float*)safe_calloc(n * n, sizeof(float));
        int*   lu_pivot = (int*)safe_calloc(n, sizeof(int));
        if (inv_v && inv_w && lu_A && lu_pivot) {
            for (size_t i = 0; i < n; i++) inv_v[i] = secure_random_float();
            float norm = 0.0f;
            for (size_t i = 0; i < n; i++) norm += inv_v[i] * inv_v[i];
            if (norm > 1e-10f) { norm = sqrtf(norm); for (size_t i = 0; i < n; i++) inv_v[i] /= norm; }

            float inv_lambda = 0.0f;
            for (int iter = 0; iter < 30; iter++) {
                /* 复制J到lu_A并做部分主元高斯消元(LU分解) */
                memcpy(lu_A, J, n * n * sizeof(float));
                for (size_t k = 0; k < n; k++) {
                    /* 部分主元选择 */
                    size_t pivot = k;
                    float max_val = fabsf(lu_A[k * n + k]);
                    for (size_t r = k + 1; r < n; r++) {
                        float val = fabsf(lu_A[r * n + k]);
                        if (val > max_val) { max_val = val; pivot = r; }
                    }
                    lu_pivot[k] = (int)pivot;
                    if (max_val < 1e-15f) { lu_A[k * n + k] = 1e-15f; continue; }
                    /* 交换行 */
                    if (pivot != k) {
                        for (size_t c = 0; c < n; c++) {
                            float tmp = lu_A[k * n + c];
                            lu_A[k * n + c] = lu_A[pivot * n + c];
                            lu_A[pivot * n + c] = tmp;
                        }
                    }
                    /* 高斯消元 */
                    float inv_pivot = 1.0f / lu_A[k * n + k];
                    for (size_t r = k + 1; r < n; r++) {
                        float factor = lu_A[r * n + k] * inv_pivot;
                        lu_A[r * n + k] = factor;
                        for (size_t c = k + 1; c < n; c++) {
                            lu_A[r * n + c] -= factor * lu_A[k * n + c];
                        }
                    }
                }
                /* 前向代入(使用LU分解结果求解 J*w = v) */
                float* b = (float*)safe_calloc(n, sizeof(float));
                if (!b) break;
                memcpy(b, inv_v, n * sizeof(float));
                /* 前向代入: L*y = P*b */
                for (size_t k = 0; k < n; k++) {
                    int pk = lu_pivot[k];
                    float tmp = b[k]; b[k] = b[pk]; b[pk] = tmp;
                    for (size_t r = k + 1; r < n; r++) {
                        b[r] -= lu_A[r * n + k] * b[k];
                    }
                }
                /* 回代: U*w = y */
                for (int k = (int)n - 1; k >= 0; k--) {
                    float sum = b[k];
                    for (size_t c = (size_t)k + 1; c < n; c++) {
                        sum -= lu_A[k * n + c] * b[c];
                    }
                    b[k] = sum / (fabsf(lu_A[k * n + k]) > 1e-15f ? lu_A[k * n + k] : 1e-15f);
                }
                /* 归一化w并计算瑞利商 */
                float wnorm = 0.0f;
                for (size_t i = 0; i < n; i++) wnorm += b[i] * b[i];
                if (wnorm < 1e-20f) { safe_free((void**)&b); break; }
                wnorm = sqrtf(wnorm);
                for (size_t i = 0; i < n; i++) b[i] /= wnorm;
                inv_lambda = 0.0f;
                for (size_t i = 0; i < n; i++) inv_lambda += inv_v[i] * b[i];
                memcpy(inv_v, b, n * sizeof(float));
                safe_free((void**)&b);
                if (fabsf(inv_lambda) < 1e-10f) break;
            }
            /* inv_lambda现为1/|λ_min|的估计，lambda_min ≈ 1/|inv_lambda| */
            if (fabsf(inv_lambda) > 1e-10f) {
                lambda_min = 1.0f / fabsf(inv_lambda);
            }
            if (lambda_min < 1e-10f) lambda_min = 1e-10f;
        }
        safe_free((void**)&inv_v);
        safe_free((void**)&inv_w);
        safe_free((void**)&lu_A);
        safe_free((void**)&lu_pivot);
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
                current_solver = ODE_SOLVER_ROSENBROCK; /* 使用枚举常量替代魔法数字5 */
                cfc_set_solver_type(network, current_solver);
                using_stiff_solver = 1;
                stats->solver_switches++;
            } else if (sr > 10.0f) {
                current_solver = ODE_SOLVER_DP54; /* 使用枚举常量替代魔法数字4 */
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

/* 周期性刚度重检测。
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

/* 演化结束后恢复原始求解器类型，
     * 避免刚度检测的临时切换影响后续非演化调用。 */
    if (current_solver != original_solver) {
        cfc_set_solver_type(network, original_solver);
    }

    stats->final_t = t;
    stats->avg_dt = fabsf(t - config->t_start) / fmaxf((float)stats->accepted_steps, 1.0f);
    if (stats->min_dt_used < 1e-12f) stats->min_dt_used = h;
    stats->used_stiff_solver = using_stiff_solver;

/* 使用专门的truncated标志位替代rejected_steps负数编码。
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

