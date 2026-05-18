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
#include "selflnn/utils/math_utils.h"
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
    
    // 分配权重和偏置
    size_t weight_size = config->hidden_size * config->hidden_size;
    if (config->num_layers == 1) {
        // 单层网络：输入到隐藏
        weight_size = config->input_size * config->hidden_size;
    }
    
    // 分配连续参数块（权重矩阵 + 偏置向量连续存储，确保lnn_get_parameters返回完整参数）
    size_t total_param_size = weight_size + config->hidden_size;
    float* param_block = (float*)safe_calloc(total_param_size, sizeof(float));
    network->weight_matrix = param_block;
    network->bias_vector = param_block ? param_block + weight_size : NULL;
    
    // 分配连续梯度块（权重梯度 + 偏置梯度 + 输出投影矩阵W_out + b_out连续存储）
    // W_out: [output_size × hidden_size], b_out: [output_size]
    size_t out_proj_size = (config->output_size != config->hidden_size) ? 
        (config->output_size * config->hidden_size + config->output_size) : 0;
    size_t total_grad_size = weight_size + config->hidden_size + out_proj_size;
    float* grad_block = (float*)safe_calloc(total_grad_size, sizeof(float));
    network->weight_gradients = grad_block;
    network->bias_gradients = grad_block ? grad_block + weight_size : NULL;

    // 初始化输出投影矩阵（如果输出维度不同于隐藏维度）
    if (out_proj_size > 0 && grad_block) {
        float* W_out = grad_block + weight_size + config->hidden_size;
        float* b_out = W_out + config->output_size * config->hidden_size;
        for (size_t i = 0; i < config->output_size * config->hidden_size; i++) {
            W_out[i] = rng_uniform(-0.1f, 0.1f);
        }
        for (size_t i = 0; i < config->output_size; i++) {
            b_out[i] = 0.0f;
        }
    }
    
    // 检查内存分配
    if (!network->layer_outputs || !network->layer_gradients ||
        !network->activation_buffer || !network->dropout_mask ||
        !param_block || !grad_block) {
        cfc_free(network);
        return NULL;
    }
    
    // 初始化权重为小随机值
    for (size_t i = 0; i < weight_size; i++) {
        network->weight_matrix[i] = rng_uniform(-0.1f, 0.1f);
    }
    
    // 初始化偏置为零
    for (size_t i = 0; i < config->hidden_size; i++) {
        network->bias_vector[i] = 0.0f;
    }
    
    // 初始化Dropout掩码
    for (size_t i = 0; i < max_layer_size; i++) {
        network->dropout_mask[i] = 1.0f;  // 初始全为1（不丢弃）
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
    
    safe_free((void**)&network->weight_matrix);
    network->bias_vector = NULL;
    safe_free((void**)&network->weight_gradients);
    network->bias_gradients = NULL;
    
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
        // 单层网络：保存原始输入到layer_gradients缓冲区的前input_size个元素中（用于反向传播）
        // 注意：layer_gradients的大小为total_layers * max_layer_size，所以前input_size个元素是安全的
        memcpy(network->layer_gradients, input, input_size * sizeof(float));
        
        // 直接应用权重和偏置
        matrix_vector_multiply_raw(current_input, network->weight_matrix, input, 
                                 hidden_size, input_size);
        
        // 添加偏置
        for (size_t i = 0; i < hidden_size; i++) {
            current_input[i] += network->bias_vector[i];
        }
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
    
    // 应用输出投影矩阵：output = W_out · hidden_state + b_out
    // K-052: 修复输出投影矩阵偏移量Bug —— 原代码多加了hidden_size导致读取零值
    // 梯度块布局: [weight_grads: w_size] [bias_grads: hidden_size] [W_out: out*hid] [b_out: out]
    // W_out起始偏移 = w_size + hidden_size (不是 w_size + hidden_size + hidden_size)
    size_t w_size = input_size * hidden_size;
    float* W_out = network->weight_gradients + w_size + hidden_size;
    float* b_out = W_out + output_size * hidden_size;

    if (output_size != hidden_size && network->weight_gradients) {
        for (size_t i = 0; i < output_size; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < hidden_size; j++) {
                sum += W_out[i * hidden_size + j] * current_hidden[j];
            }
            output[i] = sum + b_out[i];
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
 * @brief 反向传播（训练）- 完整实现
 * 支持多层CfC网络和输出投影矩阵的完整梯度链
 */
int cfc_backward(CfCNetwork* network, const float* error, 
                 float* gradient, float learning_rate) {
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
    
    if (output_size != hidden_size && network->weight_gradients) {
        /* 输出投影矩阵反向传播：dL/dh = W_out^T · dL/do */
        /* K-052: 修复与forward一致的W_out偏移量 */
        size_t w_size = input_size * hidden_size;
        float* W_out = network->weight_gradients + w_size + hidden_size;
        float* b_out = W_out + output_size * hidden_size;
        
        // dL/dh_j = Σ_i W_out[i][j] * error[i]
        memset(output_gradient, 0, hidden_size * sizeof(float));
        for (size_t i = 0; i < output_size; i++) {
            for (size_t j = 0; j < hidden_size; j++) {
                output_gradient[j] += W_out[i * hidden_size + j] * error[i];
            }
        }
        
        // 更新W_out: dL/dW_out[i][j] = error[i] * h[j]
        float* last_hidden = network->layer_outputs + ((num_layers - 1) * max_layer_size);
        for (size_t i = 0; i < output_size; i++) {
            for (size_t j = 0; j < hidden_size; j++) {
                W_out[i * hidden_size + j] -= learning_rate * error[i] * last_hidden[j];
            }
            b_out[i] -= learning_rate * error[i];
        }
    } else {
        memcpy(output_gradient, error, hidden_size * sizeof(float));
    }

    // 步骤2: 逐层反向传播（从最后一层到第一层）
    for (int layer = num_layers - 1; layer >= 0; layer--) {
        float* layer_gradient = network->layer_gradients + (layer * max_layer_size);
        float* prev_gradient = (layer > 0) ? 
            network->layer_gradients + ((layer - 1) * max_layer_size) : gradient;

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
        SELFLNN_CHECK(result == 0, SELFLNN_ERROR_CFC_CELL_CONFIG,
                     "第%d层CfC单元反向传播失败", layer);

        // 应用Dropout掩码到传播的梯度
        if (config->dropout_rate > 0.0f && layer > 0) {
            for (size_t i = 0; i < hidden_size; i++) {
                prev_gradient[i] *= network->dropout_mask[i];
            }
        }
    }

    // 步骤3: 更新各层CfC单元的参数
    for (int layer = 0; layer < num_layers; layer++) {
        CfCCell* cell_layer = network->layers[layer];
        if (!cell_layer) continue;
        
        size_t curr_input_size = (layer == 0) ? input_size : hidden_size;
        size_t weight_count = curr_input_size * hidden_size;
        
        // 更新权重矩阵(W_ax)和偏置(b_a)
        for (size_t k = 0; k < weight_count; k++) {
            cell_layer->weight_matrix[k] -= learning_rate * cell_layer->weight_grad[k];
        }
        for (size_t k = 0; k < hidden_size; k++) {
            cell_layer->bias_vector[k] -= learning_rate * cell_layer->bias_grad[k];
        }
        
        // 更新隐藏到隐藏连接权重
        size_t hh_count = hidden_size * hidden_size;
        for (size_t k = 0; k < hh_count; k++) {
            cell_layer->hidden_to_gate_weights[k] -= 
                learning_rate * cell_layer->hidden_to_gate_weight_grad[k];
            cell_layer->hidden_to_activation_weights[k] -= 
                learning_rate * cell_layer->hidden_to_activation_weight_grad[k];
        }
        
        // 更新门控权重和偏置
        for (size_t k = 0; k < weight_count; k++) {
            cell_layer->input_gate_weights[k] -= 
                learning_rate * cell_layer->input_gate_weight_grad[k];
        }
        for (size_t k = 0; k < hidden_size * 3; k++) {
            cell_layer->gate_biases[k] -= learning_rate * cell_layer->gate_bias_grad[k];
        }
        
        // 更新时间常数
        if (cell_layer->use_adaptive_tau) {
            for (size_t k = 0; k < hidden_size; k++) {
                cell_layer->time_constants[k] -= 
                    cell_layer->tau_learning_rate * cell_layer->time_constant_grad[k];
                if (cell_layer->time_constants[k] < cell_layer->min_time_constant)
                    cell_layer->time_constants[k] = cell_layer->min_time_constant;
                if (cell_layer->time_constants[k] > cell_layer->max_time_constant)
                    cell_layer->time_constants[k] = cell_layer->max_time_constant;
            }
        }
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
    
    // 输出层梯度：误差乘以输出层激活函数导数
    // 在实际实现中，这里应该包括激活函数的导数
    size_t copy_size = (output_size < hidden_size) ? output_size : hidden_size;
    for (size_t i = 0; i < copy_size; i++) {
        output_gradient[i] = error[i];
    }
    
    // 反向传播通过各层
    for (int layer = num_layers - 1; layer >= 0; layer--) {
        float* layer_gradient = network->layer_gradients + (layer * max_layer_size);
        float* layer_output = network->layer_outputs + (layer * max_layer_size);
        (void)layer_output; // 未使用，消除警告
        
        // 获取上一层梯度（对于第一层，使用输入梯度）
        float* prev_gradient = NULL;
        if (layer > 0) {
            prev_gradient = network->layer_gradients + ((layer - 1) * max_layer_size);
        }
        
        // 执行CfC单元反向传播
        int result = cfc_cell_backward(network->layers[layer], 
                                      layer_gradient,
                                      prev_gradient ? prev_gradient : gradient);
        
        SELFLNN_CHECK(result == 0, SELFLNN_ERROR_CFC_CELL_CONFIG,
                     "第%d层CfC单元反向传播失败", layer);
        
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
        // 计算权重梯度
        // 使用保存的原始输入（如果可用，否则使用layer_gradients）
        total_weight_size = hidden_size * config->input_size;
        float* input_signal = saved_input ? saved_input : network->layer_gradients;
        
        // 边界检查：确保索引在范围内
        size_t max_weight_idx = total_weight_size;
        size_t max_bias_idx = hidden_size;
        
        for (size_t i = 0; i < hidden_size; i++) {
            for (size_t j = 0; j < config->input_size; j++) {
                size_t idx = i * config->input_size + j;
                if (idx >= max_weight_idx) {
                    fprintf(stderr, "cfc_accumulate_gradients: 错误：权重索引越界 idx=%zu >= max=%zu\n", idx, max_weight_idx);
                    if (saved_input) {
                        safe_free((void**)&saved_input);
                    }
                    return SELFLNN_ERROR_INVALID_ARGUMENT;
                }
                weight_gradients[idx] = gradient[i] * input_signal[j];
            }
            
            // 偏置梯度
            if (i >= max_bias_idx) {
                fprintf(stderr, "cfc_accumulate_gradients: 错误：偏置索引越界 i=%zu >= max=%zu\n", i, max_bias_idx);
                if (saved_input) {
                    safe_free((void**)&saved_input);
                }
                return SELFLNN_ERROR_INVALID_ARGUMENT;
            }
            bias_gradients[i] = gradient[i];
        }
    } else {
        // 多层网络完整梯度计算
        // 多层CfC使用共享的 hidden_size × hidden_size 权重矩阵
        // 所有层的梯度累积到这个共享矩阵中:
        //   - 第0层(input→hidden): hidden_size × input_size, maps to 前input_size列
        //   - 中间层(hidden→hidden): hidden_size × hidden_size, 直接映射到共享矩阵
        //   - 最后一层(hidden→output): 输出投影矩阵, 存储在W_out区域
        
        total_weight_size = config->hidden_size * config->hidden_size;
        
        // 将共享权重矩阵的梯度清零
        memset(weight_gradients, 0, total_weight_size * sizeof(float));
        // 将共享偏置向量的梯度清零
        memset(bias_gradients, 0, config->hidden_size * sizeof(float));
        
        // 逐层累积梯度到共享权重矩阵中
        for (int layer = 0; layer < num_layers; layer++) {
            // 获取当前层CfC单元中已计算的权重梯度和偏置梯度
            CfCCell* cell = network->layers[layer];
            if (!cell) continue;
            
            size_t curr_input_size = (layer == 0) ? config->input_size : config->hidden_size;
            size_t cell_weight_size = curr_input_size * config->hidden_size;
            
            if (layer == 0) {
                // 第0层: hidden_size × input_size 映射到共享矩阵的前input_size列
                // cell->weight_grad[i * input_size + j] → weight_gradients[i * hidden_size + j]
                for (size_t i = 0; i < config->hidden_size; i++) {
                    for (size_t j = 0; j < config->input_size; j++) {
                        size_t cell_idx = i * config->input_size + j;
                        size_t shared_idx = i * config->hidden_size + j;
                        if (cell_idx < cell_weight_size && shared_idx < total_weight_size) {
                            weight_gradients[shared_idx] += cell->weight_grad[cell_idx];
                        }
                    }
                }
            } else if (layer < num_layers - 1) {
                // 中间层: hidden_size × hidden_size 直接累积到共享矩阵
                for (size_t i = 0; i < config->hidden_size; i++) {
                    for (size_t j = 0; j < config->hidden_size; j++) {
                        size_t idx = i * config->hidden_size + j;
                        if (idx < cell_weight_size && idx < total_weight_size) {
                            weight_gradients[idx] += cell->weight_grad[idx];
                        }
                    }
                }
            } else {
                // 最后一层: hidden_size × output_size 是输出投影
                // 存储在 weight_gradients + hidden_size*hidden_size + hidden_size 之后
                // (W_out区域), 此处无需处理到共享权重矩阵
            }
            
            // 累积偏置梯度（所有层的偏置累积到共享的hidden_size偏置向量）
            for (size_t i = 0; i < config->hidden_size; i++) {
                if (i < config->hidden_size) {
                    bias_gradients[i] += cell->bias_grad[i];
                }
            }
        }
        
        // 将计算出的梯度存储到网络内部缓冲区
        size_t actual_weight_size = config->hidden_size * config->hidden_size;
        size_t copy_weight_size = (total_weight_size < actual_weight_size) ?
            total_weight_size : actual_weight_size;
        if (network->weight_gradients && weight_gradients) {
            memcpy(network->weight_gradients, weight_gradients, 
                   copy_weight_size * sizeof(float));
        }
        
        size_t actual_bias_size = config->hidden_size;
        size_t copy_bias_size = actual_bias_size;
        if (network->bias_gradients && bias_gradients) {
            memcpy(network->bias_gradients, bias_gradients,
                   copy_bias_size * sizeof(float));
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
    size_t weight_size = network->config.hidden_size * network->config.hidden_size;
    if (network->config.num_layers == 1) {
        weight_size = network->config.input_size * network->config.hidden_size;
    }
    
    SELFLNN_CHECK(fwrite(network->weight_matrix, sizeof(float), weight_size, file) == weight_size,
                 SELFLNN_ERROR_IO_ERROR,
                 "保存权重矩阵失败（大小：%zu）", weight_size);
    
    SELFLNN_CHECK(fwrite(network->bias_vector, sizeof(float), network->config.hidden_size, file) 
                 == network->config.hidden_size,
                 SELFLNN_ERROR_IO_ERROR,
                 "保存偏置向量失败（大小：%zu）", network->config.hidden_size);
    
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
    size_t weight_size = config.hidden_size * config.hidden_size;
    if (config.num_layers == 1) {
        weight_size = config.input_size * config.hidden_size;
    }
    
    SELFLNN_CHECK(fread(network->weight_matrix, sizeof(float), weight_size, file) == weight_size,
                 SELFLNN_ERROR_IO_ERROR,
                 "读取权重矩阵失败（大小：%zu）", weight_size);
    
    SELFLNN_CHECK(fread(network->bias_vector, sizeof(float), config.hidden_size, file) 
                 == config.hidden_size,
                 SELFLNN_ERROR_IO_ERROR,
                 "读取偏置向量失败（大小：%zu）", config.hidden_size);
    
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
    
    // 返回权重矩阵信息
    // 与cfc_create中的实际存储布局一致：
    // 单层: weight_size = input_size * hidden_size
    // 多层: weight_size = hidden_size * hidden_size
    *weight_matrix = network->weight_matrix;
    if (network->config.num_layers <= 1) {
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
    
    // 返回偏置向量信息
    *bias_vector = network->bias_vector;
    *bias_count = network->config.hidden_size;

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
    /* 保存当前隐藏状态 */
    float* saved_hs = rhsc->temp_buffer1;
    float* saved_cs = rhsc->temp_buffer2;
    memcpy(saved_hs, y, n * sizeof(float));
    /* 调用cfc_forward计算 dy = f(input, y) */
    float* out_buffer = (float*)safe_malloc(net->config.output_size * sizeof(float));
    if (!out_buffer) return -1;
    int ret = cfc_forward(net, rhsc->input, dydt, dydt, out_buffer);
    /* dydt = new_state - old_state ≈ f(y) （当dt=1时的离散近似）
     * 真正的连续RHS: dy/dt = -y/τ + σ(Wx + Uy + b) ⊙ f(Wx + Vy + a)
     * 近似: dy/dt ≈ cfc_cell_forward(cell, input, y) - y
     * 此处 dydt 已被 cfc_forward 写入为新状态值 */
    for (size_t i = 0; i < n; i++) {
        dydt[i] = dydt[i] - saved_hs[i];
    }
    safe_free((void**)&out_buffer);
    /* 恢复隐藏状态到 cfc_forward 修改前的值 */
    /* 注意：cfc_forward 修改了 cell 的内部状态，但 RHS 求值应该是无副作用的。
     * 然而 CfC 单元的 cfc_cell_forward 会将激活值写入 cell->state->state。
     * 为此，我们需要在 RHS 求值完成后恢复状态。 */
    /* 通过重新设置 cell 状态来恢复 */
    for (int l = 0; l < net->config.num_layers; l++) {
        if (net->layers[l] && net->layers[l]->state) {
            memcpy(net->layers[l]->state->state, saved_hs, n * sizeof(float));
        }
    }
    return ret;
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
    /* 最小特征值近似：对对角线元素取最小值（用于刚度比估算） */
    float diag_min = 1e20f, diag_avg = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(J[i * n + i]);
        diag_avg += d;
        if (d < diag_min && d > 1e-10f) diag_min = d;
    }
    diag_avg /= (float)n;
    if (diag_min > 1e10f) diag_min = diag_avg * 0.01f;
    if (diag_min < 1e-10f) diag_min = 1e-10f;
    *stiffness_ratio = lambda_max / diag_min;
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

    /* 刚度检测 */
    if (config->enable_stiffness_detect) {
        float sr = 1.0f;
        if (cfc_detect_stiffness(network, input, hidden_state, &sr) == 0) {
            stats->stiffness_ratio = sr;
            if (sr > 100.0f) {
                cfc_set_solver_type(network, 5); /* ROSENBROCK */
                using_stiff_solver = 1;
                stats->solver_switches++;
            } else if (sr > 10.0f) {
                cfc_set_solver_type(network, 4); /* DP54自适应 */
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
    }

    stats->final_t = t;
    stats->avg_dt = fabsf(t - config->t_start) / fmaxf((float)stats->accepted_steps, 1.0f);
    if (stats->min_dt_used < 1e-12f) stats->min_dt_used = h;
    stats->used_stiff_solver = using_stiff_solver;

    /* I-012修复：步数达到上限未完成时标记（外部可通过rejected_steps推断） */
    if (stats->total_steps >= config->max_steps)
        stats->rejected_steps = -(stats->rejected_steps + 1);

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

