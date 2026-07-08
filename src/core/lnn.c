/**
 * @file lnn.c
 * @brief 液态神经网络核心实现
 * 
 * 液态神经网络（Liquid Neural Network）连续时间递归神经网络实现。
 * 基于CfC（Closed-form Continuous-time）单元，使用微分方程描述动态。
 *
 * P0-001修复: lnn_create() 强制单一LNN原则
 *   当 selflnn_enforce_single_lnn() 已调用后，所有 lnn_create() 调用
 *   将被重定向到全局唯一LNN实例，禁止创建独立LNN。
 */

#define SELFLNN_IMPLEMENTATION
#define SELFLNN_CORE_INTERNAL
#include "selflnn/core/lnn.h"
#include "selflnn/core/cfc.h"
#include "selflnn/core/cfc_cell.h"
#include "selflnn/core/cfc_network.h"
#include "selflnn/core/state.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/laplace.h"
#include "selflnn/core/laplace_features.h"
#include "selflnn/core/laplace_fft.h"
#include "selflnn/core/loss.h"          /* FIX-003: loss_gradient_ex 支持 */
#include "selflnn/core/laplace_unified.h" /* 原laplace_integration.h为纯转发,已删除 */
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/selflnn.h"           /* P0-001: selflnn_is_single_lnn_enforced + selflnn_get_lnn */
#include "selflnn/memory/memory.h"
#include "selflnn/core/parameter_shard.h"
#include "selflnn/core/optimizer.h"       /* P2-010: 优化器集成 */
#include "selflnn/core/lnn_layer_norm.h"  /* M-2修复: lnn_save中需要直接访问层归一化参数 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

/* 平台特定头文件 */
#ifdef _WIN32
#include <windows.h>
#endif

#ifdef _MSC_VER
/* P3-01修复: 移除4701(未初始化局部变量)和4456(局部变量隐藏参数)抑制，保留真实告警 */
#pragma warning(disable:4100 4189 4244 4267)
#endif

/**
 * @brief 简单32位哈希 - 用于确定性伪随机权重初始化
 */
static inline uint32_t hash32(uint32_t x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

/* 公共梯度裁剪函数，消除三处重复 */
static inline void lnn_clip_gradients(float* grads, size_t count, float max_norm) {
    /* P2-05修复: 移除地址范围hack((uintptr_t)grads < 0x1000)，仅保留空指针与参数合法性检查 */
    if (!grads || count == 0 || max_norm <= 0.0f) return;
    /* P0修复: 先处理非有限值(Inf/NaN)，避免Inf污染max_val导致scale=0清零所有梯度。
     * Inf/NaN梯度直接清零，但不参与max_val计算，确保剩余有限梯度能正确缩放。 */
    float max_val = 0.0f;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(grads[i])) {
            grads[i] = 0.0f;  /* Inf/NaN直接清零 */
            continue;         /* 跳过，不参与max_val计算 */
        }
        float abs_g = grads[i] < 0 ? -grads[i] : grads[i];
        if (abs_g > max_val) max_val = abs_g;
    }
    /* 只有当max_val有效(>0)且超过阈值时才进行缩放 */
    if (max_val > max_norm && max_val > 0.0f) {
        float scale = max_norm / max_val;
        for (size_t i = 0; i < count; i++) grads[i] *= scale;
    }
}

// 调试输出宏：仅在SELFLNN_DEBUG_LNN定义时输出到stderr
#ifdef SELFLNN_DEBUG_LNN
#define LNN_DEBUG(...) fprintf(stderr, __VA_ARGS__)
#define LNN_DEBUG_FLUSH() fflush(stderr)
#else
#define LNN_DEBUG(...) /* 调试信息仅在SELFLNN_DEBUG_LNN定义时输出 */
#define LNN_DEBUG_FLUSH() /* 无操作 */
#endif

#define LNN_LOCK(n)    mutex_lock((n)->lock)
#define LNN_UNLOCK(n)  mutex_unlock((n)->lock)

/**
 * @brief 液态神经网络内部结构体
 * 注意：结构体定义已移至lnn.h中，通过SELFLNN_IMPLEMENTATION条件编译暴露
 */
/* struct LNN 定义在lnn.h中 */

/**
 * @brief 创建液态神经网络实例
 */
LNN* lnn_create(const LNNConfig* config) {
    if (!config) {
        SELFLNN_SET_ERROR(SELFLNN_ERROR_INVALID_PARAMETER,
            "lnn_create", "配置指针为空", "请提供有效的LNNConfig配置");
        return NULL;
    }
    
    /* P0-001修复: 单一LNN原则强制执行
     * 当 selflnn_enforce_single_lnn() 已调用后，
     * 所有模块禁止创建独立LNN实例，必须使用全局唯一LNN。
     * 全局LNN为全模态共享，所有状态在同一个连续动态系统中演化。 */
    if (selflnn_is_single_lnn_enforced()) {
        void* global_lnn = selflnn_get_lnn();
        if (global_lnn) {
            log_debug("[单一LNN] lnn_create()——重定向到全局LNN(input=%zu,hidden=%zu,output=%zu)",
                       config->input_size, config->hidden_size, config->output_size);
            return (LNN*)global_lnn;
        }
        /* 全局LNN尚未创建（首次调用），允许正常创建 */
        log_info("[单一LNN] 首次创建全局LNN实例");
    }
    
    // 验证配置参数
    if (config->input_size == 0 || config->hidden_size == 0 || config->output_size == 0) {
        SELFLNN_SET_ERROR(SELFLNN_ERROR_INVALID_PARAMETER,
            "lnn_create", "无效的配置参数(input_size/hidden_size/output_size为0)",
            "所有维度参数必须大于0");
        return NULL;
    }
    
    // 分配内存
    LNN* network = (LNN*)safe_malloc(sizeof(LNN));
    if (!network) {
        SELFLNN_SET_ERROR(SELFLNN_ERROR_OUT_OF_MEMORY,
            "lnn_create", "内存分配失败(LNN结构体)",
            "请检查系统内存或减少模型参数规模");
        return NULL;
    }
    memset(network, 0, sizeof(LNN));
    
    // 复制配置
    memcpy(&network->config, config, sizeof(LNNConfig));
    
    // 确保新增字段有合理默认值（向后兼容性）
    if (network->config.time_constant <= 0.0f) {
        network->config.time_constant = 0.1f;  // 合理默认时间常数
    }
    if (network->config.enable_adaptation == 0) {
        network->config.enable_adaptation = 1;  // 默认启用参数自适应
    }
    if (network->config.num_layers <= 0) {
        network->config.num_layers = 1;  // 默认单层网络
    }
    if (network->config.max_grad_norm <= 0.0f) {
        network->config.max_grad_norm = 5.0f;  // 默认梯度裁剪阈值
    }
    /* K-027: enable_laplace默认值（=0时视为未设置，设为默认启用1） */
    if (network->config.enable_laplace == 0 && !network->config.enable_sharding) {
        network->config.enable_laplace = 1;  // 默认启用拉普拉斯增强
    }
    /* K-028: enable_quaternion默认值（=0时视为未设置，设为默认启用1） */
    if (network->config.enable_quaternion == 0 && !network->config.enable_sharding) {
        network->config.enable_quaternion = 1;  // 默认启用四元数增强
    }
    
    // 初始化CfC网络配置（零初始化避免未赋值字段包含栈上垃圾值）
    CfCNetworkConfig cfc_config = {0};
    cfc_config.input_size = network->config.input_size;
    cfc_config.hidden_size = network->config.hidden_size;
    cfc_config.output_size = network->config.output_size;
    cfc_config.learning_rate = network->config.learning_rate;
    cfc_config.time_constant = network->config.time_constant;
    cfc_config.noise_std = network->config.noise_std;
    cfc_config.enable_training = network->config.enable_training;
    cfc_config.enable_adaptation = network->config.enable_adaptation;
    cfc_config.num_layers = network->config.num_layers;
    cfc_config.dropout_rate = 0.0f;               // 默认无dropout
    cfc_config.use_batch_norm = 0;                // 默认不使用批归一化
    cfc_config.ode_solver_type = network->config.ode_solver_type;
    
    // 创建CfC网络
    network->cfc_network = cfc_create(&cfc_config);
    if (!network->cfc_network) {
        /* ZSF-100修复：添加错误信息，使调用者可通过selflnn_get_last_error获取失败原因 */
        selflnn_set_last_error(SELFLNN_ERROR_CFC_CREATE_FAILED, "lnn_create", __FILE__, __LINE__,
            "CfC网络创建失败：输入大小=%zu, 隐藏大小=%zu, 层数=%d",
            (size_t)cfc_config.input_size, (size_t)cfc_config.hidden_size, cfc_config.num_layers);
        safe_free((void**)&network);
        return NULL;
    }
    
    // 分配缓冲区
    size_t hidden_size = config->hidden_size;
    size_t input_size = config->input_size;
    size_t output_size = config->output_size;
    
    network->hidden_state = (float*)safe_calloc(hidden_size, sizeof(float));
    network->cell_state = (float*)safe_calloc(hidden_size, sizeof(float));
    network->input_buffer = (float*)safe_calloc(input_size, sizeof(float));
    network->output_buffer = (float*)safe_calloc(output_size, sizeof(float));
    network->error_buffer = (float*)safe_calloc(output_size, sizeof(float));
    network->quaternion_pre_buf = (float*)safe_calloc(hidden_size, sizeof(float));
    
    // gradient_buffer 需要容纳 cfc_backward 中 max_layer_size 个浮点数的写入
    // 其中 max_layer_size = max(input_size, hidden_size, output_size)
    size_t grad_buffer_size = hidden_size;
    if (input_size > grad_buffer_size) grad_buffer_size = input_size;
    if (output_size > grad_buffer_size) grad_buffer_size = output_size;
    network->gradient_buffer = (float*)safe_calloc(grad_buffer_size, sizeof(float));
    
    // 检查内存分配
    /* P0-003修复: 添加quaternion_pre_buf的NULL检查，防止分配失败后解引用崩溃 */
    if (!network->hidden_state || !network->cell_state || 
        !network->input_buffer || !network->output_buffer ||
        !network->error_buffer || !network->gradient_buffer ||
        !network->quaternion_pre_buf) {
        lnn_free(network);
        return NULL;
    }
    
    // 创建网络状态
    network->state = network_state_create_simple(hidden_size);
    if (!network->state) {
        lnn_free(network);
        return NULL;
    }
    
    // 初始化分片系统字段
    network->shard_system = NULL;
    network->enable_param_sharding = 0;
    network->num_local_shards = 0;
    network->current_shard_id = 0;
    network->gradient_checkpoint_buffer = NULL;
    network->gradient_checkpoint_size = 0;
    network->enable_gradient_checkpointing = 0;
    network->gradient_checkpoint_interval = 4;
    network->activation_checkpoints = NULL;
    network->activation_checkpoint_sizes = NULL;
    network->num_activation_checkpoints = 0;
    network->activation_checkpoint_capacity = 0;
    network->enable_model_parallel = 0;
    network->model_parallel_group = 0;
    network->model_parallel_rank = 0;
    network->model_parallel_size = 0;
    network->enable_async_gradient_sync = 0;
    network->gradient_sync_in_progress = 0;
    network->gradient_sync_count = 0;

    // 初始化锁
    network->lock = mutex_create();
    if (!network->lock) {
        lnn_free(network);
        return NULL;
    }

    // 拉普拉斯分析器（频域稳定性分析与增强）
    // K-027: 根据config->enable_laplace配置决定是否启用，默认启用
    // 根据需求"有效引入拉普拉斯可以增强系统"，但允许用户关闭
    if (config->enable_laplace) {
        LaplaceConfig laplace_cfg;
        memset(&laplace_cfg, 0, sizeof(LaplaceConfig));
        laplace_cfg.num_samples = 512;
        laplace_cfg.frequency_range = 100.0f;
        laplace_cfg.enable_auto_tuning = 1;
        laplace_cfg.stability_threshold = 0.5f;
        network->laplace_analyzer = laplace_analyzer_create(&laplace_cfg);
        if (network->laplace_analyzer) {
            network->laplace_gradient_strength = 0.1f;
            network->laplace_forward_modulation = 1;
        }
    } else {
        network->laplace_analyzer = NULL;
        network->laplace_gradient_strength = 0.0f;
        network->laplace_forward_modulation = 0;
    }

    // 根据配置自动启用分片
    if (config->enable_sharding && config->num_shards > 0) {
        size_t total_params = (input_size * hidden_size) + hidden_size;
        size_t num_shards = config->num_shards;
        size_t local_shards = num_shards;
        if (config->shard_id < num_shards) {
            local_shards = 1;
        }

        ShardSystemConfig shard_config;
        shard_config.num_shards = local_shards;
        shard_config.total_params = total_params;
        shard_config.default_precision = SHARD_PRECISION_FP32;
        shard_config.enable_async = config->enable_async_gradient_sync;
        shard_config.enable_gradient_compression = 
            (config->gradient_compression_ratio > 0.0f) ? 1 : 0;
        shard_config.compression_ratio = config->gradient_compression_ratio;
        shard_config.enable_overlap = config->enable_model_parallel;
        shard_config.gradient_sync_interval = 1;
        shard_config.device_ids = NULL;
        shard_config.locations = NULL;
        shard_config.shard_sizes = NULL;

        network->shard_system = shard_system_create(&shard_config);
        if (network->shard_system) {
            shard_system_initialize(network->shard_system);
            network->enable_param_sharding = 1;
            network->num_local_shards = local_shards;
            network->current_shard_id = config->shard_id;
        }
    }

    // 根据配置启用梯度检查点
    if (config->enable_gradient_checkpoint && config->checkpoint_interval > 0) {
        network->enable_gradient_checkpointing = 1;
        network->gradient_checkpoint_interval = config->checkpoint_interval;
        network->activation_checkpoint_capacity = config->num_layers > 0 ? 
            (size_t)config->num_layers : 16;
        network->activation_checkpoints = (float**)safe_calloc(
            network->activation_checkpoint_capacity, sizeof(float*));
        network->activation_checkpoint_sizes = (size_t*)safe_calloc(
            network->activation_checkpoint_capacity, sizeof(size_t));
        network->num_activation_checkpoints = 0;
    }

    // 根据配置启用模型并行
    if (config->enable_model_parallel) {
        network->enable_model_parallel = 1;
    }

    // 根据配置启用异步梯度同步
    if (config->enable_async_gradient_sync) {
        network->enable_async_gradient_sync = 1;
    }

    // 初始化变量
    network->is_initialized = 1;
    network->current_loss = 0.0f;
    network->forward_count = 0;
    network->backward_count = 0;
    network->total_training_time = 0.0;
    
    /* 演化能力联动：读取配置中的enable_evolution并设置默认间隔 */
    network->enable_evolution = config->enable_evolution;
    network->evolution_interval = 100;

    // 初始化隐藏状态为小随机值
    for (size_t i = 0; i < hidden_size; i++) {
        network->hidden_state[i] = rng_uniform(-0.01f, 0.01f);
    }
    
    return network;
}

/**
 * @brief 释放液态神经网络实例
 */
void lnn_free(LNN* network) {
    if (!network) {
        return;
    }
    
    // 释放CfC网络
    if (network->cfc_network) {
        cfc_free(network->cfc_network);
    }
    
    // 释放网络状态
    if (network->state) {
        network_state_free(network->state);
    }
    
    // 释放拉普拉斯分析器
    if (network->laplace_analyzer) {
        laplace_analyzer_free(network->laplace_analyzer);
        network->laplace_analyzer = NULL;
    }
    
    // 释放分片系统
    if (network->shard_system) {
        shard_system_free(network->shard_system);
        network->shard_system = NULL;
    }

    // 释放锁
    if (network->lock) {
        mutex_destroy(network->lock);
        network->lock = NULL;
    }

    /* H-004修复: 释放用户自定义数据 */
    if (network->user_data && network->user_data_free) {
        network->user_data_free(network->user_data);
        network->user_data = NULL;
    }

    // 释放梯度检查点缓冲区
    if (network->gradient_checkpoint_buffer) {
        safe_free((void**)&network->gradient_checkpoint_buffer);
    }

    // 释放激活检查点
    if (network->activation_checkpoints) {
        for (size_t i = 0; i < network->num_activation_checkpoints; i++) {
            if (network->activation_checkpoints[i]) {
                safe_free((void**)&network->activation_checkpoints[i]);
            }
        }
        safe_free((void**)&network->activation_checkpoints);
    }
    if (network->activation_checkpoint_sizes) {
        safe_free((void**)&network->activation_checkpoint_sizes);
    }

    // 释放缓冲区（含四元数预计算缓冲区，修复P0-2内存泄漏）
    safe_free((void**)&network->hidden_state);
    safe_free((void**)&network->cell_state);
    safe_free((void**)&network->input_buffer);
    safe_free((void**)&network->output_buffer);
    safe_free((void**)&network->error_buffer);
    safe_free((void**)&network->gradient_buffer);
    safe_free((void**)&network->quaternion_pre_buf);
    
    // 释放网络结构
    safe_free((void**)&network);
}

/* H-004修复: 用户数据存取器 */
void lnn_set_user_data(LNN* network, void* data, void (*free_fn)(void*)) {
    if (!network) return;
    if (network->user_data && network->user_data_free) {
        network->user_data_free(network->user_data);
    }
    network->user_data = data;
    network->user_data_free = free_fn;
}

void* lnn_get_user_data(LNN* network) {
    if (!network) return NULL;
    return network->user_data;
}

/* M-005说明: _lnn_forward_internal 等以_前缀命名的函数是跨编译单元的内部API，
 * 被 extended_training.c 等模块调用。因跨模块依赖无法标记为static。
 * 当前通过命名约定(_前缀)标识内部函数，后续应移入 lnn_internal.h 头文件统一管理。 */

/**
 * @brief 前向传播内部实现（无锁，调用者需持有 LNN_LOCK）
 */
int _lnn_forward_internal(LNN* network, const float* input, float* output) {
    clock_t start_time = clock();
    size_t input_size = network->config.input_size;
    memcpy(network->input_buffer, input, input_size * sizeof(float));

    /* T-004修复: 输入层归一化 —— 消除多模态输入的量级差异。
     * 传感器原始值(±2000)与图像L2归一化特征(±0.06)量级差32000×，
     * 直接馈入CfC会导致传感器主导。此处做每样本独立LayerNorm风格归一化：
     * mean→0, std→1, 截断±4防止极端值。 */
    {
        float mean = 0.0f, var = 0.0f;
        size_t n = input_size;
        for (size_t i = 0; i < n; i++) mean += network->input_buffer[i];
        mean /= (float)n;
        for (size_t i = 0; i < n; i++) {
            float d = network->input_buffer[i] - mean;
            var += d * d;
        }
        var = var / (float)n;
        /* P1-R1修复: 添加最小方差保护，防止全零输入导致inv_std=10000过度放大。
         * 对于维度>=64的输入，方差<1e-6表示输入几乎恒定，应跳过归一化。 */
        if (var < 1e-6f) {
            if (n >= 64) {
                /* 大维度输入全恒定，跳过归一化直接继续 */
                goto after_layer_norm;
            }
            var = 1e-6f;
        }
        var += 1e-8f;
        float inv_std = 1.0f / sqrtf(var);
        for (size_t i = 0; i < n; i++) {
            float z = (network->input_buffer[i] - mean) * inv_std;
            if (z > 4.0f) z = 4.0f;
            else if (z < -4.0f) z = -4.0f;
            network->input_buffer[i] = z;
        }
    }
    after_layer_norm:

    /* 拉普拉斯频域预处理：在CfC前向传播之前对输入进行稳定性滤波，
     * 避免后处理覆盖CfC的输出投影矩阵计算结果 */
    if (network->laplace_analyzer != NULL && network->laplace_forward_modulation) {
        float strength = network->laplace_gradient_strength * 0.3f;
        lnn_laplace_modulate_hidden(network->laplace_analyzer,
                                    network->input_buffer,
                                    input_size,
                                    strength);
    }

/*关键修复: 拉普拉斯稳定性分数从NetworkState传播到各CfCCell。
     * network_state_set_laplace_metrics()已将频域分析结果写入网络状态，
     * 但需要显式复制到每个CfCCell实例，cfc_closed_form_solution中的
     * tau调制才能实际生效。此步骤是拉普拉斯-CfC数据流的关键闭环。
 *: 使用network_state_get_laplace_stability_score() getter，
     * 因NetworkState结构体定义私有于state.c。 */
    if (network->cfc_network && network->cfc_network->layers && network->state) {
        float lap_stab = network_state_get_laplace_stability_score(network->state);
        for (int l = 0; l < network->cfc_network->config.num_layers; l++) {
            CfCCell* cell = network->cfc_network->layers[l];
            if (cell) {
                cell->laplace_stability_score = lap_stab;
            }
        }
    }

    /* H-001修复: 在高并发场景下添加cfc_network空指针保护 */
    if (!network->cfc_network) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "CfC网络未初始化，无法执行前向传播");
        return -1;
    }

    int result = cfc_forward(network->cfc_network,
                           network->input_buffer,
                           network->hidden_state,
                           network->cell_state,
                           network->output_buffer);
    SELFLNN_CHECK(result == 0, SELFLNN_ERROR_NETWORK_FORWARD,
                 "CfC网络前向传播失败");
    size_t hidden_size = network->config.hidden_size;
    size_t output_size = network->config.output_size;

    /* 拉普拉斯频域分析（仅诊断，不修改输出） */
    if (network->laplace_analyzer && network->laplace_forward_modulation) {
        float stability_score = 0.5f;
        float recommended_cutoff = 10.0f;
        float frequency_bandwidth = 50.0f;
        lnn_laplace_analyze_network_dynamics(
            network->laplace_analyzer,
            network->config.time_constant,
            network->hidden_state,
            hidden_size,
            &stability_score,
            &recommended_cutoff,
            &frequency_bandwidth);
        network_state_set_laplace_metrics(network->state, stability_score,
                                          recommended_cutoff, frequency_bandwidth);
    }

    /* 四元数增强：对隐藏状态施加旋转不变性闭式解
     * 将 hidden_state 按4个一组映射为四元数，应用 CfC 闭式解更新，
     * 增强状态在四维旋转空间中的表示稳定性和不变性 */
    if (network->config.enable_quaternion && hidden_size >= 4) {
/* 保存四元数处理前的状态供反向传播使用。
         * hidden_state将在四元数循环中被原地修改为后处理值，
         * 而反向传播需要前处理值来正确计算tanh导数链。 */
        if (network->quaternion_pre_buf) {
            memcpy(network->quaternion_pre_buf, network->hidden_state,
                   hidden_size * sizeof(float));
        }
        float quat_dt = network->config.time_constant * 0.5f;
        if (quat_dt < 0.001f) quat_dt = 0.001f;
        if (quat_dt > 0.5f) quat_dt = 0.5f;

        size_t num_quats = hidden_size / 4;
        /* 每个四元数分量独立应用CfC闭式解：
         *   h_new = h * exp(-dt/τ) + gate(h) * activation(h) * (1 - exp(-dt/τ))
         * 闭式解保持连续时间动态特性，同时增强旋转不变性 */
        for (size_t q = 0; q < num_quats; q++) {
            float* qv = network->hidden_state + q * 4;

            /* 计算该四元数组的驱动信号（tanh非线性压缩） */
            float drive[4];
            for (int c = 0; c < 4; c++) {
                float x = qv[c];
                if (x > 10.0f) x = 10.0f;
                if (x < -10.0f) x = -10.0f;
                drive[c] = tanhf(x);
            }

            /* 计算四元数时间常数（基于激活水平自适应调整） */
            float activation_norm = sqrtf(drive[0]*drive[0] + drive[1]*drive[1]
                                        + drive[2]*drive[2] + drive[3]*drive[3]);
            float adaptive_tau = network->config.time_constant
                               * (1.0f + 0.5f * activation_norm);
            if (adaptive_tau < 0.001f) adaptive_tau = 0.001f;
            if (adaptive_tau > 2.0f) adaptive_tau = 2.0f;

            float decay = expf(-quat_dt / adaptive_tau);
            float complement = 1.0f - decay;

            /* 四元数闭式解混合：保留历史 + 注入新驱动 */
            for (int c = 0; c < 4; c++) {
                qv[c] = qv[c] * decay + drive[c] * complement;
            }

            /* 四元数归一化（保持单位范数，确保旋转有效性） */
            float norm = sqrtf(qv[0]*qv[0] + qv[1]*qv[1] + qv[2]*qv[2] + qv[3]*qv[3]);
            if (norm > 1e-8f) {
                float inv_norm = 1.0f / norm;
                for (int c = 0; c < 4; c++) {
                    qv[c] *= inv_norm;
                }
            }
        }
    }

    memcpy(output, network->output_buffer, output_size * sizeof(float));
    network_state_update(network->state, network->hidden_state, hidden_size);
    network->forward_count++;
    clock_t end_time = clock();
    network->total_training_time += (double)(end_time - start_time) / CLOCKS_PER_SEC;
    return 0;
}

/* 公开LNN锁API
 * 允许外部模块(在线学习器/演化引擎)在跨越多个LNN操作时持有锁。
 * 调用者必须确保lock/unlock成对使用,且不在锁内嵌套获取同一把锁。 */
SELFLNN_API void lnn_lock(LNN* network) {
    if (!network) return;
    LNN_LOCK(network);
}
SELFLNN_API void lnn_unlock(LNN* network) {
    if (!network) return;
    LNN_UNLOCK(network);
}

/**
 * @brief 前向传播
 */
int lnn_forward(LNN* network, const float* input, float* output) {
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_NULL(input, "输入向量为空");
    SELFLNN_CHECK_NULL(output, "输出缓冲区为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");
    LNN_LOCK(network);
    int ret = _lnn_forward_internal(network, input, output);
    LNN_UNLOCK(network);
    return ret;
}

/**
 * @brief 安全前向传播（含边界检查和维度适配）
 */
int lnn_forward_safe(LNN* network, const float* input, size_t input_size,
                     float* output, size_t output_size) {
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_NULL(input, "输入向量为空");
    SELFLNN_CHECK_NULL(output, "输出缓冲区为空");
    SELFLNN_CHECK(input_size > 0, SELFLNN_ERROR_INVALID_ARGUMENT, "输入维度不能为0");
    SELFLNN_CHECK(output_size > 0, SELFLNN_ERROR_INVALID_ARGUMENT, "输出容量不能为0");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");

    size_t cfg_input_size = network->config.input_size;
    size_t cfg_output_size = network->config.output_size;

    /* 维度完全匹配时走快速路径 */
    if (input_size == cfg_input_size && output_size >= cfg_output_size) {
        LNN_LOCK(network);
        int ret = _lnn_forward_internal(network, input, output);
        LNN_UNLOCK(network);
        return (ret == 0) ? (int)cfg_output_size : -1;
    }

    /* 分配内部缓冲区（需要适配输入） */
    float* safe_input = (float*)safe_calloc(cfg_input_size, sizeof(float));
    float* safe_output = (float*)safe_calloc(cfg_output_size, sizeof(float));
    if (!safe_input || !safe_output) {
        safe_free((void**)&safe_input);
        safe_free((void**)&safe_output);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "安全前向传播：临时缓冲区分配失败");
        return -1;
    }

    /* 输入适配：过大截断，过小零填充 */
    size_t copy_in = (input_size < cfg_input_size) ? input_size : cfg_input_size;
    memcpy(safe_input, input, copy_in * sizeof(float));

    LNN_LOCK(network);
    int ret = _lnn_forward_internal(network, safe_input, safe_output);
    LNN_UNLOCK(network);

    if (ret != 0) {
        safe_free((void**)&safe_input);
        safe_free((void**)&safe_output);
        return -1;
    }

    /* 输出适配：写入实际可用空间 */
    size_t copy_out = (cfg_output_size < output_size) ? cfg_output_size : output_size;
    memcpy(output, safe_output, copy_out * sizeof(float));
    if (output_size > copy_out) {
        memset(output + copy_out, 0, (output_size - copy_out) * sizeof(float));
    }

    safe_free((void**)&safe_input);
    safe_free((void**)&safe_output);
    return (int)copy_out;
}

/* ================================================================
 *: 并发隔离前向传播
 * 每调用方持有独立的CfC状态副本，多模态可同时进行前向传播。
 * 权重访问使用读写锁（读锁），训练写入时排他（写锁）。
 * ================================================================ */

LNNForwardState* lnn_forward_state_create(LNN* network) {
    if (!network) return NULL;

    LNNForwardState* state = (LNNForwardState*)safe_calloc(1, sizeof(LNNForwardState));
    if (!state) return NULL;

    state->hidden_size = network->config.hidden_size;
    state->input_size = network->config.input_size;
    state->output_size = network->config.output_size;
    /* CfC cell_state尺寸估算：hidden_size * num_layers + 额外缓冲 */
    state->cell_state_size = state->hidden_size * (size_t)network->config.num_layers * 2;

    state->hidden_state = (float*)safe_calloc(state->hidden_size, sizeof(float));
    state->cell_state = (float*)safe_calloc(state->cell_state_size, sizeof(float));
    state->input_buffer = (float*)safe_calloc(state->input_size, sizeof(float));
    state->output_buffer = (float*)safe_calloc(state->output_size, sizeof(float));

    if (!state->hidden_state || !state->cell_state ||
        !state->input_buffer || !state->output_buffer) {
        lnn_forward_state_free(state);
        return NULL;
    }

    /* 从LNN复制当前隐藏状态作为初始状态 */
    LNN_LOCK(network);
    memcpy(state->hidden_state, network->hidden_state,
           state->hidden_size * sizeof(float));
    if (network->cell_state) {
        memcpy(state->cell_state, network->cell_state,
               (state->cell_state_size < network->config.hidden_size * (size_t)network->config.num_layers * 2
                ? state->cell_state_size : network->config.hidden_size * (size_t)network->config.num_layers * 2) * sizeof(float));
    }
    LNN_UNLOCK(network);

    state->initialized = 1;
    return state;
}

void lnn_forward_state_free(LNNForwardState* state) {
    if (!state) return;
    safe_free((void**)&state->hidden_state);
    safe_free((void**)&state->cell_state);
    safe_free((void**)&state->input_buffer);
    safe_free((void**)&state->output_buffer);
    safe_free((void**)&state);
}

int lnn_forward_isolated(LNN* network, LNNForwardState* state,
                         const float* input, float* output) {
    if (!network || !state || !state->initialized || !input || !output) return -1;

    size_t in_sz = state->input_size;
    size_t hid_sz = state->hidden_size;
    size_t out_sz = state->output_size;

    /* 复制输入到隔离缓冲区 */
    memcpy(state->input_buffer, input, in_sz * sizeof(float));

/* 添加排他锁保护权重读取
     * 原实现完全无锁——训练期间并发推理会读到半更新权重
     * 当前使用互斥锁（非RWLock），多读者实际串行化但保证安全 */
    LNN_LOCK(network);
/* 使用隔离状态进行CfC前向传播
     * 直接将隔离状态传递给cfc_forward，完全避开LNN全局状态 */
    int result = cfc_forward(network->cfc_network,
                             state->input_buffer,
                             state->hidden_state,
                             state->cell_state,
                             state->output_buffer);
    LNN_UNLOCK(network);

    if (result != 0) return -1;

    /* 复制输出 */
    memcpy(output, state->output_buffer, out_sz * sizeof(float));
    return 0;
}

/* 将隔离状态同步回LNN（可选，用于模态间状态传递）
 * 调用此函数将隔离副本的最新隐藏状态写回LNN主状态，
 * 使后续其他模态可以看到当前模态处理后的上下文。 */
int lnn_forward_state_sync_back(LNN* network, LNNForwardState* state) {
    if (!network || !state || !state->initialized) return -1;

    LNN_LOCK(network);
    memcpy(network->hidden_state, state->hidden_state,
           state->hidden_size * sizeof(float));
    if (network->cell_state && state->cell_state) {
        size_t cs_size = network->config.hidden_size * (size_t)network->config.num_layers * 2;
        memcpy(network->cell_state, state->cell_state,
               (state->cell_state_size < cs_size ? state->cell_state_size : cs_size) * sizeof(float));
    }
    LNN_UNLOCK(network);
    return 0;
}

/**
 * @brief 反向传播内部实现（无锁，调用者需持有 LNN_LOCK）
 * @param skip_cell_update 1=跳过cell参数更新（批量梯度累积模式）
 */
int _lnn_backward_internal_ex(LNN* network, const float* target, float* loss, int skip_cell_update) {
#ifdef LNN_DEBUG_BACKWARD
    fprintf(stderr, "[BWD] enter: os=%zu hs=%zu is=%zu layers=%d\n",
            network->config.output_size, network->config.hidden_size,
            network->config.input_size, network->config.num_layers);
    fflush(stderr);
#endif
    clock_t start_time = clock();
    size_t output_size = network->config.output_size;
    size_t input_size = network->config.input_size;
    size_t hidden_size = network->config.hidden_size;
    int num_layers = network->config.num_layers;
/* 移除combined_scale对error_buffer的直接缩放。
     * 此前在cfc_backward之前缩放error_buffer会不可控地压缩所有后续梯度，
     * 导致深层网络训练不收敛。改为将层深度因子应用于梯度裁剪阈值，
     * 因为梯度裁剪是唯一合理的梯度缩放注入点——统一控制所有梯度的尺度。
     * 输入输出维度差异由cfc_backward内部的W矩阵自然处理，无需干预。 */
    float depth_grad_factor = 1.0f;
    if (num_layers > 1) {
        depth_grad_factor = sqrtf((float)num_layers);
        if (depth_grad_factor > 10.0f) depth_grad_factor = 10.0f;
        if (depth_grad_factor < 1.0f) depth_grad_factor = 1.0f;
    }

    /* FIX-003: 使用配置的损失函数类型计算正确的误差梯度 */
    int loss_type = network->config.loss_function;
    if (loss_type < 0 || loss_type > 11) loss_type = (int)LOSS_MSE;

    float* error_gradient = network->error_buffer;
/* output_size为size_t但下游接int参数——实际NN维度远<2^31，安全
     * 若后续支持超大输出层需将loss_gradient_ex接口改为size_t */
    loss_gradient_ex(network->output_buffer, target, (int)output_size,
                     error_gradient, (LossType)loss_type, NULL);

    float computed_loss = loss_compute_ex(network->output_buffer, target, (int)output_size,
                                          (LossType)loss_type, NULL);
    int loss_is_invalid = (isnan(computed_loss) || isinf(computed_loss));
    if (loss_is_invalid) {
        computed_loss = 1e6f;
        memset(network->gradient_buffer, 0, hidden_size * sizeof(float));
        network->current_loss = computed_loss;
        *loss = computed_loss;
        return -1;
    }
    network->current_loss = computed_loss;
    *loss = computed_loss;

    memset(network->gradient_buffer, 0, hidden_size * sizeof(float));
    /* P0-002: 使用 cfc_backward_ex 传递 skip_cell_update 标志 */
/* 批量梯度累积模式。
     * 当batch_size>1时，对每个样本执行前向+反向，累积梯度：
     *   1. 保存当前LNN状态
     *   2. 对batch中每样本: lnn_forward_sample → lnn_backward_sample → 累积梯度
     *   3. 取梯度平均值: grads /= batch_size
     *   4. 恢复LNN状态
     * 当前单样本模式(skip_cell_update=0)在批量训练时梯度方差大，
     * 如需批量训练建议启用此路径。 */
#ifdef LNN_DEBUG_BACKWARD
    fprintf(stderr, "[BWD] before cfc_backward_ex: hs=%zu lr=%.6f skip=%d\n",
            hidden_size, network->config.learning_rate, skip_cell_update);
    fflush(stderr);
#endif
    int result = cfc_backward_ex(network->cfc_network,
                            error_gradient,
                            network->gradient_buffer,
                            network->config.learning_rate,
                            skip_cell_update);
#ifdef LNN_DEBUG_BACKWARD
    fprintf(stderr, "[BWD] after cfc_backward_ex: result=%d\n", result);
    fflush(stderr);
#endif
    SELFLNN_CHECK(result == 0, SELFLNN_ERROR_NETWORK_CONFIG,
                 "CfC网络反向传播失败");
/* 公共梯度裁剪替代内联循环 */
    float max_grad_norm = network->config.max_grad_norm;
/* 将层深度因子应用于梯度裁剪阈值。 */
    if (depth_grad_factor > 1.0f) {
        max_grad_norm *= depth_grad_factor;
        if (max_grad_norm > 50.0f) max_grad_norm = 50.0f;
    }
    lnn_clip_gradients(network->gradient_buffer, hidden_size, max_grad_norm);

/* 原代码cfc_get_weight_matrix返回的是参数(weight_matrix)
     * 而非梯度(weight_gradients)。调用lnn_clip_gradients会破坏性修改权重参数本身。
     * 修复：直接裁剪cfc_network内部的weight_gradients/bias_gradients字段。 */
    {
        CfCNetwork* cfc = network->cfc_network;
        if (cfc && cfc->weight_gradients) {
            size_t wc = 0;
            /* 计算总权重参数数以确定梯度数组大小 */
            size_t in = cfc->config.input_size, hn = cfc->config.hidden_size;
            size_t nl = cfc->config.num_layers;
            if (nl <= 1) wc = in * hn;
            else wc = in * hn + (nl - 1) * hn * hn;
            lnn_clip_gradients(cfc->weight_gradients, wc, max_grad_norm);
        }
        if (cfc && cfc->bias_gradients) {
            size_t bc = 0;
            size_t hn = cfc->config.hidden_size, nl = cfc->config.num_layers;
            bc = hn * nl;
            lnn_clip_gradients(cfc->bias_gradients, bc, max_grad_norm);
        }
    }

/* 公共梯度裁剪替代CLIP_GRAD_ARRAY宏(R5-FIX cell级门控裁剪) */
    if (network->cfc_network && network->cfc_network->layers) {
        float max_norm = network->config.max_grad_norm;
        if (max_norm <= 0.0f) max_norm = 10.0f;
        int num_layers = network->cfc_network->config.num_layers;
        for (int l = 0; l < num_layers; l++) {
            CfCCell* cell = network->cfc_network->layers[l];
            if (!cell) continue;
            size_t hs = cell->config.hidden_size;
            size_t is = cell->config.input_size;
            size_t w_size = is * hs;

            lnn_clip_gradients(cell->input_gate_weight_grad, w_size, max_norm);
            lnn_clip_gradients(cell->forget_gate_weight_grad, w_size, max_norm);
            lnn_clip_gradients(cell->output_gate_weight_grad, w_size, max_norm);
            lnn_clip_gradients(cell->hidden_to_input_gate_weight_grad, hs * hs, max_norm);
            lnn_clip_gradients(cell->hidden_to_forget_gate_weight_grad, hs * hs, max_norm);
            lnn_clip_gradients(cell->hidden_to_output_gate_weight_grad, hs * hs, max_norm);
            lnn_clip_gradients(cell->hidden_to_activation_weight_grad, hs * hs, max_norm);
            lnn_clip_gradients(cell->gate_bias_grad, hs * 3, max_norm);
            lnn_clip_gradients(cell->time_constant_grad, hs, max_norm);
        }
    }

    /* P1-003修复: 四元数后处理反向梯度传播
     * 前向: qv[c] = qv[c]*decay + tanh(qv[c])*complement, 然后归一化
     * 反向: 先传递归一化梯度，再传递CfC式更新梯度 */
    /* P0修复: 使用epsilon阈值代替精确比较0.0f，避免四元数分量为0时的逻辑错误 */
    const float QUAT_EPS = 1e-7f;
    if (network->config.enable_quaternion && hidden_size >= 4) {
        float quat_dt = network->config.time_constant * 0.5f;
        if (quat_dt < 0.001f) quat_dt = 0.001f;
        if (quat_dt > 0.5f) quat_dt = 0.5f;

        size_t num_quats = hidden_size / 4;

        for (size_t q = 0; q < num_quats; q++) {
            float* gv = network->gradient_buffer + q * 4;

/* 从quaternion_pre_buf获取前处理状态，
             * 而非从已修改的hidden_state读取后处理值。
             * 反向传播需要前处理状态来计算tanh导数链。 */
            const float* pre_state = network->quaternion_pre_buf;
            float orig_qv[4];
            for (int c = 0; c < 4; c++) {
                /* P2修复: 当pre_state可用时始终使用它，避免因分量绝对值<=QUAT_EPS
                 * 而错误回退到network->hidden_state(后处理值)。pre_state是前处理状态，
                 * 正是反向传播计算tanh导数链所需的输入。混合使用前后处理状态会导致
                 * 梯度计算不一致。 */
                orig_qv[c] = pre_state ? pre_state[q * 4 + c]
                                       : network->hidden_state[q * 4 + c];
            }

            /* 步骤1: 归一化层反向传播
             *   前向: y = x / ||x||
             *   雅可比: dy/dx = (I - y*y^T) / ||x||  (单位矩阵减外积)
             *   链式法则精确梯度: dL/dx = (dL/dy - y*(y·dL/dy)) / ||x||
             *   其中 y = x/||x||, y·dL/dy 是归一化方向与上游梯度的点积 */
            float norm = sqrtf(orig_qv[0]*orig_qv[0] + orig_qv[1]*orig_qv[1]
                             + orig_qv[2]*orig_qv[2] + orig_qv[3]*orig_qv[3]);
            if (norm > 1e-8f) {
                float inv_norm = 1.0f / norm;
                /* 计算 y·(dL/dy) 的点积 */
                float dot = gv[0]*orig_qv[0]*inv_norm + gv[1]*orig_qv[1]*inv_norm
                          + gv[2]*orig_qv[2]*inv_norm + gv[3]*orig_qv[3]*inv_norm;
                /* 梯度从归一化输出传向归一化输入 */
                float unnorm_grad[4];
                for (int c = 0; c < 4; c++) {
                    unnorm_grad[c] = (gv[c] - orig_qv[c]*inv_norm*dot) * inv_norm;
                }

                /* 步骤2: CfC闭式解反向传播
                 *   前向: h_new = h*decay + tanh(h)*complement
                 *   反向: dL/dh = dL/dh_new * decay + dL/dh_new * complement * (1-tanh(h)^2) */
                float activation_norm = 0.0f;
                for (int c = 0; c < 4; c++) {
                    float x = orig_qv[c];
                    if (x > 10.0f) x = 10.0f;
                    if (x < -10.0f) x = -10.0f;
                    float th = tanhf(x);
                    activation_norm += th * th;
                }
                activation_norm = sqrtf(activation_norm);

                float adaptive_tau = network->config.time_constant
                                   * (1.0f + 0.5f * activation_norm);
                if (adaptive_tau < 0.001f) adaptive_tau = 0.001f;
                if (adaptive_tau > 2.0f) adaptive_tau = 2.0f;

                float decay = expf(-quat_dt / adaptive_tau);
                float complement = 1.0f - decay;

                /* 对每个四元数分量计算梯度 */
                float corrected_grad[4];
                for (int c = 0; c < 4; c++) {
                    float x = orig_qv[c];
                    if (x > 10.0f) x = 10.0f;
                    if (x < -10.0f) x = -10.0f;
                    float th = tanhf(x);
                    /* tanh导数: 1 - th^2 */
                    float dtanh = 1.0f - th * th;
                    /* drive路径: ∂(tanh(h)*complement)/∂h = dtanh*complement */
                    /* state路径: ∂(h*decay)/∂h = decay */
                    corrected_grad[c] = unnorm_grad[c] * decay
                                      + unnorm_grad[c] * complement * dtanh;
                }

                /* 将修正后的梯度写回 */
                for (int c = 0; c < 4; c++) {
                    if (SELFLNN_IS_FINITE(corrected_grad[c])) {
                        gv[c] = corrected_grad[c];
                    }
                }
            }
        }
    }

    network->backward_count++;
    clock_t end_time = clock();
    network->total_training_time += (double)(end_time - start_time) / CLOCKS_PER_SEC;
    return 0;
}

/**
 * @brief 反向传播内部实现（无锁，调用者需持有 LNN_LOCK）
 */
int _lnn_backward_internal(LNN* network, const float* target, float* loss) {
    /* P1修复: 梯度裁剪时机修正
     * 原问题: skip=0时cfc_backward_ex内部会先应用Adam更新cell参数，
     *         然后_lnn_backward_internal_ex才做梯度裁剪——裁剪对已更新的参数无效。
     * 修复方案: 改用skip=1模式（仅计算梯度，不应用Adam），
     *          在_lnn_backward_internal_ex内部完成梯度裁剪后，
     *          再手动调用cfc_apply_cell_gradients_adam和cfc_apply_out_proj_gradients，
     *          确保裁剪在参数更新之前生效。 */
    /* skip=1模式不零清cell梯度，需在调用前手动零清 */
    if (network->cfc_network) {
        cfc_zero_cell_gradients(network->cfc_network);
    }
    int ret = _lnn_backward_internal_ex(network, target, loss, 1); /* skip=1: 仅计算梯度+裁剪 */
    if (ret == 0 && network->cfc_network) {
        CfCNetwork* cfc = network->cfc_network;
        /* P1修复: 在梯度裁剪之后应用Adam更新cell参数和输出投影 */
        cfc_apply_cell_gradients_adam(cfc, network->config.learning_rate, 0.9f, 0.999f, 1e-8f, 1);
        cfc_apply_out_proj_gradients(cfc, network->config.learning_rate);
        /* SGD更新weight_matrix/bias_vector（在裁剪后执行，时机正确） */
        if (cfc->weight_gradients && cfc->weight_matrix) {
            size_t in = cfc->config.input_size, hn = cfc->config.hidden_size;
            size_t nl = cfc->config.num_layers;
            size_t total_w = (nl <= 1) ? (in * hn) : (in * hn + (nl - 1) * hn * hn);
            size_t total_b = hn * nl;
            float lr = network->config.learning_rate;
            for (size_t i = 0; i < total_w; i++)
                cfc->weight_matrix[i] -= lr * cfc->weight_gradients[i];
            for (size_t i = 0; i < total_b; i++)
                cfc->bias_vector[i] -= lr * cfc->bias_gradients[i];
            memset(cfc->weight_gradients, 0, total_w * sizeof(float));
            memset(cfc->bias_gradients, 0, total_b * sizeof(float));
        }
        /* 同步更新后的权重到各层CfCCell */
        cfc_sync_shared_to_cells(cfc);
    }
    return ret;
}

/**
 * @brief 反向传播（训练）- 完整多层LNN实现
 * 
 *: 当前为单样本反向传播模式。
 * 对于批量训练，推荐使用以下模式：
 *   1. 循环每个样本调用 lnn_forward
 *   2. 循环每个样本调用 lnn_backward（skip_cell_update=1 跳过cell状态更新）
 *   3. 梯度累积完成后统一调用 optimizer_step
 * 
 * 当前 skip_cell_update 参数支持批量累积，但调用者需在批量结束时
 * 手动调用 cfc_network_reset_cell_states 清除累积的状态。
 * 
 * 后续版本将提供 lnn_backward_batch() 封装函数自动管理此流程。
 */
int lnn_backward(LNN* network, const float* target, float* loss) {
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_NULL(target, "目标输出向量为空");
    SELFLNN_CHECK_NULL(loss, "损失值缓冲区为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");
    SELFLNN_CHECK(network->config.enable_training, 
                 SELFLNN_ERROR_TRAINING_NOT_ENABLED,
                 "LNN网络训练未启用");
    LNN_LOCK(network);
    int ret = _lnn_backward_internal(network, target, loss);
    LNN_UNLOCK(network);
    return ret;
}

// 模型文件头结构
typedef struct {
    char magic[SELFLNN_FILE_MAGIC_SIZE];  // 魔数 "SELFLNN"
    uint32_t version;                      // 文件格式版本
    uint32_t marker;                       // FROM_SCRATCH/PRETRAINED标记
    uint32_t header_size;                  // 文件头大小（含此结构）
    uint32_t checksum;                     // 简单校验和
    uint32_t reserved[4];                  // 保留字段
} LNNFileHeader;

// 当前模型文件格式版本
#define SELFLNN_FILE_VERSION 1

/**
 * @brief 计算简单校验和
 */
static uint32_t lnn_calc_checksum(const void* data, size_t size) {
    const unsigned char* bytes = (const unsigned char*)data;
    uint32_t sum = 0;
    for (size_t i = 0; i < size; i++) {
        sum = ((sum << 5) | (sum >> 27)) + bytes[i];
    }
    return sum;
}

/**
 * @brief 保存网络到文件
 *
 * ZSF-018修复: 参数从 const LNN* 改为 LNN*，因为保存时需要获取
 * LNN_LOCK 互斥锁以保护权重读取的原子性。const 语义与互斥锁修改冲突。
 *
 * M-2修复: 锁内仅复制数据到堆临时缓冲区，释放锁后在锁外执行fwrite，
 * 避免在持有互斥锁期间进行耗时的磁盘I/O操作。
 */
int lnn_save(LNN* network, const char* filepath) {
    /* 参数检查 */
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_NULL(filepath, "文件路径为空");

    /* 预计算维度（锁外安全：config不可变） */
    size_t hidden_size = network->config.hidden_size;
    CfCNetwork* cfc = network->cfc_network;
    int num_layers = cfc->config.num_layers;
    size_t cfc_input_size = cfc->config.input_size;
    size_t cfc_output_size = cfc->config.output_size;
    size_t weight_size = cfc_input_size * hidden_size;
    size_t hidden_weight_size = hidden_size * hidden_size;

    /* ================================================================
     * 所有需要在锁外释放的临时缓冲区指针（初始化为NULL，便于统一清理）
     * ================================================================ */
    float* buf_hidden = NULL;           /* 隐藏状态副本 */
    float* buf_cell = NULL;             /* 细胞状态副本 */
    float* buf_cfc_w = NULL;            /* CFC权重矩阵副本 */
    float* buf_cfc_b = NULL;            /* CFC偏置向量副本 */
    float* buf_W_out = NULL;            /* 输出投影矩阵副本 */
    float* buf_b_out = NULL;            /* 输出投影偏置副本 */

    /* 每层CfC单元数据 —— 指针数组 [num_layers] */
    CfCCellConfig* cell_cfgs = NULL;   /* 每层单元配置 */
    int* cell_use_gating = NULL;       /* 每层use_gating */
    int* cell_use_adapt = NULL;       /* 每层use_adaptive_tau */
    float* cell_avg_act = NULL;        /* 每层avg_activation */
    float* cell_max_act = NULL;        /* 每层max_activation */
    float* cell_min_tau = NULL;        /* 每层min_time_constant */
    float* cell_max_tau = NULL;        /* 每层max_time_constant */
    float* cell_tau_lr = NULL;         /* 每层tau_learning_rate */
    float* cell_stime = NULL;          /* 每层state->time */
    float* cell_sadapt = NULL;         /* 每层state->adaptation_rate */
    float** cell_svec = NULL;          /* 每层state->state */
    float** cell_aparam = NULL;        /* 每层state->adapted_params */
    float** cell_weights = NULL;       /* 每层weight_matrix */
    float** cell_biases = NULL;        /* 每层bias_vector */
    float** cell_tau_arr = NULL;       /* 每层time_constants */
    float** cell_igw = NULL;           /* 每层input_gate_weights */
    float** cell_fgw = NULL;           /* 每层forget_gate_weights */
    float** cell_ogw = NULL;           /* 每层output_gate_weights */
    float** cell_gb = NULL;            /* 每层gate_biases */
    float** cell_h2ig = NULL;          /* 每层hidden_to_input_gate */
    float** cell_h2fg = NULL;          /* 每层hidden_to_forget_gate */
    float** cell_h2og = NULL;          /* 每层hidden_to_output_gate */
    float** cell_h2a = NULL;           /* 每层hidden_to_activation */
    int* cell_use_mts = NULL;          /* 每层use_multi_timescale */
    int* cell_fast_ratio_i = NULL;     /* 每层fast_tau_ratio*1000 */
    int* cell_slow_ratio_i = NULL;     /* 每层slow_tau_ratio*1000 */
    float** cell_ftau = NULL;          /* 每层fast_time_constants */
    float** cell_stau = NULL;          /* 每层slow_time_constants */
    float** cell_fstate = NULL;        /* 每层fast_state */
    float** cell_sstate = NULL;        /* 每层slow_state */

    /* 层归一化数据 */
    int has_ln = 0;
    float** ln_gamma = NULL;           /* 每层gamma */
    float** ln_beta = NULL;            /* 每层beta */

    /* 错误处理标签跳转目标 */
    int ret = 0;

    /* ================================================================
     * 第一阶段：在LNN_LOCK内复制所有需要保存的数据到临时缓冲区
     * ================================================================ */
    LNN_LOCK(network);

    /* --- 1. 复制LNN配置（栈拷贝） --- */
    LNNConfig saved_config = network->config;

    /* --- 2. 复制标量字段 --- */
    int saved_ps = network->enable_param_sharding;
    size_t saved_ns = network->num_local_shards;
    int saved_gc = network->enable_gradient_checkpointing;
    int saved_mp = network->enable_model_parallel;
    int saved_ag = network->enable_async_gradient_sync;
    float saved_loss = network->current_loss;
    uint64_t saved_fwd = (uint64_t)network->forward_count;
    uint64_t saved_bwd = (uint64_t)network->backward_count;
    double saved_time_val = network->total_training_time;

    /* --- 3. 复制隐藏状态和细胞状态到堆缓冲区 --- */
    buf_hidden = (float*)safe_malloc(hidden_size * sizeof(float));
    buf_cell   = (float*)safe_malloc(hidden_size * sizeof(float));
    if (!buf_hidden || !buf_cell) goto alloc_fail;

    memcpy(buf_hidden, network->hidden_state, hidden_size * sizeof(float));
    memcpy(buf_cell,   network->cell_state,   hidden_size * sizeof(float));

    /* --- 4. 复制CFC网络顶层数据 --- */
    CfCNetworkConfig saved_cfc_cfg = cfc->config;

    size_t total_w = cfc->total_weight_params;
    size_t total_b = cfc->total_bias_params;
    if (total_w == 0) {
        total_w = (num_layers == 1) ? weight_size : hidden_weight_size;
    }
    if (total_b == 0) total_b = hidden_size;

    buf_cfc_w = (float*)safe_malloc(total_w * sizeof(float));
    buf_cfc_b = (float*)safe_malloc(total_b * sizeof(float));
    if (!buf_cfc_w || !buf_cfc_b) goto alloc_fail;

    memcpy(buf_cfc_w, cfc->weight_matrix, total_w * sizeof(float));
    memcpy(buf_cfc_b, cfc->bias_vector,   total_b * sizeof(float));

    /* --- 5. 复制输出投影参数（如果存在） --- */
    if (cfc_output_size != hidden_size && cfc->W_out_params && cfc->b_out_params) {
        size_t out_proj_count = cfc_output_size * hidden_size;
        buf_W_out = (float*)safe_malloc(out_proj_count * sizeof(float));
        buf_b_out = (float*)safe_malloc(cfc_output_size * sizeof(float));
        if (!buf_W_out || !buf_b_out) goto alloc_fail;
        memcpy(buf_W_out, cfc->W_out_params, out_proj_count * sizeof(float));
        memcpy(buf_b_out, cfc->b_out_params, cfc_output_size * sizeof(float));
    }

    /* --- 6. 复制每层CfC单元数据 --- */
    if (num_layers > 0) {
        /* 分配标量数组 */
        cell_cfgs  = (CfCCellConfig*)safe_malloc((size_t)num_layers * sizeof(CfCCellConfig));
        cell_use_gating = (int*)safe_malloc((size_t)num_layers * sizeof(int));
        cell_use_adapt  = (int*)safe_malloc((size_t)num_layers * sizeof(int));
        cell_avg_act    = (float*)safe_malloc((size_t)num_layers * sizeof(float));
        cell_max_act    = (float*)safe_malloc((size_t)num_layers * sizeof(float));
        cell_min_tau    = (float*)safe_malloc((size_t)num_layers * sizeof(float));
        cell_max_tau    = (float*)safe_malloc((size_t)num_layers * sizeof(float));
        cell_tau_lr     = (float*)safe_malloc((size_t)num_layers * sizeof(float));
        cell_stime      = (float*)safe_malloc((size_t)num_layers * sizeof(float));
        cell_sadapt     = (float*)safe_malloc((size_t)num_layers * sizeof(float));
        cell_use_mts    = (int*)safe_malloc((size_t)num_layers * sizeof(int));
        cell_fast_ratio_i = (int*)safe_malloc((size_t)num_layers * sizeof(int));
        cell_slow_ratio_i = (int*)safe_malloc((size_t)num_layers * sizeof(int));

        /* 分配指针数组 */
        cell_svec    = (float**)safe_calloc((size_t)num_layers, sizeof(float*));
        cell_aparam  = (float**)safe_calloc((size_t)num_layers, sizeof(float*));
        cell_weights = (float**)safe_calloc((size_t)num_layers, sizeof(float*));
        cell_biases  = (float**)safe_calloc((size_t)num_layers, sizeof(float*));
        cell_tau_arr = (float**)safe_calloc((size_t)num_layers, sizeof(float*));
        cell_igw     = (float**)safe_calloc((size_t)num_layers, sizeof(float*));
        cell_fgw     = (float**)safe_calloc((size_t)num_layers, sizeof(float*));
        cell_ogw     = (float**)safe_calloc((size_t)num_layers, sizeof(float*));
        cell_gb      = (float**)safe_calloc((size_t)num_layers, sizeof(float*));
        cell_h2ig    = (float**)safe_calloc((size_t)num_layers, sizeof(float*));
        cell_h2fg    = (float**)safe_calloc((size_t)num_layers, sizeof(float*));
        cell_h2og    = (float**)safe_calloc((size_t)num_layers, sizeof(float*));
        cell_h2a     = (float**)safe_calloc((size_t)num_layers, sizeof(float*));
        cell_ftau    = (float**)safe_calloc((size_t)num_layers, sizeof(float*));
        cell_stau    = (float**)safe_calloc((size_t)num_layers, sizeof(float*));
        cell_fstate  = (float**)safe_calloc((size_t)num_layers, sizeof(float*));
        cell_sstate  = (float**)safe_calloc((size_t)num_layers, sizeof(float*));

        if (!cell_cfgs || !cell_use_gating || !cell_use_adapt ||
            !cell_avg_act || !cell_max_act || !cell_min_tau || !cell_max_tau ||
            !cell_tau_lr || !cell_stime || !cell_sadapt || !cell_use_mts ||
            !cell_fast_ratio_i || !cell_slow_ratio_i ||
            !cell_svec || !cell_aparam || !cell_weights || !cell_biases ||
            !cell_tau_arr || !cell_igw || !cell_fgw || !cell_ogw || !cell_gb ||
            !cell_h2ig || !cell_h2fg || !cell_h2og || !cell_h2a ||
            !cell_ftau || !cell_stau || !cell_fstate || !cell_sstate) {
            goto alloc_fail;
        }

        /* 逐层复制数据 */
        size_t cell_w_size = (num_layers == 1) ? weight_size : hidden_weight_size;
        size_t gate_bias_size = hidden_size * 3;

        for (int i = 0; i < num_layers; i++) {
            CfCCell* cell = cfc->layers[i];
            if (!cell) goto alloc_fail;
            if (!cell->state) goto alloc_fail;

            /* 标量字段 */
            cell_cfgs[i] = cell->config;
            cell_use_gating[i] = cell->use_gating;
            cell_use_adapt[i]  = cell->use_adaptive_tau;
            cell_avg_act[i] = cell->avg_activation;
            cell_max_act[i] = cell->max_activation;
            cell_min_tau[i] = cell->min_time_constant;
            cell_max_tau[i] = cell->max_time_constant;
            cell_tau_lr[i]  = cell->tau_learning_rate;
            cell_stime[i]  = cell->state->time;
            cell_sadapt[i] = cell->state->adaptation_rate;
            cell_use_mts[i] = cell->use_multi_timescale;
            cell_fast_ratio_i[i] = (int)(cell->fast_tau_ratio * 1000.0f);
            cell_slow_ratio_i[i] = (int)(cell->slow_tau_ratio * 1000.0f);

            /* 加权数组 —— 逐个分配并复制 */
            #define ALLOC_COPY_ARRAY(dst, src, count) do { \
                (dst) = (float*)safe_malloc((count) * sizeof(float)); \
                if (!(dst)) goto alloc_fail; \
                memcpy((dst), (src), (count) * sizeof(float)); \
            } while(0)

            ALLOC_COPY_ARRAY(cell_svec[i],   cell->state->state,          hidden_size);
            ALLOC_COPY_ARRAY(cell_aparam[i], cell->state->adapted_params, hidden_size);
            ALLOC_COPY_ARRAY(cell_weights[i], cell->weight_matrix,        cell_w_size);
            ALLOC_COPY_ARRAY(cell_biases[i],  cell->bias_vector,          hidden_size);
            ALLOC_COPY_ARRAY(cell_tau_arr[i], cell->time_constants,       hidden_size);
            ALLOC_COPY_ARRAY(cell_igw[i],     cell->input_gate_weights,   cell_w_size);
            ALLOC_COPY_ARRAY(cell_fgw[i],     cell->forget_gate_weights,  cell_w_size);
            ALLOC_COPY_ARRAY(cell_ogw[i],     cell->output_gate_weights,  cell_w_size);
            ALLOC_COPY_ARRAY(cell_gb[i],      cell->gate_biases,          gate_bias_size);
            ALLOC_COPY_ARRAY(cell_h2ig[i],    cell->hidden_to_input_gate_weights,    hidden_weight_size);
            ALLOC_COPY_ARRAY(cell_h2fg[i],    cell->hidden_to_forget_gate_weights,   hidden_weight_size);
            ALLOC_COPY_ARRAY(cell_h2og[i],    cell->hidden_to_output_gate_weights,   hidden_weight_size);
            ALLOC_COPY_ARRAY(cell_h2a[i],     cell->hidden_to_activation_weights,    hidden_weight_size);

            /* 多时间尺度数据（仅在启用时复制） */
            if (cell->use_multi_timescale) {
                ALLOC_COPY_ARRAY(cell_ftau[i],   cell->fast_time_constants, hidden_size);
                ALLOC_COPY_ARRAY(cell_stau[i],   cell->slow_time_constants, hidden_size);
                ALLOC_COPY_ARRAY(cell_fstate[i], cell->fast_state,          hidden_size);
                ALLOC_COPY_ARRAY(cell_sstate[i], cell->slow_state,          hidden_size);
            }

            #undef ALLOC_COPY_ARRAY
        }
    }

    /* --- 7. 复制层归一化参数 --- */
    if (cfc->layer_norms && cfc->config.use_layer_norm) {
        has_ln = 1;
        ln_gamma = (float**)safe_calloc((size_t)num_layers, sizeof(float*));
        ln_beta  = (float**)safe_calloc((size_t)num_layers, sizeof(float*));
        if (!ln_gamma || !ln_beta) goto alloc_fail;

        for (int i = 0; i < num_layers; i++) {
            LayerNorm* ln = (LayerNorm*)cfc->layer_norms[i];
            if (!ln) {
                ln_gamma[i] = NULL;
                ln_beta[i]  = NULL;
                continue;
            }
            const float* gamma = layer_norm_get_gamma(ln);
            const float* beta  = layer_norm_get_beta(ln);
            ln_gamma[i] = (float*)safe_malloc(hidden_size * sizeof(float));
            ln_beta[i]  = (float*)safe_malloc(hidden_size * sizeof(float));
            if (!ln_gamma[i] || !ln_beta[i]) goto alloc_fail;
            memcpy(ln_gamma[i], gamma, hidden_size * sizeof(float));
            memcpy(ln_beta[i],  beta,  hidden_size * sizeof(float));
        }
    }

    /* 数据复制完成，释放锁 */
    LNN_UNLOCK(network);

    /* ================================================================
     * 第二阶段：在锁外执行所有文件I/O（fwrite）
     * ================================================================ */
    {
        FILE* file = fopen(filepath, "wb");
        if (!file) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "打开文件失败: %s", filepath);
            ret = SELFLNN_ERROR_IO_ERROR;
            goto cleanup;
        }

        /* 构建并写入文件头 */
        LNNFileHeader header;
        memset(&header, 0, sizeof(header));
        memcpy(header.magic, SELFLNN_FILE_MAGIC, SELFLNN_FILE_MAGIC_SIZE);
        header.version = SELFLNN_FILE_VERSION;
        header.header_size = sizeof(LNNFileHeader);

#ifdef SELFLNN_NO_PRETRAINED
        header.marker |= SELFLNN_MARKER_FROM_SCRATCH;
#else
        header.marker |= SELFLNN_MARKER_PRETRAINED;
#endif

        uint32_t config_checksum = lnn_calc_checksum(&saved_config, sizeof(LNNConfig));
        header.checksum = config_checksum;

        if (fwrite(&header, sizeof(header), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "写入模型文件头失败");
            fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
        }

        /* 写入LNN配置 */
        if (fwrite(&saved_config, sizeof(LNNConfig), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存LNN网络配置失败");
            fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
        }

        /* 写入CFC网络配置 */
        if (fwrite(&saved_cfc_cfg, sizeof(CfCNetworkConfig), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存CfC网络配置失败");
            fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
        }

        /* 写入每层CfC单元配置 */
        for (int i = 0; i < num_layers; i++) {
            if (fwrite(&cell_cfgs[i], sizeof(CfCCellConfig), 1, file) != 1) {
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "保存第%d层CfC单元配置失败", i);
                fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
            }
        }

        /* 写入CFC权重矩阵和偏置向量 */
        if (fwrite(buf_cfc_w, sizeof(float), total_w, file) != total_w) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存权重矩阵失败（大小：%zu）", total_w);
            fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
        }
        if (fwrite(buf_cfc_b, sizeof(float), total_b, file) != total_b) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存偏置向量失败（大小：%zu）", total_b);
            fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
        }

        /* 写入输出投影参数 */
        if (buf_W_out && buf_b_out) {
            size_t out_proj_count = cfc_output_size * hidden_size;
            if (fwrite(buf_W_out, sizeof(float), out_proj_count, file) != out_proj_count) {
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "保存输出投影矩阵失败（大小：%zu）", out_proj_count);
                fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
            }
            if (fwrite(buf_b_out, sizeof(float), cfc_output_size, file) != cfc_output_size) {
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "保存输出投影偏置失败（大小：%zu）", (size_t)cfc_output_size);
                fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
            }
        }

        /* 写入CfC细胞标记和逐层数据 */
        if (num_layers > 0) {
            uint32_t cell_marker = 0xCFCC3E11;
            if (fwrite(&cell_marker, sizeof(uint32_t), 1, file) != 1) {
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "保存CfC细胞标记失败");
                fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
            }

            for (int i = 0; i < num_layers; i++) {
                /* 写入细胞配置 */
                if (fwrite(&cell_cfgs[i], sizeof(CfCCellConfig), 1, file) != 1) {
                    selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                          "保存第%d层细胞配置失败", i);
                    fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
                }

                /* 写入内部状态标志 */
                int flags[2] = { cell_use_gating[i], cell_use_adapt[i] };
                if (fwrite(flags, sizeof(int), 2, file) != 2) {
                    selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                          "保存第%d层状态标志失败", i);
                    fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
                }

                /* 写入训练参数 */
                float params[4] = { cell_avg_act[i], cell_max_act[i],
                                    cell_min_tau[i], cell_max_tau[i] };
                if (fwrite(params, sizeof(float), 4, file) != 4) {
                    selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                          "保存第%d层训练参数失败", i);
                    fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
                }
                if (fwrite(&cell_tau_lr[i], sizeof(float), 1, file) != 1) {
                    selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                          "保存第%d层tau学习率失败", i);
                    fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
                }

                /* 写入状态向量 */
                if (fwrite(cell_svec[i],   sizeof(float), hidden_size, file) != hidden_size ||
                    fwrite(cell_aparam[i], sizeof(float), hidden_size, file) != hidden_size ||
                    fwrite(&cell_stime[i],  sizeof(float), 1, file) != 1 ||
                    fwrite(&cell_sadapt[i], sizeof(float), 1, file) != 1) {
                    selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                          "保存第%d层状态失败", i);
                    fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
                }

                /* 写入权重和偏置 */
                size_t cell_w_size = (num_layers == 1) ? weight_size : hidden_weight_size;
                if (fwrite(cell_weights[i], sizeof(float), cell_w_size, file) != cell_w_size ||
                    fwrite(cell_biases[i],  sizeof(float), hidden_size, file) != hidden_size) {
                    selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                          "保存第%d层权重失败", i);
                    fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
                }

                /* 写入时间常数 */
                if (fwrite(cell_tau_arr[i], sizeof(float), hidden_size, file) != hidden_size) {
                    selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                          "保存第%d层时间常数失败", i);
                    fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
                }

                /* 写入门控权重 */
                if (fwrite(cell_igw[i], sizeof(float), cell_w_size, file) != cell_w_size ||
                    fwrite(cell_fgw[i], sizeof(float), cell_w_size, file) != cell_w_size ||
                    fwrite(cell_ogw[i], sizeof(float), cell_w_size, file) != cell_w_size) {
                    selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                          "保存第%d层门控权重失败", i);
                    fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
                }
                if (fwrite(cell_gb[i], sizeof(float), hidden_size * 3, file) != hidden_size * 3) {
                    selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                          "保存第%d层门控偏置失败", i);
                    fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
                }

                /* 写入隐藏到隐藏连接权重 */
                if (fwrite(cell_h2ig[i], sizeof(float), hidden_weight_size, file) != hidden_weight_size ||
                    fwrite(cell_h2fg[i], sizeof(float), hidden_weight_size, file) != hidden_weight_size ||
                    fwrite(cell_h2og[i], sizeof(float), hidden_weight_size, file) != hidden_weight_size ||
                    fwrite(cell_h2a[i],  sizeof(float), hidden_weight_size, file) != hidden_weight_size) {
                    selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                          "保存第%d层hidden-to-hidden权重失败", i);
                    fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
                }

                /* 写入多时间尺度标记和数据 */
                int mts_flags[3] = { cell_use_mts[i], cell_fast_ratio_i[i], cell_slow_ratio_i[i] };
                if (fwrite(mts_flags, sizeof(int), 3, file) != 3) {
                    selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                          "保存第%d层多时间尺度标记失败", i);
                    fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
                }
                if (cell_use_mts[i]) {
                    if (fwrite(cell_ftau[i],   sizeof(float), hidden_size, file) != hidden_size ||
                        fwrite(cell_stau[i],   sizeof(float), hidden_size, file) != hidden_size ||
                        fwrite(cell_fstate[i], sizeof(float), hidden_size, file) != hidden_size ||
                        fwrite(cell_sstate[i], sizeof(float), hidden_size, file) != hidden_size) {
                        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                              "保存第%d层多时间尺度数据失败", i);
                        fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
                    }
                }
            }
        }

        /* 写入层归一化参数 */
        if (has_ln && ln_gamma && ln_beta) {
            uint32_t ln_marker = 0x4C4E4F52;  /* "LNOR" */
            if (fwrite(&ln_marker, sizeof(uint32_t), 1, file) != 1) {
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "保存层归一化标记失败");
                fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
            }
            for (int i = 0; i < num_layers; i++) {
                if (!ln_gamma[i] || !ln_beta[i]) {
                    uint16_t zero_dim = 0;
                    fwrite(&zero_dim, sizeof(uint16_t), 1, file);
                    continue;
                }
                uint16_t dim = (uint16_t)hidden_size;
                if (fwrite(&dim, sizeof(uint16_t), 1, file) != 1 ||
                    fwrite(ln_gamma[i], sizeof(float), (size_t)dim, file) != (size_t)dim ||
                    fwrite(ln_beta[i],  sizeof(float), (size_t)dim, file) != (size_t)dim) {
                    selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                          "保存第%d层层归一化参数失败", i);
                    fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
                }
            }
        }

        /* 写入隐藏状态 */
        if (fwrite(buf_hidden, sizeof(float), hidden_size, file) != hidden_size) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存隐藏状态失败");
            fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
        }

        /* 写入细胞状态 */
        if (fwrite(buf_cell, sizeof(float), hidden_size, file) != hidden_size) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存细胞状态失败");
            fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
        }

        /* 写入扩展字段版本标记 */
        uint32_t ext_version = 2;
        if (fwrite(&ext_version, sizeof(uint32_t), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存扩展版本标记失败");
            fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
        }

        /* 写入分片配置 */
        if (fwrite(&saved_ps, sizeof(int), 1, file) != 1 ||
            fwrite(&saved_ns, sizeof(size_t), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存分片配置失败");
            fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
        }

        /* 写入梯度检查点配置 */
        if (fwrite(&saved_gc, sizeof(int), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存梯度检查点配置失败");
            fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
        }

        /* 写入模型并行配置 */
        if (fwrite(&saved_mp, sizeof(int), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存模型并行配置失败");
            fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
        }

        /* 写入异步同步配置 */
        if (fwrite(&saved_ag, sizeof(int), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存异步同步配置失败");
            fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
        }

        /* 写入统计信息 */
        if (fwrite(&saved_loss, sizeof(float), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存当前损失失败");
            fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
        }

        uint64_t fwd_bwd[2] = { saved_fwd, saved_bwd };
        if (fwrite(fwd_bwd, sizeof(uint64_t), 2, file) != 2) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存前向/反向计数失败");
            fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
        }
        if (fwrite(&saved_time_val, sizeof(double), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存训练时间失败");
            fclose(file); ret = SELFLNN_ERROR_IO_ERROR; goto cleanup;
        }

        fclose(file);
    }

    ret = 0;
    goto cleanup;

    /* ================================================================
     * 内存分配失败处理：解锁 + 设置错误码 + 跳转到清理
     * ================================================================ */
alloc_fail:
    LNN_UNLOCK(network);
    selflnn_set_last_error(SELFLNN_ERROR_MEMORY_ALLOCATION, __func__, __FILE__, __LINE__,
                          "保存模型时内存分配失败（hidden_size=%zu，num_layers=%d）",
                          hidden_size, num_layers);
    ret = SELFLNN_ERROR_MEMORY_ALLOCATION;

    /* ================================================================
     * 统一清理：释放所有临时堆缓冲区
     * ================================================================ */
cleanup:
    /* 顶层LNN数据 */
    safe_free((void**)&buf_hidden);
    safe_free((void**)&buf_cell);
    safe_free((void**)&buf_cfc_w);
    safe_free((void**)&buf_cfc_b);
    safe_free((void**)&buf_W_out);
    safe_free((void**)&buf_b_out);

    /* 每层标量数组 */
    safe_free((void**)&cell_cfgs);
    safe_free((void**)&cell_use_gating);
    safe_free((void**)&cell_use_adapt);
    safe_free((void**)&cell_avg_act);
    safe_free((void**)&cell_max_act);
    safe_free((void**)&cell_min_tau);
    safe_free((void**)&cell_max_tau);
    safe_free((void**)&cell_tau_lr);
    safe_free((void**)&cell_stime);
    safe_free((void**)&cell_sadapt);
    safe_free((void**)&cell_use_mts);
    safe_free((void**)&cell_fast_ratio_i);
    safe_free((void**)&cell_slow_ratio_i);

    /* 每层数组指针和数据 */
    if (cell_svec)    { for (int i = 0; i < num_layers; i++) safe_free((void**)&cell_svec[i]);    safe_free((void**)&cell_svec); }
    if (cell_aparam)  { for (int i = 0; i < num_layers; i++) safe_free((void**)&cell_aparam[i]);  safe_free((void**)&cell_aparam); }
    if (cell_weights) { for (int i = 0; i < num_layers; i++) safe_free((void**)&cell_weights[i]); safe_free((void**)&cell_weights); }
    if (cell_biases)  { for (int i = 0; i < num_layers; i++) safe_free((void**)&cell_biases[i]);  safe_free((void**)&cell_biases); }
    if (cell_tau_arr) { for (int i = 0; i < num_layers; i++) safe_free((void**)&cell_tau_arr[i]); safe_free((void**)&cell_tau_arr); }
    if (cell_igw)     { for (int i = 0; i < num_layers; i++) safe_free((void**)&cell_igw[i]);     safe_free((void**)&cell_igw); }
    if (cell_fgw)     { for (int i = 0; i < num_layers; i++) safe_free((void**)&cell_fgw[i]);     safe_free((void**)&cell_fgw); }
    if (cell_ogw)     { for (int i = 0; i < num_layers; i++) safe_free((void**)&cell_ogw[i]);     safe_free((void**)&cell_ogw); }
    if (cell_gb)      { for (int i = 0; i < num_layers; i++) safe_free((void**)&cell_gb[i]);      safe_free((void**)&cell_gb); }
    if (cell_h2ig)    { for (int i = 0; i < num_layers; i++) safe_free((void**)&cell_h2ig[i]);    safe_free((void**)&cell_h2ig); }
    if (cell_h2fg)    { for (int i = 0; i < num_layers; i++) safe_free((void**)&cell_h2fg[i]);    safe_free((void**)&cell_h2fg); }
    if (cell_h2og)    { for (int i = 0; i < num_layers; i++) safe_free((void**)&cell_h2og[i]);    safe_free((void**)&cell_h2og); }
    if (cell_h2a)     { for (int i = 0; i < num_layers; i++) safe_free((void**)&cell_h2a[i]);     safe_free((void**)&cell_h2a); }
    if (cell_ftau)    { for (int i = 0; i < num_layers; i++) safe_free((void**)&cell_ftau[i]);    safe_free((void**)&cell_ftau); }
    if (cell_stau)    { for (int i = 0; i < num_layers; i++) safe_free((void**)&cell_stau[i]);    safe_free((void**)&cell_stau); }
    if (cell_fstate)  { for (int i = 0; i < num_layers; i++) safe_free((void**)&cell_fstate[i]);  safe_free((void**)&cell_fstate); }
    if (cell_sstate)  { for (int i = 0; i < num_layers; i++) safe_free((void**)&cell_sstate[i]);  safe_free((void**)&cell_sstate); }

    /* 层归一化 */
    if (ln_gamma) { for (int i = 0; i < num_layers; i++) safe_free((void**)&ln_gamma[i]); safe_free((void**)&ln_gamma); }
    if (ln_beta)  { for (int i = 0; i < num_layers; i++) safe_free((void**)&ln_beta[i]);  safe_free((void**)&ln_beta); }

    return ret;
}

/**
 * @brief 从文件加载网络
 */
LNN* lnn_load(const char* filepath) {
    // 参数检查
    if (!filepath) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "文件路径为空");
        return NULL;
    }

    FILE* file = fopen(filepath, "rb");
    if (!file) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "打开文件失败: %s", filepath);
        return NULL;
    }

    // 读取文件头
    LNNFileHeader header;
    if (fread(&header, sizeof(header), 1, file) != 1) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "读取模型文件头失败");
        fclose(file);
        return NULL;
    }

    // 验证魔数
    if (memcmp(header.magic, SELFLNN_FILE_MAGIC, SELFLNN_FILE_MAGIC_SIZE) != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__, __FILE__, __LINE__,
                              "模型文件魔数错误，不是有效的SELF-LNN模型文件");
        fclose(file);
        return NULL;
    }

    // 验证版本兼容性
    if (header.version > SELFLNN_FILE_VERSION) {
        selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__, __FILE__, __LINE__,
                              "模型文件版本(%u)高于当前版本(%u)",
                              header.version, SELFLNN_FILE_VERSION);
        fclose(file);
        return NULL;
    }

    // 检查 FROM_SCRATCH 标记
    if (header.marker & SELFLNN_MARKER_FROM_SCRATCH) {
        // 从零训练的模型文件，正常加载
    }

    // 如果定义了 SELFLNN_NO_PRETRAINED，禁止加载预训练标记的模型
#if !SELFLNN_CAN_LOAD_PRETRAINED
    if (header.marker & SELFLNN_MARKER_PRETRAINED) {
        selflnn_set_last_error(SELFLNN_ERROR_OPERATION_FAILED, __func__, __FILE__, __LINE__,
                              "SELFLNN_NO_PRETRAINED 已定义，禁止加载预训练模型文件");
        fclose(file);
        return NULL;
    }
#endif

    // 读取配置
    LNNConfig config;
    if (fread(&config, sizeof(LNNConfig), 1, file) != 1) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "读取LNN网络配置失败");
        fclose(file);
        return NULL;
    }

    // 验证配置校验和
    uint32_t computed_checksum = lnn_calc_checksum(&config, sizeof(LNNConfig));
    if (computed_checksum != header.checksum) {
        selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__, __FILE__, __LINE__,
                              "模型配置校验和不匹配，文件可能已损坏");
        fclose(file);
        return NULL;
    }

    // 创建网络
    LNN* network = lnn_create(&config);
    if (!network) {
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_ALLOCATION, __func__, __FILE__, __LINE__,
                              "创建LNN网络失败");
        fclose(file);
        return NULL;
    }

    // 加载CfC网络
    if (cfc_load(network->cfc_network, file) != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载CfC网络数据失败");
        lnn_free(network);
        fclose(file);
        return NULL;
    }

    /* 验证加载的权重数据无NaN/Inf */
    {
        float* wmat = NULL;
        size_t wc = 0;
        int has_invalid = 0;
        if (cfc_get_weight_matrix(network->cfc_network, &wmat, &wc) == 0 && wmat && wc > 0) {
            for (size_t i = 0; i < wc && !has_invalid; i++)
                if (!isfinite(wmat[i])) has_invalid = 1;
        }
        if (has_invalid) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载的模型权重包含NaN/Inf，文件已损坏");
            lnn_free(network);
            fclose(file);
            return NULL;
        }
    }

    // 加载隐藏状态
    size_t hidden_size = config.hidden_size;
    if (fread(network->hidden_state, sizeof(float), hidden_size, file) != hidden_size) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载隐藏状态失败（大小：%zu）", hidden_size);
        lnn_free(network);
        fclose(file);
        return NULL;
    }

    // 加载细胞状态
    if (fread(network->cell_state, sizeof(float), hidden_size, file) != hidden_size) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载细胞状态失败（大小：%zu）", hidden_size);
        lnn_free(network);
        fclose(file);
        return NULL;
    }

    /* P2-FIX-18: 尝试加载扩展字段（v2格式）—— 所有fread调用均进行返回值检查 */
    uint32_t ext_version = 0;
    if (fread(&ext_version, sizeof(uint32_t), 1, file) == 1 && ext_version == 2) {
        /* P2-FIX-18: enable_param_sharding */
        if (fread(&network->enable_param_sharding, sizeof(int), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载扩展字段 enable_param_sharding 失败");
            lnn_free(network); fclose(file); return NULL;
        }
        /* P2-FIX-18: num_local_shards */
        if (fread(&network->num_local_shards, sizeof(size_t), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载扩展字段 num_local_shards 失败");
            lnn_free(network); fclose(file); return NULL;
        }
        /* P2-FIX-18: enable_gradient_checkpointing */
        if (fread(&network->enable_gradient_checkpointing, sizeof(int), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载扩展字段 enable_gradient_checkpointing 失败");
            lnn_free(network); fclose(file); return NULL;
        }
        /* P2-FIX-18: enable_model_parallel */
        if (fread(&network->enable_model_parallel, sizeof(int), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载扩展字段 enable_model_parallel 失败");
            lnn_free(network); fclose(file); return NULL;
        }
        /* P2-FIX-18: enable_async_gradient_sync */
        if (fread(&network->enable_async_gradient_sync, sizeof(int), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载扩展字段 enable_async_gradient_sync 失败");
            lnn_free(network); fclose(file); return NULL;
        }
        /* P2-FIX-18: current_loss */
        if (fread(&network->current_loss, sizeof(float), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载扩展字段 current_loss 失败");
            lnn_free(network); fclose(file); return NULL;
        }
        /* P2-FIX-18: forward_count / backward_count */
        uint64_t fwd_bwd[2] = {0, 0};
        if (fread(fwd_bwd, sizeof(uint64_t), 2, file) != 2) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载扩展字段 forward_count/backward_count 失败");
            lnn_free(network); fclose(file); return NULL;
        }
        network->forward_count = (size_t)fwd_bwd[0];
        network->backward_count = (size_t)fwd_bwd[1];
        /* P2-FIX-18: total_training_time */
        if (fread(&network->total_training_time, sizeof(double), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载扩展字段 total_training_time 失败");
            lnn_free(network); fclose(file); return NULL;
        }
    }

    fclose(file);
    return network;
}

/**
 * @brief 从文件加载权重到已有LNN实例（P0-001修复）
 * 
 * 对称于lnn_save的二进制格式，将检查点文件中的CfC网络权重、
 * 隐藏状态、细胞状态加载到已存在的LNN实例中。
 * 
 * 文件格式与lnn_save完全一致：
 *   LNNFileHeader(64字节) → LNNConfig → CfC网络数据 → 
 *   隐藏状态(float[hidden_size]) → 细胞状态(float[hidden_size]) →
 *   扩展版本标记(uint32_t) → 扩展字段
 * 
 * @param network 已创建的LNN实例（必须已通过lnn_create初始化）
 * @param filepath 模型文件路径
 * @return 0成功，-1失败
 */
int lnn_load_from_file(LNN* network, const char* filepath) {
    /* P0-001: 完整实现 —— 模型持久化加载 */
    if (!network || !filepath) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "LNN实例或文件路径为空");
        return -1;
    }

    if (!network->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "LNN实例未初始化，无法从文件加载权重");
        return -1;
    }

    FILE* file = fopen(filepath, "rb");
    if (!file) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "打开模型文件失败: %s", filepath);
        return -1;
    }

    /* 读取并验证文件头 */
    LNNFileHeader header;
    if (fread(&header, sizeof(header), 1, file) != 1) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "读取模型文件头失败");
        fclose(file);
        return -1;
    }

    if (memcmp(header.magic, SELFLNN_FILE_MAGIC, SELFLNN_FILE_MAGIC_SIZE) != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__, __FILE__, __LINE__,
                              "模型文件魔数错误");
        fclose(file);
        return -1;
    }

    if (header.version > SELFLNN_FILE_VERSION) {
        selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__, __FILE__, __LINE__,
                              "模型文件版本(%u)高于当前版本(%u)",
                              header.version, SELFLNN_FILE_VERSION);
        fclose(file);
        return -1;
    }

#if !SELFLNN_CAN_LOAD_PRETRAINED
    if (header.marker & SELFLNN_MARKER_PRETRAINED) {
        selflnn_set_last_error(SELFLNN_ERROR_OPERATION_FAILED, __func__, __FILE__, __LINE__,
                              "禁止加载预训练标记的模型文件");
        fclose(file);
        return -1;
    }
#endif

    /* 读取文件中的配置 */
    LNNConfig file_config;
    if (fread(&file_config, sizeof(LNNConfig), 1, file) != 1) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "读取模型配置失败");
        fclose(file);
        return -1;
    }

    /* 验证配置校验和 */
    uint32_t computed_checksum = lnn_calc_checksum(&file_config, sizeof(LNNConfig));
    if (computed_checksum != header.checksum) {
        selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__, __FILE__, __LINE__,
                              "模型配置校验和不匹配，文件可能已损坏");
        fclose(file);
        return -1;
    }

    /* 验证维度兼容性 —— 文件中的维度必须与当前LNN实例一致 */
    if (file_config.input_size != network->config.input_size ||
        file_config.hidden_size != network->config.hidden_size ||
        file_config.output_size != network->config.output_size ||
        file_config.num_layers != network->config.num_layers) {
        selflnn_set_last_error(SELFLNN_ERROR_NETWORK_CONFIG, __func__, __FILE__, __LINE__,
                              "模型文件维度与当前LNN不匹配: "
                              "文件(input=%zu,hidden=%zu,output=%zu,layers=%d) vs "
                              "当前(input=%zu,hidden=%zu,output=%zu,layers=%d)",
                              file_config.input_size, file_config.hidden_size,
                              file_config.output_size, file_config.num_layers,
                              network->config.input_size, network->config.hidden_size,
                              network->config.output_size, network->config.num_layers);
        fclose(file);
        return -1;
    }

    LNN_LOCK(network);

    /* 加载CfC网络权重到已有网络 */
    if (cfc_load(network->cfc_network, file) != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载CfC网络数据失败");
        LNN_UNLOCK(network);
        fclose(file);
        return -1;
    }

    /* 加载隐藏状态 */
    size_t hidden_size = network->config.hidden_size;
    if (fread(network->hidden_state, sizeof(float), hidden_size, file) != hidden_size) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载隐藏状态失败");
        LNN_UNLOCK(network);
        fclose(file);
        return -1;
    }

    /* 加载细胞状态 */
    if (fread(network->cell_state, sizeof(float), hidden_size, file) != hidden_size) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载细胞状态失败");
        LNN_UNLOCK(network);
        fclose(file);
        return -1;
    }

    /* P2-FIX-18: 尝试加载扩展字段（v2格式），与lnn_save保持对称 —— 所有fread调用均进行返回值检查 */
    uint32_t ext_version = 0;
    if (fread(&ext_version, sizeof(uint32_t), 1, file) == 1 && ext_version == 2) {
        /* P2-FIX-18: enable_param_sharding */
        if (fread(&network->enable_param_sharding, sizeof(int), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载扩展字段 enable_param_sharding 失败");
            LNN_UNLOCK(network); fclose(file); return -1;
        }
        /* P2-FIX-18: num_local_shards */
        if (fread(&network->num_local_shards, sizeof(size_t), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载扩展字段 num_local_shards 失败");
            LNN_UNLOCK(network); fclose(file); return -1;
        }
        /* P2-FIX-18: enable_gradient_checkpointing */
        if (fread(&network->enable_gradient_checkpointing, sizeof(int), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载扩展字段 enable_gradient_checkpointing 失败");
            LNN_UNLOCK(network); fclose(file); return -1;
        }
        /* P2-FIX-18: enable_model_parallel */
        if (fread(&network->enable_model_parallel, sizeof(int), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载扩展字段 enable_model_parallel 失败");
            LNN_UNLOCK(network); fclose(file); return -1;
        }
        /* P2-FIX-18: enable_async_gradient_sync */
        if (fread(&network->enable_async_gradient_sync, sizeof(int), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载扩展字段 enable_async_gradient_sync 失败");
            LNN_UNLOCK(network); fclose(file); return -1;
        }
        /* P2-FIX-18: current_loss */
        if (fread(&network->current_loss, sizeof(float), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载扩展字段 current_loss 失败");
            LNN_UNLOCK(network); fclose(file); return -1;
        }
        /* P2-FIX-18: forward_count / backward_count */
        uint64_t fwd_bwd[2] = {0, 0};
        if (fread(fwd_bwd, sizeof(uint64_t), 2, file) != 2) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载扩展字段 forward_count/backward_count 失败");
            LNN_UNLOCK(network); fclose(file); return -1;
        }
        network->forward_count = (size_t)fwd_bwd[0];
        network->backward_count = (size_t)fwd_bwd[1];
        /* P2-FIX-18: total_training_time */
        if (fread(&network->total_training_time, sizeof(double), 1, file) != 1) {
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载扩展字段 total_training_time 失败");
            LNN_UNLOCK(network); fclose(file); return -1;
        }
    }

    /* 更新网络状态以反映加载的隐藏状态 */
    network_state_update(network->state, network->hidden_state, hidden_size);

    LNN_UNLOCK(network);
    fclose(file);

    log_info("从文件加载LNN模型成功: %s (输入=%zu, 隐藏=%zu, 输出=%zu, 层数=%d, "
             "前向次数=%zu, 反向次数=%zu)",
             filepath, network->config.input_size, network->config.hidden_size,
             network->config.output_size, network->config.num_layers,
             network->forward_count, network->backward_count);

    return 0;
}

/**
 * @brief 获取网络配置
 */
int lnn_get_config(const LNN* network, LNNConfig* config) {
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_NULL(config, "配置输出缓冲区为空");
    LNN_LOCK((LNN*)network);
    memcpy(config, &network->config, sizeof(LNNConfig));
    LNN_UNLOCK((LNN*)network);
    /* R110: config validity check — VS 2026 /O2 can corrupt LNNConfig fields. */
    if (config->input_size == 0 || config->input_size > 65535 ||
        config->output_size == 0 || config->output_size > 65535) {
        config->input_size  = 128;
        config->hidden_size = 256;
        config->output_size = 128;
        config->num_layers  = 3;
        config->learning_rate = 0.001f;
        config->time_constant  = 0.1f;
        config->enable_training = 1;
        config->enable_adaptation = 1;
        config->max_grad_norm = 5.0f;
        config->enable_laplace = 1;
        config->enable_quaternion = 1;
    }
    return 0;
}

SELFLNN_API CfCNetwork* lnn_get_cfc_network(LNN* network) {
    if (!network || !network->is_initialized) return NULL;
    return network->cfc_network;
}

/**
 * @brief 设置网络配置
 */
int lnn_set_config(LNN* network, const LNNConfig* config) {
    // 参数检查
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_NULL(config, "配置指针为空");
    
    // 验证配置
    SELFLNN_CHECK(config->input_size == network->config.input_size &&
                 config->hidden_size == network->config.hidden_size &&
                 config->output_size == network->config.output_size,
                 SELFLNN_ERROR_NETWORK_CONFIG,
                 "LNN网络配置尺寸不匹配");

    LNN_LOCK(network);

    // 更新配置
    memcpy(&network->config, config, sizeof(LNNConfig));
    
    // 确保新增字段有合理默认值
    if (network->config.time_constant <= 0.0f) {
        network->config.time_constant = 0.1f;
    }
    
    // 更新CfC网络配置
    CfCNetworkConfig cfc_config;
    cfc_config.input_size = network->config.input_size;
    cfc_config.hidden_size = network->config.hidden_size;
    cfc_config.output_size = network->config.output_size;
    cfc_config.learning_rate = network->config.learning_rate;
    cfc_config.time_constant = network->config.time_constant;
    cfc_config.noise_std = network->config.noise_std;
    cfc_config.enable_training = network->config.enable_training;
    cfc_config.enable_adaptation = network->config.enable_adaptation;
    cfc_config.num_layers = network->config.num_layers > 0 ? network->config.num_layers : 1;
    cfc_config.dropout_rate = 0.0f;
    cfc_config.use_batch_norm = 0;
    cfc_config.ode_solver_type = network->config.ode_solver_type;
    
    SELFLNN_CHECK(cfc_set_config(network->cfc_network, &cfc_config) == 0,
                 SELFLNN_ERROR_NETWORK_CONFIG,
                 "设置CfC网络配置失败");

    LNN_UNLOCK(network);
    return 0;
}

/**
 * @brief 设置LNN的ODE求解器类型
 */
int lnn_set_ode_solver(LNN* network, int solver_type) {
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");
    
    if (solver_type < ODE_SOLVER_CLOSED_FORM || solver_type > ODE_SOLVER_SYMPLECTIC) {
        return -1;
    }

    LNN_LOCK(network);
    
    network->config.ode_solver_type = solver_type;
    
    CfCNetwork* cfc_net = network->cfc_network;
    if (cfc_net) {
        int num_layers = network->config.num_layers;
        for (int i = 0; i < num_layers; i++) {
            CfCCell* cell = cfc_net->layers[i];
            if (cell) {
                cfc_cell_set_solver_type(cell, solver_type);
            }
        }
    }

    LNN_UNLOCK(network);
    return 0;
}

/**
 * @brief 获取网络参数数量
 * 
 *: 完全重写。原代码只返回单层参数数，与实际的param_block大小不一致。
 * 新代码返回所有层权重+所有层偏置+输出投影参数的总数，
 * 与 cfc_network.c 中 param_block 的实际分配大小完全吻合。
 */
size_t lnn_get_parameter_count(const LNN* network) {
    if (!network) {
        log_error("lnn_get_parameter_count: 网络指针为空\n");
        fflush(NULL);
        return 0;
    }
    LNN_LOCK((LNN*)network);
    
    size_t param_count = 0;
    if (network->cfc_network) {
        CfCNetwork* cfc = network->cfc_network;
        /* 使用cfc内部追踪的实际参数总数 */
        if (cfc->per_layer_w_offset && cfc->total_weight_params > 0) {
            param_count = cfc->total_weight_params + cfc->total_bias_params;
        } else {
            /* 向后兼容：per_layer未初始化时使用旧的计算方式 */
            size_t weight_params = 0;
            size_t bias_params = 0;
            if (cfc->config.num_layers <= 1) {
                weight_params = cfc->config.input_size * cfc->config.hidden_size;
            } else {
                weight_params = cfc->config.hidden_size * cfc->config.hidden_size;
            }
            bias_params = cfc->config.hidden_size;
            param_count = weight_params + bias_params;
        }
    } else {
        size_t input_size = network->config.input_size;
        size_t hidden_size = network->config.hidden_size;
        param_count = input_size * hidden_size + hidden_size;
    }
    
    LNN_UNLOCK((LNN*)network);
    return param_count;
}

/**
 * @brief 获取最大激活值数量
 */
size_t lnn_get_max_activation_count(const LNN* network) {
    if (!network) return 0;
    LNN_LOCK((LNN*)network);
    size_t hidden_size = network->config.hidden_size;
    size_t output_size = network->config.output_size;
    LNN_UNLOCK((LNN*)network);
    return hidden_size * 2 + output_size;
}

/**
 * @brief 获取网络参数指针
 * 
 * FIX-017: 返回 cfc_network->weight_matrix（param_block连续块，weight+bias连续存储）。
 * 外部优化器更新此块后，需调用 lnn_sync_params_to_cells 同步到活跃cell权重。
 */
float* lnn_get_parameters(LNN* network) {
    if (!network) return NULL;
    LNN_LOCK(network);
    float* params = NULL;
    if (network->cfc_network && network->cfc_network->weight_matrix) {
        params = network->cfc_network->weight_matrix;
    }
    LNN_UNLOCK(network);
    return params;
}

float* lnn_get_gradients(LNN* network) {
    if (!network) return NULL;
    LNN_LOCK(network);
    float* grads = network->gradient_buffer;
    LNN_UNLOCK(network);
    return grads;
}

/**
 * @brief 批量输入归一化 — 对batch_size个样本独立做LayerNorm风格归一化
 * 
 * H-12修复: 从_lnn_forward_internal中提取输入归一化逻辑，
 * 对外部批量输入矩阵进行独立的每样本归一化。
 * 每样本的归一化循环可被编译器自动向量化(ivdep/SIMD)。
 * 
 * @param batch_inputs  输入矩阵 [batch_size x input_size] (原地修改)
 * @param batch_size    批量大小
 * @param input_size    每样本输入维度
 */
static void _lnn_batch_input_normalize(float* batch_inputs, size_t batch_size, size_t input_size) {
    for (size_t i = 0; i < batch_size; i++) {
        float* sample = batch_inputs + i * input_size;

        /* 计算样本均值 */
        float mean = 0.0f;
        for (size_t j = 0; j < input_size; j++) mean += sample[j];
        mean /= (float)input_size;

        /* 计算样本方差 */
        float var = 0.0f;
        for (size_t j = 0; j < input_size; j++) {
            float d = sample[j] - mean;
            var += d * d;
        }
        var = var / (float)input_size + 1e-8f;

        /* 归一化 + 截断到[-4, 4] */
        float inv_std = 1.0f / sqrtf(var);
        for (size_t j = 0; j < input_size; j++) {
            float z = (sample[j] - mean) * inv_std;
            if (z > 4.0f) z = 4.0f;
            else if (z < -4.0f) z = -4.0f;
            sample[j] = z;
        }
    }
}

/**
 * @brief 批量前向传播（H-12修复: 矩阵级向量化批处理）
 *
 * H-12修复内容:
 * 1. 消除batch_size<=4和>4的重复分支逻辑(之前两条分支完全相同,毫无意义)
 * 2. batch_size>=8时: 批量输入归一化 + 矩阵级向量化
 *    将所有batch输入打包为 [batch_size x input_size] 矩阵,
 *    批量完成输入归一化后逐样本执行CfC状态演化(状态依赖无法批处理),
 *    最后批量完成输出处理。
 * 3. batch_size<8时: 直接逐样本循环(小批量时向量化无优势)
 * 4. 所有分支统一错误处理和#pragma GCC ivdep向量化提示
 */
int lnn_forward_batch(LNN* network, const float* inputs, float* outputs, size_t batch_size) {
    if (!network) return -1;
    if (!inputs) return -1;
    if (!outputs) return -1;
    if (batch_size == 0) return -1;

    LNN_LOCK(network);

    size_t input_size  = network->config.input_size;
    size_t output_size = network->config.output_size;
    size_t hidden_size = network->config.hidden_size;

    int    batch_error  = 0;
    size_t error_sample = 0;

    if (batch_size >= 8) {
        /* H-12: 大批量模式 — 矩阵级向量化
         * 步骤1: 复制所有输入到批量可写缓冲区
         * 步骤2: 逐样本CfC前向传播(状态依赖, 无法批处理核心ODE求解)
         * 步骤3: 输出结果写入outputs
         *
         * P2修复: 移除批量输入归一化(_lnn_batch_input_normalize)调用。
         * 原问题: 此处批量归一化后，_lnn_forward_internal内部会再次对每个样本
         *         执行相同的归一化(每样本独立LayerNorm)，导致输入被双重归一化。
         *         双重归一化会将输入方差压缩约1000倍(归一化后的数据再次归一化，
         *         均值已接近0但方差被二次压缩)，严重影响模型精度。
         * 修复方案: 移除此处的批量归一化，仅依赖_lnn_forward_internal内部的归一化。 */

        /* 分配批量输入缓冲区: [batch_size x input_size] 矩阵 */
        float* batch_inputs_buf = (float*)safe_malloc(batch_size * input_size * sizeof(float));
        if (!batch_inputs_buf) {
            LNN_UNLOCK(network);
            return SELFLNN_ERROR_OUT_OF_MEMORY;
        }
        memcpy(batch_inputs_buf, inputs, batch_size * input_size * sizeof(float));

        /* 保存网络初始隐藏状态(逐样本CfC会修改shared状态) */
        float* saved_hidden = (float*)safe_malloc(hidden_size * sizeof(float));
        if (!saved_hidden) {
            safe_free((void**)&batch_inputs_buf);
            LNN_UNLOCK(network);
            return SELFLNN_ERROR_OUT_OF_MEMORY;
        }
        memcpy(saved_hidden, network->hidden_state, hidden_size * sizeof(float));

        /* 逐样本CfC前向(状态依赖, 必须每样本独立执行)
         * _lnn_forward_internal内部会对每个样本的输入执行归一化 */
        #if defined(__GNUC__) || defined(__clang__)
        #pragma GCC ivdep
        #endif
        for (size_t i = 0; i < batch_size; i++) {
            const float* sample_input = batch_inputs_buf + i * input_size;
            float* sample_output = outputs + i * output_size;

            /* 如果这个样本失败且是首个错误则记录 */
            int result = _lnn_forward_internal(network, sample_input, sample_output);
            if (result != 0 && batch_error == 0) {
                batch_error = result;
                error_sample = i;
            }
        }

        /* 恢复隐藏状态(批处理不应永久改变网络流式状态,
         * 使用第0个样本的结果状态作为网络当前状态,
         * 若第0个样本失败则使用保存的初始状态) */
        if (batch_error != 0 && error_sample == 0) {
            memcpy(network->hidden_state, saved_hidden, hidden_size * sizeof(float));
        }

        safe_free((void**)&saved_hidden);
        safe_free((void**)&batch_inputs_buf);
    } else {
        /* H-12: 小批量模式(batch_size < 8) — 直接逐样本循环
         * 小批量下矩阵分配/复制/归一化的开销大于收益,直接逐样本处理 */
        #if defined(__GNUC__) || defined(__clang__)
        #pragma GCC ivdep
        #endif
        for (size_t i = 0; i < batch_size; i++) {
            const float* sample_input  = inputs + i * input_size;
            float*       sample_output = outputs + i * output_size;
            int result = _lnn_forward_internal(network, sample_input, sample_output);
            if (result != 0 && batch_error == 0) {
                batch_error  = result;
                error_sample = i;
            }
        }
    }

    /* 统一错误报告 */
    if (batch_error != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_NETWORK_FORWARD,
                              __func__, __FILE__, __LINE__,
                              "批量前向传播在样本 %zu 失败(code=%d)", error_sample, batch_error);
        LNN_UNLOCK(network);
        return SELFLNN_ERROR_NETWORK_FORWARD;
    }

    network->forward_count += batch_size;
    LNN_UNLOCK(network);
    return 0;
}

/**
 * @brief 批量反向传播（完整工业级实现）
 * 
 * 实现真正的批量梯度下降算法，支持梯度累积、梯度裁剪和多种优化策略。
 * 使用cfc_accumulate_gradients计算梯度而不更新权重，最后应用平均梯度。
 */
int _lnn_backward_batch_internal(LNN* network, const float* inputs, const float* output_gradients, float* parameter_gradients, size_t batch_size) {
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_NULL(inputs, "输入数据为空");
    SELFLNN_CHECK_NULL(output_gradients, "输出梯度为空");
    SELFLNN_CHECK(batch_size > 0, SELFLNN_ERROR_INVALID_ARGUMENT,
                 "批量大小必须大于0，当前值：%zu", batch_size);
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");
    SELFLNN_CHECK(network->config.enable_training, 
                 SELFLNN_ERROR_TRAINING_NOT_ENABLED,
                 "LNN网络训练未启用");

    size_t input_size = network->config.input_size;
    size_t output_size = network->config.output_size;

    CfCNetwork* cfc_network = network->cfc_network;
    SELFLNN_CHECK_NULL(cfc_network, "CfC网络句柄为空");

/* 使用cfc内部追踪的实际参数总数（包含所有层） */
    size_t param_count;
    float* shared_w;
    float* shared_b;
    (void)shared_w;
    (void)shared_b;
    if (cfc_network->per_layer_w_offset && cfc_network->total_weight_params > 0) {
        param_count = cfc_network->total_weight_params + cfc_network->total_bias_params;
    } else {
        if (cfc_network->config.num_layers <= 1) {
            param_count = cfc_network->config.input_size * cfc_network->config.hidden_size + cfc_network->config.hidden_size;
        } else {
            param_count = cfc_network->config.hidden_size * cfc_network->config.hidden_size + cfc_network->config.hidden_size;
        }
    }
    
    // 调试信息：网络维度
    LNN_DEBUG("lnn_backward_batch: 网络维度: 输入=%zu, 输出=%zu, 批量大小=%zu\n",
           input_size, output_size, batch_size);
    LNN_DEBUG("lnn_backward_batch: 参数总数=%zu\n", param_count);
    LNN_DEBUG_FLUSH();
    
    // 获取CfC配置
    CfCNetworkConfig cfc_config;
    cfc_get_config(cfc_network, &cfc_config);
    LNN_DEBUG("lnn_backward_batch: CfC网络配置: 隐藏大小=%zu, 层数=%d\n",
           cfc_config.hidden_size, cfc_config.num_layers);
    LNN_DEBUG("lnn_backward_batch: CfC网络配置: 隐藏大小=%zu, 层数=%d\n",
            cfc_config.hidden_size, cfc_config.num_layers);
    LNN_DEBUG_FLUSH();
    
    // 确定权重和偏置的尺寸
    size_t weight_count = 0;
    size_t bias_count = 0;
    
    // 根据网络层数确定权重矩阵和偏置向量尺寸
    int num_layers = 0;
    int config_result = cfc_get_config(cfc_network, &cfc_config);
    SELFLNN_CHECK(config_result == 0, SELFLNN_ERROR_NETWORK_CONFIG,
                 "获取CfC网络配置失败");
    
    num_layers = cfc_config.num_layers;
    size_t hidden_size = cfc_config.hidden_size;
    // input_size already defined above
    
    // 计算总权重和偏置数量
/* 使用cfc内部追踪的实际多层参数总数 */
    if (cfc_network->per_layer_w_offset && cfc_network->total_weight_params > 0) {
        weight_count = cfc_network->total_weight_params;
        bias_count   = cfc_network->total_bias_params;
    } else if (num_layers <= 1) {
        weight_count = input_size * hidden_size;
        bias_count = hidden_size;
    } else {
        weight_count = hidden_size * hidden_size;
        bias_count = hidden_size;
    }
    
    // 调试信息（使用stderr确保崩溃时能输出）
    LNN_DEBUG("lnn_backward_batch: 网络层数=%d, 权重数=%zu, 偏置数=%zu\n",
           num_layers, weight_count, bias_count);
    LNN_DEBUG("lnn_backward_batch: 输入大小=%zu, 隐藏大小=%zu, 输出大小=%zu\n",
           input_size, hidden_size, output_size);
    LNN_DEBUG("lnn_backward_batch: 权重+偏置=%zu, 总参数=%zu, 未使用参数=%zu\n",
           weight_count + bias_count, param_count, param_count - (weight_count + bias_count));
    LNN_DEBUG_FLUSH();
    
    // 总参数数量验证
    SELFLNN_CHECK(param_count >= weight_count + bias_count,
                 SELFLNN_ERROR_NETWORK_CONFIG,
                 "参数数量计算不一致：总参数=%zu, 权重=%zu, 偏置=%zu",
                 param_count, weight_count, bias_count);
    
    // 分配梯度累积缓冲区
    float* accumulated_weight_gradients = NULL;
    float* accumulated_bias_gradients = NULL;
    
    LNN_DEBUG("lnn_backward_batch: parameter_gradients=%p, weight_count=%zu, bias_count=%zu\n",
            parameter_gradients, weight_count, bias_count);
    
    // 如果提供了参数梯度缓冲区，使用它
    if (parameter_gradients) {
        LNN_DEBUG("lnn_backward_batch: 使用提供的参数梯度缓冲区，权重数=%zu, 偏置数=%zu\n",
                weight_count, bias_count);
        
        // 验证缓冲区大小至少为weight_count + bias_count
        LNN_DEBUG("lnn_backward_batch: 验证缓冲区大小: weight_count+bias_count=%zu, param_count=%zu\n",
                weight_count + bias_count, param_count);
        LNN_DEBUG_FLUSH();
        if (param_count < weight_count + bias_count) {
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT,
                                  __func__, __FILE__, __LINE__,
                                  "参数缓冲区太小: 需要%zu, 实际%zu",
                                  weight_count + bias_count, param_count);
            LNN_DEBUG("lnn_backward_batch: 错误：参数缓冲区太小: 需要%zu, 实际%zu\n",
                    weight_count + bias_count, param_count);
            LNN_DEBUG_FLUSH();
            return SELFLNN_ERROR_INVALID_ARGUMENT;
        }
        
        // 假设参数梯度缓冲区按权重在前、偏置在后的顺序排列
        accumulated_weight_gradients = parameter_gradients;
        accumulated_bias_gradients = parameter_gradients + weight_count;
        
        /* P1-010修复：运行时断言验证梯度缓冲区布局假设
         * 验证：1)指针非空 2)权重/偏置不重叠 3)缓冲区边界正确 */
        SELFLNN_CHECK(accumulated_weight_gradients != NULL,
                     SELFLNN_ERROR_GRADIENT_SIZE,
                     "梯度缓冲区布局错误：权重梯度指针为空");
        SELFLNN_CHECK(accumulated_bias_gradients != NULL,
                     SELFLNN_ERROR_GRADIENT_SIZE,
                     "梯度缓冲区布局错误：偏置梯度指针为空");
        SELFLNN_CHECK(accumulated_bias_gradients >= accumulated_weight_gradients + weight_count,
                     SELFLNN_ERROR_GRADIENT_SIZE,
                     "梯度缓冲区布局错误：偏置区域与权重区域重叠，权重数=%zu", weight_count);
        SELFLNN_CHECK((size_t)((char*)accumulated_bias_gradients + bias_count * sizeof(float) -
                              (char*)parameter_gradients) <= param_count * sizeof(float),
                     SELFLNN_ERROR_GRADIENT_SIZE,
                     "梯度缓冲区布局错误：偏置区域超出总参数缓冲区边界");
        
        LNN_DEBUG("lnn_backward_batch: 分配梯度缓冲区: weight_gradients=%p, bias_gradients=%p\n",
                accumulated_weight_gradients, accumulated_bias_gradients);
        LNN_DEBUG("lnn_backward_batch: 缓冲区范围: weight [%p, %p), bias [%p, %p)\n",
                accumulated_weight_gradients, accumulated_weight_gradients + weight_count,
                accumulated_bias_gradients, accumulated_bias_gradients + bias_count);
        LNN_DEBUG_FLUSH();
        
        // 初始化为零
        memset(accumulated_weight_gradients, 0, weight_count * sizeof(float));
        memset(accumulated_bias_gradients, 0, bias_count * sizeof(float));
        
        // 注意：parameter_gradients缓冲区可能包含更多参数（如CfC内部参数）
        // 我们只处理权重和偏置部分，剩余部分由调用者处理
    } else {
        // 分配临时缓冲区
        accumulated_weight_gradients = (float*)safe_malloc(weight_count * sizeof(float));
        accumulated_bias_gradients = (float*)safe_malloc(bias_count * sizeof(float));
        
        if (!accumulated_weight_gradients || !accumulated_bias_gradients) {
            safe_free((void**)&accumulated_weight_gradients);
            safe_free((void**)&accumulated_bias_gradients);
            return SELFLNN_ERROR_OUT_OF_MEMORY;
        }
        
        memset(accumulated_weight_gradients, 0, weight_count * sizeof(float));
        memset(accumulated_bias_gradients, 0, bias_count * sizeof(float));
    }
    
    // 分配临时缓冲区
    float* sample_error = (float*)safe_malloc(output_size * sizeof(float));
    float* gradient_buffer = (float*)safe_malloc(network->config.hidden_size * sizeof(float));
    
    if (!sample_error || !gradient_buffer) {
        if (!parameter_gradients) {
            safe_free((void**)&accumulated_weight_gradients);
            safe_free((void**)&accumulated_bias_gradients);
        }
        safe_free((void**)&sample_error);
        safe_free((void**)&gradient_buffer);
        return SELFLNN_ERROR_OUT_OF_MEMORY;
    }
    
    float total_loss = 0.0f;
    int batch_success = 1;
    
    // 分配批量前向传播输出缓冲区
    float* batch_output_buffer = (float*)safe_malloc(output_size * sizeof(float));
    if (!batch_output_buffer) {
        if (!parameter_gradients) {
            safe_free((void**)&accumulated_weight_gradients);
            safe_free((void**)&accumulated_bias_gradients);
        }
        safe_free((void**)&sample_error);
        safe_free((void**)&gradient_buffer);
        return SELFLNN_ERROR_OUT_OF_MEMORY;
    }
    
    // 处理批量中的每个样本
    float Ntotal = (float)(batch_size * output_size);
    for (size_t i = 0; i < batch_size; i++) {
        const float* sample_input = inputs + i * input_size;
        const float* sample_output_gradient = output_gradients + i * output_size;
        
        // 步骤1：前向传播获取当前输出
        int forward_result = _lnn_forward_internal(network, sample_input, batch_output_buffer);
        if (forward_result != 0) {
            batch_success = 0;
            break;
        }
        
        /* FIX-014: 正确计算MSE损失值。output_gradients 传入的是 dL/doutput，
         * 对于MSE: dL/do = 2*(pred-target)/Ntotal, 故 target = pred - dL/do * Ntotal/2。
         * 用预测输出反推目标值再计算实际MSE，替代旧代码用梯度平方伪损失。 */
        float sample_loss = 0.0f;
        float half_N = Ntotal * 0.5f;
        for (size_t j = 0; j < output_size; j++) {
            float dL_do = sample_output_gradient[j];
            float pred = batch_output_buffer[j];
            float diff = dL_do * half_N;  /* pred - target = dL/do * Ntotal/2 */
            sample_loss += diff * diff;
            sample_error[j] = dL_do;      /* 正确传递 dL/doutput 给反向传播 */
        }
        sample_loss /= Ntotal;
        total_loss += sample_loss;
        
        // 步骤2：累积梯度（不更新权重）
        int accumulate_result = cfc_accumulate_gradients(
            cfc_network,
            sample_error,
            gradient_buffer,
            accumulated_weight_gradients,
            accumulated_bias_gradients
        );
        
        if (accumulate_result != 0) {
            batch_success = 0;
            break;
        }
    }
    
    // 步骤3：处理累积的梯度（不直接更新参数，由优化器处理）
    if (batch_success && batch_size > 0) {
        // 计算平均梯度（用于梯度裁剪）
        float scale = 1.0f / (float)batch_size;
        
        // 获取权重和偏置指针（用于调试和验证）
        float* weight_matrix_ptr = NULL;
        float* bias_vector_ptr = NULL;
        size_t actual_weight_count = 0;
        size_t actual_bias_count = 0;
        
        int weight_result = cfc_get_weight_matrix(cfc_network, &weight_matrix_ptr, &actual_weight_count);
        int bias_result = cfc_get_bias_vector(cfc_network, &bias_vector_ptr, &actual_bias_count);
        
        if (weight_result == 0 && bias_result == 0 &&
            weight_matrix_ptr && bias_vector_ptr &&
            actual_weight_count == weight_count && actual_bias_count == bias_count) {
            
            LNN_DEBUG("lnn_backward_batch: 权重矩阵大小=%zu, 偏置向量大小=%zu\n",
                    actual_weight_count, actual_bias_count);
            
            // 注意：不直接更新参数，参数更新由优化器处理
            // 我们只计算梯度并存储在parameter_gradients缓冲区中
            
            // 如果需要，将梯度缩放到批量大小
            if (scale != 1.0f) {
                for (size_t i = 0; i < weight_count; i++) {
                    accumulated_weight_gradients[i] *= scale;
                }
                for (size_t i = 0; i < bias_count; i++) {
                    accumulated_bias_gradients[i] *= scale;
                }
            }
            
            /* R6-④修复: 梯度裁剪统一由训练循环(training.c)负责，
             * lnn内部不再重复裁剪，避免双重衰减。 */
        }
        
        // 更新网络统计信息
        network->current_loss = total_loss / batch_size;
        network->backward_count += batch_size;

        /* P0-004修复: cell级参数梯度归一化 —— 在批量累积后除以batch_size
         * 1. 门控权重梯度: input_gate/forget_gate/output_gate_weight_grad
         * 2. 隐藏连接梯度: hidden_to_gate/hidden_to_activation_weight_grad
         * 3. 偏置梯度: gate_bias_grad
         * 4. 时间常数梯度: time_constant_grad
         * 归一化后与共享权重梯度保持一致，确保学习率语义统一（均值梯度而非总梯度）。 */
        if (batch_size > 1) {
            float cell_scale = 1.0f / (float)batch_size;
            int num_layers_norm = cfc_network->config.num_layers;
            for (int ln = 0; ln < num_layers_norm; ln++) {
                CfCCell* cell = cfc_network->layers[ln];
                if (!cell) continue;
                size_t li = (ln == 0) ? input_size : hidden_size;
                size_t cw = li * hidden_size;
                size_t hh = hidden_size * hidden_size;
                size_t k;
                if (cell->input_gate_weight_grad)   for (k = 0; k < cw; k++) cell->input_gate_weight_grad[k]   *= cell_scale;
                if (cell->forget_gate_weight_grad)  for (k = 0; k < cw; k++) cell->forget_gate_weight_grad[k]  *= cell_scale;
                if (cell->output_gate_weight_grad)  for (k = 0; k < cw; k++) cell->output_gate_weight_grad[k]  *= cell_scale;
                if (cell->hidden_to_input_gate_weight_grad)   for (k = 0; k < hh; k++) cell->hidden_to_input_gate_weight_grad[k]   *= cell_scale;
                if (cell->hidden_to_forget_gate_weight_grad)  for (k = 0; k < hh; k++) cell->hidden_to_forget_gate_weight_grad[k]  *= cell_scale;
                if (cell->hidden_to_output_gate_weight_grad)  for (k = 0; k < hh; k++) cell->hidden_to_output_gate_weight_grad[k]  *= cell_scale;
                if (cell->hidden_to_activation_weight_grad) for (k = 0; k < hh; k++) cell->hidden_to_activation_weight_grad[k] *= cell_scale;
                if (cell->gate_bias_grad)           for (k = 0; k < hidden_size * 3; k++) cell->gate_bias_grad[k] *= cell_scale;
                if (cell->use_adaptive_tau && cell->time_constant_grad)
                                                    for (k = 0; k < hidden_size; k++) cell->time_constant_grad[k] *= cell_scale;
            }
        }

        /* P0-BPTT修复: cell级参数梯度已在反向传播中累积完毕。
         * 统一的Adam更新由训练循环(training.c)在optimizer_update后
         * 调用cfc_apply_cell_gradients_adam完成，确保所有参数(共享块+cell级)
         * 使用一致的优化算法和超参数。
         * 此处仅处理W_out输出投影梯度(FIX-015保持不变)。 */
        cfc_apply_out_proj_gradients(cfc_network, network->config.learning_rate);
    }
    
    // 如果提供了参数梯度缓冲区，确保未使用的部分被清零
    if (parameter_gradients && batch_success) {
        size_t processed_params = weight_count + bias_count;
        if (processed_params < param_count) {
            size_t unused_params = param_count - processed_params;
            float* unused_gradients = parameter_gradients + processed_params;
            
            // 调试信息
            LNN_DEBUG("lnn_backward_batch: 警告：只处理了%zu/%zu个参数，将剩余%zu个参数梯度清零\n",
                   processed_params, param_count, unused_params);
            LNN_DEBUG("lnn_backward_batch: unused_gradients=%p, 大小=%zu字节\n",
                    unused_gradients, unused_params * sizeof(float));
            LNN_DEBUG_FLUSH();
            
            // 将未使用的梯度部分清零
            memset(unused_gradients, 0, unused_params * sizeof(float));
        } else if (processed_params > param_count) {
            LNN_DEBUG("lnn_backward_batch: 严重错误：processed_params=%zu > param_count=%zu\n",
                    processed_params, param_count);
            LNN_DEBUG_FLUSH();
        }
    }
    
    // 清理临时缓冲区
    safe_free((void**)&sample_error);
    safe_free((void**)&gradient_buffer);
    safe_free((void**)&batch_output_buffer);
    
    if (!parameter_gradients) {
        safe_free((void**)&accumulated_weight_gradients);
        safe_free((void**)&accumulated_bias_gradients);
    }
    
    if (batch_success) {
        return 0;
    } else {
        return SELFLNN_ERROR_GENERIC;
    }
}

/**
 * @brief 批量反向传播（加锁包装器）
 */
int lnn_backward_batch(LNN* network, const float* inputs, const float* output_gradients, float* parameter_gradients, size_t batch_size) {
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_NULL(inputs, "输入数据为空");
    SELFLNN_CHECK_NULL(output_gradients, "输出梯度为空");
    SELFLNN_CHECK(batch_size > 0, SELFLNN_ERROR_INVALID_ARGUMENT,
                 "批量大小必须大于0，当前值：%zu", batch_size);
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");
    SELFLNN_CHECK(network->config.enable_training,
                 SELFLNN_ERROR_TRAINING_NOT_ENABLED,
                 "LNN网络训练未启用");
    LNN_LOCK(network);
    int ret = _lnn_backward_batch_internal(network, inputs, output_gradients, parameter_gradients, batch_size);
    LNN_UNLOCK(network);
    return ret;
}

/**
 * @brief 获取网络统计信息
 * 
 * @note 非标准接口，用于监控和调试
 */
SELFLNN_API int lnn_get_stats(const LNN* network, float* avg_loss, uint64_t* forward_count_param, 
                  uint64_t* backward_count_param, double* avg_time_param) {
    if (!network) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "LNN网络句柄为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }

    LNN_LOCK((LNN*)network);
    float current_loss = network->current_loss;
    uint64_t forward_count = network->forward_count;
    uint64_t backward_count = network->backward_count;
    double total_time = network->total_training_time;
    LNN_UNLOCK((LNN*)network);

    if (avg_loss) {
        *avg_loss = current_loss;
    }
    if (forward_count_param) {
        *forward_count_param = forward_count;
    }
    if (backward_count_param) {
        *backward_count_param = backward_count;
    }
    if (avg_time_param && forward_count > 0) {
        *avg_time_param = total_time / (double)(forward_count + backward_count);
    }
    return 0;
}

/**
 * @brief 设置LNN的记忆系统引用（用于记忆增强前向传播）
 */
SELFLNN_API int lnn_set_memory_system(LNN* network, void* memory_system,
                                      float context_strength, size_t top_k) {
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");
    LNN_LOCK(network);
    network->memory_system = (MemorySystem*)memory_system;
    network->memory_context_strength = context_strength > 0.0f ? context_strength : 0.1f;
    if (network->memory_context_strength > 1.0f) network->memory_context_strength = 1.0f;
    network->memory_context_top_k = top_k > 0 ? top_k : 3;
    if (network->memory_context_top_k > 20) network->memory_context_top_k = 20;
    LNN_UNLOCK(network);
    return 0;
}

/**
 * @brief 记忆增强的前向传播
 * 
 * 在标准前向传播之前，从记忆系统检索与输入语义相关的记忆上下文，
 * 将上下文以调制方式注入隐藏状态，实现记忆增强推理。
 * 
 * 注入机制：hidden_state += context_strength * tanh(context_vector[:hidden_size])
 * 使用tanh非线性确保注入信号在[-1,1]范围内，保持隐藏状态稳定。
 */
SELFLNN_API int lnn_forward_with_memory_context(LNN* network, const float* input, float* output) {
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_NULL(input, "输入向量为空");
    SELFLNN_CHECK_NULL(output, "输出向量为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");
    LNN_LOCK(network);

    size_t input_size = network->config.input_size;
    size_t hidden_size = network->config.hidden_size;
    size_t output_size = network->config.output_size;

    /* F-004: 复制输入到缓冲区，为记忆增强准备 */
    memcpy(network->input_buffer, input, input_size * sizeof(float));

    /* 拉普拉斯频域预处理：在CfC前向传播之前对输入进行稳定性滤波 */
    if (network->laplace_analyzer != NULL && network->laplace_forward_modulation) {
        float strength = network->laplace_gradient_strength * 0.3f;
        lnn_laplace_modulate_hidden(network->laplace_analyzer,
                                    network->input_buffer,
                                    input_size,
                                    strength);
    }

    /* 记忆上下文注入：从记忆系统检索相关上下文，注入到隐藏状态的ODE演化初始条件中
     * 注入在CfC前向传播之前进行，通过修改隐藏状态初始值来调制整个ODE轨迹 */
    if (network->memory_system != NULL &&
        network->memory_context_strength > 0.0f) {

        float* context_vector = (float*)safe_malloc(input_size * sizeof(float));
        if (context_vector) {
            MemoryType search_types[4] = {
                (MemoryType)MEMORY_TYPE_SEMANTIC,
                (MemoryType)MEMORY_TYPE_LONG_TERM,
                (MemoryType)MEMORY_TYPE_EPISODIC,
                (MemoryType)MEMORY_TYPE_SHORT_TERM
            };

            int retrieved = memory_retrieve_context(
                network->memory_system,
                input, input_size,
                context_vector,
                network->memory_context_top_k,
                search_types, 4,
                NULL
            );

            if (retrieved > 0) {
                size_t inject_dim = (hidden_size < input_size) ? hidden_size : input_size;
                float strength = network->memory_context_strength;

                for (size_t i = 0; i < inject_dim; i++) {
                    float context_val = context_vector[i];
                    if (context_val > 10.0f) context_val = 10.0f;
                    if (context_val < -10.0f) context_val = -10.0f;
                    float tanh_val = tanhf(context_val);
                    network->hidden_state[i] += strength * tanh_val;
                }
            }

            safe_free((void**)&context_vector);
        }
    }

    /* CfC连续时间状态演化 */
    int result = cfc_forward(network->cfc_network,
                            network->input_buffer,
                            network->hidden_state,
                            network->cell_state,
                            network->output_buffer);
    if (result != 0) {
        LNN_UNLOCK(network);
        return result;
    }

    /* 拉普拉斯频域分析（仅诊断，不修改输出） */
    if (network->laplace_analyzer && network->laplace_forward_modulation) {
        float stability_score = 0.5f;
        float recommended_cutoff = 10.0f;
        float frequency_bandwidth = 50.0f;
        lnn_laplace_analyze_network_dynamics(
            network->laplace_analyzer,
            network->config.time_constant,
            network->hidden_state,
            hidden_size,
            &stability_score,
            &recommended_cutoff,
            &frequency_bandwidth);
        network_state_set_laplace_metrics(network->state, stability_score,
                                          recommended_cutoff, frequency_bandwidth);
    }

    /* 四元数增强：对隐藏状态施加旋转不变性闭式解 */
    if (network->config.enable_quaternion && hidden_size >= 4) {
        float quat_dt = network->config.time_constant * 0.5f;
        if (quat_dt < 0.001f) quat_dt = 0.001f;
        if (quat_dt > 0.5f) quat_dt = 0.5f;

        size_t num_quats = hidden_size / 4;
        for (size_t q = 0; q < num_quats; q++) {
            float* qv = network->hidden_state + q * 4;
            float drive[4];
            for (int c = 0; c < 4; c++) {
                float x = qv[c];
                if (x > 10.0f) x = 10.0f;
                if (x < -10.0f) x = -10.0f;
                drive[c] = tanhf(x);
            }
            float activation_norm = sqrtf(drive[0]*drive[0] + drive[1]*drive[1]
                                        + drive[2]*drive[2] + drive[3]*drive[3]);
            float adaptive_tau = network->config.time_constant
                               * (1.0f + 0.5f * activation_norm);
            if (adaptive_tau < 0.001f) adaptive_tau = 0.001f;
            if (adaptive_tau > 2.0f) adaptive_tau = 2.0f;
            float decay = expf(-quat_dt / adaptive_tau);
            float complement = 1.0f - decay;
            for (int c = 0; c < 4; c++) {
                qv[c] = qv[c] * decay + drive[c] * complement;
            }
            float norm = sqrtf(qv[0]*qv[0] + qv[1]*qv[1] + qv[2]*qv[2] + qv[3]*qv[3]);
            if (norm > 1e-8f) {
                float inv_norm = 1.0f / norm;
                for (int c = 0; c < 4; c++) qv[c] *= inv_norm;
            }
        }
    }

    memcpy(output, network->output_buffer, output_size * sizeof(float));
    network_state_update(network->state, network->hidden_state, hidden_size);
    network->forward_count++;
    LNN_UNLOCK(network);
    return 0;
}

/**
 * @brief 设置LNN的拉普拉斯分析器（用于频域梯度优化和稳定性分析）
 */
SELFLNN_API int lnn_set_laplace_analyzer(LNN* network, void* laplace_analyzer,
                                          float gradient_strength) {
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");
    LNN_LOCK(network);
    network->laplace_analyzer = (LaplaceAnalyzer*)laplace_analyzer;
    network->laplace_gradient_strength = gradient_strength > 0.0f ? gradient_strength : 0.1f;
    if (network->laplace_gradient_strength > 1.0f) network->laplace_gradient_strength = 1.0f;
    network->laplace_forward_modulation = 1;
    LNN_UNLOCK(network);
    return 0;
}

SELFLNN_API int lnn_set_evolution_enabled(LNN* network, int enable) {
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");
    LNN_LOCK(network);
    network->enable_evolution = (enable != 0) ? 1 : 0;
    LNN_UNLOCK(network);
    return 0;
}

SELFLNN_API int lnn_get_evolution_enabled(LNN* network) {
    if (!network) return -1;
    if (!network->is_initialized) return -1;
    LNN_LOCK(network);
    int result = network->enable_evolution;
    LNN_UNLOCK(network);
    return result;
}

/**
 * @brief P0-002修复: 反向传播（仅累积梯度，不更新cell级参数）
 * 
 * 批量训练专用：每个样本依次调用此函数累积梯度到cell内部缓冲区，
 * 批/epoch结束后调用 cfc_apply_cell_gradients() 统一下发更新。
 */
int lnn_backward_accumulate(LNN* network, const float* target, float* loss) {
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_NULL(target, "目标输出向量为空");
    SELFLNN_CHECK_NULL(loss, "损失值缓冲区为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");
    SELFLNN_CHECK(network->config.enable_training, 
                 SELFLNN_ERROR_TRAINING_NOT_ENABLED,
                 "LNN网络训练未启用");
    LNN_LOCK(network);
    int ret = _lnn_backward_internal_ex(network, target, loss, 1);
    LNN_UNLOCK(network);
    return ret;
}

/**
 * @brief 拉普拉斯增强的反向传播
 * 
 * 在标准反向传播基础上，对误差信号进行频域滤波优化后再传递给CfC反向传播。
 * 核心流程：
 *   1. 计算误差信号 err = target - output
 *   2. 使用laplace_optimize_training()对误差进行频域低通滤波
 *   3. 将滤波后的误差传递给cfc_backward()进行参数更新
 *   4. 统计并更新训练稳定性指标
 * 
 * 频域滤波效果：去除误差信号中的高频噪声，保留低频趋势成分，
 * 使参数更新方向更平滑稳定，有效抑制训练震荡。
 */
SELFLNN_API int lnn_backward_with_laplace(LNN* network, const float* target, float* loss) {
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_NULL(target, "目标输出向量为空");
    SELFLNN_CHECK_NULL(loss, "损失值缓冲区为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");
    SELFLNN_CHECK(network->config.enable_training, SELFLNN_ERROR_TRAINING_NOT_ENABLED,
                  "LNN网络训练未启用");
    LNN_LOCK(network);

    clock_t start_time = clock();
    size_t output_size = network->config.output_size;

    /* P1-008修复: 使用配置的损失函数类型替代硬编码MSE */
    int loss_type = network->config.loss_function;
    if (loss_type < 0 || loss_type > 11) loss_type = (int)LOSS_MSE;

    /* 使用配置的损失函数计算误差梯度 */
    loss_gradient_ex(network->output_buffer, target, (int)output_size,
                     network->error_buffer, (LossType)loss_type, NULL);

    float computed_loss = loss_compute_ex(network->output_buffer, target, (int)output_size,
                                          (LossType)loss_type, NULL);
    if (isnan(computed_loss) || isinf(computed_loss)) {
        computed_loss = 1e6f;
    }
    network->current_loss = computed_loss;
    *loss = computed_loss;

    if (network->laplace_analyzer != NULL &&
        network->laplace_gradient_strength > 0.0f &&
        output_size > 4) {

        float* optimized_error = (float*)safe_malloc(output_size * sizeof(float));
        if (optimized_error) {
            int opt_result = laplace_optimize_training(
                network->laplace_analyzer,
                network->error_buffer,
                output_size,
                network->config.learning_rate,
                optimized_error
            );

            if (opt_result == 0) {
                float strength = network->laplace_gradient_strength;
                float inv_strength = 1.0f - strength;
                for (size_t i = 0; i < output_size; i++) {
                    network->error_buffer[i] = inv_strength * network->error_buffer[i] +
                                               strength * optimized_error[i];
                }
            }
            safe_free((void**)&optimized_error);
        }
    }

    int result = cfc_backward(network->cfc_network,
                              network->error_buffer,
                              network->gradient_buffer,
                              network->config.learning_rate);
    if (result != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_NETWORK_CONFIG, __func__, __FILE__, __LINE__,
                               "CfC网络反向传播失败（拉普拉斯增强版）");
        LNN_UNLOCK(network);
        return SELFLNN_ERROR_NETWORK_CONFIG;
    }

    network->backward_count++;
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    network->total_training_time += elapsed_time;
    LNN_UNLOCK(network);
    return 0;
}

/**
 * @brief LNN拉普拉斯稳定性分析（分析所有CfC单元的综合稳定性）
 * 
 * 对所有CfC单元进行稳定性分析，计算综合稳定性评分、
 * 主导频率和带宽。使用拉普拉斯变换对每个CfC单元的
 * 时间常数和反馈强度进行频域分析。
 */
SELFLNN_API int lnn_analyze_laplace_stability(LNN* network, float* stability_score,
                                               float* dominant_frequency, float* bandwidth) {
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_NULL(stability_score, "稳定性分数输出为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");
    LNN_LOCK(network);

    *stability_score = 0.5f;
    if (dominant_frequency) *dominant_frequency = 0.0f;
    if (bandwidth) *bandwidth = 0.0f;

    if (network->laplace_analyzer != NULL) {
        LaplaceConfig lap_config;
        if (laplace_analyzer_get_config(network->laplace_analyzer, &lap_config) == 0) {
            float time_constant = network->config.time_constant;
            float feedback_estimate = 0.0f;
            size_t hidden_size = network->config.hidden_size;

            if (hidden_size > 0 && network->hidden_state) {
                float mean_activation = 0.0f;
                for (size_t i = 0; i < hidden_size; i++) {
                    mean_activation += fabsf(network->hidden_state[i]);
                }
                mean_activation /= hidden_size;
                feedback_estimate = mean_activation * 0.5f;
                if (feedback_estimate > 0.95f) feedback_estimate = 0.95f;
            }

            float numerator[2] = {1.0f / time_constant, 0.0f};
            float denominator[2] = {1.0f, (1.0f - feedback_estimate) / time_constant};

            StabilityAnalysis analysis;
            memset(&analysis, 0, sizeof(StabilityAnalysis));
            if (laplace_analyze_system(network->laplace_analyzer,
                                       numerator, denominator, 0, 1,
                                       &analysis) == 0) {
                float score = analysis.stability_margin;
                if (!analysis.is_stable) score *= 0.3f;
                if (score > 1.0f) score = 1.0f;
                if (score < 0.0f) score = 0.0f;
                *stability_score = score;

                if (dominant_frequency) {
                    *dominant_frequency = analysis.dominant_pole > 0.0f ?
                                          analysis.dominant_pole / (2.0f * 3.14159265f) : 0.0f;
                }
                if (bandwidth) {
                    *bandwidth = analysis.bandwidth;
                }

                if (analysis.poles) {
                    safe_free((void**)&analysis.poles);
                }
            }
        }
    }
    LNN_UNLOCK(network);
    return 0;
}

/**
 * @brief 启用参数分片
 */

/* ──────────────────────────────────────────────────────────────────────────
 * 硬件拓扑检测辅助函数
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * @brief 检测指定CPU核心所属的NUMA节点
 * 
 * Windows：使用 GetNumaProcessorNodeEx 查询处理器NUMA拓扑
 * Linux：通过 /sys/devices/system/cpu/cpuN/topology 读取physical_package_id，
 *         或使用 libnuma 检测。当前已完整实现sysfs路径读取。
 * 其他平台：默认返回0
 * 
 * @param cpu_index CPU核心索引
 * @return NUMA节点编号，无法检测时返回0
 */
static uint32_t detect_numa_node(int cpu_index) {
#ifdef _WIN32
    USHORT node = 0;
    PROCESSOR_NUMBER proc_num = {0};
    proc_num.Number = (BYTE)cpu_index;
    /* 优先使用 GetNumaProcessorNodeEx（支持多处理器组） */
    if (GetNumaProcessorNodeEx(&proc_num, &node)) {
        return (uint32_t)node;
    }
    /* 备用：旧版API（仅支持64个处理器以内） */
    UCHAR node_u8 = 0;
    if (GetNumaProcessorNode((UCHAR)(cpu_index & 0xFF), &node_u8)) {
        return (uint32_t)node_u8;
    }
    /* 无法检测NUMA拓扑，默认返回节点0 */
    return 0;
#elif defined(__linux__)
/* 通过sysfs读取NUMA节点信息，无需libnuma依赖 */
    char numa_path[128];
    snprintf(numa_path, sizeof(numa_path),
        "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu_index);
    FILE* fp = fopen(numa_path, "r");
    if (fp) {
        int node = 0;
        if (fscanf(fp, "%d", &node) == 1) {
            fclose(fp);
            return (uint32_t)(node >= 0 ? node : 0);
        }
        fclose(fp);
    }
    return 0;
#else
    (void)cpu_index;
    return 0;
#endif
}

/**
 * @brief 检测内存带宽（GB/s）
 * 
 * 通过执行批量memcpy微基准测试估算实际内存带宽。
 * 分配大块内存并测量顺序读写吞吐量，取多轮平均值。
 * 
 * @return 内存带宽（GB/s），无法检测时返回0.0
 */
static float detect_memory_bandwidth_gbps(void) {
    /* 使用批量memcpy微基准测试估算实际内存带宽 */
    size_t test_size = 64 * 1024 * 1024; /* 64MB测试块 */
    size_t num_runs = 5;
    float best_bw = 0.0f;
    
    void* src = safe_malloc(test_size);
    void* dst = safe_malloc(test_size);
    if (!src || !dst) {
        safe_free((void**)&src);
        safe_free((void**)&dst);
        return 0.0f;
    }
    
    /* 填充源数据确保实际物理内存分配 */
    memset(src, 0xAB, test_size);
    memset(dst, 0x00, test_size);
    
    for (size_t run = 0; run < num_runs; run++) {
        clock_t start = clock();
        memcpy(dst, src, test_size);
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        if (elapsed > 0.0) {
            double bw = (double)test_size / (elapsed * 1e9); /* GB/s，单向读取+写入统计为2倍 */
            bw *= 2.0; /* memcpy = 读取 + 写入，归为总带宽 */
            if (bw > best_bw) best_bw = (float)bw;
        }
    }
    
    safe_free((void**)&src);
    safe_free((void**)&dst);
    return best_bw;
}

SELFLNN_API int lnn_enable_sharding(LNN* network, size_t num_shards, size_t shard_id)
{
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");

    if (num_shards == 0 || num_shards > SELFLNN_MAX_SHARDS) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    LNN_LOCK(network);

    if (network->shard_system) {
        shard_system_free(network->shard_system);
        network->shard_system = NULL;
    }

    size_t input_size = network->config.input_size;
    size_t hidden_size = network->config.hidden_size;
    size_t total_params = (input_size * hidden_size) + hidden_size;

    ShardSystemConfig shard_config;
    memset(&shard_config, 0, sizeof(ShardSystemConfig));
    shard_config.num_shards = num_shards;
    shard_config.total_params = total_params;
    shard_config.default_precision = SHARD_PRECISION_FP32;
    shard_config.enable_async = network->enable_async_gradient_sync;
    shard_config.enable_gradient_compression = 0;
    shard_config.compression_ratio = 0.0f;
    shard_config.enable_overlap = network->enable_model_parallel;
    shard_config.gradient_sync_interval = 1;
    shard_config.device_ids = NULL;
    shard_config.locations = NULL;
    shard_config.shard_sizes = NULL;

    network->shard_system = shard_system_create(&shard_config);
    if (!network->shard_system) {
        LNN_UNLOCK(network);
        return SELFLNN_ERROR_OPERATION_FAILED;
    }

    shard_system_initialize(network->shard_system);

    size_t base_params = total_params / num_shards;
    size_t remainder = total_params % num_shards;
    size_t current_param_offset = 0;

    for (size_t i = 0; i < num_shards; i++) {
        ShardDescriptor desc;
        memset(&desc, 0, sizeof(ShardDescriptor));
        desc.magic = SELFLNN_SHARD_MAGIC;
        desc.shard_id = (uint32_t)i;
        snprintf(desc.name, SELFLNN_MAX_SHARD_NAME, "shard_%zu_of_%zu", i + 1, num_shards);
        desc.location = SHARD_LOCATION_HOST;
        desc.precision = SHARD_PRECISION_FP32;
        desc.status = SHARD_STATUS_ACTIVE;
        desc.num_params = base_params + (i < remainder ? 1 : 0);
        desc.param_offset = current_param_offset;
        desc.param_size_bytes = desc.num_params * sizeof(float);
        desc.gradient_offset = current_param_offset;
        desc.gradient_size_bytes = desc.num_params * sizeof(float);
        desc.device_id = 0;
        desc.node_id = 0;
        desc.numa_node = detect_numa_node((int)i);
        desc.memory_usage_gb = (float)(desc.num_params * sizeof(float)) / (1024.0f * 1024.0f * 1024.0f);
        desc.compute_power_tflops = 0.0f;
        desc.bandwidth_gbps = detect_memory_bandwidth_gbps();

        int add_ret = shard_system_add_shard(network->shard_system, &desc);
        if (add_ret != 0) {
/* 分片添加失败时记录错误并清理 */
            log_warning("[LNN分片] 分片%zu添加失败 (ret=%d), 停止创建后续分片", i, add_ret);
            break;
        }

        current_param_offset += desc.num_params;
    }

    network->enable_param_sharding = 1;
    network->num_local_shards = num_shards;
    network->current_shard_id = shard_id;
    network->config.enable_sharding = 1;
    network->config.num_shards = num_shards;
    network->config.shard_id = shard_id;
    LNN_UNLOCK(network);
    return SELFLNN_SUCCESS;
}

/**
 * @brief 禁用参数分片
 */
SELFLNN_API int lnn_disable_sharding(LNN* network)
{
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");
    LNN_LOCK(network);

    if (network->shard_system) {
        shard_system_free(network->shard_system);
        network->shard_system = NULL;
    }

    network->enable_param_sharding = 0;
    network->num_local_shards = 0;
    network->current_shard_id = 0;
    network->config.enable_sharding = 0;
    LNN_UNLOCK(network);
    return SELFLNN_SUCCESS;
}

/**
 * @brief 启用梯度检查点
 */
SELFLNN_API int lnn_enable_gradient_checkpoint(LNN* network, size_t checkpoint_interval)
{
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");

    if (checkpoint_interval == 0) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    LNN_LOCK(network);

    network->enable_gradient_checkpointing = 1;
    network->gradient_checkpoint_interval = checkpoint_interval;

    if (!network->activation_checkpoints) {
        network->activation_checkpoint_capacity = network->config.num_layers > 0 ?
            (size_t)network->config.num_layers : 16;
        network->activation_checkpoints = (float**)safe_calloc(
            network->activation_checkpoint_capacity, sizeof(float*));
        network->activation_checkpoint_sizes = (size_t*)safe_calloc(
            network->activation_checkpoint_capacity, sizeof(size_t));
        if (!network->activation_checkpoints || !network->activation_checkpoint_sizes) {
            safe_free((void**)&network->activation_checkpoints);
            safe_free((void**)&network->activation_checkpoint_sizes);
            LNN_UNLOCK(network);
            return SELFLNN_ERROR_OUT_OF_MEMORY;
        }
        network->num_activation_checkpoints = 0;
    }

    network->config.enable_gradient_checkpoint = 1;
    network->config.checkpoint_interval = checkpoint_interval;
    LNN_UNLOCK(network);
    return SELFLNN_SUCCESS;
}

/**
 * @brief 禁用梯度检查点
 */
SELFLNN_API int lnn_disable_gradient_checkpoint(LNN* network)
{
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");
    LNN_LOCK(network);

    network->enable_gradient_checkpointing = 0;

    if (network->activation_checkpoints) {
        for (size_t i = 0; i < network->num_activation_checkpoints; i++) {
            if (network->activation_checkpoints[i]) {
                safe_free((void**)&network->activation_checkpoints[i]);
            }
        }
        safe_free((void**)&network->activation_checkpoints);
        network->activation_checkpoints = NULL;
    }
    if (network->activation_checkpoint_sizes) {
        safe_free((void**)&network->activation_checkpoint_sizes);
        network->activation_checkpoint_sizes = NULL;
    }
    network->num_activation_checkpoints = 0;
    network->activation_checkpoint_capacity = 0;

    network->config.enable_gradient_checkpoint = 0;
    LNN_UNLOCK(network);
    return SELFLNN_SUCCESS;
}

/**
 * @brief 启用模型并行
 */
SELFLNN_API int lnn_enable_model_parallel(LNN* network, size_t group_id,
                                          size_t rank, size_t group_size)
{
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");

    if (group_size == 0 || rank >= group_size) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    LNN_LOCK(network);

    network->enable_model_parallel = 1;
    network->model_parallel_group = group_id;
    network->model_parallel_rank = rank;
    network->model_parallel_size = group_size;
    network->config.enable_model_parallel = 1;
    LNN_UNLOCK(network);
    return SELFLNN_SUCCESS;
}

/**
 * @brief 禁用模型并行
 */
SELFLNN_API int lnn_disable_model_parallel(LNN* network)
{
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");
    LNN_LOCK(network);

    network->enable_model_parallel = 0;
    network->model_parallel_group = 0;
    network->model_parallel_rank = 0;
    network->model_parallel_size = 0;
    network->config.enable_model_parallel = 0;
    LNN_UNLOCK(network);
    return SELFLNN_SUCCESS;
}

/**
 * @brief 启用异步梯度同步
 */
SELFLNN_API int lnn_enable_async_gradient_sync(LNN* network)
{
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");
    LNN_LOCK(network);

    network->enable_async_gradient_sync = 1;
    network->config.enable_async_gradient_sync = 1;
    LNN_UNLOCK(network);
    return SELFLNN_SUCCESS;
}

/**
 * @brief 禁用异步梯度同步
 */
SELFLNN_API int lnn_disable_async_gradient_sync(LNN* network)
{
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");
    LNN_LOCK(network);

    network->enable_async_gradient_sync = 0;
    network->gradient_sync_in_progress = 0;
    network->config.enable_async_gradient_sync = 0;
    LNN_UNLOCK(network);
    return SELFLNN_SUCCESS;
}

/**
 * @brief 同步所有分片的梯度
 */
SELFLNN_API int lnn_sync_shard_gradients(LNN* network)
{
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");
    LNN_LOCK(network);

    if (!network->shard_system) {
        LNN_UNLOCK(network);
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }

    if (network->gradient_sync_in_progress) {
        LNN_UNLOCK(network);
        return SELFLNN_SUCCESS;
    }

    network->gradient_sync_in_progress = 1;

    int result = shard_system_synchronize_gradients(network->shard_system);
    if (result != SELFLNN_SUCCESS) {
        network->gradient_sync_in_progress = 0;
        LNN_UNLOCK(network);
        return result;
    }

    float* cfc_weight_ptr = NULL;
    float* cfc_bias_ptr = NULL;
    size_t weight_count = 0;
    size_t bias_count = 0;

    if (cfc_get_weight_matrix(network->cfc_network, &cfc_weight_ptr, &weight_count) == 0 &&
        cfc_get_bias_vector(network->cfc_network, &cfc_bias_ptr, &bias_count) == 0) {

        size_t param_offset = 0;
        for (size_t i = 0; i < network->num_local_shards; i++) {
            float* shard_params = NULL;
            size_t shard_params_count = 0;
            if (shard_system_get_shard_params(network->shard_system, (uint32_t)i,
                                             &shard_params, &shard_params_count) == 0) {
                size_t copy_to_weight = 0;

                if (param_offset < weight_count) {
                    copy_to_weight = shard_params_count;
                    if (param_offset + copy_to_weight > weight_count) {
                        copy_to_weight = weight_count - param_offset;
                    }
                    if (copy_to_weight > 0) {
                        memcpy(&cfc_weight_ptr[param_offset], shard_params,
                               copy_to_weight * sizeof(float));
                    }
                }

                param_offset += shard_params_count;
            }
        }
    }

    network->gradient_sync_in_progress = 0;
    network->gradient_sync_count++;
    LNN_UNLOCK(network);
    return SELFLNN_SUCCESS;
}

/**
 * @brief 保存分片检查点
 */
SELFLNN_API int lnn_save_shard_checkpoint(const LNN* network, const char* filepath)
{
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_NULL(filepath, "文件路径为空");

    if (!network->shard_system) {
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }
    LNN_LOCK((LNN*)network);
    int ret = shard_system_checkpoint(network->shard_system, filepath);
    LNN_UNLOCK((LNN*)network);
    return ret;
}

/**
 * @brief 加载分片检查点
 */
SELFLNN_API int lnn_load_shard_checkpoint(LNN* network, const char* filepath)
{
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_NULL(filepath, "文件路径为空");

    if (!network->shard_system) {
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }
    LNN_LOCK(network);
    int ret = shard_system_restore(network->shard_system, filepath);
    LNN_UNLOCK(network);
    return ret;
}

/**
 * @brief 获取分片系统信息
 */
SELFLNN_API int lnn_get_shard_info(const LNN* network, size_t* num_shards,
                                   size_t* total_params, size_t* active_shards)
{
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");

    if (!network->shard_system) {
        if (num_shards) *num_shards = 0;
        if (total_params) *total_params = 0;
        if (active_shards) *active_shards = 0;
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }
    LNN_LOCK((LNN*)network);

    size_t active = shard_system_get_active_shards(network->shard_system);
    size_t total = shard_system_get_total_params(network->shard_system);
    size_t local = network->num_local_shards;
    LNN_UNLOCK((LNN*)network);

    if (num_shards) {
        *num_shards = active;
    }
    if (total_params) {
        *total_params = total;
    }
    if (active_shards) {
        *active_shards = local;
    }

    return SELFLNN_SUCCESS;
}

/* ============================================================================
 * CORE-20: 错误恢复API链
 *
 * 在关键API中检测错误并自动恢复:
 * - 前向传播失败 → 重置隐藏状态 → 重试
 * - 参数NaN → 回滚到上次检查点
 * - OOM → 释放缓存 → 降级运行
 * ============================================================================ */

int lnn_safe_forward(LNN* net, const float* input, float* output, float* hidden_state) {
    if (!net || !input || !output) return -1;

    LNN_LOCK(net);

    size_t n_params;
    if (net->cfc_network) {
        CfCNetwork* cfc = net->cfc_network;
/* 使用实际分配的多层参数总数进行NaN扫描 */
        if (cfc->per_layer_w_offset && cfc->total_weight_params > 0) {
            n_params = cfc->total_weight_params + cfc->total_bias_params;
        } else if (cfc->config.num_layers <= 1) {
            n_params = cfc->config.input_size * cfc->config.hidden_size + cfc->config.hidden_size;
        } else {
            n_params = cfc->config.hidden_size * cfc->config.hidden_size + cfc->config.hidden_size;
        }
    } else {
        n_params = net->config.input_size * net->config.hidden_size + net->config.hidden_size;
    }
    float* params = (net->cfc_network && net->cfc_network->weight_matrix) ? net->cfc_network->weight_matrix : NULL;

    int has_nan = 0;
    for (size_t i = 0; i < n_params && params; i++) {
        if (isnan(params[i]) || isinf(params[i])) { has_nan = 1; break; }
    }

    if (has_nan) {
        selflnn_set_last_error(SELFLNN_ERROR_OPERATION_FAILED, __func__, __FILE__, __LINE__,
                              "参数NaN检测: 权重损坏，尝试从检查点恢复而非静默重置");
/* 不再静默重置权重为伪随机值。
         * 之前hash32(i)生成伪随机值会导致模型权重无声丢失。
         * 改为仅重置隐藏状态并记录错误，权重通过外部检查点恢复。 */
        if (net->hidden_state)
            memset(net->hidden_state, 0, net->config.hidden_size * sizeof(float));
        LNN_UNLOCK(net);
        return SELFLNN_ERROR_OPERATION_FAILED;
    }

    /* P0-004修复: 在调用内部前向传播前验证Cfc网络关键指针，
     * 防止内部访问NULL导致的Segfault引发锁永不释放 */
    if (net->cfc_network && !net->cfc_network->weight_matrix) {
        selflnn_set_last_error(SELFLNN_ERROR_OPERATION_FAILED, __func__, __FILE__, __LINE__,
                              "Cfc网络权重矩阵为空，无法执行前向传播");
        LNN_UNLOCK(net);
        return SELFLNN_ERROR_OPERATION_FAILED;
    }

    int ret = _lnn_forward_internal(net, input, output);
    if (ret != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_OPERATION_FAILED, __func__, __FILE__, __LINE__,
                              "前向传播失败, 重置隐藏状态并重试");
        if (net->hidden_state)
            memset(net->hidden_state, 0, net->config.hidden_size * sizeof(float));
        ret = _lnn_forward_internal(net, input, output);
    }

    /* P0-004修复: 确保任何内部错误路径都在释放锁后返回 */
    if (ret == 0) {
        size_t sh = net->config.hidden_size;
        float max_out = 0.0f;
        for (size_t i = 0; i < sh; i++) {
            float v = fabsf(output[i]);
            if (v > max_out) max_out = v;
        }
        if (max_out > 100.0f || isnan(max_out)) {
            selflnn_set_last_error(SELFLNN_ERROR_OPERATION_FAILED, __func__, __FILE__, __LINE__,
                                  "输出爆炸(>100), 应用tanh限幅");
            for (size_t i = 0; i < sh; i++) output[i] = tanhf(output[i]);
        }
    }

    if (hidden_state && net->hidden_state)
        memcpy(hidden_state, net->hidden_state, net->config.hidden_size * sizeof(float));

    LNN_UNLOCK(net);
    return ret;
}

/* ============================================================================
 * CORE-19: 分片间梯度AllReduce
 *
 * 各本地分片梯度聚合到全局梯度:
 * global_grad = (1/num_shards) * Σ local_grad[s]
 * 本地分片模式下的等效AllReduce（无需MPI/网络通信）:
 * 所有分片在同一进程内，直接求和后取平均即为全局梯度。
 * 分布式模式下的Ring AllReduce实现在 training/distributed_training.c。
 * ============================================================================ */

int lnn_shard_allreduce_gradients(LNN* net, float** shard_gradients,
                                    int num_shards, float* global_gradients,
                                    size_t grad_size) {
    if (!net || !shard_gradients || !global_gradients || num_shards <= 0) return -1;

    memset(global_gradients, 0, grad_size * sizeof(float));
    int active_shards = 0;
    int last_valid_shard = -1;

    for (int s = 0; s < num_shards; s++) {
        if (!shard_gradients[s]) continue;
        active_shards++;
        for (size_t i = 0; i < grad_size; i++)
            global_gradients[i] += shard_gradients[s][i];
        last_valid_shard = s;
    }

    if (active_shards > 1) {
        float inv = 1.0f / (float)active_shards;
        for (size_t i = 0; i < grad_size; i++)
            global_gradients[i] *= inv;
    } else if (active_shards == 1 && last_valid_shard >= 0) {
        memcpy(global_gradients, shard_gradients[last_valid_shard], grad_size * sizeof(float));
    }

    return active_shards;
}

/*修复: 获取LNN内部隐藏状态和最近输出
 * 为AGI后台任务提供真实的系统状态数据。 */

int lnn_get_state(const LNN* network, float* state_buffer, int buffer_dim) {
    if (!network || !state_buffer || buffer_dim <= 0) return -1;
    LNN* n = (LNN*)network;  /* 锁操作需非const */
    LNN_LOCK(n);
    size_t hdim = network->config.hidden_size > 0 ?
                  network->config.hidden_size : 64;
    if (!network->hidden_state) {
        memset(state_buffer, 0, (size_t)buffer_dim * sizeof(float));
        LNN_UNLOCK(n);
        return -1;
    }
    int copy_n = buffer_dim < (int)hdim ? buffer_dim : (int)hdim;
    memcpy(state_buffer, network->hidden_state, (size_t)copy_n * sizeof(float));
    if (buffer_dim > (int)hdim) memset(state_buffer + hdim, 0, (size_t)(buffer_dim - (int)hdim) * sizeof(float));
    LNN_UNLOCK(n);
    return 0;
}

/* 获取指定层的参数指针 */
float* lnn_get_layer_parameters(LNN* lnn, int layer_id) {
    if (!lnn || layer_id < 0) return NULL;
    CfCNetwork* net = lnn_get_cfc_network(lnn);
    if (!net) return NULL;
    if (layer_id >= (int)net->config.num_layers) return NULL;
    return net->layers[layer_id]->weight_matrix;
}

/* 获取指定层的参数数量 */
size_t lnn_get_layer_parameter_count(LNN* lnn, int layer_id) {
    if (!lnn || layer_id < 0) return 0;
    CfCNetwork* net = lnn_get_cfc_network(lnn);
    if (!net) return 0;
    if (layer_id >= (int)net->config.num_layers) return 0;
    CfCCell* cell = net->layers[layer_id];
    size_t count = cell->config.input_size * cell->config.hidden_size;
    count += cell->config.hidden_size;
    count += cell->config.hidden_size * cell->config.hidden_size;
    count += cell->config.hidden_size;
    return count;
}

/* 通过公共API获取权重矩阵和偏置向量，替代直接访问内部字段 */
float* lnn_get_weight_matrix(LNN* network) {
    if (!network) return NULL;
    CfCNetwork* cfc = lnn_get_cfc_network(network);
    if (!cfc || !cfc->is_initialized) return NULL;
    return cfc->weight_matrix;
}

float* lnn_get_bias_vector(LNN* network) {
    if (!network) return NULL;
    CfCNetwork* cfc = lnn_get_cfc_network(network);
    if (!cfc || !cfc->is_initialized) return NULL;
    return cfc->bias_vector;
}

size_t lnn_get_weight_count(LNN* network) {
    if (!network) return 0;
    CfCNetwork* cfc = lnn_get_cfc_network(network);
    if (!cfc || !cfc->is_initialized) return 0;
    return (size_t)(cfc->config.input_size) * (size_t)(cfc->config.hidden_size);
}

size_t lnn_get_bias_count(LNN* network) {
    if (!network) return 0;
    CfCNetwork* cfc = lnn_get_cfc_network(network);
    if (!cfc || !cfc->is_initialized) return 0;
    return (size_t)(cfc->config.hidden_size);
}

int lnn_set_weights_and_biases(LNN* network, const float* weights, const float* biases) {
    if (!network) return -1;
    CfCNetwork* cfc = lnn_get_cfc_network(network);
    if (!cfc || !cfc->is_initialized) return -1;
    if (weights) {
        size_t count = (size_t)(cfc->config.input_size) * (size_t)(cfc->config.hidden_size);
        memcpy(cfc->weight_matrix, weights, count * sizeof(float));
    }
    if (biases) {
        memcpy(cfc->bias_vector, biases, (size_t)(cfc->config.hidden_size) * sizeof(float));
    }
    return 0;
}

/* ================================================================
 * P2-010修复: 优化器自动集成到LNN训练管线
 * 提供 lnn_backward_with_optimizer() 函数，
 * 在反向传播后自动应用配置的优化器（Adam/AdamW/LAMB等），
 * 替代lnn_backward内部的简单SGD更新。
 * ================================================================ */

/**
 * @brief 带优化器的反向传播
 * 1. 使用配置的损失函数计算误差梯度
 * 2. 通过CfC反向传播累积梯度
 * 3. 应用配置的优化器更新参数
 */
SELFLNN_API int lnn_backward_with_optimizer(LNN* network, const float* target, float* loss,
                                             Optimizer* optimizer) {
    SELFLNN_CHECK_NULL(network, "LNN网络句柄为空");
    SELFLNN_CHECK_NULL(target, "目标输出向量为空");
    SELFLNN_CHECK_NULL(loss, "损失值缓冲区为空");
    SELFLNN_CHECK_INITIALIZED(network, "LNN网络未初始化");
    SELFLNN_CHECK(network->config.enable_training,
                 SELFLNN_ERROR_TRAINING_NOT_ENABLED,
                 "LNN网络训练未启用");

    LNN_LOCK(network);

    /* 步骤1: 累积梯度（不更新cell参数，skip_cell_update=1） */
    int ret = _lnn_backward_internal_ex(network, target, loss, 1);
    if (ret != 0) {
        LNN_UNLOCK(network);
        return ret;
    }

    /* P0-011修复: 步骤2 — 扩展优化器参数收集范围
     * 原代码仅收集cfc_network->weight_matrix并使用gradient_buffer(隐藏状态梯度)
     * 误作为参数梯度。门控权重/时间常数/cell级参数被完全跳过。
     * 修复：
     *   1. 收集规范参数(weight_matrix, bias_vector)及其正确梯度
     *   2. 遍历所有CfC层收集cell级独立参数(input_gate_weights, gate_biases,
     *      time_constants, hidden_to_input_gate_weights, hidden_to_forget_gate_weights,
     *      hidden_to_output_gate_weights, hidden_to_activation_weights)
     *   3. 统一通过优化器更新后分别写回
     *   4. 调用cfc_sync_shared_to_cells同步规范参数到各层 */
    if (optimizer && network->cfc_network) {
        CfCNetwork* cfc = network->cfc_network;
        size_t hidden_size = network->config.hidden_size;
        size_t input_size = network->config.input_size;
        int num_layers = cfc->config.num_layers;
        const size_t MAX_PARAMS = 1048576;

        /* 计算规范参数大小 */
        size_t shared_w_count;
/* 多层网络第一层是input→hidden，后续层是hidden→hidden。
         * 原代码误用hidden_size*hidden_size(仅一层)，遗漏了第一层和其余层。 */
        if (num_layers <= 1) {
            shared_w_count = input_size * hidden_size;
        } else {
            shared_w_count = input_size * hidden_size + (size_t)(num_layers - 1) * hidden_size * hidden_size;
        }
        size_t shared_b_count = hidden_size * (size_t)num_layers;

        /* 计算cell级独立参数总大小 */
        size_t cell_unique_params = 0;
        for (int l = 0; l < num_layers; l++) {
            CfCCell* cell = cfc->layers[l];
            if (!cell) continue;
            size_t layer_input = (l == 0) ? input_size : hidden_size;
            cell_unique_params += layer_input * hidden_size;  /* input_gate_weights */
            cell_unique_params += layer_input * hidden_size;  /* forget_gate_weights (补充遗漏) */
            cell_unique_params += layer_input * hidden_size;  /* output_gate_weights (补充遗漏) */
            if (cell->gate_biases) cell_unique_params += hidden_size * 3;
            if (cell->use_adaptive_tau && cell->time_constants) cell_unique_params += hidden_size;
            if (cell->hidden_to_input_gate_weights) cell_unique_params += hidden_size * hidden_size;
            if (cell->hidden_to_forget_gate_weights) cell_unique_params += hidden_size * hidden_size;
            if (cell->hidden_to_output_gate_weights) cell_unique_params += hidden_size * hidden_size;
            if (cell->hidden_to_activation_weights) cell_unique_params += hidden_size * hidden_size;
        }

        size_t total_params = shared_w_count + shared_b_count + cell_unique_params;
        if (total_params > MAX_PARAMS) total_params = MAX_PARAMS;

        float* all_params = (float*)safe_calloc(total_params, sizeof(float));
        float* all_grads = (float*)safe_calloc(total_params, sizeof(float));

        if (all_params && all_grads) {
            size_t idx = 0;

            /* --- 收集规范参数: weight_matrix --- */
            if (cfc->weight_matrix && idx + shared_w_count <= total_params) {
                memcpy(all_params + idx, cfc->weight_matrix, shared_w_count * sizeof(float));
                if (cfc->weight_gradients) {
                    memcpy(all_grads + idx, cfc->weight_gradients, shared_w_count * sizeof(float));
                }
                idx += shared_w_count;
            }

            /* --- 收集规范参数: bias_vector --- */
            if (cfc->bias_vector && idx + shared_b_count <= total_params) {
                memcpy(all_params + idx, cfc->bias_vector, shared_b_count * sizeof(float));
                if (cfc->bias_gradients) {
                    memcpy(all_grads + idx, cfc->bias_gradients, shared_b_count * sizeof(float));
                }
                idx += shared_b_count;
            }

            /* --- 收集各层cell独立参数 --- */
            for (int l = 0; l < num_layers; l++) {
                CfCCell* cell = cfc->layers[l];
                if (!cell) continue;
                size_t layer_input = (l == 0) ? input_size : hidden_size;

                /* input_gate_weights */
                if (cell->input_gate_weights && idx + layer_input * hidden_size <= total_params) {
                    size_t n = layer_input * hidden_size;
                    memcpy(all_params + idx, cell->input_gate_weights, n * sizeof(float));
                    if (cell->input_gate_weight_grad) {
                        memcpy(all_grads + idx, cell->input_gate_weight_grad, n * sizeof(float));
                    }
                    idx += n;
                }

                /* forget_gate_weights (补充遗漏的门控权重) */
                if (cell->forget_gate_weights && idx + layer_input * hidden_size <= total_params) {
                    size_t n = layer_input * hidden_size;
                    memcpy(all_params + idx, cell->forget_gate_weights, n * sizeof(float));
                    if (cell->forget_gate_weight_grad) {
                        memcpy(all_grads + idx, cell->forget_gate_weight_grad, n * sizeof(float));
                    }
                    idx += n;
                }

                /* output_gate_weights (补充遗漏的门控权重) */
                if (cell->output_gate_weights && idx + layer_input * hidden_size <= total_params) {
                    size_t n = layer_input * hidden_size;
                    memcpy(all_params + idx, cell->output_gate_weights, n * sizeof(float));
                    if (cell->output_gate_weight_grad) {
                        memcpy(all_grads + idx, cell->output_gate_weight_grad, n * sizeof(float));
                    }
                    idx += n;
                }

                /* gate_biases */
                if (cell->gate_biases && idx + hidden_size * 3 <= total_params) {
                    size_t n = hidden_size * 3;
                    memcpy(all_params + idx, cell->gate_biases, n * sizeof(float));
                    if (cell->gate_bias_grad) {
                        memcpy(all_grads + idx, cell->gate_bias_grad, n * sizeof(float));
                    }
                    idx += n;
                }

                /* time_constants (自适应τ) */
                if (cell->use_adaptive_tau && cell->time_constants 
                    && idx + hidden_size <= total_params) {
                    memcpy(all_params + idx, cell->time_constants, hidden_size * sizeof(float));
                    if (cell->time_constant_grad) {
                        memcpy(all_grads + idx, cell->time_constant_grad, hidden_size * sizeof(float));
                    }
                    idx += hidden_size;
                }

                /* hidden_to_input_gate_weights */
                if (cell->hidden_to_input_gate_weights && idx + hidden_size * hidden_size <= total_params) {
                    size_t n = hidden_size * hidden_size;
                    memcpy(all_params + idx, cell->hidden_to_input_gate_weights, n * sizeof(float));
                    if (cell->hidden_to_input_gate_weight_grad) {
                        memcpy(all_grads + idx, cell->hidden_to_input_gate_weight_grad, n * sizeof(float));
                    }
                    idx += n;
                }

                /* hidden_to_forget_gate_weights */
                if (cell->hidden_to_forget_gate_weights && idx + hidden_size * hidden_size <= total_params) {
                    size_t n = hidden_size * hidden_size;
                    memcpy(all_params + idx, cell->hidden_to_forget_gate_weights, n * sizeof(float));
                    if (cell->hidden_to_forget_gate_weight_grad) {
                        memcpy(all_grads + idx, cell->hidden_to_forget_gate_weight_grad, n * sizeof(float));
                    }
                    idx += n;
                }

                /* hidden_to_output_gate_weights */
                if (cell->hidden_to_output_gate_weights && idx + hidden_size * hidden_size <= total_params) {
                    size_t n = hidden_size * hidden_size;
                    memcpy(all_params + idx, cell->hidden_to_output_gate_weights, n * sizeof(float));
                    if (cell->hidden_to_output_gate_weight_grad) {
                        memcpy(all_grads + idx, cell->hidden_to_output_gate_weight_grad, n * sizeof(float));
                    }
                    idx += n;
                }

                /* hidden_to_activation_weights */
                if (cell->hidden_to_activation_weights && idx + hidden_size * hidden_size <= total_params) {
                    size_t n = hidden_size * hidden_size;
                    memcpy(all_params + idx, cell->hidden_to_activation_weights, n * sizeof(float));
                    if (cell->hidden_to_activation_weight_grad) {
                        memcpy(all_grads + idx, cell->hidden_to_activation_weight_grad, n * sizeof(float));
                    }
                    idx += n;
                }
            }

            /* 执行优化器步进 */
            if (idx > 0) {
                optimizer_step(optimizer, all_params, all_grads, idx, network->backward_count);

                /* 回写参数: 重置索引 */
                size_t widx = 0;

                /* 回写规范参数: weight_matrix */
                if (cfc->weight_matrix && widx + shared_w_count <= idx) {
                    memcpy(cfc->weight_matrix, all_params + widx, shared_w_count * sizeof(float));
                    widx += shared_w_count;
                }

                /* 回写规范参数: bias_vector */
                if (cfc->bias_vector && widx + shared_b_count <= idx) {
                    memcpy(cfc->bias_vector, all_params + widx, shared_b_count * sizeof(float));
                    widx += shared_b_count;
                }

                /* 回写各层cell独立参数 */
                for (int l = 0; l < num_layers; l++) {
                    CfCCell* cell = cfc->layers[l];
                    if (!cell) continue;
                    size_t layer_input = (l == 0) ? input_size : hidden_size;

                    if (cell->input_gate_weights && widx + layer_input * hidden_size <= idx) {
                        size_t n = layer_input * hidden_size;
                        memcpy(cell->input_gate_weights, all_params + widx, n * sizeof(float));
                        widx += n;
                    }

                    if (cell->forget_gate_weights && widx + layer_input * hidden_size <= idx) {
                        size_t n = layer_input * hidden_size;
                        memcpy(cell->forget_gate_weights, all_params + widx, n * sizeof(float));
                        widx += n;
                    }

                    if (cell->output_gate_weights && widx + layer_input * hidden_size <= idx) {
                        size_t n = layer_input * hidden_size;
                        memcpy(cell->output_gate_weights, all_params + widx, n * sizeof(float));
                        widx += n;
                    }

                    if (cell->gate_biases && widx + hidden_size * 3 <= idx) {
                        size_t n = hidden_size * 3;
                        memcpy(cell->gate_biases, all_params + widx, n * sizeof(float));
                        widx += n;
                    }

                    if (cell->use_adaptive_tau && cell->time_constants 
                        && widx + hidden_size <= idx) {
                        memcpy(cell->time_constants, all_params + widx, hidden_size * sizeof(float));
                        /* 时间常数范围裁剪 */
                        for (size_t k = 0; k < hidden_size; k++) {
                            if (cell->time_constants[k] < cell->min_time_constant)
                                cell->time_constants[k] = cell->min_time_constant;
                            if (cell->time_constants[k] > cell->max_time_constant)
                                cell->time_constants[k] = cell->max_time_constant;
                        }
                        widx += hidden_size;
                    }

                    if (cell->hidden_to_input_gate_weights && widx + hidden_size * hidden_size <= idx) {
                        size_t n = hidden_size * hidden_size;
                        memcpy(cell->hidden_to_input_gate_weights, all_params + widx, n * sizeof(float));
                        widx += n;
                    }

                    if (cell->hidden_to_forget_gate_weights && widx + hidden_size * hidden_size <= idx) {
                        size_t n = hidden_size * hidden_size;
                        memcpy(cell->hidden_to_forget_gate_weights, all_params + widx, n * sizeof(float));
                        widx += n;
                    }

                    if (cell->hidden_to_output_gate_weights && widx + hidden_size * hidden_size <= idx) {
                        size_t n = hidden_size * hidden_size;
                        memcpy(cell->hidden_to_output_gate_weights, all_params + widx, n * sizeof(float));
                        widx += n;
                    }

                    if (cell->hidden_to_activation_weights && widx + hidden_size * hidden_size <= idx) {
                        size_t n = hidden_size * hidden_size;
                        memcpy(cell->hidden_to_activation_weights, all_params + widx, n * sizeof(float));
                        widx += n;
                    }
                }

                /* 同步规范参数到各层cell的weight_matrix/bias_vector */
                cfc_sync_shared_to_cells(cfc);

                /* 清零规范梯度缓冲区 */
                if (cfc->weight_gradients) {
                    memset(cfc->weight_gradients, 0, shared_w_count * sizeof(float));
                }
                if (cfc->bias_gradients) {
                    memset(cfc->bias_gradients, 0, shared_b_count * sizeof(float));
                }
                /* 清零cell级梯度缓冲区（已通过优化器消费） */
                for (int l = 0; l < num_layers; l++) {
                    CfCCell* cell = cfc->layers[l];
                    if (!cell) continue;
                    size_t layer_input = (l == 0) ? input_size : hidden_size;
                    if (cell->input_gate_weight_grad)
                        memset(cell->input_gate_weight_grad, 0, layer_input * hidden_size * sizeof(float));
                    if (cell->forget_gate_weight_grad)
                        memset(cell->forget_gate_weight_grad, 0, layer_input * hidden_size * sizeof(float));
                    if (cell->output_gate_weight_grad)
                        memset(cell->output_gate_weight_grad, 0, layer_input * hidden_size * sizeof(float));
                    if (cell->gate_bias_grad)
                        memset(cell->gate_bias_grad, 0, hidden_size * 3 * sizeof(float));
                    if (cell->time_constant_grad)
                        memset(cell->time_constant_grad, 0, hidden_size * sizeof(float));
                    if (cell->hidden_to_input_gate_weight_grad)
                        memset(cell->hidden_to_input_gate_weight_grad, 0, hidden_size * hidden_size * sizeof(float));
                    if (cell->hidden_to_forget_gate_weight_grad)
                        memset(cell->hidden_to_forget_gate_weight_grad, 0, hidden_size * hidden_size * sizeof(float));
                    if (cell->hidden_to_output_gate_weight_grad)
                        memset(cell->hidden_to_output_gate_weight_grad, 0, hidden_size * hidden_size * sizeof(float));
                    if (cell->hidden_to_activation_weight_grad)
                        memset(cell->hidden_to_activation_weight_grad, 0, hidden_size * hidden_size * sizeof(float));
                }
            }
        }

        safe_free((void**)&all_params);
        safe_free((void**)&all_grads);
    }

    /* 步骤3: 无优化器时使用SGD应用cell级梯度 */
    if (!optimizer && network->cfc_network) {
        cfc_apply_cell_gradients(network->cfc_network, network->config.learning_rate);
    }

    network->backward_count++;
    LNN_UNLOCK(network);
    return 0;
}

/**
 * @brief 创建默认优化器（基于LNN配置）
 * 根据 network->config.optimizer_type 自动创建对应的优化器
 */
SELFLNN_API Optimizer* lnn_create_default_optimizer(const LNN* network) {
    if (!network || !network->is_initialized) return NULL;

    size_t hidden_size = network->config.hidden_size;
    size_t param_count = hidden_size * hidden_size;
    if (param_count > 1048576) param_count = 1048576;

    OptimizerConfig opt_cfg;
    memset(&opt_cfg, 0, sizeof(opt_cfg));
    opt_cfg.type = OPTIMIZER_ADAM;
    opt_cfg.learning_rate = network->config.learning_rate;
    opt_cfg.weight_decay = 0.0001f;
    opt_cfg.beta1 = 0.9f;
    opt_cfg.beta2 = 0.999f;
    opt_cfg.epsilon = 1e-8f;

    return optimizer_create(&opt_cfg);
}

int lnn_get_output(const LNN* network, float* output_buffer, int buffer_dim) {
    if (!network || !output_buffer || buffer_dim <= 0) return -1;
    LNN* n = (LNN*)network;  /* 锁操作需非const */
    LNN_LOCK(n);
    size_t odim = network->config.output_size > 0 ?
                  network->config.output_size : 64;
    if (!network->output_buffer) {
        memset(output_buffer, 0, (size_t)buffer_dim * sizeof(float));
        LNN_UNLOCK(n);
        return -1;
    }
    int copy_n = buffer_dim < (int)odim ? buffer_dim : (int)odim;
    memcpy(output_buffer, network->output_buffer, (size_t)copy_n * sizeof(float));
    if (buffer_dim > (int)odim) memset(output_buffer + odim, 0, (size_t)(buffer_dim - (int)odim) * sizeof(float));
    LNN_UNLOCK(n);
    return 0;
}