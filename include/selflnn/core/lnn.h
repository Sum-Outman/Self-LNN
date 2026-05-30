#ifndef SELFLNN_LNN_H
#define SELFLNN_LNN_H

#include <stddef.h>
#include <stdint.h>
#include "selflnn/core/common.h"
#include "selflnn/core/cfc_network.h"
#include "selflnn/core/state.h"
#include "selflnn/core/parameter_shard.h"
#include "selflnn/utils/platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file lnn.h
 * @brief 液态神经网络核心接口
 * 
 * 液态神经网络（Liquid Neural Network）是一种连续时间递归神经网络，
 * 其动态由微分方程描述，能够处理多模态输入并产生统一输出。
 */

/**
 * @brief 液态神经网络配置结构体
 */
typedef struct {
    size_t input_size;       /**< 输入向量大小 */
    size_t hidden_size;      /**< 隐藏状态大小 */
    size_t output_size;      /**< 输出向量大小 */
    float learning_rate;     /**< 学习率 */
    float time_constant;     /**< 时间常数（秒） */
    float noise_std;         /**< 噪声标准差 */
    int enable_training;     /**< 是否启用训练 */
    int enable_adaptation;   /**< 是否启用参数自适应 */
    int enable_evolution;    /**< 是否启用演化 */
    int num_layers;          /**< 网络层数（默认1） */
    int ode_solver_type;     /**< ODE求解器类型: 0=封闭形式解, 1=RK4, 2=RK45, 3=CTBP */

    int enable_sharding;             /**< 是否启用参数分片 */
    size_t num_shards;               /**< 分片数量 */
    size_t shard_id;                 /**< 当前分片ID（多机时使用） */
    int enable_gradient_checkpoint;  /**< 是否启用梯度检查点 */
    size_t checkpoint_interval;      /**< 梯度检查点间隔层数 */
    int enable_model_parallel;       /**< 是否启用模型并行 */
    int enable_async_gradient_sync;  /**< 是否启用异步梯度同步 */
    float gradient_compression_ratio;/**< 梯度压缩比（0.0-1.0） */
    float max_grad_norm;            /**< 梯度裁剪阈值（默认5.0），统一控制梯度裁剪 */
    int enable_laplace;             /**< 是否启用拉普拉斯频域分析增强（默认1=启用） */
    int enable_quaternion;          /**< 是否启用四元数增强（默认1=启用，兼容CfCCell） */
    float dropout_rate;             /**< Dropout率（0.0-1.0，默认0.0=无dropout） */
    float weight_decay;             /**< 权重衰减系数（L2正则化，默认0.0） */
    int loss_function;              /**< FIX-003: 损失函数类型（LossType枚举值，默认0=MSE）。
                                        反向传播时使用此字段计算正确的误差梯度，
                                        替代之前硬编码MSE梯度导致的所有损失类型均错误。 */
} LNNConfig;

/**
 * @brief 液态神经网络句柄
 */
typedef struct LNN LNN;

/* 前向声明依赖类型（由 cfc_network.h 提供 CfCNetwork/NetworkState） */
typedef struct MemorySystem MemorySystem;
typedef struct LaplaceAnalyzer LaplaceAnalyzer;

/* 内部实现：仅在SELFLNN_IMPLEMENTATION定义时暴露结构体定义 */
#ifdef SELFLNN_IMPLEMENTATION
/**
 * @brief 液态神经网络内部结构体
 */
struct LNN {
    LNNConfig config;                /**< 网络配置 */
    CfCNetwork* cfc_network;         /**< CfC网络实例 */
    NetworkState* state;             /**< 网络状态 */
    float* hidden_state;             /**< 隐藏状态向量 */
    float* cell_state;               /**< 细胞状态向量 */
    float* input_buffer;             /**< 输入缓冲区 */
    float* output_buffer;            /**< 输出缓冲区 */
    float* error_buffer;             /**< 误差缓冲区 */
    float* gradient_buffer;          /**< 梯度缓冲区 */
    int is_initialized;              /**< 是否已初始化 */
    float current_loss;              /**< 当前损失值 */
    uint64_t forward_count;          /**< 前向传播计数 */
    uint64_t backward_count;         /**< 反向传播计数 */
    double total_training_time;      /**< 总训练时间 */
    int enable_evolution;            /**< 是否启用演化（与演化引擎联动） */
    int evolution_interval;          /**< 演化间隔（前向传播次数），默认100 */

    /* 关联的记忆系统（用于记忆增强前向传播） */
    MemorySystem* memory_system;     /**< 记忆系统实例 */
    float memory_context_strength;   /**< 记忆上下文注入强度（默认0.1） */
    size_t memory_context_top_k;     /**< 记忆检索top_k（默认3） */

    /* 关联的拉普拉斯分析器（用于频域梯度优化和稳定性分析） */
    LaplaceAnalyzer* laplace_analyzer;       /**< 拉普拉斯分析器实例 */
    float laplace_gradient_strength;         /**< 拉普拉斯梯度优化强度（默认0.1） */
    int laplace_forward_modulation;          /**< 是否启用前向传播频域调制 */

    /* 参数分片系统（用于万亿级参数分布式存储和计算） */
    ParameterShardSystem* shard_system;               /**< 参数分片系统实例 */
    int enable_param_sharding;                        /**< 是否启用参数分片 */
    size_t num_local_shards;                          /**< 本地分片数量 */
    size_t current_shard_id;                          /**< 当前分片ID */
    float* gradient_checkpoint_buffer;                /**< 梯度检查点缓冲区 */
    size_t gradient_checkpoint_size;                  /**< 梯度检查点缓冲区大小 */
    int enable_gradient_checkpointing;                /**< 是否启用梯度检查点 */
    size_t gradient_checkpoint_interval;              /**< 梯度检查点间隔 */

    /* 中间激活值缓存（用于梯度检查点重计算） */
    float** activation_checkpoints;                   /**< 激活检查点数组 */
    size_t* activation_checkpoint_sizes;              /**< 各检查点大小 */
    size_t* activation_checkpoint_layers;             /**< ZSFWS-M-011: 各检查点对应的层索引 */
    size_t num_activation_checkpoints;                /**< 检查点数量 */
    size_t activation_checkpoint_capacity;            /**< 检查点容量 */

    /* 模型并行相关 */
    int enable_model_parallel;                        /**< 是否启用模型并行 */
    size_t model_parallel_group;                      /**< 模型并行组ID */
    size_t model_parallel_rank;                       /**< 模型并行组内排名 */
    size_t model_parallel_size;                       /**< 模型并行组大小 */

    /* 异步梯度同步 */
    int enable_async_gradient_sync;                   /**< 是否启用异步梯度同步 */
    volatile int gradient_sync_in_progress;           /**< 梯度同步是否在进行中 */
    uint64_t gradient_sync_count;                     /**< 梯度同步计数 */

    MutexHandle lock;                                 /**< 网络内部锁（线程安全） */

    /* ZSFWS-P6修复: 四元数预处理状态缓冲区
     * 保存四元数处理前的hidden_state，供反向传播计算正确的梯度链 */
    float* quaternion_pre_buf;                        /**< 四元数处理前状态（反向传播用） */
};

/* 内部无锁函数声明（供扩展训练模块使用，调用者需持有 LNN_LOCK） */
int _lnn_forward_internal(LNN* network, const float* input, float* output);
int _lnn_backward_internal(LNN* network, const float* target, float* loss);
int _lnn_backward_batch_internal(LNN* network, const float* inputs, const float* output_gradients, float* parameter_gradients, size_t batch_size);
#endif /* SELFLNN_IMPLEMENTATION */

/**
 * @brief 创建液态神经网络实例
 * 
 * @param config 网络配置
 * @return LNN* 网络句柄，失败返回NULL
 */
LNN* lnn_create(const LNNConfig* config);

/**
 * @brief 释放液态神经网络实例
 * 
 * @param network 网络句柄
 */
void lnn_free(LNN* network);

/**
 * @brief 前向传播
 * 
 * @param network 网络句柄
 * @param input 输入向量
 * @param output 输出向量缓冲区
 * @return int 成功返回0，失败返回-1
 */
int lnn_forward(LNN* network, const float* input, float* output);

/* ZSFX-DEEP-R9-001: 公开LNN锁API - 外部模块跨操作原子性 */
SELFLNN_API void lnn_lock(LNN* network);
SELFLNN_API void lnn_unlock(LNN* network);

/**
 * @brief 安全前向传播（含边界检查和维度适配）
 * 
 * 当输入/输出维度与LNN配置不匹配时自动进行投影适配。
 * 输入过大时截断，输入过小时零填充；输出过大时截断，输出过小时只写入实际可用空间。
 * 
 * @param network 网络句柄
 * @param input 输入向量
 * @param input_size 输入向量实际维度
 * @param output 输出向量缓冲区
 * @param output_size 输出缓冲区实际容量
 * @return int 成功返回写入的浮点数数量，失败返回-1
 */
int lnn_forward_safe(LNN* network, const float* input, size_t input_size,
                     float* output, size_t output_size);

/* ================================================================
 * ZSFWS-P1-008: 并发前向传播——每调用方独立的隐藏状态副本
 * 解决互斥锁瓶颈：多个模态（视觉/音频/文本/传感器等）可同时
 * 使用隔离的CfC状态进行前向传播，无需等待彼此释放LNN全局锁。
 * 权重读取使用RWLock的读锁（多读者并发），仅训练写入时排他。
 * ================================================================ */

/** @brief 隔离前向传播状态（每模态/每调用方独立分配） */
typedef struct {
    float* hidden_state;     /**< 隐藏状态副本 [hidden_size] */
    float* cell_state;       /**< CfC细胞状态副本 [cell_state_size] */
    float* input_buffer;     /**< 输入缓冲区 [input_size] */
    float* output_buffer;    /**< 输出缓冲区 [output_size] */
    size_t hidden_size;
    size_t cell_state_size;
    size_t input_size;
    size_t output_size;
    int initialized;
} LNNForwardState;

/** @brief 创建隔离前向传播状态 */
LNNForwardState* lnn_forward_state_create(LNN* network);

/** @brief 释放隔离前向传播状态 */
void lnn_forward_state_free(LNNForwardState* state);

/** @brief 使用隔离状态的前向传播（无全局锁，可并发调用）
 *  注意：此函数仅对LNN权重加读锁（RWLock），隐藏状态使用state副本。
 *  多个调用方使用各自的LNNForwardState实例可安全并发。
 *  训练期间(写锁持有中)调用此函数会阻塞等待。
 *  @return 0=成功, -1=失败 */
int lnn_forward_isolated(LNN* network, LNNForwardState* state,
                         const float* input, float* output);

/**
 * @brief 反向传播（训练）
 * 
 * @param network 网络句柄
 * @param target 目标输出向量
 * @param loss 损失值输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int lnn_backward(LNN* network, const float* target, float* loss);

/**
 * @defgroup batch_training 批量训练API
 * @brief 液态神经网络批量训练接口族
 * 
 * 批量训练的三种推荐模式：
 * 
 * 模式一：单样本逐次累积（当前默认）
 *   for each sample in batch:
 *       lnn_forward(network, input[i], output[i])
 *       lnn_backward_accumulate(network, target[i], &loss)
 *   cfc_apply_cell_gradients(network->cfc_network)
 *   optimizer_step(optimizer, params, grads, param_count, learning_rate)
 * 
 * 模式二：批量反向传播（高性能GPU）
 *   lnn_forward_batch(network, inputs, outputs, batch_size)
 *   lnn_backward_batch(network, inputs, output_gradients, param_gradients, batch_size)
 *   optimizer_step(optimizer, params, param_gradients, param_count, learning_rate)
 * 
 * 模式三：手动skip_cell_update累积（高级用法）
 *   for each sample in batch:
 *       lnn_forward(network, input[i], output[i])
 *       _lnn_backward_internal_ex(network, target[i], &loss, 1)
 *   cfc_network_reset_cell_states(network->cfc_network)
 *   optimizer_step(optimizer, params, grads, param_count, learning_rate)
 * 
 * 注意：
 *   - lnn_backward() 始终以 skip_cell_update=0 调用，每样本独立反向，无梯度累积
 *   - lnn_backward_accumulate() 以 skip_cell_update=1 调用，仅累积梯度不更新
 *   - lnn_backward_batch() 为完整的批量梯度下降实现，支持梯度裁剪和优化策略
 * 
 * @{
 */

/**
 * @brief P0-002修复: 反向传播（仅累积梯度，不更新cell级参数）
 * 
 * 与 lnn_backward 的区别：
 *   - 跳过 cfc_backward 的 Step3（cell参数直接更新）
 *   - cell级梯度保留在各cell内部缓冲区中（用于随后批量统一下发）
 *   - 调用方须在批/epoch结束后调用 cfc_apply_cell_gradients()
 * 
 * @param network LNN网络句柄
 * @param target 目标输出 [output_size]
 * @param loss 损失值输出
 * @return 0成功，负值失败
 */
int lnn_backward_accumulate(LNN* network, const float* target, float* loss);

/**
 * @brief 保存网络到文件
 *
 * K-027: SELF-LNN 模型文件格式规范
 *
 * 文件格式: 二进制，小端字节序
 *
 * 文件结构:
 * ┌──────────────────────────────────────────────────┐
 * │ LNNFileHeader (64字节)                           │
 * │   magic[16]: "SELFLNN\0\0\0\0\0\0\0\0\0"       │
 * │   version: uint32_t (当前=1)                      │
 * │   marker: uint32_t (0=FROM_SCRATCH, 1=PRETRAINED)│
 * │   header_size: uint32_t (=64)                    │
 * │   checksum: uint32_t (header+data的简单校验和)    │
 * │   reserved[16]: 保留字段，写入0                    │
 * ├──────────────────────────────────────────────────┤
 * │ 网络配置段 (variable)                             │
 * │   input_size: uint32_t                           │
 * │   hidden_size: uint32_t                          │
 * │   output_size: uint32_t                          │
 * │   num_layers: uint32_t                           │
 * │   activation: uint32_t                           │
 * │   enable_training: uint32_t                      │
 * │   use_cfc: uint32_t                              │
 * │   cfc_config: (CFCConfig结构体)                   │
 * │   reserved_config[32]: 保留                       │
 * ├──────────────────────────────────────────────────┤
 * │ 权重数据段 (variable, float32小端)                 │
 * │   每层: [weights(W) | biases(b)]                 │
 * │   优化器状态(如果是预训练模型):                      │
 * │     [momentum | velocity | adam_m | adam_v]       │
 * ├──────────────────────────────────────────────────┤
 * │ 优化器状态段 (variable)                            │
 * │   optimizer_type: uint32_t                       │
 * │   learning_rate: float32                         │
 * │   step_count: uint32_t                           │
 * │   per_param_states: (variable)                    │
 * └──────────────────────────────────────────────────┘
 *
 * 校验和算法: 旋转哈希
 *   checksum = 0
 *   for each byte b in (header+data):
 *       checksum = ((checksum << 5) | (checksum >> 27)) + b
 *
 * @param network 网络句柄
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int lnn_save(const LNN* network, const char* filepath);

/**
 * @brief 从文件加载网络
 * 
 * @param filepath 文件路径
 * @return LNN* 网络句柄，失败返回NULL
 */
LNN* lnn_load(const char* filepath);

/**
 * @brief 从文件加载权重到已有LNN实例（P0-001修复）
 * 
 * 对称于lnn_save，将模型文件中的权重、隐藏状态、细胞状态加载到
 * 已经创建的LNN实例中。不需要重新创建LNN，避免破坏已有的子系统绑定。
 * 
 * @param lnn 已创建的LNN实例指针
 * @param filepath 模型文件路径
 * @return int 成功返回0，失败返回-1
 */
int lnn_load_from_file(LNN* lnn, const char* filepath);

/**
 * @brief 获取LNN网络配置
 * 
 * @param network 网络句柄
 * @param config 输出配置缓冲区
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int lnn_get_config(const LNN* network, LNNConfig* config);

/* ZSF-001修复: 获取LNN内部隐藏状态和最近输出 */
SELFLNN_API int lnn_get_state(const LNN* network, float* state_buffer, int buffer_dim);
SELFLNN_API int lnn_get_output(const LNN* network, float* output_buffer, int buffer_dim);

/**
 * @brief 获取LNN内部CfC网络句柄（高级用途：直接操作原始CfC反向传播）
 *
 * 仅用于需要精细控制梯度流的算法（如强化学习的actor-critic训练）。
 * 一般前向传播请使用 lnn_forward()。
 *
 * @param network 网络句柄
 * @return CfCNetwork* 内部CfC网络句柄，失败返回NULL
 */
SELFLNN_API CfCNetwork* lnn_get_cfc_network(LNN* network);

/**
 * @brief 获取LNN输入维度
 * 
 * @param network 网络句柄
 * @return size_t 输入维度，失败返回0
 */
static inline size_t lnn_get_input_size(const LNN* network) {
    LNNConfig cfg;
    if (lnn_get_config(network, &cfg) != 0) return 0;
    return cfg.input_size;
}

/**
 * @brief 获取LNN输出维度
 * 
 * @param network 网络句柄
 * @return size_t 输出维度，失败返回0
 */
static inline size_t lnn_get_output_size(const LNN* network) {
    LNNConfig cfg;
    if (lnn_get_config(network, &cfg) != 0) return 0;
    return cfg.output_size;
}

/**
 * @brief 设置网络配置
 * 
 * @param network 网络句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int lnn_set_config(LNN* network, const LNNConfig* config);

/**
 * @brief 获取网络参数数量
 * 
 * @param network 网络句柄
 * @return size_t 参数数量，失败返回0
 */
size_t lnn_get_parameter_count(const LNN* network);

/**
 * @brief 获取网络统计信息（非标准接口，用于监控和调试）
 * 
 * @param network 网络句柄
 * @param avg_loss 平均损失（可选，可以为NULL）
 * @param forward_count 前向传播次数（可选，可以为NULL）
 * @param backward_count 反向传播次数（可选，可以为NULL）
 * @param avg_time 平均处理时间（秒，可选，可以为NULL）
 * @return int 成功返回0，失败返回错误码
 */
SELFLNN_API int lnn_get_stats(const LNN* network, float* avg_loss, uint64_t* forward_count,
                  uint64_t* backward_count, double* avg_time);

/**
 * @brief 获取最大激活值数量
 * 
 * @param network 网络句柄
 * @return size_t 最大激活值数量，失败返回0
 */
size_t lnn_get_max_activation_count(const LNN* network);

/**
 * @brief 获取网络参数指针
 * 
 * @param network 网络句柄
 * @return float* 参数指针，失败返回NULL
 */
float* lnn_get_parameters(LNN* network);

/**
 * @brief 获取当前梯度缓冲区指针
 * 
 * 返回最后一次反向传播计算得到的梯度数据指针，
 * 用于自定义梯度更新（如IRL奖励权重更新）。
 * 
 * @param network 网络句柄
 * @return float* 梯度缓冲区指针，失败返回NULL
 */
float* lnn_get_gradients(LNN* network);

/* ZSFUSA: 获取指定层的参数指针和参数数量 */
float* lnn_get_layer_parameters(LNN* lnn, int layer_id);
size_t lnn_get_layer_parameter_count(LNN* lnn, int layer_id);

/**
 * @brief 批量前向传播
 * 
 * @param network 网络句柄
 * @param inputs 输入数据（形状为batch_size × input_size）
 * @param outputs 输出数据缓冲区（形状为batch_size × output_size）
 * @param batch_size 批量大小
 * @return int 成功返回0，失败返回-1
 */
int lnn_forward_batch(LNN* network, const float* inputs, float* outputs, size_t batch_size);

/**
 * @brief 批量反向传播
 * 
 * @param network 网络句柄
 * @param inputs 输入数据（形状为batch_size × input_size）
 * @param output_gradients 输出梯度数据（形状为batch_size × output_size）
 * @param parameter_gradients 参数梯度缓冲区（可选，可以为NULL）
 * @param batch_size 批量大小
 * @return int 成功返回0，失败返回-1
 */
int lnn_backward_batch(LNN* network, const float* inputs, const float* output_gradients, float* parameter_gradients, size_t batch_size);

/** @} */ /* 批量训练API组结束 */

/**
 * @brief 设置LNN的记忆系统引用（用于记忆增强前向传播）
 * 
 * @param network 网络句柄
 * @param memory_system 记忆系统句柄
 * @param context_strength 记忆上下文注入强度（推荐0.05-0.3范围）
 * @param top_k 记忆检索top_k数量（推荐3-5）
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int lnn_set_memory_system(LNN* network, void* memory_system,
                                      float context_strength, size_t top_k);

/**
 * @brief 记忆增强的前向传播
 * 
 * 在标准前向传播之前，基于输入从记忆系统检索语义上相关的记忆上下文，
 * 将上下文注入隐藏状态，使网络输出受到过往经验的调制。
 * 实现"记忆增强推理"：网络不仅处理当前输入，还参考了相关记忆中存储的模式。
 * 
 * @param network 网络句柄
 * @param input 输入向量（大小为input_size）
 * @param output 输出向量缓冲区（大小为output_size）
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int lnn_forward_with_memory_context(LNN* network, const float* input, float* output);

/**
 * @brief 设置LNN的拉普拉斯分析器（用于频域梯度优化和稳定性分析）
 * 
 * @param network 网络句柄
 * @param laplace_analyzer 拉普拉斯分析器句柄
 * @param gradient_strength 梯度优化强度（推荐0.05-0.3范围）
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int lnn_set_laplace_analyzer(LNN* network, void* laplace_analyzer, float gradient_strength);

/**
 * @brief 设置LNN演化能力开关（与演化引擎联动）
 * 
 * @param network 网络句柄
 * @param enable 1启用演化，0禁用
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int lnn_set_evolution_enabled(LNN* network, int enable);

/**
 * @brief 获取LNN演化能力状态
 * 
 * @param network 网络句柄
 * @return int 1已启用，0已禁用，-1失败
 */
SELFLNN_API int lnn_get_evolution_enabled(LNN* network);

/**
 * @brief 拉普拉斯增强的反向传播
 * 
 * 在标准反向传播基础上，对误差信号进行拉普拉斯频域滤波优化：
 * 1. 计算误差信号（target - output）
 * 2. 使用拉普拉斯变换对误差进行频域低通滤波（去除高频噪声）
 * 3. 将滤波后的误差传递给CfC反向传播
 * 4. 统计训练稳定性指标
 * 
 * @param network 网络句柄
 * @param target 目标输出向量
 * @param loss 损失值输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int lnn_backward_with_laplace(LNN* network, const float* target, float* loss);

/**
 * @brief LNN拉普拉斯稳定性分析（分析所有CfC单元的综合稳定性）
 * 
 * @param network 网络句柄
 * @param stability_score 输出稳定性分数（0.0-1.0）
 * @param dominant_frequency 输出主导频率（Hz）
 * @param bandwidth 输出带宽（Hz）
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int lnn_analyze_laplace_stability(LNN* network, float* stability_score,
                                              float* dominant_frequency, float* bandwidth);

/**
 * @brief 设置LNN的ODE求解器类型
 * 
 * 设置网络中所有CfC单元的ODE求解器类型。
 * ODE_SOLVER_CLOSED_FORM=0: 封闭形式解（默认，精确解析解）
 * ODE_SOLVER_RK4=1: 四阶龙格-库塔法（固定步长数值解）
 * ODE_SOLVER_RK45=2: 五阶龙格-库塔-费尔伯格法（自适应步长数值解）
 * ODE_SOLVER_CTBP=3: 连续时间反向传播法（精确梯度计算）
 * 
 * @param network 网络句柄
 * @param solver_type 求解器类型
 * @return int 成功返回0，失败返回-1
 */
int lnn_set_ode_solver(LNN* network, int solver_type);

/**
 * @brief 启用参数分片系统
 * 
 * @param network 网络句柄
 * @param num_shards 分片数量
 * @param shard_id 当前分片ID
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int lnn_enable_sharding(LNN* network, size_t num_shards, size_t shard_id);

/**
 * @brief 禁用参数分片系统
 * 
 * @param network 网络句柄
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int lnn_disable_sharding(LNN* network);

/**
 * @brief 启用梯度检查点
 * 
 * @param network 网络句柄
 * @param checkpoint_interval 检查点间隔层数
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int lnn_enable_gradient_checkpoint(LNN* network, size_t checkpoint_interval);

/**
 * @brief 禁用梯度检查点
 * 
 * @param network 网络句柄
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int lnn_disable_gradient_checkpoint(LNN* network);

/**
 * @brief 启用模型并行
 * 
 * @param network 网络句柄
 * @param group_id 并行组ID
 * @param rank 组内排名
 * @param group_size 组大小
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int lnn_enable_model_parallel(LNN* network, size_t group_id,
                                          size_t rank, size_t group_size);

/**
 * @brief 禁用模型并行
 * 
 * @param network 网络句柄
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int lnn_disable_model_parallel(LNN* network);

/**
 * @brief 启用异步梯度同步
 * 
 * @param network 网络句柄
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int lnn_enable_async_gradient_sync(LNN* network);

/**
 * @brief 禁用异步梯度同步
 * 
 * @param network 网络句柄
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int lnn_disable_async_gradient_sync(LNN* network);

/**
 * @brief 同步所有分片的梯度
 * 
 * @param network 网络句柄
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int lnn_sync_shard_gradients(LNN* network);

/**
 * @brief 保存分片检查点
 * 
 * @param network 网络句柄
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int lnn_save_shard_checkpoint(const LNN* network, const char* filepath);

/**
 * @brief 加载分片检查点
 * 
 * @param network 网络句柄
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int lnn_load_shard_checkpoint(LNN* network, const char* filepath);

/**
 * @brief 获取参数分片系统状态信息
 * 
 * @param network 网络句柄
 * @param num_shards 输出分片数量（可选）
 * @param total_params 输出总参数数量（可选）
 * @param active_shards 输出活跃分片数量（可选）
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int lnn_get_shard_info(const LNN* network, size_t* num_shards,
                                   size_t* total_params, size_t* active_shards);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_LNN_H