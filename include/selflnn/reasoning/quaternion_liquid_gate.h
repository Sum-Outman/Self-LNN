/**
 * @file quaternion_liquid_gate.h
 * @brief 四元数液态门控接口
 *
 * 基于四元数代数的CfC液态门控机制。使用四元数的4维超复数表示，
 * 通过门控(σ)和激活(tanh)双路径计算调制信号：
 *   M = σ(W_gate · x + b_gate) ⊙ tanh(W_act · x + b_act)
 *
 * 完全基于CfC闭式解门控方程，无QKV投影、无softmax、无注意力权重。
 */

#ifndef SELFLNN_QUATERNION_LIQUID_GATE_H
#define SELFLNN_QUATERNION_LIQUID_GATE_H

#include <stddef.h>
#include "selflnn/utils/math_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 四元数液态门控配置
 */
typedef struct {
    size_t input_dim;              /**< 输入特征维度(标量) */
    size_t quaternion_dim;         /**< 四元数输出维度(四元数数量) */
    float gate_bias_init;          /**< 门控偏置初始化值，默认2.0（初始偏向开启） */
    float init_scale;              /**< 权重初始化缩放，默认0.1 */
    int use_bias;                  /**< 是否使用偏置，默认1 */
    float learning_rate;           /**< 学习率，默认0.001 */
} QuaternionLiquidGateConfig;

/**
 * @brief 四元数液态门控权重结构
 */
typedef struct {
    float* gate_kernel;            /**< 门控核权重 [input_dim][quaternion_dim][4] */
    float* act_kernel;             /**< 激活核权重 [input_dim][quaternion_dim][4] */
    float* gate_bias;              /**< 门控偏置 [quaternion_dim][4] */
    float* act_bias;               /**< 激活偏置 [quaternion_dim][4] */
} QuaternionLiquidGateWeights;

/**
 * @brief 四元数液态门控层
 */
typedef struct {
    QuaternionLiquidGateConfig config;       /**< 配置 */
    QuaternionLiquidGateWeights weights;     /**< 权重 */

    /* 优化器状态(Adam) */
    float* adam_m;                           /**< 一阶动量 */
    float* adam_v;                           /**< 二阶动量 */
    size_t adam_step;                        /**< 优化步数 */

    /* 缓存(前向传播时使用) */
    Quaternion* cached_gate;                 /**< 缓存的门控信号 [seq_len][quaternion_dim] */
    Quaternion* cached_act;                  /**< 缓存的激活信号 [seq_len][quaternion_dim] */
    size_t cached_seq_len;                   /**< 缓存序列长度 */
} QuaternionLiquidGate;

/**
 * @brief 创建四元数液态门控层
 *
 * @param config 配置参数
 * @return QuaternionLiquidGate* 成功返回指针，失败返回NULL
 */
QuaternionLiquidGate* quaternion_liquid_gate_create(const QuaternionLiquidGateConfig* config);

/**
 * @brief 销毁四元数液态门控层
 *
 * @param gate 门控层指针
 */
void quaternion_liquid_gate_destroy(QuaternionLiquidGate* gate);

/**
 * @brief 前向传播(单序列)
 *
 * 计算液态门控调制：
 *   gate = σ(W_gate · x + b_gate)     // CfC门控
 *   act = tanh(W_act · x + b_act)     // CfC激活
 *   output = gate ⊙ act                // 门控调制
 *
 * @param gate 门控层指针
 * @param input 输入序列 [seq_len][input_dim]
 * @param seq_len 序列长度
 * @param output 输出序列 [seq_len][input_dim]，预先分配
 * @return int 成功返回0，失败返回-1
 */
int quaternion_liquid_gate_forward(QuaternionLiquidGate* gate,
                                   const float* input,
                                   size_t seq_len,
                                   float* output);

/**
 * @brief 前向传播(批量)
 *
 * @param gate 门控层指针
 * @param input 输入序列 [batch][seq_len][input_dim]
 * @param batch 批大小
 * @param seq_len 序列长度
 * @param output 输出序列 [batch][seq_len][input_dim]
 * @return int 成功返回0，失败返回-1
 */
int quaternion_liquid_gate_forward_batch(QuaternionLiquidGate* gate,
                                         const float* input,
                                         size_t batch, size_t seq_len,
                                         float* output);

/**
 * @brief 训练步骤(单步梯度下降)
 *
 * @param gate 门控层指针
 * @param input 输入 [batch][seq_len][input_dim]
 * @param target 目标 [batch][seq_len][input_dim]
 * @param batch 批大小
 * @param seq_len 序列长度
 * @return float 当前损失值
 */
float quaternion_liquid_gate_train_step(QuaternionLiquidGate* gate,
                                        const float* input,
                                        const float* target,
                                        size_t batch, size_t seq_len);

/**
 * @brief 训练步骤(解析梯度链式法则反向传播) — M-008修复
 *
 * 使用四元数链式法则计算解析梯度，替代逐参数数值梯度，
 * 大幅减少前向传播次数（从 O(kernel_size) 降为 O(1)）。
 * 需要前向传播缓存（cached_gate/cached_act），否则回退到数值梯度。
 *
 * @param gate 门控层指针
 * @param input 输入 [batch][seq_len][input_dim]
 * @param target 目标 [batch][seq_len][input_dim]
 * @param batch 批大小
 * @param seq_len 序列长度
 * @return float 当前损失值
 */
float quaternion_liquid_gate_train_step_analytic(QuaternionLiquidGate* gate,
                                                  const float* input,
                                                  const float* target,
                                                  size_t batch, size_t seq_len);

/**
 * @brief 重置优化器状态
 *
 * @param gate 门控层指针
 */
void quaternion_liquid_gate_reset_optimizer(QuaternionLiquidGate* gate);

/**
 * @brief 设置学习率
 *
 * @param gate 门控层指针
 * @param lr 新学习率
 */
void quaternion_liquid_gate_set_learning_rate(QuaternionLiquidGate* gate, float lr);

/**
 * @brief 保存模型参数到文件
 *
 * @param gate 门控层指针
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int quaternion_liquid_gate_save(const QuaternionLiquidGate* gate, const char* filepath);

/**
 * @brief 从文件加载模型参数
 *
 * @param gate 门控层指针
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int quaternion_liquid_gate_load(QuaternionLiquidGate* gate, const char* filepath);

/* ===========================================================================
 * A04.5.1 四元数相位位置编码
 *
 * 将序列位置编码为四元数相位旋转。使用多频段正弦/余弦函数生成相位角，
 * 组合为四元数表示。支持可学习频率，可直接替代RoPE位置编码。
 *
 * 编码方式：对于位置pos和频率ω_k，生成四元数：
 *   q_k = cos(ω_k · pos) + i · sin(ω_k · pos)
 * 多频段组合：Q = [q_0, q_1, ..., q_{K-1}]
 * =========================================================================== */

/**
 * @brief 相位位置编码配置
 */
typedef struct {
    size_t freq_count;           /**< 频率数量(相位维度)，编码输出四元数维度 = freq_count */
    float freq_min;              /**< 最小频率，默认1.0 */
    float freq_max;              /**< 最大频率，默认10000.0 */
    int learnable;               /**< 频率是否可学习(1)或固定(0)，默认0 */
} PhaseEncodingConfig;

/**
 * @brief 四元数相位位置编码器
 */
typedef struct {
    PhaseEncodingConfig config;  /**< 配置 */
    float* frequencies;          /**< 频率数组[freq_count]，log-spaced或可学习 */
    int is_initialized;          /**< 是否已初始化 */
} QuaternionPhaseEncoding;

/**
 * @brief 创建四元数相位位置编码器
 *
 * 初始化频率数组，从freq_min到freq_max对数间隔分布。
 *
 * @param config 配置参数
 * @return QuaternionPhaseEncoding* 成功返回指针，失败返回NULL
 */
QuaternionPhaseEncoding* quaternion_phase_encoding_create(const PhaseEncodingConfig* config);

/**
 * @brief 销毁四元数相位位置编码器
 *
 * @param pe 编码器指针
 */
void quaternion_phase_encoding_destroy(QuaternionPhaseEncoding* pe);

/**
 * @brief 前向计算四元数相位位置编码
 *
 * 对每个位置pos ∈ positions，计算：
 *   for k in 0..freq_count-1:
 *     θ_k = pos · freq_k
 *     q_k = (cos θ_k, sin θ_k, 0, 0)
 *   输出Q = [q_0, q_1, ..., q_{K-1}, 0, ...] 填充到quaternion_dim
 *
 * @param pe 编码器指针
 * @param positions 位置索引数组 [seq_len]
 * @param seq_len 序列长度
 * @param quaternion_dim 输出四元数维度(>= freq_count，多余补零)
 * @param output 输出四元数序列 [seq_len][quaternion_dim]
 * @return int 成功返回0，失败返回-1
 */
int quaternion_phase_encoding_forward(const QuaternionPhaseEncoding* pe,
                                       const float* positions, size_t seq_len,
                                       size_t quaternion_dim, Quaternion* output);

/**
 * @brief 更新可学习频率（训练时使用）
 *
 * @param pe 编码器指针
 * @param grad 频率梯度数组 [freq_count]
 * @param lr 学习率
 * @return int 成功返回0，失败返回-1
 */
int quaternion_phase_encoding_update(QuaternionPhaseEncoding* pe,
                                      const float* grad, float lr);

/**
 * @brief 保存相位编码器参数到文件
 *
 * @param pe 编码器指针
 * @param path 文件路径
 * @return int 成功返回0，失败返回-1
 */
int quaternion_phase_encoding_save(const QuaternionPhaseEncoding* pe, const char* path);

/**
 * @brief 从文件加载相位编码器参数
 *
 * @param pe 编码器指针
 * @param path 文件路径
 * @return int 成功返回0，失败返回-1
 */
int quaternion_phase_encoding_load(QuaternionPhaseEncoding* pe, const char* path);

/* ===========================================================================
 * A04.5.1 四元数液态状态聚合
 *
 * 将多个四元数液态状态融合为单一聚合状态。支持简单平均和注意力加权
 * 两种聚合方式。注意力聚合使用可学习的查询投影向量计算各状态的权重。
 * =========================================================================== */

/**
 * @brief 液态状态聚合配置
 */
typedef struct {
    size_t state_dim;            /**< 状态维度(四元数数量) */
    int use_attention;           /**< 是否使用注意力加权(1)或简单平均(0)，默认1 */
    float temperature;           /**< 注意力softmax温度，默认1.0 */
} LiquidAggregationConfig;

/**
 * @brief 四元数液态状态聚合器
 */
typedef struct {
    LiquidAggregationConfig config; /**< 配置 */
    float* query_proj;           /**< 查询投影权重[state_dim][4] */
    int is_initialized;          /**< 是否已初始化 */
} QuaternionLiquidAggregation;

/**
 * @brief 创建四元数液态状态聚合器
 *
 * @param config 配置参数
 * @return QuaternionLiquidAggregation* 成功返回指针，失败返回NULL
 */
QuaternionLiquidAggregation* quaternion_liquid_aggregation_create(const LiquidAggregationConfig* config);

/**
 * @brief 销毁四元数液态状态聚合器
 *
 * @param agg 聚合器指针
 */
void quaternion_liquid_aggregation_destroy(QuaternionLiquidAggregation* agg);

/**
 * @brief 前向聚合多个四元数状态
 *
 * 简单平均：Q_agg = (1/N) * Σ Q_i
 * 注意力加权：w_i = softmax(Q_i · Q_query / T), Q_agg = Σ w_i * Q_i
 *
 * @param agg 聚合器指针
 * @param states 多状态输入 [num_states][state_dim]
 * @param num_states 状态数量
 * @param output 聚合输出 [state_dim]
 * @return int 成功返回0，失败返回-1
 */
int quaternion_liquid_aggregation_forward(QuaternionLiquidAggregation* agg,
                                           const Quaternion* states,
                                           size_t num_states,
                                           Quaternion* output);

/**
 * @brief 更新查询投影权重（训练时使用）
 *
 * @param agg 聚合器指针
 * @param grad 梯度数组 [state_dim][4]
 * @param lr 学习率
 * @return int 成功返回0，失败返回-1
 */
int quaternion_liquid_aggregation_update(QuaternionLiquidAggregation* agg,
                                          const float* grad, float lr);

/* ===========================================================================
 * A04.5.2 CfC隐式状态稀疏激活
 *
 * 对CfC门控信号施加稀疏约束，仅保留最重要的门控维度。
 * 支持top-k稀疏化和阈值稀疏化两种模式。
 * =========================================================================== */

/**
 * @brief CfC稀疏激活配置
 */
typedef struct {
    float sparsity_ratio;        /**< 稀疏比例[0,1]，保留比例，默认0.3 */
    int use_topk;                /**< 使用top-k(1)或阈值(0)模式，默认1 */
    float threshold;             /**< 固定阈值(use_topk=0时使用)，默认0.5 */
} CfCSparseConfig;

/**
 * @brief 对四元数门控信号施加CfC稀疏激活
 *
 * top-k模式：按四元数幅值排序，保留前sparsity_ratio比例
 * 阈值模式：幅值低于threshold的置零
 * 稀疏化后对门控信号进行重缩放以保持能量
 *
 * @param gate_signal 门控信号序列 [seq_len][quaternion_dim]，原地修改
 * @param seq_len 序列长度
 * @param quaternion_dim 四元数维度
 * @param config 稀疏配置
 * @return int 成功返回0，失败返回-1
 */
int quaternion_cfc_sparse_activate(Quaternion* gate_signal,
                                    size_t seq_len, size_t quaternion_dim,
                                    const CfCSparseConfig* config);

/* ===========================================================================
 * A04.5.2 CfC线性时间演化
 *
 * 基于ODE的连续时间状态演化，使用欧拉法或RK4进行数值积分。
 * 时间复杂度O(n)，无需矩阵求逆。
 * =========================================================================== */

/**
 * @brief CfC线性时间演化配置
 */
typedef struct {
    float dt;                    /**< 时间步长，默认0.01 */
    int use_rk4;                 /**< 使用RK4(1)或欧拉法(0)，默认0 */
} CfCLinearTimeConfig;

/**
 * @brief 四元数CfC线性时间演化
 *
 * 执行一步连续时间演化：
 *   欧拉法：state += dt * deriv
 *   RK4：  k1=dt*f(state), k2=dt*f(state+k1/2), k3=dt*f(state+k2/2), k4=dt*f(state+k3)
 *          state += (k1+2*k2+2*k3+k4)/6
 *
 * @param state 当前四元数状态 [dim]，原地更新
 * @param deriv 状态导数 [dim](欧拉法) 或 NULL(使用RK4时需要状态函数)
 * @param dim 状态维度
 * @param dt 时间步长
 * @return int 成功返回0，失败返回-1
 */
int quaternion_cfc_linear_time_evolve(Quaternion* state,
                                       const Quaternion* deriv,
                                       size_t dim, float dt);

/* ===========================================================================
 * A04.5.2 CfC分块状态演化
 *
 * 将四元数状态分为独立块，每块独立进行连续时间演化。
 * 块间解耦允许并行计算，降低全状态耦合复杂度。
 * =========================================================================== */

/**
 * @brief CfC分块状态演化配置
 */
typedef struct {
    size_t block_size;           /**< 每块包含的四元数数量，默认4 */
    int use_independent;         /**< 块间是否完全独立(1)或弱耦合(0)，默认1 */
} CfCBlockEvolveConfig;

/**
 * @brief 四元数CfC分块状态演化
 *
 * 将state分为block_size大小的块，对每块调用block_func进行处理。
 * 块函数签名：void func(Quaternion* block, size_t block_size, void* ctx)
 * 其中ctx为用户提供的上下文指针。
 *
 * @param state 四元数状态 [state_dim]，原地更新
 * @param state_dim 状态维度
 * @param config 分块配置
 * @param block_func 块处理函数指针
 * @param ctx 用户上下文(传递给block_func)
 * @return int 成功返回0，失败返回-1
 */
int quaternion_cfc_block_evolve(Quaternion* state,
                                 size_t state_dim,
                                 const CfCBlockEvolveConfig* config,
                                 void (*block_func)(Quaternion* block, size_t block_size, void* ctx),
                                 void* ctx);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_QUATERNION_LIQUID_GATE_H
