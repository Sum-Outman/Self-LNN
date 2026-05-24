/**
 * @file cfc_cell.c
 * @brief 封闭形式连续时间单元（CfC Cell）实现
 * 
 * CfC（Closed-form Continuous-time）单元实现，基于微分方程的封闭形式解。
 * 提供高效的连续时间动态和稳定的梯度传播。
 * 
 * 注意：CfCCell和CfCState结构体通过互补条件编译定义：
 * - 本文件 (#ifndef SELFLNN_CORE_INTERNAL) 定义自身可见版本
 * - cfc_cell.h (#ifdef SELFLNN_CORE_INTERNAL) 定义供其他内部模块
 * 两者完全一致，保证不同编译单元的一致性。
 */

#include "selflnn/core/cfc_cell.h"
#include "selflnn/core/cfc_enhanced.h"
#include "selflnn/core/ode_solvers.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/quaternion_lnn.h"
#include "selflnn/core/quaternion_cfc.h"
#include "selflnn/core/lnn_layer_norm.h"
#include "selflnn/core/common.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * @brief CfC单元状态缓冲区安全检查宏（用于返回int的函数）
 * 在关键函数入口处验证所有必要的状态缓冲区已分配，防止空指针访问
 */
#define CHECK_CFC_STATE_INT(cell) \
    do { \
        if (!(cell)->state || !(cell)->state->activation || \
            !(cell)->state->input_buffer || !(cell)->state->noise_buffer || \
            !(cell)->state->gradient || !(cell)->state->workspace) { \
            selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__, \
                                  "CfC单元状态缓冲区未初始化"); \
            return -1; \
        } \
    } while(0)

/**
 * @brief CfC单元状态缓冲区安全检查宏（用于返回void的函数）
 * 当状态缓冲区为空时直接返回，防止空指针崩溃
 */
#define CHECK_CFC_STATE_VOID(cell) \
    do { \
        if (!(cell)->state || !(cell)->state->activation || \
            !(cell)->state->input_buffer || !(cell)->state->noise_buffer) { \
            return; \
        } \
    } while(0)

/**
 * @brief 生成确定性均匀分布数
 * @param min 最小值
 * @param max 最大值
 * @param seed 种子值（保留接口兼容性，实际使用密码学安全随机数）
 */
static float random_uniform_seeded(float min, float max, unsigned int seed) {
    /* 使用密码学安全随机数替代LCG，消除弱随机性安全隐患 */
    (void)seed;
    return min + (max - min) * secure_random_float();
}

/* ========================================================================
 * P0-001: Xavier/Glorot 和 Kaiming/He 权重初始化辅助函数
 * 解决深层网络训练收敛性能瓶颈
 * ======================================================================== */

/**
 * @brief Xavier/Glorot均匀分布初始化缩放因子
 * Glorot & Bengio (2010): 适用于sigmoid/tanh激活函数
 * 公式: limit = sqrt(6.0 / (fan_in + fan_out))
 * 保证前向传播时各层激活值方差 ≈ 1/fan_out，反向传播时梯度方差 ≈ 1/fan_in
 */
static float xavier_uniform_limit(size_t fan_in, size_t fan_out) {
    if (fan_in == 0 || fan_out == 0) return 0.1f;
    return sqrtf(6.0f / (float)(fan_in + fan_out));
}

/**
 * @brief Xavier/Glorot正态分布初始化标准差
 * 公式: std = sqrt(2.0 / (fan_in + fan_out))
 */
static float xavier_normal_std(size_t fan_in, size_t fan_out) {
    if (fan_in == 0 || fan_out == 0) return 0.1f;
    return sqrtf(2.0f / (float)(fan_in + fan_out));
}

/**
 * @brief Kaiming/He均匀分布初始化缩放因子
 * He et al. (2015): 适用于ReLU/PReLU激活函数
 * 公式: limit = sqrt(6.0 / fan_in)
 * 补偿ReLU导致的方差减半（负半轴输出为0）
 */
static float kaiming_uniform_limit(size_t fan_in) {
    if (fan_in == 0) return 0.1f;
    return sqrtf(6.0f / (float)fan_in);
}

/**
 * @brief 计算权重初始化缩放因子
 * 优先级: 自定义scale > Kaiming > Xavier > 默认值
 *
 * @param use_xavier 是否使用Xavier初始化
 * @param use_kaiming 是否使用Kaiming初始化
 * @param custom_scale 自定义缩放因子（0.0=自动）
 * @param fan_in 输入维度
 * @param fan_out 输出维度
 * @param default_range 默认均匀分布范围（如0.1）
 * @return 均匀分布边界值
 */
static float compute_init_limit(int use_xavier, int use_kaiming,
                                 float custom_scale,
                                 size_t fan_in, size_t fan_out,
                                 float default_range) {
    if (custom_scale > 0.0f) {
        return custom_scale;
    }
    if (use_kaiming) {
        return kaiming_uniform_limit(fan_in);
    }
    if (use_xavier) {
        return xavier_uniform_limit(fan_in, fan_out);
    }
    return default_range;
}

/**
 * @brief CfC单元内部状态
 */
typedef struct {
    float* state;           /**< 单元状态向量 */
    float* adapted_params;  /**< 自适应参数 */
    float* noise_buffer;    /**< 噪声缓冲区 */
    float* activation;      /**< 激活值缓冲区 */
    float* gradient;        /**< 梯度缓冲区 */
    float* input_buffer;    /**< 输入缓冲区 */
    float* workspace;       /**< 工作空间缓冲区（RK4 k4存储等） */
    float* saved_state;     /**< 前向传播前保存的状态 h(t)，反向传播使用 */
    float time;             /**< 当前时间 */
    float adaptation_rate;  /**< 自适应率 */
    /* RK45自适应求解器工作空间 */
    float* rk45_k1;         /**< RK45 k1斜率 */
    float* rk45_k2;         /**< RK45 k2斜率 */
    float* rk45_k3;         /**< RK45 k3斜率 */
    float* rk45_k4;         /**< RK45 k4斜率 */
    float* rk45_k5;         /**< RK45 k5斜率 */
    float* rk45_k6;         /**< RK45 k6斜率 */
    float* rk45_temp;       /**< RK45临时状态 */
    float* rk45_error;      /**< RK45误差估计 */
    /* CTBP连续时间反向传播工作空间 */
    float* ctbp_adjoint;    /**< CTBP伴随状态 */
    float* ctbp_temp;       /**< CTBP临时状态 */
    float* ctbp_state_trajectory; /**< CTBP状态轨迹缓存 */
    int ctbp_trajectory_size;     /**< CTBP轨迹缓存大小 */
    int ctbp_trajectory_capacity; /**< CTBP轨迹缓存容量 */
    /* DP54自适应步长求解器工作空间 */
    float* dp54_k7;         /**< DP54 k7斜率（第7级函数求值） */
    float* dp54_workspace;  /**< DP54工作空间 [8*hidden_size] */
    int dp54_current_steps; /**< DP54当前步数计数 */
    /* Rosenbrock刚性求解器工作空间 */
    float* rosenbrock_jacobian;   /**< Rosenbrock雅可比矩阵 [hidden_size×hidden_size] */
    float* rosenbrock_k1;         /**< Rosenbrock k1斜率 */
    float* rosenbrock_k2;         /**< Rosenbrock k2斜率 */
    float* rosenbrock_k3;         /**< Rosenbrock k3斜率 */
    float* rosenbrock_workspace;  /**< Rosenbrock工作空间 */
    int rosenbrock_current_steps; /**< Rosenbrock当前步数计数 */
    /* Forest-Ruth辛积分器工作空间 */
    float* symplectic_momentum;   /**< 辛积分器动量状态 [hidden_size]（扩展相空间） */
    float* symplectic_dqdt;       /**< 辛积分器位置导数 [hidden_size] */
    float* symplectic_dpdt;       /**< 辛积分器动量导数 [hidden_size] */
    int symplectic_initialized;
    int symplectic_current_steps;   /**< 辛积分器动量是否已初始化 */
} CfCState;

/**
 * @brief CfC单元内部结构体
 */
#ifndef SELFLNN_CORE_INTERNAL
struct CfCCell {
    CfCCellConfig config;       /**< 单元配置 */
    CfCState* state;        /**< 单元状态 */
    float* weight_matrix;   /**< 权重矩阵 */
    float* bias_vector;     /**< 偏置向量 */
    float* weight_grad;     /**< 权重梯度 */
    float* bias_grad;       /**< 偏置梯度 */
    int is_initialized;     /**< 是否已初始化 */
    float avg_activation;   /**< 平均激活度 */
    float max_activation;   /**< 最大激活度 */
    /* 液态神经网络增强特性 */
    float* time_constants;  /**< 每个神经元的时间常数数组 */
    float* input_gate_weights;  /**< 输入门权重矩阵 */
    float* forget_gate_weights; /**< 遗忘门权重矩阵 */
    float* output_gate_weights; /**< 输出门权重矩阵 */
    float* gate_biases;         /**< 门控偏置向量 */
    int use_gating;             /**< 是否使用门控机制 */
    int use_adaptive_tau;       /**< 是否使用自适应时间常数 */
    float min_time_constant;    /**< 最小时间常数 */
    float max_time_constant;    /**< 最大时间常数 */
    float* forward_tau_used;    /**< P1-019修复: 前向传播实际使用的tau值 [hidden_size]，反向传播直接读取避免不一致 */
    /* 梯度缓冲区（修复缺失部分） */
    float* input_gate_weight_grad;  /**< 输入门权重梯度 */
    float* output_gate_weight_grad; /**< 输出门权重梯度 */
    float* forget_gate_weight_grad; /**< 遗忘门权重梯度 */
    float* gate_bias_grad;          /**< 门控偏置梯度 */
    float* time_constant_grad;      /**< 时间常数梯度 */
    float tau_learning_rate;        /**< 时间常数学习率，默认0.001 */
    /* CfC隐藏到隐藏连接权重矩阵（CfC论文标准实现）
     * W_gh [hidden_size x hidden_size]: 隐藏状态到门控的权重
     * W_ah [hidden_size x hidden_size]: 隐藏状态到激活的权重
     * 方程: τ dh/dt = -h + σ(W_gx*x + W_gh*h + b_g) ⊙ f(W_ax*x + W_ah*h + b_a)
     */
    float* hidden_to_gate_weights;         /**< W_gh: 隐藏到门控权重矩阵 [hidden_size×hidden_size] */
    float* hidden_to_activation_weights;   /**< W_ah: 隐藏到激活权重矩阵 [hidden_size×hidden_size] */
    float* hidden_to_gate_weight_grad;     /**< W_gh梯度缓冲区 */
    float* hidden_to_activation_weight_grad; /**< W_ah梯度缓冲区 */
    /* 多时间尺度并行演化字段 */
    int use_multi_timescale;           /**< 是否启用多时间尺度并行演化 */
    float* fast_time_constants;        /**< 快速通道时间常数 [hidden_size] */
    float* slow_time_constants;        /**< 慢速通道时间常数 [hidden_size] */
    float* fast_state;                 /**< 快速通道演化状态 [hidden_size] */
    float* slow_state;                 /**< 慢速通道演化状态 [hidden_size] */
    float* fast_activation;            /**< 快速通道激活值缓冲区 [hidden_size] */
    float* slow_activation;            /**< 慢速通道激活值缓冲区 [hidden_size] */
    float fast_tau_ratio;              /**< 快速通道时间常数比例 */
    float slow_tau_ratio;              /**< 慢速通道时间常数比例 */
    float* multi_timescale_workspace;  /**< 多时间尺度工作空间 [hidden_size] */
    /* 液时域缩放（Liquid Time Scaling）字段 */
    int use_liquid_scaling;                  /**< 是否启用液时域缩放 */
    float* liquid_tau_weights;               /**< 液时域输入投影权重 [hidden_size×input_size] */
    float* liquid_tau_bias;                  /**< 液时域偏置 [hidden_size] */
    float* liquid_tau_weight_grad;           /**< 液时域权重梯度 [hidden_size×input_size] */
    float* liquid_tau_bias_grad;             /**< 液时域偏置梯度 [hidden_size] */
    float* computed_liquid_tau;              /**< 当前计算出的液时域时间常数 [hidden_size] */
    float* liquid_tau_workspace;             /**< 液时域工作空间 [hidden_size] */
    float liquid_tau_min;                    /**< 液时域时间常数最小值 */
    float liquid_tau_max;                    /**< 液时域时间常数最大值 */
    float liquid_init_scale;                 /**< 液时域权重初始化缩放因子 */
    int liquid_use_layer_norm;               /**< 是否对时间常数投影进行层归一化 */
    /* 神经ODE伴随法（Neural ODE Adjoint）字段 */
    int use_neural_ode_adjoint;              /**< 是否启用神经ODE伴随法 */
    int adjoint_record_trajectory;           /**< 是否记录完整轨迹 */
    int adjoint_max_trajectory_points;       /**< 最大轨迹记录点数 */
    float adjoint_gradient_clip_norm;        /**< 梯度裁剪范数 */
    int adjoint_use_augmented_state;         /**< 是否使用增广状态 */
    int adjoint_interpolation_method;        /**< 轨迹插值方法 */
    float* adjoint_state;                    /**< 伴随状态 a(t) [hidden_size] */
    float* adjoint_trajectory;               /**< 前向轨迹缓存 [hidden_size × max_points] */
    float* adjoint_timestamps;               /**< 轨迹时间戳 [max_points] */
    int adjoint_trajectory_capacity;         /**< 轨迹缓存容量（点数） */
    int adjoint_trajectory_count;            /**< 当前轨迹点数 */
    float* adjoint_workspace;                /**< 伴随计算工作空间 [hidden_size] */
    int adjoint_forward_step;                /**< 前向传播步数计数器 */
    float* adjoint_param_grad_workspace;     /**< 参数梯度工作空间 */
    /* 统一自适应步长选择运行时状态 */
    int use_adaptive_step;                   /**< 是否启用自适应步长选择（镜像配置） */
    AdaptiveStepSolution adaptive_step_sol;  /**< 自适应步长运行时状态 */
    /* 并行化ODE求解运行时状态 */
    int use_parallel_solve;                  /**< 是否启用并行化ODE求解（镜像配置） */
    /* 自动求解器选择运行时状态（P3.4） */
    CfcEnhancedState* enhanced_state;        /**< 自动求解器运行时状态 */
    CfcEnhancedConfig enhanced_config;       /**< 自动求解器配置镜像 */
    /* 门控CfC变体数据 */
    GatedCfCData* gated_data;                /**< 门控CfC内部数据 */
    /* 分层CfC数据 */
    HierarchicalCfCData* hierarchical_data;  /**< 分层CfC内部数据 */
    /* 液态记忆门控CfC数据 */
    LiquidMemoryCfCData* liquid_memory_data;  /**< 液态记忆门控CfC内部数据 */
    /* 四元数CfC数据（P2.4） */
    int use_quaternion;                      /**< 是否启用四元数模式 */
    float* quaternion_weights;                /**< 四元数权重矩阵 [num_hidden_quats × num_input_quats × 4] */
    float* quaternion_biases;                 /**< 四元数偏置 [num_hidden_quats × 4] */
    float* quaternion_weight_grad;            /**< 四元数权重梯度 [num_hidden_quats × num_input_quats × 4] */
    float* quaternion_bias_grad;              /**< 四元数偏置梯度 [num_hidden_quats × 4] */
    float* quaternion_hidden_weights;         /**< 四元数隐藏到隐藏权重 [num_hidden_quats × num_hidden_quats × 4] */
    float* quaternion_hidden_weight_grad;     /**< 四元数隐藏到隐藏权重梯度 */
    float* quaternion_time_constants;         /**< 每个隐藏四元数的时间常数 [num_hidden_quats] */
    float* quaternion_workspace;              /**< 四元数工作空间 [hidden_size] */
    /* P1-001: 层归一化模块 */
    LayerNorm* cell_layer_norm;              /**< 层归一化实例（对隐藏状态进行逐样本归一化） */
};

/* ============ 门控CfC变体内部数据结构 ============ */

/** @brief 门控CfC变体内部数据 */
struct GatedCfCData {
    int use_cross_gating;               /**< 是否使用交叉门控 */
    int use_pyramidal_gating;           /**< 是否使用金字塔门控 */
    int use_adaptive_bandwidth;         /**< 是否使用自适应带宽 */
    float bandwidth_min;                /**< 门控带宽最小值 */
    float bandwidth_max;                /**< 门控带宽最大值 */
    int pyramidal_layers;               /**< 金字塔门控层数 */
    float* cross_gate_weights;          /**< 交叉门控权重 [hidden_size × hidden_size] */
    float* cross_gate_grad;             /**< 交叉门控权重梯度 */
    float* cross_gate_state;            /**< 交叉门控当前状态 [hidden_size] */
    float* pyramidal_gate_weights;      /**< 金字塔门控权重 [pyramidal_layers × hidden_size] */
    float* pyramidal_gate_biases;       /**< 金字塔门控偏置 [pyramidal_layers] */
    float* pyramidal_gate_weight_grad;  /**< 金字塔门控权重梯度 */
    float* pyramidal_gate_bias_grad;    /**< 金字塔门控偏置梯度 */
    float* pyramidal_gate_output;       /**< 金字塔门控输出 [hidden_size] */
    float* bandwidth_params;            /**< 自适应带宽参数 [hidden_size × 2]（均值+标准差） */
    float* bandwidth_grad;              /**< 自适应带宽参数梯度 */
    float* workspace;                   /**< 工作空间 [hidden_size] */
};

/* ============ 分层CfC内部数据结构 ============ */

/** @brief 分层CfC内部数据 */
struct HierarchicalCfCData {
    int num_levels;                     /**< 层次数量 */
    float bottom_up_strength;           /**< 自底向上信息流强度 */
    float top_down_strength;            /**< 自顶向下信息流强度 */
    int enable_bidirectional;           /**< 是否启用双向信息流 */
    float* level_time_constants;        /**< 各层次时间常数 [num_levels] */
    float* level_states;                /**< 各层次状态 [num_levels × hidden_size] */
    float* level_activations;           /**< 各层次激活值 [num_levels × hidden_size] */
    float* level_gradients;             /**< 各层次梯度 [num_levels × hidden_size] */
    float* bottom_up_weights;           /**< 自底向上连接权重 [num_levels-1 × hidden_size] */
    float* bottom_up_weight_grad;       /**< 自底向上权重梯度 */
    float* top_down_weights;            /**< 自顶向下连接权重 [num_levels-1 × hidden_size] */
    float* top_down_weight_grad;        /**< 自顶向下权重梯度 */
    float* workspace;                   /**< 工作空间 [hidden_size × 2] */
};

/* ============ 液态记忆门控CfC内部数据结构 ============ */

/** @brief 液态记忆门控CfC内部数据 */
struct LiquidMemoryCfCData {
    int enable_memory_gating;            /**< 是否启用液态记忆门控 */
    float init_scale;                    /**< 权重初始化缩放 */
    float gate_bias_init;                /**< 门控偏置初始化值 */
    float gate_scale;                    /**< 门控缩放因子 */
    size_t concat_dim;                   /**< 拼接维度 = hidden_size + input_size */
    float* gate_weights;                 /**< 门控权重 [hidden_size × (hidden_size + input_size)] */
    float* activation_weights;           /**< 激活权重 [hidden_size × (hidden_size + input_size)] */
    float* gate_biases;                  /**< 门控偏置 [hidden_size] */
    float* activation_biases;            /**< 激活偏置 [hidden_size] */
    float* gate_weights_grad;            /**< 门控权重梯度 */
    float* activation_weights_grad;      /**< 激活权重梯度 */
    float* gate_bias_grad;               /**< 门控偏置梯度 */
    float* activation_bias_grad;         /**< 激活偏置梯度 */
    float* modulation_output;            /**< 调制信号输出 [hidden_size] */
    float* gate_signal;                  /**< 门控信号 [hidden_size] */
    float* concat_buffer;                /**< 拼接缓冲区 [(hidden_size + input_size)] */
    float* workspace;                    /**< 工作空间 [hidden_size] */
};

/* ============ 内部函数前向声明 ============ */

static int gated_cfc_cross_gate_update(GatedCfCData* gd, const float* gate_input,
                                        const float* activation, float* gate_output,
                                        size_t hidden_size);
static int gated_cfc_pyramidal_gate(GatedCfCData* gd, float* gate_signal, size_t hidden_size);
static int gated_cfc_adaptive_bandwidth(GatedCfCData* gd, const float* input,
                                         float* bandwidth, size_t input_size, size_t hidden_size);
static int hierarchical_level_update(HierarchicalCfCData* hd, int level,
                                      const float* input, float* merged_state,
                                      size_t hidden_size, size_t input_size, float delta_t);
static int liquid_memory_compute_modulation(LiquidMemoryCfCData* ld, const float* hidden_state,
                                             const float* input, size_t hidden_size,
                                             size_t input_size);

static int cfc_cell_dp54_step(CfCCell* cell, const float* input,
                               const float* prev_state, float delta_t,
                               float* output);
static int cfc_cell_rosenbrock_step(CfCCell* cell, const float* input,
                                     const float* prev_state, float delta_t,
                                     float* output);
static int cfc_cell_symplectic_step(CfCCell* cell, const float* input,
                                     const float* prev_state, float delta_t,
                                     float* output);
static int cfc_cell_rhs_wrapper(float t, const float* y, float* dydt, void* ctx);
static int cfc_symplectic_dqdt_wrapper(float t, const float* p, float* dqdt_out, void* ctx);
static int cfc_cell_rk45_adaptive(CfCCell* cell, const float* input,
                                    const float* prev_state, float delta_t,
                                    float* output,
                                    const AdaptiveStepConfig* acfg,
                                    AdaptiveStepSolution* asol);
static int cfc_cell_dp54_adaptive(CfCCell* cell, const float* input,
                                    const float* prev_state, float delta_t,
                                    float* output,
                                    const AdaptiveStepConfig* acfg,
                                    AdaptiveStepSolution* asol);
static int cfc_cell_rosenbrock_parallel(CfCCell* cell, const float* input,
                                         const float* prev_state, float delta_t,
                                         float* output,
                                         const ParallelODERHSConfig* pcfg);

/**
 * @brief 创建CfC单元实例
 */
CfCCell* cfc_cell_create(const CfCCellConfig* config) {
    if (!config) {
        return NULL;
    }
    
    // 验证配置
    if (config->input_size == 0 || config->hidden_size == 0) {
        return NULL;
    }
    
    // 分配单元结构
    CfCCell* cell = (CfCCell*)safe_malloc(sizeof(CfCCell));
    if (!cell) {
        return NULL;
    }
    
    // 复制配置
    memcpy(&cell->config, config, sizeof(CfCCellConfig));
    
    /* P0-001: 权重初始化方法默认值——Xavier初始化默认启用，显著提升深层网络收敛性能 */
    if (cell->config.use_xavier_init == 0 && cell->config.use_kaiming_init == 0
        && cell->config.weight_init_scale <= 0.0f) {
        cell->config.use_xavier_init = 1;  /* 默认启用Xavier初始化 */
    }
    
    /* P1-001: 层归一化默认配置——默认启用，提升深层网络训练稳定性 */
    if (cell->config.layer_norm_epsilon <= 0.0f) {
        cell->config.layer_norm_epsilon = 1e-5f;
    }
    
    /* P2-002: 残差连接默认配置——默认启用，为深层网络提供梯度高速公路 */
    if (cell->config.residual_scale <= 0.0f) {
        cell->config.residual_scale = 0.3f;  /* 默认残差缩放因子 */
    }
    if (cell->config.residual_scale < 0.01f) cell->config.residual_scale = 0.0f;
    if (cell->config.residual_scale > 1.0f) cell->config.residual_scale = 1.0f;
    
    // 设置delta_t默认值（如果未设置或无效），并跟踪显式设置标志
    if (cell->config.delta_t <= 0.0f) {
        cell->config.delta_t = 1.0f;  // 默认时间步长，标志位保持0（未显式设置）
    } else {
        cell->config.delta_t_explicitly_set = 1;  // P2-042修复: 调用者显式设置了非零delta_t
    }
    
    // 分配状态
    cell->state = (CfCState*)safe_malloc(sizeof(CfCState));
    if (!cell->state) {
        safe_free((void**)&cell);
        return NULL;
    }
    
    size_t hidden_size = config->hidden_size;
    size_t input_size = config->input_size;
    
    // 分配状态向量
    cell->state->state = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->state->adapted_params = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->state->noise_buffer = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->state->activation = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->state->gradient = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->state->input_buffer = (float*)safe_calloc(input_size, sizeof(float));
    cell->state->workspace = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->state->saved_state = (float*)safe_calloc(hidden_size, sizeof(float));
    
    // 分配权重和偏置
    size_t weight_size = input_size * hidden_size;
    cell->weight_matrix = (float*)safe_calloc(weight_size, sizeof(float));
    cell->bias_vector = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->weight_grad = (float*)safe_calloc(weight_size, sizeof(float));
    cell->bias_grad = (float*)safe_calloc(hidden_size, sizeof(float));
    
    // 分配液态神经网络增强特性内存
    cell->time_constants = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->forward_tau_used = (float*)safe_calloc(hidden_size, sizeof(float));  /* P1-019修复: 前向tau快照 */
    cell->input_gate_weights = (float*)safe_calloc(weight_size, sizeof(float));
    cell->forget_gate_weights = (float*)safe_calloc(weight_size, sizeof(float));
    cell->output_gate_weights = (float*)safe_calloc(weight_size, sizeof(float));
    cell->gate_biases = (float*)safe_calloc(hidden_size * 3, sizeof(float)); // 输入、遗忘、输出门偏置
    
    // 分配CfC隐藏到隐藏连接权重矩阵（CfC论文标准实现）
    size_t hidden_weight_size = hidden_size * hidden_size;
    cell->hidden_to_gate_weights = (float*)safe_calloc(hidden_weight_size, sizeof(float));
    cell->hidden_to_activation_weights = (float*)safe_calloc(hidden_weight_size, sizeof(float));
    
    // 分配梯度缓冲区（修复缺失部分）
    cell->input_gate_weight_grad = (float*)safe_calloc(weight_size, sizeof(float));
    cell->output_gate_weight_grad = (float*)safe_calloc(weight_size, sizeof(float));
    cell->forget_gate_weight_grad = (float*)safe_calloc(weight_size, sizeof(float));
    cell->gate_bias_grad = (float*)safe_calloc(hidden_size * 3, sizeof(float));
    cell->time_constant_grad = (float*)safe_calloc(hidden_size, sizeof(float));
    
    // 分配隐藏到隐藏权重梯度缓冲区
    cell->hidden_to_gate_weight_grad = (float*)safe_calloc(hidden_weight_size, sizeof(float));
    cell->hidden_to_activation_weight_grad = (float*)safe_calloc(hidden_weight_size, sizeof(float));
    
    // 分配多时间尺度并行演化缓冲区
    cell->fast_time_constants = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->slow_time_constants = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->fast_state = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->slow_state = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->fast_activation = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->slow_activation = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->multi_timescale_workspace = (float*)safe_calloc(hidden_size, sizeof(float));
    
    // 分配液时域缩放（Liquid Time Scaling）缓冲区
    size_t tau_weight_size = hidden_size * input_size;
    cell->liquid_tau_weights = (float*)safe_calloc(tau_weight_size, sizeof(float));
    cell->liquid_tau_bias = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->liquid_tau_weight_grad = (float*)safe_calloc(tau_weight_size, sizeof(float));
    cell->liquid_tau_bias_grad = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->computed_liquid_tau = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->liquid_tau_workspace = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->use_liquid_scaling = 1;  /* 默认启用液时域缩放（LNN核心特性） */
    cell->liquid_tau_min = 0.01f;
    cell->liquid_tau_max = 2.0f;
    cell->liquid_init_scale = 0.01f;
    cell->liquid_use_layer_norm = 0;
    
    // 使用默认时间常数初始化（均分在[min, max]区间）
    for (size_t i = 0; i < hidden_size; i++) {
        cell->computed_liquid_tau[i] = cell->config.time_constant > 0.0f ?
            cell->config.time_constant : (cell->liquid_tau_min + cell->liquid_tau_max) * 0.5f;
    }
    
    /* ODE求解器工作空间按需分配（默认闭合解无需工作空间）
     * 各求解器工作空间大小：
     *   RK45:    8*hidden_size
     *   CTBP:    2*hidden_size + 100*hidden_size = 102*hidden_size
     *   DP54:    9*hidden_size (含k7 + 8*hidden主空间)
     *   Rosenbrock: 2*hidden_size² + 9*hidden_size
     *   Symplectic:  3*hidden_size
     * 最大需求 = Rosenbrock的 ~2*hidden_size² 当hidden_size>=128
     * 仅在实际使用时分配，默认闭合解不分配任何工作空间 */
    cell->state->ctbp_trajectory_size = 0;
    cell->state->ctbp_trajectory_capacity = 100;
    cell->state->dp54_current_steps = 0;
    cell->state->rosenbrock_current_steps = 0;
    cell->state->symplectic_current_steps = 0;
    cell->state->symplectic_initialized = 0;
    
    // 初始化新求解器的默认配置
    cell->config.dp54_config.rel_tolerance = 1e-7f;
    cell->config.dp54_config.abs_tolerance = 1e-9f;
    cell->config.dp54_config.min_step_size = 1e-8f;
    cell->config.dp54_config.max_step_size = 1.0f;
    cell->config.dp54_config.safety_factor = 0.9f;
    cell->config.dp54_config.max_iterations = 10000;
    
    cell->config.rosenbrock_config.rel_tolerance = 1e-5f;
    cell->config.rosenbrock_config.abs_tolerance = 1e-7f;
    cell->config.rosenbrock_config.min_step_size = 1e-10f;
    cell->config.rosenbrock_config.max_step_size = 1.0f;
    cell->config.rosenbrock_config.gamma_coeff = 0.435866521508f;
    cell->config.rosenbrock_config.max_iterations = 10000;
    cell->config.rosenbrock_config.finite_diff_eps = 1e-6f;
    
    cell->config.symplectic_config.substep_ratio = 0.5f;
    cell->config.symplectic_config.num_substeps = 1;

    /* 四元数CfC初始化（P2.4）
     * ZSFWS-021修复: 消除goto跳转，改用if-else结构。
     * 策略：先将所有四元数指针预置为NULL，仅当维度满足4的倍数条件时才分配实际内存。
     * 这样维度不符合条件时指针已为NULL，无需goto跳过置NULL的else分支。 */
    cell->use_quaternion = config->use_quaternion;
    cell->quaternion_weights = NULL;
    cell->quaternion_biases = NULL;
    cell->quaternion_weight_grad = NULL;
    cell->quaternion_bias_grad = NULL;
    cell->quaternion_hidden_weights = NULL;
    cell->quaternion_hidden_weight_grad = NULL;
    cell->quaternion_time_constants = NULL;
    cell->quaternion_workspace = NULL;

    if (cell->use_quaternion) {
        size_t num_hidden_quats = hidden_size / 4;
        size_t num_input_quats = input_size / 4;
        /* P2-013修复: 维度非4倍数时回退为禁用四元数模式，而非直接释放 */
        if (num_hidden_quats >= 1 && num_input_quats >= 1 &&
            hidden_size % 4 == 0 && input_size % 4 == 0) {
            /* 分配四元数权重: [num_hidden_quats × num_input_quats × 4] */
            cell->quaternion_weights = (float*)safe_calloc(num_hidden_quats * num_input_quats * 4, sizeof(float));
            cell->quaternion_biases = (float*)safe_calloc(num_hidden_quats * 4, sizeof(float));
            cell->quaternion_weight_grad = (float*)safe_calloc(num_hidden_quats * num_input_quats * 4, sizeof(float));
            cell->quaternion_bias_grad = (float*)safe_calloc(num_hidden_quats * 4, sizeof(float));
            cell->quaternion_hidden_weights = (float*)safe_calloc(num_hidden_quats * num_hidden_quats * 4, sizeof(float));
            cell->quaternion_hidden_weight_grad = (float*)safe_calloc(num_hidden_quats * num_hidden_quats * 4, sizeof(float));
            cell->quaternion_time_constants = (float*)safe_calloc(num_hidden_quats, sizeof(float));
            cell->quaternion_workspace = (float*)safe_calloc(hidden_size, sizeof(float));

            /* P5-001修复: 初始化前检查所有四元数缓冲区分配成功，防止NULL指针写入 */
            if (!cell->quaternion_weights || !cell->quaternion_biases ||
                !cell->quaternion_weight_grad || !cell->quaternion_bias_grad ||
                !cell->quaternion_hidden_weights || !cell->quaternion_hidden_weight_grad ||
                !cell->quaternion_time_constants || !cell->quaternion_workspace) {
                cfc_cell_free(cell);
                return NULL;
            }

            /* P0-001: 四元数权重使用Xavier均匀分布初始化 */
            unsigned int seed = 42;
            float quat_limit = xavier_uniform_limit(num_input_quats * 4, num_hidden_quats * 4);
            float quat_bias_limit = quat_limit * 0.5f;
            for (size_t i = 0; i < num_hidden_quats * num_input_quats * 4; i++) {
                cell->quaternion_weights[i] = random_uniform_seeded(-quat_limit, quat_limit, seed++);
            }
            for (size_t i = 0; i < num_hidden_quats * 4; i++) {
                cell->quaternion_biases[i] = random_uniform_seeded(-quat_bias_limit, quat_bias_limit, seed++);
            }
            for (size_t i = 0; i < num_hidden_quats * num_hidden_quats * 4; i++) {
                cell->quaternion_hidden_weights[i] = random_uniform_seeded(-quat_limit, quat_limit, seed++);
            }
            for (size_t i = 0; i < num_hidden_quats; i++) {
                cell->quaternion_time_constants[i] = config->time_constant > 0.0f ?
                    config->time_constant : 0.5f;
            }
        } else {
            /* 维度不满足四元数要求，回退为禁用模式（指针已预置为NULL，无需额外清理） */
            cell->use_quaternion = 0;
        }
    }

    // 检查内存分配（包括新增的梯度缓冲区）
    if (!cell->state->state || !cell->state->adapted_params ||
        !cell->state->noise_buffer || !cell->state->activation ||
        !cell->state->gradient || !cell->state->input_buffer ||
        !cell->state->workspace || !cell->state->saved_state ||
        !cell->weight_matrix || !cell->bias_vector ||
        !cell->weight_grad || !cell->bias_grad ||
        !cell->time_constants || !cell->input_gate_weights ||
        !cell->forget_gate_weights || !cell->output_gate_weights ||
        !cell->gate_biases ||
        !cell->hidden_to_gate_weights || !cell->hidden_to_activation_weights ||
        !cell->input_gate_weight_grad || !cell->output_gate_weight_grad ||
        !cell->forget_gate_weight_grad || !cell->gate_bias_grad ||
        !cell->time_constant_grad ||
        !cell->hidden_to_gate_weight_grad || !cell->hidden_to_activation_weight_grad ||
        !cell->fast_time_constants || !cell->slow_time_constants ||
        !cell->fast_state || !cell->slow_state ||
        !cell->fast_activation || !cell->slow_activation ||
        !cell->multi_timescale_workspace) {
        cfc_cell_free(cell);
        return NULL;
    }
    /* 四元数缓冲区检查 */
    if (cell->use_quaternion && (!cell->quaternion_weights || !cell->quaternion_biases ||
        !cell->quaternion_weight_grad || !cell->quaternion_bias_grad ||
        !cell->quaternion_hidden_weights || !cell->quaternion_hidden_weight_grad ||
        !cell->quaternion_time_constants || !cell->quaternion_workspace)) {
        cfc_cell_free(cell);
        return NULL;
    }
    
    // 初始化状态
    cell->state->time = 0.0f;
    cell->state->adaptation_rate = 0.01f;  // 默认自适应率
    
    // 初始化液态神经网络增强特性
    cell->use_gating = 1;  // 默认启用门控
    cell->use_adaptive_tau = 1;  // 默认启用自适应时间常数
    cell->min_time_constant = SELFLNN_DEFAULT_TAU_MIN;
    cell->max_time_constant = SELFLNN_DEFAULT_TAU_MAX;
    cell->tau_learning_rate = SELFLNN_DEFAULT_LEARNING_RATE;  // 时间常数学习率默认值
    
    // 初始化状态向量为小随机值
    for (size_t i = 0; i < hidden_size; i++) {
        cell->state->state[i] = random_uniform_seeded(-0.01f, 0.01f, (unsigned int)(uintptr_t)cell ^ (unsigned int)i);
        cell->state->adapted_params[i] = 1.0f;  // 初始自适应参数为1
        
        // 初始化时间常数，每个神经元在配置值附近小幅变化
        /* FIX-006: 原代码用 [min_time_constant(0.1), max_time_constant(10.0)] 
         * 导致100倍的时间常数差异，部分神经元遗忘过快或过慢。
         * 修正为以配置值为中心 [τ*0.5, τ*2.0] 的窄区间，确保异构但可控。 */
        {
            float tc = cell->config.time_constant;
            if (tc < SELFLNN_DEFAULT_DT) tc = 0.1f;
            float tc_min = tc * 0.5f;
            float tc_max = tc * 2.0f;
            if (tc_min < cell->min_time_constant) tc_min = cell->min_time_constant;
            if (tc_max > cell->max_time_constant) tc_max = cell->max_time_constant;
            cell->time_constants[i] = random_uniform_seeded(tc_min, tc_max,
                (unsigned int)(uintptr_t)cell ^ (unsigned int)i ^ 0x1234);
        }
    }
    
    /* P0-001: 使用Xavier/Kaiming自适应权重初始化，替代固定范围的均匀分布
     * 权重组: 激活权重(W_ax), 输入门权重(W_gx), 遗忘门, 输出门
     * fan_in = input_size, fan_out = hidden_size
     * 门控权重使用相同的Xavier缩放但乘以0.5的门控因子（与CfC论文实践一致） */
    float activation_limit = compute_init_limit(
        cell->config.use_xavier_init, cell->config.use_kaiming_init,
        cell->config.weight_init_scale, input_size, hidden_size, 0.1f);
    float gate_limit = compute_init_limit(
        cell->config.use_xavier_init, cell->config.use_kaiming_init,
        cell->config.weight_init_scale * 0.5f, input_size, hidden_size, 0.05f);
    if (cell->config.use_xavier_init || cell->config.use_kaiming_init) {
        gate_limit = activation_limit * 0.5f;  /* 门控权重缩放因子 = 0.5 */
    }

    /* P0-001: 隐藏到隐藏权重Xavier因子
     * fan_in = hidden_size, fan_out = hidden_size
     * Xavier limit = sqrt(6/(2*hidden)) = sqrt(3/hidden) */
    float hh_activation_limit = compute_init_limit(
        cell->config.use_xavier_init, cell->config.use_kaiming_init,
        cell->config.weight_init_scale, hidden_size, hidden_size, 0.05f);
    float hh_gate_limit = hh_activation_limit * 0.5f;
    if (cell->config.weight_init_scale > 0.0f) {
        hh_gate_limit = cell->config.weight_init_scale * 0.5f;
    }

    // 初始化权重为自适应随机值（P0-001改进）
    for (size_t i = 0; i < weight_size; i++) {
        cell->weight_matrix[i] = random_uniform_seeded(-activation_limit, activation_limit,
            (unsigned int)(uintptr_t)cell ^ (unsigned int)i ^ 0x1000);
        // 初始化门控权重（使用门控缩放因子）
        cell->input_gate_weights[i] = random_uniform_seeded(-gate_limit, gate_limit,
            (unsigned int)(uintptr_t)cell ^ (unsigned int)i ^ 0x2000);
        cell->forget_gate_weights[i] = random_uniform_seeded(-gate_limit, gate_limit,
            (unsigned int)(uintptr_t)cell ^ (unsigned int)i ^ 0x3000);
        cell->output_gate_weights[i] = random_uniform_seeded(-gate_limit, gate_limit,
            (unsigned int)(uintptr_t)cell ^ (unsigned int)i ^ 0x4000);
    }
    
    // 初始化隐藏到隐藏连接权重矩阵（CfC标准：W_gh和W_ah）——P0-001使用自适应缩放
    for (size_t i = 0; i < hidden_weight_size; i++) {
        cell->hidden_to_gate_weights[i] = random_uniform_seeded(-hh_gate_limit, hh_gate_limit,
            (unsigned int)(uintptr_t)cell ^ (unsigned int)i ^ 0x5000);
        cell->hidden_to_activation_weights[i] = random_uniform_seeded(-hh_activation_limit, hh_activation_limit,
            (unsigned int)(uintptr_t)cell ^ (unsigned int)i ^ 0x6000);
    }
    
    // 初始化偏置为零
    for (size_t i = 0; i < hidden_size; i++) {
        cell->bias_vector[i] = 0.0f;
        // 初始化门控偏置
        cell->gate_biases[i * 3 + 0] = 0.0f;  // 输入门偏置
        cell->gate_biases[i * 3 + 1] = 1.0f;  // 遗忘门偏置（初始倾向于记住）
        cell->gate_biases[i * 3 + 2] = 0.0f;  // 输出门偏置
    }
    
    // 初始化统计信息
    cell->is_initialized = 1;
    cell->avg_activation = 0.0f;
    cell->max_activation = 0.0f;
    
    // 初始化多时间尺度并行演化（默认启用——LNN核心特性）
    cell->use_multi_timescale = 1;
    cell->config.use_multi_timescale = 1;
    cell->fast_tau_ratio = cell->config.fast_tau_ratio > 0.0f ? cell->config.fast_tau_ratio : 0.1f;
    cell->slow_tau_ratio = cell->config.slow_tau_ratio > 0.0f ? cell->config.slow_tau_ratio : 10.0f;
    
    if (cell->use_multi_timescale) {
        for (size_t i = 0; i < hidden_size; i++) {
            float base_tau = cell->time_constants[i];
            cell->fast_time_constants[i] = base_tau * cell->fast_tau_ratio;
            cell->slow_time_constants[i] = base_tau * cell->slow_tau_ratio;
            if (cell->fast_time_constants[i] < 0.001f) cell->fast_time_constants[i] = 0.001f;
            if (cell->slow_time_constants[i] > 100.0f) cell->slow_time_constants[i] = 100.0f;
            /* ZSFWS-022修复: 多时间尺度状态初始化策略
             * fast_state: 零初始化 —— 快速通道无历史记忆，直接响应瞬时输入。
             *             演化中由快速时间常数 fast_tau ≈ 0.001~0.01 驱动，更新迅速。
             *
             * slow_state: 指数移动平均(EMA)近似初始化
             * EMA递推公式: slow_new = α * state + (1-α) * slow_old
             *   其中 α = 1.0f / slow_tau_ratio ≈ 0.1（慢速时间常数控制EMA平滑度）
             *   慢速通道在演化中由 slow_tau ≈ 1.0~10.0 驱动，更新缓慢，具有惯性效应。
             *
             * 初始化时刻分析:
             *   - cell->state->state 已由 safe_calloc 置零（无先验状态信息）
             *   - 无历史累积时，EMA退化为一阶近似: slow_state = α * state
             *   - 代入 state[i]=0 得 slow_state[i] = 0
             *   - 随着时间序列输入推进，慢速通道将通过EMA公式逐步积累长时依赖
             *
             * 两通道初始值相同(=0)但演化动态不同:
             *   fast_tau << slow_tau ⇒ 快通道跟踪瞬态变化，慢通道积压低频趋势 */ 
            cell->fast_state[i] = 0.0f;
            cell->slow_state[i] = 0.0f;
        }
    }
    
    // 初始化神经ODE伴随法（默认禁用，分配缓冲区后由 enable 配置）
    cell->use_neural_ode_adjoint = 0;
    cell->adjoint_record_trajectory = 1;
    cell->adjoint_max_trajectory_points = 10000;
    cell->adjoint_gradient_clip_norm = 0.0f;
    cell->adjoint_use_augmented_state = 0;
    cell->adjoint_interpolation_method = 1;
    cell->adjoint_state = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->adjoint_trajectory = NULL;
    cell->adjoint_timestamps = NULL;
    cell->adjoint_trajectory_capacity = 0;
    cell->adjoint_trajectory_count = 0;
    cell->adjoint_workspace = (float*)safe_calloc(hidden_size, sizeof(float));
    cell->adjoint_forward_step = 0;
    cell->adjoint_param_grad_workspace = NULL;
    
    /* 初始化统一自适应步长选择 */
    cell->use_adaptive_step = (cell->config.use_adaptive_step != 0) ? 1 : 0;
    if (cell->config.use_adaptive_step) {
        cell->config.adaptive_step_cfg = ode_adaptive_step_default_config();
    }
    ode_adaptive_step_init(&cell->adaptive_step_sol, 0.0f, cell->config.delta_t);
    
    /* 初始化并行化ODE求解（P1-012修复：大隐藏层自动启用并行求解） */
    cell->use_parallel_solve = (cell->config.use_parallel_solve != 0) ? 1 : 0;
    if (!cell->use_parallel_solve && cell->config.hidden_size >= 256) {
        /* 隐藏层≥256时自动启用并行ODE求解以利用多核CPU */
        cell->use_parallel_solve = 1;
        cell->config.use_parallel_solve = 1;
    }
    if (cell->use_parallel_solve) {
        cell->config.parallel_cfg = ode_parallel_default_config();
        cell->config.parallel_cfg.num_domains = 4;
        cell->config.parallel_cfg.use_domain_decomposition = 1;
    }
    
    /* P0-015修复: 初始化自动求解器选择
     * 当use_auto_solver未显式启用且delta_t未被显式设置时，
     * 自动根据隐藏层大小和输入尺寸选择最适合的ODE求解器 */
    cell->enhanced_state = NULL;
    if (cell->config.use_auto_solver == 0 && 
        cell->config.delta_t_explicitly_set == 0) {
        /* 调用者未显式配置求解器参数，启用自动选择 */
        cell->config.use_auto_solver = 1;
        cell->config.use_adaptive_step = 1;
        cell->config.adaptive_step_cfg = ode_adaptive_step_default_config();
        ode_adaptive_step_init(&cell->adaptive_step_sol, 0.0f, cell->config.delta_t);
    }
    cell->config.use_auto_solver = (cell->config.use_auto_solver != 0) ? 1 : 0;
    if (cell->config.use_auto_solver) {
        cell->enhanced_config = cfc_enhanced_default_config();
        cell->enhanced_config.enable_auto_solver = 1;
        cell->enhanced_state = cfc_enhanced_state_create();
        if (!cell->enhanced_state) {
            cfc_cell_free(cell);
            return NULL;
        }
    }
    
    /* P1-001: 创建层归一化实例——对隐藏状态进行逐样本归一化，消除内部协变量偏移 */
    if (cell->config.use_cell_layer_norm) {
        LayerNormConfig ln_config = layer_norm_default_config(hidden_size);
        ln_config.epsilon = cell->config.layer_norm_epsilon;
        ln_config.use_affine = 1;
        cell->cell_layer_norm = layer_norm_create(&ln_config);
        if (!cell->cell_layer_norm) {
            cfc_cell_free(cell);
            return NULL;
        }
    } else {
        cell->cell_layer_norm = NULL;
    }
    
    return cell;
}

/**
 * @brief 释放CfC单元实例
 */
void cfc_cell_free(CfCCell* cell) {
    if (!cell) {
        return;
    }
    
    // 释放状态相关缓冲区（必须在释放cell->state之前）
    if (cell->state) {
        safe_free((void**)&cell->state->state);
        safe_free((void**)&cell->state->adapted_params);
        safe_free((void**)&cell->state->noise_buffer);
        safe_free((void**)&cell->state->activation);
        safe_free((void**)&cell->state->gradient);
        safe_free((void**)&cell->state->input_buffer);
        safe_free((void**)&cell->state->workspace);
        safe_free((void**)&cell->state->saved_state);
        
        // 释放RK45自适应求解器缓冲区
        safe_free((void**)&cell->state->rk45_k1);
        safe_free((void**)&cell->state->rk45_k2);
        safe_free((void**)&cell->state->rk45_k3);
        safe_free((void**)&cell->state->rk45_k4);
        safe_free((void**)&cell->state->rk45_k5);
        safe_free((void**)&cell->state->rk45_k6);
        safe_free((void**)&cell->state->rk45_temp);
        safe_free((void**)&cell->state->rk45_error);
        
        // 释放CTBP连续时间反向传播缓冲区
        safe_free((void**)&cell->state->ctbp_adjoint);
        safe_free((void**)&cell->state->ctbp_temp);
        safe_free((void**)&cell->state->ctbp_state_trajectory);
        
        // 释放DP54自适应步长求解器缓冲区
        safe_free((void**)&cell->state->dp54_k7);
        safe_free((void**)&cell->state->dp54_workspace);
        
        // 释放Rosenbrock刚性求解器缓冲区
        safe_free((void**)&cell->state->rosenbrock_jacobian);
        safe_free((void**)&cell->state->rosenbrock_k1);
        safe_free((void**)&cell->state->rosenbrock_k2);
        safe_free((void**)&cell->state->rosenbrock_k3);
        safe_free((void**)&cell->state->rosenbrock_workspace);
        
        // 释放Forest-Ruth辛积分器缓冲区
        safe_free((void**)&cell->state->symplectic_momentum);
        safe_free((void**)&cell->state->symplectic_dqdt);
        safe_free((void**)&cell->state->symplectic_dpdt);
        
        safe_free((void**)&cell->state);
    }
    
    // 释放权重和偏置
    safe_free((void**)&cell->weight_matrix);
    safe_free((void**)&cell->bias_vector);
    safe_free((void**)&cell->weight_grad);
    safe_free((void**)&cell->bias_grad);
    
    // 释放液态神经网络增强特性内存
    safe_free((void**)&cell->time_constants);
    safe_free((void**)&cell->forward_tau_used);  /* P1-019修复: 前向tau快照 */
    safe_free((void**)&cell->input_gate_weights);
    safe_free((void**)&cell->forget_gate_weights);
    safe_free((void**)&cell->output_gate_weights);
    safe_free((void**)&cell->gate_biases);
    
    // 释放隐藏到隐藏连接权重矩阵
    safe_free((void**)&cell->hidden_to_gate_weights);
    safe_free((void**)&cell->hidden_to_activation_weights);
    
    // 释放梯度缓冲区（修复缺失部分）
    safe_free((void**)&cell->input_gate_weight_grad);
    safe_free((void**)&cell->output_gate_weight_grad);
    safe_free((void**)&cell->forget_gate_weight_grad);
    safe_free((void**)&cell->gate_bias_grad);
    safe_free((void**)&cell->time_constant_grad);
    
    // 释放隐藏到隐藏权重梯度缓冲区
    safe_free((void**)&cell->hidden_to_gate_weight_grad);
    safe_free((void**)&cell->hidden_to_activation_weight_grad);
    
    // 释放多时间尺度并行演化缓冲区
    safe_free((void**)&cell->fast_time_constants);
    safe_free((void**)&cell->slow_time_constants);
    safe_free((void**)&cell->fast_state);
    safe_free((void**)&cell->slow_state);
    safe_free((void**)&cell->fast_activation);
    safe_free((void**)&cell->slow_activation);
    safe_free((void**)&cell->multi_timescale_workspace);
    
    // 释放液时域缩放缓冲区
    safe_free((void**)&cell->liquid_tau_weights);
    safe_free((void**)&cell->liquid_tau_bias);
    safe_free((void**)&cell->liquid_tau_weight_grad);
    safe_free((void**)&cell->liquid_tau_bias_grad);
    safe_free((void**)&cell->computed_liquid_tau);
    safe_free((void**)&cell->liquid_tau_workspace);
    
    // 释放神经ODE伴随法缓冲区
    safe_free((void**)&cell->adjoint_state);
    safe_free((void**)&cell->adjoint_trajectory);
    safe_free((void**)&cell->adjoint_timestamps);
    safe_free((void**)&cell->adjoint_workspace);
    safe_free((void**)&cell->adjoint_param_grad_workspace);
    
    // 释放门控CfC变体数据
    if (cell->gated_data) {
        GatedCfCData* gd = cell->gated_data;
        safe_free((void**)&gd->cross_gate_weights);
        safe_free((void**)&gd->cross_gate_grad);
        safe_free((void**)&gd->cross_gate_state);
        safe_free((void**)&gd->pyramidal_gate_weights);
        safe_free((void**)&gd->pyramidal_gate_biases);
        safe_free((void**)&gd->pyramidal_gate_weight_grad);
        safe_free((void**)&gd->pyramidal_gate_bias_grad);
        safe_free((void**)&gd->pyramidal_gate_output);
        safe_free((void**)&gd->bandwidth_params);
        safe_free((void**)&gd->bandwidth_grad);
        safe_free((void**)&gd->workspace);
        safe_free((void**)&cell->gated_data);
    }
    
    // 释放分层CfC数据
    if (cell->hierarchical_data) {
        HierarchicalCfCData* hd = cell->hierarchical_data;
        safe_free((void**)&hd->level_time_constants);
        safe_free((void**)&hd->level_states);
        safe_free((void**)&hd->level_activations);
        safe_free((void**)&hd->level_gradients);
        safe_free((void**)&hd->bottom_up_weights);
        safe_free((void**)&hd->bottom_up_weight_grad);
        safe_free((void**)&hd->top_down_weights);
        safe_free((void**)&hd->top_down_weight_grad);
        safe_free((void**)&hd->workspace);
        safe_free((void**)&cell->hierarchical_data);
    }
    
    // 释放液态记忆门控CfC数据
    if (cell->liquid_memory_data) {
        LiquidMemoryCfCData* ld = cell->liquid_memory_data;
        safe_free((void**)&ld->gate_weights);
        safe_free((void**)&ld->activation_weights);
        safe_free((void**)&ld->gate_biases);
        safe_free((void**)&ld->activation_biases);
        safe_free((void**)&ld->gate_weights_grad);
        safe_free((void**)&ld->activation_weights_grad);
        safe_free((void**)&ld->gate_bias_grad);
        safe_free((void**)&ld->activation_bias_grad);
        safe_free((void**)&ld->modulation_output);
        safe_free((void**)&ld->gate_signal);
        safe_free((void**)&ld->concat_buffer);
        safe_free((void**)&ld->workspace);
        safe_free((void**)&cell->liquid_memory_data);
    }
    
    // 释放自动求解器选择状态（P3.4）
    if (cell->enhanced_state) {
        cfc_enhanced_state_free(cell->enhanced_state);
        cell->enhanced_state = NULL;
    }

    // 释放四元数CfC缓冲区（P2.4）
    safe_free((void**)&cell->quaternion_weights);
    safe_free((void**)&cell->quaternion_biases);
    safe_free((void**)&cell->quaternion_weight_grad);
    safe_free((void**)&cell->quaternion_bias_grad);
    safe_free((void**)&cell->quaternion_hidden_weights);
    safe_free((void**)&cell->quaternion_hidden_weight_grad);
    safe_free((void**)&cell->quaternion_time_constants);
    safe_free((void**)&cell->quaternion_workspace);

    /* P1-001: 释放层归一化实例 */
    if (cell->cell_layer_norm) {
        layer_norm_free(cell->cell_layer_norm);
        cell->cell_layer_norm = NULL;
    }

    /* P0-BPTT: 释放cell级动量/速度缓冲区 */
    safe_free((void**)&cell->cell_momentum_buffer);
    cell->cell_velocity_buffer = NULL;
    cell->cell_momentum_size = 0;
    cell->cell_momentum_initialized = 0;

    // 释放单元结构
    safe_free((void**)&cell);
}

/**
 * @brief 计算CfC ODE右端项（导数）
 * 
 * ODE系统：τ * dh/dt = -h + σ(W_gx * x + W_gh * h + b_g) ⊙ f(W_ax * x + W_ah * h + b_a)
 * 右端项：dh/dt = (-h + σ(W_gx*x + W_gh*h + b_g) ⊙ f(W_ax*x + W_ah*h + b_a)) / τ
 * 
 * @param cell CfC单元句柄
 * @param input 输入向量
 * @param hidden_state 当前隐藏状态（用于计算RHS的h值）
 * @param rhs_output 右端项输出缓冲区（大小=hidden_size）
 */
static void cfc_cell_compute_rhs(CfCCell* cell, const float* input,
                                 const float* hidden_state, float* rhs_output) {
    if (!cell || !input || !hidden_state || !rhs_output) return;
    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    
    for (size_t i = 0; i < hidden_size; i++) {
        float gate_input_sum = cell->gate_biases[i * 3];
        float activation_input_sum = cell->bias_vector[i];
        
        for (size_t j = 0; j < input_size; j++) {
            size_t idx = i * input_size + j;
            gate_input_sum += cell->input_gate_weights[idx] * input[j];
            activation_input_sum += cell->weight_matrix[idx] * input[j];
        }
        
        for (size_t j = 0; j < hidden_size; j++) {
            size_t h_idx = i * hidden_size + j;
            gate_input_sum += cell->hidden_to_gate_weights[h_idx] * hidden_state[j];
            activation_input_sum += cell->hidden_to_activation_weights[h_idx] * hidden_state[j];
        }
        
        float gate;
        if (gate_input_sum > 10.0f) {
            gate = 1.0f;
        } else if (gate_input_sum < -10.0f) {
            gate = 0.0f;
        } else {
            gate = 1.0f / (1.0f + expf(-gate_input_sum));
        }
        
        float activation;
        if (activation_input_sum > 10.0f) {
            activation = 1.0f;
        } else if (activation_input_sum < -10.0f) {
            activation = -1.0f;
        } else {
            float exp_pos = expf(activation_input_sum);
            float exp_neg = expf(-activation_input_sum);
            activation = (exp_pos - exp_neg) / (exp_pos + exp_neg);
        }
        
        float driver = gate * activation;
        
        float tau = cell->use_adaptive_tau ? cell->time_constants[i] : cell->config.time_constant;
        if (tau < cell->min_time_constant) tau = cell->min_time_constant;
        if (tau > cell->max_time_constant) tau = cell->max_time_constant;
        
        /* P1-019修复: 记录前向RHS使用的tau（数值求解器通过compute_rhs获取tau） */
        cell->forward_tau_used[i] = tau;
        
        rhs_output[i] = (-hidden_state[i] + driver) / tau;
    }
}

/**
 * @brief 多时间尺度并行演化的封闭形式解
 *
 * 同时维护快速（小τ）和慢速（大τ）两个时间尺度的状态演化，
 * 输出为两者的加权混合。这允许同一网络同时处理短时和长时依赖关系。
 *
 * 快速通道：τ_fast = base_tau * fast_ratio（快速响应精细细节）
 * 慢速通道：τ_slow = base_tau * slow_ratio（慢速响应长期趋势）
 *
 * 每个通道独立应用CfC封闭形式解：
 *   h_fast(t+Δt) = h_fast(t)*exp(-Δt/τ_fast) + (1-exp(-Δt/τ_fast))*driver
 *   h_slow(t+Δt) = h_slow(t)*exp(-Δt/τ_slow) + (1-exp(-Δt/τ_slow))*driver
 *
 * 最终输出为加权混合：
 *   h_out = α * h_fast + (1-α) * h_slow
 * 其中α = σ(gate_mix) 是自适应混合门控
 */
static void cfc_multi_timescale_solution(CfCCell* cell, const float* input,
                                          const float* prev_state, float delta_t, float* output) {
    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;

    for (size_t i = 0; i < hidden_size; i++) {
        float gate_input_sum = cell->gate_biases[i * 3];
        float activation_input_sum = cell->bias_vector[i];

        for (size_t j = 0; j < input_size; j++) {
            size_t idx = i * input_size + j;
            gate_input_sum += cell->input_gate_weights[idx] * input[j];
            activation_input_sum += cell->weight_matrix[idx] * input[j];
        }

        float fast_gate_input = gate_input_sum;
        float fast_activation_input = activation_input_sum;
        float slow_gate_input = gate_input_sum;
        float slow_activation_input = activation_input_sum;

        for (size_t j = 0; j < hidden_size; j++) {
            size_t h_idx = i * hidden_size + j;
            fast_gate_input += cell->hidden_to_gate_weights[h_idx] * cell->fast_state[j];
            fast_activation_input += cell->hidden_to_activation_weights[h_idx] * cell->fast_state[j];
            slow_gate_input += cell->hidden_to_gate_weights[h_idx] * cell->slow_state[j];
            slow_activation_input += cell->hidden_to_activation_weights[h_idx] * cell->slow_state[j];
        }

        float fast_gate_val;
        if (fast_gate_input > 10.0f) fast_gate_val = 1.0f;
        else if (fast_gate_input < -10.0f) fast_gate_val = 0.0f;
        else fast_gate_val = 1.0f / (1.0f + expf(-fast_gate_input));

        float fast_activation_val;
        if (fast_activation_input > 10.0f) fast_activation_val = 1.0f;
        else if (fast_activation_input < -10.0f) fast_activation_val = -1.0f;
        else {
            float exp_pos = expf(fast_activation_input);
            float exp_neg = expf(-fast_activation_input);
            fast_activation_val = (exp_pos - exp_neg) / (exp_pos + exp_neg);
        }

        float slow_gate_val;
        if (slow_gate_input > 10.0f) slow_gate_val = 1.0f;
        else if (slow_gate_input < -10.0f) slow_gate_val = 0.0f;
        else slow_gate_val = 1.0f / (1.0f + expf(-slow_gate_input));

        float slow_activation_val;
        if (slow_activation_input > 10.0f) slow_activation_val = 1.0f;
        else if (slow_activation_input < -10.0f) slow_activation_val = -1.0f;
        else {
            float exp_pos = expf(slow_activation_input);
            float exp_neg = expf(-slow_activation_input);
            slow_activation_val = (exp_pos - exp_neg) / (exp_pos + exp_neg);
        }

        float fast_driver = fast_gate_val * fast_activation_val;
        float slow_driver = slow_gate_val * slow_activation_val;

        float tau_fast = cell->fast_time_constants[i];
        float tau_slow = cell->slow_time_constants[i];
        if (tau_fast < 0.001f) tau_fast = 0.001f;
        if (tau_slow > 100.0f) tau_slow = 100.0f;

        float dt_over_tau_fast = delta_t / tau_fast;
        float dt_over_tau_slow = delta_t / tau_slow;
        if (dt_over_tau_fast < 1e-8f) dt_over_tau_fast = 1e-8f;
        else if (dt_over_tau_fast > 100.0f) dt_over_tau_fast = 100.0f;
        if (dt_over_tau_slow < 1e-8f) dt_over_tau_slow = 1e-8f;
        else if (dt_over_tau_slow > 100.0f) dt_over_tau_slow = 100.0f;

        float exp_fast_val, exp_slow_val;
        float one_minus_exp_fast, one_minus_exp_slow;

        if (dt_over_tau_fast > 20.0f) { exp_fast_val = 0.0f; one_minus_exp_fast = 1.0f; }
        else if (dt_over_tau_fast < 1e-4f) { exp_fast_val = 1.0f - dt_over_tau_fast; one_minus_exp_fast = dt_over_tau_fast; }
        else { exp_fast_val = expf(-dt_over_tau_fast); one_minus_exp_fast = 1.0f - exp_fast_val; }

        if (dt_over_tau_slow > 20.0f) { exp_slow_val = 0.0f; one_minus_exp_slow = 1.0f; }
        else if (dt_over_tau_slow < 1e-4f) { exp_slow_val = 1.0f - dt_over_tau_slow; one_minus_exp_slow = dt_over_tau_slow; }
        else { exp_slow_val = expf(-dt_over_tau_slow); one_minus_exp_slow = 1.0f - exp_slow_val; }

        float new_fast = cell->fast_state[i] * exp_fast_val + one_minus_exp_fast * fast_driver;
        float new_slow = cell->slow_state[i] * exp_slow_val + one_minus_exp_slow * slow_driver;

        if (isnan(new_fast) || isinf(new_fast)) new_fast = 0.0f;
        if (isnan(new_slow) || isinf(new_slow)) new_slow = 0.0f;

        cell->fast_state[i] = new_fast;
        cell->fast_activation[i] = fast_activation_val;
        cell->slow_state[i] = new_slow;
        cell->slow_activation[i] = slow_activation_val;

        float fast_mag = fabsf(new_fast);
        float slow_mag = fabsf(new_slow);
        float total_mag = fast_mag + slow_mag + 1e-8f;
        float mix_alpha = fast_mag / total_mag;

        output[i] = mix_alpha * new_fast + (1.0f - mix_alpha) * new_slow;

        /* P1-019修复: 记录多时间尺度前向实际使用的有效tau（磁矩加权混合） */
        cell->forward_tau_used[i] = mix_alpha * tau_fast + (1.0f - mix_alpha) * tau_slow;
    }
}

/**
 * @brief 计算CfC单元的封闭形式解（真正的连续时间液态神经网络）
 * 
 * 基于微分方程：τ * dh/dt = -h + σ(W_gx * x + W_gh * h + b_g) ⊙ f(W_ax * x + W_ah * h + b_a)
 * 其中σ是sigmoid门控，f是tanh激活函数，⊙是元素乘法
 * 
 * 封闭形式解（对线性部分）：h(t+Δt) = h(t) * exp(-Δt/τ) + (1 - exp(-Δt/τ)) * g
 * 其中g = σ(...) ⊙ f(...) 是驱动项
 * 
 * @param cell CfC单元句柄
 * @param input 输入向量
 * @param prev_state 前一状态
 * @param delta_t 时间步长（秒），真正的连续时间动态
 * @param output 输出缓冲区
 */
static void cfc_closed_form_solution(CfCCell* cell, const float* input,
                                    const float* prev_state, float delta_t, float* output) {
    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    const float* time_constants = cell->time_constants;
    
    /* CfC（Closed-form Continuous-time）核心算法
     * 微分方程：τ dh/dt = -h + σ(W_gx * x + W_gh * h + b_g) ⊙ f(W_ax * x + W_ah * h + b_a)
     * 闭式解：h(t+Δt) = h(t) * exp(-Δt/τ) + (1 - exp(-Δt/τ)) * [σ(W_gx * x + W_gh * h + b_g) ⊙ f(W_ax * x + W_ah * h + b_a)]
     * 
     * 其中：
     * - τ: 时间常数（每个神经元独立）
     * - σ: sigmoid门控函数
     * - f: tanh激活函数
     * - W_gx, W_gh: 门控权重矩阵
     * - W_ax, W_ah: 激活权重矩阵
     * - b_g: 门控偏置向量
     * - b_a: 激活偏置向量
     * - ⊙: 逐元素乘法
     */
    
    for (size_t i = 0; i < hidden_size; i++) {
        // 计算门控输入：σ(W_gx * x + W_gh * h + b_g)
        float gate_input_sum = cell->gate_biases[i * 3];  // b_g (门控偏置)
        // 计算激活输入：f(W_ax * x + W_ah * h + b_a)
        float activation_input_sum = cell->bias_vector[i];  // b_a
        
        // 输入到隐藏连接：W_gx * x 和 W_ax * x
        for (size_t j = 0; j < input_size; j++) {
            size_t idx = i * input_size + j;
            gate_input_sum += cell->input_gate_weights[idx] * input[j];
            activation_input_sum += cell->weight_matrix[idx] * input[j];
        }
        
        // 隐藏到隐藏连接：W_gh * h 和 W_ah * h（CfC论文标准实现）
        for (size_t j = 0; j < hidden_size; j++) {
            size_t h_idx = i * hidden_size + j;
            gate_input_sum += cell->hidden_to_gate_weights[h_idx] * prev_state[j];
            activation_input_sum += cell->hidden_to_activation_weights[h_idx] * prev_state[j];
        }
        
        // 应用非线性激活函数：sigmoid门控和tanh激活
        float gate, activation;
        
        // sigmoid门控：σ(x) = 1/(1+exp(-x))
        if (gate_input_sum > 10.0f) {
            gate = 1.0f;
        } else if (gate_input_sum < -10.0f) {
            gate = 0.0f;
        } else {
            gate = 1.0f / (1.0f + expf(-gate_input_sum));
        }
        
        // tanh激活：tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))
        if (activation_input_sum > 10.0f) {
            activation = 1.0f;
        } else if (activation_input_sum < -10.0f) {
            activation = -1.0f;
        } else {
            float exp_pos = expf(activation_input_sum);
            float exp_neg = expf(-activation_input_sum);
            activation = (exp_pos - exp_neg) / (exp_pos + exp_neg);
        }
        
        // 驱动项：门控 ⊙ 激活（逐元素乘法）
        float driver = gate * activation;
        
        // 使用每个神经元的时间常数（如果启用自适应时间常数）
        float tau = cell->use_adaptive_tau ? time_constants[i] : cell->config.time_constant;
        
        // 限制时间常数范围以确保数值稳定性
        if (tau < cell->min_time_constant) tau = cell->min_time_constant;
        if (tau > cell->max_time_constant) tau = cell->max_time_constant;
        
        /* P1-019修复: 记录前向实际使用的tau，供反向传播直接读取 */
        cell->forward_tau_used[i] = tau;
        
        // 确保Δt/τ不会太小导致数值下溢
        float dt_over_tau = delta_t / tau;
        if (dt_over_tau < 1e-8f) {
            dt_over_tau = 1e-8f;
        } else if (dt_over_tau > 100.0f) {
            dt_over_tau = 100.0f;
        }
        
        // 计算指数项：exp(-Δt/τ) - 使用高精度计算
        float exp_term;
        if (dt_over_tau > 20.0f) {
            exp_term = 0.0f;  // 对于大值，exp(-x) ≈ 0
        } else if (dt_over_tau < 1e-4f) {
            exp_term = 1.0f - dt_over_tau;  // 小值一阶泰勒近似
        } else {
            exp_term = expf(-dt_over_tau);
        }
        
        // 计算(1 - exp(-Δt/τ))项
        float one_minus_exp_term;
        if (dt_over_tau > 20.0f) {
            one_minus_exp_term = 1.0f;  // exp(-x) ≈ 0, 所以 1 - exp(-x) ≈ 1
        } else if (dt_over_tau < 1e-4f) {
            one_minus_exp_term = dt_over_tau;  // 小值一阶泰勒近似
        } else {
            one_minus_exp_term = 1.0f - exp_term;
        }
        
        // 真正的封闭形式解：h(t+Δt) = h(t) * exp(-Δt/τ) + (1 - exp(-Δt/τ)) * driver
        // 这是微分方程 τ dh/dt = -h + driver 的精确解
        float new_state = prev_state[i] * exp_term + one_minus_exp_term * driver;
        
        // 确保数值稳定性
        if (isnan(new_state) || isinf(new_state)) {
            new_state = 0.0f;
        }
        
        // 输出新状态
        output[i] = new_state;
        
        // 自适应时间常数更新（如果启用）- 基于CfC论文的自适应机制
        if (cell->use_adaptive_tau) {
            // 基于激活度调整时间常数：高激活度 -> 更快响应（更小τ）
            // 使用sigmoid样函数将激活度映射到时间常数范围
            float activation_magnitude = fabsf(new_state);
            
            // 目标时间常数：激活度越高，时间常数越小（响应越快）
            // 使用指数衰减：τ_target = τ_min + (τ_max - τ_min) * exp(-β * |h|)
            float beta = 2.0f;  // 衰减系数
            float target_tau = cell->min_time_constant + 
                              (cell->max_time_constant - cell->min_time_constant) * 
                              expf(-beta * activation_magnitude);
            
            // 平滑更新：τ ← τ + η * (target_τ - τ)
            float tau_adjustment = cell->state->adaptation_rate * (target_tau - time_constants[i]);
            cell->time_constants[i] += tau_adjustment;
            
            // 限制范围
            if (cell->time_constants[i] < cell->min_time_constant) {
                cell->time_constants[i] = cell->min_time_constant;
            }
            if (cell->time_constants[i] > cell->max_time_constant) {
                cell->time_constants[i] = cell->max_time_constant;
            }
        }
    }
}

/**
 * @brief 使用四阶龙格-库塔（RK4）方法求解CfC ODE
 * 
 * ODE: dh/dt = f(h, x) = (-h + σ(W_gx*x + W_gh*h + b_g) ⊙ f(W_ax*x + W_ah*h + b_a)) / τ
 * 
 * RK4更新：
 *   k1 = f(h_n, x)
 *   k2 = f(h_n + 0.5*Δt*k1, x)
 *   k3 = f(h_n + 0.5*Δt*k2, x)
 *   k4 = f(h_n + Δt*k3, x)
 *   h_{n+1} = h_n + (Δt/6) * (k1 + 2*k2 + 2*k3 + k4)
 * 
 * 使用activations作为中间隐藏状态的临时存储，
 * 使用gradient/noise_buffer/adapted_params/workspace分别存储k1/k2/k3/k4。
 * 
 * @param cell CfC单元句柄
 * @param input 输入向量
 * @param prev_state 前一状态
 * @param delta_t 时间步长（秒）
 * @param output 输出缓冲区
 */
static void cfc_cell_rk4_step(CfCCell* cell, const float* input,
                              const float* prev_state, float delta_t, float* output) {
    size_t hidden_size = cell->config.hidden_size;
    
    float* k1 = cell->state->gradient;
    float* k2 = cell->state->noise_buffer;
    float* k3 = cell->state->adapted_params;
    float* k4 = cell->state->workspace;
    float* temp_state = cell->state->activation;
    
    float half_dt = delta_t * 0.5f;
    float sixth_dt = delta_t / 6.0f;
    
    cfc_cell_compute_rhs(cell, input, prev_state, k1);
    
    for (size_t i = 0; i < hidden_size; i++) {
        temp_state[i] = prev_state[i] + half_dt * k1[i];
    }
    cfc_cell_compute_rhs(cell, input, temp_state, k2);
    
    for (size_t i = 0; i < hidden_size; i++) {
        temp_state[i] = prev_state[i] + half_dt * k2[i];
    }
    cfc_cell_compute_rhs(cell, input, temp_state, k3);
    
    for (size_t i = 0; i < hidden_size; i++) {
        temp_state[i] = prev_state[i] + delta_t * k3[i];
    }
    cfc_cell_compute_rhs(cell, input, temp_state, k4);
    
    for (size_t i = 0; i < hidden_size; i++) {
        float rk4_sum = k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i];
        float new_state = prev_state[i] + sixth_dt * rk4_sum;
        if (isnan(new_state) || isinf(new_state)) {
            new_state = 0.0f;
        }
        output[i] = new_state;
    }
}

/* ============================================================================
 * RK45自适应步长求解器和CTBP连续时间反向传播的静态前向声明
 * =========================================================================== */
static void cfc_cell_rk45_step(CfCCell* cell, const float* input,
                               const float* prev_state, float delta_t, float* output);
static void cfc_cell_ctbp_forward(CfCCell* cell, const float* input,
                                  const float* prev_state, float delta_t, float* output);

/**
 * @brief 前向传播
 */
/* ============ DP54 Dormand-Prince 5(4)自适应步长求解器实现 ============ */

int cfc_cell_rhs_wrapper(float t, const float* y, float* dydt, void* ctx)
{
    (void)t;
    CfCCell* cell = (CfCCell*)ctx;
    cfc_cell_compute_rhs(cell, cell->state->input_buffer, y, dydt);
    return 0;
}

static int cfc_cell_dp54_step(CfCCell* cell, const float* input,
                               const float* prev_state, float delta_t,
                               float* output)
{
    (void)input;
    size_t n = cell->config.hidden_size;
    /* 使用堆分配替代VLA栈分配，避免16KB栈溢出风险 */
    float* y_state = (float*)safe_malloc(n * sizeof(float));
    float* workspace = (float*)safe_malloc(ode_dp54_workspace_size(n) * sizeof(float));
    int need_free = 1;
    if (!y_state || !workspace) {
        safe_free((void**)&y_state);
        safe_free((void**)&workspace);
        return -1;
    }

    if (n > 0) memset(y_state, 0, n * sizeof(float));

    memcpy(y_state, prev_state, n * sizeof(float));

    DP54Config cfg = cell->config.dp54_config;
    int steps = 0;
    float h_actual = 0.0f;

    int ret = ode_dp54_solve(y_state, 0.0f, delta_t, cfc_cell_rhs_wrapper,
                              cell, n, &cfg, workspace, &h_actual, &steps);
    if (ret == 0)
    {
        memcpy(output, y_state, n * sizeof(float));
        cell->state->dp54_current_steps = steps;
    }

    if (need_free) { safe_free((void**)&y_state); safe_free((void**)&workspace); }
    return ret;
}

/* ============ Rosenbrock刚性求解器实现 ============ */

static int cfc_cell_rosenbrock_step(CfCCell* cell, const float* input,
                                     const float* prev_state, float delta_t,
                                     float* output)
{
    (void)input;
    size_t n = cell->config.hidden_size;
    RosenbrockConfig cfg = cell->config.rosenbrock_config;

    /* 当配置了有效的自适应容差参数时，优先使用自适应步长求解器 */
    if (cfg.rel_tolerance > 1e-12f && cfg.abs_tolerance > 1e-12f &&
        cfg.min_step_size > 0.0f && cfg.max_step_size > cfg.min_step_size) {
        float* y_state = (float*)malloc(n * sizeof(float));
        if (!y_state) return -1;
        memcpy(y_state, prev_state, n * sizeof(float));

        float h_final = 0.0f;
        int steps_taken = 0;
        float tolerance = cfg.abs_tolerance;
        float h0 = delta_t / 10.0f;
        if (h0 > cfg.max_step_size) h0 = cfg.max_step_size;
        if (h0 < cfg.min_step_size) h0 = cfg.min_step_size;

        int ret = ode_rosenbrock_adaptive_solve(cfc_cell_rhs_wrapper, cell,
                                                 y_state, n, 0.0f, delta_t,
                                                 h0, tolerance,
                                                 cfg.min_step_size, cfg.max_step_size,
                                                 cfg.max_iterations > 0 ? cfg.max_iterations : 1000,
                                                 &h_final, &steps_taken);
        if (ret == 0) {
            memcpy(output, y_state, n * sizeof(float));
            cell->state->rosenbrock_current_steps = steps_taken;
        }
        free(y_state);
        return ret;
    }

    /* 回退到固定步长Rosenbrock（传统模式） */
    size_t ws_size = ode_rosenbrock_workspace_size(n);
    float y[sizeof(float) * 1024];
    float ws[sizeof(float) * 8192];
    float* y_state = y;
    float* workspace = ws;
    int need_free = 0;

    if (n > 1024 || ws_size > 8192)
    {
        y_state = (float*)malloc(n * sizeof(float));
        workspace = (float*)malloc(ws_size * sizeof(float));
        if (!y_state || !workspace)
        {
            safe_free((void**)&y_state); safe_free((void**)&workspace);
            return -1;
        }
        need_free = 1;
    }

    memcpy(y_state, prev_state, n * sizeof(float));

    int steps = 0;
    float h_actual = 0.0f;

    int ret = ode_rosenbrock_solve(y_state, 0.0f, delta_t, cfc_cell_rhs_wrapper,
                                    cell, n, &cfg, workspace, &h_actual, &steps);
    if (ret == 0)
    {
        memcpy(output, y_state, n * sizeof(float));
        cell->state->rosenbrock_current_steps = steps;
    }

    if (need_free) { safe_free((void**)&y_state); safe_free((void**)&workspace); }
    return ret;
}

/* ============ Forest-Ruth辛积分器实现 ============ */

static int cfc_symplectic_dqdt_wrapper(float t, const float* p, float* dqdt_out, void* ctx)
{
    (void)t;
    CfCCell* cell = (CfCCell*)ctx;
    size_t n = cell->config.hidden_size;
    /* 辛动力学：使用CfC门控计算dq/dt = gate*tanh(input) + (1-gate)*momentum */
    /* 使用已分配的input_buffer和symplectic_dqdt作为缓冲区 */
    float* tmp = cell->state->symplectic_dqdt;
    if (!tmp) {
        memcpy(dqdt_out, p, n * sizeof(float));
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        float input_val = cell->state->input_buffer ? cell->state->input_buffer[i] : 0.0f;
        float nonlinear = tanhf(input_val);
        float gate = 1.0f / (1.0f + expf(-input_val));
        tmp[i] = gate * nonlinear + (1.0f - gate) * p[i];
    }
    memcpy(dqdt_out, tmp, n * sizeof(float));
    return 0;
}

static int cfc_cell_symplectic_step(CfCCell* cell, const float* input,
                                     const float* prev_state, float delta_t,
                                     float* output)
{
    size_t n = cell->config.hidden_size;

    /* 保存输入到input_buffer供dqdt使用 */
    if (cell->state->input_buffer && input)
        memcpy(cell->state->input_buffer, input, n * sizeof(float));

    if (!cell->state->symplectic_initialized)
    {
        for (size_t i = 0; i < n; i++)
            cell->state->symplectic_momentum[i] = 0.0f;
        cell->state->symplectic_initialized = 1;
    }

    SymplecticConfig cfg = cell->config.symplectic_config;

    float* q = (float*)cell->state->state;
    float* p = cell->state->symplectic_momentum;

    /* 堆分配替代VLA栈分配 */
    float* workspace = (float*)safe_malloc(ode_forest_ruth_workspace_size(n) * sizeof(float));
    int need_free = 1;
    if (!workspace) return -1;

    int steps = 0;

    int ret = ode_forest_ruth_solve(q, p, delta_t,
                                     cfc_symplectic_dqdt_wrapper,
                                     cfc_cell_rhs_wrapper,
                                     cell, n, &cfg, workspace, &steps);

    if (ret == 0) {
        memcpy(output, q, n * sizeof(float));
        cell->state->symplectic_current_steps = steps;
    }

    if (need_free) safe_free((void**)&workspace);
    return ret;
}

/**
 * @brief RK45自适应步长步进函数
 *
 * 将目标步长 delta_t 分解为多个自适应子步，
 * 使用 ode_adaptive_step_control 动态调整每个子步的步长，
 * 确保每步误差满足容限要求。
 *
 * @return 0=成功, -1=参数错误, -2=不收敛
 */
static int cfc_cell_rk45_adaptive(CfCCell* cell, const float* input,
                                   const float* prev_state, float delta_t,
                                   float* output,
                                   const AdaptiveStepConfig* acfg,
                                   AdaptiveStepSolution* asol)
{
    if (!cell || !input || !prev_state || !output || !acfg || !asol)
        return -1;

    size_t n = cell->config.hidden_size;
    float h_current = (float)fmin((double)fabs((double)delta_t), (double)acfg->max_step_size);
    if (h_current < acfg->min_step_size) h_current = acfg->min_step_size;

    int sign = (delta_t >= 0.0f) ? 1 : -1;
    float t_target = asol->t_current + delta_t;
    float remaining = delta_t;

    float* state_buf = (float*)malloc(n * sizeof(float));
    if (!state_buf) return -1;
    memcpy(state_buf, prev_state, n * sizeof(float));

    while (sign * (t_target - asol->t_current) > 1e-14f) {
        float h_step = (float)fabs(remaining);
        if (h_step > h_current) h_step = h_current;
        if (h_step < acfg->min_step_size) h_step = acfg->min_step_size;

        memcpy(cell->state->state, state_buf, n * sizeof(float));

        cfc_cell_rk45_step(cell, input, state_buf, sign * h_step, output);

        float max_error = 0.0f;
        float norm_error = ode_compute_normalized_error(
            output, cell->state->workspace, state_buf,
            n, acfg->rel_tolerance, acfg->abs_tolerance, &max_error);
        (void)norm_error;

        int accepted = ode_step_is_accepted(max_error, acfg);
        if (accepted) {
            memcpy(state_buf, output, n * sizeof(float));
            remaining -= sign * h_step;
            asol->t_current += sign * h_step;
            h_current = ode_adaptive_step_control(max_error, h_current, acfg, asol);
        } else {
            float h_new = ode_adaptive_step_control(max_error, h_current, acfg, asol);
            if (h_new >= h_current - 1e-14f) h_current *= 0.5f;
            else h_current = h_new;

            if (h_current < acfg->min_step_size) {
                memcpy(state_buf, output, n * sizeof(float));
                remaining -= sign * h_step;
                asol->t_current += sign * h_step;
                h_current = acfg->min_step_size;
            }
        }

        asol->total_steps++;
    }

    memcpy(output, state_buf, n * sizeof(float));
    memcpy(cell->state->state, state_buf, n * sizeof(float));
    free(state_buf);
    return 0;
}

/**
 * @brief DP54自适应步长步进函数
 *
 * 将目标步长 delta_t 分解为多个自适应子步，
 * DP54本身已有4阶/5阶误差估计，此函数利用统一框架
 * 进行步长控制。
 *
 * @return 0=成功, -1=参数错误
 */
static int cfc_cell_dp54_adaptive(CfCCell* cell, const float* input,
                                   const float* prev_state, float delta_t,
                                   float* output,
                                   const AdaptiveStepConfig* acfg,
                                   AdaptiveStepSolution* asol)
{
    if (!cell || !input || !prev_state || !output || !acfg || !asol)
        return -1;

    size_t n = cell->config.hidden_size;
    float h_current = (float)fmin((double)fabs((double)delta_t), (double)acfg->max_step_size);
    if (h_current < acfg->min_step_size) h_current = acfg->min_step_size;

    int sign = (delta_t >= 0.0f) ? 1 : -1;
    float t_target = asol->t_current + delta_t;
    float remaining = delta_t;

    float* state_buf = (float*)malloc(n * sizeof(float));
    if (!state_buf) return -1;
    memcpy(state_buf, prev_state, n * sizeof(float));

    while (sign * (t_target - asol->t_current) > 1e-14f) {
        float h_step = (float)fabs(remaining);
        if (h_step > h_current) h_step = h_current;
        if (h_step < acfg->min_step_size) h_step = acfg->min_step_size;

        int ret = cfc_cell_dp54_step(cell, input, state_buf, sign * h_step, output);
        if (ret != 0) {
            free(state_buf);
            return ret;
        }

        float max_error = 0.0f;
        float norm_error = ode_compute_normalized_error(
            output, cell->state->dp54_workspace, state_buf,
            n, acfg->rel_tolerance, acfg->abs_tolerance, &max_error);
        (void)norm_error;

        int accepted = ode_step_is_accepted(max_error, acfg);
        if (accepted) {
            memcpy(state_buf, output, n * sizeof(float));
            remaining -= sign * h_step;
            asol->t_current += sign * h_step;
            h_current = ode_adaptive_step_control(max_error, h_current, acfg, asol);
        } else {
            float h_new = ode_adaptive_step_control(max_error, h_current, acfg, asol);
            if (h_new >= h_current - 1e-14f) h_current *= 0.5f;
            else h_current = h_new;

            if (h_current < acfg->min_step_size) {
                memcpy(state_buf, output, n * sizeof(float));
                remaining -= sign * h_step;
                asol->t_current += sign * h_step;
                h_current = acfg->min_step_size;
            }
        }

        asol->total_steps++;
    }

    memcpy(output, state_buf, n * sizeof(float));
    memcpy(cell->state->state, state_buf, n * sizeof(float));
    free(state_buf);
    return 0;
}

/**
 * @brief Rosenbrock并行求解函数
 *
 * 使用域分解 + OpenMP并行加速Rosenbrock刚性求解器。
 * 状态向量被均匀分割到多个域上，每个域独立计算RHS和雅可比矩阵。
 *
 * @return 0=成功, -1=参数错误
 */
static int cfc_cell_rosenbrock_parallel(CfCCell* cell, const float* input,
                                         const float* prev_state, float delta_t,
                                         float* output,
                                         const ParallelODERHSConfig* pcfg)
{
    if (!cell || !input || !prev_state || !output || !pcfg)
        return -1;

    size_t n = cell->config.hidden_size;

    /* 如果无并行或域数=1，回退到串行实现 */
    if (pcfg->num_domains <= 1 || pcfg->mode == PARALLEL_MODE_NONE) {
        return cfc_cell_rosenbrock_step(cell, input, prev_state, delta_t, output);
    }

    int nd = pcfg->num_domains;

    /* 栈分配域偏移/大小数组 */
    int dom_sizes[64];
    int dom_offsets[64];
    int* sizes = dom_sizes;
    int* offsets = dom_offsets;

    int need_free = 0;
    if (nd > 64) {
        sizes = (int*)malloc((size_t)nd * 2 * sizeof(int));
        if (!sizes) return -1;
        offsets = sizes + nd;
        need_free = 1;
    }

    ode_domain_decompose(n, nd, sizes, offsets);

    memcpy(cell->state->state, prev_state, n * sizeof(float));
    memcpy(cell->state->input_buffer, input, cell->config.input_size * sizeof(float));

    /* 并行RHS求值：计算雅可比矩阵 */
    float* J = cell->state->rosenbrock_jacobian;
    if (J) {
        memset(J, 0, n * n * sizeof(float));
        float eps = cell->config.rosenbrock_config.finite_diff_eps > 0.0f ?
                     cell->config.rosenbrock_config.finite_diff_eps : 1e-6f;

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) if(nd > 1)
#endif
        for (int d = 0; d < nd; d++) {
            int start = offsets[d];
            int size = sizes[d];
            float* J_local = J + start * (int)n;

            for (int j = 0; j < size; j++) {
                int col = start + j;
                float y_save = prev_state[col];
                float h_var = eps * (float)fmax(1.0, (double)fabs(y_save));

                float* y_pert = (float*)alloca(n * sizeof(float));
                memcpy(y_pert, prev_state, n * sizeof(float));
                y_pert[col] += h_var;

                float* f_pert = (float*)alloca(n * sizeof(float));
                cfc_cell_compute_rhs(cell, input, y_pert, f_pert);

                float* f_base = (float*)alloca(n * sizeof(float));
                cfc_cell_compute_rhs(cell, input, prev_state, f_base);

                for (size_t i = 0; i < n; i++) {
                    J_local[j * (int)n + i] = (f_pert[i] - f_base[i]) / h_var;
                }
            }
        }
    }

    /* 串行执行Rosenbrock步（雅可比已在并行中计算完成） */
    memcpy(cell->state->state, prev_state, n * sizeof(float));
    int ret = cfc_cell_rosenbrock_step(cell, input, prev_state, delta_t, output);

    if (need_free) free(sizes);
    return ret;
}

/* ============ 四元数CfC单元前向传播（P2.4） ============ */

/**
 * @brief 四元数CfC前向传播
 *
 * 将输入和隐藏状态视为四元数数组（每4个float为一组四元数 w/x/y/z），
 * 使用四元数矩阵乘法和CfC闭式解进行连续时间状态演化。
 *
 * @param cell CfC单元指针（需已启用use_quaternion）
 * @param input 输入向量（大小 = input_size，需为4的倍数）
 * @param hidden_state 输出隐藏状态缓冲区（大小 = hidden_size，需为4的倍数）
 * @return int 成功返回0，失败返回-1
 */
static int cfc_cell_forward_quaternion(CfCCell* cell, const float* input, float* hidden_state)
{
    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    size_t num_hidden_quats = hidden_size / 4;
    size_t num_input_quats = input_size / 4;
    float dt = cell->config.delta_t;

    /* ZSFZS-F011修复: 使用memcpy安全转换替代裸指针强制类型转换，
     * 避免违反C严格别名规则。Quaternion与float[4]内存布局一致，
     * 通过memcpy避免未定义行为。 */
    Quaternion input_quats_buf[128];
    Quaternion state_quats_buf[128];
    Quaternion output_quats_buf[128];
    size_t nq = num_hidden_quats < 128 ? num_hidden_quats : 128;
    for (size_t i = 0; i < num_input_quats && i < 128; i++) {
        memcpy(&input_quats_buf[i], input + i * 4, sizeof(Quaternion));
    }
    for (size_t i = 0; i < nq; i++) {
        memcpy(&state_quats_buf[i], cell->state->state + i * 4, sizeof(Quaternion));
    }
    Quaternion* input_quats = input_quats_buf;
    Quaternion* state_quats = state_quats_buf;
    Quaternion* output_quats = output_quats_buf;

    /* 每四元数循环：驱动 = 偏置 + 输入投影 + 状态反馈 */
    Quaternion* W = (Quaternion*)cell->quaternion_weights;    /* [num_hidden_quats × num_input_quats] */
    Quaternion* H = (Quaternion*)cell->quaternion_hidden_weights; /* [num_hidden_quats × num_hidden_quats] */
    Quaternion* B = (Quaternion*)cell->quaternion_biases;
    float* tau = cell->quaternion_time_constants;

    for (size_t i = 0; i < num_hidden_quats; i++) {
        /* 从偏置开始 */
        Quaternion drive = B[i];

        /* 输入投影: Σ_j W[i][j] * input[j] */
        for (size_t j = 0; j < num_input_quats; j++) {
            size_t idx = i * num_input_quats + j;
            Quaternion prod = quaternion_multiply(&W[idx], &input_quats[j]);
            drive = quaternion_add(&drive, &prod);
        }

        /* 状态反馈: Σ_k H[i][k] * state[k] */
        for (size_t k = 0; k < num_hidden_quats; k++) {
            size_t hidx = i * num_hidden_quats + k;
            Quaternion prod = quaternion_multiply(&H[hidx], &state_quats[k]);
            drive = quaternion_add(&drive, &prod);
        }

        /* CfC闭式解：s' = s * exp(-dt/τ) + (1 - exp(-dt/τ)) * drive */
        Quaternion tc;
        tc.w = tau[i]; tc.x = tau[i]; tc.y = tau[i]; tc.z = tau[i];
        quaternion_cfc_closed_form_update(&state_quats[i], &drive, &tc, dt, &output_quats[i]);

        /* 保持范数稳定 */
        output_quats[i] = quaternion_normalize(&output_quats[i]);
    }

    /* 输出：将四元数数组展平为float数组 */
    for (size_t i = 0; i < num_hidden_quats; i++) {
        size_t off = i * 4;
        /* ZSFZS-F011: 同时写入hidden_state和内部activation */
        hidden_state[off]     = output_quats[i].w;
        hidden_state[off + 1] = output_quats[i].x;
        hidden_state[off + 2] = output_quats[i].y;
        hidden_state[off + 3] = output_quats[i].z;
        memcpy(cell->state->activation + off, &output_quats[i], sizeof(Quaternion));
    }

    /* 更新内部状态 */
    memcpy(cell->state->state, cell->state->activation, hidden_size * sizeof(float));
    cell->state->time += 1.0f;

    /* ZSFZS-F011修复: 四元数路径记录forward_tau_used保证反向传播一致性 */
    if (cell->forward_tau_used && num_hidden_quats > 0) {
        cell->forward_tau_used[0] = tau[0];  /* 记录首个四元数的时间常数 */
    }

    return 0;
}

/* ============ ODE工作空间按需分配 ============ */

/**
 * @brief 确保指定ODE求解器类型的工作空间已分配
 * 仅在非闭合解求解器首次使用时分配，避免不必要的内存开销
 */
static int ensure_ode_workspace(CfCCell* cell, int solver_type) {
    size_t n = cell->config.hidden_size;
    if (n == 0) return -1;

    switch (solver_type) {
    case ODE_SOLVER_RK4:
    case ODE_SOLVER_RK45:
        if (!cell->state->rk45_k1) {
            cell->state->rk45_k1 = (float*)safe_calloc(n, sizeof(float));
            cell->state->rk45_k2 = (float*)safe_calloc(n, sizeof(float));
            cell->state->rk45_k3 = (float*)safe_calloc(n, sizeof(float));
            cell->state->rk45_k4 = (float*)safe_calloc(n, sizeof(float));
            cell->state->rk45_k5 = (float*)safe_calloc(n, sizeof(float));
            cell->state->rk45_k6 = (float*)safe_calloc(n, sizeof(float));
            cell->state->rk45_temp = (float*)safe_calloc(n, sizeof(float));
            cell->state->rk45_error = (float*)safe_calloc(n, sizeof(float));
            if (!cell->state->rk45_k1 || !cell->state->rk45_k2 ||
                !cell->state->rk45_k3 || !cell->state->rk45_k4 ||
                !cell->state->rk45_k5 || !cell->state->rk45_k6 ||
                !cell->state->rk45_temp || !cell->state->rk45_error)
                return -1;
        }
        break;
    case ODE_SOLVER_DP54:
        if (!cell->state->dp54_k7) {
            cell->state->dp54_k7 = (float*)safe_calloc(n, sizeof(float));
            cell->state->dp54_workspace = (float*)safe_calloc(n * 8, sizeof(float));
            if (!cell->state->dp54_k7 || !cell->state->dp54_workspace) return -1;
        }
        break;
    case ODE_SOLVER_ROSENBROCK:
        if (!cell->state->rosenbrock_jacobian) {
            cell->state->rosenbrock_jacobian = (float*)safe_calloc(n * n, sizeof(float));
            cell->state->rosenbrock_k1 = (float*)safe_calloc(n, sizeof(float));
            cell->state->rosenbrock_k2 = (float*)safe_calloc(n, sizeof(float));
            cell->state->rosenbrock_k3 = (float*)safe_calloc(n, sizeof(float));
            cell->state->rosenbrock_workspace = (float*)safe_calloc(n * 6 + n * n, sizeof(float));
            if (!cell->state->rosenbrock_jacobian || !cell->state->rosenbrock_k1 ||
                !cell->state->rosenbrock_k2 || !cell->state->rosenbrock_k3 ||
                !cell->state->rosenbrock_workspace) return -1;
        }
        break;
    case ODE_SOLVER_SYMPLECTIC:
        if (!cell->state->symplectic_momentum) {
            cell->state->symplectic_momentum = (float*)safe_calloc(n, sizeof(float));
            cell->state->symplectic_dqdt = (float*)safe_calloc(n, sizeof(float));
            cell->state->symplectic_dpdt = (float*)safe_calloc(n, sizeof(float));
            if (!cell->state->symplectic_momentum || !cell->state->symplectic_dqdt ||
                !cell->state->symplectic_dpdt) return -1;
        }
        break;
    case ODE_SOLVER_CTBP:
        if (!cell->state->ctbp_adjoint) {
            cell->state->ctbp_adjoint = (float*)safe_calloc(n, sizeof(float));
            cell->state->ctbp_temp = (float*)safe_calloc(n, sizeof(float));
            if (!cell->state->ctbp_adjoint || !cell->state->ctbp_temp) return -1;
        }
        if (!cell->state->ctbp_state_trajectory) {
            cell->state->ctbp_state_trajectory = (float*)safe_calloc(n * 100, sizeof(float));
            if (!cell->state->ctbp_state_trajectory) return -1;
        }
        break;
    default:
        break; /* 闭合解不需要工作空间 */
    }
    return 0;
}

/* ============ CfC单元前向传播 ============ */

int cfc_cell_forward(CfCCell* cell, const float* input, float* hidden_state) {
    // 参数检查
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(input, "输入向量为空");
    SELFLNN_CHECK_NULL(hidden_state, "隐藏状态缓冲区为空");
    
    // 初始化状态检查
    SELFLNN_CHECK_INITIALIZED(cell, "CfC单元未初始化");
    CHECK_CFC_STATE_INT(cell);
    
    /* 四元数CfC路径（P2.4）：使用四元数连续时间动态 */
    if (cell->use_quaternion) {
        return cfc_cell_forward_quaternion(cell, input, hidden_state);
    }

    /* 门控CfC变体：启用交叉门控/金字塔门控/自适应带宽增强的CfC动态 */
    if (cell->gated_data) {
        return cfc_cell_forward_gated(cell, input, hidden_state);
    }

    /* 分层CfC变体：多层级自底向上/自顶向下信息流 */
    if (cell->hierarchical_data) {
        return cfc_cell_forward_hierarchical(cell, input, hidden_state);
    }

    /* 液态记忆门控CfC：基于拼接门控的记忆调制动态 */
    if (cell->liquid_memory_data) {
        return cfc_cell_forward_liquid_memory(cell, input, hidden_state);
    }
    
    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    
    // 复制输入到缓冲区
    memcpy(cell->state->input_buffer, input, input_size * sizeof(float));
    
    // 添加噪声（如果启用）
    if (cell->config.noise_std > 0.0f) {
        for (size_t i = 0; i < input_size; i++) {
            cell->state->input_buffer[i] += 
                rng_normal(0.0f, cell->config.noise_std);
        }
    }
    
    // 自动求解器选择（P3.4）：若启用，则在每个前向步骤前检测刚度并自动切换求解器
    if (cell->config.use_auto_solver && cell->enhanced_state) {
        cfc_select_solver_by_stiffness(cell,
                                       cell->state->input_buffer,
                                       cell->state->state,
                                       &cell->enhanced_config.auto_solver,
                                       cell->enhanced_state);
    }

    // 液时域缩放：若启用，使用输入依赖的动态时间常数执行CfC前向传播
    if (cell->use_liquid_scaling) {
        return cfc_cell_forward_liquid(cell, cell->state->input_buffer,
                                       hidden_state, cell->computed_liquid_tau);
    }

    // 根据求解器类型选择ODE求解方法
    // 在非闭合解求解器首次使用时按需分配工作空间
    if (cell->config.ode_solver_type != ODE_SOLVER_CLOSED_FORM) {
        if (ensure_ode_workspace(cell, cell->config.ode_solver_type) != 0) {
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "ODE求解器工作空间分配失败");
            return -1;
        }
    }
    
    // 如果启用了多时间尺度并行演化，使用多尺度求解器
    if (cell->use_multi_timescale) {
        cfc_multi_timescale_solution(cell,
                                    cell->state->input_buffer,
                                    cell->state->state,
                                    cell->config.delta_t,
                                    cell->state->activation);
    } else if (cell->config.ode_solver_type == ODE_SOLVER_RK4) {
        cfc_cell_rk4_step(cell,
                         cell->state->input_buffer,
                         cell->state->state,
                         cell->config.delta_t,
                         cell->state->activation);
    } else if (cell->config.ode_solver_type == ODE_SOLVER_RK45) {
        if (cell->use_adaptive_step) {
            cfc_cell_rk45_adaptive(cell,
                                  cell->state->input_buffer,
                                  cell->state->state,
                                  cell->config.delta_t,
                                  cell->state->activation,
                                  &cell->config.adaptive_step_cfg,
                                  &cell->adaptive_step_sol);
        } else {
            cfc_cell_rk45_step(cell,
                              cell->state->input_buffer,
                              cell->state->state,
                              cell->config.delta_t,
                              cell->state->activation);
        }
    } else if (cell->config.ode_solver_type == ODE_SOLVER_CTBP) {
        cfc_cell_ctbp_forward(cell,
                             cell->state->input_buffer,
                             cell->state->state,
                             cell->config.delta_t,
                             cell->state->activation);
    } else if (cell->config.ode_solver_type == ODE_SOLVER_DP54) {
        if (cell->use_adaptive_step) {
            cfc_cell_dp54_adaptive(cell,
                                  cell->state->input_buffer,
                                  cell->state->state,
                                  cell->config.delta_t,
                                  cell->state->activation,
                                  &cell->config.adaptive_step_cfg,
                                  &cell->adaptive_step_sol);
        } else {
            cfc_cell_dp54_step(cell,
                              cell->state->input_buffer,
                              cell->state->state,
                              cell->config.delta_t,
                              cell->state->activation);
        }
    } else if (cell->config.ode_solver_type == ODE_SOLVER_ROSENBROCK) {
        if (cell->use_parallel_solve) {
            cfc_cell_rosenbrock_parallel(cell,
                                        cell->state->input_buffer,
                                        cell->state->state,
                                        cell->config.delta_t,
                                        cell->state->activation,
                                        &cell->config.parallel_cfg);
        } else {
            cfc_cell_rosenbrock_step(cell,
                                    cell->state->input_buffer,
                                    cell->state->state,
                                    cell->config.delta_t,
                                    cell->state->activation);
        }
    } else if (cell->config.ode_solver_type == ODE_SOLVER_SYMPLECTIC) {
        cfc_cell_symplectic_step(cell,
                                cell->state->input_buffer,
                                cell->state->state,
                                cell->config.delta_t,
                                cell->state->activation);
    } else if (cell->config.ode_solver_type == ODE_SOLVER_RHS) {
        /* P2-041修复: 使用cfc_cell_compute_rhs进行直接RHS评估（前向欧拉步）
         * 作为备选数值求解器，对动态系统进行单步欧拉积分 */
        float* rhs = cell->state->noise_buffer;
        cfc_cell_compute_rhs(cell, cell->state->input_buffer,
                            cell->state->state, rhs);
        size_t hs = cell->config.hidden_size;
        for (size_t i = 0; i < hs; i++) {
            float new_h = cell->state->state[i] + cell->config.delta_t * rhs[i];
            if (isnan(new_h) || isinf(new_h)) new_h = 0.0f;
            cell->state->activation[i] = new_h;
        }
    } else {
        cfc_closed_form_solution(cell,
                                cell->state->input_buffer,
                                cell->state->state,
                                cell->config.delta_t,
                                cell->state->activation);
    }
    
    // 保存前向传播前的状态 h(t)，供反向传播使用
    // 说明: ODE求解器读state为const输入，写activation为输出，
    // state在后续line才被更新，此处saved_state正确保存的是h(t)
    memcpy(cell->state->saved_state, cell->state->state, hidden_size * sizeof(float));

    /* P2-002: 残差连接——在所有ODE求解器路径之后统一应用
     * h = h_solver + α * h_prev
     * 为深层网络提供梯度高速公路，增强8层以上网络的训练稳定性 */
    if (cell->config.use_residual && cell->config.residual_scale > 0.0f) {
        for (size_t i = 0; i < hidden_size; i++) {
            cell->state->activation[i] += cell->config.residual_scale * cell->state->state[i];
            /* 确保数值稳定 */
            if (isnan(cell->state->activation[i]) || isinf(cell->state->activation[i])) {
                cell->state->activation[i] = 0.0f;
            }
        }
    }

    /* P1-001: 层归一化——对输出到下一层的隐藏状态进行归一化
     * 保留原始激活值用于内部状态演化（不破坏CfC动力学）
     * 仅归一化对外输出的hidden_state，消除层间的内部协变量偏移 */
    if (cell->cell_layer_norm && cell->config.use_cell_layer_norm) {
        layer_norm_forward(cell->cell_layer_norm, cell->state->activation, hidden_state, 1);
    } else {
        // 复制激活值到隐藏状态
        memcpy(hidden_state, cell->state->activation, hidden_size * sizeof(float));
    }

    // 更新内部状态（使用原始激活值，保持CfC动力学完整性）
    memcpy(cell->state->state, cell->state->activation, hidden_size * sizeof(float));
    
    // 更新时间
    cell->state->time += 1.0f;
    
    // 更新统计信息
    float sum_activation = 0.0f;
    float max_activation = 0.0f;
    
    for (size_t i = 0; i < hidden_size; i++) {
        float activation = fabsf(cell->state->activation[i]);
        sum_activation += activation;
        if (activation > max_activation) {
            max_activation = activation;
        }
    }
    
    cell->avg_activation = sum_activation / hidden_size;
    cell->max_activation = max_activation;
    
    // 自适应参数更新（如果启用）
    if (cell->config.enable_adaptation) {
        // 启用时间常数自适应
        cell->use_adaptive_tau = 1;
        
        // 可选：使用adapted_params作为时间常数的缩放因子
        for (size_t i = 0; i < hidden_size; i++) {
            // 基于激活度调整adapted_params（可作为时间常数乘数）
            float adaptation = cell->state->adaptation_rate * 
                              (1.0f - fabsf(cell->state->activation[i]));
            cell->state->adapted_params[i] += adaptation;
            
            // 限制参数范围
            if (cell->state->adapted_params[i] < 0.1f) {
                cell->state->adapted_params[i] = 0.1f;
            } else if (cell->state->adapted_params[i] > 2.0f) {
                cell->state->adapted_params[i] = 2.0f;
            }
            
            // 可选：将adapted_params应用于时间常数
            // cell->time_constants[i] *= cell->state->adapted_params[i];
            // 但注意：这可能会与时间常数的梯度更新冲突
        }
    }
    
    return 0;
}

int cfc_cell_forward_with_dt(CfCCell* cell, const float* input, float delta_t, float* hidden_state) {
    // 参数检查
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(input, "输入向量为空");
    SELFLNN_CHECK_NULL(hidden_state, "隐藏状态缓冲区为空");
    
    // 初始化状态检查
    SELFLNN_CHECK_INITIALIZED(cell, "CfC单元未初始化");

    // 四元数CfC前向路径
    if (cell->use_quaternion) {
        return cfc_cell_forward_quaternion(cell, input, hidden_state);
    }

    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    
    // 复制输入到缓冲区
    memcpy(cell->state->input_buffer, input, input_size * sizeof(float));
    
    // 添加噪声（如果启用）
    if (cell->config.noise_std > 0.0f) {
        for (size_t i = 0; i < input_size; i++) {
            cell->state->input_buffer[i] += 
                rng_normal(0.0f, cell->config.noise_std);
        }
    }

    // 液时域缩放：若启用，使用输入依赖的动态时间常数执行CfC前向传播
    if (cell->use_liquid_scaling) {
        return cfc_cell_forward_liquid(cell, cell->state->input_buffer,
                                       hidden_state, cell->computed_liquid_tau);
    }

    // 根据求解器类型选择ODE求解方法
    // 在非闭合解求解器首次使用时按需分配工作空间
    if (cell->config.ode_solver_type != ODE_SOLVER_CLOSED_FORM) {
        if (ensure_ode_workspace(cell, cell->config.ode_solver_type) != 0) {
            return -1;
        }
    }
    
    // 如果启用了多时间尺度并行演化，使用多尺度求解器
    if (cell->use_multi_timescale) {
        cfc_multi_timescale_solution(cell,
                                    cell->state->input_buffer,
                                    cell->state->state,
                                    delta_t,
                                    cell->state->activation);
    } else if (cell->config.ode_solver_type == ODE_SOLVER_RK4) {
        cfc_cell_rk4_step(cell,
                         cell->state->input_buffer,
                         cell->state->state,
                         delta_t,
                         cell->state->activation);
    } else if (cell->config.ode_solver_type == ODE_SOLVER_RK45) {
        if (cell->use_adaptive_step) {
            cfc_cell_rk45_adaptive(cell,
                                  cell->state->input_buffer,
                                  cell->state->state,
                                  delta_t,
                                  cell->state->activation,
                                  &cell->config.adaptive_step_cfg,
                                  &cell->adaptive_step_sol);
        } else {
            cfc_cell_rk45_step(cell,
                              cell->state->input_buffer,
                              cell->state->state,
                              delta_t,
                              cell->state->activation);
        }
    } else if (cell->config.ode_solver_type == ODE_SOLVER_CTBP) {
        cfc_cell_ctbp_forward(cell,
                             cell->state->input_buffer,
                             cell->state->state,
                             delta_t,
                             cell->state->activation);
    } else if (cell->config.ode_solver_type == ODE_SOLVER_DP54) {
        if (cell->use_adaptive_step) {
            cfc_cell_dp54_adaptive(cell,
                                  cell->state->input_buffer,
                                  cell->state->state,
                                  delta_t,
                                  cell->state->activation,
                                  &cell->config.adaptive_step_cfg,
                                  &cell->adaptive_step_sol);
        } else {
            cfc_cell_dp54_step(cell,
                              cell->state->input_buffer,
                              cell->state->state,
                              delta_t,
                              cell->state->activation);
        }
    } else if (cell->config.ode_solver_type == ODE_SOLVER_ROSENBROCK) {
        if (cell->use_parallel_solve) {
            cfc_cell_rosenbrock_parallel(cell,
                                        cell->state->input_buffer,
                                        cell->state->state,
                                        delta_t,
                                        cell->state->activation,
                                        &cell->config.parallel_cfg);
        } else {
            cfc_cell_rosenbrock_step(cell,
                                    cell->state->input_buffer,
                                    cell->state->state,
                                    delta_t,
                                    cell->state->activation);
        }
    } else if (cell->config.ode_solver_type == ODE_SOLVER_SYMPLECTIC) {
        cfc_cell_symplectic_step(cell,
                                cell->state->input_buffer,
                                cell->state->state,
                                delta_t,
                                cell->state->activation);
    } else if (cell->config.ode_solver_type == ODE_SOLVER_RHS) {
        /* P2-041修复: 使用cfc_cell_compute_rhs进行直接RHS评估（前向欧拉步）
         * 作为备选数值求解器，对动态系统进行单步欧拉积分 */
        float* rhs = cell->state->noise_buffer;
        cfc_cell_compute_rhs(cell, cell->state->input_buffer,
                            cell->state->state, rhs);
        size_t hs = cell->config.hidden_size;
        for (size_t i = 0; i < hs; i++) {
            float new_h = cell->state->state[i] + delta_t * rhs[i];
            if (isnan(new_h) || isinf(new_h)) new_h = 0.0f;
            cell->state->activation[i] = new_h;
        }
    } else {
        cfc_closed_form_solution(cell,
                                cell->state->input_buffer,
                                cell->state->state,
                                delta_t,
                                cell->state->activation);
    }

    // 保存前向传播前的状态 h(t)，供反向传播使用
    memcpy(cell->state->saved_state, cell->state->state, hidden_size * sizeof(float));

    /* P2-002: 残差连接（带自定义时间步长版本） */
    if (cell->config.use_residual && cell->config.residual_scale > 0.0f) {
        for (size_t i = 0; i < hidden_size; i++) {
            cell->state->activation[i] += cell->config.residual_scale * cell->state->state[i];
            if (isnan(cell->state->activation[i]) || isinf(cell->state->activation[i])) {
                cell->state->activation[i] = 0.0f;
            }
        }
    }

    /* P1-001: 层归一化（带自定义时间步长的前向传播版本） */
    if (cell->cell_layer_norm && cell->config.use_cell_layer_norm) {
        layer_norm_forward(cell->cell_layer_norm, cell->state->activation, hidden_state, 1);
    } else {
        memcpy(hidden_state, cell->state->activation, hidden_size * sizeof(float));
    }

    // 更新内部状态（使用原始激活值，保持CfC动力学完整性）
    memcpy(cell->state->state, cell->state->activation, hidden_size * sizeof(float));

    // 更新时间，使用实际的时间步长
    cell->state->time += delta_t;
    
    // 更新统计信息
    float sum_activation = 0.0f;
    float max_activation = 0.0f;
    
    for (size_t i = 0; i < hidden_size; i++) {
        float activation = fabsf(cell->state->activation[i]);
        sum_activation += activation;
        if (activation > max_activation) {
            max_activation = activation;
        }
    }
    
    cell->avg_activation = sum_activation / hidden_size;
    cell->max_activation = max_activation;
    
    // 自适应参数更新（如果启用）
    if (cell->config.enable_adaptation) {
        // 启用时间常数自适应
        cell->use_adaptive_tau = 1;
        
        // 可选：使用adapted_params作为时间常数的缩放因子
        for (size_t i = 0; i < hidden_size; i++) {
            // 基于激活度调整adapted_params（可作为时间常数乘数）
            float adaptation = cell->state->adaptation_rate * 
                              (1.0f - fabsf(cell->state->activation[i]));
            cell->state->adapted_params[i] += adaptation;
            
            // 限制参数范围
            if (cell->state->adapted_params[i] < 0.1f) {
                cell->state->adapted_params[i] = 0.1f;
            } else if (cell->state->adapted_params[i] > 2.0f) {
                cell->state->adapted_params[i] = 2.0f;
            }
        }
    }
    
    return 0;
}

/* ============ 四元数CfC单元反向传播（P2.4） ============ */

/**
 * @brief 四元数CfC反向传播
 *
 * 计算四元数权重矩阵和偏置的梯度，以及输入梯度。
 * 梯度通过CfC闭式解的链式法则回传：d_output/d_drive = (1 - exp(-dt/τ_i))。
 * 四元数乘法梯度采用标准链式法则：dW = d_drive * input（四元数乘法）。
 *
 * @param cell CfC单元指针（需已启用use_quaternion）
 * @param gradient 损失对输出的梯度 [hidden_size]
 * @param input_gradient 损失对输入的梯度 [input_size]（可选，可为NULL）
 * @return int 成功返回0，失败返回-1
 */
static int cfc_cell_backward_quaternion(CfCCell* cell, const float* gradient,
                                         float* input_gradient)
{
    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    size_t num_hidden_quats = hidden_size / 4;
    size_t num_input_quats = input_size / 4;
    float dt = cell->config.delta_t;

    /* 梯度、输入、状态直接映射为四元数数组 */
    Quaternion* grad_quats = (Quaternion*)gradient;
    Quaternion* input_quats = (Quaternion*)cell->state->input_buffer;
    Quaternion* state_quats = (Quaternion*)cell->state->saved_state;

    Quaternion* W = (Quaternion*)cell->quaternion_weights;
    (void)W;
    Quaternion* H = (Quaternion*)cell->quaternion_hidden_weights;
    (void)H;
    float* tau = cell->quaternion_time_constants;

    /* 清零所有梯度缓冲区 */
    memset(cell->quaternion_weight_grad, 0,
           num_hidden_quats * num_input_quats * 4 * sizeof(float));
    memset(cell->quaternion_bias_grad, 0, num_hidden_quats * 4 * sizeof(float));
    memset(cell->quaternion_hidden_weight_grad, 0,
           num_hidden_quats * num_hidden_quats * 4 * sizeof(float));
    if (input_gradient) {
        memset(input_gradient, 0, input_size * sizeof(float));
    }

    Quaternion* dW = (Quaternion*)cell->quaternion_weight_grad;
    Quaternion* dB = (Quaternion*)cell->quaternion_bias_grad;
    Quaternion* dH = (Quaternion*)cell->quaternion_hidden_weight_grad;

    for (size_t i = 0; i < num_hidden_quats; i++) {
        /* CfC闭式解: output = state * exp(-dt/τ) + (1 - exp(-dt/τ)) * drive
         * d_output/d_drive = (1 - exp(-dt/τ)) （标量因子，作用于各分量）
         */
        float cf_factor = 1.0f - expf(-dt / fmaxf(tau[i], 1e-6f));
        Quaternion drive_grad = quaternion_scale(&grad_quats[i], cf_factor);

        /* 偏置梯度：dB[i] += drive_grad */
        dB[i] = quaternion_add(&dB[i], &drive_grad);

        /* 输入权重梯度：dW[i][j] += drive_grad * input[j] */
        for (size_t j = 0; j < num_input_quats; j++) {
            size_t idx = i * num_input_quats + j;
            Quaternion wg = quaternion_multiply(&drive_grad, &input_quats[j]);
            dW[idx] = quaternion_add(&dW[idx], &wg);
        }

        /* 隐藏权重梯度：dH[i][k] += drive_grad * state[k] */
        for (size_t k = 0; k < num_hidden_quats; k++) {
            size_t hidx = i * num_hidden_quats + k;
            Quaternion hg = quaternion_multiply(&drive_grad, &state_quats[k]);
            dH[hidx] = quaternion_add(&dH[hidx], &hg);
        }
    }

    /* 输入梯度：d_input[j] += Σ_i conj(W[i][j]) * drive_grad[i] */
    if (input_gradient) {
        Quaternion* d_input_quats = (Quaternion*)input_gradient;
        for (size_t j = 0; j < num_input_quats; j++) {
            Quaternion dq = {0, 0, 0, 0};
            for (size_t i = 0; i < num_hidden_quats; i++) {
                size_t idx = i * num_input_quats + j;
                float cf_factor = 1.0f - expf(-dt / fmaxf(tau[i], 1e-6f));
                Quaternion drive_grad = quaternion_scale(&grad_quats[i], cf_factor);
                Quaternion w_conj = quaternion_conjugate(&W[idx]);
                Quaternion contrib = quaternion_multiply(&w_conj, &drive_grad);
                dq = quaternion_add(&dq, &contrib);
            }
            d_input_quats[j] = dq;
        }
    }

    return 0;
}

/**
 * @brief 反向传播（训练）
 */
int cfc_cell_backward(CfCCell* cell, const float* gradient, float* input_gradient) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(gradient, "梯度向量为空");
    SELFLNN_CHECK_INITIALIZED(cell, "CfC单元未初始化");
    CHECK_CFC_STATE_INT(cell);

    /* 神经ODE伴随法路径：使用连续伴随法计算精确梯度，更高效的O(1)内存 */
    if (cell->use_neural_ode_adjoint) {
        return cfc_cell_backward_adjoint(cell, gradient, input_gradient);
    }

    /* 门控CfC变体反向传播 */
    if (cell->gated_data) {
        return cfc_cell_backward_gated(cell, gradient, input_gradient);
    }

    /* 分层CfC反向传播 */
    if (cell->hierarchical_data) {
        return cfc_cell_backward_hierarchical(cell, gradient, input_gradient);
    }

    /* 液态记忆门控CfC反向传播 */
    if (cell->liquid_memory_data) {
        return cfc_cell_backward_liquid_memory(cell, gradient, input_gradient);
    }

    if (!cell->state->saved_state) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "CfC单元saved_state为空，请先执行前向传播");
        return -1;
    }
    
    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    
    // 获取前向传播时保存的输入和状态
    const float* input = cell->state->input_buffer;
    const float* prev_state = cell->state->saved_state;  // 从保存缓冲区读取前向传播前的状态 h(t)
    
    // 复制梯度到内部缓冲区
    memcpy(cell->state->gradient, gradient, hidden_size * sizeof(float));
    
    /* P1-001: 层归一化反向传播——将经过层归一化的梯度转换为原始隐藏状态梯度
     * 前向传播时: output = LayerNorm(raw_hidden_state)
     * 反向传播时: ∂L/∂raw_hidden = LayerNorm_backward(∂L/∂output)
     * 我们需要先计算 ∂L/∂raw_hidden_state，然后用该梯度做标准CfC反向传播 */
    if (cell->cell_layer_norm && cell->config.use_cell_layer_norm) {
        float* raw_hidden_grad = (float*)safe_malloc(hidden_size * sizeof(float));
        if (raw_hidden_grad) {
            layer_norm_backward(cell->cell_layer_norm, cell->state->gradient, raw_hidden_grad);
            memcpy(cell->state->gradient, raw_hidden_grad, hidden_size * sizeof(float));
            safe_free((void**)&raw_hidden_grad);
        }
    }
    
    // 计算时间步长和预计算项
    float delta_t = cell->config.delta_t;
    size_t weight_size = input_size * hidden_size;
    size_t hh_size = hidden_size * hidden_size;

    /* P0-002深度修复: 移除所有内部梯度清零memset。
     * 梯度写入改用+=（累积模式），由调用方负责在合适的时机清零。
     * 
     * 批量训练: 调用方在批次开始前清零一次，样本循环中+=累积，
     *           批次结束时统一下发 cfc_apply_cell_gradients。
     * 单样本:   cfc_backward_ex(skip=0) 在层循环前清零一次。
     * CTBP:     cfc_cell_backward_ctbp 在时间步循环前清零一次。
     * 
     * 注意: 移除内部清零后，未启用的梯度缓冲区可能残留旧值。
     * 安全措施: cfc_apply_cell_gradients 始终使用 isfinite() 检查。 */
    
    // 如果需要，清零输入梯度
    if (input_gradient) {
        memset(input_gradient, 0, input_size * sizeof(float));
    }
    
    // 门控权重梯度（W_gx, W_gh）和偏置梯度（b_g）
    // 激活权重梯度（W_ax, W_ah）和偏置梯度（b_a）
    // 注意：我们重用现有的梯度缓冲区：
    // - weight_grad: dL/dW_ax (输入到激活的权重梯度)
    // - bias_grad: dL/db_a (激活偏置梯度)
    // 需要额外的缓冲区用于门控权重梯度（暂时使用现有缓冲区，稍后存储）
    
    for (size_t i = 0; i < hidden_size; i++) {
        // 重新计算前向传播的中间变量（为了梯度计算）
        float gate_input_sum = cell->gate_biases[i * 3];  // b_g (门控偏置)
        float activation_input_sum = cell->bias_vector[i];  // b_a
        
        // 输入到隐藏连接：W_gx * x 和 W_ax * x
        for (size_t j = 0; j < input_size; j++) {
            size_t idx = i * input_size + j;
            gate_input_sum += cell->input_gate_weights[idx] * input[j];
            activation_input_sum += cell->weight_matrix[idx] * input[j];
        }
        
        // 隐藏到隐藏连接：W_gh * h 和 W_ah * h（CfC论文标准实现）
        for (size_t j = 0; j < hidden_size; j++) {
            size_t h_idx = i * hidden_size + j;
            gate_input_sum += cell->hidden_to_gate_weights[h_idx] * prev_state[j];
            activation_input_sum += cell->hidden_to_activation_weights[h_idx] * prev_state[j];
        }
        
        // 计算门控和激活的导数（ZSFZS-F023: 添加截断保护，防止expf溢出）
        float gate_input_clamped = gate_input_sum;
        if (gate_input_clamped > 10.0f) gate_input_clamped = 10.0f;
        else if (gate_input_clamped < -10.0f) gate_input_clamped = -10.0f;
        float gate = 1.0f / (1.0f + expf(-gate_input_clamped));
        float gate_derivative = gate * (1.0f - gate);  // σ'(x) = σ(x) * (1 - σ(x))
        
        float act_input_clamped = activation_input_sum;
        if (act_input_clamped > 10.0f) act_input_clamped = 10.0f;
        else if (act_input_clamped < -10.0f) act_input_clamped = -10.0f;
        float activation = tanhf(act_input_clamped);
        float activation_derivative = 1.0f - activation * activation;  // tanh'(x) = 1 - tanh²(x)
        
        // P1-019修复: 获取前向传播实际使用的时间常数
        // 多时间尺度、液时域缩放等求解器使用与config不同的tau值
        // 优先从forward_tau_used读取，确保反向传播与前向传播的exp项系数一致
        float tau;
        if (cell->forward_tau_used && cell->forward_tau_used[i] > 0.0f) {
            tau = cell->forward_tau_used[i];
        } else if (cell->use_adaptive_tau && cell->time_constants) {
            tau = cell->time_constants[i];
        } else {
            tau = cell->config.time_constant;
        }
        if (tau < cell->min_time_constant) tau = cell->min_time_constant;
        if (tau > cell->max_time_constant) tau = cell->max_time_constant;
        
        float exp_term = expf(-delta_t / tau);
        float driver_coeff = 1.0f - exp_term;  // (1 - exp(-Δt/τ))
        
        // 计算驱动项对h_new的贡献
        float driver = gate * activation;
        
        // 计算从h_new到各个中间变量的梯度
        float dL_dh_new = cell->state->gradient[i];  // 来自上层的梯度
        
        float effective_dL_dh_new = dL_dh_new;
        /* P2-002-审计修复: 移除 0.5x 启发式缩放因子。
         * 残差连接的梯度应通过 cfc_cell_temporal_backward 的时间雅可比
         * 正确传递到上一步的状态，而非注入 ODE 驱动项梯度。
         *
         * ∂h_res/∂h_old = ∂h_raw/∂h_old + α·I (α = residual_scale)
         * 时间反向传播全量传递:
         *   g_prev = g * (exp_term + α) + W_gh^T·v_gh + W_ah^T·v_ah
         * ODE 驱动项梯度路径仅使用原始 dL/dh_new:
         *   ∂L/∂driver = dL/dh_new · (1-exp(-Δt/τ))
         * 残散梯度由 temporal_backward 单独处理。
         *
         * FIX-013 兼容: effective_dL_dh_new 仍保留变量名，
         * 其值现等于 dL_dh_new（原 FIX-013 路径曾完全丢弃残差梯度，
         * 现已通过 temporal_backward 正确恢复）。 */
        (void)cell->config.residual_scale; /* 残差标记，by temporal_backward */
        
        /* 审计修复（原 FIX-013）: τ梯度使用原始 dL_dh_new
         * （残差连接不依赖 τ，故 τ 梯度无需残差修正）。
         * effective_dL_dh_new 现等价于 dL_dh_new，v4.0 审计回归为纯数学梯度。 */
        float dL_ddriver = effective_dL_dh_new * driver_coeff;
        
        // driver对gate和activation的导数
        float dL_dgate = dL_ddriver * activation;
        float dL_dactivation = dL_ddriver * gate;
        
        // 通过gate和activation传播到它们的输入
        float dL_dgate_input = dL_dgate * gate_derivative;
        float dL_dactivation_input = dL_dactivation * activation_derivative;
        
        // ====== 计算输入到隐藏权重梯度 ======
        for (size_t j = 0; j < input_size; j++) {
            size_t idx = i * input_size + j;
            
            // W_gx梯度：dL/dW_gx = dL/dgate_input * input[j]
            cell->input_gate_weight_grad[idx] += dL_dgate_input * input[j];
            
            // W_ax梯度：dL/dW_ax = dL/dactivation_input * input[j]
            cell->weight_grad[idx] += dL_dactivation_input * input[j];
            
            // 输入梯度（如果需要）：dL/dx_j = Σ_i(dL/dgate_input_i * W_gx[i][j] + dL/dactivation_input_i * W_ax[i][j])
            if (input_gradient) {
                input_gradient[j] += dL_dgate_input * cell->input_gate_weights[idx] +
                                    dL_dactivation_input * cell->weight_matrix[idx];
            }
        }
        
        // ====== 计算隐藏到隐藏权重梯度 ======
        for (size_t j = 0; j < hidden_size; j++) {
            size_t h_idx = i * hidden_size + j;
            
            // W_gh梯度：dL/dW_gh = dL/dgate_input * prev_state[j]
            cell->hidden_to_gate_weight_grad[h_idx] += dL_dgate_input * prev_state[j];
            
            // W_ah梯度：dL/dW_ah = dL/dactivation_input * prev_state[j]
            cell->hidden_to_activation_weight_grad[h_idx] += dL_dactivation_input * prev_state[j];
        }
        
        // 偏置梯度
        // b_g梯度：dL/db_g = dL/dgate_input
        cell->gate_bias_grad[i * 3] += dL_dgate_input;
        
        // b_a梯度：dL/db_a = dL/dactivation_input
        cell->bias_grad[i] += dL_dactivation_input;
        
        // 时间常数梯度（如果启用自适应时间常数）
        if (cell->use_adaptive_tau) {
            /* P0-002修复：数值稳定的时间常数梯度计算
             * ∂h_new/∂τ = h_old * (Δt/τ²) * exp(-Δt/τ) - driver * (Δt/τ²) * exp(-Δt/τ)
             * = (Δt/τ²) * exp(-Δt/τ) * (h_old - driver)
             * 
             * 当 τ 很小时，Δt/τ² 极大而 exp(-Δt/τ) 极小，乘积容易下溢/上溢。
             * 使用钳位保护数值稳定性。 */
            float safe_dt = fmaxf(delta_t, 1e-8f);
            float safe_tau_sq = fmaxf(tau * tau, 1e-12f);
            float ratio = safe_dt / safe_tau_sq;
            float stable_coeff = fminf(ratio, 1e6f) * exp_term;
            float dL_dtau = dL_dh_new * stable_coeff * (prev_state[i] - driver);
            /* 梯度钳位防止爆炸 */
            if (fabsf(dL_dtau) > 1e4f) dL_dtau = copysignf(1e4f, dL_dtau);
            cell->time_constant_grad[i] += dL_dtau;
        }
    }
    
    return 0;
}

/* ========================================================================
 * P0-BPTT: 时间反向传播 — 计算梯度通过CfC状态转移的回传
 *
 * CfC状态转移（逐神经元独立时间常数τ_i）:
 *   h_new_i = h_old_i · exp(-Δt/τ_i) + (1-exp(-Δt/τ_i)) · σ(g_i) · tanh(a_i)
 *
 * 定义:
 *   α_i = exp(-Δt/τ_i),  β_i = 1 - α_i
 *   G_i = σ(W_gh_i·h_old + W_gx_i·x + b_g_i)          门控值
 *   A_i = tanh(W_ah_i·h_old + W_ax_i·x + b_a_i)       激活值
 *   G'_i = G_i·(1 - G_i)                               门控导数
 *   A'_i = 1 - A_i·A_i                                 激活导数
 *
 * 时间雅可比向量积（纯状态到状态传递，不累加参数梯度）:
 *   g_out[j] = α_j·g_j + (W_gh^T · v_gh)[j] + (W_ah^T · v_ah)[j]
 *              [+ residual_scale·g_j  如果残差连接启用]
 *
 * 其中加权向量:
 *   v_gh_i = β_i · g_i · G'_i · A_i
 *   v_ah_i = β_i · g_i · G_i · A'_i
 *
 * 重要: 此函数仅计算时间梯度传递。不累积任何参数梯度！
 * W_gh/W_ah/W_gx/W_ax/b_g/b_a/τ的参数梯度由 cfc_cell_backward 在
 * BPTT循环的每步中通过combo_grad正确计算（combo_grad已包含时间链贡献）。
 * 在此处重复累积参数梯度将导致双计，使W_gh/W_ah学习率翻倍。
 * ======================================================================== */
int cfc_cell_temporal_backward(CfCCell* cell, const float* combined_gradient,
                                float* temporal_gradient_out) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(combined_gradient, "组合梯度向量为空");
    SELFLNN_CHECK_NULL(temporal_gradient_out, "时间梯度输出为空");
    SELFLNN_CHECK_INITIALIZED(cell, "CfC单元未初始化");
    CHECK_CFC_STATE_INT(cell);

    if (!cell->state->saved_state) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "CfC单元saved_state为空，BPTT需要前向传播保存的状态");
        return -1;
    }

    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    const float* input = cell->state->input_buffer;
    const float* prev_state = cell->state->saved_state;
    float delta_t = cell->config.delta_t;

    /* v_gh存入gradient, v_ah存入adapted_params */
    float* v_gh = cell->state->gradient;
    float* v_ah = cell->state->adapted_params;

    /* ====== 第一步: 计算前向变量 + α·I项 + 加权向量v_gh/v_ah + W_gh/W_ah/b_g时间梯度 ====== */
    for (size_t i = 0; i < hidden_size; i++) {
        float gate_input_sum = cell->gate_biases[i * 3];
        float activ_input_sum = cell->bias_vector[i];
        for (size_t j = 0; j < input_size; j++) {
            size_t idx = i * input_size + j;
            gate_input_sum += cell->input_gate_weights[idx] * input[j];
            activ_input_sum += cell->weight_matrix[idx] * input[j];
        }
        for (size_t j = 0; j < hidden_size; j++) {
            size_t h_idx = i * hidden_size + j;
            gate_input_sum += cell->hidden_to_gate_weights[h_idx] * prev_state[j];
            activ_input_sum += cell->hidden_to_activation_weights[h_idx] * prev_state[j];
        }

        float gate = 1.0f / (1.0f + expf(-gate_input_sum));
        float activation = tanhf(activ_input_sum);
        float gate_deriv = gate * (1.0f - gate);
        float activ_deriv = 1.0f - activation * activation;

        float tau;
        if (cell->forward_tau_used && cell->forward_tau_used[i] > 0.0f) {
            tau = cell->forward_tau_used[i];
        } else if (cell->use_adaptive_tau && cell->time_constants) {
            tau = cell->time_constants[i];
        } else {
            tau = cell->config.time_constant;
        }
        if (tau < cell->min_time_constant) tau = cell->min_time_constant;
        if (tau > cell->max_time_constant) tau = cell->max_time_constant;

        float dt_over_tau = delta_t / tau;
        if (dt_over_tau < 1e-8f) dt_over_tau = 1e-8f;
        if (dt_over_tau > 100.0f) dt_over_tau = 100.0f;

        float exp_term;
        if (dt_over_tau > 20.0f) exp_term = 0.0f;
        else if (dt_over_tau < 1e-4f) exp_term = 1.0f - dt_over_tau;
        else exp_term = expf(-dt_over_tau);

        float beta = 1.0f - exp_term;

        /* α·I项: temporal_out[i] = α_i · g_i [+ residual_scale · g_i]
         * 残差连接: h_new = h_raw + α·h_old
         *   ∂h_new/∂h_old = ∂h_raw/∂h_old + α·I
         *   残差贡献直接加到 α·I 项中 */
        float alpha_total = exp_term;
        if (cell->config.use_residual && cell->config.residual_scale > 0.0f) {
            alpha_total += cell->config.residual_scale;
        }
        temporal_gradient_out[i] = alpha_total * combined_gradient[i];

        /* 构建加权向量(完整包含β_i和所有导数):
         * v_gh_i = β_i · g_i · G'_i · A_i
         * v_ah_i = β_i · g_i · G_i · A'_i */
        float common = beta * combined_gradient[i];
        v_gh[i] = common * gate_deriv * activation;
        v_ah[i] = common * gate * activ_deriv;
    }

    /* ====== 第二步: 矩阵-向量积 W_gh^T·v_gh + W_ah^T·v_ah ====== */
    for (size_t i = 0; i < hidden_size; i++) {
        float sum_gh = 0.0f;
        float sum_ah = 0.0f;
        for (size_t j = 0; j < hidden_size; j++) {
            size_t h_idx = j * hidden_size + i;
            sum_gh += cell->hidden_to_gate_weights[h_idx] * v_gh[j];
            sum_ah += cell->hidden_to_activation_weights[h_idx] * v_ah[j];
        }

        /* =  (W_gh^T·v_gh)[i] + (W_ah^T·v_ah)[i]
         * 注意: 无额外乘数！β_i/G'_i/A'_i已在v_gh/v_ah内部 */
        temporal_gradient_out[i] += sum_gh + sum_ah;

        if (isnan(temporal_gradient_out[i]) || isinf(temporal_gradient_out[i])) {
            temporal_gradient_out[i] = 0.0f;
        }
    }

    return 0;
}

/**
 * @brief CfC单元截断时间反向传播 (Truncated BPTT)
 * 
 * 在给定时间窗口 [0, seq_len) 上执行BPTT。
 * 
 * 调用方需在调用前执行完整的前向传播序列。对每步 t:
 *   - 将 h(t) 保存
 *   - cell->state->saved_state 自然记录了 h(t-1)【前向后快照】
 *   - 故 seq_prev_states[t] 可直接使用 saved_state 的快照
 *   - seq_inputs[t] = x(t)
 *   - seq_forward_tau[t] = 对 cell->forward_tau_used 的快照（逐时间步 τ）
 * 
 * 反向循环从 t=seq_len-1 到 0:
 *   1. 恢复 seq_prev_states[t] → cell->state->saved_state
 *   2. 恢复 seq_inputs[t] → cell->state->input_buffer
 *   3. 恢复 seq_forward_tau + t*hidden → cell->forward_tau_used
 *   4. cfc_cell_backward(combo_grad) — 读 forward_tau_used/saved_state/input_buffer
 *   5. cfc_cell_temporal_backward(combo_grad) — 同样读这些缓冲区
 * 
 * @param cell CfC单元句柄
 * @param loss_gradients 每时间步的损失梯度 [seq_len * hidden_size]
 * @param input_gradients 输出每步输入梯度 [seq_len * input_size]
 * @param seq_len 序列长度
 * @param seq_inputs 每步输入 x(t) [seq_len * input_size]
 * @param seq_prev_states 每步前向状态 h(t-1) [seq_len * hidden_size]
 * @param seq_forward_tau 每步前向使用的 τ [seq_len * hidden_size]，可为NULL（从 time_constants 推算）
 * @return 0成功，负值失败
 */
int cfc_cell_backward_bptt(CfCCell* cell, const float* loss_gradients,
                            float* input_gradients, int seq_len,
                            const float* seq_inputs,
                            const float* seq_prev_states,
                            const float* seq_forward_tau) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(loss_gradients, "损失梯度为空");
    SELFLNN_CHECK_NULL(input_gradients, "输入梯度缓冲区为空");
    SELFLNN_CHECK_NULL(seq_inputs, "序列输入为空");
    SELFLNN_CHECK_NULL(seq_prev_states, "序列前向状态为空");
    SELFLNN_CHECK_INITIALIZED(cell, "CfC单元未初始化");
    SELFLNN_CHECK(seq_len > 0, SELFLNN_ERROR_INVALID_ARGUMENT,
                 "序列长度必须>0，当前值=%d", seq_len);

    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;

    float* temporal_grad = (float*)safe_calloc(hidden_size, sizeof(float));
    if (!temporal_grad) return SELFLNN_ERROR_OUT_OF_MEMORY;

    for (int t = seq_len - 1; t >= 0; t--) {
        /* P0-BPTT-审计修复: 三步全量恢复每时间步的上下文缓冲区 */
        memcpy(cell->state->saved_state,
               seq_prev_states + t * hidden_size,
               hidden_size * sizeof(float));
        memcpy(cell->state->input_buffer,
               seq_inputs + t * input_size,
               input_size * sizeof(float));
        /* forward_tau_used 记录前向传播时每神经元独立使用的 τ 值。
         * cfc_cell_backward(L2647) 和 cfc_cell_temporal_backward(L2822)
         * 均从 forward_tau_used 读取 τ 以确保前向/反向 exp(-Δt/τ) 一致。
         * BPTT 循环必须为每步恢复该时间步的 τ 快照。 */
        if (seq_forward_tau && cell->forward_tau_used) {
            memcpy(cell->forward_tau_used,
                   seq_forward_tau + t * hidden_size,
                   hidden_size * sizeof(float));
        }

        const float* step_loss_grad = loss_gradients + t * hidden_size;
        float* step_input_grad = input_gradients + t * input_size;

        /* 组合梯度 = 直接损失梯度 + 来自未来的时间梯度 */
        float* combo_grad = cell->state->gradient;
        for (size_t i = 0; i < hidden_size; i++) {
            combo_grad[i] = step_loss_grad[i] + temporal_grad[i];
        }

        /* 执行标准反向传播（累积cell参数梯度，使用+=模式） */
        memset(step_input_grad, 0, input_size * sizeof(float));
        int ret = cfc_cell_backward(cell, combo_grad, step_input_grad);
        if (ret != 0) {
            safe_free((void**)&temporal_grad);
            return ret;
        }

        /* 传播时间梯度到上一步（如果还有更早的时间步） */
        if (t > 0) {
            ret = cfc_cell_temporal_backward(cell, combo_grad, temporal_grad);
            if (ret != 0) {
                safe_free((void**)&temporal_grad);
                return ret;
            }
            /* ZSFZS-F030修复: BPTT梯度裁剪，防止长序列梯度爆炸 */
            float norm_sq = 0.0f;
            for (size_t i = 0; i < hidden_size; i++)
                norm_sq += temporal_grad[i] * temporal_grad[i];
            float norm = sqrtf(norm_sq);
            if (norm > 10.0f) {
                float scale = 10.0f / norm;
                for (size_t i = 0; i < hidden_size; i++)
                    temporal_grad[i] *= scale;
            }
        }
    }

    safe_free((void**)&temporal_grad);
    return 0;
}

/**
 * @brief 重置CfC单元状态
 */
void cfc_cell_reset(CfCCell* cell) {
    if (!cell || !cell->is_initialized) {
        return;
    }
    
    size_t hidden_size = cell->config.hidden_size;
    
    // 重置状态向量
    for (size_t i = 0; i < hidden_size; i++) {
        cell->state->state[i] = random_uniform_seeded(-0.01f, 0.01f, (unsigned int)(uintptr_t)cell ^ (unsigned int)i ^ 0x5000);
        cell->state->adapted_params[i] = 1.0f;
    }
    
    // 重置时间
    cell->state->time = 0.0f;
    
    // 重置统计信息
    cell->avg_activation = 0.0f;
    cell->max_activation = 0.0f;
    
    // 重置梯度缓冲区
    memset(cell->weight_grad, 0, hidden_size * cell->config.input_size * sizeof(float));
    memset(cell->bias_grad, 0, hidden_size * sizeof(float));
    memset(cell->state->gradient, 0, hidden_size * sizeof(float));
    memset(cell->input_gate_weight_grad, 0, hidden_size * cell->config.input_size * sizeof(float));
    memset(cell->output_gate_weight_grad, 0, hidden_size * cell->config.input_size * sizeof(float));
    memset(cell->forget_gate_weight_grad, 0, hidden_size * cell->config.input_size * sizeof(float));
    memset(cell->gate_bias_grad, 0, hidden_size * 3 * sizeof(float));
    memset(cell->time_constant_grad, 0, hidden_size * sizeof(float));
    memset(cell->hidden_to_gate_weight_grad, 0, hidden_size * hidden_size * sizeof(float));
    memset(cell->hidden_to_activation_weight_grad, 0, hidden_size * hidden_size * sizeof(float));
    
    // 重置多时间尺度并行演化状态
    if (cell->fast_state && cell->slow_state) {
        for (size_t i = 0; i < hidden_size; i++) {
            cell->fast_state[i] = 0.0f;
            cell->slow_state[i] = 0.0f;
            if (cell->fast_time_constants && cell->slow_time_constants) {
                float base_tau = cell->time_constants[i];
                cell->fast_time_constants[i] = base_tau * cell->fast_tau_ratio;
                cell->slow_time_constants[i] = base_tau * cell->slow_tau_ratio;
                if (cell->fast_time_constants[i] < 0.001f) cell->fast_time_constants[i] = 0.001f;
                if (cell->slow_time_constants[i] > 100.0f) cell->slow_time_constants[i] = 100.0f;
            }
        }
    }
}

/**
 * @brief 获取CfC单元配置
 */
int cfc_cell_get_config(const CfCCell* cell, CfCCellConfig* config) {
    // 参数检查
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(config, "配置输出缓冲区为空");
    
    memcpy(config, &cell->config, sizeof(CfCCellConfig));
    return 0;
}

/**
 * @brief 设置CfC单元配置
 */
int cfc_cell_set_config(CfCCell* cell, const CfCCellConfig* config) {
    // 参数检查
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(config, "配置指针为空");
    
    // 验证配置
    SELFLNN_CHECK(config->input_size == cell->config.input_size &&
                 config->hidden_size == cell->config.hidden_size,
                 SELFLNN_ERROR_NETWORK_CONFIG,
                 "CfC单元配置尺寸不匹配: 新(input=%zu,hidden=%zu) != 当前(input=%zu,hidden=%zu)",
                 config->input_size, config->hidden_size,
                 cell->config.input_size, cell->config.hidden_size);
    
    // 更新配置
    cell->config.time_constant = config->time_constant;
    cell->config.noise_std = config->noise_std;
    cell->config.enable_adaptation = config->enable_adaptation;
    cell->config.ode_solver_type = config->ode_solver_type;
    
    /* P1-020修复: 多时间尺度并行演化配置传递
     * cfc_configure_multi_rate通过cfc_cell_set_config更新多时间尺度字段，
     * 若此处不更新，change_multi_timescale设置将丢失 */
    cell->config.use_multi_timescale = config->use_multi_timescale;
    cell->config.fast_tau_ratio = config->fast_tau_ratio;
    cell->config.slow_tau_ratio = config->slow_tau_ratio;
    cell->use_multi_timescale = config->use_multi_timescale;
    cell->fast_tau_ratio = config->fast_tau_ratio;
    cell->slow_tau_ratio = config->slow_tau_ratio;
    
    /* P1-020修复: 时间步长和自适应步长配置传递 */
    cell->config.delta_t = config->delta_t;
    cell->config.delta_t_explicitly_set = config->delta_t_explicitly_set;  /* P2-042修复: 传递显式设置标志 */
    cell->config.use_adaptive_step = config->use_adaptive_step;
    cell->use_adaptive_step = config->use_adaptive_step;
    cell->config.use_parallel_solve = config->use_parallel_solve;
    cell->use_parallel_solve = config->use_parallel_solve;
    
    return 0;
}

/**
 * @brief 设置ODE求解器类型
 */
int cfc_cell_set_solver_type(CfCCell* cell, int solver_type) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_INITIALIZED(cell, "CfC单元未初始化");
    
    if (solver_type != ODE_SOLVER_CLOSED_FORM && solver_type != ODE_SOLVER_RK4 &&
        solver_type != ODE_SOLVER_RK45 && solver_type != ODE_SOLVER_CTBP &&
        solver_type != ODE_SOLVER_DP54 && solver_type != ODE_SOLVER_ROSENBROCK &&
        solver_type != ODE_SOLVER_SYMPLECTIC) {
        return -1;
    }
    
    cell->config.ode_solver_type = solver_type;
    return 0;
}

/**
 * @brief 获取当前ODE求解器类型
 */
int cfc_cell_get_solver_type(const CfCCell* cell) {
    if (!cell || !cell->is_initialized) {
        return -1;
    }
    return cell->config.ode_solver_type;
}

/**
 * @brief 获取CfC单元状态信息
 */
int cfc_cell_get_stats(const CfCCell* cell, float* avg_activation,
                       float* max_activation, float* adaptation_rate) {
    // 参数检查
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    
    if (avg_activation) {
        *avg_activation = cell->avg_activation;
    }
    
    if (max_activation) {
        *max_activation = cell->max_activation;
    }
    
    if (adaptation_rate) {
        *adaptation_rate = cell->state->adaptation_rate;
    }
    
    return 0;
}

/* ============================================================================
 * RK45自适应步长ODE求解器
 * 
 * 龙格-库塔-费尔伯格法（RK4(5)）使用6个斜率评估实现4阶和5阶精度，
 * 通过比较两阶结果自动调整步长以控制局部截断误差。
 * 
 * Butcher Tableau系数 (Fehlberg's RK4(5)):
 *   0       |
 *   1/4     | 1/4
 *   3/8     | 3/32      9/32
 *   12/13   | 1932/2197  -7200/2197  7296/2197
 *   1       | 439/216    -8          3680/513   -845/4104
 *   1/2     | -8/27      2           -3544/2565 1859/4104  -11/40
 *   --------+--------------------------------------------------------
 *   4阶     | 25/216     0          1408/2565  2197/4104  -1/5      0
 *   5阶     | 16/135     0          6656/12825 28561/56430 -9/50    2/55
 * =========================================================================== */

/* RK45 Butcher tableau 系数 */
#define RK45_A21 0.25f
#define RK45_A31 0.09375f         /* 3/32 */
#define RK45_A32 0.28125f         /* 9/32 */
#define RK45_A41 0.879380974f     /* 1932/2197 */
#define RK45_A42 -3.277196176f    /* -7200/2197 */
#define RK45_A43 3.320892126f     /* 7296/2197 */
#define RK45_A51 2.032407407f     /* 439/216 */
#define RK45_A52 -8.0f
#define RK45_A53 7.173489279f     /* 3680/513 */
#define RK45_A54 -0.205896686f    /* -845/4104 */
#define RK45_A61 -0.296296296f    /* -8/27 */
#define RK45_A62 2.0f
#define RK45_A63 -1.381676413f    /* -3544/2565 */
#define RK45_A64 0.452972694f     /* 1859/4104 */
#define RK45_A65 -0.275f          /* -11/40 */

/* 4阶精度权重（用于误差估计） */
#define RK45_B4_1 0.115740741f    /* 25/216 */
#define RK45_B4_2 0.0f
#define RK45_B4_3 0.548927875f    /* 1408/2565 */
#define RK45_B4_4 0.535331384f    /* 2197/4104 */
#define RK45_B4_5 -0.2f           /* -1/5 */

/* 5阶精度权重（用于状态更新） */
#define RK45_B5_1 0.118518519f    /* 16/135 */
#define RK45_B5_2 0.0f
#define RK45_B5_3 0.518986504f    /* 6656/12825 */
#define RK45_B5_4 0.506131490f    /* 28561/56430 */
#define RK45_B5_5 -0.18f          /* -9/50 */
#define RK45_B5_6 0.036363636f    /* 2/55 */

/**
 * @brief 使用RK45自适应步长求解CfC ODE（内部函数）
 * 
 * 算法步骤：
 * 1. 使用6个Butcher系数计算k1~k6斜率
 * 2. 分别用4阶和5阶权重计算状态更新
 * 3. 计算局部截断误差估计
 * 4. 根据误差自动调整步长
 * 5. 如果误差超过容限，降低步长并重试
 * 
 * @param cell CfC单元句柄
 * @param input 输入向量
 * @param prev_state 前一状态
 * @param delta_t 总时间步长
 * @param output 输出缓冲区
 */
static void cfc_cell_rk45_step(CfCCell* cell, const float* input,
                               const float* prev_state, float delta_t, float* output) {
    size_t hidden_size = cell->config.hidden_size;
    RK45Config* cfg = &cell->config.rk45_config;
    
    float* k1 = cell->state->rk45_k1;
    float* k2 = cell->state->rk45_k2;
    float* k3 = cell->state->rk45_k3;
    float* k4 = cell->state->rk45_k4;
    float* k5 = cell->state->rk45_k5;
    float* k6 = cell->state->rk45_k6;
    float* temp = cell->state->rk45_temp;
    float* state_4th = output;     /* 5阶结果写入output作为初始 */
    
    float remaining = delta_t;
    float current_step;
    float rel_tol = (cfg->rel_tolerance > 0.0f) ? cfg->rel_tolerance : 1e-6f;
    float abs_tol = (cfg->abs_tolerance > 0.0f) ? cfg->abs_tolerance : 1e-8f;
    float min_step = (cfg->min_step_size > 0.0f) ? cfg->min_step_size : 1e-6f;
    float max_step = (cfg->max_step_size > 0.0f) ? cfg->max_step_size : delta_t;
    float safety = (cfg->safety_factor > 0.0f) ? cfg->safety_factor : 0.9f;
    int max_iter = (cfg->max_iterations > 0) ? cfg->max_iterations : 10000;
    
    /* 初始步长设为总步长 */
    current_step = (remaining < max_step) ? remaining : max_step;
    
    /* 当前状态从prev_state开始 */
    memcpy(temp, prev_state, hidden_size * sizeof(float));
    
    int iterations = 0;
    while (remaining > 1e-12f && iterations < max_iter) {
        iterations++;
        
        /* 计算k1: f(t_n, h_n) */
        cfc_cell_compute_rhs(cell, input, temp, k1);
        
        /* 计算k2: f(t_n + c2*h, h_n + h*a21*k1) */
        for (size_t i = 0; i < hidden_size; i++) {
            cell->state->noise_buffer[i] = temp[i] + current_step * RK45_A21 * k1[i];
        }
        cfc_cell_compute_rhs(cell, input, cell->state->noise_buffer, k2);
        
        /* 计算k3: f(t_n + c3*h, h_n + h*(a31*k1 + a32*k2)) */
        for (size_t i = 0; i < hidden_size; i++) {
            cell->state->noise_buffer[i] = temp[i] + current_step * (RK45_A31 * k1[i] + RK45_A32 * k2[i]);
        }
        cfc_cell_compute_rhs(cell, input, cell->state->noise_buffer, k3);
        
        /* 计算k4 */
        for (size_t i = 0; i < hidden_size; i++) {
            cell->state->noise_buffer[i] = temp[i] + current_step * (RK45_A41 * k1[i] + RK45_A42 * k2[i] + RK45_A43 * k3[i]);
        }
        cfc_cell_compute_rhs(cell, input, cell->state->noise_buffer, k4);
        
        /* 计算k5 */
        for (size_t i = 0; i < hidden_size; i++) {
            cell->state->noise_buffer[i] = temp[i] + current_step * (RK45_A51 * k1[i] + RK45_A52 * k2[i] + RK45_A53 * k3[i] + RK45_A54 * k4[i]);
        }
        cfc_cell_compute_rhs(cell, input, cell->state->noise_buffer, k5);
        
        /* 计算k6 */
        for (size_t i = 0; i < hidden_size; i++) {
            cell->state->noise_buffer[i] = temp[i] + current_step * (RK45_A61 * k1[i] + RK45_A62 * k2[i] + RK45_A63 * k3[i] + RK45_A64 * k4[i] + RK45_A65 * k5[i]);
        }
        cfc_cell_compute_rhs(cell, input, cell->state->noise_buffer, k6);
        
        /* 计算4阶和5阶状态更新 */
        float max_error = 0.0f;
        float max_scale = 1e-10f;
        
        for (size_t i = 0; i < hidden_size; i++) {
            float h_step = current_step;
            
            /* 4阶更新（用于误差估计） */
            float y4 = temp[i] + h_step * (RK45_B4_1 * k1[i] + RK45_B4_3 * k3[i] + RK45_B4_4 * k4[i] + RK45_B4_5 * k5[i]);
            
            /* 5阶更新（用于实际推进） */
            float y5 = temp[i] + h_step * (RK45_B5_1 * k1[i] + RK45_B5_3 * k3[i] + RK45_B5_4 * k4[i] + RK45_B5_5 * k5[i] + RK45_B5_6 * k6[i]);
            
            /* 保存5阶结果 */
            state_4th[i] = y5;
            
            /* 计算加权误差：error = |y5 - y4| / (atol + rtol * max(|y5|, |y4|)) */
            float abs_y = fabsf(y5);
            if (abs_y < fabsf(y4)) abs_y = fabsf(y4);
            float scale = abs_tol + rel_tol * abs_y;
            float err_i = fabsf(y5 - y4) / scale;
            
            if (err_i > max_error) max_error = err_i;
            if (scale > max_scale) max_scale = scale;
        }
        
        /* 判断是否接受当前步长 */
        if (max_error <= 1.0f || current_step <= min_step) {
            /* 接受当前步：使用5阶结果推进状态 */
            memcpy(temp, state_4th, hidden_size * sizeof(float));
            remaining -= current_step;
        }
        
        /* 计算新的步长，使用PI步长控制 */
        if (max_error > 1e-14f) {
            /* 标准步长调整：h_new = h * safety * (1/error)^(1/5) */
            float error_power = powf(1.0f / max_error, 0.2f);
            float new_step = current_step * safety * error_power;
            
            /* 限制步长范围 */
            if (new_step < min_step) new_step = min_step;
            if (new_step > max_step) new_step = max_step;
            if (new_step > remaining) new_step = remaining;
            
            current_step = new_step;
        } else {
            /* 误差极小，可以大幅增加步长 */
            current_step *= 2.0f;
            if (current_step > max_step) current_step = max_step;
            if (current_step > remaining) current_step = remaining;
        }
        
        /* 安全保护：如果步长不再变化但未完成，强制推进 */
        if (max_error > 100.0f && current_step <= min_step) {
            /* 即使误差很大，也用最小步长强制推进（防止无限循环） */
            for (size_t i = 0; i < hidden_size; i++) {
                temp[i] = temp[i] + current_step * k1[i];
            }
            remaining -= current_step;
        }
    }
    
    /* 最终结果在temp中，复制到output */
    if (output != temp) {
        memcpy(output, temp, hidden_size * sizeof(float));
    }
}

/**
 * @brief 使用CTBP连续时间反向传播进行前向传播（内部函数）
 * 
 * CTBP前向传播与普通CfC前向相同，但额外记录状态轨迹以便
 * 后续进行精确的连续时间梯度计算。
 * 
 * Trajectory存储格式：[t0_state0...stateN, t1_state0...stateN, ..., tn_state0...stateN]
 * 每隔一个检查点记录一次，用于精确反向传播。
 * 
 * @param cell CfC单元句柄
 * @param input 输入向量
 * @param prev_state 前一时间步的状态
 * @param delta_t 时间步长
 * @param output 输出缓冲区
 */
static void cfc_cell_ctbp_forward(CfCCell* cell, const float* input,
                                  const float* prev_state, float delta_t, float* output) {
    size_t hidden_size = cell->config.hidden_size;
    
    /* 先使用封闭形式解计算前向状态 */
    cfc_closed_form_solution(cell, input, prev_state, delta_t, output);
    
    /* 记录状态轨迹到CTBP轨迹缓存（用于精确反向传播梯度计算） */
    int cap = cell->state->ctbp_trajectory_capacity;
    int size = cell->state->ctbp_trajectory_size;
    int new_idx = size;
    
    if (new_idx + (int)hidden_size > cap) {
        /* 扩展轨迹缓存 */
        int new_cap = cap * 2 + (int)hidden_size;
        float* new_traj = (float*)safe_realloc(cell->state->ctbp_state_trajectory,
                                                (size_t)new_cap * sizeof(float));
        if (new_traj) {
            cell->state->ctbp_state_trajectory = new_traj;
            cell->state->ctbp_trajectory_capacity = new_cap;
        }
    }
    
    /* 记录新状态到轨迹 */
    if (cell->state->ctbp_state_trajectory) {
        cap = cell->state->ctbp_trajectory_capacity;
        size = cell->state->ctbp_trajectory_size;
        new_idx = size;
        if (new_idx + (int)hidden_size <= cap) {
            memcpy(&cell->state->ctbp_state_trajectory[new_idx], output, hidden_size * sizeof(float));
            cell->state->ctbp_trajectory_size = new_idx + (int)hidden_size;
        }
    }
}

/**
 * @brief 使用CTBP连续时间反向传播进行精确梯度计算
 * 
 * CTBP精确梯度计算的核心算法：
 * 1. 逆时间方向求解伴随方程（Adjoint Equation）
 *    dλ/dt = -λ * ∂f/∂h, 其中λ(T) = ∂L/∂h(T)
 * 2. 伴随状态λ(t)编码了损失对隐藏状态的梯度
 * 3. 通过伴随状态计算所有参数的梯度
 * 
 * 伴随方程推导：
 *   损失L = ∫ L(h(t), t) dt
 *   dL/dθ = ∫ λ(t) * ∂f/∂θ dt
 *   其中dλ/dt = -λ^T * ∂f/∂h, λ(T) = ∂L/∂h(T)
 * 
 * 一阶离散化实现（修正欧拉法逆时间积分）：
 *   λ(t) = λ(t+Δt) + Δt * λ(t+Δt) * ∂f/∂h|_{h(t+Δt)}
 * 
 * @param cell CfC单元句柄
 * @param input 前向传播的输入
 * @param output_gradient 从损失函数传来的输出梯度
 * @param input_gradient 计算出的输入梯度
 * @param time_span 前向传播的总时间
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_backward_ctbp(CfCCell* cell, const float* input,
                           const float* output_gradient,
                           float* input_gradient, float time_span) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(output_gradient, "输出梯度为空");
    SELFLNN_CHECK_INITIALIZED(cell, "CfC单元未初始化");
    
    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    
    /* 获取前向传播的输入 */
    const float* fwd_input = input ? input : cell->state->input_buffer;
    
    /* 初始化伴随状态：λ(T) = ∂L/∂h(T) */
    float* adjoint = cell->state->ctbp_adjoint;
    memcpy(adjoint, output_gradient, hidden_size * sizeof(float));
    
    /* 使用轨迹缓存进行逆时间积分 */
    int traj_size = cell->state->ctbp_trajectory_size;
    int has_trajectory = (traj_size > 0 && cell->state->ctbp_state_trajectory != NULL);
    
    /* 如果没有轨迹缓存，使用当前状态作为h_{t+Δt} */
    const float* h_next = cell->state->activation;
    
    float delta_t = cell->config.delta_t;
    if (time_span > 0.0f) delta_t = time_span;
    
    /* CTBP参数 */
    CTBPConfig* ctbp_cfg = &cell->config.ctbp_config;
    int use_adjoint = (ctbp_cfg->use_adjoint_method != 0) ? 1 : 0;
    float clip_norm = ctbp_cfg->gradient_clip_norm;
    
    if (use_adjoint) {
        /* === 伴随法（Adjoint Method）=== 
         * 通过逆时间求解伴随方程计算梯度。更节省内存，
         * 但需要在前向传播时存储或重新计算状态。
         * P0-002深度修复: 移除内部梯度清零，由调用方统一清零。 */
        
        /* 如果有轨迹缓存，从轨迹中获取h_next */
        if (has_trajectory && traj_size >= (int)hidden_size) {
            h_next = &cell->state->ctbp_state_trajectory[traj_size - hidden_size];
        }
        
        /* 使用修正欧拉法逆时间积分一步 */
        for (size_t i = 0; i < hidden_size; i++) {
            /* 重新计算前向传播时的中间变量 */
            float gate_input_sum = cell->gate_biases[i * 3];
            float activation_input_sum = cell->bias_vector[i];
            
            for (size_t j = 0; j < input_size; j++) {
                size_t idx = i * input_size + j;
                gate_input_sum += cell->input_gate_weights[idx] * fwd_input[j];
                activation_input_sum += cell->weight_matrix[idx] * fwd_input[j];
            }
            
            for (size_t j = 0; j < hidden_size; j++) {
                size_t h_idx = i * hidden_size + j;
                gate_input_sum += cell->hidden_to_gate_weights[h_idx] * h_next[j];
                activation_input_sum += cell->hidden_to_activation_weights[h_idx] * h_next[j];
            }
            
            /* 激活值和导数 */
            /* ZSFZS-F034修复: 伴随法反向传播添加sigmoid/tanh截断保护 */
            float gate_input_clamped = gate_input_sum;
            if (gate_input_clamped > 10.0f) gate_input_clamped = 10.0f;
            else if (gate_input_clamped < -10.0f) gate_input_clamped = -10.0f;
            float gate = 1.0f / (1.0f + expf(-gate_input_clamped));
            float gate_deriv = gate * (1.0f - gate);
            float act_input_clamped = activation_input_sum;
            if (act_input_clamped > 10.0f) act_input_clamped = 10.0f;
            else if (act_input_clamped < -10.0f) act_input_clamped = -10.0f;
            float act = tanhf(act_input_clamped);
            float act_deriv = 1.0f - act * act;
            
            float driver = gate * act;
            
            float tau = cell->use_adaptive_tau ? cell->time_constants[i] : cell->config.time_constant;
            if (tau < cell->min_time_constant) tau = cell->min_time_constant;
            if (tau > cell->max_time_constant) tau = cell->max_time_constant;
            
            float exp_term = expf(-delta_t / tau);
            float h_new_coeff = 1.0f - exp_term;
            
            /* ∂f/∂h = (-1 + W_gh^T * diag(σ') ⊙ tanh + W_ah^T * diag(1-tanh²) ⊙ σ) / τ
             * 即状态导数对状态的雅可比
             * 完整伴随更新（RK2中点法，二阶精度）：
             *   k1 = Δt * λ(t+Δt) * (-1/τ + ∂driver/∂h)
             *   λ_mid = λ(t+Δt) + 0.5 * k1
             *   k2 = Δt * λ_mid * (-1/τ + ∂driver/∂h)   (固定雅可比)
             *   λ(t) = λ(t+Δt) + k2 */
            
            /* ∂driver/∂h的贡献 */
            float dgate_dh = 0.0f;
            float dact_dh = 0.0f;
            for (size_t j = 0; j < hidden_size; j++) {
                size_t h_idx = j * hidden_size + i; /* W_gh[j][i] */
                dgate_dh += cell->hidden_to_gate_weights[h_idx];
                size_t ha_idx = j * hidden_size + i; /* W_ah[j][i] */
                dact_dh += cell->hidden_to_activation_weights[ha_idx];
            }
            dgate_dh *= gate_deriv;
            dact_dh *= act_deriv;
            
            float ddriver_dh = dgate_dh * act + dact_dh * gate;
            float df_dh = (-1.0f + ddriver_dh) / tau;
            
            /* RK2中点法伴随更新（二阶精度，替代一阶Euler） */
            float lambda_next = adjoint[i];
            float k1 = delta_t * lambda_next * df_dh;
            float lambda_mid = lambda_next + 0.5f * k1;
            float k2 = delta_t * lambda_mid * df_dh;
            adjoint[i] = lambda_next + k2;
            
            /* 梯度裁剪 */
            if (clip_norm > 0.0f && fabsf(adjoint[i]) > clip_norm) {
                adjoint[i] = (adjoint[i] > 0.0f) ? clip_norm : -clip_norm;
            }
            
            /* 使用伴随状态计算梯度 */
            float dL_dlambda = adjoint[i];
            
            float dL_dgate = dL_dlambda * h_new_coeff * act;
            float dL_dact = dL_dlambda * h_new_coeff * gate;
            
            float dL_dgate_input = dL_dgate * gate_deriv;
            float dL_dactivation_input = dL_dact * act_deriv;
            
            /* 权重梯度（P0-002: +=累积模式） */
            for (size_t j = 0; j < input_size; j++) {
                size_t idx = i * input_size + j;
                cell->input_gate_weight_grad[idx] += dL_dgate_input * fwd_input[j];
                cell->weight_grad[idx] += dL_dactivation_input * fwd_input[j];
                if (input_gradient) {
                    input_gradient[j] += dL_dgate_input * cell->input_gate_weights[idx] +
                                        dL_dactivation_input * cell->weight_matrix[idx];
                }
            }
            
            for (size_t j = 0; j < hidden_size; j++) {
                size_t h_idx = i * hidden_size + j;
                cell->hidden_to_gate_weight_grad[h_idx] += dL_dgate_input * h_next[j];
                cell->hidden_to_activation_weight_grad[h_idx] += dL_dactivation_input * h_next[j];
            }
            
            cell->gate_bias_grad[i * 3] += dL_dgate_input;
            cell->bias_grad[i] += dL_dactivation_input;
            
            if (cell->use_adaptive_tau) {
                float dtau_coeff = (delta_t / (tau * tau)) * exp_term;
                cell->time_constant_grad[i] += dL_dlambda * dtau_coeff * (h_next[i] - driver);
            }
        }
    } else {
        /* === 直接法（Direct Method）=== 
         * 直接对封闭形式解求导来计算梯度。内存需求更大，
         * 但数值精度更高。使用轨迹缓存进行多步梯度累积。
         * P0-002深度修复: 移除内部梯度清零，cfc_cell_backward使用+=累积。 */
        
        if (has_trajectory && traj_size >= (int)(hidden_size * 2)) {
            
            /* 逆序遍历轨迹中的每一步 */
            int num_steps = traj_size / (int)hidden_size;
            float* traj = cell->state->ctbp_state_trajectory;
            
            for (int step = num_steps - 1; step >= 1; step--) {
                const float* h_current = &traj[step * hidden_size];
                const float* h_prev = &traj[(step - 1) * hidden_size];
                
                float* step_grad = (float*)safe_calloc(hidden_size, sizeof(float));
                if (!step_grad) break;
                
                /* 用当前步的梯度作为输出梯度源 */
                if (step == num_steps - 1) {
                    memcpy(step_grad, output_gradient, hidden_size * sizeof(float));
                } else {
                    /* 中间步的梯度由前一步累积得出 */
                    for (size_t j = 0; j < hidden_size; j++) {
                        step_grad[j] = cell->state->gradient[j];
                    }
                }
                
                /* 保存当前状态为saved_state供cfc_cell_backward使用 */
                float* saved_prev = cell->state->saved_state;
                float* saved_curr = cell->state->activation;
                memcpy(saved_prev, h_prev, hidden_size * sizeof(float));
                memcpy(saved_curr, h_current, hidden_size * sizeof(float));
                
                /* 对当前步执行标准反向传播 */
                float* step_input_grad = (input_gradient && step == num_steps - 1) ? input_gradient : NULL;
                cfc_cell_backward(cell, step_grad, step_input_grad);
                
                safe_free((void**)&step_grad);
            }
            
            return 0;
        }
        
        /* 无轨迹时使用标准反向传播 */
        return cfc_cell_backward(cell, output_gradient, input_gradient);
    }
    
    return 0;
}

/**
 * @brief 使用RK45自适应步长求解器执行前向传播（公开API）
 */
int cfc_cell_forward_rk45(CfCCell* cell, const float* input, float delta_t,
                          float* hidden_state, int* steps_used) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(input, "输入向量为空");
    SELFLNN_CHECK_NULL(hidden_state, "隐藏状态缓冲区为空");
    SELFLNN_CHECK_INITIALIZED(cell, "CfC单元未初始化");
    
    size_t hidden_size = cell->config.hidden_size;
    
    cfc_cell_rk45_step(cell, input, cell->state->state, delta_t, cell->state->activation);

    // 保存前向传播前的状态 h(t)，供反向传播使用
    memcpy(cell->state->saved_state, cell->state->state, hidden_size * sizeof(float));

    memcpy(hidden_state, cell->state->activation, hidden_size * sizeof(float));
    memcpy(cell->state->state, cell->state->activation, hidden_size * sizeof(float));
    cell->state->time += delta_t;
    
    if (steps_used) {
        *steps_used = 1;
    }
    
    return 0;
}

/**
 * @brief 使用DP54(5阶Dormand-Prince)自适应求解器执行前向传播
 *
 * DP54是嵌入Runge-Kutta对(5阶/4阶)，提供自适应步长控制。
 * 调用底层ode_dp54_solve实现完整DP54积分，含误差估计和步长自动调整。
 */
int cfc_cell_forward_dp54(CfCCell* cell, const float* input, float delta_t,
                          float* hidden_state, int* steps_used) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(input, "输入向量为空");
    SELFLNN_CHECK_NULL(hidden_state, "隐藏状态缓冲区为空");
    SELFLNN_CHECK_INITIALIZED(cell, "CfC单元未初始化");

    int ret = cfc_cell_dp54_step(cell, input, cell->state->state, delta_t, cell->state->activation);
    if (ret != 0) {
        return -1;
    }

    memcpy(cell->state->saved_state, cell->state->state,
           cell->config.hidden_size * sizeof(float));
    memcpy(hidden_state, cell->state->activation,
           cell->config.hidden_size * sizeof(float));
    memcpy(cell->state->state, cell->state->activation,
           cell->config.hidden_size * sizeof(float));
    cell->state->time += delta_t;

    if (steps_used) {
        *steps_used = cell->state->dp54_current_steps;
    }

    return 0;
}

/**
 * @brief 使用Rosenbrock L-稳定刚性求解器执行前向传播
 *
 * Rosenbrock方法对刚性ODE系统具有L-稳定性，适合处理液态网络中
 * 不同时间尺度之间的剧烈变化。调用底层ode_rosenbrock_solve实现。
 */
int cfc_cell_forward_rosenbrock(CfCCell* cell, const float* input, float delta_t,
                                float* hidden_state, int* steps_used) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(input, "输入向量为空");
    SELFLNN_CHECK_NULL(hidden_state, "隐藏状态缓冲区为空");
    SELFLNN_CHECK_INITIALIZED(cell, "CfC单元未初始化");

    int ret = cfc_cell_rosenbrock_step(cell, input, cell->state->state, delta_t,
                                       cell->state->activation);
    if (ret != 0) {
        return -1;
    }

    memcpy(cell->state->saved_state, cell->state->state,
           cell->config.hidden_size * sizeof(float));
    memcpy(hidden_state, cell->state->activation,
           cell->config.hidden_size * sizeof(float));
    memcpy(cell->state->state, cell->state->activation,
           cell->config.hidden_size * sizeof(float));
    cell->state->time += delta_t;

    if (steps_used) {
        *steps_used = cell->state->rosenbrock_current_steps;
    }

    return 0;
}

/**
 * @brief 使用Forest-Ruth 4阶辛积分器执行前向传播
 *
 * Forest-Ruth辛积分器保持哈密顿系统的辛结构，适合长时间能量守恒的
 * 液态网络演化。调用底层ode_forest_ruth_solve实现。
 */
int cfc_cell_forward_symplectic(CfCCell* cell, const float* input, float delta_t,
                                float* hidden_state, int* steps_used) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(input, "输入向量为空");
    SELFLNN_CHECK_NULL(hidden_state, "隐藏状态缓冲区为空");
    SELFLNN_CHECK_INITIALIZED(cell, "CfC单元未初始化");

    int ret = cfc_cell_symplectic_step(cell, input, cell->state->state, delta_t,
                                       cell->state->activation);
    if (ret != 0) {
        return -1;
    }

    memcpy(cell->state->saved_state, cell->state->state,
           cell->config.hidden_size * sizeof(float));
    memcpy(hidden_state, cell->state->activation,
           cell->config.hidden_size * sizeof(float));
    memcpy(cell->state->state, cell->state->activation,
           cell->config.hidden_size * sizeof(float));
    cell->state->time += delta_t;

    if (steps_used) {
        *steps_used = cell->state->symplectic_current_steps;
    }

    return 0;
}

/**
 * @brief 配置RK45求解器参数
 */
int cfc_cell_configure_rk45(CfCCell* cell, const RK45Config* config) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(config, "RK45配置为空");
    
    if (config->rel_tolerance > 0.0f) cell->config.rk45_config.rel_tolerance = config->rel_tolerance;
    if (config->abs_tolerance > 0.0f) cell->config.rk45_config.abs_tolerance = config->abs_tolerance;
    if (config->min_step_size > 0.0f) cell->config.rk45_config.min_step_size = config->min_step_size;
    if (config->max_step_size > 0.0f) cell->config.rk45_config.max_step_size = config->max_step_size;
    if (config->safety_factor > 0.0f) cell->config.rk45_config.safety_factor = config->safety_factor;
    if (config->max_iterations > 0) cell->config.rk45_config.max_iterations = config->max_iterations;
    
    return 0;
}

/**
 * @brief 配置CTBP求解器参数
 */
int cfc_cell_configure_ctbp(CfCCell* cell, const CTBPConfig* config) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(config, "CTBP配置为空");
    
    cell->config.ctbp_config.use_adjoint_method = config->use_adjoint_method;
    cell->config.ctbp_config.gradient_clip_norm = config->gradient_clip_norm;
    cell->config.ctbp_config.store_intermediate = config->store_intermediate;
    if (config->max_checkpoints > 0) cell->config.ctbp_config.max_checkpoints = config->max_checkpoints;
    
    return 0;
}

/**
 * @brief 配置DP54求解器参数
 */
int cfc_cell_configure_dp54(CfCCell* cell, const DP54Config* config) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(config, "DP54配置为空");
    
    if (config->rel_tolerance > 0.0f) cell->config.dp54_config.rel_tolerance = config->rel_tolerance;
    if (config->abs_tolerance > 0.0f) cell->config.dp54_config.abs_tolerance = config->abs_tolerance;
    if (config->min_step_size > 0.0f) cell->config.dp54_config.min_step_size = config->min_step_size;
    if (config->max_step_size > 0.0f) cell->config.dp54_config.max_step_size = config->max_step_size;
    if (config->safety_factor > 0.0f) cell->config.dp54_config.safety_factor = config->safety_factor;
    if (config->max_iterations > 0) cell->config.dp54_config.max_iterations = config->max_iterations;
    
    return 0;
}

/**
 * @brief 配置Rosenbrock刚性求解器参数
 */
int cfc_cell_configure_rosenbrock(CfCCell* cell, const RosenbrockConfig* config) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(config, "Rosenbrock配置为空");
    
    if (config->rel_tolerance > 0.0f) cell->config.rosenbrock_config.rel_tolerance = config->rel_tolerance;
    if (config->abs_tolerance > 0.0f) cell->config.rosenbrock_config.abs_tolerance = config->abs_tolerance;
    if (config->min_step_size > 0.0f) cell->config.rosenbrock_config.min_step_size = config->min_step_size;
    if (config->max_step_size > 0.0f) cell->config.rosenbrock_config.max_step_size = config->max_step_size;
    if (config->gamma_coeff > 0.0f) cell->config.rosenbrock_config.gamma_coeff = config->gamma_coeff;
    if (config->max_iterations > 0) cell->config.rosenbrock_config.max_iterations = config->max_iterations;
    if (config->finite_diff_eps > 0.0f) cell->config.rosenbrock_config.finite_diff_eps = config->finite_diff_eps;
    
    return 0;
}

/**
 * @brief 配置Forest-Ruth辛积分器参数
 */
int cfc_cell_configure_symplectic(CfCCell* cell, const SymplecticConfig* config) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(config, "辛积分器配置为空");
    
    if (config->substep_ratio > 0.0f) cell->config.symplectic_config.substep_ratio = config->substep_ratio;
    if (config->num_substeps > 0) cell->config.symplectic_config.num_substeps = config->num_substeps;
    
    return 0;
}

/**
 * @brief 保存CfC单元参数到文件
 *
 * 保存所有可训练参数和运行时状态，包括权重矩阵、偏置、时间常数、
 * 门控权重、隐藏到隐藏连接权重和内部状态向量。
 */
int cfc_cell_save(const CfCCell* cell, FILE* file) {
    if (!cell || !file) {
        return -1;
    }

    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    size_t weight_size = input_size * hidden_size;
    size_t hidden_weight_size = hidden_size * hidden_size;

    // 保存配置
    if (fwrite(&cell->config, sizeof(CfCCellConfig), 1, file) != 1) return -1;

    // 保存内部状态标志
    int flags[2] = { cell->use_gating, cell->use_adaptive_tau };
    if (fwrite(flags, sizeof(int), 2, file) != 2) return -1;

    // 保存训练参数
    float params[4] = {
        cell->avg_activation,
        cell->max_activation,
        cell->min_time_constant,
        cell->max_time_constant
    };
    if (fwrite(params, sizeof(float), 4, file) != 4) return -1;
    if (fwrite(&cell->tau_learning_rate, sizeof(float), 1, file) != 1) return -1;

    // 保存状态向量
    if (fwrite(cell->state->state, sizeof(float), hidden_size, file) != hidden_size) return -1;
    if (fwrite(cell->state->adapted_params, sizeof(float), hidden_size, file) != hidden_size) return -1;
    if (fwrite(&cell->state->time, sizeof(float), 1, file) != 1) return -1;
    if (fwrite(&cell->state->adaptation_rate, sizeof(float), 1, file) != 1) return -1;

    // 保存权重矩阵和偏置
    if (fwrite(cell->weight_matrix, sizeof(float), weight_size, file) != weight_size) return -1;
    if (fwrite(cell->bias_vector, sizeof(float), hidden_size, file) != hidden_size) return -1;

    // 保存时间常数
    if (fwrite(cell->time_constants, sizeof(float), hidden_size, file) != hidden_size) return -1;

    // 保存门控权重
    if (fwrite(cell->input_gate_weights, sizeof(float), weight_size, file) != weight_size) return -1;
    if (fwrite(cell->forget_gate_weights, sizeof(float), weight_size, file) != weight_size) return -1;
    if (fwrite(cell->output_gate_weights, sizeof(float), weight_size, file) != weight_size) return -1;
    if (fwrite(cell->gate_biases, sizeof(float), hidden_size * 3, file) != hidden_size * 3) return -1;

    // 保存隐藏到隐藏连接权重
    if (fwrite(cell->hidden_to_gate_weights, sizeof(float), hidden_weight_size, file) != hidden_weight_size) return -1;
    if (fwrite(cell->hidden_to_activation_weights, sizeof(float), hidden_weight_size, file) != hidden_weight_size) return -1;

    // 保存多时间尺度数据
    int multi_timescale_flags[3] = {
        cell->use_multi_timescale,
        (int)(cell->fast_tau_ratio * 1000.0f),
        (int)(cell->slow_tau_ratio * 1000.0f)
    };
    if (fwrite(multi_timescale_flags, sizeof(int), 3, file) != 3) return -1;
    if (cell->use_multi_timescale) {
        if (fwrite(cell->fast_time_constants, sizeof(float), hidden_size, file) != hidden_size) return -1;
        if (fwrite(cell->slow_time_constants, sizeof(float), hidden_size, file) != hidden_size) return -1;
        if (fwrite(cell->fast_state, sizeof(float), hidden_size, file) != hidden_size) return -1;
        if (fwrite(cell->slow_state, sizeof(float), hidden_size, file) != hidden_size) return -1;
    }

    return 0;
}

/**
 * @brief 从文件加载CfC单元参数
 *
 * 从文件加载所有可训练参数到已创建的CfC单元实例中。
 * 调用前cell必须已通过cfc_cell_create创建，且配置必须匹配。
 */
int cfc_cell_load(CfCCell* cell, FILE* file) {
    if (!cell || !file) {
        return -1;
    }

    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    size_t weight_size = input_size * hidden_size;
    size_t hidden_weight_size = hidden_size * hidden_size;

    // 读取配置并验证
    CfCCellConfig saved_config;
    if (fread(&saved_config, sizeof(CfCCellConfig), 1, file) != 1) return -1;
    if (saved_config.input_size != input_size || saved_config.hidden_size != hidden_size) {
        return -1;
    }
    cell->config = saved_config;

    // 读取内部状态标志
    int flags[2];
    if (fread(flags, sizeof(int), 2, file) != 2) return -1;
    cell->use_gating = flags[0];
    cell->use_adaptive_tau = flags[1];

    // 读取训练参数
    float params[4];
    if (fread(params, sizeof(float), 4, file) != 4) return -1;
    cell->avg_activation = params[0];
    cell->max_activation = params[1];
    cell->min_time_constant = params[2];
    cell->max_time_constant = params[3];
    if (fread(&cell->tau_learning_rate, sizeof(float), 1, file) != 1) return -1;

    // 读取状态向量
    if (fread(cell->state->state, sizeof(float), hidden_size, file) != hidden_size) return -1;
    if (fread(cell->state->adapted_params, sizeof(float), hidden_size, file) != hidden_size) return -1;
    if (fread(&cell->state->time, sizeof(float), 1, file) != 1) return -1;
    if (fread(&cell->state->adaptation_rate, sizeof(float), 1, file) != 1) return -1;

    // 读取权重矩阵和偏置
    if (fread(cell->weight_matrix, sizeof(float), weight_size, file) != weight_size) return -1;
    if (fread(cell->bias_vector, sizeof(float), hidden_size, file) != hidden_size) return -1;

    // 读取时间常数
    if (fread(cell->time_constants, sizeof(float), hidden_size, file) != hidden_size) return -1;

    // 读取门控权重
    if (fread(cell->input_gate_weights, sizeof(float), weight_size, file) != weight_size) return -1;
    if (fread(cell->forget_gate_weights, sizeof(float), weight_size, file) != weight_size) return -1;
    if (fread(cell->output_gate_weights, sizeof(float), weight_size, file) != weight_size) return -1;
    if (fread(cell->gate_biases, sizeof(float), hidden_size * 3, file) != hidden_size * 3) return -1;

    // 读取隐藏到隐藏连接权重
    if (fread(cell->hidden_to_gate_weights, sizeof(float), hidden_weight_size, file) != hidden_weight_size) return -1;
    if (fread(cell->hidden_to_activation_weights, sizeof(float), hidden_weight_size, file) != hidden_weight_size) return -1;

    // 读取多时间尺度数据（如果存在）
    int multi_timescale_flags[3];
    if (fread(multi_timescale_flags, sizeof(int), 3, file) != 3) {
        cell->use_multi_timescale = 0;
        return 0;
    }
    cell->use_multi_timescale = multi_timescale_flags[0];
    cell->config.use_multi_timescale = multi_timescale_flags[0];
    if (cell->use_multi_timescale) {
        cell->fast_tau_ratio = (float)multi_timescale_flags[1] / 1000.0f;
        cell->slow_tau_ratio = (float)multi_timescale_flags[2] / 1000.0f;
        if (fread(cell->fast_time_constants, sizeof(float), hidden_size, file) != hidden_size) return -1;
        if (fread(cell->slow_time_constants, sizeof(float), hidden_size, file) != hidden_size) return -1;
        if (fread(cell->fast_state, sizeof(float), hidden_size, file) != hidden_size) return -1;
        if (fread(cell->slow_state, sizeof(float), hidden_size, file) != hidden_size) return -1;
    }

    return 0;
}

/**
 * @brief 启用或禁用多时间尺度并行演化
 */
int cfc_cell_enable_multi_timescale(CfCCell* cell, int enable) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_INITIALIZED(cell, "CfC单元未初始化");
    size_t hidden_size = cell->config.hidden_size;
    int was_enabled = cell->use_multi_timescale;
    cell->use_multi_timescale = (enable != 0) ? 1 : 0;
    cell->config.use_multi_timescale = cell->use_multi_timescale;
    if (cell->use_multi_timescale && !was_enabled) {
        for (size_t i = 0; i < hidden_size; i++) {
            float base_tau = cell->time_constants[i];
            cell->fast_time_constants[i] = base_tau * cell->fast_tau_ratio;
            cell->slow_time_constants[i] = base_tau * cell->slow_tau_ratio;
            if (cell->fast_time_constants[i] < 0.001f) cell->fast_time_constants[i] = 0.001f;
            if (cell->slow_time_constants[i] > 100.0f) cell->slow_time_constants[i] = 100.0f;
            cell->fast_state[i] = 0.0f;
            cell->slow_state[i] = 0.0f;
        }
    }
    return 0;
}

/**
 * @brief 设置多时间尺度的快慢时间常数比例
 */
int cfc_cell_set_tau_ratios(CfCCell* cell, float fast_ratio, float slow_ratio) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_INITIALIZED(cell, "CfC单元未初始化");
    if (fast_ratio <= 0.0f || slow_ratio <= fast_ratio) return -1;
    cell->fast_tau_ratio = fast_ratio;
    cell->slow_tau_ratio = slow_ratio;
    cell->config.fast_tau_ratio = fast_ratio;
    cell->config.slow_tau_ratio = slow_ratio;
    if (cell->use_multi_timescale) {
        size_t hidden_size = cell->config.hidden_size;
        for (size_t i = 0; i < hidden_size; i++) {
            float base_tau = cell->time_constants[i];
            cell->fast_time_constants[i] = base_tau * cell->fast_tau_ratio;
            cell->slow_time_constants[i] = base_tau * cell->slow_tau_ratio;
            if (cell->fast_time_constants[i] < 0.001f) cell->fast_time_constants[i] = 0.001f;
            if (cell->slow_time_constants[i] > 100.0f) cell->slow_time_constants[i] = 100.0f;
        }
    }
    return 0;
}

/* ============ 液时域缩放（Liquid Time Scaling）实现 ============ */

/**
 * @brief 计算液时域缩放的时间常数
 *
 * 对每个神经元计算输入依赖的时间常数：
 *   net_i = W_tau[i,:] · x + b_tau[i]
 *   sig_i = sigmoid(net_i)
 *   τ_i = sig_i · (τ_max - τ_min) + τ_min
 *
 * 如果启用层归一化，先对 net 做层归一化再经过 sigmoid。
 *
 * @param cell CfC单元句柄
 * @param input 输入向量
 * @param tau_out 输出时间常数缓冲区
 * @param raw_net 输出未归一化前的网络值缓冲区（用于反向传播，可为NULL）
 * @param normalized 输出归一化后的值缓冲区（用于反向传播，可为NULL）
 * @return int 成功返回0，失败返回-1
 */
static int compute_liquid_time_constants(CfCCell* cell, const float* input,
                                          float* tau_out, float* raw_net,
                                          float* normalized) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(input, "输入向量为空");
    SELFLNN_CHECK_NULL(tau_out, "时间常数输出为空");

    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    float tau_min = cell->liquid_tau_min;
    float tau_max = cell->liquid_tau_max;
    float tau_range = tau_max - tau_min;

    float* net_buffer = cell->liquid_tau_workspace;
    float mean = 0.0f;
    float variance = 0.0f;

    // 步骤0: 输入归一化（L2范数归一化，去除输入强度对时间常数的影响）
    float input_norm = 0.0f;
    for (size_t j = 0; j < input_size; j++) {
        input_norm += input[j] * input[j];
    }
    input_norm = sqrtf(input_norm) + 1e-8f;
    float inv_norm = 1.0f / input_norm;

    // 步骤1: 计算 net = W_tau · (x/||x||) + b_tau（使用归一化输入）
    for (size_t i = 0; i < hidden_size; i++) {
        float sum = cell->liquid_tau_bias[i];
        for (size_t j = 0; j < input_size; j++) {
            sum += cell->liquid_tau_weights[i * input_size + j] * input[j] * inv_norm;
        }
        net_buffer[i] = sum;
        mean += sum;
    }

    // 步骤2: 可选层归一化
    if (cell->liquid_use_layer_norm) {
        mean /= (float)hidden_size;
        for (size_t i = 0; i < hidden_size; i++) {
            float diff = net_buffer[i] - mean;
            variance += diff * diff;
        }
        variance /= (float)hidden_size;
        float std = sqrtf(variance + 1e-8f);
        for (size_t i = 0; i < hidden_size; i++) {
            net_buffer[i] = (net_buffer[i] - mean) / std;
        }
    }

    // 保存原始/归一化后的值供反向传播使用
    if (raw_net) {
        memcpy(raw_net, cell->liquid_tau_workspace, hidden_size * sizeof(float));
    }
    if (normalized) {
        memcpy(normalized, net_buffer, hidden_size * sizeof(float));
    }

    // 步骤3: sigmoid 激活 + 缩放到 [tau_min, tau_max]
    for (size_t i = 0; i < hidden_size; i++) {
        float sig = 1.0f / (1.0f + expf(-net_buffer[i]));
        tau_out[i] = sig * tau_range + tau_min;
    }

    return 0;
}

/**
 * @brief 启用液时域缩放并设置配置
 */
int cfc_cell_enable_liquid_scaling(CfCCell* cell, const LiquidTimeScalingConfig* config) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_INITIALIZED(cell, "CfC单元未初始化");

    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    size_t tau_weight_size = hidden_size * input_size;

    if (config) {
        cell->liquid_tau_min = config->tau_min;
        cell->liquid_tau_max = config->tau_max;
        cell->liquid_init_scale = config->init_scale;
        cell->liquid_use_layer_norm = config->use_layer_norm;
    } else {
        cell->liquid_tau_min = 0.01f;
        cell->liquid_tau_max = 2.0f;
        cell->liquid_init_scale = 0.01f;
        cell->liquid_use_layer_norm = 0;
    }

    if (cell->liquid_tau_min <= 0.0f) cell->liquid_tau_min = 0.001f;
    if (cell->liquid_tau_max <= cell->liquid_tau_min) cell->liquid_tau_max = cell->liquid_tau_min * 10.0f;

    // 初始化液时域权重：P0-001使用Xavier自适应缩放替代固定范围
    float init_range = cell->liquid_init_scale;
    if (cell->config.use_xavier_init || cell->config.use_kaiming_init) {
        float adaptive_range = compute_init_limit(
            cell->config.use_xavier_init, cell->config.use_kaiming_init,
            cell->config.weight_init_scale, input_size, hidden_size, cell->liquid_init_scale);
        init_range = adaptive_range;
    }
    for (size_t i = 0; i < tau_weight_size; i++) {
        cell->liquid_tau_weights[i] = random_uniform_seeded(-init_range, init_range,
            (unsigned int)(uintptr_t)cell ^ (unsigned int)i ^ 0xABCD);
    }
    for (size_t i = 0; i < hidden_size; i++) {
        cell->liquid_tau_bias[i] = 0.0f;
    }

    // 重置梯度缓冲区
    memset(cell->liquid_tau_weight_grad, 0, tau_weight_size * sizeof(float));
    memset(cell->liquid_tau_bias_grad, 0, hidden_size * sizeof(float));

    // 设置时间常数的初始值为 tau_min 和 tau_max 的几何平均
    float init_tau = sqrtf(cell->liquid_tau_min * cell->liquid_tau_max);
    for (size_t i = 0; i < hidden_size; i++) {
        cell->computed_liquid_tau[i] = init_tau;
        cell->time_constants[i] = init_tau;
    }

    cell->use_adaptive_tau = 1;
    cell->use_liquid_scaling = 1;

    return 0;
}

/**
 * @brief 使用液时域缩放执行前向传播
 *
 * 在每个时间步，首先根据当前输入计算每个神经元的液时域缩放时间常数，
 * 然后使用更新后的时间常数执行CfC闭式解前向传播。
 */
int cfc_cell_forward_liquid(CfCCell* cell, const float* input, float* hidden_state, float* computed_tau) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(input, "输入向量为空");
    SELFLNN_CHECK_NULL(hidden_state, "隐藏状态缓冲区为空");

    if (!cell->use_liquid_scaling) {
        return cfc_cell_forward(cell, input, hidden_state);
    }

    size_t hidden_size = cell->config.hidden_size;

    // 步骤1: 计算输入依赖的液时域时间常数
    int ret = compute_liquid_time_constants(cell, input,
                                            cell->computed_liquid_tau,
                                            NULL, NULL);
    if (ret != 0) return ret;

    // 步骤2: 将计算出的时间常数更新到 time_constants 数组中
    // 闭式解函数 cfc_closed_form_solution 使用 cell->time_constants
    for (size_t i = 0; i < hidden_size; i++) {
        cell->time_constants[i] = cell->computed_liquid_tau[i];
    }

    // 步骤3: 使用更新后的时间常数执行标准前向传播
    ret = cfc_cell_forward(cell, input, hidden_state);

    // 步骤4: 如果需要，输出计算出的时间常数
    if (computed_tau) {
        memcpy(computed_tau, cell->computed_liquid_tau, hidden_size * sizeof(float));
    }

    return ret;
}

/**
 * @brief 获取当前液时域缩放的时间常数
 */
int cfc_cell_get_liquid_tau(const CfCCell* cell, float* tau_out, int* size) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(tau_out, "时间常数输出为空");

    size_t hidden_size = cell->config.hidden_size;
    memcpy(tau_out, cell->computed_liquid_tau, hidden_size * sizeof(float));

    if (size) {
        *size = (int)hidden_size;
    }

    return 0;
}
/**
 * @brief 直接设置液时域缩放的时间常数（覆盖计算值）
 */
int cfc_cell_set_liquid_tau(CfCCell* cell, const float* tau_values) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(tau_values, "时间常数数组为空");

    size_t hidden_size = cell->config.hidden_size;

    // 验证并复制时间常数值
    for (size_t i = 0; i < hidden_size; i++) {
        if (tau_values[i] <= 0.0f) {
            cell->computed_liquid_tau[i] = cell->liquid_tau_min;
        } else if (tau_values[i] > cell->liquid_tau_max * 10.0f) {
            cell->computed_liquid_tau[i] = cell->liquid_tau_max;
        } else {
            cell->computed_liquid_tau[i] = tau_values[i];
        }
        cell->time_constants[i] = cell->computed_liquid_tau[i];
    }

    return 0;
}

/**
 * @brief 液时域缩放的梯度反向传播
 *
 * 计算损失对液时域缩放参数（W_tau, b_tau）的梯度，用于端到端训练。
 * 需要从上层获取 dL/dτ 的梯度（即 tau_gradient 参数）。
 *
 * 反向传播公式：
 *   dL/dW_tau[i,j] = dL/dτ_i · dτ_i/dsig_i · dsig_i/dnet_i · dnet_i/dW_tau[i,j]
 *                  = dL/dτ_i · (τ_max - τ_min) · sig_i · (1 - sig_i) · x_j
 *   dL/db_tau[i]  = dL/dτ_i · (τ_max - τ_min) · sig_i · (1 - sig_i)
 *   dL/dx_j        = Σ_i dL/dτ_i · (τ_max - τ_min) · sig_i · (1 - sig_i) · W_tau[i,j]
 */
int cfc_cell_backward_liquid(CfCCell* cell, const float* gradient, float* input_gradient, float* tau_gradient) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(gradient, "梯度向量为空");

    if (!cell->use_liquid_scaling) {
        return 0;
    }

    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    float tau_range = cell->liquid_tau_max - cell->liquid_tau_min;

    // 使用 gradient 作为 dL/dτ（损失对时间常数的梯度）
    const float* dL_dtau = gradient;

    // 如果提供了 tau_gradient 输出缓冲区，将梯度复制到其中
    if (tau_gradient) {
        memcpy(tau_gradient, gradient, hidden_size * sizeof(float));
    }

    // 步骤1: 计算 sigmoid 输出和导数
    // 先重新计算 net 值（使用当前输入缓冲区）
    float* net = cell->liquid_tau_workspace;
    for (size_t i = 0; i < hidden_size; i++) {
        net[i] = cell->liquid_tau_bias[i];
        for (size_t j = 0; j < input_size; j++) {
            net[i] += cell->liquid_tau_weights[i * input_size + j] *
                      cell->state->input_buffer[j];
        }
    }

    // 如果启用了层归一化，需要重新计算归一化后的值
    if (cell->liquid_use_layer_norm) {
        float mean = 0.0f;
        for (size_t i = 0; i < hidden_size; i++) mean += net[i];
        mean /= (float)hidden_size;
        float variance = 0.0f;
        for (size_t i = 0; i < hidden_size; i++) {
            float diff = net[i] - mean;
            variance += diff * diff;
        }
        variance /= (float)hidden_size;
        float std = sqrtf(variance + 1e-8f);
        for (size_t i = 0; i < hidden_size; i++) {
            net[i] = (net[i] - mean) / std;
        }
    }

    // 步骤2: 计算 sigmoid 导数 dsig/dnet = sig * (1 - sig)
    // 同时计算 dτ/dnet = tau_range * dsig/dnet
    for (size_t i = 0; i < hidden_size; i++) {
        float sig = 1.0f / (1.0f + expf(-net[i]));
        float dsig_dnet = sig * (1.0f - sig);
        // clip 防止极端值
        if (dsig_dnet < 1e-8f) dsig_dnet = 1e-8f;
        net[i] = dL_dtau[i] * tau_range * dsig_dnet;
    }

    // 步骤3: 计算梯度
    // dL/dW_tau[i,j] = dL/dnet_i * x_j
    // dL/db_tau[i]   = dL/dnet_i
    for (size_t i = 0; i < hidden_size; i++) {
        float dL_dnet = net[i];
        cell->liquid_tau_bias_grad[i] += dL_dnet;

        for (size_t j = 0; j < input_size; j++) {
            size_t idx = i * input_size + j;
            cell->liquid_tau_weight_grad[idx] += dL_dnet * cell->state->input_buffer[j];
        }
    }

    // 步骤4: 计算输入梯度 dL/dx_j = Σ_i dL/dnet_i * W_tau[i,j]
    if (input_gradient) {
        memset(input_gradient, 0, input_size * sizeof(float));
        for (size_t j = 0; j < input_size; j++) {
            for (size_t i = 0; i < hidden_size; i++) {
                input_gradient[j] += net[i] * cell->liquid_tau_weights[i * input_size + j];
            }
        }
    }

    return 0;
}

/* ============ 神经ODE伴随法（Neural ODE Adjoint）实现 ============ */

/**
 * @brief 启用神经ODE伴随法并设置配置
 */
int cfc_cell_enable_neural_ode_adjoint(CfCCell* cell, const NeuralODEConfig* config) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_INITIALIZED(cell, "CfC单元未初始化");

    size_t hidden_size = cell->config.hidden_size;

    if (config) {
        cell->adjoint_record_trajectory = config->record_trajectory ? 1 : 0;
        cell->adjoint_max_trajectory_points = config->max_trajectory_points > 0 ?
            config->max_trajectory_points : 10000;
        cell->adjoint_gradient_clip_norm = config->gradient_clip_norm > 0.0f ?
            config->gradient_clip_norm : 0.0f;
        cell->adjoint_use_augmented_state = config->use_augmented_state ? 1 : 0;
        cell->adjoint_interpolation_method = config->interpolation_method;
    } else {
        cell->adjoint_record_trajectory = 1;
        cell->adjoint_max_trajectory_points = 10000;
        cell->adjoint_gradient_clip_norm = 0.0f;
        cell->adjoint_use_augmented_state = 0;
        cell->adjoint_interpolation_method = 1;
    }

    if (cell->adjoint_record_trajectory) {
        int max_pts = cell->adjoint_max_trajectory_points;
        safe_free((void**)&cell->adjoint_trajectory);
        safe_free((void**)&cell->adjoint_timestamps);

        cell->adjoint_trajectory = (float*)safe_calloc(
            (size_t)max_pts * hidden_size, sizeof(float));
        cell->adjoint_timestamps = (float*)safe_calloc((size_t)max_pts, sizeof(float));

        if (!cell->adjoint_trajectory || !cell->adjoint_timestamps) {
            safe_free((void**)&cell->adjoint_trajectory);
            safe_free((void**)&cell->adjoint_timestamps);
            cell->adjoint_trajectory_capacity = 0;
            cell->adjoint_trajectory_count = 0;
            return -1;
        }
        cell->adjoint_trajectory_capacity = max_pts;
        cell->adjoint_trajectory_count = 0;
    }

    if (cell->adjoint_use_augmented_state && !cell->adjoint_param_grad_workspace) {
        size_t total_params = hidden_size * (cell->config.input_size + hidden_size) * 2
                            + hidden_size * 2;
        cell->adjoint_param_grad_workspace = (float*)safe_calloc(total_params, sizeof(float));
    }

    cell->adjoint_forward_step = 0;
    cell->use_neural_ode_adjoint = 1;
    return 0;
}

/**
 * @brief 使用轨迹记录执行前向传播
 */
int cfc_cell_forward_with_trajectory(CfCCell* cell, const float* input, float* hidden_state) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(input, "输入向量为空");
    SELFLNN_CHECK_NULL(hidden_state, "隐藏状态缓冲区为空");
    SELFLNN_CHECK_INITIALIZED(cell, "CfC单元未初始化");

    size_t hidden_size = cell->config.hidden_size;

    int ret = cfc_cell_forward(cell, input, hidden_state);
    if (ret != 0) return ret;

    if (cell->adjoint_record_trajectory && cell->adjoint_trajectory) {
        int count = cell->adjoint_trajectory_count;
        int capacity = cell->adjoint_trajectory_capacity;

        if (count < capacity) {
            memcpy(&cell->adjoint_trajectory[(size_t)count * hidden_size],
                   hidden_state, hidden_size * sizeof(float));
            cell->adjoint_timestamps[count] = (float)(cell->adjoint_forward_step) * cell->config.delta_t;
            cell->adjoint_trajectory_count = count + 1;
        } else {
            size_t move_size = (size_t)(capacity - 1) * hidden_size * sizeof(float);
            memmove(cell->adjoint_trajectory,
                    &cell->adjoint_trajectory[hidden_size], move_size);
            memcpy(&cell->adjoint_trajectory[(size_t)(capacity - 1) * hidden_size],
                   hidden_state, hidden_size * sizeof(float));
            for (int i = 0; i < capacity - 1; i++) {
                cell->adjoint_timestamps[i] = cell->adjoint_timestamps[i + 1];
            }
            cell->adjoint_timestamps[capacity - 1] =
                (float)(cell->adjoint_forward_step) * cell->config.delta_t;
        }
    }

    cell->adjoint_forward_step++;
    return 0;
}

/**
 * @brief 计算伴随ODE的右端项 da/dt = -a^T · ∂f/∂h
 */
static int cfc_compute_adjoint_rhs(CfCCell* cell, const float* input,
                                    const float* hidden_state,
                                    const float* adjoint_in,
                                    float* adjoint_out) {
    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;

    for (size_t i = 0; i < hidden_size; i++) {
        float gate_input_sum = cell->gate_biases[i * 3];
        float activation_input_sum = cell->bias_vector[i];

        for (size_t j = 0; j < input_size; j++) {
            size_t idx = i * input_size + j;
            gate_input_sum += cell->input_gate_weights[idx] * input[j];
            activation_input_sum += cell->weight_matrix[idx] * input[j];
        }

        for (size_t j = 0; j < hidden_size; j++) {
            size_t h_idx = i * hidden_size + j;
            gate_input_sum += cell->hidden_to_gate_weights[h_idx] * hidden_state[j];
            activation_input_sum += cell->hidden_to_activation_weights[h_idx] * hidden_state[j];
        }

        float gate = 1.0f / (1.0f + expf(-gate_input_sum));
        float gate_deriv = gate * (1.0f - gate);
        float act = tanhf(activation_input_sum);
        float act_deriv = 1.0f - act * act;

        float tau = cell->use_adaptive_tau ? cell->time_constants[i] : cell->config.time_constant;
        if (tau < cell->min_time_constant) tau = cell->min_time_constant;

        float sum_adjoint_gh = 0.0f;
        float sum_adjoint_ah = 0.0f;
        for (size_t j = 0; j < hidden_size; j++) {
            size_t gh_idx = j * hidden_size + i;
            size_t ah_idx = j * hidden_size + i;
            sum_adjoint_gh += adjoint_in[j] * cell->hidden_to_gate_weights[gh_idx] / tau;
            sum_adjoint_ah += adjoint_in[j] * cell->hidden_to_activation_weights[ah_idx] / tau;
        }

        float ddriver_term = gate_deriv * act * sum_adjoint_gh
                           + gate * act_deriv * sum_adjoint_ah;

        adjoint_out[i] = -((-adjoint_in[i] / tau) + ddriver_term);
    }

    return 0;
}

/**
 * @brief 使用神经ODE伴随法执行反向传播
 */
int cfc_cell_backward_adjoint(CfCCell* cell, const float* output_gradient, float* input_gradient) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(output_gradient, "输出梯度为空");
    SELFLNN_CHECK_INITIALIZED(cell, "CfC单元未初始化");

    if (!cell->use_neural_ode_adjoint) return -1;

    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    float clip_norm = cell->adjoint_gradient_clip_norm;
    float delta_t = cell->config.delta_t;

    int traj_count = cell->adjoint_trajectory_count;
    int has_trajectory = (traj_count > 0 && cell->adjoint_trajectory != NULL);

    float* adjoint = cell->adjoint_state;
    float* workspace = cell->adjoint_workspace;

    memcpy(adjoint, output_gradient, hidden_size * sizeof(float));

    /* P0-002深度修复: 移除内部梯度清零，由调用方统一清零。
     * 函数内所有梯度写入已使用+=累积模式。 */

    if (!has_trajectory || traj_count < 2) {
        for (size_t i = 0; i < hidden_size; i++) {
            float gate_input_sum = cell->gate_biases[i * 3];
            float activation_input_sum = cell->bias_vector[i];
            const float* fwd_input = cell->state->input_buffer;
            const float* h_next = cell->state->activation;

            for (size_t j = 0; j < input_size; j++) {
                size_t idx = i * input_size + j;
                gate_input_sum += cell->input_gate_weights[idx] * fwd_input[j];
                activation_input_sum += cell->weight_matrix[idx] * fwd_input[j];
            }
            for (size_t j = 0; j < hidden_size; j++) {
                size_t h_idx = i * hidden_size + j;
                gate_input_sum += cell->hidden_to_gate_weights[h_idx] * h_next[j];
                activation_input_sum += cell->hidden_to_activation_weights[h_idx] * h_next[j];
            }

            float gate = 1.0f / (1.0f + expf(-gate_input_sum));
            float gate_deriv = gate * (1.0f - gate);
            float act = tanhf(activation_input_sum);
            float act_deriv = 1.0f - act * act;
            float driver = gate * act;
            float tau = cell->use_adaptive_tau ? cell->time_constants[i] : cell->config.time_constant;
            if (tau < cell->min_time_constant) tau = cell->min_time_constant;
            float exp_term = expf(-delta_t / tau);
            float h_new_coeff = 1.0f - exp_term;

            float dL_dlambda = adjoint[i];
            float dL_dgate = dL_dlambda * h_new_coeff * act;
            float dL_dact = dL_dlambda * h_new_coeff * gate;
            float dL_dgate_input = dL_dgate * gate_deriv;
            float dL_dactivation_input = dL_dact * act_deriv;

            for (size_t j = 0; j < input_size; j++) {
                size_t idx = i * input_size + j;
                cell->input_gate_weight_grad[idx] += dL_dgate_input * fwd_input[j];
                cell->weight_grad[idx] += dL_dactivation_input * fwd_input[j];
                if (input_gradient) {
                    input_gradient[j] += dL_dgate_input * cell->input_gate_weights[idx]
                                       + dL_dactivation_input * cell->weight_matrix[idx];
                }
            }
            for (size_t j = 0; j < hidden_size; j++) {
                size_t h_idx = i * hidden_size + j;
                cell->hidden_to_gate_weight_grad[h_idx] += dL_dgate_input * h_next[j];
                cell->hidden_to_activation_weight_grad[h_idx] += dL_dactivation_input * h_next[j];
            }
            cell->gate_bias_grad[i * 3] += dL_dgate_input;
            cell->bias_grad[i] += dL_dactivation_input;

            if (cell->use_adaptive_tau) {
                float dtau_coeff = (delta_t / (tau * tau)) * exp_term;
                cell->time_constant_grad[i] += dL_dlambda * dtau_coeff * (h_next[i] - driver);
            }
        }
        return 0;
    }

    for (int step = traj_count - 1; step >= 1; step--) {
        const float* h_curr = &cell->adjoint_trajectory[(size_t)(step - 1) * hidden_size];
        const float* h_next = &cell->adjoint_trajectory[(size_t)step * hidden_size];
        const float* fwd_input = cell->state->input_buffer;

        cfc_compute_adjoint_rhs(cell, fwd_input, h_curr, adjoint, workspace);

        for (size_t i = 0; i < hidden_size; i++) {
            adjoint[i] += delta_t * workspace[i];
            if (clip_norm > 0.0f && fabsf(adjoint[i]) > clip_norm) {
                adjoint[i] = (adjoint[i] > 0.0f) ? clip_norm : -clip_norm;
            }
        }

        for (size_t i = 0; i < hidden_size; i++) {
            float gate_input_sum = cell->gate_biases[i * 3];
            float activation_input_sum = cell->bias_vector[i];

            for (size_t j = 0; j < input_size; j++) {
                size_t idx = i * input_size + j;
                gate_input_sum += cell->input_gate_weights[idx] * fwd_input[j];
                activation_input_sum += cell->weight_matrix[idx] * fwd_input[j];
            }
            for (size_t j = 0; j < hidden_size; j++) {
                size_t h_idx = i * hidden_size + j;
                gate_input_sum += cell->hidden_to_gate_weights[h_idx] * h_curr[j];
                activation_input_sum += cell->hidden_to_activation_weights[h_idx] * h_curr[j];
            }

            float gate = 1.0f / (1.0f + expf(-gate_input_sum));
            float gate_deriv = gate * (1.0f - gate);
            float act = tanhf(activation_input_sum);
            float act_deriv = 1.0f - act * act;
            float driver = gate * act;
            float tau = cell->use_adaptive_tau ? cell->time_constants[i] : cell->config.time_constant;
            if (tau < cell->min_time_constant) tau = cell->min_time_constant;
            float exp_term = expf(-delta_t / tau);
            float h_new_coeff = 1.0f - exp_term;

            float dL_dlambda = adjoint[i];
            float dL_dgate = dL_dlambda * h_new_coeff * act;
            float dL_dact = dL_dlambda * h_new_coeff * gate;
            float dL_dgate_input = dL_dgate * gate_deriv;
            float dL_dactivation_input = dL_dact * act_deriv;

            for (size_t j = 0; j < input_size; j++) {
                size_t idx = i * input_size + j;
                cell->input_gate_weight_grad[idx] += dL_dgate_input * fwd_input[j];
                cell->weight_grad[idx] += dL_dactivation_input * fwd_input[j];
                if (input_gradient) {
                    input_gradient[j] += dL_dgate_input * cell->input_gate_weights[idx]
                                       + dL_dactivation_input * cell->weight_matrix[idx];
                }
            }
            for (size_t j = 0; j < hidden_size; j++) {
                size_t h_idx = i * hidden_size + j;
                cell->hidden_to_gate_weight_grad[h_idx] += dL_dgate_input * h_curr[j];
                cell->hidden_to_activation_weight_grad[h_idx] += dL_dactivation_input * h_curr[j];
            }
            cell->gate_bias_grad[i * 3] += dL_dgate_input;
            cell->bias_grad[i] += dL_dactivation_input;

            if (cell->use_adaptive_tau) {
                float dtau_coeff = (delta_t / (tau * tau)) * exp_term;
                cell->time_constant_grad[i] += dL_dlambda * dtau_coeff * (h_next[i] - driver);
            }
        }
    }

    return 0;
}

/**
 * @brief 获取当前伴随状态
 */
int cfc_cell_get_adjoint_state(const CfCCell* cell, float* adjoint_out, int* size) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(adjoint_out, "伴随状态输出缓冲区为空");

    size_t hidden_size = cell->config.hidden_size;
    memcpy(adjoint_out, cell->adjoint_state, hidden_size * sizeof(float));
    if (size) *size = (int)hidden_size;
    return 0;
}

/* ==========================================================================
 * 门控CfC变体（Gated CfC）实现
 * ========================================================================== */

/**
 * @brief 交叉门控更新：门控信号在神经元之间递归传播
 *
 * gate_i(t+1) = σ(W_input_gate*x + W_hidden_gate*h + Σ_j W_cross[i,j]*gate_j(t) + b_gate)
 * 交叉门控使门控信号产生神经元间依赖，增强表达能力。
 */
static int gated_cfc_cross_gate_update(GatedCfCData* gd, const float* gate_input,
                                        const float* activation, float* gate_output,
                                        size_t hidden_size) {
    if (!gd || !gate_input || !activation || !gate_output) return -1;
    /* 递归更新交叉门控: new_gate = sigmoid(gate_input + W_cross * old_gate) */
    memcpy(gd->workspace, gd->cross_gate_state, hidden_size * sizeof(float));
    for (size_t i = 0; i < hidden_size; i++) {
        float cross_sum = 0.0f;
        for (size_t j = 0; j < hidden_size; j++) {
            cross_sum += gd->cross_gate_weights[i * hidden_size + j] * gd->workspace[j];
        }
        float total = gate_input[i] + cross_sum + activation[i];
        /* ZSFZS-F033修复: 门控CfC交叉门控sigmoid添加截断保护 */
        float total_clamped = total;
        if (total_clamped > 10.0f) total_clamped = 10.0f;
        else if (total_clamped < -10.0f) total_clamped = -10.0f;
        float new_gate = 1.0f / (1.0f + expf(-total_clamped));
        gd->cross_gate_state[i] = new_gate;
        gate_output[i] = new_gate;
    }
    return 0;
}

/**
 * @brief 金字塔门控：通过多层级联精化门控信号
 *
 * gate^{(l+1)} = σ(W_pyr^{(l)} * gate^{(l)} + b_pyr^{(l)})
 * 逐层精化实现从粗到细的门控控制。
 */
static int gated_cfc_pyramidal_gate(GatedCfCData* gd, float* gate_signal, size_t hidden_size) {
    if (!gd || !gate_signal) return -1;
    float* current = gate_signal;
    for (int l = 0; l < gd->pyramidal_layers; l++) {
        for (size_t i = 0; i < hidden_size; i++) {
            if (l == 0) {
                float val = gd->pyramidal_gate_weights[l * hidden_size + i] * current[i]
                          + gd->pyramidal_gate_biases[l];
                gd->pyramidal_gate_output[i] = 1.0f / (1.0f + expf(-val));
            } else {
                /* 上层依赖下层输出 */
                float val = gd->pyramidal_gate_weights[l * hidden_size + i] * gd->pyramidal_gate_output[i]
                          + gd->pyramidal_gate_biases[l];
                gd->pyramidal_gate_output[i] = 1.0f / (1.0f + expf(-val));
            }
        }
    }
    memcpy(gate_signal, gd->pyramidal_gate_output, hidden_size * sizeof(float));
    return 0;
}

/**
 * @brief 自适应带宽：根据输入动态调整门控带宽
 *
 * bandwidth_i = bandwidth_min + sigmoid(mean_i * x_std + std_i) * (bandwidth_max - bandwidth_min)
 * 高不确定性时带宽增大（更快响应），低不确定性时带宽减小（更稳定）。
 */
static int gated_cfc_adaptive_bandwidth(GatedCfCData* gd, const float* input,
                                         float* bandwidth, size_t input_size, size_t hidden_size) {
    if (!gd || !input || !bandwidth) return -1;
    (void)hidden_size;
    float input_mean = 0.0f, input_var = 0.0f;
    for (size_t i = 0; i < input_size; i++) input_mean += input[i];
    input_mean /= (float)input_size;
    for (size_t i = 0; i < input_size; i++) input_var += (input[i] - input_mean) * (input[i] - input_mean);
    input_var /= (float)input_size;
    float input_std = sqrtf(input_var + 1e-8f);
    float raw = gd->bandwidth_params[0] * input_std + gd->bandwidth_params[1];
    float gate = 1.0f / (1.0f + expf(-raw));
    bandwidth[0] = gd->bandwidth_min + gate * (gd->bandwidth_max - gd->bandwidth_min);
    return 0;
}

/**
 * @brief 启用门控CfC变体
 */
int cfc_cell_enable_gated_cfc(CfCCell* cell, const GatedCfCConfig* config) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    if (cell->gated_data) {
        /* 已启用，先清理 */
        GatedCfCData* gd = cell->gated_data;
        safe_free((void**)&gd->cross_gate_weights);
        safe_free((void**)&gd->cross_gate_grad);
        safe_free((void**)&gd->cross_gate_state);
        safe_free((void**)&gd->pyramidal_gate_weights);
        safe_free((void**)&gd->pyramidal_gate_biases);
        safe_free((void**)&gd->pyramidal_gate_weight_grad);
        safe_free((void**)&gd->pyramidal_gate_bias_grad);
        safe_free((void**)&gd->pyramidal_gate_output);
        safe_free((void**)&gd->bandwidth_params);
        safe_free((void**)&gd->bandwidth_grad);
        safe_free((void**)&gd->workspace);
        safe_free((void**)&cell->gated_data);
    }
    GatedCfCData* gd = (GatedCfCData*)safe_calloc(1, sizeof(GatedCfCData));
    if (!gd) return -1;
    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    GatedCfCConfig default_config = {
        1, 0, 0, 0.01f, 0.01f, 0.1f, 10.0f, 3
    };
    if (!config) config = &default_config;
    gd->use_cross_gating = config->enable_cross_gating;
    gd->use_pyramidal_gating = config->enable_pyramidal_gating;
    gd->use_adaptive_bandwidth = config->enable_adaptive_bandwidth;
    gd->bandwidth_min = config->bandwidth_min;
    gd->bandwidth_max = config->bandwidth_max;
    gd->pyramidal_layers = config->pyramidal_layers;
    if (gd->pyramidal_layers < 1) gd->pyramidal_layers = 1;
    if (gd->pyramidal_layers > 10) gd->pyramidal_layers = 10;
    if (gd->use_cross_gating) {
        size_t n_w = hidden_size * hidden_size;
        gd->cross_gate_weights = (float*)safe_calloc(n_w, sizeof(float));
        gd->cross_gate_grad = (float*)safe_calloc(n_w, sizeof(float));
        gd->cross_gate_state = (float*)safe_calloc(hidden_size, sizeof(float));
        if (!gd->cross_gate_weights || !gd->cross_gate_grad || !gd->cross_gate_state) {
            safe_free((void**)&gd->cross_gate_weights);
            safe_free((void**)&gd->cross_gate_grad);
            safe_free((void**)&gd->cross_gate_state);
            safe_free((void**)&cell->gated_data);
            return -1;
        }
        float scale = config->cross_gate_init_scale;
        unsigned int seed = 42;
        for (size_t i = 0; i < n_w; i++) {
            seed = seed * 1103515245 + 12345;
            gd->cross_gate_weights[i] = ((float)((seed >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 2.0f * scale;
        }
    }
    if (gd->use_pyramidal_gating) {
        size_t n_pw = (size_t)gd->pyramidal_layers * hidden_size;
        gd->pyramidal_gate_weights = (float*)safe_calloc(n_pw, sizeof(float));
        gd->pyramidal_gate_biases = (float*)safe_calloc(gd->pyramidal_layers, sizeof(float));
        gd->pyramidal_gate_weight_grad = (float*)safe_calloc(n_pw, sizeof(float));
        gd->pyramidal_gate_bias_grad = (float*)safe_calloc(gd->pyramidal_layers, sizeof(float));
        gd->pyramidal_gate_output = (float*)safe_calloc(hidden_size, sizeof(float));
        if (!gd->pyramidal_gate_weights || !gd->pyramidal_gate_biases ||
            !gd->pyramidal_gate_weight_grad || !gd->pyramidal_gate_bias_grad ||
            !gd->pyramidal_gate_output) {
            safe_free((void**)&gd->pyramidal_gate_weights);
            safe_free((void**)&gd->pyramidal_gate_biases);
            safe_free((void**)&gd->pyramidal_gate_weight_grad);
            safe_free((void**)&gd->pyramidal_gate_bias_grad);
            safe_free((void**)&gd->pyramidal_gate_output);
            safe_free((void**)&gd);
            return -1;
        }
        float scale = config->pyramidal_gate_init_scale;
        unsigned int seed = 84;
        for (size_t i = 0; i < n_pw; i++) {
            seed = seed * 1103515245 + 12345;
            gd->pyramidal_gate_weights[i] = ((float)((seed >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 2.0f * scale;
        }
    }
    if (gd->use_adaptive_bandwidth) {
        gd->bandwidth_params = (float*)safe_calloc(2, sizeof(float));
        gd->bandwidth_grad = (float*)safe_calloc(2, sizeof(float));
        if (!gd->bandwidth_params || !gd->bandwidth_grad) {
            safe_free((void**)&gd->bandwidth_params);
            safe_free((void**)&gd->bandwidth_grad);
            safe_free((void**)&gd);
            return -1;
        }
        gd->bandwidth_params[0] = 0.5f;
        gd->bandwidth_params[1] = 0.0f;
    }
    gd->workspace = (float*)safe_calloc(hidden_size + input_size, sizeof(float));
    if (!gd->workspace) {
        safe_free((void**)&gd);
        return -1;
    }
    cell->gated_data = gd;
    return 0;
}

/**
 * @brief 门控CfC前向传播
 */
int cfc_cell_forward_gated(CfCCell* cell, const float* input, float* hidden_state) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(input, "输入向量为空");
    SELFLNN_CHECK_NULL(hidden_state, "隐藏状态缓冲区为空");
    GatedCfCData* gd = cell->gated_data;
    if (!gd) return cfc_cell_forward(cell, input, hidden_state);
    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    float delta_t = cell->config.delta_t;
    float* h = hidden_state;
    float* h_ws = cell->state->state;
    // 保存前向传播前的状态 h(t)，供反向传播使用
    memcpy(cell->state->saved_state, h_ws, hidden_size * sizeof(float));
    float tau = cell->config.time_constant;
    float bandwidth = 1.0f;
    if (gd->use_adaptive_bandwidth) {
        gated_cfc_adaptive_bandwidth(gd, input, &bandwidth, input_size, hidden_size);
    }
    for (size_t i = 0; i < hidden_size; i++) {
        float gate_sum = cell->gate_biases[i * 3];
        float act_sum = cell->bias_vector[i];
        for (size_t j = 0; j < input_size; j++) {
            size_t idx = i * input_size + j;
            gate_sum += cell->input_gate_weights[idx] * input[j];
            act_sum += cell->weight_matrix[idx] * input[j];
        }
        for (size_t j = 0; j < hidden_size; j++) {
            size_t h_idx = i * hidden_size + j;
            gate_sum += cell->hidden_to_gate_weights[h_idx] * h[j];
            act_sum += cell->hidden_to_activation_weights[h_idx] * h[j];
        }
        float raw_gate = gate_sum * bandwidth;
        if (gd->use_cross_gating) {
            gated_cfc_cross_gate_update(gd, &raw_gate, &act_sum, &raw_gate, hidden_size);
        }
        if (gd->use_pyramidal_gating) {
            gated_cfc_pyramidal_gate(gd, &raw_gate, hidden_size);
        }
        float gate = 1.0f / (1.0f + expf(-raw_gate));
        float activation = tanhf(act_sum);
        float driver = gate * activation;
        float tau_i = cell->use_adaptive_tau ? cell->time_constants[i] : tau;
        if (tau_i < cell->min_time_constant) tau_i = cell->min_time_constant;
        float exp_term = expf(-delta_t / (tau_i * bandwidth));
        h_ws[i] = h_ws[i] * exp_term + driver * (1.0f - exp_term);
        h[i] = h_ws[i];
        cell->state->activation[i] = activation;
    }
    return 0;
}

/**
 * @brief 门控CfC反向传播
 */
int cfc_cell_backward_gated(CfCCell* cell, const float* gradient, float* input_gradient) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(gradient, "梯度向量为空");
    GatedCfCData* gd = cell->gated_data;
    if (!gd) return cfc_cell_backward(cell, gradient, input_gradient);
    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    float delta_t = cell->config.delta_t;
    float tau = cell->config.time_constant;
    float* h = cell->state->saved_state;  // 使用前向传播前保存的状态 h(t)
    float* act = cell->state->activation;
    for (size_t i = 0; i < hidden_size; i++) {
        float tau_i = cell->use_adaptive_tau ? cell->time_constants[i] : tau;
        if (tau_i < cell->min_time_constant) tau_i = cell->min_time_constant;
        float exp_term = expf(-delta_t / tau_i);
        float dL_dh = gradient[i];
        float d_driver = dL_dh * (1.0f - exp_term);
        float gate = 1.0f / (1.0f + expf(-gd->cross_gate_state[i]));
        float gate_deriv = gate * (1.0f - gate);
        float act_deriv = 1.0f - act[i] * act[i];
        float dL_dgate = d_driver * act[i];
        float dL_dact = d_driver * gate;
        float dL_dgate_input = dL_dgate * gate_deriv;
        float dL_dact_input = dL_dact * act_deriv;

        for (size_t j = 0; j < input_size; j++) {
            size_t idx = i * input_size + j;
            cell->input_gate_weight_grad[idx] += dL_dgate_input * cell->state->input_buffer[j];
            cell->weight_grad[idx] += dL_dact_input * cell->state->input_buffer[j];
            if (input_gradient) {
                input_gradient[j] += dL_dgate_input * cell->input_gate_weights[idx]
                                   + dL_dact_input * cell->weight_matrix[idx];
            }
        }
        for (size_t j = 0; j < hidden_size; j++) {
            size_t h_idx = i * hidden_size + j;
            cell->hidden_to_gate_weight_grad[h_idx] += dL_dgate_input * h[j];
            cell->hidden_to_activation_weight_grad[h_idx] += dL_dact_input * h[j];
            if (gd->use_cross_gating) {
                gd->cross_gate_grad[i * hidden_size + j] += dL_dgate_input * gd->cross_gate_state[j];
            }
        }
        cell->gate_bias_grad[i * 3] += dL_dgate_input;
        cell->bias_grad[i] += dL_dact_input;
        if (gd->use_pyramidal_gating) {
            for (int l = 0; l < gd->pyramidal_layers; l++) {
                float pyr_gate = 1.0f / (1.0f + expf(-(gd->pyramidal_gate_weights[l * hidden_size + i]
                    * (l == 0 ? gd->cross_gate_state[i] : gd->pyramidal_gate_output[i])
                    + gd->pyramidal_gate_biases[l])));
                float pyr_gate_deriv = pyr_gate * (1.0f - pyr_gate);
                gd->pyramidal_gate_weight_grad[l * hidden_size + i] += dL_dgate_input * pyr_gate_deriv * gd->pyramidal_gate_output[i];
                gd->pyramidal_gate_bias_grad[l] += dL_dgate_input * pyr_gate_deriv;
            }
        }
    }
    if (gd->use_adaptive_bandwidth) {
        float bw_grad = 0.0f;
        for (size_t i = 0; i < hidden_size; i++) {
            float tau_i = cell->use_adaptive_tau ? cell->time_constants[i] : tau;
            float exp_term = expf(-delta_t / tau_i);
            float dbw = (delta_t / (tau_i * tau_i)) * exp_term * (h[i] - act[i]);
            bw_grad += gradient[i] * dbw;
        }
        float input_std = 0.0f;
        float input_mean = 0.0f;
        for (size_t i = 0; i < input_size; i++) input_mean += cell->state->input_buffer[i];
        input_mean /= (float)input_size;
        for (size_t i = 0; i < input_size; i++) input_std += (cell->state->input_buffer[i] - input_mean)
            * (cell->state->input_buffer[i] - input_mean);
        input_std = sqrtf(input_std / (float)input_size + 1e-8f);
        float sig = 1.0f / (1.0f + expf(-(gd->bandwidth_params[0] * input_std + gd->bandwidth_params[1])));
        float sig_deriv = sig * (1.0f - sig);
        gd->bandwidth_grad[0] += bw_grad * (gd->bandwidth_max - gd->bandwidth_min) * sig_deriv * input_std;
        gd->bandwidth_grad[1] += bw_grad * (gd->bandwidth_max - gd->bandwidth_min) * sig_deriv;
    }
    return 0;
}

/**
 * @brief 获取门控CfC状态信息
 */
int cfc_cell_get_gated_stats(const CfCCell* cell, float* cross_gate_values,
                             float* pyramidal_gate_values, float* bandwidth_values) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    GatedCfCData* gd = cell->gated_data;
    if (!gd) return -1;
    size_t hidden_size = cell->config.hidden_size;
    if (cross_gate_values && gd->use_cross_gating) {
        memcpy(cross_gate_values, gd->cross_gate_state, hidden_size * sizeof(float));
    }
    if (pyramidal_gate_values && gd->use_pyramidal_gating) {
        memcpy(pyramidal_gate_values, gd->pyramidal_gate_output, hidden_size * sizeof(float));
    }
    if (bandwidth_values && gd->use_adaptive_bandwidth) {
        bandwidth_values[0] = gd->bandwidth_params[0];
        bandwidth_values[1] = gd->bandwidth_params[1];
    }
    return 0;
}

/* ==========================================================================
 * 分层CfC（Hierarchical CfC）实现
 * ========================================================================== */

/**
 * @brief 更新单个层次的状态
 *
 * 第l层使用自身的状态、时间常数以及来自上下层的信息进行更新。
 */
static int hierarchical_level_update(HierarchicalCfCData* hd, int level,
                                      const float* input, float* merged_state,
                                      size_t hidden_size, size_t input_size, float delta_t) {
    if (!hd || !input || !merged_state) return -1;
    float* h_level = hd->level_states + (size_t)level * hidden_size;
    float* a_level = hd->level_activations + (size_t)level * hidden_size;
    float tau_level = hd->level_time_constants[level];
    for (size_t i = 0; i < hidden_size; i++) {
        float gate_sum = 0.0f, act_sum = 0.0f;
        for (size_t j = 0; j < input_size; j++) {
            gate_sum += input[j] * 0.1f;
            act_sum += input[j] * 0.1f;
        }
        if (level > 0 && hd->enable_bidirectional) {
            float* h_lower = hd->level_states + (size_t)(level - 1) * hidden_size;
            gate_sum += hd->bottom_up_strength * h_lower[i];
            act_sum += hd->bottom_up_strength * h_lower[i] * 0.5f;
        }
        if (level < hd->num_levels - 1 && hd->enable_bidirectional) {
            float* h_upper = hd->level_states + (size_t)(level + 1) * hidden_size;
            gate_sum += hd->top_down_strength * h_upper[i];
            act_sum += hd->top_down_strength * h_upper[i] * 0.5f;
        }
        float gate = 1.0f / (1.0f + expf(-gate_sum));
        float activation = tanhf(act_sum);
        float driver = gate * activation;
        float exp_term = expf(-delta_t / tau_level);
        h_level[i] = h_level[i] * exp_term + driver * (1.0f - exp_term);
        a_level[i] = activation;
        merged_state[i] += h_level[i] / (float)hd->num_levels;
    }
    return 0;
}

/**
 * @brief 启用分层CfC
 */
int cfc_cell_enable_hierarchical(CfCCell* cell, const HierarchicalCfCConfig* config) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    if (cell->hierarchical_data) {
        HierarchicalCfCData* hd = cell->hierarchical_data;
        safe_free((void**)&hd->level_time_constants);
        safe_free((void**)&hd->level_states);
        safe_free((void**)&hd->level_activations);
        safe_free((void**)&hd->level_gradients);
        safe_free((void**)&hd->bottom_up_weights);
        safe_free((void**)&hd->bottom_up_weight_grad);
        safe_free((void**)&hd->top_down_weights);
        safe_free((void**)&hd->top_down_weight_grad);
        safe_free((void**)&hd->workspace);
        safe_free((void**)&cell->hierarchical_data);
    }
    HierarchicalCfCData* hd = (HierarchicalCfCData*)safe_calloc(1, sizeof(HierarchicalCfCData));
    if (!hd) return -1;
    size_t hidden_size = cell->config.hidden_size;
    HierarchicalCfCConfig default_config = {3, 0.01f, 1.0f, 0.5f, 0.3f, 1, 0.01f};
    if (!config) config = &default_config;
    hd->num_levels = config->num_levels;
    if (hd->num_levels < 2) hd->num_levels = 2;
    if (hd->num_levels > 10) hd->num_levels = 10;
    hd->bottom_up_strength = config->bottom_up_strength;
    hd->top_down_strength = config->top_down_strength;
    hd->enable_bidirectional = config->enable_bidirectional;
    size_t n_levels = (size_t)hd->num_levels;
    hd->level_time_constants = (float*)safe_calloc(n_levels, sizeof(float));
    hd->level_states = (float*)safe_calloc(n_levels * hidden_size, sizeof(float));
    hd->level_activations = (float*)safe_calloc(n_levels * hidden_size, sizeof(float));
    hd->level_gradients = (float*)safe_calloc(n_levels * hidden_size, sizeof(float));
    hd->bottom_up_weights = (float*)safe_calloc((n_levels - 1) * hidden_size, sizeof(float));
    hd->bottom_up_weight_grad = (float*)safe_calloc((n_levels - 1) * hidden_size, sizeof(float));
    hd->top_down_weights = (float*)safe_calloc((n_levels - 1) * hidden_size, sizeof(float));
    hd->top_down_weight_grad = (float*)safe_calloc((n_levels - 1) * hidden_size, sizeof(float));
    hd->workspace = (float*)safe_calloc(hidden_size * 2, sizeof(float));
    if (!hd->level_time_constants || !hd->level_states || !hd->level_activations ||
        !hd->level_gradients || !hd->bottom_up_weights || !hd->bottom_up_weight_grad ||
        !hd->top_down_weights || !hd->top_down_weight_grad || !hd->workspace) {
        safe_free((void**)&hd->level_time_constants);
        safe_free((void**)&hd->level_states);
        safe_free((void**)&hd->level_activations);
        safe_free((void**)&hd->level_gradients);
        safe_free((void**)&hd->bottom_up_weights);
        safe_free((void**)&hd->bottom_up_weight_grad);
        safe_free((void**)&hd->top_down_weights);
        safe_free((void**)&hd->top_down_weight_grad);
        safe_free((void**)&hd->workspace);
        safe_free((void**)&hd);
        return -1;
    }
    float fast_tau = config->fast_time_constant;
    float slow_tau = config->slow_time_constant;
    for (int l = 0; l < hd->num_levels; l++) {
        float ratio = (float)l / (float)(hd->num_levels - 1);
        hd->level_time_constants[l] = fast_tau * powf(slow_tau / fast_tau, ratio);
    }
    float scale = config->level_init_scale;
    unsigned int seed = 168;
    size_t n_bu = (n_levels - 1) * hidden_size;
    for (size_t i = 0; i < n_bu; i++) {
        seed = seed * 1103515245 + 12345;
        float v = ((float)((seed >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 2.0f * scale;
        hd->bottom_up_weights[i] = v;
        hd->top_down_weights[i] = v * 0.5f;
    }
    cell->hierarchical_data = hd;
    return 0;
}

/**
 * @brief 分层CfC前向传播
 */
int cfc_cell_forward_hierarchical(CfCCell* cell, const float* input, float* hidden_state) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(input, "输入向量为空");
    SELFLNN_CHECK_NULL(hidden_state, "隐藏状态缓冲区为空");
    HierarchicalCfCData* hd = cell->hierarchical_data;
    if (!hd) return cfc_cell_forward(cell, input, hidden_state);
    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    float delta_t = cell->config.delta_t;
    memset(hidden_state, 0, hidden_size * sizeof(float));
    /* 自底向上更新 */
    for (int l = 0; l < hd->num_levels; l++) {
        hierarchical_level_update(hd, l, input, hidden_state, hidden_size, input_size, delta_t);
    }
    if (hd->enable_bidirectional) {
        /* 自顶向下精化 */
        for (int l = hd->num_levels - 1; l >= 0; l--) {
            float* h_level = hd->level_states + (size_t)l * hidden_size;
            if (l < hd->num_levels - 1) {
                float* h_upper = hd->level_states + (size_t)(l + 1) * hidden_size;
                for (size_t i = 0; i < hidden_size; i++) {
                    h_level[i] += hd->top_down_strength * hd->top_down_weights[(size_t)l * hidden_size + i] * h_upper[i];
                }
            }
        }
        /* 重新融合 */
        memset(hidden_state, 0, hidden_size * sizeof(float));
        for (int l = 0; l < hd->num_levels; l++) {
            float* h_level = hd->level_states + (size_t)l * hidden_size;
            for (size_t i = 0; i < hidden_size; i++) {
                hidden_state[i] += h_level[i] / (float)hd->num_levels;
            }
        }
    }
    memcpy(cell->state->state, hidden_state, hidden_size * sizeof(float));
    return 0;
}

/**
 * @brief 分层CfC反向传播
 */
int cfc_cell_backward_hierarchical(CfCCell* cell, const float* gradient, float* input_gradient) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(gradient, "梯度向量为空");
    HierarchicalCfCData* hd = cell->hierarchical_data;
    if (!hd) return cfc_cell_backward(cell, gradient, input_gradient);
    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    float delta_t = cell->config.delta_t;
    /* 跨层次分配合并梯度 */
    for (int l = 0; l < hd->num_levels; l++) {
        for (size_t i = 0; i < hidden_size; i++) {
            hd->level_gradients[(size_t)l * hidden_size + i] = gradient[i] / (float)hd->num_levels;
        }
    }
    /* 自顶向下传播梯度 */
    for (int l = hd->num_levels - 1; l >= 0; l--) {
        float* h_level = hd->level_states + (size_t)l * hidden_size;
        float* a_level = hd->level_activations + (size_t)l * hidden_size;
        float* h_grad = hd->level_gradients + (size_t)l * hidden_size;
        float tau_level = hd->level_time_constants[l];
        for (size_t i = 0; i < hidden_size; i++) {
            float tau_i = tau_level;
            float exp_term = expf(-delta_t / tau_i);
            float d_driver = h_grad[i] * (1.0f - exp_term);
            float gate_val = 0.0f, act_val = a_level[i];
            gate_val = 1.0f / (1.0f + expf(-h_level[i]));
            float gate_deriv = gate_val * (1.0f - gate_val);
            float act_deriv = 1.0f - act_val * act_val;
            float dL_dgate = d_driver * act_val;
            float dL_dact = d_driver * gate_val;
            float dL_dgate_input = dL_dgate * gate_deriv;
            float dL_dact_input = dL_dact * act_deriv;
            if (input_gradient) {
                for (size_t j = 0; j < input_size; j++) {
                    input_gradient[j] += (dL_dgate_input + dL_dact_input) * 0.1f;
                }
            }
            if (l > 0) {
                hd->bottom_up_weight_grad[(size_t)(l - 1) * hidden_size + i] += dL_dgate_input * h_level[i];
            }
            if (l < hd->num_levels - 1) {
                float* h_upper_grad = hd->level_gradients + (size_t)(l + 1) * hidden_size;
                h_upper_grad[i] += hd->top_down_strength * hd->top_down_weights[(size_t)l * hidden_size + i] * h_grad[i];
                hd->top_down_weight_grad[(size_t)l * hidden_size + i] += h_grad[i] * hd->top_down_strength * h_level[i];
            }
        }
    }
    return 0;
}

/**
 * @brief 获取分层CfC指定层次状态
 */
int cfc_cell_get_hierarchical_state(const CfCCell* cell, int level, float* state_out) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(state_out, "状态输出缓冲区为空");
    HierarchicalCfCData* hd = cell->hierarchical_data;
    if (!hd || level < 0 || level >= hd->num_levels) return -1;
    size_t hidden_size = cell->config.hidden_size;
    memcpy(state_out, hd->level_states + (size_t)level * hidden_size, hidden_size * sizeof(float));
    return 0;
}

/* ==========================================================================
 * 液态记忆门控CfC（Liquid Memory Gating CfC）实现
 * ========================================================================== */

/**
 * @brief 计算液态记忆调制信号
 *
 * 使用CfC自身门控机制计算调制信号：
 *   concat = [hidden_state, input]          // 状态-输入拼接
 *   gate = σ(W_gate · concat + b_gate)      // CfC门控
 *   act = tanh(W_act · concat + b_act)      // CfC激活
 *   modulation = gate ⊙ act                 // 调制信号
 *
 * 完全基于CfC闭式解门控方程，无QKV投影、无softmax、无注意力权重。
 */
static int liquid_memory_compute_modulation(LiquidMemoryCfCData* ld, const float* hidden_state,
                                             const float* input, size_t hidden_size,
                                             size_t input_size) {
    if (!ld || !hidden_state || !input) return -1;
    size_t concat_dim = hidden_size + input_size;
    /* 拼接隐藏状态和输入 */
    memcpy(ld->concat_buffer, hidden_state, hidden_size * sizeof(float));
    memcpy(ld->concat_buffer + hidden_size, input, input_size * sizeof(float));
    /* 计算门控信号: gate = σ(W_gate · concat + b_gate) */
    for (size_t i = 0; i < hidden_size; i++) {
        float sum = ld->gate_biases[i];
        for (size_t j = 0; j < concat_dim; j++) {
            sum += ld->gate_weights[i * concat_dim + j] * ld->concat_buffer[j];
        }
        ld->gate_signal[i] = 1.0f / (1.0f + expf(-sum * ld->gate_scale));
    }
    /* 计算激活信号: act = tanh(W_act · concat + b_act) */
    for (size_t i = 0; i < hidden_size; i++) {
        float sum = ld->activation_biases[i];
        for (size_t j = 0; j < concat_dim; j++) {
            sum += ld->activation_weights[i * concat_dim + j] * ld->concat_buffer[j];
        }
        ld->modulation_output[i] = tanhf(sum);
    }
    /* 调制信号 = gate ⊙ act */
    for (size_t i = 0; i < hidden_size; i++) {
        ld->modulation_output[i] = ld->gate_signal[i] * ld->modulation_output[i];
    }
    return 0;
}

/**
 * @brief 启用液态记忆门控CfC
 */
int cfc_cell_enable_liquid_memory(CfCCell* cell, const LiquidMemoryCfCConfig* config) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    if (cell->liquid_memory_data) {
        LiquidMemoryCfCData* ld = cell->liquid_memory_data;
        safe_free((void**)&ld->gate_weights);
        safe_free((void**)&ld->activation_weights);
        safe_free((void**)&ld->gate_biases);
        safe_free((void**)&ld->activation_biases);
        safe_free((void**)&ld->gate_weights_grad);
        safe_free((void**)&ld->activation_weights_grad);
        safe_free((void**)&ld->gate_bias_grad);
        safe_free((void**)&ld->activation_bias_grad);
        safe_free((void**)&ld->modulation_output);
        safe_free((void**)&ld->gate_signal);
        safe_free((void**)&ld->concat_buffer);
        safe_free((void**)&ld->workspace);
        safe_free((void**)&cell->liquid_memory_data);
    }
    LiquidMemoryCfCData* ld = (LiquidMemoryCfCData*)safe_calloc(1, sizeof(LiquidMemoryCfCData));
    if (!ld) return -1;
    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    LiquidMemoryCfCConfig default_config = {1, 0.1f, 2.0f, 1.0f};
    if (!config) config = &default_config;
    ld->enable_memory_gating = config->enable_memory_gating;
    ld->init_scale = config->init_scale;
    ld->gate_bias_init = config->gate_bias_init;
    ld->gate_scale = config->gate_scale;
    ld->concat_dim = hidden_size + input_size;
    size_t concat_dim = ld->concat_dim;
    size_t n_w = hidden_size * concat_dim;
    ld->gate_weights = (float*)safe_calloc(n_w, sizeof(float));
    ld->activation_weights = (float*)safe_calloc(n_w, sizeof(float));
    ld->gate_biases = (float*)safe_calloc(hidden_size, sizeof(float));
    ld->activation_biases = (float*)safe_calloc(hidden_size, sizeof(float));
    ld->gate_weights_grad = (float*)safe_calloc(n_w, sizeof(float));
    ld->activation_weights_grad = (float*)safe_calloc(n_w, sizeof(float));
    ld->gate_bias_grad = (float*)safe_calloc(hidden_size, sizeof(float));
    ld->activation_bias_grad = (float*)safe_calloc(hidden_size, sizeof(float));
    ld->modulation_output = (float*)safe_calloc(hidden_size, sizeof(float));
    ld->gate_signal = (float*)safe_calloc(hidden_size, sizeof(float));
    ld->concat_buffer = (float*)safe_calloc(concat_dim, sizeof(float));
    ld->workspace = (float*)safe_calloc(hidden_size, sizeof(float));
    if (!ld->gate_weights || !ld->activation_weights || !ld->gate_biases ||
        !ld->activation_biases || !ld->gate_weights_grad || !ld->activation_weights_grad ||
        !ld->gate_bias_grad || !ld->activation_bias_grad ||
        !ld->modulation_output || !ld->gate_signal || !ld->concat_buffer || !ld->workspace) {
        safe_free((void**)&ld->gate_weights);
        safe_free((void**)&ld->activation_weights);
        safe_free((void**)&ld->gate_biases);
        safe_free((void**)&ld->activation_biases);
        safe_free((void**)&ld->gate_weights_grad);
        safe_free((void**)&ld->activation_weights_grad);
        safe_free((void**)&ld->gate_bias_grad);
        safe_free((void**)&ld->activation_bias_grad);
        safe_free((void**)&ld->modulation_output);
        safe_free((void**)&ld->gate_signal);
        safe_free((void**)&ld->concat_buffer);
        safe_free((void**)&ld->workspace);
        safe_free((void**)&ld);
        return -1;
    }
    float scale = ld->init_scale;
    float gate_bias = ld->gate_bias_init;
    unsigned int seed = 336;
    for (size_t i = 0; i < hidden_size; i++) {
        for (size_t j = 0; j < concat_dim; j++) {
            seed = seed * 1103515245 + 12345;
            ld->gate_weights[i * concat_dim + j] = ((float)((seed >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 2.0f * scale;
            seed = seed * 1103515245 + 12345;
            ld->activation_weights[i * concat_dim + j] = ((float)((seed >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 2.0f * scale;
        }
        ld->gate_biases[i] = gate_bias;
        ld->activation_biases[i] = 0.0f;
    }
    cell->liquid_memory_data = ld;
    return 0;
}

/**
 * @brief 液态记忆门控CfC前向传播
 */
int cfc_cell_forward_liquid_memory(CfCCell* cell, const float* input, float* hidden_state) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(input, "输入向量为空");
    SELFLNN_CHECK_NULL(hidden_state, "隐藏状态缓冲区为空");
    LiquidMemoryCfCData* ld = cell->liquid_memory_data;
    if (!ld) return cfc_cell_forward(cell, input, hidden_state);
    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    float delta_t = cell->config.delta_t;
    float* h = cell->state->state;  // 前向传播计算使用 h(t)，结果直接写回 state
    // 保存前向传播前的状态 h(t)，供反向传播使用
    memcpy(cell->state->saved_state, h, hidden_size * sizeof(float));
    /* 计算液态记忆调制信号 */
    memset(ld->modulation_output, 0, hidden_size * sizeof(float));
    if (ld->enable_memory_gating) {
        liquid_memory_compute_modulation(ld, h, input, hidden_size, input_size);
    }
    float* modulation = ld->modulation_output;
    /* 标准CfC闭式解 + 液态记忆调制 */
    for (size_t i = 0; i < hidden_size; i++) {
        float gate_sum = cell->gate_biases[i * 3];
        float act_sum = cell->bias_vector[i];
        for (size_t j = 0; j < input_size; j++) {
            size_t idx = i * input_size + j;
            gate_sum += cell->input_gate_weights[idx] * input[j];
            act_sum += cell->weight_matrix[idx] * input[j];
        }
        for (size_t j = 0; j < hidden_size; j++) {
            size_t h_idx = i * hidden_size + j;
            gate_sum += cell->hidden_to_gate_weights[h_idx] * h[j];
            act_sum += cell->hidden_to_activation_weights[h_idx] * h[j];
        }
        act_sum += modulation[i];
        float gate = 1.0f / (1.0f + expf(-gate_sum));
        float activation = tanhf(act_sum);
        float driver = gate * activation;
        float tau_i = cell->use_adaptive_tau ? cell->time_constants[i] : cell->config.time_constant;
        if (tau_i < cell->min_time_constant) tau_i = cell->min_time_constant;
        float exp_term = expf(-delta_t / tau_i);
        h[i] = h[i] * exp_term + driver * (1.0f - exp_term);
        hidden_state[i] = h[i];
        cell->state->activation[i] = activation;
    }
    return 0;
}

/**
 * @brief 液态记忆门控CfC反向传播
 */
int cfc_cell_backward_liquid_memory(CfCCell* cell, const float* gradient, float* input_gradient) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(gradient, "梯度向量为空");
    LiquidMemoryCfCData* ld = cell->liquid_memory_data;
    if (!ld) return cfc_cell_backward(cell, gradient, input_gradient);
    size_t hidden_size = cell->config.hidden_size;
    size_t input_size = cell->config.input_size;
    size_t concat_dim = ld->concat_dim;
    float delta_t = cell->config.delta_t;
    float* h = cell->state->saved_state;  // 使用前向传播前保存的状态 h(t)
    float* act = cell->state->activation;
    for (size_t i = 0; i < hidden_size; i++) {
        float tau_i = cell->use_adaptive_tau ? cell->time_constants[i] : cell->config.time_constant;
        if (tau_i < cell->min_time_constant) tau_i = cell->min_time_constant;
        float exp_term = expf(-delta_t / tau_i);
        float dL_dh = gradient[i];
        float d_driver = dL_dh * (1.0f - exp_term);
        float gate_val = 1.0f / (1.0f + expf(-(h[i] + ld->modulation_output[i])));
        float gate_deriv = gate_val * (1.0f - gate_val);
        float act_deriv = 1.0f - act[i] * act[i];
        float dL_dgate = d_driver * act[i];
        float dL_dact = d_driver * gate_val;
        float dL_dgate_input = dL_dgate * gate_deriv;
        float dL_dact_input = dL_dact * act_deriv;
        for (size_t j = 0; j < input_size; j++) {
            size_t idx = i * input_size + j;
            cell->input_gate_weight_grad[idx] += dL_dgate_input * cell->state->input_buffer[j];
            cell->weight_grad[idx] += dL_dact_input * cell->state->input_buffer[j];
            if (input_gradient) {
                input_gradient[j] += dL_dgate_input * cell->input_gate_weights[idx]
                                   + dL_dact_input * cell->weight_matrix[idx];
            }
        }
        for (size_t j = 0; j < hidden_size; j++) {
            size_t h_idx = i * hidden_size + j;
            cell->hidden_to_gate_weight_grad[h_idx] += dL_dgate_input * h[j];
            cell->hidden_to_activation_weight_grad[h_idx] += dL_dact_input * h[j];
        }
        cell->gate_bias_grad[i * 3] += dL_dgate_input;
        cell->bias_grad[i] += dL_dact_input;
        ld->workspace[i] = dL_dact_input;
    }
    /* 液态记忆门控模块梯度 */
    if (ld->enable_memory_gating) {
        for (size_t i = 0; i < hidden_size; i++) {
            float dL_dmod = ld->workspace[i];
            float gate_i = ld->gate_signal[i];
            float mod_act = ld->modulation_output[i] / (gate_i + 1e-8f);
            float dL_dgate = dL_dmod * mod_act;
            float dL_dact = dL_dmod * gate_i;
            float act_deriv_local = 1.0f - mod_act * mod_act;
            float dL_dgate_raw = dL_dgate * gate_i * (1.0f - gate_i) * ld->gate_scale;
            float dL_dact_raw = dL_dact * act_deriv_local;
            for (size_t j = 0; j < concat_dim; j++) {
                ld->gate_weights_grad[i * concat_dim + j] += dL_dgate_raw * ld->concat_buffer[j];
                ld->activation_weights_grad[i * concat_dim + j] += dL_dact_raw * ld->concat_buffer[j];
            }
            ld->gate_bias_grad[i] += dL_dgate_raw;
            ld->activation_bias_grad[i] += dL_dact_raw;
        }
    }
    return 0;
}

/**
 * @brief 获取液态记忆门控CfC的当前记忆调制状态
 */
int cfc_cell_get_liquid_memory_state(const CfCCell* cell, float* modulation_out, float* gate_out) {
    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    SELFLNN_CHECK_NULL(modulation_out, "调制信号输出缓冲区为空");
    LiquidMemoryCfCData* ld = cell->liquid_memory_data;
    if (!ld) return -1;
    size_t hidden_size = cell->config.hidden_size;
    memcpy(modulation_out, ld->modulation_output, hidden_size * sizeof(float));
    if (gate_out) {
        memcpy(gate_out, ld->gate_signal, hidden_size * sizeof(float));
    }
    return 0;
}

/* ============================================================================
 * CORE-15: 分层CfC双向信息流 (Hierarchical Bidirectional Flow)
 *
 * 自底向上(BU): 底层特征→顶层抽象, 逐层池化压缩
 * 自顶向下(TD): 顶层上下文→底层细化, 逐层上采样调制
 * ============================================================================ */

int cfc_cell_hierarchical_forward(CfCCell* cell, const float* input,
                                    float* output, float* bu_flows, float* td_flows,
                                    int num_levels) {
    if (!cell || !input || !output || num_levels <= 1) return -1;
    size_t h = cell->config.hidden_size;
    int per_level = (int)h / num_levels;
    if (per_level < 8) per_level = 8;

    /* 自底向上: 逐层压缩 */
    for (int l = 0; l < num_levels; l++) {
        int start = l * per_level;
        for (int i = 0; i < per_level && start + i < (int)h; i++) {
            float sum = 0.0f;
            int base_n = (l > 0) ? per_level : (int)cell->config.input_size;
            for (int j = 0; j < base_n && j < (int)h; j++)
                sum += input[j] * 0.5f;
            bu_flows[start + i] = sum / (float)(base_n + 1) + input[start + i] * 0.5f;
        }
    }

    /* CfC核心处理 */
    cfc_cell_forward(cell, input, output);

    /* 自顶向下: 顶层调制底层 */
    for (int l = num_levels - 1; l >= 0; l--) {
        int start = l * per_level;
        for (int i = 0; i < per_level && start + i < (int)h; i++) {
            float td_mod = 0.0f;
            if (l < num_levels - 1)
                for (int j = 0; j < per_level; j++)
                    td_mod += output[(l + 1) * per_level + j] * 0.3f;
            td_flows[start + i] = output[start + i] * (1.0f + tanhf(td_mod * 0.1f));
        }
    }

    for (size_t i = 0; i < h; i++) output[i] = td_flows[i];
    return 0;
}

/* ============================================================================
 * CORE-16: 液态记忆门控调制
 *
 * 记忆调制信号: m_t = σ(W_m·[h_{t-1}, x_t])
 * 调制后的状态: h'_t = h_t ⊙ m_t (element-wise gate)
 * ============================================================================ */

int cfc_cell_memory_gate_apply(CfCCell* cell, const float* hidden_pre,
                                 const float* input, float* hidden_post,
                                 float memory_decay) {
    if (!cell || !hidden_pre || !input || !hidden_post) return -1;
    size_t h = cell->config.hidden_size;

    if (memory_decay <= 0.0f) memory_decay = 0.5f;

    for (size_t i = 0; i < h; i++) {
        float gate_input = hidden_pre[i] * 0.3f + input[i % cell->config.input_size] * 0.7f;
        float memory_gate = 1.0f / (1.0f + expf(-gate_input));
        hidden_post[i] = hidden_pre[i] * memory_gate * (1.0f - memory_decay)
                       + hidden_pre[i] * memory_decay;
        if (fabsf(hidden_post[i]) > 10.0f)
            hidden_post[i] = (hidden_post[i] > 0.0f) ? 10.0f : -10.0f;
    }
    return 0;
}

/* CORE-23: 液时域层归一化 (LayerNorm on liquid tau projections) */
int cfc_liquid_layernorm(float* tau_projection, size_t hidden_size) {
    if (!tau_projection || hidden_size == 0) return -1;
    float mean = 0.0f, var = 0.0f;
    for (size_t i = 0; i < hidden_size; i++) mean += tau_projection[i];
    mean /= (float)hidden_size;
    for (size_t i = 0; i < hidden_size; i++) { float d = tau_projection[i] - mean; var += d * d; }
    var = sqrtf(var / (float)hidden_size + 1e-5f);
    for (size_t i = 0; i < hidden_size; i++) tau_projection[i] = (tau_projection[i] - mean) / var;
    return 0;
}
#endif /* SELFLNN_CORE_INTERNAL */
