/**
 * @file cfc_cell.h
 * @brief 封闭形式连续时间单元（CfC Cell）
 * 
 * CfC（Closed-form Continuous-time）单元是液态神经网络的基本构建块，
 * 提供连续时间动态和封闭形式解，提高训练效率和稳定性。
 */

#ifndef SELFLNN_CFC_CELL_H
#define SELFLNN_CFC_CELL_H

#include <stddef.h>
#include <stdio.h>

#include "selflnn/core/ode_solvers.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============ 前向声明内部数据结构 ============ */

/** @brief 门控CfC变体内部数据（不透明指针） */
typedef struct GatedCfCData GatedCfCData;

/** @brief 分层CfC内部数据（不透明指针） */
typedef struct HierarchicalCfCData HierarchicalCfCData;

/** @brief 液态记忆门控CfC内部数据（不透明指针） */
typedef struct LiquidMemoryCfCData LiquidMemoryCfCData;

/**
 * @brief ODE求解器类型
 */
typedef enum {
    ODE_SOLVER_CLOSED_FORM = 0, /**< 封闭形式解（默认，精确解析解） */
    ODE_SOLVER_RK4 = 1,         /**< 四阶龙格-库塔法（固定步长数值解） */
    ODE_SOLVER_RK45 = 2,        /**< 五阶龙格-库塔-费尔伯格法（自适应步长数值解） */
    ODE_SOLVER_CTBP = 3,        /**< 连续时间反向传播法（精确梯度计算） */
    ODE_SOLVER_DP54 = 4,        /**< Dormand-Prince 5(4)自适应步长求解器（7阶Butcher table，更优误差控制） */
    ODE_SOLVER_ROSENBROCK = 5,  /**< Rosenbrock刚性方程求解器（隐式RK-L稳定，支持刚性动态） */
    ODE_SOLVER_SYMPLECTIC = 6,  /**< Forest-Ruth辛积分器（4阶辛几何，保守系统能量守恒） */
    ODE_SOLVER_RHS = 7          /**< P2-041修复: 直接RHS评估器（前向欧拉步，使用cfc_cell_compute_rhs作为备选求解器） */
} OdeSolverType;

/**
 * @brief RK45自适应求解器配置
 */
typedef struct {
    float rel_tolerance;     /**< 相对误差容限，默认1e-6 */
    float abs_tolerance;     /**< 绝对误差容限，默认1e-8 */
    float min_step_size;     /**< 最小步长，默认1e-6 */
    float max_step_size;     /**< 最大步长，默认1.0 */
    float safety_factor;     /**< 步长调整安全因子，默认0.9 */
    int max_iterations;      /**< 最大迭代次数，默认10000 */
} RK45Config;

/**
 * @brief CTBP连续时间反向传播配置
 */
typedef struct {
    int use_adjoint_method;      /**< 是否使用伴随法（Adjoint Method），1=是0=直接法 */
    float gradient_clip_norm;    /**< 梯度裁剪范数，0=不裁剪 */
    int store_intermediate;      /**< 是否存储中间状态用于精确梯度计算 */
    int max_checkpoints;         /**< 最大检查点数量，用于权衡内存和精度 */
} CTBPConfig;

/**
 * @brief CfC单元配置
 */
typedef struct {
    size_t input_size;             /**< 输入向量大小 */
    size_t hidden_size;            /**< 隐藏状态大小 */
    float time_constant;           /**< 时间常数 */
    float noise_std;               /**< 噪声标准差 */
    int enable_adaptation;         /**< 是否启用参数自适应 */
    float delta_t;                 /**< 时间步长（秒），默认1.0，支持连续时间动态 */
    int delta_t_explicitly_set;    /**< P2-042修复: 调用者是否显式设置了delta_t（0=使用默认，1=显式设置），解决1.0既是合法配置又是默认值的歧义 */
    int ode_solver_type;           /**< ODE求解器类型：0=闭式解，1=RK4，2=RK45，3=CTBP，4=DP54，5=Rosenbrock，6=Forest-Ruth辛，7=RHS直接评估 */
    RK45Config rk45_config;        /**< RK45自适应求解器配置 */
    CTBPConfig ctbp_config;        /**< CTBP连续时间反向传播配置 */
    DP54Config dp54_config;        /**< DP54自适应步长求解器配置 */
    RosenbrockConfig rosenbrock_config; /**< Rosenbrock刚性求解器配置 */
    SymplecticConfig symplectic_config; /**< Forest-Ruth辛积分器配置 */
    /* 拉普拉斯变换分析增强参数 */
    float feedback_strength;       /**< 反馈强度，用于稳定性分析 */
    float input_gain;              /**< 输入增益，用于频域分析 */
    float output_gain;             /**< 输出增益，用于频域分析 */
    /* 多时间尺度并行演化参数 */
    int use_multi_timescale;       /**< 是否启用多时间尺度并行演化（快速/慢速双通道） */
    float fast_tau_ratio;          /**< 快速通道时间常数比例，默认0.1（τ_fast = τ * ratio） */
    float slow_tau_ratio;          /**< 慢速通道时间常数比例，默认10.0（τ_slow = τ * ratio） */
/* 拉普拉斯频域调制参数。
     * laplace_stability_score: 由拉普拉斯分析器计算，范围[0,1]，
     *   0=系统不稳定(需要强阻尼) 1=系统稳定(正常时间常数)
     * laplace_stability_alpha: 调制强度系数，默认0.3，越大阻尼越强
     * use_laplace_modulation: 开关，0=禁用 1=启用拉普拉斯tau调制 */
    float laplace_stability_score;
    float laplace_stability_alpha;
    int use_laplace_modulation;
    /* 统一自适应步长选择配置 */
    int use_adaptive_step;                 /**< 是否启用统一自适应步长选择，默认0=禁用（使用固定delta_t） */
    AdaptiveStepConfig adaptive_step_cfg;  /**< 自适应步长配置 */
    /* 并行化ODE求解配置 */
    int use_parallel_solve;                /**< 是否启用并行化ODE求解，默认0=禁用 */
    ParallelODERHSConfig parallel_cfg;     /**< 并行求解配置 */
    /* 自动求解器选择（P3.4） */
    int use_auto_solver;                   /**< 是否启用自动求解器选择，默认0=禁用（使用固定的ode_solver_type） */
    /* 四元数CfC集成（P2.4） */
    int use_quaternion;                    /**< 是否启用四元数CfC模式，默认0=禁用。启用后hidden_size和input_size需为4的倍数 */
    /* P0-001修复: 权重初始化方法 */
    int use_xavier_init;                   /**< 是否使用Xavier/Glorot初始化（默认1=启用），基于fan_in和fan_out自适应缩放，显著提升深层网络收敛性能 */
    int use_kaiming_init;                  /**< 是否使用Kaiming/He初始化（默认0=禁用），适用于ReLU类激活函数。使用fan_in/2缩放 */
    float weight_init_scale;               /**< 自定义权重初始化缩放因子（默认0.0=自动计算），>0时覆盖Xavier/Kaiming计算 */
    /* P1-001: 层归一化（LayerNorm）配置 */
    int use_cell_layer_norm;               /**< 是否启用层归一化（默认1=启用），对隐藏状态进行逐样本归一化，消除内部协变量偏移，显著提升深层网络训练稳定性 */
    float layer_norm_epsilon;              /**< 层归一化epsilon（默认1e-5） */
    /* P2-002: 残差连接配置 */
    int use_residual;                      /**< 是否启用残差/跳跃连接（默认1=启用），h_new = CfC_update + α*h_prev，为深层网络提供梯度高速公路 */
    float residual_scale;                  /**< 残差缩放因子（默认0.3），控制跳跃连接的强度。0=无残差，1=恒等映射 */
} CfCCellConfig;

/**
 * @brief CfC单元句柄
 */
typedef struct CfCCell CfCCell;

#ifdef SELFLNN_CORE_INTERNAL

/**
 * @brief CfC状态结构体（内部实现）
 * 包含ODE求解器所需的所有中间状态缓冲区
 */
typedef struct CfCState {
    float* state;                  /**< 隐藏状态 */
    float* adapted_params;         /**< 自适应参数 */
    float* noise_buffer;           /**< 噪声缓冲区 */
    float* ode_rhs_temp;           /**< P2-003修复: ODE RHS求解器RK2中间态预分配缓冲区 */
    float* activation;             /**< 激活值 */
    float* gradient;               /**< 梯度 */
    float* input_buffer;           /**< 输入缓冲区 */
    float* workspace;              /**< 工作空间 */

    /* 时间跟踪 */
    float time;                    /**< 当前时间 */
    float adaptation_rate;         /**< 自适应率 */

/* 保存前向传播的output_gate用于CTBP反向传播
     * 原代码用 og_approx=0.5+0.5*tanh(h) 保守估计，存在系统性偏差。
     * 现在前向传播时存储真实的sigmoid(W*x+U*h+b)值。 */
    float* saved_output_gate;      /**< 保存前向传播的output_gate [hidden_size] */

    /* RK45求解器中间状态 */
    float* rk45_k1;                /**< RK45 k1 */
    float* rk45_k2;                /**< RK45 k2 */
    float* rk45_k3;                /**< RK45 k3 */
    float* rk45_k4;                /**< RK45 k4 */
    float* rk45_k5;                /**< RK45 k5 */
    float* rk45_k6;                /**< RK45 k6 */
    float* rk45_temp;              /**< RK45临时 */
    float* rk45_error;             /**< RK45误差估计 */

    /* CTBP (连续时间反向传播) */
    float* ctbp_adjoint;           /**< CTBP伴随状态 */
    float* ctbp_temp;              /**< CTBP临时 */
    float* ctbp_state_trajectory;  /**< CTBP状态轨迹 */
    int ctbp_trajectory_size;      /**< CTBP轨迹大小 */
    int ctbp_trajectory_capacity;  /**< CTBP轨迹容量 */

    /* DP54求解器中间状态 */
    float* dp54_k7;                /**< DP54第7阶 */
    float* dp54_workspace;         /**< DP54工作空间 */
    int dp54_current_steps;        /**< DP54当前步数 */

    /* Rosenbrock求解器中间状态 */
    float* rosenbrock_jacobian;    /**< Rosenbrock雅可比矩阵 */
    float* rosenbrock_k1;          /**< Rosenbrock k1 */
    float* rosenbrock_k2;          /**< Rosenbrock k2 */
    float* rosenbrock_k3;          /**< Rosenbrock k3 */
    float* rosenbrock_workspace;   /**< Rosenbrock工作空间 */
    int rosenbrock_current_steps;  /**< Rosenbrock当前步数 */

    /* Symplectic辛积分器中间状态 */
    float* symplectic_momentum;    /**< 辛动量 */
    float* symplectic_dqdt;        /**< 辛dq/dt */
    float* symplectic_dpdt;        /**< 辛dp/dt */
    int symplectic_initialized;    /**< 辛积分器是否已初始化 */
    int symplectic_current_steps;   /**< 辛求解器当前步数 */
    float* saved_state;             /**< 前向状态快照 (ODR fix) */
} CfCState;

struct CfCCell {
    CfCCellConfig config;
    CfCState* state;
    float* weight_matrix;
    float* bias_vector;
    float* weight_grad;
    float* bias_grad;
    int is_initialized;
    int is_trained;             /**< 训练状态 — cfc_cell.c引用 */
    float avg_activation;
    float max_activation;
    float* time_constants;
    float* input_gate_weights;
    float* forget_gate_weights;
    float* output_gate_weights;
    float* gate_biases;
    int use_gating;
    int use_adaptive_tau;
    float min_time_constant;
    float max_time_constant;
    float* forward_tau_used;    /**< P1-019修复: 前向传播实际使用的tau值 [hidden_size]，反向传播直接读取避免不一致 */
    float* input_gate_weight_grad;
    float* output_gate_weight_grad;
    float* forget_gate_weight_grad;
    float* gate_bias_grad;
    float* input_gate_grads;        /**< 输入门激活梯度 [hidden_size] */
    float* forget_gate_grads;       /**< 遗忘门激活梯度 [hidden_size] */
    float* output_gate_grads;       /**< 输出门激活梯度 [hidden_size] */
    float* time_constant_grad;
    float tau_learning_rate;
    float* hidden_to_input_gate_weights;      /**< W_ghi: 隐藏到输入门权重矩阵 [hidden_size×hidden_size] */
    float* hidden_to_forget_gate_weights;    /**< W_ghf: 隐藏到遗忘门权重矩阵 [hidden_size×hidden_size] */
    float* hidden_to_output_gate_weights;    /**< W_gho: 隐藏到输出门权重矩阵 [hidden_size×hidden_size] */
    float* hidden_to_activation_weights;
    float* hidden_to_input_gate_weight_grad;  /**< W_ghi梯度缓冲区 */
    float* hidden_to_forget_gate_weight_grad; /**< W_ghf梯度缓冲区 */
    float* hidden_to_output_gate_weight_grad; /**< W_gho梯度缓冲区 */
    float* hidden_to_activation_weight_grad;
    int use_multi_timescale;
    float* fast_time_constants;
    float* slow_time_constants;
    float* fast_state;
    float* slow_state;
    float* fast_activation;
    float* slow_activation;
    float fast_tau_ratio;
    float slow_tau_ratio;
    float* multi_timescale_workspace;
    int use_adaptive_step;
    AdaptiveStepSolution adaptive_step_sol;
    int use_parallel_solve;
    int use_liquid_scaling;
    float* liquid_tau_weights;
    float* liquid_tau_bias;
    float* liquid_tau_weight_grad;
    float* liquid_tau_bias_grad;
    float* computed_liquid_tau;
    float* liquid_tau_workspace;
    float liquid_tau_min;
    float liquid_tau_max;
    float liquid_init_scale;
    int liquid_use_layer_norm;
    int use_neural_ode_adjoint;
    int adjoint_record_trajectory;
    int adjoint_max_trajectory_points;
    float adjoint_gradient_clip_norm;
    int adjoint_use_augmented_state;
    int adjoint_interpolation_method;
    float* adjoint_state;
    float* adjoint_trajectory;
    float* adjoint_timestamps;
    int adjoint_trajectory_capacity;
    int adjoint_trajectory_count;
    float* adjoint_workspace;
    int adjoint_forward_step;
    float* adjoint_param_grad_workspace;
    GatedCfCData* gated_data;
    HierarchicalCfCData* hierarchical_data;
    LiquidMemoryCfCData* liquid_memory_data;
    /* 分层CfC门控权重矩阵（可学习参数，替代硬编码0.1f） */
    float* hierarchical_gate_weights;         /**< 分层CfC门控权重矩阵 [hidden_size × input_size] */
    float* hierarchical_activation_weights;   /**< 分层CfC激活融合权重 [hidden_size × input_size] */
    float* hierarchical_gate_weight_grad;     /**< 分层CfC门控权重梯度 */
    float* hierarchical_activation_weight_grad; /**< 分层CfC激活权重梯度 */
    /* P0-BPTT: 统一参数更新 — cell级动量缓冲区 */
    float* cell_momentum_buffer;     /**< 所有cell级参数的动量缓冲区（一维平坦） */
    float* cell_velocity_buffer;     /**< 所有cell级参数的速度缓冲区（Adam二阶矩） */
    size_t cell_momentum_size;       /**< 动量缓冲区总大小（元素数） */
    int cell_momentum_initialized;   /**< 动量缓冲区是否已初始化 */
/* 拉普拉斯频域tau调制（运行时字段）
     * LNN层周期性将network_state的laplace_stability_score复制到这些字段，
     * cfc_closed_form_solution据此动态调整有效时间常数。
     * use_laplace_modulation和laplace_stability_alpha来自CfCCellConfig。 */
    float laplace_stability_score;
    void* liquid_scaling_mutex;      /**< 液时域缩放递归防护互斥锁 */
    /* ODR fix: cfc_cell.c本地字段合并到header */
    int use_quaternion;
    float* quaternion_weights;
    float* quaternion_biases;
    float* quaternion_weight_grad;
    float* quaternion_bias_grad;
    float* quaternion_hidden_weights;
    float* quaternion_hidden_weight_grad;
    float* quaternion_time_constants;
    float* quaternion_workspace;
    void* cell_layer_norm;           /* LayerNorm* */
    void* enhanced_state;            /* CfcEnhancedState* */
    void* enhanced_config;           /* CfcEnhancedConfig* (曾为值类型) */
    uint32_t enhanced_config_type_tag; /* D-011修复: 类型标签 0x43464345="CFCE" 用于void*安全检查 */
};
#endif

/**
 * @brief 创建CfC单元实例
 * 
 * @param config 单元配置
 * @return CfCCell* 单元句柄，失败返回NULL
 */
CfCCell* cfc_cell_create(const CfCCellConfig* config);

/**
 * @brief 释放CfC单元实例
 * 
 * @param cell 单元句柄
 */
void cfc_cell_free(CfCCell* cell);

/**
 * @brief 前向传播
 * 
 * @param cell 单元句柄
 * @param input 输入向量
 * @param hidden_state 隐藏状态缓冲区
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_forward(CfCCell* cell, const float* input, float* hidden_state);

/**
 * @brief 前向传播（带时间步长）
 * 
 * @param cell 单元句柄
 * @param input 输入向量
 * @param delta_t 时间步长（秒），覆盖配置中的delta_t
 * @param hidden_state 隐藏状态缓冲区
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_forward_with_dt(CfCCell* cell, const float* input, float delta_t, float* hidden_state);

/**
 * @brief 反向传播（训练）
 * 
 * @param cell 单元句柄
 * @param gradient 梯度向量
 * @param input_gradient 输入梯度输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_backward(CfCCell* cell, const float* gradient, float* input_gradient);

/**
 * @brief 前向+反向传播自测 — 全程在cfc_cell.c同一TU内执行
 *
 * 创建临时CfC单元 → forward → zero grad → backward → 验证梯度非零 → free
 * 绕过GCC 15.1/MinGW跨编译单元内存访问崩溃。
 *
 * @return int 0=成功，-1=失败
 */
int cfc_cell_backward_self_test(void);

/**
 * @brief 清零所有梯度缓冲区 — 在反向传播前调用
 *
 * 在 cfc_cell.c (梯度缓冲区所属编译单元) 内清零，
 * 避免 GCC 15.1/MinGW 跨编译单元 memset 崩溃。
 *
 * @param cell CfC单元
 */
void cfc_cell_zero_gradients(CfCCell* cell);

/**
 * @brief 时间梯度反向传播 — 计算梯度通过CfC状态转移矩阵的回传
 * 
 * 对给定组合梯度combined_gradient (已包含当前时间步的损失梯度和
 * 来自未来的时间梯度)，计算通过状态转移 h_new→h_old 传播回上一步的梯度。
 * 
 * 同时累积时间链对 W_gh、W_ah、gate_bias 的额外梯度贡献。
 * 
 * @param cell CfC单元句柄（需已执行前向传播保存saved_state）
 * @param combined_gradient 组合梯度向量 [hidden_size]
 * @param temporal_gradient_out 输出时间梯度 [hidden_size]（传回上一步的梯度）
 * @return 0成功，负值失败
 */
int cfc_cell_temporal_backward(CfCCell* cell, const float* combined_gradient,
                                float* temporal_gradient_out);

/**
 * @brief 截断时间反向传播 (Truncated BPTT)
 * 
 * 在给定时间窗口上执行完整的BPTT。
 * 
 * 调用方需在调用前执行完整的前向传播序列，保存每步的快照:
 *   seq_inputs[t]      = x(t)
 *   seq_prev_states[t] = cfc_cell_forward 调用后 cell->state->saved_state 的快照
 *   seq_forward_tau[t] = cell->forward_tau_used 的快照（可为NULL自动推算）
 * 
 * @param cell CfC单元句柄
 * @param loss_gradients 每时间步的损失梯度 [seq_len * hidden_size]
 * @param input_gradients 输出每步输入梯度 [seq_len * input_size]
 * @param seq_len 序列长度
 * @param seq_inputs 每步输入 [seq_len * input_size]
 * @param seq_prev_states 每步前向状态快照 [seq_len * hidden_size]
 * @param seq_forward_tau 每步 τ 快照 [seq_len * hidden_size]，NULL从time_constants推算
 * @return 0成功，负值失败
 */
int cfc_cell_backward_bptt(CfCCell* cell, const float* loss_gradients,
                            float* input_gradients, int seq_len,
                            const float* seq_inputs,
                            const float* seq_prev_states,
                            const float* seq_forward_tau);

/**
 * @brief 重置CfC单元状态
 * 
 * @param cell 单元句柄
 */
void cfc_cell_reset(CfCCell* cell);

/**
 * @brief 获取CfC单元配置
 * 
 * @param cell 单元句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_get_config(const CfCCell* cell, CfCCellConfig* config);

/**
 * @brief 设置CfC单元配置
 * 
 * @param cell 单元句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_set_config(CfCCell* cell, const CfCCellConfig* config);

/**
 * @brief 设置ODE求解器类型
 * 
 * @param cell 单元句柄
 * @param solver_type 求解器类型（0=闭式解，1=RK4，2=RK45，3=CTBP，4=DP54，5=Rosenbrock，6=Forest-Ruth辛）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_set_solver_type(CfCCell* cell, int solver_type);

/**
 * @brief 获取当前ODE求解器类型
 * 
 * @param cell 单元句柄
 * @return int 当前求解器类型，失败返回-1
 */
int cfc_cell_get_solver_type(const CfCCell* cell);

/**
 * @brief 使用RK45自适应步长求解器执行前向传播
 * 
 * RK45（龙格-库塔-费尔伯格法）是一种具有误差估计的自适应步长ODE求解器。
 * 它同时计算四阶和五阶近似，利用两者之差估计局部截断误差，
 * 并根据误差自动调整步长，在保证精度的同时提高计算效率。
 * 
 * @param cell CfC单元句柄
 * @param input 输入向量
 * @param delta_t 总时间步长
 * @param hidden_state 隐藏状态输出缓冲区
 * @param steps_used 实际使用的步数（输出参数，可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_forward_rk45(CfCCell* cell, const float* input, float delta_t, float* hidden_state, int* steps_used);

/**
 * @brief 使用Dormand-Prince 5(4)自适应步长求解器执行前向传播
 *
 * DP54是Dormand-Prince 5(4)对，使用7级Butcher表（7次函数求值）。
 * 与RK45（Fehlberg公式）相比，DP54具有：
 *   - 更优的精度效率比（同样误差下步长更大）
 *   - 更好的稳定性区域（在纯虚数轴上更稳定）
 *   - 步长控制采用数字PID控制器（更平滑的步长变化）
 *
 * 本实现采用标准的DP5(4)7M系数，5阶解用于推进，4阶解用于误差估计，
 * 步长控制使用H211b数字PI控制器。
 *
 * @param cell CfC单元句柄
 * @param input 输入向量
 * @param delta_t 总时间步长
 * @param hidden_state 隐藏状态输出缓冲区
 * @param steps_used 实际使用的步数（输出参数，可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_forward_dp54(CfCCell* cell, const float* input, float delta_t, float* hidden_state, int* steps_used);

/**
 * @brief 使用Rosenbrock刚性求解器执行前向传播
 *
 * Rosenbrock方法是线性隐式RK方法的变体，专门用于刚性ODE系统。
 * 本实现采用ROS3P（3阶L-稳定方案），具有以下特性：
 *   - L-稳定性：确保刚性特征值被数值阻尼快速衰减
 *   - 3阶精度：在非刚性分量上保持良好精度
 *   - 每步只需求解一个线性系统（比全隐式RK更高效）
 *
 * 刚性系统的Jacobian矩阵通过有限差分逼近计算：
 *   J_ij = (f_i(y + ε_j) - f_i(y)) / ε_j
 * 其中ε_j = max(1, |y_j|) * √(machine_epsilon)
 *
 * @param cell CfC单元句柄
 * @param input 输入向量
 * @param delta_t 总时间步长
 * @param hidden_state 隐藏状态输出缓冲区
 * @param steps_used 实际使用的步数（输出参数，可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_forward_rosenbrock(CfCCell* cell, const float* input, float delta_t, float* hidden_state, int* steps_used);

/**
 * @brief 使用Forest-Ruth辛积分器执行前向传播
 *
 * Forest-Ruth是4阶辛积分器，通过组合辛欧拉步(p←p+c*h*F(q), q←q+d*h*G(p))
 * 实现哈密顿系统的保结构积分。适用于需要长时间能量守恒的场景。
 *
 * 系数选择确保整体映射为4阶精度且严格辛：
 *   θ = 1/(2 - 2^(1/3))
 *   c = [θ/2, (1-θ)/2, (1-θ)/2, θ/2]
 *   d = [θ, (1-θ), 0, (1-θ)]
 *
 * @param cell CfC单元句柄
 * @param input 输入向量
 * @param delta_t 总时间步长
 * @param hidden_state 隐藏状态输出缓冲区
 * @param steps_used 实际使用的步数（输出参数，可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_forward_symplectic(CfCCell* cell, const float* input, float delta_t, float* hidden_state, int* steps_used);

/**
 * @brief 使用CTBP连续时间反向传播进行精确梯度计算
 * 
 * CTBP通过求解伴随方程（Adjoint Equation）实现精确的连续时间梯度计算，
 * 避免了离散化带来的梯度误差。支持直接法和伴随法两种模式。
 * 
 * @param cell CfC单元句柄
 * @param input 输入向量（前向传播时的输入）
 * @param output_gradient 输出梯度（从损失函数传来的梯度）
 * @param input_gradient 输入梯度输出缓冲区
 * @param time_span 时间跨度（前向传播的总时间）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_backward_ctbp(CfCCell* cell, const float* input, const float* output_gradient, float* input_gradient, float time_span);

/**
 * @brief 配置RK45求解器参数
 * 
 * @param cell CfC单元句柄
 * @param config RK45配置
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_configure_rk45(CfCCell* cell, const RK45Config* config);

/**
 * @brief 配置CTBP求解器参数
 * 
 * @param cell CfC单元句柄
 * @param config CTBP配置
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_configure_ctbp(CfCCell* cell, const CTBPConfig* config);

/**
 * @brief 配置DP54求解器参数
 * 
 * @param cell CfC单元句柄
 * @param config DP54配置
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_configure_dp54(CfCCell* cell, const DP54Config* config);

/**
 * @brief 配置Rosenbrock刚性求解器参数
 * 
 * @param cell CfC单元句柄
 * @param config Rosenbrock配置
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_configure_rosenbrock(CfCCell* cell, const RosenbrockConfig* config);

/**
 * @brief 配置Forest-Ruth辛积分器参数
 * 
 * @param cell CfC单元句柄
 * @param config 辛积分器配置
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_configure_symplectic(CfCCell* cell, const SymplecticConfig* config);

/**
 * @brief 获取CfC单元状态信息
 * 
 * @param cell 单元句柄
 * @param avg_activation 平均激活度输出缓冲区
 * @param max_activation 最大激活度输出缓冲区
 * @param adaptation_rate 自适应率输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_get_stats(const CfCCell* cell, float* avg_activation,
                       float* max_activation, float* adaptation_rate);

/**
 * @brief 启用或禁用多时间尺度并行演化
 *
 * 启用后，CfC单元将并行维护快速（小τ）和慢速（大τ）两个时间尺度的状态演化，
 * 输出为两者的加权混合。这允许同一网络同时处理短时和长时依赖关系。
 *
 * @param cell CfC单元句柄
 * @param enable 非零启用，零禁用
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_enable_multi_timescale(CfCCell* cell, int enable);

/**
 * @brief 设置多时间尺度的快慢时间常数比例
 *
 * @param cell CfC单元句柄
 * @param fast_ratio 快速通道时间常数比例（τ_fast = base_tau * ratio），推荐0.05~0.2
 * @param slow_ratio 慢速通道时间常数比例（τ_slow = base_tau * ratio），推荐5.0~20.0
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_set_tau_ratios(CfCCell* cell, float fast_ratio, float slow_ratio);

/* ============ 液时域缩放（Liquid Time Scaling）API ============ */

/**
 * @brief 液时域缩放配置
 *
 * 液时域缩放是液态神经网络的核心特性。与固定时间常数的传统RNN不同，
 * 液时域缩放使每个神经元的时间常数成为输入的函数：
 * τ_i(x) = sigmoid(W_tau[i,:] · x + b_tau[i]) · (τ_max - τ_min) + τ_min
 *
 * 这使得网络能根据输入信号自动调整时间动态特性，
 * 需要快速响应时加快动态，需要长期记忆时减慢动态。
 */
typedef struct {
    int enable_liquid_scaling;    /**< 是否启用液时域缩放，1=启用，0=禁用 */
    float tau_min;                /**< 时间常数最小值，默认0.01f */
    float tau_max;                /**< 时间常数最大值，默认2.0f */
    float init_scale;             /**< 输入投影权重初始化缩放因子，默认0.01f */
    int use_layer_norm;           /**< 是否对时间常数投影进行层归一化，默认0 */
} LiquidTimeScalingConfig;

/**
 * @brief 启用液时域缩放并设置配置
 *
 * 启用后，CfC单元将为每个神经元计算输入依赖的时间常数。
 * 需要额外内存：hidden_size × (input_size + 1) 个float（权重 + 偏置）
 *
 * @param cell CfC单元句柄
 * @param config 液时域缩放配置，为NULL时使用默认配置
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_enable_liquid_scaling(CfCCell* cell, const LiquidTimeScalingConfig* config);

/**
 * @brief 使用液时域缩放执行前向传播
 *
 * 在每个时间步，首先根据当前输入计算每个神经元的液时域缩放时间常数：
 *   liquid_tau = sigmoid(W_tau · x + b_tau) · (tau_max - tau_min) + tau_min
 * 然后使用这些输入依赖的时间常数执行CfC闭式解前向传播。
 *
 * @param cell CfC单元句柄
 * @param input 输入向量
 * @param hidden_state 隐藏状态输出缓冲区
 * @param computed_tau 计算得到的液时域时间常数（输出，可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_forward_liquid(CfCCell* cell, const float* input, float* hidden_state, float* computed_tau);

/**
 * @brief 获取当前液时域缩放的时间常数
 *
 * 返回最近一次 forward_liquid 调用计算出的时间常数。
 * 可用于可视化或调试。
 *
 * @param cell CfC单元句柄
 * @param tau_out 时间常数输出缓冲区（至少 hidden_size 个float）
 * @param size 输出参数：实际神经元数量
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_get_liquid_tau(const CfCCell* cell, float* tau_out, int* size);

/**
 * @brief 直接设置液时域缩放的时间常数（覆盖计算值）
 *
 * 允许外部直接指定每个神经元的时间常数，覆盖输入依赖的计算结果。
 * 可用于手动调控或注入先验知识。
 *
 * @param cell CfC单元句柄
 * @param tau_values 新的时间常数数组（至少 hidden_size 个float）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_set_liquid_tau(CfCCell* cell, const float* tau_values);

/**
 * @brief 液时域缩放的梯度反向传播
 *
 * 计算损失对液时域缩放参数（W_tau, b_tau）的梯度，
 * 用于端到端训练。
 *
 * @param cell CfC单元句柄
 * @param gradient 输出梯度（从上层传来的梯度）
 * @param input_gradient 输入梯度输出缓冲区（可为NULL）
 * @param tau_gradient 液时域时间常数梯度输出缓冲区（可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_backward_liquid(CfCCell* cell, const float* gradient, float* input_gradient, float* tau_gradient);

/* ============ 神经ODE伴随法（Neural ODE Adjoint）API ============ */

/**
 * @brief 神经ODE伴随法配置
 *
 * 神经ODE伴随法（Neural ODE Adjoint）是一种通过求解伴随方程（Adjoint Equation）
 * 来计算连续时间梯度的方法。与标准的反向传播不同，它通过逆时间积分一个伴随ODE
 * 来计算梯度，避免了存储完整前向计算图。
 *
 * 核心思想：
 *   1. 前向传播时求解ODE：dh/dt = f(h, t, θ)，并记录状态轨迹
 *   2. 反向传播时求解伴随ODE：da/dt = -a(t)ᵀ · ∂f/∂h
 *   3. a(t₁) = ∂L/∂h(t₁) 作为伴随方程的初始条件
 *   4. 参数梯度：dL/dθ = ∫ a(t)ᵀ · ∂f/∂θ dt
 *
 * 优势：
 *   - O(1)级别的内存复杂度（相对于深度网络的O(N)）
 *   - 精确的连续时间梯度（不依赖于离散化近似）
 *   - 可与任意ODE求解器组合使用
 */
typedef struct {
    int enable_adjoint;              /**< 是否启用伴随法，1=启用，0=禁用 */
    int record_trajectory;           /**< 是否记录完整轨迹，1=记录（默认），0=不记录 */
    int max_trajectory_points;       /**< 最大轨迹记录点数，默认10000 */
    float gradient_clip_norm;        /**< 梯度裁剪范数，0=不裁剪，默认0 */
    int use_augmented_state;         /**< 是否使用增广状态（同时计算参数梯度），默认0 */
    int interpolation_method;        /**< 轨迹插值方法：0=最近邻，1=线性（默认） */
} NeuralODEConfig;

/**
 * @brief 启用神经ODE伴随法并设置配置
 *
 * 启用后，前向传播将自动记录状态轨迹，反向传播使用伴随法计算精确梯度。
 *
 * @param cell CfC单元句柄
 * @param config 神经ODE伴随法配置，为NULL时使用默认配置
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_enable_neural_ode_adjoint(CfCCell* cell, const NeuralODEConfig* config);

/**
 * @brief 使用轨迹记录执行前向传播（神经ODE伴随法用）
 *
 * 执行标准前向传播的同时，记录隐藏状态的完整演化轨迹。
 * 轨迹用于反向传播时的伴随ODE求解。
 *
 * @param cell CfC单元句柄
 * @param input 输入向量
 * @param hidden_state 隐藏状态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_forward_with_trajectory(CfCCell* cell, const float* input, float* hidden_state);

/**
 * @brief 使用神经ODE伴随法执行反向传播
 *
 * 通过逆时间求解伴随方程计算精确梯度：
 *   1. 从终点状态开始：a(t₁) = gradient
 *   2. 逆时间积分伴随ODE：da/dt = -a(t)ᵀ · ∂f/∂h
 *   3. 积分参数梯度：dL/dθ = ∫ a(t)ᵀ · ∂f/∂θ dt
 *
 * @param cell CfC单元句柄
 * @param output_gradient 输出梯度（从损失函数传来的梯度）
 * @param input_gradient 输入梯度输出缓冲区（可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_backward_adjoint(CfCCell* cell, const float* output_gradient, float* input_gradient);

/**
 * @brief 获取当前伴随状态
 *
 * 返回最近一次 backward_adjoint 调用中计算出的伴随状态。
 * 伴随状态编码了损失对隐藏状态在连续时间内的梯度信息。
 *
 * @param cell CfC单元句柄
 * @param adjoint_out 伴随状态输出缓冲区（至少 hidden_size 个float）
 * @param size 输出参数：实际神经元数量
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_get_adjoint_state(const CfCCell* cell, float* adjoint_out, int* size);

/**
 * @brief 保存CfC单元参数到文件
 * 
 * @param cell 单元句柄
 * @param file 已打开的可写文件指针
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_save(const CfCCell* cell, FILE* file);

/**
 * @brief 从文件加载CfC单元参数
 * 
 * @param cell 单元句柄（必须已通过cfc_cell_create创建）
 * @param file 已打开的只读文件指针
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_load(CfCCell* cell, FILE* file);

/* ============ 门控CfC变体（Gated CfC）API ============ */

/**
 * @brief 门控CfC变体配置
 *
 * 增强CfC单元的门控机制，引入交叉门控、金字塔门控和自适应带宽。
 * 交叉门控（Cross-gating）：不同神经元之间的门控信号互相调制
 * 金字塔门控（Pyramidal gating）：多层级联门控结构实现精细控制
 * 自适应带宽（Adaptive bandwidth）：门控信号的带宽根据输入动态调整
 */
typedef struct {
    int enable_cross_gating;          /**< 是否启用交叉门控，1=启用，0=禁用 */
    int enable_pyramidal_gating;      /**< 是否启用金字塔门控，1=启用，0=禁用 */
    int enable_adaptive_bandwidth;    /**< 是否启用自适应带宽，1=启用，0=禁用 */
    float cross_gate_init_scale;      /**< 交叉门控权重初始化缩放，默认0.01 */
    float pyramidal_gate_init_scale;  /**< 金字塔门控权重初始化缩放，默认0.01 */
    float bandwidth_min;              /**< 门控带宽最小值，默认0.1 */
    float bandwidth_max;              /**< 门控带宽最大值，默认10.0 */
    int pyramidal_layers;             /**< 金字塔门控层数，默认3 */
} GatedCfCConfig;

/**
 * @brief 启用门控CfC变体并设置配置
 *
 * 启用交叉门控后，每个神经元的门控值受其他神经元的影响：
 *   gate_i = σ(W_gx*x + W_gh*h + Σ_j W_cross[i,j]*gate_j + b_g)
 * 形成递归门控结构，增强了门控的表达能力。
 *
 * 启用金字塔门控后，门控信号通过多层级联精化：
 *   gate^{(l+1)} = σ(W_pyr^{(l)} * gate^{(l)} + b_pyr^{(l)})
 * 实现从粗到细的门控控制。
 *
 * @param cell CfC单元句柄
 * @param config 门控CfC配置，为NULL时使用默认配置
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_enable_gated_cfc(CfCCell* cell, const GatedCfCConfig* config);

/**
 * @brief 使用门控CfC变体执行前向传播
 *
 * 在标准CfC前向传播的基础上，应用增强的门控机制。
 * 交叉门控使门控信号在神经元之间传递，金字塔门控逐层精化门控信号。
 *
 * @param cell CfC单元句柄
 * @param input 输入向量
 * @param hidden_state 隐藏状态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_forward_gated(CfCCell* cell, const float* input, float* hidden_state);

/**
 * @brief 门控CfC变体的梯度反向传播
 *
 * 计算损失对门控CfC参数（交叉门控权重、金字塔门控权重、自适应带宽参数）的梯度。
 *
 * @param cell CfC单元句柄
 * @param gradient 输出梯度
 * @param input_gradient 输入梯度输出缓冲区（可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_backward_gated(CfCCell* cell, const float* gradient, float* input_gradient);

/**
 * @brief 获取门控CfC的状态信息
 *
 * @param cell CfC单元句柄
 * @param cross_gate_values 交叉门控值输出缓冲区（可为NULL）
 * @param pyramidal_gate_values 金字塔门控值输出缓冲区（可为NULL）
 * @param bandwidth_values 自适应带宽值输出缓冲区（可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_get_gated_stats(const CfCCell* cell, float* cross_gate_values,
                             float* pyramidal_gate_values, float* bandwidth_values);

/* ============ 分层CfC（Hierarchical CfC）API ============ */

/**
 * @brief 分层CfC配置
 *
 * 分层CfC将CfC单元组织为多级时间层次结构。
 * 底层：快动态（毫秒级），中层：中动态（秒级），顶层：慢动态（分钟级）。
 * 支持双向信息流：自底向上（快速→慢速）和自顶向下（慢速→快速）。
 */
typedef struct {
    int num_levels;                   /**< 层次数量，默认3（快/中/慢） */
    float fast_time_constant;         /**< 底层（最快层）时间常数，默认0.01 */
    float slow_time_constant;         /**< 顶层（最慢层）时间常数，默认1.0 */
    float bottom_up_strength;         /**< 自底向上信息流强度，默认0.5 */
    float top_down_strength;          /**< 自顶向下信息流强度，默认0.3 */
    int enable_bidirectional;         /**< 是否启用双向信息流，1=启用，0=仅自底向上 */
    float level_init_scale;           /**< 层间连接权重初始化缩放，默认0.01 */
} HierarchicalCfCConfig;

/**
 * @brief 启用分层CfC并设置配置
 *
 * 启用后，CfC单元内部维护多个不同时间尺度的时间常数。
 * 各层次使用标准CfC闭式解独立演化，层间通过双向连接交换信息。
 * 底层对快速变化敏感，顶层捕捉长期依赖。
 *
 * @param cell CfC单元句柄
 * @param config 分层CfC配置，为NULL时使用默认配置
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_enable_hierarchical(CfCCell* cell, const HierarchicalCfCConfig* config);

/**
 * @brief 使用分层CfC执行前向传播
 *
 * 每个层次独立演化：
 *   对于第l层：dh_l/dt = -h_l/τ_l + f_l(h_l, x, h_{l-1}, h_{l+1})
 * 层次间信息流通过加权连接实现。
 * 最终输出为所有层次的加权融合。
 *
 * @param cell CfC单元句柄
 * @param input 输入向量
 * @param hidden_state 合并后的隐藏状态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_forward_hierarchical(CfCCell* cell, const float* input, float* hidden_state);

/**
 * @brief 分层CfC的梯度反向传播
 *
 * 跨层次传播梯度，包括自底向上和自顶向下的梯度路径。
 *
 * @param cell CfC单元句柄
 * @param gradient 输出梯度
 * @param input_gradient 输入梯度输出缓冲区（可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_backward_hierarchical(CfCCell* cell, const float* gradient, float* input_gradient);

/**
 * @brief 获取分层CfC指定层次的状态
 *
 * @param cell CfC单元句柄
 * @param level 层次索引（0=最快层，num_levels-1=最慢层）
 * @param state_out 状态输出缓冲区（hidden_size个float）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_get_hierarchical_state(const CfCCell* cell, int level, float* state_out);

/* ============ 液态记忆门控CfC（Liquid Memory Gating CfC）API ============ */

/**
 * @brief 液态记忆门控CfC配置
 *
 * 使用CfC自身门控机制替代注意力机制，实现隐藏状态的自适应记忆调制。
 * 液态记忆门控通过CfC的闭式解门控方程计算调制信号：
 *   M = σ(W_gate · [h, x]) ⊙ tanh(W_act · [h, x])
 * 调制信号直接注入CfC ODE驱动项，无外部注意力机制。
 */
typedef struct {
    int enable_memory_gating;         /**< 是否启用液态记忆门控，1=启用，0=禁用 */
    float init_scale;                 /**< 权重初始化缩放，默认0.1 */
    float gate_bias_init;             /**< 门控偏置初始化值，默认2.0（初始偏向开启） */
    float gate_scale;                 /**< 门控缩放因子，默认1.0 */
} LiquidMemoryCfCConfig;

/**
 * @brief 启用液态记忆门控CfC并设置配置
 *
 * 分配并初始化液态记忆门控内部数据：
 *   门控权重 W_gate ∈ R^{hidden_size × (hidden_size + input_size)}
 *   激活权重 W_act ∈ R^{hidden_size × (hidden_size + input_size)}
 *   门控偏置 b_gate ∈ R^{hidden_size}
 *   激活偏置 b_act ∈ R^{hidden_size}
 *
 * 前向计算：
 *   concat = [hidden_state, input]    // 拼接
 *   gate = σ(W_gate · concat + b_gate)
 *   act = tanh(W_act · concat + b_act)
 *   modulation = gate ⊙ act
 *   drive_adjusted = σ(gate_orig) ⊙ tanh(act_orig + modulation)
 *
 * @param cell CfC单元句柄
 * @param config 液态记忆门控配置，为NULL时使用默认配置
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_enable_liquid_memory(CfCCell* cell, const LiquidMemoryCfCConfig* config);

/**
 * @brief 使用液态记忆门控CfC执行前向传播
 *
 * 在标准CfC前向传播的基础上，集成液态记忆门控调制：
 *   1. 拼接隐藏状态和输入向量
 *   2. 计算门控信号 gate = σ(W_gate · concat + b_gate)
 *   3. 计算激活信号 act = tanh(W_act · concat + b_act)
 *   4. 计算调制信号 modulation = gate ⊙ act
 *   5. 将调制信号注入CfC驱动项
 *   6. 执行标准CfC闭式解更新
 *
 * @param cell CfC单元句柄
 * @param input 输入向量
 * @param hidden_state 隐藏状态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_forward_liquid_memory(CfCCell* cell, const float* input, float* hidden_state);

/**
 * @brief 液态记忆门控CfC的梯度反向传播
 *
 * 通过液态记忆门控传播梯度，包括对门控/激活权重的梯度计算。
 *
 * @param cell CfC单元句柄
 * @param gradient 输出梯度
 * @param input_gradient 输入梯度输出缓冲区（可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_backward_liquid_memory(CfCCell* cell, const float* gradient, float* input_gradient);

/**
 * @brief 获取液态记忆门控CfC的当前记忆调制状态
 *
 * 返回最近一次forward_liquid_memory调用计算出的记忆调制信号。
 *
 * @param cell CfC单元句柄
 * @param modulation_out 记忆调制信号输出缓冲区（hidden_size个float）
 * @param gate_out 门控信号输出缓冲区（hidden_size个float，可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_get_liquid_memory_state(const CfCCell* cell, float* modulation_out, float* gate_out);

/* ============ D-004修复: 分层CfC与液态归一化扩展接口 ============ */

/**
 * @brief 分层CfC的分层前向传播（自底向上+自顶向下流）
 *
 * 支持多层级信息处理，每层CfC单元之间传递底部向上(bottom-up)和
 * 顶部向下(top-down)信号流，适用于分层抽象学习。
 *
 * @param cell CfC单元句柄
 * @param input 输入向量
 * @param output 输出隐藏状态缓冲区
 * @param bu_flows 底部向上流缓冲区（num_levels * hidden_size）
 * @param td_flows 顶部向下流缓冲区（num_levels * hidden_size）
 * @param num_levels 层级数量（必须 > 1）
 * @return 0=成功, -1=参数无效
 */
int cfc_cell_hierarchical_forward(CfCCell* cell, const float* input,
                                   float* output, float* bu_flows, float* td_flows,
                                   int num_levels);

/**
 * @brief 液态记忆门控应用
 *
 * 将记忆门控信号应用于隐藏状态，使用记忆衰减因子
 * 进行门控调制，实现长期记忆的逐步衰减。
 *
 * @param cell CfC单元句柄
 * @param hidden_pre 门控前的隐藏状态
 * @param input 输入向量
 * @param hidden_post 门控后的隐藏状态输出
 * @param memory_decay 记忆衰减因子 [0,1]
 * @return 0=成功, -1=参数无效
 */
int cfc_cell_memory_gate_apply(CfCCell* cell, const float* hidden_pre,
                                const float* input, float* hidden_post,
                                float memory_decay);

/**
 * @brief 液时域层归一化
 *
 * 对CfC时间常数投影进行LayerNorm归一化，确保
 * 时间常数在不同时间尺度下的数值稳定性。
 *
 * @param tau_projection 时间常数投影向量（原地修改）
 * @param hidden_size 向量维度
 * @return 0=成功, -1=参数无效
 */
int cfc_liquid_layernorm(float* tau_projection, size_t hidden_size);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_CFC_CELL_H