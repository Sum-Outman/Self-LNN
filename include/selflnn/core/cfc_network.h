/**
 * @file cfc_network.h
 * @brief CfC（闭合形式连续时间）液态神经网络接口
 * 
 * 由多个CfC细胞单元组成的完整液态神经网络，提供完整的前向传播和反向传播功能。
 * 实现文件：src/core/cfc_network.c（命名一致性：头文件 cfc_network.h ↔ 实现文件 cfc_network.c）
 */

#ifndef SELFLNN_CFC_NETWORK_H
#define SELFLNN_CFC_NETWORK_H

#include <stddef.h>
#include <stdio.h>
#include "selflnn/core/cfc_cell.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CfC网络配置
 */
typedef struct {
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
    float learning_rate;
    float time_constant;
    float noise_std;
    int enable_training;
    int enable_adaptation;
    int num_layers;
    float dropout_rate;
    int use_batch_norm;
    int use_layer_norm;          /**< P0-001修复: 层归一化开关，默认启用，消除内部协变量偏移 */
    int ode_solver_type;
} CfCNetworkConfig;

typedef struct CfCNetwork CfCNetwork;

CfCNetwork* cfc_create(const CfCNetworkConfig* config);
void cfc_free(CfCNetwork* network);
int cfc_forward(CfCNetwork* network, const float* input, 
                float* hidden_state, float* cell_state, float* output);
int cfc_backward(CfCNetwork* network, const float* error, 
                 float* gradient, float learning_rate);

/**
 * @brief CfC网络反向传播（扩展版——支持批量训练梯度累积模式）
 * 
 * P0-002修复: skip_cell_update=1时跳过Step3的cell参数直接更新，
 * 仅将梯度累积到cell内部缓冲区。调用方需在批量结束后调用
 * cfc_apply_cell_gradients() 统一下发参数更新。
 *
 * @param network CfC网络句柄
 * @param error 误差向量 [output_size]
 * @param gradient 输入梯度输出缓冲区 [input_size]
 * @param learning_rate 学习率
 * @param skip_cell_update 0=单样本直接更新, 1=仅累积梯度不更新cell参数
 * @return 0成功，负值失败
 */
int cfc_backward_ex(CfCNetwork* network, const float* error,
                    float* gradient, float learning_rate, int skip_cell_update);

int cfc_accumulate_gradients(CfCNetwork* network, const float* error, 
                            float* gradient,
                            float* weight_gradients, float* bias_gradients);
int cfc_save(const CfCNetwork* network, FILE* file);
int cfc_load(CfCNetwork* network, FILE* file);
int cfc_set_config(CfCNetwork* network, const CfCNetworkConfig* config);
int cfc_get_config(const CfCNetwork* network, CfCNetworkConfig* config);
void cfc_reset(CfCNetwork* network);
int cfc_get_stats(const CfCNetwork* network, float* avg_activation,
                  float* max_activation, float* gradient_norm);
int cfc_get_weight_matrix(CfCNetwork* network, float** weight_matrix, size_t* weight_count);
int cfc_get_bias_vector(CfCNetwork* network, float** bias_vector, size_t* bias_count);

/**
 * @brief 应用各层CfC单元的cell级参数梯度（门控权重、门控偏置、时间常数）
 * 
 * FIX-011: cfc_accumulate_gradients 只处理共享权重/偏置梯度到外部缓冲区，
 * cell级参数（W_gx/W_gh/W_ah/gate_bias/time_constants）的梯度仅存储在各cell内部。
 * 此函数将这些cell级梯度应用到对应参数上。
 *
 * @param network CfC网络
 * @param learning_rate 学习率（用于门控权重/偏置更新）
 * @return 0成功，-1失败
 */
int cfc_apply_cell_gradients(CfCNetwork* network, float learning_rate);

/**
 * @brief P0-BPTT: 使用Adam风格更新应用cell级参数梯度
 * 
 * 为所有cell级参数（W_gh, W_ah, W_gx, W_fx, W_ox, b_g, τ）使用
 * Adam自适应学习率更新，解决双层参数更新分裂问题。
 * 内部维护每参数的动量(m)和速度(v)缓冲区。
 * 
 * @param network CfC网络
 * @param learning_rate 基础学习率
 * @param beta1 一阶矩衰减率（默认0.9）
 * @param beta2 二阶矩衰减率（默认0.999）
 * @param epsilon 数值稳定常数（默认1e-8）
 * @param t 当前全局步数（用于偏差校正）
 * @return 0成功，负值失败
 */
int cfc_apply_cell_gradients_adam(CfCNetwork* network, float learning_rate,
                                   float beta1, float beta2, float epsilon, size_t t);

/**
 * @brief 应用输出投影矩阵(W_out+b_out)梯度
 * 
 * FIX-015: W_out 参数不在共享权重块(param_block)中，而是在grad_block尾部。
 * 外部优化器(lnn_get_parameters→weight_matrix)无法触及W_out。
 * 此函数在批量训练结束时将累积的W_out梯度应用至W_out参数。
 *
 * @param network CfC网络
 * @param learning_rate 学习率
 * @return 0成功，-1失败
 */
int cfc_apply_out_proj_gradients(CfCNetwork* network, float learning_rate);

/**
 * @brief P0-002深度修复: 清零所有cell级梯度缓冲区
 * 在batch开始/单样本反向传播前调用，确保+=累积从零开始。
 */
void cfc_zero_cell_gradients(CfCNetwork* network);

/**
 * @brief 将共享参数块同步到各层CfC单元的活跃权重
 * 
 * FIX-017: cell->weight_matrix/bias_vector 是 cfc_forward 使用的活跃权重，
 * 但 optimizer_update 更新的是共享 param_block（cfc_network->weight_matrix）。
 * 此函数在优化器更新后将共享块参数复制到各层的活跃 cell 权重中。
 *
 * @param network CfC网络
 * @return 0成功，-1失败
 */
int cfc_sync_shared_to_cells(CfCNetwork* network);

/* ============================================================================
 * 长时连续动态系统求解器
 *
 * 为液态神经网络提供完整的连续时间动力学支持：
 *   - 可变时间步长前向传播（cfc_forward_dt 已存在）
 *   - 长时轨迹积分（dt累积，记录中间状态）
 *   - 任意时刻状态插值（Hermite三次插值）
 *   - 网络级自适应步长 + 刚度检测 + 自动求解器切换
 *   - 连续动力学RHS暴露（用于外部ODE积分器）
 * ============================================================================ */

/**
 * @brief 连续时间轨迹记录
 * 存储长时演化过程中各时间点的网络状态快照
 */
typedef struct {
    float* state_buffer;        /**< 状态快照缓冲区 [capacity × (state_dim + 1)] */
    float* timestamps;          /**< 时间戳数组 [capacity] */
    int capacity;               /**< 最大快照数 */
    int count;                  /**< 当前快照数 */
    float t_start;              /**< 轨迹起始时间 */
    float t_end;                /**< 轨迹结束时间 */
    size_t state_dim;           /**< 每个快照的状态维度 */
} CfCTrajectory;

/**
 * @brief 长时连续演化配置
 */
typedef struct {
    float t_start;              /**< 起始时间 */
    float t_end;                /**< 结束时间 */
    float dt_init;              /**< 初始时间步长 */
    float dt_min;               /**< 最小允许步长 */
    float dt_max;               /**< 最大允许步长 */
    float rel_tol;              /**< 相对误差容限 */
    float abs_tol;              /**< 绝对误差容限 */
    int max_steps;              /**< 最大步数限制 */
    int trajectory_capacity;    /**< 轨迹缓冲区容量 */
    int enable_stiffness_detect; /**< 是否启用刚度检测自动切换求解器 */
    int enable_trajectory_output; /**< 是否输出完整轨迹 */
    float trajectory_sample_dt; /**< 轨迹采样间隔（0=每步都记录） */
} CfCContinuousConfig;

/**
 * @brief 连续动力学右端函数上下文
 * 将CfC网络的ODE系统暴露为可调用的RHS函数，
 * 兼容 ode_solvers.h 中定义的 ODERHSFunc 类型。
 */
typedef struct {
    CfCNetwork* network;        /**< 目标CfC网络 */
    const float* input;         /**< 固定输入（零阶保持） */
    float* temp_buffer1;        /**< 临时缓冲区1 [hidden_size] */
    float* temp_buffer2;        /**< 临时缓冲区2 [hidden_size] */
    int current_layer;          /**< 当前层索引 */
} CfCRHSContext;

/**
 * @brief 长时连续演化结果统计
 */
typedef struct {
    int total_steps;            /**< 总步数 */
    int accepted_steps;         /**< 接受步数 */
    int rejected_steps;         /**< 拒绝步数 */
    int solver_switches;        /**< 求解器切换次数 */
    float final_t;              /**< 最终时间 */
    float avg_dt;               /**< 平均步长 */
    float min_dt_used;          /**< 最窄步长 */
    float max_dt_used;          /**< 最宽步长 */
    float stiffness_ratio;      /**< 刚度比（max|Re(λ)| / min|Re(λ)|） */
    int used_stiff_solver;      /**< 是否使用了刚性求解器 */
} CfCContinuousStats;

/**
 * @brief 获取默认长时连续演化配置
 */
CfCContinuousConfig cfc_continuous_default_config(void);

/**
 * @brief 创建连续时间轨迹记录器
 * @param capacity 最大快照数
 * @param state_dim 每个快照的状态维度
 * @return 轨迹对象，失败返回NULL
 */
CfCTrajectory* cfc_trajectory_create(int capacity, size_t state_dim);

/**
 * @brief 销毁轨迹对象
 */
void cfc_trajectory_free(CfCTrajectory* traj);

/**
 * @brief 将状态快照追加到轨迹
 * @param traj 轨迹对象
 * @param t 时间戳
 * @param state 状态向量 [state_dim]
 * @return 0成功，-1轨迹满
 */
int cfc_trajectory_append(CfCTrajectory* traj, float t, const float* state);

/**
 * @brief 在轨迹中插值任意时刻的状态（Hermite三次插值）
 * @param traj 轨迹对象
 * @param t_query 查询时间
 * @param state_out 输出状态 [state_dim]
 * @return 0成功，-1失败（t超出范围或缺数据）
 */
int cfc_trajectory_interpolate(const CfCTrajectory* traj, float t_query, float* state_out);

/**
 * @brief 长时连续演化：从t_start到t_end积分网络动态
 *
 * 使用自适应步长ODE求解器对CfC网络进行长时连续积分。
 * 支持刚度检测自动切换求解器（显式→隐式/Rosenbrock）。
 * 可选输出完整状态轨迹。
 *
 * @param network CfC网络
 * @param input 固定输入向量（零阶保持，随时间不变）[input_size]
 * @param hidden_state 输入/输出隐藏状态 [hidden_size]
 * @param cell_state 输入/输出细胞状态 [hidden_size]
 * @param output 输出向量 [output_size]（可为NULL仅演化不发散）
 * @param config 连续演化配置
 * @param stats 输出统计信息（可为NULL）
 * @return 0成功，-1失败
 */
int cfc_continuous_evolve(CfCNetwork* network, const float* input,
                          float* hidden_state, float* cell_state,
                          float* output,
                          const CfCContinuousConfig* config,
                          CfCContinuousStats* stats);

/**
 * @brief 长时连续演化（带轨迹输出）
 * 与 cfc_continuous_evolve 相同，但将完整轨迹写入 traj。
 *
 * @param traj 轨迹输出对象（由调用者预创建）
 * @return 0成功，-1失败
 */
int cfc_continuous_evolve_with_trajectory(CfCNetwork* network, const float* input,
                                          float* hidden_state, float* cell_state,
                                          float* output,
                                          const CfCContinuousConfig* config,
                                          CfCTrajectory* traj,
                                          CfCContinuousStats* stats);

/**
 * @brief 构建CfC网络的连续ODE右端函数上下文
 *
 * 将网络动态表示为 dy/dt = f(t, y) 形式，
 * 其中 y = [hidden_state] 为状态向量。
 * 返回的ctx可直接传入 ode_solvers.h 中定义的ODE积分器。
 *
 * @param network CfC网络
 * @param input 固定输入 [input_size]
 * @return CfCRHSContext* 可为NULL（内存不足）
 */
CfCRHSContext* cfc_create_rhs_context(CfCNetwork* network, const float* input);

/**
 * @brief 销毁RHS上下文
 */
void cfc_free_rhs_context(CfCRHSContext* ctx);

/**
 * @brief CfC网络连续RHS求值：计算 dy/dt = f(t, y)
 *
 * 兼容 ODERHSFunc 签名：
 *   int (*)(float t, const float* y, float* dydt, void* ctx)
 *
 * 其中 ctx 为 CfCRHSContext*。
 *
 * @param t 当前时间
 * @param y 当前状态 [hidden_size]
 * @param dydt 输出导数 [hidden_size]
 * @param ctx CfCRHSContext*
 * @return 0成功，-1失败
 */
int cfc_continuous_rhs(float t, const float* y, float* dydt, void* ctx);

/**
 * @brief 网络级刚度检测：估算ODE系统的刚度比
 *
 * 使用有限差分逼近雅可比矩阵，计算最大/最小特征值实部之比。
 * 刚度比 > 10 建议切换刚性求解器，> 100 必须使用刚性求解器。
 *
 * @param network CfC网络
 * @param input 当前输入 [input_size]
 * @param state 当前状态 [hidden_size]
 * @param stiffness_ratio 输出刚度比
 * @return 0成功，-1失败
 */
int cfc_detect_stiffness(CfCNetwork* network, const float* input,
                         const float* state, float* stiffness_ratio);

/**
 * @brief 设置网络级ODE求解器类型（为所有层统一设置）
 * @param network CfC网络
 * @param solver_type 求解器类型（参见ODE_SOLVER_*常量）
 * @return 0成功，-1失败
 */
int cfc_set_solver_type(CfCNetwork* network, int solver_type);

/**
 * @brief 启用网络级自适应时间步长
 * @param network CfC网络
 * @param enable 1=启用, 0=禁用
 * @return 0成功，-1失败
 */
int cfc_set_adaptive_step(CfCNetwork* network, int enable);

#ifdef SELFLNN_IMPLEMENTATION
/* CfCCell 前向声明由 cfc_cell.h 提供，此处无需重复 */
struct CfCNetwork {
    CfCNetworkConfig config;
    CfCCell** layers;
    float* layer_outputs;
    float* layer_gradients;
    float* weight_matrix;
    float* bias_vector;
    float* weight_gradients;
    float* bias_gradients;
    float* activation_buffer;
    float* dropout_mask;
    int is_initialized;
    float current_avg_activation;
    float current_max_activation;
    float current_gradient_norm;
    /* P0-001: 每层级联层归一化，消除深层网络的内部协变量偏移 */
    void** layer_norms;           /**< LayerNorm* 数组 [num_layers]，每层一个LN */
    float* ln_temp_buffer;        /**< LN前向临時缓冲区 (max_layer_size) */
    /* P0-014修复: W_out/b_out参数与梯度分离存储，防止梯度更新破坏参数值 */
    float* W_out_params;          /**< 输出投影矩阵参数 [output_size * hidden_size] */
    float* b_out_params;          /**< 输出投影偏置参数 [output_size] */
    float* W_out_gradients;       /**< 输出投影矩阵梯度 [output_size * hidden_size] */
    float* b_out_gradients;       /**< 输出投影偏置梯度 [output_size] */
};
#endif

#ifdef __cplusplus
}
#endif

#endif
