/**
 * @file dynamics.h
 * @brief 动态系统接口
 * 
 * 连续时间动态系统建模，用于描述液态神经网络的演化动态。
 */

#ifndef SELFLNN_DYNAMICS_H
#define SELFLNN_DYNAMICS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief ODE求解器类型
 */
typedef enum {
    SOLVER_EULER = 0,          /**< 欧拉法（一阶精度） */
    SOLVER_RK2 = 1,            /**< 二阶Runge-Kutta方法 */
    SOLVER_RK4 = 2,            /**< 经典四阶Runge-Kutta方法 */
    SOLVER_ADAPTIVE_RK = 3,    /**< 自适应步长Runge-Kutta方法 */
    SOLVER_VERLET = 4,         /**< Verlet积分方法（用于保守系统） */
    SOLVER_SYMPLECTIC = 5,     /**< 辛积分方法（保持哈密顿结构） */
    SOLVER_IMPLICIT_EULER = 6, /**< 隐式欧拉法（L-稳定，处理刚性方程） */
    SOLVER_BDF2 = 7,           /**< 二阶后向微分公式（BDF-2，处理刚性方程） */
    SOLVER_DP54 = 8,           /**< Dormand-Prince 5(4)自适应步长求解器 */
    SOLVER_ROSENBROCK = 9,     /**< Rosenbrock刚性方程求解器（线性隐式） */
    SOLVER_FOREST_RUTH = 10    /**< Forest-Ruth四阶辛积分器（保守系统） */
} ODESolverType;

/**
 * @brief 动态系统配置
 */
typedef struct {
    size_t state_size;      /**< 状态向量大小 */
    float time_scale;       /**< 时间尺度 */
    float damping;          /**< 阻尼系数 */
    float nonlinearity;     /**< 非线性强度 */
    int enable_noise;       /**< 是否启用噪声 */
    float noise_std;        /**< 噪声标准差 */
    ODESolverType solver_type; /**< ODE求解器类型 */
    float solver_tolerance;   /**< 求解器容差（用于自适应方法） */
    int enable_adaptive_step; /**< 是否启用自适应时间步长 */
    float min_step_size;      /**< 最小时间步长 */
    float max_step_size;      /**< 最大时间步长 */
    float state_clamp; /**< 状态限幅值，默认10.0f */
    float velocity_clamp; /**< 速度限幅值，默认5.0f */
} DynamicsConfig;

/**
 * @brief 动态系统句柄
 */
typedef struct DynamicsSystem DynamicsSystem;

/**
 * @brief 创建动态系统实例
 * 
 * @param config 系统配置
 * @return DynamicsSystem* 系统句柄，失败返回NULL
 */
DynamicsSystem* dynamics_create(const DynamicsConfig* config);

/**
 * @brief 释放动态系统实例
 * 
 * @param system 系统句柄
 */
void dynamics_free(DynamicsSystem* system);

/**
 * @brief 更新动态系统状态
 * 
 * @param system 系统句柄
 * @param input 输入向量
 * @param dt 时间步长
 * @param state 状态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int dynamics_update(DynamicsSystem* system, const float* input, 
                    float dt, float* state);

/**
 * @brief 重置动态系统状态
 * 
 * @param system 系统句柄
 */
void dynamics_reset(DynamicsSystem* system);

/**
 * @brief 获取动态系统配置
 * 
 * @param system 系统句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int dynamics_get_config(const DynamicsSystem* system, DynamicsConfig* config);

/**
 * @brief 设置动态系统配置
 * 
 * @param system 系统句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int dynamics_set_config(DynamicsSystem* system, const DynamicsConfig* config);

/**
 * @brief 获取动态系统统计信息
 * 
 * @param system 系统句柄
 * @param avg_velocity 平均速度输出缓冲区
 * @param max_velocity 最大速度输出缓冲区
 * @param stability 稳定性指标输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int dynamics_get_stats(const DynamicsSystem* system, float* avg_velocity,
                       float* max_velocity, float* stability);

/* ============ D-003修复: 微分动力学扩展接口 ============ */

/**
 * @brief 可微分前向动力学（生成平滑轨迹）
 *
 * 执行离散时间步的可微分前向仿真，生成完整状态轨迹。
 * 支持接触动力学和ODE积分，适用于梯度优化（微分物理）。
 *
 * @param system 动态系统实例
 * @param initial_state 初始状态向量（dim=state_dim）
 * @param state_dim 状态向量维度
 * @param control_input 控制输入向量（dim=control_dim, 可为NULL）
 * @param control_dim 控制输入维度
 * @param dt 时间步长（秒）
 * @param num_steps 仿真步数
 * @param trajectory 输出轨迹缓冲区（大小 >= state_dim * (num_steps+1)）
 * @param trajectory_size 轨迹缓冲区大小
 * @return 0=成功, -1=参数无效, -2=维度不匹配, -3=缓冲区不足
 */
int dynamics_differentiable_forward(DynamicsSystem* system,
                                     const float* initial_state, size_t state_dim,
                                     const float* control_input, size_t control_dim,
                                     float dt, int num_steps,
                                     float* trajectory, size_t trajectory_size);

/**
 * @brief 可微分反向动力学（伴随法梯度传播）
 *
 * 通过轨迹反向传播损失梯度，使用伴随灵敏度法(CfC伴随法)
 * 计算关于动力学参数的梯度。支持通过整个物理仿真过程进行端到端学习。
 *
 * @param system 动态系统实例
 * @param trajectory 前向传播生成的状态轨迹
 * @param state_dim 状态向量维度
 * @param loss_gradient 损失关于最终状态的梯度（dim=grad_dim）
 * @param grad_dim 梯度维度
 * @param dt 时间步长（秒）
 * @param num_steps 仿真步数
 * @param param_gradient 输出参数梯度缓冲区（大小 >= param_grad_size）
 * @param param_grad_size 参数梯度缓冲区大小
 * @return 0=成功, -1=参数无效
 */
int dynamics_differentiable_backward(DynamicsSystem* system,
                                      const float* trajectory, size_t state_dim,
                                      const float* loss_gradient, size_t grad_dim,
                                      float dt, int num_steps,
                                      float* param_gradient, size_t param_grad_size);

/**
 * @brief 接触力梯度计算
 *
 * 计算两物体间接触力的空间梯度，用于机器人抓取/操作
 * 的力控制优化和接触动力学学习。
 *
 * @param system 动态系统实例
 * @param object_pos_a 物体A位置向量
 * @param object_pos_b 物体B位置向量
 * @param contact_distance 接触判定距离阈值
 * @param force_gradient 输出力梯度矩阵（dim=grad_size）
 * @param grad_size 梯度向量大小
 * @return 0=成功, -1=参数无效
 */
int dynamics_compute_contact_gradient(DynamicsSystem* system,
                                       const float* object_pos_a, const float* object_pos_b,
                                       float contact_distance,
                                       float* force_gradient, size_t grad_size);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_DYNAMICS_H