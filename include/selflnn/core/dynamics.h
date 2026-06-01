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

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_DYNAMICS_H