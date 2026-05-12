#ifndef SELFLNN_QUATERNION_CFC_H
#define SELFLNN_QUATERNION_CFC_H

#include "selflnn/core/common.h"
#include "selflnn/core/quaternion_lnn.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file quaternion_cfc.h
 * @brief 四元数CfC单元接口
 * 
 * 将四元数表示与CfC动力学深度集成，实现四维超复杂连续时间动态系统，
 * 支持旋转不变性和四维状态演化。
 */

/**
 * @brief 四元数CfC单元配置
 */
typedef struct {
    Quaternion initial_state;        /**< 初始四元数状态 */
    Quaternion input_gain;           /**< 输入增益四元数 */
    Quaternion feedback_gain;        /**< 反馈增益四元数 */
    Quaternion time_constant;        /**< 时间常数四元数（各分量可独立） */
    Quaternion output_gain;          /**< 输出增益四元数 */
    
    float noise_std;                 /**< 噪声标准差 */
    int enable_adaptation;           /**< 是否启用参数自适应 */
    int enable_evolution;            /**< 是否启用演化 */
    int solver_type;                 /**< ODE求解器类型: 0=闭式解, 1=RK4, 2=RK45, 3=DP54, 4=Forest-Ruth, 5=Rosenbrock, 6=Verlet, 7=BDF2 */
    float adaptive_rel_tol;          /**< 自适应步长相对容限（solver_type=2,3,5时生效） */
    float adaptive_abs_tol;          /**< 自适应步长绝对容限 */
    float min_dt;                    /**< 最小步长 */
    float max_dt;                    /**< 最大步长 */
    int use_cfc_closed_form;         /**< 是否使用CfC闭式解（solver_type=0时生效） */
} QuaternionCfcConfig;

/**
 * @brief 四元数CfC单元句柄（不透明类型）
 */
typedef struct QuaternionCfcCell QuaternionCfcCell;

/**
 * @brief 创建四元数CfC单元
 * 
 * @param config 配置参数
 * @return QuaternionCfcCell* 成功返回单元指针，失败返回NULL
 */
QuaternionCfcCell* quaternion_cfc_cell_create(const QuaternionCfcConfig* config);

/**
 * @brief 销毁四元数CfC单元
 * 
 * @param cell 单元指针
 */
void quaternion_cfc_cell_destroy(QuaternionCfcCell* cell);

/**
 * @brief 更新四元数CfC单元状态
 * 
 * @param cell 单元指针
 * @param input 输入四元数
 * @param dt 时间步长（秒）
 * @param output 输出四元数（可选，可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int quaternion_cfc_cell_update(QuaternionCfcCell* cell, const Quaternion* input, 
                               float dt, Quaternion* output);

/**
 * @brief 获取四元数CfC单元状态
 * 
 * @param cell 单元指针
 * @return const Quaternion* 状态四元数指针
 */
const Quaternion* quaternion_cfc_cell_get_state(const QuaternionCfcCell* cell);

/**
 * @brief 设置四元数CfC单元状态
 * 
 * @param cell 单元指针
 * @param state 状态四元数
 * @return int 成功返回0，失败返回-1
 */
int quaternion_cfc_cell_set_state(QuaternionCfcCell* cell, const Quaternion* state);

/**
 * @brief 配置四元数CfC单元参数
 * 
 * @param cell 单元指针
 * @param config 配置参数
 * @return int 成功返回0，失败返回-1
 */
int quaternion_cfc_cell_configure(QuaternionCfcCell* cell, const QuaternionCfcConfig* config);

/**
 * @brief 执行四元数CfC单元自适应
 * 
 * @param cell 单元指针
 * @param error 误差四元数
 * @param learning_rate 学习率
 * @return int 成功返回0，失败返回-1
 */
int quaternion_cfc_cell_adapt(QuaternionCfcCell* cell, const Quaternion* error, float learning_rate);

/**
 * @brief 执行四元数CfC单元演化
 * 
 * @param cell 单元指针
 * @param mutation_rate 突变率
 * @return int 成功返回0，失败返回-1
 */
int quaternion_cfc_cell_evolve(QuaternionCfcCell* cell, float mutation_rate);

/**
 * @brief 计算四元数CfC单元旋转不变性度量
 * 
 * @param cell 单元指针
 * @param rotation 旋转四元数
 * @return float 旋转不变性分数（0.0-1.0）
 */
float quaternion_cfc_cell_rotation_invariance(const QuaternionCfcCell* cell, 
                                              const Quaternion* rotation);

/**
 * @brief 重置四元数CfC单元状态
 * 
 * @param cell 单元指针
 * @return int 成功返回0，失败返回-1
 */
int quaternion_cfc_cell_reset(QuaternionCfcCell* cell);

/**
 * @brief 使用CfC闭式解更新四元数状态（增强版）
 * 
 * 对每个四元数分量应用CfC闭式解:
 *   new = prev * exp(-dt/τ) + (1 - exp(-dt/τ)) * drive_normalized
 * 支持四参数独立时间常数，保留更多旋转变换特性
 * 
 * @param state 当前状态四元数（输入输出）
 * @param drive 驱动四元数（输入*增益 + 反馈*状态）
 * @param time_constant 时间常数四元数（各分量独立）
 * @param dt 时间步长
 * @param output 输出四元数（可选）
 * @return int 成功返回0，失败返回-1
 */
int quaternion_cfc_closed_form_update(Quaternion* state, const Quaternion* drive,
                                       const Quaternion* time_constant, float dt,
                                       Quaternion* output);

/**
 * @brief 四元数CfC ODE右端项函数（用于标准求解器）
 * 
 * 计算 dq/dt = -q/τ + f(q, u)
 * 其中 f(q, u) = input_gain * u + feedback_gain * q
 * 四元数乘法确保旋转结构保留
 * 
 * @param t 当前时间
 * @param q 状态数组 [w,x,y,z,w2,x2,y2,z2,...] 每4个一组
 * @param dqdt 导数输出数组
 * @param ctx 上下文指针 (QuaternionCfcCell*)
 * @return int 成功返回0，失败返回-1
 */
int quaternion_cfc_rhs(float t, const float* q, float* dqdt, void* ctx);

/**
 * @brief 增强四元数ODE求解器（支持所有求解器类型）
 * 
 * 增强版:
 * - 使用实际四元数CfC动力学 RHS
 * - 支持自适应步长（DP54/Rosenbrock）
 * - 支持并行化求解
 * - 求解后自动归一化
 * 
 * @param qcfc_cell 四元数CfC单元指针
 * @param quat_input 输入四元数数组 [w,x,y,z,...]
 * @param input_dim 输入维度（必须是4的倍数）
 * @param solver_type 求解器类型 (0=闭式解,1=RK4,2=RK45,3=DP54,4=Forest-Ruth,5=Rosenbrock,6=Verlet,7=BDF2)
 * @param dt 时间步长
 * @param quat_output 输出四元数数组
 * @param steps_taken 实际步数（可选输出）
 * @return int 成功返回0，失败返回-1
 */
int quaternion_cfc_solve_with_solver(void* qcfc_cell, const float* quat_input,
                                      int input_dim, int solver_type, float dt,
                                      float* quat_output, int* steps_taken);

/**
 * @brief 批量并行四元数ODE求解
 * 
 * 对多个独立的四元数CfC单元并行执行求解
 * 
 * @param cells 单元指针数组
 * @param inputs 输入数组 [cell0_w,cell0_x,cell0_y,cell0_z,cell1_w,...]
 * @param n_cells 单元数量
 * @param solver_type 求解器类型
 * @param dt 时间步长
 * @param outputs 输出数组
 * @return int 成功返回0，失败返回-1
 */
int quaternion_cfc_parallel_solve(QuaternionCfcCell** cells, const float* inputs,
                                   int n_cells, int solver_type, float dt,
                                   float* outputs);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_QUATERNION_CFC_H */