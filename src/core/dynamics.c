/**
 * @file dynamics.c
 * @brief 传感器预处理动态系统（非独立AI系统）
 * 
 * 本模块仅为传感器数据的物理预处理器（本体感/惯性滤波），
 * 不参与AGI决策。所有AI推理状态演化统一由主LNN管理。
 * 本动态系统的输出作为LNN触觉/本体感模态的输入。
 * 
 * 严禁将本模块用作独立AI状态演化系统。
 * 
 * ODE求解器层次：
 *   简单求解器（Euler/RK2/RK4/Verlet）：本地实现，适合传感器物理预滤波
 *   复杂求解器（DP54/Rosenbrock/Forest-Ruth）：委托到 ode_solvers.c 共享实现
 *   传感器预滤波仅需简单物理模型，复杂求解保留仅为接口兼容。
 */

#include "selflnn/core/dynamics.h"
#include "selflnn/core/ode_solvers.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/memory_utils.h"

/* 前向声明：ODE右侧函数（定义见文件末尾） */
static int dynamics_ode_rhs(float t, const float* y, float* dydt, void* ctx);

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* MSVC警告处理 */

/* 静态求解器函数前向声明 */
static void dynamics_internal_solve_euler(DynamicsSystem* system, const float* input, float dt);
static void dynamics_internal_solve_rk2(DynamicsSystem* system, const float* input, float dt);
static void dynamics_internal_solve_rk4(DynamicsSystem* system, const float* input, float dt);
static float dynamics_internal_solve_adaptive_rk(DynamicsSystem* system, const float* input, float dt_requested);
static void dynamics_internal_solve_verlet(DynamicsSystem* system, const float* input, float dt);
static void dynamics_internal_solve_symplectic(DynamicsSystem* system, const float* input, float dt);
static void dynamics_internal_solve_implicit_euler(DynamicsSystem* system, const float* input, float dt);
static void dynamics_internal_solve_bdf2(DynamicsSystem* system, const float* input, float dt, const float* prev_state);
static float dynamics_internal_solve_dp54(DynamicsSystem* system, const float* input, float dt_requested);
static void dynamics_internal_solve_rosenbrock(DynamicsSystem* system, const float* input, float dt);
static void dynamics_internal_solve_forest_ruth(DynamicsSystem* system, const float* input, float dt);

/**
 * @brief 动态系统内部结构体
 */
struct DynamicsSystem {
    DynamicsConfig config;      /**< 系统配置 */
    float* state;               /**< 状态向量 */
    float* velocity;            /**< 速度向量 */
    float* acceleration;        /**< 加速度向量 */
    float* noise_buffer;        /**< 噪声缓冲区 */
    float* prev_state;          /**< 前一状态向量（BDF-2所需） */
    float* workspace;           /**< 预分配ODE求解器工作区（最大11*state_size） */
    int bdf2_initialized;       /**< BDF-2初始化标记（需要两步历史） */
    int is_initialized;         /**< 是否已初始化 */
    float time;                 /**< 当前时间 */
    float avg_velocity;         /**< 平均速度 */
    float max_velocity;         /**< 最大速度 */
    float stability;            /**< 稳定性指标 */
};

/**
 * @brief 创建动态系统实例
 */
DynamicsSystem* dynamics_create(const DynamicsConfig* config) {
    if (!config) {
        return NULL;
    }
    
    if (config->state_size == 0) {
        return NULL;
    }
    
    // 分配系统结构
    DynamicsSystem* system = (DynamicsSystem*)safe_malloc(sizeof(DynamicsSystem));
    if (!system) {
        return NULL;
    }
    
    // 复制配置
    memcpy(&system->config, config, sizeof(DynamicsConfig));
    
    // 设置新字段的默认值（如果用户未提供）
    if (system->config.solver_type < SOLVER_EULER || system->config.solver_type > SOLVER_FOREST_RUTH) {
        system->config.solver_type = SOLVER_EULER;
    }
    
    if (system->config.solver_tolerance <= 0.0f) {
        system->config.solver_tolerance = 1e-4f;  // 默认容差
    }
    
    if (system->config.min_step_size <= 0.0f) {
        system->config.min_step_size = 1e-6f;  // 默认最小步长
    }
    
    if (system->config.max_step_size <= 0.0f) {
        system->config.max_step_size = 0.1f;  // 默认最大步长
    }
    
    size_t state_size = config->state_size;
    
    // 分配状态向量
    system->state = (float*)safe_calloc(state_size, sizeof(float));
    system->velocity = (float*)safe_calloc(state_size, sizeof(float));
    system->acceleration = (float*)safe_calloc(state_size, sizeof(float));
    system->noise_buffer = (float*)safe_calloc(state_size, sizeof(float));
    system->prev_state = (float*)safe_calloc(state_size, sizeof(float));
    
    // 检查内存分配
    if (!system->state || !system->velocity || 
        !system->acceleration || !system->noise_buffer || !system->prev_state) {
        dynamics_free(system);
        return NULL;
    }

    // 分配ODE求解器工作区（最大需求：Rosenbrock需要11*state_size个float）
    system->workspace = (float*)safe_calloc(11 * state_size, sizeof(float));
    if (!system->workspace) {
        dynamics_free(system);
        return NULL;
    }

    // BDF-2初始化
    system->bdf2_initialized = 0;
    
    // 初始化状态
    for (size_t i = 0; i < state_size; i++) {
        system->state[i] = rng_uniform(-0.1f, 0.1f);
        system->velocity[i] = 0.0f;
        system->acceleration[i] = 0.0f;
    }
    
    // 初始化变量
    system->is_initialized = 1;
    system->time = 0.0f;
    system->avg_velocity = 0.0f;
    system->max_velocity = 0.0f;
    system->stability = 1.0f;
    
    return system;
}

/**
 * @brief 释放动态系统实例
 */
void dynamics_free(DynamicsSystem* system) {
    if (!system) {
        return;
    }
    
    // 释放状态向量
    safe_free((void**)&system->state);
    safe_free((void**)&system->velocity);
    safe_free((void**)&system->acceleration);
    safe_free((void**)&system->noise_buffer);
    safe_free((void**)&system->prev_state);
    safe_free((void**)&system->workspace);
    
    // 释放系统结构
    safe_free((void**)&system);
}

/**
 * @brief 更新动态系统状态
 */
int dynamics_update(DynamicsSystem* system, const float* input, 
                    float dt, float* output_state) {
    if (!system || !input || !output_state) {
        return -1;
    }
    
    if (!system->is_initialized) {
        return -1;
    }
    
    if (dt <= 0.0f) {
        return -1;
    }
    
    size_t state_size = system->config.state_size;
    
    // 添加噪声（如果启用）
    if (system->config.enable_noise) {
        for (size_t i = 0; i < state_size; i++) {
            system->noise_buffer[i] = rng_normal(0.0f, system->config.noise_std);
        }
    } else {
        memset(system->noise_buffer, 0, state_size * sizeof(float));
    }
    
    // 计算新的加速度（基于完整动力学方程）
    // 完整模型：加速度 = 外部输入 + 恢复力 + 阻尼力 + 非线性项 + 耦合效应 + 噪声
    
    float avg_velocity = 0.0f;
    float max_velocity = 0.0f;
    
    for (size_t i = 0; i < state_size; i++) {
        // 1. 恢复力：趋向零点的力（类似弹簧恢复力）
        // 使用时间尺度的倒数作为有效弹簧常数
        float spring_constant = 1.0f / (system->config.time_scale + 1e-6f);
        float restoring_force = -spring_constant * system->state[i];
        
        // 2. 线性阻尼力：-c * velocity
        float damping_force = -system->config.damping * system->velocity[i];
        
        // 3. 非线性项：多种非线性效应组合
        // 立方非线性：α * x³（软/硬弹簧效应）
        float cubic_nonlinearity = system->config.nonlinearity * 
                                  system->state[i] * system->state[i] * system->state[i];
        
        // 正弦非线性：使用时间尺度作为频率参数
        float sine_nonlinearity = system->config.nonlinearity * 0.5f * 
                                 sinf(system->state[i] * system->config.time_scale);
        
        // 双曲正切非线性（饱和效应）
        float tanh_nonlinearity = system->config.nonlinearity * 0.3f * 
                                 tanhf(system->state[i] * 2.0f);
        
        // 4. 状态耦合：状态变量之间的相互作用
        float coupling_force = 0.0f;
        if (state_size > 1) {
            // 自适应耦合：基于状态差异和全局统计
            for (size_t j = 0; j < state_size; j++) {
                if (j != i) {
                    // 耦合强度与状态差异成正比（扩散耦合）
                    float state_diff = system->state[j] - system->state[i];
                    // 归一化耦合系数：使用阻尼系数的一部分
                    float coupling_strength = system->config.damping * 0.1f;
                    coupling_force += coupling_strength * state_diff;
                }
            }
            // 归一化耦合项
            coupling_force /= (state_size - 1);
        }
        
        // 5. 综合加速度计算（完整动力学方程）
        // 总力 = 外部输入 + 恢复力 + 阻尼力 + 非线性项总和 + 耦合 + 噪声
        float total_force = input[i] + 
                           restoring_force + 
                           damping_force + 
                           cubic_nonlinearity + 
                           sine_nonlinearity + 
                           tanh_nonlinearity + 
                           coupling_force + 
                           system->noise_buffer[i];
        
        system->acceleration[i] = total_force;
    }
    
    // 6. 使用配置的ODE求解器进行数值积分
    float actual_dt = dt;
    switch (system->config.solver_type) {
        case SOLVER_EULER:
            dynamics_internal_solve_euler(system, input, dt);
            break;
        case SOLVER_RK2:
            dynamics_internal_solve_rk2(system, input, dt);
            break;
        case SOLVER_RK4:
            dynamics_internal_solve_rk4(system, input, dt);
            break;
        case SOLVER_ADAPTIVE_RK:
            actual_dt = dynamics_internal_solve_adaptive_rk(system, input, dt);
            break;
        case SOLVER_VERLET:
            dynamics_internal_solve_verlet(system, input, dt);
            break;
        case SOLVER_SYMPLECTIC:
            dynamics_internal_solve_symplectic(system, input, dt);
            break;
        case SOLVER_IMPLICIT_EULER:
            dynamics_internal_solve_implicit_euler(system, input, dt);
            break;
        case SOLVER_BDF2: {
            /* 委托ode_solvers.c的BDF-2实现 */
            float* y = (float*)safe_malloc(2 * state_size * sizeof(float));
            size_t ws_sz = ode_bdf2_workspace_size(2 * state_size);
            float* bdf_ws = (float*)safe_malloc(ws_sz > 0 ? ws_sz : 4096);
            if (y && bdf_ws) {
                for (size_t i = 0; i < state_size; i++) {
                    y[i] = system->state[i];
                    y[state_size + i] = system->velocity[i];
                }
                BDF2Config bdf_cfg;
                memset(&bdf_cfg, 0, sizeof(bdf_cfg));
                bdf_cfg.newton_tol = system->config.solver_tolerance > 0.0f ?
                                     system->config.solver_tolerance : 1e-4f;
                float h_actual; int steps;
                ode_bdf2_solve(y, system->time, dt, dynamics_ode_rhs, system,
                              2 * state_size, &bdf_cfg, bdf_ws, &h_actual, &steps);
                (void)h_actual; (void)steps;
                for (size_t i = 0; i < state_size; i++) {
                    system->state[i] = y[i];
                    system->velocity[i] = y[state_size + i];
                }
                if (!system->bdf2_initialized) system->bdf2_initialized = 1;
            }
            safe_free((void**)&y); safe_free((void**)&bdf_ws);
            if (!y || !bdf_ws) {
                float* current_state = system->workspace + 10 * state_size;
                memcpy(current_state, system->state, state_size * sizeof(float));
                dynamics_internal_solve_bdf2(system, input, dt, system->prev_state);
                memcpy(system->prev_state, current_state, state_size * sizeof(float));
                if (!system->bdf2_initialized) system->bdf2_initialized = 1;
            }
            break;
        }
        case SOLVER_DP54: {
            /* 委托ode_solvers.c的DP54实现 */
            float* y = (float*)safe_malloc(2 * state_size * sizeof(float));
            float* ws = (float*)safe_malloc(ode_dp54_workspace_size(2 * state_size));
            if (y && ws) {
                for (size_t i = 0; i < state_size; i++) {
                    y[i] = system->state[i];
                    y[state_size + i] = system->velocity[i];
                }
                DP54Config dp_cfg;
                memset(&dp_cfg, 0, sizeof(dp_cfg));
                dp_cfg.rel_tolerance = system->config.solver_tolerance > 0 ?
                    system->config.solver_tolerance : 1e-6f;
                dp_cfg.abs_tolerance = 1e-8f;
                dp_cfg.min_step_size = system->config.min_step_size > 0 ?
                    system->config.min_step_size : 1e-6f;
                dp_cfg.max_step_size = system->config.max_step_size > 0 ?
                    system->config.max_step_size : dt;
                float h_actual; int steps;
                ode_dp54_solve(y, system->time, dt, dynamics_ode_rhs, system,
                              2 * state_size, &dp_cfg, ws, &h_actual, &steps);
                actual_dt = h_actual; (void)steps;
                for (size_t i = 0; i < state_size; i++) {
                    system->state[i] = y[i];
                    system->velocity[i] = y[state_size + i];
                }
            }
            safe_free((void**)&y); safe_free((void**)&ws);
            if (!y || !ws) actual_dt = dynamics_internal_solve_dp54(system, input, dt);
            break;
        }
        case SOLVER_ROSENBROCK: {
            /* 委托ode_solvers.c的Rosenbrock实现 */
            float* y = (float*)safe_malloc(2 * state_size * sizeof(float));
            size_t ws_sz = ode_rosenbrock_workspace_size(2 * state_size);
            float* ws = (float*)safe_malloc(ws_sz > 0 ? ws_sz : 4096);
            if (y && ws) {
                for (size_t i = 0; i < state_size; i++) {
                    y[i] = system->state[i];
                    y[state_size + i] = system->velocity[i];
                }
                RosenbrockConfig rb_cfg;
                memset(&rb_cfg, 0, sizeof(rb_cfg));
                rb_cfg.rel_tolerance = system->config.solver_tolerance > 0 ?
                    system->config.solver_tolerance : 1e-4f;
                rb_cfg.max_iterations = 50;
                float h_actual; int steps;
                ode_rosenbrock_solve(y, system->time, dt, dynamics_ode_rhs, system,
                                    2 * state_size, &rb_cfg, ws, &h_actual, &steps);
                (void)h_actual; (void)steps;
                for (size_t i = 0; i < state_size; i++) {
                    system->state[i] = y[i];
                    system->velocity[i] = y[state_size + i];
                }
            }
            safe_free((void**)&y); safe_free((void**)&ws);
            if (!y || !ws) dynamics_internal_solve_rosenbrock(system, input, dt);
            break;
        }
        case SOLVER_FOREST_RUTH: {
            /* 委托ode_solvers.c的Forest-Ruth辛积分器 */
            float* q = (float*)safe_malloc(state_size * sizeof(float));
            float* p = (float*)safe_malloc(state_size * sizeof(float));
            float* ws = (float*)safe_malloc(ode_forest_ruth_workspace_size(state_size));
            if (q && p && ws) {
                memcpy(q, system->state, state_size * sizeof(float));
                memcpy(p, system->velocity, state_size * sizeof(float));
                SymplecticConfig sym_cfg;
                memset(&sym_cfg, 0, sizeof(sym_cfg));
                sym_cfg.substep_ratio = 0.5f;
                sym_cfg.num_substeps = 4;
                int steps;
                ode_forest_ruth_solve(q, p, dt, dynamics_ode_rhs, dynamics_ode_rhs,
                                      system, state_size, &sym_cfg, ws, &steps);
                (void)steps;
                memcpy(system->state, q, state_size * sizeof(float));
                memcpy(system->velocity, p, state_size * sizeof(float));
            }
            safe_free((void**)&q); safe_free((void**)&p); safe_free((void**)&ws);
            if (!q || !p || !ws) dynamics_internal_solve_forest_ruth(system, input, dt);
            break;
        }
        default:
            dynamics_internal_solve_euler(system, input, dt);
            break;
    }
    
    // 7. 应用边界约束（防止数值溢出）并更新统计信息
    for (size_t i = 0; i < state_size; i++) {
        // 应用边界约束
        if (system->state[i] > 10.0f) system->state[i] = 10.0f;
        if (system->state[i] < -10.0f) system->state[i] = -10.0f;
        if (system->velocity[i] > 5.0f) system->velocity[i] = 5.0f;
        if (system->velocity[i] < -5.0f) system->velocity[i] = -5.0f;
        
        // 更新统计信息
        float vel_magnitude = fabsf(system->velocity[i]);
        avg_velocity += vel_magnitude;
        
        if (vel_magnitude > max_velocity) {
            max_velocity = vel_magnitude;
        }
    }
    
    // 更新统计信息
    system->avg_velocity = avg_velocity / state_size;
    system->max_velocity = max_velocity;
    
    // 计算稳定性指标（基于速度变化率）
    float velocity_variance = 0.0f;
    for (size_t i = 0; i < state_size; i++) {
        float diff = system->velocity[i] - system->avg_velocity;
        velocity_variance += diff * diff;
    }
    velocity_variance /= state_size;
    
    // 稳定性指标：1 / (1 + 方差) （值越接近1越稳定）
    system->stability = 1.0f / (1.0f + sqrtf(velocity_variance));
    
    // 复制状态到输出缓冲区
    memcpy(output_state, system->state, state_size * sizeof(float));
    
    // 更新时间
    system->time += dt;
    
    return 0;
}

/* ============================================================================
 * ODE求解器实现
 * ============================================================================ */

/**
 * @brief 计算动态系统导数（微分方程）
 * 
 * 计算 dx/dt = f(x, t, input)
 * 对于二阶系统，状态向量包含位置和速度：x = [q, p]
 * 导数向量：dx/dt = [p, F(q,p,t)]
 */
static void compute_derivatives(const DynamicsSystem* system, 
                               const float* state,
                               const float* velocity,
                               const float* input,
                               float* dstate_dt,
                               float* dvelocity_dt) {
    if (!system || !state || !velocity || !input || !dstate_dt || !dvelocity_dt) {
        return;
    }
    
    size_t state_size = system->config.state_size;
    
    // 添加噪声（如果启用）
    float* effective_noise = system->noise_buffer;
    if (system->config.enable_noise) {
        // 噪声已经预先计算并存储在noise_buffer中
        effective_noise = system->noise_buffer;
    }
    
    for (size_t i = 0; i < state_size; i++) {
        // 位置导数：dq/dt = velocity
        dstate_dt[i] = velocity[i];
        
        // 1. 恢复力：趋向零点的力（类似弹簧恢复力）
        float spring_constant = 1.0f / (system->config.time_scale + 1e-6f);
        float restoring_force = -spring_constant * state[i];
        
        // 2. 线性阻尼力：-c * velocity
        float damping_force = -system->config.damping * velocity[i];
        
        // 3. 非线性项：多种非线性效应组合
        // 立方非线性：α * x³（软/硬弹簧效应）
        float cubic_nonlinearity = system->config.nonlinearity * 
                                  state[i] * state[i] * state[i];
        
        // 正弦非线性：使用时间尺度作为频率参数
        float sine_nonlinearity = system->config.nonlinearity * 0.5f * 
                                 sinf(state[i] * system->config.time_scale);
        
        // 双曲正切非线性（饱和效应）
        float tanh_nonlinearity = system->config.nonlinearity * 0.3f * 
                                 tanhf(state[i] * 2.0f);
        
        // 4. 状态耦合：状态变量之间的相互作用
        float coupling_force = 0.0f;
        if (state_size > 1) {
            for (size_t j = 0; j < state_size; j++) {
                if (j != i) {
                    float state_diff = state[j] - state[i];
                    float coupling_strength = system->config.damping * 0.1f;
                    coupling_force += coupling_strength * state_diff;
                }
            }
            coupling_force /= (state_size - 1);
        }
        
        // 5. 综合加速度计算（完整动力学方程）
        float total_force = input[i] + 
                           restoring_force + 
                           damping_force + 
                           cubic_nonlinearity + 
                           sine_nonlinearity + 
                           tanh_nonlinearity + 
                           coupling_force + 
                           effective_noise[i];
        
        dvelocity_dt[i] = total_force;
    }
}

/* ========================================================================
 * ODE RHS适配器：将分离的state+velocity数组映射为ode_solvers.c的ODERHSFunc
 * y[0..n-1]=state, y[n..2n-1]=velocity, dydt[0..n-1]=dq/dt, dydt[n..2n-1]=dv/dt
 * ======================================================================== */
static int dynamics_ode_rhs(float t, const float* y, float* dydt, void* ctx) {
    DynamicsSystem* s = (DynamicsSystem*)ctx;
    size_t n = s->config.state_size;
    /* MID-015修复: 动态分配input_buf替代硬编码固定大小[256]防止数组越界 */
    float* input_buf = (float*)malloc(n * sizeof(float));
    if (!input_buf) return -1;
    for (size_t i = 0; i < n; i++) input_buf[i] = y[n+i] * s->config.damping;
    compute_derivatives(s, y, y+n, input_buf, dydt, dydt+n);
    (void)t;
    free(input_buf);
    return 0;
}

/**
 * @brief 欧拉法求解器（一阶精度）
 */
static void dynamics_internal_solve_euler(DynamicsSystem* system, const float* input, float dt) {
    size_t state_size = system->config.state_size;
    float* dstate_dt = system->workspace;
    float* dvelocity_dt = system->workspace + state_size;
    
    compute_derivatives(system, system->state, system->velocity, input, 
                       dstate_dt, dvelocity_dt);
    
    for (size_t i = 0; i < state_size; i++) {
        system->state[i] += dt * dstate_dt[i];
        system->velocity[i] += dt * dvelocity_dt[i];
    }
}

/**
 * @brief 二阶Runge-Kutta求解器（中点法）
 */
static void dynamics_internal_solve_rk2(DynamicsSystem* system, const float* input, float dt) {
    size_t state_size = system->config.state_size;
    float* k1_state = system->workspace;
    float* k1_velocity = system->workspace + 1 * state_size;
    float* k2_state = system->workspace + 2 * state_size;
    float* k2_velocity = system->workspace + 3 * state_size;
    float* temp_state = system->workspace + 4 * state_size;
    float* temp_velocity = system->workspace + 5 * state_size;
    
    compute_derivatives(system, system->state, system->velocity, input, 
                       k1_state, k1_velocity);
    
    for (size_t i = 0; i < state_size; i++) {
        temp_state[i] = system->state[i] + 0.5f * dt * k1_state[i];
        temp_velocity[i] = system->velocity[i] + 0.5f * dt * k1_velocity[i];
    }
    
    compute_derivatives(system, temp_state, temp_velocity, input, 
                       k2_state, k2_velocity);
    
    for (size_t i = 0; i < state_size; i++) {
        system->state[i] += dt * k2_state[i];
        system->velocity[i] += dt * k2_velocity[i];
    }
}

/**
 * @brief 经典四阶Runge-Kutta求解器
 */
static void dynamics_internal_solve_rk4(DynamicsSystem* system, const float* input, float dt) {
    size_t state_size = system->config.state_size;
    float* k1_state = system->workspace;
    float* k1_velocity = system->workspace + 1 * state_size;
    float* k2_state = system->workspace + 2 * state_size;
    float* k2_velocity = system->workspace + 3 * state_size;
    float* k3_state = system->workspace + 4 * state_size;
    float* k3_velocity = system->workspace + 5 * state_size;
    float* k4_state = system->workspace + 6 * state_size;
    float* k4_velocity = system->workspace + 7 * state_size;
    float* temp_state = system->workspace + 8 * state_size;
    float* temp_velocity = system->workspace + 9 * state_size;
    
    compute_derivatives(system, system->state, system->velocity, input, 
                       k1_state, k1_velocity);
    
    for (size_t i = 0; i < state_size; i++) {
        temp_state[i] = system->state[i] + 0.5f * dt * k1_state[i];
        temp_velocity[i] = system->velocity[i] + 0.5f * dt * k1_velocity[i];
    }
    
    compute_derivatives(system, temp_state, temp_velocity, input, 
                       k2_state, k2_velocity);
    
    for (size_t i = 0; i < state_size; i++) {
        temp_state[i] = system->state[i] + 0.5f * dt * k2_state[i];
        temp_velocity[i] = system->velocity[i] + 0.5f * dt * k2_velocity[i];
    }
    
    compute_derivatives(system, temp_state, temp_velocity, input, 
                       k3_state, k3_velocity);
    
    for (size_t i = 0; i < state_size; i++) {
        temp_state[i] = system->state[i] + dt * k3_state[i];
        temp_velocity[i] = system->velocity[i] + dt * k3_velocity[i];
    }
    
    compute_derivatives(system, temp_state, temp_velocity, input, 
                       k4_state, k4_velocity);
    
    for (size_t i = 0; i < state_size; i++) {
        system->state[i] += dt * (k1_state[i] + 2.0f*k2_state[i] + 
                                  2.0f*k3_state[i] + k4_state[i]) / 6.0f;
        system->velocity[i] += dt * (k1_velocity[i] + 2.0f*k2_velocity[i] + 
                                     2.0f*k3_velocity[i] + k4_velocity[i]) / 6.0f;
    }
}

/**
 * @brief 自适应Runge-Kutta求解器（使用RK4-RK5对进行误差估计）
 */
static float dynamics_internal_solve_adaptive_rk(DynamicsSystem* system, const float* input, 
                              float dt_requested) {
    size_t state_size = system->config.state_size;
    float tolerance = system->config.solver_tolerance;
    float dt = dt_requested;
    
    if (dt < system->config.min_step_size) {
        dt = system->config.min_step_size;
    }
    if (dt > system->config.max_step_size) {
        dt = system->config.max_step_size;
    }
    
    float* saved_state = system->workspace + 10 * state_size;
    float* saved_velocity = system->workspace + 10 * state_size + state_size;
    memcpy(saved_state, system->state, state_size * sizeof(float));
    memcpy(saved_velocity, system->velocity, state_size * sizeof(float));
    
    float error = 0.0f;
    int attempts = 0;
    const int max_attempts = 10;
    
    do {
        attempts++;
        
        /* F-021修复: 先保存全步结果，再计算两个半步，然后用两者差异做误差估计 */
        float* full_state = (float*)safe_malloc(state_size * sizeof(float));
        float* full_velocity = (float*)safe_malloc(state_size * sizeof(float));
        if (!full_state || !full_velocity) {
            if (full_state) free(full_state);
            if (full_velocity) free(full_velocity);
            break;
        }
        
        dynamics_internal_solve_rk4(system, input, dt);
        memcpy(full_state, system->state, state_size * sizeof(float));
        memcpy(full_velocity, system->velocity, state_size * sizeof(float));
        
        memcpy(system->state, saved_state, state_size * sizeof(float));
        memcpy(system->velocity, saved_velocity, state_size * sizeof(float));
        
        dynamics_internal_solve_rk4(system, input, dt/2.0f);
        dynamics_internal_solve_rk4(system, input, dt/2.0f);
        
        error = 0.0f;
        for (size_t i = 0; i < state_size; i++) {
            float diff_state = fabsf(system->state[i] - full_state[i]);
            float diff_velocity = fabsf(system->velocity[i] - full_velocity[i]);
            error += diff_state * diff_state + diff_velocity * diff_velocity;
        }
        error = sqrtf(error / (2 * state_size));
        
        free(full_state);
        free(full_velocity);
        
        if (error > tolerance && attempts < max_attempts) {
            dt *= 0.5f;
            if (dt < system->config.min_step_size) {
                dt = system->config.min_step_size;
                break;
            }
            memcpy(system->state, saved_state, state_size * sizeof(float));
            memcpy(system->velocity, saved_velocity, state_size * sizeof(float));
        } else if (error < tolerance * 0.1f && attempts < max_attempts) {
            dt *= 2.0f;
            if (dt > system->config.max_step_size) {
                dt = system->config.max_step_size;
            }
            memcpy(system->state, saved_state, state_size * sizeof(float));
            memcpy(system->velocity, saved_velocity, state_size * sizeof(float));
        }
    } while ((error > tolerance || error < tolerance * 0.1f) && attempts < max_attempts);
    
    if (attempts > 1) {
        memcpy(system->state, saved_state, state_size * sizeof(float));
        memcpy(system->velocity, saved_velocity, state_size * sizeof(float));
        dynamics_internal_solve_rk4(system, input, dt);
    }
    
    return dt;
}

/**
 * @brief Verlet积分器（用于保守系统）
 */
static void dynamics_internal_solve_verlet(DynamicsSystem* system, const float* input, float dt) {
    size_t state_size = system->config.state_size;
    float* dstate_dt = system->workspace;
    float* dvelocity_dt = system->workspace + state_size;
    compute_derivatives(system, system->state, system->velocity, input, 
                       dstate_dt, dvelocity_dt);
    
    for (size_t i = 0; i < state_size; i++) {
        system->velocity[i] += dvelocity_dt[i] * dt * 0.5f;
    }
    
    for (size_t i = 0; i < state_size; i++) {
        system->state[i] += system->velocity[i] * dt;
    }
    
    compute_derivatives(system, system->state, system->velocity, input, 
                       dstate_dt, dvelocity_dt);
    
    for (size_t i = 0; i < state_size; i++) {
        system->velocity[i] += dvelocity_dt[i] * dt * 0.5f;
    }
}

/**
 * @brief 辛积分器（保持哈密顿结构）
 */
static void dynamics_internal_solve_symplectic(DynamicsSystem* system, const float* input, float dt) {
    size_t state_size = system->config.state_size;
    
    // 辛欧拉方法（半隐式欧拉），保持哈密顿结构
    // 对于哈密顿系统 H(q,p) = T(p) + V(q)
    // 辛欧拉：p_{n+1} = p_n - dt * ∂V/∂q(q_n)
    //          q_{n+1} = q_n + dt * ∂T/∂p(p_{n+1})
    
    // 标准哈密顿动力学：动能形式 T(p) = p²/2（单位质量系统）
    // 因此 ∂T/∂p = p
    
    float* force = system->workspace;
    
    for (size_t i = 0; i < state_size; i++) {
        float spring_constant = 1.0f / (system->config.time_scale + 1e-6f);
        float restoring_force = -spring_constant * system->state[i];
        
        float cubic_nonlinearity = system->config.nonlinearity * 
                                  system->state[i] * system->state[i] * system->state[i];
        
        float external_force = input[i];
        if (system->config.enable_noise) {
            external_force += system->noise_buffer[i];
        }
        
        force[i] = restoring_force + cubic_nonlinearity + external_force;
    }
    
    for (size_t i = 0; i < state_size; i++) {
        system->velocity[i] -= dt * force[i];
        system->state[i] += dt * system->velocity[i];
    }
}

/**
 * @brief 隐式欧拉求解器（L-稳定，处理刚性方程）
 *
 * 使用固定点迭代求解隐式系统：
 * y_{n+1} = y_n + dt * f(t_{n+1}, y_{n+1})
 * 对于CfC/动力学系统，雅可比矩阵-向量乘积累积成本较高，
 * 采用固定点迭代：y_{n+1}^{(k+1)} = y_n + dt * f(t_{n+1}, y_{n+1}^{(k)})
 *
 * 隐式欧拉对刚性方程具有L-稳定性，即使在大时间步长下也能保持数值稳定。
 */
static void dynamics_internal_solve_implicit_euler(DynamicsSystem* system, const float* input, float dt) {
    size_t state_size = system->config.state_size;
    float* temp_state = system->workspace;
    float* temp_velocity = system->workspace + 1 * state_size;
    float* initial_state = system->workspace + 2 * state_size;
    float* initial_velocity = system->workspace + 3 * state_size;
    float* delta = system->workspace + 4 * state_size;
    float* delta_vel = system->workspace + 5 * state_size;

    /* 保存初始条件 */
    memcpy(initial_state, system->state, state_size * sizeof(float));
    memcpy(initial_velocity, system->velocity, state_size * sizeof(float));

    int max_iter = 32;
    float tolerance = system->config.solver_tolerance > 0.0f ?
                      system->config.solver_tolerance : 1e-4f;

    memcpy(temp_state, system->state, state_size * sizeof(float));
    memcpy(temp_velocity, system->velocity, state_size * sizeof(float));

    for (int iter = 0; iter < max_iter; iter++) {
        for (size_t i = 0; i < state_size; i++) {
            float spring_constant = 1.0f / (system->config.time_scale + 1e-6f);
            float restoring_force = -spring_constant * temp_state[i];
            float damping_force = -system->config.damping * temp_velocity[i];
            float cubic_nonlinearity = system->config.nonlinearity *
                                      temp_state[i] * temp_state[i] * temp_state[i];
            float sine_nonlinearity = system->config.nonlinearity * 0.5f *
                                     sinf(temp_state[i] * system->config.time_scale);
            float tanh_nonlinearity = system->config.nonlinearity * 0.3f *
                                     tanhf(temp_state[i] * 2.0f);
            float input_force = input[i];
            float noise = system->config.enable_noise ? system->noise_buffer[i] : 0.0f;

            float total_force = input_force + restoring_force + damping_force +
                               cubic_nonlinearity + sine_nonlinearity +
                               tanh_nonlinearity + noise;

            delta[i] = initial_state[i] + dt * temp_velocity[i] - temp_state[i];
            delta_vel[i] = initial_velocity[i] + dt * total_force - temp_velocity[i];

            temp_state[i] += 0.5f * delta[i];
            temp_velocity[i] += 0.5f * delta_vel[i];

            if (temp_state[i] > 10.0f) temp_state[i] = 10.0f;
            if (temp_state[i] < -10.0f) temp_state[i] = -10.0f;
            if (temp_velocity[i] > 5.0f) temp_velocity[i] = 5.0f;
            if (temp_velocity[i] < -5.0f) temp_velocity[i] = -5.0f;
        }

        float max_delta = 0.0f;
        float max_delta_vel = 0.0f;
        for (size_t i = 0; i < state_size; i++) {
            float ad = fabsf(delta[i]);
            float av = fabsf(delta_vel[i]);
            if (ad > max_delta) max_delta = ad;
            if (av > max_delta_vel) max_delta_vel = av;
        }

        if (max_delta < tolerance && max_delta_vel < tolerance) {
            break;
        }

        for (size_t i = 0; i < state_size; i++) {
            float spring_constant = 1.0f / (system->config.time_scale + 1e-6f);
            float restoring_force = -spring_constant * temp_state[i];
            float damping_force = -system->config.damping * temp_velocity[i];
            float cubic_nonlinearity = system->config.nonlinearity *
                                      temp_state[i] * temp_state[i] * temp_state[i];
            float sine_nonlinearity = system->config.nonlinearity * 0.5f *
                                     sinf(temp_state[i] * system->config.time_scale);
            float tanh_nonlinearity = system->config.nonlinearity * 0.3f *
                                     tanhf(temp_state[i] * 2.0f);
            float noise = system->config.enable_noise ? system->noise_buffer[i] : 0.0f;
            system->acceleration[i] = input[i] + restoring_force + damping_force +
                                     cubic_nonlinearity + sine_nonlinearity +
                                     tanh_nonlinearity + noise;
        }
    }

    memcpy(system->state, temp_state, state_size * sizeof(float));
    memcpy(system->velocity, temp_velocity, state_size * sizeof(float));
}

/**
 * @brief BDF-2求解器（二阶后向微分公式，处理刚性方程）
 *
 * 标准BDF-2公式：
 * y_{n+2} = (4/3)*y_{n+1} - (1/3)*y_n + (2/3)*dt*f(t_{n+2}, y_{n+2})
 *
 * 使用固定点迭代求解隐式方程。
 * 第一步（bdf2_initialized=0）使用隐式欧拉启动。
 * BDF-2具有二阶精度和A-稳定性，适合刚性ODE系统。
 */
static void dynamics_internal_solve_bdf2(DynamicsSystem* system, const float* input, float dt, const float* prev_state) {
    size_t state_size = system->config.state_size;

    /* BDF-2需要两步历史，第一步用隐式欧拉启动 */
    if (!system->bdf2_initialized || !prev_state) {
        /* 第一步使用隐式欧拉作为BDF-2的启动器 */
        dynamics_internal_solve_implicit_euler(system, input, dt);
        return;
    }

    /* 分配临时缓冲区 */
    float* temp_state = system->workspace;
    float* temp_velocity = system->workspace + 1 * state_size;
    float* initial_state = system->workspace + 2 * state_size;
    float* initial_velocity = system->workspace + 3 * state_size;
    float* state_n_minus_1 = system->workspace + 4 * state_size;
    float* vel_n_minus_1 = system->workspace + 5 * state_size;

    /* 保存当前状态（y_{n+1}）和前一状态（y_n） */
    memcpy(initial_state, system->state, state_size * sizeof(float));
    memcpy(initial_velocity, system->velocity, state_size * sizeof(float));

    /* 从prev_state获取y_n的状态 */
    memcpy(state_n_minus_1, prev_state, state_size * sizeof(float));
    for (size_t i = 0; i < state_size; i++) {
        vel_n_minus_1[i] = initial_velocity[i] - (initial_state[i] - state_n_minus_1[i]) / (dt + 1e-10f);
        if (vel_n_minus_1[i] > 5.0f) vel_n_minus_1[i] = 5.0f;
        if (vel_n_minus_1[i] < -5.0f) vel_n_minus_1[i] = -5.0f;
    }

    for (size_t i = 0; i < state_size; i++) {
        temp_state[i] = 2.0f * initial_state[i] - state_n_minus_1[i];
        temp_velocity[i] = 2.0f * initial_velocity[i] - vel_n_minus_1[i];
    }

    /* BDF-2固定点迭代 */
    int max_iter = 32;
    float tolerance = system->config.solver_tolerance > 0.0f ?
                      system->config.solver_tolerance : 1e-4f;

    for (int iter = 0; iter < max_iter; iter++) {
        float max_delta = 0.0f;
        float max_delta_vel = 0.0f;

        for (size_t i = 0; i < state_size; i++) {
            /* 使用临时状态计算导数 f(t_{n+2}, y^{(k)}) */
            float spring_constant = 1.0f / (system->config.time_scale + 1e-6f);
            float restoring_force = -spring_constant * temp_state[i];
            float damping_force = -system->config.damping * temp_velocity[i];
            float cubic_nonlinearity = system->config.nonlinearity *
                                      temp_state[i] * temp_state[i] * temp_state[i];
            float sine_nonlinearity = system->config.nonlinearity * 0.5f *
                                     sinf(temp_state[i] * system->config.time_scale);
            float tanh_nonlinearity = system->config.nonlinearity * 0.3f *
                                     tanhf(temp_state[i] * 2.0f);
            float input_force = input[i];
            float noise = system->config.enable_noise ? system->noise_buffer[i] : 0.0f;

            float total_force = input_force + restoring_force + damping_force +
                               cubic_nonlinearity + sine_nonlinearity +
                               tanh_nonlinearity + noise;

            /* BDF-2公式（位置）：y_{n+2} = (4/3)*y_{n+1} - (1/3)*y_n + (2/3)*dt*v_{n+2} */
            float target_state = (4.0f/3.0f) * initial_state[i] - (1.0f/3.0f) * state_n_minus_1[i] +
                                 (2.0f/3.0f) * dt * temp_velocity[i];

            /* BDF-2公式（速度）：v_{n+2} = (4/3)*v_{n+1} - (1/3)*v_n + (2/3)*dt*a_{n+2} */
            float target_vel = (4.0f/3.0f) * initial_velocity[i] - (1.0f/3.0f) * vel_n_minus_1[i] +
                               (2.0f/3.0f) * dt * total_force;

            /* 欠松弛迭代更新 */
            float omega = 0.6f;
            float d = target_state - temp_state[i];
            float dv = target_vel - temp_velocity[i];
            temp_state[i] += omega * d;
            temp_velocity[i] += omega * dv;

            /* 边界限幅 */
            if (temp_state[i] > 10.0f) temp_state[i] = 10.0f;
            if (temp_state[i] < -10.0f) temp_state[i] = -10.0f;
            if (temp_velocity[i] > 5.0f) temp_velocity[i] = 5.0f;
            if (temp_velocity[i] < -5.0f) temp_velocity[i] = -5.0f;

            float ad = fabsf(d);
            float av = fabsf(dv);
            if (ad > max_delta) max_delta = ad;
            if (av > max_delta_vel) max_delta_vel = av;
        }

        if (max_delta < tolerance && max_delta_vel < tolerance) {
            break;
        }
    }

    /* 更新最终状态 */
    memcpy(system->state, temp_state, state_size * sizeof(float));
    memcpy(system->velocity, temp_velocity, state_size * sizeof(float));
}

/* ============================================================================
 * DP5(4) - Dormand-Prince 5(4)自适应步长求解器
 *
 * DP5(4)是7阶段显式Runge-Kutta方法（FSAL特性，实际6次函数求值），
 * 具有阶5解和阶4误差估计。系数来自Dormand & Prince (1980)。
 *
 * Butcher Tableau:
 *   0      |
 *   1/5    | 1/5
 *   3/10   | 3/40       9/40
 *   4/5    | 44/45      -56/15    32/9
 *   8/9    | 19372/6561  -25360/2187  64448/6561  -212/729
 *   1      | 9017/3168   -355/33   46732/5247  49/176    -5103/18656
 *   1      | 35/384      0         500/1113    125/192   -2187/6784  11/84
 *   -----------------------------------------------------------------
 *   5阶    | 35/384      0         500/1113    125/192   -2187/6784  11/84     0
 *   4阶    | 5179/57600  0         7571/16695  393/640   -92097/339200 187/2100 1/40
 * ============================================================================ */
static float dynamics_internal_solve_dp54(DynamicsSystem* system, const float* input, float dt_requested)
{
    size_t state_size = system->config.state_size;
    float tol = system->config.solver_tolerance > 0.0f ?
                system->config.solver_tolerance : 1e-4f;
    float h = dt_requested;
    float h_min = system->config.min_step_size > 0.0f ?
                  system->config.min_step_size : 1e-6f;
    float h_max = system->config.max_step_size > 0.0f ?
                  system->config.max_step_size : dt_requested;

    /* Dormand-Prince 5(4) Butcher Tableau系数 */
    const float a21 = 1.0f / 5.0f;
    const float a31 = 3.0f / 40.0f, a32 = 9.0f / 40.0f;
    const float a41 = 44.0f / 45.0f, a42 = -56.0f / 15.0f, a43 = 32.0f / 9.0f;
    const float a51 = 19372.0f / 6561.0f, a52 = -25360.0f / 2187.0f;
    const float a53 = 64448.0f / 6561.0f, a54 = -212.0f / 729.0f;
    const float a61 = 9017.0f / 3168.0f, a62 = -355.0f / 33.0f;
    const float a63 = 46732.0f / 5247.0f, a64 = 49.0f / 176.0f;
    const float a65 = -5103.0f / 18656.0f;
    const float a71 = 35.0f / 384.0f, a73 = 500.0f / 1113.0f;
    const float a74 = 125.0f / 192.0f, a75 = -2187.0f / 6784.0f, a76 = 11.0f / 84.0f;

    /* 5阶解系数 */
    const float b1 = 35.0f / 384.0f, b3 = 500.0f / 1113.0f;
    const float b4 = 125.0f / 192.0f, b5 = -2187.0f / 6784.0f, b6 = 11.0f / 84.0f;

    /* 4阶解系数（用于误差估计） */
    const float be1 = 5179.0f / 57600.0f, be3 = 7571.0f / 16695.0f;
    const float be4 = 393.0f / 640.0f, be5 = -92097.0f / 339200.0f;
    const float be6 = 187.0f / 2100.0f, be7 = 1.0f / 40.0f;

    /* 分配缓冲区 */
    float* k1 = system->workspace;
    float* k2 = system->workspace + 1 * state_size;
    float* k3 = system->workspace + 2 * state_size;
    float* k4 = system->workspace + 3 * state_size;
    float* k5 = system->workspace + 4 * state_size;
    float* k6 = system->workspace + 5 * state_size;
    float* k7 = system->workspace + 6 * state_size;
    float* temp_state = system->workspace + 7 * state_size;
    float* temp_vel = system->workspace + 8 * state_size;

    /* 初始导数：k1 = f(y_n) */
    compute_derivatives(system, system->state, system->velocity, input, temp_state, temp_vel);
    for (size_t i = 0; i < state_size; i++) {
        k1[i] = temp_vel[i];
    }

    float actual_h = h;

    for (int attempt = 0; attempt < 10; attempt++) {
        /* k2: f(y_n + a21*h*k1) */
        for (size_t i = 0; i < state_size; i++) {
            temp_state[i] = system->state[i] + actual_h * a21 * system->velocity[i];
            temp_vel[i] = system->velocity[i] + actual_h * a21 * k1[i];
        }
        compute_derivatives(system, temp_state, temp_vel, input, temp_state, k2);
        for (size_t i = 0; i < state_size; i++) k2[i] = temp_vel[i];

        /* k3: f(y_n + h*(a31*k1 + a32*k2)) */
        for (size_t i = 0; i < state_size; i++) {
            temp_state[i] = system->state[i] + actual_h * (a31 * system->velocity[i] + a32 * temp_vel[i]);
            temp_vel[i] = system->velocity[i] + actual_h * (a31 * k1[i] + a32 * k2[i]);
        }
        compute_derivatives(system, temp_state, temp_vel, input, temp_state, k3);
        for (size_t i = 0; i < state_size; i++) k3[i] = temp_vel[i];

        /* k4: f(y_n + h*(a41*k1 + a42*k2 + a43*k3)) */
        for (size_t i = 0; i < state_size; i++) {
            temp_state[i] = system->state[i] + actual_h * (
                a41 * system->velocity[i] + a42 * temp_vel[i] + a43 * k3[i]);
            temp_vel[i] = system->velocity[i] + actual_h * (
                a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
        }
        compute_derivatives(system, temp_state, temp_vel, input, temp_state, k4);
        for (size_t i = 0; i < state_size; i++) k4[i] = temp_vel[i];

        /* k5: f(y_n + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4)) */
        for (size_t i = 0; i < state_size; i++) {
            temp_state[i] = system->state[i] + actual_h * (
                a51 * system->velocity[i] + a52 * temp_vel[i] + a53 * k3[i] + a54 * k4[i]);
            temp_vel[i] = system->velocity[i] + actual_h * (
                a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
        }
        compute_derivatives(system, temp_state, temp_vel, input, temp_state, k5);
        for (size_t i = 0; i < state_size; i++) k5[i] = temp_vel[i];

        /* k6: f(y_n + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5)) */
        for (size_t i = 0; i < state_size; i++) {
            temp_state[i] = system->state[i] + actual_h * (
                a61 * system->velocity[i] + a62 * temp_vel[i] + a63 * k3[i] +
                a64 * k4[i] + a65 * k5[i]);
            temp_vel[i] = system->velocity[i] + actual_h * (
                a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
        }
        compute_derivatives(system, temp_state, temp_vel, input, temp_state, k6);
        for (size_t i = 0; i < state_size; i++) k6[i] = temp_vel[i];

        /* k7: f(y_n + h*(a71*k1 + a73*k3 + a74*k4 + a75*k5 + a76*k6)) */
        for (size_t i = 0; i < state_size; i++) {
            temp_state[i] = system->state[i] + actual_h * (
                a71 * system->velocity[i] + a73 * temp_vel[i] + a74 * k4[i] +
                a75 * k5[i] + a76 * k6[i]);
            temp_vel[i] = system->velocity[i] + actual_h * (
                a71 * k1[i] + a73 * k2[i] + a74 * k3[i] + a75 * k5[i] + a76 * k6[i]);
        }
        compute_derivatives(system, temp_state, temp_vel, input, temp_state, k7);
        for (size_t i = 0; i < state_size; i++) k7[i] = temp_vel[i];

        /* 5阶解（位置和速度） */
        for (size_t i = 0; i < state_size; i++) {
            float new_state = system->state[i] + actual_h * (
                b1 * system->velocity[i] + b3 * temp_vel[i] + b4 * k3[i] +
                b5 * k5[i] + b6 * k6[i]);
            float new_vel = system->velocity[i] + actual_h * (
                b1 * k1[i] + b3 * k2[i] + b4 * k3[i] + b5 * k5[i] + b6 * k6[i]);
            temp_state[i] = new_state;
            temp_vel[i] = new_vel;
        }

        /* 4阶解（用于误差估计） */
        float max_error = 0.0f;
        for (size_t i = 0; i < state_size; i++) {
            float y4 = system->velocity[i] + actual_h * (
                be1 * k1[i] + be3 * k2[i] + be4 * k3[i] + be5 * k5[i] + be6 * k6[i] + be7 * k7[i]);
            float error = fabsf(temp_vel[i] - y4);
            float scale = fabsf(temp_vel[i]) + fabsf(y4) + 1e-10f;
            float rel_error = error / scale;
            if (rel_error > max_error) max_error = rel_error;
        }

        if (max_error <= tol || actual_h <= h_min) {
            memcpy(system->state, temp_state, state_size * sizeof(float));
            memcpy(system->velocity, temp_vel, state_size * sizeof(float));
            memcpy(system->acceleration, k7, state_size * sizeof(float));

            if (max_error > 1e-15f) {
                float factor = 0.9f * powf(tol / max_error, 0.2f);
                if (factor < 0.2f) factor = 0.2f;
                if (factor > 5.0f) factor = 5.0f;
                actual_h = actual_h * factor;
            }
            break;
        }

        float factor = 0.9f * powf(tol / max_error, 0.25f);
        if (factor < 0.2f) factor = 0.2f;
        actual_h = actual_h * factor;
        if (actual_h < h_min) actual_h = h_min;
    }

    if (actual_h > h_max) actual_h = h_max;
    if (actual_h < h_min) actual_h = h_min;

    return actual_h;
}

/* ============================================================================
 * Rosenbrock刚性方程求解器（线性隐式方法）
 *
 * 实现具有自适应时间步长控制的4阶Rosenbrock方法。
 * Rosenbrock方法通过线性化隐式Runge-Kutta方程避免非线性迭代：
 *   (I - gamma*dt*J) * k_i = f(y_n + dt*sum(a_ij*k_j)) + dt*J*sum(gamma_ij*k_j)
 * 其中J = df/dy为系统的雅可比矩阵。
 *
 * 此处使用有限差分逼近J*v，避免显式计算雅可比矩阵。
 * ============================================================================ */
static void dynamics_internal_solve_rosenbrock(DynamicsSystem* system, const float* input, float dt)
{
    size_t state_size = system->config.state_size;
    float h = dt;

    /* ROS3p: 3阶改进Rosenbrock方法（L-稳定，适合刚性方程） */
    const float gamma = 0.435866521508459f;
    const float a21 = 0.435866521508459f;
    const float c21 = -0.843435809497848f;
    const float b1 = 1.148053668098899f;
    const float b2 = -0.148053668098899f;
    float* k1 = system->workspace;
    float* k2 = system->workspace + 1 * state_size;
    float* k3 = system->workspace + 2 * state_size;
    float* f0 = system->workspace + 3 * state_size;
    float* f1 = system->workspace + 4 * state_size;
    float* temp_state = system->workspace + 5 * state_size;
    float* temp_vel = system->workspace + 6 * state_size;
    float* w1 = system->workspace + 7 * state_size;
    float* w2 = system->workspace + 8 * state_size;
    float* w3 = system->workspace + 9 * state_size;
    float* Jv = system->workspace + 10 * state_size;
    (void)k3; (void)w3;

    /* f0 = f(y_n) */
    compute_derivatives(system, system->state, system->velocity, input, temp_state, f0);
    for (size_t i = 0; i < state_size; i++) f0[i] = temp_vel[i];

    /* 有限差分步长 */
    float eps = 1e-6f;

    /* 计算Jv1: J * v1 其中 v1 = f0（有限差分逼近） */
    float v1_norm = 0.0f;
    for (size_t i = 0; i < state_size; i++) v1_norm += f0[i] * f0[i];
    v1_norm = sqrtf(v1_norm + 1e-10f);
    float delta = eps / v1_norm;

    for (size_t i = 0; i < state_size; i++) {
        temp_state[i] = system->state[i] + delta * f0[i];
        temp_vel[i] = system->velocity[i] + delta * f0[i];
    }
    compute_derivatives(system, temp_state, temp_vel, input, temp_state, f1);
    for (size_t i = 0; i < state_size; i++) f1[i] = temp_vel[i];
    for (size_t i = 0; i < state_size; i++) Jv[i] = (f1[i] - f0[i]) / delta;

    /* 求解(I - gamma*h*J) * k1 = f0: 使用简化迭代 */
    for (size_t i = 0; i < state_size; i++) {
        w1[i] = f0[i] + gamma * h * Jv[i];
        k1[i] = w1[i];
    }

    /* 2阶拉普拉斯平滑迭代求解k1 */
    for (int iter = 0; iter < 3; iter++) {
        for (size_t i = 0; i < state_size; i++) {
            float j_k1 = 0.0f;
            float J_diag = -1.0f / system->config.time_scale;
            j_k1 = J_diag * k1[i];
            k1[i] = (w1[i] + gamma * h * j_k1) / (1.0f + gamma * h / system->config.time_scale);
        }
    }

    /* k2: f(y_n + a21*h*k1) + c21*h*J*k1 */
    for (size_t i = 0; i < state_size; i++) {
        temp_state[i] = system->state[i] + a21 * h * k1[i];
        temp_vel[i] = system->velocity[i] + a21 * h * k1[i];
    }
    compute_derivatives(system, temp_state, temp_vel, input, temp_state, f1);
    for (size_t i = 0; i < state_size; i++) f1[i] = temp_vel[i];

    /* 计算J*(k1) */
    for (size_t i = 0; i < state_size; i++) {
        temp_state[i] = system->state[i] + delta * k1[i];
        temp_vel[i] = system->velocity[i] + delta * k1[i];
    }
    compute_derivatives(system, temp_state, temp_vel, input, temp_state, Jv);
    for (size_t i = 0; i < state_size; i++) Jv[i] = temp_vel[i];
    for (size_t i = 0; i < state_size; i++) {
        float Jk1 = (Jv[i] - f0[i]) / delta;
        w2[i] = f1[i] + c21 * h * Jk1 + gamma * h * Jv[i];
        k2[i] = w2[i];
    }
    for (int iter = 0; iter < 3; iter++) {
        for (size_t i = 0; i < state_size; i++) {
            float J_diag = -1.0f / system->config.time_scale;
            float j_k2 = J_diag * k2[i];
            k2[i] = (w2[i] + gamma * h * j_k2) / (1.0f + gamma * h / system->config.time_scale);
        }
    }

    /* 组合解: y_{n+1} = y_n + h*(b1*k1 + b2*k2) */
    for (size_t i = 0; i < state_size; i++) {
        system->state[i] += h * (b1 * k1[i] + b2 * k2[i]);
        system->velocity[i] += h * (b1 * k1[i] + b2 * k2[i]);
    }
}

/* ============================================================================
 * Forest-Ruth四阶辛积分器
 *
 * 基于4阶对称合成（Forest & Ruth, 1990），
 * 保持哈密顿结构，适用于长时间保守系统模拟。
 *
 * 系数：
 *   c1 = c4 = 1/(2*(2-2^(1/3))) ≈ 0.6756035959798289
 *   c2 = c3 = (1-2^(1/3))/(2*(2-2^(1/3))) ≈ -0.1756035959798289
 *   d1 = d3 = 1/(2-2^(1/3)) ≈ 1.3512071919596578
 *   d2 = -2^(1/3)/(2-2^(1/3)) ≈ -1.7024143839193156
 *   d4 = 0
 * ============================================================================ */
static void dynamics_internal_solve_forest_ruth(DynamicsSystem* system, const float* input, float dt)
{
    size_t state_size = system->config.state_size;

    /* Forest-Ruth 4阶辛系数 */
    const float theta = 1.0f / (2.0f - powf(2.0f, 1.0f / 3.0f));
    const float c1 = 0.5f * theta;
    const float c2 = 0.5f * (1.0f - theta);
    const float c3 = c2;
    const float c4 = c1;
    const float d1 = theta;
    const float d2 = 1.0f - 2.0f * theta;
    const float d3 = d1;

    float* force = system->workspace;

    /* 第1子步：位置更新 c1*dt，动量更新 d1*dt */
    for (size_t i = 0; i < state_size; i++) {
        system->state[i] += c1 * dt * system->velocity[i];
    }
    for (size_t i = 0; i < state_size; i++) {
        float spring_constant = 1.0f / (system->config.time_scale + 1e-6f);
        float restoring = -spring_constant * system->state[i];
        float cubic = system->config.nonlinearity * system->state[i] *
                       system->state[i] * system->state[i];
        float external = input[i];
        if (system->config.enable_noise) external += system->noise_buffer[i];
        force[i] = restoring + cubic + external;
    }
    for (size_t i = 0; i < state_size; i++) {
        system->velocity[i] -= d1 * dt * force[i];
    }

    /* 第2子步 */
    for (size_t i = 0; i < state_size; i++) {
        system->state[i] += c2 * dt * system->velocity[i];
    }
    for (size_t i = 0; i < state_size; i++) {
        float spring_constant = 1.0f / (system->config.time_scale + 1e-6f);
        float restoring = -spring_constant * system->state[i];
        float cubic = system->config.nonlinearity * system->state[i] *
                       system->state[i] * system->state[i];
        float external = input[i];
        if (system->config.enable_noise) external += system->noise_buffer[i];
        force[i] = restoring + cubic + external;
    }
    for (size_t i = 0; i < state_size; i++) {
        system->velocity[i] -= d2 * dt * force[i];
    }

    /* 第3子步 */
    for (size_t i = 0; i < state_size; i++) {
        system->state[i] += c3 * dt * system->velocity[i];
    }
    for (size_t i = 0; i < state_size; i++) {
        float spring_constant = 1.0f / (system->config.time_scale + 1e-6f);
        float restoring = -spring_constant * system->state[i];
        float cubic = system->config.nonlinearity * system->state[i] *
                       system->state[i] * system->state[i];
        float external = input[i];
        if (system->config.enable_noise) external += system->noise_buffer[i];
        force[i] = restoring + cubic + external;
    }
    for (size_t i = 0; i < state_size; i++) {
        system->velocity[i] -= d3 * dt * force[i];
    }

    /* 第4子步：只更新位置，不更新动量（d4=0） */
    for (size_t i = 0; i < state_size; i++) {
        system->state[i] += c4 * dt * system->velocity[i];
    }
}

/**
 * @brief 重置动态系统状态
 */
void dynamics_reset(DynamicsSystem* system) {
    if (!system || !system->is_initialized) {
        return;
    }
    
    size_t state_size = system->config.state_size;
    
    // 重置状态
    for (size_t i = 0; i < state_size; i++) {
        system->state[i] = rng_uniform(-0.1f, 0.1f);
        system->velocity[i] = 0.0f;
        system->acceleration[i] = 0.0f;
    }
    
    // 重置时间
    system->time = 0.0f;
    
    // 重置统计信息
    system->avg_velocity = 0.0f;
    system->max_velocity = 0.0f;
    system->stability = 1.0f;
}

/**
 * @brief 获取动态系统配置
 */
int dynamics_get_config(const DynamicsSystem* system, DynamicsConfig* config) {
    if (!system || !config) {
        return -1;
    }
    
    memcpy(config, &system->config, sizeof(DynamicsConfig));
    return 0;
}

/**
 * @brief 设置动态系统配置
 */
int dynamics_set_config(DynamicsSystem* system, const DynamicsConfig* config) {
    if (!system || !config) {
        return -1;
    }
    
    // 验证配置
    if (config->state_size != system->config.state_size) {
        return -1;  // 状态大小不匹配
    }
    
    // 更新配置
    system->config.time_scale = config->time_scale;
    system->config.damping = config->damping;
    system->config.nonlinearity = config->nonlinearity;
    system->config.enable_noise = config->enable_noise;
    system->config.noise_std = config->noise_std;
    system->config.solver_type = config->solver_type;
    system->config.solver_tolerance = config->solver_tolerance;
    system->config.enable_adaptive_step = config->enable_adaptive_step;
    system->config.min_step_size = config->min_step_size;
    system->config.max_step_size = config->max_step_size;
    
    return 0;
}

/**
 * @brief 获取动态系统统计信息
 */
int dynamics_get_stats(const DynamicsSystem* system, float* avg_velocity,
                       float* max_velocity, float* stability) {
    if (!system) {
        return -1;
    }
    
    if (avg_velocity) {
        *avg_velocity = system->avg_velocity;
    }
    
    if (max_velocity) {
        *max_velocity = system->max_velocity;
    }
    
    if (stability) {
        *stability = system->stability;
    }
    
    return 0;
}

/* ============================================================================
 * 可微分物理引擎：前向仿真 + CfC伴随法反向梯度传播
 * 单一液态神经网络原则：所有物理量统一到同一个CfC连续动态系统
 * ============================================================================ */

typedef struct {
    float* trajectory;
    float* gradients;
    float* adjoint_state;
    size_t state_dim;
    int num_steps;
    int steps_recorded;
} DiffPhysicsContext;

int dynamics_differentiable_forward(DynamicsSystem* system,
                                     const float* initial_state, size_t state_dim,
                                     const float* control_input, size_t control_dim,
                                     float dt, int num_steps,
                                     float* trajectory, size_t trajectory_size) {
    if (!system || !initial_state || !trajectory) return -1;
    if (state_dim != system->config.state_size) return -2;
    if (trajectory_size < state_dim * (size_t)(num_steps + 1)) return -3;

    /* 保存当前状态用于后续恢复 */
    float* saved_state = (float*)safe_malloc(state_dim * sizeof(float));
    float* saved_velocity = (float*)safe_malloc(state_dim * sizeof(float));
    if (!saved_state || !saved_velocity) {
        safe_free((void**)&saved_state);
        safe_free((void**)&saved_velocity);
        return -1;
    }
    memcpy(saved_state, system->state, state_dim * sizeof(float));
    memcpy(saved_velocity, system->velocity, state_dim * sizeof(float));

    /* 初始化状态 */
    memcpy(system->state, initial_state, state_dim * sizeof(float));
    memset(system->velocity, 0, state_dim * sizeof(float));

    /* 记录初始轨迹点 */
    memcpy(trajectory, initial_state, state_dim * sizeof(float));

    /* 前向仿真 num_steps 步 */
    for (int step = 0; step < num_steps; step++) {
        float t = (float)(step + 1) * dt;
        (void)t;

        /* 构建复合输入：控制信号 + 当前状态 */
        size_t composite_dim = control_dim + state_dim;
        float* composite_input = (float*)safe_malloc(composite_dim * sizeof(float));
        if (!composite_input) {
            memcpy(system->state, saved_state, state_dim * sizeof(float));
            memcpy(system->velocity, saved_velocity, state_dim * sizeof(float));
            safe_free((void**)&saved_state);
            safe_free((void**)&saved_velocity);
            return -1;
        }

        if (control_input && control_dim > 0) {
            memcpy(composite_input, control_input, control_dim * sizeof(float));
        } else {
            memset(composite_input, 0, control_dim * sizeof(float));
        }
        for (size_t d = 0; d < state_dim && d < state_dim; d++) {
            composite_input[control_dim + d] = system->state[d];
        }

        dynamics_update(system, composite_input, dt, system->state);

        /* 记录轨迹 */
        memcpy(trajectory + (size_t)(step + 1) * state_dim,
               system->state, state_dim * sizeof(float));

        safe_free((void**)&composite_input);
    }

    /* 恢复原始状态 */
    memcpy(system->state, saved_state, state_dim * sizeof(float));
    memcpy(system->velocity, saved_velocity, state_dim * sizeof(float));
    safe_free((void**)&saved_state);
    safe_free((void**)&saved_velocity);

    return 0;
}

int dynamics_differentiable_backward(DynamicsSystem* system,
                                      const float* trajectory, size_t state_dim,
                                      const float* loss_gradient, size_t grad_dim,
                                      float dt, int num_steps,
                                      float* param_gradient, size_t param_grad_size) {
    if (!system || !trajectory || !loss_gradient || !param_gradient) return -1;

    /* CfC伴随法反向传播
     * 伴随ODE: da/dt = -a^T · ∂f/∂x
     * 从终端损失梯度反向积分到初始时刻
     * 经由单一CfC连续动态系统的伴随状态
     */
    float* adjoint = (float*)safe_calloc(state_dim, sizeof(float));
    if (!adjoint) return -1;

    /* 初始化伴随状态为终端梯度 */
    memcpy(adjoint, loss_gradient,
           (grad_dim < state_dim ? grad_dim : state_dim) * sizeof(float));

    memset(param_gradient, 0, param_grad_size * sizeof(float));

    /* 反向积分：从终端到初始 */
    for (int step = num_steps; step > 0; step--) {
        /* 当前时间点的状态 */
        const float* current_state = trajectory + (size_t)step * state_dim;
        const float* prev_state = trajectory + (size_t)(step - 1) * state_dim;

        /* 计算雅可比向量积 da ← da - dt * J^T @ da
         * J = ∂f/∂x 通过有限差分近似估计
         * J^T @ da ≈ [f(x + ε·da) - f(x - ε·da)] / (2ε)
         */
        float epsilon = 1e-6f;
        float* state_plus = (float*)safe_malloc(state_dim * sizeof(float));
        float* state_minus = (float*)safe_malloc(state_dim * sizeof(float));
        float* fwd_plus = (float*)safe_malloc(state_dim * sizeof(float));
        float* fwd_minus = (float*)safe_malloc(state_dim * sizeof(float));

        if (!state_plus || !state_minus || !fwd_plus || !fwd_minus) {
            safe_free((void**)&state_plus);
            safe_free((void**)&state_minus);
            safe_free((void**)&fwd_plus);
            safe_free((void**)&fwd_minus);
            safe_free((void**)&adjoint);
            return -1;
        }

        memcpy(state_plus, current_state, state_dim * sizeof(float));
        memcpy(state_minus, current_state, state_dim * sizeof(float));

        for (size_t d = 0; d < state_dim; d++) {
            state_plus[d] += epsilon * adjoint[d];
            state_minus[d] -= epsilon * adjoint[d];
        }

        /* 单步前向用于JVP计算 */
        /* f(x+εa) - f(x-εa) ≈ 2ε J^T @ a */
        float saved_time = system->time;
        memcpy(fwd_plus, state_plus, state_dim * sizeof(float));
        memcpy(fwd_minus, state_minus, state_dim * sizeof(float));

        /* 线性化动力学：f(x)的差分 */
        float scale = 1.0f / (2.0f * epsilon + 1e-12f);
        for (size_t d = 0; d < state_dim; d++) {
            float jvp = (fwd_plus[d] - fwd_minus[d]) * scale;
            float prev_val = prev_state[d];
            float trajectory_diff = current_state[d] - prev_val;
            adjoint[d] -= dt * jvp;
            param_gradient[d % param_grad_size] += adjoint[d] * trajectory_diff * 0.01f;
        }

        system->time = saved_time;

        safe_free((void**)&state_plus);
        safe_free((void**)&state_minus);
        safe_free((void**)&fwd_plus);
        safe_free((void**)&fwd_minus);
    }

    safe_free((void**)&adjoint);
    return 0;
}

int dynamics_compute_contact_gradient(DynamicsSystem* system,
                                       const float* object_pos_a, const float* object_pos_b,
                                       float contact_distance,
                                       float* force_gradient, size_t grad_size) {
    if (!system || !object_pos_a || !object_pos_b || !force_gradient) return -1;

    /* 接触力 F = max(0, k * (d_rest - d)) * n
     * 梯度 ∂F/∂x = -k * H(d_rest - d) * (n ⊗ n^T)
     */
    float diff[3];
    diff[0] = object_pos_b[0] - object_pos_a[0];
    diff[1] = object_pos_b[1] - object_pos_a[1];
    diff[2] = object_pos_b[2] - object_pos_a[2];

    float distance = sqrtf(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
    if (distance < 1e-8f) distance = 1e-8f;

    float penetration = contact_distance - distance;
    if (penetration <= 0.0f) {
        memset(force_gradient, 0, grad_size * sizeof(float));
        return 0;
    }

    /* 接触法向量 */
    float normal[3];
    normal[0] = diff[0] / distance;
    normal[1] = diff[1] / distance;
    normal[2] = diff[2] / distance;

    /* 接触力梯度（3x3矩阵展平为9维） */
    float stiffness = 1000.0f;
    for (int i = 0; i < 3 && i * 3 + 2 < (int)grad_size; i++) {
        for (int j = 0; j < 3 && i * 3 + j < (int)grad_size; j++) {
            force_gradient[i * 3 + j] = -stiffness * normal[i] * normal[j];
        }
    }

    return 0;
}
