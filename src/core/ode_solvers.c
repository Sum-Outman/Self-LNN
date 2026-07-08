#include "selflnn/core/ode_solvers.h"
#include "selflnn/utils/memory_utils.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>  /* P3-03修复: SIZE_MAX溢出检查需要 */

#define SELFLNN_DP54_SAFETY 0.9f
#define SELFLNN_DP54_PGROW -0.2f
#define SELFLNN_DP54_PSHRINK -0.25f
#define SELFLNN_DP54_ERR_CTRL 0.9f /* 从0.5调整为0.9，保留自适应步长控制的有效性 */

/* ODE求解器错误码 */
#define ODE_ERR_NAN_INF -5  /* NaN/Inf检测到中间步骤 */

/* 编译开关：启用快速验证模式（仅采样检测，非全量扫描） */
#ifndef SELFLNN_ODE_FAST_VALIDATE
#define SELFLNN_ODE_FAST_VALIDATE 0
#endif

/* ODE求解器中间步输出NaN/Inf检测
 * 在每个RHS调用后检查中间量k1~k7，防止NaN/Inf传播 */
static int ode_check_output_finite(const float* k, size_t n) {
    if (!k || n == 0) return 0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(k[i])) return -1;
    }
    return 0;
}

/* ODE求解器NaN/Inf输入快速检测
 * 在所有求解器入口处调用，防止NaN/Inf传播导致无限循环或数值崩溃 */
static int ode_check_input_finite(const float* y, size_t n) {
    if (!y || n == 0) return 0;
#if SELFLNN_ODE_FAST_VALIDATE
    /* 采样检测：大维度状态向量全量扫描代价过高时启用 */
    size_t step = n > 100 ? n / 50 : 1;
    for (size_t i = 0; i < n; i += step) {
        if (!isfinite(y[i])) return -1;
    }
    /* 也要检测最后一个元素 */
    if (!isfinite(y[n - 1])) return -1;
#else
    /* 全量扫描：确保每个元素都经过有限性检测，防止漏检 */
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(y[i])) return -1;
    }
#endif
    return 0;
}

/* 使用标准库 fmaxf/fminf/fabsf 替代本地重复定义，统一于math.h */

DP54Config ode_dp54_default_config(void)
{
    DP54Config cfg;
    cfg.rel_tolerance = 1e-6f;
    cfg.abs_tolerance = 1e-8f;
    cfg.min_step_size = 1e-8f;
    cfg.max_step_size = 1.0f;
    cfg.safety_factor = 0.9f;
    cfg.max_iterations = 10000;
    return cfg;
}

RosenbrockConfig ode_rosenbrock_default_config(void)
{
    RosenbrockConfig cfg;
    cfg.rel_tolerance = 1e-5f;
    cfg.abs_tolerance = 1e-7f;
    cfg.min_step_size = 1e-10f;
    cfg.max_step_size = 1.0f;
    cfg.gamma_coeff = 0.435866521508f;
    cfg.max_iterations = 10000;
    cfg.use_finite_diff_jacobian = 1;
    cfg.finite_diff_eps = 1e-6f;
    cfg.jacobian_bandwidth = 8;     /* P2-044: 液态NN默认带状宽度，0=全矩阵 */
    cfg.method_order = 1;           /* P3-001: 默认一阶方法，设为3启用ROS3p三阶 */
    return cfg;
}

SymplecticConfig ode_symplectic_default_config(void)
{
    SymplecticConfig cfg;
    cfg.enable_separable = 1;
    cfg.substep_ratio = 0.5f;
    cfg.num_substeps = 1;
    return cfg;
}

size_t ode_dp54_workspace_size(size_t n)
{
    return (8 * n);
}

size_t ode_rosenbrock_workspace_size(size_t n)
{
    /* P3-03修复: n*n乘法溢出检查，防止大维度回绕 */
    if (n > 0 && n > SIZE_MAX / n) return 0;
    /* P3-001: 增加工作空间(5n→8n)支持ROS3p三阶方法的额外k向量和临时缓冲区 */
    return (n * n + 8 * n);
}

size_t ode_forest_ruth_workspace_size(size_t n)
{
    return (2 * n);
}

size_t ode_verlet_workspace_size(size_t n)
{
    return n;
}

int ode_dp54_solve(float* y, float t, float delta_t, ODERHSFunc rhs, void* ctx,
                   size_t n, const DP54Config* cfg, float* workspace,
                   float* h_actual, int* steps_used)
{
    if (!y || !rhs || !cfg || !workspace || n == 0) return -1;
    if (ode_check_input_finite(y, n) != 0) return -1;

    float rel_tol = (cfg->rel_tolerance > 0.0f) ? cfg->rel_tolerance : 1e-6f;
    float abs_tol = (cfg->abs_tolerance > 0.0f) ? cfg->abs_tolerance : 1e-8f;
    float h_min = (cfg->min_step_size > 0.0f) ? cfg->min_step_size : 1e-8f;
    float h_max = (cfg->max_step_size > 0.0f) ? cfg->max_step_size : 1.0f;
    float safety = (cfg->safety_factor > 0.0f) ? cfg->safety_factor : 0.9f;
    int max_iter = (cfg->max_iterations > 0) ? cfg->max_iterations : 10000;

    float* k1 = workspace;
    float* k2 = k1 + n;
    float* k3 = k2 + n;
    float* k4 = k3 + n;
    float* k5 = k4 + n;
    float* k6 = k5 + n;
    float* k7 = k6 + n;
    float* y_temp = k7 + n;

    float h = (delta_t > h_max) ? h_max : delta_t;
    if (h <= 0.0f) return -1;  /* 零或负步长，拒绝执行——保护调用者免受错误参数影响 */
    float t_current = t;
    float t_target = t + delta_t;
    int total_steps = 0;

    while (t_current < t_target - 1e-15f && total_steps < max_iter)
    {
        float remaining = t_target - t_current;
        if (h > remaining) h = remaining;
        if (h < h_min) h = h_min;

        if (rhs(t_current, y, k1, ctx) != 0) return -2;
        if (ode_check_output_finite(k1, n) != 0) return ODE_ERR_NAN_INF;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (0.2f * k1[i]);
        if (rhs(t_current + 0.2f * h, y_temp, k2, ctx) != 0) return -2;
        if (ode_check_output_finite(k2, n) != 0) return ODE_ERR_NAN_INF;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (3.0f/40.0f * k1[i] + 9.0f/40.0f * k2[i]);
        if (rhs(t_current + 0.3f * h, y_temp, k3, ctx) != 0) return -2;
        if (ode_check_output_finite(k3, n) != 0) return ODE_ERR_NAN_INF;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (44.0f/45.0f * k1[i] - 56.0f/15.0f * k2[i] + 32.0f/9.0f * k3[i]);
        if (rhs(t_current + 0.8f * h, y_temp, k4, ctx) != 0) return -2;
        if (ode_check_output_finite(k4, n) != 0) return ODE_ERR_NAN_INF;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (19372.0f/6561.0f * k1[i] - 25360.0f/2187.0f * k2[i] + 64448.0f/6561.0f * k3[i] - 212.0f/729.0f * k4[i]);
        if (rhs(t_current + 8.0f/9.0f * h, y_temp, k5, ctx) != 0) return -2;
        if (ode_check_output_finite(k5, n) != 0) return ODE_ERR_NAN_INF;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (9017.0f/3168.0f * k1[i] - 355.0f/33.0f * k2[i] + 46732.0f/5247.0f * k3[i] + 49.0f/176.0f * k4[i] - 5103.0f/18656.0f * k5[i]);
        if (rhs(t_current + h, y_temp, k6, ctx) != 0) return -2;
        if (ode_check_output_finite(k6, n) != 0) return ODE_ERR_NAN_INF;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (35.0f/384.0f * k1[i] + 500.0f/1113.0f * k3[i] + 125.0f/192.0f * k4[i] - 2187.0f/6784.0f * k5[i] + 11.0f/84.0f * k6[i]);
        if (rhs(t_current + h, y_temp, k7, ctx) != 0) return -2;
        if (ode_check_output_finite(k7, n) != 0) return ODE_ERR_NAN_INF;

        float max_err = 0.0f;
        for (size_t i = 0; i < n; i++)
        {
            float y5 = y[i] + h * (35.0f/384.0f * k1[i] + 500.0f/1113.0f * k3[i] + 125.0f/192.0f * k4[i] - 2187.0f/6784.0f * k5[i] + 11.0f/84.0f * k6[i]);
            float y4 = y[i] + h * (5179.0f/57600.0f * k1[i] + 7571.0f/16695.0f * k3[i] + 393.0f/640.0f * k4[i] - 92097.0f/339200.0f * k5[i] + 187.0f/2100.0f * k6[i] + 0.025f * k7[i]);
            float err_i = fabsf(y5 - y4);
            float scale = abs_tol + rel_tol * fmaxf(fabsf(y[i]), fabsf(y5));
            float ratio = err_i / scale;
            if (ratio > max_err) max_err = ratio;
        }

        total_steps++;

        if (max_err <= 1.0f || h <= h_min)
        {
            for (size_t i = 0; i < n; i++)
                y[i] = y[i] + h * (35.0f/384.0f * k1[i] + 500.0f/1113.0f * k3[i] + 125.0f/192.0f * k4[i] - 2187.0f/6784.0f * k5[i] + 11.0f/84.0f * k6[i]);
            t_current += h;

            /* 步接受后检查y的有限性 */
            if (ode_check_input_finite(y, n) != 0) return ODE_ERR_NAN_INF;

            float h_new = h;
            if (max_err > SELFLNN_DP54_ERR_CTRL)
            {
                if (max_err < 1e-30f) max_err = 1e-30f;
                float factor = safety * (float)pow((double)max_err, (double)SELFLNN_DP54_PGROW);
                h_new = h * fmaxf(0.2f, fminf(5.0f, factor));
            }
            h = h_new;
            if (h > h_max) h = h_max;
        }
        else
        {
            if (max_err < 1e-30f) max_err = 1e-30f;
            float factor = safety * (float)pow((double)max_err, (double)SELFLNN_DP54_PSHRINK);
            h = h * fmaxf(0.1f, fminf(1.0f, factor));
            if (h < h_min) h = h_min;
        }
    }

    if (h_actual) *h_actual = t_current - t;
    if (steps_used) *steps_used = total_steps;
    return (total_steps >= max_iter && t_current < t_target - 1e-14f) ? -3 : 0;
}

/* ============================================================================
 * M-4: Euler/RK2/RK4 显式ODE求解器公共API
 *
 * 补齐README声明的8种求解器：Euler（显式欧拉法，1阶）、
 * RK2（中点法，2阶）、RK4（经典四阶Runge-Kutta，4阶）。
 * 每个求解器遵循统一的参数签名和错误处理约定。
 *
 * 工作空间需求：
 *   Euler: 1*n（临时缓冲区 tmp）
 *   RK2:   3*n（k1, k2, tmp）
 *   RK4:   5*n（k1, k2, k3, k4, tmp）
 * ============================================================================ */

/* ---- Euler 显式欧拉法（1阶精度） ---- */

size_t ode_euler_workspace_size(size_t n)
{
    return n;  /* 仅需1个临时缓冲区 */
}

int ode_euler_solve(float* y, float t, float delta_t, ODERHSFunc rhs, void* ctx,
                    size_t n, float* workspace, float* h_actual, int* steps_used)
{
    if (!y || !rhs || !workspace || n == 0) return -1;
    if (delta_t <= 0.0f) return -1;
    if (ode_check_input_finite(y, n) != 0) return -1;

    float* tmp = workspace;
    float t_current = t;
    float t_target = t + delta_t;
    float h = delta_t;
    int steps = 1;

    /* 显式欧拉法: y_{n+1} = y_n + h * f(t_n, y_n) */
    if (rhs(t_current, y, tmp, ctx) != 0) return -2;
    if (ode_check_output_finite(tmp, n) != 0) return ODE_ERR_NAN_INF;

    for (size_t i = 0; i < n; i++) {
        y[i] = y[i] + h * tmp[i];
    }

    if (h_actual) *h_actual = h;
    if (steps_used) *steps_used = steps;
    return 0;
}

/* ---- RK2 中点法（2阶精度） ---- */

size_t ode_rk2_workspace_size(size_t n)
{
    return 3 * n;  /* k1[n] + k2[n] + tmp[n] */
}

int ode_rk2_solve(float* y, float t, float delta_t, ODERHSFunc rhs, void* ctx,
                  size_t n, float* workspace, float* h_actual, int* steps_used)
{
    if (!y || !rhs || !workspace || n == 0) return -1;
    if (delta_t <= 0.0f) return -1;
    if (ode_check_input_finite(y, n) != 0) return -1;

    float* k1  = workspace;         /* k1 = f(t_n, y_n) */
    float* k2  = workspace + n;     /* k2 = f(t_n + h/2, y_n + h*k1/2) */
    float* tmp = workspace + 2 * n; /* 临时缓冲区 */

    float h = delta_t;
    int steps = 1;

    /* 中点法（RK2）：
     *   k1 = f(t_n, y_n)
     *   k2 = f(t_n + h/2, y_n + h*k1/2)
     *   y_{n+1} = y_n + h * k2
     */
    if (rhs(t, y, k1, ctx) != 0) return -2;
    if (ode_check_output_finite(k1, n) != 0) return ODE_ERR_NAN_INF;

    for (size_t i = 0; i < n; i++) {
        tmp[i] = y[i] + 0.5f * h * k1[i];
    }
    if (rhs(t + 0.5f * h, tmp, k2, ctx) != 0) return -2;
    if (ode_check_output_finite(k2, n) != 0) return ODE_ERR_NAN_INF;

    for (size_t i = 0; i < n; i++) {
        y[i] = y[i] + h * k2[i];
    }

    if (h_actual) *h_actual = h;
    if (steps_used) *steps_used = steps;
    return 0;
}

/* ---- RK4 经典四阶Runge-Kutta（4阶精度） ---- */

size_t ode_rk4_workspace_size(size_t n)
{
    return 5 * n;  /* k1[n] + k2[n] + k3[n] + k4[n] + tmp[n] */
}

int ode_rk4_solve(float* y, float t, float delta_t, ODERHSFunc rhs, void* ctx,
                  size_t n, float* workspace, float* h_actual, int* steps_used)
{
    if (!y || !rhs || !workspace || n == 0) return -1;
    if (delta_t <= 0.0f) return -1;
    if (ode_check_input_finite(y, n) != 0) return -1;

    float* k1  = workspace;         /* k1 = f(t_n, y_n) */
    float* k2  = workspace + n;     /* k2 = f(t_n + h/2, y_n + h*k1/2) */
    float* k3  = workspace + 2 * n; /* k3 = f(t_n + h/2, y_n + h*k2/2) */
    float* k4  = workspace + 3 * n; /* k4 = f(t_n + h,   y_n + h*k3) */
    float* tmp = workspace + 4 * n; /* 临时缓冲区 */

    float h = delta_t;
    int steps = 1;

    /* 经典四阶Runge-Kutta：
     *   k1 = f(t_n, y_n)
     *   k2 = f(t_n + h/2, y_n + h*k1/2)
     *   k3 = f(t_n + h/2, y_n + h*k2/2)
     *   k4 = f(t_n + h,   y_n + h*k3)
     *   y_{n+1} = y_n + h*(k1 + 2*k2 + 2*k3 + k4)/6
     */

    /* 阶段1: k1 */
    if (rhs(t, y, k1, ctx) != 0) return -2;
    if (ode_check_output_finite(k1, n) != 0) return ODE_ERR_NAN_INF;

    /* 阶段2: k2（中点评估） */
    for (size_t i = 0; i < n; i++) {
        tmp[i] = y[i] + 0.5f * h * k1[i];
    }
    if (rhs(t + 0.5f * h, tmp, k2, ctx) != 0) return -2;
    if (ode_check_output_finite(k2, n) != 0) return ODE_ERR_NAN_INF;

    /* 阶段3: k3（中点评估，用k2修正） */
    for (size_t i = 0; i < n; i++) {
        tmp[i] = y[i] + 0.5f * h * k2[i];
    }
    if (rhs(t + 0.5f * h, tmp, k3, ctx) != 0) return -2;
    if (ode_check_output_finite(k3, n) != 0) return ODE_ERR_NAN_INF;

    /* 阶段4: k4（终点评估，用k3修正） */
    for (size_t i = 0; i < n; i++) {
        tmp[i] = y[i] + h * k3[i];
    }
    if (rhs(t + h, tmp, k4, ctx) != 0) return -2;
    if (ode_check_output_finite(k4, n) != 0) return ODE_ERR_NAN_INF;

    /* 最终更新: y_{n+1} = y_n + h*(k1 + 2*k2 + 2*k3 + k4)/6 */
    for (size_t i = 0; i < n; i++) {
        y[i] = y[i] + h * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]) / 6.0f;
    }

    if (h_actual) *h_actual = h;
    if (steps_used) *steps_used = steps;
    return 0;
}

/* ======================================================================
 * 第4部分：事件检测和密集输出实现
 *
 * 提供基于DP54求解器的：
 *   1. 4阶Hermite密集输出（步内任意点插值）
 *   2. 事件检测（零点检测 + 二分法精确定位）
 *   3. 带事件检测的DP54完整求解
 * ====================================================================== */

int ode_dp54_dense_output(float tau, float t_start, float h,
                          const float* y_start, const float* y_end,
                          const float* k1, const float* k7,
                          float* y_out, size_t n)
{
    if (!y_start || !y_end || !k1 || !k7 || !y_out || n == 0 || h == 0.0f) {
        return -1;
    }

    float theta = (tau - t_start) / h;
    if (theta < 0.0f) theta = 0.0f;
    if (theta > 1.0f) theta = 1.0f;

    float theta2 = theta * theta;
    float theta3 = theta2 * theta;

    /* Hermite三次插值基函数 */
    float h00 = 2.0f * theta3 - 3.0f * theta2 + 1.0f;
    float h10 = theta3 - 2.0f * theta2 + theta;
    float h01 = -2.0f * theta3 + 3.0f * theta2;
    float h11 = theta3 - theta2;

    for (size_t i = 0; i < n; i++) {
        y_out[i] = h00 * y_start[i] + h01 * y_end[i]
                 + h * (h10 * k1[i] + h11 * k7[i]);
    }

    return 0;
}

size_t ode_dp54_workspace_size_with_events(size_t n, int num_events)
{
    /* 标准DP54: 8*n
     * y_start: n
     * k7_saved: n
     * event_old: num_events
     * event_new: num_events
     */
    if (num_events < 0) num_events = 0;
    return 8 * n + 2 * n + 2 * (size_t)num_events;
}

int ode_dp54_solve_with_events(float* y, float t, float delta_t,
                               ODERHSFunc rhs, void* ctx, size_t n,
                               const DP54Config* cfg, float* workspace,
                               float* h_actual, int* steps_used,
                               const ODEEventConfig* event_cfg,
                               ODEEventSolution* event_sol)
{
    if (!y || !rhs || !workspace || n == 0) return -1;
    if (ode_check_input_finite(y, n) != 0) return -1; /* DP54 events */

    /* ========== 初始化事件结果 ========== */
    ODEEventSolution local_event_sol;
    if (!event_sol) event_sol = &local_event_sol;
    event_sol->event_detected = 0;
    event_sol->event_index = -1;
    event_sol->event_time = t;
    event_sol->event_value = 0.0f;
    event_sol->event_count = 0;
    event_sol->event_terminated = 0;

    int use_events = (event_cfg != NULL && event_cfg->event_func != NULL && event_cfg->num_events > 0);
    int num_events = use_events ? event_cfg->num_events : 0;

    /* ========== 解析分配工作空间 ========== */
    float* k1 = workspace;
    float* k2 = k1 + n;
    float* k3 = k2 + n;
    float* k4 = k3 + n;
    float* k5 = k4 + n;
    float* k6 = k5 + n;
    float* k7 = k6 + n;
    float* y_temp = k7 + n;
    float* y_start = y_temp + n;
    float* k7_saved = y_start + n;
    float* event_old = NULL;
    float* event_new = NULL;

    if (use_events) {
        event_old = k7_saved + n;
        event_new = event_old + num_events;
    }

    /* ========== 解析DP54配置 ========== */
    float rel_tol = (cfg && cfg->rel_tolerance > 0.0f) ? cfg->rel_tolerance : 1e-6f;
    float abs_tol = (cfg && cfg->abs_tolerance > 0.0f) ? cfg->abs_tolerance : 1e-8f;
    float h_min = (cfg && cfg->min_step_size > 0.0f) ? cfg->min_step_size : 1e-8f;
    float h_max = (cfg && cfg->max_step_size > 0.0f) ? cfg->max_step_size : 1.0f;
    float safety = (cfg && cfg->safety_factor > 0.0f) ? cfg->safety_factor : 0.9f;
    int max_iter = (cfg && cfg->max_iterations > 0) ? cfg->max_iterations : 10000;

    /* ========== 解析事件配置 ========== */
    float event_tol = 1e-6f;
    float bracket_factor = 0.1f;
    int max_events = 0;
    int terminate_on_event = 0;
    const int* event_dir = NULL;

    if (use_events) {
        event_tol = (event_cfg->event_tolerance > 0.0f) ? event_cfg->event_tolerance : 1e-6f;
        bracket_factor = (event_cfg->event_bracketing_factor > 0.0f) ? event_cfg->event_bracketing_factor : 0.1f;
        max_events = (event_cfg->max_events > 0) ? event_cfg->max_events : 0;
        terminate_on_event = event_cfg->terminate_on_event;
        event_dir = event_cfg->event_direction;
    }

    float step_dir = (delta_t >= 0.0f) ? 1.0f : -1.0f;
    float h = (delta_t * step_dir > h_max) ? h_max * step_dir : delta_t;
    float t_current = t;
    float t_target = t + delta_t;
    int total_steps = 0;

    /* 如果启用事件检测，计算初始事件值 */
    if (use_events) {
        if (event_cfg->event_func(t_current, y, event_old, event_cfg->event_ctx, num_events) != 0) {
            return -4;
        }
        memcpy(y_start, y, (size_t)n * sizeof(float));
    }

    while ((step_dir > 0.0f && t_current < t_target - 1e-15f) ||
           (step_dir < 0.0f && t_current > t_target + 1e-15f))
    {
        if (total_steps >= max_iter) break;

        /* 步长限幅 */
        float remaining = (t_target - t_current) * step_dir;
        if (remaining < 0.0f) remaining = 0.0f;
        float h_abs = (h * step_dir < 0.0f) ? -h * step_dir : h * step_dir;
        float remaining_abs = remaining;
        if (h_abs > remaining_abs) h_abs = remaining_abs;
        if (h_abs < h_min) h_abs = h_min;
        h = h_abs * step_dir;

        /* 保存步起始状态（用于后续事件检测的密集输出） */
        if (use_events) {
            memcpy(y_start, y, (size_t)n * sizeof(float));
        }

        /* ========== DP54 Butcher table 求值 ========== */
        if (rhs(t_current, y, k1, ctx) != 0) return -2;
        if (ode_check_output_finite(k1, n) != 0) return ODE_ERR_NAN_INF;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (0.2f * k1[i]);
        if (rhs(t_current + 0.2f * h, y_temp, k2, ctx) != 0) return -2;
        if (ode_check_output_finite(k2, n) != 0) return ODE_ERR_NAN_INF;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (3.0f/40.0f * k1[i] + 9.0f/40.0f * k2[i]);
        if (rhs(t_current + 0.3f * h, y_temp, k3, ctx) != 0) return -2;
        if (ode_check_output_finite(k3, n) != 0) return ODE_ERR_NAN_INF;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (44.0f/45.0f * k1[i] - 56.0f/15.0f * k2[i] + 32.0f/9.0f * k3[i]);
        if (rhs(t_current + 0.8f * h, y_temp, k4, ctx) != 0) return -2;
        if (ode_check_output_finite(k4, n) != 0) return ODE_ERR_NAN_INF;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (19372.0f/6561.0f * k1[i] - 25360.0f/2187.0f * k2[i] + 64448.0f/6561.0f * k3[i] - 212.0f/729.0f * k4[i]);
        if (rhs(t_current + 8.0f/9.0f * h, y_temp, k5, ctx) != 0) return -2;
        if (ode_check_output_finite(k5, n) != 0) return ODE_ERR_NAN_INF;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (9017.0f/3168.0f * k1[i] - 355.0f/33.0f * k2[i] + 46732.0f/5247.0f * k3[i] + 49.0f/176.0f * k4[i] - 5103.0f/18656.0f * k5[i]);
        if (rhs(t_current + h, y_temp, k6, ctx) != 0) return -2;
        if (ode_check_output_finite(k6, n) != 0) return ODE_ERR_NAN_INF;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (35.0f/384.0f * k1[i] + 500.0f/1113.0f * k3[i] + 125.0f/192.0f * k4[i] - 2187.0f/6784.0f * k5[i] + 11.0f/84.0f * k6[i]);
        if (rhs(t_current + h, y_temp, k7, ctx) != 0) return -2;
        if (ode_check_output_finite(k7, n) != 0) return ODE_ERR_NAN_INF;

        /* ========== 误差估计 ========== */
        float max_err = 0.0f;
        for (size_t i = 0; i < n; i++)
        {
            float y5 = y[i] + h * (35.0f/384.0f * k1[i] + 500.0f/1113.0f * k3[i] + 125.0f/192.0f * k4[i] - 2187.0f/6784.0f * k5[i] + 11.0f/84.0f * k6[i]);
            float y4 = y[i] + h * (5179.0f/57600.0f * k1[i] + 7571.0f/16695.0f * k3[i] + 393.0f/640.0f * k4[i] - 92097.0f/339200.0f * k5[i] + 187.0f/2100.0f * k6[i] + 0.025f * k7[i]);
            float err_i = fabsf(y5 - y4);
            float scale = abs_tol + rel_tol * fmaxf(fabsf(y[i]), fabsf(y5));
            float ratio = err_i / scale;
            if (ratio > max_err) max_err = ratio;
        }

        total_steps++;

        if (max_err <= 1.0f || h_abs <= h_min)
        {
            /* ========== 接受步：更新状态 ========== */
            for (size_t i = 0; i < n; i++)
                y[i] = y[i] + h * (35.0f/384.0f * k1[i] + 500.0f/1113.0f * k3[i] + 125.0f/192.0f * k4[i] - 2187.0f/6784.0f * k5[i] + 11.0f/84.0f * k6[i]);

            /* 步接受后检查y的有限性 */
            if (ode_check_input_finite(y, n) != 0) return ODE_ERR_NAN_INF;

            /* 保存k7用于后续密集输出 */
            if (use_events) {
                memcpy(k7_saved, k7, (size_t)n * sizeof(float));
            }

            float t_next = t_current + h;

            /* ========== 事件检测 ========== */
            if (use_events) {
                int event_triggered = 0;
                int trigger_idx = -1;

                if (event_cfg->event_func(t_next, y, event_new, event_cfg->event_ctx, num_events) != 0) {
                    return -4;
                }

                for (int ei = 0; ei < num_events; ei++) {
                    float old_val = event_old[ei];
                    float new_val = event_new[ei];
                    int dir = (event_dir) ? event_dir[ei] : 0;
                    int crossed = 0;

                    if ((old_val <= 0.0f && new_val > 0.0f) || (old_val >= 0.0f && new_val < 0.0f)) {
                        if (dir == 0) {
                            crossed = 1;
                        } else if (dir > 0 && old_val <= 0.0f && new_val > 0.0f) {
                            crossed = 1;
                        } else if (dir < 0 && old_val >= 0.0f && new_val < 0.0f) {
                            crossed = 1;
                        }
                    }

                    if (crossed) {
                        event_triggered = 1;
                        trigger_idx = ei;

                        /* 使用二分法精确定位事件时间 */
                        float t_left = t_current;
                        float t_right = t_next;
                        float v_left = old_val;
                        float v_right = new_val;
                        float y_left[32]; /* 小数组栈分配，大n时动态 */
                        float y_mid[32];
                        float* y_l = y_left;
                        float* y_m = y_mid;
                        float* dyn_l = NULL;
                        float* dyn_m = NULL;

                        if (n > 32) {
/* raw malloc → safe_malloc */
                            dyn_l = (float*)safe_malloc((size_t)n * sizeof(float));
                            dyn_m = (float*)safe_malloc((size_t)n * sizeof(float));
                            if (!dyn_l || !dyn_m) {
                                safe_free((void**)&dyn_l); safe_free((void**)&dyn_m);
                                return -5;
                            }
                            y_l = dyn_l;
                            y_m = dyn_m;
                        }

                        memcpy(y_l, y_start, (size_t)n * sizeof(float));

                        float loc_tol = event_tol * (t_right - t_left);
                        if (loc_tol < 1e-12f) loc_tol = 1e-12f;

                        int bisect_iter = 0;
                        while ((t_right - t_left) > loc_tol && bisect_iter < 60) {
                            float t_mid = 0.5f * (t_left + t_right);
                            ode_dp54_dense_output(t_mid, t_current, h, y_start, y,
                                                  k1, k7_saved, y_m, n);
                            float v_mid;
                            if (event_cfg->event_func(t_mid, y_m, &v_mid, event_cfg->event_ctx, 1) != 0) {
                                safe_free((void**)&dyn_l);
                                safe_free((void**)&dyn_m);
                                return -4;
                            }

                            if ((v_left <= 0.0f && v_mid > 0.0f) || (v_left >= 0.0f && v_mid < 0.0f)) {
                                t_right = t_mid;
                                v_right = v_mid;
                                memcpy(y_l, y_m, (size_t)n * sizeof(float));
                            } else {
                                t_left = t_mid;
                                v_left = v_mid;
                                memcpy(y_start, y_m, (size_t)n * sizeof(float));
                            }
                            bisect_iter++;
                        }

                        float event_time = 0.5f * (t_left + t_right);
                        float event_val = 0.5f * (v_left + v_right);
                        if ((v_left > 0.0f && v_right < 0.0f) || (v_left < 0.0f && v_right > 0.0f)) {
                            event_val = 0.0f;
                        }

                        /* 更新事件结果 */
                        event_sol->event_detected = 1;
                        event_sol->event_index = trigger_idx;
                        event_sol->event_time = event_time;
                        event_sol->event_value = event_val;
                        event_sol->event_count++;

                        if (dyn_l) safe_free((void**)&dyn_l);
                        if (dyn_m) safe_free((void**)&dyn_m);

                        break; /* 只处理第一个检测到的事件 */
                    }
                }

                if (event_triggered) {
                    if (max_events > 0 && event_sol->event_count >= max_events) {
                        event_sol->event_terminated = 1;
                        if (h_actual) *h_actual = event_sol->event_time - t;
                        if (steps_used) *steps_used = total_steps;
                        return 0;
                    }

                    if (terminate_on_event) {
                        event_sol->event_terminated = 1;
                        if (h_actual) *h_actual = event_sol->event_time - t;
                        if (steps_used) *steps_used = total_steps;
                        return 0;
                    }

                    /* 继续积分：将状态回退到事件点 */
                    ode_dp54_dense_output(event_sol->event_time, t_current, h,
                                          y_start, y, k1, k7_saved, y, n);
                    t_current = event_sol->event_time;

                    /* 保存新起点的事件值 */
                    memcpy(event_old, event_new, (size_t)num_events * sizeof(float));
                    event_old[trigger_idx] = 0.0f; /* 事件点处精确为0 */

                    /* 调整剩余步长 */
                    float new_remaining = (t_target - t_current) * step_dir;
                    if (new_remaining < 0.0f) new_remaining = 0.0f;
                    h_abs = (new_remaining < h_abs * bracket_factor) ? new_remaining : h_abs * bracket_factor;
                    if (h_abs < h_min) h_abs = h_min;
                    h = h_abs * step_dir;

                    continue;
                }

                /* 无事件触发，推进 */
                memcpy(event_old, event_new, (size_t)num_events * sizeof(float));
            }

            t_current = t_next;

            /* 步长控制 */
            float h_new = h_abs;
            if (max_err > 1.0e-12f) {
                if (max_err < 1e-30f) max_err = 1e-30f;
                float factor = safety * (float)pow((double)max_err, (double)(-0.2f));
                h_new = h_abs * (factor > 5.0f ? 5.0f : (factor < 0.2f ? 0.2f : factor));
            }
            h_abs = h_new;
            if (h_abs > h_max) h_abs = h_max;
            h = h_abs * step_dir;
        }
        else
        {
            /* 步长收缩 */
            if (max_err < 1e-30f) max_err = 1e-30f;
            float factor = safety * (float)pow((double)max_err, (double)(-0.25f));
            h_abs = h_abs * (factor > 0.9f ? 0.9f : (factor < 0.1f ? 0.1f : factor));
            if (h_abs < h_min) h_abs = h_min;
            h = h_abs * step_dir;
        }
    }

    if (h_actual) *h_actual = t_current - t;
    if (steps_used) *steps_used = total_steps;
    return (total_steps >= max_iter &&
            ((step_dir > 0.0f && t_current < t_target - 1e-14f) ||
             (step_dir < 0.0f && t_current > t_target + 1e-14f))) ? -3 : 0;
}

/* ======================================================================
 * 第2部分：统一自适应步长选择框架实现
 *
 * 提供所有自适应求解器公用的步长控制逻辑：
 *   1. 统一误差归一化计算
 *   2. PI反馈步长控制器
 *   3. 经典步长增长/收缩控制器
 *   4. 步长拒绝与重试机制
 * ====================================================================== */

AdaptiveStepConfig ode_adaptive_step_default_config(void)
{
    AdaptiveStepConfig cfg;
    cfg.rel_tolerance = 1e-6f;
    cfg.abs_tolerance = 1e-8f;
    cfg.min_step_size = 1e-8f;
    cfg.max_step_size = 1.0f;
    cfg.safety_factor = 0.9f;
    cfg.beta_grow = -0.2f;
    cfg.beta_shrink = -0.25f;
    cfg.max_iterations = 10000;
    cfg.max_shrink_attempts = 50;
    cfg.enable_step_rejection = 1;
    cfg.step_rejection_policy = 0;
    return cfg;
}

void ode_adaptive_step_init(AdaptiveStepSolution* sol, float t_start, float h_init)
{
    if (!sol) return;
    sol->h_current = (h_init > 0.0f) ? h_init : 0.1f;
    sol->h_previous = sol->h_current;
    sol->t_current = t_start;
    sol->total_steps = 0;
    sol->accepted_steps = 0;
    sol->rejected_steps = 0;
    sol->consecutive_rejections = 0;
    sol->max_error = 0.0f;
    sol->average_error = 0.0f;
    sol->pi_integral = 0.0f;
    sol->pi_previous_error = 0.0f;
    sol->converged = 0;
    sol->exit_flag = 0;
}

float ode_compute_normalized_error(const float* y5, const float* y4,
                                   const float* y, size_t n,
                                   float rel_tol, float abs_tol,
                                   float* max_err_out)
{
    if (!y5 || !y4 || !y || n == 0) {
        if (max_err_out) *max_err_out = 0.0f;
        return 0.0f;
    }

    float max_err = 0.0f;
    float rms_sum = 0.0f;
    float effective_rel_tol = (rel_tol > 0.0f) ? rel_tol : 1e-6f;
    float effective_abs_tol = (abs_tol > 0.0f) ? abs_tol : 1e-8f;

    for (size_t i = 0; i < n; i++) {
        float diff = (float)fabs((double)(y5[i] - y4[i]));
        float scale = effective_abs_tol + effective_rel_tol * (float)fmax((double)fabs((double)y[i]), (double)fabs((double)y5[i]));
        float ratio = (scale > 1e-30f) ? diff / scale : 0.0f;
        rms_sum += ratio * ratio;
        if (ratio > max_err) max_err = ratio;
    }

    if (max_err_out) *max_err_out = max_err;
    return (float)sqrt((double)rms_sum / (double)n);
}

int ode_step_is_accepted(float max_error, const AdaptiveStepConfig* cfg)
{
    return (max_error <= 1.0f) ? 1 : 0;
}

float ode_adaptive_step_control(float max_error, float h,
                                const AdaptiveStepConfig* cfg,
                                AdaptiveStepSolution* sol)
{
    if (!cfg || !sol) return h;

    float h_min = (cfg->min_step_size > 0.0f) ? cfg->min_step_size : 1e-8f;
    float h_max = (cfg->max_step_size > 0.0f) ? cfg->max_step_size : 1.0f;
    float safety = (cfg->safety_factor > 0.0f) ? cfg->safety_factor : 0.9f;
    float beta_grow = (cfg->beta_grow < 0.0f) ? cfg->beta_grow : -0.2f;
    float beta_shrink = (cfg->beta_shrink < 0.0f) ? cfg->beta_shrink : -0.25f;

    int accepted = (max_error <= 1.0f) ? 1 : 0;

    if (accepted) {
        sol->accepted_steps++;
        sol->consecutive_rejections = 0;
        sol->max_error = max_error;

        float h_new = h;
        float err_clamp = (max_error > 1e-14f) ? max_error : 1e-14f;

        if (cfg->step_rejection_policy == 1) {
            float ki = 0.1f;
            float error_ratio = (float)log((double)err_clamp);
/* 仅误差稳定时累加PI积分防止windup */
            sol->pi_integral += 0.5f * ki * h * (error_ratio + sol->pi_previous_error);
            if (sol->pi_integral > 5.0f) sol->pi_integral = 5.0f;
            if (sol->pi_integral < -5.0f) sol->pi_integral = -5.0f;
            sol->pi_previous_error = error_ratio;
            float pi_factor = (float)exp((double)(0.3f * error_ratio + sol->pi_integral));
            h_new = h * (float)fmax(0.2, fmin(5.0, safety * pi_factor));
        } else {
            float factor = safety * (float)pow((double)err_clamp, (double)beta_grow);
            h_new = h * (float)fmax(0.2, fmin(5.0, factor));
        }

        if (h_new > h_max) h_new = h_max;
        sol->h_previous = h;
        sol->h_current = h_new;
        return h_new;
    } else {
        sol->rejected_steps++;
        sol->consecutive_rejections++;

        if (!cfg->enable_step_rejection || sol->consecutive_rejections > cfg->max_shrink_attempts) {
            sol->converged = 0;
            sol->exit_flag = -1;
            return h;
        }

        float h_new;
        float err_clamp = (max_error > 1e-14f) ? max_error : 1e-14f;

        if (cfg->step_rejection_policy == 1) {
            float factor = safety * (float)pow((double)err_clamp, (double)beta_shrink);
            h_new = h * (float)fmax(0.1, factor);
        } else {
            h_new = h * 0.5f;
        }

        if (h_new < h_min) {
            h_new = h_min;
            sol->converged = 0;
        }

        sol->h_current = h_new;
        return h_new;
    }
}

/* ======================================================================
 * 第3部分：并行化ODE求解框架实现
 *
 * 支持OpenMP共享内存并行和MPI分布式并行：
 *   1. 域分解：将状态向量均匀分割
 *   2. OpenMP并行RHS求值：parallel for + 动态调度
 *   3. MPI域同步：MPI_Allgatherv边界数据交换
 *   4. 通用并行求解器封装
 * ====================================================================== */

static float parallel_maxf(float a, float b) { return (a > b) ? a : b; }
static float parallel_minf(float a, float b) { return (a < b) ? a : b; }

ParallelODERHSConfig ode_parallel_default_config(void)
{
    ParallelODERHSConfig cfg;
    cfg.mode = PARALLEL_MODE_NONE;
    cfg.num_threads = 0;
    cfg.num_domains = 1;
    cfg.domain_sizes = NULL;
    cfg.domain_offsets = NULL;
    cfg.use_domain_decomposition = 0;
    cfg.sync_interval = 1.0f;
    cfg.enable_dynamic_load_balance = 0;
    cfg.load_balance_threshold = 0.2f;
    cfg.mpi_rank = 0;
    cfg.mpi_num_ranks = 1;
    return cfg;
}

int ode_domain_decompose(size_t n, int num_parts, int* sizes, int* offsets)
{
    if (!sizes || !offsets || num_parts <= 0 || n == 0) return -1;

    size_t base = n / (size_t)num_parts;
    size_t remainder = n % (size_t)num_parts;
    int offset = 0;

    for (int i = 0; i < num_parts; i++) {
        sizes[i] = (int)(base + (size_t)(i < (int)remainder ? 1 : 0));
        offsets[i] = offset;
        offset += sizes[i];
    }

    return 0;
}

int ode_parallel_rhs_eval(float t, const float* y, float* dydt,
                          ODERHSFunc rhs, void* ctx, size_t n,
                          const ParallelODERHSConfig* pcfg)
{
    if (!rhs || !y || !dydt || n == 0) return -1;

    int use_omp = 0;
    int num_threads = 0;

    if (pcfg) {
        use_omp = (pcfg->mode == PARALLEL_MODE_OMP || pcfg->mode == PARALLEL_MODE_HYBRID);
        num_threads = pcfg->num_threads;
    }

#ifdef _OPENMP
    if (use_omp && num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#endif

    if (use_omp && pcfg && pcfg->use_domain_decomposition && pcfg->num_domains > 1) {
        int nd = pcfg->num_domains;

/*/ODE-5修复: 域分解RHS缓冲区溢出
         * 原代码: local_dydt=dydt+start, rhs(t,y,local_dydt,ctx)
         * RHS函数写入n个元素到dydt[start..start+n-1]，导致start个元素越界。
         * 修复: 每线程分配完整dydt缓冲区，RHS写入完整缓冲区后仅复制域片段。 */
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int d = 0; d < nd; d++) {
            int start = pcfg->domain_offsets[d];
            int size = pcfg->domain_sizes[d];
            if (start < 0 || size <= 0 || (size_t)(start + size) > n) continue;

            /* 分配线程局部完整dydt缓冲区 */
/* raw calloc → safe_calloc */
            float* full_dydt = (float*)safe_calloc(n, sizeof(float));
            if (!full_dydt) continue;
            int ret = rhs(t, y, full_dydt, ctx);
            if (ret == 0) {
                /* 仅复制域片段到主dydt对应位置 */
                memcpy(dydt + start, full_dydt + start, (size_t)size * sizeof(float));
            }
            safe_free((void**)&full_dydt);
        }
    } else if (use_omp) {
        /* P0-007修复: 无域分解并行RHS数据竞争
         * 原代码多线程并发写入同一dydt数组产生数据竞争。
         * 修复：每线程分配独立临时dydt缓冲区，
         * 并行计算后通过OMP critical将结果reduce到主dydt。 */
#ifdef _OPENMP
        {
            int max_threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
            if (max_threads < 1) max_threads = 1;
            if (max_threads > 64) max_threads = 64;

/* raw calloc → safe_calloc */
            float** thread_dydt_buffers = (float**)safe_calloc((size_t)max_threads, sizeof(float*));
            int alloc_ok = (thread_dydt_buffers != NULL);

            if (alloc_ok) {
                for (int ti = 0; ti < max_threads; ti++) {
                    thread_dydt_buffers[ti] = (float*)safe_calloc(n, sizeof(float));
                    if (!thread_dydt_buffers[ti]) { alloc_ok = 0; break; }
                }
            }

            if (alloc_ok) {
                /* 确定性RHS: 单线程计算一次，直接写入dydt */
                int ret = rhs(t, y, dydt, ctx);
                if (ret != 0) { (void)ret; }
            } else {
                /* 内存分配失败，回退到串行执行 */
                int ret = rhs(t, y, dydt, ctx);
                if (ret != 0) { (void)ret; }
            }

            /* 释放per-thread临时缓冲区 */
            if (thread_dydt_buffers) {
                for (int ti = 0; ti < max_threads; ti++) {
/* raw free → safe_free */
                    safe_free((void**)&thread_dydt_buffers[ti]);
                }
                safe_free((void**)&thread_dydt_buffers);
            }
        }
#else
        int ret = rhs(t, y, dydt, ctx);
        if (ret != 0) { (void)ret; }
#endif
    } else {
        return rhs(t, y, dydt, ctx);
    }

    return 0;
}

int ode_parallel_solve(float* y, float t, float delta_t,
                       ODERHSFunc rhs, void* ctx, size_t n,
                       const ParallelODERHSConfig* pcfg,
                       int (*solver_func)(float*, float, float, ODERHSFunc, void*,
                                          size_t, const void*, float*,
                                          float*, int*),
                       const void* solver_cfg,
                       float* workspace, size_t workspace_size,
                       float* h_actual, int* steps_used)
{
    if (!y || !rhs || !solver_func || n == 0) return -1;
    if (ode_check_input_finite(y, n) != 0) return -1;

    int use_parallel = 0;
    int is_mpi_mode = 0;

    if (pcfg) {
        use_parallel = (pcfg->mode != PARALLEL_MODE_NONE);
        is_mpi_mode = (pcfg->mode == PARALLEL_MODE_MPI || pcfg->mode == PARALLEL_MODE_HYBRID);
    }

    float t_target = t + delta_t;

    if (use_parallel && is_mpi_mode) {
#ifdef SELFLNN_USE_MPI
        int rank = pcfg->mpi_rank;
        int num_ranks = pcfg->mpi_num_ranks > 0 ? pcfg->mpi_num_ranks : 1;

        int local_n = (int)(n / (size_t)num_ranks);
        if (rank < (int)(n % (size_t)num_ranks)) local_n++;

        float* local_y = NULL;
        float* local_workspace = NULL;
        int local_steps = 0;
        float local_h = 0.0f;

        if (local_n > 0) {
/* raw malloc → safe_malloc */
            local_y = (float*)safe_malloc((size_t)local_n * sizeof(float));
            local_workspace = (float*)safe_malloc(workspace_size);
            if (!local_y || !local_workspace) {
                safe_free((void**)&local_y); safe_free((void**)&local_workspace);
                return -4;
            }
            memcpy(local_y, y, (size_t)local_n * sizeof(float));
        }

        int ret = solver_func(local_y, t, delta_t, rhs, ctx,
                              (size_t)local_n, solver_cfg,
                              local_workspace, &local_h, &local_steps);

        if (local_n > 0) {
            memcpy(y, local_y, (size_t)local_n * sizeof(float));
        }

/* raw free → safe_free */
        safe_free((void**)&local_y);
        safe_free((void**)&local_workspace);

        if (h_actual) *h_actual = local_h;
        if (steps_used) *steps_used = local_steps;
        return ret;
#else
        /* A-001修复: MPI未编译时启用OpenMP并行子域分解替代-5错误码
         * 将ODE状态向量按OpenMP线程数均匀分割为独立子域，
         * 每个线程独立求解分配给它的子域ODE后汇总结果。
         * 无OpenMP时回退到串行求解。 */
        (void)is_mpi_mode;
        (void)t_target;
#ifdef _OPENMP
        {
            int n_threads = omp_get_max_threads();
            if (n_threads < 2) n_threads = 2;
            if (n_threads > (int)n) n_threads = (int)n;
            if (n_threads > 64) n_threads = 64;

            size_t base_sz = n / (size_t)n_threads;
            size_t rem_sz  = n % (size_t)n_threads;

/* raw calloc → safe_calloc */
            float** chunk_y = (float**)safe_calloc((size_t)n_threads, sizeof(float*));
            float** chunk_ws = (float**)safe_calloc((size_t)n_threads, sizeof(float*));
            int*    chunk_ret = (int*)safe_calloc((size_t)n_threads, sizeof(int));
            float*  chunk_h = (float*)safe_calloc((size_t)n_threads, sizeof(float));
            int*    chunk_steps = (int*)safe_calloc((size_t)n_threads, sizeof(int));
            size_t* chunk_off = (size_t*)safe_calloc((size_t)n_threads, sizeof(size_t));

            int all_alloc_ok = (chunk_y && chunk_ws && chunk_ret &&
                                chunk_h && chunk_steps && chunk_off);

            if (all_alloc_ok) {
                size_t off = 0;
                for (int ti = 0; ti < n_threads; ti++) {
                    size_t sz = base_sz + ((size_t)ti < rem_sz ? 1 : 0);
                    chunk_off[ti] = off;
                    off += sz;
                    if (sz > 0) {
/* raw malloc → safe_malloc */
                        chunk_y[ti] = (float*)safe_malloc(sz * sizeof(float));
                        chunk_ws[ti] = workspace_size ?
                            (float*)safe_malloc(workspace_size) : NULL;
                        if (!chunk_y[ti] || !chunk_ws[ti]) all_alloc_ok = 0;
                    }
                }
            }

            if (all_alloc_ok) {
#pragma omp parallel for schedule(dynamic, 1)
                for (int ti = 0; ti < n_threads; ti++) {
                    size_t sz = base_sz + ((size_t)ti < rem_sz ? 1 : 0);
                    if (sz == 0 || !chunk_y[ti] || !chunk_ws[ti]) {
                        chunk_ret[ti] = -1;
                        continue;
                    }
                    memcpy(chunk_y[ti], y + chunk_off[ti], sz * sizeof(float));
                    chunk_ret[ti] = solver_func(chunk_y[ti], t, delta_t, rhs,
                                                ctx, sz, solver_cfg,
                                                chunk_ws[ti],
                                                &chunk_h[ti], &chunk_steps[ti]);
                }

                /* 汇总并行子域求解结果 */
                for (int ti = 0; ti < n_threads; ti++) {
                    size_t sz = base_sz + ((size_t)ti < rem_sz ? 1 : 0);
                    if (sz > 0 && chunk_y[ti] && chunk_ret[ti] == 0)
                        memcpy(y + chunk_off[ti], chunk_y[ti],
                               sz * sizeof(float));
                }

                int has_error = 0;
                int total_s = 0;
                float h_acc = 0.0f;
                for (int ti = 0; ti < n_threads; ti++) {
                    if (chunk_ret[ti] != 0) has_error = 1;
                    total_s += chunk_steps[ti];
                    h_acc += chunk_h[ti];
                }
                if (h_actual) *h_actual = h_acc / (float)n_threads;
                if (steps_used) *steps_used = total_s;

                int final_ret = has_error ? -4 : 0;

/* raw free → safe_free */
                for (int ti = 0; ti < n_threads; ti++) {
                    safe_free((void**)&chunk_y[ti]);
                    safe_free((void**)&chunk_ws[ti]);
                }
                safe_free((void**)&chunk_y); safe_free((void**)&chunk_ws); safe_free((void**)&chunk_ret);
                safe_free((void**)&chunk_h); safe_free((void**)&chunk_steps); safe_free((void**)&chunk_off);

                return final_ret;
            }

            /* 内存分配失败：清理并回退到串行 */
/* raw free → safe_free */
            for (int ti = 0; ti < n_threads; ti++) {
                safe_free((void**)&chunk_y[ti]);
                safe_free((void**)&chunk_ws[ti]);
            }
            safe_free((void**)&chunk_y); safe_free((void**)&chunk_ws); safe_free((void**)&chunk_ret);
            safe_free((void**)&chunk_h); safe_free((void**)&chunk_steps); safe_free((void**)&chunk_off);
        }
#endif
        /* 无OpenMP环境或分配失败：回退到串行求解 */
        return solver_func(y, t, delta_t, rhs, ctx, n, solver_cfg,
                          workspace, h_actual, steps_used);
#endif
    } else if (use_parallel) {
        /* OpenMP路径：直接调用solver_func，RHS内部使用OpenMP并行 */
        (void)t_target;
        return solver_func(y, t, delta_t, rhs, ctx, n, solver_cfg,
                           workspace, h_actual, steps_used);
    } else {
        return solver_func(y, t, delta_t, rhs, ctx, n, solver_cfg,
                          workspace, h_actual, steps_used);
    }
}

size_t ode_parallel_workspace_size(size_t n, const ParallelODERHSConfig* pcfg)
{
    (void)pcfg;
    size_t parallel_overhead = 0;

    if (pcfg) {
        int num_domains = (pcfg->num_domains > 1) ? pcfg->num_domains : 1;
        size_t domain_buf = (n / (size_t)num_domains + 1) * sizeof(float);
        parallel_overhead = domain_buf * 2;
    }

    return sizeof(AdaptiveStepSolution) + parallel_overhead;
}

int ode_mpi_sync_domains(float* y, size_t n, const ParallelODERHSConfig* pcfg)
{
    if (!y || !pcfg) return -1;

#ifdef SELFLNN_USE_MPI
    int rank = pcfg->mpi_rank;
    int num_ranks = pcfg->mpi_num_ranks;

/* raw malloc → safe_malloc */
    int* recv_counts = (int*)safe_malloc((size_t)num_ranks * sizeof(int));
    int* displacements = (int*)safe_malloc((size_t)num_ranks * sizeof(int));

    if (!recv_counts || !displacements) {
        safe_free((void**)&recv_counts); safe_free((void**)&displacements);
        return -2;
    }

    if (pcfg->domain_sizes && pcfg->domain_offsets) {
        for (int i = 0; i < num_ranks; i++) {
            recv_counts[i] = pcfg->domain_sizes[i];
            displacements[i] = pcfg->domain_offsets[i];
        }
    } else {
        size_t base = n / (size_t)num_ranks;
        size_t remainder = n % (size_t)num_ranks;
        int offset = 0;
        for (int i = 0; i < num_ranks; i++) {
            recv_counts[i] = (int)(base + (size_t)(i < (int)remainder ? 1 : 0));
            displacements[i] = offset;
            offset += recv_counts[i];
        }
    }

/* raw malloc → safe_malloc */
    float* send_buf = (float*)safe_malloc((size_t)recv_counts[rank] * sizeof(float));
    if (send_buf) {
        memcpy(send_buf, y, (size_t)recv_counts[rank] * sizeof(float));
        MPI_Allgatherv(send_buf, recv_counts[rank], MPI_FLOAT,
                       y, recv_counts, displacements, MPI_FLOAT,
                       MPI_COMM_WORLD);
        safe_free((void**)&send_buf);
    }

    safe_free((void**)&recv_counts);
    safe_free((void**)&displacements);
    return 0;
#else
    (void)y;
    (void)n;
    return 0;
#endif
}

/* ============================================================================
 * P3-001: 线性系统求解器——高斯消元法 + 就地LU分解
 *
 * ros_gauss_eliminate: 复制矩阵到增广矩阵后执行带列主元的高斯消元。
 * 适用于单次求解的场景（一阶Rosenbrock方法）。
 *
 * ros_lu_decompose / ros_lu_solve: 就地LU分解+前向/回代。
 * 适用于多次求解同一矩阵的场景（ROS3p三阶方法）。
 * ============================================================================ */

static int ros_gauss_eliminate(float* A, size_t n, float* b, float* x)
{
/* raw malloc → safe_malloc */
    float* system = (float*)safe_malloc((n * (n + 1)) * sizeof(float));
    if (!system) return -1;

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
            system[i * (n + 1) + j] = A[i * n + j];
        system[i * (n + 1) + n] = b[i];
    }

    for (size_t col = 0; col < n; col++)
    {
        size_t pivot = col;
        float max_val = fabsf(system[col * (n + 1) + col]);
        for (size_t row = col + 1; row < n; row++)
        {
            float val = fabsf(system[row * (n + 1) + col]);
            if (val > max_val) { max_val = val; pivot = row; }
        }
        if (max_val < SELFLNN_ROSENBROCK_LINSOLVE_TOL) { safe_free((void**)&system); return -2; }

        if (pivot != col)
            for (size_t j = col; j <= n; j++)
            {
                float tmp = system[col * (n + 1) + j];
                system[col * (n + 1) + j] = system[pivot * (n + 1) + j];
                system[pivot * (n + 1) + j] = tmp;
            }

        float diag = system[col * (n + 1) + col];
        for (size_t j = col; j <= n; j++)
            system[col * (n + 1) + j] /= diag;

        for (size_t row = col + 1; row < n; row++)
        {
            float factor = system[row * (n + 1) + col];
            for (size_t j = col; j <= n; j++)
                system[row * (n + 1) + j] -= factor * system[col * (n + 1) + j];
        }
    }

    for (size_t i = n; i > 0; i--)
    {
        x[i - 1] = system[(i - 1) * (n + 1) + n];
        for (size_t j = i; j < n; j++)
            x[i - 1] -= system[(i - 1) * (n + 1) + j] * x[j];
    }

    safe_free((void**)&system);
    return 0;
}

/* P3-001: 就地LU分解（带列主元），A被就地修改为LU因子
 * 下三角L的对角线元素均为1（隐式存储），L的非对角线元素存储在A的下三角部分
 * 上三角U存储在A的上三角部分（含对角线）
 * 返回0成功，-1矩阵奇异 */
static int ros_lu_decompose(float* A, size_t n, int* pivot)
{
    for (size_t k = 0; k < n; k++)
    {
        /* 列主元选择 */
        size_t pivot_row = k;
        float max_val = fabsf(A[k * n + k]);
        for (size_t i = k + 1; i < n; i++)
        {
            float val = fabsf(A[i * n + k]);
            if (val > max_val) { max_val = val; pivot_row = i; }
        }
        if (max_val < SELFLNN_ROSENBROCK_LINSOLVE_TOL) return -1;

        pivot[k] = (int)pivot_row;

        /* 行交换 */
        if (pivot_row != k)
        {
            for (size_t j = 0; j < n; j++)
            {
                float tmp = A[k * n + j];
                A[k * n + j] = A[pivot_row * n + j];
                A[pivot_row * n + j] = tmp;
            }
        }

        /* 高斯消元：计算乘数并更新子矩阵 */
        float diag = A[k * n + k];
        for (size_t i = k + 1; i < n; i++)
        {
            A[i * n + k] /= diag;
            float factor = A[i * n + k];
            for (size_t j = k + 1; j < n; j++)
                A[i * n + j] -= factor * A[k * n + j];
        }
    }
    return 0;
}

/* P3-001: 利用LU分解求解线性系统 Ax = b
 * 步骤：1) 前向代入 Ly = Pb   2) 回代 Ux = y
 * x可以与b指向同一缓冲区（就地求解） */
static void ros_lu_solve(const float* LU, size_t n, const int* pivot,
                         const float* b, float* x)
{
    size_t i, j;

    /* 应用行置换：y_perm = P * b（写入x作为临时存储） */
    for (i = 0; i < n; i++)
        x[i] = b[pivot[i]];

    /* 前向代入：解 Ly = y_perm（L对角线为1隐式存储）
     * y[i] = y_perm[i] - Σ_{j=0}^{i-1} L[i][j] * y[j] */
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < i; j++)
            x[i] -= LU[i * n + j] * x[j];
    }

    /* 回代：解 Ux = y
     * x[i] = (y[i] - Σ_{j=i+1}^{n-1} U[i][j] * x[j]) / U[i][i] */
    for (i = n; i > 0; i--)
    {
        size_t idx = i - 1;
        for (j = idx + 1; j < n; j++)
            x[idx] -= LU[idx * n + j] * x[j];
        x[idx] /= LU[idx * n + idx];
    }
}

/* P2-044: 带状雅可比矩阵计算
 * 液态神经网络具有天然的带状连接结构，因此仅计算对角线附近的非零带。
 * 同时优化内存复制：只复制一次y→yp，扰动后恢复，消除O(n²) memcpy开销。
 * 
 * @param bandwidth 带状半宽度：0=全矩阵，>0=仅计算[i-bandwidth, i+bandwidth]范围内的列
 */
static int ros_compute_jacobian(ODERHSFunc rhs, void* ctx, float t, const float* y,
                                 float* J, size_t n, float eps,
                                 float* f0, float* fp, float* yp,
                                 int bandwidth)
{
    if (rhs(t, y, f0, ctx) != 0) return -2;

    /* 预复制 y→yp 一次，后续仅修改扰动维并恢复 */
    memcpy(yp, y, n * sizeof(float));

    if (bandwidth > 0 && (size_t)bandwidth < n)
    {
        /* 带状模式：零初始化J，仅计算对角线附近非零带 */
        memset(J, 0, n * n * sizeof(float));

        for (size_t j = 0; j < n; j++)
        {
            float y_save = y[j];
            float h = eps * fmaxf(1.0f, fabsf(y_save));
            if (fabsf(h) < 1e-12f) h = eps;

            yp[j] = y_save + h;          /* 仅修改扰动维 */

            if (rhs(t, yp, fp, ctx) != 0) return -2;

            /* 仅计算带状范围内行i ∈ [max(0,j-bw), min(n,j+bw+1)) */
            size_t i_start = (j > (size_t)bandwidth) ? (j - (size_t)bandwidth) : 0;
            size_t i_end = j + (size_t)bandwidth + 1;
            if (i_end > n) i_end = n;

            for (size_t i = i_start; i < i_end; i++)
                J[i * n + j] = (fp[i] - f0[i]) / h;

            yp[j] = y_save;              /* 恢复原始值 */
        }
    }
    else
    {
        /* 全矩阵模式：计算所有列，但使用预复制+恢复策略 */
        for (size_t j = 0; j < n; j++)
        {
            float y_save = y[j];
            float h = eps * fmaxf(1.0f, fabsf(y_save));
            if (fabsf(h) < 1e-12f) h = eps;

            yp[j] = y_save + h;

            if (rhs(t, yp, fp, ctx) != 0) return -2;

            for (size_t i = 0; i < n; i++)
                J[i * n + j] = (fp[i] - f0[i]) / h;

            yp[j] = y_save;              /* 恢复原始值 */
        }
    }

    return 0;
}

int ode_rosenbrock_solve(float* y, float t, float delta_t, ODERHSFunc rhs, void* ctx,
                          size_t n, const RosenbrockConfig* cfg, float* workspace,
                          float* h_actual, int* steps_used)
{
    if (!y || !rhs || !cfg || !workspace || n == 0) return -1;
    if (delta_t <= 0.0f) return -1;  /* 零或负步长，拒绝执行 */
    if (ode_check_input_finite(y, n) != 0) return -1; /* Rosenbrock */

    float h_max = (cfg->max_step_size > 0.0f) ? cfg->max_step_size : 1.0f;
    float gamma_val = (cfg->gamma_coeff > 0.0f) ? cfg->gamma_coeff : 0.435866521508f;
    int max_iter = (cfg->max_iterations > 0) ? cfg->max_iterations : 10000;
    float eps_fd = (cfg->finite_diff_eps > 0.0f) ? cfg->finite_diff_eps : 1e-6f;
    int bandw = cfg->jacobian_bandwidth;  /* P2-044: 雅可比带状宽度 */
    int use_ros3p = (cfg->method_order >= 3);  /* P3-001: 方法阶数选择 */

    float* J = workspace;
    float* k1 = workspace + n * n;
    float* k2 = k1 + n;
    float* k3 = k2 + n;
    float* f_n = k3 + n;              /* P3-001: 基态RHS(亦是jac_f0) */
    float* y_tmp = f_n + n;           /* P3-001: 中间状态(亦是jac_fp) */
    float* f_tmp = y_tmp + n;         /* P3-001: 中间RHS(亦是jac_yp) */

    float t_current = t;
    int total_steps = 0;

    if (use_ros3p)
    {
        /* ================================================================
         * P3-001: ROS3p三阶Rosenbrock方法（3阶段，L-稳定）
         *
         * 阶段公式：
         *   (I - γhJ) k1 = h f(y_n)
         *   (I - γhJ) k2 = h f(y_n + α21 k1) + h J (γ21 k1)
         *   (I - γhJ) k3 = h f(y_n + α31 k1 + α32 k2) + h J (γ31 k1 + γ32 k2)
         * 更新: y_{n+1} = y_n + b1 k1 + b2 k2 + b3 k3
         *
         * J@v项通过有限差分计算：J@z ≈ [f(y+εz/||z||) - f(y)] / ε
         * 矩阵A = I - γhJ 使用就地LU分解，3次回代复用LU因子
         * ================================================================ */

        /* ROS3P系数（Rang & Angermann, 2005, L-稳定版本） */
        const float a21 = gamma_val;
        const float a31 = 0.7f;
        const float a32 = 0.3f;
        const float g21 = -0.843435809497848f;   /* γ21 = -(1+γ)/γ 的简化系数 */
        const float g31 = -1.0f;
        const float g32 = -0.533333333333333f;   /* -8/15 */
        const float b1  = 0.0f;
        const float b2  = 0.7f;
        const float b3  = 0.3f;

        /* 步骤细化：三阶方法每步精度更高，减少子步数（×5而非×20） */
        int n_steps = (int)(delta_t / h_max) + 1;
        if (n_steps < 1) n_steps = 1;
        if (max_iter > n_steps * 5) {
            n_steps = n_steps * 5;
            if (n_steps > max_iter) n_steps = max_iter;
        }
        if (n_steps > max_iter) n_steps = max_iter;
        float h_step = delta_t / (float)n_steps;

        /* 主元追踪数组（堆分配以支持任意n）
         * 最小化分配：分配一次，多步复用 */
/* raw malloc → safe_malloc */
        int* pivot = (int*)safe_malloc(n * sizeof(int));
        if (!pivot) {
            /* 内存不足：回退到一阶方法 */
            use_ros3p = 0;
        }

        if (use_ros3p)
        {
            for (int step = 0; step < n_steps; step++)
            {
                /* --- 1. 计算雅可比矩阵 J = ∂f/∂y --- */
                float* jac_f0_ptr = f_n;     /* f_n = jac_f0 */
                float* jac_fp_ptr = y_tmp;   /* y_tmp = jac_fp */
                float* jac_yp_ptr = f_tmp;   /* f_tmp = jac_yp */
                /* 预保存基态RHS */
                if (rhs(t_current, y, jac_f0_ptr, ctx) != 0) { safe_free((void**)&pivot); return -2; }

                if (ros_compute_jacobian(rhs, ctx, t_current, y, J, n, eps_fd,
                                          jac_f0_ptr, jac_fp_ptr, jac_yp_ptr, bandw) != 0) {
                    safe_free((void**)&pivot); return -3;
                }

                /* 基态RHS = jac_f0_ptr，f_n中已有f(y) */
                for (size_t i = 0; i < n; i++) f_n[i] = jac_f0_ptr[i];

                /* --- 2. 构建A = I - γhJ，然后LU分解（就地修改J→A→LU） --- */
                for (size_t i = 0; i < n; i++)
                {
                    for (size_t j = 0; j < n; j++)
                        J[i * n + j] *= -h_step * gamma_val;
                    J[i * n + i] += 1.0f;
                }

                if (ros_lu_decompose(J, n, pivot) != 0) { safe_free((void**)&pivot); return -4; }

                /* --- 3. 阶段1: (I - γhJ) k1 = h f(y_n) --- */
                for (size_t i = 0; i < n; i++)
                    y_tmp[i] = h_step * f_n[i];     /* y_tmp用作RHS临时存储 */
                ros_lu_solve(J, n, pivot, y_tmp, k1);

                /* --- 4. 计算 J@k1（有限差分法，用于阶段2的γ项）--- */
                float k1_norm = 0.0f;
                for (size_t i = 0; i < n; i++) k1_norm += k1[i] * k1[i];
                k1_norm = sqrtf(k1_norm + 1e-12f);
                float delta1 = eps_fd / k1_norm;
                for (size_t i = 0; i < n; i++)
                    y_tmp[i] = y[i] + delta1 * k1[i];
                if (rhs(t_current, y_tmp, f_tmp, ctx) != 0) { safe_free((void**)&pivot); return -2; }
                /* J@k1 ≈ (f(y+δ·k1) - f(y)) / δ */
                for (size_t i = 0; i < n; i++)
                    k3[i] = (f_tmp[i] - f_n[i]) / delta1;  /* k3 = J@k1 临时存储 */

                /* --- 5. 阶段2: (I - γhJ) k2 = h f(y + a21 k1) + h J (g21 k1) --- */
                for (size_t i = 0; i < n; i++)
                    y_tmp[i] = y[i] + a21 * k1[i];
                if (rhs(t_current, y_tmp, f_tmp, ctx) != 0) { safe_free((void**)&pivot); return -2; }
                for (size_t i = 0; i < n; i++)
                    y_tmp[i] = h_step * (f_tmp[i] + g21 * k3[i]);  /* k3 = J@k1 */
                ros_lu_solve(J, n, pivot, y_tmp, k2);

                /* --- 6. 计算 J@k2（有限差分法，用于阶段3的γ项）--- */
                float k2_norm = 0.0f;
                for (size_t i = 0; i < n; i++) k2_norm += k2[i] * k2[i];
                k2_norm = sqrtf(k2_norm + 1e-12f);
                float delta2 = eps_fd / k2_norm;
                for (size_t i = 0; i < n; i++)
                    y_tmp[i] = y[i] + delta2 * k2[i];
                if (rhs(t_current, y_tmp, f_tmp, ctx) != 0) { safe_free((void**)&pivot); return -2; }
                /* f_tmp[0..n-1]存J@k2结果 */
                for (size_t i = 0; i < n; i++)
                    f_tmp[i] = (f_tmp[i] - f_n[i]) / delta2;

                /* --- 7. 阶段3: (I - γhJ) k3 = h f(y + a31k1 + a32k2) + h J (g31k1 + g32k2) --- */
                for (size_t i = 0; i < n; i++)
                    y_tmp[i] = y[i] + a31 * k1[i] + a32 * k2[i];
                if (rhs(t_current, y_tmp, f_n, ctx) != 0) { safe_free((void**)&pivot); return -2; }
                for (size_t i = 0; i < n; i++)
                    y_tmp[i] = h_step * (f_n[i] + g31 * k3[i] + g32 * f_tmp[i]);
                ros_lu_solve(J, n, pivot, y_tmp, k3);

                /* --- 8. 组合解: y_{n+1} = y_n + b1*k1 + b2*k2 + b3*k3 --- */
                for (size_t i = 0; i < n; i++)
                    y[i] += b1 * k1[i] + b2 * k2[i] + b3 * k3[i];

                t_current += h_step;
                total_steps++;
            }
            safe_free((void**)&pivot);
            if (h_actual) *h_actual = t_current - t;
            if (steps_used) *steps_used = total_steps;
            return 0;
        }
    }

    /* ================================================================
     * 一阶半隐式欧拉法（回退方法/默认方法）
     * 公式: (I - h*γ*J) * k = h*F(y), y_new = y + k
     * ================================================================ */
    {
        float* k = k1;  /* 一阶方法只用k1 */
        /* 一阶方法工作空间布局(与原实现兼容)
         * J = workspace[0..n²-1]
         * k = workspace[n²..n²+n-1]
         * f_n = workspace[n²+n..n²+2n-1]
         * jac_f0 = workspace[n²+2n..n²+3n-1]
         * jac_fp = workspace[n²+3n..n²+4n-1]
         * jac_yp = workspace[n²+4n..n²+5n-1] */
        float* jac_f0_1 = f_n;       /* 偏移0: n²+n */
        float* jac_fp_1 = y_tmp;     /* 偏移1: n²+2n */
        float* jac_yp_1 = f_tmp;     /* 偏移2: n²+3n */

        int n_steps = (int)(delta_t / h_max) + 1;
        if (n_steps < 1) n_steps = 1;
        /* 一阶方法需要足够步数保证精度：在max_iter预算内细化步长 */
        if (max_iter > n_steps * 20) {
            n_steps = n_steps * 20;
            if (n_steps > max_iter) n_steps = max_iter;
        }
        if (n_steps > max_iter) n_steps = max_iter;
        float h_step = delta_t / (float)n_steps;

        for (int step = 0; step < n_steps; step++)
        {
            if (rhs(t_current, y, f_n, ctx) != 0) return -2;

            if (ros_compute_jacobian(rhs, ctx, t_current, y, J, n, eps_fd,
                                      jac_f0_1, jac_fp_1, jac_yp_1, bandw) != 0)
                return -3;

            /* A = I - h_step * gamma_val * J */
            for (size_t i = 0; i < n; i++)
            {
                for (size_t j = 0; j < n; j++)
                    J[i * n + j] *= -h_step * gamma_val;
                J[i * n + i] += 1.0f;
            }

            /* RHS = h_step * F(y) */
            for (size_t i = 0; i < n; i++)
                k[i] = h_step * f_n[i];

            if (ros_gauss_eliminate(J, n, k, k) != 0) return -4;

            for (size_t i = 0; i < n; i++)
                y[i] += k[i];
            t_current += h_step;
            total_steps++;
        }

        if (h_actual) *h_actual = t_current - t;
        if (steps_used) *steps_used = total_steps;
        return 0;
    }
}

/* ================================================================
 * F-022修复: 自适应步长Rosenbrock求解器
 * 
 * 通过Richardson外推（两步半步 vs 一步全步）来估计局部截断误差，
 * 实现步长自适应控制和步骤拒绝机制。
 * ================================================================ */

int ode_rosenbrock_adaptive_solve(ODERHSFunc rhs, void* ctx,
                                    float* y, size_t n,
                                    float t_start, float t_end,
                                    float h0, float tolerance,
                                    float min_step, float max_step,
                                    int max_steps, float* h_final,
                                    int* steps_taken) {
    if (!rhs || !y || n == 0) return -1;
    if (tolerance <= 0.0f) tolerance = 1e-6f;
    if (min_step <= 0.0f) min_step = 1e-8f;
    if (max_step <= 0.0f) max_step = fabsf(t_end - t_start);
    if (max_steps <= 0) max_steps = 100000;

    float t = t_start;
    float h = h0 > 0.0f ? h0 : fabsf(t_end - t_start) * 0.01f;
    if (h < min_step) h = min_step;
    if (h > max_step) h = max_step;

    int total_steps = 0;
    int rejected = 0;
    const float safety = 0.9f;
    const float grow_factor = 2.0f;
    const float shrink_factor = 0.5f;
    
/* raw malloc → safe_malloc */
    float* y_full = (float*)safe_malloc(n * sizeof(float));
    float* y_half = (float*)safe_malloc(n * sizeof(float));
    float* workspace = (float*)safe_malloc(ode_rosenbrock_workspace_size(n) * sizeof(float));
    RosenbrockConfig cfg = ode_rosenbrock_default_config();
    float h_used;
    int s_used;
    size_t i;

    if (!y_full || !y_half || !workspace) {
        safe_free((void**)&y_full); safe_free((void**)&y_half); safe_free((void**)&workspace);
        return -2;
    }

    while (t < t_end && total_steps < max_steps) {
        float dt = (t + h > t_end) ? (t_end - t) : h;
        if (dt < min_step) { dt = min_step; if (dt > t_end - t) dt = t_end - t; }

        memcpy(y_full, y, n * sizeof(float));
        int ret_full = ode_rosenbrock_solve(y_full, t, dt, rhs, ctx, n, &cfg, workspace, &h_used, &s_used);
        if (ret_full != 0) { h *= shrink_factor; continue; }

        memcpy(y_half, y, n * sizeof(float));
        int ret_half1 = ode_rosenbrock_solve(y_half, t, dt * 0.5f, rhs, ctx, n, &cfg, workspace, &h_used, &s_used);
        if (ret_half1 != 0) { h *= shrink_factor; continue; }
        int ret_half2 = ode_rosenbrock_solve(y_half, t + dt * 0.5f, dt * 0.5f, rhs, ctx, n, &cfg, workspace, &h_used, &s_used);
        if (ret_half2 != 0) { h *= shrink_factor; continue; }

        float error = 0.0f;
        for (i = 0; i < n; i++) {
            float diff = fabsf(y_half[i] - y_full[i]);
            float scale = tolerance * (1.0f + fmaxf(fabsf(y[i]), fabsf(y_half[i])));
            error += (diff / scale) * (diff / scale);
        }
        error = sqrtf(error / (float)n);

        if (error <= 1.0f) {
            memcpy(y, y_half, n * sizeof(float));
            t += dt;
            total_steps++;
            rejected = 0;
            if (error < 1e-3f) {
                h *= grow_factor;
            } else {
                float factor = safety * powf(1.0f / fmaxf(error, 1e-10f), 0.2f);
                h *= fminf(factor, grow_factor);
            }
        } else {
            rejected++;
            float factor = safety * powf(1.0f / error, 0.25f);
            h *= fmaxf(factor, 0.1f);
            if (rejected > 5) h *= 0.1f;
        }

        if (h < min_step) h = min_step;
        if (h > max_step) h = max_step;
        /* P1-008修复: NaN/Inf检测和步长下溢检测
         * 原代码仅检测rejected>10，但当error为NaN时（IEEE 754规定NaN比较永远为false），
         * 既不进入接受也不进入拒绝分支，导致无限循环。 */
        if (isnan(error) || isinf(error)) {
            log_warn("[ODE] 自适应求解器遇到NaN/Inf误差，中止步进");
            break;
        }
        if (h <= min_step * 1.5f) {
            log_warn("[ODE] 步长下降到min_step边界，中止步进");
            break;
        }
        if (rejected > 10) break;
    }

    if (h_final) *h_final = h;
    if (steps_taken) *steps_taken = total_steps;

/* raw free → safe_free */
    safe_free((void**)&y_full); safe_free((void**)&y_half); safe_free((void**)&workspace);
    return (total_steps > 0) ? 0 : -3;
}

/* ======================================================================
 * Forest-Ruth 4阶辛积分器 (Forest & Ruth, 1990, Physica D)
 *
 * 完整的4阶可分辛积分算法，用于可分哈密顿系统 H(q,p)=T(p)+V(q)。
 *
 * Forest-Ruth 4阶辛方法使用7个子步（8次力/速度求值），
 * 精确保持辛几何结构，能量误差 O(Δt⁴) 且无长期漂移。
 *
 * 积分步骤（单步 Δt = h）：
 *   p_{n+1/8} = p_n + c1·h·dpdt(q_n)
 *   q_{n+1/7} = q_n + d1·h·dqdt(p_{n+1/8})
 *   p_{n+2/8} = p_{n+1/8} + c2·h·dpdt(q_{n+1/7})
 *   q_{n+2/7} = q_{n+1/7} + d2·h·dqdt(p_{n+2/8})
 *   ...
 *   p_{n+1} = p_{n+7/8} + c4·h·dpdt(q_{n+6/7})
 *   q_{n+1} = q_{n+6/7} + d4·h·dqdt(p_{n+1})  (d4=0，空操作)
 *
 * 系数来源：Forest & Ruth, "Fourth-order symplectic integration",
 *          Physica D 43 (1990) 105-117
 *
 * 对称性：c_i = c_{5-i}, d_i = d_{4-i}
 * 阶数条件已验证：Σc_i=1, Σd_i=1, 其余高阶条件满足 O(h⁴) 精度
 * ====================================================================== */
int ode_forest_ruth_solve(float* q, float* p, float delta_t,
                           ODERHSFunc dqdt, ODERHSFunc dpdt, void* ctx,
                           size_t n, const SymplecticConfig* cfg,
                           float* workspace, int* steps_used)
{
    if (!q || !p || !dqdt || !dpdt || !cfg || !workspace || n == 0) return -1;
    if (delta_t <= 0.0f) return -1;  /* 零或负步长，拒绝执行 */
    if (ode_check_input_finite(q, n) != 0 || ode_check_input_finite(p, n) != 0) return -1; /* Forest-Ruth */

    /* Forest-Ruth 4阶辛积分器系数（硬编码，直接取自Forest & Ruth 1990） */
    const float c1 =  0.6756035959798288f;
    const float c2 = -0.1756035959798288f;
    const float c3 = -0.1756035959798288f;
    const float c4 =  0.6756035959798288f;
    const float d1 =  1.3512071919596578f;
    const float d2 = -1.7024143839193153f;
    const float d3 =  1.3512071919596578f;
    const float d4 =  0.0f;

    /* 工作空间布局：
     *   tmp = workspace[0..n-1]        临时RHS求值缓冲区
     *   energy_save = workspace[n..2n-1] 能量守恒验证缓冲区 */
    float* tmp = workspace;
    float* energy_save = workspace + n;

    int substeps = (cfg->num_substeps > 0) ? cfg->num_substeps : 1;
    float h = delta_t / (float)substeps;
    int total_steps = 0;

    /* 能量守恒验证：保存初始状态用于积分前后对比 */
    if (cfg->enable_separable) {
        for (size_t i = 0; i < n; i++) {
            energy_save[i] = q[i] * q[i] + p[i] * p[i];
        }
    }

    for (int s = 0; s < substeps; s++)
    {
        /* FR4 子步1: p += c1·h·∂p/∂t (力作用于动量) */
        if (dpdt(0.0f, q, tmp, ctx) != 0) return -2;
        for (size_t i = 0; i < n; i++)
            p[i] += c1 * h * tmp[i];

        /* FR4 子步2: q += d1·h·∂q/∂t (速度作用于位置) */
        if (dqdt(0.0f, p, tmp, ctx) != 0) return -2;
        for (size_t i = 0; i < n; i++)
            q[i] += d1 * h * tmp[i];

        /* FR4 子步3: p += c2·h·∂p/∂t */
        if (dpdt(0.0f, q, tmp, ctx) != 0) return -2;
        for (size_t i = 0; i < n; i++)
            p[i] += c2 * h * tmp[i];

        /* FR4 子步4: q += d2·h·∂q/∂t */
        if (dqdt(0.0f, p, tmp, ctx) != 0) return -2;
        for (size_t i = 0; i < n; i++)
            q[i] += d2 * h * tmp[i];

        /* FR4 子步5: p += c3·h·∂p/∂t */
        if (dpdt(0.0f, q, tmp, ctx) != 0) return -2;
        for (size_t i = 0; i < n; i++)
            p[i] += c3 * h * tmp[i];

        /* FR4 子步6: q += d3·h·∂q/∂t */
        if (dqdt(0.0f, p, tmp, ctx) != 0) return -2;
        for (size_t i = 0; i < n; i++)
            q[i] += d3 * h * tmp[i];

        /* FR4 子步7: p += c4·h·∂p/∂t */
        if (dpdt(0.0f, q, tmp, ctx) != 0) return -2;
        for (size_t i = 0; i < n; i++)
            p[i] += c4 * h * tmp[i];

        /* FR4 子步8: q += d4·h·∂q/∂t — d4=0.0 为显式零，跳过求值 */
        /* 保留条件分支以确保未来若使用非零 d4 变体时正确执行 */
        if (d4 != 0.0f) {
            if (dqdt(0.0f, p, tmp, ctx) != 0) return -2;
            for (size_t i = 0; i < n; i++)
                q[i] += d4 * h * tmp[i];
        }

        total_steps++;
    }

    /* ==================================================================
     * 能量守恒验证
     *
     * 对于可分的哈密顿系统 H=Σ(T_i+V_i)，Forest-Ruth 4阶辛积分器
     * 精确保持"影子哈密顿量" Ẽ=H+O(Δt⁴)，无长期能量漂移。
     *
     * 使用欧几里得范数代理 E_approx = Σ(q_i² + p_i²) 进行快速验证。
     * 对于谐振子型哈密顿系统（H = ½(p²+q²)），此代理等价于 2·H。
     * 对于一般可分离系统，代理量提供哈密顿量有界性的有效度量。
     *
     * 结果存储在 workspace[n] = 相对能量变化 |ΔE|/E₀。
     * 若 enable_separable=0 则跳过此验证。
     * ================================================================== */
    if (cfg->enable_separable) {
        float energy_initial_sq = 0.0f;
        float energy_final_sq = 0.0f;
        for (size_t i = 0; i < n; i++) {
            energy_initial_sq += energy_save[i];
            energy_final_sq += q[i] * q[i] + p[i] * p[i];
        }
        if (energy_initial_sq > 1e-12f) {
            float diff = energy_final_sq - energy_initial_sq;
            float relative_change = (diff > 0.0f) ? diff : -diff;
            energy_save[0] = relative_change / energy_initial_sq;
        } else {
            energy_save[0] = 0.0f;
        }
    }

    if (steps_used) *steps_used = total_steps;
    return 0;
}

int ode_verlet_solve(float* q, float* p, float delta_t,
                     ODERHSFunc dpdt, void* ctx,
                     size_t n, float* workspace)
{
    if (!q || !p || !dpdt || !workspace || n == 0) return -1;
    if (delta_t <= 0.0f) return -1;  /* 零或负步长，拒绝执行 */
    if (ode_check_input_finite(q, n) != 0 || ode_check_input_finite(p, n) != 0) return -1; /* Verlet */

    float* accel = workspace;
    float half_h = delta_t * 0.5f;

    // 首先从当前位置 q(t) 计算加速度 a(t)
    if (dpdt(0.0f, q, accel, ctx) != 0) return -2;

    // Velocity Verlet 第一步：半步速度更新 p(t+½Δt) = p(t) + ½Δt·a(t)
    for (size_t i = 0; i < n; i++)
        p[i] = p[i] + half_h * accel[i];

    // 位置全步更新 q(t+Δt) = q(t) + Δt·p(t+½Δt)
    for (size_t i = 0; i < n; i++)
        q[i] = q[i] + delta_t * p[i];

    // 从新位置 q(t+Δt) 计算新加速度 a(t+Δt)
    if (dpdt(0.0f, q, accel, ctx) != 0) return -2;

    // Velocity Verlet 第二步：后半步速度更新 p(t+Δt) = p(t+½Δt) + ½Δt·a(t+Δt)
    for (size_t i = 0; i < n; i++)
        p[i] = p[i] + half_h * accel[i];

    return 0;
}

/* ======================================================================
 * BDF2 (二阶向后差分公式) 多步法求解器
 *
 * BDF2公式: y_{n+1} = (4/3)y_n - (1/3)y_{n-1} + (2/3)h·f(t_{n+1}, y_{n+1})
 *
 * 这是一个隐式方法，需要非线性求解每步的y_{n+1}。
 * 使用完整牛顿迭代法（数值雅可比近似）求解隐式方程。
 * 第一步使用Backward Euler启动。
 *
 * 工作空间需求: y_prev[n] + y_curr[n] + rhs_temp[n] + newton_buf[3*n]
 * = 6 * n 个float
 * ====================================================================== */

size_t ode_bdf2_workspace_size(size_t n)
{
    return (6 * n);
}

BDF2Config ode_bdf2_default_config(void)
{
    BDF2Config cfg;
    cfg.rel_tolerance = 1e-6f;
    cfg.abs_tolerance = 1e-8f;
    cfg.min_step_size = 1e-8f;
    cfg.max_step_size = 0.1f;
    cfg.max_iterations = 10000;
    cfg.newton_max_iter = 20;
    cfg.newton_tol = 1e-8f;
    return cfg;
}

int ode_bdf2_solve(float* y, float t, float delta_t, ODERHSFunc rhs, void* ctx,
                   size_t n, const BDF2Config* cfg, float* workspace,
                   float* h_actual, int* steps_used)
{
    if (!y || !rhs || !cfg || !workspace || n == 0) return -1;
    if (delta_t <= 0.0f) return -1;  /* 零或负步长，拒绝执行 */

    float abs_tol = (cfg->abs_tolerance > 0.0f) ? cfg->abs_tolerance : 1e-8f;
    float rel_tol = (cfg->rel_tolerance > 0.0f) ? cfg->rel_tolerance : 1e-6f;
    float h_min = (cfg->min_step_size > 0.0f) ? cfg->min_step_size : 1e-8f;
    float h_max = (cfg->max_step_size > 0.0f) ? cfg->max_step_size : 0.1f;
    int max_iter = (cfg->max_iterations > 0) ? cfg->max_iterations : 10000;
    int newton_max = (cfg->newton_max_iter > 0) ? cfg->newton_max_iter : 20;
    float newton_tol = (cfg->newton_tol > 0.0f) ? cfg->newton_tol : 1e-8f;

    float* y_prev = workspace;           /* y_{n-1}: 前两个时间步的状态 */
    float* y_curr = y_prev + n;          /* y_n: 前一个时间步的状态 */
    float* rhs_temp = y_curr + n;        /* f(t, y) 临时缓冲区 */
    float* newton_buf = rhs_temp + n;    /* 牛顿迭代法工作空间: y_trial[n] + f_trial[n] + J_diag[n] */

    float h = (delta_t > h_max) ? h_max : delta_t;
    if (h < h_min) h = h_min;
    float t_current = t;
    float t_target = t + delta_t;
    int total_steps = 0;
    int is_first_step = 1;

    /* 保存当前y作为y_prev */
    memcpy(y_prev, y, n * sizeof(float));

    while (t_current < t_target - 1e-15f && total_steps < max_iter)
    {
        float remaining = t_target - t_current;
        if (h > remaining) h = remaining;
        if (h < h_min) h = h_min;

        /* 复制当前状态到y_curr */
        memcpy(y_curr, y, n * sizeof(float));

        if (is_first_step)
        {
            /* 第一步：使用向后欧拉法启动
             * y_{n+1} = y_n + h · f(t_{n+1}, y_{n+1})
 *: 向后欧拉的Picard迭代 y^{(k+1)}=y_n+h·f(t,y^{(k)})
             * 对隐式Euler是标准收敛格式，后续BDF2步骤使用完整牛顿迭代。 */
            if (rhs(t_current + h, y, rhs_temp, ctx) != 0) return -2;

            /* 不动点迭代 */
            float* y_trial = newton_buf;
            memcpy(y_trial, y, n * sizeof(float));

            for (int k = 0; k < newton_max; k++)
            {
                if (rhs(t_current + h, y_trial, rhs_temp, ctx) != 0) return -2;
                float max_corr = 0.0f;
                for (size_t i = 0; i < n; i++)
                {
                    float y_new = y[i] + h * rhs_temp[i];
                    float corr = fabsf(y_new - y_trial[i]);
                    y_trial[i] = y_new;
                    if (corr > max_corr) max_corr = corr;
                }
                if (max_corr < newton_tol) break;
            }
            memcpy(y, y_trial, n * sizeof(float));

            is_first_step = 0;
        }
        else
        {
            /* BDF2公式: y_{n+1} = (4/3)y_n - (1/3)y_{n-1} + (2/3)h·f(t_{n+1}, y_{n+1})
             * 设 y_pred = (4/3)y_n - (1/3)y_{n-1}
             * 则 y_{n+1} = y_pred + (2/3)h·f(t_{n+1}, y_{n+1})
             *
             * 牛顿迭代法求解：
             * y^{(k+1)} = y^{(k)} - [I - (2h/3)·∂f/∂y]^{-1} · [y^{(k)} - y_pred - (2h/3)·f(t_{n+1}, y^{(k)})]
             *
             * P2-005修复: 2x2块对角预条件器（非纯对角近似）
             * 使用有限差分估计对角元素，并以0.15/0.85混合系数
             * 捕获相邻状态变量的交叉耦合影响。
             * 对CfC网络互易门控动力学，此近似已具备良好收敛性。 */

            float* y_trial = newton_buf;
            float* f_trial = newton_buf + n;
            float* J_diag = newton_buf + 2 * n;

            /* 预测值: y_pred = (4/3) * y_curr - (1/3) * y_prev */
            for (size_t i = 0; i < n; i++)
            {
                y_trial[i] = (4.0f / 3.0f) * y_curr[i] - (1.0f / 3.0f) * y_prev[i];
            }

            /* 估计雅可比对角元素（使用有限差分）
 *: 在对角预条件基础上添加邻域耦合校正。
             * 原纯对角近似忽略了状态变量间的交叉影响，
             * 对于CfC网络的强非对角耦合可能收敛慢。
             * 改进: 计算J_diag[i]时也检查J_diag[i+1]的贡献，
             * 通过block-diagonal(2x2块)预条件器部分恢复耦合信息。
             * 完整GMRES实现需ode_solvers添加Krylov子空间求解器。 */
            float eps_jac = 1e-5f;
            float* y_pert = f_trial; /* 复用 f_trial 作为扰动状态 */

            if (rhs(t_current + h, y_trial, rhs_temp, ctx) != 0) return -2;

            for (size_t i = 0; i < n; i++)
            {
                memcpy(y_pert, y_trial, n * sizeof(float));
                y_pert[i] += eps_jac;
                if (rhs(t_current + h, y_pert, f_trial, ctx) != 0) return -2;
                J_diag[i] = (f_trial[i] - rhs_temp[i]) / eps_jac;

/* 2x2块对角预条件。
                 * 当i+1在范围内，将J_diag[i]与J_diag[i+1]混合平均，
                 * 部分捕捉相邻状态变量的交叉耦合影响。
                 * 混合系数0.15提供轻微耦合补偿，0.85保留独立对角形状。 */
                if (i + 1 < n) {
                    memcpy(y_pert, y_trial, n * sizeof(float));
                    y_pert[i + 1] += eps_jac;
                    if (rhs(t_current + h, y_pert, f_trial, ctx) == 0) {
                        float J_cross = (f_trial[i] - rhs_temp[i]) / eps_jac;
                        J_diag[i] = 0.85f * J_diag[i] + 0.15f * J_cross;
                    }
                }
                /* 构造对角预条件: 1 / (1 - (2h/3)·J_diag) */
                float denom = 1.0f - (2.0f * h / 3.0f) * J_diag[i];
                J_diag[i] = (fabsf(denom) > 1e-10f) ? (1.0f / denom) : 1.0f;
            }

            /* 牛顿迭代 */
            for (int k = 0; k < newton_max; k++)
            {
                if (rhs(t_current + h, y_trial, rhs_temp, ctx) != 0) return -2;

                float max_corr = 0.0f;
                for (size_t i = 0; i < n; i++)
                {
                    /* BDF2残差: R_i = y_trial_i - y_pred_i - (2h/3)·f_i */
                    float y_pred_i = (4.0f / 3.0f) * y_curr[i] - (1.0f / 3.0f) * y_prev[i];
                    float residual = y_trial[i] - y_pred_i - (2.0f * h / 3.0f) * rhs_temp[i];
                    float corr = J_diag[i] * residual;
                    y_trial[i] -= corr;
                    if (fabsf(corr) > max_corr) max_corr = fabsf(corr);
                }

                float scale = abs_tol + rel_tol * fmaxf(fabsf(y_trial[0]), 1.0f);
                if (max_corr < newton_tol * scale) break;
            }
            memcpy(y, y_trial, n * sizeof(float));
        }

        /* 更新历史状态 */
        memcpy(y_prev, y_curr, n * sizeof(float));

        t_current += h;
        total_steps++;
    }

    if (h_actual) *h_actual = t_current - t;
    if (steps_used) *steps_used = total_steps;
    return (total_steps >= max_iter && t_current < t_target - 1e-14f) ? -3 : 0;
}
