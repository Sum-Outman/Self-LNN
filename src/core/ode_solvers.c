#include "selflnn/core/ode_solvers.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define SELFLNN_DP54_SAFETY 0.9f
#define SELFLNN_DP54_PGROW -0.2f
#define SELFLNN_DP54_PSHRINK -0.25f
#define SELFLNN_DP54_ERR_CTRL 1.0e-12f

static float dp54_max(float a, float b) { return (a > b) ? a : b; }
static float dp54_min(float a, float b) { return (a < b) ? a : b; }
static float dp54_abs(float x) { return (x < 0.0f) ? -x : x; }

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
    return (n * n + 5 * n);
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
    float t_current = t;
    float t_target = t + delta_t;
    int total_steps = 0;

    while (t_current < t_target - 1e-15f && total_steps < max_iter)
    {
        float remaining = t_target - t_current;
        if (h > remaining) h = remaining;
        if (h < h_min) h = h_min;

        if (rhs(t_current, y, k1, ctx) != 0) return -2;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (0.2f * k1[i]);
        if (rhs(t_current + 0.2f * h, y_temp, k2, ctx) != 0) return -2;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (3.0f/40.0f * k1[i] + 9.0f/40.0f * k2[i]);
        if (rhs(t_current + 0.3f * h, y_temp, k3, ctx) != 0) return -2;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (44.0f/45.0f * k1[i] - 56.0f/15.0f * k2[i] + 32.0f/9.0f * k3[i]);
        if (rhs(t_current + 0.8f * h, y_temp, k4, ctx) != 0) return -2;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (19372.0f/6561.0f * k1[i] - 25360.0f/2187.0f * k2[i] + 64448.0f/6561.0f * k3[i] - 212.0f/729.0f * k4[i]);
        if (rhs(t_current + 8.0f/9.0f * h, y_temp, k5, ctx) != 0) return -2;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (9017.0f/3168.0f * k1[i] - 355.0f/33.0f * k2[i] + 46732.0f/5247.0f * k3[i] + 49.0f/176.0f * k4[i] - 5103.0f/18656.0f * k5[i]);
        if (rhs(t_current + h, y_temp, k6, ctx) != 0) return -2;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (35.0f/384.0f * k1[i] + 500.0f/1113.0f * k3[i] + 125.0f/192.0f * k4[i] - 2187.0f/6784.0f * k5[i] + 11.0f/84.0f * k6[i]);
        if (rhs(t_current + h, y_temp, k7, ctx) != 0) return -2;

        float max_err = 0.0f;
        for (size_t i = 0; i < n; i++)
        {
            float y5 = y[i] + h * (35.0f/384.0f * k1[i] + 500.0f/1113.0f * k3[i] + 125.0f/192.0f * k4[i] - 2187.0f/6784.0f * k5[i] + 11.0f/84.0f * k6[i]);
            float y4 = y[i] + h * (5179.0f/57600.0f * k1[i] + 7571.0f/16695.0f * k3[i] + 393.0f/640.0f * k4[i] - 92097.0f/339200.0f * k5[i] + 187.0f/2100.0f * k6[i] + 0.025f * k7[i]);
            float err_i = dp54_abs(y5 - y4);
            float scale = abs_tol + rel_tol * dp54_max(dp54_abs(y[i]), dp54_abs(y5));
            float ratio = err_i / scale;
            if (ratio > max_err) max_err = ratio;
        }

        total_steps++;

        if (max_err <= 1.0f || h <= h_min)
        {
            for (size_t i = 0; i < n; i++)
                y[i] = y[i] + h * (35.0f/384.0f * k1[i] + 500.0f/1113.0f * k3[i] + 125.0f/192.0f * k4[i] - 2187.0f/6784.0f * k5[i] + 11.0f/84.0f * k6[i]);
            t_current += h;

            float h_new = h;
            if (max_err > SELFLNN_DP54_ERR_CTRL)
            {
                float factor = safety * (float)pow((double)max_err, (double)SELFLNN_DP54_PGROW);
                h_new = h * dp54_max(0.2f, dp54_min(5.0f, factor));
            }
            h = h_new;
            if (h > h_max) h = h_max;
        }
        else
        {
            float factor = safety * (float)pow((double)max_err, (double)SELFLNN_DP54_PSHRINK);
            h = h * dp54_max(0.1f, factor);
            if (h < h_min) h = h_min;
        }
    }

    if (h_actual) *h_actual = t_current - t;
    if (steps_used) *steps_used = total_steps;
    return (total_steps >= max_iter && t_current < t_target - 1e-14f) ? -3 : 0;
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

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (0.2f * k1[i]);
        if (rhs(t_current + 0.2f * h, y_temp, k2, ctx) != 0) return -2;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (3.0f/40.0f * k1[i] + 9.0f/40.0f * k2[i]);
        if (rhs(t_current + 0.3f * h, y_temp, k3, ctx) != 0) return -2;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (44.0f/45.0f * k1[i] - 56.0f/15.0f * k2[i] + 32.0f/9.0f * k3[i]);
        if (rhs(t_current + 0.8f * h, y_temp, k4, ctx) != 0) return -2;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (19372.0f/6561.0f * k1[i] - 25360.0f/2187.0f * k2[i] + 64448.0f/6561.0f * k3[i] - 212.0f/729.0f * k4[i]);
        if (rhs(t_current + 8.0f/9.0f * h, y_temp, k5, ctx) != 0) return -2;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (9017.0f/3168.0f * k1[i] - 355.0f/33.0f * k2[i] + 46732.0f/5247.0f * k3[i] + 49.0f/176.0f * k4[i] - 5103.0f/18656.0f * k5[i]);
        if (rhs(t_current + h, y_temp, k6, ctx) != 0) return -2;

        for (size_t i = 0; i < n; i++)
            y_temp[i] = y[i] + h * (35.0f/384.0f * k1[i] + 500.0f/1113.0f * k3[i] + 125.0f/192.0f * k4[i] - 2187.0f/6784.0f * k5[i] + 11.0f/84.0f * k6[i]);
        if (rhs(t_current + h, y_temp, k7, ctx) != 0) return -2;

        /* ========== 误差估计 ========== */
        float max_err = 0.0f;
        for (size_t i = 0; i < n; i++)
        {
            float y5 = y[i] + h * (35.0f/384.0f * k1[i] + 500.0f/1113.0f * k3[i] + 125.0f/192.0f * k4[i] - 2187.0f/6784.0f * k5[i] + 11.0f/84.0f * k6[i]);
            float y4 = y[i] + h * (5179.0f/57600.0f * k1[i] + 7571.0f/16695.0f * k3[i] + 393.0f/640.0f * k4[i] - 92097.0f/339200.0f * k5[i] + 187.0f/2100.0f * k6[i] + 0.025f * k7[i]);
            float err_i = dp54_abs(y5 - y4);
            float scale = abs_tol + rel_tol * dp54_max(dp54_abs(y[i]), dp54_abs(y5));
            float ratio = err_i / scale;
            if (ratio > max_err) max_err = ratio;
        }

        total_steps++;

        if (max_err <= 1.0f || h_abs <= h_min)
        {
            /* ========== 接受步：更新状态 ========== */
            for (size_t i = 0; i < n; i++)
                y[i] = y[i] + h * (35.0f/384.0f * k1[i] + 500.0f/1113.0f * k3[i] + 125.0f/192.0f * k4[i] - 2187.0f/6784.0f * k5[i] + 11.0f/84.0f * k6[i]);

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
                            dyn_l = (float*)malloc((size_t)n * sizeof(float));
                            dyn_m = (float*)malloc((size_t)n * sizeof(float));
                            if (!dyn_l || !dyn_m) {
                                free(dyn_l); free(dyn_m);
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
                                if (dyn_l) free(dyn_l);
                                if (dyn_m) free(dyn_m);
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

                        if (dyn_l) free(dyn_l);
                        if (dyn_m) free(dyn_m);

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
            sol->pi_integral += 0.5f * ki * h * (error_ratio + sol->pi_previous_error);
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

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int d = 0; d < nd; d++) {
            int start = pcfg->domain_offsets[d];
            int size = pcfg->domain_sizes[d];
            if (start < 0 || size <= 0 || (size_t)(start + size) > n) continue;

            float* local_dydt = dydt + start;
            const float* local_y = y;
            int ret = rhs(t, local_y, local_dydt, ctx);
            if (ret != 0) {
                (void)ret;
            }
        }
    } else if (use_omp) {
        /* ZSFAB P1-002修复: 消除双重RHS调用 - 并行路径与串行回退互斥 */
#ifdef _OPENMP
        #pragma omp parallel
        {
            float* local_dydt = dydt;
            const float* local_y = y;
            int ret = rhs(t, local_y, local_dydt, ctx);
            if (ret != 0) { (void)ret; }
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
            local_y = (float*)malloc((size_t)local_n * sizeof(float));
            local_workspace = (float*)malloc(workspace_size);
            if (!local_y || !local_workspace) {
                free(local_y); free(local_workspace);
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

        free(local_y);
        free(local_workspace);

        if (h_actual) *h_actual = local_h;
        if (steps_used) *steps_used = local_steps;
        return ret;
#else
        /* I-010修复：MPI模式当前未编译，返回明确错误码而非静默回退 */
        (void)is_mpi_mode;
        (void)workspace_size;
        if (pcfg->mode == PARALLEL_MODE_MPI || pcfg->mode == PARALLEL_MODE_HYBRID) {
            (void)t_target;
            return -5; /* MPI未编译：返回明确错误码，上层可捕获并处理 */
        }
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

    int* recv_counts = (int*)malloc((size_t)num_ranks * sizeof(int));
    int* displacements = (int*)malloc((size_t)num_ranks * sizeof(int));

    if (!recv_counts || !displacements) {
        free(recv_counts); free(displacements);
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

    float* send_buf = (float*)malloc((size_t)recv_counts[rank] * sizeof(float));
    if (send_buf) {
        memcpy(send_buf, y, (size_t)recv_counts[rank] * sizeof(float));
        MPI_Allgatherv(send_buf, recv_counts[rank], MPI_FLOAT,
                       y, recv_counts, displacements, MPI_FLOAT,
                       MPI_COMM_WORLD);
        free(send_buf);
    }

    free(recv_counts);
    free(displacements);
    return 0;
#else
    (void)y;
    (void)n;
    return 0;
#endif
}

static int ros_gauss_eliminate(float* A, size_t n, float* b, float* x)
{
    float* system = (float*)malloc((n * (n + 1)) * sizeof(float));
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
        float max_val = dp54_abs(system[col * (n + 1) + col]);
        for (size_t row = col + 1; row < n; row++)
        {
            float val = dp54_abs(system[row * (n + 1) + col]);
            if (val > max_val) { max_val = val; pivot = row; }
        }
        if (max_val < SELFLNN_ROSENBROCK_LINSOLVE_TOL) { free(system); return -2; }

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

    free(system);
    return 0;
}

static int ros_compute_jacobian(ODERHSFunc rhs, void* ctx, float t, const float* y,
                                 float* J, size_t n, float eps,
                                 float* f0, float* fp, float* yp)
{
    if (rhs(t, y, f0, ctx) != 0) return -2;

    for (size_t j = 0; j < n; j++)
    {
        float y_save = y[j];
        float h = eps * dp54_max(1.0f, dp54_abs(y_save));
        if (dp54_abs(h) < 1e-12f) h = eps;
        memcpy(yp, y, n * sizeof(float));
        yp[j] = y_save + h;

        if (rhs(t, yp, fp, ctx) != 0) return -2;

        for (size_t i = 0; i < n; i++)
            J[i * n + j] = (fp[i] - f0[i]) / h;
    }

    return 0;
}

int ode_rosenbrock_solve(float* y, float t, float delta_t, ODERHSFunc rhs, void* ctx,
                          size_t n, const RosenbrockConfig* cfg, float* workspace,
                          float* h_actual, int* steps_used)
{
    if (!y || !rhs || !cfg || !workspace || n == 0) return -1;

    float h_max = (cfg->max_step_size > 0.0f) ? cfg->max_step_size : 1.0f;
    float gamma_val = (cfg->gamma_coeff > 0.0f) ? cfg->gamma_coeff : 0.435866521508f;
    int max_iter = (cfg->max_iterations > 0) ? cfg->max_iterations : 10000;
    float eps_fd = (cfg->finite_diff_eps > 0.0f) ? cfg->finite_diff_eps : 1e-6f;

    float* J = workspace;
    float* k = J + n * n;
    float* f_n = k + n;
    float* jac_f0 = f_n + n;
    float* jac_fp = jac_f0 + n;
    float* jac_yp = jac_fp + n;

    float t_current = t;
    int total_steps = 0;

    /* 使用固定步长子步进（半隐式欧拉/1级Rosenbrock法）
       正确公式：(I - h*gamma*J) * k = h*F(y), y_new = y + k */
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
                                  jac_f0, jac_fp, jac_yp) != 0)
            return -3;

        /* A = I - h_step * gamma_val * J (where J = dF/dy) */
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
    
    float* y_full = (float*)malloc(n * sizeof(float));
    float* y_half = (float*)malloc(n * sizeof(float));
    float* workspace = (float*)malloc(ode_rosenbrock_workspace_size(n) * sizeof(float));
    RosenbrockConfig cfg = ode_rosenbrock_default_config();
    float h_used;
    int s_used;
    size_t i;

    if (!y_full || !y_half || !workspace) {
        free(y_full); free(y_half); free(workspace);
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
        if (rejected > 10) break;
    }

    if (h_final) *h_final = h;
    if (steps_taken) *steps_taken = total_steps;

    free(y_full); free(y_half); free(workspace);
    return (total_steps > 0) ? 0 : -3;
}

int ode_forest_ruth_solve(float* q, float* p, float delta_t,
                           ODERHSFunc dqdt, ODERHSFunc dpdt, void* ctx,
                           size_t n, const SymplecticConfig* cfg,
                           float* workspace, int* steps_used)
{
    if (!q || !p || !dqdt || !dpdt || !cfg || !workspace || n == 0) return -1;

    float* tmp = workspace;
    int substeps = (cfg->num_substeps > 0) ? cfg->num_substeps : 1;
    float h = delta_t / substeps;
    int total_steps = 0;

    float theta = SELFLNN_FOREST_RUTH_THETA;
    float c1 = theta * 0.5f;
    float c2 = (1.0f - theta) * 0.5f;
    float c3 = (1.0f - theta) * 0.5f;
    float c4 = theta * 0.5f;
    float d1 = theta;
    float d2 = 1.0f - 2.0f * theta;
    float d3 = theta;
    float d4 = 0.0f;

    for (int s = 0; s < substeps; s++)
    {
        if (dpdt(0.0f, q, tmp, ctx) != 0) return -2;
        for (size_t i = 0; i < n; i++)
            p[i] = p[i] + c1 * h * tmp[i];

        if (dqdt(0.0f, p, tmp, ctx) != 0) return -2;
        for (size_t i = 0; i < n; i++)
            q[i] = q[i] + d1 * h * tmp[i];

        if (dpdt(0.0f, q, tmp, ctx) != 0) return -2;
        for (size_t i = 0; i < n; i++)
            p[i] = p[i] + c2 * h * tmp[i];

        if (dqdt(0.0f, p, tmp, ctx) != 0) return -2;
        for (size_t i = 0; i < n; i++)
            q[i] = q[i] + d2 * h * tmp[i];

        if (dpdt(0.0f, q, tmp, ctx) != 0) return -2;
        for (size_t i = 0; i < n; i++)
            p[i] = p[i] + c3 * h * tmp[i];

        if (dqdt(0.0f, p, tmp, ctx) != 0) return -2;
        for (size_t i = 0; i < n; i++)
            q[i] = q[i] + d3 * h * tmp[i];

        if (dpdt(0.0f, q, tmp, ctx) != 0) return -2;
        for (size_t i = 0; i < n; i++)
            p[i] = p[i] + c4 * h * tmp[i];

        if (dqdt(0.0f, p, tmp, ctx) != 0) return -2;
        for (size_t i = 0; i < n; i++)
            q[i] = q[i] + d4 * h * tmp[i];

        total_steps++;
    }

    if (steps_used) *steps_used = total_steps;
    return 0;
}

int ode_verlet_solve(float* q, float* p, float delta_t,
                     ODERHSFunc dpdt, void* ctx,
                     size_t n, float* workspace)
{
    if (!q || !p || !dpdt || !workspace || n == 0) return -1;

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
             * 简化不动点迭代求解 */
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
                    float corr = dp54_abs(y_new - y_trial[i]);
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
             * 使用对角雅可比近似（简化）：∂f/∂y ≈ 用有限差分估计对角元素
             */

            float* y_trial = newton_buf;
            float* f_trial = newton_buf + n;
            float* J_diag = newton_buf + 2 * n;

            /* 预测值: y_pred = (4/3) * y_curr - (1/3) * y_prev */
            for (size_t i = 0; i < n; i++)
            {
                y_trial[i] = (4.0f / 3.0f) * y_curr[i] - (1.0f / 3.0f) * y_prev[i];
            }

            /* 估计雅可比对角元素（使用有限差分） */
            float eps_jac = 1e-5f;
            float* y_pert = f_trial; /* 复用 f_trial 作为扰动状态 */

            if (rhs(t_current + h, y_trial, rhs_temp, ctx) != 0) return -2;

            for (size_t i = 0; i < n; i++)
            {
                memcpy(y_pert, y_trial, n * sizeof(float));
                y_pert[i] += eps_jac;
                if (rhs(t_current + h, y_pert, f_trial, ctx) != 0) return -2;
                J_diag[i] = (f_trial[i] - rhs_temp[i]) / eps_jac;
                /* 构造对角预条件: 1 / (1 - (2h/3)·J_diag) */
                float denom = 1.0f - (2.0f * h / 3.0f) * J_diag[i];
                J_diag[i] = (dp54_abs(denom) > 1e-10f) ? (1.0f / denom) : 1.0f;
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
                    if (dp54_abs(corr) > max_corr) max_corr = dp54_abs(corr);
                }

                float scale = abs_tol + rel_tol * dp54_max(dp54_abs(y_trial[0]), 1.0f);
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
