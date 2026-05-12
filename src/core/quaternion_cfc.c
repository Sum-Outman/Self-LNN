/**
 * @file quaternion_cfc.c
 * @brief 四元数CfC单元实现
 * 
 * 将四元数表示与CfC动力学深度集成，实现四维超复杂连续时间动态系统，
 * 支持旋转不变性和四维状态演化。
 */

#include "selflnn/core/quaternion_cfc.h"
#include "selflnn/core/quaternion_lnn.h"
#include "selflnn/core/cfc_cell.h"
#include "selflnn/core/cfc.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/ode_solvers.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* K-114: Box-Muller线程安全 —— static变量无锁保护,并发调用产生错误分布 */
static float gaussian_random(float mean, float stddev) {
    float u, v, s;
    do {
        u = rng_uniform(0.0f, 1.0f) * 2.0f - 1.0f;
        v = rng_uniform(0.0f, 1.0f) * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);
    
    s = sqrtf(-2.0f * logf(s) / s);
    return mean + stddev * u * s;
}

/**
 * @brief 均匀随机数生成器（静态辅助函数）
 */
static float uniform_random(float min, float max) {
    return min + rng_uniform(0.0f, 1.0f) * (max - min);
}

/**
 * @brief 四元数CfC单元内部结构
 */
struct QuaternionCfcCell {
    Quaternion state;                /**< 四元数状态 (q = w + xi + yj + zk) */
    Quaternion input_gain;           /**< 输入增益四元数 */
    Quaternion feedback_gain;        /**< 反馈增益四元数 */
    Quaternion time_constant;        /**< 时间常数四元数（各分量可独立） */
    Quaternion output_gain;          /**< 输出增益四元数 */
    
    float noise_std;                 /**< 噪声标准差 */
    int enable_adaptation;           /**< 是否启用参数自适应 */
    int enable_evolution;            /**< 是否启用演化 */
    
    // 内部状态
    Quaternion last_input;           /**< 上一次输入 */
    Quaternion integrated_error;     /**< 积分误差 */
    Quaternion adaptive_params;      /**< 自适应参数 */
    
    // 性能统计
    uint64_t update_count;           /**< 更新次数 */
    float average_update_time_ns;    /**< 平均更新时间（纳秒） */
};

/**
 * @brief 创建四元数CfC单元
 */
QuaternionCfcCell* quaternion_cfc_cell_create(const QuaternionCfcConfig* config) {
    if (!config) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建四元数CfC单元：配置参数为空");
        return NULL;
    }
    
    QuaternionCfcCell* cell = (QuaternionCfcCell*)safe_calloc(1, sizeof(QuaternionCfcCell));
    if (!cell) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建四元数CfC单元：内存分配失败");
        return NULL;
    }
    
    cell->state = config->initial_state;
    cell->input_gain = config->input_gain;
    cell->feedback_gain = config->feedback_gain;
    cell->time_constant = config->time_constant;
    cell->output_gain = config->output_gain;
    
    cell->noise_std = config->noise_std;
    cell->enable_adaptation = config->enable_adaptation;
    cell->enable_evolution = config->enable_evolution;
    
    cell->last_input.w = 0.0f; cell->last_input.x = 0.0f; cell->last_input.y = 0.0f; cell->last_input.z = 0.0f;
    cell->integrated_error.w = 0.0f; cell->integrated_error.x = 0.0f; cell->integrated_error.y = 0.0f; cell->integrated_error.z = 0.0f;
    cell->adaptive_params.w = 1.0f; cell->adaptive_params.x = 0.0f; cell->adaptive_params.y = 0.0f; cell->adaptive_params.z = 0.0f;
    
    cell->update_count = 0;
    cell->average_update_time_ns = 0.0f;
    
    return cell;
}

/**
 * @brief 销毁四元数CfC单元
 */
void quaternion_cfc_cell_destroy(QuaternionCfcCell* cell) {
    if (!cell) {
        return;
    }
    
    safe_free((void**)&cell);
}

/**
 * @brief 使用CfC闭式解更新四元数状态
 * 
 * 对每个四元数分量应用CfC闭式解:
 *   new = prev * exp(-dt/τ) + (1 - exp(-dt/τ)) * drive
 * 
 * 这是四元数化的CfC闭式解：每个分量（w,x,y,z）独立演化，
 * 具有各自的时间常数，最终输出通过四元数乘法保持旋转结构。
 */
int quaternion_cfc_closed_form_update(Quaternion* state, const Quaternion* drive,
                                       const Quaternion* time_constant, float dt,
                                       Quaternion* output) {
    if (!state || !drive || !time_constant || dt <= 0.0f) {
        return -1;
    }

    float exp_w = expf(-dt / fmaxf(time_constant->w, 1e-6f));
    float exp_x = expf(-dt / fmaxf(time_constant->x, 1e-6f));
    float exp_y = expf(-dt / fmaxf(time_constant->y, 1e-6f));
    float exp_z = expf(-dt / fmaxf(time_constant->z, 1e-6f));

    state->w = state->w * exp_w + (1.0f - exp_w) * drive->w;
    state->x = state->x * exp_x + (1.0f - exp_x) * drive->x;
    state->y = state->y * exp_y + (1.0f - exp_y) * drive->y;
    state->z = state->z * exp_z + (1.0f - exp_z) * drive->z;

    float norm = sqrtf(state->w * state->w + state->x * state->x +
                       state->y * state->y + state->z * state->z);
    if (norm > 1e-8f) {
        float inv = 1.0f / norm;
        state->w *= inv; state->x *= inv;
        state->y *= inv; state->z *= inv;
    }

    if (output) *output = *state;
    return 0;
}

/**
 * @brief 更新四元数CfC单元状态（增强版）
 * 
 * 使用CfC闭式解替代简单欧拉积分。
 * solver_type=0时默认用闭式解，其他类型调用标准ODE求解器。
 */
int quaternion_cfc_cell_update(QuaternionCfcCell* cell, const Quaternion* input, 
                               float dt, Quaternion* output) {
    if (!cell || !input || dt <= 0.0f) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "更新四元数CfC单元：参数无效");
        return -1;
    }
    
    cell->last_input = *input;
    
    Quaternion input_term = quaternion_multiply(&cell->input_gain, input);
    Quaternion feedback_term = quaternion_multiply(&cell->feedback_gain, &cell->state);
    Quaternion drive = quaternion_add(&input_term, &feedback_term);
    
    /* CfC闭式解 */
    int ret = quaternion_cfc_closed_form_update(&cell->state, &drive,
                                                 &cell->time_constant, dt, NULL);
    if (ret != 0) return ret;
    
    if (cell->noise_std > 0.0f) {
        cell->state.w += gaussian_random(0.0f, cell->noise_std) * dt;
        cell->state.x += gaussian_random(0.0f, cell->noise_std) * dt;
        cell->state.y += gaussian_random(0.0f, cell->noise_std) * dt;
        cell->state.z += gaussian_random(0.0f, cell->noise_std) * dt;
    }
    
    if (output) {
        *output = quaternion_multiply(&cell->output_gain, &cell->state);
    }
    
    cell->update_count++;
    return 0;
}

/**
 * @brief 获取四元数CfC单元状态
 */
const Quaternion* quaternion_cfc_cell_get_state(const QuaternionCfcCell* cell) {
    if (!cell) {
        return NULL;
    }
    
    return &cell->state;
}

/**
 * @brief 设置四元数CfC单元状态
 */
int quaternion_cfc_cell_set_state(QuaternionCfcCell* cell, const Quaternion* state) {
    if (!cell || !state) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "设置四元数CfC单元状态：参数无效");
        return -1;
    }
    
    cell->state = *state;
    return 0;
}

/**
 * @brief 配置四元数CfC单元参数
 */
int quaternion_cfc_cell_configure(QuaternionCfcCell* cell, const QuaternionCfcConfig* config) {
    if (!cell || !config) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "配置四元数CfC单元：参数无效");
        return -1;
    }
    
    cell->input_gain = config->input_gain;
    cell->feedback_gain = config->feedback_gain;
    cell->time_constant = config->time_constant;
    cell->output_gain = config->output_gain;
    cell->noise_std = config->noise_std;
    cell->enable_adaptation = config->enable_adaptation;
    cell->enable_evolution = config->enable_evolution;
    
    return 0;
}

/**
 * @brief 执行四元数CfC单元自适应
 */
int quaternion_cfc_cell_adapt(QuaternionCfcCell* cell, const Quaternion* error, float learning_rate) {
    if (!cell || !error || learning_rate <= 0.0f) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "四元数CfC单元自适应：参数无效");
        return -1;
    }
    
    if (!cell->enable_adaptation) {
        return 0;  // 自适应未启用
    }
    
    // 更新积分误差：∫e dt
    Quaternion new_integrated_error;
    new_integrated_error.w = cell->integrated_error.w + error->w;
    new_integrated_error.x = cell->integrated_error.x + error->x;
    new_integrated_error.y = cell->integrated_error.y + error->y;
    new_integrated_error.z = cell->integrated_error.z + error->z;
    
    cell->integrated_error = new_integrated_error;
    
    // 自适应调整时间常数：τ = τ₀ * exp(-α * ∫e dt)
    // 确保时间常数保持正数
    float alpha = learning_rate * 0.01f;
    
    cell->time_constant.w = cell->time_constant.w * expf(-alpha * cell->integrated_error.w);
    cell->time_constant.x = cell->time_constant.x * expf(-alpha * cell->integrated_error.x);
    cell->time_constant.y = cell->time_constant.y * expf(-alpha * cell->integrated_error.y);
    cell->time_constant.z = cell->time_constant.z * expf(-alpha * cell->integrated_error.z);
    
    // 限制时间常数范围
    float min_tau = 0.001f;
    float max_tau = 10.0f;
    
    cell->time_constant.w = fmaxf(min_tau, fminf(max_tau, cell->time_constant.w));
    cell->time_constant.x = fmaxf(min_tau, fminf(max_tau, cell->time_constant.x));
    cell->time_constant.y = fmaxf(min_tau, fminf(max_tau, cell->time_constant.y));
    cell->time_constant.z = fmaxf(min_tau, fminf(max_tau, cell->time_constant.z));
    
    return 0;
}

/**
 * @brief 执行四元数CfC单元演化
 */
int quaternion_cfc_cell_evolve(QuaternionCfcCell* cell, float mutation_rate) {
    if (!cell || mutation_rate < 0.0f) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "四元数CfC单元演化：参数无效");
        return -1;
    }
    
    if (!cell->enable_evolution) {
        return 0;  // 演化未启用
    }
    
    // 对增益参数添加随机突变
    if (mutation_rate > 0.0f) {
        // 输入增益突变
        cell->input_gain.w += gaussian_random(0.0f, mutation_rate);
        cell->input_gain.x += gaussian_random(0.0f, mutation_rate);
        cell->input_gain.y += gaussian_random(0.0f, mutation_rate);
        cell->input_gain.z += gaussian_random(0.0f, mutation_rate);
        
        // 反馈增益突变
        cell->feedback_gain.w += gaussian_random(0.0f, mutation_rate * 0.1f);
        cell->feedback_gain.x += gaussian_random(0.0f, mutation_rate * 0.1f);
        cell->feedback_gain.y += gaussian_random(0.0f, mutation_rate * 0.1f);
        cell->feedback_gain.z += gaussian_random(0.0f, mutation_rate * 0.1f);
        
        // 输出增益突变
        cell->output_gain.w += gaussian_random(0.0f, mutation_rate * 0.5f);
        cell->output_gain.x += gaussian_random(0.0f, mutation_rate * 0.5f);
        cell->output_gain.y += gaussian_random(0.0f, mutation_rate * 0.5f);
        cell->output_gain.z += gaussian_random(0.0f, mutation_rate * 0.5f);
    }
    
    return 0;
}

/**
 * @brief 计算四元数CfC单元旋转不变性度量
 */
float quaternion_cfc_cell_rotation_invariance(const QuaternionCfcCell* cell, 
                                              const Quaternion* rotation) {
    if (!cell || !rotation) {
        return 0.0f;
    }
    
    // 应用旋转：q_rotated = r ⊗ q ⊗ r⁻¹
    Quaternion r_conj = quaternion_conjugate(rotation);
    
    Quaternion temp1 = quaternion_multiply(rotation, &cell->state);
    Quaternion temp2 = quaternion_multiply(&temp1, &r_conj);
    
    // 计算旋转前后状态差异
    Quaternion diff = quaternion_subtract(&cell->state, &temp2);
    
    // 计算差异范数
    float diff_norm = sqrtf(diff.w * diff.w + diff.x * diff.x + 
                           diff.y * diff.y + diff.z * diff.z);
    
    // 旋转不变性分数：1 - 差异范数
    float invariance = 1.0f - fminf(1.0f, diff_norm);
    
    return invariance;
}

/**
 * @brief 重置四元数CfC单元状态
 */
int quaternion_cfc_cell_reset(QuaternionCfcCell* cell) {
    if (!cell) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "重置四元数CfC单元：参数无效");
        return -1;
    }
    
    // 重置状态为零四元数
    cell->state.w = 0.0f; cell->state.x = 0.0f; cell->state.y = 0.0f; cell->state.z = 0.0f;
    
    // 重置积分误差
    cell->integrated_error.w = 0.0f; cell->integrated_error.x = 0.0f; cell->integrated_error.y = 0.0f; cell->integrated_error.z = 0.0f;
    
    // 重置性能统计
    cell->update_count = 0;
    cell->average_update_time_ns = 0.0f;

    return 0;
}

/* ============================================================================
 * 四元数CfC与标准求解器联合（增强版）
 *
 * 将四元数CfC的Hamilton乘积→ODE形式，送入增强型ODE求解器:
 * 使用实际四元数CfC动力学的RHS函数，替换旧的 -state[i]*0.5f 伪RHS。
 *
 * 支持求解器:
 *   0 = CfC闭式解（推荐）
 *   1 = RK4（固定步长）
 *   2 = RK45（自适应步长）
 *   3 = DP54（自适应步长）
 *   4 = Forest-Ruth（辛积分）
 *   5 = Rosenbrock（自适应步长，刚性）
 *   6 = Verlet（辛积分，分离变量）
 *   7 = BDF2（自适应步长，刚性）
 * ============================================================================ */

/**
 * @brief 四元数CfC ODE右端项（RHS）回调函数
 *
 * 计算分量独立CfC导数:
 *   dq_i/dt = -q_i/τ_i + q_i  (自我驱动CfC动力学)
 *
 * 不主动保持范数——求解后由 quaternion_cfc_solve_with_solver 重归一化。
 */
int quaternion_cfc_rhs(float t, const float* q, float* dqdt, void* ctx) {
    (void)t;
    if (!q || !dqdt || !ctx) return -1;

    QuaternionCfcCell* cell = (QuaternionCfcCell*)ctx;

    float inv_tau_w = 1.0f / fmaxf(cell->time_constant.w, 1e-6f);
    float inv_tau_x = 1.0f / fmaxf(cell->time_constant.x, 1e-6f);
    float inv_tau_y = 1.0f / fmaxf(cell->time_constant.y, 1e-6f);
    float inv_tau_z = 1.0f / fmaxf(cell->time_constant.z, 1e-6f);

    dqdt[0] = -q[0] * inv_tau_w + q[0];
    dqdt[1] = -q[1] * inv_tau_x + q[1];
    dqdt[2] = -q[2] * inv_tau_y + q[2];
    dqdt[3] = -q[3] * inv_tau_z + q[3];

    return 0;
}

/**
 * @brief 增强四元数ODE求解器（支持所有求解器类型）
 */
int quaternion_cfc_solve_with_solver(void* qcfc_cell, const float* quat_input,
                                      int input_dim, int solver_type, float dt,
                                      float* quat_output, int* steps_taken) {
    if (!qcfc_cell || !quat_input || !quat_output) return -1;
    if (steps_taken) *steps_taken = 0;

    int n_quat = input_dim / 4;
    if (n_quat < 1) n_quat = 1;

    float* state = (float*)safe_calloc((size_t)(n_quat * 4), sizeof(float));
    if (!state) return -1;

    memcpy(state, quat_input, (size_t)(n_quat * 4) * sizeof(float));

    float h = dt > 0.0f ? dt : 0.01f;
    int ret = 0;
    int steps = 1;

    switch (solver_type) {
        case 0: {
            /* CfC闭式解：逐四元数应用 closed-form update */
            QuaternionCfcCell* cell = (QuaternionCfcCell*)qcfc_cell;
            for (int qi = 0; qi < n_quat; qi++) {
                float* qs = state + qi * 4;
                Quaternion q = {qs[0], qs[1], qs[2], qs[3]};
                Quaternion drive = {qs[0], qs[1], qs[2], qs[3]};
                quaternion_cfc_closed_form_update(&q, &drive,
                                                   qi == 0 ? &cell->time_constant : &cell->time_constant,
                                                   h, NULL);
                qs[0] = q.w; qs[1] = q.x; qs[2] = q.y; qs[3] = q.z;
            }
            steps = 1;
            break;
        }
        case 1: {
            /* RK4固定步长 */
            size_t n = (size_t)n_quat * 4;
            float* k1 = (float*)safe_calloc(n * 4, sizeof(float));
            float* tmp = (float*)safe_calloc(n, sizeof(float));
            if (!k1 || !tmp) { safe_free((void**)&k1); safe_free((void**)&tmp); ret = -1; break; }

            float* k2 = k1 + n;
            float* k3 = k2 + n;
            float* k4 = k3 + n;

            quaternion_cfc_rhs(0.0f, state, k1, qcfc_cell);
            for (size_t i = 0; i < n; i++) tmp[i] = state[i] + 0.5f * h * k1[i];
            quaternion_cfc_rhs(0.5f * h, tmp, k2, qcfc_cell);
            for (size_t i = 0; i < n; i++) tmp[i] = state[i] + 0.5f * h * k2[i];
            quaternion_cfc_rhs(0.5f * h, tmp, k3, qcfc_cell);
            for (size_t i = 0; i < n; i++) tmp[i] = state[i] + h * k3[i];
            quaternion_cfc_rhs(h, tmp, k4, qcfc_cell);

            for (size_t i = 0; i < n; i++)
                state[i] += h / 6.0f * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);

            steps = 4;
            safe_free((void**)&k1); safe_free((void**)&tmp);
            break;
        }
        case 2:
        case 3: {
            /* RK45/DP54自适应步长 */
            size_t n = (size_t)n_quat * 4;
            size_t ws = ode_dp54_workspace_size(n);
            float* workspace = (float*)safe_calloc(ws, sizeof(float));
            if (!workspace) { ret = -1; break; }

            float h_actual = 0.0f;
            DP54Config dp_cfg = ode_dp54_default_config();
            dp_cfg.rel_tolerance = 1e-4f;
            dp_cfg.abs_tolerance = 1e-6f;

            ret = ode_dp54_solve(state, 0.0f, h, quaternion_cfc_rhs, qcfc_cell,
                                 n, &dp_cfg, workspace, &h_actual, &steps);
            safe_free((void**)&workspace);
            break;
        }
        case 4: {
            /* Forest-Ruth 辛积分（无分离版本：用前一半作为位置） */
            const float theta = SELFLNN_FOREST_RUTH_THETA;
            const float d1 = theta * 0.5f;
            const float d2 = (1.0f - theta) * 0.5f;
            const float d3 = d1;

            size_t n = (size_t)n_quat * 4;
            float* derivs = (float*)safe_calloc(n, sizeof(float));
            if (!derivs) { ret = -1; break; }

            for (int stage = 0; stage < 3; stage++) {
                float coeff = (stage == 0) ? d1 : ((stage == 1) ? d2 : d3);
                quaternion_cfc_rhs(0.0f, state, derivs, qcfc_cell);
                for (size_t i = 0; i < n; i++)
                    state[i] += coeff * h * derivs[i];
            }

            steps = 3;
            safe_free((void**)&derivs);
            break;
        }
        case 5: {
            /* Rosenbrock自适应步长 */
            size_t n = (size_t)n_quat * 4;
            size_t ws = ode_rosenbrock_workspace_size(n);
            float* workspace = (float*)safe_calloc(ws, sizeof(float));
            if (!workspace) { ret = -1; break; }

            float h_actual = 0.0f;
            RosenbrockConfig rb_cfg = ode_rosenbrock_default_config();
            rb_cfg.rel_tolerance = 1e-4f;
            rb_cfg.abs_tolerance = 1e-6f;

            ret = ode_rosenbrock_solve(state, 0.0f, h, quaternion_cfc_rhs, qcfc_cell,
                                       n, &rb_cfg, workspace, &h_actual, &steps);
            safe_free((void**)&workspace);
            break;
        }
        case 6: {
            /* Verlet辛积分（分离变量：前半位置，后半动量） */
            size_t n_half = (size_t)n_quat * 2;
            size_t n_total = (size_t)n_quat * 4;
            float* workspace = (float*)safe_calloc(n_total, sizeof(float));
            if (!workspace) { ret = -1; break; }

            ode_verlet_solve(state, state + n_half, h, quaternion_cfc_rhs,
                             qcfc_cell, n_half, workspace);
            steps = 1;
            safe_free((void**)&workspace);
            break;
        }
        case 7: {
            /* BDF2自适应步长 */
            size_t n = (size_t)n_quat * 4;
            size_t ws = ode_bdf2_workspace_size(n);
            float* workspace = (float*)safe_calloc(ws, sizeof(float));
            if (!workspace) { ret = -1; break; }

            float h_actual = 0.0f;
            BDF2Config bdf2_cfg = ode_bdf2_default_config();
            bdf2_cfg.rel_tolerance = 1e-4f;
            bdf2_cfg.abs_tolerance = 1e-6f;

            ret = ode_bdf2_solve(state, 0.0f, h, quaternion_cfc_rhs, qcfc_cell,
                                 n, &bdf2_cfg, workspace, &h_actual, &steps);
            safe_free((void**)&workspace);
            break;
        }
        default: {
            /* 默认：CfC闭式解 */
            QuaternionCfcCell* cell = (QuaternionCfcCell*)qcfc_cell;
            for (int qi = 0; qi < n_quat; qi++) {
                float* qs = state + qi * 4;
                Quaternion q = {qs[0], qs[1], qs[2], qs[3]};
                Quaternion drive = {qs[0], qs[1], qs[2], qs[3]};
                quaternion_cfc_closed_form_update(&q, &drive,
                                                   qi == 0 ? &cell->time_constant : &cell->time_constant,
                                                   h, NULL);
                qs[0] = q.w; qs[1] = q.x; qs[2] = q.y; qs[3] = q.z;
            }
            steps = 1;
            break;
        }
    }

    if (ret == 0) {
        for (int qi = 0; qi < n_quat; qi++) {
            float* qs = state + qi * 4;
            float norm = sqrtf(qs[0] * qs[0] + qs[1] * qs[1] + qs[2] * qs[2] + qs[3] * qs[3]);
            if (norm > 1e-8f) {
                float inv = 1.0f / norm;
                qs[0] *= inv; qs[1] *= inv; qs[2] *= inv; qs[3] *= inv;
            }
        }
        memcpy(quat_output, state, (size_t)(n_quat * 4) * sizeof(float));
        if (steps_taken) *steps_taken = steps;
    }

    safe_free((void**)&state);
    return ret;
}

/**
 * @brief 批量并行四元数ODE求解
 * 
 * 对多个独立的四元数CfC单元并行执行求解。
 * 使用OMP并行化（如果可用）。
 */
int quaternion_cfc_parallel_solve(QuaternionCfcCell** cells, const float* inputs,
                                   int n_cells, int solver_type, float dt,
                                   float* outputs) {
    if (!cells || !inputs || n_cells <= 0 || !outputs) return -1;

    int global_ret = 0;

#pragma omp parallel for reduction(&&:global_ret) if(n_cells >= 4)
    for (int i = 0; i < n_cells; i++) {
        if (!cells[i]) { global_ret = 0; continue; }

        const float* cell_in = inputs + (size_t)i * 4;
        float* cell_out = outputs + (size_t)i * 4;
        int steps;

        int ret = quaternion_cfc_solve_with_solver(cells[i], cell_in, 4,
                                                     solver_type, dt,
                                                     cell_out, &steps);
        if (ret != 0) global_ret = 0;
    }

    return global_ret;
}