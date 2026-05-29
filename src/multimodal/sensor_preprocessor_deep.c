/**
 * @file sensor_preprocessor_deep.c
 * @brief 传感器预处理深度实现：EKF/UKF/ESKF/粒子滤波/信息滤波
 *
 * 所有算法基于纯C实现，不依赖任何第三方库。
 * 每个滤波器集成都包含完整的 CfC ODE 状态转移模型，
 * 实现真正的连续时间动态传感器预处理。
 * 严格遵循：此为传感器原始信号预处理层，不涉及多模型融合，
 * 预处理后的信号直接输入单一LNN连续动态系统。
 */

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include "selflnn/multimodal/sensor.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/utils/math_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* 四元数转旋转矩阵辅助函数 */
static void _sf_quat_to_rotmat(const float q[4], float R[9]) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    R[0] = 1.0f - 2.0f * (yy + zz);
    R[1] = 2.0f * (xy - wz);
    R[2] = 2.0f * (xz + wy);
    R[3] = 2.0f * (xy + wz);
    R[4] = 1.0f - 2.0f * (xx + zz);
    R[5] = 2.0f * (yz - wx);
    R[6] = 2.0f * (xz - wy);
    R[7] = 2.0f * (yz + wx);
    R[8] = 1.0f - 2.0f * (xx + yy);
}

/* ============================================================================
 * 内部辅助函数
 * ============================================================================ */

/* M-027修复：使用统一activation_sigmoid替代_sf_sig */
#define _sf_sig(x) activation_sigmoid(x)

static float _sf_tanh(float x) {
    return tanhf(x);
}

static float _sf_dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static float _sf_norm(const float* a, int n) {
    return sqrtf(_sf_dot(a, a, n) + 1e-30f);
}

static void _sf_mat_vec_mul(const float* mat, const float* vec, float* out, int r, int c) {
    for (int i = 0; i < r; i++) {
        out[i] = 0.0f;
        for (int j = 0; j < c; j++)
            out[i] += mat[i * c + j] * vec[j];
    }
}

static void _sf_vec_add(float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

static void _sf_vec_sub(float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) a[i] -= b[i];
}

static void _sf_vec_scale(float* a, float s, int n) {
    for (int i = 0; i < n; i++) a[i] *= s;
}

static void _sf_mat_mul(const float* A, const float* B, float* C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * p + j];
            C[i * p + j] = sum;
        }
    }
}

static void _sf_mat_transpose(const float* A, float* AT, int m, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            AT[j * m + i] = A[i * n + j];
}

static void _sf_mat_add(float* C, const float* A, const float* B, int m, int n) {
    for (int i = 0; i < m * n; i++) C[i] = A[i] + B[i];
}

static void _sf_mat_sub(float* C, const float* A, const float* B, int m, int n) {
    for (int i = 0; i < m * n; i++) C[i] = A[i] - B[i];
}

static void _sf_mat_scale(float* A, float s, int m, int n) {
    for (int i = 0; i < m * n; i++) A[i] *= s;
}

static void _sf_mat_identity(float* A, int n) {
    memset(A, 0, n * n * sizeof(float));
    for (int i = 0; i < n; i++) A[i * n + i] = 1.0f;
}

static void _sf_mat_copy(float* dst, const float* src, int m, int n) {
    memcpy(dst, src, m * n * sizeof(float));
}

/**
 * @brief 求解线性方程组 A*x = b，其中A为n×n对称正定矩阵（Cholesky分解）
 * @return 0成功，-1失败
 */
static int _sf_cholesky_solve(const float* A, const float* b, float* x, int n) {
    float* L = (float*)safe_malloc(n * n * sizeof(float));
    if (!L) return -1;
    memset(L, 0, n * n * sizeof(float));
    for (int j = 0; j < n; j++) {
        float sum = 0.0f;
        for (int k = 0; k < j; k++) sum += L[j * n + k] * L[j * n + k];
        float val = A[j * n + j] - sum;
        if (val <= 1e-15f) { safe_free((void**)&L); return -1; }
        L[j * n + j] = sqrtf(val);
        for (int i = j + 1; i < n; i++) {
            sum = 0.0f;
            for (int k = 0; k < j; k++) sum += L[i * n + k] * L[j * n + k];
            L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
        }
    }
    float* y = (float*)safe_malloc(n * sizeof(float));
    if (!y) { safe_free((void**)&L); return -1; }
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int k = 0; k < i; k++) sum += L[i * n + k] * y[k];
        y[i] = (b[i] - sum) / L[i * n + i];
    }
    for (int i = n - 1; i >= 0; i--) {
        float sum = 0.0f;
        for (int k = i + 1; k < n; k++) sum += L[k * n + i] * x[k];
        x[i] = (y[i] - sum) / L[i * n + i];
    }
    safe_free((void**)&y);
    safe_free((void**)&L);
    return 0;
}

/**
 * @brief 计算n×n矩阵的逆（通过Cholesky分解，用于对称正定矩阵）
 */
static int _sf_mat_inv_sym(const float* A, float* inv, int n) {
    float* L = (float*)safe_malloc(n * n * sizeof(float));
    if (!L) return -1;
    memset(L, 0, n * n * sizeof(float));
    for (int j = 0; j < n; j++) {
        float sum = 0.0f;
        for (int k = 0; k < j; k++) sum += L[j * n + k] * L[j * n + k];
        float val = A[j * n + j] - sum;
        if (val <= 1e-15f) { safe_free((void**)&L); return -1; }
        L[j * n + j] = sqrtf(val);
        for (int i = j + 1; i < n; i++) {
            sum = 0.0f;
            for (int k = 0; k < j; k++) sum += L[i * n + k] * L[j * n + k];
            L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
        }
    }
    for (int j = 0; j < n; j++) {
        float* e = (float*)safe_malloc(n * sizeof(float));
        if (!e) { safe_free((void**)&L); return -1; }
        memset(e, 0, n * sizeof(float));
        e[j] = 1.0f;
        float* y = (float*)safe_malloc(n * sizeof(float));
        if (!y) { safe_free((void**)&e); safe_free((void**)&L); return -1; }
        for (int i = 0; i < n; i++) {
            float sum = 0.0f;
            for (int k = 0; k < i; k++) sum += L[i * n + k] * y[k];
            y[i] = (e[i] - sum) / L[i * n + i];
        }
        for (int i = n - 1; i >= 0; i--) {
            float sum = 0.0f;
            for (int k = i + 1; k < n; k++) sum += L[k * n + i] * inv[k * n + j];
            inv[i * n + j] = (y[i] - sum) / L[i * n + i];
        }
        safe_free((void**)&y);
        safe_free((void**)&e);
    }
    safe_free((void**)&L);
    return 0;
}

/**
 * @brief 内部CfC ODE步进（与slam_enhance.c一致）
 *
 * CfC闭式解单步演化：
 * gate = σ(W_g*h + b_g)
 * act = tanh(W_a*h + b_a)
 * h_new = h * exp(-dt/τ) + (1 - exp(-dt/τ)) * gate * act
 */
static void _sf_cfc_ode_step(const float* h, const float* input, float* h_out,
                             const float* W_h, const float* W_in,
                             const float* b_g, const float* b_a,
                             const float* tau, int hidden_size, float dt)
{
    float* gate = (float*)safe_malloc(hidden_size * sizeof(float));
    float* act  = (float*)safe_malloc(hidden_size * sizeof(float));
    float* Wh   = (float*)safe_malloc(hidden_size * sizeof(float));
    float* Win  = (float*)safe_malloc(hidden_size * sizeof(float));
    if (!gate || !act || !Wh || !Win) {
        safe_free((void**)&gate); safe_free((void**)&act); safe_free((void**)&Wh); safe_free((void**)&Win);
        memcpy(h_out, h, hidden_size * sizeof(float));
        return;
    }
    _sf_mat_vec_mul(W_h, h, Wh, hidden_size, hidden_size);
    _sf_mat_vec_mul(W_in, input, Win, hidden_size, hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        float g = _sf_sig(Wh[i] + Win[i] + b_g[i]);
        float a = _sf_tanh(Wh[i] + b_a[i]);
        gate[i] = g;
        act[i]  = a;
    }
    for (int i = 0; i < hidden_size; i++) {
        float decay = expf(-dt / fmaxf(tau[i], 1e-6f));
        h_out[i] = h[i] * decay + (1.0f - decay) * gate[i] * act[i];
    }
    safe_free((void**)&gate); safe_free((void**)&act); safe_free((void**)&Wh); safe_free((void**)&Win);
}

/**
 * @brief CfC ODE步进（多步，用于生成更准确的预测）
 */
static void _sf_cfc_ode_predict(const float* h, const float* input, float* h_out,
                                const float* W_h, const float* W_in,
                                const float* b_g, const float* b_a,
                                const float* tau, int hidden_size,
                                float dt, int num_steps)
{
    float* temp = (float*)safe_malloc(hidden_size * sizeof(float));
    if (!temp) { memcpy(h_out, h, hidden_size * sizeof(float)); return; }
    memcpy(temp, h, hidden_size * sizeof(float));
    float step_dt = dt / fmaxf((float)num_steps, 1.0f);
    for (int s = 0; s < num_steps; s++) {
        _sf_cfc_ode_step(temp, input, temp, W_h, W_in, b_g, b_a, tau, hidden_size, step_dt);
    }
    memcpy(h_out, temp, hidden_size * sizeof(float));
    safe_free((void**)&temp);
}

/* ============================================================================
 * EKF - 扩展卡尔曼滤波器
 * ============================================================================ */

struct EKFFilter {
    EKFConfig config;
    float* state;            /* 状态向量 [state_dim] */
    float* covariance;       /* 协方差矩阵 [state_dim x state_dim] */
    float* process_noise;    /* 过程噪声矩阵 Q [state_dim x state_dim] */
    float* obs_noise;        /* 观测噪声矩阵 R [obs_dim x obs_dim] */
    /* CfC ODE参数（use_cfc_transition时使用） */
    float* cfc_W_h;          /* CfC隐藏到隐藏权重 [state_dim x state_dim] */
    float* cfc_W_in;         /* CfC输入权重 [state_dim x state_dim] (状态作为输入) */
    float* cfc_W_control;    /* CfC控制权重 [state_dim x control_dim] */
    float* cfc_b_g;          /* CfC门控偏置 [state_dim] */
    float* cfc_b_a;          /* CfC激活偏置 [state_dim] */
    float* cfc_tau;          /* CfC时间常数 [state_dim] */
    float* buffer1;          /* 工作缓冲区1 [state_dim x state_dim] */
    float* buffer2;          /* 工作缓冲区2 [state_dim x obs_dim] */
    float* jacobian_F;       /* 状态转移雅可比 [state_dim x state_dim] */
    float* innovation_buf;   /* innovation缓冲区 [obs_dim] */
};

static EKFFilter* ekf_create(const EKFConfig* config) {
    if (!config) return NULL;
    EKFFilter* ekf = (EKFFilter*)safe_calloc(1, sizeof(EKFFilter));
    if (!ekf) return NULL;
    memcpy(&ekf->config, config, sizeof(EKFConfig));
    int s = config->state_dim;
    int o = config->obs_dim;
    int c = config->control_dim;
    if (s <= 0 || o <= 0) { safe_free((void**)&ekf); return NULL; }
    ekf->state = (float*)safe_calloc(s, sizeof(float));
    ekf->covariance = (float*)safe_calloc(s * s, sizeof(float));
    ekf->process_noise = (float*)safe_calloc(s * s, sizeof(float));
    ekf->obs_noise = (float*)safe_calloc(o * o, sizeof(float));
    ekf->buffer1 = (float*)safe_calloc(s * s, sizeof(float));
    ekf->buffer2 = (float*)safe_calloc(s * o, sizeof(float));
    ekf->jacobian_F = (float*)safe_calloc(s * s, sizeof(float));
    ekf->innovation_buf = (float*)safe_calloc(o, sizeof(float));
    if (!ekf->state || !ekf->covariance || !ekf->process_noise ||
        !ekf->obs_noise || !ekf->buffer1 || !ekf->buffer2 ||
        !ekf->jacobian_F || !ekf->innovation_buf) {
        ekf_free(ekf);
        return NULL;
    }
    _sf_mat_identity(ekf->covariance, s);
    _sf_mat_scale(ekf->covariance, config->initial_state_std * config->initial_state_std, s, s);
    _sf_mat_identity(ekf->process_noise, s);
    _sf_mat_scale(ekf->process_noise, config->process_noise_std * config->process_noise_std, s, s);
    _sf_mat_identity(ekf->obs_noise, o);
    _sf_mat_scale(ekf->obs_noise, config->observation_noise_std * config->observation_noise_std, o, o);
    if (config->use_cfc_transition) {
        int hs = config->cfc_hidden_size > 0 ? config->cfc_hidden_size : s;
        ekf->cfc_W_h = (float*)safe_calloc(hs * hs, sizeof(float));
        ekf->cfc_W_in = (float*)safe_calloc(hs * s, sizeof(float));
        ekf->cfc_W_control = (float*)safe_calloc(hs * (c > 1 ? c : 1), sizeof(float));
        ekf->cfc_b_g = (float*)safe_calloc(hs, sizeof(float));
        ekf->cfc_b_a = (float*)safe_calloc(hs, sizeof(float));
        ekf->cfc_tau = (float*)safe_calloc(hs, sizeof(float));
        if (!ekf->cfc_W_h || !ekf->cfc_W_in || !ekf->cfc_W_control ||
            !ekf->cfc_b_g || !ekf->cfc_b_a || !ekf->cfc_tau) {
            ekf_free(ekf);
            return NULL;
        }
        for (int i = 0; i < hs; i++) {
            ekf->cfc_tau[i] = 0.1f;
            /* K-012修复：安全随机数初始化 */
            ekf->cfc_b_g[i] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
            ekf->cfc_b_a[i] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
            for (int j = 0; j < hs; j++)
                ekf->cfc_W_h[i * hs + j] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
            for (int j = 0; j < s; j++)
                ekf->cfc_W_in[i * s + j] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
        }
    }
    return ekf;
}

static void ekf_free(EKFFilter* ekf) {
    if (!ekf) return;
    safe_free((void**)&ekf->state);
    safe_free((void**)&ekf->covariance);
    safe_free((void**)&ekf->process_noise);
    safe_free((void**)&ekf->obs_noise);
    safe_free((void**)&ekf->cfc_W_h);
    safe_free((void**)&ekf->cfc_W_in);
    safe_free((void**)&ekf->cfc_W_control);
    safe_free((void**)&ekf->cfc_b_g);
    safe_free((void**)&ekf->cfc_b_a);
    safe_free((void**)&ekf->cfc_tau);
    safe_free((void**)&ekf->buffer1);
    safe_free((void**)&ekf->buffer2);
    safe_free((void**)&ekf->jacobian_F);
    safe_free((void**)&ekf->innovation_buf);
    safe_free((void**)&ekf);
}

static int ekf_predict(EKFFilter* ekf, const float* control, float dt) {
    if (!ekf || dt <= 0.0f) return -1;
    int s = ekf->config.state_dim;
    int c = ekf->config.control_dim;
    /* 状态预测：通过CfC ODE演化或简单线性预测 */
    if (ekf->config.use_cfc_transition && ekf->cfc_W_h) {
        int hs = ekf->config.cfc_hidden_size > 0 ? ekf->config.cfc_hidden_size : s;
        float* input = (float*)safe_malloc((s + (c > 0 ? c : 0)) * sizeof(float));
        if (!input) return -1;
        memcpy(input, ekf->state, s * sizeof(float));
        if (control && c > 0)
            memcpy(input + s, control, c * sizeof(float));
        float* new_state = (float*)safe_malloc(hs * sizeof(float));
        if (!new_state) { safe_free((void**)&input); return -1; }
        _sf_cfc_ode_predict(ekf->state, input, new_state,
                           ekf->cfc_W_h, ekf->cfc_W_in,
                           ekf->cfc_b_g, ekf->cfc_b_a,
                           ekf->cfc_tau, hs, dt, 1);
        memcpy(ekf->state, new_state, (s < hs ? s : hs) * sizeof(float));
        safe_free((void**)&new_state);
        safe_free((void**)&input);
        /* 数值雅可比计算: F_ij = df_i/dx_j ≈ (f(x+δ) - f(x-δ)) / (2*δ) */
        float eps = 1e-4f;
        float* x_plus  = (float*)safe_malloc(s * sizeof(float));
        float* x_minus = (float*)safe_malloc(s * sizeof(float));
        float* f_plus  = (float*)safe_malloc(hs * sizeof(float));
        float* f_minus = (float*)safe_malloc(hs * sizeof(float));
        float* inp     = (float*)safe_malloc((s + (c > 0 ? c : 0)) * sizeof(float));
        float* f0      = (float*)safe_malloc(hs * sizeof(float));
        if (!x_plus || !x_minus || !f_plus || !f_minus || !inp || !f0) {
            safe_free((void**)&x_plus); safe_free((void**)&x_minus); safe_free((void**)&f_plus); safe_free((void**)&f_minus);
            safe_free((void**)&inp); safe_free((void**)&f0);
            return -1;
        }
        memcpy(inp, ekf->state, s * sizeof(float));
        if (control && c > 0) memcpy(inp + s, control, c * sizeof(float));
        memcpy(f0, new_state, hs * sizeof(float));
        for (int j = 0; j < s; j++) {
            memcpy(x_plus, ekf->state, s * sizeof(float));
            memcpy(x_minus, ekf->state, s * sizeof(float));
            x_plus[j] += eps;
            x_minus[j] -= eps;
            memcpy(inp, x_plus, s * sizeof(float));
            if (control && c > 0) memcpy(inp + s, control, c * sizeof(float));
            _sf_cfc_ode_predict(x_plus, inp, f_plus,
                               ekf->cfc_W_h, ekf->cfc_W_in,
                               ekf->cfc_b_g, ekf->cfc_b_a,
                               ekf->cfc_tau, hs, dt, 1);
            memcpy(inp, x_minus, s * sizeof(float));
            if (control && c > 0) memcpy(inp + s, control, c * sizeof(float));
            _sf_cfc_ode_predict(x_minus, inp, f_minus,
                               ekf->cfc_W_h, ekf->cfc_W_in,
                               ekf->cfc_b_g, ekf->cfc_b_a,
                               ekf->cfc_tau, hs, dt, 1);
            for (int i = 0; i < s && i < hs; i++)
                ekf->jacobian_F[i * s + j] = (f_plus[i] - f_minus[i]) / (2.0f * eps);
        }
        safe_free((void**)&x_plus); safe_free((void**)&x_minus); safe_free((void**)&f_plus); safe_free((void**)&f_minus);
        safe_free((void**)&inp); safe_free((void**)&f0);
    } else {
        /* 简单线性预测（单位转移） */
        _sf_mat_identity(ekf->jacobian_F, s);
    }
    /* 协方差预测: P_pred = F * P * F^T + Q */
    float* FP = ekf->buffer1;
    float* FPFt = (float*)safe_malloc(s * s * sizeof(float));
    if (!FPFt) return -1;
    _sf_mat_mul(ekf->jacobian_F, ekf->covariance, FP, s, s, s);
    _sf_mat_transpose(ekf->jacobian_F, FPFt, s, s);
    _sf_mat_mul(FP, FPFt, ekf->buffer1, s, s, s);
    _sf_mat_add(ekf->covariance, ekf->buffer1, ekf->process_noise, s, s);
    safe_free((void**)&FPFt);
    return 0;
}

static int ekf_update(EKFFilter* ekf, const float* observation) {
    if (!ekf || !observation) return -1;
    int s = ekf->config.state_dim;
    int o = ekf->config.obs_dim;
    /* 使用线性观测模型 H = [I_o, 0] */
    float* H = (float*)safe_malloc(o * s * sizeof(float));
    float* P_HT = (float*)safe_malloc(s * o * sizeof(float));
    float* S = (float*)safe_malloc(o * o * sizeof(float));
    float* K = (float*)safe_malloc(s * o * sizeof(float));
    float* innovation = ekf->innovation_buf;
    if (!H || !P_HT || !S || !K) {
        safe_free((void**)&H); safe_free((void**)&P_HT); safe_free((void**)&S); safe_free((void**)&K);
        return -1;
    }
    memset(H, 0, o * s * sizeof(float));
    int min_dim = s < o ? s : o;
    for (int i = 0; i < min_dim; i++) H[i * s + i] = 1.0f;
    /* innovation: y = z - H*x_pred */
    for (int i = 0; i < o; i++) {
        innovation[i] = observation[i];
        for (int j = 0; j < s; j++)
            innovation[i] -= H[i * s + j] * ekf->state[j];
    }
    /* S = H * P_pred * H^T + R */
    _sf_mat_mul(ekf->covariance, H, P_HT, s, s, o);
    _sf_mat_transpose(H, ekf->buffer2, o, s);
    _sf_mat_mul(H, ekf->covariance, S, o, s, s);
    _sf_mat_mul(S, ekf->buffer2, ekf->buffer1, o, s, o);
    _sf_mat_add(S, ekf->buffer1, ekf->obs_noise, o, o);
    /* 求解 S * K^T = (H*P)^T，即 K^T = S^(-1) * (H*P)^T */
    float* HPT = (float*)safe_malloc(o * s * sizeof(float));
    if (!HPT) { safe_free((void**)&H); safe_free((void**)&P_HT); safe_free((void**)&S); safe_free((void**)&K); return -1; }
    _sf_mat_transpose(P_HT, HPT, s, o);
    for (int j = 0; j < s; j++) {
        if (_sf_cholesky_solve(S, HPT + j * o, K + j * o, o) != 0) {
            /* S可能奇异，使用伪逆近似 */
            for (int i = 0; i < o; i++)
                K[j * o + i] = (o == 1 && fabsf(S[0]) > 1e-15f) ? HPT[j] / S[0] : 0.0f;
        }
    }
    /* 状态更新: x = x_pred + K * y */
    float* Ky = (float*)safe_malloc(s * sizeof(float));
    if (!Ky) { safe_free((void**)&H); safe_free((void**)&P_HT); safe_free((void**)&S); safe_free((void**)&K); safe_free((void**)&HPT); return -1; }
    _sf_mat_vec_mul(K, innovation, Ky, s, o);
    _sf_vec_add(ekf->state, Ky, s);
    /* 协方差更新: P = (I - K*H) * P_pred */
    float* I_KH = (float*)safe_malloc(s * s * sizeof(float));
    if (!I_KH) { safe_free((void**)&H); safe_free((void**)&P_HT); safe_free((void**)&S); safe_free((void**)&K); safe_free((void**)&HPT); safe_free((void**)&Ky); return -1; }
    _sf_mat_identity(I_KH, s);
    _sf_mat_mul(K, H, ekf->buffer1, s, o, s);
    _sf_mat_sub(I_KH, I_KH, ekf->buffer1, s, s);
    _sf_mat_mul(I_KH, ekf->covariance, ekf->buffer2, s, s, s);
    _sf_mat_copy(ekf->covariance, ekf->buffer2, s, s);
    safe_free((void**)&H); safe_free((void**)&P_HT); safe_free((void**)&S); safe_free((void**)&K); safe_free((void**)&HPT); safe_free((void**)&Ky); safe_free((void**)&I_KH);
    return 0;
}

static int ekf_get_state(const EKFFilter* ekf, float* state, float* covariance) {
    if (!ekf || !state) return -1;
    int s = ekf->config.state_dim;
    memcpy(state, ekf->state, s * sizeof(float));
    if (covariance)
        memcpy(covariance, ekf->covariance, s * s * sizeof(float));
    return 0;
}

static void ekf_reset(EKFFilter* ekf, const float* state, const float* covariance) {
    if (!ekf) return;
    int s = ekf->config.state_dim;
    if (state) memcpy(ekf->state, state, s * sizeof(float));
    else memset(ekf->state, 0, s * sizeof(float));
    if (covariance) memcpy(ekf->covariance, covariance, s * s * sizeof(float));
    else {
        _sf_mat_identity(ekf->covariance, s);
        _sf_mat_scale(ekf->covariance, ekf->config.initial_state_std * ekf->config.initial_state_std, s, s);
    }
}

/* ============================================================================
 * UKF - 无迹卡尔曼滤波器
 * ============================================================================ */

struct UKFFilter {
    UKFConfig config;
    float* state;            /* 状态向量 [state_dim] */
    float* covariance;       /* 协方差矩阵 [state_dim x state_dim] */
    float* process_noise;    /* Q [state_dim x state_dim] */
    float* obs_noise;        /* R [obs_dim x obs_dim] */
    float* sigma_points;     /* Sigma点 [(2*state_dim+1) x state_dim] */
    float* sigma_weights_m;  /* Sigma点均值权重 [2*state_dim+1] */
    float* sigma_weights_c;  /* Sigma点协方差权重 [2*state_dim+1] */
    /* CfC参数 */
    float* cfc_W_h;
    float* cfc_W_in;
    float* cfc_b_g;
    float* cfc_b_a;
    float* cfc_tau;
    /* 工作缓冲区 */
    float* buffer;
};

static UKFFilter* ukf_create(const UKFConfig* config) {
    if (!config) return NULL;
    UKFFilter* ukf = (UKFFilter*)safe_calloc(1, sizeof(UKFFilter));
    if (!ukf) return NULL;
    memcpy(&ukf->config, config, sizeof(UKFConfig));
    int s = config->state_dim;
    int o = config->obs_dim;
    if (s <= 0 || o <= 0) { safe_free((void**)&ukf); return NULL; }
    int n_sp = 2 * s + 1;
    ukf->state = (float*)safe_calloc(s, sizeof(float));
    ukf->covariance = (float*)safe_calloc(s * s, sizeof(float));
    ukf->process_noise = (float*)safe_calloc(s * s, sizeof(float));
    ukf->obs_noise = (float*)safe_calloc(o * o, sizeof(float));
    ukf->sigma_points = (float*)safe_calloc(n_sp * s, sizeof(float));
    ukf->sigma_weights_m = (float*)safe_calloc(n_sp, sizeof(float));
    ukf->sigma_weights_c = (float*)safe_calloc(n_sp, sizeof(float));
    ukf->buffer = (float*)safe_calloc((s * s > s * o ? s * s : s * o) * 2, sizeof(float));
    if (!ukf->state || !ukf->covariance || !ukf->process_noise ||
        !ukf->obs_noise || !ukf->sigma_points || !ukf->sigma_weights_m ||
        !ukf->sigma_weights_c || !ukf->buffer) {
        ukf_free(ukf);
        return NULL;
    }
    _sf_mat_identity(ukf->covariance, s);
    _sf_mat_scale(ukf->covariance, config->initial_state_std, s, s);
    _sf_mat_identity(ukf->process_noise, s);
    _sf_mat_scale(ukf->process_noise, config->process_noise_std * config->process_noise_std, s, s);
    _sf_mat_identity(ukf->obs_noise, o);
    _sf_mat_scale(ukf->obs_noise, config->observation_noise_std * config->observation_noise_std, o, o);
    float alpha = config->alpha > 0.0f ? config->alpha : 1e-3f;
    float beta  = config->beta;
    float kappa = config->kappa;
    float lambda = alpha * alpha * ((float)s + kappa) - (float)s;
    ukf->sigma_weights_m[0] = lambda / ((float)s + lambda);
    ukf->sigma_weights_c[0] = lambda / ((float)s + lambda) + (1.0f - alpha * alpha + beta);
    for (int i = 1; i < n_sp; i++) {
        float w = 1.0f / (2.0f * ((float)s + lambda));
        ukf->sigma_weights_m[i] = w;
        ukf->sigma_weights_c[i] = w;
    }
    if (config->use_cfc_transition) {
        int hs = config->cfc_hidden_size > 0 ? config->cfc_hidden_size : s;
        ukf->cfc_W_h = (float*)safe_calloc(hs * hs, sizeof(float));
        ukf->cfc_W_in = (float*)safe_calloc(hs * s, sizeof(float));
        ukf->cfc_b_g = (float*)safe_calloc(hs, sizeof(float));
        ukf->cfc_b_a = (float*)safe_calloc(hs, sizeof(float));
        ukf->cfc_tau = (float*)safe_calloc(hs, sizeof(float));
        if (!ukf->cfc_W_h || !ukf->cfc_W_in || !ukf->cfc_b_g ||
            !ukf->cfc_b_a || !ukf->cfc_tau) {
            ukf_free(ukf);
            return NULL;
        }
        for (int i = 0; i < hs; i++) {
            ukf->cfc_tau[i] = 0.1f;
            /* K-012修复：安全随机数初始化 */
            ukf->cfc_b_g[i] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
            ukf->cfc_b_a[i] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
            for (int j = 0; j < hs; j++)
                ukf->cfc_W_h[i * hs + j] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
            for (int j = 0; j < s; j++)
                ukf->cfc_W_in[i * s + j] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
        }
    }
    return ukf;
}

static void ukf_free(UKFFilter* ukf) {
    if (!ukf) return;
    safe_free((void**)&ukf->state);
    safe_free((void**)&ukf->covariance);
    safe_free((void**)&ukf->process_noise);
    safe_free((void**)&ukf->obs_noise);
    safe_free((void**)&ukf->sigma_points);
    safe_free((void**)&ukf->sigma_weights_m);
    safe_free((void**)&ukf->sigma_weights_c);
    safe_free((void**)&ukf->cfc_W_h);
    safe_free((void**)&ukf->cfc_W_in);
    safe_free((void**)&ukf->cfc_b_g);
    safe_free((void**)&ukf->cfc_b_a);
    safe_free((void**)&ukf->cfc_tau);
    safe_free((void**)&ukf->buffer);
    safe_free((void**)&ukf);
}

static int ukf_predict(UKFFilter* ukf, const float* control, float dt) {
    (void)control;
    if (!ukf || dt <= 0.0f) return -1;
    int s = ukf->config.state_dim;
    int n_sp = 2 * s + 1;
    float alpha = ukf->config.alpha > 0.0f ? ukf->config.alpha : 1e-3f;
    float kappa = ukf->config.kappa;
    float lambda = alpha * alpha * ((float)s + kappa) - (float)s;
    float sqrt_term = sqrtf((float)s + lambda);
    /* Cholesky分解协方差矩阵 */
    float* L = (float*)safe_malloc(s * s * sizeof(float));
    if (!L) return -1;
    memset(L, 0, s * s * sizeof(float));
    for (int j = 0; j < s; j++) {
        float sum = 0.0f;
        for (int k = 0; k < j; k++) sum += L[j * s + k] * L[j * s + k];
        float val = ukf->covariance[j * s + j] - sum;
        if (val <= 0.0f) val = 1e-6f;
        L[j * s + j] = sqrtf(val);
        for (int i = j + 1; i < s; i++) {
            sum = 0.0f;
            for (int k = 0; k < j; k++) sum += L[i * s + k] * L[j * s + k];
            L[i * s + j] = (ukf->covariance[i * s + j] - sum) / L[j * s + j];
        }
    }
    /* 生成Sigma点 */
    memcpy(ukf->sigma_points, ukf->state, s * sizeof(float));
    for (int i = 0; i < s; i++) {
        for (int j = 0; j < s; j++) {
            ukf->sigma_points[(i + 1) * s + j]     = ukf->state[j] + sqrt_term * L[j * s + i];
            ukf->sigma_points[(i + 1 + s) * s + j] = ukf->state[j] - sqrt_term * L[j * s + i];
        }
    }
    safe_free((void**)&L);
    /* 传播Sigma点通过CfC ODE */
    int use_cfc = (ukf->config.use_cfc_transition && ukf->cfc_W_h);
    int hs = use_cfc ? (ukf->config.cfc_hidden_size > 0 ? ukf->config.cfc_hidden_size : s) : s;
    for (int i = 0; i < n_sp; i++) {
        float* sp = ukf->sigma_points + i * s;
        if (use_cfc) {
            _sf_cfc_ode_predict(sp, sp, sp,
                               ukf->cfc_W_h, ukf->cfc_W_in,
                               ukf->cfc_b_g, ukf->cfc_b_a,
                               ukf->cfc_tau, hs, dt, 1);
        }
        /* 否则Sigma点保持不变（线性转移） */
    }
    /* 加权计算预测均值和协方差 */
    memset(ukf->state, 0, s * sizeof(float));
    for (int i = 0; i < n_sp; i++) {
        float* sp = ukf->sigma_points + i * s;
        for (int j = 0; j < s; j++)
            ukf->state[j] += ukf->sigma_weights_m[i] * sp[j];
    }
    memset(ukf->covariance, 0, s * s * sizeof(float));
    for (int i = 0; i < n_sp; i++) {
        float* sp = ukf->sigma_points + i * s;
        float* diff = (float*)safe_malloc(s * sizeof(float));
        if (!diff) continue;
        for (int j = 0; j < s; j++) diff[j] = sp[j] - ukf->state[j];
        for (int r = 0; r < s; r++)
            for (int c = 0; c < s; c++)
                ukf->covariance[r * s + c] += ukf->sigma_weights_c[i] * diff[r] * diff[c];
        safe_free((void**)&diff);
    }
    _sf_mat_add(ukf->covariance, ukf->covariance, ukf->process_noise, s, s);
    return 0;
}

static int ukf_update(UKFFilter* ukf, const float* observation) {
    if (!ukf || !observation) return -1;
    int s = ukf->config.state_dim;
    int o = ukf->config.obs_dim;
    int n_sp = 2 * s + 1;
    float alpha = ukf->config.alpha > 0.0f ? ukf->config.alpha : 1e-3f;
    float kappa = ukf->config.kappa;
    float lambda = alpha * alpha * ((float)s + kappa) - (float)s;
    float sqrt_term = sqrtf((float)s + lambda);
    /* 从预测状态重新生成Sigma点 */
    float* L = (float*)safe_malloc(s * s * sizeof(float));
    float* sigma_new = (float*)safe_malloc(n_sp * s * sizeof(float));
    float* sigma_obs = (float*)safe_malloc(n_sp * o * sizeof(float));
    float* z_pred = (float*)safe_malloc(o * sizeof(float));
    float* S = (float*)safe_malloc(o * o * sizeof(float));
    float* cross_cov = (float*)safe_malloc(s * o * sizeof(float));
    float* K = (float*)safe_malloc(s * o * sizeof(float));
    if (!L || !sigma_new || !sigma_obs || !z_pred || !S || !cross_cov || !K) {
        safe_free((void**)&L); safe_free((void**)&sigma_new); safe_free((void**)&sigma_obs); safe_free((void**)&z_pred);
        safe_free((void**)&S); safe_free((void**)&cross_cov); safe_free((void**)&K);
        return -1;
    }
    memset(L, 0, s * s * sizeof(float));
    for (int j = 0; j < s; j++) {
        float sum = 0.0f;
        for (int k = 0; k < j; k++) sum += L[j * s + k] * L[j * s + k];
        float val = ukf->covariance[j * s + j] - sum;
        if (val <= 0.0f) val = 1e-6f;
        L[j * s + j] = sqrtf(val);
        for (int i = j + 1; i < s; i++) {
            sum = 0.0f;
            for (int k = 0; k < j; k++) sum += L[i * s + k] * L[j * s + k];
            L[i * s + j] = (ukf->covariance[i * s + j] - sum) / L[j * s + j];
        }
    }
    memcpy(sigma_new, ukf->state, s * sizeof(float));
    for (int i = 0; i < s; i++) {
        for (int j = 0; j < s; j++) {
            sigma_new[(i + 1) * s + j]     = ukf->state[j] + sqrt_term * L[j * s + i];
            sigma_new[(i + 1 + s) * s + j] = ukf->state[j] - sqrt_term * L[j * s + i];
        }
    }
    safe_free((void**)&L);
    /* 通过观测模型传播（线性观测模型 H: s→o） */
    int min_dim = s < o ? s : o;
    memset(z_pred, 0, o * sizeof(float));
    for (int i = 0; i < n_sp; i++) {
        float* sp = sigma_new + i * s;
        float* zo = sigma_obs + i * o;
        memset(zo, 0, o * sizeof(float));
        for (int j = 0; j < min_dim; j++) zo[j] = sp[j];
        for (int j = 0; j < o; j++)
            z_pred[j] += ukf->sigma_weights_m[i] * zo[j];
    }
    /* 观测协方差 S */
    memset(S, 0, o * o * sizeof(float));
    for (int i = 0; i < n_sp; i++) {
        float* zo = sigma_obs + i * o;
        float* diff = (float*)safe_malloc(o * sizeof(float));
        if (!diff) continue;
        for (int j = 0; j < o; j++) diff[j] = zo[j] - z_pred[j];
        for (int r = 0; r < o; r++)
            for (int c = 0; c < o; c++)
                S[r * o + c] += ukf->sigma_weights_c[i] * diff[r] * diff[c];
        safe_free((void**)&diff);
    }
    _sf_mat_add(S, S, ukf->obs_noise, o, o);
    /* 互协方差 */
    memset(cross_cov, 0, s * o * sizeof(float));
    for (int i = 0; i < n_sp; i++) {
        float* sp = sigma_new + i * s;
        float* zo = sigma_obs + i * o;
        float* dx = (float*)safe_malloc(s * sizeof(float));
        float* dz = (float*)safe_malloc(o * sizeof(float));
        if (!dx || !dz) { safe_free((void**)&dx); safe_free((void**)&dz); continue; }
        for (int j = 0; j < s; j++) dx[j] = sp[j] - ukf->state[j];
        for (int j = 0; j < o; j++) dz[j] = zo[j] - z_pred[j];
        for (int r = 0; r < s; r++)
            for (int c = 0; c < o; c++)
                cross_cov[r * o + c] += ukf->sigma_weights_c[i] * dx[r] * dz[c];
        safe_free((void**)&dx); safe_free((void**)&dz);
    }
    /* Kalman增益: K = cross_cov * S^(-1) */
    float* innovation = (float*)safe_malloc(o * sizeof(float));
    if (!innovation) { safe_free((void**)&sigma_new); safe_free((void**)&sigma_obs); safe_free((void**)&z_pred);
        safe_free((void**)&S); safe_free((void**)&cross_cov); safe_free((void**)&K); return -1; }
    for (int j = 0; j < s; j++) {
        float* col = K + j * o;
        if (_sf_cholesky_solve(S, cross_cov + j * o, col, o) != 0) {
            for (int i = 0; i < o; i++) col[i] = 0.0f;
        }
    }
    /* innovation: y = z - z_pred */
    for (int i = 0; i < o; i++)
        innovation[i] = observation[i] - z_pred[i];
    /* 状态更新 */
    float* Ky = (float*)safe_malloc(s * sizeof(float));
    if (!Ky) { safe_free((void**)&sigma_new); safe_free((void**)&sigma_obs); safe_free((void**)&z_pred);
        safe_free((void**)&S); safe_free((void**)&cross_cov); safe_free((void**)&K); safe_free((void**)&innovation); return -1; }
    _sf_mat_vec_mul(K, innovation, Ky, s, o);
    _sf_vec_add(ukf->state, Ky, s);
    /* 协方差更新: P = P_pred - K*S*K^T */
    float* K_S = (float*)safe_malloc(s * o * sizeof(float));
    float* K_S_Kt = (float*)safe_malloc(s * s * sizeof(float));
    float* Kt = (float*)safe_malloc(o * s * sizeof(float));
    if (!K_S || !K_S_Kt || !Kt) {
        safe_free((void**)&sigma_new); safe_free((void**)&sigma_obs); safe_free((void**)&z_pred); safe_free((void**)&S);
        safe_free((void**)&cross_cov); safe_free((void**)&K); safe_free((void**)&innovation); safe_free((void**)&Ky);
        safe_free((void**)&K_S); safe_free((void**)&K_S_Kt); safe_free((void**)&Kt);
        return -1;
    }
    _sf_mat_mul(K, S, K_S, s, o, o);
    _sf_mat_transpose(K, Kt, s, o);
    _sf_mat_mul(K_S, Kt, K_S_Kt, s, o, s);
    _sf_mat_sub(ukf->covariance, ukf->covariance, K_S_Kt, s, s);
    safe_free((void**)&sigma_new); safe_free((void**)&sigma_obs); safe_free((void**)&z_pred); safe_free((void**)&S);
    safe_free((void**)&cross_cov); safe_free((void**)&K); safe_free((void**)&innovation); safe_free((void**)&Ky);
    safe_free((void**)&K_S); safe_free((void**)&K_S_Kt); safe_free((void**)&Kt);
    return 0;
}

static int ukf_get_state(const UKFFilter* ukf, float* state, float* covariance) {
    if (!ukf || !state) return -1;
    int s = ukf->config.state_dim;
    memcpy(state, ukf->state, s * sizeof(float));
    if (covariance)
        memcpy(covariance, ukf->covariance, s * s * sizeof(float));
    return 0;
}

static void ukf_reset(UKFFilter* ukf, const float* state, const float* covariance) {
    if (!ukf) return;
    int s = ukf->config.state_dim;
    if (state) memcpy(ukf->state, state, s * sizeof(float));
    else memset(ukf->state, 0, s * sizeof(float));
    if (covariance) memcpy(ukf->covariance, covariance, s * s * sizeof(float));
    else {
        _sf_mat_identity(ukf->covariance, s);
        _sf_mat_scale(ukf->covariance, ukf->config.initial_state_std, s, s);
    }
}

/* ============================================================================
 * ESKF - 误差状态卡尔曼滤波器
 * ============================================================================ */

struct ESKFFilter {
    ESKFConfig config;
    float* nom_state;        /* 名义状态 [nom_state_dim] */
    float* error_state;      /* 误差状态 [error_state_dim] */
    float* covariance;       /* 误差状态协方差 [error_state_dim x error_state_dim] */
    float* process_noise;    /* Q [error_state_dim x error_state_dim] */
    float* obs_noise;        /* R [obs_dim x obs_dim] */
    float* F_error;          /* 误差状态转移矩阵 [error_state_dim x error_state_dim] */
    /* CfC参数 */
    float* cfc_W_h;
    float* cfc_W_in;
    float* cfc_b_g;
    float* cfc_b_a;
    float* cfc_tau;
    /* 工作缓冲区 */
    float* buffer;
};

ESKFFilter* eskf_create(const ESKFConfig* config) {
    if (!config) return NULL;
    ESKFFilter* eskf = (ESKFFilter*)safe_calloc(1, sizeof(ESKFFilter));
    if (!eskf) return NULL;
    memcpy(&eskf->config, config, sizeof(ESKFConfig));
    int ns = config->nom_state_dim;
    int es = config->error_state_dim;
    int o  = config->obs_dim;
    if (ns <= 0 || es <= 0 || o <= 0) { safe_free((void**)&eskf); return NULL; }
    eskf->nom_state   = (float*)safe_calloc(ns, sizeof(float));
    eskf->error_state = (float*)safe_calloc(es, sizeof(float));
    eskf->covariance  = (float*)safe_calloc(es * es, sizeof(float));
    eskf->process_noise = (float*)safe_calloc(es * es, sizeof(float));
    eskf->obs_noise     = (float*)safe_calloc(o * o, sizeof(float));
    eskf->F_error = (float*)safe_calloc(es * es, sizeof(float));
    eskf->buffer  = (float*)safe_calloc((es * es > es * o ? es * es : es * o) * 2, sizeof(float));
    if (!eskf->nom_state || !eskf->error_state || !eskf->covariance ||
        !eskf->process_noise || !eskf->obs_noise || !eskf->F_error || !eskf->buffer) {
        eskf_free(eskf);
        return NULL;
    }
    _sf_mat_identity(eskf->covariance, es);
    _sf_mat_identity(eskf->process_noise, es);
    _sf_mat_identity(eskf->obs_noise, o);
    if (config->use_cfc_transition) {
        int hs = config->cfc_hidden_size > 0 ? config->cfc_hidden_size : ns;
        eskf->cfc_W_h = (float*)safe_calloc(hs * hs, sizeof(float));
        eskf->cfc_W_in = (float*)safe_calloc(hs * ns, sizeof(float));
        eskf->cfc_b_g = (float*)safe_calloc(hs, sizeof(float));
        eskf->cfc_b_a = (float*)safe_calloc(hs, sizeof(float));
        eskf->cfc_tau = (float*)safe_calloc(hs, sizeof(float));
        if (!eskf->cfc_W_h || !eskf->cfc_W_in || !eskf->cfc_b_g ||
            !eskf->cfc_b_a || !eskf->cfc_tau) {
            eskf_free(eskf);
            return NULL;
        }
        for (int i = 0; i < hs; i++) {
            eskf->cfc_tau[i] = 0.1f;
            /* K-012修复：安全随机数初始化 */
            eskf->cfc_b_g[i] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
            eskf->cfc_b_a[i] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
            for (int j = 0; j < hs; j++)
                eskf->cfc_W_h[i * hs + j] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
            for (int j = 0; j < ns; j++)
                eskf->cfc_W_in[i * ns + j] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
        }
    }
    return eskf;
}

static void eskf_free(ESKFFilter* eskf) {
    if (!eskf) return;
    safe_free((void**)&eskf->nom_state);
    safe_free((void**)&eskf->error_state);
    safe_free((void**)&eskf->covariance);
    safe_free((void**)&eskf->process_noise);
    safe_free((void**)&eskf->obs_noise);
    safe_free((void**)&eskf->F_error);
    safe_free((void**)&eskf->cfc_W_h);
    safe_free((void**)&eskf->cfc_W_in);
    safe_free((void**)&eskf->cfc_b_g);
    safe_free((void**)&eskf->cfc_b_a);
    safe_free((void**)&eskf->cfc_tau);
    safe_free((void**)&eskf->buffer);
    safe_free((void**)&eskf);
}

static int eskf_predict_nominal(ESKFFilter* eskf, const float gyro[3], const float acc[3], float dt) {
    if (!eskf || !gyro || !acc || dt <= 0.0f) return -1;
    int ns = eskf->config.nom_state_dim;
    /* 名义状态结构: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, bgx, bgy, bgz, bax, bay, baz, ...] */
    if (ns < 16) return -1;
    float* nom = eskf->nom_state;
    /* 提取状态分量 */
    float px = nom[0], py = nom[1], pz = nom[2];
    float vx = nom[3], vy = nom[4], vz = nom[5];
    float qw = nom[6], qx = nom[7], qy = nom[8], qz = nom[9];
    float bgx = nom[10], bgy = nom[11], bgz = nom[12];
    float bax = nom[13], bay = nom[14], baz = nom[15];
    /* 校正陀螺仪和加速度计偏置 */
    float wx = gyro[0] - bgx;
    float wy = gyro[1] - bgy;
    float wz = gyro[2] - bgz;
    float ax = acc[0] - bax;
    float ay = acc[1] - bay;
    float az = acc[2] - baz;
    /* 姿态更新（四元数运动学） */
    float wx2 = wx * 0.5f, wy2 = wy * 0.5f, wz2 = wz * 0.5f;
    float dqw = -qx * wx2 - qy * wy2 - qz * wz2;
    float dqx =  qw * wx2 + qy * wz2 - qz * wy2;
    float dqy =  qw * wy2 - qx * wz2 + qz * wx2;
    float dqz =  qw * wz2 + qx * wy2 - qy * wx2;
    qw += dqw * dt; qx += dqx * dt; qy += dqy * dt; qz += dqz * dt;
    float qnorm = sqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
    if (qnorm > 1e-10f) { qw /= qnorm; qx /= qnorm; qy /= qnorm; qz /= qnorm; }
    else { qw = 1.0f; qx = 0.0f; qy = 0.0f; qz = 0.0f; }
    /* 旋转加速度到世界坐标系 */
    float gx_w = (1.0f - 2.0f*(qy*qy + qz*qz)) * ax
               + 2.0f*(qx*qy - qw*qz) * ay
               + 2.0f*(qx*qz + qw*qy) * az;
    float gy_w = 2.0f*(qx*qy + qw*qz) * ax
               + (1.0f - 2.0f*(qx*qx + qz*qz)) * ay
               + 2.0f*(qy*qz - qw*qx) * az;
    float gz_w = 2.0f*(qx*qz - qw*qy) * ax
               + 2.0f*(qy*qz + qw*qx) * ay
               + (1.0f - 2.0f*(qx*qx + qy*qy)) * az;
    /* 速度更新（使用CfC ODE或标准IMU运动学） */
    if (eskf->config.use_cfc_transition && eskf->cfc_W_h) {
        float* cfc_input = (float*)safe_malloc(ns * sizeof(float));
        float* cfc_out   = (float*)safe_malloc(ns * sizeof(float));
        if (cfc_input && cfc_out) {
            memcpy(cfc_input, nom, ns * sizeof(float));
            _sf_cfc_ode_predict(cfc_input, cfc_input, cfc_out,
                               eskf->cfc_W_h, eskf->cfc_W_in,
                               eskf->cfc_b_g, eskf->cfc_b_a,
                               eskf->cfc_tau,
                               eskf->config.cfc_hidden_size > 0 ?
                               eskf->config.cfc_hidden_size : ns, dt, 1);
            memcpy(nom, cfc_out, ns * sizeof(float));
        }
        safe_free((void**)&cfc_input);
        safe_free((void**)&cfc_out);
    }
    /* 标准IMU运动学更新（在CfC输出基础上叠加） */
    nom[0] = px + vx * dt + 0.5f * gx_w * dt * dt;
    nom[1] = py + vy * dt + 0.5f * gy_w * dt * dt;
    nom[2] = pz + vz * dt + 0.5f * gz_w * dt * dt;
    nom[3] = vx + gx_w * dt;
    nom[4] = vy + gy_w * dt;
    nom[5] = vz + gz_w * dt;
    nom[6] = qw; nom[7] = qx; nom[8] = qy; nom[9] = qz;
    nom[10] = bgx; nom[11] = bgy; nom[12] = bgz;
    nom[13] = bax; nom[14] = bay; nom[15] = baz;
    return 0;
}

static int eskf_predict_error(ESKFFilter* eskf, float dt) {
    if (!eskf || dt <= 0.0f) return -1;
    int es = eskf->config.error_state_dim;
    /* 构建误差状态转移矩阵 F_error (15维标准ESKF: δp, δv, δθ, δbg, δba) */
    if (es < 15) return -1;
    float* nom = eskf->nom_state;
    float qw = nom[6], qx = nom[7], qy = nom[8], qz = nom[9];
    float* F = eskf->F_error;
    memset(F, 0, es * es * sizeof(float));
    /* F[0:3, 0:3] = I_3 (位置误差转移) */
    for (int i = 0; i < 3; i++) F[i * es + i] = 1.0f;
    /* F[0:3, 3:6] = I_3 * dt (位置←速度) */
    for (int i = 0; i < 3; i++) F[i * es + 3 + i] = dt;
    /* F[3:6, 3:6] = I_3 (速度误差转移) */
    for (int i = 0; i < 3; i++) F[(3 + i) * es + 3 + i] = 1.0f;
    /* F[3:6, 6:9] = -R * [acc]× * dt (速度←姿态) */
    float ax = nom[13], ay = nom[14], az = nom[15];
    float acc_skew[9] = {
        0.0f, -az,  ay,
         az, 0.0f, -ax,
        -ay,  ax, 0.0f
    };
    float R[9];
    _sf_quat_to_rotmat((float[4]){qw, qx, qy, qz}, R);
    float R_acc_skew[9];
    _sf_mat_mul(R, acc_skew, R_acc_skew, 3, 3, 3);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            F[(3 + i) * es + 6 + j] = -R_acc_skew[i * 3 + j] * dt;
    /* F[3:6, 12:15] = -R * dt (速度←加速度偏置) */
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            F[(3 + i) * es + 12 + j] = -R[i * 3 + j] * dt;
    /* F[6:9, 6:9] = I_3 (姿态误差转移) */
    for (int i = 0; i < 3; i++) F[(6 + i) * es + 6 + i] = 1.0f;
    /* F[6:9, 9:12] = -I_3 * dt (姿态←陀螺偏置) */
    for (int i = 0; i < 3; i++) F[(6 + i) * es + 9 + i] = -dt;
    /* F[9:12, 9:12] = I_3 (陀螺偏置) */
    for (int i = 0; i < 3; i++) F[(9 + i) * es + 9 + i] = 1.0f;
    /* F[12:15, 12:15] = I_3 (加速度偏置) */
    for (int i = 0; i < 3; i++) F[(12 + i) * es + 12 + i] = 1.0f;
    /* 误差状态协方差预测: P = F * P * F^T + Q */
    float* FP = (float*)safe_malloc(es * es * sizeof(float));
    float* FPFt = (float*)safe_malloc(es * es * sizeof(float));
    float* Ft = (float*)safe_malloc(es * es * sizeof(float));
    if (!FP || !FPFt || !Ft) {
        safe_free((void**)&FP); safe_free((void**)&FPFt); safe_free((void**)&Ft);
        return -1;
    }
    _sf_mat_mul(F, eskf->covariance, FP, es, es, es);
    _sf_mat_transpose(F, Ft, es, es);
    _sf_mat_mul(FP, Ft, FPFt, es, es, es);
    _sf_mat_add(eskf->covariance, FPFt, eskf->process_noise, es, es);
    safe_free((void**)&FP); safe_free((void**)&FPFt); safe_free((void**)&Ft);
    return 0;
}

static int eskf_update(ESKFFilter* eskf, const float* observation, int obs_model_type) {
    if (!eskf || !observation) return -1;
    int es = eskf->config.error_state_dim;
    int o  = eskf->config.obs_dim;
    /* 构建观测矩阵 H（取决于obs_model_type） */
    float* H = (float*)safe_malloc(o * es * sizeof(float));
    float* innovation = (float*)safe_malloc(o * sizeof(float));
    if (!H || !innovation) { safe_free((void**)&H); safe_free((void**)&innovation); return -1; }
    memset(H, 0, o * es * sizeof(float));
    if (obs_model_type == 0) {
        for (int i = 0; i < o && i < 3; i++) H[i * es + i] = 1.0f;
    } else if (obs_model_type == 1) {
        for (int i = 0; i < o && i < 3; i++) H[i * es + 6 + i] = 1.0f;
    } else if (obs_model_type == 2) {
        for (int i = 0; i < o && i < 3; i++) H[i * es + 3 + i] = 1.0f;
    } else {
        for (int i = 0; i < o && i < es; i++) H[i * es + i] = 1.0f;
    }
    /* innovation: y = z - h(x_nom) */
    for (int i = 0; i < o; i++) {
        innovation[i] = observation[i];
        for (int j = 0; j < es; j++)
            innovation[i] -= H[i * es + j] * eskf->error_state[j];
    }
    /* S = H * P * H^T + R */
    float* S = (float*)safe_malloc(o * o * sizeof(float));
    float* P_HT = (float*)safe_malloc(es * o * sizeof(float));
    float* K = (float*)safe_malloc(es * o * sizeof(float));
    if (!S || !P_HT || !K) { safe_free((void**)&H); safe_free((void**)&innovation); safe_free((void**)&S); safe_free((void**)&P_HT); safe_free((void**)&K); return -1; }
    _sf_mat_mul(eskf->covariance, H, P_HT, es, es, o);
    _sf_mat_transpose(H, eskf->buffer, o, es);
    _sf_mat_mul(H, eskf->covariance, S, o, es, es);
    _sf_mat_mul(S, eskf->buffer, eskf->buffer + es * o, o, es, o);
    _sf_mat_add(S, eskf->buffer + es * o, eskf->obs_noise, o, o);
    /* Kalman增益 */
    float* HT_PT = (float*)safe_malloc(o * es * sizeof(float));
    if (!HT_PT) { safe_free((void**)&H); safe_free((void**)&innovation); safe_free((void**)&S); safe_free((void**)&P_HT); safe_free((void**)&K); return -1; }
    _sf_mat_transpose(P_HT, HT_PT, es, o);
    for (int j = 0; j < es; j++) {
        if (_sf_cholesky_solve(S, HT_PT + j * o, K + j * o, o) != 0) {
            for (int i = 0; i < o; i++) K[j * o + i] = 0.0f;
        }
    }
    /* 误差状态更新: δx = K * y */
    float* Ky_es = (float*)safe_malloc(es * sizeof(float));
    if (!Ky_es) { safe_free((void**)&H); safe_free((void**)&innovation); safe_free((void**)&S); safe_free((void**)&P_HT); safe_free((void**)&K); safe_free((void**)&HT_PT); return -1; }
    _sf_mat_vec_mul(K, innovation, Ky_es, es, o);
    _sf_vec_add(eskf->error_state, Ky_es, es);
    /* 注入名义状态: x_nom = x_nom ⊕ δx */
    float* nom = eskf->nom_state;
    nom[0] += eskf->error_state[0];
    nom[1] += eskf->error_state[1];
    nom[2] += eskf->error_state[2];
    nom[3] += eskf->error_state[3];
    nom[4] += eskf->error_state[4];
    nom[5] += eskf->error_state[5];
    float dphi[3] = {eskf->error_state[6], eskf->error_state[7], eskf->error_state[8]};
    float dtheta = _sf_norm(dphi, 3);
    if (dtheta > 1e-10f) {
        float half = dtheta * 0.5f;
        float dq[4] = {cosf(half),
                       sinf(half) * dphi[0] / dtheta,
                       sinf(half) * dphi[1] / dtheta,
                       sinf(half) * dphi[2] / dtheta};
        float q_old[4] = {nom[6], nom[7], nom[8], nom[9]};
        float q_new[4];
        q_new[0] = dq[0] * q_old[0] - dq[1] * q_old[1] - dq[2] * q_old[2] - dq[3] * q_old[3];
        q_new[1] = dq[0] * q_old[1] + dq[1] * q_old[0] + dq[2] * q_old[3] - dq[3] * q_old[2];
        q_new[2] = dq[0] * q_old[2] - dq[1] * q_old[3] + dq[2] * q_old[0] + dq[3] * q_old[1];
        q_new[3] = dq[0] * q_old[3] + dq[1] * q_old[2] - dq[2] * q_old[1] + dq[3] * q_old[0];
        float qn = sqrtf(q_new[0]*q_new[0] + q_new[1]*q_new[1] + q_new[2]*q_new[2] + q_new[3]*q_new[3]);
        if (qn > 1e-10f) { nom[6]=q_new[0]/qn; nom[7]=q_new[1]/qn; nom[8]=q_new[2]/qn; nom[9]=q_new[3]/qn; }
    }
    nom[10] += eskf->error_state[9];
    nom[11] += eskf->error_state[10];
    nom[12] += eskf->error_state[11];
    nom[13] += eskf->error_state[12];
    nom[14] += eskf->error_state[13];
    nom[15] += eskf->error_state[14];
    /* 协方差更新: P = (I - K*H) * P_pred */
    float* I_KH = (float*)safe_malloc(es * es * sizeof(float));
    float* KH = (float*)safe_malloc(es * es * sizeof(float));
    if (!I_KH || !KH) { safe_free((void**)&H); safe_free((void**)&innovation); safe_free((void**)&S); safe_free((void**)&P_HT); safe_free((void**)&K);
        safe_free((void**)&HT_PT); safe_free((void**)&Ky_es); safe_free((void**)&I_KH); safe_free((void**)&KH); return -1; }
    _sf_mat_identity(I_KH, es);
    _sf_mat_mul(K, H, KH, es, o, es);
    _sf_mat_sub(I_KH, I_KH, KH, es, es);
    _sf_mat_mul(I_KH, eskf->covariance, eskf->buffer, es, es, es);
    _sf_mat_copy(eskf->covariance, eskf->buffer, es, es);
    /* 重置误差状态为0 */
    memset(eskf->error_state, 0, es * sizeof(float));
    safe_free((void**)&H); safe_free((void**)&innovation); safe_free((void**)&S); safe_free((void**)&P_HT); safe_free((void**)&K);
    safe_free((void**)&HT_PT); safe_free((void**)&Ky_es); safe_free((void**)&I_KH); safe_free((void**)&KH);
    return 0;
}

static int deep_eskf_get_state(const ESKFFilter* eskf, float* nom_state, float* error_state, float* covariance) {
    if (!eskf || !nom_state) return -1;
    int ns = eskf->config.nom_state_dim;
    int es = eskf->config.error_state_dim;
    memcpy(nom_state, eskf->nom_state, ns * sizeof(float));
    if (error_state)
        memcpy(error_state, eskf->error_state, es * sizeof(float));
    if (covariance)
        memcpy(covariance, eskf->covariance, es * es * sizeof(float));
    return 0;
}

static void eskf_reset(ESKFFilter* eskf, const float* init_nom_state) {
    if (!eskf) return;
    int ns = eskf->config.nom_state_dim;
    int es = eskf->config.error_state_dim;
    if (init_nom_state) memcpy(eskf->nom_state, init_nom_state, ns * sizeof(float));
    else memset(eskf->nom_state, 0, ns * sizeof(float));
    memset(eskf->error_state, 0, es * sizeof(float));
    _sf_mat_identity(eskf->covariance, es);
}

/* ============================================================================
 * Particle Filter - 粒子滤波器（SIR）
 * ============================================================================ */

struct DeepParticleFilter {
    ParticleFilterConfig config;
    float* particles;        /* 粒子状态 [num_particles x state_dim] */
    float* weights;          /* 粒子权重 [num_particles] */
    float* weighted_sum;     /* 加权和缓冲区 [state_dim] */
    /* CfC参数 */
    float* cfc_W_h;
    float* cfc_W_in;
    float* cfc_b_g;
    float* cfc_b_a;
    float* cfc_tau;
};

typedef struct DeepParticleFilter DeepParticleFilter;

static void deep_particle_filter_free(DeepParticleFilter* pf);

static DeepParticleFilter* deep_particle_filter_create(const ParticleFilterConfig* config) {
    if (!config) return NULL;
    DeepParticleFilter* pf = (DeepParticleFilter*)safe_calloc(1, sizeof(DeepParticleFilter));
    if (!pf) return NULL;
    memcpy(&pf->config, config, sizeof(ParticleFilterConfig));
    int s = config->state_dim;
    int n = config->num_particles;
    if (s <= 0 || n <= 0) { safe_free((void**)&pf); return NULL; }
    pf->particles = (float*)safe_calloc(n * s, sizeof(float));
    pf->weights = (float*)safe_calloc(n, sizeof(float));
    pf->weighted_sum = (float*)safe_calloc(s, sizeof(float));
    if (!pf->particles || !pf->weights || !pf->weighted_sum) {
        deep_particle_filter_free(pf);
        return NULL;
    }
    for (int i = 0; i < n; i++) pf->weights[i] = 1.0f / (float)n;
    if (config->use_cfc_motion_model) {
        int hs = config->cfc_hidden_size > 0 ? config->cfc_hidden_size : s;
        pf->cfc_W_h = (float*)safe_calloc(hs * hs, sizeof(float));
        pf->cfc_W_in = (float*)safe_calloc(hs * s, sizeof(float));
        pf->cfc_b_g = (float*)safe_calloc(hs, sizeof(float));
        pf->cfc_b_a = (float*)safe_calloc(hs, sizeof(float));
        pf->cfc_tau = (float*)safe_calloc(hs, sizeof(float));
        if (!pf->cfc_W_h || !pf->cfc_W_in || !pf->cfc_b_g ||
            !pf->cfc_b_a || !pf->cfc_tau) {
            deep_particle_filter_free(pf);
            return NULL;
        }
        for (int i = 0; i < hs; i++) {
            pf->cfc_tau[i] = 0.1f;
            /* K-012修复：安全随机数初始化 */
            pf->cfc_b_g[i] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
            pf->cfc_b_a[i] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
            for (int j = 0; j < hs; j++)
                pf->cfc_W_h[i * hs + j] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
            for (int j = 0; j < s; j++)
                pf->cfc_W_in[i * s + j] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
        }
    }
    return pf;
}

static void deep_particle_filter_free(DeepParticleFilter* pf) {
    if (!pf) return;
    safe_free((void**)&pf->particles);
    safe_free((void**)&pf->weights);
    safe_free((void**)&pf->weighted_sum);
    safe_free((void**)&pf->cfc_W_h);
    safe_free((void**)&pf->cfc_W_in);
    safe_free((void**)&pf->cfc_b_g);
    safe_free((void**)&pf->cfc_b_a);
    safe_free((void**)&pf->cfc_tau);
    safe_free((void**)&pf);
}

static int deep_particle_filter_predict(DeepParticleFilter* pf, const float* control, float dt) {
    (void)control;
    if (!pf || dt <= 0.0f) return -1;
    int s = pf->config.state_dim;
    int n = pf->config.num_particles;
    float noise_std = pf->config.process_noise_std;
    int use_cfc = (pf->config.use_cfc_motion_model && pf->cfc_W_h);
    int hs = use_cfc ? (pf->config.cfc_hidden_size > 0 ? pf->config.cfc_hidden_size : s) : s;
    for (int i = 0; i < n; i++) {
        float* particle = pf->particles + i * s;
        if (use_cfc) {
            _sf_cfc_ode_predict(particle, particle, particle,
                               pf->cfc_W_h, pf->cfc_W_in,
                               pf->cfc_b_g, pf->cfc_b_a,
                               pf->cfc_tau, hs, dt, 1);
        }
        for (int j = 0; j < s; j++) {
            /* K-012修复：安全随机数 */
            float noise = (secure_random_float() * 2.0f - 1.0f) * noise_std * sqrtf(dt);
            particle[j] += noise;
        }
    }
    return 0;
}

static int deep_particle_filter_update(DeepParticleFilter* pf, const float* observation) {
    if (!pf || !observation) return -1;
    int s = pf->config.state_dim;
    int o = pf->config.obs_dim;
    int n = pf->config.num_particles;
    float obs_noise_std = pf->config.observation_noise_std;
    float inv_sigma2 = 1.0f / (obs_noise_std * obs_noise_std + 1e-30f);
    float norm_const = 1.0f / sqrtf(2.0f * (float)M_PI * obs_noise_std * obs_noise_std + 1e-30f);
    float log_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float* particle = pf->particles + i * s;
        float sq_err = 0.0f;
        int min_dim = s < o ? s : o;
        for (int j = 0; j < min_dim; j++) {
            float diff = observation[j] - particle[j];
            sq_err += diff * diff;
        }
        pf->weights[i] *= norm_const * expf(-0.5f * sq_err * inv_sigma2);
        log_sum += pf->weights[i];
    }
    if (log_sum < 1e-30f) {
        for (int i = 0; i < n; i++) pf->weights[i] = 1.0f / (float)n;
        return 1;
    }
    float inv_sum = 1.0f / log_sum;
    for (int i = 0; i < n; i++) pf->weights[i] *= inv_sum;
    float neff = 0.0f;
    for (int i = 0; i < n; i++) neff += pf->weights[i] * pf->weights[i];
    neff = 1.0f / (neff + 1e-30f);
    float threshold = pf->config.resampling_threshold * (float)n;
    if (neff < threshold) {
        deep_particle_filter_resample(pf);
    }
    return 0;
}

static int deep_particle_filter_resample(DeepParticleFilter* pf) {
    if (!pf) return -1;
    int s = pf->config.state_dim;
    int n = pf->config.num_particles;
    float* new_particles = (float*)safe_malloc(n * s * sizeof(float));
    float* new_weights = (float*)safe_malloc(n * sizeof(float));
    if (!new_particles || !new_weights) { safe_free((void**)&new_particles); safe_free((void**)&new_weights); return -1; }
    float* cdf = (float*)safe_malloc((n + 1) * sizeof(float));
    if (!cdf) { safe_free((void**)&new_particles); safe_free((void**)&new_weights); return -1; }
    cdf[0] = 0.0f;
    for (int i = 0; i < n; i++) cdf[i + 1] = cdf[i] + pf->weights[i];
    float inv_n = 1.0f / (float)n;
    /* K-012修复：安全随机数 */
    float u0 = secure_random_float() * inv_n;
    int idx = 0;
    for (int i = 0; i < n; i++) {
        float u = u0 + (float)i * inv_n;
        while (idx < n && cdf[idx + 1] < u) idx++;
        if (idx >= n) idx = n - 1;
        memcpy(new_particles + i * s, pf->particles + idx * s, s * sizeof(float));
        new_weights[i] = inv_n;
    }
    memcpy(pf->particles, new_particles, n * s * sizeof(float));
    memcpy(pf->weights, new_weights, n * sizeof(float));
    safe_free((void**)&new_particles); safe_free((void**)&new_weights); safe_free((void**)&cdf);
    return 0;
}

static int deep_particle_filter_get_state(const DeepParticleFilter* pf, float* mean_state, float* covariance, int* effective_particles) {
    if (!pf || !mean_state) return -1;
    int s = pf->config.state_dim;
    int n = pf->config.num_particles;
    memset(mean_state, 0, s * sizeof(float));
    for (int i = 0; i < n; i++) {
        float* p = pf->particles + i * s;
        for (int j = 0; j < s; j++)
            mean_state[j] += pf->weights[i] * p[j];
    }
    if (covariance) {
        memset(covariance, 0, s * s * sizeof(float));
        for (int i = 0; i < n; i++) {
            float* p = pf->particles + i * s;
            for (int r = 0; r < s; r++)
                for (int c = 0; c < s; c++)
                    covariance[r * s + c] += pf->weights[i] * (p[r] - mean_state[r]) * (p[c] - mean_state[c]);
        }
    }
    if (effective_particles) {
        float w2 = 0.0f;
        for (int i = 0; i < n; i++) w2 += pf->weights[i] * pf->weights[i];
        *effective_particles = (int)(1.0f / (w2 + 1e-30f));
    }
    return 0;
}

static int deep_particle_filter_get_particles(const DeepParticleFilter* pf, float* particles, float* weights, int max_particles) {
    if (!pf || !particles) return -1;
    int s = pf->config.state_dim;
    int n = pf->config.num_particles;
    int copy_n = n < max_particles ? n : max_particles;
    memcpy(particles, pf->particles, copy_n * s * sizeof(float));
    if (weights)
        memcpy(weights, pf->weights, copy_n * sizeof(float));
    return copy_n;
}

static int deep_particle_filter_reset(DeepParticleFilter* pf, const float* init_mean, float init_std) {
    if (!pf || !init_mean) return -1;
    int s = pf->config.state_dim;
    int n = pf->config.num_particles;
    for (int i = 0; i < n; i++) {
        float* particle = pf->particles + i * s;
        for (int j = 0; j < s; j++) {
            /* K-012修复：安全随机数 */
            float noise = (secure_random_float() * 2.0f - 1.0f) * init_std;
            particle[j] = init_mean[j] + noise;
        }
        pf->weights[i] = 1.0f / (float)n;
    }
    return 0;
}

/* ============================================================================
 * Info Filter - 信息滤波器
 * ============================================================================ */

struct DeepInfoFilter {
    InfoFilterConfig config;
    float* information_vector;   /* 信息向量 y [state_dim] */
    float* information_matrix;   /* 信息矩阵 Y [state_dim x state_dim] */
    float* state;                /* 缓存的状态向量 [state_dim] */
    float* covariance;           /* 缓存的协方差矩阵 [state_dim x state_dim] */
    float* process_info;         /* 过程信息 Q^(-1) [state_dim x state_dim] */
    float* obs_info;             /* 观测信息 R^(-1) [obs_dim x obs_dim] */
    /* CfC参数 */
    float* cfc_W_h;
    float* cfc_W_in;
    float* cfc_b_g;
    float* cfc_b_a;
    float* cfc_tau;
    /* 工作缓冲区 */
    float* buffer;
};

typedef struct DeepInfoFilter DeepInfoFilter;

static void deep_info_filter_free(DeepInfoFilter* inf);

static DeepInfoFilter* deep_info_filter_create(const InfoFilterConfig* config) {
    if (!config) return NULL;
    DeepInfoFilter* inf = (DeepInfoFilter*)safe_calloc(1, sizeof(DeepInfoFilter));
    if (!inf) return NULL;
    memcpy(&inf->config, config, sizeof(InfoFilterConfig));
    int s = config->state_dim;
    int o = config->obs_dim;
    if (s <= 0 || o <= 0) { safe_free((void**)&inf); return NULL; }
    inf->information_vector  = (float*)safe_calloc(s, sizeof(float));
    inf->information_matrix  = (float*)safe_calloc(s * s, sizeof(float));
    inf->state      = (float*)safe_calloc(s, sizeof(float));
    inf->covariance = (float*)safe_calloc(s * s, sizeof(float));
    inf->process_info = (float*)safe_calloc(s * s, sizeof(float));
    inf->obs_info     = (float*)safe_calloc(o * o, sizeof(float));
    inf->buffer = (float*)safe_calloc(s * s * 4, sizeof(float));
    if (!inf->information_vector || !inf->information_matrix || !inf->state ||
        !inf->covariance || !inf->process_info || !inf->obs_info || !inf->buffer) {
        deep_info_filter_free(inf);
        return NULL;
    }
    _sf_mat_identity(inf->information_matrix, s);
    _sf_mat_identity(inf->covariance, s);
    _sf_mat_identity(inf->process_info, s);
    _sf_mat_scale(inf->process_info, 1.0f / (config->process_info_std * config->process_info_std + 1e-30f), s, s);
    _sf_mat_identity(inf->obs_info, o);
    _sf_mat_scale(inf->obs_info, 1.0f / (config->observation_info_std * config->observation_info_std + 1e-30f), o, o);
    if (config->use_cfc_transition) {
        int hs = config->cfc_hidden_size > 0 ? config->cfc_hidden_size : s;
        inf->cfc_W_h = (float*)safe_calloc(hs * hs, sizeof(float));
        inf->cfc_W_in = (float*)safe_calloc(hs * s, sizeof(float));
        inf->cfc_b_g = (float*)safe_calloc(hs, sizeof(float));
        inf->cfc_b_a = (float*)safe_calloc(hs, sizeof(float));
        inf->cfc_tau = (float*)safe_calloc(hs, sizeof(float));
        if (!inf->cfc_W_h || !inf->cfc_W_in || !inf->cfc_b_g ||
            !inf->cfc_b_a || !inf->cfc_tau) {
            deep_info_filter_free(inf);
            return NULL;
        }
        for (int i = 0; i < hs; i++) {
            inf->cfc_tau[i] = 0.1f;
            /* K-012修复：安全随机数初始化 */
            inf->cfc_b_g[i] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
            inf->cfc_b_a[i] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
            for (int j = 0; j < hs; j++)
                inf->cfc_W_h[i * hs + j] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
            for (int j = 0; j < s; j++)
                inf->cfc_W_in[i * s + j] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
        }
    }
    return inf;
}

static void deep_info_filter_free(DeepInfoFilter* inf) {
    if (!inf) return;
    safe_free((void**)&inf->information_vector);
    safe_free((void**)&inf->information_matrix);
    safe_free((void**)&inf->state);
    safe_free((void**)&inf->covariance);
    safe_free((void**)&inf->process_info);
    safe_free((void**)&inf->obs_info);
    safe_free((void**)&inf->cfc_W_h);
    safe_free((void**)&inf->cfc_W_in);
    safe_free((void**)&inf->cfc_b_g);
    safe_free((void**)&inf->cfc_b_a);
    safe_free((void**)&inf->cfc_tau);
    safe_free((void**)&inf->buffer);
    safe_free((void**)&inf);
}

static int deep_info_filter_predict(DeepInfoFilter* inf, const float* control, float dt) {
    (void)control;
    if (!inf || dt <= 0.0f) return -1;
    int s = inf->config.state_dim;
    /* 从信息形式恢复状态和协方差: P = Y^(-1), x = P * y */
    if (_sf_mat_inv_sym(inf->information_matrix, inf->covariance, s) != 0) {
        _sf_mat_identity(inf->covariance, s);
    }
    for (int i = 0; i < s; i++) {
        inf->state[i] = 0.0f;
        for (int j = 0; j < s; j++)
            inf->state[i] += inf->covariance[i * s + j] * inf->information_vector[j];
    }
    /* 通过CfC ODE预测状态 */
    if (inf->config.use_cfc_transition && inf->cfc_W_h) {
        int hs = inf->config.cfc_hidden_size > 0 ? inf->config.cfc_hidden_size : s;
        _sf_cfc_ode_predict(inf->state, inf->state, inf->state,
                           inf->cfc_W_h, inf->cfc_W_in,
                           inf->cfc_b_g, inf->cfc_b_a,
                           inf->cfc_tau, hs, dt, 1);
    }
    /* 数值雅可比 F */
    float* F = (float*)safe_malloc(s * s * sizeof(float));
    if (!F) return -1;
    float eps = 1e-4f;
    float* x_plus = (float*)safe_malloc(s * sizeof(float));
    float* x_minus = (float*)safe_malloc(s * sizeof(float));
    float* f_plus = (float*)safe_malloc(s * sizeof(float));
    float* f_minus = (float*)safe_malloc(s * sizeof(float));
    if (!x_plus || !x_minus || !f_plus || !f_minus) {
        safe_free((void**)&F); safe_free((void**)&x_plus); safe_free((void**)&x_minus); safe_free((void**)&f_plus); safe_free((void**)&f_minus);
        return -1;
    }
    for (int j = 0; j < s; j++) {
        memcpy(x_plus, inf->state, s * sizeof(float));
        memcpy(x_minus, inf->state, s * sizeof(float));
        x_plus[j] += eps;
        x_minus[j] -= eps;
        if (inf->config.use_cfc_transition && inf->cfc_W_h) {
            int hs = inf->config.cfc_hidden_size > 0 ? inf->config.cfc_hidden_size : s;
            _sf_cfc_ode_predict(x_plus, x_plus, f_plus,
                               inf->cfc_W_h, inf->cfc_W_in,
                               inf->cfc_b_g, inf->cfc_b_a,
                               inf->cfc_tau, hs, dt, 1);
            _sf_cfc_ode_predict(x_minus, x_minus, f_minus,
                               inf->cfc_W_h, inf->cfc_W_in,
                               inf->cfc_b_g, inf->cfc_b_a,
                               inf->cfc_tau, hs, dt, 1);
        } else {
            memcpy(f_plus, x_plus, s * sizeof(float));
            memcpy(f_minus, x_minus, s * sizeof(float));
        }
        for (int i = 0; i < s; i++)
            F[i * s + j] = (f_plus[i] - f_minus[i]) / (2.0f * eps);
    }
    /* 信息形式预测: Y_pred = (F * P * F^T + Q)^(-1), y_pred = Y_pred * (F * x) */
    float* FP = (float*)safe_malloc(s * s * sizeof(float));
    float* FPFt = (float*)safe_malloc(s * s * sizeof(float));
    float* Ft = (float*)safe_malloc(s * s * sizeof(float));
    float* P_pred = (float*)safe_malloc(s * s * sizeof(float));
    float* Q_inv = (float*)safe_malloc(s * s * sizeof(float));
    if (!FP || !FPFt || !Ft || !P_pred || !Q_inv) {
        safe_free((void**)&F); safe_free((void**)&x_plus); safe_free((void**)&x_minus); safe_free((void**)&f_plus); safe_free((void**)&f_minus);
        safe_free((void**)&FP); safe_free((void**)&FPFt); safe_free((void**)&Ft); safe_free((void**)&P_pred); safe_free((void**)&Q_inv);
        return -1;
    }
    _sf_mat_mul(F, inf->covariance, FP, s, s, s);
    _sf_mat_transpose(F, Ft, s, s);
    _sf_mat_mul(FP, Ft, FPFt, s, s, s);
    memcpy(Q_inv, inf->process_info, s * s * sizeof(float));
    _sf_mat_inv_sym(FPFt, P_pred, s);
    /* ... 简化: P_pred = FPFt + Q, 但Q已经是信息形式，需要转换
     * 实际: P_pred = F*P*F^T + Q (其中Q = process_info^(-1))
     */
    float* Q_cov = (float*)safe_malloc(s * s * sizeof(float));
    if (!Q_cov) {
        safe_free((void**)&F); safe_free((void**)&x_plus); safe_free((void**)&x_minus); safe_free((void**)&f_plus); safe_free((void**)&f_minus);
        safe_free((void**)&FP); safe_free((void**)&FPFt); safe_free((void**)&Ft); safe_free((void**)&P_pred); safe_free((void**)&Q_inv);
        return -1;
    }
    if (_sf_mat_inv_sym(inf->process_info, Q_cov, s) != 0)
        _sf_mat_identity(Q_cov, s);
    _sf_mat_add(P_pred, FPFt, Q_cov, s, s);
    /* 新信息矩阵 Y_pred = P_pred^(-1) */
    float* Y_pred = (float*)safe_malloc(s * s * sizeof(float));
    float* x_pred = (float*)safe_malloc(s * sizeof(float));
    if (!Y_pred || !x_pred) {
        safe_free((void**)&F); safe_free((void**)&x_plus); safe_free((void**)&x_minus); safe_free((void**)&f_plus); safe_free((void**)&f_minus);
        safe_free((void**)&FP); safe_free((void**)&FPFt); safe_free((void**)&Ft); safe_free((void**)&P_pred); safe_free((void**)&Q_inv); safe_free((void**)&Q_cov);
        safe_free((void**)&Y_pred); safe_free((void**)&x_pred);
        return -1;
    }
    if (_sf_mat_inv_sym(P_pred, Y_pred, s) != 0) {
        memcpy(Y_pred, inf->information_matrix, s * s * sizeof(float));
    }
    /* x_pred = F * x */
    _sf_mat_vec_mul(F, inf->state, x_pred, s, s);
    /* y_pred = Y_pred * x_pred */
    memset(inf->information_vector, 0, s * sizeof(float));
    _sf_mat_vec_mul(Y_pred, x_pred, inf->information_vector, s, s);
    memcpy(inf->information_matrix, Y_pred, s * s * sizeof(float));
    /* 更新缓存的状态和协方差 */
    if (_sf_mat_inv_sym(inf->information_matrix, inf->covariance, s) != 0)
        _sf_mat_identity(inf->covariance, s);
    for (int i = 0; i < s; i++) {
        inf->state[i] = 0.0f;
        for (int j = 0; j < s; j++)
            inf->state[i] += inf->covariance[i * s + j] * inf->information_vector[j];
    }
    safe_free((void**)&F); safe_free((void**)&x_plus); safe_free((void**)&x_minus); safe_free((void**)&f_plus); safe_free((void**)&f_minus);
    safe_free((void**)&FP); safe_free((void**)&FPFt); safe_free((void**)&Ft); safe_free((void**)&P_pred); safe_free((void**)&Q_inv); safe_free((void**)&Q_cov);
    safe_free((void**)&Y_pred); safe_free((void**)&x_pred);
    return 0;
}

static int deep_info_filter_update(DeepInfoFilter* inf, const float* observation, const float* obs_jacobian) {
    if (!inf || !observation) return -1;
    int s = inf->config.state_dim;
    int o = inf->config.obs_dim;
    /* 使用观测雅可比构建信息更新 */
    float* H = (float*)safe_malloc(o * s * sizeof(float));
    if (!H) return -1;
    if (obs_jacobian) {
        memcpy(H, obs_jacobian, o * s * sizeof(float));
    } else {
        memset(H, 0, o * s * sizeof(float));
        int min_dim = s < o ? s : o;
        for (int i = 0; i < min_dim; i++) H[i * s + i] = 1.0f;
    }
    /* 信息形式更新（加法）:
     * Y = Y_pred + H^T * R^(-1) * H
     * y = y_pred + H^T * R^(-1) * z
     */
    float* HT = (float*)safe_malloc(s * o * sizeof(float));
    float* HT_Rinv = (float*)safe_malloc(s * o * sizeof(float));
    float* HT_Rinv_H = (float*)safe_malloc(s * s * sizeof(float));
    float* HT_Rinv_z = (float*)safe_malloc(s * sizeof(float));
    float* Rinv_z = (float*)safe_malloc(o * sizeof(float));
    if (!HT || !HT_Rinv || !HT_Rinv_H || !HT_Rinv_z || !Rinv_z) {
        safe_free((void**)&H); safe_free((void**)&HT); safe_free((void**)&HT_Rinv); safe_free((void**)&HT_Rinv_H); safe_free((void**)&HT_Rinv_z); safe_free((void**)&Rinv_z);
        return -1;
    }
    _sf_mat_transpose(H, HT, o, s);
    /* HT_Rinv = H^T * R^(-1) */
    _sf_mat_mul(HT, inf->obs_info, HT_Rinv, s, o, o);
    /* HT_Rinv_H = H^T * R^(-1) * H */
    _sf_mat_mul(HT_Rinv, H, HT_Rinv_H, s, o, s);
    /* Y = Y_pred + HT_Rinv_H */
    _sf_mat_add(inf->information_matrix, inf->information_matrix, HT_Rinv_H, s, s);
    /* Rinv_z = R^(-1) * z */
    _sf_mat_vec_mul(inf->obs_info, observation, Rinv_z, o, o);
    /* HT_Rinv_z = H^T * R^(-1) * z */
    _sf_mat_vec_mul(HT, Rinv_z, HT_Rinv_z, s, o);
    _sf_vec_add(inf->information_vector, HT_Rinv_z, s);
    /* 更新缓存的状态和协方差 */
    if (_sf_mat_inv_sym(inf->information_matrix, inf->covariance, s) != 0)
        _sf_mat_identity(inf->covariance, s);
    for (int i = 0; i < s; i++) {
        inf->state[i] = 0.0f;
        for (int j = 0; j < s; j++)
            inf->state[i] += inf->covariance[i * s + j] * inf->information_vector[j];
    }
    safe_free((void**)&H); safe_free((void**)&HT); safe_free((void**)&HT_Rinv); safe_free((void**)&HT_Rinv_H); safe_free((void**)&HT_Rinv_z); safe_free((void**)&Rinv_z);
    return 0;
}

static int deep_info_filter_get_state(const DeepInfoFilter* inf, float* state, float* information_matrix, float* covariance) {
    if (!inf || !state) return -1;
    int s = inf->config.state_dim;
    memcpy(state, inf->state, s * sizeof(float));
    if (information_matrix)
        memcpy(information_matrix, inf->information_matrix, s * s * sizeof(float));
    if (covariance)
        memcpy(covariance, inf->covariance, s * s * sizeof(float));
    return 0;
}

static void deep_info_filter_reset(DeepInfoFilter* inf, const float* init_state, const float* init_covariance) {
    if (!inf) return;
    int s = inf->config.state_dim;
    if (init_state) memcpy(inf->state, init_state, s * sizeof(float));
    else memset(inf->state, 0, s * sizeof(float));
    if (init_covariance) memcpy(inf->covariance, init_covariance, s * s * sizeof(float));
    else {
        _sf_mat_identity(inf->covariance, s);
    }
    if (_sf_mat_inv_sym(inf->covariance, inf->information_matrix, s) != 0) {
        _sf_mat_identity(inf->information_matrix, s);
    }
    for (int i = 0; i < s; i++) {
        inf->information_vector[i] = 0.0f;
        for (int j = 0; j < s; j++)
            inf->information_vector[i] += inf->information_matrix[i * s + j] * inf->state[j];
    }
}

/* ============================================================================
 * 公开API：深度传感器预处理集成接口
 * ============================================================================ */

/**
 * @brief 深度传感器预处理上下文
 *
 * 封装内部静态滤波器实例，对外提供统一的传感器深度预处理接口。
 * 内部使用 CfC ODE 增强的 ESKF + 粒子滤波 + 信息滤波组合管线。
 */
typedef struct SensorDeepPreprocessor {
    ESKFFilter* eskf;
    DeepParticleFilter* pf;
    DeepInfoFilter* inf;
    int is_initialized;
    size_t state_dim;
    size_t obs_dim;
    float* output_buffer;
} SensorDeepPreprocessor;

/**
 * @brief 创建深度传感器预处理器
 *
 * @param state_dim 状态向量维度
 * @param obs_dim 观测向量维度
 * @return 预处理器句柄，失败返回NULL
 */
SensorDeepPreprocessor* sensor_deep_preprocessor_create(size_t state_dim, size_t obs_dim) {
    if (state_dim == 0 || obs_dim == 0) return NULL;

    /* 前向声明（实现见文件末尾） */
    void sensor_deep_preprocessor_free(SensorDeepPreprocessor* sdp);

    SensorDeepPreprocessor* sdp = (SensorDeepPreprocessor*)safe_calloc(1, sizeof(SensorDeepPreprocessor));
    if (!sdp) return NULL;

    sdp->state_dim = state_dim;
    sdp->obs_dim = obs_dim;

    /* 创建ESKF（含CfC ODE状态转移） */
    ESKFConfig eskf_cfg;
    memset(&eskf_cfg, 0, sizeof(ESKFConfig));
    eskf_cfg.nom_state_dim = (int)(state_dim > 16 ? state_dim : 16);
    eskf_cfg.error_state_dim = 15;
    eskf_cfg.obs_dim = (int)obs_dim;
    /* ZSFWS-M007修复: use_cfc_transition 默认值改为0
     * 标准IMU运动学(陀螺仪+加速度计积分)本身具备确定性姿态估计能力
     * CfC ODE叠加随机权重(未训练时)只会引入微小噪声扰动，长期运行产生漂移
     * 仅在CfC权重已完成训练后才启用此增强路径 */
    eskf_cfg.use_cfc_transition = 0;
    eskf_cfg.cfc_hidden_size = (int)state_dim;
    sdp->eskf = eskf_create(&eskf_cfg);
    if (!sdp->eskf) { sensor_deep_preprocessor_free(sdp); return NULL; }

    /* 创建粒子滤波器（CfC运动模型） */
    ParticleFilterConfig pf_cfg;
    memset(&pf_cfg, 0, sizeof(ParticleFilterConfig));
    pf_cfg.state_dim = (int)state_dim;
    pf_cfg.obs_dim = (int)obs_dim;
    pf_cfg.num_particles = 500;
    pf_cfg.process_noise_std = 0.01f;
    pf_cfg.observation_noise_std = 0.1f;
    pf_cfg.resampling_threshold = 0.5f;
    pf_cfg.use_cfc_motion_model = 0; /* ZSFWS-M007: 默认禁用随机CfC运动模型 */
    pf_cfg.cfc_hidden_size = (int)state_dim;
    sdp->pf = deep_particle_filter_create(&pf_cfg);
    if (!sdp->pf) { sensor_deep_preprocessor_free(sdp); return NULL; }

    /* 创建信息滤波器（CfC状态转移） */
    InfoFilterConfig inf_cfg;
    memset(&inf_cfg, 0, sizeof(InfoFilterConfig));
    inf_cfg.state_dim = (int)state_dim;
    inf_cfg.obs_dim = (int)obs_dim;
    inf_cfg.process_info_std = 0.1f;
    inf_cfg.observation_info_std = 0.01f;
    inf_cfg.use_cfc_transition = 0; /* ZSFWS-M007: 默认禁用随机CfC转移 */
    inf_cfg.cfc_hidden_size = (int)state_dim;
    sdp->inf = deep_info_filter_create(&inf_cfg);
    if (!sdp->inf) { sensor_deep_preprocessor_free(sdp); return NULL; }

    sdp->output_buffer = (float*)safe_calloc(state_dim, sizeof(float));
    if (!sdp->output_buffer) { sensor_deep_preprocessor_free(sdp); return NULL; }

    sdp->is_initialized = 1;
    return sdp;
}

/**
 * @brief 释放深度传感器预处理器
 */
void sensor_deep_preprocessor_free(SensorDeepPreprocessor* sdp) {
    if (!sdp) return;
    if (sdp->eskf) { eskf_free(sdp->eskf); sdp->eskf = NULL; }
    if (sdp->pf) { deep_particle_filter_free(sdp->pf); sdp->pf = NULL; }
    if (sdp->inf) { deep_info_filter_free(sdp->inf); sdp->inf = NULL; }
    safe_free((void**)&sdp->output_buffer);
    safe_free((void**)&sdp);
}

/**
 * @brief 深度预处理单帧传感器数据
 *
 * 通过ESKF+粒子滤波+信息滤波管线处理原始传感器数据，
 * 输出优化后的状态估计。
 *
 * @param sdp 深度预处理器句柄
 * @param raw_data 原始传感器数据 [obs_dim]
 * @param gyro 陀螺仪数据 [3]（IMU模式时使用）
 * @param acc 加速度计数据 [3]（IMU模式时使用）
 * @param dt 时间步长
 * @param output 输出状态估计 [state_dim]
 * @return 0成功，-1失败
 */
int sensor_deep_preprocess_frame(SensorDeepPreprocessor* sdp,
                                  const float* raw_data,
                                  const float* gyro,
                                  const float* acc,
                                  float dt,
                                  float* output) {
    if (!sdp || !sdp->is_initialized || !raw_data || !output) return -1;
    if (dt <= 0.0f) dt = 0.01f;

    /* ESKF预测+更新（IMU运动学模式） */
    if (sdp->eskf && gyro && acc) {
        eskf_predict_nominal(sdp->eskf, gyro, acc, dt);
        eskf_predict_error(sdp->eskf, dt);
        eskf_update(sdp->eskf, raw_data, 0);
        float nom_state[64];
        float error_state[32];
        deep_eskf_get_state(sdp->eskf, nom_state, error_state, NULL);
        /* 拷贝ESKF名义状态前sdp->state_dim维 */
        size_t copy_dim = sdp->state_dim < 16 ? sdp->state_dim : 16;
        memcpy(output, nom_state, copy_dim * sizeof(float));
    }

    /* 粒子滤波预测+更新 */
    if (sdp->pf) {
        deep_particle_filter_predict(sdp->pf, NULL, dt);
        deep_particle_filter_update(sdp->pf, raw_data);
        float pf_state[64];
        int effective;
        deep_particle_filter_get_state(sdp->pf, pf_state, NULL, &effective);
        /* 融合粒子滤波估计 */
        for (size_t i = 0; i < sdp->state_dim && i < 64; i++) {
            output[i] = output[i] * 0.5f + pf_state[i] * 0.5f;
        }
    }

    /* 信息滤波预测+更新 */
    if (sdp->inf) {
        deep_info_filter_predict(sdp->inf, NULL, dt);
        deep_info_filter_update(sdp->inf, raw_data, NULL);
        float inf_state[64];
        deep_info_filter_get_state(sdp->inf, inf_state, NULL, NULL);
        /* 三重滤波融合平均 */
        for (size_t i = 0; i < sdp->state_dim && i < 64; i++) {
            output[i] = output[i] * 0.667f + inf_state[i] * 0.333f;
        }
    }

    return 0;
}

/**
 * @brief 深度传感器预处理信号质量评估
 *
 * 对各传感器模态进行深度信号质量分析，辅助自适应路由决策。
 *
 * @param sdp 深度预处理器句柄
 * @param modality_data 各模态原始数据 [4][dim]
 * @param modality_dims 各模态数据维度 [4]
 * @param quality_scores 输出质量评分 [4]
 * @return 0成功，-1失败
 */
int sensor_deep_quality_assess(SensorDeepPreprocessor* sdp,
                                const float** modality_data,
                                const size_t* modality_dims,
                                float* quality_scores) {
    if (!sdp || !modality_data || !modality_dims || !quality_scores) return -1;

    for (int m = 0; m < 4; m++) {
        if (!modality_data[m] || modality_dims[m] == 0) {
            quality_scores[m] = 0.0f;
            continue;
        }

        /* 基于信号统计特性评估质量 */
        const float* data = modality_data[m];
        size_t n = modality_dims[m];

        float sum = 0.0f, sum_sq = 0.0f;
        int nonzero = 0;
        for (size_t i = 0; i < n; i++) {
            float v = data[i];
            sum += v;
            sum_sq += v * v;
            if (fabsf(v) > 1e-6f) nonzero++;
        }

        float mean = sum / (float)n;
        float variance = sum_sq / (float)n - mean * mean;
        if (variance < 0.0f) variance = 0.0f;
        float energy = sum_sq / (float)n;
        float sparsity = (float)(n - nonzero) / (float)n;

        /* 综合质量 = 能量*0.3 + 1/稀疏度*0.3 + 方差归一化*0.2 + 信号存在性*0.2 */
        float q_energy = (energy > 0.0f) ? (1.0f - expf(-energy * 10.0f)) : 0.0f;
        float q_sparsity = 1.0f - sparsity;
        float q_var = (variance > 0.0f) ? (1.0f - expf(-variance * 5.0f)) : 0.0f;
        float q_present = (nonzero > (int)(n / 4)) ? 1.0f : ((float)nonzero / (float)(n / 4));

        quality_scores[m] = 0.3f * q_energy + 0.3f * q_sparsity + 0.2f * q_var + 0.2f * q_present;
        if (quality_scores[m] > 1.0f) quality_scores[m] = 1.0f;
        if (quality_scores[m] < 0.0f) quality_scores[m] = 0.0f;
    }

    return 0;
}
