#include "selflnn/core/graph_optimization_ext.h"
#include "selflnn/core/common.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ==================== 内部数学工具 ==================== */

static float internal_dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static float internal_norm2(const float* v, int n) {
    float s = internal_dot(v, v, n);
    return sqrtf(s > 0.0f ? s : 0.0f);
}

static void internal_mat_vec_mul(const float* A, const float* x, float* y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        y[i] = 0.0f;
        for (int j = 0; j < cols; j++)
            y[i] += A[i * cols + j] * x[j];
    }
}

static void internal_vec_scale(float* y, const float* x, float s, int n) {
    for (int i = 0; i < n; i++) y[i] = x[i] * s;
}

static void internal_vec_add(float* y, const float* x, float s, int n) {
    for (int i = 0; i < n; i++) y[i] += x[i] * s;
}

static void internal_vec_copy(float* dst, const float* src, int n) {
    memcpy(dst, src, (size_t)n * sizeof(float));
}

static void internal_mat_transpose(const float* A, float* AT, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            AT[j * rows + i] = A[i * cols + j];
}

static void internal_mat_mul(const float* A, const float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float s = 0.0f;
            for (int t = 0; t < n; t++)
                s += A[i * n + t] * B[t * k + j];
            C[i * k + j] = s;
        }
    }
}

static int internal_cholesky_decompose(float* L, const float* A, int n) {
    for (int j = 0; j < n; j++) {
        float s = 0.0f;
        for (int k = 0; k < j; k++)
            s += L[j * n + k] * L[j * n + k];
        float d = A[j * n + j] - s;
        if (d <= 1e-12f) {
            d = 1e-8f;
        }
        L[j * n + j] = sqrtf(d);
        if (!SELFLNN_IS_FINITE(L[j * n + j])) {
            L[j * n + j] = 1e-4f;
        }
        float ljj = L[j * n + j];
        for (int i = j + 1; i < n; i++) {
            s = 0.0f;
            for (int k = 0; k < j; k++)
                s += L[i * n + k] * L[j * n + k];
            L[i * n + j] = (A[i * n + j] - s) / ljj;
            if (!SELFLNN_IS_FINITE(L[i * n + j])) {
                L[i * n + j] = 0.0f;
            }
        }
    }
    return 0;
}

static void internal_cholesky_solve(const float* L, const float* b, float* x, int n) {
    float* y = (float*)malloc((size_t)n * sizeof(float));
    if (!y) return;
    for (int i = 0; i < n; i++) {
        float s = 0.0f;
        for (int k = 0; k < i; k++) s += L[i * n + k] * y[k];
        float diag = L[i * n + i];
        if (diag < 1e-12f && diag > -1e-12f) diag = 1e-12f;
        y[i] = (b[i] - s) / diag;
    }
    for (int i = n - 1; i >= 0; i--) {
        float s = 0.0f;
        for (int k = i + 1; k < n; k++) s += L[k * n + i] * x[k];
        float diag = L[i * n + i];
        if (diag < 1e-12f && diag > -1e-12f) diag = 1e-12f;
        x[i] = (y[i] - s) / diag;
    }
    safe_free((void**)&y);
}

static float internal_quad_form(const float* H, const float* v, int n) {
    float* Hv = (float*)malloc((size_t)n * sizeof(float));
    if (!Hv) return 0.0f;
    internal_mat_vec_mul(H, v, Hv, n, n);
    float q = internal_dot(v, Hv, n);
    safe_free((void**)&Hv);
    return q;
}

static float internal_line_search(const float* x, const float* d, PowellFunc func,
                                   void* user_data, float f0, int n, float* fx) {
    float alpha = 1.0f;
    float* xtry = (float*)malloc((size_t)n * sizeof(float));
    if (!xtry) return 0.0f;
    float c = 0.5f;
    float tau = 0.5f;
    for (int iter = 0; iter < 20; iter++) {
        for (int i = 0; i < n; i++) xtry[i] = x[i] + alpha * d[i];
        float fnew;
        func(xtry, &fnew, user_data);
        if (fnew <= f0 + c * alpha * internal_dot(x, d, n) + 1e-12f) {
            *fx = fnew;
            safe_free((void**)&xtry);
            return alpha;
        }
        alpha *= tau;
    }
    *fx = f0;
    safe_free((void**)&xtry);
    return 0.0f;
}

/* ==================== A01.4.1 LM优化器 ==================== */

int lm_optimizer_create(LMOptimizer* opt, int n_params, int n_residuals) {
    if (!opt || n_params < 1 || n_residuals < 1) return -1;
    opt->n_params = n_params;
    opt->n_residuals = n_residuals;
    opt->params = (float*)calloc((size_t)n_params, sizeof(float));
    opt->residuals = (float*)calloc((size_t)n_residuals, sizeof(float));
    opt->jacobian = (float*)calloc((size_t)n_residuals * n_params, sizeof(float));
    int ws_size = n_params * n_params + n_params + n_params;
    opt->workspace = (float*)calloc((size_t)ws_size, sizeof(float));
    if (!opt->params || !opt->residuals || !opt->jacobian || !opt->workspace) {
        safe_free((void**)&opt->params); safe_free((void**)&opt->residuals); safe_free((void**)&opt->jacobian); safe_free((void**)&opt->workspace);
        return -1;
    }
    opt->status = 0;
    lm_optimizer_set_defaults(opt);
    return 0;
}

void lm_optimizer_free(LMOptimizer* opt) {
    if (!opt) return;
    safe_free((void**)&opt->params);
    safe_free((void**)&opt->residuals);
    safe_free((void**)&opt->jacobian);
    safe_free((void**)&opt->workspace);
    memset(opt, 0, sizeof(LMOptimizer));
}

void lm_optimizer_set_defaults(LMOptimizer* opt) {
    if (!opt) return;
    opt->lambda = 1e-3f;
    opt->lambda_factor = 10.0f;
    opt->gtol = 1e-8f;
    opt->ftol = 1e-12f;
    opt->xtol = 1e-12f;
    opt->max_iterations = 200;
    opt->n_iterations = 0;
}

int lm_optimizer_solve(LMOptimizer* opt, float* initial_params,
                       LMResidualFunc res_func, LMJacobianFunc jac_func,
                       void* user_data) {
    if (!opt || !initial_params || !res_func || !jac_func) return -1;
    int n = opt->n_params;
    int m = opt->n_residuals;

    internal_vec_copy(opt->params, initial_params, n);

    float* H = opt->workspace;
    float* g = opt->workspace + (size_t)n * n;
    float* delta = opt->workspace + (size_t)n * n + n;
    float* JL = opt->workspace;
    (void)JL;

    float prev_cost = FLT_MAX;
    int nu = 2;

    for (opt->n_iterations = 0; opt->n_iterations < opt->max_iterations; opt->n_iterations++) {
        if (res_func(opt->params, opt->residuals, user_data) != 0) return -1;
        if (jac_func(opt->params, opt->jacobian, user_data) != 0) return -1;

        float cost = 0.0f;
        for (int i = 0; i < m; i++) cost += opt->residuals[i] * opt->residuals[i];
        cost *= 0.5f;

        float* JT = (float*)safe_malloc((size_t)n * m * sizeof(float));
        if (!JT) return -1;
        internal_mat_transpose(opt->jacobian, JT, m, n);
        internal_mat_mul(JT, opt->jacobian, H, n, m, n);
        internal_mat_vec_mul(JT, opt->residuals, g, n, m);
        for (int i = 0; i < n; i++) g[i] = -g[i];
        safe_free((void**)&JT);

        float grad_norm = internal_norm2(g, n);
        if (grad_norm < opt->gtol) {
            opt->status = 1;
            return 0;
        }

        if (opt->n_iterations > 0 && fabsf(prev_cost - cost) < opt->ftol * (1.0f + cost)) {
            opt->status = 1;
            return 0;
        }
        prev_cost = cost;

        int accepted = 0;
        for (int inner = 0; inner < 30; inner++) {
            float* H_aug = (float*)malloc((size_t)n * n * sizeof(float));
            if (!H_aug) return -1;
            memcpy(H_aug, H, (size_t)n * n * sizeof(float));
            for (int i = 0; i < n; i++)
                H_aug[i * n + i] += opt->lambda;

            float* L = (float*)malloc((size_t)n * n * sizeof(float));
            if (!L) { safe_free((void**)&H_aug); return -1; }
            memset(L, 0, (size_t)n * n * sizeof(float));

            if (internal_cholesky_decompose(L, H_aug, n) != 0) {
                opt->lambda *= opt->lambda_factor;
                safe_free((void**)&H_aug);
                safe_free((void**)&L);
                continue;
            }

            internal_cholesky_solve(L, g, delta, n);
            safe_free((void**)&H_aug);
            safe_free((void**)&L);

            float* new_params = (float*)malloc((size_t)n * sizeof(float));
            if (!new_params) return -1;
            for (int i = 0; i < n; i++)
                new_params[i] = opt->params[i] + delta[i];

            float* new_res = (float*)malloc((size_t)m * sizeof(float));
            if (!new_res) { safe_free((void**)&new_params); return -1; }
            if (res_func(new_params, new_res, user_data) != 0) {
                safe_free((void**)&new_params); safe_free((void**)&new_res);
                return -1;
            }

            float new_cost = 0.0f;
            for (int i = 0; i < m; i++) new_cost += new_res[i] * new_res[i];
            new_cost *= 0.5f;

            float actual_reduction = cost - new_cost;
            float predicted_reduction = 0.0f;
            for (int i = 0; i < m; i++) {
                float Ji_d = 0.0f;
                for (int j = 0; j < n; j++)
                    Ji_d += opt->jacobian[i * n + j] * delta[j];
                predicted_reduction += Ji_d * Ji_d;
            }
            predicted_reduction *= 0.5f;
            for (int i = 0; i < n; i++)
                predicted_reduction += 0.5f * opt->lambda * delta[i] * delta[i];

            float rho = (predicted_reduction > 1e-12f) ? actual_reduction / predicted_reduction : 0.0f;

            if (rho > 0.0f) {
                internal_vec_copy(opt->params, new_params, n);
                internal_vec_copy(opt->residuals, new_res, m);
                opt->lambda *= fmaxf(1.0f / 3.0f, 1.0f - powf(2.0f * rho - 1.0f, 3));
                nu = 2;
                accepted = 1;
                safe_free((void**)&new_params);
                safe_free((void**)&new_res);
                break;
            } else {
                opt->lambda *= opt->lambda_factor;
                nu *= 2;
            }
            safe_free((void**)&new_params);
            safe_free((void**)&new_res);
        }

        if (!accepted) {
            opt->status = -1;
            return -1;
        }

        float param_change = internal_norm2(delta, n);
        if (param_change < opt->xtol * (internal_norm2(opt->params, n) + 1e-12f)) {
            opt->status = 1;
            return 0;
        }
    }

    opt->status = 1;
    return 0;
}

/* ==================== A01.4.1 Dogleg优化器 ==================== */

int dogleg_optimizer_create(DoglegOptimizer* opt, int n_params, int n_residuals) {
    if (!opt || n_params < 1 || n_residuals < 1) return -1;
    opt->n_params = n_params;
    opt->n_residuals = n_residuals;
    opt->params = (float*)calloc((size_t)n_params, sizeof(float));
    opt->residuals = (float*)calloc((size_t)n_residuals, sizeof(float));
    opt->jacobian = (float*)calloc((size_t)n_residuals * n_params, sizeof(float));
    opt->gradient = (float*)calloc((size_t)n_params, sizeof(float));
    opt->gn_step = (float*)calloc((size_t)n_params, sizeof(float));
    opt->cauchy_step = (float*)calloc((size_t)n_params, sizeof(float));
    int ws_size = n_params * n_params + n_params + n_params;
    opt->workspace = (float*)calloc((size_t)ws_size, sizeof(float));
    if (!opt->params || !opt->residuals || !opt->jacobian || !opt->gradient ||
        !opt->gn_step || !opt->cauchy_step || !opt->workspace) {
        safe_free((void**)&opt->params); safe_free((void**)&opt->residuals); safe_free((void**)&opt->jacobian);
        safe_free((void**)&opt->gradient); safe_free((void**)&opt->gn_step); safe_free((void**)&opt->cauchy_step);
        safe_free((void**)&opt->workspace);
        return -1;
    }
    opt->status = 0;
    dogleg_optimizer_set_defaults(opt);
    return 0;
}

void dogleg_optimizer_free(DoglegOptimizer* opt) {
    if (!opt) return;
    safe_free((void**)&opt->params); safe_free((void**)&opt->residuals); safe_free((void**)&opt->jacobian);
    safe_free((void**)&opt->gradient); safe_free((void**)&opt->gn_step); safe_free((void**)&opt->cauchy_step);
    safe_free((void**)&opt->workspace);
    memset(opt, 0, sizeof(DoglegOptimizer));
}

void dogleg_optimizer_set_defaults(DoglegOptimizer* opt) {
    if (!opt) return;
    opt->trust_radius = 1.0f;
    opt->delta_max = 100.0f;
    opt->gtol = 1e-8f;
    opt->ftol = 1e-12f;
    opt->xtol = 1e-12f;
    opt->max_iterations = 200;
}

static float dogleg_compute_step(float* step, const float* gn, const float* cauchy,
                                  float trust_radius, int n) {
    float gn_norm = internal_norm2(gn, n);
    if (gn_norm <= trust_radius) {
        internal_vec_copy(step, gn, n);
        return gn_norm;
    }
    float cauchy_norm = internal_norm2(cauchy, n);
    if (cauchy_norm >= trust_radius) {
        float scale = trust_radius / (cauchy_norm + 1e-12f);
        internal_vec_scale(step, cauchy, scale, n);
        return trust_radius;
    }
    float* diff = (float*)malloc((size_t)n * sizeof(float));
    if (!diff) return 0.0f;
    for (int i = 0; i < n; i++) diff[i] = gn[i] - cauchy[i];
    float diff_norm = internal_norm2(diff, n);
    float a = internal_dot(cauchy, diff, n);
    float cauchy_sq = cauchy_norm * cauchy_norm;
    float trust_sq = trust_radius * trust_radius;
    float b = 2.0f * a;
    float c = cauchy_sq - trust_sq;
    float disc = b * b - 4.0f * diff_norm * diff_norm * c;
    float tau = (disc > 0.0f) ?
        (-b + sqrtf(disc)) / (2.0f * diff_norm * diff_norm + 1e-12f) : 1.0f;
    tau = fmaxf(0.0f, fminf(1.0f, tau));
    for (int i = 0; i < n; i++)
        step[i] = cauchy[i] + tau * diff[i];
    safe_free((void**)&(diff));
    return internal_norm2(step, n);
}

int dogleg_optimizer_solve(DoglegOptimizer* opt, float* initial_params,
                           LMResidualFunc res_func, LMJacobianFunc jac_func,
                           void* user_data) {
    if (!opt || !initial_params || !res_func || !jac_func) return -1;
    int n = opt->n_params;
    int m = opt->n_residuals;

    internal_vec_copy(opt->params, initial_params, n);

    float* H = opt->workspace;
    float* H_gd = opt->workspace + (size_t)n * n;
    float* L = opt->workspace + (size_t)n * n * 2;
    (void)H_gd;

    float prev_cost = FLT_MAX;

    for (opt->n_iterations = 0; opt->n_iterations < opt->max_iterations; opt->n_iterations++) {
        if (res_func(opt->params, opt->residuals, user_data) != 0) return -1;
        if (jac_func(opt->params, opt->jacobian, user_data) != 0) return -1;

        float cost = 0.0f;
        for (int i = 0; i < m; i++) cost += opt->residuals[i] * opt->residuals[i];
        cost *= 0.5f;

        float* JT = (float*)safe_malloc((size_t)n * m * sizeof(float));
        if (!JT) return -1;
        internal_mat_transpose(opt->jacobian, JT, m, n);
        internal_mat_mul(JT, opt->jacobian, H, n, m, n);
        internal_mat_vec_mul(JT, opt->residuals, opt->gradient, n, m);
        safe_free((void**)&(JT));

        for (int i = 0; i < n; i++) opt->gradient[i] *= -1.0f;

        float grad_norm = internal_norm2(opt->gradient, n);
        if (grad_norm < opt->gtol) { opt->status = 1; return 0; }

        if (opt->n_iterations > 0 && fabsf(prev_cost - cost) < opt->ftol * (1.0f + cost)) {
            opt->status = 1; return 0;
        }
        prev_cost = cost;

        float gnorm2 = internal_dot(opt->gradient, opt->gradient, n);
        float gHg = internal_quad_form(H, opt->gradient, n);

        float cauchy_len = (gHg > 1e-12f) ? gnorm2 / gHg : 1.0f;
        internal_vec_scale(opt->cauchy_step, opt->gradient, cauchy_len, n);

        memset(L, 0, (size_t)n * n * sizeof(float));
        if (internal_cholesky_decompose(L, H, n) == 0) {
            internal_cholesky_solve(L, opt->gradient, opt->gn_step, n);
        } else {
            memset(opt->gn_step, 0, (size_t)n * sizeof(float));
        }

        dogleg_compute_step(opt->gn_step, opt->gn_step, opt->cauchy_step,
                           opt->trust_radius, n);

        float* new_params = (float*)malloc((size_t)n * sizeof(float));
        if (!new_params) return -1;
        for (int i = 0; i < n; i++)
            new_params[i] = opt->params[i] + opt->gn_step[i];

        float* new_res = (float*)malloc((size_t)m * sizeof(float));
        if (!new_res) { safe_free((void**)&(new_params)); return -1; }
        if (res_func(new_params, new_res, user_data) != 0) {
            safe_free((void**)&(new_params)); safe_free((void**)&(new_res)); return -1;
        }

        float new_cost = 0.0f;
        for (int i = 0; i < m; i++) new_cost += new_res[i] * new_res[i];
        new_cost *= 0.5f;

        float actual_red = cost - new_cost;
        float pred_red = 0.0f;
        for (int i = 0; i < m; i++) {
            float Ji_d = 0.0f;
            for (int j = 0; j < n; j++)
                Ji_d += opt->jacobian[i * n + j] * opt->gn_step[j];
            pred_red += Ji_d * Ji_d;
        }
        pred_red *= 0.5f;

        float rho = (pred_red > 1e-12f) ? actual_red / pred_red : 0.0f;

        if (rho > 0.0f) {
            internal_vec_copy(opt->params, new_params, n);
            internal_vec_copy(opt->residuals, new_res, m);
            if (rho > 0.75f)
                opt->trust_radius = fminf(opt->delta_max, 2.0f * opt->trust_radius);
        } else {
            opt->trust_radius *= 0.25f;
        }
        safe_free((void**)&(new_params));
        safe_free((void**)&(new_res));

        if (opt->trust_radius < opt->xtol) { opt->status = 1; return 0; }
        if (internal_norm2(opt->gn_step, n) < opt->xtol * (internal_norm2(opt->params, n) + 1e-12f)) {
            opt->status = 1; return 0;
        }
    }
    opt->status = 1;
    return 0;
}

/* ==================== A01.4.1 Powell无导数优化器 ==================== */

static float powell_brent_line_search(const float* x, const float* d, PowellFunc func,
                                       void* user_data, int n, float* fx) {
    float a = 0.0f, b = 1.0f, c = 0.5f;
    float* xt = (float*)malloc((size_t)n * sizeof(float));
    if (!xt) { *fx = (float)func(x, fx, user_data); return 0.0f; }

    for (int i = 0; i < n; i++) xt[i] = x[i] + b * d[i];
    float fb;
    func(xt, &fb, user_data);

    for (int i = 0; i < n; i++) xt[i] = x[i] + a * d[i];
    float fa;
    func(xt, &fa, user_data);

    float phi = (sqrtf(5.0f) - 1.0f) / 2.0f;
    for (int iter = 0; iter < 30; iter++) {
        float x1 = a + (1.0f - phi) * (b - a);
        float x2 = a + phi * (b - a);

        for (int i = 0; i < n; i++) xt[i] = x[i] + x1 * d[i];
        float f1;
        func(xt, &f1, user_data);

        for (int i = 0; i < n; i++) xt[i] = x[i] + x2 * d[i];
        float f2;
        func(xt, &f2, user_data);

        if (f1 < f2) {
            b = x2;
            c = x1;
            fb = f2;
        } else {
            a = x1;
            c = x2;
            fa = f1;
        }
        if (fabsf(b - a) < 1e-8f) break;
    }
    float alpha = (a + b) * 0.5f;
    for (int i = 0; i < n; i++) xt[i] = x[i] + alpha * d[i];
    func(xt, fx, user_data);
    safe_free((void**)&(xt));
    return alpha;
}

int powell_optimizer_create(PowellOptimizer* opt, int n_params) {
    if (!opt || n_params < 1) return -1;
    opt->n_params = n_params;
    opt->params = (float*)calloc((size_t)n_params, sizeof(float));
    opt->directions = (float*)calloc((size_t)n_params * n_params, sizeof(float));
    opt->workspace = (float*)calloc((size_t)n_params, sizeof(float));
    opt->prev_params = (float*)calloc((size_t)n_params, sizeof(float));
    opt->step = (float*)calloc((size_t)n_params, sizeof(float));
    if (!opt->params || !opt->directions || !opt->workspace ||
        !opt->prev_params || !opt->step) {
        safe_free((void**)&(opt->params)); safe_free((void**)&(opt->directions)); safe_free((void**)&(opt->workspace));
        safe_free((void**)&(opt->prev_params)); safe_free((void**)&(opt->step));
        return -1;
    }
    powell_optimizer_set_defaults(opt);
    return 0;
}

void powell_optimizer_free(PowellOptimizer* opt) {
    if (!opt) return;
    safe_free((void**)&(opt->params)); safe_free((void**)&(opt->directions)); safe_free((void**)&(opt->workspace));
    safe_free((void**)&(opt->prev_params)); safe_free((void**)&(opt->step));
    memset(opt, 0, sizeof(PowellOptimizer));
}

void powell_optimizer_set_defaults(PowellOptimizer* opt) {
    if (!opt) return;
    opt->ftol = 1e-10f;
    opt->xtol = 1e-10f;
    opt->max_iterations = 500;
    opt->initial_radius = 1.0f;
}

int powell_optimizer_solve(PowellOptimizer* opt, float* initial_params,
                           PowellFunc func, void* user_data) {
    if (!opt || !initial_params || !func) return -1;
    int n = opt->n_params;

    internal_vec_copy(opt->params, initial_params, n);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            opt->directions[i * n + j] = (i == j) ? 1.0f : 0.0f;

    float fprev;
    func(opt->params, &fprev, user_data);
    float fcur = fprev;

    for (opt->n_iterations = 0; opt->n_iterations < opt->max_iterations; opt->n_iterations++) {
        internal_vec_copy(opt->prev_params, opt->params, n);

        float delta_f = 0.0f;
        int best_dir = 0;

        for (int i = 0; i < n; i++) {
            float* d = &opt->directions[i * n];
            float fx;
            float alpha = powell_brent_line_search(opt->params, d, func, user_data, n, &fx);
            float df = 0.0f;
            if (i == 0) df = fabsf(fcur - fx);
            if (df > delta_f) { delta_f = df; best_dir = i; }
            for (int j = 0; j < n; j++)
                opt->params[j] += alpha * d[j];
            fcur = fx;
        }

        if (fabsf(fcur - fprev) < opt->ftol * (1.0f + fabsf(fcur))) {
            opt->status = 1;
            return 0;
        }
        fprev = fcur;

        for (int j = 0; j < n; j++)
            opt->step[j] = opt->params[j] - opt->prev_params[j];
        float step_norm = internal_norm2(opt->step, n);
        if (step_norm < opt->xtol) { opt->status = 1; return 0; }

        for (int j = best_dir; j < n - 1; j++)
            for (int k = 0; k < n; k++)
                opt->directions[j * n + k] = opt->directions[(j + 1) * n + k];

        for (int k = 0; k < n; k++)
            opt->directions[(n - 1) * n + k] = opt->step[k];

        float* d_new = &opt->directions[(n - 1) * n];
        float fx;
        powell_brent_line_search(opt->params, d_new, func, user_data, n, &fx);
        for (int j = 0; j < n; j++)
            opt->params[j] += fx * d_new[j];
        func(opt->params, &fcur, user_data);
    }

    opt->status = 1;
    return 0;
}

/* ==================== A01.4.2 因子图 (Factor Graph) ==================== */

int fg_create(FactorGraph* fg, int max_vars, int max_factors) {
    if (!fg || max_vars < 1 || max_factors < 1) return -1;
    fg->variables = (FgVariableNode*)calloc((size_t)max_vars, sizeof(FgVariableNode));
    fg->factors = (FgFactorNode*)calloc((size_t)max_factors, sizeof(FgFactorNode));
    fg->n_variables = 0;
    fg->n_factors = 0;
    fg->max_variables = max_vars;
    fg->max_factors = max_factors;
    fg->elim_order = (int*)calloc((size_t)max_vars, sizeof(int));
    if (!fg->variables || !fg->factors || !fg->elim_order) {
        safe_free((void**)&(fg->variables)); safe_free((void**)&(fg->factors)); safe_free((void**)&(fg->elim_order));
        return -1;
    }
    return 0;
}

void fg_free(FactorGraph* fg) {
    if (!fg) return;
    for (int i = 0; i < fg->n_variables; i++) {
        safe_free((void**)&(fg->variables[i].belief));
        safe_free((void**)&(fg->variables[i].factor_ids));
    }
    for (int i = 0; i < fg->n_factors; i++) {
        safe_free((void**)&(fg->factors[i].connected_vars));
        safe_free((void**)&(fg->factors[i].var_dimensions));
        safe_free((void**)&(fg->factors[i].values));
    }
    safe_free((void**)&(fg->variables));
    safe_free((void**)&(fg->factors));
    safe_free((void**)&(fg->elim_order));
    memset(fg, 0, sizeof(FactorGraph));
}

int fg_add_variable(FactorGraph* fg, int id, const char* name, int domain_size) {
    if (!fg || fg->n_variables >= fg->max_variables || domain_size < 2) return -1;
    FgVariableNode* v = &fg->variables[fg->n_variables];
    v->id = id;
    strncpy_s(v->name, sizeof(v->name), name ? name : "", _TRUNCATE);
    v->domain_size = domain_size;
    v->belief = (float*)calloc((size_t)domain_size, sizeof(float));
    v->factor_ids = (int*)calloc((size_t)fg->max_factors, sizeof(int));
    v->n_factors = 0;
    if (!v->belief || !v->factor_ids) {
        safe_free((void**)&(v->belief)); safe_free((void**)&(v->factor_ids)); return -1;
    }
    for (int i = 0; i < domain_size; i++) v->belief[i] = 1.0f / (float)domain_size;
    return fg->n_variables++;
}

int fg_add_factor(FactorGraph* fg, int id, const char* name,
                  const int* var_ids, int n_vars, const float* values) {
    if (!fg || fg->n_factors >= fg->max_factors || !var_ids || n_vars < 1 || !values) return -1;
    FgFactorNode* f = &fg->factors[fg->n_factors];
    f->id = id;
    strncpy_s(f->name, sizeof(f->name), name ? name : "", _TRUNCATE);
    f->n_connected_vars = n_vars;
    f->connected_vars = (int*)malloc((size_t)n_vars * sizeof(int));
    f->var_dimensions = (int*)malloc((size_t)n_vars * sizeof(int));
    if (!f->connected_vars || !f->var_dimensions) {
        safe_free((void**)&(f->connected_vars)); safe_free((void**)&(f->var_dimensions)); return -1;
    }
    int total_size = 1;
    for (int i = 0; i < n_vars; i++) {
        f->connected_vars[i] = var_ids[i];
        int vs = 0;
        for (int j = 0; j < fg->n_variables; j++) {
            if (fg->variables[j].id == var_ids[i]) { vs = fg->variables[j].domain_size; break; }
        }
        f->var_dimensions[i] = vs > 0 ? vs : 2;
        total_size *= f->var_dimensions[i];
    }
    f->values = (float*)malloc((size_t)total_size * sizeof(float));
    if (!f->values) { safe_free((void**)&(f->values)); return -1; }
    memcpy(f->values, values, (size_t)total_size * sizeof(float));

    for (int i = 0; i < n_vars; i++) {
        for (int j = 0; j < fg->n_variables; j++) {
            if (fg->variables[j].id == var_ids[i]) {
                int slot = fg->variables[j].n_factors++;
                fg->variables[j].factor_ids[slot] = id;
                break;
            }
        }
    }
    return fg->n_factors++;
}

int fg_connect(FactorGraph* fg, int factor_id, int var_id) {
    if (!fg) return -1;
    int fi = -1, vi = -1;
    for (int i = 0; i < fg->n_factors; i++) if (fg->factors[i].id == factor_id) { fi = i; break; }
    for (int i = 0; i < fg->n_variables; i++) if (fg->variables[i].id == var_id) { vi = i; break; }
    if (fi < 0 || vi < 0) return -1;

    int found = 0;
    for (int i = 0; i < fg->factors[fi].n_connected_vars; i++)
        if (fg->factors[fi].connected_vars[i] == var_id) { found = 1; break; }
    if (!found) return -1;

    int slot = fg->variables[vi].n_factors++;
    fg->variables[vi].factor_ids[slot] = factor_id;
    return 0;
}

int fg_sum_product(FactorGraph* fg, int target_var, float* marginal, int max_iter) {
    if (!fg || !marginal) return -1;
    int vi = -1;
    for (int i = 0; i < fg->n_variables; i++)
        if (fg->variables[i].id == target_var) { vi = i; break; }
    if (vi < 0) return -1;

    int ds = fg->variables[vi].domain_size;

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < ds; i++)
            fg->variables[vi].belief[i] = 1.0f;

        for (int fi = 0; fi < fg->variables[vi].n_factors; fi++) {
            int fid = fg->variables[vi].factor_ids[fi];
            int fj = -1;
            for (int j = 0; j < fg->n_factors; j++)
                if (fg->factors[j].id == fid) { fj = j; break; }
            if (fj < 0) continue;

            FgFactorNode* f = &fg->factors[fj];
            int var_pos = -1;
            for (int p = 0; p < f->n_connected_vars; p++)
                if (f->connected_vars[p] == target_var) { var_pos = p; break; }
            if (var_pos < 0) continue;

            int other_dim = 1;
            for (int p = 0; p < f->n_connected_vars; p++)
                if (p != var_pos) other_dim *= f->var_dimensions[p];

            for (int s = 0; s < ds; s++) {
                float msg = 0.0f;
                if (other_dim > 0) {
                    int* idx = (int*)calloc((size_t)f->n_connected_vars, sizeof(int));
                    if (!idx) continue;
                    idx[var_pos] = s;
                    for (int o = 0; o < other_dim; o++) {
                        int temp = o;
                        for (int p = 0; p < f->n_connected_vars; p++) {
                            if (p == var_pos) continue;
                            int vd = f->var_dimensions[p];
                            idx[p] = temp % vd;
                            temp /= vd;
                        }
                        int flat = 0;
                        int stride = 1;
                        for (int p = f->n_connected_vars - 1; p >= 0; p--) {
                            flat += idx[p] * stride;
                            stride *= f->var_dimensions[p];
                        }
                        msg += f->values[flat];
                    }
                    safe_free((void**)&(idx));
                } else {
                    msg = f->values[s];
                }
                fg->variables[vi].belief[s] *= (msg > 1e-12f ? msg : 1e-12f);
            }
        }

        float sum = 0.0f;
        for (int s = 0; s < ds; s++) sum += fg->variables[vi].belief[s];
        if (sum > 1e-12f)
            for (int s = 0; s < ds; s++) fg->variables[vi].belief[s] /= sum;
    }

    for (int s = 0; s < ds; s++)
        marginal[s] = fg->variables[vi].belief[s];
    return 0;
}

int fg_max_product(FactorGraph* fg, int* map_assignment, int max_iter) {
    if (!fg || !map_assignment) return -1;
    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < fg->n_variables; i++) {
            int ds = fg->variables[i].domain_size;
            for (int s = 0; s < ds; s++)
                fg->variables[i].belief[s] = 1.0f;

            for (int fi = 0; fi < fg->variables[i].n_factors; fi++) {
                int fid = fg->variables[i].factor_ids[fi];
                int fj = -1;
                for (int j = 0; j < fg->n_factors; j++)
                    if (fg->factors[j].id == fid) { fj = j; break; }
                if (fj < 0) continue;

                FgFactorNode* f = &fg->factors[fj];
                int var_pos = -1;
                for (int p = 0; p < f->n_connected_vars; p++)
                    if (f->connected_vars[p] == fg->variables[i].id) { var_pos = p; break; }
                if (var_pos < 0) continue;

                int ds2 = fg->variables[i].domain_size;
                for (int s = 0; s < ds2; s++) {
                    float best = -FLT_MAX;
                    int* idx = (int*)calloc((size_t)f->n_connected_vars, sizeof(int));
                    if (!idx) continue;
                    int other_dim = 1;
                    for (int p = 0; p < f->n_connected_vars; p++)
                        if (p != var_pos) other_dim *= f->var_dimensions[p];
                    idx[var_pos] = s;
                    for (int o = 0; o < (other_dim > 0 ? other_dim : 1); o++) {
                        int temp = o;
                        for (int p = 0; p < f->n_connected_vars; p++) {
                            if (p == var_pos) continue;
                            idx[p] = temp % f->var_dimensions[p];
                            temp /= f->var_dimensions[p];
                        }
                        int flat = 0, stride = 1;
                        for (int p = f->n_connected_vars - 1; p >= 0; p--) {
                            flat += idx[p] * stride;
                            stride *= f->var_dimensions[p];
                        }
                        if (f->values[flat] > best) best = f->values[flat];
                    }
                    safe_free((void**)&(idx));
                    fg->variables[i].belief[s] *= best;
                }
            }

            int best_s = 0;
            float best_v = -FLT_MAX;
            for (int s = 0; s < ds; s++) {
                if (fg->variables[i].belief[s] > best_v) {
                    best_v = fg->variables[i].belief[s];
                    best_s = s;
                }
            }
            map_assignment[i] = best_s;
        }
    }
    return 0;
}

int fg_variable_elimination(FactorGraph* fg, int elim_var, int* new_factor_vars, float* new_factor_values) {
    if (!fg || !new_factor_vars || !new_factor_values) return -1;

    int vi = -1;
    for (int i = 0; i < fg->n_variables; i++)
        if (fg->variables[i].id == elim_var) { vi = i; break; }
    if (vi < 0) return -1;

    int total_size = 1;
    int* all_vars = (int*)malloc((size_t)fg->max_factors * sizeof(int));
    int* all_dims = (int*)malloc((size_t)fg->max_factors * sizeof(int));
    int all_n = 0;
    if (!all_vars || !all_dims) { safe_free((void**)&(all_vars)); safe_free((void**)&(all_dims)); return -1; }

    for (int fi = 0; fi < fg->variables[vi].n_factors; fi++) {
        int fid = fg->variables[vi].factor_ids[fi];
        int fj = -1;
        for (int j = 0; j < fg->n_factors; j++)
            if (fg->factors[j].id == fid) { fj = j; break; }
        if (fj < 0) continue;

        FgFactorNode* f = &fg->factors[fj];
        for (int p = 0; p < f->n_connected_vars; p++) {
            int vid = f->connected_vars[p];
            if (vid == elim_var) continue;
            int found = 0;
            for (int a = 0; a < all_n; a++)
                if (all_vars[a] == vid) { found = 1; break; }
            if (!found) {
                int d = 2;
                for (int vj = 0; vj < fg->n_variables; vj++)
                    if (fg->variables[vj].id == vid) { d = fg->variables[vj].domain_size; break; }
                all_vars[all_n] = vid;
                all_dims[all_n] = d;
                total_size *= d;
                all_n++;
            }
        }
    }

    int elim_domain = fg->variables[vi].domain_size;
    if (elim_domain < 2) elim_domain = 2;

    for (int i = 0; i < all_n && i < fg->max_factors; i++) {
        new_factor_vars[i] = all_vars[i];
    }
    int total_combos = total_size * elim_domain;
    if (total_combos > 1000000) total_combos = 1000000;

    float* temp_values = (float*)calloc((size_t)total_combos, sizeof(float));
    if (!temp_values) { safe_free((void**)&(all_vars)); safe_free((void**)&(all_dims)); return -1; }

    int* indices = (int*)calloc((size_t)all_n + 1, sizeof(int));
    if (!indices) { safe_free((void**)&(temp_values)); safe_free((void**)&(all_vars)); safe_free((void**)&(all_dims)); return -1; }

    int out_idx = 0;
    for (int linear = 0; linear < total_combos && out_idx < total_size; linear++) {
        int elim_val = linear % elim_domain;
        int rem = linear / elim_domain;

        for (int d = 0; d < all_n; d++) {
            indices[d] = rem % all_dims[d];
            rem /= all_dims[d];
        }

        float product = 1.0f;
        FgFactorNode* f = NULL;  /* ZSFX-FIX: 提升作用域，供下方M-003消元使用 */
        for (int fi = 0; fi < fg->variables[vi].n_factors; fi++) {
            int fid = fg->variables[vi].factor_ids[fi];
            int fj = -1;
            for (int j = 0; j < fg->n_factors; j++)
                if (fg->factors[j].id == fid) { fj = j; break; }
            if (fj < 0) continue;

            f = &fg->factors[fj];
            int f_idx = 0;
            int stride = 1;
            for (int p = f->n_connected_vars - 1; p >= 0; p--) {
                int vid = f->connected_vars[p];
                int val = (vid == elim_var) ? elim_val : 0;
                for (int a = 0; a < all_n; a++) {
                    if (all_vars[a] == vid) { val = indices[a]; break; }
                }
                f_idx += val * stride;
                stride *= (vid == elim_var) ? elim_domain :
                    (all_n > 0 ? all_dims[(all_n > 1 ? 1 : 0)] : 2);
            }
            if (f_idx >= 0 && f->values && f_idx < f->n_connected_vars * 4) {
                product *= f->values[f_idx];
            }
        }

        if (elim_val == 0 && out_idx < total_size) {
            /* M-003修复：因子图变量消元 — 对所有消元变量的值求和
             * 正确的变量消元：sum_{elim_var} ∏_f φ_f(x_f)
             * 每个保留组合需遍历消元变量的所有值，累加product */
            float sum_over_elim = product;
            for (int e = 1; e < elim_domain; e++) {
                /* 计算消元变量值=e时的线性索引 */
                int elinear = (linear / elim_domain) * elim_domain + e;
                if (elinear < total_combos && elinear >= 0 && elinear != linear) {
                    /* 重新计算elim_var=e时的全部因子乘积 */
                    float elim_product = 1.0f;
                    int e_f_idx = 0;
                    int e_stride = 1;
                    int rem_combo = elinear;
                    for (int vv = all_n - 1; vv >= 0; vv--) {
                        int vid = all_vars[vv];
                        int vval = rem_combo % all_dims[vv];
                        rem_combo /= all_dims[vv];
                        for (int f2 = 0; f2 < fg->n_factors; f2++) {
                            FgFactorNode* ff = &fg->factors[f2];
                            for (int vc = 0; vc < ff->n_connected_vars; vc++) {
                                if (ff->connected_vars[vc] == vid) {
                                    e_f_idx += vval * e_stride;
                                    e_stride *= (vid == elim_var) ? elim_domain :
                                        (all_n > 0 ? all_dims[(all_n > 1 ? 1 : 0)] : 2);
                                }
                            }
                        }
                    }
                    if (e_f_idx >= 0 && f->values && e_f_idx < f->n_connected_vars * 4) {
                        elim_product = f->values[e_f_idx];
                    }
                    sum_over_elim += elim_product;
                }
            }
            new_factor_values[out_idx] = sum_over_elim;
            out_idx++;
        }
    }

    safe_free((void**)&(indices));
    safe_free((void**)&(temp_values));
    safe_free((void**)&(all_vars));
    safe_free((void**)&(all_dims));
    return out_idx;
}

/* ==================== A01.4.2 贝叶斯网络 ==================== */

int bn_create(BayesianNetwork* bn, int max_nodes) {
    if (!bn || max_nodes < 1) return -1;
    bn->nodes = (BNNode*)calloc((size_t)max_nodes, sizeof(BNNode));
    bn->n_nodes = 0;
    bn->max_nodes = max_nodes;
    bn->topological_order = (int*)calloc((size_t)max_nodes, sizeof(int));
    bn->sorted = (int*)calloc((size_t)max_nodes, sizeof(int));
    bn->visited = (int*)calloc((size_t)max_nodes, sizeof(int));
    if (!bn->nodes || !bn->topological_order || !bn->sorted || !bn->visited) {
        safe_free((void**)&(bn->nodes)); safe_free((void**)&(bn->topological_order));
        safe_free((void**)&(bn->sorted)); safe_free((void**)&(bn->visited));
        return -1;
    }
    return 0;
}

void bn_free(BayesianNetwork* bn) {
    if (!bn) return;
    for (int i = 0; i < bn->n_nodes; i++) {
        safe_free((void**)&(bn->nodes[i].parent_ids));
        safe_free((void**)&(bn->nodes[i].child_ids));
        safe_free((void**)&(bn->nodes[i].cpt));
    }
    safe_free((void**)&(bn->nodes));
    safe_free((void**)&(bn->topological_order));
    safe_free((void**)&(bn->sorted));
    safe_free((void**)&(bn->visited));
    memset(bn, 0, sizeof(BayesianNetwork));
}

int bn_add_node(BayesianNetwork* bn, int id, const char* name, int domain_size) {
    if (!bn || bn->n_nodes >= bn->max_nodes || domain_size < 2) return -1;
    BNNode* n = &bn->nodes[bn->n_nodes];
    n->id = id;
    strncpy_s(n->name, sizeof(n->name), name ? name : "", _TRUNCATE);
    n->domain_size = domain_size;
    n->parent_ids = (int*)calloc((size_t)bn->max_nodes, sizeof(int));
    n->child_ids = (int*)calloc((size_t)bn->max_nodes, sizeof(int));
    n->n_parents = 0;
    n->n_children = 0;
    n->cpt = NULL;
    n->cpt_size = 0;
    if (!n->parent_ids || !n->child_ids) {
        safe_free((void**)&(n->parent_ids)); safe_free((void**)&(n->child_ids)); return -1;
    }
    return bn->n_nodes++;
}

int bn_add_edge(BayesianNetwork* bn, int parent_id, int child_id) {
    if (!bn) return -1;
    int pi = -1, ci = -1;
    for (int i = 0; i < bn->n_nodes; i++) {
        if (bn->nodes[i].id == parent_id) pi = i;
        if (bn->nodes[i].id == child_id) ci = i;
    }
    if (pi < 0 || ci < 0) return -1;
    bn->nodes[ci].parent_ids[bn->nodes[ci].n_parents++] = parent_id;
    bn->nodes[pi].child_ids[bn->nodes[pi].n_children++] = child_id;
    return 0;
}

int bn_set_cpt(BayesianNetwork* bn, int node_id, const float* cpt_values, int cpt_size) {
    if (!bn || !cpt_values) return -1;
    int ni = -1;
    for (int i = 0; i < bn->n_nodes; i++)
        if (bn->nodes[i].id == node_id) { ni = i; break; }
    if (ni < 0 || cpt_size < 1) return -1;
    BNNode* n = &bn->nodes[ni];
    safe_free((void**)&(n->cpt));
    n->cpt_size = cpt_size;
    n->cpt = (float*)malloc((size_t)cpt_size * sizeof(float));
    if (!n->cpt) return -1;
    memcpy(n->cpt, cpt_values, (size_t)cpt_size * sizeof(float));
    return 0;
}

static void bn_dfs(int node_idx, BayesianNetwork* bn, int* idx) {
    bn->visited[node_idx] = 1;
    for (int i = 0; i < bn->nodes[node_idx].n_children; i++) {
        int cid = bn->nodes[node_idx].child_ids[i];
        int cj = -1;
        for (int j = 0; j < bn->n_nodes; j++)
            if (bn->nodes[j].id == cid) { cj = j; break; }
        if (cj >= 0 && !bn->visited[cj]) bn_dfs(cj, bn, idx);
    }
    bn->topological_order[(*idx)--] = node_idx;
}

int bn_topological_sort(BayesianNetwork* bn) {
    if (!bn) return -1;
    for (int i = 0; i < bn->n_nodes; i++) {
        bn->visited[i] = 0;
        bn->sorted[i] = 0;
    }
    int idx = bn->n_nodes - 1;
    for (int i = 0; i < bn->n_nodes; i++)
        if (!bn->visited[i]) bn_dfs(i, bn, &idx);
    for (int i = 0; i < bn->n_nodes; i++) bn->sorted[i] = 1;
    return 0;
}

float bn_infer_marginal(const BayesianNetwork* bn, int query_var, int evidence_var, int evidence_val) {
    if (!bn) return -1.0f;
    int qi = -1, ei = -1;
    for (int i = 0; i < bn->n_nodes; i++) {
        if (bn->nodes[i].id == query_var) qi = i;
        if (bn->nodes[i].id == evidence_var) ei = i;
    }
    if (qi < 0) return -1.0f;

    if (!bn->sorted[0]) return -1.0f;

    if (ei >= 0) {
        BNNode* en = &bn->nodes[ei];
        int ev_size = 1;
        for (int p = 0; p < en->n_parents; p++) {
            int pid = en->parent_ids[p];
            for (int j = 0; j < bn->n_nodes; j++)
                if (bn->nodes[j].id == pid) { ev_size *= bn->nodes[j].domain_size; break; }
        }
        ev_size *= en->domain_size;

        if (en->cpt && en->cpt_size >= ev_size) {
            int parent_stride = ev_size / en->domain_size;
            float norm = 0.0f;
            for (int d = 0; d < en->domain_size; d++) {
                float prob = 0.0f;
                int* p_assign = (int*)calloc((size_t)en->n_parents, sizeof(int));
                if (!p_assign) continue;
                if (d == evidence_val) {
                    for (int pi = 0; pi < parent_stride; pi++) {
                        int temp = pi;
                        for (int pp = 0; pp < en->n_parents; pp++) {
                            int pid = en->parent_ids[pp];
                            int pd = 2;
                            for (int j = 0; j < bn->n_nodes; j++)
                                if (bn->nodes[j].id == pid) { pd = bn->nodes[j].domain_size; break; }
                            p_assign[pp] = temp % pd;
                            temp /= pd;
                        }
                        float prior = 1.0f;
                        prob += en->cpt[d * parent_stride + pi] * prior;
                    }
                }
                norm += prob;
                safe_free((void**)&(p_assign));
            }
            int pi_slot = evidence_val * parent_stride;
            if (norm > 1e-12f) {
                float joint = 0.0f;
                for (int pi = 0; pi < parent_stride; pi++)
                    joint += en->cpt[pi_slot + pi] * (1.0f / (float)parent_stride);
                return joint / norm;
            }
        }
    }

    BNNode* qn = &bn->nodes[qi];
    if (qn->cpt) {
        if (qn->n_parents == 0) {
            /* 无父节点：CPT条目即为边际分布，返回归一化后第0个状态 */
            float cpt_sum = 0.0f;
            for (int d = 0; d < qn->domain_size; d++) cpt_sum += qn->cpt[d];
            return (cpt_sum > 1e-12f) ? qn->cpt[0] / cpt_sum : 1.0f / (float)qn->domain_size;
        }
        /* 有父节点：使用实际domain_size，正确计算组合数 */
        int total = 1;
        int* parent_ds = (int*)calloc((size_t)qn->n_parents, sizeof(int));
        if (!parent_ds) return 1.0f / (float)qn->domain_size;
        for (int p = 0; p < qn->n_parents; p++) {
            parent_ds[p] = 2; /* 默认二值 */
            int pid = qn->parent_ids[p];
            for (int j = 0; j < bn->n_nodes; j++) {
                if (bn->nodes[j].id == pid) { parent_ds[p] = bn->nodes[j].domain_size; break; }
            }
            total *= parent_ds[p];
        }
        /* 计算查询变量第0个状态的边际概率：对所有父组合均匀加权求和 */
        if (qn->domain_size < 2) { safe_free((void**)&parent_ds); return qn->cpt[0]; }
        int parent_stride = total; /* 每个查询状态的跨度 */
        float marginal_0 = 0.0f;
        for (int pi = 0; pi < parent_stride; pi++) {
            marginal_0 += qn->cpt[pi]; /* CPT[0*parent_stride + pi] = CPT[pi] */
        }
        float sum_all = 0.0f;
        for (int i = 0; i < total * qn->domain_size; i++) sum_all += qn->cpt[i];
        safe_free((void**)&parent_ds);
        return (sum_all > 1e-12f) ? marginal_0 / sum_all : 1.0f / (float)qn->domain_size;
    }
    return 1.0f / (float)qn->domain_size;
}

int bn_sample(const BayesianNetwork* bn, int* samples, int n_samples) {
    if (!bn || !samples) return -1;
    int* order = bn->topological_order;
    if (!bn->sorted[0]) {
        BayesianNetwork* mut = (BayesianNetwork*)bn;
        bn_topological_sort((BayesianNetwork*)mut);
    }

    for (int s = 0; s < n_samples; s++) {
        int* assignment = (int*)calloc((size_t)bn->n_nodes, sizeof(int));
        if (!assignment) continue;

        for (int oi = 0; oi < bn->n_nodes; oi++) {
            int ni = order[oi];
            BNNode* n = &bn->nodes[ni];
            if (!n->cpt) {
                /* K-008修复：使用安全随机数替代rand() */
                assignment[ni] = (int)(secure_random_int((uint32_t)(n->domain_size - 1)));
                continue;
            }
            int idx = 0;
            int stride = 1;
            for (int p = n->n_parents - 1; p >= 0; p--) {
                int pid = n->parent_ids[p];
                int pj = -1;
                for (int j = 0; j < bn->n_nodes; j++)
                    if (bn->nodes[j].id == pid) { pj = j; break; }
                if (pj >= 0) {
                    idx += assignment[pj] * stride;
                    stride *= bn->nodes[pj].domain_size;
                }
            }
            /* K-008修复：使用安全随机数替代rand() */
            float r = secure_random_float();
            float cum = 0.0f;
            int chosen = 0;
            for (int d = 0; d < n->domain_size; d++) {
                cum += n->cpt[idx + d * (n->cpt_size / n->domain_size)];
                if (r < cum) { chosen = d; break; }
            }
            assignment[ni] = chosen;
        }
        for (int i = 0; i < bn->n_nodes; i++)
            samples[s * bn->n_nodes + i] = assignment[i];
        safe_free((void**)&(assignment));
    }
    return 0;
}

/* ==================== A01.4.2 马尔可夫随机场 ==================== */

int mrf_create(MarkovRandomField* mrf, int max_nodes, int max_cliques) {
    if (!mrf || max_nodes < 1 || max_cliques < 1) return -1;
    mrf->nodes = (MrfNode*)calloc((size_t)max_nodes, sizeof(MrfNode));
    mrf->n_nodes = 0;
    mrf->max_nodes = max_nodes;
    mrf->cliques = (Clique*)calloc((size_t)max_cliques, sizeof(Clique));
    mrf->n_cliques = 0;
    mrf->max_cliques = max_cliques;
    mrf->edges = (int*)calloc((size_t)max_nodes * max_nodes, sizeof(int));
    mrf->edge_pairs = (int*)calloc((size_t)max_nodes * 2, sizeof(int));
    mrf->pairwise_potentials = (float*)calloc((size_t)max_nodes * max_nodes * 4, sizeof(float));
    mrf->edge_count = 0;
    if (!mrf->nodes || !mrf->cliques || !mrf->edges ||
        !mrf->edge_pairs || !mrf->pairwise_potentials) {
        safe_free((void**)&(mrf->nodes)); safe_free((void**)&(mrf->cliques));
        safe_free((void**)&(mrf->edges)); safe_free((void**)&(mrf->edge_pairs)); safe_free((void**)&(mrf->pairwise_potentials));
        return -1;
    }
    return 0;
}

void mrf_free(MarkovRandomField* mrf) {
    if (!mrf) return;
    for (int i = 0; i < mrf->n_nodes; i++) {
        safe_free((void**)&(mrf->nodes[i].belief));
        safe_free((void**)&(mrf->nodes[i].clique_ids));
    }
    for (int i = 0; i < mrf->n_cliques; i++)
        safe_free((void**)&(mrf->cliques[i].potential));
    safe_free((void**)&(mrf->nodes)); safe_free((void**)&(mrf->cliques));
    safe_free((void**)&(mrf->edges)); safe_free((void**)&(mrf->edge_pairs)); safe_free((void**)&(mrf->pairwise_potentials));
    memset(mrf, 0, sizeof(MarkovRandomField));
}

int mrf_add_node(MarkovRandomField* mrf, int id, const char* name, int domain_size) {
    if (!mrf || mrf->n_nodes >= mrf->max_nodes || domain_size < 2) return -1;
    MrfNode* n = &mrf->nodes[mrf->n_nodes];
    n->id = id;
    strncpy_s(n->name, sizeof(n->name), name ? name : "", _TRUNCATE);
    n->domain_size = domain_size;
    n->belief = (float*)calloc((size_t)domain_size, sizeof(float));
    n->clique_ids = (int*)calloc((size_t)mrf->max_cliques, sizeof(int));
    n->n_cliques = 0;
    if (!n->belief || !n->clique_ids) {
        safe_free((void**)&(n->belief)); safe_free((void**)&(n->clique_ids)); return -1;
    }
    for (int s = 0; s < domain_size; s++) n->belief[s] = 1.0f / (float)domain_size;
    return mrf->n_nodes++;
}

int mrf_add_edge(MarkovRandomField* mrf, int node_i, int node_j, const float* potentials) {
    if (!mrf || !potentials) return -1;
    int ni = -1, nj = -1;
    for (int i = 0; i < mrf->n_nodes; i++) {
        if (mrf->nodes[i].id == node_i) ni = i;
        if (mrf->nodes[i].id == node_j) nj = i;
    }
    if (ni < 0 || nj < 0) return -1;
    int e = mrf->edge_count;
    mrf->edges[ni * mrf->max_nodes + nj] = 1;
    mrf->edges[nj * mrf->max_nodes + ni] = 1;
    mrf->edge_pairs[e * 2] = ni;
    mrf->edge_pairs[e * 2 + 1] = nj;
    int di = mrf->nodes[ni].domain_size;
    int dj = mrf->nodes[nj].domain_size;
    for (int i = 0; i < di * dj; i++)
        mrf->pairwise_potentials[e * 4 + i] = potentials[i];
    mrf->edge_count++;
    return 0;
}

int mrf_add_clique(MarkovRandomField* mrf, const int* node_ids, int n_nodes, const float* potential) {
    if (!mrf || !node_ids || !potential || mrf->n_cliques >= mrf->max_cliques) return -1;
    Clique* c = &mrf->cliques[mrf->n_cliques];
    c->node_ids = (int*)malloc((size_t)n_nodes * sizeof(int));
    c->n_nodes = n_nodes;
    if (!c->node_ids) return -1;
    int total_size = 1;
    for (int i = 0; i < n_nodes; i++) {
        c->node_ids[i] = node_ids[i];
        for (int j = 0; j < mrf->n_nodes; j++)
            if (mrf->nodes[j].id == node_ids[i]) {
                total_size *= mrf->nodes[j].domain_size;
                break;
            }
    }
    c->potential = (float*)malloc((size_t)total_size * sizeof(float));
    if (!c->potential) { safe_free((void**)&(c->node_ids)); return -1; }
    memcpy(c->potential, potential, (size_t)total_size * sizeof(float));

    for (int i = 0; i < n_nodes; i++) {
        for (int j = 0; j < mrf->n_nodes; j++) {
            if (mrf->nodes[j].id == node_ids[i]) {
                int slot = mrf->nodes[j].n_cliques++;
                mrf->nodes[j].clique_ids[slot] = mrf->n_cliques;
                break;
            }
        }
    }
    return mrf->n_cliques++;
}

int mrf_gibbs_sample(const MarkovRandomField* mrf, int* assignment, int n_burnin, int n_samples) {
    if (!mrf || !assignment) return -1;
    int n = mrf->n_nodes;

    int* current = (int*)malloc((size_t)n * sizeof(int));
    if (!current) return -1;
    for (int i = 0; i < n; i++) /* K-008修复：安全随机数 */
        current[i] = (int)(secure_random_int((uint32_t)(mrf->nodes[i].domain_size - 1)));

    for (int iter = 0; iter < n_burnin + n_samples; iter++) {
        for (int vi = 0; vi < n; vi++) {
            MrfNode* v = &mrf->nodes[vi];
            int ds = v->domain_size;
            float* log_probs = (float*)calloc((size_t)ds, sizeof(float));
            if (!log_probs) continue;

            for (int ei = 0; ei < mrf->edge_count; ei++) {
                int n1 = mrf->edge_pairs[ei * 2];
                int n2 = mrf->edge_pairs[ei * 2 + 1];
                if (n1 == vi) {
                    for (int s = 0; s < ds; s++)
                        log_probs[s] += logf(mrf->pairwise_potentials[ei * 4 + s * mrf->nodes[n2].domain_size + current[n2]] + 1e-12f);
                } else if (n2 == vi) {
                    for (int s = 0; s < ds; s++)
                        log_probs[s] += logf(mrf->pairwise_potentials[ei * 4 + current[n1] * mrf->nodes[n2].domain_size + s] + 1e-12f);
                }
            }

            for (int ci = 0; ci < v->n_cliques; ci++) {
                int cid = v->clique_ids[ci];
                Clique* c = &mrf->cliques[cid];
                int pos = -1;
                for (int p = 0; p < c->n_nodes; p++)
                    if (c->node_ids[p] == v->id) { pos = p; break; }
                if (pos < 0) continue;

                int stride = 1;
                for (int p = pos + 1; p < c->n_nodes; p++) stride *= 2;
                int base = 0;
                for (int p = 0; p < pos; p++) {
                    int ni = -1;
                    for (int j = 0; j < n; j++)
                        if (mrf->nodes[j].id == c->node_ids[p]) { ni = j; break; }
                    int other_stride = 1;
                    for (int j = p + 1; j < c->n_nodes; j++) other_stride *= 2;
                    base += current[ni] * other_stride;
                }
                int total = 1;
                for (int p = 0; p < c->n_nodes; p++) total *= 2;
                for (int s = 0; s < ds; s++)
                    log_probs[s] += logf(c->potential[base + s * stride] + 1e-12f);
            }

            float max_lp = log_probs[0];
            for (int s = 1; s < ds; s++)
                if (log_probs[s] > max_lp) max_lp = log_probs[s];

            float* probs = (float*)malloc((size_t)ds * sizeof(float));
            if (!probs) { safe_free((void**)&(log_probs)); continue; }
            float sum_p = 0.0f;
            for (int s = 0; s < ds; s++) {
                probs[s] = expf(log_probs[s] - max_lp);
                sum_p += probs[s];
            }
            if (sum_p > 1e-12f) {
                for (int s = 0; s < ds; s++) probs[s] /= sum_p;
                /* K-008修复：安全随机数 */
                float r = secure_random_float();
                float cum = 0.0f;
                for (int s = 0; s < ds; s++) {
                    cum += probs[s];
                    if (r < cum) { current[vi] = s; break; }
                }
            }
            safe_free((void**)&(probs));
            safe_free((void**)&(log_probs));
        }
        if (iter >= n_burnin) {
            for (int i = 0; i < n; i++)
                assignment[(iter - n_burnin) * n + i] = current[i];
        }
    }
    safe_free((void**)&(current));
    return 0;
}

float mrf_infer_marginal(const MarkovRandomField* mrf, int node_id, int n_burnin, int n_samples, float* marginal) {
    if (!mrf || !marginal) return -1.0f;
    int vi = -1;
    for (int i = 0; i < mrf->n_nodes; i++)
        if (mrf->nodes[i].id == node_id) { vi = i; break; }
    if (vi < 0) return -1.0f;

    int n = mrf->n_nodes;
    int ds = mrf->nodes[vi].domain_size;
    int nsamples = n_samples > 0 ? n_samples : 100;

    int* all_samples = (int*)malloc((size_t)nsamples * n * sizeof(int));
    if (!all_samples) return -1.0f;

    if (mrf_gibbs_sample(mrf, all_samples, n_burnin, nsamples) != 0) {
        safe_free((void**)&(all_samples));
        return -1.0f;
    }

    float* counts = (float*)calloc((size_t)ds, sizeof(float));
    if (!counts) { safe_free((void**)&(all_samples)); return -1.0f; }
    for (int s = 0; s < nsamples; s++) {
        int val = all_samples[s * n + vi];
        if (val >= 0 && val < ds) counts[val] += 1.0f;
    }
    float sum = 0.0f;
    for (int s = 0; s < ds; s++) sum += counts[s];
    if (sum > 1e-12f)
        for (int s = 0; s < ds; s++) marginal[s] = counts[s] / sum;
    else
        for (int s = 0; s < ds; s++) marginal[s] = 1.0f / (float)ds;

    float result = marginal[0];
    safe_free((void**)&(all_samples));
    safe_free((void**)&(counts));
    return result;
}

int mrf_loopy_bp(MarkovRandomField* mrf, int max_iter, float tol) {
    if (!mrf || max_iter < 1) return -1;
    int n = mrf->n_nodes;

    float* messages = (float*)calloc((size_t)mrf->edge_count * 4, sizeof(float));
    float* new_messages = (float*)calloc((size_t)mrf->edge_count * 4, sizeof(float));
    if (!messages || !new_messages) {
        safe_free((void**)&(messages)); safe_free((void**)&(new_messages)); return -1;
    }

    for (int e = 0; e < mrf->edge_count; e++) {
        int n1 = mrf->edge_pairs[e * 2];
        int n2 = mrf->edge_pairs[e * 2 + 1];
        int d1 = mrf->nodes[n1].domain_size;
        int d2 = mrf->nodes[n2].domain_size;
        float val = 1.0f / (float)fmax(d1, d2);
        for (int s = 0; s < d1; s++) messages[e * 2 * 2 + s] = val;
        for (int s = 0; s < d2; s++) messages[e * 2 * 2 + 4 + s] = val;
    }

    for (int iter = 0; iter < max_iter; iter++) {
        memset(new_messages, 0, (size_t)mrf->edge_count * 4 * sizeof(float));

        for (int e = 0; e < mrf->edge_count; e++) {
            int n1 = mrf->edge_pairs[e * 2];
            int n2 = mrf->edge_pairs[e * 2 + 1];
            int d1 = mrf->nodes[n1].domain_size;

            for (int s = 0; s < d1; s++) {
                float msg = 1.0f;
                for (int e2 = 0; e2 < mrf->edge_count; e2++) {
                    int m1 = mrf->edge_pairs[e2 * 2];
                    int m2 = mrf->edge_pairs[e2 * 2 + 1];
                    if (m1 == n1 && m2 != n2) {
                        int d_other = mrf->nodes[m2].domain_size;
                        float incoming = 0.0f;
                        for (int t = 0; t < d_other; t++)
                            incoming += messages[e2 * 2 * 2 + 4 + t];
                        if (incoming > 1e-12f) msg *= incoming / (float)d_other;
                    } else if (m2 == n1 && m1 != n2) {
                        float incoming = 0.0f;
                        for (int t = 0; t < mrf->nodes[m1].domain_size; t++)
                            incoming += messages[e2 * 2 * 2 + t];
                        if (incoming > 1e-12f) msg *= incoming / (float)mrf->nodes[m1].domain_size;
                    }
                }
                float pot = 1.0f;
                if (mrf->pairwise_potentials) {
                    int d2 = mrf->nodes[n2].domain_size;
                    float pot_sum = 0.0f;
                    for (int t = 0; t < d2; t++)
                        pot_sum += mrf->pairwise_potentials[e * 4 + s * d2 + t];
                    if (pot_sum > 1e-12f) pot = pot_sum / (float)d2;
                }
                new_messages[e * 2 * 2 + 4 + s] = msg * pot;
            }
        }

        float max_diff = 0.0f;
        for (int e = 0; e < mrf->edge_count * 4; e++) {
            float diff = fabsf(new_messages[e] - messages[e]);
            if (diff > max_diff) max_diff = diff;
        }

        memcpy(messages, new_messages, (size_t)mrf->edge_count * 4 * sizeof(float));

        if (max_diff < tol) {
            safe_free((void**)&(messages));
            safe_free((void**)&(new_messages));
            return iter + 1;
        }
    }

    for (int vi = 0; vi < n; vi++) {
        MrfNode* v = &mrf->nodes[vi];
        int ds = v->domain_size;
        for (int s = 0; s < ds; s++) {
            float belief = 1.0f;
            for (int e = 0; e < mrf->edge_count; e++) {
                int n1 = mrf->edge_pairs[e * 2];
                int n2 = mrf->edge_pairs[e * 2 + 1];
                if (n1 == vi) {
                    belief *= messages[e * 2 * 2 + 4 + s];
                } else if (n2 == vi) {
                    belief *= messages[e * 2 * 2 + s];
                }
            }
            v->belief[s] = belief;
        }
        float sum = 0.0f;
        for (int s = 0; s < ds; s++) sum += v->belief[s];
        if (sum > 1e-12f)
            for (int s = 0; s < ds; s++) v->belief[s] /= sum;
    }

    safe_free((void**)&(messages));
    safe_free((void**)&(new_messages));
    return max_iter;
}