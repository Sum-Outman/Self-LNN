#include "selflnn/core/cma_es.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#ifndef _WIN32
#define _POSIX_C_SOURCE 199309L
#endif
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <unistd.h>
#include <sys/time.h>
#endif

#define CMAES_EIGEN_MAX_ITER 50
#define CMAES_EIGEN_TOL 1e-10f

static unsigned int cmaes_rng_next(unsigned int* state) {
    *state = *state * 1103515245u + 12345u;
    return *state;
}

static float cmaes_rng_uniform(unsigned int* state) {
    return (float)(cmaes_rng_next(state) & 0x7FFFFFFFu) / 2147483648.0f;
}

static float cmaes_rng_gaussian(unsigned int* state) {
    float u1 = cmaes_rng_uniform(state);
    float u2 = cmaes_rng_uniform(state);
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

static void cmaes_jacobi_eigen(float* C, size_t n, float* diag, float* Q) {
    memset(Q, 0, n * n * sizeof(float));
    for (size_t i = 0; i < n; i++) Q[i * n + i] = 1.0f;
    memcpy(diag, C, n * sizeof(float));

    float* offdiag = (float*)safe_malloc(n * sizeof(float));
    if (!offdiag) return;
    for (size_t i = 1; i < n; i++) offdiag[i] = C[i * n + (i - 1)];

    for (int iter = 0; iter < CMAES_EIGEN_MAX_ITER * n * n; iter++) {
        size_t p = n - 1;
        size_t q = n - 1;
        float max_off = 0.0f;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = i + 1; j < n; j++) {
                float val = fabsf(C[i * n + j]);
                if (val > max_off) {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_off < CMAES_EIGEN_TOL) break;

        float app = diag[p];
        float aqq = diag[q];
        float apq = C[p * n + q];

        float theta = 0.5f * (aqq - app) / apq;
        float t = 1.0f / (fabsf(theta) + sqrtf(1.0f + theta * theta));
        if (theta < 0) t = -t;
        float c = 1.0f / sqrtf(1.0f + t * t);
        float s = t * c;

        diag[p] = app - t * apq;
        diag[q] = aqq + t * apq;
        C[p * n + q] = 0.0f;
        C[q * n + p] = 0.0f;

        for (size_t i = 0; i < n; i++) {
            if (i != p && i != q) {
                float aip = C[i * n + p];
                float aiq = C[i * n + q];
                C[i * n + p] = c * aip - s * aiq;
                C[p * n + i] = C[i * n + p];
                C[i * n + q] = s * aip + c * aiq;
                C[q * n + i] = C[i * n + q];
            }
            float qip = Q[i * n + p];
            float qiq = Q[i * n + q];
            Q[i * n + p] = c * qip - s * qiq;
            Q[i * n + q] = s * qip + c * qiq;
        }
    }
    safe_free((void**)&offdiag);

    for (size_t i = 0; i < n; i++) {
        if (diag[i] < 0) diag[i] = 0;
    }
}

/* 矩阵平方根/逆平方根公式修正
 * 原理: 矩阵C的特征分解 C = Q * D * Q^T
 * sqrt(C) = Q * sqrt(D) * Q^T
 * inv_sqrt(C) = Q * inv_sqrt(D) * Q^T
 * 原代码错误: sqrt_d_i被用于所有k而非sqrt_d_k，且Q^T用Q[k][i]替代Q[i][k] */
static void cmaes_sym_matrix_sqrt(const float* C, size_t n, float* sqrt_C, float* inv_sqrt_C) {
    float* work = (float*)safe_malloc(n * n * sizeof(float));
    float* diag = (float*)safe_malloc(n * sizeof(float));
    float* Q = (float*)safe_malloc(n * n * sizeof(float));
    float* temp_inv = (float*)safe_malloc(n * n * sizeof(float));
    if (!work || !diag || !Q || !temp_inv) {
        safe_free((void**)&work); safe_free((void**)&diag); safe_free((void**)&Q); safe_free((void**)&temp_inv);
        return;
    }
    memcpy(work, C, n * n * sizeof(float));
    cmaes_jacobi_eigen(work, n, diag, Q);

    /* 步骤1: 计算 Q * sqrt(D) 存入 work, Q * inv_sqrt(D) 存入 temp_inv */
    for (size_t i = 0; i < n; i++) {
        float sd = sqrtf(diag[i] > 0 ? diag[i] : 0);
        float isd = (diag[i] > 1e-15f) ? 1.0f / sqrtf(diag[i]) : 0;
        for (size_t j = 0; j < n; j++) {
            work[j * n + i] = Q[j * n + i] * sd;
            temp_inv[j * n + i] = Q[j * n + i] * isd;
        }
    }

    /* 步骤2: sqrt_C = (Q * sqrt(D)) * Q^T, inv_sqrt_C = (Q * inv_sqrt(D)) * Q^T */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum_sqrt = 0, sum_inv = 0;
            for (size_t k = 0; k < n; k++) {
                sum_sqrt += work[i * n + k] * Q[j * n + k];
                sum_inv += temp_inv[i * n + k] * Q[j * n + k];
            }
            sqrt_C[i * n + j] = sum_sqrt;
            inv_sqrt_C[i * n + j] = sum_inv;
        }
    }

    safe_free((void**)&work); safe_free((void**)&diag); safe_free((void**)&Q); safe_free((void**)&temp_inv);
}

CMAESState* cmaes_alloc(size_t dimension, float sigma, int lambda, int seed) {
    /* P1-012修复: 补充sigma和lambda参数的合法性校验
     * 原代码仅校验dimension，sigma<=0和lambda范围非法都会被静默接受 */
    if (dimension == 0 || dimension > CMAES_MAX_DIM) return NULL;
    if (sigma <= 0.0f || sigma > 1000.0f) return NULL;  /* sigma必须为正且不超过合理上限 */
    /* P1-012修复: 检测sigma为NaN或Inf */
    if (!(sigma == sigma) || sigma != sigma) return NULL; /* isnan(sigma) */
    if (lambda < CMAES_MIN_POP || lambda > CMAES_MAX_POP) return NULL; /* 种群大小范围 */
    CMAESState* state = (CMAESState*)safe_calloc(1, sizeof(CMAESState));
    if (!state) return NULL;
    if (cmaes_init(state, dimension, sigma, lambda, seed) != 0) {
        safe_free((void**)&state);
        return NULL;
    }
    return state;
}

void cmaes_free(CMAESState* state) {
    if (!state) return;
    safe_free((void**)&state->mean);
    safe_free((void**)&state->covariance);
    safe_free((void**)&state->evolution_path_sigma);
    safe_free((void**)&state->evolution_path_cov);
    safe_free((void**)&state->temp_vec);
    safe_free((void**)&state->temp_vec2);
    safe_free((void**)&state->weights);
    safe_free((void**)&state->sample_pop);
    safe_free((void**)&state->fitness);
    safe_free((void**)&state->index_order);
    safe_free((void**)&state->eigen_values);
    safe_free((void**)&state->eigen_vectors);
    safe_free((void**)&state->inv_sqrt_cov);
    safe_free((void**)&state->best_solution);
    safe_free((void**)&state->lower_bounds);
    safe_free((void**)&state->upper_bounds);
    memset(state, 0, sizeof(CMAESState));
    safe_free((void**)&state);
}

int cmaes_init(CMAESState* state, size_t dimension, float sigma, int lambda, int seed) {
    if (!state || dimension == 0 || dimension > CMAES_MAX_DIM) return -1;

    memset(state, 0, sizeof(CMAESState));
    state->dimension = dimension;
    state->sigma = sigma > 0 ? sigma : CMAES_DEFAULT_SIGMA;
    if (seed != 0) {
        state->rng_state = (unsigned int)seed;
    } else {
        /* 混合高精度计时器和进程ID作为随机种子 */
        unsigned int time_seed, pid_seed;
#ifdef _WIN32
        LARGE_INTEGER counter;
        QueryPerformanceCounter(&counter);
        time_seed = (unsigned int)(counter.QuadPart & 0xFFFFFFFFu);
        pid_seed = (unsigned int)_getpid();
#else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        time_seed = (unsigned int)((ts.tv_sec * 1000000000u + (unsigned int)ts.tv_nsec) & 0xFFFFFFFFu);
        pid_seed = (unsigned int)getpid();
#endif
        state->rng_state = time_seed ^ (pid_seed << 16) ^ (pid_seed >> 16) ^ 0x9E3779B9u;
        if (state->rng_state == 0) state->rng_state = 0xDEADBEEFu;
    }

    state->lambda = lambda < CMAES_MIN_POP ? CMAES_DEFAULT_LAMBDA : lambda;
    if (state->lambda > CMAES_MAX_POP) state->lambda = CMAES_MAX_POP;
    state->mu = state->lambda / 2;
    if (state->mu < 1) state->mu = 1;

    state->max_generations = CMAES_DEFAULT_MAX_GEN;
    state->stop_fitness = CMAES_DEFAULT_STOP_FITNESS;
    state->tol_x = CMAES_DEFAULT_TOL_X;
    state->tol_cov = CMAES_DEFAULT_TOL_COV;
    state->tol_fun = CMAES_DEFAULT_TOL_FUN;
    state->tol_hist_fun = CMAES_DEFAULT_TOL_HIST_FUN;
    state->best_fitness = FLT_MAX;
    state->best_fitness_history = FLT_MAX;
    state->termination_reason = CMAES_TERM_NONE;
    state->use_boundary = 0;

    state->mean = (float*)safe_calloc(dimension, sizeof(float));
    state->covariance = (float*)safe_calloc(dimension * dimension, sizeof(float));
    state->evolution_path_sigma = (float*)safe_calloc(dimension, sizeof(float));
    state->evolution_path_cov = (float*)safe_calloc(dimension, sizeof(float));
    state->temp_vec = (float*)safe_calloc(dimension, sizeof(float));
    state->temp_vec2 = (float*)safe_calloc(dimension, sizeof(float));
    state->weights = (float*)safe_calloc(state->mu, sizeof(float));
    state->sample_pop = (float*)safe_calloc((size_t)state->lambda * dimension, sizeof(float));
    state->fitness = (float*)safe_calloc((size_t)state->lambda, sizeof(float));
    state->index_order = (int*)safe_calloc((size_t)state->lambda, sizeof(int));
    state->eigen_values = (float*)safe_calloc(dimension, sizeof(float));
    state->eigen_vectors = (float*)safe_calloc(dimension * dimension, sizeof(float));
    state->inv_sqrt_cov = (float*)safe_calloc(dimension * dimension, sizeof(float));
    state->best_solution = (float*)safe_calloc(dimension, sizeof(float));

    if (!state->mean || !state->covariance || !state->evolution_path_sigma ||
        !state->evolution_path_cov || !state->temp_vec || !state->temp_vec2 ||
        !state->weights || !state->sample_pop || !state->fitness ||
        !state->index_order || !state->eigen_values || !state->eigen_vectors ||
        !state->inv_sqrt_cov || !state->best_solution) {
        cmaes_free(state);
        return -1;
    }

    for (size_t i = 0; i < dimension; i++) {
        state->covariance[i * dimension + i] = 1.0f;
    }
    for (size_t i = 0; i < dimension * dimension; i++) {
        state->eigen_vectors[i] = 0;
        state->inv_sqrt_cov[i] = 0;
    }
    for (size_t i = 0; i < dimension; i++) {
        state->eigen_vectors[i * dimension + i] = 1.0f;
        state->inv_sqrt_cov[i * dimension + i] = 1.0f;
    }
    for (size_t i = 0; i < dimension; i++) {
        state->eigen_values[i] = 1.0f;
    }

    float sum_weights = 0;
    float sum_weights_sq = 0;
    for (int i = 0; i < state->mu; i++) {
        float w = logf((float)(state->mu + 1)) - logf((float)(i + 1));
        state->weights[i] = w;
        sum_weights += w;
        sum_weights_sq += w * w;
    }
    for (int i = 0; i < state->mu; i++) {
        state->weights[i] /= sum_weights;
        sum_weights_sq /= (sum_weights * sum_weights);
    }
    state->mueff = 1.0f / sum_weights_sq;

    float n = (float)dimension;
    state->cc = (4.0f + state->mueff / n) / (n + 4.0f + 2.0f * state->mueff / n);
    state->cs = (state->mueff + 2.0f) / (n + state->mueff + 5.0f);
    state->c1 = 2.0f / ((n + 1.3f) * (n + 1.3f) + state->mueff);
    state->cmu = fminf(1.0f - state->c1,
                        2.0f * (state->mueff - 2.0f + 1.0f / state->mueff) /
                        ((n + 2.0f) * (n + 2.0f) + state->mueff));
    state->ds = 1.0f + 2.0f * fmaxf(0.0f, sqrtf((state->mueff - 1.0f) / (n + 1.0f)) - 1.0f) + state->cs;
    state->chi_n = sqrtf(n) * (1.0f - 1.0f / (4.0f * n) + 1.0f / (21.0f * n * n));

    return 0;
}

void cmaes_set_bounds(CMAESState* state, const float* lower, const float* upper) {
    if (!state || !lower || !upper) return;
    if (!state->lower_bounds) {
        state->lower_bounds = (float*)safe_calloc(state->dimension, sizeof(float));
    }
    if (!state->upper_bounds) {
        state->upper_bounds = (float*)safe_calloc(state->dimension, sizeof(float));
    }
    if (state->lower_bounds && state->upper_bounds) {
        memcpy(state->lower_bounds, lower, state->dimension * sizeof(float));
        memcpy(state->upper_bounds, upper, state->dimension * sizeof(float));
        state->use_boundary = 1;
    }
}

void cmaes_set_stop_conditions(CMAESState* state, float stop_fitness, int max_generations,
                                float tol_x, float tol_cov, float tol_fun, float tol_hist_fun) {
    if (!state) return;
    /* P1-013修复: 将">0"改为">=0"允许用户设置零值的停止条件
     * 使用-1.0f作为哨兵值表示"不修改此条件"，NaN检测保护 */
    if (!(stop_fitness != stop_fitness) && stop_fitness >= 0.0f) state->stop_fitness = stop_fitness;
    if (max_generations >= 0) state->max_generations = max_generations;
    if (!(tol_x != tol_x) && tol_x >= 0.0f) state->tol_x = tol_x;
    if (!(tol_cov != tol_cov) && tol_cov >= 0.0f) state->tol_cov = tol_cov;
    if (!(tol_fun != tol_fun) && tol_fun >= 0.0f) state->tol_fun = tol_fun;
    if (!(tol_hist_fun != tol_hist_fun) && tol_hist_fun >= 0.0f) state->tol_hist_fun = tol_hist_fun;
}

void cmaes_sample_population(CMAESState* state) {
    if (!state) return;

    size_t dim = state->dimension;
    size_t n = dim;
    float* eigenvec = state->eigen_vectors;
    float* eigenval = state->eigen_values;
    (void)eigenval;

    cmaes_sym_matrix_sqrt(state->covariance, n, eigenvec, state->inv_sqrt_cov);

    for (int k = 0; k < state->lambda; k++) {
        float* x = state->sample_pop + (size_t)k * dim;
        for (size_t i = 0; i < dim; i++) {
            state->temp_vec[i] = cmaes_rng_gaussian(&state->rng_state);
        }
        for (size_t i = 0; i < dim; i++) {
            x[i] = state->mean[i];
            for (size_t j = 0; j < dim; j++) {
                x[i] += state->sigma * eigenvec[i * dim + j] * state->temp_vec[j];
            }
        }
        if (state->use_boundary) {
            for (size_t i = 0; i < dim; i++) {
                if (x[i] < state->lower_bounds[i]) x[i] = state->lower_bounds[i];
                if (x[i] > state->upper_bounds[i]) x[i] = state->upper_bounds[i];
            }
        }
    }
}

int cmaes_update(CMAESState* state, const float* fitness_values) {
    if (!state || !fitness_values) return -1;

    size_t dim = state->dimension;
    memcpy(state->fitness, fitness_values, (size_t)state->lambda * sizeof(float));

    for (int i = 0; i < state->lambda; i++) {
        state->index_order[i] = i;
    }
    for (int i = 0; i < state->lambda - 1; i++) {
        for (int j = 0; j < state->lambda - i - 1; j++) {
            if (state->fitness[state->index_order[j]] > state->fitness[state->index_order[j + 1]]) {
                int tmp = state->index_order[j];
                state->index_order[j] = state->index_order[j + 1];
                state->index_order[j + 1] = tmp;
            }
        }
    }

    if (state->fitness[state->index_order[0]] < state->best_fitness) {
        state->best_fitness = state->fitness[state->index_order[0]];
        memcpy(state->best_solution,
               state->sample_pop + (size_t)state->index_order[0] * dim,
               dim * sizeof(float));
    }

    memset(state->temp_vec, 0, dim * sizeof(float));
    for (int i = 0; i < state->mu; i++) {
        float* x = state->sample_pop + (size_t)state->index_order[i] * dim;
        for (size_t j = 0; j < dim; j++) {
            state->temp_vec[j] += state->weights[i] * x[j];
        }
    }
    for (size_t i = 0; i < dim; i++) {
        state->temp_vec2[i] = state->temp_vec[i] - state->mean[i];
    }
    memcpy(state->mean, state->temp_vec, dim * sizeof(float));

    float* ps = state->evolution_path_sigma;
    for (size_t i = 0; i < dim; i++) {
        float sum = 0;
        for (size_t j = 0; j < dim; j++) {
            sum += state->inv_sqrt_cov[i * dim + j] * state->temp_vec2[j];
        }
        state->temp_vec[i] = sum;
    }
    float factor_ps = sqrtf(state->cs * (2.0f - state->cs) * state->mueff) / state->sigma;
    for (size_t i = 0; i < dim; i++) {
        ps[i] = (1.0f - state->cs) * ps[i] + factor_ps * state->temp_vec[i];
    }

    float* pc = state->evolution_path_cov;
    float factor_pc = sqrtf(state->cc * (2.0f - state->cc) * state->mueff) / state->sigma;
    for (size_t i = 0; i < dim; i++) {
        pc[i] = (1.0f - state->cc) * pc[i] + factor_pc * state->temp_vec2[i];
    }

    float* C = state->covariance;
    float c1 = state->c1;
    float cmu = state->cmu;
    float alpha_cov = 1.0f;
    float hs = 0;
    float ps_norm = 0;
    for (size_t i = 0; i < dim; i++) ps_norm += ps[i] * ps[i];
    ps_norm = sqrtf(ps_norm);
    float expected_ps = state->chi_n;
    {
        float gen_scale = powf(1.0f - state->cs, 2.0f * (float)(state->generation + 1));
        float denom = sqrtf(fmaxf(1.0f - gen_scale, 1e-20f));
        if (ps_norm / denom < 1.5f * expected_ps) {
            hs = 1.0f;
        }
    }

    if (hs > 0) {
        alpha_cov = 1.0f - c1 - cmu;
/* 当演化路径长度hs>0时，delta_h不应被丢弃，
         * 需要在下一次合并到协方差更新中以补偿路径衰减 */
        float delta_h = (1.0f - hs) * c1 * (2.0f - c1);
        alpha_cov += delta_h; /* 将路径衰减补偿纳入协方差收缩因子 */
    } else {
        alpha_cov = 1.0f - c1 - cmu + c1 * hs * (2.0f - c1);
    }

    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            C[i * dim + j] = alpha_cov * C[i * dim + j];
        }
    }

    float factor_rank1 = c1 * hs;
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            C[i * dim + j] += factor_rank1 * pc[i] * pc[j];
        }
    }

    /* 数值稳定性：条件数检查与正则化 */
    float max_diag = 0.0f, min_diag = 1e30f;
    for (size_t i = 0; i < dim; i++) {
        float d = C[i * dim + i];
        if (d > max_diag) max_diag = d;
        if (d < min_diag) min_diag = d;
    }
    if (min_diag < 1e-20f || max_diag > 1e20f || (min_diag > 0 && max_diag / min_diag > 1e12f)) {
        float reg = 0.001f * (max_diag + 1e-6f);
        for (size_t i = 0; i < dim; i++) {
            C[i * dim + i] += reg;
        }
    }
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            if (isnan(C[i * dim + j]) || isinf(C[i * dim + j])) {
                C[i * dim + j] = (i == j) ? 0.01f : 0.0f;
            }
        }
    }

    float denom_c1cmu = fmaxf(1.0f - c1 - cmu, 1e-10f);
    float factor_rankmu = cmu * (1.0f - hs * c1 * (2.0f - c1) / denom_c1cmu);
    for (int k = 0; k < state->mu; k++) {
        float* xk = state->sample_pop + (size_t)state->index_order[k] * dim;
        for (size_t i = 0; i < dim; i++) {
            state->temp_vec[i] = (xk[i] - (state->mean[i] - state->temp_vec2[i])) / state->sigma;
        }
        for (size_t i = 0; i < dim; i++) {
            for (size_t j = 0; j < dim; j++) {
                C[i * dim + j] += factor_rankmu * state->weights[k] * state->temp_vec[i] * state->temp_vec[j];
            }
        }
    }

    /* Active CMA：使用最差个体主动减少不良方向的方差 */
    if (state->enable_active && state->lambda > state->mu + 2) {
        float active_scale = 0.5f * cmu;
        int n_active = state->mu;
        float sum_active_neg = 0.0f;
        for (int k = 0; k < n_active; k++) {
            int worst_idx = state->index_order[state->lambda - 1 - k];
            float* xk = state->sample_pop + (size_t)worst_idx * dim;
            for (size_t i = 0; i < dim; i++) {
                state->temp_vec[i] = (xk[i] - (state->mean[i] - state->temp_vec2[i])) / state->sigma;
            }
            float w_neg = -state->weights[k] * active_scale;
            sum_active_neg += w_neg;
            for (size_t i = 0; i < dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    C[i * dim + j] += w_neg * state->temp_vec[i] * state->temp_vec[j];
                }
            }
        }
    }

    for (size_t i = 0; i < dim; i++) {
        C[i * dim + i] = fmaxf(C[i * dim + i], 1e-20f);
    }

    float ps_length = ps_norm;
    float sigma_factor = expf((state->cs / state->ds) * (ps_length / state->chi_n - 1.0f));
    state->sigma *= sigma_factor;
    if (state->sigma < 1e-20f) state->sigma = 1e-20f;

    state->generation++;

    return 0;
}

float* cmaes_get_solution(const CMAESState* state, int index) {
    if (!state || index < 0 || index >= state->lambda) return NULL;
    return state->sample_pop + (size_t)index * state->dimension;
}

int cmaes_get_population_size(const CMAESState* state) {
    return state ? state->lambda : 0;
}

int cmaes_get_generation(const CMAESState* state) {
    return state ? state->generation : 0;
}

void cmaes_get_best_solution(const CMAESState* state, float* out_x, float* out_fitness) {
    if (!state) return;
    if (out_x) memcpy(out_x, state->best_solution, state->dimension * sizeof(float));
    if (out_fitness) *out_fitness = state->best_fitness;
}

float cmaes_get_current_sigma(const CMAESState* state) {
    return state ? state->sigma : 0;
}

int cmaes_test_stop_conditions(CMAESState* state) {
    if (!state) return 1;

    if (state->generation >= state->max_generations) {
        state->termination_reason = CMAES_TERM_MAXGEN;
        return 1;
    }

    if (state->best_fitness <= state->stop_fitness) {
        state->termination_reason = CMAES_TERM_TOLFUN;
        return 1;
    }

/* 原条件 sigma*tol_x < tol_x 恒为真
     * 正确条件: 步长×最大标准差 < tol_x，即检查搜索步长是否已收敛 */
    {
        size_t dim = state->dimension;
        float max_diag = state->covariance[0];
        for (size_t di = 0; di < dim; di++) {
            float val = state->covariance[di * dim + di];
            if (val > max_diag) max_diag = val;
        }
        if (state->sigma * sqrtf(max_diag > 0.0f ? max_diag : 1e-30f) < state->tol_x) {
            state->termination_reason = CMAES_TERM_TOLX;
            return 1;
        }
    }

    if (state->generation > 2 &&
        fabsf(state->best_fitness - state->best_fitness_history) < state->tol_hist_fun &&
        state->best_fitness > state->stop_fitness) {
        state->termination_reason = CMAES_TERM_TOLHISTFUN;
        return 1;
    }
    state->best_fitness_history = state->best_fitness;

    size_t dim = state->dimension;
    float max_cov = state->covariance[0];
    float min_cov = state->covariance[0];
    for (size_t i = 0; i < dim; i++) {
        float val = state->covariance[i * dim + i];
        if (val > max_cov) max_cov = val;
        if (val < min_cov) min_cov = val;
    }
    if (min_cov > 0 && max_cov / min_cov > 1e14f) {
        state->termination_reason = CMAES_TERM_CONDITION;
        return 1;
    }

    return 0;
}

void cmaes_reset(CMAESState* state) {
    if (!state) return;
    size_t dim = state->dimension;
    memset(state->mean, 0, dim * sizeof(float));
    memset(state->covariance, 0, dim * dim * sizeof(float));
    for (size_t i = 0; i < dim; i++) state->covariance[i * dim + i] = 1.0f;
    memset(state->evolution_path_sigma, 0, dim * sizeof(float));
    memset(state->evolution_path_cov, 0, dim * sizeof(float));
    memset(state->eigen_values, 0, dim * sizeof(float));
    for (size_t i = 0; i < dim; i++) state->eigen_values[i] = 1.0f;
    memset(state->eigen_vectors, 0, dim * dim * sizeof(float));
    for (size_t i = 0; i < dim; i++) state->eigen_vectors[i * dim + i] = 1.0f;
    memset(state->inv_sqrt_cov, 0, dim * dim * sizeof(float));
    for (size_t i = 0; i < dim; i++) state->inv_sqrt_cov[i * dim + i] = 1.0f;
    state->sigma = CMAES_DEFAULT_SIGMA;
    state->generation = 0;
    state->best_fitness = FLT_MAX;
    state->best_fitness_history = FLT_MAX;
    state->termination_reason = CMAES_TERM_NONE;
}

int cmaes_optimize(CMAESState* state, CMAESFitnessFunction func, void* user_data) {
    if (!state || !func) return -1;

    while (cmaes_test_stop_conditions(state) == 0) {
        cmaes_sample_population(state);

        for (int i = 0; i < state->lambda; i++) {
            float* x = state->sample_pop + (size_t)i * state->dimension;
            state->fitness[i] = func(x, state->dimension, user_data);
        }

        if (cmaes_update(state, state->fitness) != 0) return -1;
    }

    return 0;
}

void cmaes_enable_active_cma(CMAESState* state, int enable) {
    if (state) state->enable_active = enable;
}

/**
 * @brief IPOP重启（增量种群优化）
 * 
 * S-002修复: 完整的IPOP(Increasing Population)策略实现。
 * 
 * IPOP策略原理:
 * - CMA-ES收敛到局部最优时，将种群大小翻倍并重新启动
 * - 每次重启: λ_new = λ_base * (IPOP_MULTIPLIER)^restart
 * - 更大的种群提供更全局的搜索能力，有助于跳出局部最优
 * 
 * 修复要点:
 * - 保存dimension在cmaes_free之前（避免zero-out丢失维度信息）
 * - 保存旧的最优均值作为重启起点
 * - 递增种群倍数保证搜索空间逐步扩大
 */
int cmaes_ipop_optimize(CMAESState* state, CMAESFitnessFunction func, void* user_data,
                         int max_restarts) {
    if (!state || !func) return -1;
    if (max_restarts <= 0) max_restarts = CMAES_IPOP_MAX_RESTARTS;
    if (max_restarts > CMAES_IPOP_MAX_RESTARTS) max_restarts = CMAES_IPOP_MAX_RESTARTS;

    float global_best_fitness = FLT_MAX;
    float* global_best_solution = (float*)safe_malloc(state->dimension * sizeof(float));
    if (!global_best_solution) return -1;

    int old_lambda = state->lambda;
    float old_sigma = state->sigma;
    /* S-002: 保存dimension，防止cmaes_free清零 */
    size_t saved_dim = state->dimension;

    for (int restart = 0; restart <= max_restarts; restart++) {
        state->ipop_restart_count = restart;

        if (restart > 0) {
            /* IPOP策略: 每次重启种群大小翻倍 */
            int new_lambda = (int)((float)old_lambda * powf(CMAES_IPOP_POP_MULTIPLIER, (float)restart));
            if (new_lambda > CMAES_MAX_POP) new_lambda = CMAES_MAX_POP;
            if (new_lambda <= state->lambda) new_lambda = state->lambda + CMAES_MIN_POP;

            /* 保存当前最优均值作为重启种子 */
            float* old_mean = (float*)safe_malloc(saved_dim * sizeof(float));
            if (old_mean) {
                memcpy(old_mean, state->mean, saved_dim * sizeof(float));
            }

            /* S-002: 保存sigma，因为在cmaes_free中会清零 */
            float saved_sigma_restart = state->sigma;
            int seed = (int)(state->rng_state + (unsigned int)restart * 12345u);

            /* 释放旧状态（会清零整个结构体） */
            cmaes_free(state);

            /* S-002: 使用保存的dimension初始化新状态 */
            if (cmaes_init(state, saved_dim, saved_sigma_restart, new_lambda, seed) != 0) {
                safe_free((void**)&global_best_solution);
                if (old_mean) safe_free((void**)&old_mean);
                return -1;
            }

            state->enable_active = 1;
            state->ipop_restart_count = restart;

            /* 从旧最优均值继续搜索 */
            if (old_mean) {
                memcpy(state->mean, old_mean, saved_dim * sizeof(float));
                safe_free((void**)&old_mean);
            }
        }

        /* R6-007修复: 使用cmaes_optimize返回值，而非(void)丢弃 */
        int cma_result = cmaes_optimize(state, func, user_data);
        if (cma_result != 0) {
            /* CMA-ES优化步失败，记录但继续（可能因停滞/收敛） */
            if (state->termination_reason == CMAES_TERM_TOLFUN) {
                break; /* 收敛终止 */
            }
        }

        if (state->best_fitness < global_best_fitness) {
            global_best_fitness = state->best_fitness;
            memcpy(global_best_solution, state->best_solution, saved_dim * sizeof(float));
        }

        /* 终止条件检查 */
        if (state->generation >= state->max_generations) break;

        if (state->termination_reason == CMAES_TERM_TOLFUN && global_best_fitness <= state->stop_fitness) {
            break;
        }

        /* S-002: 如果未收敛到足够好的解，继续增加种群 */
        if (restart >= max_restarts) break;

        /* 更新保存的尺寸引用，sigma在cmaes_init内已初始化无需手动更新 */
    }

    state->best_fitness = global_best_fitness;
    memcpy(state->best_solution, global_best_solution, saved_dim * sizeof(float));
    safe_free((void**)&global_best_solution);

    return 0;
}

/* S-002修复: BIPOP双种群重启策略
 *
 * BIPOP (BI-Population) 策略原理:
 * - 在IPOP基础上交替使用小种群(快速收敛)和大种群(全局探索)
 * - 偶数重启: 小种群 → λ = base * (0.5)^(restart/2+1) → 快速找到局部最优
 * - 奇数重启: 大种群 → λ = base * 2^((restart+1)/2) → 跳出局部最优搜索全局
 * - 交替策略确保既有局部精细化又有全局探索能力
 *
 * 修复要点:
 * - 保存dimension在cmaes_free之前（避免zero-out丢失维度信息）
 * - 保存旧的最优均值作为重启起点
 * - 小种群比例递减、大种群比例递增 */
int cmaes_bipop_optimize(CMAESState* state, CMAESFitnessFunction func, void* user_data,
                          int max_restarts) {
    if (!state || !func) return -1;
    if (max_restarts <= 0) max_restarts = CMAES_BIPOP_MAX_RESTARTS;
    if (max_restarts > CMAES_BIPOP_MAX_RESTARTS) max_restarts = CMAES_BIPOP_MAX_RESTARTS;

    float global_best_fitness = FLT_MAX;
    float* global_best_solution = (float*)safe_malloc(state->dimension * sizeof(float));
    if (!global_best_solution) return -1;

    int base_lambda = CMAES_BIPOP_LAMBDA_DEFAULT;
    float old_sigma = state->sigma;
    /* S-002: 保存dimension，防止cmaes_free清零 */
    size_t saved_dim = state->dimension;

    for (int restart = 0; restart <= max_restarts; restart++) {
        state->ipop_restart_count = restart;

        if (restart > 0) {
            /* BIPOP交替策略: 偶数重启用小种群(快速收敛)，奇数重启用大种群(全局探索) */
            int new_lambda;
            if (restart % 2 == 0) {
                /* 小种群: λ = base * 0.5^(restart/2+1) */
                new_lambda = (int)((float)base_lambda *
                    powf(CMAES_BIPOP_SMALL_POP_RATIO, (float)(restart / 2 + 1)));
            } else {
                /* 大种群: λ = base * 2^(restart/2+1) */
                new_lambda = (int)((float)base_lambda *
                    powf(CMAES_BIPOP_LARGE_POP_RATIO, (float)((restart + 1) / 2)));
            }

            if (new_lambda > CMAES_MAX_POP) new_lambda = CMAES_MAX_POP;
            if (new_lambda < CMAES_MIN_POP) new_lambda = CMAES_MIN_POP;
            if (new_lambda <= base_lambda / 4) new_lambda = base_lambda / 4;
            if (new_lambda == state->lambda) new_lambda = state->lambda + 1;

            /* 保存当前最优均值作为重启种子 */
            float* old_mean = (float*)safe_malloc(saved_dim * sizeof(float));
            if (old_mean) {
                memcpy(old_mean, state->mean, saved_dim * sizeof(float));
            }

            /* S-002: 保存sigma，因为在cmaes_free中会清零 */
            float saved_sigma = state->sigma;
            int seed = (int)(state->rng_state + (unsigned int)restart * 67890u);

            /* 释放旧状态（会清零整个结构体） */
            cmaes_free(state);

            /* S-002: 使用保存的dimension初始化新状态 */
            if (cmaes_init(state, saved_dim, saved_sigma, new_lambda, seed) != 0) {
                safe_free((void**)&global_best_solution);
                if (old_mean) safe_free((void**)&old_mean);
                return -1;
            }

            state->enable_active = 1;
            state->ipop_restart_count = restart;

            /* 从旧最优均值继续搜索 */
            if (old_mean) {
                memcpy(state->mean, old_mean, saved_dim * sizeof(float));
                safe_free((void**)&old_mean);
            }
        }

        int result = cmaes_optimize(state, func, user_data);
        if (result != 0) {
/* BIPOP中的CMA-ES优化失败处理 */
            log_warning("[CMA-ES BIPOP] 子优化失败 (result=%d), 继续下次重启", result);
        }

        if (state->best_fitness < global_best_fitness) {
            global_best_fitness = state->best_fitness;
            memcpy(global_best_solution, state->best_solution, saved_dim * sizeof(float));
        }

        /* 终止条件检查 */
        if (state->generation >= state->max_generations) break;

        if (state->termination_reason == CMAES_TERM_TOLFUN &&
            global_best_fitness <= state->stop_fitness) {
            break;
        }

        /* S-002: 如果未收敛到足够好的解，继续交替大小种群 */
        if (restart >= max_restarts) break;

        /* sigma在cmaes_init内已初始化无需手动更新 */
    }

    state->best_fitness = global_best_fitness;
    memcpy(state->best_solution, global_best_solution, saved_dim * sizeof(float));
    safe_free((void**)&global_best_solution);
    return 0;
}
