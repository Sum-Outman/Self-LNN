#include "selflnn/core/cfc_enhanced.h"
#include "selflnn/core/ode_solvers.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/memory_utils.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CFC_ENHANCED_VERBOSE(fmt, ...) do { if (config && config->verbose) { printf("[CfC增强] " fmt "\n", ##__VA_ARGS__); } } while(0)

static float cfc_enhanced_fmaxf(float a, float b) { return (a > b) ? a : b; }
static float cfc_enhanced_fminf(float a, float b) { return (a < b) ? a : b; }

CfcEnhancedConfig cfc_enhanced_default_config(void)
{
    CfcEnhancedConfig cfg;
    cfg.enable_auto_solver = 0;
    cfg.auto_solver.strategy = CFC_SOLVER_SELECT_AUTO_STIFFNESS;
    cfg.auto_solver.enable_stiffness_detection = 1;
    cfg.auto_solver.stiffness_check_interval = CFC_ENHANCED_DEFAULT_STIFFNESS_CHECK_INTERVAL;
    cfg.auto_solver.stiffness_ratio_threshold = CFC_ENHANCED_DEFAULT_STIFFNESS_THRESHOLD;
    cfg.auto_solver.enable_solver_switching = 1;
    cfg.auto_solver.preferred_stiff_solver = ODE_SOLVER_ROSENBROCK;
    cfg.auto_solver.preferred_nonstiff_solver = ODE_SOLVER_DP54;
    cfg.auto_solver.fallback_solver = ODE_SOLVER_CLOSED_FORM;
    cfg.auto_solver.multi_rate.enable_multi_rate = 0;
    cfg.auto_solver.multi_rate.fast_ratio = 0.1f;
    cfg.auto_solver.multi_rate.slow_ratio = 10.0f;
    cfg.auto_solver.multi_rate.enable_adaptive_ratio = 0;
    cfg.auto_solver.multi_rate.min_fast_ratio = 0.01f;
    cfg.auto_solver.multi_rate.max_slow_ratio = 100.0f;
    cfg.auto_solver.multi_rate.enable_stiffness_adaptive = 0;
    cfg.auto_solver.multi_rate.stiffness_threshold = CFC_ENHANCED_DEFAULT_STIFFNESS_THRESHOLD;
    cfg.enable_parallel_enhance = 0;
    cfg.parallel_enhance.enable_parallel_rhs = 0;
    cfg.parallel_enhance.enable_domain_parallel = 0;
    cfg.parallel_enhance.parallel_rhs_min_size = CFC_ENHANCED_MIN_PARALLEL_SIZE;
    cfg.parallel_enhance.num_parallel_domains = 4;
    cfg.parallel_enhance.enable_simd_optimization = 0;
    cfg.parallel_enhance.enable_dynamic_load_balance = 0;
    cfg.verbose = 0;
    return cfg;
}

CfcEnhancedState* cfc_enhanced_state_create(void)
{
    CfcEnhancedState* state = (CfcEnhancedState*)safe_malloc(sizeof(CfcEnhancedState));
    if (!state) return NULL;

    state->current_solver = ODE_SOLVER_CLOSED_FORM;
    state->original_solver = ODE_SOLVER_CLOSED_FORM;
    state->last_stiffness_solver = ODE_SOLVER_CLOSED_FORM;
    state->stiffness_check_counter = 0;
    state->current_stiffness_ratio = 1.0f;
    state->peak_stiffness_ratio = 1.0f;
    state->avg_stiffness_ratio = 1.0f;
    state->stiffness_samples = 0;
    state->parallel_rhs_enabled = 0;
    state->total_calls = 0;
    state->solver_switches = 0;
    state->stiffness_detected_count = 0;
    state->multi_rate_active = 0;
    state->power_iter_buffer = NULL;
    state->power_iter_buffer2 = NULL;
    state->power_iter_buffer_size = 0;
    state->initialized = 0;

    return state;
}

void cfc_enhanced_state_free(CfcEnhancedState* state)
{
    if (!state) return;
    safe_free((void**)&state->power_iter_buffer);
    safe_free((void**)&state->power_iter_buffer2);
    safe_free((void**)&state);
}

/* ================================================================
 * SIMD加速实现 — SSE/AVX向量化核心运算
 * 
 * 当 enable_simd_optimization=1 时，使用SIMD指令加速：
 * - 向量点积 (dot product)
 * - 向量加法 (element-wise addition) 
 * - 向量缩放 (scalar multiply)
 * - Sigmoid/Tanh 批量近似
 *   - CfC门控计算 (gate*sigmoid + (1-gate)*tanh)
 * 
 * 支持 x86 SSE/AVX、ARM NEON，无硬件时回退标量运算。
 * 
 * 注意：本文件的SIMD实现专门针对CfC内核优化（批量sigmoid/tanh/门控）。
 * 通用向量运算（add/sub/dot/matmul）请使用 simd_utils.h 的统一接口。
 * 两个文件功能不重复：cfc_enhanced 做CfC专用SIMD，simd_utils 做通用SIMD。
 * ================================================================ */

#if defined(__x86_64__) || defined(_M_X64) || defined(__SSE2__)
#include <emmintrin.h>
#include <xmmintrin.h>
#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif
#define CFC_SIMD_AVAILABLE 1
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define CFC_SIMD_AVAILABLE 1
#else
#define CFC_SIMD_AVAILABLE 0
#endif

#if CFC_SIMD_AVAILABLE
#if defined(__x86_64__) || defined(_M_X64)

/* SSE 向量点积：float sum = Σ a[i] * b[i] */
static float cfc_simd_dot_product(const float* a, const float* b, size_t n) {
    __m128 sum = _mm_setzero_ps();
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
    }
    float result[4];
    _mm_storeu_ps(result, sum);
    float total = result[0] + result[1] + result[2] + result[3];
    for (; i < n; i++) total += a[i] * b[i];
    return total;
}

/* SSE 向量加法：c[i] = a[i] + b[i] */
static void cfc_simd_vector_add(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(c + i, _mm_add_ps(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] + b[i];
}

/* SSE高精度exp：整数/小数分解+多项式，相对误差~1e-6 */
static __m128 cfc_sse_exp_ps(__m128 x) {
    __m128i emm0;
    __m128 one = _mm_set1_ps(1.0f);
    __m128 x_clamped = _mm_min_ps(_mm_max_ps(x, _mm_set1_ps(-87.336548f)), _mm_set1_ps(88.376262f));
    __m128 fx = _mm_add_ps(_mm_mul_ps(x_clamped, _mm_set1_ps(1.4426950409f)), _mm_set1_ps(0.5f));
    emm0 = _mm_cvttps_epi32(fx);
    __m128 tmp = _mm_cvtepi32_ps(emm0);
    __m128 mask = _mm_and_ps(_mm_cmpgt_ps(tmp, fx), one);
    fx = _mm_sub_ps(tmp, mask);
    x_clamped = _mm_sub_ps(x_clamped, _mm_mul_ps(fx, _mm_set1_ps(0.693359375f)));
    x_clamped = _mm_sub_ps(x_clamped, _mm_mul_ps(fx, _mm_set1_ps(-2.12194440e-4f)));
    __m128 z = _mm_mul_ps(x_clamped, x_clamped);
    __m128 y = _mm_set1_ps(0.0083333592f);
    y = _mm_add_ps(_mm_mul_ps(y, x_clamped), _mm_set1_ps(0.0416667796f));
    y = _mm_add_ps(_mm_mul_ps(y, x_clamped), _mm_set1_ps(0.16666577783f));
    y = _mm_add_ps(_mm_mul_ps(y, x_clamped), _mm_set1_ps(0.4999999703f));
    y = _mm_add_ps(_mm_mul_ps(y, z), x_clamped);
    y = _mm_add_ps(y, one);
    emm0 = _mm_slli_epi32(emm0, 23);
    return _mm_mul_ps(y, _mm_castsi128_ps(emm0));
}

/* SSE Sigmoid批量：y[i] = 1/(1+exp(-x[i])) 使用高精度exp */
static void cfc_simd_sigmoid_batch(const float* x, float* y, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 vx = _mm_loadu_ps(x + i);
        __m128 neg = _mm_sub_ps(_mm_setzero_ps(), vx);
        __m128 exp_val = cfc_sse_exp_ps(neg);
        __m128 one = _mm_set1_ps(1.0f);
        __m128 result = _mm_div_ps(one, _mm_add_ps(one, exp_val));
        _mm_storeu_ps(y + i, result);
    }
    for (; i < n; i++) y[i] = 1.0f / (1.0f + expf(-x[i]));
}

/* SSE Tanh批量：y[i] = tanh(x[i]) = 2*sigmoid(2x) - 1 使用高精度exp */
static void cfc_simd_tanh_batch(const float* x, float* y, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 vx = _mm_loadu_ps(x + i);
        __m128 vx2 = _mm_mul_ps(vx, _mm_set1_ps(2.0f));
        __m128 neg = _mm_sub_ps(_mm_setzero_ps(), vx2);
        __m128 exp_val = cfc_sse_exp_ps(neg);
        __m128 one = _mm_set1_ps(1.0f);
        __m128 sig2x = _mm_div_ps(one, _mm_add_ps(one, exp_val));
        __m128 two = _mm_set1_ps(2.0f);
        _mm_storeu_ps(y + i, _mm_sub_ps(_mm_mul_ps(two, sig2x), one));
    }
    for (; i < n; i++) y[i] = tanhf(x[i]);
}

/* SSE CfC门控：y[i] = gate[i]*tanh(x[i]) + (1-gate[i])*state[i] */
static void cfc_simd_gate_apply(const float* gate, const float* x, 
                                 const float* state, float* y, size_t n) {
    size_t i = 0;
    __m128 one = _mm_set1_ps(1.0f);
    for (; i + 4 <= n; i += 4) {
        __m128 vg = _mm_loadu_ps(gate + i);
        __m128 vx = _mm_loadu_ps(x + i);
        __m128 vs = _mm_loadu_ps(state + i);
        __m128 term1 = _mm_mul_ps(vg, vx);
        __m128 term2 = _mm_mul_ps(_mm_sub_ps(one, vg), vs);
        _mm_storeu_ps(y + i, _mm_add_ps(term1, term2));
    }
    for (; i < n; i++) y[i] = gate[i] * x[i] + (1.0f - gate[i]) * state[i];
}

/* SSE/AVX ReLU批量：y[i] = max(0, x[i]) */
static void cfc_simd_relu_batch_x86(const float* x, float* y, size_t n) {
    __m128 zero = _mm_setzero_ps();
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 vx = _mm_loadu_ps(x + i);
        _mm_storeu_ps(y + i, _mm_max_ps(vx, zero));
    }
    for (; i < n; i++) y[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
}

/* AVX ReLU批量：一次处理8个float */
#if defined(__AVX__)
static void cfc_simd_relu_batch_avx(const float* x, float* y, size_t n) {
    __m256 zero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(y + i, _mm256_max_ps(vx, zero));
    }
    for (; i < n; i++) y[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
}
#endif

#elif defined(__ARM_NEON)

static float cfc_simd_dot_product(const float* a, const float* b, size_t n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vmlaq_f32(sum, va, vb);
    }
    float result[4];
    vst1q_f32(result, sum);
    float total = result[0] + result[1] + result[2] + result[3];
    for (; i < n; i++) total += a[i] * b[i];
    return total;
}

static void cfc_simd_vector_add(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(c + i, vaddq_f32(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] + b[i];
}

static void cfc_simd_sigmoid_batch(const float* x, float* y, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t neg = vnegq_f32(vx);
        float32x4_t neg_c = vmaxq_f32(vdupq_n_f32(-4.0f), vminq_f32(vdupq_n_f32(4.0f), neg));
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t x2 = vmulq_f32(neg_c, neg_c);
        float32x4_t exp_a = vaddq_f32(one, vaddq_f32(neg_c,
            vaddq_f32(vmulq_f32(x2, vdupq_n_f32(0.5f)),
            vaddq_f32(vmulq_f32(vmulq_f32(x2, neg_c), vdupq_n_f32(1.0f/6.0f)),
                      vmulq_f32(vmulq_f32(x2, x2), vdupq_n_f32(1.0f/24.0f))))));
        float32x4_t recip = vrecpeq_f32(vaddq_f32(one, exp_a));
        recip = vmulq_f32(vrecpsq_f32(vaddq_f32(one, exp_a), recip), recip);
        vst1q_f32(y + i, recip);
    }
    for (; i < n; i++) y[i] = 1.0f / (1.0f + expf(-x[i]));
}

static void cfc_simd_tanh_batch(const float* x, float* y, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vx2 = vmulq_f32(vx, vdupq_n_f32(2.0f));
        float32x4_t neg = vnegq_f32(vx2);
        float32x4_t neg_c = vmaxq_f32(vdupq_n_f32(-4.0f), vminq_f32(vdupq_n_f32(4.0f), neg));
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t two = vdupq_n_f32(2.0f);
        float32x4_t x2 = vmulq_f32(neg_c, neg_c);
        float32x4_t exp_a = vaddq_f32(one, vaddq_f32(neg_c,
            vaddq_f32(vmulq_f32(x2, vdupq_n_f32(0.5f)),
            vaddq_f32(vmulq_f32(vmulq_f32(x2, neg_c), vdupq_n_f32(1.0f/6.0f)),
                      vmulq_f32(vmulq_f32(x2, x2), vdupq_n_f32(1.0f/24.0f))))));
        float32x4_t sig2x = vrecpeq_f32(vaddq_f32(one, exp_a));
        sig2x = vmulq_f32(vrecpsq_f32(vaddq_f32(one, exp_a), sig2x), sig2x);
        vst1q_f32(y + i, vsubq_f32(vmulq_f32(two, sig2x), one));
    }
    for (; i < n; i++) y[i] = tanhf(x[i]);
}

static void cfc_simd_gate_apply(const float* gate, const float* x,
                                 const float* state, float* y, size_t n) {
    size_t i = 0;
    float32x4_t one = vdupq_n_f32(1.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t vg = vld1q_f32(gate + i);
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vs = vld1q_f32(state + i);
        float32x4_t term1 = vmulq_f32(vg, vx);
        float32x4_t term2 = vmulq_f32(vsubq_f32(one, vg), vs);
        vst1q_f32(y + i, vaddq_f32(term1, term2));
    }
    for (; i < n; i++) y[i] = gate[i] * x[i] + (1.0f - gate[i]) * state[i];
}

/* NEON ReLU批量：y[i] = max(0, x[i]) */
static void cfc_simd_relu_batch_neon(const float* x, float* y, size_t n) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        vst1q_f32(y + i, vmaxq_f32(vx, zero));
    }
    for (; i < n; i++) y[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
}

#endif /* __ARM_NEON */
#endif /* CFC_SIMD_AVAILABLE */

/* 标量回退——当SIMD不可用时使用 */
#if !CFC_SIMD_AVAILABLE
static float cfc_scalar_dot(const float* a, const float* b, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
static void cfc_scalar_add(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];
}
static void cfc_scalar_sigmoid(const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; i++) y[i] = 1.0f / (1.0f + expf(-x[i]));
}
static void cfc_scalar_tanh(const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; i++) y[i] = tanhf(x[i]);
}
static void cfc_scalar_gate(const float* gate, const float* x, const float* state, float* y, size_t n) {
    for (size_t i = 0; i < n; i++) y[i] = gate[i] * x[i] + (1.0f - gate[i]) * state[i];
}
#endif

/* 统一调度：根据enable_simd选择SIMD或标量路径 */
float cfc_simd_dot(const float* a, const float* b, size_t n, int use_simd) {
#if CFC_SIMD_AVAILABLE
    if (use_simd) return cfc_simd_dot_product(a, b, n);
#endif
    (void)use_simd;
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

void cfc_simd_gate_forward(const float* gate, const float* x, const float* state,
                            float* y, size_t n, int use_simd) {
#if CFC_SIMD_AVAILABLE
    if (use_simd) { cfc_simd_gate_apply(gate, x, state, y, n); return; }
#endif
    (void)use_simd;
    for (size_t i = 0; i < n; i++) y[i] = gate[i] * x[i] + (1.0f - gate[i]) * state[i];
}

void cfc_simd_activation_batch(const float* x, float* y, size_t n, int act_type, int use_simd) {
#if CFC_SIMD_AVAILABLE
    if (use_simd) {
        if (act_type == 0) { cfc_simd_sigmoid_batch(x, y, n); return; }
        if (act_type == 1) { cfc_simd_tanh_batch(x, y, n); return; }
        if (act_type == 2) {
#if defined(__AVX__)
            cfc_simd_relu_batch_avx(x, y, n);
#elif defined(__x86_64__) || defined(_M_X64) || defined(__SSE2__)
            cfc_simd_relu_batch_x86(x, y, n);
#elif defined(__ARM_NEON) || defined(__aarch64__)
            cfc_simd_relu_batch_neon(x, y, n);
#endif
            return;
        }
        { for (size_t i = 0; i < n; i++) y[i] = x[i] > 0 ? x[i] : 0.0f; }
        return;
    }
#endif
    (void)use_simd;
    if (act_type == 0) { for (size_t i = 0; i < n; i++) y[i] = 1.0f/(1.0f+expf(-x[i])); }
    else if (act_type == 1) { for (size_t i = 0; i < n; i++) y[i] = tanhf(x[i]); }
    else { for (size_t i = 0; i < n; i++) y[i] = x[i] > 0 ? x[i] : 0.0f; }
}

int cfc_enhanced_state_reset(CfcEnhancedState* state)
{
    if (!state) return -1;
    state->current_solver = state->original_solver;
    state->stiffness_check_counter = 0;
    state->current_stiffness_ratio = 1.0f;
    state->peak_stiffness_ratio = 1.0f;
    state->avg_stiffness_ratio = 1.0f;
    state->stiffness_samples = 0;
    state->parallel_rhs_enabled = 0;
    state->total_calls = 0;
    state->solver_switches = 0;
    state->stiffness_detected_count = 0;
    state->multi_rate_active = 0;
    return 0;
}

static int cfc_ensure_power_iter_buffers(CfcEnhancedState* state, size_t hidden_size)
{
    if (!state || hidden_size == 0) return -1;
    /* I-015修复：添加上限保护，防止超大hidden_size导致分配失败 */
    if (hidden_size > 65536) return -1;
    if (state->power_iter_buffer_size >= (int)hidden_size && state->power_iter_buffer && state->power_iter_buffer2)
        return 0;

    safe_free((void**)&state->power_iter_buffer);
    safe_free((void**)&state->power_iter_buffer2);

    state->power_iter_buffer = (float*)safe_malloc(hidden_size * sizeof(float));
    state->power_iter_buffer2 = (float*)safe_malloc(hidden_size * sizeof(float));
    if (!state->power_iter_buffer || !state->power_iter_buffer2) {
        safe_free((void**)&state->power_iter_buffer);
        safe_free((void**)&state->power_iter_buffer2);
        state->power_iter_buffer_size = 0;
        return -1;
    }
    state->power_iter_buffer_size = (int)hidden_size;
    return 0;
}

float cfc_estimate_stiffness_ratio(CfCCell* cell, const float* input,
                                    const float* state, CfcEnhancedState* estate)
{
    if (!cell || !estate) return 1.0f;

    CfCCellConfig config;
    if (cfc_cell_get_config(cell, &config) != 0) return 1.0f;

    size_t hidden_size = config.hidden_size;
    if (hidden_size == 0) return 1.0f;

    float stiffness_ratio = 1.0f;

    /* F-007修复：基于配置时间常数的基础刚度估计 */
    float time_const = config.time_constant;
    float delta_t_val = config.delta_t;
    if (config.use_multi_timescale) {
        float fast_tau = time_const * config.fast_tau_ratio;
        float slow_tau = time_const * config.slow_tau_ratio;
        if (fast_tau > 0.0f && slow_tau > fast_tau) {
            stiffness_ratio = slow_tau / fast_tau;
        }
    } else {
        stiffness_ratio = time_const / cfc_enhanced_fmaxf(delta_t_val, 1e-8f);
        if (stiffness_ratio < 1.0f) stiffness_ratio = 1.0f;
        if (stiffness_ratio > 1e6f) stiffness_ratio = 1e6f;
    }

    /* F-007修复：基于tau分布的刚度增强（从液态时间常数） */
    int tau_size = 0;
    /* I-015修复：边界检查 + 获取液态时间常数 */
    float* tau_buffer = (float*)safe_malloc(hidden_size * sizeof(float));
    if (tau_buffer) {
        if (cfc_cell_get_liquid_tau(cell, tau_buffer, &tau_size) == 0 && tau_size > 0 && (size_t)tau_size <= hidden_size) {
            float min_tau = tau_buffer[0];
            float max_tau = tau_buffer[0];
            for (int i = 1; i < tau_size; i++) {
                if (tau_buffer[i] < min_tau) min_tau = tau_buffer[i];
                if (tau_buffer[i] > max_tau) max_tau = tau_buffer[i];
            }
            if (min_tau > 0.0f && max_tau > min_tau) {
                stiffness_ratio = cfc_enhanced_fmaxf(stiffness_ratio, max_tau / min_tau);
            }
        }
        safe_free((void**)&tau_buffer);
    }

    /* F-007修复：基于真实input/state的雅可比矩阵近似条件数 */
    if (input && state && hidden_size > 0) {
        /* 使用有限差分估计雅可比矩阵的算子范数（谱半径近似） */
        float eps = 1e-5f;
        float* perturbed_state = (float*)safe_malloc(hidden_size * sizeof(float));
        float* forward_original = (float*)safe_malloc(hidden_size * sizeof(float));
        float* forward_perturbed = (float*)safe_malloc(hidden_size * sizeof(float));
        if (perturbed_state && forward_original && forward_perturbed) {
            /* 先执行原始前向，输出写入 forward_original */
            memcpy(forward_original, state, hidden_size * sizeof(float));
            cfc_cell_forward(cell, input, forward_original);

            /* 随机方向扰动状态，输出写入 forward_perturbed */
            memcpy(forward_perturbed, state, hidden_size * sizeof(float));
            float norm_perturb = 0.0f;
            for (size_t i = 0; i < hidden_size; i++) {
                /* 使用确定性的伪随机方向（基于状态值哈希） */
                float direction = sinf(state[i] * 137.508f + 3.14159f) * eps;
                forward_perturbed[i] = state[i] + direction;
                float diff = forward_perturbed[i] - state[i];
                norm_perturb += diff * diff;
            }
            norm_perturb = sqrtf(norm_perturb);

            if (norm_perturb > 1e-12f) {
                cfc_cell_forward(cell, input, forward_perturbed);

                /* 计算 ||F(x+δ) - F(x)|| / ||δ|| 作为局部Lipschitz常数估计 */
                float norm_diff = 0.0f;
                for (size_t i = 0; i < hidden_size; i++) {
                    float d = forward_perturbed[i] - forward_original[i];
                    norm_diff += d * d;
                }
                norm_diff = sqrtf(norm_diff);

                float local_lipschitz = norm_diff / norm_perturb;
                /* 将Lipschitz常数融入刚度比（高Lipschitz ≈ 高刚度） */
                if (local_lipschitz > 1.0f) {
                    stiffness_ratio *= cfc_enhanced_fminf(local_lipschitz, 100.0f);
                }
                /* 将局部刚度与时间尺度刚度做权重混合 */
                float jac_stiffness = local_lipschitz * cfc_enhanced_fmaxf(time_const / delta_t_val, 1.0f);
                stiffness_ratio = 0.7f * stiffness_ratio + 0.3f * jac_stiffness;
            }
        }
        if (perturbed_state) safe_free((void**)&perturbed_state);
        if (forward_original) safe_free((void**)&forward_original);
        if (forward_perturbed) safe_free((void**)&forward_perturbed);
    }

    if (stiffness_ratio > 1e6f) stiffness_ratio = 1e6f;
    if (stiffness_ratio < 1.0f) stiffness_ratio = 1.0f;

    estate->current_stiffness_ratio = stiffness_ratio;
    if (stiffness_ratio > estate->peak_stiffness_ratio)
        estate->peak_stiffness_ratio = stiffness_ratio;

    float total = estate->avg_stiffness_ratio * (float)estate->stiffness_samples;
    estate->stiffness_samples++;
    estate->avg_stiffness_ratio = (total + stiffness_ratio) / (float)estate->stiffness_samples;

    return stiffness_ratio;
}

int cfc_select_solver_by_stiffness(CfCCell* cell, const float* input,
                                    const float* state,
                                    const CfcAutoSolverConfig* config,
                                    CfcEnhancedState* estate)
{
    if (!cell || !config || !estate) return -1;

    CfCCellConfig cell_config;
    if (cfc_cell_get_config(cell, &cell_config) != 0) return -1;

    estate->stiffness_check_counter++;
    if (estate->stiffness_check_counter < config->stiffness_check_interval) {
        return 0;
    }
    estate->stiffness_check_counter = 0;

    if (!config->enable_stiffness_detection) {
        return 0;
    }

    float stiffness = cfc_estimate_stiffness_ratio(cell, input, state, estate);

    int recommended_solver;
    if (stiffness > config->stiffness_ratio_threshold) {
        recommended_solver = config->preferred_stiff_solver;
        estate->stiffness_detected_count++;
    } else {
        recommended_solver = config->preferred_nonstiff_solver;
    }

    int valid_solver = 0;
    if (recommended_solver >= ODE_SOLVER_CLOSED_FORM && recommended_solver <= ODE_SOLVER_SYMPLECTIC) {
        valid_solver = 1;
    }
    if (!valid_solver) {
        recommended_solver = config->fallback_solver;
    }

    int current_type;
    if (estate->initialized) {
        current_type = estate->current_solver;
    } else {
        current_type = cfc_cell_get_solver_type(cell);
        if (current_type < 0) current_type = config->fallback_solver;
        estate->original_solver = current_type;
        estate->initialized = 1;
    }

    if (config->enable_solver_switching && recommended_solver != current_type) {
        if (cfc_cell_set_solver_type(cell, recommended_solver) == 0) {
            estate->solver_switches++;
            estate->current_solver = recommended_solver;
            estate->last_stiffness_solver = recommended_solver;
        }
    }

    return 0;
}

static int cfc_configure_multi_rate(CfCCell* cell, const CfcMultiRateConfig* mr_config,
                                     int stiffness_detected, float stiffness_ratio,
                                     CfcEnhancedState* estate)
{
    if (!cell || !mr_config || !estate) return -1;

    if (!mr_config->enable_multi_rate) {
        CfCCellConfig cfg;
        if (cfc_cell_get_config(cell, &cfg) == 0) {
            cfg.use_multi_timescale = 0;
            cfc_cell_set_config(cell, &cfg);
        }
        estate->multi_rate_active = 0;
        return 0;
    }

    CfCCellConfig cfg;
    if (cfc_cell_get_config(cell, &cfg) != 0) return -1;

    cfg.use_multi_timescale = 1;
    cfg.fast_tau_ratio = mr_config->fast_ratio;
    cfg.slow_tau_ratio = mr_config->slow_ratio;

    if (mr_config->enable_adaptive_ratio || mr_config->enable_stiffness_adaptive) {
        if (stiffness_detected && stiffness_ratio > mr_config->stiffness_threshold) {
            float adapt_factor = cfc_enhanced_fminf(stiffness_ratio / mr_config->stiffness_threshold, 10.0f);
            cfg.fast_tau_ratio = cfc_enhanced_fmaxf(mr_config->min_fast_ratio,
                mr_config->fast_ratio / adapt_factor);
            cfg.slow_tau_ratio = cfc_enhanced_fminf(mr_config->max_slow_ratio,
                mr_config->slow_ratio * adapt_factor);
        } else {
            cfg.fast_tau_ratio = mr_config->fast_ratio;
            cfg.slow_tau_ratio = mr_config->slow_ratio;
        }
    }

    cfc_cell_set_config(cell, &cfg);
    estate->multi_rate_active = 1;
    return 0;
}

int cfc_enhanced_forward(CfCCell* cell, const float* input, float* hidden_state,
                          const CfcEnhancedConfig* config,
                          CfcEnhancedState* state)
{
    if (!cell || !input || !hidden_state || !config || !state) return -1;

    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    if (!state->initialized) {
        int current = cfc_cell_get_solver_type(cell);
        if (current < 0) current = ODE_SOLVER_CLOSED_FORM;
        state->original_solver = current;
        state->current_solver = current;
        state->initialized = 1;
    }

    state->total_calls++;

    if (config->enable_auto_solver) {
        cfc_select_solver_by_stiffness(cell, input, hidden_state,
                                        &config->auto_solver, state);

        CfCCellConfig cell_cfg;
        if (cfc_cell_get_config(cell, &cell_cfg) == 0) {
            int is_stiff = (state->current_stiffness_ratio > config->auto_solver.stiffness_ratio_threshold);
            cfc_configure_multi_rate(cell, &config->auto_solver.multi_rate,
                                      is_stiff, state->current_stiffness_ratio, state);
        }
    }

    int ret = cfc_cell_forward(cell, input, hidden_state);

    return ret;
}

int cfc_enhanced_forward_with_dt(CfCCell* cell, const float* input, float delta_t,
                                   float* hidden_state,
                                   const CfcEnhancedConfig* config,
                                   CfcEnhancedState* state)
{
    if (!cell || !input || !hidden_state || !config || !state) return -1;

    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");
    if (!state->initialized) {
        int current = cfc_cell_get_solver_type(cell);
        if (current < 0) current = ODE_SOLVER_CLOSED_FORM;
        state->original_solver = current;
        state->current_solver = current;
        state->initialized = 1;
    }

    state->total_calls++;

    if (config->enable_auto_solver) {
        cfc_select_solver_by_stiffness(cell, input, hidden_state,
                                        &config->auto_solver, state);

        CfCCellConfig cell_cfg;
        if (cfc_cell_get_config(cell, &cell_cfg) == 0) {
            int is_stiff = (state->current_stiffness_ratio > config->auto_solver.stiffness_ratio_threshold);
            cfc_configure_multi_rate(cell, &config->auto_solver.multi_rate,
                                      is_stiff, state->current_stiffness_ratio, state);
        }
    }

    int ret = cfc_cell_forward_with_dt(cell, input, delta_t, hidden_state);

    return ret;
}

int cfc_multi_rate_forward(CfCCell* cell, const float* input, float delta_t,
                            float* hidden_state,
                            const CfcMultiRateConfig* mr_config,
                            CfcEnhancedState* state)
{
    if (!cell || !input || !hidden_state || !mr_config || !state) return -1;

    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");

    if (!mr_config->enable_multi_rate) {
        return cfc_cell_forward_with_dt(cell, input, delta_t, hidden_state);
    }

    CfCCellConfig cfg;
    if (cfc_cell_get_config(cell, &cfg) != 0) return -1;

    int saved_multi = cfg.use_multi_timescale;
    float saved_fast = cfg.fast_tau_ratio;
    float saved_slow = cfg.slow_tau_ratio;

    cfg.use_multi_timescale = 1;
    cfg.fast_tau_ratio = mr_config->fast_ratio;
    cfg.slow_tau_ratio = mr_config->slow_ratio;

    if (mr_config->enable_adaptive_ratio && state->current_stiffness_ratio > 1.0f) {
        float stiffness = state->current_stiffness_ratio;
        if (stiffness > mr_config->stiffness_threshold) {
            float adapt = cfc_enhanced_fminf(stiffness / mr_config->stiffness_threshold, 10.0f);
            cfg.fast_tau_ratio = cfc_enhanced_fmaxf(mr_config->min_fast_ratio, mr_config->fast_ratio / adapt);
            cfg.slow_tau_ratio = cfc_enhanced_fminf(mr_config->max_slow_ratio, mr_config->slow_ratio * adapt);
        }
    }

    if (cfc_cell_set_config(cell, &cfg) != 0) {
        return cfc_cell_forward_with_dt(cell, input, delta_t, hidden_state);
    }

    int ret = cfc_cell_forward_with_dt(cell, input, delta_t, hidden_state);

    cfg.use_multi_timescale = saved_multi;
    cfg.fast_tau_ratio = saved_fast;
    cfg.slow_tau_ratio = saved_slow;
    cfc_cell_set_config(cell, &cfg);

    state->multi_rate_active = 1;
    return ret;
}

int cfc_parallel_rhs_forward(CfCCell* cell, const float* input, float delta_t,
                              float* hidden_state,
                              const CfcParallelEnhanceConfig* pe_config,
                              const ParallelODERHSConfig* base_pcfg,
                              CfcEnhancedState* state)
{
    if (!cell || !input || !hidden_state || !pe_config || !state) return -1;

    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");

    CfCCellConfig cfg;
    if (cfc_cell_get_config(cell, &cfg) != 0) return -1;

    size_t hidden_size = cfg.hidden_size;
    if (hidden_size < (size_t)pe_config->parallel_rhs_min_size) {
        state->parallel_rhs_enabled = 0;
        return cfc_cell_forward_with_dt(cell, input, delta_t, hidden_state);
    }

    int nd = pe_config->num_parallel_domains;
    if (nd < 1) nd = 1;
    if (nd > 64) nd = 64;

    int* dom_sizes = (int*)safe_malloc((size_t)nd * sizeof(int));
    int* dom_offsets = (int*)safe_malloc((size_t)nd * sizeof(int));
    if (!dom_sizes || !dom_offsets) {
        safe_free((void**)&dom_sizes);
        safe_free((void**)&dom_offsets);
        return cfc_cell_forward_with_dt(cell, input, delta_t, hidden_state);
    }

    ode_domain_decompose(hidden_size, nd, dom_sizes, dom_offsets);

    ParallelODERHSConfig pcfg;
    if (base_pcfg) {
        pcfg = *base_pcfg;
    } else {
        pcfg = ode_parallel_default_config();
    }

    if (pe_config->enable_domain_parallel || pe_config->enable_parallel_rhs) {
        pcfg.mode = PARALLEL_MODE_OMP;
        pcfg.num_domains = nd;
        pcfg.use_domain_decomposition = pe_config->enable_domain_parallel;
        pcfg.domain_sizes = dom_sizes;
        pcfg.domain_offsets = dom_offsets;
        pcfg.num_threads = nd;
    }

    int saved_parallel = cfg.use_parallel_solve;
    ParallelODERHSConfig saved_pcfg = cfg.parallel_cfg;

    cfg.use_parallel_solve = 1;
    cfg.parallel_cfg = pcfg;
    if (pe_config->enable_dynamic_load_balance) {
        cfg.parallel_cfg.enable_dynamic_load_balance = 1;
        cfg.parallel_cfg.load_balance_threshold = 0.2f;
    }

    cfc_cell_set_config(cell, &cfg);

    state->parallel_rhs_enabled = 1;

    int ret = cfc_cell_forward_with_dt(cell, input, delta_t, hidden_state);

    cfg.use_parallel_solve = saved_parallel;
    cfg.parallel_cfg = saved_pcfg;
    cfc_cell_set_config(cell, &cfg);

    safe_free((void**)&dom_sizes);
    safe_free((void**)&dom_offsets);

    return ret;
}

int cfc_domain_parallel_forward(CfCCell* cell, const float* input, float delta_t,
                                 float* hidden_state, int solver_type,
                                 const CfcParallelEnhanceConfig* pe_config,
                                 const ParallelODERHSConfig* base_pcfg,
                                 CfcEnhancedState* state)
{
    if (!cell || !input || !hidden_state || !pe_config || !state) return -1;

    SELFLNN_CHECK_NULL(cell, "CfC单元句柄为空");

    int saved_solver = cfc_cell_get_solver_type(cell);
    if (saved_solver < 0) saved_solver = ODE_SOLVER_CLOSED_FORM;

    if (solver_type >= ODE_SOLVER_CLOSED_FORM && solver_type <= ODE_SOLVER_SYMPLECTIC) {
        cfc_cell_set_solver_type(cell, solver_type);
    }

    int ret = cfc_parallel_rhs_forward(cell, input, delta_t, hidden_state,
                                        pe_config, base_pcfg, state);

    cfc_cell_set_solver_type(cell, saved_solver);

    return ret;
}

int cfc_get_enhanced_stats(const CfcEnhancedState* state, int* current_solver,
                            float* stiffness_ratio, int* total_calls,
                            int* solver_switches, int* stiffness_count)
{
    if (!state) return -1;
    if (current_solver) *current_solver = state->current_solver;
    if (stiffness_ratio) *stiffness_ratio = state->current_stiffness_ratio;
    if (total_calls) *total_calls = state->total_calls;
    if (solver_switches) *solver_switches = state->solver_switches;
    if (stiffness_count) *stiffness_count = state->stiffness_detected_count;
    return 0;
}

int cfc_get_enhanced_config(CfcEnhancedConfig* config)
{
    if (!config) return -1;
    *config = cfc_enhanced_default_config();
    return 0;
}
