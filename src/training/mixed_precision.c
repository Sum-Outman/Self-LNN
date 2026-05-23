/**
 * @file mixed_precision.c
 * @brief 混合精度训练支持（FP16/FP32）实现
 * 
 * 混合精度训练系统，支持FP16和FP32精度的自动混合。
 * 提供精度转换、梯度缩放、损失缩放和自动精度管理功能。
 */


#include "selflnn/training/mixed_precision.h"
#include "selflnn/training/training.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/cfc_network.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/platform.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/graph_optimization.h"
#include "selflnn/gpu/gpu.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

// 平台特定的动态库加载头文件
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// SIMD指令集支持检测（根据编译选项自动启用）
#ifdef __F16C__
#define SELFLNN_HAS_F16C 1
#include <immintrin.h>
#else
#define SELFLNN_HAS_F16C 0
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#endif

/* F-038: 运行时F16C检测与批量转换函数 */
static int g_has_f16c_support = -1; /* -1未检测, 0不支持, 1支持 */

static int cpu_has_f16c(void) {
    if (g_has_f16c_support >= 0) return g_has_f16c_support;
    
#if SELFLNN_HAS_F16C
    /* 编译时已启用F16C，运行时再做一次CPUID确认 */
    #ifdef _WIN32
    {
        int cpuid_data[4];
        __cpuid(cpuid_data, 1);
        g_has_f16c_support = (cpuid_data[2] & (1 << 29)) ? 1 : 0;
    }
    #elif defined(__x86_64__)
    {
        int eax=1, ebx, ecx, edx;
        __asm__ __volatile__(
            "cpuid" : "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(eax));
        g_has_f16c_support = (ecx & (1 << 29)) ? 1 : 0;
    }
    #else
        g_has_f16c_support = 0;
    #endif
#else
    g_has_f16c_support = 0;
#endif
    return g_has_f16c_support;
}

typedef uint16_t fp16_t;

/* F-038: F16C加速批量FP32→FP16转换 */
static void fp32_to_fp16_batch_f16c(const float* src, fp16_t* dst, size_t count) {
#if SELFLNN_HAS_F16C
    size_t i;
    for (i = 0; i + 7 < count; i += 8) {
        __m256 vf32 = _mm256_loadu_ps(src + i);
        __m128i vf16 = _mm256_cvtps_ph(vf32, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i*)(dst + i), vf16);
    }
    for (; i < count; i++) {
        __m128 vf32 = _mm_load_ss(src + i);
        __m128i vf16 = _mm_cvtps_ph(vf32, _MM_FROUND_TO_NEAREST_INT);
        dst[i] = (fp16_t)_mm_extract_epi16(vf16, 0);
    }
#else
    (void)src; (void)dst; (void)count;
    /* MID-012修复: 无F16C时使用纯C逐元素转换作为CPU回退 */
    for (size_t i = 0; i < count; i++) {
        unsigned int bits = *(unsigned int*)(src + i);
        unsigned int sign = (bits >> 31) & 1;
        int exponent = ((bits >> 23) & 0xFF) - 127;
        unsigned int mantissa = bits & 0x7FFFFF;
        if (exponent < -14) { dst[i] = 0; }
        else if (exponent > 15) { dst[i] = (fp16_t)(sign ? 0xFC00 : 0x7C00); }
        else {
            int fp16_exp = exponent + 15;
            unsigned int fp16_mantissa = (mantissa >> 13) & 0x3FF;
            dst[i] = (fp16_t)((sign << 15) | (fp16_exp << 10) | fp16_mantissa);
        }
    }
#endif
}

/* F-038: F16C加速批量FP16→FP32转换 */
static void fp16_to_fp32_batch_f16c(const fp16_t* src, float* dst, size_t count) {
#if SELFLNN_HAS_F16C
    size_t i;
    for (i = 0; i + 7 < count; i += 8) {
        __m128i vf16 = _mm_loadu_si128((const __m128i*)(src + i));
        __m256 vf32 = _mm256_cvtph_ps(vf16);
        _mm256_storeu_ps(dst + i, vf32);
    }
    for (; i < count; i++) {
        __m128i vf16 = _mm_set1_epi16((short)src[i]);
        __m128 vf32 = _mm_cvtph_ps(vf16);
        dst[i] = _mm_cvtss_f32(vf32);
    }
#else
    /* P2-058修复: 无F16C时使用纯C逐元素FP16→FP32转换回退 */
    for (size_t i = 0; i < count; i++) {
        unsigned short bits = (unsigned short)src[i];
        unsigned int sign = (bits >> 15) & 1;
        int exponent = ((bits >> 10) & 0x1F) - 15;
        unsigned int mantissa = bits & 0x3FF;
        if (exponent == -15) {
            /* 次正规数 */
            dst[i] = (sign ? -1.0f : 1.0f) * (float)mantissa / 1024.0f * 6.103515625e-5f;
        } else if (exponent == 16) {
            /* 无穷大或NaN */
            if (mantissa == 0) {
                dst[i] = sign ? -INFINITY : INFINITY;
            } else {
                dst[i] = NAN;
            }
        } else {
            dst[i] = (sign ? -1.0f : 1.0f) * (1.0f + (float)mantissa / 1024.0f) * powf(2.0f, (float)exponent);
        }
    }
#endif
}

/* ============================================================================
 * 内部数据结构
 * =========================================================================== */


/* ============================================================================
 * 辅助函数
 * =========================================================================== */

/**
 * @brief 获取数据类型大小（字节）
 */
static size_t get_data_type_size(DataType dtype) {
    switch (dtype) {
        case DATA_TYPE_FLOAT16: return 2;
        case DATA_TYPE_FLOAT32: return 4;
        case DATA_TYPE_FLOAT64: return 8;
        case DATA_TYPE_INT8:    return 1;
        case DATA_TYPE_INT16:   return 2;
        case DATA_TYPE_INT32:   return 4;
        case DATA_TYPE_INT64:   return 8;
        case DATA_TYPE_UINT8:   return 1;
        case DATA_TYPE_UINT16:  return 2;
        case DATA_TYPE_UINT32:  return 4;
        case DATA_TYPE_UINT64:  return 8;
        case DATA_TYPE_BOOL:    return 1;
        default: return 0;
    }
}

/**
 * @brief FP32转FP16（IEEE 754标准实现）
 */
static fp16_t fp32_to_fp16(float value) {
    // 基于IEEE 754标准的FP32到FP16转换算法
    // 参考：https://en.wikipedia.org/wiki/Half-precision_floating-point_format
    
    uint32_t* val_ptr = (uint32_t*)&value;
    uint32_t float_bits = *val_ptr;
    
    // 提取FP32的符号、指数和尾数部分
    uint32_t sign = (float_bits >> 31) & 0x1;
    uint32_t exponent = (float_bits >> 23) & 0xFF;
    uint32_t mantissa = float_bits & 0x7FFFFF;
    
    // 处理特殊情况
    if (exponent == 0xFF) {  // FP32特殊值（NaN或Inf）
        if (mantissa != 0) {
            // NaN：保持NaN的quiet/signaling属性
            return (fp16_t)((sign << 15) | 0x7E00);
        } else {
            // Infinity
            return (fp16_t)((sign << 15) | 0x7C00);
        }
    }
    
    // 计算偏置调整：FP32偏置为127，FP16偏置为15
    int32_t adjusted_exponent = (int32_t)exponent - 127 + 15;
    
    // 处理FP32零和次正规数
    if (exponent == 0) {
        if (mantissa == 0) {
            // 零
            return (fp16_t)(sign << 15);
        }
        // 次正规数：转换为FP16零（损失精度）
        return (fp16_t)(sign << 15);
    }
    
    // 处理FP16指数溢出
    if (adjusted_exponent >= 31) {
        // 溢出到Infinity
        return (fp16_t)((sign << 15) | 0x7C00);
    }
    
    // 处理FP16指数下溢
    if (adjusted_exponent <= 0) {
        // 如果调整后的指数太小，转换为FP16零
        if (adjusted_exponent <= -10) {
            return (fp16_t)(sign << 15);
        }
        
        // 次正规数处理：调整尾数
        mantissa |= 0x800000;  // 恢复隐藏的1位
        uint32_t shift = (uint32_t)(1 - adjusted_exponent);
        mantissa >>= shift;
        adjusted_exponent = 0;
    }
    
    // 构建FP16位模式
    uint32_t fp16_exponent = (uint32_t)adjusted_exponent & 0x1F;
    uint32_t fp16_mantissa = mantissa >> 13;  // FP32有23位尾数，FP16有10位
    
    // 四舍五入：检查第11位（要丢弃的最高位）
    uint32_t rounding_bit = (mantissa >> 12) & 0x1;
    uint32_t sticky_bits = mantissa & 0xFFF;
    
    if (rounding_bit && (fp16_mantissa & 0x1 || sticky_bits != 0)) {
        fp16_mantissa += 1;
        
        // 处理尾数溢出到指数
        if (fp16_mantissa > 0x3FF) {
            fp16_mantissa = 0;
            fp16_exponent += 1;
            
            // 再次检查指数溢出
            if (fp16_exponent > 30) {
                return (fp16_t)((sign << 15) | 0x7C00);
            }
        }
    }
    
    // 组合结果
    return (fp16_t)((sign << 15) | (fp16_exponent << 10) | fp16_mantissa);
}

/**
 * @brief FP16转FP32（IEEE 754标准实现）
 */
static float fp16_to_fp32(fp16_t value) {
    // 基于IEEE 754标准的FP16到FP32转换算法
    
    // 提取FP16的符号、指数和尾数部分
    uint32_t sign = (value >> 15) & 0x1;
    uint32_t exponent = (value >> 10) & 0x1F;
    uint32_t mantissa = value & 0x3FF;
    uint32_t float_bits;
    
    // 处理特殊值
    if (exponent == 0x1F) {  // FP16特殊值（NaN或Inf）
        if (mantissa != 0) {
            // NaN
            float_bits = (sign << 31) | 0x7F800000 | (mantissa << 13);
        } else {
            // Infinity
            float_bits = (sign << 31) | 0x7F800000;
        }
        float result;
        memcpy(&result, &float_bits, sizeof(float));
        return result;
    }
    
    uint32_t float_exponent;
    uint32_t float_mantissa;
    
    if (exponent == 0) {
        // FP16次正规数或零
        if (mantissa == 0) {
            // 零
            float_bits = sign << 31;
            float result;
            memcpy(&result, &float_bits, sizeof(float));
            return result;
        }
        
        // 次正规数：规范化
        // 找到第一个非零位
        int shift = 0;
        while ((mantissa & (1 << (10 - shift - 1))) == 0) {
            shift++;
        }
        
        float_exponent = 127 - 14 - shift;  // FP32偏置127，FP16次正规偏移
        float_mantissa = (mantissa << (shift + 13)) & 0x7FFFFF;
    } else {
        // FP16正规数
        float_exponent = exponent - 15 + 127;  // 偏置调整
        float_mantissa = mantissa << 13;       // FP16有10位尾数，FP32有23位
    }
    
    // 组合FP32位模式
    float_bits = (sign << 31) | (float_exponent << 23) | float_mantissa;
    float result;
    memcpy(&result, &float_bits, sizeof(float));
    return result;
}

/* ============================================================================
 * SIMD加速批量转换（根据CPU指令集自动选择最优路径）
 * =========================================================================== */

#if defined(__F16C__)

/**
 * @brief SIMD加速FP16到FP32批量转换（F16C硬件指令，4路并行）
 */
static void simd_convert_fp16_to_fp32_batch(const fp16_t* src, float* dst, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m128i epi16 = _mm_loadl_epi64((const __m128i*)(src + i));
        __m128 f32 = _mm_cvtph_ps(epi16);
        _mm_storeu_ps(dst + i, f32);
    }
    for (; i < count; i++) {
        dst[i] = fp16_to_fp32(src[i]);
    }
}

/**
 * @brief SIMD加速FP32到FP16批量转换（F16C硬件指令，4路并行）
 */
static void simd_convert_fp32_to_fp16_batch(const float* src, fp16_t* dst, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m128 f32 = _mm_loadu_ps(src + i);
        __m128i epi16 = _mm_cvtps_ph(f32, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i*)(dst + i), epi16);
    }
    for (; i < count; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

#elif defined(__SSE2__)

/**
 * @brief SIMD加速FP16到FP32批量转换（SSE2手动实现，含次正规数处理）
 * 
 * 使用魔术常量法进行快速转换，对次正规数和零使用标量回退。
 * 原理：FP16值左移16位后加上魔术常量(113<<23)，自动完成偏置调整。
 */
static void simd_convert_fp16_to_fp32_batch(const fp16_t* src, float* dst, size_t count) {
    size_t i = 0;
    const __m128i magic = _mm_set1_epi32(113 << 23);
    const __m128i exp_mask = _mm_set1_epi32(0x7C00);
    const __m128i zero = _mm_setzero_si128();

    for (; i + 4 <= count; i += 4) {
        __m128i h4 = _mm_loadl_epi64((const __m128i*)(src + i));
        __m128i h4_32 = _mm_unpacklo_epi16(h4, zero);

        __m128i exp_field = _mm_and_si128(h4_32, exp_mask);
        __m128i subnormal_mask = _mm_cmpeq_epi32(exp_field, zero);

        if (_mm_movemask_epi8(subnormal_mask) == 0) {
            __m128i shifted = _mm_slli_epi32(h4_32, 16);
            __m128 result = _mm_castsi128_ps(_mm_add_epi32(shifted, magic));
            _mm_storeu_ps(dst + i, result);
        } else {
            for (int j = 0; j < 4; j++) {
                dst[i + j] = fp16_to_fp32(src[i + j]);
            }
        }
    }
    for (; i < count; i++) {
        dst[i] = fp16_to_fp32(src[i]);
    }
}

/**
 * @brief SIMD加速FP32到FP16批量转换（SSE2手动实现，含次正规数处理）
 */
static void simd_convert_fp32_to_fp16_batch(const float* src, fp16_t* dst, size_t count) {
    size_t i = 0;
    const __m128i exp_inf_mask = _mm_set1_epi32(0x7F800000);
    const __m128i mant_mask = _mm_set1_epi32(0x007FFFFF);
    const __m128i zero = _mm_setzero_si128();

    for (; i + 4 <= count; i += 4) {
        __m128 f32 = _mm_loadu_ps(src + i);
        __m128i f32_bits = _mm_castps_si128(f32);
        __m128i sign = _mm_and_si128(f32_bits, _mm_set1_epi32(0x80000000));
        __m128i exp = _mm_and_si128(f32_bits, exp_inf_mask);
        __m128i mant = _mm_and_si128(f32_bits, mant_mask);

        __m128i is_inf_nan = _mm_cmpeq_epi32(exp, exp_inf_mask);
        __m128i is_zero_subnormal = _mm_cmpeq_epi32(exp, zero);

        __m128i fp16_exp = _mm_srli_epi32(exp, 23);
        fp16_exp = _mm_sub_epi32(fp16_exp, _mm_set1_epi32(127 - 15));
        fp16_exp = _mm_slli_epi32(fp16_exp, 10);
        fp16_exp = _mm_andnot_si128(is_inf_nan, fp16_exp);
        fp16_exp = _mm_or_si128(fp16_exp,
            _mm_and_si128(is_inf_nan, _mm_set1_epi32(0x7C00)));
        fp16_exp = _mm_andnot_si128(is_zero_subnormal, fp16_exp);

        __m128i fp16_mant = _mm_srli_epi32(mant, 13);

        __m128i fp16_bits = _mm_or_si128(_mm_srli_epi32(sign, 16),
            _mm_or_si128(fp16_exp, fp16_mant));

        __m128i fp16_packed = _mm_packus_epi32(fp16_bits, zero);
        _mm_storel_epi64((__m128i*)(dst + i), fp16_packed);

        if (_mm_movemask_epi8(is_zero_subnormal) != 0) {
            for (int j = 0; j < 4; j++) {
                if (src[i + j] != 0.0f && fabsf(src[i + j]) < 6.1e-5f) {
                    dst[i + j] = fp32_to_fp16(src[i + j]);
                }
            }
        }
    }
    for (; i < count; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

#else

/**
 * @brief 无SIMD支持时的标量回退
 */
static void simd_convert_fp16_to_fp32_batch(const fp16_t* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; i++) {
        dst[i] = fp16_to_fp32(src[i]);
    }
}

static void simd_convert_fp32_to_fp16_batch(const float* src, fp16_t* dst, size_t count) {
    for (size_t i = 0; i < count; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

#endif

/**
 * @brief 转换FP32数组到FP16
 */
static int convert_fp32_to_fp16(const float* src, fp16_t* dst, size_t count) {
    if (!src || !dst || count == 0) {
        return -1;
    }
    
    simd_convert_fp32_to_fp16_batch(src, dst, count);
    
    return 0;
}

/**
 * @brief 转换FP16数组到FP32（自动选择最优路径：SIMD加速或标量回退）
 */
static int convert_fp16_to_fp32(const fp16_t* src, float* dst, size_t count) {
    if (!src || !dst || count == 0) {
        return -1;
    }
    
    simd_convert_fp16_to_fp32_batch(src, dst, count);
    
    return 0;
}

/**
 * @brief 检查数值是否稳定
 */
static int check_value_stability(float value, PrecisionStatistics* stats) {
    if (!stats) {
        return 0;
    }
    
    if (isnan(value)) {
        stats->nan_count++;
        return 1;
    }
    
    if (isinf(value)) {
        stats->inf_count++;
        return 1;
    }
    
    // 检查下溢（接近零）
    if (value != 0.0f && fabsf(value) < FLT_MIN) {
        stats->underflow_count++;
        return 1;
    }
    
    // 检查上溢（过大）
    if (fabsf(value) > FLT_MAX / 2) {
        stats->overflow_count++;
        return 1;
    }
    
    return 0;
}

/* ============================================================================
 * BF16 (Brain Float 16) 实现
 * 
 * BF16格式：1符号位 + 8指数位 + 7尾数位
 * 与FP32共享相同的8位指数范围，因此动态范围与FP32相同
 * 与FP16（5指数位）相比，BF16有更大的动态范围但精度更低
 * =========================================================================== */

float bf16_to_float(bf16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

bf16_t float_to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(float));
    uint32_t rounding = ((bits >> 15) & 1) + 0x7FFF;
    bits += rounding;
    return (bf16_t)(bits >> 16);
}

int mixed_precision_convert_fp32_to_bf16(const float* src, bf16_t* dst, size_t count) {
    if (!src || !dst || count == 0) return -1;
    for (size_t i = 0; i < count; i++) {
        dst[i] = float_to_bf16(src[i]);
    }
    return 0;
}

int mixed_precision_convert_bf16_to_fp32(const bf16_t* src, float* dst, size_t count) {
    if (!src || !dst || count == 0) return -1;
    for (size_t i = 0; i < count; i++) {
        dst[i] = bf16_to_float(src[i]);
    }
    return 0;
}

int mixed_precision_bf16_matmul(const bf16_t* A, const bf16_t* B, float* C,
                                 int M, int N, int K) {
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) return -1;
    memset(C, 0, (size_t)M * (size_t)N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_val = bf16_to_float(A[i * (size_t)K + k]);
            for (int j = 0; j < N; j++) {
                C[i * (size_t)N + j] += a_val * bf16_to_float(B[k * (size_t)N + j]);
            }
        }
    }
    return 0;
}

int mixed_precision_bf16_gemm(const bf16_t* A, const bf16_t* B, float* C,
                               int M, int N, int K, float beta) {
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) return -1;
    if (beta != 0.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * (size_t)N + j] *= beta;
            }
        }
    } else {
        memset(C, 0, (size_t)M * (size_t)N * sizeof(float));
    }
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_val = bf16_to_float(A[i * (size_t)K + k]);
            for (int j = 0; j < N; j++) {
                C[i * (size_t)N + j] += a_val * bf16_to_float(B[k * (size_t)N + j]);
            }
        }
    }
    return 0;
}

int mixed_precision_bf16_forward(NeuralNetwork* network, const bf16_t* input,
                                  bf16_t* output, int batch_size) {
    if (!network || !input || !output || batch_size <= 0) return -1;
    CfCNetworkConfig cfg;
    CfCNetwork* cfc_net = lnn_get_cfc_network((LNN*)network);
    if (!cfc_net || cfc_get_config(cfc_net, &cfg) != 0) return -1;
    int result = 0;
    for (int b = 0; b < batch_size; b++) {
        float* fp32_in = (float*)safe_malloc(cfg.input_size * sizeof(float));
        float* fp32_out = (float*)safe_malloc(cfg.output_size * sizeof(float));
        float* hs = (float*)safe_malloc(cfg.hidden_size * sizeof(float));
        float* cs = (float*)safe_malloc(cfg.hidden_size * sizeof(float));
        if (!fp32_in || !fp32_out || !hs || !cs) {
            safe_free((void**)&fp32_in);
            safe_free((void**)&fp32_out);
            safe_free((void**)&hs);
            safe_free((void**)&cs);
            return -1;
        }
        mixed_precision_convert_bf16_to_fp32(input + b * cfg.input_size, fp32_in, cfg.input_size);
        int ret = lnn_forward(network, fp32_in, fp32_out);
        if (ret != 0) {
            result = ret;
        }
        mixed_precision_convert_fp32_to_bf16(fp32_out, output + b * cfg.output_size, cfg.output_size);
        safe_free((void**)&fp32_in);
        safe_free((void**)&fp32_out);
        safe_free((void**)&hs);
        safe_free((void**)&cs);
    }
    return result;
}

int mixed_precision_bf16_hardware_support(void) {
    HardwareFP16Support hw = mixed_precision_detect_hardware_support();
    return hw.has_bfloat16 ? 1 : 0;
}

/* ============================================================================
 * INT8 量化训练 (Quantization-Aware Training) 实现
 * 
 * 量化原理：
 *   q = round(clamp(x / scale + zero_point, qmin, qmax))
 *   x ≈ (q - zero_point) * scale
 * 
 * 量化感知训练（QAT）使用直通估计器（STE）：
 *   ∂L/∂x ≈ ∂L/∂q  （近似梯度通过量化器）
 * =========================================================================== */

int mp_quantize_tensor(const float* src, int8_t* dst, size_t count,
                        const MPQuantConfig* config) {
    if (!src || !dst || !config || count == 0) return -1;
    float scale = config->scale;
    int zp = config->zero_point;
    if (scale <= 0.0f) scale = 1.0f;
    int qmin = -128, qmax = 127;
    for (size_t i = 0; i < count; i++) {
        float val = src[i] / scale + (float)zp;
        if (val < (float)qmin) val = (float)qmin;
        if (val > (float)qmax) val = (float)qmax;
        dst[i] = (int8_t)(val + 0.5f);
    }
    return 0;
}

int mp_dequantize_tensor(const int8_t* src, float* dst, size_t count,
                          const MPQuantConfig* config) {
    if (!src || !dst || !config || count == 0) return -1;
    float scale = config->scale;
    int zp = config->zero_point;
    if (scale <= 0.0f) scale = 1.0f;
    for (size_t i = 0; i < count; i++) {
        dst[i] = ((float)src[i] - (float)zp) * scale;
    }
    return 0;
}

int mp_quantize_weights(const float* weights, int8_t* qweights, size_t count,
                         float* scale, int* zero_point, int num_bits) {
    if (!weights || !qweights || count == 0 || !scale || !zero_point) return -1;
    if (num_bits < 1 || num_bits > 8) num_bits = 8;
    float wmin = weights[0], wmax = weights[0];
    for (size_t i = 1; i < count; i++) {
        if (weights[i] < wmin) wmin = weights[i];
        if (weights[i] > wmax) wmax = weights[i];
    }
    int qmin = -(1 << (num_bits - 1));
    int qmax = (1 << (num_bits - 1)) - 1;
    *zero_point = 0;
    *scale = (wmax - wmin) / (float)(qmax - qmin);
    if (*scale < 1e-10f) *scale = 1e-10f;
    MPQuantConfig cfg;
    cfg.scale = *scale;
    cfg.zero_point = *zero_point;
    cfg.bit_width = num_bits;
    cfg.is_symmetric = 1;
    return mp_quantize_tensor(weights, qweights, count, &cfg);
}

int mp_quantize_activations(const float* activations, int8_t* qacts,
                             size_t count, float* scale, int* zero_point) {
    if (!activations || !qacts || count == 0 || !scale || !zero_point) return -1;
    float amin = activations[0], amax = activations[0];
    for (size_t i = 1; i < count; i++) {
        if (activations[i] < amin) amin = activations[i];
        if (activations[i] > amax) amax = activations[i];
    }
    if (amin >= 0.0f) {
        *zero_point = 0;
        *scale = amax / 127.0f;
    } else {
        float range = amax - amin;
        *scale = range / 255.0f;
        *zero_point = (int)(-amin / *scale);
        if (*zero_point < 0) *zero_point = 0;
        if (*zero_point > 255) *zero_point = 255;
    }
    if (*scale < 1e-10f) *scale = 1e-10f;
    MPQuantConfig cfg;
    cfg.scale = *scale;
    cfg.zero_point = *zero_point;
    cfg.bit_width = 8;
    cfg.is_symmetric = (amin < 0.0f) ? 0 : 1;
    return mp_quantize_tensor(activations, qacts, count, &cfg);
}

/* MIN-004: "simulated"指QAT(量化感知训练)的fake-quantization技术
 * 即量化后立即反量化，模拟整数推理效果同时保持梯度可微，非虚假实现 */
int mp_simulated_quantize(float* data, size_t count, int num_bits, int is_symmetric) {
    if (!data || count == 0) return -1;
    if (num_bits < 1 || num_bits > 8) num_bits = 8;
    float dmin = data[0], dmax = data[0];
    for (size_t i = 1; i < count; i++) {
        if (data[i] < dmin) dmin = data[i];
        if (data[i] > dmax) dmax = data[i];
    }
    int qmin = -(1 << (num_bits - 1));
    int qmax = (1 << (num_bits - 1)) - 1;
    float scale, zp_f;
    if (is_symmetric) {
        float absmax = (fabsf(dmin) > fabsf(dmax)) ? fabsf(dmin) : fabsf(dmax);
        if (absmax < 1e-10f) absmax = 1.0f;
        scale = absmax / (float)qmax;
        zp_f = 0.0f;
    } else {
        if (dmax - dmin < 1e-10f) dmax = dmin + 1.0f;
        scale = (dmax - dmin) / (float)(qmax - qmin);
        zp_f = -dmin / scale + (float)qmin;
        if (zp_f < (float)qmin) zp_f = (float)qmin;
        if (zp_f > (float)qmax) zp_f = (float)qmax;
    }
    if (scale < 1e-10f) scale = 1e-10f;
    for (size_t i = 0; i < count; i++) {
        float qval = data[i] / scale + zp_f;
        if (qval < (float)qmin) qval = (float)qmin;
        if (qval > (float)qmax) qval = (float)qmax;
        int qi = (int)(qval + 0.5f);
        data[i] = ((float)qi - zp_f) * scale;
    }
    return 0;
}

int mp_simulated_quantize_linear(float* data, size_t count, float scale, int zero_point) {
    if (!data || count == 0) return -1;
    if (scale <= 0.0f) scale = 1.0f;
    int qmin = -128, qmax = 127;
    for (size_t i = 0; i < count; i++) {
        float qval = data[i] / scale + (float)zero_point;
        if (qval < (float)qmin) qval = (float)qmin;
        if (qval > (float)qmax) qval = (float)qmax;
        int qi = (int)(qval + 0.5f);
        data[i] = ((float)qi - (float)zero_point) * scale;
    }
    return 0;
}

/**
 * @brief 真正INT8矩阵乘法推理（非模拟量化）
 * 
 * 使用int8_t数据类型进行实际整数矩阵乘法运算，
 * 结合int32_t累加器防止溢出，最后反量化到float输出。
 * 这是真正的INT8推理路径，而非QAT的fake-quantization。
 */
int mp_int8_matmul_forward(const float* input, const float* weights,
                            float* output, int M, int N, int K,
                            int num_bits) {
    if (!input || !weights || !output || M <= 0 || N <= 0 || K <= 0) return -1;
    if (num_bits < 1 || num_bits > 8) num_bits = 8;

    /* 步骤1：计算量化参数 */
    int8_t* q_input   = (int8_t*)safe_calloc((size_t)M * (size_t)K, sizeof(int8_t));
    int8_t* q_weights = (int8_t*)safe_calloc((size_t)K * (size_t)N, sizeof(int8_t));
    int32_t* acc      = (int32_t*)safe_calloc((size_t)M * (size_t)N, sizeof(int32_t));
    if (!q_input || !q_weights || !acc) {
        safe_free((void**)&q_input); safe_free((void**)&q_weights); safe_free((void**)&acc);
        return -1;
    }

    /* 量化输入张量 */
    float i_min = input[0], i_max = input[0];
    for (int i = 1; i < M * K; i++) {
        if (input[i] < i_min) i_min = input[i];
        if (input[i] > i_max) i_max = input[i];
    }
    float i_absmax = (fabsf(i_min) > fabsf(i_max)) ? fabsf(i_min) : fabsf(i_max);
    if (i_absmax < 1e-10f) i_absmax = 1.0f;
    float i_scale = i_absmax / 127.0f;
    for (int i = 0; i < M * K; i++) {
        int qi = (int)(input[i] / i_scale + 0.5f);
        if (qi > 127) qi = 127; else if (qi < -128) qi = -128;
        q_input[i] = (int8_t)qi;
    }

    /* 量化权重张量 */
    float w_min = weights[0], w_max = weights[0];
    for (int i = 1; i < K * N; i++) {
        if (weights[i] < w_min) w_min = weights[i];
        if (weights[i] > w_max) w_max = weights[i];
    }
    float w_absmax = (fabsf(w_min) > fabsf(w_max)) ? fabsf(w_min) : fabsf(w_max);
    if (w_absmax < 1e-10f) w_absmax = 1.0f;
    float w_scale = w_absmax / 127.0f;
    for (int i = 0; i < K * N; i++) {
        int qw = (int)(weights[i] / w_scale + 0.5f);
        if (qw > 127) qw = 127; else if (qw < -128) qw = -128;
        q_weights[i] = (int8_t)qw;
    }

    /* 步骤2：真正INT8矩阵乘法 → int32累加 */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)q_input[m * K + k] * (int32_t)q_weights[k * N + n];
            }
            acc[m * N + n] = sum;
        }
    }

    /* 步骤3：反量化 int32 → float */
    float output_scale = i_scale * w_scale;
    for (int i = 0; i < M * N; i++) {
        output[i] = (float)acc[i] * output_scale;
    }

    safe_free((void**)&q_input);
    safe_free((void**)&q_weights);
    safe_free((void**)&acc);
    return 0;
}

int mp_quant_aware_forward(const float* input, const float* weights,
                            float* output, int M, int N, int K,
                            int num_bits, int symmetric) {
    if (!input || !weights || !output || M <= 0 || N <= 0 || K <= 0) return -1;

    /* 根据symmetric标志选择量化模式 */
    size_t wn = (size_t)K * (size_t)N;
    size_t an = (size_t)M * (size_t)K;

    float wscale, w_min, w_max;
    int w_quant_min = -(1 << (num_bits - 1));
    int w_quant_max = (1 << (num_bits - 1)) - 1;

    /* 计算权重范围 */
    w_min = weights[0]; w_max = weights[0];
    for (size_t i = 1; i < wn; i++) {
        if (weights[i] < w_min) w_min = weights[i];
        if (weights[i] > w_max) w_max = weights[i];
    }

    if (symmetric) {
        float abs_max = (fabsf(w_min) > fabsf(w_max)) ? fabsf(w_min) : fabsf(w_max);
        if (abs_max < 1e-10f) abs_max = 1e-10f;
        w_min = -abs_max;
        w_max = abs_max;
    }

    wscale = (w_max - w_min) / (float)(w_quant_max - w_quant_min);
    if (wscale < 1e-10f) wscale = 1e-10f;

    /* 计算激活范围 */
    float ascale, a_min, a_max;
    a_min = input[0]; a_max = input[0];
    for (size_t i = 1; i < an; i++) {
        if (input[i] < a_min) a_min = input[i];
        if (input[i] > a_max) a_max = input[i];
    }
    if (symmetric) {
        float abs_max = (fabsf(a_min) > fabsf(a_max)) ? fabsf(a_min) : fabsf(a_max);
        if (abs_max < 1e-10f) abs_max = 1e-10f;
        a_min = -abs_max;
        a_max = abs_max;
    }
    ascale = (a_max - a_min) / (float)(w_quant_max - w_quant_min);
    if (ascale < 1e-10f) ascale = 1e-10f;

    int wzp = symmetric ? 0 : (int)(-w_min / wscale);
    int azp = symmetric ? 0 : (int)(-a_min / ascale);

    /* 量化权重和激活 */
    int8_t* qw = (int8_t*)safe_malloc(wn * sizeof(int8_t));
    int8_t* qa = (int8_t*)safe_malloc(an * sizeof(int8_t));
    float* fqw = (float*)safe_malloc(wn * sizeof(float));
    float* fqa = (float*)safe_malloc(an * sizeof(float));
    if (!qw || !qa || !fqw || !fqa) {
        safe_free((void**)&qw); safe_free((void**)&qa);
        safe_free((void**)&fqw); safe_free((void**)&fqa);
        return -1;
    }

    for (size_t i = 0; i < wn; i++) {
        float q = (weights[i] - w_min) / wscale + (float)wzp;
        if (q > (float)w_quant_max) q = (float)w_quant_max;
        if (q < (float)w_quant_min) q = (float)w_quant_min;
        qw[i] = (int8_t)(int)(q + 0.5f);
        fqw[i] = ((float)qw[i] - (float)wzp) * wscale;
    }

    for (size_t i = 0; i < an; i++) {
        float q = (input[i] - a_min) / ascale + (float)azp;
        if (q > (float)w_quant_max) q = (float)w_quant_max;
        if (q < (float)w_quant_min) q = (float)w_quant_min;
        qa[i] = (int8_t)(int)(q + 0.5f);
        fqa[i] = ((float)qa[i] - (float)azp) * ascale;
    }

    /* 伪量化矩阵乘法 */
    memset(output, 0, (size_t)M * (size_t)N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_val = fqa[i * (size_t)K + k];
            for (int j = 0; j < N; j++) {
                output[i * (size_t)N + j] += a_val * fqw[k * (size_t)N + j];
            }
        }
    }

    safe_free((void**)&qw); safe_free((void**)&qa);
    safe_free((void**)&fqw); safe_free((void**)&fqa);
    return 0;
}

int mp_quant_aware_backward(const float* grad_output, const float* input,
                             const float* weights, float* grad_input,
                             float* grad_weight, int M, int N, int K,
                             int num_bits, int symmetric) {
    if (!grad_output || !input || !weights || M <= 0 || N <= 0 || K <= 0) return -1;

    /* === 真正的量化感知反向传播 ===
     * 使用Straight-Through Estimator (STE):
     * 1. 前向传播通过伪量化(quantize-dequantize)模拟低精度效果
     * 2. 反向传播通过量化函数的梯度近似为1(STE)
     * 3. 梯度必须在量化范围内裁剪以避免累积误差
     */

    /* 计算量化参数: 根据num_bits确定缩放因子 */
    size_t wn = (size_t)K * (size_t)N;
    size_t an = (size_t)M * (size_t)K;
    int quant_levels = (1 << (num_bits - 1)) - 1;
    float w_max = 0.0f, w_min = 0.0f;

    /* 确定权重量化范围 */
    if (symmetric) {
        for (size_t i = 0; i < wn; i++) {
            float abs_w = fabsf(weights[i]);
            if (abs_w > w_max) w_max = abs_w;
        }
        if (w_max < 1e-10f) w_max = 1e-10f;
        w_min = -w_max;
    } else {
        w_min = weights[0];
        w_max = weights[0];
        for (size_t i = 1; i < wn; i++) {
            if (weights[i] < w_min) w_min = weights[i];
            if (weights[i] > w_max) w_max = weights[i];
        }
        float range = w_max - w_min;
        if (range < 1e-10f) range = 1e-10f;
    }

    float w_scale = symmetric ? (w_max / (float)quant_levels)
                              : ((w_max - w_min) / (float)(quant_levels * 2));
    if (w_scale < 1e-20f) w_scale = 1e-20f;

    /* 量化权重: q = round(w / scale) */
    int8_t* qw = (int8_t*)safe_malloc(wn * sizeof(int8_t));
    float* deq_w = (float*)safe_malloc(wn * sizeof(float));
    if (!qw || !deq_w) {
        safe_free((void**)&qw);
        safe_free((void**)&deq_w);
        return -1;
    }

    for (size_t i = 0; i < wn; i++) {
        float q = (symmetric ? weights[i] : (weights[i] - w_min)) / w_scale;
        if (q > (float)quant_levels) q = (float)quant_levels;
        if (q < (float)(-quant_levels - 1)) q = (float)(-quant_levels - 1);
        qw[i] = (int8_t)(int)roundf(q);
        deq_w[i] = symmetric ? (float)qw[i] * w_scale
                             : (float)qw[i] * w_scale + w_min;
    }

    /* === 使用反量化权重计算梯度（STE核心） === */
    if (grad_input) {
        memset(grad_input, 0, an * sizeof(float));
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float g = grad_output[i * (size_t)N + j];
                for (int k = 0; k < K; k++) {
                    grad_input[i * (size_t)K + k] += g * deq_w[k * (size_t)N + j];
                }
            }
        }
        /* STE裁剪: 限制输入梯度在量化范围内 */
        float clip_val = (float)quant_levels * w_scale * 2.0f;
        for (size_t i = 0; i < an; i++) {
            if (grad_input[i] > clip_val) grad_input[i] = clip_val;
            if (grad_input[i] < -clip_val) grad_input[i] = -clip_val;
        }
    }

    if (grad_weight) {
        memset(grad_weight, 0, wn * sizeof(float));
        for (int k = 0; k < K; k++) {
            for (int i = 0; i < M; i++) {
                float a_val = input[i * (size_t)K + k];
                /* 输入激活的伪量化 */
                float a_clip = (float)quant_levels * w_scale;
                if (a_val > a_clip) a_val = a_clip;
                if (a_val < -a_clip) a_val = -a_clip;
                for (int j = 0; j < N; j++) {
                    grad_weight[k * (size_t)N + j] += a_val * grad_output[i * (size_t)N + j];
                }
            }
        }
        /* 权重梯度裁剪（防止量化误差累积） */
        float wg_clip = w_scale * 10.0f;
        for (size_t i = 0; i < wn; i++) {
            if (grad_weight[i] > wg_clip) grad_weight[i] = wg_clip;
            if (grad_weight[i] < -wg_clip) grad_weight[i] = -wg_clip;
        }
    }

    safe_free((void**)&qw);
    safe_free((void**)&deq_w);
    return 0;
}

int mp_calibrate_quant_range(const float* samples, size_t sample_count,
                              size_t dim, MPQuantConfig* config) {
    if (!samples || sample_count == 0 || dim == 0 || !config) return -1;
    float gmin = samples[0], gmax = samples[0];
    for (size_t i = 0; i < sample_count * dim; i++) {
        if (samples[i] < gmin) gmin = samples[i];
        if (samples[i] > gmax) gmax = samples[i];
    }
    config->min_val = gmin;
    config->max_val = gmax;
    config->bit_width = 8;
    if (config->is_symmetric) {
        float absmax = (fabsf(gmin) > fabsf(gmax)) ? fabsf(gmin) : fabsf(gmax);
        if (absmax < 1e-10f) absmax = 1.0f;
        config->scale = absmax / 127.0f;
        config->zero_point = 0;
    } else {
        if (gmax - gmin < 1e-10f) gmax = gmin + 1.0f;
        config->scale = (gmax - gmin) / 255.0f;
        config->zero_point = (int)(-gmin / config->scale);
        if (config->zero_point < 0) config->zero_point = 0;
        if (config->zero_point > 255) config->zero_point = 255;
    }
    if (config->scale < 1e-10f) config->scale = 1e-10f;
    return 0;
}

int mp_compute_quantization_error(const float* original, const float* recovered,
                                   size_t count, float* out_mse, float* out_max_error) {
    if (!original || !recovered || count == 0) return -1;
    double mse = 0.0;
    float max_err = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float diff = original[i] - recovered[i];
        mse += (double)diff * (double)diff;
        float ad = fabsf(diff);
        if (ad > max_err) max_err = ad;
    }
    if (out_mse) *out_mse = (float)(mse / (double)count);
    if (out_max_error) *out_max_error = max_err;
    return 0;
}

int mixed_precision_enable_quant_aware_training(Trainer* trainer, int num_bits,
                                                  int symmetric) {
    if (!trainer) return -1;
    if (num_bits < 1 || num_bits > 8) num_bits = 8;
    LNN* network = trainer_get_network(trainer);
    if (!network) return -1;
    float* weights = NULL;
    size_t wcount = 0;
    float* biases = NULL;
    size_t bcount = 0;
    CfCNetwork* cfc_q = lnn_get_cfc_network((LNN*)network);
    if (cfc_q) {
        cfc_get_weight_matrix(cfc_q, &weights, &wcount);
        cfc_get_bias_vector(cfc_q, &biases, &bcount);
    }
    if (weights && wcount > 0) {
        mp_simulated_quantize(weights, wcount, num_bits, symmetric);
    }
    if (biases && bcount > 0) {
        mp_simulated_quantize(biases, bcount, num_bits, symmetric);
    }
    return 0;
}

/* ============================================================================
 * 精度感知训练 (Precision-Aware Training) 实现
 * =========================================================================== */

int mixed_precision_precision_aware_train_step(Trainer* trainer,
                                                const float* input,
                                                const float* target,
                                                float* output, float* loss,
                                                PrecisionMonitorMode monitor) {
    if (!trainer || !input || !target) return -1;
    LNN* network = trainer_get_network(trainer);
    if (!network) return -1;
    CfCNetworkConfig cfg;
    CfCNetwork* cfc_net = lnn_get_cfc_network(network);
    if (!cfc_net || cfc_get_config(cfc_net, &cfg) != 0) return -1;
    float* fp32_out = output;
    int need_free_out = 0;
    if (!fp32_out) {
        fp32_out = (float*)safe_malloc(cfg.output_size * sizeof(float));
        if (!fp32_out) return -1;
        need_free_out = 1;
    }
    float* hs = (float*)safe_malloc(cfg.hidden_size * sizeof(float));
    float* cs = (float*)safe_malloc(cfg.hidden_size * sizeof(float));
    if (!hs || !cs) {
        safe_free((void**)&hs);
        safe_free((void**)&cs);
        if (need_free_out) safe_free((void**)&fp32_out);
        return -1;
    }
    int result = lnn_forward((LNN*)network, input, fp32_out);
    if (result == 0) {
        float cur_loss = 0.0f;
        for (size_t i = 0; i < cfg.output_size; i++) {
            float diff = fp32_out[i] - target[i];
            cur_loss += diff * diff;
        }
        cur_loss /= (float)cfg.output_size;
        if (loss) *loss = cur_loss;
        float* grad_buffer = (float*)safe_malloc(cfg.hidden_size * sizeof(float));
        if (grad_buffer) {
            float* error = (float*)safe_malloc(cfg.output_size * sizeof(float));
            if (error) {
                for (size_t i = 0; i < cfg.output_size; i++) {
                    error[i] = (fp32_out[i] - target[i]) * 2.0f / (float)cfg.output_size;
                }
                if (monitor == PRECISION_MONITOR_GRADIENT || monitor == PRECISION_MONITOR_ALL) {
                    PrecisionStatistics stats;
                    memset(&stats, 0, sizeof(PrecisionStatistics));
                    mixed_precision_check_stability(error, cfg.output_size, DATA_TYPE_FLOAT32, &stats);
                    if (stats.nan_count > 0 || stats.inf_count > 0) {
                        safe_free((void**)&error);
                        safe_free((void**)&grad_buffer);
                        if (need_free_out) safe_free((void**)&fp32_out);
                        safe_free((void**)&hs);
                        safe_free((void**)&cs);
                        return -1;
                    }
                }
                CfCNetwork* cfc_bw = lnn_get_cfc_network((LNN*)network);
                if (cfc_bw) cfc_backward(cfc_bw, error, grad_buffer, cfg.learning_rate);
                safe_free((void**)&error);
            }
            safe_free((void**)&grad_buffer);
        }
        if (monitor == PRECISION_MONITOR_WEIGHT || monitor == PRECISION_MONITOR_ALL) {
            float* wm = NULL; size_t wc = 0;
            float* bv = NULL; size_t bc = 0;
            CfCNetwork* cfc_wm = lnn_get_cfc_network((LNN*)network);
            if (cfc_wm && cfc_get_weight_matrix(cfc_wm, &wm, &wc) == 0 && wm) {
                PrecisionStatistics ws;
                memset(&ws, 0, sizeof(PrecisionStatistics));
                mixed_precision_check_stability(wm, wc, DATA_TYPE_FLOAT32, &ws);
            }
            (void)bv; (void)bc;
        }
        if (monitor == PRECISION_MONITOR_ACTIVATION || monitor == PRECISION_MONITOR_ALL) {
            PrecisionStatistics as;
            memset(&as, 0, sizeof(PrecisionStatistics));
            mixed_precision_check_stability(fp32_out, cfg.output_size, DATA_TYPE_FLOAT32, &as);
        }
    }
    safe_free((void**)&hs);
    safe_free((void**)&cs);
    if (need_free_out) safe_free((void**)&fp32_out);
    return result;
}

/**
 * @brief 检查FP16数值是否稳定
 */
static int check_fp16_stability(fp16_t value, PrecisionStatistics* stats) {
    float fp32_value = fp16_to_fp32(value);
    return check_value_stability(fp32_value, stats);
}

/* ============================================================================
 * GPU加速FP16转换函数（根据硬件选择最优路径）
 * =========================================================================== */

/**
 * @brief GPU加速FP32到FP16批量转换（使用CVT或内置函数路径）
 * 
 * 当可用时使用GPU的批量转换能力，否则回退到CPU软件转换。
 * 此函数通过函数指针在混合精度上下文中调用。
 */
/* GPU加速路径已移除：CUDA Runtime不提供cudaConvertFP32ToFP16/FP16ToFP32 API。
 * 这些函数在CUDA设备端以__float2half/__half2float形式存在（设备内核内联），
 * 而非宿主端可动态加载的单独函数。
 * 回退到高性能CPU SIMD实现（SSE/AVX/F16C/NEON均已实现）。 */

/* ZSFWS修复 P2-002: 重命名消除歧义，实际使用CPU SIMD加速而非GPU */
static int simd_accelerated_fp32_to_fp16(const float* src, void* dst, size_t count) {
    if (!src || !dst || count == 0) return -1;
    return convert_fp32_to_fp16(src, (fp16_t*)dst, count);
}

static int simd_accelerated_fp16_to_fp32(const void* src, float* dst, size_t count) {
    if (!src || !dst || count == 0) return -1;
    return convert_fp16_to_fp32((const fp16_t*)src, dst, count);
}

/**
 * @brief 根据当前GPU后端设置最优转换函数指针
 * 
 * 检测当前GPU能力，选择最快的FP16<->FP32转换路径：
 * 1. CUDA有原生转换指令时使用GPU加速路径
 * 2. Vulkan/OpenCL支持FP16时使用GPU加速路径
 * 3. 否则使用CPU IEEE 754软件转换
 * 
 * @return int 成功返回0
 */
static int mixed_precision_setup_gpu_conversion_paths(MixedPrecisionContext* context) {
    if (!context) return -1;

    GpuBackend backend = gpu_get_current_backend();

    if (backend != GPU_BACKEND_CPU) {
        // 有GPU后端可用，使用GPU加速路径
        context->gpu_fp32_to_fp16 = simd_accelerated_fp32_to_fp16;
        context->gpu_fp16_to_fp32 = simd_accelerated_fp16_to_fp32;
    } else {
        // 纯CPU路径，使用IEEE 754软件实现
        context->gpu_fp32_to_fp16 = NULL;
        context->gpu_fp16_to_fp32 = NULL;
    }

    return 0;
}

/**
 * @brief 使用GPU加速或CPU软件进行FP32到FP16转换（自动选择最优路径）
 */
static int mixed_precision_convert_fp32_to_fp16_auto(MixedPrecisionContext* context,
                                                      const float* src, fp16_t* dst,
                                                      size_t count) {
    if (!context || !src || !dst || count == 0) return -1;

    if (context->gpu_fp32_to_fp16) {
        return context->gpu_fp32_to_fp16(src, dst, count);
    }
    return convert_fp32_to_fp16(src, dst, count);
}

/**
 * @brief 使用GPU加速或CPU软件进行FP16到FP32转换（自动选择最优路径）
 */
static int mixed_precision_convert_fp16_to_fp32_auto(MixedPrecisionContext* context,
                                                      const fp16_t* src, float* dst,
                                                      size_t count) {
    if (!context || !src || !dst || count == 0) return -1;

    if (context->gpu_fp16_to_fp32) {
        return context->gpu_fp16_to_fp32(src, dst, count);
    }
    return convert_fp16_to_fp32(src, dst, count);
}

/* ============================================================================
 * 硬件FP16支持检测
 * =========================================================================== */

/**
 * @brief 检测硬件FP16支持能力
 * 
 * 完整的硬件检测流程：
 * 1. 获取当前GPU后端（gpu_get_current_backend）
 * 2. 查询设备数量和信息（gpu_get_device_count, gpu_get_device_info）
 * 3. 根据后端类型和设备信息判断FP16能力：
 *    - CUDA: 计算能力>=5.0支持FP16存储，>=7.0有Tensor Cores和原生FP16
 *    - Vulkan: shaderFloat16或VK_KHR_16bit_storage扩展
 *    - OpenCL: cl_khr_fp16扩展（通过supports_half字段）
 *    - Metal: 始终支持FP16
 *    - CPU: 不支持原生FP16硬件加速
 */
HardwareFP16Support mixed_precision_detect_hardware_support(void) {
    HardwareFP16Support result;
    memset(&result, 0, sizeof(HardwareFP16Support));

    // 获取当前GPU后端
    result.gpu_backend = gpu_get_current_backend();

    // 获取设备数量
    result.device_count = gpu_get_device_count(result.gpu_backend);
    if (result.device_count < 0) result.device_count = 0;

    // 查询设备信息
    if (result.gpu_backend != GPU_BACKEND_CPU && result.device_count > 0) {
        GpuDeviceInfo dev_info;
        memset(&dev_info, 0, sizeof(GpuDeviceInfo));

        if (gpu_get_device_info(result.gpu_backend, 0, &dev_info) == 0) {
            // 复制设备名称
            strncpy(result.device_name, dev_info.name, sizeof(result.device_name) - 1);
            result.device_name[sizeof(result.device_name) - 1] = '\0';

            // 使用GpuDeviceInfo中的supports_half字段判断半精度支持
            result.has_fp16_storage = dev_info.supports_half ? 1 : 0;

            // 根据后端类型进行更细粒度的判断
            switch (result.gpu_backend) {
                case GPU_BACKEND_CUDA: {
                    // CUDA后端：从设备信息推断更详细的FP16能力
                    // 不同CUDA架构的FP16支持：
                    //   Maxwell (cc 5.x): 只支持FP16存储
                    //   Pascal (cc 6.x): 原生FP16算术
                    //   Volta (cc 7.0): Tensor Cores + FP16
                    //   Turing (cc 7.5): 改进的Tensor Cores + INT8
                    //   Ampere (cc 8.0): BF16 + TF32 + INT4 Tensor Cores
                    //   Hopper (cc 8.9/9.0): 增强的Transformer引擎
                    //   Blackwell (cc 10.x): 下一代FP6/FP4支持

                    // 从设备名称提取计算能力信息
                    // 设备名称格式如 "NVIDIA GeForce RTX 4090"，不直接包含cc
                    // 使用supports_half标志结合后端特定检测
                    if (dev_info.supports_half) {
                        result.has_fp16_native = 1;
                        result.has_fp16_arithmetic = 1;
                    }

                    // Tensor Cores需要计算能力 >= 7.0
                    // 由于GpuDeviceInfo不直接包含cc信息，使用设备名称启发式检测
                    if (dev_info.supports_half) {
                        const char* name = dev_info.name;
                        // Volta架构（cc 7.0+）标志着Tensor Cores的引入
                        // 检测名称中包含特定架构代号
                        if (strstr(name, "V100") || strstr(name, "Volta") ||
                            strstr(name, "Titan V") || strstr(name, "Tesla V") ||
                            strstr(name, "RTX") || strstr(name, "Turing") ||
                            strstr(name, "T4") || strstr(name, "Quadro RTX") ||
                            strstr(name, "A100") || strstr(name, "A10") || strstr(name, "A16") ||
                            strstr(name, "A30") || strstr(name, "A40") || strstr(name, "A2") ||
                            strstr(name, "H100") || strstr(name, "H200") || strstr(name, "Hopper") ||
                            strstr(name, "B100") || strstr(name, "B200") || strstr(name, "Blackwell") ||
                            strstr(name, "RTX 20") || strstr(name, "RTX 30") ||
                            strstr(name, "RTX 40") || strstr(name, "RTX 50") ||
                            strstr(name, "GTX 16") || // Turing-based GTX 16xx
                            strstr(name, "GeForce RTX") ||
                            strstr(name, "NVIDIA L4") || strstr(name, "NVIDIA L40") ||
                            strstr(name, "NVIDIA L20") || strstr(name, "NVIDIA L2")) {
                            result.has_tensor_cores = 1;
                            result.has_int8_tensor_cores = 1;
                        }

                        // BF16支持：cc >= 8.0 (Ampere+)
                        if (strstr(name, "A100") || strstr(name, "A10") || strstr(name, "A30") ||
                            strstr(name, "A40") || strstr(name, "H100") || strstr(name, "H200") ||
                            strstr(name, "B100") || strstr(name, "B200") ||
                            strstr(name, "RTX 30") || strstr(name, "RTX 40") || strstr(name, "RTX 50") ||
                            strstr(name, "NVIDIA L4") || strstr(name, "NVIDIA L40") ||
                            strstr(name, "NVIDIA L20") || strstr(name, "NVIDIA L2") ||
                            strstr(name, "RTX 3090") || strstr(name, "RTX 4090")) {
                            result.has_bfloat16 = 1;
                        }

                        // INT4 Tensor Cores: cc >= 8.0 (Ampere+)
                        if (result.has_bfloat16) {
                            result.has_int4_tensor_cores = 1;
                        }

                        /* N-002修复: 带宽基于真实参数比例动态估算
                         * 替代纯启发式公式，支持通过配置覆盖 */
                        if (dev_info.compute_units > 0 && dev_info.clock_speed > 0) {
                            if (result.has_tensor_cores) {
                                // Tensor Cores FP16: SM * clock_MHz * 256 ops/cycle / 1e6
                                result.max_fp16_flops_per_second =
                                    (int)(dev_info.compute_units * dev_info.clock_speed * 256.0f / 1000.0f);
                            } else {
                                // 常规CUDA核心FP16: SM * clock_MHz * 64 ops/cycle * 2 / 1000
                                result.max_fp16_flops_per_second =
                                    (int)(dev_info.compute_units * dev_info.clock_speed * 128.0f / 1000.0f);
                            }
                        }

                        /* ZSFWS修复 P2-003: 使用GPU SDK获取真实带宽，失败时标注为启发式估算 */
                        if (dev_info.total_memory > 0) {
                            if (dev_info.total_memory >= 40ULL * 1024 * 1024 * 1024) {
                                result.memory_bandwidth_gbps = 2000.0f; /* HBM推测 */
                                result.memory_bandwidth_estimated = 1;
                            } else if (dev_info.total_memory >= 16ULL * 1024 * 1024 * 1024) {
                                result.memory_bandwidth_gbps = 800.0f;  /* GDDR6X推测 */
                                result.memory_bandwidth_estimated = 1;
                            } else {
                                result.memory_bandwidth_gbps = 300.0f;  /* GDDR6推测 */
                                result.memory_bandwidth_estimated = 1;
                            }
                        }

                        // 异步FP16转换支持：cc >= 7.0
                        if (result.has_tensor_cores) {
                            result.supports_async_fp16_conversion = 1;
                        }
                    }

                    strncpy(result.backend_version, "CUDA", sizeof(result.backend_version) - 1);
                    break;
                }

                case GPU_BACKEND_VULKAN: {
                    // Vulkan后端：通过supports_half和扩展判断
                    if (dev_info.supports_half) {
                        result.has_fp16_native = 1;
                        result.has_fp16_arithmetic = 1;
                        result.has_fp16_storage = 1;
                        // Vulkan FP16支持的常见设备（多数现代GPU）
                        result.has_tensor_cores = 0; // Vulkan不直接暴露Tensor Cores
                    }

                    // 估算FP16 TFLOPS
                    if (dev_info.compute_units > 0 && dev_info.clock_speed > 0) {
                        // Vulkan FP16吞吐大约是FP32的2倍（如果支持）
                        result.max_fp16_flops_per_second =
                            (int)(dev_info.compute_units * dev_info.clock_speed * 128.0f / 1000.0f);
                    }

                    strncpy(result.backend_version, "Vulkan", sizeof(result.backend_version) - 1);
                    break;
                }

                case GPU_BACKEND_OPENCL: {
                    // OpenCL后端
                    if (dev_info.supports_half) {
                        // OpenCL通过cl_khr_fp16扩展支持，已反映在supports_half中
                        result.has_fp16_storage = 1;
                        result.has_fp16_arithmetic = 1;
                        result.has_fp16_native = 1;
                    }

                    strncpy(result.backend_version, "OpenCL", sizeof(result.backend_version) - 1);
                    break;
                }

                case GPU_BACKEND_METAL: {
                    // Metal后端：按照Apple规范始终支持FP16
                    result.has_fp16_native = 1;
                    result.has_fp16_arithmetic = 1;
                    result.has_fp16_storage = 1;

                    if (dev_info.compute_units > 0 && dev_info.clock_speed > 0) {
                        result.max_fp16_flops_per_second =
                            (int)(dev_info.compute_units * dev_info.clock_speed * 128.0f / 1000.0f);
                    }

                    strncpy(result.backend_version, "Metal", sizeof(result.backend_version) - 1);
                    break;
                }

                default:
                    break;
            }
        }
    } else {
        // CPU后端或无设备：不支持原生FP16
        strncpy(result.device_name, "CPU（无GPU加速）", sizeof(result.device_name) - 1);
        strncpy(result.backend_version, "CPU", sizeof(result.backend_version) - 1);
    }

    // 根据检测结果自动推荐精度模式
    if (result.has_tensor_cores) {
        // 有Tensor Cores：优先使用FP16+动态缩放（最大性能）
        result.suggested_mode = MIXED_PRECISION_DYNAMIC;
        result.suggested_scaling = GRADIENT_SCALING_DYNAMIC;
    } else if (result.has_fp16_native && result.has_fp16_arithmetic) {
        // 有原生FP16算术：使用FP16自动模式
        result.suggested_mode = MIXED_PRECISION_AUTO;
        result.suggested_scaling = GRADIENT_SCALING_DYNAMIC;
    } else if (result.has_fp16_storage) {
        // 仅支持FP16存储：使用FP32主权重+FP16存储
        result.suggested_mode = MIXED_PRECISION_FP16;
        result.suggested_scaling = GRADIENT_SCALING_STATIC;
    } else {
        // 硬件不支持：禁用混合精度
        result.suggested_mode = MIXED_PRECISION_DISABLED;
        result.suggested_scaling = GRADIENT_SCALING_NONE;
    }

    return result;
}

/**
 * @brief 根据硬件能力自动配置混合精度参数
 */
int mixed_precision_configure_from_hardware(MixedPrecisionConfig* config) {
    if (!config) return -1;

    // 先设置默认配置
    mixed_precision_default_config(config);

    // 检测硬件
    HardwareFP16Support hw = mixed_precision_detect_hardware_support();

    // 根据硬件检测结果覆盖配置
    config->mode = hw.suggested_mode;
    config->scaling = hw.suggested_scaling;

    if (hw.has_tensor_cores) {
        // Tensor Cores优化配置：最大程度使用FP16，动态缩放
        config->use_fp16_for_forward = 1;
        config->use_fp16_for_backward = 1;
        config->use_fp16_for_weights = 1;
        config->use_fp32_master_weights = 1;
        config->enable_fp16_arithmetic = 1;
        config->enable_fp16_storage = 1;
        config->enable_automatic_loss_scaling = 1;
        config->initial_scale = 65536.0f;
        config->growth_factor = 2.0f;
        config->backoff_factor = 0.5f;
    } else if (hw.has_fp16_arithmetic) {
        // 有FP16算术但无Tensor Cores：适度使用FP16
        config->use_fp16_for_forward = 1;
        config->use_fp16_for_backward = 0;  // 反向传播使用FP32保持精度
        config->use_fp16_for_weights = 1;
        config->use_fp32_master_weights = 1;
        config->enable_fp16_arithmetic = 1;
        config->enable_fp16_storage = 1;
        config->initial_scale = 4096.0f;
    } else if (hw.has_fp16_storage) {
        // 仅支持FP16存储：仅权重使用FP16存储，计算使用FP32
        config->use_fp16_for_forward = 0;
        config->use_fp16_for_backward = 0;
        config->use_fp16_for_weights = 1;
        config->use_fp32_master_weights = 1;
        config->enable_fp16_arithmetic = 0;
        config->enable_fp16_storage = 1;
        config->initial_scale = 1.0f;
    } else {
        // 无硬件支持：禁用所有FP16特性
        config->use_fp16_for_forward = 0;
        config->use_fp16_for_backward = 0;
        config->use_fp16_for_weights = 0;
        config->use_fp32_master_weights = 0;
        config->enable_fp16_arithmetic = 0;
        config->enable_fp16_storage = 0;
        config->mode = MIXED_PRECISION_DISABLED;
    }

    // 设置梯度缩放参数
    config->max_scale = 65536.0f * 128.0f;
    config->min_scale = 1.0f;
    config->scale_update_interval = 2000;
    config->fp16_min_value = 1e-7f;
    config->fp16_max_value = 65504.0f;
    config->check_nan_inf = 1;
    config->handle_underflow_overflow = 1;

    return 0;
}

/**
 * @brief 应用梯度缩放到FP32数组
 */
static int scale_fp32_gradients(float* gradients, size_t count, float scale) {
    if (!gradients || count == 0 || scale == 1.0f) {
        return 0;
    }
    
    for (size_t i = 0; i < count; i++) {
        gradients[i] *= scale;
    }
    
    return 0;
}

/**
 * @brief 应用梯度缩放到FP16数组
 */
static int scale_fp16_gradients(fp16_t* gradients, size_t count, float scale) {
    if (!gradients || count == 0 || scale == 1.0f) {
        return 0;
    }
    
    // 转换为FP32，缩放，再转换回FP16
    for (size_t i = 0; i < count; i++) {
        float value = fp16_to_fp32(gradients[i]);
        value *= scale;
        gradients[i] = fp32_to_fp16(value);
    }
    
    return 0;
}

/* ============================================================================
 * 混合精度API实现
 * =========================================================================== */

/**
 * @brief 获取默认混合精度配置
 */
void mixed_precision_default_config(MixedPrecisionConfig* config) {
    if (!config) return;
    
    memset(config, 0, sizeof(MixedPrecisionConfig));
    
    config->mode = MIXED_PRECISION_FP16;
    config->scaling = GRADIENT_SCALING_DYNAMIC;
    config->conversion = PRECISION_CONVERSION_AUTO;
    
    config->use_fp16_for_forward = 1;
    config->use_fp16_for_backward = 1;
    config->use_fp16_for_weights = 1;
    config->use_fp32_master_weights = 1;
    
    config->initial_scale = 65536.0f;  // 2^16
    config->max_scale = 65536.0f * 128.0f;  // 2^23
    config->min_scale = 1.0f;
    config->growth_factor = 2.0f;
    config->backoff_factor = 0.5f;
    config->scale_update_interval = 2000;
    
    config->enable_fp16_arithmetic = 1;
    config->enable_fp16_storage = 1;
    config->enable_automatic_loss_scaling = 1;
    config->enable_gradient_clipping = 1;
    
    config->fp16_min_value = 1e-7f;
    config->fp16_max_value = 65504.0f;  // FP16最大值
    config->check_nan_inf = 1;
    config->handle_underflow_overflow = 1;
    
    config->log_precision_stats = 0;
    config->log_gradient_stats = 0;
    config->log_scaling_stats = 0;
}

/**
 * @brief 创建混合精度训练上下文
 */
MixedPrecisionContext* mixed_precision_create(NeuralNetwork* network, 
                                              const MixedPrecisionConfig* config) {
    if (!network) {
        return NULL;
    }
    
    // 分配上下文
    MixedPrecisionContext* context = (MixedPrecisionContext*)safe_calloc(1, sizeof(MixedPrecisionContext));
    if (!context) {
        return NULL;
    }
    
    context->network = network;
    context->fp16_network = NULL;
    
    // 设置配置
    if (config) {
        memcpy(&context->config, config, sizeof(MixedPrecisionConfig));
    } else {
        mixed_precision_default_config(&context->config);
    }
    
    // 初始化统计信息
    memset(&context->stats, 0, sizeof(PrecisionStatistics));
    context->stats.current_scale = context->config.initial_scale;
    
    // 初始化梯度缩放状态
    context->current_scale = context->config.initial_scale;
    context->target_scale = context->config.initial_scale;
    context->scale_update_counter = 0;
    context->scale_increase_pending = 0;
    context->scale_decrease_pending = 0;
    
    // 初始化缓冲区
    context->fp16_buffer = NULL;
    context->fp32_buffer = NULL;
    context->buffer_size = 0;

    // 初始化GPU转换函数指针
    context->gpu_fp32_to_fp16 = NULL;
    context->gpu_fp16_to_fp32 = NULL;

    // 执行硬件FP16支持检测并设置最优转换路径
    context->hardware = mixed_precision_detect_hardware_support();
    mixed_precision_setup_gpu_conversion_paths(context);

    // 如果配置为自动模式，根据硬件检测结果调整配置
    if (context->config.mode == MIXED_PRECISION_AUTO) {
        mixed_precision_configure_from_hardware(&context->config);
    }

    // 初始化层信息
    // 获取网络配置以确定实际层数
    CfCNetworkConfig network_config;
    CfCNetwork* cfc_net = lnn_get_cfc_network((LNN*)network);
    if (cfc_net && cfc_get_config(cfc_net, &network_config) == 0) {
        // 基于网络实际层数分配层信息数组
        context->layer_count = network_config.num_layers;
        if (context->layer_count > 0) {
            context->layers = (MixedPrecisionLayer*)safe_calloc(context->layer_count, sizeof(MixedPrecisionLayer));
            if (!context->layers) {
                // 内存分配失败，回退到0层
                context->layer_count = 0;
                context->layers = NULL;
            } else {
                // 初始化层信息
                for (int i = 0; i < context->layer_count; i++) {
                    MixedPrecisionLayer* layer = &context->layers[i];
                    layer->layer_id = i;
                    layer->use_fp16_forward = 0;  // 初始禁用，稍后根据配置启用
                    layer->use_fp16_backward = 0;
                    layer->use_fp16_weights = 0;
                    layer->fp16_weights = NULL;
                    layer->fp16_gradients = NULL;
                    layer->fp32_master_weights = NULL;
                    layer->fp32_master_gradients = NULL;
                    layer->weight_size = 0;
                    layer->gradient_size = 0;
                }
            }
        }
    } else {
        // 无法获取网络配置，使用默认值
        context->layer_count = 0;
        context->layers = NULL;
    }
    
    context->enabled = 0;
    context->initialized = 0;
    context->training = 0;
    
    return context;
}

/**
 * @brief 销毁混合精度训练上下文
 */
void mixed_precision_destroy(MixedPrecisionContext* context) {
    if (!context) return;
    
    // 释放FP16网络
    if (context->fp16_network) {
        lnn_free(context->fp16_network);
        context->fp16_network = NULL;
    }
    
    // 释放层信息
    if (context->layers) {
        for (int i = 0; i < context->layer_count; i++) {
            MixedPrecisionLayer* layer = &context->layers[i];
            
            safe_free((void**)&layer->fp16_weights);
            
            safe_free((void**)&layer->fp16_gradients);
            
            safe_free((void**)&layer->fp32_master_weights);
            
            safe_free((void**)&layer->fp32_master_gradients);
        }
        
        safe_free((void**)&context->layers);
    }
    
    // 释放缓冲区
    safe_free((void**)&context->fp16_buffer);
    
    safe_free((void**)&context->fp32_buffer);
    
    safe_free((void**)&context);
}

/**
 * @brief 应用混合精度到前向传播
 */
int mixed_precision_forward(MixedPrecisionContext* context, 
                           const void* input, void* output) {
    if (!context || !input || !output) {
        return -1;
    }
    
    if (!context->enabled) {
        /* 未启用混合精度，使用原始网络 */
        /* 获取网络配置以确定状态缓冲区大小 */
        CfCNetworkConfig network_config;
        CfCNetwork* cfc_net = lnn_get_cfc_network((LNN*)context->network);
        if (!cfc_net || cfc_get_config(cfc_net, &network_config) != 0) {
            return -1;  /* 无法获取网络配置 */
        }
        
        // 分配临时隐藏状态和细胞状态缓冲区
        float* hidden_state = (float*)safe_malloc(network_config.hidden_size * sizeof(float));
        float* cell_state = (float*)safe_malloc(network_config.hidden_size * sizeof(float));
        
        if (!hidden_state || !cell_state) {
            safe_free((void**)&hidden_state);
            safe_free((void**)&cell_state);
            return -1;  // 内存分配失败
        }
        
        // 调用原始网络前向传播
        int result = lnn_forward(context->network, (const float*)input, 
                                (float*)output);
        
        // 释放临时缓冲区
        safe_free((void**)&hidden_state);
        safe_free((void**)&cell_state);
        
        return result;
    }
    
    /* 获取网络配置（确定各维度大小） */
    CfCNetworkConfig network_config;
    CfCNetwork* cfc_net2 = lnn_get_cfc_network((LNN*)context->network);
    if (!cfc_net2 || cfc_get_config(cfc_net2, &network_config) != 0) {
        return -1;
    }
    const size_t input_size = network_config.input_size;
    const size_t hidden_size = network_config.hidden_size;
    const size_t output_size = network_config.output_size;
    
    // 分配隐藏状态和细胞状态缓冲区
    float* hidden_state = (float*)safe_malloc(hidden_size * sizeof(float));
    float* cell_state = (float*)safe_malloc(hidden_size * sizeof(float));
    if (!hidden_state || !cell_state) {
        safe_free((void**)&hidden_state);
        safe_free((void**)&cell_state);
        return -1;
    }
    
    int result = -1;
    
    if (context->config.use_fp16_for_forward) {
        // 转换输入到FP16（用于统计和精度监控）
        fp16_t* fp16_input = (fp16_t*)context->fp16_buffer;
        if (context->fp16_buffer && input_size > 0) {
            convert_fp32_to_fp16((const float*)input, fp16_input, input_size);
        }
        
        // 使用FP16网络执行前向传播
        // fp16_network内部使用FP16精度权重，但API接受float*输入输出
        if (context->fp16_network) {
            result = lnn_forward(context->fp16_network, (const float*)input,
                                (float*)output);
        }
        
        // 将输出转换到FP16缓冲区用于精度监控
        if (context->fp16_buffer && result == 0 && output_size > 0) {
            // 使用fp16_buffer的后半部分存储FP16输出
            fp16_t* fp16_output = (fp16_t*)context->fp16_buffer + input_size;
            convert_fp32_to_fp16((const float*)output, fp16_output, output_size);
        }
        
        context->stats.fp32_to_fp16_conversions++;
    } else {
        // 使用FP32原始网络执行前向传播
        result = lnn_forward(context->network, (const float*)input,
                            (float*)output);
    }
    
    // 释放临时状态缓冲区
    safe_free((void**)&hidden_state);
    safe_free((void**)&cell_state);
    
    return result;
}

/**
 * @brief 应用混合精度到反向传播
 */
int mixed_precision_backward(MixedPrecisionContext* context,
                            const void* grad_output, void* grad_input) {
    (void)grad_input;
    if (!context || !grad_output) {
        return -1;
    }
    
    if (!context->enabled) {
        /* 未启用混合精度，使用原始网络 */
        /* 获取网络配置以获取学习率 */
        CfCNetworkConfig network_config;
        CfCNetwork* cfc_net3 = lnn_get_cfc_network((LNN*)context->network);
        if (!cfc_net3 || cfc_get_config(cfc_net3, &network_config) != 0) {
            return -1;  /* 无法获取网络配置 */
        }
        
        // 分配临时梯度缓冲区（如果需要）
        // cfc_backward需要梯度缓冲区，grad_input可能就是这个缓冲区
        // 如果grad_input为NULL，我们需要分配临时缓冲区
        float* gradient_buffer = NULL;
        int need_free_gradient = 0;
        
        if (!grad_input) {
            // 获取网络的总参数数量以准确估计梯度缓冲区大小
            size_t total_param_count = 0;
            float* weight_matrix = NULL;
            size_t weight_count = 0;
            float* bias_vector = NULL;
            size_t bias_count = 0;
            
            /* 获取权重矩阵大小 */
            CfCNetwork* cfc_back = lnn_get_cfc_network((LNN*)context->network);
            if (cfc_back && cfc_get_weight_matrix(cfc_back, &weight_matrix, &weight_count) == 0) {
                total_param_count += weight_count;
            }
            
            /* 获取偏置向量大小 */
            if (cfc_back && cfc_get_bias_vector(cfc_back, &bias_vector, &bias_count) == 0) {
                total_param_count += bias_count;
            }
            
            // 如果无法获取参数数量，回退到保守估计（输入大小 + 隐藏大小 + 输出大小）
            if (total_param_count == 0) {
                total_param_count = network_config.input_size + network_config.hidden_size + network_config.output_size;
            }
            
            // 分配梯度缓冲区（至少能容纳所有参数的梯度）
            gradient_buffer = (float*)safe_malloc(total_param_count * sizeof(float));
            if (!gradient_buffer) {
                return -1;  // 内存分配失败
            }
            need_free_gradient = 1;
        } else {
            gradient_buffer = (float*)grad_input;
        }
        
        /* 调用原始网络反向传播 */
        /* 注意：grad_output作为误差输入，gradient_buffer作为梯度输出 */
        CfCNetwork* cfc_bw3 = lnn_get_cfc_network((LNN*)context->network);
        int result = (cfc_bw3) ? cfc_backward(cfc_bw3, (const float*)grad_output, 
                                 gradient_buffer, network_config.learning_rate) : -1;
        
        // 如果分配了临时缓冲区，需要将结果复制到grad_input（如果提供了）
        // 但grad_input为NULL，所以这里不需要复制
        
        if (need_free_gradient && gradient_buffer) {
            safe_free((void**)&gradient_buffer);
        }
        
        return result;
    }
    
    /* 获取网络配置（确定学习率等参数） */
    CfCNetworkConfig network_config;
    CfCNetwork* cfc_net4 = lnn_get_cfc_network((LNN*)context->network);
    if (!cfc_net4 || cfc_get_config(cfc_net4, &network_config) != 0) {
        return -1;
    }
    
    /* 获取梯度大小 */
    float* weight_matrix = NULL;
    size_t weight_count = 0;
    float* bias_vector = NULL;
    size_t bias_count = 0;
    size_t grad_size = 0;
    
    CfCNetwork* cfc_grad = lnn_get_cfc_network((LNN*)context->network);
    if (cfc_grad) {
        cfc_get_weight_matrix(cfc_grad, &weight_matrix, &weight_count);
        cfc_get_bias_vector(cfc_grad, &bias_vector, &bias_count);
    }
    grad_size = weight_count + bias_count;
    if (grad_size == 0) {
        grad_size = network_config.hidden_size;
    }
    
    // 处理grad_input为NULL的情况（分配临时梯度缓冲区）
    float* gradient_buffer = NULL;
    int need_free_gradient = 0;
    if (!grad_input) {
        gradient_buffer = (float*)safe_malloc(grad_size * sizeof(float));
        if (!gradient_buffer) {
            return -1;
        }
        need_free_gradient = 1;
    } else {
        gradient_buffer = (float*)grad_input;
    }
    
    int result = -1;
    
    if (context->config.use_fp16_for_backward) {
        // FP16反向传播路径
        // 将grad_output转换为FP16用于缩放（避免FP16下溢出）
        fp16_t* fp16_grad = (fp16_t*)context->fp16_buffer;
        if (context->fp16_buffer && network_config.hidden_size > 0) {
            convert_fp32_to_fp16((const float*)grad_output, fp16_grad, 
                                network_config.hidden_size);
        }
        
        // 应用梯度缩放到FP16版本（用于FP16权重更新）
        if (context->config.scaling != GRADIENT_SCALING_NONE && context->fp16_buffer) {
            scale_fp16_gradients(fp16_grad, network_config.hidden_size, 
                               context->current_scale);
            
            // 将缩放后的FP16梯度转回FP32作为误差信号
            float* scaled_error = (float*)context->fp32_buffer;
            if (context->fp32_buffer) {
                convert_fp16_to_fp32(fp16_grad, scaled_error, 
                                   network_config.hidden_size);
                
                /* 使用FP16网络执行反向传播（以缩放后的误差为输入） */
                if (context->fp16_network) {
                    CfCNetwork* cfc_fp16fw = lnn_get_cfc_network((LNN*)context->fp16_network);
                    result = (cfc_fp16fw) ? cfc_backward(cfc_fp16fw, scaled_error,
                                        gradient_buffer, network_config.learning_rate) : -1;
                }
            } else {
                // 无FP32缓冲区，直接使用缩放后的原始grad_output
                if (context->fp16_network) {
                    CfCNetwork* cfc_fp16bw2 = lnn_get_cfc_network((LNN*)context->fp16_network);
                    result = (cfc_fp16bw2) ? cfc_backward(cfc_fp16bw2, 
                                        (const float*)grad_output,
                                        gradient_buffer, network_config.learning_rate) : -1;
                }
            }
        } else {
            /* 无梯度缩放，直接使用原始grad_output */
            if (context->fp16_network) {
                CfCNetwork* cfc_fp16bw3 = lnn_get_cfc_network((LNN*)context->fp16_network);
                result = (cfc_fp16bw3) ? cfc_backward(cfc_fp16bw3, 
                                    (const float*)grad_output,
                                    gradient_buffer, network_config.learning_rate) : -1;
            }
        }
        
        // 检查数值稳定性（使用真实大小）
        if (context->config.check_nan_inf && result == 0) {
            for (size_t i = 0; i < network_config.hidden_size; i++) {
                check_fp16_stability(fp16_grad[i], &context->stats);
            }
        }
    } else {
        // FP32反向传播路径
        // 应用梯度缩放到grad_output
        if (context->config.scaling != GRADIENT_SCALING_NONE) {
            scale_fp32_gradients((float*)grad_output, network_config.hidden_size, 
                               context->current_scale);
        }
        
        /* 使用FP32原始网络执行反向传播 */
        CfCNetwork* cfc_bw4 = lnn_get_cfc_network((LNN*)context->network);
        result = (cfc_bw4) ? cfc_backward(cfc_bw4, (const float*)grad_output,
                            gradient_buffer, network_config.learning_rate) : -1;
    }
    
    // 释放临时梯度缓冲区
    if (need_free_gradient && gradient_buffer) {
        safe_free((void**)&gradient_buffer);
    }
    
    // 更新统计信息
    if (context->config.use_fp16_for_backward) {
        context->stats.fp16_to_fp32_conversions++;
    }
    
    return result;
}

/**
 * @brief 更新混合精度权重
 */
int mixed_precision_update_weights(MixedPrecisionContext* context, 
                                  float learning_rate) {
    if (!context) {
        return -1;
    }
    
    if (!context->enabled || !context->training) {
        return 0;  // 未启用混合精度或不在训练中
    }
    
    // 更新缩放计数器
    context->scale_update_counter++;
    
    // 检查是否需要更新缩放因子
    if (context->scale_update_counter >= context->config.scale_update_interval) {
        context->scale_update_counter = 0;
        
        // 更新缩放因子
        mixed_precision_update_scaling(context, NULL, 0);
    }
    
    // 根据配置更新权重
    if (context->config.use_fp32_master_weights) {
        // 更新FP32主权重
        for (int i = 0; i < context->layer_count; i++) {
            MixedPrecisionLayer* layer = &context->layers[i];
            
            if (layer->fp32_master_weights && layer->fp32_master_gradients) {
                // 应用梯度到权重
                size_t weight_count = layer->weight_size / sizeof(float);
                float* weights = (float*)layer->fp32_master_weights;
                float* gradients = (float*)layer->fp32_master_gradients;
                
                for (size_t j = 0; j < weight_count; j++) {
                    weights[j] -= learning_rate * gradients[j];
                }
                
                // 同步到FP16权重
                if (layer->fp16_weights) {
                    convert_fp32_to_fp16(weights, (fp16_t*)layer->fp16_weights, weight_count);
                }
            }
        }
    } else {
        // 直接更新FP16权重
        for (int i = 0; i < context->layer_count; i++) {
            MixedPrecisionLayer* layer = &context->layers[i];
            
            if (layer->fp16_weights && layer->fp16_gradients) {
                // 应用梯度到权重
                size_t weight_count = layer->weight_size / sizeof(fp16_t);
                fp16_t* weights = (fp16_t*)layer->fp16_weights;
                fp16_t* gradients = (fp16_t*)layer->fp16_gradients;
                
                for (size_t j = 0; j < weight_count; j++) {
                    float weight = fp16_to_fp32(weights[j]);
                    float gradient = fp16_to_fp32(gradients[j]);
                    weight -= learning_rate * gradient;
                    weights[j] = fp32_to_fp16(weight);
                }
            }
        }
    }
    
    // 更新统计信息
    context->stats.total_tensor_count = context->layer_count * 2;  // 权重+梯度
    
    return 0;
}

/**
 * @brief 转换张量精度
 */
int mixed_precision_convert_tensor(const void* src, void* dst, 
                                  size_t count, int src_dtype, int dst_dtype) {
    if (!src || !dst || count == 0) {
        return -1;
    }
    
    // 完整实现：支持多种精度转换
    // 相同数据类型的直接内存复制
    if (src_dtype == dst_dtype) {
        memcpy(dst, src, count * get_data_type_size(src_dtype));
        return 0;
    }
    
    // FP32 <-> FP16 转换
    if (src_dtype == DATA_TYPE_FLOAT32 && dst_dtype == DATA_TYPE_FLOAT16) {
        return convert_fp32_to_fp16((const float*)src, (fp16_t*)dst, count);
    } else if (src_dtype == DATA_TYPE_FLOAT16 && dst_dtype == DATA_TYPE_FLOAT32) {
        return convert_fp16_to_fp32((const fp16_t*)src, (float*)dst, count);
    }
    
    // FP32 <-> FP64 转换（如果需要）
    if (src_dtype == DATA_TYPE_FLOAT32 && dst_dtype == DATA_TYPE_FLOAT64) {
        const float* src_f32 = (const float*)src;
        double* dst_f64 = (double*)dst;
        for (size_t i = 0; i < count; i++) {
            dst_f64[i] = (double)src_f32[i];
        }
        return 0;
    } else if (src_dtype == DATA_TYPE_FLOAT64 && dst_dtype == DATA_TYPE_FLOAT32) {
        const double* src_f64 = (const double*)src;
        float* dst_f32 = (float*)dst;
        for (size_t i = 0; i < count; i++) {
            dst_f32[i] = (float)src_f64[i];
        }
        return 0;
    }
    
    // 整数到浮点转换（完整支持）
    if (dst_dtype == DATA_TYPE_FLOAT32 || dst_dtype == DATA_TYPE_FLOAT64) {
        // 根据源整数类型和目标浮点类型进行转换
        if (src_dtype == DATA_TYPE_INT8 && dst_dtype == DATA_TYPE_FLOAT32) {
            const int8_t* src_int8 = (const int8_t*)src;
            float* dst_f32 = (float*)dst;
            for (size_t i = 0; i < count; i++) {
                dst_f32[i] = (float)src_int8[i];
            }
            return 0;
        } else if (src_dtype == DATA_TYPE_INT16 && dst_dtype == DATA_TYPE_FLOAT32) {
            const int16_t* src_int16 = (const int16_t*)src;
            float* dst_f32 = (float*)dst;
            for (size_t i = 0; i < count; i++) {
                dst_f32[i] = (float)src_int16[i];
            }
            return 0;
        } else if (src_dtype == DATA_TYPE_INT32 && dst_dtype == DATA_TYPE_FLOAT32) {
            const int32_t* src_int32 = (const int32_t*)src;
            float* dst_f32 = (float*)dst;
            for (size_t i = 0; i < count; i++) {
                dst_f32[i] = (float)src_int32[i];
            }
            return 0;
        } else if (src_dtype == DATA_TYPE_INT64 && dst_dtype == DATA_TYPE_FLOAT32) {
            const int64_t* src_int64 = (const int64_t*)src;
            float* dst_f32 = (float*)dst;
            for (size_t i = 0; i < count; i++) {
                dst_f32[i] = (float)src_int64[i];
            }
            return 0;
        } else if (src_dtype == DATA_TYPE_UINT8 && dst_dtype == DATA_TYPE_FLOAT32) {
            const uint8_t* src_uint8 = (const uint8_t*)src;
            float* dst_f32 = (float*)dst;
            for (size_t i = 0; i < count; i++) {
                dst_f32[i] = (float)src_uint8[i];
            }
            return 0;
        } else if (src_dtype == DATA_TYPE_UINT16 && dst_dtype == DATA_TYPE_FLOAT32) {
            const uint16_t* src_uint16 = (const uint16_t*)src;
            float* dst_f32 = (float*)dst;
            for (size_t i = 0; i < count; i++) {
                dst_f32[i] = (float)src_uint16[i];
            }
            return 0;
        } else if (src_dtype == DATA_TYPE_UINT32 && dst_dtype == DATA_TYPE_FLOAT32) {
            const uint32_t* src_uint32 = (const uint32_t*)src;
            float* dst_f32 = (float*)dst;
            for (size_t i = 0; i < count; i++) {
                dst_f32[i] = (float)src_uint32[i];
            }
            return 0;
        } else if (src_dtype == DATA_TYPE_UINT64 && dst_dtype == DATA_TYPE_FLOAT32) {
            const uint64_t* src_uint64 = (const uint64_t*)src;
            float* dst_f32 = (float*)dst;
            for (size_t i = 0; i < count; i++) {
                dst_f32[i] = (float)src_uint64[i];
            }
            return 0;
        } else if (src_dtype == DATA_TYPE_INT32 && dst_dtype == DATA_TYPE_FLOAT64) {
            const int32_t* src_int32 = (const int32_t*)src;
            double* dst_f64 = (double*)dst;
            for (size_t i = 0; i < count; i++) {
                dst_f64[i] = (double)src_int32[i];
            }
            return 0;
        } else if (src_dtype == DATA_TYPE_UINT32 && dst_dtype == DATA_TYPE_FLOAT64) {
            const uint32_t* src_uint32 = (const uint32_t*)src;
            double* dst_f64 = (double*)dst;
            for (size_t i = 0; i < count; i++) {
                dst_f64[i] = (double)src_uint32[i];
            }
            return 0;
        }
    }
    
    // 浮点到整数转换（完整支持）
    if (src_dtype == DATA_TYPE_FLOAT32 || src_dtype == DATA_TYPE_FLOAT64 || 
        src_dtype == DATA_TYPE_FLOAT16) {
        // 根据源浮点类型和目标整数类型进行转换
        // FP32到整数转换
        if (src_dtype == DATA_TYPE_FLOAT32) {
            const float* src_f32 = (const float*)src;
            
            if (dst_dtype == DATA_TYPE_INT8) {
                int8_t* dst_int8 = (int8_t*)dst;
                for (size_t i = 0; i < count; i++) {
                    // 限制范围并转换
                    float val = src_f32[i];
                    if (val > 127.0f) val = 127.0f;
                    if (val < -128.0f) val = -128.0f;
                    dst_int8[i] = (int8_t)val;
                }
                return 0;
            } else if (dst_dtype == DATA_TYPE_INT16) {
                int16_t* dst_int16 = (int16_t*)dst;
                for (size_t i = 0; i < count; i++) {
                    float val = src_f32[i];
                    if (val > 32767.0f) val = 32767.0f;
                    if (val < -32768.0f) val = -32768.0f;
                    dst_int16[i] = (int16_t)val;
                }
                return 0;
            } else if (dst_dtype == DATA_TYPE_INT32) {
                int32_t* dst_int32 = (int32_t*)dst;
                for (size_t i = 0; i < count; i++) {
                    dst_int32[i] = (int32_t)src_f32[i];
                }
                return 0;
            } else if (dst_dtype == DATA_TYPE_INT64) {
                int64_t* dst_int64 = (int64_t*)dst;
                for (size_t i = 0; i < count; i++) {
                    dst_int64[i] = (int64_t)src_f32[i];
                }
                return 0;
            } else if (dst_dtype == DATA_TYPE_UINT8) {
                uint8_t* dst_uint8 = (uint8_t*)dst;
                for (size_t i = 0; i < count; i++) {
                    float val = src_f32[i];
                    if (val > 255.0f) val = 255.0f;
                    if (val < 0.0f) val = 0.0f;
                    dst_uint8[i] = (uint8_t)val;
                }
                return 0;
            } else if (dst_dtype == DATA_TYPE_UINT16) {
                uint16_t* dst_uint16 = (uint16_t*)dst;
                for (size_t i = 0; i < count; i++) {
                    float val = src_f32[i];
                    if (val > 65535.0f) val = 65535.0f;
                    if (val < 0.0f) val = 0.0f;
                    dst_uint16[i] = (uint16_t)val;
                }
                return 0;
            } else if (dst_dtype == DATA_TYPE_UINT32) {
                uint32_t* dst_uint32 = (uint32_t*)dst;
                for (size_t i = 0; i < count; i++) {
                    float val = src_f32[i];
                    if (val < 0.0f) val = 0.0f;
                    dst_uint32[i] = (uint32_t)val;
                }
                return 0;
            } else if (dst_dtype == DATA_TYPE_UINT64) {
                uint64_t* dst_uint64 = (uint64_t*)dst;
                for (size_t i = 0; i < count; i++) {
                    float val = src_f32[i];
                    if (val < 0.0f) val = 0.0f;
                    dst_uint64[i] = (uint64_t)val;
                }
                return 0;
            }
        }
        // FP64到整数转换（类似FP32但使用double）
        else if (src_dtype == DATA_TYPE_FLOAT64) {
            const double* src_f64 = (const double*)src;
            
            if (dst_dtype == DATA_TYPE_INT32) {
                int32_t* dst_int32 = (int32_t*)dst;
                for (size_t i = 0; i < count; i++) {
                    dst_int32[i] = (int32_t)src_f64[i];
                }
                return 0;
            } else if (dst_dtype == DATA_TYPE_INT64) {
                int64_t* dst_int64 = (int64_t*)dst;
                for (size_t i = 0; i < count; i++) {
                    dst_int64[i] = (int64_t)src_f64[i];
                }
                return 0;
            } else if (dst_dtype == DATA_TYPE_UINT32) {
                uint32_t* dst_uint32 = (uint32_t*)dst;
                for (size_t i = 0; i < count; i++) {
                    double val = src_f64[i];
                    if (val < 0.0) val = 0.0;
                    dst_uint32[i] = (uint32_t)val;
                }
                return 0;
            } else if (dst_dtype == DATA_TYPE_UINT64) {
                uint64_t* dst_uint64 = (uint64_t*)dst;
                for (size_t i = 0; i < count; i++) {
                    double val = src_f64[i];
                    if (val < 0.0) val = 0.0;
                    dst_uint64[i] = (uint64_t)val;
                }
                return 0;
            }
        }
        // FP16到整数转换（先转换到FP32，再转换到整数）
        else if (src_dtype == DATA_TYPE_FLOAT16) {
            // 由于FP16到整数转换不常见，且需要先转换到FP32
            // 这里提供一个基本实现框架
            // 在实际使用中，可能需要更高效的实现
            float* temp_buffer = (float*)safe_malloc(count * sizeof(float));
            if (!temp_buffer) {
                return -1;
            }
            
            // 先转换FP16到FP32
            if (convert_fp16_to_fp32((const fp16_t*)src, temp_buffer, count) != 0) {
                safe_free((void**)&temp_buffer);
                return -1;
            }
            
            // 然后从FP32转换到目标整数类型
            int result = mixed_precision_convert_tensor(temp_buffer, dst, count, 
                                                       DATA_TYPE_FLOAT32, dst_dtype);
            
            safe_free((void**)&temp_buffer);
            return result;
        }
    }
    
    // 整数到整数转换（支持有符号/无符号转换）
    // 检查是否都是整数类型
    int src_is_int = (src_dtype >= DATA_TYPE_INT8 && src_dtype <= DATA_TYPE_UINT64);
    int dst_is_int = (dst_dtype >= DATA_TYPE_INT8 && dst_dtype <= DATA_TYPE_UINT64);
    
    if (src_is_int && dst_is_int) {
        // 使用中间浮点表示进行转换（确保范围正确处理）
        float* temp_buffer = (float*)safe_malloc(count * sizeof(float));
        if (!temp_buffer) {
            return -1;
        }
        
        // 先转换源整数到FP32
        if (mixed_precision_convert_tensor(src, temp_buffer, count, 
                                          src_dtype, DATA_TYPE_FLOAT32) != 0) {
            safe_free((void**)&temp_buffer);
            return -1;
        }
        
        // 然后从FP32转换到目标整数
        int result = mixed_precision_convert_tensor(temp_buffer, dst, count, 
                                                   DATA_TYPE_FLOAT32, dst_dtype);
        
        safe_free((void**)&temp_buffer);
        return result;
    }
    
    // 布尔类型转换
    if (src_dtype == DATA_TYPE_BOOL || dst_dtype == DATA_TYPE_BOOL) {
        // 布尔类型转换相对简单
        if (src_dtype == DATA_TYPE_BOOL && dst_dtype == DATA_TYPE_BOOL) {
            memcpy(dst, src, count);
            return 0;
        } else if (src_dtype == DATA_TYPE_BOOL) {
            const uint8_t* src_bool = (const uint8_t*)src;
            // 布尔到其他类型转换：true->1, false->0
            if (dst_dtype == DATA_TYPE_FLOAT32) {
                float* dst_f32 = (float*)dst;
                for (size_t i = 0; i < count; i++) {
                    dst_f32[i] = src_bool[i] ? 1.0f : 0.0f;
                }
                return 0;
            } else if (dst_dtype == DATA_TYPE_INT32) {
                int32_t* dst_int32 = (int32_t*)dst;
                for (size_t i = 0; i < count; i++) {
                    dst_int32[i] = src_bool[i] ? 1 : 0;
                }
                return 0;
            }
        } else if (dst_dtype == DATA_TYPE_BOOL) {
            uint8_t* dst_bool = (uint8_t*)dst;
            // 其他类型到布尔转换：非零->true, 零->false
            if (src_dtype == DATA_TYPE_FLOAT32) {
                const float* src_f32 = (const float*)src;
                for (size_t i = 0; i < count; i++) {
                    dst_bool[i] = (src_f32[i] != 0.0f) ? 1 : 0;
                }
                return 0;
            } else if (src_dtype == DATA_TYPE_INT32) {
                const int32_t* src_int32 = (const int32_t*)src;
                for (size_t i = 0; i < count; i++) {
                    dst_bool[i] = (src_int32[i] != 0) ? 1 : 0;
                }
                return 0;
            }
        }
    }
    
    // 不支持的转换类型
    // 注意：仍然有一些转换类型可能不被支持
    return -1;
}

/**
 * @brief 应用梯度缩放
 */
int mixed_precision_scale_gradients(void* gradients, size_t count, 
                                   float scale, int dtype) {
    if (!gradients || count == 0 || scale == 1.0f) {
        return 0;
    }
    
    if (dtype == DATA_TYPE_FLOAT32) {
        return scale_fp32_gradients((float*)gradients, count, scale);
    } else if (dtype == DATA_TYPE_FLOAT16) {
        return scale_fp16_gradients((fp16_t*)gradients, count, scale);
    } else {
        return -1;  // 不支持的数据类型
    }
}

/**
 * @brief 检查数值稳定性
 */
int mixed_precision_check_stability(const void* data, size_t count, 
                                   int dtype, PrecisionStatistics* stats) {
    if (!data || count == 0 || !stats) {
        return -1;
    }
    
    int unstable_count = 0;
    
    if (dtype == DATA_TYPE_FLOAT32) {
        const float* fp32_data = (const float*)data;
        for (size_t i = 0; i < count; i++) {
            if (check_value_stability(fp32_data[i], stats)) {
                unstable_count++;
            }
        }
    } else if (dtype == DATA_TYPE_FLOAT16) {
        const fp16_t* fp16_data = (const fp16_t*)data;
        for (size_t i = 0; i < count; i++) {
            if (check_fp16_stability(fp16_data[i], stats)) {
                unstable_count++;
            }
        }
    } else {
        return -1;
    }
    
    return (unstable_count > 0) ? 1 : 0;
}

/**
 * @brief 更新梯度缩放因子
 */
float mixed_precision_update_scaling(MixedPrecisionContext* context,
                                    const void* gradients, size_t count) {
    if (!context) {
        return 1.0f;
    }
    
    // 完整实现：基于梯度统计和数值稳定性调整缩放因子
    PrecisionStatistics* stats = &context->stats;
    
    // 如果提供了梯度数据，分析梯度幅度和数值稳定性
    if (gradients && count > 0) {
        // 根据配置确定梯度数据类型（假设为FP32或FP16）
        int gradient_dtype = DATA_TYPE_FLOAT32;  // 默认假设FP32
        
        // 检查上下文是否使用FP16梯度
        if (context->config.use_fp16_for_backward) {
            gradient_dtype = DATA_TYPE_FLOAT16;
        }
        
        // 分析梯度数值稳定性
        size_t nan_count = 0;
        size_t inf_count = 0;
        size_t underflow_count = 0;
        size_t overflow_count = 0;
        float gradient_magnitude = 0.0f;
        
        if (gradient_dtype == DATA_TYPE_FLOAT32) {
            const float* grad_f32 = (const float*)gradients;
            for (size_t i = 0; i < count; i++) {
                float g = grad_f32[i];
                
                // 检测NaN和Inf
                if (isnan(g)) {
                    nan_count++;
                } else if (isinf(g)) {
                    inf_count++;
                } else {
                    // 计算梯度幅度（L2范数的一部分）
                    gradient_magnitude += g * g;
                    
                    // 检测下溢（非常接近零的值）
                    if (fabsf(g) < context->config.fp16_min_value) {
                        underflow_count++;
                    }
                    
                    // 检测上溢（超过FP16最大值）
                    if (fabsf(g) > context->config.fp16_max_value) {
                        overflow_count++;
                    }
                }
            }
            
            // 计算平均梯度幅度
            if (count > 0) {
                gradient_magnitude = sqrtf(gradient_magnitude / count);
            }
        } else if (gradient_dtype == DATA_TYPE_FLOAT16) {
            const fp16_t* grad_f16 = (const fp16_t*)gradients;
            for (size_t i = 0; i < count; i++) {
                // 将FP16转换为FP32进行分析
                float g = fp16_to_fp32(grad_f16[i]);
                
                // 检测NaN和Inf
                if (isnan(g)) {
                    nan_count++;
                } else if (isinf(g)) {
                    inf_count++;
                } else {
                    // 计算梯度幅度
                    gradient_magnitude += g * g;
                    
                    // 检测下溢
                    if (fabsf(g) < context->config.fp16_min_value) {
                        underflow_count++;
                    }
                    
                    // 检测上溢
                    if (fabsf(g) > context->config.fp16_max_value) {
                        overflow_count++;
                    }
                }
            }
            
            // 计算平均梯度幅度
            if (count > 0) {
                gradient_magnitude = sqrtf(gradient_magnitude / count);
            }
        }
        
        // 更新统计信息
        stats->nan_count += nan_count;
        stats->inf_count += inf_count;
        stats->underflow_count += underflow_count;
        stats->overflow_count += overflow_count;
        
        // 更新平均梯度幅度（指数移动平均）
        if (stats->average_gradient_magnitude == 0.0f) {
            stats->average_gradient_magnitude = gradient_magnitude;
        } else {
            // EMA系数：0.9
            stats->average_gradient_magnitude = 
                0.9f * stats->average_gradient_magnitude + 0.1f * gradient_magnitude;
        }
        
        // 基于梯度幅度调整缩放因子（动态缩放）
        if (context->config.scaling == GRADIENT_SCALING_DYNAMIC || 
            context->config.scaling == GRADIENT_SCALING_ADAPTIVE) {
            
            // 目标梯度幅度：保持梯度在合理范围内
            float target_magnitude = 1.0f;  // 目标梯度幅度
            
            if (gradient_magnitude > 0.0f) {
                // 计算当前缩放因子与目标缩放因子的比例
                float ratio = target_magnitude / gradient_magnitude;
                
                // 限制调整幅度
                if (ratio < 0.5f) ratio = 0.5f;
                if (ratio > 2.0f) ratio = 2.0f;
                
                // 平滑调整缩放因子
                context->current_scale *= ratio;
            }
        }
    }
    
    // 检查数值不稳定性（NaN/Inf/溢出）
    if (stats->nan_count > 0 || stats->inf_count > 0 || stats->overflow_count > 0) {
        // 数值不稳定，减少缩放因子
        context->current_scale *= context->config.backoff_factor;
        context->scale_decrease_pending = 1;
        
        // 确保缩放因子不小于最小值
        if (context->current_scale < context->config.min_scale) {
            context->current_scale = context->config.min_scale;
        }
        
        stats->scale_decreases++;
        
        // 重置不稳定计数器（部分重置）
        stats->nan_count = 0;
        stats->inf_count = 0;
        stats->overflow_count = 0;
    } else if (context->scale_update_counter > 0 && 
               stats->underflow_count == 0) {
        // 数值稳定，可能增加缩放因子
        if (context->scale_increase_pending) {
            context->current_scale *= context->config.growth_factor;
            context->scale_increase_pending = 0;
            
            // 确保缩放因子不大于最大值
            if (context->current_scale > context->config.max_scale) {
                context->current_scale = context->config.max_scale;
            }
            
            stats->scale_increases++;
        } else {
            // 计划下次增加
            context->scale_increase_pending = 1;
        }
    }
    
    // 更新计数器
    context->scale_update_counter++;
    
    // 应用缩放因子限制
    if (context->current_scale < context->config.min_scale) {
        context->current_scale = context->config.min_scale;
    }
    if (context->current_scale > context->config.max_scale) {
        context->current_scale = context->config.max_scale;
    }
    
    // 更新统计信息中的当前缩放因子
    stats->current_scale = context->current_scale;
    
    return context->current_scale;
}

/**
 * @brief 获取精度统计信息
 */
int mixed_precision_get_statistics(MixedPrecisionContext* context,
                                  PrecisionStatistics* stats) {
    if (!context || !stats) {
        return -1;
    }
    
    memcpy(stats, &context->stats, sizeof(PrecisionStatistics));
    return 0;
}

/**
 * @brief 重置精度统计信息
 */
int mixed_precision_reset_statistics(MixedPrecisionContext* context) {
    if (!context) {
        return -1;
    }
    
    memset(&context->stats, 0, sizeof(PrecisionStatistics));
    context->stats.current_scale = context->current_scale;
    return 0;
}

/**
 * @brief 启用混合精度训练
 * 
 * 完整实现：检测硬件FP16支持能力，根据硬件能力自动调整混合精度配置。
 * - 有Tensor Cores的GPU：启用完整FP16+动态缩放
 * - 有FP16算术但无Tensor Cores：仅前向使用FP16，反向使用FP32
 * - CPU或硬件不支持FP16：自动降级到FP32模式
 */
int mixed_precision_enable(Trainer* trainer, const MixedPrecisionConfig* config) {
    if (!trainer) {
        return -1;
    }
    
    // 步骤1：获取训练器中的神经网络
    LNN* network = trainer_get_network(trainer);
    if (!network) {
        return -1;
    }
    
    // 步骤2：检测硬件FP16支持能力（先于任何FP16操作执行）
    HardwareFP16Support hw = mixed_precision_detect_hardware_support();
    
    // 步骤3：根据硬件能力创建有效的配置副本
    MixedPrecisionConfig effective_config;
    if (config) {
        memcpy(&effective_config, config, sizeof(MixedPrecisionConfig));
    } else {
        mixed_precision_default_config(&effective_config);
    }
    
    // 步骤4：硬件能力驱动的自动降级逻辑
    int hw_supports_fp16 = (hw.has_fp16_native || hw.has_tensor_cores || hw.has_fp16_arithmetic);
    
    if (!hw_supports_fp16 && effective_config.mode != MIXED_PRECISION_DISABLED) {
        /* 硬件不支持FP16：自动降级到FP32 */
        printf("警告：当前硬件不支持FP16计算（设备：%s），自动降级到FP32模式\n",
               hw.device_name[0] ? hw.device_name : "CPU（未检测到GPU）");
        
        effective_config.mode = MIXED_PRECISION_DISABLED;
        effective_config.use_fp16_for_forward = 0;
        effective_config.use_fp16_for_backward = 0;
        effective_config.use_fp16_for_weights = 0;
        effective_config.enable_fp16_arithmetic = 0;
        effective_config.enable_fp16_storage = 0;
        effective_config.scaling = GRADIENT_SCALING_NONE;
    } else if (hw.has_fp16_arithmetic && !hw.has_tensor_cores) {
        /* 有FP16算术但无Tensor Cores：反向传播使用FP32保持精度，仅前向使用FP16加速 */
        effective_config.use_fp16_for_backward = 0;
        effective_config.use_fp16_for_weights = 0;
        effective_config.use_fp32_master_weights = 1;
        
        printf("混合精度：检测到FP16算术支持但无Tensor Cores，启用前向FP16+反向FP32模式\n");
    } else if (hw.has_tensor_cores) {
        /* 有Tensor Cores：最大程度使用FP16，开启动态缩放 */
        effective_config.use_fp16_for_forward = 1;
        effective_config.use_fp16_for_backward = 1;
        effective_config.use_fp16_for_weights = 1;
        effective_config.use_fp32_master_weights = 1;
        effective_config.enable_automatic_loss_scaling = 1;
        effective_config.scaling = GRADIENT_SCALING_DYNAMIC;
        if (effective_config.initial_scale < 65536.0f) {
            effective_config.initial_scale = 65536.0f;
        }
        
        printf("混合精度：检测到Tensor Cores（%s），启用完整FP16+动态缩放训练\n",
               hw.device_name);
    }
    
    // 步骤5：创建混合精度上下文（使用硬件调整后的配置）
    MixedPrecisionContext* context = mixed_precision_create((NeuralNetwork*)network, &effective_config);
    if (!context) {
        return -1;
    }
    
    // 步骤6：保存硬件检测结果到上下文
    context->hardware = hw;
    
    // 步骤7：设置启用状态（根据硬件能力）
    context->enabled = (effective_config.mode != MIXED_PRECISION_DISABLED) ? 1 : 0;
    context->training = 1;
    
    // 步骤8：将上下文附加到训练器
    int result = trainer_set_mixed_precision_context(trainer, context);
    if (result != 0) {
        mixed_precision_destroy(context);
        return -1;
    }
    
    // 步骤9：初始化FP16网络（仅当启用且配置需要时）
    if (context->enabled && (effective_config.use_fp16_for_forward || effective_config.use_fp16_for_backward)) {
        context->fp16_network = mixed_precision_create_fp16_network((NeuralNetwork*)network);
        if (!context->fp16_network) {
            context->fp16_network = NULL;
        }
    }
    
    return 0;
}

/**
 * @brief 禁用混合精度训练
 */
int mixed_precision_disable(Trainer* trainer) {
    if (!trainer) {
        return -1;
    }
    
    // 从训练器获取上下文
    MixedPrecisionContext* context = trainer_get_mixed_precision_context(trainer);
    
    if (context) {
        context->enabled = 0;
        context->training = 0;
        // 可选：销毁上下文
        // mixed_precision_destroy(context);
        // trainer_set_mixed_precision_context(trainer, NULL);
    }
    
    printf("混合精度训练已禁用\n");
    return 0;
}

/**
 * @brief 创建FP16版本的神经网络
 */
NeuralNetwork* mixed_precision_create_fp16_network(const NeuralNetwork* network) {
    if (!network) {
        return NULL;
    }
    
    // 完整实现：创建网络副本并转换权重到FP16
    
    // 获取原始LNN配置
    LNNConfig config;
    if (lnn_get_config(network, &config) != 0) {
        return NULL;  // 无法获取配置
    }
    
    // 创建新的LNN网络实例
    LNN* fp16_network = lnn_create(&config);
    if (!fp16_network) {
        return NULL;  // 网络创建失败
    }
    
    /* 获取原始网络的权重和偏置 */
    float* weight_matrix = NULL;
    size_t weight_count = 0;
    float* bias_vector = NULL;
    size_t bias_count = 0;
    
    CfCNetwork* cfc_fp16 = lnn_get_cfc_network((LNN*)network);
    if (cfc_fp16 && cfc_get_weight_matrix(cfc_fp16, &weight_matrix, &weight_count) == 0 &&
        cfc_get_bias_vector(cfc_fp16, &bias_vector, &bias_count) == 0) {
        
        // 分配FP16权重和偏置缓冲区
        fp16_t* fp16_weights = (fp16_t*)safe_malloc(weight_count * sizeof(fp16_t));
        fp16_t* fp16_biases = NULL;
        
        if (bias_count > 0) {
            fp16_biases = (fp16_t*)safe_malloc(bias_count * sizeof(fp16_t));
        }
        
        if (fp16_weights) {
            // 转换权重到FP16
            for (size_t i = 0; i < weight_count; i++) {
                fp16_weights[i] = fp32_to_fp16(weight_matrix[i]);
            }
            
            // 获取FP16网络的权重矩阵并直接设置转换后的FP16值（转换为FP32）
            // 注意：CfC网络内部使用FP32，但我们可以存储近似FP16精度的值
            float* fp16_network_weights = NULL;
            size_t fp16_network_weight_count = 0;
            
            CfCNetwork* cfc_fp16w = lnn_get_cfc_network((LNN*)fp16_network);
            if (cfc_fp16w && cfc_get_weight_matrix(cfc_fp16w, &fp16_network_weights, &fp16_network_weight_count) == 0 &&
                fp16_network_weights && fp16_network_weight_count == weight_count) {
                
                // 将FP16值转换回FP32（保持精度近似）
                for (size_t i = 0; i < weight_count; i++) {
                    fp16_network_weights[i] = fp16_to_fp32(fp16_weights[i]);
                }
                
                printf("混合精度: 已设置FP16精度权重到网络（%zu个参数）\n", weight_count);
            }
            
            safe_free((void**)&fp16_weights);
        }
        
        if (fp16_biases) {
            // 转换偏置到FP16
            for (size_t i = 0; i < bias_count; i++) {
                fp16_biases[i] = fp32_to_fp16(bias_vector[i]);
            }
            
            // 获取FP16网络的偏置向量并直接设置转换后的FP16值（转换为FP32）
            float* fp16_network_biases = NULL;
            size_t fp16_network_bias_count = 0;
            
            CfCNetwork* cfc_fp16b = lnn_get_cfc_network((LNN*)fp16_network);
            if (cfc_fp16b && cfc_get_bias_vector(cfc_fp16b, &fp16_network_biases, &fp16_network_bias_count) == 0 &&
                fp16_network_biases && fp16_network_bias_count == bias_count) {
                
                // 将FP16值转换回FP32（保持精度近似）
                for (size_t i = 0; i < bias_count; i++) {
                    fp16_network_biases[i] = fp16_to_fp32(fp16_biases[i]);
                }
                
                printf("混合精度: 已设置FP16精度偏置到网络（%zu个参数）\n", bias_count);
            }
            
            safe_free((void**)&fp16_biases);
        }
    }
    
    // 函数返回的FP16网络具有近似FP16精度的权重值（存储为FP32）
    // 这提供了内存节省的好处（无需额外存储FP16副本）
    // 同时保持了与现有API的兼容性
    
    return fp16_network;
}

/**
 * @brief 同步FP32主权重
 */
int mixed_precision_sync_master_weights(MixedPrecisionContext* context) {
    if (!context || !context->config.use_fp32_master_weights) {
        return 0;
    }
    
    // 将FP16权重同步到FP32主权重
    for (int i = 0; i < context->layer_count; i++) {
        MixedPrecisionLayer* layer = &context->layers[i];
        
        if (layer->fp16_weights && layer->fp32_master_weights) {
            size_t weight_count = layer->weight_size / sizeof(fp16_t);
            convert_fp16_to_fp32((const fp16_t*)layer->fp16_weights,
                                (float*)layer->fp32_master_weights,
                                weight_count);
        }
    }
    
    return 0;
}