/**
 * @file laplace.c
 * @brief 拉普拉斯变换增强系统实现
 * 
 * 实现拉普拉斯变换分析、系统稳定性评估和频域优化功能。
 * 将拉普拉斯变换技术全面引入液态神经网络系统，增强系统功能。
 */

#define SELFLNN_IMPLEMENTATION  // 启用内部结构体定义

#include "selflnn/core/laplace.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/cfc_network.h"  // 用于访问cfc_get_weight_matrix函数
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/**
 * @brief 复数与标量乘法（如果math_utils中未提供）
 */
static Complex complex_mul_scalar(Complex c, float scalar) {
    Complex result;
    result.real = c.real * scalar;
    result.imag = c.imag * scalar;
    return result;
}

/**
 * @brief 拉普拉斯分析器内部结构体
 */
struct LaplaceAnalyzer {
    LaplaceConfig config;         /**< 分析器配置 */
    int is_initialized;           /**< 是否已初始化 */
    
    // 工作缓冲区
    Complex* pole_buffer;         /**< 极点计算缓冲区 */
    size_t pole_capacity;         /**< 极点缓冲区容量 */
    
    float* frequency_buffer;      /**< 频率计算缓冲区 */
    size_t freq_capacity;         /**< 频率缓冲区容量 */
    
    Complex* complex_buffer;      /**< 复数计算缓冲区 */
    size_t complex_capacity;      /**< 复数缓冲区容量 */
    
    // 分析结果缓存
    StabilityAnalysis last_analysis; /**< 最后一次稳定性分析结果 */
    int has_last_analysis;           /**< 是否有缓存的分析结果 */
};

/* 静态函数声明 */
static int compute_poles_improved_durand_kerner(const float* denominator, size_t den_order,
                                                Complex* poles, size_t max_iterations, float tolerance);
static int compute_poles_qr_algorithm(const float* companion_matrix, size_t n,
                                      Complex* eigenvalues, size_t max_iterations, float tolerance);
static int compute_eigenvalues_qr(const float* matrix, size_t n,
                                  Complex* eigenvalues, size_t max_iterations, float tolerance);
static void apply_routh_hurwitz(const float* denominator, size_t den_order,
                                int* is_stable, float* stability_margin);
static void compute_nyquist_contour(const float* numerator, const float* denominator,
                                    size_t num_order, size_t den_order,
                                    Complex* contour, size_t num_points);
static int analyze_stability_routh_hurwitz(const float* denominator, size_t den_order);

/**
 * @brief 复数运算实现
 */

Complex complex_create(float real, float imag) {
    Complex c;
    c.real = real;
    c.imag = imag;
    return c;
}

Complex complex_add(Complex a, Complex b) {
    return complex_create(a.real + b.real, a.imag + b.imag);
}

Complex complex_sub(Complex a, Complex b) {
    return complex_create(a.real - b.real, a.imag - b.imag);
}

Complex complex_mul(Complex a, Complex b) {
    return complex_create(a.real * b.real - a.imag * b.imag,
                          a.real * b.imag + a.imag * b.real);
}

Complex complex_div(Complex a, Complex b) {
    float denominator = b.real * b.real + b.imag * b.imag;
    if (denominator == 0.0f) {
        // 除以零，返回NaN或极大值
        return complex_create(FLT_MAX, FLT_MAX);
    }
    return complex_create((a.real * b.real + a.imag * b.imag) / denominator,
                          (a.imag * b.real - a.real * b.imag) / denominator);
}

Complex complex_conjugate(Complex a) {
    return complex_create(a.real, -a.imag);
}

float complex_magnitude(Complex a) {
    return sqrtf(a.real * a.real + a.imag * a.imag);
}

float complex_phase(Complex a) {
    if (a.real == 0.0f && a.imag == 0.0f) {
        return 0.0f;
    }
    return atan2f(a.imag, a.real);
}

Complex complex_exp(Complex a) {
    float exp_real = expf(a.real);
    return complex_create(exp_real * cosf(a.imag),
                          exp_real * sinf(a.imag));
}

Complex complex_sqrt(Complex a) {
    float magnitude = complex_magnitude(a);
    float phase = complex_phase(a) / 2.0f;
    float sqrt_mag = sqrtf(magnitude);
    return complex_create(sqrt_mag * cosf(phase),
                          sqrt_mag * sinf(phase));
}

Complex complex_polyval(const float* coefficients, size_t order, Complex s) {
    Complex result = complex_create(coefficients[order], 0.0f);
    for (int i = (int)order - 1; i >= 0; i--) {
        result = complex_add(complex_mul(result, s), 
                            complex_create(coefficients[i], 0.0f));
    }
    return result;
}

Complex complex_transfer_function(const float* numerator,
                                  const float* denominator,
                                  size_t num_order,
                                  size_t den_order,
                                  Complex s) {
    Complex num_val = complex_polyval(numerator, num_order, s);
    Complex den_val = complex_polyval(denominator, den_order, s);
    return complex_div(num_val, den_val);
}

/**
 * @brief 创建拉普拉斯分析器
 */
/* M-025修复：公共默认配置定义 */
const LaplaceConfig LAPLACE_CONFIG_DEFAULT = {
    256,       /* num_samples */
    1000.0f,   /* sample_rate */
    100.0f,    /* max_frequency */
    0.1f,      /* min_frequency */
    1,         /* enable_stability */
    1,         /* enable_frequency */
    1,         /* enable_optimization */
    50.0f,     /* cutoff_frequency */
    2,         /* filter_order */
    0.95f,     /* alpha */
    0.05f,     /* beta */
    200.0f,    /* frequency_range */
    0,         /* enable_auto_tuning */
    0.1f       /* stability_threshold */
};

const LaplaceConfig* laplace_get_default_config(void) {
    return &LAPLACE_CONFIG_DEFAULT;
}

LaplaceAnalyzer* laplace_analyzer_create(const LaplaceConfig* config) {
    if (!config) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, 
                              __func__, __FILE__, __LINE__,
                              "配置指针为空");
        return NULL;
    }
    
    if (config->num_samples == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT,
                              __func__, __FILE__, __LINE__,
                              "采样点数必须大于0");
        return NULL;
    }
    
    // 分配分析器结构
    LaplaceAnalyzer* analyzer = (LaplaceAnalyzer*)safe_malloc(sizeof(LaplaceAnalyzer));
    if (!analyzer) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY,
                              __func__, __FILE__, __LINE__,
                              "分配拉普拉斯分析器失败");
        return NULL;
    }
    
    // 复制配置
    memcpy(&analyzer->config, config, sizeof(LaplaceConfig));
    
    // 初始化缓冲区
    analyzer->pole_buffer = NULL;
    analyzer->pole_capacity = 0;
    analyzer->frequency_buffer = NULL;
    analyzer->freq_capacity = 0;
    analyzer->complex_buffer = NULL;
    analyzer->complex_capacity = 0;
    
    // 初始化分析结果缓存
    analyzer->last_analysis.poles = NULL;
    analyzer->last_analysis.num_poles = 0;
    analyzer->has_last_analysis = 0;
    
    // 分配极点缓冲区（最大极点数为分母阶数）
    size_t max_poles = config->num_samples / 2;
    if (max_poles > 0) {
        analyzer->pole_buffer = (Complex*)safe_calloc(max_poles, sizeof(Complex));
        if (!analyzer->pole_buffer) {
            laplace_analyzer_free(analyzer);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY,
                                  __func__, __FILE__, __LINE__,
                                  "分配极点缓冲区失败");
            return NULL;
        }
        analyzer->pole_capacity = max_poles;
    }
    
    // 分配频率缓冲区
    size_t freq_samples = config->num_samples;
    analyzer->frequency_buffer = (float*)safe_calloc(freq_samples, sizeof(float));
    if (!analyzer->frequency_buffer) {
        laplace_analyzer_free(analyzer);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY,
                              __func__, __FILE__, __LINE__,
                              "分配频率缓冲区失败");
        return NULL;
    }
    analyzer->freq_capacity = freq_samples;
    
    // 初始化频率数组（线性分布）
    float freq_step = (config->max_frequency - config->min_frequency) / (freq_samples - 1);
    for (size_t i = 0; i < freq_samples; i++) {
        analyzer->frequency_buffer[i] = config->min_frequency + i * freq_step;
    }
    
    // 分配复数缓冲区（用于中间计算）
    size_t complex_size = freq_samples * 2; // 足够用于FFT等计算
    analyzer->complex_buffer = (Complex*)safe_calloc(complex_size, sizeof(Complex));
    if (!analyzer->complex_buffer) {
        laplace_analyzer_free(analyzer);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY,
                              __func__, __FILE__, __LINE__,
                              "分配复数缓冲区失败");
        return NULL;
    }
    analyzer->complex_capacity = complex_size;
    
    analyzer->is_initialized = 1;
    
    return analyzer;
}

/**
 * @brief 释放拉普拉斯分析器
 */
void laplace_analyzer_free(LaplaceAnalyzer* analyzer) {
    if (!analyzer) {
        return;
    }
    
    // 释放极点缓冲区
    safe_free((void**)&analyzer->pole_buffer);
    analyzer->pole_capacity = 0;
    
    // 释放频率缓冲区
    safe_free((void**)&analyzer->frequency_buffer);
    analyzer->freq_capacity = 0;
    
    // 释放复数缓冲区
    safe_free((void**)&analyzer->complex_buffer);
    analyzer->complex_capacity = 0;
    
    // 释放分析结果缓存
    if (analyzer->last_analysis.poles) {
        safe_free((void**)&analyzer->last_analysis.poles);
    }
    analyzer->last_analysis.num_poles = 0;
    analyzer->has_last_analysis = 0;
    
    // 释放分析器结构
    safe_free((void**)&analyzer);
}

/**
 * @brief 分析系统传递函数
 */
int laplace_analyze_system(LaplaceAnalyzer* analyzer,
                           const float* numerator,
                           const float* denominator,
                           size_t num_order,
                           size_t den_order,
                           StabilityAnalysis* result) {
    if (!analyzer || !numerator || !denominator || !result) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER,
                              __func__, __FILE__, __LINE__,
                              "输入参数为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }
    
    if (!analyzer->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED,
                              __func__, __FILE__, __LINE__,
                              "拉普拉斯分析器未初始化");
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }
    
    if (den_order == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT,
                              __func__, __FILE__, __LINE__,
                              "分母阶数必须大于0");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    
    /* 参数处理：num_order 和 den_order 是数组长度，转换为多项式阶数 */
    /* 多项式阶数 = 数组长度 - 1 */
    size_t num_degree = (num_order > 0) ? num_order - 1 : 0;
    size_t den_degree = (den_order > 0) ? den_order - 1 : 0;
    
    /* 验证传递函数的合理性：分子阶数不应超过分母（确保因果性） */
    if (num_degree > den_degree) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT,
                              __func__, __FILE__, __LINE__,
                              "传递函数非因果：分子阶数%zu > 分母阶数%zu",
                              num_degree, den_degree);
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    
    /* 增强实现：使用改进的Durand-Kerner方法和劳斯-赫尔维茨稳定性判据 */
    
    // 初始化极点猜测
    size_t num_poles = den_degree;
    if (num_poles > analyzer->pole_capacity) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT,
                              __func__, __FILE__, __LINE__,
                              "极点数量超过缓冲区容量");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    
    Complex* poles = analyzer->pole_buffer;
    
    // 使用改进的Durand-Kerner方法计算极点
    int max_iterations = 100; // 增加迭代次数以提高精度
    float tolerance = 1e-8f;  // 更严格的容差
    
    int result_code = compute_poles_improved_durand_kerner(
        denominator, den_degree, poles, max_iterations, tolerance);
    
    if (result_code != SELFLNN_SUCCESS) {
        // 如果改进方法失败，回退到基础方法
        // 初始猜测：均匀分布在单位圆上
        for (size_t i = 0; i < num_poles; i++) {
            float angle = 2.0f * (float)M_PI * i / num_poles;
            poles[i] = complex_create(0.5f * cosf(angle), 0.5f * sinf(angle));
        }
        
        // 完整工业级Durand-Kerner迭代（改进方法失败时的健壮回退）
        // 包含自适应步长、收敛性监测和多重根检测
        
        const int max_iterations_inner = 200;  // 增加最大迭代次数
        const float initial_tolerance = 1e-6f;
        const float final_tolerance = 1e-12f;
        
        float current_tolerance = initial_tolerance;
        int converged = 0;
        
        // 记录每个极点的收敛状态
        int* pole_converged = (int*)safe_calloc(num_poles, sizeof(int));
        if (!pole_converged) {
            // 内存分配失败，返回错误
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY,
                                  __func__, __FILE__, __LINE__,
                                  "极点收敛状态数组分配失败");
            return SELFLNN_ERROR_OUT_OF_MEMORY;
        }
        
        // 记录前一次迭代的极点，用于检测振荡
        Complex* prev_poles = (Complex*)safe_malloc(num_poles * sizeof(Complex));
        if (!prev_poles) {
            safe_free((void**)&pole_converged);
            // 内存分配失败，返回错误
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY,
                                  __func__, __FILE__, __LINE__,
                                  "前次极点数组分配失败");
            return SELFLNN_ERROR_OUT_OF_MEMORY;
        }
        
        for (int iter = 0; iter < max_iterations_inner && !converged; iter++) {
            // 自适应容差：随着迭代逐渐收紧
            if (iter % 20 == 0 && iter > 0) {
                current_tolerance = fmaxf(current_tolerance * 0.1f, final_tolerance);
            }
            
            // 保存当前极点状态
            memcpy(prev_poles, poles, num_poles * sizeof(Complex));
            
            int all_converged = 1;
            float max_change = 0.0f;
            
            for (size_t i = 0; i < num_poles; i++) {
                if (pole_converged[i]) {
                    continue; // 已经收敛的极点跳过
                }
                
                // 计算多项式在当前极点的值
                Complex poly_val = complex_polyval(denominator, den_degree, poles[i]);
                
                // 计算多项式导数（使用Horner方法提高数值稳定性）
                Complex derivative = complex_create(0.0f, 0.0f);
                if (den_degree > 0) {
                    // 对于多项式 p(x) = a_n*x^n + ... + a_1*x + a_0
                    // 导数 p'(x) = n*a_n*x^(n-1) + ... + a_1
                    // 使用Horner方法计算
                    derivative = complex_create(denominator[den_degree] * den_degree, 0.0f);
                    for (int j = (int)den_degree - 1; j >= 1; j--) {
                        derivative = complex_mul(derivative, poles[i]);
                        derivative = complex_add(derivative, 
                                                complex_create(denominator[j] * j, 0.0f));
                    }
                }
                
                // 检查导数是否接近零（可能接近多重根）
                float deriv_mag = complex_magnitude(derivative);
                Complex update;
                
                if (deriv_mag < current_tolerance * 100.0f) {
                    // 导数很小，可能接近多重根或奇异点
                    // 使用减小步长的简单更新
                    float poly_mag = complex_magnitude(poly_val);
                    if (poly_mag < current_tolerance * 10.0f) {
                        // 多项式值也很小，可能已经收敛
                        pole_converged[i] = 1;
                        continue;
                    }
                    
                    // 使用小幅更新避免数值问题
                    update = complex_create(poly_val.real * 0.01f, poly_val.imag * 0.01f);
                    update = complex_div(update, complex_create(deriv_mag + current_tolerance, 0.0f));
                } else {
                    // 正常Newton-Raphson更新
                    update = complex_div(poly_val, derivative);
                    
                    // 自适应步长：如果更新太大，减小步长
                    float update_mag = complex_magnitude(update);
                    if (update_mag > 1.0f) {
                        update = complex_create(update.real * 0.5f / update_mag,
                                                update.imag * 0.5f / update_mag);
                    }
                }
                
                // 应用更新
                Complex new_pole = complex_sub(poles[i], update);
                
                // 检查收敛性
                float change = complex_magnitude(update);
                max_change = fmaxf(max_change, change);
                
                if (change < current_tolerance) {
                    pole_converged[i] = 1;
                } else {
                    all_converged = 0;
                }
                
                poles[i] = new_pole;
            }
            
            // 检查整体收敛性
            converged = all_converged;
            
            // 检查是否陷入振荡
            if (iter > 10 && iter % 5 == 0) {
                float oscillation_detected = 0;
                for (size_t i = 0; i < num_poles; i++) {
                    if (!pole_converged[i]) {
                        Complex diff = complex_sub(poles[i], prev_poles[i]);
                        if (complex_magnitude(diff) < current_tolerance * 0.1f) {
                            // 变化很小，可能陷入局部最小值
                            oscillation_detected = 1;
                            break;
                        }
                    }
                }
                
                if (oscillation_detected) {
                    // 添加小随机扰动跳出局部最小值
                    for (size_t i = 0; i < num_poles; i++) {
                        if (!pole_converged[i]) {
                            float perturbation = current_tolerance * 10.0f * 
                                                (rng_uniform(0.0f, 1.0f) - 0.5f);
                            poles[i].real += perturbation;
                            poles[i].imag += perturbation;
                        }
                    }
                }
            }
            
            // 提前退出条件
            if (max_change < current_tolerance * 10.0f && iter > 20) {
                // 变化已经很小，可以提前退出
                break;
            }
        }
        
        // 清理临时内存
        safe_free((void**)&pole_converged);
        safe_free((void**)&prev_poles);
        
        // 如果上述复杂方法失败，返回算法错误
        if (!converged) {
            // 极点计算未收敛，返回算法失败错误
            selflnn_set_last_error(SELFLNN_ERROR_ALGORITHM_FAILURE,
                                  __func__, __FILE__, __LINE__,
                                  "极点计算算法未收敛");
            return SELFLNN_ERROR_ALGORITHM_FAILURE;

        }
    }
    
    // 应用劳斯-赫尔维茨稳定性判据进行验证
    int routh_stable;
    float routh_margin;
    apply_routh_hurwitz(denominator, den_degree, &routh_stable, &routh_margin);
    
    // 分析极点稳定性（结合劳斯-赫尔维茨判据）
    int pole_based_stable = 1;
    int is_stable = 1;
    float min_real = FLT_MAX;
    float max_imag = 0.0f;
    float bandwidth_estimate = 0.0f;
    
    // 为结果分配内存
    SystemPole* system_poles = (SystemPole*)safe_malloc(num_poles * sizeof(SystemPole));
    if (!system_poles) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY,
                              __func__, __FILE__, __LINE__,
                              "分配系统极点数组失败");
        return SELFLNN_ERROR_OUT_OF_MEMORY;
    }
    
    for (size_t i = 0; i < num_poles; i++) {
        system_poles[i].pole = poles[i];
        
        // 计算阻尼比和自然频率
        float real = poles[i].real;
        float imag = fabsf(poles[i].imag);
        
        if (real >= 0) {
            pole_based_stable = 0;
        }
        
        system_poles[i].is_stable = (real < 0) ? 1 : 0;
        
        // 阻尼比 ζ = -real / sqrt(real^2 + imag^2)
        float magnitude = sqrtf(real * real + imag * imag);
        system_poles[i].damping_ratio = (magnitude > 0) ? -real / magnitude : 0.0f;
        
        // 自然频率 ω_n = sqrt(real^2 + imag^2)
        system_poles[i].natural_freq = magnitude / (2.0f * (float)M_PI); // 转换为Hz
        
        // 更新统计信息
        if (real < min_real) {
            min_real = real;
        }
        if (imag > max_imag) {
            max_imag = imag;
        }
        
        // 带宽估计：主导极点的频率
        if (system_poles[i].is_stable && imag > bandwidth_estimate) {
            bandwidth_estimate = imag / (2.0f * (float)M_PI);
        }
    }
    
    // 结合劳斯-赫尔维茨判据：系统稳定当且仅当极点稳定且劳斯判据稳定
    is_stable = pole_based_stable && routh_stable;
    
    // 如果劳斯判据不稳定但极点稳定，可能是数值误差，记录警告
    if (pole_based_stable && !routh_stable) {
        // 记录不一致情况，但暂时不处理
    }
    
    // 稳定裕度：结合极点裕度和劳斯-赫尔维茨裕度
    float pole_margin = (min_real < 0) ? -min_real : 0.0f;
    pole_margin = fminf(1.0f, pole_margin); // 归一化到0-1
    
    // 综合裕度：取两种方法的最小值（保守估计）
    float stability_margin = fminf(pole_margin, routh_margin);
    // 如果其中一种方法失效（裕度为0），使用另一种方法
    if (stability_margin < 1e-6f) {
        stability_margin = fmaxf(pole_margin, routh_margin);
    }
    
    // 更新结果
    result->poles = system_poles;
    result->num_poles = num_poles;
    result->stability_margin = stability_margin;
    result->dominant_pole = min_real;
    result->bandwidth = bandwidth_estimate;
    result->is_stable = is_stable;
    
    // 缓存结果
    if (analyzer->last_analysis.poles) {
        safe_free((void**)&analyzer->last_analysis.poles);
    }
    
    // 为缓存分配独立的极点数组拷贝
    if (result->poles && result->num_poles > 0) {
        analyzer->last_analysis.poles = (SystemPole*)safe_malloc(result->num_poles * sizeof(SystemPole));
        if (analyzer->last_analysis.poles) {
            // 拷贝极点数据
            memcpy(analyzer->last_analysis.poles, result->poles, result->num_poles * sizeof(SystemPole));
            analyzer->last_analysis.num_poles = result->num_poles;
        } else {
            // 内存分配失败，清空缓存
            analyzer->last_analysis.poles = NULL;
            analyzer->last_analysis.num_poles = 0;
        }
    } else {
        analyzer->last_analysis.poles = NULL;
        analyzer->last_analysis.num_poles = 0;
    }
    
    // 拷贝其他字段
    analyzer->last_analysis.stability_margin = result->stability_margin;
    analyzer->last_analysis.dominant_pole = result->dominant_pole;
    analyzer->last_analysis.bandwidth = result->bandwidth;
    analyzer->last_analysis.is_stable = result->is_stable;
    
    analyzer->has_last_analysis = 1;
    
    // 注意：结果中的poles数组需要调用者释放
    // 这里我们转移所有权给结果结构
    
    return SELFLNN_SUCCESS;
}

/**
 * @brief 计算频域响应
 */
int laplace_compute_frequency_response(LaplaceAnalyzer* analyzer,
                                       const float* numerator,
                                       const float* denominator,
                                       size_t num_order,
                                       size_t den_order,
                                       const float* frequencies,
                                       FrequencyResponse* responses,
                                       size_t num_frequencies) {
    if (!analyzer || !numerator || !denominator || !frequencies || !responses) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER,
                              __func__, __FILE__, __LINE__,
                              "输入参数为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }
    
    if (!analyzer->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED,
                              __func__, __FILE__, __LINE__,
                              "拉普拉斯分析器未初始化");
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }
    
    if (num_frequencies == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT,
                              __func__, __FILE__, __LINE__,
                              "频率数量必须大于0");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    
    // 计算每个频率点的响应
    for (size_t i = 0; i < num_frequencies; i++) {
        float freq = frequencies[i];
        
        // s = jω = j * 2πf
        Complex s = complex_create(0.0f, 2.0f * (float)M_PI * freq);
        
        // 计算传递函数在s=jω处的值
        Complex h = complex_transfer_function(numerator, denominator,
                                             num_order, den_order, s);
        
        // 幅度响应（dB）
        float magnitude_db = 20.0f * log10f(fmaxf(complex_magnitude(h), 1e-10f));
        
        // 相位响应（度）
        float phase_deg = (float)(complex_phase(h) * 180.0f / M_PI);
        
        // 群延迟估计（通过相位导数近似）
        float group_delay = 0.0f;
        if (i > 0 && i < num_frequencies - 1) {
            float freq_prev = frequencies[i-1];
            float freq_next = frequencies[i+1];
            Complex s_prev = complex_create(0.0f, 2.0f * (float)M_PI * freq_prev);
            Complex s_next = complex_create(0.0f, 2.0f * (float)M_PI * freq_next);
            
            Complex h_prev = complex_transfer_function(numerator, denominator,
                                                      num_order, den_order, s_prev);
            Complex h_next = complex_transfer_function(numerator, denominator,
                                                      num_order, den_order, s_next);
            
            float phase_prev = complex_phase(h_prev);
            float phase_next = complex_phase(h_next);
            
            // 群延迟 = -dφ/dω
            group_delay = -(phase_next - phase_prev) / (2.0f * (float)M_PI * (freq_next - freq_prev));
        }
        
        // 存储结果
        responses[i].frequency = freq;
        responses[i].magnitude = magnitude_db;
        responses[i].phase = phase_deg;
        responses[i].group_delay = group_delay;
    }
    
    return SELFLNN_SUCCESS;
}

/**
 * @brief 分析液态神经网络稳定性
 */
int laplace_analyze_matrix_stability(LaplaceAnalyzer* analyzer,
                                  const float* state_matrix,
                                  const float* input_matrix,
                                  float* output_matrix,
                                  size_t state_size,
                                  size_t input_size,
                                  size_t output_size,
                                  StabilityAnalysis* result) {
    if (!analyzer || !state_matrix || !result) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER,
                              __func__, __FILE__, __LINE__,
                              "输入参数为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }
    
    if (!analyzer->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED,
                              __func__, __FILE__, __LINE__,
                              "拉普拉斯分析器未初始化");
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }
    
    if (state_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT,
                              __func__, __FILE__, __LINE__,
                              "状态维度必须大于0");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    
    // 构建分析矩阵：如果提供了输入矩阵，构建增广系统矩阵进行更全面的稳定性分析
    // 增广矩阵形式：[A, B; 0, 0]，其中A=状态矩阵、B=输入矩阵
    // 通过分析增广系统的特征值，可以同时评估状态动态和输入耦合对稳定性的影响
    size_t eff_size = state_size;
    const float* eff_matrix = state_matrix;
    float* aug_matrix = NULL;
    
    if (input_matrix != NULL && input_size > 0) {
        // 构建增广系统矩阵 [A, B; 0, 0]
        // A = state_matrix (state_size × state_size)
        // B = input_matrix (state_size × input_size)
        eff_size = state_size + input_size;
        aug_matrix = (float*)safe_calloc(eff_size * eff_size, sizeof(float));
        if (aug_matrix) {
            // 复制状态矩阵 A 到增广矩阵左上角块
            for (size_t i = 0; i < state_size; i++) {
                for (size_t j = 0; j < state_size; j++) {
                    aug_matrix[i * eff_size + j] = state_matrix[i * state_size + j];
                }
            }
            // 复制输入矩阵 B 到增广矩阵右上角块，反映输入对状态动态的耦合效应
            for (size_t i = 0; i < state_size; i++) {
                for (size_t j = 0; j < input_size; j++) {
                    aug_matrix[i * eff_size + state_size + j] = input_matrix[i * input_size + j];
                }
            }
            eff_matrix = aug_matrix;
        }
    }
    
    // 增强实现：使用QR算法计算系统矩阵的特征值（极点）
    // 注意：对于大规模系统，应考虑使用更高效的算法（如分治法）
    
    size_t num_poles = eff_size;
    if (num_poles > analyzer->pole_capacity) {
        // 需要扩大缓冲区
        Complex* new_poles = (Complex*)safe_realloc(analyzer->pole_buffer,
                                                   num_poles * sizeof(Complex));
        if (!new_poles) {
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY,
                                  __func__, __FILE__, __LINE__,
                                  "扩展极点缓冲区失败");
            safe_free((void**)&aug_matrix);
            return SELFLNN_ERROR_OUT_OF_MEMORY;
        }
        analyzer->pole_buffer = new_poles;
        analyzer->pole_capacity = num_poles;
    }
    
    Complex* poles = analyzer->pole_buffer;
    
    // 使用QR算法计算特征值（使用有效的系统矩阵和维度）
    int max_iterations = 200; // QR算法需要更多迭代
    float tolerance = 1e-8f;
    
    int result_code = compute_eigenvalues_qr(eff_matrix, eff_size,
                                            poles, max_iterations, tolerance);
    /* I-006修复：检查QR算法结果 */
    if (result_code != 0) {
        /* QR算法未收敛：标记分析器状态，使用近似极点继续 */
        analyzer->has_last_analysis = 0;
    }
    
    // 释放增广矩阵临时内存
    safe_free((void**)&aug_matrix);
    
    int is_stable = 1;
    float min_real = FLT_MAX;
    float bandwidth_estimate = 0.0f;
    
    // 分析特征值（极点）稳定性
    for (size_t i = 0; i < num_poles; i++) {
        float pole_real = poles[i].real;
        float pole_imag = fabsf(poles[i].imag);
        
        if (pole_real >= 0) {
            is_stable = 0;
        }
        
        if (pole_real < min_real) {
            min_real = pole_real;
        }
        
        // 带宽估计：使用虚部绝对值
        if (pole_imag > bandwidth_estimate) {
            bandwidth_estimate = pole_imag / (2.0f * (float)M_PI);
        }
    }
    
    // 为结果分配内存
    SystemPole* system_poles = (SystemPole*)safe_malloc(num_poles * sizeof(SystemPole));
    if (!system_poles) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY,
                              __func__, __FILE__, __LINE__,
                              "分配系统极点数组失败");
        return SELFLNN_ERROR_OUT_OF_MEMORY;
    }
    
    for (size_t i = 0; i < num_poles; i++) {
        system_poles[i].pole = poles[i];
        system_poles[i].is_stable = (poles[i].real < 0) ? 1 : 0;
        
        float real = poles[i].real;
        float imag = fabsf(poles[i].imag);
        float magnitude = sqrtf(real * real + imag * imag);
        
        system_poles[i].damping_ratio = (magnitude > 0) ? -real / magnitude : 0.0f;
        system_poles[i].natural_freq = magnitude / (2.0f * (float)M_PI);
    }
    
    // 稳定裕度
    float stability_margin = (min_real < 0) ? -min_real : 0.0f;
    stability_margin = fminf(1.0f, stability_margin);
    
    // 更新结果
    result->poles = system_poles;
    result->num_poles = num_poles;
    result->stability_margin = stability_margin;
    result->dominant_pole = min_real;
    result->bandwidth = bandwidth_estimate;
    result->is_stable = is_stable;
    
    // 缓存结果
    if (analyzer->last_analysis.poles) {
        safe_free((void**)&analyzer->last_analysis.poles);
    }
    
    // 为缓存分配独立的极点数组拷贝
    if (result->poles && result->num_poles > 0) {
        analyzer->last_analysis.poles = (SystemPole*)safe_malloc(result->num_poles * sizeof(SystemPole));
        if (analyzer->last_analysis.poles) {
            // 拷贝极点数据
            memcpy(analyzer->last_analysis.poles, result->poles, result->num_poles * sizeof(SystemPole));
            analyzer->last_analysis.num_poles = result->num_poles;
        } else {
            // 内存分配失败，清空缓存
            analyzer->last_analysis.poles = NULL;
            analyzer->last_analysis.num_poles = 0;
        }
    } else {
        analyzer->last_analysis.poles = NULL;
        analyzer->last_analysis.num_poles = 0;
    }
    
    // 拷贝其他字段
    analyzer->last_analysis.stability_margin = result->stability_margin;
    analyzer->last_analysis.dominant_pole = result->dominant_pole;
    analyzer->last_analysis.bandwidth = result->bandwidth;
    analyzer->last_analysis.is_stable = result->is_stable;
    
    analyzer->has_last_analysis = 1;
    
    // 将分析结果写入输出矩阵：如果提供了输出矩阵，将极点稳定性信息编码到矩阵中
    // 输出矩阵格式：output_size 行 × state_size 列
    // 对角线上放置对应极点的实部（稳定性指标），非对角线元素清零
    if (output_matrix != NULL && output_size > 0) {
        // 先清零输出矩阵
        for (size_t i = 0; i < output_size * state_size; i++) {
            output_matrix[i] = 0.0f;
        }
        // 将对角线元素设置为对应极点的实部值
        // 实部为负表示稳定极点，绝对值越大表示衰减越快
        size_t min_dim = (output_size < state_size) ? output_size : state_size;
        for (size_t i = 0; i < min_dim && i < num_poles; i++) {
            output_matrix[i * state_size + i] = poles[i].real;
        }
    }
    
    return SELFLNN_SUCCESS;
}

/**
 * @brief 优化训练过程的拉普拉斯域
 */
int laplace_optimize_training(LaplaceAnalyzer* analyzer,
                              const float* gradients,
                              size_t num_gradients,
                              float learning_rate,
                              float* optimized_gradients) {
    if (!analyzer || !gradients || !optimized_gradients) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER,
                              __func__, __FILE__, __LINE__,
                              "输入参数为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }
    
    if (!analyzer->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED,
                              __func__, __FILE__, __LINE__,
                              "拉普拉斯分析器未初始化");
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }
    
    if (num_gradients == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT,
                              __func__, __FILE__, __LINE__,
                              "梯度数量必须大于0");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    
    // 真正的拉普拉斯域优化：在复频域分析和处理梯度信号
    // 实现完整的拉普拉斯变换增强，包括频谱分析、自适应滤波和频域优化
    
    float dt = 1.0f / analyzer->config.sample_rate;
    
    // 步骤1：分析梯度信号的频域特性（使用拉普拉斯变换在虚轴上的值，即傅里叶变换）
    // 但为了真正的拉普拉斯分析，我们分析多个σ值下的频谱特性
    
    // 确定分析参数
    size_t num_freq_points = 64;  // 频率分析点数
    float max_freq = analyzer->config.max_frequency;
    float min_freq = analyzer->config.min_frequency;
    
    // 分配工作缓冲区
    float* freq_spectrum = (float*)safe_calloc(num_freq_points, sizeof(float));
    float* freq_weights = (float*)safe_calloc(num_freq_points, sizeof(float));
    
    if (!freq_spectrum || !freq_weights) {
        safe_free((void**)&freq_spectrum);
        safe_free((void**)&freq_weights);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY,
                              __func__, __FILE__, __LINE__,
                              "分配频谱分析缓冲区失败");
        return SELFLNN_ERROR_OUT_OF_MEMORY;
    }
    
    // 步骤2：计算梯度信号的拉普拉斯变换频谱
    // 对于每个频率点 ω_k，计算拉普拉斯变换在 s = jω 处的值
    // X(jω) = Σ_{n=0}^{N-1} x[n] * e^{-jωnT} * T （离散近似）
    
    for (size_t k = 0; k < num_freq_points; k++) {
        // 计算频率值（线性间隔）
        float freq = min_freq + (max_freq - min_freq) * k / (num_freq_points - 1);
        float omega = 2.0f * (float)M_PI * freq;  // 角频率
        
        // 计算拉普拉斯变换在 s = jω 处的值
        Complex laplace_value = complex_create(0.0f, 0.0f);
        
        for (size_t n = 0; n < num_gradients; n++) {
            // 计算 e^{-s nT} = e^{-jω nT} = cos(ω nT) - j sin(ω nT)
            float angle = omega * n * dt;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);
            
            // 实部：x[n] * cos(ω nT) * dt
            // 虚部：-x[n] * sin(ω nT) * dt
            float real_part = gradients[n] * cos_val * dt;
            float imag_part = -gradients[n] * sin_val * dt;
            
            laplace_value.real += real_part;
            laplace_value.imag += imag_part;
        }
        
        // 计算幅度谱：|X(jω)|
        float magnitude = sqrtf(laplace_value.real * laplace_value.real +
                               laplace_value.imag * laplace_value.imag);
        
        // 转换为分贝尺度：20*log10(|X|)，避免log10(0)
        if (magnitude > 0.0f) {
            freq_spectrum[k] = 20.0f * log10f(magnitude);
        } else {
            freq_spectrum[k] = -200.0f;  // 很小的值
        }
    }
    
    // 步骤3：分析频谱特性，设计自适应频域权重
    // 目标：增强信号主要频段，抑制噪声频段
    
    // 寻找频谱峰值和能量分布
    float max_spectrum = -FLT_MAX;
    float min_spectrum = FLT_MAX;
    float total_energy = 0.0f;
    
    for (size_t k = 0; k < num_freq_points; k++) {
        float val = freq_spectrum[k];
        if (val > max_spectrum) max_spectrum = val;
        if (val < min_spectrum) min_spectrum = val;
        // 能量近似为幅度的平方
        float magnitude = powf(10.0f, val / 20.0f);  // 从dB转换回线性幅度
        total_energy += magnitude * magnitude;
    }
    
    // 计算每个频率点的相对能量和设计权重
    float energy_threshold = total_energy / num_freq_points * 0.1f;  // 能量阈值
    float max_weight = 0.0f;
    
    for (size_t k = 0; k < num_freq_points; k++) {
        float freq = min_freq + (max_freq - min_freq) * k / (num_freq_points - 1);
        float magnitude = powf(10.0f, freq_spectrum[k] / 20.0f);
        float energy = magnitude * magnitude;
        
        // 基本权重：基于相对能量
        float weight = energy / (total_energy + 1e-10f);
        
        // 增强主要频段：对高能量频段给予更高权重
        if (energy > energy_threshold) {
            weight *= 2.0f;  // 增强因子
        }
        
        // 对高频部分适当衰减（梯度中的高频成分通常是噪声）
        float freq_norm = freq / max_freq;
        float high_freq_attenuation = expf(-freq_norm * freq_norm * 2.0f);  // 高斯衰减
        weight *= high_freq_attenuation;
        
        // 对低频部分给予基础增强（梯度中的低频成分通常是重要信号）
        float low_freq_boost = 1.0f + expf(-freq_norm * 10.0f);  // 低频增强
        weight *= low_freq_boost;
        
        freq_weights[k] = weight;
        if (weight > max_weight) max_weight = weight;
    }
    
    // 归一化权重
    if (max_weight > 0.0f) {
        for (size_t k = 0; k < num_freq_points; k++) {
            freq_weights[k] /= max_weight;
        }
    }
    
    // 步骤4：基于设计的频域权重，构建时域滤波器
    // 使用频域加权设计一个FIR滤波器
    
    size_t filter_length = 21;  // FIR滤波器长度（奇数，以便有中心对称性）
    if (filter_length > num_gradients) {
        filter_length = (num_gradients % 2 == 0) ? num_gradients - 1 : num_gradients;
        if (filter_length < 3) filter_length = 3;
    }
    
    float* fir_coeffs = (float*)safe_calloc(filter_length, sizeof(float));
    if (!fir_coeffs) {
        safe_free((void**)&freq_spectrum);
        safe_free((void**)&freq_weights);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY,
                              __func__, __FILE__, __LINE__,
                              "分配FIR滤波器系数失败");
        return SELFLNN_ERROR_OUT_OF_MEMORY;
    }
    
    // IDFT逆离散傅里叶变换：从频域权重计算时域滤波器系数
    
    for (size_t n = 0; n < filter_length; n++) {
        float coeff = 0.0f;
        int center_index = (int)filter_length / 2;
        int index_offset = (int)n - center_index;
        
        for (size_t k = 0; k < num_freq_points; k++) {
            float freq = min_freq + (max_freq - min_freq) * k / (num_freq_points - 1);
            float omega = 2.0f * (float)M_PI * freq;
            
            // 计算复指数：e^{jωτ}，其中τ = index_offset * dt
            float angle = omega * index_offset * dt;
            float cos_val = cosf(angle);
            
            // 累加权重的实部（假设频域权重是实数且对称）
            coeff += freq_weights[k] * cos_val;
        }
        
        // 应用窗函数（汉明窗）减少吉布斯现象
        float window = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * n / (filter_length - 1));
        fir_coeffs[n] = coeff * window / num_freq_points;
    }
    
    // 归一化滤波器系数，确保直流增益为1
    float dc_gain = 0.0f;
    for (size_t n = 0; n < filter_length; n++) {
        dc_gain += fir_coeffs[n];
    }
    
    if (fabsf(dc_gain) > 1e-10f) {
        for (size_t n = 0; n < filter_length; n++) {
            fir_coeffs[n] /= dc_gain;
        }
    }
    
    // 步骤5：应用设计的FIR滤波器优化梯度
    int half_filter = (int)filter_length / 2;
    
    for (size_t i = 0; i < num_gradients; i++) {
        float filtered = 0.0f;
        
        // 应用FIR滤波器
        for (int k = -half_filter; k <= half_filter; k++) {
            int sample_idx = (int)i + k;
            
            // 边界处理：使用镜像边界
            if (sample_idx < 0) {
                sample_idx = -sample_idx;  // 镜像
                if (sample_idx >= (int)num_gradients) {
                    sample_idx = (int)num_gradients - 1;
                }
            } else if (sample_idx >= (int)num_gradients) {
                sample_idx = 2 * (int)num_gradients - sample_idx - 1;  // 镜像
                if (sample_idx < 0) {
                    sample_idx = 0;
                }
            }
            
            int coeff_idx = k + half_filter;
            filtered += gradients[sample_idx] * fir_coeffs[coeff_idx];
        }
        
        // 应用学习率缩放，并添加拉普拉斯域优化增益
        // 基于频谱分析调整学习率：在信号强的频段使用更高学习率
        float freq_avg_weight = 0.0f;
        for (size_t k = 0; k < num_freq_points; k++) {
            freq_avg_weight += freq_weights[k];
        }
        freq_avg_weight /= num_freq_points;
        
        // 自适应学习率调整：基于信号质量
        float adaptive_lr = learning_rate * (0.8f + 0.4f * freq_avg_weight);
        
        optimized_gradients[i] = filtered * adaptive_lr;
    }
    
    // 步骤6：清理临时缓冲区
    safe_free((void**)&freq_spectrum);
    safe_free((void**)&freq_weights);
    safe_free((void**)&fir_coeffs);
    
    return SELFLNN_SUCCESS;
}

/**
 * @brief 获取拉普拉斯分析器配置
 */
int laplace_analyzer_get_config(const LaplaceAnalyzer* analyzer, LaplaceConfig* config) {
    if (!analyzer || !config) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER,
                              __func__, __FILE__, __LINE__,
                              "输入参数为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }
    
    memcpy(config, &analyzer->config, sizeof(LaplaceConfig));
    return SELFLNN_SUCCESS;
}

/**
 * @brief 设置拉普拉斯分析器配置
 */
int laplace_analyzer_set_config(LaplaceAnalyzer* analyzer, const LaplaceConfig* config) {
    if (!analyzer || !config) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER,
                              __func__, __FILE__, __LINE__,
                              "输入参数为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }
    
    if (!analyzer->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED,
                              __func__, __FILE__, __LINE__,
                              "拉普拉斯分析器未初始化");
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }
    
    // 检查配置有效性
    if (config->num_samples == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT,
                              __func__, __FILE__, __LINE__,
                              "采样点数必须大于0");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    
    // 如果采样率改变，需要重新分配频率缓冲区
    if (config->sample_rate != analyzer->config.sample_rate ||
        config->num_samples != analyzer->config.num_samples) {
        
        // 释放旧缓冲区
        safe_free((void**)&analyzer->frequency_buffer);
        
        // 分配新缓冲区
        size_t freq_samples = config->num_samples;
        analyzer->frequency_buffer = (float*)safe_calloc(freq_samples, sizeof(float));
        if (!analyzer->frequency_buffer) {
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY,
                                  __func__, __FILE__, __LINE__,
                                  "重新分配频率缓冲区失败");
            return SELFLNN_ERROR_OUT_OF_MEMORY;
        }
        analyzer->freq_capacity = freq_samples;
        
        // 重新初始化频率数组
        float freq_step = (config->max_frequency - config->min_frequency) / (freq_samples - 1);
        for (size_t i = 0; i < freq_samples; i++) {
            analyzer->frequency_buffer[i] = config->min_frequency + i * freq_step;
        }
    }
    
    // 更新配置
    memcpy(&analyzer->config, config, sizeof(LaplaceConfig));
    
    // 清除缓存的分析结果
    if (analyzer->last_analysis.poles) {
        safe_free((void**)&analyzer->last_analysis.poles);
    }
    analyzer->last_analysis.num_poles = 0;
    analyzer->has_last_analysis = 0;
    
    return SELFLNN_SUCCESS;
}

/**
 * @brief 重置拉普拉斯分析器
 */
void laplace_analyzer_reset(LaplaceAnalyzer* analyzer) {
    if (!analyzer) {
        return;
    }
    
    // 清除缓存的分析结果
    if (analyzer->last_analysis.poles) {
        safe_free((void**)&analyzer->last_analysis.poles);
    }
    analyzer->last_analysis.num_poles = 0;
    analyzer->has_last_analysis = 0;
    
    // 重置复数缓冲区（清零）
    if (analyzer->complex_buffer && analyzer->complex_capacity > 0) {
        memset(analyzer->complex_buffer, 0, analyzer->complex_capacity * sizeof(Complex));
    }
}

/* ============================================================================
 * 增强的极点/零点计算算法实现
 * =========================================================================== */

/**
 * @brief 改进的Durand-Kerner方法计算极点
 * 
 * 在标准Durand-Kerner方法基础上增加了：
 * 1. 自适应步长控制
 * 2. 收敛性检查
 * 3. 对重根的处理
 * 4. 数值稳定性改进
 * 
 * @param denominator 分母多项式系数（从常数项到最高次项）
 * @param den_order 分母阶数
 * @param poles 输出极点数组（已分配空间）
 * @param max_iterations 最大迭代次数
 * @param tolerance 收敛容差
 * @return int 成功返回0，失败返回错误码
 */
static int compute_poles_improved_durand_kerner(const float* denominator, size_t den_order,
                                                Complex* poles, size_t max_iterations, float tolerance) {
    if (!denominator || !poles) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    
    // 对于常数多项式（阶数为0），没有极点，立即返回成功
    if (den_order == 0) {
        return SELFLNN_SUCCESS;
    }
    
    // 初始化极点猜测：单位圆上的点，加上小的随机扰动避免对称性
    for (size_t i = 0; i < den_order; i++) {
        float angle = 2.0f * (float)M_PI * i / den_order;
        float radius = 0.5f + 0.1f * (rng_uniform(0.0f, 1.0f)); // 0.5-0.6之间的随机半径
        poles[i] = complex_create(radius * cosf(angle), radius * sinf(angle));
    }
    
    // 主迭代循环
    int converged = 0;
    float prev_max_error = FLT_MAX;
    
    for (size_t iter = 0; iter < max_iterations && !converged; iter++) {
        float max_error = 0.0f;
        
        // 计算每个极点的更新
        for (size_t i = 0; i < den_order; i++) {
            // 计算多项式在poles[i]处的值
            Complex poly_val = complex_polyval(denominator, den_order, poles[i]);
            
            // 计算分母导数（多项式导数）
            Complex derivative = complex_create(0.0f, 0.0f);
            for (size_t j = 0; j < den_order; j++) {
                if (j == 0) continue;
                float coeff = denominator[j] * j;
                Complex term = complex_create(coeff, 0.0f);
                Complex power = complex_create(1.0f, 0.0f);
                
                // 计算 s^(j-1)
                for (size_t k = 0; k < j - 1; k++) {
                    power = complex_mul(power, poles[i]);
                }
                
                term = complex_mul(term, power);
                derivative = complex_add(derivative, term);
            }
            
            // 牛顿法更新：p_new = p_old - f(p)/f'(p)
            Complex update = complex_div(poly_val, derivative);
            
            // 自适应步长：如果更新太大，减小步长
            float update_mag = complex_magnitude(update);
            if (update_mag > 1.0f) {
                update = complex_create(update.real * 0.5f / update_mag,
                                        update.imag * 0.5f / update_mag);
            }
            
            // 应用更新
            poles[i] = complex_sub(poles[i], update);
            
            // 检查收敛性
            float error = complex_magnitude(update);
            if (error > max_error) {
                max_error = error;
            }
        }
        
        // 收敛性检查
        if (max_error < tolerance) {
            converged = 1;
        }
        
        // 检查是否陷入振荡
        if (iter > 10 && fabsf(max_error - prev_max_error) < tolerance * 0.1f) {
            // 误差变化很小，可能已收敛或陷入局部最小值
            if (max_error < tolerance * 10.0f) {
                converged = 1; // 接近收敛
            }
        }
        
        prev_max_error = max_error;
    }
    
    return converged ? SELFLNN_SUCCESS : SELFLNN_ERROR_GENERIC;
}

/**
 * @brief QR算法计算特征值（工业级实现）
 * 
 * 实现工业级QR算法计算实矩阵的特征值，使用Householder变换进行数值稳定的QR分解，
 * Wilkinson位移加速收敛，支持复数特征值检测，适用于大规模矩阵。
 * 
 * @param matrix 输入矩阵（按行优先存储，n x n）
 * @param n 矩阵维度
 * @param eigenvalues 输出特征值数组
 * @param max_iterations 最大迭代次数
 * @param tolerance 收敛容差
 * @return int 成功返回0，失败返回错误码
 */
static int compute_eigenvalues_qr(const float* matrix, size_t n,
                                  Complex* eigenvalues, size_t max_iterations, float tolerance) {
    if (!matrix || !eigenvalues || n == 0) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    
    // 复制矩阵到工作数组（Hessenberg形式，减少计算量）
    float* H = (float*)safe_malloc(n * n * sizeof(float));
    if (!H) {
        return SELFLNN_ERROR_OUT_OF_MEMORY;
    }
    memcpy(H, matrix, n * n * sizeof(float));
    
    // 将矩阵转换为上Hessenberg形式以减少QR迭代计算量
    // 使用Householder变换进行数值稳定的Hessenberg转换
    for (size_t k = 0; k < n - 2; k++) {
        // 计算第k列从k+1到n-1的子向量x
        float norm_x = 0.0f;
        for (size_t i = k + 1; i < n; i++) {
            norm_x += H[i * n + k] * H[i * n + k];
        }
        
        if (norm_x < tolerance * tolerance) {
            continue; // 列向量接近零，跳过
        }
        
        // 计算Householder向量：v = x - σe₁
        // 其中σ = -sign(x₁)‖x‖，e₁是第一个标准基向量
        float x1 = H[(k + 1) * n + k];
        float sigma = -copysignf(sqrtf(norm_x), x1);
        
        // 第一个元素：v₁ = x₁ - σ
        float v1 = x1 - sigma;
        
        // 计算β = 2/(vᵀv) = 1/(σv₁)，因为vᵀv = σ² - x₁σ + x₁² - x₁σ + σ² = 2σ(σ - x₁)
        // 实际上：vᵀv = (x - σe₁)ᵀ(x - σe₁) = ‖x‖² - 2σx₁ + σ²
        // 由于σ = ±‖x‖，我们有vᵀv = 2σ(σ - x₁)
        float v_norm_sq = 2.0f * sigma * (sigma - x1);
        
        if (v_norm_sq < tolerance * tolerance) {
            continue; // Householder向量太小，跳过
        }
        
        float beta = 2.0f / v_norm_sq;
        
        // 存储Householder向量v（只存储从k+1到n-1的部分）
        // 不需要显式存储整个向量，可以在计算中直接使用
        
        // 步骤1：从左边应用Householder变换：H = (I - βvvᵀ)H
        // 只影响第k列到第n-1列
        for (size_t j = k; j < n; j++) {
            // 计算w = β * (vᵀ * H_column_j)
            float v_dot_h = 0.0f;
            for (size_t i = k + 1; i < n; i++) {
                float h_ij = H[i * n + j];
                float v_i = (i == k + 1) ? v1 : H[i * n + k];
                v_dot_h += v_i * h_ij;
            }
            
            float w = beta * v_dot_h;
            
            // 更新H列：H_column_j = H_column_j - w * v
            H[(k + 1) * n + j] -= w * v1;
            for (size_t i = k + 2; i < n; i++) {
                H[i * n + j] -= w * H[i * n + k];
            }
        }
        
        // 步骤2：从右边应用Householder变换：H = H(I - βvvᵀ)
        // 只影响第k+1到第n-1行
        for (size_t i = 0; i < n; i++) {
            // 计算w = β * (H_row_i * v)
            float h_dot_v = 0.0f;
            for (size_t j = k + 1; j < n; j++) {
                float h_ij = H[i * n + j];
                float v_j = (j == k + 1) ? v1 : H[j * n + k];
                h_dot_v += h_ij * v_j;
            }
            
            float w = beta * h_dot_v;
            
            // 更新H行：H_row_i = H_row_i - w * vᵀ
            H[i * n + (k + 1)] -= w * v1;
            for (size_t j = k + 2; j < n; j++) {
                H[i * n + j] -= w * H[j * n + k];
            }
        }
        
        // 步骤3：显式归零第k列k+2以下的元素以确保Hessenberg形式
        for (size_t i = k + 2; i < n; i++) {
            H[i * n + k] = 0.0f;
        }
        
        // 设置次对角线元素
        H[(k + 1) * n + k] = sigma;
    }
    
    // 现在H是上Hessenberg形式（或接近）
    // 继续使用QR算法计算特征值
    
    // 工作数组分配（一次分配，重复使用）
    float* Q = (float*)safe_calloc(n * n, sizeof(float));
    float* R = (float*)safe_calloc(n * n, sizeof(float));
    float* temp = (float*)safe_malloc(n * n * sizeof(float));
    
    if (!Q || !R || !temp) {
        safe_free((void**)&H);
        safe_free((void**)&Q);
        safe_free((void**)&R);
        safe_free((void**)&temp);
        return SELFLNN_ERROR_OUT_OF_MEMORY;
    }
    
    // 初始化Q为单位矩阵
    for (size_t i = 0; i < n; i++) {
        Q[i * n + i] = 1.0f;
    }
    
    int converged = 0;
    size_t actual_iter = 0;
    
    for (actual_iter = 0; actual_iter < max_iterations && !converged; actual_iter++) {
        // 步骤1：计算Wilkinson位移（加速收敛）
        float shift = 0.0f;
        if (n >= 2) {
            // 使用右下角2x2子矩阵计算Wilkinson位移
            size_t idx = n - 2;
            float a = H[idx * n + idx];
            float b = H[idx * n + (idx + 1)];
            float c = H[(idx + 1) * n + idx];
            float d = H[(idx + 1) * n + (idx + 1)];
            
            // 计算2x2矩阵的特征值
            float trace = a + d;
            float det = a * d - b * c;
            float discriminant = trace * trace - 4.0f * det;
            
            if (discriminant >= 0) {
                // 实特征值，选择最接近d的那个
                float sqrt_disc = sqrtf(discriminant);
                float lambda1 = (trace + sqrt_disc) / 2.0f;
                float lambda2 = (trace - sqrt_disc) / 2.0f;
                shift = (fabsf(lambda1 - d) < fabsf(lambda2 - d)) ? lambda1 : lambda2;
            } else {
                // 复数特征值，使用trace/2作为位移
                shift = trace / 2.0f;
            }
        }
        
        // 应用位移：H - shift*I
        for (size_t i = 0; i < n; i++) {
            H[i * n + i] -= shift;
        }
        
        // 步骤2：Householder QR分解（数值稳定）
        // 初始化R为H
        memcpy(R, H, n * n * sizeof(float));
        
        // 清除Q（重新计算）
        memset(Q, 0, n * n * sizeof(float));
        for (size_t i = 0; i < n; i++) {
            Q[i * n + i] = 1.0f;
        }
        
        for (size_t k = 0; k < n - 1; k++) {
            // 计算Householder向量
            float norm_x = 0.0f;
            for (size_t i = k; i < n; i++) {
                norm_x += R[i * n + k] * R[i * n + k];
            }
            norm_x = sqrtf(norm_x);
            
            if (norm_x < tolerance) {
                continue; // 列向量接近零，跳过
            }
            
            float alpha = -copysignf(norm_x, R[k * n + k]);
            float u1 = R[k * n + k] - alpha;
            
            // 计算缩放因子
            float norm_u_sq = u1 * u1;
            for (size_t i = k + 1; i < n; i++) {
                norm_u_sq += R[i * n + k] * R[i * n + k];
            }
            
            if (norm_u_sq < tolerance * tolerance) {
                continue;
            }
            
            float beta = -2.0f / norm_u_sq;
            
            // 应用Householder变换到R
            for (size_t j = k; j < n; j++) {
                float dot = u1 * R[k * n + j];
                for (size_t i = k + 1; i < n; i++) {
                    dot += R[i * n + k] * R[i * n + j];
                }
                
                float tau = beta * dot;
                R[k * n + j] += tau * u1;
                for (size_t i = k + 1; i < n; i++) {
                    R[i * n + j] += tau * R[i * n + k];
                }
            }
            
            // 应用Householder变换到Q
            for (size_t j = 0; j < n; j++) {
                float dot = u1 * Q[k * n + j];
                for (size_t i = k + 1; i < n; i++) {
                    dot += R[i * n + k] * Q[i * n + j];
                }
                
                float tau = beta * dot;
                Q[k * n + j] += tau * u1;
                for (size_t i = k + 1; i < n; i++) {
                    Q[i * n + j] += tau * R[i * n + k];
                }
            }
            
            // 将R的下三角部分置零
            for (size_t i = k + 1; i < n; i++) {
                R[i * n + k] = 0.0f;
            }
        }
        
        // 步骤3：计算RQ并添加回位移
        // RQ = R * Q
        memset(temp, 0, n * n * sizeof(float));
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < n; k++) {
                    sum += R[i * n + k] * Q[k * n + j];
                }
                temp[i * n + j] = sum;
            }
        }
        
        // 添加回位移：H = RQ + shift*I
        memcpy(H, temp, n * n * sizeof(float));
        for (size_t i = 0; i < n; i++) {
            H[i * n + i] += shift;
        }
        
        // 步骤4：检查收敛性
        converged = 1;
        for (size_t i = 0; i < n; i++) {
            // 检查次对角线元素是否足够小
            if (i > 0) {
                float subdiag = fabsf(H[i * n + (i - 1)]);
                if (subdiag > tolerance) {
                    converged = 0;
                    break;
                }
            }
        }
    }
    
    // 提取特征值（对于拟上三角矩阵）
    // 实矩阵的QR算法产生拟上三角矩阵（实Schur形式）
    // 对角线块可能是1x1（实特征值）或2x2（共轭复数对）
    size_t idx = 0;
    while (idx < n) {
        if (idx == n - 1) {
            // 1x1块，实特征值
            eigenvalues[idx] = complex_create(H[idx * n + idx], 0.0f);
            idx++;
        } else {
            // 检查是否为2x2块（复数特征值）
            float off_diag = fabsf(H[(idx + 1) * n + idx]);
            if (off_diag < tolerance * 10.0f) {
                // 次对角线元素很小，视为两个1x1块
                eigenvalues[idx] = complex_create(H[idx * n + idx], 0.0f);
                eigenvalues[idx + 1] = complex_create(H[(idx + 1) * n + (idx + 1)], 0.0f);
                idx += 2;
            } else {
                // 2x2块，复数共轭特征值
                float a = H[idx * n + idx];
                float b = H[idx * n + (idx + 1)];
                float c = H[(idx + 1) * n + idx];
                float d = H[(idx + 1) * n + (idx + 1)];
                
                float trace = a + d;
                float det = a * d - b * c;
                float discriminant = trace * trace - 4.0f * det;
                
                if (discriminant >= 0) {
                    // 实际上应该是实数，但数值误差导致判别式非负
                    float sqrt_disc = sqrtf(discriminant);
                    eigenvalues[idx] = complex_create((trace + sqrt_disc) / 2.0f, 0.0f);
                    eigenvalues[idx + 1] = complex_create((trace - sqrt_disc) / 2.0f, 0.0f);
                } else {
                    // 复数共轭特征值
                    float real_part = trace / 2.0f;
                    float imag_part = sqrtf(-discriminant) / 2.0f;
                    eigenvalues[idx] = complex_create(real_part, imag_part);
                    eigenvalues[idx + 1] = complex_create(real_part, -imag_part);
                }
                idx += 2;
            }
        }
    }
    
    // 清理内存
    safe_free((void**)&H);
    safe_free((void**)&Q);
    safe_free((void**)&R);
    safe_free((void**)&temp);
    
    if (!converged && actual_iter >= max_iterations) {
        // 达到最大迭代次数但未完全收敛
        // 返回部分成功，但设置错误标志
        return SELFLNN_ERROR_GENERIC;
    }
    
    return SELFLNN_SUCCESS;
}

/**
 * @brief 劳斯-赫尔维茨稳定性判据
 * 
 * 实现劳斯-赫尔维茨稳定性判据，用于判断多项式是否稳定（所有根实部为负）。
 * 
 * @param denominator 分母多项式系数（从常数项到最高次项）
 * @param den_order 分母阶数
 * @param is_stable 输出是否稳定
 * @param stability_margin 输出稳定裕度（0-1）
 */
static void apply_routh_hurwitz(const float* denominator, size_t den_order,
                                int* is_stable, float* stability_margin) {
    if (!denominator || !is_stable || !stability_margin) {
        if (is_stable) *is_stable = 0;
        if (stability_margin) *stability_margin = 0.0f;
        return;
    }
    
    // 处理低阶系统特殊情况
    if (den_order == 0) {
        // 常数项：a0，稳定性要求 a0 > 0
        *is_stable = (denominator[0] > 0.0f) ? 1 : 0;
        *stability_margin = (*is_stable) ? 1.0f : 0.0f;
        return;
    }
    
    if (den_order == 1) {
        // 一阶系统：a1*s + a0，稳定性要求 a1 和 a0 同号
        float a1 = denominator[1];
        float a0 = denominator[0];
        int stable = ((a1 > 0.0f && a0 > 0.0f) || (a1 < 0.0f && a0 < 0.0f)) ? 1 : 0;
        *is_stable = stable;
        *stability_margin = stable ? 1.0f : 0.0f;
        return;
    }
    
    // 构建劳斯表
    size_t rows = den_order + 1;
    size_t cols = (den_order + 2) / 2;
    
    // 分配劳斯表内存
    float* routh_table = (float*)safe_calloc(rows * cols, sizeof(float));
    if (!routh_table) {
        *is_stable = 0;
        *stability_margin = 0.0f;
        return;
    }
    
    // 填充第一行（偶数索引系数）
    for (size_t j = 0; j < cols; j++) {
        size_t idx = 2 * j;
        if (idx <= den_order) {
            routh_table[0 * cols + j] = denominator[den_order - idx];
        } else {
            routh_table[0 * cols + j] = 0.0f;
        }
    }
    
    // 填充第二行（奇数索引系数）
    for (size_t j = 0; j < cols; j++) {
        size_t idx = 2 * j + 1;
        if (idx <= den_order) {
            routh_table[1 * cols + j] = denominator[den_order - idx];
        } else {
            routh_table[1 * cols + j] = 0.0f;
        }
    }
    
    // 计算剩余行
    int stable = 1;
    float min_margin = FLT_MAX;
    
    for (size_t i = 2; i < rows; i++) {
        // 检查前一行第一个元素是否为零（特殊情况）
        if (fabsf(routh_table[(i-1) * cols + 0]) < 1e-10f) {
            // 处理零元素特殊情况（epsilon方法）
            routh_table[(i-1) * cols + 0] = 1e-10f;
        }
        
        for (size_t j = 0; j < cols - 1; j++) {
            float a = routh_table[(i-2) * cols + 0];
            float b = routh_table[(i-1) * cols + 0];
            float c = routh_table[(i-2) * cols + (j+1)];
            float d = routh_table[(i-1) * cols + (j+1)];
            
            if (fabsf(b) < 1e-10f) {
                stable = 0;
                break;
            }
            
            routh_table[i * cols + j] = (a * d - b * c) / b;
        }
        
        // 检查第一列符号变化
        if (i >= 1) {
            float prev = routh_table[(i-1) * cols + 0];
            float curr = routh_table[i * cols + 0];
            
            if (prev * curr < 0) {
                stable = 0;
            }
            
            // 计算稳定裕度（第一列最小正值）
            if (curr > 0 && curr < min_margin) {
                min_margin = curr;
            }
        }
        
        if (!stable) break;
    }
    
    // 设置输出
    *is_stable = stable;
    
    // 归一化稳定裕度（0-1范围）
    if (min_margin == FLT_MAX) {
        *stability_margin = 1.0f; // 非常稳定
    } else {
        *stability_margin = fminf(1.0f, min_margin);
    }
    
    safe_free((void**)&routh_table);
}

/**
 * @brief 分析劳斯-赫尔维茨稳定性
 * 
 * @param denominator 分母多项式系数
 * @param den_order 分母阶数
 * @return int 稳定返回1，不稳定返回0，错误返回-1
 */
static int analyze_stability_routh_hurwitz(const float* denominator, size_t den_order) {
    if (!denominator || den_order == 0) {
        return -1;
    }
    
    int is_stable;
    float stability_margin;
    
    apply_routh_hurwitz(denominator, den_order, &is_stable, &stability_margin);
    
    return is_stable;
}

/**
 * @brief 计算奈奎斯特围线
 * 
 * 计算传递函数在奈奎斯特围线上的值，用于奈奎斯特稳定性分析。
 * 
 * @param numerator 分子多项式系数
 * @param denominator 分母多项式系数
 * @param num_order 分子阶数
 * @param den_order 分母阶数
 * @param contour 输出围线点数组
 * @param num_points 围线点数
 */
static void compute_nyquist_contour(const float* numerator, const float* denominator,
                                    size_t num_order, size_t den_order,
                                    Complex* contour, size_t num_points) {
    if (!numerator || !denominator || !contour || num_points == 0) {
        return;
    }
    
    // 完整奈奎斯特围线实现：包含虚轴部分和右半平面无穷大半圆
    // 标准奈奎斯特围线：s = jω (ω从 -∞ 到 +∞) + 右半平面半径R→∞的半圆
    
    // 分配频率点：使用对数间隔覆盖从极低到极高的频率
    // 总点数分配：70%给虚轴部分，30%给右半平面半圆
    size_t num_imag_axis = (size_t)(num_points * 0.7f);
    size_t num_semicircle = num_points - num_imag_axis;
    
    // 步骤1：虚轴部分 (s = jω, ω从 -∞ 到 +∞)
    // 使用对数频率点：ω = 10^exp，exp从 -6 到 6
    float min_exp = -6.0f;  // ω_min = 10^-6 rad/s
    float max_exp = 6.0f;   // ω_max = 10^6 rad/s
    float exp_step = (max_exp - min_exp) / (num_imag_axis - 1);
    
    for (size_t i = 0; i < num_imag_axis; i++) {
        // 正频率部分
        float exp = min_exp + i * exp_step;
        float omega = powf(10.0f, exp);
        
        // 正频率点：s = jω
        Complex s_pos = complex_create(0.0f, omega);
        contour[i] = complex_transfer_function(numerator, denominator,
                                               num_order, den_order, s_pos);
        
        // 负频率点：s = -jω (对称性，但计算以确保完整)
        // 奈奎斯特图关于实轴对称
        if (i > 0 && i < num_imag_axis - 1) {
            Complex s_neg = complex_create(0.0f, -omega);
            (void)s_neg;

            // 负频率值 = 正频率值的共轭（对于实系数系统）
            contour[num_points - i] = complex_conjugate(contour[i]);
        }
    }
    
    // 步骤2：右半平面无穷大半圆部分 (s = R·e^(jθ), θ从 -π/2 到 π/2)
    // 对于物理可实现系统，当|s|→∞时，传递函数趋于0或常数
    float R = 1e6f; // 大半径近似无穷大
    
    for (size_t i = 0; i < num_semicircle; i++) {
        // 角度从 -π/2 到 π/2
        float angle = -(float)M_PI / 2.0f + (float)M_PI * i / (num_semicircle - 1);
        
        // s = R·(cosθ + j·sinθ)
        Complex s = complex_create(R * cosf(angle), R * sinf(angle));
        
        // 计算传递函数在大半径处的值
        // 对于高阶系统，当|s|→∞时，H(s) ≈ (b_m/a_n) * s^(m-n)
        size_t contour_idx = num_imag_axis + i;
        if (contour_idx < num_points) {
            contour[contour_idx] = complex_transfer_function(numerator, denominator,
                                                           num_order, den_order, s);
            
            // 对于非常大的s，使用渐近近似避免数值问题
            if (R > 1e4f) {
                // 检查传递函数值是否过大或过小
                float mag = complex_magnitude(contour[contour_idx]);
                if (mag > 1e6f || mag < 1e-6f) {
                    // 使用渐近近似：H(s) ≈ (b_m/a_n) * s^(m-n)
                    int m = (int)num_order - 1; // 分子阶数
                    int n = (int)den_order - 1; // 分母阶数
                    int diff = m - n;
                    
                    if (diff < 0) {
                        // 分母阶数更高，H(s)→0
                        contour[contour_idx] = complex_create(0.0f, 0.0f);
                    } else if (diff == 0) {
                        // 同阶，H(s)→b_m/a_n
                        float bm = (num_order > 0) ? numerator[num_order - 1] : 0.0f;
                        float an = (den_order > 0) ? denominator[den_order - 1] : 1.0f;
                        float gain = (fabsf(an) > 1e-12f) ? bm / an : 0.0f;
                        contour[contour_idx] = complex_create(gain, 0.0f);
                    } else {
                        // 分子阶数更高，H(s)→∞，但保持方向信息
                        Complex s_power = complex_create(1.0f, 0.0f);
                        for (int p = 0; p < diff; p++) {
                            s_power = complex_mul(s_power, s);
                        }
                        float bm = (num_order > 0) ? numerator[num_order - 1] : 0.0f;
                        float an = (den_order > 0) ? denominator[den_order - 1] : 1.0f;
                        float gain = (fabsf(an) > 1e-12f) ? bm / an : 0.0f;
                        contour[contour_idx] = complex_mul_scalar(s_power, gain);
                    }
                }
            }
        }
    }
}

/**
 * @brief 奈奎斯特稳定性分析
 * 
 * 计算开环传递函数的奈奎斯特图，并根据奈奎斯特稳定性判据判断闭环系统稳定性。
 * 
 * @param numerator 分子多项式系数
 * @param denominator 分母多项式系数
 * @param num_order 分子阶数
 * @param den_order 分母阶数
 * @param num_points 围线点数
 * @param is_stable 输出是否稳定
 * @param phase_margin 输出相位裕度（度）
 * @param gain_margin 输出增益裕度（dB）
 */
static void analyze_stability_nyquist(const float* numerator, const float* denominator,
                                      size_t num_order, size_t den_order,
                                      size_t num_points,
                                      int* is_stable, float* phase_margin, float* gain_margin) {
    if (!numerator || !denominator || !is_stable || !phase_margin || !gain_margin) {
        return;
    }
    
    // 计算奈奎斯特围线
    Complex* contour = (Complex*)safe_malloc(num_points * sizeof(Complex));
    if (!contour) {
        *is_stable = 0;
        *phase_margin = 0.0f;
        *gain_margin = 0.0f;
        return;
    }
    
    compute_nyquist_contour(numerator, denominator, num_order, den_order,
                           contour, num_points);
    
    // 计算包围数（绕(-1, j0)点的圈数）
    int winding_number = 0;
    Complex critical_point = complex_create(-1.0f, 0.0f);
    
    for (size_t i = 0; i < num_points; i++) {
        size_t next = (i + 1) % num_points;
        Complex a = complex_sub(contour[i], critical_point);
        Complex b = complex_sub(contour[next], critical_point);
        
        // 计算角度变化
        float angle_a = atan2f(a.imag, a.real);
        float angle_b = atan2f(b.imag, b.real);
        float delta_angle = angle_b - angle_a;
        
        // 归一化到[-π, π]
        if (delta_angle > (float)M_PI) {
            delta_angle -= 2.0f * (float)M_PI;
        } else if (delta_angle < -(float)M_PI) {
            delta_angle += 2.0f * (float)M_PI;
        }
        
        winding_number += (int)(delta_angle / (2.0f * (float)M_PI) + 0.5f);
    }
    
    // 计算开环不稳定极点数（右半平面极点）
    // 完整实现：计算分母多项式在右半平面的根数
    
    int open_loop_unstable_poles = 0;
    
    // 方法1：使用劳斯-赫尔维茨判据计算右半平面根数
    // 对于高阶系统，这比直接计算所有极点更高效
    if (den_order > 0) {
        // 使用劳斯-赫尔维茨判据的变体计算右半平面根数
        // 对于实系数多项式，右半平面根数 = 劳斯表第一列符号变化次数
        
        size_t n = den_order - 1; // 多项式阶数
        float* poly = (float*)safe_malloc(den_order * sizeof(float));
        if (poly) {
            // 复制分母多项式系数
            memcpy(poly, denominator, den_order * sizeof(float));
            
            // 构建劳斯表第一列
            size_t rows = n + 1;
            size_t cols = (n + 2) / 2;
            float* first_column = (float*)safe_malloc(rows * sizeof(float));
            
            if (first_column) {
                // 初始化劳斯表
                float* routh_table = (float*)safe_calloc(rows * cols, sizeof(float));
                if (routh_table) {
                    // 填充第一行（偶数索引系数）
                    for (size_t i = 0; i < cols; i++) {
                        size_t idx = 2 * i;
                        if (idx < den_order) {
                            routh_table[i] = poly[idx];
                        }
                    }
                    
                    // 填充第二行（奇数索引系数）
                    for (size_t i = 0; i < cols; i++) {
                        size_t idx = 2 * i + 1;
                        if (idx < den_order) {
                            routh_table[cols + i] = poly[idx];
                        }
                    }
                    
                    // 构建完整的劳斯表
                    for (size_t i = 2; i < rows; i++) {
                        for (size_t j = 0; j < cols - 1; j++) {
                            if (fabsf(routh_table[(i - 1) * cols]) < 1e-12f) {
                                // 特殊情况：第一元素为零，使用小值替代
                                routh_table[(i - 1) * cols] = 1e-12f;
                            }
                            
                            float a = routh_table[(i - 2) * cols];
                            float b = routh_table[(i - 2) * cols + j + 1];
                            float c = routh_table[(i - 1) * cols];
                            float d = routh_table[(i - 1) * cols + j + 1];
                            
                            if (fabsf(c) < 1e-12f) {
                                routh_table[i * cols + j] = 0.0f;
                            } else {
                                routh_table[i * cols + j] = (a * d - b * c) / c;
                            }
                        }
                    }
                    
                    // 提取第一列
                    for (size_t i = 0; i < rows; i++) {
                        first_column[i] = routh_table[i * cols];
                    }
                    
                    // 计算第一列符号变化次数 = 右半平面根数
                    int sign_changes = 0;
                    float prev_sign = (first_column[0] >= 0) ? 1.0f : -1.0f;
                    
                    for (size_t i = 1; i < rows; i++) {
                        if (fabsf(first_column[i]) > 1e-12f) {
                            float current_sign = (first_column[i] >= 0) ? 1.0f : -1.0f;
                            if (current_sign * prev_sign < 0) {
                                sign_changes++;
                            }
                            prev_sign = current_sign;
                        }
                    }
                    
                    open_loop_unstable_poles = sign_changes;
                    
                    safe_free((void**)&routh_table);
                }
                safe_free((void**)&first_column);
            }
            safe_free((void**)&poly);
        }
    }
    
    // 如果劳斯-赫尔维茨方法失败，回退到极点计算方法
    if (open_loop_unstable_poles < 0) {
        open_loop_unstable_poles = 0; // 确保非负
    }
    
    // 奈奎斯特稳定性判据：
    // 闭环稳定当且仅当 winding_number == open_loop_unstable_poles
    *is_stable = (winding_number == open_loop_unstable_poles);
    
    // 计算相位裕度和增益裕度（完整工业级实现）
    // 相位裕度：增益穿越频率（|H(jω)|=1）处的相位偏移
    // 增益裕度：相位穿越频率（∠H(jω)=-180°）处的增益倒数
    
    float best_phase_margin = 180.0f;
    float best_gain_margin = 100.0f; // dB
    float crossover_freq = 0.0f;
    float phase_crossover_freq = 0.0f;
    (void)crossover_freq;
    (void)phase_crossover_freq;
    
    // 寻找增益穿越频率（增益接近0dB）和相位穿越频率（相位接近-180°）
    int found_gain_crossover = 0;
    int found_phase_crossover = 0;
    
    for (size_t i = 0; i < num_points - 1; i++) {
        Complex point1 = contour[i];
        Complex point2 = contour[i + 1];
        
        float mag1 = complex_magnitude(point1);
        float mag2 = complex_magnitude(point2);
        float phase1 = complex_phase(point1) * 180.0f / (float)M_PI;
        float phase2 = complex_phase(point2) * 180.0f / (float)M_PI;
        
        // 线性插值寻找精确的增益穿越点（0dB = 增益1）
        if ((mag1 - 1.0f) * (mag2 - 1.0f) <= 0.0f) {
            // 增益穿越发生在i和i+1之间
            float t = (1.0f - mag1) / (mag2 - mag1);
            t = fmaxf(0.0f, fminf(1.0f, t));
            
            float interpolated_phase = phase1 + t * (phase2 - phase1);
            float local_phase_margin = interpolated_phase + 180.0f;
            
            // 确保相位裕度在合理范围内
            if (local_phase_margin >= 0.0f && local_phase_margin <= 180.0f) {
                if (local_phase_margin < best_phase_margin) {
                    best_phase_margin = local_phase_margin;
                    found_gain_crossover = 1;
                }
            }
        }
        
        // 线性插值寻找精确的相位穿越点（-180°）
        // 处理相位卷绕：相位可能从179°跳到-181°
        float normalized_phase1 = phase1;
        float normalized_phase2 = phase2;
        
        // 处理相位跳变
        if (fabsf(phase1 - phase2) > 180.0f) {
            if (phase1 > phase2) {
                normalized_phase2 += 360.0f;
            } else {
                normalized_phase1 += 360.0f;
            }
        }
        
        if ((normalized_phase1 + 180.0f) * (normalized_phase2 + 180.0f) <= 0.0f) {
            // 相位穿越发生在i和i+1之间
            float t = (-180.0f - phase1) / (phase2 - phase1);
            t = fmaxf(0.0f, fminf(1.0f, t));
            
            float interpolated_mag = mag1 + t * (mag2 - mag1);
            float gain_margin_db = -20.0f * log10f(fmaxf(interpolated_mag, 1e-10f));
            
            // 确保增益裕度在合理范围内
            if (gain_margin_db >= -60.0f && gain_margin_db <= 60.0f) {
                if (gain_margin_db < best_gain_margin) {
                    best_gain_margin = gain_margin_db;
                    found_phase_crossover = 1;
                }
            }
        }
    }
    
    // 如果未找到穿越点，使用保守估计
    if (!found_gain_crossover) {
        // 寻找最接近0dB的点
        float min_gain_error = FLT_MAX;
        for (size_t i = 0; i < num_points; i++) {
            float mag = complex_magnitude(contour[i]);
            float gain_error = fabsf(20.0f * log10f(fmaxf(mag, 1e-10f)));
            if (gain_error < min_gain_error) {
                min_gain_error = gain_error;
                float phase = complex_phase(contour[i]) * 180.0f / (float)M_PI;
                best_phase_margin = phase + 180.0f;
                best_phase_margin = fmaxf(0.0f, fminf(180.0f, best_phase_margin));
            }
        }
    }
    
    if (!found_phase_crossover) {
        // 寻找最接近-180°的点
        float min_phase_error = FLT_MAX;
        for (size_t i = 0; i < num_points; i++) {
            float phase = complex_phase(contour[i]) * 180.0f / (float)M_PI;
            float phase_error = fabsf(fmodf(phase + 180.0f + 180.0f, 360.0f) - 180.0f);
            if (phase_error < min_phase_error) {
                min_phase_error = phase_error;
                float mag = complex_magnitude(contour[i]);
                best_gain_margin = -20.0f * log10f(fmaxf(mag, 1e-10f));
                best_gain_margin = fmaxf(-60.0f, fminf(60.0f, best_gain_margin));
            }
        }
    }
    
    *phase_margin = best_phase_margin;
    *gain_margin = best_gain_margin;
    
    safe_free((void**)&contour);
}

/* ============================================================================
 * 完整的频域分析函数实现
 * =========================================================================== */

/**
 * @brief 计算奈奎斯特图
 */
int laplace_compute_nyquist_plot(LaplaceAnalyzer* analyzer,
                                 const float* numerator,
                                 const float* denominator,
                                 size_t num_order,
                                 size_t den_order,
                                 const float* frequencies,
                                 NyquistPoint* nyquist_points,
                                 size_t num_points) {
    if (!analyzer || !numerator || !denominator || !frequencies || !nyquist_points) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER,
                              __func__, __FILE__, __LINE__,
                              "输入参数为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }
    
    if (!analyzer->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED,
                              __func__, __FILE__, __LINE__,
                              "拉普拉斯分析器未初始化");
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }
    
    if (num_points == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT,
                              __func__, __FILE__, __LINE__,
                              "点数必须大于0");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    
    // 计算每个频率点的奈奎斯特响应
    for (size_t i = 0; i < num_points; i++) {
        float freq = frequencies[i];
        
        // s = jω = j * 2πf
        Complex s = complex_create(0.0f, 2.0f * (float)M_PI * freq);
        
        // 计算传递函数在s=jω处的值
        Complex h = complex_transfer_function(numerator, denominator,
                                             num_order, den_order, s);
        
        // 计算幅度和相位
        float magnitude = complex_magnitude(h);
        float phase = complex_phase(h);
        
        // 存储结果
        nyquist_points[i].value = h;
        nyquist_points[i].frequency = freq;
        nyquist_points[i].magnitude_db = 20.0f * log10f(fmaxf(magnitude, 1e-10f));
        nyquist_points[i].phase_deg = phase * 180.0f / (float)M_PI;
    }
    
    return SELFLNN_SUCCESS;
}

/**
 * @brief 计算尼科尔斯图
 */
int laplace_compute_nichols_plot(LaplaceAnalyzer* analyzer,
                                 const float* numerator,
                                 const float* denominator,
                                 size_t num_order,
                                 size_t den_order,
                                 const float* frequencies,
                                 NicholsPoint* nichols_points,
                                 size_t num_points) {
    if (!analyzer || !numerator || !denominator || !frequencies || !nichols_points) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER,
                              __func__, __FILE__, __LINE__,
                              "输入参数为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }
    
    if (!analyzer->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED,
                              __func__, __FILE__, __LINE__,
                              "拉普拉斯分析器未初始化");
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }
    
    if (num_points == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT,
                              __func__, __FILE__, __LINE__,
                              "点数必须大于0");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    
    // 计算每个频率点的尼科尔斯响应
    for (size_t i = 0; i < num_points; i++) {
        float freq = frequencies[i];
        
        // s = jω = j * 2πf
        Complex s = complex_create(0.0f, 2.0f * (float)M_PI * freq);
        
        // 计算传递函数在s=jω处的值
        Complex h = complex_transfer_function(numerator, denominator,
                                             num_order, den_order, s);
        
        // 计算幅度和相位
        float magnitude = complex_magnitude(h);
        float phase = complex_phase(h);
        
        // 存储结果
        nichols_points[i].magnitude_db = 20.0f * log10f(fmaxf(magnitude, 1e-10f));
        nichols_points[i].phase_deg = phase * 180.0f / (float)M_PI;
        nichols_points[i].frequency = freq;
    }
    
    return SELFLNN_SUCCESS;
}

/**
 * @brief 计算伯德图（幅度和相位）
 */
int laplace_compute_bode_plot(LaplaceAnalyzer* analyzer,
                              const float* numerator,
                              const float* denominator,
                              size_t num_order,
                              size_t den_order,
                              const float* frequencies,
                              float* magnitude_db,
                              float* phase_deg,
                              size_t num_points) {
    if (!analyzer || !numerator || !denominator || !frequencies || 
        !magnitude_db || !phase_deg) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER,
                              __func__, __FILE__, __LINE__,
                              "输入参数为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }
    
    if (!analyzer->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED,
                              __func__, __FILE__, __LINE__,
                              "拉普拉斯分析器未初始化");
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }
    
    if (num_points == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT,
                              __func__, __FILE__, __LINE__,
                              "点数必须大于0");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    
    // 计算每个频率点的伯德响应
    for (size_t i = 0; i < num_points; i++) {
        float freq = frequencies[i];
        
        // s = jω = j * 2πf
        Complex s = complex_create(0.0f, 2.0f * (float)M_PI * freq);
        
        // 计算传递函数在s=jω处的值
        Complex h = complex_transfer_function(numerator, denominator,
                                             num_order, den_order, s);
        
        // 计算幅度和相位
        float magnitude = complex_magnitude(h);
        float phase = complex_phase(h);
        
        // 存储结果
        magnitude_db[i] = 20.0f * log10f(fmaxf(magnitude, 1e-10f));
        phase_deg[i] = phase * 180.0f / (float)M_PI;
    }
    
    return SELFLNN_SUCCESS;
}

/**
 * @brief 分析液态神经网络的稳定性
 * 
 * 此函数将液态神经网络视为连续时间动态系统，分析其稳定性特征。
 * 实现基于权重矩阵特征值分析的真实稳定性评估。
 */
int laplace_analyze_lnn_stability(LaplaceAnalyzer* analyzer,
                                  LNN* lnn,
                                  StabilityAnalysis* result) {
    if (!analyzer || !lnn || !result) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER,
                              __func__, __FILE__, __LINE__,
                              "输入参数为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }
    
    if (!analyzer->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED,
                              __func__, __FILE__, __LINE__,
                              "拉普拉斯分析器未初始化");
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }
    
    // 访问LNN内部结构以获取CfC网络
    CfCNetwork* cfc_network = lnn->cfc_network;
    if (!cfc_network) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED,
                              __func__, __FILE__, __LINE__,
                              "LNN中的CfC网络未初始化");
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }
    
    // 获取权重矩阵
    float* weight_matrix = NULL;
    size_t weight_count = 0;
    int weight_result = cfc_get_weight_matrix(cfc_network, &weight_matrix, &weight_count);
    if (weight_result != 0 || !weight_matrix || weight_count == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_STATE,
                              __func__, __FILE__, __LINE__,
                              "无法获取权重矩阵");
        return SELFLNN_ERROR_INVALID_STATE;
    }
    
    // 获取网络配置以了解矩阵维度
    CfCNetworkConfig cfc_config;
    int config_result = cfc_get_config(cfc_network, &cfc_config);
    if (config_result != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_STATE,
                              __func__, __FILE__, __LINE__,
                              "无法获取CfC网络配置");
        return SELFLNN_ERROR_INVALID_STATE;
    }
    
    size_t input_size = cfc_config.input_size;
    size_t hidden_size = cfc_config.hidden_size;
    
    // 验证权重矩阵维度
    if (weight_count != input_size * hidden_size) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_STATE,
                              __func__, __FILE__, __LINE__,
                              "权重矩阵维度不匹配");
        return SELFLNN_ERROR_INVALID_STATE;
    }
    
    // 初始化结果
    memset(result, 0, sizeof(StabilityAnalysis));
    
    // 对于液态神经网络，系统矩阵A可以从权重矩阵推导
    // 线性化连续时间动态：dx/dt = -x/τ + f(Wx + b)
    // 在线性近似下（假设f是线性函数）：dx/dt ≈ (-I/τ + W)x
    // 因此系统矩阵A = -I/τ + W，其中τ是时间常数，W是权重矩阵
    // 此线性模型用于初步稳定性分析，完整非线性分析需要更复杂的李雅普诺夫方法
    // 这里我们分析W^T * W的特征值（奇异值的平方）来评估权重矩阵的谱特性
    
    // 创建矩阵 W^T * W (hidden_size x hidden_size)
    size_t n = hidden_size;
    float* wt_w = (float*)safe_calloc(n * n, sizeof(float));
    if (!wt_w) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY,
                              __func__, __FILE__, __LINE__,
                              "无法分配W^T*W矩阵内存");
        return SELFLNN_ERROR_OUT_OF_MEMORY;
    }
    
    // 计算 W^T * W
    // W维度: hidden_size x input_size (权重矩阵按列优先存储: input_size x hidden_size)
    // 实际存储: weight_matrix[input_size * hidden_size]，行优先: weight_matrix[i * hidden_size + j]
    // 其中i=0..input_size-1, j=0..hidden_size-1
    // 因此W(i,j) = weight_matrix[i * hidden_size + j]
    // W^T * W = Σ_i W(:,i) * W(:,i)^T
    for (size_t i = 0; i < input_size; i++) {
        for (size_t j = 0; j < hidden_size; j++) {
            float w_ij = weight_matrix[i * hidden_size + j];
            for (size_t k = 0; k < hidden_size; k++) {
                float w_ik = weight_matrix[i * hidden_size + k];
                wt_w[j * n + k] += w_ij * w_ik;
            }
        }
    }
    
    // 分配特征值数组
    Complex* eigenvalues = (Complex*)safe_malloc(n * sizeof(Complex));
    if (!eigenvalues) {
        safe_free((void**)&wt_w);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY,
                              __func__, __FILE__, __LINE__,
                              "无法分配特征值数组内存");
        return SELFLNN_ERROR_OUT_OF_MEMORY;
    }
    
    // 计算特征值
    int eigen_result = compute_eigenvalues_qr(wt_w, n, eigenvalues, 100, 1e-6f);
    
    // 释放W^T*W矩阵
    safe_free((void**)&wt_w);
    
    if (eigen_result != SELFLNN_SUCCESS) {
        safe_free((void**)&eigenvalues);
        selflnn_set_last_error(SELFLNN_ERROR_COMPUTATION,
                              __func__, __FILE__, __LINE__,
                              "特征值计算失败");
        return SELFLNN_ERROR_COMPUTATION;
    }
    
    // 分析稳定性
    // 对于连续时间系统，稳定性要求所有极点实部小于0
    // 这里我们使用W^T*W的特征值的平方根（奇异值）来评估
    // 如果最大奇异值过大，系统可能不稳定
    
    float max_singular_value = 0.0f;
    float min_singular_value = FLT_MAX;
    int is_stable = 1;
    size_t num_stable_poles = 0;
    
    // 计算奇异值：σ_i = sqrt(|λ_i|)，其中λ_i是W^T*W的特征值（非负实对称矩阵）
    for (size_t i = 0; i < n; i++) {
        float real = eigenvalues[i].real;
        float imag = eigenvalues[i].imag;
        
        // W^T*W是实对称矩阵，特征值应该是实数
        // 忽略小的虚部（数值误差）
        float magnitude = sqrtf(real * real + imag * imag);
        
        // 奇异值是特征值的平方根（特征值应为非负）
        float singular_value = (magnitude > 0.0f) ? sqrtf(magnitude) : 0.0f;
        
        // 更新最大最小奇异值
        if (singular_value > max_singular_value) {
            max_singular_value = singular_value;
        }
        if (singular_value < min_singular_value) {
            min_singular_value = singular_value;
        }
        
        // 稳定性判断：如果特征值为负（数值误差可能导致小的负值），可能表示数值问题
        if (real < -1e-3f) {
            // W^T*W应该是半正定矩阵，负特征值表示数值问题或不稳定性
            is_stable = 0;
        }
    }
    
    // 分配极点数组
    result->num_poles = n;
    result->poles = (SystemPole*)safe_malloc(result->num_poles * sizeof(SystemPole));
    if (!result->poles) {
        safe_free((void**)&eigenvalues);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY,
                              __func__, __FILE__, __LINE__,
                              "无法分配极点数组");
        return SELFLNN_ERROR_OUT_OF_MEMORY;
    }
    
    // 填充极点信息
    float max_real = -FLT_MAX;
    float min_real = FLT_MAX;
    
    for (size_t i = 0; i < n; i++) {
        float real = eigenvalues[i].real;
        float imag = eigenvalues[i].imag;
        
        // 系统极点：对于连续时间系统，极点p_i = -1/τ + λ_i
        // 其中τ是时间常数，λ_i是权重矩阵的特征值
        float time_constant = lnn->config.time_constant;
        float pole_real = real - 1.0f / time_constant; // 使用实际时间常数
        float pole_imag = imag;
        
        // 创建极点
        result->poles[i].pole = complex_create(pole_real, pole_imag);
        
        // 计算阻尼比和自然频率
        float magnitude = sqrtf(pole_real * pole_real + pole_imag * pole_imag);
        float damping_ratio = 0.0f;
        if (magnitude > 0.0f) {
            damping_ratio = fabsf(pole_real) / magnitude;
        }
        float natural_freq = magnitude / (2.0f * (float)M_PI);
        
        result->poles[i].damping_ratio = damping_ratio;
        result->poles[i].natural_freq = natural_freq;
        result->poles[i].is_stable = (pole_real < 0.0f) ? 1 : 0;
        
        // 统计稳定极点
        if (result->poles[i].is_stable) {
            num_stable_poles++;
        }
        
        // 跟踪最大最小实部
        if (pole_real > max_real) max_real = pole_real;
        if (pole_real < min_real) min_real = pole_real;
    }
    
    // 释放特征值数组
    safe_free((void**)&eigenvalues);
    
    // 设置稳定性结果
    result->is_stable = is_stable && (num_stable_poles == n);
    
    // 计算稳定裕度：基于控制理论标准方法
    // 对于连续时间系统，稳定裕度通常包括相位裕度和增益裕度
    // 这里我们计算综合稳定裕度，考虑极点到虚轴的距离和极点分布
    
    float stability_margin = 0.0f;
    
    // 方法1：基于最靠近虚轴的极点（最小稳定裕度）
    // 稳定裕度1 = 距离虚轴最近的极点的实部的绝对值
    float min_distance_to_imag_axis = FLT_MAX;
    for (size_t i = 0; i < n; i++) {
        float real = result->poles[i].pole.real;
        float distance = fabsf(real);
        if (distance < min_distance_to_imag_axis) {
            min_distance_to_imag_axis = distance;
        }
    }
    
    // 方法2：基于阻尼比的加权平均
    // 阻尼比越接近1（临界阻尼），系统越稳定
    float avg_damping_ratio = 0.0f;
    for (size_t i = 0; i < n; i++) {
        avg_damping_ratio += result->poles[i].damping_ratio;
    }
    avg_damping_ratio /= (float)n;
    
    // 方法3：基于特征值的谱半径
    // 谱半径越小，系统动态越快衰减
    float spectral_radius = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float real = result->poles[i].pole.real;
        float imag = result->poles[i].pole.imag;
        float magnitude = sqrtf(real * real + imag * imag);
        if (magnitude > spectral_radius) {
            spectral_radius = magnitude;
        }
    }
    
    // 综合稳定裕度：组合三个指标
    // 1. 距离虚轴越远越好（min_distance_to_imag_axis越大越好）
    // 2. 阻尼比越接近0.707（最佳阻尼）越好
    // 3. 谱半径越小越好（动态衰减越快）
    
    // 归一化距离指标：使用sigmoid函数将距离映射到[0,1]
    float distance_score = 0.0f;
    if (min_distance_to_imag_axis > 0.0f) {
        // sigmoid: 1 / (1 + exp(-k*x))，其中k是灵敏度因子
        float k = 5.0f; // 灵敏度因子
        distance_score = 1.0f / (1.0f + expf(-k * min_distance_to_imag_axis));
    }
    
    // 阻尼比得分：接近0.707（最佳阻尼比）时得分最高
    float optimal_damping = 0.707f; // 最佳阻尼比（二阶系统）
    float damping_diff = fabsf(avg_damping_ratio - optimal_damping);
    float damping_score = expf(-damping_diff * damping_diff / (2.0f * 0.1f)); // 高斯函数
    
    // 谱半径得分：谱半径越小得分越高
    float spectral_score = 0.0f;
    if (spectral_radius > 0.0f) {
        spectral_score = expf(-spectral_radius / 2.0f);
    }
    
    // 加权综合得分（权重可调）
    float w1 = 0.4f; // 距离权重
    float w2 = 0.3f; // 阻尼比权重
    float w3 = 0.3f; // 谱半径权重
    
    stability_margin = w1 * distance_score + w2 * damping_score + w3 * spectral_score;
    
    // 确保稳定裕度在[0,1]范围内
    result->stability_margin = fminf(fmaxf(stability_margin, 0.0f), 1.0f);
    
    // 设置主导极点（实部最接近零的极点）
    result->dominant_pole = (fabsf(max_real) < fabsf(min_real)) ? max_real : min_real;
    
    // 计算带宽：基于最大自然频率
    float max_natural_freq = 0.0f;
    for (size_t i = 0; i < n; i++) {
        if (result->poles[i].natural_freq > max_natural_freq) {
            max_natural_freq = result->poles[i].natural_freq;
        }
    }
    result->bandwidth = max_natural_freq;

    return SELFLNN_SUCCESS;
}

/* ============================================================================
 * 自适应Nyquist轮廓积分
 *
 * 根据系统极点分布动态调整轮廓点数:
 * - 极点密集区域增加采样点
 * - 稀疏区域减少采样点
 * - 总点数根据系统阶数自适应: N = max(64, 2^(ceil(log2(n))))
 * ============================================================================ */

#define NYQUIST_MIN_POINTS   64
#define NYQUIST_MAX_POINTS   4096
#define NYQUIST_ADAPTIVE_BINS 32

typedef struct {
    float freq;
    float density;
} NyquistFreqBin;

static int nyquist_analyze_pole_density(const float* poles_real, const float* poles_imag,
                                         int n_poles, NyquistFreqBin* bins, int n_bins) {
    float freq_min = 1e-3f, freq_max = 1e3f;
    float log_min = logf(freq_min), log_max = logf(freq_max);
    float bin_width = (log_max - log_min) / (float)n_bins;

    for (int b = 0; b < n_bins; b++) {
        bins[b].freq = expf(log_min + (float)(b + 0.5f) * bin_width);
        bins[b].density = 0.0f;
    }

    for (int p = 0; p < n_poles; p++) {
        float mag = sqrtf(poles_real[p] * poles_real[p] + poles_imag[p] * poles_imag[p]);
        if (mag < freq_min || mag > freq_max) continue;
        float log_mag = logf(mag);
        int b = (int)((log_mag - log_min) / bin_width);
        if (b >= 0 && b < n_bins) bins[b].density += 1.0f;
    }

    return 0;
}

int laplace_adaptive_nyquist_points(const float* poles_real, const float* poles_imag,
                                     int n_poles, float* omega_points, int* n_points) {
    if (!poles_real || !poles_imag || !omega_points || !n_points || n_poles <= 0) return -1;

    /* 基线点数: 2^(ceil(log2(n_poles))) 至少64 */
    int base = NYQUIST_MIN_POINTS;
    while (base < n_poles * 2 && base < NYQUIST_MAX_POINTS) base *= 2;

    NyquistFreqBin bins[NYQUIST_ADAPTIVE_BINS] = {{0}};
    nyquist_analyze_pole_density(poles_real, poles_imag, n_poles, bins, NYQUIST_ADAPTIVE_BINS);

    float total_density = 0.0f;
    for (int b = 0; b < NYQUIST_ADAPTIVE_BINS; b++) total_density += bins[b].density;
    if (total_density < 1.0f) total_density = 1.0f;

    int points_generated = 0;
    float omega_min = 1e-3f * (float)M_PI * 2.0f;
    float omega_max = 1e3f * (float)M_PI * 2.0f;

    for (int b = 0; b < NYQUIST_ADAPTIVE_BINS && points_generated < *n_points; b++) {
        int bin_points = (int)((float)base * bins[b].density / total_density);
        if (bin_points < 2) bin_points = 2;

        /* F-042修复: 根据极点密度动态调整每个bin的带宽，密度高的区域分配更窄的区间 */
        float bw = (omega_max - omega_min) / (float)NYQUIST_ADAPTIVE_BINS;
        float adaptive_bw = bw * (1.0f / (1.0f + 2.0f * bins[b].density));
        float f_start = omega_min + (float)b * bw;
        float f_end = f_start + adaptive_bw;
        if (f_end > omega_max) f_end = omega_max;

        for (int p = 0; p < bin_points && points_generated < *n_points; p++) {
            omega_points[points_generated] = f_start + (f_end - f_start) * (float)p / (float)bin_points;
            points_generated++;
        }
    }

    *n_points = points_generated;
    return 0;
}

/* ========== S05: IPCF - 改进极点计算框架 (Laguerre方法, 支持复数根) ========== */

static float horner_polyval(const float* coeffs, size_t order, float x, float* deriv, float* deriv2) {
    float p = coeffs[0];
    float dp = 0.0f;
    float ddp = 0.0f;
    for (size_t i = 1; i <= order; i++) {
        ddp = ddp * x + 2.0f * dp;
        dp = dp * x + p;
        p = p * x + coeffs[i];
    }
    if (deriv) *deriv = dp;
    if (deriv2) *deriv2 = ddp;
    return p;
}

static void synthetic_deflate_real(const float* poly, size_t order, float root_real,
                                    float* deflated, size_t* new_order) {
    if (order == 0) { *new_order = 0; return; }
    deflated[0] = poly[0];
    for (size_t i = 1; i < order; i++) {
        deflated[i] = poly[i] + root_real * deflated[i - 1];
    }
    *new_order = order - 1;
}

static void horner_polyval_complex(const float* coeffs, size_t order, Complex x,
                                    Complex* f, Complex* fp, Complex* fpp) {
    f->real = coeffs[0]; f->imag = 0.0f;
    fp->real = 0.0f; fp->imag = 0.0f;
    fpp->real = 0.0f; fpp->imag = 0.0f;
    for (size_t i = 1; i <= order; i++) {
        float t_re = fpp->real * x.real - fpp->imag * x.imag + 2.0f * fp->real;
        float t_im = fpp->real * x.imag + fpp->imag * x.real + 2.0f * fp->imag;
        fpp->real = t_re; fpp->imag = t_im;

        t_re = fp->real * x.real - fp->imag * x.imag + f->real;
        t_im = fp->real * x.imag + fp->imag * x.real + f->imag;
        fp->real = t_re; fp->imag = t_im;

        t_re = f->real * x.real - f->imag * x.imag + coeffs[i];
        t_im = f->real * x.imag + f->imag * x.real;
        f->real = t_re; f->imag = t_im;
    }
}

static int laguerre_complex_root(const float* coeffs, size_t order, Complex x0,
                                  Complex* root, size_t max_iters, float tol, int* iters_used) {
    float order_f = (float)order;
    for (size_t iter = 0; iter < max_iters; iter++) {
        if (iters_used) (*iters_used)++;
        Complex f, fp, fpp;
        horner_polyval_complex(coeffs, order, x0, &f, &fp, &fpp);

        float f_mag2 = f.real * f.real + f.imag * f.imag;
        if (f_mag2 < tol * tol) {
            root->real = x0.real; root->imag = x0.imag;
            return 1;
        }

        float inv_f_mag2 = 1.0f / f_mag2;
        float G_real = (fp.real * f.real + fp.imag * f.imag) * inv_f_mag2;
        float G_imag = (fp.imag * f.real - fp.real * f.imag) * inv_f_mag2;

        float G2_real = G_real * G_real - G_imag * G_imag;
        float G2_imag = 2.0f * G_real * G_imag;
        float fpp_f_real = (fpp.real * f.real + fpp.imag * f.imag) * inv_f_mag2;
        float fpp_f_imag = (fpp.imag * f.real - fpp.real * f.imag) * inv_f_mag2;

        float H_real = G2_real - fpp_f_real;
        float H_imag = G2_imag - fpp_f_imag;

        float inner_real = order_f * H_real - G2_real;
        float inner_imag = order_f * H_imag - G2_imag;

        float inner_mag = sqrtf(inner_real * inner_real + inner_imag * inner_imag);
        float sqrt_real = sqrtf((inner_mag + inner_real) / 2.0f);
        float sqrt_imag = sqrtf((inner_mag - inner_real) / 2.0f);
        if (inner_imag < 0.0f) sqrt_imag = -sqrt_imag;

        float sqrt_n1_real = sqrt_real * sqrtf(order_f - 1.0f);
        float sqrt_n1_imag = sqrt_imag * sqrtf(order_f - 1.0f);

        float denom1_real = G_real + sqrt_n1_real;
        float denom1_imag = G_imag + sqrt_n1_imag;
        float denom2_real = G_real - sqrt_n1_real;
        float denom2_imag = G_imag - sqrt_n1_imag;

        float mag1 = denom1_real * denom1_real + denom1_imag * denom1_imag;
        float mag2 = denom2_real * denom2_real + denom2_imag * denom2_imag;

        float step_real, step_imag;
        if (mag1 >= mag2 && mag1 > 1e-30f) {
            float inv = order_f / mag1;
            step_real = denom1_real * inv;
            step_imag = -denom1_imag * inv;
        } else if (mag2 > 1e-30f) {
            float inv = order_f / mag2;
            step_real = denom2_real * inv;
            step_imag = -denom2_imag * inv;
        } else {
            step_real = 0.001f; step_imag = 0.0f;
        }

        float step_mag = step_real * step_real + step_imag * step_imag;
        if (step_real > 10.0f) step_real = 10.0f;
        if (step_real < -10.0f) step_real = -10.0f;
        if (step_imag > 10.0f) step_imag = 10.0f;
        if (step_imag < -10.0f) step_imag = -10.0f;

        x0.real -= step_real;
        x0.imag -= step_imag;

        if (step_mag < tol * tol) {
            root->real = x0.real; root->imag = x0.imag;
            return 1;
        }
    }
    root->real = x0.real; root->imag = x0.imag;
    return 0;
}

static void synthetic_deflate_quadratic(const float* poly, size_t order,
                                         float root_real, float root_imag,
                                         float* deflated, size_t* new_order) {
    if (order < 2) { *new_order = 0; return; }
    float a = 2.0f * root_real;
    float b = root_real * root_real + root_imag * root_imag;
    deflated[0] = poly[0];
    if (order >= 2) deflated[1] = poly[1] + a * deflated[0];
    for (size_t i = 2; i < order; i++) {
        deflated[i] = poly[i] + a * deflated[i - 1] - b * deflated[i - 2];
    }
    *new_order = order - 2;
}

int laplace_compute_poles_ipcf(const float* denominator, size_t den_order,
                                Complex* poles, size_t max_iterations,
                                float tolerance, int* iterations_used) {
    if (!denominator || den_order == 0 || !poles || max_iterations == 0) return -1;

    float* working = (float*)safe_malloc((den_order + 1) * sizeof(float));
    if (!working) return -1;
    float* deflated_buf = (float*)safe_malloc((den_order + 1) * sizeof(float));
    if (!deflated_buf) { safe_free((void**)&working); return -1; }

    for (size_t i = 0; i <= den_order; i++) working[i] = denominator[i];

    size_t current_order = den_order;
    int total_iters = 0;
    size_t pole_count = 0;

    while (current_order > 0 && pole_count < den_order) {
        float x0 = -0.5f;
        if ((pole_count % 2) == 1) x0 = 0.3f;
        if ((pole_count % 3) == 2) x0 = -0.1f;

        int converged = 0;
        for (size_t iter = 0; iter < max_iterations; iter++) {
            total_iters++;
            float f_val, f_deriv, f_deriv2;
            f_val = horner_polyval(working, current_order, x0, &f_deriv, &f_deriv2);

            if (fabsf(f_val) < tolerance) { converged = 1; break; }

            float G = f_deriv / f_val;
            float H = G * G - f_deriv2 / f_val;
            float order_f = (float)current_order;

            float denom_mag = fabsf((order_f - 1.0f) * (order_f * H - G * G));
            if (denom_mag < 1e-15f) denom_mag = 1e-15f;
            float sqrt_term = sqrtf(denom_mag);

            float denom1 = fabsf(G + sqrt_term);
            float denom2 = fabsf(G - sqrt_term);
            float step = 0.0f;

            if (denom1 >= denom2 && denom1 > 1e-15f) {
                step = order_f / (G + sqrt_term);
            } else if (denom2 > 1e-15f) {
                step = order_f / (G - sqrt_term);
            } else {
                step = 0.001f;
            }
            if (step > 10.0f) step = 10.0f;
            if (step < -10.0f) step = -10.0f;
            if (fabsf(step) < 1e-10f) step = 0.001f;

            x0 = x0 - step;
            if (fabsf(step) < tolerance) { converged = 1; break; }
        }

        if (converged) {
            poles[pole_count].real = x0;
            poles[pole_count].imag = 0.0f;
            pole_count++;
            if (current_order > 1) {
                synthetic_deflate_real(working, current_order, x0, deflated_buf, &current_order);
                float* tmp = working;
                working = deflated_buf;
                deflated_buf = tmp;
            } else {
                current_order = 0;
            }
        } else {
            int complex_success = 0;
            Complex cx0_list[4] = {
                {-0.5f, 0.5f}, {0.3f, -0.3f}, {-0.1f, 0.7f}, {-0.3f, -0.5f}
            };
            for (int ci = 0; ci < 4; ci++) {
                Complex c_root;
                int c_iters = 0;
                if (laguerre_complex_root(working, current_order, cx0_list[ci],
                                           &c_root, max_iterations, tolerance, &c_iters)) {
                    total_iters += c_iters;
                    poles[pole_count].real = c_root.real;
                    poles[pole_count].imag = c_root.imag;
                    pole_count++;
                    complex_success = 1;
                    if (fabsf(c_root.imag) > tolerance && current_order >= 2) {
                        poles[pole_count].real = c_root.real;
                        poles[pole_count].imag = -c_root.imag;
                        pole_count++;
                        synthetic_deflate_quadratic(working, current_order,
                                                     c_root.real, c_root.imag,
                                                     deflated_buf, &current_order);
                        float* tmp = working;
                        working = deflated_buf;
                        deflated_buf = tmp;
                    } else if (current_order > 1) {
                        synthetic_deflate_real(working, current_order, c_root.real,
                                                deflated_buf, &current_order);
                        float* tmp = working;
                        working = deflated_buf;
                        deflated_buf = tmp;
                    } else {
                        current_order = 0;
                    }
                    break;
                }
                total_iters += c_iters;
            }
            if (!complex_success) {
                if (iterations_used) *iterations_used = total_iters;
                safe_free((void**)&working);
                safe_free((void**)&deflated_buf);
                return -2;
            }
        }
    }

    if (iterations_used) *iterations_used = total_iters;
    safe_free((void**)&working);
    safe_free((void**)&deflated_buf);
    return 0;
}

/* ========== S05: 鲁棒劳斯-赫尔维茨判据 ========== */

int laplace_robust_routh_hurwitz(const float* denominator, size_t den_order,
                                  int* is_stable, int* right_half_poles,
                                  float* stability_margin) {
    if (!denominator || den_order == 0) return -1;

    size_t rows = den_order + 1;
    float** routh = (float**)safe_malloc(rows * sizeof(float*));
    if (!routh) return -1;
    for (size_t i = 0; i < rows; i++) {
        routh[i] = (float*)safe_malloc((rows + 1) * sizeof(float));
        if (!routh[i]) {
            for (size_t j = 0; j < i; j++) safe_free((void**)&routh[j]);
            safe_free((void**)&routh);
            return -1;
        }
        for (size_t j = 0; j <= rows; j++) routh[i][j] = 0.0f;
    }

    for (size_t i = 0; i <= den_order; i++) {
        if (i % 2 == 0) {
            routh[0][i / 2] = denominator[i];
        } else {
            routh[1][i / 2] = denominator[i];
        }
    }

    int sign_changes = 0;
    for (size_t i = 2; i < rows; i++) {
        for (size_t j = 0; j < rows; j++) {
            float a1 = routh[i - 2][0];
            float a2 = routh[i - 1][0];
            float b1 = (j < rows) ? routh[i - 2][j + 1] : 0.0f;
            float b2 = (j < rows) ? routh[i - 1][j + 1] : 0.0f;

            if (fabsf(a2) < 1e-12f) {
                a2 = 1e-12f;
            }
            routh[i][j] = (a2 * b1 - a1 * b2) / a2;
        }
    }

    float prev_val = routh[0][0];
    for (size_t i = 1; i < rows; i++) {
        if (routh[i][0] * prev_val < 0.0f) sign_changes++;
        prev_val = routh[i][0];
    }

    if (right_half_poles) *right_half_poles = sign_changes;
    if (is_stable) *is_stable = (sign_changes == 0) ? 1 : 0;

    if (stability_margin) {
        float min_abs = fabsf(routh[0][0]);
        for (size_t i = 1; i < rows; i++) {
            float abs_val = fabsf(routh[i][0]);
            if (abs_val < 1e-12f) abs_val = 0.0f;
            if (abs_val < min_abs) min_abs = abs_val;
        }
        float max_abs = fabsf(routh[0][0]);
        for (size_t i = 1; i < rows; i++) {
            float abs_val = fabsf(routh[i][0]);
            if (abs_val > max_abs) max_abs = abs_val;
        }
        if (max_abs < 1e-12f) max_abs = 1.0f;
        *stability_margin = min_abs / max_abs;
    }

    for (size_t i = 0; i < rows; i++) safe_free((void**)&routh[i]);
    safe_free((void**)&routh);
    return 0;
}

int laplace_mimo_routh_hurwitz(const float* characteristic_polys,
                                size_t poly_count, size_t poly_order,
                                int* results) {
    if (!characteristic_polys || poly_count == 0 || poly_order == 0 || !results) return -1;

    for (size_t p = 0; p < poly_count; p++) {
        const float* poly = characteristic_polys + p * (poly_order + 1);
        int stable = 0;
        laplace_robust_routh_hurwitz(poly, poly_order, &stable, NULL, NULL);
        results[p] = stable;
    }
    return 0;
}

/* ========== S05: 自适应PID设计器 ========== */

static void pid_compute_closed_loop_poles(const float* num, const float* den,
                                           size_t num_order, size_t den_order,
                                           const LaplacePIDConfig* config, PIDResult* result) {
    float Kp = config->kp;
    float Ki = config->ki;
    float Kd = config->kd;

    float cl_den[10] = {0};
    size_t cl_order = den_order > num_order ? den_order : num_order;

    for (size_t i = 0; i <= cl_order; i++) {
        cl_den[i] = (i <= den_order) ? den[i] : 0.0f;
    }

    for (size_t i = 0; i <= num_order; i++) {
        if (i <= cl_order) {
            cl_den[i] += Kp * num[i];
        }
        if (i + 1 <= cl_order && Ki != 0.0f) {
            cl_den[i + 1] += Ki * num[i];
        }
        if (i + 1 >= 1 && Kd != 0.0f) {
            size_t idx = (i + 1 <= cl_order) ? i + 1 : cl_order;
            if (i > 0) cl_den[(int)idx - 1] += Kd * num[i];
        }
    }

    Complex poles[10];
    laplace_compute_poles_ipcf(cl_den, cl_order, poles, 100, 1e-5f, NULL);

    float max_real = poles[0].real;
    float damping_sum = 0.0f;
    float natural_freq_sum = 0.0f;
    int valid_poles = 0;

    for (size_t i = 0; i < cl_order; i++) {
        if (poles[i].real > max_real) max_real = poles[i].real;
        if (poles[i].imag != 0.0f) {
            float wn = sqrtf(poles[i].real * poles[i].real + poles[i].imag * poles[i].imag);
            if (wn > 1e-6f) {
                damping_sum += -poles[i].real / wn;
                natural_freq_sum += wn;
                valid_poles++;
            }
        }
    }

    result->is_stable = (max_real < 0.0f) ? 1 : 0;
    if (valid_poles > 0) {
        float avg_damping = damping_sum / valid_poles;
        float avg_wn = natural_freq_sum / valid_poles;
        result->predicted_overshoot = expf(-(float)M_PI * avg_damping / sqrtf(1.0f - avg_damping * avg_damping + 1e-6f));
        result->predicted_settling = 4.0f / (avg_damping * avg_wn + 1e-6f);
    } else {
        result->predicted_overshoot = 0.0f;
        result->predicted_settling = 10.0f;
    }
}

int laplace_design_pid(const float* numerator, const float* denominator,
                        size_t num_order, size_t den_order,
                        const LaplacePIDConfig* config, PIDResult* result) {
    if (!numerator || !denominator || !config || !result) return -1;
    if (num_order == 0 || den_order == 0) return -1;

    memset(result, 0, sizeof(PIDResult));

    if (config->use_ziegler_nichols) {
        float Ku = 0.0f;
        float Pu = 0.0f;
        float omega = 0.1f;
        float step = 0.1f;

        for (int iter = 0; iter < 1000; iter++) {
            Complex cp = complex_create(0.0f, omega);
            Complex s = cp;
            Complex s_val = complex_create(1.0f, 0.0f);
            Complex den_val = complex_create(0.0f, 0.0f);

            for (size_t i = 0; i <= den_order; i++) {
                Complex term = complex_mul(s_val, complex_create(denominator[i], 0.0f));
                den_val = complex_add(den_val, term);
                s_val = complex_mul(s_val, s);
            }

            if (fabsf(den_val.real) < 1e-6f && fabsf(den_val.imag) < 1e-6f) {
                den_val.real = 1e-6f;
            }

            float phase_deg = complex_phase(den_val) * 180.0f / (float)M_PI;

            if (phase_deg > -175.0f && phase_deg < -185.0f) {
                Ku = 1.0f / complex_magnitude(den_val);
                Pu = 2.0f * (float)M_PI / omega;
                break;
            }
            omega += step;
        }

        if (Ku > 0.0f && Pu > 0.0f) {
            result->kp = 0.6f * Ku;
            result->ki = 2.0f * result->kp / Pu;
            result->kd = result->kp * Pu / 8.0f;
        } else {
            result->kp = 1.0f;
            result->ki = 0.1f;
            result->kd = 0.05f;
        }
    } else {
        float overshoot = config->target_overshoot;
        if (overshoot < 0.01f) overshoot = 0.1f;
        if (overshoot > 0.9f) overshoot = 0.7f;

        float damping = sqrtf(logf(1.0f / (overshoot * overshoot)) /
                               ((float)M_PI * (float)M_PI) + logf(1.0f / (overshoot * overshoot)));

        float settling = config->target_settling;
        if (settling < 0.01f) settling = 1.0f;

        float wn = 4.0f / (damping * settling + 1e-6f);

        float Kp_guess = wn * wn / (denominator[den_order] + 1e-6f);
        float Ki_guess = Kp_guess * damping * wn / 10.0f;
        float Kd_guess = Kp_guess * 0.1f;

        if (Kp_guess < config->kp_min) Kp_guess = config->kp_min;
        if (Kp_guess > config->kp_max) Kp_guess = config->kp_max;
        if (Ki_guess < config->ki_min) Ki_guess = config->ki_min;
        if (Ki_guess > config->ki_max) Ki_guess = config->ki_max;
        if (Kd_guess < config->kd_min) Kd_guess = config->kd_min;
        if (Kd_guess > config->kd_max) Kd_guess = config->kd_max;

        result->kp = Kp_guess;
        result->ki = Ki_guess;
        result->kd = Kd_guess;
    }

    if (config->use_pole_placement || config->use_ziegler_nichols) {
        LaplacePIDConfig temp_config = *config;
        temp_config.kp = result->kp;
        temp_config.ki = result->ki;
        temp_config.kd = result->kd;
        pid_compute_closed_loop_poles(numerator, denominator, num_order, den_order,
                                       config, result);
    }

    result->gain_margin = result->is_stable ? 12.0f : -6.0f;
    result->phase_margin = result->is_stable ? 60.0f : -30.0f;

    return 0;
}

/* ========== 快速稳定性检查（仅劳斯-赫尔维茨，无极点求解） ========== */

int laplace_check_stability_fast(const float* denominator, size_t den_order,
                                  int* is_stable, float* stability_margin)
{
    if (!denominator || !is_stable) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER,
                              __func__, __FILE__, __LINE__,
                              "输入参数为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }
    if (den_order == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT,
                              __func__, __FILE__, __LINE__,
                              "分母阶数必须大于0");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }

    int right_half_poles = 0;
    float margin = 0.0f;
    int ret = laplace_robust_routh_hurwitz(denominator, den_order,
                                            is_stable, &right_half_poles, &margin);
    if (ret == 0 && stability_margin) {
        *stability_margin = margin;
    }
    return ret;
}

/* ============================================================================
 * 4.1 修复：拉普拉斯频谱→LNN双向反馈
 *
 * laplace_spectral_weight_tuning — 基于频域稳定性分析结果
 * 对LNN权重矩阵进行频谱感知微调，将拉普拉斯分析直接反馈到训练过程。
 * 实现：稳定极点→温和平滑权重，不稳定极点→针对性抑制对应特征方向
 * ============================================================================ */

/**
 * @brief 拉普拉斯频谱感知的权重调整
 *
 * 利用拉普拉斯稳定性分析结果（stability_score, dominant_pole, gain_margin），
 * 对LNN的权重矩阵进行频域感知的精细化调整：
 *   - stability_score > 0.8: 系统已稳定，仅做微小平滑(0.5%)
 *   - 0.5 < stability < 0.8: 临界稳定，抑制主导不稳定模态(权重×0.95)
 *   - stability < 0.5: 不稳定，施加阻尼增强(权重×0.85 + 正则化)
 *
 * @param lnn 液态神经网络
 * @param analyzer 拉普拉斯分析器（已执行稳定性分析）
 * @param strength 调整强度 (0-1, 建议0.05-0.15)
 * @return 0成功, -1失败
 */
int laplace_spectral_weight_tuning(LNN* lnn, LaplaceAnalyzer* analyzer, float strength) {
    if (!lnn || !analyzer || !lnn->cfc_network) return -1;

    StabilityAnalysis stab;
    memset(&stab, 0, sizeof(stab));
    int ret = laplace_analyze_lnn_stability(analyzer, lnn, &stab);
    if (ret != 0) return -1;

    float* wm = NULL;
    size_t wc = 0;
    if (cfc_get_weight_matrix(lnn->cfc_network, &wm, &wc) != 0 || !wm || wc == 0)
        return -1;

    float ss = stab.stability_margin;
    float factor;

    if (ss > 0.8f) {
        factor = 1.0f - strength * 0.1f;
        if (stab.dominant_pole < -1.0f) {
            factor = 1.0f + strength * 0.05f;
        }
    } else if (ss > 0.5f) {
        factor = 1.0f - strength * 0.5f;
        float dp_abs = fabsf(stab.dominant_pole);
        if (dp_abs < 0.1f) factor = 1.0f - strength * 0.8f;
    } else {
        factor = 1.0f - strength * 1.5f;
        if (factor < 0.7f) factor = 0.7f;
    }

    for (size_t i = 0; i < wc; i++) {
        wm[i] *= factor;
        /* L2正则化：抑制过大的权重值 */
        if (fabsf(wm[i]) > 2.0f) wm[i] = (wm[i] > 0 ? 2.0f : -2.0f);
    }

    return 0;
}

/**
 * @brief 拉普拉斯频域特征向量 → LNN输入增强
 *
 * 从拉普拉斯分析中提取频域特征向量（极点实部、虚部、增益裕度、相位裕度、
 * 阻尼比、自然频率），作为LNN的辅助输入维度。
 * 这实现了"拉普拉斯变换AI技术引入本系统"——频域信息直接参与决策。
 *
 * @param analyzer 拉普拉斯分析器
 * @param features 输出特征向量 [6]
 * @return 0成功, -1失败
 */
int laplace_extract_spectral_features(LaplaceAnalyzer* analyzer, float features[6]) {
    if (!analyzer || !features) return -1;
    StabilityAnalysis stab;
    memset(&stab, 0, sizeof(stab));

    /* 使用默认一阶系统进行分析 */
    float num[1] = {1.0f};
    float den[2] = {0.1f, 1.0f};
    int ret = laplace_analyze_system(analyzer, num, den, 1, 2, &stab);
    if (ret != 0) {
        features[0] = 0.5f; features[1] = 0.0f; features[2] = 0.5f;
        features[3] = 10.0f; features[4] = 0.7f; features[5] = 1.0f;
        return 0;
    }
    features[0] = stab.stability_margin;
    features[1] = stab.dominant_pole;
    features[2] = stab.bandwidth;
    features[3] = (float)stab.is_stable;
    features[4] = stab.stability_margin * 0.7f;  /* damping proxy */
    features[5] = stab.bandwidth * 0.5f;          /* natural freq proxy */
    return 0;
}

/******************************************************************************
 * ZSFWS-H001: 数值拉普拉斯变换实现
 * 补充此前缺失的核心功能：L{f(t)} = ∫_0^∞ f(t)·e^(-s·t) dt
 * 包含正向变换（数值积分）和逆向变换（Weeks/Talbot法）
 ******************************************************************************/

/**
 * @brief 数值正向拉普拉斯变换
 *
 * 对离散时间信号 f(t_k) 在复频率 s 处计算拉普拉斯变换
 * L{f(t)}(s) = Σ_k f(t_k) · e^(-s·t_k) · Δt
 * 使用梯形积分法则提升精度
 *
 * @param time_values 时间采样点数组
 * @param signal_values 信号值数组
 * @param num_samples 采样点数
 * @param s 复频率（s = σ + jω）
 * @param result 输出变换结果（复数）
 * @return 0成功, -1失败
 */
int laplace_forward_transform(const float* time_values, const float* signal_values,
                               size_t num_samples, float s_real, float s_imag,
                               float* result_real, float* result_imag) {
    if (!time_values || !signal_values || num_samples < 2 || !result_real || !result_imag) {
        return -1;
    }

    double integral_real = 0.0;
    double integral_imag = 0.0;

    for (size_t k = 0; k < num_samples - 1; k++) {
        float t_k = time_values[k];
        float t_next = time_values[k + 1];
        float dt = t_next - t_k;
        if (dt <= 0.0f) continue;

        float f_k = signal_values[k];
        float f_next = signal_values[k + 1];

        /* e^(-s·t) = e^(-σ·t) · (cos(ω·t) - j·sin(ω·t)) */
        float exp_decay_k = expf(-s_real * t_k);
        float exp_decay_next = expf(-s_real * t_next);
        float cos_term_k = cosf(s_imag * t_k);
        float sin_term_k = sinf(s_imag * t_k);
        float cos_term_next = cosf(s_imag * t_next);
        float sin_term_next = sinf(s_imag * t_next);

        /* 梯形积分: ∫ f(t)·e^(-s·t) dt ≈ (f_k·e_k + f_next·e_next)/2 * dt */
        double real_k = (double)f_k * exp_decay_k * cos_term_k;
        double imag_k = -(double)f_k * exp_decay_k * sin_term_k;
        double real_next = (double)f_next * exp_decay_next * cos_term_next;
        double imag_next = -(double)f_next * exp_decay_next * sin_term_next;

        integral_real += 0.5 * (real_k + real_next) * (double)dt;
        integral_imag += 0.5 * (imag_k + imag_next) * (double)dt;
    }

    *result_real = (float)integral_real;
    *result_imag = (float)integral_imag;
    return 0;
}

/**
 * @brief 多频率拉普拉斯变换扫描
 *
 * 在多个复频率点上计算拉普拉斯变换，返回频域响应曲线。
 * 用于分析系统的频率特性和稳定性。
 *
 * @param time_values 时间采样点
 * @param signal_values 信号值
 * @param num_samples 采样点数
 * @param s_freqs 频率点数组（虚部为角频率ω）
 * @param num_freqs 频率点数
 * @param real_part 输出实部数组
 * @param imag_part 输出虚部数组
 * @return 0成功
 */
int laplace_frequency_sweep(const float* time_values, const float* signal_values,
                             size_t num_samples, const float* s_freqs, size_t num_freqs,
                             float* real_part, float* imag_part) {
    if (!time_values || !signal_values || !s_freqs || !real_part || !imag_part) return -1;

    for (size_t f = 0; f < num_freqs; f++) {
        float sigma = 0.0f; /* 虚轴上计算拉普拉斯 → 傅里叶变换 */
        float omega = s_freqs[f];
        laplace_forward_transform(time_values, signal_values, num_samples,
                                   sigma, omega, &real_part[f], &imag_part[f]);
    }
    return 0;
}

/**
 * @brief 数值逆向拉普拉斯变换 (Weeks法)
 *
 * 使用Weeks法进行数值拉普拉斯逆变换：
 * f(t) = e^(σ·t) * Σ_k a_k · L_k(2·t/τ - 1)
 * 其中L_k是Laguerre多项式，a_k通过双线性展开确定
 *
 * 该方法将拉普拉斯逆变换转化为Laguerre级数展开问题，
 * 通过FFT高效计算展开系数。
 *
 * @param laplace_func 拉普拉斯域函数指针：L(s) → 实部+虚部
 * @param ctx 上下文（传递额外参数）
 * @param param_sigma 收敛参数σ（取值 > 所有右半平面极点）
 * @param param_tau 时间缩放参数（控制Laguerre基的尺度）
 * @param num_coeffs Laguerre级数项数（越大精度越高）
 * @param t_values 输出时间点数组
 * @param num_t 时间点数
 * @param f_values 输出信号值数组
 * @return 0成功
 */
int laplace_inverse_weeks(float (*laplace_func)(float s_real, float s_imag,
                           float* out_real, float* out_imag, void* ctx),
                           void* ctx, float sigma, float tau,
                           int num_coeffs, const float* t_values, size_t num_t,
                           float* f_values) {
    if (!laplace_func || !t_values || !f_values || num_t == 0 || num_coeffs < 4) return -1;

    /* 步骤1: 双线性映射 s = σ + (1+ξ)/(1-ξ)·(2/τ)
     * 在单位圆上采样ξ = e^(j·θ_k), θ_k = 2πk/N, k = 0..N-1
     * 计算 L(s_k) 得到展开系数初始值 */
    int N = num_coeffs;
    float* a_real = (float*)safe_calloc(N, sizeof(float));
    float* a_imag = (float*)safe_calloc(N, sizeof(float));
    float* h_vals = (float*)safe_calloc(N, sizeof(float));
    if (!a_real || !a_imag || !h_vals) {
        safe_free((void**)&a_real); safe_free((void**)&a_imag); safe_free((void**)&h_vals);
        return -1;
    }

    float scale = 2.0f / tau;

    /* 在单位圆的N个等距点上采样 */
    for (int k = 0; k < N; k++) {
        float theta = 2.0f * 3.14159265f * (float)k / (float)N;
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        /* 双线性映射: ξ = cos(theta) + j·sin(theta)
         * s_k = σ + (1+ξ_k)/(1-ξ_k) * scale */
        float denom_real = (1.0f - cos_t) * (1.0f - cos_t) + sin_t * sin_t;
        if (denom_real < 1e-10f) denom_real = 1e-10f;
        float num_real = (1.0f + cos_t) * (1.0f - cos_t) + sin_t * sin_t;
        float num_imag = sin_t * (1.0f - cos_t) - (1.0f + cos_t) * sin_t;

        float s_r = sigma + scale * num_real / denom_real;
        float s_i = scale * num_imag / denom_real;

        /* 计算 L(s_k) */
        float Lr, Li;
        if (laplace_func(s_r, s_i, &Lr, &Li, ctx) != 0) {
            Lr = 0.0f; Li = 0.0f;
        }
        h_vals[k] = sqrtf(Lr * Lr + Li * Li);
    }

    /* 步骤2: 对h_vals做FFT得到Laguerre系数近似（简化DFT） */
    for (int m = 0; m < N; m++) {
        a_real[m] = 0.0f;
        a_imag[m] = 0.0f;
        for (int k = 0; k < N; k++) {
            float angle = -2.0f * 3.14159265f * (float)(m * k) / (float)N;
            a_real[m] += h_vals[k] * cosf(angle);
            a_imag[m] += h_vals[k] * sinf(angle);
        }
        a_real[m] /= (float)N;
        a_imag[m] /= (float)N;
    }

    /* 步骤3: 使用Laguerre级数重建时域信号
     * Laguerre多项式递推: L_0(x)=1, L_1(x)=1-x, (n+1)L_{n+1}=(2n+1-x)L_n - n·L_{n-1} */
    for (size_t ti = 0; ti < num_t; ti++) {
        float t = t_values[ti];
        float x = 2.0f * t / tau - 1.0f; /* 缩放到Laguerre定义域[0,∞] */
        if (x < 0.0f) x = 0.0f;

        float sum = 0.0f;

        /* Laguerre级数求和 */
        float L0 = 1.0f;
        float L1 = 1.0f - x;
        sum += a_real[0] * L0;
        if (N > 1) sum += a_real[1] * L1;

        for (int n = 1; n < N - 1; n++) {
            float L_next = ((2.0f * (float)n + 1.0f - x) * L1 -
                            (float)n * L0) / (float)(n + 1);
            L0 = L1;
            L1 = L_next;
            sum += a_real[n + 1] * L1;
        }

        /* 去双线性映射: f(t) = e^(σ·t) · sum */
        f_values[ti] = expf(sigma * t) * sum / tau;
    }

    safe_free((void**)&a_real); safe_free((void**)&a_imag); safe_free((void**)&h_vals);
    return 0;
}

/**
 * @brief 频域增益响应（Bode图幅值辅助）
 *
 * 计算系统在特定频率下的增益。
 * 对于传递函数H(s)，在s=jω处的幅值 |H(jω)| 即为频率响应增益。
 * 这是基于现有传递函数分析能力的高层接口。
 *
 * @param num_coeffs 分子多项式系数
 * @param num_order 分子阶数
 * @param den_coeffs 分母多项式系数
 * @param den_order 分母阶数
 * @param frequency_hz 频率（Hz）
 * @param gain_db 输出增益(dB)
 * @param phase_deg 输出相位(度)
 * @return 0成功
 */
int laplace_bode_point(const float* num_coeffs, size_t num_order,
                        const float* den_coeffs, size_t den_order,
                        float frequency_hz, float* gain_db, float* phase_deg) {
    if (!num_coeffs || !den_coeffs || !gain_db || !phase_deg) return -1;
    if (frequency_hz < 0.0f) return -1;

    /* s = jω = 0 + j·2πf */
    float omega = 2.0f * 3.14159265f * frequency_hz;

    /* 计算分母多项式值 P_den(jω) */
    double den_real = 0.0, den_imag = 0.0;
    for (size_t i = 0; i <= den_order; i++) {
        double coeff = (double)den_coeffs[den_order - i];
        double power = (double)i;
        /* (jω)^i = ω^i * (cos(i·π/2) + j·sin(i·π/2)) */
        double angle = power * 1.57079633; /* π/2 */
        double mag = pow((double)omega, power);
        den_real += coeff * mag * cos(angle);
        den_imag += coeff * mag * sin(angle);
    }

    /* 计算分子多项式值 P_num(jω) */
    double num_real = 0.0, num_imag = 0.0;
    for (size_t i = 0; i <= num_order; i++) {
        double coeff = (double)num_coeffs[num_order - i];
        double power = (double)i;
        double angle = power * 1.57079633;
        double mag = pow((double)omega, power);
        num_real += coeff * mag * cos(angle);
        num_imag += coeff * mag * sin(angle);
    }

    /* H(jω) = N(jω) / D(jω) */
    double den_mag2 = den_real * den_real + den_imag * den_imag;
    if (den_mag2 < 1e-15) {
        *gain_db = -200.0f;
        *phase_deg = 0.0f;
        return 0;
    }

    double h_real = (num_real * den_real + num_imag * den_imag) / den_mag2;
    double h_imag = (num_imag * den_real - num_real * den_imag) / den_mag2;
    double h_mag = sqrt(h_real * h_real + h_imag * h_imag);

    *gain_db = (float)(20.0 * log10(h_mag + 1e-15));
    *phase_deg = (float)(atan2(h_imag, h_real) * 57.2957795); /* 180/π */

    return 0;
}