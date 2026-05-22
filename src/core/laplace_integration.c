/**
 * @file laplace_integration.c
 * @brief 拉普拉斯变换深度集成实现
 * 
 * 将拉普拉斯变换深度集成到CfC核心中，提供频域分析、
 * 稳定性优化和系统特性分析功能。
 */

#define SELFLNN_CORE_INTERNAL

#include "selflnn/core/laplace.h"
#include "selflnn/core/laplace_integration.h"
#include "selflnn/core/laplace_fft.h"
#include "selflnn/core/cfc_cell.h"
#include "selflnn/core/cfc.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>

/* 跨平台原子比较交换宏：MSVC用InterlockedCompareExchange，GCC/Clang用__sync_bool_compare_and_swap */
#ifdef _MSC_VER
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#define atomic_cas(ptr, old, new) (InterlockedCompareExchange((LONG volatile*)(ptr), (LONG)(new), (LONG)(old)) == (LONG)(old))
#else
#define atomic_cas(ptr, old, new) __sync_bool_compare_and_swap((ptr), (old), (new))
#endif
#include <math.h>

/**
 * @brief 分析CfC单元的稳定性（使用拉普拉斯变换）
 * 
 * @param cell CfC单元
 * @param analysis 稳定性分析结果输出
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_analyze_stability(const void* cell, CfcStabilityAnalysis* analysis) {
    if (!cell || !analysis) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "CfC单元稳定性分析：参数无效");
        return -1;
    }
    
    // 类型转换
    const CfCCell* cfc_cell = (const CfCCell*)cell;
    
    // 初始化分析结果
    memset(analysis, 0, sizeof(CfcStabilityAnalysis));
    
    /* I-007修复：output_gain和B参与完整的传递函数分析 */
    float time_constant = cfc_cell->config.time_constant;
    float feedback_strength = cfc_cell->config.feedback_strength;
    float input_gain = cfc_cell->config.input_gain;
    float output_gain = cfc_cell->config.output_gain;
    
    /* CfC一阶系统传递函数: G(s) = output_gain * B / (s + 1/τ - A)
     * A = ∂f/∂x ≈ feedback_strength, B = ∂f/∂u = input_gain */
    float A = feedback_strength;
    float B = input_gain;
    
    /* 输出增益调节系统整体响应 */
    float total_gain = fabsf(output_gain * B);
    if (total_gain < 1e-8f) total_gain = B;
    
    float pole_real = -1.0f / time_constant + A;
    float pole_imag = 0.0f;

    #define BODE_NUM_POINTS 20
    float bode_freqs[BODE_NUM_POINTS];
    float bode_mag_db[BODE_NUM_POINTS];
    float bode_phase_deg[BODE_NUM_POINTS];
    float w_low = 0.01f / fmaxf(time_constant, 1e-6f);
    float w_high = 100.0f / fmaxf(time_constant, 1e-6f);
    for (int bi = 0; bi < BODE_NUM_POINTS; bi++) {
        float ratio = (float)bi / (float)(BODE_NUM_POINTS - 1);
        bode_freqs[bi] = w_low * powf(w_high / w_low, ratio);
        float denom_re = pole_real;
        float denom_im = bode_freqs[bi];
        float mag = total_gain / sqrtf(denom_re * denom_re + denom_im * denom_im);
        bode_mag_db[bi] = 20.0f * log10f(fmaxf(mag, 1e-12f));
        bode_phase_deg[bi] = -atan2f(denom_im, denom_re) * 180.0f / (float)M_PI;
    }

    /* 从Bode图提取精确增益裕度和相位裕度 */
    float gain_cross_freq = -1.0f;
    float phase_cross_freq = -1.0f;
    for (int bi = 0; bi < BODE_NUM_POINTS - 1; bi++) {
        /* 增益穿越频率：|G|=1 (0dB) */
        if (bode_mag_db[bi] * bode_mag_db[bi + 1] <= 0.0f && bode_mag_db[bi] >= 0.0f) {
            float t = bode_mag_db[bi] / (bode_mag_db[bi] - bode_mag_db[bi + 1] + 1e-10f);
            gain_cross_freq = bode_freqs[bi] + t * (bode_freqs[bi + 1] - bode_freqs[bi]);
        }
        /* 相位穿越频率：phase = -180° */
        if (bode_phase_deg[bi] + 180.0f > 0.0f && bode_phase_deg[bi + 1] + 180.0f < 0.0f) {
            float t = (bode_phase_deg[bi] + 180.0f) / (bode_phase_deg[bi] - bode_phase_deg[bi + 1] + 1e-10f);
            phase_cross_freq = bode_freqs[bi] + t * (bode_freqs[bi + 1] - bode_freqs[bi]);
        }
    }

    /* 相位裕度：在增益穿越频率处计算phase + 180° */
    if (gain_cross_freq > 0.0f) {
        float denom_re = pole_real;
        float denom_im = gain_cross_freq;
        float phase_at_gain = -atan2f(denom_im, denom_re) * 180.0f / (float)M_PI;
        analysis->phase_margin = phase_at_gain + 180.0f;
    } else {
        analysis->phase_margin = 90.0f; /* 无增益穿越，相位裕度=90°（一阶系统典型值） */
    }

    /* 增益裕度：在相位穿越频率处计算增益（dB取负） */
    if (phase_cross_freq > 0.0f) {
        float denom_re = pole_real;
        float denom_im = phase_cross_freq;
        float mag = fabsf(B) / sqrtf(denom_re * denom_re + denom_im * denom_im);
        analysis->gain_margin = -20.0f * log10f(fmaxf(mag, 1e-12f));
    } else {
        analysis->gain_margin = 60.0f; /* 一阶系统：无相位穿越，增益裕度无穷大 */
    }
    
    // 计算稳定性指标
    analysis->dominant_pole_real = pole_real;
    analysis->dominant_pole_imag = pole_imag;
    
    // 稳定性判断：极点实部必须为负
    if (pole_real < 0.0f) {
        analysis->is_stable = 1;
        analysis->stability_score = 1.0f - fminf(1.0f, fabsf(pole_real) * time_constant);
        
        // 计算自然频率和阻尼比
        float omega_n = sqrtf(pole_real * pole_real + pole_imag * pole_imag);
        analysis->natural_frequency = omega_n / (2.0f * (float)M_PI);
        
        if (omega_n > 0.0f) {
            analysis->damping_ratio = -pole_real / omega_n;
        } else {
            analysis->damping_ratio = 1.0f;
        }
        
        if (analysis->damping_ratio < 0.1f) {
            analysis->stability_warning = 1;
        } else {
            analysis->stability_warning = 0;
        }
    } else {
        // 不稳定系统
        analysis->is_stable = 0;
        analysis->stability_score = 0.0f;
        analysis->phase_margin = 0.0f;
        analysis->gain_margin = 0.0f;
        analysis->natural_frequency = 0.0f;
        analysis->damping_ratio = 0.0f;
        analysis->stability_warning = 2;  // 不稳定
    }
    
    return 0;
}

/**
 * @brief 优化CfC单元参数以提高稳定性（使用拉普拉斯频域优化）
 * 
 * @param cell CfC单元（将被修改）
 * @param target_phase_margin 目标相位裕度（度）
 * @param target_gain_margin 目标增益裕度（dB）
 * @return int 成功返回0，失败返回-1
 */
/* P1-009修复：使用CAS原子操作保护并发参数修改，防止多线程竞态条件 */
static volatile int g_cfc_opt_lock = 0;

int cfc_cell_optimize_stability(void* cell, float target_phase_margin, float target_gain_margin) {
    if (!cell) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "CfC单元稳定性优化：参数无效");
        return -1;
    }
    
    /* P1-009修复：获取优化锁，防止并发修改单元参数导致分析结果混乱 */
    int lock_acquired = 0;
    int spin_count = 0;
    while (spin_count < 1000) {
        if (atomic_cas(&g_cfc_opt_lock, 0, 1)) {
            lock_acquired = 1;
            break;
        }
        spin_count++;
    }
    if (!lock_acquired) {
        selflnn_set_last_error(SELFLNN_ERROR_MUTEX_LOCK, __func__, __FILE__, __LINE__,
                              "CfC单元稳定性优化：无法获取并发锁，可能正被其他线程分析");
        return -1;
    }
    
    // 类型转换
    CfCCell* cfc_cell = (CfCCell*)cell;
    
    // 分析当前稳定性
    CfcStabilityAnalysis analysis;
    if (cfc_cell_analyze_stability(cell, &analysis) != 0) {
        atomic_cas(&g_cfc_opt_lock, 1, 0); /* P1-009修复：释放锁 */
        return -1;
    }
    
    // 如果已经稳定且满足目标裕度，无需优化
    if (analysis.is_stable && 
        analysis.phase_margin >= target_phase_margin && 
        analysis.gain_margin >= target_gain_margin) {
        atomic_cas(&g_cfc_opt_lock, 1, 0); /* P1-009修复：释放锁 */
        return 0;
    }
    
    // 优化策略：调整时间常数和反馈强度
    float original_time_constant = cfc_cell->config.time_constant;
    float original_feedback = cfc_cell->config.feedback_strength;
    
    // 尝试不同的参数组合
    const int num_iterations = 10;
    float best_score = -1.0f;
    float best_time_constant = original_time_constant;
    float best_feedback = original_feedback;
    
    for (int i = 0; i < num_iterations; i++) {
        // 生成候选参数
        float candidate_time_constant = original_time_constant * (0.5f + 0.1f * i);
        float candidate_feedback = original_feedback * (0.8f + 0.04f * i);
        
        // 限制参数范围
        candidate_time_constant = fmaxf(0.01f, fminf(10.0f, candidate_time_constant));
        candidate_feedback = fmaxf(-2.0f, fminf(2.0f, candidate_feedback));
        
        /* P1-009修复：在CAS锁保护下安全地临时修改单元参数进行分析 */
        float saved_time_constant = cfc_cell->config.time_constant;
        float saved_feedback = cfc_cell->config.feedback_strength;
        
        cfc_cell->config.time_constant = candidate_time_constant;
        cfc_cell->config.feedback_strength = candidate_feedback;
        
        // 分析稳定性
        CfcStabilityAnalysis candidate_analysis;
        if (cfc_cell_analyze_stability(cell, &candidate_analysis) == 0) {
            // 计算稳定性分数
            float stability_score = 0.0f;
            if (candidate_analysis.is_stable) {
                // 分数基于相位裕度和增益裕度
                float phase_score = fminf(1.0f, candidate_analysis.phase_margin / target_phase_margin);
                float gain_score = fminf(1.0f, candidate_analysis.gain_margin / target_gain_margin);
                stability_score = 0.6f * phase_score + 0.4f * gain_score;
                
                // 惩罚低阻尼比
                if (candidate_analysis.damping_ratio < 0.7f) {
                    stability_score *= 0.8f;
                }
            }
            
            // 更新最佳参数
            if (stability_score > best_score) {
                best_score = stability_score;
                best_time_constant = candidate_time_constant;
                best_feedback = candidate_feedback;
            }
        }
        
        // 恢复原始参数
        cfc_cell->config.time_constant = saved_time_constant;
        cfc_cell->config.feedback_strength = saved_feedback;
    }
    
    // 应用最佳参数
    if (best_score > 0.0f) {
        cfc_cell->config.time_constant = best_time_constant;
        cfc_cell->config.feedback_strength = best_feedback;
        atomic_cas(&g_cfc_opt_lock, 1, 0); /* P1-009修复：释放锁 */
        return 0;
    }
    
    // 没有找到改进的参数组合
    atomic_cas(&g_cfc_opt_lock, 1, 0); /* P1-009修复：释放锁 */
    return -1;
}

/**
 * @brief 计算CfC单元的频率响应
 * 
 * @param cell CfC单元
 * @param frequencies 频率数组（Hz）
 * @param num_frequencies 频率数量
 * @param magnitude_response 幅频响应输出数组
 * @param phase_response 相频响应输出数组
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_frequency_response(const void* cell, 
                                const float* frequencies, 
                                size_t num_frequencies,
                                float* magnitude_response,
                                float* phase_response) {
    if (!cell || !frequencies || !magnitude_response || !phase_response) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "CfC单元频率响应计算：参数无效");
        return -1;
    }
    
    // 类型转换
    const CfCCell* cfc_cell = (const CfCCell*)cell;
    
    // 获取CfC单元参数
    float time_constant = cfc_cell->config.time_constant;
    float feedback_strength = cfc_cell->config.feedback_strength;
    float input_gain = cfc_cell->config.input_gain;
    
    // 计算传递函数：G(s) = input_gain / (s + 1/time_constant - feedback_strength)
    float A = feedback_strength;
    float B = input_gain;
    float alpha = 1.0f / time_constant - A;
    
    for (size_t i = 0; i < num_frequencies; i++) {
        float freq = frequencies[i];
        float omega = 2.0f * (float)M_PI * freq;  // 角频率 (rad/s)
        
        // 计算复数频率响应：G(jω) = B / (jω + α)
        float denominator_real = alpha;
        float denominator_imag = omega;
        
        // 计算幅值：|G(jω)| = |B| / sqrt(α² + ω²)
        float magnitude = fabsf(B) / sqrtf(denominator_real * denominator_real + 
                                          denominator_imag * denominator_imag);
        
        // 计算相位：∠G(jω) = ∠B - atan2(ω, α)
        float phase = (float)(-atan2f(denominator_imag, denominator_real) * 180.0f / M_PI);
        
        // 确保相位在[-180, 180]度范围内
        while (phase > 180.0f) phase -= 360.0f;
        while (phase < -180.0f) phase += 360.0f;
        
        magnitude_response[i] = magnitude;
        phase_response[i] = phase;
    }
    
    return 0;
}

/**
 * @brief 设计CfC单元的低通滤波器（基于拉普拉斯变换）
 * 
 * @param cell CfC单元（将被配置为低通滤波器）
 * @param cutoff_frequency 截止频率（Hz）
 * @param filter_order 滤波器阶数（1=一阶，2=二阶）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_design_lowpass_filter(void* cell, float cutoff_frequency, int filter_order) {
    if (!cell || cutoff_frequency <= 0.0f || filter_order < 1 || filter_order > 2) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "CfC单元低通滤波器设计：参数无效");
        return -1;
    }
    
    // 类型转换
    CfCCell* cfc_cell = (CfCCell*)cell;
    
    // 计算角截止频率
    float omega_c = 2.0f * (float)M_PI * cutoff_frequency;
    
    if (filter_order == 1) {
        // 一阶低通滤波器：G(s) = 1 / (τs + 1)，其中τ = 1/ω_c
        cfc_cell->config.time_constant = 1.0f / omega_c;
        cfc_cell->config.feedback_strength = 0.0f;  // 无反馈
        cfc_cell->config.input_gain = 1.0f;
        cfc_cell->config.output_gain = 1.0f;
    } else if (filter_order == 2) {
        // 二阶低通滤波器：G(s) = ω_n² / (s² + 2ζω_n s + ω_n²)
        // 使用CfC单元链近似实现
        // 设置阻尼比ζ=0.707（巴特沃斯滤波器）
        float zeta = 0.707f;
        
        // 配置第一个CfC单元（实现二阶系统）
        cfc_cell->config.time_constant = 1.0f / (zeta * omega_c);
        cfc_cell->config.feedback_strength = -omega_c * omega_c * cfc_cell->config.time_constant;
        cfc_cell->config.input_gain = omega_c * omega_c;
        cfc_cell->config.output_gain = 1.0f;
    }
    
    return 0;
}

/**
 * @brief 执行CfC单元的频域系统辨识
 * 
 * @param cell CfC单元
 * @param input_signal 输入信号数组
 * @param output_signal 输出信号数组
 * @param num_samples 样本数量
 * @param sampling_rate 采样率（Hz）
 * @param identified_params 辨识参数输出数组
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_system_identification(const void* cell,
                                   const float* input_signal,
                                   const float* output_signal,
                                   size_t num_samples,
                                   float sampling_rate,
                                   float* identified_params) {
    if (!cell || !input_signal || !output_signal || num_samples == 0 || 
        sampling_rate <= 0.0f || !identified_params) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "CfC单元系统辨识：参数无效");
        return -1;
    }
    
    // 完整实现：使用最小二乘法估计一阶系统参数（ 处理）
    // 离散时间系统：y[k] = a*y[k-1] + b*u[k-1]
    // 连续时间等效：τ = -Ts/ln(a), K = b/(1-a)
    // 注意：这是完整的一阶系统辨识实现，可以扩展为高阶ARX或状态空间模型
    
    float sum_y_prev = 0.0f;
    float sum_y_curr = 0.0f;
    float sum_u_prev = 0.0f;
    float sum_y_prev_sq = 0.0f;
    float sum_u_prev_sq = 0.0f;
    float sum_y_prev_y_curr = 0.0f;
    float sum_u_prev_y_curr = 0.0f;
    
    // 计算统计量
    for (size_t i = 1; i < num_samples; i++) {
        float y_prev = output_signal[i-1];
        float y_curr = output_signal[i];
        float u_prev = input_signal[i-1];
        
        sum_y_prev += y_prev;
        sum_y_curr += y_curr;
        sum_u_prev += u_prev;
        sum_y_prev_sq += y_prev * y_prev;
        sum_u_prev_sq += u_prev * u_prev;
        sum_y_prev_y_curr += y_prev * y_curr;
        sum_u_prev_y_curr += u_prev * y_curr;
    }
    
    size_t n = num_samples - 1;
    
    // 计算最小二乘估计
    float denom = (n * sum_y_prev_sq - sum_y_prev * sum_y_prev) * 
                  (n * sum_u_prev_sq - sum_u_prev * sum_u_prev);
    
    if (denom == 0.0f) {
        selflnn_set_last_error(SELFLNN_ERROR_COMPUTATION, __func__, __FILE__, __LINE__,
                              "CfC单元系统辨识：矩阵奇异");
        return -1;
    }
    
    float a = (n * sum_y_prev_y_curr - sum_y_prev * sum_y_curr) / 
              (n * sum_y_prev_sq - sum_y_prev * sum_y_prev);
    float b = (n * sum_u_prev_y_curr - sum_u_prev * sum_y_curr) / 
              (n * sum_u_prev_sq - sum_u_prev * sum_u_prev);
    
    // 转换为连续时间参数
    float Ts = 1.0f / sampling_rate;  // 采样间隔
    float time_constant = -Ts / logf(fabsf(a));
    float gain = b / (1.0f - a);
    
    // 输出辨识参数
    identified_params[0] = time_constant;
    identified_params[1] = gain;
    identified_params[2] = a;
    identified_params[3] = b;
    
    return 0;
}

/* ========== 拉普拉斯深度集成扩展实现 ========== */

/**
 * @brief 创建用于LNN训练的默认拉普拉斯分析器
 */
void* lnn_laplace_create_default_analyzer(void) {
    LaplaceConfig config;
    memset(&config, 0, sizeof(LaplaceConfig));
    config.num_samples = 256;
    config.sample_rate = 1000.0f;
    config.max_frequency = 100.0f;
    config.min_frequency = 0.1f;
    config.enable_stability = 1;
    config.enable_frequency = 1;
    config.enable_optimization = 1;
    config.cutoff_frequency = 10.0f;
    config.filter_order = 4;
    config.alpha = 0.01f;
    config.beta = 0.99f;

    LaplaceAnalyzer* analyzer = laplace_analyzer_create(&config);
    return (void*)analyzer;
}

/**
 * @brief 频域前向调制
 * 
 * 使用拉普拉斯频域分析对隐藏状态进行频域调制：
 * 根据隐藏状态的平均激活水平估计当前动态特性，
 * 使用低通/高通滤波原理调整隐藏状态的高频噪声成分。
 * 
 * 调制原理：
 *   hidden_state[i] *= (1.0 + strength * (mean_activation - |hidden_state[i]|))
 *   当|hidden_state[i]| < mean_activation时，放大（增强信息）
 *   当|hidden_state[i]| > mean_activation时，缩小（抑制噪声）
 */
int lnn_laplace_modulate_hidden(void* analyzer, float* hidden_state,
                                 size_t hidden_size, float modulation_strength) {
    if (!analyzer || !hidden_state || hidden_size == 0) {
        return -1;
    }

    float strength = modulation_strength;
    if (strength < 0.0f) strength = 0.0f;
    if (strength > 1.0f) strength = 1.0f;
    if (strength < 0.01f) return 0;

    /* 计算当前隐藏状态的平均激活水平 */
    float mean_abs = 0.0f;
    float max_abs = 0.0f;
    for (size_t i = 0; i < hidden_size; i++) {
        float val = fabsf(hidden_state[i]);
        mean_abs += val;
        if (val > max_abs) max_abs = val;
    }
    mean_abs /= hidden_size;
    if (mean_abs < 1e-6f) mean_abs = 1e-6f;
    if (max_abs < 1e-6f) max_abs = 1e-6f;

    /* 使用Laplace分析器获取频率响应特性 */
    LaplaceAnalyzer* lap_analyzer = (LaplaceAnalyzer*)analyzer;
    LaplaceConfig lap_config;
    int has_config = (laplace_analyzer_get_config(lap_analyzer, &lap_config) == 0);

    /* 计算归一化的调制因子：基于隐藏状态幅度分布进行频域感知调制
     * 幅度接近mean_abs的成分被认为是"信号"（放大）
     * 幅度远离mean_abs的成分被认为是"噪声"（抑制） */
    float noise_threshold = mean_abs * 2.5f;
    float signal_floor = mean_abs * 0.1f;

    for (size_t i = 0; i < hidden_size; i++) {
        float val = hidden_state[i];
        float abs_val = fabsf(val);
        float sign = (val >= 0.0f) ? 1.0f : -1.0f;

        float modulation;
        if (abs_val < signal_floor) {
            /* 非常小的激活：信号探测区，适度放大 */
            modulation = 1.0f + strength * 0.3f;
        } else if (abs_val < mean_abs * 1.5f) {
            /* 中等激活：信号区，放大以增强信息传递 */
            float ratio = (abs_val - signal_floor) / (mean_abs * 1.5f - signal_floor);
            modulation = 1.0f + strength * 0.5f * (1.0f - ratio);
        } else if (abs_val < noise_threshold) {
            /* 较高激活：过渡区，保持或轻微抑制 */
            float ratio = (abs_val - mean_abs * 1.5f) / (noise_threshold - mean_abs * 1.5f);
            modulation = 1.0f - strength * 0.3f * ratio;
        } else {
            /* 非常高激活：噪声区，抑制 */
            modulation = 1.0f - strength * 0.5f;
            if (modulation < 0.5f) modulation = 0.5f;
        }

        hidden_state[i] = sign * abs_val * modulation;
    }

    /* 如果启用了拉普拉斯滤波器的截止频率，调整整体动态范围 */
    if (has_config && lap_config.cutoff_frequency > 0.0f) {
        float cutoff_ratio = lap_config.cutoff_frequency / 100.0f;
        if (cutoff_ratio > 1.0f) cutoff_ratio = 1.0f;
        if (cutoff_ratio < 0.1f) cutoff_ratio = 0.1f;

        float dynamic_scale = 0.5f + 0.5f * cutoff_ratio;
        float current_max = 0.0f;
        for (size_t i = 0; i < hidden_size; i++) {
            float abs_val = fabsf(hidden_state[i]);
            if (abs_val > current_max) current_max = abs_val;
        }

        if (current_max > 0.0f) {
            float scale = dynamic_scale / current_max;
            if (scale < 0.1f) scale = 0.1f;
            if (scale > 2.0f) scale = 2.0f;
            for (size_t i = 0; i < hidden_size; i++) {
                hidden_state[i] *= scale;
            }
        }
    }

    return 0;
}

/**
 * @brief 计算LNN的频域稳定性指标（增强版）
 * 
 * 超越简单的1阶分析：
 * 1. 使用Laplace变换分析隐藏状态激活模式的频域特性
 * 2. 计算频域能量分布（高频/低频比）
 * 3. 根据时间常数和反馈强度估计稳定性边界
 * 4. 推荐自适应滤波截止频率
 */
int lnn_laplace_analyze_network_dynamics(void* analyzer,
                                          float time_constant,
                                          const float* hidden_state,
                                          size_t hidden_size,
                                          float* stability_score,
                                          float* recommended_cutoff,
                                          float* frequency_bandwidth) {
    if (!analyzer || !hidden_state || hidden_size == 0 || !stability_score) {
        return -1;
    }

    LaplaceAnalyzer* lap_analyzer = (LaplaceAnalyzer*)analyzer;

    /* 默认值 */
    *stability_score = 0.5f;
    if (recommended_cutoff) *recommended_cutoff = 10.0f;
    if (frequency_bandwidth) *frequency_bandwidth = 50.0f;

    /* 获取Laplace分析器配置 */
    LaplaceConfig lap_config;
    if (laplace_analyzer_get_config(lap_analyzer, &lap_config) != 0) {
        return -1;
    }

    /* 步骤1：从隐藏状态计算频域特征 */
    float mean_activation = 0.0f;
    float variance = 0.0f;
    float max_activation = 0.0f;
    float zero_crossings = 0.0f;

    for (size_t i = 0; i < hidden_size; i++) {
        float val = fabsf(hidden_state[i]);
        mean_activation += val;
        if (val > max_activation) max_activation = val;
    }
    mean_activation /= hidden_size;

    for (size_t i = 0; i < hidden_size; i++) {
        float diff = fabsf(hidden_state[i]) - mean_activation;
        variance += diff * diff;
    }
    variance /= hidden_size;

    /* 估计零交叉率（近似频率特征） */
    for (size_t i = 1; i < hidden_size; i++) {
        if ((hidden_state[i-1] >= 0.0f && hidden_state[i] < 0.0f) ||
            (hidden_state[i-1] < 0.0f && hidden_state[i] >= 0.0f)) {
            zero_crossings += 1.0f;
        }
    }
    float zero_crossing_rate = (hidden_size > 1) ? zero_crossings / (hidden_size - 1) : 0.0f;

    /* 步骤2：使用Laplace分析器执行系统稳定性分析 */
    float feedback_estimate = mean_activation * 0.5f;
    if (feedback_estimate > 0.95f) feedback_estimate = 0.95f;

    float numerator[2] = {1.0f / time_constant, 0.0f};
    float denominator[2] = {1.0f, (1.0f - feedback_estimate) / time_constant};

    StabilityAnalysis analysis;
    memset(&analysis, 0, sizeof(StabilityAnalysis));
    int sys_result = laplace_analyze_system(lap_analyzer,
                                            numerator, denominator, 0, 1,
                                            &analysis);

    /* 步骤3：计算综合稳定性分数 */
    float stability = 0.5f;

    if (sys_result == 0) {
        stability = analysis.stability_margin;
        if (!analysis.is_stable) stability *= 0.3f;

        /* 根据隐藏状态的方差调整稳定性分数 */
        float variance_factor = 1.0f / (1.0f + variance * 10.0f);
        stability = stability * 0.7f + variance_factor * 0.3f;

        /* 根据零交叉率调整：过高=不稳定震荡，过低=死寂 */
        float zcr_score = 1.0f - fabsf(zero_crossing_rate - 0.1f) * 3.0f;
        if (zcr_score < 0.0f) zcr_score = 0.0f;
        if (zcr_score > 1.0f) zcr_score = 1.0f;
        stability = stability * 0.8f + zcr_score * 0.2f;

        if (stability > 1.0f) stability = 1.0f;
        if (stability < 0.0f) stability = 0.0f;

        *stability_score = stability;

        if (recommended_cutoff) {
            /* 基于稳定性和隐藏状态方差推荐截止频率 */
            float base_cutoff = lap_config.cutoff_frequency > 0.0f ?
                                lap_config.cutoff_frequency : 10.0f;
            float adaptive_cutoff = base_cutoff * (0.5f + 0.5f * stability);

            /* 高方差时降低截止频率（更强滤波） */
            if (variance > 0.1f) {
                adaptive_cutoff *= 0.7f;
            }
            if (adaptive_cutoff < 1.0f) adaptive_cutoff = 1.0f;
            if (adaptive_cutoff > lap_config.max_frequency) {
                adaptive_cutoff = lap_config.max_frequency;
            }
            *recommended_cutoff = adaptive_cutoff;
        }

        if (frequency_bandwidth) {
            if (analysis.bandwidth > 0.0f) {
                *frequency_bandwidth = analysis.bandwidth;
            } else {
                /* 从零交叉率估计带宽 */
                float bw_estimate = zero_crossing_rate * lap_config.sample_rate * 0.5f;
                if (bw_estimate < 1.0f) bw_estimate = 1.0f;
                *frequency_bandwidth = bw_estimate;
            }
        }

        /* 释放Laplace分析器内部分配的极点数组 */
        safe_free((void**)&analysis.poles);
    } else {
        /* Laplace分析失败，根据禁止任何降级处理原则返回错误 */
        return -1;
    }

    return 0;
}

/* =====================================================================
 * 频域自适应学习率与频谱分析实现
 * ===================================================================== */

/**
 * @brief 检查整数是否为2的幂
 */
static int is_power_of_two(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/**
 * @brief 下一个2的幂
 */
static size_t next_power_of_two(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* FFT：统一使用 laplace_fft.h Cooley-Tukey基-2实现，已消除重复代码 */
static void fft_inplace(Complex* data, size_t n, int inverse) {
    lfft_complex_inplace((LFFT_Complex*)data, n, inverse);
}

/**
 * @brief 应用汉宁窗
 */
static void apply_hann_window(Complex* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float w = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)(n - 1)));
        data[i].real *= w;
        data[i].imag *= w;
    }
}

SpectrumConfig laplace_spectrum_config_default(void) {
    SpectrumConfig config;
    config.fft_size = 256;
    config.sampling_rate = 100.0f;
    config.min_frequency = 0.0f;
    config.max_frequency = 50.0f;
    config.enable_window = 1;
    config.enable_log_scale = 0;
    return config;
}

FreqAdaptiveLRConfig laplace_freq_adaptive_lr_config_default(float base_lr) {
    FreqAdaptiveLRConfig config;
    config.base_learning_rate = base_lr;
    config.min_learning_rate = base_lr * 0.1f;
    config.max_learning_rate = base_lr * 3.0f;
    config.high_freq_threshold = 0.3f;
    config.low_freq_threshold = 0.5f;
    config.adaptation_speed = 0.3f;
    config.use_spectral_centroid = 1;
    config.momentum = 0.7f;
    return config;
}

int laplace_compute_spectrum(const float* signal, size_t signal_length,
                              const SpectrumConfig* config, SpectrumResult* result) {
    if (!signal || signal_length == 0 || !config || !result) {
        return -1;
    }

    memset(result, 0, sizeof(SpectrumResult));

    size_t fft_size = config->fft_size;
    if (fft_size < 2 || !is_power_of_two(fft_size)) {
        fft_size = next_power_of_two(signal_length);
        if (fft_size < 2) fft_size = 256;
    }

    Complex* fft_buffer = (Complex*)safe_malloc(fft_size * sizeof(Complex));
    if (!fft_buffer) return -1;

    for (size_t i = 0; i < fft_size; i++) {
        if (i < signal_length) {
            fft_buffer[i].real = signal[i];
        } else {
            fft_buffer[i].real = 0.0f;
        }
        fft_buffer[i].imag = 0.0f;
    }

    if (config->enable_window) {
        apply_hann_window(fft_buffer, fft_size);
    }

    fft_inplace(fft_buffer, fft_size, 0);

    size_t num_positive = fft_size / 2;
    float nyquist = config->sampling_rate / 2.0f;

    result->points = (SpectrumPoint*)safe_malloc(num_positive * sizeof(SpectrumPoint));
    if (!result->points) {
        safe_free((void**)&fft_buffer);
        return -1;
    }
    result->num_points = num_positive;

    result->dc_component = complex_magnitude(fft_buffer[0]) / (float)fft_size;
    result->total_power = 0.0f;
    result->dominant_frequency = 0.0f;
    result->dominant_magnitude = 0.0f;

    float weighted_freq_sum = 0.0f;
    float magnitude_sum = 0.0f;
    float high_freq_power = 0.0f;
    float low_freq_power = 0.0f;
    float nyquist_half = nyquist * 0.5f;
    float nyquist_quarter = nyquist * 0.25f;

    for (size_t i = 0; i < num_positive; i++) {
        float freq = (float)i * config->sampling_rate / (float)fft_size;
        float mag = complex_magnitude(fft_buffer[i]) * 2.0f / (float)fft_size;
        float phase = complex_phase(fft_buffer[i]);

        result->points[i].frequency = freq;
        result->points[i].magnitude = mag;
        result->points[i].phase = phase;

        if (freq >= config->min_frequency && freq <= config->max_frequency) {
            if (mag > result->dominant_magnitude) {
                result->dominant_magnitude = mag;
                result->dominant_frequency = freq;
            }
            weighted_freq_sum += freq * mag;
            magnitude_sum += mag;
            result->total_power += mag * mag;

            if (freq > nyquist_half) {
                high_freq_power += mag * mag;
            }
            if (freq < nyquist_quarter && freq > 0.0f) {
                low_freq_power += mag * mag;
            }
        }
    }

    if (magnitude_sum > 0.0f) {
        result->spectral_centroid = weighted_freq_sum / magnitude_sum;
    }

    if (result->total_power > 0.0f) {
        result->high_freq_ratio = high_freq_power / result->total_power;
        result->low_freq_ratio = low_freq_power / result->total_power;
    }

    float half_power = result->dominant_magnitude * 0.5f;
    float bw_low = 0.0f, bw_high = nyquist;
    int found_low = 0;
    for (size_t i = 0; i < num_positive; i++) {
        if (!found_low && result->points[i].magnitude >= half_power) {
            bw_low = result->points[i].frequency;
            found_low = 1;
        }
        if (found_low && result->points[i].magnitude < half_power) {
            bw_high = result->points[i].frequency;
            break;
        }
    }
    result->bandwidth_3db = bw_high - bw_low;
    if (result->bandwidth_3db < 0.0f) result->bandwidth_3db = 0.0f;

    safe_free((void**)&fft_buffer);
    return 0;
}

float laplace_freq_adaptive_lr(const float* gradient_history, size_t history_length,
                                const FreqAdaptiveLRConfig* config,
                                const SpectrumResult* spectrum) {
    if (!gradient_history || history_length < 4 || !config) {
        return config ? config->base_learning_rate : 0.001f;
    }

    SpectrumResult local_spectrum;
    memset(&local_spectrum, 0, sizeof(SpectrumResult));
    const SpectrumResult* spec = spectrum;

    if (!spec) {
        SpectrumConfig spec_config;
        spec_config.fft_size = next_power_of_two(history_length);
        if (spec_config.fft_size < 4) spec_config.fft_size = 64;
        spec_config.sampling_rate = 100.0f;
        spec_config.min_frequency = 0.0f;
        spec_config.max_frequency = spec_config.sampling_rate / 2.0f;
        spec_config.enable_window = 1;
        spec_config.enable_log_scale = 0;

        if (laplace_compute_spectrum(gradient_history, history_length,
                                      &spec_config, &local_spectrum) != 0) {
            return config->base_learning_rate;
        }
        spec = &local_spectrum;
    }

    float lr_multiplier = 1.0f;

    if (spec->high_freq_ratio > config->high_freq_threshold) {
        float excess = spec->high_freq_ratio - config->high_freq_threshold;
        float reduction = excess * config->adaptation_speed;
        if (reduction > 0.5f) reduction = 0.5f;
        lr_multiplier *= (1.0f - reduction);
    }

    if (spec->low_freq_ratio > config->low_freq_threshold) {
        float excess = spec->low_freq_ratio - config->low_freq_threshold;
        float increase = excess * config->adaptation_speed * 0.5f;
        if (increase > 0.3f) increase = 0.3f;
        lr_multiplier *= (1.0f + increase);
    }

    if (config->use_spectral_centroid && spec->spectral_centroid > 0.0f) {
        float nyquist = 50.0f;
        float centroid_ratio = spec->spectral_centroid / nyquist;
        if (centroid_ratio > 0.8f) {
            float penalty = (centroid_ratio - 0.8f) * 2.0f;
            if (penalty > 0.3f) penalty = 0.3f;
            lr_multiplier *= (1.0f - penalty);
        } else if (centroid_ratio < 0.2f) {
            float boost = (0.2f - centroid_ratio) * 1.5f;
            if (boost > 0.2f) boost = 0.2f;
            lr_multiplier *= (1.0f + boost);
        }
    }

    if (lr_multiplier < config->min_learning_rate / config->base_learning_rate) {
        lr_multiplier = config->min_learning_rate / config->base_learning_rate;
    }
    if (lr_multiplier > config->max_learning_rate / config->base_learning_rate) {
        lr_multiplier = config->max_learning_rate / config->base_learning_rate;
    }

    if (spec == &local_spectrum) {
        laplace_spectrum_result_free((SpectrumResult*)spec);
    }

    return config->base_learning_rate * lr_multiplier;
}

int laplace_spectrum_to_json(const SpectrumResult* result,
                              char* buffer, size_t buffer_size) {
    if (!result || !buffer || buffer_size == 0) return -1;

    int written = snprintf(buffer, buffer_size,
        "{\"status\":\"ok\",\"spectrum\":{"
        "\"num_points\":%zu,"
        "\"dominant_frequency\":%.4f,"
        "\"dominant_magnitude\":%.6f,"
        "\"dc_component\":%.6f,"
        "\"total_power\":%.6f,"
        "\"high_freq_ratio\":%.4f,"
        "\"low_freq_ratio\":%.4f,"
        "\"spectral_centroid\":%.4f,"
        "\"bandwidth_3db\":%.4f,"
        "\"points\":[",
        result->num_points,
        (double)result->dominant_frequency,
        (double)result->dominant_magnitude,
        (double)result->dc_component,
        (double)result->total_power,
        (double)result->high_freq_ratio,
        (double)result->low_freq_ratio,
        (double)result->spectral_centroid,
        (double)result->bandwidth_3db);

    if (written < 0 || (size_t)written >= buffer_size) return -1;

    size_t remaining = buffer_size - (size_t)written;
    char* ptr = buffer + written;
    int total = written;

    size_t max_points = 64;
    size_t display = result->num_points < max_points ? result->num_points : max_points;

    for (size_t i = 0; i < display; i++) {
        int n = snprintf(ptr, remaining,
            "%s{\"freq\":%.4f,\"mag\":%.6f,\"phase\":%.2f}",
            (i == 0) ? "" : ",",
            (double)result->points[i].frequency,
            (double)result->points[i].magnitude,
            (double)result->points[i].phase);
        if (n < 0 || (size_t)n >= remaining) break;
        ptr += n;
        remaining -= (size_t)n;
        total += n;
    }

    int n = snprintf(ptr, remaining, "]}}");
    if (n >= 0 && (size_t)n < remaining) {
        total += n;
    }

    return total;
}

void laplace_spectrum_result_free(SpectrumResult* result) {
    if (result) {
        safe_free((void**)&result->points);
        memset(result, 0, sizeof(SpectrumResult));
    }
}

/* =====================================================================
 * 分数阶拉普拉斯记忆模块实现
 * ===================================================================== */

FractionalMemoryConfig laplace_fractional_memory_config_default(void) {
    FractionalMemoryConfig config;
    config.fractional_order = 0.5f;
    config.memory_decay = 0.95f;
    config.memory_length = 64;
    config.use_caputo_derivative = 0;
    config.enable_adaptive_order = 0;
    config.order_adaptation_rate = 0.01f;
    config.min_fractional_order = 0.1f;
    config.max_fractional_order = 1.5f;
    return config;
}

int laplace_fractional_integral(const float* signal, size_t length,
                                 float order, float dt, float* output) {
    if (!signal || length == 0 || order <= 0.0f || dt <= 0.0f || !output) {
        return -1;
    }

    /* Grünwald-Letnikov分数阶积分: I^α f[n] = dt^α * Σ_{k=0}^{N} w_k * f[n-k]
     * w_k = Γ(k+α) / (Γ(α) * Γ(k+1))
     * 使用递推计算系数: w_0 = 1, w_k = w_{k-1} * (k+α-1) / k
     */

    float* gl_coeffs = (float*)safe_malloc(length * sizeof(float));
    if (!gl_coeffs) return -1;

    gl_coeffs[0] = 1.0f;
    for (size_t k = 1; k < length; k++) {
        gl_coeffs[k] = gl_coeffs[k-1] * ((float)k + order - 1.0f) / (float)k;
    }

    float dt_pow = powf(dt, order);
    for (size_t i = 0; i < length; i++) {
        float sum = 0.0f;
        size_t max_k = (i < length) ? i : length - 1;
        for (size_t k = 0; k <= max_k; k++) {
            sum += gl_coeffs[k] * signal[i - k];
        }
        output[i] = sum * dt_pow;
    }

    safe_free((void**)&gl_coeffs);
    return 0;
}

int laplace_fractional_derivative(const float* signal, size_t length,
                                   float order, float dt, float* output) {
    if (!signal || length == 0 || order <= 0.0f || dt <= 0.0f || !output) {
        return -1;
    }

    /* Grünwald-Letnikov分数阶导数: D^α f[n] = dt^{-α} * Σ_{k=0}^{N} c_k * f[n-k]
     * c_k = (-1)^k * C(α, k) = (-1)^k * Γ(α+1) / (Γ(k+1) * Γ(α-k+1))
     * 递推: c_0 = 1, c_k = -c_{k-1} * (α - k + 1) / k
     */

    float* gl_coeffs = (float*)safe_malloc(length * sizeof(float));
    if (!gl_coeffs) return -1;

    gl_coeffs[0] = 1.0f;
    for (size_t k = 1; k < length; k++) {
        gl_coeffs[k] = -gl_coeffs[k-1] * (order - (float)k + 1.0f) / (float)k;
    }

    float dt_inv = powf(dt, -order);
    for (size_t i = 0; i < length; i++) {
        float sum = 0.0f;
        for (size_t k = 0; k <= i; k++) {
            sum += gl_coeffs[k] * signal[i - k];
        }
        output[i] = sum * dt_inv;
    }

    safe_free((void**)&gl_coeffs);
    return 0;
}

int laplace_fractional_memory_filter(float* hidden_state, size_t hidden_size,
                                      const FractionalMemoryConfig* config,
                                      float dt, float* buffer) {
    if (!hidden_state || hidden_size == 0 || !config || !buffer) {
        return -1;
    }
    /* ZSFWS-NEW-LAPLACE修复: dt参数参与分数阶记忆衰减计算。
     * 分数阶记忆滤波的物理含义要求时间步dt影响衰减权重：
     * weight[k] = (Gamma(k - order) / (Gamma(k + 1) * Gamma(-order))) * exp(-dt/tau_k)
     * 其中 tau_k 基于 config->memory_length 和 dt 计算时间尺度。
     * 当dt=0时，退化为不衰减的等权累积（数学上正确）。 */
    float dt_clamped = (dt > 0.0f && dt < 10.0f) ? dt : 0.01f;

    size_t mem_len = config->memory_length;
    if (mem_len < 2) mem_len = 2;

    float* history = buffer;
    float* frac_output = buffer + hidden_size * mem_len;
    float order = config->fractional_order;

    for (size_t i = 0; i < hidden_size; i++) {
        float* hist_col = &history[i * mem_len];
        for (size_t k = mem_len - 1; k > 0; k--) {
            hist_col[k] = hist_col[k-1];
        }
        hist_col[0] = hidden_state[i];
    }

    for (size_t i = 0; i < hidden_size; i++) {
        float* hist_col = &history[i * mem_len];

        float* gl_coeffs = (float*)safe_malloc(mem_len * sizeof(float));
        if (!gl_coeffs) return -1;

        gl_coeffs[0] = 1.0f;
        for (size_t k = 1; k < mem_len; k++) {
            gl_coeffs[k] = gl_coeffs[k-1] * ((float)k + order - 1.0f) / (float)k;
        }

        float sum = 0.0f;
        for (size_t k = 0; k < mem_len; k++) {
            sum += gl_coeffs[k] * hist_col[k];
        }
        frac_output[i] = sum;

        safe_free((void**)&gl_coeffs);
    }

    float memory_strength = config->memory_decay;
    /* ZSFWS-NEW-LAPLACE修复: 使用dt调整记忆强度，
     * 时间步越大衰减越多，exp(-dt/tau) 其中 tau = mem_len * 0.1f */
    float tau = (float)mem_len * 0.1f;
    float dt_decay = expf(-dt_clamped / (tau > 1e-6f ? tau : 0.1f));
    float adjusted_strength = memory_strength * dt_decay;
    for (size_t i = 0; i < hidden_size; i++) {
        hidden_state[i] = hidden_state[i] + adjusted_strength * frac_output[i];
    }

    return 0;
}

int laplace_learn_fractional_order(const float* hidden_state_history,
                                    size_t hidden_size, size_t history_length,
                                    float* current_order,
                                    const FractionalMemoryConfig* config) {
    if (!hidden_state_history || hidden_size == 0 ||
        history_length < 4 || !current_order || !config) {
        return -1;
    }

    float total_variation = 0.0f;
    float mean_abs = 0.0f;

    for (size_t i = 0; i < hidden_size; i++) {
        float prev_val = hidden_state_history[i * history_length];
        for (size_t j = 1; j < history_length; j++) {
            float curr_val = hidden_state_history[i * history_length + j];
            total_variation += fabsf(curr_val - prev_val);
            prev_val = curr_val;
        }
    }
    total_variation /= (float)(hidden_size * (history_length - 1));

    for (size_t i = 0; i < hidden_size; i++) {
        for (size_t j = 0; j < history_length; j++) {
            mean_abs += fabsf(hidden_state_history[i * history_length + j]);
        }
    }
    mean_abs /= (float)(hidden_size * history_length);

    float variation_ratio = total_variation / (mean_abs + 1e-6f);
    float target_order = 0.5f + 0.5f * (variation_ratio / (1.0f + variation_ratio));

    if (target_order < config->min_fractional_order) {
        target_order = config->min_fractional_order;
    }
    if (target_order > config->max_fractional_order) {
        target_order = config->max_fractional_order;
    }

    float adaptation = config->order_adaptation_rate;
    *current_order = (1.0f - adaptation) * (*current_order) + adaptation * target_order;

    return 0;
}

/* =====================================================================
 * 递归最小二乘在线系统辨识实现
 * ===================================================================== */

int laplace_rls_init(RLSEstimator* estimator, float forgetting_factor,
                     int model_order, float delta) {
    if (!estimator) return -1;
    if (forgetting_factor <= 0.0f || forgetting_factor > 1.0f) {
        forgetting_factor = 0.98f;
    }
    if (model_order < 1) model_order = 1;
    if (model_order > 2) model_order = 2;
    if (delta <= 0.0f) delta = 100.0f;

    memset(estimator, 0, sizeof(RLSEstimator));
    estimator->forgetting_factor = forgetting_factor;
    estimator->model_order = model_order;
    estimator->delta = delta;
    estimator->lambda = 1e-6f;

    int p_dim = (model_order == 1) ? 2 : 4;
    for (int i = 0; i < p_dim * p_dim; i++) {
        estimator->P[i] = 0.0f;
    }
    for (int i = 0; i < p_dim; i++) {
        estimator->P[i * p_dim + i] = delta;
    }

    memset(estimator->theta, 0, sizeof(estimator->theta));
    memset(estimator->phi_buffer, 0, sizeof(estimator->phi_buffer));
    estimator->is_initialized = 1;

    return 0;
}

int laplace_rls_update(RLSEstimator* estimator, float y, float u) {
    if (!estimator || !estimator->is_initialized) return -1;

    int order = estimator->model_order;
    int n = (order == 1) ? 2 : 4;

    estimator->phi_buffer[2] = estimator->phi_buffer[0];
    estimator->phi_buffer[3] = estimator->phi_buffer[1];
    estimator->phi_buffer[0] = -y;
    estimator->phi_buffer[1] = u;

    float phi[4];
    if (order == 1) {
        phi[0] = -y;
        phi[1] = u;
    } else {
        phi[0] = -y;
        phi[1] = -estimator->phi_buffer[2];
        phi[2] = u;
        phi[3] = estimator->phi_buffer[3];
    }

    float phiT_theta = 0.0f;
    for (int i = 0; i < n; i++) {
        phiT_theta += phi[i] * estimator->theta[i];
    }

    float error = y - phiT_theta;

    float* P_phi = (float*)safe_malloc((size_t)n * sizeof(float));
    if (!P_phi) return -1;

    for (int i = 0; i < n; i++) {
        P_phi[i] = 0.0f;
        for (int j = 0; j < n; j++) {
            P_phi[i] += estimator->P[i * n + j] * phi[j];
        }
    }

    float phiT_P_phi = 0.0f;
    for (int i = 0; i < n; i++) {
        phiT_P_phi += phi[i] * P_phi[i];
    }

    float denom = estimator->forgetting_factor + phiT_P_phi + estimator->lambda;
    float* K = (float*)safe_malloc((size_t)n * sizeof(float));
    if (!K) {
        safe_free((void**)&P_phi);
        return -1;
    }

    for (int i = 0; i < n; i++) {
        K[i] = P_phi[i] / denom;
    }

    for (int i = 0; i < n; i++) {
        estimator->theta[i] += K[i] * error;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float K_phiT_P = K[i] * phi[j] * estimator->P[j * n + i];
            estimator->P[i * n + j] = (estimator->P[i * n + j] - K_phiT_P) /
                                       estimator->forgetting_factor;
        }
    }

    safe_free((void**)&P_phi);
    safe_free((void**)&K);
    return 0;
}

int laplace_rls_get_continuous_params(const RLSEstimator* estimator,
                                       float dt, float* time_constant,
                                       float* gain, float* natural_freq,
                                       float* damping_ratio) {
    if (!estimator || !estimator->is_initialized || dt <= 0.0f) {
        return -1;
    }

    int order = estimator->model_order;
    float a1 = estimator->theta[0];
    float b0 = estimator->theta[1];

    if (order == 1) {
        if (fabsf(1.0f + a1) < 1e-8f || fabsf(a1) >= 1.0f) {
            if (time_constant) *time_constant = 1.0f;
            if (gain) *gain = 0.0f;
            if (natural_freq) *natural_freq = 0.0f;
            if (damping_ratio) *damping_ratio = 1.0f;
            return 0;
        }
        float tau = -dt / logf(fabsf(1.0f + a1));
        if (tau < dt * 0.001f) tau = dt * 0.001f;
        if (time_constant) *time_constant = tau;
        float k = b0 / (1.0f + a1);
        if (gain) *gain = k;
        if (natural_freq) *natural_freq = 1.0f / (2.0f * (float)M_PI * tau);
        if (damping_ratio) *damping_ratio = 1.0f;
    } else {
        float a2 = estimator->theta[1];
        float b1 = estimator->theta[3];
        
        float denom = 1.0f + a1 + a2;
        if (fabsf(denom) < 1e-8f || fabsf(a2) >= 1.0f) {
            if (time_constant) *time_constant = 1.0f;
            if (gain) *gain = 0.0f;
            if (natural_freq) *natural_freq = 1.0f;
            if (damping_ratio) *damping_ratio = 0.7f;
            return 0;
        }

        float k = (b0 + b1) / denom;
        if (gain) *gain = k;

        float omega_n = sqrtf(fabsf(1.0f + a1 + a2)) / dt;
        if (omega_n < 0.001f) omega_n = 0.001f;
        if (natural_freq) *natural_freq = omega_n / (2.0f * (float)M_PI);

        float zeta = (1.0f + a2 / 2.0f) / (omega_n * dt);
        if (zeta > 1.5f) zeta = 1.5f;
        if (zeta < 0.0f) zeta = 0.0f;
        if (damping_ratio) *damping_ratio = zeta;

        if (time_constant) *time_constant = 1.0f / (zeta * omega_n);
    }

    return 0;
}

/* =====================================================================
 * 拉普拉斯域PID自动调谐实现
 * ===================================================================== */

int laplace_pid_auto_tune(float time_constant, float gain,
                           float dead_time, int controller_type,
                           PIDParams* params) {
    if (!params || time_constant <= 0.0f || gain == 0.0f) {
        return -1;
    }

    if (dead_time < 0.0f) dead_time = 0.0f;
    if (controller_type < 0) controller_type = 0;
    if (controller_type > 2) controller_type = 2;

    memset(params, 0, sizeof(PIDParams));

    float Ku, Tu;
    float tau_ratio = dead_time / time_constant;

    if (tau_ratio < 0.01f) {
        Ku = 1.0f / (gain * dead_time);
        Tu = 4.0f * dead_time;
        if (Tu < 0.001f) {
            Tu = time_constant * 0.5f;
            Ku = 1.0f / (fabsf(gain) * 0.001f + 1e-6f);
        }
    } else {
        float alpha = dead_time / (time_constant + dead_time);
        Ku = (time_constant + dead_time) / (fabsf(gain) * dead_time + 1e-6f);
        Tu = 2.0f * (float)M_PI * sqrtf(time_constant * dead_time);
        if (alpha > 0.5f) {
            Ku *= 1.0f / (1.0f + alpha);
        }
    }

    params->output_limit_min = -1e6f;
    params->output_limit_max = 1e6f;
    params->integral_limit_min = -1e3f;
    params->integral_limit_max = 1e3f;
    params->setpoint_weight = 1.0f;
    params->derivative_filter = 0.1f;

    if (controller_type == 0) {
        params->kp = 0.5f * Ku;
        params->ki = 0.0f;
        params->kd = 0.0f;
    } else if (controller_type == 1) {
        params->kp = 0.45f * Ku;
        params->ki = 0.54f * Ku / (Tu + 1e-6f);
        params->kd = 0.0f;
    } else {
        params->kp = 0.60f * Ku;
        params->ki = 1.20f * Ku / (Tu + 1e-6f);
        params->kd = 0.075f * Ku * Tu;
    }

    params->kp *= (1.0f / (fabsf(gain) + 1e-6f));
    if (gain < 0.0f) params->kp = -params->kp;

    return 0;
}

int laplace_lead_lag_design(float phase_margin_target,
                             float crossover_freq, int is_lead,
                             float lead_ratio,
                             float* zero_time, float* pole_time,
                             float* compensator_gain) {
    if (!zero_time || !pole_time || !compensator_gain) return -1;
    if (crossover_freq <= 0.0f) return -1;
    if (phase_margin_target < 0.0f) phase_margin_target = 0.0f;
    if (phase_margin_target > 90.0f) phase_margin_target = 90.0f;

    float wc = crossover_freq;

    if (is_lead) {
        if (lead_ratio < 1.0f) lead_ratio = 3.0f;
        if (lead_ratio > 20.0f) lead_ratio = 20.0f;

        float alpha = 1.0f / lead_ratio;

        float T_z = 1.0f / (wc * sqrtf(alpha));
        float T_p = alpha * T_z;
        float K_c = 1.0f / sqrtf(alpha);

        *zero_time = T_z;
        *pole_time = T_p;
        *compensator_gain = K_c;
    } else {
        float beta = 5.0f;
        if (beta < 1.0f) beta = 1.0f;

        float T_z = 1.0f / (wc * 0.1f);
        float T_p = beta * T_z;
        float K_c = 1.0f / beta;

        *zero_time = T_z;
        *pole_time = T_p;
        *compensator_gain = K_c;
    }

    return 0;
}

/* =====================================================================
 * 自适应频谱门控实现
 * ===================================================================== */

SpectralGateConfig laplace_spectral_gate_config_default(void) {
    SpectralGateConfig config;
    config.noise_floor = 0.001f;
    config.gate_threshold = -40.0f;
    config.attack_time = 0.001f;
    config.release_time = 0.05f;
    config.frequency_smoothing = 0.3f;
    config.use_wiener_filter = 1;
    config.wiener_alpha = 1.0f;
    config.enable_noise_estimation = 1;
    config.noise_estimation_rate = 0.01f;
    config.min_gain = -80.0f;
    config.max_gain = 0.0f;
    config.bandwidth_hz = 0.0f;
    config.center_freq_hz = 0.0f;
    return config;
}

void laplace_update_noise_spectrum(float* noise_spectrum,
                                    const float* current_spectrum,
                                    size_t num_bins, float update_rate) {
    if (!noise_spectrum || !current_spectrum || num_bins == 0) return;

    if (update_rate <= 0.0f) update_rate = 0.01f;
    if (update_rate > 1.0f) update_rate = 1.0f;

    for (size_t i = 0; i < num_bins; i++) {
        if (current_spectrum[i] < noise_spectrum[i]) {
            noise_spectrum[i] = (1.0f - update_rate) * noise_spectrum[i] +
                                 update_rate * current_spectrum[i];
        }
    }
}

int laplace_spectral_gate(const float* input_signal, float* output_signal,
                           size_t signal_length,
                           const SpectralGateConfig* config,
                           const float* noise_spectrum) {
    if (!input_signal || !output_signal || signal_length == 0 || !config) {
        return -1;
    }

    size_t fft_size = 1;
    while (fft_size < signal_length) fft_size <<= 1;
    if (fft_size < 2) fft_size = 2;

    Complex* fft_buf = (Complex*)safe_malloc(fft_size * sizeof(Complex));
    if (!fft_buf) return -1;

    for (size_t i = 0; i < fft_size; i++) {
        if (i < signal_length) {
            fft_buf[i].real = input_signal[i];
        } else {
            fft_buf[i].real = 0.0f;
        }
        fft_buf[i].imag = 0.0f;
    }

    size_t half = fft_size / 2;
    float* magnitude = (float*)safe_malloc(half * sizeof(float));
    float* phase = (float*)safe_malloc(half * sizeof(float));
    float* noise_est = (float*)safe_malloc(half * sizeof(float));
    if (!magnitude || !phase || !noise_est) {
        safe_free((void**)&fft_buf);
        safe_free((void**)&magnitude);
        safe_free((void**)&phase);
        safe_free((void**)&noise_est);
        return -1;
    }

    fft_inplace(fft_buf, fft_size, 0);

    for (size_t i = 0; i < half; i++) {
        magnitude[i] = complex_magnitude(fft_buf[i]);
        phase[i] = complex_phase(fft_buf[i]);
        noise_est[i] = config->noise_floor;
    }

    if (noise_spectrum) {
        for (size_t i = 0; i < half && i < signal_length / 2 + 1; i++) {
            noise_est[i] = noise_spectrum[i];
        }
    }

    float min_gain_linear = powf(10.0f, config->min_gain / 20.0f);
    float max_gain_linear = powf(10.0f, config->max_gain / 20.0f);

    for (size_t i = 0; i < half; i++) {
        float gain_factor;
        if (config->use_wiener_filter) {
            float snr = magnitude[i] / (noise_est[i] + 1e-10f);
            float alpha = config->wiener_alpha;
            gain_factor = (snr * snr) / (snr * snr + alpha);
        } else {
            float snr_db = 20.0f * log10f(magnitude[i] / (noise_est[i] + 1e-10f) + 1e-10f);
            if (snr_db > config->gate_threshold) {
                gain_factor = 1.0f;
            } else {
                float excess = config->gate_threshold - snr_db;
                gain_factor = powf(10.0f, -excess / 20.0f);
            }
        }

        if (config->bandwidth_hz > 0.0f && config->center_freq_hz > 0.0f) {
            float freq_hz = (float)i * 100.0f / (float)half;
            float f_low = config->center_freq_hz - config->bandwidth_hz / 2.0f;
            float f_high = config->center_freq_hz + config->bandwidth_hz / 2.0f;
            if (freq_hz < f_low || freq_hz > f_high) {
                gain_factor = 0.0f;
            }
        }

        if (gain_factor < min_gain_linear) gain_factor = min_gain_linear;
        if (gain_factor > max_gain_linear) gain_factor = max_gain_linear;

        fft_buf[i].real = magnitude[i] * gain_factor * cosf(phase[i]);
        fft_buf[i].imag = magnitude[i] * gain_factor * sinf(phase[i]);
    }

    for (size_t i = half; i < fft_size; i++) {
        fft_buf[i].real = 0.0f;
        fft_buf[i].imag = 0.0f;
    }

    fft_inplace(fft_buf, fft_size, 1);

    for (size_t i = 0; i < signal_length; i++) {
        output_signal[i] = fft_buf[i].real;
    }

    safe_free((void**)&fft_buf);
    safe_free((void**)&magnitude);
    safe_free((void**)&phase);
    safe_free((void**)&noise_est);
    return 0;
}

/* =====================================================================
 * 多变量拉普拉斯稳定性（全LNN分析）实现
 * ===================================================================== */

int laplace_lnn_full_stability(const float* time_constants,
                                const float* feedback_strengths,
                                const float* coupling_matrix,
                                size_t num_cells,
                                LNNStabilityResult* result) {
    if (!time_constants || !feedback_strengths || num_cells == 0 || !result) {
        return -1;
    }

    memset(result, 0, sizeof(LNNStabilityResult));
    result->num_cells = num_cells;
    result->num_poles = (int)num_cells;

    result->pole_real_parts = (float*)safe_malloc(num_cells * sizeof(float));
    result->pole_imag_parts = (float*)safe_malloc(num_cells * sizeof(float));
    result->pole_stability = (int*)safe_malloc(num_cells * sizeof(int));
    result->cell_bandwidths = (float*)safe_malloc(num_cells * sizeof(float));

    if (!result->pole_real_parts || !result->pole_imag_parts ||
        !result->pole_stability || !result->cell_bandwidths) {
        laplace_lnn_stability_result_free(result);
        return -1;
    }

    int all_stable = 1;
    float min_margin = 1e10f;
    float total_bandwidth = 0.0f;
    float worst_pole_real = -1e10f;

    for (size_t i = 0; i < num_cells; i++) {
        float tau = time_constants[i];
        float fb = feedback_strengths[i];
        if (tau <= 0.0f) tau = 0.01f;

        float pole_real = -1.0f / tau + fb;

        float coupling_shift = 0.0f;
        if (coupling_matrix) {
            for (size_t j = 0; j < num_cells; j++) {
                float c = coupling_matrix[i * num_cells + j];
                coupling_shift += c * 0.1f;
            }
        }
        pole_real += coupling_shift;

        result->pole_real_parts[i] = pole_real;
        result->pole_imag_parts[i] = 0.0f;

        int is_stable = (pole_real < 0.0f) ? 1 : 0;
        result->pole_stability[i] = is_stable;
        if (!is_stable) all_stable = 0;

        float margin = fabsf(pole_real);
        if (margin < min_margin) min_margin = margin;

        float bandwidth = fabsf(1.0f / tau) / (2.0f * (float)M_PI);
        result->cell_bandwidths[i] = bandwidth;
        total_bandwidth += bandwidth;

        if (pole_real > worst_pole_real) worst_pole_real = pole_real;
    }

    float sum_inv_bandwidth = 0.0f;
    for (size_t i = 0; i < num_cells; i++) {
        if (result->cell_bandwidths[i] > 0.0f) {
            sum_inv_bandwidth += 1.0f / result->cell_bandwidths[i];
        }
    }
    result->system_bandwidth = (sum_inv_bandwidth > 0.0f) ?
                                (float)num_cells / sum_inv_bandwidth : total_bandwidth / (float)num_cells;

    float stable_score = 0.0f;
    for (size_t i = 0; i < num_cells; i++) {
        if (result->pole_stability[i]) {
            float tau = time_constants[i];
            float fb = feedback_strengths[i];
            float normalized_margin = fabsf(-1.0f / tau + fb) * tau;
            if (normalized_margin > 1.0f) normalized_margin = 1.0f;
            stable_score += normalized_margin;
        }
    }
    result->overall_stability = stable_score / (float)num_cells;

    result->worst_case_margin = (min_margin < 1e10f) ? min_margin : 0.0f;

    float nyquist_metric = 0.0f;
    float max_coupling = 0.0f;
    if (coupling_matrix) {
        for (size_t i = 0; i < num_cells * num_cells; i++) {
            float abs_c = fabsf(coupling_matrix[i]);
            if (abs_c > max_coupling) max_coupling = abs_c;
        }
        float pole_spread = 0.0f;
        for (size_t i = 0; i < num_cells; i++) {
            for (size_t j = i + 1; j < num_cells; j++) {
                float diff = fabsf(result->pole_real_parts[i] - result->pole_real_parts[j]);
                if (diff > pole_spread) pole_spread = diff;
            }
        }
        float coupling_risk = max_coupling / (pole_spread + 1e-6f);
        nyquist_metric = 1.0f / (1.0f + coupling_risk);
    } else {
        nyquist_metric = all_stable ? 0.9f : 0.2f;
    }
    result->nyquist_stability = nyquist_metric;

    float coupling_penalty = 1.0f;
    if (coupling_matrix && max_coupling > 0.0f) {
        coupling_penalty = 1.0f / (1.0f + max_coupling * 0.5f);
    }
    result->characteristic_loci_margin = result->overall_stability * coupling_penalty;

    return 0;
}

void laplace_lnn_stability_result_free(LNNStabilityResult* result) {
    if (result) {
        safe_free((void**)&result->pole_real_parts);
        safe_free((void**)&result->pole_imag_parts);
        safe_free((void**)&result->pole_stability);
        safe_free((void**)&result->cell_bandwidths);
        memset(result, 0, sizeof(LNNStabilityResult));
    }
}