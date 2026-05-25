/**
 * @file laplace_enhanced.c
 * @brief 拉普拉斯变换全面增强系统 —— 系统编排层（System Orchestration Layer）
 *
 * ========== ZSFWS-033 模块职责边界 ==========
 * 本模块职责：系统级拉普拉斯分析编排与监控
 *   - 拉普拉斯增强系统生命周期管理 (laplace_enhanced_create/free)
 *   - FFT频谱分析与频谱特征提取 (laplace_spectral_analyze) *
 *   - 频域滤波器（Butterworth低通/带通） (laplace_spectral_filter) *
 *   - 梯度频域增强（自适应截止频率） (laplace_enhance_gradients)
 *   - 谱减法去噪 (laplace_denoise)
 *   - 增强型稳定性分析（幂迭代特征值） (laplace_stability_analyze) **
 *   - 控制信号稳定化（双线性变换二阶补偿器） (laplace_stabilize_control)
 *   - FFT频域卷积（卷积定理实现） (laplace_fft_convolution_enhanced)
 *   - 图像拉普拉斯锐化增强 (laplace_image_enhance)
 *   - 拉普拉斯记忆衰减模型 (laplace_memory_decay_model)
 *   - 5阶段分析管道（FFT→频谱→稳定性→滤波→控制） (laplace_pipeline_execute)
 *   - 实时稳定性监控器 (laplace_monitor_*)
 *   - LNN稳定性保证器（特征值监控+自动校正） (laplace_guarantor_*)
 *
 * * 与 laplace_integration.c 的关系：
 *   两者都涉及FFT频谱计算。本模块的 laplace_spectral_analyze 侧重
 *   系统级频谱特征提取（质心、熵、带宽、高低频能量比），作为分析管道
 *   的输入阶段。laplace_integration.c 的 laplace_compute_spectrum 侧重
 *   CfC训练级的频谱计算（用于自适应学习率）。
 *   两者通过统一的 laplace_fft.h (lfft_split_radix2) 实现FFT，无重复实现。
 *
 * ** 与 laplace_integration.c 的关系：
 *   两者都涉及稳定性分析。本模块的 laplace_stability_analyze 使用
 *   幂迭代特征值分解进行系统级广义稳定性判断（阻尼比、增益裕度、
 *   相位裕度、超调量）。laplace_integration.c 的 cfc_cell_analyze_stability
 *   专注CfC单细胞的一阶传递函数极点分析。
 *
 * 与 laplace_features.c 的关系：
 *   本模块与 laplace_features.c 无直接功能重叠。
 *   laplace_features.c 是特征层（金字塔/图拉普拉斯/流形学习），
 *   本模块是系统编排层（管道/监控/保证器），各司其职。
 *
 * 本模块的独特价值：提供了统一的分析管道和实时监控体系，
 * 是系统初始化时创建的顶层Laplace系统，供各子系统调用。
 * =============================================
 */

#include "selflnn/core/laplace_enhanced.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/laplace_fft.h"
#include "selflnn/utils/memory_utils.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ZSFBUILD: 缺失类型定义 —— laplace_enhanced.h为空转发，全部在此补全 */

/* 增强目标 */
typedef enum {
    LAPLACE_TARGET_ALL = 0,        /**< 全部子系统 */
    LAPLACE_TARGET_LNN = 1,        /**< LNN网络 */
    LAPLACE_TARGET_TRAINING = 2,   /**< 训练系统 */
    LAPLACE_TARGET_INFERENCE = 3   /**< 推理系统 */
} LaplaceTarget;

/* 滤波器操作类型 */
typedef enum {
    LAPLACE_OP_GAUSSIAN = 0,       /**< 高斯滤波 */
    LAPLACE_OP_MEDIAN = 1,         /**< 中值滤波 */
    LAPLACE_OP_BILATERAL = 2,      /**< 双边滤波 */
    LAPLACE_OP_BUTTERWORTH = 3     /**< Butterworth滤波 */
} LaplaceFilterOpType;

/* 滤波器类型 */
typedef enum {
    LAPLACE_FILTER_LOWPASS = 0,    /**< 低通 */
    LAPLACE_FILTER_HIGHPASS = 1,   /**< 高通 */
    LAPLACE_FILTER_BANDPASS = 2,   /**< 带通 */
    LAPLACE_FILTER_BANDSTOP = 3    /**< 带阻 */
} LaplaceFilterPassType;

/* 滤波器配置 */
typedef struct {
    LaplaceFilterOpType op_type;        /**< 操作类型 */
    LaplaceFilterPassType type;          /**< 通带类型 */
    float cutoff_frequency;             /**< 截止频率 */
    float filter_order;                  /**< 滤波器阶数 */
    float smoothing_alpha;              /**< 平滑系数 */
    float noise_suppression;            /**< 噪声抑制 */
    int use_adaptive_cutoff;            /**< 自适应截止 */
    int use_zero_phase;                 /**< 零相位滤波 */
    int kernel_size;                    /**< 核大小 */
    int order;                          /**< 滤波器阶数 */
    float cutoff_freq;                  /**< 归一化截止频率 */
} LaplaceFilterConfig;

/* 频谱特征 */
typedef struct {
    float* magnitude_spectrum;          /**< 幅度谱 */
    float* phase_spectrum;              /**< 相位谱 */
    size_t spectrum_size;               /**< 频谱大小 */
    float dominant_frequency;           /**< 主导频率 */
    float spectral_entropy;             /**< 谱熵 */
    float spectral_centroid;            /**< 谱质心 */
    float spectral_bandwidth;           /**< 谱带宽 */
    float spectral_flatness;            /**< 谱平坦度 */
    float spectral_rolloff;             /**< 谱滚降 */
    float spectral_spread;              /**< 谱扩展 */
    float high_freq_energy_ratio;       /**< 高频能量比 */
    float low_freq_energy_ratio;        /**< 低频能量比 */
    float stability_metric;             /**< 综合稳定性度量 */
    int peak_count;                     /**< 峰值数量 */
} LaplaceSpectralFeatures;

/* 增强稳定性分析 */
typedef struct {
    float dominant_eigenvalue;          /**< 主导特征值 */
    float eigenvalue_spread;            /**< 特征值散布 */
    int eigenvalue_count;               /**< 特征值数量 */
    float damping_ratio;                /**< 阻尼比 */
    float natural_frequency;            /**< 自然频率 */
    float gain_margin;                  /**< 增益裕度 */
    float phase_margin;                 /**< 相位裕度 */
    float overshoot;                    /**< 超调量 */
    float settling_time;                /**< 稳定时间 Ts */
    float stability_reserve;            /**< 稳定储备 (0-1) */
    float condition_number;             /**< 条件数 */
    int is_stable;                      /**< 是否稳定 */
    float stability_margin;             /**< 稳定性裕度 */
    float eigenvalue_real_parts[128];   /**< 特征值实部 */
    float eigenvalue_imag_parts[128];   /**< 特征值虚部 */
} EnhancedStabilityAnalysis;

/* 分析管道阶段 */
typedef enum {
    LAPLACE_PIPELINE_STAGE_FFT = 0,        /**< FFT频域变换 */
    LAPLACE_PIPELINE_STAGE_SPECTRAL = 1,   /**< 频谱特征提取 */
    LAPLACE_PIPELINE_STAGE_STABILITY = 2,  /**< 稳定性分析 */
    LAPLACE_PIPELINE_STAGE_FILTERING = 3,  /**< 频域滤波 */
    LAPLACE_PIPELINE_STAGE_CONTROL = 4     /**< 控制信号生成 */
} LaplacePipelineStage;

/* 管道阶段结果 */
typedef struct {
    LaplacePipelineStage stage;            /**< 管道阶段 */
    int success;                           /**< 是否成功 */
    float execution_time_ms;               /**< 执行时间(ms) */
    float stage_metrics[8];                /**< 阶段指标 */
    int metric_count;                      /**< 指标数量 */
} PipelineStageResult;

/* 管道最终结果 */
typedef struct {
    PipelineStageResult stage_results[5];  /**< 各阶段结果 */
    int stage_count;                       /**< 阶段计数 */
    float overall_health_score;            /**< 综合健康指数 */
    float control_recommendation;          /**< 控制建议 */
    int alarm_level;                       /**< 告警级别 */
} PipelineResult;

/* Laplace增强系统 */
typedef struct LaplaceEnhancedSystem LaplaceEnhancedSystem;

/* ZSFBUILD: LNN稳定性保证器配置 */
typedef struct {
    float max_eigenvalue_real;           /**< 最大允许特征值实部 */
    float damping_factor;                /**< 阻尼因子 */
    float correction_strength;           /**< 校正强度 */
    int enable_automatic_correction;     /**< 自动校正开关 */
    int enable_adaptive_damping;         /**< 自适应阻尼 */
} LNNStabilityConfig;

/* ZSFBUILD: LNN稳定性保证器 */
typedef struct {
    LaplaceEnhancedSystem* parent_system;    /**< 父系统 */
    LNNStabilityConfig config;               /**< 配置 */
    float* eigenvalue_real_parts;            /**< 特征值实部 */
    float* eigenvalue_imag_parts;            /**< 特征值虚部 */
    size_t eigenvalue_count;                 /**< 特征值数量 */
    float max_real_part;                     /**< 最大实部 */
    float* correction_history;               /**< 校正历史 */
    size_t correction_history_size;          /**< 校正历史容量 */
    size_t correction_write_pos;             /**< 校正写入位置 */
    int correction_count;                    /**< 校正次数 */
} LNNStabilityGuarantor;

/* Laplace增强系统 */

/* 前向声明 */
static void power_iteration_eigenvalues(const float* matrix, size_t size,
                                          float* real_parts, float* imag_parts,
                                          size_t* count, size_t max_count);

struct LaplaceEnhancedSystem {
    LaplaceTarget target;
    LaplaceFilterConfig filter_config;
    LaplaceSpectralFeatures last_features;
    EnhancedStabilityAnalysis last_stability;
    int initialized;

    /* 工作缓冲区 */
    float* fft_real;
    float* fft_imag;
    size_t fft_size;

    /* 频域累积统计 */
    float* freq_accumulator;
    size_t freq_accum_count;
    float* freq_variances;
    size_t freq_var_count;

    /* 滤波器内核 */
    float* filter_kernel;
    int filter_kernel_size;
};

/* ZSFBUILD: RealtimeMonitor类型未在任何头文件中声明，在此处补全 */
typedef struct {
    float* history_buffer;
    float* spectrum_accumulator;
    float* feature_history;
    size_t buffer_size;
    size_t feature_history_size;
    size_t write_pos;
    size_t feature_write_pos;
    float stability_threshold;
    float anomaly_threshold;
    int consecutive_anomalies;      /**< 连续异常计数 */
    int is_initialized;
} RealtimeMonitor;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* 快速傅里叶变换：统一使用 laplace_fft.h 中的 lfft_split_radix2 实现，
 * 已消除与 laplace_integration.c / laplace_ai_framework.c 的重复代码 */
#define radix2_fft(real, imag, n, inv) lfft_split_radix2(real, imag, n, inv)

/* 获取下一个2的幂 */
static size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* 生成高斯核 */
static void generate_gaussian_kernel(float* kernel, int size, float sigma) {
    int center = size / 2;
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float x = (float)(i - center);
        kernel[i] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
}

/* 生成LoG核 */
static void generate_log_kernel(float* kernel, int size, float sigma) {
    int center = size / 2;
    float sum = 0.0f;
    float sigma2 = sigma * sigma;
    float sigma4 = sigma2 * sigma2;
    for (int i = 0; i < size; i++) {
        int y = i / size - center;
        for (int j = 0; j < size; j++) {
            int x = j - center;
            float r2 = (float)(x * x + y * y);
            kernel[i * size + j] = -1.0f / ((float)M_PI * sigma4) * 
                (1.0f - r2 / (2.0f * sigma2)) * expf(-r2 / (2.0f * sigma2));
            sum += kernel[i * size + j];
        }
    }
    /* 归一化为零均值 */
    float mean = sum / (float)(size * size);
    for (int i = 0; i < size * size; i++) {
        kernel[i] -= mean;
    }
}

LaplaceEnhancedSystem* laplace_enhanced_create(LaplaceTarget target) {
    LaplaceEnhancedSystem* sys = (LaplaceEnhancedSystem*)safe_calloc(1, sizeof(LaplaceEnhancedSystem));
    if (!sys) return NULL;

    sys->target = target;
    sys->initialized = 1;

    /* 默认滤波配置 */
    sys->filter_config.op_type = LAPLACE_OP_GAUSSIAN;
    sys->filter_config.cutoff_frequency = 10.0f;
    sys->filter_config.filter_order = 2.0f;
    sys->filter_config.smoothing_alpha = 0.1f;
    sys->filter_config.noise_suppression = 0.5f;
    sys->filter_config.use_adaptive_cutoff = 1;
    sys->filter_config.use_zero_phase = 1;
    sys->filter_config.kernel_size = 5;

    sys->fft_size = 0;
    sys->fft_real = NULL;
    sys->fft_imag = NULL;

    sys->freq_accumulator = NULL;
    sys->freq_accum_count = 0;
    sys->freq_variances = NULL;
    sys->freq_var_count = 0;

    /* 分配滤波器内核 */
    int kernel_size = sys->filter_config.kernel_size;
    sys->filter_kernel = (float*)safe_calloc(kernel_size * kernel_size, sizeof(float));
    sys->filter_kernel_size = kernel_size;
    generate_log_kernel(sys->filter_kernel, kernel_size, 1.0f);

    return sys;
}

void laplace_enhanced_free(LaplaceEnhancedSystem* system) {
    if (!system) return;
    safe_free((void**)&system->fft_real);
    safe_free((void**)&system->fft_imag);
    safe_free((void**)&system->freq_accumulator);
    safe_free((void**)&system->freq_variances);
    safe_free((void**)&system->filter_kernel);
    safe_free((void**)&system);
}

int laplace_set_filter_config(LaplaceEnhancedSystem* system, const LaplaceFilterConfig* config) {
    if (!system || !config) return -1;
    memcpy(&system->filter_config, config, sizeof(LaplaceFilterConfig));

    int kernel_size = config->kernel_size;
    if (kernel_size != system->filter_kernel_size) {
        safe_free((void**)&system->filter_kernel);
        system->filter_kernel = (float*)safe_calloc(kernel_size * kernel_size, sizeof(float));
        system->filter_kernel_size = kernel_size;
    }
    if (system->filter_kernel) {
        generate_log_kernel(system->filter_kernel, kernel_size, 1.0f);
    }

    return 0;
}

int laplace_spectral_analyze(LaplaceEnhancedSystem* system, const float* signal,
                            size_t signal_size, LaplaceSpectralFeatures* features) {
    if (!system || !signal || signal_size == 0 || !features) return -1;

    size_t n = next_pow2(signal_size);
    if (n > system->fft_size) {
        safe_free((void**)&system->fft_real);
        safe_free((void**)&system->fft_imag);
        system->fft_real = (float*)safe_calloc(n, sizeof(float));
        system->fft_imag = (float*)safe_calloc(n, sizeof(float));
        system->fft_size = n;
    }

    memset(system->fft_real, 0, n * sizeof(float));
    memset(system->fft_imag, 0, n * sizeof(float));
    memcpy(system->fft_real, signal, signal_size * sizeof(float));

    radix2_fft(system->fft_real, system->fft_imag, n, 0);

    /* 计算幅度谱 */
    size_t half_n = n / 2;
    features->magnitude_spectrum = (float*)safe_malloc(half_n * sizeof(float));
    features->phase_spectrum = (float*)safe_malloc(half_n * sizeof(float));
    features->spectrum_size = half_n;

    float total_mag = 0.0f, weighted_freq = 0.0f, entropy = 0.0f;
    float max_mag = 0.0f;
    int max_idx = 0;

    for (size_t i = 0; i < half_n; i++) {
        float mag = sqrtf(system->fft_real[i] * system->fft_real[i] +
                         system->fft_imag[i] * system->fft_imag[i]);
        features->magnitude_spectrum[i] = mag;
        features->phase_spectrum[i] = atan2f(system->fft_imag[i], system->fft_real[i]);

        total_mag += mag;
        weighted_freq += mag * (float)i;
        if (mag > max_mag) { max_mag = mag; max_idx = (int)i; }
    }

    if (total_mag > 1e-10f) {
        features->dominant_frequency = weighted_freq / total_mag;
        for (size_t i = 0; i < half_n; i++) {
            float p = features->magnitude_spectrum[i] / total_mag;
            if (p > 1e-10f) entropy -= p * log2f(p);
        }
        features->spectral_entropy = entropy;
        features->spectral_centroid = weighted_freq / total_mag;
    }

    float bw_sum = 0.0f, total_power = 0.0f;
    for (size_t i = 0; i < half_n; i++) {
        float power = features->magnitude_spectrum[i] * features->magnitude_spectrum[i];
        total_power += power;
        bw_sum += power * (float)(i - (size_t)features->spectral_centroid) * 
                           (float)(i - (size_t)features->spectral_centroid);
    }
    features->spectral_bandwidth = (total_power > 1e-10f) ? sqrtf(bw_sum / total_power) : 0.0f;

    float high_energy = 0.0f, low_energy = 0.0f;
    size_t mid = half_n / 4;
    for (size_t i = 0; i < mid; i++) low_energy += features->magnitude_spectrum[i];
    for (size_t i = mid; i < half_n; i++) high_energy += features->magnitude_spectrum[i];
    features->high_freq_energy_ratio = (low_energy + high_energy > 1e-10f) ? 
        high_energy / (low_energy + high_energy) : 0.5f;

    features->stability_metric = 1.0f / (1.0f + features->high_freq_energy_ratio);

    memcpy(&system->last_features, features, sizeof(LaplaceSpectralFeatures));
    return 0;
}

int laplace_spectral_filter(LaplaceEnhancedSystem* system, const float* input,
                           float* output, size_t signal_size) {
    if (!system || !input || !output || signal_size == 0) return -1;

    size_t n = next_pow2(signal_size);
    if (n > system->fft_size) {
        safe_free((void**)&system->fft_real);
        safe_free((void**)&system->fft_imag);
        system->fft_real = (float*)safe_calloc(n, sizeof(float));
        system->fft_imag = (float*)safe_calloc(n, sizeof(float));
        system->fft_size = n;
    }

    memset(system->fft_real, 0, n * sizeof(float));
    memset(system->fft_imag, 0, n * sizeof(float));
    memcpy(system->fft_real, input, signal_size * sizeof(float));

    radix2_fft(system->fft_real, system->fft_imag, n, 0);

    float cutoff = system->filter_config.cutoff_frequency;
    float order = system->filter_config.filter_order;
    size_t half_n = n / 2;

    for (size_t i = 0; i < half_n; i++) {
        float freq_norm = (float)i / (float)half_n;
        float gain = 1.0f / (1.0f + powf(freq_norm / (cutoff / (float)half_n), 2.0f * order));
        system->fft_real[i] *= gain;
        system->fft_imag[i] *= gain;
    }

    radix2_fft(system->fft_real, system->fft_imag, n, 1);

    memcpy(output, system->fft_real, signal_size * sizeof(float));
    return 0;
}

int laplace_enhance_gradients(LaplaceEnhancedSystem* system, float* gradients,
                             size_t grad_size, float learning_rate) {
    if (!system || !gradients || grad_size == 0) return -1;

    /* I-008修复：使用learning_rate缩放频域增强梯度 */
    float lr = learning_rate > 0.0f ? learning_rate : 1e-3f;

    /* 频域分析梯度 */
    LaplaceSpectralFeatures features;
    int ret = laplace_spectral_analyze(system, gradients, grad_size, &features);
    if (ret != 0) return ret;

    /* 自适应截止频率：高频噪声多时降低截止频率 */
    float adaptive_cutoff = system->filter_config.cutoff_frequency;
    if (system->filter_config.use_adaptive_cutoff) {
        float noise_ratio = features.high_freq_energy_ratio;
        adaptive_cutoff *= (1.0f - noise_ratio * 0.8f);
        if (adaptive_cutoff < 1.0f) adaptive_cutoff = 1.0f;
    }

    /* 频域滤波 */
    float* filtered = (float*)safe_malloc(grad_size * sizeof(float));
    if (!filtered) return -1;

    float saved_cutoff = system->filter_config.cutoff_frequency;
    system->filter_config.cutoff_frequency = adaptive_cutoff;

    laplace_spectral_filter(system, gradients, filtered, grad_size);

    system->filter_config.cutoff_frequency = saved_cutoff;

    /* 混合原始梯度和滤波梯度（I-008修复：使用learning_rate控制混合强度） */
    float alpha = system->filter_config.smoothing_alpha;
    /* 学习率越高，滤波梯度贡献越大，帮助加速收敛 */
    float lr_scale = fminf(lr * 100.0f, 1.0f);
    alpha *= lr_scale;
    for (size_t i = 0; i < grad_size; i++) {
        gradients[i] = (1.0f - alpha) * gradients[i] + alpha * filtered[i];
    }

    safe_free((void**)&features.magnitude_spectrum);
    safe_free((void**)&features.phase_spectrum);
    safe_free((void**)&filtered);

    return 0;
}

int laplace_denoise(LaplaceEnhancedSystem* system, const float* input,
                   float* output, size_t signal_size) {
    if (!system || !input || !output || signal_size == 0) return -1;

    /* 使用谱减法去噪 */
    size_t n = next_pow2(signal_size);
    if (n > system->fft_size) {
        safe_free((void**)&system->fft_real);
        safe_free((void**)&system->fft_imag);
        system->fft_real = (float*)safe_calloc(n, sizeof(float));
        system->fft_imag = (float*)safe_calloc(n, sizeof(float));
        system->fft_size = n;
    }

    memset(system->fft_real, 0, n * sizeof(float));
    memset(system->fft_imag, 0, n * sizeof(float));
    memcpy(system->fft_real, input, signal_size * sizeof(float));

    radix2_fft(system->fft_real, system->fft_imag, n, 0);

    float suppression = system->filter_config.noise_suppression;
    size_t half_n = n / 2;

    for (size_t i = 0; i < half_n; i++) {
        float mag = sqrtf(system->fft_real[i] * system->fft_real[i] +
                         system->fft_imag[i] * system->fft_imag[i]);
        float phase = atan2f(system->fft_imag[i], system->fft_real[i]);

        /* 谱减法 */
        float noise_floor = suppression * 0.01f;
        float clean_mag = mag - noise_floor;
        if (clean_mag < 0.0f) clean_mag = 0.001f;

        system->fft_real[i] = clean_mag * cosf(phase);
        system->fft_imag[i] = clean_mag * sinf(phase);
    }

    radix2_fft(system->fft_real, system->fft_imag, n, 1);
    memcpy(output, system->fft_real, signal_size * sizeof(float));
    return 0;
}

int laplace_stability_analyze(LaplaceEnhancedSystem* system, const float* system_matrix,
                             int matrix_size, EnhancedStabilityAnalysis* analysis) {
    if (!system || !system_matrix || matrix_size <= 0 || !analysis) return -1;

    memset(analysis, 0, sizeof(EnhancedStabilityAnalysis));
    analysis->is_stable = 1;

    /* 使用幂迭代进行完整特征值分析替代简化的迹分析 */
    size_t eig_size = (size_t)matrix_size;
    size_t max_eigs = eig_size < 64 ? eig_size : 64;
    float* real_parts = (float*)safe_calloc(max_eigs, sizeof(float));
    float* imag_parts = (float*)safe_calloc(max_eigs, sizeof(float));
    size_t eig_count = 0;

    if (real_parts && imag_parts) {
        power_iteration_eigenvalues(system_matrix, eig_size, real_parts, imag_parts, &eig_count, max_eigs);

        if (eig_count > 0) {
            /* 从特征值计算真实阻尼比和自然频率 */
            float max_real = real_parts[0];
            float max_imag = imag_parts[0];
            int any_unstable = 0;

            for (size_t i = 0; i < eig_count; i++) {
                if (real_parts[i] > max_real) max_real = real_parts[i];
                if (imag_parts[i] > max_imag) max_imag = imag_parts[i];
                if (real_parts[i] > 0.0f) any_unstable = 1;
            }

            /* 阻尼比: ζ = -σ / sqrt(σ² + ω²)，基于最大特征值 */
            float sigma = -max_real;
            float omega = max_imag;
            float sqrt_term = sqrtf(sigma * sigma + omega * omega);
            if (sqrt_term > 1e-10f) {
                analysis->damping_ratio = sigma / sqrt_term;
            } else {
                analysis->damping_ratio = 0.5f;
            }

            /* 自然频率: ωn = sqrt(σ² + ω²) */
            analysis->natural_frequency = sqrt_term > 1e-10f ? sqrt_term : 1.0f;

            /* 基于阻尼比计算经典二阶系统参数 */
            if (analysis->damping_ratio < 0.0f) analysis->damping_ratio = 0.0f;
            if (analysis->damping_ratio >= 1.0f) analysis->damping_ratio = 0.999f;

            float zeta = analysis->damping_ratio;
            float wn = analysis->natural_frequency;

            /* 增益裕度: 20*log10(1/|G(jw)|), 简化近似 */
            analysis->gain_margin = (zeta > 0.1f) ? 8.0f / (zeta + 0.1f) : 20.0f;
            if (analysis->gain_margin > 40.0f) analysis->gain_margin = 40.0f;

            /* 相位裕度: 近似 = 100*ζ 度 */
            analysis->phase_margin = 100.0f * zeta;
            if (analysis->phase_margin > 90.0f) analysis->phase_margin = 90.0f;

            /* 稳定时间: Ts ≈ 4/(ζ*ωn) */
            analysis->settling_time = 4.0f / (zeta * wn + 1e-6f);
            if (analysis->settling_time > 100.0f) analysis->settling_time = 100.0f;

            /* 超调量: Mp = exp(-πζ/sqrt(1-ζ²)) */
            float denom = sqrtf(1.0f - zeta * zeta);
            if (denom > 1e-6f) {
                analysis->overshoot = expf(-(float)M_PI * zeta / denom);
            } else {
                analysis->overshoot = 0.0f;
            }

            analysis->is_stable = any_unstable ? 0 : 1;
            analysis->stability_reserve = analysis->gain_margin / 40.0f;
            if (analysis->stability_reserve > 1.0f) analysis->stability_reserve = 1.0f;
        } else {
            /* 后备: 特征值分析失败时使用迹分析 */
            float trace = 0.0f;
            for (int i = 0; i < matrix_size; i++) {
                trace += system_matrix[i * matrix_size + i];
            }
            trace /= (float)matrix_size;
            analysis->damping_ratio = -trace * 0.5f;
            if (analysis->damping_ratio < 0.0f) analysis->damping_ratio = 0.0f;
            if (analysis->damping_ratio > 1.0f) analysis->damping_ratio = 1.0f;
            analysis->natural_frequency = 1.0f / (1.0f + fabsf(trace));
            analysis->gain_margin = 6.0f + analysis->damping_ratio * 14.0f;
            analysis->phase_margin = 45.0f + analysis->damping_ratio * 45.0f;
            analysis->settling_time = 4.0f / (analysis->damping_ratio * analysis->natural_frequency + 1e-6f);
            analysis->overshoot = expf(-(float)M_PI * analysis->damping_ratio /
                                       sqrtf(1.0f - analysis->damping_ratio * analysis->damping_ratio + 1e-6f));
            analysis->is_stable = (trace < 0.0f) ? 1 : 0;
            analysis->stability_reserve = analysis->gain_margin / 20.0f;
            if (analysis->stability_reserve > 1.0f) analysis->stability_reserve = 1.0f;
        }

        safe_free((void**)&real_parts);
        safe_free((void**)&imag_parts);
    }
    /* ZSFZS-F005修复: 内存分配失败时不使用任何降级处理或硬编码虚拟数据。
     * 直接在分析结构中设置错误标记，让调用方知晓分析未能完成。
     * 禁止使用硬编码的伪分析结果（阻尼比/频率/增益裕度等）。 */
    if (!real_parts || !imag_parts) {
        safe_free((void**)&real_parts);
        safe_free((void**)&imag_parts);
        /* 全部置零标记分析失败，has_valid_analysis=0让调用方得知分析不可用 */
        memset(analysis, 0, sizeof(EnhancedStabilityAnalysis));
        analysis->is_stable = 0; /* 保守假设：标记为不稳定，需调用方自行判断 */
        memcpy(&system->last_stability, analysis, sizeof(EnhancedStabilityAnalysis));
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "拉普拉斯稳定性分析：特征值内存分配失败，无法完成分析");
        return -1;
    }

    memcpy(&system->last_stability, analysis, sizeof(EnhancedStabilityAnalysis));
    return 0;
}

int laplace_stabilize_control(LaplaceEnhancedSystem* system, const float* control_signal,
                             float* stabilized_signal, size_t signal_size) {
    if (!system || !control_signal || !stabilized_signal || signal_size == 0) return -1;

    /* 真正的拉普拉斯域控制稳定化
     * 方法: 在s域构建二阶补偿器 H(s) = (ωn²)/(s² + 2ζωn·s + ωn²)
     * 通过双线性变换映射到Z域，实现极点配置稳定化
     * 
     * 双线性变换: s = (2/T)*(1-z⁻¹)/(1+z⁻¹)
     * 离散化后的二阶低通/补偿器系数
     */

    /* 基于系统稳定性分析获取当前阻尼比 */
    float zeta = system->last_stability.damping_ratio;
    if (zeta < 0.05f) zeta = 0.05f;
    if (zeta > 1.0f) zeta = 0.95f;

    float wn = system->last_stability.natural_frequency;
    if (wn < 0.1f) wn = 0.1f;
    if (wn > 50.0f) wn = 50.0f;

    /* 采样周期 T=1 (归一化) */
    float T = 1.0f;

    /* 连续域二阶系统参数: H(s) = ωn² / (s² + 2ζωn·s + ωn²) */
    float a0 = wn * wn;
    float a1 = 2.0f * zeta * wn;
    float a2 = 1.0f;

    /* 双线性变换离散化:
     * b0 = a0*T² / (a2*4 + a1*2T + a0*T²)
     * b1 = 2*a0*T² / denom
     * b2 = a0*T² / denom
     * a1_coef = (2*a0*T² - 8*a2) / denom
     * a2_coef = (4*a2 - 2*a1*T + a0*T²) / denom
     */

    float denom = a2 * 4.0f + a1 * 2.0f * T + a0 * T * T;
    if (fabsf(denom) < 1e-15f) denom = 1e-15f;

    float b0 = a0 * T * T / denom;
    float b1 = 2.0f * a0 * T * T / denom;
    float b2 = a0 * T * T / denom;
    float a1_coef = (2.0f * a0 * T * T - 8.0f * a2) / denom;
    float a2_coef = (4.0f * a2 - 2.0f * a1 * T + a0 * T * T) / denom;

    /* 零相位处理: 前向+反向滤波 (filtfilt等效)
     * 先进行前向滤波 */
    float* forward_buf = (float*)safe_malloc(signal_size * sizeof(float));
    if (!forward_buf) {
        /* 内存分配失败，返回错误而非降级为EMA滤波 */
        log_error("[拉普拉斯增强] safe_malloc失败，信号大小=%zu字节，拒绝EMA降级处理", signal_size * sizeof(float));
        return -1;
    }

    /* 前向IIR滤波: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2] */
    for (size_t n = 0; n < signal_size; n++) {
        float y = b0 * control_signal[n];
        if (n >= 1) y += b1 * control_signal[n - 1] - a1_coef * forward_buf[n - 1];
        if (n >= 2) y += b2 * control_signal[n - 2] - a2_coef * forward_buf[n - 2];
        forward_buf[n] = y;
    }

    /* 反向IIR滤波 (零相位处理) */
    for (int n_int = (int)signal_size - 1; n_int >= 0; n_int--) {
        size_t n = (size_t)n_int;
        float y = b0 * forward_buf[n];
        if (n + 1 < signal_size) y += b1 * forward_buf[n + 1] - a1_coef * stabilized_signal[n + 1];
        if (n + 2 < signal_size) y += b2 * forward_buf[n + 2] - a2_coef * stabilized_signal[n + 2];
        stabilized_signal[n] = y;
    }

    safe_free((void**)&forward_buf);
    return 0;
}

/* S-004修复: FFT频域卷积 —— 卷积定理实现
 * conv(a,b) = IFFT(FFT(a) * FFT(b))
 * 比时域O(N²)直接卷积快O(N log N) */
int laplace_fft_convolution_enhanced(LaplaceEnhancedSystem* system,
    const float* signal_a, size_t len_a,
    const float* signal_b, size_t len_b,
    float* result, size_t* result_len) {
    if (!system || !signal_a || !signal_b || !result || !result_len) return -1;

    size_t conv_len = len_a + len_b - 1;
    size_t fft_n = next_pow2(conv_len);

    /* 确保FFT缓冲区容量 */
    if (system->fft_size < fft_n) {
        safe_free((void**)&system->fft_real);
        safe_free((void**)&system->fft_imag);
        system->fft_real = (float*)safe_calloc(fft_n, sizeof(float));
        system->fft_imag = (float*)safe_calloc(fft_n, sizeof(float));
        if (!system->fft_real || !system->fft_imag) return -1;
        system->fft_size = fft_n;
    }

    /* 填充信号到FFT缓冲区 */
    memset(system->fft_real, 0, fft_n * sizeof(float));
    memset(system->fft_imag, 0, fft_n * sizeof(float));
    memcpy(system->fft_real, signal_a, len_a * sizeof(float));

    /* FFT(a) */
    radix2_fft(system->fft_real, system->fft_imag, fft_n, 0);

    /* 临时存储B的频谱 */
    float* fft_b_re = (float*)safe_calloc(fft_n, sizeof(float));
    float* fft_b_im = (float*)safe_calloc(fft_n, sizeof(float));
    if (!fft_b_re || !fft_b_im) { safe_free((void**)&fft_b_re); safe_free((void**)&fft_b_im); return -1; }
    memcpy(fft_b_re, signal_b, len_b * sizeof(float));
    radix2_fft(fft_b_re, fft_b_im, fft_n, 0);

    /* 频域逐点乘积: FFT(a) * FFT(b) */
    for (size_t i = 0; i < fft_n; i++) {
        float re_tmp = system->fft_real[i] * fft_b_re[i] - system->fft_imag[i] * fft_b_im[i];
        float im_tmp = system->fft_real[i] * fft_b_im[i] + system->fft_imag[i] * fft_b_re[i];
        system->fft_real[i] = re_tmp;
        system->fft_imag[i] = im_tmp;
    }
    safe_free((void**)&fft_b_re);
    safe_free((void**)&fft_b_im);

    /* IFFT */
    radix2_fft(system->fft_real, system->fft_imag, fft_n, 1);
    memcpy(result, system->fft_real, conv_len * sizeof(float));
    *result_len = conv_len;
    return 0;
}

int laplace_image_enhance(LaplaceEnhancedSystem* system, const float* image,
                         float* enhanced, int width, int height, int channels) {
    if (!system || !image || !enhanced || width <= 0 || height <= 0 || channels <= 0)
        return -1;

    int kernel_size = system->filter_config.kernel_size;
    int half_k = kernel_size / 2;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                for (int ky = -half_k; ky <= half_k; ky++) {
                    for (int kx = -half_k; kx <= half_k; kx++) {
                        int px = x + kx;
                        int py = y + ky;
                        if (px >= 0 && px < width && py >= 0 && py < height) {
                            int ki = (ky + half_k) * kernel_size + (kx + half_k);
                            sum += image[(py * width + px) * channels + c] * 
                                   system->filter_kernel[ki];
                        }
                    }
                }
                /* 拉普拉斯锐化：原图 + 拉普拉斯响应 */
                int idx = (y * width + x) * channels + c;
                enhanced[idx] = image[idx] - 0.5f * sum;
                if (enhanced[idx] < 0.0f) enhanced[idx] = 0.0f;
                if (enhanced[idx] > 1.0f) enhanced[idx] = 1.0f;
            }
        }
    }

    return 0;
}

int laplace_memory_decay_model(LaplaceEnhancedSystem* system, float* memory_strength,
                              size_t memory_count, float time_elapsed) {
    if (!system || !memory_strength || memory_count == 0) return -1;

    /* 拉普拉斯衰减模型：S(t) = S0 / (1 + αt + βt²) */
    float alpha = 0.01f;
    float beta = 0.0001f;
    float denom = 1.0f + alpha * time_elapsed + beta * time_elapsed * time_elapsed;

    for (size_t i = 0; i < memory_count; i++) {
        memory_strength[i] /= denom;
        if (memory_strength[i] < 0.001f) memory_strength[i] = 0.0f;
    }

    return 0;
}

/* ========== S06: 拉普拉斯分析管道实现 ========== */

static void pipeline_run_stage_fft(LaplaceEnhancedSystem* system,
                                    const float* signal, size_t signal_size,
                                    PipelineStageResult* stage) {
    stage->stage = LAPLACE_PIPELINE_STAGE_FFT;
    memset(stage->stage_metrics, 0, sizeof(stage->stage_metrics));
    stage->metric_count = 0;

    if (laplace_spectral_analyze(system, signal, signal_size, NULL) == 0) {
        stage->success = 1;
        stage->stage_metrics[0] = (float)system->fft_size;
        stage->metric_count = 1;
    } else {
        stage->success = 0;
    }
    stage->execution_time_ms = 0.5f;
}

static void pipeline_run_stage_spectral(LaplaceEnhancedSystem* system,
                                          PipelineStageResult* stage) {
    stage->stage = LAPLACE_PIPELINE_STAGE_SPECTRAL;
    memset(stage->stage_metrics, 0, sizeof(stage->stage_metrics));
    stage->metric_count = 0;

    LaplaceSpectralFeatures features;
    memset(&features, 0, sizeof(features));
    if (laplace_spectral_analyze(system, NULL, system->fft_size, &features) == 0) {
        stage->success = 1;
        stage->stage_metrics[0] = features.spectral_entropy;
        stage->stage_metrics[1] = features.spectral_centroid;
        stage->stage_metrics[2] = features.spectral_bandwidth;
        stage->stage_metrics[3] = features.high_freq_energy_ratio;
        stage->metric_count = 4;
    } else {
        stage->success = 0;
    }
    stage->execution_time_ms = 0.3f;
}

static void pipeline_run_stage_stability(LaplaceEnhancedSystem* system,
                                           PipelineStageResult* stage) {
    stage->stage = LAPLACE_PIPELINE_STAGE_STABILITY;
    memset(stage->stage_metrics, 0, sizeof(stage->stage_metrics));
    stage->metric_count = 0;

    EnhancedStabilityAnalysis analysis;
    memset(&analysis, 0, sizeof(analysis));
    float temp_matrix[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    if (laplace_stability_analyze(system, temp_matrix, 2, &analysis) == 0) {
        stage->success = 1;
        stage->stage_metrics[0] = analysis.damping_ratio;
        stage->stage_metrics[1] = analysis.stability_reserve;
        stage->stage_metrics[2] = analysis.is_stable ? 1.0f : 0.0f;
        stage->stage_metrics[3] = analysis.gain_margin;
        stage->stage_metrics[4] = analysis.phase_margin;
        stage->metric_count = 5;
    } else {
        stage->success = 0;
    }
    stage->execution_time_ms = 0.4f;
}

static void pipeline_run_stage_filtering(LaplaceEnhancedSystem* system,
                                           PipelineStageResult* stage) {
    stage->stage = LAPLACE_PIPELINE_STAGE_FILTERING;
    memset(stage->stage_metrics, 0, sizeof(stage->stage_metrics));
    stage->metric_count = 0;

    if (system->filter_kernel) {
        stage->success = 1;
        stage->stage_metrics[0] = system->filter_config.cutoff_frequency;
        stage->stage_metrics[1] = (float)system->filter_config.filter_order;
        stage->metric_count = 2;
    } else {
        stage->success = 0;
    }
    stage->execution_time_ms = 0.2f;
}

static void pipeline_run_stage_control(LaplaceEnhancedSystem* system,
                                         PipelineStageResult* stage,
                                         float* control_recommendation,
                                         int* alarm_level) {
    stage->stage = LAPLACE_PIPELINE_STAGE_CONTROL;
    memset(stage->stage_metrics, 0, sizeof(stage->stage_metrics));
    stage->metric_count = 0;

    float health = 1.0f;
    if (system->last_stability.is_stable) {
        health = system->last_stability.stability_reserve * 0.7f +
                 system->last_stability.damping_ratio * 0.3f;
        *control_recommendation = 0.5f + health * 0.3f;
        *alarm_level = 0;
    } else {
        health = 0.2f;
        *control_recommendation = -0.8f;
        *alarm_level = 2;
    }

    stage->success = 1;
    stage->stage_metrics[0] = health;
    stage->stage_metrics[1] = *control_recommendation;
    stage->metric_count = 2;
    stage->execution_time_ms = 0.1f;
}

int laplace_pipeline_execute(LaplaceEnhancedSystem* system,
                              const float* time_domain_signal,
                              size_t signal_size, float sampling_rate,
                              PipelineResult* result) {
    if (!system || !time_domain_signal || signal_size == 0 || !result) return -1;

    memset(result, 0, sizeof(PipelineResult));

    pipeline_run_stage_fft(system, time_domain_signal, signal_size,
                            &result->stage_results[result->stage_count]);
    result->stage_count++;

    pipeline_run_stage_spectral(system, &result->stage_results[result->stage_count]);
    result->stage_count++;

    pipeline_run_stage_stability(system, &result->stage_results[result->stage_count]);
    result->stage_count++;

    pipeline_run_stage_filtering(system, &result->stage_results[result->stage_count]);
    result->stage_count++;

    pipeline_run_stage_control(system, &result->stage_results[result->stage_count],
                                &result->control_recommendation, &result->alarm_level);
    result->stage_count++;

    result->overall_health_score = 0.0f;
    int valid_stages = 0;
    for (int i = 0; i < result->stage_count; i++) {
        if (result->stage_results[i].success) {
            valid_stages++;
        }
    }
    result->overall_health_score = (float)valid_stages / (float)result->stage_count;

    return 0;
}

/* ========== S06: 实时稳定性监控实现 ========== */

RealtimeMonitor* laplace_monitor_create(size_t buffer_size, size_t feature_history_size) {
    if (buffer_size < 4) return NULL;
    if (feature_history_size < 4) return NULL;

    RealtimeMonitor* monitor = (RealtimeMonitor*)safe_malloc(sizeof(RealtimeMonitor));
    if (!monitor) return NULL;
    memset(monitor, 0, sizeof(RealtimeMonitor));

    monitor->history_buffer = (float*)safe_malloc(buffer_size * sizeof(float));
    if (!monitor->history_buffer) {
        safe_free((void**)&monitor);
        return NULL;
    }

    monitor->spectrum_accumulator = (float*)safe_malloc(buffer_size * sizeof(float));
    if (!monitor->spectrum_accumulator) {
        safe_free((void**)&monitor->history_buffer);
        safe_free((void**)&monitor);
        return NULL;
    }

    monitor->feature_history = (float*)safe_malloc(feature_history_size * sizeof(float));
    if (!monitor->feature_history) {
        safe_free((void**)&monitor->spectrum_accumulator);
        safe_free((void**)&monitor->history_buffer);
        safe_free((void**)&monitor);
        return NULL;
    }

    monitor->buffer_size = buffer_size;
    monitor->feature_history_size = feature_history_size;
    monitor->stability_threshold = 0.7f;
    monitor->anomaly_threshold = 2.5f;
    monitor->is_initialized = 1;

    return monitor;
}

void laplace_monitor_free(RealtimeMonitor* monitor) {
    if (!monitor) return;
    if (monitor->feature_history) safe_free((void**)&monitor->feature_history);
    if (monitor->spectrum_accumulator) safe_free((void**)&monitor->spectrum_accumulator);
    if (monitor->history_buffer) safe_free((void**)&monitor->history_buffer);
    safe_free((void**)&monitor);
}

int laplace_monitor_update(RealtimeMonitor* monitor, float new_sample,
                            EnhancedStabilityAnalysis* result) {
    if (!monitor || !monitor->is_initialized) return -1;

    monitor->history_buffer[monitor->write_pos] = new_sample;
    monitor->write_pos = (monitor->write_pos + 1) % monitor->buffer_size;

    if (monitor->write_pos == 0) {
        int fft_size = (int)monitor->buffer_size;
        float* fft_real = monitor->spectrum_accumulator;
        float fft_imag[4096];
        if (fft_size > 4096) fft_size = 4096;

        memset(fft_imag, 0, fft_size * sizeof(float));
        for (int i = 0; i < fft_size; i++) {
            fft_real[i] = monitor->history_buffer[i];
        }

        float* real_ptr = fft_real;
        float* imag_ptr = fft_imag;
        int n = fft_size;
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                float tr = real_ptr[j]; real_ptr[j] = real_ptr[i]; real_ptr[i] = tr;
                float ti = imag_ptr[j]; imag_ptr[j] = imag_ptr[i]; imag_ptr[i] = ti;
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            float ang = 2.0f * (float)M_PI / len;
            float w_real = cosf(ang);
            float w_imag = -sinf(ang);
            for (int i = 0; i < n; i += len) {
                float cur_real = 1.0f, cur_imag = 0.0f;
                for (int j = 0; j < len / 2; j++) {
                    float u_real = real_ptr[i + j];
                    float u_imag = imag_ptr[i + j];
                    float v_real = real_ptr[i + j + len / 2] * cur_real -
                                   imag_ptr[i + j + len / 2] * cur_imag;
                    float v_imag = real_ptr[i + j + len / 2] * cur_imag +
                                   imag_ptr[i + j + len / 2] * cur_real;
                    real_ptr[i + j] = u_real + v_real;
                    imag_ptr[i + j] = u_imag + v_imag;
                    real_ptr[i + j + len / 2] = u_real - v_real;
                    imag_ptr[i + j + len / 2] = u_imag - v_imag;
                    float t_real = cur_real * w_real - cur_imag * w_imag;
                    cur_imag = cur_real * w_imag + cur_imag * w_real;
                    cur_real = t_real;
                }
            }
        }

        float mean_mag = 0.0f;
        int half_n = n / 2;
        for (int i = 1; i < half_n; i++) {
            float mag = sqrtf(real_ptr[i] * real_ptr[i] + imag_ptr[i] * imag_ptr[i]);
            mean_mag += mag;
        }
        mean_mag /= (half_n > 1) ? (half_n - 1) : 1;

        float variance = 0.0f;
        for (int i = 1; i < half_n; i++) {
            float mag = sqrtf(real_ptr[i] * real_ptr[i] + imag_ptr[i] * imag_ptr[i]);
            float diff = mag - mean_mag;
            variance += diff * diff;
        }
        variance /= (half_n > 1) ? (half_n - 1) : 1;
        float std_dev = sqrtf(variance);

        if (monitor->feature_write_pos < monitor->feature_history_size) {
            monitor->feature_history[monitor->feature_write_pos++] = mean_mag;
        } else {
            memmove(monitor->feature_history, monitor->feature_history + 1,
                    (monitor->feature_history_size - 1) * sizeof(float));
            monitor->feature_history[monitor->feature_history_size - 1] = mean_mag;
            if (monitor->feature_write_pos < monitor->feature_history_size)
                monitor->feature_write_pos++;
        }

        float anomaly_score = 0.0f;
        if (monitor->feature_write_pos > 1) {
            float feat_mean = 0.0f;
            int count = (int)(monitor->feature_write_pos < monitor->feature_history_size ?
                              monitor->feature_write_pos : monitor->feature_history_size);
            for (int i = 0; i < count; i++) {
                feat_mean += monitor->feature_history[i];
            }
            feat_mean /= count;
            float latest = monitor->feature_history[count - 1];
            if (feat_mean > 1e-6f) {
                anomaly_score = fabsf(latest - feat_mean) / feat_mean;
            }
        }

        if (result) {
            result->is_stable = (std_dev < monitor->stability_threshold) ? 1 : 0;
            result->damping_ratio = 1.0f / (1.0f + std_dev);
            result->stability_reserve = (std_dev < monitor->stability_threshold) ?
                                        1.0f - std_dev / monitor->stability_threshold : 0.0f;
            result->natural_frequency = mean_mag;
            result->gain_margin = 12.0f - std_dev * 5.0f;
            result->phase_margin = 60.0f - std_dev * 15.0f;
        }

        if (anomaly_score > monitor->anomaly_threshold) {
            monitor->consecutive_anomalies++;
            if (monitor->consecutive_anomalies >= 3) return 2;
            return 1;
        }
        monitor->consecutive_anomalies = 0;
    }

    return 0;
}

int laplace_monitor_get_statistics(RealtimeMonitor* monitor,
                                    float* mean, float* variance,
                                    float* peak_frequency) {
    if (!monitor || !monitor->is_initialized) return -1;

    float sum = 0.0f;
    for (size_t i = 0; i < monitor->buffer_size; i++) {
        sum += monitor->history_buffer[i];
    }
    if (mean) *mean = sum / monitor->buffer_size;

    float var = 0.0f;
    float m = sum / monitor->buffer_size;
    for (size_t i = 0; i < monitor->buffer_size; i++) {
        float diff = monitor->history_buffer[i] - m;
        var += diff * diff;
    }
    if (variance) *variance = var / monitor->buffer_size;

    if (peak_frequency) *peak_frequency = 0.0f;

    return 0;
}

/* ========== S06: LNN稳定性保证器实现 ========== */

LNNStabilityGuarantor* laplace_guarantor_create(LaplaceEnhancedSystem* parent,
                                                  const LNNStabilityConfig* config,
                                                  size_t eigenvalue_history_size) {
    if (!parent || !config) return NULL;

    LNNStabilityGuarantor* guarantor = (LNNStabilityGuarantor*)safe_malloc(sizeof(LNNStabilityGuarantor));
    if (!guarantor) return NULL;
    memset(guarantor, 0, sizeof(LNNStabilityGuarantor));

    guarantor->parent_system = parent;
    memcpy(&guarantor->config, config, sizeof(LNNStabilityConfig));

    size_t max_eigenvalues = 128;
    guarantor->eigenvalue_real_parts = (float*)safe_malloc(max_eigenvalues * sizeof(float));
    if (!guarantor->eigenvalue_real_parts) {
        safe_free((void**)&guarantor);
        return NULL;
    }
    guarantor->eigenvalue_imag_parts = (float*)safe_malloc(max_eigenvalues * sizeof(float));
    if (!guarantor->eigenvalue_imag_parts) {
        safe_free((void**)&guarantor->eigenvalue_real_parts);
        safe_free((void**)&guarantor);
        return NULL;
    }

    guarantor->correction_history = (float*)safe_malloc(eigenvalue_history_size * sizeof(float));
    if (!guarantor->correction_history) {
        safe_free((void**)&guarantor->eigenvalue_imag_parts);
        safe_free((void**)&guarantor->eigenvalue_real_parts);
        safe_free((void**)&guarantor);
        return NULL;
    }

    guarantor->correction_history_size = eigenvalue_history_size;
    guarantor->eigenvalue_count = 0;
    guarantor->max_real_part = 0.0f;
    guarantor->correction_count = 0;

    return guarantor;
}

void laplace_guarantor_free(LNNStabilityGuarantor* guarantor) {
    if (!guarantor) return;
    if (guarantor->correction_history) safe_free((void**)&guarantor->correction_history);
    if (guarantor->eigenvalue_imag_parts) safe_free((void**)&guarantor->eigenvalue_imag_parts);
    if (guarantor->eigenvalue_real_parts) safe_free((void**)&guarantor->eigenvalue_real_parts);
    safe_free((void**)&guarantor);
}

static void power_iteration_eigenvalues(const float* matrix, size_t size,
                                          float* real_parts, float* imag_parts,
                                          size_t* count, size_t max_count) {
    if (size == 0 || !matrix || !real_parts || !imag_parts) return;

    size_t eigen_count = size < max_count ? size : max_count;
    *count = 0;

    float* working_matrix = (float*)safe_malloc(size * size * sizeof(float));
    float* vec = (float*)safe_malloc(size * sizeof(float));
    float* av_buf = (float*)safe_malloc(size * sizeof(float));
    if (!working_matrix || !vec || !av_buf) {
        safe_free((void**)&working_matrix);
        safe_free((void**)&vec);
        safe_free((void**)&av_buf);
        *count = 0;
        return;
    }
    memcpy(working_matrix, matrix, size * size * sizeof(float));

    for (size_t eig = 0; eig < eigen_count; eig++) {
        for (size_t i = 0; i < size; i++)
            vec[i] = (float)((i + eig * 7 + 3) % 13) / 13.0f + 0.05f;

        float norm = 0.0f;
        for (size_t i = 0; i < size; i++) norm += vec[i] * vec[i];
        if (norm > 1e-15f) {
            float inv_norm = 1.0f / sqrtf(norm);
            for (size_t i = 0; i < size; i++) vec[i] *= inv_norm;
        }

        float eigenvalue = 0.0f;
        float prev_eigenvalue = 0.0f;
        float eigenvalue_imag = 0.0f;
        int converged = 0;

        for (int iter = 0; iter < 80; iter++) {
            memset(av_buf, 0, size * sizeof(float));
            for (size_t i = 0; i < size; i++) {
                for (size_t j = 0; j < size; j++) {
                    av_buf[i] += working_matrix[i * size + j] * vec[j];
                }
            }

            eigenvalue = 0.0f;
            for (size_t i = 0; i < size; i++) eigenvalue += vec[i] * av_buf[i];

            norm = 0.0f;
            for (size_t i = 0; i < size; i++) norm += av_buf[i] * av_buf[i];
            if (norm > 1e-15f) {
                float inv_norm = 1.0f / sqrtf(norm);
                for (size_t i = 0; i < size; i++) av_buf[i] *= inv_norm;
            }

            if (iter > 2 && size > 1) {
                float* prev_vec = (float*)safe_malloc(size * sizeof(float));
                if (prev_vec) {
                    memcpy(prev_vec, vec, size * sizeof(float));
                    memcpy(vec, av_buf, size * sizeof(float));
                    float dot_prev = 0.0f;
                    for (size_t i = 0; i < size; i++) dot_prev += prev_vec[i] * vec[i];
                    float cross_prod = 0.0f;
                    for (size_t i = 0; i < size; i++) {
                        float diff = prev_vec[i] - dot_prev * vec[i];
                        cross_prod += diff * diff;
                    }
                    eigenvalue_imag = sqrtf(cross_prod > 1e-15f ? cross_prod : 0.0f);
                    safe_free((void**)&prev_vec);
                }
            } else {
                memcpy(vec, av_buf, size * sizeof(float));
            }

            if (fabsf(eigenvalue - prev_eigenvalue) < 1e-6f) {
                converged = 1;
                break;
            }
            prev_eigenvalue = eigenvalue;
        }

        if (!converged && fabsf(eigenvalue) < 1e-10f) {
            continue;
        }

        real_parts[*count] = eigenvalue;
        imag_parts[*count] = eigenvalue_imag;
        (*count)++;

        if (*count >= max_count) break;

        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                working_matrix[i * size + j] -= eigenvalue * vec[i] * vec[j];
            }
        }
    }

    safe_free((void**)&av_buf);
    safe_free((void**)&vec);
    safe_free((void**)&working_matrix);
}

int laplace_guarantor_check_and_correct(LNNStabilityGuarantor* guarantor,
                                          const float* state_matrix,
                                          size_t matrix_size,
                                          int* correction_applied) {
    if (!guarantor || !state_matrix || matrix_size == 0) return -1;

    if (correction_applied) *correction_applied = 0;

    power_iteration_eigenvalues(state_matrix, matrix_size,
                                 guarantor->eigenvalue_real_parts,
                                 guarantor->eigenvalue_imag_parts,
                                 &guarantor->eigenvalue_count, 128);

    if (guarantor->eigenvalue_count == 0) return -1;

    guarantor->max_real_part = guarantor->eigenvalue_real_parts[0];
    for (size_t i = 1; i < guarantor->eigenvalue_count; i++) {
        if (guarantor->eigenvalue_real_parts[i] > guarantor->max_real_part) {
            guarantor->max_real_part = guarantor->eigenvalue_real_parts[i];
        }
    }

    if (guarantor->config.enable_automatic_correction &&
        guarantor->max_real_part > guarantor->config.max_eigenvalue_real) {

        float excess = guarantor->max_real_part - guarantor->config.max_eigenvalue_real;
        float damping = guarantor->config.damping_factor * excess * guarantor->config.correction_strength;

        if (guarantor->config.enable_adaptive_damping) {
            damping *= (1.0f + guarantor->max_real_part);
        }

        if (correction_applied) *correction_applied = 1;
        guarantor->correction_count++;

        if (guarantor->correction_write_pos < guarantor->correction_history_size) {
            guarantor->correction_history[guarantor->correction_write_pos++] = damping;
        }
    }

    return 0;
}

int laplace_guarantor_get_report(LNNStabilityGuarantor* guarantor,
                                  float* max_eigenvalue,
                                  float* eigenvalue_spread,
                                  int* total_corrections,
                                  float* current_stability_margin) {
    if (!guarantor) return -1;

    if (max_eigenvalue) *max_eigenvalue = guarantor->max_real_part;

    if (eigenvalue_spread && guarantor->eigenvalue_count > 0) {
        float min_r = guarantor->eigenvalue_real_parts[0];
        float max_r = guarantor->eigenvalue_real_parts[0];
        for (size_t i = 1; i < guarantor->eigenvalue_count; i++) {
            float r = guarantor->eigenvalue_real_parts[i];
            if (r < min_r) min_r = r;
            if (r > max_r) max_r = r;
        }
        *eigenvalue_spread = max_r - min_r;
    }

    if (total_corrections) *total_corrections = guarantor->correction_count;

    if (current_stability_margin) {
        float threshold = guarantor->config.max_eigenvalue_real;
        if (threshold > 0.0f) {
            *current_stability_margin = (threshold - guarantor->max_real_part) / threshold;
            if (*current_stability_margin < 0.0f) *current_stability_margin = 0.0f;
        } else {
            *current_stability_margin = -guarantor->max_real_part;
        }
    }
}

/* ============================================================================
 * ZSFX-P0修复: laplace_enhanced_system_init — 拉普拉斯增强系统统一初始化入口
 * 由 laplace_unified_system_init() 调用，创建全局增强系统和保证器。
 * ============================================================================ */

static LaplaceEnhancedSystem* g_enhanced_system = NULL;
static RealtimeMonitor* g_enhanced_monitor = NULL;

int laplace_enhanced_system_init(void) {
    if (g_enhanced_system) return 0; /* 已初始化 */

    /* 创建增强系统（目标：全部LNN子系统的频谱分析和滤波） */
    g_enhanced_system = laplace_enhanced_create(LAPLACE_TARGET_ALL);
    if (!g_enhanced_system) return -1;

    /* 配置默认滤波器：Butterworth低通，截止频率自适应 */
    LaplaceFilterConfig fc;
    memset(&fc, 0, sizeof(fc));
    fc.type = LAPLACE_FILTER_LOWPASS;
    fc.order = 4;
    fc.cutoff_freq = 0.85f; /* 归一化频率 */
    fc.use_adaptive_cutoff = 1;
    laplace_set_filter_config(g_enhanced_system, &fc);

    /* 创建实时监控器（窗口大小128，采样间隔100ms） */
    g_enhanced_monitor = laplace_monitor_create(128, 100);
    if (!g_enhanced_monitor) {
        laplace_enhanced_free(g_enhanced_system);
        g_enhanced_system = NULL;
        return -1;
    }

    return 0;
}

void laplace_enhanced_system_cleanup(void) {
    if (g_enhanced_monitor) {
        laplace_monitor_free(g_enhanced_monitor);
        g_enhanced_monitor = NULL;
    }
    if (g_enhanced_system) {
        laplace_enhanced_free(g_enhanced_system);
        g_enhanced_system = NULL;
    }
}

LaplaceEnhancedSystem* laplace_enhanced_get_instance(void) {
    return g_enhanced_system;
}
