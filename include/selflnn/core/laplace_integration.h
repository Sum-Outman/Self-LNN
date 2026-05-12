#ifndef SELFLNN_LAPLACE_INTEGRATION_H
#define SELFLNN_LAPLACE_INTEGRATION_H

#include "selflnn/core/common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file laplace_integration.h
 * @brief 拉普拉斯变换深度集成接口
 * 
 * 将拉普拉斯变换深度集成到CfC核心中，提供频域分析、
 * 稳定性优化和系统特性分析功能。
 */

/**
 * @brief CfC单元稳定性分析结果
 */
typedef struct {
    float stability_score;          /**< 稳定性分数 (0.0-1.0) */
    float phase_margin;             /**< 相位裕度 (度) */
    float gain_margin;              /**< 增益裕度 (dB) */
    float natural_frequency;        /**< 自然频率 (Hz) */
    float damping_ratio;            /**< 阻尼比 */
    float dominant_pole_real;       /**< 主导极点实部 */
    float dominant_pole_imag;       /**< 主导极点虚部 */
    int is_stable;                  /**< 是否稳定 (1=稳定) */
    int stability_warning;          /**< 稳定性警告 (0=无, 1=临界, 2=不稳定) */
} CfcStabilityAnalysis;

/**
 * @brief 分析CfC单元的稳定性（使用拉普拉斯变换）
 * 
 * @param cell CfC单元
 * @param analysis 稳定性分析结果输出
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_analyze_stability(const void* cell, CfcStabilityAnalysis* analysis);

/**
 * @brief 优化CfC单元参数以提高稳定性（使用拉普拉斯频域优化）
 * 
 * @param cell CfC单元（将被修改）
 * @param target_phase_margin 目标相位裕度（度）
 * @param target_gain_margin 目标增益裕度（dB）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_optimize_stability(void* cell, float target_phase_margin, float target_gain_margin);

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
                                float* phase_response);

/**
 * @brief 设计CfC单元的低通滤波器（基于拉普拉斯变换）
 * 
 * @param cell CfC单元（将被配置为低通滤波器）
 * @param cutoff_frequency 截止频率（Hz）
 * @param filter_order 滤波器阶数（1=一阶，2=二阶）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_design_lowpass_filter(void* cell, float cutoff_frequency, int filter_order);

/**
 * @brief 执行CfC单元的频域系统辨识
 * 
 * @param cell CfC单元
 * @param input_signal 输入信号数组
 * @param output_signal 输出信号数组
 * @param num_samples 样本数量
 * @param sampling_rate 采样率（Hz）
 * @param identified_params 辨识参数输出数组（至少4个元素）
 * @return int 成功返回0，失败返回-1
 */
int cfc_cell_system_identification(const void* cell,
                                   const float* input_signal,
                                   const float* output_signal,
                                   size_t num_samples,
                                   float sampling_rate,
                                   float* identified_params);

/* ========== 频域自适应学习率与频谱分析API ========== */

/**
 * @brief 频谱分析配置
 */
typedef struct {
    size_t fft_size;             /**< FFT点数（必须是2的幂，推荐256/512/1024） */
    float sampling_rate;         /**< 采样率（Hz） */
    float min_frequency;         /**< 最小关注频率（Hz） */
    float max_frequency;         /**< 最大关注频率（Hz） */
    int enable_window;           /**< 是否启用汉宁窗 */
    int enable_log_scale;        /**< 是否使用对数幅度 */
} SpectrumConfig;

/**
 * @brief 频谱数据点
 */
typedef struct {
    float frequency;             /**< 频率（Hz） */
    float magnitude;             /**< 幅度 */
    float phase;                 /**< 相位（度） */
} SpectrumPoint;

/**
 * @brief 频谱分析结果
 */
typedef struct {
    SpectrumPoint* points;       /**< 频谱数据点数组 */
    size_t num_points;           /**< 数据点数量 */
    float dominant_frequency;    /**< 主导频率（Hz） */
    float dominant_magnitude;    /**< 主导幅度 */
    float dc_component;          /**< 直流分量 */
    float total_power;           /**< 总功率 */
    float high_freq_ratio;       /**< 高频成分占比（>奈奎斯特频率/2） */
    float low_freq_ratio;        /**< 低频成分占比（<奈奎斯特频率/4） */
    float spectral_centroid;     /**< 频谱质心（Hz） */
    float bandwidth_3db;         /**< 3dB带宽（Hz） */
} SpectrumResult;

/**
 * @brief 频域自适应学习率配置
 */
typedef struct {
    float base_learning_rate;         /**< 基础学习率 */
    float min_learning_rate;          /**< 最小学习率（基础学习率的倍数） */
    float max_learning_rate;          /**< 最大学习率（基础学习率的倍数） */
    float high_freq_threshold;        /**< 高频噪声阈值（高频占比超过此值降低LR） */
    float low_freq_threshold;         /**< 低频漂移阈值（低频占比超过此值增加LR） */
    float adaptation_speed;           /**< 自适应速度（0.0-1.0） */
    int use_spectral_centroid;        /**< 是否使用频谱质心调节 */
    float momentum;                   /**< 动量因子（平滑LR变化） */
} FreqAdaptiveLRConfig;

/**
 * @brief 获取默认频谱分析配置
 *
 * @return SpectrumConfig 默认配置（FFT=256, 采样率=100Hz, 启用汉宁窗）
 */
SpectrumConfig laplace_spectrum_config_default(void);

/**
 * @brief 获取默认频域自适应学习率配置
 *
 * @param base_lr 基础学习率
 * @return FreqAdaptiveLRConfig 默认配置
 */
FreqAdaptiveLRConfig laplace_freq_adaptive_lr_config_default(float base_lr);

/**
 * @brief 计算信号的频谱（使用FFT）
 *
 * 对输入信号进行FFT变换，计算幅度谱和相位谱。
 * FFT点数必须是2的幂，信号长度不足时自动补零。
 *
 * @param signal 输入信号数组
 * @param signal_length 信号长度
 * @param config 频谱分析配置
 * @param result 频谱分析结果输出（需要调用 laplace_spectrum_result_free 释放）
 * @return int 成功返回0，失败返回-1
 */
int laplace_compute_spectrum(const float* signal, size_t signal_length,
                              const SpectrumConfig* config, SpectrumResult* result);

/**
 * @brief 基于频谱分析的自适应学习率计算
 *
 * 分析梯度历史数据的频谱特性，根据频域分布自适应调整学习率：
 * - 高频噪声占比高 -> 减小学习率（防止震荡）
 * - 低频漂移占比高 -> 增大学习率（加速收敛）
 * - 频谱质心偏移 -> 根据偏移方向调整
 *
 * @param gradient_history 梯度历史数据数组
 * @param history_length 历史数据长度
 * @param config 自适应学习率配置
 * @param spectrum 当前频谱分析结果（可选，如果为NULL则内部计算）
 * @return float 调整后的学习率
 */
float laplace_freq_adaptive_lr(const float* gradient_history, size_t history_length,
                                const FreqAdaptiveLRConfig* config,
                                const SpectrumResult* spectrum);

/**
 * @brief 将频谱分析结果序列化为JSON字符串
 *
 * @param result 频谱分析结果
 * @param buffer 输出缓冲区
 * @param buffer_size 缓冲区大小
 * @return int 成功返回写入的字符数，失败返回-1
 */
int laplace_spectrum_to_json(const SpectrumResult* result,
                              char* buffer, size_t buffer_size);

/**
 * @brief 释放频谱分析结果
 *
 * @param result 频谱分析结果
 */
void laplace_spectrum_result_free(SpectrumResult* result);

/* ========== 拉普拉斯深度集成扩展API ========== */

/**
 * @brief 创建用于LNN训练的默认拉普拉斯分析器
 * 
 * 提供一个预配置的拉普拉斯分析器，适用于LNN训练场景：
 * - 采样率: 1000Hz
 * - 最大频率: 100Hz  
 * - 最小频率: 0.1Hz
 * - 采样点数: 256
 * - 启用稳定性和频域分析
 * - 截止频率: 10Hz（低通滤波）
 * - 滤波器阶数: 4
 * 
 * @return void* 分析器句柄，失败返回NULL（需要调用laplace_analyzer_free释放）
 */
void* lnn_laplace_create_default_analyzer(void);

/**
 * @brief 频域前向调制
 * 
 * 使用拉普拉斯频域分析对隐藏状态进行频域调制：
 * 1. 计算隐藏状态的频域特性（主导频率、带宽）
 * 2. 根据频域特性调整隐藏状态的高频/低频成分
 * 3. 使网络动态适应输入的频域特征
 * 
 * @param analyzer 拉普拉斯分析器句柄
 * @param hidden_state 隐藏状态向量（会被就地修改）
 * @param hidden_size 隐藏状态维度
 * @param modulation_strength 调制强度（0.0-1.0，推荐0.1-0.3）
 * @return int 成功返回0，失败返回-1
 */
int lnn_laplace_modulate_hidden(void* analyzer, float* hidden_state,
                                 size_t hidden_size, float modulation_strength);

/**
 * @brief 计算LNN的频域稳定性指标（增强版）
 * 
 * 超越简单的1阶稳定性分析，结合所有CfC单元的时间常数和
 * 当前激活状态进行综合频域分析。提供：
 * - 多极点稳定性评估
 * - 频域响应估计
 * - 自适应滤波建议
 * 
 * @param analyzer 拉普拉斯分析器句柄
 * @param time_constant 网络时间常数（秒）
 * @param hidden_state 当前隐藏状态（用于估计反馈强度）
 * @param hidden_size 隐藏状态维度
 * @param stability_score 输出稳定性分数（0.0-1.0）
 * @param recommended_cutoff 输出推荐的滤波截止频率（Hz）
 * @param frequency_bandwidth 输出频域带宽（Hz）
 * @return int 成功返回0，失败返回-1
 */
int lnn_laplace_analyze_network_dynamics(void* analyzer,
                                          float time_constant,
                                          const float* hidden_state,
                                          size_t hidden_size,
                                          float* stability_score,
                                          float* recommended_cutoff,
                                          float* frequency_bandwidth);

/* ========== 分数阶拉普拉斯记忆模块 ========== */

/**
 * @brief 分数阶记忆配置
 */
typedef struct {
    float fractional_order;          /**< 分数阶阶数 (0.0~2.0, 0.5=半阶记忆) */
    float memory_decay;              /**< 记忆衰减因子 (0.0~1.0) */
    size_t memory_length;            /**< 记忆长度 (历史窗口大小) */
    int use_caputo_derivative;       /**< 是否使用Caputo导数 (0=Grünwald-Letnikov) */
    int enable_adaptive_order;       /**< 是否自适应调整分数阶 */
    float order_adaptation_rate;     /**< 阶数自适应速率 */
    float min_fractional_order;      /**< 最小分数阶 */
    float max_fractional_order;      /**< 最大分数阶 */
} FractionalMemoryConfig;

/**
 * @brief 获取默认分数阶记忆配置
 *
 * @return FractionalMemoryConfig 默认配置（阶数=0.5, 衰减=0.95, 长度=64）
 */
FractionalMemoryConfig laplace_fractional_memory_config_default(void);

/**
 * @brief 分数阶积分（Grünwald-Letnikov定义）
 *
 * 实现D^{-α} f(t) = 1/Γ(α) * ∫_0^t (t-τ)^{α-1} f(τ) dτ
 * 离散近似: I^α f[n] = T^α * Σ_{k=0}^{N} w_k * f[n-k]
 * w_k = Γ(k+α) / (Γ(α) * Γ(k+1))
 *
 * @param signal 输入信号数组
 * @param length 信号长度
 * @param order 积分阶数 (正数, 0.5=半阶积分)
 * @param dt 采样时间步长
 * @param output 输出数组（与输入同长度）
 * @return int 成功返回0，失败返回-1
 */
int laplace_fractional_integral(const float* signal, size_t length,
                                 float order, float dt, float* output);

/**
 * @brief 分数阶导数（Grünwald-Letnikov定义）
 *
 * 实现D^{α} f(t) = lim_{h→0} h^{-α} * Σ_{k=0}^{∞} (-1)^k * C(α,k) * f(t-kh)
 *
 * @param signal 输入信号数组
 * @param length 信号长度
 * @param order 导数阶数 (正数, 0.5=半阶导数)
 * @param dt 采样时间步长
 * @param output 输出数组（与输入同长度）
 * @return int 成功返回0，失败返回-1
 */
int laplace_fractional_derivative(const float* signal, size_t length,
                                   float order, float dt, float* output);

/**
 * @brief 分数阶记忆滤波器
 *
 * 将分数阶动态应用于隐藏状态，实现长期记忆建模：
 * h_new[i] = h_old[i] + α * D^{-β} h_old[i]
 * 其中β是分数阶，α是记忆强度
 *
 * @param hidden_state 隐藏状态（会被修改）
 * @param hidden_size 隐藏状态维度
 * @param config 分数阶记忆配置
 * @param dt 时间步长
 * @param buffer 工作缓冲区（至少hidden_size * config.memory_length大小）
 * @return int 成功返回0，失败返回-1
 */
int laplace_fractional_memory_filter(float* hidden_state, size_t hidden_size,
                                      const FractionalMemoryConfig* config,
                                      float dt, float* buffer);

/**
 * @brief 自适应学习分数阶
 *
 * 根据隐藏状态的动态特性自动调整最优分数阶：
 * - 快速变化→高阶导数（接近1.0）
 * - 慢速变化→低阶导数/积分（接近0.0）
 *
 * @param hidden_state_history 隐藏状态历史（memory_length x hidden_size）
 * @param hidden_size 隐藏状态维度
 * @param history_length 历史长度
 * @param current_order 当前分数阶（会被修改）
 * @param config 自适应配置
 * @return int 成功返回0，失败返回-1
 */
int laplace_learn_fractional_order(const float* hidden_state_history,
                                    size_t hidden_size, size_t history_length,
                                    float* current_order,
                                    const FractionalMemoryConfig* config);

/* ========== 递归最小二乘在线系统辨识 ========== */

/**
 * @brief RLS估计器状态
 */
typedef struct {
    float theta[4];              /**< 参数估计 [a1, a2, b0, b1] */
    float P[16];                 /**< 协方差矩阵 (4x4, 按行存储) */
    float forgetting_factor;     /**< 遗忘因子 (0.95~1.0) */
    float delta;                 /**< 初始化协方差参数 */
    int model_order;             /**< 模型阶数 (1或2) */
    float lambda;                /**< 正则化参数 */
    float phi_buffer[4];         /**< 回归向量缓冲区 */
    int is_initialized;          /**< 是否已初始化 */
} RLSEstimator;

/**
 * @brief 初始化RLS估计器
 *
 * @param estimator RLS估计器（会被初始化）
 * @param forgetting_factor 遗忘因子 (0.95~1.0, 推荐0.98)
 * @param model_order 模型阶数 (1=一阶, 2=二阶)
 * @param delta 初始协方差参数 (推荐100.0)
 * @return int 成功返回0，失败返回-1
 */
int laplace_rls_init(RLSEstimator* estimator, float forgetting_factor,
                     int model_order, float delta);

/**
 * @brief RLS在线更新一步
 *
 * 执行一步RLS更新：
 * 1. 构建回归向量 φ[n] = [-y[n-1], -y[n-2], u[n-1], u[n-2]]
 * 2. 计算先验误差 e[n] = y[n] - φ^T[n] * θ[n-1]
 * 3. 计算卡尔曼增益 K[n] = P[n-1] * φ[n] / (λ + φ^T[n] * P[n-1] * φ[n])
 * 4. 更新参数 θ[n] = θ[n-1] + K[n] * e[n]
 * 5. 更新协方差 P[n] = (P[n-1] - K[n] * φ^T[n] * P[n-1]) / λ
 *
 * @param estimator RLS估计器
 * @param y 当前输出测量值
 * @param u 当前输入值
 * @return int 成功返回0，失败返回-1
 */
int laplace_rls_update(RLSEstimator* estimator, float y, float u);

/**
 * @brief 获取RLS估计的连续时间参数
 *
 * 将离散时间参数转换为连续时间参数：
 * 一阶：G(s) = K / (τs + 1)
 * 二阶：G(s) = ω_n^2 / (s^2 + 2ζω_n s + ω_n^2)
 *
 * @param estimator RLS估计器
 * @param dt 采样时间
 * @param time_constant 输出时间常数 τ
 * @param gain 输出增益 K
 * @param natural_freq 输出自然频率 ω_n（二阶）
 * @param damping_ratio 输出阻尼比 ζ（二阶）
 * @return int 成功返回0，失败返回-1
 */
int laplace_rls_get_continuous_params(const RLSEstimator* estimator,
                                       float dt, float* time_constant,
                                       float* gain, float* natural_freq,
                                       float* damping_ratio);

/* ========== 拉普拉斯域PID自动调谐 ========== */

/**
 * @brief PID控制器参数
 */
typedef struct {
    float kp;                    /**< 比例增益 */
    float ki;                    /**< 积分增益 */
    float kd;                    /**< 微分增益 */
    float setpoint_weight;       /**< 设定值加权 (0~1) */
    float derivative_filter;     /**< 微分滤波系数 (0~1, 0=无滤波) */
    float output_limit_min;      /**< 输出下限 */
    float output_limit_max;      /**< 输出上限 */
    float integral_limit_min;    /**< 积分项下限 */
    float integral_limit_max;    /**< 积分项上限 */
} PIDParams;

/**
 * @brief 基于系统辨识的PID自动调谐（Ziegler-Nichols方法）
 *
 * 使用RLS辨识的系统模型，自动计算PID参数：
 * 1. 从模型参数计算临界增益Ku和临界周期Tu
 * 2. 应用Ziegler-Nichols整定规则
 * 3. 可选精细调节（基于幅值裕度和相位裕度）
 *
 * @param time_constant 系统时间常数
 * @param gain 系统增益
 * @param dead_time 系统延迟时间（秒）
 * @param controller_type 控制器类型 (0=P, 1=PI, 2=PID)
 * @param params PID参数输出
 * @return int 成功返回0，失败返回-1
 */
int laplace_pid_auto_tune(float time_constant, float gain,
                           float dead_time, int controller_type,
                           PIDParams* params);

/**
 * @brief 超前-滞后补偿器设计
 *
 * 设计超前或滞后补偿器来改善系统性能：
 * - 超前补偿器：增加相位裕度，提高响应速度
 * - 滞后补偿器：降低稳态误差，抑制高频噪声
 *
 * @param phase_margin_target 目标相位裕度（度）
 * @param crossover_freq 穿越频率（rad/s）
 * @param is_lead 超前(1)或滞后(0)补偿
 * @param lead_ratio 超前比（超前用，推荐3~10）
 * @param zero_time 输出补偿器零点时间常数
 * @param pole_time 输出补偿器极点时间常数
 * @param compensator_gain 输出补偿器增益
 * @return int 成功返回0，失败返回-1
 */
int laplace_lead_lag_design(float phase_margin_target,
                             float crossover_freq, int is_lead,
                             float lead_ratio,
                             float* zero_time, float* pole_time,
                             float* compensator_gain);

/* ========== 自适应频谱门控 ========== */

/**
 * @brief 频谱门控配置
 */
typedef struct {
    float noise_floor;             /**< 噪声底噪估计 */
    float gate_threshold;          /**< 门控阈值 (dB, 推荐-40) */
    float attack_time;             /**< 启动时间 (秒, 推荐0.001) */
    float release_time;            /**< 释放时间 (秒, 推荐0.050) */
    float frequency_smoothing;     /**< 频域平滑系数 (0~1) */
    int use_wiener_filter;         /**< 是否使用维纳滤波 */
    float wiener_alpha;            /**< 维纳滤波指数 (0.5~2.0) */
    int enable_noise_estimation;   /**< 是否启用噪声估计 */
    float noise_estimation_rate;   /**< 噪声估计更新率 */
    float min_gain;                /**< 最小增益 (dB, 推荐-80) */
    float max_gain;                /**< 最大增益 (dB, 推荐0) */
    float bandwidth_hz;            /**< 处理带宽 (Hz, 0=全频段) */
    float center_freq_hz;          /**< 中心频率（如果带宽>0） */
} SpectralGateConfig;

/**
 * @brief 获取默认频谱门控配置
 *
 * @return SpectralGateConfig 默认配置
 */
SpectralGateConfig laplace_spectral_gate_config_default(void);

/**
 * @brief 自适应频谱门控
 *
 * 在频域中应用自适应门控，根据信号-噪声比抑制噪声：
 * 1. 对输入信号做FFT
 * 2. 计算每个频点的SNR估计
 * 3. 应用门控/维纳增益
 * 4. IFFT恢复时域信号
 *
 * @param input_signal 输入时域信号
 * @param output_signal 输出时域信号
 * @param signal_length 信号长度
 * @param config 频谱门控配置
 * @param config 可选的噪声谱估计 (NULL=自动估计)
 * @return int 成功返回0，失败返回-1
 */
int laplace_spectral_gate(const float* input_signal, float* output_signal,
                           size_t signal_length,
                           const SpectralGateConfig* config,
                           const float* noise_spectrum);

/**
 * @brief 更新噪声谱估计（滑动平均）
 *
 * @param noise_spectrum 当前噪声谱估计（会被更新）
 * @param current_spectrum 当前频谱幅度
 * @param num_bins 频点数量
 * @param update_rate 更新率 (0~1)
 */
void laplace_update_noise_spectrum(float* noise_spectrum,
                                    const float* current_spectrum,
                                    size_t num_bins, float update_rate);

/* ========== 多变量拉普拉斯稳定性（全LNN分析） ========== */

/**
 * @brief LNN全网络频域分析结果
 */
typedef struct {
    size_t num_cells;              /**< CfC单元数量 */
    float* pole_real_parts;        /**< 所有极点实部数组 */
    float* pole_imag_parts;        /**< 所有极点虚部数组 */
    int* pole_stability;           /**< 每个极点的稳定性标志 */
    float* cell_bandwidths;        /**< 每个单元的带宽 (Hz) */
    float overall_stability;       /**< 整体稳定性 (0.0~1.0) */
    float worst_case_margin;       /**< 最坏情况稳定裕度 */
    float system_bandwidth;        /**< 系统总带宽 (Hz) */
    int num_poles;                 /**< 总极点数量 */
    float nyquist_stability;       /**< 奈奎斯特稳定性指数 */
    float characteristic_loci_margin; /**< 特征轨迹稳定裕度 */
} LNNStabilityResult;

/**
 * @brief 分析全LNN网络的频域稳定性
 *
 * 超越单单元分析，考虑全网络交互：
 * 1. 收集所有CfC单元的极点
 * 2. 评估单元间耦合的稳定性影响
 * 3. 计算整体奈奎斯特稳定性
 * 4. 提供频域稳定性报告
 *
 * @param time_constants 所有单元的时间常数数组
 * @param feedback_strengths 所有单元的反馈强度数组
 * @param coupling_matrix 单元间耦合矩阵 (num_cells x num_cells, NULL=无耦合)
 * @param num_cells 单元数量
 * @param result 分析结果输出
 * @return int 成功返回0，失败返回-1
 */
int laplace_lnn_full_stability(const float* time_constants,
                                const float* feedback_strengths,
                                const float* coupling_matrix,
                                size_t num_cells,
                                LNNStabilityResult* result);

/**
 * @brief 释放LNN稳定性结果
 *
 * @param result 稳定性结果
 */
void laplace_lnn_stability_result_free(LNNStabilityResult* result);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_LAPLACE_INTEGRATION_H */