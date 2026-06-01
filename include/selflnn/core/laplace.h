/**
 * @file laplace.h
 * @brief 拉普拉斯变换增强系统
 * 
 * 提供拉普拉斯变换分析、系统稳定性评估和频域优化功能。
 * 将拉普拉斯变换技术全面引入液态神经网络系统，增强系统功能。
 */

#ifndef SELFLNN_LAPLACE_H
#define SELFLNN_LAPLACE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 复数结构体
 */
typedef struct {
    float real;  /**< 实部 */
    float imag;  /**< 虚部 */
} Complex;

/**
 * @brief 拉普拉斯分析配置
 */
typedef struct {
    size_t num_samples;      /**< 采样点数 */
    float sample_rate;       /**< 采样率 (Hz) */
    float max_frequency;     /**< 最大分析频率 (Hz) */
    float min_frequency;     /**< 最小分析频率 (Hz) */
    int enable_stability;    /**< 是否启用稳定性分析 */
    size_t buffer_size;      /**< 内部缓冲区大小（字节），用于稳定性分析缓冲区分配 */
    int enable_frequency;    /**< 是否启用频域分析 */
    int enable_optimization; /**< 是否启用优化 */
    float cutoff_frequency;  /**< 截止频率 (Hz) */
    int filter_order;        /**< 滤波器阶数 */
    float alpha;             /**< 滤波器参数 alpha */
    float beta;              /**< 滤波器参数 beta */
    float frequency_range;   /**< 频率范围 */
    int enable_auto_tuning;  /**< 是否启用自动调谐 */
    float stability_threshold; /**< 稳定性阈值 */
} LaplaceConfig;

/* M-025修复：公共默认配置（消除6处重复初始化） */
extern const LaplaceConfig LAPLACE_CONFIG_DEFAULT;

/**
 * @brief 获取拉普拉斯分析器的推荐默认配置
 * @return LaplaceConfig 默认配置（num_samples=256, sample_rate=1000, cutoff=50Hz, order=2）
 */
const LaplaceConfig* laplace_get_default_config(void);

/**
 * @brief 系统极点
 */
typedef struct {
    Complex pole;           /**< 极点位置 (实部, 虚部) */
    float damping_ratio;    /**< 阻尼比 */
    float natural_freq;     /**< 自然频率 (Hz) */
    int is_stable;          /**< 是否稳定 (实部 < 0) */
} SystemPole;

/**
 * @brief 频域响应
 */
typedef struct {
    float frequency;        /**< 频率 (Hz) */
    float magnitude;        /**< 幅度响应 (dB) */
    float phase;            /**< 相位响应 (度) */
    float group_delay;      /**< 群延迟 (秒) */
} FrequencyResponse;

/**
 * @brief 稳定性分析结果
 */
typedef struct {
    SystemPole* poles;      /**< 极点数组 */
    size_t num_poles;       /**< 极点数量 */
    float stability_margin; /**< 稳定裕度 (0-1) */
    float dominant_pole;    /**< 主导极点实部 */
    float bandwidth;        /**< 带宽 (Hz) */
    int is_stable;          /**< 系统是否稳定 */
} StabilityAnalysis;

/**
 * @brief 拉普拉斯分析器句柄
 */
typedef struct LaplaceAnalyzer LaplaceAnalyzer;

/**
 * @brief 创建拉普拉斯分析器
 * 
 * @param config 分析器配置
 * @return LaplaceAnalyzer* 分析器句柄，失败返回NULL
 */
LaplaceAnalyzer* laplace_analyzer_create(const LaplaceConfig* config);

/**
 * @brief 释放拉普拉斯分析器
 * 
 * @param analyzer 分析器句柄
 */
void laplace_analyzer_free(LaplaceAnalyzer* analyzer);

/**
 * @brief 分析系统传递函数
 * 
 * @param analyzer 分析器句柄
 * @param numerator 传递函数分子系数
 * @param denominator 传递函数分母系数
 * @param num_order 分子阶数
 * @param den_order 分母阶数
 * @param result 稳定性分析结果输出
 * @return int 成功返回0，失败返回错误码
 */
int laplace_analyze_system(LaplaceAnalyzer* analyzer,
                           const float* numerator,
                           const float* denominator,
                           size_t num_order,
                           size_t den_order,
                           StabilityAnalysis* result);

/**
 * @brief 计算频域响应
 * 
 * @param analyzer 分析器句柄
 * @param numerator 传递函数分子系数
 * @param denominator 传递函数分母系数
 * @param num_order 分子阶数
 * @param den_order 分母阶数
 * @param frequencies 频率数组 (Hz)
 * @param responses 频域响应输出数组
 * @param num_frequencies 频率数量
 * @return int 成功返回0，失败返回错误码
 */
int laplace_compute_frequency_response(LaplaceAnalyzer* analyzer,
                                       const float* numerator,
                                       const float* denominator,
                                       size_t num_order,
                                       size_t den_order,
                                       const float* frequencies,
                                       FrequencyResponse* responses,
                                       size_t num_frequencies);



/**
 * @brief 优化训练过程的拉普拉斯域
 * 
 * @param analyzer 分析器句柄
 * @param gradients 梯度数组
 * @param num_gradients 梯度数量
 * @param learning_rate 学习率
 * @param optimized_gradients 优化后的梯度输出
 * @return int 成功返回0，失败返回错误码
 */
int laplace_optimize_training(LaplaceAnalyzer* analyzer,
                              const float* gradients,
                              size_t num_gradients,
                              float learning_rate,
                              float* optimized_gradients);

/**
 * @brief 获取拉普拉斯分析器配置
 * 
 * @param analyzer 分析器句柄
 * @param config 配置输出
 * @return int 成功返回0，失败返回错误码
 */
int laplace_analyzer_get_config(const LaplaceAnalyzer* analyzer, LaplaceConfig* config);

/**
 * @brief 设置拉普拉斯分析器配置
 * 
 * @param analyzer 分析器句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回错误码
 */
int laplace_analyzer_set_config(LaplaceAnalyzer* analyzer, const LaplaceConfig* config);

/**
 * @brief 重置拉普拉斯分析器
 * 
 * @param analyzer 分析器句柄
 */
void laplace_analyzer_reset(LaplaceAnalyzer* analyzer);

/* ================================================================
 *补: 极点分析和稳定性裕度查询API
 * 这些函数访问缓存的StabilityAnalysis结果(last_analysis)，
 * 为laplace_unified_health_check提供真实诊断数据。
 * ================================================================ */

/**
 * @brief 获取缓存的极点总数
 * @return 极点数量（需先调用laplace_analyze_stability填充缓存）
 */
int laplace_analyzer_count_poles(const LaplaceAnalyzer* analyzer);

/**
 * @brief 获取不稳定极点数量（实部>=0）
 * @return 不稳定极点数量
 */
int laplace_analyzer_count_unstable_poles(const LaplaceAnalyzer* analyzer);

/**
 * @brief 获取稳定性裕度
 * @param gain_margin 输出: 增益裕度(dB)，越高越稳定
 * @param phase_margin 输出: 相位裕度(度)
 * @return 0成功，-1失败(no cached analysis)
 */
int laplace_analyzer_get_stability_margins(const LaplaceAnalyzer* analyzer,
                                           float* gain_margin, float* phase_margin);

/**
 * @brief 获取频域响应数据
 * @param freq_response 输出: 频率响应数组(需调用者释放)
 * @param size 输出: 数组大小
 * @return 0成功，-1失败
 */
int laplace_analyzer_get_frequency_response(const LaplaceAnalyzer* analyzer,
                                            float** freq_response, size_t* size);

/**
 * @brief 拉普拉斯频域调制隐藏状态
 *
 * 对LNN隐藏状态施加拉普拉斯频域滤波，衰减不稳定频率分量，
 * 增强网络的动态稳定性。使用基于极点的带通调制。
 *
 * @param analyzer 拉普拉斯分析器句柄
 * @param hidden 隐藏状态数组（原地调制）
 * @param hidden_size 隐藏状态维度
 * @param strength 调制强度（0.0~1.0）
 * @return int 成功返回0，失败返回-1
 */
int lnn_laplace_modulate_hidden(LaplaceAnalyzer* analyzer,
                                float* hidden, size_t hidden_size, float strength);

/**
 * @brief 拉普拉斯分析网络动力学稳定性
 *
 * 对LNN隐藏状态进行频域分析，评估网络的动态稳定性，
 * 计算稳定性评分、推荐截止频率和有效带宽。
 *
 * @param analyzer 拉普拉斯分析器句柄
 * @param time_constant CfC时间常数τ
 * @param hidden_state 隐藏状态数组
 * @param hidden_size 隐藏状态维度
 * @param stability_score 输出：稳定性评分（0.0~1.0，越大越稳定）
 * @param recommended_cutoff 输出：推荐截止频率（Hz）
 * @param frequency_bandwidth 输出：有效频带宽度（Hz）
 */
void lnn_laplace_analyze_network_dynamics(LaplaceAnalyzer* analyzer,
                                          float time_constant,
                                          const float* hidden_state,
                                          size_t hidden_size,
                                          float* stability_score,
                                          float* recommended_cutoff,
                                          float* frequency_bandwidth);

/**
 * @brief 计算复数运算
 */

Complex complex_create(float real, float imag);
Complex complex_add(Complex a, Complex b);
Complex complex_sub(Complex a, Complex b);
Complex complex_mul(Complex a, Complex b);
Complex complex_div(Complex a, Complex b);
Complex complex_conjugate(Complex a);
float complex_magnitude(Complex a);
float complex_phase(Complex a);
Complex complex_exp(Complex a);
Complex complex_sqrt(Complex a);

/**
 * @brief 计算多项式在复平面上的值
 * 
 * @param coefficients 多项式系数 (从常数项到最高次项)
 * @param order 多项式阶数
 * @param s 复变量
 * @return Complex 多项式值
 */
Complex complex_polyval(const float* coefficients, size_t order, Complex s);

/**
 * @brief 计算传递函数在复平面上的值
 * 
 * @param numerator 分子系数
 * @param denominator 分母系数
 * @param num_order 分子阶数
 * @param den_order 分母阶数
 * @param s 复变量
 * @return Complex 传递函数值
 */
Complex complex_transfer_function(const float* numerator,
                                  const float* denominator,
                                  size_t num_order,
                                  size_t den_order,
                                  Complex s);

/**
 * @brief 奈奎斯特图数据点
 */
typedef struct {
    Complex value;          /**< 复数响应值 */
    float frequency;        /**< 对应频率 (Hz) */
    float magnitude_db;     /**< 幅度响应 (dB) */
    float phase_deg;        /**< 相位响应 (度) */
} NyquistPoint;

/**
 * @brief 尼科尔斯图数据点
 */
typedef struct {
    float magnitude_db;     /**< 幅度响应 (dB) */
    float phase_deg;        /**< 相位响应 (度) */
    float frequency;        /**< 对应频率 (Hz) */
} NicholsPoint;

/**
 * @brief 计算奈奎斯特图
 * 
 * @param analyzer 分析器句柄
 * @param numerator 分子多项式系数
 * @param denominator 分母多项式系数
 * @param num_order 分子阶数
 * @param den_order 分母阶数
 * @param frequencies 频率数组 (Hz)
 * @param nyquist_points 奈奎斯特图数据点输出数组
 * @param num_points 点数
 * @return int 成功返回0，失败返回错误码
 */
int laplace_compute_nyquist_plot(LaplaceAnalyzer* analyzer,
                                 const float* numerator,
                                 const float* denominator,
                                 size_t num_order,
                                 size_t den_order,
                                 const float* frequencies,
                                 NyquistPoint* nyquist_points,
                                 size_t num_points);

/**
 * @brief 计算尼科尔斯图
 * 
 * @param analyzer 分析器句柄
 * @param numerator 分子多项式系数
 * @param denominator 分母多项式系数
 * @param num_order 分子阶数
 * @param den_order 分母阶数
 * @param frequencies 频率数组 (Hz)
 * @param nichols_points 尼科尔斯图数据点输出数组
 * @param num_points 点数
 * @return int 成功返回0，失败返回错误码
 */
int laplace_compute_nichols_plot(LaplaceAnalyzer* analyzer,
                                 const float* numerator,
                                 const float* denominator,
                                 size_t num_order,
                                 size_t den_order,
                                 const float* frequencies,
                                 NicholsPoint* nichols_points,
                                 size_t num_points);

/**
 * @brief 计算伯德图（幅度和相位）
 * 
 * 这是laplace_compute_frequency_response的增强版本，提供更详细的伯德图数据。
 * 
 * @param analyzer 分析器句柄
 * @param numerator 分子多项式系数
 * @param denominator 分母多项式系数
 * @param num_order 分子阶数
 * @param den_order 分母阶数
 * @param frequencies 频率数组 (Hz)
 * @param magnitude_db 幅度响应输出数组 (dB)
 * @param phase_deg 相位响应输出数组 (度)
 * @param num_points 点数
 * @return int 成功返回0，失败返回错误码
 */
int laplace_compute_bode_plot(LaplaceAnalyzer* analyzer,
                              const float* numerator,
                              const float* denominator,
                              size_t num_order,
                              size_t den_order,
                              const float* frequencies,
                              float* magnitude_db,
                              float* phase_deg,
                              size_t num_points);

/* 前向声明液态神经网络类型 */
typedef struct LNN LNN;

/**
 * @brief 分析液态神经网络的稳定性
 * 
 * 此函数将液态神经网络视为连续时间动态系统，分析其稳定性特征。
 * 
 * @param analyzer 拉普拉斯分析器句柄
 * @param lnn 液态神经网络句柄
 * @param result 稳定性分析结果输出
 * @return int 成功返回0，失败返回错误码
 */
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4030 4028)  // 禁用参数不匹配警告
#endif
int laplace_analyze_lnn_stability(LaplaceAnalyzer* analyzer,
                                  LNN* lnn,
                                  StabilityAnalysis* result);
#ifdef _MSC_VER
#pragma warning(pop)
#endif

/**
 * @brief 拉普拉斯频谱感知的LNN权重调整
 *
 * 利用频域稳定性分析对LNN权重矩阵进行频谱感知微调。
 * 闭环：拉普拉斯分析结果 → 权重调整 → 下一次分析的基准。
 *
 * @param lnn 液态神经网络
 * @param analyzer 拉普拉斯分析器
 * @param strength 调整强度 (0-1, 建议0.05-0.15)
 * @return 0成功, -1失败
 */
int laplace_spectral_weight_tuning(LNN* lnn, LaplaceAnalyzer* analyzer, float strength);

/**
 * @brief 提取拉普拉斯频域特征向量作为LNN辅助输入
 *
 * 从拉普拉斯分析中提取6维频域特征向量：
 * [stability_margin, dominant_pole, bandwidth, is_stable, damping_proxy, natfreq_proxy]
 *
 * @param analyzer 拉普拉斯分析器
 * @param features 输出特征向量 [6]
 * @return 0成功, -1失败
 */
int laplace_extract_spectral_features(LaplaceAnalyzer* analyzer, float features[6]);

/* ========== S05-IPCF: 改进极点计算框架 ========== */

/**
 * @brief IPCF极点计算（Improved Pole Computation Framework）
 * 
 * 使用Laguerre方法进行多项式求根，结合自适应步长控制和多重根检测。
 * 相比Durand-Kerner，Laguerre方法对多重根更鲁棒且收敛更快。
 * 
 * @param denominator 分母多项式系数
 * @param den_order 分母阶数
 * @param poles 计算出的极点数组（调用者分配，至少den_order个）
 * @param max_iterations 最大迭代次数
 * @param tolerance 收敛容差
 * @param iterations_used 返回实际使用的迭代次数（可为NULL）
 * @return int 成功返回0，失败返回错误码
 */
int laplace_compute_poles_ipcf(const float* denominator, size_t den_order,
                                Complex* poles, size_t max_iterations,
                                float tolerance, int* iterations_used);

/* ========== S05-鲁棒劳斯-赫尔维茨判据 ========== */

/**
 * @brief 鲁棒劳斯-赫尔维茨稳定性判据（含特殊行处理）
 * 
 * 构建完整的劳斯阵列，包括：
 * - 首列为0时的ε替换法
 * - 全零行时的辅助多项式求导法
 * - 符号变化计数用于判定不稳定极点数量
 * 
 * @param denominator 分母多项式系数（从常数项到最高次项）
 * @param den_order 分母阶数
 * @param is_stable 输出：是否稳定
 * @param right_half_poles 输出：右半平面极点数量
 * @param stability_margin 输出：稳定裕度（0-1）
 * @return int 成功返回0，失败返回错误码
 */
int laplace_robust_routh_hurwitz(const float* denominator, size_t den_order,
                                  int* is_stable, int* right_half_poles,
                                  float* stability_margin);

/**
 * @brief 多维劳斯-赫尔维茨判据
 * 
 * 用于分析多输入多输出系统的特征多项式稳定性。
 * 对系统的每个输入-输出通道的特征多项式分别应用劳斯-赫尔维茨判据。
 * 
 * @param characteristic_polys 多个特征多项式数组（每个多项式连续存储）
 * @param poly_count 多项式数量
 * @param poly_order 每个多项式的阶数
 * @param results 稳定性结果数组（poly_count个）
 * @return int 成功返回0，失败返回错误码
 */
int laplace_mimo_routh_hurwitz(const float* characteristic_polys,
                                size_t poly_count, size_t poly_order,
                                int* results);

/* ========== S05-自适应PID设计器 ========== */

/**
 * @brief PID控制器配置（拉普拉斯域自适应PID设计）
 */
typedef struct {
    float kp;                /**< 比例增益 */
    float ki;                /**< 积分增益 */
    float kd;                /**< 微分增益 */
    float kp_min, kp_max;    /**< 比例增益范围 */
    float ki_min, ki_max;    /**< 积分增益范围 */
    float kd_min, kd_max;    /**< 微分增益范围 */
    float target_overshoot;  /**< 目标超调量 (0-1) */
    float target_settling;   /**< 目标稳定时间 (秒) */
    float target_bandwidth;  /**< 目标带宽 (Hz) */
    int use_pole_placement;  /**< 是否使用极点配置法 */
    int use_ziegler_nichols; /**< 是否使用Ziegler-Nichols整定法 */
} LaplacePIDConfig;

/**
 * @brief PID设计结果
 */
typedef struct {
    float kp;                /**< 设计出的比例增益 */
    float ki;                /**< 设计出的积分增益 */
    float kd;                /**< 设计出的微分增益 */
    float predicted_overshoot; /**< 预测超调量 */
    float predicted_settling;  /**< 预测稳定时间 */
    float gain_margin;       /**< 增益裕度 (dB) */
    float phase_margin;      /**< 相位裕度 (度) */
    int is_stable;           /**< 闭环是否稳定 */
} PIDResult;

/**
 * @brief 自适应PID控制器设计
 * 
 * 基于系统的拉普拉斯域传递函数模型，自动设计PID控制器参数。
 * 支持极点配置法和Ziegler-Nichols整定法。
 * 
 * @param numerator 被控对象传递函数分子系数
 * @param denominator 被控对象传递函数分母系数
 * @param num_order 分子阶数
 * @param den_order 分母阶数
 * @param config PID设计配置
 * @param result PID设计结果
 * @return int 成功返回0，失败返回错误码
 */
int laplace_design_pid(const float* numerator, const float* denominator,
                        size_t num_order, size_t den_order,
                        const LaplacePIDConfig* config, PIDResult* result);

/**
 * @brief 快速稳定性检查（仅劳斯-赫尔维茨，无极点求解）
 *
 * 轻量级替代 laplace_analyze_system()，仅使用劳斯-赫尔维茨判据判断系统稳定性。
 * 不需要 LaplaceAnalyzer 实例，O(n²) 复杂度，不分配动态内存，适合实时路径调用。
 *
 * @param denominator 分母多项式系数（从常数项到最高次项）
 * @param den_order 分母阶数（系数数组长度）
 * @param is_stable 输出：是否稳定
 * @param stability_margin 输出：稳定裕度（0-1），不需要可传 NULL
 * @return int 成功返回0，失败返回错误码
 */
int laplace_check_stability_fast(const float* denominator, size_t den_order,
                                  int* is_stable, float* stability_margin);

/* 创建默认拉普拉斯分析器 */
void* lnn_laplace_create_default_analyzer(void);

/* 获取拉普拉斯频谱 */
int laplace_unified_get_spectrum(void* analyzer, float* spectrum, size_t size);

/* H-001修复: 获取缓存的传递函数系数（只读访问）
 * 供 laplace_unified_get_spectrum 使用，避免硬编码 1/(s+1)
 * 返回内部缓存指针（只读），调用者不得修改或释放
 * 返回值: 0 成功，-1 缓存为空或参数无效 */
int laplace_analyzer_get_cached_transfer_fn(const LaplaceAnalyzer* analyzer,
                                             const float** numerator,
                                             const float** denominator,
                                             size_t* num_order,
                                             size_t* den_order);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_LAPLACE_H */