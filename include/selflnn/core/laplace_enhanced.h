/**
 * @file laplace_enhanced.h
 * @brief 拉普拉斯变换全面增强系统接口
 */

#ifndef SELFLNN_LAPLACE_ENHANCED_H
#define SELFLNN_LAPLACE_ENHANCED_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 拉普拉斯算子类型 */
typedef enum {
    LAPLACE_OP_STANDARD = 0,      /* 标准拉普拉斯算子 */
    LAPLACE_OP_GAUSSIAN = 1,      /* 高斯-拉普拉斯 (LoG) */
    LAPLACE_OP_FRACTIONAL = 2,    /* 分数阶拉普拉斯 */
    LAPLACE_OP_ANISOTROPIC = 3,   /* 各向异性拉普拉斯 */
    LAPLACE_OP_SPECTRAL = 4       /* 谱拉普拉斯 */
} LaplaceOperatorType;

/* 增强目标模块 */
typedef enum {
    LAPLACE_TARGET_TRAINING = 0,  /* 训练模块 */
    LAPLACE_TARGET_REASONING = 1, /* 推理模块 */
    LAPLACE_TARGET_VISION = 2,    /* 视觉模块 */
    LAPLACE_TARGET_AUDIO = 3,     /* 音频模块 */
    LAPLACE_TARGET_CONTROL = 4,   /* 控制模块 */
    LAPLACE_TARGET_MEMORY = 5,    /* 记忆模块 */
    LAPLACE_TARGET_ALL = 6        /* 所有模块 */
} LaplaceTarget;

/* 滤波配置 */
typedef struct {
    LaplaceOperatorType op_type;
    float cutoff_frequency;        /* 截止频率 */
    float filter_order;            /* 滤波器阶数 */
    float smoothing_alpha;         /* 平滑系数 */
    float noise_suppression;       /* 噪声抑制强度 */
    int use_adaptive_cutoff;       /* 自适应截止频率 */
    int use_zero_phase;            /* 零相位滤波 */
    int kernel_size;               /* 卷积核大小 */
} LaplaceFilterConfig;

/* 频域特征 */
typedef struct {
    float* magnitude_spectrum;     /* 幅度谱 */
    float* phase_spectrum;         /* 相位谱 */
    size_t spectrum_size;          /* 频谱大小 */
    float dominant_frequency;      /* 主频 */
    float spectral_entropy;        /* 谱熵 */
    float spectral_centroid;       /* 谱质心 */
    float spectral_bandwidth;      /* 谱带宽 */
    float spectral_flatness;       /* 谱平坦度 */
    float high_freq_energy_ratio;  /* 高频能量比 */
    float stability_metric;        /* 稳定性度量 */
} LaplaceSpectralFeatures;

/* 系统稳定性分析 */
typedef struct {
    float damping_ratio;           /* 阻尼比 */
    float natural_frequency;       /* 自然频率 */
    float gain_margin;             /* 增益裕度 */
    float phase_margin;            /* 相位裕度 */
    float settling_time;           /* 稳定时间 */
    float overshoot;               /* 超调量 */
    int is_stable;                 /* 是否稳定 */
    float stability_reserve;       /* 稳定性储备 */
} EnhancedStabilityAnalysis;

/* 拉普拉斯增强系统句柄 */
typedef struct LaplaceEnhancedSystem LaplaceEnhancedSystem;

/**
 * @brief 创建拉普拉斯增强系统
 */
LaplaceEnhancedSystem* laplace_enhanced_create(LaplaceTarget target);

/**
 * @brief 释放拉普拉斯增强系统
 */
void laplace_enhanced_free(LaplaceEnhancedSystem* system);

/**
 * @brief 配置滤波器参数
 */
int laplace_set_filter_config(LaplaceEnhancedSystem* system, const LaplaceFilterConfig* config);

/**
 * @brief 频域分析
 */
int laplace_spectral_analyze(LaplaceEnhancedSystem* system, const float* signal, 
                            size_t signal_size, LaplaceSpectralFeatures* features);

/**
 * @brief 频域滤波
 */
int laplace_spectral_filter(LaplaceEnhancedSystem* system, const float* input,
                           float* output, size_t signal_size);

/**
 * @brief 梯度频域增强（训练用）
 */
int laplace_enhance_gradients(LaplaceEnhancedSystem* system, float* gradients,
                             size_t grad_size, float learning_rate);

/**
 * @brief 信号去噪
 */
int laplace_denoise(LaplaceEnhancedSystem* system, const float* input, 
                   float* output, size_t signal_size);

/**
 * @brief 稳定性分析
 */
int laplace_stability_analyze(LaplaceEnhancedSystem* system, const float* system_matrix,
                             int matrix_size, EnhancedStabilityAnalysis* analysis);

/**
 * @brief 控制稳定性增强
 */
int laplace_stabilize_control(LaplaceEnhancedSystem* system, const float* control_signal,
                             float* stabilized_signal, size_t signal_size);

/**
 * @brief 图像拉普拉斯增强（视觉用）
 */
int laplace_image_enhance(LaplaceEnhancedSystem* system, const float* image,
                         float* enhanced, int width, int height, int channels);

/**
 * @brief 记忆衰减建模（记忆用）
 */
int laplace_memory_decay_model(LaplaceEnhancedSystem* system, float* memory_strength,
                              size_t memory_count, float time_elapsed);

/* ========== S06: 全系统拉普拉斯分析管道 ========== */

/**
 * @brief 分析管道阶段枚举
 */
typedef enum {
    PIPELINE_STAGE_FFT,            /**< FFT频谱分析 */
    PIPELINE_STAGE_SPECTRAL,       /**< 频谱特征提取 */
    PIPELINE_STAGE_STABILITY,      /**< 稳定性分析 */
    PIPELINE_STAGE_FILTERING,      /**< 滤波增强 */
    PIPELINE_STAGE_CONTROL,        /**< 控制输出生成 */
    PIPELINE_STAGE_COUNT           /**< 阶段总数 */
} PipelineStage;

/**
 * @brief 分析管道阶段状态
 */
typedef struct {
    PipelineStage stage;       /**< 当前阶段 */
    int success;               /**< 该阶段是否成功 */
    float execution_time_ms;   /**< 执行时间(毫秒) */
    float stage_metrics[10];   /**< 阶段输出指标数组 */
    int metric_count;          /**< 指标数量 */
} PipelineStageResult;

/**
 * @brief 全系统分析管道结果
 */
typedef struct {
    LaplaceSpectralFeatures spectral;    /**< 频谱分析结果 */
    EnhancedStabilityAnalysis stability;         /**< 稳定性分析结果 */
    float control_recommendation;        /**< 控制推荐值(-1到1) */
    float overall_health_score;          /**< 系统健康度(0-1) */
    int alarm_level;                     /**< 告警级别(0=正常,1=警告,2=严重) */
    PipelineStageResult stage_results[PIPELINE_STAGE_COUNT]; /**< 各阶段详情 */
    int stage_count;                     /**< 实际执行阶段数 */
} PipelineResult;

/**
 * @brief 执行全系统拉普拉斯分析管道
 * 
 * 将时间域信号通过完整的分析管道：
 * 信号→FFT→频谱特征提取→稳定性分析→滤波增强→控制推荐
 * 
 * @param system 拉普拉斯增强系统句柄
 * @param time_domain_signal 时域输入信号
 * @param signal_size 信号长度（须为2的幂）
 * @param sampling_rate 采样率(Hz)
 * @param result 管道分析结果输出
 * @return int 成功返回0，失败返回错误码
 */
int laplace_pipeline_execute(LaplaceEnhancedSystem* system,
                              const float* time_domain_signal,
                              size_t signal_size, float sampling_rate,
                              PipelineResult* result);

/* ========== S06: 实时稳定性监控 ========== */

/**
 * @brief 实时稳定性监控器
 */
typedef struct {
    float* history_buffer;         /**< 历史信号环形缓冲区 */
    size_t buffer_size;            /**< 缓冲区大小 */
    size_t write_pos;              /**< 写入位置 */
    float* spectrum_accumulator;   /**< 频谱累加器（滑动平均） */
    size_t accumulation_count;     /**< 累加次数 */
    float stability_threshold;     /**< 稳定性阈值 */
    float anomaly_threshold;       /**< 异常检测阈值 */
    int consecutive_anomalies;     /**< 连续异常计数 */
    float* feature_history;        /**< 特征历史记录 */
    size_t feature_history_size;   /**< 特征历史长度 */
    size_t feature_write_pos;      /**< 特征历史写入位置 */
    int is_initialized;            /**< 是否已初始化 */
} RealtimeMonitor;

/**
 * @brief 创建实时稳定性监控器
 * 
 * @param buffer_size 环形缓冲区大小（须为2的幂）
 * @param feature_history_size 特征历史记录长度
 * @return RealtimeMonitor* 监控器句柄，失败返回NULL
 */
RealtimeMonitor* laplace_monitor_create(size_t buffer_size, 
                                         size_t feature_history_size);

/**
 * @brief 销毁实时稳定性监控器
 */
void laplace_monitor_free(RealtimeMonitor* monitor);

/**
 * @brief 更新监控状态（推入新数据点）
 * 
 * @param monitor 监控器句柄
 * @param new_sample 新的时域样本点
 * @param result 输出：当前稳定性结果（可为NULL）
 * @return int 0=正常, 1=警告, 2=异常
 */
int laplace_monitor_update(RealtimeMonitor* monitor, float new_sample,
                            EnhancedStabilityAnalysis* result);

/**
 * @brief 获取监控器统计信息
 * 
 * @param monitor 监控器句柄
 * @param mean 输出：均值
 * @param variance 输出：方差
 * @param peak_frequency 输出：主频
 * @return int 成功返回0
 */
int laplace_monitor_get_statistics(RealtimeMonitor* monitor,
                                    float* mean, float* variance,
                                    float* peak_frequency);

/* ========== S06: LNN稳定性保证器 ========== */

/**
 * @brief LNN稳定性保证器配置
 */
typedef struct {
    float max_eigenvalue_real;       /**< 最大允许特征值实部 */
    float correction_strength;       /**< 修正强度 (0-1) */
    float damping_factor;            /**< 阻尼因子 */
    int enable_automatic_correction; /**< 是否启用自动修正 */
    int enable_adaptive_damping;     /**< 是否启用自适应阻尼 */
    int max_correction_iterations;   /**< 最大修正迭代次数 */
} LNNStabilityConfig;

/**
 * @brief LNN稳定性保证器
 */
typedef struct {
    LaplaceEnhancedSystem* parent_system;    /**< 父系统 */
    LNNStabilityConfig config;               /**< 配置 */
    float* eigenvalue_real_parts;            /**< 特征值实部阵列 */
    float* eigenvalue_imag_parts;            /**< 特征值虚部阵列 */
    size_t eigenvalue_count;                 /**< 特征值数量 */
    float max_real_part;                     /**< 最大实部 */
    int correction_count;                    /**< 修正次数 */
    float* correction_history;               /**< 修正历史 */
    size_t correction_history_size;          /**< 修正历史大小 */
    size_t correction_write_pos;             /**< 修正历史写入位置 */
} LNNStabilityGuarantor;

/**
 * @brief 创建LNN稳定性保证器
 * 
 * @param parent 拉普拉斯增强系统句柄
 * @param config 稳定性保证器配置
 * @param eigenvalue_history_size 修正历史大小
 * @return LNNStabilityGuarantor* 保证器句柄，失败返回NULL
 */
LNNStabilityGuarantor* laplace_guarantor_create(LaplaceEnhancedSystem* parent,
                                                  const LNNStabilityConfig* config,
                                                  size_t eigenvalue_history_size);

/**
 * @brief 销毁LNN稳定性保证器
 */
void laplace_guarantor_free(LNNStabilityGuarantor* guarantor);

/**
 * @brief 检查LNN稳定性并自动修正
 * 
 * 分析当前LNN状态矩阵的特征值分布，如果任何特征值的实部超过阈值，
 * 则施加阻尼修正将极点拉回左半平面。
 * 
 * @param guarantor 稳定性保证器句柄
 * @param state_matrix LNN状态矩阵（一维平铺）
 * @param matrix_size 矩阵尺寸（行数/列数）
 * @param correction_applied 输出：是否施加了修正
 * @return int 成功返回0，失败返回错误码。修正后系统保证稳定。
 */
int laplace_guarantor_check_and_correct(LNNStabilityGuarantor* guarantor,
                                          const float* state_matrix,
                                          size_t matrix_size,
                                          int* correction_applied);

/**
 * @brief 获取稳定性保证器状态报告
 * 
 * @param guarantor 稳定性保证器句柄
 * @param max_eigenvalue 输出：最大特征值实部
 * @param eigenvalue_spread 输出：特征值分布范围
 * @param total_corrections 输出：总修正次数
 * @param current_stability_margin 输出：当前稳定裕度
 * @return int 成功返回0
 */
int laplace_guarantor_get_report(LNNStabilityGuarantor* guarantor,
                                  float* max_eigenvalue,
                                  float* eigenvalue_spread,
                                  int* total_corrections,
                                  float* current_stability_margin);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_LAPLACE_ENHANCED_H */
