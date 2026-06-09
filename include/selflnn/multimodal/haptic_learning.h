/**
 * @file haptic_learning.h
 * @brief 触觉/力觉感官学习系统接口
 */

#ifndef SELFLNN_HAPTIC_LEARNING_H
#define SELFLNN_HAPTIC_LEARNING_H

#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HL_MAX_FEATURES 256
#define HL_MAX_MATERIALS 64
#define HL_MAX_GRASP_CONFIGS 32

typedef struct {
    float pressure[16];
    float temperature[4];
    float vibration[8];
    float force[6];
    float torque[6];
    time_t timestamp;
    int sensor_count;
} HapticReading;

typedef struct {
    float flatness;
    float roughness;
    float hardness;
    float friction;
    float thermal_conductivity;
    float compliance;
    float density;
    float damping_ratio;
    char material_name[64];
    float confidence;
} MaterialProperty;

typedef struct {
    int grasp_id;
    float finger_positions[5][3];
    float contact_forces[5];
    float grasp_quality;
    int success_count;
    int attempt_count;
    float success_rate;
    char object_type[64];
} GraspConfiguration;

typedef struct HapticLearner HapticLearner;

HapticLearner* haptic_learner_create(void);
void haptic_learner_free(HapticLearner* hl);

/* 触觉特征提取 */
int hl_extract_features(HapticLearner* hl, const HapticReading* reading, float* features, int max_dim);

/* 材料识别 */
int hl_classify_material(HapticLearner* hl, const float* features, int dim, MaterialProperty* out);
int hl_learn_material(HapticLearner* hl, const float* features, int dim, const char* material_name);
int hl_list_materials(const HapticLearner* hl, MaterialProperty* out, int max_count);

/* 力觉反馈学习 */
int hl_learn_force_profile(HapticLearner* hl, const float* force_trajectory, int steps, const char* label);
int hl_predict_force(HapticLearner* hl, const float* current_state, float* predicted_force, int dim);

/* 抓取学习 */
int hl_learn_grasp(HapticLearner* hl, const float* finger_positions, const float* contact_forces, int fingers, const char* object_type, int success);
int hl_get_best_grasp(const HapticLearner* hl, const char* object_type, GraspConfiguration* out);
int hl_list_grasps(const HapticLearner* hl, const char* object_type, GraspConfiguration* out, int max_count);

/* 联合表征 */
int hl_fuse_vision_haptic(HapticLearner* hl, const float* visual_features, int vdim, const float* haptic_features, int hdim, float* fused, int fdim);

/* LNN集成（闭环触觉→LNN参数更新） */
int haptic_learner_set_lnn(HapticLearner* hl, void* lnn_instance);
int haptic_learner_update_lnn(HapticLearner* hl, const float* teaching_signal, int signal_dim, float learning_rate);

/* 传感器反馈闭环 */
int hl_feedback_measured_force(HapticLearner* hl, const float* measured_force, int dim);

#ifdef __cplusplus
}
#endif

// ==================== CfC触觉信号深度处理器 ====================

/**
 * @brief CfC触觉处理器配置
 */
typedef struct {
    int signal_dim;                  /**< 触觉信号维度（默认16通道） */
    int cfc_hidden_size;             /**< CfC隐藏状态大小（默认64） */
    float cfc_time_constant;         /**< CfC时间常数（默认0.1） */
    float cfc_noise_std;             /**< CfC噪声标准差（默认0.01） */
    int enable_adaptive_timestep;    /**< 启用自适应时间步长（默认1） */
    float min_timestep;              /**< 最小时间步长（秒，默认0.001） */
    float max_timestep;              /**< 最大时间步长（秒，默认0.1） */
    int enable_signal_filtering;     /**< 启用信号预滤波（默认1） */
    float filter_cutoff_freq;        /**< 滤波器截止频率（Hz，默认100） */
    int enable_contact_detection;    /**< 启用接触检测（默认1） */
    float contact_threshold;         /**< 接触检测阈值（默认0.05） */
    int enable_slip_detection;       /**< 启用滑移检测（默认1） */
    float slip_detection_threshold;  /**< 滑移检测阈值（默认0.02） */
} HapticCfcConfig;

/**
 * @brief CfC触觉处理器句柄（不透明）
 *
 * 使用CfC ODE对触觉信号进行连续时间动态演化：
 * dh/dt = CfC_ODE(h, x(t), θ)
 * 其中h为隐藏状态，x(t)为触觉传感器信号，θ为可学习参数
 * 输出：演化后的触觉特征向量 f = W*tanh(h) + b
 */
typedef struct HapticCfcProcessor HapticCfcProcessor;

/**
 * @brief 获取默认CfC触觉处理器配置
 *
 * @return HapticCfcConfig 默认配置
 */
HapticCfcConfig haptic_cfc_get_default_config(void);

/**
 * @brief 创建CfC触觉处理器
 *
 * @param config CfC触觉处理器配置
 * @return HapticCfcProcessor* 句柄，失败返回NULL
 */
HapticCfcProcessor* haptic_cfc_create(const HapticCfcConfig* config);

/**
 * @brief 释放CfC触觉处理器
 *
 * @param proc CfC触觉处理器句柄
 */
void haptic_cfc_free(HapticCfcProcessor* proc);

/**
 * @brief 处理触觉信号通过CfC ODE演化
 *
 * 核心算法：
 * 1. 对输入触觉信号进行预滤波（低通滤波去除高频噪声）
 * 2. 检测接触事件（信号幅度超过阈值）
 * 3. 检测滑移事件（信号高频分量变化）
 * 4. 将触觉信号x(t)输入CfC ODE：dh/dt = σ(W_h*h + W_x*x + b)
 * 5. ODE积分步进（RK4或自适应步长）
 * 6. 输出演化后的触觉特征向量
 *
 * @param proc CfC触觉处理器句柄
 * @param reading 触觉传感器读数
 * @param dt 时间步长（秒）
 * @param features_out 输出触觉特征向量（预分配cfc_hidden_size个float）
 * @param feature_dim 输出特征维度
 * @param contact_detected 输出是否检测到接触（可为NULL）
 * @param slip_detected 输出是否检测到滑移（可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int haptic_cfc_process(HapticCfcProcessor* proc,
                       const HapticReading* reading,
                       float dt,
                       float* features_out, int feature_dim,
                       int* contact_detected,
                       int* slip_detected);

/**
 * @brief 重置CfC触觉处理器内部状态
 *
 * @param proc CfC触觉处理器句柄
 */
void haptic_cfc_reset(HapticCfcProcessor* proc);

/**
 * @brief 获取CfC触觉处理器当前内部状态
 *
 * @param proc CfC触觉处理器句柄
 * @param hidden_state 输出隐藏状态（预分配cfc_hidden_size个float）
 * @param state_dim 状态维度
 * @return int 成功返回0，失败返回-1
 */
int haptic_cfc_get_state(const HapticCfcProcessor* proc,
                         float* hidden_state, int state_dim);

// ==================== 触觉纹理深度分析 ====================

/**
 * @brief 触觉纹理特征结构体
 */
typedef struct {
    float spatial_frequency;         /**< 空间频率（Hz） */
    float roughness_index;           /**< 粗糙度指数（0-1） */
    float hardness_index;            /**< 硬度指数（0-1） */
    float friction_coefficient;      /**< 摩擦系数 */
    float thermal_diffusivity;       /**< 热扩散率（m²/s） */
    float compliance_index;          /**< 柔顺性指数（0-1） */
    float periodicity;               /**< 周期性（m） */
    float amplitude_variance;        /**< 幅度方差 */
    float feature_vector[64];        /**< 纹理特征向量 */
    int feature_count;               /**< 有效特征数 */
    char texture_class[64];          /**< 纹理类别名称 */
    float classification_confidence; /**< 分类置信度（0-1） */
} HapticTextureDescriptor;

/**
 * @brief 触觉纹理分析配置
 */
typedef struct {
    int fft_window_size;             /**< FFT窗口大小（默认256） */
    int fft_overlap;                 /**< FFT重叠样本数（默认128） */
    float frequency_resolution;      /**< 频率分辨率（Hz，默认0.5） */
    int enable_wavelet_analysis;     /**< 启用小波分析（默认1） */
    int wavelet_decomposition_level; /**< 小波分解层数（默认4） */
    int enable_statistical_features; /**< 启用统计特征（默认1） */
    int enable_fractal_analysis;     /**< 启用分形分析（默认1） */
    float fractal_box_size_min;      /**< 分形盒计数最小尺寸（默认2） */
    float fractal_box_size_max;      /**< 分形盒计数最大尺寸（默认64） */
    int num_texture_classes;         /**< 预定义纹理类别数（默认20） */
    int cfc_hidden_size;             /**< CfC纹理特征演化维度（默认64） */
} HapticTextureConfig;

/**
 * @brief 触觉纹理分析器句柄（不透明）
 *
 * 深度纹理分析引擎：
 * 1. 时频分析（FFT + 小波变换）
 * 2. 统计特征提取（均值、方差、偏度、峰度、自相关）
 * 3. 分形维度计算（盒计数法）
 * 4. CfC ODE纹理特征连续演化
 * 5. 基于特征向量的纹理分类
 */
typedef struct HapticTextureAnalyzer HapticTextureAnalyzer;

/**
 * @brief 获取默认触觉纹理分析配置
 *
 * @return HapticTextureConfig 默认配置
 */
HapticTextureConfig haptic_texture_get_default_config(void);

/**
 * @brief 创建触觉纹理分析器
 *
 * @param config 纹理分析配置
 * @return HapticTextureAnalyzer* 句柄，失败返回NULL
 */
HapticTextureAnalyzer* haptic_texture_create(const HapticTextureConfig* config);

/**
 * @brief 释放触觉纹理分析器
 *
 * @param ta 纹理分析器句柄
 */
void haptic_texture_free(HapticTextureAnalyzer* ta);

/**
 * @brief 分析触觉信号提取纹理描述
 *
 * 核心算法：
 * 1. 对振动信号进行FFT分析，提取频谱特征
 *    X(k) = Σ x(n)*exp(-j*2π*n*k/N)
 *    计算频谱峰值频率、能量分布、频谱质心
 * 2. 进行小波包分解，提取多尺度时频特征
 *    W(j,k) = Σ x(n)*ψ_j,k(n)，ψ为小波基函数
 * 3. 提取统计特征：均值、方差、偏度、峰度、自相关系数
 * 4. 计算分形维度（盒计数法）：
 *    D = lim(log(N(ε))/log(1/ε))，ε→0
 *    N(ε)为覆盖信号所需边长为ε的盒子数量
 * 5. 通过CfC ODE演化纹理特征向量
 * 6. 特征匹配分类（余弦相似度或最近邻）
 *
 * @param ta 纹理分析器句柄
 * @param reading 触觉传感器读数（振动通道用于纹理分析）
 * @param dt 采样时间间隔（秒）
 * @param descriptor 输出纹理描述符
 * @return int 成功返回0，失败返回-1
 */
int haptic_texture_analyze(HapticTextureAnalyzer* ta,
                           const HapticReading* reading,
                           float dt,
                           HapticTextureDescriptor* descriptor);

/**
 * @brief 学习新的纹理类别
 *
 * @param ta 纹理分析器句柄
 * @param descriptor 纹理描述符
 * @param texture_name 纹理类别名称
 * @return int 成功返回0，失败返回-1
 */
int haptic_texture_learn(HapticTextureAnalyzer* ta,
                         const HapticTextureDescriptor* descriptor,
                         const char* texture_name);

/**
 * @brief 查询已知纹理类别列表
 *
 * @param ta 纹理分析器句柄
 * @param descriptors 输出纹理描述符数组
 * @param max_count 最大数量
 * @return int 成功返回类别数，失败返回-1
 */
int haptic_texture_list(const HapticTextureAnalyzer* ta,
                        HapticTextureDescriptor* descriptors,
                        int max_count);

/**
 * @brief 重置纹理分析器
 *
 * @param ta 纹理分析器句柄
 */
void haptic_texture_reset(HapticTextureAnalyzer* ta);

// ==================== CfC增强物体识别 ====================

/**
 * @brief CfC触觉物体识别配置
 */
typedef struct {
    int cfc_hidden_size;             /**< CfC隐藏状态大小（默认64） */
    int num_known_objects;           /**< 已知物体最大数量（默认100） */
    float recognition_threshold;     /**< 识别置信度阈值（默认0.6） */
    int enable_multi_grasp_fusion;   /**< 启用多抓取融合（默认1） */
    int max_grasps_per_object;       /**< 每物体最大抓取数（默认10） */
    int enable_continuous_learning;  /**< 启用持续学习（默认1） */
    float learning_rate;             /**< 学习率（默认0.01） */
    int num_material_features;       /**< 材料特征维度（默认32） */
} HapticObjectRecognitionConfig;

/**
 * @brief CfC触觉物体识别器句柄（不透明）
 *
 * 基于CfC ODE的触觉物体识别引擎：
 * 1. 融合多角度触觉信号通过CfC ODE演化
 * 2. 材料属性特征提取
 * 3. 与已知物体数据库匹配（余弦相似度）
 * 4. 持续学习更新物体表征
 */
typedef struct HapticObjectRecognizer HapticObjectRecognizer;

/**
 * @brief 获取默认触觉物体识别配置
 *
 * @return HapticObjectRecognitionConfig 默认配置
 */
HapticObjectRecognitionConfig haptic_object_recognition_get_default_config(void);

/**
 * @brief 创建触觉物体识别器
 *
 * @param config 物体识别配置
 * @return HapticObjectRecognizer* 句柄，失败返回NULL
 */
HapticObjectRecognizer* haptic_object_recognizer_create(const HapticObjectRecognitionConfig* config);

/**
 * @brief 释放触觉物体识别器
 *
 * @param rec 物体识别器句柄
 */
void haptic_object_recognizer_free(HapticObjectRecognizer* rec);

/**
 * @brief 通过触觉信号识别物体
 *
 * 核心算法：
 * 1. 触觉信号通过CfC ODE演化：dh/dt = f(h, x(t), θ)
 * 2. 提取CfC隐藏状态作为物体嵌入向量
 * 3. 计算与已知物体嵌入的余弦相似度：
 *    sim(o_i, o_j) = e_i · e_j / (||e_i|| * ||e_j||)
 * 4. 选择相似度最高的类别（超过阈值）
 * 5. 更新物体置信度
 *
 * @param rec 物体识别器句柄
 * @param readings 触觉读数数组（可多次抓取）
 * @param num_readings 读数数量
 * @param dt 时间步长数组（每个读数对应）
 * @param object_name 输出识别物体名称
 * @param max_name_len 名称缓冲区最大长度
 * @param confidence 输出识别置信度
 * @return int 成功返回0，识别失败（低于阈值）返回1，错误返回-1
 */
int haptic_object_recognize(HapticObjectRecognizer* rec,
                            const HapticReading* readings, int num_readings,
                            const float* dt,
                            char* object_name, int max_name_len,
                            float* confidence);

/**
 * @brief 学习新的物体触觉表征
 *
 * @param rec 物体识别器句柄
 * @param readings 触觉读数数组
 * @param num_readings 读数数量
 * @param dt 时间步长数组
 * @param object_name 物体名称
 * @return int 成功返回0，失败返回-1
 */
int haptic_object_learn(HapticObjectRecognizer* rec,
                        const HapticReading* readings, int num_readings,
                        const float* dt,
                        const char* object_name);

/**
 * @brief 重置物体识别器
 *
 * @param rec 物体识别器句柄
 */
void haptic_object_recognizer_reset(HapticObjectRecognizer* rec);

// ==================== CfC增强抓取学习 ====================

/**
 * @brief CfC抓取学习配置
 */
typedef struct {
    int cfc_hidden_size;             /**< CfC隐藏状态大小（默认64） */
    int num_fingers;                 /**< 手指数量（默认5） */
    float force_control_gain;        /**< 力控制增益（默认0.8） */
    float position_control_gain;     /**< 位置控制增益（默认0.6） */
    int enable_slip_compensation;    /**< 启用滑移补偿（默认1） */
    float slip_compensation_gain;    /**< 滑移补偿增益（默认0.3） */
    int enable_adaptive_grip;        /**< 启用自适应抓取力（默认1） */
    float min_grip_force;            /**< 最小抓取力（N，默认0.5） */
    float max_grip_force;            /**< 最大抓取力（N，默认20.0） */
    int enable_learning_from_demo;   /**< 启用人机示教学习（默认1） */
    float imitation_learning_rate;   /**< 模仿学习率（默认0.05） */
} HapticGraspLearningConfig;

/**
 * @brief CfC抓取学习器句柄（不透明）
 *
 * 基于CfC ODE的抓取学习引擎：
 * 1. 通过CfC ODE学习抓取力/位置动态映射
 * 2. 滑移检测与实时补偿
 * 3. 自适应抓取力调节
 * 4. 从示教中模仿学习
 */
typedef struct HapticGraspLearner HapticGraspLearner;

/**
 * @brief 获取默认抓取学习配置
 *
 * @return HapticGraspLearningConfig 默认配置
 */
HapticGraspLearningConfig haptic_grasp_learning_get_default_config(void);

/**
 * @brief 创建抓取学习器
 *
 * @param config 抓取学习配置
 * @return HapticGraspLearner* 句柄，失败返回NULL
 */
HapticGraspLearner* haptic_grasp_learner_create(const HapticGraspLearningConfig* config);

/**
 * @brief 释放抓取学习器
 *
 * @param gl 抓取学习器句柄
 */
void haptic_grasp_learner_free(HapticGraspLearner* gl);

/**
 * @brief 执行抓取控制（CfC ODE预测）
 *
 * 核心算法：
 * 1. 触觉反馈信号通过CfC ODE演化：
 *    dh/dt = σ(W_h*h + W_f*f_contact + W_p*p_finger + b)
 * 2. 从CfC隐藏状态解码控制信号：
 *    Δf = W_out * tanh(h) + b_out
 * 3. 检测滑移（振动信号高频分量变化率）
 * 4. 滑移补偿：Δf_comp = K_slip * vibration_energy
 * 5. 自适应力调节：f_target = min(max(f_base + Δf, f_min), f_max)
 *
 * @param gl 抓取学习器句柄
 * @param reading 当前触觉读数
 * @param finger_positions 当前手指位置 [num_fingers][3]
 * @param contact_forces 当前接触力 [num_fingers]
 * @param dt 时间步长
 * @param target_forces 输出目标接触力 [num_fingers]
 * @param target_positions 输出目标手指位置 [num_fingers][3]
 * @return int 成功返回0，失败返回-1
 */
int haptic_grasp_control(HapticGraspLearner* gl,
                         const HapticReading* reading,
                         const float* finger_positions,
                         const float* contact_forces,
                         float dt,
                         float* target_forces,
                         float* target_positions);

/**
 * @brief 从示教轨迹学习抓取
 *
 * 核心算法：
 * 1. 记录示教的力/位置轨迹
 * 2. 计算示教的抓取动态：d = (f_demo, p_demo)
 * 3. CfC ODE拟合示教轨迹：min ||CfC(h, x) - d||²
 * 4. 更新CfC权重：θ = θ - α*∇L(θ)
 *
 * @param gl 抓取学习器句柄
 * @param readings 示教读数数组
 * @param finger_positions 示教手指位置数组
 * @param contact_forces 示教接触力数组
 * @param num_steps 示教步数
 * @param dt 时间步长
 * @param grasp_quality 输出抓取质量评分（0-1）
 * @return int 成功返回0，失败返回-1
 */
int haptic_grasp_learn_from_demo(HapticGraspLearner* gl,
                                 const HapticReading* readings,
                                 const float* finger_positions,
                                 const float* contact_forces,
                                 int num_steps, float dt,
                                 float* grasp_quality);

/**
 * @brief 评估当前抓取质量
 *
 * @param gl 抓取学习器句柄
 * @param reading 当前触觉读数
 * @param contact_forces 当前接触力
 * @param num_fingers 手指数
 * @return float 抓取质量评分（0-1），-1表示失败
 */
float haptic_grasp_evaluate(HapticGraspLearner* gl,
                            const HapticReading* reading,
                            const float* contact_forces,
                            int num_fingers);

/**
 * @brief 重置抓取学习器
 *
 * @param gl 抓取学习器句柄
 */
void haptic_grasp_learner_reset(HapticGraspLearner* gl);

// ==================== 视触联合融合 ====================

/**
 * @brief 视触融合配置
 */
typedef struct {
    int visual_feature_dim;          /**< 视觉特征维度 */
    int haptic_feature_dim;          /**< 触觉特征维度 */
    int fused_feature_dim;           /**< 融合特征维度 */
    int cfc_hidden_size;             /**< CfC融合隐藏状态大小 */
    int enable_cfc_fusion;           /**< 启用CfC动态融合（Phase2: 默认0，启用投影拼接替代独立ODE） */
    float fusion_temporal_constant;  /**< 融合时间常数（默认0.05） */
} VisionHapticFusionConfig;

/**
 * @brief 视触融合器句柄（不透明）
 *
 * 多模态CfC融合：
 * f_fused = CfC_ODE(f_vis, f_hap, θ)
 * 通过CfC ODE自然融合视觉和触觉特征
 */
typedef struct VisionHapticFusion VisionHapticFusion;

/**
 * @brief 创建视触融合器
 *
 * @param config 融合配置
 * @return VisionHapticFusion* 句柄，失败返回NULL
 */
VisionHapticFusion* vision_haptic_fusion_create(const VisionHapticFusionConfig* config);

/**
 * @brief 释放视触融合器
 *
 * @param vf 融合器句柄
 */
void vision_haptic_fusion_free(VisionHapticFusion* vf);

/**
 * @brief 融合视觉和触觉特征（CfC ODE）
 *
 * 核心算法：
 * 1. 视觉特征f_vis和触觉特征f_hap拼接为联合输入
 * 2. 通过CfC ODE在连续时间中演化：
 *    dh/dt = σ(W_h*h + W_v*f_vis + W_hap*f_hap + b)
 * 3. 输出融合特征：f_fused = W_o*tanh(h) + b_o
 *
 * @param vf 融合器句柄
 * @param visual_features 视觉特征向量
 * @param haptic_features 触觉特征向量（可由haptic_cfc_process生成）
 * @param fused_features 输出融合特征向量
 * @param dt 时间步长
 * @return int 成功返回0，失败返回-1
 */
int vision_haptic_fusion_fuse(VisionHapticFusion* vf,
                              const float* visual_features,
                              const float* haptic_features,
                              float* fused_features,
                              float dt);

/**
 * @brief 重置融合器
 *
 * @param vf 融合器句柄
 */
void vision_haptic_fusion_reset(VisionHapticFusion* vf);

#endif // SELFLNN_HAPTIC_LEARNING_H
