/**
 * @file sensor.h
 * @brief 传感器处理模块接口
 * 
 * 传感器数据处理接口，支持传感器特征提取。
 * 时域动态演化由 multimodal.c 中的主CfC统一管理。
 */

#ifndef SELFLNN_SENSOR_H
#define SELFLNN_SENSOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 传感器类型枚举
 */
typedef enum {
    SENSOR_TYPE_ACCELEROMETER = 0,   /**< 加速度计 */
    SENSOR_TYPE_GYROSCOPE = 1,       /**< 陀螺仪 */
    SENSOR_TYPE_MAGNETOMETER = 2,    /**< 磁力计 */
    SENSOR_TYPE_TEMPERATURE = 3,     /**< 温度传感器 */
    SENSOR_TYPE_PRESSURE = 4,        /**< 压力传感器 */
    SENSOR_TYPE_HUMIDITY = 5,        /**< 湿度传感器 */
    SENSOR_TYPE_LIGHT = 6,           /**< 光传感器 */
    SENSOR_TYPE_PROXIMITY = 7,       /**< 接近传感器 */
    SENSOR_TYPE_GAS = 8,             /**< 气体/气味传感器（味觉模拟） */
    SENSOR_TYPE_TASTE = 9,           /**< 味觉传感器（pH/电导率/离子浓度） */
    SENSOR_TYPE_3D_SHAPE = 10        /**< 3D形状特征传感器 */
} SensorType;

/**
 * @brief 传感器处理配置
 */
typedef struct {
    SensorType sensor_type;          /**< 传感器类型 */
    int sampling_rate;               /**< 采样率 */
    int window_size;                 /**< 窗口大小 */
    int feature_dimension;           /**< 特征维度 */
    int enable_cfc;                  /**< 是否启用CfC连续时间动态演化 */
    int cfc_hidden_size;             /**< CfC隐藏状态大小 */
    float cfc_time_constant;         /**< CfC时间常数 */
} SensorConfig;

/**
 * @brief 传感器处理器句柄
 */
typedef struct SensorProcessor SensorProcessor;

/**
 * @brief 创建传感器处理器
 * 
 * @param config 传感器配置
 * @return SensorProcessor* 处理器句柄，失败返回NULL
 */
SensorProcessor* sensor_processor_create(const SensorConfig* config);

/**
 * @brief 释放传感器处理器
 * 
 * @param processor 处理器句柄
 */
void sensor_processor_free(SensorProcessor* processor);

/**
 * @brief 处理传感器数据
 * 
 * 纯数值归一化处理，原始信号由 multimodal.c 中的主CfC统一演化。
 * 
 * @param processor 处理器句柄
 * @param values 传感器值数组
 * @param num_values 值数量
 * @param timestamp 时间戳
 * @param features 特征输出缓冲区
 * @param max_features 最大特征数
 * @return int 成功返回特征数，失败返回-1
 */
int sensor_process_data(SensorProcessor* processor,
                       const float* values, size_t num_values,
                       long timestamp,
                       float* features, size_t max_features);

/**
 * @brief 提取时域统计特征
 * 
 * 纯数值归一化复制，原始信号由主CfC统一处理。
 * 
 * @param processor 处理器句柄
 * @param values 传感器值数组
 * @param num_values 值数量
 * @param stat_features 统计特征输出缓冲区
 * @param max_features 最大特征数
 * @return int 成功返回特征数，失败返回-1
 */
int sensor_extract_stat_features(SensorProcessor* processor,
                                const float* values, size_t num_values,
                                float* stat_features, size_t max_features);

/**
 * @brief 检测传感器事件
 * 
 * 全局RMS能量检测，由主CfC统一决策。
 * 
 * @param processor 处理器句柄
 * @param values 传感器值数组
 * @param num_values 值数量
 * @param event_type 事件类型输出缓冲区
 * @param event_confidence 事件置信度输出缓冲区
 * @return int 成功返回事件类型，失败返回-1
 */
int sensor_detect_event(SensorProcessor* processor,
                       const float* values, size_t num_values,
                       int* event_type, float* event_confidence);

/**
 * @brief 获取传感器处理器配置
 * 
 * @param processor 处理器句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int sensor_processor_get_config(const SensorProcessor* processor, SensorConfig* config);

/**
 * @brief 设置传感器处理器配置
 * 
 * @param processor 处理器句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int sensor_processor_set_config(SensorProcessor* processor, const SensorConfig* config);

/**
 * @brief 重置传感器处理器
 * 
 * @param processor 处理器句柄
 */
void sensor_processor_reset(SensorProcessor* processor);

/* ---------------------------------------------------------------------------
 * P2-23: 传感器噪声模型在线校准
 * --------------------------------------------------------------------------- */

/** 对传感器进行在线噪声校准，计算偏差和方差 */
int sensor_calibrate_noise(SensorProcessor* processor,
                           const float* values, size_t num_values,
                           int num_calibration_samples);

/** 对原始传感器数据应用校准（去偏+归一化） */
int sensor_apply_calibration(const SensorProcessor* processor,
                             float* values, size_t num_values);

/** 获取当前校准参数 */
int sensor_get_calibration(const SensorProcessor* processor,
                           float* offset, float* scale, int max_channels);

/** 重置传感器校准 */
void sensor_reset_calibration(SensorProcessor* processor);

// ==================== 扩展卡尔曼滤波器（EKF） ====================

/**
 * @brief EKF滤波器配置
 */
typedef struct {
    int state_dim;                   /**< 状态向量维度 */
    int obs_dim;                     /**< 观测向量维度 */
    int control_dim;                 /**< 控制输入维度 */
    float process_noise_std;         /**< 过程噪声标准差 */
    float observation_noise_std;     /**< 观测噪声标准差 */
    float initial_state_std;         /**< 初始状态不确定性标准差 */
    int use_cfc_transition;          /**< 使用CfC作为状态转移模型（默认1） */
    int cfc_hidden_size;             /**< CfC隐藏状态维度（use_cfc_transition时使用） */
} EKFConfig;

typedef struct EKFFilter EKFFilter;

/**
 * @brief 创建扩展卡尔曼滤波器
 *
 * @param config EKF配置
 * @return EKFFilter* 句柄，失败返回NULL
 */
EKFFilter* ekf_create(const EKFConfig* config);

/**
 * @brief 释放扩展卡尔曼滤波器
 */
void ekf_free(EKFFilter* ekf);

/**
 * @brief EKF预测步骤
 *
 * 使用CfC ODE状态转移模型预测下一时刻状态：
 * x_pred = CfC_ODE(x, u, dt)
 * P_pred = F*P*F^T + Q
 * 其中F为状态转移雅可比（通过CfC自动微分或数值差分）
 *
 * @param ekf EKF句柄
 * @param control 控制输入向量（大小为control_dim，可为NULL）
 * @param dt 时间步长
 * @return int 成功返回0，失败返回-1
 */
int ekf_predict(EKFFilter* ekf, const float* control, float dt);

/**
 * @brief EKF更新步骤
 *
 * 使用观测更新状态估计：
 * y = z - h(x_pred)
 * S = H*P_pred*H^T + R
 * K = P_pred*H^T*S^(-1)
 * x = x_pred + K*y
 * P = (I - K*H)*P_pred
 *
 * @param ekf EKF句柄
 * @param observation 观测向量（大小为obs_dim）
 * @return int 成功返回0，失败返回-1
 */
int ekf_update(EKFFilter* ekf, const float* observation);

/**
 * @brief 获取EKF状态估计和协方差
 *
 * @param ekf EKF句柄
 * @param state 输出状态向量
 * @param covariance 输出协方差矩阵（可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int ekf_get_state(const EKFFilter* ekf, float* state, float* covariance);

/**
 * @brief 重置EKF状态
 *
 * @param ekf EKF句柄
 * @param state 初始状态向量（可为NULL则用0）
 * @param covariance 初始协方差矩阵（可为NULL则用默认）
 */
void ekf_reset(EKFFilter* ekf, const float* state, const float* covariance);

// ==================== 无迹卡尔曼滤波器（UKF） ====================

/**
 * @brief UKF滤波器配置
 */
typedef struct {
    int state_dim;                   /**< 状态向量维度 */
    int obs_dim;                     /**< 观测向量维度 */
    int control_dim;                 /**< 控制输入维度 */
    float process_noise_std;         /**< 过程噪声标准差 */
    float observation_noise_std;     /**< 观测噪声标准差 */
    float initial_state_std;         /**< 初始状态不确定性标准差 */
    float alpha;                     /**< Sigma点分布参数（默认1e-3） */
    float beta;                      /**< 状态分布先验参数 */
    float kappa;                     /**< 次级缩放参数（默认0） */
    int use_cfc_transition;          /**< 使用CfC作为状态转移模型（默认1） */
    int cfc_hidden_size;             /**< CfC隐藏状态维度 */
} UKFConfig;

typedef struct UKFFilter UKFFilter;

/**
 * @brief 创建无迹卡尔曼滤波器
 *
 * @param config UKF配置
 * @return UKFFilter* 句柄，失败返回NULL
 */
UKFFilter* ukf_create(const UKFConfig* config);

/**
 * @brief 释放无迹卡尔曼滤波器
 */
void ukf_free(UKFFilter* ukf);

/**
 * @brief UKF预测步骤
 *
 * 1. 生成Sigma点
 * 2. 每个Sigma点通过CfC ODE传播
 * 3. 加权计算预测均值和协方差
 *
 * @param ukf UKF句柄
 * @param control 控制输入（可为NULL）
 * @param dt 时间步长
 * @return int 成功返回0，失败返回-1
 */
int ukf_predict(UKFFilter* ukf, const float* control, float dt);

/**
 * @brief UKF更新步骤
 *
 * @param ukf UKF句柄
 * @param observation 观测向量
 * @return int 成功返回0，失败返回-1
 */
int ukf_update(UKFFilter* ukf, const float* observation);

/**
 * @brief 获取UKF状态估计和协方差
 *
 * @param ukf UKF句柄
 * @param state 输出状态向量
 * @param covariance 输出协方差矩阵（可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int ukf_get_state(const UKFFilter* ukf, float* state, float* covariance);

/**
 * @brief 重置UKF状态
 */
void ukf_reset(UKFFilter* ukf, const float* state, const float* covariance);

// ==================== 误差状态卡尔曼滤波器（ESKF） ====================

/**
 * @brief ESKF滤波器配置
 */
typedef struct {
    int nom_state_dim;               /**< 名义状态维度 */
    int error_state_dim;             /**< 误差状态维度 */
    int obs_dim;                     /**< 观测维度 */
    int control_dim;                 /**< 控制输入维度 */
    float gyro_noise_std;            /**< 陀螺仪噪声标准差 */
    float acc_noise_std;             /**< 加速度计噪声标准差 */
    float gyro_bias_noise_std;       /**< 陀螺仪偏置噪声标准差 */
    float acc_bias_noise_std;        /**< 加速度计偏置噪声标准差 */
    int use_cfc_transition;          /**< 使用CfC作为状态转移（默认1） */
    int cfc_hidden_size;             /**< CfC隐藏状态维度 */
} ESKFConfig;

typedef struct ESKFFilter ESKFFilter;

/**
 * @brief 创建误差状态卡尔曼滤波器
 *
 * @param config ESKF配置
 * @return ESKFFilter* 句柄，失败返回NULL
 */
ESKFFilter* eskf_create(const ESKFConfig* config);

/**
 * @brief 释放ESKF
 */
void eskf_free(ESKFFilter* eskf);

/**
 * @brief ESKF名义状态预测
 *
 * 通过CfC ODE传播名义状态
 *
 * @param eskf ESKF句柄
 * @param gyro 陀螺仪测量 (3)
 * @param acc 加速度计测量 (3)
 * @param dt 时间步长
 * @return int 成功返回0，失败返回-1
 */
int eskf_predict_nominal(ESKFFilter* eskf, const float gyro[3], const float acc[3], float dt);

/**
 * @brief ESKF误差状态预测
 *
 * @param eskf ESKF句柄
 * @param dt 时间步长
 * @return int 成功返回0，失败返回-1
 */
int eskf_predict_error(ESKFFilter* eskf, float dt);

/**
 * @brief ESKF更新步骤
 *
 * @param eskf ESKF句柄
 * @param observation 观测向量
 * @param obs_model_type 观测模型类型
 * @return int 成功返回0，失败返回-1
 */
int eskf_update(ESKFFilter* eskf, const float* observation, int obs_model_type);

/**
 * @brief 获取ESKF状态
 *
 * @param eskf ESKF句柄
 * @param nom_state 名义状态输出
 * @param error_state 误差状态输出（可为NULL）
 * @param covariance 误差状态协方差输出（可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int eskf_get_state(const ESKFFilter* eskf, float* nom_state, float* error_state, float* covariance);

/**
 * @brief 重置ESKF
 */
void eskf_reset(ESKFFilter* eskf, const float* init_nom_state);

// ==================== 粒子滤波器（Particle Filter） ====================

/**
 * @brief 粒子滤波器配置
 */
typedef struct {
    int state_dim;                   /**< 状态维度 */
    int obs_dim;                     /**< 观测维度 */
    int num_particles;               /**< 粒子数量（默认500） */
    float process_noise_std;         /**< 过程噪声标准差 */
    float observation_noise_std;     /**< 观测噪声标准差 */
    float resampling_threshold;      /**< 有效粒子数阈值比例（默认0.5） */
    int resampling_method;           /**< 重采样方法：0=系统，1=分层，2=残差 */
    int use_systematic_resampling;   /**< 使用系统重采样（默认1） */
    int use_cfc_motion_model;        /**< 使用CfC作为运动模型（默认1） */
    int cfc_hidden_size;             /**< CfC隐藏状态维度 */
} ParticleFilterConfig;

typedef struct ParticleFilter ParticleFilter;

/**
 * @brief 创建粒子滤波器
 *
 * @param config 粒子滤波器配置
 * @return ParticleFilter* 句柄，失败返回NULL
 */
ParticleFilter* particle_filter_create(const ParticleFilterConfig* config);

/**
 * @brief 释放粒子滤波器
 */
void particle_filter_free(ParticleFilter* pf);

/**
 * @brief 粒子滤波器预测步骤
 *
 * 每个粒子通过CfC ODE运动模型传播
 *
 * @param pf 粒子滤波器句柄
 * @param control 控制输入（可为NULL）
 * @param dt 时间步长
 * @return int 成功返回0，失败返回-1
 */
int particle_filter_predict(ParticleFilter* pf, const float* control, float dt);

/**
 * @brief 粒子滤波器更新步骤
 *
 * @param pf 粒子滤波器句柄
 * @param observation 观测向量
 * @return int 成功返回0，失败返回-1
 */
int particle_filter_update(ParticleFilter* pf, const float* observation);

/**
 * @brief 执行粒子重采样
 *
 * @param pf 粒子滤波器句柄
 * @return int 成功返回0，失败返回-1
 */
int particle_filter_resample(ParticleFilter* pf);

/**
 * @brief 获取粒子滤波器状态估计
 *
 * @param pf 粒子滤波器句柄
 * @param mean_state 输出加权平均状态
 * @param covariance 输出状态协方差（可为NULL）
 * @param effective_particles 输出有效粒子数（可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int particle_filter_get_state(const ParticleFilter* pf, float* mean_state, float* covariance, int* effective_particles);

/**
 * @brief 获取所有粒子状态和权重
 *
 * @param pf 粒子滤波器句柄
 * @param particles 输出粒子状态数组
 * @param weights 输出粒子权重数组
 * @param max_particles 最大粒子数
 * @return int 成功返回粒子数，失败返回-1
 */
int particle_filter_get_particles(const ParticleFilter* pf, float* particles, float* weights, int max_particles);

/**
 * @brief 重置粒子滤波器
 *
 * @param pf 粒子滤波器句柄
 * @param init_mean 初始状态均值
 * @param init_std 初始状态标准差
 * @return int 成功返回0，失败返回-1
 */
int particle_filter_reset(ParticleFilter* pf, const float* init_mean, float init_std);

// ==================== 信息滤波器（Information Filter） ====================

/**
 * @brief 信息滤波器配置
 */
typedef struct {
    int state_dim;                   /**< 状态维度 */
    int obs_dim;                     /**< 观测维度 */
    int control_dim;                 /**< 控制输入维度 */
    float process_info_std;          /**< 过程信息标准差 */
    float observation_info_std;      /**< 观测信息标准差 */
    int use_cfc_transition;          /**< 使用CfC状态转移（默认1） */
    int cfc_hidden_size;             /**< CfC隐藏状态维度 */
} InfoFilterConfig;

typedef struct InfoFilter InfoFilter;

/**
 * @brief 创建信息滤波器
 *
 * @param config 信息滤波器配置
 * @return InfoFilter* 句柄，失败返回NULL
 */
InfoFilter* info_filter_create(const InfoFilterConfig* config);

/**
 * @brief 释放信息滤波器
 */
void info_filter_free(InfoFilter* inf);

/**
 * @brief 信息滤波器预测步骤
 *
 * @param inf 信息滤波器句柄
 * @param control 控制输入（可为NULL）
 * @param dt 时间步长
 * @return int 成功返回0，失败返回-1
 */
int info_filter_predict(InfoFilter* inf, const float* control, float dt);

/**
 * @brief 信息滤波器更新步骤
 *
 * @param inf 信息滤波器句柄
 * @param observation 观测向量
 * @param obs_jacobian 观测雅可比矩阵（可为NULL则用单位阵）
 * @return int 成功返回0，失败返回-1
 */
int info_filter_update(InfoFilter* inf, const float* observation, const float* obs_jacobian);

/**
 * @brief 获取信息滤波器状态
 *
 * @param inf 信息滤波器句柄
 * @param state 输出状态向量
 * @param information_matrix 输出信息矩阵（可为NULL）
 * @param covariance 输出协方差矩阵（可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int info_filter_get_state(const InfoFilter* inf, float* state, float* information_matrix, float* covariance);

/**
 * @brief 重置信息滤波器
 */
void info_filter_reset(InfoFilter* inf, const float* init_state, const float* init_covariance);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_SENSOR_H
