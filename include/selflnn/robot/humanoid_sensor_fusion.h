#ifndef SELFLNN_HUMANOID_SENSOR_FUSION_H
#define SELFLNN_HUMANOID_SENSOR_FUSION_H

#include "selflnn/robot/robot.h"
#include "selflnn/robot/sensor_pipeline.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HUMAN_SENSOR_FUSION_MAX_JOINTS          32
#define HUMAN_SENSOR_FUSION_MAX_IMUS            8
#define HUMAN_SENSOR_FUSION_MAX_CONTACTS        4
#define HUMAN_SENSOR_FUSION_FILTER_ORDER        4
#define HUMAN_SENSOR_FUSION_CALIB_SAMPLES       200
#define HUMAN_SENSOR_FUSION_KALMAN_STATE_DIM    15
#define HUMAN_SENSOR_FUSION_KALMAN_MEASURE_DIM  9
#define HUMAN_SENSOR_FUSION_NAME_MAX            64

typedef enum {
    FUSION_STATE_INITIALIZING = 0,
    FUSION_STATE_CALIBRATING = 1,
    FUSION_STATE_RUNNING = 2,
    FUSION_STATE_ERROR = 3,
    FUSION_STATE_SUSPENDED = 4
} FusionState;

typedef enum {
    GAIT_PHASE_UNKNOWN = 0,
    GAIT_PHASE_STANCE = 1,
    GAIT_PHASE_SWING = 2,
    GAIT_PHASE_DOUBLE_SUPPORT = 3,
    GAIT_PHASE_SINGLE_SUPPORT_LEFT = 4,
    GAIT_PHASE_SINGLE_SUPPORT_RIGHT = 5,
    GAIT_PHASE_LANDING = 6,
    GAIT_PHASE_LIFTOFF = 7,
    GAIT_PHASE_STANDING = 8,
    GAIT_PHASE_SITTING = 9,
    GAIT_PHASE_JUMPING = 10,
    GAIT_PHASE_RUNNING = 11
} GaitPhase;

typedef enum {
    FUSION_CONTACT_NONE = 0,
    FUSION_CONTACT_HEEL = 1,
    FUSION_CONTACT_TOE = 2,
    FUSION_CONTACT_FULL = 3,
    FUSION_CONTACT_BALL = 4
} FusionContactType;

typedef enum {
    FUSION_IMU_LOCATION_HEAD = 0,
    FUSION_IMU_LOCATION_CHEST = 1,
    FUSION_IMU_LOCATION_PELVIS = 2,
    FUSION_IMU_LOCATION_LEFT_UPPER_ARM = 3,
    FUSION_IMU_LOCATION_RIGHT_UPPER_ARM = 4,
    FUSION_IMU_LOCATION_LEFT_THIGH = 5,
    FUSION_IMU_LOCATION_RIGHT_THIGH = 6,
    FUSION_IMU_LOCATION_LEFT_SHANK = 7,
    FUSION_IMU_LOCATION_RIGHT_SHANK = 8,
    FUSION_IMU_LOCATION_LEFT_FOOT = 9,
    FUSION_IMU_LOCATION_RIGHT_FOOT = 10
} FusionImuLocation;

typedef struct {
    double qw, qx, qy, qz;
    double gyro_bias_x, gyro_bias_y, gyro_bias_z;
    double accel_bias_x, accel_bias_y, accel_bias_z;
    double gyro_scale_x, gyro_scale_y, gyro_scale_z;
    double accel_scale_x, accel_scale_y, accel_scale_z;
    double misalignment_matrix[3][3];
    double temperature_coefficient[3];
} FusionImuCalibration;

typedef struct {
    double orientation[4];
    double angular_velocity[3];
    double linear_acceleration[3];
    double orientation_covariance[9];
    double angular_velocity_covariance[9];
    double linear_acceleration_covariance[9];
    double world_linear_acceleration[3];
    double world_velocity[3];
    double world_position[3];
    double gravity_vector[3];
    double yaw, pitch, roll;
    FusionImuCalibration calibration;
    FusionImuLocation location;
    int is_calibrated;
    double last_update_time;
    double update_rate_hz;
    double temperature;
} FusionImuData;

typedef struct {
    double position[3];
    double velocity[3];
    double orientation[4];
    double angular_velocity[3];
    double joint_positions[HUMAN_SENSOR_FUSION_MAX_JOINTS];
    double joint_velocities[HUMAN_SENSOR_FUSION_MAX_JOINTS];
    double joint_torques[HUMAN_SENSOR_FUSION_MAX_JOINTS];
    int num_joints;
    double mass_total;
    double com_position[3];
    double com_velocity[3];
    double zmp_position[2];
    double zmp_estimated[2];
    double foot_positions[2][3];
    double foot_orientations[2][4];
    double foot_contact_forces[2][3];
    double foot_contact_torques[2][3];
    FusionContactType left_foot_contact;
    FusionContactType right_foot_contact;
    double left_foot_force_magnitude;
    double right_foot_force_magnitude;
    double torso_orientation[4];
    double head_orientation[4];
    double base_height;
    double walking_speed;
    double step_length;
    double step_height;
    double stance_width;
    GaitPhase gait_phase;
    double gait_cycle_progress;
    double stability_measure;
    FusionImuData imus[HUMAN_SENSOR_FUSION_MAX_IMUS];
    int imu_count;
    double sensor_timestamp;
    double fusion_timestamp;
    double update_rate_hz;
    FusionState state;
    /* GNSS全局定位数据 */
    double global_position[3];
    float gnss_accuracy;
    int gnss_updated;
} FusionHumanoidState;

typedef struct {
    double process_noise_q[15];
    double measurement_noise_r[9];
    double initial_covariance_p[15];
    double dt;
    double gyro_trust_factor;
    double accel_trust_factor;
    double vision_trust_factor;
    double joint_trust_factor;
    double foot_contact_threshold;
    double com_height_offset;
    double zmp_margin_x;
    double zmp_margin_y;
    int enable_magnetometer;
    int enable_vision_fusion;
    int enable_joint_fusion;
    int enable_foot_contact;
    int enable_gait_detection;
    int auto_calibrate_imus;
    int calibration_samples;
    double gravity_magnitude;
    char robot_name[HUMAN_SENSOR_FUSION_NAME_MAX];
} FusionConfig;

typedef struct {
    double P[15][15];
    double Q[15];
    double R[9];
    double x_est[15];
    int initialized;
} KalmanFilterState;

typedef struct {
    double previous_foot_positions[2][3];
    double previous_joint_positions[HUMAN_SENSOR_FUSION_MAX_JOINTS];
    double previous_timestamps[5];
    GaitPhase previous_phase;
    double phase_duration;
    double phase_start_time;
    double cycle_count;
    double swing_height[2];
    double step_length_measured[2];
    double cadence;
    int gait_initialized;
} GaitAnalyzerState;

typedef struct {
    double com_alpha;
    double com_pos_filtered[3];
    double com_vel_filtered[3];
    double zmp_alpha;
    double zmp_filtered[2];
    int filter_initialized;
} LowPassFilterState;

typedef struct HumanoidSensorFusion HumanoidSensorFusion;

HumanoidSensorFusion* humanoid_sensor_fusion_create(const FusionConfig* config);
void humanoid_sensor_fusion_destroy(HumanoidSensorFusion* fusion);

int humanoid_sensor_fusion_init(HumanoidSensorFusion* fusion);
int humanoid_sensor_fusion_reset(HumanoidSensorFusion* fusion);
int humanoid_sensor_fusion_calibrate(HumanoidSensorFusion* fusion);
int humanoid_sensor_fusion_suspend(HumanoidSensorFusion* fusion);
int humanoid_sensor_fusion_resume(HumanoidSensorFusion* fusion);

int humanoid_sensor_fusion_update(HumanoidSensorFusion* fusion, double dt);
int humanoid_sensor_fusion_update_with_sensor_pipeline(HumanoidSensorFusion* fusion,
                                                        SensorPipeline* pipeline, double dt);

int humanoid_sensor_fusion_feed_imu(HumanoidSensorFusion* fusion, int imu_index,
                                     const double* acceleration, const double* angular_velocity,
                                     const double* orientation, FusionImuLocation location,
                                     double temperature, double timestamp);

int humanoid_sensor_fusion_feed_joint_states(HumanoidSensorFusion* fusion,
                                              const double* positions, const double* velocities,
                                              const double* torques, int num_joints,
                                              double timestamp);

int humanoid_sensor_fusion_feed_vision_pose(HumanoidSensorFusion* fusion,
                                             const double* position, const double* orientation,
                                             const double* covariance, double timestamp);

int humanoid_sensor_fusion_feed_foot_contact(HumanoidSensorFusion* fusion,
                                              int is_left_foot,
                                              const double* contact_force,
                                              const double* contact_torque,
                                              FusionContactType contact_type,
                                              double timestamp, double force_magnitude);

int humanoid_sensor_fusion_feed_gnss(HumanoidSensorFusion* fusion,
                                      double latitude, double longitude, double altitude,
                                      double timestamp, float accuracy);

int humanoid_sensor_fusion_get_state(HumanoidSensorFusion* fusion,
                                     FusionHumanoidState* state);
int humanoid_sensor_fusion_get_imu(HumanoidSensorFusion* fusion, int imu_index,
                                   FusionImuData* imu_data);
int humanoid_sensor_fusion_get_com(HumanoidSensorFusion* fusion,
                                    double* com_position, double* com_velocity);
int humanoid_sensor_fusion_get_zmp(HumanoidSensorFusion* fusion,
                                    double* zmp_position, double* zmp_estimated);
int humanoid_sensor_fusion_get_gait(HumanoidSensorFusion* fusion,
                                     GaitPhase* phase, double* progress);
int humanoid_sensor_fusion_get_orientation(HumanoidSensorFusion* fusion,
                                            double* orientation, double* yaw_pitch_roll);

int humanoid_sensor_fusion_set_parameter(HumanoidSensorFusion* fusion,
                                          const char* param_name, double value);
double humanoid_sensor_fusion_get_parameter(HumanoidSensorFusion* fusion,
                                             const char* param_name);

void humanoid_sensor_fusion_get_fusion_config(const HumanoidSensorFusion* fusion,
                                               FusionConfig* config);
int humanoid_sensor_fusion_set_fusion_config(HumanoidSensorFusion* fusion,
                                              const FusionConfig* config);
void humanoid_sensor_fusion_set_default_config(FusionConfig* config);

int humanoid_sensor_fusion_is_calibrated(HumanoidSensorFusion* fusion);
int humanoid_sensor_fusion_get_error_code(HumanoidSensorFusion* fusion);
const char* humanoid_sensor_fusion_get_last_error(HumanoidSensorFusion* fusion);

const char* humanoid_sensor_fusion_gait_phase_string(GaitPhase phase);
const char* humanoid_sensor_fusion_state_string(FusionState state);

typedef struct {
    int imu_count;
    int joint_count;
    int vision_updates;
    int contact_updates;
    int gnss_updates;
    int kalman_updates;
    double avg_update_time_us;
    double max_update_time_us;
    double last_update_time_us;
    double stability_measure;
    double com_estimation_error;
    double zmp_estimation_error;
    int is_stable;
    int calibration_progress;
} FusionPerformanceStats;

int humanoid_sensor_fusion_get_stats(HumanoidSensorFusion* fusion,
                                      FusionPerformanceStats* stats);

/* ============================
 * 传感器校准增强（M17）
 * ============================ */

typedef enum {
    FUSION_CALIB_NONE = 0,
    FUSION_CALIB_GYRO_BIAS,
    FUSION_CALIB_ACCEL_BIAS,
    FUSION_CALIB_SCALE_FACTOR,
    FUSION_CALIB_MISALIGNMENT,
    FUSION_CALIB_TEMPERATURE,
    FUSION_CALIB_COMPLETE
} FusionCalibStage;

typedef struct {
    double gyro_bias[3];
    double accel_bias[3];
    double gyro_scale[3];
    double accel_scale[3];
    double misalignment[3][3];
    double temp_coeff_gyro[3];
    double temp_coeff_accel[3];
    double reference_temperature;
    int is_valid;
    double calibration_quality;
    FusionCalibStage stage;
    int sample_count;
    int samples_required;
} FusionCalibrationData;

typedef struct {
    int imu_index;
    FusionImuLocation location;
    FusionCalibrationData calib;
    double bias_stability[3];
    double noise_density[3];
    double random_walk[3];
    int is_calibrated;
    int calibration_progress;
    double last_calibration_time;
} FusionImuCalibState;

/**
 * @brief 开始IMU自动校准流程
 *
 * 采集静态数据估计陀螺仪零偏、加速度计零偏和比例因子。
 * 校准完成后自动切换到融合运行状态。
 *
 * @param fusion 传感器融合句柄
 * @param imu_index IMU索引（-1表示所有IMU）
 * @return int 成功返回0，失败返回-1
 */
int humanoid_sensor_fusion_start_calibration(HumanoidSensorFusion* fusion, int imu_index);

/**
 * @brief 获取IMU校准进度
 *
 * @param fusion 传感器融合句柄
 * @param imu_index IMU索引
 * @param progress 输出校准进度（0~100）
 * @return int 成功返回0，失败返回-1
 */
int humanoid_sensor_fusion_get_calibration_progress(HumanoidSensorFusion* fusion,
                                                     int imu_index, int* progress);

/**
 * @brief 获取IMU校准数据
 *
 * @param fusion 传感器融合句柄
 * @param imu_index IMU索引
 * @param calib 输出校准数据
 * @return int 成功返回0，失败返回-1
 */
int humanoid_sensor_fusion_get_calibration_data(HumanoidSensorFusion* fusion,
                                                 int imu_index,
                                                 FusionCalibrationData* calib);

/**
 * @brief 应用温度补偿
 *
 * 根据当前温度修正IMU的偏差和比例因子。
 *
 * @param fusion 传感器融合句柄
 * @param imu_index IMU索引
 * @param current_temperature 当前温度（摄氏度）
 * @return int 成功返回0，失败返回-1
 */
int humanoid_sensor_fusion_apply_temperature_compensation(HumanoidSensorFusion* fusion,
                                                           int imu_index,
                                                           double current_temperature);

/* ============================
 * 传感器故障检测（M17）
 * ============================ */

typedef enum {
    FUSION_FAULT_NONE = 0,
    FUSION_FAULT_SIGNAL_LOST,
    FUSION_FAULT_SATURATION,
    FUSION_FAULT_NOISE_EXCEEDED,
    FUSION_FAULT_STUCK_VALUE,
    FUSION_FAULT_OUTLIER,
    FUSION_FAULT_DRIFT_EXCEEDED,
    FUSION_FAULT_TEMPERATURE_ABNORMAL,
    FUSION_FAULT_TIMEOUT,
    FUSION_FAULT_CALIBRATION_INVALID,
    FUSION_FAULT_HARDWARE
} FusionSensorFaultType;

typedef struct {
    FusionSensorFaultType fault_type;
    int imu_index;
    double fault_score;
    double detection_time;
    int consecutive_faults;
    char description[128];
} FusionSensorFaultEvent;

typedef struct {
    int sensor_online;
    int data_valid;
    double signal_strength;
    double noise_level;
    double drift_rate[3];
    double last_valid_timestamp;
    int fault_count;
    double fault_rate;
    double last_fault_time;
    double health_score;
    int is_recovering;
    double recovery_progress;
} FusionSensorHealth;

typedef struct {
    FusionSensorHealth imu_health[HUMAN_SENSOR_FUSION_MAX_IMUS];
    FusionSensorHealth joint_health;
    FusionSensorHealth vision_health;
    FusionSensorHealth foot_health[2];
    FusionSensorHealth gnss_health;
    int total_faults;
    int active_faults;
    double last_fault_detection_time;
    int fault_events_capacity;
    int fault_events_count;
    FusionSensorFaultEvent* fault_events;
} FusionFaultDetector;

/**
 * @brief 创建传感器故障检测器
 *
 * @param fusion 传感器融合句柄
 * @return int 成功返回0，失败返回-1
 */
int humanoid_sensor_fusion_create_fault_detector(HumanoidSensorFusion* fusion);

/**
 * @brief 销毁传感器故障检测器
 *
 * @param fusion 传感器融合句柄
 */
void humanoid_sensor_fusion_destroy_fault_detector(HumanoidSensorFusion* fusion);

/**
 * @brief 执行传感器故障检测
 *
 * 检查所有传感器的数据有效性、噪声水平、漂移等指标。
 * 检测到的故障事件会记录到故障检测器中。
 *
 * @param fusion 传感器融合句柄
 * @param dt 时间步长（秒）
 * @return int 检测到的故障数量，失败返回-1
 */
int humanoid_sensor_fusion_detect_faults(HumanoidSensorFusion* fusion, double dt);

/**
 * @brief 获取传感器健康状态
 *
 * @param fusion 传感器融合句柄
 * @param imu_index IMU索引（-1表示整体状态）
 * @param health 输出健康状态
 * @return int 成功返回0，失败返回-1
 */
int humanoid_sensor_fusion_get_sensor_health(HumanoidSensorFusion* fusion,
                                              int imu_index,
                                              FusionSensorHealth* health);

/**
 * @brief 获取最近的故障事件
 *
 * @param fusion 传感器融合句柄
 * @param events 输出故障事件数组
 * @param max_count 最大数量
 * @param count 输出实际数量
 * @return int 成功返回0，失败返回-1
 */
int humanoid_sensor_fusion_get_fault_events(HumanoidSensorFusion* fusion,
                                             FusionSensorFaultEvent* events,
                                             size_t max_count, size_t* count);

/**
 * @brief 获取故障检测器统计信息
 *
 * @param fusion 传感器融合句柄
 * @param total_faults 输出总故障数
 * @param active_faults 输出当前活跃故障数
 * @param avg_health_score 输出平均健康评分（0~1）
 * @return int 成功返回0，失败返回-1
 */
int humanoid_sensor_fusion_get_fault_stats(HumanoidSensorFusion* fusion,
                                            int* total_faults,
                                            int* active_faults,
                                            double* avg_health_score);

/**
 * @brief 重置传感器故障检测器
 *
 * @param fusion 传感器融合句柄
 * @return int 成功返回0，失败返回-1
 */
int humanoid_sensor_fusion_reset_fault_detector(HumanoidSensorFusion* fusion);

/**
 * @brief 执行传感器数据验证与修复
 *
 * 检查传感器数据的合理性，对可疑数据进行标记或修正。
 *
 * @param fusion 传感器融合句柄
 * @param imu_index IMU索引
 * @param acceleration 加速度数据（输入输出）
 * @param angular_velocity 角速度数据（输入输出）
 * @return int 数据有效返回0，数据已修复返回1，数据不可用返回-1
 */
int humanoid_sensor_fusion_validate_sensor_data(HumanoidSensorFusion* fusion,
                                                  int imu_index,
                                                  double acceleration[3],
                                                  double angular_velocity[3]);

#ifdef __cplusplus
}
#endif

#endif
