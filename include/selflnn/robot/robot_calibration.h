#ifndef SELFLNN_ROBOT_CALIBRATION_H
#define SELFLNN_ROBOT_CALIBRATION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CALIBRATION_MAX_JOINTS 16
#define CALIBRATION_MAX_SAMPLES 5000
#define CALIBRATION_MAX_PARAMS (CALIBRATION_MAX_JOINTS * 4)
#define CALIBRATION_MAX_FOURIER_TERMS 10
#define CALIBRATION_NAME_MAX 64
#define CALIBRATION_MIN_SAMPLES 50

typedef enum {
    CALIB_TYPE_KINEMATIC = 0,
    CALIB_TYPE_DYNAMIC = 1,
    CALIB_TYPE_SENSOR = 2,
    CALIB_TYPE_TOOL = 3,
    CALIB_TYPE_JOINT_OFFSET = 4,
    CALIB_TYPE_FULL = 5
} CalibrationType;

typedef enum {
    CALIB_STATUS_IDLE = 0,
    CALIB_STATUS_INITIALIZING = 1,
    CALIB_STATUS_COLLECTING = 2,
    CALIB_STATUS_COMPUTING = 3,
    CALIB_STATUS_VERIFYING = 4,
    CALIB_STATUS_COMPLETED = 5,
    CALIB_STATUS_FAILED = 6
} CalibrationStatus;

typedef struct {
    float a;  // 连杆长度
    float d;  // 连杆偏距
    float alpha;  // 连杆扭角
    float theta_offset;  // 关节零位偏移
} DHParameters;

typedef struct {
    float position[CALIBRATION_MAX_JOINTS];
    float velocity[CALIBRATION_MAX_JOINTS];
    float torque[CALIBRATION_MAX_JOINTS];
    float actual_position[CALIBRATION_MAX_JOINTS];
    float residual[CALIBRATION_MAX_JOINTS];
    double timestamp;
} CalibrationSample;

typedef struct {
    int use_fourier_trajectory;
    float fourier_freq_base;
    float fourier_amplitudes[CALIBRATION_MAX_FOURIER_TERMS];
    float fourier_phases[CALIBRATION_MAX_FOURIER_TERMS];
    float trajectory_duration;
    float sampling_rate;
    float max_velocity;
    float max_acceleration;
    int num_excitation_cycles;
} ExcitationTrajectoryConfig;

typedef struct {
    int num_joints;
    DHParameters nominal_dh[CALIBRATION_MAX_JOINTS];
    DHParameters calibrated_dh[CALIBRATION_MAX_JOINTS];
    float joint_offsets[CALIBRATION_MAX_JOINTS];
    float joint_gear_ratios[CALIBRATION_MAX_JOINTS];
    float joint_friction[CALIBRATION_MAX_JOINTS];
    float link_masses[CALIBRATION_MAX_JOINTS];
    CalibrationStatus status;
    CalibrationType type;
    float calibration_error;
    float max_position_error;
    float rms_position_error;
    int sample_count;
    CalibrationSample samples[CALIBRATION_MAX_SAMPLES];
    ExcitationTrajectoryConfig excitation;
    char name[CALIBRATION_NAME_MAX];
    double start_time;
    double elapsed_time;
    int converged;
    int iteration_count;
} RobotCalibration;

typedef struct {
    CalibrationType type;
    int num_joints;
    int max_samples;
    float max_velocity;
    float max_acceleration;
    float trajectory_duration;
    float sampling_rate;
    int use_excitation;
    int use_least_squares;
    float convergence_threshold;
    int max_iterations;
    char name[CALIBRATION_NAME_MAX];
} CalibrationConfig;

extern const CalibrationConfig CALIBRATION_CONFIG_DEFAULT;

RobotCalibration* robot_calibration_create(const CalibrationConfig* config);
void robot_calibration_free(RobotCalibration* calib);
int robot_calibration_init(RobotCalibration* calib, const CalibrationConfig* config);
int robot_calibration_set_dh_parameters(RobotCalibration* calib,
                                         const DHParameters* dh_params);
int robot_calibration_get_dh_parameters(const RobotCalibration* calib,
                                         DHParameters* dh_params, int calibrated);
int robot_calibration_add_sample(RobotCalibration* calib,
                                  const float* joint_positions,
                                  const float* joint_velocities,
                                  const float* joint_torques,
                                  const float* actual_positions);
int robot_calibration_generate_excitation(RobotCalibration* calib,
                                           float* trajectory,
                                           size_t* num_points);
int robot_calibration_compute(RobotCalibration* calib);
int robot_calibration_verify(RobotCalibration* calib,
                              const float* test_positions,
                              const float* actual_positions,
                              int num_samples);
int robot_calibration_get_status(const RobotCalibration* calib,
                                  CalibrationStatus* status, float* error);
int robot_calibration_get_joint_offsets(const RobotCalibration* calib,
                                         float* offsets);
int robot_calibration_reset(RobotCalibration* calib);
int robot_calibration_save(const RobotCalibration* calib, const char* filepath);
int robot_calibration_load(RobotCalibration* calib, const char* filepath);
int robot_calibration_apply(RobotCalibration* calib, float* joint_positions,
                             int num_joints);
int robot_calibration_get_summary(const RobotCalibration* calib,
                                   char* buffer, size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif
