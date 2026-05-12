#include "selflnn/robot/humanoid_sensor_fusion.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct HumanoidSensorFusion {
    FusionConfig config;
    FusionHumanoidState state;
    KalmanFilterState kf;
    GaitAnalyzerState gait;
    LowPassFilterState lpf;
    FusionPerformanceStats stats;

    int calibration_sample_count;
    double calibration_accel_samples[HUMAN_SENSOR_FUSION_CALIB_SAMPLES][3];
    double calibration_gyro_samples[HUMAN_SENSOR_FUSION_CALIB_SAMPLES][3];
    double calibration_accel_sum[3];
    double calibration_gyro_sum[3];
    double calibration_accel_sq_sum[3];
    double calibration_gyro_sq_sum[3];

    // M17: 传感器校准增强
    FusionImuCalibState imu_calib[HUMAN_SENSOR_FUSION_MAX_IMUS];
    FusionImuCalibState* current_calib_target;
    int calibration_active;
    int calibration_target_imu;
    double calibration_timer;
    double calibration_temp_samples[HUMAN_SENSOR_FUSION_CALIB_SAMPLES];
    double calibration_temp_sum;

    // M17: 传感器故障检测
    FusionFaultDetector fault_detector;
    int fault_detector_created;
    double previous_accel[HUMAN_SENSOR_FUSION_MAX_IMUS][3];
    double previous_gyro[HUMAN_SENSOR_FUSION_MAX_IMUS][3];
    double prev_accel_variance[HUMAN_SENSOR_FUSION_MAX_IMUS][3];
    double stuck_check_timer[HUMAN_SENSOR_FUSION_MAX_IMUS];

    double previous_update_time;
    double integration_timer;
    int kalman_initialized;

    char last_error[256];
    int error_code;
};

static void quaternion_normalize(double q[4])
{
    double norm = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (norm < 1e-15) { q[0] = 1.0; q[1] = q[2] = q[3] = 0.0; return; }
    double inv = 1.0 / norm;
    q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
}

static void quaternion_multiply(const double a[4], const double b[4], double out[4])
{
    out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

static void quaternion_to_euler(const double q[4], double* yaw, double* pitch, double* roll)
{
    double sinr_cosp = 2.0 * (q[0] * q[1] + q[2] * q[3]);
    double cosr_cosp = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]);
    *roll = atan2(sinr_cosp, cosr_cosp);

    double sinp = 2.0 * (q[0] * q[2] - q[3] * q[1]);
    if (fabs(sinp) >= 1.0)
        *pitch = (sinp >= 0.0) ? M_PI / 2.0 : -M_PI / 2.0;
    else
        *pitch = asin(sinp);

    double siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2]);
    double cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]);
    *yaw = atan2(siny_cosp, cosy_cosp);
}

static void euler_to_quaternion(double yaw, double pitch, double roll, double q[4])
{
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);

    q[0] = cr * cp * cy + sr * sp * sy;
    q[1] = sr * cp * cy - cr * sp * sy;
    q[2] = cr * sp * cy + sr * cp * sy;
    q[3] = cr * cp * sy - sr * sp * cy;
}

static void rotate_vector_by_quaternion(const double v[3], const double q[4], double out[3])
{
    double qv[4] = {0, v[0], v[1], v[2]};
    double q_conj[4] = {q[0], -q[1], -q[2], -q[3]};
    double temp[4];
    quaternion_multiply(q, qv, temp);
    quaternion_multiply(temp, q_conj, temp);
    out[0] = temp[1]; out[1] = temp[2]; out[2] = temp[3];
}

static void matrix_3x3_multiply(const double a[3][3], const double b[3][3], double out[3][3])
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            out[i][j] = 0;
            for (int k = 0; k < 3; k++)
                out[i][j] += a[i][k] * b[k][j];
        }
}

static void matrix_3x3_vector_multiply(const double m[3][3], const double v[3], double out[3])
{
    for (int i = 0; i < 3; i++)
    {
        out[i] = 0;
        for (int j = 0; j < 3; j++)
            out[i] += m[i][j] * v[j];
    }
}

static double quaternion_dot_product(const double a[4], const double b[4])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

static void kalman_predict(KalmanFilterState* kf, double dt)
{
    if (!kf->initialized) return;

    for (int i = 0; i < 3; i++)
        kf->x_est[i] += kf->x_est[6 + i] * dt;

    for (int i = 3; i < 6; i++)
        kf->x_est[i] += kf->x_est[9 + i - 3] * dt;

    double dt2 = dt * dt;
    for (int i = 0; i < 15; i++)
        kf->P[i][i] += kf->Q[i] * dt;
    for (int i = 0; i < 3; i++)
        kf->P[i][i] += kf->Q[6 + i] * dt2 * 0.5;
    for (int i = 0; i < 3; i++)
        kf->P[6 + i][6 + i] += kf->Q[6 + i] * dt;
}

static void kalman_update_position(KalmanFilterState* kf,
                                    const double measured_pos[3],
                                    const double measured_cov[9])
{
    if (!kf->initialized) return;

    for (int i = 0; i < 3; i++)
    {
        double innovation = measured_pos[i] - kf->x_est[i];
        double s = kf->P[i][i] + measured_cov[i * 3 + i] + 1e-10;
        double k = kf->P[i][i] / s;
        kf->x_est[i] += k * innovation;
        kf->P[i][i] = (1.0 - k) * kf->P[i][i];
    }
}

static void kalman_update_orientation(KalmanFilterState* kf,
                                       const double measured_ori[4],
                                       const double measured_cov[9])
{
    if (!kf->initialized) return;
    double yaw, pitch, roll;
    quaternion_to_euler(measured_ori, &yaw, &pitch, &roll);

    double euler_est[3];
    quaternion_to_euler(&kf->x_est[3], &euler_est[0], &euler_est[1], &euler_est[2]);

    double measured_euler[3] = {yaw, pitch, roll};
    for (int i = 0; i < 3; i++)
    {
        double innovation = measured_euler[i] - euler_est[i];
        int p_idx = 3 + i;
        double s = kf->P[p_idx][p_idx] + measured_cov[i * 3 + i] + 1e-10;
        double k = kf->P[p_idx][p_idx] / s;
        euler_est[i] += k * innovation;
        kf->P[p_idx][p_idx] = (1.0 - k) * kf->P[p_idx][p_idx];
    }

    euler_to_quaternion(euler_est[0], euler_est[1], euler_est[2], &kf->x_est[3]);
    quaternion_normalize(&kf->x_est[3]);
}

static void kalman_init(KalmanFilterState* kf, const FusionConfig* config)
{
    memset(kf, 0, sizeof(KalmanFilterState));
    for (int i = 0; i < 15; i++)
    {
        kf->Q[i] = config->process_noise_q[i];
        kf->P[i][i] = config->initial_covariance_p[i];
    }
    for (int i = 0; i < 9; i++)
        kf->R[i] = config->measurement_noise_r[i];
    kf->x_est[0] = 0.0;
    kf->x_est[1] = 0.0;
    kf->x_est[2] = config->com_height_offset;
    kf->x_est[3] = 1.0;
    kf->initialized = 1;
}

static void lowpass_init(LowPassFilterState* lpf, double alpha)
{
    memset(lpf, 0, sizeof(LowPassFilterState));
    lpf->com_alpha = (alpha > 0.0 && alpha < 1.0) ? alpha : 0.1;
    lpf->zmp_alpha = lpf->com_alpha;
    lpf->filter_initialized = 0;
}

static void lowpass_update_3d(LowPassFilterState* lpf, const double input[3],
                               double output[3], int is_com)
{
    double alpha = is_com ? lpf->com_alpha : lpf->zmp_alpha;
    if (!lpf->filter_initialized)
    {
        output[0] = input[0]; output[1] = input[1]; output[2] = input[2];
        return;
    }
    for (int i = 0; i < 3; i++)
        output[i] = alpha * input[i] + (1.0 - alpha) * (is_com ? lpf->com_pos_filtered[i] : lpf->com_vel_filtered[i]);
}

static void lowpass_update_2d(LowPassFilterState* lpf, const double input[2],
                               double output[2])
{
    if (!lpf->filter_initialized)
    {
        output[0] = input[0]; output[1] = input[1];
        return;
    }
    output[0] = lpf->zmp_alpha * input[0] + (1.0 - lpf->zmp_alpha) * lpf->zmp_filtered[0];
    output[1] = lpf->zmp_alpha * input[1] + (1.0 - lpf->zmp_alpha) * lpf->zmp_filtered[1];
}

static void lowpass_finalize(LowPassFilterState* lpf, const double com_pos[3],
                              const double com_vel[3], const double zmp[2])
{
    if (!lpf->filter_initialized)
    {
        lpf->com_pos_filtered[0] = com_pos[0];
        lpf->com_pos_filtered[1] = com_pos[1];
        lpf->com_pos_filtered[2] = com_pos[2];
        lpf->com_vel_filtered[0] = com_vel[0];
        lpf->com_vel_filtered[1] = com_vel[1];
        lpf->com_vel_filtered[2] = com_vel[2];
        lpf->zmp_filtered[0] = zmp[0];
        lpf->zmp_filtered[1] = zmp[1];
        lpf->filter_initialized = 1;
    }
    else
    {
        lowpass_update_3d(lpf, com_pos, lpf->com_pos_filtered, 1);
        lowpass_update_3d(lpf, com_vel, lpf->com_vel_filtered, 0);
        lowpass_update_2d(lpf, zmp, lpf->zmp_filtered);
    }
}

static void estimate_com_from_joints(const double* joint_positions, int num_joints,
                                      double total_mass,
                                      double com_out[3])
{
    if (!joint_positions || num_joints <= 0)
    {
        com_out[0] = com_out[1] = 0.0;
        com_out[2] = 0.8;
        return;
    }

    double total_weighted_x = 0.0, total_weighted_y = 0.0, total_weighted_z = 0.0;
    double mass_sum = 0.0;

    double segment_masses[6] = {
        total_mass * 0.388,
        total_mass * 0.161,
        total_mass * 0.161,
        total_mass * 0.145,
        total_mass * 0.0725,
        total_mass * 0.0725
    };
    int num_segments = (num_joints < 6) ? num_joints : 6;

    for (int i = 0; i < num_segments; i++)
    {
        double segment_com[3];
        segment_com[0] = 0.1 * sin(joint_positions[i]);
        segment_com[1] = 0.05 * cos(joint_positions[i]);
        segment_com[2] = 0.2 + 0.15 * i;

        total_weighted_x += segment_masses[i] * segment_com[0];
        total_weighted_y += segment_masses[i] * segment_com[1];
        total_weighted_z += segment_masses[i] * segment_com[2];
        mass_sum += segment_masses[i];
    }

    if (mass_sum > 0.0)
    {
        com_out[0] = total_weighted_x / mass_sum;
        com_out[1] = total_weighted_y / mass_sum;
        com_out[2] = total_weighted_z / mass_sum;
    }
    else
    {
        com_out[0] = com_out[1] = 0.0;
        com_out[2] = 0.8;
    }
}

static void estimate_zmp_from_com(const double com_pos[3], const double com_accel[3],
                                   double zmp_out[2], double gravity)
{
    if (gravity < 0.1) gravity = 9.81;
    double height = com_pos[2];
    if (height < 0.1) height = 0.8;

    zmp_out[0] = com_pos[0] - (height / gravity) * com_accel[0];
    zmp_out[1] = com_pos[1] - (height / gravity) * com_accel[1];
}

static GaitPhase detect_gait_phase(GaitAnalyzerState* gait,
                                    double left_force, double right_force,
                                    FusionContactType left_contact,
                                    FusionContactType right_contact,
                                    const double com_pos[3],
                                    const double com_vel[3],
                                    double timestamp, double threshold)
{
    (void)com_pos;
    int left_contact_binary = (left_force > threshold &&
        (left_contact == FUSION_CONTACT_FULL || left_contact == FUSION_CONTACT_HEEL ||
         left_contact == FUSION_CONTACT_TOE || left_contact == FUSION_CONTACT_BALL));
    int right_contact_binary = (right_force > threshold &&
        (right_contact == FUSION_CONTACT_FULL || right_contact == FUSION_CONTACT_HEEL ||
         right_contact == FUSION_CONTACT_TOE || right_contact == FUSION_CONTACT_BALL));

    if (!gait->gait_initialized)
    {
        gait->gait_initialized = 1;
        gait->previous_phase = GAIT_PHASE_UNKNOWN;
        gait->phase_start_time = timestamp;
        if (left_contact_binary && right_contact_binary)
            return GAIT_PHASE_DOUBLE_SUPPORT;
        else if (left_contact_binary)
            return GAIT_PHASE_SINGLE_SUPPORT_LEFT;
        else if (right_contact_binary)
            return GAIT_PHASE_SINGLE_SUPPORT_RIGHT;
        return GAIT_PHASE_STANDING;
    }

    double com_speed = sqrt(com_vel[0] * com_vel[0] + com_vel[1] * com_vel[1]);
    GaitPhase new_phase;

    if (left_contact_binary && right_contact_binary)
    {
        if (com_speed < 0.05)
            new_phase = GAIT_PHASE_STANDING;
        else
            new_phase = GAIT_PHASE_DOUBLE_SUPPORT;
    }
    else if (left_contact_binary && !right_contact_binary)
    {
        if (com_speed > 0.3)
            new_phase = GAIT_PHASE_RUNNING;
        else
            new_phase = GAIT_PHASE_SINGLE_SUPPORT_LEFT;
    }
    else if (!left_contact_binary && right_contact_binary)
    {
        if (com_speed > 0.3)
            new_phase = GAIT_PHASE_RUNNING;
        else
            new_phase = GAIT_PHASE_SINGLE_SUPPORT_RIGHT;
    }
    else
    {
        new_phase = GAIT_PHASE_JUMPING;
    }

    if (new_phase != gait->previous_phase)
    {
        gait->phase_duration = timestamp - gait->phase_start_time;
        gait->phase_start_time = timestamp;
        if (new_phase == GAIT_PHASE_SINGLE_SUPPORT_LEFT ||
            new_phase == GAIT_PHASE_SINGLE_SUPPORT_RIGHT ||
            new_phase == GAIT_PHASE_DOUBLE_SUPPORT)
        {
            if (gait->previous_phase != GAIT_PHASE_UNKNOWN &&
                gait->previous_phase != GAIT_PHASE_STANDING)
            {
                gait->cycle_count += 0.5;
            }
        }
        gait->previous_phase = new_phase;
    }

    gait->phase_duration = timestamp - gait->phase_start_time;
    return new_phase;
}

static void calibrate_imu_from_samples(HumanoidSensorFusion* fusion,
                                        FusionImuData* imu)
{
    if (fusion->calibration_sample_count < 10) return;

    double accel_mean[3] = {0}, gyro_mean[3] = {0};
    double accel_std[3] = {0}, gyro_std[3] = {0};
    int n = fusion->calibration_sample_count;

    for (int i = 0; i < n; i++)
    {
        accel_mean[0] += fusion->calibration_accel_samples[i][0];
        accel_mean[1] += fusion->calibration_accel_samples[i][1];
        accel_mean[2] += fusion->calibration_accel_samples[i][2];
        gyro_mean[0] += fusion->calibration_gyro_samples[i][0];
        gyro_mean[1] += fusion->calibration_gyro_samples[i][1];
        gyro_mean[2] += fusion->calibration_gyro_samples[i][2];
    }
    double inv_n = 1.0 / n;
    accel_mean[0] *= inv_n; accel_mean[1] *= inv_n; accel_mean[2] *= inv_n;
    gyro_mean[0] *= inv_n; gyro_mean[1] *= inv_n; gyro_mean[2] *= inv_n;

    for (int i = 0; i < n; i++)
    {
        accel_std[0] += (fusion->calibration_accel_samples[i][0] - accel_mean[0]) *
                        (fusion->calibration_accel_samples[i][0] - accel_mean[0]);
        accel_std[1] += (fusion->calibration_accel_samples[i][1] - accel_mean[1]) *
                        (fusion->calibration_accel_samples[i][1] - accel_mean[1]);
        accel_std[2] += (fusion->calibration_accel_samples[i][2] - accel_mean[2]) *
                        (fusion->calibration_accel_samples[i][2] - accel_mean[2]);
        gyro_std[0] += (fusion->calibration_gyro_samples[i][0] - gyro_mean[0]) *
                       (fusion->calibration_gyro_samples[i][0] - gyro_mean[0]);
        gyro_std[1] += (fusion->calibration_gyro_samples[i][1] - gyro_mean[1]) *
                       (fusion->calibration_gyro_samples[i][1] - gyro_mean[1]);
        gyro_std[2] += (fusion->calibration_gyro_samples[i][2] - gyro_mean[2]) *
                       (fusion->calibration_gyro_samples[i][2] - gyro_mean[2]);
    }
    double inv_n1 = 1.0 / (n - 1);
    accel_std[0] = sqrt(accel_std[0] * inv_n1);
    accel_std[1] = sqrt(accel_std[1] * inv_n1);
    accel_std[2] = sqrt(accel_std[2] * inv_n1);
    gyro_std[0] = sqrt(gyro_std[0] * inv_n1);
    gyro_std[1] = sqrt(gyro_std[1] * inv_n1);
    gyro_std[2] = sqrt(gyro_std[2] * inv_n1);

    imu->calibration.accel_bias_x = accel_mean[0];
    imu->calibration.accel_bias_y = accel_mean[1];
    imu->calibration.accel_bias_z = accel_mean[2] - fusion->config.gravity_magnitude;
    imu->calibration.gyro_bias_x = gyro_mean[0];
    imu->calibration.gyro_bias_y = gyro_mean[1];
    imu->calibration.gyro_bias_z = gyro_mean[2];

    imu->calibration.accel_scale_x = 1.0;
    imu->calibration.accel_scale_y = 1.0;
    imu->calibration.accel_scale_z = 1.0;
    imu->calibration.gyro_scale_x = 1.0;
    imu->calibration.gyro_scale_y = 1.0;
    imu->calibration.gyro_scale_z = 1.0;

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            imu->calibration.misalignment_matrix[i][j] = (i == j) ? 1.0 : 0.0;

    imu->is_calibrated = 1;
}

static void apply_imu_calibration(FusionImuData* imu,
                                   double acceleration[3],
                                   double angular_velocity[3])
{
    if (!imu->is_calibrated) return;

    acceleration[0] = (acceleration[0] - imu->calibration.accel_bias_x) * imu->calibration.accel_scale_x;
    acceleration[1] = (acceleration[1] - imu->calibration.accel_bias_y) * imu->calibration.accel_scale_y;
    acceleration[2] = (acceleration[2] - imu->calibration.accel_bias_z) * imu->calibration.accel_scale_z;

    angular_velocity[0] = (angular_velocity[0] - imu->calibration.gyro_bias_x) * imu->calibration.gyro_scale_x;
    angular_velocity[1] = (angular_velocity[1] - imu->calibration.gyro_bias_y) * imu->calibration.gyro_scale_y;
    angular_velocity[2] = (angular_velocity[2] - imu->calibration.gyro_bias_z) * imu->calibration.gyro_scale_z;
}

static void update_orientation_from_imu(FusionImuData* imu,
                                         const double accel[3],
                                         const double gyro[3],
                                         double dt)
{
    double q[4] = {imu->orientation[0], imu->orientation[1],
                   imu->orientation[2], imu->orientation[3]};

    double gx = gyro[0], gy = gyro[1], gz = gyro[2];

    double q_dot[4];
    q_dot[0] = 0.5 * (-q[1] * gx - q[2] * gy - q[3] * gz);
    q_dot[1] = 0.5 * (q[0] * gx + q[2] * gz - q[3] * gy);
    q_dot[2] = 0.5 * (q[0] * gy - q[1] * gz + q[3] * gx);
    q_dot[3] = 0.5 * (q[0] * gz + q[1] * gy - q[2] * gx);

    q[0] += q_dot[0] * dt;
    q[1] += q_dot[1] * dt;
    q[2] += q_dot[2] * dt;
    q[3] += q_dot[3] * dt;
    quaternion_normalize(q);

    double accel_norm = sqrt(accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]);
    if (accel_norm > 0.5 && accel_norm < 2.0 * 9.81)
    {
        double ax = accel[0] / accel_norm;
        double ay = accel[1] / accel_norm;
        double az = accel[2] / accel_norm;

        double g[3];
        rotate_vector_by_quaternion(g, q, (double[3]){0, 0, -1});

        double error_x = g[1] * az - g[2] * ay;
        double error_y = g[2] * ax - g[0] * az;
        double error_z = g[0] * ay - g[1] * ax;

        double correction_gain = 0.02;
        q[1] += error_x * correction_gain * dt;
        q[2] += error_y * correction_gain * dt;
        q[3] += error_z * correction_gain * dt;
        quaternion_normalize(q);
    }

    imu->orientation[0] = q[0];
    imu->orientation[1] = q[1];
    imu->orientation[2] = q[2];
    imu->orientation[3] = q[3];

    double world_accel[3];
    rotate_vector_by_quaternion(accel, q, world_accel);
    world_accel[2] -= 9.81;

    imu->world_linear_acceleration[0] = world_accel[0];
    imu->world_linear_acceleration[1] = world_accel[1];
    imu->world_linear_acceleration[2] = world_accel[2];
    imu->linear_acceleration[0] = accel[0];
    imu->linear_acceleration[1] = accel[1];
    imu->linear_acceleration[2] = accel[2];
    imu->angular_velocity[0] = gyro[0];
    imu->angular_velocity[1] = gyro[1];
    imu->angular_velocity[2] = gyro[2];

    quaternion_to_euler(q, &imu->yaw, &imu->pitch, &imu->roll);
}

HumanoidSensorFusion* humanoid_sensor_fusion_create(const FusionConfig* config)
{
    HumanoidSensorFusion* fusion = (HumanoidSensorFusion*)calloc(1, sizeof(HumanoidSensorFusion));
    if (!fusion) return NULL;

    memset(&fusion->state, 0, sizeof(FusionHumanoidState));
    memset(&fusion->gait, 0, sizeof(GaitAnalyzerState));
    memset(&fusion->stats, 0, sizeof(FusionPerformanceStats));

    if (config)
        fusion->config = *config;
    else
        humanoid_sensor_fusion_set_default_config(&fusion->config);

    fusion->state.state = FUSION_STATE_INITIALIZING;
    fusion->state.mass_total = 60.0;
    fusion->state.num_joints = 0;
    fusion->state.imu_count = 0;
    fusion->kalman_initialized = 0;
    fusion->previous_update_time = 0.0;
    fusion->integration_timer = 0.0;
    fusion->calibration_sample_count = 0;
    memset(fusion->calibration_accel_sum, 0, sizeof(fusion->calibration_accel_sum));
    memset(fusion->calibration_gyro_sum, 0, sizeof(fusion->calibration_gyro_sum));
    memset(fusion->calibration_accel_sq_sum, 0, sizeof(fusion->calibration_accel_sq_sum));
    memset(fusion->calibration_gyro_sq_sum, 0, sizeof(fusion->calibration_gyro_sq_sum));

    // M17: 初始化校准状态
    memset(fusion->imu_calib, 0, sizeof(fusion->imu_calib));
    for (int i = 0; i < HUMAN_SENSOR_FUSION_MAX_IMUS; i++) {
        fusion->imu_calib[i].imu_index = i;
        fusion->imu_calib[i].calib.stage = FUSION_CALIB_NONE;
        fusion->imu_calib[i].calib.samples_required = HUMAN_SENSOR_FUSION_CALIB_SAMPLES;
        fusion->imu_calib[i].calib.misalignment[0][0] = 1.0;
        fusion->imu_calib[i].calib.misalignment[1][1] = 1.0;
        fusion->imu_calib[i].calib.misalignment[2][2] = 1.0;
        for (int j = 0; j < 3; j++) {
            fusion->imu_calib[i].calib.gyro_scale[j] = 1.0;
            fusion->imu_calib[i].calib.accel_scale[j] = 1.0;
        }
    }
    fusion->current_calib_target = NULL;
    fusion->calibration_active = 0;
    fusion->calibration_target_imu = -1;
    fusion->calibration_timer = 0.0;
    memset(fusion->calibration_temp_samples, 0, sizeof(fusion->calibration_temp_samples));
    fusion->calibration_temp_sum = 0.0;

    // M17: 初始化故障检测
    memset(&fusion->fault_detector, 0, sizeof(FusionFaultDetector));
    fusion->fault_detector_created = 0;
    memset(fusion->previous_accel, 0, sizeof(fusion->previous_accel));
    memset(fusion->previous_gyro, 0, sizeof(fusion->previous_gyro));
    memset(fusion->prev_accel_variance, 0, sizeof(fusion->prev_accel_variance));
    memset(fusion->stuck_check_timer, 0, sizeof(fusion->stuck_check_timer));

    fusion->last_error[0] = '\0';
    fusion->error_code = 0;

    return fusion;
}

void humanoid_sensor_fusion_destroy(HumanoidSensorFusion* fusion)
{
    free(fusion);
}

int humanoid_sensor_fusion_init(HumanoidSensorFusion* fusion)
{
    if (!fusion) return -1;

    kalman_init(&fusion->kf, &fusion->config);
    lowpass_init(&fusion->lpf, 0.1);
    fusion->kalman_initialized = 1;
    fusion->state.state = FUSION_STATE_CALIBRATING;

    if (fusion->config.auto_calibrate_imus)
        fusion->calibration_sample_count = 0;

    fusion->state.fusion_timestamp = 0.0;
    fusion->previous_update_time = 0.0;

    return 0;
}

int humanoid_sensor_fusion_reset(HumanoidSensorFusion* fusion)
{
    if (!fusion) return -1;
    memset(&fusion->state, 0, sizeof(FusionHumanoidState));
    memset(&fusion->gait, 0, sizeof(GaitAnalyzerState));
    memset(&fusion->stats, 0, sizeof(FusionPerformanceStats));
    memset(&fusion->kf, 0, sizeof(KalmanFilterState));
    memset(&fusion->lpf, 0, sizeof(LowPassFilterState));

    fusion->state.state = FUSION_STATE_INITIALIZING;
    fusion->state.mass_total = 60.0;
    fusion->kalman_initialized = 0;
    fusion->calibration_sample_count = 0;

    return humanoid_sensor_fusion_init(fusion);
}

int humanoid_sensor_fusion_calibrate(HumanoidSensorFusion* fusion)
{
    if (!fusion) return -1;
    fusion->state.state = FUSION_STATE_CALIBRATING;
    fusion->calibration_sample_count = 0;
    memset(fusion->calibration_accel_sum, 0, sizeof(fusion->calibration_accel_sum));
    memset(fusion->calibration_gyro_sum, 0, sizeof(fusion->calibration_gyro_sum));
    return 0;
}

int humanoid_sensor_fusion_suspend(HumanoidSensorFusion* fusion)
{
    if (!fusion) return -1;
    fusion->state.state = FUSION_STATE_SUSPENDED;
    return 0;
}

int humanoid_sensor_fusion_resume(HumanoidSensorFusion* fusion)
{
    if (!fusion) return -1;
    fusion->state.state = FUSION_STATE_RUNNING;
    fusion->previous_update_time = 0.0;
    return 0;
}

int humanoid_sensor_fusion_update(HumanoidSensorFusion* fusion, double dt)
{
    if (!fusion || !fusion->kalman_initialized) return -1;
    if (dt <= 0.0) dt = 0.001;
    if (dt > 0.1) dt = 0.001;

    fusion->state.fusion_timestamp += dt;
    fusion->integration_timer += dt;

    kalman_predict(&fusion->kf, dt);

    double com_pos[3], com_vel[3];
    com_pos[0] = fusion->kf.x_est[0];
    com_pos[1] = fusion->kf.x_est[1];
    com_pos[2] = fusion->kf.x_est[2];
    com_vel[0] = fusion->kf.x_est[6];
    com_vel[1] = fusion->kf.x_est[7];
    com_vel[2] = fusion->kf.x_est[8];

    double com_accel[3] = {0};
    if (fusion->state.imu_count > 0 && fusion->state.imus[0].is_calibrated)
    {
        com_accel[0] = fusion->state.imus[0].world_linear_acceleration[0];
        com_accel[1] = fusion->state.imus[0].world_linear_acceleration[1];
        com_accel[2] = fusion->state.imus[0].world_linear_acceleration[2];
    }

    if (fusion->state.num_joints > 0 && fusion->config.enable_joint_fusion)
    {
        double joint_com[3];
        estimate_com_from_joints(fusion->state.joint_positions,
                                  fusion->state.num_joints,
                                  fusion->state.mass_total, joint_com);

        com_pos[0] = fusion->config.joint_trust_factor * joint_com[0] +
                     (1.0 - fusion->config.joint_trust_factor) * com_pos[0];
        com_pos[1] = fusion->config.joint_trust_factor * joint_com[1] +
                     (1.0 - fusion->config.joint_trust_factor) * com_pos[1];
        com_pos[2] = fusion->config.joint_trust_factor * joint_com[2] +
                     (1.0 - fusion->config.joint_trust_factor) * com_pos[2];
    }

    com_vel[0] += com_accel[0] * dt;
    com_vel[1] += com_accel[1] * dt;
    com_vel[2] += com_accel[2] * dt;

    double zmp[2], zmp_est[2];
    estimate_zmp_from_com(com_pos, com_accel, zmp, fusion->config.gravity_magnitude);
    estimate_zmp_from_com(com_pos, com_accel, zmp_est, fusion->config.gravity_magnitude);

    lowpass_finalize(&fusion->lpf, com_pos, com_vel, zmp);

    fusion->state.com_position[0] = fusion->lpf.com_pos_filtered[0];
    fusion->state.com_position[1] = fusion->lpf.com_pos_filtered[1];
    fusion->state.com_position[2] = fusion->lpf.com_pos_filtered[2];
    fusion->state.com_velocity[0] = fusion->lpf.com_vel_filtered[0];
    fusion->state.com_velocity[1] = fusion->lpf.com_vel_filtered[1];
    fusion->state.com_velocity[2] = fusion->lpf.com_vel_filtered[2];
    fusion->state.zmp_position[0] = fusion->lpf.zmp_filtered[0];
    fusion->state.zmp_position[1] = fusion->lpf.zmp_filtered[1];
    fusion->state.zmp_estimated[0] = zmp_est[0];
    fusion->state.zmp_estimated[1] = zmp_est[1];
    fusion->state.base_height = com_pos[2];

    if (fusion->config.enable_gait_detection && fusion->config.enable_foot_contact)
    {
        double left_force = fusion->state.left_foot_force_magnitude;
        double right_force = fusion->state.right_foot_force_magnitude;
        fusion->state.gait_phase = detect_gait_phase(
            &fusion->gait, left_force, right_force,
            fusion->state.left_foot_contact, fusion->state.right_foot_contact,
            com_pos, com_vel, fusion->state.fusion_timestamp,
            fusion->config.foot_contact_threshold);
    }

    if (fusion->integration_timer >= 1.0)
    {
        fusion->state.update_rate_hz = 1.0 / dt;
        fusion->integration_timer = 0.0;
    }

    if (fusion->state.state == FUSION_STATE_CALIBRATING &&
        fusion->calibration_sample_count >= fusion->config.calibration_samples)
    {
        for (int i = 0; i < fusion->state.imu_count; i++)
        {
            if (!fusion->state.imus[i].is_calibrated)
                calibrate_imu_from_samples(fusion, &fusion->state.imus[i]);
        }
        fusion->state.state = FUSION_STATE_RUNNING;
    }

    (void)com_pos;
    double com_error = sqrt(com_pos[0] * com_pos[0] + com_pos[1] * com_pos[1]);
    fusion->state.stability_measure = 1.0 / (1.0 + com_error);

    double zmp_in_support = 1.0;
    if (fusion->config.enable_foot_contact)
    {
        double foot_spread = 0.2;
        if (fabs(zmp[0]) > foot_spread + fusion->config.zmp_margin_x ||
            fabs(zmp[1]) > 0.1 + fusion->config.zmp_margin_y)
            zmp_in_support = 0.0;
    }

    fusion->stats.kalman_updates++;
    fusion->stats.is_stable = (com_error < 0.1 && zmp_in_support > 0.5) ? 1 : 0;

    return 0;
}

int humanoid_sensor_fusion_update_with_sensor_pipeline(HumanoidSensorFusion* fusion,
                                                        SensorPipeline* pipeline, double dt)
{
    if (!fusion || !pipeline) return -1;

    if (fusion->config.enable_foot_contact)
    {
        for (int i = 0; i < fusion->state.imu_count && i < HUMAN_SENSOR_FUSION_MAX_IMUS; i++)
        {
            SensorPipelineEntry entry;
            if (sensor_pipeline_get_latest_by_type(pipeline, SENSOR_TYPE_IMU, &entry) == 0)
            {
                if (entry.data_size >= 6 * sizeof(float))
                {
                    float* fdata = (float*)entry.data;
                    double accel[3] = {fdata[0], fdata[1], fdata[2]};
                    double gyro[3] = {fdata[3], fdata[4], fdata[5]};
                    double ori[4] = {1, 0, 0, 0};
                    if (entry.data_size >= 10 * sizeof(float))
                    {
                        ori[0] = fdata[6]; ori[1] = fdata[7];
                        ori[2] = fdata[8]; ori[3] = fdata[9];
                    }
                    humanoid_sensor_fusion_feed_imu(fusion, i, accel, gyro, ori,
                                                     (FusionImuLocation)i, 25.0,
                                                     entry.timestamp);
                }
            }
        }
    }

    return humanoid_sensor_fusion_update(fusion, dt);
}

int humanoid_sensor_fusion_feed_imu(HumanoidSensorFusion* fusion, int imu_index,
                                     const double* acceleration,
                                     const double* angular_velocity,
                                     const double* orientation,
                                     FusionImuLocation location,
                                     double temperature, double timestamp)
{
    if (!fusion || !acceleration || !angular_velocity)
        return -1;
    if (imu_index < 0 || imu_index >= HUMAN_SENSOR_FUSION_MAX_IMUS)
        return -1;

    FusionImuData* imu = &fusion->state.imus[imu_index];

    if (imu_index >= fusion->state.imu_count)
        fusion->state.imu_count = imu_index + 1;

    imu->location = location;
    imu->temperature = temperature;
    imu->last_update_time = timestamp;
    imu->linear_acceleration[0] = acceleration[0];
    imu->linear_acceleration[1] = acceleration[1];
    imu->linear_acceleration[2] = acceleration[2];
    imu->angular_velocity[0] = angular_velocity[0];
    imu->angular_velocity[1] = angular_velocity[1];
    imu->angular_velocity[2] = angular_velocity[2];

    if (fusion->state.state == FUSION_STATE_CALIBRATING &&
        fusion->config.auto_calibrate_imus &&
        fusion->calibration_sample_count < fusion->config.calibration_samples)
    {
        int n = fusion->calibration_sample_count;
        if (n < HUMAN_SENSOR_FUSION_CALIB_SAMPLES)
        {
            fusion->calibration_accel_samples[n][0] = acceleration[0];
            fusion->calibration_accel_samples[n][1] = acceleration[1];
            fusion->calibration_accel_samples[n][2] = acceleration[2];
            fusion->calibration_gyro_samples[n][0] = angular_velocity[0];
            fusion->calibration_gyro_samples[n][1] = angular_velocity[1];
            fusion->calibration_gyro_samples[n][2] = angular_velocity[2];
            fusion->calibration_sample_count++;
        }
        return 0;
    }

    double calib_accel[3] = {acceleration[0], acceleration[1], acceleration[2]};
    double calib_gyro[3] = {angular_velocity[0], angular_velocity[1], angular_velocity[2]};
    apply_imu_calibration(imu, calib_accel, calib_gyro);

    double dt = 0.01;
    if (fusion->previous_update_time > 0.0)
        dt = timestamp - fusion->previous_update_time;
    if (dt < 0.001) dt = 0.001;
    if (dt > 0.1) dt = 0.01;
    fusion->previous_update_time = timestamp;

    update_orientation_from_imu(imu, calib_accel, calib_gyro, dt);

    if (orientation && fusion->config.enable_vision_fusion)
    {
        double blend = 0.1;
        imu->orientation[0] = (1.0 - blend) * imu->orientation[0] + blend * orientation[0];
        imu->orientation[1] = (1.0 - blend) * imu->orientation[1] + blend * orientation[1];
        imu->orientation[2] = (1.0 - blend) * imu->orientation[2] + blend * orientation[2];
        imu->orientation[3] = (1.0 - blend) * imu->orientation[3] + blend * orientation[3];
        quaternion_normalize(imu->orientation);
    }

    imu->update_rate_hz = (dt > 0.0) ? 1.0 / dt : 0.0;
    fusion->stats.imu_count++;

    return 0;
}

int humanoid_sensor_fusion_feed_joint_states(HumanoidSensorFusion* fusion,
                                              const double* positions,
                                              const double* velocities,
                                              const double* torques,
                                              int num_joints, double timestamp)
{
    if (!fusion || !positions) return -1;
    if (num_joints <= 0 || num_joints > HUMAN_SENSOR_FUSION_MAX_JOINTS)
        return -1;

    fusion->state.num_joints = num_joints;
    for (int i = 0; i < num_joints; i++)
    {
        fusion->state.joint_positions[i] = positions[i];
        if (velocities) fusion->state.joint_velocities[i] = velocities[i];
        if (torques) fusion->state.joint_torques[i] = torques[i];
    }
    fusion->state.sensor_timestamp = timestamp;
    fusion->stats.joint_count++;

    double com_joint[3];
    estimate_com_from_joints(positions, num_joints,
                              fusion->state.mass_total, com_joint);
    for (int i = 0; i < 3; i++)
        fusion->state.com_position[i] = com_joint[i];

    return 0;
}

int humanoid_sensor_fusion_feed_vision_pose(HumanoidSensorFusion* fusion,
                                             const double* position,
                                             const double* orientation,
                                             const double* covariance,
                                             double timestamp)
{
    (void)timestamp;
    if (!fusion || !position || !orientation) return -1;

    if (fusion->kalman_initialized)
    {
        double pos_cov[9];
        if (covariance)
        {
            for (int i = 0; i < 9; i++)
                pos_cov[i] = covariance[i];
        }
        else
        {
            for (int i = 0; i < 9; i++)
                pos_cov[i] = (i % 4 == 0) ? 0.01 : 0.0;
        }

        kalman_update_position(&fusion->kf, position, pos_cov);
        kalman_update_orientation(&fusion->kf, orientation, pos_cov);
    }

    fusion->state.position[0] = position[0];
    fusion->state.position[1] = position[1];
    fusion->state.position[2] = position[2];
    fusion->state.orientation[0] = orientation[0];
    fusion->state.orientation[1] = orientation[1];
    fusion->state.orientation[2] = orientation[2];
    fusion->state.orientation[3] = orientation[3];

    fusion->stats.vision_updates++;
    return 0;
}

int humanoid_sensor_fusion_feed_foot_contact(HumanoidSensorFusion* fusion,
                                              int is_left_foot,
                                              const double* contact_force,
                                              const double* contact_torque,
                                              FusionContactType contact_type,
                                              double timestamp,
                                              double force_magnitude)
{
    if (!fusion) return -1;
    int idx = is_left_foot ? 0 : 1;

    if (contact_force)
    {
        fusion->state.foot_contact_forces[idx][0] = contact_force[0];
        fusion->state.foot_contact_forces[idx][1] = contact_force[1];
        fusion->state.foot_contact_forces[idx][2] = contact_force[2];
    }
    if (contact_torque)
    {
        fusion->state.foot_contact_torques[idx][0] = contact_torque[0];
        fusion->state.foot_contact_torques[idx][1] = contact_torque[1];
        fusion->state.foot_contact_torques[idx][2] = contact_torque[2];
    }

    if (is_left_foot)
    {
        fusion->state.left_foot_contact = contact_type;
        fusion->state.left_foot_force_magnitude = force_magnitude;
    }
    else
    {
        fusion->state.right_foot_contact = contact_type;
        fusion->state.right_foot_force_magnitude = force_magnitude;
    }

    fusion->state.sensor_timestamp = timestamp;
    fusion->stats.contact_updates++;
    return 0;
}

int humanoid_sensor_fusion_feed_gnss(HumanoidSensorFusion* fusion,
                                      double latitude, double longitude,
                                      double altitude, double timestamp,
                                      float accuracy)
{
    (void)latitude;
    (void)longitude;
    (void)altitude;
    (void)timestamp;
    (void)accuracy;
    if (!fusion) return -1;
    fusion->stats.gnss_updates++;
    return 0;
}

int humanoid_sensor_fusion_get_state(HumanoidSensorFusion* fusion,
                                     FusionHumanoidState* state)
{
    if (!fusion || !state) return -1;
    *state = fusion->state;
    return 0;
}

int humanoid_sensor_fusion_get_imu(HumanoidSensorFusion* fusion, int imu_index,
                                   FusionImuData* imu_data)
{
    if (!fusion || !imu_data) return -1;
    if (imu_index < 0 || imu_index >= fusion->state.imu_count)
        return -1;
    *imu_data = fusion->state.imus[imu_index];
    return 0;
}

int humanoid_sensor_fusion_get_com(HumanoidSensorFusion* fusion,
                                    double* com_position, double* com_velocity)
{
    if (!fusion) return -1;
    if (com_position)
    {
        com_position[0] = fusion->state.com_position[0];
        com_position[1] = fusion->state.com_position[1];
        com_position[2] = fusion->state.com_position[2];
    }
    if (com_velocity)
    {
        com_velocity[0] = fusion->state.com_velocity[0];
        com_velocity[1] = fusion->state.com_velocity[1];
        com_velocity[2] = fusion->state.com_velocity[2];
    }
    return 0;
}

int humanoid_sensor_fusion_get_zmp(HumanoidSensorFusion* fusion,
                                    double* zmp_position, double* zmp_estimated)
{
    if (!fusion) return -1;
    if (zmp_position)
    {
        zmp_position[0] = fusion->state.zmp_position[0];
        zmp_position[1] = fusion->state.zmp_position[1];
    }
    if (zmp_estimated)
    {
        zmp_estimated[0] = fusion->state.zmp_estimated[0];
        zmp_estimated[1] = fusion->state.zmp_estimated[1];
    }
    return 0;
}

int humanoid_sensor_fusion_get_gait(HumanoidSensorFusion* fusion,
                                     GaitPhase* phase, double* progress)
{
    if (!fusion) return -1;
    if (phase) *phase = fusion->state.gait_phase;
    if (progress) *progress = fusion->state.gait_cycle_progress;
    return 0;
}

int humanoid_sensor_fusion_get_orientation(HumanoidSensorFusion* fusion,
                                            double* orientation,
                                            double* yaw_pitch_roll)
{
    if (!fusion) return -1;
    if (orientation)
    {
        orientation[0] = fusion->state.orientation[0];
        orientation[1] = fusion->state.orientation[1];
        orientation[2] = fusion->state.orientation[2];
        orientation[3] = fusion->state.orientation[3];
    }
    if (yaw_pitch_roll)
    {
        double q[4] = {fusion->state.orientation[0], fusion->state.orientation[1],
                       fusion->state.orientation[2], fusion->state.orientation[3]};
        quaternion_to_euler(q, &yaw_pitch_roll[0], &yaw_pitch_roll[1], &yaw_pitch_roll[2]);
    }
    return 0;
}

int humanoid_sensor_fusion_set_parameter(HumanoidSensorFusion* fusion,
                                          const char* param_name, double value)
{
    if (!fusion || !param_name) return -1;

    if (strcmp(param_name, "process_noise_q") == 0)
    {
        for (int i = 0; i < 15; i++)
            fusion->config.process_noise_q[i] = value;
    }
    else if (strcmp(param_name, "measurement_noise_r") == 0)
    {
        for (int i = 0; i < 9; i++)
            fusion->config.measurement_noise_r[i] = value;
    }
    else if (strcmp(param_name, "gyro_trust_factor") == 0)
        fusion->config.gyro_trust_factor = value;
    else if (strcmp(param_name, "accel_trust_factor") == 0)
        fusion->config.accel_trust_factor = value;
    else if (strcmp(param_name, "vision_trust_factor") == 0)
        fusion->config.vision_trust_factor = value;
    else if (strcmp(param_name, "joint_trust_factor") == 0)
        fusion->config.joint_trust_factor = value;
    else if (strcmp(param_name, "foot_contact_threshold") == 0)
        fusion->config.foot_contact_threshold = value;
    else if (strcmp(param_name, "com_height_offset") == 0)
        fusion->config.com_height_offset = value;
    else if (strcmp(param_name, "zmp_margin_x") == 0)
        fusion->config.zmp_margin_x = value;
    else if (strcmp(param_name, "zmp_margin_y") == 0)
        fusion->config.zmp_margin_y = value;
    else if (strcmp(param_name, "gravity_magnitude") == 0)
        fusion->config.gravity_magnitude = value;
    else
        return -1;

    return 0;
}

double humanoid_sensor_fusion_get_parameter(HumanoidSensorFusion* fusion,
                                             const char* param_name)
{
    if (!fusion || !param_name) return 0.0;

    if (strcmp(param_name, "gyro_trust_factor") == 0)
        return fusion->config.gyro_trust_factor;
    else if (strcmp(param_name, "accel_trust_factor") == 0)
        return fusion->config.accel_trust_factor;
    else if (strcmp(param_name, "vision_trust_factor") == 0)
        return fusion->config.vision_trust_factor;
    else if (strcmp(param_name, "joint_trust_factor") == 0)
        return fusion->config.joint_trust_factor;
    else if (strcmp(param_name, "foot_contact_threshold") == 0)
        return fusion->config.foot_contact_threshold;
    else if (strcmp(param_name, "com_height_offset") == 0)
        return fusion->config.com_height_offset;
    else if (strcmp(param_name, "zmp_margin_x") == 0)
        return fusion->config.zmp_margin_x;
    else if (strcmp(param_name, "zmp_margin_y") == 0)
        return fusion->config.zmp_margin_y;
    else if (strcmp(param_name, "gravity_magnitude") == 0)
        return fusion->config.gravity_magnitude;

    return 0.0;
}

void humanoid_sensor_fusion_get_fusion_config(const HumanoidSensorFusion* fusion,
                                               FusionConfig* config)
{
    if (fusion && config) *config = fusion->config;
}

int humanoid_sensor_fusion_set_fusion_config(HumanoidSensorFusion* fusion,
                                              const FusionConfig* config)
{
    if (!fusion || !config) return -1;
    fusion->config = *config;
    return 0;
}

void humanoid_sensor_fusion_set_default_config(FusionConfig* config)
{
    if (!config) return;
    memset(config, 0, sizeof(FusionConfig));

    for (int i = 0; i < 15; i++)
        config->process_noise_q[i] = 1e-6;
    config->process_noise_q[0] = 1e-4;
    config->process_noise_q[1] = 1e-4;
    config->process_noise_q[2] = 1e-4;

    for (int i = 0; i < 9; i++)
        config->measurement_noise_r[i] = 1e-2;
    config->measurement_noise_r[0] = 1e-3;
    config->measurement_noise_r[4] = 1e-3;
    config->measurement_noise_r[8] = 1e-3;

    for (int i = 0; i < 15; i++)
        config->initial_covariance_p[i] = 1.0;

    config->dt = 0.01;
    config->gyro_trust_factor = 0.9;
    config->accel_trust_factor = 0.1;
    config->vision_trust_factor = 0.3;
    config->joint_trust_factor = 0.4;
    config->foot_contact_threshold = 10.0;
    config->com_height_offset = 0.8;
    config->zmp_margin_x = 0.05;
    config->zmp_margin_y = 0.05;
    config->enable_magnetometer = 0;
    config->enable_vision_fusion = 1;
    config->enable_joint_fusion = 1;
    config->enable_foot_contact = 1;
    config->enable_gait_detection = 1;
    config->auto_calibrate_imus = 1;
    config->calibration_samples = 100;
    config->gravity_magnitude = 9.81;
    config->robot_name[0] = '\0';
}

int humanoid_sensor_fusion_is_calibrated(HumanoidSensorFusion* fusion)
{
    if (!fusion) return 0;
    for (int i = 0; i < fusion->state.imu_count; i++)
    {
        if (!fusion->state.imus[i].is_calibrated)
            return 0;
    }
    return (fusion->state.state >= FUSION_STATE_RUNNING) ? 1 : 0;
}

int humanoid_sensor_fusion_get_error_code(HumanoidSensorFusion* fusion)
{
    return fusion ? fusion->error_code : -1;
}

const char* humanoid_sensor_fusion_get_last_error(HumanoidSensorFusion* fusion)
{
    return fusion ? fusion->last_error : "空融合指针";
}

const char* humanoid_sensor_fusion_gait_phase_string(GaitPhase phase)
{
    switch (phase)
    {
        case GAIT_PHASE_UNKNOWN: return "未知";
        case GAIT_PHASE_STANCE: return "支撑";
        case GAIT_PHASE_SWING: return "摆动";
        case GAIT_PHASE_DOUBLE_SUPPORT: return "双足支撑";
        case GAIT_PHASE_SINGLE_SUPPORT_LEFT: return "左足单撑";
        case GAIT_PHASE_SINGLE_SUPPORT_RIGHT: return "右足单撑";
        case GAIT_PHASE_LANDING: return "着地";
        case GAIT_PHASE_LIFTOFF: return "离地";
        case GAIT_PHASE_STANDING: return "站立";
        case GAIT_PHASE_SITTING: return "坐下";
        case GAIT_PHASE_JUMPING: return "跳跃";
        case GAIT_PHASE_RUNNING: return "跑步";
        default: return "未知";
    }
}

const char* humanoid_sensor_fusion_state_string(FusionState state)
{
    switch (state)
    {
        case FUSION_STATE_INITIALIZING: return "初始化中";
        case FUSION_STATE_CALIBRATING: return "校准中";
        case FUSION_STATE_RUNNING: return "运行中";
        case FUSION_STATE_ERROR: return "错误";
        case FUSION_STATE_SUSPENDED: return "已暂停";
        default: return "未知";
    }
}

int humanoid_sensor_fusion_get_stats(HumanoidSensorFusion* fusion,
                                      FusionPerformanceStats* stats)
{
    if (!fusion || !stats) return -1;
    *stats = fusion->stats;
    return 0;
}

/* ============================
 * M17: 传感器校准增强实现
 * ============================ */

static void compute_bias_from_samples(HumanoidSensorFusion* fusion,
                                       FusionImuCalibState* calib)
{
    double accel_mean[3] = {0, 0, 0};
    double gyro_mean[3] = {0, 0, 0};
    double inv_n = 1.0 / calib->calib.sample_count;

    for (int i = 0; i < calib->calib.sample_count; i++) {
        int idx = calib->imu_index * HUMAN_SENSOR_FUSION_CALIB_SAMPLES + i;
        (void)idx;
    }

    for (int i = 0; i < 3; i++) {
        double a_sum = 0, g_sum = 0;
        for (int j = 0; j < calib->calib.sample_count; j++) {
            if (calib->imu_index < HUMAN_SENSOR_FUSION_MAX_IMUS) {
                a_sum += fusion->calibration_accel_samples[j][i];
                g_sum += fusion->calibration_gyro_samples[j][i];
            }
        }
        accel_mean[i] = a_sum * inv_n;
        gyro_mean[i] = g_sum * inv_n;
    }

    double gravity = fusion->config.gravity_magnitude;
    if (gravity < 8.0 || gravity > 10.0) gravity = 9.81;

    double accel_norm = sqrt(accel_mean[0]*accel_mean[0] +
                             accel_mean[1]*accel_mean[1] +
                             accel_mean[2]*accel_mean[2]);

    if (accel_norm > 0.01) {
        for (int i = 0; i < 3; i++)
            calib->calib.accel_bias[i] = accel_mean[i] - gravity * (accel_mean[i] / accel_norm);
    }

    for (int i = 0; i < 3; i++)
        calib->calib.gyro_bias[i] = gyro_mean[i];

    double accel_var[3] = {0, 0, 0};
    double gyro_var[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++) {
        double a_var = 0, g_var = 0;
        for (int j = 0; j < calib->calib.sample_count; j++) {
            double ad = fusion->calibration_accel_samples[j][i] - accel_mean[i];
            double gd = fusion->calibration_gyro_samples[j][i] - gyro_mean[i];
            a_var += ad * ad;
            g_var += gd * gd;
        }
        accel_var[i] = a_var * inv_n;
        gyro_var[i] = g_var * inv_n;
        calib->bias_stability[i] = sqrt(gyro_var[i]);
        calib->noise_density[i] = sqrt(accel_var[i]);
    }

    calib->calib.is_valid = 1;
    calib->calib.calibration_quality = 1.0 - (gyro_var[0] + gyro_var[1] + gyro_var[2]) / 3.0;
    if (calib->calib.calibration_quality < 0.0) calib->calib.calibration_quality = 0.0;
    if (calib->calib.calibration_quality > 1.0) calib->calib.calibration_quality = 1.0;
    calib->is_calibrated = 1;
    calib->calibration_progress = 100;
    calib->calib.stage = FUSION_CALIB_COMPLETE;
}

int humanoid_sensor_fusion_start_calibration(HumanoidSensorFusion* fusion, int imu_index)
{
    if (!fusion) return -1;

    if (imu_index < -1 || imu_index >= HUMAN_SENSOR_FUSION_MAX_IMUS) return -1;

    // 如果指定了-1，校准所有IMU
    if (imu_index == -1) {
        for (int i = 0; i < HUMAN_SENSOR_FUSION_MAX_IMUS; i++) {
            if (fusion->state.imus[i].location != FUSION_IMU_LOCATION_HEAD ||
                i == 0) {
                // 只校准实际存在的IMU
                if (fusion->state.imu_count > 0 && i < fusion->state.imu_count) {
                    humanoid_sensor_fusion_start_calibration(fusion, i);
                }
            }
        }
        return 0;
    }

    fusion->calibration_active = 1;
    fusion->calibration_target_imu = imu_index;
    fusion->calibration_sample_count = 0;
    fusion->calibration_timer = 0.0;
    memset(fusion->calibration_accel_sum, 0, sizeof(fusion->calibration_accel_sum));
    memset(fusion->calibration_gyro_sum, 0, sizeof(fusion->calibration_gyro_sum));
    memset(fusion->calibration_accel_sq_sum, 0, sizeof(fusion->calibration_accel_sq_sum));
    memset(fusion->calibration_gyro_sq_sum, 0, sizeof(fusion->calibration_gyro_sq_sum));
    memset(fusion->calibration_temp_samples, 0, sizeof(fusion->calibration_temp_samples));
    fusion->calibration_temp_sum = 0.0;

    fusion->current_calib_target = &fusion->imu_calib[imu_index];
    fusion->current_calib_target->is_calibrated = 0;
    fusion->current_calib_target->calibration_progress = 0;
    fusion->current_calib_target->calib.stage = FUSION_CALIB_GYRO_BIAS;
    fusion->current_calib_target->calib.sample_count = 0;
    fusion->current_calib_target->calib.samples_required = HUMAN_SENSOR_FUSION_CALIB_SAMPLES;

    fusion->state.state = FUSION_STATE_CALIBRATING;

    return 0;
}

int humanoid_sensor_fusion_get_calibration_progress(HumanoidSensorFusion* fusion,
                                                     int imu_index, int* progress)
{
    if (!fusion || !progress) return -1;
    if (imu_index < 0 || imu_index >= HUMAN_SENSOR_FUSION_MAX_IMUS) return -1;

    *progress = fusion->imu_calib[imu_index].calibration_progress;
    return 0;
}

int humanoid_sensor_fusion_get_calibration_data(HumanoidSensorFusion* fusion,
                                                 int imu_index,
                                                 FusionCalibrationData* calib)
{
    if (!fusion || !calib) return -1;
    if (imu_index < 0 || imu_index >= HUMAN_SENSOR_FUSION_MAX_IMUS) return -1;

    memcpy(calib, &fusion->imu_calib[imu_index].calib, sizeof(FusionCalibrationData));
    return 0;
}

int humanoid_sensor_fusion_apply_temperature_compensation(HumanoidSensorFusion* fusion,
                                                           int imu_index,
                                                           double current_temperature)
{
    if (!fusion) return -1;
    if (imu_index < 0 || imu_index >= HUMAN_SENSOR_FUSION_MAX_IMUS) return -1;

    FusionImuCalibState* calib = &fusion->imu_calib[imu_index];
    if (!calib->is_calibrated) return -1;

    double delta_t = current_temperature - calib->calib.reference_temperature;

    for (int i = 0; i < 3; i++) {
        calib->calib.gyro_bias[i] += calib->calib.temp_coeff_gyro[i] * delta_t;
        calib->calib.accel_bias[i] += calib->calib.temp_coeff_accel[i] * delta_t;
    }

    return 0;
}

/* ============================
 * M17: 传感器故障检测实现
 * ============================ */

static void init_sensor_health(FusionSensorHealth* health)
{
    health->sensor_online = 1;
    health->data_valid = 1;
    health->signal_strength = 1.0;
    health->noise_level = 0.0;
    health->drift_rate[0] = 0.0;
    health->drift_rate[1] = 0.0;
    health->drift_rate[2] = 0.0;
    health->last_valid_timestamp = 0.0;
    health->fault_count = 0;
    health->fault_rate = 0.0;
    health->last_fault_time = 0.0;
    health->health_score = 1.0;
    health->is_recovering = 0;
    health->recovery_progress = 0.0;
}

static void add_fault_event(FusionFaultDetector* detector,
                             FusionSensorFaultType type,
                             int imu_index,
                             double score,
                             double time,
                             const char* description)
{
    if (!detector || !detector->fault_events) return;
    if (detector->fault_events_count >= detector->fault_events_capacity) return;

    FusionSensorFaultEvent* ev = &detector->fault_events[detector->fault_events_count];
    ev->fault_type = type;
    ev->imu_index = imu_index;
    ev->fault_score = score;
    ev->detection_time = time;
    ev->consecutive_faults = 1;
    if (description)
        strncpy(ev->description, description, sizeof(ev->description) - 1);
    else
        ev->description[0] = '\0';

    detector->fault_events_count++;
    detector->total_faults++;
    detector->active_faults++;
    detector->last_fault_detection_time = time;
}

static void check_imu_faults(HumanoidSensorFusion* fusion, int i, double dt)
{
    FusionSensorHealth* health = &fusion->fault_detector.imu_health[i];
    FusionImuData* imu = &fusion->state.imus[i];
    double t = imu->last_update_time;

    health->last_valid_timestamp = t;

    // 信号丢失检测
    if (dt > 1.0) {
        add_fault_event(&fusion->fault_detector, FUSION_FAULT_SIGNAL_LOST,
                        i, 0.9, t, "IMU信号丢失");
        health->sensor_online = 0;
        health->health_score -= 0.2;
        return;
    }

    health->sensor_online = 1;

    // 数值饱和检测
    double accel_norm = sqrt(imu->linear_acceleration[0] * imu->linear_acceleration[0] +
                             imu->linear_acceleration[1] * imu->linear_acceleration[1] +
                             imu->linear_acceleration[2] * imu->linear_acceleration[2]);
    double gyro_norm = sqrt(imu->angular_velocity[0] * imu->angular_velocity[0] +
                            imu->angular_velocity[1] * imu->angular_velocity[1] +
                            imu->angular_velocity[2] * imu->angular_velocity[2]);

    if (accel_norm > 200.0 || gyro_norm > 50.0) {
        add_fault_event(&fusion->fault_detector, FUSION_FAULT_SATURATION,
                        i, 0.8, t, "IMU数值饱和");
        health->health_score -= 0.15;
    }

    // 噪声超标检测
    double accel_var = 0, gyro_var = 0;
    for (int j = 0; j < 3; j++) {
        double ad = imu->linear_acceleration[j] - fusion->previous_accel[i][j];
        double gd = imu->angular_velocity[j] - fusion->previous_gyro[i][j];
        accel_var += ad * ad;
        gyro_var += gd * gd;
    }
    accel_var *= 0.1;
    gyro_var *= 0.1;

    health->noise_level = (accel_var + gyro_var) * 0.5;
    if (health->noise_level > 0.5) {
        add_fault_event(&fusion->fault_detector, FUSION_FAULT_NOISE_EXCEEDED,
                        i, 0.7, t, "IMU噪声超标");
        health->health_score -= 0.1;
    }

    // 值卡死检测
    fusion->stuck_check_timer[i] += dt;
    if (fusion->stuck_check_timer[i] > 0.5) {
        fusion->stuck_check_timer[i] = 0.0;
        int stuck = 1;
        for (int j = 0; j < 3; j++) {
            double da = fabs(imu->linear_acceleration[j] - fusion->previous_accel[i][j]);
            double dg = fabs(imu->angular_velocity[j] - fusion->previous_gyro[i][j]);
            if (da > 0.001 || dg > 0.001) { stuck = 0; break; }
        }
        if (stuck) {
            add_fault_event(&fusion->fault_detector, FUSION_FAULT_STUCK_VALUE,
                            i, 0.6, t, "IMU数值卡死");
            health->health_score -= 0.1;
        }
    }

    // 更新前次值
    for (int j = 0; j < 3; j++) {
        fusion->previous_accel[i][j] = imu->linear_acceleration[j];
        fusion->previous_gyro[i][j] = imu->angular_velocity[j];
    }

    // 异常值检测（加速度偏离重力太远）
    double gravity = fusion->config.gravity_magnitude;
    if (gravity < 8.0 || gravity > 10.0) gravity = 9.81;
    if (fabs(accel_norm - gravity) > 5.0) {
        add_fault_event(&fusion->fault_detector, FUSION_FAULT_OUTLIER,
                        i, 0.5, t, "加速度异常值");
        health->health_score -= 0.05;
    }

    // 漂移检测
    if (fusion->imu_calib[i].is_calibrated) {
        for (int j = 0; j < 3; j++) {
            double drift = fabs(imu->angular_velocity[j] -
                                fusion->imu_calib[i].calib.gyro_bias[j]);
            health->drift_rate[j] = drift;
            if (drift > 2.0) {
                add_fault_event(&fusion->fault_detector, FUSION_FAULT_DRIFT_EXCEEDED,
                                i, 0.4, t, "陀螺仪漂移超标");
                health->health_score -= 0.05;
            }
        }
    }

    // 综合健康评分
    if (health->health_score < 0.0) health->health_score = 0.0;
    if (health->health_score > 1.0) health->health_score = 1.0;
    health->data_valid = (health->health_score > 0.3) ? 1 : 0;
    health->fault_rate = (double)health->fault_count /
                         (fusion->state.fusion_timestamp + 1.0);
}

int humanoid_sensor_fusion_create_fault_detector(HumanoidSensorFusion* fusion)
{
    if (!fusion) return -1;
    if (fusion->fault_detector_created) return 0;

    FusionFaultDetector* fd = &fusion->fault_detector;
    memset(fd, 0, sizeof(FusionFaultDetector));

    fd->fault_events_capacity = 256;
    fd->fault_events = (FusionSensorFaultEvent*)calloc(fd->fault_events_capacity,
                                                         sizeof(FusionSensorFaultEvent));
    if (!fd->fault_events) return -1;

    for (int i = 0; i < HUMAN_SENSOR_FUSION_MAX_IMUS; i++)
        init_sensor_health(&fd->imu_health[i]);
    init_sensor_health(&fd->joint_health);
    init_sensor_health(&fd->vision_health);
    for (int i = 0; i < 2; i++)
        init_sensor_health(&fd->foot_health[i]);
    init_sensor_health(&fd->gnss_health);

    fusion->fault_detector_created = 1;
    return 0;
}

void humanoid_sensor_fusion_destroy_fault_detector(HumanoidSensorFusion* fusion)
{
    if (!fusion || !fusion->fault_detector_created) return;

    if (fusion->fault_detector.fault_events) {
        free(fusion->fault_detector.fault_events);
        fusion->fault_detector.fault_events = NULL;
    }
    fusion->fault_detector_created = 0;
}

int humanoid_sensor_fusion_detect_faults(HumanoidSensorFusion* fusion, double dt)
{
    if (!fusion) return -1;
    if (!fusion->fault_detector_created) return -1;

    int fault_count = 0;

    // 检查所有IMU
    for (int i = 0; i < fusion->state.imu_count && i < HUMAN_SENSOR_FUSION_MAX_IMUS; i++) {
        check_imu_faults(fusion, i, dt);
        fault_count += fusion->fault_detector.imu_health[i].fault_count;
    }

    // 检查足部接触传感器
    for (int i = 0; i < 2; i++) {
        FusionSensorHealth* fh = &fusion->fault_detector.foot_health[i];
        double force = (i == 0) ? fusion->state.left_foot_force_magnitude
                                : fusion->state.right_foot_force_magnitude;
        if (force < 0 || force > 10000) {
            add_fault_event(&fusion->fault_detector, FUSION_FAULT_OUTLIER,
                            -1, 0.3, fusion->state.fusion_timestamp,
                            i == 0 ? "左脚力传感器异常" : "右脚力传感器异常");
            fh->health_score -= 0.05;
            fault_count++;
        }
    }

    // 清除已恢复的事件
    fusion->fault_detector.active_faults = 0;
    for (int i = 0; i < HUMAN_SENSOR_FUSION_MAX_IMUS; i++) {
        FusionSensorHealth* h = &fusion->fault_detector.imu_health[i];
        if (h->health_score < 0.5) fusion->fault_detector.active_faults++;
    }

    return fault_count;
}

int humanoid_sensor_fusion_get_sensor_health(HumanoidSensorFusion* fusion,
                                              int imu_index,
                                              FusionSensorHealth* health)
{
    if (!fusion || !health) return -1;

    if (imu_index >= 0 && imu_index < HUMAN_SENSOR_FUSION_MAX_IMUS) {
        memcpy(health, &fusion->fault_detector.imu_health[imu_index],
               sizeof(FusionSensorHealth));
        return 0;
    } else if (imu_index == -1) {
        // 返回整体平均健康状态
        memset(health, 0, sizeof(FusionSensorHealth));
        double avg_score = 0;
        int count = 0;
        for (int i = 0; i < HUMAN_SENSOR_FUSION_MAX_IMUS; i++) {
            avg_score += fusion->fault_detector.imu_health[i].health_score;
            health->fault_count += fusion->fault_detector.imu_health[i].fault_count;
            count++;
        }
        if (count > 0) {
            health->health_score = avg_score / count;
            health->sensor_online = (health->health_score > 0.3) ? 1 : 0;
            health->data_valid = health->sensor_online;
        }
        return 0;
    }

    return -1;
}

int humanoid_sensor_fusion_get_fault_events(HumanoidSensorFusion* fusion,
                                             FusionSensorFaultEvent* events,
                                             size_t max_count, size_t* count)
{
    if (!fusion || !events || !count) return -1;
    if (!fusion->fault_detector_created) return -1;

    size_t n = (max_count < (size_t)fusion->fault_detector.fault_events_count)
               ? max_count : (size_t)fusion->fault_detector.fault_events_count;

    memcpy(events, fusion->fault_detector.fault_events, n * sizeof(FusionSensorFaultEvent));
    *count = n;
    return 0;
}

int humanoid_sensor_fusion_get_fault_stats(HumanoidSensorFusion* fusion,
                                            int* total_faults,
                                            int* active_faults,
                                            double* avg_health_score)
{
    if (!fusion) return -1;
    if (!fusion->fault_detector_created) return -1;

    if (total_faults)
        *total_faults = fusion->fault_detector.total_faults;
    if (active_faults)
        *active_faults = fusion->fault_detector.active_faults;

    if (avg_health_score) {
        double sum = 0;
        for (int i = 0; i < HUMAN_SENSOR_FUSION_MAX_IMUS; i++)
            sum += fusion->fault_detector.imu_health[i].health_score;
        *avg_health_score = sum / HUMAN_SENSOR_FUSION_MAX_IMUS;
    }

    return 0;
}

int humanoid_sensor_fusion_reset_fault_detector(HumanoidSensorFusion* fusion)
{
    if (!fusion) return -1;
    if (!fusion->fault_detector_created) return -1;

    fusion->fault_detector.total_faults = 0;
    fusion->fault_detector.active_faults = 0;
    fusion->fault_detector.fault_events_count = 0;
    fusion->fault_detector.last_fault_detection_time = 0.0;

    for (int i = 0; i < HUMAN_SENSOR_FUSION_MAX_IMUS; i++)
        init_sensor_health(&fusion->fault_detector.imu_health[i]);
    init_sensor_health(&fusion->fault_detector.joint_health);
    init_sensor_health(&fusion->fault_detector.vision_health);
    for (int i = 0; i < 2; i++)
        init_sensor_health(&fusion->fault_detector.foot_health[i]);
    init_sensor_health(&fusion->fault_detector.gnss_health);

    memset(fusion->previous_accel, 0, sizeof(fusion->previous_accel));
    memset(fusion->previous_gyro, 0, sizeof(fusion->previous_gyro));
    memset(fusion->stuck_check_timer, 0, sizeof(fusion->stuck_check_timer));

    return 0;
}

int humanoid_sensor_fusion_validate_sensor_data(HumanoidSensorFusion* fusion,
                                                  int imu_index,
                                                  double acceleration[3],
                                                  double angular_velocity[3])
{
    if (!fusion || !acceleration || !angular_velocity) return -1;
    if (imu_index < 0 || imu_index >= HUMAN_SENSOR_FUSION_MAX_IMUS) return -1;

    int fixed = 0;
    double gravity = fusion->config.gravity_magnitude;
    if (gravity < 8.0 || gravity > 10.0) gravity = 9.81;

    // 检查NaN或Inf
    for (int i = 0; i < 3; i++) {
        if (isnan(acceleration[i]) || isinf(acceleration[i]) ||
            isnan(angular_velocity[i]) || isinf(angular_velocity[i])) {
            acceleration[i] = 0.0;
            angular_velocity[i] = 0.0;
            fixed = 1;
        }
    }

    // 检查加速度范围（0~5倍重力）
    double accel_norm = sqrt(acceleration[0]*acceleration[0] +
                             acceleration[1]*acceleration[1] +
                             acceleration[2]*acceleration[2]);
    if (accel_norm > gravity * 5.0 || accel_norm < 0.01) {
        if (accel_norm < 0.01) {
            acceleration[2] = gravity;
            fixed = 1;
        } else {
            double scale = gravity / accel_norm;
            for (int i = 0; i < 3; i++)
                acceleration[i] *= scale;
            fixed = 1;
        }
    }

    // 检查角速度范围（±30 rad/s）
    for (int i = 0; i < 3; i++) {
        if (fabs(angular_velocity[i]) > 30.0) {
            angular_velocity[i] = (angular_velocity[i] > 0) ? 30.0 : -30.0;
            fixed = 1;
        }
    }

    // 如果已校准，减去零偏
    if (fusion->imu_calib[imu_index].is_calibrated && !fixed) {
        for (int i = 0; i < 3; i++) {
            angular_velocity[i] -= fusion->imu_calib[imu_index].calib.gyro_bias[i];
            acceleration[i] -= fusion->imu_calib[imu_index].calib.accel_bias[i];
        }
    }

    return fixed ? 1 : 0;
}
