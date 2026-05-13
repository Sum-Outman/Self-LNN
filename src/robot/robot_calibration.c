#include "selflnn/robot/robot_calibration.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/time_utils.h"
#include "selflnn/core/errors.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CALIB_SVD_THRESHOLD 1e-10
#define CALIB_JACOBIAN_MAX_ITER 100

const CalibrationConfig CALIBRATION_CONFIG_DEFAULT = {
    .type = CALIB_TYPE_KINEMATIC,
    .num_joints = 6,
    .max_samples = CALIBRATION_MAX_SAMPLES,
    .max_velocity = 2.0f,
    .max_acceleration = 5.0f,
    .trajectory_duration = 30.0f,
    .sampling_rate = 100.0f,
    .use_excitation = 1,
    .use_least_squares = 1,
    .convergence_threshold = 1e-4f,
    .max_iterations = 50,
    .name = "robot_calib"
};

static void dh_identity(DHParameters* dh) {
    dh->a = 0.0f;
    dh->d = 0.0f;
    dh->alpha = 0.0f;
    dh->theta_offset = 0.0f;
}

static void dh_copy(DHParameters* dst, const DHParameters* src) {
    dst->a = src->a;
    dst->d = src->d;
    dst->alpha = src->alpha;
    dst->theta_offset = src->theta_offset;
}

static void dh_transform(const DHParameters* dh, float theta, float T[16]) {
    float ct = cosf(theta + dh->theta_offset);
    float st = sinf(theta + dh->theta_offset);
    float ca = cosf(dh->alpha);
    float sa = sinf(dh->alpha);
    T[0] = ct;  T[1] = -st*ca; T[2] = st*sa;  T[3] = dh->a * ct;
    T[4] = st;  T[5] = ct*ca;  T[6] = -ct*sa; T[7] = dh->a * st;
    T[8] = 0.0f; T[9] = sa;    T[10] = ca;    T[11] = dh->d;
    T[12] = 0.0f; T[13] = 0.0f; T[14] = 0.0f; T[15] = 1.0f;
}

static void mat4_mul(const float A[16], const float B[16], float C[16]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            C[i*4+j] = A[i*4+0]*B[0*4+j] + A[i*4+1]*B[1*4+j] +
                       A[i*4+2]*B[2*4+j] + A[i*4+3]*B[3*4+j];
        }
    }
}

static void mat4_vec4_mul(const float T[16], const float v[4], float out[4]) {
    for (int i = 0; i < 4; i++) {
        out[i] = T[i*4+0]*v[0] + T[i*4+1]*v[1] + T[i*4+2]*v[2] + T[i*4+3]*v[3];
    }
}

static void forward_kinematics(const DHParameters* dh, int n,
                                const float* q, float T[16]) {
    float result[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    for (int i = 0; i < n; i++) {
        float Ti[16];
        dh_transform(&dh[i], q[i], Ti);
        float tmp[16];
        mat4_mul(result, Ti, tmp);
        memcpy(result, tmp, sizeof(tmp));
    }
    memcpy(T, result, sizeof(result));
}

static void build_identification_matrix(const DHParameters* dh, int n,
                                         const float* q, float* J_geom) {
    float T_total[16];
    forward_kinematics(dh, n, q, T_total);

    float p_ee[4] = {T_total[3], T_total[7], T_total[11], 1.0f};
    float T_accum[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

    for (int j = 0; j < n; j++) {
        float Tj[16];
        dh_transform(&dh[j], q[j], Tj);
        float T_next[16];
        mat4_mul(T_accum, Tj, T_next);

        float z_axis[3] = {T_accum[2], T_accum[6], T_accum[10]};
        float p_diff[3] = {p_ee[0] - T_accum[3],
                           p_ee[1] - T_accum[7],
                           p_ee[2] - T_accum[11]};

        J_geom[j*6+0] = z_axis[1]*p_diff[2] - z_axis[2]*p_diff[1];
        J_geom[j*6+1] = z_axis[2]*p_diff[0] - z_axis[0]*p_diff[2];
        J_geom[j*6+2] = z_axis[0]*p_diff[1] - z_axis[1]*p_diff[0];
        J_geom[j*6+3] = z_axis[0];
        J_geom[j*6+4] = z_axis[1];
        J_geom[j*6+5] = z_axis[2];

        memcpy(T_accum, T_next, sizeof(T_next));
    }
}

static void svd_3x3(const float A[9], float U[9], float S[3], float V[9]) {
    float a[9];
    memcpy(a, A, 9 * sizeof(float));
    float u[9] = {1,0,0, 0,1,0, 0,0,1};
    float v[9] = {1,0,0, 0,1,0, 0,0,1};
    float s[3] = {0,0,0};

    for (int iter = 0; iter < CALIB_JACOBIAN_MAX_ITER; iter++) {
        float max_off = 0.0f;
        int p = 0, q = 1;
        for (int i = 0; i < 3; i++) {
            for (int j = i+1; j < 3; j++) {
                float val = fabsf(a[i*3+j]);
                if (val > max_off) { max_off = val; p = i; q = j; }
            }
        }
        if (max_off < 1e-10f) break;

        float app = a[p*3+p], aqq = a[q*3+q], apq = a[p*3+q];
        float theta = (aqq - app) / (2.0f * apq + 1e-30f);
        float t = (theta >= 0.0f ? 1.0f : -1.0f) /
                  (fabsf(theta) + sqrtf(1.0f + theta*theta));
        float c = 1.0f / sqrtf(1.0f + t*t);
        float s_j = t * c;

        float ap[3], aq[3];
        for (int k = 0; k < 3; k++) {
            ap[k] = a[p*3+k]; aq[k] = a[q*3+k];
            a[p*3+k] = c*ap[k] - s_j*aq[k];
            a[q*3+k] = s_j*ap[k] + c*aq[k];
        }
        for (int k = 0; k < 3; k++) {
            ap[k] = a[k*3+p]; aq[k] = a[k*3+q];
            a[k*3+p] = c*ap[k] - s_j*aq[k];
            a[k*3+q] = s_j*ap[k] + c*aq[k];
        }

        for (int k = 0; k < 3; k++) {
            ap[k] = u[k*3+p]; aq[k] = u[k*3+q];
            u[k*3+p] = c*ap[k] - s_j*aq[k];
            u[k*3+q] = s_j*ap[k] + c*aq[k];
        }
        for (int k = 0; k < 3; k++) {
            ap[k] = v[k*3+p]; aq[k] = v[k*3+q];
            v[k*3+p] = c*ap[k] - s_j*aq[k];
            v[k*3+q] = s_j*ap[k] + c*aq[k];
        }
    }

    for (int i = 0; i < 3; i++) s[i] = fabsf(a[i*3+i]);
    for (int i = 0; i < 9; i++) { U[i] = u[i]; V[i] = v[i]; }
    S[0] = s[0]; S[1] = s[1]; S[2] = s[2];
}

static void pseudo_inverse_3x6(const float* J, int m, float* J_pinv) {
    float JT[6*3];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < 6; j++) {
            JT[j*m + i] = J[i*6 + j];
        }
    }
    float JTJ[9] = {0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 6; k++) {
                sum += JT[i*6 + k] * JT[j*6 + k];
            }
            JTJ[i*3 + j] = sum;
        }
    }
    float U[9], V[9], S[3];
    svd_3x3(JTJ, U, S, V);
    float S_inv[9] = {0};
    for (int i = 0; i < 3; i++) {
        if (S[i] > CALIB_SVD_THRESHOLD) {
            S_inv[i*3 + i] = 1.0f / S[i];
        }
    }
    float US_inv[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += U[i*3 + k] * S_inv[k*3 + j];
            }
            US_inv[i*3 + j] = sum;
        }
    }
    float JTJ_inv[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += US_inv[i*3 + k] * V[j*3 + k];
            }
            JTJ_inv[i*3 + j] = sum;
        }
    }
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += JT[i*3 + k] * JTJ_inv[k*3 + j];
            }
            J_pinv[j*6 + i] = sum;
        }
    }
}

RobotCalibration* robot_calibration_create(const CalibrationConfig* config) {
    RobotCalibration* calib = (RobotCalibration*)safe_calloc(1, sizeof(RobotCalibration));
    if (!calib) return NULL;
    CalibrationConfig cfg = config ? *config : CALIBRATION_CONFIG_DEFAULT;
    robot_calibration_init(calib, &cfg);
    return calib;
}

void robot_calibration_free(RobotCalibration* calib) {
    safe_free((void**)&calib);
}

int robot_calibration_init(RobotCalibration* calib, const CalibrationConfig* config) {
    if (!calib || !config) return -1;
    memset(calib, 0, sizeof(RobotCalibration));
    calib->num_joints = config->num_joints;
    calib->status = CALIB_STATUS_IDLE;
    calib->type = config->type;
    calib->excitation.trajectory_duration = config->trajectory_duration;
    calib->excitation.sampling_rate = config->sampling_rate;
    calib->excitation.max_velocity = config->max_velocity;
    calib->excitation.max_acceleration = config->max_acceleration;
    calib->excitation.use_fourier_trajectory = config->use_excitation;
    calib->converged = 0;
    calib->iteration_count = 0;
    strncpy(calib->name, config->name, CALIBRATION_NAME_MAX - 1);
    calib->name[CALIBRATION_NAME_MAX - 1] = '\0';
    for (int i = 0; i < CALIBRATION_MAX_JOINTS; i++) {
        dh_identity(&calib->nominal_dh[i]);
        dh_identity(&calib->calibrated_dh[i]);
        calib->joint_offsets[i] = 0.0f;
        calib->joint_gear_ratios[i] = 1.0f;
        calib->joint_friction[i] = 0.0f;
        calib->link_masses[i] = 0.0f;
    }
    return 0;
}

int robot_calibration_set_dh_parameters(RobotCalibration* calib,
                                         const DHParameters* dh_params) {
    if (!calib || !dh_params) return -1;
    for (int i = 0; i < calib->num_joints && i < CALIBRATION_MAX_JOINTS; i++) {
        dh_copy(&calib->nominal_dh[i], &dh_params[i]);
        dh_copy(&calib->calibrated_dh[i], &dh_params[i]);
    }
    return 0;
}

int robot_calibration_get_dh_parameters(const RobotCalibration* calib,
                                         DHParameters* dh_params, int calibrated) {
    if (!calib || !dh_params) return -1;
    const DHParameters* src = calibrated ? calib->calibrated_dh : calib->nominal_dh;
    for (int i = 0; i < calib->num_joints && i < CALIBRATION_MAX_JOINTS; i++) {
        dh_copy(&dh_params[i], &src[i]);
    }
    return 0;
}

int robot_calibration_add_sample(RobotCalibration* calib,
                                  const float* joint_positions,
                                  const float* joint_velocities,
                                  const float* joint_torques,
                                  const float* actual_positions) {
    if (!calib || !joint_positions || !actual_positions) return -1;
    if (calib->sample_count >= CALIBRATION_MAX_SAMPLES) return -1;

    CalibrationSample* s = &calib->samples[calib->sample_count];
    for (int i = 0; i < calib->num_joints && i < CALIBRATION_MAX_JOINTS; i++) {
        s->position[i] = joint_positions ? joint_positions[i] : 0.0f;
        s->velocity[i] = joint_velocities ? joint_velocities[i] : 0.0f;
        s->torque[i] = joint_torques ? joint_torques[i] : 0.0f;
        s->actual_position[i] = actual_positions[i];
        s->residual[i] = s->actual_position[i] - s->position[i];
    }
    s->timestamp = time_utils_get_time_s();
    calib->sample_count++;
    return 0;
}

int robot_calibration_generate_excitation(RobotCalibration* calib,
                                           float* trajectory,
                                           size_t* num_points) {
    if (!calib || !trajectory || !num_points) return -1;

    int n_joints = calib->num_joints;
    float duration = calib->excitation.trajectory_duration;
    float fs = calib->excitation.sampling_rate;
    int n_pts = (int)(duration * fs);
    if (n_pts <= 0) n_pts = 1000;

    float w0 = 2.0f * (float)M_PI / duration;
    int n_terms = CALIBRATION_MAX_FOURIER_TERMS;

    /* 动态优化傅里叶参数：基于可观测性指标和条件数优化激励轨迹 */
    /* 初始化基频振幅（指数衰减） */
    for (int j = 0; j < n_terms && j < CALIBRATION_MAX_FOURIER_TERMS; j++) {
        calib->excitation.fourier_amplitudes[j] = 0.5f / (float)(j + 1);
        calib->excitation.fourier_phases[j] = (float)j * 0.1f;
    }
    calib->excitation.fourier_freq_base = w0;

    /* 条件数优化迭代：最大化识别矩阵的最小奇异值(=最小条件数) */
    float best_cond = 1e10f;
    float best_amp[CALIBRATION_MAX_FOURIER_TERMS];
    float best_phase[CALIBRATION_MAX_FOURIER_TERMS];
    memcpy(best_amp, calib->excitation.fourier_amplitudes, sizeof(best_amp));
    memcpy(best_phase, calib->excitation.fourier_phases, sizeof(best_phase));

    for (int opt_iter = 0; opt_iter < 20; opt_iter++) {
        /* 随机扰动振幅和相位 */
        float perturb_amp = 0.1f / (float)(opt_iter + 1);
        for (int j = 0; j < n_terms && j < CALIBRATION_MAX_FOURIER_TERMS; j++) {
            calib->excitation.fourier_amplitudes[j] = best_amp[j] * (1.0f + ((float)((j * 2654435761U) % 2001) / 1000.0f - 1.0f) * perturb_amp);
            calib->excitation.fourier_phases[j] = best_phase[j] + ((float)((j * 314159U) % 6283) / 1000.0f - 3.1415f) * perturb_amp;
            if (calib->excitation.fourier_amplitudes[j] < 0.01f) calib->excitation.fourier_amplitudes[j] = 0.01f;
            if (calib->excitation.fourier_amplitudes[j] > 2.0f) calib->excitation.fourier_amplitudes[j] = 2.0f;
        }

        /* 评估识别矩阵的条件数（简化：利用Hessian迹近似） */
        size_t n = (size_t)calib->num_joints;
        float Happrox[6*6] = {0};
        int n_eval = n_pts < 50 ? n_pts : 50;
        for (int i = 0; i < n_eval; i++) {
            float t = (float)i * duration / (float)n_eval;
            float q_eval[16] = {0};
            for (int j = 0; j < (int)n && j < 16; j++) {
                for (int k = 0; k < n_terms && k < CALIBRATION_MAX_FOURIER_TERMS; k++) {
                    q_eval[j] += calib->excitation.fourier_amplitudes[k] * (1.0f + 0.1f * j)
                              * sinf((float)(k + 1) * w0 * t + calib->excitation.fourier_phases[k] + 0.5f * j);
                }
            }
            /* M-035修复: J6需要6×max_cols空间（16列）而非6×6 */
            float J6[6*16] = {0};
            build_identification_matrix(calib->nominal_dh, (int)n, q_eval, J6);
            for (int ri = 0; ri < 6; ri++)
                for (int ci = 0; ci < 6; ci++)
                    Happrox[ri*6+ci] += J6[ri*16+ci] * J6[ri*16+ci];
        }
        /* 用最大/最小对角线比近似条件数 */
        float min_diag = 1e10f, max_diag = 0.0f;
        for (int di = 0; di < 6; di++) {
            if (Happrox[di*6+di] < min_diag) min_diag = Happrox[di*6+di];
            if (Happrox[di*6+di] > max_diag) max_diag = Happrox[di*6+di];
        }
        float cond_approx = (min_diag > 1e-10f) ? max_diag / min_diag : 1e10f;

        if (cond_approx < best_cond) {
            best_cond = cond_approx;
            memcpy(best_amp, calib->excitation.fourier_amplitudes, sizeof(best_amp));
            memcpy(best_phase, calib->excitation.fourier_phases, sizeof(best_phase));
        }
    }
    /* 应用最优参数 */
    memcpy(calib->excitation.fourier_amplitudes, best_amp, sizeof(best_amp));
    memcpy(calib->excitation.fourier_phases, best_phase, sizeof(best_phase));

    for (int i = 0; i < n_pts && i < CALIBRATION_MAX_SAMPLES; i++) {
        float t = (float)i / fs;
        for (int j = 0; j < n_joints && j < CALIBRATION_MAX_JOINTS; j++) {
            float q = 0.0f;
            for (int k = 0; k < n_terms && k < CALIBRATION_MAX_FOURIER_TERMS; k++) {
                float amp = calib->excitation.fourier_amplitudes[k] * (1.0f + 0.1f * j);
                float phase = calib->excitation.fourier_phases[k] + 0.5f * j;
                q += amp * sinf((float)(k + 1) * w0 * t + phase);
            }
            trajectory[i * n_joints + j] = q;
        }
    }
    *num_points = (size_t)n_pts;
    return 0;
}

int robot_calibration_compute(RobotCalibration* calib) {
    if (!calib) return -1;
    if (calib->sample_count < CALIBRATION_MIN_SAMPLES) return -1;

    calib->status = CALIB_STATUS_COMPUTING;
    calib->start_time = time_utils_get_time_s();
    calib->iteration_count = 0;

    int n = calib->num_joints;
    float prev_error = 1e10f;

    for (int iter = 0; iter < 50; iter++) {
        calib->iteration_count = iter + 1;

        float total_error = 0.0f;
        float max_error = 0.0f;
        float sum_sq = 0.0f;

        float H[6*6] = {0};
        float b[6] = {0};

        for (int s = 0; s < calib->sample_count; s++) {
            float q[16];
            for (int j = 0; j < n; j++) {
                q[j] = calib->samples[s].position[j];
            }

            float T_nominal[16];
            forward_kinematics(calib->nominal_dh, n, q, T_nominal);

            float J_geom[6*16] = {0};
            build_identification_matrix(calib->nominal_dh, n, q, J_geom);

            float p_nominal[3] = {T_nominal[3], T_nominal[7], T_nominal[11]};
            (void)p_nominal;

            float dx = calib->samples[s].actual_position[0] - calib->samples[s].position[0];
            float dy = calib->samples[s].actual_position[1] - calib->samples[s].position[1];
            float dz = calib->samples[s].actual_position[2] - calib->samples[s].position[2];
            float dr = sqrtf(dx*dx + dy*dy + dz*dz);

            float err = dr;
            total_error += err;
            if (err > max_error) max_error = err;
            sum_sq += err * err;

            for (int j = 0; j < 6 && j < n*6; j++) {
                float J_pos[3] = {J_geom[j*6+0], J_geom[j*6+1], J_geom[j*6+2]};
                for (int k = 0; k < 6 && k < n*6; k++) {
                    float Jk_pos[3] = {J_geom[k*6+0], J_geom[k*6+1], J_geom[k*6+2]};
                    H[j*6 + k] += J_pos[0]*Jk_pos[0] + J_pos[1]*Jk_pos[1] + J_pos[2]*Jk_pos[2];
                }
                b[j] += J_pos[0]*dx + J_pos[1]*dy + J_pos[2]*dz;
            }
        }

        float error = total_error / (float)calib->sample_count;
        calib->calibration_error = error;
        calib->max_position_error = max_error;
        calib->rms_position_error = sqrtf(sum_sq / (float)calib->sample_count);

        if (fabsf(error - prev_error) < 1e-4f || error < 1e-4f) {
            calib->converged = 1;
            break;
        }
        prev_error = error;

        float H_inv[6*6] = {0};
        /* 完整Gauss-Jordan消元法求6x6矩阵逆 */
        float aug[6*12];
        memset(aug, 0, sizeof(aug));
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) aug[i*12+j] = H[i*6+j];
            aug[i*12+6+i] = 1.0f;
        }
        int valid = 1;
        for (int col = 0; col < 6 && valid; col++) {
            /* 主元选取 */
            int pivot = col;
            float max_v = fabsf(aug[col*12+col]);
            for (int r = col+1; r < 6; r++) {
                if (fabsf(aug[r*12+col]) > max_v) { max_v = fabsf(aug[r*12+col]); pivot = r; }
            }
            if (max_v < 1e-10f) { valid = 0; break; }
            /* 交换行 */
            if (pivot != col) {
                for (int c = 0; c < 12; c++) {
                    float tmp = aug[col*12+c]; aug[col*12+c] = aug[pivot*12+c]; aug[pivot*12+c] = tmp;
                }
            }
            /* 归一化主元行 */
            float inv_pivot = 1.0f / aug[col*12+col];
            for (int c = 0; c < 12; c++) aug[col*12+c] *= inv_pivot;
            /* 消元其他行 */
            for (int r = 0; r < 6; r++) {
                if (r == col) continue;
                float factor = aug[r*12+col];
                for (int c = 0; c < 12; c++) aug[r*12+c] -= factor * aug[col*12+c];
            }
        }
        if (valid) {
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 6; j++)
                    H_inv[i*6+j] = aug[i*12+6+j];
        } else { break; }

        float delta[6] = {0};
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                delta[i] += H_inv[i*6 + j] * b[j];
            }
        }

        for (int i = 0; i < n && i < 6; i++) {
            calib->nominal_dh[i].theta_offset += delta[i] * 0.1f;
            calib->joint_offsets[i] = calib->nominal_dh[i].theta_offset;
        }
    }

    for (int i = 0; i < n; i++) {
        dh_copy(&calib->calibrated_dh[i], &calib->nominal_dh[i]);
        calib->calibrated_dh[i].theta_offset = calib->joint_offsets[i];
    }

    calib->elapsed_time = time_utils_get_time_s() - calib->start_time;
    calib->status = CALIB_STATUS_COMPLETED;
    return 0;
}

int robot_calibration_verify(RobotCalibration* calib,
                              const float* test_positions,
                              const float* actual_positions,
                              int num_samples) {
    if (!calib || !test_positions || !actual_positions || num_samples <= 0) return -1;

    float total_error = 0.0f;
    float max_error = 0.0f;

    for (int s = 0; s < num_samples; s++) {
        const float* cmd = &test_positions[s * calib->num_joints];
        const float* actual = &actual_positions[s * calib->num_joints];

        float T_cmd[16], T_actual[16];
        forward_kinematics(calib->nominal_dh, calib->num_joints, cmd, T_cmd);
        forward_kinematics(calib->calibrated_dh, calib->num_joints, actual, T_actual);

        float err = sqrtf(powf(T_cmd[3]-T_actual[3], 2) +
                          powf(T_cmd[7]-T_actual[7], 2) +
                          powf(T_cmd[11]-T_actual[11], 2));
        total_error += err;
        if (err > max_error) max_error = err;
    }

    float avg_error = total_error / (float)num_samples;
    calib->calibration_error = avg_error;
    calib->max_position_error = max_error;
    calib->rms_position_error = avg_error;
    calib->status = CALIB_STATUS_VERIFYING;

    return (avg_error < 0.001f) ? 0 : (avg_error < 0.01f) ? 1 : -1;
}

int robot_calibration_get_status(const RobotCalibration* calib,
                                  CalibrationStatus* status, float* error) {
    if (!calib || !status || !error) return -1;
    *status = calib->status;
    *error = calib->calibration_error;
    return 0;
}

int robot_calibration_get_joint_offsets(const RobotCalibration* calib,
                                         float* offsets) {
    if (!calib || !offsets) return -1;
    for (int i = 0; i < calib->num_joints; i++) {
        offsets[i] = calib->joint_offsets[i];
    }
    return 0;
}

int robot_calibration_reset(RobotCalibration* calib) {
    if (!calib) return -1;
    calib->status = CALIB_STATUS_IDLE;
    calib->sample_count = 0;
    calib->calibration_error = 0.0f;
    calib->max_position_error = 0.0f;
    calib->rms_position_error = 0.0f;
    calib->converged = 0;
    calib->iteration_count = 0;
    calib->start_time = 0.0;
    calib->elapsed_time = 0.0;
    for (int i = 0; i < calib->num_joints; i++) {
        dh_copy(&calib->calibrated_dh[i], &calib->nominal_dh[i]);
        calib->joint_offsets[i] = 0.0f;
    }
    return 0;
}

int robot_calibration_save(const RobotCalibration* calib, const char* filepath) {
    if (!calib || !filepath) return -1;
    FILE* f = fopen(filepath, "w");
    if (!f) return -1;
    fprintf(f, "SELF-LNN Robot Calibration File\n");
    fprintf(f, "name=%s\n", calib->name);
    fprintf(f, "num_joints=%d\n", calib->num_joints);
    fprintf(f, "error=%.10f\n", calib->calibration_error);
    fprintf(f, "samples=%d\n", calib->sample_count);
    fprintf(f, "converged=%d\n", calib->converged);
    fprintf(f, "iterations=%d\n", calib->iteration_count);
    for (int i = 0; i < calib->num_joints; i++) {
        fprintf(f, "joint_%d_offsets=%.10f\n", i, calib->joint_offsets[i]);
        fprintf(f, "joint_%d_dh_a=%.10f d=%.10f alpha=%.10f theta_offset=%.10f\n",
                i, calib->calibrated_dh[i].a, calib->calibrated_dh[i].d,
                calib->calibrated_dh[i].alpha, calib->calibrated_dh[i].theta_offset);
    }
    fclose(f);
    return 0;
}

int robot_calibration_load(RobotCalibration* calib, const char* filepath) {
    if (!calib || !filepath) return -1;
    FILE* f = fopen(filepath, "r");
    if (!f) return -1;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "num_joints=%d", &calib->num_joints) == 1) continue;
        if (sscanf(line, "error=%f", &calib->calibration_error) == 1) continue;
        if (sscanf(line, "samples=%d", &calib->sample_count) == 1) continue;
        if (sscanf(line, "converged=%d", &calib->converged) == 1) continue;
        if (sscanf(line, "iterations=%d", &calib->iteration_count) == 1) continue;
        int idx; float val;
        if (sscanf(line, "joint_%d_offsets=%f", &idx, &val) == 2) {
            if (idx >= 0 && idx < CALIBRATION_MAX_JOINTS) calib->joint_offsets[idx] = val;
        }
        float a, d, alpha, th;
        if (sscanf(line, "joint_%d_dh_a=%f d=%f alpha=%f theta_offset=%f",
                   &idx, &a, &d, &alpha, &th) == 5) {
            if (idx >= 0 && idx < CALIBRATION_MAX_JOINTS) {
                calib->calibrated_dh[idx].a = a;
                calib->calibrated_dh[idx].d = d;
                calib->calibrated_dh[idx].alpha = alpha;
                calib->calibrated_dh[idx].theta_offset = th;
            }
        }
    }
    fclose(f);
    calib->status = CALIB_STATUS_COMPLETED;
    return 0;
}

int robot_calibration_apply(RobotCalibration* calib, float* joint_positions,
                             int num_joints) {
    if (!calib || !joint_positions) return -1;
    int n = num_joints < calib->num_joints ? num_joints : calib->num_joints;
    for (int i = 0; i < n; i++) {
        joint_positions[i] += calib->joint_offsets[i];
    }
    return 0;
}

int robot_calibration_get_summary(const RobotCalibration* calib,
                                   char* buffer, size_t buffer_size) {
    if (!calib || !buffer || buffer_size == 0) return -1;
    snprintf(buffer, buffer_size,
             "标定[%s] 关节=%d 状态=%d 误差=%.6frad 最大=%.6frad RMS=%.6frad "
             "样本=%d 收敛=%d 迭代=%d 耗时=%.2fs",
             calib->name, calib->num_joints, (int)calib->status,
             calib->calibration_error, calib->max_position_error,
             calib->rms_position_error, calib->sample_count,
             calib->converged, calib->iteration_count, calib->elapsed_time);
    return 0;
}
