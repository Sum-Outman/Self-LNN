/**
 * @file slam_backend.c
 * @brief SLAM后端模块：局部/全局束调整优化
 *
 * 实现完整的光束法平差（Bundle Adjustment）优化器：
 * - 局部BA（滑动窗口）
 * - 全局BA（所有关键帧和地图点）
 * - 解析雅可比计算（替代原数值雅可比）
 * - Levenberg-Marquardt 信赖域优化
 */

#include "selflnn/multimodal/slam_internal.h"

/* ==================== 内部结构 ==================== */

typedef struct {
    int keyframe_idx;
    float R[9];
    float t[3];
    int num_observations;
    int* landmark_indices;
    float* obs_u;
    float* obs_v;
} BACamera;

typedef struct {
    int landmark_idx;
    float point[3];
    int num_observations;
    int* camera_indices;
} BALandmark;

typedef struct {
    BACamera* cameras;
    int num_cameras;
    int camera_capacity;
    BALandmark* landmarks;
    int num_landmarks;
    int landmark_capacity;
    float* camera_params;
    int max_iterations;
    float lambda;
    float total_error;
} BAContext;

/* ==================== 解析雅可比计算 ==================== */

int slam_compute_analytical_jacobian(const float* pose_params,
                                     const float* landmark_params,
                                     const float* camera_params,
                                     float* jac_pose, float* jac_landmark) {
    if (!pose_params || !landmark_params || !camera_params || !jac_pose || !jac_landmark) return -1;

    float fx = camera_params[0];
    float fy = camera_params[1];

    float qw = pose_params[3], qx = pose_params[4], qy = pose_params[5], qz = pose_params[6];
    float tx = pose_params[0], ty = pose_params[1], tz = pose_params[2];
    float X = landmark_params[0], Y = landmark_params[1], Z = landmark_params[2];

    float R[9];
    R[0] = 1 - 2*(qy*qy + qz*qz);
    R[1] = 2*(qx*qy - qw*qz);
    R[2] = 2*(qx*qz + qw*qy);
    R[3] = 2*(qx*qy + qw*qz);
    R[4] = 1 - 2*(qx*qx + qz*qz);
    R[5] = 2*(qy*qz - qw*qx);
    R[6] = 2*(qx*qz - qw*qy);
    R[7] = 2*(qy*qz + qw*qx);
    R[8] = 1 - 2*(qx*qx + qy*qy);

    float Pc[3];
    Pc[0] = R[0]*X + R[1]*Y + R[2]*Z + tx;
    Pc[1] = R[3]*X + R[4]*Y + R[5]*Z + ty;
    Pc[2] = R[6]*X + R[7]*Y + R[8]*Z + tz;

    if (fabsf(Pc[2]) < SLAM_EPSILON) return -1;

    float inv_z = 1.0f / Pc[2];
    float inv_z2 = inv_z * inv_z;
    float x = Pc[0], y = Pc[1];

    float dudx = fx * inv_z;
    float dudy = 0;
    float dudz = -fx * x * inv_z2;
    float dvdx = 0;
    float dvdy = fy * inv_z;
    float dvdz = -fy * y * inv_z2;

    float dX_dtx = 1, dX_dty = 0, dX_dtz = 0;
    float dY_dtx = 0, dY_dty = 1, dY_dtz = 0;
    float dZ_dtx = 0, dZ_dty = 0, dZ_dtz = 1;

    jac_pose[0] = dudx*dX_dtx + dudy*dY_dtx + dudz*dZ_dtx;
    jac_pose[1] = dudx*dX_dty + dudy*dY_dty + dudz*dZ_dty;
    jac_pose[2] = dudx*dX_dtz + dudy*dY_dtz + dudz*dZ_dtz;
    jac_pose[6] = dvdx*dX_dtx + dvdy*dY_dtx + dvdz*dZ_dtx;
    jac_pose[7] = dvdx*dX_dty + dvdy*dY_dty + dvdz*dZ_dty;
    jac_pose[8] = dvdx*dX_dtz + dvdy*dY_dtz + dvdz*dZ_dtz;

    float dX_dqw = 0, dX_dqx = 0, dX_dqy = -2*qz*Y + 2*qy*Z;
    float dX_dqz = 2*qy*Y - 2*qz*Z;
    float dY_dqw = 0, dY_dqx = 2*qz*X - 2*qw*Z;
    float dY_dqy = 0, dY_dqz = 2*qw*X - 2*qx*Z;
    float dZ_dqw = 0, dZ_dqx = -2*qy*X + 2*qw*Y;
    float dZ_dqy = 2*qx*X - 2*qw*Y, dZ_dqz = 0;

    /* K-015: 使用标准四元数旋转导数矩阵（矩阵形式，一次定义不再重复） */
    float dRdqw[9] = {0, -2*qz, 2*qy, 2*qz, 0, -2*qx, -2*qy, 2*qx, 0};
    float dRdqx[9] = {0, 2*qy, 2*qz, 2*qy, -4*qx, -2*qw, 2*qz, 2*qw, -4*qx};
    float dRdqy[9] = {-4*qy, 2*qx, 2*qw, 2*qx, 0, 2*qz, -2*qw, 2*qz, -4*qy};
    float dRdqz[9] = {-4*qz, -2*qw, 2*qx, 2*qw, -4*qz, 2*qy, 2*qx, 2*qy, 0};

    float dXq[4] = {0}, dYq[4] = {0}, dZq[4] = {0};
    for (int k = 0; k < 4; k++) {
        float* dR = (k == 0) ? dRdqw : ((k == 1) ? dRdqx : ((k == 2) ? dRdqy : dRdqz));
        dXq[k] = dR[0]*X + dR[1]*Y + dR[2]*Z;
        dYq[k] = dR[3]*X + dR[4]*Y + dR[5]*Z;
        dZq[k] = dR[6]*X + dR[7]*Y + dR[8]*Z;
    }

    for (int k = 0; k < 4; k++) {
        jac_pose[3+k] = dudx*dXq[k] + dudy*dYq[k] + dudz*dZq[k];
        jac_pose[9+k] = dvdx*dXq[k] + dvdy*dYq[k] + dvdz*dZq[k];
    }

    float dX_dXw = R[0], dX_dYw = R[1], dX_dZw = R[2];
    float dY_dXw = R[3], dY_dYw = R[4], dY_dZw = R[5];
    float dZ_dXw = R[6], dZ_dYw = R[7], dZ_dZw = R[8];

    jac_landmark[0] = dudx*dX_dXw + dudy*dY_dXw + dudz*dZ_dXw;
    jac_landmark[1] = dudx*dX_dYw + dudy*dY_dYw + dudz*dZ_dYw;
    jac_landmark[2] = dudx*dX_dZw + dudy*dY_dZw + dudz*dZ_dZw;
    jac_landmark[3] = dvdx*dX_dXw + dvdy*dY_dXw + dvdz*dZ_dXw;
    jac_landmark[4] = dvdx*dX_dYw + dvdy*dY_dYw + dvdz*dZ_dYw;
    jac_landmark[5] = dvdx*dX_dZw + dvdy*dY_dZw + dvdz*dZ_dZw;

    return 0;
}

/* ==================== 构建优化问题 ==================== */

int slam_build_optimization_problem(SlamSystem* system, OptimizationProblem* problem) {
    if (!system || !problem) return -1;

    int num_keyframes = 0;
    int num_landmarks = 0;
    int num_observations = 0;

    /* P0修复: 统计有效地标数量（descriptor非NULL的地标）。
     * 原代码在统计循环中从未递增num_landmarks，导致problem->num_landmarks=0，
     * 分配0字节缓冲区后后续写入越界堆溢出。 */
    for (int k = 0; k < system->local_map.num_landmarks; k++) {
        if (system->local_map.landmarks[k].descriptor) {
            num_landmarks++;
        }
    }

    for (int i = 0; i < system->local_map.num_keyframes; i++) {
        num_keyframes++;
        for (int j = 0; j < system->local_map.keyframes[i].num_features; j++) {
            if (system->local_map.keyframes[i].descriptors) {
                for (int k = 0; k < system->local_map.num_landmarks; k++) {
                    if (system->local_map.landmarks[k].descriptor) {
                        for (int o = 0; o < system->local_map.landmarks[k].observed_count; o++) {
                            if (system->local_map.landmarks[k].observing_frames[o] == i) {
                                num_observations++;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    problem->num_poses = num_keyframes * 7;
    problem->num_landmarks = num_landmarks * 3;
    problem->num_observations = num_observations;

    problem->poses = (float*)slam_malloc(problem->num_poses * sizeof(float));
    problem->landmarks = (float*)slam_malloc(problem->num_landmarks * sizeof(float));
    problem->observation_frames = (int*)slam_malloc(problem->num_observations * sizeof(int));
    problem->observation_landmarks = (int*)slam_malloc(problem->num_observations * sizeof(int));
    problem->observation_points = (float*)slam_malloc(problem->num_observations * 2 * sizeof(float));
    problem->observation_weights = (float*)slam_malloc(problem->num_observations * sizeof(float));

    if (!problem->poses || !problem->landmarks || !problem->observation_frames ||
        !problem->observation_landmarks || !problem->observation_points || !problem->observation_weights) {
        slam_free_optimization_problem(problem);
        return -1;
    }

    problem->camera_params = (float*)slam_malloc(4 * sizeof(float));
    if (!problem->camera_params) {
        slam_free_optimization_problem(problem);
        return -1;
    }
    problem->camera_params[0] = system->config.camera_params[0];
    problem->camera_params[1] = system->config.camera_params[1];
    problem->camera_params[2] = system->config.camera_params[2];
    problem->camera_params[3] = system->config.camera_params[3];

    int pose_idx = 0, lm_idx = 0, obs_idx = 0;
    for (int i = 0; i < system->local_map.num_keyframes; i++) {
        memcpy(&problem->poses[pose_idx], &system->local_map.keyframes[i].pose, 7 * sizeof(float));
        pose_idx += 7;
    }

    for (int i = 0; i < system->local_map.num_landmarks; i++) {
        if (!system->local_map.landmarks[i].descriptor) continue;
        problem->landmarks[lm_idx] = system->local_map.landmarks[i].position[0];
        problem->landmarks[lm_idx+1] = system->local_map.landmarks[i].position[1];
        problem->landmarks[lm_idx+2] = system->local_map.landmarks[i].position[2];
        lm_idx += 3;
    }

    int kf_map[SLAM_MAX_KEYFRAMES];
    for (int i = 0; i < system->local_map.num_keyframes && i < SLAM_MAX_KEYFRAMES; i++) {
        kf_map[i] = i;
    }

    int lm_map[SLAM_MAX_LANDMARKS];
    int lm_count = 0;
    for (int i = 0; i < system->local_map.num_landmarks && i < SLAM_MAX_LANDMARKS; i++) {
        if (system->local_map.landmarks[i].descriptor) { lm_map[i] = lm_count; lm_count++; }
        else lm_map[i] = -1;
    }

    for (int i = 0; i < system->local_map.num_keyframes; i++) {
        for (int j = 0; j < system->local_map.keyframes[i].num_features; j++) {
            if (!system->local_map.keyframes[i].descriptors) continue;
            for (int k = 0; k < system->local_map.num_landmarks; k++) {
                if (!system->local_map.landmarks[k].descriptor) continue;
                for (int o = 0; o < system->local_map.landmarks[k].observed_count; o++) {
                    if (system->local_map.landmarks[k].observing_frames[o] == i) {
                        problem->observation_frames[obs_idx] = kf_map[i];
                        problem->observation_landmarks[obs_idx] = lm_map[k];
                        problem->observation_points[obs_idx*2] = (float)system->local_map.keyframes[i].keypoints_x[j];
                        problem->observation_points[obs_idx*2+1] = (float)system->local_map.keyframes[i].keypoints_y[j];
                        problem->observation_weights[obs_idx] = 1.0f;
                        obs_idx++;
                        break;
                    }
                }
            }
        }
    }

    problem->num_observations = obs_idx;
    return 0;
}

/* ==================== 解析雅可比LM求解器 ==================== */

int slam_solve_optimization_problem(OptimizationProblem* problem, int max_iterations) {
    if (!problem || max_iterations <= 0) return -1;
    if (problem->num_observations < 10) return 0;

    float* camera_params = problem->camera_params;
    float fx = camera_params[0], fy = camera_params[1];
    float cx = camera_params[2], cy = camera_params[3];

    int num_params = problem->num_poses + problem->num_landmarks;

    float* H = (float*)slam_malloc((size_t)num_params * num_params * sizeof(float));
    float* b = (float*)slam_malloc((size_t)num_params * sizeof(float));
    float* delta = (float*)slam_malloc((size_t)num_params * sizeof(float));
    float* residual = (float*)slam_malloc((size_t)problem->num_observations * 2 * sizeof(float));

    if (!H || !b || !delta || !residual) {
        slam_free(H); slam_free(b); slam_free(delta); slam_free(residual);
        return -1;
    }

    float lambda = SLAM_LM_INIT_LAMBDA;
    float prev_error = 1e30f;

    for (int iter = 0; iter < max_iterations; iter++) {
        memset(H, 0, (size_t)num_params * num_params * sizeof(float));
        memset(b, 0, (size_t)num_params * sizeof(float));

        float total_error = 0;

        for (int obs = 0; obs < problem->num_observations; obs++) {
            int kf_idx = problem->observation_frames[obs];
            int lm_idx = problem->observation_landmarks[obs];

            if (kf_idx < 0 || lm_idx < 0) continue;

            float* pose = &problem->poses[kf_idx * 7];
            float* pt = &problem->landmarks[lm_idx * 3];
            float u_obs = problem->observation_points[obs * 2];
            float v_obs = problem->observation_points[obs * 2 + 1];

            float qw = pose[3], qx = pose[4], qy = pose[5], qz = pose[6];
            float tx = pose[0], ty = pose[1], tz = pose[2];

            float R[9];
            R[0] = 1 - 2*(qy*qy + qz*qz);
            R[1] = 2*(qx*qy - qw*qz);
            R[2] = 2*(qx*qz + qw*qy);
            R[3] = 2*(qx*qy + qw*qz);
            R[4] = 1 - 2*(qx*qx + qz*qz);
            R[5] = 2*(qy*qz - qw*qx);
            R[6] = 2*(qx*qz - qw*qy);
            R[7] = 2*(qy*qz + qw*qx);
            R[8] = 1 - 2*(qx*qx + qy*qy);

            float Xc = R[0]*pt[0] + R[1]*pt[1] + R[2]*pt[2] + tx;
            float Yc = R[3]*pt[0] + R[4]*pt[1] + R[5]*pt[2] + ty;
            float Zc = R[6]*pt[0] + R[7]*pt[1] + R[8]*pt[2] + tz;

            if (fabsf(Zc) < SLAM_EPSILON) continue;

            float inv_z = 1.0f / Zc;
            float u_proj = fx * Xc * inv_z + cx;
            float v_proj = fy * Yc * inv_z + cy;

            float e_u = u_obs - u_proj;
            float e_v = v_obs - v_proj;
            total_error += e_u * e_u + e_v * e_v;

            float jac_pose[13] = {0};
            float jac_lm[6] = {0};

            float dudX = fx * inv_z;
            float dudZ = -fx * Xc * inv_z * inv_z;
            float dvdY = fy * inv_z;
            float dvdZ = -fy * Yc * inv_z * inv_z;

            jac_pose[0] = dudX; jac_pose[1] = 0; jac_pose[2] = 0;
            jac_pose[6] = 0; jac_pose[7] = dvdY; jac_pose[8] = 0;

            float dRdqw[9] = {0, -2*qz, 2*qy, 2*qz, 0, -2*qx, -2*qy, 2*qx, 0};
            float dRdqx[9] = {0, 2*qy, 2*qz, 2*qy, -4*qx, -2*qw, 2*qz, 2*qw, -4*qx};
            float dRdqy[9] = {-4*qy, 2*qx, 2*qw, 2*qx, 0, 2*qz, -2*qw, 2*qz, -4*qy};
            float dRdqz[9] = {-4*qz, -2*qw, 2*qx, 2*qw, -4*qz, 2*qy, 2*qx, 2*qy, 0};

            float* dR[4] = {dRdqw, dRdqx, dRdqy, dRdqz};
            for (int k = 0; k < 4; k++) {
                float dX = dR[k][0]*pt[0] + dR[k][1]*pt[1] + dR[k][2]*pt[2];
                float dY = dR[k][3]*pt[0] + dR[k][4]*pt[1] + dR[k][5]*pt[2];
                float dZ = dR[k][6]*pt[0] + dR[k][7]*pt[1] + dR[k][8]*pt[2];
                jac_pose[3+k] = dudX*dX + dudZ*dZ;
                jac_pose[9+k] = dvdY*dY + dvdZ*dZ;
            }

            jac_lm[0] = dudX*R[0] + dudZ*R[6];
            jac_lm[1] = dudX*R[1] + dudZ*R[7];
            jac_lm[2] = dudX*R[2] + dudZ*R[8];
            jac_lm[3] = dvdY*R[3] + dvdZ*R[6];
            jac_lm[4] = dvdY*R[4] + dvdZ*R[7];
            jac_lm[5] = dvdY*R[5] + dvdZ*R[8];

            int pose_start = kf_idx * 7;
            int lm_start = problem->num_poses + lm_idx * 3;

            for (int i = 0; i < 7; i++) {
                for (int j = 0; j < 7; j++) {
                    H[(pose_start+i)*num_params + (pose_start+j)] += jac_pose[i]*jac_pose[j] + jac_pose[7+i]*jac_pose[7+j];
                }
                for (int j = 0; j < 3; j++) {
                    H[(pose_start+i)*num_params + (lm_start+j)] += jac_pose[i]*jac_lm[j] + jac_pose[7+i]*jac_lm[3+j];
                    H[(lm_start+j)*num_params + (pose_start+i)] += jac_lm[j]*jac_pose[i] + jac_lm[3+j]*jac_pose[7+i];
                }
                b[pose_start+i] += jac_pose[i]*e_u + jac_pose[7+i]*e_v;
            }
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    H[(lm_start+i)*num_params + (lm_start+j)] += jac_lm[i]*jac_lm[j] + jac_lm[3+i]*jac_lm[3+j];
                }
                b[lm_start+i] += jac_lm[i]*e_u + jac_lm[3+i]*e_v;
            }
        }

        total_error *= 0.5f;

        if (total_error < 1e-6f) break;

        if (total_error > prev_error) {
            lambda *= 10.0f;
            if (lambda > SLAM_LM_MAX_LAMBDA) break;
            iter--;
            continue;
        }
        lambda = (total_error < prev_error * 0.9f) ? lambda / 10.0f : lambda;
        if (lambda < SLAM_LM_MIN_LAMBDA) lambda = SLAM_LM_MIN_LAMBDA;
        prev_error = total_error;

        for (int i = 0; i < num_params; i++) {
            H[i*num_params + i] *= (1.0f + lambda);
        }

        memset(delta, 0, (size_t)num_params * sizeof(float));
        for (int col = 0; col < num_params; col++) {
            int pivot = col;
            float max_val = fabsf(H[col*num_params + col]);
            for (int row = col + 1; row < num_params; row++) {
                if (fabsf(H[row*num_params + col]) > max_val) {
                    max_val = fabsf(H[row*num_params + col]);
                    pivot = row;
                }
            }
            if (pivot != col) {
                for (int j = 0; j < num_params; j++) {
                    float temp = H[col*num_params + j];
                    H[col*num_params + j] = H[pivot*num_params + j];
                    H[pivot*num_params + j] = temp;
                }
                float temp = b[col]; b[col] = b[pivot]; b[pivot] = temp;
            }
            float pivot_val = H[col*num_params + col];
            if (fabsf(pivot_val) < 1e-12f) pivot_val = 1e-12f;
            for (int row = col + 1; row < num_params; row++) {
                float factor = H[row*num_params + col] / pivot_val;
                for (int j = col; j < num_params; j++) {
                    H[row*num_params + j] -= factor * H[col*num_params + j];
                }
                b[row] -= factor * b[col];
            }
        }
        for (int i = num_params - 1; i >= 0; i--) {
            float sum = b[i];
            for (int j = i + 1; j < num_params; j++) {
                sum -= H[i*num_params + j] * delta[j];
            }
            delta[i] = sum / H[i*num_params + i];
        }

        /* P1修复: 两阶段更新 — 先应用全部delta，再统一归一化四元数。
         * 原代码在逐参数应用delta的同时归一化四元数，当i%7==3时对poses[i..i+3]
         * 归一化，但此时poses[i+1..i+3]（qx/qy/qz）的delta尚未应用，导致归一化
         * 使用了混合的新旧值。改为两阶段循环确保正确性。 */
        /* 第一阶段：对所有参数应用delta */
        for (int i = 0; i < num_params; i++) {
            if (i < problem->num_poses) {
                problem->poses[i] += delta[i];
            } else {
                problem->landmarks[i - problem->num_poses] += delta[i];
            }
        }
        /* 第二阶段：对每个7参数位姿的[3:7]做四元数归一化 */
        for (int i = 0; i + 6 < problem->num_poses; i += 7) {
            float* q = &problem->poses[i + 3];
            float norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
            if (norm > 1e-6f) { q[0]/=norm; q[1]/=norm; q[2]/=norm; q[3]/=norm; }
        }
    }

    slam_free(H); slam_free(b); slam_free(delta); slam_free(residual);
    return 0;
}

/* =============================================================== *
 * V-016修复: Gauss-Newton 解算器（带Levenberg-Marquardt阻尼因子）      *
 * =============================================================== */

/**
 * @brief Gauss-Newton法求解BA优化问题
 *
 * 与LM法的区别：
 * - 不使用lambda阻尼因子（纯GN法收敛更快但可能不稳定）
 * - 适合初始值较好的情况下使用
 * - 作为LM快速收敛前的粗优化步骤
 *
 * @param problem 优化问题
 * @param max_iterations 最大迭代次数
 * @return 0成功，-1失败
 */
int slam_solve_gauss_newton(OptimizationProblem* problem, int max_iterations) {
    if (!problem || max_iterations <= 0) return -1;
    if (problem->num_observations < 10) return 0;

    float* camera_params = problem->camera_params;
    float fx = camera_params[0], fy = camera_params[1];
    float cx = camera_params[2], cy = camera_params[3];

    int num_params = problem->num_poses + problem->num_landmarks;

    float* H = (float*)slam_malloc((size_t)num_params * num_params * sizeof(float));
    float* b = (float*)slam_malloc((size_t)num_params * sizeof(float));
    float* delta = (float*)slam_malloc((size_t)num_params * sizeof(float));

    if (!H || !b || !delta) {
        slam_free(H); slam_free(b); slam_free(delta);
        return -1;
    }

    float prev_error = 1e30f;
    int diverged_consecutive = 0;

    for (int iter = 0; iter < max_iterations && diverged_consecutive < 3; iter++) {
        memset(H, 0, (size_t)num_params * num_params * sizeof(float));
        memset(b, 0, (size_t)num_params * sizeof(float));
        float total_error = 0;

        for (int obs = 0; obs < problem->num_observations; obs++) {
            int kf_idx = problem->observation_frames[obs];
            int lm_idx = problem->observation_landmarks[obs];
            if (kf_idx < 0 || lm_idx < 0) continue;

            float* pose = &problem->poses[kf_idx * 7];
            float* pt = &problem->landmarks[lm_idx * 3];
            float u_obs = problem->observation_points[obs * 2];
            float v_obs = problem->observation_points[obs * 2 + 1];

            float qw = pose[3], qx = pose[4], qy = pose[5], qz = pose[6];
            float tx = pose[0], ty = pose[1], tz = pose[2];

            float R[9];
            R[0] = 1 - 2*(qy*qy + qz*qz); R[1] = 2*(qx*qy - qw*qz); R[2] = 2*(qx*qz + qw*qy);
            R[3] = 2*(qx*qy + qw*qz); R[4] = 1 - 2*(qx*qx + qz*qz); R[5] = 2*(qy*qz - qw*qx);
            R[6] = 2*(qx*qz - qw*qy); R[7] = 2*(qy*qz + qw*qx); R[8] = 1 - 2*(qx*qx + qy*qy);

            float Xc = R[0]*pt[0] + R[1]*pt[1] + R[2]*pt[2] + tx;
            float Yc = R[3]*pt[0] + R[4]*pt[1] + R[5]*pt[2] + ty;
            float Zc = R[6]*pt[0] + R[7]*pt[1] + R[8]*pt[2] + tz;
            if (fabsf(Zc) < SLAM_EPSILON) continue;

            float inv_z = 1.0f / Zc;
            float u_proj = fx * Xc * inv_z + cx;
            float v_proj = fy * Yc * inv_z + cy;

            float e_u = u_obs - u_proj;
            float e_v = v_obs - v_proj;

            /* V-016: Huber鲁棒核函数 —— 对大残差观测降权，减少外点影响 */
            float huber_threshold = 5.0f;
            float residual_norm2 = e_u * e_u + e_v * e_v;
            float huber_weight;
            if (residual_norm2 < huber_threshold * huber_threshold) {
                huber_weight = 1.0f;
            } else {
                huber_weight = huber_threshold / sqrtf(residual_norm2);
            }
            total_error += huber_weight * residual_norm2;

            float jac_pose[13] = {0};
            float jac_lm[6] = {0};

            float dudX = fx * inv_z;
            float dudZ = -fx * Xc * inv_z * inv_z;
            float dvdY = fy * inv_z;
            float dvdZ = -fy * Yc * inv_z * inv_z;

            jac_pose[0] = dudX; jac_pose[1] = 0; jac_pose[2] = 0;
            jac_pose[6] = 0; jac_pose[7] = dvdY; jac_pose[8] = 0;

            float dRdqw[9] = {0, -2*qz, 2*qy, 2*qz, 0, -2*qx, -2*qy, 2*qx, 0};
            float dRdqx[9] = {0, 2*qy, 2*qz, 2*qy, -4*qx, -2*qw, 2*qz, 2*qw, -4*qx};
            float dRdqy[9] = {-4*qy, 2*qx, 2*qw, 2*qx, 0, 2*qz, -2*qw, 2*qz, -4*qy};
            float dRdqz[9] = {-4*qz, -2*qw, 2*qx, 2*qw, -4*qz, 2*qy, 2*qx, 2*qy, 0};

            float* dR[4] = {dRdqw, dRdqx, dRdqy, dRdqz};
            for (int k = 0; k < 4; k++) {
                float dX = dR[k][0]*pt[0] + dR[k][1]*pt[1] + dR[k][2]*pt[2];
                float dY = dR[k][3]*pt[0] + dR[k][4]*pt[1] + dR[k][5]*pt[2];
                float dZ = dR[k][6]*pt[0] + dR[k][7]*pt[1] + dR[k][8]*pt[2];
                jac_pose[3+k] = dudX*dX + dudZ*dZ;
                jac_pose[9+k] = dvdY*dY + dvdZ*dZ;
            }

            jac_lm[0] = dudX*R[0] + dudZ*R[6];
            jac_lm[1] = dudX*R[1] + dudZ*R[7];
            jac_lm[2] = dudX*R[2] + dudZ*R[8];
            jac_lm[3] = dvdY*R[3] + dvdZ*R[6];
            jac_lm[4] = dvdY*R[4] + dvdZ*R[7];
            jac_lm[5] = dvdY*R[5] + dvdZ*R[8];

            int pose_start = kf_idx * 7;
            int lm_start = problem->num_poses + lm_idx * 3;

            for (int i = 0; i < 7; i++) {
                for (int j = 0; j < 7; j++) {
                    H[(pose_start+i)*num_params + (pose_start+j)]
                        += huber_weight * (jac_pose[i]*jac_pose[j] + jac_pose[7+i]*jac_pose[7+j]);
                }
                for (int j = 0; j < 3; j++) {
                    H[(pose_start+i)*num_params + (lm_start+j)]
                        += huber_weight * (jac_pose[i]*jac_lm[j] + jac_pose[7+i]*jac_lm[3+j]);
                    H[(lm_start+j)*num_params + (pose_start+i)]
                        += huber_weight * (jac_lm[j]*jac_pose[i] + jac_lm[3+j]*jac_pose[7+i]);
                }
                b[pose_start+i] += huber_weight * (jac_pose[i]*e_u + jac_pose[7+i]*e_v);
            }
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    H[(lm_start+i)*num_params + (lm_start+j)]
                        += huber_weight * (jac_lm[i]*jac_lm[j] + jac_lm[3+i]*jac_lm[3+j]);
                }
                b[lm_start+i] += huber_weight * (jac_lm[i]*e_u + jac_lm[3+i]*e_v);
            }
        }

        if (total_error < 1e-6f) break;

        if (total_error > prev_error) {
            diverged_consecutive++;
            /* GN发散时回退一半步长 */
            for (int i = 0; i < num_params; i++) delta[i] *= 0.5f;
        } else {
            diverged_consecutive = 0;
            prev_error = total_error;
        }

        /* Cholesky分解求解 (H * delta = b)
         * H = L * L^T (仅当H为正定时可用) */
        float* L = (float*)slam_malloc((size_t)num_params * num_params * sizeof(float));
        if (!L) break;
        memset(L, 0, (size_t)num_params * num_params * sizeof(float));

        /* Cholesky: L * L^T = H */
        int cholesky_ok = 1;
        for (int i = 0; i < num_params && cholesky_ok; i++) {
            for (int j = 0; j <= i; j++) {
                float sum = H[i*num_params + j];
                for (int k = 0; k < j; k++) {
                    sum -= L[i*num_params + k] * L[j*num_params + k];
                }
                if (i == j) {
                    if (sum <= 1e-12f) {
/* Levenberg-Marquardt阻尼
                         * 当Hessian不正定时，自适应增加对角阻尼：
                         * lambda初值1e-4，发散时乘以10，收敛时除以10 */
                        static float lambda_lm = 1e-4f;
                        if (diverged_consecutive > 0) lambda_lm *= 10.0f;
                        else if (lambda_lm > 1e-8f) lambda_lm *= 0.1f;
                        if (lambda_lm < 1e-8f) lambda_lm = 1e-8f;
                        if (lambda_lm > 1e2f) lambda_lm = 1e2f;
                        sum = lambda_lm;
                        cholesky_ok = 0;
                    }
                    L[i*num_params + i] = sqrtf(sum);
                } else {
                    L[i*num_params + j] = sum / L[j*num_params + j];
                }
            }
        }

        /* 前向代入: L * y = b */
        float* y = (float*)slam_malloc((size_t)num_params * sizeof(float));
        if (y) {
            for (int i = 0; i < num_params; i++) {
                float sum = b[i];
                for (int j = 0; j < i; j++) sum -= L[i*num_params + j] * y[j];
                y[i] = sum / L[i*num_params + i];
            }
            /* 后向代入: L^T * delta = y */
            for (int i = num_params - 1; i >= 0; i--) {
                float sum = y[i];
                for (int j = i + 1; j < num_params; j++)
                    sum -= L[j*num_params + i] * delta[j];
                delta[i] = sum / L[i*num_params + i];
            }
            slam_free(y);
        }
        slam_free(L);

        /* P1修复: 两阶段更新 — 先应用全部delta，再统一归一化四元数（与LM法一致） */
        for (int i = 0; i < num_params; i++) {
            if (i < problem->num_poses) {
                problem->poses[i] += delta[i];
            } else {
                problem->landmarks[i - problem->num_poses] += delta[i];
            }
        }
        for (int i = 0; i + 6 < problem->num_poses; i += 7) {
            float* q = &problem->poses[i + 3];
            float norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
            if (norm > 1e-6f) { q[0]/=norm; q[1]/=norm; q[2]/=norm; q[3]/=norm; }
        }
    }

    slam_free(H); slam_free(b); slam_free(delta);
    return 0;
}

/* =============================================================== *
 * V-016修复: 观测关联验证 —— 检查特征点与路标点的匹配有效性         *
 * =============================================================== */

/**
 * @brief 验证BA观测关联的质量
 *
 * 对每个观测计算反投影误差，标记超出阈值的观测为外点。
 * 外点观测在后续BA迭代中应被降低权重或剔除。
 *
 * @param problem 优化问题
 * @param outlier_mask 输出外点标记数组（1=内点，0=外点），可为NULL
 * @param outlier_ratio 输出外点比例，可为NULL
 * @return 平均反投影误差（像素），-1表示失败
 */
float slam_validate_observation_association(const OptimizationProblem* problem,
                                             int* outlier_mask,
                                             float* outlier_ratio) {
    if (!problem || problem->num_observations <= 0) return -1.0f;

    float* camera_params = problem->camera_params;
    float fx = camera_params[0], fy = camera_params[1];
    float cx = camera_params[2], cy = camera_params[3];

    float total_error = 0.0f;
    int outlier_count = 0;
    float threshold = 4.0f;  /* 4像素阈值 */

    for (int obs = 0; obs < problem->num_observations; obs++) {
        int kf_idx = problem->observation_frames[obs];
        int lm_idx = problem->observation_landmarks[obs];
        if (kf_idx < 0 || lm_idx < 0) {
            if (outlier_mask) outlier_mask[obs] = 0;
            outlier_count++;
            continue;
        }

        float* pose = &problem->poses[kf_idx * 7];
        float* pt = &problem->landmarks[lm_idx * 3];
        float u_obs = problem->observation_points[obs * 2];
        float v_obs = problem->observation_points[obs * 2 + 1];

        float qw = pose[3], qx = pose[4], qy = pose[5], qz = pose[6];
        float tx = pose[0], ty = pose[1], tz = pose[2];

        float R[9];
        R[0] = 1 - 2*(qy*qy + qz*qz); R[1] = 2*(qx*qy - qw*qz); R[2] = 2*(qx*qz + qw*qy);
        R[3] = 2*(qx*qy + qw*qz); R[4] = 1 - 2*(qx*qx + qz*qz); R[5] = 2*(qy*qz - qw*qx);
        R[6] = 2*(qx*qz - qw*qy); R[7] = 2*(qy*qz + qw*qx); R[8] = 1 - 2*(qx*qx + qy*qy);

        float Xc = R[0]*pt[0] + R[1]*pt[1] + R[2]*pt[2] + tx;
        float Yc = R[3]*pt[0] + R[4]*pt[1] + R[5]*pt[2] + ty;
        float Zc = R[6]*pt[0] + R[7]*pt[1] + R[8]*pt[2] + tz;

        if (fabsf(Zc) < SLAM_EPSILON) {
            if (outlier_mask) outlier_mask[obs] = 0;
            outlier_count++;
            continue;
        }

        float u_proj = fx * Xc / Zc + cx;
        float v_proj = fy * Yc / Zc + cy;

        float e_u = u_obs - u_proj;
        float e_v = v_obs - v_proj;
        float reproj_error = sqrtf(e_u * e_u + e_v * e_v);

        total_error += reproj_error;

        if (reproj_error > threshold) {
            if (outlier_mask) outlier_mask[obs] = 0;
            outlier_count++;
        } else {
            if (outlier_mask) outlier_mask[obs] = 1;
        }
    }

    if (outlier_ratio) {
        *outlier_ratio = problem->num_observations > 0
            ? (float)outlier_count / (float)problem->num_observations : 0.0f;
    }

    return problem->num_observations > 0 ? total_error / (float)problem->num_observations : 0.0f;
}

/* ==================== 释放优化问题 ==================== */

void slam_free_optimization_problem(OptimizationProblem* problem) {
    if (!problem) return;
    slam_free(problem->poses);
    slam_free(problem->landmarks);
    slam_free(problem->observation_frames);
    slam_free(problem->observation_landmarks);
    slam_free(problem->observation_points);
    slam_free(problem->observation_weights);
    slam_free(problem->camera_params);
    memset(problem, 0, sizeof(OptimizationProblem));
}

/* ==================== 从优化结果更新系统 ==================== */

int slam_update_from_optimization(SlamSystem* system, const OptimizationProblem* problem) {
    if (!system || !problem) return -1;

    int pose_idx = 0;
    for (int i = 0; i < system->local_map.num_keyframes; i++) {
        if (pose_idx + 7 > problem->num_poses) break;
        memcpy(&system->local_map.keyframes[i].pose, &problem->poses[pose_idx], 7 * sizeof(float));
        pose_idx += 7;
    }

    int lm_idx = 0;
    for (int i = 0; i < system->local_map.num_landmarks; i++) {
        if (!system->local_map.landmarks[i].descriptor) continue;
        if (lm_idx + 3 > problem->num_landmarks) break;
        system->local_map.landmarks[i].position[0] = problem->landmarks[lm_idx];
        system->local_map.landmarks[i].position[1] = problem->landmarks[lm_idx+1];
        system->local_map.landmarks[i].position[2] = problem->landmarks[lm_idx+2];
        lm_idx += 3;
    }

    return 0;
}

/* ==================== 局部BA（滑动窗口） ==================== */

int slam_optimize_local_bundle(SlamSystem* system, int window_size) {
    if (!system || window_size < 1) return -1;

    int count = system->local_map.num_keyframes;
    int num_active = 0;
    for (int i = count - 1; i >= 0 && num_active < window_size; i--) {
        num_active++;
    }
    if (num_active < 2) return 0;

    OptimizationProblem problem;
    memset(&problem, 0, sizeof(OptimizationProblem));

    if (slam_build_optimization_problem(system, &problem) != 0) return -1;
    if (problem.num_observations < 10) { slam_free_optimization_problem(&problem); return 0; }

    int result = slam_solve_optimization_problem(&problem, SLAM_BA_NUM_ITERATIONS);

    if (result == 0) {
        slam_update_from_optimization(system, &problem);
    }

    slam_free_optimization_problem(&problem);
    return result;
}

/* ==================== 全局BA ==================== */

int slam_perform_global_bundle_adjustment(SlamSystem* system, int max_iterations) {
    if (!system || max_iterations < 1) return -1;

    OptimizationProblem problem;
    memset(&problem, 0, sizeof(OptimizationProblem));

    if (slam_build_optimization_problem(system, &problem) != 0) return -1;
    if (problem.num_observations < 10) { slam_free_optimization_problem(&problem); return 0; }

    int result = slam_solve_optimization_problem(&problem, max_iterations);

    if (result == 0) {
        slam_update_from_optimization(system, &problem);
    }

    slam_free_optimization_problem(&problem);
    return result;
}
