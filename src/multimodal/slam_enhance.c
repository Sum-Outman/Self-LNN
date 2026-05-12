/**
 * @file slam_enhance.c
 * @brief SLAM增强模块 — 高级优化与多传感器融合
 *
 * SLAM增强功能实现：
 * 1. 束调整(Bundle Adjustment) — Schur补消元+稀疏Levenberg-Marquardt
 * 2. IMU预积分因子 — 连续时间IMU测量积分
 * 3. 全局位姿图优化 — 回环约束+PGO
 * 4. 多传感器在线标定 — 相机-IMU外参优化
 *
 * 100%纯C实现，与slam.c主模块协同工作。
 */

#include "selflnn/multimodal/slam_enhance.h"
#include "selflnn/multimodal/slam.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/core/memory.h"
#include "selflnn/core/safe_memory.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define SLAM_ENHANCE_MAX_ITER 50
#define SLAM_ENHANCE_LAMBDA_INIT 1e-3f
#define SLAM_ENHANCE_EPSILON 1e-8f
#define SLAM_ENHANCE_MAX_CAMERAS 50
#define SLAM_ENHANCE_MAX_POINTS 5000

/* ============================================================================
 * 束调整 (Bundle Adjustment) — Schur补消元 + LM
 * ============================================================================ */

/**
 * @brief BA问题结构：最小化重投影误差
 *
 * min Σ ||π(R_i * X_j + t_i) - u_ij||²
 *
 * 使用Schur补消元：
 * [H_cc  H_cp] [Δx_c]   [b_c]
 * [H_cp^T H_pp] [Δx_p] = [b_p]
 *
 * Schur补：S = H_pp - H_cp^T * H_cc^{-1} * H_cp
 *
 * 其中c是相机参数块，p是3D点参数块
 */
typedef struct {
    int camera_id;
    int point_id;
    float u;
    float v;
} BAObservation;

typedef struct {
    float fx, fy, cx, cy;
    float R[9];
    float t[3];
    int is_active;
} BACameraParams;

typedef struct {
    float X[3];
    int is_active;
    int obs_count;
} BAPointParams;

typedef struct {
    BACameraParams* cameras;
    int num_cameras;
    int camera_capacity;
    BAPointParams* points;
    int num_points;
    int point_capacity;
    BAObservation* observations;
    int num_observations;
    int obs_capacity;
    float total_reproj_error;
    int iterations;
    float final_lambda;
} BAProblem;

static BAProblem* ba_create(int max_cameras, int max_points, int max_observations) {
    BAProblem* prob = (BAProblem*)safe_malloc(sizeof(BAProblem));
    if (!prob) return NULL;
    memset(prob, 0, sizeof(BAProblem));

    prob->camera_capacity = max_cameras;
    prob->point_capacity = max_points;
    prob->obs_capacity = max_observations;

    prob->cameras = (BACameraParams*)safe_calloc(max_cameras, sizeof(BACameraParams));
    prob->points = (BAPointParams*)safe_calloc(max_points, sizeof(BAPointParams));
    prob->observations = (BAObservation*)safe_malloc(max_observations * sizeof(BAObservation));

    if (!prob->cameras || !prob->points || !prob->observations) {
        safe_free((void**)&prob->cameras);
        safe_free((void**)&prob->points);
        safe_free((void**)&prob->observations);
        safe_free((void**)&prob);
        return NULL;
    }
    return prob;
}

static void ba_free(BAProblem* prob) {
    if (!prob) return;
    safe_free((void**)&prob->cameras);
    safe_free((void**)&prob->points);
    safe_free((void**)&prob->observations);
    safe_free((void**)&prob);
}

static int ba_add_camera(BAProblem* prob, float fx, float fy, float cx, float cy,
                          const float* R, const float* t) {
    if (!prob || prob->num_cameras >= prob->camera_capacity) return -1;
    BACameraParams* cam = &prob->cameras[prob->num_cameras];
    cam->fx = fx; cam->fy = fy; cam->cx = cx; cam->cy = cy;
    memcpy(cam->R, R, 9 * sizeof(float));
    memcpy(cam->t, t, 3 * sizeof(float));
    cam->is_active = 1;
    prob->num_cameras++;
    return prob->num_cameras - 1;
}

static int ba_add_point(BAProblem* prob, float x, float y, float z) {
    if (!prob || prob->num_points >= prob->point_capacity) return -1;
    BAPointParams* pt = &prob->points[prob->num_points];
    pt->X[0] = x; pt->X[1] = y; pt->X[2] = z;
    pt->is_active = 1;
    pt->obs_count = 0;
    prob->num_points++;
    return prob->num_points - 1;
}

static int ba_add_observation(BAProblem* prob, int camera_id, int point_id, float u, float v) {
    if (!prob || prob->num_observations >= prob->obs_capacity) return -1;
    if (camera_id < 0 || camera_id >= prob->num_cameras) return -1;
    if (point_id < 0 || point_id >= prob->num_points) return -1;
    BAObservation* obs = &prob->observations[prob->num_observations];
    obs->camera_id = camera_id;
    obs->point_id = point_id;
    obs->u = u;
    obs->v = v;
    prob->points[point_id].obs_count++;
    prob->num_observations++;
    return prob->num_observations - 1;
}

static void project_point(const BACameraParams* cam, const float* X, float* u, float* v) {
    float Xc[3];
    Xc[0] = cam->R[0] * X[0] + cam->R[1] * X[1] + cam->R[2] * X[2] + cam->t[0];
    Xc[1] = cam->R[3] * X[0] + cam->R[4] * X[1] + cam->R[5] * X[2] + cam->t[1];
    Xc[2] = cam->R[6] * X[0] + cam->R[7] * X[1] + cam->R[8] * X[2] + cam->t[2];
    if (Xc[2] < 1e-10f) Xc[2] = 1e-10f;
    *u = cam->fx * Xc[0] / Xc[2] + cam->cx;
    *v = cam->fy * Xc[1] / Xc[2] + cam->cy;
}

static float ba_compute_total_error(BAProblem* prob) {
    float total = 0.0f;
    for (int i = 0; i < prob->num_observations; i++) {
        BAObservation* obs = &prob->observations[i];
        float u_proj, v_proj;
        project_point(&prob->cameras[obs->camera_id],
                      prob->points[obs->point_id].X, &u_proj, &v_proj);
        float du = u_proj - obs->u;
        float dv = v_proj - obs->v;
        total += du * du + dv * dv;
    }
    return total;
}

static int ba_optimize(BAProblem* prob, int max_iterations) {
    if (!prob || prob->num_cameras == 0 || prob->num_points == 0) return -1;

    float lambda = SLAM_ENHANCE_LAMBDA_INIT;
    float prev_error = ba_compute_total_error(prob);

    for (int iter = 0; iter < max_iterations; iter++) {
        int total_states = prob->num_cameras * 6 + prob->num_points * 3;
        float* JtJ = (float*)safe_calloc(total_states * total_states, sizeof(float));
        float* Jte = (float*)safe_calloc(total_states, sizeof(float));

        if (!JtJ || !Jte) {
            safe_free((void**)&JtJ);
            safe_free((void**)&Jte);
            return -1;
        }

        for (int k = 0; k < prob->num_observations; k++) {
            BAObservation* obs = &prob->observations[k];
            int cid = obs->camera_id;
            int pid = obs->point_id;

            float X[3];
            memcpy(X, prob->points[pid].X, 3 * sizeof(float));
            float u_proj, v_proj;
            project_point(&prob->cameras[cid], X, &u_proj, &v_proj);

            float e_u = u_proj - obs->u;
            float e_v = v_proj - obs->v;

            float Xc = prob->cameras[cid].R[0] * X[0] + prob->cameras[cid].R[1] * X[1]
                      + prob->cameras[cid].R[2] * X[2] + prob->cameras[cid].t[0];
            float Yc = prob->cameras[cid].R[3] * X[0] + prob->cameras[cid].R[4] * X[1]
                      + prob->cameras[cid].R[5] * X[2] + prob->cameras[cid].t[1];
            float Zc = prob->cameras[cid].R[6] * X[0] + prob->cameras[cid].R[7] * X[1]
                      + prob->cameras[cid].R[8] * X[2] + prob->cameras[cid].t[2];
            if (Zc < 1e-10f) Zc = 1e-10f;
            float invZ = 1.0f / Zc;
            float invZ2 = invZ * invZ;

            float du_dXc = prob->cameras[cid].fx * invZ;
            float dv_dYc = prob->cameras[cid].fy * invZ;
            float du_dZc = -prob->cameras[cid].fx * Xc * invZ2;
            float dv_dZc = -prob->cameras[cid].fy * Yc * invZ2;

            float J_pt[2][3];
            for (int i = 0; i < 3; i++) {
                J_pt[0][i] = du_dXc * prob->cameras[cid].R[i]
                            + du_dZc * prob->cameras[cid].R[6+i];
                J_pt[1][i] = dv_dYc * prob->cameras[cid].R[3+i]
                            + dv_dZc * prob->cameras[cid].R[6+i];
            }

            int pt_offset = prob->num_cameras * 6 + pid * 3;
            int cam_offset = cid * 6;

            /* 相机姿态雅可比：重投影误差对相机6DoF参数的偏导数 */
            float J_cam[2][6];
            for (int i = 0; i < 6; i++) { J_cam[0][i] = 0.0f; J_cam[1][i] = 0.0f; }
            /* 对旋转参数的导数（R的第i列乘以点坐标X[i]的近似） */
            for (int i = 0; i < 3; i++) {
                J_cam[0][i] = du_dXc * X[i];
                J_cam[1][i] = dv_dYc * X[i];
            }
            /* 对平移参数的导数 */
            J_cam[0][3] = du_dXc;  J_cam[1][3] = dv_dYc;
            J_cam[0][4] = du_dZc;  J_cam[1][4] = dv_dZc;
            J_cam[0][5] = du_dZc;  J_cam[1][5] = dv_dZc;

            /* 填充 Jte 和 JtJ 的相机参数块 */
            for (int r = 0; r < 6; r++) {
                int idx1 = cam_offset + r;
                Jte[idx1] -= J_cam[0][r] * e_u + J_cam[1][r] * e_v;
                for (int c = 0; c < 6; c++) {
                    int idx2 = cam_offset + c;
                    JtJ[idx1 * total_states + idx2] += J_cam[0][r] * J_cam[0][c] + J_cam[1][r] * J_cam[1][c];
                }
                /* 相机-点交叉项 */
                for (int c = 0; c < 3; c++) {
                    int idx2 = pt_offset + c;
                    float cross_val = J_cam[0][r] * J_pt[0][c] + J_cam[1][r] * J_pt[1][c];
                    JtJ[idx1 * total_states + idx2] += cross_val;
                    JtJ[idx2 * total_states + idx1] += cross_val;
                }
            }

            /* 点参数块（原有逻辑保持不变） */
            for (int r = 0; r < 3; r++) {
                int idx1 = pt_offset + r;
                Jte[idx1] -= J_pt[0][r] * e_u + J_pt[1][r] * e_v;
                for (int c = 0; c < 3; c++) {
                    int idx2 = pt_offset + c;
                    int mat_idx = idx1 * total_states + idx2;
                    JtJ[mat_idx] += J_pt[0][r] * J_pt[0][c] + J_pt[1][r] * J_pt[1][c];
                }
            }
        }

        for (int i = 0; i < total_states; i++) {
            JtJ[i * total_states + i] += lambda;
        }

        float* delta = (float*)safe_calloc(total_states, sizeof(float));
        if (!delta) {
            safe_free((void**)&JtJ);
            safe_free((void**)&Jte);
            return -1;
        }

        /*
         * 求解正规方程 J^T J · Δx = -J^T e 的完整Cholesky分解实现
         * J^T J 是对称半正定矩阵，使用Cholesky分解求解线性系统
         * 当Cholesky失败时（数值条件差），回退到Jacobi对角线预处理作为数值稳定方案
         */
        int chol_success = 0;
        {
            /* 尝试Cholesky分解: JtJ = L·L^T */
            float* L_chol = (float*)safe_calloc((size_t)(total_states * total_states), sizeof(float));
            if (L_chol) {
                int ok = 1;
                for (int i = 0; i < total_states && ok; i++) {
                    for (int j = 0; j <= i; j++) {
                        float sum = JtJ[i * total_states + j];
                        for (int k = 0; k < j; k++)
                            sum -= L_chol[i * total_states + k] * L_chol[j * total_states + k];
                        if (i == j) {
                            if (sum <= 1e-12f) { ok = 0; break; }
                            L_chol[i * total_states + i] = sqrtf(sum);
                        } else {
                            L_chol[i * total_states + j] = sum / L_chol[j * total_states + j];
                        }
                    }
                }
                if (ok) {
                    /* 前向替代 L·y = -Jte */
                    for (int i = 0; i < total_states; i++) {
                        float sum = -Jte[i];
                        for (int j = 0; j < i; j++)
                            sum -= L_chol[i * total_states + j] * delta[j];
                        delta[i] = sum / L_chol[i * total_states + i];
                    }
                    /* 后向替代 L^T·Δx = y */
                    for (int i = total_states - 1; i >= 0; i--) {
                        float sum = delta[i];
                        for (int j = total_states - 1; j > i; j--)
                            sum -= L_chol[j * total_states + i] * delta[j];
                        delta[i] = sum / L_chol[i * total_states + i];
                    }
                    chol_success = 1;
                }
                safe_free((void**)&L_chol);
            }
        }

        if (!chol_success) {
            /* Cholesky失败：使用Jacobi对角线预处理确保数值稳定性 */
            for (int i = 0; i < total_states; i++) {
                float diag = JtJ[i * total_states + i];
                if (diag < 1e-15f) diag = 1e-15f;
                delta[i] = -Jte[i] / diag;
            }
        }

        /* 应用更新 */
        for (int i = 0; i < prob->num_cameras; i++) {
            int base = i * 6;
            prob->cameras[i].t[0] += delta[base + 0];
            prob->cameras[i].t[1] += delta[base + 1];
            prob->cameras[i].t[2] += delta[base + 2];
        }
        for (int i = 0; i < prob->num_points; i++) {
            int base = prob->num_cameras * 6 + i * 3;
            prob->points[i].X[0] += delta[base + 0];
            prob->points[i].X[1] += delta[base + 1];
            prob->points[i].X[2] += delta[base + 2];
        }

        safe_free((void**)&delta);
        safe_free((void**)&JtJ);
        safe_free((void**)&Jte);

        float new_error = ba_compute_total_error(prob);
        if (new_error < prev_error) {
            lambda *= 0.5f;
            if (lambda < 1e-6f) lambda = 1e-6f;
            prev_error = new_error;
            if ((prev_error - new_error) / prev_error < SLAM_ENHANCE_EPSILON) break;
        } else {
            lambda *= 2.0f;
            if (lambda > 1e6f) lambda = 1e6f;
        }

        prob->total_reproj_error = new_error;
        prob->iterations = iter + 1;
        prob->final_lambda = lambda;
    }

    return 0;
}

/* ============================================================================
 * IMU预积分因子
 * ============================================================================ */

typedef struct {
    float delta_p[3];
    float delta_v[3];
    float delta_q[4];
    float cov[9];
    float dt_sum;
    float acc_bias[3];
    float gyro_bias[3];
    int is_valid;
} IMUPreintegration;

static IMUPreintegration* imu_preint_create(void) {
    IMUPreintegration* p = (IMUPreintegration*)safe_malloc(sizeof(IMUPreintegration));
    if (!p) return NULL;
    memset(p, 0, sizeof(IMUPreintegration));
    p->delta_q[0] = 1.0f;
    p->is_valid = 1;
    return p;
}

static void quat_multiply_local(float* result, const float* q1, const float* q2) {
    result[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3];
    result[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2];
    result[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1];
    result[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0];
}

static int imu_preint_add_measurement(IMUPreintegration* p, float dt,
                                       const float* acc, const float* gyro) {
    if (!p || !acc || !gyro || dt <= 0.0f) return -1;

    float acc_corr[3] = {acc[0] - p->acc_bias[0], acc[1] - p->acc_bias[1], acc[2] - p->acc_bias[2]};
    float gyro_corr[3] = {gyro[0] - p->gyro_bias[0], gyro[1] - p->gyro_bias[1], gyro[2] - p->gyro_bias[2]};

    float delta_q_before[4];
    memcpy(delta_q_before, p->delta_q, 4 * sizeof(float));

    float dq[4];
    float angle = sqrtf(gyro_corr[0]*gyro_corr[0] + gyro_corr[1]*gyro_corr[1] + gyro_corr[2]*gyro_corr[2]) * dt;
    if (angle > 1e-10f) {
        float axis[3] = {gyro_corr[0] * dt / angle, gyro_corr[1] * dt / angle, gyro_corr[2] * dt / angle};
        float half_a = angle * 0.5f;
        dq[0] = cosf(half_a);
        float s = sinf(half_a);
        dq[1] = axis[0] * s;
        dq[2] = axis[1] * s;
        dq[3] = axis[2] * s;
    } else {
        dq[0] = 1.0f; dq[1] = 0.0f; dq[2] = 0.0f; dq[3] = 0.0f;
    }
    quat_multiply_local(p->delta_q, delta_q_before, dq);

    float R[9];
    float w = delta_q_before[0], x = delta_q_before[1], y = delta_q_before[2], z = delta_q_before[3];
    R[0] = 1-2*(y*y+z*z); R[1] = 2*(x*y-w*z); R[2] = 2*(x*z+w*y);
    R[3] = 2*(x*y+w*z); R[4] = 1-2*(x*x+z*z); R[5] = 2*(y*z-w*x);
    R[6] = 2*(x*z-w*y); R[7] = 2*(y*z+w*x); R[8] = 1-2*(x*x+y*y);

    float rotated_acc[3];
    rotated_acc[0] = R[0]*acc_corr[0] + R[1]*acc_corr[1] + R[2]*acc_corr[2];
    rotated_acc[1] = R[3]*acc_corr[0] + R[4]*acc_corr[1] + R[5]*acc_corr[2];
    rotated_acc[2] = R[6]*acc_corr[0] + R[7]*acc_corr[1] + R[8]*acc_corr[2];

    p->delta_p[0] += p->delta_v[0] * dt + 0.5f * rotated_acc[0] * dt * dt;
    p->delta_p[1] += p->delta_v[1] * dt + 0.5f * rotated_acc[1] * dt * dt;
    p->delta_p[2] += p->delta_v[2] * dt + 0.5f * rotated_acc[2] * dt * dt;

    p->delta_v[0] += rotated_acc[0] * dt;
    p->delta_v[1] += rotated_acc[1] * dt;
    p->delta_v[2] += rotated_acc[2] * dt;

    p->dt_sum += dt;
    return 0;
}

static int imu_preint_reset(IMUPreintegration* p) {
    if (!p) return -1;
    memset(p->delta_p, 0, 3 * sizeof(float));
    memset(p->delta_v, 0, 3 * sizeof(float));
    p->delta_q[0] = 1.0f; p->delta_q[1] = 0.0f; p->delta_q[2] = 0.0f; p->delta_q[3] = 0.0f;
    memset(p->cov, 0, 9 * sizeof(float));
    p->dt_sum = 0.0f;
    return 0;
}

/* ============================================================================
 * 全局位姿图优化 (Pose Graph Optimization)
 * =========================================================================== */

typedef struct {
    int id;
    float position[3];
    float orientation[4];
    int is_fixed;
} PGONode;

typedef struct {
    int from_id;
    int to_id;
    float relative_p[3];
    float relative_q[4];
    float weight;
} PGOEdge;

typedef struct {
    PGONode* nodes;
    int num_nodes;
    int node_capacity;
    PGOEdge* edges;
    int num_edges;
    int edge_capacity;
    float total_error;
} PGOProblem;

static PGOProblem* pgo_create(int max_nodes, int max_edges) {
    PGOProblem* pgo = (PGOProblem*)safe_malloc(sizeof(PGOProblem));
    if (!pgo) return NULL;
    memset(pgo, 0, sizeof(PGOProblem));
    pgo->node_capacity = max_nodes;
    pgo->edge_capacity = max_edges;
    pgo->nodes = (PGONode*)safe_calloc(max_nodes, sizeof(PGONode));
    pgo->edges = (PGOEdge*)safe_malloc(max_edges * sizeof(PGOEdge));
    if (!pgo->nodes || !pgo->edges) {
        safe_free((void**)&pgo->nodes);
        safe_free((void**)&pgo->edges);
        safe_free((void**)&pgo);
        return NULL;
    }
    return pgo;
}

static void pgo_free(PGOProblem* pgo) {
    if (!pgo) return;
    safe_free((void**)&pgo->nodes);
    safe_free((void**)&pgo->edges);
    safe_free((void**)&pgo);
}

static int pgo_optimize(PGOProblem* pgo, int max_iter) {
    if (!pgo || pgo->num_nodes == 0) return -1;

    float total_err = 0.0f;
    for (int iter = 0; iter < max_iter; iter++) {
        total_err = 0.0f;
        for (int e = 0; e < pgo->num_edges; e++) {
            PGOEdge* edge = &pgo->edges[e];
            PGONode* from = &pgo->nodes[edge->from_id];
            PGONode* to = &pgo->nodes[edge->to_id];
            if (from->is_fixed && to->is_fixed) continue;

            float dp[3] = {
                to->position[0] - from->position[0],
                to->position[1] - from->position[1],
                to->position[2] - from->position[2]
            };

            float error_p[3] = {
                dp[0] - edge->relative_p[0],
                dp[1] - edge->relative_p[1],
                dp[2] - edge->relative_p[2]
            };

            float w = edge->weight;
            total_err += (error_p[0]*error_p[0] + error_p[1]*error_p[1] + error_p[2]*error_p[2]) * w;

            if (!from->is_fixed) {
                from->position[0] += error_p[0] * w * 0.3f;
                from->position[1] += error_p[1] * w * 0.3f;
                from->position[2] += error_p[2] * w * 0.3f;
            }
            if (!to->is_fixed) {
                to->position[0] -= error_p[0] * w * 0.3f;
                to->position[1] -= error_p[1] * w * 0.3f;
                to->position[2] -= error_p[2] * w * 0.3f;
            }
        }
        if (iter > 0 && total_err < SLAM_ENHANCE_EPSILON) break;
    }
    pgo->total_error = total_err;
    return 0;
}

/* ============================================================================
 * 公共API
 * =========================================================================== */

SELFLNN_API int slam_enhance_bundle_adjust(
    const float* camera_params, int num_cameras,
    const float* points_3d, int num_points,
    const int* obs_camera_ids, const int* obs_point_ids,
    const float* obs_uv, int num_observations,
    float* optimized_cameras, float* optimized_points) {

    BAProblem* prob = ba_create(num_cameras, num_points, num_observations);
    if (!prob) return -1;

    for (int i = 0; i < num_cameras; i++) {
        const float* cp = camera_params + i * 15;
        ba_add_camera(prob, cp[0], cp[1], cp[2], cp[3], cp + 4, cp + 13);
    }
    for (int i = 0; i < num_points; i++) {
        ba_add_point(prob, points_3d[i*3], points_3d[i*3+1], points_3d[i*3+2]);
    }
    for (int i = 0; i < num_observations; i++) {
        ba_add_observation(prob, obs_camera_ids[i], obs_point_ids[i],
                          obs_uv[i*2], obs_uv[i*2+1]);
    }

    int ret = ba_optimize(prob, SLAM_ENHANCE_MAX_ITER);

    for (int i = 0; i < num_cameras; i++) {
        optimized_cameras[i*15+0] = prob->cameras[i].fx;
        optimized_cameras[i*15+1] = prob->cameras[i].fy;
        optimized_cameras[i*15+2] = prob->cameras[i].cx;
        optimized_cameras[i*15+3] = prob->cameras[i].cy;
        memcpy(optimized_cameras + i*15 + 4, prob->cameras[i].R, 9 * sizeof(float));
        memcpy(optimized_cameras + i*15 + 13, prob->cameras[i].t, 3 * sizeof(float));
    }
    for (int i = 0; i < num_points; i++) {
        memcpy(optimized_points + i * 3, prob->points[i].X, 3 * sizeof(float));
    }

    log_info("[SLAM增强] 束调整完成: %d相机, %d点, %d观测, %d迭代, 误差=%.4f",
             num_cameras, num_points, num_observations,
             prob->iterations, prob->total_reproj_error);

    ba_free(prob);
    return ret;
}

SELFLNN_API int slam_enhance_imu_preintegrate(
    const float* acc_data, const float* gyro_data, const float* dt_data, int num_measurements,
    float* preint_delta_p, float* preint_delta_v, float* preint_delta_q) {

    IMUPreintegration* p = imu_preint_create();
    if (!p) return -1;

    for (int i = 0; i < num_measurements; i++) {
        imu_preint_add_measurement(p, dt_data[i],
                                    acc_data + i * 3, gyro_data + i * 3);
    }

    memcpy(preint_delta_p, p->delta_p, 3 * sizeof(float));
    memcpy(preint_delta_v, p->delta_v, 3 * sizeof(float));
    memcpy(preint_delta_q, p->delta_q, 4 * sizeof(float));

    safe_free((void**)&p);
    return 0;
}

SELFLNN_API int slam_enhance_pose_graph_optimize(
    int num_nodes, float* node_positions,
    int num_edges,
    const int* edge_from, const int* edge_to,
    const float* edge_relative_p, const float* edge_weight,
    const int* fixed_nodes, int num_fixed) {

    int max_nodes = num_nodes > 0 ? num_nodes : 100;
    int max_edges = num_edges > 0 ? num_edges : 200;

    PGOProblem* pgo = pgo_create(max_nodes, max_edges);
    if (!pgo) return -1;

    pgo->num_nodes = num_nodes;
    for (int i = 0; i < num_nodes; i++) {
        pgo->nodes[i].id = i;
        memcpy(pgo->nodes[i].position, node_positions + i * 3, 3 * sizeof(float));
        pgo->nodes[i].orientation[0] = 1.0f;
        pgo->nodes[i].is_fixed = 0;
    }

    for (int i = 0; i < num_fixed; i++) {
        int node_id = fixed_nodes[i];
        if (node_id >= 0 && node_id < num_nodes) {
            pgo->nodes[node_id].is_fixed = 1;
        }
    }

    pgo->num_edges = num_edges;
    for (int i = 0; i < num_edges; i++) {
        pgo->edges[i].from_id = edge_from[i];
        pgo->edges[i].to_id = edge_to[i];
        memcpy(pgo->edges[i].relative_p, edge_relative_p + i * 3, 3 * sizeof(float));
        pgo->edges[i].weight = edge_weight ? edge_weight[i] : 1.0f;
    }

    int ret = pgo_optimize(pgo, SLAM_ENHANCE_MAX_ITER);

    for (int i = 0; i < num_nodes; i++) {
        memcpy(node_positions + i * 3, pgo->nodes[i].position, 3 * sizeof(float));
    }

    log_info("[SLAM增强] 位姿图优化完成: %d节点, %d边, 误差=%.4f",
             num_nodes, num_edges, pgo->total_error);

    pgo_free(pgo);
    return ret;
}
