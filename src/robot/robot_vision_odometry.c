#include "selflnn/robot/robot_vision_odometry.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/time_utils.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/core/errors.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define VO_SQRT2 1.4142135623730951f
/* K-012修复：安全随机数 */
#define VO_RAND_FLOAT secure_random_float()

const CameraIntrinsics CAMERA_INTRINSICS_DEFAULT = {
    .K = {500,0,320, 0,500,240, 0,0,1},
    .K_inv = {0.002f,0,-0.64f, 0,0.002f,-0.48f, 0,0,1},
    .fx = 500.0f,
    .fy = 500.0f,
    .cx = 320.0f,
    .cy = 240.0f,
    .width = 640.0f,
    .height = 480.0f
};

static void quaternion_from_axis_angle(float ax, float ay, float az,
                                        float angle, float q[4]) {
    float half = angle * 0.5f;
    float s = sinf(half);
    float c = cosf(half);
    float norm = sqrtf(ax*ax + ay*ay + az*az);
    if (norm < 1e-10f) { q[0] = 1.0f; q[1] = 0.0f; q[2] = 0.0f; q[3] = 0.0f; return; }
    q[0] = c;
    q[1] = ax / norm * s;
    q[2] = ay / norm * s;
    q[3] = az / norm * s;
}

static void quaternion_multiply(const float q1[4], const float q2[4],
                                 float q_out[4]) {
    q_out[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3];
    q_out[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2];
    q_out[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1];
    q_out[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0];
}

static void rotation_from_axis_angle(float ax, float ay, float az,
                                      float angle, float R[9]) {
    float c = cosf(angle);
    float s = sinf(angle);
    float v = 1.0f - c;
    float norm = sqrtf(ax*ax + ay*ay + az*az);
    if (norm < 1e-10f) {
        for (int i = 0; i < 9; i++) R[i] = (i % 4 == 0) ? 1.0f : 0.0f;
        return;
    }
    float nx = ax/norm, ny = ay/norm, nz = az/norm;
    R[0] = nx*nx*v + c;   R[1] = nx*ny*v - nz*s; R[2] = nx*nz*v + ny*s;
    R[3] = nx*ny*v + nz*s; R[4] = ny*ny*v + c;   R[5] = ny*nz*v - nx*s;
    R[6] = nx*nz*v - ny*s; R[7] = ny*nz*v + nx*s; R[8] = nz*nz*v + c;
}

static void rotation_to_axis_angle(const float R[9], float* ax, float* ay,
                                    float* az, float* angle) {
    float trace = R[0] + R[4] + R[8];
    *angle = acosf((trace - 1.0f) * 0.5f);
    if (*angle < 1e-10f) { *ax = 1; *ay = 0; *az = 0; return; }
    float s = 1.0f / (2.0f * sinf(*angle));
    *ax = (R[7] - R[5]) * s;
    *ay = (R[2] - R[6]) * s;
    *az = (R[3] - R[1]) * s;
}

static void mat3_vec3_mul(const float R[9], const float v[3], float out[3]) {
    out[0] = R[0]*v[0] + R[1]*v[1] + R[2]*v[2];
    out[1] = R[3]*v[0] + R[4]*v[1] + R[5]*v[2];
    out[2] = R[6]*v[0] + R[7]*v[1] + R[8]*v[2];
}

static void mat3_transpose(const float R[9], float Rt[9]) {
    Rt[0]=R[0]; Rt[1]=R[3]; Rt[2]=R[6];
    Rt[3]=R[1]; Rt[4]=R[4]; Rt[5]=R[7];
    Rt[6]=R[2]; Rt[7]=R[5]; Rt[8]=R[8];
}

static void cross3(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static float compute_harris_score(const unsigned char* img, int x, int y,
                                   int w, int stride) {
    if (x < 3 || y < 3 || x >= w-3 || y >= 480-3) return 0.0f;

    /* Sobel梯度计算 */
    const float sobel_x[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    const float sobel_y[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

    float grad_Ix = 0.0f, grad_Iy = 0.0f;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            unsigned char pixel = img[(y+dy)*stride + (x+dx)];
            grad_Ix += (float)pixel * sobel_x[dy+1][dx+1];
            grad_Iy += (float)pixel * sobel_y[dy+1][dx+1];
        }
    }
    grad_Ix /= 4.0f;
    grad_Iy /= 4.0f;

    /* 7x7高斯加权窗口累加二阶矩 */
    float A = 0.0f, B = 0.0f, C = 0.0f;
    const float gauss_weights[7][7] = {
        {0.006f,0.013f,0.021f,0.025f,0.021f,0.013f,0.006f},
        {0.013f,0.028f,0.046f,0.054f,0.046f,0.028f,0.013f},
        {0.021f,0.046f,0.074f,0.088f,0.074f,0.046f,0.021f},
        {0.025f,0.054f,0.088f,0.104f,0.088f,0.054f,0.025f},
        {0.021f,0.046f,0.074f,0.088f,0.074f,0.046f,0.021f},
        {0.013f,0.028f,0.046f,0.054f,0.046f,0.028f,0.013f},
        {0.006f,0.013f,0.021f,0.025f,0.021f,0.013f,0.006f}
    };

    for (int dy = -3; dy <= 3; dy++) {
        for (int dx = -3; dx <= 3; dx++) {
            int px = x + dx, py = y + dy;
            if (px < 0 || py < 0 || px >= w || py >= 480) continue;

            unsigned char pixel = img[py*stride + px];
            float local_Ix = 0.0f, local_Iy = 0.0f;
            for (int kdy = -1; kdy <= 1; kdy++) {
                for (int kdx = -1; kdx <= 1; kdx++) {
                    int qx = px + kdx, qy = py + kdy;
                    if (qx < 0 || qy < 0 || qx >= w || qy >= 480) continue;
                    local_Ix += (float)img[qy*stride + qx] * sobel_x[kdy+1][kdx+1];
                    local_Iy += (float)img[qy*stride + qx] * sobel_y[kdy+1][kdx+1];
                }
            }
            local_Ix /= 4.0f; local_Iy /= 4.0f;

            float wg = gauss_weights[dy+3][dx+3];
            A += wg * local_Ix * local_Ix;
            B += wg * local_Ix * local_Iy;
            C += wg * local_Iy * local_Iy;
        }
    }

    float det   = A * C - B * B;
    float trace = A + C;
    if (trace < 1e-10f) return 0.0f;

    /* Harris响应: R = det(M) - k * trace(M)^2, k=0.04 */
    float response = det - 0.04f * trace * trace;
    return response > 0.0f ? response : 0.0f;
}

static void extract_patch_descriptor(const unsigned char* img, int x, int y,
                                      int w, int stride, float* desc) {
    for (int i = 0; i < VO_DESC_LENGTH; i++) desc[i] = 0.0f;
    int patch_size = 8;
    for (int dy = -patch_size; dy < patch_size; dy++) {
        for (int dx = -patch_size; dx < patch_size; dx++) {
            int px = x + dx, py = y + dy;
            if (px < 0 || py < 0 || px >= w || py >= 480) continue;
            int bin = ((dx + patch_size) * 2 / (patch_size/4)) +
                      ((dy + patch_size) * 2 / (patch_size/4)) * 4;
            if (bin >= VO_DESC_LENGTH) bin = VO_DESC_LENGTH - 1;
            float val = (float)img[py * stride + px] / 255.0f;
            float Ix = 0, Iy = 0;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    int idx = (py+k)*stride + (px+l);
                    if (idx >= 0) {
                        Ix += img[idx] * (l == 1 ? 1.0f : (l == -1 ? -1.0f : 0.0f));
                        Iy += img[idx] * (k == 1 ? 1.0f : (k == -1 ? -1.0f : 0.0f));
                    }
                }
            }
            float mag = sqrtf(Ix*Ix + Iy*Iy);
            desc[bin] += mag * val;
        }
    }
    float norm = 0.0f;
    for (int i = 0; i < VO_DESC_LENGTH; i++) norm += desc[i] * desc[i];
    norm = sqrtf(norm);
    if (norm > 1e-6f) {
        for (int i = 0; i < VO_DESC_LENGTH; i++) desc[i] /= norm;
    }
}

static float descriptor_distance(const float* a, const float* b) {
    float dist = 0.0f;
    for (int i = 0; i < VO_DESC_LENGTH; i++) {
        float d = a[i] - b[i];
        dist += d * d;
    }
    return sqrtf(dist);
}

/* 3x3矩阵SVD分解（Jacobi迭代法）
 * 输入: M[9] => 输出: U[9]*S[3]*Vt[9] */
static void svd3_jacobi(const float M[9], float U[9], float S[3], float Vt[9]) {
    float A[9];
    memcpy(A, M, 9 * sizeof(float));

    /* 初始化U=I, Vt=I */
    for (int i = 0; i < 9; i++) {
        U[i]  = (i % 4 == 0) ? 1.0f : 0.0f;
        Vt[i] = (i % 4 == 0) ? 1.0f : 0.0f;
    }

    /* Jacobi迭代 —— 最多50次扫描 */
    for (int sweep = 0; sweep < 50; sweep++) {
        float max_off = 0.0f;
        int p = 0, q = 1;

        /* 寻找最大非对角元素 */
        for (int i = 0; i < 3; i++) {
            for (int j = i + 1; j < 3; j++) {
                float val = fabsf(A[j * 3 + i]);
                if (val > max_off) {
                    max_off = val;
                    p = i; q = j;
                }
            }
        }
        if (max_off < 1e-10f) break;

        /* 计算Jacobi旋转 */
        float app = A[p * 3 + p];
        float aqq = A[q * 3 + q];
        float apq = A[q * 3 + p];

        float theta = 0.5f * atan2f(2.0f * apq, aqq - app);
        float c = cosf(theta);
        float s_val = sinf(theta);

        /* 旋转A */
        for (int j = 0; j < 3; j++) {
            float ajp = A[j * 3 + p];
            float ajq = A[j * 3 + q];
            A[j * 3 + p] = ajp * c - ajq * s_val;
            A[j * 3 + q] = ajp * s_val + ajq * c;
        }
        for (int j = 0; j < 3; j++) {
            float apj = A[p * 3 + j];
            float aqj = A[q * 3 + j];
            A[p * 3 + j] =  c * apj - s_val * aqj;
            A[q * 3 + j] =  s_val * apj + c * aqj;
        }

        /* 更新U */
        for (int j = 0; j < 3; j++) {
            float upj = U[p * 3 + j];
            float uqj = U[q * 3 + j];
            U[p * 3 + j] = upj * c - uqj * s_val;
            U[q * 3 + j] = upj * s_val + uqj * c;
        }

        /* 更新Vt */
        for (int i = 0; i < 3; i++) {
            float vip = Vt[i * 3 + p];
            float viq = Vt[i * 3 + q];
            Vt[i * 3 + p] = vip * c - viq * s_val;
            Vt[i * 3 + q] = vip * s_val + viq * c;
        }
    }

    /* 提取奇异值（对角线元素）并归一化符号 */
    for (int i = 0; i < 3; i++) {
        S[i] = fabsf(A[i * 3 + i]);
        if (A[i * 3 + i] < 0.0f) {
            for (int j = 0; j < 3; j++) U[j * 3 + i] = -U[j * 3 + i];
        }
    }

    /* 按奇异值降序排列 */
    for (int i = 0; i < 2; i++) {
        int max_idx = i;
        for (int j = i + 1; j < 3; j++) {
            if (S[j] > S[max_idx]) max_idx = j;
        }
        if (max_idx != i) {
            float tmp = S[i]; S[i] = S[max_idx]; S[max_idx] = tmp;
            for (int j = 0; j < 3; j++) {
                float tu = U[j * 3 + i]; U[j * 3 + i] = U[j * 3 + max_idx]; U[j * 3 + max_idx] = tu;
                float tv = Vt[i * 3 + j]; Vt[i * 3 + j] = Vt[max_idx * 3 + j]; Vt[max_idx * 3 + j] = tv;
            }
        }
    }

    /* 确保U的行列式为正 */
    float detU = U[0] * (U[4] * U[8] - U[5] * U[7])
               - U[1] * (U[3] * U[8] - U[5] * U[6])
               + U[2] * (U[3] * U[7] - U[4] * U[6]);
    if (detU < 0.0f) {
        for (int j = 0; j < 3; j++) U[j * 3 + 2] = -U[j * 3 + 2];
        S[2] = -S[2];
        for (int j = 0; j < 3; j++) Vt[2 * 3 + j] = -Vt[2 * 3 + j];
    }
}

/* 矩阵乘法: C = A * B (均为3x3) */
static void mat3_mul(const float A[9], const float B[9], float C[9]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            C[i * 3 + j] = A[i * 3 + 0] * B[0 * 3 + j]
                         + A[i * 3 + 1] * B[1 * 3 + j]
                         + A[i * 3 + 2] * B[2 * 3 + j];
        }
    }
}

/* 8点法计算本质矩阵 + RANSAC内点筛选 */
static int essential_from_points(const float* pts1, const float* pts2,
                                  int n, float E[9], int* inliers,
                                  float threshold) {
    if (n < 8) return 0;

    /* 构建8点法矩阵 A [8x9] */
    float A[8 * 9];
    for (int i = 0; i < 8 && i < n; i++) {
        float x1 = pts1[i * 2],     y1 = pts1[i * 2 + 1];
        float x2 = pts2[i * 2],     y2 = pts2[i * 2 + 1];
        A[i * 9 + 0] = x1 * x2; A[i * 9 + 1] = x1 * y2; A[i * 9 + 2] = x1;
        A[i * 9 + 3] = y1 * x2; A[i * 9 + 4] = y1 * y2; A[i * 9 + 5] = y1;
        A[i * 9 + 6] = x2;      A[i * 9 + 7] = y2;       A[i * 9 + 8] = 1.0f;
    }

    /* 构建 A^T * A [9x9] 用于求最小特征值对应的特征向量（零空间） */
    float AtA[81] = {0};
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 9; j++) {
            for (int k = 0; k < 9; k++) {
                AtA[j * 9 + k] += A[i * 9 + j] * A[i * 9 + k];
            }
        }
    }

    /* 幂迭代法求最小特征值对应的特征向量（用逆迭代）
     * QL迭代求所有特征值更可靠，这里使用QR分解近似最小特征向量 */
    float v[9] = {1,1,1,1,1,1,1,1,1};
    /* 逆迭代: 解 (AtA + eps*I)*v_new = v */
    for (int iter = 0; iter < 20; iter++) {
        /* 高斯消元求解 (AtA + 1e-6*I) * v_new = v */
        float M[81];
        memcpy(M, AtA, 81 * sizeof(float));
        for (int i = 0; i < 9; i++) M[i * 9 + i] += 1e-6f;

        float b[9];
        memcpy(b, v, 9 * sizeof(float));

        /* 部分主元高斯消元 */
        for (int col = 0; col < 9; col++) {
            float max_val = fabsf(M[col * 9 + col]);
            int max_row = col;
            for (int row = col + 1; row < 9; row++) {
                if (fabsf(M[row * 9 + col]) > max_val) {
                    max_val = fabsf(M[row * 9 + col]);
                    max_row = row;
                }
            }
            if (max_val < 1e-12f) continue;
            if (max_row != col) {
                for (int j = 0; j < 9; j++) {
                    float t = M[col * 9 + j]; M[col * 9 + j] = M[max_row * 9 + j]; M[max_row * 9 + j] = t;
                }
                float t = b[col]; b[col] = b[max_row]; b[max_row] = t;
            }
            float pivot = M[col * 9 + col];
            for (int row = col + 1; row < 9; row++) {
                float factor = M[row * 9 + col] / pivot;
                for (int j = col; j < 9; j++) M[row * 9 + j] -= factor * M[col * 9 + j];
                b[row] -= factor * b[col];
            }
        }
        /* 回代 */
        for (int i = 8; i >= 0; i--) {
            float sum = b[i];
            for (int j = i + 1; j < 9; j++) sum -= M[i * 9 + j] * v[j];
            v[i] = (fabsf(M[i * 9 + i]) > 1e-12f) ? sum / M[i * 9 + i] : 0.0f;
        }
        /* 归一化 */
        float nv = 0.0f;
        for (int i = 0; i < 9; i++) nv += v[i] * v[i];
        nv = sqrtf(nv);
        if (nv > 0.0f) for (int i = 0; i < 9; i++) v[i] /= nv;
    }

    for (int i = 0; i < 9; i++) E[i] = v[i];

    /* 强制本质矩阵约束：E = U * diag(1,1,0) * V^T */
    float Ue[9], Se[3], Vte[9];
    svd3_jacobi(E, Ue, Se, Vte);

    /* 本质矩阵的两个非零奇异值应相等，取平均值 */
    float s_avg = (Se[0] + Se[1]) * 0.5f;
    float Se_corrected[3] = {s_avg, s_avg, 0.0f};

    /* 重建: E = U * diag(s_avg, s_avg, 0) * V^T */
    float US[9] = {0};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            US[i * 3 + j] = Ue[i * 3 + j] * Se_corrected[j];
    mat3_mul(US, Vte, E);

    /* 归一化 */
    float norm_e = 0.0f;
    for (int i = 0; i < 9; i++) norm_e += E[i] * E[i];
    norm_e = sqrtf(norm_e);
    if (norm_e > 0.0f) for (int i = 0; i < 9; i++) E[i] /= norm_e;

    /* 计算内点 */
    int inlier_count = 0;
    for (int i = 0; i < n; i++) {
        float p1[3] = {pts1[i * 2], pts1[i * 2 + 1], 1.0f};
        float p2[3] = {pts2[i * 2], pts2[i * 2 + 1], 1.0f};
        float Ep1[3];
        mat3_vec3_mul(E, p1, Ep1);
        float err = fabsf(p2[0] * Ep1[0] + p2[1] * Ep1[1] + p2[2] * Ep1[2]);

        /* Sampson距离近似 */
        float denom = Ep1[0] * Ep1[0] + Ep1[1] * Ep1[1] + 1e-10f;
        float sampson_err = err * err / denom;

        int is_inlier = (sampson_err < threshold * threshold) ? 1 : 0;
        if (inliers) inliers[i] = is_inlier;
        if (is_inlier) inlier_count++;
    }
    return inlier_count;
}

/* 从本质矩阵恢复旋转R和平移t（使用真正的SVD分解） */
static void decompose_essential(const float E[9], float R[9], float t[3]) {
    float U[9], Vt[9], S[3];
    svd3_jacobi(E, U, S, Vt);

    /* V = Vt^T */
    float V[9];
    mat3_transpose(Vt, V);

    /* W 矩阵: 绕Z轴旋转90度 */
    float W[9] = {0, -1, 0,
                   1,  0, 0,
                   0,  0, 1};
    float Wt[9] = {0,  1, 0,
                   -1, 0, 0,
                    0, 0, 1};

    /* 两种可能的旋转: R1 = U * W * Vt, R2 = U * Wt * Vt */
    float UW[9], R1[9], R2[9];
    mat3_mul(U, W, UW);
    mat3_mul(UW, Vt, R1);
    mat3_mul(U, Wt, UW);
    mat3_mul(UW, Vt, R2);

    /* 选择行列式为正的旋转矩阵 */
    float detR1 = R1[0] * (R1[4] * R1[8] - R1[5] * R1[7])
                - R1[1] * (R1[3] * R1[8] - R1[5] * R1[6])
                + R1[2] * (R1[3] * R1[7] - R1[4] * R1[6]);

    if (detR1 > 0.0f) {
        memcpy(R, R1, 9 * sizeof(float));
        /* 平移向量 = U的第三列 */
        t[0] = U[2];
        t[1] = U[5];
        t[2] = U[8];
    } else {
        memcpy(R, R2, 9 * sizeof(float));
        t[0] = -U[2];
        t[1] = -U[5];
        t[2] = -U[8];
    }

    /* 归一化平移向量 */
    float tnorm = sqrtf(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
    if (tnorm > 1e-10f) {
        t[0] /= tnorm;
        t[1] /= tnorm;
        t[2] /= tnorm;
    }
}

VisualOdometry* visual_odometry_create(const CameraIntrinsics* intrinsics) {
    VisualOdometry* vo = (VisualOdometry*)safe_calloc(1, sizeof(VisualOdometry));
    if (!vo) return NULL;
    CameraIntrinsics ci = intrinsics ? *intrinsics : CAMERA_INTRINSICS_DEFAULT;
    visual_odometry_init(vo, &ci);
    return vo;
}

void visual_odometry_free(VisualOdometry* vo) {
    safe_free((void**)&vo);
}

int visual_odometry_init(VisualOdometry* vo, const CameraIntrinsics* intrinsics) {
    if (!vo || !intrinsics) return -1;
    memset(vo, 0, sizeof(VisualOdometry));
    vo->intrinsics = *intrinsics;
    vo->max_features = VO_MAX_FEATURES;
    vo->max_response = 0.01f;
    vo->min_distance = 20.0f;
    vo->max_distance = 200.0f;
    vo->ransac_threshold = 0.005f;
    vo->ransac_iterations = 200;
    for (int i = 0; i < 9; i++) {
        vo->pose.orientation[i/4] = (float)((i%4==0)?1:0);
    }
    vo->pose.orientation[0] = 1.0f;
    vo->initialized = 1;
    vo->frame_count = 0;
    strncpy(vo->name, "visual_odometry", 63);
    return 0;
}

int visual_odometry_reset(VisualOdometry* vo) {
    if (!vo) return -1;
    CameraIntrinsics ci = vo->intrinsics;
    int tl = vo->trajectory_length;
    memset(vo, 0, sizeof(VisualOdometry));
    vo->intrinsics = ci;
    vo->trajectory_length = tl;
    vo->pose.orientation[0] = 1.0f;
    vo->initialized = 1;
    return 0;
}

int visual_odometry_extract_features(VisualOdometry* vo,
                                      const unsigned char* image,
                                      int width, int height, int stride) {
    if (!vo || !image || width <= 0 || height <= 0) return -1;

    memcpy(vo->prev_features, vo->features,
           vo->feature_count * sizeof(FeaturePoint));
    vo->prev_feature_count = vo->feature_count;
    vo->feature_count = 0;

    int grid_cols = (width + VO_GRID_SIZE - 1) / VO_GRID_SIZE;
    int grid_rows = (height + VO_GRID_SIZE - 1) / VO_GRID_SIZE;
    int max_per_cell = vo->max_features / (grid_cols * grid_rows);
    if (max_per_cell < 1) max_per_cell = 1;

    for (int gy = 0; gy < grid_rows; gy++) {
        for (int gx = 0; gx < grid_cols; gx++) {
            int x_start = gx * VO_GRID_SIZE;
            int y_start = gy * VO_GRID_SIZE;
            int x_end = x_start + VO_GRID_SIZE;
            int y_end = y_start + VO_GRID_SIZE;
            if (x_end > width) x_end = width;
            if (y_end > height) y_end = height;

            int cell_count = 0;
            for (int y = y_start; y < y_end && cell_count < max_per_cell; y++) {
                for (int x = x_start; x < x_end && cell_count < max_per_cell; x++) {
                    float score = compute_harris_score(image, x, y, width, stride);
                    if (score > vo->max_response) {
                        if (vo->feature_count < VO_MAX_FEATURES) {
                            FeaturePoint* fp = &vo->features[vo->feature_count];
                            fp->x = (float)x;
                            fp->y = (float)y;
                            fp->response = score;
                            fp->tracked = 1;
                            fp->age = 0;
                            vo->feature_count++;
                            cell_count++;
                        }
                    }
                }
            }
        }
    }
    return vo->feature_count;
}

int visual_odometry_compute_descriptors(VisualOdometry* vo,
                                         const unsigned char* image,
                                         int width, int height, int stride) {
    if (!vo || !image) return -1;
    for (int i = 0; i < vo->feature_count; i++) {
        FeaturePoint* fp = &vo->features[i];
        extract_patch_descriptor(image, (int)fp->x, (int)fp->y,
                                  width, stride, fp->descriptor);
    }
    return 0;
}

int visual_odometry_match_features(VisualOdometry* vo) {
    if (!vo) return -1;
    vo->match_count = 0;

    for (int i = 0; i < vo->feature_count && vo->match_count < VO_MAX_MATCHES; i++) {
        float best_dist = 1e10f;
        float second_best = 1e10f;
        int best_j = -1;

        for (int j = 0; j < vo->prev_feature_count; j++) {
            float dist = descriptor_distance(vo->features[i].descriptor,
                                              vo->prev_features[j].descriptor);
            if (dist < best_dist) {
                second_best = best_dist;
                best_dist = dist;
                best_j = j;
            } else if (dist < second_best) {
                second_best = dist;
            }
        }

        if (best_j >= 0 && best_dist < second_best * 0.8f && best_dist < 1.0f) {
            FeatureMatch* m = &vo->matches[vo->match_count];
            m->idx_src = i;
            m->idx_dst = best_j;
            m->distance = best_dist;
            m->ratio = best_dist / (second_best + 1e-10f);
            m->is_inlier = 0;
            vo->match_count++;
        }
    }
    return vo->match_count;
}

int visual_odometry_compute_essential_matrix(VisualOdometry* vo) {
    if (!vo) return -1;
    if (vo->match_count < 8) return -1;

    float pts1[VO_MAX_MATCHES * 2];
    float pts2[VO_MAX_MATCHES * 2];

    CameraIntrinsics* K = &vo->intrinsics;
    int n = vo->match_count;

    for (int i = 0; i < n; i++) {
        FeatureMatch* m = &vo->matches[i];
        FeaturePoint* src = &vo->features[m->idx_src];
        FeaturePoint* dst = &vo->prev_features[m->idx_dst];
        pts1[i*2] = (src->x - K->cx) / K->fx;
        pts1[i*2+1] = (src->y - K->cy) / K->fy;
        pts2[i*2] = (dst->x - K->cx) / K->fx;
        pts2[i*2+1] = (dst->y - K->cy) / K->fy;
    }

    int best_inliers = 0;
    float best_E[9] = {0};
    int best_inlier_flags[VO_MAX_MATCHES] = {0};

    for (int iter = 0; iter < vo->ransac_iterations; iter++) {
        int idx[8];
        for (int i = 0; i < 8; i++) {
            /* K-012修复：安全随机数 */
            idx[i] = (int)(secure_random_int((uint32_t)(n - 1)));
            for (int j = 0; j < i; j++) {
                if (idx[i] == idx[j]) { idx[i] = (int)(secure_random_int((uint32_t)(n - 1))); j = -1; }
            }
        }
        float sample_pts1[16], sample_pts2[16];
        for (int i = 0; i < 8; i++) {
            sample_pts1[i*2] = pts1[idx[i]*2];
            sample_pts1[i*2+1] = pts1[idx[i]*2+1];
            sample_pts2[i*2] = pts2[idx[i]*2];
            sample_pts2[i*2+1] = pts2[idx[i]*2+1];
        }

        float E_test[9];
        int inlier_flags[VO_MAX_MATCHES] = {0};
        int inliers = essential_from_points(sample_pts1, sample_pts2, 8,
                                             E_test, inlier_flags,
                                             vo->ransac_threshold);
        (void)inliers;

        int total_inliers = 0;
        for (int i = 0; i < 8 && i < n; i++) {
            float p1[3] = {pts1[i*2], pts1[i*2+1], 1.0f};
            float p2[3] = {pts2[i*2], pts2[i*2+1], 1.0f};
            float Ep1[3];
            mat3_vec3_mul(E_test, p1, Ep1);
            float err = fabsf(p2[0]*Ep1[0] + p2[1]*Ep1[1] + p2[2]*Ep1[2]);
            if (err < vo->ransac_threshold) {
                inlier_flags[i] = 1;
                total_inliers++;
            }
        }

        if (total_inliers > best_inliers) {
            best_inliers = total_inliers;
            memcpy(best_E, E_test, 9 * sizeof(float));
            memcpy(best_inlier_flags, inlier_flags, n * sizeof(int));
        }
    }

    memcpy(vo->essential.E, best_E, 9 * sizeof(float));
    vo->essential.num_inliers = best_inliers;

    vo->inlier_count = 0;
    for (int i = 0; i < n && vo->inlier_count < VO_MAX_INLIERS; i++) {
        if (best_inlier_flags[i]) {
            vo->inliers[vo->inlier_count] = vo->matches[i];
            vo->inliers[vo->inlier_count].is_inlier = 1;
            vo->inlier_count++;
        }
    }

    return vo->inlier_count;
}

int visual_odometry_estimate_pose(VisualOdometry* vo) {
    if (!vo) return -1;
    if (vo->essential.num_inliers < 8) return -1;

    decompose_essential(vo->essential.E, vo->essential.R, vo->essential.t);

    memcpy(&vo->prev_pose, &vo->pose, sizeof(CameraPose));

    float delta_R[9];
    memcpy(delta_R, vo->essential.R, 9 * sizeof(float));
    float new_pos[3];
    mat3_vec3_mul(delta_R, vo->pose.position, new_pos);
    vo->pose.position[0] = new_pos[0] + vo->essential.t[0];
    vo->pose.position[1] = new_pos[1] + vo->essential.t[1];
    vo->pose.position[2] = new_pos[2] + vo->essential.t[2];

    float ax, ay, az, angle;
    rotation_to_axis_angle(delta_R, &ax, &ay, &az, &angle);
    float dq[4];
    quaternion_from_axis_angle(ax, ay, az, angle, dq);
    float new_q[4];
    quaternion_multiply(vo->pose.orientation, dq, new_q);
    memcpy(vo->pose.orientation, new_q, 4 * sizeof(float));

    float norm_q = sqrtf(vo->pose.orientation[0]*vo->pose.orientation[0] +
                         vo->pose.orientation[1]*vo->pose.orientation[1] +
                         vo->pose.orientation[2]*vo->pose.orientation[2] +
                         vo->pose.orientation[3]*vo->pose.orientation[3]);
    if (norm_q > 0) {
        for (int i = 0; i < 4; i++) vo->pose.orientation[i] /= norm_q;
    }

    vo->pose.velocity[0] = vo->pose.position[0] - vo->prev_pose.position[0];
    vo->pose.velocity[1] = vo->pose.position[1] - vo->prev_pose.position[1];
    vo->pose.velocity[2] = vo->pose.position[2] - vo->prev_pose.position[2];
    vo->pose.initialized = 1;
    vo->pose.timestamp = (unsigned long long)(time_utils_get_time_us());

    if (vo->trajectory_length < 10000) {
        vo->trajectory[vo->trajectory_length][0] = vo->pose.position[0];
        vo->trajectory[vo->trajectory_length][1] = vo->pose.position[1];
        vo->trajectory[vo->trajectory_length][2] = vo->pose.position[2];
        vo->trajectory_length++;
    }

    float translation = sqrtf(vo->essential.t[0]*vo->essential.t[0] +
                               vo->essential.t[1]*vo->essential.t[1] +
                               vo->essential.t[2]*vo->essential.t[2]);
    vo->total_translation += translation;

    return 0;
}

int visual_odometry_update(VisualOdometry* vo,
                            const unsigned char* image,
                            int width, int height, int stride) {
    if (!vo || !image) return -1;

    visual_odometry_extract_features(vo, image, width, height, stride);
    visual_odometry_compute_descriptors(vo, image, width, height, stride);

    if (vo->frame_count > 0) {
        visual_odometry_match_features(vo);
        if (vo->match_count >= 8) {
            visual_odometry_compute_essential_matrix(vo);
            if (vo->essential.num_inliers >= 8) {
                visual_odometry_estimate_pose(vo);
            }
        }
    }

    vo->frame_count++;
    return 0;
}

int visual_odometry_get_pose(const VisualOdometry* vo,
                              float* position, float* orientation) {
    if (!vo || !position || !orientation) return -1;
    memcpy(position, vo->pose.position, 3 * sizeof(float));
    memcpy(orientation, vo->pose.orientation, 4 * sizeof(float));
    return 0;
}

int visual_odometry_get_trajectory(const VisualOdometry* vo,
                                    float* trajectory, size_t* length) {
    if (!vo || !trajectory || !length) return -1;
    size_t n = (size_t)vo->trajectory_length;
    for (size_t i = 0; i < n; i++) {
        trajectory[i*3] = vo->trajectory[i][0];
        trajectory[i*3+1] = vo->trajectory[i][1];
        trajectory[i*3+2] = vo->trajectory[i][2];
    }
    *length = n;
    return 0;
}

int visual_odometry_get_status(const VisualOdometry* vo,
                                char* buffer, size_t size) {
    if (!vo || !buffer || size == 0) return -1;
    snprintf(buffer, size,
             "视觉里程计[%s] 帧=%d 特征=%d 匹配=%d 内点=%d "
             "位置=(%.3f,%.3f,%.3f) 总位移=%.3f 轨迹长度=%d",
             vo->name, vo->frame_count, vo->feature_count,
             vo->match_count, vo->inlier_count,
             vo->pose.position[0], vo->pose.position[1],
             vo->pose.position[2],
             vo->total_translation, vo->trajectory_length);
    return 0;
}

int visual_odometry_triangulate(const VisualOdometry* vo,
                                 const FeatureMatch* match,
                                 float* point_3d) {
    if (!vo || !match || !point_3d) return -1;
    const CameraIntrinsics* K = &vo->intrinsics;
    (void)K;
    const FeaturePoint* f1 = &vo->features[match->idx_src];
    const FeaturePoint* f2 = &vo->prev_features[match->idx_dst];
    float u1 = (f1->x - K->cx) / K->fx;
    float v1 = (f1->y - K->cy) / K->fy;
    float u2 = (f2->x - K->cx) / K->fx;
    float v2 = (f2->y - K->cy) / K->fy;
    float R[9], t[3];
    memcpy(R, vo->essential.R, 9 * sizeof(float));
    memcpy(t, vo->essential.t, 3 * sizeof(float));
    float p1[3] = {u1, v1, 1.0f};
    float p2[3] = {u2, v2, 1.0f};
    float p2_rot[3];
    mat3_vec3_mul(R, p2, p2_rot);
    float A[4] = {0};
    float b[2] = {0};
    A[0] = p1[0]; A[1] = -p2_rot[0]; b[0] = t[0];
    A[2] = p1[1]; A[3] = -p2_rot[1]; b[1] = t[1];
    float det = A[0]*A[3] - A[1]*A[2];
    if (fabsf(det) < 1e-10f) { point_3d[0]=0; point_3d[1]=0; point_3d[2]=0; return -1; }
    float lambda = (b[0]*A[3] - b[1]*A[1]) / det;
    point_3d[0] = p1[0] * lambda;
    point_3d[1] = p1[1] * lambda;
    point_3d[2] = lambda;
    return 0;
}

int visual_odometry_compute_reprojection_error(const VisualOdometry* vo,
                                                 float* mean_error,
                                                 float* max_error) {
    if (!vo || !mean_error || !max_error) return -1;
    float total = 0.0f;
    float max_e = 0.0f;
    int count = 0;
    const CameraIntrinsics* K = &vo->intrinsics;
    (void)K;
    for (int i = 0; i < vo->inlier_count; i++) {
        const FeatureMatch* m = &vo->inliers[i];
        const FeaturePoint* f1 = &vo->features[m->idx_src];
        const FeaturePoint* f2 = &vo->prev_features[m->idx_dst];
        float dx = f1->x - f2->x;
        float dy = f1->y - f2->y;
        float err = sqrtf(dx*dx + dy*dy);
        total += err;
        if (err > max_e) max_e = err;
        count++;
    }
    *mean_error = count > 0 ? total / (float)count : 0.0f;
    *max_error = max_e;
    return count;
}
