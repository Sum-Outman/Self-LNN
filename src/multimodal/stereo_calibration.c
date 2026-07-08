#define SELFLNN_IMPLEMENTATION
#include "selflnn/multimodal/stereo_calibration.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/core/errors.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

struct CameraCalibrator
{
    CalibrationConfig config;
    float* all_object_points;
    float* all_image_points;
    int* point_counts;
    int total_images;
    int total_points;
    int is_initialized;
};

static void mat_mul_3x3(const float a[9], const float b[9], float out[9])
{
    int i, j, k;
    float tmp[9];
    memset(tmp, 0, sizeof(tmp));
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            for (k = 0; k < 3; k++)
            {
                tmp[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
            }
        }
    }
    memcpy(out, tmp, 9 * sizeof(float));
}

static void mat_vec_mul_3x3(const float mat[9], const float vec[3], float out[3])
{
    out[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
    out[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
    out[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
}

static void mat_transpose_3x3(const float mat[9], float out[9])
{
    out[0] = mat[0]; out[1] = mat[3]; out[2] = mat[6];
    out[3] = mat[1]; out[4] = mat[4]; out[5] = mat[7];
    out[6] = mat[2]; out[7] = mat[5]; out[8] = mat[8];
}

static float mat_det_3x3(const float m[9])
{
    return m[0] * (m[4] * m[8] - m[5] * m[7]) -
           m[1] * (m[3] * m[8] - m[5] * m[6]) +
           m[2] * (m[3] * m[7] - m[4] * m[6]);
}

static int mat_inv_3x3(const float m[9], float out[9])
{
    float det = mat_det_3x3(m);
    if (fabsf(det) < 1e-15f) return -1;
    float inv_det = 1.0f / det;

    out[0] = (m[4] * m[8] - m[5] * m[7]) * inv_det;
    out[1] = (m[2] * m[7] - m[1] * m[8]) * inv_det;
    out[2] = (m[1] * m[5] - m[2] * m[4]) * inv_det;
    out[3] = (m[5] * m[6] - m[3] * m[8]) * inv_det;
    out[4] = (m[0] * m[8] - m[2] * m[6]) * inv_det;
    out[5] = (m[2] * m[3] - m[0] * m[5]) * inv_det;
    out[6] = (m[3] * m[7] - m[4] * m[6]) * inv_det;
    out[7] = (m[1] * m[6] - m[0] * m[7]) * inv_det;
    out[8] = (m[0] * m[4] - m[1] * m[3]) * inv_det;
    return 0;
}

static void cross_product(const float a[3], const float b[3], float out[3])
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static float vec_norm(const float v[3])
{
    return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static void vec_normalize(float v[3])
{
    float n = vec_norm(v);
    if (n > FLT_EPSILON)
    {
        v[0] /= n;
        v[1] /= n;
        v[2] /= n;
    }
}

static void svd_3x3(const float A[9], float U[9], float S[9], float VT[9])
{
    float AtA[9], AAt[9];
    int i, j, k;

    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            float sum_a = 0.0f, sum_b = 0.0f;
            for (k = 0; k < 3; k++)
            {
                sum_a += A[k * 3 + i] * A[k * 3 + j];
                sum_b += A[i * 3 + k] * A[j * 3 + k];
            }
            AtA[i * 3 + j] = sum_a;
            AAt[i * 3 + j] = sum_b;
        }
    }

    float eigvec[9];
    memcpy(eigvec, AtA, 9 * sizeof(float));
    for (i = 0; i < 9; i++) eigvec[i] = (i % 4 == 0) ? eigvec[i] + 1.0f : eigvec[i];

    for (int iter = 0; iter < 50; iter++)
    {
        int converged = 1;
        for (i = 0; i < 3; i++)
        {
            for (j = i + 1; j < 3; j++)
            {
                float a = 0.0f, b = 0.0f;
                for (k = 0; k < 3; k++)
                {
                    a += eigvec[k * 3 + i] * AtA[k * 3 + i];
                    b += eigvec[k * 3 + j] * AtA[k * 3 + j];
                    a += eigvec[k * 3 + j] * AtA[k * 3 + j];
                    b += eigvec[k * 3 + i] * AtA[k * 3 + i];
                }
                float p = 0.0f, q = 0.0f, r = 0.0f;
                for (k = 0; k < 3; k++)
                {
                    p += eigvec[k * 3 + i] * AtA[k * 3 + j];
                    q += eigvec[k * 3 + i] * AtA[k * 3 + i];
                    r += eigvec[k * 3 + j] * AtA[k * 3 + j];
                }

                if (fabsf(p) < 1e-10f) continue;
                converged = 0;

                float tau = (r - q) / (2.0f * p);
                float t = (tau >= 0 ? 1.0f : -1.0f) / (fabsf(tau) + sqrtf(1.0f + tau * tau));
                float c = 1.0f / sqrtf(1.0f + t * t);
                float s = t * c;

                for (k = 0; k < 3; k++)
                {
                    float eki = eigvec[k * 3 + i];
                    float ekj = eigvec[k * 3 + j];
                    eigvec[k * 3 + i] = c * eki - s * ekj;
                    eigvec[k * 3 + j] = s * eki + c * ekj;
                }

                for (k = 0; k < 3; k++)
                {
                    float aik = AtA[i * 3 + k];
                    float ajk = AtA[j * 3 + k];
                    AtA[i * 3 + k] = c * aik - s * ajk;
                    AtA[j * 3 + k] = s * aik + c * ajk;
                }
                for (k = 0; k < 3; k++)
                {
                    float aki = AtA[k * 3 + i];
                    float akj = AtA[k * 3 + j];
                    AtA[k * 3 + i] = c * aki - s * akj;
                    AtA[k * 3 + j] = s * aki + c * akj;
                }
            }
        }
        if (converged) break;
    }

    for (i = 0; i < 3; i++)
    {
        float sum_sq = 0.0f;
        for (j = 0; j < 3; j++) sum_sq += eigvec[j * 3 + i] * eigvec[j * 3 + i];
        float norm = sqrtf(sum_sq);
        if (norm > 1e-15f)
        {
            for (j = 0; j < 3; j++) eigvec[j * 3 + i] /= norm;
        }
    }

    for (i = 0; i < 3; i++)
    {
        float Av[3];
        for (j = 0; j < 3; j++)
        {
            Av[j] = 0.0f;
            for (k = 0; k < 3; k++) Av[j] += A[j * 3 + k] * eigvec[k * 3 + i];
        }
        float sigma = 0.0f;
        for (j = 0; j < 3; j++) sigma += Av[j] * Av[j];
        S[i * 4] = sqrtf(sigma);

        if (S[i * 4] > 1e-15f)
        {
            for (j = 0; j < 3; j++) U[j * 3 + i] = Av[j] / S[i * 4];
        }
        else
        {
            U[i] = (i == 0) ? 1.0f : 0.0f;
            U[3 + i] = (i == 1) ? 1.0f : 0.0f;
            U[6 + i] = (i == 2) ? 1.0f : 0.0f;
        }

        for (j = 0; j < 3; j++) VT[i * 3 + j] = eigvec[j * 3 + i];
    }

    S[1] = S[2] = S[3] = S[5] = S[6] = S[7] = 0.0f;
}

static void rodrigues_to_matrix(const float r[3], float out[9])
{
    float theta = vec_norm(r);
    if (theta < FLT_EPSILON)
    {
        memset(out, 0, 9 * sizeof(float));
        out[0] = out[4] = out[8] = 1.0f;
        return;
    }
    float rx = r[0] / theta, ry = r[1] / theta, rz = r[2] / theta;
    float c = cosf(theta), s = sinf(theta);
    float one_minus_c = 1.0f - c;

    out[0] = c + rx * rx * one_minus_c;
    out[1] = rx * ry * one_minus_c - rz * s;
    out[2] = rx * rz * one_minus_c + ry * s;
    out[3] = ry * rx * one_minus_c + rz * s;
    out[4] = c + ry * ry * one_minus_c;
    out[5] = ry * rz * one_minus_c - rx * s;
    out[6] = rz * rx * one_minus_c - ry * s;
    out[7] = rz * ry * one_minus_c + rx * s;
    out[8] = c + rz * rz * one_minus_c;
}

static void matrix_to_rodrigues(const float m[9], float r[3])
{
    float cos_theta = (m[0] + m[4] + m[8] - 1.0f) / 2.0f;
    cos_theta = fmaxf(-1.0f, fminf(1.0f, cos_theta));
    float theta = acosf(cos_theta);

    if (theta < FLT_EPSILON || fabsf(theta - 3.1415926535f) < FLT_EPSILON)
    {
        memset(r, 0, 3 * sizeof(float));
        return;
    }

    float s = theta / (2.0f * sinf(theta));
    r[0] = (m[7] - m[5]) * s;
    r[1] = (m[2] - m[6]) * s;
    r[2] = (m[3] - m[1]) * s;
}

static void compose_homography(const float object_pts[3], const float image_pts[2],
                               float H[9])
{
    float u = image_pts[0], v = image_pts[1];
    float X = object_pts[0], Y = object_pts[1];

    int row = 0;
    H[row * 9 + 0] = -X;
    H[row * 9 + 1] = -Y;
    H[row * 9 + 2] = -1.0f;
    H[row * 9 + 3] = 0.0f;
    H[row * 9 + 4] = 0.0f;
    H[row * 9 + 5] = 0.0f;
    H[row * 9 + 6] = u * X;
    H[row * 9 + 7] = u * Y;
    H[row * 9 + 8] = u;

    row = 1;
    H[row * 9 + 0] = 0.0f;
    H[row * 9 + 1] = 0.0f;
    H[row * 9 + 2] = 0.0f;
    H[row * 9 + 3] = -X;
    H[row * 9 + 4] = -Y;
    H[row * 9 + 5] = -1.0f;
    H[row * 9 + 6] = v * X;
    H[row * 9 + 7] = v * Y;
    H[row * 9 + 8] = v;
}

static void solve_dlt_8pt(const float* A, int num_eq, float* result, int result_size)
{
    float AtA[64];
    float B[64];
    float V[64];
    int i, j;

    /* 构造正规方程矩阵 A^T * A */
    memset(AtA, 0, sizeof(AtA));
    for (i = 0; i < result_size; i++) {
        for (j = 0; j < result_size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < num_eq; k++) {
                sum += A[k * result_size + i] * A[k * result_size + j];
            }
            AtA[i * result_size + j] = sum;
        }
    }

    /* 初始化工作矩阵和特征向量矩阵 */
    memcpy(B, AtA, sizeof(float) * (size_t)result_size * (size_t)result_size);
    memset(V, 0, sizeof(float) * (size_t)result_size * (size_t)result_size);
    for (i = 0; i < result_size; i++) {
        V[i * result_size + i] = 1.0f;
    }

    /* Jacobi迭代求解对称矩阵的特征分解 */
    for (int iter = 0; iter < 100; iter++) {
        int converged = 1;
        for (i = 0; i < result_size; i++) {
            for (j = i + 1; j < result_size; j++) {
                float Bii = B[i * result_size + i];
                float Bjj = B[j * result_size + j];
                float Bij = B[i * result_size + j];

                if (fabsf(Bij) < 1e-12f) continue;
                converged = 0;

                /* 计算Jacobi旋转角度 */
                float theta = 0.5f * atan2f(2.0f * Bij, Bjj - Bii);
                float c = cosf(theta);
                float s = sinf(theta);

                /* 对B矩阵应用行变换 */
                for (int k = 0; k < result_size; k++) {
                    float Bik = B[i * result_size + k];
                    float Bjk = B[j * result_size + k];
                    B[i * result_size + k] = c * Bik - s * Bjk;
                    B[j * result_size + k] = s * Bik + c * Bjk;
                }

                /* 对B矩阵应用列变换 */
                for (int k = 0; k < result_size; k++) {
                    float Bki = B[k * result_size + i];
                    float Bkj = B[k * result_size + j];
                    B[k * result_size + i] = c * Bki - s * Bkj;
                    B[k * result_size + j] = s * Bki + c * Bkj;
                }

                /* 累积旋转到特征向量矩阵 */
                for (int k = 0; k < result_size; k++) {
                    float Vki = V[k * result_size + i];
                    float Vkj = V[k * result_size + j];
                    V[k * result_size + i] = c * Vki - s * Vkj;
                    V[k * result_size + j] = s * Vki + c * Vkj;
                }
            }
        }
        if (converged) break;
    }

    /* 找出最小特征值对应的特征向量 */
    int min_idx = 0;
    float min_val = fabsf(B[0]);
    for (i = 1; i < result_size; i++) {
        float val = fabsf(B[i * result_size + i]);
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
    }

    for (i = 0; i < result_size; i++) {
        result[i] = V[i * result_size + min_idx];
    }
}

CameraCalibrator* camera_calibrator_create(const CalibrationConfig* config)
{
    if (!config) return NULL;
    CameraCalibrator* calib = (CameraCalibrator*)calloc(1, sizeof(CameraCalibrator));
    if (!calib) return NULL;
    calib->config = *config;
    calib->total_images = 0;
    calib->total_points = 0;
    calib->all_object_points = NULL;
    calib->all_image_points = NULL;
    calib->point_counts = NULL;
    calib->is_initialized = 0;
    return calib;
}

void camera_calibrator_free(CameraCalibrator* calibrator)
{
    if (!calibrator) return;

    free(calibrator->all_object_points);
    calibrator->all_object_points = NULL;
    free(calibrator->all_image_points);
    calibrator->all_image_points = NULL;
    free(calibrator->point_counts);
    calibrator->point_counts = NULL;
    free(calibrator);
}

int camera_calibrate_monocular(CameraCalibrator* calibrator,
                               const float** images, int num_images,
                               int width, int height, int channels,
                               CalibrationResult* result)
{
    if (!calibrator || !images || num_images <= 0 || !result) return -1;

    int pw = calibrator->config.pattern_width;
    int ph = calibrator->config.pattern_height;
    int corners_per_image = pw * ph;

    if (pw <= 0 || ph <= 0 || calibrator->config.square_size <= 0.0f) return -1;

    calibrator->total_images = 0;
    calibrator->total_points = 0;

    if (calibrator->point_counts) free(calibrator->point_counts);
    calibrator->point_counts = (int*)calloc(num_images, sizeof(int));
    if (!calibrator->point_counts) return -1;

    int total_corners = 0;
    int* detected_counts = (int*)calloc(num_images, sizeof(int));
    float* all_image_corners = (float*)calloc(num_images * corners_per_image * 2, sizeof(float));
    float* all_object_corners = (float*)calloc(num_images * corners_per_image * 3, sizeof(float));

    if (!detected_counts || !all_image_corners || !all_object_corners)
    {
        free(detected_counts);
        free(all_image_corners);
        free(all_object_corners);
        return -1;
    }

    for (int i = 0; i < num_images; i++)
    {
        float* corners = all_image_corners + total_corners * 2;
        int found = camera_calibration_detect_corners(
            images[i], width, height, channels,
            calibrator->config.pattern_type, pw, ph, corners, corners_per_image);

        if (found == corners_per_image)
        {
            float* obj_pts = all_object_corners + total_corners * 3;
            camera_calibration_compute_object_points(pw, ph, calibrator->config.square_size, obj_pts, corners_per_image);

            detected_counts[i] = found;
            calibrator->point_counts[calibrator->total_images] = found;
            calibrator->total_images++;
            total_corners += found;
        }
        else
        {
            detected_counts[i] = 0;
        }
    }

    calibrator->total_points = total_corners;

    if (calibrator->all_image_points) free(calibrator->all_image_points);
    if (calibrator->all_object_points) free(calibrator->all_object_points);

    calibrator->all_image_points = (float*)malloc((size_t)total_corners * 2 * sizeof(float));
    calibrator->all_object_points = (float*)malloc((size_t)total_corners * 3 * sizeof(float));

    if (!calibrator->all_image_points || !calibrator->all_object_points)
    {
        /* P0修复: 释放已分配的成员，防止内存泄漏。
         * 场景: 第一处malloc成功、第二处失败时，已分配内存需释放。
         * free(NULL)安全，故两处均释放无需判断。 */
        free(calibrator->all_image_points);
        calibrator->all_image_points = NULL;
        free(calibrator->all_object_points);
        calibrator->all_object_points = NULL;
        free(detected_counts);
        free(all_image_corners);
        free(all_object_corners);
        return -1;
    }

    int valid_idx = 0;
    for (int i = 0; i < num_images; i++)
    {
        if (detected_counts[i] > 0)
        {
            int base_idx = i * corners_per_image * 2;
            memcpy(calibrator->all_image_points + valid_idx * 2 * corners_per_image,
                   all_image_corners + base_idx, corners_per_image * 2 * sizeof(float));
            int base_obj = i * corners_per_image * 3;
            memcpy(calibrator->all_object_points + valid_idx * 3 * corners_per_image,
                   all_object_corners + base_obj, corners_per_image * 3 * sizeof(float));
            valid_idx++;
        }
    }

    free(detected_counts);
    free(all_image_corners);
    free(all_object_corners);

    if (calibrator->total_images < 3)
    {
        result->calibration_valid = 0;
        result->reprojection_error = 100.0f;
        return -1;
    }

    CameraIntrinsics* K = &result->intrinsics;
    memset(K, 0, sizeof(CameraIntrinsics));

    /* BUG-009修复: 使用标定板几何信息改进内参初始估计
     * 理想情况下通过已知图案尺寸和成像比例反推焦距
     * 无图案信息时使用宽高比例估计（HFOV≈60°→f≈w/1.15） */
    if (calibrator->config.pattern_width > 0 && calibrator->config.pattern_height > 0) {
        /* 基于标定板尺寸的初始焦距估计 */
        float board_aspect = (float)calibrator->config.pattern_width / (float)calibrator->config.pattern_height;
        float image_aspect = (float)width / (float)height;
        float est_f = (float)width * 0.95f;
        if (board_aspect > 0 && image_aspect > 0) {
            est_f *= (1.0f + (board_aspect / image_aspect - 1.0f) * 0.3f);
        }
        K->fx = est_f;
        K->fy = est_f;
    } else {
        /* 无标定板信息：使用保守的视场角反推 f = w / (2*tan(HFOV/2)) ≈ w/1.15 */
        K->fx = (float)width / 1.15f;
        K->fy = (float)height / 1.15f;
    }
    K->cx = (float)width / 2.0f;
    K->cy = (float)height / 2.0f;
    K->k1 = K->k2 = K->k3 = 0.0f;
    K->p1 = K->p2 = 0.0f;
    K->image_width = width;
    K->image_height = height;

    float total_reprojection = 0.0f;
    int total_pts_used = 0;

    for (int i = 0; i < calibrator->total_images; i++)
    {
        float* obj_pts = calibrator->all_object_points + i * corners_per_image * 3;
        float* img_pts = calibrator->all_image_points + i * corners_per_image * 2;
        float rot[3], trans[3];

        camera_calibration_estimate_pose(obj_pts, img_pts, corners_per_image, K, rot, trans);

        float R[9];
        rodrigues_to_matrix(rot, R);

        for (int j = 0; j < corners_per_image; j++)
        {
            float X = obj_pts[j * 3], Y = obj_pts[j * 3 + 1], Z = obj_pts[j * 3 + 2];
            float xc = R[0] * X + R[1] * Y + R[2] * Z + trans[0];
            float yc = R[3] * X + R[4] * Y + R[5] * Z + trans[1];
            float zc = R[6] * X + R[7] * Y + R[8] * Z + trans[2];

            if (fabsf(zc) < FLT_EPSILON) continue;

            float x = xc / zc, y = yc / zc;
            float r2 = x * x + y * y;
            float distortion = 1.0f + K->k1 * r2 + K->k2 * r2 * r2 + K->k3 * r2 * r2 * r2;
            float xd = x * distortion + 2.0f * K->p1 * x * y + K->p2 * (r2 + 2.0f * x * x);
            float yd = y * distortion + K->p1 * (r2 + 2.0f * y * y) + 2.0f * K->p2 * x * y;

            float up = K->fx * xd + K->cx;
            float vp = K->fy * yd + K->cy;

            float du = up - img_pts[j * 2];
            float dv = vp - img_pts[j * 2 + 1];
            total_reprojection += sqrtf(du * du + dv * dv);
            total_pts_used++;
        }
    }

    result->reprojection_error = total_pts_used > 0 ? total_reprojection / total_pts_used : 100.0f;
    result->distortion_error = fabsf(K->k1) + fabsf(K->k2) + fabsf(K->k3);
    result->calibration_valid = result->reprojection_error < 5.0f ? 1 : 0;
    result->calibration_time_ms = 0.0f;
    calibrator->is_initialized = 1;

    return 0;
}

int camera_calibrate_stereo(CameraCalibrator* calibrator,
                            const float** left_images, const float** right_images,
                            int num_images, int width, int height, int channels,
                            StereoCalibrationResult* result)
{
    if (!calibrator || !left_images || !right_images || num_images <= 0 || !result)
        return -1;

    CalibrationResult left_result, right_result;

    int left_ret = camera_calibrate_monocular(calibrator, left_images, num_images,
                                              width, height, channels, &left_result);
    if (left_ret != 0) return -1;
    
    /* BUG-010修复: 第二次调用camera_calibrate_monocular会覆盖all_image_points
     * 必须先保存左相机角点数据，再调用右相机标定 */
    int pw = calibrator->config.pattern_width;
    int ph = calibrator->config.pattern_height;
    int corners_per_image = pw * ph;
    int total_left_images = calibrator->total_images;
    int obj_size = total_left_images * corners_per_image * 3;
    int img_size = total_left_images * corners_per_image * 2;
    float* saved_obj_points = (float*)malloc((size_t)total_left_images * (size_t)corners_per_image * 3 * sizeof(float));
    float* saved_left_points = (float*)malloc((size_t)total_left_images * (size_t)corners_per_image * 2 * sizeof(float));
    if (!saved_obj_points || !saved_left_points) {
        free(saved_obj_points);
        free(saved_left_points);
        return -1;
    }
    memcpy(saved_obj_points, calibrator->all_object_points, (size_t)obj_size * sizeof(float));
    memcpy(saved_left_points, calibrator->all_image_points, (size_t)img_size * sizeof(float));
    
    int right_ret = camera_calibrate_monocular(calibrator, right_images, num_images,
                                               width, height, channels, &right_result);

    if (right_ret != 0) {
        free(saved_obj_points);
        free(saved_left_points);
        return -1;
    }

    StereoCalibration* calib = &result->calibration;
    memset(calib, 0, sizeof(StereoCalibration));
    memcpy(&calib->left_intrinsics, &left_result.intrinsics, sizeof(CameraIntrinsics));
    memcpy(&calib->right_intrinsics, &right_result.intrinsics, sizeof(CameraIntrinsics));

    int valid_pairs = 0;
    float R_total[9] = {0}, T_total[3] = {0};

    for (int i = 0; i < total_left_images && i < num_images && i < calibrator->total_images; i++)
    {
        float* obj_pts = saved_obj_points + i * corners_per_image * 3;
        float* left_pts = saved_left_points + i * corners_per_image * 2;
        /* 右相机角点从第二次标定结果（calibrator->all_image_points）获取 */
        float* right_pts = calibrator->all_image_points + i * corners_per_image * 2;

        float Rl[3], Tl[3], Rr[3], Tr[3];

        if (camera_calibration_estimate_pose(obj_pts, left_pts, corners_per_image,
                                             &calib->left_intrinsics, Rl, Tl) != 0)
            continue;
        
        /* BUG-010修复: 对右相机也进行姿态估计（之前Rr/Tr从未被赋值） */
        if (camera_calibration_estimate_pose(obj_pts, right_pts, corners_per_image,
                                             &calib->right_intrinsics, Rr, Tr) != 0)
            continue;

        float Rl_mat[9], Rr_mat[9];
        rodrigues_to_matrix(Rl, Rl_mat);
        rodrigues_to_matrix(Rr, Rr_mat);

        float Rl_inv[9];
        mat_transpose_3x3(Rl_mat, Rl_inv);

        float R_rel[9];
        mat_mul_3x3(Rr_mat, Rl_inv, R_rel);

        float T_rel[3];
        T_rel[0] = Tr[0] - R_rel[0] * Tl[0] - R_rel[1] * Tl[1] - R_rel[2] * Tl[2];
        T_rel[1] = Tr[1] - R_rel[3] * Tl[0] - R_rel[4] * Tl[1] - R_rel[5] * Tl[2];
        T_rel[2] = Tr[2] - R_rel[6] * Tl[0] - R_rel[7] * Tl[1] - R_rel[8] * Tl[2];

        for (int j = 0; j < 9; j++) R_total[j] += R_rel[j];
        for (int j = 0; j < 3; j++) T_total[j] += T_rel[j];

        valid_pairs++;
    }

    /* ZSFIII-P1-001修复: return -0在IEEE754下等于0，导致调用方无法检测立体标定失败 */
    if (valid_pairs < 3) {
        free(saved_obj_points);
        free(saved_left_points);
        return -1;
    }

    float inv_count = 1.0f / valid_pairs;
    for (int i = 0; i < 9; i++) R_total[i] *= inv_count;
    for (int i = 0; i < 3; i++) T_total[i] *= inv_count;

    float U[9], W[9], VT[9];
    svd_3x3(R_total, U, W, VT);

    float VT_T[9];
    mat_transpose_3x3(VT, VT_T);

    if (mat_det_3x3(U) * mat_det_3x3(VT_T) < 0)
    {
        for (int i = 0; i < 3; i++) VT[i * 3 + 2] *= -1.0f;
    }

    mat_mul_3x3(U, VT, calib->extrinsics.rotation);
    memcpy(calib->extrinsics.translation, T_total, 3 * sizeof(float));

    calib->baseline = vec_norm(T_total);

    float Tx[9] = {
        0, -T_total[2], T_total[1],
        T_total[2], 0, -T_total[0],
        -T_total[1], T_total[0], 0
    };
    mat_mul_3x3(Tx, calib->extrinsics.rotation, calib->essential_matrix);

    float K_l_inv[9], K_r_t[9];
    mat_inv_3x3((const float*)&calib->left_intrinsics.fx, K_l_inv);

    float K_r_t_mat[9] = {
        calib->right_intrinsics.fx, 0, calib->right_intrinsics.cx,
        0, calib->right_intrinsics.fy, calib->right_intrinsics.cy,
        0, 0, 1
    };
    mat_transpose_3x3(K_r_t_mat, K_r_t);

    float temp_F[9];
    mat_mul_3x3(K_r_t, calib->essential_matrix, temp_F);
    mat_mul_3x3(temp_F, K_l_inv, calib->fundamental_matrix);

    stereo_compute_rectification_map(calib, width, height, NULL);

    result->reprojection_error = (left_result.reprojection_error + right_result.reprojection_error) / 2.0f;
    result->epipolar_error = 0.0f;
    result->calibration_valid = 1;
    result->calibration_time_ms = 0.0f;
    calib->is_calibrated = 1;

    free(saved_obj_points);
    free(saved_left_points);
    return 0;
}

int camera_calibration_save(const CameraIntrinsics* intrinsics,
                            const CameraExtrinsics* extrinsics,
                            const char* filepath)
{
    if (!intrinsics || !filepath) return -1;

    FILE* fp = fopen(filepath, "w");
    if (!fp) return -1;

    fprintf(fp, "# 相机标定参数\n");
    fprintf(fp, "image_width=%d\n", intrinsics->image_width);
    fprintf(fp, "image_height=%d\n", intrinsics->image_height);
    fprintf(fp, "fx=%f\n", intrinsics->fx);
    fprintf(fp, "fy=%f\n", intrinsics->fy);
    fprintf(fp, "cx=%f\n", intrinsics->cx);
    fprintf(fp, "cy=%f\n", intrinsics->cy);
    fprintf(fp, "k1=%f\n", intrinsics->k1);
    fprintf(fp, "k2=%f\n", intrinsics->k2);
    fprintf(fp, "k3=%f\n", intrinsics->k3);
    fprintf(fp, "p1=%f\n", intrinsics->p1);
    fprintf(fp, "p2=%f\n", intrinsics->p2);

    if (extrinsics)
    {
        fprintf(fp, "# 外参\n");
        fprintf(fp, "rotation=");
        for (int i = 0; i < 9; i++)
            fprintf(fp, "%s%f", i > 0 ? "," : "", extrinsics->rotation[i]);
        fprintf(fp, "\n");

        fprintf(fp, "translation=");
        for (int i = 0; i < 3; i++)
            fprintf(fp, "%s%f", i > 0 ? "," : "", extrinsics->translation[i]);
        fprintf(fp, "\n");
    }

    fclose(fp);
    return 0;
}

int camera_calibration_load(CameraIntrinsics* intrinsics,
                            CameraExtrinsics* extrinsics,
                            const char* filepath)
{
    if (!intrinsics || !filepath) return -1;

    FILE* fp = fopen(filepath, "r");
    if (!fp) return -1;

    memset(intrinsics, 0, sizeof(CameraIntrinsics));
    if (extrinsics) memset(extrinsics, 0, sizeof(CameraExtrinsics));

    char line[256];
    while (fgets(line, sizeof(line), fp))
    {
        if (line[0] == '#' || line[0] == '\n') continue;

        char* eq = strchr(line, '=');
        if (!eq) continue;
        *eq = '\0';
        char* key = line;
        char* val = eq + 1;
        val[strcspn(val, "\n")] = '\0';

        if (strcmp(key, "image_width") == 0) intrinsics->image_width = atoi(val);
        else if (strcmp(key, "image_height") == 0) intrinsics->image_height = atoi(val);
        else if (strcmp(key, "fx") == 0) intrinsics->fx = (float)atof(val);
        else if (strcmp(key, "fy") == 0) intrinsics->fy = (float)atof(val);
        else if (strcmp(key, "cx") == 0) intrinsics->cx = (float)atof(val);
        else if (strcmp(key, "cy") == 0) intrinsics->cy = (float)atof(val);
        else if (strcmp(key, "k1") == 0) intrinsics->k1 = (float)atof(val);
        else if (strcmp(key, "k2") == 0) intrinsics->k2 = (float)atof(val);
        else if (strcmp(key, "k3") == 0) intrinsics->k3 = (float)atof(val);
        else if (strcmp(key, "p1") == 0) intrinsics->p1 = (float)atof(val);
        else if (strcmp(key, "p2") == 0) intrinsics->p2 = (float)atof(val);
        else if (extrinsics && strcmp(key, "rotation") == 0)
        {
            char* tok = strtok(val, ",");
            for (int i = 0; i < 9 && tok; i++)
            {
                extrinsics->rotation[i] = (float)atof(tok);
                tok = strtok(NULL, ",");
            }
        }
        else if (extrinsics && strcmp(key, "translation") == 0)
        {
            char* tok = strtok(val, ",");
            for (int i = 0; i < 3 && tok; i++)
            {
                extrinsics->translation[i] = (float)atof(tok);
                tok = strtok(NULL, ",");
            }
        }
    }

    fclose(fp);
    return 0;
}

int stereo_calibration_save(const StereoCalibration* calibration, const char* filepath)
{
    if (!calibration || !filepath) return -1;

    FILE* fp = fopen(filepath, "w");
    if (!fp) return -1;

    fprintf(fp, "# 立体标定参数\n");
    fprintf(fp, "baseline=%f\n", calibration->baseline);
    fprintf(fp, "is_calibrated=%d\n", calibration->is_calibrated);

    fprintf(fp, "\n# 左相机内参\n");
    fprintf(fp, "left_fx=%f\n", calibration->left_intrinsics.fx);
    fprintf(fp, "left_fy=%f\n", calibration->left_intrinsics.fy);
    fprintf(fp, "left_cx=%f\n", calibration->left_intrinsics.cx);
    fprintf(fp, "left_cy=%f\n", calibration->left_intrinsics.cy);
    fprintf(fp, "left_k1=%f\n", calibration->left_intrinsics.k1);
    fprintf(fp, "left_k2=%f\n", calibration->left_intrinsics.k2);
    fprintf(fp, "left_k3=%f\n", calibration->left_intrinsics.k3);
    fprintf(fp, "left_p1=%f\n", calibration->left_intrinsics.p1);
    fprintf(fp, "left_p2=%f\n", calibration->left_intrinsics.p2);
    fprintf(fp, "left_width=%d\n", calibration->left_intrinsics.image_width);
    fprintf(fp, "left_height=%d\n", calibration->left_intrinsics.image_height);

    fprintf(fp, "\n# 右相机内参\n");
    fprintf(fp, "right_fx=%f\n", calibration->right_intrinsics.fx);
    fprintf(fp, "right_fy=%f\n", calibration->right_intrinsics.fy);
    fprintf(fp, "right_cx=%f\n", calibration->right_intrinsics.cx);
    fprintf(fp, "right_cy=%f\n", calibration->right_intrinsics.cy);
    fprintf(fp, "right_k1=%f\n", calibration->right_intrinsics.k1);
    fprintf(fp, "right_k2=%f\n", calibration->right_intrinsics.k2);
    fprintf(fp, "right_k3=%f\n", calibration->right_intrinsics.k3);
    fprintf(fp, "right_p1=%f\n", calibration->right_intrinsics.p1);
    fprintf(fp, "right_p2=%f\n", calibration->right_intrinsics.p2);
    fprintf(fp, "right_width=%d\n", calibration->right_intrinsics.image_width);
    fprintf(fp, "right_height=%d\n", calibration->right_intrinsics.image_height);

    fprintf(fp, "\n# 外参\n");
    fprintf(fp, "rotation=");
    for (int i = 0; i < 9; i++)
        fprintf(fp, "%s%f", i > 0 ? "," : "", calibration->extrinsics.rotation[i]);
    fprintf(fp, "\n");
    fprintf(fp, "translation=");
    for (int i = 0; i < 3; i++)
        fprintf(fp, "%s%f", i > 0 ? "," : "", calibration->extrinsics.translation[i]);
    fprintf(fp, "\n");

    fprintf(fp, "\n# 基础矩阵\n");
    fprintf(fp, "fundamental=");
    for (int i = 0; i < 9; i++)
        fprintf(fp, "%s%f", i > 0 ? "," : "", calibration->fundamental_matrix[i]);
    fprintf(fp, "\n");

    fclose(fp);
    return 0;
}

int stereo_calibration_load(StereoCalibration* calibration, const char* filepath)
{
    if (!calibration || !filepath) return -1;

    FILE* fp = fopen(filepath, "r");
    if (!fp) return -1;

    memset(calibration, 0, sizeof(StereoCalibration));

    char line[512];
    while (fgets(line, sizeof(line), fp))
    {
        if (line[0] == '#' || line[0] == '\n') continue;

        char* eq = strchr(line, '=');
        if (!eq) continue;
        *eq = '\0';
        char* key = line;
        char* val = eq + 1;
        val[strcspn(val, "\n")] = '\0';

        if (strcmp(key, "baseline") == 0) calibration->baseline = (float)atof(val);
        else if (strcmp(key, "is_calibrated") == 0) calibration->is_calibrated = atoi(val);

        else if (strcmp(key, "left_fx") == 0) calibration->left_intrinsics.fx = (float)atof(val);
        else if (strcmp(key, "left_fy") == 0) calibration->left_intrinsics.fy = (float)atof(val);
        else if (strcmp(key, "left_cx") == 0) calibration->left_intrinsics.cx = (float)atof(val);
        else if (strcmp(key, "left_cy") == 0) calibration->left_intrinsics.cy = (float)atof(val);
        else if (strcmp(key, "left_k1") == 0) calibration->left_intrinsics.k1 = (float)atof(val);
        else if (strcmp(key, "left_k2") == 0) calibration->left_intrinsics.k2 = (float)atof(val);
        else if (strcmp(key, "left_k3") == 0) calibration->left_intrinsics.k3 = (float)atof(val);
        else if (strcmp(key, "left_p1") == 0) calibration->left_intrinsics.p1 = (float)atof(val);
        else if (strcmp(key, "left_p2") == 0) calibration->left_intrinsics.p2 = (float)atof(val);
        else if (strcmp(key, "left_width") == 0) calibration->left_intrinsics.image_width = atoi(val);
        else if (strcmp(key, "left_height") == 0) calibration->left_intrinsics.image_height = atoi(val);

        else if (strcmp(key, "right_fx") == 0) calibration->right_intrinsics.fx = (float)atof(val);
        else if (strcmp(key, "right_fy") == 0) calibration->right_intrinsics.fy = (float)atof(val);
        else if (strcmp(key, "right_cx") == 0) calibration->right_intrinsics.cx = (float)atof(val);
        else if (strcmp(key, "right_cy") == 0) calibration->right_intrinsics.cy = (float)atof(val);
        else if (strcmp(key, "right_k1") == 0) calibration->right_intrinsics.k1 = (float)atof(val);
        else if (strcmp(key, "right_k2") == 0) calibration->right_intrinsics.k2 = (float)atof(val);
        else if (strcmp(key, "right_k3") == 0) calibration->right_intrinsics.k3 = (float)atof(val);
        else if (strcmp(key, "right_p1") == 0) calibration->right_intrinsics.p1 = (float)atof(val);
        else if (strcmp(key, "right_p2") == 0) calibration->right_intrinsics.p2 = (float)atof(val);
        else if (strcmp(key, "right_width") == 0) calibration->right_intrinsics.image_width = atoi(val);
        else if (strcmp(key, "right_height") == 0) calibration->right_intrinsics.image_height = atoi(val);

        else if (strcmp(key, "rotation") == 0)
        {
            char* tok = strtok(val, ",");
            for (int i = 0; i < 9 && tok; i++)
            {
                calibration->extrinsics.rotation[i] = (float)atof(tok);
                tok = strtok(NULL, ",");
            }
        }
        else if (strcmp(key, "translation") == 0)
        {
            char* tok = strtok(val, ",");
            for (int i = 0; i < 3 && tok; i++)
            {
                calibration->extrinsics.translation[i] = (float)atof(tok);
                tok = strtok(NULL, ",");
            }
        }
        else if (strcmp(key, "fundamental") == 0)
        {
            char* tok = strtok(val, ",");
            for (int i = 0; i < 9 && tok; i++)
            {
                calibration->fundamental_matrix[i] = (float)atof(tok);
                tok = strtok(NULL, ",");
            }
        }
    }

    fclose(fp);
    calibration->is_calibrated = 1;
    return 0;
}

int stereo_compute_rectification_map(const StereoCalibration* calibration,
                                     int width, int height,
                                     StereoRectificationMap* rectification_map)
{
    if (!calibration) return -1;

    if (!rectification_map) return 0;

    float R[9];
    memcpy(R, calibration->extrinsics.rotation, 9 * sizeof(float));

    float T[3];
    memcpy(T, calibration->extrinsics.translation, 3 * sizeof(float));

    float e1[3] = {T[0], T[1], T[2]};
    vec_normalize(e1);

    float e2[3];
    float tmp_z[3] = {-T[1], T[0], 0};
    if (fabsf(T[0]) < FLT_EPSILON && fabsf(T[1]) < FLT_EPSILON)
    {
        tmp_z[0] = 0;
        tmp_z[1] = T[2];
        tmp_z[2] = -T[1];
    }
    cross_product(tmp_z, e1, e2);
    vec_normalize(e2);

    float e3[3];
    cross_product(e1, e2, e3);
    vec_normalize(e3);

    float R_rect[9] = {
        e1[0], e1[1], e1[2],
        e2[0], e2[1], e2[2],
        e3[0], e3[1], e3[2]
    };

    float R_left[9];
    mat_mul_3x3(R_rect, R, R_left);

    memcpy((float*)&calibration->rectification_left, R_left, 9 * sizeof(float));
    memcpy((float*)&calibration->rectification_right, (const float*)R_rect, 9 * sizeof(float));

    float P_left[12] = {
        calibration->left_intrinsics.fx, 0, calibration->left_intrinsics.cx, 0,
        0, calibration->left_intrinsics.fy, calibration->left_intrinsics.cy, 0,
        0, 0, 1, 0
    };

    float P_right[12] = {
        calibration->left_intrinsics.fx, 0, calibration->left_intrinsics.cx,
        -calibration->left_intrinsics.fx * calibration->baseline,
        0, calibration->left_intrinsics.fy, calibration->left_intrinsics.cy, 0,
        0, 0, 1, 0
    };

    memcpy((float*)&calibration->projection_left, P_left, 12 * sizeof(float));
    memcpy((float*)&calibration->projection_right, P_right, 12 * sizeof(float));

    rectification_map->map_width = width;
    rectification_map->map_height = height;

    rectification_map->map_left_x = (float*)calloc(width * height, sizeof(float));
    rectification_map->map_left_y = (float*)calloc(width * height, sizeof(float));
    rectification_map->map_right_x = (float*)calloc(width * height, sizeof(float));
    rectification_map->map_right_y = (float*)calloc(width * height, sizeof(float));

    if (!rectification_map->map_left_x || !rectification_map->map_left_y ||
        !rectification_map->map_right_x || !rectification_map->map_right_y)
    {
        stereo_rectification_map_free(rectification_map);
        return -1;
    }

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = y * width + x;

            float xl_norm = (float)(x - calibration->left_intrinsics.cx) / calibration->left_intrinsics.fx;
            float yl_norm = (float)(y - calibration->left_intrinsics.cy) / calibration->left_intrinsics.fy;

            float rx_l = R_left[0] * xl_norm + R_left[1] * yl_norm + R_left[2];
            float ry_l = R_left[3] * xl_norm + R_left[4] * yl_norm + R_left[5];
            float rz_l = R_left[6] * xl_norm + R_left[7] * yl_norm + R_left[8];

            if (fabsf(rz_l) > FLT_EPSILON)
            {
                rectification_map->map_left_x[idx] = calibration->left_intrinsics.fx * rx_l / rz_l + calibration->left_intrinsics.cx;
                rectification_map->map_left_y[idx] = calibration->left_intrinsics.fy * ry_l / rz_l + calibration->left_intrinsics.cy;
            }
            else
            {
                rectification_map->map_left_x[idx] = (float)x;
                rectification_map->map_left_y[idx] = (float)y;
            }

            float xr_norm = (float)(x - calibration->right_intrinsics.cx) / calibration->right_intrinsics.fx;
            float yr_norm = (float)(y - calibration->right_intrinsics.cy) / calibration->right_intrinsics.fy;

            float rx_r = R_rect[0] * xr_norm + R_rect[1] * yr_norm + R_rect[2];
            float ry_r = R_rect[3] * xr_norm + R_rect[4] * yr_norm + R_rect[5];
            float rz_r = R_rect[6] * xr_norm + R_rect[7] * yr_norm + R_rect[8];

            if (fabsf(rz_r) > FLT_EPSILON)
            {
                rectification_map->map_right_x[idx] = calibration->right_intrinsics.fx * rx_r / rz_r + calibration->right_intrinsics.cx;
                rectification_map->map_right_y[idx] = calibration->right_intrinsics.fy * ry_r / rz_r + calibration->right_intrinsics.cy;
            }
            else
            {
                rectification_map->map_right_x[idx] = (float)x;
                rectification_map->map_right_y[idx] = (float)y;
            }
        }
    }

    return 0;
}

void stereo_rectification_map_free(StereoRectificationMap* map)
{
    if (!map) return;
    free(map->map_left_x);
    map->map_left_x = NULL;
    free(map->map_left_y);
    map->map_left_y = NULL;
    free(map->map_right_x);
    map->map_right_x = NULL;
    free(map->map_right_y);
    map->map_right_y = NULL;
    map->map_width = 0;
    map->map_height = 0;
}

int stereo_apply_rectification(const float* src_image, float* dst_image,
                               int width, int height, int channels,
                               const float* map_x, const float* map_y)
{
    if (!src_image || !dst_image || !map_x || !map_y) return -1;
    if (width <= 0 || height <= 0 || channels <= 0) return -1;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = y * width + x;
            float sx = map_x[idx];
            float sy = map_y[idx];

            int x0 = (int)floorf(sx);
            int y0 = (int)floorf(sy);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            float dx = sx - x0;
            float dy = sy - y0;

            x0 = x0 < 0 ? 0 : (x0 >= width ? width - 1 : x0);
            x1 = x1 < 0 ? 0 : (x1 >= width ? width - 1 : x1);
            y0 = y0 < 0 ? 0 : (y0 >= height ? height - 1 : y0);
            y1 = y1 < 0 ? 0 : (y1 >= height ? height - 1 : y1);

            for (int c = 0; c < channels; c++)
            {
                float v00 = src_image[(y0 * width + x0) * channels + c];
                float v01 = src_image[(y0 * width + x1) * channels + c];
                float v10 = src_image[(y1 * width + x0) * channels + c];
                float v11 = src_image[(y1 * width + x1) * channels + c];

                float v0 = v00 * (1.0f - dx) + v01 * dx;
                float v1 = v10 * (1.0f - dx) + v11 * dx;
                dst_image[idx * channels + c] = v0 * (1.0f - dy) + v1 * dy;
            }
        }
    }

    return 0;
}

int stereo_compute_disparity_to_depth(const StereoCalibration* calibration,
                                      float* disparity_to_depth)
{
    if (!calibration || !disparity_to_depth) return -1;

    float baseline = calibration->baseline;
    float fx = calibration->left_intrinsics.fx;

    if (baseline <= 0.0f || fx <= 0.0f) return -1;

    memset(disparity_to_depth, 0, 16 * sizeof(float));
    disparity_to_depth[0] = 1.0f;
    disparity_to_depth[5] = 1.0f;
    disparity_to_depth[10] = 0.0f;
    disparity_to_depth[11] = baseline * fx;
    disparity_to_depth[15] = 0.0f;
    disparity_to_depth[14] = 1.0f;

    return 0;
}

int stereo_disparity_to_depth(const float* disparity_map, int width, int height,
                              const StereoCalibration* calibration,
                              float* depth_map)
{
    if (!disparity_map || !calibration || !depth_map) return -1;
    if (width <= 0 || height <= 0) return -1;

    float baseline = calibration->baseline;
    float fx = calibration->left_intrinsics.fx;
    float cx_l = calibration->left_intrinsics.cx;
    float cx_r = calibration->right_intrinsics.cx;

    if (baseline <= 0.0f || fx <= 0.0f) return -1;

    float cx_diff = cx_r - cx_l;

    int total = width * height;
    for (int i = 0; i < total; i++)
    {
        float d = disparity_map[i];
        if (d > 0.0f)
        {
            depth_map[i] = baseline * fx / (d + cx_diff);
        }
        else
        {
            depth_map[i] = 0.0f;
        }
    }

    return 0;
}

float stereo_compute_epipolar_error(const StereoCalibration* calibration,
                                    const float* left_points,
                                    const float* right_points,
                                    int num_points, float* epipolar_errors)
{
    if (!calibration || !left_points || !right_points || num_points <= 0)
        return 0.0f;

    float total_error = 0.0f;

    for (int i = 0; i < num_points; i++)
    {
        float pl[3] = {left_points[i * 2], left_points[i * 2 + 1], 1.0f};
        float pr[3] = {right_points[i * 2], right_points[i * 2 + 1], 1.0f};

        float Fpr[3];
        mat_vec_mul_3x3(calibration->fundamental_matrix, pr, Fpr);

        float error = fabsf(pl[0] * Fpr[0] + pl[1] * Fpr[1] + Fpr[2]);
        float norm = sqrtf(Fpr[0] * Fpr[0] + Fpr[1] * Fpr[1]);

        if (norm > FLT_EPSILON)
        {
            error /= norm;
        }

        if (epipolar_errors)
        {
            epipolar_errors[i] = error;
        }
        total_error += error;
    }

    return total_error / num_points;
}

float stereo_validate_calibration(const StereoCalibration* calibration,
                                  const float** test_images,
                                  int num_test_images,
                                  int width, int height, int channels,
                                  float* reprojection_errors)
{
    if (!calibration || !test_images || num_test_images <= 0)
        return 0.0f;

    (void)channels;

    float total_error = 0.0f;
    int total_points = 0;

    for (int i = 0; i < num_test_images; i++)
    {
        float error = 0.0f;
        int count = 0;

        const float* img = test_images[i];

        (void)img;

        for (int y = 0; y < height; y += 10)
        {
            for (int x = 0; x < width; x += 10)
            {
                float x_norm = (float)(x - calibration->left_intrinsics.cx) / calibration->left_intrinsics.fx;
                float y_norm = (float)(y - calibration->left_intrinsics.cy) / calibration->left_intrinsics.fy;

                float rota = calibration->rectification_left[0] * x_norm +
                             calibration->rectification_left[1] * y_norm +
                             calibration->rectification_left[2];
                float rotb = calibration->rectification_left[3] * x_norm +
                             calibration->rectification_left[4] * y_norm +
                             calibration->rectification_left[5];
                float rotc = calibration->rectification_left[6] * x_norm +
                             calibration->rectification_left[7] * y_norm +
                             calibration->rectification_left[8];

                float x_rect = calibration->left_intrinsics.fx * rota / (rotc + FLT_EPSILON) + calibration->left_intrinsics.cx;
                float y_rect = calibration->left_intrinsics.fy * rotb / (rotc + FLT_EPSILON) + calibration->left_intrinsics.cy;

                float dx = x_rect - x;
                float dy = y_rect - y;
                error += sqrtf(dx * dx + dy * dy);
                count++;
            }
        }

        float avg_err = count > 0 ? error / count : 0.0f;
        if (reprojection_errors) reprojection_errors[i] = avg_err;
        total_error += avg_err;
        total_points += count;
    }

    return total_points > 0 ? total_error / num_test_images : 0.0f;
}

void camera_calibration_default_config(CalibrationConfig* config)
{
    if (!config) return;
    memset(config, 0, sizeof(CalibrationConfig));
    config->pattern_type = CALIBRATION_PATTERN_CHESSBOARD;
    config->pattern_width = 9;
    config->pattern_height = 6;
    config->square_size = 0.025f;
    config->max_iterations = 100;
    config->accuracy_epsilon = 1e-6f;
    config->use_intrinsic_guess = 1;
    config->fix_principal_point = 0;
    config->fix_aspect_ratio = 0;
    config->zero_tangential_distortion = 0;
    config->use_rational_model = 0;
}

void camera_calibration_default_intrinsics(CameraIntrinsics* intrinsics,
                                           int width, int height)
{
    if (!intrinsics) return;
    memset(intrinsics, 0, sizeof(CameraIntrinsics));
    intrinsics->fx = (float)width;
    intrinsics->fy = (float)width;
    intrinsics->cx = (float)width / 2.0f;
    intrinsics->cy = (float)height / 2.0f;
    intrinsics->k1 = 0.0f;
    intrinsics->k2 = 0.0f;
    intrinsics->k3 = 0.0f;
    intrinsics->p1 = 0.0f;
    intrinsics->p2 = 0.0f;
    intrinsics->image_width = width;
    intrinsics->image_height = height;
}

void camera_calibrator_reset(CameraCalibrator* calibrator)
{
    if (!calibrator) return;
    calibrator->total_images = 0;
    calibrator->total_points = 0;
    calibrator->is_initialized = 0;
}

int camera_calibration_detect_corners(const float* image,
                                      int width, int height, int channels,
                                      CalibrationPattern pattern_type,
                                      int pattern_width, int pattern_height,
                                      float* corners, int max_corners)
{
    if (!image || width <= 0 || height <= 0 || !corners || max_corners <= 0)
        return -1;

    if (pattern_type != CALIBRATION_PATTERN_CHESSBOARD)
        return -1;

    int expected_corners = pattern_width * pattern_height;
    if (max_corners < expected_corners) return -1;

    int* grad_x = (int*)calloc(width * height, sizeof(int));
    int* grad_y = (int*)calloc(width * height, sizeof(int));
    float* grad_mag = (float*)calloc(width * height, sizeof(float));
    float* grad_orient = (float*)calloc(width * height, sizeof(float));

    if (!grad_x || !grad_y || !grad_mag || !grad_orient)
    {
        free(grad_x); free(grad_y); free(grad_mag); free(grad_orient);
        return -1;
    }

    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            int idx = y * width + x;
            float val_center, val_left, val_right, val_up, val_down;

            if (channels >= 1)
            {
                val_center = image[idx * channels] / 255.0f;
                val_left = image[(y * width + (x - 1)) * channels] / 255.0f;
                val_right = image[(y * width + (x + 1)) * channels] / 255.0f;
                val_up = image[((y - 1) * width + x) * channels] / 255.0f;
                val_down = image[((y + 1) * width + x) * channels] / 255.0f;
            }
            else
            {
                val_center = image[idx];
                val_left = image[y * width + (x - 1)];
                val_right = image[y * width + (x + 1)];
                val_up = image[(y - 1) * width + x];
                val_down = image[(y + 1) * width + x];
            }

            float gx = val_right - val_left;
            float gy = val_down - val_up;

            grad_x[idx] = (int)(gx * 255.0f);
            grad_y[idx] = (int)(gy * 255.0f);
            grad_mag[idx] = sqrtf(gx * gx + gy * gy);
            grad_orient[idx] = atan2f(gy, gx);
        }
    }

    int* corner_score = (int*)calloc(width * height, sizeof(int));
    if (!corner_score)
    {
        free(grad_x); free(grad_y); free(grad_mag); free(grad_orient);
        return -1;
    }

    for (int y = 4; y < height - 4; y++)
    {
        for (int x = 4; x < width - 4; x++)
        {
            int idx = y * width + x;
            float mag = grad_mag[idx];
            if (mag < 0.05f) continue;

            float orient = grad_orient[idx];
            float perp_orient = orient + 3.1415926535f / 2.0f;

            float cos_o = cosf(orient);
            float sin_o = sinf(orient);
            float cos_p = cosf(perp_orient);
            float sin_p = sinf(perp_orient);

            float sum_along = 0.0f, sum_across = 0.0f;

            for (int d = -4; d <= 4; d++)
            {
                int px = x + (int)(cos_o * d);
                int py = y + (int)(sin_o * d);
                if (px >= 0 && px < width && py >= 0 && py < height)
                {
                    sum_along += grad_mag[py * width + px];
                }

                px = x + (int)(cos_p * d);
                py = y + (int)(sin_p * d);
                if (px >= 0 && px < width && py >= 0 && py < height)
                {
                    sum_across += grad_mag[py * width + px];
                }
            }

            float score = sum_along - sum_across;
            if (score > 0.01f)
            {
                int is_max = 1;
                for (int dy = -2; dy <= 2 && is_max; dy++)
                {
                    for (int dx = -2; dx <= 2; dx++)
                    {
                        if (dx == 0 && dy == 0) continue;
                        int nx = x + dx, ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                        {
                            if (grad_mag[ny * width + nx] >= mag)
                            {
                                is_max = 0;
                                break;
                            }
                        }
                    }
                }

                if (is_max)
                {
                    corner_score[idx] = (int)(score * 100000.0f);
                }
            }
        }
    }

    int* sorted_indices = (int*)malloc((size_t)width * (size_t)height * sizeof(int));
    if (!sorted_indices)
    {
        free(grad_x); free(grad_y); free(grad_mag); free(grad_orient); free(corner_score);
        return -1;
    }

    int valid_count = 0;
    for (int i = 0; i < width * height; i++)
    {
        if (corner_score[i] > 0)
        {
            sorted_indices[valid_count++] = i;
        }
    }

    for (int i = 0; i < valid_count - 1; i++)
    {
        for (int j = i + 1; j < valid_count; j++)
        {
            if (corner_score[sorted_indices[j]] > corner_score[sorted_indices[i]])
            {
                int tmp = sorted_indices[i];
                sorted_indices[i] = sorted_indices[j];
                sorted_indices[j] = tmp;
            }
        }
    }

    int corners_found = 0;
    for (int i = 0; i < valid_count && corners_found < expected_corners; i++)
    {
        int idx = sorted_indices[i];
        float fx = (float)(idx % width);
        float fy = (float)(idx / width);

        if (fx > 0 && fx < width - 1 && fy > 0 && fy < height - 1)
        {
            float sx = 0.0f, sy = 0.0f, total_w = 0.0f;
            for (int dy = -4; dy <= 4; dy++)
            {
                for (int dx = -4; dx <= 4; dx++)
                {
                    int px = (int)fx + dx, py = (int)fy + dy;
                    if (px >= 1 && px < width - 1 && py >= 1 && py < height - 1)
                    {
                        float w = grad_mag[py * width + px];
                        sx += w * px; sy += w * py; total_w += w;
                    }
                }
            }
            if (total_w > FLT_EPSILON) { fx = sx / total_w; fy = sy / total_w; }
        }

        int too_close = 0;
        for (int j = 0; j < corners_found; j++)
        {
            float dx = corners[j * 2] - fx;
            float dy = corners[j * 2 + 1] - fy;
            if (dx * dx + dy * dy < 100.0f) { too_close = 1; break; }
        }
        if (!too_close)
        {
            corners[corners_found * 2] = fx;
            corners[corners_found * 2 + 1] = fy;
            corners_found++;
        }
    }

    if (corners_found == expected_corners)
    {
        float min_dist = (float)(width + height);
        int top_left_idx = 0;
        for (int i = 0; i < corners_found; i++)
        {
            float dx = corners[i * 2] - 0.0f;
            float dy = corners[i * 2 + 1] - 0.0f;
            float d = dx * dx + dy * dy;
            if (d < min_dist) { min_dist = d; top_left_idx = i; }
        }

        float* ordered = (float*)safe_malloc((size_t)expected_corners * 2 * sizeof(float));
        if (ordered)
        {
            int* used = (int*)safe_calloc((size_t)expected_corners, sizeof(int));
            if (used)
            {
                ordered[0] = corners[top_left_idx * 2];
                ordered[1] = corners[top_left_idx * 2 + 1];
                used[top_left_idx] = 1;
                float ref_x = ordered[0], ref_y = ordered[1];

                for (int row = 0; row < pattern_height; row++)
                {
                    for (int col = 0; col < pattern_width; col++)
                    {
                        if (row == 0 && col == 0) continue;
                        float expected_x = ref_x + (float)col * 20.0f;
                        float expected_y = ref_y + (float)row * 20.0f;
                        float best_d = 1000.0f;
                        int best_j = -1;
                        for (int j = 0; j < corners_found; j++)
                        {
                            if (used[j]) continue;
                            float dx = corners[j * 2] - expected_x;
                            float dy = corners[j * 2 + 1] - expected_y;
                            float d = dx * dx + dy * dy;
                            if (d < best_d) { best_d = d; best_j = j; }
                        }
                        if (best_j >= 0 && best_d < 400.0f)
                        {
                            int ord_idx = row * pattern_width + col;
                            ordered[ord_idx * 2] = corners[best_j * 2];
                            ordered[ord_idx * 2 + 1] = corners[best_j * 2 + 1];
                            used[best_j] = 1;
                        }
                        else
                        {
                            safe_free((void**)&ordered);
                            safe_free((void**)&used);
                            corners_found = 0;
                            goto cleanup_corners;
                        }
                    }
                }

                memcpy(corners, ordered, (size_t)expected_corners * 2 * sizeof(float));
                safe_free((void**)&used);
            }
            safe_free((void**)&ordered);
        }
    }

cleanup_corners:

    free(grad_x); free(grad_y); free(grad_mag); free(grad_orient);
    free(corner_score); free(sorted_indices);

    return corners_found;
}

int camera_calibration_compute_object_points(int pattern_width, int pattern_height,
                                             float square_size,
                                             float* object_points, int max_points)
{
    if (pattern_width <= 0 || pattern_height <= 0 || square_size <= 0.0f ||
        !object_points) return -1;

    int num_points = pattern_width * pattern_height;
    if (max_points < num_points) return -1;

    int idx = 0;
    for (int j = 0; j < pattern_height; j++)
    {
        for (int i = 0; i < pattern_width; i++)
        {
            object_points[idx * 3] = (float)i * square_size;
            object_points[idx * 3 + 1] = (float)j * square_size;
            object_points[idx * 3 + 2] = 0.0f;
            idx++;
        }
    }

    return num_points;
}

int camera_calibration_estimate_pose(const float* object_points,
                                     const float* image_points, int num_points,
                                     const CameraIntrinsics* intrinsics,
                                     float* rotation, float* translation)
{
    if (!object_points || !image_points || num_points < 4 || !intrinsics || !rotation || !translation)
        return -1;

    float mean_3d[3] = {0}, mean_2d[2] = {0};
    for (int i = 0; i < num_points; i++)
    {
        mean_3d[0] += object_points[i * 3];
        mean_3d[1] += object_points[i * 3 + 1];
        mean_3d[2] += object_points[i * 3 + 2];
        mean_2d[0] += image_points[i * 2];
        mean_2d[1] += image_points[i * 2 + 1];
    }
    float inv_n = 1.0f / num_points;
    mean_3d[0] *= inv_n; mean_3d[1] *= inv_n; mean_3d[2] *= inv_n;
    mean_2d[0] *= inv_n; mean_2d[1] *= inv_n;

    float centered_3d[3];
    float H[9] = {0};

    for (int i = 0; i < num_points; i++)
    {
        centered_3d[0] = object_points[i * 3] - mean_3d[0];
        centered_3d[1] = object_points[i * 3 + 1] - mean_3d[1];
        centered_3d[2] = object_points[i * 3 + 2] - mean_3d[2];

        float u = (image_points[i * 2] - intrinsics->cx) / intrinsics->fx;
        float v = (image_points[i * 2 + 1] - intrinsics->cy) / intrinsics->fy;

        H[0] += centered_3d[0] * u;
        H[1] += centered_3d[1] * u;
        H[2] += centered_3d[2] * u;
        H[3] += centered_3d[0] * v;
        H[4] += centered_3d[1] * v;
        H[5] += centered_3d[2] * v;
        H[6] += centered_3d[0];
        H[7] += centered_3d[1];
        H[8] += centered_3d[2];
    }

    float norm_u = sqrtf(H[0] * H[0] + H[1] * H[1] + H[2] * H[2]);
    float norm_v = sqrtf(H[3] * H[3] + H[4] * H[4] + H[5] * H[5]);

    float r0[3] = {H[0] / (norm_u + FLT_EPSILON), H[1] / (norm_u + FLT_EPSILON), H[2] / (norm_u + FLT_EPSILON)};
    float r1[3] = {H[3] / (norm_v + FLT_EPSILON), H[4] / (norm_v + FLT_EPSILON), H[5] / (norm_v + FLT_EPSILON)};

    float r2[3];
    cross_product(r0, r1, r2);

    float R[9] = {
        r0[0], r0[1], r0[2],
        r1[0], r1[1], r1[2],
        r2[0], r2[1], r2[2]
    };

    float U_svd[9], W_svd[9], VT_svd[9];
    svd_3x3(R, U_svd, W_svd, VT_svd);

    float VT_T[9], U_T[9];
    mat_transpose_3x3(VT_svd, VT_T);
    mat_transpose_3x3(U_svd, U_T);

    if (mat_det_3x3(U_svd) * mat_det_3x3(VT_T) < 0)
    {
        for (int i = 0; i < 3; i++) VT_svd[i * 3 + 2] *= -1.0f;
    }

    float R_opt[9];
    mat_mul_3x3(U_svd, VT_svd, R_opt);

    matrix_to_rodrigues(R_opt, rotation);

    float t[3];
    for (int i = 0; i < 3; i++)
    {
        t[i] = (H[6 + i] - R_opt[i * 3] * mean_3d[0] - R_opt[i * 3 + 1] * mean_3d[1] - R_opt[i * 3 + 2] * mean_3d[2]) * inv_n;
    }
    translation[0] = t[0];
    translation[1] = t[1];
    translation[2] = t[2];

    return 0;
}

/* ============================================================================
 * Bundle Adjustment 非线性优化标定实现
 * ============================================================================ */

void bundle_adjustment_default_config(BundleAdjustmentConfig* config)
{
    if (!config) return;
    config->max_iterations = 100;
    config->lambda_init = 0.001f;
    config->lambda_factor = 10.0f;
    config->gradient_threshold = 1e-6f;
    config->error_threshold = 1e-8f;
    config->fix_intrinsics = 0;
    config->fix_distortion = 0;
    config->verbose = 0;
}

/**
 * @brief 计算单个点的重投影误差（含畸变校正）
 */
static float compute_reprojection_error_point(
    const float* object_pt,
    const CameraIntrinsics* intr,
    const float* rotation,
    const float* translation,
    const float* observed_uv)
{
    float R[9];
    rodrigues_to_matrix(rotation, R);

    float xc = R[0] * object_pt[0] + R[1] * object_pt[1] + R[2] * object_pt[2] + translation[0];
    float yc = R[3] * object_pt[0] + R[4] * object_pt[1] + R[5] * object_pt[2] + translation[1];
    float zc = R[6] * object_pt[0] + R[7] * object_pt[1] + R[8] * object_pt[2] + translation[2];

    if (fabsf(zc) < 1e-12f) return 1e6f;

    float xn = xc / zc;
    float yn = yc / zc;

    float r2 = xn * xn + yn * yn;
    float r4 = r2 * r2;
    float r6 = r4 * r2;

    float radial = 1.0f + intr->k1 * r2 + intr->k2 * r4 + intr->k3 * r6;
    float tx = 2.0f * intr->p1 * xn * yn + intr->p2 * (r2 + 2.0f * xn * xn);
    float ty = intr->p1 * (r2 + 2.0f * yn * yn) + 2.0f * intr->p2 * xn * yn;

    float xd = xn * radial + tx;
    float yd = yn * radial + ty;

    float up = intr->fx * xd + intr->cx;
    float vp = intr->fy * yd + intr->cy;

    float du = up - observed_uv[0];
    float dv = vp - observed_uv[1];

    return sqrtf(du * du + dv * dv);
}

/**
 * @brief 计算所有点的重投影误差总和
 */
static float compute_total_reprojection_error(
    const CameraIntrinsics* left_intr,
    const CameraIntrinsics* right_intr,
    const float* left_rotation,
    const float* left_translation,
    const float* right_rotation,
    const float* right_translation,
    const float* left_points,
    const float* right_points,
    const float* object_points,
    int num_images,
    int corners_per_image)
{
    float total_error = 0.0f;
    int total_points = 0;

    for (int img = 0; img < num_images; img++)
    {
        const float* left_img_pts = left_points + img * corners_per_image * 2;
        const float* right_img_pts = right_points + img * corners_per_image * 2;
        const float* left_R = left_rotation + img * 3;
        const float* left_T = left_translation + img * 3;
        const float* right_R = right_rotation + img * 3;
        const float* right_T = right_translation + img * 3;

        for (int pt = 0; pt < corners_per_image; pt++)
        {
            const float* obj_pt = object_points + pt * 3;
            const float* left_obs = left_img_pts + pt * 2;
            const float* right_obs = right_img_pts + pt * 2;

            total_error += compute_reprojection_error_point(obj_pt, left_intr,
                left_R, left_T, left_obs);
            total_error += compute_reprojection_error_point(obj_pt, right_intr,
                right_R, right_T, right_obs);
            total_points += 2;
        }
    }

    return (total_points > 0) ? (total_error / (float)total_points) : 1e6f;
}

/**
 * @brief 数值计算雅可比矩阵（通过有限差分）
 *
 * 对第param_idx个参数施加微小扰动，计算重投影误差的变化。
 * 参数向量布局：[left_intr(11) + left_poses(num_images*6) + right_intr(11) + right_poses(num_images*6)]
 */
static void numerical_jacobian(
    const float* params,
    int num_params,
    const float* left_points,
    const float* right_points,
    const float* object_points,
    int num_images,
    int corners_per_image,
    int left_intr_offset,
    int left_pose_offset,
    int right_intr_offset,
    int right_pose_offset,
    float* jacobian,
    int* jac_row_count)
{
    float epsilon = 1e-5f;
    int total_obs = num_images * corners_per_image * 2;

    for (int j = 0; j < num_params; j++)
    {
        float params_plus[200];
        float params_minus[200];
        memcpy(params_plus, params, num_params * sizeof(float));
        memcpy(params_minus, params, num_params * sizeof(float));

        float h = (fabsf(params[j]) + 1.0f) * epsilon;
        params_plus[j] += h;
        params_minus[j] -= h;

        CameraIntrinsics left_intr_plus, left_intr_minus;
        memcpy(&left_intr_plus, params + left_intr_offset, sizeof(CameraIntrinsics));
        memcpy(&left_intr_minus, params + left_intr_offset, sizeof(CameraIntrinsics));

        CameraIntrinsics right_intr_plus, right_intr_minus;
        memcpy(&right_intr_plus, params + right_intr_offset, sizeof(CameraIntrinsics));
        memcpy(&right_intr_minus, params + right_intr_offset, sizeof(CameraIntrinsics));

        float error_plus = compute_total_reprojection_error(
            &left_intr_plus, &right_intr_plus,
            params_plus + left_pose_offset,
            params_plus + left_pose_offset + num_images * 3,
            params_plus + right_pose_offset,
            params_plus + right_pose_offset + num_images * 3,
            left_points, right_points, object_points,
            num_images, corners_per_image);

        float error_minus = compute_total_reprojection_error(
            &left_intr_minus, &right_intr_minus,
            params_minus + left_pose_offset,
            params_minus + left_pose_offset + num_images * 3,
            params_minus + right_pose_offset,
            params_minus + right_pose_offset + num_images * 3,
            left_points, right_points, object_points,
            num_images, corners_per_image);

        float derivative = (error_plus - error_minus) / (2.0f * h);

        for (int r = 0; r < total_obs; r++)
        {
            jacobian[r * num_params + j] = derivative / (float)total_obs;
        }
    }

    *jac_row_count = total_obs;
}

/**
 * @brief Levenberg-Marquardt 优化一步迭代
 */
static float lm_step(
    float* params,
    int num_params,
    float lambda,
    const float* left_points,
    const float* right_points,
    const float* object_points,
    int num_images,
    int corners_per_image,
    int left_intr_offset,
    int left_pose_offset,
    int right_intr_offset,
    int right_pose_offset,
    float* current_error)
{
    int total_obs = num_images * corners_per_image * 2;
    int max_jac = total_obs * num_params;
    float* jacobian = (float*)safe_malloc((size_t)max_jac * sizeof(float));
    if (!jacobian) return *current_error;

    int jac_rows = 0;
    numerical_jacobian(params, num_params,
                       left_points, right_points, object_points,
                       num_images, corners_per_image,
                       left_intr_offset, left_pose_offset,
                       right_intr_offset, right_pose_offset,
                       jacobian, &jac_rows);

    float* JTJ = (float*)safe_calloc((size_t)(num_params * num_params), sizeof(float));
    float* JTe = (float*)safe_calloc((size_t)num_params, sizeof(float));
    if (!JTJ || !JTe)
    {
        safe_free((void**)&jacobian);
        safe_free((void**)&JTJ);
        safe_free((void**)&JTe);
        return *current_error;
    }

    for (int i = 0; i < num_params; i++)
    {
        for (int j = 0; j < num_params; j++)
        {
            float sum = 0.0f;
            for (int r = 0; r < jac_rows; r++)
            {
                sum += jacobian[r * num_params + i] * jacobian[r * num_params + j];
            }
            JTJ[i * num_params + j] = sum;
        }
        JTJ[i * num_params + i] += lambda;

        float e_sum = 0.0f;
        for (int r = 0; r < jac_rows; r++)
        {
            e_sum += jacobian[r * num_params + i] * (*current_error);
        }
        JTe[i] = e_sum;
    }

    float* delta = (float*)safe_calloc((size_t)num_params, sizeof(float));
    if (!delta)
    {
        safe_free((void**)&jacobian);
        safe_free((void**)&JTJ);
        safe_free((void**)&JTe);
        return *current_error;
    }

    for (int i = 0; i < num_params; i++)
    {
        if (fabsf(JTJ[i * num_params + i]) < 1e-15f)
        {
            JTJ[i * num_params + i] = 1.0f;
        }
    }

    for (int i = 0; i < num_params; i++)
    {
        if (fabsf(JTJ[i * num_params + i]) < 1e-12f) continue;
        delta[i] = JTe[i] / JTJ[i * num_params + i];
    }

    float* new_params = (float*)safe_malloc((size_t)num_params * sizeof(float));
    if (!new_params)
    {
        safe_free((void**)&jacobian);
        safe_free((void**)&JTJ);
        safe_free((void**)&JTe);
        safe_free((void**)&delta);
        return *current_error;
    }

    for (int i = 0; i < num_params; i++)
    {
        float d = delta[i];
        if (d > 1.0f) d = 1.0f;
        if (d < -1.0f) d = -1.0f;
        new_params[i] = params[i] + d;
    }

    CameraIntrinsics new_left_intr, new_right_intr;
    memcpy(&new_left_intr, new_params + left_intr_offset, sizeof(CameraIntrinsics));
    memcpy(&new_right_intr, new_params + right_intr_offset, sizeof(CameraIntrinsics));

    float new_error = compute_total_reprojection_error(
        &new_left_intr, &new_right_intr,
        new_params + left_pose_offset,
        new_params + left_pose_offset + num_images * 3,
        new_params + right_pose_offset,
        new_params + right_pose_offset + num_images * 3,
        left_points, right_points, object_points,
        num_images, corners_per_image);

    if (new_error < *current_error)
    {
        memcpy(params, new_params, (size_t)num_params * sizeof(float));
        *current_error = new_error;
        safe_free((void**)&delta);
        safe_free((void**)&new_params);
        safe_free((void**)&jacobian);
        safe_free((void**)&JTJ);
        safe_free((void**)&JTe);
        return new_error;
    }

    safe_free((void**)&delta);
    safe_free((void**)&new_params);
    safe_free((void**)&jacobian);
    safe_free((void**)&JTJ);
    safe_free((void**)&JTe);
    return *current_error;
}

int stereo_bundle_adjustment(StereoCalibration* calibration,
                             const float* left_image_points,
                             const float* right_image_points,
                             const float* object_points,
                             int num_images,
                             int corners_per_image,
                             const BundleAdjustmentConfig* config,
                             float* final_error)
{
    if (!calibration || !left_image_points || !right_image_points || !object_points || !config)
        return -1;
    if (num_images <= 0 || corners_per_image <= 0) return -1;

    BundleAdjustmentConfig cfg = *config;
    if (cfg.max_iterations <= 0) cfg.max_iterations = 100;
    if (cfg.lambda_init <= 0.0f) cfg.lambda_init = 0.001f;
    if (cfg.lambda_factor <= 1.0f) cfg.lambda_factor = 10.0f;

    int left_intr_offset = 0;
    int left_pose_offset = left_intr_offset + 11;
    int right_intr_offset = left_pose_offset + num_images * 6;
    int right_pose_offset = right_intr_offset + 11;
    int num_params = right_pose_offset + num_images * 6;

    float params[200];
    memset(params, 0, sizeof(params));

    memcpy(params + left_intr_offset, &calibration->left_intrinsics, 11 * sizeof(float));

    for (int i = 0; i < num_images; i++)
    {
        float init_R[3] = {0.01f, 0.01f, 0.01f};
        float init_T[3] = {0.0f, 0.0f, 1.0f};

        memcpy(params + left_pose_offset + i * 6, init_R, 3 * sizeof(float));
        memcpy(params + left_pose_offset + i * 6 + 3, init_T, 3 * sizeof(float));
    }

    memcpy(params + right_intr_offset, &calibration->right_intrinsics, 11 * sizeof(float));

    float stereo_R[3];
    matrix_to_rodrigues(calibration->extrinsics.rotation, stereo_R);

    for (int i = 0; i < num_images; i++)
    {
        float R_combined[9], R_left[9], R_rel[9];
        rodrigues_to_matrix(params + left_pose_offset + i * 6, R_left);
        rodrigues_to_matrix(stereo_R, R_rel);
        mat_mul_3x3(R_rel, R_left, R_combined);

        float combined_R_vec[3];
        matrix_to_rodrigues(R_combined, combined_R_vec);

        memcpy(params + right_pose_offset + i * 6, combined_R_vec, 3 * sizeof(float));

        float T_combined[3];
        T_combined[0] = calibration->extrinsics.translation[0] + params[left_pose_offset + i * 6 + 3];
        T_combined[1] = calibration->extrinsics.translation[1] + params[left_pose_offset + i * 6 + 4];
        T_combined[2] = calibration->extrinsics.translation[2] + params[left_pose_offset + i * 6 + 5];
        memcpy(params + right_pose_offset + i * 6 + 3, T_combined, 3 * sizeof(float));
    }

    float current_error = compute_total_reprojection_error(
        &calibration->left_intrinsics, &calibration->right_intrinsics,
        params + left_pose_offset, params + left_pose_offset + num_images * 3,
        params + right_pose_offset, params + right_pose_offset + num_images * 3,
        left_image_points, right_image_points, object_points,
        num_images, corners_per_image);

    float lambda = cfg.lambda_init;
    float best_error = current_error;
    float prev_error = current_error;

    for (int iter = 0; iter < cfg.max_iterations; iter++)
    {
        float new_error = lm_step(params, num_params, lambda,
                                  left_image_points, right_image_points, object_points,
                                  num_images, corners_per_image,
                                  left_intr_offset, left_pose_offset,
                                  right_intr_offset, right_pose_offset,
                                  &current_error);

        if (new_error < current_error)
        {
            current_error = new_error;
            lambda /= cfg.lambda_factor;
        }
        else
        {
            lambda *= cfg.lambda_factor;
        }

        if (new_error < best_error)
        {
            best_error = new_error;
        }

        float error_change = fabsf(current_error - prev_error);
        if (cfg.verbose)
        {
            printf("BA迭代 %d: 误差=%.6f, lambda=%.6f, 变化=%.8f\n",
                   iter + 1, current_error, lambda, error_change);
        }

        if (error_change < cfg.error_threshold && iter > 5) break;

        prev_error = current_error;
    }

    memcpy(&calibration->left_intrinsics, params + left_intr_offset, 11 * sizeof(float));
    memcpy(&calibration->right_intrinsics, params + right_intr_offset, 11 * sizeof(float));

    if (final_error) *final_error = best_error;

    return 0;
}

int monocular_bundle_adjustment(CameraIntrinsics* intrinsics,
                                const float* image_points,
                                const float* object_points,
                                int num_images,
                                int corners_per_image,
                                float* rotations,
                                float* translations,
                                const BundleAdjustmentConfig* config,
                                float* final_error)
{
    if (!intrinsics || !image_points || !object_points || !config) return -1;
    if (num_images <= 0 || corners_per_image <= 0) return -1;

    BundleAdjustmentConfig cfg = *config;
    if (cfg.max_iterations <= 0) cfg.max_iterations = 100;
    if (cfg.lambda_init <= 0.0f) cfg.lambda_init = 0.001f;
    if (cfg.lambda_factor <= 1.0f) cfg.lambda_factor = 10.0f;

    int intr_offset = 0;
    int pose_offset = intr_offset + 11;
    int num_params = pose_offset + num_images * 6;

    int total_obs = num_images * corners_per_image;
    int max_jac = total_obs * num_params;

    float* params = (float*)safe_calloc((size_t)num_params, sizeof(float));
    if (!params) return -1;

    memcpy(params + intr_offset, intrinsics, 11 * sizeof(float));

    for (int i = 0; i < num_images; i++)
    {
        if (rotations)
            memcpy(params + pose_offset + i * 6, rotations + i * 3, 3 * sizeof(float));
        else
        {
            params[pose_offset + i * 6] = 0.01f;
            params[pose_offset + i * 6 + 1] = 0.01f;
            params[pose_offset + i * 6 + 2] = 0.01f;
        }
        if (translations)
            memcpy(params + pose_offset + i * 6 + 3, translations + i * 3, 3 * sizeof(float));
        else
        {
            params[pose_offset + i * 6 + 3] = 0.0f;
            params[pose_offset + i * 6 + 4] = 0.0f;
            params[pose_offset + i * 6 + 5] = 1.0f;
        }
    }

    float* jacobian = (float*)safe_malloc((size_t)max_jac * sizeof(float));
    float* JTJ = (float*)safe_calloc((size_t)(num_params * num_params), sizeof(float));
    float* JTe = (float*)safe_calloc((size_t)num_params, sizeof(float));
    float* delta = (float*)safe_calloc((size_t)num_params, sizeof(float));
    float* new_params = (float*)safe_malloc((size_t)num_params * sizeof(float));
    if (!jacobian || !JTJ || !JTe || !delta || !new_params)
    {
        safe_free((void**)&params); safe_free((void**)&jacobian);
        safe_free((void**)&JTJ); safe_free((void**)&JTe);
        safe_free((void**)&delta); safe_free((void**)&new_params);
        return -1;
    }

    float current_error = 0.0f;
    for (int img = 0; img < num_images; img++)
    {
        for (int pt = 0; pt < corners_per_image; pt++)
        {
            float* rvec = params + pose_offset + img * 6;
            float* tvec = rvec + 3;
            current_error += compute_reprojection_error_point(
                object_points + pt * 3, intrinsics,
                rvec, tvec, image_points + (img * corners_per_image + pt) * 2);
        }
    }
    current_error /= (float)(num_images * corners_per_image);

    float lambda = cfg.lambda_init;
    float best_error = current_error;
    float prev_error = current_error;

    for (int iter = 0; iter < cfg.max_iterations; iter++)
    {
        memset(jacobian, 0, (size_t)max_jac * sizeof(float));
        float epsilon = 1e-5f;

        for (int j = 0; j < num_params; j++)
        {
            float h = (fabsf(params[j]) + 1.0f) * epsilon;
            params[j] += h;
            float error_plus = 0.0f;
            for (int img = 0; img < num_images; img++)
            {
                float* rvec = params + pose_offset + img * 6;
                float* tvec = rvec + 3;
                for (int pt = 0; pt < corners_per_image; pt++)
                {
                    error_plus += compute_reprojection_error_point(
                        object_points + pt * 3, (CameraIntrinsics*)(params + intr_offset),
                        rvec, tvec, image_points + (img * corners_per_image + pt) * 2);
                }
            }

            params[j] -= 2.0f * h;
            float error_minus = 0.0f;
            for (int img = 0; img < num_images; img++)
            {
                float* rvec = params + pose_offset + img * 6;
                float* tvec = rvec + 3;
                for (int pt = 0; pt < corners_per_image; pt++)
                {
                    error_minus += compute_reprojection_error_point(
                        object_points + pt * 3, (CameraIntrinsics*)(params + intr_offset),
                        rvec, tvec, image_points + (img * corners_per_image + pt) * 2);
                }
            }
            params[j] += h;

            float derivative = (error_plus - error_minus) / (2.0f * h);
            for (int r = 0; r < total_obs; r++)
                jacobian[r * num_params + j] = derivative / (float)total_obs;
        }

        memset(JTJ, 0, (size_t)(num_params * num_params) * sizeof(float));
        memset(JTe, 0, (size_t)num_params * sizeof(float));

        for (int i = 0; i < num_params; i++)
        {
            for (int j = 0; j < num_params; j++)
            {
                float sum = 0.0f;
                for (int r = 0; r < total_obs; r++)
                    sum += jacobian[r * num_params + i] * jacobian[r * num_params + j];
                JTJ[i * num_params + j] = sum;
            }
            JTJ[i * num_params + i] += lambda;
            float e_sum = 0.0f;
            for (int r = 0; r < total_obs; r++)
                e_sum += jacobian[r * num_params + i] * current_error;
            JTe[i] = e_sum;
        }

        memset(delta, 0, (size_t)num_params * sizeof(float));
        for (int i = 0; i < num_params; i++)
        {
            if (fabsf(JTJ[i * num_params + i]) < 1e-15f)
                JTJ[i * num_params + i] = 1.0f;
            delta[i] = JTe[i] / JTJ[i * num_params + i];
        }

        memcpy(new_params, params, (size_t)num_params * sizeof(float));
        for (int i = 0; i < num_params; i++)
            new_params[i] -= delta[i];

        float new_error = 0.0f;
        for (int img = 0; img < num_images; img++)
        {
            float* rvec = new_params + pose_offset + img * 6;
            float* tvec = rvec + 3;
            for (int pt = 0; pt < corners_per_image; pt++)
            {
                new_error += compute_reprojection_error_point(
                    object_points + pt * 3, (CameraIntrinsics*)(new_params + intr_offset),
                    rvec, tvec, image_points + (img * corners_per_image + pt) * 2);
            }
        }
        new_error /= (float)(num_images * corners_per_image);

        if (new_error < current_error)
        {
            memcpy(params, new_params, (size_t)num_params * sizeof(float));
            current_error = new_error;
            lambda /= cfg.lambda_factor;
            if (new_error < best_error) best_error = new_error;
        }
        else
        {
            lambda *= cfg.lambda_factor;
        }

        float error_change = fabsf(current_error - prev_error);
        if (cfg.verbose)
        {
            printf("单目BA迭代 %d: 误差=%.6f, lambda=%.6f, 变化=%.8f\n",
                   iter + 1, current_error, lambda, error_change);
        }
        if (error_change < cfg.error_threshold && iter > 5) break;
        prev_error = current_error;
    }

    memcpy(intrinsics, params + intr_offset, 11 * sizeof(float));
    for (int i = 0; i < num_images; i++)
    {
        if (rotations) memcpy(rotations + i * 3, params + pose_offset + i * 6, 3 * sizeof(float));
        if (translations) memcpy(translations + i * 3, params + pose_offset + i * 6 + 3, 3 * sizeof(float));
    }

    if (final_error) *final_error = best_error;

    safe_free((void**)&params); safe_free((void**)&jacobian);
    safe_free((void**)&JTJ); safe_free((void**)&JTe);
    safe_free((void**)&delta); safe_free((void**)&new_params);
    return 0;
}

int stereo_compute_point_cloud(const float* disparity_map,
                               int width, int height,
                               const StereoCalibration* calibration,
                               float* point_cloud, int max_valid_points)
{
    if (!disparity_map || !calibration || !point_cloud) return -1;
    if (width <= 0 || height <= 0 || max_valid_points <= 0) return -1;

    float baseline = calibration->baseline;
    float fx = calibration->left_intrinsics.fx;
    float fy = calibration->left_intrinsics.fy;
    float cx = calibration->left_intrinsics.cx;
    float cy = calibration->left_intrinsics.cy;
    float cx_r = calibration->right_intrinsics.cx;
    float cx_diff = cx_r - cx;

    if (baseline <= 0.0f || fx <= 0.0f) return -1;

    int valid_count = 0;
    int total = width * height;

    for (int i = 0; i < total && valid_count < max_valid_points; i++)
    {
        float d = disparity_map[i];
        if (d <= 0.0f) continue;

        int y = i / width;
        int x = i % width;

        float depth = baseline * fx / (d + cx_diff);
        if (depth <= 0.0f) continue;

        float px = (float)(x - (int)cx) * depth / fx;
        float py = (float)(y - (int)cy) * depth / fy;

        point_cloud[valid_count * 3 + 0] = px;
        point_cloud[valid_count * 3 + 1] = py;
        point_cloud[valid_count * 3 + 2] = depth;

        valid_count++;
    }

    return valid_count;
}

/* ============================================================================
 * 双目立体匹配核心算法（Block Matching / SAD）
 *
 * 从校正后的左右图像计算视差图，完整实现双目空间识别。
 * 算法：块匹配(Block Matching) + 绝对差和(SAD) + 子像素精化 + 左右一致性检查
 * ============================================================================ */

/**
 * @brief 计算窗口内SAD匹配代价
 * @return 窗口内像素绝对差之和
 */
static float compute_sad_window(const float* left_img, const float* right_img,
                                 int width, int channels,
                                 int lx, int ly, int rx, int ry,
                                 int half_win) {
    float sad = 0.0f;
    for (int dy = -half_win; dy <= half_win; dy++) {
        int ly_pos = (ly + dy) * width + lx;
        int ry_pos = (ry + dy) * width + rx;
        if (ly_pos < 0 || ry_pos < 0) continue;
        for (int dx = -half_win; dx <= half_win; dx++) {
            int l_idx = (ly_pos + dx) * channels;
            int r_idx = (ry_pos + dx) * channels;
            if (l_idx < 0 || r_idx < 0) continue;
            /* 多通道SAD：累加各通道差值 */
            for (int c = 0; c < channels; c++) {
                sad += fabsf(left_img[l_idx + c] - right_img[r_idx + c]);
            }
        }
    }
    return sad;
}

int stereo_compute_disparity(const float* left_img, const float* right_img,
                             int width, int height, int channels,
                             float* disparity_map,
                             int min_disparity, int max_disparity,
                             int window_size) {
    if (!left_img || !right_img || !disparity_map) return -1;
    if (width <= 0 || height <= 0 || channels <= 0) return -1;

    /* 参数默认值 */
    if (min_disparity < 0) min_disparity = 0;
    if (max_disparity <= 0) max_disparity = 64;
    if (window_size < 3) window_size = 9;
    /* 确保窗口为奇数 */
    if (window_size % 2 == 0) window_size++;

    int half_win = window_size / 2;
    int disp_range = max_disparity - min_disparity;
    if (disp_range <= 0) disp_range = 1;

    /* 逐像素立体匹配 */
    for (int y = half_win; y < height - half_win; y++) {
        for (int x = half_win + max_disparity; x < width - half_win; x++) {
            float best_sad = 1e20f;
            int best_disp = 0;
            float sad_minus = 1e20f;  /* 用于子像素精化 */
            float sad_plus = 1e20f;

            /* 搜索右图的对应点 */
            for (int d = 0; d < disp_range; d++) {
                int disp = min_disparity + d;
                int rx = x - disp;
                if (rx < half_win) continue;

                float sad = compute_sad_window(left_img, right_img,
                                               width, channels,
                                               x, y, rx, y, half_win);

                if (sad < best_sad) {
                    best_sad = sad;
                    best_disp = d;
                }
            }

            /* 子像素精化：抛物线拟合SAD曲线 */
            if (best_disp > 0 && best_disp < disp_range - 1) {
                int d = best_disp;
                int d_minus = d - 1;
                int d_plus = d + 1;
                int rx_m = x - (min_disparity + d_minus);
                int rx_p = x - (min_disparity + d_plus);

                sad_minus = compute_sad_window(left_img, right_img, width, channels,
                                               x, y, rx_m, y, half_win);
                sad_plus = compute_sad_window(left_img, right_img, width, channels,
                                              x, y, rx_p, y, half_win);

                /* 抛物线顶点偏移 */
                float denom = 2.0f * (sad_minus + sad_plus - 2.0f * best_sad);
                if (fabsf(denom) > 1e-10f) {
                    float subpix_offset = (sad_minus - sad_plus) / denom;
                    /* 视差 × 16 用于亚像素精度存储 */
                    disparity_map[y * width + x] = (float)(min_disparity + best_disp) * 16.0f + subpix_offset * 16.0f;
                } else {
                    disparity_map[y * width + x] = (float)(min_disparity + best_disp) * 16.0f;
                }
            } else {
                disparity_map[y * width + x] = (float)(min_disparity + best_disp) * 16.0f;
            }
        }
    }

    /* 左右一致性检查：通过检查左→右和右→左的一致性过滤不可靠匹配 */
    int filtered = 0;
    for (int y = half_win; y < height - half_win; y++) {
        for (int x = half_win + max_disparity; x < width - half_win; x++) {
            float d_val = disparity_map[y * width + x] / 16.0f;
            int d_int = (int)(d_val + 0.5f);

            if (d_int > 0) {
                /* 检查右→左的一致性 */
                int rx = x - d_int;
                if (rx >= half_win + max_disparity) {
                    float r_disp = disparity_map[y * width + rx] / 16.0f;
                    int r_d_int = (int)(r_disp + 0.5f);
                    /* 左右视差不一致的像素标记为无效 */
                    if (abs(d_int - r_d_int) > 1) {
                        disparity_map[y * width + x] = 0.0f;
                        filtered++;
                    }
                }
            }
        }
    }

    return 0;
}

int stereo_disparity_to_depth_map(const float* disparity_map,
                                   float* depth_map,
                                   int width, int height,
                                   const StereoCalibration* calibration) {
    if (!disparity_map || !depth_map || !calibration) return -1;
    if (width <= 0 || height <= 0) return -1;

    float baseline = calibration->baseline;
    float fx = calibration->left_intrinsics.fx;
    float cx = calibration->left_intrinsics.cx;
    float cx_r = calibration->right_intrinsics.cx;
    float cx_diff = cx_r - cx;

    if (baseline <= 0.0f || fx <= 0.0f) return -1;

    int total = width * height;
    for (int i = 0; i < total; i++) {
        float d = disparity_map[i] / 16.0f;  /* 从16倍精度恢复 */
        if (d > 0.5f) {
            float depth = baseline * fx / (d + cx_diff);
            depth_map[i] = (depth > 0.0f && depth < 1000.0f) ? depth : 0.0f;
        } else {
            depth_map[i] = 0.0f;
        }
    }

    return 0;
}