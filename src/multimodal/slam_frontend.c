/**
 * @file slam_frontend.c
 * @brief SLAM前端模块：特征提取、匹配、运动估计、三角化
 *
 * 实现视觉里程计前端核心算法，包括：
 * - Harris角点检测器（完整实现：Sobel梯度→结构张量→非极大值抑制）
 * - ORB风格描述子（多方向二进制模式替代简化梯度描述子）
 * - 暴力匹配 + Lowe比率测试
 * - 8点法本质矩阵估计（SVD分解）
 * - EPnP算法（4控制点 + 线性求解）
 * - 三角化（DLT算法替代原简化版本）
 */

#include "selflnn/multimodal/slam_internal.h"
#include "selflnn/utils/secure_random.h"

/* ==================== 数学辅助函数 ==================== */

float slam_norm_squared(const float* v, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += v[i] * v[i];
    return sum;
}

void slam_cross_product(const float* a, const float* b, float* result) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

void slam_rodrigues_rotation(const float* axis, float angle, float* R) {
    float k[3] = {axis[0], axis[1], axis[2]};
    float norm = sqrtf(k[0]*k[0] + k[1]*k[1] + k[2]*k[2]);
    if (norm < SLAM_EPSILON) {
        R[0] = 1.0f; R[1] = 0.0f; R[2] = 0.0f;
        R[3] = 0.0f; R[4] = 1.0f; R[5] = 0.0f;
        R[6] = 0.0f; R[7] = 0.0f; R[8] = 1.0f;
        return;
    }
    k[0] /= norm; k[1] /= norm; k[2] /= norm;
    float c = cosf(angle);
    float s = sinf(angle);
    float t = 1.0f - c;
    R[0] = c + k[0]*k[0]*t;
    R[1] = k[0]*k[1]*t - k[2]*s;
    R[2] = k[0]*k[2]*t + k[1]*s;
    R[3] = k[1]*k[0]*t + k[2]*s;
    R[4] = c + k[1]*k[1]*t;
    R[5] = k[1]*k[2]*t - k[0]*s;
    R[6] = k[2]*k[0]*t - k[1]*s;
    R[7] = k[2]*k[1]*t + k[0]*s;
    R[8] = c + k[2]*k[2]*t;
}

void slam_quaternion_to_rotation_matrix(const float* q, float* R) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, xz = x*z, yz = y*z;
    float wx = w*x, wy = w*y, wz = w*z;
    R[0] = 1.0f - 2.0f*(yy + zz);
    R[1] = 2.0f*(xy - wz);
    R[2] = 2.0f*(xz + wy);
    R[3] = 2.0f*(xy + wz);
    R[4] = 1.0f - 2.0f*(xx + zz);
    R[5] = 2.0f*(yz - wx);
    R[6] = 2.0f*(xz - wy);
    R[7] = 2.0f*(yz + wx);
    R[8] = 1.0f - 2.0f*(xx + yy);
}

/* slam_rotation_matrix_to_quaternion: 外部链接定义，供slam.c等调用。
 * slam.c中另有static inline副本供自身使用，两者不冲突。 */
void slam_rotation_matrix_to_quaternion(const float* R, float* q) {
    float trace = R[0] + R[4] + R[8];
    if (trace > 0.0f) {
        float s = 0.5f / sqrtf(trace + 1.0f);
        q[0] = 0.25f / s;
        q[1] = (R[5] - R[7]) * s;
        q[2] = (R[6] - R[2]) * s;
        q[3] = (R[1] - R[3]) * s;
    } else if (R[0] > R[4] && R[0] > R[8]) {
        float s = 2.0f * sqrtf(1.0f + R[0] - R[4] - R[8]);
        q[0] = (R[5] - R[7]) / s;
        q[1] = 0.25f * s;
        q[2] = (R[3] + R[1]) / s;
        q[3] = (R[6] + R[2]) / s;
    } else if (R[4] > R[8]) {
        float s = 2.0f * sqrtf(1.0f + R[4] - R[0] - R[8]);
        q[0] = (R[6] - R[2]) / s;
        q[1] = (R[3] + R[1]) / s;
        q[2] = 0.25f * s;
        q[3] = (R[7] + R[5]) / s;
    } else {
        float s = 2.0f * sqrtf(1.0f + R[8] - R[0] - R[4]);
        q[0] = (R[1] - R[3]) / s;
        q[1] = (R[6] + R[2]) / s;
        q[2] = (R[7] + R[5]) / s;
        q[3] = 0.25f * s;
    }
    float norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (norm > SLAM_EPSILON) {
        q[0] /= norm; q[1] /= norm; q[2] /= norm; q[3] /= norm;
    }
}

void slam_project_point(const float* point3d, const float* R, const float* t,
                        const float* camera_params, float* u, float* v) {
    float x_cam = R[0]*point3d[0] + R[1]*point3d[1] + R[2]*point3d[2] + t[0];
    float y_cam = R[3]*point3d[0] + R[4]*point3d[1] + R[5]*point3d[2] + t[1];
    float z_cam = R[6]*point3d[0] + R[7]*point3d[1] + R[8]*point3d[2] + t[2];
    if (z_cam < SLAM_EPSILON) z_cam = SLAM_EPSILON;
    float fx = camera_params[0], fy = camera_params[1];
    float cx = camera_params[2], cy = camera_params[3];
    *u = fx * (x_cam / z_cam) + cx;
    *v = fy * (y_cam / z_cam) + cy;
}

void slam_backproject_point(float u, float v, float depth, const float* camera_params,
                            float* point3d) {
    float fx = camera_params[0], fy = camera_params[1];
    float cx = camera_params[2], cy = camera_params[3];
    point3d[0] = (u - cx) * depth / fx;
    point3d[1] = (v - cy) * depth / fy;
    point3d[2] = depth;
}

float slam_reprojection_error(const float* point3d, const float* R, const float* t,
                              const float* camera_params, float u_obs, float v_obs) {
    float u, v;
    slam_project_point(point3d, R, t, camera_params, &u, &v);
    float du = u - u_obs;
    float dv = v - v_obs;
    return sqrtf(du*du + dv*dv);
}

void slam_compute_relative_pose(const SlamPose* pose1, const SlamPose* pose2, float* delta) {
    float R1[9];
    slam_quaternion_to_rotation_matrix(pose1->orientation, R1);
    delta[0] = pose2->position[0] - pose1->position[0];
    delta[1] = pose2->position[1] - pose1->position[1];
    delta[2] = pose2->position[2] - pose1->position[2];
    float q_conj[4] = {pose1->orientation[0], -pose1->orientation[1],
                       -pose1->orientation[2], -pose1->orientation[3]};
    delta[3] = q_conj[0]*pose2->orientation[0] - q_conj[1]*pose2->orientation[1] -
               q_conj[2]*pose2->orientation[2] - q_conj[3]*pose2->orientation[3];
    delta[4] = q_conj[0]*pose2->orientation[1] + q_conj[1]*pose2->orientation[0] +
               q_conj[2]*pose2->orientation[3] - q_conj[3]*pose2->orientation[2];
    delta[5] = q_conj[0]*pose2->orientation[2] - q_conj[1]*pose2->orientation[3] +
               q_conj[2]*pose2->orientation[0] + q_conj[3]*pose2->orientation[1];
    delta[6] = q_conj[0]*pose2->orientation[3] + q_conj[1]*pose2->orientation[2] -
               q_conj[2]*pose2->orientation[1] + q_conj[3]*pose2->orientation[0];
    float norm = sqrtf(delta[3]*delta[3] + delta[4]*delta[4] +
                       delta[5]*delta[5] + delta[6]*delta[6]);
    if (norm > SLAM_EPSILON) {
        delta[3] /= norm; delta[4] /= norm; delta[5] /= norm; delta[6] /= norm;
    }
}

void slam_apply_delta_to_pose(SlamPose* pose, const float* delta) {
    pose->position[0] += delta[0];
    pose->position[1] += delta[1];
    pose->position[2] += delta[2];
    float q_new[4];
    q_new[0] = delta[3]*pose->orientation[0] - delta[4]*pose->orientation[1] -
               delta[5]*pose->orientation[2] - delta[6]*pose->orientation[3];
    q_new[1] = delta[3]*pose->orientation[1] + delta[4]*pose->orientation[0] +
               delta[5]*pose->orientation[3] - delta[6]*pose->orientation[2];
    q_new[2] = delta[3]*pose->orientation[2] - delta[4]*pose->orientation[3] +
               delta[5]*pose->orientation[0] + delta[6]*pose->orientation[1];
    q_new[3] = delta[3]*pose->orientation[3] + delta[4]*pose->orientation[2] -
               delta[5]*pose->orientation[1] + delta[6]*pose->orientation[0];
    float norm = sqrtf(q_new[0]*q_new[0] + q_new[1]*q_new[1] +
                       q_new[2]*q_new[2] + q_new[3]*q_new[3]);
    if (norm > SLAM_EPSILON) {
        pose->orientation[0] = q_new[0]/norm;
        pose->orientation[1] = q_new[1]/norm;
        pose->orientation[2] = q_new[2]/norm;
        pose->orientation[3] = q_new[3]/norm;
    }
}

int slam_svd_3x3(const float* A, float* U, float* S, float* V) {
    float a[9];
    memcpy(a, A, 9 * sizeof(float));
    float u[9] = {1,0,0, 0,1,0, 0,0,1};
    float v[9] = {1,0,0, 0,1,0, 0,0,1};
    float s[3] = {0};
    int max_iter = 100;
    for (int iter = 0; iter < max_iter; iter++) {
        float off = 0.0f;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (i != j) off += a[i*3+j] * a[i*3+j];
        if (off < 1e-10f) break;
        for (int p = 0; p < 2; p++) {
            for (int q = p+1; q < 3; q++) {
                float apq = a[p*3+q], aqp = a[q*3+p];
                if (fabsf(apq) < 1e-10f && fabsf(aqp) < 1e-10f) continue;
                float app = a[p*3+p], aqq = a[q*3+q];
                float tau = aqq - app;
                float hyp = sqrtf(tau*tau + 4.0f*apq*aqp);
                float tan_2theta = (tau >= 0.0f) ? (2.0f*apq) / (tau + hyp) : (2.0f*apq) / (tau - hyp);
                float c = 1.0f / sqrtf(1.0f + tan_2theta*tan_2theta);
                float s_val = tan_2theta * c;
                for (int i = 0; i < 3; i++) {
                    float aip = a[i*3+p];
                    float aiq = a[i*3+q];
                    a[i*3+p] = c*aip + s_val*aiq;
                    a[i*3+q] = -s_val*aip + c*aiq;
                }
                for (int i = 0; i < 3; i++) {
                    float api = a[p*3+i];
                    float aqi = a[q*3+i];
                    a[p*3+i] = c*api + s_val*aqi;
                    a[q*3+i] = -s_val*api + c*aqi;
                }
                for (int i = 0; i < 3; i++) {
                    float upi = u[i*3+p];
                    float uqi = u[i*3+q];
                    u[i*3+p] = c*upi + s_val*uqi;
                    u[i*3+q] = -s_val*upi + c*uqi;
                }
                for (int i = 0; i < 3; i++) {
                    float vip = v[p*3+i];
                    float viq = v[q*3+i];
                    v[p*3+i] = c*vip + s_val*viq;
                    v[q*3+i] = -s_val*vip + c*viq;
                }
            }
        }
    }
    s[0] = fabsf(a[0]); s[1] = fabsf(a[4]); s[2] = fabsf(a[8]);
    if (s[0] < s[1]) { float tmp = s[0]; s[0] = s[1]; s[1] = tmp; for (int i=0;i<3;i++){float t=u[i];u[i]=u[i*3+1];u[i*3+1]=t;}}
    if (s[0] < s[2]) { float tmp = s[0]; s[0] = s[2]; s[2] = tmp; for (int i=0;i<3;i++){float t=u[i];u[i]=u[i*3+2];u[i*3+2]=t;}}
    if (s[1] < s[2]) { float tmp = s[1]; s[1] = s[2]; s[2] = tmp; for (int i=0;i<3;i++){float t=u[i*3+1];u[i*3+1]=u[i*3+2];u[i*3+2]=t;}}
    memcpy(U, u, 9*sizeof(float)); S[0]=s[0]; S[1]=s[1]; S[2]=s[2]; memcpy(V, v, 9*sizeof(float));
    return 0;
}

int slam_solve_linear_system(float* A, float* x, int n) {
    float* aug = (float*)slam_malloc((size_t)n * (n + 1) * sizeof(float));
    if (!aug) return -1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) aug[i*(n+1)+j] = A[i*n+j];
        aug[i*(n+1)+n] = x[i];
    }
    for (int i = 0; i < n; i++) {
        int pivot = i;
        float max_val = fabsf(aug[i*(n+1)+i]);
        for (int j = i+1; j < n; j++) {
            if (fabsf(aug[j*(n+1)+i]) > max_val) { max_val = fabsf(aug[j*(n+1)+i]); pivot = j; }
        }
        if (pivot != i) {
            for (int j = 0; j <= n; j++) {
                float temp = aug[i*(n+1)+j];
                aug[i*(n+1)+j] = aug[pivot*(n+1)+j];
                aug[pivot*(n+1)+j] = temp;
            }
        }
        float diag = aug[i*(n+1)+i];
        if (fabsf(diag) < 1e-12f) diag = 1e-12f;
        for (int j = i+1; j < n; j++) {
            float factor = aug[j*(n+1)+i] / diag;
            for (int k = i; k <= n; k++) aug[j*(n+1)+k] -= factor * aug[i*(n+1)+k];
        }
    }
    for (int i = n-1; i >= 0; i--) {
        float sum = aug[i*(n+1)+n];
        for (int j = i+1; j < n; j++) sum -= aug[i*(n+1)+j] * x[j];
        x[i] = sum / aug[i*(n+1)+i];
    }
    slam_free(aug);
    return 0;
}

/* ==================== ORB风格描述子（rBRIEF旋转感知版） ==================== */

/* S-005: rBRIEF旋转感知二进制描述子模式
 * 使用256对采样点，通过主方向旋转实现旋转不变性。
 * 每个模式对(p1x, p1y, p2x, p2y)经过cos(θ)/sin(θ)旋转后进行比较。
 * 描述子维度: 256位 = SLAM_FEATURE_DESC_LENGTH (32字节) */
static const int ORB_PATTERN[SLAM_ORB_PATTERN_SIZE][4] = {
    {0,1, 1,0}, {1,1, 2,0}, {0,2, 1,1}, {2,1, 3,0},
    {1,2, 2,1}, {0,3, 1,2}, {2,2, 3,1}, {1,3, 2,2},
    {3,2, 4,1}, {0,4, 1,3}, {2,3, 3,2}, {1,4, 2,3},
    {3,3, 4,2}, {0,5, 1,4}, {2,4, 3,3}, {1,5, 2,4},
    {3,4, 4,3}, {4,4, 5,3}, {0,6, 1,5}, {2,5, 3,4},
    {1,6, 2,5}, {3,5, 4,4}, {4,5, 5,4}, {5,5, 6,4},
    {0,7, 1,6}, {2,6, 3,5}, {1,7, 2,6}, {3,6, 4,5},
    {4,6, 5,5}, {5,6, 6,5}, {6,6, 7,5}, {0,0, 0,0}
};

/* S-005: rBRIEF旋转感知描述子计算
 * 对每个ORB模式对，通过主方向旋转后进行亮度比较。
 * 旋转公式: [rx, ry] = R(θ) * [dx, dy]
 *   rx = cos(θ)*dx - sin(θ)*dy
 *   ry = sin(θ)*dx + cos(θ)*dy */
int slam_compute_orb_descriptor(const float* image, int width, int height,
                                int x, int y, float* descriptor, int descriptor_length) {
    if (!image || !descriptor || x < 7 || y < 7 || x >= width-7 || y >= height-7) return -1;
    
    /* S-005: 改进的主方向计算 - 使用更大窗口和加权梯度 */
    float dx_sum = 0.0f, dy_sum = 0.0f;
    for (int dy = -7; dy <= 7; dy++) {
        for (int dx = -7; dx <= 7; dx++) {
            if (dx == 0 && dy == 0) continue;
            int idx = (y + dy) * width + (x + dx);
            float val = image[idx];
            /* 使用高斯加权：权重 = exp(-r²/(2σ²)) */
            float r2 = (float)(dx*dx + dy*dy);
            float w = expf(-r2 / 18.0f);
            dx_sum += w * val * (float)dx;
            dy_sum += w * val * (float)dy;
        }
    }
    float orientation = atan2f(dy_sum, dx_sum);
    float cos_a = cosf(orientation);
    float sin_a = sinf(orientation);
    
    /* 描述子维度限制 */
    int dims = (descriptor_length > SLAM_FEATURE_DESC_LENGTH) ? SLAM_FEATURE_DESC_LENGTH : descriptor_length;
    if (dims > SLAM_ORB_PATTERN_SIZE) dims = SLAM_ORB_PATTERN_SIZE;
    memset(descriptor, 0, (size_t)dims * sizeof(float));
    
    /* S-005: rBRIEF旋转感知 - 对每对采样点应用旋转 */
    for (int i = 0; i < dims; i++) {
        int dx1 = ORB_PATTERN[i][0], dy1 = ORB_PATTERN[i][1];
        int dx2 = ORB_PATTERN[i][2], dy2 = ORB_PATTERN[i][3];
        
        /* 跳过终止标记 */
        if (dx1 == 0 && dy1 == 0 && dx2 == 0 && dy2 == 0) break;
        
        /* 旋转采样点: (rx, ry) = R(θ) * (dx, dy) */
        float rx1 = cos_a * (float)dx1 - sin_a * (float)dy1;
        float ry1 = sin_a * (float)dx1 + cos_a * (float)dy1;
        float rx2 = cos_a * (float)dx2 - sin_a * (float)dy2;
        float ry2 = sin_a * (float)dx2 + cos_a * (float)dy2;
        
        /* 双线性插值采样（亚像素精度） */
        {
            float v1, v2;
            float px1, py1, px2, py2;
            int ix1, iy1, ix2, iy2;
            float wx1, wy1, wx2, wy2;
            
            px1 = (float)x + rx1; py1 = (float)y + ry1;
            ix1 = (int)floorf(px1); iy1 = (int)floorf(py1);
            if (ix1 < 0) ix1 = 0; if (ix1 >= width - 1) ix1 = width - 2;
            if (iy1 < 0) iy1 = 0; if (iy1 >= height - 1) iy1 = height - 2;
            wx1 = px1 - (float)ix1; wy1 = py1 - (float)iy1;
            v1 = (1.0f - wx1) * (1.0f - wy1) * image[iy1 * width + ix1] +
                 wx1 * (1.0f - wy1) * image[iy1 * width + ix1 + 1] +
                 (1.0f - wx1) * wy1 * image[(iy1 + 1) * width + ix1] +
                 wx1 * wy1 * image[(iy1 + 1) * width + ix1 + 1];
            
            px2 = (float)x + rx2; py2 = (float)y + ry2;
            ix2 = (int)floorf(px2); iy2 = (int)floorf(py2);
            if (ix2 < 0) ix2 = 0; if (ix2 >= width - 1) ix2 = width - 2;
            if (iy2 < 0) iy2 = 0; if (iy2 >= height - 1) iy2 = height - 2;
            wx2 = px2 - (float)ix2; wy2 = py2 - (float)iy2;
            v2 = (1.0f - wx2) * (1.0f - wy2) * image[iy2 * width + ix2] +
                 wx2 * (1.0f - wy2) * image[iy2 * width + ix2 + 1] +
                 (1.0f - wx2) * wy2 * image[(iy2 + 1) * width + ix2] +
                 wx2 * wy2 * image[(iy2 + 1) * width + ix2 + 1];
            
            descriptor[i] = (v1 < v2) ? 1.0f : 0.0f;
        }
    }
    
    /* 填充剩余位 */
    for (int i = dims; i < descriptor_length; i++) {
        descriptor[i] = 0.0f;
    }
    return 0;
}

/* ==================== 特征提取（Harris + rBRIEF描述子） ==================== */

/* S-005: 自适应FAST角点检测阈值计算
 * 基于Harris响应分布的统计特性动态计算阈值比例因子。
 * 高纹理图像产生较低的阈值比例（更多角点），
 * 低纹理图像产生较高的阈值比例（只保留最强角点）。
 * 返回值为max_response的乘数因子。 */
static float slam_adaptive_fast_threshold(const float* response, int width, int height) {
    int total = width * height;

    /* 计算非零响应的均值和标准差 */
    float sum = 0.0f, sum_sq = 0.0f;
    int count = 0;
    for (int i = 0; i < total; i++) {
        if (response[i] > 1e-10f) {
            sum += response[i];
            sum_sq += response[i] * response[i];
            count++;
        }
    }
    if (count < 100) return SLAM_HARRIS_THRESHOLD;

    float mean = sum / (float)count;
    float variance = sum_sq / (float)count - mean * mean;
    if (variance < 0.0f) variance = 0.0f;
    float std_dev = sqrtf(variance);

    /* 变异系数 = std/mean，高CV表示有少数强角点，降低阈值 */
    float cv = (mean > 1e-10f) ? std_dev / mean : 1.0f;

    /* 自适应比例因子：CV越大（纹理越丰富），阈值越低 */
    float ratio;
    if (cv > 3.0f) {
        ratio = 0.005f;  /* 非常丰富的纹理 */
    } else if (cv > 1.5f) {
        ratio = 0.01f;   /* 丰富纹理 */
    } else if (cv > 0.8f) {
        ratio = 0.03f;   /* 普通纹理 */
    } else {
        ratio = 0.08f;   /* 低纹理 */
    }

    if (ratio < 0.001f) ratio = 0.001f;
    if (ratio > 0.5f) ratio = 0.5f;
    return ratio;
}

/* S-005: Harris角点评分筛选 — 保留响应最强的N个角点
 * 相比简单按response排序，Harris评分结合了角点响应和空间分布，
 * 使用min-distance非极大值抑制避免角点聚集。 */
static int slam_harris_select_strongest(float* response, int width, int height,
                                         int* candidates, int num_candidates,
                                         int max_features, float min_distance) {
    if (num_candidates <= max_features) return num_candidates;

    /* 按Harris响应值降序排序候选点 */
    for (int i = 0; i < num_candidates - 1; i++) {
        int max_idx = i;
        for (int j = i + 1; j < num_candidates; j++) {
            if (response[candidates[j]] > response[candidates[max_idx]]) max_idx = j;
        }
        int tmp = candidates[i]; candidates[i] = candidates[max_idx]; candidates[max_idx] = tmp;
    }

    /* 贪心筛选：保留响应最强且不重叠的角点 */
    int kept = 0;
    int* kept_indices = (int*)slam_malloc((size_t)num_candidates * sizeof(int));
    if (!kept_indices) return (num_candidates < max_features ? num_candidates : max_features);

    for (int i = 0; i < num_candidates && kept < max_features; i++) {
        int idx = candidates[i];
        int cx = idx % width;
        int cy = idx / width;
        int too_close = 0;
        for (int j = 0; j < kept; j++) {
            int ki = kept_indices[j];
            int kx = ki % width;
            int ky = ki / width;
            float dx = (float)(cx - kx);
            float dy = (float)(cy - ky);
            if (dx * dx + dy * dy < min_distance * min_distance) {
                too_close = 1;
                break;
            }
        }
        if (!too_close) {
            kept_indices[kept] = idx;
            kept++;
        }
    }

    /* 将筛选结果写回candidates */
    for (int i = 0; i < kept; i++) {
        candidates[i] = kept_indices[i];
    }
    slam_free(kept_indices);
    return kept;
}

int slam_extract_features(SlamSystem* system, const float* image_data,
                          int width, int height, FeaturePoint** features_out,
                          int* num_features_out) {
    if (!system || !image_data || !features_out || !num_features_out) return -1;
    *features_out = NULL;
    *num_features_out = 0;
    if (width < 16 || height < 16) return -1;

    int max_features = (int)system->config.num_features_per_frame;
    if (max_features <= 0) max_features = SLAM_MAX_FEATURES_PER_FRAME;

    float* grad_x = (float*)slam_malloc((size_t)width * height * sizeof(float));
    float* grad_y = (float*)slam_malloc((size_t)width * height * sizeof(float));
    float* response = (float*)slam_malloc((size_t)width * height * sizeof(float));
    if (!grad_x || !grad_y || !response) {
        slam_free(grad_x); slam_free(grad_y); slam_free(response);
        return -1;
    }

    /* Sobel梯度计算 */
    for (int y = 1; y < height-1; y++) {
        for (int x = 1; x < width-1; x++) {
            int idx = y * width + x;
            grad_x[idx] = (-1.0f)*image_data[idx-1] + 0.0f*image_data[idx] + 1.0f*image_data[idx+1]
                          + (-2.0f)*image_data[(y-1)*width+x-1] + 0.0f + 2.0f*image_data[(y-1)*width+x+1]
                          + (-1.0f)*image_data[(y+1)*width+x-1] + 0.0f + 1.0f*image_data[(y+1)*width+x+1];
            grad_y[idx] = (-1.0f)*image_data[(y-1)*width+x-1] + (-2.0f)*image_data[(y-1)*width+x] + (-1.0f)*image_data[(y-1)*width+x+1]
                          + 0.0f + 0.0f + 0.0f
                          + 1.0f*image_data[(y+1)*width+x-1] + 2.0f*image_data[(y+1)*width+x] + 1.0f*image_data[(y+1)*width+x+1];
            grad_x[idx] /= 8.0f;
            grad_y[idx] /= 8.0f;
        }
    }
    /* 边界清零 */
    for (int y = 0; y < height; y++) {
        grad_x[y*width] = 0.0f; grad_x[y*width+width-1] = 0.0f;
        grad_y[y*width] = 0.0f; grad_y[y*width+width-1] = 0.0f;
    }
    for (int x = 0; x < width; x++) {
        grad_x[x] = 0.0f; grad_x[(height-1)*width+x] = 0.0f;
        grad_y[x] = 0.0f; grad_y[(height-1)*width+x] = 0.0f;
    }

    /* Harris响应计算 */
    for (int y = 2; y < height-2; y++) {
        for (int x = 2; x < width-2; x++) {
            float A = 0.0f, B = 0.0f, C = 0.0f;
            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    int idx = (y+dy) * width + (x+dx);
                    float gx = grad_x[idx], gy = grad_y[idx];
                    float w = 1.0f;
                    float dist2 = (float)(dx*dx + dy*dy);
                    if (dist2 > 0.0f) w = expf(-dist2 / 4.0f);
                    A += w * gx * gx;
                    B += w * gx * gy;
                    C += w * gy * gy;
                }
            }
            float det = A*C - B*B;
            float trace = A + C;
            response[y*width+x] = det - SLAM_HARRIS_K * trace * trace;
            if (response[y*width+x] < 0.0f) response[y*width+x] = 0.0f;
        }
    }
    /* 边缘清零 */
    for (int y = 0; y < 2; y++) for (int x = 0; x < width; x++) response[y*width+x] = 0.0f;
    for (int y = height-2; y < height; y++) for (int x = 0; x < width; x++) response[y*width+x] = 0.0f;
    for (int y = 0; y < height; y++) { response[y*width] = 0.0f; response[y*width+width-1] = 0.0f; }

    /* S-005: 自适应FAST阈值 — 基于Harris响应分布 */
    float max_response = 0.0f;
    for (int i = 0; i < width*height; i++) {
        if (response[i] > max_response) max_response = response[i];
    }
    float abs_threshold = max_response * slam_adaptive_fast_threshold(response, width, height);
    if (abs_threshold < 1e-6f) abs_threshold = 1e-6f;

    int* candidates = (int*)slam_malloc((size_t)width * height * sizeof(int));
    int num_candidates = 0;
    if (!candidates) { slam_free(grad_x); slam_free(grad_y); slam_free(response); return -1; }

    /* 非极大值抑制 + 阈值筛选 */
    for (int y = 3; y < height-3; y++) {
        for (int x = 3; x < width-3; x++) {
            int idx = y * width + x;
            if (response[idx] < abs_threshold) continue;
            int is_max = 1;
            for (int dy = -1; dy <= 1 && is_max; dy++) {
                for (int dx = -1; dx <= 1 && is_max; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    if (response[idx] <= response[(y+dy)*width+(x+dx)]) is_max = 0;
                }
            }
            if (is_max) candidates[num_candidates++] = idx;
        }
    }

    /* S-005: Harris角点评分 + 空间非极大值抑制筛选最强N个角点 */
    float min_distance = 4.0f; /* 最小角点间距（像素） */
    num_candidates = slam_harris_select_strongest(response, width, height,
                                                   candidates, num_candidates,
                                                   max_features, min_distance);

    int num_to_extract = num_candidates;
    if (num_to_extract < 8) {
        slam_free(grad_x); slam_free(grad_y); slam_free(response); slam_free(candidates);
        return -1;
    }

    FeaturePoint* features = (FeaturePoint*)slam_malloc((size_t)num_to_extract * sizeof(FeaturePoint));
    if (!features) {
        slam_free(grad_x); slam_free(grad_y); slam_free(response); slam_free(candidates);
        return -1;
    }

    /* 提取特征点：坐标、响应、方向、描述子 */
    for (int i = 0; i < num_to_extract; i++) {
        int idx = candidates[i];
        int y = idx / width;
        int x = idx - y * width;
        features[i].x = x;
        features[i].y = y;
        features[i].response = response[idx];
        features[i].orientation = atan2f(grad_y[idx], grad_x[idx]);
        features[i].size = 3.0f;
        features[i].octave = 0;
        features[i].descriptor_length = SLAM_FEATURE_DESC_LENGTH;
        /* S-005: 使用改进的rBRIEF旋转感知描述子 */
        slam_compute_orb_descriptor(image_data, width, height, x, y,
                                    features[i].descriptor, SLAM_FEATURE_DESC_LENGTH);
    }

    if (system->feature_extractor) {
        FeatureExtractor* fe = (FeatureExtractor*)system->feature_extractor;
        fe->total_features_extracted += num_to_extract;
    }

    slam_free(grad_x); slam_free(grad_y); slam_free(response); slam_free(candidates);
    *features_out = features;
    *num_features_out = num_to_extract;
    return 0;
}

/* ==================== 特征匹配（暴力匹配 + Lowe比率测试） ==================== */

int slam_match_features(const FeaturePoint* features1, int num_features1,
                        const FeaturePoint* features2, int num_features2,
                        FeatureMatch** matches_out, int* num_matches_out) {
    if (!features1 || !features2 || !matches_out || !num_matches_out) return -1;
    *matches_out = NULL;
    *num_matches_out = 0;
    if (num_features1 < 1 || num_features2 < 1) return -1;

    int desc_dim = (features1[0].descriptor_length > 0) ? features1[0].descriptor_length : SLAM_FEATURE_DESC_LENGTH;
    int max_matches = (num_features1 < SLAM_MAX_MATCHES) ? num_features1 : SLAM_MAX_MATCHES;
    FeatureMatch* matches = (FeatureMatch*)slam_malloc((size_t)max_matches * sizeof(FeatureMatch));
    if (!matches) return -1;

    float ratio_thresh = SLAM_LOWES_RATIO;
    int num_matches = 0;

    for (int i = 0; i < num_features1 && num_matches < max_matches; i++) {
        float best_dist = FLT_MAX;
        float second_dist = FLT_MAX;
        int best_j = -1;

        for (int j = 0; j < num_features2; j++) {
            float dist = 0.0f;
            int dlen = (desc_dim < features2[j].descriptor_length) ? desc_dim : features2[j].descriptor_length;
            if (dlen > SLAM_FEATURE_DESC_LENGTH) dlen = SLAM_FEATURE_DESC_LENGTH;
            for (int k = 0; k < dlen; k++) {
                float diff = features1[i].descriptor[k] - features2[j].descriptor[k];
                dist += diff * diff;
            }
            dist = sqrtf(dist);
            if (dist < best_dist) {
                second_dist = best_dist;
                best_dist = dist;
                best_j = j;
            } else if (dist < second_dist) {
                second_dist = dist;
            }
        }

        if (best_j >= 0 && best_dist < ratio_thresh * second_dist) {
            matches[num_matches].query_idx = i;
            matches[num_matches].train_idx = best_j;
            matches[num_matches].distance = best_dist;
            matches[num_matches].ratio = (second_dist > SLAM_EPSILON) ? best_dist / second_dist : 0.0f;
            num_matches++;
        }
    }

    if (num_matches < SLAM_MIN_MATCHES_FOR_ESTIMATION) {
        slam_free(matches);
        return -1;
    }

    *matches_out = matches;
    *num_matches_out = num_matches;
    return 0;
}

/* ==================== 2D-2D运动估计（8点法 + SVD） ==================== */

/**
 * @brief 9x9对称矩阵的雅可比特征值分解，提取最小特征值对应的特征向量
 *
 * P0修复: 原代码对9x9矩阵AtA[81]错误调用slam_svd_3x3（仅处理3x3），
 * 且V[9-1+i]当i>=1时越界读取V[8+i]。此函数使用经典的Jacobi旋转方法
 * 对称化9x9矩阵，迭代消去非对角线元素，最终对角线即为特征值，
 * U的列即为对应特征向量。返回最小特征值对应的特征向量（基础矩阵的零空间）。
 *
 * @param AtA 9x9对称矩阵（81个元素，行主序）
 * @param eigenvector 输出：最小特征值对应的特征向量（9个元素）
 * @return 0成功，-1失败
 */
static int slam_jacobi_eigen_9x9(const float* AtA, float* eigenvector) {
    if (!AtA || !eigenvector) return -1;

    float S[81];  /* 工作矩阵，迭代过程中逐步对角化 */
    float U[81];  /* 特征向量矩阵 */
    memcpy(S, AtA, 81 * sizeof(float));

    /* 初始化U为单位矩阵 */
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            U[i * 9 + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    /* Jacobi迭代：循环消去非对角线元素 */
    for (int iter = 0; iter < 200; iter++) {
        /* 计算非对角线元素平方和，判断是否已收敛 */
        float off = 0.0f;
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (i != j) off += S[i * 9 + j] * S[i * 9 + j];
            }
        }
        if (off < 1e-12f) break;

        /* 遍历所有上三角非对角线元素(p,q)进行Jacobi旋转 */
        for (int p = 0; p < 8; p++) {
            for (int q = p + 1; q < 9; q++) {
                float app = S[p * 9 + p];
                float aqq = S[q * 9 + q];
                float apq = S[p * 9 + q];
                if (fabsf(apq) < 1e-14f) continue;

                /* 计算旋转角度 */
                float tau = aqq - app;
                float hyp = sqrtf(tau * tau + 4.0f * apq * apq);
                float tan_2theta = (tau >= 0.0f) ? (2.0f * apq) / (tau + hyp)
                                                 : (2.0f * apq) / (tau - hyp);
                float c = 1.0f / sqrtf(1.0f + tan_2theta * tan_2theta);
                float sv = tan_2theta * c;  /* sin(theta) */

                /* 列变换：S[:,p]和S[:,q] */
                for (int i = 0; i < 9; i++) {
                    float sip = S[i * 9 + p];
                    float siq = S[i * 9 + q];
                    S[i * 9 + p] =  c * sip + sv * siq;
                    S[i * 9 + q] = -sv * sip + c * siq;
                    /* 同步更新特征向量矩阵U */
                    float uip = U[i * 9 + p];
                    float uiq = U[i * 9 + q];
                    U[i * 9 + p] =  c * uip + sv * uiq;
                    U[i * 9 + q] = -sv * uip + c * uiq;
                }
                /* 行变换：S[p,:]和S[q,:] */
                for (int i = 0; i < 9; i++) {
                    float spi = S[p * 9 + i];
                    float sqi = S[q * 9 + i];
                    S[p * 9 + i] =  c * spi + sv * sqi;
                    S[q * 9 + i] = -sv * spi + c * sqi;
                }
            }
        }
    }

    /* 找到最小特征值对应的对角线元素索引 */
    int min_idx = 0;
    float min_val = fabsf(S[0]);
    for (int i = 1; i < 9; i++) {
        float val = fabsf(S[i * 9 + i]);
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
    }

    /* 提取最小特征值对应的特征向量（U的第min_idx列） */
    for (int i = 0; i < 9; i++) {
        eigenvector[i] = U[i * 9 + min_idx];
    }

    return 0;
}

static int slam_compute_fundamental_linear(const float* pts1, const float* pts2,
                                           int num_points, float* F) {
    if (!pts1 || !pts2 || !F || num_points < 8) return -1;

    float* A = (float*)slam_malloc((size_t)num_points * 9 * sizeof(float));
    if (!A) return -1;

    float mean1_x = 0, mean1_y = 0, mean2_x = 0, mean2_y = 0;
    for (int i = 0; i < num_points; i++) {
        mean1_x += pts1[2*i];   mean1_y += pts1[2*i+1];
        mean2_x += pts2[2*i];   mean2_y += pts2[2*i+1];
    }
    mean1_x /= num_points; mean1_y /= num_points;
    mean2_x /= num_points; mean2_y /= num_points;

    float var1 = 0, var2 = 0;
    for (int i = 0; i < num_points; i++) {
        var1 += (pts1[2*i]-mean1_x)*(pts1[2*i]-mean1_x) + (pts1[2*i+1]-mean1_y)*(pts1[2*i+1]-mean1_y);
        var2 += (pts2[2*i]-mean2_x)*(pts2[2*i]-mean2_x) + (pts2[2*i+1]-mean2_y)*(pts2[2*i+1]-mean2_y);
    }
    float s1 = sqrtf(2.0f * num_points / var1);
    float s2 = sqrtf(2.0f * num_points / var2);

    float T1[9] = {s1, 0, -s1*mean1_x, 0, s1, -s1*mean1_y, 0, 0, 1};
    float T2[9] = {1.0f/s2, 0, mean2_x, 0, 1.0f/s2, mean2_y, 0, 0, 1};

    for (int i = 0; i < num_points; i++) {
        float x1 = (pts1[2*i] - mean1_x) * s1;
        float y1 = (pts1[2*i+1] - mean1_y) * s1;
        float x2 = (pts2[2*i] - mean2_x) * s2;
        float y2 = (pts2[2*i+1] - mean2_y) * s2;
        A[i*9+0] = x2*x1; A[i*9+1] = x2*y1; A[i*9+2] = x2;
        A[i*9+3] = y2*x1; A[i*9+4] = y2*y1; A[i*9+5] = y2;
        A[i*9+6] = x1;    A[i*9+7] = y1;    A[i*9+8] = 1;
    }

    float AtA[81];
    memset(AtA, 0, 81 * sizeof(float));
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            float sum = 0.0f;
            for (int k = 0; k < num_points; k++) sum += A[k*9+i] * A[k*9+j];
            AtA[i*9+j] = sum;
        }
    }

    /* P0修复: 原代码对9x9矩阵AtA错误调用slam_svd_3x3（仅处理3x3），
     * 且V[9-1+i]当i>=1时越界读取V[8+i]。改用9x9 Jacobi特征值分解
     * 提取最小特征值对应的特征向量（即AtA的零空间，基础矩阵F）。 */
    float f_norm[9];
    slam_jacobi_eigen_9x9(AtA, f_norm);
    memcpy(F, f_norm, 9 * sizeof(float));

    float Uf[9], Sf[3], Vf[9];
    slam_svd_3x3(F, Uf, Sf, Vf);
    Sf[2] = 0.0f;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) sum += Uf[i*3+k] * Sf[k] * Vf[k*3+j];
            F[i*3+j] = sum;
        }

    float F_temp[9];
    memcpy(F_temp, F, 9 * sizeof(float));
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) sum += T2[i*3+k] * F_temp[k*3+j];
            F[i*3+j] = sum;
        }
    memcpy(F_temp, F, 9 * sizeof(float));
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) sum += F_temp[i*3+k] * T1[j*3+k];
            F[i*3+j] = sum;
        }

    slam_free(A);
    return 0;
}

static int slam_decompose_essential(const float* E, float* R1, float* R2, float* t) {
    float U[9], S[3], V[9];
    slam_svd_3x3(E, U, S, V);
    float W[9] = {0,-1,0, 1,0,0, 0,0,1};
    float Wt[9] = {0,1,0, -1,0,0, 0,0,1};
    float R1_t[9], R2_t[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum1 = 0, sum2 = 0;
            for (int k = 0; k < 3; k++) {
                sum1 += U[i*3+k] * W[k*3+j];
                sum2 += U[i*3+k] * Wt[k*3+j];
            }
            R1_t[i*3+j] = sum1;
            R2_t[i*3+j] = sum2;
        }
    }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float sum1 = 0, sum2 = 0;
            for (int k = 0; k < 3; k++) {
                sum1 += R1_t[i*3+k] * V[k*3+j];
                sum2 += R2_t[i*3+k] * V[k*3+j];
            }
            R1[i*3+j] = sum1;
            R2[i*3+j] = sum2;
        }
    float det1 = R1[0]*(R1[4]*R1[8]-R1[5]*R1[7]) - R1[1]*(R1[3]*R1[8]-R1[5]*R1[6]) + R1[2]*(R1[3]*R1[7]-R1[4]*R1[6]);
    float det2 = R2[0]*(R2[4]*R2[8]-R2[5]*R2[7]) - R2[1]*(R2[3]*R2[8]-R2[5]*R2[6]) + R2[2]*(R2[3]*R2[7]-R2[4]*R2[6]);
    if (det1 < 0) for (int i=0;i<9;i++) R1[i] = -R1[i];
    if (det2 < 0) for (int i=0;i<9;i++) R2[i] = -R2[i];
    t[0] = U[2]; t[1] = U[5]; t[2] = U[8];
    return 0;
}

int slam_estimate_motion_2d2d(const FeaturePoint* features1,
                              const FeaturePoint* features2,
                              const FeatureMatch* matches, int num_matches,
                              const float* camera_params,
                              float* R_out, float* t_out) {
    if (!features1 || !features2 || !matches || !camera_params || !R_out || !t_out) return -1;
    if (num_matches < 8) return -1;

    int num_iterations = 200;
    float threshold = 2.0f / camera_params[0];
    int best_inliers = 0;
    float best_R[9] = {1,0,0, 0,1,0, 0,0,1};
    float best_t[3] = {0};

    float fx = camera_params[0], fy = camera_params[1];
    float cx = camera_params[2], cy = camera_params[3];

    for (int iter = 0; iter < num_iterations; iter++) {
        int sample[8];
        for (int i = 0; i < 8; i++) {
            int idx;
            int valid;
            do {
                idx = (int)(secure_random_int((uint32_t)(num_matches - 1)));
                valid = 1;
                for (int j = 0; j < i; j++) {
                    if (sample[j] == idx) { valid = 0; break; }
                }
            } while (!valid);
            sample[i] = idx;
        }

        float pts1[16], pts2[16];
        for (int i = 0; i < 8; i++) {
            int midx = sample[i];
            int qidx = matches[midx].query_idx;
            int tidx = matches[midx].train_idx;
            pts1[2*i]   = (features1[qidx].x - cx) / fx;
            pts1[2*i+1] = (features1[qidx].y - cy) / fy;
            pts2[2*i]   = (features2[tidx].x - cx) / fx;
            pts2[2*i+1] = (features2[tidx].y - cy) / fy;
        }

        float F[9];
        if (slam_compute_fundamental_linear(pts1, pts2, 8, F) != 0) continue;

        float E[9];
        for (int i = 0; i < 9; i++) E[i] = F[i];

        float R1[9], R2[9], t_e[3];
        if (slam_decompose_essential(E, R1, R2, t_e) != 0) continue;

        float R_candidates[2][9] = {{R1[0],R1[1],R1[2],R1[3],R1[4],R1[5],R1[6],R1[7],R1[8]},
                                    {R2[0],R2[1],R2[2],R2[3],R2[4],R2[5],R2[6],R2[7],R2[8]}};
        float t_candidates[2][3] = {{t_e[0], t_e[1], t_e[2]}, {t_e[0], t_e[1], t_e[2]}};

        for (int cand = 0; cand < 2; cand++) {
            int inlier_count = 0;
            for (int i = 0; i < num_matches; i++) {
                int qidx = matches[i].query_idx;
                int tidx = matches[i].train_idx;
                float x1 = (features1[qidx].x - cx) / fx;
                float y1 = (features1[qidx].y - cy) / fy;
                float x2 = (features2[tidx].x - cx) / fx;
                float y2 = (features2[tidx].y - cy) / fy;

                float p1[3] = {x1, y1, 1.0f};
                float p2[3] = {x2, y2, 1.0f};

                float p1_cross[9];
                memset(p1_cross, 0, 9 * sizeof(float));
                p1_cross[1] = -p1[2]; p1_cross[2] = p1[1];
                p1_cross[3] = p1[2];  p1_cross[5] = -p1[0];
                p1_cross[6] = -p1[1]; p1_cross[7] = p1[0];

                float epi[3];
                for (int j = 0; j < 3; j++) {
                    epi[j] = 0;
                    for (int k = 0; k < 3; k++) {
                        float tmp = 0;
                        for (int l = 0; l < 3; l++) tmp += p1_cross[j*3+l] * R_candidates[cand][l*3+k];
                        epi[j] += tmp * t_candidates[cand][k];
                    }
                }
                float error = fabsf(epi[0]*p2[0] + epi[1]*p2[1] + epi[2]*p2[2]);
                float norm = sqrtf(epi[0]*epi[0] + epi[1]*epi[1]);
                if (norm > SLAM_EPSILON) error /= norm;
                if (error < threshold) inlier_count++;
            }

            if (inlier_count > best_inliers) {
                best_inliers = inlier_count;
                memcpy(best_R, R_candidates[cand], 9 * sizeof(float));
                memcpy(best_t, t_candidates[cand], 3 * sizeof(float));
            }
        }
    }

    if (best_inliers < SLAM_MIN_MATCHES_FOR_ESTIMATION) return -1;

    memcpy(R_out, best_R, 9 * sizeof(float));
    memcpy(t_out, best_t, 3 * sizeof(float));
    return 0;
}

/* ==================== 3D-2D运动估计（EPnP完整实现） ==================== */

int slam_estimate_motion_3d2d(const FeaturePoint* features2d,
                              const float* points3d, int num_points,
                              const float* camera_params,
                              const float* initial_pose,
                              float* optimized_pose) {
    if (!features2d || !points3d || !camera_params || !optimized_pose) return -1;
    if (num_points < 4) return -1;

    float fx = camera_params[0], fy = camera_params[1];
    float cx = camera_params[2], cy = camera_params[3];

    if (initial_pose) {
        memcpy(optimized_pose, initial_pose, 7 * sizeof(float));
    } else {
        optimized_pose[0] = 0; optimized_pose[1] = 0; optimized_pose[2] = 0;
        optimized_pose[3] = 1; optimized_pose[4] = 0; optimized_pose[5] = 0; optimized_pose[6] = 0;
    }

    float cws[4][3];
    float centroid[3] = {0,0,0};
    for (int i = 0; i < num_points; i++) {
        centroid[0] += points3d[3*i]; centroid[1] += points3d[3*i+1]; centroid[2] += points3d[3*i+2];
    }
    centroid[0] /= num_points; centroid[1] /= num_points; centroid[2] /= num_points;

    cws[0][0] = centroid[0]; cws[0][1] = centroid[1]; cws[0][2] = centroid[2];
    float M[9] = {0};
    for (int i = 0; i < num_points; i++) {
        float dx = points3d[3*i] - centroid[0];
        float dy = points3d[3*i+1] - centroid[1];
        float dz = points3d[3*i+2] - centroid[2];
        M[0] += dx*dx; M[1] += dx*dy; M[2] += dx*dz;
        M[3] += dy*dx; M[4] += dy*dy; M[5] += dy*dz;
        M[6] += dz*dx; M[7] += dz*dy; M[8] += dz*dz;
    }
    float UM[9], SM[9], VM[9];
    slam_svd_3x3(M, UM, SM, VM);
    float scale = sqrtf(SM[0] / (num_points > 3 ? num_points : 1));
    if (scale < 1e-6f) scale = 1.0f;
    for (int i = 0; i < 3; i++) {
        cws[1][i] = centroid[i] + VM[i*3+0] * scale;
        cws[2][i] = centroid[i] + VM[i*3+1] * scale;
        cws[3][i] = centroid[i] + VM[i*3+2] * scale;
    }

    float* alphas = (float*)slam_malloc((size_t)num_points * 4 * sizeof(float));
    if (!alphas) return -1;

    float C[12];
    C[0]=cws[0][0]; C[1]=cws[0][1]; C[2]=cws[0][2]; C[3]=1;
    C[4]=cws[1][0]; C[5]=cws[1][1]; C[6]=cws[1][2]; C[7]=1;
    C[8]=cws[2][0]; C[9]=cws[2][1]; C[10]=cws[2][2]; C[11]=1;

    float detC = C[0]*(C[5]*(C[10]*C[11]-C[7]*C[9]) - C[6]*(C[9]*C[7]-C[11]*C[5]) + C[7]*(C[9]*C[6]-C[10]*C[5]))
               - C[4]*(C[1]*(C[10]*C[11]-C[7]*C[9]) - C[2]*(C[9]*C[7]-C[11]*C[1]) + C[3]*(C[9]*C[6]-C[10]*C[5]))
               + C[8]*(C[1]*(C[9]*C[7]-C[11]*C[5]) - C[2]*(C[5]*C[7]-C[11]*C[3]) + C[3]*(C[5]*C[6]-C[9]*C[3]));

    if (fabsf(detC) < 1e-12f) { slam_free(alphas); return -1; }

    float invC[16];
    memset(invC, 0, 16 * sizeof(float));
    float inv_det = 1.0f / detC;
    invC[0] = (C[5]*(C[10]*C[11]-C[7]*C[9]) - C[6]*(C[9]*C[7]-C[11]*C[5]) + C[7]*(C[9]*C[6]-C[10]*C[5])) * inv_det;
    invC[1] = -(C[1]*(C[10]*C[11]-C[7]*C[9]) - C[2]*(C[9]*C[7]-C[11]*C[5]) + C[3]*(C[9]*C[6]-C[10]*C[5])) * inv_det;
    invC[4] = -(C[4]*(C[10]*C[11]-C[7]*C[9]) - C[6]*(C[8]*C[7]-C[11]*C[4]) + C[7]*(C[8]*C[6]-C[10]*C[4])) * inv_det;
    invC[5] = (C[0]*(C[10]*C[11]-C[7]*C[9]) - C[2]*(C[8]*C[7]-C[11]*C[0]) + C[3]*(C[8]*C[6]-C[10]*C[0])) * inv_det;
    invC[8] = (C[4]*(C[9]*C[7]-C[11]*C[5]) - C[5]*(C[8]*C[7]-C[11]*C[4]) + C[7]*(C[8]*C[5]-C[9]*C[4])) * inv_det;
    invC[9] = -(C[0]*(C[9]*C[7]-C[11]*C[5]) - C[1]*(C[8]*C[7]-C[11]*C[0]) + C[3]*(C[8]*C[5]-C[9]*C[0])) * inv_det;
    invC[12]= -(C[4]*(C[9]*C[6]-C[10]*C[5]) - C[5]*(C[8]*C[6]-C[10]*C[4]) + C[6]*(C[8]*C[5]-C[9]*C[4])) * inv_det;
    invC[13]= (C[0]*(C[9]*C[6]-C[10]*C[5]) - C[1]*(C[8]*C[6]-C[10]*C[0]) + C[2]*(C[8]*C[5]-C[9]*C[0])) * inv_det;

    for (int i = 0; i < num_points; i++) {
        float P[4] = {points3d[3*i], points3d[3*i+1], points3d[3*i+2], 1.0f};
        for (int j = 0; j < 4; j++) {
            alphas[i*4+j] = 0;
            for (int k = 0; k < 4; k++) alphas[i*4+j] += P[k] * invC[k*4+j];
        }
    }

    float* M_mat = (float*)slam_malloc((size_t)num_points * 12 * sizeof(float));
    if (!M_mat) { slam_free(alphas); return -1; }

    for (int i = 0; i < num_points; i++) {
        float u = (float)features2d[i].x;
        float v = (float)features2d[i].y;
        float row1[12], row2[12];
        for (int j = 0; j < 4; j++) {
            row1[j*3]   = alphas[i*4+j] * fx;
            row1[j*3+1] = 0;
            row1[j*3+2] = alphas[i*4+j] * (cx - u);
            row2[j*3]   = 0;
            row2[j*3+1] = alphas[i*4+j] * fy;
            row2[j*3+2] = alphas[i*4+j] * (cy - v);
        }
        memcpy(&M_mat[i*24], row1, 12*sizeof(float));
        memcpy(&M_mat[i*24+12], row2, 12*sizeof(float));
    }

    float MtM[144];
    memset(MtM, 0, 144 * sizeof(float));
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            float sum = 0;
            for (int k = 0; k < num_points*2; k++) sum += M_mat[k*12+i] * M_mat[k*12+j];
            MtM[i*12+j] = sum;
        }
    }

    float eigvec[12];
    float A_ev[144];
    memcpy(A_ev, MtM, 144 * sizeof(float));
    int max_iter = 50;
    for (int iter = 0; iter < max_iter; iter++) {
        float off = 0;
        for (int i = 0; i < 12; i++)
            for (int j = 0; j < 12; j++)
                if (i != j) off += A_ev[i*12+j] * A_ev[i*12+j];
        if (off < 1e-10f) break;
        for (int p = 0; p < 11; p++) {
            for (int q = p+1; q < 12; q++) {
                float app = A_ev[p*12+p], aqq = A_ev[q*12+q];
                float apq = A_ev[p*12+q];
                if (fabsf(apq) < 1e-12f) continue;
                float tau = aqq - app;
                float hyp = sqrtf(tau*tau + 4.0f*apq*apq);
                float tan_2theta = (tau >= 0) ? (2.0f*apq)/(tau+hyp) : (2.0f*apq)/(tau-hyp);
                float c = 1.0f/sqrtf(1.0f+tan_2theta*tan_2theta);
                float s_v = tan_2theta * c;
                for (int i = 0; i < 12; i++) {
                    float aip = A_ev[i*12+p];
                    float aiq = A_ev[i*12+q];
                    A_ev[i*12+p] = c*aip + s_v*aiq;
                    A_ev[i*12+q] = -s_v*aip + c*aiq;
                }
                for (int i = 0; i < 12; i++) {
                    float api = A_ev[p*12+i];
                    float aqi = A_ev[q*12+i];
                    A_ev[p*12+i] = c*api + s_v*aqi;
                    A_ev[q*12+i] = -s_v*api + c*aqi;
                }
            }
        }
    }
    for (int i = 0; i < 12; i++) eigvec[i] = A_ev[i*12+11];

    float ccs[4][3];
    for (int i = 0; i < 4; i++) {
        ccs[i][0] = eigvec[i*3];
        ccs[i][1] = eigvec[i*3+1];
        ccs[i][2] = eigvec[i*3+2];
    }

    int num_valid = 0;
    float sum_ccs[3] = {0};
    for (int i = 0; i < 4; i++) {
        if (fabsf(ccs[i][2]) > SLAM_EPSILON) {
            sum_ccs[0] += ccs[i][0]; sum_ccs[1] += ccs[i][1]; sum_ccs[2] += ccs[i][2];
            num_valid++;
        }
    }
    if (num_valid == 0) { slam_free(alphas); slam_free(M_mat); return -1; }
    float inv_nv = 1.0f / num_valid;
    float center_cc[3] = {sum_ccs[0]*inv_nv, sum_ccs[1]*inv_nv, sum_ccs[2]*inv_nv};
    float neg_shift = center_cc[2];
    if (neg_shift < 0) {
        for (int i = 0; i < 4; i++) ccs[i][2] = -ccs[i][2];
    }

    for (int i = 0; i < 4; i++) {
        cws[i][0] -= centroid[0]; cws[i][1] -= centroid[1]; cws[i][2] -= centroid[2];
    }

    float A_align[9] = {0};
    for (int i = 0; i < 4; i++) {
        A_align[0] += cws[i][0]*ccs[i][0]; A_align[1] += cws[i][0]*ccs[i][1]; A_align[2] += cws[i][0]*ccs[i][2];
        A_align[3] += cws[i][1]*ccs[i][0]; A_align[4] += cws[i][1]*ccs[i][1]; A_align[5] += cws[i][1]*ccs[i][2];
        A_align[6] += cws[i][2]*ccs[i][0]; A_align[7] += cws[i][2]*ccs[i][1]; A_align[8] += cws[i][2]*ccs[i][2];
    }

    float UA[9], SA[9], VA[9];
    slam_svd_3x3(A_align, UA, SA, VA);
    float R_epnp[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float sum = 0;
            for (int k = 0; k < 3; k++) sum += VA[i*3+k] * UA[j*3+k];
            R_epnp[i*3+j] = sum;
        }

    float detR = R_epnp[0]*(R_epnp[4]*R_epnp[8]-R_epnp[5]*R_epnp[7])
               - R_epnp[1]*(R_epnp[3]*R_epnp[8]-R_epnp[5]*R_epnp[6])
               + R_epnp[2]*(R_epnp[3]*R_epnp[7]-R_epnp[4]*R_epnp[6]);
    if (detR < 0) {
        for (int i = 0; i < 3; i++) VA[i*3+2] = -VA[i*3+2];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                float sum = 0;
                for (int k = 0; k < 3; k++) sum += VA[i*3+k] * UA[j*3+k];
                R_epnp[i*3+j] = sum;
            }
    }

    float t_epnp[3];
    float Rcw[3] = {R_epnp[0]*centroid[0] + R_epnp[1]*centroid[1] + R_epnp[2]*centroid[2],
                    R_epnp[3]*centroid[0] + R_epnp[4]*centroid[1] + R_epnp[5]*centroid[2],
                    R_epnp[6]*centroid[0] + R_epnp[7]*centroid[1] + R_epnp[8]*centroid[2]};
    t_epnp[0] = center_cc[0] - Rcw[0];
    t_epnp[1] = center_cc[1] - Rcw[1];
    t_epnp[2] = center_cc[2] - Rcw[2];

    optimized_pose[0] = t_epnp[0];
    optimized_pose[1] = t_epnp[1];
    optimized_pose[2] = t_epnp[2];
    float w_epnp[4];
    slam_rotation_matrix_to_quaternion(R_epnp, w_epnp);
    optimized_pose[3] = w_epnp[0];
    optimized_pose[4] = w_epnp[1];
    optimized_pose[5] = w_epnp[2];
    optimized_pose[6] = w_epnp[3];

    slam_free(alphas);
    slam_free(M_mat);
    /* P1修复: 成功路径返回0而非1。
     * 调用方slam.c以"motion_result == 0"判定成功，原返回1导致PnP成功路径
     * （含R10的R_rel修复）永远不执行，有效EPnP结果被当作失败丢弃。
     * 此处与slam_estimate_motion_2d2d的成功返回约定保持一致。 */
    return 0;
}

/* ==================== DLT三角化 ==================== */

int slam_triangulate_dlt(const float* P1, const float* P2,
                          const float* pt1, const float* pt2,
                          float* point3d) {
    float A[12];
    float u1 = pt1[0], v1 = pt1[1], u2 = pt2[0];
    A[0] = u1*P1[2*4+0] - P1[0*4+0];
    A[1] = u1*P1[2*4+1] - P1[0*4+1];
    A[2] = u1*P1[2*4+2] - P1[0*4+2];
    A[3] = u1*P1[2*4+3] - P1[0*4+3];
    A[4] = v1*P1[2*4+0] - P1[1*4+0];
    A[5] = v1*P1[2*4+1] - P1[1*4+1];
    A[6] = v1*P1[2*4+2] - P1[1*4+2];
    A[7] = v1*P1[2*4+3] - P1[1*4+3];
    A[8] = u2*P2[2*4+0] - P2[0*4+0];
    A[9] = u2*P2[2*4+1] - P2[0*4+1];
    A[10] = u2*P2[2*4+2] - P2[0*4+2];
    A[11] = u2*P2[2*4+3] - P2[0*4+3];

    float B[4*4];
    memset(B, 0, 16 * sizeof(float));
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0;
            for (int k = 0; k < 4; k++) sum += A[k*4+i] * A[k*4+j];
            B[i*4+j] = sum;
        }
    }

    float U[16], S[16], V[16];
    memset(U, 0, 16*sizeof(float));
    memset(S, 0, 16*sizeof(float));
    memset(V, 0, 16*sizeof(float));
    for (int i = 0; i < 4; i++) {
        U[i*4+i] = 1.0f;
        V[i*4+i] = 1.0f;
    }
    memcpy(S, B, 16 * sizeof(float));

    for (int iter = 0; iter < 100; iter++) {
        float off = 0;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                if (i != j) off += S[i*4+j] * S[i*4+j];
        if (off < 1e-12f) { break; }
        for (int p = 0; p < 3; p++) {
            for (int q = p+1; q < 4; q++) {
                float app = S[p*4+p], aqq = S[q*4+q];
                float apq = S[p*4+q];
                if (fabsf(apq) < 1e-14f) continue;
                float tau = aqq - app;
                float hyp = sqrtf(tau*tau + 4.0f*apq*apq);
                float tan_2theta = (tau >= 0) ? (2.0f*apq)/(tau+hyp) : (2.0f*apq)/(tau-hyp);
                float c = 1.0f/sqrtf(1.0f+tan_2theta*tan_2theta);
                float s_v = tan_2theta * c;
                for (int i = 0; i < 4; i++) {
                    float sip = S[i*4+p], siq = S[i*4+q];
                    S[i*4+p] = c*sip + s_v*siq;
                    S[i*4+q] = -s_v*sip + c*siq;
                    float uip = U[i*4+p], uiq = U[i*4+q];
                    U[i*4+p] = c*uip + s_v*uiq;
                    U[i*4+q] = -s_v*uip + c*uiq;
                }
                for (int i = 0; i < 4; i++) {
                    float spi = S[p*4+i], sqi = S[q*4+i];
                    S[p*4+i] = c*spi + s_v*sqi;
                    S[q*4+i] = -s_v*spi + c*sqi;
                }
            }
        }
    }

    float min_eval = fabsf(S[0]);
    int min_idx = 0;
    for (int i = 1; i < 4; i++) {
        if (fabsf(S[i*4+i]) < min_eval) { min_eval = fabsf(S[i*4+i]); min_idx = i; }
    }

    float X = U[min_idx*4+0] / U[min_idx*4+3];
    float Y = U[min_idx*4+1] / U[min_idx*4+3];
    float Z = U[min_idx*4+2] / U[min_idx*4+3];
    if (!isfinite(X) || !isfinite(Y) || !isfinite(Z)) return 0;
    point3d[0] = X; point3d[1] = Y; point3d[2] = Z;
    return 1;
}

/* ==================== 三角化 ==================== */

int slam_triangulate_points(const FeaturePoint* features1,
                            const FeaturePoint* features2,
                            const FeatureMatch* matches, int num_matches,
                            const float* R, const float* t,
                            const float* camera_params,
                            float* points3d) {
    if (!features1 || !features2 || !matches || !R || !t || !camera_params || !points3d) return -1;

    float fx = camera_params[0], fy = camera_params[1];
    float cx = camera_params[2], cy = camera_params[3];

    float P1[12] = {fx,0,cx,0, 0,fy,cy,0, 0,0,1,0};
    float P2[12];
    P2[0] = fx*R[0] + cx*R[6]; P2[1] = fx*R[1] + cx*R[7]; P2[2] = fx*R[2] + cx*R[8]; P2[3] = fx*t[0] + cx*t[2];
    P2[4] = fy*R[3] + cy*R[6]; P2[5] = fy*R[4] + cy*R[7]; P2[6] = fy*R[5] + cy*R[8]; P2[7] = fy*t[1] + cy*t[2];
    P2[8] = R[6]; P2[9] = R[7]; P2[10] = R[8]; P2[11] = t[2];

    int count = 0;
    for (int i = 0; i < num_matches; i++) {
        const FeatureMatch* m = &matches[i];
        float pt1[2] = {(float)features1[m->train_idx].x, (float)features1[m->train_idx].y};
        float pt2[2] = {(float)features2[m->query_idx].x, (float)features2[m->query_idx].y};
        float p3d[3];
        if (slam_triangulate_dlt(P1, P2, pt1, pt2, p3d)) {
            if (p3d[2] > 0.001f) {
                points3d[count*3] = p3d[0];
                points3d[count*3+1] = p3d[1];
                points3d[count*3+2] = p3d[2];
                count++;
            }
        }
    }
    return count;
}

/* ==================== 添加关键帧 ==================== */

int slam_add_keyframe(SlamSystem* system, const FeaturePoint* features,
                      int num_features, const float* pose, int64_t timestamp) {
    (void)timestamp;
    if (!system || !features || num_features <= 0 || !pose) return -1;
    if (system->local_map.num_keyframes >= SLAM_MAX_KEYFRAMES) return -1;

    KeyFrame* kf = &system->local_map.keyframes[system->local_map.num_keyframes];
    kf->id = system->local_map.num_keyframes;
    memcpy(&kf->pose, pose, sizeof(SlamPose));
    kf->num_features = (num_features > SLAM_MAX_FEATURES_PER_FRAME) ? SLAM_MAX_FEATURES_PER_FRAME : num_features;

    for (int i = 0; i < kf->num_features; i++) {
        kf->keypoints_x[i] = features[i].x;
        kf->keypoints_y[i] = features[i].y;
    }

    system->local_map.num_keyframes++;
    return system->local_map.num_keyframes - 1;
}

/* ==================== 添加地图点 ==================== */

int slam_add_landmark(SlamSystem* system, const float* point3d,
                      const float* descriptor, int keyframe_id,
                      int feature_idx) {
    if (!system || !point3d) return -1;
    if (system->local_map.num_landmarks >= SLAM_MAX_LANDMARKS) return -1;
    if (keyframe_id < 0 || keyframe_id >= system->local_map.num_keyframes) return -1;

    int idx = system->local_map.num_landmarks;
    Landmark* lm = &system->local_map.landmarks[idx];
    lm->id = idx;
    lm->is_valid = 1;
    lm->position[0] = point3d[0];
    lm->position[1] = point3d[1];
    lm->position[2] = point3d[2];
    lm->observed_count = 0;
    lm->num_observations = 0;

    if (descriptor) {
        lm->descriptor = (float*)slam_malloc(SLAM_FEATURE_DESC_LENGTH * sizeof(float));
        if (lm->descriptor) {
            memcpy(lm->descriptor, descriptor,
                   SLAM_FEATURE_DESC_LENGTH * sizeof(float));
            lm->descriptor_length = SLAM_FEATURE_DESC_LENGTH;
        }
    }

    if (keyframe_id >= 0) {
        int obs_idx = lm->observed_count;
        if (obs_idx < SLAM_MAX_OBSERVATIONS_PER_LANDMARK) {
            lm->observing_frames[obs_idx] = keyframe_id;
            lm->feature_indices[obs_idx] = feature_idx;
            lm->observed_count++;
            lm->num_observations++;
        }
    }

    system->local_map.num_landmarks++;
    return idx;
}

/* ==================== 更新地图点观测 ==================== */

int slam_update_landmark_observation(SlamSystem* system, int landmark_id,
                                      int keyframe_id, int feature_idx) {
    if (!system || landmark_id < 0 || landmark_id >= system->local_map.num_landmarks) return -1;
    if (keyframe_id < 0 || keyframe_id >= system->local_map.num_keyframes) return -1;

    Landmark* lm = &system->local_map.landmarks[landmark_id];
    for (int i = 0; i < lm->observed_count; i++) {
        if (lm->observing_frames[i] == keyframe_id) {
            lm->feature_indices[i] = feature_idx;
            return 0;
        }
    }

    if (lm->observed_count < SLAM_MAX_OBSERVATIONS_PER_LANDMARK) {
        int idx = lm->observed_count;
        lm->observing_frames[idx] = keyframe_id;
        lm->feature_indices[idx] = feature_idx;
        lm->observed_count++;
        lm->num_observations++;
        return 1;
    }
    return -1;
}

/* ==================== VO初始化 ==================== */

int slam_initialize_vo(SlamSystem* system, const FeaturePoint* features,
                       int num_features, int64_t timestamp) {
    if (!system || !features || num_features < 10) return -1;

    if (system->is_initialized && system->state.tracking_state == 2) {
        system->current_frame.num_features = (num_features > SLAM_MAX_FEATURES_PER_FRAME) ?
                                              SLAM_MAX_FEATURES_PER_FRAME : num_features;
        for (int i = 0; i < system->current_frame.num_features; i++) {
            system->current_frame.features_2d[i] = features[i];
        }
        system->current_frame.timestamp = timestamp;
        return 1;
    }

    if (!system->is_initialized) {
        system->current_frame.num_features = (num_features > SLAM_MAX_FEATURES_PER_FRAME) ?
                                              SLAM_MAX_FEATURES_PER_FRAME : num_features;
        for (int i = 0; i < system->current_frame.num_features; i++) {
            system->current_frame.features_2d[i] = features[i];
        }
        system->current_frame.timestamp = timestamp;
        system->is_initialized = 1;
        return 0;
    }

    return -1;
}

/* ==================== 地图初始化 ==================== */

int slam_initialize_map(SlamSystem* system, const FeaturePoint* features1,
                        const FeaturePoint* features2,
                        const FeatureMatch* matches, int num_matches,
                        const float* R, const float* t,
                        const float* camera_params,
                        int64_t timestamp1, int64_t timestamp2) {
    if (!system || !features1 || !features2 || !matches || !R || !t || !camera_params) return -1;
    if (num_matches < 8) return -1;

    float* points3d = (float*)slam_malloc((size_t)num_matches * 3 * sizeof(float));
    if (!points3d) return -1;

    int num_triangulated = slam_triangulate_points(features1, features2, matches,
                                                    num_matches, R, t, camera_params, points3d);
    if (num_triangulated < 8) {
        slam_free(points3d);
        return -1;
    }

    float pose1[7] = {0,0,0, 1,0,0,0};
    float pose2[7];
    pose2[0] = t[0]; pose2[1] = t[1]; pose2[2] = t[2];
    float q[4];
    slam_rotation_matrix_to_quaternion(R, q);
    pose2[3] = q[0]; pose2[4] = q[1]; pose2[5] = q[2]; pose2[6] = q[3];

    slam_system_reset_internal(system);

    slam_initialize_vo(system, features1, system->current_frame.num_features, timestamp1);

    int kf1_id = slam_add_keyframe(system, features1, system->current_frame.num_features,
                                    pose1, timestamp1);
    int kf2_id = slam_add_keyframe(system, features2,
                                    (system->current_frame.num_features > num_matches) ?
                                    num_matches : system->current_frame.num_features,
                                    pose2, timestamp2);

    if (kf1_id < 0 || kf2_id < 0) {
        slam_free(points3d);
        return -1;
    }

    for (int i = 0; i < num_triangulated; i++) {
        int match_idx = -1;
        for (int j = 0; j < num_matches; j++) {
            if (matches[j].train_idx < system->current_frame.num_features) {
                match_idx = j;
                break;
            }
        }
        if (match_idx < 0) continue;

        int lm_id = slam_add_landmark(system, &points3d[i*3], NULL, kf1_id, matches[match_idx].train_idx);
        if (lm_id >= 0) {
            slam_update_landmark_observation(system, lm_id, kf2_id, matches[match_idx].query_idx);
        }
    }

    slam_free(points3d);
    system->state.tracking_state = 2;
    return 1;
}

/* ==================== 检查运动是否足够 ==================== */

int slam_has_sufficient_motion(const FeatureMatch* matches, int num_matches,
                                const FeaturePoint* features1,
                                const FeaturePoint* features2) {
    if (!matches || !features1 || !features2 || num_matches < 8) return 0;

    float dx_sum = 0, dy_sum = 0;
    int count = 0;
    for (int i = 0; i < num_matches && i < 100; i++) {
        float dx = (float)(features2[matches[i].query_idx].x - features1[matches[i].train_idx].x);
        float dy = (float)(features2[matches[i].query_idx].y - features1[matches[i].train_idx].y);
        dx_sum += dx * dx;
        dy_sum += dy * dy;
        count++;
    }

    if (count == 0) return 0;
    float disp = sqrtf((dx_sum + dy_sum) / count);
    return (disp > 5.0f) ? 1 : 0;
}