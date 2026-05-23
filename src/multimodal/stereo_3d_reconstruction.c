#include "selflnn/multimodal/stereo_3d_reconstruction.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#define SR3D_MAX_DISPARITY 256
#define SR3D_CENSUS_WIN 5
#define SR3D_CENSUS_HALF 2

struct SR3DReconstructor {
    SR3DCalibration calib;
    int calibrated;
    float* disparity_buffer;
    size_t disp_size;
};

SR3DReconstructor* sr3d_create(void) {
    SR3DReconstructor* sr = (SR3DReconstructor*)safe_calloc(1, sizeof(SR3DReconstructor));
    if (!sr) return NULL;
    sr->calibrated = 0;
    sr->disparity_buffer = NULL;
    sr->disp_size = 0;
    return sr;
}

void sr3d_free(SR3DReconstructor* sr) {
    if (!sr) return;
    safe_free((void**)&sr->disparity_buffer);
    sr->disp_size = 0;
    safe_free((void**)&sr);
}

int sr3d_set_calibration(SR3DReconstructor* sr, const SR3DCalibration* calib) {
    if (!sr || !calib) return -1;
    memcpy(&sr->calib, calib, sizeof(SR3DCalibration));
    sr->calibrated = 1;
    return 0;
}

static unsigned long long sr3d_census_transform(const float* img, int x, int y, int w, int h) {
    unsigned long long code = 0;
    int bit = 0;
    float center = img[y * w + x];
    for (int wy = -SR3D_CENSUS_HALF; wy <= SR3D_CENSUS_HALF; wy++) {
        for (int wx = -SR3D_CENSUS_HALF; wx <= SR3D_CENSUS_HALF; wx++) {
            if (wx == 0 && wy == 0) continue;
            int px = x + wx, py = y + wy;
            if (px >= 0 && px < w && py >= 0 && py < h) {
                if (img[py * w + px] < center)
                    code |= (1ULL << bit);
            }
            bit++;
        }
    }
    return code;
}

static int sr3d_popcount64(unsigned long long x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (int)((x * 0x0101010101010101ULL) >> 56);
}

int sr3d_compute_dense_disparity(SR3DReconstructor* sr, const float* left, const float* right,
    int w, int h, float* disparity) {
    if (!sr || !left || !right || !disparity || w <= 0 || h <= 0) return -1;

    size_t needed = (size_t)w * h * sizeof(float);
    if (needed > sr->disp_size) {
        safe_free((void**)&sr->disparity_buffer);
        sr->disparity_buffer = (float*)safe_malloc(needed * 2);
        if (!sr->disparity_buffer) return -1;
        sr->disp_size = needed * 2;
    }
    float* right_disp = sr->disparity_buffer;
    int max_disp = SR3D_MAX_DISPARITY;
    float* census_left = (float*)safe_malloc((size_t)w * h * sizeof(float));
    float* census_right = (float*)safe_malloc((size_t)w * h * sizeof(float));
    if (!census_left || !census_right) {
        safe_free((void**)&census_left);
        safe_free((void**)&census_right);
        return -1;
    }

    for (int i = 0; i < w * h; i++) {
        unsigned long long cl = sr3d_census_transform(left, i % w, i / w, w, h);
        unsigned long long cr = sr3d_census_transform(right, i % w, i / w, w, h);
        census_left[i] = (float)cl;
        census_right[i] = (float)cr;
    }

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float best_cost = FLT_MAX;
            int best_disp = 0;
            float second_best_cost = FLT_MAX;
            unsigned long long ref_census = (unsigned long long)census_left[y * w + x];

            int d_start = 0, d_end = max_disp;
            if (x < max_disp) d_start = 0;
            else d_start = 1;
            if (x < max_disp) d_end = x;

            for (int d = d_start; d <= d_end; d++) {
                unsigned long long target_census = (unsigned long long)census_right[y * w + (x - d)];
                int hamming = sr3d_popcount64(ref_census ^ target_census);

                float cost = (float)hamming;
                if (cost < best_cost) {
                    second_best_cost = best_cost;
                    best_cost = cost;
                    best_disp = d;
                } else if (cost < second_best_cost) {
                    second_best_cost = cost;
                }
            }

            float uniqueness = (second_best_cost > 0) ? best_cost / second_best_cost : 0;
            if (best_disp > 0 && uniqueness < 0.8f) {
                disparity[y * w + x] = (float)best_disp;
            } else {
                disparity[y * w + x] = 0.0f;
            }
        }
    }

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float best_cost = FLT_MAX;
            int best_disp = 0;
            unsigned long long ref_census = (unsigned long long)census_right[y * w + x];

            for (int d = 0; d < max_disp && x + d < w; d++) {
                unsigned long long target_census = (unsigned long long)census_left[y * w + (x + d)];
                int hamming = sr3d_popcount64(ref_census ^ target_census);
                if ((float)hamming < best_cost) {
                    best_cost = (float)hamming;
                    best_disp = d;
                }
            }
            right_disp[y * w + x] = (float)best_disp;
        }
    }

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float d = disparity[y * w + x];
            if (d > 0) {
                int xr = x - (int)d;
                if (xr >= 0 && xr < w) {
                    float rd = right_disp[y * w + xr];
                    if (fabsf(d - rd) > 2.0f) {
                        disparity[y * w + x] = 0.0f;
                    }
                }
            }
        }
    }

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float d = disparity[y * w + x];
            if (d > 0 && x > 0 && x < w - 1 && y > 0 && y < h - 1) {
                float c1 = disparity[(y - 1) * w + x];
                float c2 = disparity[(y + 1) * w + x];
                float c3 = disparity[y * w + (x - 1)];
                float c4 = disparity[y * w + (x + 1)];
                if (c1 > 0 && c2 > 0 && c3 > 0 && c4 > 0) {
                    float sum = c1 + c2 + c3 + c4;
                    if (fabsf(d * 4 - sum) > 6.0f) {
                        disparity[y * w + x] = 0.0f;
                    }
                }
            }
        }
    }

    for (int y = 0; y < h; y++) {
        for (int x = 1; x < w - 1; x++) {
            float d = disparity[y * w + x];
            if (d > 0) {
                float dm1 = disparity[y * w + (x - 1)];
                float dp1 = disparity[y * w + (x + 1)];
                if (dm1 > 0 && dp1 > 0) {
                    float denom = 2.0f * dm1 - 2.0f * d + 2.0f * dp1;
                    if (fabsf(denom) > 1e-6f) {
                        float sub = (dm1 - dp1) / denom;
                        if (fabsf(sub) <= 1.0f) {
                            disparity[y * w + x] = d + sub;
                        }
                    }
                }
            }
        }
    }

    safe_free((void**)&census_left);
    safe_free((void**)&census_right);
    return 0;
}

/* =============================================================== *
 * V-017修复: 多视图一致性立体匹配 —— 光一致性验证和深度一致性检查    *
 * =============================================================== */

/**
 * @brief 计算局部窗口内的归一化互相关（NCC）
 *
 * NCC衡量左右图像对应像素窗口的相似度，用于光一致性验证。
 * NCC范围[-1, 1]，1表示完全匹配，-1表示完全负相关。
 *
 * @param left 左图像
 * @param right 右图像
 * @param w 图像宽度
 * @param h 图像高度
 * @param xl 左图像x坐标
 * @param y 左图像y坐标
 * @param xr 右图像x坐标
 * @param yr 右图像y坐标
 * @param win_size 窗口半边长
 * @return NCC值[-1,1]，匹配失败返回-999
 */
static float sr3d_ncc_window(const float* left, const float* right,
                              int w, int h,
                              int xl, int y, int xr, int yr,
                              int win_size) {
    float sum_l = 0, sum_r = 0, sum_l2 = 0, sum_r2 = 0, sum_lr = 0;
    int count = 0;

    for (int wy = -win_size; wy <= win_size; wy++) {
        for (int wx = -win_size; wx <= win_size; wx++) {
            int plx = xl + wx, ply = y + wy;
            int prx = xr + wx, pry = yr + wy;
            if (plx >= 0 && plx < w && ply >= 0 && ply < h &&
                prx >= 0 && prx < w && pry >= 0 && pry < h) {
                float lv = left[ply * w + plx];
                float rv = right[pry * w + prx];
                sum_l += lv;
                sum_r += rv;
                sum_l2 += lv * lv;
                sum_r2 += rv * rv;
                sum_lr += lv * rv;
                count++;
            }
        }
    }

    if (count < 9) return -999.0f;

    float mean_l = sum_l / (float)count;
    float mean_r = sum_r / (float)count;
    float var_l = sum_l2 / (float)count - mean_l * mean_l;
    float var_r = sum_r2 / (float)count - mean_r * mean_r;
    float cov_lr = sum_lr / (float)count - mean_l * mean_r;

    float denom = sqrtf(var_l * var_r);
    if (denom < 1e-8f) return 0.0f;

    float ncc = cov_lr / denom;
    if (ncc > 1.0f) ncc = 1.0f;
    if (ncc < -1.0f) ncc = -1.0f;
    return ncc;
}

/**
 * @brief 计算立体匹配的光一致性代价（NCC + 梯度一致性联合）
 *
 * 联合NCC和梯度幅度一致性，生成更鲁棒的匹配代价。
 * 低代价 = 高一致性。
 *
 * @param left 左图像
 * @param right 右图像
 * @param w 图像宽度
 * @param h 图像高度
 * @param x 左图像x坐标
 * @param y 行坐标
 * @param d 视差值
 * @param win_size NCC窗口大小
 * @return 光一致性代价（0=完美匹配，越大越不匹配）
 */
float sr3d_photo_consistency_cost(const float* left, const float* right,
                                   int w, int h, int x, int y, int d,
                                   int win_size) {
    int xr = x - d;
    if (xr < 0 || xr >= w) return 1000.0f;

    /* NCC得分 */
    float ncc = sr3d_ncc_window(left, right, w, h, x, y, xr, y, win_size);
    if (ncc < -900.0f) return 1000.0f;

    /* 梯度一致性 */
    float grad_cost = 0.0f;
    int grad_count = 0;
    for (int wy = -win_size; wy <= win_size; wy++) {
        for (int wx = -win_size; wx <= win_size; wx++) {
            int plx = x + wx, ply = y + wy;
            int prx = xr + wx, pry = y + wy;
            if (plx > 0 && plx < w - 1 && ply > 0 && ply < h - 1 &&
                prx > 0 && prx < w - 1 && pry > 0 && pry < h - 1) {
                float grad_lx = left[ply * w + plx + 1] - left[ply * w + plx - 1];
                float grad_ly = left[(ply + 1) * w + plx] - left[(ply - 1) * w + plx];
                float grad_rx = right[pry * w + prx + 1] - right[pry * w + prx - 1];
                float grad_ry = right[(pry + 1) * w + prx] - right[(pry - 1) * w + prx];
                float mag_l = sqrtf(grad_lx * grad_lx + grad_ly * grad_ly);
                float mag_r = sqrtf(grad_rx * grad_rx + grad_ry * grad_ry);
                grad_cost += fabsf(mag_l - mag_r);
                grad_count++;
            }
        }
    }

    if (grad_count > 0) grad_cost /= (float)grad_count;

    /* NCC代价: 高NCC = 低成本，低NCC = 高成本 */
    float ncc_cost = (1.0f - ncc) * 50.0f;

    return ncc_cost + grad_cost * 5.0f;
}

/**
 * @brief 基于光一致性的视差图优化
 *
 * 对当前视差图进行局部光一致性验证和亚像素细化。
 * 对每个像素，在其邻域搜索光一致性最优的亚像素视差值。
 *
 * @param sr 重建器句柄
 * @param left 左图像
 * @param right 右图像
 * @param w 图像宽度
 * @param h 图像高度
 * @param disparity 输入/输出视差图
 * @return 0成功，-1失败
 */
int sr3d_refine_disparity_photo_consistency(SR3DReconstructor* sr,
                                             const float* left,
                                             const float* right,
                                             int w, int h,
                                             float* disparity) {
    if (!sr || !left || !right || !disparity || w <= 0 || h <= 0) return -1;

    float* refined = (float*)safe_malloc((size_t)w * h * sizeof(float));
    if (!refined) return -1;
    memcpy(refined, disparity, (size_t)w * h * sizeof(float));

    int win_size = 3;  /* 小窗口用于局部细化 */

    for (int y = win_size; y < h - win_size; y++) {
        for (int x = win_size; x < w - win_size; x++) {
            float d_cur = disparity[y * w + x];
            if (d_cur < 0.5f) continue;

            float best_cost = sr3d_photo_consistency_cost(left, right, w, h, x, y, (int)d_cur, win_size);
            float best_d = d_cur;

            /* 局部搜索±2像素，步长0.5 */
            for (int dd = -4; dd <= 4; dd++) {
                float d_try = d_cur + (float)dd * 0.5f;
                if (d_try < 0.5f) continue;
                float cost = sr3d_photo_consistency_cost(left, right, w, h, x, y, (int)d_try, win_size);
                if (cost < best_cost) {
                    best_cost = cost;
                    best_d = d_try;
                }
            }
            refined[y * w + x] = best_d;
        }
    }

    memcpy(disparity, refined, (size_t)w * h * sizeof(float));
    safe_free((void**)&refined);
    return 0;
}

/**
 * @brief 多视图深度一致性验证
 *
 * 对点云中的每个点，验证其在左右视图间的深度一致性。
 * 如果深度差异过大或NCC过低，降低该点的置信度或将其标记为无效。
 *
 * @param sr 重建器句柄
 * @param left 左图像
 * @param right 右图像
 * @param disparity 视差图
 * @param w 图像宽度
 * @param h 图像高度
 * @param points 点云数组
 * @param point_count 点云数量
 * @param ncc_threshold NCC阈值（低于此值标记为不可靠）
 * @return 0成功，-1失败
 */
int sr3d_validate_multiview_consistency(SR3DReconstructor* sr,
                                         const float* left,
                                         const float* right,
                                         const float* disparity,
                                         int w, int h,
                                         SR3DPoint* points,
                                         int point_count,
                                         float ncc_threshold) {
    if (!sr || !left || !right || !disparity || !points || point_count <= 0) return -1;

    /* ZSFWS修复 P3-007: 未标定时使用默认值，明确标注状态 */
    float focal = sr->calibrated ? sr->calib.camera_matrix_left[0] : 525.0f;  /* 默认焦距（未标定估计值） */
    float baseline = sr->calibrated ? sr->calib.baseline_m : 0.12f;           /* 默认基线（未标定估计值） */
    int win_size = 4;

    for (int p = 0; p < point_count; p++) {
        /* 反投影点到图像坐标 */
        float z = points[p].z;
        if (z < 0.01f) { points[p].confidence = 0.0f; continue; }

        float u_l = points[p].x * focal / z + (float)w / 2.0f;
        float v_l = points[p].y * focal / z + (float)h / 2.0f;
        int ux = (int)(u_l + 0.5f);
        int uy = (int)(v_l + 0.5f);

        if (ux < 0 || ux >= w || uy < 0 || uy >= h) {
            points[p].confidence = 0.01f;
            continue;
        }

        /* 在右视图中的对应点 */
        float d = disparity[uy * w + ux];
        if (d < 0.5f) {
            points[p].confidence = 0.05f;
            continue;
        }

        int rx = ux - (int)(d + 0.5f);
        if (rx < 0 || rx >= w) {
            points[p].confidence = 0.05f;
            continue;
        }

        /* 光一致性验证：左右视图窗口NCC */
        float ncc = sr3d_ncc_window(left, right, w, h, ux, uy, rx, uy, win_size);
        if (ncc < -900.0f) {
            points[p].confidence = 0.01f;
            continue;
        }

        /* 深度一致性：反算的深度 vs 视差推导的深度 */
        float z_from_disp = focal * baseline / d;
        float depth_ratio = fabsf(z - z_from_disp) / fmaxf(z, 0.01f);

        /* 综合置信度：NCC（80%）+ 深度一致性（20%） */
        float depth_score = 1.0f / (1.0f + depth_ratio * 5.0f);
        float ncc_score = (ncc + 1.0f) * 0.5f;
        float conf = ncc_score * 0.8f + depth_score * 0.2f;

        if (ncc < ncc_threshold) conf *= 0.2f;
        if (depth_ratio > 0.3f) conf *= 0.5f;

        if (conf < 0.0f) conf = 0.0f;
        if (conf > 1.0f) conf = 1.0f;
        points[p].confidence = conf;
    }

    return 0;
}

int sr3d_generate_point_cloud(SR3DReconstructor* sr, const float* left, const float* right,
    const float* disparity, int w, int h, SR3DPoint* points, int* count) {
    (void)right;
    if (!sr || !left || !disparity || !points || !count) return -1;

    /* ZSFWS修复 P3-007: 未标定时使用默认值，明确标注状态 */
    float focal = sr->calibrated ? sr->calib.camera_matrix_left[0] : 525.0f;  /* 默认焦距（未标定估计值） */
    float baseline = sr->calibrated ? sr->calib.baseline_m : 0.12f;           /* 默认基线（未标定估计值） */
    float cx = (float)w / 2.0f, cy = (float)h / 2.0f;

    int pc = 0;
    for (int y = 0; y < h && pc < SR3D_MAX_POINTS; y++) {
        for (int x = 0; x < w && pc < SR3D_MAX_POINTS; x++) {
            float d = disparity[y * w + x];
            if (d < 0.5f || d > 100.0f) continue;

            float z = focal * baseline / d;
            if (z < 0.05f || z > 100.0f) continue;

            points[pc].x = (float)(x - (int)cx) * z / focal;
            points[pc].y = (float)(y - (int)cy) * z / focal;
            points[pc].z = z;
            points[pc].confidence = 1.0f / (1.0f + d * 0.005f);

            int idx = y * w + x;
            points[pc].r = (unsigned char)(fmaxf(0.0f, fminf(1.0f, left[idx])) * 255.0f);
            points[pc].g = points[pc].r;
            points[pc].b = points[pc].r;
            points[pc].segment_id = 0;
            points[pc].nx = 0.0f;
            points[pc].ny = 0.0f;
            points[pc].nz = 0.0f;
            pc++;
        }
    }
    *count = pc;

    if (pc > 3) {
        float* cov = (float*)safe_malloc((size_t)pc * 9 * sizeof(float));
        if (cov) {
            float search_radius = 0.1f;
            for (int i = 0; i < pc; i++) {
                int neighbors[256];
                int nn = 0;
                for (int j = 0; j < pc && nn < 256; j++) {
                    if (i == j) continue;
                    float dx = points[j].x - points[i].x;
                    float dy = points[j].y - points[i].y;
                    float dz = points[j].z - points[i].z;
                    float dist2 = dx * dx + dy * dy + dz * dz;
                    if (dist2 < search_radius * search_radius && dist2 > 1e-10f) {
                        neighbors[nn++] = j;
                    }
                }
                if (nn >= 3) {
                    float mx = 0, my = 0, mz = 0;
                    for (int k = 0; k < nn; k++) {
                        mx += points[neighbors[k]].x;
                        my += points[neighbors[k]].y;
                        mz += points[neighbors[k]].z;
                    }
                    mx /= nn; my /= nn; mz /= nn;

                    float c11 = 0, c12 = 0, c13 = 0, c22 = 0, c23 = 0, c33 = 0;
                    for (int k = 0; k < nn; k++) {
                        float dx = points[neighbors[k]].x - mx;
                        float dy = points[neighbors[k]].y - my;
                        float dz = points[neighbors[k]].z - mz;
                        c11 += dx * dx; c12 += dx * dy; c13 += dx * dz;
                        c22 += dy * dy; c23 += dy * dz;
                        c33 += dz * dz;
                    }
                    c11 /= nn; c12 /= nn; c13 /= nn;
                    c22 /= nn; c23 /= nn; c33 /= nn;

                    float a = 1.0f, b = -(c11 + c22 + c33);
                    (void)a;
                    float c = c11 * c22 + c11 * c33 + c22 * c33 - c12 * c12 - c13 * c13 - c23 * c23;
                    float d2 = -(c11 * c22 * c33 + 2 * c12 * c13 * c23
                                - c11 * c23 * c23 - c22 * c13 * c13 - c33 * c12 * c12);
                    float q = (3.0f * c - b * b) / 9.0f;
                    float r2 = (9.0f * b * c - 27.0f * d2 - 2.0f * b * b * b) / 54.0f;
                    float theta = acosf(fmaxf(-1.0f, fminf(1.0f, r2 / sqrtf(fmaxf(0, -q * q * q)))));
                    float eig1 = 2.0f * sqrtf(-q) * cosf(theta / 3.0f) - b / 3.0f;
                    float eig2 = 2.0f * sqrtf(-q) * cosf((theta + 2.0f * 3.14159265f) / 3.0f) - b / 3.0f;
                    float eig3 = 2.0f * sqrtf(-q) * cosf((theta + 4.0f * 3.14159265f) / 3.0f) - b / 3.0f;
                    (void)eig1; (void)eig2;

                    float A[9] = {c11 - eig3, c12, c13, c12, c22 - eig3, c23, c13, c23, c33 - eig3};
                    float det = A[0] * (A[4] * A[8] - A[5] * A[7])
                              - A[1] * (A[3] * A[8] - A[5] * A[6])
                              + A[2] * (A[3] * A[7] - A[4] * A[6]);
                    if (fabsf(det) > 1e-12f) {
                        float nx = A[1] * A[5] - A[2] * A[4];
                        float ny = A[2] * A[3] - A[0] * A[5];
                        float nz = A[0] * A[4] - A[1] * A[3];
                        float len = sqrtf(nx * nx + ny * ny + nz * nz);
                        if (len > 1e-10f) {
                            points[i].nx = nx / len;
                            points[i].ny = ny / len;
                            points[i].nz = nz / len;
                        }
                    }
                }
                if (points[i].nx == 0 && points[i].ny == 0 && points[i].nz == 0) {
                    points[i].nx = 0; points[i].ny = 0; points[i].nz = 1;
                }
            }
            safe_free((void**)&cov);
        }
    }

    return 0;
}

int sr3d_filter_point_cloud(SR3DReconstructor* sr, SR3DPoint* points, int count, int* filtered_count) {
    if (!sr || !points || !filtered_count) return -1;

    float* mean_k = (float*)safe_calloc(count, sizeof(float));
    float* std_k = (float*)safe_calloc(count, sizeof(float));
    if (!mean_k || !std_k) {
        safe_free((void**)&mean_k);
        safe_free((void**)&std_k);
        return -1;
    }

    int k = 20;
    for (int i = 0; i < count; i++) {
        float sum = 0, sum2 = 0;
        int n = 0;
        for (int j = 0; j < count && n < k; j++) {
            if (i == j) continue;
            float dx = points[j].x - points[i].x;
            float dy = points[j].y - points[i].y;
            float dz = points[j].z - points[i].z;
            float d = sqrtf(dx * dx + dy * dy + dz * dz);
            if (d < 0.5f) {
                sum += d;
                sum2 += d * d;
                n++;
            }
        }
        if (n > 0) {
            mean_k[i] = sum / n;
            std_k[i] = sqrtf(sum2 / n - mean_k[i] * mean_k[i]);
        } else {
            mean_k[i] = FLT_MAX;
            std_k[i] = 0;
        }
    }

    float global_mean = 0, global_std = 0;
    int valid = 0;
    for (int i = 0; i < count; i++) {
        if (mean_k[i] < FLT_MAX) {
            global_mean += mean_k[i];
            valid++;
        }
    }
    if (valid > 0) global_mean /= valid;
    for (int i = 0; i < count; i++) {
        if (mean_k[i] < FLT_MAX) {
            global_std += (mean_k[i] - global_mean) * (mean_k[i] - global_mean);
        }
    }
    if (valid > 1) global_std = sqrtf(global_std / (valid - 1));
    else global_std = 0.1f;

    float threshold = global_mean + 1.5f * global_std;
    int fc = 0;
    for (int i = 0; i < count; i++) {
        if (mean_k[i] > threshold) continue;
        if (points[i].confidence < 0.01f) continue;
        points[fc++] = points[i];
    }
    *filtered_count = fc;

    safe_free((void**)&mean_k);
    safe_free((void**)&std_k);
    return 0;
}

int sr3d_reconstruct_surface(SR3DReconstructor* sr, const SR3DPoint* points, int count,
    SR3DTriangle* triangles, int* tri_count) {
    if (!sr || !points || !triangles || !tri_count) return -1;

    if (count < 4) {
        *tri_count = 0;
        return 0;
    }

    float min_x = FLT_MAX, max_x = -FLT_MAX, min_y = FLT_MAX, max_y = -FLT_MAX;
    float min_z = FLT_MAX, max_z = -FLT_MAX;
    for (int i = 0; i < count; i++) {
        if (points[i].x < min_x) min_x = points[i].x;
        if (points[i].x > max_x) max_x = points[i].x;
        if (points[i].y < min_y) min_y = points[i].y;
        if (points[i].y > max_y) max_y = points[i].y;
        if (points[i].z < min_z) min_z = points[i].z;
        if (points[i].z > max_z) max_z = points[i].z;
    }

    float range = fmaxf(max_x - min_x, fmaxf(max_y - min_y, max_z - min_z));
    if (range < 1e-6f) {
        *tri_count = 0;
        return 0;
    }

    int grid_res = (int)(sqrtf((float)count) + 0.5f);
    if (grid_res < 4) grid_res = 4;
    if (grid_res > 128) grid_res = 128;

    int* grid_idx = (int*)safe_malloc((size_t)grid_res * grid_res * sizeof(int));
    if (!grid_idx) return -1;
    memset(grid_idx, -1, (size_t)grid_res * grid_res * sizeof(int));

    for (int i = 0; i < count; i++) {
        int gx = (int)((points[i].x - min_x) / (max_x - min_x + 1e-6f) * (grid_res - 1));
        int gy = (int)((points[i].y - min_y) / (max_y - min_y + 1e-6f) * (grid_res - 1));
        gx = gx < 0 ? 0 : (gx >= grid_res ? grid_res - 1 : gx);
        gy = gy < 0 ? 0 : (gy >= grid_res ? grid_res - 1 : gy);
        int idx = gy * grid_res + gx;
        if (grid_idx[idx] < 0) {
            grid_idx[idx] = i;
        } else {
            float existing_z = points[grid_idx[idx]].z;
            if (points[i].confidence > points[grid_idx[idx]].confidence ||
                (fabsf(points[i].z) < fabsf(existing_z) && points[i].confidence > 0.5f)) {
                grid_idx[idx] = i;
            }
        }
    }

    int tc = 0;
    for (int gy = 0; gy < grid_res - 1 && tc < SR3D_MAX_TRIANGLES; gy++) {
        for (int gx = 0; gx < grid_res - 1 && tc < SR3D_MAX_TRIANGLES; gx++) {
            int i00 = grid_idx[gy * grid_res + gx];
            int i10 = grid_idx[gy * grid_res + gx + 1];
            int i01 = grid_idx[(gy + 1) * grid_res + gx];
            int i11 = grid_idx[(gy + 1) * grid_res + gx + 1];

            if (i00 < 0 || i01 < 0 || i10 < 0 || i11 < 0) {
                if (i00 >= 0 && i10 >= 0 && i01 >= 0) {
                    float d1 = fabsf(points[i00].z - points[i10].z);
                    float d2 = fabsf(points[i00].z - points[i01].z);
                    float d3 = fabsf(points[i10].z - points[i01].z);
                    if (d1 < range * 0.3f && d2 < range * 0.3f && d3 < range * 0.3f) {
                        triangles[tc].v1 = i00; triangles[tc].v2 = i10; triangles[tc].v3 = i01;
                        float ex1 = points[i10].x - points[i00].x, ey1 = points[i10].y - points[i00].y, ez1 = points[i10].z - points[i00].z;
                        float ex2 = points[i01].x - points[i00].x, ey2 = points[i01].y - points[i00].y, ez2 = points[i01].z - points[i00].z;
                        float nx = ey1 * ez2 - ez1 * ey2;
                        float ny = ez1 * ex2 - ex1 * ez2;
                        float nz = ex1 * ey2 - ey1 * ex2;
                        float nl = sqrtf(nx * nx + ny * ny + nz * nz);
                        if (nl > 1e-10f) { triangles[tc].normal[0] = nx / nl; triangles[tc].normal[1] = ny / nl; triangles[tc].normal[2] = nz / nl; }
                        else { triangles[tc].normal[0] = 0; triangles[tc].normal[1] = 0; triangles[tc].normal[2] = 1; }
                        tc++;
                    }
                }
                continue;
            }

            float d1 = fabsf(points[i00].z - points[i10].z);
            float d2 = fabsf(points[i00].z - points[i01].z);
            float d3 = fabsf(points[i10].z - points[i01].z);
            float d4 = fabsf(points[i11].z - points[i10].z);
            float d5 = fabsf(points[i11].z - points[i01].z);

            int valid1 = (d1 < range * 0.3f && d2 < range * 0.3f && d3 < range * 0.3f);
            int valid2 = (d4 < range * 0.3f && d5 < range * 0.3f && fabsf(points[i11].z - points[i00].z) < range * 0.3f);

            if (valid1 && tc < SR3D_MAX_TRIANGLES) {
                triangles[tc].v1 = i00; triangles[tc].v2 = i10; triangles[tc].v3 = i01;
                float ex1 = points[i10].x - points[i00].x, ey1 = points[i10].y - points[i00].y, ez1 = points[i10].z - points[i00].z;
                float ex2 = points[i01].x - points[i00].x, ey2 = points[i01].y - points[i00].y, ez2 = points[i01].z - points[i00].z;
                float nx = ey1 * ez2 - ez1 * ey2;
                float ny = ez1 * ex2 - ex1 * ez2;
                float nz = ex1 * ey2 - ey1 * ex2;
                float nl = sqrtf(nx * nx + ny * ny + nz * nz);
                if (nl > 1e-10f) { triangles[tc].normal[0] = nx / nl; triangles[tc].normal[1] = ny / nl; triangles[tc].normal[2] = nz / nl; }
                else { triangles[tc].normal[0] = 0; triangles[tc].normal[1] = 0; triangles[tc].normal[2] = 1; }
                tc++;
            }
            if (valid2 && tc < SR3D_MAX_TRIANGLES) {
                triangles[tc].v1 = i00; triangles[tc].v2 = i11; triangles[tc].v3 = i01;
                float ex1 = points[i11].x - points[i00].x, ey1 = points[i11].y - points[i00].y, ez1 = points[i11].z - points[i00].z;
                float ex2 = points[i01].x - points[i00].x, ey2 = points[i01].y - points[i00].y, ez2 = points[i01].z - points[i00].z;
                float nx = ey1 * ez2 - ez1 * ey2;
                float ny = ez1 * ex2 - ex1 * ez2;
                float nz = ex1 * ey2 - ey1 * ex2;
                float nl = sqrtf(nx * nx + ny * ny + nz * nz);
                if (nl > 1e-10f) { triangles[tc].normal[0] = nx / nl; triangles[tc].normal[1] = ny / nl; triangles[tc].normal[2] = nz / nl; }
                else { triangles[tc].normal[0] = 0; triangles[tc].normal[1] = 0; triangles[tc].normal[2] = 1; }
                tc++;
            }
        }
    }

    *tri_count = tc;
    safe_free((void**)&grid_idx);
    return 0;
}

int sr3d_texture_map(SR3DReconstructor* sr, SR3DPoint* points, int count,
    SR3DTriangle* triangles, int tri_count, const float* image, int w, int h) {
    if (!sr || !points || !triangles || !image) return -1;

    for (int i = 0; i < tri_count; i++) {
        int indices[3] = {triangles[i].v1, triangles[i].v2, triangles[i].v3};
        float avg_r = 0, avg_g = 0, avg_b = 0;
        int valid = 0;
        for (int k = 0; k < 3; k++) {
            int idx = indices[k];
            if (idx >= 0 && idx < count) {
                float px = points[idx].x, py = points[idx].y;
                int u = (int)(px + (float)w / 2.0f);
                int v = (int)(py + (float)h / 2.0f);
                if (u >= 0 && u < w && v >= 0 && v < h) {
                    float val = image[v * w + u];
                    avg_r += val; avg_g += val; avg_b += val;
                    valid++;
                }
            }
        }
        if (valid > 0) {
            avg_r /= valid; avg_g /= valid; avg_b /= valid;
            for (int k = 0; k < 3; k++) {
                int idx = indices[k];
                if (idx >= 0 && idx < count) {
                    points[idx].r = (unsigned char)(fmaxf(0, fminf(255, avg_r * 255)));
                    points[idx].g = (unsigned char)(fmaxf(0, fminf(255, avg_g * 255)));
                    points[idx].b = (unsigned char)(fmaxf(0, fminf(255, avg_b * 255)));
                }
            }
        }
    }
    return 0;
}

int sr3d_segment_objects(SR3DReconstructor* sr, const SR3DPoint* points, int count,
    SR3DSegment* segments, int* seg_count) {
    if (!sr || !points || !segments || !seg_count) return -1;

    int sc = 0;
    int* assigned = (int*)safe_calloc(count, sizeof(int));
    if (!assigned) return -1;

    for (int i = 0; i < count && sc < SR3D_MAX_SEGMENTS; i++) {
        if (assigned[i]) continue;

        int queue[4096];
        int qh = 0, qt = 0;
        queue[qt++] = i;
        assigned[i] = sc + 1;
        int cluster_pts = 0;

        while (qh < qt && cluster_pts < 10000) {
            int ci = queue[qh++];
            cluster_pts++;

            for (int j = 0; j < count && qt < 4096; j++) {
                if (assigned[j]) continue;
                float dx = points[j].x - points[ci].x;
                float dy = points[j].y - points[ci].y;
                float dz = points[j].z - points[ci].z;
                float dist = sqrtf(dx * dx + dy * dy + dz * dz);
                if (dist < 0.15f) {
                    assigned[j] = sc + 1;
                    queue[qt++] = j;
                }
            }
        }

        if (cluster_pts > 10) {
            SR3DSegment* seg = &segments[sc];
            memset(seg, 0, sizeof(SR3DSegment));
            seg->segment_id = sc + 1;
            seg->point_count = cluster_pts;
            snprintf(seg->label, sizeof(seg->label), "物体_%d", sc + 1);

            float sx = 0, sy = 0, sz = 0;
            float min_x = FLT_MAX, max_x = -FLT_MAX;
            float min_y = FLT_MAX, max_y = -FLT_MAX;
            float min_z = FLT_MAX, max_z = -FLT_MAX;
            for (int j = 0; j < count; j++) {
                if (assigned[j] == sc + 1) {
                    sx += points[j].x; sy += points[j].y; sz += points[j].z;
                    if (points[j].x < min_x) min_x = points[j].x;
                    if (points[j].x > max_x) max_x = points[j].x;
                    if (points[j].y < min_y) min_y = points[j].y;
                    if (points[j].y > max_y) max_y = points[j].y;
                    if (points[j].z < min_z) min_z = points[j].z;
                    if (points[j].z > max_z) max_z = points[j].z;
                }
            }
            seg->center[0] = sx / cluster_pts;
            seg->center[1] = sy / cluster_pts;
            seg->center[2] = sz / cluster_pts;
            seg->extents[0] = max_x - min_x;
            seg->extents[1] = max_y - min_y;
            seg->extents[2] = max_z - min_z;
            seg->volume = seg->extents[0] * seg->extents[1] * seg->extents[2];
            seg->confidence = fminf(1.0f, (float)cluster_pts / 100.0f);
            sc++;
        }
    }

    safe_free((void**)&assigned);
    *seg_count = sc;
    return 0;
}

int sr3d_measure_object(SR3DReconstructor* sr, const SR3DPoint* points, int count,
    int segment_id, SR3DMeasurement* measurement) {
    if (!sr || !points || !measurement) return -1;
    memset(measurement, 0, sizeof(SR3DMeasurement));

    float min_x = FLT_MAX, max_x = -FLT_MAX, min_y = FLT_MAX, max_y = -FLT_MAX, min_z = FLT_MAX, max_z = -FLT_MAX;
    int found = 0;
    for (int i = 0; i < count; i++) {
        if (points[i].segment_id != segment_id) continue;
        if (points[i].x < min_x) min_x = points[i].x;
        if (points[i].x > max_x) max_x = points[i].x;
        if (points[i].y < min_y) min_y = points[i].y;
        if (points[i].y > max_y) max_y = points[i].y;
        if (points[i].z < min_z) min_z = points[i].z;
        if (points[i].z > max_z) max_z = points[i].z;
        found = 1;
    }
    if (!found) return -1;

    measurement->width_m = max_x - min_x;
    measurement->height_m = max_y - min_y;
    measurement->depth_m = max_z - min_z;
    measurement->area_m2 = measurement->width_m * measurement->depth_m;
    measurement->volume_m3 = measurement->area_m2 * measurement->height_m;
    snprintf(measurement->unit, sizeof(measurement->unit), "m");
    return 0;
}

int sr3d_measure_distance(SR3DReconstructor* sr, const float* p1, const float* p2, float* distance) {
    if (!sr || !p1 || !p2 || !distance) return -1;
    float dx = p2[0] - p1[0], dy = p2[1] - p1[1], dz = p2[2] - p1[2];
    *distance = sqrtf(dx * dx + dy * dy + dz * dz);
    return 0;
}

int sr3d_label_scene(SR3DReconstructor* sr, const SR3DPoint* points, int count,
    const SR3DSegment* segments, int seg_count, char* scene_desc, size_t max_len) {
    if (!sr || !points || !segments || !scene_desc) return -1;

    float total_volume = 0;
    float max_vol = 0;
    int largest_idx = -1;
    for (int i = 0; i < seg_count; i++) {
        total_volume += segments[i].volume;
        if (segments[i].volume > max_vol) {
            max_vol = segments[i].volume;
            largest_idx = i;
        }
    }

    if (seg_count > 0 && largest_idx >= 0) {
        snprintf(scene_desc, max_len,
            "3D场景: %d个物体, %d个点云点, 最大物体'%s'(%.2fm³), 空间范围(%.2f×%.2f×%.2f)m",
            seg_count, count, segments[largest_idx].label, segments[largest_idx].volume,
            segments[largest_idx].extents[0], segments[largest_idx].extents[1], segments[largest_idx].extents[2]);
    } else {
        snprintf(scene_desc, max_len, "3D场景: %d个点云点, 未检测到独立物体", count);
    }
    return 0;
}

#define SR3D_DELAUNAY_MAX_TRIANGLES 65536
#define SR3D_MAX_VERTICES  32768
#define SR3D_SUPER_TRI_SCALE 100.0f

typedef struct {
    int v0, v1, v2;
    float circum_x, circum_y, circum_r2;
} SR3DDelaunayTri;

typedef struct {
    float x, y;
} SR3DVertex2D;

static float sr3d_circumradius_sq(float ax, float ay, float bx, float by, float cx, float cy,
                                   float* ccx, float* ccy) {
    float d = 2.0f * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if (fabsf(d) < 1e-12f) return -1.0f;
    float ax2 = ax * ax + ay * ay;
    float bx2 = bx * bx + by * by;
    float cx2 = cx * cx + cy * cy;
    *ccx = (ax2 * (by - cy) + bx2 * (cy - ay) + cx2 * (ay - by)) / d;
    *ccy = (ax2 * (cx - bx) + bx2 * (ax - cx) + cx2 * (bx - ax)) / d;
    float dx = ax - *ccx, dy = ay - *ccy;
    return dx * dx + dy * dy;
}

static int sr3d_in_circumcircle(float px, float py, const SR3DDelaunayTri* tri) {
    float dx = px - tri->circum_x;
    float dy = py - tri->circum_y;
    return (dx * dx + dy * dy) < tri->circum_r2;
}

int sr3d_generate_mesh(SR3DReconstructor* sr, const SR3DPoint* points, int point_count,
                        int* p_tri_count, int* p_tri_indices, int max_tri_indices) {
    if (!sr || !points || point_count < 3 || !p_tri_count || !p_tri_indices) return -1;

    SR3DVertex2D verts[SR3D_MAX_VERTICES];
    int n_verts = point_count < SR3D_MAX_VERTICES ? point_count : SR3D_MAX_VERTICES;
    float min_x = FLT_MAX, max_x = -FLT_MAX, min_y = FLT_MAX, max_y = -FLT_MAX;

    for (int i = 0; i < n_verts; i++) {
        verts[i].x = points[i].x;
        verts[i].y = points[i].y;
        if (verts[i].x < min_x) min_x = verts[i].x;
        if (verts[i].x > max_x) max_x = verts[i].x;
        if (verts[i].y < min_y) min_y = verts[i].y;
        if (verts[i].y > max_y) max_y = verts[i].y;
    }

    float dx = max_x - min_x, dy = max_y - min_y;
    float dmax = (dx > dy ? dx : dy) * SR3D_SUPER_TRI_SCALE;
    float mid_x = (min_x + max_x) * 0.5f, mid_y = (min_y + max_y) * 0.5f;

    SR3DDelaunayTri tri_buf[SR3D_DELAUNAY_MAX_TRIANGLES];
    int tri_count = 0;

    tri_buf[0].v0 = -1; tri_buf[0].v1 = -2; tri_buf[0].v2 = -3;
    tri_buf[0].circum_r2 = sr3d_circumradius_sq(
        mid_x, mid_y + dmax, mid_x - dmax, mid_y - dmax,
        mid_x + dmax, mid_y - dmax,
        &tri_buf[0].circum_x, &tri_buf[0].circum_y);
    tri_count = 1;

    for (int p = 0; p < n_verts; p++) {
        float px = verts[p].x, py = verts[p].y;
        int edge_buf[SR3D_DELAUNAY_MAX_TRIANGLES * 3];
        int edge_count = 0;

        int new_tri_count = 0;
        for (int t = 0; t < tri_count; t++) {
            if (sr3d_in_circumcircle(px, py, &tri_buf[t])) {
                int ev[3] = {tri_buf[t].v0, tri_buf[t].v1, tri_buf[t].v2};
                for (int e = 0; e < 3; e++) {
                    int a = ev[e], b = ev[(e + 1) % 3];
                    int found = 0;
                    for (int k = 0; k < edge_count; k++) {
                        if ((edge_buf[k * 2] == b && edge_buf[k * 2 + 1] == a)) {
                            if (k < edge_count - 1) {
                                edge_buf[k * 2] = edge_buf[(edge_count - 1) * 2];
                                edge_buf[k * 2 + 1] = edge_buf[(edge_count - 1) * 2 + 1];
                            }
                            edge_count--;
                            found = 1;
                            break;
                        }
                    }
                    if (!found) {
                        edge_buf[edge_count * 2] = a;
                        edge_buf[edge_count * 2 + 1] = b;
                        edge_count++;
                    }
                }
            } else {
                tri_buf[new_tri_count++] = tri_buf[t];
            }
        }

        for (int e = 0; e < edge_count; e++) {
            int a = edge_buf[e * 2], b = edge_buf[e * 2 + 1];
            if (p >= 0 && new_tri_count < SR3D_DELAUNAY_MAX_TRIANGLES) {
                SR3DDelaunayTri* nt = &tri_buf[new_tri_count];
                nt->v0 = a; nt->v1 = b; nt->v2 = p;
                float ax_v = (a >= 0) ? verts[a].x : ((a == -1) ? mid_x : (a == -2 ? mid_x - dmax : mid_x + dmax));
                float ay_v = (a >= 0) ? verts[a].y : ((a == -1) ? mid_y + dmax : (a == -2 ? mid_y - dmax : mid_y - dmax));
                float bx_v = (b >= 0) ? verts[b].x : ((b == -1) ? mid_x : (b == -2 ? mid_x - dmax : mid_x + dmax));
                float by_v = (b >= 0) ? verts[b].y : ((b == -1) ? mid_y + dmax : (b == -2 ? mid_y - dmax : mid_y - dmax));
                nt->circum_r2 = sr3d_circumradius_sq(ax_v, ay_v, bx_v, by_v, px, py,
                                                      &nt->circum_x, &nt->circum_y);
                new_tri_count++;
            }
        }
        tri_count = new_tri_count;
        if (tri_count >= SR3D_DELAUNAY_MAX_TRIANGLES - 10) break;
    }

    int out_tri = 0;
    for (int t = 0; t < tri_count && out_tri * 3 + 2 < max_tri_indices; t++) {
        if (tri_buf[t].v0 < 0 || tri_buf[t].v1 < 0 || tri_buf[t].v2 < 0) continue;
        p_tri_indices[out_tri * 3 + 0] = tri_buf[t].v0;
        p_tri_indices[out_tri * 3 + 1] = tri_buf[t].v1;
        p_tri_indices[out_tri * 3 + 2] = tri_buf[t].v2;
        out_tri++;
    }
    *p_tri_count = out_tri;
    return 0;
}
