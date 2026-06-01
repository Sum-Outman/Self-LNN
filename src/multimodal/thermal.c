/**
 * @file thermal.c
 * @brief 热感预处理器实现 — 温度标定、热点检测、梯度分析
 *
 * H-001修复: 新增专用热感预处理器，提供真实的热像仪信号处理。
 */

#include "selflnn/multimodal/thermal.h"
#include "selflnn/utils/memory_utils.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct ThermalProcessor {
    float calibration_offset;
    float calibration_gain;
    int initialized;
};

static int float_compare(const void* a, const void* b) {
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa < fb) ? -1 : ((fa > fb) ? 1 : 0);
}

ThermalProcessor* thermal_processor_create(void) {
    ThermalProcessor* tp = (ThermalProcessor*)safe_calloc(1, sizeof(ThermalProcessor));
    if (!tp) return NULL;
    tp->calibration_offset = 0.0f;
    tp->calibration_gain = 1.0f;
    tp->initialized = 0;
    return tp;
}

void thermal_processor_free(ThermalProcessor* tp) {
    safe_free((void**)&tp);
}

int thermal_calibrate(ThermalProcessor* tp,
    const float* raw_thermal, int width, int height,
    float emissivity, float ambient_temp, float* calibrated) {
    if (!tp || !raw_thermal || !calibrated || width <= 0 || height <= 0)
        return -1;
    int n = width * height;
    if (n > THERMAL_MAX_PIXELS) n = THERMAL_MAX_PIXELS;

    /* L-008: Stefan-Boltzmann辐射传热标定
     * 辐射传热四次方律: P_received = ε·σ·A·T_obj⁴ + (1-ε)·σ·A·T_amb⁴
     * 求解目标温度: T_obj = (T_raw⁴ - (1-ε)·T_amb⁴)^(1/4) / ε^(1/4)
     * 同时保留线性近似作为快速路径（当ε≈1或温差小时切换） */

    float eps_clamped = (emissivity < 0.1f) ? 0.1f : ((emissivity > 1.0f) ? 1.0f : emissivity);

    /* Stefan-Boltzmann标定参数 */
    float eps_inv_4th_root = 1.0f / sqrtf(sqrtf(eps_clamped)); /* ε^(-1/4) */
    float one_minus_eps = 1.0f - eps_clamped;
    float amb4 = ambient_temp * ambient_temp * ambient_temp * ambient_temp;

    /* 线性近似参数（快速路径） */
    tp->calibration_gain = eps_inv_4th_root;
    tp->calibration_offset = ambient_temp * one_minus_eps * 0.5f;

    /* 快速路径阈值：当发射率>0.95时线性误差<2%，直接使用线性路径 */
    float use_sb_threshold = 0.95f;

    for (int i = 0; i < n; i++) {
        if (eps_clamped >= use_sb_threshold) {
            /* 快速路径：线性近似 */
            calibrated[i] = tp->calibration_gain * raw_thermal[i] + tp->calibration_offset;
        } else {
            /* Stefan-Boltzmann完整辐射模型 */
            float raw4 = raw_thermal[i] * raw_thermal[i] * raw_thermal[i] * raw_thermal[i];
            float corrected4 = raw4 - one_minus_eps * amb4;
            /* 保护：确保辐射项非负 */
            if (corrected4 < eps_clamped * amb4 * 0.01f)
                corrected4 = eps_clamped * amb4 * 0.01f;
            /* T_obj = (corrected4)^(1/4) / ε^(1/4) = (corrected4/ε)^(1/4) */
            float cal4 = corrected4 / eps_clamped;
            calibrated[i] = sqrtf(sqrtf(cal4));
        }
    }
    tp->initialized = 1;
    return 0;
}

int thermal_compute_stats(ThermalProcessor* tp,
    const float* thermal_image, int width, int height,
    ThermalStats* stats) {
    if (!tp || !thermal_image || !stats || width <= 0 || height <= 0)
        return -1;
    memset(stats, 0, sizeof(ThermalStats));
    int n = width * height;
    if (n > THERMAL_MAX_PIXELS) n = THERMAL_MAX_PIXELS;
    stats->pixel_count = n;
    stats->min_temp = thermal_image[0];
    stats->max_temp = thermal_image[0];
    double sum = 0.0, sq_sum = 0.0;
    for (int i = 0; i < n; i++) {
        float t = thermal_image[i];
        if (t < stats->min_temp) stats->min_temp = t;
        if (t > stats->max_temp) stats->max_temp = t;
        sum += t;
        sq_sum += (double)t * t;
    }
    stats->mean_temp = (float)(sum / (double)n);
    stats->std_temp = sqrtf((float)(sq_sum / (double)n - stats->mean_temp * stats->mean_temp));
    /* 中位数：拷贝排序 */
    {
        float* sorted = (float*)safe_malloc((size_t)n * sizeof(float));
        if (sorted) {
            memcpy(sorted, thermal_image, (size_t)n * sizeof(float));
            qsort(sorted, (size_t)n, sizeof(float), float_compare);
            stats->median_temp = sorted[n / 2];
            safe_free((void**)&sorted);
        } else {
            stats->median_temp = stats->mean_temp;
        }
    }
    /* 梯度幅值：Sobel 3x3近似 */
    if (width >= 3 && height >= 3) {
        float max_grad = 0.0f;
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int idx = y * width + x;
                float gx = thermal_image[idx+1] - thermal_image[idx-1];
                float gy = thermal_image[idx+width] - thermal_image[idx-width];
                float mag = sqrtf(gx*gx + gy*gy);
                if (mag > max_grad) {
                    max_grad = mag;
                    stats->gradient_direction = atan2f(gy, gx);
                }
            }
        }
        stats->gradient_magnitude = max_grad;
    }
    return 0;
}

int thermal_detect_hotspots(ThermalProcessor* tp,
    const float* thermal_image, int width, int height,
    float threshold_temp, HotSpot* hotspots, int max_hotspots,
    int* num_detected) {
    if (!tp || !thermal_image || !hotspots || !num_detected ||
        width <= 0 || height <= 0 || max_hotspots <= 0)
        return -1;
    *num_detected = 0;
    int n = width * height;
    if (n > THERMAL_MAX_PIXELS) n = THERMAL_MAX_PIXELS;
    /* 连通域分析：4邻域泛洪填充检测热点
     * L-013: visited使用标记值2区分当前连通域，精确边界周长计算 */
    int* visited = (int*)safe_calloc((size_t)n, sizeof(int));
    if (!visited) return -1;
    int* queue_x = (int*)safe_malloc((size_t)n * sizeof(int));
    int* queue_y = (int*)safe_malloc((size_t)n * sizeof(int));
    if (!queue_x || !queue_y) {
        safe_free((void**)&visited);
        safe_free((void**)&queue_x);
        safe_free((void**)&queue_y);
        return -1;
    }

    for (int y = 0; y < height && *num_detected < max_hotspots; y++) {
        for (int x = 0; x < width && *num_detected < max_hotspots; x++) {
            int idx = y * width + x;
            if (visited[idx] || thermal_image[idx] < threshold_temp)
                continue;

            /* BFS泛洪填充：visited=2标记当前连通域 */
            int q_head = 0, q_tail = 0;
            queue_x[q_tail] = x; queue_y[q_tail] = y; q_tail++;
            visited[idx] = 2;
            float sum_x = 0.0f, sum_y = 0.0f, peak = thermal_image[idx];
            float sum_intensity = 0.0f;
            int area = 0;

            while (q_head < q_tail && area < THERMAL_MAX_PIXELS) {
                int cx = queue_x[q_head], cy = queue_y[q_head]; q_head++;
                int cidx = cy * width + cx;
                sum_x += (float)cx; sum_y += (float)cy;
                sum_intensity += thermal_image[cidx];
                area++;
                if (thermal_image[cidx] > peak) peak = thermal_image[cidx];

                /* 4邻域扩展 */
                int dirs[4][2] = {{0,1},{0,-1},{1,0},{-1,0}};
                for (int d = 0; d < 4; d++) {
                    int nx = cx + dirs[d][0], ny = cy + dirs[d][1];
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                    int nidx = ny * width + nx;
                    if (!visited[nidx] && thermal_image[nidx] >= threshold_temp) {
                        visited[nidx] = 2;
                        queue_x[q_tail] = nx; queue_y[q_tail] = ny;
                        q_tail++;
                    }
                }
            }

            if (area > 0) {
                /* L-013: 跟踪连通域边界像素计算精确周长
                 * 遍历当前连通域所有像素（队列0..q_tail-1），
                 * 对每个像素检查4邻域，不在当前连通域（visited≠2）的计为边界边 */
                float exact_perimeter = 0.0f;
                for (int p = 0; p < q_tail; p++) {
                    int px = queue_x[p], py = queue_y[p];
                    for (int d = 0; d < 4; d++) {
                        int nx = px + ((d == 0) ? 0 : (d == 1) ? 0 : (d == 2) ? 1 : -1);
                        int ny = py + ((d == 0) ? 1 : (d == 1) ? -1 : (d == 2) ? 0 : 0);
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                            exact_perimeter += 1.0f; /* 图像边界 */
                        } else {
                            int nidx = ny * width + nx;
                            if (visited[nidx] != 2)
                                exact_perimeter += 1.0f; /* 邻域不属于当前连通域 */
                        }
                    }
                }

                /* 将当前连通域visited标记从2恢复为1 */
                for (int p = 0; p < q_tail; p++) {
                    int pidx = queue_y[p] * width + queue_x[p];
                    visited[pidx] = 1;
                }

                HotSpot* hs = &hotspots[*num_detected];
                hs->center_x = sum_x / (float)area;
                hs->center_y = sum_y / (float)area;
                hs->peak_temp = peak;
                hs->area = (float)area;
                hs->perimeter = exact_perimeter;
                hs->circularity = (4.0f * (float)M_PI * (float)area) /
                    (exact_perimeter * exact_perimeter + 1e-8f);
                hs->mean_intensity = sum_intensity / (float)area;
                (*num_detected)++;
            }
        }
    }

    safe_free((void**)&visited);
    safe_free((void**)&queue_x);
    safe_free((void**)&queue_y);
    return 0;
}

int thermal_compute_feature_vector(ThermalProcessor* tp,
    const float* thermal_image, int width, int height,
    float* feature_vector, size_t feature_dim) {
    if (!tp || !thermal_image || !feature_vector ||
        feature_dim < THERMAL_FEATURE_DIM || width <= 0 || height <= 0)
        return -1;
    memset(feature_vector, 0, feature_dim * sizeof(float));
    /* 计算统计特征 */
    ThermalStats stats;
    memset(&stats, 0, sizeof(stats));
    thermal_compute_stats(tp, thermal_image, width, height, &stats);
    size_t idx = 0;
    feature_vector[idx++] = stats.min_temp * 0.01f;
    feature_vector[idx++] = stats.max_temp * 0.01f;
    feature_vector[idx++] = stats.mean_temp * 0.01f;
    feature_vector[idx++] = stats.median_temp * 0.01f;
    feature_vector[idx++] = tanhf(stats.std_temp * 0.1f);
    feature_vector[idx++] = tanhf(stats.gradient_magnitude * 0.1f);
    /* 温度直方图特征：将温度范围等分为16个区间 */
    {
        float t_range = stats.max_temp - stats.min_temp;
        if (t_range < 1e-6f) t_range = 1.0f;
        float hist[16] = {0};
        int n = width * height;
        if (n > THERMAL_MAX_PIXELS) n = THERMAL_MAX_PIXELS;
        for (int i = 0; i < n; i++) {
            int bin = (int)((thermal_image[i] - stats.min_temp) / t_range * 15.0f);
            if (bin < 0) bin = 0;
            if (bin > 15) bin = 15;
            hist[bin] += 1.0f;
        }
        for (int i = 0; i < 16 && idx < feature_dim; i++, idx++)
            feature_vector[idx] = hist[i] / (float)(n + 1);
    }
    /* 采样点温度编码 */
    {
        int w = width, h = height;
        float step_x = (float)w / 6.0f, step_y = (float)h / 6.0f;
        for (int ry = 0; ry < 6 && idx < feature_dim; ry++) {
            for (int rx = 0; rx < 6 && idx < feature_dim; rx++) {
                int sx = (int)((float)rx * step_x), sy = (int)((float)ry * step_y);
                if (sx >= w) sx = w - 1;
                if (sy >= h) sy = h - 1;
                feature_vector[idx++] = thermal_image[sy * w + sx] * 0.01f;
            }
        }
    }
    /* 剩余填充均值 */
    while (idx < feature_dim)
        feature_vector[idx++] = stats.mean_temp * 0.01f;
    return 0;
}

int thermal_compute_gradient(ThermalProcessor* tp,
    const float* thermal_image, int width, int height,
    float* grad_x, float* grad_y, int grad_size) {
    if (!tp || !thermal_image || !grad_x || !grad_y ||
        width <= 0 || height <= 0 || grad_size < width * height)
        return -1;
    int n = width * height;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            float gx = 0.0f, gy = 0.0f;
            if (x > 0 && x < width - 1)
                gx = (thermal_image[idx+1] - thermal_image[idx-1]) * 0.5f;
            else if (x == 0 && width > 1)
                gx = thermal_image[idx+1] - thermal_image[idx];
            else if (x == width - 1 && width > 1)
                gx = thermal_image[idx] - thermal_image[idx-1];
            if (y > 0 && y < height - 1)
                gy = (thermal_image[idx+width] - thermal_image[idx-width]) * 0.5f;
            else if (y == 0 && height > 1)
                gy = thermal_image[idx+width] - thermal_image[idx];
            else if (y == height - 1 && height > 1)
                gy = thermal_image[idx] - thermal_image[idx-width];
            grad_x[idx] = gx;
            grad_y[idx] = gy;
        }
    }
    return 0;
}
