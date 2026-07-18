#include "selflnn/multimodal/stereo_depth_enhance.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SDE_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define SDE_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define SDE_CLAMP(v, lo, hi) SDE_MAX((lo), SDE_MIN((hi), (v)))

struct SDEHandler {
    SDEStereoConfig config;
    SDECameraParams cam;
    int has_camera_params;
    float left_k[9];
    float right_k[9];
    float rotation[9];
    float translation[3];
    float dist_left[5];
    float dist_right[5];
    int has_calibration;
    float* rect_map_left_x;
    float* rect_map_left_y;
    float* rect_map_right_x;
    float* rect_map_right_y;
    int rect_map_width;
    int rect_map_height;
    float* temporal_buffer[SDE_TEMPORAL_BUFFER_SIZE];
    float* temporal_confidence[SDE_TEMPORAL_BUFFER_SIZE];
    int temporal_frame_count;
    int temporal_write_idx;
};

struct SDEFusionHandler {
    SDEFusionConfig config;
    float* accumulated_depth;
    float* accumulated_weight;
    int accum_width;
    int accum_height;
    int accum_initialized;
    float* temporal_buffer[SDE_TEMPORAL_BUFFER_SIZE];
    float* temporal_conf_buffer[SDE_TEMPORAL_BUFFER_SIZE];
    int temporal_idx;
    int temporal_count;
};

static unsigned long long _census_transform_9x7(const float* patch, int patch_size)
{
    unsigned long long code = 0;
    float center = patch[patch_size / 2];
    for (int i = 0; i < patch_size && i < 63; i++) {
        if (patch[i] < center) {
            code |= (1ULL << i);
        }
    }
    return code;
}

static unsigned int _hamming_distance(unsigned long long a, unsigned long long b)
{
    unsigned long long x = a ^ b;
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (unsigned int)((x * 0x0101010101010101ULL) >> 56);
}

static int _extract_patch(const float* image, int x, int y,
                           int width, int height, int half,
                           float* patch_out)
{
    int size = 2 * half + 1;
    int idx = 0;
    for (int dy = -half; dy <= half; dy++) {
        for (int dx = -half; dx <= half; dx++) {
            int px = x + dx;
            int py = y + dy;
            if (px >= 0 && px < width && py >= 0 && py < height) {
                patch_out[idx++] = image[py * width + px];
            } else {
                patch_out[idx++] = 0.0f;
            }
        }
    }
    return size * size;
}

static float _compute_bt_cost(float l_val, float r_val)
{
    float diff = l_val - r_val;
    if (diff < -0.5f) return -diff - 0.25f;
    if (diff > 0.5f) return diff - 0.25f;
    return diff * diff;
}

static float _compute_gradient_x(const float* image, int x, int y, int width)
{
    if (x <= 0 || x >= width - 1) return 0.0f;
    return image[y * width + x + 1] - image[y * width + x - 1];
}

static void _compute_sobel_x(const float* image, int width, int height,
                              float* grad_out)
{
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float gx = -image[(y - 1) * width + (x - 1)]
                       - 2.0f * image[y * width + (x - 1)]
                       - image[(y + 1) * width + (x - 1)]
                       + image[(y - 1) * width + (x + 1)]
                       + 2.0f * image[y * width + (x + 1)]
                       + image[(y + 1) * width + (x + 1)];
            grad_out[y * width + x] = gx / 8.0f;
        }
    }
    for (int y = 0; y < height; y++) {
        grad_out[y * width + 0] = 0.0f;
        grad_out[y * width + width - 1] = 0.0f;
    }
    for (int x = 0; x < width; x++) {
        grad_out[0 * width + x] = 0.0f;
        grad_out[(height - 1) * width + x] = 0.0f;
    }
}

static float _compute_pixel_cost_buffered(const float* left_img, const float* right_img,
                                           const float* grad_left, const float* grad_right,
                                           int x, int y, int d, int width, int height,
                                           SDECostType cost_type,
                                           float* patch_buf, int patch_buf_size)
{
    if (x - d < 0) return FLT_MAX;

    switch (cost_type) {
    case SDE_COST_CENSUS_SAD:
    case SDE_COST_CENSUS_SSD: {
        int half = 4;
        int patch_size = 81;
        if (patch_buf_size < patch_size * 2) return 100.0f;
        float* left_patch = patch_buf;
        float* right_patch = patch_buf + patch_size;
        _extract_patch(left_img, x, y, width, height, half, left_patch);
        _extract_patch(right_img, x - d, y, width, height, half, right_patch);

        unsigned long long lc = _census_transform_9x7(left_patch, patch_size);
        unsigned long long rc = _census_transform_9x7(right_patch, patch_size);
        unsigned int hamming = _hamming_distance(lc, rc);

        float app_cost = 0.0f;
        if (cost_type == SDE_COST_CENSUS_SAD) {
            for (int i = 0; i < patch_size; i++) {
                app_cost += fabsf(left_patch[i] - right_patch[i]);
            }
            app_cost /= (float)patch_size;
        } else {
            for (int i = 0; i < patch_size; i++) {
                float diff_val = left_patch[i] - right_patch[i];
                app_cost += diff_val * diff_val;
            }
            app_cost /= (float)patch_size;
        }
        return (float)hamming * 0.7f + app_cost * 0.3f;
    }
    case SDE_COST_GRADIENT_CENSUS: {
        if (!grad_left || !grad_right) return 100.0f;
        int half = 3;
        int patch_size = 49;
        if (patch_buf_size < patch_size * 2) return 100.0f;
        float* lp = patch_buf;
        float* rp = patch_buf + patch_size;
        _extract_patch(grad_left, x, y, width, height, half, lp);
        _extract_patch(grad_right, x - d, y, width, height, half, rp);

        unsigned long long lc = _census_transform_9x7(lp, patch_size);
        unsigned long long rc = _census_transform_9x7(rp, patch_size);
        float census_cost = (float)_hamming_distance(lc, rc);
        float app_cost = fabsf(left_img[y * width + x] - right_img[y * width + (x - d)]);
        return census_cost * 0.5f + app_cost * 0.5f;
    }
    case SDE_COST_BT:
    default: {
        float l = left_img[y * width + x];
        float r = right_img[y * width + (x - d)];
        float r_m1 = (x - d - 1 >= 0) ? right_img[y * width + (x - d - 1)] : r;
        float r_p1 = (x - d + 1 < width) ? right_img[y * width + (x - d + 1)] : r;
        float cost_m1 = _compute_bt_cost(l, r_m1);
        float cost_0 = _compute_bt_cost(l, r);
        float cost_p1 = _compute_bt_cost(l, r_p1);
        return SDE_MIN(cost_m1, SDE_MIN(cost_0, cost_p1));
    }
    }
}

static void _aggregate_path(float* cost_volume, int w, int h, int disp_range,
                             int dx, int dy,
                             float p1, float p2)
{
    int start_y = (dy >= 0) ? 0 : h - 1;
    int end_y = (dy >= 0) ? h : -1;
    int step_y = (dy >= 0) ? 1 : -1;

    int start_x = (dx >= 0) ? 0 : w - 1;
    int end_x = (dx >= 0) ? w : -1;
    int step_x = (dx >= 0) ? 1 : -1;

    int wh = w * h;

    for (int y = start_y; y != end_y; y += step_y) {
        for (int x = start_x; x != end_x; x += step_x) {
            int px = x - dx;
            int py = y - dy;
            if (px < 0 || px >= w || py < 0 || py >= h) continue;

            float min_prev_all = FLT_MAX;
            for (int d = 0; d < disp_range; d++) {
                int prev_idx = d * wh + py * w + px;
                if (cost_volume[prev_idx] < min_prev_all) {
                    min_prev_all = cost_volume[prev_idx];
                }
            }

            for (int d = 0; d < disp_range; d++) {
                int idx = d * wh + y * w + x;
                int prev_idx = d * wh + py * w + px;

                float best_prev = cost_volume[prev_idx];

                if (d > 0) {
                    int pd = (d - 1) * wh + py * w + px;
                    float v = cost_volume[pd] + p1;
                    if (v < best_prev) best_prev = v;
                }
                if (d < disp_range - 1) {
                    int nd = (d + 1) * wh + py * w + px;
                    float v = cost_volume[nd] + p1;
                    if (v < best_prev) best_prev = v;
                }

                float p2_pen = min_prev_all + p2;
                if (p2_pen < best_prev) best_prev = p2_pen;

                cost_volume[idx] += best_prev - min_prev_all;
            }
        }
    }
}

static int _disparity_to_depth(float disp, float fx, float baseline)
{
    if (disp < 0.5f) return -1;
    return (int)(fx * baseline / disp * 1000.0f);
}

static void _median_filter_2d(float* data, int w, int h, int ksize,
                               float* out)
{
    int half = ksize / 2;
    int* values = (int*)safe_malloc((size_t)ksize * (size_t)ksize * sizeof(int));
    if (!values) return;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int count = 0;
            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    int px = x + kx;
                    int py = y + ky;
                    if (px >= 0 && px < w && py >= 0 && py < h) {
                        values[count++] = (int)data[py * w + px];
                    }
                }
            }
            for (int i = 0; i < count - 1; i++) {
                for (int j = 0; j < count - 1 - i; j++) {
                    if (values[j] > values[j + 1]) {
                        int t = values[j];
                        values[j] = values[j + 1];
                        values[j + 1] = t;
                    }
                }
            }
            out[y * w + x] = (float)values[count / 2];
        }
    }
    safe_free((void**)&values);
}

static void _gaussian_kernel(int size, float sigma, float* kernel)
{
    int half = size / 2;
    float sum = 0.0f;
    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float v = expf(-(float)(x * x + y * y) / (2.0f * sigma * sigma));
            kernel[(y + half) * size + (x + half)] = v;
            sum += v;
        }
    }
    for (int i = 0; i < size * size; i++) kernel[i] /= sum;
}

static void _bilateral_filter_2d(float* data, int w, int h,
                                  int ksize, float sigma_s, float sigma_r,
                                  float* guide, float* out)
{
    int half = ksize / 2;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sum_w = 0.0f;
            float sum_v = 0.0f;
            float center_val = data[y * w + x];
            float center_guide = guide ? guide[y * w + x] : center_val;

            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    int px = x + kx;
                    int py = y + ky;
                    if (px < 0 || px >= w || py < 0 || py >= h) continue;

                    float spatial_w = expf(-(float)(kx * kx + ky * ky) / (2.0f * sigma_s * sigma_s));
                    float range_diff = (guide ? guide[py * w + px] : data[py * w + px]) - center_guide;
                    float range_w = expf(-(range_diff * range_diff) / (2.0f * sigma_r * sigma_r));
                    float wgt = spatial_w * range_w;

                    sum_w += wgt;
                    sum_v += wgt * data[py * w + px];
                }
            }
            out[y * w + x] = (sum_w > 1e-10f) ? sum_v / sum_w : 0.0f;
        }
    }
}

static void _interpolate_invalid_disparity(float* disparity, int w, int h)
{
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (disparity[y * w + x] > 0.5f) continue;
            float sum_d = 0.0f;
            float sum_w = 0.0f;
            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int px = x + dx;
                    int py = y + dy;
                    if (px < 0 || px >= w || py < 0 || py >= h) continue;
                    float d = disparity[py * w + px];
                    if (d < 0.5f) continue;
                    float wgt = 1.0f / (sqrtf((float)(dx * dx + dy * dy)) + 1.0f);
                    sum_d += d * wgt;
                    sum_w += wgt;
                }
            }
            if (sum_w > 1e-10f) {
                disparity[y * w + x] = sum_d / sum_w;
            }
        }
    }
}

SDEHandler* sde_create(const SDEStereoConfig* config)
{
    SDEHandler* handler = (SDEHandler*)safe_calloc(1, sizeof(SDEHandler));
    if (!handler) return NULL;

    if (config) {
        memcpy(&handler->config, config, sizeof(SDEStereoConfig));
    } else {
        handler->config = sde_get_default_config();
    }

    handler->has_camera_params = 0;
    handler->has_calibration = 0;
    handler->rect_map_left_x = NULL;
    handler->rect_map_left_y = NULL;
    handler->rect_map_right_x = NULL;
    handler->rect_map_right_y = NULL;
    handler->rect_map_width = 0;
    handler->rect_map_height = 0;
    handler->temporal_frame_count = 0;
    handler->temporal_write_idx = 0;
    for (int i = 0; i < SDE_TEMPORAL_BUFFER_SIZE; i++) {
        handler->temporal_buffer[i] = NULL;
        handler->temporal_confidence[i] = NULL;
    }

    return handler;
}

void sde_free(SDEHandler* handler)
{
    if (!handler) return;
    safe_free((void**)&handler->rect_map_left_x);
    safe_free((void**)&handler->rect_map_left_y);
    safe_free((void**)&handler->rect_map_right_x);
    safe_free((void**)&handler->rect_map_right_y);
    for (int i = 0; i < SDE_TEMPORAL_BUFFER_SIZE; i++) {
        safe_free((void**)&handler->temporal_buffer[i]);
        safe_free((void**)&handler->temporal_confidence[i]);
    }
    safe_free((void**)&handler);
}

int sde_set_calibration(SDEHandler* handler,
                         const float left_k[9], const float right_k[9],
                         const float rotation[9], const float translation[3],
                         const float dist_left[5], const float dist_right[5])
{
    if (!handler || !left_k || !right_k) return -1;

    memcpy(handler->left_k, left_k, 9 * sizeof(float));
    memcpy(handler->right_k, right_k, 9 * sizeof(float));
    if (rotation) memcpy(handler->rotation, rotation, 9 * sizeof(float));
    if (translation) memcpy(handler->translation, translation, 3 * sizeof(float));
    if (dist_left) memcpy(handler->dist_left, dist_left, 5 * sizeof(float));
    if (dist_right) memcpy(handler->dist_right, dist_right, 5 * sizeof(float));

    handler->cam.fx = left_k[0];
    handler->cam.fy = left_k[4];
    handler->cam.cx = left_k[2];
    handler->cam.cy = left_k[5];
    if (translation) {
        handler->cam.baseline = fabsf(translation[0]);
    }

    handler->has_calibration = 1;
    handler->has_camera_params = 1;

    safe_free((void**)&handler->rect_map_left_x);
    safe_free((void**)&handler->rect_map_left_y);
    safe_free((void**)&handler->rect_map_right_x);
    safe_free((void**)&handler->rect_map_right_y);
    handler->rect_map_width = 0;
    handler->rect_map_height = 0;

    return 0;
}

int sde_set_camera_params(SDEHandler* handler, const SDECameraParams* params)
{
    if (!handler || !params) return -1;
    memcpy(&handler->cam, params, sizeof(SDECameraParams));
    handler->has_camera_params = 1;
    return 0;
}

SDEStereoConfig sde_get_default_config(void)
{
    SDEStereoConfig cfg;
    cfg.min_disparity = 0;
    cfg.max_disparity = 128;
    cfg.window_size = 9;
    cfg.num_paths = 8;
    cfg.p1 = 8.0f;
    cfg.p2 = 32.0f;
    cfg.cost_type = SDE_COST_CENSUS_SAD;
    cfg.enable_lr_check = 1;
    cfg.lr_check_threshold = 1.0f;
    cfg.enable_speckle_filter = 1;
    cfg.speckle_window_size = 100;
    cfg.speckle_range = 2;
    cfg.enable_subpixel = 1;
    cfg.enable_median_filter = 1;
    cfg.median_filter_size = 5;
    cfg.enable_bilateral_filter = 0;
    cfg.bilateral_sigma_color = 0.1f;
    cfg.bilateral_sigma_space = 5.0f;
    cfg.bilateral_kernel_size = 7;
    return cfg;
}

int sde_set_config(SDEHandler* handler, const SDEStereoConfig* config)
{
    if (!handler || !config) return -1;
    memcpy(&handler->config, config, sizeof(SDEStereoConfig));
    return 0;
}

int sde_get_config(SDEHandler* handler, SDEStereoConfig* config)
{
    if (!handler || !config) return -1;
    memcpy(config, &handler->config, sizeof(SDEStereoConfig));
    return 0;
}

int sde_compute_disparity(SDEHandler* handler,
                           const float* left_image, const float* right_image,
                           int width, int height, int channels,
                           float* disparity_out, float* confidence_out)
{
    if (!handler || !left_image || !right_image || !disparity_out ||
        width <= 0 || height <= 0) return -1;

    SDEStereoConfig* cfg = &handler->config;
    int disp_range = cfg->max_disparity - cfg->min_disparity;
    if (disp_range <= 0 || disp_range > SDE_MAX_DISPARITY) return -1;

    const float* left_gray = left_image;
    const float* right_gray = right_image;
    float* left_gray_buf = NULL;
    float* right_gray_buf = NULL;
    float* left_grad = NULL;
    float* right_grad = NULL;

    if (channels >= 3) {
        left_gray_buf = (float*)safe_malloc((size_t)width * (size_t)height * sizeof(float));
        right_gray_buf = (float*)safe_malloc((size_t)width * (size_t)height * sizeof(float));
        if (!left_gray_buf || !right_gray_buf) {
            safe_free((void**)&left_gray_buf);
            safe_free((void**)&right_gray_buf);
            return -1;
        }
        for (int i = 0; i < (size_t)width * (size_t)height; i++) {
            left_gray_buf[i] = 0.299f * left_image[i * 3] +
                               0.587f * left_image[i * 3 + 1] +
                               0.114f * left_image[i * 3 + 2];
            right_gray_buf[i] = 0.299f * right_image[i * 3] +
                                0.587f * right_image[i * 3 + 1] +
                                0.114f * right_image[i * 3 + 2];
        }
        left_gray = left_gray_buf;
        right_gray = right_gray_buf;
    }

    size_t cv_size = (size_t)disp_range * height * width;
    float* cost_volume = (float*)safe_calloc(cv_size, sizeof(float));
    if (!cost_volume) {
        safe_free((void**)&left_gray_buf);
        safe_free((void**)&right_gray_buf);
        return -1;
    }

    float p1 = cfg->p1;
    float p2 = cfg->p2;

    if (cfg->cost_type == SDE_COST_GRADIENT_CENSUS) {
        left_grad = (float*)safe_malloc((size_t)width * (size_t)height * sizeof(float));
        right_grad = (float*)safe_malloc((size_t)width * (size_t)height * sizeof(float));
        if (left_grad && right_grad) {
            _compute_sobel_x(left_gray, width, height, left_grad);
            _compute_sobel_x(right_gray, width, height, right_grad);
        }
    }

    int patch_half = (cfg->cost_type == SDE_COST_GRADIENT_CENSUS) ? 3 : 4;
    int patch_pixels = (2 * patch_half + 1) * (2 * patch_half + 1);
    int patch_buf_size = patch_pixels * 2;
    float* patch_buf = (float*)safe_malloc(patch_buf_size * sizeof(float));

    for (int d = 0; d < disp_range; d++) {
        int disp_val = d + cfg->min_disparity;
        for (int y = 0; y < height; y++) {
            for (int x = disp_val; x < width; x++) {
                int idx = (d * height + y) * width + x;
                cost_volume[idx] = _compute_pixel_cost_buffered(
                    left_gray, right_gray, left_grad, right_grad,
                    x, y, disp_val, width, height, cfg->cost_type,
                    patch_buf, patch_buf_size);
            }
        }
    }

    safe_free((void**)&patch_buf);
    safe_free((void**)&left_grad);
    safe_free((void**)&right_grad);

    int directions[SDE_MAX_PATH_DIRECTIONS][2] = {
        {1, 0}, {-1, 0}, {0, 1}, {0, -1},
        {1, 1}, {-1, -1}, {1, -1}, {-1, 1}
    };

    float* aggregated = (float*)safe_calloc(cv_size, sizeof(float));
    if (!aggregated) {
        safe_free((void**)&cost_volume);
        safe_free((void**)&left_gray_buf);
        safe_free((void**)&right_gray_buf);
        return -1;
    }
    memcpy(aggregated, cost_volume, cv_size * sizeof(float));

    float** path_costs = (float**)safe_calloc(cfg->num_paths, sizeof(float*));
    if (!path_costs) {
        safe_free((void**)&aggregated);
        safe_free((void**)&cost_volume);
        safe_free((void**)&left_gray_buf);
        safe_free((void**)&right_gray_buf);
        return -1;
    }

    for (int pi = 0; pi < cfg->num_paths && pi < SDE_MAX_PATH_DIRECTIONS; pi++) {
        path_costs[pi] = (float*)safe_calloc(cv_size, sizeof(float));
        if (!path_costs[pi]) {
            for (int j = 0; j < pi; j++) safe_free((void**)&path_costs[j]);
            safe_free((void**)&path_costs);
            safe_free((void**)&aggregated);
            safe_free((void**)&cost_volume);
            safe_free((void**)&left_gray_buf);
            safe_free((void**)&right_gray_buf);
            return -1;
        }
        memcpy(path_costs[pi], cost_volume, cv_size * sizeof(float));
        _aggregate_path(path_costs[pi], width, height, disp_range,
                         directions[pi][0], directions[pi][1], p1, p2);
    }

    for (int i = 0; i < (int)cv_size; i++) {
        aggregated[i] = 0.0f;
    }
    for (int pi = 0; pi < cfg->num_paths && pi < SDE_MAX_PATH_DIRECTIONS; pi++) {
        for (int i = 0; i < (int)cv_size; i++) {
            aggregated[i] += path_costs[pi][i];
        }
    }

    for (int i = 0; i < (int)cv_size; i++) {
        aggregated[i] /= (float)cfg->num_paths;
    }

    for (int pi = 0; pi < cfg->num_paths && pi < SDE_MAX_PATH_DIRECTIONS; pi++) {
        safe_free((void**)&path_costs[pi]);
    }
    safe_free((void**)&path_costs);

    float* left_disparity = (float*)safe_calloc((size_t)width * (size_t)height, sizeof(float));
    float* left_confidence = (float*)safe_calloc((size_t)width * (size_t)height, sizeof(float));
    if (!left_disparity || !left_confidence) {
        safe_free((void**)&left_disparity);
        safe_free((void**)&left_confidence);
        safe_free((void**)&aggregated);
        safe_free((void**)&cost_volume);
        safe_free((void**)&left_gray_buf);
        safe_free((void**)&right_gray_buf);
        safe_free((void**)&left_grad);
        safe_free((void**)&right_grad);
        safe_free((void**)&patch_buf);
        return -1;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float min_cost = FLT_MAX;
            float second_min_cost = FLT_MAX;
            int best_d = 0;

            for (int d = 0; d < disp_range; d++) {
                float c = aggregated[(d * height + y) * width + x];
                if (c < min_cost) {
                    second_min_cost = min_cost;
                    min_cost = c;
                    best_d = d;
                } else if (c < second_min_cost) {
                    second_min_cost = c;
                }
            }

            left_disparity[y * width + x] = (float)(best_d + cfg->min_disparity);

            if (second_min_cost > min_cost + 1e-10f) {
                float ratio = min_cost / second_min_cost;
                left_confidence[y * width + x] = 1.0f - SDE_CLAMP(ratio, 0.0f, 1.0f);
            } else {
                left_confidence[y * width + x] = 0.0f;
            }

            if (cfg->enable_subpixel && best_d > 0 && best_d < disp_range - 1) {
                float c_1 = aggregated[((best_d - 1) * height + y) * width + x];
                float c0 = aggregated[(best_d * height + y) * width + x];
                float c1 = aggregated[((best_d + 1) * height + y) * width + x];
                float denom = c_1 - 2.0f * c0 + c1;
                if (fabsf(denom) > 1e-10f) {
                    float offset = (c_1 - c1) / (2.0f * denom);
                    left_disparity[y * width + x] = (float)(best_d + cfg->min_disparity) + SDE_CLAMP(offset, -0.5f, 0.5f);
                }
            }
        }
    }

    if (cfg->enable_lr_check) {
        float* right_disparity = (float*)safe_calloc((size_t)width * (size_t)height, sizeof(float));
        if (right_disparity) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    float min_cost = FLT_MAX;
                    int best_d = 0;
                    for (int d = 0; d < disp_range; d++) {
                        int rx = x + d + cfg->min_disparity;
                        if (rx >= width) break;
                        float c = aggregated[(d * height + y) * width + rx];
                        if (c < min_cost) {
                            min_cost = c;
                            best_d = d;
                        }
                    }
                    right_disparity[y * width + x] = (float)(best_d + cfg->min_disparity);
                }
            }

            int invalid_count = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    float dl = left_disparity[y * width + x];
                    int rx = x - (int)(dl + 0.5f);
                    if (rx >= 0 && rx < width) {
                        float dr = right_disparity[y * width + rx];
                        if (fabsf(dl - dr) > cfg->lr_check_threshold) {
                            left_disparity[y * width + x] = 0.0f;
                            left_confidence[y * width + x] = 0.0f;
                            invalid_count++;
                        }
                    }
                }
            }
            safe_free((void**)&right_disparity);
        }
    }

    safe_free((void**)&aggregated);
    safe_free((void**)&cost_volume);

    if (cfg->enable_speckle_filter && cfg->speckle_window_size > 0) {
        int* visited = (int*)safe_calloc((size_t)width * (size_t)height, sizeof(int));
        if (visited) {
            int* region_stack_x = (int*)safe_malloc((size_t)width * (size_t)height * sizeof(int));
            int* region_stack_y = (int*)safe_malloc((size_t)width * (size_t)height * sizeof(int));
            if (region_stack_x && region_stack_y) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        if (visited[y * width + x] || left_disparity[y * width + x] < 0.5f) continue;
                        int stack_top = 0;
                        region_stack_x[stack_top] = x;
                        region_stack_y[stack_top] = y;
                        stack_top++;
                        visited[y * width + x] = 1;
                        int region_size = 1;

                        while (stack_top > 0) {
                            stack_top--;
                            int cx = region_stack_x[stack_top];
                            int cy = region_stack_y[stack_top];
                            float cd = left_disparity[cy * width + cx];
                            int neigh_dx[] = {-1, 1, 0, 0};
                            int neigh_dy[] = {0, 0, -1, 1};
                            for (int n = 0; n < 4; n++) {
                                int nx = cx + neigh_dx[n];
                                int ny = cy + neigh_dy[n];
                                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                                if (visited[ny * width + nx]) continue;
                                float nd = left_disparity[ny * width + nx];
                                if (nd < 0.5f) continue;
                                if (fabsf(cd - nd) <= cfg->speckle_range) {
                                    visited[ny * width + nx] = 1;
                                    region_stack_x[stack_top] = nx;
                                    region_stack_y[stack_top] = ny;
                                    stack_top++;
                                    region_size++;
                                }
                            }
                        }

                        if (region_size < cfg->speckle_window_size) {
                            for (int ry = 0; ry < height; ry++) {
                                for (int rx = 0; rx < width; rx++) {
                                    if (visited[ry * width + rx] == 1) {
                                        left_disparity[ry * width + rx] = 0.0f;
                                        left_confidence[ry * width + rx] = 0.0f;
                                    }
                                }
                            }
                            for (int ry = 0; ry < height; ry++) {
                                for (int rx = 0; rx < width; rx++) {
                                    if (left_disparity[ry * width + rx] < 0.5f &&
                                        visited[ry * width + rx] == 1) {
                                        visited[ry * width + rx] = 2;
                                    }
                                }
                            }
                        } else {
                            for (int ry = 0; ry < height; ry++) {
                                for (int rx = 0; rx < width; rx++) {
                                    if (visited[ry * width + rx] == 1) {
                                        visited[ry * width + rx] = 2;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            safe_free((void**)&region_stack_x);
            safe_free((void**)&region_stack_y);
            safe_free((void**)&visited);
        }
    }

    if (cfg->enable_median_filter && cfg->median_filter_size >= 3) {
        float* filtered = (float*)safe_malloc((size_t)width * (size_t)height * sizeof(float));
        if (filtered) {
            _median_filter_2d(left_disparity, width, height,
                              cfg->median_filter_size, filtered);
            memcpy(left_disparity, filtered, (size_t)width * (size_t)height * sizeof(float));
        }
        safe_free((void**)&filtered);
    }

    _interpolate_invalid_disparity(left_disparity, width, height);

    memcpy(disparity_out, left_disparity, (size_t)width * (size_t)height * sizeof(float));
    if (confidence_out) {
        memcpy(confidence_out, left_confidence, (size_t)width * (size_t)height * sizeof(float));
    }

    safe_free((void**)&left_disparity);
    safe_free((void**)&left_confidence);
    safe_free((void**)&left_gray_buf);
    safe_free((void**)&right_gray_buf);

    return 0;
}

int sde_compute_depth(SDEHandler* handler,
                       const float* disparity, int width, int height,
                       float* depth_out, float* confidence_out)
{
    if (!handler || !disparity || !depth_out || width <= 0 || height <= 0) return -1;
    if (!handler->has_camera_params && !handler->has_calibration) return -1;

    float fx = handler->cam.fx;
    float baseline = handler->cam.baseline;

    if (fx < 1.0f || baseline < 0.001f) return -1;

    for (int i = 0; i < (size_t)width * (size_t)height; i++) {
        float d = disparity[i];
        if (d > 0.5f) {
            depth_out[i] = fx * baseline / d;
        } else {
            depth_out[i] = 0.0f;
        }
    }

    if (confidence_out) {
        for (int i = 0; i < (size_t)width * (size_t)height; i++) {
            float d = disparity[i];
            if (d > 0.5f) {
                float depth = depth_out[i];
                float depth_conf = 1.0f / (1.0f + depth * 0.01f);
                float disp_conf = 1.0f - SDE_CLAMP(d / (float)handler->config.max_disparity, 0.0f, 1.0f);
                confidence_out[i] = depth_conf * disp_conf;
            } else {
                confidence_out[i] = 0.0f;
            }
        }
    }

    return 0;
}

int sde_generate_point_cloud(SDEHandler* handler,
                              const float* left_image,
                              const float* depth_map,
                              int width, int height, int channels,
                              float* point_cloud_out, int max_points,
                              int* out_point_count)
{
    if (!handler || !left_image || !depth_map || !point_cloud_out || !out_point_count ||
        width <= 0 || height <= 0 || max_points <= 0) return -1;
    if (!handler->has_camera_params && !handler->has_calibration) return -1;

    float fx = handler->cam.fx;
    float fy = handler->cam.fy;
    float cx = handler->cam.cx;
    float cy = handler->cam.cy;

    if (cx < 1.0f) cx = (float)width / 2.0f;
    if (cy < 1.0f) cy = (float)height / 2.0f;

    int pc = 0;
    for (int y = 0; y < height && pc < max_points; y++) {
        for (int x = 0; x < width && pc < max_points; x++) {
            float depth = depth_map[y * width + x];
            if (depth < 0.001f) continue;

            float p_x = (float)(x - cx) * depth / fx;
            float p_y = (float)(y - cy) * depth / fy;
            float p_z = depth;

            point_cloud_out[pc * 6 + 0] = p_x;
            point_cloud_out[pc * 6 + 1] = p_y;
            point_cloud_out[pc * 6 + 2] = p_z;

            if (channels >= 3 && left_image) {
                point_cloud_out[pc * 6 + 3] = left_image[(y * width + x) * channels + 0];
                point_cloud_out[pc * 6 + 4] = left_image[(y * width + x) * channels + 1];
                point_cloud_out[pc * 6 + 5] = left_image[(y * width + x) * channels + 2];
            } else {
                point_cloud_out[pc * 6 + 3] = 0.5f;
                point_cloud_out[pc * 6 + 4] = 0.5f;
                point_cloud_out[pc * 6 + 5] = 0.5f;
            }
            pc++;
        }
    }

    *out_point_count = pc;
    return 0;
}

int sde_compute_stereo_pipeline(SDEHandler* handler,
                                 const float* left_image, const float* right_image,
                                 int width, int height, int channels,
                                 float* disparity_out, float* depth_out,
                                 float* point_cloud_out, int max_points,
                                 int* out_point_count,
                                 float* confidence_out)
{
    if (!handler || !left_image || !right_image || !disparity_out ||
        width <= 0 || height <= 0) return -1;

    int ret = sde_compute_disparity(handler, left_image, right_image,
                                     width, height, channels,
                                     disparity_out, confidence_out);
    if (ret != 0) return ret;

    if (depth_out) {
        ret = sde_compute_depth(handler, disparity_out, width, height,
                                 depth_out, NULL);
        if (ret != 0) return ret;
    }

    if (point_cloud_out && out_point_count && depth_out) {
        ret = sde_generate_point_cloud(handler, left_image, depth_out,
                                        width, height, channels,
                                        point_cloud_out, max_points,
                                        out_point_count);
    }

    return ret;
}

SDEFusionHandler* sde_fusion_create(const SDEFusionConfig* config)
{
    SDEFusionHandler* fh = (SDEFusionHandler*)safe_calloc(1, sizeof(SDEFusionHandler));
    if (!fh) return NULL;

    if (config) {
        memcpy(&fh->config, config, sizeof(SDEFusionConfig));
    } else {
        fh->config = sde_fusion_get_default_config();
    }

    fh->accumulated_depth = NULL;
    fh->accumulated_weight = NULL;
    fh->accum_width = 0;
    fh->accum_height = 0;
    fh->accum_initialized = 0;
    fh->temporal_idx = 0;
    fh->temporal_count = 0;
    for (int i = 0; i < SDE_TEMPORAL_BUFFER_SIZE; i++) {
        fh->temporal_buffer[i] = NULL;
        fh->temporal_conf_buffer[i] = NULL;
    }

    return fh;
}

void sde_fusion_free(SDEFusionHandler* fh)
{
    if (!fh) return;
    safe_free((void**)&fh->accumulated_depth);
    safe_free((void**)&fh->accumulated_weight);
    for (int i = 0; i < SDE_TEMPORAL_BUFFER_SIZE; i++) {
        safe_free((void**)&fh->temporal_buffer[i]);
        safe_free((void**)&fh->temporal_conf_buffer[i]);
    }
    safe_free((void**)&fh);
}

SDEFusionConfig sde_fusion_get_default_config(void)
{
    SDEFusionConfig cfg;
    cfg.method = SDE_FUSION_WEIGHTED_AVERAGE;
    cfg.confidence_threshold = 0.1f;
    cfg.temporal_alpha = 0.3f;
    cfg.num_fusion_views = 4;
    cfg.enable_hole_filling = 1;
    cfg.hole_filling_radius = 5;
    cfg.enable_edge_preserve = 1;
    cfg.edge_preserve_sigma = 0.05f;
    cfg.enable_temporal_fusion = 0;
    cfg.temporal_smoothing_factor = 0.7f;
    return cfg;
}

int sde_fusion_add_depth(SDEFusionHandler* fh,
                          const float* depth_map, const float* confidence,
                          int width, int height,
                          const float* pose_matrix)
{
    if (!fh || !depth_map || width <= 0 || height <= 0) return -1;

    if (!fh->accum_initialized) {
        fh->accumulated_depth = (float*)safe_calloc((size_t)width * (size_t)height, sizeof(float));
        fh->accumulated_weight = (float*)safe_calloc((size_t)width * (size_t)height, sizeof(float));
        if (!fh->accumulated_depth || !fh->accumulated_weight) {
            safe_free((void**)&fh->accumulated_depth);
            safe_free((void**)&fh->accumulated_weight);
            return -1;
        }
        fh->accum_width = width;
        fh->accum_height = height;
        fh->accum_initialized = 1;
    }

    if (width != fh->accum_width || height != fh->accum_height) return -1;

    for (int i = 0; i < (size_t)width * (size_t)height; i++) {
        float w = confidence ? confidence[i] : 0.5f;
        if (w > fh->config.confidence_threshold && depth_map[i] > 0.001f) {
            fh->accumulated_depth[i] += depth_map[i] * w;
            fh->accumulated_weight[i] += w;
        }
    }

    if (fh->config.enable_temporal_fusion) {
        int idx = fh->temporal_idx;
        if (fh->temporal_buffer[idx]) {
            if (fh->temporal_buffer[idx] &&
                fh->temporal_conf_buffer[idx]) {
                memcpy(fh->temporal_buffer[idx], depth_map, (size_t)width * (size_t)height * sizeof(float));
                if (confidence) {
                    memcpy(fh->temporal_conf_buffer[idx], confidence, (size_t)width * (size_t)height * sizeof(float));
                }
            }
        } else {
            fh->temporal_buffer[idx] = (float*)safe_malloc((size_t)width * (size_t)height * sizeof(float));
            fh->temporal_conf_buffer[idx] = (float*)safe_malloc((size_t)width * (size_t)height * sizeof(float));
            if (fh->temporal_buffer[idx] && fh->temporal_conf_buffer[idx]) {
                memcpy(fh->temporal_buffer[idx], depth_map, (size_t)width * (size_t)height * sizeof(float));
                if (confidence) {
                    memcpy(fh->temporal_conf_buffer[idx], confidence, (size_t)width * (size_t)height * sizeof(float));
                } else {
                    memset(fh->temporal_conf_buffer[idx], 0, (size_t)width * (size_t)height * sizeof(float));
                }
            }
        }
        fh->temporal_idx = (idx + 1) % SDE_TEMPORAL_BUFFER_SIZE;
        if (fh->temporal_count < SDE_TEMPORAL_BUFFER_SIZE) fh->temporal_count++;
    }

    (void)pose_matrix;
    return 0;
}

int sde_fusion_fuse(SDEFusionHandler* fh,
                     float* fused_depth_out, float* fused_confidence_out)
{
    if (!fh || !fused_depth_out) return -1;
    if (!fh->accum_initialized) return -1;

    int total = fh->accum_width * fh->accum_height;

    for (int i = 0; i < total; i++) {
        float w = fh->accumulated_weight[i];
        if (w > 1e-10f) {
            fused_depth_out[i] = fh->accumulated_depth[i] / w;
            if (fused_confidence_out) {
                fused_confidence_out[i] = SDE_CLAMP(w / 3.0f, 0.0f, 1.0f);
            }
        } else {
            fused_depth_out[i] = 0.0f;
            if (fused_confidence_out) fused_confidence_out[i] = 0.0f;
        }
    }

    if (fh->config.enable_hole_filling) {
        sde_fusion_hole_fill(fused_depth_out, NULL,
                              fh->accum_width, fh->accum_height,
                              fh->config.hole_filling_radius,
                              fused_depth_out);
    }

    memset(fh->accumulated_depth, 0, total * sizeof(float));
    memset(fh->accumulated_weight, 0, total * sizeof(float));
    fh->accum_initialized = 0;

    return 0;
}

int sde_fusion_fuse_multi_view(SDEFusionHandler* fh,
                                 const float** depth_maps, const float** confidence_maps,
                                 int num_views, int width, int height,
                                 float* fused_out, float* confidence_out)
{
    if (!fh || !depth_maps || !fused_out || num_views <= 0 ||
        num_views > SDE_MAX_FUSION_VIEWS || width <= 0 || height <= 0) return -1;

    memset(fused_out, 0, (size_t)width * (size_t)height * sizeof(float));
    float* weight_sum = (float*)safe_calloc((size_t)width * (size_t)height, sizeof(float));
    if (!weight_sum) return -1;

    for (int v = 0; v < num_views; v++) {
        if (!depth_maps[v]) continue;
        const float* conf = confidence_maps ? confidence_maps[v] : NULL;

        for (int i = 0; i < (size_t)width * (size_t)height; i++) {
            float d = depth_maps[v][i];
            if (d < 0.001f) continue;

            float w = conf ? conf[i] : 1.0f / (float)num_views;
            if (w < fh->config.confidence_threshold) w = 0.0f;

            fused_out[i] += d * w;
            weight_sum[i] += w;
        }
    }

    for (int i = 0; i < (size_t)width * (size_t)height; i++) {
        if (weight_sum[i] > 1e-10f) {
            fused_out[i] /= weight_sum[i];
            if (confidence_out) {
                confidence_out[i] = SDE_CLAMP(weight_sum[i] / (float)num_views * 2.0f, 0.0f, 1.0f);
            }
        } else {
            fused_out[i] = 0.0f;
            if (confidence_out) confidence_out[i] = 0.0f;
        }
    }

    if (fh->config.enable_hole_filling) {
        sde_fusion_hole_fill(fused_out, NULL, width, height,
                              fh->config.hole_filling_radius, fused_out);
    }

    safe_free((void**)&weight_sum);
    return 0;
}

int sde_fusion_temporal(SDEFusionHandler* fh,
                         const float* current_depth, const float* current_confidence,
                         int width, int height,
                         float* smoothed_out, float* confidence_out)
{
    if (!fh || !current_depth || !smoothed_out || width <= 0 || height <= 0) return -1;

    if ((size_t)width * (size_t)height == 0) return -1;

    if (fh->temporal_count == 0) {
        memcpy(smoothed_out, current_depth, (size_t)width * (size_t)height * sizeof(float));
        if (confidence_out && current_confidence) {
            memcpy(confidence_out, current_confidence, (size_t)width * (size_t)height * sizeof(float));
        } else if (confidence_out) {
            memset(confidence_out, 0, (size_t)width * (size_t)height * sizeof(float));
        }

        int idx = fh->temporal_idx;
        if (!fh->temporal_buffer[idx]) {
            fh->temporal_buffer[idx] = (float*)safe_malloc((size_t)width * (size_t)height * sizeof(float));
        }
        if (fh->temporal_buffer[idx]) {
            memcpy(fh->temporal_buffer[idx], current_depth, (size_t)width * (size_t)height * sizeof(float));
        }
        if (!fh->temporal_conf_buffer[idx]) {
            fh->temporal_conf_buffer[idx] = (float*)safe_malloc((size_t)width * (size_t)height * sizeof(float));
        }
        if (fh->temporal_conf_buffer[idx] && current_confidence) {
            memcpy(fh->temporal_conf_buffer[idx], current_confidence, (size_t)width * (size_t)height * sizeof(float));
        }
        fh->temporal_idx = (fh->temporal_idx + 1) % SDE_TEMPORAL_BUFFER_SIZE;
        fh->temporal_count = 1;
        return 0;
    }

    float* accum = (float*)safe_calloc((size_t)width * (size_t)height, sizeof(float));
    float* weights = (float*)safe_calloc((size_t)width * (size_t)height, sizeof(float));
    if (!accum || !weights) {
        safe_free((void**)&accum);
        safe_free((void**)&weights);
        return -1;
    }

    float alpha = fh->config.temporal_smoothing_factor;
    float beta = 1.0f - alpha;

    for (int i = 0; i < (size_t)width * (size_t)height; i++) {
        float cur_w = current_confidence ? current_confidence[i] : 0.5f;
        if (cur_w > fh->config.confidence_threshold && current_depth[i] > 0.001f) {
            accum[i] += current_depth[i] * cur_w * beta;
            weights[i] += cur_w * beta;
        }
    }

    int count = fh->temporal_count < SDE_TEMPORAL_BUFFER_SIZE ?
                fh->temporal_count : SDE_TEMPORAL_BUFFER_SIZE;
    for (int t = 0; t < count; t++) {
        if (!fh->temporal_buffer[t]) continue;
        float decay = powf(alpha, (float)(t + 1));
        for (int i = 0; i < (size_t)width * (size_t)height; i++) {
            float d = fh->temporal_buffer[t][i];
            float c = fh->temporal_conf_buffer[t] ? fh->temporal_conf_buffer[t][i] : 0.3f;
            if (d > 0.001f && c > fh->config.confidence_threshold) {
                accum[i] += d * c * decay;
                weights[i] += c * decay;
            }
        }
    }

    for (int i = 0; i < (size_t)width * (size_t)height; i++) {
        if (weights[i] > 1e-10f) {
            smoothed_out[i] = accum[i] / weights[i];
            if (confidence_out) {
                confidence_out[i] = SDE_CLAMP(weights[i], 0.0f, 1.0f);
            }
        } else {
            smoothed_out[i] = current_depth[i];
            if (confidence_out) confidence_out[i] = 0.0f;
        }
    }

    int idx = fh->temporal_idx;
    if (!fh->temporal_buffer[idx]) {
        fh->temporal_buffer[idx] = (float*)safe_malloc((size_t)width * (size_t)height * sizeof(float));
    }
    if (fh->temporal_buffer[idx]) {
        memcpy(fh->temporal_buffer[idx], smoothed_out, (size_t)width * (size_t)height * sizeof(float));
    }
    if (!fh->temporal_conf_buffer[idx]) {
        fh->temporal_conf_buffer[idx] = (float*)safe_malloc((size_t)width * (size_t)height * sizeof(float));
    }
    if (fh->temporal_conf_buffer[idx]) {
        if (confidence_out) {
            memcpy(fh->temporal_conf_buffer[idx], confidence_out, (size_t)width * (size_t)height * sizeof(float));
        } else if (current_confidence) {
            memcpy(fh->temporal_conf_buffer[idx], current_confidence, (size_t)width * (size_t)height * sizeof(float));
        }
    }

    fh->temporal_idx = (idx + 1) % SDE_TEMPORAL_BUFFER_SIZE;
    if (fh->temporal_count < SDE_TEMPORAL_BUFFER_SIZE) fh->temporal_count++;

    safe_free((void**)&accum);
    safe_free((void**)&weights);
    return 0;
}

int sde_fusion_hole_fill(const float* depth_map, const float* confidence,
                          int width, int height, int radius,
                          float* filled_out)
{
    if (!depth_map || !filled_out || width <= 0 || height <= 0 || radius <= 0) return -1;

    memcpy(filled_out, depth_map, (size_t)width * (size_t)height * sizeof(float));
    float* temp = (float*)safe_malloc((size_t)width * (size_t)height * sizeof(float));
    if (!temp) return -1;

    for (int iter = 0; iter < 3; iter++) {
        memcpy(temp, filled_out, (size_t)width * (size_t)height * sizeof(float));
        int changed = 0;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (temp[y * width + x] > 0.001f) continue;

                float sum_d = 0.0f;
                float sum_w = 0.0f;

                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dx = -radius; dx <= radius; dx++) {
                        int px = x + dx;
                        int py = y + dy;
                        if (px < 0 || px >= width || py < 0 || py >= height) continue;
                        float d = temp[py * width + px];
                        if (d < 0.001f) continue;

                        float dist = sqrtf((float)(dx * dx + dy * dy));
                        float w = 1.0f / (dist + 1.0f);
                        if (confidence) {
                            w *= confidence[py * width + px];
                        }
                        sum_d += d * w;
                        sum_w += w;
                    }
                }

                if (sum_w > 1e-10f) {
                    filled_out[y * width + x] = sum_d / sum_w;
                    changed++;
                }
            }
        }

        if (changed == 0) break;
    }

    safe_free((void**)&temp);
    return 0;
}

int sde_fusion_edge_preserve_filter(const float* depth_map, const float* guide_image,
                                      int width, int height,
                                      float sigma_depth, float sigma_image,
                                      int kernel_size,
                                      float* filtered_out)
{
    if (!depth_map || !filtered_out || width <= 0 || height <= 0) return -1;

    _bilateral_filter_2d((float*)depth_map, width, height,
                          kernel_size, sigma_image, sigma_depth,
                          (float*)guide_image, filtered_out);
    return 0;
}

int sde_save_disparity(const float* disparity, int width, int height,
                        const char* filepath)
{
    if (!disparity || !filepath || width <= 0 || height <= 0) return -1;
    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;
    int header[3] = {width, height, 1};
    fwrite(header, sizeof(int), 3, fp);
    fwrite(disparity, sizeof(float), (size_t)width * height, fp);
    fclose(fp);
    return 0;
}

int sde_save_depth(const float* depth, int width, int height,
                    const char* filepath)
{
    if (!depth || !filepath || width <= 0 || height <= 0) return -1;
    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;
    int header[3] = {width, height, 2};
    fwrite(header, sizeof(int), 3, fp);
    fwrite(depth, sizeof(float), (size_t)width * height, fp);
    fclose(fp);
    return 0;
}

int sde_load_disparity(float* disparity, int width, int height,
                        const char* filepath)
{
    if (!disparity || !filepath || width <= 0 || height <= 0) return -1;
    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;
    int header[3];
    if (fread(header, sizeof(int), 3, fp) != 3) { fclose(fp); return -1; }
    fread(disparity, sizeof(float), (size_t)width * height, fp);
    fclose(fp);
    return 0;
}

int sde_load_depth(float* depth, int width, int height,
                    const char* filepath)
{
    if (!depth || !filepath || width <= 0 || height <= 0) return -1;
    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;
    int header[3];
    if (fread(header, sizeof(int), 3, fp) != 3) { fclose(fp); return -1; }
    fread(depth, sizeof(float), (size_t)width * height, fp);
    fclose(fp);
    return 0;
}
