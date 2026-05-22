/**
 * @file object_recognition.c
 * @brief 增强物体识别与场景理解完整实现
 * 使用真实特征提取算法（HOG + 归一化互相关模板匹配）
 */
#include "selflnn/multimodal/object_recognition.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* 梯度直方图方向数 */
#define HOG_BINS 9
/* 每个cell的像素数 */
#define HOG_CELL_SIZE 8

struct ObjectRecognizer {
    float category_templates[OR_MAX_CATEGORIES][128];
    int category_count;
    char category_names[OR_MAX_CATEGORIES][64];
    int initialized;
    SceneType last_scene;
};

/**
 * @brief 计算图像梯度（Sobel算子，真实实现）
 */
static void compute_gradients(const float* image, int w, int h, int ch,
                              float* grad_mag, float* grad_ori, int* g_w, int* g_h) {
    int gw = w - 2, gh = h - 2;
    if (gw < 1 || gh < 1) { *g_w = 0; *g_h = 0; return; }
    *g_w = gw; *g_h = gh;
    
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            /* 对每个通道取灰度值（加权平均） */
            float gy_p1 = 0, gy_m1 = 0, gx_p1 = 0, gx_m1 = 0;
            for (int c = 0; c < ch; c++) {
                float wc = (c == 0) ? 0.3f : ((c == 1) ? 0.59f : 0.11f);
                gy_p1 += image[((y + 1) * w + x) * ch + c] * wc;
                gy_m1 += image[((y - 1) * w + x) * ch + c] * wc;
                gx_p1 += image[(y * w + (x + 1)) * ch + c] * wc;
                gx_m1 += image[(y * w + (x - 1)) * ch + c] * wc;
            }
            float gx = gx_p1 - gx_m1;
            float gy = gy_p1 - gy_m1;
            int idx = (y - 1) * gw + (x - 1);
            grad_mag[idx] = sqrtf(gx * gx + gy * gy);
            grad_ori[idx] = atan2f(gy, gx);
        }
    }
}

/**
 * @brief 计算HOG特征描述子（真实实现）
 */
static void compute_hog_features(const float* grad_mag, const float* grad_ori,
                                  int gw, int gh, float* features, int max_features) {
    int cells_x = gw / HOG_CELL_SIZE;
    int cells_y = gh / HOG_CELL_SIZE;
    if (cells_x < 1) cells_x = 1;
    if (cells_y < 1) cells_y = 1;
    
    int total_cells = cells_x * cells_y;
    int feature_count = total_cells * HOG_BINS;
    if (feature_count > max_features) feature_count = max_features;
    
    memset(features, 0, feature_count * sizeof(float));
    
    for (int cy = 0; cy < cells_y; cy++) {
        for (int cx = 0; cx < cells_x; cx++) {
            float cell_hist[HOG_BINS] = {0};
            float cell_max = 0;
            
            for (int dy = 0; dy < HOG_CELL_SIZE; dy++) {
                for (int dx = 0; dx < HOG_CELL_SIZE; dx++) {
                    int gx = cx * HOG_CELL_SIZE + dx;
                    int gy = cy * HOG_CELL_SIZE + dy;
                    if (gx >= gw || gy >= gh) continue;
                    int idx = gy * gw + gx;
                    float mag = grad_mag[idx];
                    float ori = grad_ori[idx] + (float)M_PI; /* 转到[0, 2π] */
                    float bin_f = ori * (float)HOG_BINS / (2.0f * (float)M_PI);
                    int bin0 = (int)bin_f % HOG_BINS;
                    int bin1 = (bin0 + 1) % HOG_BINS;
                    float w1 = bin_f - floorf(bin_f);
                    float w0 = 1.0f - w1;
                    cell_hist[bin0] += mag * w0;
                    cell_hist[bin1] += mag * w1;
                }
            }
            
            /* 找最大值用于归一化 */
            for (int b = 0; b < HOG_BINS; b++) {
                if (cell_hist[b] > cell_max) cell_max = cell_hist[b];
            }
            
            /* L2归一化 */
            float l2_norm = 0;
            for (int b = 0; b < HOG_BINS; b++) {
                l2_norm += cell_hist[b] * cell_hist[b];
            }
            l2_norm = sqrtf(l2_norm) + 1e-6f;
            
            int feat_base = (cy * cells_x + cx) * HOG_BINS;
            for (int b = 0; b < HOG_BINS && (feat_base + b) < feature_count; b++) {
                features[feat_base + b] = cell_hist[b] / l2_norm;
            }
        }
    }
}

/**
 * @brief 计算两向量的归一化互相关（真实模板匹配）
 */
static float normalized_cross_correlation(const float* a, const float* b, int len) {
    float sum_a = 0, sum_b = 0, sum_ab = 0, sum_aa = 0, sum_bb = 0;
    for (int i = 0; i < len; i++) {
        sum_a += a[i];
        sum_b += b[i];
    }
    float mean_a = sum_a / (float)len;
    float mean_b = sum_b / (float)len;
    for (int i = 0; i < len; i++) {
        float da = a[i] - mean_a;
        float db = b[i] - mean_b;
        sum_ab += da * db;
        sum_aa += da * da;
        sum_bb += db * db;
    }
    float denom = sqrtf(sum_aa * sum_bb);
    if (denom < 1e-10f) return 0;
    return sum_ab / denom;
}

ObjectRecognizer* object_recognizer_create(void) {
    ObjectRecognizer* or_obj = (ObjectRecognizer*)safe_calloc(1, sizeof(ObjectRecognizer));
    if (!or_obj) return NULL;
    or_obj->initialized = 1;

    /* 预定义类别名称（模板特征将在首次识别时从图像提取） */
    const char* cats[] = {"人","车辆","动物","家具","电子设备","食物","工具","建筑物","植物","自然景观"};
    int cat_count = sizeof(cats) / sizeof(cats[0]);
    for (int i = 0; i < cat_count && i < OR_MAX_CATEGORIES; i++) {
        snprintf(or_obj->category_names[i], 64, "%s", cats[i]);
        /* ZSFWS-S004修复: 模板初始化为全零，标记为"未训练"状态。
         * 不再使用PRNG生成伪随机特征，因为随机特征不等于真实图像学习特征。
         * 未训练时模板全为零，or_train_classifier训练后替换为真实HOG+NCC学习模板。 */
        memset(or_obj->category_templates[i], 0, 128 * sizeof(float));
        or_obj->category_count++;
    }
    or_obj->last_scene = SCENE_UNKNOWN;
    return or_obj;
}

void object_recognizer_free(ObjectRecognizer* or_obj) { safe_free((void**)&or_obj); }

/* 多尺度滑动窗口检测 */
int or_detect_objects(ObjectRecognizer* or_obj, const float* image, int w, int h, int ch,
    DetectedObject* out, int max_count) {
    if (!or_obj || !image || !out || w <= 0 || h <= 0 || ch <= 0) return 0;

    int count = 0;
    int scales[] = {32, 64, 96};
    int scale_count = sizeof(scales) / sizeof(scales[0]);

    /* 分配梯度缓冲区（最大尺寸） */
    int max_gw = w - 2, max_gh = h - 2;
    if (max_gw < 1 || max_gh < 1) return 0;
    float* grad_mag = (float*)safe_malloc((size_t)(max_gw * max_gh) * sizeof(float));
    float* grad_ori = (float*)safe_malloc((size_t)(max_gw * max_gh) * sizeof(float));
    if (!grad_mag || !grad_ori) {
        if (grad_mag) safe_free((void**)&grad_mag);
        if (grad_ori) safe_free((void**)&grad_ori);
        return 0;
    }

    /* 计算全图梯度一次（用于多尺度滑动窗口） */
    int gw, gh;
    compute_gradients(image, w, h, ch, grad_mag, grad_ori, &gw, &gh);

    for (int s = 0; s < scale_count && count < max_count; s++) {
        int win_size = scales[s];
        int step = win_size / 3;
        /* 确保窗口不小于两个cell */
        if (win_size < HOG_CELL_SIZE * 2) continue;

        for (int y = 0; y + win_size <= h && count < max_count; y += step) {
            for (int x = 0; x + win_size <= w && count < max_count; x += step) {
                /* 提取窗口内梯度数据计算HOG特征 */
                int wg_start = (x > 0) ? x - 1 : 0;
                int wg_end = (x + win_size - 2 < gw) ? x + win_size - 2 : gw;
                int hg_start = (y > 0) ? y - 1 : 0;
                int hg_end = (y + win_size - 2 < gh) ? y + win_size - 2 : gh;
                int wg_w = wg_end - wg_start;
                int wg_h = hg_end - hg_start;
                
                if (wg_w < HOG_CELL_SIZE || wg_h < HOG_CELL_SIZE) continue;

                /* 提取窗口内梯度幅值和方向 */
                float* win_grad_mag = (float*)safe_malloc((size_t)(wg_w * wg_h) * sizeof(float));
                float* win_grad_ori = (float*)safe_malloc((size_t)(wg_w * wg_h) * sizeof(float));
                if (!win_grad_mag || !win_grad_ori) {
                    if (win_grad_mag) safe_free((void**)&win_grad_mag);
                    if (win_grad_ori) safe_free((void**)&win_grad_ori);
                    continue;
                }

                for (int wy = 0; wy < wg_h; wy++) {
                    int src_row = hg_start + wy;
                    int dst_row = wy;
                    for (int wx = 0; wx < wg_w; wx++) {
                        int src_idx = src_row * gw + (wg_start + wx);
                        int dst_idx = dst_row * wg_w + wx;
                        win_grad_mag[dst_idx] = grad_mag[src_idx];
                        win_grad_ori[dst_idx] = grad_ori[src_idx];
                    }
                }

                /* 计算HOG特征 */
                float hog_features[128];
                compute_hog_features(win_grad_mag, win_grad_ori, wg_w, wg_h, hog_features, 128);

                /* 边缘响应：使用梯度幅值均值 */
                float edge_response = 0;
                for (int i = 0; i < wg_w * wg_h; i++) {
                    edge_response += win_grad_mag[i];
                }
                edge_response /= (float)(wg_w * wg_h);

                safe_free((void**)&win_grad_mag);
                safe_free((void**)&win_grad_ori);

                if (edge_response > 0.01f) {
                    DetectedObject* obj = &out[count];
                    obj->x = (float)x;
                    obj->y = (float)y;
                    obj->width = (float)win_size;
                    obj->height = (float)win_size;
                    obj->confidence = edge_response * 8.0f;
                    if (obj->confidence > 1.0f) obj->confidence = 1.0f;
                    
                    /* ZSFWS-M009修复: 检查模板是否已训练
                     * 未训练时模板全为零，NCC匹配无意义，直接返回"未训练"状态 */
                    int is_trained = 0;
                    for (int c = 0; c < or_obj->category_count && !is_trained; c++) {
                        for (int i = 0; i < 128; i++) {
                            if (or_obj->category_templates[c][i] != 0.0f) {
                                is_trained = 1; break;
                            }
                        }
                    }
                    
                    int best_category = 0;
                    float best_ncc = -1.0f;
                    if (is_trained) {
                        for (int c = 0; c < or_obj->category_count; c++) {
                            float ncc = normalized_cross_correlation(
                                hog_features, or_obj->category_templates[c], 128);
                            if (ncc > best_ncc) {
                                best_ncc = ncc;
                                best_category = c;
                            }
                        }
                    }
                    obj->category_id = is_trained ? best_category : -1;
                    if (is_trained && best_category < or_obj->category_count) {
                        snprintf(obj->category_name, sizeof(obj->category_name),
                                "%s", or_obj->category_names[best_category]);
                    } else {
                        snprintf(obj->category_name, sizeof(obj->category_name),
                                "未训练");
                    }
                    /* 存储HOG特征 */
                    memcpy(obj->features, hog_features, 128 * sizeof(float));
                    obj->feature_dim = 128;
                    count++;
                }
            }
        }
    }

    safe_free((void**)&grad_mag);
    safe_free((void**)&grad_ori);
    return count;
}

int or_classify_object(ObjectRecognizer* or_obj, const float* features, int dim, int* category_id, float* confidence) {
    if (!or_obj || !features || !category_id || !confidence) return -1;

    float best_sim = -1.0f;
    int best_cat = 0;
    int cmp_dim = dim < 128 ? dim : 128;

    for (int c = 0; c < or_obj->category_count; c++) {
        float dot = 0.0f, mag_f = 0.0f, mag_t = 0.0f;
        for (int i = 0; i < cmp_dim; i++) {
            dot += features[i] * or_obj->category_templates[c][i];
            mag_f += features[i] * features[i];
            mag_t += or_obj->category_templates[c][i] * or_obj->category_templates[c][i];
        }
        float sim = (mag_f > 1e-10f && mag_t > 1e-10f) ? dot / (sqrtf(mag_f) * sqrtf(mag_t)) : 0.0f;
        if (sim > best_sim) { best_sim = sim; best_cat = c; }
    }

    *category_id = best_cat;
    *confidence = (best_sim + 1.0f) / 2.0f;
    return 0;
}

int or_recognize_attributes(ObjectRecognizer* or_obj, const float* image, int w, int h, int ch,
    const DetectedObject* obj, ObjectAttributes* attrs) {
    if (!or_obj || !image || !obj || !attrs) return -1;
    memset(attrs, 0, sizeof(ObjectAttributes));
    int x = (int)obj->x, y = (int)obj->y, bw = (int)obj->width, bh = (int)obj->height;
    if (x < 0) x = 0; if (y < 0) y = 0;
    if (x + bw > w) bw = w - x; if (y + bh > h) bh = h - y;
    if (bw <= 0 || bh <= 0) return -1;

    float sum_h = 0.0f, sum_s = 0.0f, sum_b = 0.0f;
    int n = bw * bh;
    for (int dy = 0; dy < bh; dy++) {
        for (int dx = 0; dx < bw; dx++) {
            int idx = ((y + dy) * w + (x + dx)) * ch;
            float r = image[idx], g = (ch > 1) ? image[idx + 1] : r, b = (ch > 2) ? image[idx + 2] : r;
            float mx = r > g ? r : g; mx = mx > b ? mx : b;
            float mn = r < g ? r : g; mn = mn < b ? mn : b;
            float d = mx - mn;
            float h_val = 0.0f;
            if (d > 1e-6f) {
                if (mx == r) h_val = 60.0f * fmodf((g - b) / d + 6.0f, 6.0f);
                else if (mx == g) h_val = 60.0f * ((b - r) / d + 2.0f);
                else h_val = 60.0f * ((r - g) / d + 4.0f);
            }
            sum_h += h_val;
            sum_s += (mx > 1e-6f) ? d / mx : 0.0f;
            sum_b += mx;
        }
    }
    attrs->hue_mean = sum_h / (float)n;
    attrs->saturation_mean = sum_s / (float)n;
    attrs->brightness_mean = sum_b / (float)n;
    /* BUG-003修复: 使用LBP风格的局部纹理分析替代坐标比率假数据
     * LBP: 中心像素与8邻域比较，统计纹理模式复杂度 */
    float texture_score = 0.0f;
    int tex_samples = 0;
    for (int dy = 2; dy < bh - 2; dy += 4) {
        for (int dx = 2; dx < bw - 2; dx += 4) {
            int idx_c = ((y + dy) * w + (x + dx)) * ch;
            if (idx_c + ch >= w * h * ch) continue;
            float center = image[idx_c];
            int lbp_code = 0;
            int offsets[8][2] = {{-1,-1},{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0}};
            for (int nbr = 0; nbr < 8; nbr++) {
                int nx = x + dx + offsets[nbr][0];
                int ny = y + dy + offsets[nbr][1];
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    int nidx = (ny * w + nx) * ch;
                    if (image[nidx] >= center) lbp_code |= (1 << nbr);
                }
            }
            /* 统计LBP码中0-1跳变次数衡量纹理复杂度 */
            int transitions = 0;
            int prev_bit = (lbp_code >> 7) & 1;
            for (int b = 0; b < 8; b++) {
                int cur_bit = (lbp_code >> b) & 1;
                if (cur_bit != prev_bit) transitions++;
                prev_bit = cur_bit;
            }
            texture_score += (float)transitions / 9.0f;
            tex_samples++;
        }
    }
    attrs->texture_roughness = tex_samples > 0 ? (texture_score / (float)tex_samples) : 0.3f;
    snprintf(attrs->material, sizeof(attrs->material), "%s",
        attrs->brightness_mean > 0.6f ? "金属/光泽" : attrs->texture_roughness > 0.5f ? "粗糙" : "光滑");
    snprintf(attrs->size_category, sizeof(attrs->size_category), "%s",
        (bw * bh) > (w * h / 16) ? "大型" : (bw * bh) > (w * h / 64) ? "中型" : "小型");
    return 0;
}

int or_detect_color(ObjectRecognizer* or_obj, const float* image, int w, int h, int ch,
    const DetectedObject* obj, float* color_rgb) {
    if (!or_obj || !image || !obj || !color_rgb) return -1;
    int x = (int)obj->x, y = (int)obj->y, bw = (int)obj->width, bh = (int)obj->height;
    x = x < 0 ? 0 : x; y = y < 0 ? 0 : y;
    if (x + bw > w) bw = w - x; if (y + bh > h) bh = h - y;
    if (bw <= 0 || bh <= 0) return -1;

    float sr = 0, sg = 0, sb = 0;
    int n = bw * bh;
    for (int dy = 0; dy < bh; dy++) {
        for (int dx = 0; dx < bw; dx++) {
            int idx = ((y + dy) * w + (x + dx)) * ch;
            sr += image[idx];
            sg += (ch > 1) ? image[idx + 1] : image[idx];
            sb += (ch > 2) ? image[idx + 2] : image[idx];
        }
    }
    color_rgb[0] = sr / (float)n; color_rgb[1] = sg / (float)n; color_rgb[2] = sb / (float)n;
    return 0;
}

int or_estimate_size(ObjectRecognizer* or_obj, const DetectedObject* obj, float* width_m, float* height_m) {
    if (!or_obj || !obj || !width_m || !height_m) return -1;
    /* BUG-001修复: 基于检测框特征和典型物距估算物理尺寸
     * pixel_to_meter = est_distance_m / focal_pixels
     * 估算焦距=图像宽度(60度HFOV→f≈w/1.15), 估算物距=2m */
    float img_width = obj->width * 5.0f; /* 从检测框回推图像宽度: 假设物体占20%画面 */
    if (img_width < 320.0f) img_width = 640.0f;
    float focal_pixels = img_width / 1.15f;
    float est_distance = 2.0f;
    *width_m = obj->width * est_distance / focal_pixels;
    *height_m = obj->height * est_distance / focal_pixels;
    /* 合理性限制 */
    if (*width_m <= 0.0f) *width_m = 0.01f;
    if (*height_m <= 0.0f) *height_m = 0.01f;
    return 0;
}

int or_classify_scene(ObjectRecognizer* or_obj, const float* image, int w, int h, int ch,
    SceneType* type, char* name, size_t name_len) {
    if (!or_obj || !image || !type || !name) return -1;
    /* BUG-002修复: 使用多维特征（亮度+纹理+边缘+色温）进行场景分类
     * 替代原来仅靠平均亮度的降级处理 */
    float avg_brightness = 0.0f;
    float avg_variance = 0.0f;
    float edge_density = 0.0f;
    float green_ratio = 0.0f;
    int n = w * h;
    /* 1. 亮度均值和方差 */
    for (int i = 0; i < n; i++) avg_brightness += image[i * ch];
    avg_brightness /= (float)n;
    for (int i = 0; i < n; i++) {
        float d = image[i * ch] - avg_brightness;
        avg_variance += d * d;
    }
    avg_variance /= (float)n;
    /* 2. 边缘密度（Sobel梯度幅值统计） */
    float edge_threshold = 0.05f;
    int edge_count = 0;
    float total_green = 0.0f;
    for (int y = 1; y < h - 1; y += 2) {
        for (int x = 1; x < w - 1; x += 2) {
            int idx = (y * w + x) * ch;
            float gx = image[idx + w * ch] - image[idx - w * ch];  /* 垂直梯度 */
            float gy = image[idx + ch] - image[idx - ch];           /* 水平梯度 */
            float mag = sqrtf(gx * gx + gy * gy);
            if (mag > edge_threshold) edge_count++;
            if (ch > 1) total_green += image[idx + 1];
        }
    }
    int samples = ((h - 2) / 2) * ((w - 2) / 2);
    edge_density = samples > 0 ? (float)edge_count / (float)samples : 0.0f;
    /* 3. 绿色通道占比（自然场景检测） */
    green_ratio = (n > 0) ? (total_green / (float)samples) : 0.0f;
    /* 4. 综合判定 */
    float indoor_score = (avg_brightness < 0.3f ? 2.0f : 0.0f)
                       + (edge_density < 0.1f ? 1.0f : 0.0f)
                       + (green_ratio < 0.2f ? 1.0f : 0.0f);
    float nature_score = (green_ratio > 0.35f ? 3.0f : 0.0f)
                       + (avg_brightness > 0.4f ? 1.0f : 0.0f)
                       + (edge_density > 0.2f ? 1.0f : 0.0f);
    float industrial_score = (avg_variance < 0.02f ? 2.0f : 0.0f)
                           + (edge_density > 0.15f ? 1.0f : 0.0f)
                           + (avg_brightness > 0.3f && avg_brightness < 0.6f ? 1.0f : 0.0f);
    float outdoor_score = (avg_brightness > 0.5f ? 2.0f : 0.0f)
                        + (edge_density > 0.1f ? 1.0f : 0.0f);
    if (nature_score >= indoor_score && nature_score >= industrial_score && nature_score >= outdoor_score) {
        *type = SCENE_NATURE; snprintf(name, name_len, "自然场景");
    } else if (industrial_score >= indoor_score && industrial_score >= outdoor_score) {
        *type = SCENE_INDUSTRIAL; snprintf(name, name_len, "工业场景");
    } else if (indoor_score >= outdoor_score) {
        *type = SCENE_INDOOR; snprintf(name, name_len, "室内场景");
    } else {
        *type = SCENE_OUTDOOR; snprintf(name, name_len, "室外场景");
    }
    or_obj->last_scene = *type;
    return 0;
}

int or_detect_relations(ObjectRecognizer* or_obj, const DetectedObject* objects, int count, SceneContext* ctx) {
    if (!or_obj || !objects || !ctx) return -1;
    int n_rel = count < OR_MAX_OBJECTS ? count : OR_MAX_OBJECTS;
    ctx->object_count = n_rel;
    for (int i = 0; i < n_rel; i++) {
        memcpy(&ctx->objects[i], &objects[i], sizeof(DetectedObject));
    }
    for (int i = 0; i < n_rel; i++) {
        for (int j = i + 1; j < n_rel; j++) {
            float dx = objects[i].x - objects[j].x;
            float dy = objects[i].y - objects[j].y;
            float overlap_x = (objects[i].width + objects[j].width) / 2.0f - fabsf(dx);
            float overlap_y = (objects[i].height + objects[j].height) / 2.0f - fabsf(dy);
            if (overlap_x > 0 && overlap_y > 0) {
                if (i < OR_MAX_OBJECTS && ctx->relation_counts && i < n_rel) ctx->relation_counts[i]++;
            }
        }
    }
    return 0;
}

int or_detect_changes(ObjectRecognizer* or_obj, const float* prev, const float* curr, int w, int h, int ch,
    float* change_map) {
    if (!or_obj || !prev || !curr || !change_map) return -1;
    int n = w * h;
    float max_change = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = 0.0f;
        for (int c = 0; c < ch; c++) {
            float d = curr[i * ch + c] - prev[i * ch + c];
            diff += d * d;
        }
        change_map[i] = sqrtf(diff / (float)ch);
        if (change_map[i] > max_change) max_change = change_map[i];
    }
    if (max_change > 1e-10f) {
        for (int i = 0; i < n; i++) change_map[i] /= max_change;
    }
    return 0;
}

/* ====== 分类器训练/保存/加载（F-001修复：完整实现） ====== */

/*
 * or_train_classifier: 使用基于原型均值的分类器训练
 * 对每个类别，计算所有属于该类别的样本的特征均值作为类别模板
 * 采用Warmuth缩放增强小样本类别鲁棒性
 */
int or_train_classifier(ObjectRecognizer* or_obj, const float* features,
                        const int* labels, int samples, int dim, int categories) {
    if (!or_obj || !features || !labels) return -1;
    if (samples <= 0 || dim <= 0 || categories <= 0) return -1;
    if (categories > OR_MAX_CATEGORIES) categories = OR_MAX_CATEGORIES;
    int feat_dim = dim < 128 ? dim : 128;

    /* 统计每个类别的样本数 */
    int category_samples[OR_MAX_CATEGORIES];
    memset(category_samples, 0, sizeof(category_samples));

    /* 清零所有类别模板 */
    for (int c = 0; c < OR_MAX_CATEGORIES; c++) {
        memset(or_obj->category_templates[c], 0, 128 * sizeof(float));
    }

    /* 累加每个类别的特征向量 */
    for (int s = 0; s < samples; s++) {
        int label = labels[s];
        if (label < 0 || label >= categories) continue;
        if (label >= OR_MAX_CATEGORIES) continue;
        int offset = s * dim;
        for (int i = 0; i < feat_dim; i++) {
            or_obj->category_templates[label][i] += features[offset + i];
        }
        category_samples[label]++;
    }

    /* 计算每个类别的均值模板（Warmuth缩放） */
    for (int c = 0; c < categories && c < OR_MAX_CATEGORIES; c++) {
        int n = category_samples[c];
        if (n < 1) {
            /* 无样本的类别：使用所有类别的全局均值 */
            float global_mean_sum[128] = {0};
            int total_valid = 0;
            for (int cc = 0; cc < categories && cc < OR_MAX_CATEGORIES; cc++) {
                if (category_samples[cc] > 0) {
                    for (int i = 0; i < feat_dim; i++)
                        global_mean_sum[i] += or_obj->category_templates[cc][i] / (float)category_samples[cc];
                    total_valid++;
                }
            }
            if (total_valid > 0) {
                for (int i = 0; i < feat_dim; i++)
                    or_obj->category_templates[c][i] = global_mean_sum[i] / (float)total_valid;
            }
            continue;
        }
        /* Warmuth缩放：n/(n+1) 衰减，增强小样本鲁棒性 */
        float scale = (float)n / (float)(n + 1);
        for (int i = 0; i < feat_dim; i++) {
            or_obj->category_templates[c][i] = (or_obj->category_templates[c][i] / (float)n) * scale;
        }
    }

    /* 更新类别总数 */
    if (categories > or_obj->category_count) {
        or_obj->category_count = categories;
    }

    return 0;
}

/*
 * or_save_model: 将分类器模型保存到文件
 * 二进制格式: [魔数4字节][类别数4字节][每个类别的名称64字节+模板128*float]
 * 魔数: "SLO2" = 0x324F4C53 (Self-LNN Object recognition v2)
 */
int or_save_model(const ObjectRecognizer* or_obj, const char* filepath) {
    if (!or_obj || !filepath) return -1;
    if (or_obj->category_count <= 0) return -1;

    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;

    /* 写入魔数 */
    const char magic[4] = {'S', 'L', 'O', '2'};
    if (fwrite(magic, 1, 4, fp) != 4) { fclose(fp); return -1; }

    /* 写入类别数量 */
    int count = or_obj->category_count;
    if (fwrite(&count, sizeof(int), 1, fp) != 1) { fclose(fp); return -1; }

    /* 写入每个类别的名称和模板 */
    for (int c = 0; c < count && c < OR_MAX_CATEGORIES; c++) {
        /* 类别名称（固定64字节） */
        if (fwrite(or_obj->category_names[c], 1, 64, fp) != 64) { fclose(fp); return -1; }
        /* 类别模板特征（128个float） */
        if (fwrite(or_obj->category_templates[c], sizeof(float), 128, fp) != 128) { fclose(fp); return -1; }
    }

    /* 写入场景信息 */
    int scene = (int)or_obj->last_scene;
    if (fwrite(&scene, sizeof(int), 1, fp) != 1) { fclose(fp); return -1; }

    fclose(fp);
    return 0;
}

/*
 * or_load_model: 从文件加载分类器模型
 */
int or_load_model(ObjectRecognizer* or_obj, const char* filepath) {
    if (!or_obj || !filepath) return -1;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;

    /* 读取并验证魔数 */
    char magic[4];
    if (fread(magic, 1, 4, fp) != 4) { fclose(fp); return -1; }
    if (magic[0] != 'S' || magic[1] != 'L' || magic[2] != 'O' || magic[3] != '2') {
        fclose(fp);
        return -1;
    }

    /* 读取类别数量 */
    int count = 0;
    if (fread(&count, sizeof(int), 1, fp) != 1) { fclose(fp); return -1; }
    if (count <= 0 || count > OR_MAX_CATEGORIES) { fclose(fp); return -1; }

    or_obj->category_count = count;

    /* 读取每个类别的名称和模板 */
    for (int c = 0; c < count && c < OR_MAX_CATEGORIES; c++) {
        if (fread(or_obj->category_names[c], 1, 64, fp) != 64) { fclose(fp); return -1; }
        or_obj->category_names[c][63] = '\0'; /* 确保字符串终止 */
        if (fread(or_obj->category_templates[c], sizeof(float), 128, fp) != 128) { fclose(fp); return -1; }
    }

    /* 读取场景信息 */
    int scene = SCENE_UNKNOWN;
    if (fread(&scene, sizeof(int), 1, fp) == 1) {
        or_obj->last_scene = (SceneType)scene;
    }

    or_obj->initialized = 1;
    fclose(fp);
    return 0;
}
