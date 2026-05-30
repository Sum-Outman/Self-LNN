/**
 * @file text_detection.c
 * @brief 文字区域检测和定位实现
 * 
 * 实现图像中文字区域的检测和定位功能。
 * 100%纯C实现，不依赖任何第三方库。
 */

#include "selflnn/multimodal/text_detection.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/perf.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/logging.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ==================== 内部结构体定义 ==================== */

/**
 * @brief 文本检测处理器内部结构体
 */
struct TextDetectionProcessor {
    TextDetectionConfig config;      /**< 检测配置 */
    float* gray_buffer;              /**< 灰度图像缓冲区 */
    float* edge_buffer;              /**< 边缘图像缓冲区 */
    float* binary_buffer;            /**< 二值化缓冲区 */
    int* label_buffer;               /**< 标签缓冲区 */
    size_t buffer_size;              /**< 缓冲区大小 */
};

/**
 * @brief 边缘检测结果
 */
typedef struct {
    float* magnitude;    /**< 梯度幅值 */
    float* direction;    /**< 梯度方向 */
    int width;          /**< 图像宽度 */
    int height;         /**< 图像高度 */
} EdgeDetectionResult;

/**
 * @brief 连通域信息
 */
typedef struct {
    int label;          /**< 标签 */
    int x_min;          /**< 最小x坐标 */
    int x_max;          /**< 最大x坐标 */
    int y_min;          /**< 最小y坐标 */
    int y_max;          /**< 最大y坐标 */
    int area;           /**< 面积 */
    float centroid_x;   /**< 质心x坐标 */
    float centroid_y;   /**< 质心y坐标 */
    float aspect_ratio; /**< 宽高比 */
    float solidity;     /**< 固体度（面积/凸包面积） */
} ConnectedComponent;

/* ==================== 静态函数声明 ==================== */

static int convert_to_grayscale(const float* image, int width, int height, int channels,
                               float* gray);
static int detect_edges_sobel(const float* gray, int width, int height,
                             float* magnitude, float* direction);
static int binarize_edge(const float* magnitude, int width, int height,
                        float threshold, float* binary);
static int find_connected_components(const float* binary, int width, int height,
                                    int* labels, int* num_components);
static int extract_component_info(const int* labels, int width, int height,
                                 int num_components, ConnectedComponent* components);
static int filter_text_candidates(const ConnectedComponent* components, int num_components,
                                 const TextDetectionConfig* config,
                                 TextRegion* regions, int max_regions, int* num_regions);
static float calculate_component_solidity(const ConnectedComponent* comp,
                                         const int* labels, int width, int height);
static void apply_non_maximum_suppression(TextRegion* regions, int num_regions,
                                         float threshold,
                                         TextRegion* result_regions, int max_result,
                                         int* num_result);
static float calculate_region_overlap(const TextRegion* r1, const TextRegion* r2);
static int compare_regions_by_confidence(const void* a, const void* b);

/* ==================== 公开函数实现 ==================== */

TextDetectionProcessor* text_detection_processor_create(const TextDetectionConfig* config) {
    if (!config) {
        return NULL;
    }
    
    TextDetectionProcessor* processor = (TextDetectionProcessor*)safe_calloc(1, sizeof(TextDetectionProcessor));
    if (!processor) {
        return NULL;
    }
    
    // 复制配置
    memcpy(&processor->config, config, sizeof(TextDetectionConfig));
    
    // 初始化缓冲区指针
    processor->gray_buffer = NULL;
    processor->edge_buffer = NULL;
    processor->binary_buffer = NULL;
    processor->label_buffer = NULL;
    processor->buffer_size = 0;
    
    return processor;
}

void text_detection_processor_free(TextDetectionProcessor* processor) {
    if (!processor) {
        return;
    }
    
    safe_free((void**)&processor->gray_buffer);
    safe_free((void**)&processor->edge_buffer);
    safe_free((void**)&processor->binary_buffer);
    safe_free((void**)&processor->label_buffer);
    safe_free((void**)&processor);
}

int text_detection_detect(TextDetectionProcessor* processor,
                         const float* image_data, int width, int height, int channels,
                         TextRegion* regions, int max_regions, int* num_regions) {
    if (!processor || !image_data || !regions || !num_regions) {
        return -1;
    }
    
    if (width <= 0 || height <= 0 || channels <= 0) {
        return -2;
    }
    
    // 确保缓冲区足够大
    size_t image_size = (size_t)width * height;
    size_t edge_buffer_size = image_size * 2; // 幅值和方向
    
    if (processor->buffer_size < image_size * 4) { // 安全边界
        safe_free((void**)&processor->gray_buffer);
        safe_free((void**)&processor->edge_buffer);
        safe_free((void**)&processor->binary_buffer);
        safe_free((void**)&processor->label_buffer);
        
        processor->gray_buffer = (float*)safe_calloc(image_size, sizeof(float));
        processor->edge_buffer = (float*)safe_calloc(edge_buffer_size, sizeof(float));
        processor->binary_buffer = (float*)safe_calloc(image_size, sizeof(float));
        processor->label_buffer = (int*)safe_calloc(image_size, sizeof(int));
        
        if (!processor->gray_buffer || !processor->edge_buffer || 
            !processor->binary_buffer || !processor->label_buffer) {
            safe_free((void**)&processor->gray_buffer);
            safe_free((void**)&processor->edge_buffer);
            safe_free((void**)&processor->binary_buffer);
            safe_free((void**)&processor->label_buffer);
            return -3;
        }
        
        processor->buffer_size = image_size * 4;
    }
    
    // 根据算法类型选择检测方法
    int result = 0;
    *num_regions = 0;
    
    // 转换为灰度图
    result = convert_to_grayscale(image_data, width, height, channels,
                                 processor->gray_buffer);
    if (result != 0) {
        return result;
    }
    
    // 边缘检测
    float* magnitude = processor->edge_buffer;
    float* direction = processor->edge_buffer + image_size;
    
    result = detect_edges_sobel(processor->gray_buffer, width, height,
                               magnitude, direction);
    if (result != 0) {
        return result;
    }
    
    // 二值化边缘图像
    float edge_threshold = 0.1f; // 可配置
    result = binarize_edge(magnitude, width, height, edge_threshold,
                          processor->binary_buffer);
    if (result != 0) {
        return result;
    }
    
    // 查找连通域
    int num_components = 0;
    result = find_connected_components(processor->binary_buffer, width, height,
                                      processor->label_buffer, &num_components);
    if (result != 0) {
        return result;
    }
    
    // 提取连通域信息
    if (num_components > 0) {
        ConnectedComponent* components = (ConnectedComponent*)safe_calloc(num_components, sizeof(ConnectedComponent));
        if (!components) {
            return -4;
        }
        
        result = extract_component_info(processor->label_buffer, width, height,
                                       num_components, components);
        if (result == 0) {
            // 计算每个连通域的固体度
            for (int i = 0; i < num_components; i++) {
                components[i].solidity = calculate_component_solidity(&components[i],
                                                                     processor->label_buffer,
                                                                     width, height);
            }
            
            // 过滤文本候选区域
            result = filter_text_candidates(components, num_components, &processor->config,
                                           regions, max_regions, num_regions);
        }
        
        safe_free((void**)&components);
    }
    
    // 应用非极大值抑制
    if (result == 0 && processor->config.use_nms && *num_regions > 1) {
        TextRegion* nms_result = (TextRegion*)safe_calloc(*num_regions, sizeof(TextRegion));
        if (nms_result) {
            int num_nms = 0;
            apply_non_maximum_suppression(regions, *num_regions,
                                         processor->config.nms_threshold,
                                         nms_result, *num_regions, &num_nms);
            if (num_nms > 0) {
                memcpy(regions, nms_result, num_nms * sizeof(TextRegion));
                *num_regions = num_nms;
            }
            safe_free((void**)&nms_result);
        }
    }
    
    return result;
}

void text_detection_nms(TextRegion* regions, int num_regions, float threshold,
                       TextRegion* result_regions, int max_result, int* num_result) {
    apply_non_maximum_suppression(regions, num_regions, threshold,
                                 result_regions, max_result, num_result);
}

void text_detection_filter(TextRegion* regions, int num_regions,
                          const TextDetectionConfig* config,
                          TextRegion* filtered_regions, int max_filtered,
                          int* num_filtered) {
    if (!regions || !config || !filtered_regions || !num_filtered) {
        return;
    }
    
    int count = 0;
    
    for (int i = 0; i < num_regions && count < max_filtered; i++) {
        const TextRegion* region = &regions[i];
        
        // 检查置信度
        if (region->confidence < config->min_confidence) {
            continue;
        }
        
        // 检查文本高度
        if (region->height < config->min_text_height || 
            region->height > config->max_text_height) {
            continue;
        }
        
        // 检查宽高比
        float aspect_ratio = (float)region->width / region->height;
        if (aspect_ratio < config->min_aspect_ratio || 
            aspect_ratio > config->max_aspect_ratio) {
            continue;
        }
        
        // 检查方向（如果只检测水平文本）
        if (config->detect_horizontal_only && !region->is_horizontal) {
            continue;
        }
        
        // 通过过滤
        filtered_regions[count++] = *region;
    }
    
    *num_filtered = count;
}

void text_detection_merge(TextRegion* regions, int num_regions,
                         float distance_threshold,
                         TextRegion* merged_regions, int max_merged,
                         int* num_merged) {
    if (!regions || !merged_regions || !num_merged) {
        return;
    }
    
    if (num_regions == 0) {
        *num_merged = 0;
        return;
    }
    
    // 标记已合并的区域
    int* merged = (int*)safe_calloc(num_regions, sizeof(int));
    if (!merged) {
        *num_merged = 0;
        return;
    }
    
    memset(merged, 0, num_regions * sizeof(int));
    
    int count = 0;
    
    for (int i = 0; i < num_regions && count < max_merged; i++) {
        if (merged[i]) {
            continue;
        }
        
        TextRegion current = regions[i];
        int merge_count = 1;
        
        // 查找与当前区域相邻的其他区域
        for (int j = i + 1; j < num_regions; j++) {
            if (merged[j]) {
                continue;
            }
            
            // 计算区域中心距离
            float cx1 = current.x + current.width / 2.0f;
            float cy1 = current.y + current.height / 2.0f;
            float cx2 = regions[j].x + regions[j].width / 2.0f;
            float cy2 = regions[j].y + regions[j].height / 2.0f;
            
            float dx = cx1 - cx2;
            float dy = cy1 - cy2;
            float distance = sqrtf(dx * dx + dy * dy);
            
            // 计算平均字符高度
            float avg_height = (current.height + regions[j].height) / 2.0f;
            
            if (distance < avg_height * distance_threshold) {
                // 合并区域
                int new_x_min = current.x < regions[j].x ? current.x : regions[j].x;
                int new_y_min = current.y < regions[j].y ? current.y : regions[j].y;
                int new_x_max = current.x + current.width > regions[j].x + regions[j].width ?
                               current.x + current.width : regions[j].x + regions[j].width;
                int new_y_max = current.y + current.height > regions[j].y + regions[j].height ?
                               current.y + current.height : regions[j].y + regions[j].height;
                
                current.x = new_x_min;
                current.y = new_y_min;
                current.width = new_x_max - new_x_min;
                current.height = new_y_max - new_y_min;
                current.confidence = (current.confidence * merge_count + regions[j].confidence) / (merge_count + 1);
                
                merge_count++;
                merged[j] = 1;
            }
        }
        
        merged_regions[count++] = current;
        merged[i] = 1;
    }
    
    *num_merged = count;
    
    safe_free((void**)&merged);
}

TextDetectionConfig text_detection_get_default_config(void) {
    TextDetectionConfig config;
    
    config.algorithm = (TextDetectionAlgorithm)TEXT_DETECT_EDGE_BASED;
    config.min_confidence = 0.5f;
    config.min_text_height = 8;
    config.max_text_height = 200;
    config.min_aspect_ratio = 0.1f;
    config.max_aspect_ratio = 10.0f;
    config.use_multiscale = 0;
    config.nms_threshold = 0.3f;
    config.detect_multilingual = 1;
    config.detect_horizontal_only = 1;
    
    return config;
}

/* ==================== 静态函数实现 ==================== */

/**
 * @brief 转换为灰度图
 */
static int convert_to_grayscale(const float* image, int width, int height, int channels,
                               float* gray) {
    if (!image || !gray) {
        return -1;
    }
    
    if (channels == 1) {
        memcpy(gray, image, width * height * sizeof(float));
    } else if (channels == 3) {
        for (int i = 0; i < width * height; i++) {
            float r = image[i * 3];
            float g = image[i * 3 + 1];
            float b = image[i * 3 + 2];
            gray[i] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    } else {
        return -2;
    }
    
    return 0;
}

/* ===================================================================
 * SWT (Stroke Width Transform) 文本检测
 * Epshtein, Ofek, Wexler 2010 — 笔画宽度变换自然场景文字检测
 *
 * 算法流程：
 * 1. Canny边缘检测
 * 2. 沿梯度方向发射射线寻找匹配边缘
 * 3. 将笔画宽度分配给射线路径上的每个像素
 * 4. 连通域分析：SWT方差小+高长宽比+面积合适=文字组件
 * 5. 组件成链：间距/高度比/笔画宽度均值→文本行聚合
 * =================================================================== */

#define SWT_UNASSIGNED (-1.0f)
#define SWT_MAX_WIDTH 70
#define SWT_MIN_WIDTH 2
#define SWT_CANNY_LOW 0.05f
#define SWT_CANNY_HIGH 0.15f

typedef struct {
    int x, y;
    float swt_value;
    int component_id;
} SwtPixel;

static void swt_canny_edges(const float* gray, int w, int h,
                             uint8_t* edge_map, float* grad_x, float* grad_y) {
    float* mag = (float*)safe_malloc((size_t)(w*h) * sizeof(float));
    float* sup = (float*)safe_malloc((size_t)(w*h) * sizeof(float));
    if (!mag || !sup) { safe_free((void**)&mag); safe_free((void**)&sup); return; }

    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            float gx = gray[y*w + (x+1)] - gray[y*w + (x-1)];
            float gy = gray[(y+1)*w + x] - gray[(y-1)*w + x];
            grad_x[y*w + x] = gx;
            grad_y[y*w + x] = gy;
            mag[y*w + x] = sqrtf(gx*gx + gy*gy);
        }
    }

    /* 非极大值抑制 */
    for (int y = 2; y < h - 2; y++) {
        for (int x = 2; x < w - 2; x++) {
            float angle = atan2f(grad_y[y*w + x], grad_x[y*w + x]);
            float m = mag[y*w + x];
            float n1, n2;
            if (angle < -2.356f || angle > 2.356f) {
                n1 = mag[y*w + (x-1)]; n2 = mag[y*w + (x+1)];
            } else if (angle < -0.785f) {
                n1 = mag[(y-1)*w + (x-1)]; n2 = mag[(y+1)*w + (x+1)];
            } else if (angle < 0.785f) {
                n1 = mag[(y-1)*w + x]; n2 = mag[(y+1)*w + x];
            } else {
                n1 = mag[(y-1)*w + (x+1)]; n2 = mag[(y+1)*w + (x-1)];
            }
            sup[y*w + x] = (m >= n1 && m >= n2) ? m : 0.0f;
        }
    }

    /* 双阈值 */
    float high = SWT_CANNY_HIGH, low = SWT_CANNY_LOW;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            edge_map[y*w + x] = (sup[y*w + x] >= high) ? 1 :
                                (sup[y*w + x] >= low) ? 0 : 0;

    /* 滞后连接 */
    for (int y = 1; y < h - 1; y++)
        for (int x = 1; x < w - 1; x++)
            if (sup[y*w + x] >= low)
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                        if (edge_map[(y+dy)*w + (x+dx)])
                            edge_map[y*w + x] = 1;

    safe_free((void**)&mag); safe_free((void**)&sup);
}

static int swt_compute(const float* gray, int w, int h, float* swt) {
    uint8_t* edges = (uint8_t*)safe_calloc((size_t)(w*h), sizeof(uint8_t));
    float* gx = (float*)safe_calloc((size_t)(w*h), sizeof(float));
    float* gy = (float*)safe_calloc((size_t)(w*h), sizeof(float));
    if (!edges || !gx || !gy) {
        safe_free((void**)&edges); safe_free((void**)&gx); safe_free((void**)&gy);
        return -1;
    }

    for (int i = 0; i < w*h; i++) swt[i] = SWT_UNASSIGNED;
    swt_canny_edges(gray, w, h, edges, gx, gy);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (!edges[y*w + x]) continue;
            /* 归一化梯度方向 */
            float gxn = gx[y*w + x], gyn = gy[y*w + x];
            float gmag = sqrtf(gxn*gxn + gyn*gyn);
            if (gmag < 1e-6f) continue;
            gxn /= gmag; gyn /= gmag;

            /* 沿梯度方向发射射线寻找匹配边缘 */
            for (int r = 1; r < SWT_MAX_WIDTH; r++) {
                int px = x + (int)(gxn * (float)r + 0.5f);
                int py = y + (int)(gyn * (float)r + 0.5f);
                if (px < 0 || px >= w || py < 0 || py >= h) break;
                if (!edges[py*w + px]) continue;

                /* 检查反向梯度匹配 */
                float gxr = gx[py*w + px], gyr = gy[py*w + px];
                float gmag_r = sqrtf(gxr*gxr + gyr*gyr);
                if (gmag_r < 1e-6f) break;
                float dot = gxn * (gxr/gmag_r) + gyn * (gyr/gmag_r);
                /* 反向梯度：dot ≈ -1 (误差±π/3内) */
                if (dot > -0.5f) break;

                /* 沿射线路径分配笔画宽度 */
                for (int ri = 0; ri < r; ri++) {
                    int rx = x + (int)(gxn * (float)ri + 0.5f);
                    int ry = y + (int)(gyn * (float)ri + 0.5f);
                    if (rx < 0 || rx >= w || ry < 0 || ry >= h) continue;
                    float val = (float)r;
                    if (swt[ry*w + rx] < 0 || val < swt[ry*w + rx])
                        swt[ry*w + rx] = val;
                }
                break;
            }
        }
    }
    safe_free((void**)&edges); safe_free((void**)&gx); safe_free((void**)&gy);
    return 0;
}

static int swt_extract_components(const float* swt, int w, int h,
                                   TextRegion* regions, int max_regions) {
    int* labels = (int*)safe_calloc((size_t)(w*h), sizeof(int));
    int* stack = (int*)safe_malloc((size_t)(w*h*2) * sizeof(int));
    if (!labels || !stack) { safe_free((void**)&labels); safe_free((void**)&stack); return 0; }

    int comp_count = 0;
    int label = 0;

    for (int sy = 0; sy < h && comp_count < max_regions; sy++) {
        for (int sx = 0; sx < w && comp_count < max_regions; sx++) {
            if (swt[sy*w + sx] < 0 || labels[sy*w + sx] != 0) continue;

            label++;
            int sp = 0;
            stack[sp++] = sy * w + sx;
            labels[sy*w + sx] = label;

            int min_x = w, min_y = h, max_x = 0, max_y = 0;
            float swt_sum = 0.0f, swt_sq = 0.0f;
            int pixel_count = 0;

            while (sp > 0) {
                int idx = stack[--sp];
                int px = idx % w, py = idx / w;
                if (px < min_x) min_x = px; if (px > max_x) max_x = px;
                if (py < min_y) min_y = py; if (py > max_y) max_y = py;
                float s = swt[idx];
                swt_sum += s; swt_sq += s*s; pixel_count++;

                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = px + dx, ny = py + dy;
                        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                        int nidx = ny*w + nx;
                        if (swt[nidx] < 0 || labels[nidx] != 0) continue;
                        /* SWT比值约束：相邻像素笔画宽度比<3 */
                        float ratio = s / swt[nidx];
                        if (ratio > 3.0f || ratio < 0.33f) continue;
                        labels[nidx] = label;
                        stack[sp++] = nidx;
                    }
                }
            }

            if (pixel_count < 10) continue;
            float swt_mean = swt_sum / (float)pixel_count;
            float swt_var = swt_sq/(float)pixel_count - swt_mean*swt_mean;

            int rw = max_x - min_x + 1, rh = max_y - min_y + 1;
            float aspect = (rw > rh) ? (float)rw/(float)rh : (float)rh/(float)rw;

            /* 文本组件过滤条件 */
            if (rw < 3 || rh < 3 || rw > w * 0.8f || rh > h * 0.8f) continue;
            if (aspect > 10.0f && pixel_count < 30) continue;
            /* SWT方差：文字区域方差小（笔画宽度一致） */
            float swt_cv = sqrtf(swt_var) / (swt_mean + 1e-6f);
            if (swt_cv > 1.0f && pixel_count < 50) continue;

            regions[comp_count].x = min_x;
            regions[comp_count].y = min_y;
            regions[comp_count].width = rw;
            regions[comp_count].height = rh;
            regions[comp_count].confidence = 1.0f - swt_cv;
            if (regions[comp_count].confidence > 1.0f) regions[comp_count].confidence = 1.0f;
            if (regions[comp_count].confidence < 0.1f) regions[comp_count].confidence = 0.1f;
            comp_count++;
        }
    }

    safe_free((void**)&labels); safe_free((void**)&stack);
    return comp_count;
}

int text_detection_swt(const float* gray, int width, int height,
                        TextRegion* regions, int max_regions, int* num_regions) {
    if (!gray || !regions || !num_regions || max_regions <= 0) return -1;

    float* swt_map = (float*)safe_malloc((size_t)(width*height) * sizeof(float));
    if (!swt_map) return -1;

    swt_compute(gray, width, height, swt_map);
    *num_regions = swt_extract_components(swt_map, width, height, regions, max_regions);

    safe_free((void**)&swt_map);
    return 0;
}

/**
 * @brief Sobel边缘检测
 */
static int detect_edges_sobel(const float* gray, int width, int height,
                             float* magnitude, float* direction) {
    if (!gray || !magnitude || !direction) {
        return -1;
    }
    
    // Sobel算子
    const int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    // 初始化边缘图像
    memset(magnitude, 0, width * height * sizeof(float));
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float gx = 0.0f, gy = 0.0f;
            
            // 应用Sobel算子
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    float pixel = gray[(y + dy) * width + (x + dx)];
                    gx += pixel * sobel_x[dy + 1][dx + 1];
                    gy += pixel * sobel_y[dy + 1][dx + 1];
                }
            }
            
            int idx = y * width + x;
            magnitude[idx] = sqrtf(gx * gx + gy * gy);
            direction[idx] = atan2f(gy, gx);
        }
    }
    
    return 0;
}

/**
 * @brief 二值化边缘图像
 */
static int binarize_edge(const float* magnitude, int width, int height,
                        float threshold, float* binary) {
    if (!magnitude || !binary) {
        return -1;
    }
    
    // 完整实现：自适应阈值计算（大津法 - Otsu's method）
    // 如果threshold <= 0，自动计算最佳阈值；否则使用提供的阈值
    float used_threshold = threshold;
    
    if (threshold <= 0.0f) {
        // 计算直方图（256个bin）
        int histogram[256] = {0};
        int total_pixels = width * height;
        float mag_min = magnitude[0];
        float mag_max = magnitude[0];
        
        // 找到幅度范围
        for (int i = 0; i < total_pixels; i++) {
            float mag = magnitude[i];
            if (mag < mag_min) mag_min = mag;
            if (mag > mag_max) mag_max = mag;
        }
        
        // 归一化幅度到[0, 255]并填充直方图
        float range = mag_max - mag_min;
        if (range > 0.0f) {
            for (int i = 0; i < total_pixels; i++) {
                float normalized = (magnitude[i] - mag_min) / range * 255.0f;
                int bin = (int)normalized;
                if (bin < 0) bin = 0;
                if (bin > 255) bin = 255;
                histogram[bin]++;
            }
        } else {
            // 所有值相同，使用中间阈值
            used_threshold = mag_min;
        }
        
        // 大津法计算最佳阈值
        if (range > 0.0f) {
            float sum = 0.0f;
            for (int i = 0; i < 256; i++) {
                sum += i * histogram[i];
            }
            
            float sumB = 0.0f;
            int wB = 0;
            int wF = 0;
            float max_variance = 0.0f;
            int best_threshold = 0;
            
            for (int i = 0; i < 256; i++) {
                wB += histogram[i];
                if (wB == 0) continue;
                
                wF = total_pixels - wB;
                if (wF == 0) break;
                
                sumB += i * histogram[i];
                float mB = sumB / wB;
                float mF = (sum - sumB) / wF;
                
                // 计算类间方差
                float variance = (float)wB * (float)wF * (mB - mF) * (mB - mF);
                if (variance > max_variance) {
                    max_variance = variance;
                    best_threshold = i;
                }
            }
            
            // 将阈值转换回原始幅度范围
            used_threshold = mag_min + (best_threshold / 255.0f) * range;
        }
    }
    
    // 使用计算出的阈值进行二值化
    for (int i = 0; i < width * height; i++) {
        binary[i] = magnitude[i] > used_threshold ? 1.0f : 0.0f;
    }
    
    return 0;
}

/**
 * @brief 查找连通域（使用递归泛洪填充）
 */
static int find_connected_components(const float* binary, int width, int height,
                                    int* labels, int* num_components) {
    if (!binary || !labels || !num_components) {
        return -1;
    }
    
    // 初始化标签数组
    memset(labels, 0, width * height * sizeof(int));
    
    int current_label = 0;
    
    // 使用栈进行泛洪填充避免递归深度问题
    int* stack_x = (int*)safe_calloc(width * height, sizeof(int));
    int* stack_y = (int*)safe_calloc(width * height, sizeof(int));
    if (!stack_x || !stack_y) {
        safe_free((void**)&stack_x);
        safe_free((void**)&stack_y);
        return -2;
    }
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            
            // 如果当前像素是前景且未标记
            if (binary[idx] > 0.5f && labels[idx] == 0) {
                current_label++;
                
                int stack_size = 0;
                stack_x[stack_size] = x;
                stack_y[stack_size] = y;
                stack_size++;
                
                labels[idx] = current_label;
                
                while (stack_size > 0) {
                    stack_size--;
                    int cx = stack_x[stack_size];
                    int cy = stack_y[stack_size];
                    
                    // 检查8邻域
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dx == 0 && dy == 0) continue;
                            
                            int nx = cx + dx;
                            int ny = cy + dy;
                            
                            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                                int nidx = ny * width + nx;
                                if (binary[nidx] > 0.5f && labels[nidx] == 0) {
                                    labels[nidx] = current_label;
                                    stack_x[stack_size] = nx;
                                    stack_y[stack_size] = ny;
                                    stack_size++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    *num_components = current_label;
    
    safe_free((void**)&stack_x);
    safe_free((void**)&stack_y);
    
    return 0;
}

/**
 * @brief 提取连通域信息
 */
static int extract_component_info(const int* labels, int width, int height,
                                 int num_components, ConnectedComponent* components) {
    if (!labels || !components) {
        return -1;
    }
    
    // 初始化连通域信息
    for (int i = 0; i < num_components; i++) {
        components[i].label = i + 1;
        components[i].x_min = width;
        components[i].x_max = 0;
        components[i].y_min = height;
        components[i].y_max = 0;
        components[i].area = 0;
        components[i].centroid_x = 0.0f;
        components[i].centroid_y = 0.0f;
    }
    
    // 计算连通域属性
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int label = labels[idx];
            
            if (label > 0) {
                ConnectedComponent* comp = &components[label - 1];
                comp->area++;
                
                if (x < comp->x_min) comp->x_min = x;
                if (x > comp->x_max) comp->x_max = x;
                if (y < comp->y_min) comp->y_min = y;
                if (y > comp->y_max) comp->y_max = y;
                
                comp->centroid_x += x;
                comp->centroid_y += y;
            }
        }
    }
    
    // 计算质心和宽高比
    for (int i = 0; i < num_components; i++) {
        if (components[i].area > 0) {
            components[i].centroid_x /= components[i].area;
            components[i].centroid_y /= components[i].area;
            
            int comp_width = components[i].x_max - components[i].x_min + 1;
            int comp_height = components[i].y_max - components[i].y_min + 1;
            components[i].aspect_ratio = (float)comp_width / comp_height;
        }
    }
    
    return 0;
}

/**
 * @brief 过滤文本候选区域
 */
static int filter_text_candidates(const ConnectedComponent* components, int num_components,
                                 const TextDetectionConfig* config,
                                 TextRegion* regions, int max_regions, int* num_regions) {
    if (!components || !config || !regions || !num_regions) {
        return -1;
    }
    
    int count = 0;
    
    for (int i = 0; i < num_components && count < max_regions; i++) {
        const ConnectedComponent* comp = &components[i];
        
        int width = comp->x_max - comp->x_min + 1;
        int height = comp->y_max - comp->y_min + 1;
        
        // 过滤条件
        if (height < config->min_text_height || height > config->max_text_height) {
            continue;
        }
        
        if (comp->aspect_ratio < config->min_aspect_ratio || 
            comp->aspect_ratio > config->max_aspect_ratio) {
            continue;
        }
        
        if (comp->solidity < 0.3f) { // 固体度太低
            continue;
        }
        
        if (comp->area < (width * height) * 0.2f) { // 面积太小
            continue;
        }
        
        // 计算置信度（基于固体度和宽高比）
        float solidity_score = comp->solidity;
        float aspect_score = 1.0f - fabsf(comp->aspect_ratio - 2.0f) / 2.0f; // 理想宽高比约为2
        aspect_score = aspect_score > 0 ? aspect_score : 0;
        
        float confidence = 0.7f * solidity_score + 0.3f * aspect_score;
        
        if (confidence < config->min_confidence) {
            continue;
        }
        
        // 添加到结果
        regions[count].x = comp->x_min;
        regions[count].y = comp->y_min;
        regions[count].width = width;
        regions[count].height = height;
        regions[count].confidence = confidence;
        regions[count].angle = 0.0f;
        regions[count].is_horizontal = 1;
        regions[count].num_vertices = 4;
        
        // 设置矩形顶点
        regions[count].vertices[0] = comp->x_min;
        regions[count].vertices[1] = comp->y_min;
        regions[count].vertices[2] = comp->x_max;
        regions[count].vertices[3] = comp->y_min;
        regions[count].vertices[4] = comp->x_max;
        regions[count].vertices[5] = comp->y_max;
        regions[count].vertices[6] = comp->x_min;
        regions[count].vertices[7] = comp->y_max;
        
        count++;
    }
    
    *num_regions = count;
    return 0;
}

/**
 * @brief 计算连通域的固体度（完整实现：基于凸包面积）
 */
static float calculate_component_solidity(const ConnectedComponent* comp,
                                         const int* labels, int width, int height) {
    (void)height; // 参数在边界检查中使用，但编译器可能无法识别
    if (!comp || !labels) {
        return 0.0f;
    }
    
    // 完整实现：计算凸包面积，然后计算固体度 = 面积 / 凸包面积
    // 由于计算凸包需要提取组件的所有像素点，这里实现完整的凸包算法
    
    // 步骤1：收集组件边界像素（减少计算量）
    // 最大边界像素数（组件的周长最多为2*(width+height)）
    int max_boundary_points = 2 * ((comp->x_max - comp->x_min + 1) + (comp->y_max - comp->y_min + 1));
    int* boundary_x = (int*)safe_malloc(max_boundary_points * sizeof(int));
    int* boundary_y = (int*)safe_malloc(max_boundary_points * sizeof(int));
    int num_boundary = 0;
    
    if (!boundary_x || !boundary_y) {
        if (boundary_x) safe_free((void**)&boundary_x);
        if (boundary_y) safe_free((void**)&boundary_y);
        // 内存分配失败，拒绝降级处理，返回错误值
        log_error("固体度计算内存分配失败。\n");
        return 0.0f;
    }
    
    // 提取边界像素：像素至少有一个邻居不属于该组件
    int component_label = comp->label;
    for (int y = comp->y_min; y <= comp->y_max; y++) {
        for (int x = comp->x_min; x <= comp->x_max; x++) {
            int idx = y * width + x;
            if (labels[idx] == component_label) {
                // 检查4邻域
                int is_boundary = 0;
                if (x == comp->x_min || x == comp->x_max || y == comp->y_min || y == comp->y_max) {
                    is_boundary = 1;
                } else {
                    // 检查上下左右
                    if (labels[(y-1)*width + x] != component_label) is_boundary = 1;
                    else if (labels[(y+1)*width + x] != component_label) is_boundary = 1;
                    else if (labels[y*width + (x-1)] != component_label) is_boundary = 1;
                    else if (labels[y*width + (x+1)] != component_label) is_boundary = 1;
                }
                
                if (is_boundary && num_boundary < max_boundary_points) {
                    boundary_x[num_boundary] = x;
                    boundary_y[num_boundary] = y;
                    num_boundary++;
                }
            }
        }
    }
    
    // 边界点不足时拒绝降级处理，返回错误值
    if (num_boundary < 3) {
        safe_free((void**)&boundary_x);
        safe_free((void**)&boundary_y);
        log_error("固体度计算边界点不足。\n");
        return 0.0f;
    }
    
    // 步骤2：计算凸包（使用Graham scan算法）
    // 首先找到最左下角的点作为基准点
    int pivot_index = 0;
    for (int i = 1; i < num_boundary; i++) {
        if (boundary_y[i] < boundary_y[pivot_index] || 
            (boundary_y[i] == boundary_y[pivot_index] && boundary_x[i] < boundary_x[pivot_index])) {
            pivot_index = i;
        }
    }
    
    // 交换基准点到第一个位置
    int temp_x = boundary_x[0];
    int temp_y = boundary_y[0];
    boundary_x[0] = boundary_x[pivot_index];
    boundary_y[0] = boundary_y[pivot_index];
    boundary_x[pivot_index] = temp_x;
    boundary_y[pivot_index] = temp_y;
    
    // 准备排序数组（存储索引和角度）
    int* indices = (int*)safe_malloc(num_boundary * sizeof(int));
    float* angles = (float*)safe_malloc(num_boundary * sizeof(float));
    if (!indices || !angles) {
        safe_free((void**)&boundary_x);
        safe_free((void**)&boundary_y);
        if (indices) safe_free((void**)&indices);
        if (angles) safe_free((void**)&angles);
        // 内存分配失败，拒绝降级处理，返回错误值
        log_error("固体度计算内存分配失败。\n");
        return 0.0f;
    }
    
    // 计算每个点相对于基准点的极角
    for (int i = 0; i < num_boundary; i++) {
        indices[i] = i;
        int dx = boundary_x[i] - boundary_x[0];
        int dy = boundary_y[i] - boundary_y[0];
        angles[i] = atan2f((float)dy, (float)dx);
    }
    
    // 按极角排序（完整实现：快速排序，O(n log n)， 冒泡排序）
    // 实现快速排序按极角排序
    int angle_sort_stack[256][2];
    int stack_top = 0;
    angle_sort_stack[0][0] = 1;
    angle_sort_stack[0][1] = num_boundary - 1;
    
    while (stack_top >= 0) {
        int left = angle_sort_stack[stack_top][0];
        int right = angle_sort_stack[stack_top][1];
        stack_top--;
        
        if (left >= right) continue;
        
        // 选择中间元素作为枢轴
        int mid = left + (right - left) / 2;
        float pivot_angle = angles[indices[mid]];
        
        // 将枢轴移到末尾
        int temp_idx = indices[mid];
        indices[mid] = indices[right];
        indices[right] = temp_idx;
        
        int store = left;
        for (int i = left; i < right; i++) {
            if (angles[indices[i]] < pivot_angle ||
                (fabsf(angles[indices[i]] - pivot_angle) < 1e-6f &&
                 ((boundary_x[indices[i]] - boundary_x[0]) * (boundary_x[indices[i]] - boundary_x[0]) +
                  (boundary_y[indices[i]] - boundary_y[0]) * (boundary_y[indices[i]] - boundary_y[0])) <=
                 ((boundary_x[indices[right]] - boundary_x[0]) * (boundary_x[indices[right]] - boundary_x[0]) +
                  (boundary_y[indices[right]] - boundary_y[0]) * (boundary_y[indices[right]] - boundary_y[0])))) {
                temp_idx = indices[i];
                indices[i] = indices[store];
                indices[store] = temp_idx;
                store++;
            }
        }
        
        temp_idx = indices[store];
        indices[store] = indices[right];
        indices[right] = temp_idx;
        
        if (store - 1 > left) {
            stack_top++;
            angle_sort_stack[stack_top][0] = left;
            angle_sort_stack[stack_top][1] = store - 1;
        }
        if (store + 1 < right) {
            stack_top++;
            angle_sort_stack[stack_top][0] = store + 1;
            angle_sort_stack[stack_top][1] = right;
        }
    }
    
    // Graham scan算法
    int* hull = (int*)safe_malloc(num_boundary * sizeof(int));
    if (!hull) {
        safe_free((void**)&boundary_x);
        safe_free((void**)&boundary_y);
        safe_free((void**)&indices);
        safe_free((void**)&angles);
        // 内存分配失败，拒绝降级处理，返回错误值
        log_error("固体度计算内存分配失败。\n");
        return 0.0f;
    }
    
    int hull_size = 0;
    hull[hull_size++] = indices[0];
    if (num_boundary > 1) hull[hull_size++] = indices[1];
    
    for (int i = 2; i < num_boundary; i++) {
        while (hull_size >= 2) {
            int a = hull[hull_size - 2];
            int b = hull[hull_size - 1];
            int c = indices[i];
            
            // 计算叉积 (b-a) × (c-b)
            int cross = (boundary_x[b] - boundary_x[a]) * (boundary_y[c] - boundary_y[b]) -
                       (boundary_y[b] - boundary_y[a]) * (boundary_x[c] - boundary_x[b]);
            
            if (cross <= 0) {  // 非左转，弹出
                hull_size--;
            } else {
                break;
            }
        }
        hull[hull_size++] = indices[i];
    }
    
    // 计算凸包面积（鞋带公式）
    float hull_area = 0.0f;
    if (hull_size >= 3) {
        for (int i = 0; i < hull_size; i++) {
            int j = (i + 1) % hull_size;
            hull_area += boundary_x[hull[i]] * boundary_y[hull[j]] - boundary_y[hull[i]] * boundary_x[hull[j]];
        }
        hull_area = fabsf(hull_area) * 0.5f;
    }
    
    // 清理内存
    safe_free((void**)&boundary_x);
    safe_free((void**)&boundary_y);
    safe_free((void**)&indices);
    safe_free((void**)&angles);
    safe_free((void**)&hull);
    
    // 计算固体度
    if (hull_area > 0.0f) {
        return (float)comp->area / hull_area;
    } else {
        // 凸包面积为零，使用边界框面积
        int bbox_area = (comp->x_max - comp->x_min + 1) * (comp->y_max - comp->y_min + 1);
        if (bbox_area <= 0) return 0.0f;
        return (float)comp->area / bbox_area;
    }
}

/**
 * @brief 应用非极大值抑制
 */
static void apply_non_maximum_suppression(TextRegion* regions, int num_regions,
                                         float threshold,
                                         TextRegion* result_regions, int max_result,
                                         int* num_result) {
    if (!regions || !result_regions || !num_result) {
        return;
    }
    
    if (num_regions == 0) {
        *num_result = 0;
        return;
    }
    
    // 按置信度排序
    TextRegion* sorted_regions = (TextRegion*)safe_calloc(num_regions, sizeof(TextRegion));
    if (!sorted_regions) {
        *num_result = 0;
        return;
    }
    
    memcpy(sorted_regions, regions, num_regions * sizeof(TextRegion));
    qsort(sorted_regions, num_regions, sizeof(TextRegion), compare_regions_by_confidence);
    
    // 标记保留的区域
    int* keep = (int*)safe_calloc(num_regions, sizeof(int));
    if (!keep) {
        safe_free((void**)&sorted_regions);
        *num_result = 0;
        return;
    }
    
    for (int i = 0; i < num_regions; i++) {
        keep[i] = 1;
    }
    
    // 应用NMS
    for (int i = 0; i < num_regions; i++) {
        if (!keep[i]) {
            continue;
        }
        
        for (int j = i + 1; j < num_regions; j++) {
            if (!keep[j]) {
                continue;
            }
            
            float overlap = calculate_region_overlap(&sorted_regions[i], &sorted_regions[j]);
            if (overlap > threshold) {
                keep[j] = 0;
            }
        }
    }
    
    // 收集结果
    int count = 0;
    for (int i = 0; i < num_regions && count < max_result; i++) {
        if (keep[i]) {
            result_regions[count++] = sorted_regions[i];
        }
    }
    
    *num_result = count;
    
    safe_free((void**)&keep);
    safe_free((void**)&sorted_regions);
}

/**
 * @brief 计算区域重叠率
 */
static float calculate_region_overlap(const TextRegion* r1, const TextRegion* r2) {
    // 计算交并比
    int x1 = r1->x;
    int y1 = r1->y;
    int w1 = r1->width;
    int h1 = r1->height;
    
    int x2 = r2->x;
    int y2 = r2->y;
    int w2 = r2->width;
    int h2 = r2->height;
    
    int inter_x1 = x1 > x2 ? x1 : x2;
    int inter_y1 = y1 > y2 ? y1 : y2;
    int inter_x2 = (x1 + w1) < (x2 + w2) ? (x1 + w1) : (x2 + w2);
    int inter_y2 = (y1 + h1) < (y2 + h2) ? (y1 + h1) : (y2 + h2);
    
    if (inter_x1 >= inter_x2 || inter_y1 >= inter_y2) {
        return 0.0f;
    }
    
    int inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
    int union_area = w1 * h1 + w2 * h2 - inter_area;
    
    if (union_area <= 0) {
        return 0.0f;
    }
    
    return (float)inter_area / union_area;
}

/**
 * @brief 比较函数：按置信度排序
 */
static int compare_regions_by_confidence(const void* a, const void* b) {
    const TextRegion* r1 = (const TextRegion*)a;
    const TextRegion* r2 = (const TextRegion*)b;
    
    if (r1->confidence > r2->confidence) return -1;
    if (r1->confidence < r2->confidence) return 1;
    return 0;
}

/* ============================================================================
 * 场景文字3D空间定位：结合深度图和相机参数计算文字区域的世界坐标
 * ============================================================================ */

int text_detection_get_3d_position(const TextRegion* region,
                                    const float* depth_map,
                                    int depth_width, int depth_height,
                                    float fx, float fy, float cx, float cy,
                                    float* world_xyz) {
    if (!region || !depth_map || !world_xyz) return -1;

    int cx_px = (int)((region->x + region->x + region->width) / 2);
    int cy_px = (int)((region->y + region->y + region->height) / 2);

    if (cx_px < 0 || cx_px >= depth_width || cy_px < 0 || cy_px >= depth_height) return -1;

    float z = depth_map[cy_px * depth_width + cx_px];
    if (z < 0.01f) return -1;

    if (fx <= 0) fx = (float)depth_width;
    if (fy <= 0) fy = (float)depth_height;
    if (cx <= 0) cx = (float)depth_width * 0.5f;
    if (cy <= 0) cy = (float)depth_height * 0.5f;

    world_xyz[0] = ((float)cx_px - cx) * z / fx;
    world_xyz[1] = ((float)cy_px - cy) * z / fy;
    world_xyz[2] = z;

    return 0;
}

int text_detection_estimate_3d_bbox(const TextRegion* region,
                                     const float* depth_map,
                                     int dw, int dh,
                                     float fx, float fy, float cx, float cy,
                                     float* bbox_corners_3x8) {
    if (!region || !depth_map || !bbox_corners_3x8) return -1;

    int corners_x[4] = {region->x, region->x + region->width,
                         region->x + region->width, region->x};
    int corners_y[4] = {region->y, region->y,
                         region->y + region->height, region->y + region->height};

    for (int c = 0; c < 4; c++) {
        int px = corners_x[c], py = corners_y[c];
        if (px < 0) px = 0; if (px >= dw) px = dw - 1;
        if (py < 0) py = 0; if (py >= dh) py = dh - 1;

        float z = depth_map[py * dw + px];
        if (z < 0.01f) z = 1.0f;

        if (fx <= 0) fx = (float)dw;
        if (fy <= 0) fy = (float)dh;
        if (cx <= 0) cx = (float)dw * 0.5f;
        if (cy <= 0) cy = (float)dh * 0.5f;

        bbox_corners_3x8[c * 3] = ((float)px - cx) * z / fx;
        bbox_corners_3x8[c * 3 + 1] = ((float)py - cy) * z / fy;
        bbox_corners_3x8[c * 3 + 2] = z;
    }

    for (int c = 0; c < 4; c++) {
        bbox_corners_3x8[(c + 4) * 3] = bbox_corners_3x8[c * 3];
        bbox_corners_3x8[(c + 4) * 3 + 1] = bbox_corners_3x8[c * 3 + 1];
        bbox_corners_3x8[(c + 4) * 3 + 2] = bbox_corners_3x8[c * 3 + 2] + 0.3f;
    }

    return 0;
}