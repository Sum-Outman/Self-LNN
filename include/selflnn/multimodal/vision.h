#ifndef SELFLNN_VISION_H
#define SELFLNN_VISION_H

#include <stddef.h>

#include "selflnn/core/lnn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int target_width;
    int target_height;
    int grayscale;
    int feature_dimension;
    int enable_cfc;
    size_t cfc_hidden_size;
    float cfc_time_constant;
    int enable_multiscale_pyramid;
    int enable_hog;
    int enable_color_histogram;
} VisionConfig;

typedef struct VisionProcessor VisionProcessor;

VisionProcessor* vision_processor_create(const VisionConfig* config);

void vision_processor_free(VisionProcessor* processor);

int vision_process_image(VisionProcessor* processor,
                        int width, int height, int channels,
                        const float* data,
                        float* features, size_t max_features);

int vision_processor_get_config(const VisionProcessor* processor, VisionConfig* config);

int vision_processor_set_config(VisionProcessor* processor, const VisionConfig* config);

void vision_processor_reset(VisionProcessor* processor);

int vision_resize_bilinear(int src_width, int src_height, int channels,
                            const float* src_data,
                            int dst_width, int dst_height, float* dst_data);

int vision_resize_bilinear_float(int src_width, int src_height, int channels,
                            const float* src_data,
                            int dst_width, int dst_height, float* dst_data);

int vision_yuv420_to_rgb(int width, int height,
                          const unsigned char* y_plane, int y_stride,
                          const unsigned char* u_plane, int uv_stride,
                          const unsigned char* v_plane,
                          float* rgb_output);

/* 增强检测结构体——支持50个目标×80类别 */
#define VISION_ENHANCED_MAX_OBJECTS 50
#define VISION_ENHANCED_CLASS_MAX 80

typedef struct {
    float cx, cy, w, h;
    float confidence;
    int class_id;
    char class_name[32];
    float class_probs[VISION_ENHANCED_CLASS_MAX];
} VisionEnhancedDetect;

/* HOG特征提取：9方向×N_cells */
int vision_extract_hog_features(const float* gray, int width, int height,
                                float* features, int max_features);

/* 多尺度LBP特征提取 */
int vision_extract_multiscale_lbp(const float* gray, int width, int height,
                                  float* features, int max_features);

/* HSV色彩直方图提取 */
int vision_extract_color_histogram(const float* rgb, int width, int height, int channels,
                                   float* features, int max_features);

/* 增强版CfC目标检测 */
int vision_enhanced_cfc_detect(const float* features, int feature_dim,
                                LNN* vision_lnn,
                                VisionEnhancedDetect* detections,
                                int max_detections, int* num_found);

/* 原生LNN视觉目标检测（像素到检测框的完整LNN管线） */
typedef struct {
    float cx, cy, w, h;
    float confidence;
    int class_id;
} CfCVisionDetection;

int vision_cfc_detect(const float* image, int width, int height, int channels,
                       LNN* vision_lnn, CfCVisionDetection* detections,
                       int max_detections, int* num_found);

/* 原始像素到LNN视觉特征编码 */
int vision_pixel_to_cfc_features(const uint8_t* raw_pixels, int width, int height,
                                  int channels, LNN* vision_lnn,
                                  float* visual_features, int feature_dim);

/* K-018: 视觉类别ID到名称映射函数 */
const char* vision_get_class_name_zh(int class_id);
const char* vision_get_class_name_en(int class_id);

/* 非极大值抑制 */
int vision_nms(CfCVisionDetection* detections, int count, float iou_threshold);

#ifdef __cplusplus
}
#endif

#endif
