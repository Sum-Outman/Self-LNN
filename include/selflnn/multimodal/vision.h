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

/* ================================================================
 * 动态可扩展类别系统 (P0-004修复)
 *
 * 保留80类COCO兼容类别作为初始默认类别，但系统完全动态可扩展。
 * 通过多模态教学和学习过程可以动态注册新类别，
 * 类别ID从80开始自动递增分配。
 * 所有模型权重为本地原生模型，从零开始训练。
 * 类别映射仅为标签字典，不包含任何预训练权重或模型参数。
 * ================================================================ */

#define VISION_CLASS_DEFAULT_COUNT 80       /**< 初始默认类别数（COCO兼容） */
#define VISION_CLASS_MAX_CAPACITY  100000   /**< 最大可扩展类别容量 */
#define VISION_ENHANCED_MAX_OBJECTS 50      /**< 单次检测最大目标数 */
#define VISION_CLASS_NAME_MAX_LEN   64      /**< 类别名称最大长度 */

/**
 * @brief 视觉类别注册表（动态可扩展）
 *
 * 初始包含80类COCO兼容类别，可通过学习动态添加新类别。
 * 线程安全，支持并发读写。
 */
typedef struct VisionClassRegistry VisionClassRegistry;

/**
 * @brief 类别条目结构体
 */
typedef struct {
    int class_id;                              /**< 类别ID（0开始自增） */
    char name_zh[VISION_CLASS_NAME_MAX_LEN];   /**< 中文名称 */
    char name_en[VISION_CLASS_NAME_MAX_LEN];   /**< 英文名称 */
    int is_learned;                            /**< 是否通过学习添加（1=是） */
    int sample_count;                          /**< 该类别已学习的样本数 */
    float confidence_threshold;                /**< 该类别检测置信度阈值 */
} VisionClassEntry;

/**
 * @brief 创建类别注册表（自动加载80类COCO默认类别）
 * @return 注册表句柄，失败返回NULL
 */
VisionClassRegistry* vision_class_registry_create(void);

/**
 * @brief 释放类别注册表
 * @param registry 注册表句柄
 */
void vision_class_registry_free(VisionClassRegistry* registry);

/**
 * @brief 获取全局视觉类别注册表（单例模式）
 * @return 全局注册表句柄，未初始化返回NULL
 */
VisionClassRegistry* vision_class_registry_get_global(void);

/**
 * @brief 通过多模态学习注册新类别
 * @param registry 注册表句柄
 * @param name_zh 中文名称
 * @param name_en 英文名称
 * @return 新分配的class_id，失败返回-1
 */
int vision_class_register(VisionClassRegistry* registry,
                          const char* name_zh, const char* name_en);

/**
 * @brief 获取当前类别总数（随学习动态增长）
 * @param registry 注册表句柄
 * @return 类别总数，失败返回0
 */
int vision_class_get_count(const VisionClassRegistry* registry);

/**
 * @brief 获取指定类别条目详情
 * @param registry 注册表句柄
 * @param class_id 类别ID
 * @param entry 输出条目（可为NULL仅检测存在性）
 * @return 成功返回0，不存在返回-1
 */
int vision_class_get_entry(const VisionClassRegistry* registry, int class_id,
                           VisionClassEntry* entry);

/**
 * @brief 增加某类别的学习样本计数
 * @param registry 注册表句柄
 * @param class_id 类别ID
 * @param count 增加的样本数
 * @return 成功返回0，失败返回-1
 */
int vision_class_add_samples(VisionClassRegistry* registry, int class_id, int count);

/* 增强检测结构体——动态类别数（P0-004修复） */
typedef struct {
    float cx, cy, w, h;
    float confidence;
    int class_id;
    char class_name[VISION_CLASS_NAME_MAX_LEN];
    float* class_probs;          /**< 动态分配的类别概率数组，长度为当前类别总数 */
    int class_probs_count;       /**< class_probs的实际长度 */
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

/* 增强版CfC目标检测（动态类别数） */
int vision_enhanced_cfc_detect(const float* features, int feature_dim,
                                LNN* vision_lnn,
                                VisionEnhancedDetect* detections,
                                int max_detections, int* num_found);

/* 原生LNN视觉目标检测（像素到检测框的完整LNN管线，动态类别数） */
typedef struct {
    float cx, cy, w, h;
    float confidence;
    int class_id;
    char class_name[VISION_CLASS_NAME_MAX_LEN];
    float* class_probs;          /**< 动态分配的类别概率数组 */
    int class_probs_count;       /**< class_probs的实际长度 */
} CfCVisionDetection;

int vision_cfc_detect(const float* image, int width, int height, int channels,
                       LNN* vision_lnn, CfCVisionDetection* detections,
                       int max_detections, int* num_found);

/* 原始像素到LNN视觉特征编码 */
int vision_pixel_to_cfc_features(const uint8_t* raw_pixels, int width, int height,
                                  int channels, LNN* vision_lnn,
                                  float* visual_features, int feature_dim);

/* 视觉类别ID到名称映射函数（动态查询） */
const char* vision_get_class_name_zh(int class_id);
const char* vision_get_class_name_en(int class_id);

/* 非极大值抑制 */
int vision_nms(CfCVisionDetection* detections, int count, float iou_threshold);

/* ZSFWS-L002: 释放检测结果中动态分配的class_probs内存
 * 每个检测结果的class_probs由safe_malloc分配，调用者必须使用此函数释放 */
void vision_free_detections(CfCVisionDetection* detections, int count);

#ifdef __cplusplus
}
#endif

#endif
