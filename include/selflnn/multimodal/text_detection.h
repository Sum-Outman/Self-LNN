/**
 * @file text_detection.h
 * @brief 文字区域检测和定位
 * 
 * 提供图像中文字区域的检测和定位功能，支持多种检测算法。
 * 100%纯C实现，不依赖任何第三方库。
 */

#ifndef SELFLNN_MULTIMODAL_TEXT_DETECTION_H
#define SELFLNN_MULTIMODAL_TEXT_DETECTION_H

#include <stddef.h>
#include "ocr.h"  /* 包含TextRegion定义 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 文本检测算法类型
 */
#ifndef SELFLNN_TEXT_DETECTION_ALGORITHM_DEFINED
#define SELFLNN_TEXT_DETECTION_ALGORITHM_DEFINED
typedef enum {
    TEXT_DETECT_MSER = 0,          /**< 最大稳定极值区域 */
    TEXT_DETECT_SWT = 1,           /**< 笔画宽度变换 */
    TEXT_DETECT_EDGE_BASED = 2,    /**< 基于边缘的方法 */
    TEXT_DETECT_CNN = 3,           /**< 卷积神经网络 */
    TEXT_DETECT_EAST = 4           /**< EAST文本检测 */
} TextDetectionAlgorithm;
#endif /* SELFLNN_TEXT_DETECTION_ALGORITHM_DEFINED */

/**
 * @brief 文本检测配置
 */
typedef struct {
    TextDetectionAlgorithm algorithm; /**< 检测算法 */
    float min_confidence;            /**< 最小置信度 */
    int min_text_height;             /**< 最小文本高度 */
    int max_text_height;             /**< 最大文本高度 */
    float min_aspect_ratio;          /**< 最小宽高比 */
    float max_aspect_ratio;          /**< 最大宽高比 */
    int use_multiscale;              /**< 是否使用多尺度检测 */
    float scale_factor;              /**< 尺度因子 */
    int num_scales;                  /**< 尺度数量 */
    int use_nms;                     /**< 是否使用非极大值抑制 */
    float nms_threshold;             /**< NMS阈值 */
    int detect_multilingual;         /**< 是否检测多语言文本 */
    int detect_horizontal_only;      /**< 是否只检测水平文本 */
} TextDetectionConfig;

/**
 * @brief 文本检测处理器结构体（不透明）
 */
typedef struct TextDetectionProcessor TextDetectionProcessor;

/**
 * @brief 获取默认的文本检测配置
 * 
 * @return TextDetectionConfig 默认配置
 */
TextDetectionConfig text_detection_get_default_config(void);

/**
 * @brief 创建文本检测处理器
 * 
 * @param config 检测配置
 * @return TextDetectionProcessor* 处理器指针，失败返回NULL
 */
TextDetectionProcessor* text_detection_processor_create(const TextDetectionConfig* config);

/**
 * @brief 释放文本检测处理器
 * 
 * @param processor 处理器指针
 */
void text_detection_processor_free(TextDetectionProcessor* processor);

/**
 * @brief 检测图像中的文本区域
 * 
 * @param processor 处理器
 * @param image_data 图像数据（灰度图或RGB图）
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 图像通道数（1=灰度，3=RGB）
 * @param regions 输出文本区域数组
 * @param max_regions 最大区域数
 * @param num_regions 实际区域数（输出）
 * @return int 成功返回0，失败返回错误码
 */
int text_detection_detect(TextDetectionProcessor* processor,
                         const float* image_data, int width, int height, int channels,
                         TextRegion* regions, int max_regions, int* num_regions);

/**
 * @brief 非极大值抑制（NMS）处理
 * 
 * @param regions 文本区域数组
 * @param num_regions 区域数量
 * @param threshold NMS阈值
 * @param result_regions 结果区域数组
 * @param max_result 最大结果数
 * @param num_result 实际结果数（输出）
 */
void text_detection_nms(TextRegion* regions, int num_regions, float threshold,
                       TextRegion* result_regions, int max_result, int* num_result);

/**
 * @brief 过滤文本区域（基于几何属性）
 * 
 * @param regions 文本区域数组
 * @param num_regions 区域数量
 * @param config 检测配置
 * @param filtered_regions 过滤后的区域数组
 * @param max_filtered 最大过滤区域数
 * @param num_filtered 实际过滤区域数（输出）
 */
void text_detection_filter(TextRegion* regions, int num_regions,
                          const TextDetectionConfig* config,
                          TextRegion* filtered_regions, int max_filtered,
                          int* num_filtered);

/**
 * @brief 合并相邻文本区域
 * 
 * @param regions 文本区域数组
 * @param num_regions 区域数量
 * @param distance_threshold 距离阈值
 * @param merged_regions 合并后的区域数组
 * @param max_merged 最大合并区域数
 * @param num_merged 实际合并区域数（输出）
 */
void text_detection_merge(TextRegion* regions, int num_regions,
                         float distance_threshold,
                         TextRegion* merged_regions, int max_merged,
                         int* num_merged);

/**
 * @brief SWT笔画宽度变换文本检测
 * Epshtein, Ofek, Wexler 2010
 * @param gray 灰度图像
 * @param width 图像宽度
 * @param height 图像高度
 * @param regions 检测结果区域数组
 * @param max_regions 最大区域数
 * @param num_regions 实际区域数（输出）
 * @return int 成功返回0，失败返回-1
 */
int text_detection_swt(const float* gray, int width, int height,
                        TextRegion* regions, int max_regions, int* num_regions);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_MULTIMODAL_TEXT_DETECTION_H