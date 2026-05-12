/**
 * @file character_segmentation.h
 * @brief 字符分割算法
 * 
 * 提供从图像中分割字符的算法，支持连通域分析、投影分析、深度学习等方法。
 * 100%纯C实现，不依赖任何第三方库。
 */

#ifndef SELFLNN_MULTIMODAL_CHARACTER_SEGMENTATION_H
#define SELFLNN_MULTIMODAL_CHARACTER_SEGMENTATION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 字符分割算法类型
 */
typedef enum {
    CHAR_SEG_CONNECTED_COMPONENTS = 0, /**< 连通域分析 */
    CHAR_SEG_PROJECTION_ANALYSIS = 1,  /**< 投影分析 */
    CHAR_SEG_WATERSHED = 2,            /**< 分水岭算法 */
    CHAR_SEG_DEEP_LEARNING = 3         /**< 深度学习分割 */
} CharSegmentationAlgorithm;

/**
 * @brief 字符区域结构体
 */
typedef struct {
    int x;          /**< 区域左上角x坐标 */
    int y;          /**< 区域左上角y坐标 */
    int width;      /**< 区域宽度 */
    int height;     /**< 区域高度 */
    float confidence; /**< 置信度 */
    int char_code;  /**< 字符编码（如果已知） */
} CharRegion;

/**
 * @brief 字符分割配置
 */
typedef struct {
    CharSegmentationAlgorithm algorithm; /**< 分割算法 */
    int min_char_width;                  /**< 最小字符宽度 */
    int max_char_width;                  /**< 最大字符宽度 */
    int min_char_height;                 /**< 最小字符高度 */
    int max_char_height;                 /**< 最大字符高度 */
    float min_aspect_ratio;              /**< 最小宽高比 */
    float max_aspect_ratio;              /**< 最大宽高比 */
    int merge_overlapping;               /**< 是否合并重叠区域 */
    float merge_threshold;               /**< 合并阈值 */
    int use_binarization;                /**< 是否使用二值化 */
    float binarization_threshold;        /**< 二值化阈值 */
} CharSegmentationConfig;

/**
 * @brief 字符分割处理器结构体（不透明）
 */
typedef struct CharSegmentationProcessor CharSegmentationProcessor;

/**
 * @brief 获取默认的字符分割配置
 * 
 * @return CharSegmentationConfig 默认配置
 */
CharSegmentationConfig char_segmentation_get_default_config(void);

/**
 * @brief 创建字符分割处理器
 * 
 * @param config 分割配置
 * @return CharSegmentationProcessor* 处理器指针，失败返回NULL
 */
CharSegmentationProcessor* char_segmentation_processor_create(const CharSegmentationConfig* config);

/**
 * @brief 释放字符分割处理器
 * 
 * @param processor 处理器指针
 */
void char_segmentation_processor_free(CharSegmentationProcessor* processor);

/**
 * @brief 从图像中分割字符
 * 
 * @param processor 处理器
 * @param image_data 图像数据（灰度图，值范围0-1）
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 图像通道数（1=灰度，3=RGB）
 * @param regions 输出字符区域数组
 * @param max_regions 最大区域数
 * @param num_regions 实际区域数（输出）
 * @return int 成功返回0，失败返回错误码
 */
int char_segmentation_segment(CharSegmentationProcessor* processor,
                             const float* image_data, int width, int height, int channels,
                             CharRegion* regions, int max_regions, int* num_regions);

/**
 * @brief 从文本行图像中分割字符
 * 
 * @param processor 处理器
 * @param line_image 文本行图像数据（灰度图）
 * @param line_width 文本行宽度
 * @param line_height 文本行高度
 * @param regions 输出字符区域数组（相对于文本行）
 * @param max_regions 最大区域数
 * @param num_regions 实际区域数（输出）
 * @return int 成功返回0，失败返回错误码
 */
int char_segmentation_segment_line(CharSegmentationProcessor* processor,
                                  const float* line_image, int line_width, int line_height,
                                  CharRegion* regions, int max_regions, int* num_regions);

/**
 * @brief 合并重叠的字符区域
 * 
 * @param regions 字符区域数组
 * @param num_regions 区域数量
 * @param threshold 合并阈值（重叠率）
 * @param merged_regions 合并后的区域数组
 * @param max_merged 最大合并区域数
 * @param num_merged 实际合并区域数（输出）
 * @return int 成功返回0，失败返回错误码
 */
int char_segmentation_merge_overlapping(CharRegion* regions, int num_regions,
                                       float threshold,
                                       CharRegion* merged_regions, int max_merged,
                                       int* num_merged);

/**
 * @brief 按阅读顺序排序字符区域（从左到右，从上到下）
 * 
 * @param regions 字符区域数组
 * @param num_regions 区域数量
 */
void char_segmentation_sort_regions(CharRegion* regions, int num_regions);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_MULTIMODAL_CHARACTER_SEGMENTATION_H