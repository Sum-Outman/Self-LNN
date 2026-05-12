/**
 * @file image_loader.h
 * @brief 纯C图像格式解码器头文件
 *
 * K-004/K-008修复: BMP/PPM格式原生解码，供训练数据加载和视觉处理管线使用。
 */

#ifndef SELFLNN_IMAGE_LOADER_H
#define SELFLNN_IMAGE_LOADER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 解码BMP文件为RGB字节数组
 * @param filepath BMP文件路径
 * @param width_out 输出宽度
 * @param height_out 输出高度
 * @return RGB字节数组(需调用者free), 失败返回NULL
 */
uint8_t* image_load_bmp(const char* filepath, int* width_out, int* height_out);

/**
 * @brief 解码PPM(P6)文件为RGB字节数组
 * @param filepath PPM文件路径
 * @param width_out 输出宽度
 * @param height_out 输出高度
 * @return RGB字节数组(需调用者free), 失败返回NULL
 */
uint8_t* image_load_ppm(const char* filepath, int* width_out, int* height_out);

/**
 * @brief 将RGB uint8数据转换为float数组(归一化到[0,1])
 * @param rgb_data RGB字节数据
 * @param width 宽度
 * @param height 高度
 * @param channels_out 输出通道数(1=灰度, 3=RGB)
 * @return float数组(需调用者以safe_free释放), 失败返回NULL
 */
float* image_rgb_to_float(const uint8_t* rgb_data, int width, int height, int channels_out);

/**
 * @brief 从文件加载图像并转换为归一化float数组
 *
 * 自动检测格式(BMP/PPM)并解码。
 * 为视觉处理管线提供统一的图像加载接口。
 *
 * @param filepath 图像文件路径 (.bmp/.ppm)
 * @param width_out 输出宽度
 * @param height_out 输出高度
 * @param channels 期望通道数(1=灰度, 3=RGB)
 * @return float数组(需调用者以safe_free释放), 失败返回NULL
 */
float* image_load_float(const char* filepath, int* width_out, int* height_out, int channels);

/**
 * @brief 释放通过image_load_float加载的图像数据
 */
void image_float_free(float* data);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_IMAGE_LOADER_H */
