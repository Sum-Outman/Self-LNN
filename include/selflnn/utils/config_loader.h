/**
 * @file config_loader.h
 * @brief SELF-LNN 配置文件加载和保存模块
 * ZSFWS-S012: 对应src/utils/config_loader.c，提供模块化头文件
 */
#ifndef SELFLNN_UTILS_CONFIG_LOADER_H
#define SELFLNN_UTILS_CONFIG_LOADER_H

#include "selflnn/selflnn.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 从JSON配置文件加载SELF-LNN系统配置
 * @param config 配置输出结构体指针
 * @param config_path 配置文件路径（NULL则使用默认路径）
 * @return 0成功，-1失败
 */
int selflnn_config_load_from_file(SystemConfig* config, const char* config_path);

/**
 * @brief 将SELF-LNN系统配置保存到JSON文件
 * @param config 配置结构体指针
 * @param config_path 保存路径（NULL则使用默认路径）
 * @return 0成功，-1失败
 */
int selflnn_config_save_to_file(const SystemConfig* config, const char* config_path);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_UTILS_CONFIG_LOADER_H */
