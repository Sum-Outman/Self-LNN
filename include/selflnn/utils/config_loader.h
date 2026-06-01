/**
 * @file config_loader.h
 * @brief 系统配置文件加载与保存接口 (纯C, 零依赖)
 * 
 *修复: 为config_loader.c提供公开API头文件声明
 */
#ifndef SELFLNN_CONFIG_LOADER_H
#define SELFLNN_CONFIG_LOADER_H

#include "selflnn/selflnn.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 从JSON文件加载系统配置
 * @param filepath 配置文件路径, NULL时使用默认路径"config/system_config.json"
 * @param config   输出配置结构体
 * @return 0成功, -1失败
 */
int selflnn_config_load_from_file(const char* filepath, SystemConfig* config);

/**
 * @brief 保存系统配置到JSON文件
 * @param filepath 配置文件路径, NULL时使用默认路径"config/system_config.json"
 * @param config   配置结构体
 * @return 0成功, -1失败
 */
int selflnn_config_save_to_file(const char* filepath, const SystemConfig* config);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_CONFIG_LOADER_H */
