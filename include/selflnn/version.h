/**
 * @file version.h
 * @brief SELF-LNN版本号 (DEEP-005: 替代CMake生成的version.h.in)
 * 
 * 原系统使用CMake的configure_file从version.h.in生成，MSVC构建时该文件缺失。
 */

#ifndef SELFLNN_VERSION_H
#define SELFLNN_VERSION_H

#define SELFLNN_VERSION_MAJOR 1
#define SELFLNN_VERSION_MINOR 5
#define SELFLNN_VERSION_PATCH 0

#define SELFLNN_VERSION_STRING "1.5.0"

#endif /* SELFLNN_VERSION_H */
