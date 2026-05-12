#ifndef SELFLNN_CORE_PLATFORM_TYPES_H
#define SELFLNN_CORE_PLATFORM_TYPES_H

/**
 * @file platform_types.h
 * @brief 平台无关的基础类型和宏定义
 *
 * 提取自 common.h，作为零依赖的第0层头文件。
 * 打破 common.h ↔ errors.h 之间的循环依赖。
 * errors.h 只依赖此文件（而非完整的 common.h）。
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============ 平台导出宏 ============ */

#if defined(_WIN32) || defined(_WIN64)
    #define SELFLNN_API __declspec(dllexport)
    #define SELFLNN_CALL __cdecl
#else
    #define SELFLNN_API __attribute__((visibility("default")))
    #define SELFLNN_CALL
#endif

#ifndef SELFLNN_CALL
    #define SELFLNN_CALL
#endif

/* ============ 平台检测宏 ============ */

#if defined(_WIN32) || defined(_WIN64)
    #define SELFLNN_PLATFORM_WINDOWS 1
#elif defined(__linux__)
    #define SELFLNN_PLATFORM_LINUX 1
#elif defined(__APPLE__)
    #define SELFLNN_PLATFORM_MACOS 1
#endif

/* ============ 布尔类型（纯C兼容） ============ */

typedef enum {
    SELFLNN_FALSE = 0,
    SELFLNN_TRUE = 1
} selflnn_bool_t;

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_CORE_PLATFORM_TYPES_H */
