/**
 * @file memory.h
 * @brief 内存管理核心头文件
 * 
 * 提供内存分配、释放和安全内存操作函数。
 * 此文件重定向到 memory_utils.h 以保持向后兼容性。
 */

#ifndef SELFLNN_CORE_MEMORY_H
#define SELFLNN_CORE_MEMORY_H

#include "../utils/memory_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

// 使用 memory_utils.h 中定义的函数，不重复定义宏

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_CORE_MEMORY_H