/**
 * @file memory.h
 * @brief 内存管理核心头文件（兼容性重定向）
 * 
 * 【ZSF-011】此文件为兼容性重定向头文件，所有声明已迁移至 utils/memory_utils.h。
 * 新代码应直接使用 #include "selflnn/utils/memory_utils.h"。
 * 保留此文件仅为向后兼容，未来版本将移除此文件。
 *
 * 重定向至: selflnn/utils/memory_utils.h
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