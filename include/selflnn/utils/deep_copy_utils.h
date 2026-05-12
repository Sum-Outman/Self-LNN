/**
 * @file deep_copy_utils.h
 * @brief 结构体深拷贝通用辅助宏（F-038: 消除copy_knowledge_entry与allocate_alternative_copy模式重复）
 *
 * 提供标准的深拷贝模式宏：字符串字段、float数组字段、blob字段。
 * 每个宏失败时自动回滚已分配资源，确保无内存泄漏。
 * 100%纯C，零外部依赖。
 *
 * 使用示例：
 *   DEEP_COPY_STRING(dest->name, src->name);
 *   DEEP_COPY_FLOAT_ARRAY(dest->values, dest->count, src->values, src->count);
 *   DEEP_COPY_BLOB(dest->data, dest->size, src->data, src->size);
 *   DEEP_COPY_SCALAR(dest->weight, src->weight);
 */

#ifndef SELFLNN_DEEP_COPY_UTILS_H
#define SELFLNN_DEEP_COPY_UTILS_H

#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * 宏：字符串深拷贝 — 源有值则分配新副本，否则置NULL
 * 失败时跳转到 cleanup 标签释放已分配资源
 *
 * 前置条件：dest_field 需已初始化为 NULL
 * ================================================================ */
#define DEEP_COPY_STRING(dest_field, src_field)                                                  \
    do {                                                                                         \
        if ((src_field) != NULL) {                                                               \
            size_t _dcl = strlen(src_field) + 1;                                                 \
            (dest_field) = (char*)malloc(_dcl);                                                  \
            if ((dest_field) == NULL) goto deep_copy_cleanup;                                    \
            memcpy((dest_field), (src_field), _dcl);                                             \
        }                                                                                        \
    } while(0)

#define DEEP_COPY_STRING_SAFE(dest_field, src_field, free_fn)                                    \
    do {                                                                                         \
        if ((src_field) != NULL) {                                                               \
            size_t _dcl = strlen(src_field) + 1;                                                 \
            (dest_field) = (char*)malloc(_dcl);                                                  \
            if ((dest_field) == NULL) goto deep_copy_cleanup;                                    \
            memcpy((dest_field), (src_field), _dcl);                                             \
        } else {                                                                                 \
            (dest_field) = NULL;                                                                 \
        }                                                                                        \
    } while(0)

/* ================================================================
 * 宏：float数组深拷贝 — 源非空且count>0时分配新数组
 * ================================================================ */
#define DEEP_COPY_FLOAT_ARRAY(dest_arr, dest_count, src_arr, src_count)                          \
    do {                                                                                         \
        (dest_count) = (src_count);                                                              \
        if ((src_arr) != NULL && (src_count) > 0) {                                              \
            (dest_arr) = (float*)malloc((size_t)(src_count) * sizeof(float));                    \
            if ((dest_arr) == NULL) goto deep_copy_cleanup;                                      \
            memcpy((dest_arr), (src_arr), (size_t)(src_count) * sizeof(float));                  \
        } else {                                                                                 \
            (dest_arr) = NULL;                                                                   \
        }                                                                                        \
    } while(0)

/* ================================================================
 * 宏：blob深拷贝 — 任意类型的原始字节块
 * ================================================================ */
#define DEEP_COPY_BLOB(dest_ptr, dest_size, src_ptr, src_size)                                   \
    do {                                                                                         \
        (dest_size) = (src_size);                                                                \
        if ((src_ptr) != NULL && (src_size) > 0) {                                               \
            (dest_ptr) = malloc((src_size));                                                     \
            if ((dest_ptr) == NULL) goto deep_copy_cleanup;                                      \
            memcpy((dest_ptr), (src_ptr), (src_size));                                           \
        } else {                                                                                 \
            (dest_ptr) = NULL;                                                                   \
        }                                                                                        \
    } while(0)

/* ================================================================
 * 宏：标量拷贝 — 记录同名字段的直接赋值
 * ================================================================ */
#define DEEP_COPY_SCALAR(dest_field, src_field)   ((dest_field) = (src_field))

/* ================================================================
 * 宏：深拷贝清理序列（在函数末尾定义cleanup标签）
 *
 * 用法：
 *   int my_copy(MyType* dest, const MyType* src) {
 *       memset(dest, 0, sizeof(MyType));
 *       DEEP_COPY_STRING(dest->name, src->name);
 *       DEEP_COPY_FLOAT_ARRAY(dest->data, dest->n, src->data, src->n);
 *       return 0;
 *   deep_copy_cleanup:
 *       DEEP_COPY_CLEANUP_FREE(dest->name);
 *       DEEP_COPY_CLEANUP_FREE(dest->data);
 *       return -1;
 *   }
 * ================================================================ */
#define DEEP_COPY_CLEANUP_FREE(ptr)                                                              \
    do { if ((ptr) != NULL) { free((ptr)); (ptr) = NULL; } } while(0)

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_DEEP_COPY_UTILS_H */
