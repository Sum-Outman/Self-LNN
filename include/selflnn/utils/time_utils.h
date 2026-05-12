#ifndef SELFLNN_UTILS_TIME_UTILS_H
#define SELFLNN_UTILS_TIME_UTILS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 获取当前时间（毫秒）
 * @return uint64_t 当前时间戳（毫秒）
 */
uint64_t time_utils_get_time_ms(void);

/**
 * @brief 获取当前时间（微秒）
 * @return uint64_t 当前时间戳（微秒）
 */
uint64_t time_utils_get_time_us(void);

/**
 * @brief 获取当前时间（秒）
 * @return double 当前时间戳（秒）
 */
double time_utils_get_time_s(void);

/**
 * K-041: 获取当前时间（纳秒）—— 高精度计时器
 * @return uint64_t 当前时间戳（纳秒）
 */
uint64_t time_utils_get_time_ns(void);

/**
 * K-041: 获取单调时钟（毫秒）—— 不受系统时间调整影响
 * @return uint64_t 单调时钟时间戳（毫秒）
 */
uint64_t time_utils_get_time_monotonic_ms(void);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_UTILS_TIME_UTILS_H */
