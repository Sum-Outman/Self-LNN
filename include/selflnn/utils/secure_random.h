#ifndef SELFLNN_SECURE_RANDOM_H
#define SELFLNN_SECURE_RANDOM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 填充缓冲区为密码学安全随机字节
 * @param buffer 输出缓冲区
 * @param length 填充字节数
 * @return 0=成功，-1=失败
 */
int secure_random_bytes(uint8_t* buffer, size_t length);

/**
 * @brief 生成 [0, 1) 范围的浮点随机数
 * @return 浮点随机数
 */
float secure_random_float(void);

/**
 * @brief 生成 [0, max) 范围的整数随机数
 * @param max 上限（不包含）
 * @return 整数随机数
 */
uint32_t secure_random_int(uint32_t max);

/**
 * @brief 使用密码学安全随机数初始化系统种子
 * 在系统启动时调用一次
 */
void secure_random_init(void);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_SECURE_RANDOM_H */
