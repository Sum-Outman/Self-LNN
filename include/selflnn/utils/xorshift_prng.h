#ifndef SELFLNN_XORSHIFT_PRNG_H
#define SELFLNN_XORSHIFT_PRNG_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Xorshift128+ PRNG 状态
 *
 * Xorshift128+ 是一种快速、高质量的伪随机数生成器。
 * 周期为 2^128 - 1，通过 BigCrush 所有测试。
 * 适用于训练数据生成、仿真噪声、权重初始化等场景。
 * 不适用于密码学安全场景——密码学用途请使用 secure_random_*。
 */
typedef struct {
    uint64_t state[2];
} XorshiftPrng;

/**
 * @brief 使用种子初始化 PRNG
 *
 * @param prng PRNG 状态指针
 * @param seed 64位种子（0也是有效种子，但建议使用非零值）
 */
void xorshift_prng_seed(XorshiftPrng* prng, uint64_t seed);

/**
 * @brief 使用高熵种子初始化 PRNG
 *
 * 从 secure_random_bytes() 获取高质量种子。
 * 每次调用产生不同的序列。
 *
 * @param prng PRNG 状态指针
 */
void xorshift_prng_seed_secure(XorshiftPrng* prng);

/**
 * @brief 生成下一个 64 位伪随机数
 *
 * @param prng PRNG 状态指针
 * @return uint64_t 均匀分布的 64 位随机数
 */
uint64_t xorshift_prng_next_u64(XorshiftPrng* prng);

/**
 * @brief 生成 [0, max) 范围的 32 位伪随机整数
 *
 * 无偏生成，使用拒绝采样避免模数偏差。
 *
 * @param prng PRNG 状态指针
 * @param max 上限（不包含），必须 > 0
 * @return uint32_t 均匀分布在 [0, max) 的整数
 */
uint32_t xorshift_prng_next_u32(XorshiftPrng* prng, uint32_t max);

/**
 * @brief 生成 [0.0f, 1.0f) 范围的 32 位浮点伪随机数
 *
 * 53 位尾数精度（利用 64 位随机数的高 53 位）。
 *
 * @param prng PRNG 状态指针
 * @return float 均匀分布在 [0.0f, 1.0f) 的浮点数
 */
float xorshift_prng_next_float(XorshiftPrng* prng);

/**
 * @brief 生成 [-1.0f, 1.0f) 范围的 32 位浮点伪随机数
 *
 * @param prng PRNG 状态指针
 * @return float 均匀分布在 [-1.0f, 1.0f) 的浮点数
 */
float xorshift_prng_next_signed_float(XorshiftPrng* prng);

/**
 * @brief 生成均值为 0、标准差为 1 的正态分布浮点数
 *
 * 使用 Box-Muller 变换从均匀分布生成正态分布。
 *
 * @param prng PRNG 状态指针
 * @return float 标准正态分布 N(0,1) 的采样值
 */
float xorshift_prng_next_gaussian(XorshiftPrng* prng);

/**
 * @brief 用随机值填充缓冲区
 *
 * @param prng PRNG 状态指针
 * @param buffer 输出缓冲区
 * @param count 填充的 uint64_t 元素数量
 */
void xorshift_prng_fill_u64(XorshiftPrng* prng, uint64_t* buffer, size_t count);

/**
 * @brief 用 [0.0f, 1.0f) 浮点随机值填充缓冲区
 *
 * @param prng PRNG 状态指针
 * @param buffer 输出缓冲区
 * @param count 填充的 float 元素数量
 */
void xorshift_prng_fill_float(XorshiftPrng* prng, float* buffer, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_XORSHIFT_PRNG_H */
