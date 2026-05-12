#include "selflnn/utils/xorshift_prng.h"
#include "selflnn/utils/secure_random.h"
#include <math.h>

void xorshift_prng_seed(XorshiftPrng* prng, uint64_t seed) {
    if (!prng) return;
    /* 使用 SplitMix64 从单一种子初始化 Xorshift128+ 的状态 */
    uint64_t z = seed;
    prng->state[0] = z;
    z = z * 6364136223846793005ULL + 1442695040888963407ULL;
    prng->state[1] = z;
}

void xorshift_prng_seed_secure(XorshiftPrng* prng) {
    if (!prng) return;
    uint8_t buf[16];
    if (secure_random_bytes(buf, 16) == 0) {
        prng->state[0] = ((uint64_t)buf[0]) | ((uint64_t)buf[1]) << 8 |
                         ((uint64_t)buf[2]) << 16 | ((uint64_t)buf[3]) << 24 |
                         ((uint64_t)buf[4]) << 32 | ((uint64_t)buf[5]) << 40 |
                         ((uint64_t)buf[6]) << 48 | ((uint64_t)buf[7]) << 56;
        prng->state[1] = ((uint64_t)buf[8]) | ((uint64_t)buf[9]) << 8 |
                         ((uint64_t)buf[10]) << 16 | ((uint64_t)buf[11]) << 24 |
                         ((uint64_t)buf[12]) << 32 | ((uint64_t)buf[13]) << 40 |
                         ((uint64_t)buf[14]) << 48 | ((uint64_t)buf[15]) << 56;
        /* 确保状态不全为0（Xorshift128+ 零状态会退化） */
        if (prng->state[0] == 0 && prng->state[1] == 0) {
            prng->state[0] = 0x9E3779B97F4A7C15ULL;
            prng->state[1] = 0xBF58476D1CE4E5B9ULL;
        }
    } else {
        /* secure_random_bytes 失败时的回退方案 */
        xorshift_prng_seed(prng, (uint64_t)(uintptr_t)prng ^ 0x9E3779B97F4A7C15ULL);
    }
}

uint64_t xorshift_prng_next_u64(XorshiftPrng* prng) {
    if (!prng) return 0;
    uint64_t s1 = prng->state[0];
    uint64_t s0 = prng->state[1];
    prng->state[0] = s0;
    s1 ^= s1 << 23;
    prng->state[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return prng->state[1] + s0;
}

uint32_t xorshift_prng_next_u32(XorshiftPrng* prng, uint32_t max) {
    if (!prng || max == 0) return 0;
    if (max == 1) return 0;
    /* 拒绝采样避免模偏差 */
    uint32_t threshold = (uint32_t)((uint64_t)(max - 1) << 32) / max;
    uint32_t val;
    do {
        val = (uint32_t)(xorshift_prng_next_u64(prng) >> 32);
    } while (val < threshold);
    return val % max;
}

float xorshift_prng_next_float(XorshiftPrng* prng) {
    if (!prng) return 0.0f;
    /* 使用高 53 位生成 [0, 1) 浮点数（双精度分辨率） */
    uint64_t raw = xorshift_prng_next_u64(prng) >> 11;
    return (float)(raw * 0x1.0p-53);
}

float xorshift_prng_next_signed_float(XorshiftPrng* prng) {
    if (!prng) return 0.0f;
    return 2.0f * xorshift_prng_next_float(prng) - 1.0f;
}

float xorshift_prng_next_gaussian(XorshiftPrng* prng) {
    if (!prng) return 0.0f;
    /* Box-Muller 变换 */
    float u1, u2;
    do {
        u1 = xorshift_prng_next_float(prng);
    } while (u1 <= 1e-10f);
    u2 = xorshift_prng_next_float(prng);
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * 3.14159265358979323846f * u2;
    return r * cosf(theta);
}

void xorshift_prng_fill_u64(XorshiftPrng* prng, uint64_t* buffer, size_t count) {
    if (!prng || !buffer) return;
    for (size_t i = 0; i < count; i++) {
        buffer[i] = xorshift_prng_next_u64(prng);
    }
}

void xorshift_prng_fill_float(XorshiftPrng* prng, float* buffer, size_t count) {
    if (!prng || !buffer) return;
    for (size_t i = 0; i < count; i++) {
        buffer[i] = xorshift_prng_next_float(prng);
    }
}
