#ifndef SELFLNN_MATH_VEC3_OPS_H
#define SELFLNN_MATH_VEC3_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <string.h>

/* ============================================================================
 * M-008修复：统一的vec3向量运算 — static inline，消除5个文件中的重复定义
 *
 * 支持两种接口风格：
 *   1. float[3] 数组（与dynamics/phyiscs_engine/gait_generator/vision_odometry兼容）
 *   2. Vec3 结构体（与kinematics兼容）
 * ============================================================================ */

#ifndef SELFLNN_VEC3_EPSILON
#define SELFLNN_VEC3_EPSILON 1e-8f
#endif

/* ========== float[3] 数组风格接口 ========== */

#ifndef SELFLNN_VEC3_OPS_DEFINED_COPY
#define SELFLNN_VEC3_OPS_DEFINED_COPY
static inline void vec3_copy(float dst[3], const float src[3]) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
}
#endif

#ifndef SELFLNN_VEC3_OPS_DEFINED_ADD
#define SELFLNN_VEC3_OPS_DEFINED_ADD
static inline void vec3_add(float dst[3], const float a[3], const float b[3]) {
    dst[0] = a[0] + b[0]; dst[1] = a[1] + b[1]; dst[2] = a[2] + b[2];
}
#endif

#ifndef SELFLNN_VEC3_OPS_DEFINED_SUB
#define SELFLNN_VEC3_OPS_DEFINED_SUB
static inline void vec3_sub(float dst[3], const float a[3], const float b[3]) {
    dst[0] = a[0] - b[0]; dst[1] = a[1] - b[1]; dst[2] = a[2] - b[2];
}
#endif

#ifndef SELFLNN_VEC3_OPS_DEFINED_DOT
#define SELFLNN_VEC3_OPS_DEFINED_DOT
static inline float vec3_dot(const float a[3], const float b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
#endif

#ifndef SELFLNN_VEC3_OPS_DEFINED_CROSS
#define SELFLNN_VEC3_OPS_DEFINED_CROSS
static inline void vec3_cross(float dst[3], const float a[3], const float b[3]) {
    dst[0] = a[1] * b[2] - a[2] * b[1];
    dst[1] = a[2] * b[0] - a[0] * b[2];
    dst[2] = a[0] * b[1] - a[1] * b[0];
}
#endif

#ifndef SELFLNN_VEC3_OPS_DEFINED_SCALE
#define SELFLNN_VEC3_OPS_DEFINED_SCALE
static inline void vec3_scale(float dst[3], const float a[3], float s) {
    dst[0] = a[0] * s; dst[1] = a[1] * s; dst[2] = a[2] * s;
}
#endif

#ifndef SELFLNN_VEC3_OPS_DEFINED_LENGTH
#define SELFLNN_VEC3_OPS_DEFINED_LENGTH
static inline float vec3_length(const float a[3]) {
    return sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}
#endif

#ifndef SELFLNN_VEC3_OPS_DEFINED_LENGTH_SQ
#define SELFLNN_VEC3_OPS_DEFINED_LENGTH_SQ
static inline float vec3_length_sq(const float a[3]) {
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}
#endif

#ifndef SELFLNN_VEC3_OPS_DEFINED_DIST_SQ
#define SELFLNN_VEC3_OPS_DEFINED_DIST_SQ
static inline float vec3_dist_sq(const float a[3], const float b[3]) {
    float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}
#endif

#ifndef SELFLNN_VEC3_OPS_DEFINED_NORMALIZE
#define SELFLNN_VEC3_OPS_DEFINED_NORMALIZE
static inline void vec3_normalize(float dst[3], const float a[3]) {
    float len = vec3_length(a);
    if (len < SELFLNN_VEC3_EPSILON) {
        dst[0] = 0.0f; dst[1] = 0.0f; dst[2] = 0.0f;
        return;
    }
    float inv = 1.0f / len;
    dst[0] = a[0] * inv; dst[1] = a[1] * inv; dst[2] = a[2] * inv;
}
#endif

#ifndef SELFLNN_VEC3_OPS_DEFINED_LERP
#define SELFLNN_VEC3_OPS_DEFINED_LERP
static inline void vec3_lerp(float dst[3], const float a[3], const float b[3], float t) {
    dst[0] = a[0] + (b[0] - a[0]) * t;
    dst[1] = a[1] + (b[1] - a[1]) * t;
    dst[2] = a[2] + (b[2] - a[2]) * t;
}
#endif

#ifndef SELFLNN_VEC3_OPS_DEFINED_ZERO
#define SELFLNN_VEC3_OPS_DEFINED_ZERO
static inline void vec3_zero(float v[3]) {
    v[0] = 0.0f; v[1] = 0.0f; v[2] = 0.0f;
}
#endif

#ifndef SELFLNN_VEC3_OPS_DEFINED_MUL
#define SELFLNN_VEC3_OPS_DEFINED_MUL
static inline void vec3_mul(float dst[3], const float a[3], const float b[3]) {
    dst[0] = a[0] * b[0]; dst[1] = a[1] * b[1]; dst[2] = a[2] * b[2];
}
#endif

/* ========== 3×3矩阵运算 ========== */

static inline void mat3_mul_vec3(float dst[3], const float mat[9], const float v[3]) {
    dst[0] = mat[0] * v[0] + mat[1] * v[1] + mat[2] * v[2];
    dst[1] = mat[3] * v[0] + mat[4] * v[1] + mat[5] * v[2];
    dst[2] = mat[6] * v[0] + mat[7] * v[1] + mat[8] * v[2];
}

static inline void mat3_transpose(float dst[9], const float mat[9]) {
    dst[0] = mat[0]; dst[1] = mat[3]; dst[2] = mat[6];
    dst[3] = mat[1]; dst[4] = mat[4]; dst[5] = mat[7];
    dst[6] = mat[2]; dst[7] = mat[5]; dst[8] = mat[8];
}

static inline void mat3_mul(float dst[9], const float a[9], const float b[9]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) sum += a[i * 3 + k] * b[k * 3 + j];
            dst[i * 3 + j] = sum;
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_MATH_VEC3_OPS_H */
