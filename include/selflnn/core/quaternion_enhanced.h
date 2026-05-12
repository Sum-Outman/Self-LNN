/**
 * @file quaternion_enhanced.h
 * @brief 四元数增强全面引入系统接口
 */

#ifndef SELFLNN_QUATERNION_ENHANCED_H
#define SELFLNN_QUATERNION_ENHANCED_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 四元数结构 */
typedef struct {
    float w, x, y, z;
} Quaternion;

/* 对偶四元数 */
typedef struct {
    Quaternion real;     /* 实部（旋转） */
    Quaternion dual;     /* 对偶部（平移） */
} DualQuaternion;

/* 四元数插值类型 */
typedef enum {
    QUAT_INTERP_LERP = 0,    /* 线性插值 */
    QUAT_INTERP_SLERP = 1,   /* 球面线性插值 */
    QUAT_INTERP_SQUAD = 2,   /* 球面四边形插值 */
    QUAT_INTERP_SLERP_NO = 3 /* 无归一化SLERP */
} QuatInterpType;

/* 四元数工具函数 */
static inline Quaternion quat_identity(void) {
    Quaternion q = {1.0f, 0.0f, 0.0f, 0.0f};
    return q;
}

static inline Quaternion quat_from_axis_angle(float ax, float ay, float az, float angle) {
    float half = angle * 0.5f;
    float s = sinf(half);
    Quaternion q = {cosf(half), ax * s, ay * s, az * s};
    return q;
}

static inline Quaternion quat_mul(const Quaternion a, const Quaternion b) {
    Quaternion q;
    q.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    q.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    q.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
    q.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
    return q;
}

static inline Quaternion quat_conjugate(const Quaternion q) {
    Quaternion r = {q.w, -q.x, -q.y, -q.z};
    return r;
}

static inline float quat_norm(const Quaternion q) {
    return sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
}

static inline Quaternion quat_normalize(const Quaternion q) {
    float n = quat_norm(q);
    if (n < 1e-10f) return quat_identity();
    Quaternion r = {q.w / n, q.x / n, q.y / n, q.z / n};
    return r;
}

static inline Quaternion quat_slerp(Quaternion a, Quaternion b, float t) {
    a = quat_normalize(a);
    b = quat_normalize(b);
    float dot = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
    if (dot < 0.0f) {
        dot = -dot;
        b.w = -b.w; b.x = -b.x; b.y = -b.y; b.z = -b.z;
    }
    if (dot > 0.9995f) {
        Quaternion r;
        r.w = a.w + t * (b.w - a.w);
        r.x = a.x + t * (b.x - a.x);
        r.y = a.y + t * (b.y - a.y);
        r.z = a.z + t * (b.z - a.z);
        return quat_normalize(r);
    }
    float theta = acosf(dot);
    float sin_theta = sinf(theta);
    float wa = sinf((1.0f - t) * theta) / sin_theta;
    float wb = sinf(t * theta) / sin_theta;
    Quaternion r;
    r.w = wa * a.w + wb * b.w;
    r.x = wa * a.x + wb * b.x;
    r.y = wa * a.y + wb * b.y;
    r.z = wa * a.z + wb * b.z;
    return quat_normalize(r);
}

static inline void quat_to_matrix(const Quaternion q, float* matrix) {
    float w = q.w, x = q.x, y = q.y, z = q.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;

    matrix[0] = 1.0f - 2.0f * (y2 + z2);
    matrix[1] = 2.0f * (xy + wz);
    matrix[2] = 2.0f * (xz - wy);
    matrix[3] = 0.0f;

    matrix[4] = 2.0f * (xy - wz);
    matrix[5] = 1.0f - 2.0f * (x2 + z2);
    matrix[6] = 2.0f * (yz + wx);
    matrix[7] = 0.0f;

    matrix[8] = 2.0f * (xz + wy);
    matrix[9] = 2.0f * (yz - wx);
    matrix[10] = 1.0f - 2.0f * (x2 + y2);
    matrix[11] = 0.0f;

    matrix[12] = 0.0f; matrix[13] = 0.0f; matrix[14] = 0.0f; matrix[15] = 1.0f;
}

static inline void quat_rotate_vector(const Quaternion q, const float* v, float* out) {
    Quaternion qv = {0.0f, v[0], v[1], v[2]};
    Quaternion qc = quat_conjugate(q);
    Quaternion result = quat_mul(quat_mul(q, qv), qc);
    out[0] = result.x;
    out[1] = result.y;
    out[2] = result.z;
}

static inline Quaternion quat_from_euler(float roll, float pitch, float yaw) {
    float cr = cosf(roll * 0.5f), sr = sinf(roll * 0.5f);
    float cp = cosf(pitch * 0.5f), sp = sinf(pitch * 0.5f);
    float cy = cosf(yaw * 0.5f), sy = sinf(yaw * 0.5f);
    Quaternion q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;
    return q;
}

static inline void quat_to_euler(const Quaternion q, float* roll, float* pitch, float* yaw) {
    float sinr_cosp = 2.0f * (q.w * q.x + q.y * q.z);
    float cosr_cosp = 1.0f - 2.0f * (q.x * q.x + q.y * q.y);
    *roll = atan2f(sinr_cosp, cosr_cosp);

    float sinp = 2.0f * (q.w * q.y - q.z * q.x);
    if (fabsf(sinp) >= 1.0f)
        *pitch = copysignf(M_PI / 2.0f, sinp);
    else
        *pitch = asinf(sinp);

    float siny_cosp = 2.0f * (q.w * q.z + q.x * q.y);
    float cosy_cosp = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
    *yaw = atan2f(siny_cosp, cosy_cosp);
}

/* 对偶四元数 */
static inline DualQuaternion dual_quat_from_pose(const Quaternion rotation, const float* translation) {
    DualQuaternion dq;
    dq.real = rotation;
    Quaternion qt = {0.0f, translation[0] * 0.5f, translation[1] * 0.5f, translation[2] * 0.5f};
    dq.dual = quat_mul(qt, rotation);
    return dq;
}

static inline Quaternion quat_log(const Quaternion q) {
    float n = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z);
    Quaternion result = {0.0f, 0.0f, 0.0f, 0.0f};
    if (n > 1e-10f) {
        float theta = atan2f(n, q.w);
        float scale = theta / n;
        result.x = scale * q.x;
        result.y = scale * q.y;
        result.z = scale * q.z;
    }
    return result;
}

static inline Quaternion quat_exp(const Quaternion q) {
    float n = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z);
    Quaternion result;
    if (n > 1e-10f) {
        float s = sinf(n) / n;
        result.w = cosf(n);
        result.x = s * q.x;
        result.y = s * q.y;
        result.z = s * q.z;
    } else {
        result.w = 1.0f;
        result.x = 0.0f; result.y = 0.0f; result.z = 0.0f;
    }
    return result;
}

static inline float quat_distance(const Quaternion a, const Quaternion b) {
    Quaternion diff = quat_mul(a, quat_conjugate(b));
    float angle = 2.0f * acosf(fminf(fabsf(diff.w), 1.0f));
    return angle;
}

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_QUATERNION_ENHANCED_H */
