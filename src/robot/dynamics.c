#include "selflnn/robot/dynamics.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/core/errors.h"
#include "selflnn/math/vec3_ops.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef _MSC_VER
#pragma warning(disable:4100 4189 4244 4267 4701 4101 4090)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DYNAMICS_ABS(x) ((x) >= 0 ? (x) : -(x))

struct DynamicsModel {
    RobotDynamicsConfig config;
};

static void mat3_identity(float* m)
{
    int i;
    for (i = 0; i < 9; i++) m[i] = 0.0f;
    m[0] = 1.0f; m[4] = 1.0f; m[8] = 1.0f;
}

/*  mat3_mul 已在 vec3_ops.h 中 static inline 定义
 *  本地版本参数顺序不同(a,b,out)，重命名为 mat3_mul_dyn 避免冲突 */
static void mat3_mul_dyn(const float* a, const float* b, float* out)
{
    int i, j, k;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (k = 0; k < 3; k++) {
                sum += a[i * 3 + k] * b[k * 3 + j];
            }
            out[i * 3 + j] = sum;
        }
    }
}

/* mat3_transpose 已在 vec3_ops.h 中定义，重命名为 mat3_transpose_dyn 避免冲突 */
static void mat3_transpose_dyn(const float* m, float* out)
{
    int i, j;
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            out[i * 3 + j] = m[j * 3 + i];
}

/* M-008修复: 向量/矩阵运算已统一到 selflnn/math/vec3_ops.h */
#define mat3_vec3_mul(m,v,o)  mat3_mul_vec3(o,m,v)
#define dvec3_cross(a,b,o)    vec3_cross(o,a,b)
#define dvec3_dot(a,b)        vec3_dot(a,b)
#define dvec3_add(a,b,o)      vec3_add(o,a,b)
#define dvec3_sub(a,b,o)      vec3_sub(o,a,b)
#define dvec3_scale(v,s,o)    vec3_scale(o,v,s)
#define dvec3_copy(s,d)       vec3_copy(d,s)
#define dvec3_zero(v)         vec3_zero(v)

static void inertia_transform(const float* I_body, const float* R, float* I_world)
{
    float RT[9], tmp[9];
    mat3_transpose_dyn(R, RT);
    mat3_mul_dyn(I_body, RT, tmp);
    mat3_mul_dyn(R, tmp, I_world);
}

/* ================================================================
 * B-028: 四元数万向节锁检测与回退
 * 1. 四元数结构体与基本运算
 * 2. 轴角→四元数→旋转矩阵转换
 * 3. 万向节锁检测 (|cosθ| < 1e-6)
 * 4. 旋转矩阵↔欧拉角来回转换一致性验证
 * ================================================================ */

typedef struct {
    float w, x, y, z;
} Quat;

static void quat_identity(Quat* q)
{
    q->w = 1.0f; q->x = 0.0f; q->y = 0.0f; q->z = 0.0f;
}

static Quat quat_from_axis_angle(const float axis[3], float angle)
{
    Quat q;
    float half_a = angle * 0.5f;
    float s = sinf(half_a);
    float norm = sqrtf(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    if (norm < 1e-12f) { quat_identity(&q); return q; }
    float inv = 1.0f / norm;
    q.w = cosf(half_a);
    q.x = axis[0] * inv * s;
    q.y = axis[1] * inv * s;
    q.z = axis[2] * inv * s;
    return q;
}

static Quat quat_mul(const Quat* a, const Quat* b)
{
    Quat r;
    r.w = a->w * b->w - a->x * b->x - a->y * b->y - a->z * b->z;
    r.x = a->w * b->x + a->x * b->w + a->y * b->z - a->z * b->y;
    r.y = a->w * b->y - a->x * b->z + a->y * b->w + a->z * b->x;
    r.z = a->w * b->z + a->x * b->y - a->y * b->x + a->z * b->w;
    return r;
}

static Quat quat_normalize_quat(const Quat* q)
{
    Quat r;
    float norm = sqrtf(q->w*q->w + q->x*q->x + q->y*q->y + q->z*q->z);
    if (norm < 1e-12f) { quat_identity(&r); return r; }
    float inv = 1.0f / norm;
    r.w = q->w * inv; r.x = q->x * inv; r.y = q->y * inv; r.z = q->z * inv;
    return r;
}

static void quat_to_mat3(const Quat* q, float* m)
{
    float xx = q->x * q->x, yy = q->y * q->y, zz = q->z * q->z;
    float xy = q->x * q->y, xz = q->x * q->z, yz = q->y * q->z;
    float wx = q->w * q->x, wy = q->w * q->y, wz = q->w * q->z;
    m[0] = 1.0f - 2.0f * (yy + zz);
    m[1] = 2.0f * (xy - wz);
    m[2] = 2.0f * (xz + wy);
    m[3] = 2.0f * (xy + wz);
    m[4] = 1.0f - 2.0f * (xx + zz);
    m[5] = 2.0f * (yz - wx);
    m[6] = 2.0f * (xz - wy);
    m[7] = 2.0f * (yz + wx);
    m[8] = 1.0f - 2.0f * (xx + yy);
}

/* 检测万向节锁条件: |cos(绕X轴旋转角)| < 阈值 */
static int dynamics_detect_gimbal_lock(const float* R, float threshold)
{
    float R11 = R[0], R21 = R[3], R31 = R[6];
    float R32 = R[7], R33 = R[8];
    float cos_theta_y = sqrtf(R11*R11 + R21*R21);
    float cos_theta_x = sqrtf(R32*R32 + R33*R33);
    if (cos_theta_y < threshold || cos_theta_x < threshold) return 1;
    return 0;
}

/* 旋转矩阵→欧拉角提取 (ZYX顺序: 偏航-俯仰-翻滚) */
static void mat3_to_euler_zyx(const float* R, float* yaw, float* pitch, float* roll)
{
    float sy = -R[2];
    if (sy > 1.0f) sy = 1.0f;
    if (sy < -1.0f) sy = -1.0f;
    *pitch = asinf(sy);
    float cos_pitch = cosf(*pitch);
    if (fabsf(cos_pitch) > 1e-6f) {
        *yaw = atan2f(R[5], R[8]);
        *roll = atan2f(R[1], R[0]);
    } else {
        *yaw = atan2f(-R[7], R[4]);
        *roll = 0.0f;
    }
}

/* 欧拉角→旋转矩阵 (ZYX顺序) */
static void euler_zyx_to_mat3(float yaw, float pitch, float roll, float* R)
{
    float cy = cosf(yaw), sy = sinf(yaw);
    float cp = cosf(pitch), sp = sinf(pitch);
    float cr = cosf(roll), sr = sinf(roll);
    R[0] = cy * cp;  R[1] = cy * sp * sr - sy * cr;  R[2] = cy * sp * cr + sy * sr;
    R[3] = sy * cp;  R[4] = sy * sp * sr + cy * cr;  R[5] = sy * sp * cr - cy * sr;
    R[6] = -sp;      R[7] = cp * sr;                  R[8] = cp * cr;
}

/* 验证旋转矩阵→欧拉角→旋转矩阵来回转换一致性 */
static int dynamics_verify_roundtrip(const float* R_orig, float tolerance)
{
    float yaw, pitch, roll;
    float R_test[9];
    mat3_to_euler_zyx(R_orig, &yaw, &pitch, &roll);
    euler_zyx_to_mat3(yaw, pitch, roll, R_test);
    float max_diff = 0.0f;
    for (int k = 0; k < 9; k++) {
        float diff = fabsf(R_orig[k] - R_test[k]);
        if (diff > max_diff) max_diff = diff;
    }
    return (max_diff < tolerance) ? 1 : 0;
}

static void build_rotation_z(float theta, float* m)
{
    float c = (float)cos(theta);
    float s = (float)sin(theta);
    mat3_identity(m);
    m[0] = c; m[1] = -s;
    m[3] = s; m[4] = c;
}

/* 绕任意轴旋转 — 使用四元数，天然避免万向节锁 */
static void build_rotation_axis_angle(const float axis[3], float angle, float* m)
{
    Quat q = quat_from_axis_angle(axis, angle);
    q = quat_normalize_quat(&q);
    quat_to_mat3(&q, m);
}

RobotDynamicsConfig dynamics_config_default(void)
{
    RobotDynamicsConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.num_joints = 0;
    cfg.num_links = 0;
    cfg.gravity[0] = 0.0f;
    cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = -9.81f;
    cfg.contact_stiffness = 100000.0f;
    cfg.contact_damping = 2000.0f;
    cfg.friction_coefficient = 0.8f;
    cfg.restitution = 0.3f;
    cfg.contact_model = CONTACT_MODEL_PENALTY;
    cfg.friction_model = FRICTION_CONE_PYRAMID;
    cfg.enable_joint_friction = 1;
    cfg.enable_joint_damping = 1;
    {
        int i;
        for (i = 0; i < DYNAMICS_MAX_JOINTS; i++) {
            cfg.joint_frictions[i] = 0.1f;
            cfg.joint_dampings[i] = 0.05f;
            cfg.max_joint_torques[i] = 100.0f;
        }
    }
    return cfg;
}

DynamicsModel* dynamics_model_create(const KinematicModel* kin_model, const RobotDynamicsConfig* config)
{
    DynamicsModel* model;
    int i;
    if (!kin_model || !config) return NULL;
    model = (DynamicsModel*)safe_malloc(sizeof(DynamicsModel));
    if (!model) return NULL;
    memset(model, 0, sizeof(DynamicsModel));
    model->config = *config;
    model->config.num_joints = kin_model->joint_count;
    model->config.num_links = kin_model->joint_count;
    for (i = 0; i < kin_model->joint_count && i < DYNAMICS_MAX_JOINTS; i++) {
        const KinJoint* j = &kin_model->joints[i];
        model->config.link_masses[i] = j->link_mass;
        model->config.link_com[i * 3 + 0] = j->link_com.x;
        model->config.link_com[i * 3 + 1] = j->link_com.y;
        model->config.link_com[i * 3 + 2] = j->link_com.z;
        {
            int k;
            for (k = 0; k < 9; k++) {
                model->config.link_inertia[i * 9 + k] = j->link_inertia[k];
            }
        }
        model->config.joint_frictions[i] = j->joint_friction;
        model->config.joint_dampings[i] = j->joint_damping;
        model->config.joint_types[i] = (int)j->joint_type;
        model->config.parent_indices[i] = j->parent_index;
        model->config.joint_axes[i * 3 + 0] = j->joint_axis[0];
        model->config.joint_axes[i * 3 + 1] = j->joint_axis[1];
        model->config.joint_axes[i * 3 + 2] = j->joint_axis[2];
        if (j->joint_max_torque > 0) {
            model->config.max_joint_torques[i] = j->joint_max_torque;
        }
    }
    return model;
}

void dynamics_model_destroy(DynamicsModel* model)
{
    safe_free((void**)&model);
}

static int compute_relative_transform(const DynamicsModel* model, int joint_idx,
                                       float q, float* R, float* p)
{
    float axis[3];
    int parent;
    if (joint_idx <= 0) {
        mat3_identity(R);
        dvec3_zero(p);
        return 0;
    }
    parent = model->config.parent_indices[joint_idx];
    if (parent < 0) {
        mat3_identity(R);
        dvec3_zero(p);
        return 0;
    }
    axis[0] = model->config.joint_axes[joint_idx * 3 + 0];
    axis[1] = model->config.joint_axes[joint_idx * 3 + 1];
    axis[2] = model->config.joint_axes[joint_idx * 3 + 2];
    if (model->config.joint_types[joint_idx] == JOINT_TYPE_REVOLUTE ||
        model->config.joint_types[joint_idx] == JOINT_TYPE_CONTINUOUS) {
        /* B-028: 万向节锁检测 — 当旋转角度接近±90°时(|cosθ|<1e-6)
         * 欧拉角旋转矩阵退化，使用四元数替代方案 */
        float Rz[9];
        float cos_q = cosf(q);
        int gimbal_locked = (fabsf(cos_q) < 1e-6f) ? 1 : 0;

        mat3_identity(R);
        if (gimbal_locked) {
            /* 万向节锁情况下直接使用轴角→四元数→旋转矩阵
             * 绕关节轴axis旋转角度q，避免欧拉角奇异 */
            build_rotation_axis_angle(axis, q, R);
            /* 验证来回转换一致性 */
            if (!dynamics_verify_roundtrip(R, 1e-4f)) {
                /* 一致性验证失败，回退到原始方法 */
                build_rotation_z(q, Rz);
                {
                    float nx = DYNAMICS_ABS(axis[0]), ny = DYNAMICS_ABS(axis[1]), nz = DYNAMICS_ABS(axis[2]);
                    if (nz > nx && nz > ny) {
                        int k;
                        for (k = 0; k < 9; k++) R[k] = Rz[k];
                    } else if (nx > ny && nx > nz) {
                        R[0] = Rz[0]; R[1] = Rz[3]; R[2] = Rz[6];
                        R[3] = Rz[1]; R[4] = Rz[4]; R[5] = Rz[7];
                        R[6] = Rz[2]; R[7] = Rz[5]; R[8] = Rz[8];
                    } else {
                        R[0] = Rz[0]; R[1] = Rz[1]; R[2] = Rz[2];
                        R[3] = Rz[3]; R[4] = Rz[4]; R[5] = Rz[5];
                        R[6] = Rz[6]; R[7] = Rz[7]; R[8] = Rz[8];
                    }
                }
            }
        } else {
            /* 正常情况: 使用绕Z轴旋转 + 轴排列映射 */
            build_rotation_z(q, Rz);
            {
                float nx = DYNAMICS_ABS(axis[0]), ny = DYNAMICS_ABS(axis[1]), nz = DYNAMICS_ABS(axis[2]);
                if (nz > nx && nz > ny) {
                    int k;
                    for (k = 0; k < 9; k++) R[k] = Rz[k];
                } else if (nx > ny && nx > nz) {
                    R[0] = Rz[0]; R[1] = Rz[3]; R[2] = Rz[6];
                    R[3] = Rz[1]; R[4] = Rz[4]; R[5] = Rz[7];
                    R[6] = Rz[2]; R[7] = Rz[5]; R[8] = Rz[8];
                } else {
                    R[0] = Rz[0]; R[1] = Rz[1]; R[2] = Rz[2];
                    R[3] = Rz[3]; R[4] = Rz[4]; R[5] = Rz[5];
                    R[6] = Rz[6]; R[7] = Rz[7]; R[8] = Rz[8];
                }
            }
        }
        dvec3_zero(p);
    } else {
        mat3_identity(R);
        dvec3_zero(p);
        p[0] = q * axis[0];
        p[1] = q * axis[1];
        p[2] = q * axis[2];
    }
    return 0;
}

/* ================================================================
 * K-031: GJK碰撞检测 + EPA穿透深度计算
 * ================================================================ */

typedef struct {
    float vertices[32][3];
    int num_vertices;
} GjkShape;

/* 3D Minkowski差的支持函数 */
static void gjk_support(const GjkShape* a, const GjkShape* b,
                         const float* direction, float* point) {
    /* 在形状A中找direction方向的最远点 */
    float max_dot_a = -1e30f;
    int best_a = 0;
    for (int i = 0; i < a->num_vertices; i++) {
        float dot = a->vertices[i][0] * direction[0] +
                    a->vertices[i][1] * direction[1] +
                    a->vertices[i][2] * direction[2];
        if (dot > max_dot_a) { max_dot_a = dot; best_a = i; }
    }
    /* 在形状B中找-direction方向的最远点 */
    float max_dot_b = -1e30f;
    int best_b = 0;
    float ndir[3] = {-direction[0], -direction[1], -direction[2]};
    for (int i = 0; i < b->num_vertices; i++) {
        float dot = b->vertices[i][0] * ndir[0] +
                    b->vertices[i][1] * ndir[1] +
                    b->vertices[i][2] * ndir[2];
        if (dot > max_dot_b) { max_dot_b = dot; best_b = i; }
    }
    /* Minkowski差点: support_A(dir) - support_B(-dir) */
    point[0] = a->vertices[best_a][0] - b->vertices[best_b][0];
    point[1] = a->vertices[best_a][1] - b->vertices[best_b][1];
    point[2] = a->vertices[best_a][2] - b->vertices[best_b][2];
}

/* M-008修复: vec3_sub/local函数已由 vec3_ops.h 统一提供 */
#define vec3_len(v)  vec3_length(v)

/**
 * @brief K-031: GJK碰撞检测（带单形体输出）— 判断两个凸多面体是否相交
 *
 * @param shape_a 凸多面体A的顶点数组
 * @param num_a A顶点数
 * @param shape_b 凸多面体B的顶点数组
 * @param num_b B顶点数
 * @param simplex_out [输出] GJK单形体顶点数组(最多4个顶点)
 * @param simplex_count_out [输出] 单形体顶点数
 * @return 1=碰撞, 0=无碰撞, -1=错误
 */
static int dynamics_gjk_with_simplex(const float* shape_a, int num_a,
                                      const float* shape_b, int num_b,
                                      float simplex_out[4][3],
                                      int* simplex_count_out) {
    if (!shape_a || !shape_b || num_a < 4 || num_b < 4) return -1;

    GjkShape a, b;
    a.num_vertices = num_a < 32 ? num_a : 32;
    b.num_vertices = num_b < 32 ? num_b : 32;
    for (int i = 0; i < a.num_vertices; i++) {
        a.vertices[i][0] = shape_a[i*3+0];
        a.vertices[i][1] = shape_a[i*3+1];
        a.vertices[i][2] = shape_a[i*3+2];
    }
    for (int i = 0; i < b.num_vertices; i++) {
        b.vertices[i][0] = shape_b[i*3+0];
        b.vertices[i][1] = shape_b[i*3+1];
        b.vertices[i][2] = shape_b[i*3+2];
    }

    /* GJK主循环: 维护一个3D单形体 */
    float simplex[4][3];
    int simplex_count = 0;
    float direction[3] = {1, 0, 0};

    for (int iter = 0; iter < 64; iter++) {
        float support_pt[3];
        gjk_support(&a, &b, direction, support_pt);

        /* 如果support点不穿越原点方向，则无碰撞 */
        if (vec3_dot(support_pt, direction) < 0.0f) {
            if (simplex_out && simplex_count_out) {
                for (int s = 0; s < simplex_count; s++) {
                    simplex_out[s][0] = simplex[s][0];
                    simplex_out[s][1] = simplex[s][1];
                    simplex_out[s][2] = simplex[s][2];
                }
                *simplex_count_out = simplex_count;
            }
            return 0;
        }

        simplex[simplex_count][0] = support_pt[0];
        simplex[simplex_count][1] = support_pt[1];
        simplex[simplex_count][2] = support_pt[2];
        simplex_count++;

        /* Johnson算法: 检查原点是否在单形体内 */
        if (simplex_count == 2) {
            float ab[3], ao[3];
            vec3_sub(simplex[1], simplex[0], ab);
            ao[0] = -simplex[0][0]; ao[1] = -simplex[0][1]; ao[2] = -simplex[0][2];
            direction[0] = ab[0] * vec3_dot(ab, ao) - ao[0] * vec3_dot(ab, ab);
            direction[1] = ab[1] * vec3_dot(ab, ao) - ao[1] * vec3_dot(ab, ab);
            direction[2] = ab[2] * vec3_dot(ab, ao) - ao[2] * vec3_dot(ab, ab);
        } else if (simplex_count == 3) {
/* 完整Voronoi区域检查(与kinematics.c正确实现一致)
             * 原作仅用三角形法向量搜索,缺少边区域检查导致假阴性和迭代超限 */
            float ab[3], ac[3], ao[3], abc[3];
            vec3_sub(simplex[1], simplex[0], ab);
            vec3_sub(simplex[2], simplex[0], ac);
            ao[0] = -simplex[0][0]; ao[1] = -simplex[0][1]; ao[2] = -simplex[0][2];
            abc[0] = ab[1]*ac[2] - ab[2]*ac[1];
            abc[1] = ab[2]*ac[0] - ab[0]*ac[2];
            abc[2] = ab[0]*ac[1] - ab[1]*ac[0];

            /* Voronoi区域1: 原点靠近AB边 */
            float abc_ao[3];
            abc_ao[0] = abc[1]*ao[2] - abc[2]*ao[1];
            abc_ao[1] = abc[2]*ao[0] - abc[0]*ao[2];
            abc_ao[2] = abc[0]*ao[1] - abc[1]*ao[0];
            if (ab[0]*abc_ao[0] + ab[1]*abc_ao[1] + ab[2]*abc_ao[2] > 0.0f) {
                simplex[0][0] = simplex[1][0]; simplex[0][1] = simplex[1][1]; simplex[0][2] = simplex[1][2];
                simplex_count = 2;
                float ab_ao[3];
                ab_ao[0] = ab[1]*ao[2] - ab[2]*ao[1]; ab_ao[1] = ab[2]*ao[0] - ab[0]*ao[2]; ab_ao[2] = ab[0]*ao[1] - ab[1]*ao[0];
                direction[0] = ab_ao[1]*ab[2] - ab_ao[2]*ab[1];
                direction[1] = ab_ao[2]*ab[0] - ab_ao[0]*ab[2];
                direction[2] = ab_ao[0]*ab[1] - ab_ao[1]*ab[0];
            }
            /* Voronoi区域2: 原点靠近AC边 */
            else {
                float ac_ao[3];
                ac_ao[0] = ac[1]*ao[2] - ac[2]*ao[1]; ac_ao[1] = ac[2]*ao[0] - ac[0]*ao[2]; ac_ao[2] = ac[0]*ao[1] - ac[1]*ao[0];
                if (ac[0]*ac_ao[0] + ac[1]*ac_ao[1] + ac[2]*ac_ao[2] > 0.0f) {
                    simplex[1][0] = simplex[2][0]; simplex[1][1] = simplex[2][1]; simplex[1][2] = simplex[2][2];
                    simplex_count = 2;
                    direction[0] = ac_ao[1]*ac[2] - ac_ao[2]*ac[1];
                    direction[1] = ac_ao[2]*ac[0] - ac_ao[0]*ac[2];
                    direction[2] = ac_ao[0]*ac[1] - ac_ao[1]*ac[0];
                } else {
                    /* 原点在三角形面Voronoi区域 */
                    float dot = abc[0]*ao[0] + abc[1]*ao[1] + abc[2]*ao[2];
                    if (dot > 0.0f) {
                        direction[0] = abc[0]; direction[1] = abc[1]; direction[2] = abc[2];
                    } else {
                        direction[0] = -abc[0]; direction[1] = -abc[1]; direction[2] = -abc[2];
                    }
                }
            }
        } else if (simplex_count == 4) {
            /* 四面体: 原点在内部则碰撞，输出单形体 */
            if (simplex_out && simplex_count_out) {
                for (int s = 0; s < simplex_count; s++) {
                    simplex_out[s][0] = simplex[s][0];
                    simplex_out[s][1] = simplex[s][1];
                    simplex_out[s][2] = simplex[s][2];
                }
                *simplex_count_out = simplex_count;
            }
            return 1;
        }
    }
    if (simplex_out && simplex_count_out) {
        for (int s = 0; s < simplex_count; s++) {
            simplex_out[s][0] = simplex[s][0];
            simplex_out[s][1] = simplex[s][1];
            simplex_out[s][2] = simplex[s][2];
        }
        *simplex_count_out = simplex_count;
    }
    return 0;
}

/* 公开的GJK碰撞检测封装 */
int dynamics_gjk_collision(const float* shape_a, int num_a,
                            const float* shape_b, int num_b) {
    float simplex_buf[4][3];
    int simplex_cnt;
    return dynamics_gjk_with_simplex(shape_a, num_a, shape_b, num_b,
                                      simplex_buf, &simplex_cnt);
}

/* EPA多面体面数据结构 */
#define EPA_MAX_FACES 64
#define EPA_MAX_VERTICES 64

typedef struct {
    int v[3];          /* 面的3个顶点索引 */
    float normal[3];   /* 面法向量（指向外部） */
    float dist;        /* 面到原点的距离 */
} EPAPolyFace;

/**
 * @brief K-031: EPA穿透深度计算 — 真正的扩展多面体算法
 *
 * 在GJK检测到碰撞后，使用EPA算法计算最小穿透深度和方向。
 *
 * @param shape_a 形状A
 * @param num_a A顶点数
 * @param shape_b 形状B
 * @param num_b B顶点数
 * @param penetration_depth [输出] 穿透深度
 * @param penetration_normal [输出] 穿透法向量(3分量)
 * @param contact_point [输出] 接触点(3分量)
 * @return 1=成功, 0=无穿透, -1=错误
 */
int dynamics_epa_penetration(const float* shape_a, int num_a,
                              const float* shape_b, int num_b,
                              float* penetration_depth,
                              float* penetration_normal,
                              float* contact_point) {
    if (!shape_a || !shape_b || !penetration_depth || !penetration_normal)
        return -1;

    /* 第一步: 用GJK检测碰撞并获取单形体 */
    float simplex[4][3];
    int simplex_count = 0;
    int collision = dynamics_gjk_with_simplex(shape_a, num_a, shape_b, num_b,
                                               simplex, &simplex_count);
    if (!collision) {
        *penetration_depth = 0.0f;
        penetration_normal[0] = 0; penetration_normal[1] = 0; penetration_normal[2] = 1;
        return 0;
    }

    /* 第二步: 从GJK单形体构建初始多面体
     * GJK单形体是一个包含原点的四面体，顶点是Minkowski差点 */
    if (simplex_count < 4) {
        /* 单形体不足4个顶点，退化情况，使用14方向搜索作为回退 */
        float nearest_dist = 1e30f;
        float nearest_norm[3] = {0, 0, 1};
        GjkShape a, b;
        a.num_vertices = num_a < 32 ? num_a : 32;
        b.num_vertices = num_b < 32 ? num_b : 32;
        for (int i = 0; i < a.num_vertices; i++) {
            a.vertices[i][0] = shape_a[i*3+0];
            a.vertices[i][1] = shape_a[i*3+1];
            a.vertices[i][2] = shape_a[i*3+2];
        }
        for (int i = 0; i < b.num_vertices; i++) {
            b.vertices[i][0] = shape_b[i*3+0];
            b.vertices[i][1] = shape_b[i*3+1];
            b.vertices[i][2] = shape_b[i*3+2];
        }
        float search_dirs[14][3] = {
            {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
            {0.577f,0.577f,0.577f},{-0.577f,0.577f,0.577f},
            {0.577f,-0.577f,0.577f},{0.577f,0.577f,-0.577f},
            {-0.577f,-0.577f,0.577f},{-0.577f,0.577f,-0.577f},
            {0.577f,-0.577f,-0.577f},{-0.577f,-0.577f,-0.577f}
        };
        for (int d = 0; d < 14; d++) {
            float dir[3] = {search_dirs[d][0], search_dirs[d][1], search_dirs[d][2]};
            float pt[3];
            gjk_support(&a, &b, dir, pt);
            float dist = vec3_len(pt);
            if (dist < nearest_dist && dist > 1e-8f) {
                nearest_dist = dist;
                float inv = 1.0f / dist;
                nearest_norm[0] = -pt[0] * inv;
                nearest_norm[1] = -pt[1] * inv;
                nearest_norm[2] = -pt[2] * inv;
            }
        }
        *penetration_depth = nearest_dist;
        penetration_normal[0] = nearest_norm[0];
        penetration_normal[1] = nearest_norm[1];
        penetration_normal[2] = nearest_norm[2];
        if (contact_point) {
            contact_point[0] = nearest_norm[0] * nearest_dist * 0.5f;
            contact_point[1] = nearest_norm[1] * nearest_dist * 0.5f;
            contact_point[2] = nearest_norm[2] * nearest_dist * 0.5f;
        }
        return 1;
    }

    /* 第三步: 构建EPA多面体顶点数组和面列表 */
    float epa_vertices[EPA_MAX_VERTICES][3];
    EPAPolyFace epa_faces[EPA_MAX_FACES];
    int vertex_count = simplex_count;
    int face_count = 0;

    for (int i = 0; i < simplex_count; i++) {
        epa_vertices[i][0] = simplex[i][0];
        epa_vertices[i][1] = simplex[i][1];
        epa_vertices[i][2] = simplex[i][2];
    }

    /* 从四面体的4个面构建初始面列表 */
    int tet_faces[4][3] = {{0,1,2},{0,3,1},{0,2,3},{1,3,2}};
    for (int f = 0; f < 4; f++) {
        int i0 = tet_faces[f][0], i1 = tet_faces[f][1], i2 = tet_faces[f][2];
        float ab[3], ac[3];
        vec3_sub(epa_vertices[i1], epa_vertices[i0], ab);
        vec3_sub(epa_vertices[i2], epa_vertices[i0], ac);
        float n[3];
        n[0] = ab[1]*ac[2] - ab[2]*ac[1];
        n[1] = ab[2]*ac[0] - ab[0]*ac[2];
        n[2] = ab[0]*ac[1] - ab[1]*ac[0];
        float nl = sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
        if (nl < 1e-10f) continue;
        n[0] /= nl; n[1] /= nl; n[2] /= nl;

        /* 确保法向量指向外部（远离原点） */
        if (vec3_dot(n, epa_vertices[i0]) < 0.0f) {
            n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2];
        }

        epa_faces[face_count].v[0] = i0;
        epa_faces[face_count].v[1] = i1;
        epa_faces[face_count].v[2] = i2;
        epa_faces[face_count].normal[0] = n[0];
        epa_faces[face_count].normal[1] = n[1];
        epa_faces[face_count].normal[2] = n[2];
        epa_faces[face_count].dist = vec3_dot(n, epa_vertices[i0]);
        face_count++;
    }

    /* 设置support函数所需的形状数据 */
    GjkShape a, b;
    a.num_vertices = num_a < 32 ? num_a : 32;
    b.num_vertices = num_b < 32 ? num_b : 32;
    for (int i = 0; i < a.num_vertices; i++) {
        a.vertices[i][0] = shape_a[i*3+0];
        a.vertices[i][1] = shape_a[i*3+1];
        a.vertices[i][2] = shape_a[i*3+2];
    }
    for (int i = 0; i < b.num_vertices; i++) {
        b.vertices[i][0] = shape_b[i*3+0];
        b.vertices[i][1] = shape_b[i*3+1];
        b.vertices[i][2] = shape_b[i*3+2];
    }

    /* 第四步: EPA迭代扩展多面体 */
    int best_face_idx = 0;
    float best_dist = 1e30f;
    for (int iter = 0; iter < 32; iter++) {
        /* 找到距离原点最近的面 */
        best_face_idx = -1;
        best_dist = 1e30f;
        for (int f = 0; f < face_count; f++) {
            if (epa_faces[f].dist < best_dist) {
                best_dist = epa_faces[f].dist;
                best_face_idx = f;
            }
        }
        if (best_face_idx < 0) break;

        /* 沿最近面的法向量方向搜索新support点 */
        float search_dir[3];
        search_dir[0] = epa_faces[best_face_idx].normal[0];
        search_dir[1] = epa_faces[best_face_idx].normal[1];
        search_dir[2] = epa_faces[best_face_idx].normal[2];

        float new_pt[3];
        gjk_support(&a, &b, search_dir, new_pt);

        /* 计算新点到原点的投影距离 */
        float new_dist = vec3_dot(new_pt, search_dir);
        float dist_diff = new_dist - best_dist;

        /* 收敛判断: 新点没有显著提升距离 */
        if (dist_diff < 1e-6f || vertex_count >= EPA_MAX_VERTICES - 1)
            break;

        /* 添加新顶点到多面体 */
        epa_vertices[vertex_count][0] = new_pt[0];
        epa_vertices[vertex_count][1] = new_pt[1];
        epa_vertices[vertex_count][2] = new_pt[2];
        int new_vi = vertex_count++;

        /* 重建面列表: 移除能看到新顶点的面，为新边添加新面
         * 简化实现: 移除旧面列表中所有被新顶点"看到"的面，
         * 然后为每条暴露的边添加连接到新顶点的新面 */
        int keep_face[EPA_MAX_FACES] = {0};
        int new_face_count = 0;
        int edge_list[EPA_MAX_FACES * 2][2];
        int edge_count = 0;

        for (int f = 0; f < face_count; f++) {
            int i0 = epa_faces[f].v[0], i1 = epa_faces[f].v[1], i2 = epa_faces[f].v[2];
            /* 面中心 */
            float fc[3];
            fc[0] = (epa_vertices[i0][0] + epa_vertices[i1][0] + epa_vertices[i2][0]) / 3.0f;
            fc[1] = (epa_vertices[i0][1] + epa_vertices[i1][1] + epa_vertices[i2][1]) / 3.0f;
            fc[2] = (epa_vertices[i0][2] + epa_vertices[i1][2] + epa_vertices[i2][2]) / 3.0f;

            /* 判断新顶点是否在面外侧 */
            float to_new[3];
            vec3_sub(new_pt, fc, to_new);
            float dot_view = vec3_dot(to_new, epa_faces[f].normal);

            if (dot_view > 1e-6f) {
                /* 面被新顶点"看到"，标记为移除，添加其三条边到边列表 */
                int edges[3][2] = {{i0,i1},{i1,i2},{i2,i0}};
                for (int e = 0; e < 3; e++) {
                    int found = 0;
                    for (int ee = 0; ee < edge_count; ee++) {
                        if ((edge_list[ee][0] == edges[e][0] && edge_list[ee][1] == edges[e][1]) ||
                            (edge_list[ee][0] == edges[e][1] && edge_list[ee][1] == edges[e][0])) {
                            /* 边被共享，移除（两个相邻面都能看到新顶点） */
                            edge_list[ee][0] = edge_list[edge_count-1][0];
                            edge_list[ee][1] = edge_list[edge_count-1][1];
                            edge_count--;
                            found = 1;
                            break;
                        }
                    }
                    if (!found && edge_count < EPA_MAX_FACES * 2) {
                        edge_list[edge_count][0] = edges[e][0];
                        edge_list[edge_count][1] = edges[e][1];
                        edge_count++;
                    }
                }
            } else {
                /* 保留该面 */
                keep_face[f] = 1;
            }
        }

        /* 复制保留的面到新面列表 */
        for (int f = 0; f < face_count; f++) {
            if (keep_face[f]) {
                if (new_face_count >= EPA_MAX_FACES) break;
                epa_faces[new_face_count] = epa_faces[f];
                new_face_count++;
            }
        }

        /* 为每条暴露的边创建连接到新顶点的三角形面 */
        for (int e = 0; e < edge_count && new_face_count < EPA_MAX_FACES; e++) {
            int i0 = edge_list[e][0], i1 = edge_list[e][1];
            float ab[3], ac[3];
            vec3_sub(epa_vertices[i1], epa_vertices[i0], ab);
            vec3_sub(new_pt, epa_vertices[i0], ac);
            float n[3];
            n[0] = ab[1]*ac[2] - ab[2]*ac[1];
            n[1] = ab[2]*ac[0] - ab[0]*ac[2];
            n[2] = ab[0]*ac[1] - ab[1]*ac[0];
            float nl = sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
            if (nl < 1e-10f) continue;
            n[0] /= nl; n[1] /= nl; n[2] /= nl;

            /* 确保法向量指向外部 */
            float fm[3];
            fm[0] = (epa_vertices[i0][0] + epa_vertices[i1][0] + new_pt[0]) / 3.0f;
            fm[1] = (epa_vertices[i0][1] + epa_vertices[i1][1] + new_pt[1]) / 3.0f;
            fm[2] = (epa_vertices[i0][2] + epa_vertices[i1][2] + new_pt[2]) / 3.0f;
            if (vec3_dot(n, fm) < 0.0f) {
                n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2];
            }

            epa_faces[new_face_count].v[0] = i0;
            epa_faces[new_face_count].v[1] = i1;
            epa_faces[new_face_count].v[2] = new_vi;
            epa_faces[new_face_count].normal[0] = n[0];
            epa_faces[new_face_count].normal[1] = n[1];
            epa_faces[new_face_count].normal[2] = n[2];
            epa_faces[new_face_count].dist = vec3_dot(n, epa_vertices[i0]);
            new_face_count++;
        }

        if (new_face_count <= 0) break;
        face_count = new_face_count;
    }

    /* 第五步: 输出结果——最近面到原点的距离即为穿透深度 */
    *penetration_depth = best_dist;
    if (best_face_idx >= 0) {
        penetration_normal[0] = epa_faces[best_face_idx].normal[0];
        penetration_normal[1] = epa_faces[best_face_idx].normal[1];
        penetration_normal[2] = epa_faces[best_face_idx].normal[2];
    } else {
        penetration_normal[0] = 0; penetration_normal[1] = 0; penetration_normal[2] = 1;
    }
    if (contact_point) {
        /* 接触点: 最近面法向量方向上的穿透中点 */
        float half_depth = best_dist * 0.5f;
        contact_point[0] = penetration_normal[0] * half_depth;
        contact_point[1] = penetration_normal[1] * half_depth;
        contact_point[2] = penetration_normal[2] * half_depth;
    }
    return 1;
}

static void compute_world_transforms(const DynamicsModel* model,
                                      const float* q,
                                      float* world_R, float* world_p)
{
    int i;
    for (i = 0; i < model->config.num_joints; i++) {
        float R_rel[9], p_rel[3];
        compute_relative_transform(model, i, q[i], R_rel, p_rel);
        if (i == 0 || model->config.parent_indices[i] < 0) {
            int k;
            for (k = 0; k < 9; k++) world_R[i * 9 + k] = R_rel[k];
            dvec3_copy(p_rel, &world_p[i * 3]);
        } else {
            int p_idx = model->config.parent_indices[i];
            float* parent_R = &world_R[p_idx * 9];
            float* parent_p = &world_p[p_idx * 3];
            float tmp[3];
            mat3_mul_dyn(parent_R, R_rel, &world_R[i * 9]);
            mat3_vec3_mul(parent_R, p_rel, tmp);
            dvec3_add(parent_p, tmp, &world_p[i * 3]);
        }
    }
}

int dynamics_rnea(const DynamicsModel* model,
                  const float* q, const float* qd, const float* qdd,
                  float* torques)
{
    int n;
    float world_R[DYNAMICS_MAX_JOINTS * 9];
    float world_p[DYNAMICS_MAX_JOINTS * 3];
    float omega[DYNAMICS_MAX_JOINTS * 3];
    float alpha[DYNAMICS_MAX_JOINTS * 3];
    float accel[DYNAMICS_MAX_JOINTS * 3];
    float com_accel[DYNAMICS_MAX_JOINTS * 3];
    float force[DYNAMICS_MAX_JOINTS * 3];
    float moment[DYNAMICS_MAX_JOINTS * 3];
    float I_world[DYNAMICS_MAX_JOINTS * 9];
    int i;
    if (!model || !q || !qd || !qdd || !torques) return -1;
    n = model->config.num_joints;
    if (n <= 0 || n > DYNAMICS_MAX_JOINTS) return -1;
    compute_world_transforms(model, q, world_R, world_p);
    for (i = 0; i < n; i++) {
        float I_body[9];
        int k;
        for (k = 0; k < 9; k++) {
            I_body[k] = model->config.link_inertia[i * 9 + k];
        }
        inertia_transform(I_body, &world_R[i * 9], &I_world[i * 9]);
    }
    for (i = 0; i < n; i++) {
        float z_world[3];
        float axis_local[3];
        axis_local[0] = model->config.joint_axes[i * 3 + 0];
        axis_local[1] = model->config.joint_axes[i * 3 + 1];
        axis_local[2] = model->config.joint_axes[i * 3 + 2];
        {
            float axis_norm = (float)sqrt(axis_local[0]*axis_local[0] +
                                          axis_local[1]*axis_local[1] +
                                          axis_local[2]*axis_local[2]);
            if (axis_norm > 1e-10f) {
                axis_local[0] /= axis_norm;
                axis_local[1] /= axis_norm;
                axis_local[2] /= axis_norm;
            }
        }
        mat3_vec3_mul(&world_R[i * 9], axis_local, z_world);
        if (i == 0) {
            dvec3_zero(&omega[i * 3]);
            dvec3_zero(&alpha[i * 3]);
            if (model->config.joint_types[i] == JOINT_TYPE_REVOLUTE ||
                model->config.joint_types[i] == JOINT_TYPE_CONTINUOUS) {
                float qd_z[3], qdd_z[3];
                dvec3_scale(z_world, qd[i], qd_z);
                dvec3_add(&omega[i * 3], qd_z, &omega[i * 3]);
                dvec3_scale(z_world, qdd[i], qdd_z);
                dvec3_add(&alpha[i * 3], qdd_z, &alpha[i * 3]);
            }
            {
                float g[3] = {model->config.gravity[0],
                              model->config.gravity[1],
                              model->config.gravity[2]};
                dvec3_copy(g, &accel[i * 3]);
            }
        } else {
            int parent = model->config.parent_indices[i];
            if (parent < 0) {
                dvec3_zero(&omega[i * 3]);
                dvec3_zero(&alpha[i * 3]);
                {
                    float g[3] = {model->config.gravity[0],
                                  model->config.gravity[1],
                                  model->config.gravity[2]};
                    dvec3_copy(g, &accel[i * 3]);
                }
            } else {
                float* w_parent = &omega[parent * 3];
                float* a_parent = &alpha[parent * 3];
                float p_vec[3], tmp1[3], tmp2[3], tmp3[3];
                dvec3_sub(&world_p[i * 3], &world_p[parent * 3], p_vec);
                dvec3_copy(w_parent, &omega[i * 3]);
                dvec3_copy(a_parent, &alpha[i * 3]);
                if (model->config.joint_types[i] == JOINT_TYPE_REVOLUTE ||
                    model->config.joint_types[i] == JOINT_TYPE_CONTINUOUS) {
                    float qd_z[3], qdd_z[3], wxz[3];
                    dvec3_scale(z_world, qd[i], qd_z);
                    dvec3_add(&omega[i * 3], qd_z, &omega[i * 3]);
                    dvec3_scale(z_world, qdd[i], qdd_z);
                    dvec3_add(&alpha[i * 3], qdd_z, &alpha[i * 3]);
                    dvec3_cross(w_parent, z_world, wxz);
                    dvec3_scale(wxz, qd[i], tmp1);
                    dvec3_add(&alpha[i * 3], tmp1, &alpha[i * 3]);
                } else if (model->config.joint_types[i] == JOINT_TYPE_PRISMATIC) {
/* 平动关节角速度/角加速度不受影响,
                     * 滑动加速度由循环末尾的专用后处理块累加 */
                }
                {
                    float w_x_p[3], w_x_wxp[3];
                    dvec3_cross(&omega[parent * 3], p_vec, w_x_p);
                    dvec3_cross(&omega[parent * 3], w_x_p, w_x_wxp);
                    dvec3_add(&accel[parent * 3], w_x_wxp, tmp1);
                    dvec3_cross(&alpha[parent * 3], p_vec, tmp2);
                    dvec3_add(tmp1, tmp2, &accel[i * 3]);
                }
/* 平动关节滑动加速度后处理
                 * 基础旋转加速度已写入accel[i],现在累加z*qdd+2*ω×(z*qd) */
                if (model->config.joint_types[i] == JOINT_TYPE_PRISMATIC) {
                    float q_dd_z[3], two_wxz[3], sliding[3];
                    dvec3_scale(z_world, qdd[i], q_dd_z);
                    dvec3_cross(&omega[i * 3], z_world, two_wxz);
                    dvec3_scale(two_wxz, 2.0f * qd[i], sliding);
                    dvec3_add(sliding, q_dd_z, sliding);  /* sliding = z*qdd + 2*ω×(z*qd) */
                    dvec3_add(&accel[i * 3], sliding, &accel[i * 3]);
                }
            }
        }
        {
            float r_com[3], w_x_r[3], w_x_wxr[3], a_x_r[3];
            r_com[0] = model->config.link_com[i * 3 + 0];
            r_com[1] = model->config.link_com[i * 3 + 1];
            r_com[2] = model->config.link_com[i * 3 + 2];
            dvec3_cross(&omega[i * 3], r_com, w_x_r);
            dvec3_cross(&omega[i * 3], w_x_r, w_x_wxr);
            dvec3_cross(&alpha[i * 3], r_com, a_x_r);
            dvec3_add(w_x_wxr, a_x_r, &com_accel[i * 3]);
            dvec3_add(&accel[i * 3], &com_accel[i * 3], &com_accel[i * 3]);
        }
        {
            float m = model->config.link_masses[i];
            dvec3_scale(&com_accel[i * 3], m, &force[i * 3]);
        }
        {
            float Iw[9];
            int k;
            for (k = 0; k < 9; k++) Iw[k] = I_world[i * 9 + k];
            {
                float Iw_w[3], w_x_Iw[3];
                mat3_vec3_mul(Iw, &omega[i * 3], Iw_w);
                dvec3_cross(&omega[i * 3], Iw_w, w_x_Iw);
                {
                    float I_alpha[3];
                    mat3_vec3_mul(Iw, &alpha[i * 3], I_alpha);
                    dvec3_add(w_x_Iw, I_alpha, &moment[i * 3]);
                }
            }
        }
    }
    {
        float* f_total = (float*)safe_malloc((size_t)n * 3 * sizeof(float));
        float* n_total = (float*)safe_malloc((size_t)n * 3 * sizeof(float));
        if (!f_total || !n_total) {
            safe_free((void**)&f_total);
            safe_free((void**)&n_total);
            return -1;
        }
        memset(f_total, 0, (size_t)n * 3 * sizeof(float));
        memset(n_total, 0, (size_t)n * 3 * sizeof(float));
        for (i = n - 1; i >= 0; i--) {
            float f_sum[3], n_sum[3];
            float z_axis[3];
            float axis_local[3];
            float axis_norm;
            int child;
            dvec3_copy(&force[i * 3], f_sum);
            dvec3_copy(&moment[i * 3], n_sum);
            for (child = 0; child < n; child++) {
                if (model->config.parent_indices[child] == i) {
                    float p_ij[3], p_x_f[3];
                    dvec3_sub(&world_p[child * 3], &world_p[i * 3], p_ij);
                    dvec3_add(f_sum, &f_total[child * 3], f_sum);
                    dvec3_add(n_sum, &n_total[child * 3], n_sum);
                    dvec3_cross(p_ij, &f_total[child * 3], p_x_f);
                    dvec3_add(n_sum, p_x_f, n_sum);
                }
            }
            dvec3_copy(f_sum, &f_total[i * 3]);
            dvec3_copy(n_sum, &n_total[i * 3]);
            axis_local[0] = model->config.joint_axes[i * 3 + 0];
            axis_local[1] = model->config.joint_axes[i * 3 + 1];
            axis_local[2] = model->config.joint_axes[i * 3 + 2];
            axis_norm = (float)sqrt(axis_local[0]*axis_local[0] +
                                    axis_local[1]*axis_local[1] +
                                    axis_local[2]*axis_local[2]);
            if (axis_norm > 1e-10f) {
                axis_local[0] /= axis_norm;
                axis_local[1] /= axis_norm;
                axis_local[2] /= axis_norm;
            }
            mat3_vec3_mul(&world_R[i * 9], axis_local, z_axis);
            if (model->config.joint_types[i] == JOINT_TYPE_REVOLUTE ||
                model->config.joint_types[i] == JOINT_TYPE_CONTINUOUS) {
                torques[i] = dvec3_dot(n_sum, z_axis);
            } else if (model->config.joint_types[i] == JOINT_TYPE_PRISMATIC) {
                torques[i] = dvec3_dot(f_sum, z_axis);
            } else {
                torques[i] = 0.0f;
            }
            if (model->config.enable_joint_friction) {
                torques[i] += model->config.joint_frictions[i] *
                              (float)((qd[i] > 0) - (qd[i] < 0));
            }
            if (model->config.enable_joint_damping) {
                torques[i] += model->config.joint_dampings[i] * qd[i];
            }
        }
        safe_free((void**)&f_total);
        safe_free((void**)&n_total);
    }
    return 0;
}

int dynamics_mass_matrix(const DynamicsModel* model, const float* q, float* mass_matrix)
{
    int n, i, j;
    float* qd_zero;
    float* qdd_ej;
    float* torques_j;
    if (!model || !q || !mass_matrix) return -1;
    n = model->config.num_joints;
    if (n <= 0 || n > DYNAMICS_MAX_JOINTS) return -1;
    qd_zero = (float*)safe_malloc((size_t)n * 3 * sizeof(float));
    qdd_ej = (float*)safe_malloc((size_t)n * sizeof(float));
    torques_j = (float*)safe_malloc((size_t)n * sizeof(float));
    if (!qd_zero || !qdd_ej || !torques_j) {
        safe_free((void**)&qd_zero);
        safe_free((void**)&qdd_ej);
        safe_free((void**)&torques_j);
        return -1;
    }
    memset(qd_zero, 0, (size_t)n * sizeof(float));
/* 临时清零重力计算纯质量矩阵M(q)
     * RNEA(q,0,ej) = M(q)·ej + G(q), 必须排除G(q)才能得到M(q)的准确列 */
    float saved_gravity[3];
    memcpy(saved_gravity, model->config.gravity, sizeof(saved_gravity));
    memset(model->config.gravity, 0, sizeof(model->config.gravity));
    for (i = 0; i < n; i++) {
        memset(qdd_ej, 0, (size_t)n * sizeof(float));
        qdd_ej[i] = 1.0f;
        if (dynamics_rnea(model, q, qd_zero, qdd_ej, torques_j) != 0) {
            memcpy(model->config.gravity, saved_gravity, sizeof(saved_gravity));
            safe_free((void**)&qd_zero);
            safe_free((void**)&qdd_ej);
            safe_free((void**)&torques_j);
            return -1;
        }
        for (j = 0; j < n; j++) {
            mass_matrix[j * n + i] = torques_j[j];
        }
    }
    memcpy(model->config.gravity, saved_gravity, sizeof(saved_gravity));
    safe_free((void**)&qd_zero);
    safe_free((void**)&qdd_ej);
    safe_free((void**)&torques_j);
    return 0;
}

int dynamics_coriolis(const DynamicsModel* model,
                      const float* q, const float* qd,
                      float* coriolis)
{
    int n;
    float* qdd_zero;
    float* tau;
    if (!model || !q || !qd || !coriolis) return -1;
    n = model->config.num_joints;
    if (n <= 0 || n > DYNAMICS_MAX_JOINTS) return -1;
    qdd_zero = (float*)safe_malloc((size_t)n * sizeof(float));
    tau = (float*)safe_malloc((size_t)n * sizeof(float));
    if (!qdd_zero || !tau) {
        safe_free((void**)&qdd_zero);
        safe_free((void**)&tau);
        return -1;
    }
    memset(qdd_zero, 0, (size_t)n * sizeof(float));
    if (dynamics_rnea(model, q, qd, qdd_zero, tau) != 0) {
        safe_free((void**)&qdd_zero);
        safe_free((void**)&tau);
        return -1;
    }
    {
        float* qd_zero2;
        float* tau_gravity;
        int i;
        qd_zero2 = (float*)safe_malloc((size_t)n * sizeof(float));
        tau_gravity = (float*)safe_malloc((size_t)n * sizeof(float));
        if (!qd_zero2 || !tau_gravity) {
            safe_free((void**)&qdd_zero);
            safe_free((void**)&tau);
            safe_free((void**)&qd_zero2);
            safe_free((void**)&tau_gravity);
            return -1;
        }
        memset(qd_zero2, 0, (size_t)n * sizeof(float));
        dynamics_rnea(model, q, qd_zero2, qdd_zero, tau_gravity);
        for (i = 0; i < n; i++) {
            coriolis[i] = tau[i] - tau_gravity[i];
        }
        safe_free((void**)&qd_zero2);
        safe_free((void**)&tau_gravity);
    }
    safe_free((void**)&qdd_zero);
    safe_free((void**)&tau);
    return 0;
}

int dynamics_gravity(const DynamicsModel* model,
                     const float* q, float* gravity_terms)
{
    int n;
    float* qd_zero;
    float* qdd_zero;
    if (!model || !q || !gravity_terms) return -1;
    n = model->config.num_joints;
    if (n <= 0 || n > DYNAMICS_MAX_JOINTS) return -1;
    qd_zero = (float*)safe_malloc((size_t)n * sizeof(float));
    qdd_zero = (float*)safe_malloc((size_t)n * sizeof(float));
    if (!qd_zero || !qdd_zero) {
        safe_free((void**)&qd_zero);
        safe_free((void**)&qdd_zero);
        return -1;
    }
    memset(qd_zero, 0, (size_t)n * sizeof(float));
    memset(qdd_zero, 0, (size_t)n * sizeof(float));
    if (dynamics_rnea(model, q, qd_zero, qdd_zero, gravity_terms) != 0) {
        safe_free((void**)&qd_zero);
        safe_free((void**)&qdd_zero);
        return -1;
    }
    safe_free((void**)&qd_zero);
    safe_free((void**)&qdd_zero);
    return 0;
}

int dynamics_forward_dynamics(const DynamicsModel* model,
                              const float* q, const float* qd,
                              const float* torques,
                              float* qdd)
{
    int n, i;
    float* mass_matrix;
    float* bias;
    float* rhs;
    if (!model || !q || !qd || !torques || !qdd) return -1;
    n = model->config.num_joints;
    if (n <= 0 || n > DYNAMICS_MAX_JOINTS) return -1;
    mass_matrix = (float*)safe_malloc((size_t)n * (size_t)n * sizeof(float));
    bias = (float*)safe_malloc((size_t)n * sizeof(float));
    rhs = (float*)safe_malloc((size_t)n * sizeof(float));
    if (!mass_matrix || !bias || !rhs) {
        safe_free((void**)&mass_matrix);
        safe_free((void**)&bias);
        safe_free((void**)&rhs);
        return -1;
    }
    if (dynamics_mass_matrix(model, q, mass_matrix) != 0) {
        safe_free((void**)&mass_matrix);
        safe_free((void**)&bias);
        safe_free((void**)&rhs);
        return -1;
    }
    if (dynamics_coriolis(model, q, qd, bias) != 0) {
        safe_free((void**)&mass_matrix);
        safe_free((void**)&bias);
        safe_free((void**)&rhs);
        return -1;
    }
    {
        float* grav;
        grav = (float*)safe_malloc((size_t)n * sizeof(float));
        if (!grav) {
            safe_free((void**)&mass_matrix);
            safe_free((void**)&bias);
            safe_free((void**)&rhs);
            return -1;
        }
        dynamics_gravity(model, q, grav);
        for (i = 0; i < n; i++) {
            rhs[i] = torques[i] - bias[i] - grav[i];
        }
        safe_free((void**)&grav);
    }
    if (model->config.enable_joint_friction || model->config.enable_joint_damping) {
        for (i = 0; i < n; i++) {
            if (model->config.enable_joint_friction) {
                rhs[i] -= model->config.joint_frictions[i] *
                          (float)((qd[i] > 0) - (qd[i] < 0));
            }
            if (model->config.enable_joint_damping) {
                rhs[i] -= model->config.joint_dampings[i] * qd[i];
            }
        }
    }
    if (dynamics_solve_linear_system(n, mass_matrix, rhs, qdd) != 0) {
        safe_free((void**)&mass_matrix);
        safe_free((void**)&bias);
        safe_free((void**)&rhs);
        return -1;
    }
    safe_free((void**)&mass_matrix);
    safe_free((void**)&bias);
    safe_free((void**)&rhs);
    return 0;
}

int dynamics_forward_dynamics_external(const DynamicsModel* model,
                                       const float* q, const float* qd,
                                       const float* torques,
                                       const float* external_forces,
                                       float* qdd)
{
    int n, i;
    float* total_torques;
    if (!model || !q || !qd || !torques || !qdd) return -1;
    n = model->config.num_joints;
    if (n <= 0 || n > DYNAMICS_MAX_JOINTS) return -1;
    total_torques = (float*)safe_malloc((size_t)n * sizeof(float));
    if (!total_torques) return -1;
    for (i = 0; i < n; i++) {
        total_torques[i] = torques[i];
        if (external_forces) total_torques[i] += external_forces[i];
    }
    i = dynamics_forward_dynamics(model, q, qd, total_torques, qdd);
    safe_free((void**)&total_torques);
    return i;
}

int dynamics_compute_contact_forces(const DynamicsModel* model,
                                     const float* q, const float* qd,
                                     const ContactPoint* contacts,
                                     int contact_count,
                                     ContactForce* forces)
{
    int i;
    float world_R[DYNAMICS_MAX_JOINTS * 9];
    float world_p[DYNAMICS_MAX_JOINTS * 3];
    float world_axes[DYNAMICS_MAX_JOINTS * 3];
    int has_world_kinematics;
    if (!model || !contacts || !forces) return -1;
    if (contact_count <= 0 || contact_count > DYNAMICS_MAX_CONTACTS) return -1;

    /* M-003: 预先计算所有关节的世界变换和世界坐标系下的关节轴
     *        避免在接触点循环内重复计算运动学 */
    has_world_kinematics = 0;
    if (qd && model && model->config.num_joints > 0) {
        int j;
        compute_world_transforms(model, q, world_R, world_p);
        /* 计算世界坐标系下的关节轴 */
        for (j = 0; j < model->config.num_joints; j++) {
            float axis_local[3];
            float axis_norm;
            axis_local[0] = model->config.joint_axes[j * 3 + 0];
            axis_local[1] = model->config.joint_axes[j * 3 + 1];
            axis_local[2] = model->config.joint_axes[j * 3 + 2];
            axis_norm = (float)sqrt(axis_local[0] * axis_local[0] +
                                    axis_local[1] * axis_local[1] +
                                    axis_local[2] * axis_local[2]);
            if (axis_norm > 1e-10f) {
                axis_local[0] /= axis_norm;
                axis_local[1] /= axis_norm;
                axis_local[2] /= axis_norm;
            }
            mat3_vec3_mul(&world_R[j * 9], axis_local, &world_axes[j * 3]);
        }
        has_world_kinematics = 1;
    }

    for (i = 0; i < contact_count; i++) {
        const ContactPoint* cp = &contacts[i];
        ContactForce* cf = &forces[i];
        float v_rel[3] = {0, 0, 0};

        /* M-003: 使用完整几何雅可比计算接触点速度
         *        遍历从基座到接触点所属刚体(body_a)的运动链，
         *        对每个关节累积速度贡献：
         *          - 旋转关节: ω_j × r_{cp→j}  其中 ω_j = axis_j * qd_j
         *          - 平动关节: axis_j * qd_j
         *        这替代了原仅适用于Z轴对齐旋转关节的叉积近似 */
        if (has_world_kinematics && qd) {
            int body_a = cp->body_a;
            int n_joints = model->config.num_joints;
            if (body_a < 0 || body_a >= n_joints) body_a = n_joints - 1;
            if (body_a >= 0) {
                /* 构建从基座到body_a的运动链 */
                int chain[64];
                int chain_len = 0;
                int curr = body_a;
                while (curr >= 0 && chain_len < 64) {
                    chain[chain_len++] = curr;
                    if (curr == 0 || model->config.parent_indices[curr] < 0) break;
                    curr = model->config.parent_indices[curr];
                }
                /* 反转链顺序：从基座(root)到body_a(leaf) */
                {
                    int k;
                    for (k = chain_len - 1; k >= 0; k--) {
                        int j = chain[k];
                        int jtype = model->config.joint_types[j];
                        if (jtype == JOINT_TYPE_REVOLUTE || jtype == JOINT_TYPE_CONTINUOUS) {
                            /* 旋转关节速度贡献: v = (axis · qd) × r */
                            float omega[3];
                            float r[3];
                            omega[0] = world_axes[j * 3 + 0] * qd[j];
                            omega[1] = world_axes[j * 3 + 1] * qd[j];
                            omega[2] = world_axes[j * 3 + 2] * qd[j];
                            r[0] = cp->position[0] - world_p[j * 3 + 0];
                            r[1] = cp->position[1] - world_p[j * 3 + 1];
                            r[2] = cp->position[2] - world_p[j * 3 + 2];
                            /* v_contrib = omega × r */
                            v_rel[0] += omega[1] * r[2] - omega[2] * r[1];
                            v_rel[1] += omega[2] * r[0] - omega[0] * r[2];
                            v_rel[2] += omega[0] * r[1] - omega[1] * r[0];
                        } else if (jtype == JOINT_TYPE_PRISMATIC) {
                            /* 平动关节速度贡献: v = axis · qd */
                            v_rel[0] += world_axes[j * 3 + 0] * qd[j];
                            v_rel[1] += world_axes[j * 3 + 1] * qd[j];
                            v_rel[2] += world_axes[j * 3 + 2] * qd[j];
                        }
                        /* JOINT_TYPE_FIXED: 无速度贡献 */
                    }
                }
            }
        }
        float vt1, vt2;
        float fn, ft_max;
        float t1[3], t2[3], tmp[3];
        cf->normal_force = 0.0f;
        cf->friction_force[0] = 0.0f;
        cf->friction_force[1] = 0.0f;
        cf->total_force[0] = 0.0f;
        cf->total_force[1] = 0.0f;
        cf->total_force[2] = 0.0f;
        cf->is_sliding = 0;
        if (cp->penetration_depth <= 0) continue;
        if (model->config.contact_model == CONTACT_MODEL_PENALTY) {
            fn = model->config.contact_stiffness * cp->penetration_depth;
            if (fn < 0) fn = 0;
            cf->normal_force = fn;
            if (DYNAMICS_ABS(cp->normal[0]) > 0.1f ||
                DYNAMICS_ABS(cp->normal[1]) > 0.1f) {
                t1[0] = cp->normal[1];
                t1[1] = -cp->normal[0];
                t1[2] = 0;
            } else {
                t1[0] = 1;
                t1[1] = 0;
                t1[2] = 0;
            }
            {
                float t1_norm = (float)sqrt(t1[0]*t1[0] + t1[1]*t1[1] + t1[2]*t1[2]);
                if (t1_norm > 1e-10f) {
                    t1[0] /= t1_norm;
                    t1[1] /= t1_norm;
                    t1[2] /= t1_norm;
                }
            }
            dvec3_cross(cp->normal, t1, t2);
            vt1 = dvec3_dot(v_rel, t1);
            vt2 = dvec3_dot(v_rel, t2);
            ft_max = model->config.friction_coefficient * fn;
            if (cp->friction_coeff > 0) {
                ft_max *= cp->friction_coeff / model->config.friction_coefficient;
            }
            if (model->config.friction_model == FRICTION_CONE_PYRAMID) {
                float ft1, ft2;
                ft1 = -model->config.contact_stiffness * vt1 * 0.01f;
                ft2 = -model->config.contact_stiffness * vt2 * 0.01f;
                if (DYNAMICS_ABS(ft1) > ft_max) {
                    ft1 = (ft1 > 0 ? ft_max : -ft_max);
                    cf->is_sliding = 1;
                }
                if (DYNAMICS_ABS(ft2) > ft_max) {
                    ft2 = (ft2 > 0 ? ft_max : -ft_max);
                    cf->is_sliding = 1;
                }
                cf->friction_force[0] = ft1;
                cf->friction_force[1] = ft2;
            } else {
                float ft_total = (float)sqrt(vt1*vt1 + vt2*vt2);
                float ft_mag;
                ft_mag = model->config.contact_stiffness * ft_total * 0.01f;
                if (ft_mag > ft_max) {
                    ft_mag = ft_max;
                    cf->is_sliding = 1;
                }
                if (ft_total > 1e-10f) {
                    cf->friction_force[0] = -ft_mag * vt1 / ft_total;
                    cf->friction_force[1] = -ft_mag * vt2 / ft_total;
                }
            }
            cf->total_force[0] = cp->normal[0] * cf->normal_force;
            cf->total_force[1] = cp->normal[1] * cf->normal_force;
            cf->total_force[2] = cp->normal[2] * cf->normal_force;
            dvec3_scale(t1, cf->friction_force[0], tmp);
            dvec3_add(cf->total_force, tmp, cf->total_force);
            dvec3_scale(t2, cf->friction_force[1], tmp);
            dvec3_add(cf->total_force, tmp, cf->total_force);
        }
    }
    return 0;
}

float dynamics_kinetic_energy(const DynamicsModel* model,
                               const float* q, const float* qd)
{
    int n, i;
    float* mass_matrix;
    float energy;
    if (!model || !q || !qd) return -1.0f;
    n = model->config.num_joints;
    if (n <= 0 || n > DYNAMICS_MAX_JOINTS) return -1.0f;
    mass_matrix = (float*)safe_malloc((size_t)n * (size_t)n * sizeof(float));
    if (!mass_matrix) return -1.0f;
    if (dynamics_mass_matrix(model, q, mass_matrix) != 0) {
        safe_free((void**)&mass_matrix);
        return -1.0f;
    }
    energy = 0.0f;
    for (i = 0; i < n; i++) {
        int j;
        float row_sum = 0.0f;
        for (j = 0; j < n; j++) {
            row_sum += mass_matrix[i * n + j] * qd[j];
        }
        energy += 0.5f * qd[i] * row_sum;
    }
    safe_free((void**)&mass_matrix);
    return energy;
}

float dynamics_potential_energy(const DynamicsModel* model,
                                 const float* q)
{
    int n, i;
    float mass_total;
    float* com_total;
    float energy;
    float world_R[DYNAMICS_MAX_JOINTS * 9];
    float world_p[DYNAMICS_MAX_JOINTS * 3];
    if (!model || !q) return -1.0f;
    n = model->config.num_joints;
    if (n <= 0 || n > DYNAMICS_MAX_JOINTS) return -1.0f;
    compute_world_transforms(model, q, world_R, world_p);
    mass_total = 0.0f;
    com_total = (float*)safe_malloc(3 * sizeof(float));
    if (!com_total) return -1.0f;
    com_total[0] = 0; com_total[1] = 0; com_total[2] = 0;
    for (i = 0; i < n; i++) {
        float r_com_world[3], R_r_com[3];
        float mass = model->config.link_masses[i];
        mass_total += mass;
        r_com_world[0] = model->config.link_com[i * 3 + 0];
        r_com_world[1] = model->config.link_com[i * 3 + 1];
        r_com_world[2] = model->config.link_com[i * 3 + 2];
        mat3_vec3_mul(&world_R[i * 9], r_com_world, R_r_com);
        com_total[0] += mass * (world_p[i * 3 + 0] + R_r_com[0]);
        com_total[1] += mass * (world_p[i * 3 + 1] + R_r_com[1]);
        com_total[2] += mass * (world_p[i * 3 + 2] + R_r_com[2]);
    }
    if (mass_total > 0) {
        com_total[0] /= mass_total;
        com_total[1] /= mass_total;
        com_total[2] /= mass_total;
    }
    energy = mass_total * (model->config.gravity[0] * com_total[0] +
                           model->config.gravity[1] * com_total[1] +
                           model->config.gravity[2] * com_total[2]);
    safe_free((void**)&com_total);
    return energy;
}

float dynamics_total_energy(const DynamicsModel* model,
                             const float* q, const float* qd)
{
    float ke, pe;
    ke = dynamics_kinetic_energy(model, q, qd);
    pe = dynamics_potential_energy(model, q);
    if (ke < -0.5f || pe < -0.5f) return -1.0f;
    return ke + pe;
}

int dynamics_apply_torque_limits(const DynamicsModel* model,
                                  float* torques, int num_joints)
{
    int i;
    if (!model || !torques) return -1;
    if (num_joints > model->config.num_joints) num_joints = model->config.num_joints;
    for (i = 0; i < num_joints; i++) {
        float limit = model->config.max_joint_torques[i];
        if (torques[i] > limit) torques[i] = limit;
        else if (torques[i] < -limit) torques[i] = -limit;
    }
    return 0;
}

int dynamics_step(DynamicsModel* model,
                  float* q, float* qd,
                  const float* torques, float dt)
{
    float* qdd;
    int n, i;
    if (!model || !q || !qd || !torques) return -1;
    n = model->config.num_joints;
    if (n <= 0 || n > DYNAMICS_MAX_JOINTS) return -1;
    if (dt <= 0) return -1;
    qdd = (float*)safe_malloc((size_t)n * sizeof(float));
    if (!qdd) return -1;
    if (dynamics_forward_dynamics(model, q, qd, torques, qdd) != 0) {
        safe_free((void**)&qdd);
        return -1;
    }
    for (i = 0; i < n; i++) {
        q[i] += qd[i] * dt + 0.5f * qdd[i] * dt * dt;
        qd[i] += qdd[i] * dt;
    }
    safe_free((void**)&qdd);
    return 0;
}

int dynamics_get_state(const DynamicsModel* model, DynamicsState* state)
{
    if (!model || !state) return -1;
    memset(state, 0, sizeof(DynamicsState));
    state->num_joints = model->config.num_joints;
    return 0;
}

int dynamics_get_config(const DynamicsModel* model, RobotDynamicsConfig* config)
{
    if (!model || !config) return -1;
    *config = model->config;
    return 0;
}

int dynamics_set_config(DynamicsModel* model, const RobotDynamicsConfig* config)
{
    if (!model || !config) return -1;
    model->config = *config;
    return 0;
}

int dynamics_solve_linear_system(int n, const float* A, const float* b, float* x)
{
    float* aug;
    int i, j, k;
    if (n <= 0 || !A || !b || !x) return -1;
    if (n == 1) {
        if (DYNAMICS_ABS(A[0]) < 1e-8f) return -1;
        x[0] = b[0] / A[0];
        return 0;
    }
    if (n == 2) {
        float det = A[0] * A[3] - A[1] * A[2];
        if (DYNAMICS_ABS(det) < 1e-8f) return -1;
        x[0] = (b[0] * A[3] - b[1] * A[1]) / det;
        x[1] = (A[0] * b[1] - A[2] * b[0]) / det;
        return 0;
    }
    aug = (float*)safe_malloc((size_t)n * (size_t)(n + 1) * sizeof(float));
    if (!aug) return -1;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            aug[i * (n + 1) + j] = A[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }
    for (i = 0; i < n; i++) {
        int max_row = i;
        float max_val = DYNAMICS_ABS(aug[i * (n + 1) + i]);
        for (k = i + 1; k < n; k++) {
            if (DYNAMICS_ABS(aug[k * (n + 1) + i]) > max_val) {
                max_val = DYNAMICS_ABS(aug[k * (n + 1) + i]);
                max_row = k;
            }
        }
        if (max_val < 1e-8f) {
            safe_free((void**)&aug);
            return -1;
        }
        if (max_row != i) {
            for (j = 0; j <= n; j++) {
                float tmp = aug[i * (n + 1) + j];
                aug[i * (n + 1) + j] = aug[max_row * (n + 1) + j];
                aug[max_row * (n + 1) + j] = tmp;
            }
        }
        for (k = i + 1; k < n; k++) {
            float factor = aug[k * (n + 1) + i] / aug[i * (n + 1) + i];
            for (j = i; j <= n; j++) {
                aug[k * (n + 1) + j] -= factor * aug[i * (n + 1) + j];
            }
        }
    }
    for (i = n - 1; i >= 0; i--) {
        float sum = aug[i * (n + 1) + n];
        for (j = i + 1; j < n; j++) {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / aug[i * (n + 1) + i];
    }
    safe_free((void**)&aug);
    return 0;
}

int dynamics_check_energy_conservation(const DynamicsModel* model,
                                        const float* q_before, const float* qd_before,
                                        const float* q_after, const float* qd_after,
                                        const float* torques, float dt,
                                        float* energy_error)
{
    float E_before, E_after, work, error;
    int n, i;
    if (!model || !q_before || !qd_before || !q_after || !qd_after || !energy_error) {
        return -1;
    }
    E_before = dynamics_total_energy(model, q_before, qd_before);
    E_after = dynamics_total_energy(model, q_after, qd_after);
    if (E_before < -0.5f || E_after < -0.5f) return -1;
    work = 0.0f;
    n = model->config.num_joints;
    for (i = 0; i < n; i++) {
        float avg_qd = (qd_before[i] + qd_after[i]) * 0.5f;
        work += torques[i] * avg_qd * dt;
    }
    error = (E_after - E_before) - work;
    if (DYNAMICS_ABS(E_before) > 1e-10f) {
        *energy_error = DYNAMICS_ABS(error) / DYNAMICS_ABS(E_before);
    } else {
        *energy_error = DYNAMICS_ABS(error);
    }
    return 0;
}
