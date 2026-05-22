#include "selflnn/robot/kinematics.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"   /* F-030: 显式包含四元数函数 */
#include "selflnn/core/errors.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/* 线程安全TLS宏 */
#if defined(_MSC_VER)
#define LAI_THREAD_LOCAL __declspec(thread)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define LAI_THREAD_LOCAL _Thread_local
#else
#define LAI_THREAD_LOCAL __thread
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define DEG2RAD(d) ((d) * M_PI / 180.0f)
#define RAD2DEG(r) ((r) * 180.0f / M_PI)
#define KINEMATICS_EPSILON 1e-8f
#define KINEMATICS_MAX(a,b) ((a)>(b)?(a):(b))
#define KINEMATICS_MIN(a,b) ((a)<(b)?(a):(b))
#define KINEMATICS_CLAMP(v,lo,hi) ((v)<(lo)?(lo):((v)>(hi)?(hi):(v)))

/* ============================================================================
 * 向量运算
 * ============================================================================ */

static void k_vec3_add(Vec3* out, const Vec3* a, const Vec3* b) {
    out->x = a->x + b->x;
    out->y = a->y + b->y;
    out->z = a->z + b->z;
}

static void k_vec3_sub(Vec3* out, const Vec3* a, const Vec3* b) {
    out->x = a->x - b->x;
    out->y = a->y - b->y;
    out->z = a->z - b->z;
}

static float k_vec3_dot(const Vec3* a, const Vec3* b) {
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

static void k_vec3_cross(Vec3* out, const Vec3* a, const Vec3* b) {
    out->x = a->y * b->z - a->z * b->y;
    out->y = a->z * b->x - a->x * b->z;
    out->z = a->x * b->y - a->y * b->x;
}

static void k_vec3_scale(Vec3* out, const Vec3* a, float s) {
    out->x = a->x * s;
    out->y = a->y * s;
    out->z = a->z * s;
}

float vec3_length(const Vec3* a) {
    return sqrtf(a->x * a->x + a->y * a->y + a->z * a->z);
}

static void k_vec3_normalize(Vec3* out, const Vec3* a) {
    float len = vec3_length(a);
    if (len < KINEMATICS_EPSILON) {
        out->x = 0.0f; out->y = 0.0f; out->z = 0.0f;
        return;
    }
    float inv = 1.0f / len;
    out->x = a->x * inv;
    out->y = a->y * inv;
    out->z = a->z * inv;
}

float vec3_distance(const Vec3* a, const Vec3* b) {
    Vec3 d;
    k_vec3_sub(&d, a, b);
    return vec3_length(&d);
}

void vec3_transform_quat(Vec3* out, const Vec3* v, const float* q) {
    float qx = q[0], qy = q[1], qz = q[2], qw = q[3];
    float x2 = qx + qx, y2 = qy + qy, z2 = qz + qz;
    float xx = qx * x2, xy = qx * y2, xz = qx * z2;
    float yy = qy * y2, yz = qy * z2, zz = qz * z2;
    float wx = qw * x2, wy = qw * y2, wz = qw * z2;
    out->x = v->x * (1.0f - (yy + zz)) + v->y * (xy - wz) + v->z * (xz + wy);
    out->y = v->x * (xy + wz) + v->y * (1.0f - (xx + zz)) + v->z * (yz - wx);
    out->z = v->x * (xz - wy) + v->y * (yz + wx) + v->z * (1.0f - (xx + yy));
}

static void mat3_mul_vec3(Vec3* out, const float* mat, const Vec3* v) {
    out->x = mat[0]*v->x + mat[1]*v->y + mat[2]*v->z;
    out->y = mat[3]*v->x + mat[4]*v->y + mat[5]*v->z;
    out->z = mat[6]*v->x + mat[7]*v->y + mat[8]*v->z;
}

/* ============================================================================
 * 碰撞结果管理
 * ============================================================================ */

void collision_result_init(CollisionResult* result, int initial_capacity) {
    result->contact_count = 0;
    result->contact_capacity = initial_capacity > 0 ? initial_capacity : 8;
    result->contacts = (CollisionContact*)safe_malloc((size_t)result->contact_capacity * sizeof(CollisionContact));
}

void collision_result_free(CollisionResult* result) {
    if (result->contacts) {
        safe_free((void**)&result->contacts);
    }
    result->contact_count = 0;
    result->contact_capacity = 0;
}

int collision_result_add(CollisionResult* result, const CollisionContact* contact) {
    if (result->contact_count >= result->contact_capacity) {
        int new_cap = result->contact_capacity * 2;
        CollisionContact* new_contacts = (CollisionContact*)safe_realloc(
            result->contacts, (size_t)new_cap * sizeof(CollisionContact));
        if (!new_contacts) return -1;
        result->contacts = new_contacts;
        result->contact_capacity = new_cap;
    }
    result->contacts[result->contact_count++] = *contact;
    return 0;
}

/* ============================================================================
 * 碰撞形状初始化
 * ============================================================================ */

void collision_shape_init_sphere(CollisionShape* shape, const Vec3* center, float radius) {
    memset(shape, 0, sizeof(CollisionShape));
    shape->type = COLLISION_SHAPE_SPHERE;
    shape->data.sphere.center = *center;
    shape->data.sphere.radius = radius;
    shape->local_transform[0] = center->x;
    shape->local_transform[1] = center->y;
    shape->local_transform[2] = center->z;
    shape->local_transform[6] = 1.0f;
}

void collision_shape_init_box(CollisionShape* shape, const Vec3* center, const Vec3* half_extents) {
    memset(shape, 0, sizeof(CollisionShape));
    shape->type = COLLISION_SHAPE_BOX;
    shape->data.box.center = *center;
    shape->data.box.half_extents = *half_extents;
    shape->local_transform[0] = center->x;
    shape->local_transform[1] = center->y;
    shape->local_transform[2] = center->z;
    shape->local_transform[6] = 1.0f;
}

void collision_shape_init_capsule(CollisionShape* shape, const Vec3* center, float radius, float height) {
    memset(shape, 0, sizeof(CollisionShape));
    shape->type = COLLISION_SHAPE_CAPSULE;
    shape->data.capsule.center = *center;
    shape->data.capsule.radius = radius;
    shape->data.capsule.height = height;
    shape->local_transform[0] = center->x;
    shape->local_transform[1] = center->y;
    shape->local_transform[2] = center->z;
    shape->local_transform[6] = 1.0f;
}

void collision_shape_init_mesh(CollisionShape* shape, Vec3* vertices, int vertex_count,
                                int* indices, int index_count) {
    memset(shape, 0, sizeof(CollisionShape));
    shape->type = COLLISION_SHAPE_MESH;
    shape->data.mesh.vertices = vertices;
    shape->data.mesh.vertex_count = vertex_count;
    shape->data.mesh.indices = indices;
    shape->data.mesh.index_count = index_count;
    shape->data.mesh.center.x = 0.0f;
    shape->data.mesh.center.y = 0.0f;
    shape->data.mesh.center.z = 0.0f;
    shape->data.mesh.bounding_radius = 0.0f;
    if (vertices && vertex_count > 0) {
        Vec3 sum = {0,0,0};
        float max_r = 0.0f;
        for (int i = 0; i < vertex_count; i++) {
            k_vec3_add(&sum, &sum, &vertices[i]);
        }
        shape->data.mesh.center.x = sum.x / vertex_count;
        shape->data.mesh.center.y = sum.y / vertex_count;
        shape->data.mesh.center.z = sum.z / vertex_count;
        for (int i = 0; i < vertex_count; i++) {
            float d = vec3_distance(&vertices[i], &shape->data.mesh.center);
            if (d > max_r) max_r = d;
        }
        shape->data.mesh.bounding_radius = max_r;
    }
    shape->local_transform[6] = 1.0f;
}

/* ============================================================================
 * GJK 碰撞检测核心
 * ============================================================================ */

static Vec3 get_support_sphere(const CollisionSphere* sphere, const Vec3* dir) {
    Vec3 n;
    k_vec3_normalize(&n, dir);
    Vec3 s;
    s.x = sphere->center.x + n.x * sphere->radius;
    s.y = sphere->center.y + n.y * sphere->radius;
    s.z = sphere->center.z + n.z * sphere->radius;
    return s;
}

static Vec3 get_support_box(const CollisionBox* box, const Vec3* dir) {
    Vec3 s;
    s.x = box->center.x + (dir->x > 0 ? box->half_extents.x : -box->half_extents.x);
    s.y = box->center.y + (dir->y > 0 ? box->half_extents.y : -box->half_extents.y);
    s.z = box->center.z + (dir->z > 0 ? box->half_extents.z : -box->half_extents.z);
    return s;
}

static Vec3 get_support_capsule(const CollisionCapsule* capsule, const Vec3* dir) {
    Vec3 n;
    k_vec3_normalize(&n, dir);
    Vec3 top = {capsule->center.x, capsule->center.y + capsule->height * 0.5f, capsule->center.z};
    Vec3 bot = {capsule->center.x, capsule->center.y - capsule->height * 0.5f, capsule->center.z};
    float dt = k_vec3_dot(&top, dir);
    float db = k_vec3_dot(&bot, dir);
    Vec3 base = dt > db ? top : bot;
    Vec3 s;
    s.x = base.x + n.x * capsule->radius;
    s.y = base.y + n.y * capsule->radius;
    s.z = base.z + n.z * capsule->radius;
    return s;
}

static Vec3 get_support_cylinder(const CollisionCylinder* cyl, const Vec3* dir) {
    Vec3 n;
    k_vec3_normalize(&n, dir);
    float half_h = cyl->height * 0.5f;
    float dy = dir->y;
    float y_comp = dy > 0 ? half_h : -half_h;
    float rad_scale = 1.0f;
    if (fabsf(dy) > 0.001f) {
        float horiz = sqrtf(KINEMATICS_MAX(0, 1 - dy*dy));
        rad_scale = horiz;
    }
    Vec3 s;
    s.x = cyl->center.x + n.x * cyl->radius * rad_scale;
    s.y = cyl->center.y + y_comp;
    s.z = cyl->center.z + n.z * cyl->radius * rad_scale;
    return s;
}

static Vec3 get_support_mesh(const CollisionMesh* mesh, const Vec3* dir) {
    Vec3 s = mesh->vertices[0];
    float max_dot = k_vec3_dot(&s, dir);
    for (int i = 1; i < mesh->vertex_count; i++) {
        float d = k_vec3_dot(&mesh->vertices[i], dir);
        if (d > max_dot) {
            max_dot = d;
            s = mesh->vertices[i];
        }
    }
    return s;
}

Vec3 collision_shape_support(const CollisionShape* shape, const Vec3* dir) {
    switch (shape->type) {
        case COLLISION_SHAPE_SPHERE:
            return get_support_sphere(&shape->data.sphere, dir);
        case COLLISION_SHAPE_BOX:
            return get_support_box(&shape->data.box, dir);
        case COLLISION_SHAPE_CAPSULE:
            return get_support_capsule(&shape->data.capsule, dir);
        case COLLISION_SHAPE_CYLINDER:
            return get_support_cylinder(&shape->data.cylinder, dir);
        case COLLISION_SHAPE_MESH:
            return get_support_mesh(&shape->data.mesh, dir);
        default: {
            Vec3 zero = {0,0,0};
            return zero;
        }
    }
}

static Vec3 get_support_minkowski_diff(const CollisionShape* a, const CollisionShape* b, const Vec3* dir) {
    Vec3 sa = collision_shape_support(a, dir);
    Vec3 neg_dir;
    neg_dir.x = -dir->x; neg_dir.y = -dir->y; neg_dir.z = -dir->z;
    Vec3 sb = collision_shape_support(b, &neg_dir);
    Vec3 diff;
    k_vec3_sub(&diff, &sa, &sb);
    return diff;
}

static void simplex_add(Simplex* s, const Vec3* p) {
    if (s->num_vertices < 4) {
        s->support[s->num_vertices] = *p;
        s->num_vertices++;
    }
}

static int simplex_contains_origin(Simplex* s, Vec3* new_dir) {
    Vec3 a = s->support[s->num_vertices - 1];
    Vec3 ao;
    ao.x = -a.x; ao.y = -a.y; ao.z = -a.z;

    if (s->num_vertices == 2) {
        Vec3 b = s->support[0];
        Vec3 ab;
        k_vec3_sub(&ab, &b, &a);
        k_vec3_cross(new_dir, &ab, &ao);
        k_vec3_cross(new_dir, new_dir, &ab);
        if (vec3_length(new_dir) < KINEMATICS_EPSILON) {
            k_vec3_scale(new_dir, &ab, -1.0f);
        }
        return 0;
    }

    if (s->num_vertices == 3) {
        Vec3 b = s->support[1], c = s->support[0];
        Vec3 ab, ac;
        k_vec3_sub(&ab, &b, &a);
        k_vec3_sub(&ac, &c, &a);
        Vec3 abc;
        k_vec3_cross(&abc, &ab, &ac);

        Vec3 abc_ao;
        k_vec3_cross(&abc_ao, &abc, &ao);
        if (k_vec3_dot(&abc_ao, &ab) > 0) {
            s->support[0] = b;
            s->num_vertices = 2;
            k_vec3_cross(new_dir, &ab, &ao);
            k_vec3_cross(new_dir, new_dir, &ab);
            return 0;
        }

        Vec3 ac_ao;
        k_vec3_cross(&ac_ao, &ac, &ao);
        if (k_vec3_dot(&ac_ao, &abc) > 0) {
            s->support[0] = c;
            s->num_vertices = 2;
            k_vec3_cross(new_dir, &ac, &ao);
            k_vec3_cross(new_dir, new_dir, &ac);
            return 0;
        }

        if (k_vec3_dot(&abc, &ao) > 0) {
            *new_dir = abc;
        } else {
            Vec3 neg_abc;
            k_vec3_scale(&neg_abc, &abc, -1.0f);
            *new_dir = neg_abc;
        }
        return 0;
    }

    if (s->num_vertices == 4) {
        Vec3 b = s->support[2], c = s->support[1], d = s->support[0];
        Vec3 ab, ac, ad;
        k_vec3_sub(&ab, &b, &a);
        k_vec3_sub(&ac, &c, &a);
        k_vec3_sub(&ad, &d, &a);
        Vec3 abc, acd, adb;
        k_vec3_cross(&abc, &ab, &ac);
        k_vec3_cross(&acd, &ac, &ad);
        k_vec3_cross(&adb, &ad, &ab);
        if (k_vec3_dot(&abc, &ao) > 0) {
            s->support[0] = b; s->support[1] = c;
            s->num_vertices = 3;
            *new_dir = abc;
            return 0;
        }
        if (k_vec3_dot(&acd, &ao) > 0) {
            s->support[0] = c; s->support[1] = d;
            s->num_vertices = 3;
            *new_dir = acd;
            return 0;
        }
        if (k_vec3_dot(&adb, &ao) > 0) {
            s->support[0] = d; s->support[1] = b;
            s->num_vertices = 3;
            *new_dir = adb;
            return 0;
        }
        return 1;
    }
    return 0;
}

int gjk_intersection(const CollisionShape* shape_a, const CollisionShape* shape_b,
                      CollisionContact* contact) {
    Simplex simplex;
    simplex.num_vertices = 0;
    Vec3 dir = {1.0f, 0.0f, 0.0f};

    Vec3 p = get_support_minkowski_diff(shape_a, shape_b, &dir);
    simplex_add(&simplex, &p);
    k_vec3_scale(&dir, &p, -1.0f);

    int intersected = 0;
    for (int iter = 0; iter < KINEMATICS_GJK_MAX_ITER; iter++) {
        p = get_support_minkowski_diff(shape_a, shape_b, &dir);
        float dot = k_vec3_dot(&p, &dir);
        if (dot < 0) {
            intersected = 0;
            break;
        }
        simplex_add(&simplex, &p);
        if (simplex_contains_origin(&simplex, &dir)) {
            intersected = 1;
            break;
        }
        if (vec3_length(&dir) < KINEMATICS_EPSILON) {
            intersected = 0;
            break;
        }
    }

    if (intersected && contact) {
        Vec3 dir_a = {1,0,0};
        Vec3 sa = collision_shape_support(shape_a, &dir_a);
        Vec3 sb = collision_shape_support(shape_b, &dir_a);
        Vec3 diff;
        k_vec3_sub(&diff, &sa, &sb);
        contact->penetration_depth = vec3_length(&diff);
        if (contact->penetration_depth < KINEMATICS_EPSILON)
            contact->penetration_depth = 0.01f;
        k_vec3_normalize(&contact->normal, &diff);
        Vec3 mid;
        k_vec3_add(&mid, &sa, &sb);
        k_vec3_scale(&contact->point, &mid, 0.5f);
        contact->shape_a_id = 0;
        contact->shape_b_id = 1;
    }

    return intersected ? 1 : 0;
}

/* ============================================================================
 * 批量碰撞检测
 * ============================================================================ */

int collision_detect(const CollisionShape* shapes, int shape_count,
                      CollisionResult* result, CollisionCallback callback, void* user_data) {
    if (!shapes || shape_count <= 0 || !result) return -1;
    int collision_count = 0;
    for (int i = 0; i < shape_count; i++) {
        for (int j = i + 1; j < shape_count; j++) {
            CollisionContact contact;
            memset(&contact, 0, sizeof(contact));
            if (gjk_intersection(&shapes[i], &shapes[j], &contact)) {
                contact.shape_a_id = i;
                contact.shape_b_id = j;
                collision_result_add(result, &contact);
                collision_count++;
                if (callback) {
                    if (!callback(i, j, &contact, user_data)) return collision_count;
                }
            }
        }
    }
    return collision_count;
}

int collision_detect_robot_self(const KinematicModel* model, const float* joint_angles,
                                 CollisionResult* result) {
    /* ZS-005修复: 使用joint_angles和正运动学计算各链节当前世界位姿 */
    if (!model || !result) return -1;
    CollisionShape shapes[KINEMATICS_MAX_LINKS];
    int shape_count = 0;

    /* 首先计算所有链节的正运动学位姿 */
    Vec3 link_positions[KINEMATICS_MAX_LINKS];
    float link_orientations[KINEMATICS_MAX_LINKS * 4];
    int use_transformed = 0;
    if (joint_angles) {
        if (forward_kinematics_full(model, joint_angles,
                                     link_positions, link_orientations) == 0) {
            use_transformed = 1;
        }
    }

    for (int i = 0; i < model->joint_count && shape_count < KINEMATICS_MAX_LINKS; i++) {
        if (model->joints[i].has_collision) {
            shapes[shape_count] = model->joints[i].collision_shape;
            if (use_transformed) {
                shapes[shape_count].local_transform[0] = link_positions[i].x;
                shapes[shape_count].local_transform[1] = link_positions[i].y;
                shapes[shape_count].local_transform[2] = link_positions[i].z;
                shapes[shape_count].local_transform[3] = link_orientations[i * 4 + 0];
                shapes[shape_count].local_transform[4] = link_orientations[i * 4 + 1];
                shapes[shape_count].local_transform[5] = link_orientations[i * 4 + 2];
                shapes[shape_count].local_transform[6] = link_orientations[i * 4 + 3];
            }
            shape_count++;
        }
    }
    return collision_detect(shapes, shape_count, result, NULL, NULL);
}

/* ============================================================================
 * 运动学模型管理
 * ============================================================================ */

void kinematic_model_init(KinematicModel* model) {
    memset(model, 0, sizeof(KinematicModel));
    model->base_orientation[3] = 1.0f;
    model->end_effector_joint = -1;
}

int kinematic_model_add_joint(KinematicModel* model, int parent_index,
                                const char* name, KinJointType type,
                                const DHParameter* dh,
                                float limit_lower, float limit_upper) {
    if (!model || model->joint_count >= KINEMATICS_MAX_JOINTS) return -1;
    int idx = model->joint_count;
    KinJoint* j = &model->joints[idx];
    memset(j, 0, sizeof(KinJoint));
    if (name) {
        strncpy(j->name, name, sizeof(j->name) - 1);
        j->name[sizeof(j->name) - 1] = '\0';
    }
    j->joint_id = idx;
    j->joint_type = type;
    j->dh = *dh;
    j->joint_limit_lower = limit_lower;
    j->joint_limit_upper = limit_upper;
    j->joint_max_velocity = 10.0f;
    j->parent_index = parent_index;
    j->child_count = 0;
    j->joint_axis[0] = 0.0f; j->joint_axis[1] = 0.0f; j->joint_axis[2] = 1.0f;
    if (type == JOINT_TYPE_REVOLUTE || type == JOINT_TYPE_CONTINUOUS) {
        j->joint_axis[2] = 1.0f;
    } else if (type == JOINT_TYPE_PRISMATIC) {
        j->joint_axis[2] = 1.0f;
    }
    if (parent_index >= 0 && parent_index < idx) {
        KinJoint* parent = &model->joints[parent_index];
        if (parent->child_count < KINEMATICS_MAX_URDF_CHILDREN) {
            parent->child_indices[parent->child_count++] = idx;
        }
    }
    model->joint_count++;
    if (type != JOINT_TYPE_FIXED) {
        model->num_dof++;
    }
    return idx;
}

void kinematic_model_set_end_effector(KinematicModel* model, int joint_index,
                                       const float* offset) {
    if (!model || joint_index < 0 || joint_index >= model->joint_count) return;
    model->end_effector_joint = joint_index;
    if (offset) {
        model->end_effector_offset[0] = offset[0];
        model->end_effector_offset[1] = offset[1];
        model->end_effector_offset[2] = offset[2];
    }
}

int kinematic_model_set_base_pose(KinematicModel* model, const Vec3* position,
                                   const float* orientation) {
    if (!model) return -1;
    if (position) model->base_position = *position;
    if (orientation) memcpy(model->base_orientation, orientation, 4 * sizeof(float));
    return 0;
}

/* ============================================================================
 * 正运动学（D-H 参数法）
 * ============================================================================ */

static void dh_transform(float* mat, float a, float alpha, float d, float theta) {
    float ca = cosf(alpha), sa = sinf(alpha);
    float ct = cosf(theta), st = sinf(theta);
    mat[0] = ct;       mat[1] = -st * ca;  mat[2] = st * sa;   mat[3] = a * ct;
    mat[4] = st;       mat[5] = ct * ca;   mat[6] = -ct * sa;  mat[7] = a * st;
    mat[8] = 0.0f;     mat[9] = sa;        mat[10] = ca;        mat[11] = d;
    mat[12] = 0.0f;    mat[13] = 0.0f;     mat[14] = 0.0f;     mat[15] = 1.0f;
}

static void mat4_mul(const float* a, const float* b, float* out) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            out[i * 4 + j] = a[i * 4 + 0] * b[0 * 4 + j]
                           + a[i * 4 + 1] * b[1 * 4 + j]
                           + a[i * 4 + 2] * b[2 * 4 + j]
                           + a[i * 4 + 3] * b[3 * 4 + j];
        }
    }
}

static void mat4_transform_point(const float* mat, const Vec3* p, Vec3* out) {
    out->x = mat[0] * p->x + mat[1] * p->y + mat[2] * p->z + mat[3];
    out->y = mat[4] * p->x + mat[5] * p->y + mat[6] * p->z + mat[7];
    out->z = mat[8] * p->x + mat[9] * p->y + mat[10] * p->z + mat[11];
}

static void mat4_extract_quat(const float* mat, float* q) {
    float trace = mat[0] + mat[5] + mat[10];
    if (trace > 0) {
        float s = sqrtf(trace + 1.0f) * 2.0f;
        q[3] = 0.25f * s;
        q[0] = (mat[6] - mat[9]) / s;
        q[1] = (mat[8] - mat[2]) / s;
        q[2] = (mat[1] - mat[4]) / s;
    } else if (mat[0] > mat[5] && mat[0] > mat[10]) {
        float s = sqrtf(1.0f + mat[0] - mat[5] - mat[10]) * 2.0f;
        q[3] = (mat[6] - mat[9]) / s;
        q[0] = 0.25f * s;
        q[1] = (mat[4] + mat[1]) / s;
        q[2] = (mat[8] + mat[2]) / s;
    } else if (mat[5] > mat[10]) {
        float s = sqrtf(1.0f + mat[5] - mat[0] - mat[10]) * 2.0f;
        q[3] = (mat[8] - mat[2]) / s;
        q[0] = (mat[4] + mat[1]) / s;
        q[1] = 0.25f * s;
        q[2] = (mat[9] + mat[6]) / s;
    } else {
        float s = sqrtf(1.0f + mat[10] - mat[0] - mat[5]) * 2.0f;
        q[3] = (mat[1] - mat[4]) / s;
        q[0] = (mat[8] + mat[2]) / s;
        q[1] = (mat[9] + mat[6]) / s;
        q[2] = 0.25f * s;
    }
    float norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (norm > 0) {
        float inv = 1.0f / norm;
        q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
    }
}

int forward_kinematics(const KinematicModel* model, const float* joint_angles,
                        EndEffectorState* result) {
    if (!model || !joint_angles || !result) return -1;
    float T[16];
    float base_mat[16];
    quat_to_matrix(model->base_orientation, base_mat);
    base_mat[3] = model->base_position.x;
    base_mat[7] = model->base_position.y;
    base_mat[11] = model->base_position.z;
    base_mat[12] = 0; base_mat[13] = 0; base_mat[14] = 0; base_mat[15] = 1;
    memcpy(T, base_mat, sizeof(T));
    int ee_idx = model->end_effector_joint;
    if (ee_idx < 0) ee_idx = model->joint_count - 1;
    if (ee_idx < 0) return -1;
    for (int i = 0; i <= ee_idx && i < model->joint_count; i++) {
        const KinJoint* j = &model->joints[i];
        float theta = j->dh.theta;
        if (j->joint_type == JOINT_TYPE_REVOLUTE || j->joint_type == JOINT_TYPE_CONTINUOUS) {
            theta += joint_angles[i];
        }
        float d = j->dh.d;
        if (j->joint_type == JOINT_TYPE_PRISMATIC) {
            d += joint_angles[i];
        }
        float Ti[16];
        dh_transform(Ti, j->dh.a, j->dh.alpha, d, theta);
        float newT[16];
        mat4_mul(T, Ti, newT);
        memcpy(T, newT, sizeof(T));
    }
    Vec3 offset = {model->end_effector_offset[0],
                   model->end_effector_offset[1],
                   model->end_effector_offset[2]};
    Vec3 ee_pos;
    mat4_transform_point(T, &offset, &ee_pos);
    result->position = ee_pos;
    mat4_extract_quat(T, result->orientation);

    /* F-041修复: 使用高精度时间戳计算速度，替代静态60Hz假设。
     * 函数内速度计算使用线程局部存储(TLS)，确保多线程安全。 */
    static LAI_THREAD_LOCAL clock_t prev_tick = 0;
    static LAI_THREAD_LOCAL float prev_pos_x = 0, prev_pos_y = 0, prev_pos_z = 0;
    
    clock_t now_tick = clock();
    double dt_sec = 0.0;
    if (prev_tick > 0) {
        dt_sec = (double)(now_tick - prev_tick) / (double)CLOCKS_PER_SEC;
    }
    
    if (dt_sec > 1e-6 && dt_sec < 1.0) { /* 合法时间间隔：1us~1s */
        result->linear_velocity.x = (ee_pos.x - prev_pos_x) / (float)dt_sec;
        result->linear_velocity.y = (ee_pos.y - prev_pos_y) / (float)dt_sec;
        result->linear_velocity.z = (ee_pos.z - prev_pos_z) / (float)dt_sec;
    } else {
        /* 首次调用或时间异常，速度设为0 */
        result->linear_velocity.x = 0;
        result->linear_velocity.y = 0;
        result->linear_velocity.z = 0;
    }
    
    prev_tick = now_tick;
    prev_pos_x = ee_pos.x; prev_pos_y = ee_pos.y; prev_pos_z = ee_pos.z;
    result->angular_velocity.x = 0;
    result->angular_velocity.y = 0;
    result->angular_velocity.z = 0;
    return 0;
}

int forward_kinematics_full(const KinematicModel* model, const float* joint_angles,
                             Vec3* link_positions, float* link_orientations) {
    if (!model || !joint_angles || !link_positions) return -1;
    float T[16];
    float base_mat[16];
    quat_to_matrix(model->base_orientation, base_mat);
    base_mat[3] = model->base_position.x;
    base_mat[7] = model->base_position.y;
    base_mat[11] = model->base_position.z;
    base_mat[12] = 0; base_mat[13] = 0; base_mat[14] = 0; base_mat[15] = 1;
    memcpy(T, base_mat, sizeof(T));
    link_positions[0].x = T[3]; link_positions[0].y = T[7]; link_positions[0].z = T[11];
    if (link_orientations) {
        mat4_extract_quat(T, link_orientations);
    }
    for (int i = 0; i < model->joint_count; i++) {
        int idx = i + 1;
        const KinJoint* j = &model->joints[i];
        float theta = j->dh.theta;
        if (j->joint_type == JOINT_TYPE_REVOLUTE || j->joint_type == JOINT_TYPE_CONTINUOUS) {
            theta += joint_angles[i];
        }
        float d = j->dh.d;
        if (j->joint_type == JOINT_TYPE_PRISMATIC) {
            d += joint_angles[i];
        }
        float Ti[16];
        dh_transform(Ti, j->dh.a, j->dh.alpha, d, theta);
        float newT[16];
        mat4_mul(T, Ti, newT);
        memcpy(T, newT, sizeof(T));
        link_positions[idx].x = T[3];
        link_positions[idx].y = T[7];
        link_positions[idx].z = T[11];
        if (link_orientations) {
            mat4_extract_quat(T, link_orientations + idx * 4);
        }
    }
    return 0;
}

/* ============================================================================
 * 雅可比矩阵计算
 * ============================================================================ */

int compute_jacobian(const KinematicModel* model, const float* joint_angles,
                      float* jacobian, int jacobian_rows) {
    if (!model || !joint_angles || !jacobian) return -1;
    int ee_idx = model->end_effector_joint;
    if (ee_idx < 0) ee_idx = model->joint_count - 1;
    if (ee_idx < 0) return -1;
    int dof = 0;
    for (int i = 0; i <= ee_idx; i++) {
        if (model->joints[i].joint_type != JOINT_TYPE_FIXED) dof++;
    }
    if (dof <= 0) return -1;
    float T[16];
    float base_mat[16];
    quat_to_matrix(model->base_orientation, base_mat);
    base_mat[3] = model->base_position.x;
    base_mat[7] = model->base_position.y;
    base_mat[11] = model->base_position.z;
    base_mat[12] = 0; base_mat[13] = 0; base_mat[14] = 0; base_mat[15] = 1;
    memcpy(T, base_mat, sizeof(T));
    Vec3 ee_pos;
    {
        float Ttmp[16];
        memcpy(Ttmp, T, sizeof(T));
        for (int i = 0; i <= ee_idx; i++) {
            const KinJoint* j = &model->joints[i];
            float theta = j->dh.theta;
            if (j->joint_type == JOINT_TYPE_REVOLUTE || j->joint_type == JOINT_TYPE_CONTINUOUS)
                theta += joint_angles[i];
            float d = j->dh.d;
            if (j->joint_type == JOINT_TYPE_PRISMATIC)
                d += joint_angles[i];
            float Ti[16];
            dh_transform(Ti, j->dh.a, j->dh.alpha, d, theta);
            float newT[16];
            mat4_mul(Ttmp, Ti, newT);
            memcpy(Ttmp, newT, sizeof(T));
        }
        Vec3 offset = {model->end_effector_offset[0],
                       model->end_effector_offset[1],
                       model->end_effector_offset[2]};
        mat4_transform_point(Ttmp, &offset, &ee_pos);
    }
    int col = 0;
    for (int i = 0; i <= ee_idx; i++) {
        const KinJoint* j = &model->joints[i];
        if (j->joint_type == JOINT_TYPE_FIXED) continue;
        float theta = j->dh.theta;
        if (j->joint_type == JOINT_TYPE_REVOLUTE || j->joint_type == JOINT_TYPE_CONTINUOUS) {
            theta += joint_angles[i];
        }
        float d = j->dh.d;
        if (j->joint_type == JOINT_TYPE_PRISMATIC) {
            d += joint_angles[i];
        }
        float Ti[16];
        dh_transform(Ti, j->dh.a, j->dh.alpha, d, theta);
        float newT[16];
        mat4_mul(T, Ti, newT);
        memcpy(T, newT, sizeof(T));
        Vec3 joint_pos = {T[3], T[7], T[11]};
        Vec3 axis = {T[0]*j->joint_axis[0] + T[4]*j->joint_axis[1] + T[8]*j->joint_axis[2],
                     T[1]*j->joint_axis[0] + T[5]*j->joint_axis[1] + T[9]*j->joint_axis[2],
                     T[2]*j->joint_axis[0] + T[6]*j->joint_axis[1] + T[10]*j->joint_axis[2]};
        if (j->joint_type == JOINT_TYPE_REVOLUTE || j->joint_type == JOINT_TYPE_CONTINUOUS) {
            Vec3 diff;
            k_vec3_sub(&diff, &ee_pos, &joint_pos);
            Vec3 jv;
            k_vec3_cross(&jv, &axis, &diff);
            if (col < jacobian_rows) {
                jacobian[col * 6 + 0] = jv.x;
                jacobian[col * 6 + 1] = jv.y;
                jacobian[col * 6 + 2] = jv.z;
                jacobian[col * 6 + 3] = axis.x;
                jacobian[col * 6 + 4] = axis.y;
                jacobian[col * 6 + 5] = axis.z;
            }
        } else {
            if (col < jacobian_rows) {
                jacobian[col * 6 + 0] = axis.x;
                jacobian[col * 6 + 1] = axis.y;
                jacobian[col * 6 + 2] = axis.z;
                jacobian[col * 6 + 3] = 0;
                jacobian[col * 6 + 4] = 0;
                jacobian[col * 6 + 5] = 0;
            }
        }
        col++;
    }
    return dof;
}

/* ============================================================================
 * 逆运动学 — CCD 算法
 * ============================================================================ */

int inverse_kinematics_ccd(const KinematicModel* model, const Vec3* target_pos,
                            const float* target_orient, float* joint_angles,
                            int max_iter, float tolerance) {
    if (!model || !target_pos || !joint_angles) return -1;
    if (max_iter <= 0) max_iter = KINEMATICS_IK_MAX_ITER;
    if (tolerance <= 0) tolerance = KINEMATICS_IK_TOLERANCE;
    int ee_idx = model->end_effector_joint;
    if (ee_idx < 0) ee_idx = model->joint_count - 1;
    if (ee_idx < 0) return -1;
    (void)target_orient;
    for (int iter = 0; iter < max_iter; iter++) {
        EndEffectorState state;
        forward_kinematics(model, joint_angles, &state);
        Vec3 to_target;
        k_vec3_sub(&to_target, target_pos, &state.position);
        float error = vec3_length(&to_target);
        if (error < tolerance) return iter + 1;
        for (int j = ee_idx; j >= 0; j--) {
            const KinJoint* joint = &model->joints[j];
            if (joint->joint_type == JOINT_TYPE_FIXED) continue;
            float T[16];
            float base_mat[16];
            quat_to_matrix(model->base_orientation, base_mat);
            base_mat[3] = model->base_position.x;
            base_mat[7] = model->base_position.y;
            base_mat[11] = model->base_position.z;
            base_mat[12] = 0; base_mat[13] = 0; base_mat[14] = 0; base_mat[15] = 1;
            memcpy(T, base_mat, sizeof(T));
            for (int k = 0; k <= j; k++) {
                const KinJoint* jk = &model->joints[k];
                float theta = jk->dh.theta;
                if (jk->joint_type == JOINT_TYPE_REVOLUTE || jk->joint_type == JOINT_TYPE_CONTINUOUS)
                    theta += joint_angles[k];
                float d = jk->dh.d;
                if (jk->joint_type == JOINT_TYPE_PRISMATIC)
                    d += joint_angles[k];
                float Ti[16];
                dh_transform(Ti, jk->dh.a, jk->dh.alpha, d, theta);
                float newT[16];
                mat4_mul(T, Ti, newT);
                memcpy(T, newT, sizeof(T));
            }
            Vec3 joint_pos = {T[3], T[7], T[11]};
            Vec3 to_ee;
            k_vec3_sub(&to_ee, &state.position, &joint_pos);
            Vec3 to_target_local;
            k_vec3_sub(&to_target_local, target_pos, &joint_pos);
            float d1 = vec3_length(&to_ee);
            float d2 = vec3_length(&to_target_local);
            if (d1 < KINEMATICS_EPSILON || d2 < KINEMATICS_EPSILON) continue;
            Vec3 dir_ee, dir_target;
            k_vec3_normalize(&dir_ee, &to_ee);
            k_vec3_normalize(&dir_target, &to_target_local);
            float cos_angle = k_vec3_dot(&dir_ee, &dir_target);
            if (cos_angle > 0.9999f) continue;
            float angle = acosf(KINEMATICS_CLAMP(cos_angle, -1.0f, 1.0f));
            Vec3 cross_axis;
            k_vec3_cross(&cross_axis, &dir_ee, &dir_target);
            float clen = vec3_length(&cross_axis);
            if (clen < KINEMATICS_EPSILON) continue;
            Vec3 axis;
            k_vec3_normalize(&axis, &cross_axis);
            Vec3 joint_axis = {T[0]*joint->joint_axis[0] + T[4]*joint->joint_axis[1] + T[8]*joint->joint_axis[2],
                               T[1]*joint->joint_axis[0] + T[5]*joint->joint_axis[1] + T[9]*joint->joint_axis[2],
                               T[2]*joint->joint_axis[0] + T[6]*joint->joint_axis[1] + T[10]*joint->joint_axis[2]};
            float ja_len = vec3_length(&joint_axis);
            if (ja_len < KINEMATICS_EPSILON) continue;
            float dot = k_vec3_dot(&axis, &joint_axis);
            if (joint->joint_type == JOINT_TYPE_REVOLUTE || joint->joint_type == JOINT_TYPE_CONTINUOUS) {
                joint_angles[j] += angle * KINEMATICS_CLAMP(dot / ja_len, -1.0f, 1.0f) * 0.5f;
            } else if (joint->joint_type == JOINT_TYPE_PRISMATIC) {
                joint_angles[j] += k_vec3_dot(&to_target, &joint_axis) * 0.3f;
            }
            if (joint->joint_type != JOINT_TYPE_CONTINUOUS) {
                joint_angles[j] = KINEMATICS_CLAMP(joint_angles[j],
                    joint->joint_limit_lower, joint->joint_limit_upper);
            }
        }
    }
    EndEffectorState final;
    forward_kinematics(model, joint_angles, &final);
    Vec3 final_diff;
    k_vec3_sub(&final_diff, target_pos, &final.position);
    return vec3_length(&final_diff) < tolerance ? max_iter : -1;
}

/* ============================================================================
 * 逆运动学 — DLS（阻尼最小二乘法）
 * ============================================================================ */

/**
 * @brief 逆运动学求解 - DLS阻尼最小二乘法（私有实现，quaternion姿态版本）
 * 
 * 本函数与 ik_solve_dls() 是两种独立的DLS实现，各有不同算法取舍：
 * - inverse_kinematics_dls: 使用四元数姿态误差、动态堆分配、固定阻尼因子
 * - ik_solve_dls: 使用欧拉角姿态误差、静态栈分配、自适应阻尼因子(lambda)
 * 两者均为有效的DLS实现，根据调用场景选择使用。
 * 
 * @param model 运动学模型
 * @param target_pos 目标末端位置
 * @param target_orient 目标姿态（四元数格式，NULL表示忽略姿态）
 * @param joint_angles [输入/输出] 当前关节角，求解结果写回
 * @param damping DLS阻尼因子
 * @param max_iter 最大迭代次数
 * @param tolerance 收敛容差
 * @return 迭代次数（成功），-1（失败）
 */
int inverse_kinematics_dls(const KinematicModel* model, const Vec3* target_pos,
                            const float* target_orient, float* joint_angles,
                            float damping, int max_iter, float tolerance) {
    if (!model || !target_pos || !joint_angles) return -1;
    if (max_iter <= 0) max_iter = KINEMATICS_IK_MAX_ITER;
    if (tolerance <= 0) tolerance = KINEMATICS_IK_TOLERANCE;
    if (damping <= 0) damping = KINEMATICS_IK_DAMPING;
    int ee_idx = model->end_effector_joint;
    if (ee_idx < 0) ee_idx = model->joint_count - 1;
    int dof = 0;
    for (int i = 0; i <= ee_idx; i++) {
        if (model->joints[i].joint_type != JOINT_TYPE_FIXED) dof++;
    }
    if (dof <= 0) return -1;
    float* jacobian = (float*)safe_malloc((size_t)dof * 6 * sizeof(float));
    float* jtj = (float*)safe_malloc((size_t)dof * dof * sizeof(float));
    float* error = (float*)safe_malloc(6 * sizeof(float));
    float* dq = (float*)safe_malloc((size_t)dof * sizeof(float));
    float* temp = (float*)safe_malloc((size_t)dof * sizeof(float));
    if (!jacobian || !jtj || !error || !dq || !temp) {
        safe_free((void**)&jacobian); safe_free((void**)&jtj);
        safe_free((void**)&error); safe_free((void**)&dq); safe_free((void**)&temp);
        return -1;
    }
    int result = -1;
    for (int iter = 0; iter < max_iter; iter++) {
        EndEffectorState state;
        forward_kinematics(model, joint_angles, &state);
        error[0] = target_pos->x - state.position.x;
        error[1] = target_pos->y - state.position.y;
        error[2] = target_pos->z - state.position.z;
        float pos_err = sqrtf(error[0]*error[0] + error[1]*error[1] + error[2]*error[2]);
        if (target_orient) {
            float q_err[4];
            quat_conjugate(q_err, state.orientation);
            float q_diff[4];
            quat_multiply(q_diff, target_orient, q_err);
            error[3] = q_diff[0] * 2.0f;
            error[4] = q_diff[1] * 2.0f;
            error[5] = q_diff[2] * 2.0f;
        }
        if (pos_err < tolerance) { result = iter + 1; break; }
        int ncols = compute_jacobian(model, joint_angles, jacobian, dof);
        if (ncols <= 0) { result = -1; break; }
        for (int i = 0; i < ncols; i++) {
            for (int j = 0; j < ncols; j++) {
                float sum = 0;
                for (int k = 0; k < 6; k++) {
                    sum += jacobian[i * 6 + k] * jacobian[j * 6 + k];
                }
                jtj[i * ncols + j] = sum;
                if (i == j) jtj[i * ncols + j] += damping;
            }
        }
        for (int i = 0; i < ncols; i++) {
            float sum = 0;
            for (int k = 0; k < 6; k++) {
                sum += jacobian[i * 6 + k] * error[k];
            }
            temp[i] = sum;
        }
        for (int i = 0; i < ncols; i++) {
            if (fabsf(jtj[i * ncols + i]) < KINEMATICS_EPSILON) {
                dq[i] = temp[i] / (jtj[i * ncols + i] + damping);
            } else {
                dq[i] = temp[i] / jtj[i * ncols + i];
            }
        }
        int j_idx = 0;
        for (int i = 0; i <= ee_idx; i++) {
            if (model->joints[i].joint_type == JOINT_TYPE_FIXED) continue;
            joint_angles[i] += dq[j_idx] * 0.5f;
            if (model->joints[i].joint_type != JOINT_TYPE_CONTINUOUS) {
                joint_angles[i] = KINEMATICS_CLAMP(joint_angles[i],
                    model->joints[i].joint_limit_lower, model->joints[i].joint_limit_upper);
            }
            j_idx++;
        }
    }
    if (result < 0) {
        EndEffectorState final;
        forward_kinematics(model, joint_angles, &final);
        Vec3 diff;
        k_vec3_sub(&diff, target_pos, &final.position);
        if (vec3_length(&diff) < tolerance * 10) result = max_iter;
    }
    safe_free((void**)&jacobian); safe_free((void**)&jtj);
    safe_free((void**)&error); safe_free((void**)&dq); safe_free((void**)&temp);
    return result;
}

/* ============================================================================
 * 逆运动学 — 雅可比转置法
 * ============================================================================ */

int inverse_kinematics_transpose(const KinematicModel* model, const Vec3* target_pos,
                                  const float* target_orient, float* joint_angles,
                                  float step_size, int max_iter, float tolerance) {
    if (!model || !target_pos || !joint_angles) return -1;
    if (max_iter <= 0) max_iter = KINEMATICS_IK_MAX_ITER;
    if (tolerance <= 0) tolerance = KINEMATICS_IK_TOLERANCE;
    if (step_size <= 0) step_size = 0.1f;
    int ee_idx = model->end_effector_joint;
    if (ee_idx < 0) ee_idx = model->joint_count - 1;
    int dof = 0;
    for (int i = 0; i <= ee_idx; i++) {
        if (model->joints[i].joint_type != JOINT_TYPE_FIXED) dof++;
    }
    if (dof <= 0) return -1;
    float* jacobian = (float*)safe_malloc((size_t)dof * 6 * sizeof(float));
    if (!jacobian) return -1;
    int result = -1;
    for (int iter = 0; iter < max_iter; iter++) {
        EndEffectorState state;
        forward_kinematics(model, joint_angles, &state);
        float dx = target_pos->x - state.position.x;
        float dy = target_pos->y - state.position.y;
        float dz = target_pos->z - state.position.z;
        float error = sqrtf(dx*dx + dy*dy + dz*dz);
        if (error < tolerance) { result = iter + 1; break; }
        int ncols = compute_jacobian(model, joint_angles, jacobian, dof);
        if (ncols <= 0) { result = -1; break; }
        (void)target_orient;
        int j_idx = 0;
        for (int i = 0; i <= ee_idx; i++) {
            if (model->joints[i].joint_type == JOINT_TYPE_FIXED) continue;
            float grad = jacobian[j_idx * 6 + 0] * dx
                       + jacobian[j_idx * 6 + 1] * dy
                       + jacobian[j_idx * 6 + 2] * dz;
            joint_angles[i] += step_size * grad;
            if (model->joints[i].joint_type != JOINT_TYPE_CONTINUOUS) {
                joint_angles[i] = KINEMATICS_CLAMP(joint_angles[i],
                    model->joints[i].joint_limit_lower, model->joints[i].joint_limit_upper);
            }
            j_idx++;
        }
    }
    safe_free((void**)&jacobian);
    return result;
}

/* ============================================================================
 * 关节约束求解器
 * ============================================================================ */

void constraint_solver_init(JointConstraintSolver* solver) {
    memset(solver, 0, sizeof(JointConstraintSolver));
}

int constraint_solver_add(JointConstraintSolver* solver, KinJointType type,
                           float lower, float upper, float max_velocity) {
    if (!solver || solver->constraint_count >= KINEMATICS_MAX_JOINTS) return -1;
    int idx = solver->constraint_count;
    solver->constraints[idx].type = type;
    solver->constraints[idx].lower = lower;
    solver->constraints[idx].upper = upper;
    solver->constraints[idx].max_velocity = max_velocity > 0 ? max_velocity : 10.0f;
    solver->constraint_count++;
    return idx;
}

void constraint_solver_apply(const JointConstraintSolver* solver,
                              float* joint_angles, int joint_count) {
    if (!solver || !joint_angles) return;
    int n = solver->constraint_count < joint_count ? solver->constraint_count : joint_count;
    for (int i = 0; i < n; i++) {
        if (solver->constraints[i].type == JOINT_TYPE_CONTINUOUS) {
            while (joint_angles[i] > M_PI) joint_angles[i] -= 2.0f * (float)M_PI;
            while (joint_angles[i] < -M_PI) joint_angles[i] += 2.0f * (float)M_PI;
        } else {
            joint_angles[i] = KINEMATICS_CLAMP(joint_angles[i],
                solver->constraints[i].lower, solver->constraints[i].upper);
        }
    }
}

/* ============================================================================
 * URDF 解析器 — 轻量级 XML 解析
 * ============================================================================ */

typedef struct {
    const char* content;
    size_t length;
    size_t pos;
} XmlParser;

static XmlParser xml_parser_create(const char* xml, size_t len) {
    XmlParser p;
    p.content = xml;
    p.length = len;
    p.pos = 0;
    return p;
}

static void xml_skip_whitespace(XmlParser* p) {
    while (p->pos < p->length) {
        char c = p->content[p->pos];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
        p->pos++;
    }
}

static int xml_peek(XmlParser* p) {
    xml_skip_whitespace(p);
    if (p->pos >= p->length) return -1;
    return (unsigned char)p->content[p->pos];
}

static int xml_eat(XmlParser* p, char c) {
    xml_skip_whitespace(p);
    if (p->pos < p->length && p->content[p->pos] == c) {
        p->pos++;
        return 1;
    }
    return 0;
}

static int xml_match_str(XmlParser* p, const char* s) {
    xml_skip_whitespace(p);
    size_t slen = strlen(s);
    if (p->pos + slen > p->length) return 0;
    if (strncmp(p->content + p->pos, s, slen) == 0) {
        p->pos += slen;
        return 1;
    }
    return 0;
}

static int xml_read_until(XmlParser* p, char delim, char* out, size_t out_size) {
    size_t written = 0;
    while (p->pos < p->length && written + 1 < out_size) {
        char c = p->content[p->pos];
        if (c == delim) break;
        out[written++] = c;
        p->pos++;
    }
    out[written] = '\0';
    return (int)written;
}

static int xml_read_attr_value(XmlParser* p, char* out, size_t out_size) {
    if (!xml_eat(p, '=')) return -1;
    xml_skip_whitespace(p);
    if (p->pos >= p->length) return -1;
    char quote = p->content[p->pos];
    if (quote != '"' && quote != '\'') return -1;
    p->pos++;
    size_t written = 0;
    while (p->pos < p->length && written + 1 < out_size) {
        char c = p->content[p->pos];
        if (c == quote) break;
        out[written++] = c;
        p->pos++;
    }
    out[written] = '\0';
    if (p->pos < p->length && p->content[p->pos] == quote) p->pos++;
    return (int)written;
}

static int xml_read_tag_name(XmlParser* p, char* out, size_t out_size) {
    xml_skip_whitespace(p);
    size_t written = 0;
    while (p->pos < p->length && written + 1 < out_size) {
        char c = p->content[p->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '>' || c == '/') break;
        out[written++] = c;
        p->pos++;
    }
    out[written] = '\0';
    return (int)written;
}

static int xml_find_attr(XmlParser* p, const char* attr_name, char* value, size_t value_size) {
    xml_skip_whitespace(p);
    while (p->pos < p->length) {
        char c = p->content[p->pos];
        if (c == '>' || c == '/') break;
        char name[128];
        size_t nw = 0;
        while (p->pos < p->length && nw + 1 < sizeof(name)) {
            c = p->content[p->pos];
            if (c == '=' || c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '>' || c == '/') break;
            name[nw++] = c;
            p->pos++;
        }
        name[nw] = '\0';
        xml_skip_whitespace(p);
        if (xml_eat(p, '=')) {
            xml_skip_whitespace(p);
            char val[512];
            size_t vw = 0;
            if (p->pos < p->length && (p->content[p->pos] == '"' || p->content[p->pos] == '\'')) {
                char q = p->content[p->pos];
                p->pos++;
                while (p->pos < p->length && vw + 1 < sizeof(val)) {
                    if (p->content[p->pos] == q) break;
                    val[vw++] = p->content[p->pos];
                    p->pos++;
                }
                val[vw] = '\0';
                if (p->pos < p->length) p->pos++;
            }
            if (strcmp(name, attr_name) == 0) {
                strncpy(value, val, value_size - 1);
                value[value_size - 1] = '\0';
                return 1;
            }
        } else {
            if (strcmp(name, attr_name) == 0) {
                value[0] = '\0';
                return 1;
            }
        }
    }
    return 0;
}

static float xml_parse_float(const char* s) {
    if (!s || !*s) return 0.0f;
    return (float)atof(s);
}

static void xml_parse_vec3(const char* s, Vec3* v) {
    v->x = 0; v->y = 0; v->z = 0;
    if (!s || !*s) return;
    float vals[3] = {0,0,0};
    int count = sscanf(s, "%f %f %f", &vals[0], &vals[1], &vals[2]);
    if (count >= 1) v->x = vals[0];
    if (count >= 2) v->y = vals[1];
    if (count >= 3) v->z = vals[2];
}

static KinJointType xml_parse_joint_type(const char* type_str) {
    if (!type_str) return JOINT_TYPE_UNKNOWN;
    if (strcmp(type_str, "revolute") == 0) return JOINT_TYPE_REVOLUTE;
    if (strcmp(type_str, "prismatic") == 0) return JOINT_TYPE_PRISMATIC;
    if (strcmp(type_str, "fixed") == 0) return JOINT_TYPE_FIXED;
    if (strcmp(type_str, "continuous") == 0) return JOINT_TYPE_CONTINUOUS;
    if (strcmp(type_str, "floating") == 0) return JOINT_TYPE_FLOATING;
    if (strcmp(type_str, "planar") == 0) return JOINT_TYPE_PLANAR;
    return JOINT_TYPE_UNKNOWN;
}

typedef struct {
    char name[64];
    char parent[64];
    char child[64];
    float origin_xyz[3];
    float origin_rpy[3];
    float axis_xyz[3];
    float limit_lower;
    float limit_upper;
    float limit_effort;
    float limit_velocity;
    char type_str[32];
    int parsed;
} XmlUrdfJoint;

typedef struct {
    char name[64];
    char joint_name[64];
    float origin_xyz[3];
    float origin_rpy[3];
    float visual_geometry_xyz[3];
    float visual_geometry_rpy[3];
    float visual_geometry_size[3];
    float visual_geometry_radius;
    float visual_geometry_length;
    float collision_origin_xyz[3];
    float collision_origin_rpy[3];
    float collision_geometry_size[3];
    float collision_geometry_radius;
    float collision_geometry_length;
    char geometry_type[32];
    char collision_geometry_type[32];
    float inertial_origin_xyz[3];
    float inertial_mass;
    float inertial_ixx, inertial_iyy, inertial_izz;
    float inertial_ixy, inertial_ixz, inertial_iyz;
    int parsed;
} XmlUrdfLink;

static int xml_skip_element(XmlParser* p) {
    int depth = 0;
    while (p->pos < p->length) {
        char c = p->content[p->pos];
        if (c == '<') {
            if (p->pos + 1 < p->length) {
                if (p->content[p->pos + 1] == '/') {
                    depth--;
                    if (depth < 0) {
                        while (p->pos < p->length && p->content[p->pos] != '>') p->pos++;
                        if (p->pos < p->length) p->pos++;
                        return 1;
                    }
                    while (p->pos < p->length && p->content[p->pos] != '>') p->pos++;
                    if (p->pos < p->length) p->pos++;
                } else {
                    depth++;
                    while (p->pos < p->length && p->content[p->pos] != '>') p->pos++;
                    if (p->pos < p->length) p->pos++;
                }
            } else {
                p->pos++;
            }
        } else {
            p->pos++;
        }
    }
    return 1;
}

static int xml_parse_float_array(const char* s, float* out, int max_count) {
    if (!s || !*s) return 0;
    int count = 0;
    const char* p = s;
    while (*p && count < max_count) {
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        out[count++] = (float)atof(p);
        while (*p && *p != ' ' && *p != '\t') p++;
    }
    return count;
}

static int xml_find_element(XmlParser* p, const char* tag_name) {
    while (p->pos < p->length) {
        xml_skip_whitespace(p);
        if (p->pos >= p->length || p->content[p->pos] != '<') {
            p->pos++;
            continue;
        }
        p->pos++;
        if (p->pos < p->length && p->content[p->pos] == '/') {
            while (p->pos < p->length && p->content[p->pos] != '>') p->pos++;
            if (p->pos < p->length) p->pos++;
            continue;
        }
        size_t saved = p->pos;
        char name[128];
        int nlen = 0;
        while (p->pos < p->length && nlen < 127) {
            char c = p->content[p->pos];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '>' || c == '/') break;
            name[nlen++] = c;
            p->pos++;
        }
        name[nlen] = '\0';
        if (strcmp(name, tag_name) == 0) return 1;
        p->pos = saved - 1;
        p->pos++;
    }
    return 0;
}

static int xml_read_element_content(XmlParser* p, char* out, size_t out_size) {
    xml_skip_whitespace(p);
    size_t written = 0;
    while (p->pos < p->length && written + 1 < out_size) {
        char c = p->content[p->pos];
        if (c == '<') break;
        out[written++] = c;
        p->pos++;
    }
    out[written] = '\0';
    return (int)written;
}

static int xml_parse_urdf_joint(XmlParser* p, XmlUrdfJoint* joint) {
    memset(joint, 0, sizeof(XmlUrdfJoint));
    joint->axis_xyz[2] = 1.0f;
    joint->limit_upper = 3.14159265f;
    joint->limit_lower = -3.14159265f;
    joint->limit_effort = 100.0f;
    joint->limit_velocity = 10.0f;
    xml_find_attr(p, "name", joint->name, sizeof(joint->name));
    while (p->pos < p->length) {
        int c = xml_peek(p);
        if (c < 0) break;
        if (c == '<') {
            p->pos++;
            if (p->pos < p->length && p->content[p->pos] == '/') {
                p->pos++;
                char name[64];
                xml_read_tag_name(p, name, sizeof(name));
                if (strcmp(name, "joint") == 0) { joint->parsed = 1; return 1; }
                xml_eat(p, '>');
                continue;
            }
            char name[64];
            xml_read_tag_name(p, name, sizeof(name));
            if (strcmp(name, "parent") == 0) {
                xml_find_attr(p, "link", joint->parent, sizeof(joint->parent));
                xml_skip_element(p);
            } else if (strcmp(name, "child") == 0) {
                xml_find_attr(p, "link", joint->child, sizeof(joint->child));
                xml_skip_element(p);
            } else if (strcmp(name, "origin") == 0) {
                char xyz[128] = "", rpy[128] = "";
                xml_find_attr(p, "xyz", xyz, sizeof(xyz));
                xml_find_attr(p, "rpy", rpy, sizeof(rpy));
                if (*xyz) xml_parse_float_array(xyz, joint->origin_xyz, 3);
                if (*rpy) xml_parse_float_array(rpy, joint->origin_rpy, 3);
                xml_skip_element(p);
            } else if (strcmp(name, "axis") == 0) {
                char xyz[128] = "";
                xml_find_attr(p, "xyz", xyz, sizeof(xyz));
                if (*xyz) xml_parse_float_array(xyz, joint->axis_xyz, 3);
                xml_skip_element(p);
            } else if (strcmp(name, "limit") == 0) {
                char lower[64] = "", upper[64] = "", effort[64] = "", vel[64] = "";
                xml_find_attr(p, "lower", lower, sizeof(lower));
                xml_find_attr(p, "upper", upper, sizeof(upper));
                xml_find_attr(p, "effort", effort, sizeof(effort));
                xml_find_attr(p, "velocity", vel, sizeof(vel));
                if (*lower) joint->limit_lower = xml_parse_float(lower);
                if (*upper) joint->limit_upper = xml_parse_float(upper);
                if (*effort) joint->limit_effort = xml_parse_float(effort);
                if (*vel) joint->limit_velocity = xml_parse_float(vel);
                xml_skip_element(p);
            } else if (strcmp(name, "type") == 0 || strcmp(name, "dynamics") == 0 || strcmp(name, "calibration") == 0 || strcmp(name, "mimic") == 0 || strcmp(name, "safety_controller") == 0) {
                xml_skip_element(p);
            } else {
                xml_skip_element(p);
            }
        } else {
            p->pos++;
        }
    }
    joint->parsed = 1;
    return 1;
}

static int xml_parse_urdf_link(XmlParser* p, XmlUrdfLink* link) {
    memset(link, 0, sizeof(XmlUrdfLink));
    link->collision_geometry_radius = 0.05f;
    link->visual_geometry_radius = 0.05f;
    xml_find_attr(p, "name", link->name, sizeof(link->name));
    while (p->pos < p->length) {
        int c = xml_peek(p);
        if (c < 0) break;
        if (c == '<') {
            p->pos++;
            if (p->pos < p->length && p->content[p->pos] == '/') {
                p->pos++;
                char name[64];
                xml_read_tag_name(p, name, sizeof(name));
                if (strcmp(name, "link") == 0) { link->parsed = 1; return 1; }
                xml_eat(p, '>');
                continue;
            }
            char name[64];
            xml_read_tag_name(p, name, sizeof(name));
            if (strcmp(name, "inertial") == 0) {
                char sub_origin_xyz[128] = "";
                xml_find_attr(p, "xyz", sub_origin_xyz, sizeof(sub_origin_xyz));
                if (*sub_origin_xyz) xml_parse_float_array(sub_origin_xyz, link->inertial_origin_xyz, 3);
                XmlParser sub = *p;
                while (sub.pos < sub.length) {
                    if (sub.content[sub.pos] == '<' && sub.pos + 1 < sub.length) {
                        if (sub.content[sub.pos + 1] == '/') break;
                        sub.pos++;
                        char subname[64];
                        int sn = 0;
                        while (sub.pos < sub.length && sn < 63) {
                            char sc = sub.content[sub.pos];
                            if (sc == ' ' || sc == '\t' || sc == '\n' || sc == '\r' || sc == '>' || sc == '/') break;
                            subname[sn++] = sc;
                            sub.pos++;
                        }
                        subname[sn] = '\0';
                        if (strcmp(subname, "mass") == 0) {
                            char val[64] = "";
                            xml_find_attr(&sub, "value", val, sizeof(val));
                            if (*val) link->inertial_mass = xml_parse_float(val);
                            xml_skip_element(&sub);
                        } else if (strcmp(subname, "inertia") == 0) {
                            char ixx[64]="", iyy[64]="", izz[64]="", ixy[64]="", ixz[64]="", iyz[64]="";
                            xml_find_attr(&sub, "ixx", ixx, sizeof(ixx));
                            xml_find_attr(&sub, "iyy", iyy, sizeof(iyy));
                            xml_find_attr(&sub, "izz", izz, sizeof(izz));
                            xml_find_attr(&sub, "ixy", ixy, sizeof(ixy));
                            xml_find_attr(&sub, "ixz", ixz, sizeof(ixz));
                            xml_find_attr(&sub, "iyz", iyz, sizeof(iyz));
                            if (*ixx) link->inertial_ixx = xml_parse_float(ixx);
                            if (*iyy) link->inertial_iyy = xml_parse_float(iyy);
                            if (*izz) link->inertial_izz = xml_parse_float(izz);
                            if (*ixy) link->inertial_ixy = xml_parse_float(ixy);
                            if (*ixz) link->inertial_ixz = xml_parse_float(ixz);
                            if (*iyz) link->inertial_iyz = xml_parse_float(iyz);
                            xml_skip_element(&sub);
                        } else {
                            xml_skip_element(&sub);
                        }
                    } else {
                        sub.pos++;
                    }
                }
                xml_skip_element(p);
            } else if (strcmp(name, "visual") == 0) {
                char origin_xyz[128] = "", origin_rpy[128] = "";
                xml_find_attr(p, "xyz", origin_xyz, sizeof(origin_xyz));
                xml_find_attr(p, "rpy", origin_rpy, sizeof(origin_rpy));
                if (*origin_xyz) xml_parse_float_array(origin_xyz, link->visual_geometry_xyz, 3);
                if (*origin_rpy) xml_parse_float_array(origin_rpy, link->visual_geometry_rpy, 3);
                XmlParser sub = *p;
                while (sub.pos < sub.length) {
                    if (sub.content[sub.pos] == '<' && sub.pos + 1 < sub.length) {
                        if (sub.content[sub.pos + 1] == '/') break;
                        sub.pos++;
                        char subname[64];
                        int sn = 0;
                        while (sub.pos < sub.length && sn < 63) {
                            char sc = sub.content[sub.pos];
                            if (sc == ' ' || sc == '\t' || sc == '\n' || sc == '\r' || sc == '>' || sc == '/') break;
                            subname[sn++] = sc;
                            sub.pos++;
                        }
                        subname[sn] = '\0';
                        if (strcmp(subname, "geometry") == 0) {
                            XmlParser geo = sub;
                            while (geo.pos < geo.length) {
                                if (geo.content[geo.pos] == '<' && geo.pos + 1 < geo.length) {
                                    if (geo.content[geo.pos + 1] == '/') break;
                                    geo.pos++;
                                    char gname[64];
                                    int gn = 0;
                                    while (geo.pos < geo.length && gn < 63) {
                                        char gc = geo.content[geo.pos];
                                        if (gc == ' ' || gc == '\t' || gc == '\n' || gc == '\r' || gc == '>' || gc == '/') break;
                                        gname[gn++] = gc;
                                        geo.pos++;
                                    }
                                    gname[gn] = '\0';
                                    strncpy(link->geometry_type, gname, sizeof(link->geometry_type) - 1);
                                    if (strcmp(gname, "box") == 0) {
                                        char size[128] = "";
                                        xml_find_attr(&geo, "size", size, sizeof(size));
                                        if (*size) xml_parse_float_array(size, link->visual_geometry_size, 3);
                                    } else if (strcmp(gname, "sphere") == 0) {
                                        char rad[64] = "";
                                        xml_find_attr(&geo, "radius", rad, sizeof(rad));
                                        if (*rad) link->visual_geometry_radius = xml_parse_float(rad);
                                    } else if (strcmp(gname, "cylinder") == 0) {
                                        char rad[64] = "", len[64] = "";
                                        xml_find_attr(&geo, "radius", rad, sizeof(rad));
                                        xml_find_attr(&geo, "length", len, sizeof(len));
                                        if (*rad) link->visual_geometry_radius = xml_parse_float(rad);
                                        if (*len) link->visual_geometry_length = xml_parse_float(len);
                                    } else if (strcmp(gname, "mesh") == 0) {
                                    }
                                    xml_skip_element(&geo);
                                } else {
                                    geo.pos++;
                                }
                            }
                            xml_skip_element(&sub);
                        } else {
                            xml_skip_element(&sub);
                        }
                    } else {
                        sub.pos++;
                    }
                }
                xml_skip_element(p);
            } else if (strcmp(name, "collision") == 0) {
                char origin_xyz[128] = "", origin_rpy[128] = "";
                xml_find_attr(p, "xyz", origin_xyz, sizeof(origin_xyz));
                xml_find_attr(p, "rpy", origin_rpy, sizeof(origin_rpy));
                if (*origin_xyz) xml_parse_float_array(origin_xyz, link->collision_origin_xyz, 3);
                if (*origin_rpy) xml_parse_float_array(origin_rpy, link->collision_origin_rpy, 3);
                XmlParser sub = *p;
                while (sub.pos < sub.length) {
                    if (sub.content[sub.pos] == '<' && sub.pos + 1 < sub.length) {
                        if (sub.content[sub.pos + 1] == '/') break;
                        sub.pos++;
                        char subname[64];
                        int sn = 0;
                        while (sub.pos < sub.length && sn < 63) {
                            char sc = sub.content[sub.pos];
                            if (sc == ' ' || sc == '\t' || sc == '\n' || sc == '\r' || sc == '>' || sc == '/') break;
                            subname[sn++] = sc;
                            sub.pos++;
                        }
                        subname[sn] = '\0';
                        if (strcmp(subname, "geometry") == 0) {
                            XmlParser geo = sub;
                            while (geo.pos < geo.length) {
                                if (geo.content[geo.pos] == '<' && geo.pos + 1 < geo.length) {
                                    if (geo.content[geo.pos + 1] == '/') break;
                                    geo.pos++;
                                    char gname[64];
                                    int gn = 0;
                                    while (geo.pos < geo.length && gn < 63) {
                                        char gc = geo.content[geo.pos];
                                        if (gc == ' ' || gc == '\t' || gc == '\n' || gc == '\r' || gc == '>' || gc == '/') break;
                                        gname[gn++] = gc;
                                        geo.pos++;
                                    }
                                    gname[gn] = '\0';
                                    strncpy(link->collision_geometry_type, gname, sizeof(link->collision_geometry_type) - 1);
                                    if (strcmp(gname, "box") == 0) {
                                        char size[128] = "";
                                        xml_find_attr(&geo, "size", size, sizeof(size));
                                        if (*size) xml_parse_float_array(size, link->collision_geometry_size, 3);
                                    } else if (strcmp(gname, "sphere") == 0) {
                                        char rad[64] = "";
                                        xml_find_attr(&geo, "radius", rad, sizeof(rad));
                                        if (*rad) link->collision_geometry_radius = xml_parse_float(rad);
                                    } else if (strcmp(gname, "cylinder") == 0) {
                                        char rad[64] = "", len[64] = "";
                                        xml_find_attr(&geo, "radius", rad, sizeof(rad));
                                        xml_find_attr(&geo, "length", len, sizeof(len));
                                        if (*rad) link->collision_geometry_radius = xml_parse_float(rad);
                                        if (*len) link->collision_geometry_length = xml_parse_float(len);
                                    }
                                    xml_skip_element(&geo);
                                } else {
                                    geo.pos++;
                                }
                            }
                            xml_skip_element(&sub);
                        } else {
                            xml_skip_element(&sub);
                        }
                    } else {
                        sub.pos++;
                    }
                }
                xml_skip_element(p);
            } else {
                xml_skip_element(p);
            }
        } else {
            p->pos++;
        }
    }
    link->parsed = 1;
    return 1;
}

static int urdf_build_model(KinematicModel* model, XmlUrdfLink* links, int link_count,
                             XmlUrdfJoint* joints, int joint_count) {
    (void)links;
    (void)link_count;
    for (int i = 0; i < joint_count; i++) {
        XmlUrdfJoint* j = &joints[i];
        KinJointType type = xml_parse_joint_type(j->type_str);
        if (type == JOINT_TYPE_UNKNOWN) {
            if (strstr(j->type_str, "revolute")) type = JOINT_TYPE_REVOLUTE;
            else if (strstr(j->type_str, "prismatic")) type = JOINT_TYPE_PRISMATIC;
            else if (strstr(j->type_str, "fixed")) type = JOINT_TYPE_FIXED;
            else if (strstr(j->type_str, "continuous")) type = JOINT_TYPE_CONTINUOUS;
            else type = JOINT_TYPE_REVOLUTE;
        }
        int parent_idx = -1;
        if (i > 0) parent_idx = i - 1;
        DHParameter dh;
        memset(&dh, 0, sizeof(dh));
        dh.a = 0;
        dh.alpha = 0;
        dh.d = j->origin_xyz[2];
        dh.theta = 0;
        float limit_lower = j->limit_lower;
        float limit_upper = j->limit_upper;
        if (type == JOINT_TYPE_CONTINUOUS) {
            limit_lower = -2.0f * (float)M_PI;
            limit_upper = 2.0f * (float)M_PI;
        }
        int idx = kinematic_model_add_joint(model, parent_idx, j->name, type, &dh,
                                             limit_lower, limit_upper);
        if (idx >= 0) {
            model->joints[idx].joint_max_velocity = j->limit_velocity;
            model->joints[idx].joint_max_torque = j->limit_effort;
            model->joints[idx].parent_to_joint_pos[0] = j->origin_xyz[0];
            model->joints[idx].parent_to_joint_pos[1] = j->origin_xyz[1];
            model->joints[idx].parent_to_joint_pos[2] = j->origin_xyz[2];
        }
    }
    return 0;
}

int urdf_parse_string(KinematicModel* model, const char* urdf_xml) {
    if (!model || !urdf_xml) return -1;
    kinematic_model_init(model);
    size_t len = strlen(urdf_xml);
    XmlParser p = xml_parser_create(urdf_xml, len);
    int found_robot = xml_find_element(&p, "robot");
    if (!found_robot) return -2;
    char robot_name[128] = "";
    xml_find_attr(&p, "name", robot_name, sizeof(robot_name));
    if (*robot_name) strncpy(model->name, robot_name, sizeof(model->name) - 1);
    XmlUrdfLink links[KINEMATICS_MAX_LINKS];
    int link_count = 0;
    XmlUrdfJoint joints[KINEMATICS_MAX_JOINTS];
    int joint_count = 0;
    memset(links, 0, sizeof(links));
    memset(joints, 0, sizeof(joints));
    while (p.pos < p.length && (link_count < KINEMATICS_MAX_LINKS || joint_count < KINEMATICS_MAX_JOINTS)) {
        int c = xml_peek(&p);
        if (c < 0) break;
        if (c == '<') {
            size_t saved = p.pos;
            p.pos++;
            if (p.pos < p.length && p.content[p.pos] == '/') {
                while (p.pos < p.length && p.content[p.pos] != '>') p.pos++;
                if (p.pos < p.length) p.pos++;
                continue;
            }
            char tag[64];
            xml_read_tag_name(&p, tag, sizeof(tag));
            p.pos = saved;
            if (strcmp(tag, "link") == 0 && link_count < KINEMATICS_MAX_LINKS) {
                xml_parse_urdf_link(&p, &links[link_count++]);
            } else if ((strcmp(tag, "joint") == 0) && joint_count < KINEMATICS_MAX_JOINTS) {
                char jt[32] = "";
                int saved_pos = (int)p.pos;
                xml_find_attr(&p, "type", jt, sizeof(jt));
                p.pos = (size_t)saved_pos;
                strncpy(joints[joint_count].type_str, jt, sizeof(joints[joint_count].type_str) - 1);
                xml_parse_urdf_joint(&p, &joints[joint_count++]);
            } else {
                xml_skip_element(&p);
            }
        } else {
            p.pos++;
        }
    }
    model->source_urdf[0] = '\0';
    strncat(model->source_urdf, urdf_xml, sizeof(model->source_urdf) - 1);
    return urdf_build_model(model, links, link_count, joints, joint_count);
}

int urdf_parse_file(KinematicModel* model, const char* filepath) {
    if (!model || !filepath) return -1;
    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -3;
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    if (fsize <= 0) { fclose(fp); return -3; }
    fseek(fp, 0, SEEK_SET);
    char* xml = (char*)safe_malloc((size_t)fsize + 1);
    if (!xml) { fclose(fp); return -4; }
    size_t read_size = fread(xml, 1, (size_t)fsize, fp);
    fclose(fp);
    xml[read_size] = '\0';
    int result = urdf_parse_string(model, xml);
    safe_free((void**)&xml);
    return result;
}

const char* kinematics_error_string(int error_code) {
    switch (error_code) {
        case -1: return "无效参数或模型为空";
        case -2: return "URDF 格式错误 - 未找到 <robot> 标签";
        case -3: return "无法打开 URDF 文件";
        case -4: return "URDF 文件内存分配失败";
        case -5: return "关节数量超过最大限制";
        case -6: return "未找到末端执行器关节";
        case -7: return "逆运动学求解未收敛";
        default: return "未知错误";
    }
}

/* ============================================================================
 * 内部辅助：6×6 对称矩阵运算（Gram矩阵/Cholesky/SVD）
 * ============================================================================ */

static void kin_gram_6x6(const float* jac, int dof, float* G) {
    memset(G, 0, 36 * sizeof(float));
    for (int j = 0; j < dof; j++) {
        const float* col = jac + j * 6;
        for (int r = 0; r < 6; r++) {
            for (int c = r; c < 6; c++) {
                G[r * 6 + c] += col[r] * col[c];
            }
        }
    }
    for (int r = 0; r < 6; r++)
        for (int c = 0; c < r; c++)
            G[r * 6 + c] = G[c * 6 + r];
}

static int kin_cholesky_6x6(const float* A, float* L) {
    memset(L, 0, 36 * sizeof(float));
    for (int i = 0; i < 6; i++) {
        float sum = 0.0f;
        for (int k = 0; k < i; k++)
            sum += L[i * 6 + k] * L[i * 6 + k];
        float val = A[i * 6 + i] - sum;
        if (val < 1e-12f) return -1;
        L[i * 6 + i] = sqrtf(val);
        for (int j = i + 1; j < 6; j++) {
            sum = 0.0f;
            for (int k = 0; k < i; k++)
                sum += L[j * 6 + k] * L[i * 6 + k];
            L[j * 6 + i] = (A[j * 6 + i] - sum) / L[i * 6 + i];
        }
    }
    return 0;
}

static void kin_cholesky_solve_6x6(const float* L, const float* b, float* x) {
    float y[6];
    for (int i = 0; i < 6; i++) {
        float sum = 0.0f;
        for (int k = 0; k < i; k++) sum += L[i * 6 + k] * y[k];
        y[i] = (b[i] - sum) / L[i * 6 + i];
    }
    for (int i = 5; i >= 0; i--) {
        float sum = 0.0f;
        for (int k = i + 1; k < 6; k++) sum += L[k * 6 + i] * x[k];
        x[i] = (y[i] - sum) / L[i * 6 + i];
    }
}

static int kin_invert_6x6_cholesky(const float* A, float* Ainv) {
    float L[36];
    if (kin_cholesky_6x6(A, L) != 0) return -1;
    float ei[6];
    for (int col = 0; col < 6; col++) {
        memset(ei, 0, 6 * sizeof(float));
        ei[col] = 1.0f;
        kin_cholesky_solve_6x6(L, ei, Ainv + col * 6);
    }
    return 0;
}

static int kin_svd_sym_6x6(float* A, float* U, float* S) {
    memset(U, 0, 36 * sizeof(float));
    for (int i = 0; i < 6; i++) U[i * 6 + i] = 1.0f;
    int iter = 0;
    const float tol = 1e-12f;
    for (int max_iter = 100; iter < max_iter; iter++) {
        float max_off = 0.0f;
        int p = 0, q = 0;
        for (int i = 0; i < 6; i++)
            for (int j = i + 1; j < 6; j++) {
                float v = fabsf(A[i * 6 + j]);
                if (v > max_off) { max_off = v; p = i; q = j; }
            }
        if (max_off < tol) break;
        float App = A[p * 6 + p], Aqq = A[q * 6 + q], Apq = A[p * 6 + q];
        float theta = 0.5f * atan2f(2.0f * Apq, Aqq - App);
        float c = cosf(theta), s = sinf(theta);
        float npp = c * c * App + s * s * Aqq - 2.0f * s * c * Apq;
        float nqq = s * s * App + c * c * Aqq + 2.0f * s * c * Apq;
        A[p * 6 + p] = npp; A[q * 6 + q] = nqq;
        A[p * 6 + q] = 0.0f; A[q * 6 + p] = 0.0f;
        for (int r = 0; r < 6; r++) {
            if (r == p || r == q) continue;
            float Arp = A[r * 6 + p], Arq = A[r * 6 + q];
            A[r * 6 + p] = c * Arp - s * Arq; A[p * 6 + r] = A[r * 6 + p];
            A[r * 6 + q] = s * Arp + c * Arq; A[q * 6 + r] = A[r * 6 + q];
        }
        for (int r = 0; r < 6; r++) {
            float Urp = U[r * 6 + p], Urq = U[r * 6 + q];
            U[r * 6 + p] = c * Urp - s * Urq;
            U[r * 6 + q] = s * Urp + c * Urq;
        }
    }
    for (int i = 0; i < 6; i++)
        S[i] = sqrtf(fmaxf(0.0f, A[i * 6 + i]));
    for (int i = 0; i < 5; i++)
        for (int j = i + 1; j < 6; j++)
            if (S[j] > S[i]) {
                float tmp = S[i]; S[i] = S[j]; S[j] = tmp;
                for (int r = 0; r < 6; r++) {
                    tmp = U[r * 6 + i]; U[r * 6 + i] = U[r * 6 + j]; U[r * 6 + j] = tmp;
                }
            }
    return iter;
}

/* ============================================================================
 * 运动学奇异性处理
 * ============================================================================ */

float kinematics_manipulability(const float* jacobian, int dof) {
    if (!jacobian || dof <= 0) return 0.0f;
    float G[36];
    kin_gram_6x6(jacobian, dof, G);
    float L[36];
    if (kin_cholesky_6x6(G, L) != 0) return 0.0f;
    float det = 1.0f;
    for (int i = 0; i < 6; i++) det *= L[i * 6 + i];
    return det * det;
}

float kinematics_condition_number(const float* jacobian, int dof) {
    if (!jacobian || dof <= 0) return 1e10f;
    float G[36], U[36], S[6];
    kin_gram_6x6(jacobian, dof, G);
    kin_svd_sym_6x6(G, U, S);
    (void)U;
    if (S[5] < 1e-12f) return 1e10f;
    return S[0] / S[5];
}

float kinematics_min_singular_value(const float* jacobian, int dof) {
    if (!jacobian || dof <= 0) return 0.0f;
    float G[36], U[36], S[6];
    kin_gram_6x6(jacobian, dof, G);
    kin_svd_sym_6x6(G, U, S);
    (void)U;
    return S[5];
}

int kinematics_svd(const float* jacobian, int rows, int cols,
                    float* U, float* S, float* Vt) {
    if (!jacobian || !U || !S || !Vt) return -1;
    if (rows != 6 || cols <= 0) return -1;
    float G[36];
    kin_gram_6x6(jacobian, cols, G);
    float local_U[36];
    kin_svd_sym_6x6(G, local_U, S);
    for (int r = 0; r < 6; r++)
        for (int c = 0; c < 6; c++)
            U[r * 6 + c] = local_U[r * 6 + c];
    for (int i = 0; i < 6; i++) {
        if (S[i] < 1e-14f) {
            for (int j = 0; j < cols; j++) Vt[i * cols + j] = 0.0f;
            continue;
        }
        float inv_s = 1.0f / S[i];
        for (int j = 0; j < cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 6; k++)
                sum += local_U[k * 6 + i] * jacobian[j * 6 + k];
            Vt[i * cols + j] = sum * inv_s;
        }
    }
    return 0;
}

int kinematics_sr_inverse(const float* jacobian, int rows, int cols,
                           float* jacobian_pinv,
                           float* damping_out, float manipulability_threshold) {
    if (!jacobian || !jacobian_pinv) return -1;
    if (rows != 6 || cols <= 0) return -1;
    float G[36];
    kin_gram_6x6(jacobian, cols, G);
    float m = kinematics_manipulability(jacobian, cols);
    float thresh = (manipulability_threshold > 0.0f) ? manipulability_threshold : 0.01f;
    float lambda_sq = 0.0f;
    if (m < thresh) {
        float ratio = m / thresh;
        lambda_sq = (1.0f - ratio * ratio) * 0.01f;
    }
    if (damping_out) *damping_out = sqrtf(lambda_sq);
    for (int i = 0; i < 6; i++) G[i * 6 + i] += lambda_sq;
    float Ginv[36];
    if (kin_invert_6x6_cholesky(G, Ginv) != 0) {
        for (int i = 0; i < 6; i++) G[i * 6 + i] -= lambda_sq;
        return -1;
    }
    for (int j = 0; j < cols; j++) {
        for (int r = 0; r < 6; r++) {
            float sum = 0.0f;
            for (int k = 0; k < 6; k++)
                sum += jacobian[j * 6 + k] * Ginv[k * 6 + r];
            jacobian_pinv[j * 6 + r] = sum;
        }
    }
    for (int i = 0; i < 6; i++) G[i * 6 + i] -= lambda_sq;
    return 0;
}

int kinematics_singularity_detect(const float* jacobian, int dof,
                                   float threshold) {
    if (!jacobian || dof <= 0) return -1;
    float m = kinematics_manipulability(jacobian, dof);
    float t = (threshold > 0.0f) ? threshold : 0.01f;
    if (m < t * 0.1f) return 2;
    if (m < t) return 1;
    return 0;
}

/* ============================================================================
 * 冗余机器人运动学 — 辅助
 * ============================================================================ */

static int kin_jacobian_multiply(const float* jac, int dof,
                                  const float* x, float* result) {
    for (int r = 0; r < 6; r++) {
        float sum = 0.0f;
        for (int j = 0; j < dof; j++)
            sum += jac[j * 6 + r] * x[j];
        result[r] = sum;
    }
    return 0;
}

static int kin_jacobian_transpose_mult(const float* jac, int dof,
                                        const float* y, float* result) {
    for (int j = 0; j < dof; j++) {
        float sum = 0.0f;
        for (int r = 0; r < 6; r++)
            sum += jac[j * 6 + r] * y[r];
        result[j] = sum;
    }
    return 0;
}

static int kin_jacobian_pinv_mult(const float* jac, int dof,
                                   const float* y, float* result, float damping) {
    float G[36];
    kin_gram_6x6(jac, dof, G);
    float lambda_sq = damping * damping;
    for (int i = 0; i < 6; i++) G[i * 6 + i] += lambda_sq;
    float L[36];
    if (kin_cholesky_6x6(G, L) != 0) {
        for (int i = 0; i < 6; i++) G[i * 6 + i] -= lambda_sq;
        return -1;
    }
    float x[6];
    kin_cholesky_solve_6x6(L, y, x);
    kin_jacobian_transpose_mult(jac, dof, x, result);
    for (int i = 0; i < 6; i++) G[i * 6 + i] -= lambda_sq;
    return 0;
}

/* ============================================================================
 * 冗余机器人运动学 — 零空间投影
 * ============================================================================ */

int kinematics_null_space_project(const float* jacobian, int dof,
                                   const float* gradient,
                                   float* null_space_qdot) {
    if (!jacobian || !gradient || !null_space_qdot) return -1;
    if (dof <= 0) return -1;
    float jg[6];
    kin_jacobian_multiply(jacobian, dof, gradient, jg);
    float pinv_jg[KINEMATICS_MAX_DOF];
    if (kin_jacobian_pinv_mult(jacobian, dof, jg, pinv_jg, 0.001f) != 0) {
        memcpy(null_space_qdot, gradient, (size_t)dof * sizeof(float));
        return -1;
    }
    for (int i = 0; i < dof; i++)
        null_space_qdot[i] = gradient[i] - pinv_jg[i];
    return 0;
}

int kinematics_joint_limit_gradient(const KinematicModel* model,
                                     const float* joint_angles,
                                     float* gradient) {
    if (!model || !joint_angles || !gradient) return -1;
    for (int i = 0; i < model->joint_count; i++) {
        float lo = model->joints[i].joint_limit_lower;
        float hi = model->joints[i].joint_limit_upper;
        float q = joint_angles[i];
        float mid = 0.5f * (lo + hi);
        float range = hi - lo;
        if (range < KINEMATICS_EPSILON) { gradient[i] = 0.0f; continue; }
        float normalized = (q - mid) / (0.5f * range);
        float clamped = KINEMATICS_CLAMP(normalized, -1.0f, 1.0f);
        gradient[i] = -clamped / (1.0f - clamped * clamped + 0.1f);
    }
    return 0;
}

int kinematics_singularity_avoidance_gradient(const float* jacobian,
                                               int dof, float* gradient) {
    if (!jacobian || !gradient) return -1;
    if (dof <= 0) return -1;
    float G[36], U[36], S[6];
    kin_gram_6x6(jacobian, dof, G);
    kin_svd_sym_6x6(G, U, S);
    float inv_sum = 0.0f;
    for (int i = 0; i < 6; i++) {
        if (S[i] > 1e-12f) inv_sum += 1.0f / (S[i] * S[i]);
    }
    if (inv_sum < 1e-12f) { memset(gradient, 0, (size_t)dof * sizeof(float)); return 0; }
    float grad_task[6];
    for (int r = 0; r < 6; r++) {
        float sum = 0.0f;
        for (int i = 0; i < 6; i++) {
            if (S[i] < 1e-12f) continue;
            float inv_s3 = 1.0f / (S[i] * S[i] * S[i]);
            sum += inv_s3 * U[r * 6 + i];
        }
        grad_task[r] = sum * inv_sum;
    }
    kin_jacobian_transpose_mult(jacobian, dof, grad_task, gradient);
    return 0;
}

/* ============================================================================
 * 冗余机器人运动学 — 任务优先级 IK
 * ============================================================================ */

int inverse_kinematics_priority(const KinematicModel* model,
                                 const Vec3* primary_target,
                                 const float* primary_orient,
                                 const Vec3* secondary_target,
                                 const float* secondary_orient,
                                 float* joint_angles,
                                 float damping, int max_iter, float tolerance) {
    if (!model || !primary_target || !joint_angles) return -1;
    int ee_idx = model->end_effector_joint;
    if (ee_idx < 0) ee_idx = model->joint_count - 1;
    if (ee_idx < 0) return -1;
    int dof = 0;
    for (int i = 0; i <= ee_idx; i++)
        if (model->joints[i].joint_type != JOINT_TYPE_FIXED) dof++;
    if (dof <= 0) return -1;
    if (max_iter <= 0) max_iter = KINEMATICS_IK_MAX_ITER;
    if (tolerance <= 0) tolerance = KINEMATICS_IK_TOLERANCE;
    if (damping <= 0) damping = KINEMATICS_IK_DAMPING;
    float dls_damping = damping;
    int result = -1;
    for (int iter = 0; iter < max_iter; iter++) {
        EndEffectorState state;
        forward_kinematics(model, joint_angles, &state);
        float dx = primary_target->x - state.position.x;
        float dy = primary_target->y - state.position.y;
        float dz = primary_target->z - state.position.z;
        float pos_err = sqrtf(dx * dx + dy * dy + dz * dz);
        float error_6[6] = { dx, dy, dz, 0, 0, 0 };
        if (primary_orient) {
            float q_conj[4];
            quat_conjugate(q_conj, state.orientation);
            float q_diff[4];
            quat_multiply(q_diff, primary_orient, q_conj);
            error_6[3] = q_diff[0] * 2.0f;
            error_6[4] = q_diff[1] * 2.0f;
            error_6[5] = q_diff[2] * 2.0f;
        }
        if (pos_err < tolerance && (primary_orient == NULL ||
            (fabsf(error_6[3]) < 0.01f && fabsf(error_6[4]) < 0.01f && fabsf(error_6[5]) < 0.01f))) {
            result = iter + 1;
            break;
        }
        float* jac = (float*)safe_malloc((size_t)dof * 6 * sizeof(float));
        if (!jac) break;
        int ndof = compute_jacobian(model, joint_angles, jac, dof);
        if (ndof <= 0) { safe_free((void**)&jac); break; }
        float jac_pinv_dq[KINEMATICS_MAX_DOF];
        if (kin_jacobian_pinv_mult(jac, ndof, error_6, jac_pinv_dq, dls_damping) != 0) {
            safe_free((void**)&jac); break;
        }
        if (secondary_target) {
            EndEffectorState sec_state;
            forward_kinematics(model, joint_angles, &sec_state);
            float sx = secondary_target->x - sec_state.position.x;
            float sy = secondary_target->y - sec_state.position.y;
            float sz = secondary_target->z - sec_state.position.z;
            float serr[6] = { sx, sy, sz, 0, 0, 0 };
            if (secondary_orient) {
                float qc[4];
                quat_conjugate(qc, sec_state.orientation);
                float qd[4];
                quat_multiply(qd, secondary_orient, qc);
                serr[3] = qd[0] * 2.0f; serr[4] = qd[1] * 2.0f; serr[5] = qd[2] * 2.0f;
            }
            float sec_dq[KINEMATICS_MAX_DOF];
            if (kin_jacobian_pinv_mult(jac, ndof, serr, sec_dq, dls_damping) == 0) {
                float ns_dq[KINEMATICS_MAX_DOF];
                kinematics_null_space_project(jac, ndof, sec_dq, ns_dq);
                for (int i = 0; i < ndof; i++)
                    jac_pinv_dq[i] += ns_dq[i] * 0.3f;
            }
        }
        int j_idx = 0;
        for (int i = 0; i <= ee_idx; i++) {
            if (model->joints[i].joint_type == JOINT_TYPE_FIXED) continue;
            joint_angles[i] += jac_pinv_dq[j_idx] * 0.5f;
            if (model->joints[i].joint_type != JOINT_TYPE_CONTINUOUS)
                joint_angles[i] = KINEMATICS_CLAMP(joint_angles[i],
                    model->joints[i].joint_limit_lower, model->joints[i].joint_limit_upper);
            j_idx++;
        }
        safe_free((void**)&jac);
    }
    if (result < 0) {
        EndEffectorState final;
        forward_kinematics(model, joint_angles, &final);
        float df = sqrtf((final.position.x - primary_target->x) * (final.position.x - primary_target->x) +
                         (final.position.y - primary_target->y) * (final.position.y - primary_target->y) +
                         (final.position.z - primary_target->z) * (final.position.z - primary_target->z));
        if (df < tolerance * 10) result = max_iter;
    }
    return result;
}

/* ============================================================================
 * 冗余机器人运动学 — 梯度投影 IK
 * ============================================================================ */

int inverse_kinematics_gradient_projection(const KinematicModel* model,
                                            const Vec3* target_pos,
                                            const float* target_orient,
                                            float* joint_angles,
                                            float joint_limit_weight,
                                            float singularity_weight,
                                            float step_size,
                                            int max_iter, float tolerance) {
    if (!model || !target_pos || !joint_angles) return -1;
    int ee_idx = model->end_effector_joint;
    if (ee_idx < 0) ee_idx = model->joint_count - 1;
    if (ee_idx < 0) return -1;
    int dof = 0;
    for (int i = 0; i <= ee_idx; i++)
        if (model->joints[i].joint_type != JOINT_TYPE_FIXED) dof++;
    if (dof <= 0) return -1;
    if (max_iter <= 0) max_iter = KINEMATICS_IK_MAX_ITER * 2;
    if (tolerance <= 0) tolerance = KINEMATICS_IK_TOLERANCE;
    if (step_size <= 0) step_size = 0.3f;
    float w_limit = (joint_limit_weight > 0.0f) ? joint_limit_weight : 0.1f;
    float w_sing = (singularity_weight > 0.0f) ? singularity_weight : 0.05f;
    float dls_damping = KINEMATICS_IK_DAMPING;
    int result = -1;
    for (int iter = 0; iter < max_iter; iter++) {
        EndEffectorState state;
        forward_kinematics(model, joint_angles, &state);
        float dx = target_pos->x - state.position.x;
        float dy = target_pos->y - state.position.y;
        float dz = target_pos->z - state.position.z;
        float pos_err = sqrtf(dx * dx + dy * dy + dz * dz);
        float error_6[6] = { dx, dy, dz, 0, 0, 0 };
        if (target_orient) {
            float qc[4];
            quat_conjugate(qc, state.orientation);
            float qd[4];
            quat_multiply(qd, target_orient, qc);
            error_6[3] = qd[0] * 2.0f; error_6[4] = qd[1] * 2.0f; error_6[5] = qd[2] * 2.0f;
        }
        if (pos_err < tolerance && (target_orient == NULL ||
            (fabsf(error_6[3]) < 0.01f && fabsf(error_6[4]) < 0.01f && fabsf(error_6[5]) < 0.01f))) {
            result = iter + 1;
            break;
        }
        float* jac = (float*)safe_malloc((size_t)dof * 6 * sizeof(float));
        if (!jac) break;
        int ndof = compute_jacobian(model, joint_angles, jac, dof);
        if (ndof <= 0) { safe_free((void**)&jac); break; }
        float dq_primary[KINEMATICS_MAX_DOF];
        if (kin_jacobian_pinv_mult(jac, ndof, error_6, dq_primary, dls_damping) != 0) {
            safe_free((void**)&jac); break;
        }
        float gradient_ns[KINEMATICS_MAX_DOF] = { 0 };
        if (w_limit > 0.0f || w_sing > 0.0f) {
            float grad_composite[KINEMATICS_MAX_DOF] = { 0 };
            if (w_limit > 0.0f) {
                float lim_grad[KINEMATICS_MAX_DOF];
                kinematics_joint_limit_gradient(model, joint_angles, lim_grad);
                for (int i = 0; i < ndof; i++)
                    grad_composite[i] += w_limit * lim_grad[i];
            }
            if (w_sing > 0.0f) {
                float sing_grad[KINEMATICS_MAX_DOF];
                kinematics_singularity_avoidance_gradient(jac, ndof, sing_grad);
                for (int i = 0; i < ndof; i++)
                    grad_composite[i] += w_sing * sing_grad[i];
            }
            kinematics_null_space_project(jac, ndof, grad_composite, gradient_ns);
        }
        int j_idx = 0;
        for (int i = 0; i <= ee_idx; i++) {
            if (model->joints[i].joint_type == JOINT_TYPE_FIXED) continue;
            joint_angles[i] += step_size * (dq_primary[j_idx] + gradient_ns[j_idx]);
            if (model->joints[i].joint_type != JOINT_TYPE_CONTINUOUS)
                joint_angles[i] = KINEMATICS_CLAMP(joint_angles[i],
                    model->joints[i].joint_limit_lower, model->joints[i].joint_limit_upper);
            j_idx++;
        }
        safe_free((void**)&jac);
    }
    if (result < 0) {
        EndEffectorState final;
        forward_kinematics(model, joint_angles, &final);
        float df = sqrtf((final.position.x - target_pos->x) * (final.position.x - target_pos->x) +
                         (final.position.y - target_pos->y) * (final.position.y - target_pos->y) +
                         (final.position.z - target_pos->z) * (final.position.z - target_pos->z));
        if (df < tolerance * 10) result = max_iter;
    }
    return result;
}

/* ============================================================================
 * DLS (阻尼最小二乘法) 逆运动学求解器
 * 相比传统Jacobian伪逆，DLS在接近奇异点时通过阻尼避免数值发散
 * 公式: Δθ = J^T (J·J^T + λ²I)^{-1} Δx
 * λ = λ_max * (1 - ω/ω_max)² 当可操作性低于阈值时激活
 * ============================================================================ */
/**
 * @brief 逆运动学求解 - DLS阻尼最小二乘法（公共API，自适应lambda版本）
 * 
 * 本函数与 inverse_kinematics_dls() 是两种独立的DLS实现：
 * - ik_solve_dls: 使用欧拉角姿态误差、静态栈分配(32关节)、自适应阻尼因子
 *   根据可操作性 ω = sqrt(trace(J·J^T)/6) 动态调整阻尼因子lambda
 * - inverse_kinematics_dls: 使用四元数姿态误差、动态堆分配、固定阻尼因子
 * 两者均为有效的DLS实现，根据调用场景选择使用。
 * 
 * @param model 运动学模型
 * @param target_pos 目标末端位置
 * @param target_orient 目标姿态（欧拉角:[roll,pitch,yaw]，NULL表示忽略）
 * @param joint_angles [输入/输出] 关节角
 * @param max_iter 最大迭代次数
 * @param tolerance 收敛容差
 * @param lambda_max 最大阻尼系数
 * @param manipulability_threshold 可操作性阈值
 * @return 0成功，-1失败
 */
int ik_solve_dls(const KinematicModel* model, const Vec3* target_pos,
                 const float* target_orient, float* joint_angles,
                 int max_iter, float tolerance, float lambda_max, float manipulability_threshold) {
    if (!model || !target_pos || !joint_angles || model->joint_count <= 0) return -1;
    int n = model->joint_count;
    float lambda = 0.0f;
    float damp_thresh = (manipulability_threshold > 0.0f) ? manipulability_threshold : 0.01f;
    float lmax = (lambda_max > 0.0f) ? lambda_max : 0.5f;

    for (int iter = 0; iter < max_iter; iter++) {
        EndEffectorState current;
        forward_kinematics(model, joint_angles, &current);

        float dx[6];
        dx[0] = target_pos->x - current.position.x;
        dx[1] = target_pos->y - current.position.y;
        dx[2] = target_pos->z - current.position.z;

        float rot_err[3];
        if (target_orient) {
            for (int i = 0; i < 3; i++)
                rot_err[i] = target_orient[i] - current.orientation[i];
            dx[3] = rot_err[0]; dx[4] = rot_err[1]; dx[5] = rot_err[2];
        } else {
            dx[3] = dx[4] = dx[5] = 0.0f;
        }

        float pos_err = sqrtf(dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]);
        if (pos_err < tolerance) break;

        /* 计算雅可比矩阵 J [6×n] */
        float J[6 * 32]; /* 最多32关节 */
        memset(J, 0, sizeof(J));
        compute_jacobian(model, joint_angles, J, model->joint_count);

        /* ZSFABC-M008深度修复: 计算完整Yoshikawa可操作性指标 ω = sqrt(det(J·J^T))
         * 不再使用简化的迹近似（trace(JJ^T)/N）。
         * 对于6×n雅可比矩阵，JJ^T是6×6矩阵，直接计算其行列式。 */
        float jj[6 * 6] = {0};
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++) {
                    sum += J[i * n + k] * J[j * n + k];
                }
                jj[i * 6 + j] = sum;
            }
        }
        /* 直接计算6x6矩阵的行列式（使用LU分解） */
        float lu[6 * 6];
        memcpy(lu, jj, sizeof(lu));
        float det = 1.0f;
        int sign = 1;
        for (int col = 0; col < 6; col++) {
            /* 选主元 */
            int pivot = col;
            float max_v = fabsf(lu[col * 6 + col]);
            for (int row = col + 1; row < 6; row++) {
                float abs_v = fabsf(lu[row * 6 + col]);
                if (abs_v > max_v) { max_v = abs_v; pivot = row; }
            }
            if (max_v < 1e-12f) { det = 0.0f; break; }
            if (pivot != col) {
                for (int j = 0; j < 6; j++) {
                    float tmp = lu[col * 6 + j];
                    lu[col * 6 + j] = lu[pivot * 6 + j];
                    lu[pivot * 6 + j] = tmp;
                }
                sign = -sign;
            }
            det *= lu[col * 6 + col];
            float inv_pivot = 1.0f / lu[col * 6 + col];
            for (int row = col + 1; row < 6; row++) {
                float factor = lu[row * 6 + col] * inv_pivot;
                for (int j = col + 1; j < 6; j++) {
                    lu[row * 6 + j] -= factor * lu[col * 6 + j];
                }
            }
        }
        det *= (float)sign;
        if (det < 0.0f) det = -det;
        float omega = sqrtf(det);

        /* DLS阻尼因子 */
        if (omega < damp_thresh) {
            float ratio = omega / damp_thresh;
            lambda = lmax * (1.0f - ratio) * (1.0f - ratio);
        } else {
            lambda = 0.0f;
        }

        /* (J·J^T + λ²I)^{-1} */
        for (int i = 0; i < 6; i++) jj[i*6+i] += lambda * lambda;

        /* 高斯消元求 J·J^T + λ²I 的逆 × dx */
        float rhs[6];
        memcpy(rhs, dx, sizeof(rhs));
        for (int i = 0; i < 6; i++) {
            float pivot = jj[i*6+i];
            if (fabsf(pivot) < 1e-12f) continue;
            for (int j = i+1; j < 6; j++) {
                float factor = jj[j*6+i] / pivot;
                for (int k = i; k < 6; k++) jj[j*6+k] -= factor * jj[i*6+k];
                rhs[j] -= factor * rhs[i];
            }
        }
        float inv_dx[6];
        for (int i = 5; i >= 0; i--) {
            inv_dx[i] = rhs[i];
            for (int j = i+1; j < 6; j++) inv_dx[i] -= jj[i*6+j] * inv_dx[j];
            if (fabsf(jj[i*6+i]) > 1e-12f) inv_dx[i] /= jj[i*6+i];
        }

        /* Δθ = J^T · inv(J·J^T+λ²I) · Δx */
        for (int j = 0; j < n; j++) {
            float dtheta = 0.0f;
            for (int i = 0; i < 6; i++) dtheta += J[i*n+j] * inv_dx[i];
            joint_angles[j] += dtheta;
        }

        if (lambda < 1e-8f && pos_err < tolerance) return iter + 1;
    }
    return max_iter;
}

/* ================================================================
 * 人体姿态→机器人关节映射（模仿学习关键桥接）
 * 将3D人体骨架关键点映射到机器人关节角度
 * ================================================================ */

#define BODY_POSE_KEYPOINTS 17
#define BODY_POSE_2D_FACTOR 0.5f

typedef struct {
    float position[3];    /* 3D关键点位置(m)，以髋部为原点 */
    float confidence;     /* 关键点检测置信度 */
} BodyKeypoint;

typedef struct {
    BodyKeypoint keypoints[BODY_POSE_KEYPOINTS];
    float timestamp;
    int width, height;   /* 图像尺寸（用于2D回退） */
} BodyPoseEstimation;

/* COCO 17关键点映射索引 */
enum {
    KP_NOSE=0, KP_NECK=1, KP_R_SHOULDER=2, KP_R_ELBOW=3, KP_R_WRIST=4,
    KP_L_SHOULDER=5, KP_L_ELBOW=6, KP_L_WRIST=7,
    KP_R_HIP=8, KP_R_KNEE=9, KP_R_ANKLE=10,
    KP_L_HIP=11, KP_L_KNEE=12, KP_L_ANKLE=13,
    KP_R_EYE=14, KP_L_EYE=15, KP_R_EAR=16
};

/**
 * @brief 计算两个关键点之间的3D方向向量
 */
static void kp_direction(const BodyKeypoint* a, const BodyKeypoint* b, float dir[3]) {
    dir[0] = b->position[0] - a->position[0];
    dir[1] = b->position[1] - a->position[1];
    dir[2] = b->position[2] - a->position[2];
    float len = sqrtf(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);
    if (len > 1e-6f) { dir[0]/=len; dir[1]/=len; dir[2]/=len; }
}

/**
 * @brief 计算两个关键点之间的距离（用于肢体长度估计）
 */
static float kp_distance(const BodyKeypoint* a, const BodyKeypoint* b) {
    float dx = b->position[0]-a->position[0];
    float dy = b->position[1]-a->position[1];
    float dz = b->position[2]-a->position[2];
    return sqrtf(dx*dx+dy*dy+dz*dz);
}

/**
 * @brief 利用向量投影计算关节弯曲角度
 * @param shoulder_dir 上臂方向
 * @param elbow_dir 前臂方向
 * @return 关节角度(rad)
 */
static float compute_bend_angle(const float upper_dir[3], const float lower_dir[3]) {
    float dot = upper_dir[0]*lower_dir[0] + upper_dir[1]*lower_dir[1] + upper_dir[2]*lower_dir[2];
    if (dot > 1.0f) dot = 1.0f;
    if (dot < -1.0f) dot = -1.0f;
    return acosf(dot);
}

/**
 * @brief 人体3D姿态估计→机器人关节角度重定向
 *
 * 将检测到的3D人体骨架关键点映射到目标机器人的关节角度。
 * 使用几何重定向方法：通过3D骨架的肢体方向向量计算对应关节的弯曲角度。
 *
 * @param pose 人体姿态估计结果（17个COCO关键点）
 * @param robot_limits_min 机器人各关节最小限位(rad)
 * @param robot_limits_max 机器人各关节最大限位(rad)
 * @param robot_joint_angles 输出：重定向后的机器人关节角度(rad)
 * @param num_joints 机器人关节数（6=工业臂, >6=人形）
 * @param scaling_factors 身体到机器人的缩放因子[num_joints]
 * @return 0=成功, -1=失败
 */
int body_pose_to_robot_joints(const BodyPoseEstimation* pose,
                               const float* robot_limits_min,
                               const float* robot_limits_max,
                               float* robot_joint_angles,
                               int num_joints,
                               const float* scaling_factors) {
    if (!pose || !robot_joint_angles || num_joints <= 0) return -1;

    const BodyKeypoint* kp = pose->keypoints;

    /* 验证关键点置信度 */
    int required_kps[] = {KP_R_SHOULDER, KP_R_ELBOW, KP_R_WRIST, KP_L_SHOULDER,
                          KP_L_ELBOW, KP_L_WRIST, KP_R_HIP, KP_R_KNEE,
                          KP_L_HIP, KP_L_KNEE, KP_NECK};
    int n_required = sizeof(required_kps) / sizeof(required_kps[0]);
    for (int i = 0; i < n_required; i++) {
        if (kp[required_kps[i]].confidence < 0.3f) return -1;
    }

    float r_upper_arm[3], r_forearm[3];
    float l_upper_arm[3], l_forearm[3];
    float r_thigh[3], r_shin[3];
    float l_thigh[3], l_shin[3];
    float neck_to_hip[3];

    kp_direction(&kp[KP_R_SHOULDER], &kp[KP_R_ELBOW], r_upper_arm);
    kp_direction(&kp[KP_R_ELBOW], &kp[KP_R_WRIST], r_forearm);
    kp_direction(&kp[KP_L_SHOULDER], &kp[KP_L_ELBOW], l_upper_arm);
    kp_direction(&kp[KP_L_ELBOW], &kp[KP_L_WRIST], l_forearm);
    kp_direction(&kp[KP_R_HIP], &kp[KP_R_KNEE], r_thigh);
    kp_direction(&kp[KP_R_KNEE], &kp[KP_R_ANKLE], r_shin);
    kp_direction(&kp[KP_L_HIP], &kp[KP_L_KNEE], l_thigh);
    kp_direction(&kp[KP_L_KNEE], &kp[KP_L_ANKLE], l_shin);

    /* 清除输出 */
    memset(robot_joint_angles, 0, (size_t)num_joints * sizeof(float));

    float scale = scaling_factors ? scaling_factors[0] : 1.0f;

    /* 右臂（对应机器人肩部俯仰/滚动/肘部） */
    if (num_joints > 0) robot_joint_angles[0] = atan2f(r_upper_arm[2], r_upper_arm[1]) * scale;
    if (num_joints > 1) robot_joint_angles[1] = atan2f(r_upper_arm[0], r_upper_arm[1]) * scale * 0.5f;
    if (num_joints > 2) robot_joint_angles[2] = compute_bend_angle(r_upper_arm, r_forearm) * scale;

    /* 左臂 */
    if (num_joints > 3) robot_joint_angles[3] = atan2f(l_upper_arm[2], l_upper_arm[1]) * scale;
    if (num_joints > 4) robot_joint_angles[4] = atan2f(l_upper_arm[0], l_upper_arm[1]) * scale * 0.5f;
    if (num_joints > 5) robot_joint_angles[5] = compute_bend_angle(l_upper_arm, l_forearm) * scale;

    /* 腿部（人形机器人） */
    if (num_joints > 6) robot_joint_angles[6] = compute_bend_angle(r_thigh, r_shin) * scale;
    if (num_joints > 7) robot_joint_angles[7] = compute_bend_angle(l_thigh, l_shin) * scale;

    /* 躯干弯曲（颈部到髋部的倾斜） */
    kp_direction(&kp[KP_NECK], kp + KP_R_HIP, neck_to_hip);
    if (num_joints > 8) robot_joint_angles[8] = atan2f(neck_to_hip[0], neck_to_hip[2]) * scale;

    /* 关节限位裁剪 */
    for (int j = 0; j < num_joints; j++) {
        if (robot_limits_min) {
            if (robot_joint_angles[j] < robot_limits_min[j])
                robot_joint_angles[j] = robot_limits_min[j];
        }
        if (robot_limits_max) {
            if (robot_joint_angles[j] > robot_limits_max[j])
                robot_joint_angles[j] = robot_limits_max[j];
        }
    }

    return 0;
}