#include "selflnn/robot/physics_engine.h"
#include <string.h>
#include "selflnn/utils/math_utils.h"
#include <math.h>
#include <float.h>

#define PE_PI 3.14159265358979323846f
#define PE_EPS 1e-8f
#define PE_INV_EPS 1e8f

static float pe_min(float a, float b) { return a < b ? a : b; }
static float pe_max(float a, float b) { return a > b ? a : b; }
static float pe_clamp(float v, float lo, float hi) { return pe_min(pe_max(v, lo), hi); }
static float pe_abs(float v) { return v < 0.0f ? -v : v; }
static float pe_sqrt(float v) { return sqrtf(v); }
static float pe_sign(float v) { return v < 0.0f ? -1.0f : 1.0f; }

static void pe_vec3_zero(float* v) { v[0] = 0.0f; v[1] = 0.0f; v[2] = 0.0f; }
static void pe_vec3_copy(const float* src, float* dst) { dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; }
static float pe_vec3_dot(const float* a, const float* b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
static float pe_vec3_len_sq(const float* v) { return v[0]*v[0] + v[1]*v[1] + v[2]*v[2]; }
static float pe_vec3_len(const float* v) { return pe_sqrt(pe_vec3_len_sq(v)); }

static void pe_vec3_add(const float* a, const float* b, float* out) {
    out[0] = a[0] + b[0]; out[1] = a[1] + b[1]; out[2] = a[2] + b[2];
}
static void pe_vec3_sub(const float* a, const float* b, float* out) {
    out[0] = a[0] - b[0]; out[1] = a[1] - b[1]; out[2] = a[2] - b[2];
}
static void pe_vec3_scale(const float* v, float s, float* out) {
    out[0] = v[0]*s; out[1] = v[1]*s; out[2] = v[2]*s;
}
static void pe_vec3_cross(const float* a, const float* b, float* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}
static void pe_vec3_mul(const float* a, const float* b, float* out) {
    out[0] = a[0]*b[0]; out[1] = a[1]*b[1]; out[2] = a[2]*b[2];
}
static float pe_vec3_dist_sq(const float* a, const float* b) {
    float dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
    return dx*dx + dy*dy + dz*dz;
}

static void pe_quat_identity(float* q) { q[0] = 0.0f; q[1] = 0.0f; q[2] = 0.0f; q[3] = 1.0f; }
/* quat_conjugate: use math_utils.h static inline */

static void pe_quat_mul(const float* a, const float* b, float* out) {
    out[0] = a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1];
    out[1] = a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0];
    out[2] = a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3];
    out[3] = a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2];
}

/* 四元数加法: out = a + b（RK4中间步骤使用） */
static void pe_quat_add(const float* a, const float* b, float* out) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    out[3] = a[3] + b[3];
}

/* quat_normalize: use math_utils.h static inline */

/* quat_from_axis_angle: use math_utils.h static inline */

/* quat_rotate: use math_utils.h static inline */

static void pe_quat_from_angular_vel(const float* w, float dt, float* out) {
    float angle = pe_vec3_len(w) * dt;
    if (angle < PE_EPS) { pe_quat_identity(out); return; }
    float axis[3]; pe_vec3_scale(w, 1.0f / (angle / dt + PE_EPS), axis);
    float axis_len = pe_vec3_len(axis);
    if (axis_len < PE_EPS) { pe_quat_identity(out); return; }
    pe_vec3_scale(axis, 1.0f / axis_len, axis);
    quat_from_axis_angle(axis, angle, out);
}

static void pe_vec3_rotate_by_quat(const float* v, const float* q, float* out) {
    quat_rotate(q, v, out);
}

static void pe_mat3_from_quat(const float* q, float* m) {
    float x=q[0], y=q[1], z=q[2], w=q[3];
    float x2=x*x, y2=y*y, z2=z*z;
    m[0] = 1.0f - 2.0f*(y2+z2); m[1] = 2.0f*(x*y-w*z);     m[2] = 2.0f*(x*z+w*y);
    m[3] = 2.0f*(x*y+w*z);     m[4] = 1.0f - 2.0f*(x2+z2); m[5] = 2.0f*(y*z-w*x);
    m[6] = 2.0f*(x*z-w*y);     m[7] = 2.0f*(y*z+w*x);     m[8] = 1.0f - 2.0f*(x2+y2);
}

int pe_world_init(PEWorld* world) {
    if (!world) return -1;
    memset(world, 0, sizeof(PEWorld));
    world->gravity[0] = 0.0f;
    world->gravity[1] = PE_GRAVITY;
    world->gravity[2] = 0.0f;
    world->dt = 0.01f;
    world->substeps = 4;
    world->solver_iters = PE_SOLVER_ITERATIONS;
    world->enable_ccd = 1;
    world->integrator_type = PE_INTEGRATOR_EULER; /* 默认半隐式Euler（快速） */
    world->enable_adaptive_substep = 0;
    return 0;
}

/* ============================================================================
 * NGS (非线性高斯-赛德尔) 约束求解器
 *
 * 相比传统PGS(投影高斯-赛德尔)，NGS通过以下增强提升收敛性：
 * 1. 非线性接触力模型：法向力 = max(0, k·penetration^n - d·v_contact)
 * 2. 子步长细化：在约束求解内部使用更小的子步长
 * 3. 自适应松弛：根据收敛速率动态调整SOR因子
 * 4. 摩擦锥约束：库仑摩擦锥的精确投影
 * ============================================================================ */

/* 前向声明 */
static void pe_integrate_velocity(PEBody* body, float dt);
static void pe_body_aabb(const PEBody* body, float* min_out, float* max_out);
static int pe_aabb_overlap(const float* min_a, const float* max_a, const float* min_b, const float* max_b);
static int pe_gjk_penetration(const PEBody* body_a, const PEBody* body_b, float* normal, float* point, float* penetration);
static void pe_solve_joint_hinge(PEJoint* j, PEBody* ba, PEBody* bb, float dt);
static void pe_solve_joint_ball(PEJoint* j, PEBody* ba, PEBody* bb, float dt);
static void pe_solve_joint_fixed(PEJoint* j, PEBody* ba, PEBody* bb, float dt);
static void pe_solve_joint_spring(PEJoint* j, PEBody* ba, PEBody* bb, float dt);

static void pe_body_aabb(const PEBody* body, float* min_out, float* max_out) {
    /* ZSFWS-F002修复: 使用刚体自身的bounding_radius而非硬编码0.5m */
    float radius = body->bounding_radius > 0.0f ? body->bounding_radius : 0.5f;
    for (int i = 0; i < 3; i++) {
        min_out[i] = body->pos[i] - radius;
        max_out[i] = body->pos[i] + radius;
    }
}

int pe_world_step_ngs(PEWorld* world, float dt) {
    if (!world || world->body_count < 2) return -1;

    int max_iters = world->solver_iters > 0 ? world->solver_iters : 10;
    float omega = 1.2f; /* SOR over-relaxation factor */
    int sub_steps = 4;
    float sub_dt = dt / (float)sub_steps;

    for (int substep = 0; substep < sub_steps; substep++) {
        for (int i = 0; i < world->body_count; i++) {
            PEBody* b = &world->bodies[i];
            if (!b->props.active || b->props.type != PE_BODY_DYNAMIC) continue;
            /* R4-007修复: 使用局部变量gravity_impulse而非修改全局gravity */
            float gravity_impulse[3];
            gravity_impulse[0] = world->gravity[0] * b->props.inv_mass * sub_dt;
            gravity_impulse[1] = world->gravity[1] * b->props.inv_mass * sub_dt;
            gravity_impulse[2] = world->gravity[2] * b->props.inv_mass * sub_dt;
            if (pe_vec3_len_sq(gravity_impulse) > PE_EPS) {
                b->vel[0] += gravity_impulse[0] * sub_dt;
                b->vel[1] += gravity_impulse[1] * sub_dt;
                b->vel[2] += gravity_impulse[2] * sub_dt;
            }
            pe_integrate_velocity(b, sub_dt);
        }

        /* 碰撞检测 */
        int contact_count = 0;
        for (int i = 0; i < world->body_count && contact_count < PE_MAX_CONTACTS; i++) {
            PEBody* ba = &world->bodies[i];
            if (!ba->props.active) continue;
            for (int j = i + 1; j < world->body_count && contact_count < PE_MAX_CONTACTS; j++) {
                PEBody* bb = &world->bodies[j];
                if (!bb->props.active) continue;
                float aabb_min_a[3], aabb_max_a[3], aabb_min_b[3], aabb_max_b[3];
                pe_body_aabb(ba, aabb_min_a, aabb_max_a);
                pe_body_aabb(bb, aabb_min_b, aabb_max_b);
                if (!pe_aabb_overlap(aabb_min_a, aabb_max_a, aabb_min_b, aabb_max_b)) continue;

                float normal[3], point[3], penetration;
                if (pe_gjk_penetration(ba, bb, normal, point, &penetration)) {
                    PEContact* ct = &world->contacts[contact_count++];
                    ct->body_a = i; ct->body_b = j;
                    pe_vec3_copy(normal, ct->normal);
                    pe_vec3_copy(point, ct->point);
                    ct->penetration = penetration;
                    ct->friction = 0.5f; ct->restitution = 0.2f;
                    ct->impulse_n = 0.0f;
                    ct->impulse_t[0] = ct->impulse_t[1] = 0.0f;
                    ct->active = 1;
                }
            }
        }
        world->contact_count = contact_count;

        /* NGS约束求解循环 */
        float prev_error = 1e10f;
        for (int iter = 0; iter < max_iters; iter++) {
            float max_error = 0.0f;

            for (int c = 0; c < world->contact_count; c++) {
                PEContact* ct = &world->contacts[c];
                PEBody* ba = &world->bodies[ct->body_a];
                PEBody* bb = &world->bodies[ct->body_b];
                if (!ba->props.active || !bb->props.active) continue;

                /* 法向约束求解 */
                float inv_mass_eff = ba->props.inv_mass + bb->props.inv_mass;
                float vn = (ba->vel[0] - bb->vel[0]) * ct->normal[0] +
                           (ba->vel[1] - bb->vel[1]) * ct->normal[1] +
                           (ba->vel[2] - bb->vel[2]) * ct->normal[2];

                /* 非线性恢复力: F = k * pen^n - d * v  (Hertz接触) */
                float k_hertz = 1e5f;
                float d_damp = 50.0f;
                float pen = ct->penetration;
                float fn_target = k_hertz * pen * sqrtf(pen) - d_damp * vn;
                if (fn_target < 0.0f) fn_target = 0.0f;

                float delta_v = fn_target * inv_mass_eff * sub_dt;
                float dlambda = (delta_v - vn) / (inv_mass_eff + PE_EPS);

                /* 动量守恒 */
                float accum = ct->impulse_n + dlambda;
                if (accum < 0.0f) { dlambda -= accum; accum = 0.0f; }
                ct->impulse_n = accum;

                float impulse_n[3];
                pe_vec3_scale(ct->normal, omega * dlambda, impulse_n);
                ba->vel[0] += impulse_n[0] * ba->props.inv_mass;
                ba->vel[1] += impulse_n[1] * ba->props.inv_mass;
                ba->vel[2] += impulse_n[2] * ba->props.inv_mass;
                bb->vel[0] -= impulse_n[0] * bb->props.inv_mass;
                bb->vel[1] -= impulse_n[1] * bb->props.inv_mass;
                bb->vel[2] -= impulse_n[2] * bb->props.inv_mass;

                /* 摩擦约束 (库仑摩擦锥) */
                float vt1 = (ba->vel[0] - bb->vel[0]) * (ct->point[0] - ba->pos[0]);
                float vt2 = (ba->vel[1] - bb->vel[1]) * (ct->point[1] - ba->pos[1]);
                float mu = ct->friction;
                float max_friction = mu * pe_abs(ct->impulse_n);

                float ft_mag = sqrtf(vt1*vt1 + vt2*vt2 + 1e-12f);
                if (ft_mag > PE_EPS) {
                    float ft_clamp = ft_mag > max_friction ? max_friction / ft_mag : 1.0f;
                    ba->vel[0] -= vt1 * ft_clamp * ba->props.inv_mass * omega;
                    ba->vel[1] -= vt2 * ft_clamp * ba->props.inv_mass * omega;
                    bb->vel[0] += vt1 * ft_clamp * bb->props.inv_mass * omega;
                    bb->vel[1] += vt2 * ft_clamp * bb->props.inv_mass * omega;
                }

                float err = pe_abs(vn);
                if (err > max_error) max_error = err;
            }

            /* 自适应松弛因子调整 */
            if (max_error > prev_error * 0.9f && omega > 1.0f) omega *= 0.9f;
            if (max_error < prev_error * 0.5f && omega < 1.5f) omega *= 1.05f;
            if (omega < 1.0f) omega = 1.0f;
            if (omega > 1.5f) omega = 1.5f;
            prev_error = max_error;

            if (max_error < 1e-6f) break;

            /* 关节求解 */
            for (int j = 0; j < world->joint_count; j++) {
                PEJoint* jnt = &world->joints[j];
                if (!jnt->active) continue;
                PEBody* ba = &world->bodies[jnt->body_a];
                PEBody* bb = &world->bodies[jnt->body_b];
                if (!ba->props.active || !bb->props.active) continue;
                switch (jnt->type) {
                    case PE_JOINT_HINGE: pe_solve_joint_hinge(jnt, ba, bb, sub_dt); break;
                    case PE_JOINT_BALL: pe_solve_joint_ball(jnt, ba, bb, sub_dt); break;
                    case PE_JOINT_FIXED: pe_solve_joint_fixed(jnt, ba, bb, sub_dt); break;
                    case PE_JOINT_SPRING: pe_solve_joint_spring(jnt, ba, bb, sub_dt); break;
                    default: break;
                }
            }
        }
    }
    return 0;
}

int pe_body_add(PEWorld* world, const float* pos, const float* quat, float mass, PEBodyType type) {
    if (!world || world->body_count >= PE_MAX_BODIES) return -1;
    int id = world->body_count;
    PEBody* b = &world->bodies[id];
    memset(b, 0, sizeof(PEBody));
    b->body_id = id;
    pe_vec3_copy(pos, b->pos);
    if (quat) { b->quat[0]=quat[0]; b->quat[1]=quat[1]; b->quat[2]=quat[2]; b->quat[3]=quat[3]; }
    else pe_quat_identity(b->quat);
    b->props.mass = mass;
    b->props.type = type;
    if (type == PE_BODY_STATIC || type == PE_BODY_KINEMATIC) {
        b->props.inv_mass = 0.0f;
        b->props.inv_inertia[0] = 0.0f;
        b->props.inv_inertia[1] = 0.0f;
        b->props.inv_inertia[2] = 0.0f;
    } else {
        b->props.inv_mass = 1.0f / (mass + PE_EPS);
        /* ZSFWS-F002修复: 使用刚体bounding_radius计算惯性张量，替代硬编码0.5m球体 */
        float radius = b->bounding_radius > 0.0f ? b->bounding_radius : 0.5f;
        float ixx = 0.4f * mass * radius * radius;
        b->props.inertia[0] = ixx;
        b->props.inertia[1] = ixx;
        b->props.inertia[2] = ixx;
        b->props.inv_inertia[0] = 1.0f / (ixx + PE_EPS);
        b->props.inv_inertia[1] = 1.0f / (ixx + PE_EPS);
        b->props.inv_inertia[2] = 1.0f / (ixx + PE_EPS);
    }
    b->props.active = 1;
    world->body_count++;
    return id;
}

int pe_body_remove(PEWorld* world, int body_id) {
    if (!world || body_id < 0 || body_id >= world->body_count) return -1;
    world->bodies[body_id].props.active = 0;
    return 0;
}

int pe_body_apply_force(PEWorld* world, int body_id, const float* force) {
    if (!world || body_id < 0 || body_id >= world->body_count) return -1;
    PEBody* b = &world->bodies[body_id];
    if (!b->props.active) return -1;
    b->force_acc[0] += force[0];
    b->force_acc[1] += force[1];
    b->force_acc[2] += force[2];
    return 0;
}

int pe_body_apply_torque(PEWorld* world, int body_id, const float* torque) {
    if (!world || body_id < 0 || body_id >= world->body_count) return -1;
    PEBody* b = &world->bodies[body_id];
    if (!b->props.active) return -1;
    b->torque_acc[0] += torque[0];
    b->torque_acc[1] += torque[1];
    b->torque_acc[2] += torque[2];
    return 0;
}

int pe_body_get_state(const PEWorld* world, int body_id, float* pos, float* quat, float* vel, float* ang_vel) {
    if (!world || body_id < 0 || body_id >= world->body_count) return -1;
    const PEBody* b = &world->bodies[body_id];
    if (pos) pe_vec3_copy(b->pos, pos);
    if (quat) pe_vec3_copy(b->quat, quat);
    if (vel) pe_vec3_copy(b->vel, vel);
    if (ang_vel) pe_vec3_copy(b->ang_vel, ang_vel);
    return 0;
}

int pe_body_set_kinematic(PEWorld* world, int body_id, const float* pos, const float* quat, const float* vel, const float* ang_vel) {
    if (!world || body_id < 0 || body_id >= world->body_count) return -1;
    PEBody* b = &world->bodies[body_id];
    if (b->props.type != PE_BODY_KINEMATIC) return -1;
    if (pos) pe_vec3_copy(pos, b->pos);
    if (quat) { b->quat[0]=quat[0]; b->quat[1]=quat[1]; b->quat[2]=quat[2]; b->quat[3]=quat[3]; }
    if (vel) pe_vec3_copy(vel, b->vel);
    if (ang_vel) pe_vec3_copy(ang_vel, b->ang_vel);
    return 0;
}

int pe_joint_add(PEWorld* world, PEJointType type, int body_a, int body_b, const float* pivot_a, const float* pivot_b, const float* axis, float limit_lower, float limit_upper) {
    if (!world || world->joint_count >= PE_MAX_JOINTS) return -1;
    if (body_a < 0 || body_a >= world->body_count || body_b < 0 || body_b >= world->body_count) return -1;
    int id = world->joint_count;
    PEJoint* j = &world->joints[id];
    memset(j, 0, sizeof(PEJoint));
    j->type = type;
    j->body_a = body_a;
    j->body_b = body_b;
    if (pivot_a) pe_vec3_copy(pivot_a, j->pivot_a);
    if (pivot_b) pe_vec3_copy(pivot_b, j->pivot_b);
    if (axis) {
        float len = pe_vec3_len(axis);
        if (len > PE_EPS) { pe_vec3_scale(axis, 1.0f/len, j->axis_a); pe_vec3_copy(j->axis_a, j->axis_b); }
        else { j->axis_a[1] = 1.0f; j->axis_b[1] = 1.0f; }
    } else {
        j->axis_a[1] = 1.0f; j->axis_b[1] = 1.0f;
    }
    j->limit_lower = limit_lower;
    j->limit_upper = limit_upper;
    j->max_force = 1e6f;
    j->stiffness = 0.0f;
    j->damping = 0.0f;
    j->motor_speed = 0.0f;
    j->motor_max_torque = 0.0f;
    j->angle = 0.0f;
    j->active = 1;
    world->joint_count++;
    return id;
}

int pe_joint_set_motor(PEWorld* world, int joint_id, float speed, float max_torque) {
    if (!world || joint_id < 0 || joint_id >= world->joint_count) return -1;
    PEJoint* j = &world->joints[joint_id];
    j->motor_speed = speed;
    j->motor_max_torque = max_torque;
    return 0;
}

int pe_joint_get_angle(const PEWorld* world, int joint_id, float* angle) {
    if (!world || joint_id < 0 || joint_id >= world->joint_count || !angle) return -1;
    *angle = world->joints[joint_id].angle;
    return 0;
}

void pe_collision_aabb(const PEBody* body, float* aabb_min, float* aabb_max) {
    float extent = 0.5f;
    aabb_min[0] = body->pos[0] - extent;
    aabb_min[1] = body->pos[1] - extent;
    aabb_min[2] = body->pos[2] - extent;
    aabb_max[0] = body->pos[0] + extent;
    aabb_max[1] = body->pos[1] + extent;
    aabb_max[2] = body->pos[2] + extent;
}

/* ============================================================================
 * 6自由度刚体动力学 — 力/力矩积分 + 欧拉方程 + 四元数积分
 * ============================================================================ */

static void pe_integrate_forces(PEBody* body, float dt) {
    if (!body->props.active || body->props.type != PE_BODY_DYNAMIC) return;

    float total_force[3];
    pe_vec3_copy(body->force_acc, total_force);

    float acc[3] = {
        total_force[0] * body->props.inv_mass,
        total_force[1] * body->props.inv_mass,
        total_force[2] * body->props.inv_mass
    };

    body->vel[0] += acc[0] * dt;
    body->vel[1] += acc[1] * dt;
    body->vel[2] += acc[2] * dt;

    float R[9];
    pe_mat3_from_quat(body->quat, R);

    float world_inv_inertia[9];
    world_inv_inertia[0] = R[0]*body->props.inv_inertia[0]*R[0] + R[1]*body->props.inv_inertia[1]*R[1] + R[2]*body->props.inv_inertia[2]*R[2];
    world_inv_inertia[1] = R[0]*body->props.inv_inertia[0]*R[3] + R[1]*body->props.inv_inertia[1]*R[4] + R[2]*body->props.inv_inertia[2]*R[5];
    world_inv_inertia[2] = R[0]*body->props.inv_inertia[0]*R[6] + R[1]*body->props.inv_inertia[1]*R[7] + R[2]*body->props.inv_inertia[2]*R[8];
    world_inv_inertia[3] = R[3]*body->props.inv_inertia[0]*R[0] + R[4]*body->props.inv_inertia[1]*R[1] + R[5]*body->props.inv_inertia[2]*R[2];
    world_inv_inertia[4] = R[3]*body->props.inv_inertia[0]*R[3] + R[4]*body->props.inv_inertia[1]*R[4] + R[5]*body->props.inv_inertia[2]*R[5];
    world_inv_inertia[5] = R[3]*body->props.inv_inertia[0]*R[6] + R[4]*body->props.inv_inertia[1]*R[7] + R[5]*body->props.inv_inertia[2]*R[8];
    world_inv_inertia[6] = R[6]*body->props.inv_inertia[0]*R[0] + R[7]*body->props.inv_inertia[1]*R[1] + R[8]*body->props.inv_inertia[2]*R[2];
    world_inv_inertia[7] = R[6]*body->props.inv_inertia[0]*R[3] + R[7]*body->props.inv_inertia[1]*R[4] + R[8]*body->props.inv_inertia[2]*R[5];
    world_inv_inertia[8] = R[6]*body->props.inv_inertia[0]*R[6] + R[7]*body->props.inv_inertia[1]*R[7] + R[8]*body->props.inv_inertia[2]*R[8];

    float ang_acc[3];
    pe_vec3_mul(world_inv_inertia, body->torque_acc, ang_acc);

    body->ang_vel[0] += ang_acc[0] * dt;
    body->ang_vel[1] += ang_acc[1] * dt;
    body->ang_vel[2] += ang_acc[2] * dt;

    pe_vec3_zero(body->force_acc);
    pe_vec3_zero(body->torque_acc);
}

static void pe_integrate_velocity(PEBody* body, float dt) {
    if (!body->props.active || body->props.type != PE_BODY_DYNAMIC) return;

    body->pos[0] += body->vel[0] * dt;
    body->pos[1] += body->vel[1] * dt;
    body->pos[2] += body->vel[2] * dt;

    float dq[4];
    pe_quat_from_angular_vel(body->ang_vel, dt, dq);
    float new_q[4];
    pe_quat_mul(dq, body->quat, new_q);
    quat_normalize(new_q);
    pe_vec3_copy(new_q, body->quat);
}

/* ============================================================================
 * RK4 (4阶龙格-库塔) 刚体动力学积分器
 *
 * 相比半隐式Euler的O(dt)截断误差，RK4提供O(dt⁴)截断误差，
 * 显著提升仿真精度，尤其适用于：
 * - 高加速度场景（碰撞、爆炸）
 * - 长时间仿真（轨道力学、空间机器人）
 * - 高精度训练（RL策略需要精确物理反馈）
 *
 * 代价：每步4次力计算（vs Euler的1次），约3-4倍计算开销。
 * 通过自适应子步长机制可在精度与性能间平衡。
 * ============================================================================ */

#define PE_RK4_MAX_BODIES PE_MAX_BODIES

typedef struct {
    float pos[3], vel[3], quat[4], ang_vel[3];
} PERK4State;

/* 计算刚体在给定状态下的加速度（力累积→加速度） */
static void pe_rk4_compute_accel(PEWorld* world, const float* pos, const float* vel,
                                  const float* quat, const float* ang_vel,
                                  float mass, float inv_mass,
                                  const float* inv_inertia,
                                  float* out_acc, float* out_ang_acc) {
    if (inv_mass < 1e-10f) { pe_vec3_zero(out_acc); pe_vec3_zero(out_ang_acc); return; }

    /* 外力：重力 */
    out_acc[0] = world->gravity[0];
    out_acc[1] = world->gravity[1];
    out_acc[2] = world->gravity[2];

    /* 线性阻尼 */
    float linear_damping = 0.01f;
    out_acc[0] -= linear_damping * vel[0] * inv_mass;
    out_acc[1] -= linear_damping * vel[1] * inv_mass;
    out_acc[2] -= linear_damping * vel[2] * inv_mass;

    /* 旋转世界惯性到局部，计算角加速度 */
    float R[9];
    pe_mat3_from_quat(quat, R);
    float world_inv_inertia[9];
    world_inv_inertia[0]=R[0]*inv_inertia[0]*R[0]+R[1]*inv_inertia[1]*R[1]+R[2]*inv_inertia[2]*R[2];
    world_inv_inertia[1]=R[0]*inv_inertia[0]*R[3]+R[1]*inv_inertia[1]*R[4]+R[2]*inv_inertia[2]*R[5];
    world_inv_inertia[2]=R[0]*inv_inertia[0]*R[6]+R[1]*inv_inertia[1]*R[7]+R[2]*inv_inertia[2]*R[8];
    world_inv_inertia[3]=R[3]*inv_inertia[0]*R[0]+R[4]*inv_inertia[1]*R[1]+R[5]*inv_inertia[2]*R[2];
    world_inv_inertia[4]=R[3]*inv_inertia[0]*R[3]+R[4]*inv_inertia[1]*R[4]+R[5]*inv_inertia[2]*R[5];
    world_inv_inertia[5]=R[3]*inv_inertia[0]*R[6]+R[4]*inv_inertia[1]*R[7]+R[5]*inv_inertia[2]*R[8];
    world_inv_inertia[6]=R[6]*inv_inertia[0]*R[0]+R[7]*inv_inertia[1]*R[1]+R[8]*inv_inertia[2]*R[2];
    world_inv_inertia[7]=R[6]*inv_inertia[0]*R[3]+R[7]*inv_inertia[1]*R[4]+R[8]*inv_inertia[2]*R[5];
    world_inv_inertia[8]=R[6]*inv_inertia[0]*R[6]+R[7]*inv_inertia[1]*R[7]+R[8]*inv_inertia[2]*R[8];

    /* 角速度阻尼+世界坐标系角加速度 */
    float ang_damping = 0.01f;
    float torque_local[3];
    pe_vec3_rotate_by_quat(ang_vel, quat, torque_local);
    float damp_torque[3] = {
        -ang_damping * torque_local[0],
        -ang_damping * torque_local[1],
        -ang_damping * torque_local[2]
    };
    pe_vec3_mul(world_inv_inertia, damp_torque, out_ang_acc);
}

/* 单步RK4推进一个刚体 */
static void pe_rk4_step_body(PEBody* body, float dt, const float* gravity) {
    if (!body->props.active || body->props.type != PE_BODY_DYNAMIC) return;

    float mass = body->props.mass;
    float inv_mass = body->props.inv_mass;
    const float* inv_inertia = body->props.inv_inertia;

    /* 保存初始状态 y0 */
    PERK4State y0;
    pe_vec3_copy(body->pos, y0.pos);
    pe_vec3_copy(body->vel, y0.vel);
    pe_vec3_copy(body->quat, y0.quat);
    pe_vec3_copy(body->ang_vel, y0.ang_vel);

    /* k1 = f(y0) */
    float k1_v[3], k1_w[3], k1_pos[3], k1_ang[3];
    pe_rk4_compute_accel(NULL, y0.pos, y0.vel, y0.quat, y0.ang_vel,
                          mass, inv_mass, inv_inertia, k1_v, k1_w);
    pe_vec3_copy(y0.vel, k1_pos);  /* dr/dt = v */
    /* 角速度→四元数导数：dq/dt = 0.5 * omega * q */
    { float dq[4]; pe_quat_from_angular_vel(y0.ang_vel, 1.0f, dq); k1_ang[0]=dq[0]; k1_ang[1]=dq[1]; k1_ang[2]=dq[2]; }

    /* k2 = f(y0 + 0.5*dt*k1) */
    float h2 = 0.5f * dt;
    float y1_pos[3], y1_vel[3], y1_quat[4], y1_angvel[3];
    for (int i=0;i<3;i++) { y1_pos[i]=y0.pos[i]+h2*k1_pos[i]; y1_vel[i]=y0.vel[i]+h2*k1_v[i]; y1_angvel[i]=y0.ang_vel[i]+h2*k1_w[i]; }
    { float tmp_q[4], dq[4]; pe_vec3_scale(k1_ang, h2, dq); pe_quat_add(y0.quat, dq, tmp_q); quat_normalize(tmp_q); pe_vec3_copy(tmp_q, y1_quat); }

    float k2_v[3], k2_w[3], k2_pos[3], k2_ang[3];
    pe_rk4_compute_accel(NULL, y1_pos, y1_vel, y1_quat, y1_angvel,
                          mass, inv_mass, inv_inertia, k2_v, k2_w);
    pe_vec3_copy(y1_vel, k2_pos);
    { float dq[4]; pe_quat_from_angular_vel(y1_angvel, 1.0f, dq); k2_ang[0]=dq[0]; k2_ang[1]=dq[1]; k2_ang[2]=dq[2]; }

    /* k3 = f(y0 + 0.5*dt*k2) */
    float y2_pos[3], y2_vel[3], y2_quat[4], y2_angvel[3];
    for (int i=0;i<3;i++) { y2_pos[i]=y0.pos[i]+h2*k2_pos[i]; y2_vel[i]=y0.vel[i]+h2*k2_v[i]; y2_angvel[i]=y0.ang_vel[i]+h2*k2_w[i]; }
    { float tmp_q[4], dq[4]; pe_vec3_scale(k2_ang, h2, dq); pe_quat_add(y0.quat, dq, tmp_q); quat_normalize(tmp_q); pe_vec3_copy(tmp_q, y2_quat); }

    float k3_v[3], k3_w[3], k3_pos[3], k3_ang[3];
    pe_rk4_compute_accel(NULL, y2_pos, y2_vel, y2_quat, y2_angvel,
                          mass, inv_mass, inv_inertia, k3_v, k3_w);
    pe_vec3_copy(y2_vel, k3_pos);
    { float dq[4]; pe_quat_from_angular_vel(y2_angvel, 1.0f, dq); k3_ang[0]=dq[0]; k3_ang[1]=dq[1]; k3_ang[2]=dq[2]; }

    /* k4 = f(y0 + dt*k3) */
    float y3_pos[3], y3_vel[3], y3_quat[4], y3_angvel[3];
    for (int i=0;i<3;i++) { y3_pos[i]=y0.pos[i]+dt*k3_pos[i]; y3_vel[i]=y0.vel[i]+dt*k3_v[i]; y3_angvel[i]=y0.ang_vel[i]+dt*k3_w[i]; }
    { float tmp_q[4], dq[4]; pe_vec3_scale(k3_ang, dt, dq); pe_quat_add(y0.quat, dq, tmp_q); quat_normalize(tmp_q); pe_vec3_copy(tmp_q, y3_quat); }

    float k4_v[3], k4_w[3], k4_pos[3], k4_ang[3];
    pe_rk4_compute_accel(NULL, y3_pos, y3_vel, y3_quat, y3_angvel,
                          mass, inv_mass, inv_inertia, k4_v, k4_w);
    pe_vec3_copy(y3_vel, k4_pos);
    { float dq[4]; pe_quat_from_angular_vel(y3_angvel, 1.0f, dq); k4_ang[0]=dq[0]; k4_ang[1]=dq[1]; k4_ang[2]=dq[2]; }

    /* 6维RK4更新: y = y0 + (dt/6)*(k1+2k2+2k3+k4) */
    float dt6 = dt / 6.0f;
    for (int i = 0; i < 3; i++) {
        body->pos[i] = y0.pos[i] + dt6*(k1_pos[i] + 2.0f*k2_pos[i] + 2.0f*k3_pos[i] + k4_pos[i]);
        body->vel[i] = y0.vel[i] + dt6*(k1_v[i]   + 2.0f*k2_v[i]   + 2.0f*k3_v[i]   + k4_v[i]);
        body->ang_vel[i] = y0.ang_vel[i] + dt6*(k1_w[i] + 2.0f*k2_w[i] + 2.0f*k3_w[i] + k4_w[i]);
    }
    /* 四元数更新: q += dt6*(k1_q + 2k2_q + 2k3_q + k4_q) 然后归一化 */
    float new_q[4];
    for (int i = 0; i < 3; i++) {
        /* 四元数分量只用前3个（标量部分从归一化隐含） */
        new_q[i] = y0.quat[i] + dt6*(k1_ang[i] + 2.0f*k2_ang[i] + 2.0f*k3_ang[i] + k4_ang[i]);
    }
    new_q[3] = y0.quat[3];
    quat_normalize(new_q);
    pe_vec3_copy(new_q, body->quat);
}

/* 完整的RK4世界步进：所有刚体用RK4推进 */
static int pe_world_step_rk4(PEWorld* world, float dt) {
    if (!world || world->body_count < 2) return -1;

    /* 自适应子步长：根据最大速度/角速度调整 */
    int substeps = world->substeps > 0 ? world->substeps : 4;
    if (world->enable_adaptive_substep) {
        float max_speed = 0.1f;
        for (int i = 0; i < world->body_count; i++) {
            PEBody* b = &world->bodies[i];
            if (!b->props.active) continue;
            float spd = sqrtf(b->vel[0]*b->vel[0]+b->vel[1]*b->vel[1]+b->vel[2]*b->vel[2]);
            if (spd > max_speed) max_speed = spd;
        }
        int adaptive_count = (int)(max_speed * 10.0f / dt) + 2;
        if (adaptive_count < 2) adaptive_count = 2;
        if (adaptive_count > 16) adaptive_count = 16;
        substeps = adaptive_count;
    }

    float sub_dt = dt / (float)substeps;

    for (int substep = 0; substep < substeps; substep++) {
        /* 保存所有刚体状态（用于碰撞回退） */
        float saved_pos[PE_RK4_MAX_BODIES][3];
        float saved_quat[PE_RK4_MAX_BODIES][4];
        for (int i = 0; i < world->body_count; i++) {
            pe_vec3_copy(world->bodies[i].pos, saved_pos[i]);
            pe_vec3_copy(world->bodies[i].quat, saved_quat[i]);
        }

        /* RK4积分所有刚体 */
        for (int i = 0; i < world->body_count; i++) {
            PEBody* b = &world->bodies[i];
            if (!b->props.active || b->props.type != PE_BODY_DYNAMIC) continue;
            pe_rk4_step_body(b, sub_dt, world->gravity);
            pe_vec3_zero(b->force_acc);
            pe_vec3_zero(b->torque_acc);
        }

        /* 碰撞检测与响应（与半隐式Euler共享相同逻辑） */
        int contact_count = 0;
        for (int i = 0; i < world->body_count && contact_count < PE_MAX_CONTACTS; i++) {
            PEBody* ba = &world->bodies[i];
            if (!ba->props.active) continue;
            for (int j = i + 1; j < world->body_count && contact_count < PE_MAX_CONTACTS; j++) {
                PEBody* bb = &world->bodies[j];
                if (!bb->props.active) continue;
                float aabb_min_a[3], aabb_max_a[3], aabb_min_b[3], aabb_max_b[3];
                pe_body_aabb(ba, aabb_min_a, aabb_max_a);
                pe_body_aabb(bb, aabb_min_b, aabb_max_b);
                if (!pe_aabb_overlap(aabb_min_a, aabb_max_a, aabb_min_b, aabb_max_b)) continue;

                float normal[3], point[3], penetration;
                if (pe_gjk_penetration(ba, bb, normal, point, &penetration)) {
                    PEContact* ct = &world->contacts[contact_count++];
                    ct->body_a = i; ct->body_b = j;
                    pe_vec3_copy(normal, ct->normal);
                    pe_vec3_copy(point, ct->point);
                    ct->penetration = penetration;
                    ct->friction = 0.5f; ct->restitution = 0.2f;
                    ct->impulse_n = 0.0f;
                    ct->impulse_t[0] = ct->impulse_t[1] = 0.0f;
                    ct->active = 1;
                }
            }
        }
        world->contact_count = contact_count;

        /* 碰撞响应（NGS求解器 + 冲量施加） */
        float omega = 1.2f;
        int max_iters = world->solver_iters > 0 ? world->solver_iters : 10;
        for (int iter = 0; iter < max_iters; iter++) {
            for (int c = 0; c < contact_count; c++) {
                PEContact* ct = &world->contacts[c];
                if (!ct->active) continue;
                PEBody* ba = &world->bodies[ct->body_a];
                PEBody* bb = &world->bodies[ct->body_b];

                float rel_vel[3];
                pe_vec3_sub(ba->vel, bb->vel, rel_vel);
                float vn = pe_vec3_dot(rel_vel, ct->normal);

                if (ct->penetration > 0.0f) {
                    float ba_inv = (ba->props.type == PE_BODY_DYNAMIC) ? ba->props.inv_mass : 0.0f;
                    float bb_inv = (bb->props.type == PE_BODY_DYNAMIC) ? bb->props.inv_mass : 0.0f;
                    float total_inv = ba_inv + bb_inv;
                    if (total_inv < PE_EPS) continue;

                    float bias = 0.2f * (ct->penetration > 0.05f ? ct->penetration : 0.0f) / sub_dt;
                    float lambda = (-vn + bias) / total_inv;
                    lambda *= omega / (float)max_iters;
                    if (lambda < 0.0f) lambda = 0.0f;

                    float impulse[3];
                    pe_vec3_scale(ct->normal, lambda, impulse);
                    if (ba->props.type == PE_BODY_DYNAMIC) { ba->vel[0]+=impulse[0]*ba_inv; ba->vel[1]+=impulse[1]*ba_inv; ba->vel[2]+=impulse[2]*ba_inv; }
                    if (bb->props.type == PE_BODY_DYNAMIC) { bb->vel[0]-=impulse[0]*bb_inv; bb->vel[1]-=impulse[1]*bb_inv; bb->vel[2]-=impulse[2]*bb_inv; }
                    ct->penetration *= 0.6f;
                }
            }
        }

        /* 关节约束求解 */
        float joint_dt = sub_dt;
        for (int j = 0; j < world->joint_count; j++) {
            PEJoint* jt = &world->joints[j];
            if (!jt->active) continue;
            PEBody* ba = &world->bodies[jt->body_a];
            PEBody* bb = (jt->body_b >= 0) ? &world->bodies[jt->body_b] : NULL;
            switch (jt->type) {
                case PE_JOINT_HINGE: pe_solve_joint_hinge(jt, ba, bb, joint_dt); break;
                case PE_JOINT_BALL:  pe_solve_joint_ball(jt, ba, bb, joint_dt);  break;
                case PE_JOINT_FIXED: pe_solve_joint_fixed(jt, ba, bb, joint_dt); break;
                case PE_JOINT_SPRING: pe_solve_joint_spring(jt, ba, bb, joint_dt); break;
                default: break;
            }
        }
    }

    return 0;
}

/* ============================================================================
 * 增强碰撞检测 — AABB广阶段 + GJK狭阶段 + 接触点生成
 * ============================================================================ */

static int pe_aabb_overlap(const float* min_a, const float* max_a, const float* min_b, const float* max_b) {
    return (min_a[0] <= max_b[0] && max_a[0] >= min_b[0] &&
            min_a[1] <= max_b[1] && max_a[1] >= min_b[1] &&
            min_a[2] <= max_b[2] && max_a[2] >= min_b[2]);
}

static int pe_gjk_penetration(const PEBody* body_a, const PEBody* body_b, float* normal, float* point, float* penetration) {
    float support_a[3], support_b[3], dir[3], mink_diff[3];
    float simplex[4][3];
    int simplex_size = 0;
    dir[0] = 1.0f; dir[1] = 0.0f; dir[2] = 0.0f;
    int max_iter = 50;

    float radius = 0.5f;
    for (int iter = 0; iter < max_iter; iter++) {
        float len = pe_vec3_len(dir);
        if (len < PE_EPS) break;
        float inv_len = 1.0f / len;
        float ndir[3] = { dir[0]*inv_len, dir[1]*inv_len, dir[2]*inv_len };

        float r_dir[3];
        pe_vec3_rotate_by_quat(ndir, body_a->quat, r_dir);
        float r_len = pe_vec3_len(r_dir);
        if (r_len < PE_EPS) { support_a[0] = body_a->pos[0]; support_a[1] = body_a->pos[1]; support_a[2] = body_a->pos[2]; }
        else { pe_vec3_scale(r_dir, radius / r_len, support_a); pe_vec3_add(body_a->pos, support_a, support_a); }

        float neg_dir[3] = { -ndir[0], -ndir[1], -ndir[2] };
        pe_vec3_rotate_by_quat(neg_dir, body_b->quat, r_dir);
        r_len = pe_vec3_len(r_dir);
        if (r_len < PE_EPS) { support_b[0] = body_b->pos[0]; support_b[1] = body_b->pos[1]; support_b[2] = body_b->pos[2]; }
        else { pe_vec3_scale(r_dir, radius / r_len, support_b); pe_vec3_add(body_b->pos, support_b, support_b); }

        pe_vec3_sub(support_a, support_b, mink_diff);

        float dot_dir = pe_vec3_dot(mink_diff, ndir);
        if (simplex_size == 0) {
            pe_vec3_copy(mink_diff, simplex[0]);
            simplex_size = 1;
            pe_vec3_scale(ndir, -1.0f, dir);
            continue;
        }

        if (simplex_size < 4) {
            pe_vec3_copy(mink_diff, simplex[simplex_size]);
            simplex_size++;
        }

        if (simplex_size == 2) {
            float ab[3]; pe_vec3_sub(simplex[1], simplex[0], ab);
            float a0[3]; pe_vec3_scale(simplex[0], -1.0f, a0);
            float cross_dir[3]; pe_vec3_cross(ab, a0, cross_dir);
            pe_vec3_cross(cross_dir, ab, dir);
        } else if (simplex_size == 3) {
            float ab[3], ac[3], a0[3];
            pe_vec3_sub(simplex[1], simplex[0], ab);
            pe_vec3_sub(simplex[2], simplex[0], ac);
            pe_vec3_scale(simplex[0], -1.0f, a0);
            float abc[3]; pe_vec3_cross(ab, ac, abc);
            float tmp[3]; pe_vec3_cross(abc, ac, tmp);
            if (pe_vec3_dot(tmp, a0) > 0) { pe_vec3_cross(abc, ac, dir); }
            else { pe_vec3_cross(ab, abc, dir); }
        }

        if (dot_dir < 0) {
            if (normal) {
                float sep[3]; pe_vec3_sub(body_b->pos, body_a->pos, sep);
                float sep_len = pe_vec3_len(sep);
                if (sep_len > PE_EPS) { pe_vec3_scale(sep, 1.0f/sep_len, normal); }
                else { normal[1] = 1.0f; }
            }
            if (penetration) *penetration = 0.0f;
            if (point) { pe_vec3_add(body_a->pos, body_b->pos, point); pe_vec3_scale(point, 0.5f, point); }
            return 0;
        }

        if (dot_dir > 0 && pe_vec3_dot(mink_diff, ndir) - pe_vec3_dot(simplex[0], ndir) < PE_EPS) break;
    }

    if (normal) {
        float sep[3]; pe_vec3_sub(body_b->pos, body_a->pos, sep);
        float sep_len = pe_vec3_len(sep);
        if (sep_len > PE_EPS) { pe_vec3_scale(sep, 1.0f/sep_len, normal); }
        else { normal[1] = 1.0f; }
    }
    /* ZSFWS-F001修复: 穿透深度使用pe_vec3_dist（欧氏距离）而非pe_vec3_dist_sq（距离平方）。
     * 原代码将线性距离(2*radius)与距离平方混用，导致穿透深度量纲错误。
     * pe_vec3_dist返回线性距离，减法结果量纲一致。 */
    if (penetration) *penetration = radius * 2.0f - pe_vec3_dist(body_a->pos, body_b->pos);
    if (penetration && *penetration < 0) *penetration = 0.0f;
    if (point) {
        pe_vec3_add(body_a->pos, body_b->pos, point);
        pe_vec3_scale(point, 0.5f, point);
    }
    return 1;
}

/* ============================================================================
 * 增强约束求解器 — 顺序脉冲（SI）接触求解 + 摩擦锥
 * ============================================================================ */

static void pe_solve_contact(PEContact* ct, PEBody* body_a, PEBody* body_b, float dt) {
    if (!ct->active) return;

    float inv_mass_a = body_a->props.inv_mass;
    float inv_mass_b = body_b->props.inv_mass;
    float inv_inertia_a[3], inv_inertia_b[3];
    pe_vec3_copy(body_a->props.inv_inertia, inv_inertia_a);
    pe_vec3_copy(body_b->props.inv_inertia, inv_inertia_b);

    float ra[3], rb[3];
    pe_vec3_sub(ct->point, body_a->pos, ra);
    pe_vec3_sub(ct->point, body_b->pos, rb);

    float ra_cross_n[3], rb_cross_n[3];
    pe_vec3_cross(ra, ct->normal, ra_cross_n);
    pe_vec3_cross(rb, ct->normal, rb_cross_n);

    float inv_inertia_ra[3], inv_inertia_rb[3];
    pe_vec3_mul(inv_inertia_a, ra_cross_n, inv_inertia_ra);
    pe_vec3_mul(inv_inertia_b, rb_cross_n, inv_inertia_rb);

    float ja[3], jb[3];
    pe_vec3_cross(inv_inertia_ra, ra, ja);
    pe_vec3_cross(inv_inertia_rb, rb, jb);

    float inv_mass_normal = inv_mass_a + inv_mass_b + pe_vec3_dot(ja, ct->normal) + pe_vec3_dot(jb, ct->normal);
    if (inv_mass_normal < PE_EPS) { ct->active = 0; return; }

    float rel_vel[3];
    pe_vec3_sub(body_a->vel, body_b->vel, rel_vel);
    float ra_cross_w_a[3], rb_cross_w_b[3];
    pe_vec3_cross(body_a->ang_vel, ra, ra_cross_w_a);
    pe_vec3_cross(body_b->ang_vel, rb, rb_cross_w_b);
    rel_vel[0] += ra_cross_w_a[0] - rb_cross_w_b[0];
    rel_vel[1] += ra_cross_w_a[1] - rb_cross_w_b[1];
    rel_vel[2] += ra_cross_w_a[2] - rb_cross_w_b[2];

    float rel_vel_n = pe_vec3_dot(rel_vel, ct->normal);

    float bias = PE_BAUMGARTE * ct->penetration / (dt + PE_EPS);
    if (bias > 0.1f) bias = 0.1f;

    float restitution_vel = rel_vel_n * ct->restitution;
    float denom = 1.0f / inv_mass_normal;
    float d_n = (-rel_vel_n + pe_max(-restitution_vel, 0.0f) + bias) * denom;

    float old_impulse_n = ct->impulse_n;
    ct->impulse_n += d_n;
    if (ct->impulse_n < 0.0f) ct->impulse_n = 0.0f;
    d_n = ct->impulse_n - old_impulse_n;

    float impulse_vec[3];
    pe_vec3_scale(ct->normal, d_n, impulse_vec);

    float dv_a[3], dv_b[3];
    pe_vec3_scale(impulse_vec, inv_mass_a, dv_a);
    pe_vec3_scale(impulse_vec, -inv_mass_b, dv_b);
    body_a->vel[0] += dv_a[0]; body_a->vel[1] += dv_a[1]; body_a->vel[2] += dv_a[2];
    body_b->vel[0] += dv_b[0]; body_b->vel[1] += dv_b[1]; body_b->vel[2] += dv_b[2];

    float torque_a[3], torque_b[3];
    pe_vec3_cross(ra, impulse_vec, torque_a);
    pe_vec3_cross(rb, impulse_vec, torque_b);
    float dw_a[3], dw_b[3];
    pe_vec3_mul(inv_inertia_a, torque_a, dw_a);
    pe_vec3_mul(inv_inertia_b, torque_b, dw_b);
    body_a->ang_vel[0] += dw_a[0]; body_a->ang_vel[1] += dw_a[1]; body_a->ang_vel[2] += dw_a[2];
    body_b->ang_vel[0] -= dw_b[0]; body_b->ang_vel[1] -= dw_b[1]; body_b->ang_vel[2] -= dw_b[2];

    float tangent[2][3];
    float t1[3] = { 1.0f, 0.0f, 0.0f };
    if (pe_abs(ct->normal[0]) > 0.9f) { t1[0] = 0.0f; t1[1] = 1.0f; t1[2] = 0.0f; }
    float cross_t[3];
    pe_vec3_cross(ct->normal, t1, cross_t);
    float ct_len = pe_vec3_len(cross_t);
    if (ct_len > PE_EPS) { pe_vec3_scale(cross_t, 1.0f/ct_len, tangent[0]); }
    else { tangent[0][1] = 1.0f; }
    pe_vec3_cross(ct->normal, tangent[0], tangent[1]);
    float t1_len = pe_vec3_len(tangent[1]);
    if (t1_len > PE_EPS) pe_vec3_scale(tangent[1], 1.0f/t1_len, tangent[1]);

    float max_friction = ct->friction * ct->impulse_n;

    for (int ti = 0; ti < 2; ti++) {
        float ra_cross_t[3], rb_cross_t[3];
        pe_vec3_cross(ra, tangent[ti], ra_cross_t);
        pe_vec3_cross(rb, tangent[ti], rb_cross_t);
        float inv_inertia_ra_t[3], inv_inertia_rb_t[3];
        pe_vec3_mul(inv_inertia_a, ra_cross_t, inv_inertia_ra_t);
        pe_vec3_mul(inv_inertia_b, rb_cross_t, inv_inertia_rb_t);
        float ja_t[3], jb_t[3];
        pe_vec3_cross(inv_inertia_ra_t, ra, ja_t);
        pe_vec3_cross(inv_inertia_rb_t, rb, jb_t);
        float inv_mass_t = inv_mass_a + inv_mass_b + pe_vec3_dot(ja_t, tangent[ti]) + pe_vec3_dot(jb_t, tangent[ti]);
        if (inv_mass_t < PE_EPS) continue;

        float rel_vel_t = pe_vec3_dot(rel_vel, tangent[ti]);
        float denom_t = 1.0f / inv_mass_t;
        float d_t = -rel_vel_t * denom_t;

        float old_t = ct->impulse_t[ti];
        ct->impulse_t[ti] += d_t;
        ct->impulse_t[ti] = pe_clamp(ct->impulse_t[ti], -max_friction, max_friction);
        d_t = ct->impulse_t[ti] - old_t;

        float impulse_t_vec[3];
        pe_vec3_scale(tangent[ti], d_t, impulse_t_vec);

        dv_a[0] = impulse_t_vec[0] * inv_mass_a;
        dv_a[1] = impulse_t_vec[1] * inv_mass_a;
        dv_a[2] = impulse_t_vec[2] * inv_mass_a;
        dv_b[0] = -impulse_t_vec[0] * inv_mass_b;
        dv_b[1] = -impulse_t_vec[1] * inv_mass_b;
        dv_b[2] = -impulse_t_vec[2] * inv_mass_b;
        body_a->vel[0] += dv_a[0]; body_a->vel[1] += dv_a[1]; body_a->vel[2] += dv_a[2];
        body_b->vel[0] += dv_b[0]; body_b->vel[1] += dv_b[1]; body_b->vel[2] += dv_b[2];

        pe_vec3_cross(ra, impulse_t_vec, torque_a);
        pe_vec3_cross(rb, impulse_t_vec, torque_b);
        pe_vec3_mul(inv_inertia_a, torque_a, dw_a);
        pe_vec3_mul(inv_inertia_b, torque_b, dw_b);
        body_a->ang_vel[0] += dw_a[0]; body_a->ang_vel[1] += dw_a[1]; body_a->ang_vel[2] += dw_a[2];
        body_b->ang_vel[0] -= dw_b[0]; body_b->ang_vel[1] -= dw_b[1]; body_b->ang_vel[2] -= dw_b[2];
    }
}

/* ============================================================================
 * 增强关节约束求解器 — 速度级约束（Velocity-Level Constraint）
 * ============================================================================ */

static void pe_solve_joint_hinge(PEJoint* j, PEBody* ba, PEBody* bb, float dt) {
    (void)dt;
    float ra[3], rb[3];
    pe_vec3_sub(j->pivot_a, ba->pos, ra);
    pe_vec3_sub(j->pivot_b, bb->pos, rb);

    float rel_vel[3];
    pe_vec3_sub(ba->vel, bb->vel, rel_vel);
    float ra_cross_wa[3], rb_cross_wb[3];
    pe_vec3_cross(ba->ang_vel, ra, ra_cross_wa);
    pe_vec3_cross(bb->ang_vel, rb, rb_cross_wb);
    rel_vel[0] += ra_cross_wa[0] - rb_cross_wb[0];
    rel_vel[1] += ra_cross_wa[1] - rb_cross_wb[1];
    rel_vel[2] += ra_cross_wa[2] - rb_cross_wb[2];

    float axis_world_a[3], axis_world_b[3];
    pe_vec3_rotate_by_quat(j->axis_a, ba->quat, axis_world_a);
    pe_vec3_rotate_by_quat(j->axis_b, bb->quat, axis_world_b);
    float a_len = pe_vec3_len(axis_world_a);
    float b_len = pe_vec3_len(axis_world_b);
    if (a_len > PE_EPS) pe_vec3_scale(axis_world_a, 1.0f/a_len, axis_world_a);
    if (b_len > PE_EPS) pe_vec3_scale(axis_world_b, 1.0f/b_len, axis_world_b);

    float error[3];
    pe_vec3_sub(axis_world_a, axis_world_b, error);

    float inv_mass_ba = ba->props.inv_mass;
    float inv_mass_bb = bb->props.inv_mass;
    float inv_ia[3], inv_ib[3];
    pe_vec3_copy(ba->props.inv_inertia, inv_ia);
    pe_vec3_copy(bb->props.inv_inertia, inv_ib);

    for (int dim = 0; dim < 3; dim++) {
        float axis[3] = { 0.0f, 0.0f, 0.0f };
        axis[dim] = 1.0f;
        float ra_cross_a[3], rb_cross_a[3];
        pe_vec3_cross(ra, axis, ra_cross_a);
        pe_vec3_cross(rb, axis, rb_cross_a);
        float inv_ia_ra_a[3], inv_ib_rb_a[3];
        pe_vec3_mul(inv_ia, ra_cross_a, inv_ia_ra_a);
        pe_vec3_mul(inv_ib, rb_cross_a, inv_ib_rb_a);
        float ja[3], jb[3];
        pe_vec3_cross(inv_ia_ra_a, ra, ja);
        pe_vec3_cross(inv_ib_rb_a, rb, jb);
        float inv_mass_dim = inv_mass_ba + inv_mass_bb + pe_vec3_dot(ja, axis) + pe_vec3_dot(jb, axis);
        if (inv_mass_dim < PE_EPS) continue;

        float err_scale = 0.1f;
        float bias = error[dim] * err_scale / (dt + PE_EPS);
        if (pe_abs(bias) > 1.0f) bias = pe_sign(bias) * 1.0f;

        float rel_v = pe_vec3_dot(rel_vel, axis) + pe_vec3_dot(ba->ang_vel, ra_cross_a) - pe_vec3_dot(bb->ang_vel, rb_cross_a);
        float lambda = (-rel_v + bias) / inv_mass_dim;

        float impulse[3];
        pe_vec3_scale(axis, lambda, impulse);
        float dv_a[3], dv_b[3];
        pe_vec3_scale(impulse, inv_mass_ba, dv_a);
        pe_vec3_scale(impulse, -inv_mass_bb, dv_b);
        ba->vel[0] += dv_a[0]; ba->vel[1] += dv_a[1]; ba->vel[2] += dv_a[2];
        bb->vel[0] += dv_b[0]; bb->vel[1] += dv_b[1]; bb->vel[2] += dv_b[2];
        float torque_a[3], torque_b[3];
        pe_vec3_cross(ra, impulse, torque_a);
        pe_vec3_cross(rb, impulse, torque_b);
        float dw_a[3], dw_b[3];
        pe_vec3_mul(inv_ia, torque_a, dw_a);
        pe_vec3_mul(inv_ib, torque_b, dw_b);
        ba->ang_vel[0] += dw_a[0]; ba->ang_vel[1] += dw_a[1]; ba->ang_vel[2] += dw_a[2];
        bb->ang_vel[0] -= dw_b[0]; bb->ang_vel[1] -= dw_b[1]; bb->ang_vel[2] -= dw_b[2];
    }

    float rel_ang_vel = pe_vec3_dot(ba->ang_vel, axis_world_a) - pe_vec3_dot(bb->ang_vel, axis_world_b);
    if (j->motor_max_torque > 0.0f) {
        float motor_error = j->motor_speed - rel_ang_vel;
        float motor_lambda = motor_error / (inv_ia[0] + inv_ib[0] + PE_EPS);
        motor_lambda = pe_clamp(motor_lambda, -j->motor_max_torque, j->motor_max_torque);
        ba->ang_vel[0] += motor_lambda * inv_ia[0] * axis_world_a[0];
        ba->ang_vel[1] += motor_lambda * inv_ia[1] * axis_world_a[1];
        ba->ang_vel[2] += motor_lambda * inv_ia[2] * axis_world_a[2];
        bb->ang_vel[0] -= motor_lambda * inv_ib[0] * axis_world_b[0];
        bb->ang_vel[1] -= motor_lambda * inv_ib[1] * axis_world_b[1];
        bb->ang_vel[2] -= motor_lambda * inv_ib[2] * axis_world_b[2];
    }

    if (j->limit_lower < j->limit_upper) {
        float current_angle = j->angle;
        float limit_bias = 0.0f;
        if (current_angle < j->limit_lower) limit_bias = (j->limit_lower - current_angle) * 5.0f;
        if (current_angle > j->limit_upper) limit_bias = (j->limit_upper - current_angle) * 5.0f;
        if (pe_abs(limit_bias) > PE_EPS) {
            float limit_lambda = limit_bias / (inv_ia[0] + inv_ib[0] + PE_EPS);
            ba->ang_vel[0] += limit_lambda * inv_ia[0] * axis_world_a[0];
            ba->ang_vel[1] += limit_lambda * inv_ia[1] * axis_world_a[1];
            ba->ang_vel[2] += limit_lambda * inv_ia[2] * axis_world_a[2];
            bb->ang_vel[0] -= limit_lambda * inv_ib[0] * axis_world_b[0];
            bb->ang_vel[1] -= limit_lambda * inv_ib[1] * axis_world_b[1];
            bb->ang_vel[2] -= limit_lambda * inv_ib[2] * axis_world_b[2];
        }
    }

    j->angle += rel_ang_vel * dt;
}

static void pe_solve_joint_ball(PEJoint* j, PEBody* ba, PEBody* bb, float dt) {
    float ra[3], rb[3];
    pe_vec3_sub(j->pivot_a, ba->pos, ra);
    pe_vec3_sub(j->pivot_b, bb->pos, rb);

    float error[3];
    pe_vec3_sub(ba->pos, bb->pos, error);
    error[0] += ra[0] - rb[0];
    error[1] += ra[1] - rb[1];
    error[2] += ra[2] - rb[2];

    float inv_mass_ba = ba->props.inv_mass;
    float inv_mass_bb = bb->props.inv_mass;
    float inv_ia[3], inv_ib[3];
    pe_vec3_copy(ba->props.inv_inertia, inv_ia);
    pe_vec3_copy(bb->props.inv_inertia, inv_ib);

    for (int dim = 0; dim < 3; dim++) {
        float axis[3] = { 0.0f, 0.0f, 0.0f };
        axis[dim] = 1.0f;
        float ra_cross_a[3], rb_cross_a[3];
        pe_vec3_cross(ra, axis, ra_cross_a);
        pe_vec3_cross(rb, axis, rb_cross_a);
        float inv_ia_ra_a[3], inv_ib_rb_a[3];
        pe_vec3_mul(inv_ia, ra_cross_a, inv_ia_ra_a);
        pe_vec3_mul(inv_ib, rb_cross_a, inv_ib_rb_a);
        float ja[3], jb[3];
        pe_vec3_cross(inv_ia_ra_a, ra, ja);
        pe_vec3_cross(inv_ib_rb_a, rb, jb);
        float inv_mass_dim = inv_mass_ba + inv_mass_bb + pe_vec3_dot(ja, axis) + pe_vec3_dot(jb, axis);
        if (inv_mass_dim < PE_EPS) continue;

        float err_scale = 0.2f;
        float bias = error[dim] * err_scale / (dt + PE_EPS);
        if (pe_abs(bias) > 2.0f) bias = pe_sign(bias) * 2.0f;

        float rel_v = ba->vel[dim] - bb->vel[dim] + pe_vec3_dot(ba->ang_vel, ra_cross_a) - pe_vec3_dot(bb->ang_vel, rb_cross_a);
        float lambda = (-rel_v + bias) / inv_mass_dim;

        float impulse[3];
        pe_vec3_scale(axis, lambda, impulse);
        ba->vel[0] += impulse[0] * inv_mass_ba;
        ba->vel[1] += impulse[1] * inv_mass_ba;
        ba->vel[2] += impulse[2] * inv_mass_ba;
        bb->vel[0] -= impulse[0] * inv_mass_bb;
        bb->vel[1] -= impulse[1] * inv_mass_bb;
        bb->vel[2] -= impulse[2] * inv_mass_bb;
        float torque_a[3], torque_b[3];
        pe_vec3_cross(ra, impulse, torque_a);
        pe_vec3_cross(rb, impulse, torque_b);
        float dw_a[3], dw_b[3];
        pe_vec3_mul(inv_ia, torque_a, dw_a);
        pe_vec3_mul(inv_ib, torque_b, dw_b);
        ba->ang_vel[0] += dw_a[0]; ba->ang_vel[1] += dw_a[1]; ba->ang_vel[2] += dw_a[2];
        bb->ang_vel[0] -= dw_b[0]; bb->ang_vel[1] -= dw_b[1]; bb->ang_vel[2] -= dw_b[2];
    }
}

static void pe_solve_joint_slider(PEJoint* j, PEBody* ba, PEBody* bb, float dt) {
    float axis_w[3];
    pe_vec3_rotate_by_quat(j->axis_a, ba->quat, axis_w);
    float a_len = pe_vec3_len(axis_w);
    if (a_len > PE_EPS) pe_vec3_scale(axis_w, 1.0f/a_len, axis_w);

    float ra[3], rb[3];
    pe_vec3_sub(j->pivot_a, ba->pos, ra);
    pe_vec3_sub(j->pivot_b, bb->pos, rb);

    float offset[3];
    pe_vec3_sub(bb->pos, ba->pos, offset);
    float slide_dist = pe_vec3_dot(offset, axis_w);

    float inv_mass_ba = ba->props.inv_mass;
    float inv_mass_bb = bb->props.inv_mass;
    float inv_ia[3], inv_ib[3];
    pe_vec3_copy(ba->props.inv_inertia, inv_ia);
    pe_vec3_copy(bb->props.inv_inertia, inv_ib);

    for (int dim = 0; dim < 3; dim++) {
        float axis[3] = { 0.0f, 0.0f, 0.0f };
        axis[dim] = 1.0f;
        float ra_cross_a[3], rb_cross_a[3];
        pe_vec3_cross(ra, axis, ra_cross_a);
        pe_vec3_cross(rb, axis, rb_cross_a);
        float inv_ia_ra_a[3], inv_ib_rb_a[3];
        pe_vec3_mul(inv_ia, ra_cross_a, inv_ia_ra_a);
        pe_vec3_mul(inv_ib, rb_cross_a, inv_ib_rb_a);
        float ja[3], jb[3];
        pe_vec3_cross(inv_ia_ra_a, ra, ja);
        pe_vec3_cross(inv_ib_rb_a, rb, jb);
        float inv_mass_dim = inv_mass_ba + inv_mass_bb + pe_vec3_dot(ja, axis) + pe_vec3_dot(jb, axis);
        if (inv_mass_dim < PE_EPS) continue;

        float err_scale = 0.15f;
        float bias = 0.0f;
        if (dim != 1) {
            float err = (dim == 0) ? (offset[0] - axis_w[0] * slide_dist) : (offset[2] - axis_w[2] * slide_dist);
            bias = err * err_scale / (dt + PE_EPS);
            if (pe_abs(bias) > 1.0f) bias = pe_sign(bias) * 1.0f;
        }
        if (dim == 1 && j->limit_lower < j->limit_upper) {
            if (slide_dist < j->limit_lower) bias = (j->limit_lower - slide_dist) * err_scale / (dt + PE_EPS);
            if (slide_dist > j->limit_upper) bias = (j->limit_upper - slide_dist) * err_scale / (dt + PE_EPS);
            if (pe_abs(bias) > 2.0f) bias = pe_sign(bias) * 2.0f;
        }

        float rel_v = ba->vel[dim] - bb->vel[dim] + pe_vec3_dot(ba->ang_vel, ra_cross_a) - pe_vec3_dot(bb->ang_vel, rb_cross_a);
        float lambda = (-rel_v + bias) / inv_mass_dim;

        float impulse[3];
        pe_vec3_scale(axis, lambda, impulse);
        ba->vel[0] += impulse[0] * inv_mass_ba;
        ba->vel[1] += impulse[1] * inv_mass_ba;
        ba->vel[2] += impulse[2] * inv_mass_ba;
        bb->vel[0] -= impulse[0] * inv_mass_bb;
        bb->vel[1] -= impulse[1] * inv_mass_bb;
        bb->vel[2] -= impulse[2] * inv_mass_bb;
        float torque_a[3], torque_b[3];
        pe_vec3_cross(ra, impulse, torque_a);
        pe_vec3_cross(rb, impulse, torque_b);
        float dw_a[3], dw_b[3];
        pe_vec3_mul(inv_ia, torque_a, dw_a);
        pe_vec3_mul(inv_ib, torque_b, dw_b);
        ba->ang_vel[0] += dw_a[0]; ba->ang_vel[1] += dw_a[1]; ba->ang_vel[2] += dw_a[2];
        bb->ang_vel[0] -= dw_b[0]; bb->ang_vel[1] -= dw_b[1]; bb->ang_vel[2] -= dw_b[2];
    }
}

static void pe_solve_joint_fixed(PEJoint* j, PEBody* ba, PEBody* bb, float dt) {
    pe_solve_joint_ball(j, ba, bb, dt);
}

static void pe_solve_joint_spring(PEJoint* j, PEBody* ba, PEBody* bb, float dt) {
    float offset[3];
    pe_vec3_sub(bb->pos, ba->pos, offset);
    float dist = pe_vec3_len(offset);
    if (dist < PE_EPS) return;
    float dir[3]; pe_vec3_scale(offset, 1.0f/dist, dir);
    float rest_len = pe_vec3_dist_sq(j->pivot_a, j->pivot_b);
    rest_len = pe_sqrt(rest_len);
    float stretch = dist - rest_len;

    float rel_vel_dir = (ba->vel[0] - bb->vel[0])*dir[0] + (ba->vel[1] - bb->vel[1])*dir[1] + (ba->vel[2] - bb->vel[2])*dir[2];
    float force_mag = -j->stiffness * stretch - j->damping * rel_vel_dir;
    float max_f = j->max_force;
    if (force_mag > max_f) force_mag = max_f;
    if (force_mag < -max_f) force_mag = -max_f;

    float force[3];
    pe_vec3_scale(dir, force_mag, force);
    ba->vel[0] += force[0] * ba->props.inv_mass * dt;
    ba->vel[1] += force[1] * ba->props.inv_mass * dt;
    ba->vel[2] += force[2] * ba->props.inv_mass * dt;
    bb->vel[0] -= force[0] * bb->props.inv_mass * dt;
    bb->vel[1] -= force[1] * bb->props.inv_mass * dt;
    bb->vel[2] -= force[2] * bb->props.inv_mass * dt;
}

/* ============================================================================
 * 主步进函数
 * ============================================================================ */

int pe_step(PEWorld* world, float dt) {
    if (!world || dt <= 0.0f) return -1;

    world->dt = dt;

    /* 根据积分器类型调度 */
    if (world->integrator_type == PE_INTEGRATOR_RK4) {
        return pe_world_step_rk4(world, dt);
    }

    /* 默认：半隐式Euler NGS步进 */
    float sub_dt = dt / world->substeps;

    for (int s = 0; s < world->substeps; s++) {
        for (int i = 0; i < world->body_count; i++) {
            PEBody* b = &world->bodies[i];
            if (!b->props.active || b->props.type == PE_BODY_STATIC) continue;
            if (b->props.type == PE_BODY_KINEMATIC) continue;

            b->force_acc[1] += world->gravity[1] * (b->props.type == PE_BODY_DYNAMIC ? b->props.mass : 0.0f);
            pe_integrate_forces(b, sub_dt);
        }

        int contact_count = 0;
        for (int i = 0; i < world->body_count && contact_count < PE_MAX_CONTACTS; i++) {
            PEBody* ba = &world->bodies[i];
            if (!ba->props.active) continue;
            float aabb_min_a[3], aabb_max_a[3];
            pe_collision_aabb(ba, aabb_min_a, aabb_max_a);
            for (int j = i + 1; j < world->body_count && contact_count < PE_MAX_CONTACTS; j++) {
                PEBody* bb = &world->bodies[j];
                if (!bb->props.active) continue;
                if (ba->props.type == PE_BODY_STATIC && bb->props.type == PE_BODY_STATIC) continue;
                float aabb_min_b[3], aabb_max_b[3];
                pe_collision_aabb(bb, aabb_min_b, aabb_max_b);
                if (!pe_aabb_overlap(aabb_min_a, aabb_max_a, aabb_min_b, aabb_max_b)) continue;

                float normal[3], point[3], penetration;
                if (pe_gjk_penetration(ba, bb, normal, point, &penetration)) {
                    PEContact* ct = &world->contacts[contact_count];
                    ct->body_a = i;
                    ct->body_b = j;
                    pe_vec3_copy(normal, ct->normal);
                    pe_vec3_copy(point, ct->point);
                    ct->penetration = penetration;
                    ct->friction = 0.5f;
                    ct->restitution = 0.2f;
                    ct->impulse_n = 0.0f;
                    ct->impulse_t[0] = 0.0f;
                    ct->impulse_t[1] = 0.0f;
                    ct->active = 1;
                    contact_count++;
                }
            }
        }
        world->contact_count = contact_count;

        for (int iter = 0; iter < world->solver_iters; iter++) {
            for (int c = 0; c < world->contact_count; c++) {
                PEContact* ct = &world->contacts[c];
                PEBody* ba = &world->bodies[ct->body_a];
                PEBody* bb = &world->bodies[ct->body_b];
                if (!ba->props.active || !bb->props.active) continue;
                pe_solve_contact(ct, ba, bb, sub_dt);
            }
        }

        for (int j = 0; j < world->joint_count; j++) {
            PEJoint* jnt = &world->joints[j];
            if (!jnt->active) continue;
            PEBody* ba = &world->bodies[jnt->body_a];
            PEBody* bb = &world->bodies[jnt->body_b];
            if (!ba->props.active || !bb->props.active) continue;
            switch (jnt->type) {
                case PE_JOINT_HINGE: pe_solve_joint_hinge(jnt, ba, bb, sub_dt); break;
                case PE_JOINT_BALL: pe_solve_joint_ball(jnt, ba, bb, sub_dt); break;
                case PE_JOINT_SLIDER: pe_solve_joint_slider(jnt, ba, bb, sub_dt); break;
                case PE_JOINT_FIXED: pe_solve_joint_fixed(jnt, ba, bb, sub_dt); break;
                case PE_JOINT_SPRING: pe_solve_joint_spring(jnt, ba, bb, sub_dt); break;
            }
        }

        for (int i = 0; i < world->body_count; i++) {
            PEBody* b = &world->bodies[i];
            if (!b->props.active || b->props.type != PE_BODY_DYNAMIC) continue;
            pe_integrate_velocity(b, sub_dt);
        }
    }
    return 0;
}
