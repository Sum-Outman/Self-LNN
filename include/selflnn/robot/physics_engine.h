#ifndef SELFLNN_PHYSICS_ENGINE_H
#define SELFLNN_PHYSICS_ENGINE_H

#include "selflnn/robot/kinematics.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PE_MAX_BODIES 256
#define PE_MAX_CONTACTS 128
#define PE_MAX_JOINTS 128
#define PE_SOLVER_ITERATIONS 15
#define PE_BAUMGARTE 0.15f
#define PE_GRAVITY -9.81f

typedef enum {
    PE_BODY_DYNAMIC = 0,
    PE_BODY_STATIC = 1,
    PE_BODY_KINEMATIC = 2
} PEBodyType;

typedef struct {
    float mass;
    float inv_mass;
    float inertia[3];
    float inv_inertia[3];
    PEBodyType type;
    int active;
} PEBodyProps;

typedef struct {
    float pos[3];
    float vel[3];
    float ang_vel[3];
    float quat[4];
    float force_acc[3];
    float torque_acc[3];
    PEBodyProps props;
    int body_id;
    float bounding_radius; /**< 刚体包围球半径（用于AABB广阶段碰撞检测和惯性张量计算）*/
} PEBody;

typedef struct {
    int body_a;
    int body_b;
    float normal[3];
    float point[3];
    float penetration;
    float friction;
    float restitution;
    float impulse_n;
    float impulse_t[2];
    int active;
} PEContact;

typedef enum {
    PE_JOINT_HINGE = 0,
    PE_JOINT_BALL = 1,
    PE_JOINT_SLIDER = 2,
    PE_JOINT_FIXED = 3,
    PE_JOINT_SPRING = 4
} PEJointType;

typedef struct {
    PEJointType type;
    int body_a;
    int body_b;
    float pivot_a[3];
    float pivot_b[3];
    float axis_a[3];
    float axis_b[3];
    float limit_lower;
    float limit_upper;
    float max_force;
    float stiffness;
    float damping;
    float motor_speed;
    float motor_max_torque;
    float angle;
    int active;
} PEJoint;

typedef struct {
    PEBody bodies[PE_MAX_BODIES];
    int body_count;
    PEContact contacts[PE_MAX_CONTACTS];
    int contact_count;
    PEJoint joints[PE_MAX_JOINTS];
    int joint_count;
    float gravity[3];
    float dt;
    int substeps;
    int solver_iters;
    int enable_ccd;
    int integrator_type; /**< 0=半隐式Euler, 1=RK4(4阶龙格-库塔) */
    int enable_adaptive_substep; /**< 自适应子步长（仅RK4模式） */
} PEWorld;

/* 积分器类型常量 */
#define PE_INTEGRATOR_EULER      0  /**< 半隐式Euler（默认，快速） */
#define PE_INTEGRATOR_RK4        1  /**< 4阶龙格-库塔（高精度） */

int pe_world_init(PEWorld* world);
int pe_body_add(PEWorld* world, const float* pos, const float* quat, float mass, PEBodyType type);
int pe_body_remove(PEWorld* world, int body_id);
int pe_body_apply_force(PEWorld* world, int body_id, const float* force);
int pe_body_apply_torque(PEWorld* world, int body_id, const float* torque);
int pe_body_get_state(const PEWorld* world, int body_id, float* pos, float* quat, float* vel, float* ang_vel);
int pe_body_set_kinematic(PEWorld* world, int body_id, const float* pos, const float* quat, const float* vel, const float* ang_vel);
int pe_joint_add(PEWorld* world, PEJointType type, int body_a, int body_b, const float* pivot_a, const float* pivot_b, const float* axis, float limit_lower, float limit_upper);
int pe_joint_set_motor(PEWorld* world, int joint_id, float speed, float max_torque);
int pe_joint_get_angle(const PEWorld* world, int joint_id, float* angle);
int pe_step(PEWorld* world, float dt);
void pe_collision_aabb(const PEBody* body, float* aabb_min, float* aabb_max);

#ifdef __cplusplus
}
#endif

#endif
