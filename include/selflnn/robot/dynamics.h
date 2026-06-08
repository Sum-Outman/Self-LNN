#ifndef SELFLNN_ROBOT_DYNAMICS_H
#define SELFLNN_ROBOT_DYNAMICS_H

#include "selflnn/robot/kinematics.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DYNAMICS_MAX_JOINTS 32
#define DYNAMICS_MAX_CONTACTS 64
#define DYNAMICS_MATRIX_SIZE (DYNAMICS_MAX_JOINTS * DYNAMICS_MAX_JOINTS)
#define DYNAMICS_IK_MIN_DAMPING 1e-8f

typedef enum {
    CONTACT_MODEL_PENALTY = 0,
    CONTACT_MODEL_IMPULSE = 1,
    CONTACT_MODEL_LCP = 2
} ContactModelType;

typedef enum {
    FRICTION_CONE_PYRAMID = 0,
    FRICTION_CONE_CONE = 1
} FrictionConeType;

typedef struct {
    int num_joints;
    int num_links;
    float link_masses[DYNAMICS_MAX_JOINTS];
    float link_com[DYNAMICS_MAX_JOINTS * 3];
    float link_inertia[DYNAMICS_MAX_JOINTS * 9];
    float joint_frictions[DYNAMICS_MAX_JOINTS];
    float joint_dampings[DYNAMICS_MAX_JOINTS];
    float joint_axes[DYNAMICS_MAX_JOINTS * 3];
    int joint_types[DYNAMICS_MAX_JOINTS];
    int parent_indices[DYNAMICS_MAX_JOINTS];
    float gravity[3];
    float contact_stiffness;
    float contact_damping;
    float friction_coefficient;
    float restitution;
    ContactModelType contact_model;
    FrictionConeType friction_model;
    int enable_joint_friction;
    int enable_joint_damping;
    float max_joint_torques[DYNAMICS_MAX_JOINTS];
} RobotDynamicsConfig;

typedef struct {
    float position[3];
    float normal[3];
    float penetration_depth;
    float friction_coeff;
    int body_a;
    int body_b;
    int contact_id;
} ContactPoint;

typedef struct {
    float normal_force;
    float friction_force[2];
    float total_force[3];
    int is_sliding;
} ContactForce;

typedef struct {
    float joint_positions[DYNAMICS_MAX_JOINTS];
    float joint_velocities[DYNAMICS_MAX_JOINTS];
    float joint_accelerations[DYNAMICS_MAX_JOINTS];
    float joint_torques[DYNAMICS_MAX_JOINTS];
    int num_joints;
    float total_energy;
    float kinetic_energy;
    float potential_energy;
} DynamicsState;

typedef struct DynamicsModel DynamicsModel;

RobotDynamicsConfig dynamics_config_default(void);

DynamicsModel* dynamics_model_create(const KinematicModel* kin_model, const RobotDynamicsConfig* config);
void dynamics_model_destroy(DynamicsModel* model);

int dynamics_rnea(const DynamicsModel* model,
                  const float* q, const float* qd, const float* qdd,
                  float* torques);

int dynamics_mass_matrix(const DynamicsModel* model, const float* q, float* mass_matrix);

int dynamics_coriolis(const DynamicsModel* model,
                      const float* q, const float* qd,
                      float* coriolis);

int dynamics_gravity(const DynamicsModel* model,
                     const float* q, float* gravity_terms);

int dynamics_forward_dynamics(const DynamicsModel* model,
                              const float* q, const float* qd,
                              const float* torques,
                              float* qdd);

int dynamics_forward_dynamics_external(const DynamicsModel* model,
                                       const float* q, const float* qd,
                                       const float* torques,
                                       const float* external_forces,
                                       float* qdd);

int dynamics_compute_contact_forces(const DynamicsModel* model,
                                     const float* q, const float* qd,
                                     const ContactPoint* contacts,
                                     int contact_count,
                                     ContactForce* forces);

float dynamics_kinetic_energy(const DynamicsModel* model,
                               const float* q, const float* qd);

float dynamics_potential_energy(const DynamicsModel* model,
                                 const float* q);

float dynamics_total_energy(const DynamicsModel* model,
                             const float* q, const float* qd);

int dynamics_apply_torque_limits(const DynamicsModel* model,
                                  float* torques, int num_joints);

int dynamics_step(DynamicsModel* model,
                  float* q, float* qd,
                  const float* torques, float dt);

int dynamics_get_state(const DynamicsModel* model, DynamicsState* state);

int robot_dynamics_get_config(const DynamicsModel* model, RobotDynamicsConfig* config);
int robot_dynamics_set_config(DynamicsModel* model, const RobotDynamicsConfig* config);

int dynamics_solve_linear_system(int n, const float* A, const float* b, float* x);

int dynamics_check_energy_conservation(const DynamicsModel* model,
                                        const float* q_before, const float* qd_before,
                                        const float* q_after, const float* qd_after,
                                        const float* torques, float dt,
                                        float* energy_error);

/* ================================================================
 * K-031: GJK碰撞检测 + EPA穿透深度
 * ================================================================ */

/**
 * @brief K-031: GJK碰撞检测 — 判断两个凸多面体是否相交
 * @param shape_a 形状A顶点数组 [num_a*3]
 * @param num_a A顶点数
 * @param shape_b 形状B顶点数组 [num_b*3]
 * @param num_b B顶点数
 * @return 1=碰撞, 0=无碰撞, -1=错误
 */
int dynamics_gjk_collision(const float* shape_a, int num_a,
                            const float* shape_b, int num_b);

/**
 * @brief K-031: EPA穿透深度计算
 * @param shape_a A顶点数组
 * @param num_a A顶点数
 * @param shape_b B顶点数组
 * @param num_b B顶点数
 * @param penetration_depth [输出] 穿透深度
 * @param penetration_normal [输出] 穿透法向量(3分量)
 * @param contact_point [输出] 接触点(3分量,可NULL)
 * @return 1=有穿透, 0=无穿透, -1=错误
 */
int dynamics_epa_penetration(const float* shape_a, int num_a,
                              const float* shape_b, int num_b,
                              float* penetration_depth,
                              float* penetration_normal,
                              float* contact_point);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_ROBOT_DYNAMICS_H */
