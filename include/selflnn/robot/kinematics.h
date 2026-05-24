#ifndef SELFLNN_KINEMATICS_H
#define SELFLNN_KINEMATICS_H

#include "selflnn/robot/robot.h"
#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KINEMATICS_MAX_JOINTS 32
#define KINEMATICS_MAX_LINKS 32
#define KINEMATICS_MAX_URDF_CHILDREN 16
#define KINEMATICS_MAX_COLLISION_SHAPES 64
#define KINEMATICS_GJK_MAX_ITER 256
#define KINEMATICS_IK_MAX_ITER 200
#define KINEMATICS_IK_DAMPING 0.01f
#define KINEMATICS_IK_TOLERANCE 1e-6f

typedef enum {
    COLLISION_SHAPE_SPHERE = 0,
    COLLISION_SHAPE_BOX = 1,
    COLLISION_SHAPE_CAPSULE = 2,
    COLLISION_SHAPE_CYLINDER = 3,
    COLLISION_SHAPE_MESH = 4,
    COLLISION_SHAPE_PLANE = 5
} CollisionShapeType;

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    Vec3 center;
    float radius;
} CollisionSphere;

typedef struct {
    Vec3 half_extents;
    Vec3 center;
} CollisionBox;

typedef struct {
    Vec3 center;
    float radius;
    float height;
} CollisionCapsule;

typedef struct {
    Vec3 center;
    float radius;
    float height;
} CollisionCylinder;

typedef struct {
    Vec3* vertices;
    int vertex_count;
    int* indices;
    int index_count;
    Vec3 center;
    float bounding_radius;
} CollisionMesh;

typedef struct {
    Vec3 normal;
    float d;
} CollisionPlane;

typedef struct {
    CollisionShapeType type;
    union {
        CollisionSphere sphere;
        CollisionBox box;
        CollisionCapsule capsule;
        CollisionCylinder cylinder;
        CollisionMesh mesh;
        CollisionPlane plane;
    } data;
    float local_transform[7];
    void* user_data;
} CollisionShape;

typedef struct {
    Vec3 point;
    Vec3 normal;
    float penetration_depth;
    int shape_a_id;
    int shape_b_id;
} CollisionContact;

typedef struct {
    CollisionContact* contacts;
    int contact_count;
    int contact_capacity;
} CollisionResult;

typedef struct {
    Vec3 support[4];
    Vec3 search_dir;
    int num_vertices;
} Simplex;

typedef enum {
    JOINT_TYPE_REVOLUTE = 0,
    JOINT_TYPE_PRISMATIC = 1,
    JOINT_TYPE_FIXED = 2,
    JOINT_TYPE_CONTINUOUS = 3,
    JOINT_TYPE_FLOATING = 4,
    JOINT_TYPE_PLANAR = 5,
    JOINT_TYPE_UNKNOWN = 6
} KinJointType;

typedef struct {
    float a;    /* 连杆长度 (mm) */
    float alpha; /* 连杆扭转 (rad) */
    float d;    /* 连杆偏距 (mm) */
    float theta; /* 关节角 (rad) */
} DHParameter;

typedef struct {
    char name[64];
    int joint_id;
    KinJointType joint_type;
    DHParameter dh;
    float joint_limit_lower;
    float joint_limit_upper;
    float joint_max_velocity;
    float joint_max_torque;
    float joint_friction;
    float joint_damping;
    float parent_to_joint_pos[3];
    float joint_to_child_pos[3];
    float joint_axis[3];
    float link_mass;
    Vec3 link_com;
    float link_inertia[9];
    int parent_index;
    int child_indices[KINEMATICS_MAX_URDF_CHILDREN];
    int child_count;
    int has_collision;
    CollisionShape collision_shape;
    float color[4];
} KinJoint;

typedef struct {
    char name[64];
    KinJoint joints[KINEMATICS_MAX_JOINTS];
    int joint_count;
    int num_dof;
    Vec3 base_position;
    float base_orientation[4];
    float end_effector_offset[3];
    int end_effector_joint;
    float total_mass;
    char source_urdf[2048];
} KinematicModel;

/* S-001修复: 机器人运动学实例状态结构体
 * 每个RobotKinematics实例独立持有速度计算所需的前一帧状态，
 * 消除TLS线程局部存储跨不同模型/机器人实例共享的问题。
 * 每个机器人/模型创建独立的RobotKinematics实例，互不干扰。 */
typedef struct {
    clock_t prev_tick;                 /* 上一帧的时间戳(clock ticks) */
    float prev_pos_x;                  /* 上一帧末端位置X */
    float prev_pos_y;                  /* 上一帧末端位置Y */
    float prev_pos_z;                  /* 上一帧末端位置Z */
    float prev_quat_x;                 /* 上一帧末端姿态四元数X */
    float prev_quat_y;                 /* 上一帧末端姿态四元数Y */
    float prev_quat_z;                 /* 上一帧末端姿态四元数Z */
    float prev_quat_w;                 /* 上一帧末端姿态四元数W */
    int is_initialized;               /* 是否已初始化（首次调用标记） */
    int robot_instance_id;            /* 绑定的机器人实例ID，-1表示未绑定 */
} RobotKinematics;

typedef struct {
    Vec3 position;
    float orientation[4];
    Vec3 linear_velocity;
    Vec3 angular_velocity;
} EndEffectorState;

typedef int (*CollisionCallback)(int shape_a, int shape_b,
                                  const CollisionContact* contact, void* user_data);

void k_vec3_add(Vec3* out, const Vec3* a, const Vec3* b);
void k_vec3_sub(Vec3* out, const Vec3* a, const Vec3* b);
void k_vec3_lerp(Vec3* out, const Vec3* a, const Vec3* b, float t);
float k_vec3_dot(const Vec3* a, const Vec3* b);
void k_vec3_cross(Vec3* out, const Vec3* a, const Vec3* b);
void k_vec3_scale(Vec3* out, const Vec3* a, float s);
float k_vec3_length(const Vec3* a);
void k_vec3_normalize(Vec3* out, const Vec3* a);
float k_vec3_distance(const Vec3* a, const Vec3* b);
void k_vec3_transform_quat(Vec3* out, const Vec3* v, const float* q);

/* F-032修复: 移除与math_utils.h中static inline版本冲突的四元数函数声明
 * 四元数函数(quat_multiply/quat_conjugate/quat_to_matrix/quat_from_axis_angle)
 * 现在统一使用 include/selflnn/utils/math_utils.h 中的 static inline 版本
 * src/robot/kinematics.c 已显式包含 math_utils.h */

void collision_result_init(CollisionResult* result, int initial_capacity);
void collision_result_free(CollisionResult* result);
int collision_result_add(CollisionResult* result, const CollisionContact* contact);
void collision_shape_init_sphere(CollisionShape* shape, const Vec3* center, float radius);
void collision_shape_init_box(CollisionShape* shape, const Vec3* center, const Vec3* half_extents);
void collision_shape_init_capsule(CollisionShape* shape, const Vec3* center, float radius, float height);
void collision_shape_init_mesh(CollisionShape* shape, Vec3* vertices, int vertex_count,
                                int* indices, int index_count);
Vec3 collision_shape_support(const CollisionShape* shape, const Vec3* dir);
int gjk_intersection(const CollisionShape* shape_a, const CollisionShape* shape_b,
                      CollisionContact* contact);
int collision_detect(const CollisionShape* shapes, int shape_count,
                      CollisionResult* result, CollisionCallback callback, void* user_data);
int collision_detect_robot_self(const KinematicModel* model, const float* joint_angles,
                                 CollisionResult* result);

void kinematic_model_init(KinematicModel* model);
int kinematic_model_add_joint(KinematicModel* model, int parent_index,
                                const char* name, KinJointType type,
                                const DHParameter* dh,
                                float limit_lower, float limit_upper);
void kinematic_model_set_end_effector(KinematicModel* model, int joint_index,
                                       const float* offset);
int kinematic_model_set_base_pose(KinematicModel* model, const Vec3* position,
                                   const float* orientation);

int forward_kinematics(const KinematicModel* model, const float* joint_angles,
                        EndEffectorState* result);
int forward_kinematics_full(const KinematicModel* model, const float* joint_angles,
                             Vec3* link_positions, float* link_orientations);

/* S-001修复: 带实例状态的正运动学速度计算
 * RobotKinematics实例由调用方创建和管理，每个机器人独立持有一个。
 * 通过robot_kinematics_init()初始化，robot_kinematics_reset()重置。 */
void robot_kinematics_init(RobotKinematics* state);
void robot_kinematics_reset(RobotKinematics* state);
int forward_kinematics_stateful(const KinematicModel* model, const float* joint_angles,
                                 RobotKinematics* state, EndEffectorState* result);
int compute_jacobian(const KinematicModel* model, const float* joint_angles,
                      float* jacobian, int jacobian_rows);
int inverse_kinematics_ccd(const KinematicModel* model, const Vec3* target_pos,
                            const float* target_orient, float* joint_angles,
                            int max_iter, float tolerance);
int inverse_kinematics_dls(const KinematicModel* model, const Vec3* target_pos,
                            const float* target_orient, float* joint_angles,
                            float damping, int max_iter, float tolerance);
int inverse_kinematics_transpose(const KinematicModel* model, const Vec3* target_pos,
                                  const float* target_orient, float* joint_angles,
                                  float step_size, int max_iter, float tolerance);

typedef struct {
    KinJointType type;
    float lower;
    float upper;
    float max_velocity;
} JointConstraint;

typedef struct {
    JointConstraint constraints[KINEMATICS_MAX_JOINTS];
    int constraint_count;
} JointConstraintSolver;

void constraint_solver_init(JointConstraintSolver* solver);
int constraint_solver_add(JointConstraintSolver* solver, KinJointType type,
                           float lower, float upper, float max_velocity);
void constraint_solver_apply(const JointConstraintSolver* solver,
                              float* joint_angles, int joint_count);

int urdf_parse_string(KinematicModel* model, const char* urdf_xml);
int urdf_parse_file(KinematicModel* model, const char* filepath);
const char* kinematics_error_string(int error_code);

/* ============================================================================
 * 运动学奇异性处理
 * ============================================================================ */

#define KINEMATICS_MAX_DOF 32

/* 可操作度椭球度量 - 返回 Yoshikawa 可操作度 w = sqrt(det(J * J^T)) */
float kinematics_manipulability(const float* jacobian, int dof);

/* 条件数 - 返回 σ_max / σ_min，越大越接近奇异 */
float kinematics_condition_number(const float* jacobian, int dof);

/* 最小奇异值 - 返回最小奇异值，接近0表示奇异 */
float kinematics_min_singular_value(const float* jacobian, int dof);

/* 奇异值分解（6×n 雅可比的经济 SVD） */
int kinematics_svd(const float* jacobian, int rows, int cols,
                    float* U, float* S, float* Vt);

/* 奇异鲁棒逆（SR-Inverse/DLS）：J⁺ = J^T (J J^T + λ²I)⁻¹，λ自动调节 */
int kinematics_sr_inverse(const float* jacobian, int rows, int cols,
                           float* jacobian_pinv,
                           float* damping_out, float manipulability_threshold);

/* 奇异检测 - 返回奇异状态：0=正常, 1=接近奇异, 2=奇异 */
int kinematics_singularity_detect(const float* jacobian, int dof,
                                   float threshold);

/* ============================================================================
 * 冗余机器人运动学
 * ============================================================================ */

/* 零空间投影矩阵：N = I - J⁺J */
int kinematics_null_space_project(const float* jacobian, int dof,
                                   const float* gradient,
                                   float* null_space_qdot);

/* 梯度投影法避关节限位 - 关节空间二次优化梯度 */
int kinematics_joint_limit_gradient(const KinematicModel* model,
                                     const float* joint_angles,
                                     float* gradient);

/* 梯度投影法避奇异 - 最大化可操作度的梯度 */
int kinematics_singularity_avoidance_gradient(const float* jacobian,
                                               int dof, float* gradient);

/* 任务优先级 IK（主任务 + 零空间副任务） */
int inverse_kinematics_priority(const KinematicModel* model,
                                 const Vec3* primary_target,
                                 const float* primary_orient,
                                 const Vec3* secondary_target,
                                 const float* secondary_orient,
                                 float* joint_angles,
                                 float damping, int max_iter, float tolerance);

/* 梯度投影 IK（主任务 + 加权零空间优化） */
int inverse_kinematics_gradient_projection(const KinematicModel* model,
                                            const Vec3* target_pos,
                                            const float* target_orient,
                                            float* joint_angles,
                                            float joint_limit_weight,
                                            float singularity_weight,
                                            float step_size,
                                            int max_iter, float tolerance);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_KINEMATICS_H */
