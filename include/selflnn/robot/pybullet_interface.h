#ifndef SELFLNN_PYBULLET_INTERFACE_H
#define SELFLNN_PYBULLET_INTERFACE_H

#include "selflnn/core/common.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PB_MAX_NAME_LEN 256
#define PB_MAX_JOINTS 64
#define PB_MAX_LINKS 64
#define PB_MAX_SENSORS 32
#define PB_BUF_SIZE 65536
#define PB_CMD_BUF_SIZE 8192
#define PB_RESP_BUF_SIZE 262144
#define PB_PIPE_PATH_LEN 512
#define PB_MAX_ROBOTS 64
#define PB_IK_MAX_ITERATIONS 100
#define PB_CONTACT_MAX_POINTS 64
#define PB_MAX_LOG_ENTRIES 10000
#define PB_MAX_DEBUG_ITEMS 1024
#define PB_LNN_INPUT_DIM 32
#define PB_LNN_HIDDEN_DIM 64
#define PB_LNN_OUTPUT_DIM 16
#define PB_MAX_PRIMITIVES 128
#define PB_MAX_CONSTRAINTS 64

typedef enum {
    PB_STATE_DISCONNECTED = 0,
    PB_STATE_CONNECTING,
    PB_STATE_CONNECTED,
    PB_STATE_RUNNING,
    PB_STATE_ERROR
} PBConnectionState;

typedef enum {
    PB_GUI_NONE = 0,
    PB_GUI_GUI = 1,
    PB_GUI_SHARED_MEMORY = 2,
    PB_GUI_DIRECT = 3,
    PB_GUI_UDP = 4
} PBGuiMode;

typedef enum {
    PB_CONSTRAINT_POINT2POINT = 0,
    PB_CONSTRAINT_FIXED,
    PB_CONSTRAINT_HINGE,
    PB_CONSTRAINT_SLIDER,
    PB_CONSTRAINT_SPRING,
    PB_CONSTRAINT_GEAR
} PBConstraintType;

typedef enum {
    PB_SHAPE_BOX = 0,
    PB_SHAPE_SPHERE,
    PB_SHAPE_CYLINDER,
    PB_SHAPE_CAPSULE,
    PB_SHAPE_MESH,
    PB_SHAPE_PLANE
} PBCollisionShape;

typedef enum {
    PB_CONTROL_POSITION = 0,
    PB_CONTROL_VELOCITY,
    PB_CONTROL_TORQUE,
    PB_CONTROL_PD,
    PB_CONTROL_PID
} PBControlMode;

typedef enum {
    PB_IK_MODE_POSITION = 0,
    PB_IK_MODE_ORIENTATION,
    PB_IK_MODE_POSE,
    PB_IK_MODE_CARTESIAN
} PBIKMode;

typedef struct {
    int use_gui;
    PBGuiMode gui_mode;
    int connect_timeout_ms;
    int enable_visualization;
    float timestep;
    float gravity_z;
    int num_solver_iterations;
    int enable_real_time_simulation;
    float real_time_factor;
    int use_external_pybullet;      /**< 1=连接外部真实PyBullet, 0=使用内部纯C仿真器 */
} PBConfig;

#define PB_CONFIG_DEFAULT { 0, PB_GUI_DIRECT, 10000, 1, 0.001f, -9.81f, 50, 0, 1.0f, 0 }

typedef struct {
    int body_unique_id;
    int num_joints;
    int num_links;
    float base_position[3];
    float base_orientation[4];
    char urdf_path[PB_MAX_NAME_LEN];
} PBRobotInfo;

typedef struct {
    int body_id;
    int joint_index;
    float joint_position;
    float joint_velocity;
    float joint_reaction_force[3];
    float joint_reaction_torque[3];
    float motor_torque;
    float joint_lower_limit;
    float joint_upper_limit;
    float max_force;
    float max_velocity;
    char joint_name[64];
    char link_name[64];
} PBJointState;

typedef struct {
    float position[3];
    float orientation[4];
    float linear_velocity[3];
    float angular_velocity[3];
} PBLinkState;

typedef struct {
    int width;
    int height;
    unsigned char* rgb_data;
    size_t rgb_size;
    float* depth_data;
    size_t depth_size;
    float* segmentation_mask;
    size_t seg_size;
    float view_matrix[16];
    float projection_matrix[16];
} PBCameraImage;

typedef struct {
    float ray_from[3];
    float ray_to[3];
    int body_unique_id;
    int link_index;
    float hit_fraction;
    float hit_position[3];
    float hit_normal[3];
} PBRayHit;

typedef struct {
    PBConnectionState state;
    int is_connected;
    int body_count;
    float simulation_time;
    int step_count;
    float real_time_factor_actual;
    char last_error[256];
} PBStatus;

typedef struct {
    int body_a;
    int link_a;
    int body_b;
    int link_b;
    float position_on_a[3];
    float position_on_b[3];
    float normal_on_b[3];
    float distance;
    float normal_force;
    float lateral_friction;
} PBContactPoint;

typedef struct {
    int num_contacts;
    PBContactPoint points[PB_CONTACT_MAX_POINTS];
} PBContactInfo;

typedef struct {
    int body_id;
    int link_index;
    float collision_radius;
    float collision_margin;
    int has_collision;
    float contact_normal[3];
    float penetration_depth;
    int num_overlapping_bodies;
    int overlapping_bodies[32];
} PBCollisionInfo;

typedef struct {
    int body_id;
    int gripper_link_index;
    float open_position;
    float closed_position;
    float current_position;
    float grip_force;
    int is_closed;
    int has_object;
    int grasped_body_id;
    float grasp_strength;
} PBGripperControl;

typedef struct {
    int body_id;
    int num_joints;
    int joint_indices[PB_MAX_JOINTS];
    float target_positions[PB_MAX_JOINTS];
    float target_orientations[4];
    float solution[PB_MAX_JOINTS];
    float solution_confidence;
    int iterations_used;
    int success;
} PBIKResult;

typedef struct {
    int num_robots;
    int num_steps;
    int* body_ids;
    float** joint_positions;
    float** joint_velocities;
    float* timestamps;
    int entry_count;
    int capacity;
} PBBatchStep;

typedef struct {
    int constraint_id;
    PBConstraintType type;
    int body_a;
    int body_b;
    int link_a;
    int link_b;
    float pivot_in_a[3];
    float pivot_in_b[3];
    float axis_in_a[3];
    float axis_in_b[3];
    float lower_limit;
    float upper_limit;
    float max_force;
    int is_active;
} PBConstraintInfo;

typedef struct {
    int debug_id;
    int is_active;
    float color_r;
    float color_g;
    float color_b;
    float line_width;
    float life_time;
} PBDebugItem;

typedef struct {
    float position[3];
    float half_extents[3];
    float color_r;
    float color_g;
    float color_b;
    float mass;
    PBCollisionShape shape;
} PBPrimitiveInfo;

typedef struct {
    int enabled;
    int num_entries;
    float log_interval;
    int max_entries;
    float* timestamps;
    float** joint_positions;
    float** joint_velocities;
    float* end_effector_positions;
    int* entry_count_ptr;
} PBStateLogger;

int pb_init(PBConfig* config);
int pb_connect(void);
int pb_disconnect(void);

/**
 * @brief 检测外部PyBullet/Gazebo是否可用（网络连接方式）
 * @return 1可用，0不可用（使用内部纯C仿真器）
 */
int pybullet_is_external_available(void);

/**
 * @brief 检测PyBullet仿真系统是否已初始化且可用
 * @return 1可用，0不可用
 */
int pybullet_is_simulation_available(void);

int pb_load_urdf(const char* urdf_path, float base_x, float base_y, float base_z,
                 int use_fixed_base, PBRobotInfo* out_info);
int pb_step_simulation(void);
int pb_step_simulation_n(int num_steps);
int pb_reset_simulation(void);
int pb_set_joint_position(int body_id, int joint_index, float target_position, float force);
int pb_set_joint_velocity(int body_id, int joint_index, float target_velocity);
int pb_set_joint_torque(int body_id, int joint_index, float torque);
int pb_get_joint_states(int body_id, PBJointState* states, int max_count, int* count);
int pb_get_link_state(int body_id, int link_index, PBLinkState* state);
int pb_get_base_position(int body_id, float* position, float* orientation);
int pb_reset_base_position(int body_id, float x, float y, float z, float qx, float qy, float qz, float qw);
int pb_get_camera_image(int width, int height, const float* view_matrix, const float* proj_matrix,
                        PBCameraImage* image);
int pb_ray_cast(const float* ray_from, const float* ray_to, PBRayHit* hit);
int pb_remove_body(int body_id);
int pb_get_num_bodies(int* count);
int pb_get_body_info(int body_id, PBRobotInfo* info);
int pb_set_gravity(float gravity_z);
int pb_set_real_time_simulation(int enable);
int pb_get_status(PBStatus* status);
void pb_ray_hit_free(PBRayHit* hit);
void pb_camera_image_free(PBCameraImage* image);
void pb_shutdown(void);
const char* pb_state_str(PBConnectionState state);

int pb_get_contacts(int body_id, PBContactInfo* info);
int pb_get_all_contacts(PBContactInfo* infos, int max_count, int* count);
int pb_check_collision(int body_a, int body_b, PBCollisionInfo* info);
int pb_get_collision_data(int body_id, PBCollisionInfo* info);
int pb_gripper_init(int body_id, int gripper_link, float open_pos, float closed_pos, float force);
int pb_gripper_open(PBGripperControl* gripper);
int pb_gripper_close(PBGripperControl* gripper);
int pb_gripper_set_force(PBGripperControl* gripper, float force);
int pb_gripper_get_state(PBGripperControl* gripper);
int pb_compute_ik(int body_id, int end_effector_link, const float* target_pos,
                  const float* target_orn, PBIKResult* result, PBIKMode mode);
int pb_compute_ik_chain(int body_id, const int* link_chain, int chain_len,
                        const float* target_positions, PBIKResult* results);
int pb_batch_step(int num_steps, float* out_states);
int pb_batch_set_joints(const int* body_ids, const int* joint_indices,
                        const float* positions, int count);
int pb_batch_get_states(const int* body_ids, int num_bodies,
                        PBJointState* out_states, int max_joints, int* out_count);
int pb_set_link_friction(int body_id, int link_index, float friction);
int pb_set_link_restitution(int body_id, int link_index, float restitution);
int pb_set_link_damping(int body_id, int link_index, float linear_damping, float angular_damping);
int pb_set_link_linear_damping(int body_id, int link_index, float damping);
int pb_set_joint_limit(int body_id, int joint_index, float lower, float upper);
int pb_set_joint_max_force(int body_id, int joint_index, float max_force);
int pb_set_joint_max_velocity(int body_id, int joint_index, float max_vel);
int pb_load_scene(const char* scene_file);
int pb_clear_scene(void);
int pb_save_scene(const char* scene_file);
int pb_add_primitive(PBPrimitiveInfo* info, int* out_body_id);
int pb_remove_primitive(int body_id);
int pb_create_constraint(PBConstraintInfo* info, int* out_constraint_id);
int pb_remove_constraint(int constraint_id);
int pb_get_constraints(PBConstraintInfo* infos, int max_count, int* count);
int pb_add_debug_line(const float* from, const float* to,
                      float r, float g, float b, float line_width, float life_time);
int pb_add_debug_text(const float* position, const char* text,
                      float r, float g, float b, float size);
int pb_remove_debug(int debug_id);
int pb_remove_all_debug(void);
int pb_set_joint_group_position(int body_id, const int* joint_indices,
                                const float* positions, int num_joints);
int pb_set_joint_group_velocity(int body_id, const int* joint_indices,
                                const float* velocities, int num_joints);
int pb_get_joint_group_states(int body_id, const int* joint_indices, int num_joints,
                              PBJointState* out_states);
int pb_start_logging(int body_id, float log_interval, int max_entries);
int pb_stop_logging(int body_id);
int pb_get_log(int body_id, PBStateLogger* logger);
int pb_predict_trajectory(int body_id, int num_steps, float* out_positions, float* out_orientations);
int pb_predict_joint_trajectory(int body_id, const int* joint_indices, int num_joints,
                                int num_steps, float* out_positions);
int pb_set_control_mode(int body_id, PBControlMode mode);
int pb_get_control_mode(int body_id, PBControlMode* mode);
int pb_set_pid_gains(int body_id, float kp, float ki, float kd);
int pb_get_pid_gains(int body_id, float* kp, float* ki, float* kd);
int pb_get_point_cloud(int body_id, float* out_points, int max_points, int* out_count);
int pb_get_depth_image(int width, int height, const float* view_matrix,
                       const float* proj_matrix, float* depth_out);
int pb_set_collision_margin(int body_id, float margin);
int pb_set_collision_filter(int body_a, int body_b, int enable_collision);
int pb_set_timeout(int timeout_ms);
int pb_get_simulation_stats(int* total_steps, float* total_time,
                            int* num_bodies, int* num_constraints);

#ifdef __cplusplus
}
#endif

#endif
