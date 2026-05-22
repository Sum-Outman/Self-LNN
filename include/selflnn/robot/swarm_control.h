/**
 * @file swarm_control.h
 * @brief 多机器人集群控制接口
 *
 * 提供集群编队控制、蜂拥算法、一致性协议和任务分配算法。
 * 纯C实现，不依赖任何第三方库。
 */

#ifndef SELFLNN_SWARM_CONTROL_H
#define SELFLNN_SWARM_CONTROL_H

#include "selflnn/robot/robot.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SWARM_MAX_ROBOTS 64
#define SWARM_MAX_NEIGHBORS 32
#define SWARM_NAME_MAX 64
#define SWARM_FORMATION_MAX 64
#define SWARM_MAX_TASKS 256

typedef enum {
    SWARM_ALGORITHM_FLOCKING = 0,
    SWARM_ALGORITHM_CONSENSUS = 1,
    SWARM_ALGORITHM_FORMATION = 2,
    SWARM_ALGORITHM_TASK_ALLOCATION = 3,
    SWARM_ALGORITHM_SWARM_INTELLIGENCE = 4
} SwarmAlgorithm;

typedef enum {
    FORMATION_TYPE_LINE = 0,
    FORMATION_TYPE_VEE = 1,
    FORMATION_TYPE_CIRCLE = 2,
    FORMATION_TYPE_WEDGE = 3,
    FORMATION_TYPE_SQUARE = 4,
    FORMATION_TYPE_CUSTOM = 5,
    FORMATION_TYPE_COLUMN = 6,
    FORMATION_TYPE_DIAMOND = 7,
    FORMATION_TYPE_CROSS = 8
} FormationType;

typedef enum {
    TASK_ALLOCATION_AUCTION = 0,
    TASK_ALLOCATION_CONSENSUS = 1,
    TASK_ALLOCATION_HUNGARIAN = 2,
    TASK_ALLOCATION_BEES = 3
} TaskAllocationMethod;

typedef struct {
    float position[3];
    float velocity[3];
    float orientation[4];
    float angular_velocity[3];
    int robot_id;
    int is_active;
    float battery_level;
    float task_progress;
    uint32_t udp_seq_counter; /**< ZSFWS-M011: UDP报文递增序号，区分新旧数据包 */
} SwarmRobotState;

typedef struct {
    int robot_id;
    float distance;
    float relative_position[3];
    float relative_velocity[3];
    int is_visible;
} SwarmNeighbor;

typedef struct {
    float cohesion_weight;
    float separation_weight;
    float alignment_weight;
    float obstacle_avoidance_weight;
    float goal_attraction_weight;
    float max_force;
    float neighbor_radius;
    float separation_radius;
    float max_velocity;
    float max_acceleration;
} FlockingParams;

typedef struct {
    float speed;
    float noise_amplitude;
    float interaction_radius;
} VicsekParams;

typedef struct {
    float kappa;
    float theta;
    float beta;
    float lambda;
    float interaction_radius;
} CuckerSmaleParams;

typedef struct {
    float position_gain;
    float velocity_gain;
    float convergence_threshold;
    float communication_delay;
} ConsensusParams;

typedef struct {
    FormationType type;
    float positions[SWARM_FORMATION_MAX][3];
    int position_count;
    float spacing;
    float orientation_offset;
    float rotation_angle;
    float scale;
    int rotate_with_leader;
    int dynamic_reconfiguration;
} FormationConfig;

typedef struct {
    char task_id[64];
    char task_type[64];
    float priority;
    float deadline;
    float required_capability;
    float position[3];
    int assigned_robot_id;
    int is_assigned;
    int is_completed;
    float estimated_cost;
    float reward;
} SwarmTask;

typedef struct {
    TaskAllocationMethod method;
    float communication_range;
    int max_tasks_per_robot;
    int enable_load_balancing;
    float bid_noise;
    int consensus_rounds;
} TaskAllocationConfig;

typedef struct {
    int robot_count;
    int active_count;
    SwarmRobotState robots[SWARM_MAX_ROBOTS];
    SwarmNeighbor neighbors[SWARM_MAX_ROBOTS][SWARM_MAX_NEIGHBORS];
    int neighbor_counts[SWARM_MAX_ROBOTS];
    int leader_id;
    int is_initialized;
    float swarm_center[3];
    float target_positions[SWARM_FORMATION_MAX][3];
    int step_count;
    float swarm_spread;
    float average_speed;
    float formation_error;
} SwarmState;

typedef struct SwarmController SwarmController;

typedef struct {
    int max_robots;
    int enable_collision_avoidance;
    int enable_obstacle_avoidance;
    int enable_leader_following;
    int enable_self_healing;
    int update_rate_hz;
    float control_loop_dt;
    FlockingParams flocking_params;
    VicsekParams vicsek_params;
    CuckerSmaleParams cucker_smale_params;
    ConsensusParams consensus_params;
    FormationConfig formation_config;
    TaskAllocationConfig task_config;
    char name[SWARM_NAME_MAX];
    void* user_data;
} SwarmConfig;

SwarmConfig swarm_config_default(void);

SwarmController* swarm_controller_create(const SwarmConfig* config);
void swarm_controller_destroy(SwarmController* controller);

int swarm_add_robot(SwarmController* controller, int robot_id, const RobotConfig* config);
int swarm_remove_robot(SwarmController* controller, int robot_id);
int swarm_set_leader(SwarmController* controller, int robot_id);

int swarm_init_formation(SwarmController* controller, FormationType type, float spacing);
int swarm_set_formation(SwarmController* controller, const FormationConfig* config);
int swarm_get_formation_positions(SwarmController* controller, int robot_id, float* out_pos, float* out_orient);

int swarm_update_flocking(SwarmController* controller, float dt);
int swarm_update_vicsek(SwarmController* controller, float dt);
int swarm_update_cucker_smale(SwarmController* controller, float dt);
int swarm_update_flocking_enhanced(SwarmController* controller, float dt);
int swarm_update_consensus(SwarmController* controller, float* shared_state, int state_dim, float dt);
int swarm_update_formation(SwarmController* controller, float dt);
int swarm_update_formation_virtual(SwarmController* controller, float dt);
int swarm_update_formation_behavior(SwarmController* controller, float dt);
int swarm_update_all(SwarmController* controller, float dt);
int swarm_deadlock_detect_and_resolve(SwarmController* controller);

int swarm_get_state(SwarmController* controller, SwarmState* state);
int swarm_get_robot_command(SwarmController* controller, int robot_id, RobotCommand* command);
int swarm_set_goal(SwarmController* controller, const float* goal_position, float radius);

int swarm_add_task(SwarmController* controller, const SwarmTask* task);
int swarm_allocate_tasks(SwarmController* controller);
int swarm_allocate_hungarian(SwarmController* controller);
int swarm_allocate_auction_refined(SwarmController* controller);
int swarm_allocate_cbba(SwarmController* controller);
int swarm_get_task_status(SwarmController* controller, const char* task_id, int* assigned_robot, float* progress);

int swarm_enable_algorithm(SwarmController* controller, SwarmAlgorithm algo, int enable);
int swarm_is_algorithm_enabled(const SwarmController* controller, SwarmAlgorithm algo);

int swarm_set_flocking_params(SwarmController* controller, const FlockingParams* params);
int swarm_set_consensus_params(SwarmController* controller, const ConsensusParams* params);
int swarm_set_task_allocation(SwarmController* controller, const TaskAllocationConfig* config);

#ifdef __cplusplus
}
#endif

#endif
