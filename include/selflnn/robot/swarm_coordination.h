/**
 * @file swarm_coordination.h
 * @brief 多机器人群体协同控制系统接口
 */

#ifndef SELFLNN_SWARM_COORDINATION_H
#define SELFLNN_SWARM_COORDINATION_H

#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SW_MAX_ROBOTS 32
#define SW_MAX_FORMATIONS 16
#define SW_MAX_TASKS 128
#define SW_MAX_PATHS 1024

typedef enum {
    FORMATION_LINE = 0,
    FORMATION_V = 1,
    FORMATION_CIRCLE = 2,
    FORMATION_GRID = 3,
    FORMATION_DIAMOND = 4,
    FORMATION_WEDGE = 5,
    FORMATION_COLUMN = 6,
    FORMATION_CUSTOM = 7
} FormationType;

typedef struct {
    FormationType type;
    int formation_id;
    int robot_count;
    int robot_ids[SW_MAX_ROBOTS];
    float spacing;
    float orientation;
    float center[3];
    int leader_id;
    int active;
} SwarmFormation;

typedef struct {
    int task_id;
    char name[64];
    int required_robots;
    char required_capabilities[16][32];
    int capability_count;
    float priority;
    float estimated_duration;
    time_t assigned_at;
    time_t deadline;
    int* assigned_robots;
    int assigned_count;
    int completed;
    float progress;
} SwarmTask;

typedef struct {
    int robot_id;
    char robot_name[32];
    float position[3];
    float velocity[3];
    float battery;
    float load;
    int task_id;
    int formation_id;
    int is_leader;
    char capabilities[8][32];
    int capability_count;
    int online;
    time_t last_heartbeat;
    float goal_position[3];       /**< 目标位置 */
    float velocity_cmd[3];        /**< 速度命令输出 */
    int goal_active;              /**< 目标活跃标志 */
} SwarmRobot;

typedef struct {
    int conflict_id;
    int robot_a;
    int robot_b;
    char resource[64];
    float distance;
    int resolved;
    int resolution_strategy;
} SwarmConflict;

typedef struct SwarmCoordinator SwarmCoordinator;

SwarmCoordinator* swarm_coordinator_create(void);
void swarm_coordinator_free(SwarmCoordinator* sc);

/* 机器人管理 */
int sw_register_robot(SwarmCoordinator* sc, const SwarmRobot* robot);
int sw_unregister_robot(SwarmCoordinator* sc, int robot_id);
int sw_get_robot(const SwarmCoordinator* sc, int robot_id, SwarmRobot* out);
int sw_list_robots(const SwarmCoordinator* sc, SwarmRobot* out, int max_count);
int sw_heartbeat(SwarmCoordinator* sc, int robot_id);

/* 编队控制 */
int sw_create_formation(SwarmCoordinator* sc, FormationType type, const int* robot_ids, int count, int leader_id, float spacing);
int sw_disband_formation(SwarmCoordinator* sc, int formation_id);
int sw_get_formation_targets(const SwarmCoordinator* sc, int formation_id, float* targets, int max_robots);
int sw_move_formation(SwarmCoordinator* sc, int formation_id, const float* delta);
int sw_rotate_formation(SwarmCoordinator* sc, int formation_id, float angle);

/* 任务分配 */
int sw_assign_task(SwarmCoordinator* sc, const SwarmTask* task);
int sw_complete_task(SwarmCoordinator* sc, int task_id);
int sw_get_task(const SwarmCoordinator* sc, int task_id, SwarmTask* out);
int sw_list_tasks(const SwarmCoordinator* sc, SwarmTask* out, int max_count);
int sw_reassign_failed_task(SwarmCoordinator* sc, int task_id, int failed_robot);

/* 冲突解决 */
int sw_detect_conflicts(SwarmCoordinator* sc, SwarmConflict* out, int max_count);
int sw_resolve_conflict(SwarmCoordinator* sc, int conflict_id, int strategy);

/* 调度与负载均衡 */
int sw_schedule_tasks(SwarmCoordinator* sc);
int sw_load_balance(SwarmCoordinator* sc);

/* 协同操作 */
int sw_cooperative_lift(SwarmCoordinator* sc, const int* robot_ids, int count, const float* object_pos, float weight);
int sw_cooperative_transport(SwarmCoordinator* sc, const int* robot_ids, int count, const float* start, const float* end);

#ifdef __cplusplus
}
#endif
#endif
