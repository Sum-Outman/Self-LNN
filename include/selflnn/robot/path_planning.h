/**
 * @file path_planning.h
 * @brief 路径规划算法接口
 *
 * 提供A*全局路径规划、DWA动态窗口局部规划、RRT快速扩展随机树等算法。
 * 纯C实现，不依赖任何第三方库。
 */

#ifndef SELFLNN_PATH_PLANNING_H
#define SELFLNN_PATH_PLANNING_H

#include "selflnn/robot/robot.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PLAN_PATH_MAX_WAYPOINTS 512
#define PLAN_ASTAR_GRID_MAX 256
#define PLAN_DWA_SAMPLES 72
#define PLAN_RRT_MAX_NODES 2048
#define PLAN_RRT_STEP_SIZE 0.5f
#define PLAN_RRT_GOAL_BIAS 0.1f
#define PLAN_RRT_MAX_ITER 5000
#define PLAN_DWA_PREDICT_TIME 2.0f
#define PLAN_DWA_DT 0.1f
#define PLAN_RRT_STAR_MAX_NODES 4096
#define PLAN_RRT_STAR_REWIRE_RADIUS 2.0f
#define PLAN_RRT_STAR_MAX_ITER 10000
#define PLAN_PRM_MAX_NODES 2048
#define PLAN_PRM_MAX_NEIGHBORS 30
#define PLAN_PRM_CONNECT_RADIUS 3.0f
#define PLAN_PRM_MAX_EDGES 8192
#define PLAN_DSTAR_MAX_CELLS 65536
#define PLAN_CHOMP_MAX_POINTS 128
#define PLAN_CHOMP_MAX_ITER 50
#define PLAN_CHOMP_LAMBDA_SMOOTH 0.01f
#define PLAN_CHOMP_LAMBDA_OBS 10.0f
#define PLAN_CHOMP_ETA 0.1f
#define PLAN_STOMP_MAX_POINTS 128
#define PLAN_STOMP_MAX_ITER 50
#define PLAN_STOMP_NUM_NOISY 30
#define PLAN_STOMP_NOISE_STD 0.5f
#define PLAN_STOMP_EXPLORATION 0.1f

typedef enum {
    PLAN_ALGORITHM_ASTAR = 0,
    PLAN_ALGORITHM_DWA = 1,
    PLAN_ALGORITHM_RRT = 2,
    PLAN_ALGORITHM_HYBRID_ASTAR_DWA = 3,
    PLAN_ALGORITHM_RRT_STAR = 4,
    PLAN_ALGORITHM_PRM_STAR = 5,
    PLAN_ALGORITHM_DSTAR_LITE = 6,
    PLAN_ALGORITHM_CHOMP = 7,
    PLAN_ALGORITHM_STOMP = 8
} PlanAlgorithm;

typedef struct {
    float x, y, z;
    float yaw;
    float cost;
    int is_valid;
} PlanWaypoint;

typedef struct {
    float origin_x, origin_y;
    float resolution;
    int width, height;
    uint8_t* cells;
} PlanGridMap;

typedef struct {
    float min_velocity;
    float max_velocity;
    float min_yaw_rate;
    float max_yaw_rate;
    float max_acceleration;
    float max_yaw_acceleration;
    float velocity_resolution;
    float yaw_rate_resolution;
} PlanDWAConfig;

typedef struct {
    float max_iterations;
    float step_size;
    float goal_bias;
    float goal_tolerance;
    float obstacle_clearance;
    int max_nodes;
} PlanRRTConfig;

typedef struct {
    float goal_tolerance;
    float rewire_radius;
    int max_iterations;
    int max_nodes;
} PlanRRTStarConfig;

typedef struct {
    int max_nodes;
    int max_neighbors;
    float connect_radius;
} PlanPRMConfig;

typedef struct {
    float goal_tolerance;
    float obstacle_margin;
    float weight_obstacle;
    float weight_path_length;
} PlanDStarLiteConfig;

typedef struct {
    float smoothness_cost_weight;
    float obstacle_cost_weight;
    float learning_rate;
    int max_iterations;
    int num_points;
    float obstacle_margin;
    float min_step_size;
} PlanCHOMPConfig;

typedef struct {
    int max_iterations;
    int num_noisy;
    float noise_std;
    float exploration;
    int num_points;
    float obstacle_margin;
} PlanSTOMPConfig;

typedef struct {
    PlanAlgorithm algorithm;
    PlanWaypoint start;
    PlanWaypoint goal;
    float robot_radius;
    float obstacle_margin;
    int smooth_path;
    int enable_z_axis;
    PlanDWAConfig dwa_config;
    PlanRRTConfig rrt_config;
    PlanRRTStarConfig rrt_star_config;
    PlanPRMConfig prm_config;
    PlanDStarLiteConfig dstar_lite_config;
    PlanCHOMPConfig chomp_config;
    PlanSTOMPConfig stomp_config;
    void* user_data;
} PlanConfig;

typedef struct {
    PlanWaypoint waypoints[PLAN_PATH_MAX_WAYPOINTS];
    int waypoint_count;
    float total_length;
    float total_cost;
    float computation_time_ms;
    int success;
} PlanResult;

PlanConfig plan_config_default(void);

int plan_set_grid_map(PlanConfig* config, const PlanGridMap* map);

int plan_astar(const PlanConfig* config, PlanResult* result);
int plan_dwa(const PlanConfig* config, const float* current_state, PlanResult* result);
int plan_rrt(const PlanConfig* config, PlanResult* result);

int plan_hybrid_astar_dwa(const PlanConfig* config, const float* current_state, PlanResult* result);

int plan_rrt_star(const PlanConfig* config, PlanResult* result);
int plan_prm_star(const PlanConfig* config, PlanResult* result);
int plan_dstar_lite(const PlanConfig* config, PlanResult* result);
int plan_chomp(const PlanConfig* config, PlanResult* result);
int plan_stomp(const PlanConfig* config, PlanResult* result);

int plan_smooth_path(PlanWaypoint* waypoints, int count, float smooth_factor);
int plan_simplify_path(PlanWaypoint* waypoints, int* count, float tolerance);
int plan_interpolate_path(const PlanWaypoint* waypoints, int count, PlanWaypoint* output, int* output_count, float step);

float plan_path_length(const PlanWaypoint* waypoints, int count);
int plan_check_path_valid(const PlanWaypoint* waypoints, int count, const PlanGridMap* map, float robot_radius);

int plan_waypoint_to_command(const PlanWaypoint* waypoint, RobotCommand* command);
int plan_path_to_trajectory(const PlanWaypoint* waypoints, int count, float speed, float* trajectory, int* trajectory_count);

#ifdef __cplusplus
}
#endif

#endif
