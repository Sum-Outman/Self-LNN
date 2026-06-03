/**
 * @file path_planning.c
 * @brief 路径规划算法实现
 *
 * 实现A*全局路径规划、DWA动态窗口局部规划、RRT快速扩展随机树。
 * 纯C实现，浮点运算，无第三方库依赖。
 */

#include "selflnn/robot/path_planning.h"
#include "selflnn/robot/robot.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/utils/memory_utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h> /* ZSFJJJ-FIX: CLOCKS_PER_SEC 需要 time.h */

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ZSFJJJ-FIX: slam_map_update_fn 回调类型定义。
 * 该类型在头文件中缺失，导致编译错误 */
typedef int (*slam_map_update_fn)(PlanGridMap* map, void* ctx);

#define PLAN_MIN(a,b) (((a)<(b))?(a):(b))
#define PLAN_MAX(a,b) (((a)>(b))?(a):(b))
#define PLAN_CLAMP(v,lo,hi) (((v)<(lo))?(lo):(((v)>(hi))?(hi):(v)))
#define PLAN_ABS(x) (((x)>=0)?(x):-(x))
#define PLAN_SQ(x) ((x)*(x))
#define PLAN_DIST2(x1,y1,x2,y2) (PLAN_SQ((x1)-(x2))+PLAN_SQ((y1)-(y2)))
#define PLAN_DIST(x1,y1,x2,y2) (sqrtf(PLAN_DIST2(x1,y1,x2,y2)))
#define PLAN_WRAP_ANGLE(a) \
    do { \
        while ((a) > (float)M_PI) (a) -= 2.0f*(float)M_PI; \
        while ((a) < -(float)M_PI) (a) += 2.0f*(float)M_PI; \
    } while(0)

/* S-010修复: 考虑运动学约束的真实距离函数 */

/**
 * @brief 计算Dubins曲线长度的可接受下界（用于A*启发式）
 * 
 * 对于具有最小转弯半径ρ的Dubins车辆，从(x1,y1)到(x2,y2,θ_goal)的最短路径
 * 长度至少为: max(欧氏距离, ρ·|Δθ|)
 * 其中Δθ是当前位置到目标朝向所需的最小角度变化。
 * 
 * 该启发式满足可接受性（不低估真实代价）。
 */
static float plan_dubins_heuristic(float x1, float y1, float gx, float gy, float goal_yaw, float min_radius) {
    float euclidean = PLAN_DIST(x1, y1, gx, gy);
    if (min_radius <= 0.0f) return euclidean;
    
    /* 估计当前朝向：从(x1,y1)到(gx,gy)的直线方向 */
    float estimated_heading = atan2f(gy - y1, gx - x1);
    float heading_diff = goal_yaw - estimated_heading;
    PLAN_WRAP_ANGLE(heading_diff);
    
    /* Dubins下界：需要至少min_radius * |Δθ|的距离来转弯 */
    float turning_cost = min_radius * fabsf(heading_diff);
    
    return PLAN_MAX(euclidean, turning_cost);
}

/**
 * @brief 计算考虑运动学约束的SE(2)加权距离
 * 
 * d_SE2 = sqrt(dx² + dy² + w_θ·(min_radius·Δθ)²)
 * 
 * 该距离度量同时考虑了平移距离和朝向差异，
 * 其中朝向差异通过最小转弯半径缩放以保持单位一致。
 * 适用于RRT/RRT*的代价函数。
 * 
 * @param x1,y1 点1坐标
 * @param heading1 点1朝向（若无朝向信息，传任意值）
 * @param x2,y2 点2坐标
 * @param heading2 点2朝向（若无朝向信息，传任意值）
 * @param min_radius 最小转弯半径（>0启用运动学约束，<=0回退为欧氏距离）
 * @param use_heading 是否使用朝向信息（0=纯欧氏距离，1=考虑朝向）
 * @return SE(2)加权距离
 */
static float plan_kinematic_distance(float x1, float y1, float heading1,
                                      float x2, float y2, float heading2,
                                      float min_radius, int use_heading) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float euclidean_sq = dx * dx + dy * dy;
    
    if (!use_heading || min_radius <= 1e-6f) {
        return sqrtf(euclidean_sq);
    }
    
    /* 朝向差异（归一化到[-π, π]） */
    float dtheta = heading2 - heading1;
    PLAN_WRAP_ANGLE(dtheta);
    
    /* SE(2)加权距离: 位置差异 + 朝向差异缩放
     * 权重w_θ控制朝向在总代价中的比重
     * 使用min_radius缩放使朝向代价与平移代价单位一致 */
    float w_theta = 0.3f;
    float angular_cost_sq = (min_radius * dtheta) * (min_radius * dtheta);
    
    return sqrtf(euclidean_sq + w_theta * angular_cost_sq);
}

/**
 * @brief 基于父节点方向推断当前节点朝向
 * @param parent_x,parent_y 父节点坐标
 * @param child_x,child_y 当前节点坐标
 * @return 推断的朝向角（弧度）
 */
static float plan_infer_heading(float parent_x, float parent_y, float child_x, float child_y) {
    return atan2f(child_y - parent_y, child_x - parent_x);
}

PlanConfig plan_config_default(void)
{
    PlanConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.algorithm = PLAN_ALGORITHM_ASTAR;
    cfg.start.x = 0.0f;
    cfg.start.y = 0.0f;
    cfg.start.z = 0.0f;
    cfg.start.yaw = 0.0f;
    cfg.goal.x = 1.0f;
    cfg.goal.y = 1.0f;
    cfg.goal.z = 0.0f;
    cfg.goal.yaw = 0.0f;
    cfg.robot_radius = 0.3f;
    cfg.obstacle_margin = 0.1f;
    cfg.smooth_path = 1;
    cfg.enable_z_axis = 0;
    cfg.dwa_config.min_velocity = -0.5f;
    cfg.dwa_config.max_velocity = 1.0f;
    cfg.dwa_config.min_yaw_rate = -(float)M_PI / 2.0f;
    cfg.dwa_config.max_yaw_rate = (float)M_PI / 2.0f;
    cfg.dwa_config.max_acceleration = 0.5f;
    cfg.dwa_config.max_yaw_acceleration = (float)M_PI / 4.0f;
    cfg.dwa_config.velocity_resolution = 0.1f;
    cfg.dwa_config.yaw_rate_resolution = (float)M_PI / 18.0f;
    cfg.rrt_config.max_iterations = PLAN_RRT_MAX_ITER;
    cfg.rrt_config.step_size = PLAN_RRT_STEP_SIZE;
    cfg.rrt_config.goal_bias = PLAN_RRT_GOAL_BIAS;
    cfg.rrt_config.goal_tolerance = 0.3f;
    cfg.rrt_config.obstacle_clearance = 0.3f;
    cfg.rrt_config.max_nodes = PLAN_RRT_MAX_NODES;
    return cfg;
}

int plan_set_grid_map(PlanConfig* config, const PlanGridMap* map)
{
    if (!config || !map || !map->cells) return -1;
    config->user_data = (void*)map;
    return 0;
}

static int grid_is_occupied(const PlanGridMap* map, float wx, float wy, float radius)
{
    if (!map || !map->cells) return 1;
    int cx = (int)((wx - map->origin_x) / map->resolution);
    int cy = (int)((wy - map->origin_y) / map->resolution);
    int rad_cells = (int)(radius / map->resolution) + 1;
    for (int dy = -rad_cells; dy <= rad_cells; dy++)
    {
        for (int dx = -rad_cells; dx <= rad_cells; dx++)
        {
            int nx = cx + dx;
            int ny = cy + dy;
            if (nx < 0 || nx >= map->width || ny < 0 || ny >= map->height) return 1;
            if (map->cells[ny * map->width + nx] > 0) return 1;
        }
    }
    return 0;
}

static int grid_is_valid(const PlanGridMap* map, int x, int y)
{
    if (!map || !map->cells) return 0;
    return (x >= 0 && x < map->width && y >= 0 && y < map->height && map->cells[y * map->width + x] == 0);
}

/* ==================== A* 全局路径规划 ==================== */

typedef struct
{
    int x, y;
    float g, f;
    int parent_idx;
    int in_open;
    int in_closed;
} AStarNode;

int plan_astar(const PlanConfig* config, PlanResult* result)
{
    if (!config || !result) return -1;
    const PlanGridMap* map = (const PlanGridMap*)config->user_data;
    if (!map || !map->cells) return -1;

    plan_slam_ensure_fresh((PlanConfig*)config);

    memset(result, 0, sizeof(PlanResult));

    int sx = (int)((config->start.x - map->origin_x) / map->resolution);
    int sy = (int)((config->start.y - map->origin_y) / map->resolution);
    int gx = (int)((config->goal.x - map->origin_x) / map->resolution);
    int gy = (int)((config->goal.y - map->origin_y) / map->resolution);

    if (sx < 0 || sx >= map->width || sy < 0 || sy >= map->height) return -1;
    if (gx < 0 || gx >= map->width || gy < 0 || gy >= map->height) return -1;
    if (map->cells[sy * map->width + sx] > 0) return -1;

    int total_cells = map->width * map->height;
    AStarNode* nodes = (AStarNode*)calloc(total_cells, sizeof(AStarNode));
    if (!nodes) return -1;

    for (int i = 0; i < total_cells; i++)
    {
        int cy = i / map->width;
        int cx = i % map->width;
        nodes[i].x = cx;
        nodes[i].y = cy;
        nodes[i].g = 1e9f;
        nodes[i].f = 1e9f;
        nodes[i].parent_idx = -1;
    }

    int start_idx = sy * map->width + sx;
    nodes[start_idx].g = 0.0f;
    /* S-010修复: 使用Dubins运动学感知启发式替代纯欧氏距离 */
    nodes[start_idx].f = plan_dubins_heuristic((float)sx, (float)sy, (float)gx, (float)gy,
        config->goal.yaw, config->min_turning_radius);
    nodes[start_idx].in_open = 1;

    int* open_list = (int*)malloc(total_cells * sizeof(int));
    if (!open_list) { free(nodes); return -1; }
    
    /* 二叉堆优先队列索引数组：heap_idx[node_idx] = 在heap中的位置(-1表示不在堆中) */
    int* heap_idx = (int*)malloc(total_cells * sizeof(int));
    if (!heap_idx) { free(open_list); free(nodes); return -1; }
    for (int i = 0; i < total_cells; i++) heap_idx[i] = -1;
    
    int heap_size = 0;
    
    /* 二叉堆上浮操作 */
    #define HEAP_SIFT_UP(h, idx_arr, n_arr) do { \
        int _ci = h; \
        while (_ci > 0) { \
            int _pi = (_ci - 1) >> 1; \
            if (n_arr[idx_arr[_ci]].f >= n_arr[idx_arr[_pi]].f) break; \
            int _tmp = idx_arr[_ci]; idx_arr[_ci] = idx_arr[_pi]; idx_arr[_pi] = _tmp; \
            heap_idx[idx_arr[_ci]] = _ci; heap_idx[idx_arr[_pi]] = _pi; \
            _ci = _pi; \
        } \
    } while(0)
    
    /* 二叉堆下沉操作 */
    #define HEAP_SIFT_DOWN(h, sz, idx_arr, n_arr) do { \
        int _ci = h; \
        while (1) { \
            int _l = (_ci << 1) + 1, _r = _l + 1, _min_i = _ci; \
            if (_l < sz && n_arr[idx_arr[_l]].f < n_arr[idx_arr[_min_i]].f) _min_i = _l; \
            if (_r < sz && n_arr[idx_arr[_r]].f < n_arr[idx_arr[_min_i]].f) _min_i = _r; \
            if (_min_i == _ci) break; \
            int _tmp = idx_arr[_ci]; idx_arr[_ci] = idx_arr[_min_i]; idx_arr[_min_i] = _tmp; \
            heap_idx[idx_arr[_ci]] = _ci; heap_idx[idx_arr[_min_i]] = _min_i; \
            _ci = _min_i; \
        } \
    } while(0)
    
    /* 初始节点入堆 */
    open_list[0] = start_idx; heap_size = 1; heap_idx[start_idx] = 0;
    
    int found = 0;
    int goal_idx = -1;
    int dirs[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{-1,1},{1,-1},{-1,-1}};
    float dir_costs[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.414f, 1.414f, 1.414f, 1.414f};

    while (heap_size > 0)
    {
        /* 二叉堆弹出最小f值节点 O(log n) */
        int current = open_list[0];
        heap_idx[current] = -1;
        if (heap_size > 1) {
            open_list[0] = open_list[--heap_size];
            heap_idx[open_list[0]] = 0;
            HEAP_SIFT_DOWN(0, heap_size, open_list, nodes);
        } else {
            heap_size = 0;
        }

        nodes[current].in_open = 0;
        nodes[current].in_closed = 1;

        if (nodes[current].x == gx && nodes[current].y == gy)
        {
            found = 1;
            goal_idx = current;
            break;
        }

        for (int d = 0; d < 8; d++)
        {
            int nx = nodes[current].x + dirs[d][0];
            int ny = nodes[current].y + dirs[d][1];
            if (nx < 0 || nx >= map->width || ny < 0 || ny >= map->height) continue;
            int nidx = ny * map->width + nx;
            if (nodes[nidx].in_closed) continue;
/* A*碰撞检测安全缺陷 — 原仅检查单个栅格,
             * 未考虑机器人本体半径。现在使用grid_is_occupied做半径感知检测 */
            if (grid_is_occupied(map,
                map->origin_x + ((float)nx + 0.5f) * map->resolution,
                map->origin_y + ((float)ny + 0.5f) * map->resolution,
                config->robot_radius)) continue;

            float tent_g = nodes[current].g + dir_costs[d];
            if (tent_g < nodes[nidx].g)
            {
                nodes[nidx].g = tent_g;
                /* S-010修复: 使用Dubins运动学感知启发式 */
                float h_val = plan_dubins_heuristic((float)nx, (float)ny,
                    (float)gx, (float)gy, config->goal.yaw, config->min_turning_radius);
                nodes[nidx].f = tent_g + h_val;
                nodes[nidx].parent_idx = current;
                if (!nodes[nidx].in_open)
                {
                    nodes[nidx].in_open = 1;
                    /* 二叉堆插入 O(log n) */
                    open_list[heap_size] = nidx;
                    heap_idx[nidx] = heap_size;
                    HEAP_SIFT_UP(heap_size, open_list, nodes);
                    heap_size++;
                }
                else if (heap_idx[nidx] >= 0) {
                    /* 节点已在堆中，调整位置 */
                    HEAP_SIFT_UP(heap_idx[nidx], open_list, nodes);
                }
            }
        }
    }

    if (found)
    {
        int path_cells[PLAN_PATH_MAX_WAYPOINTS];
        int pc = 0;
        int cur = goal_idx;
        while (cur >= 0 && pc < PLAN_PATH_MAX_WAYPOINTS)
        {
            path_cells[pc++] = cur;
            cur = nodes[cur].parent_idx;
        }
        for (int i = 0; i < pc && i < PLAN_PATH_MAX_WAYPOINTS; i++)
        {
            int idx = path_cells[pc - 1 - i];
            int cx = idx % map->width;
            int cy = idx / map->width;
            result->waypoints[i].x = map->origin_x + (cx + 0.5f) * map->resolution;
            result->waypoints[i].y = map->origin_y + (cy + 0.5f) * map->resolution;
            result->waypoints[i].z = config->start.z;
            result->waypoints[i].yaw = 0.0f;
            result->waypoints[i].cost = nodes[idx].g;
            result->waypoints[i].is_valid = 1;
        }
        result->waypoint_count = pc;
        result->total_length = plan_path_length(result->waypoints, result->waypoint_count);
        result->total_cost = nodes[goal_idx].g;
        result->success = 1;

        if (config->smooth_path && result->waypoint_count > 2)
            plan_smooth_path(result->waypoints, result->waypoint_count, 0.3f);
    }

    free(open_list);
    free(heap_idx);
    free(nodes);
    return result->success ? 0 : -1;
}

/* ==================== DWA 动态窗口局部规划 ==================== */

static float dwa_trajectory_cost(const PlanConfig* config, float v, float w,
                                  const float* state, const PlanGridMap* map,
                                  float* end_x, float* end_y, float* end_yaw)
{
    float x = state[0], y = state[1], yaw = state[2];
    float dt = PLAN_DWA_DT;
    float time = 0.0f;
    float min_obs_dist = 1e9f;
    float goal_dist = 1e9f;
    int valid = 1;

    while (time < PLAN_DWA_PREDICT_TIME)
    {
        yaw += w * dt;
        x += v * cosf(yaw) * dt;
        y += v * sinf(yaw) * dt;
        time += dt;

        if (map && grid_is_occupied(map, x, y, config->robot_radius))
        {
            valid = 0;
            break;
        }

        float od = 1e9f;
        if (map)
        {
            for (int dx = -2; dx <= 2; dx++)
            {
                for (int dy = -2; dy <= 2; dy++)
                {
                    int cx = (int)((x - map->origin_x) / map->resolution) + dx;
                    int cy = (int)((y - map->origin_y) / map->resolution) + dy;
                    if (cx >= 0 && cx < map->width && cy >= 0 && cy < map->height)
                    {
                        if (map->cells[cy * map->width + cx] > 0)
                        {
                            float ox = map->origin_x + cx * map->resolution;
                            float oy = map->origin_y + cy * map->resolution;
                            float d = PLAN_DIST(x, y, ox, oy);
                            if (d < od) od = d;
                        }
                    }
                }
            }
        }
        if (od < min_obs_dist) min_obs_dist = od;

        float gd = PLAN_DIST(x, y, config->goal.x, config->goal.y);
        if (gd < goal_dist) goal_dist = gd;
    }

    if (!valid) return 1e9f;

    *end_x = x;
    *end_y = y;
    *end_yaw = yaw;

    float heading = PLAN_ABS(yaw - atan2f(config->goal.y - state[1], config->goal.x - state[0]));
    if (heading > (float)M_PI) heading = 2.0f * (float)M_PI - heading;

    float obs_cost = (min_obs_dist < config->robot_radius + config->obstacle_margin) ? 1e9f : 0.0f;
    float heading_cost = heading / (float)M_PI;
    float goal_cost = goal_dist / PLAN_MAX(config->goal.x - config->start.x, 0.01f);
    float speed_cost = 1.0f - PLAN_ABS(v) / config->dwa_config.max_velocity;

    return 0.5f * heading_cost + 0.3f * goal_cost + 0.1f * speed_cost + 0.1f * obs_cost;
}

int plan_dwa(const PlanConfig* config, const float* current_state, PlanResult* result)
{
    if (!config || !result || !current_state) return -1;
    const PlanGridMap* map = (const PlanGridMap*)config->user_data;

    plan_slam_ensure_fresh((PlanConfig*)config);

    memset(result, 0, sizeof(PlanResult));

    float v0 = current_state[3];
    float w0 = current_state[4];

    float v_min = PLAN_MAX(config->dwa_config.min_velocity, v0 - config->dwa_config.max_acceleration * PLAN_DWA_DT);
    float v_max = PLAN_MIN(config->dwa_config.max_velocity, v0 + config->dwa_config.max_acceleration * PLAN_DWA_DT);
    float w_min = PLAN_MAX(config->dwa_config.min_yaw_rate, w0 - config->dwa_config.max_yaw_acceleration * PLAN_DWA_DT);
    float w_max = PLAN_MIN(config->dwa_config.max_yaw_rate, w0 + config->dwa_config.max_yaw_acceleration * PLAN_DWA_DT);

    float best_v = 0.0f, best_w = 0.0f;
    float best_cost = 1e9f;
    float best_ex = current_state[0], best_ey = current_state[1], best_eyaw = current_state[2];
    int sample_count = 0;

    for (float v = v_min; v <= v_max + 0.001f; v += config->dwa_config.velocity_resolution)
    {
        for (float w = w_min; w <= w_max + 0.001f; w += config->dwa_config.yaw_rate_resolution)
        {
            float ex, ey, eyaw;
            float cost = dwa_trajectory_cost(config, v, w, current_state, map, &ex, &ey, &eyaw);
            sample_count++;
            if (cost < best_cost)
            {
                best_cost = cost;
                best_v = v;
                best_w = w;
                best_ex = ex;
                best_ey = ey;
                best_eyaw = eyaw;
            }
        }
    }

    if (best_cost > 1e8f) return -1;

    float dt = PLAN_DWA_PREDICT_TIME;
    result->waypoints[0].x = current_state[0];
    result->waypoints[0].y = current_state[1];
    result->waypoints[0].z = 0.0f;
    result->waypoints[0].yaw = current_state[2];
    result->waypoints[0].cost = 0.0f;
    result->waypoints[0].is_valid = 1;

    float step_dt = 0.2f;
    int steps = (int)(dt / step_dt);
    if (steps > PLAN_PATH_MAX_WAYPOINTS - 1) steps = PLAN_PATH_MAX_WAYPOINTS - 1;
    float cx = current_state[0], cy = current_state[1], cyaw = current_state[2];
    for (int i = 1; i <= steps; i++)
    {
        cyaw += best_w * step_dt;
        cx += best_v * cosf(cyaw) * step_dt;
        cy += best_v * sinf(cyaw) * step_dt;
        result->waypoints[i].x = cx;
        result->waypoints[i].y = cy;
        result->waypoints[i].z = 0.0f;
        result->waypoints[i].yaw = cyaw;
        result->waypoints[i].cost = (float)i / (float)steps;
        result->waypoints[i].is_valid = 1;
    }
    result->waypoint_count = steps + 1;
    result->total_length = plan_path_length(result->waypoints, result->waypoint_count);
    result->total_cost = best_cost;
    result->success = 1;

    return 0;
}

/* ==================== RRT 快速扩展随机树 ==================== */

typedef struct RRTNode
{
    float x, y;
    int parent_idx;
} RRTNode;

int plan_rrt(const PlanConfig* config, PlanResult* result)
{
    if (!config || !result) return -1;
    const PlanGridMap* map = (const PlanGridMap*)config->user_data;

    plan_slam_ensure_fresh((PlanConfig*)config);

    memset(result, 0, sizeof(PlanResult));

    RRTNode* nodes = (RRTNode*)malloc(config->rrt_config.max_nodes * sizeof(RRTNode));
    if (!nodes) return -1;

    nodes[0].x = config->start.x;
    nodes[0].y = config->start.y;
    nodes[0].parent_idx = -1;
    int node_count = 1;
    int found = 0;

    float min_x = PLAN_MIN(config->start.x, config->goal.x) - 5.0f;
    float max_x = PLAN_MAX(config->start.x, config->goal.x) + 5.0f;
    float min_y = PLAN_MIN(config->start.y, config->goal.y) - 5.0f;
    float max_y = PLAN_MAX(config->start.y, config->goal.y) + 5.0f;

    for (int iter = 0; iter < config->rrt_config.max_iterations && node_count < config->rrt_config.max_nodes; iter++)
    {
        float rx, ry;
        if (secure_random_float() < config->rrt_config.goal_bias)
        {
            rx = config->goal.x;
            ry = config->goal.y;
        }
        else
        {
            rx = min_x + secure_random_float() * (max_x - min_x);
            ry = min_y + secure_random_float() * (max_y - min_y);
        }

        int nearest = 0;
        /* S-010修复: 使用运动学SE(2)距离进行最近邻搜索 */
        /* 采样点的朝向：使用随机方向（无朝向信息时） */
        float sample_heading = atan2f(ry - nodes[0].y, rx - nodes[0].x);
        float nearest_dist;
        /* 根节点的朝向：从起点朝向goal推断 */
        if (node_count > 0 && nodes[0].parent_idx < 0) {
            nearest_dist = plan_kinematic_distance(nodes[0].x, nodes[0].y,
                config->start.yaw, rx, ry, sample_heading,
                config->min_turning_radius, 1);
        } else {
            nearest_dist = PLAN_DIST2(nodes[0].x, nodes[0].y, rx, ry);
        }
        for (int i = 1; i < node_count; i++)
        {
            float heading_i;
            if (config->min_turning_radius > 0.0f && nodes[i].parent_idx >= 0) {
                heading_i = plan_infer_heading(nodes[nodes[i].parent_idx].x,
                    nodes[nodes[i].parent_idx].y, nodes[i].x, nodes[i].y);
            } else {
                heading_i = 0.0f;
            }
            float d = plan_kinematic_distance(nodes[i].x, nodes[i].y, heading_i,
                rx, ry, sample_heading, config->min_turning_radius,
                config->min_turning_radius > 0.0f ? 1 : 0);
            if (d < nearest_dist)
            {
                nearest_dist = d;
                nearest = i;
            }
        }

        float dx = rx - nodes[nearest].x;
        float dy = ry - nodes[nearest].y;
        /* 使用欧氏距离进行步进方向计算（运动学距离用于最近邻搜索） */
        float dist_xy = sqrtf(dx * dx + dy * dy);
        float step = PLAN_MIN(config->rrt_config.step_size, dist_xy);
        float nx = nodes[nearest].x + dx / dist_xy * step;
        float ny = nodes[nearest].y + dy / dist_xy * step;

        if (map && grid_is_occupied(map, nx, ny, config->robot_radius))
            continue;

        nodes[node_count].x = nx;
        nodes[node_count].y = ny;
        nodes[node_count].parent_idx = nearest;
        node_count++;

        /* S-010修复: 使用运动学距离检查是否到达目标 */
        float new_heading = plan_infer_heading(nodes[nearest].x, nodes[nearest].y, nx, ny);
        float goal_dist = plan_kinematic_distance(nx, ny, new_heading,
            config->goal.x, config->goal.y, config->goal.yaw,
            config->min_turning_radius, config->min_turning_radius > 0.0f ? 1 : 0);
        if (goal_dist < config->rrt_config.goal_tolerance)
        {
            found = 1;
            break;
        }
    }

    if (!found)
    {
        float best_dist = 1e9f;
        int best_idx = -1;
        for (int i = 0; i < node_count; i++)
        {
            /* S-010修复: 使用运动学距离评估最佳目标候选 */
            float heading_i;
            if (nodes[i].parent_idx >= 0) {
                heading_i = plan_infer_heading(nodes[nodes[i].parent_idx].x,
                    nodes[nodes[i].parent_idx].y, nodes[i].x, nodes[i].y);
            } else {
                heading_i = config->start.yaw;
            }
            float d = plan_kinematic_distance(nodes[i].x, nodes[i].y, heading_i,
                config->goal.x, config->goal.y, config->goal.yaw,
                config->min_turning_radius, config->min_turning_radius > 0.0f ? 1 : 0);
            if (d < best_dist)
            {
                best_dist = d;
                best_idx = i;
            }
        }
        int path_idx[PLAN_PATH_MAX_WAYPOINTS];
        int pc = 0;
        int cur = (best_idx >= 0) ? best_idx : 0;
        while (cur >= 0 && pc < PLAN_PATH_MAX_WAYPOINTS)
        {
            path_idx[pc++] = cur;
            cur = nodes[cur].parent_idx;
        }
        if (pc <= 1) { free(nodes); return -1; }
        for (int i = 0; i < pc; i++)
        {
            int idx = path_idx[pc - 1 - i];
            result->waypoints[i].x = nodes[idx].x;
            result->waypoints[i].y = nodes[idx].y;
            result->waypoints[i].z = config->start.z;
            result->waypoints[i].yaw = 0.0f;
            result->waypoints[i].cost = (float)i / (float)pc;
            result->waypoints[i].is_valid = 1;
        }
        result->waypoint_count = pc;
        result->total_length = plan_path_length(result->waypoints, result->waypoint_count);
        result->success = 1;
        free(nodes);
        return 0;
    }

    int path_idx[PLAN_PATH_MAX_WAYPOINTS];
    int pc = 0;
    int cur = node_count - 1;
    while (cur >= 0 && pc < PLAN_PATH_MAX_WAYPOINTS)
    {
        path_idx[pc++] = cur;
        cur = nodes[cur].parent_idx;
    }
    for (int i = 0; i < pc; i++)
    {
        int idx = path_idx[pc - 1 - i];
        result->waypoints[i].x = nodes[idx].x;
        result->waypoints[i].y = nodes[idx].y;
        result->waypoints[i].z = config->start.z;
        result->waypoints[i].yaw = 0.0f;
        result->waypoints[i].cost = (float)i / (float)pc;
        result->waypoints[i].is_valid = 1;
    }
    result->waypoint_count = pc;
    result->total_length = plan_path_length(result->waypoints, result->waypoint_count);
    result->success = 1;

    if (config->smooth_path && result->waypoint_count > 2)
        plan_smooth_path(result->waypoints, result->waypoint_count, 0.3f);

    free(nodes);
    return 0;
}

int plan_hybrid_astar_dwa(const PlanConfig* config, const float* current_state, PlanResult* result)
{
    if (!config || !result) return -1;

    PlanResult global_result;
    memset(&global_result, 0, sizeof(global_result));

    int gret = plan_astar(config, &global_result);
    if (gret != 0 || !global_result.success || global_result.waypoint_count < 2)
    {
        PlanResult local_result;
        memset(&local_result, 0, sizeof(local_result));
        int lret = plan_dwa(config, current_state, &local_result);
        if (lret == 0) memcpy(result, &local_result, sizeof(PlanResult));
        return lret;
    }

    int closest_idx = 0;
    float min_dist = 1e9f;
    for (int i = 0; i < global_result.waypoint_count; i++)
    {
        float d = PLAN_DIST(global_result.waypoints[i].x, global_result.waypoints[i].y,
                            current_state[0], current_state[1]);
        if (d < min_dist) { min_dist = d; closest_idx = i; }
    }

    int remaining = global_result.waypoint_count - closest_idx;
    if (remaining <= 0) { memcpy(result, &global_result, sizeof(PlanResult)); return 0; }

    PlanResult hybrid;
    memset(&hybrid, 0, sizeof(hybrid));

    for (int i = 0; i < remaining && i < PLAN_PATH_MAX_WAYPOINTS; i++)
    {
        int src = closest_idx + i;
        hybrid.waypoints[i] = global_result.waypoints[src];
    }
    hybrid.waypoint_count = remaining;
    hybrid.total_length = plan_path_length(hybrid.waypoints, hybrid.waypoint_count);
    hybrid.total_cost = global_result.total_cost;
    hybrid.success = 1;

    memcpy(result, &hybrid, sizeof(PlanResult));
    return 0;
}

/* ==================== 路径平滑与简化 ==================== */

int plan_smooth_path(PlanWaypoint* waypoints, int count, float smooth_factor)
{
    if (!waypoints || count < 3 || smooth_factor <= 0.0f) return -1;
    float* ox = (float*)malloc(count * sizeof(float));
    float* oy = (float*)malloc(count * sizeof(float));
    if (!ox || !oy) { free(ox); free(oy); return -1; }
    for (int i = 0; i < count; i++)
    {
        ox[i] = waypoints[i].x;
        oy[i] = waypoints[i].y;
    }
    float tol = 1e-4f;
    for (int iter = 0; iter < 100; iter++)
    {
        float max_change = 0.0f;
        for (int i = 1; i < count - 1; i++)
        {
            float nx = waypoints[i].x + smooth_factor * (ox[i] + waypoints[i - 1].x + waypoints[i + 1].x - 3.0f * waypoints[i].x);
            float ny = waypoints[i].y + smooth_factor * (oy[i] + waypoints[i - 1].y + waypoints[i + 1].y - 3.0f * waypoints[i].y);
            float dx = nx - waypoints[i].x;
            float dy = ny - waypoints[i].y;
            float change = sqrtf(dx * dx + dy * dy);
            if (change > max_change) max_change = change;
            waypoints[i].x = nx;
            waypoints[i].y = ny;
        }
        if (max_change < tol) break;
    }
    free(ox);
    free(oy);
    return 0;
}

int plan_simplify_path(PlanWaypoint* waypoints, int* count, float tolerance)
{
    if (!waypoints || !count || *count < 3 || tolerance <= 0.0f) return -1;
    int in_count = *count;
    char* keep = (char*)calloc(in_count, 1);
    if (!keep) return -1;
    keep[0] = 1;
    keep[in_count - 1] = 1;
    int* stack = (int*)malloc(in_count * sizeof(int));
    if (!stack) { free(keep); return -1; }
    int stack_count = 0;
    stack[stack_count++] = 0;
    stack[stack_count++] = in_count - 1;
    while (stack_count > 0)
    {
        int end = stack[--stack_count];
        int start = stack[--stack_count];
        float max_dist = 0.0f;
        int max_idx = -1;
        float ax = waypoints[start].x, ay = waypoints[start].y;
        float bx = waypoints[end].x, by = waypoints[end].y;
        float abx = bx - ax, aby = by - ay;
        float ab_len = sqrtf(abx * abx + aby * aby);
        float ab_nx = (ab_len > 1e-6f) ? -aby / ab_len : 0.0f;
        float ab_ny = (ab_len > 1e-6f) ? abx / ab_len : 0.0f;
        for (int i = start + 1; i < end; i++)
        {
            float px = waypoints[i].x - ax;
            float py = waypoints[i].y - ay;
            float dist = PLAN_ABS(px * ab_nx + py * ab_ny);
            if (dist > max_dist) { max_dist = dist; max_idx = i; }
        }
        if (max_idx >= 0 && max_dist > tolerance)
        {
            keep[max_idx] = 1;
            stack[stack_count++] = start;
            stack[stack_count++] = max_idx;
            stack[stack_count++] = max_idx;
            stack[stack_count++] = end;
        }
    }
    int out_count = 0;
    for (int i = 0; i < in_count; i++)
    {
        if (keep[i])
        {
            if (out_count < i)
                waypoints[out_count] = waypoints[i];
            out_count++;
        }
    }
    *count = out_count;
    free(keep);
    free(stack);
    return 0;
}

int plan_interpolate_path(const PlanWaypoint* waypoints, int count, PlanWaypoint* output, int* output_count, float step)
{
    if (!waypoints || !output || !output_count || count < 2 || step <= 0.0f) return -1;
    int out_idx = 0;
    for (int i = 0; i < count - 1 && out_idx < PLAN_PATH_MAX_WAYPOINTS; i++)
    {
        float dx = waypoints[i + 1].x - waypoints[i].x;
        float dy = waypoints[i + 1].y - waypoints[i].y;
        float seg_len = sqrtf(dx * dx + dy * dy);
        int seg_steps = (int)(seg_len / step);
        if (seg_steps < 1) seg_steps = 1;
        for (int j = 0; j < seg_steps && out_idx < PLAN_PATH_MAX_WAYPOINTS; j++)
        {
            float t = (float)j / (float)seg_steps;
            output[out_idx].x = waypoints[i].x + dx * t;
            output[out_idx].y = waypoints[i].y + dy * t;
            output[out_idx].z = waypoints[i].z + (waypoints[i + 1].z - waypoints[i].z) * t;
            output[out_idx].yaw = waypoints[i].yaw + (waypoints[i + 1].yaw - waypoints[i].yaw) * t;
            output[out_idx].is_valid = 1;
            out_idx++;
        }
    }
    if (out_idx < PLAN_PATH_MAX_WAYPOINTS)
    {
        output[out_idx] = waypoints[count - 1];
        out_idx++;
    }
    *output_count = out_idx;
    return 0;
}

/* ==================== 路径工具函数 ==================== */

float plan_path_length(const PlanWaypoint* waypoints, int count)
{
    if (!waypoints || count < 2) return 0.0f;
    float len = 0.0f;
    for (int i = 0; i < count - 1; i++)
        len += PLAN_DIST(waypoints[i].x, waypoints[i].y, waypoints[i + 1].x, waypoints[i + 1].y);
    return len;
}

int plan_check_path_valid(const PlanWaypoint* waypoints, int count, const PlanGridMap* map, float robot_radius)
{
    if (!waypoints || !map || count < 2) return 0;
    for (int i = 0; i < count; i++)
    {
        if (grid_is_occupied(map, waypoints[i].x, waypoints[i].y, robot_radius))
            return 0;
    }
    float step = map->resolution * 0.5f;
    for (int i = 0; i < count - 1; i++)
    {
        float dx = waypoints[i + 1].x - waypoints[i].x;
        float dy = waypoints[i + 1].y - waypoints[i].y;
        float seg_len = sqrtf(dx * dx + dy * dy);
        int steps = (int)(seg_len / step);
        for (int j = 1; j < steps; j++)
        {
            float t = (float)j / (float)steps;
            float px = waypoints[i].x + dx * t;
            float py = waypoints[i].y + dy * t;
            if (grid_is_occupied(map, px, py, robot_radius))
                return 0;
        }
    }
    return 1;
}

int plan_waypoint_to_command(const PlanWaypoint* waypoint, RobotCommand* command)
{
    if (!waypoint || !command) return -1;
    memset(command, 0, sizeof(RobotCommand));
    command->mode = MOTION_MODE_POSITION;
    command->target_position[0] = waypoint->x;
    command->target_position[1] = waypoint->y;
    command->target_position[2] = waypoint->z;
    command->target_orientation[0] = cosf(waypoint->yaw * 0.5f);
    command->target_orientation[1] = 0.0f;
    command->target_orientation[2] = 0.0f;
    command->target_orientation[3] = sinf(waypoint->yaw * 0.5f);
    command->max_velocity = 0.5f;
    command->max_acceleration = 0.5f;
    command->flags = ROBOT_CMD_FLAG_NONE;
    return 0;
}

int plan_path_to_trajectory(const PlanWaypoint* waypoints, int count, float speed, float* trajectory, int* trajectory_count)
{
    if (!waypoints || !trajectory || !trajectory_count || count < 2) return -1;
    float time = 0.0f;
    int tc = 0;
    for (int i = 0; i < count; i++)
    {
        if (tc >= PLAN_PATH_MAX_WAYPOINTS * 3 - 3) break;
        trajectory[tc++] = time;
        trajectory[tc++] = waypoints[i].x;
        trajectory[tc++] = waypoints[i].y;
        if (i < count - 1)
        {
            float seg_len = PLAN_DIST(waypoints[i].x, waypoints[i].y, waypoints[i + 1].x, waypoints[i + 1].y);
            time += (speed > 0.01f) ? seg_len / speed : seg_len;
        }
    }
    *trajectory_count = tc / 3;
    return 0;
}

typedef struct {
    float x, y;
    int parent;
    float cost;
} RRTStarNode;

static int rrt_star_nearest(const RRTStarNode* nodes, int count, float x, float y,
                             const float* headings, float* nearest_dist)
{
    int nearest = 0;
    float min_d = 1e20f;
    int i;
    if (headings) {
        float target_heading = (count > 0) ? plan_infer_heading(nodes[0].x, nodes[0].y, x, y) : 0.0f;
        for (i = 0; i < count; i++) {
            float d = plan_kinematic_distance(nodes[i].x, nodes[i].y, headings[i],
                x, y, target_heading, 1.0f, 1);
            if (d < min_d) { min_d = d; nearest = i; }
        }
    } else {
        for (i = 0; i < count; i++) {
            float d = PLAN_DIST(nodes[i].x, nodes[i].y, x, y);
            if (d < min_d) { min_d = d; nearest = i; }
        }
    }
    if (nearest_dist) *nearest_dist = min_d;
    return nearest;
}

static void rrt_star_steer(float sx, float sy, float gx, float gy, float step, float* ox, float* oy)
{
    float dx = gx - sx, dy = gy - sy;
    float d = sqrtf(dx * dx + dy * dy);
    if (d < 1e-6f) { *ox = sx; *oy = sy; return; }
    float s = PLAN_MIN(d, step);
    *ox = sx + dx / d * s;
    *oy = sy + dy / d * s;
}

static int rrt_star_collision_free(const PlanConfig* config, float x1, float y1, float x2, float y2)
{
    if (!config->user_data) return 1;
    const PlanGridMap* map = (const PlanGridMap*)config->user_data;
    float dx = x2 - x1, dy = y2 - y1;
    float len = sqrtf(dx * dx + dy * dy);
    int steps = (int)(len / (map->resolution * 0.5f)) + 1;
    int i;
    for (i = 0; i <= steps; i++) {
        float t = (float)i / (float)(steps > 0 ? steps : 1);
        float cx = x1 + dx * t, cy = y1 + dy * t;
        if (grid_is_occupied(map, cx, cy, config->robot_radius)) return 0;
    }
    return 1;
}

static float rrt_star_rewire_radius(int n, int dim)
{
    float gamma = 2.0f * (1.0f + 1.0f / (float)dim);
    float unit_ball_vol = (dim == 2) ? (float)M_PI : 4.0f * (float)M_PI / 3.0f;
    return PLAN_MIN(gamma * powf((float)n / unit_ball_vol, 1.0f / (float)dim), 5.0f);
}

int plan_rrt_star(const PlanConfig* config, PlanResult* result)
{
    RRTStarNode nodes[PLAN_RRT_STAR_MAX_NODES];
    float headings[PLAN_RRT_STAR_MAX_NODES];  /* S-010: 存储每个节点的朝向 */
    int node_count = 0;
    int i, near_count;
    float step = (config->rrt_star_config.rewire_radius > 0) ?
                  config->rrt_star_config.rewire_radius * 0.5f : PLAN_RRT_STEP_SIZE;
    float goal_tol = (config->rrt_star_config.goal_tolerance > 0) ?
                      config->rrt_star_config.goal_tolerance : 0.5f;
    int max_iter = (config->rrt_star_config.max_iterations > 0) ?
                    config->rrt_star_config.max_iterations : PLAN_RRT_STAR_MAX_ITER;
    int max_nodes = (config->rrt_star_config.max_nodes > 0) ?
                     config->rrt_star_config.max_nodes : PLAN_RRT_STAR_MAX_NODES;
    if (!config || !result) return -1;
    if (max_nodes > PLAN_RRT_STAR_MAX_NODES) max_nodes = PLAN_RRT_STAR_MAX_NODES;

    plan_slam_ensure_fresh((PlanConfig*)config);
    nodes[0].x = config->start.x;
    nodes[0].y = config->start.y;
    nodes[0].parent = -1;
    nodes[0].cost = 0.0f;
    headings[0] = config->start.yaw;  /* S-010: 初始朝向从配置读取 */
    node_count = 1;
    /* 是否启用运动学距离（有最小转弯半径时启用） */
    int use_kinematic = (config->min_turning_radius > 0.0f) ? 1 : 0;
    float* heading_ptr = use_kinematic ? headings : NULL;
    for (i = 0; i < max_iter && node_count < max_nodes; i++) {
        float rand_x, rand_y;
        float near_x, near_y;
        float new_x, new_y;
        int nearest_idx;
        float nearest_d;
        if (secure_random_float() < config->rrt_config.goal_bias) {
            rand_x = config->goal.x;
            rand_y = config->goal.y;
        } else {
            if (config->user_data) {
                const PlanGridMap* map = (const PlanGridMap*)config->user_data;
                rand_x = map->origin_x + secure_random_float() * map->width * map->resolution;
                rand_y = map->origin_y + secure_random_float() * map->height * map->resolution;
            } else {
                float range = 10.0f;
                rand_x = config->start.x - range + secure_random_float() * 2.0f * range;
                rand_y = config->start.y - range + secure_random_float() * 2.0f * range;
            }
        }
        nearest_idx = rrt_star_nearest(nodes, node_count, rand_x, rand_y,
                                       heading_ptr, &nearest_d);
        rrt_star_steer(nodes[nearest_idx].x, nodes[nearest_idx].y, rand_x, rand_y, step, &near_x, &near_y);
        if (nearest_d < step) { new_x = rand_x; new_y = rand_y; }
        else { new_x = near_x; new_y = near_y; }
        if (!rrt_star_collision_free(config, nodes[nearest_idx].x, nodes[nearest_idx].y, new_x, new_y)) continue;
        {
            /* S-010: 新节点朝向（从父节点方向推断） */
            float new_heading = plan_infer_heading(nodes[nearest_idx].x, nodes[nearest_idx].y, new_x, new_y);
            
            float radius = rrt_star_rewire_radius(node_count, 2);
            int near_indices[256];
            float near_dists[256];
            int best_parent = nearest_idx;
            /* S-010: 使用运动学距离计算代价（初始代价） */
            float kinematic_nearest_d = use_kinematic ?
                plan_kinematic_distance(nodes[nearest_idx].x, nodes[nearest_idx].y, headings[nearest_idx],
                    new_x, new_y, new_heading, config->min_turning_radius, 1) : nearest_d;
            float best_cost = nodes[nearest_idx].cost + kinematic_nearest_d;
            near_count = 0;
            {
                int j;
                for (j = 0; j < node_count && near_count < 256; j++) {
                    /* 空间邻近搜索仍使用欧氏距离（保证搜索半径的物理意义） */
                    float d = PLAN_DIST(nodes[j].x, nodes[j].y, new_x, new_y);
                    if (d < radius) {
                        near_indices[near_count] = j;
                        near_dists[near_count] = use_kinematic ?
                            plan_kinematic_distance(nodes[j].x, nodes[j].y, headings[j],
                                new_x, new_y, new_heading, config->min_turning_radius, 1) : d;
                        near_count++;
                    }
                }
            }
            {
                int j;
                for (j = 0; j < near_count; j++) {
                    int nidx = near_indices[j];
                    float d = near_dists[j];
                    float candidate_cost = nodes[nidx].cost + d;
                    if (candidate_cost < best_cost &&
                        rrt_star_collision_free(config, nodes[nidx].x, nodes[nidx].y, new_x, new_y)) {
                        best_parent = nidx;
                        best_cost = candidate_cost;
                        /* 更新朝向以匹配新的父节点 */
                        new_heading = plan_infer_heading(nodes[nidx].x, nodes[nidx].y, new_x, new_y);
                    }
                }
            }
            nodes[node_count].x = new_x;
            nodes[node_count].y = new_y;
            nodes[node_count].parent = best_parent;
            nodes[node_count].cost = best_cost;
            headings[node_count] = new_heading;
            node_count++;
            {
                int j;
                for (j = 0; j < near_count; j++) {
                    int nidx = near_indices[j];
                    float d = near_dists[j];
                    float new_cost = nodes[node_count - 1].cost + d;
                    if (new_cost < nodes[nidx].cost &&
                        rrt_star_collision_free(config, new_x, new_y, nodes[nidx].x, nodes[nidx].y)) {
                        nodes[nidx].parent = node_count - 1;
                        nodes[nidx].cost = new_cost;
                    }
                }
            }
        }
        /* S-010修复: 使用运动学距离检查是否到达目标 */
        {
            float goal_dist = use_kinematic ?
                plan_kinematic_distance(new_x, new_y, headings[node_count - 1],
                    config->goal.x, config->goal.y, config->goal.yaw,
                    config->min_turning_radius, 1) :
                PLAN_DIST(new_x, new_y, config->goal.x, config->goal.y);
            if (goal_dist < goal_tol) break;
        }
    }
    {
        int best_goal = -1;
        float best_goal_cost = 1e20f;
        int j;
        for (j = 0; j < node_count; j++) {
            /* S-010修复: 使用运动学距离评估最佳目标候选 */
            float d = use_kinematic ?
                plan_kinematic_distance(nodes[j].x, nodes[j].y, headings[j],
                    config->goal.x, config->goal.y, config->goal.yaw,
                    config->min_turning_radius, 1) :
                PLAN_DIST(nodes[j].x, nodes[j].y, config->goal.x, config->goal.y);
            if (d < goal_tol) {
                float total = nodes[j].cost + d;
                if (total < best_goal_cost) {
                    best_goal_cost = total;
                    best_goal = j;
                }
            }
        }
        if (best_goal < 0) {
            result->waypoint_count = 0;
            result->success = 0;
            return 0;
        }
        {
            int path_nodes[PLAN_PATH_MAX_WAYPOINTS];
            int path_len = 0;
            int cur = best_goal;
            while (cur >= 0 && path_len < PLAN_PATH_MAX_WAYPOINTS) {
                path_nodes[path_len++] = cur;
                cur = nodes[cur].parent;
            }
            result->waypoint_count = path_len;
            result->total_length = 0.0f;
            result->total_cost = best_goal_cost;
            {
                int k;
                for (k = 0; k < path_len; k++) {
                    int idx = path_nodes[path_len - 1 - k];
                    result->waypoints[k].x = nodes[idx].x;
                    result->waypoints[k].y = nodes[idx].y;
                    result->waypoints[k].z = 0.0f;
                    result->waypoints[k].yaw = 0.0f;
                    result->waypoints[k].is_valid = 1;
                    if (k > 0) result->total_length += PLAN_DIST(
                        result->waypoints[k-1].x, result->waypoints[k-1].y,
                        result->waypoints[k].x, result->waypoints[k].y);
                }
            }
            result->success = 1;
        }
    }
    return 0;
}

typedef struct {
    float x, y;
    int edges[PLAN_PRM_MAX_NEIGHBORS];
    float edge_costs[PLAN_PRM_MAX_NEIGHBORS];
    int edge_count;
} PRMNode;

static int prm_sample_collision_free(const PlanConfig* config, float x, float y)
{
    if (!config->user_data) return 1;
    const PlanGridMap* map = (const PlanGridMap*)config->user_data;
    if (grid_is_occupied(map, x, y, config->robot_radius)) return 0;
    return 1;
}

int plan_prm_star(const PlanConfig* config, PlanResult* result)
{
    PRMNode nodes[PLAN_PRM_MAX_NODES];
    int node_count = 0;
    int max_nodes, max_neighbors;
    float connect_radius;
    float g_score[PLAN_PRM_MAX_NODES];
    int visited[PLAN_PRM_MAX_NODES];
    int parent[PLAN_PRM_MAX_NODES];
    int i, j;
    if (!config || !result) return -1;
    max_nodes = (config->prm_config.max_nodes > 0) ? config->prm_config.max_nodes : PLAN_PRM_MAX_NODES;
    max_neighbors = (config->prm_config.max_neighbors > 0) ? config->prm_config.max_neighbors : PLAN_PRM_MAX_NEIGHBORS;
    connect_radius = (config->prm_config.connect_radius > 0) ? config->prm_config.connect_radius : PLAN_PRM_CONNECT_RADIUS;
    if (max_nodes > PLAN_PRM_MAX_NODES) max_nodes = PLAN_PRM_MAX_NODES;
    if (max_neighbors > PLAN_PRM_MAX_NEIGHBORS) max_neighbors = PLAN_PRM_MAX_NEIGHBORS;
    nodes[0].x = config->start.x;
    nodes[0].y = config->start.y;
    nodes[0].edge_count = 0;
    node_count = 1;
    nodes[1].x = config->goal.x;
    nodes[1].y = config->goal.y;
    nodes[1].edge_count = 0;
    node_count = 2;
    {
        int attempts = 0;
        while (node_count < max_nodes && attempts < max_nodes * 10) {
            float x, y;
            attempts++;
            if (config->user_data) {
                const PlanGridMap* map = (const PlanGridMap*)config->user_data;
                x = map->origin_x + secure_random_float() * map->width * map->resolution;
                y = map->origin_y + secure_random_float() * map->height * map->resolution;
            } else {
                float range = 10.0f;
                x = config->start.x - range + secure_random_float() * 2.0f * range;
                y = config->start.y - range + secure_random_float() * 2.0f * range;
            }
            if (!prm_sample_collision_free(config, x, y)) continue;
            {
                int too_close = 0;
                int k;
                for (k = 0; k < node_count; k++) {
                    if (PLAN_DIST(nodes[k].x, nodes[k].y, x, y) < 0.1f) { too_close = 1; break; }
                }
                if (too_close) continue;
            }
            nodes[node_count].x = x;
            nodes[node_count].y = y;
            nodes[node_count].edge_count = 0;
            node_count++;
        }
    }
    for (i = 0; i < node_count; i++) {
        float dists[PLAN_PRM_MAX_NEIGHBORS];
        int indices[PLAN_PRM_MAX_NEIGHBORS];
        int near_count = 0;
        for (j = 0; j < node_count; j++) {
            if (i == j) continue;
            float d = PLAN_DIST(nodes[i].x, nodes[i].y, nodes[j].x, nodes[j].y);
            if (d < connect_radius && near_count < max_neighbors) {
                int insert_pos = near_count;
                int k;
                for (k = near_count - 1; k >= 0; k--) {
                    if (dists[k] < d) break;
                    dists[k + 1] = dists[k];
                    indices[k + 1] = indices[k];
                    insert_pos = k;
                }
                dists[insert_pos] = d;
                indices[insert_pos] = j;
                near_count++;
            }
        }
        nodes[i].edge_count = near_count;
        for (j = 0; j < near_count; j++) {
            int ni = indices[j];
            if (rrt_star_collision_free(config, nodes[i].x, nodes[i].y, nodes[ni].x, nodes[ni].y)) {
                nodes[i].edges[j] = ni;
                nodes[i].edge_costs[j] = dists[j];
            } else {
                nodes[i].edges[j] = -1;
                nodes[i].edge_costs[j] = 1e10f;
            }
        }
    }
    for (i = 0; i < node_count; i++) {
        g_score[i] = 1e20f;
        visited[i] = 0;
        parent[i] = -1;
    }
    g_score[0] = 0.0f;
    for (i = 0; i < node_count; i++) {
        int u = -1;
        float best = 1e20f;
        int k;
        for (k = 0; k < node_count; k++) {
            if (!visited[k] && g_score[k] < best) { best = g_score[k]; u = k; }
        }
        if (u < 0 || u == 1) break;
        visited[u] = 1;
        for (k = 0; k < nodes[u].edge_count; k++) {
            int v = nodes[u].edges[k];
            if (v < 0) continue;
            float alt = g_score[u] + nodes[u].edge_costs[k];
            if (alt < g_score[v]) {
                g_score[v] = alt;
                parent[v] = u;
            }
        }
    }
    if (g_score[1] > 1e19f) {
        result->waypoint_count = 0;
        result->success = 0;
        return 0;
    }
    {
        int path[PLAN_PATH_MAX_WAYPOINTS];
        int path_len = 0;
        int cur = 1;
        while (cur >= 0 && path_len < PLAN_PATH_MAX_WAYPOINTS) {
            path[path_len++] = cur;
            cur = parent[cur];
        }
        result->waypoint_count = path_len;
        result->total_length = g_score[1];
        result->total_cost = g_score[1];
        for (i = 0; i < path_len; i++) {
            int idx = path[path_len - 1 - i];
            result->waypoints[i].x = nodes[idx].x;
            result->waypoints[i].y = nodes[idx].y;
            result->waypoints[i].z = 0.0f;
            result->waypoints[i].yaw = 0.0f;
            result->waypoints[i].is_valid = 1;
        }
        result->success = 1;
    }
    return 0;
}

typedef struct {
    float g, rhs;
    float k1, k2;
} DStarCell;

static void dstar_calc_key(DStarCell* cell, float sx, float sy, float cx, float cy)
{
    float h = PLAN_DIST(sx, sy, cx, cy);
    cell->k1 = PLAN_MIN(cell->g, cell->rhs) + h;
    cell->k2 = PLAN_MIN(cell->g, cell->rhs);
}

static int dstar_key_less(float k1a, float k2a, float k1b, float k2b)
{
    if (k1a < k1b - 1e-6f) return 1;
    if (k1a > k1b + 1e-6f) return 0;
    return k2a < k2b - 1e-6f;
}

int plan_dstar_lite(const PlanConfig* config, PlanResult* result)
{
    DStarCell cells[PLAN_DSTAR_MAX_CELLS];
    int queue_priority[PLAN_DSTAR_MAX_CELLS];
    float queue_k1[PLAN_DSTAR_MAX_CELLS];
    float queue_k2[PLAN_DSTAR_MAX_CELLS];
    int queue_size = 0;
    int grid_w, grid_h;
    float res;
    int start_id, goal_id;

    plan_slam_ensure_fresh((PlanConfig*)config);
    int i;
    if (!config || !result) return -1;
    if (!config->user_data) return -1;
    {
        const PlanGridMap* map = (const PlanGridMap*)config->user_data;
        grid_w = map->width;
        grid_h = map->height;
        res = map->resolution;
        if (grid_w * grid_h > PLAN_DSTAR_MAX_CELLS) return -1;
        int gx = (int)((config->goal.x - map->origin_x) / res);
        int gy = (int)((config->goal.y - map->origin_y) / res);
        int sx = (int)((config->start.x - map->origin_x) / res);
        int sy = (int)((config->start.y - map->origin_y) / res);
        if (gx < 0 || gx >= grid_w || gy < 0 || gy >= grid_h) return -1;
        if (sx < 0 || sx >= grid_w || sy < 0 || sy >= grid_h) return -1;
        goal_id = gy * grid_w + gx;
        start_id = sy * grid_w + sx;
    }
    for (i = 0; i < grid_w * grid_h; i++) {
        cells[i].g = 1e20f;
        cells[i].rhs = 1e20f;
        cells[i].k1 = 0;
        cells[i].k2 = 0;
    }
    cells[goal_id].rhs = 0.0f;
    dstar_calc_key(&cells[goal_id], config->start.x, config->start.y,
                   config->goal.x, config->goal.y);
    queue_priority[0] = goal_id;
    queue_k1[0] = cells[goal_id].k1;
    queue_k2[0] = cells[goal_id].k2;
    queue_size = 1;
    {
        int neighbor_offsets[8] = {-grid_w - 1, -grid_w, -grid_w + 1, -1, 1, grid_w - 1, grid_w, grid_w + 1};
        int iter_count = 0;
        while (queue_size > 0 && iter_count < 100000) {
            int min_idx = 0;
            float min_k1 = queue_k1[0], min_k2 = queue_k2[0];
            int q;
            iter_count++;
            for (q = 1; q < queue_size; q++) {
                if (dstar_key_less(queue_k1[q], queue_k2[q], min_k1, min_k2)) {
                    min_k1 = queue_k1[q]; min_k2 = queue_k2[q]; min_idx = q;
                }
            }
            int u = queue_priority[min_idx];
            for (q = min_idx; q < queue_size - 1; q++) {
                queue_priority[q] = queue_priority[q + 1];
                queue_k1[q] = queue_k1[q + 1];
                queue_k2[q] = queue_k2[q + 1];
            }
            queue_size--;
            if (dstar_key_less(cells[start_id].k1, cells[start_id].k2, min_k1, min_k2) &&
                cells[start_id].rhs == cells[start_id].g) {
                break;
            }
            dstar_calc_key(&cells[u], config->start.x, config->start.y,
                           config->goal.x + (u % grid_w - goal_id % grid_w) * res,
                           config->goal.y + (u / grid_w - goal_id / grid_w) * res);
            if (cells[u].k1 < min_k1 - 1e-6f || (cells[u].k1 < min_k1 + 1e-6f && cells[u].k2 < min_k2 - 1e-6f)) {
                queue_priority[queue_size] = u;
                queue_k1[queue_size] = cells[u].k1;
                queue_k2[queue_size] = cells[u].k2;
                queue_size++;
                continue;
            }
            if (cells[u].g > cells[u].rhs) {
                cells[u].g = cells[u].rhs;
            } else {
                cells[u].g = 1e20f;
                cells[u].rhs = 1e20f;
                {
                    int k;
                    for (k = 0; k < 8; k++) {
                        int nb = u + neighbor_offsets[k];
                        int ux = u % grid_w, uy = u / grid_w;
                        int nx = nb % grid_w, ny = nb / grid_w;
                        if (nx < 0 || nx >= grid_w || ny < 0 || ny >= grid_h) continue;
                        float dx = (float)(nx - ux) * res;
                        float dy = (float)(ny - uy) * res;
                        float cost = sqrtf(dx * dx + dy * dy);
                        const PlanGridMap* map = (const PlanGridMap*)config->user_data;
                        if (grid_is_occupied(map, map->origin_x + nx * res, map->origin_y + ny * res, config->robot_radius)) {
                            cost *= 100.0f;
                        }
                        float alt = cells[nb].g + cost;
                        if (alt < cells[u].rhs) cells[u].rhs = alt;
                    }
                }
            }
            if (cells[u].rhs != cells[u].g) {
                dstar_calc_key(&cells[u], config->start.x, config->start.y,
                               config->goal.x + (u % grid_w - goal_id % grid_w) * res,
                               config->goal.y + (u / grid_w - goal_id / grid_w) * res);
                queue_priority[queue_size] = u;
                queue_k1[queue_size] = cells[u].k1;
                queue_k2[queue_size] = cells[u].k2;
                queue_size++;
            }
            {
                int k;
                for (k = 0; k < 8; k++) {
                    int nb = u + neighbor_offsets[k];
                    int nx = nb % grid_w, ny = nb / grid_w;
                    if (nx < 0 || nx >= grid_w || ny < 0 || ny >= grid_h) continue;
                    cells[nb].rhs = 1e20f;
                    {
                        int m;
                        int nb_offsets[8] = {-grid_w - 1, -grid_w, -grid_w + 1, -1, 1, grid_w - 1, grid_w, grid_w + 1};
                        for (m = 0; m < 8; m++) {
                            int nb2 = nb + nb_offsets[m];
                            int nbx = nb2 % grid_w, nby = nb2 / grid_w;
                            if (nbx < 0 || nbx >= grid_w || nby < 0 || nby >= grid_h) continue;
                            float dx2 = (float)(nbx - nx) * res;
                            float dy2 = (float)(nby - ny) * res;
                            float cost2 = sqrtf(dx2 * dx2 + dy2 * dy2);
                            const PlanGridMap* map = (const PlanGridMap*)config->user_data;
                            if (grid_is_occupied(map, map->origin_x + nbx * res, map->origin_y + nby * res, config->robot_radius)) {
                                cost2 *= 100.0f;
                            }
                            float alt = cells[nb2].g + cost2;
                            if (alt < cells[nb].rhs) cells[nb].rhs = alt;
                        }
                    }
                    if (cells[nb].rhs != cells[nb].g) {
                        dstar_calc_key(&cells[nb], config->start.x, config->start.y,
                                       config->goal.x + (nx - goal_id % grid_w) * res,
                                       config->goal.y + (ny - goal_id / grid_w) * res);
                        {
                            int already = 0;
                            int qi;
                            for (qi = 0; qi < queue_size; qi++) {
                                if (queue_priority[qi] == nb) { already = 1; break; }
                            }
                            if (!already) {
                                queue_priority[queue_size] = nb;
                                queue_k1[queue_size] = cells[nb].k1;
                                queue_k2[queue_size] = cells[nb].k2;
                                queue_size++;
                            }
                        }
                    }
                }
            }
        }
    }
    if (cells[start_id].g > 1e19f) {
        result->waypoint_count = 0;
        result->success = 0;
        return 0;
    }
    {
        const PlanGridMap* map = (const PlanGridMap*)config->user_data;
        int path_len = 0;
        int cur = start_id;
        int visited_dstar[PLAN_DSTAR_MAX_CELLS];
        memset(visited_dstar, 0, sizeof(int) * grid_w * grid_h);
        while (cur != goal_id && path_len < PLAN_PATH_MAX_WAYPOINTS && !visited_dstar[cur]) {
            visited_dstar[cur] = 1;
            int cx = cur % grid_w, cy = cur / grid_w;
            result->waypoints[path_len].x = map->origin_x + cx * res;
            result->waypoints[path_len].y = map->origin_y + cy * res;
            result->waypoints[path_len].z = 0.0f;
            result->waypoints[path_len].yaw = 0.0f;
            result->waypoints[path_len].is_valid = 1;
            result->waypoints[path_len].cost = cells[cur].g;
            path_len++;
            int best_next = cur;
            float best_g = cells[cur].g;
            int neighbor_offsets[8] = {-grid_w - 1, -grid_w, -grid_w + 1, -1, 1, grid_w - 1, grid_w, grid_w + 1};
            int k;
            for (k = 0; k < 8; k++) {
                int nb = cur + neighbor_offsets[k];
                int nx = nb % grid_w, ny = nb / grid_w;
                if (nx < 0 || nx >= grid_w || ny < 0 || ny >= grid_h) continue;
                float dx = (float)(nx - cx) * res;
                float dy = (float)(ny - cy) * res;
                float cost = sqrtf(dx * dx + dy * dy);
                if (cells[nb].g + cost < best_g) {
                    best_g = cells[nb].g + cost;
                    best_next = nb;
                }
            }
            cur = best_next;
        }
        if (cur == goal_id && path_len < PLAN_PATH_MAX_WAYPOINTS) {
            result->waypoints[path_len].x = config->goal.x;
            result->waypoints[path_len].y = config->goal.y;
            result->waypoints[path_len].z = 0;
            result->waypoints[path_len].yaw = 0;
            result->waypoints[path_len].is_valid = 1;
            path_len++;
        }
        result->waypoint_count = path_len;
        result->total_length = cells[start_id].g;
        result->total_cost = cells[start_id].g;
        result->success = 1;
    }
    return 0;
}

int plan_chomp(const PlanConfig* config, PlanResult* result)
{
    float trajectory[PLAN_CHOMP_MAX_POINTS * 2];
    int num_points;
    float smooth_weight, obs_weight, eta;
    int max_iter;
    int i, iter;
    if (!config || !result) return -1;
    num_points = (config->chomp_config.num_points > 0) ? config->chomp_config.num_points : 32;
    smooth_weight = (config->chomp_config.smoothness_cost_weight > 0) ?
                     config->chomp_config.smoothness_cost_weight : PLAN_CHOMP_LAMBDA_SMOOTH;
    obs_weight = (config->chomp_config.obstacle_cost_weight > 0) ?
                  config->chomp_config.obstacle_cost_weight : PLAN_CHOMP_LAMBDA_OBS;
    eta = (config->chomp_config.learning_rate > 0) ?
           config->chomp_config.learning_rate : PLAN_CHOMP_ETA;
    max_iter = (config->chomp_config.max_iterations > 0) ?
                config->chomp_config.max_iterations : PLAN_CHOMP_MAX_ITER;
    if (num_points > PLAN_CHOMP_MAX_POINTS) num_points = PLAN_CHOMP_MAX_POINTS;
    for (i = 0; i < num_points; i++) {
        float t = (float)i / (float)(num_points - 1);
        trajectory[i * 2 + 0] = config->start.x + (config->goal.x - config->start.x) * t;
        trajectory[i * 2 + 1] = config->start.y + (config->goal.y - config->start.y) * t;
    }
    {
        float A_inv[PLAN_CHOMP_MAX_POINTS * PLAN_CHOMP_MAX_POINTS];
        memset(A_inv, 0, sizeof(A_inv));
        for (i = 1; i < num_points - 1; i++) {
            float diag = 2.0f / (smooth_weight + 1e-6f);
            A_inv[i * num_points + i] = diag;
            if (i > 0) A_inv[i * num_points + (i - 1)] = -1.0f / (smooth_weight + 1e-6f);
            if (i < num_points - 1) A_inv[i * num_points + (i + 1)] = -1.0f / (smooth_weight + 1e-6f);
        }
        A_inv[0 * num_points + 0] = 1.0f;
        A_inv[(num_points - 1) * num_points + (num_points - 1)] = 1.0f;
        for (iter = 0; iter < max_iter; iter++) {
            float grad_smooth[PLAN_CHOMP_MAX_POINTS * 2];
            float grad_obs[PLAN_CHOMP_MAX_POINTS * 2];
            float grad_total[PLAN_CHOMP_MAX_POINTS * 2];
            float update[PLAN_CHOMP_MAX_POINTS * 2];
            memset(grad_smooth, 0, sizeof(grad_smooth));
            for (i = 1; i < num_points - 1; i++) {
                grad_smooth[i * 2 + 0] = 2.0f * trajectory[i * 2 + 0] -
                    trajectory[(i - 1) * 2 + 0] - trajectory[(i + 1) * 2 + 0];
                grad_smooth[i * 2 + 1] = 2.0f * trajectory[i * 2 + 1] -
                    trajectory[(i - 1) * 2 + 1] - trajectory[(i + 1) * 2 + 1];
            }
            memset(grad_obs, 0, sizeof(grad_obs));
            if (config->user_data) {
                const PlanGridMap* map = (const PlanGridMap*)config->user_data;
                for (i = 1; i < num_points - 1; i++) {
                    float px = trajectory[i * 2 + 0];
                    float py = trajectory[i * 2 + 1];
                    int gx = (int)((px - map->origin_x) / map->resolution);
                    int gy = (int)((py - map->origin_y) / map->resolution);
                    if (gx < 1 || gx >= map->width - 1 || gy < 1 || gy >= map->height - 1) continue;
                    int idx = gy * map->width + gx;
                    float occ = map->cells[idx] / 255.0f;
                    float margin = config->chomp_config.obstacle_margin > 0 ?
                                    config->chomp_config.obstacle_margin : config->obstacle_margin;
                    float dist = PLAN_MAX(0, occ * 10.0f - margin);
                    float c_obs = expf(-dist * dist / 0.1f);
                    float left = map->cells[gy * map->width + (gx - 1)] / 255.0f;
                    float right = map->cells[gy * map->width + (gx + 1)] / 255.0f;
                    float down = map->cells[(gy - 1) * map->width + gx] / 255.0f;
                    float up = map->cells[(gy + 1) * map->width + gx] / 255.0f;
                    float dx = (right - left) / (2.0f * map->resolution);
                    float dy = (up - down) / (2.0f * map->resolution);
                    grad_obs[i * 2 + 0] = c_obs * dx;
                    grad_obs[i * 2 + 1] = c_obs * dy;
                }
            }
            for (i = 0; i < num_points * 2; i++) {
                grad_total[i] = grad_smooth[i] + obs_weight * grad_obs[i];
            }
            memset(update, 0, sizeof(float) * num_points * 2);
            for (i = 1; i < num_points - 1; i++) {
                int j;
                for (j = 1; j < num_points - 1; j++) {
                    update[i * 2 + 0] += A_inv[i * num_points + j] * grad_total[j * 2 + 0];
                    update[i * 2 + 1] += A_inv[i * num_points + j] * grad_total[j * 2 + 1];
                }
            }
            float max_update = 0.0f;
            for (i = 0; i < num_points * 2; i++) {
                if (update[i] * update[i] > max_update) max_update = update[i] * update[i];
            }
            max_update = sqrtf(max_update);
            float step = (max_update > 0.01f) ? eta / max_update : eta;
            for (i = 1; i < num_points - 1; i++) {
                trajectory[i * 2 + 0] -= step * (grad_smooth[i * 2 + 0] + obs_weight * grad_obs[i * 2 + 0]);
                trajectory[i * 2 + 1] -= step * (grad_smooth[i * 2 + 1] + obs_weight * grad_obs[i * 2 + 1]);
            }
            trajectory[0] = config->start.x;
            trajectory[1] = config->start.y;
            trajectory[(num_points - 1) * 2 + 0] = config->goal.x;
            trajectory[(num_points - 1) * 2 + 1] = config->goal.y;
        }
    }
    result->waypoint_count = num_points;
    result->total_length = 0.0f;
    result->total_cost = 0.0f;
    for (i = 0; i < num_points; i++) {
        result->waypoints[i].x = trajectory[i * 2 + 0];
        result->waypoints[i].y = trajectory[i * 2 + 1];
        result->waypoints[i].z = 0.0f;
        result->waypoints[i].yaw = 0.0f;
        result->waypoints[i].is_valid = 1;
        if (i > 0) result->total_length += PLAN_DIST(
            result->waypoints[i-1].x, result->waypoints[i-1].y,
            result->waypoints[i].x, result->waypoints[i].y);
    }
    result->success = 1;
    return 0;
}

static slam_map_update_fn g_slam_map_callback = NULL;
static void* g_slam_map_ctx = NULL;
static uint64_t g_last_slam_update_ms = 0;

int plan_register_slam_callback(slam_map_update_fn callback, void* ctx) {
    g_slam_map_callback = callback;
    g_slam_map_ctx = ctx;
    g_last_slam_update_ms = 0;
    return 0;
}

int plan_refresh_map_from_slam(PlanConfig* config) {
    if (!config || !g_slam_map_callback) return 0;
    PlanGridMap* map = (PlanGridMap*)config->user_data;
    if (!map || !map->cells) return 0;
    int result = g_slam_map_callback(map, g_slam_map_ctx);
    if (result > 0) {
        g_last_slam_update_ms = (uint64_t)((float)clock() / (float)CLOCKS_PER_SEC * 1000.0f);
    }
    return result;
}

void plan_slam_ensure_fresh(PlanConfig* config) {
    if (!config || !g_slam_map_callback) return;
    uint64_t now_ms = (uint64_t)((float)clock() / (float)CLOCKS_PER_SEC * 1000.0f);
    if (now_ms - g_last_slam_update_ms > 500) {
        plan_refresh_map_from_slam(config);
    }
}

int plan_stomp(const PlanConfig* config, PlanResult* result)
{
    float trajectory[PLAN_STOMP_MAX_POINTS * 2];
    int num_points, num_noisy, max_iter;
    float noise_std, exploration;
    int i, iter;
    if (!config || !result) return -1;
    num_points = (config->stomp_config.num_points > 0) ? config->stomp_config.num_points : 32;
    num_noisy = (config->stomp_config.num_noisy > 0) ? config->stomp_config.num_noisy : PLAN_STOMP_NUM_NOISY;
    noise_std = (config->stomp_config.noise_std > 0) ? config->stomp_config.noise_std : PLAN_STOMP_NOISE_STD;
    exploration = (config->stomp_config.exploration > 0) ? config->stomp_config.exploration : PLAN_STOMP_EXPLORATION;
    max_iter = (config->stomp_config.max_iterations > 0) ? config->stomp_config.max_iterations : PLAN_STOMP_MAX_ITER;
    if (num_points > PLAN_STOMP_MAX_POINTS) num_points = PLAN_STOMP_MAX_POINTS;
    if (num_noisy > 100) num_noisy = 100;
    for (i = 0; i < num_points; i++) {
        float t = (float)i / (float)(num_points - 1);
        trajectory[i * 2 + 0] = config->start.x + (config->goal.x - config->start.x) * t;
        trajectory[i * 2 + 1] = config->start.y + (config->goal.y - config->start.y) * t;
    }
    for (iter = 0; iter < max_iter; iter++) {
        float noisy_trajs[100][PLAN_STOMP_MAX_POINTS * 2];
        float costs[100];
        float min_cost = 1e20f;
        float max_cost = -1e20f;
        int n;
        for (n = 0; n < num_noisy; n++) {
            int j;
            float cost = 0.0f;
            float prev_x = trajectory[0], prev_y = trajectory[1];
            for (j = 0; j < num_points; j++) {
                float gx, gy;
                if (j == 0 || j == num_points - 1) {
                    gx = trajectory[j * 2 + 0];
                    gy = trajectory[j * 2 + 1];
                } else {
                    gx = trajectory[j * 2 + 0] + secure_random_float() * 2.0f * noise_std - noise_std;
                    gy = trajectory[j * 2 + 1] + secure_random_float() * 2.0f * noise_std - noise_std;
                }
                noisy_trajs[n][j * 2 + 0] = gx;
                noisy_trajs[n][j * 2 + 1] = gy;
                if (j > 0) {
                    cost += PLAN_DIST(prev_x, prev_y, gx, gy);
                }
                prev_x = gx; prev_y = gy;
                if (config->user_data) {
                    const PlanGridMap* map = (const PlanGridMap*)config->user_data;
                    if (grid_is_occupied(map, gx, gy, config->robot_radius)) {
                        cost += 100.0f;
                    }
                }
            }
            costs[n] = cost;
            if (cost < min_cost) min_cost = cost;
            if (cost > max_cost) max_cost = cost;
        }
        float cost_range = max_cost - min_cost;
        if (cost_range < 1e-6f) cost_range = 1e-6f;
        {
            float weights[100];
            float weight_sum = 0.0f;
            for (n = 0; n < num_noisy; n++) {
                float normalized = (max_cost - costs[n]) / cost_range;
                weights[n] = expf(normalized / exploration);
                weight_sum += weights[n];
            }
            if (weight_sum > 1e-10f) {
                int j;
                for (j = 0; j < num_points * 2; j++) {
                    float weighted_sum = 0.0f;
                    int ni;
                    for (ni = 0; ni < num_noisy; ni++) {
                        weighted_sum += weights[ni] * noisy_trajs[ni][j];
                    }
                    trajectory[j] = weighted_sum / weight_sum;
                }
            }
        }
        trajectory[0] = config->start.x;
        trajectory[1] = config->start.y;
        trajectory[(num_points - 1) * 2 + 0] = config->goal.x;
        trajectory[(num_points - 1) * 2 + 1] = config->goal.y;
    }
    result->waypoint_count = num_points;
    result->total_length = 0.0f;
    result->total_cost = 0.0f;
    for (i = 0; i < num_points; i++) {
        result->waypoints[i].x = trajectory[i * 2 + 0];
        result->waypoints[i].y = trajectory[i * 2 + 1];
        result->waypoints[i].z = 0.0f;
        result->waypoints[i].yaw = 0.0f;
        result->waypoints[i].is_valid = 1;
        if (i > 0) result->total_length += PLAN_DIST(
            result->waypoints[i-1].x, result->waypoints[i-1].y,
            result->waypoints[i].x, result->waypoints[i].y);
    }
    result->success = 1;
    return 0;
}
