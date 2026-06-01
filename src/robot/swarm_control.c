/**
 * @file swarm_control.c
 * @brief 多机器人集群控制实现
 *
 * 实现蜂拥算法、编队控制、一致性协议和任务分配。
 * 纯C实现，浮点运算，无第三方库依赖。
 */

#include "selflnn/robot/swarm_control.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/logging.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET swarm_socket_t;
#define SWARM_INVALID_SOCKET INVALID_SOCKET
#define swarm_socket_close closesocket
#define SWARM_SOCKET_ERROR SOCKET_ERROR
#else
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <fcntl.h>
#include <errno.h>
typedef int swarm_socket_t;
#define SWARM_INVALID_SOCKET (-1)
#define swarm_socket_close close
#define SWARM_SOCKET_ERROR (-1)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define SWARM_MIN(a,b) (((a)<(b))?(a):(b))
#define SWARM_MAX(a,b) (((a)>(b))?(a):(b))
#define SWARM_CLAMP(v,lo,hi) (((v)<(lo))?(lo):(((v)>(hi))?(hi):(v)))
#define SWARM_ABS(x) (((x)>=0)?(x):-(x))
#define SWARM_SQ(x) ((x)*(x))
#define SWARM_MAX_TASKS_PER_ROBOT 8
#define SWARM_CBBA_MAX_BUNDLE 16
#define HUNGARIAN_MAX 256
#define HUNGARIAN_INF 1e18f

typedef struct SwarmController
{
    SwarmConfig config;
    SwarmState state;
    SwarmTask tasks[SWARM_MAX_TASKS];
    int task_count;
    int allocated_tasks;
    int algorithms_enabled[5];
    float* consensus_state;
    int consensus_state_dim;
    int is_initialized;
    float vs_position[3];
    float vs_velocity[3];
    float vs_orientation[4];
    float vs_angular_velocity[3];
    MutexHandle mutex;

    /* UDP组播网络同步（多机群集通信） */
    swarm_socket_t udp_socket;
    struct sockaddr_in multicast_addr;
    struct sockaddr_in unicast_addrs[SWARM_MAX_ROBOTS];
    int unicast_count;
    int udp_port_base;
    int udp_initialized;
    char network_buffer[8192];
} SwarmController;

static void swarm_lock(SwarmController* controller)
{
    if (controller && controller->mutex) mutex_lock(controller->mutex);
}

static void swarm_unlock(SwarmController* controller)
{
    if (controller && controller->mutex) mutex_unlock(controller->mutex);
}

static int swarm_find_robot_index(const SwarmController* controller, int robot_id)
{
    for (int i = 0; i < controller->state.robot_count; i++)
    {
        if (controller->state.robots[i].robot_id == robot_id)
            return i;
    }
    return -1;
}

static void swarm_update_neighbors(SwarmController* controller)
{
    SwarmState* state = &controller->state;
    float range_sq = controller->config.flocking_params.neighbor_radius;
    range_sq *= range_sq;

    for (int i = 0; i < state->robot_count; i++)
    {
        state->neighbor_counts[i] = 0;
        if (!state->robots[i].is_active) continue;

        for (int j = 0; j < state->robot_count; j++)
        {
            if (i == j || !state->robots[j].is_active) continue;
            float dx = state->robots[j].position[0] - state->robots[i].position[0];
            float dy = state->robots[j].position[1] - state->robots[i].position[1];
            float dz = state->robots[j].position[2] - state->robots[i].position[2];
            float dsq = dx * dx + dy * dy + dz * dz;
            if (dsq <= range_sq && state->neighbor_counts[i] < SWARM_MAX_NEIGHBORS)
            {
                int n = state->neighbor_counts[i]++;
                state->neighbors[i][n].robot_id = state->robots[j].robot_id;
                state->neighbors[i][n].distance = sqrtf(dsq);
                state->neighbors[i][n].relative_position[0] = dx;
                state->neighbors[i][n].relative_position[1] = dy;
                state->neighbors[i][n].relative_position[2] = dz;
                state->neighbors[i][n].relative_velocity[0] = state->robots[j].velocity[0] - state->robots[i].velocity[0];
                state->neighbors[i][n].relative_velocity[1] = state->robots[j].velocity[1] - state->robots[i].velocity[1];
                state->neighbors[i][n].relative_velocity[2] = state->robots[j].velocity[2] - state->robots[i].velocity[2];
                state->neighbors[i][n].is_visible = 1;
            }
        }
    }

    state->swarm_center[0] = 0.0f;
    state->swarm_center[1] = 0.0f;
    state->swarm_center[2] = 0.0f;
    int active = 0;
    float avg_vx = 0.0f, avg_vy = 0.0f, avg_vz = 0.0f;
    for (int i = 0; i < state->robot_count; i++)
    {
        if (!state->robots[i].is_active) continue;
        active++;
        state->swarm_center[0] += state->robots[i].position[0];
        state->swarm_center[1] += state->robots[i].position[1];
        state->swarm_center[2] += state->robots[i].position[2];
        avg_vx += state->robots[i].velocity[0];
        avg_vy += state->robots[i].velocity[1];
        avg_vz += state->robots[i].velocity[2];
    }
    if (active > 0)
    {
        float inv = 1.0f / active;
        state->swarm_center[0] *= inv;
        state->swarm_center[1] *= inv;
        state->swarm_center[2] *= inv;
        state->average_speed = sqrtf(avg_vx * avg_vx + avg_vy * avg_vy + avg_vz * avg_vz) * inv;
    }
    state->active_count = active;

    float spread = 0.0f;
    for (int i = 0; i < state->robot_count; i++)
    {
        if (!state->robots[i].is_active) continue;
        float dx = state->robots[i].position[0] - state->swarm_center[0];
        float dy = state->robots[i].position[1] - state->swarm_center[1];
        float dz = state->robots[i].position[2] - state->swarm_center[2];
        spread += sqrtf(dx * dx + dy * dy + dz * dz);
    }
    state->swarm_spread = (active > 0) ? spread / active : 0.0f;
}

SwarmConfig swarm_config_default(void)
{
    SwarmConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.max_robots = SWARM_MAX_ROBOTS;
    cfg.enable_collision_avoidance = 1;
    cfg.enable_obstacle_avoidance = 1;
    cfg.enable_leader_following = 0;
    cfg.enable_self_healing = 1;
    cfg.update_rate_hz = 20;
    cfg.control_loop_dt = 0.05f;
    cfg.flocking_params.cohesion_weight = 1.0f;
    cfg.flocking_params.separation_weight = 2.0f;
    cfg.flocking_params.alignment_weight = 1.5f;
    cfg.flocking_params.obstacle_avoidance_weight = 2.0f;
    cfg.flocking_params.goal_attraction_weight = 0.5f;
    cfg.flocking_params.max_force = 5.0f;
    cfg.flocking_params.neighbor_radius = 3.0f;
    cfg.flocking_params.separation_radius = 1.0f;
    cfg.flocking_params.max_velocity = 2.0f;
    cfg.flocking_params.max_acceleration = 1.0f;
    cfg.vicsek_params.speed = 1.0f;
    cfg.vicsek_params.noise_amplitude = 0.1f;
    cfg.vicsek_params.interaction_radius = 3.0f;
    cfg.cucker_smale_params.kappa = 1.0f;
    cfg.cucker_smale_params.theta = 1.0f;
    cfg.cucker_smale_params.beta = 0.5f;
    cfg.cucker_smale_params.lambda = 0.5f;
    cfg.cucker_smale_params.interaction_radius = 5.0f;
    cfg.consensus_params.position_gain = 0.5f;
    cfg.consensus_params.velocity_gain = 0.3f;
    cfg.consensus_params.convergence_threshold = 0.01f;
    cfg.consensus_params.communication_delay = 0.0f;
    cfg.formation_config.type = FORMATION_TYPE_WEDGE;
    cfg.formation_config.position_count = 0;
    cfg.formation_config.spacing = 1.5f;
    cfg.formation_config.orientation_offset = 0.0f;
    cfg.formation_config.rotation_angle = 0.0f;
    cfg.formation_config.scale = 1.0f;
    cfg.formation_config.rotate_with_leader = 1;
    cfg.formation_config.dynamic_reconfiguration = 0;
    cfg.task_config.method = TASK_ALLOCATION_CONSENSUS;
    cfg.task_config.communication_range = 10.0f;
    cfg.task_config.max_tasks_per_robot = 3;
    cfg.task_config.enable_load_balancing = 1;
    cfg.task_config.bid_noise = 0.05f;
    cfg.task_config.consensus_rounds = 3;
    strncpy(cfg.name, "DefaultSwarm", SWARM_NAME_MAX);
    return cfg;
}

SwarmController* swarm_controller_create(const SwarmConfig* config)
{
    SwarmController* ctrl = (SwarmController*)malloc(sizeof(SwarmController));
    if (!ctrl) return NULL;
    memset(ctrl, 0, sizeof(SwarmController));

    if (config) ctrl->config = *config;
    else ctrl->config = swarm_config_default();

    ctrl->state.robot_count = 0;
    ctrl->state.active_count = 0;
    ctrl->state.leader_id = -1;
    ctrl->task_count = 0;
    ctrl->allocated_tasks = 0;
    ctrl->consensus_state = NULL;
    ctrl->consensus_state_dim = 0;
    ctrl->is_initialized = 1;

    ctrl->mutex = mutex_create();

    /* 初始化UDP网络状态 */
    ctrl->udp_socket = SWARM_INVALID_SOCKET;
    ctrl->udp_initialized = 0;
    ctrl->udp_port_base = 18880;
    ctrl->unicast_count = 0;

    for (int i = 0; i < 5; i++)
        ctrl->algorithms_enabled[i] = 1;

    return ctrl;
}

void swarm_controller_destroy(SwarmController* controller)
{
    if (!controller) return;
    if (controller->mutex) { mutex_destroy(controller->mutex); controller->mutex = NULL; }
    if (controller->consensus_state) free(controller->consensus_state);

    /* 关闭UDP网络 */
    if (controller->udp_socket != SWARM_INVALID_SOCKET) {
        swarm_socket_close(controller->udp_socket);
        controller->udp_socket = SWARM_INVALID_SOCKET;
    }
    controller->udp_initialized = 0;

    free(controller);
}

int swarm_add_robot(SwarmController* controller, int robot_id, const RobotConfig* config)
{
    if (!controller) return -1;
    swarm_lock(controller);
    if (controller->state.robot_count >= SWARM_MAX_ROBOTS) { swarm_unlock(controller); return -1; }
    int idx = swarm_find_robot_index(controller, robot_id);
    if (idx >= 0) { swarm_unlock(controller); return -1; }

    int i = controller->state.robot_count++;
    SwarmRobotState* rs = &controller->state.robots[i];
    memset(rs, 0, sizeof(SwarmRobotState));
    rs->robot_id = robot_id;
    rs->is_active = 1;
    rs->battery_level = 1.0f;
    rs->task_progress = 0.0f;
    if (config) {
        controller->config.flocking_params.max_velocity = config->max_linear_velocity;
        controller->config.flocking_params.max_acceleration = config->max_acceleration;
    }
    controller->state.active_count = controller->state.robot_count;
    swarm_unlock(controller);
    return 0;
}

/* swarm_* 兼容性桥接函数已移至 src/multisystem/swarm_intelligence.c
 * 该文件提供完整的PSO群体智能实现，消除循环依赖。
 * swarm_control.c 专注机器人群体控制（蜂拥、编队、一致性协议）。 */

int swarm_update_formation_virtual(SwarmController* controller, float dt)
{
    if (!controller || !controller->is_initialized) return -1;
    SwarmState* state = &controller->state;
    FormationConfig* fc = &controller->config.formation_config;
    if (fc->position_count <= 0) return -1;
    swarm_lock(controller);
    int i;
    float centroid[3] = {0.0f, 0.0f, 0.0f};
    int active_count = 0;
    for (i = 0; i < state->robot_count; i++) {
        if (!state->robots[i].is_active) continue;
        active_count++;
        centroid[0] += state->robots[i].position[0];
        centroid[1] += state->robots[i].position[1];
        centroid[2] += state->robots[i].position[2];
    }
    if (active_count == 0) return 0;
    float inv = 1.0f / (float)active_count;
    centroid[0] *= inv; centroid[1] *= inv; centroid[2] *= inv;
    float vs_pos[3] = {controller->vs_position[0], controller->vs_position[1], controller->vs_position[2]};
    float init = (float)(fabsf(vs_pos[0]) < 0.001f && fabsf(vs_pos[1]) < 0.001f && fabsf(vs_pos[2]) < 0.001f);
    if (init) {
        vs_pos[0] = centroid[0]; vs_pos[1] = centroid[1]; vs_pos[2] = centroid[2];
    }
    float structure_force[3] = {0.0f, 0.0f, 0.0f};
    for (i = 0; i < state->robot_count; i++) {
        if (!state->robots[i].is_active) continue;
        int pos_idx = i % SWARM_MAX(fc->position_count, 1);
        float* fp = fc->positions[pos_idx];
        float cos_r = cosf(fc->rotation_angle);
        float sin_r = sinf(fc->rotation_angle);
        float target_x = vs_pos[0] + fp[0] * cos_r * fc->scale - fp[1] * sin_r * fc->scale;
        float target_y = vs_pos[1] + fp[0] * sin_r * fc->scale + fp[1] * cos_r * fc->scale;
        float target_z = vs_pos[2] + fp[2] * fc->scale;
        float dx = target_x - state->robots[i].position[0];
        float dy = target_y - state->robots[i].position[1];
        float dz = target_z - state->robots[i].position[2];
        float err = sqrtf(dx * dx + dy * dy + dz * dz);
        if (err > 0.01f) {
            float pos_gain = 3.0f;
            float speed = pos_gain * err;
            if (speed > controller->config.flocking_params.max_velocity) speed = controller->config.flocking_params.max_velocity;
            float inv_dist = 1.0f / err;
            state->robots[i].velocity[0] = dx * inv_dist * speed;
            state->robots[i].velocity[1] = dy * inv_dist * speed;
            state->robots[i].velocity[2] = dz * inv_dist * speed;
        } else {
            state->robots[i].velocity[0] = controller->vs_velocity[0];
            state->robots[i].velocity[1] = controller->vs_velocity[1];
            state->robots[i].velocity[2] = controller->vs_velocity[2];
        }
        state->robots[i].position[0] += state->robots[i].velocity[0] * dt;
        state->robots[i].position[1] += state->robots[i].velocity[1] * dt;
        state->robots[i].position[2] += state->robots[i].velocity[2] * dt;
        structure_force[0] += dx;
        structure_force[1] += dy;
        structure_force[2] += dz;
    }
    float vs_drag = 0.1f;
    float vs_spring = 0.5f;
    structure_force[0] *= vs_spring * inv;
    structure_force[1] *= vs_spring * inv;
    structure_force[2] *= vs_spring * inv;
    controller->vs_velocity[0] = (controller->vs_velocity[0] + structure_force[0] * dt) * (1.0f - vs_drag);
    controller->vs_velocity[1] = (controller->vs_velocity[1] + structure_force[1] * dt) * (1.0f - vs_drag);
    controller->vs_velocity[2] = (controller->vs_velocity[2] + structure_force[2] * dt) * (1.0f - vs_drag);
    controller->vs_position[0] += controller->vs_velocity[0] * dt;
    controller->vs_position[1] += controller->vs_velocity[1] * dt;
    controller->vs_position[2] += controller->vs_velocity[2] * dt;
    swarm_unlock(controller);
    return 0;
}

int swarm_update_formation_behavior(SwarmController* controller, float dt)
{
    if (!controller || !controller->is_initialized) return -1;
    SwarmState* state = &controller->state;
    FormationConfig* fc = &controller->config.formation_config;
    FlockingParams* fp = &controller->config.flocking_params;
    if (fc->position_count <= 0) return -1;
    swarm_lock(controller);
    swarm_update_neighbors(controller);
    int i, j;
    for (i = 0; i < state->robot_count; i++) {
        if (!state->robots[i].is_active) continue;
        float vel[3] = {0.0f, 0.0f, 0.0f};
        float w_total = 0.0f;
        int pos_idx = i % SWARM_MAX(fc->position_count, 1);
        float* fp_pos = fc->positions[pos_idx];
        float cos_r = cosf(fc->rotation_angle);
        float sin_r = sinf(fc->rotation_angle);
        float target_x, target_y, target_z;
        if (state->leader_id >= 0) {
            int lidx = swarm_find_robot_index(controller, state->leader_id);
            if (lidx >= 0) {
                target_x = state->robots[lidx].position[0] + fp_pos[0] * cos_r * fc->scale - fp_pos[1] * sin_r * fc->scale;
                target_y = state->robots[lidx].position[1] + fp_pos[0] * sin_r * fc->scale + fp_pos[1] * cos_r * fc->scale;
                target_z = state->robots[lidx].position[2] + fp_pos[2] * fc->scale;
            } else {
                target_x = fp_pos[0] * fc->scale; target_y = fp_pos[1] * fc->scale; target_z = fp_pos[2] * fc->scale;
            }
        } else {
            target_x = state->swarm_center[0] + fp_pos[0] * fc->scale;
            target_y = state->swarm_center[1] + fp_pos[1] * fc->scale;
            target_z = state->swarm_center[2] + fp_pos[2] * fc->scale;
        }
        {
            float dx = target_x - state->robots[i].position[0];
            float dy = target_y - state->robots[i].position[1];
            float dz = target_z - state->robots[i].position[2];
            float d = sqrtf(dx * dx + dy * dy + dz * dz);
            if (d > 0.01f) {
                float w = 2.0f;
                w_total += w;
                vel[0] += w * dx / d;
                vel[1] += w * dy / d;
                vel[2] += w * dz / d;
            }
        }
        {
            float sep_sq = fp->separation_radius * fp->separation_radius;
            for (j = 0; j < state->neighbor_counts[i]; j++) {
                float dsq = state->neighbors[i][j].distance * state->neighbors[i][j].distance;
                if (dsq < sep_sq && dsq > 0.001f) {
                    float rep = 1.0f / (dsq + 0.001f);
                    vel[0] += fp->separation_weight * rep * (-state->neighbors[i][j].relative_position[0] / state->neighbors[i][j].distance);
                    vel[1] += fp->separation_weight * rep * (-state->neighbors[i][j].relative_position[1] / state->neighbors[i][j].distance);
                    vel[2] += fp->separation_weight * rep * (-state->neighbors[i][j].relative_position[2] / state->neighbors[i][j].distance);
                    w_total += fp->separation_weight * rep;
                }
            }
        }
        {
            float bound = 8.0f;
            float margin = 2.0f;
            float w_bound = 3.0f;
            if (state->robots[i].position[0] < -bound + margin) {
                vel[0] += w_bound * (-bound + margin - state->robots[i].position[0]) / margin; w_total += w_bound;
            } else if (state->robots[i].position[0] > bound - margin) {
                vel[0] += w_bound * (bound - margin - state->robots[i].position[0]) / margin; w_total += w_bound;
            }
            if (state->robots[i].position[1] < -bound + margin) {
                vel[1] += w_bound * (-bound + margin - state->robots[i].position[1]) / margin; w_total += w_bound;
            } else if (state->robots[i].position[1] > bound - margin) {
                vel[1] += w_bound * (bound - margin - state->robots[i].position[1]) / margin; w_total += w_bound;
            }
            if (state->robots[i].position[2] < -bound + margin) {
                vel[2] += w_bound * (-bound + margin - state->robots[i].position[2]) / margin; w_total += w_bound;
            } else if (state->robots[i].position[2] > bound - margin) {
                vel[2] += w_bound * (bound - margin - state->robots[i].position[2]) / margin; w_total += w_bound;
            }
        }
        {
            if (state->neighbor_counts[i] > 0) {
                float align_x = 0.0f, align_y = 0.0f, align_z = 0.0f;
                for (j = 0; j < state->neighbor_counts[i]; j++) {
                    align_x += state->neighbors[i][j].relative_velocity[0];
                    align_y += state->neighbors[i][j].relative_velocity[1];
                    align_z += state->neighbors[i][j].relative_velocity[2];
                }
                float inv_n = 1.0f / (float)state->neighbor_counts[i];
                float w_align = 0.5f;
                w_total += w_align;
                vel[0] += w_align * align_x * inv_n;
                vel[1] += w_align * align_y * inv_n;
                vel[2] += w_align * align_z * inv_n;
            }
        }
        if (w_total > 0.001f) {
            vel[0] /= w_total; vel[1] /= w_total; vel[2] /= w_total;
        }
        float max_v = fp->max_velocity;
        float v_len = sqrtf(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
        if (v_len > max_v && v_len > 0.001f) {
            float s = max_v / v_len; vel[0] *= s; vel[1] *= s; vel[2] *= s;
        }
        state->robots[i].velocity[0] = vel[0];
        state->robots[i].velocity[1] = vel[1];
        state->robots[i].velocity[2] = vel[2];
        state->robots[i].position[0] += vel[0] * dt;
        state->robots[i].position[1] += vel[1] * dt;
        state->robots[i].position[2] += vel[2] * dt;
    }
    swarm_unlock(controller);
    return 0;
}

int swarm_remove_robot(SwarmController* controller, int robot_id)
{
    if (!controller) return -1;
    swarm_lock(controller);
    int idx = swarm_find_robot_index(controller, robot_id);
    if (idx < 0) { swarm_unlock(controller); return -1; }
    for (int i = idx; i < controller->state.robot_count - 1; i++)
        controller->state.robots[i] = controller->state.robots[i + 1];
    controller->state.robot_count--;
    if (controller->state.leader_id == robot_id)
        controller->state.leader_id = (controller->state.robot_count > 0) ?
            controller->state.robots[0].robot_id : -1;
    swarm_unlock(controller);
    return 0;
}

int swarm_set_leader(SwarmController* controller, int robot_id)
{
    if (!controller) return -1;
    swarm_lock(controller);
    if (swarm_find_robot_index(controller, robot_id) < 0) { swarm_unlock(controller); return -1; }
    controller->state.leader_id = robot_id;
    swarm_unlock(controller);
    return 0;
}

int swarm_init_formation(SwarmController* controller, FormationType type, float spacing)
{
    if (!controller) return -1;
    swarm_lock(controller);
    controller->config.formation_config.type = type;
    controller->config.formation_config.spacing = spacing;
    controller->config.formation_config.position_count = 0;

    int pos_count = SWARM_MIN(controller->state.robot_count, SWARM_FORMATION_MAX);
    for (int i = 0; i < pos_count; i++)
    {
        float* p = controller->config.formation_config.positions[i];
        switch (type)
        {
            case FORMATION_TYPE_LINE:
                p[0] = (float)i * spacing;
                p[1] = 0.0f;
                p[2] = 0.0f;
                break;
            case FORMATION_TYPE_VEE:
                if (i == 0) { p[0] = 0.0f; p[1] = 0.0f; }
                else
                {
                    int side = (i % 2 == 0) ? 1 : -1;
                    int row = (i + 1) / 2;
                    p[0] = (float)row * spacing * -1.0f;
                    p[1] = (float)side * (float)row * spacing * 0.5f;
                }
                p[2] = 0.0f;
                break;
            case FORMATION_TYPE_CIRCLE:
            {
                float angle = 2.0f * (float)M_PI * (float)i / (float)pos_count;
                p[0] = spacing * cosf(angle);
                p[1] = spacing * sinf(angle);
                p[2] = 0.0f;
                break;
            }
            case FORMATION_TYPE_WEDGE:
                if (i == 0) { p[0] = spacing; p[1] = 0.0f; }
                else
                {
                    int side = (i % 2 == 0) ? 1 : -1;
                    int row = (i + 1) / 2;
                    p[0] = spacing - (float)row * spacing * 0.8f;
                    p[1] = (float)side * (float)row * spacing * 0.6f;
                }
                p[2] = 0.0f;
                break;
            case FORMATION_TYPE_SQUARE:
            {
                int cols = (int)ceilf(sqrtf((float)pos_count));
                int row = i / cols;
                int col = i % cols;
                p[0] = (float)col * spacing - (float)(cols - 1) * spacing * 0.5f;
                p[1] = (float)row * spacing - (float)((pos_count + cols - 1) / cols - 1) * spacing * 0.5f;
                p[2] = 0.0f;
                break;
            }
            case FORMATION_TYPE_COLUMN:
                p[0] = 0.0f;
                p[1] = (float)i * spacing - (float)(pos_count - 1) * spacing * 0.5f;
                p[2] = 0.0f;
                break;
            case FORMATION_TYPE_DIAMOND:
            {
                int ring = (int)((sqrtf(1.0f + 8.0f * (float)i) - 1.0f) / 2.0f);
                int idx_in_ring = i - ring * (ring + 1) / 2;
                float a = 2.0f * (float)M_PI * (float)idx_in_ring / (float)(ring * 4 + 1);
                p[0] = (float)ring * spacing * 0.5f * cosf(a);
                p[1] = (float)ring * spacing * 0.5f * sinf(a);
                p[2] = 0.0f;
                break;
            }
            case FORMATION_TYPE_CROSS:
                if (i == 0) { p[0] = 0.0f; p[1] = 0.0f; }
                else
                {
                    int arm = (i - 1) % 4;
                    int dist = (i - 1) / 4 + 1;
                    float angles[4] = {0.0f, (float)M_PI / 2.0f, (float)M_PI, 3.0f * (float)M_PI / 2.0f};
                    p[0] = (float)dist * spacing * cosf(angles[arm]);
                    p[1] = (float)dist * spacing * sinf(angles[arm]);
                }
                p[2] = 0.0f;
                break;
            default:
                p[0] = (float)i * spacing;
                p[1] = 0.0f;
                p[2] = 0.0f;
                break;
        }
    }
    controller->config.formation_config.position_count = pos_count;
    swarm_unlock(controller);
    return 0;
}

int swarm_set_formation(SwarmController* controller, const FormationConfig* config)
{
    if (!controller || !config) return -1;
    swarm_lock(controller);
    controller->config.formation_config = *config;
    swarm_unlock(controller);
    return 0;
}

int swarm_get_formation_positions(SwarmController* controller, int robot_id, float* out_pos, float* out_orient)
{
    if (!controller || !out_pos) return -1;
    swarm_lock(controller);
    int idx = swarm_find_robot_index(controller, robot_id);
    if (idx < 0) { swarm_unlock(controller); return -1; }

    FormationConfig* fc = &controller->config.formation_config;
    int pos_idx = idx % SWARM_MAX(fc->position_count, 1);
    float* fp = fc->positions[pos_idx];

    float cos_r = cosf(fc->rotation_angle);
    float sin_r = sinf(fc->rotation_angle);

    if (controller->state.leader_id >= 0)
    {
        int lidx = swarm_find_robot_index(controller, controller->state.leader_id);
        if (lidx >= 0)
        {
            out_pos[0] = controller->state.robots[lidx].position[0] + fp[0] * cos_r - fp[1] * sin_r;
            out_pos[1] = controller->state.robots[lidx].position[1] + fp[0] * sin_r + fp[1] * cos_r;
            out_pos[2] = controller->state.robots[lidx].position[2] + fp[2];
        }
    }
    else
    {
        out_pos[0] = controller->state.swarm_center[0] + fp[0] * cos_r - fp[1] * sin_r;
        out_pos[1] = controller->state.swarm_center[1] + fp[0] * sin_r + fp[1] * cos_r;
        out_pos[2] = controller->state.swarm_center[2] + fp[2];
    }

    if (out_orient)
    {
        out_orient[0] = fc->orientation_offset + fc->rotation_angle;
        out_orient[1] = 0.0f;
        out_orient[2] = 0.0f;
        out_orient[3] = 0.0f;
    }
    swarm_unlock(controller);
    return 0;
}

int swarm_update_flocking(SwarmController* controller, float dt)
{
    if (!controller || !controller->is_initialized) return -1;
    swarm_lock(controller);

    FlockingParams* fp = &controller->config.flocking_params;
    SwarmState* state = &controller->state;
    swarm_update_neighbors(controller);

    for (int i = 0; i < state->robot_count; i++)
    {
        if (!state->robots[i].is_active) continue;
        SwarmRobotState* robot = &state->robots[i];
        float force[3] = {0.0f, 0.0f, 0.0f};
        int neighbor_count = state->neighbor_counts[i];
        SwarmNeighbor* neighbors = state->neighbors[i];

        if (neighbor_count == 0) continue;

        float cohesion[3] = {0.0f, 0.0f, 0.0f};
        float separation[3] = {0.0f, 0.0f, 0.0f};
        float alignment[3] = {0.0f, 0.0f, 0.0f};
        float sep_sq = fp->separation_radius * fp->separation_radius;

        for (int j = 0; j < neighbor_count; j++)
        {
            float dsq = neighbors[j].distance * neighbors[j].distance;
            cohesion[0] += neighbors[j].relative_position[0];
            cohesion[1] += neighbors[j].relative_position[1];
            cohesion[2] += neighbors[j].relative_position[2];

            alignment[0] += neighbors[j].relative_velocity[0];
            alignment[1] += neighbors[j].relative_velocity[1];
            alignment[2] += neighbors[j].relative_velocity[2];

            if (dsq < sep_sq && dsq > 0.001f)
            {
                float repulse = 1.0f / dsq;
                separation[0] -= neighbors[j].relative_position[0] * repulse;
                separation[1] -= neighbors[j].relative_position[1] * repulse;
                separation[2] -= neighbors[j].relative_position[2] * repulse;
            }
        }

        float inv_n = 1.0f / (float)neighbor_count;
        cohesion[0] *= inv_n;
        cohesion[1] *= inv_n;
        cohesion[2] *= inv_n;
        alignment[0] *= inv_n;
        alignment[1] *= inv_n;
        alignment[2] *= inv_n;

        if (fp->cohesion_weight > 0.001f)
        {
            force[0] += cohesion[0] * fp->cohesion_weight;
            force[1] += cohesion[1] * fp->cohesion_weight;
            force[2] += cohesion[2] * fp->cohesion_weight;
        }
        if (fp->separation_weight > 0.001f)
        {
            force[0] += separation[0] * fp->separation_weight;
            force[1] += separation[1] * fp->separation_weight;
            force[2] += separation[2] * fp->separation_weight;
        }
        if (fp->alignment_weight > 0.001f)
        {
            force[0] += alignment[0] * fp->alignment_weight;
            force[1] += alignment[1] * fp->alignment_weight;
            force[2] += alignment[2] * fp->alignment_weight;
        }

        float f_len = sqrtf(force[0] * force[0] + force[1] * force[1] + force[2] * force[2]);
        if (f_len > fp->max_force && f_len > 0.001f)
        {
            float scale = fp->max_force / f_len;
            force[0] *= scale;
            force[1] *= scale;
            force[2] *= scale;
        }

        robot->velocity[0] += force[0] * dt;
        robot->velocity[1] += force[1] * dt;
        robot->velocity[2] += force[2] * dt;

        float v_len = sqrtf(robot->velocity[0] * robot->velocity[0] +
                            robot->velocity[1] * robot->velocity[1] +
                            robot->velocity[2] * robot->velocity[2]);
        if (v_len > fp->max_velocity && v_len > 0.001f)
        {
            float scale = fp->max_velocity / v_len;
            robot->velocity[0] *= scale;
            robot->velocity[1] *= scale;
            robot->velocity[2] *= scale;
        }

        robot->position[0] += robot->velocity[0] * dt;
        robot->position[1] += robot->velocity[1] * dt;
        robot->position[2] += robot->velocity[2] * dt;
    }
    swarm_unlock(controller);
    return 0;
}

int swarm_update_consensus(SwarmController* controller, float* shared_state, int state_dim, float dt)
{
    if (!controller || !controller->is_initialized) return -1;
    if (!shared_state || state_dim <= 0) return -1;
    swarm_lock(controller);

    ConsensusParams* cp = &controller->config.consensus_params;
    SwarmState* state = &controller->state;
    swarm_update_neighbors(controller);

    if (!controller->consensus_state || controller->consensus_state_dim != state_dim)
    {
        if (controller->consensus_state) free(controller->consensus_state);
        controller->consensus_state = (float*)malloc(state_dim * sizeof(float));
        controller->consensus_state_dim = state_dim;
        if (!controller->consensus_state) return -1;
        memcpy(controller->consensus_state, shared_state, state_dim * sizeof(float));
    }

    float* consensus_avg = (float*)calloc(state_dim, sizeof(float));
    if (!consensus_avg) return -1;
    int total_neighbors = 0;

    for (int i = 0; i < state->robot_count; i++)
    {
        if (!state->robots[i].is_active) continue;
        int nc = state->neighbor_counts[i];
        for (int j = 0; j < nc; j++)
        {
            int nidx = swarm_find_robot_index(controller, state->neighbors[i][j].robot_id);
            if (nidx < 0 || !state->robots[nidx].is_active) continue;
            total_neighbors++;
            if (i == 0)
            {
                for (int d = 0; d < state_dim; d++)
                    consensus_avg[d] += shared_state[d];
            }
        }
    }

    if (total_neighbors > 0)
    {
        float inv = 1.0f / (total_neighbors + 1);
        float* temp = (float*)calloc(state_dim, sizeof(float));
        if (temp)
        {
            for (int d = 0; d < state_dim; d++)
            {
                temp[d] = controller->consensus_state[d] + dt * cp->position_gain * (consensus_avg[d] * inv - controller->consensus_state[d]);
            }
            memcpy(controller->consensus_state, temp, state_dim * sizeof(float));
            free(temp);
        }
    }

    float max_diff = 0.0f;
    for (int d = 0; d < state_dim; d++)
    {
        float diff = SWARM_ABS(controller->consensus_state[d] - shared_state[d]);
        if (diff > max_diff) max_diff = diff;
    }
    if (max_diff < cp->convergence_threshold)
    {
        free(consensus_avg);
        swarm_unlock(controller);
        return 1;
    }

    memcpy(shared_state, controller->consensus_state, state_dim * sizeof(float));
    free(consensus_avg);
    swarm_unlock(controller);
    return 0;
}

int swarm_update_formation(SwarmController* controller, float dt)
{
    if (!controller || !controller->is_initialized) return -1;
    SwarmState* state = &controller->state;
    FormationConfig* fc = &controller->config.formation_config;

    if (fc->position_count <= 0) return -1;
    swarm_lock(controller);

    float cos_r = cosf(fc->rotation_angle);
    float sin_r = sinf(fc->rotation_angle);
    float center[3] = {state->swarm_center[0], state->swarm_center[1], state->swarm_center[2]};

    if (state->leader_id >= 0)
    {
        int lidx = swarm_find_robot_index(controller, state->leader_id);
        if (lidx >= 0)
        {
            center[0] = state->robots[lidx].position[0];
            center[1] = state->robots[lidx].position[1];
            center[2] = state->robots[lidx].position[2];

            if (fc->rotate_with_leader)
            {
                cos_r = cosf(state->robots[lidx].orientation[0]);
                sin_r = sinf(state->robots[lidx].orientation[0]);
            }
        }
    }

    float max_vel = controller->config.flocking_params.max_velocity;
    float pos_gain = 2.0f;
    float formation_error = 0.0f;

    for (int i = 0; i < state->robot_count; i++)
    {
        if (!state->robots[i].is_active) continue;
        int pos_idx = i % SWARM_MAX(fc->position_count, 1);
        float* fp = fc->positions[pos_idx];

        float target_x = center[0] + fp[0] * cos_r * fc->scale - fp[1] * sin_r * fc->scale;
        float target_y = center[1] + fp[0] * sin_r * fc->scale + fp[1] * cos_r * fc->scale;
        float target_z = center[2] + fp[2] * fc->scale;

        float dx = target_x - state->robots[i].position[0];
        float dy = target_y - state->robots[i].position[1];
        float dz = target_z - state->robots[i].position[2];

        float err = sqrtf(dx * dx + dy * dy + dz * dz);
        formation_error += err;

        if (err > 0.01f)
        {
            float speed = pos_gain * err;
            if (speed > max_vel) speed = max_vel;
            float inv_dist = 1.0f / err;
            state->robots[i].velocity[0] += (dx * inv_dist * speed - state->robots[i].velocity[0]) * 0.1f;
            state->robots[i].velocity[1] += (dy * inv_dist * speed - state->robots[i].velocity[1]) * 0.1f;
            state->robots[i].velocity[2] += (dz * inv_dist * speed - state->robots[i].velocity[2]) * 0.1f;
        }

        state->robots[i].position[0] += state->robots[i].velocity[0] * dt;
        state->robots[i].position[1] += state->robots[i].velocity[1] * dt;
        state->robots[i].position[2] += state->robots[i].velocity[2] * dt;
    }

    state->formation_error = formation_error / (float)state->active_count;
    swarm_unlock(controller);
    return 0;
}

int swarm_update_all(SwarmController* controller, float dt)
{
    if (!controller || !controller->is_initialized) return -1;
    swarm_lock(controller);
    swarm_update_neighbors(controller);

    if (controller->algorithms_enabled[SWARM_ALGORITHM_FLOCKING])
        swarm_update_flocking(controller, dt);

    if (controller->algorithms_enabled[SWARM_ALGORITHM_FORMATION])
        swarm_update_formation(controller, dt);

    if (controller->algorithms_enabled[SWARM_ALGORITHM_TASK_ALLOCATION] &&
        controller->task_count > controller->allocated_tasks)
        swarm_allocate_tasks(controller);

    swarm_unlock(controller);
    return 0;
}

/* 死锁检测：检测多机器人系统中是否存在死锁环
 * 死锁特征：多轮更新后所有机器人位置变化极小，但目标尚未到达
 * 解除策略：随机扰动最慢机器人的目标位置 */
int swarm_deadlock_detect_and_resolve(SwarmController* controller) {
    if (!controller || !controller->is_initialized) return 0;
    swarm_lock(controller);

    static float prev_positions[SWARM_FORMATION_MAX][3] = {{0}};
    static int stall_counter[SWARM_FORMATION_MAX] = {0};
    static int check_initialized = 0;

    if (!check_initialized) {
        for (int i = 0; i < controller->state.robot_count && i < SWARM_FORMATION_MAX; i++) {
            memcpy(prev_positions[i], controller->state.robots[i].position, 3 * sizeof(float));
        }
        check_initialized = 1;
        swarm_unlock(controller);
        return 0;
    }

    int deadlock_count = 0;
    float deadlock_threshold = 0.001f;

    for (int i = 0; i < controller->state.robot_count && i < SWARM_FORMATION_MAX; i++) {
        float dx = controller->state.robots[i].position[0] - prev_positions[i][0];
        float dy = controller->state.robots[i].position[1] - prev_positions[i][1];
        float dist = sqrtf(dx*dx + dy*dy);

        if (dist < deadlock_threshold && controller->state.robots[i].velocity[0] != 0.0f) {
            stall_counter[i]++;
        } else {
            stall_counter[i] = 0;
        }

        if (stall_counter[i] > 50) {
            deadlock_count++;
            stall_counter[i] = 0;
            /* 解除死锁：给停滞机器人的目标位置添加小随机偏移 */
            controller->state.target_positions[i][0] += 
                sinf((float)(controller->state.step_count + i * 137)) * 0.5f;
            controller->state.target_positions[i][1] += 
                cosf((float)(controller->state.step_count + i * 251)) * 0.5f;
        }

        memcpy(prev_positions[i], controller->state.robots[i].position, 3 * sizeof(float));
    }

    swarm_unlock(controller);
    return deadlock_count;
}

int swarm_get_state(SwarmController* controller, SwarmState* state)
{
    if (!controller || !state) return -1;
    swarm_lock(controller);
    memcpy(state, &controller->state, sizeof(SwarmState));
    swarm_unlock(controller);
    return 0;
}

int swarm_get_robot_command(SwarmController* controller, int robot_id, RobotCommand* command)
{
    if (!controller || !command) return -1;
    int idx = swarm_find_robot_index(controller, robot_id);
    if (idx < 0) return -1;
    swarm_lock(controller);
    memset(command, 0, sizeof(RobotCommand));
    command->mode = MOTION_MODE_POSITION;
    command->target_position[0] = controller->state.robots[idx].position[0];
    command->target_position[1] = controller->state.robots[idx].position[1];
    command->target_position[2] = controller->state.robots[idx].position[2];
    command->max_velocity = sqrtf(controller->state.robots[idx].velocity[0] *
                                  controller->state.robots[idx].velocity[0] +
                                  controller->state.robots[idx].velocity[1] *
                                  controller->state.robots[idx].velocity[1] +
                                  controller->state.robots[idx].velocity[2] *
                                  controller->state.robots[idx].velocity[2]);
    swarm_unlock(controller);
    return 0;
}

int swarm_set_goal(SwarmController* controller, const float* goal_position, float radius)
{
    if (!controller || !goal_position) return -1;
    swarm_lock(controller);
    float* center = controller->state.swarm_center;
    float dx = goal_position[0] - center[0];
    float dy = goal_position[1] - center[1];
    float dz = goal_position[2] - center[2];
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    if (dist < 0.001f) { swarm_unlock(controller); return 0; }
    if (radius > 0.0f && dist <= radius) { swarm_unlock(controller); return 1; }

    float gx = controller->config.flocking_params.goal_attraction_weight;
    for (int i = 0; i < controller->state.robot_count; i++)
    {
        if (!controller->state.robots[i].is_active) continue;
        controller->state.robots[i].velocity[0] += dx / dist * gx;
        controller->state.robots[i].velocity[1] += dy / dist * gx;
        controller->state.robots[i].velocity[2] += dz / dist * gx;
    }
    swarm_unlock(controller);
    return 0;
}

int swarm_add_task(SwarmController* controller, const SwarmTask* task)
{
    if (!controller || !task) return -1;
    swarm_lock(controller);
    if (controller->task_count >= SWARM_MAX_TASKS) { swarm_unlock(controller); return -1; }
    controller->tasks[controller->task_count++] = *task;
    swarm_unlock(controller);
    return 0;
}

int swarm_allocate_tasks(SwarmController* controller)
{
    if (!controller || controller->task_count <= 0) return -1;
    SwarmState* state = &controller->state;
    swarm_lock(controller);

    int pending = 0;
    for (int i = 0; i < controller->task_count; i++)
    {
        if (!controller->tasks[i].is_assigned && !controller->tasks[i].is_completed)
            pending++;
    }
    if (pending <= 0) { swarm_unlock(controller); return 0; }

    float* bids = (float*)malloc(controller->task_count * state->robot_count * sizeof(float));
    if (!bids) { swarm_unlock(controller); return -1; }

    for (int t = 0; t < controller->task_count; t++)
    {
        if (controller->tasks[t].is_assigned || controller->tasks[t].is_completed) continue;
        for (int r = 0; r < state->robot_count; r++)
        {
            if (!state->robots[r].is_active) continue;
            float dx = state->robots[r].position[0] - controller->tasks[t].position[0];
            float dy = state->robots[r].position[1] - controller->tasks[t].position[1];
            float dz = state->robots[r].position[2] - controller->tasks[t].position[2];
            float dist = sqrtf(dx * dx + dy * dy + dz * dz);
            float cost = dist / SWARM_MAX(state->robots[r].battery_level, 0.1f);
            cost -= controller->tasks[t].reward * 0.1f;
            cost += secure_random_float() * controller->config.task_config.bid_noise;
            bids[t * state->robot_count + r] = -cost;
        }
    }

    int* assigned = (int*)calloc(state->robot_count, sizeof(int));
    if (!assigned) { free(bids); swarm_unlock(controller); return -1; }

    if (controller->config.task_config.method == TASK_ALLOCATION_AUCTION)
    {
        for (int r = 0; r < state->robot_count; r++)
        {
            if (!state->robots[r].is_active) continue;
            float best_bid = -1e9f;
            int best_task = -1;
            for (int t = 0; t < controller->task_count; t++)
            {
                if (controller->tasks[t].is_assigned || controller->tasks[t].is_completed) continue;
                float b = bids[t * state->robot_count + r];
                if (b > best_bid) { best_bid = b; best_task = t; }
            }
            if (best_task >= 0)
            {
                controller->tasks[best_task].assigned_robot_id = state->robots[r].robot_id;
                controller->tasks[best_task].is_assigned = 1;
                assigned[r] = 1;
            }
        }
    }
    else
    {
        for (int t = 0; t < controller->task_count; t++)
        {
            if (controller->tasks[t].is_assigned || controller->tasks[t].is_completed) continue;
            float best_bid = -1e9f;
            int best_robot = -1;
            for (int r = 0; r < state->robot_count; r++)
            {
                if (!state->robots[r].is_active || assigned[r] >= controller->config.task_config.max_tasks_per_robot) continue;
                float b = bids[t * state->robot_count + r];
                if (b > best_bid) { best_bid = b; best_robot = r; }
            }
            if (best_robot >= 0)
            {
                controller->tasks[t].assigned_robot_id = state->robots[best_robot].robot_id;
                controller->tasks[t].is_assigned = 1;
                assigned[best_robot]++;
            }
        }
    }

    free(bids);
    free(assigned);
    controller->allocated_tasks = 0;
    for (int i = 0; i < controller->task_count; i++)
        if (controller->tasks[i].is_assigned) controller->allocated_tasks++;
    swarm_unlock(controller);
    return 0;
}

int swarm_get_task_status(SwarmController* controller, const char* task_id, int* assigned_robot, float* progress)
{
    if (!controller || !task_id) return -1;
    swarm_lock(controller);
    for (int i = 0; i < controller->task_count; i++)
    {
        if (strcmp(controller->tasks[i].task_id, task_id) == 0)
        {
            if (assigned_robot) *assigned_robot = controller->tasks[i].assigned_robot_id;
            if (progress) *progress = controller->tasks[i].estimated_cost;
            int ret = controller->tasks[i].is_completed ? 2 : (controller->tasks[i].is_assigned ? 1 : 0);
            swarm_unlock(controller);
            return ret;
        }
    }
    swarm_unlock(controller);
    return -1;
}

int swarm_enable_algorithm(SwarmController* controller, SwarmAlgorithm algo, int enable)
{
    if (!controller) return -1;
    if (algo < 0 || algo >= 5) return -1;
    swarm_lock(controller);
    controller->algorithms_enabled[algo] = enable;
    swarm_unlock(controller);
    return 0;
}

int swarm_is_algorithm_enabled(const SwarmController* controller, SwarmAlgorithm algo)
{
    if (!controller) return 0;
    if (algo < 0 || algo >= 5) return 0;
    int ret;
    swarm_lock((SwarmController*)controller);
    ret = controller->algorithms_enabled[algo];
    swarm_unlock((SwarmController*)controller);
    return ret;
}

int swarm_set_flocking_params(SwarmController* controller, const FlockingParams* params)
{
    if (!controller || !params) return -1;
    swarm_lock(controller);
    controller->config.flocking_params = *params;
    swarm_unlock(controller);
    return 0;
}

int swarm_set_consensus_params(SwarmController* controller, const ConsensusParams* params)
{
    if (!controller || !params) return -1;
    swarm_lock(controller);
    controller->config.consensus_params = *params;
    swarm_unlock(controller);
    return 0;
}

int swarm_set_task_allocation(SwarmController* controller, const TaskAllocationConfig* config)
{
    if (!controller || !config) return -1;
    swarm_lock(controller);
    controller->config.task_config = *config;
    swarm_unlock(controller);
    return 0;
}

int swarm_update_vicsek(SwarmController* controller, float dt)
{
    if (!controller || !controller->is_initialized) return -1;
    swarm_lock(controller);
    SwarmState* state = &controller->state;
    VicsekParams* vp = &controller->config.vicsek_params;
    float angles[SWARM_MAX_ROBOTS];
    float new_angles[SWARM_MAX_ROBOTS];
    int i, j;
    if (state->robot_count <= 0) return 0;
    swarm_update_neighbors(controller);
    for (i = 0; i < state->robot_count; i++) {
        SwarmRobotState* robot = &state->robots[i];
        angles[i] = atan2f(robot->velocity[1], robot->velocity[0]);
    }
    for (i = 0; i < state->robot_count; i++) {
        if (!state->robots[i].is_active) continue;
        float sum_sin = sinf(angles[i]);
        float sum_cos = cosf(angles[i]);
        int neighbor_count = 1;
        for (j = 0; j < state->neighbor_counts[i]; j++) {
            int nidx = swarm_find_robot_index(controller, state->neighbors[i][j].robot_id);
            if (nidx < 0 || nidx == i) continue;
            sum_sin += sinf(angles[nidx]);
            sum_cos += cosf(angles[nidx]);
            neighbor_count++;
        }
        float mean_angle = atan2f(sum_sin / (float)neighbor_count, sum_cos / (float)neighbor_count);
        /* K-012修复：使用安全随机数 */
        float noise = (secure_random_float() - 0.5f) * 2.0f * vp->noise_amplitude;
        new_angles[i] = mean_angle + noise;
    }
    for (i = 0; i < state->robot_count; i++) {
        if (!state->robots[i].is_active) continue;
        SwarmRobotState* robot = &state->robots[i];
        robot->velocity[0] = vp->speed * cosf(new_angles[i]);
        robot->velocity[1] = vp->speed * sinf(new_angles[i]);
        robot->velocity[2] = 0.0f;
        robot->position[0] += robot->velocity[0] * dt;
        robot->position[1] += robot->velocity[1] * dt;
        robot->position[2] += robot->velocity[2] * dt;
    }
    swarm_unlock(controller);
    return 0;
}

int swarm_update_cucker_smale(SwarmController* controller, float dt)
{
    if (!controller || !controller->is_initialized) return -1;
    swarm_lock(controller);
    SwarmState* state = &controller->state;
    CuckerSmaleParams* cp = &controller->config.cucker_smale_params;
    float new_vel[SWARM_MAX_ROBOTS][3];
    int i, j;
    if (state->robot_count <= 0) return 0;
    swarm_update_neighbors(controller);
    for (i = 0; i < state->robot_count; i++) {
        if (!state->robots[i].is_active) continue;
        SwarmRobotState* robot = &state->robots[i];
        float acc[3] = {0.0f, 0.0f, 0.0f};
        float weight_sum = 0.0f;
        for (j = 0; j < state->neighbor_counts[i]; j++) {
            int nidx = swarm_find_robot_index(controller, state->neighbors[i][j].robot_id);
            if (nidx < 0 || nidx == i) continue;
            float dx = state->robots[nidx].position[0] - robot->position[0];
            float dy = state->robots[nidx].position[1] - robot->position[1];
            float dz = state->robots[nidx].position[2] - robot->position[2];
            float dist_sq = dx * dx + dy * dy + dz * dz;
            float weight = cp->kappa / (1.0f + cp->theta * dist_sq);
            float dvx = state->robots[nidx].velocity[0] - robot->velocity[0];
            float dvy = state->robots[nidx].velocity[1] - robot->velocity[1];
            float dvz = state->robots[nidx].velocity[2] - robot->velocity[2];
            acc[0] += weight * dvx;
            acc[1] += weight * dvy;
            acc[2] += weight * dvz;
            weight_sum += weight;
        }
        if (weight_sum > 1e-10f) {
            float scale = cp->lambda * dt;
            new_vel[i][0] = robot->velocity[0] + scale * acc[0];
            new_vel[i][1] = robot->velocity[1] + scale * acc[1];
            new_vel[i][2] = robot->velocity[2] + scale * acc[2];
        } else {
            new_vel[i][0] = robot->velocity[0];
            new_vel[i][1] = robot->velocity[1];
            new_vel[i][2] = robot->velocity[2];
        }
    }
    for (i = 0; i < state->robot_count; i++) {
        if (!state->robots[i].is_active) continue;
        SwarmRobotState* robot = &state->robots[i];
        robot->velocity[0] = new_vel[i][0];
        robot->velocity[1] = new_vel[i][1];
        robot->velocity[2] = new_vel[i][2];
        robot->position[0] += robot->velocity[0] * dt;
        robot->position[1] += robot->velocity[1] * dt;
        robot->position[2] += robot->velocity[2] * dt;
    }
    swarm_unlock(controller);
    return 0;
}

int swarm_update_flocking_enhanced(SwarmController* controller, float dt)
{
    if (!controller || !controller->is_initialized) return -1;
    swarm_lock(controller);
    FlockingParams* fp = &controller->config.flocking_params;
    SwarmState* state = &controller->state;
    int i, j;
    swarm_update_neighbors(controller);
    for (i = 0; i < state->robot_count; i++) {
        if (!state->robots[i].is_active) continue;
        SwarmRobotState* robot = &state->robots[i];
        float force[3] = {0.0f, 0.0f, 0.0f};
        int nc = state->neighbor_counts[i];
        SwarmNeighbor* neighbors = state->neighbors[i];
        float cohesion[3] = {0.0f, 0.0f, 0.0f};
        float separation[3] = {0.0f, 0.0f, 0.0f};
        float alignment[3] = {0.0f, 0.0f, 0.0f};
        float wall_rep[3] = {0.0f, 0.0f, 0.0f};
        float leader_force[3] = {0.0f, 0.0f, 0.0f};
        float sep_sq = fp->separation_radius * fp->separation_radius;
        float neighbor_count_f = (float)(nc > 0 ? nc : 1);
        if (nc > 0) {
            for (j = 0; j < nc; j++) {
                float dsq = neighbors[j].distance * neighbors[j].distance;
                cohesion[0] += neighbors[j].relative_position[0];
                cohesion[1] += neighbors[j].relative_position[1];
                cohesion[2] += neighbors[j].relative_position[2];
                alignment[0] += neighbors[j].relative_velocity[0];
                alignment[1] += neighbors[j].relative_velocity[1];
                alignment[2] += neighbors[j].relative_velocity[2];
                if (dsq < sep_sq && dsq > 0.001f) {
                    float repulse = 1.0f / (dsq + 0.001f);
                    separation[0] -= neighbors[j].relative_position[0] * repulse;
                    separation[1] -= neighbors[j].relative_position[1] * repulse;
                    separation[2] -= neighbors[j].relative_position[2] * repulse;
                }
            }
            cohesion[0] /= neighbor_count_f;
            cohesion[1] /= neighbor_count_f;
            cohesion[2] /= neighbor_count_f;
            alignment[0] /= neighbor_count_f;
            alignment[1] /= neighbor_count_f;
            alignment[2] /= neighbor_count_f;
        }
        if (fp->cohesion_weight > 0.001f) {
            force[0] += cohesion[0] * fp->cohesion_weight;
            force[1] += cohesion[1] * fp->cohesion_weight;
            force[2] += cohesion[2] * fp->cohesion_weight;
        }
        if (fp->separation_weight > 0.001f) {
            force[0] += separation[0] * fp->separation_weight;
            force[1] += separation[1] * fp->separation_weight;
            force[2] += separation[2] * fp->separation_weight;
        }
        if (fp->alignment_weight > 0.001f) {
            force[0] += alignment[0] * fp->alignment_weight;
            force[1] += alignment[1] * fp->alignment_weight;
            force[2] += alignment[2] * fp->alignment_weight;
        }
        if (fp->goal_attraction_weight > 0.001f && controller->state.leader_id >= 0) {
            int lidx = swarm_find_robot_index(controller, state->leader_id);
            if (lidx >= 0 && lidx != i) {
                float dx = state->robots[lidx].position[0] - robot->position[0];
                float dy = state->robots[lidx].position[1] - robot->position[1];
                float dz = state->robots[lidx].position[2] - robot->position[2];
                float d = sqrtf(dx * dx + dy * dy + dz * dz);
                if (d > fp->separation_radius * 2.0f && d > 0.001f) {
                    leader_force[0] = dx / d;
                    leader_force[1] = dy / d;
                    leader_force[2] = dz / d;
                    force[0] += leader_force[0] * fp->goal_attraction_weight;
                    force[1] += leader_force[1] * fp->goal_attraction_weight;
                    force[2] += leader_force[2] * fp->goal_attraction_weight;
                }
            }
        }
        {
            float boundary = 8.0f;
            float margin = 2.0f;
            if (robot->position[0] < -boundary + margin) wall_rep[0] = (boundary - margin + robot->position[0]) / margin;
            else if (robot->position[0] > boundary - margin) wall_rep[0] = (boundary - margin - robot->position[0]) / margin;
            if (robot->position[1] < -boundary + margin) wall_rep[1] = (boundary - margin + robot->position[1]) / margin;
            else if (robot->position[1] > boundary - margin) wall_rep[1] = (boundary - margin - robot->position[1]) / margin;
            if (robot->position[2] < -boundary + margin) wall_rep[2] = (boundary - margin + robot->position[2]) / margin;
            else if (robot->position[2] > boundary - margin) wall_rep[2] = (boundary - margin - robot->position[2]) / margin;
            float wr = 3.0f;
            force[0] += wall_rep[0] * wr;
            force[1] += wall_rep[1] * wr;
            force[2] += wall_rep[2] * wr;
        }
        {
            float f_len = sqrtf(force[0] * force[0] + force[1] * force[1] + force[2] * force[2]);
            if (f_len > fp->max_force && f_len > 0.001f) {
                float scale = fp->max_force / f_len;
                force[0] *= scale; force[1] *= scale; force[2] *= scale;
            }
            robot->velocity[0] += force[0] * dt;
            robot->velocity[1] += force[1] * dt;
            robot->velocity[2] += force[2] * dt;
            float v_len = sqrtf(robot->velocity[0] * robot->velocity[0] +
                                robot->velocity[1] * robot->velocity[1] +
                                robot->velocity[2] * robot->velocity[2]);
            if (v_len > fp->max_velocity && v_len > 0.001f) {
                float scale = fp->max_velocity / v_len;
                robot->velocity[0] *= scale; robot->velocity[1] *= scale; robot->velocity[2] *= scale;
            }
            robot->position[0] += robot->velocity[0] * dt;
            robot->position[1] += robot->velocity[1] * dt;
            robot->position[2] += robot->velocity[2] * dt;
        }
    }
    swarm_unlock(controller);
    return 0;
}

static float hungarian_build_cost(SwarmController* controller, int task_idx, int robot_idx)
{
    float dx = controller->state.robots[robot_idx].position[0] - controller->tasks[task_idx].position[0];
    float dy = controller->state.robots[robot_idx].position[1] - controller->tasks[task_idx].position[1];
    float dz = controller->state.robots[robot_idx].position[2] - controller->tasks[task_idx].position[2];
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    float cost = dist / SWARM_MAX(controller->state.robots[robot_idx].battery_level, 0.1f);
    cost -= controller->tasks[task_idx].reward * 0.1f;
    return SWARM_MAX(cost, 0.001f);
}

int swarm_allocate_hungarian(SwarmController* controller)
{
    if (!controller || controller->task_count <= 0) return -1;
    swarm_lock(controller);
    int n_tasks = controller->task_count;
    int n_robots = controller->state.robot_count;
    int i, j;
    for (i = 0; i < n_tasks; i++)
        if (controller->tasks[i].is_completed || controller->tasks[i].is_assigned) { swarm_unlock(controller); return 0; }
    /* P3-071修复: 超出HUNGARIAN_MAX(256)时回退到贪婪/拍卖分配，而非直接失败 */
    if (n_tasks > HUNGARIAN_MAX || n_robots > HUNGARIAN_MAX) {
        swarm_unlock(controller);
        return swarm_allocate_tasks(controller);
    }
    int n = SWARM_MAX(n_tasks, n_robots);
    float cost[256][256];
    int m = n;
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            if (i < n_tasks && j < n_robots && controller->state.robots[j].is_active)
                cost[i][j] = hungarian_build_cost(controller, i, j);
            else
                cost[i][j] = HUNGARIAN_INF;
        }
    }
    float u[256], v[256];
    int p[256], way[256];
    for (i = 0; i < m; i++) {
        p[i] = 0;
        v[i] = 0.0f;
        u[i] = 0.0f;
    }
    for (i = 1; i <= m; i++) {
        p[0] = i;
        int j0 = 0;
        float minv[256];
        int used[256];
        for (j = 0; j < m; j++) {
            minv[j] = HUNGARIAN_INF;
            used[j] = 0;
        }
        do {
            used[j0] = 1;
            int i0 = p[j0];
            float delta = HUNGARIAN_INF;
            int j1 = 0;
            for (j = 1; j <= m; j++) {
                if (!used[j]) {
                    float cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (j = 0; j <= m; j++) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    for (i = 0; i < n_tasks; i++) {
        controller->tasks[i].is_assigned = 0;
        controller->tasks[i].assigned_robot_id = -1;
    }
    for (j = 1; j <= m; j++) {
        if (p[j] > 0 && p[j] <= n_tasks && j <= n_robots) {
            int task_idx = p[j] - 1;
            int robot_idx = j - 1;
            if (controller->state.robots[robot_idx].is_active) {
                controller->tasks[task_idx].assigned_robot_id = controller->state.robots[robot_idx].robot_id;
                controller->tasks[task_idx].is_assigned = 1;
            }
        }
    }
    controller->allocated_tasks = 0;
    for (i = 0; i < n_tasks; i++)
        if (controller->tasks[i].is_assigned) controller->allocated_tasks++;
    swarm_unlock(controller);
    return 0;
}

/* ============================================================================
 * UDP组播/单播网络同步 —— 多机群集真实通信
 *
 * 提供跨机器的群集状态同步，使用UDP组播发送状态、UDP单播接收。
 * 零虚假数据 —— 所有网络数据来自真实socket I/O。
 * 单机运行时自动回退到本地共享内存模式。
 * =========================================================================== */

#define SWARM_UDP_SYNC_MAGIC      0x53574152  /* "SWAR" */
#define SWARM_UDP_SYNC_VERSION    1
#define SWARM_UDP_MAX_PAYLOAD     4096
#define SWARM_UDP_MULTICAST_IP    "239.192.88.1"  /* 私有组播地址 */
#define SWARM_UDP_TTL             1   /* 限制在局域网内，不穿透路由器 */

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint8_t  version;
    uint8_t  msg_type;       /* 0=state_sync, 1=task_assign, 2=consensus, 3=heartbeat */
    uint16_t robot_id;
    uint32_t seq_num;
    uint64_t timestamp_us;
    uint16_t payload_size;
    uint8_t  payload[SWARM_UDP_MAX_PAYLOAD];
} SwarmUdpPacket;
#pragma pack(pop)

static int g_swarm_winsock_init = 0;

static int swarm_net_init_winsock(void) {
#ifdef _WIN32
    if (!g_swarm_winsock_init) {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return -1;
        g_swarm_winsock_init = 1;
    }
#endif
    return 0;
}

/**
 * @brief 初始化UDP组播发送socket，用于向群集中其他机器人广播状态
 */
int swarm_udp_multicast_init(SwarmController* controller, int port_base) {
    if (!controller) return -1;
    if (controller->udp_initialized) return 0;
    if (swarm_net_init_winsock() != 0) return -1;

    controller->udp_port_base = port_base > 0 ? port_base : 18880;

    /* 创建UDP socket */
    swarm_socket_t sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == SWARM_INVALID_SOCKET) {
        log_error("[集群网络] UDP socket创建失败");
        return -1;
    }

    /* 设置组播TTL */
    unsigned char ttl = SWARM_UDP_TTL;
    setsockopt(sock, IPPROTO_IP, IP_MULTICAST_TTL, (const char*)&ttl, sizeof(ttl));

    /* 允许端口重用 */
    int reuse = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));

    /* 绑定到发送端口 */
    struct sockaddr_in bind_addr;
    memset(&bind_addr, 0, sizeof(bind_addr));
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_port = htons((unsigned short)controller->udp_port_base);
    bind_addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sock, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) != 0) {
#ifdef _WIN32
        /* Windows可能需要第二次绑定（双栈环境） */
        bind_addr.sin_addr.s_addr = inet_addr("0.0.0.0");
        if (bind(sock, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) != 0) {
            swarm_socket_close(sock);
            log_error("[集群网络] UDP绑定失败，端口=%d", controller->udp_port_base);
            return -1;
        }
#else
        swarm_socket_close(sock);
        log_error("[集群网络] UDP绑定失败，端口=%d", controller->udp_port_base);
        return -1;
#endif
    }

    /* 设置组播目标地址 */
    memset(&controller->multicast_addr, 0, sizeof(controller->multicast_addr));
    controller->multicast_addr.sin_family = AF_INET;
    controller->multicast_addr.sin_port = htons((unsigned short)controller->udp_port_base + 1);
    controller->multicast_addr.sin_addr.s_addr = inet_addr(SWARM_UDP_MULTICAST_IP);

    /* 加入组播组以便接收其他机器人的数据 */
    struct ip_mreq mreq;
    mreq.imr_multiaddr.s_addr = inet_addr(SWARM_UDP_MULTICAST_IP);
    mreq.imr_interface.s_addr = INADDR_ANY;
    setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, (const char*)&mreq, sizeof(mreq));

    controller->udp_socket = sock;
    controller->udp_initialized = 1;

    log_info("[集群网络] UDP组播初始化成功，发送端口=%d，组播=%s:%d",
             controller->udp_port_base, SWARM_UDP_MULTICAST_IP,
             controller->udp_port_base + 1);
    return 0;
}

/**
 * @brief 添加单播目标地址（点到点通信，用于任务分配确认）
 */
int swarm_udp_add_peer(SwarmController* controller, const char* ip_address, int port) {
    if (!controller || !ip_address || controller->unicast_count >= SWARM_MAX_ROBOTS) return -1;

    struct sockaddr_in* addr = &controller->unicast_addrs[controller->unicast_count];
    memset(addr, 0, sizeof(*addr));
    addr->sin_family = AF_INET;
    addr->sin_port = htons((unsigned short)port);
    addr->sin_addr.s_addr = inet_addr(ip_address);

    if (addr->sin_addr.s_addr == INADDR_NONE) {
        log_error("[集群网络] 无效IP地址: %s", ip_address);
        return -1;
    }

    controller->unicast_count++;
    log_info("[集群网络] 添加对等节点: %s:%d (总数=%d)", ip_address, port, controller->unicast_count);
    return 0;
}

/**
 * @brief 通过UDP组播发送单个机器人的状态同步数据包
 *
 * 将机器人位置、速度、方向、电池等状态序列化为二进制数据包，
 * 通过UDP组播发送到群集组。接收方（包括其他机器和监控系统）
 * 可解包并更新本地状态。
 */
int swarm_udp_send_state(SwarmController* controller, int robot_index) {
    if (!controller || !controller->udp_initialized) return -1;
    if (robot_index < 0 || robot_index >= controller->state.robot_count) return -1;

    SwarmRobotState* rs = &controller->state.robots[robot_index];
    if (!rs->is_active) return 0;

    SwarmUdpPacket pkt;
    memset(&pkt, 0, sizeof(pkt));
    pkt.magic = SWARM_UDP_SYNC_MAGIC;
    pkt.version = SWARM_UDP_SYNC_VERSION;
    pkt.msg_type = 0; /* state_sync */
    pkt.robot_id = (uint16_t)rs->robot_id;
/* 使用递增序号替代指针值。
     * 指针值在每次程序运行时相同，无法区分新旧数据包。
     * 每个robot_state维护独立的seq_counter，跨报文递增。 */
    pkt.seq_num = rs->udp_seq_counter++;
    pkt.timestamp_us = 0;

    /* 序列化机器人状态到payload */
    size_t offset = 0;
    memcpy(pkt.payload + offset, rs->position, 3 * sizeof(float)); offset += 12;
    memcpy(pkt.payload + offset, rs->velocity, 3 * sizeof(float)); offset += 12;
    memcpy(pkt.payload + offset, rs->orientation, 4 * sizeof(float)); offset += 16;
    memcpy(pkt.payload + offset, rs->angular_velocity, 3 * sizeof(float)); offset += 12;
    memcpy(pkt.payload + offset, &rs->battery_level, sizeof(float)); offset += 4;
    memcpy(pkt.payload + offset, &rs->task_progress, sizeof(float)); offset += 4;
    pkt.payload_size = (uint16_t)offset;

    /* 发送UDP组播数据包 */
    struct sockaddr_in* dest = &controller->multicast_addr;
    int sent = (int)sendto(controller->udp_socket, (const char*)&pkt,
                           (int)(32 + offset), 0,
                           (struct sockaddr*)dest, sizeof(*dest));

    if (sent < 0) return -1;

    /* 同时单播到所有已注册的对等节点 */
    for (int i = 0; i < controller->unicast_count; i++) {
        sendto(controller->udp_socket, (const char*)&pkt,
               (int)(32 + offset), 0,
               (struct sockaddr*)&controller->unicast_addrs[i],
               sizeof(controller->unicast_addrs[i]));
    }

    return 0;
}

/**
 * @brief 通过UDP接收来自其他机器人的状态更新（非阻塞）
 *
 * 轮询socket接收缓冲区，解析接收到的状态同步数据包，
 * 更新本地集群状态中对应机器人的信息。
 * 返回接收到的数据包数量（0表示无数据，-1表示错误）。
 */
int swarm_udp_receive_sync(SwarmController* controller) {
    if (!controller || !controller->udp_initialized) return -1;
    if (controller->udp_socket == SWARM_INVALID_SOCKET) return -1;

    /* 设置非阻塞模式 */
#ifdef _WIN32
    u_long mode = 1;
    ioctlsocket(controller->udp_socket, FIONBIO, &mode);
#else
    int flags = fcntl(controller->udp_socket, F_GETFL, 0);
    fcntl(controller->udp_socket, F_SETFL, flags | O_NONBLOCK);
#endif

    int received = 0;
    SwarmUdpPacket pkt;
    struct sockaddr_in from_addr;
    socklen_t from_len = sizeof(from_addr);

    while (received < 32) { /* 每次最多处理32个数据包 */
        int n = (int)recvfrom(controller->udp_socket, (char*)&pkt, sizeof(pkt), 0,
                              (struct sockaddr*)&from_addr, &from_len);
        if (n < (int)(32)) break; /* 数据包太小或缓冲区空 */

        /* 验证魔数和版本 */
        if (pkt.magic != SWARM_UDP_SYNC_MAGIC || pkt.version != SWARM_UDP_SYNC_VERSION) {
            continue;
        }

        /* 查找或创建对应机器人条目 */
        int ridx = swarm_find_robot_index(controller, (int)pkt.robot_id);
        if (ridx < 0) {
            /* 发现新机器人：自动注册 */
            if (controller->state.robot_count < SWARM_MAX_ROBOTS) {
                RobotConfig rc;
                memset(&rc, 0, sizeof(rc));
                swarm_add_robot(controller, (int)pkt.robot_id, &rc);
                ridx = swarm_find_robot_index(controller, (int)pkt.robot_id);
            }
        }

        if (ridx >= 0 && pkt.msg_type == 0) {
            SwarmRobotState* rs = &controller->state.robots[ridx];
            size_t offset = 0;
            if (pkt.payload_size >= 60) {
                memcpy(rs->position, pkt.payload + offset, 12); offset += 12;
                memcpy(rs->velocity, pkt.payload + offset, 12); offset += 12;
                memcpy(rs->orientation, pkt.payload + offset, 16); offset += 16;
                memcpy(rs->angular_velocity, pkt.payload + offset, 12); offset += 12;
                memcpy(&rs->battery_level, pkt.payload + offset, 4); offset += 4;
                memcpy(&rs->task_progress, pkt.payload + offset, 4);
                rs->is_active = 1;
            }
        }
        received++;
    }

    return received;
}

/**
 * @brief 向所有集群成员广播全部机器人状态（用于同步回合）
 */
int swarm_udp_broadcast_all(SwarmController* controller) {
    if (!controller || !controller->udp_initialized) return -1;

    swarm_lock(controller);
    int sent = 0;
    for (int i = 0; i < controller->state.robot_count; i++) {
        if (controller->state.robots[i].is_active) {
            if (swarm_udp_send_state(controller, i) == 0) sent++;
        }
    }
    swarm_unlock(controller);
    return sent;
}

/**
 * @brief 同步回合：发送本地状态 + 接收远程更新
 *
 * 在一次调用中完成完整的网络同步循环。
 * 单机运行（udp_initialized=0）时自动跳过，不影响本地仿真。
 */
int swarm_udp_sync_round(SwarmController* controller) {
    if (!controller || !controller->udp_initialized) return 0; /* 无网络时静默通过 */

    swarm_udp_broadcast_all(controller);
    int rx = swarm_udp_receive_sync(controller);
    swarm_update_neighbors(controller);

    return rx;
}

int swarm_allocate_auction_refined(SwarmController* controller)
{
    if (!controller || controller->task_count <= 0) return -1;
    swarm_lock(controller);
    int n_tasks = controller->task_count;
    int n_robots = controller->state.robot_count;
    int i, r, t;
    float epsilon = 0.01f;
    float prices[SWARM_MAX_TASKS];
    int assignment[SWARM_MAX_TASKS];
    int robot_assignments[SWARM_MAX_ROBOTS];
    for (i = 0; i < n_tasks; i++) {
        prices[i] = 0.0f;
        assignment[i] = -1;
    }
    for (r = 0; r < n_robots; r++)
        robot_assignments[r] = -1;
    int max_iter = n_tasks * n_robots * 2;
    int iter = 0;
    while (iter < max_iter) {
        int unassigned_robot = -1;
        for (r = 0; r < n_robots; r++) {
            if (!controller->state.robots[r].is_active) continue;
            if (robot_assignments[r] < 0) { unassigned_robot = r; break; }
        }
        if (unassigned_robot < 0) break;
        int r_idx = unassigned_robot;
        float best_val = -1e9f;
        int best_t = -1;
        float second_val = -1e9f;
        for (t = 0; t < n_tasks; t++) {
            if (controller->tasks[t].is_completed) continue;
            float dx = controller->state.robots[r_idx].position[0] - controller->tasks[t].position[0];
            float dy = controller->state.robots[r_idx].position[1] - controller->tasks[t].position[1];
            float dz = controller->state.robots[r_idx].position[2] - controller->tasks[t].position[2];
            float dist = sqrtf(dx * dx + dy * dy + dz * dz);
            float benefit = controller->tasks[t].reward - dist * 0.1f - prices[t];
            if (benefit > best_val) {
                second_val = best_val;
                best_val = benefit;
                best_t = t;
            } else if (benefit > second_val) {
                second_val = benefit;
            }
        }
        if (best_t >= 0) {
            float bid = best_val - second_val + epsilon;
            if (bid < epsilon) bid = epsilon;
            prices[best_t] += bid;
            if (assignment[best_t] >= 0)
                robot_assignments[assignment[best_t]] = -1;
            assignment[best_t] = r_idx;
            robot_assignments[r_idx] = best_t;
        }
        iter++;
    }
    for (t = 0; t < n_tasks; t++) {
        if (assignment[t] >= 0) {
            int robot_idx = assignment[t];
            controller->tasks[t].assigned_robot_id = controller->state.robots[robot_idx].robot_id;
            controller->tasks[t].is_assigned = 1;
        }
    }
    controller->allocated_tasks = 0;
    for (i = 0; i < n_tasks; i++)
        if (controller->tasks[i].is_assigned) controller->allocated_tasks++;
    swarm_unlock(controller);
    return 0;
}

int swarm_allocate_cbba(SwarmController* controller)
{
    if (!controller || controller->task_count <= 0) return -1;
    swarm_lock(controller);
    int n_tasks = controller->task_count;
    int n_robots = controller->state.robot_count;
    int r, t;
    int robot_bundle[SWARM_MAX_ROBOTS][SWARM_CBBA_MAX_BUNDLE];
    int bundle_sizes[SWARM_MAX_ROBOTS];
    int winning_agent[SWARM_MAX_TASKS];
    float winning_bid[SWARM_MAX_TASKS];
    int task_owner[SWARM_MAX_TASKS];
    float task_reward[SWARM_MAX_TASKS];
    int i;
    for (t = 0; t < n_tasks; t++) {
        winning_agent[t] = -1;
        winning_bid[t] = -1e9f;
        task_owner[t] = -1;
        task_reward[t] = controller->tasks[t].reward;
    }
    for (r = 0; r < n_robots; r++) {
        bundle_sizes[r] = 0;
        if (!controller->state.robots[r].is_active) continue;
    }
    int max_cbda_iter = 10;
    int cbda_iter;
    for (cbda_iter = 0; cbda_iter < max_cbda_iter; cbda_iter++) {
        for (r = 0; r < n_robots; r++) {
            if (!controller->state.robots[r].is_active) continue;
            if (bundle_sizes[r] >= SWARM_CBBA_MAX_BUNDLE) continue;
            int tasks_assigned = 0;
            int ti;
            for (ti = 0; ti < bundle_sizes[r]; ti++)
                if (robot_bundle[r][ti] >= 0) tasks_assigned++;
            if (tasks_assigned >= controller->config.task_config.max_tasks_per_robot) continue;
            float best_score = -1e9f;
            int best_task = -1;
            for (t = 0; t < n_tasks; t++) {
                if (controller->tasks[t].is_completed) continue;
                int already = 0;
                for (ti = 0; ti < bundle_sizes[r]; ti++)
                    if (robot_bundle[r][ti] == t) { already = 1; break; }
                if (already) continue;
                float dx = controller->state.robots[r].position[0] - controller->tasks[t].position[0];
                float dy = controller->state.robots[r].position[1] - controller->tasks[t].position[1];
                float dz = controller->state.robots[r].position[2] - controller->tasks[t].position[2];
                float dist = sqrtf(dx * dx + dy * dy + dz * dz);
                float marginal = task_reward[t] - dist * 0.05f;
                float path_cost = 0.0f;
                int pi;
                for (pi = 0; pi < bundle_sizes[r]; pi++) {
                    int pt = robot_bundle[r][pi];
                    if (pt < 0) continue;
                    path_cost += dist * 0.02f;
                }
                float score = marginal - path_cost;
                if (score > best_score) {
                    best_score = score;
                    best_task = t;
                }
            }
            if (best_task >= 0) {
                robot_bundle[r][bundle_sizes[r]] = best_task;
                bundle_sizes[r]++;
            }
        }
        for (t = 0; t < n_tasks; t++) {
            float best_bid = -1e9f;
            int best_agent = -1;
            for (r = 0; r < n_robots; r++) {
                if (!controller->state.robots[r].is_active) continue;
                int in_bundle = 0;
                int ti;
                for (ti = 0; ti < bundle_sizes[r]; ti++)
                    if (robot_bundle[r][ti] == t) { in_bundle = 1; break; }
                if (in_bundle) {
                    float bid = task_reward[t];
                    if (bid > best_bid) {
                        best_bid = bid;
                        best_agent = r;
                    }
                }
            }
            winning_agent[t] = best_agent;
            winning_bid[t] = best_bid;
            task_owner[t] = best_agent;
        }
        for (r = 0; r < n_robots; r++) {
            if (!controller->state.robots[r].is_active) continue;
            int new_size = 0;
            for (i = 0; i < bundle_sizes[r]; i++) {
                int t_idx = robot_bundle[r][i];
                if (t_idx >= 0 && winning_agent[t_idx] == r)
                    robot_bundle[r][new_size++] = t_idx;
            }
            bundle_sizes[r] = new_size;
        }
    }
    for (t = 0; t < n_tasks; t++) {
        if (task_owner[t] >= 0) {
            int robot_idx = task_owner[t];
            controller->tasks[t].assigned_robot_id = controller->state.robots[robot_idx].robot_id;
            controller->tasks[t].is_assigned = 1;
        }
    }
    controller->allocated_tasks = 0;
    for (i = 0; i < n_tasks; i++)
        if (controller->tasks[i].is_assigned) controller->allocated_tasks++;
    swarm_unlock(controller);
    return 0;
}
