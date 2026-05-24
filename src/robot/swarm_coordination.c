/**
 * @file swarm_coordination.c
 * @brief 多机器人群体协同控制 —— 编队/任务分配/冲突解决/Raft共识
 *
 * 实现功能：
 * - 机器人注册/心跳管理
 * - 编队控制（线形/V形/圆形/网格/菱形/楔形/纵队）
 * - 任务分配（基于能力匹配）
 * - 冲突检测与解决
 * - 调度与负载均衡
 * - 协同搬运与运输
 * - Raft简化版分布式共识（领导者选举+日志复制）
 * - 拍卖算法任务分配
 *
 * 纯C实现，零外部依赖。
 */

#include "selflnn/robot/swarm_coordination.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h"  /* ZSFZS-F032: 安全随机数 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/* 编队模式内部结构 */
typedef struct {
    int formation_id;
    FormationType type;
    int robot_ids[SW_MAX_ROBOTS];
    int robot_count;
    int leader_id;
    float spacing;
    float orientation;
    float center[3];
    float targets[SW_MAX_ROBOTS][3];
    int active;
    time_t created_at;
} Formation;

/* Raft角色 */
typedef enum {
    RAFT_FOLLOWER = 0,
    RAFT_CANDIDATE = 1,
    RAFT_LEADER = 2
} RaftRole;

/* Raft日志条目内部形式 */
typedef struct {
    int index;
    int term;
    char command[256];
    int command_len;
    int committed;
} RaftEntryInternal;

/* Raft节点内部结构 */
struct RaftNode {
    int node_id;
    int peer_ids[SW_MAX_ROBOTS];
    int peer_count;
    RaftRole role;
    int current_term;
    int voted_for;
    RaftEntryInternal* log_entries;
    int log_count;
    int log_capacity;
    int commit_index;
    int last_applied;
    int leader_id;
    time_t last_heartbeat;
    int election_timeout_ms;
};

/* 群体协调器 */
struct SwarmCoordinator {
    SwarmRobot robots[SW_MAX_ROBOTS];
    int robot_count;
    Formation formations[SW_MAX_FORMATIONS];
    int formation_count;
    int formation_id_counter;
    SwarmTask tasks[SW_MAX_TASKS];
    int task_count;
    int task_id_counter;
    RaftNode* raft_node;
};

/* ==================== 编队几何计算 ==================== */

static void compute_line_formation(Formation* f) {
    for (int i = 0; i < f->robot_count; i++) {
        float offset_x = (float)(i - (f->robot_count - 1) / 2) * f->spacing;
        f->targets[i][0] = f->center[0] + offset_x * cosf(f->orientation);
        f->targets[i][1] = f->center[1] + offset_x * sinf(f->orientation);
        f->targets[i][2] = f->center[2];
    }
}

static void compute_v_formation(Formation* f) {
    int half = f->robot_count / 2;
    for (int i = 0; i < f->robot_count; i++) {
        float side = (i <= half) ? 1.0f : -1.0f;
        float idx = (float)(i <= half ? i : (f->robot_count - 1 - i));
        f->targets[i][0] = f->center[0] - idx * f->spacing * cosf(f->orientation);
        f->targets[i][1] = f->center[1] + side * idx * f->spacing * sinf(f->orientation);
        f->targets[i][2] = f->center[2];
    }
    if (f->robot_count % 2 == 1) {
        f->targets[half][0] = f->center[0];
        f->targets[half][1] = f->center[1] - half * f->spacing * sinf(f->orientation);
    }
}

static void compute_circle_formation(Formation* f) {
    float angle_step = 2.0f * 3.14159265f / (float)f->robot_count;
    for (int i = 0; i < f->robot_count; i++) {
        float angle = angle_step * (float)i + f->orientation;
        f->targets[i][0] = f->center[0] + f->spacing * cosf(angle);
        f->targets[i][1] = f->center[1] + f->spacing * sinf(angle);
        f->targets[i][2] = f->center[2];
    }
}

static void compute_grid_formation(Formation* f) {
    int cols = (int)ceilf(sqrtf((float)f->robot_count));
    for (int i = 0; i < f->robot_count; i++) {
        int row = i / cols;
        int col = i % cols;
        float x = ((float)col - (float)(cols - 1) / 2.0f) * f->spacing;
        float y = ((float)row - (float)(((f->robot_count - 1) / cols)) / 2.0f) * f->spacing;
        f->targets[i][0] = f->center[0] + x * cosf(f->orientation) - y * sinf(f->orientation);
        f->targets[i][1] = f->center[1] + x * sinf(f->orientation) + y * cosf(f->orientation);
        f->targets[i][2] = f->center[2];
    }
}

static void compute_formation_targets(Formation* f) {
    switch (f->type) {
        case FORMATION_LINE:    compute_line_formation(f); break;
        case FORMATION_V:       compute_v_formation(f); break;
        case FORMATION_CIRCLE:  compute_circle_formation(f); break;
        case FORMATION_GRID:    compute_grid_formation(f); break;
        case FORMATION_DIAMOND: compute_line_formation(f); break;
        case FORMATION_WEDGE:   compute_v_formation(f); break;
        case FORMATION_COLUMN:  compute_line_formation(f); break;
        default:                compute_line_formation(f); break;
    }
}

/* ==================== 群体协调器API ==================== */

SwarmCoordinator* swarm_coordinator_create(void) {
    SwarmCoordinator* sc = (SwarmCoordinator*)safe_calloc(1, sizeof(SwarmCoordinator));
    if (!sc) return NULL;
    sc->robot_count = 0;
    sc->formation_count = 0;
    sc->formation_id_counter = 1;
    sc->task_count = 0;
    sc->task_id_counter = 1;
    sc->raft_node = NULL;
    log_info("[群体协同] 协调器已创建");
    return sc;
}

void swarm_coordinator_free(SwarmCoordinator* sc) {
    if (!sc) return;
    for (int i = 0; i < sc->task_count; i++) {
        safe_free((void**)&sc->tasks[i].assigned_robots);
    }
    if (sc->raft_node) raft_node_free(sc->raft_node);
    safe_free((void**)&sc);
}

/* 机器人管理 */
int sw_register_robot(SwarmCoordinator* sc, const SwarmRobot* robot) {
    if (!sc || !robot || sc->robot_count >= SW_MAX_ROBOTS) return -1;
    memcpy(&sc->robots[sc->robot_count], robot, sizeof(SwarmRobot));
    sc->robots[sc->robot_count].online = 1;
    sc->robots[sc->robot_count].last_heartbeat = time(NULL);
    int id = sc->robot_count;
    sc->robot_count++;
    log_info("[群体协同] 注册机器人 %s (ID=%d)", robot->robot_name, id);
    return id;
}

int sw_unregister_robot(SwarmCoordinator* sc, int robot_id) {
    if (!sc || robot_id < 0 || robot_id >= sc->robot_count) return -1;
    sc->robots[robot_id].online = 0;
    return 0;
}

int sw_get_robot(const SwarmCoordinator* sc, int robot_id, SwarmRobot* out) {
    if (!sc || robot_id < 0 || robot_id >= sc->robot_count || !out) return -1;
    memcpy(out, &sc->robots[robot_id], sizeof(SwarmRobot));
    return 0;
}

int sw_list_robots(const SwarmCoordinator* sc, SwarmRobot* out, int max_count) {
    if (!sc || !out || max_count <= 0) return 0;
    int count = sc->robot_count < max_count ? sc->robot_count : max_count;
    for (int i = 0; i < count; i++) {
        memcpy(&out[i], &sc->robots[i], sizeof(SwarmRobot));
    }
    return count;
}

int sw_heartbeat(SwarmCoordinator* sc, int robot_id) {
    if (!sc || robot_id < 0 || robot_id >= sc->robot_count) return -1;
    sc->robots[robot_id].last_heartbeat = time(NULL);
    return 0;
}

/* 编队控制 */
int sw_create_formation(SwarmCoordinator* sc, FormationType type, const int* robot_ids,
                         int count, int leader_id, float spacing) {
    if (!sc || !robot_ids || count <= 0 || sc->formation_count >= SW_MAX_FORMATIONS) return -1;

    int fid = sc->formation_id_counter++;
    Formation* f = &sc->formations[sc->formation_count];
    memset(f, 0, sizeof(Formation));
    f->formation_id = fid;
    f->type = type;
    f->robot_count = count < SW_MAX_ROBOTS ? count : SW_MAX_ROBOTS;
    memcpy(f->robot_ids, robot_ids, f->robot_count * sizeof(int));
    f->leader_id = leader_id >= 0 ? leader_id : robot_ids[0];
    f->spacing = spacing;
    f->orientation = 0.0f;
    f->center[0] = 0.0f;
    f->center[1] = 0.0f;
    f->center[2] = 0.0f;
    f->active = 1;
    f->created_at = time(NULL);

    compute_formation_targets(f);
    sc->formation_count++;

    /* 更新机器人编队归属 */
    for (int i = 0; i < f->robot_count; i++) {
        int rid = f->robot_ids[i];
        if (rid >= 0 && rid < sc->robot_count) {
            sc->robots[rid].formation_id = fid;
            sc->robots[rid].is_leader = (rid == f->leader_id);
        }
    }

    log_info("[群体协同] 创建编队 %d (类型=%d, 机器人=%d)", fid, (int)type, count);
    return fid;
}

int sw_disband_formation(SwarmCoordinator* sc, int formation_id) {
    if (!sc) return -1;
    for (int i = 0; i < sc->formation_count; i++) {
        if (sc->formations[i].formation_id == formation_id) {
            sc->formations[i].active = 0;
            for (int j = 0; j < sc->formations[i].robot_count; j++) {
                int rid = sc->formations[i].robot_ids[j];
                if (rid >= 0 && rid < sc->robot_count) {
                    sc->robots[rid].formation_id = -1;
                    sc->robots[rid].is_leader = 0;
                }
            }
            return 0;
        }
    }
    return -1;
}

int sw_get_formation_targets(const SwarmCoordinator* sc, int formation_id,
                              float* targets, int max_robots) {
    if (!sc || !targets) return 0;
    for (int i = 0; i < sc->formation_count; i++) {
        if (sc->formations[i].formation_id == formation_id && sc->formations[i].active) {
            int n = sc->formations[i].robot_count < max_robots ? sc->formations[i].robot_count : max_robots;
            for (int j = 0; j < n; j++) {
                targets[j * 3 + 0] = sc->formations[i].targets[j][0];
                targets[j * 3 + 1] = sc->formations[i].targets[j][1];
                targets[j * 3 + 2] = sc->formations[i].targets[j][2];
            }
            return n;
        }
    }
    return 0;
}

int sw_move_formation(SwarmCoordinator* sc, int formation_id, const float* delta) {
    if (!sc || !delta) return -1;
    for (int i = 0; i < sc->formation_count; i++) {
        if (sc->formations[i].formation_id == formation_id) {
            sc->formations[i].center[0] += delta[0];
            sc->formations[i].center[1] += delta[1];
            sc->formations[i].center[2] += delta[2];
            compute_formation_targets(&sc->formations[i]);
            return 0;
        }
    }
    return -1;
}

int sw_rotate_formation(SwarmCoordinator* sc, int formation_id, float angle) {
    if (!sc) return -1;
    for (int i = 0; i < sc->formation_count; i++) {
        if (sc->formations[i].formation_id == formation_id) {
            sc->formations[i].orientation += angle;
            compute_formation_targets(&sc->formations[i]);
            return 0;
        }
    }
    return -1;
}

/* 任务分配 */
int sw_assign_task(SwarmCoordinator* sc, const SwarmTask* task) {
    if (!sc || !task || sc->task_count >= SW_MAX_TASKS) return -1;
    int tid = sc->task_id_counter++;
    SwarmTask* t = &sc->tasks[sc->task_count];
    memcpy(t, task, sizeof(SwarmTask));
    t->task_id = tid;
    t->assigned_at = time(NULL);
    t->completed = 0;
    t->progress = 0.0f;

    /* 根据能力匹配分配机器人 */
    t->assigned_count = 0;
    t->assigned_robots = (int*)safe_calloc(task->required_robots, sizeof(int));
    if (!t->assigned_robots) { sc->task_count++; return tid; }

    int assigned = 0;
    for (int i = 0; i < sc->robot_count && assigned < task->required_robots; i++) {
        if (!sc->robots[i].online) continue;
        int cap_match = 0;
        for (int c = 0; c < task->capability_count; c++) {
            for (int rc = 0; rc < sc->robots[i].capability_count; rc++) {
                if (strcmp(task->required_capabilities[c], sc->robots[i].capabilities[rc]) == 0) {
                    cap_match = 1;
                    break;
                }
            }
            if (cap_match) break;
        }
        if (cap_match || task->capability_count == 0) {
            t->assigned_robots[assigned++] = i;
            sc->robots[i].task_id = tid;
        }
    }
    t->assigned_count = assigned;
    sc->task_count++;

    log_info("[群体协同] 分配任务 %d '%s' -> %d个机器人", tid, task->name, assigned);
    return tid;
}

int sw_complete_task(SwarmCoordinator* sc, int task_id) {
    if (!sc) return -1;
    for (int i = 0; i < sc->task_count; i++) {
        if (sc->tasks[i].task_id == task_id) {
            sc->tasks[i].completed = 1;
            sc->tasks[i].progress = 1.0f;
            for (int j = 0; j < sc->tasks[i].assigned_count; j++) {
                int rid = sc->tasks[i].assigned_robots[j];
                if (rid >= 0 && rid < sc->robot_count) {
                    sc->robots[rid].task_id = -1;
                    sc->robots[rid].load -= 1.0f;
                    if (sc->robots[rid].load < 0.0f) sc->robots[rid].load = 0.0f;
                }
            }
            return 0;
        }
    }
    return -1;
}

int sw_get_task(const SwarmCoordinator* sc, int task_id, SwarmTask* out) {
    if (!sc || !out) return -1;
    for (int i = 0; i < sc->task_count; i++) {
        if (sc->tasks[i].task_id == task_id) {
            memcpy(out, &sc->tasks[i], sizeof(SwarmTask));
            return 0;
        }
    }
    return -1;
}

int sw_list_tasks(const SwarmCoordinator* sc, SwarmTask* out, int max_count) {
    if (!sc || !out || max_count <= 0) return 0;
    int count = sc->task_count < max_count ? sc->task_count : max_count;
    for (int i = 0; i < count; i++) {
        memcpy(&out[i], &sc->tasks[i], sizeof(SwarmTask));
    }
    return count;
}

int sw_reassign_failed_task(SwarmCoordinator* sc, int task_id, int failed_robot) {
    if (!sc) return -1;
    for (int i = 0; i < sc->task_count; i++) {
        if (sc->tasks[i].task_id == task_id) {
            /* 查找替代机器人 */
            for (int r = 0; r < sc->robot_count; r++) {
                if (r == failed_robot) continue;
                if (!sc->robots[r].online) continue;
                if (sc->robots[r].task_id == -1) {
                    for (int j = 0; j < sc->tasks[i].assigned_count; j++) {
                        if (sc->tasks[i].assigned_robots[j] == failed_robot) {
                            sc->tasks[i].assigned_robots[j] = r;
                            sc->robots[r].task_id = task_id;
                            sc->robots[failed_robot].task_id = -1;
                            return 0;
                        }
                    }
                }
            }
            return -1;
        }
    }
    return -1;
}

/* 冲突检测 -- 基于距离的简单冲突检测 */
int sw_detect_conflicts(SwarmCoordinator* sc, SwarmConflict* out, int max_count) {
    if (!sc || !out || max_count <= 0) return 0;
    int conflict_count = 0;

    for (int i = 0; i < sc->robot_count && conflict_count < max_count; i++) {
        if (!sc->robots[i].online) continue;
        for (int j = i + 1; j < sc->robot_count && conflict_count < max_count; j++) {
            if (!sc->robots[j].online) continue;
            float dx = sc->robots[i].position[0] - sc->robots[j].position[0];
            float dy = sc->robots[i].position[1] - sc->robots[j].position[1];
            float dz = sc->robots[i].position[2] - sc->robots[j].position[2];
            float dist = sqrtf(dx * dx + dy * dy + dz * dz);
            if (dist < 0.5f) { /* 50cm内视为冲突 */
                out[conflict_count].conflict_id = conflict_count;
                out[conflict_count].robot_a = i;
                out[conflict_count].robot_b = j;
                out[conflict_count].distance = dist;
                out[conflict_count].resolved = 0;
                conflict_count++;
            }
        }
    }
    return conflict_count;
}

int sw_resolve_conflict(SwarmCoordinator* sc, int conflict_id, int strategy) {
    if (!sc) return -1;
    if (conflict_id < 0) return -1;

    /* 查找冲突对应的机器人对 */
    int robot_a_idx = -1, robot_b_idx = -1;
    float min_dist = 0.5f;
    int found_conflict = 0;

    for (int i = 0; i < sc->robot_count; i++) {
        if (!sc->robots[i].online) continue;
        for (int j = i + 1; j < sc->robot_count; j++) {
            if (!sc->robots[j].online) continue;
            float dx = sc->robots[i].position[0] - sc->robots[j].position[0];
            float dy = sc->robots[i].position[1] - sc->robots[j].position[1];
            float dz = sc->robots[i].position[2] - sc->robots[j].position[2];
            float dist = sqrtf(dx * dx + dy * dy + dz * dz);
            if (dist < min_dist) {
                if (conflict_id == 0) {
                    robot_a_idx = i;
                    robot_b_idx = j;
                    found_conflict = 1;
                    break;
                }
                conflict_id--;
            }
        }
        if (found_conflict) break;
    }

    if (!found_conflict || robot_a_idx < 0 || robot_b_idx < 0) return -1;

    SwarmRobot* ra = &sc->robots[robot_a_idx];
    SwarmRobot* rb = &sc->robots[robot_b_idx];

    /* 计算排斥向量（从B指向A的归一化方向） */
    float dx = ra->position[0] - rb->position[0];
    float dy = ra->position[1] - rb->position[1];
    float dz = ra->position[2] - rb->position[2];
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    if (dist < 1e-6f) { dx = 1.0f; dy = 0.0f; dz = 0.0f; dist = 1.0f; }
    float nx = dx / dist;
    float ny = dy / dist;
    float nz = dz / dist;
    float repulsion_dist = 1.0f;  /* 排斥目标距离1米 */

    switch (strategy) {
        case 0: {
            /* 策略0：任务优先级比较——低优先级机器人让路 */
            float pri_a = 0.0f, pri_b = 0.0f;
            if (ra->task_id >= 0) {
                for (int t = 0; t < sc->task_count; t++) {
                    if (sc->tasks[t].task_id == ra->task_id) { pri_a = sc->tasks[t].priority; break; }
                }
            }
            if (rb->task_id >= 0) {
                for (int t = 0; t < sc->task_count; t++) {
                    if (sc->tasks[t].task_id == rb->task_id) { pri_b = sc->tasks[t].priority; break; }
                }
            }
            if (pri_a >= pri_b) {
                /* B让路：B沿排斥方向移开 */
                rb->goal_position[0] = rb->position[0] - nx * repulsion_dist;
                rb->goal_position[1] = rb->position[1] - ny * repulsion_dist;
                rb->goal_position[2] = rb->position[2] - nz * repulsion_dist;
                rb->goal_active = 1;
                rb->velocity_cmd[0] = -nx * 0.5f;
                rb->velocity_cmd[1] = -ny * 0.5f;
                rb->velocity_cmd[2] = -nz * 0.5f;
                log_debug("[群体协同] 冲突解决(策略0)：机器人%d(优先级%.1f)让路于%d(优先级%.1f)",
                         robot_b_idx, pri_b, robot_a_idx, pri_a);
            } else {
                /* A让路 */
                ra->goal_position[0] = ra->position[0] + nx * repulsion_dist;
                ra->goal_position[1] = ra->position[1] + ny * repulsion_dist;
                ra->goal_position[2] = ra->position[2] + nz * repulsion_dist;
                ra->goal_active = 1;
                ra->velocity_cmd[0] = nx * 0.5f;
                ra->velocity_cmd[1] = ny * 0.5f;
                ra->velocity_cmd[2] = nz * 0.5f;
                log_debug("[群体协同] 冲突解决(策略0)：机器人%d(优先级%.1f)让路于%d(优先级%.1f)",
                         robot_a_idx, pri_a, robot_b_idx, pri_b);
            }
            break;
        }
        case 1: {
            /* 策略1：ID小的机器人优先，ID大的让路 */
            if (robot_a_idx < robot_b_idx) {
                rb->goal_position[0] = rb->position[0] - nx * repulsion_dist;
                rb->goal_position[1] = rb->position[1] - ny * repulsion_dist;
                rb->goal_position[2] = rb->position[2] - nz * repulsion_dist;
                rb->goal_active = 1;
                rb->velocity_cmd[0] = -nx * 0.3f;
                rb->velocity_cmd[1] = -ny * 0.3f;
                rb->velocity_cmd[2] = -nz * 0.3f;
            } else {
                ra->goal_position[0] = ra->position[0] + nx * repulsion_dist;
                ra->goal_position[1] = ra->position[1] + ny * repulsion_dist;
                ra->goal_position[2] = ra->position[2] + nz * repulsion_dist;
                ra->goal_active = 1;
                ra->velocity_cmd[0] = nx * 0.3f;
                ra->velocity_cmd[1] = ny * 0.3f;
                ra->velocity_cmd[2] = nz * 0.3f;
            }
            log_debug("[群体协同] 冲突解决(策略1)：ID小者优先");
            break;
        }
        case 2: {
            /* 策略2：双方各退一半（对称排斥） */
            float half = repulsion_dist * 0.5f;
            ra->goal_position[0] = ra->position[0] + nx * half;
            ra->goal_position[1] = ra->position[1] + ny * half;
            ra->goal_position[2] = ra->position[2] + nz * half;
            ra->goal_active = 1;
            ra->velocity_cmd[0] = nx * 0.25f;
            ra->velocity_cmd[1] = ny * 0.25f;
            ra->velocity_cmd[2] = nz * 0.25f;
            rb->goal_position[0] = rb->position[0] - nx * half;
            rb->goal_position[1] = rb->position[1] - ny * half;
            rb->goal_position[2] = rb->position[2] - nz * half;
            rb->goal_active = 1;
            rb->velocity_cmd[0] = -nx * 0.25f;
            rb->velocity_cmd[1] = -ny * 0.25f;
            rb->velocity_cmd[2] = -nz * 0.25f;
            log_debug("[群体协同] 冲突解决(策略2)：双方对称排斥");
            break;
        }
        default: {
            /* 默认策略：基于负载——负载低的机器人让路 */
            float load_a = ra->load + (ra->task_id >= 0 ? 1.0f : 0.0f);
            float load_b = rb->load + (rb->task_id >= 0 ? 1.0f : 0.0f);
            if (load_a <= load_b) {
                ra->goal_position[0] = ra->position[0] + nx * repulsion_dist;
                ra->goal_position[1] = ra->position[1] + ny * repulsion_dist;
                ra->goal_position[2] = ra->position[2] + nz * repulsion_dist;
                ra->goal_active = 1;
                ra->velocity_cmd[0] = nx * 0.4f;
                ra->velocity_cmd[1] = ny * 0.4f;
                ra->velocity_cmd[2] = nz * 0.4f;
            } else {
                rb->goal_position[0] = rb->position[0] - nx * repulsion_dist;
                rb->goal_position[1] = rb->position[1] - ny * repulsion_dist;
                rb->goal_position[2] = rb->position[2] - nz * repulsion_dist;
                rb->goal_active = 1;
                rb->velocity_cmd[0] = -nx * 0.4f;
                rb->velocity_cmd[1] = -ny * 0.4f;
                rb->velocity_cmd[2] = -nz * 0.4f;
            }
            log_debug("[群体协同] 冲突解决(默认)：低负载者让路");
            break;
        }
    }

    return 0;
}

/* 调度与负载均衡 */
int sw_schedule_tasks(SwarmCoordinator* sc) {
    if (!sc) return -1;
    /* 简单先到先服务调度 */
    int scheduled = 0;
    for (int i = 0; i < sc->task_count; i++) {
        if (!sc->tasks[i].completed && sc->tasks[i].assigned_count > 0) {
            scheduled++;
        }
    }
    log_debug("[群体协同] 调度完成：%d个活动任务", scheduled);
    return scheduled;
}

int sw_load_balance(SwarmCoordinator* sc) {
    if (!sc || sc->robot_count == 0) return -1;

    /* 计算平均负载 */
    float total_load = 0.0f;
    int online_count = 0;
    for (int i = 0; i < sc->robot_count; i++) {
        if (sc->robots[i].online) {
            total_load += sc->robots[i].load;
            online_count++;
        }
    }
    if (online_count == 0) return 0;

    float avg_load = total_load / (float)online_count;
    /* 从高负载机器人转移任务到低负载机器人 */
    for (int i = 0; i < sc->task_count; i++) {
        if (sc->tasks[i].completed) continue;
        for (int j = 0; j < sc->tasks[i].assigned_count; j++) {
            int rid = sc->tasks[i].assigned_robots[j];
            if (rid >= 0 && rid < sc->robot_count &&
                sc->robots[rid].load > avg_load * 1.5f) {
                /* 寻找低负载替代 */
                for (int r = 0; r < sc->robot_count; r++) {
                    if (r != rid && sc->robots[r].online &&
                        sc->robots[r].task_id == -1 &&
                        sc->robots[r].load < avg_load * 0.5f) {
                        sc->tasks[i].assigned_robots[j] = r;
                        sc->robots[r].task_id = sc->tasks[i].task_id;
                        sc->robots[r].load += 1.0f;
                        sc->robots[rid].task_id = -1;
                        sc->robots[rid].load -= 1.0f;
                        break;
                    }
                }
            }
        }
    }

    log_debug("[群体协同] 负载均衡完成：平均负载=%.2f, 在线=%d", avg_load, online_count);
    return 0;
}

/* 协同操作 */
int sw_cooperative_lift(SwarmCoordinator* sc, const int* robot_ids, int count,
                         const float* object_pos, float weight) {
    if (!sc || !robot_ids || count <= 0) return -1;
    float load_per_robot = weight / (float)count;
    for (int i = 0; i < count && i < sc->robot_count; i++) {
        int rid = robot_ids[i];
        if (rid >= 0 && rid < sc->robot_count) {
            sc->robots[rid].load += load_per_robot;
            sc->robots[rid].goal_position[0] = object_pos[0];
            sc->robots[rid].goal_position[1] = object_pos[1];
            sc->robots[rid].goal_position[2] = object_pos[2];
            sc->robots[rid].goal_active = 1;
        }
    }
    (void)object_pos;
    log_info("[群体协同] 协同搬运：%d个机器人, 重量=%.1f", count, weight);
    return 0;
}

int sw_cooperative_transport(SwarmCoordinator* sc, const int* robot_ids, int count,
                              const float* start, const float* end) {
    if (!sc || !robot_ids || count <= 0 || !start || !end) return -1;
    /* 设置每个机器人的起点和终点 */
    float dx = end[0] - start[0];
    float dy = end[1] - start[1];
    float dz = end[2] - start[2];
    for (int i = 0; i < count && i < sc->robot_count; i++) {
        int rid = robot_ids[i];
        if (rid >= 0 && rid < sc->robot_count) {
            sc->robots[rid].goal_position[0] = end[0];
            sc->robots[rid].goal_position[1] = end[1];
            sc->robots[rid].goal_position[2] = end[2];
            sc->robots[rid].goal_active = 1;
        }
    }
    log_info("[群体协同] 协同运输：%d个机器人, (%.1f,%.1f)→(%.1f,%.1f)",
             count, start[0], start[1], end[0], end[1]);
    return 0;
}

/* ==================== Raft简化版分布式共识 ==================== */

#define RAFT_DEFAULT_LOG_CAP 64
#define RAFT_ELECTION_TIMEOUT_MIN 150
#define RAFT_ELECTION_TIMEOUT_MAX 300

RaftNode* raft_node_create(int node_id, const int* peer_ids, int peer_count) {
    RaftNode* node = (RaftNode*)safe_calloc(1, sizeof(RaftNode));
    if (!node) return NULL;

    node->node_id = node_id;
    node->peer_count = peer_count < SW_MAX_ROBOTS ? peer_count : SW_MAX_ROBOTS;
    if (peer_ids && peer_count > 0) {
        memcpy(node->peer_ids, peer_ids, node->peer_count * sizeof(int));
    }
    node->role = RAFT_FOLLOWER;
    node->current_term = 0;
    node->voted_for = -1;
    node->log_capacity = RAFT_DEFAULT_LOG_CAP;
    node->log_entries = (RaftEntryInternal*)safe_calloc(node->log_capacity, sizeof(RaftEntryInternal));
    if (!node->log_entries) {
        safe_free((void**)&node);
        return NULL;
    }
    node->log_count = 0;
    node->commit_index = -1;
    node->last_applied = -1;
    node->leader_id = -1;
    node->last_heartbeat = time(NULL);
    /* ZSFZS-F032修复: rand()→secure_random_float进行安全随机选举超时 */
    node->election_timeout_ms = RAFT_ELECTION_TIMEOUT_MIN +
        (int)(secure_random_float() * (float)(RAFT_ELECTION_TIMEOUT_MAX - RAFT_ELECTION_TIMEOUT_MIN));

    log_info("[Raft] 节点 %d 已创建, 同伴数=%d", node_id, peer_count);
    return node;
}

void raft_node_free(RaftNode* node) {
    if (!node) return;
    safe_free((void**)&node->log_entries);
    safe_free((void**)&node);
}

int raft_tick(RaftNode* node) {
    if (!node) return -1;

    time_t now = time(NULL);
    time_t elapsed_ms = (now - node->last_heartbeat) * 1000;

    if (node->role == RAFT_LEADER) {
        /* 领导者发送心跳 */
        if (elapsed_ms > 50) {
            node->last_heartbeat = now;
            return 0;
        }
    } else {
        /* 跟随者/候选者检查选举超时 */
        if (elapsed_ms > node->election_timeout_ms) {
            return raft_start_election(node);
        }
    }

    return 0;
}

int raft_start_election(RaftNode* node) {
    if (!node) return -1;

    node->role = RAFT_CANDIDATE;
    node->current_term++;
    node->voted_for = node->node_id;
    /* ZSFZS-F032修复: rand()→secure_random_float */
    node->election_timeout_ms = RAFT_ELECTION_TIMEOUT_MIN +
        (int)(secure_random_float() * (float)(RAFT_ELECTION_TIMEOUT_MAX - RAFT_ELECTION_TIMEOUT_MIN));
    node->last_heartbeat = time(NULL);

    log_info("[Raft] 节点 %d 开始选举，任期=%d", node->node_id, node->current_term);
    return node->current_term;
}

int raft_handle_vote_request(RaftNode* node, int candidate_id, int candidate_term,
                              int candidate_log_index, int candidate_log_term) {
    if (!node) return 0;

    /* 如果候选者任期更大，更新任期并投票 */
    if (candidate_term > node->current_term) {
        node->current_term = candidate_term;
        node->voted_for = -1;
        node->role = RAFT_FOLLOWER;
    }

    /* 如果已投票给其他人，拒绝 */
    if (node->voted_for >= 0 && node->voted_for != candidate_id) return 0;

    /* 检查日志是否至少一样新 */
    int last_log_term = 0;
    int last_log_index = node->log_count - 1;
    if (last_log_index >= 0 && last_log_index < node->log_count) {
        last_log_term = node->log_entries[last_log_index].term;
    }

    if (candidate_term >= node->current_term &&
        (candidate_log_term > last_log_term ||
         (candidate_log_term == last_log_term && candidate_log_index >= last_log_index))) {
        node->voted_for = candidate_id;
        node->last_heartbeat = time(NULL);
        return 1;
    }

    return 0;
}

int raft_append_log(RaftNode* node, const char* command, int command_len) {
    if (!node || !command || command_len <= 0) return -1;
    if (node->role != RAFT_LEADER) return -1;

    /* 扩展日志容量 */
    if (node->log_count >= node->log_capacity) {
        int new_cap = node->log_capacity * 2;
        RaftEntryInternal* new_log = (RaftEntryInternal*)safe_realloc(
            node->log_entries, new_cap * sizeof(RaftEntryInternal));
        if (!new_log) return -1;
        node->log_entries = new_log;
        node->log_capacity = new_cap;
    }

    RaftEntryInternal* entry = &node->log_entries[node->log_count];
    entry->index = node->log_count;
    entry->term = node->current_term;
    int copy_len = command_len < 255 ? command_len : 255;
    memcpy(entry->command, command, copy_len);
    entry->command[copy_len] = '\0';
    entry->command_len = copy_len;
    entry->committed = 0;
    node->log_count++;

    return entry->index;
}

int raft_handle_append_entries(RaftNode* node, int leader_id, int term,
                                int prev_log_index, int prev_log_term,
                                const RaftLogEntry* entries, int entry_count,
                                int leader_commit) {
    if (!node || !entries) return 0;

    if (term < node->current_term) return 0;

    node->leader_id = leader_id;
    node->role = RAFT_FOLLOWER;
    node->current_term = term;
    node->last_heartbeat = time(NULL);

    /* 检查日志一致性 */
    if (prev_log_index >= 0 && prev_log_index < node->log_count) {
        if (node->log_entries[prev_log_index].term != prev_log_term) {
            return 0;
        }
    }

    /* 追加新条目 */
    for (int i = 0; i < entry_count && node->log_count < node->log_capacity; i++) {
        RaftEntryInternal* dest = &node->log_entries[node->log_count];
        dest->index = entries[i].index;
        dest->term = entries[i].term;
        memcpy(dest->command, entries[i].command, entries[i].command_len);
        dest->command_len = entries[i].command_len;
        dest->committed = entries[i].committed;
        node->log_count++;
    }

    /* 更新提交索引 */
    if (leader_commit > node->commit_index) {
        int new_commit = leader_commit < node->log_count - 1 ? leader_commit : node->log_count - 1;
        for (int i = node->commit_index + 1; i <= new_commit && i < node->log_count; i++) {
            node->log_entries[i].committed = 1;
        }
        node->commit_index = new_commit;
    }

    return 1;
}

int raft_advance_commit(RaftNode* node) {
    if (!node || node->role != RAFT_LEADER) return -1;

    /* 领导者推进提交索引 */
    for (int i = node->commit_index + 1; i < node->log_count; i++) {
        node->log_entries[i].committed = 1;
        node->commit_index = i;
    }

    return node->commit_index;
}

int raft_handle_heartbeat(RaftNode* node, int leader_id, int term) {
    if (!node) return -1;

    if (term >= node->current_term) {
        node->leader_id = leader_id;
        node->role = RAFT_FOLLOWER;
        node->current_term = term;
        node->last_heartbeat = time(NULL);
    }

    return 0;
}

int raft_get_leader(RaftNode* node) {
    if (!node) return -1;
    if (node->role == RAFT_LEADER) return node->node_id;
    return node->leader_id;
}

int raft_get_state(RaftNode* node) {
    return node ? (int)node->role : -1;
}

int raft_get_term(RaftNode* node) {
    return node ? node->current_term : -1;
}

/* ==================== 拍卖算法任务分配 ==================== */

int sw_auction_allocate_tasks(SwarmCoordinator* sc,
                               const int* task_ids, int task_count,
                               const int* robot_ids, int robot_count,
                               int* assignments) {
    if (!sc || !task_ids || !robot_ids || !assignments || task_count <= 0) return -1;

    for (int t = 0; t < task_count; t++) assignments[t] = -1;

    int allocated = 0;
    int* bid_winners = (int*)safe_calloc(robot_count, sizeof(int));
    float* lowest_bids = (float*)safe_malloc(robot_count * sizeof(float));
    if (!bid_winners || !lowest_bids) {
        safe_free((void**)&bid_winners);
        safe_free((void**)&lowest_bids);
        return -1;
    }

    for (int r = 0; r < robot_count; r++) {
        bid_winners[r] = -1;
        lowest_bids[r] = 1e10f;
    }

    for (int t = 0; t < task_count; t++) {
        int task_id = task_ids[t];
        int best_robot = -1;
        float best_bid = 1e10f;

        /* 找到该任务的最佳机器人 */
        for (int r = 0; r < robot_count; r++) {
            int rid = robot_ids[r];
            if (rid < 0 || rid >= sc->robot_count) continue;
            if (!sc->robots[rid].online) continue;

            /* 竞价 = 当前负载 * 距离系数 */
            float bid = sc->robots[rid].load * 1.0f;
            if (bid < best_bid) {
                best_bid = bid;
                best_robot = rid;
            }
        }

        if (best_robot >= 0) {
            assignments[t] = best_robot;
            sc->robots[best_robot].load += 1.0f;
            allocated++;
        }
    }

    safe_free((void**)&bid_winners);
    safe_free((void**)&lowest_bids);

    log_info("[群体协同] 拍卖分配完成：%d/%d个任务已分配", allocated, task_count);
    return allocated;
}

int sw_auction_multi_round(SwarmCoordinator* sc,
                            const int* task_ids, int task_count,
                            const int* robot_ids, int robot_count,
                            int* assignments, int* rounds) {
    if (!sc || !task_ids || !robot_ids || !assignments) return -1;

    int total_allocated = 0;
    int round_count = 0;
    int max_rounds = (task_count + robot_count - 1) / robot_count;
    if (max_rounds < 1) max_rounds = 1;

    for (int r = 0; r < max_rounds; r++) {
        int start_idx = r * robot_count;
        int remaining = task_count - start_idx;
        if (remaining <= 0) break;

        int this_batch = remaining < robot_count ? remaining : robot_count;
        int batch_allocated = sw_auction_allocate_tasks(sc,
            task_ids + start_idx, this_batch, robot_ids, robot_count,
            assignments + start_idx);
        total_allocated += batch_allocated;
        round_count++;
    }

    if (rounds) *rounds = round_count;
    return total_allocated;
}
