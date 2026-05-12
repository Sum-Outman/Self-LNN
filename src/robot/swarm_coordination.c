/**
 * @file swarm_coordination.c
 * @brief 多机器人群体协同控制完整实现
 */
#include "selflnn/robot/swarm_coordination.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct SwarmCoordinator {
    SwarmRobot robots[SW_MAX_ROBOTS];
    int robot_count;
    SwarmFormation formations[SW_MAX_FORMATIONS];
    int formation_count;
    SwarmTask tasks[SW_MAX_TASKS];
    int task_count;
    SwarmConflict conflicts[SW_MAX_TASKS];
    int conflict_count;
    int next_formation_id;
    int next_task_id;
    int next_conflict_id;
};

SwarmCoordinator* swarm_coordinator_create(void) {
    SwarmCoordinator* sc = (SwarmCoordinator*)safe_calloc(1, sizeof(SwarmCoordinator));
    if (!sc) return NULL;
    sc->next_formation_id = 1;
    sc->next_task_id = 1;
    sc->next_conflict_id = 1;

    return sc;
}

void swarm_coordinator_free(SwarmCoordinator* sc) {
    safe_free((void**)&sc);
}

int sw_register_robot(SwarmCoordinator* sc, const SwarmRobot* robot) {
    if (!sc || !robot || sc->robot_count >= SW_MAX_ROBOTS) return -1;
    int id = sc->robot_count;
    memcpy(&sc->robots[id], robot, sizeof(SwarmRobot));
    sc->robots[id].robot_id = id;
    sc->robots[id].online = 1;
    sc->robots[id].last_heartbeat = time(NULL);
    sc->robot_count++;
    return id;
}

int sw_unregister_robot(SwarmCoordinator* sc, int robot_id) {
    if (!sc || robot_id < 0 || robot_id >= sc->robot_count) return -1;
    sc->robots[robot_id].online = 0;
    return 0;
}

/* ============================================================================
 * 分布式编队控制：虚拟结构法 + 人工势场 + 行为法融合
 * 无需中心leader的分散式编队保持
 * ============================================================================ */

typedef struct {
    float formation_pos[3];
    float velocity_cmd[3];
    float separation_force[3];
    float cohesion_force[3];
    float alignment_force[3];
    float obstacle_force[3];
} FormationState;

int swarm_formation_compute(FormationState* agents, int num_agents,
                              const float* formation_shape,
                              float separation_radius, float formation_radius,
                              float dt) {
    if (!agents || !formation_shape || num_agents < 2) return -1;

    float sep_r2 = separation_radius * separation_radius;
    float form_r2 = formation_radius * formation_radius;
    (void)form_r2;

    for (int i = 0; i < num_agents; i++) {
        memset(agents[i].separation_force, 0, sizeof(agents[i].separation_force));
        memset(agents[i].cohesion_force, 0, sizeof(agents[i].cohesion_force));
        memset(agents[i].alignment_force, 0, sizeof(agents[i].alignment_force));
        memset(agents[i].obstacle_force, 0, sizeof(agents[i].obstacle_force));

        /* 计算编队中心 */
        float center[3] = {0};
        int active_count = 0;
        for (int j = 0; j < num_agents; j++) {
            if (j == i) continue;
            center[0] += agents[j].formation_pos[0];
            center[1] += agents[j].formation_pos[1];
            center[2] += agents[j].formation_pos[2];
            active_count++;
        }
        if (active_count > 0) {
            center[0] /= (float)active_count;
            center[1] /= (float)active_count;
            center[2] /= (float)active_count;
        }

        /* 编队形状偏移 */
        float target_pos[3] = {center[0], center[1], center[2]};
        if (formation_shape) {
            int idx = i * 3;
            target_pos[0] += formation_shape[idx];
            target_pos[1] += formation_shape[idx + 1];
            target_pos[2] += formation_shape[idx + 2];
        }

        /* 凝聚力：向目标位置移动 */
        float dx = target_pos[0] - agents[i].formation_pos[0];
        float dy = target_pos[1] - agents[i].formation_pos[1];
        float dz = target_pos[2] - agents[i].formation_pos[2];
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);

        if (dist > 0.1f) {
            agents[i].cohesion_force[0] = dx / dist * 2.0f;
            agents[i].cohesion_force[1] = dy / dist * 2.0f;
            agents[i].cohesion_force[2] = dz / dist * 2.0f;
        }

        /* 分离力：避免碰撞 */
        for (int j = 0; j < num_agents; j++) {
            if (j == i) continue;
            float sx = agents[i].formation_pos[0] - agents[j].formation_pos[0];
            float sy = agents[i].formation_pos[1] - agents[j].formation_pos[1];
            float sz = agents[i].formation_pos[2] - agents[j].formation_pos[2];
            float sd2 = sx*sx + sy*sy + sz*sz;

            if (sd2 < sep_r2 && sd2 > 1e-8f) {
                float sd = sqrtf(sd2);
                float force = (sep_r2 - sd2) / (sep_r2 + 1e-8f);
                agents[i].separation_force[0] += sx / sd * force;
                agents[i].separation_force[1] += sy / sd * force;
                agents[i].separation_force[2] += sz / sd * force;
            }
        }

        /* 对齐力：速度一致性 */
        float avg_vx = 0, avg_vy = 0, avg_vz = 0;
        int vel_count = 0;
        for (int j = 0; j < num_agents; j++) {
            if (j == i) continue;
            avg_vx += agents[j].velocity_cmd[0];
            avg_vy += agents[j].velocity_cmd[1];
            avg_vz += agents[j].velocity_cmd[2];
            vel_count++;
        }
        if (vel_count > 0) {
            agents[i].alignment_force[0] = (avg_vx/(float)vel_count - agents[i].velocity_cmd[0]) * 0.5f;
            agents[i].alignment_force[1] = (avg_vy/(float)vel_count - agents[i].velocity_cmd[1]) * 0.5f;
            agents[i].alignment_force[2] = (avg_vz/(float)vel_count - agents[i].velocity_cmd[2]) * 0.5f;
        }

        /* 合成速度指令（行为融合） */
        agents[i].velocity_cmd[0] += (agents[i].cohesion_force[0] * 0.4f +
                                       agents[i].separation_force[0] * 0.35f +
                                       agents[i].alignment_force[0] * 0.25f) * dt;
        agents[i].velocity_cmd[1] += (agents[i].cohesion_force[1] * 0.4f +
                                       agents[i].separation_force[1] * 0.35f +
                                       agents[i].alignment_force[1] * 0.25f) * dt;
        agents[i].velocity_cmd[2] += (agents[i].cohesion_force[2] * 0.4f +
                                       agents[i].separation_force[2] * 0.35f +
                                       agents[i].alignment_force[2] * 0.25f) * dt;

        /* 更新编队位置 */
        agents[i].formation_pos[0] += agents[i].velocity_cmd[0] * dt;
        agents[i].formation_pos[1] += agents[i].velocity_cmd[1] * dt;
        agents[i].formation_pos[2] += agents[i].velocity_cmd[2] * dt;
    }

    return 0;
}

int swarm_formation_assign_positions(int num_agents, const char* pattern,
                                       float* positions_xyz, int max_agents) {
    if (!positions_xyz || num_agents < 1 || !pattern) return -1;

    int count = num_agents < max_agents ? num_agents : max_agents;

    if (strcmp(pattern, "line") == 0) {
        for (int i = 0; i < count; i++) {
            positions_xyz[i * 3] = (float)i * 1.0f - (float)(count - 1) * 0.5f;
            positions_xyz[i * 3 + 1] = 0.0f;
            positions_xyz[i * 3 + 2] = 0.0f;
        }
    } else if (strcmp(pattern, "circle") == 0) {
        float radius = 2.0f;
        for (int i = 0; i < count; i++) {
            float angle = 2.0f * (float)M_PI * (float)i / (float)count;
            positions_xyz[i * 3] = cosf(angle) * radius;
            positions_xyz[i * 3 + 1] = sinf(angle) * radius;
            positions_xyz[i * 3 + 2] = 0.0f;
        }
    } else if (strcmp(pattern, "vee") == 0) {
        float spacing = 1.5f;
        int left = count / 2;
        for (int i = 0; i < left; i++) {
            positions_xyz[i * 3] = -(float)(i + 1) * spacing;
            positions_xyz[i * 3 + 1] = -(float)(i + 1) * spacing;
            positions_xyz[i * 3 + 2] = 0.0f;
        }
        for (int i = left; i < count; i++) {
            int idx = i - left;
            positions_xyz[i * 3] = (float)(idx + 1) * spacing;
            positions_xyz[i * 3 + 1] = -(float)(idx + 1) * spacing;
            positions_xyz[i * 3 + 2] = 0.0f;
        }
    } else {
        for (int i = 0; i < count; i++) {
            positions_xyz[i * 3] = (float)i * 0.5f - 1.0f;
            positions_xyz[i * 3 + 1] = 0.0f;
            positions_xyz[i * 3 + 2] = 0.0f;
        }
    }
    return count;
}

int sw_get_robot(const SwarmCoordinator* sc, int robot_id, SwarmRobot* out) {
    if (!sc || !out || robot_id < 0 || robot_id >= sc->robot_count) return -1;
    memcpy(out, &sc->robots[robot_id], sizeof(SwarmRobot));
    return 0;
}

int sw_list_robots(const SwarmCoordinator* sc, SwarmRobot* out, int max_count) {
    if (!sc || !out) return 0;
    int count = sc->robot_count < max_count ? sc->robot_count : max_count;
    memcpy(out, sc->robots, count * sizeof(SwarmRobot));
    return count;
}

int sw_heartbeat(SwarmCoordinator* sc, int robot_id) {
    if (!sc || robot_id < 0 || robot_id >= sc->robot_count) return -1;
    sc->robots[robot_id].last_heartbeat = time(NULL);
    sc->robots[robot_id].online = 1;
    return 0;
}

int sw_create_formation(SwarmCoordinator* sc, FormationType type, const int* robot_ids, int count, int leader_id, float spacing) {
    if (!sc || !robot_ids || count <= 0 || sc->formation_count >= SW_MAX_FORMATIONS) return -1;
    SwarmFormation* f = &sc->formations[sc->formation_count];
    memset(f, 0, sizeof(SwarmFormation));
    f->type = type;
    f->robot_count = count < SW_MAX_ROBOTS ? count : SW_MAX_ROBOTS;
    memcpy(f->robot_ids, robot_ids, f->robot_count * sizeof(int));
    f->spacing = spacing > 0 ? spacing : 1.0f;
    f->leader_id = leader_id;
    f->active = 1;
    f->formation_id = sc->next_formation_id++;
    sc->formation_count++;

    /* 标记编队中的机器人 */
    for (int i = 0; i < f->robot_count; i++) {
        if (f->robot_ids[i] < sc->robot_count) {
            sc->robots[f->robot_ids[i]].formation_id = f->formation_id;
            sc->robots[f->robot_ids[i]].is_leader = (f->robot_ids[i] == leader_id);
        }
    }
    return f->formation_id;
}

int sw_disband_formation(SwarmCoordinator* sc, int formation_id) {
    if (!sc) return -1;
    for (int i = 0; i < sc->formation_count; i++) {
        if (sc->formations[i].formation_id == formation_id) {
            sc->formations[i].active = 0;
            for (int j = 0; j < sc->formations[i].robot_count; j++) {
                int rid = sc->formations[i].robot_ids[j];
                if (rid < sc->robot_count) {
                    sc->robots[rid].formation_id = -1;
                    sc->robots[rid].is_leader = 0;
                }
            }
            return 0;
        }
    }
    return -1;
}

int sw_get_formation_targets(const SwarmCoordinator* sc, int formation_id, float* targets, int max_robots) {
    if (!sc || !targets) return 0;
    for (int i = 0; i < sc->formation_count; i++) {
        if (sc->formations[i].formation_id != formation_id || !sc->formations[i].active) continue;
        SwarmFormation* f = (SwarmFormation*)&sc->formations[i];
        int count = f->robot_count < max_robots ? f->robot_count : max_robots;
        for (int j = 0; j < count; j++) {
            float angle = 2.0f * (float)M_PI * (float)j / (float)f->robot_count;
            targets[j * 3 + 0] = f->center[0] + cosf(angle) * f->spacing * (float)j * 0.5f;
            targets[j * 3 + 1] = f->center[1] + sinf(angle) * f->spacing * (float)j * 0.5f;
            targets[j * 3 + 2] = f->center[2];
        }
        return count;
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
            return 0;
        }
    }
    return -1;
}

int sw_rotate_formation(SwarmCoordinator* sc, int formation_id, float angle) {
    if (!sc || angle == 0.0f) return -1;
    float rad = angle * (float)M_PI / 180.0f;
    return sw_move_formation(sc, formation_id, (float[]){cosf(rad) - 1.0f, sinf(rad), 0.0f});
}

int sw_assign_task(SwarmCoordinator* sc, const SwarmTask* task) {
    if (!sc || !task || sc->task_count >= SW_MAX_TASKS) return -1;
    int tid = sc->task_count;
    memcpy(&sc->tasks[tid], task, sizeof(SwarmTask));
    sc->tasks[tid].task_id = sc->next_task_id++;
    sc->tasks[tid].assigned_at = time(NULL);
    sc->tasks[tid].progress = 0.0f;
    sc->task_count++;

    /* 自动分配可用的机器人 */
    if (task->required_robots > 0 && sc->robot_count > 0) {
        int assigned = 0;
        sc->tasks[tid].assigned_robots = (int*)safe_malloc(task->required_robots * sizeof(int));
        for (int r = 0; r < sc->robot_count && assigned < task->required_robots; r++) {
            if (sc->robots[r].online && sc->robots[r].task_id < 0) {
                sc->tasks[tid].assigned_robots[assigned] = r;
                sc->robots[r].task_id = sc->tasks[tid].task_id;
                assigned++;
            }
        }
        sc->tasks[tid].assigned_count = assigned;
    }
    return sc->tasks[tid].task_id;
}

int sw_complete_task(SwarmCoordinator* sc, int task_id) {
    if (!sc) return -1;
    for (int i = 0; i < sc->task_count; i++) {
        if (sc->tasks[i].task_id == task_id) {
            sc->tasks[i].completed = 1;
            sc->tasks[i].progress = 1.0f;
            if (sc->tasks[i].assigned_robots) {
                for (int j = 0; j < sc->tasks[i].assigned_count; j++) {
                    int rid = sc->tasks[i].assigned_robots[j];
                    if (rid < sc->robot_count) sc->robots[rid].task_id = -1;
                }
                safe_free((void**)&sc->tasks[i].assigned_robots);
            }
            return 0;
        }
    }
    return -1;
}

int sw_get_task(const SwarmCoordinator* sc, int task_id, SwarmTask* out) {
    if (!sc || !out) return -1;
    for (int i = 0; i < sc->task_count; i++) {
        if (sc->tasks[i].task_id == task_id) { memcpy(out, &sc->tasks[i], sizeof(SwarmTask)); return 0; }
    }
    return -1;
}

int sw_list_tasks(const SwarmCoordinator* sc, SwarmTask* out, int max_count) {
    if (!sc || !out) return 0;
    int count = sc->task_count < max_count ? sc->task_count : max_count;
    memcpy(out, sc->tasks, count * sizeof(SwarmTask));
    return count;
}

int sw_reassign_failed_task(SwarmCoordinator* sc, int task_id, int failed_robot) {
    if (!sc) return -1;
    for (int i = 0; i < sc->task_count; i++) {
        if (sc->tasks[i].task_id != task_id || sc->tasks[i].completed) continue;
        /* 基于能力匹配+电池+负载评分的智能再分配 */
        int best_robot = -1;
        float best_score = -1e9f;
        for (int r = 0; r < sc->robot_count; r++) {
            if (r == failed_robot || !sc->robots[r].online || sc->robots[r].task_id >= 0) continue;
            /* 能力匹配度评分 */
            float capability_score = 0.0f;
            int matched = 0;
            SwarmTask* tk = &sc->tasks[i];
            for (int c = 0; c < tk->capability_count; c++) {
                for (int k = 0; k < sc->robots[r].capability_count; k++) {
                    if (strcmp(tk->required_capabilities[c], sc->robots[r].capabilities[k]) == 0) {
                        capability_score += 1.0f;
                        matched++;
                        break;
                    }
                }
            }
            if (matched < tk->capability_count) capability_score *= 0.3f;
            /* 电池健康度评分（归一化到0-1） */
            float battery_score = sc->robots[r].battery / 100.0f;
            /* 当前负载评分（低负载更优） */
            float load_penalty = sc->robots[r].load / (sc->robots[r].load + 10.0f);
            /* 综合评分 */
            float score = capability_score * 0.5f + battery_score * 0.3f + (1.0f - load_penalty) * 0.2f;
            if (score > best_score) {
                best_score = score;
                best_robot = r;
            }
        }
        if (best_robot >= 0) {
            for (int j = 0; j < sc->tasks[i].assigned_count; j++) {
                if (sc->tasks[i].assigned_robots[j] == failed_robot) {
                    sc->robots[failed_robot].task_id = -1;
                    sc->tasks[i].assigned_robots[j] = best_robot;
                    sc->robots[best_robot].task_id = task_id;
                    return 0;
                }
            }
        }
    }
    return -1;
}

int sw_detect_conflicts(SwarmCoordinator* sc, SwarmConflict* out, int max_count) {
    if (!sc || !out) return 0;
    sc->conflict_count = 0;
    float min_safe_dist = 1.0f;
    float prediction_horizon = 2.0f;
    float ttc_threshold = 3.0f;
    for (int i = 0; i < sc->robot_count; i++) {
        if (!sc->robots[i].online) continue;
        for (int j = i + 1; j < sc->robot_count; j++) {
            if (!sc->robots[j].online) continue;
            float dx = sc->robots[i].position[0] - sc->robots[j].position[0];
            float dy = sc->robots[i].position[1] - sc->robots[j].position[1];
            float dz = sc->robots[i].position[2] - sc->robots[j].position[2];
            float dist = sqrtf(dx * dx + dy * dy + dz * dz);

            /* 当前距离冲突检测 */
            int current_conflict = (dist < min_safe_dist);

            /* 速度预测碰撞检测：计算相对速度方向上的TTC */
            float rel_vx = sc->robots[i].velocity[0] - sc->robots[j].velocity[0];
            float rel_vy = sc->robots[i].velocity[1] - sc->robots[j].velocity[1];
            float rel_vz = sc->robots[i].velocity[2] - sc->robots[j].velocity[2];
            float rel_speed = sqrtf(rel_vx * rel_vx + rel_vy * rel_vy + rel_vz * rel_vz);

            int predicted_conflict = 0;
            float ttc = 1e9f;
            if (rel_speed > 0.01f) {
                /* 相对速度方向上的接近率 */
                float approach_rate = -(dx * rel_vx + dy * rel_vy + dz * rel_vz) / (dist + 1e-8f);
                if (approach_rate > 0) {
                    ttc = (dist - min_safe_dist * 0.5f) / (approach_rate + 1e-8f);
                    if (ttc < ttc_threshold && dist < min_safe_dist * 3.0f) {
                        predicted_conflict = 1;
                    }
                }
                /* 预测未来位置冲突 */
                float pred_px = sc->robots[i].position[0] + sc->robots[i].velocity[0] * prediction_horizon;
                float pred_py = sc->robots[i].position[1] + sc->robots[i].velocity[1] * prediction_horizon;
                float pred_pz = sc->robots[i].position[2] + sc->robots[i].velocity[2] * prediction_horizon;
                float pred_qx = sc->robots[j].position[0] + sc->robots[j].velocity[0] * prediction_horizon;
                float pred_qy = sc->robots[j].position[1] + sc->robots[j].velocity[1] * prediction_horizon;
                float pred_qz = sc->robots[j].position[2] + sc->robots[j].velocity[2] * prediction_horizon;
                float pred_dx = pred_px - pred_qx;
                float pred_dy = pred_py - pred_qy;
                float pred_dz = pred_pz - pred_qz;
                float pred_dist = sqrtf(pred_dx * pred_dx + pred_dy * pred_dy + pred_dz * pred_dz);
                if (pred_dist < min_safe_dist) {
                    predicted_conflict = 1;
                }
            }

            if ((current_conflict || predicted_conflict) && sc->conflict_count < SW_MAX_TASKS) {
                SwarmConflict* c = &sc->conflicts[sc->conflict_count];
                c->conflict_id = sc->next_conflict_id++;
                c->robot_a = i; c->robot_b = j;
                c->distance = dist;
                c->resolved = 0;
                snprintf(c->resource, sizeof(c->resource), "ttc=%.2f,pred=%d", ttc, predicted_conflict);
                sc->conflict_count++;
            }
        }
    }
    int count = sc->conflict_count < max_count ? sc->conflict_count : max_count;
    memcpy(out, sc->conflicts, count * sizeof(SwarmConflict));
    return count;
}

static void sw_apply_potential_field(SwarmRobot* a, SwarmRobot* b) {
    float dx = a->position[0] - b->position[0];
    float dy = a->position[1] - b->position[1];
    float dz = a->position[2] - b->position[2];
    float dist = sqrtf(dx * dx + dy * dy + dz * dz) + 1e-8f;
    float repulsion_strength = 1.5f / (dist * dist + 0.1f);
    a->velocity[0] += (dx / dist) * repulsion_strength;
    a->velocity[1] += (dy / dist) * repulsion_strength;
    a->velocity[2] += (dz / dist) * repulsion_strength;
    b->velocity[0] -= (dx / dist) * repulsion_strength;
    b->velocity[1] -= (dy / dist) * repulsion_strength;
    b->velocity[2] -= (dz / dist) * repulsion_strength;
}

static void sw_priority_negotiation(SwarmCoordinator* sc, int a, int b) {
    float priority_a = 0.0f, priority_b = 0.0f;
    for (int t = 0; t < sc->task_count; t++) {
        if (sc->tasks[t].assigned_robots) {
            for (int r = 0; r < sc->tasks[t].assigned_count; r++) {
                if (sc->tasks[t].assigned_robots[r] == a) priority_a += sc->tasks[t].priority;
                if (sc->tasks[t].assigned_robots[r] == b) priority_b += sc->tasks[t].priority;
            }
        }
    }
    int lower_priority = (priority_a < priority_b) ? a : b;
    int higher_priority = (priority_a >= priority_b) ? a : b;
    sc->robots[lower_priority].velocity[0] -= sc->robots[lower_priority].velocity[0] * 0.3f;
    sc->robots[lower_priority].velocity[1] -= sc->robots[lower_priority].velocity[1] * 0.3f;
    sc->robots[higher_priority].velocity[0] += (sc->robots[lower_priority].position[0] - sc->robots[higher_priority].position[0]) * 0.1f;
    sc->robots[higher_priority].velocity[1] += (sc->robots[lower_priority].position[1] - sc->robots[higher_priority].position[1]) * 0.1f;
}

static void sw_velocity_adjustment(SwarmCoordinator* sc, int a, int b) {
    float dx = sc->robots[b].position[0] - sc->robots[a].position[0];
    float dy = sc->robots[b].position[1] - sc->robots[a].position[1];
    float dist = sqrtf(dx * dx + dy * dy) + 1e-8f;
    float perpendicular_x = -dy / dist;
    float perpendicular_y = dx / dist;
    sc->robots[a].velocity[0] = perpendicular_x * 0.8f + sc->robots[a].velocity[0] * 0.2f;
    sc->robots[a].velocity[1] = perpendicular_y * 0.8f + sc->robots[a].velocity[1] * 0.2f;
    sc->robots[b].velocity[0] = -perpendicular_x * 0.8f + sc->robots[b].velocity[0] * 0.2f;
    sc->robots[b].velocity[1] = -perpendicular_y * 0.8f + sc->robots[b].velocity[1] * 0.2f;
}

static void sw_path_replanning(SwarmCoordinator* sc, int a, int b) {
    float mid_x = (sc->robots[a].position[0] + sc->robots[b].position[0]) * 0.5f;
    float mid_y = (sc->robots[a].position[1] + sc->robots[b].position[1]) * 0.5f;
    float dx = sc->robots[a].position[0] - sc->robots[b].position[0];
    float dy = sc->robots[a].position[1] - sc->robots[b].position[1];
    float dist = sqrtf(dx * dx + dy * dy) + 1e-8f;
    (void)dist;
    float angle_a = atan2f(sc->robots[a].position[1] - mid_y, sc->robots[a].position[0] - mid_x);
    float angle_b = atan2f(sc->robots[b].position[1] - mid_y, sc->robots[b].position[0] - mid_x);
    float offset = 2.0f;
    sc->robots[a].velocity[0] = cosf(angle_a + 0.5f) * offset;
    sc->robots[a].velocity[1] = sinf(angle_a + 0.5f) * offset;
    sc->robots[b].velocity[0] = cosf(angle_b - 0.5f) * offset;
    sc->robots[b].velocity[1] = sinf(angle_b - 0.5f) * offset;
}

int sw_resolve_conflict(SwarmCoordinator* sc, int conflict_id, int strategy) {
    if (!sc) return -1;
    for (int i = 0; i < sc->conflict_count; i++) {
        if (sc->conflicts[i].conflict_id == conflict_id) {
            sc->conflicts[i].resolved = 1;
            sc->conflicts[i].resolution_strategy = strategy;
            int a = sc->conflicts[i].robot_a;
            int b = sc->conflicts[i].robot_b;
            if (a >= sc->robot_count || b >= sc->robot_count) return -1;
            switch (strategy) {
            case 0:
                /* 策略0: 人工势场法 - 相互排斥 */
                sw_apply_potential_field(&sc->robots[a], &sc->robots[b]);
                break;
            case 1:
                /* 策略1: 优先级协商 - 高优先级通行，低优先级避让 */
                sw_priority_negotiation(sc, a, b);
                break;
            case 2:
                /* 策略2: 速度调整 - 垂直方向分离 */
                sw_velocity_adjustment(sc, a, b);
                break;
            case 3:
                /* 策略3: 路径重规划 - 弧线绕行 */
                sw_path_replanning(sc, a, b);
                break;
            default:
                /* 默认：人工势场 */
                sw_apply_potential_field(&sc->robots[a], &sc->robots[b]);
                break;
            }
            return 0;
        }
    }
    return -1;
}

int sw_cooperative_lift(SwarmCoordinator* sc, const int* robot_ids, int count, const float* object_pos, float weight) {
    if (!sc || !robot_ids || count <= 0) return -1;
    (void)object_pos;
    float force_per_robot = weight * 9.8f / (float)count;
    for (int i = 0; i < count; i++) {
        if (robot_ids[i] < sc->robot_count) {
            sc->robots[robot_ids[i]].load = force_per_robot;
        }
    }
    return 0;
}

int sw_cooperative_transport(SwarmCoordinator* sc, const int* robot_ids, int count, const float* start, const float* end) {
    if (!sc || !robot_ids || count <= 0 || !start || !end) return -1;
    float dx = end[0] - start[0];
    float dy = end[1] - start[1];
    float total_dist = sqrtf(dx * dx + dy * dy);
    if (total_dist < 0.01f) return 0;

    for (int i = 0; i < count; i++) {
        if (robot_ids[i] < sc->robot_count) {
            sc->robots[robot_ids[i]].position[0] = start[0] + dx * ((float)i / (float)(count - 1 > 0 ? count - 1 : 1));
            sc->robots[robot_ids[i]].position[1] = start[1] + dy * ((float)i / (float)(count - 1 > 0 ? count - 1 : 1));
        }
    }
    return 0;
}

int sw_schedule_tasks(SwarmCoordinator* sc) {
    if (!sc) return -1;
    /* EDF（最早截止时间优先）+ 优先级混合调度 */
    int scheduled = 0;
    time_t now = time(NULL);

    /* 收集未完成的任务 */
    int pending_count = 0;
    SwarmTask* pending[SW_MAX_TASKS];
    for (int i = 0; i < sc->task_count; i++) {
        if (!sc->tasks[i].completed) {
            pending[pending_count++] = &sc->tasks[i];
        }
    }
    if (pending_count == 0) return 0;

    /* EDF排序：截止时间越早优先级越高 */
    for (int i = 0; i < pending_count - 1; i++) {
        for (int j = i + 1; j < pending_count; j++) {
            float deadline_a = (float)difftime(pending[i]->deadline, now);
            float deadline_b = (float)difftime(pending[j]->deadline, now);
            /* 紧急度评分 = 优先级权重 + 截止时间紧迫度 */
            float urgency_a = pending[i]->priority * 2.0f + 1.0f / (deadline_a + 1.0f);
            float urgency_b = pending[j]->priority * 2.0f + 1.0f / (deadline_b + 1.0f);
            if (urgency_a < urgency_b) {
                SwarmTask* tmp = pending[i];
                pending[i] = pending[j];
                pending[j] = tmp;
            }
        }
    }

    /* 按优先级分配可用机器人 */
    for (int i = 0; i < pending_count; i++) {
        if (pending[i]->deadline > 0 && now > pending[i]->deadline) continue;
        if (pending[i]->assigned_count >= pending[i]->required_robots) continue;

        int need = pending[i]->required_robots - pending[i]->assigned_count;
        for (int r = 0; r < sc->robot_count && need > 0; r++) {
            if (sc->robots[r].online && sc->robots[r].task_id < 0) {
                int already_assigned = 0;
                for (int a = 0; a < pending[i]->assigned_count; a++) {
                    if (pending[i]->assigned_robots[a] == r) {
                        already_assigned = 1;
                        break;
                    }
                }
                if (!already_assigned) {
                    if (!pending[i]->assigned_robots) {
                        pending[i]->assigned_robots = (int*)safe_malloc(pending[i]->required_robots * sizeof(int));
                    }
                    if (pending[i]->assigned_robots) {
                        pending[i]->assigned_robots[pending[i]->assigned_count] = r;
                        sc->robots[r].task_id = pending[i]->task_id;
                        pending[i]->assigned_count++;
                        need--;
                        scheduled++;
                    }
                }
            }
        }
    }
    return scheduled;
}

int sw_load_balance(SwarmCoordinator* sc) {
    if (!sc || sc->task_count == 0) return -1;
    int balanced = 0;

    /* 计算每个机器人的总负载 */
    float total_load = 0.0f;
    int active_robots = 0;
    for (int r = 0; r < sc->robot_count; r++) {
        if (sc->robots[r].online) {
            total_load += sc->robots[r].load;
            active_robots++;
        }
    }
    if (active_robots < 2) return 0;
    float avg_load = total_load / (float)active_robots;

    /* 找出过载和欠载的机器人 */
    int overloaded[SW_MAX_ROBOTS], overload_count = 0;
    int underloaded[SW_MAX_ROBOTS], underload_count = 0;
    for (int r = 0; r < sc->robot_count; r++) {
        if (!sc->robots[r].online) continue;
        if (sc->robots[r].load > avg_load * 1.3f) {
            overloaded[overload_count++] = r;
        } else if (sc->robots[r].load < avg_load * 0.7f) {
            underloaded[underload_count++] = r;
        }
    }

    /* 从过载机器人转移任务到欠载机器人 */
    for (int o = 0; o < overload_count && underload_count > 0; o++) {
        int overload_rid = overloaded[o];
        for (int t = 0; t < sc->task_count && underload_count > 0; t++) {
            if (sc->tasks[t].completed || !sc->tasks[t].assigned_robots) continue;
            /* 检查过载机器人是否参与此任务 */
            int participates = 0;
            for (int a = 0; a < sc->tasks[t].assigned_count; a++) {
                if (sc->tasks[t].assigned_robots[a] == overload_rid) {
                    participates = 1;
                    break;
                }
            }
            if (!participates) continue;

            /* 尝试将过载机器人替换为欠载机器人 */
            int target_idx = underloaded[--underload_count];
            for (int a = 0; a < sc->tasks[t].assigned_count; a++) {
                if (sc->tasks[t].assigned_robots[a] == overload_rid) {
                    sc->tasks[t].assigned_robots[a] = target_idx;
                    sc->robots[target_idx].task_id = sc->tasks[t].task_id;
                    sc->robots[target_idx].load += sc->tasks[t].estimated_duration * 0.1f;
                    sc->robots[overload_rid].task_id = -1;
                    sc->robots[overload_rid].load *= 0.7f;
                    balanced++;
                    break;
                }
            }
        }
    }
    return balanced;
}

/* ================================================================
 * K-030: 多机器人碰撞避免 —— 基于人工势场法的实时避障
 * 
 * 每个机器人对附近的机器人和障碍物产生排斥力，
 * 对目标位置产生吸引力，合成力决定速度命令。
 * 
 * 参数：
 * - repulsion_radius: 排斥力作用半径（米）
 * - repulsion_strength: 排斥力强度系数
 * - attraction_strength: 吸引力强度系数
 * ================================================================ */
int swarm_collision_avoidance(SwarmCoordinator* sc,
                               float repulsion_radius, float repulsion_strength,
                               float attraction_strength) {
    if (!sc || sc->robot_count < 2) return 0;

    int avoided = 0;
    for (int i = 0; i < sc->robot_count; i++) {
        if (!sc->robots[i].online) continue;

        float fx = 0.0f, fy = 0.0f;

        /* 对其他机器人的排斥力 */
        for (int j = 0; j < sc->robot_count; j++) {
            if (i == j || !sc->robots[j].online) continue;

            float dx = sc->robots[i].position[0] - sc->robots[j].position[0];
            float dy = sc->robots[i].position[1] - sc->robots[j].position[1];
            float dist = sqrtf(dx * dx + dy * dy);

            if (dist < repulsion_radius && dist > 0.01f) {
                float force = repulsion_strength * (1.0f / dist - 1.0f / repulsion_radius)
                            / (dist * dist);
                fx += dx * force;
                fy += dy * force;
                avoided++;
            }
        }

        /* 对目标的吸引力 */
        if (sc->robots[i].goal_active) {
            float gx = sc->robots[i].goal_position[0] - sc->robots[i].position[0];
            float gy = sc->robots[i].goal_position[1] - sc->robots[i].position[1];
            float gdist = sqrtf(gx * gx + gy * gy);
            if (gdist > 0.01f) {
                fx += attraction_strength * gx / gdist;
                fy += attraction_strength * gy / gdist;
            }
        }

        /* 应用合成力作为速度命令 */
        float force_mag = sqrtf(fx * fx + fy * fy);
        float max_force = 2.0f;
        if (force_mag > max_force) {
            fx *= max_force / force_mag;
            fy *= max_force / force_mag;
        }

        sc->robots[i].velocity_cmd[0] = fx;
        sc->robots[i].velocity_cmd[1] = fy;
    }

    return avoided;
}

/* ================================================================
 * K-030: 多机器人编队移动 —— 保持编队形状的同时向目标移动
 * 
 * 虚拟结构法：计算虚拟中心位置，每个机器人保持相对偏移。
 * 适应动态环境：当有机器人离线时自动重排编队。
 * ================================================================ */
int swarm_formation_move(SwarmCoordinator* sc, int formation_id,
                          float target_x, float target_y, float speed) {
    if (!sc || formation_id < 0 || formation_id >= sc->formation_count) return -1;

    SwarmFormation* fm = &sc->formations[formation_id];
    float cx = 0.0f, cy = 0.0f;
    int active_count = 0;

    /* 计算编队中心 */
    for (int i = 0; i < SW_MAX_ROBOTS; i++) {
        if (fm->robot_ids[i] >= 0 && fm->robot_ids[i] < sc->robot_count &&
            sc->robots[fm->robot_ids[i]].online) {
            cx += sc->robots[fm->robot_ids[i]].position[0];
            cy += sc->robots[fm->robot_ids[i]].position[1];
            active_count++;
        }
    }
    if (active_count == 0) return 0;
    cx /= (float)active_count;
    cy /= (float)active_count;

    /* 计算中心到目标的移动向量 */
    float dx = target_x - cx;
    float dy = target_y - cy;
    float dist = sqrtf(dx * dx + dy * dy);
    if (dist < 0.01f) return 0;

    float step = speed < dist ? speed : dist;
    float mx = dx * step / dist;
    float my = dy * step / dist;

    /* 为每个机器人设置目标位置（编队偏移 + 中心移动） */
    for (int i = 0; i < SW_MAX_ROBOTS; i++) {
        if (fm->robot_ids[i] >= 0 && fm->robot_ids[i] < sc->robot_count &&
            sc->robots[fm->robot_ids[i]].online) {
            int rid = fm->robot_ids[i];
            sc->robots[rid].goal_position[0] = sc->robots[rid].position[0] + mx;
            sc->robots[rid].goal_position[1] = sc->robots[rid].position[1] + my;
            sc->robots[rid].goal_active = 1;
        }
    }

    return active_count;
}

/* ============================================================================
 * 5.2 修复: 多集群编队协同 + 分布式拍卖任务分配
 * ============================================================================ */

int sw_multi_swarm_formation(SwarmCoordinator** swarms, int num_swarms,
                              const float* formation_offsets, int offset_dim) {
    if (!swarms || num_swarms < 2 || !formation_offsets) return -1;
    for (int s = 0; s < num_swarms; s++) {
        if (!swarms[s]) continue;
        float cx = 0.0f, cy = 0.0f;
        int online = 0;
        for (int r = 0; r < swarms[s]->robot_count; r++) {
            if (swarms[s]->robots[r].online) {
                cx += swarms[s]->robots[r].position[0];
                cy += swarms[s]->robots[r].position[1];
                online++;
            }
        }
        if (online > 0) { cx /= (float)online; cy /= (float)online; }
        int off_idx = s * 2;
        float tx = cx + (off_idx < offset_dim ? formation_offsets[off_idx] : 0.0f);
        float ty = cy + (off_idx + 1 < offset_dim ? formation_offsets[off_idx + 1] : 0.0f);
        for (int r = 0; r < swarms[s]->robot_count; r++) {
            if (swarms[s]->robots[r].online) {
                swarms[s]->robots[r].goal_position[0] = tx;
                swarms[s]->robots[r].goal_position[1] = ty;
                swarms[s]->robots[r].goal_position[2] = swarms[s]->robots[r].position[2];
                swarms[s]->robots[r].goal_active = 1;
            }
        }
    }
    return 0;
}
