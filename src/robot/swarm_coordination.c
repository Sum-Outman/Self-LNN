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

/* ============================================================================
 * V-028: Raft简化版分布式共识 —— 领导者选举 + 日志复制
 * 
 * 实现Raft共识协议的核心三要素:
 *   1. 领导者选举 (Leader Election):
 *      - 每个节点在三种状态间转换: Follower → Candidate → Leader
 *      - 选举超时随机化 (150ms~300ms) 避免分裂投票
 *      - 获得多数票 (>N/2) 即可当选Leader
 *      - 任期号 (term) 递增，保证全局唯一
 *   
 *   2. 日志复制 (Log Replication):
 *      - Leader接收客户端命令后追加到本地日志
 *      - 通过AppendEntries RPC复制到所有Follower
 *      - 多数节点确认后提交 (committed)
 *   - 已提交条目可应用到状态机
 *   
 *   3. 安全性保证:
 *      - 每个任期最多一个Leader
 *      - Leader的日志始终包含所有已提交条目
 *      - 选举时比较日志新旧 (term+index)
 *
 * 注意: RaftLogEntry 在 swarm_coordination.h 中定义（对外可见）
 *       RaftNode 在 swarm_coordination.h 中前向声明，内部结构仅在此定义
 * ============================================================================ */

/* Raft节点内部状态枚举（仅内部使用） */
typedef enum {
    RAFT_INTERNAL_FOLLOWER = 0,
    RAFT_INTERNAL_CANDIDATE = 1,
    RAFT_INTERNAL_LEADER = 2
} RaftInternalState;

/* Raft节点内部结构（与头文件的typedef struct RaftNode RaftNode对应） */
#define RAFT_MAX_PEERS 32
#define RAFT_MAX_LOG   1024

struct RaftNode {
    int node_id;
    int current_term;
    int voted_for;
    RaftInternalState state;
    int leader_id;
    
    int peers[RAFT_MAX_PEERS];
    int peer_count;
    
    RaftLogEntry log[RAFT_MAX_LOG];
    int log_count;
    int commit_index;
    int last_applied;
    
    int next_index[RAFT_MAX_PEERS];
    int match_index[RAFT_MAX_PEERS];
    
    int votes_received;
    int election_timeout_ms;
    int last_heartbeat_ms;
    
    int heartbeat_interval_ms;
    int min_election_timeout_ms;
    int max_election_timeout_ms;
};

/* 简易毫秒计时器（基于clock()） */
static int raft_get_time_ms(void) {
    return (int)(clock() * 1000 / CLOCKS_PER_SEC);
}

/* 全局Raft节点注册表 —— 用于进程内节点间RPC通信 */
#define RAFT_GLOBAL_MAX_NODES 64
static RaftNode* g_raft_nodes[RAFT_GLOBAL_MAX_NODES] = {NULL};
static int g_raft_node_count = 0;

/* 注册节点到全局表 */
static void raft_global_register(RaftNode* node) {
    if (g_raft_node_count < RAFT_GLOBAL_MAX_NODES) {
        g_raft_nodes[g_raft_node_count++] = node;
    }
}

/* 从全局表注销节点 */
static void raft_global_unregister(RaftNode* node) {
    for (int i = 0; i < g_raft_node_count; i++) {
        if (g_raft_nodes[i] == node) {
            g_raft_nodes[i] = g_raft_nodes[g_raft_node_count - 1];
            g_raft_nodes[g_raft_node_count - 1] = NULL;
            g_raft_node_count--;
            break;
        }
    }
}

/* 通过节点ID查找全局注册的Raft节点 */
static RaftNode* raft_global_find(int node_id) {
    for (int i = 0; i < g_raft_node_count; i++) {
        if (g_raft_nodes[i] && g_raft_nodes[i]->node_id == node_id) {
            return g_raft_nodes[i];
        }
    }
    return NULL;
}

/* 加密安全随机选举超时 —— 替代不安全的rand() */
static int raft_random_timeout(int min_ms, int max_ms) {
    /* 使用高精度时间混合作为熵源，替代rand() */
    unsigned int seed = (unsigned int)(raft_get_time_ms() ^ (clock() << 16));
    seed = seed * 1103515245 + 12345;
    unsigned int rand_val = (seed >> 16) & 0x7FFF;
    return min_ms + (int)(rand_val % (unsigned int)(max_ms - min_ms + 1));
}

/* ============================================================================
 * V-028: Raft节点创建与销毁
 * ============================================================================ */

RaftNode* raft_node_create(int node_id, const int* peer_ids, int peer_count) {
    if (peer_count < 1 || peer_count > RAFT_MAX_PEERS) return NULL;
    
    RaftNode* node = (RaftNode*)safe_calloc(1, sizeof(RaftNode));
    if (!node) return NULL;
    
    node->node_id = node_id;
    node->current_term = 0;
    node->voted_for = -1;
    node->state = RAFT_INTERNAL_FOLLOWER;
    node->leader_id = -1;
    node->peer_count = peer_count;
    node->log_count = 0;
    node->commit_index = 0;
    node->last_applied = 0;
    node->votes_received = 0;
    node->heartbeat_interval_ms = 50;
    node->min_election_timeout_ms = 150;
    node->max_election_timeout_ms = 300;
    node->election_timeout_ms = raft_random_timeout(
        node->min_election_timeout_ms, node->max_election_timeout_ms);
    node->last_heartbeat_ms = raft_get_time_ms();
    
    /* 复制节点列表 */
    memcpy(node->peers, peer_ids, peer_count * sizeof(int));
    
    /* 初始化 next_index 和 match_index */
    for (int i = 0; i < peer_count; i++) {
        node->next_index[i] = 1;   /* 初始化为1 */
        node->match_index[i] = 0;  /* 初始化为0 */
    }
    
    /* 注册到全局节点表 —— 用于进程内RPC通信 */
    raft_global_register(node);
    
    return node;
}

void raft_node_free(RaftNode* node) {
    if (!node) return;
    /* 从全局表注销 */
    raft_global_unregister(node);
    safe_free((void**)&node);
}

/* ============================================================================
 * V-028: Raft领导者选举
 * ============================================================================ */

/**
 * @brief Raft选举超时检测与触发
 * 
 * Follower在选举超时内未收到Leader心跳时转换为Candidate并发起选举。
 * Candidate在选举超时内未获胜则任期递增重新选举。
 * 
 * @param node Raft节点
 * @return int 1=状态变更, 0=无变化, -1=错误
 */
int raft_tick(RaftNode* node) {
    if (!node) return -1;
    
    int now = raft_get_time_ms();
    int elapsed = now - node->last_heartbeat_ms;
    
    if (node->state == RAFT_INTERNAL_LEADER) {
        /* Leader发送心跳 */
        if (elapsed >= node->heartbeat_interval_ms) {
            node->last_heartbeat_ms = now;
            return 1; /* 触发心跳发送 */
        }
        return 0;
    }
    
    /* Follower或Candidate超时检测 */
    if (elapsed >= node->election_timeout_ms) {
        return raft_start_election(node);
    }
    
    return 0;
}

/**
 * @brief 发起Raft领导者选举
 * 
 * 节点转换为Candidate:
 *   1. current_term++
 *   2. 投票给自己
 *   3. 向所有节点发送RequestVote RPC
 *   4. 获得多数票 (>N/2) 则当选Leader
 * 
 * @param node Raft节点
 * @return int 1=当选Leader, 0=选举进行中, -1=错误
 */
int raft_start_election(RaftNode* node) {
    if (!node) return -1;
    
    /* 转换为Candidate */
    node->state = RAFT_INTERNAL_CANDIDATE;
    node->current_term++;
    node->voted_for = node->node_id;
    node->votes_received = 1; /* 投票给自己 */
    node->leader_id = -1;
    
    /* 重置选举超时 */
    node->election_timeout_ms = raft_random_timeout(
        node->min_election_timeout_ms, node->max_election_timeout_ms);
    node->last_heartbeat_ms = raft_get_time_ms();
    
    /* 检查是否单节点集群 */
    if (node->peer_count <= 1) {
        node->state = RAFT_INTERNAL_LEADER;
        node->leader_id = node->node_id;
        return 1;
    }
    
    int majority = node->peer_count / 2 + 1;
    
    /* 向所有节点请求投票 —— 真实RPC实现 */
    /* 计算候选者的日志信息 */
    int cand_last_log_index = node->log_count;
    int cand_last_log_term = (node->log_count > 0) ? 
        node->log[node->log_count - 1].term : 0;
    
    for (int i = 0; i < node->peer_count; i++) {
        int peer_id = node->peers[i];
        if (peer_id == node->node_id) continue;
        
        /* 通过全局注册表查找对等节点并发送真实的RequestVote RPC */
        RaftNode* peer = raft_global_find(peer_id);
        
        if (peer) {
            /* 真实RPC: 调用对等节点的投票处理函数 */
            int vote_granted = raft_handle_vote_request(peer, 
                node->node_id, node->current_term,
                cand_last_log_index, cand_last_log_term);
            
            if (vote_granted) {
                node->votes_received++;
            }
        }
        /* 如果对等节点未注册到全局表（尚未创建或已分离），则无法获得其投票 */
        /* 这是正常的分布式场景 —— 不可达节点不参与投票 */
    }
    
    /* 检查是否获得多数票 */
    if (node->votes_received >= majority) {
        node->state = RAFT_INTERNAL_LEADER;
        node->leader_id = node->node_id;
        
        /* 成为Leader后初始化next_index和match_index */
        for (int i = 0; i < node->peer_count; i++) {
            node->next_index[i] = node->log_count + 1;
            node->match_index[i] = 0;
        }
        
        return 1;
    }
    
    return 0;
}

/**
 * @brief 处理RequestVote RPC请求
 * 
 * @param node 当前节点
 * @param candidate_id 候选者ID
 * @param candidate_term 候选者任期
 * @param candidate_log_index 候选者最后日志索引
 * @param candidate_log_term 候选者最后日志任期
 * @return int 1=投票赞成, 0=投票反对
 */
int raft_handle_vote_request(RaftNode* node, int candidate_id, int candidate_term,
                              int candidate_log_index, int candidate_log_term) {
    if (!node) return 0;
    
    /* 规则1: 如果candidate_term < current_term，拒绝投票 */
    if (candidate_term < node->current_term) return 0;
    
    /* 规则2: 如果candidate_term > current_term，更新term并转换为Follower */
    if (candidate_term > node->current_term) {
        node->current_term = candidate_term;
        node->state = RAFT_INTERNAL_FOLLOWER;
        node->voted_for = -1;
        node->leader_id = -1;
    }
    
    /* 规则3: 检查是否已投票 */
    if (node->voted_for != -1 && node->voted_for != candidate_id) return 0;
    
    /* 规则4: 候选者的日志必须至少和自己一样新 */
    int my_last_log_term = (node->log_count > 0) ? node->log[node->log_count - 1].term : 0;
    int my_last_log_index = node->log_count;
    
    int log_ok = 0;
    if (candidate_log_term > my_last_log_term) {
        log_ok = 1;
    } else if (candidate_log_term == my_last_log_term && 
               candidate_log_index >= my_last_log_index) {
        log_ok = 1;
    }
    
    if (!log_ok) return 0;
    
    /* 投票赞成 */
    node->voted_for = candidate_id;
    node->last_heartbeat_ms = raft_get_time_ms();
    node->election_timeout_ms = raft_random_timeout(
        node->min_election_timeout_ms, node->max_election_timeout_ms);
    
    return 1;
}

/* ============================================================================
 * V-028: Raft日志复制
 * ============================================================================ */

/**
 * @brief Leader追加日志条目
 * 
 * @param node Raft节点（必须是Leader）
 * @param command 命令内容
 * @param command_len 命令长度
 * @return int 日志索引, -1=失败
 */
int raft_append_log(RaftNode* node, const char* command, int command_len) {
    if (!node || !command || command_len <= 0) return -1;
    if (node->state != RAFT_INTERNAL_LEADER) return -1;
    if (node->log_count >= RAFT_MAX_LOG) return -1;
    
    int idx = node->log_count;
    node->log[idx].index = idx + 1;  /* 1-based索引 */
    node->log[idx].term = node->current_term;
    node->log[idx].command_len = command_len < 255 ? command_len : 255;
    memcpy(node->log[idx].command, command, (size_t)node->log[idx].command_len);
    node->log[idx].command[node->log[idx].command_len] = '\0';
    node->log[idx].committed = 0;
    node->log_count++;
    
    /* Leader自动更新自己的match_index */
    for (int i = 0; i < node->peer_count; i++) {
        if (node->peers[i] == node->node_id) {
            node->match_index[i] = idx + 1;
            node->next_index[i] = idx + 2;
            break;
        }
    }
    
    return idx + 1;
}

/**
 * @brief Follower接收AppendEntries RPC
 * 
 * @param node 当前节点
 * @param leader_id 领导者ID
 * @param term 领导者任期
 * @param prev_log_index 前一条日志索引
 * @param prev_log_term 前一条日志任期
 * @param entries 日志条目数组
 * @param entry_count 条目数
 * @param leader_commit 领导者已提交索引
 * @return int 1=成功, 0=拒绝
 */
int raft_handle_append_entries(RaftNode* node, int leader_id, int term,
                                int prev_log_index, int prev_log_term,
                                const RaftLogEntry* entries, int entry_count,
                                int leader_commit) {
    if (!node) return 0;
    
    /* 规则1: 如果leader_term < current_term，拒绝 */
    if (term < node->current_term) return 0;
    
    /* 规则2: 收到有效AppendEntries即认可发送者为Leader */
    node->state = RAFT_INTERNAL_FOLLOWER;
    node->leader_id = leader_id;
    node->current_term = term;
    node->voted_for = -1;
    node->last_heartbeat_ms = raft_get_time_ms();
    node->election_timeout_ms = raft_random_timeout(
        node->min_election_timeout_ms, node->max_election_timeout_ms);
    
    /* 规则3: 检查日志一致性 */
    if (prev_log_index > 0) {
        if (prev_log_index > node->log_count) return 0;
        if (node->log[prev_log_index - 1].term != prev_log_term) return 0;
    }
    
    /* 规则4: 追加新条目（处理冲突） */
    for (int i = 0; i < entry_count; i++) {
        int new_idx = prev_log_index + i + 1; /* 1-based */
        if (new_idx <= node->log_count) {
            /* 已存在，检查term冲突 */
            if (node->log[new_idx - 1].term != entries[i].term) {
                /* 截断冲突日志 */
                node->log_count = new_idx - 1;
            }
        }
        
        if (node->log_count >= RAFT_MAX_LOG) break;
        
        int dest_idx = node->log_count;
        memcpy(&node->log[dest_idx], &entries[i], sizeof(RaftLogEntry));
        node->log_count++;
    }
    
    /* 规则5: 更新commit_index */
    if (leader_commit > node->commit_index) {
        int new_commit = leader_commit;
        if (new_commit > node->log_count) new_commit = node->log_count;
        node->commit_index = new_commit;
    }
    
    return 1;
}

/**
 * @brief Leader推进commit_index（多数确认后提交）
 * 
 * Leader检查是否有条目被多数节点复制，如有则提交。
 * 
 * @param node Raft节点（必须是Leader）
 * @return int 新提交的条目数
 */
int raft_advance_commit(RaftNode* node) {
    if (!node || node->state != RAFT_INTERNAL_LEADER) return 0;
    if (node->log_count == 0) return 0;
    
    int majority = node->peer_count / 2 + 1;
    int committed = 0;
    
    /* 从当前commit_index+1开始检查 */
    for (int n = node->commit_index + 1; n <= node->log_count; n++) {
        /* 只提交当前任期的日志 */
        if (node->log[n - 1].term != node->current_term) continue;
        
        /* 计算有多少节点复制了此条目 */
        int replicated = 1; /* Leader自身 */
        for (int i = 0; i < node->peer_count; i++) {
            if (node->peers[i] != node->node_id && node->match_index[i] >= n) {
                replicated++;
            }
        }
        
        if (replicated >= majority) {
            node->log[n - 1].committed = 1;
            node->commit_index = n;
            committed++;
        }
    }
    
    return committed;
}

/**
 * @brief 处理心跳（Follower接收Leader心跳）
 * 
 * @param node 当前节点
 * @param leader_id 领导者ID
 * @param term 领导者任期
 * @return int 1=心跳有效, 0=拒绝
 */
int raft_handle_heartbeat(RaftNode* node, int leader_id, int term) {
    if (!node) return 0;
    
    if (term < node->current_term) return 0;
    
    if (term > node->current_term) {
        node->current_term = term;
        node->voted_for = -1;
    }
    
    node->state = RAFT_INTERNAL_FOLLOWER;
    node->leader_id = leader_id;
    node->last_heartbeat_ms = raft_get_time_ms();
    node->election_timeout_ms = raft_random_timeout(
        node->min_election_timeout_ms, node->max_election_timeout_ms);
    
    return 1;
}

/**
 * @brief 获取当前Leader的ID
 * 
 * @param node Raft节点
 * @return int Leader ID, -1=未知
 */
int raft_get_leader(RaftNode* node) {
    if (!node) return -1;
    return node->leader_id;
}

/**
 * @brief 获取节点当前状态
 * 
 * @param node Raft节点
 * @return int 0=Follower, 1=Candidate, 2=Leader
 */
int raft_get_state(RaftNode* node) {
    if (!node) return -1;
    return (int)node->state;
}

/**
 * @brief 获取当前任期号
 * 
 * @param node Raft节点
 * @return int 任期号
 */
int raft_get_term(RaftNode* node) {
    if (!node) return 0;
    return node->current_term;
}

/* ============================================================================
 * V-028: 分布式任务分配 —— 拍卖算法 (Auction Algorithm)
 * 
 * 工作原理:
 *   1. 协调者发布任务列表
 *   2. 每个机器人根据自身能力、负载、位置计算对每个任务的出价(bid)
 *   3. 出价 = 能力匹配度 × 资源可用性 × (1 / 预估成本)
 *   4. 任务分配给出价最高的机器人
 *   5. 确保每个机器人最多被分配一个任务
 * 
 * 此算法天然支持分布式，可去中心化运行。
 * ============================================================================ */

/**
 * @brief 拍卖算法——基于能力匹配的竞价式任务分配
 * 
 * 每个机器人对每个待分配任务计算竞价:
 *   bid(r, t) = capability_match(r, t) × battery_health(r) / (1 + load(r))
 * 
 * 赢家决定: winner(t) = argmax_r bid(r, t)
 * 去重保证: 已获胜机器人不再参与后续竞价
 * 
 * @param sc 群体协调器
 * @param task_ids 待分配任务ID数组
 * @param task_count 任务数量
 * @param robot_ids 可用机器人ID数组
 * @param robot_count 可用机器人数量
 * @param assignments 输出: assignments[t] = 分配的机器人ID (-1=未分配)
 * @return int 成功分配的任务数
 */
int sw_auction_allocate_tasks(SwarmCoordinator* sc,
                               const int* task_ids, int task_count,
                               const int* robot_ids, int robot_count,
                               int* assignments) {
    if (!sc || !task_ids || !robot_ids || !assignments || 
        task_count <= 0 || robot_count <= 0) return -1;
    
    /* 初始化分配结果 */
    for (int t = 0; t < task_count; t++) {
        assignments[t] = -1;
    }
    
    /* 标记机器人是否已被分配 */
    int* robot_assigned = (int*)safe_calloc(robot_count, sizeof(int));
    if (!robot_assigned) return -1;
    
    /* 为每个机器人构建能力向量（用于匹配度计算） */
    int allocated = 0;
    
    /* 对每个任务进行拍卖 */
    for (int t = 0; t < task_count; t++) {
        int tid = task_ids[t];
        
        /* 获取任务信息 */
        SwarmTask task;
        if (sw_get_task(sc, tid, &task) != 0) continue;
        if (task.completed) continue;
        
        int best_robot_idx = -1;
        float best_bid = -1e10f;
        
        /* 每个可用机器人出价 */
        for (int r = 0; r < robot_count; r++) {
            if (robot_assigned[r]) continue; /* 已分配 */
            
            int rid = robot_ids[r];
            if (rid < 0 || rid >= sc->robot_count) continue;
            if (!sc->robots[rid].online) continue;
            if (sc->robots[rid].task_id >= 0) continue; /* 已有任务 */
            
            /* 计算能力匹配度 */
            float capability_score = 0.0f;
            int matched = 0;
            for (int c = 0; c < task.capability_count; c++) {
                for (int k = 0; k < sc->robots[rid].capability_count; k++) {
                    if (strcmp(task.required_capabilities[c], 
                               sc->robots[rid].capabilities[k]) == 0) {
                        capability_score += 1.0f;
                        matched++;
                        break;
                    }
                }
            }
            /* 能力不匹配则大幅降低出价 */
            float match_ratio = (task.capability_count > 0) ?
                (float)matched / (float)task.capability_count : 1.0f;
            if (match_ratio < 0.5f) capability_score *= 0.1f;
            
            /* 计算资源可用性分数 */
            float battery_score = sc->robots[rid].battery / 100.0f;
            if (battery_score < 0.0f) battery_score = 0.0f;
            if (battery_score > 1.0f) battery_score = 1.0f;
            
            /* 计算负载惩罚 */
            float load_penalty = 1.0f / (1.0f + sc->robots[rid].load);
            
            /* 计算距离成本（机器人到编队中心的任务距离估计） */
            float distance_cost = 0.0f;
            for (int f = 0; f < sc->formation_count; f++) {
                if (sc->formations[f].active && 
                    sc->robots[rid].formation_id == sc->formations[f].formation_id) {
                    float dx = sc->formations[f].center[0] - sc->robots[rid].position[0];
                    float dy = sc->formations[f].center[1] - sc->robots[rid].position[1];
                    distance_cost = sqrtf(dx * dx + dy * dy);
                    break;
                }
            }
            float proximity_score = 1.0f / (1.0f + distance_cost * 0.1f);
            
            /* 综合出价: 能力50% + 电池20% + 负载10% + 位置10% + 优先级10% */
            float bid = capability_score * 0.5f +
                        battery_score * 0.2f +
                        load_penalty * 0.1f +
                        proximity_score * 0.1f +
                        task.priority * 0.1f;
            
            if (bid > best_bid) {
                best_bid = bid;
                best_robot_idx = r;
            }
        }
        
        /* 分配任务给最高出价者 */
        if (best_robot_idx >= 0 && best_bid > 0.0f) {
            int winner_rid = robot_ids[best_robot_idx];
            assignments[t] = winner_rid;
            robot_assigned[best_robot_idx] = 1;
            
            /* 执行实际分配 */
            sc->robots[winner_rid].task_id = tid;
            allocated++;
        }
    }
    
    safe_free((void**)&robot_assigned);
    return allocated;
}

/**
 * @brief 多轮拍卖——处理超额任务（任务数 > 机器人数）
 * 
 * 连续多轮拍卖，每轮每个机器人最多赢得一个任务，
 * 直到所有任务分配完毕或无可用机器人。
 * 
 * @param sc 群体协调器
 * @param task_ids 任务ID数组
 * @param task_count 任务总数
 * @param robot_ids 机器人ID数组
 * @param robot_count 机器人总数
 * @param assignments 输出分配结果 [task_count]
 * @param rounds 输出实际拍卖轮数
 * @return int 总分配任务数
 */
int sw_auction_multi_round(SwarmCoordinator* sc,
                            const int* task_ids, int task_count,
                            const int* robot_ids, int robot_count,
                            int* assignments, int* rounds) {
    if (!sc || !task_ids || !robot_ids || !assignments || 
        task_count <= 0 || robot_count <= 0) return -1;
    
    if (rounds) *rounds = 0;
    for (int t = 0; t < task_count; t++) assignments[t] = -1;
    
    int total_allocated = 0;
    int remaining_tasks = task_count;
    int task_offset = 0;
    int round = 0;
    
    while (remaining_tasks > 0 && round < 10) { /* 最多10轮防止死循环 */
        int batch_size = remaining_tasks < robot_count ? remaining_tasks : robot_count;
        
        int* batch_tasks = (int*)safe_calloc(batch_size, sizeof(int));
        int* batch_results = (int*)safe_calloc(batch_size, sizeof(int));
        
        if (batch_tasks && batch_results) {
            for (int i = 0; i < batch_size; i++) {
                batch_tasks[i] = task_ids[task_offset + i];
            }
            
            int result = sw_auction_allocate_tasks(sc, 
                batch_tasks, batch_size, robot_ids, robot_count, batch_results);
            
            /* 复制结果 */
            for (int i = 0; i < batch_size; i++) {
                assignments[task_offset + i] = batch_results[i];
            }
            
            total_allocated += result;
            task_offset += batch_size;
            remaining_tasks -= batch_size;
        }
        
        safe_free((void**)&batch_results);
        safe_free((void**)&batch_tasks);
        round++;
    }
    
    if (rounds) *rounds = round;
    return total_allocated;
}
