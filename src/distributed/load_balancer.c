#include "selflnn/distributed/load_balancer.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/core/laplace.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define TASK_ID_NONE ((uint64_t)-1)

static uint64_t get_time_ms(void) {
    TimeValue tv = time_get_current();
    return (uint64_t)tv.seconds * 1000 + (uint64_t)(tv.microseconds / 1000);
}

struct LbBalancer {
    LbConfig config;
    LbNode nodes[LB_MAX_NODES];
    int node_count;
    LbTask* tasks[LB_MAX_TASKS];
    int task_count;
    LbSchedulePolicy current_policy;
    LbStats stats;
    uint64_t next_task_id;
    uint64_t last_rebalance_time_ms;
    uint64_t last_health_check_ms;
    uint32_t rr_next_node;
    int initialized;

    /* A08.4.2 一致性哈希环 */
    LbRingSlot ring[LB_RING_SLOTS];
    int ring_size;
    int ring_initialized;
    int virtual_node_count;

    /* 拉普拉斯分析器（频域负载均衡稳定性分析与预测） */
    LaplaceAnalyzer* laplace_analyzer;
    float* laplace_spectrum_buffer;
};

void lb_default_config(LbConfig* config) {
    if (!config) return;
    memset(config, 0, sizeof(LbConfig));
    config->max_nodes = LB_MAX_NODES;
    config->max_tasks = LB_MAX_TASKS;
    config->heartbeat_interval_ms = 1000;
    config->health_check_interval_ms = 5000;
    config->metric_history_size = LB_METRIC_HISTORY;
    config->max_failures_before_remove = 3;
    config->default_policy = LB_POLICY_ROUND_ROBIN;
    config->overload_threshold = 0.85;
    config->underload_threshold = 0.20;
    config->enable_auto_rebalance = 1;
    config->enable_failover = 1;
    config->verbose = 0;
}

LbBalancer* lb_create(const LbConfig* config) {
    LbBalancer* balancer = (LbBalancer*)calloc(1, sizeof(LbBalancer));
    if (!balancer) return NULL;
    if (config) memcpy(&balancer->config, config, sizeof(LbConfig));
    else lb_default_config(&balancer->config);
    balancer->current_policy = balancer->config.default_policy;
    balancer->next_task_id = 1;
    balancer->rr_next_node = 0;
    balancer->initialized = 1;
    balancer->last_rebalance_time_ms = get_time_ms();
    balancer->last_health_check_ms = get_time_ms();
    
    /* P1-019修复：拉普拉斯分析器参数从实际负载特性动态推导
       替代硬编码值，根据配置的节点数量、心跳间隔和期望的频域分辨率推导 */
    {
        LaplaceConfig lap_cfg;
        memset(&lap_cfg, 0, sizeof(lap_cfg));
        /* 样本数：2的幂次，根据节点数量动态调整（至少128，最多2048） */
        int base_samples = 128;
        while (base_samples < (int)(balancer->config.max_nodes * 2u)) base_samples *= 2;
        if (base_samples > 2048) base_samples = 2048;
        lap_cfg.num_samples = base_samples;
        /* 采样率：基于心跳间隔计算奈奎斯特频率，至少2倍最大频率 */
        float heartbeat_hz = 1000.0f / (float)balancer->config.heartbeat_interval_ms;
        lap_cfg.sample_rate = heartbeat_hz * 2.5f;
        if (lap_cfg.sample_rate < 1.0f) lap_cfg.sample_rate = 1.0f;
        /* 最大频率：奈奎斯特频率（采样率的1/2），最小频率为直流成分 */
        lap_cfg.max_frequency = lap_cfg.sample_rate * 0.45f;
        lap_cfg.min_frequency = lap_cfg.sample_rate / (float)base_samples;
        lap_cfg.enable_stability = 1;
        lap_cfg.enable_frequency = 1;
        lap_cfg.enable_optimization = (balancer->config.enable_auto_rebalance ? 1 : 0);
        /* 截止频率：负载变化的特征频率（通常为心跳频率的1/4-1/2） */
        lap_cfg.cutoff_frequency = heartbeat_hz * 0.4f;
        lap_cfg.filter_order = 2;
        /* 指数衰减因子从过载/欠载阈值推导 */
        lap_cfg.alpha = (float)(1.0 - balancer->config.overload_threshold * 0.1);
        lap_cfg.beta  = (float)(balancer->config.underload_threshold * 0.2);
        balancer->laplace_analyzer = laplace_analyzer_create(&lap_cfg);
        balancer->laplace_spectrum_buffer = (float*)safe_malloc((size_t)base_samples * sizeof(float));
        if (balancer->laplace_spectrum_buffer) {
            memset(balancer->laplace_spectrum_buffer, 0, (size_t)base_samples * sizeof(float));
        }
    }
    
    return balancer;
}

void lb_destroy(LbBalancer* balancer) {
    if (!balancer) return;
    for (int i = 0; i < balancer->task_count; i++) {
        if (balancer->tasks[i]) free(balancer->tasks[i]);
    }
    if (balancer->laplace_analyzer) {
        laplace_analyzer_free(balancer->laplace_analyzer);
        balancer->laplace_analyzer = NULL;
    }
    safe_free((void**)&balancer->laplace_spectrum_buffer);
    free(balancer);
}

static int find_node_index(LbBalancer* balancer, uint32_t node_id) {
    for (int i = 0; i < balancer->node_count; i++) {
        if (balancer->nodes[i].node_id == node_id) return i;
    }
    return -1;
}

int lb_add_node(LbBalancer* balancer, uint32_t node_id, LbNodeType type,
                const char* name, const char* address, uint16_t port,
                const LbNodeCapability* capability) {
    if (!balancer || !name) return LB_ERROR_INVALID_PARAM;
    if (find_node_index(balancer, node_id) >= 0) return LB_ERROR_INVALID_PARAM;
    if (balancer->node_count >= (int)balancer->config.max_nodes) return LB_ERROR_MAX_NODES;
    int idx = balancer->node_count++;
    LbNode* node = &balancer->nodes[idx];
    memset(node, 0, sizeof(LbNode));
    node->node_id = node_id;
    node->node_type = type;
    node->status = LB_NODE_ONLINE;
    strncpy(node->name, name, LB_NODE_NAME_LEN - 1);
    node->name[LB_NODE_NAME_LEN - 1] = '\0';
    if (address) strncpy(node->address, address, sizeof(node->address) - 1);
    node->port = port;
    node->weight = 1.0;
    node->current_load = 0.0;
    node->last_heartbeat_ms = get_time_ms();
    node->is_local = 1;
    if (capability) memcpy(&node->capability, capability, sizeof(LbNodeCapability));
    return LB_ERROR_NONE;
}

int lb_remove_node(LbBalancer* balancer, uint32_t node_id) {
    if (!balancer) return LB_ERROR_INVALID_PARAM;
    int idx = find_node_index(balancer, node_id);
    if (idx < 0) return LB_ERROR_NODE_NOT_FOUND;
    for (int i = idx; i < balancer->node_count - 1; i++) {
        balancer->nodes[i] = balancer->nodes[i + 1];
    }
    balancer->node_count--;
    return LB_ERROR_NONE;
}

int lb_update_node_metrics(LbBalancer* balancer, uint32_t node_id,
                           const LbNodeMetrics* metrics) {
    if (!balancer || !metrics) return LB_ERROR_INVALID_PARAM;
    int idx = find_node_index(balancer, node_id);
    if (idx < 0) return LB_ERROR_NODE_NOT_FOUND;
    LbNode* node = &balancer->nodes[idx];
    memcpy(&node->metric_history[node->metric_history_index], &node->metrics, sizeof(LbNodeMetrics));
    node->metric_history_index = (node->metric_history_index + 1) % LB_METRIC_HISTORY;
    memcpy(&node->metrics, metrics, sizeof(LbNodeMetrics));
    /* P1-018修复：多维负载计算，综合GPU、内存、CPU、网络IO、磁盘IO和队列深度
       各维度权重基于分布式系统负载理论研究（CPU对非GPU节点关键，网络对远端节点关键） */
    double gpu_weight = (node->node_type == LB_NODE_CPU) ? 0.0 : 0.30;
    double cpu_weight  = (node->node_type == LB_NODE_CPU) ? 0.35 : 0.15;
    double mem_weight  = 0.20;
    double net_weight  = (node->node_type == LB_NODE_REMOTE) ? 0.25 : 0.15;
    double queue_weight = 0.10;
    double io_weight    = 0.10;
    double net_bw_ratio = 0.0;
    if (node->capability.network_bandwidth_gbps > 0.001) {
        net_bw_ratio = metrics->network_bandwidth_mbps / (node->capability.network_bandwidth_gbps * 1000.0);
        if (net_bw_ratio > 1.0) net_bw_ratio = 1.0;
    }
/* IO利用率使用队列深度+网络带宽加权估算
     * 原实现用temperature_celsius/100作为IO指标（无物理意义）
     * 改用: queue_depth/32归一化 + network_bandwidth_mbps利用率加权
     * LbNodeMetrics有: queue_depth(uint32) + network_bandwidth_mbps(double) */
    double queue_pressure = (double)(metrics->queue_depth) / 32.0;
    if (queue_pressure > 1.0) queue_pressure = 1.0;
    double io_util = queue_pressure * 0.5 + net_bw_ratio * 0.5;
    if (io_util > 1.0) io_util = 1.0;
    node->current_load = gpu_weight * metrics->gpu_utilization
                       + cpu_weight  * metrics->cpu_utilization
                       + mem_weight  * metrics->memory_utilization
                       + net_weight  * net_bw_ratio
                       + queue_weight * ((double)metrics->queue_depth / 100.0)
                       + io_weight   * io_util;
    if (node->current_load > 1.0) node->current_load = 1.0;
    if (node->current_load < 0.0) node->current_load = 0.0;
    return LB_ERROR_NONE;
}

int lb_set_node_weight(LbBalancer* balancer, uint32_t node_id, double weight) {
    if (!balancer) return LB_ERROR_INVALID_PARAM;
    int idx = find_node_index(balancer, node_id);
    if (idx < 0) return LB_ERROR_NODE_NOT_FOUND;
    balancer->nodes[idx].weight = weight > 0.0 ? weight : 0.1;
    return LB_ERROR_NONE;
}

int lb_get_node(const LbBalancer* balancer, uint32_t node_id, LbNode* node) {
    if (!balancer || !node) return LB_ERROR_INVALID_PARAM;
    int idx = find_node_index((LbBalancer*)balancer, node_id);
    if (idx < 0) return LB_ERROR_NODE_NOT_FOUND;
    memcpy(node, &balancer->nodes[idx], sizeof(LbNode));
    return LB_ERROR_NONE;
}

int lb_get_online_count(const LbBalancer* balancer) {
    if (!balancer) return 0;
    int count = 0;
    for (int i = 0; i < balancer->node_count; i++) {
        if (balancer->nodes[i].status == LB_NODE_ONLINE) count++;
    }
    return count;
}

uint64_t lb_submit_task(LbBalancer* balancer, LbTaskPriority priority,
                        uint32_t compute_cost, uint32_t memory_mb,
                        uint32_t estimated_duration_ms, int affinity_node_id,
                        void* user_data) {
    if (!balancer) return TASK_ID_NONE;
    if (balancer->task_count >= (int)balancer->config.max_tasks) return TASK_ID_NONE;
    LbTask* task = (LbTask*)calloc(1, sizeof(LbTask));
    if (!task) return TASK_ID_NONE;
    task->task_id = balancer->next_task_id++;
    task->priority = priority;
    task->estimated_compute_cost = compute_cost;
    task->estimated_memory_mb = memory_mb;
    task->estimated_duration_ms = estimated_duration_ms;
    task->assigned_node_id = (uint32_t)-1;
    task->submit_time_ms = get_time_ms();
    task->affinity_node_id = affinity_node_id;
    task->user_data = user_data;
    task->completed = 0;
    task->failed = 0;
    balancer->tasks[balancer->task_count++] = task;
    balancer->stats.total_tasks_submitted++;
    if (affinity_node_id >= 0) {
        uint32_t node = (uint32_t)affinity_node_id;
        if (find_node_index(balancer, node) >= 0) {
            task->assigned_node_id = node;
        }
    }
    return task->task_id;
}

int lb_cancel_task(LbBalancer* balancer, uint64_t task_id) {
    if (!balancer) return LB_ERROR_INVALID_PARAM;
    for (int i = 0; i < balancer->task_count; i++) {
        if (balancer->tasks[i] && balancer->tasks[i]->task_id == task_id) {
            free(balancer->tasks[i]);
            balancer->tasks[i] = balancer->tasks[--balancer->task_count];
            return LB_ERROR_NONE;
        }
    }
    return LB_ERROR_TASK_NOT_FOUND;
}

int lb_complete_task(LbBalancer* balancer, uint64_t task_id, int failed) {
    if (!balancer) return LB_ERROR_INVALID_PARAM;
    for (int i = 0; i < balancer->task_count; i++) {
        LbTask* t = balancer->tasks[i];
        if (t && t->task_id == task_id) {
            t->end_time_ms = get_time_ms();
            if (failed) {
                t->failed = 1;
                balancer->stats.total_tasks_failed++;
                uint32_t nid = t->assigned_node_id;
                int nidx = find_node_index(balancer, nid);
                if (nidx >= 0) balancer->nodes[nidx].consecutive_failures++;
            } else {
                t->completed = 1;
                balancer->stats.total_tasks_completed++;
                uint32_t nid = t->assigned_node_id;
                int nidx = find_node_index(balancer, nid);
                if (nidx >= 0) {
                    balancer->nodes[nidx].total_tasks_assigned++;
                    balancer->nodes[nidx].consecutive_failures = 0;
                }
                uint64_t exec_time = t->end_time_ms - t->start_time_ms;
                balancer->stats.avg_execution_time_ms =
                    (balancer->stats.avg_execution_time_ms * (balancer->stats.total_tasks_completed - 1) +
                     exec_time) / balancer->stats.total_tasks_completed;
            }
            free(t);
            balancer->tasks[i] = balancer->tasks[--balancer->task_count];
            return LB_ERROR_NONE;
        }
    }
    return LB_ERROR_TASK_NOT_FOUND;
}

int lb_get_task(const LbBalancer* balancer, uint64_t task_id, LbTask* task) {
    if (!balancer || !task) return LB_ERROR_INVALID_PARAM;
    for (int i = 0; i < balancer->task_count; i++) {
        if (balancer->tasks[i] && balancer->tasks[i]->task_id == task_id) {
            memcpy(task, balancer->tasks[i], sizeof(LbTask));
            return LB_ERROR_NONE;
        }
    }
    return LB_ERROR_TASK_NOT_FOUND;
}

int lb_get_queued_task_count(const LbBalancer* balancer) {
    if (!balancer) return 0;
    int count = 0;
    for (int i = 0; i < balancer->task_count; i++) {
        if (balancer->tasks[i] && !balancer->tasks[i]->completed && !balancer->tasks[i]->failed) {
            count++;
        }
    }
    return count;
}

static int lb_select_round_robin(LbBalancer* balancer) {
    int start = (int)balancer->rr_next_node;
    for (int i = 0; i < balancer->node_count; i++) {
        int idx = (start + i) % balancer->node_count;
        if (balancer->nodes[idx].status == LB_NODE_ONLINE) {
            balancer->rr_next_node = (uint32_t)((idx + 1) % balancer->node_count);
            return idx;
        }
    }
    return -1;
}

static int lb_select_least_loaded(LbBalancer* balancer) {
    int best = -1;
    double best_load = 1.0;
    for (int i = 0; i < balancer->node_count; i++) {
        if (balancer->nodes[i].status == LB_NODE_ONLINE &&
            balancer->nodes[i].current_load < best_load) {
            best = i;
            best_load = balancer->nodes[i].current_load;
        }
    }
    return best;
}

static int lb_select_weighted(LbBalancer* balancer) {
    double total_weight = 0.0;
    for (int i = 0; i < balancer->node_count; i++) {
        if (balancer->nodes[i].status == LB_NODE_ONLINE) {
            total_weight += balancer->nodes[i].weight;
        }
    }
    if (total_weight <= 0.0) return -1;
    double r = (double)secure_random_float() * total_weight;
    double cumulative = 0.0;
    for (int i = 0; i < balancer->node_count; i++) {
        if (balancer->nodes[i].status == LB_NODE_ONLINE) {
            cumulative += balancer->nodes[i].weight;
            if (r <= cumulative) return i;
        }
    }
    return -1;
}

static int lb_select_adaptive(LbBalancer* balancer) {
    int best = -1;
    double best_score = -1e9;
    for (int i = 0; i < balancer->node_count; i++) {
        if (balancer->nodes[i].status != LB_NODE_ONLINE) continue;
        LbNode* n = &balancer->nodes[i];
        double availability = 1.0 - (double)n->consecutive_failures * 0.2;
        if (availability < 0.1) availability = 0.1;
        double load_factor = 1.0 - n->current_load;
        double cap_factor = n->capability.compute_capacity;
        double score = availability * load_factor * cap_factor * n->weight;
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    return best;
}

static int lb_select_affinity(LbBalancer* balancer, int affinity_node_id) {
    if (affinity_node_id >= 0) {
        int idx = find_node_index(balancer, (uint32_t)affinity_node_id);
        if (idx >= 0 && balancer->nodes[idx].status == LB_NODE_ONLINE) return idx;
    }
    return lb_select_least_loaded(balancer);
}

int lb_schedule_next(LbBalancer* balancer, uint32_t* out_node_id) {
    if (!balancer || !out_node_id) return LB_ERROR_INVALID_PARAM;
    if (balancer->current_policy == LB_POLICY_CONSISTENT_HASH) {
        if (!balancer->ring_initialized || balancer->ring_size <= 0) {
            return LB_ERROR_NO_AVAILABLE_NODE;
        }
        uint64_t h = lb_hash_uint64(balancer->next_task_id++);
        char tmp[32];
        snprintf(tmp, sizeof(tmp), "%llu", (unsigned long long)h);
        *out_node_id = lb_lookup_key(balancer, tmp, strlen(tmp));
        if (*out_node_id == UINT32_MAX) return LB_ERROR_NO_AVAILABLE_NODE;
        return LB_ERROR_NONE;
    }
    return lb_find_best_node(balancer, out_node_id, balancer->current_policy);
}

int lb_schedule_batch(LbBalancer* balancer, uint32_t* task_ids,
                      uint32_t task_count, uint32_t* out_node_ids) {
    if (!balancer || !task_ids || !out_node_ids) return LB_ERROR_INVALID_PARAM;
    for (uint32_t i = 0; i < task_count; i++) {
        uint32_t node_id;
        int ret = lb_schedule_next(balancer, &node_id);
        if (ret == LB_ERROR_NONE) out_node_ids[i] = node_id;
        else out_node_ids[i] = (uint32_t)-1;
    }
    return LB_ERROR_NONE;
}

int lb_set_schedule_policy(LbBalancer* balancer, LbSchedulePolicy policy) {
    if (!balancer) return LB_ERROR_INVALID_PARAM;
    if (policy < LB_POLICY_ROUND_ROBIN || policy > LB_POLICY_CONSISTENT_HASH)
        return LB_ERROR_INVALID_PARAM;
    balancer->current_policy = policy;
    return LB_ERROR_NONE;
}

int lb_heartbeat(LbBalancer* balancer, uint32_t node_id) {
    if (!balancer) return LB_ERROR_INVALID_PARAM;
    int idx = find_node_index(balancer, node_id);
    if (idx < 0) return LB_ERROR_NODE_NOT_FOUND;
    balancer->nodes[idx].last_heartbeat_ms = get_time_ms();
    if (balancer->nodes[idx].consecutive_failures > 0) {
        balancer->nodes[idx].consecutive_failures--;
    }
    return LB_ERROR_NONE;
}

int lb_check_health(LbBalancer* balancer) {
    if (!balancer) return LB_ERROR_INVALID_PARAM;
    uint64_t now = get_time_ms();
    balancer->last_health_check_ms = now;
    int failures = 0;
    for (int i = 0; i < balancer->node_count; i++) {
        LbNode* n = &balancer->nodes[i];
        if (n->status == LB_NODE_OFFLINE || n->status == LB_NODE_REMOVED) continue;
        uint64_t elapsed = now - n->last_heartbeat_ms;
        if (elapsed > balancer->config.heartbeat_interval_ms * 3) {
            n->consecutive_failures++;
            if (n->consecutive_failures >= balancer->config.max_failures_before_remove) {
                n->status = LB_NODE_OFFLINE;
                if (balancer->config.verbose) {
                    printf("[LB] 节点 %u (%s) 已离线\n", n->node_id, n->name);
                }
                failures++;
            }
        } else if (n->consecutive_failures > 0) {
            n->consecutive_failures--;
        }
    }
    return failures > 0 ? 1 : 0;
}

int lb_rebalance(LbBalancer* balancer) {
    if (!balancer) return LB_ERROR_INVALID_PARAM;
    balancer->last_rebalance_time_ms = get_time_ms();
    int moved = 0;
    int overloaded = 0;
    int underloaded = 0;
    for (int i = 0; i < balancer->node_count; i++) {
        if (balancer->nodes[i].status != LB_NODE_ONLINE) continue;
        if (balancer->nodes[i].current_load > balancer->config.overload_threshold) {
            overloaded++;
        }
        if (balancer->nodes[i].current_load < balancer->config.underload_threshold) {
            underloaded++;
        }
    }
    if (overloaded > 0 && underloaded > 0) {
        for (int i = 0; i < balancer->task_count; i++) {
            LbTask* t = balancer->tasks[i];
            if (!t || t->completed || t->failed) continue;
            if (t->assigned_node_id == (uint32_t)-1) continue;
            int nidx = find_node_index(balancer, t->assigned_node_id);
            if (nidx >= 0 && balancer->nodes[nidx].current_load > balancer->config.overload_threshold) {
                uint32_t new_node;
                if (lb_schedule_next(balancer, &new_node) == LB_ERROR_NONE) {
                    t->assigned_node_id = new_node;
                    moved++;
                }
            }
        }
    }
    balancer->stats.total_rebalances++;
    return moved;
}

int lb_failover(LbBalancer* balancer, uint32_t failed_node_id) {
    if (!balancer) return LB_ERROR_INVALID_PARAM;
    int idx = find_node_index(balancer, failed_node_id);
    if (idx < 0) return LB_ERROR_NODE_NOT_FOUND;
    balancer->stats.total_failovers++;
    int failed_count = 0;
    for (int i = 0; i < balancer->task_count; i++) {
        LbTask* t = balancer->tasks[i];
        if (!t || t->completed) continue;
        if (t->assigned_node_id == failed_node_id) {
            uint32_t new_node;
            if (lb_find_best_node(balancer, &new_node, balancer->current_policy) == LB_ERROR_NONE) {
                t->assigned_node_id = new_node;
                t->start_time_ms = 0;
                failed_count++;
            }
        }
    }
    return failed_count;
}

void lb_get_stats(const LbBalancer* balancer, LbStats* stats) {
    if (!balancer || !stats) return;
    memcpy(stats, &balancer->stats, sizeof(LbStats));
    stats->online_node_count = 0;
    stats->offline_node_count = 0;
    for (int i = 0; i < balancer->node_count; i++) {
        if (balancer->nodes[i].status == LB_NODE_ONLINE) stats->online_node_count++;
        else if (balancer->nodes[i].status == LB_NODE_OFFLINE) stats->offline_node_count++;
    }
    stats->active_task_count = 0;
    stats->queued_task_count = 0;
    double total_load = 0.0;
    for (int i = 0; i < balancer->task_count; i++) {
        if (!balancer->tasks[i]) continue;
        if (balancer->tasks[i]->completed) continue;
        if (balancer->tasks[i]->start_time_ms > 0) stats->active_task_count++;
        else stats->queued_task_count++;
    }
    for (int i = 0; i < balancer->node_count; i++) {
        total_load += balancer->nodes[i].current_load;
    }
    stats->system_load_avg = balancer->node_count > 0 ? total_load / balancer->node_count : 0.0;
}

void lb_reset_stats(LbBalancer* balancer) {
    if (!balancer) return;
    memset(&balancer->stats, 0, sizeof(LbStats));
}

double lb_get_node_load(const LbBalancer* balancer, uint32_t node_id) {
    if (!balancer) return -1.0;
    int idx = find_node_index((LbBalancer*)balancer, node_id);
    if (idx < 0) return -1.0;
    return balancer->nodes[idx].current_load;
}

double lb_get_system_load(const LbBalancer* balancer) {
    if (!balancer || balancer->node_count == 0) return 0.0;
    double total = 0.0;
    for (int i = 0; i < balancer->node_count; i++) {
        total += balancer->nodes[i].current_load;
    }
    return total / balancer->node_count;
}

int lb_find_best_node(const LbBalancer* balancer, uint32_t* node_id,
                      LbSchedulePolicy policy) {
    if (!balancer || !node_id) return LB_ERROR_INVALID_PARAM;
    LbBalancer* b = (LbBalancer*)balancer;
    int idx = -1;
    switch (policy) {
        case LB_POLICY_ROUND_ROBIN: idx = lb_select_round_robin(b); break;
        case LB_POLICY_LEAST_LOADED: idx = lb_select_least_loaded(b); break;
        case LB_POLICY_WEIGHTED: idx = lb_select_weighted(b); break;
        case LB_POLICY_ADAPTIVE: idx = lb_select_adaptive(b); break;
        case LB_POLICY_AFFINITY: idx = lb_select_affinity(b, -1); break;
        case LB_POLICY_RANDOM: idx = lb_select_weighted(b); break;
        default: idx = lb_select_least_loaded(b); break;
    }
    if (idx < 0) return LB_ERROR_NO_AVAILABLE_NODE;
    *node_id = balancer->nodes[idx].node_id;
    return LB_ERROR_NONE;
}

const char* lb_error_string(int error_code) {
    switch (error_code) {
        case LB_ERROR_NONE: return "无错误";
        case LB_ERROR_INVALID_PARAM: return "无效参数";
        case LB_ERROR_NODE_NOT_FOUND: return "节点未找到";
        case LB_ERROR_NODE_OFFLINE: return "节点离线";
        case LB_ERROR_NO_AVAILABLE_NODE: return "无可用节点";
        case LB_ERROR_TASK_NOT_FOUND: return "任务未找到";
        case LB_ERROR_OUT_OF_MEMORY: return "内存不足";
        case LB_ERROR_QUEUE_FULL: return "队列已满";
        case LB_ERROR_MAX_NODES: return "已达最大节点数";
        default: return "未知错误";
    }
}

/* ============================================================================
 * 一致性哈希环实现 (A08.4.2)
 * 使用 FNV-1a 哈希 + 虚拟节点 + 二分查找
 * L-012修复: lb_hash_string 复用 math_utils.c 的公共 FNV-1a 哈希实现
 * ============================================================================ */

uint64_t lb_hash_string(const char* key, size_t len) {
    return math_fnv1a_hash64(key, len);
}

uint32_t lb_hash_uint64(uint64_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return (uint32_t)(key & 0x7FFFFFFF);
}

static int ring_slot_compare(const void* a, const void* b) {
    const LbRingSlot* sa = (const LbRingSlot*)a;
    const LbRingSlot* sb = (const LbRingSlot*)b;
    if (sa->hash_value < sb->hash_value) return -1;
    if (sa->hash_value > sb->hash_value) return 1;
    return 0;
}

int lb_enable_consistent_hashing(LbBalancer* balancer, int virtual_node_count) {
    if (!balancer) return LB_ERROR_INVALID_PARAM;
    if (virtual_node_count <= 0) return LB_ERROR_INVALID_PARAM;
    if (balancer->ring_initialized) return LB_ERROR_NONE;
    if (virtual_node_count > LB_MAX_VIRTUAL_NODES) virtual_node_count = LB_MAX_VIRTUAL_NODES;
    balancer->virtual_node_count = virtual_node_count;
    balancer->ring_size = 0;
    balancer->ring_initialized = 1;
    return lb_ring_rebuild(balancer);
}

int lb_disable_consistent_hashing(LbBalancer* balancer) {
    if (!balancer) return LB_ERROR_INVALID_PARAM;
    balancer->ring_initialized = 0;
    balancer->ring_size = 0;
    return LB_ERROR_NONE;
}

static int ring_find_node(LbBalancer* balancer, uint32_t node_id) {
    for (int i = 0; i < balancer->node_count; i++) {
        if (balancer->nodes[i].node_id == node_id) return i;
    }
    return -1;
}

int lb_ring_add_node(LbBalancer* balancer, uint32_t node_id) {
    if (!balancer || !balancer->ring_initialized) return LB_ERROR_INVALID_PARAM;
    if (ring_find_node(balancer, node_id) < 0) return LB_ERROR_NODE_NOT_FOUND;
    return lb_ring_rebuild(balancer);
}

int lb_ring_remove_node(LbBalancer* balancer, uint32_t node_id) {
    if (!balancer || !balancer->ring_initialized) return LB_ERROR_INVALID_PARAM;
    /* 验证节点存在后才能从环中移除 */
    if (ring_find_node(balancer, node_id) < 0) return LB_ERROR_NODE_NOT_FOUND;
    return lb_ring_rebuild(balancer);
}

int lb_ring_rebuild(LbBalancer* balancer) {
    if (!balancer) return LB_ERROR_INVALID_PARAM;
    int total_vnodes = 0;
    for (int i = 0; i < balancer->node_count; i++) {
        if (balancer->nodes[i].status == LB_NODE_ONLINE) {
            total_vnodes += balancer->virtual_node_count;
        }
    }
    if (total_vnodes == 0) {
        balancer->ring_size = 0;
        return LB_ERROR_NO_AVAILABLE_NODE;
    }
    if (total_vnodes > (int)LB_RING_SLOTS) total_vnodes = LB_RING_SLOTS;
    int slot_idx = 0;
    for (int i = 0; i < balancer->node_count && slot_idx < total_vnodes; i++) {
        if (balancer->nodes[i].status != LB_NODE_ONLINE) continue;
        uint32_t nid = balancer->nodes[i].node_id;
        for (int v = 0; v < balancer->virtual_node_count && slot_idx < total_vnodes; v++, slot_idx++) {
            uint64_t seed = ((uint64_t)nid << 32) | (uint64_t)v;
            uint64_t h = lb_hash_uint64(seed);
            balancer->ring[slot_idx].hash_value = (uint32_t)(h % (uint64_t)LB_RING_SLOTS);
            balancer->ring[slot_idx].node_id = nid;
        }
    }
    balancer->ring_size = slot_idx;
    qsort(balancer->ring, (size_t)balancer->ring_size, sizeof(LbRingSlot), ring_slot_compare);
    balancer->ring_initialized = 1;
    return LB_ERROR_NONE;
}

uint32_t lb_lookup_key(LbBalancer* balancer, const char* key, size_t len) {
    if (!balancer || !balancer->ring_initialized || !key || balancer->ring_size <= 0) {
        return UINT32_MAX;
    }
    uint64_t key_hash = lb_hash_string(key, len);
    uint32_t slot_hash = (uint32_t)(key_hash % (uint64_t)LB_RING_SLOTS);
    LbRingSlot target;
    target.hash_value = slot_hash;
    target.node_id = 0;
    LbRingSlot* found = (LbRingSlot*)bsearch(&target, balancer->ring,
        (size_t)balancer->ring_size, sizeof(LbRingSlot), ring_slot_compare);
    if (found) {
        return found->node_id;
    }
    int left = 0, right = balancer->ring_size - 1, idx = 0;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (balancer->ring[mid].hash_value >= slot_hash) {
            idx = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    if (idx >= balancer->ring_size) idx = 0;
    return balancer->ring[idx].node_id;
}

uint32_t lb_lookup_key_multi(LbBalancer* balancer, const char* key,
                              size_t len, uint32_t* exclude_ids,
                              int exclude_count, uint32_t* out_nodes,
                              int* out_count) {
    if (!balancer || !balancer->ring_initialized || !key || !out_nodes || !out_count) {
        return UINT32_MAX;
    }
    if (balancer->ring_size <= 0) return UINT32_MAX;
    uint32_t primary = lb_lookup_key(balancer, key, len);
    if (primary == UINT32_MAX) return UINT32_MAX;
    out_nodes[0] = primary;
    int count = 1;
    int ring_idx = 0;
    for (int i = 0; i < balancer->ring_size; i++) {
        if (balancer->ring[i].node_id == primary) {
            ring_idx = i;
            break;
        }
    }
    int seen[256];
    int seen_count = 0;
    for (int i = 1; i < balancer->ring_size && count < *out_count; i++) {
        int idx = (ring_idx + i) % balancer->ring_size;
        uint32_t nid = balancer->ring[idx].node_id;
        int excluded = 0;
        for (int e = 0; e < exclude_count; e++) {
            if (exclude_ids && nid == exclude_ids[e]) { excluded = 1; break; }
        }
        int already_seen = 0;
        for (int s = 0; s < seen_count; s++) {
            if (seen[s] == (int)nid) { already_seen = 1; break; }
        }
        if (!excluded && !already_seen && nid != primary) {
            seen[seen_count++] = (int)nid;
            out_nodes[count++] = nid;
        }
    }
    *out_count = count;
    return primary;
}
