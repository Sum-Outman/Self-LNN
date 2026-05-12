#ifndef SELFLNN_LOAD_BALANCER_H
#define SELFLNN_LOAD_BALANCER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 负载均衡器常数 */
#define LB_MAX_NODES 256
#define LB_MAX_TASKS 4096
#define LB_NODE_NAME_LEN 128
#define LB_METRIC_HISTORY 64
#define LB_RING_SLOTS 65537
#define LB_MAX_VIRTUAL_NODES 1024

/* 节点类型 */
typedef enum {
    LB_NODE_GPU_CUDA,
    LB_NODE_GPU_ROCM,
    LB_NODE_GPU_OPENCL,
    LB_NODE_GPU_VULKAN,
    LB_NODE_GPU_INTEL,
    LB_NODE_CPU,
    LB_NODE_REMOTE,
    LB_NODE_CUSTOM
} LbNodeType;

/* 节点状态 */
typedef enum {
    LB_NODE_ONLINE,
    LB_NODE_OFFLINE,
    LB_NODE_DEGRADED,
    LB_NODE_BUSY,
    LB_NODE_REMOVED
} LbNodeStatus;

/* 调度策略 */
typedef enum {
    LB_POLICY_ROUND_ROBIN,
    LB_POLICY_LEAST_LOADED,
    LB_POLICY_WEIGHTED,
    LB_POLICY_RANDOM,
    LB_POLICY_ADAPTIVE,
    LB_POLICY_AFFINITY,
    LB_POLICY_CONSISTENT_HASH
} LbSchedulePolicy;

/* 任务优先级 */
typedef enum {
    LB_PRIORITY_LOW,
    LB_PRIORITY_NORMAL,
    LB_PRIORITY_HIGH,
    LB_PRIORITY_CRITICAL
} LbTaskPriority;

/* 节点负载度量 */
typedef struct {
    double gpu_utilization;
    double memory_utilization;
    double memory_used_mb;
    double memory_total_mb;
    double temperature_celsius;
    double power_watts;
    uint32_t queue_depth;
    double avg_response_time_ms;
    double throughput_tasks_per_sec;
    uint64_t total_tasks_completed;
    uint64_t total_tasks_failed;
    double cpu_utilization;
    double network_bandwidth_mbps;
    uint64_t last_update_time_ms;
} LbNodeMetrics;

/* 节点能力 */
typedef struct {
    double compute_capacity;
    double memory_capacity_gb;
    double memory_bandwidth_gbps;
    double network_bandwidth_gbps;
    int num_cores;
    int supports_fp16;
    int supports_int8;
    int supports_tensor_cores;
    double peak_tflops_fp32;
    double peak_tflops_fp16;
} LbNodeCapability;

/* 负载均衡节点 */
typedef struct {
    uint32_t node_id;
    LbNodeType node_type;
    LbNodeStatus status;
    char name[LB_NODE_NAME_LEN];
    char address[64];
    uint16_t port;
    LbNodeCapability capability;
    LbNodeMetrics metrics;
    LbNodeMetrics metric_history[LB_METRIC_HISTORY];
    uint32_t metric_history_index;
    double weight;
    double current_load;
    uint64_t active_tasks;
    uint64_t total_tasks_assigned;
    uint64_t consecutive_failures;
    uint64_t last_heartbeat_ms;
    int is_local;
} LbNode;

/* 负载均衡任务 */
typedef struct {
    uint64_t task_id;
    LbTaskPriority priority;
    uint32_t estimated_compute_cost;
    uint32_t estimated_memory_mb;
    uint32_t estimated_duration_ms;
    uint32_t assigned_node_id;
    uint64_t submit_time_ms;
    uint64_t start_time_ms;
    uint64_t end_time_ms;
    int completed;
    int failed;
    int affinity_node_id;
    void* user_data;
} LbTask;

/* 负载均衡器配置 */
typedef struct {
    uint32_t max_nodes;
    uint32_t max_tasks;
    uint32_t heartbeat_interval_ms;
    uint32_t health_check_interval_ms;
    uint32_t metric_history_size;
    uint32_t max_failures_before_remove;
    LbSchedulePolicy default_policy;
    double overload_threshold;
    double underload_threshold;
    int enable_auto_rebalance;
    int enable_failover;
    int verbose;
} LbConfig;

/* 负载均衡器统计 */
typedef struct {
    uint64_t total_tasks_submitted;
    uint64_t total_tasks_completed;
    uint64_t total_tasks_failed;
    uint64_t total_tasks_queued;
    uint64_t total_rebalances;
    uint64_t total_failovers;
    double avg_wait_time_ms;
    double avg_execution_time_ms;
    double system_throughput;
    double system_load_avg;
    int online_node_count;
    int offline_node_count;
    uint32_t active_task_count;
    uint32_t queued_task_count;
} LbStats;

/* 负载均衡器 */
typedef struct LbBalancer LbBalancer;

/* 创建和销毁 */
LbBalancer* lb_create(const LbConfig* config);
void lb_destroy(LbBalancer* balancer);

/* 节点管理 */
int lb_add_node(LbBalancer* balancer, uint32_t node_id, LbNodeType type,
                const char* name, const char* address, uint16_t port,
                const LbNodeCapability* capability);
int lb_remove_node(LbBalancer* balancer, uint32_t node_id);
int lb_update_node_metrics(LbBalancer* balancer, uint32_t node_id,
                           const LbNodeMetrics* metrics);
int lb_set_node_weight(LbBalancer* balancer, uint32_t node_id, double weight);
int lb_get_node(const LbBalancer* balancer, uint32_t node_id, LbNode* node);
int lb_get_online_count(const LbBalancer* balancer);

/* 任务管理 */
uint64_t lb_submit_task(LbBalancer* balancer, LbTaskPriority priority,
                        uint32_t compute_cost, uint32_t memory_mb,
                        uint32_t estimated_duration_ms, int affinity_node_id,
                        void* user_data);
int lb_cancel_task(LbBalancer* balancer, uint64_t task_id);
int lb_complete_task(LbBalancer* balancer, uint64_t task_id, int failed);
int lb_get_task(const LbBalancer* balancer, uint64_t task_id, LbTask* task);
int lb_get_queued_task_count(const LbBalancer* balancer);

/* 调度 */
int lb_schedule_next(LbBalancer* balancer, uint32_t* out_node_id);
int lb_schedule_batch(LbBalancer* balancer, uint32_t* task_ids,
                      uint32_t task_count, uint32_t* out_node_ids);
int lb_set_schedule_policy(LbBalancer* balancer, LbSchedulePolicy policy);

/* 健康监控 */
int lb_heartbeat(LbBalancer* balancer, uint32_t node_id);
int lb_check_health(LbBalancer* balancer);
int lb_rebalance(LbBalancer* balancer);
int lb_failover(LbBalancer* balancer, uint32_t failed_node_id);

/* 统计和查询 */
void lb_get_stats(const LbBalancer* balancer, LbStats* stats);
void lb_reset_stats(LbBalancer* balancer);
double lb_get_node_load(const LbBalancer* balancer, uint32_t node_id);
double lb_get_system_load(const LbBalancer* balancer);
int lb_find_best_node(const LbBalancer* balancer, uint32_t* node_id,
                      LbSchedulePolicy policy);

/* 配置辅助 */
void lb_default_config(LbConfig* config);

/* ============================================================================
 * 一致性哈希 (A08.4.2)
 * ============================================================================ */

typedef struct {
    uint32_t hash_value;
    uint32_t node_id;
} LbRingSlot;

int lb_enable_consistent_hashing(LbBalancer* balancer,
                                   int virtual_node_count);
int lb_disable_consistent_hashing(LbBalancer* balancer);
uint64_t lb_hash_string(const char* key, size_t len);
uint32_t lb_hash_uint64(uint64_t key);
int lb_ring_add_node(LbBalancer* balancer, uint32_t node_id);
int lb_ring_remove_node(LbBalancer* balancer, uint32_t node_id);
int lb_ring_rebuild(LbBalancer* balancer);
uint32_t lb_lookup_key(LbBalancer* balancer, const char* key, size_t len);
uint32_t lb_lookup_key_multi(LbBalancer* balancer, const char* key,
                              size_t len, uint32_t* exclude_ids,
                              int exclude_count, uint32_t* out_nodes,
                              int* out_count);

/* 错误字符串 */
const char* lb_error_string(int error_code);

#define LB_ERROR_NONE 0
#define LB_ERROR_INVALID_PARAM -1
#define LB_ERROR_NODE_NOT_FOUND -2
#define LB_ERROR_NODE_OFFLINE -3
#define LB_ERROR_NO_AVAILABLE_NODE -4
#define LB_ERROR_TASK_NOT_FOUND -5
#define LB_ERROR_OUT_OF_MEMORY -6
#define LB_ERROR_QUEUE_FULL -7
#define LB_ERROR_MAX_NODES -8

#ifdef __cplusplus
}
#endif

#endif
