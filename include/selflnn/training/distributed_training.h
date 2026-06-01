/**
 * @file distributed_training.h
 * @brief 分布式训练模块
 *
 * 提供基于TCP网络通信的分布式训练支持。
 * 包含Ring AllReduce、Tree AllReduce、心跳检测、故障恢复、检查点同步等完整功能。
 * 所有网络通信基于platform.h跨平台套接字抽象层，不依赖任何第三方库。
 */

#ifndef SELFLNN_DISTRIBUTED_TRAINING_H
#define SELFLNN_DISTRIBUTED_TRAINING_H

#include "selflnn/core/common.h"
#include "selflnn/utils/platform.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * 常量定义
 * --------------------------------------------------------------------------- */

#define DISTRIBUTED_MAX_NODES        64    /**< 最大支持节点数 */
#define DISTRIBUTED_MAX_MSG_SIZE     (16 * 1024 * 1024) /**< 最大消息大小 16MB */
#define DISTRIBUTED_DEFAULT_PORT     8765  /**< 分布式训练默认端口 */
#define DISTRIBUTED_MAGIC            0x44495354  /**< "DIST" 魔术字 */

/* ---------------------------------------------------------------------------
 * 分布式训练配置
 * --------------------------------------------------------------------------- */

/**
 * @brief 分布式训练节点信息
 */
typedef struct {
    int node_id;                    /**< 节点ID (0-based) */
    char host[64];                  /**< 节点主机名或IP地址 */
    unsigned short port;            /**< 节点监听端口 */
    int is_leader;                  /**< 是否为领导节点 (Node 0) */
    int is_alive;                   /**< 节点是否存活 */
    uint64_t last_heartbeat;        /**< 最后心跳时间戳 (毫秒) */
    uint64_t total_bytes_sent;      /**< 总发送字节数 */
    uint64_t total_bytes_received;  /**< 总接收字节数 */
    int connection_errors;          /**< 连接错误计数 */
    int reconnection_count;         /**< 重连次数 */
} DistributedNodeInfo;

/**
 * @brief 分布式训练附加配置
 */
typedef struct {
    char master_host[64];           /**< 主节点(leader)主机地址 */
    unsigned short master_port;     /**< 主节点监听端口 */
    int num_nodes;                  /**< 总节点数 */
    int node_id;                    /**< 当前节点ID */
    int allreduce_algorithm;        /**< 全归约算法: 0=环, 1=树, 2=混合 */
    int enable_fault_tolerance;     /**< 是否启用容错 */
    int heartbeat_interval_ms;      /**< 心跳间隔毫秒 */
    int heartbeat_timeout_ms;       /**< 心跳超时毫秒 */
    int max_retries;                /**< 最大重试次数 */
    int enable_gradient_compression; /**< 是否启用梯度压缩 */
    float gradient_compression_ratio; /**< 梯度压缩率 0.01~1.0 */
    int sync_frequency;             /**< 同步频率 (每N个批次同步一次) */
    int enable_checkpointing;       /**< 是否启用检查点同步 */
    int checkpoint_frequency;       /**< 检查点保存频率 */
    int verbose;                    /**< 是否输出详细信息 */
    int enable_async_sync;          /**< 是否启用异步梯度同步 */
    char checkpoint_dir[256];       /**< 检查点保存目录 */
    int enable_quorum_consensus;    /**< 是否启用法定人数共识 */
    int quorum_percentage;          /**< 法定人数百分比 (默认51) */
    int enable_gradient_versioning; /**< 是否启用梯度版本追踪 */
    int enable_auto_rebalance;      /**< 是否启用拓扑自动重平衡 */
    int rebalance_check_interval;   /**< 重平衡检查间隔（心跳次数） */
} DistributedConfig;

/**
 * @brief 通信统计信息
 */
typedef struct {
    uint64_t total_bytes_sent;          /**< 总发送字节数 */
    uint64_t total_bytes_received;      /**< 总接收字节数 */
    uint64_t allreduce_count;           /**< 全归约操作次数 */
    uint64_t barrier_count;             /**< 屏障同步次数 */
    uint64_t broadcast_count;           /**< 广播操作次数 */
    uint64_t heartbeat_count;           /**< 心跳消息计数 */
    uint64_t total_communication_time_ms; /**< 总通信时间毫秒 */
    uint64_t last_allreduce_time_ms;    /**< 上次全归约耗时毫秒 */
    int active_connections;             /**< 活跃连接数 */
    int failed_connections;             /**< 失败连接数 */
    int reconnection_count;             /**< 重连次数 */
    int node_failures_detected;         /**< 检测到的节点故障数 */
    int checkpoint_count;               /**< 检查点操作次数 */
    int gradient_sync_count;            /**< 梯度同步次数 */
} DistributedCommStats;

/* ---------------------------------------------------------------------------
 * 分布式上下文 (不透明类型)
 * --------------------------------------------------------------------------- */

typedef struct DistributedContext DistributedContext;

/* ---------------------------------------------------------------------------
 * 初始化和清理
 * --------------------------------------------------------------------------- */

SELFLNN_API DistributedContext* distributed_init(const DistributedConfig* config);

SELFLNN_API void distributed_cleanup(DistributedContext* ctx);

/* ---------------------------------------------------------------------------
 * 拓扑建立
 * --------------------------------------------------------------------------- */

SELFLNN_API int distributed_start_server(DistributedContext* ctx);

SELFLNN_API int distributed_connect_worker(DistributedContext* ctx);

SELFLNN_API int distributed_wait_for_workers(DistributedContext* ctx, int timeout_ms);

SELFLNN_API int distributed_build_ring_topology(DistributedContext* ctx);

/* ---------------------------------------------------------------------------
 * 集合通信操作
 * --------------------------------------------------------------------------- */

SELFLNN_API int distributed_barrier(DistributedContext* ctx);

SELFLNN_API int distributed_broadcast(DistributedContext* ctx, void* data, size_t size, int root_id);

SELFLNN_API int distributed_gather(DistributedContext* ctx, const void* send_data, void* recv_data, size_t size, int root_id);

SELFLNN_API int distributed_allgather(DistributedContext* ctx, const void* send_data, void* recv_data, size_t size);

/* ---------------------------------------------------------------------------
 * 梯度同步 (Ring AllReduce / Tree AllReduce)
 * --------------------------------------------------------------------------- */

SELFLNN_API int distributed_allreduce_ring(DistributedContext* ctx, float* data, size_t count);

SELFLNN_API int distributed_allreduce_tree(DistributedContext* ctx, float* data, size_t count);

SELFLNN_API int distributed_sync_gradients_ex(DistributedContext* ctx, float* gradients, size_t num_parameters, int algorithm);

SELFLNN_API int distributed_allreduce_ring_async(DistributedContext* ctx, float* data, size_t count);

SELFLNN_API int distributed_allreduce_wait(DistributedContext* ctx);

/* ---------------------------------------------------------------------------
 * 心跳和故障检测
 * --------------------------------------------------------------------------- */

SELFLNN_API int distributed_start_heartbeat(DistributedContext* ctx);

SELFLNN_API int distributed_stop_heartbeat(DistributedContext* ctx);

SELFLNN_API int distributed_check_failures(DistributedContext* ctx, int* failed_nodes, int* num_failed);

SELFLNN_API int distributed_recover_node(DistributedContext* ctx, int node_id);

SELFLNN_API int distributed_rebuild_topology(DistributedContext* ctx, const int* failed_nodes, int num_failed);

/* ---------------------------------------------------------------------------
 * 需求20.1: 法定人数共识 — 类型定义
 * --------------------------------------------------------------------------- */

/**
 * @brief 法定人数共识状态
 */
typedef struct {
    int total_nodes;                /**< 总节点数 */
    int alive_nodes;                /**< 当前存活节点数 */
    int quorum_threshold;           /**< 法定人数阈值（需要多少节点存活才能继续） */
    int quorum_met;                 /**< 是否满足法定人数 */
    int consensus_round;            /**< 当前共识轮次 */
    uint64_t last_quorum_check_ms;  /**< 上次检查时间戳 */
    int consecutive_quorum_failures; /**< 连续法定人数失败次数 */
    int in_quorum_recovery;         /**< 是否正在恢复法定人数 */
} DistributedQuorumState;

/**
 * @brief 梯度版本追踪条目
 */
typedef struct {
    uint64_t global_version;        /**< 全局梯度版本号（单调递增） */
    uint64_t node_versions[DISTRIBUTED_MAX_NODES]; /**< 各节点已知的最新版本号 */
    uint64_t last_sync_version;     /**< 上次成功同步的版本号 */
    int stale_node_mask[DISTRIBUTED_MAX_NODES]; /**< 节点是否被标记为过时 */
    int num_stale_nodes;            /**< 过时节点数 */
} GradientVersionTracker;

/**
 * @brief 拓扑重平衡状态
 */
typedef struct {
    int topology_version;           /**< 拓扑版本号（每次变化递增） */
    int current_num_nodes;          /**< 当前实际有效节点数 */
    int node_workload[DISTRIBUTED_MAX_NODES]; /**< 各节点工作负载（样本数） */
    int total_workload;             /**< 总工作负载 */
    int rebalance_count;            /**< 重平衡次数 */
    uint64_t last_rebalance_ms;     /**< 上次重平衡时间戳 */
} DistributedTopologyState;

/* ---------------------------------------------------------------------------
 * 需求20.1: 分布式增强 — 函数声明
 * --------------------------------------------------------------------------- */

/** @brief 检查是否满足法定人数。存活节点 >= ceil(total * ratio/100) */
SELFLNN_API int distributed_quorum_check(DistributedContext* ctx, int* quorum_met);

/** @brief 获取当前法定人数状态 */
SELFLNN_API int distributed_get_quorum_state(DistributedContext* ctx, DistributedQuorumState* state);

/** @brief 设置法定人数百分比阈值（默认51） */
SELFLNN_API int distributed_set_quorum_threshold(DistributedContext* ctx, int threshold_percent);

/** @brief 更新梯度版本（同步前调用，返回当前全局版本号） */
SELFLNN_API uint64_t distributed_gradient_version_update(DistributedContext* ctx, int node_id);

/** @brief 检查指定节点的梯度是否过时。过时返回1，正常返回0 */
SELFLNN_API int distributed_is_gradient_stale(DistributedContext* ctx, int node_id);

/** @brief 获取梯度版本追踪器快照 */
SELFLNN_API int distributed_get_gradient_versions(DistributedContext* ctx, GradientVersionTracker* tracker);

/** @brief 重置过时节点标记（恢复后调用） */
SELFLNN_API int distributed_clear_stale_node(DistributedContext* ctx, int node_id);

/** @brief 自动检测拓扑变化并重平衡 */
SELFLNN_API int distributed_auto_rebalance(DistributedContext* ctx);

/** @brief 获取当前拓扑状态 */
SELFLNN_API int distributed_get_topology_state(DistributedContext* ctx, DistributedTopologyState* state);

/** @brief 手动触发工作负载重平衡 */
SELFLNN_API int distributed_rebalance_workload(DistributedContext* ctx, const int* alive_nodes, int num_alive);

/* ---------------------------------------------------------------------------
 * 检查点操作
 * --------------------------------------------------------------------------- */

SELFLNN_API int distributed_checkpoint_save(DistributedContext* ctx, const char* filepath);

SELFLNN_API int distributed_checkpoint_load(DistributedContext* ctx, const char* filepath);

SELFLNN_API int distributed_checkpoint_leader_sync(DistributedContext* ctx, const char* filepath);

/* ---------------------------------------------------------------------------
 * 状态查询
 * --------------------------------------------------------------------------- */

SELFLNN_API int distributed_get_node_info(DistributedContext* ctx, int node_id, DistributedNodeInfo* info);

SELFLNN_API int distributed_get_comm_stats(DistributedContext* ctx, DistributedCommStats* stats);

SELFLNN_API int distributed_get_connected_count(DistributedContext* ctx);

SELFLNN_API int distributed_is_leader(DistributedContext* ctx);

SELFLNN_API int distributed_get_my_node_id(DistributedContext* ctx);

SELFLNN_API int distributed_set_verbose(DistributedContext* ctx, int verbose);

SELFLNN_API int distributed_elect_leader(DistributedContext* ctx);

/* ---------------------------------------------------------------------------
 * 配置创建
 * --------------------------------------------------------------------------- */

SELFLNN_API DistributedConfig distributed_config_default(void);

/* ---------------------------------------------------------------------------
 * P1-16: UDP广播节点发现
 * --------------------------------------------------------------------------- */

/** 通过UDP广播自动发现局域网内的训练节点 */
SELFLNN_API int distributed_discover_nodes_udp(DistributedContext* ctx, int timeout_ms);

/** 启动节点发现监听器（从节点端） */
SELFLNN_API int distributed_start_discovery_listener(DistributedContext* ctx);

SELFLNN_API int distributed_stop_discovery_listener(DistributedContext* ctx);

/* ---------------------------------------------------------------------------
 * ZSFZX-FIX-P1-001: 完整内部结构定义（仅SELFLNN_IMPLEMENTATION可见）
 * 将原先在src/training/distributed_internal.h中的DistributedContext完整定义
 * 移至此处，消除core/extended_training.c的跨模块相对路径引用。
 * 外部API使用者仅能通过不透明指针访问，内部实现模块可获得完整结构体。
 * --------------------------------------------------------------------------- */
#ifdef SELFLNN_IMPLEMENTATION

typedef struct {
    int node_id;
    SocketHandle socket;
    char host[64];
    unsigned short port;
    uint64_t last_heartbeat;
    uint64_t bytes_sent;
    uint64_t bytes_received;
    int connected;
    int is_alive;
    int connection_errors;
    int reconnection_count;
    MutexHandle lock;
} NodeConnection;

typedef struct {
    float* data;
    size_t count;
    int algorithm;
    volatile int completed;
    volatile int result;
    MutexHandle lock;
    struct DistributedContext* ctx;
} AsyncAllreduceRequest;

struct DistributedContext {
    DistributedConfig config;
    int num_nodes;
    int my_node_id;
    int is_leader;
    NodeConnection nodes[DISTRIBUTED_MAX_NODES];
    MutexHandle nodes_lock;
    SocketHandle server_socket;
    int server_running;
    MutexHandle server_lock;
    int ring_established;
    int ring_successor;
    int ring_predecessor;
    int tree_established;
    int tree_parent;
    int tree_left_child;
    int tree_right_child;
    uint32_t barrier_counter;
    uint32_t barrier_generation;
    MutexHandle barrier_lock;
    volatile int heartbeat_running;
    ThreadHandle heartbeat_thread;
    MutexHandle heartbeat_lock;
    volatile int async_in_progress;
    AsyncAllreduceRequest* async_req;
    ThreadHandle async_thread;
    MutexHandle async_lock;
    DistributedCommStats stats;
    MutexHandle stats_lock;
    char checkpoint_dir[256];
    float* retry_gradient_buffer;
    size_t retry_buffer_size;
    DistributedQuorumState quorum;
    int quorum_threshold_percent;
    GradientVersionTracker grad_versions;
    uint64_t global_version_counter;
    DistributedTopologyState topo;
    int rebalance_check_counter;
    volatile int discovery_listener_running;
    ThreadHandle discovery_listener_thread;
};

#endif /* SELFLNN_IMPLEMENTATION */

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_DISTRIBUTED_TRAINING_H */
