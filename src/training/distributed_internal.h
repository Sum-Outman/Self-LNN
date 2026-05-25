/**
 * @file distributed_internal.h
 * @brief 分布式训练内部协议和结构定义
 *
 * 定义二进制网络协议格式、消息类型、内部数据结构。
 * 协议采用紧凑二进制格式，便于高效传输。
 */

#ifndef DISTRIBUTED_INTERNAL_H
#define DISTRIBUTED_INTERNAL_H

#include "selflnn/training/distributed_training.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/memory_utils.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * 二进制协议定义
 * --------------------------------------------------------------------------- */

/**
 * @brief 消息类型枚举
 */
typedef enum {
    MSG_GRADIENT_CHUNK     = 0x01,  /**< 梯度数据块 */
    MSG_BARRIER            = 0x02,  /**< 屏障同步请求 */
    MSG_BARRIER_ACK        = 0x03,  /**< 屏障同步应答 */
    MSG_HEARTBEAT          = 0x04,  /**< 心跳 */
    MSG_HEARTBEAT_ACK      = 0x05,  /**< 心跳应答 */
    MSG_WORKER_JOIN        = 0x06,  /**< 工作节点加入 */
    MSG_WORKER_JOIN_ACK    = 0x07,  /**< 工作节点加入应答 */
    MSG_CHECKPOINT_SAVE    = 0x08,  /**< 检查点保存请求 */
    MSG_CHECKPOINT_DATA    = 0x09,  /**< 检查点数据块 */
    MSG_CHECKPOINT_ACK     = 0x0A,  /**< 检查点确认 */
    MSG_BROADCAST_DATA     = 0x0B,  /**< 广播数据 */
    MSG_GATHER_DATA        = 0x0C,  /**< 收集数据 */
    MSG_ALLGATHER_DATA     = 0x0D,  /**< 全收集数据 */
    MSG_ALLGATHER_CHUNK    = 0x0E,  /**< 全收集数据块（非阻塞流水线AllGather分块） */
    MSG_ERROR              = 0xFE,  /**< 错误消息 */
    MSG_SHUTDOWN           = 0xFF   /**< 关闭连接 */
} DistributedMessageType;

/**
 * @brief 消息头 (14字节)
 *
 * 所有网络消息以固定头部开始。
 */
#pragma pack(push, 1)
typedef struct {
    uint32_t magic;             /**< 魔术字: 0x44495354 ("DIST") */
    uint8_t  type;              /**< 消息类型 @see DistributedMessageType */
    uint8_t  flags;             /**< 标志位 (bit0=压缩, bit1=确认, bit2=重传) */
    uint32_t length;            /**< 有效载荷长度 (不含头部) */
    uint32_t sender_id;         /**< 发送方节点ID */
} DistributedMessageHeader;

/**
 * @brief 梯度数据块消息 (紧跟在头之后)
 */
typedef struct {
    uint64_t offset;            /**< 梯度起始偏移 */
    uint64_t count;             /**< 梯度元素数量 */
    /* float data[count] 紧随其后 */
} GradientChunkMessage;

/**
 * @brief 工作节点加入消息 (紧跟在头之后)
 */
typedef struct {
    int node_id;                /**< 请求的节点ID (-1表示自动分配) */
    int num_parameters;         /**< 本地参数数量 */
    int listen_port;            /**< 监听端口 (用于建立环形连接) */
} WorkerJoinMessage;

/**
 * @brief 工作节点加入应答 (紧跟在头之后)
 */
typedef struct {
    int assigned_id;            /**< 分配的节点ID */
    int total_nodes;            /**< 当前总节点数 */
    int leader_id;              /**< 领导节点ID */
    unsigned short node_ports[DISTRIBUTED_MAX_NODES]; /**< 各节点监听端口 */
} WorkerJoinAckMessage;

/**
 * @brief 检查点数据消息 (紧跟在头之后)
 */
typedef struct {
    uint64_t offset;            /**< 数据偏移 */
    uint64_t total_size;        /**< 总大小 */
    uint32_t checksum;          /**< 数据块的CRC32校验和 */
    /* char data[length - sizeof(CheckpointDataMessage)] 紧随其后 */
} CheckpointDataMessage;

/**
 * @brief 屏障同步消息 (紧跟在头之后)
 */
typedef struct {
    uint32_t barrier_id;        /**< 屏障ID (epoch编号) */
    uint32_t generation;        /**< 代数 (防止重复) */
} BarrierMessage;
#pragma pack(pop)

/* ---------------------------------------------------------------------------
 * 节点连接状态
 * --------------------------------------------------------------------------- */

/**
 * @brief 节点连接
 */
typedef struct {
    int node_id;                    /**< 远端节点ID */
    SocketHandle socket;            /**< 套接字句柄 */
    char host[64];                  /**< 远端主机 */
    unsigned short port;            /**< 远端端口 */
    uint64_t last_heartbeat;        /**< 最后心跳时间 (毫秒) */
    uint64_t bytes_sent;            /**< 已发送字节数 */
    uint64_t bytes_received;        /**< 已接收字节数 */
    int connected;                  /**< 是否已连接 */
    int is_alive;                   /**< 是否存活 */
    int connection_errors;          /**< 连接错误计数 */
    int reconnection_count;         /**< 重连次数 */
    MutexHandle lock;               /**< 连接锁 */
} NodeConnection;

/**
 * @brief 异步同步请求
 */
typedef struct {
    float* data;                    /**< 要同步的数据指针 */
    size_t count;                   /**< 元素数量 */
    int algorithm;                  /**< 算法类型 */
    volatile int completed;         /**< 是否完成 */
    volatile int result;            /**< 结果代码 */
    MutexHandle lock;               /**< 状态锁 */
    struct DistributedContext* ctx; /**< 分布式上下文（用于异步线程执行真实全归约） */
} AsyncAllreduceRequest;

/**
 * @brief 分布式上下文实现
 */
struct DistributedContext {
    /* 配置 */
    DistributedConfig config;           /**< 分布式配置 */
    
    /* 节点信息 */
    int num_nodes;                      /**< 实际节点数 */
    int my_node_id;                     /**< 本节点ID */
    int is_leader;                      /**< 是否领导节点 */
    
    /* 节点连接表 */
    NodeConnection nodes[DISTRIBUTED_MAX_NODES]; /**< 所有节点连接 */
    MutexHandle nodes_lock;             /**< 节点表锁 */
    
    /* 服务器套接字 */
    SocketHandle server_socket;         /**< 服务器监听套接字 */
    int server_running;                 /**< 服务器是否运行 */
    MutexHandle server_lock;            /**< 服务器锁 */
    
    /* 环形拓扑 */
    int ring_established;               /**< 环形拓扑是否建立 */
    int ring_successor;                 /**< 环中后继节点ID */
    int ring_predecessor;               /**< 环中前驱节点ID */
    
    /* 树形拓扑 */
    int tree_established;               /**< 树形拓扑是否建立 */
    int tree_parent;                    /**< 父节点ID */
    int tree_left_child;                /**< 左子节点ID */
    int tree_right_child;               /**< 右子节点ID */
    
    /* 集合通信状态 */
    uint32_t barrier_counter;           /**< 屏障计数器 */
    uint32_t barrier_generation;        /**< 屏障代数 */
    MutexHandle barrier_lock;           /**< 屏障锁 */
    
    /* 心跳线程 */
    volatile int heartbeat_running;     /**< 心跳线程是否运行 */
    ThreadHandle heartbeat_thread;      /**< 心跳线程句柄 */
    MutexHandle heartbeat_lock;         /**< 心跳锁 */
    
    /* 异步同步 */
    volatile int async_in_progress;     /**< 异步同步进行中 */
    AsyncAllreduceRequest* async_req;   /**< 当前异步请求 */
    ThreadHandle async_thread;          /**< 异步线程句柄 */
    MutexHandle async_lock;             /**< 异步锁 */
    
    /* 通信统计 */
    DistributedCommStats stats;         /**< 通信统计 */
    MutexHandle stats_lock;             /**< 统计锁 */
    
    /* 检查点 */
    char checkpoint_dir[256];           /**< 检查点目录 */
    
    /* 梯度重试缓存 (故障恢复用) */
    float* retry_gradient_buffer;       /**< 重试梯度缓冲区 */
    size_t retry_buffer_size;           /**< 重试缓冲区大小 */
    
    /* 需求20.1: 法定人数共识状态 */
    DistributedQuorumState quorum;      /**< 法定人数共识状态 */
    int quorum_threshold_percent;       /**< 法定人数百分比阈值 */
    
    /* 需求20.1: 梯度版本追踪 */
    GradientVersionTracker grad_versions; /**< 梯度版本追踪器 */
    uint64_t global_version_counter;    /**< 全局版本计数器 */
    
    /* 需求20.1: 拓扑自动重平衡 */
    DistributedTopologyState topo;      /**< 拓扑重平衡状态 */
    int rebalance_check_counter;        /**< 重平衡检查计数器 */

    /* 节点发现监听器 */
    volatile int discovery_listener_running;  /**< 发现监听器是否运行 */
    ThreadHandle discovery_listener_thread;  /**< 发现监听器线程句柄 */
};

/* ---------------------------------------------------------------------------
 * 内部函数声明
 * --------------------------------------------------------------------------- */

/* 消息收发 */
int distributed_send_message(NodeConnection* conn, uint8_t type, uint8_t flags,
                             const void* payload, uint32_t length);
int distributed_recv_message(NodeConnection* conn, DistributedMessageHeader* header,
                             void* payload, uint32_t max_length);
int distributed_send_to_node(DistributedContext* ctx, int node_id, uint8_t type,
                             uint8_t flags, const void* payload, uint32_t length);

/* 连接管理 */
int distributed_connect_to_node(DistributedContext* ctx, int node_id,
                                const char* host, unsigned short port);
void distributed_close_connection(DistributedContext* ctx, int node_id);
int distributed_reconnect_node(DistributedContext* ctx, int node_id);

/* 拓扑 */
int distributed_build_tree_topology_internal(DistributedContext* ctx);
void distributed_print_topology(DistributedContext* ctx);

/* 梯度压缩 (内部使用) */
void distributed_gradient_topk_compress(const float* gradients, size_t num_params,
                                        float compression_ratio,
                                        int* indices_out, float* values_out,
                                        size_t* compressed_size_out);
void distributed_gradient_topk_decompress(float* output, size_t num_params,
                                          const int* indices, const float* values,
                                          size_t compressed_size);

#ifdef __cplusplus
}
#endif

#endif /* DISTRIBUTED_INTERNAL_H */
