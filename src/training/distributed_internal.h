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

/* ZSFZX-FIX-P1-001: NodeConnection/AsyncAllreduceRequest/DistributedContext
 * 完整结构定义已移至 include/selflnn/training/distributed_training.h
 * （受 SELFLNN_IMPLEMENTATION 保护，消除跨模块相对路径引用）。
 * 本头文件仅保留协议级消息类型/消息头定义。 */

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
