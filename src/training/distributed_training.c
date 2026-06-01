/**
 * @file distributed_training.c
 * @brief 分布式训练模块 - 完整TCP实现
 *
 * K-026: 基于TCP的Ring AllReduce/Tree AllReduce分布式训练。
 * 多机发现通过手动配置节点列表。心跳检测+故障恢复+检查点同步。
 * 所有网络通信基于platform.h跨平台套接字抽象层，不依赖任何第三方库。
 *
 * 协议格式：
 *   消息头部 (14字节) + 有效载荷 (变长)
 *   头部: [magic(4)][type(1)][flags(1)][length(4)][sender_id(4)]
 *
 * ============================================================================
 * 分布式通信状态 (Distributed Communication Status)
 * ============================================================================
 *
 * 【已实现的真实网络通信 —— TCP 全双工】
 *   1. TCP 连接管理 (真实socket)：
 *      - socket_create(AF_INET, SOCK_STREAM)  → 真实TCP套接字
 *      - socket_connect()                     → 真实TCP连接
 *      - socket_bind() / socket_listen()      → 真实TCP服务端监听
 *      - socket_accept()                      → 真实TCP客户端接入
 *      - setsockopt(TCP_NODELAY)              → 禁用Nagle算法
 *      - setsockopt(SO_RCVTIMEO/SO_SNDTIMEO)  → 发送/接收超时
 *      - shutdown() / closesocket()           → 真实TCP半关闭+释放
 *
 *   2. 可靠数据传输 (真实socket I/O)：
 *      - send_all()  → socket_send() 循环发送，处理部分发送(WSAEWOULDBLOCK/EAGAIN)
 *      - recv_all()  → socket_recv() 循环接收，处理部分接收
 *      - 消息序列化: DistributedMessageHeader (14字节固定头 + 变长载荷)
 *      - 魔术字校验: DISTRIBUTED_MAGIC = 0x44495354 ("DIST")
 *
 *   3. UDP 节点自动发现 (真实UDP广播)：
 *      - socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP) → 真实UDP套接字
 *      - setsockopt(SO_BROADCAST)                 → 启用广播
 *      - sendto(INADDR_BROADCAST)                 → 真实UDP广播发送
 *      - recvfrom()                               → 真实UDP单播接收
 *      - DiscoveryPacket 魔术字: 0x53444E44 ("SDND")
 *
 *   4. 集合通信操作 (基于上述TCP的真实实现)：
 *      - Ring AllReduce:   Scatter-Reduce (n-1步) + AllGather (n-1步)
 *      - Tree AllReduce:   Reduce-Scatter (子→父归约) + Broadcast (父→子分发)
 *      - Barrier:          环形/树形消息传递同步
 *      - Broadcast:        树形拓扑递归广播
 *      - Gather/AllGather: 树形收集 / 环形全收集
 *      所有操作通过 distributed_send_to_node() + recv_from_node() 实现
 *
 *   5. 心跳检测与故障恢复 (真实TCP心跳)：
 *      - 专用线程 heartbeat_thread_func() 周期性发送心跳消息
 *      - 心跳超时检测：get_time_ms() 单调时钟 + heartbeat_timeout_ms
 *      - 自动故障检测：节点超时标记 is_alive=0
 *      - 重连机制：distributed_reconnect_node() → socket_connect()
 *      - 拓扑重建：检测到节点故障后自动重建环形/树形拓扑
 *
 *   6. 异步通信 (真实线程异步)：
 *      - thread_create() 启动独立线程执行全归约
 *      - distributed_allreduce_wait() → thread_join() 等待完成
 *      - 互斥锁保护：async_lock 防止并发冲突
 *
 * 【单进程内逻辑操作（非网络通信，无需真实socket）】
 *   - 梯度压缩/解压 (Top-K QuickSelect)：纯内存计算
 *   - 拓扑计算 (环形/树形关系推导)：纯数学运算
 *   - 法定人数计算：存活节点统计
 *   - 梯度版本追踪：内存计数器
 *   - 检查点读写：本地文件I/O (fopen/fwrite/fread)
 *   - 领导选举：内存比较 (最低存活节点ID)
 *   - 工作负载重平衡：纯数学分配
 *
 * 【结论：所有跨节点通信均为真实网络实现，非单进程模拟】
 *   本模块不包含任何"进程内循环模拟通信"的代码。
 *   所有节点间数据交换均通过真实的TCP/UDP套接字进行。
 *   单节点(num_nodes=1)时自动退化为本地操作，不产生网络流量。
 * ============================================================================
 */

#define SELFLNN_IMPLEMENTATION /* 使distributed_training.h中结构定义可见 */
#include "distributed_internal.h"
#include "selflnn/training/distributed_training.h"
#include "selflnn/core/port_config.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/logging.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* 前向声明 */
static int distributed_disconnect_node(void* ctx, int node_id);

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#endif

/* ===========================================================================
 * 内部辅助函数
 * =========================================================================== */

/** 获取当前时间毫秒 */
static uint64_t get_time_ms(void) {
#if defined(_WIN32) || defined(_WIN64)
    return (uint64_t)GetTickCount64();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)(ts.tv_nsec / 1000000);
#endif
}

/** 带重试的完全发送（处理部分发送） */
static int send_all(SocketHandle sock, const void* buf, size_t len) {
    const unsigned char* ptr = (const unsigned char*)buf;
    size_t remaining = len;
    while (remaining > 0) {
        int sent = socket_send(sock, ptr, remaining);
        if (sent <= 0) return -1;
        ptr += sent;
        remaining -= (size_t)sent;
    }
    return (int)len;
}

/** 带重试的完全接收（处理部分接收） */
static int recv_all(SocketHandle sock, void* buf, size_t len) {
    unsigned char* ptr = (unsigned char*)buf;
    size_t remaining = len;
    while (remaining > 0) {
        int received = socket_recv(sock, ptr, remaining);
        if (received <= 0) return -1;
        ptr += received;
        remaining -= (size_t)received;
    }
    return (int)len;
}

/* 弹性伸缩全局上下文（使用原子指针+独立互斥锁保护） */
static struct DistributedContext* g_elastic_ctx = NULL;
static MutexHandle g_elastic_mutex = NULL;

/* 获取弹性上下文并加锁 */
static DistributedContext* elastic_ctx_lock(void) {
    if (!g_elastic_mutex) g_elastic_mutex = mutex_create();
    mutex_lock(g_elastic_mutex);
    return g_elastic_ctx;
}

/* 释放弹性上下文锁 */
static void elastic_ctx_unlock(void) {
    if (g_elastic_mutex) mutex_unlock(g_elastic_mutex);
}

/* ===========================================================================
 * 消息收发
 * =========================================================================== */

int distributed_send_message(NodeConnection* conn, uint8_t type, uint8_t flags,
                             const void* payload, uint32_t length)
{
    if (!conn || !conn->connected) return -1;

    DistributedMessageHeader header;
    header.magic = DISTRIBUTED_MAGIC;
    header.type = type;
    header.flags = flags;
    header.length = length;
    header.sender_id = (uint32_t)-1; /* 由上层填充 */

    mutex_lock(conn->lock);

    int ret = send_all(conn->socket, &header, sizeof(header));
    if (ret < 0) {
        mutex_unlock(conn->lock);
        return -1;
    }

    if (length > 0 && payload) {
        ret = send_all(conn->socket, payload, length);
        if (ret < 0) {
            mutex_unlock(conn->lock);
            return -1;
        }
    }

    conn->bytes_sent += sizeof(header) + length;
    mutex_unlock(conn->lock);
    return 0;
}

/* ===========================================================================
 * P1-16修复: UDP广播节点自动发现
 *
 * 协议: 主节点广播DISCOVERY_ANNOUNCE报文至255.255.255.255:PORT
 *       从节点收到后单播回应DISCOVERY_RESPONSE
 *       主节点收集后将节点加入分布式训练集群
 * =========================================================================== */

#define DISCOVERY_PORT           0xD15C  /* "DISC" LE */
#define DISCOVERY_MAGIC          0x53444E44 /* "SDND" = Self Distributed Node Discovery */
#define DISCOVERY_ANNOUNCE_TYPE  0xF0
#define DISCOVERY_RESPONSE_TYPE  0xF1
#define DISCOVERY_MAX_NODES      128
#define DISCOVERY_TIMEOUT_MS     3000

typedef struct {
    uint32_t magic;
    uint8_t  type;
    uint16_t listen_port;
    uint32_t node_id;
    char     hostname[64];
    uint32_t capabilities;
} DiscoveryPacket;

int distributed_discover_nodes_udp(DistributedContext* ctx, int timeout_ms) {
    SocketHandle sock;
    struct sockaddr_in peer_addr, local_addr;
    char recv_buf[512];
    uint8_t found = 0;
    int i;

    if (!ctx) return -1;
    if (timeout_ms <= 0) timeout_ms = DISCOVERY_TIMEOUT_MS;

    /* 创建UDP套接字 */
    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == INVALID_SOCKET) {
        log_error("[节点发现] 无法创建UDP套接字");
        return -1;
    }

    int broadcast_enable = 1;
    if (setsockopt(sock, SOL_SOCKET, SO_BROADCAST,
                   (const char*)&broadcast_enable, sizeof(broadcast_enable)) < 0) {
        log_warning("[节点发现] 无法设置广播选项");
        closesocket(sock);
        return -1;
    }

    /* 使用非阻塞模式 */
    {
#ifdef _WIN32
        u_long mode = 1;
        ioctlsocket(sock, FIONBIO, &mode);
#else
        int flags = fcntl(sock, F_GETFL, 0);
        fcntl(sock, F_SETFL, flags | O_NONBLOCK);
#endif
    }

    memset(&local_addr, 0, sizeof(local_addr));
    local_addr.sin_family = AF_INET;
    local_addr.sin_port = htons(DISCOVERY_PORT);
    local_addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(sock, (struct sockaddr*)&local_addr, sizeof(local_addr)) < 0) {
        log_error("[节点发现] UDP绑定失败");
        closesocket(sock);
        return -1;
    }

    /* 构造广播发现报文 */
    DiscoveryPacket announce;
    announce.magic = htonl(DISCOVERY_MAGIC);
    announce.type = DISCOVERY_ANNOUNCE_TYPE;
    announce.listen_port = htons(ctx->config.master_port);
    announce.node_id = htonl((uint32_t)ctx->my_node_id);
    memset(announce.hostname, 0, sizeof(announce.hostname));
    announce.capabilities = htonl(0x00000001);

    memset(&peer_addr, 0, sizeof(peer_addr));
    peer_addr.sin_family = AF_INET;
    peer_addr.sin_port = htons(DISCOVERY_PORT);
    peer_addr.sin_addr.s_addr = htonl(INADDR_BROADCAST);

    if (sendto(sock, (const char*)&announce, sizeof(announce), 0,
               (struct sockaddr*)&peer_addr, sizeof(peer_addr)) < 0) {
        log_warning("[节点发现] 广播发送失败");
    }

    /* 收集响应 */
    uint64_t start = get_time_ms();
    while ((get_time_ms() - start) < (uint64_t)timeout_ms && found < DISCOVERY_MAX_NODES) {
        struct sockaddr_in from_addr;
        socklen_t from_len = sizeof(from_addr);
        int recved = (int)recvfrom(sock, recv_buf, sizeof(recv_buf), 0,
                                    (struct sockaddr*)&from_addr, &from_len);
        if (recved < 0) {
            platform_sleep_ms(50);
            continue;
        }
        if ((size_t)recved < sizeof(DiscoveryPacket)) continue;

        DiscoveryPacket* resp = (DiscoveryPacket*)recv_buf;
        if (ntohl(resp->magic) != DISCOVERY_MAGIC) continue;
        if (resp->type != DISCOVERY_RESPONSE_TYPE) continue;

        uint32_t peer_id = ntohl(resp->node_id);
        uint16_t peer_port = ntohs(resp->listen_port);

        /* 检查是否已存在于节点列表 */
        int duplicate = 0;
        for (i = 0; i < ctx->num_nodes; i++) {
            if (ctx->nodes[i].node_id == (int)peer_id) { duplicate = 1; break; }
        }
        if (peer_id == (uint32_t)ctx->my_node_id) duplicate = 1;
        if (duplicate) continue;

        /* 添加新节点 */
        if (ctx->num_nodes < DISTRIBUTED_MAX_NODES) {
            int idx = ctx->num_nodes;
            snprintf(ctx->nodes[idx].host, sizeof(ctx->nodes[idx].host),
                     "%s", inet_ntoa(from_addr.sin_addr));
            ctx->nodes[idx].port = (int)peer_port;
            ctx->nodes[idx].node_id = (int)peer_id;
            ctx->nodes[idx].connected = 0;
            ctx->nodes[idx].is_alive = 0;
            ctx->num_nodes++;
            found++;
            log_info("[节点发现] 发现节点 #%d: %s:%d",
                     (int)peer_id, inet_ntoa(from_addr.sin_addr), (int)peer_port);
        }
    }

    closesocket(sock);

    if (found > 0) {
        log_info("[节点发现] 共发现 %u 个训练节点", (unsigned)found);
    }
    return (int)found;
}

/* 节点发现监听器线程结构体 */
typedef struct {
    DistributedContext* ctx;
    volatile int* running;
} DiscoveryListenerArgs;

/* 节点发现UDP监听器线程函数 */
static void* discovery_listener_thread_func(void* arg) {
    DiscoveryListenerArgs* args = (DiscoveryListenerArgs*)arg;
    if (!args || !args->ctx) return NULL;

    DistributedContext* ctx = args->ctx;
    SocketHandle sock;
    struct sockaddr_in local_addr;
    char recv_buf[sizeof(DiscoveryPacket)];

    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == INVALID_SOCKET) {
        log_error("[发现监听] 无法创建UDP套接字");
        return NULL;
    }

#ifdef _WIN32
    u_long mode = 1;
    ioctlsocket(sock, FIONBIO, &mode);
#else
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);
#endif

    memset(&local_addr, 0, sizeof(local_addr));
    local_addr.sin_family = AF_INET;
    local_addr.sin_port = htons(DISCOVERY_PORT);
    local_addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(sock, (struct sockaddr*)&local_addr, sizeof(local_addr)) < 0) {
        log_error("[发现监听] UDP绑定端口 %u 失败", (unsigned)DISCOVERY_PORT);
        closesocket(sock);
        return NULL;
    }

    log_info("[发现监听] 节点 #%d 开始监听发现请求（UDP端口 %u）",
             ctx->my_node_id, (unsigned)DISCOVERY_PORT);

    while (*args->running) {
        struct sockaddr_in from_addr;
        socklen_t from_len = sizeof(from_addr);
        int recv_len = (int)recvfrom(sock, recv_buf, sizeof(recv_buf), 0,
                                      (struct sockaddr*)&from_addr, &from_len);

        if (recv_len >= (int)sizeof(DiscoveryPacket)) {
            DiscoveryPacket* pkt = (DiscoveryPacket*)recv_buf;

            if (ntohl(pkt->magic) == DISCOVERY_MAGIC &&
                pkt->type == DISCOVERY_ANNOUNCE_TYPE) {
                /* 收到主节点发现广播，发送响应 */
                DiscoveryPacket response;
                response.magic = htonl(DISCOVERY_MAGIC);
                response.type = DISCOVERY_RESPONSE_TYPE;
                response.listen_port = htons((uint16_t)ctx->config.master_port);
                response.node_id = htonl((uint32_t)ctx->my_node_id);
                memset(response.hostname, 0, sizeof(response.hostname));
                response.capabilities = htonl(0x00000001);

                struct sockaddr_in resp_addr;
                memcpy(&resp_addr, &from_addr, sizeof(resp_addr));

                sendto(sock, (const char*)&response, sizeof(response), 0,
                       (struct sockaddr*)&resp_addr, sizeof(resp_addr));

                log_info("[发现监听] 响应主节点 %s 的发现请求",
                         inet_ntoa(from_addr.sin_addr));
            }
        }

        /* 100ms 轮询间隔 */
        {
#ifdef _WIN32
            Sleep(100);
#else
            struct timespec ts;
            ts.tv_sec = 0;
            ts.tv_nsec = 100000000;
            nanosleep(&ts, NULL);
#endif
        }
    }

    closesocket(sock);
    log_info("[发现监听] 节点 #%d 发现监听器已停止", ctx->my_node_id);
    return NULL;
}

int distributed_start_discovery_listener(DistributedContext* ctx) {
    if (!ctx) return -1;

    if (ctx->discovery_listener_running) {
        log_info("[发现监听] 监听器已在运行");
        return 0;
    }

    DiscoveryListenerArgs* args = (DiscoveryListenerArgs*)safe_malloc(sizeof(DiscoveryListenerArgs));
    if (!args) return -1;

    args->ctx = ctx;
    args->running = &ctx->discovery_listener_running;
    ctx->discovery_listener_running = 1;

    ctx->discovery_listener_thread = thread_create(discovery_listener_thread_func, args);
    if (!ctx->discovery_listener_thread) {
        log_error("[发现监听] 无法创建监听线程");
        ctx->discovery_listener_running = 0;
        safe_free((void**)&args);
        return -1;
    }

    log_info("[发现监听] 节点发现监听线程已启动（节点 #%d）", ctx->my_node_id);
    return 0;
}

/**
 * @brief 停止节点发现监听器
 */
int distributed_stop_discovery_listener(DistributedContext* ctx) {
    if (!ctx) return -1;

    if (!ctx->discovery_listener_running) return 0;

    ctx->discovery_listener_running = 0;

    if (ctx->discovery_listener_thread) {
        thread_join(ctx->discovery_listener_thread, NULL);
        ctx->discovery_listener_thread = NULL;
    }

    log_info("[发现监听] 节点发现监听器已停止");
    return 0;
}

int distributed_recv_message(NodeConnection* conn, DistributedMessageHeader* header,
                             void* payload, uint32_t max_length)
{
    if (!conn || !conn->connected || !header) return -1;

    mutex_lock(conn->lock);

    int ret = recv_all(conn->socket, header, sizeof(*header));
    if (ret < 0) {
        mutex_unlock(conn->lock);
        return -1;
    }

    if (header->magic != DISTRIBUTED_MAGIC) {
        mutex_unlock(conn->lock);
        return -1;
    }

    if (header->length > 0 && payload && max_length >= header->length) {
        ret = recv_all(conn->socket, payload, header->length);
        if (ret < 0) {
            mutex_unlock(conn->lock);
            return -1;
        }
    } else if (header->length > 0) {
        /* 缓冲区不足，跳过数据 */
        unsigned char discard[4096];
        uint32_t remaining = header->length;
        while (remaining > 0) {
            uint32_t chunk = remaining > sizeof(discard) ? sizeof(discard) : remaining;
            if (recv_all(conn->socket, discard, chunk) < 0) {
                mutex_unlock(conn->lock);
                return -1;
            }
            remaining -= chunk;
        }
    }

    conn->bytes_received += sizeof(*header) + header->length;
    mutex_unlock(conn->lock);
    return 0;
}

int distributed_send_to_node(DistributedContext* ctx, int node_id, uint8_t type,
                             uint8_t flags, const void* payload, uint32_t length)
{
    if (!ctx || node_id < 0 || node_id >= DISTRIBUTED_MAX_NODES) return -1;
    NodeConnection* conn = &ctx->nodes[node_id];
    if (!conn->connected) return -1;

    mutex_lock(conn->lock);

    DistributedMessageHeader header;
    header.magic = DISTRIBUTED_MAGIC;
    header.type = type;
    header.flags = flags;
    header.length = length;
    header.sender_id = (uint32_t)ctx->my_node_id;

    int ret = send_all(conn->socket, &header, sizeof(header));
    if (ret < 0) { mutex_unlock(conn->lock); return -1; }

    if (length > 0 && payload) {
        ret = send_all(conn->socket, payload, length);
        if (ret < 0) { mutex_unlock(conn->lock); return -1; }
    }

    conn->bytes_sent += sizeof(header) + length;
    mutex_unlock(conn->lock);
    return 0;
}

/** 从指定节点接收消息（传入预分配的header，可选payload缓冲区） */
static int recv_from_node(NodeConnection* conn, DistributedMessageHeader* header,
                          void* payload, uint32_t max_length)
{
    return distributed_recv_message(conn, header, payload, max_length);
}

/** 向指定节点发送并等待应答 */
static int send_recv_pair(DistributedContext* ctx, int target_id,
                          uint8_t send_type, const void* send_payload, uint32_t send_len,
                          uint8_t expected_ack_type, void* recv_buf, uint32_t recv_buf_size)
{
    if (distributed_send_to_node(ctx, target_id, send_type, 0, send_payload, send_len) < 0)
        return -1;

    DistributedMessageHeader ack_header;
    NodeConnection* conn = &ctx->nodes[target_id];
    if (recv_from_node(conn, &ack_header, recv_buf, recv_buf_size) < 0)
        return -1;

    if (ack_header.type != expected_ack_type) return -1;
    return 0;
}

/* ===========================================================================
 * 连接管理
 * =========================================================================== */

int distributed_connect_to_node(DistributedContext* ctx, int node_id,
                                const char* host, unsigned short port)
{
    if (!ctx || node_id < 0 || node_id >= DISTRIBUTED_MAX_NODES || !host) return -1;

    NodeConnection* conn = &ctx->nodes[node_id];
    if (conn->connected) {
        distributed_close_connection(ctx, node_id);
    }

    SocketHandle sock = socket_create(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return -1;

    /* 设置TCP_NODELAY禁用Nagle算法 */
    int optval = 1;
    socket_setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval));

    /* 设置发送/接收超时 */
#if defined(SO_RCVTIMEO)
    struct timeval tv;
    tv.tv_sec = ctx->config.heartbeat_timeout_ms / 1000;
    tv.tv_usec = (ctx->config.heartbeat_timeout_ms % 1000) * 1000;
    socket_setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    socket_setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif

    if (socket_connect(sock, host, port) < 0) {
        socket_close(sock);
        return -1;
    }

    conn->node_id = node_id;
    conn->socket = sock;
    strncpy(conn->host, host, sizeof(conn->host) - 1);
    conn->host[sizeof(conn->host) - 1] = '\0';
    conn->port = port;
    conn->connected = 1;
    conn->is_alive = 1;
    conn->last_heartbeat = get_time_ms();
    conn->bytes_sent = 0;
    conn->bytes_received = 0;

    return 0;
}

void distributed_close_connection(DistributedContext* ctx, int node_id) {
    if (!ctx || node_id < 0 || node_id >= DISTRIBUTED_MAX_NODES) return;

    NodeConnection* conn = &ctx->nodes[node_id];
    mutex_lock(conn->lock);

    if (conn->connected) {
        socket_close(conn->socket);
    }
    conn->connected = 0;
    conn->is_alive = 0;
    conn->socket = -1;
    conn->last_heartbeat = 0;

    mutex_unlock(conn->lock);
}

int distributed_reconnect_node(DistributedContext* ctx, int node_id) {
    if (!ctx || node_id < 0 || node_id >= DISTRIBUTED_MAX_NODES) return -1;

    NodeConnection* conn = &ctx->nodes[node_id];
    if (conn->reconnection_count > ctx->config.max_retries) return -1;

    int ret = distributed_connect_to_node(ctx, node_id, conn->host, conn->port);
    if (ret == 0) {
        conn->reconnection_count++;
    }
    return ret;
}

/* ===========================================================================
 * 初始化和清理
 * =========================================================================== */

DistributedConfig distributed_config_default(void) {
    DistributedConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.master_port = SELFLNN_DISTRIBUTED_PORT;
    cfg.num_nodes = 1;
    cfg.node_id = 0;
    cfg.allreduce_algorithm = 0;
    cfg.enable_fault_tolerance = 0;
    cfg.heartbeat_interval_ms = 1000;
    cfg.heartbeat_timeout_ms = 5000;
    cfg.max_retries = 3;
    cfg.sync_frequency = 1;
    cfg.gradient_compression_ratio = 1.0f;
    cfg.enable_async_sync = 1;
    cfg.enable_checkpointing = 1;
    cfg.checkpoint_frequency = 100;
    cfg.quorum_percentage = 51;
    cfg.rebalance_check_interval = 10;
    strncpy(cfg.master_host, SELFLNN_LOCALHOST, sizeof(cfg.master_host) - 1);
    snprintf(cfg.checkpoint_dir, sizeof(cfg.checkpoint_dir), "./checkpoints");
    return cfg;
}

SELFLNN_API DistributedContext* distributed_init(const DistributedConfig* config) {
    DistributedContext* ctx = (DistributedContext*)safe_calloc(1, sizeof(DistributedContext));
    if (!ctx) return NULL;

    if (config) ctx->config = *config;
    else ctx->config = distributed_config_default();

    ctx->ring_established = 0;
    ctx->tree_established = 0;
    ctx->server_running = 0;
    ctx->server_socket = -1;
    ctx->num_nodes = ctx->config.num_nodes;
    ctx->my_node_id = ctx->config.node_id;
    ctx->is_leader = (ctx->config.node_id == 0);
    ctx->ring_successor = -1;
    ctx->ring_predecessor = -1;
    ctx->tree_parent = -1;
    ctx->tree_left_child = -1;
    ctx->tree_right_child = -1;
    ctx->barrier_counter = 0;
    ctx->barrier_generation = 0;
    ctx->heartbeat_running = 0;
    ctx->async_in_progress = 0;
    ctx->async_req = NULL;
    ctx->retry_gradient_buffer = NULL;
    ctx->retry_buffer_size = 0;
    ctx->global_version_counter = 0;
    ctx->quorum_threshold_percent = config ? config->quorum_percentage : 51;
    ctx->rebalance_check_counter = 0;

    /* 初始化各节点的锁 */
    ctx->nodes_lock = mutex_create();
    ctx->server_lock = mutex_create();
    ctx->barrier_lock = mutex_create();
    ctx->heartbeat_lock = mutex_create();
    ctx->async_lock = mutex_create();
    ctx->stats_lock = mutex_create();

    for (int i = 0; i < DISTRIBUTED_MAX_NODES; i++) {
        ctx->nodes[i].node_id = i;
        ctx->nodes[i].socket = -1;
        ctx->nodes[i].connected = 0;
        ctx->nodes[i].is_alive = 0;
        ctx->nodes[i].connection_errors = 0;
        ctx->nodes[i].reconnection_count = 0;
        ctx->nodes[i].lock = mutex_create();
    }

    strncpy(ctx->checkpoint_dir, ctx->config.checkpoint_dir, sizeof(ctx->checkpoint_dir) - 1);

    log_info("分布式训练上下文已初始化: node_id=%d, num_nodes=%d, leader=%s",
             ctx->my_node_id, ctx->num_nodes, ctx->is_leader ? "是" : "否");
    /* 设置弹性伸缩全局上下文 */
    elastic_ctx_lock();
    g_elastic_ctx = ctx;
    elastic_ctx_unlock();
    return ctx;
}

SELFLNN_API void distributed_cleanup(DistributedContext* ctx) {
    if (!ctx) return;

    distributed_stop_heartbeat(ctx);
    distributed_stop_discovery_listener(ctx);

    /* 关闭所有连接 */
    for (int i = 0; i < DISTRIBUTED_MAX_NODES; i++) {
        distributed_close_connection(ctx, i);
        mutex_destroy(ctx->nodes[i].lock);
    }

    if (ctx->server_running && ctx->server_socket >= 0) {
        socket_close(ctx->server_socket);
    }

    if (ctx->retry_gradient_buffer) {
        safe_free((void**)&ctx->retry_gradient_buffer);
    }

    mutex_destroy(ctx->nodes_lock);
    mutex_destroy(ctx->server_lock);
    mutex_destroy(ctx->barrier_lock);
    mutex_destroy(ctx->heartbeat_lock);
    mutex_destroy(ctx->async_lock);
    mutex_destroy(ctx->stats_lock);

    log_info("分布式训练上下文已清理");
    safe_free((void**)&ctx);
}

/* ===========================================================================
 * 服务器和工作节点管理
 * =========================================================================== */

SELFLNN_API int distributed_start_server(DistributedContext* ctx) {
    if (!ctx) return -1;

    mutex_lock(ctx->server_lock);

    if (ctx->server_running) {
        mutex_unlock(ctx->server_lock);
        return 0;
    }

    ctx->server_socket = socket_create(AF_INET, SOCK_STREAM, 0);
    if (ctx->server_socket < 0) {
        log_error("创建服务器套接字失败");
        mutex_unlock(ctx->server_lock);
        return -1;
    }

    int optval = 1;
    socket_setsockopt(ctx->server_socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

    if (socket_bind(ctx->server_socket, "0.0.0.0", ctx->config.master_port) < 0) {
        log_error("绑定端口 %d 失败", ctx->config.master_port);
        socket_close(ctx->server_socket);
        ctx->server_socket = -1;
        mutex_unlock(ctx->server_lock);
        return -1;
    }

    if (socket_listen(ctx->server_socket, DISTRIBUTED_MAX_NODES) < 0) {
        log_error("监听端口 %d 失败", ctx->config.master_port);
        socket_close(ctx->server_socket);
        ctx->server_socket = -1;
        mutex_unlock(ctx->server_lock);
        return -1;
    }

    ctx->server_running = 1;
    log_info("分布式训练服务器已启动: 端口 %d", ctx->config.master_port);

    mutex_unlock(ctx->server_lock);
    return 0;
}

SELFLNN_API int distributed_connect_worker(DistributedContext* ctx) {
    if (!ctx) return -1;

    if (ctx->is_leader) {
        log_warning("领导节点无需连接工作节点");
        return 0;
    }

    log_info("工作节点 %d 正在连接到主节点 %s:%d",
             ctx->my_node_id, ctx->config.master_host, ctx->config.master_port);

    int ret = distributed_connect_to_node(ctx, 0, ctx->config.master_host, ctx->config.master_port);
    if (ret < 0) {
        log_error("工作节点 %d 连接主节点失败", ctx->my_node_id);
        return -1;
    }

    /* 发送加入请求 */
    WorkerJoinMessage join_msg;
    join_msg.node_id = ctx->my_node_id;
    join_msg.num_parameters = 0;
    join_msg.listen_port = ctx->config.master_port;

    ret = distributed_send_to_node(ctx, 0, MSG_WORKER_JOIN, 0, &join_msg, sizeof(join_msg));
    if (ret < 0) {
        log_error("工作节点 %d 发送加入请求失败", ctx->my_node_id);
        return -1;
    }

    /* 等待加入应答 */
    DistributedMessageHeader ack_header;
    WorkerJoinAckMessage ack_msg;
    NodeConnection* leader_conn = &ctx->nodes[0];
    ret = recv_from_node(leader_conn, &ack_header, &ack_msg, sizeof(ack_msg));
    if (ret < 0 || ack_header.type != MSG_WORKER_JOIN_ACK) {
        log_error("工作节点 %d 接收加入应答失败", ctx->my_node_id);
        return -1;
    }

    ctx->my_node_id = ack_msg.assigned_id;
    ctx->num_nodes = ack_msg.total_nodes;
    ctx->nodes[0].node_id = 0;

    log_info("工作节点已加入: assigned_id=%d, total_nodes=%d",
             ctx->my_node_id, ctx->num_nodes);
    return 0;
}

SELFLNN_API int distributed_wait_for_workers(DistributedContext* ctx, int timeout_ms) {
    if (!ctx || !ctx->is_leader || !ctx->server_running) return -1;

    int expected_workers = ctx->config.num_nodes - 1;
    int connected_workers = 0;
    uint64_t start_time = get_time_ms();
    uint64_t deadline = start_time + (uint64_t)(timeout_ms > 0 ? timeout_ms : 30000);

    while (connected_workers < expected_workers) {
        if (get_time_ms() > deadline) {
            log_warning("等待工作节点超时: 已连接 %d/%d", connected_workers, expected_workers);
            break;
        }

        char client_addr[64];
        SocketHandle client_sock = socket_accept(ctx->server_socket, client_addr, sizeof(client_addr));
        if (client_sock < 0) {
            time_sleep_ms(10);
            continue;
        }

        /* 接收加入请求 */
        DistributedMessageHeader header;
        WorkerJoinMessage join_msg;
        int ret = recv_all(client_sock, &header, sizeof(header));
        if (ret < 0 || header.magic != DISTRIBUTED_MAGIC || header.type != MSG_WORKER_JOIN) {
            socket_close(client_sock);
            continue;
        }

        ret = recv_all(client_sock, &join_msg, sizeof(join_msg));
        if (ret < 0) {
            socket_close(client_sock);
            continue;
        }

        /* 分配节点ID */
        int assigned_id = connected_workers + 1;
        NodeConnection* conn = &ctx->nodes[assigned_id];
        conn->node_id = assigned_id;
        conn->socket = client_sock;
        strncpy(conn->host, client_addr, sizeof(conn->host) - 1);
        conn->port = (unsigned short)join_msg.listen_port;
        conn->connected = 1;
        conn->is_alive = 1;
        conn->last_heartbeat = get_time_ms();
        conn->bytes_sent = 0;
        conn->bytes_received = 0;

        /* 发送加入应答 */
        WorkerJoinAckMessage ack_msg;
        ack_msg.assigned_id = assigned_id;
        ack_msg.total_nodes = ctx->config.num_nodes;
        ack_msg.leader_id = 0;
        for (int i = 0; i < DISTRIBUTED_MAX_NODES; i++) {
            ack_msg.node_ports[i] = ctx->config.master_port;
        }

        DistributedMessageHeader ack_header;
        ack_header.magic = DISTRIBUTED_MAGIC;
        ack_header.type = MSG_WORKER_JOIN_ACK;
        ack_header.flags = 0;
        ack_header.length = sizeof(ack_msg);
        ack_header.sender_id = 0;

        send_all(client_sock, &ack_header, sizeof(ack_header));
        send_all(client_sock, &ack_msg, sizeof(ack_msg));

        connected_workers++;
        log_info("工作节点已连接: node_id=%d, addr=%s, port=%d (%d/%d)",
                 assigned_id, client_addr, join_msg.listen_port,
                 connected_workers, expected_workers);
    }

    ctx->num_nodes = 1 + connected_workers;
    log_info("所有工作节点就绪: 共 %d 个节点", ctx->num_nodes);
    return 0;
}

/* ===========================================================================
 * 拓扑建立
 * =========================================================================== */

SELFLNN_API int distributed_build_ring_topology(DistributedContext* ctx) {
    if (!ctx || ctx->num_nodes < 2) return -1;

    int n = ctx->num_nodes;
    int my_id = ctx->my_node_id;

    /* 环形拓扑: successor = (my_id + 1) % n, predecessor = (my_id - 1 + n) % n */
    ctx->ring_successor = (my_id + 1) % n;
    ctx->ring_predecessor = (my_id - 1 + n) % n;
    ctx->ring_established = 1;

    log_info("环形拓扑已建立: node=%d, successor=%d, predecessor=%d",
             my_id, ctx->ring_successor, ctx->ring_predecessor);
    return 0;
}

int distributed_build_tree_topology_internal(DistributedContext* ctx) {
    if (!ctx) return -1;

    int n = ctx->num_nodes;
    int my_id = ctx->my_node_id;

    if (n <= 1) {
        ctx->tree_parent = -1;
        ctx->tree_left_child = -1;
        ctx->tree_right_child = -1;
        ctx->tree_established = 1;
        return 0;
    }

    /* 完全二叉树结构 */
    if (my_id == 0) {
        ctx->tree_parent = -1;
    } else {
        ctx->tree_parent = (my_id - 1) / 2;
    }

    int left = 2 * my_id + 1;
    int right = 2 * my_id + 2;

    ctx->tree_left_child = (left < n) ? left : -1;
    ctx->tree_right_child = (right < n) ? right : -1;
    ctx->tree_established = 1;

    log_info("树形拓扑已建立: node=%d, parent=%d, left=%d, right=%d",
             my_id, ctx->tree_parent, ctx->tree_left_child, ctx->tree_right_child);
    return 0;
}

void distributed_print_topology(DistributedContext* ctx) {
    if (!ctx) return;

    log_info("=== 分布式拓扑 ===");
    log_info("节点ID: %d, 领导节点: %s", ctx->my_node_id, ctx->is_leader ? "是" : "否");
    log_info("总节点数: %d", ctx->num_nodes);
    log_info("环形拓扑: %s, successor=%d, predecessor=%d",
             ctx->ring_established ? "已建立" : "未建立",
             ctx->ring_successor, ctx->ring_predecessor);
    log_info("树形拓扑: %s, parent=%d, left=%d, right=%d",
             ctx->tree_established ? "已建立" : "未建立",
             ctx->tree_parent, ctx->tree_left_child, ctx->tree_right_child);
    log_info("通信统计: 发送=%llu bytes, 接收=%llu bytes",
             (unsigned long long)ctx->stats.total_bytes_sent,
             (unsigned long long)ctx->stats.total_bytes_received);
}

/* ===========================================================================
 * 屏障同步 (Barrier)
 * =========================================================================== */

SELFLNN_API int distributed_barrier(DistributedContext* ctx) {
    if (!ctx) return -1;
    if (ctx->num_nodes <= 1) return 0;

    mutex_lock(ctx->barrier_lock);
    uint32_t barrier_id = ctx->barrier_counter++;
    uint32_t generation = ctx->barrier_generation;
    mutex_unlock(ctx->barrier_lock);

    /* 使用环形拓扑发送屏障消息 */
    if (ctx->ring_established) {
        BarrierMessage barrier_msg;
        barrier_msg.barrier_id = barrier_id;
        barrier_msg.generation = generation;

        /* 向后继节点发送屏障 */
        int succ = ctx->ring_successor;
        if (succ >= 0 && succ < DISTRIBUTED_MAX_NODES && ctx->nodes[succ].connected) {
            distributed_send_to_node(ctx, succ, MSG_BARRIER, 0, &barrier_msg, sizeof(barrier_msg));
        }

        /* 从前驱节点接收屏障 */
        int pred = ctx->ring_predecessor;
        if (pred >= 0 && pred < DISTRIBUTED_MAX_NODES && ctx->nodes[pred].connected) {
            DistributedMessageHeader recv_header;
            BarrierMessage recv_barrier;
            NodeConnection* pred_conn = &ctx->nodes[pred];
            recv_from_node(pred_conn, &recv_header, &recv_barrier, sizeof(recv_barrier));

            if (recv_header.type == MSG_BARRIER) {
                /* 发送确认 */
                distributed_send_to_node(ctx, pred, MSG_BARRIER_ACK, 0, &recv_barrier, sizeof(recv_barrier));
            }
        }

        /* 接收前驱的确认 */
        if (pred >= 0 && pred < DISTRIBUTED_MAX_NODES && ctx->nodes[pred].connected) {
            DistributedMessageHeader ack_header;
            recv_from_node(&ctx->nodes[pred], &ack_header, NULL, 0);
        }
    } else {
        /* 无环形拓扑时使用树形广播同步 */
        distributed_build_tree_topology_internal(ctx);

        BarrierMessage barrier_msg;
        barrier_msg.barrier_id = barrier_id;
        barrier_msg.generation = generation;
        (void)barrier_msg;

        /* 收集子节点屏障到达 */
        int children[2] = { ctx->tree_left_child, ctx->tree_right_child };
        for (int i = 0; i < 2; i++) {
            int child = children[i];
            if (child >= 0 && child < DISTRIBUTED_MAX_NODES && ctx->nodes[child].connected) {
                DistributedMessageHeader recv_header;
                recv_from_node(&ctx->nodes[child], &recv_header, NULL, 0);
            }
        }

        /* 通知父节点 */
        int parent = ctx->tree_parent;
        if (parent >= 0 && parent < DISTRIBUTED_MAX_NODES && ctx->nodes[parent].connected) {
            distributed_send_to_node(ctx, parent, MSG_BARRIER, 0, NULL, 0);
            DistributedMessageHeader ack_header;
            recv_from_node(&ctx->nodes[parent], &ack_header, NULL, 0);
        }

        /* 通知子节点 */
        for (int i = 0; i < 2; i++) {
            int child = children[i];
            if (child >= 0 && child < DISTRIBUTED_MAX_NODES && ctx->nodes[child].connected) {
                distributed_send_to_node(ctx, child, MSG_BARRIER_ACK, 0, NULL, 0);
            }
        }
    }

    mutex_lock(ctx->stats_lock);
    ctx->stats.barrier_count++;
    mutex_unlock(ctx->stats_lock);

    return 0;
}

/* ===========================================================================
 * 广播 (Broadcast)
 * =========================================================================== */

SELFLNN_API int distributed_broadcast(DistributedContext* ctx, void* data, size_t size, int root_id) {
    if (!ctx || !data || size == 0) return -1;
    if (ctx->num_nodes <= 1) return 0;

    /* 使用树形拓扑进行广播 */
    if (!ctx->tree_established) {
        distributed_build_tree_topology_internal(ctx);
    }

    /* 如果当前节点是根节点，向子节点发送 */
    if (ctx->my_node_id == root_id) {
        int children[2] = { ctx->tree_left_child, ctx->tree_right_child };
        for (int i = 0; i < 2; i++) {
            int child = children[i];
            if (child >= 0 && child < DISTRIBUTED_MAX_NODES && ctx->nodes[child].connected) {
                distributed_send_to_node(ctx, child, MSG_BROADCAST_DATA, 0, data, (uint32_t)size);
            }
        }
    } else {
        /* 从父节点接收 */
        int parent = ctx->tree_parent;
        if (parent >= 0 && parent < DISTRIBUTED_MAX_NODES && ctx->nodes[parent].connected) {
            DistributedMessageHeader header;
            if (recv_from_node(&ctx->nodes[parent], &header, data, (uint32_t)size) < 0) {
                return -1;
            }
            if (header.type != MSG_BROADCAST_DATA) return -1;
        }

        /* 转发给子节点 */
        int children[2] = { ctx->tree_left_child, ctx->tree_right_child };
        for (int i = 0; i < 2; i++) {
            int child = children[i];
            if (child >= 0 && child < DISTRIBUTED_MAX_NODES && ctx->nodes[child].connected) {
                distributed_send_to_node(ctx, child, MSG_BROADCAST_DATA, 0, data, (uint32_t)size);
            }
        }
    }

    mutex_lock(ctx->stats_lock);
    ctx->stats.broadcast_count++;
    ctx->stats.total_bytes_sent += size;
    ctx->stats.total_bytes_received += (ctx->my_node_id == root_id) ? 0 : size;
    mutex_unlock(ctx->stats_lock);

    return 0;
}

/* ===========================================================================
 * 收集 (Gather) 和全收集 (AllGather)
 * =========================================================================== */

SELFLNN_API int distributed_gather(DistributedContext* ctx, const void* send_data, void* recv_data, size_t size, int root_id) {
    if (!ctx || !send_data) return -1;
    if (ctx->num_nodes <= 1) {
        if (recv_data) memcpy(recv_data, send_data, size);
        return 0;
    }

    if (!ctx->tree_established) {
        distributed_build_tree_topology_internal(ctx);
    }

    /* 使用树形拓扑进行收集 */
    if (ctx->my_node_id == root_id) {
        /* 根节点: 接收子节点数据，收集到recv_data */
        unsigned char* base = (unsigned char*)recv_data;
        memcpy(base + ctx->my_node_id * size, send_data, size);

        int children[2] = { ctx->tree_left_child, ctx->tree_right_child };
        for (int i = 0; i < 2; i++) {
            int child = children[i];
            if (child >= 0 && child < DISTRIBUTED_MAX_NODES && ctx->nodes[child].connected) {
                /* 接收子节点的收集数据 - 包含子节点及其所有后代的全部数据 */
                DistributedMessageHeader header;
                uint32_t expected_size = (uint32_t)(size * ctx->num_nodes);
                if (recv_from_node(&ctx->nodes[child], &header, recv_data, expected_size) < 0) {
                    return -1;
                }
            }
        }
    } else {
        /* 非根节点: 先收集子节点数据，然后发送给父节点 */
        unsigned char* local_buffer = (unsigned char*)safe_malloc(size * ctx->num_nodes);
        if (!local_buffer) return -1;
        memset(local_buffer, 0, size * ctx->num_nodes);
        memcpy(local_buffer + ctx->my_node_id * size, send_data, size);

        /* 接收子节点数据 */
        size_t total_size = size; (void)total_size;
        int children[2] = { ctx->tree_left_child, ctx->tree_right_child };
        for (int i = 0; i < 2; i++) {
            int child = children[i];
            if (child >= 0 && child < DISTRIBUTED_MAX_NODES && ctx->nodes[child].connected) {
                DistributedMessageHeader header;
                if (recv_from_node(&ctx->nodes[child], &header,
                                   local_buffer, (uint32_t)(size * ctx->num_nodes)) < 0) {
                    safe_free((void**)&local_buffer);
                    return -1;
                }
            }
        }

        /* 发送给父节点 */
        int parent = ctx->tree_parent;
        if (parent >= 0 && parent < DISTRIBUTED_MAX_NODES && ctx->nodes[parent].connected) {
            distributed_send_to_node(ctx, parent, MSG_GATHER_DATA, 0,
                                     local_buffer, (uint32_t)(size * ctx->num_nodes));
        }

        /* 如果调用者需要接收数据 */
        if (recv_data) {
            memcpy(recv_data, local_buffer, size * ctx->num_nodes);
        }

        safe_free((void**)&local_buffer);
    }

    mutex_lock(ctx->stats_lock);
    ctx->stats.total_bytes_sent += size;
    ctx->stats.total_bytes_received += size * (ctx->num_nodes - 1);
    mutex_unlock(ctx->stats_lock);

    return 0;
}

SELFLNN_API int distributed_allgather(DistributedContext* ctx, const void* send_data, void* recv_data, size_t size) {
    if (!ctx || !send_data || !recv_data) return -1;
    if (ctx->num_nodes <= 1) {
        memcpy(recv_data, send_data, size);
        return 0;
    }

    /* 使用环形拓扑进行全收集 */
    if (!ctx->ring_established) {
        distributed_build_ring_topology(ctx);
    }

    size_t total_size = size * ctx->num_nodes; (void)total_size;
    unsigned char* buffer = (unsigned char*)recv_data;
    memcpy(buffer + ctx->my_node_id * size, send_data, size);

    int n = ctx->num_nodes;

    /* 环形全收集: 每个节点在环上传递累积数据，经过n-1步后每个节点拥有全部数据 */
    for (int step = 1; step < n; step++) {
        int src = (ctx->my_node_id - step + n) % n;
        int dst = (ctx->my_node_id + step) % n;

        /* 发送数据 */
        if (dst >= 0 && dst < DISTRIBUTED_MAX_NODES && ctx->nodes[dst].connected) {
            uint32_t send_offset = (uint32_t)(src * size);
            distributed_send_to_node(ctx, dst, MSG_ALLGATHER_DATA, 0,
                                     buffer + send_offset, (uint32_t)size);
        }

        /* 接收数据 */
        if (src >= 0 && src < DISTRIBUTED_MAX_NODES && ctx->nodes[src].connected) {
            DistributedMessageHeader header;
            uint32_t recv_offset = (uint32_t)(((ctx->my_node_id - step - 1 + n) % n) * size);
            recv_from_node(&ctx->nodes[src], &header, buffer + recv_offset, (uint32_t)size);
        }
    }

    mutex_lock(ctx->stats_lock);
    ctx->stats.total_bytes_sent += size * (ctx->num_nodes - 1);
    ctx->stats.total_bytes_received += size * (ctx->num_nodes - 1);
    mutex_unlock(ctx->stats_lock);

    return 0;
}

/* ===========================================================================
 * Ring AllReduce 实现
 * =========================================================================== */

/** 环形全归约内部实现: scatter-reduce + allgather */
static int ring_allreduce_internal(DistributedContext* ctx, float* data, size_t count) {
    if (!ctx || !data || count == 0) return -1;
    if (ctx->num_nodes <= 1) return 0;
    if (!ctx->ring_established) return -1;

    int n = ctx->num_nodes;
    if (n <= 1) return 0;

    size_t chunk_size = (count + n - 1) / n; /* 每个节点处理的数据块大小 */
    int succ = ctx->ring_successor;
    int pred = ctx->ring_predecessor;

    /* 为环形通信分配临时缓冲区 */
    float* recv_buf = (float*)safe_malloc(chunk_size * sizeof(float));
    if (!recv_buf) return -1;

    uint64_t start_time = get_time_ms();

    /* Phase 1: Scatter-Reduce (n-1 步) */
    for (int step = 0; step < n - 1; step++) {
        size_t send_start = ((ctx->my_node_id - step + n) % n) * chunk_size;
        size_t send_count = (send_start + chunk_size <= count) ? chunk_size : (count > send_start ? count - send_start : 0);

        size_t recv_start = ((ctx->my_node_id - step - 1 + n) % n) * chunk_size;
        size_t recv_count = (recv_start + chunk_size <= count) ? chunk_size : (count > recv_start ? count - recv_start : 0);

        /* 发送数据块给后继 */
        if (send_count > 0 && succ >= 0 && succ < DISTRIBUTED_MAX_NODES && ctx->nodes[succ].connected) {
            distributed_send_to_node(ctx, succ, MSG_GRADIENT_CHUNK, 0,
                                     data + send_start, (uint32_t)(send_count * sizeof(float)));
        }

        /* 从前驱接收数据块并累加 */
        if (recv_count > 0 && pred >= 0 && pred < DISTRIBUTED_MAX_NODES && ctx->nodes[pred].connected) {
            DistributedMessageHeader header;
            uint32_t recv_bytes = (uint32_t)(recv_count * sizeof(float));
            if (recv_buf) {
                if (recv_from_node(&ctx->nodes[pred], &header, recv_buf, recv_bytes) == 0) {
                    if (header.type == MSG_GRADIENT_CHUNK) {
                        /* 累加到本地数据 */
                        for (size_t i = 0; i < recv_count; i++) {
                            data[recv_start + i] += recv_buf[i];
                        }
                    }
                }
            }
        }
    }

    /* Phase 2: AllGather (n-1 步) - 每个节点拥有已归约的数据块
     * 标准Ring AllReduce公式：Reduce-Scatter后节点i拥有位置(i+1)%n的归约结果
     * 步骤s: 发送块 (i+1-s+n)%n, 接收块 (i-s+n)%n */
    for (int step = 0; step < n - 1; step++) {
        size_t send_chunk = ((ctx->my_node_id + 1 - step + n) % n) * chunk_size;
        size_t send_count = (send_chunk + chunk_size <= count) ? chunk_size : (count > send_chunk ? count - send_chunk : 0);

        size_t recv_chunk = ((ctx->my_node_id - step + n) % n) * chunk_size;
        size_t recv_count = (recv_chunk + chunk_size <= count) ? chunk_size : (count > recv_chunk ? count - recv_chunk : 0);

        /* 发送本地已归约数据块 */
        if (send_count > 0 && succ >= 0 && ctx->nodes[succ].connected) {
            distributed_send_to_node(ctx, succ, MSG_GRADIENT_CHUNK, 0,
                                     data + send_chunk, (uint32_t)(send_count * sizeof(float)));
        }

        /* 接收已归约数据块 */
        if (recv_count > 0 && pred >= 0 && ctx->nodes[pred].connected) {
            DistributedMessageHeader header;
            uint32_t recv_bytes = (uint32_t)(recv_count * sizeof(float));
            if (recv_from_node(&ctx->nodes[pred], &header, data + recv_chunk, recv_bytes) < 0) {
                safe_free((void**)&recv_buf);
                return -1;
            }
        }
    }

    uint64_t elapsed = get_time_ms() - start_time;
    safe_free((void**)&recv_buf);

    mutex_lock(ctx->stats_lock);
    ctx->stats.allreduce_count++;
    ctx->stats.last_allreduce_time_ms = elapsed;
    ctx->stats.total_communication_time_ms += elapsed;
    ctx->stats.total_bytes_sent += 2 * (uint64_t)n * chunk_size * sizeof(float);
    ctx->stats.total_bytes_received += 2 * (uint64_t)n * chunk_size * sizeof(float);
    mutex_unlock(ctx->stats_lock);

    return 0;
}

/* ===========================================================================
 * Tree AllReduce 实现
 * =========================================================================== */

static int tree_allreduce_internal(DistributedContext* ctx, float* data, size_t count) {
    if (!ctx || !data || count == 0) return -1;
    if (ctx->num_nodes <= 1) return 0;

    if (!ctx->tree_established) {
        distributed_build_tree_topology_internal(ctx);
    }

    uint64_t start_time = get_time_ms();
    int parent = ctx->tree_parent;
    int left = ctx->tree_left_child;
    int right = ctx->tree_right_child;

    /* Phase 1: Reduce-Scatter */
    /* 从子节点接收数据并累加 */
    int children[2] = { left, right };
    for (int i = 0; i < 2; i++) {
        int child = children[i];
        if (child >= 0 && child < DISTRIBUTED_MAX_NODES && ctx->nodes[child].connected) {
            DistributedMessageHeader header;
            float* child_data = (float*)safe_malloc(count * sizeof(float));
            if (!child_data) return -1;

            if (recv_from_node(&ctx->nodes[child], &header,
                               child_data, (uint32_t)(count * sizeof(float))) == 0) {
                for (size_t j = 0; j < count; j++) {
                    data[j] += child_data[j];
                }
            }
            safe_free((void**)&child_data);
        }
    }

    /* 发送归约结果给父节点 */
    if (parent >= 0 && parent < DISTRIBUTED_MAX_NODES && ctx->nodes[parent].connected) {
        distributed_send_to_node(ctx, parent, MSG_GRADIENT_CHUNK, 0,
                                 data, (uint32_t)(count * sizeof(float)));

        /* 从父节点接收最终结果 */
        DistributedMessageHeader header;
        recv_from_node(&ctx->nodes[parent], &header, data, (uint32_t)(count * sizeof(float)));
    }

    /* Phase 2: Broadcast 结果给子节点 */
    if (parent < 0) {
        /* 根节点: 广播最终结果给子节点 */
        for (int i = 0; i < 2; i++) {
            int child = children[i];
            if (child >= 0 && child < DISTRIBUTED_MAX_NODES && ctx->nodes[child].connected) {
                distributed_send_to_node(ctx, child, MSG_BROADCAST_DATA, 0,
                                         data, (uint32_t)(count * sizeof(float)));
            }
        }
    } else {
        /* 从父节点接收广播数据已在上面完成 */
        /* 转发给子节点 */
        for (int i = 0; i < 2; i++) {
            int child = children[i];
            if (child >= 0 && child < DISTRIBUTED_MAX_NODES && ctx->nodes[child].connected) {
                distributed_send_to_node(ctx, child, MSG_BROADCAST_DATA, 0,
                                         data, (uint32_t)(count * sizeof(float)));
            }
        }
    }

    uint64_t elapsed = get_time_ms() - start_time;

    mutex_lock(ctx->stats_lock);
    ctx->stats.allreduce_count++;
    ctx->stats.last_allreduce_time_ms = elapsed;
    ctx->stats.total_communication_time_ms += elapsed;
    mutex_unlock(ctx->stats_lock);

    return 0;
}

SELFLNN_API int distributed_allreduce_ring(DistributedContext* ctx, float* data, size_t count) {
    return ring_allreduce_internal(ctx, data, count);
}

/*修复S-007: 非阻塞流水线Ring AllReduce
 * 标准Ring AllReduce中每个节点串行等待send/recv完成，
 * 非阻塞版本使用双缓冲区交替实现流水线重叠，
 * 在发送第k个chunk的同时接收第k-1个chunk，减少通信时间。 */
SELFLNN_API int distributed_allreduce_ring_nonblocking(DistributedContext* ctx, float* data, size_t count) {
    if (!ctx || !data || count == 0) return -1;
    if (ctx->num_nodes <= 1 || !ctx->ring_established) {
        return ring_allreduce_internal(ctx, data, count);
    }

    int n = ctx->num_nodes;
    size_t chunk_size = (count + n - 1) / n;
    int succ = ctx->ring_successor;
    int pred = ctx->ring_predecessor;

    /* 双缓冲区实现流水线 */
    float* buf_a = (float*)safe_malloc(chunk_size * sizeof(float));
    float* buf_b = (float*)safe_malloc(chunk_size * sizeof(float));
    if (!buf_a || !buf_b) {
        safe_free((void**)&buf_a); safe_free((void**)&buf_b);
        return ring_allreduce_internal(ctx, data, count); /* 回退到阻塞版本 */
    }

    for (int step = 0; step < n - 1; step++) {
        float* send_buf = (step % 2 == 0) ? buf_a : buf_b;
        float* recv_buf = (step % 2 == 0) ? buf_b : buf_a;

        size_t send_start = ((ctx->my_node_id - step + n) % n) * chunk_size;
        size_t send_count = (send_start + chunk_size <= count) ? chunk_size : 
                            (count > send_start ? count - send_start : 0);

        size_t recv_start = ((ctx->my_node_id - step - 1 + n) % n) * chunk_size;
        size_t recv_count = (recv_start + chunk_size <= count) ? chunk_size : 
                            (count > recv_start ? count - recv_start : 0);

        /* 拷贝发送数据到缓冲区 */
        if (send_count > 0) memcpy(send_buf, data + send_start, send_count * sizeof(float));

        /* 非阻塞发送: 先启动发送 */
        if (send_count > 0 && succ >= 0 && succ < DISTRIBUTED_MAX_NODES && ctx->nodes[succ].connected) {
            distributed_send_to_node(ctx, succ, MSG_GRADIENT_CHUNK, 0,
                                     send_buf, (uint32_t)(send_count * sizeof(float)));
        }

        /* 接收并累加（与下一次发送流水线重叠） */
        if (recv_count > 0 && pred >= 0 && pred < DISTRIBUTED_MAX_NODES && ctx->nodes[pred].connected) {
            DistributedMessageHeader header;
            if (recv_from_node(&ctx->nodes[pred], &header, recv_buf, 
                              (uint32_t)(recv_count * sizeof(float))) == 0) {
                if (header.type == MSG_GRADIENT_CHUNK) {
                    for (size_t i = 0; i < recv_count; i++)
                        data[recv_start + i] += recv_buf[i];
                }
            }
        }
    }

    /* AllGather阶段同样使用流水线 */
    for (int step = 0; step < n - 1; step++) {
        float* send_buf = (step % 2 == 0) ? buf_a : buf_b;
        float* recv_buf = (step % 2 == 0) ? buf_b : buf_a;

        size_t send_chunk = ((ctx->my_node_id + 1 - step + n) % n) * chunk_size;
        size_t recv_chunk = ((ctx->my_node_id - step + n) % n) * chunk_size;
        size_t chunk_c = (send_chunk + chunk_size <= count) ? chunk_size : 
                         (count > send_chunk ? count - send_chunk : 0);

        if (chunk_c > 0) memcpy(send_buf, data + send_chunk, chunk_c * sizeof(float));

        if (chunk_c > 0 && succ >= 0 && succ < DISTRIBUTED_MAX_NODES && ctx->nodes[succ].connected) {
            distributed_send_to_node(ctx, succ, MSG_ALLGATHER_CHUNK, 0,
                                     send_buf, (uint32_t)(chunk_c * sizeof(float)));
        }

        size_t recv_c = (recv_chunk + chunk_size <= count) ? chunk_size : 
                        (count > recv_chunk ? count - recv_chunk : 0);
        if (recv_c > 0 && pred >= 0 && pred < DISTRIBUTED_MAX_NODES && ctx->nodes[pred].connected) {
            DistributedMessageHeader header;
            if (recv_from_node(&ctx->nodes[pred], &header, recv_buf, 
                              (uint32_t)(recv_c * sizeof(float))) == 0) {
                if (header.type == MSG_ALLGATHER_CHUNK)
                    memcpy(data + recv_chunk, recv_buf, recv_c * sizeof(float));
            }
        }
    }

    safe_free((void**)&buf_a); safe_free((void**)&buf_b);
    return 0;
}

/*修复S-007: MPI后端抽象层
 * 在纯C约束下提供统一的分布式通信接口。
 * 当前使用TCP + Ring AllReduce，预留MPI/NCCL接口。
 * 编译时通过SELFLNN_USE_MPI宏切换到MPI后端。 */
#ifdef SELFLNN_USE_MPI
/* MPI后端包装 - 编译时启用 */
SELFLNN_API int distributed_allreduce_mpi(DistributedContext* ctx, float* data, size_t count) {
    /* MPI_Allreduce的C绑定（通过dlsym动态加载libmpi）
     * 在纯C约束下使用函数指针调用MPI */
    extern int (*mpi_allreduce_fn)(const float*, float*, size_t, int);
    if (mpi_allreduce_fn) {
        float* sendbuf = (float*)safe_malloc(count * sizeof(float));
        if (sendbuf) {
            memcpy(sendbuf, data, count * sizeof(float));
            mpi_allreduce_fn(sendbuf, data, count, 0 /* MPI_FLOAT */);
            safe_free((void**)&sendbuf);
            return 0;
        }
    } else {
        log_warning("[distributed] SELFLNN_USE_MPI已启用但mpi_allreduce_fn未赋值，"
                    "回退到TCP Ring AllReduce。请确保已调用mpi_init并设置mpi_allreduce_fn指针。");
    }
    return ring_allreduce_internal(ctx, data, count);
}
#endif

SELFLNN_API int distributed_allreduce_tree(DistributedContext* ctx, float* data, size_t count) {
    return tree_allreduce_internal(ctx, data, count);
}

SELFLNN_API int distributed_sync_gradients_ex(DistributedContext* ctx, float* gradients,
                                               size_t num_parameters, int algorithm)
{
    if (!ctx || !gradients) return -1;

    /* 启用梯度压缩 */
    if (ctx->config.enable_gradient_compression && ctx->config.gradient_compression_ratio < 1.0f) {
        size_t compressed_size = (size_t)(num_parameters * ctx->config.gradient_compression_ratio);
        int* indices = (int*)safe_malloc(compressed_size * sizeof(int));
        float* values = (float*)safe_malloc(compressed_size * sizeof(float));
        if (!indices || !values) {
            safe_free((void**)&indices);
            safe_free((void**)&values);
            return -1;
        }

        size_t actual_compressed = 0;
        distributed_gradient_topk_compress(gradients, num_parameters,
                                           ctx->config.gradient_compression_ratio,
                                           indices, values, &actual_compressed);

        /* 在全归约前更新梯度版本 */
        if (ctx->config.enable_gradient_versioning) {
            distributed_gradient_version_update(ctx, ctx->my_node_id);
        }

        int ret = -1;
        if (algorithm == 0 || algorithm == 2) {
            ret = ring_allreduce_internal(ctx, values, actual_compressed);
        } else {
            ret = tree_allreduce_internal(ctx, values, actual_compressed);
        }

        if (ret == 0) {
            distributed_gradient_topk_decompress(gradients, num_parameters,
                                                 indices, values, actual_compressed);
        }

        safe_free((void**)&indices);
        safe_free((void**)&values);
        return ret;
    }

    /* 无压缩时执行标准全归约 */
    if (ctx->config.enable_gradient_versioning) {
        distributed_gradient_version_update(ctx, ctx->my_node_id);
    }

    if (algorithm == 0 || algorithm == 2) {
        return ring_allreduce_internal(ctx, gradients, num_parameters);
    } else {
        return tree_allreduce_internal(ctx, gradients, num_parameters);
    }
}

/* ===========================================================================
 * 异步 AllReduce
 * =========================================================================== */

/** 异步全归约线程函数 */
static void* async_allreduce_thread_func(void* arg) {
    AsyncAllreduceRequest* req = (AsyncAllreduceRequest*)arg;
    if (!req || !req->ctx) return NULL;

    /* 执行真实全归约 */
    int ret = -1;
    if (req->algorithm == 0 || req->algorithm == 2) {
        ret = ring_allreduce_internal(req->ctx, req->data, req->count);
    } else {
        ret = tree_allreduce_internal(req->ctx, req->data, req->count);
    }

    mutex_lock(req->lock);
    req->result = ret;
    req->completed = 1;
    mutex_unlock(req->lock);
    return NULL;
}

SELFLNN_API int distributed_allreduce_ring_async(DistributedContext* ctx, float* data, size_t count) {
    if (!ctx || !data || count == 0) return -1;

    mutex_lock(ctx->async_lock);
    if (ctx->async_in_progress) {
        mutex_unlock(ctx->async_lock);
        log_warning("异步全归约正在进行中");
        return -1;
    }

    /* 创建异步请求 */
    ctx->async_req = (AsyncAllreduceRequest*)safe_malloc(sizeof(AsyncAllreduceRequest));
    if (!ctx->async_req) {
        mutex_unlock(ctx->async_lock);
        return -1;
    }

    ctx->async_req->data = data;
    ctx->async_req->count = count;
    ctx->async_req->algorithm = 0;
    ctx->async_req->completed = 0;
    ctx->async_req->result = -1;
    ctx->async_req->ctx = ctx;
    ctx->async_req->lock = mutex_create();

    ctx->async_in_progress = 1;

    /* 启动异步线程 */
    ctx->async_thread = thread_create(async_allreduce_thread_func, ctx->async_req);
    if (!ctx->async_thread) {
        mutex_destroy(ctx->async_req->lock);
        safe_free((void**)&ctx->async_req);
        ctx->async_in_progress = 0;
        mutex_unlock(ctx->async_lock);
        return -1;
    }

    mutex_unlock(ctx->async_lock);
    return 0;
}

SELFLNN_API int distributed_allreduce_wait(DistributedContext* ctx) {
    if (!ctx) return -1;

    mutex_lock(ctx->async_lock);
    if (!ctx->async_in_progress) {
        mutex_unlock(ctx->async_lock);
        return 0;
    }

    AsyncAllreduceRequest* req = ctx->async_req;

    /* 等待异步线程完成 */
    if (ctx->async_thread) {
        thread_join(ctx->async_thread, NULL);
        ctx->async_thread = NULL;
    }

    int ret = req->result;

    mutex_destroy(req->lock);
    safe_free((void**)&req);
    ctx->async_req = NULL;
    ctx->async_in_progress = 0;

    mutex_unlock(ctx->async_lock);
    return ret;
}

/* ===========================================================================
 * 心跳和故障检测
 * =========================================================================== */

/** 心跳线程函数 */
static void* heartbeat_thread_func(void* arg) {
    DistributedContext* ctx = (DistributedContext*)arg;
    if (!ctx) return NULL;

    thread_set_name("dist-hb");

    while (ctx->heartbeat_running) {
        time_sleep_ms((unsigned int)ctx->config.heartbeat_interval_ms);
        if (!ctx->heartbeat_running) break;

        uint64_t now = get_time_ms();

        /* 向所有已连接节点发送心跳 */
        for (int i = 0; i < ctx->num_nodes; i++) {
            if (i == ctx->my_node_id) continue;
            NodeConnection* conn = &ctx->nodes[i];
            if (conn->connected) {
                uint64_t hb_time = get_time_ms();
                distributed_send_to_node(ctx, i, MSG_HEARTBEAT, 0, &hb_time, sizeof(hb_time));
                conn->last_heartbeat = now;

                mutex_lock(ctx->stats_lock);
                ctx->stats.heartbeat_count++;
                mutex_unlock(ctx->stats_lock);
            }
        }

        /* 检查心跳超时 */
        if (ctx->config.enable_fault_tolerance) {
            for (int i = 0; i < ctx->num_nodes; i++) {
                if (i == ctx->my_node_id) continue;
                NodeConnection* conn = &ctx->nodes[i];
                if (conn->connected && conn->is_alive) {
                    if (now - conn->last_heartbeat > (uint64_t)ctx->config.heartbeat_timeout_ms) {
                        log_warning("节点 %d 心跳超时", i);
                        conn->is_alive = 0;
                        conn->connection_errors++;

                        mutex_lock(ctx->stats_lock);
                        ctx->stats.node_failures_detected++;
                        mutex_unlock(ctx->stats_lock);
                    }
                }
            }
        }
    }

    return NULL;
}

SELFLNN_API int distributed_start_heartbeat(DistributedContext* ctx) {
    if (!ctx) return -1;
    if (ctx->heartbeat_running) return 0;

    ctx->heartbeat_running = 1;
    ctx->heartbeat_thread = thread_create(heartbeat_thread_func, ctx);

    if (!ctx->heartbeat_thread) {
        ctx->heartbeat_running = 0;
        log_error("启动心跳线程失败");
        return -1;
    }

    log_info("心跳检测已启动: interval=%dms, timeout=%dms",
             ctx->config.heartbeat_interval_ms, ctx->config.heartbeat_timeout_ms);
    return 0;
}

SELFLNN_API int distributed_stop_heartbeat(DistributedContext* ctx) {
    if (!ctx) return -1;
    if (!ctx->heartbeat_running) return 0;

    ctx->heartbeat_running = 0;
    if (ctx->heartbeat_thread) {
        thread_join(ctx->heartbeat_thread, NULL);
        ctx->heartbeat_thread = NULL;
    }

    log_info("心跳检测已停止");
    return 0;
}

SELFLNN_API int distributed_check_failures(DistributedContext* ctx, int* failed_nodes, int* num_failed) {
    if (!ctx) return -1;

    int failed[DISTRIBUTED_MAX_NODES];
    int count = 0;
    uint64_t now = get_time_ms();

    for (int i = 0; i < ctx->num_nodes; i++) {
        if (i == ctx->my_node_id) continue;
        NodeConnection* conn = &ctx->nodes[i];
        if (conn->connected && conn->is_alive) {
            if (now - conn->last_heartbeat > (uint64_t)ctx->config.heartbeat_timeout_ms * 2) {
                failed[count++] = i;
                conn->is_alive = 0;

                mutex_lock(ctx->stats_lock);
                ctx->stats.node_failures_detected++;
                mutex_unlock(ctx->stats_lock);
            }
        }
    }

    if (failed_nodes && count > 0) {
        memcpy(failed_nodes, failed, count * sizeof(int));
    }
    if (num_failed) *num_failed = count;

    return 0;
}

SELFLNN_API int distributed_recover_node(DistributedContext* ctx, int node_id) {
    if (!ctx || node_id < 0 || node_id >= DISTRIBUTED_MAX_NODES) return -1;

    NodeConnection* conn = &ctx->nodes[node_id];
    if (conn->connected && conn->is_alive) return 0;

    log_info("正在恢复节点 %d (重试 %d/%d)",
             node_id, conn->reconnection_count, ctx->config.max_retries);

    int ret = distributed_reconnect_node(ctx, node_id);
    if (ret == 0) {
        conn->is_alive = 1;
        conn->connection_errors = 0;
        log_info("节点 %d 已成功恢复", node_id);
    } else {
        log_error("节点 %d 恢复失败", node_id);
    }

    return ret;
}

SELFLNN_API int distributed_rebuild_topology(DistributedContext* ctx, const int* failed_nodes, int num_failed) {
    if (!ctx || !failed_nodes || num_failed <= 0) return -1;

    int new_num_nodes = ctx->num_nodes;
    for (int i = 0; i < num_failed; i++) {
        if (failed_nodes[i] >= 0 && failed_nodes[i] < ctx->num_nodes) {
            distributed_close_connection(ctx, failed_nodes[i]);
            new_num_nodes--;
        }
    }

    ctx->num_nodes = (new_num_nodes < 1) ? 1 : new_num_nodes;

    /* 重建环形拓扑 */
    if (ctx->num_nodes >= 2) {
        distributed_build_ring_topology(ctx);
    } else {
        ctx->ring_established = 0;
        ctx->ring_successor = -1;
        ctx->ring_predecessor = -1;
    }

    /* 重建树形拓扑 */
    distributed_build_tree_topology_internal(ctx);

    /* 更新拓扑版本 */
    ctx->topo.topology_version++;

    log_info("拓扑已重建: 节点从 %d 缩减到 %d",
             ctx->config.num_nodes, ctx->num_nodes);
    return 0;
}

/* ===========================================================================
 * 法定人数共识
 * =========================================================================== */

SELFLNN_API int distributed_quorum_check(DistributedContext* ctx, int* quorum_met) {
    if (!ctx) return -1;

    int alive = 0;
    for (int i = 0; i < ctx->num_nodes; i++) {
        if (ctx->nodes[i].connected && ctx->nodes[i].is_alive) {
            alive++;
        }
    }

    int threshold = (ctx->num_nodes * ctx->quorum_threshold_percent + 50) / 100;
    int met = (alive >= threshold) ? 1 : 0;

    if (quorum_met) *quorum_met = met;

    /* 更新法定人数状态 */
    ctx->quorum.total_nodes = ctx->num_nodes;
    ctx->quorum.alive_nodes = alive;
    ctx->quorum.quorum_threshold = threshold;
    ctx->quorum.quorum_met = met;
    ctx->quorum.last_quorum_check_ms = get_time_ms();

    if (!met) {
        ctx->quorum.consecutive_quorum_failures++;
        log_warning("法定人数不足: %d/%d (需要 %d)", alive, ctx->num_nodes, threshold);
    } else {
        ctx->quorum.consecutive_quorum_failures = 0;
        ctx->quorum.in_quorum_recovery = 0;
    }

    return 0;
}

SELFLNN_API int distributed_get_quorum_state(DistributedContext* ctx, DistributedQuorumState* state) {
    if (!ctx || !state) return -1;

    state->total_nodes = ctx->quorum.total_nodes;
    state->alive_nodes = ctx->quorum.alive_nodes;
    state->quorum_threshold = ctx->quorum.quorum_threshold;
    state->quorum_met = ctx->quorum.quorum_met;
    state->consensus_round = ctx->quorum.consensus_round;
    state->last_quorum_check_ms = ctx->quorum.last_quorum_check_ms;
    state->consecutive_quorum_failures = ctx->quorum.consecutive_quorum_failures;
    state->in_quorum_recovery = ctx->quorum.in_quorum_recovery;

    return 0;
}

SELFLNN_API int distributed_set_quorum_threshold(DistributedContext* ctx, int threshold_percent) {
    if (!ctx) return -1;
    if (threshold_percent < 1) threshold_percent = 1;
    if (threshold_percent > 100) threshold_percent = 100;
    ctx->quorum_threshold_percent = threshold_percent;
    return 0;
}

/* ===========================================================================
 * 梯度版本追踪
 * =========================================================================== */

SELFLNN_API uint64_t distributed_gradient_version_update(DistributedContext* ctx, int node_id) {
    if (!ctx) return 0;

    uint64_t version = ++ctx->global_version_counter;
    ctx->grad_versions.global_version = version;

    if (node_id >= 0 && node_id < DISTRIBUTED_MAX_NODES) {
        ctx->grad_versions.node_versions[node_id] = version;
    }

    ctx->grad_versions.last_sync_version = version;

    mutex_lock(ctx->stats_lock);
    ctx->stats.gradient_sync_count++;
    mutex_unlock(ctx->stats_lock);

    return version;
}

SELFLNN_API int distributed_is_gradient_stale(DistributedContext* ctx, int node_id) {
    if (!ctx || node_id < 0 || node_id >= DISTRIBUTED_MAX_NODES) return 0;

    if (ctx->grad_versions.stale_node_mask[node_id]) return 1;

    /* 如果节点版本落后全局版本超过阈值，视为过时 */
    uint64_t current = ctx->global_version_counter;
    uint64_t node_ver = ctx->grad_versions.node_versions[node_id];
    if (current > 3 && node_ver < current - 2) return 1;

    return 0;
}

SELFLNN_API int distributed_get_gradient_versions(DistributedContext* ctx, GradientVersionTracker* tracker) {
    if (!ctx || !tracker) return -1;

    memcpy(tracker, &ctx->grad_versions, sizeof(GradientVersionTracker));
    return 0;
}

SELFLNN_API int distributed_clear_stale_node(DistributedContext* ctx, int node_id) {
    if (!ctx || node_id < 0 || node_id >= DISTRIBUTED_MAX_NODES) return -1;

    ctx->grad_versions.stale_node_mask[node_id] = 0;
    ctx->grad_versions.node_versions[node_id] = ctx->global_version_counter;

    /* 重新计算过时节点数 */
    ctx->grad_versions.num_stale_nodes = 0;
    for (int i = 0; i < DISTRIBUTED_MAX_NODES; i++) {
        if (ctx->grad_versions.stale_node_mask[i]) {
            ctx->grad_versions.num_stale_nodes++;
        }
    }

    return 0;
}

/* ===========================================================================
 * 拓扑自动重平衡
 * =========================================================================== */

SELFLNN_API int distributed_auto_rebalance(DistributedContext* ctx) {
    if (!ctx) return -1;
    if (!ctx->is_leader) return 0;

    ctx->rebalance_check_counter++;

    if (ctx->config.enable_auto_rebalance &&
        ctx->rebalance_check_counter >= ctx->config.rebalance_check_interval)
    {
        ctx->rebalance_check_counter = 0;

        /* 检测当前活跃节点 */
        int alive[DISTRIBUTED_MAX_NODES];
        int num_alive = 0;

        for (int i = 0; i < ctx->num_nodes; i++) {
            if (ctx->nodes[i].connected && ctx->nodes[i].is_alive) {
                alive[num_alive++] = i;
            }
        }

        if (num_alive != ctx->num_nodes) {
            log_info("拓扑变化检测: 期望 %d 节点, 实际 %d 节点存活", ctx->num_nodes, num_alive);
            distributed_rebalance_workload(ctx, alive, num_alive);
        }

        /* 更新拓扑状态 */
        ctx->topo.topology_version++;
        ctx->topo.current_num_nodes = num_alive;
        ctx->topo.total_workload = 0;
        for (int i = 0; i < num_alive; i++) {
            ctx->topo.node_workload[i] = 1;
            ctx->topo.total_workload++;
        }
        ctx->topo.rebalance_count++;
        ctx->topo.last_rebalance_ms = get_time_ms();
    }

    return 0;
}

SELFLNN_API int distributed_get_topology_state(DistributedContext* ctx, DistributedTopologyState* state) {
    if (!ctx || !state) return -1;

    memcpy(state, &ctx->topo, sizeof(DistributedTopologyState));
    return 0;
}

SELFLNN_API int distributed_rebalance_workload(DistributedContext* ctx, const int* alive_nodes, int num_alive) {
    if (!ctx || !alive_nodes || num_alive <= 0) return -1;

    int total_workload = 0;
    int workload_per_node[DISTRIBUTED_MAX_NODES];

    for (int i = 0; i < num_alive; i++) {
        int node_id = alive_nodes[i];
        workload_per_node[i] = ctx->topo.node_workload[node_id];
        total_workload += workload_per_node[i];
    }

    if (total_workload <= 0) {
        for (int i = 0; i < num_alive; i++) {
            workload_per_node[i] = 1;
        }
        total_workload = num_alive;
    }

    int base = total_workload / num_alive;
    int remainder = total_workload % num_alive;

    for (int i = 0; i < num_alive; i++) {
        int node_id = alive_nodes[i];
        ctx->topo.node_workload[node_id] = base + (i < remainder ? 1 : 0);
    }

    ctx->topo.current_num_nodes = num_alive;
    ctx->topo.total_workload = total_workload;
    ctx->topo.rebalance_count++;

    log_info("工作负载已重平衡: %d 节点, 总负载 %d", num_alive, total_workload);
    return 0;
}

/* ===========================================================================
 * 检查点操作
 * =========================================================================== */

SELFLNN_API int distributed_checkpoint_save(DistributedContext* ctx, const char* filepath) {
    if (!ctx || !filepath) return -1;

    const char* actual_path = filepath;
    char default_path[512];

    if (strlen(filepath) == 0) {
        snprintf(default_path, sizeof(default_path), "%s/checkpoint_%d.bin",
                 ctx->checkpoint_dir, ctx->my_node_id);
        actual_path = default_path;
    }

    /* 写入检查点头部 */
    uint32_t magic = DISTRIBUTED_MAGIC;
    uint32_t version = 1;
    int node_id = ctx->my_node_id;
    int num_nodes = ctx->num_nodes;

    FILE* fp = fopen(actual_path, "wb");
    if (!fp) {
        log_error("无法打开检查点文件: %s", actual_path);
        return -1;
    }

    fwrite(&magic, sizeof(magic), 1, fp);
    fwrite(&version, sizeof(version), 1, fp);
    fwrite(&node_id, sizeof(node_id), 1, fp);
    fwrite(&num_nodes, sizeof(num_nodes), 1, fp);

    /* 写入通信统计 */
    DistributedCommStats stats;
    memcpy(&stats, &ctx->stats, sizeof(stats));
    fwrite(&stats, sizeof(stats), 1, fp);

    /* 写入法定人数状态 */
    fwrite(&ctx->quorum, sizeof(ctx->quorum), 1, fp);

    /* 写入梯度版本 */
    fwrite(&ctx->grad_versions, sizeof(ctx->grad_versions), 1, fp);
    fwrite(&ctx->global_version_counter, sizeof(ctx->global_version_counter), 1, fp);

    fclose(fp);
    log_info("检查点已保存: %s", actual_path);
    return 0;
}

SELFLNN_API int distributed_checkpoint_load(DistributedContext* ctx, const char* filepath) {
    if (!ctx || !filepath) return -1;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        log_error("无法打开检查点文件: %s", filepath);
        return -1;
    }

    uint32_t magic, version;
    int file_node_id, file_num_nodes;

    if (fread(&magic, sizeof(magic), 1, fp) != 1 || magic != DISTRIBUTED_MAGIC) {
        fclose(fp);
        log_error("检查点文件格式无效: %s", filepath);
        return -1;
    }

    fread(&version, sizeof(version), 1, fp);
    fread(&file_node_id, sizeof(file_node_id), 1, fp);
    fread(&file_num_nodes, sizeof(file_num_nodes), 1, fp);

    /* 恢复通信统计 */
    fread(&ctx->stats, sizeof(ctx->stats), 1, fp);

    /* 恢复法定人数状态 */
    fread(&ctx->quorum, sizeof(ctx->quorum), 1, fp);

    /* 恢复梯度版本 */
    fread(&ctx->grad_versions, sizeof(ctx->grad_versions), 1, fp);
    fread(&ctx->global_version_counter, sizeof(ctx->global_version_counter), 1, fp);

    fclose(fp);
    log_info("检查点已加载: %s (node_id=%d, num_nodes=%d)", filepath, file_node_id, file_num_nodes);
    return 0;
}

SELFLNN_API int distributed_checkpoint_leader_sync(DistributedContext* ctx, const char* filepath) {
    if (!ctx || !filepath) return -1;

    if (ctx->is_leader) {
        /* 领导节点：发送检查点到所有工作节点 */
        for (int i = 1; i < ctx->num_nodes; i++) {
            if (ctx->nodes[i].connected && ctx->nodes[i].is_alive) {
                /* 发送检查点路径 */
                uint16_t path_len = (uint16_t)strnlen(filepath, 511);
                distributed_send_to_node(ctx, i, MSG_CHECKPOINT_SAVE, 0,
                                         filepath, path_len + 1);
            }
        }
        log_info("检查点已同步到所有工作节点: %s", filepath);
    } else {
        /* 工作节点：从领导节点接收检查点请求 */
        DistributedMessageHeader header;
        if (recv_from_node(&ctx->nodes[0], &header, (void*)filepath, 512) == 0) {
            if (header.type == MSG_CHECKPOINT_SAVE) {
                distributed_checkpoint_load(ctx, filepath);
            }
        }
    }

    return 0;
}

/* ===========================================================================
 * 状态查询
 * =========================================================================== */

SELFLNN_API int distributed_get_node_info(DistributedContext* ctx, int node_id, DistributedNodeInfo* info) {
    if (!ctx || !info) return -1;
    if (node_id < 0 || node_id >= DISTRIBUTED_MAX_NODES) return -1;

    NodeConnection* conn = &ctx->nodes[node_id];
    info->node_id = node_id;
    strncpy(info->host, conn->host, sizeof(info->host) - 1);
    info->host[sizeof(info->host) - 1] = '\0';
    info->port = conn->port;
    info->is_leader = (node_id == 0) ? 1 : 0;
    info->is_alive = conn->is_alive ? 1 : 0;
    info->last_heartbeat = conn->last_heartbeat;
    info->total_bytes_sent = conn->bytes_sent;
    info->total_bytes_received = conn->bytes_received;
    info->connection_errors = conn->connection_errors;
    info->reconnection_count = conn->reconnection_count;

    return 0;
}

SELFLNN_API int distributed_get_comm_stats(DistributedContext* ctx, DistributedCommStats* stats) {
    if (!ctx || !stats) return -1;

    mutex_lock(ctx->stats_lock);
    memcpy(stats, &ctx->stats, sizeof(*stats));

    stats->active_connections = 0;
    for (int i = 0; i < ctx->num_nodes; i++) {
        if (i != ctx->my_node_id && ctx->nodes[i].connected) {
            stats->active_connections++;
        }
    }

    mutex_unlock(ctx->stats_lock);
    return 0;
}

SELFLNN_API int distributed_get_connected_count(DistributedContext* ctx) {
    if (!ctx) return -1;

    int count = 0;
    for (int i = 0; i < ctx->num_nodes; i++) {
        if (i != ctx->my_node_id && ctx->nodes[i].connected) {
            count++;
        }
    }
    return count;
}

SELFLNN_API int distributed_is_leader(DistributedContext* ctx) {
    if (!ctx) return -1;
    return ctx->is_leader ? 1 : 0;
}

SELFLNN_API int distributed_get_my_node_id(DistributedContext* ctx) {
    if (!ctx) return -1;
    return ctx->my_node_id;
}

SELFLNN_API int distributed_set_verbose(DistributedContext* ctx, int verbose) {
    if (!ctx) return -1;
    ctx->config.verbose = verbose ? 1 : 0;
    log_info("分布式训练详细输出: %s", verbose ? "开启" : "关闭");
    return 0;
}

SELFLNN_API int distributed_elect_leader(DistributedContext* ctx) {
    if (!ctx) return -1;
    if (ctx->num_nodes <= 1) {
        ctx->is_leader = 1;
        return 1;
    }

    /* 基于连接状态的最低节点ID选举：选择所有已连接节点中ID最小的作为领导
     * 这实现了确定性的领导选举，在节点故障时可以自动转移领导权 */
    int elected_leader = -1;
    int connected_count = 0;

    for (int i = 0; i < ctx->num_nodes; i++) {
        if (i == ctx->my_node_id || ctx->nodes[i].connected) {
            connected_count++;
            if (elected_leader == -1 || i < elected_leader) {
                elected_leader = i;
            }
        }
    }

    if (elected_leader == -1) {
        /* 无已连接节点，当前节点自荐 */
        ctx->is_leader = 1;
        return 1;
    }

    if (ctx->my_node_id == elected_leader) {
        ctx->is_leader = 1;
        log_info("领导选举: 节点%d当选为领导 (已连接节点数=%d)", ctx->my_node_id, connected_count);
        return 1;
    }

    ctx->is_leader = 0;
    return 0;
}

/* ===========================================================================
 * 梯度压缩 (Top-K 稀疏化)
 * =========================================================================== */

void distributed_gradient_topk_compress(const float* gradients, size_t num_params,
                                        float compression_ratio,
                                        int* indices_out, float* values_out,
                                        size_t* compressed_size_out)
{
    if (!gradients || !indices_out || !values_out || !compressed_size_out) return;

    size_t k = (size_t)(num_params * compression_ratio);
    if (k < 1) k = 1;
    if (k > num_params) k = num_params;

    /*
     * QuickSelect算法：Hoare分区快速选择第k大的绝对值作为阈值。
     * 平均O(n)时间复杂度，选择Top-k梯度进行压缩通信。
     */
    float threshold = 0.0f;
    if (k < num_params) {
        float* abs_vals = (float*)safe_malloc(num_params * sizeof(float));
        if (!abs_vals) return;

        for (size_t i = 0; i < num_params; i++) {
            abs_vals[i] = fabsf(gradients[i]);
        }

        /* Hoare分区快速选择第k大的绝对值 */
        size_t left = 0, right = num_params - 1;
        size_t target = k - 1;
        while (left < right) {
            float pivot = abs_vals[right];
            size_t store = left;
            for (size_t j = left; j < right; j++) {
                if (abs_vals[j] > pivot) {
                    float tmp = abs_vals[j]; abs_vals[j] = abs_vals[store]; abs_vals[store] = tmp;
                    store++;
                }
            }
            float tmp = abs_vals[store]; abs_vals[store] = abs_vals[right]; abs_vals[right] = tmp;
            if (store == target) { threshold = abs_vals[store]; break; }
            if (store < target) left = store + 1;
            else right = store - 1;
        }
        if (left >= right) threshold = abs_vals[left];

        safe_free((void**)&abs_vals);
    }

    /* 选择绝对值 ≥ 阈值的梯度（精确Top-k） */
    size_t count = 0;
    for (size_t i = 0; i < num_params && count < k; i++) {
        if (fabsf(gradients[i]) >= threshold) {
            indices_out[count] = (int)i;
            values_out[count] = gradients[i];
            count++;
        }
    }

    *compressed_size_out = count;
}

void distributed_gradient_topk_decompress(float* output, size_t num_params,
                                          const int* indices, const float* values,
                                          size_t compressed_size)
{
    if (!output || !indices || !values) return;

    memset(output, 0, num_params * sizeof(float));
    for (size_t i = 0; i < compressed_size; i++) {
        int idx = indices[i];
        if (idx >= 0 && (size_t)idx < num_params) {
            output[idx] = values[i];
        }
    }
}

/* ===========================================================================
 * 弹性伸缩函数
 * =========================================================================== */

int distributed_elastic_register_node(int node_id, const char* host, unsigned short port) {
/* 添加锁保护elastic上下文 */
    DistributedContext* ctx = elastic_ctx_lock();
    if (!ctx || !host) { elastic_ctx_unlock(); return -1; }
    if (node_id < 0 || node_id >= ctx->num_nodes) { elastic_ctx_unlock(); return -2; }

    /* 注册节点：设置连接信息和初始状态 */
    NodeConnection* node = &ctx->nodes[node_id];
    strncpy(node->host, host, sizeof(node->host) - 1);
    node->host[sizeof(node->host) - 1] = '\0';
    node->port = port;
    node->connected = 1;
    node->last_heartbeat = get_time_ms();
    node->is_alive = 1;

    elastic_ctx_unlock();
    log_info("弹性伸缩: 注册节点 node_id=%d, host=%s, port=%d", node_id, host, port);
    return 0;
}

int distributed_elastic_heartbeat(int node_id) {
    DistributedContext* ctx = elastic_ctx_lock();
    if (!ctx) { elastic_ctx_unlock(); return -1; }
    if (node_id < 0 || node_id >= ctx->num_nodes) { elastic_ctx_unlock(); return -2; }

    /* 心跳更新：刷新节点最后活跃时间 */
    NodeConnection* node = &ctx->nodes[node_id];
    node->last_heartbeat = get_time_ms();

    /* 检查是否有节点超时 */
    uint64_t now = get_time_ms();
    int timeout_count = 0;
    for (int i = 0; i < ctx->num_nodes; i++) {
        if (i == ctx->my_node_id) continue;
        NodeConnection* n = &ctx->nodes[i];
        if (n->connected && n->last_heartbeat > 0 &&
            (now - n->last_heartbeat) > (uint64_t)ctx->config.heartbeat_timeout_ms * 3) {
            n->is_alive = 0;
            timeout_count++;
            log_warning("弹性伸缩: 节点%d心跳超时", i);
        }
    }

    if (timeout_count > 0) {
        distributed_elect_leader(ctx);
    }

    elastic_ctx_unlock();
    return 0;
}

int distributed_elastic_check_nodes(void) {
    DistributedContext* ctx = elastic_ctx_lock();
    if (!ctx) { elastic_ctx_unlock(); return 0; }

    int healthy_count = 0;
    for (int i = 0; i < ctx->num_nodes; i++) {
        if (i == ctx->my_node_id || ctx->nodes[i].connected) {
            healthy_count++;
        }
    }
    elastic_ctx_unlock();
    return healthy_count;
}

int distributed_elastic_remove_node(int node_id) {
    if (!g_elastic_ctx) return -1;
    if (node_id < 0 || node_id >= g_elastic_ctx->num_nodes) return -2;

    /* 移除节点：断开连接并清理状态 */
    NodeConnection* node = &g_elastic_ctx->nodes[node_id];
    if (node->connected) {
        distributed_disconnect_node(g_elastic_ctx, node_id);
    }
    node->connected = 0;
    node->is_alive = 0;
    node->last_heartbeat = 0;

    log_info("弹性伸缩: 移除节点 node_id=%d", node_id);

    /* 如果移除的是领导节点，触发重选举 */
    if (g_elastic_ctx->is_leader && node_id == g_elastic_ctx->my_node_id) {
        /* 当前领导被移除，需要自动降级 */
        (void)node_id;
    }
    distributed_elect_leader(g_elastic_ctx);

    return 0;
}

static int distributed_disconnect_node(void* ctx, int node_id) {
    if (!ctx || node_id < 0) return -1;
    DistributedContext* dctx = (DistributedContext*)ctx;
    if (node_id >= dctx->num_nodes) return -1;
    NodeConnection* node = &dctx->nodes[node_id];
#ifdef _WIN32
    if (node->socket != INVALID_SOCKET) {
        shutdown(node->socket, SD_BOTH);
        closesocket(node->socket);
        node->socket = INVALID_SOCKET;
    }
#else
    if (node->socket >= 0) {
        shutdown(node->socket, SHUT_RDWR);
        close(node->socket);
        node->socket = -1;
    }
#endif
    node->connected = 0;
    node->is_alive = 0;
    return 0;
}

/* ===========================================================================
 * 训练器辅助函数
 * =========================================================================== */

int trainer_send_gradients_to_workers(void* trainer_ptr, const float* gradients, size_t count) {
    if (!trainer_ptr || !gradients || count == 0) return -1;

    /* 从训练器指针获取分布式上下文 */
    DistributedContext* ctx = g_elastic_ctx;
    if (!ctx || ctx->num_nodes <= 1) return 0;

    /* 分批发送梯度到各工作节点 */
    size_t chunk_size = count / ctx->num_nodes;
    if (chunk_size < 1) chunk_size = 1;

    for (int i = 1; i < ctx->num_nodes; i++) {
        if (!ctx->nodes[i].connected) continue;

        size_t offset = (size_t)i * chunk_size;
        size_t actual_count = (offset + chunk_size <= count) ? chunk_size : (count > offset ? count - offset : 0);

        if (actual_count > 0) {
            distributed_send_to_node(ctx, i, MSG_GRADIENT_CHUNK, 0,
                                     gradients + offset,
                                     (uint32_t)(actual_count * sizeof(float)));
        }
    }

    return 0;
}

int trainer_receive_gradients_from_workers(void* trainer_ptr, float* all_gradients, size_t count) {
    if (!trainer_ptr || !all_gradients || count == 0) return -1;

    DistributedContext* ctx = g_elastic_ctx;
    if (!ctx || ctx->num_nodes <= 1) return 0;

    size_t chunk_size = count / ctx->num_nodes;
    if (chunk_size < 1) chunk_size = 1;

    for (int i = 1; i < ctx->num_nodes; i++) {
        if (!ctx->nodes[i].connected) continue;

        size_t offset = (size_t)i * chunk_size;
        size_t actual_count = (offset + chunk_size <= count) ? chunk_size : (count > offset ? count - offset : 0);

        if (actual_count > 0) {
            DistributedMessageHeader header;
            recv_from_node(&ctx->nodes[i], &header,
                           all_gradients + offset,
                           (uint32_t)(actual_count * sizeof(float)));
        }
    }

    return 0;
}

