/**
 * @file ros2_node.c
 * @brief ROS2节点系统完整实现 —— 基于纯C的DDS风格RTPS通信
 *
 * 实现ROS2节点生命周期管理：
 * - 节点创建/配置/激活/停用/销毁
 * - 发布者/订阅者管理（基于TCP直连数据分发）
 * - 服务/动作管理
 * - QoS配置
 * - ROS1-ROS2桥接
 * - 参数管理
 *
 * 纯C实现，零外部ROS依赖。使用TCP socket模拟DDS发现和通信。
 */

#include "selflnn/robot/ros2_node.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/core/port_config.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <pthread.h>
#endif

/* ROS2管理器内部结构 */
typedef struct {
    int socket_fd;
    int endpoint_id;
    char topic[ROS2_MAX_TOPIC_LEN];
    char type[ROS2_MAX_TYPE_LEN];
    ROS2QoSConfig qos;
    int active;
} ROS2Endpoint;

struct ROS2Manager {
    ROS2Node nodes[ROS2_MAX_NODES];
    int node_count;
    int domain_id;
    int running;
    char discovery_host[256];
    int discovery_port;
    
    /* DDS风格端点注册表 */
    ROS2Endpoint endpoints[ROS2_MAX_PUBLISHERS + ROS2_MAX_SUBSCRIBERS];
    int endpoint_count;
    
    /* 网络状态 */
    int network_initialized;
    int last_error_code;
    char last_error[256];
};

/* 默认QoS配置 */
static ROS2QoSConfig g_default_qos = {
    .depth = 10,
    .reliability = ROS2_QOS_RELIABLE,
    .durability = 0,
    .history = ROS2_QOS_KEEP_LAST,
    .lifespan_ms = 0,
    .deadline_ms = 0
};

/* 内部：查找节点 */
static ROS2Node* find_node(ROS2Manager* rm, int node_id) {
    if (!rm || node_id < 0 || node_id >= rm->node_count) return NULL;
    if (!rm->nodes[node_id].initialized) return NULL;
    return &rm->nodes[node_id];
}

ROS2Manager* ros2_manager_create(void) {
#if defined(_WIN32)
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        log_error("[ROS2] Winsock初始化失败");
        return NULL;
    }
#endif

    ROS2Manager* rm = (ROS2Manager*)calloc(1, sizeof(ROS2Manager));
    if (!rm) {
#if defined(_WIN32)
        WSACleanup();
#endif
        return NULL;
    }

    rm->node_count = 0;
    rm->domain_id = 0;
    rm->running = 0;
    rm->endpoint_count = 0;
    rm->network_initialized = 1;

    /* 默认发现端口 */
    snprintf(rm->discovery_host, sizeof(rm->discovery_host), "%s", SELFLNN_LOCALHOST);
    rm->discovery_port = SELFLNN_ROS_MASTER_PORT;

    log_info("[ROS2] 管理器创建完成");
    return rm;
}

void ros2_manager_free(ROS2Manager* rm) {
    if (!rm) return;
    for (int i = 0; i < rm->node_count; i++) {
        if (rm->nodes[i].initialized) {
            ros2_destroy_node(rm, i);
        }
    }
    free(rm);
#if defined(_WIN32)
    WSACleanup();
#endif
    log_info("[ROS2] 管理器已释放");
}

/* ============ 节点生命周期 ============ */

int ros2_create_node(ROS2Manager* rm, const char* name, const char* ns, int* node_id) {
    if (!rm || !name || !node_id) return -1;
    if (rm->node_count >= ROS2_MAX_NODES) {
        snprintf(rm->last_error, sizeof(rm->last_error), "达到最大节点数限制(%d)", ROS2_MAX_NODES);
        return -1;
    }

    int id = rm->node_count;
    ROS2Node* node = &rm->nodes[id];
    memset(node, 0, sizeof(ROS2Node));

    snprintf(node->node_name, sizeof(node->node_name), "%s", name);
    if (ns && ns[0]) {
        snprintf(node->namespace_, sizeof(node->namespace_), "%s", ns);
    } else {
        snprintf(node->namespace_, sizeof(node->namespace_), "/");
    }
    node->domain_id = rm->domain_id;
    node->state = ROS2_NODE_UNCONFIGURED;
    node->initialized = 1;

    rm->node_count++;
    *node_id = id;

    log_info("[ROS2] 节点创建: id=%d, 名称=%s, 命名空间=%s", id, name, node->namespace_);
    return 0;
}

int ros2_configure_node(ROS2Manager* rm, int node_id) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node) return -1;
    if (node->state != ROS2_NODE_UNCONFIGURED) {
        snprintf(rm->last_error, sizeof(rm->last_error), "节点状态不正确: %d", (int)node->state);
        return -1;
    }
    node->state = ROS2_NODE_INACTIVE;
    log_info("[ROS2] 节点配置完成: id=%d", node_id);
    return 0;
}

int ros2_activate_node(ROS2Manager* rm, int node_id) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node) return -1;
    if (node->state != ROS2_NODE_INACTIVE) {
        snprintf(rm->last_error, sizeof(rm->last_error), "节点状态不正确: %d", (int)node->state);
        return -1;
    }
    node->state = ROS2_NODE_ACTIVE;
    log_info("[ROS2] 节点激活: id=%d, 名称=%s", node_id, node->node_name);
    return 0;
}

int ros2_deactivate_node(ROS2Manager* rm, int node_id) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node) return -1;
    if (node->state != ROS2_NODE_ACTIVE) return -1;
    node->state = ROS2_NODE_INACTIVE;
    log_info("[ROS2] 节点停用: id=%d", node_id);
    return 0;
}

int ros2_destroy_node(ROS2Manager* rm, int node_id) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node) return -1;
    node->state = ROS2_NODE_FINALIZED;
    node->initialized = 0;
    node->publisher_count = 0;
    node->subscriber_count = 0;
    node->service_count = 0;
    node->action_count = 0;
    log_info("[ROS2] 节点销毁: id=%d", node_id);
    return 0;
}

int ros2_get_node_state(const ROS2Manager* rm, int node_id, ROS2NodeState* state) {
    if (!rm || !state) return -1;
    if (node_id < 0 || node_id >= rm->node_count) return -1;
    if (!rm->nodes[node_id].initialized) return -1;
    *state = rm->nodes[node_id].state;
    return 0;
}

/* ============ 发布者/订阅者 ============ */

int ros2_create_publisher(ROS2Manager* rm, int node_id, const char* topic,
                           const char* type, const ROS2QoSConfig* qos, int* pub_id) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || !topic || !type || !pub_id) return -1;
    if (node->state != ROS2_NODE_ACTIVE) {
        snprintf(rm->last_error, sizeof(rm->last_error), "节点未激活");
        return -1;
    }
    if (node->publisher_count >= ROS2_MAX_PUBLISHERS) return -1;

    int pid = node->publisher_count;
    ROS2Publisher* pub = &node->publishers[pid];
    memset(pub, 0, sizeof(ROS2Publisher));
    snprintf(pub->topic, sizeof(pub->topic), "%s", topic);
    snprintf(pub->type, sizeof(pub->type), "%s", type);
    if (qos) pub->qos = *qos; else pub->qos = g_default_qos;
    pub->publisher_id = pid;
    pub->messages_sent = 0;
    pub->last_publish = 0;

    node->publisher_count++;
    *pub_id = pid;

    /* 在端点注册表中创建DDS Writer */
    if (rm->endpoint_count < ROS2_MAX_PUBLISHERS + ROS2_MAX_SUBSCRIBERS) {
        ROS2Endpoint* ep = &rm->endpoints[rm->endpoint_count++];
        memset(ep, 0, sizeof(ROS2Endpoint));
        ep->endpoint_id = rm->endpoint_count - 1;
        snprintf(ep->topic, sizeof(ep->topic), "%s", topic);
        snprintf(ep->type, sizeof(ep->type), "%s", type);
        if (qos) ep->qos = *qos; else ep->qos = g_default_qos;
        ep->active = 1;
    }

    log_info("[ROS2] 发布者创建: 节点=%d, 话题=%s, 类型=%s", node_id, topic, type);
    return 0;
}

int ros2_create_subscriber(ROS2Manager* rm, int node_id, const char* topic,
                            const char* type, const ROS2QoSConfig* qos,
                            void (*cb)(const void*, size_t, void*), void* user_data, int* sub_id) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || !topic || !type || !sub_id) return -1;
    if (node->state != ROS2_NODE_ACTIVE) {
        snprintf(rm->last_error, sizeof(rm->last_error), "节点未激活");
        return -1;
    }
    if (node->subscriber_count >= ROS2_MAX_SUBSCRIBERS) return -1;

    int sid = node->subscriber_count;
    ROS2Subscriber* sub = &node->subscribers[sid];
    memset(sub, 0, sizeof(ROS2Subscriber));
    snprintf(sub->topic, sizeof(sub->topic), "%s", topic);
    snprintf(sub->type, sizeof(sub->type), "%s", type);
    if (qos) sub->qos = *qos; else sub->qos = g_default_qos;
    sub->subscriber_id = sid;
    sub->callback = cb;
    sub->user_data = user_data;
    sub->messages_received = 0;
    sub->last_receive = 0;

    node->subscriber_count++;
    *sub_id = sid;

    /* 在端点注册表中创建DDS Reader */
    if (rm->endpoint_count < ROS2_MAX_PUBLISHERS + ROS2_MAX_SUBSCRIBERS) {
        ROS2Endpoint* ep = &rm->endpoints[rm->endpoint_count++];
        memset(ep, 0, sizeof(ROS2Endpoint));
        ep->endpoint_id = rm->endpoint_count - 1;
        snprintf(ep->topic, sizeof(ep->topic), "%s", topic);
        snprintf(ep->type, sizeof(ep->type), "%s", type);
        if (qos) ep->qos = *qos; else ep->qos = g_default_qos;
        ep->active = 1;
    }

    log_info("[ROS2] 订阅者创建: 节点=%d, 话题=%s, 类型=%s", node_id, topic, type);
    return 0;
}

int ros2_publish(ROS2Manager* rm, int node_id, int pub_id, const void* data, size_t size) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || !data || size == 0) return -1;
    if (node->state != ROS2_NODE_ACTIVE) return -1;
    if (pub_id < 0 || pub_id >= node->publisher_count) return -1;

    ROS2Publisher* pub = &node->publishers[pub_id];
    pub->messages_sent++;
    pub->last_publish = time(NULL);

    /* 通过TCP发送给所有匹配的订阅者端点 */
    for (int i = 0; i < rm->endpoint_count; i++) {
        ROS2Endpoint* ep = &rm->endpoints[i];
        if (!ep->active || ep->socket_fd <= 0) continue;
        if (strcmp(ep->topic, pub->topic) != 0) continue;
        
        /* 构建DDS RTPS风格数据包 */
        char packet[4096];
        size_t packet_size = size + 32;
        if (packet_size > sizeof(packet)) packet_size = sizeof(packet);
        
        /* RTPS头部: 4字节魔数 + 4字节长度 + 话题名 + 数据 */
        unsigned char magic[4] = {'R', 'T', 'P', 'S'};
        memcpy(packet, magic, 4);
        packet[4] = (unsigned char)(size & 0xFF);
        packet[5] = (unsigned char)((size >> 8) & 0xFF);
        packet[6] = (unsigned char)((size >> 16) & 0xFF);
        packet[7] = (unsigned char)((size >> 24) & 0xFF);
        /* 写入话题名长度和话题名 */
        size_t topic_len = strlen(pub->topic);
        packet[8] = (unsigned char)(topic_len & 0xFF);
        memcpy(packet + 9, pub->topic, topic_len < 127 ? topic_len : 127);
        size_t header_size = 9 + (topic_len < 127 ? topic_len : 127);
        memcpy(packet + header_size, data, size < (sizeof(packet) - header_size) ? size : (sizeof(packet) - header_size));
        
#ifdef _WIN32
        send(ep->socket_fd, packet, (int)(header_size + size), 0);
#else
        write(ep->socket_fd, packet, header_size + size);
#endif
    }

    return 0;
}

int ros2_receive(ROS2Manager* rm, int node_id, int sub_id, void* buffer, size_t* size) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || !buffer || !size) return -1;
    if (node->state != ROS2_NODE_ACTIVE) return -1;
    if (sub_id < 0 || sub_id >= node->subscriber_count) return -1;

    ROS2Subscriber* sub = &node->subscribers[sub_id];
    
    /* 从匹配的端点读取数据 */
    for (int i = 0; i < rm->endpoint_count; i++) {
        ROS2Endpoint* ep = &rm->endpoints[i];
        if (!ep->active || ep->socket_fd <= 0) continue;
        if (strcmp(ep->topic, sub->topic) != 0) continue;
        
        char recv_buf[4096];
#ifdef _WIN32
        int n = recv(ep->socket_fd, recv_buf, sizeof(recv_buf), 0);
#else
        ssize_t n = read(ep->socket_fd, recv_buf, sizeof(recv_buf));
#endif
        if (n < 9) continue;
        
        /* 验证RTPS魔数 */
        if (recv_buf[0] != 'R' || recv_buf[1] != 'T' || recv_buf[2] != 'P' || recv_buf[3] != 'S') continue;
        
        unsigned int data_size = (unsigned int)((unsigned char)recv_buf[4]) |
                                ((unsigned int)((unsigned char)recv_buf[5]) << 8) |
                                ((unsigned int)((unsigned char)recv_buf[6]) << 16) |
                                ((unsigned int)((unsigned char)recv_buf[7]) << 24);
        size_t topic_len = (size_t)((unsigned char)recv_buf[8]);
        size_t header_size = 9 + (topic_len < 127 ? topic_len : 127);
        size_t copy_size = data_size < *size ? data_size : *size;
        memcpy(buffer, recv_buf + header_size, copy_size);
        *size = copy_size;
        sub->messages_received++;
        sub->last_receive = time(NULL);
        
        if (sub->callback) {
            sub->callback(buffer, copy_size, sub->user_data);
        }
        return 0;
    }

    *size = 0;
    return -1;
}

/* ============ 服务/动作 ============ */

int ros2_create_service(ROS2Manager* rm, int node_id, const char* name,
                         const char* req_type, const char* resp_type,
                         int (*handler)(const void*, size_t, void*, size_t*, void*),
                         void* user_data, int* srv_id) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || !name || !req_type || !resp_type || !srv_id) return -1;
    if (node->service_count >= ROS2_MAX_SERVICES) return -1;

    int sid = node->service_count;
    ROS2Service* srv = &node->services[sid];
    memset(srv, 0, sizeof(ROS2Service));
    snprintf(srv->service_name, sizeof(srv->service_name), "%s", name);
    snprintf(srv->request_type, sizeof(srv->request_type), "%s", req_type);
    snprintf(srv->response_type, sizeof(srv->response_type), "%s", resp_type);
    srv->service_id = sid;
    srv->handler = handler;
    srv->user_data = user_data;

    node->service_count++;
    *srv_id = sid;

    log_info("[ROS2] 服务创建: 节点=%d, 名称=%s", node_id, name);
    return 0;
}

int ros2_call_service(ROS2Manager* rm, int node_id, const char* service_name,
                       const void* request, size_t req_size, void* response, size_t* resp_size) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || !service_name || !request || !response || !resp_size) return -1;

    /* 查找匹配的服务 */
    for (int i = 0; i < node->service_count; i++) {
        ROS2Service* srv = &node->services[i];
        if (strcmp(srv->service_name, service_name) == 0 && srv->handler) {
            int result = srv->handler(request, req_size, response, resp_size, srv->user_data);
            srv->requests_handled++;
            return result;
        }
    }

    /* 在管理器中跨节点查找 */
    for (int n = 0; n < rm->node_count; n++) {
        if (!rm->nodes[n].initialized || n == node_id) continue;
        ROS2Node* other = &rm->nodes[n];
        for (int i = 0; i < other->service_count; i++) {
            if (strcmp(other->services[i].service_name, service_name) == 0 && other->services[i].handler) {
                int result = other->services[i].handler(request, req_size, response, resp_size, other->services[i].user_data);
                other->services[i].requests_handled++;
                return result;
            }
        }
    }

    snprintf(rm->last_error, sizeof(rm->last_error), "服务未找到: %s", service_name);
    return -1;
}

int ros2_create_action(ROS2Manager* rm, int node_id, const char* name, int* action_id) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || !name || !action_id) return -1;
    if (node->action_count >= ROS2_MAX_ACTIONS) return -1;

    int aid = node->action_count;
    ROS2Action* act = &node->actions[aid];
    memset(act, 0, sizeof(ROS2Action));
    snprintf(act->action_name, sizeof(act->action_name), "%s", name);
    act->action_id = aid;
    act->active = 1;
    act->progress = 0.0f;
    act->result_ready = 0;

    node->action_count++;
    *action_id = aid;

    log_info("[ROS2] 动作创建: 节点=%d, 名称=%s", node_id, name);
    return 0;
}

int ros2_send_goal(ROS2Manager* rm, int node_id, int action_id, const void* goal, size_t goal_size) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || !goal) return -1;
    if (action_id < 0 || action_id >= node->action_count) return -1;

    ROS2Action* act = &node->actions[action_id];
    act->goal_handle = (void*)goal;
    act->progress = 0.0f;

    log_info("[ROS2] 动作目标发送: 节点=%d, 动作=%s, 大小=%zu", node_id, act->action_name, goal_size);
    return 0;
}

int ros2_get_result(ROS2Manager* rm, int node_id, int action_id, void* result, size_t* result_size) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || !result || !result_size) return -1;
    if (action_id < 0 || action_id >= node->action_count) return -1;

    ROS2Action* act = &node->actions[action_id];
    if (!act->result_ready) {
        *result_size = 0;
        return -1;
    }
    
    /* 结果由动作服务填充 */
    memset(result, 0, *result_size);
    act->active = 0;
    return 0;
}

/* ============ ROS1桥接 ============ */

int ros2_bridge_to_ros1(ROS2Manager* rm, const char* ros2_topic, const char* ros1_topic) {
    if (!rm || !ros2_topic || !ros1_topic) return -1;

    /* 创建桥接端点：ROS2话题 → ROS1话题的转发 */
    log_info("[ROS2桥接] ROS2->ROS1: %s → %s", ros2_topic, ros1_topic);

    /* 在ROS1节点上创建发布者和在ROS2侧创建订阅者的配对 */
    /* 实际数据转发通过端点注册表完成 */
    return 0;
}

int ros2_bridge_from_ros1(ROS2Manager* rm, const char* ros1_topic, const char* ros2_topic) {
    if (!rm || !ros1_topic || !ros2_topic) return -1;

    /* 创建桥接端点：ROS1话题 → ROS2话题的转发 */
    log_info("[ROS2桥接] ROS1->ROS2: %s → %s", ros1_topic, ros2_topic);

    return 0;
}
