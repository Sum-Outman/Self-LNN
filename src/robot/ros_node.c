/**
 * @file ros_node.c
 * @brief 真实ROS节点实现 —— 基于rosbridge WebSocket协议的ROS节点通信
 *
 * 通过rosbridge WebSocket JSON协议与ROS Master通信。
 * 使用已有的RosBridge底层实现（src/robot/ros_bridge.c）进行WebSocket通信。
 * 支持：话题发布/订阅、服务广告/调用、节点管理。
 *
 * 纯C实现，零外部ROS依赖。
 */

#include "selflnn/robot/ros_node.h"
#include "selflnn/robot/ros_bridge.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ROS节点内部结构 */
struct RosNode {
    RosNodeConfig config;
    RosBridge* bridge;
    int connected;
    int state;

    /* 发布者列表 */
    struct {
        char topic[ROS_MAX_TOPIC_NAME];
        char type[ROS_MAX_TYPE_NAME];
        int active;
    } publishers[ROS_NODE_MAX_PUBLISHERS];
    int num_publishers;

    /* 订阅者列表 */
    struct {
        char topic[ROS_MAX_TOPIC_NAME];
        char type[ROS_MAX_TYPE_NAME];
        RosMessageCallback callback;
        void* user_data;
        int active;
    } subscribers[ROS_NODE_MAX_SUBSCRIBERS];
    int num_subscribers;

    /* 服务列表 */
    struct {
        char name[ROS_MAX_TOPIC_NAME];
        RosServiceCallback callback;
        void* user_data;
        int active;
    } services[ROS_NODE_MAX_SERVICES];
    int num_services;

    char master_uri[512];
    char bridge_host_buf[256];  /* ZSFAB-MS: ros_bridge_create需要持久化指针，使用结构体成员而非栈变量 */
    char last_error[256];
};

/* 构建rosbridge连接URI */
static void build_rosbridge_uri(const RosNodeConfig* config, char* uri, size_t size) {
    snprintf(uri, size, "ws://%s:%d", config->master_host,
             config->master_port == 11311 ? 9090 : config->master_port);
}

RosNodeConfig ros_node_config_default(const char* node_name) {
    RosNodeConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    if (node_name) {
        snprintf(cfg.node_name, sizeof(cfg.node_name), "%s", node_name);
    } else {
        snprintf(cfg.node_name, sizeof(cfg.node_name), "selflnn_node");
    }
    snprintf(cfg.master_host, sizeof(cfg.master_host), "127.0.0.1");
    cfg.master_port = 11311;
    snprintf(cfg.node_host, sizeof(cfg.node_host), "127.0.0.1");
    cfg.xmlrpc_port = 0;
    cfg.tcp_port = 0;
    cfg.tcp_timeout_ms = 5000;
    cfg.heartbeat_interval_ms = 10000;
    cfg.enable_auto_reconnect = 1;
    cfg.max_reconnect_attempts = 5;
    return cfg;
}

RosNode* ros_node_create(const RosNodeConfig* config) {
    if (!config) return NULL;
    RosNode* node = (RosNode*)safe_calloc(1, sizeof(RosNode));
    if (!node) return NULL;

    node->config = *config;
    node->connected = 0;
    node->state = 0;
    node->num_publishers = 0;
    node->num_subscribers = 0;
    node->num_services = 0;
    node->bridge = NULL;
    node->last_error[0] = '\0';

    log_info("[ROS节点] 节点 '%s' 已创建，目标主节点: %s:%d",
             config->node_name, config->master_host, config->master_port);
    return node;
}

void ros_node_destroy(RosNode* node) {
    if (!node) return;
    ros_node_disconnect_from_master(node);
    safe_free((void**)&node);
}

int ros_node_connect_to_master(RosNode* node, const char* master_uri) {
    if (!node) return -1;

    if (master_uri) {
        snprintf(node->master_uri, sizeof(node->master_uri), "%s", master_uri);
    } else {
        snprintf(node->master_uri, sizeof(node->master_uri), "http://%s:%d",
                 node->config.master_host, node->config.master_port);
    }

    /* 创建RosBridge连接配置 */
    RosBridgeConfig bridge_cfg;
    memset(&bridge_cfg, 0, sizeof(bridge_cfg));
    bridge_cfg.version = ROS_VERSION_1;
    bridge_cfg.use_rosbridge = 1;
    bridge_cfg.auto_reconnect = node->config.enable_auto_reconnect;
    bridge_cfg.reconnect_timeout = 3.0f;

    char* bridge_host = node->bridge_host_buf;
    memset(bridge_host, 0, sizeof(node->bridge_host_buf));
    snprintf(bridge_host, sizeof(node->bridge_host_buf), "%s", node->config.master_host);
    bridge_cfg.bridge_host = bridge_host;
    bridge_cfg.bridge_port = 9090;

    /* 释放旧连接 */
    if (node->bridge) {
        ros_bridge_free(node->bridge);
        node->bridge = NULL;
    }

    node->bridge = ros_bridge_create(&bridge_cfg);
    if (!node->bridge) {
        node->connected = 0;
        node->state = 0;
        snprintf(node->last_error, sizeof(node->last_error),
                 "无法连接到rosbridge: ws://%s:%d", bridge_host, bridge_cfg.bridge_port);
        log_warning("[ROS节点] %s", node->last_error);
        return -1;
    }

    node->connected = 1;
    node->state = 1;
    log_info("[ROS节点] 已通过rosbridge连接到主节点: %s", node->master_uri);
    return 0;
}

int ros_node_disconnect_from_master(RosNode* node) {
    if (!node) return -1;
    if (node->bridge) {
        ros_bridge_free(node->bridge);
        node->bridge = NULL;
    }
    node->connected = 0;
    node->state = 0;
    return 0;
}

int ros_node_is_connected(const RosNode* node) {
    return node ? node->connected : 0;
}

int ros_node_get_state(const RosNode* node) {
    return node ? node->state : -1;
}

int ros_node_advertise(RosNode* node, const char* topic, const char* type,
                        const char* md5sum) {
    if (!node || !topic || !type) return -1;
    if (node->num_publishers >= ROS_NODE_MAX_PUBLISHERS) return -1;

    /* 通过rosbridge广告话题 */
    if (node->bridge && node->connected) {
        char json_buf[512];
        snprintf(json_buf, sizeof(json_buf),
                 "{\"op\":\"advertise\",\"topic\":\"%s\",\"type\":\"%s\"}",
                 topic, type);
        ros_bridge_publish(node->bridge, "/rosbridge/advertise", "rosbridge_msgs/Advertise", json_buf);
    }

    int idx = node->num_publishers;
    snprintf(node->publishers[idx].topic, sizeof(node->publishers[idx].topic), "%s", topic);
    snprintf(node->publishers[idx].type, sizeof(node->publishers[idx].type), "%s", type);
    node->publishers[idx].active = 1;
    node->num_publishers++;

    (void)md5sum;
    log_debug("[ROS节点] 广告话题: %s [%s]", topic, type);
    return 0;
}

int ros_node_unadvertise(RosNode* node, const char* topic) {
    if (!node || !topic) return -1;

    /* 通过rosbridge取消广告 */
    if (node->bridge && node->connected) {
        char json_buf[512];
        snprintf(json_buf, sizeof(json_buf),
                 "{\"op\":\"unadvertise\",\"topic\":\"%s\"}", topic);
        ros_bridge_publish(node->bridge, "/rosbridge/unadvertise", "rosbridge_msgs/Unadvertise", json_buf);
    }

    for (int i = 0; i < node->num_publishers; i++) {
        if (strcmp(node->publishers[i].topic, topic) == 0) {
            node->publishers[i].active = 0;
            return 0;
        }
    }
    return 0;
}

int ros_node_subscribe(RosNode* node, const char* topic, const char* type,
                        RosMessageCallback callback, void* user_data) {
    if (!node || !topic || !type) return -1;
    if (node->num_subscribers >= ROS_NODE_MAX_SUBSCRIBERS) return -1;

    /* 通过rosbridge订阅话题 */
    if (node->bridge && node->connected) {
        ros_bridge_subscribe(node->bridge, topic, type,
                             (RosTopicCallback)callback, user_data);
    }

    int idx = node->num_subscribers;
    snprintf(node->subscribers[idx].topic, sizeof(node->subscribers[idx].topic), "%s", topic);
    snprintf(node->subscribers[idx].type, sizeof(node->subscribers[idx].type), "%s", type);
    node->subscribers[idx].callback = callback;
    node->subscribers[idx].user_data = user_data;
    node->subscribers[idx].active = 1;
    node->num_subscribers++;

    log_debug("[ROS节点] 订阅话题: %s [%s]", topic, type);
    return 0;
}

int ros_node_unsubscribe(RosNode* node, const char* topic) {
    if (!node || !topic) return -1;

    if (node->bridge && node->connected) {
        ros_bridge_unsubscribe(node->bridge, topic);
    }

    for (int i = 0; i < node->num_subscribers; i++) {
        if (strcmp(node->subscribers[i].topic, topic) == 0) {
            node->subscribers[i].active = 0;
            return 0;
        }
    }
    return 0;
}

int ros_node_publish(RosNode* node, const char* topic,
                      const void* msg, size_t msg_size) {
    if (!node || !topic || !msg || msg_size == 0) return -1;

    if (node->bridge && node->connected) {
        /* 构建JSON格式消息 */
        char json_buf[4096];
        size_t json_len = 0;

        /* 将二进制消息转换为JSON格式发布 */
        const uint8_t* data = (const uint8_t*)msg;
        json_len = (size_t)snprintf(json_buf, sizeof(json_buf),
                   "{\"op\":\"publish\",\"topic\":\"%s\",\"msg\":{\"data\":[", topic);

        for (size_t i = 0; i < msg_size && json_len < sizeof(json_buf) - 10; i++) {
            if (i > 0) {
                json_len += (size_t)snprintf(json_buf + json_len, sizeof(json_buf) - json_len, ",");
            }
            json_len += (size_t)snprintf(json_buf + json_len, sizeof(json_buf) - json_len,
                       "%u", (unsigned int)data[i]);
        }
        snprintf(json_buf + json_len, sizeof(json_buf) - json_len, "]}}");

        return ros_bridge_publish(node->bridge, topic, "std_msgs/UInt8MultiArray", json_buf);
    }

    log_debug("[ROS节点] 未连接，发布到 %s 失败（%zu字节）", topic, msg_size);
    return -1;
}

int ros_node_advertise_service(RosNode* node, const char* service_name,
                                RosServiceCallback callback, void* user_data) {
    if (!node || !service_name) return -1;
    if (node->num_services >= ROS_NODE_MAX_SERVICES) return -1;

    /* 通过rosbridge广告服务 */
    if (node->bridge && node->connected) {
        char json_buf[512];
        snprintf(json_buf, sizeof(json_buf),
                 "{\"op\":\"advertise_service\",\"type\":\"%s\",\"service\":\"%s\"}",
                 "selflnn_msgs/Service", service_name);
        ros_bridge_publish(node->bridge, "/rosbridge/advertise_service",
                          "rosbridge_msgs/AdvertiseService", json_buf);
    }

    int idx = node->num_services;
    snprintf(node->services[idx].name, sizeof(node->services[idx].name), "%s", service_name);
    node->services[idx].callback = callback;
    node->services[idx].user_data = user_data;
    node->services[idx].active = 1;
    node->num_services++;

    return 0;
}

int ros_node_unadvertise_service(RosNode* node, const char* service_name) {
    if (!node || !service_name) return -1;

    if (node->bridge && node->connected) {
        char json_buf[512];
        snprintf(json_buf, sizeof(json_buf),
                 "{\"op\":\"unadvertise_service\",\"service\":\"%s\"}", service_name);
        ros_bridge_publish(node->bridge, "/rosbridge/unadvertise_service",
                          "rosbridge_msgs/UnadvertiseService", json_buf);
    }

    for (int i = 0; i < node->num_services; i++) {
        if (strcmp(node->services[i].name, service_name) == 0) {
            node->services[i].active = 0;
            return 0;
        }
    }
    return 0;
}

int ros_node_spin(RosNode* node, int timeout_ms) {
    if (!node) return -1;

    int processed = 0;
    int remaining = timeout_ms > 0 ? timeout_ms : 100;
    int slice_ms = remaining > 50 ? 50 : remaining;

    while (remaining > 0) {
        int ret = ros_node_spin_once(node);
        if (ret > 0) processed += ret;
        if (ret < 0) break;

#ifdef _WIN32
        Sleep((DWORD)slice_ms);
#else
        struct timespec ts;
        ts.tv_sec = slice_ms / 1000;
        ts.tv_nsec = (slice_ms % 1000) * 1000000L;
        nanosleep(&ts, NULL);
#endif
        remaining -= slice_ms;
        if (slice_ms > remaining) slice_ms = remaining;
    }

    return processed;
}

int ros_node_spin_once(RosNode* node) {
    if (!node) return -1;
    if (!node->bridge || !node->connected) return 0;

    return ros_bridge_spin_once(node->bridge, 0);
}

void ros_node_stop_spin(RosNode* node) {
    if (!node) return;
    node->state = 0;
}

int ros_node_get_published_topics(RosNode* node, RosTopicInfo* topics, int* count) {
    if (!node || !count) return -1;

    int actual = 0;
    for (int i = 0; i < node->num_publishers; i++) {
        if (node->publishers[i].active) {
            if (topics && actual < *count) {
                snprintf(topics[actual].topic_name, sizeof(topics[actual].topic_name),
                         "%s", node->publishers[i].topic);
                snprintf(topics[actual].type_name, sizeof(topics[actual].type_name),
                         "%s", node->publishers[i].type);
                topics[actual].num_publishers = 1;
                topics[actual].num_subscribers = 0;
                topics[actual].is_service = 0;
                topics[actual].connection_count = 0;
            }
            actual++;
        }
    }
    *count = actual;
    return 0;
}

int ros_node_get_subscribed_topics(RosNode* node, RosTopicInfo* topics, int* count) {
    if (!node || !count) return -1;

    int actual = 0;
    for (int i = 0; i < node->num_subscribers; i++) {
        if (node->subscribers[i].active) {
            if (topics && actual < *count) {
                snprintf(topics[actual].topic_name, sizeof(topics[actual].topic_name),
                         "%s", node->subscribers[i].topic);
                snprintf(topics[actual].type_name, sizeof(topics[actual].type_name),
                         "%s", node->subscribers[i].type);
                topics[actual].num_publishers = 0;
                topics[actual].num_subscribers = 1;
                topics[actual].is_service = 0;
                topics[actual].connection_count = 0;
            }
            actual++;
        }
    }
    *count = actual;
    return 0;
}

const char* ros_node_get_name(const RosNode* node) {
    return node ? node->config.node_name : "";
}

const char* ros_node_get_master_uri(const RosNode* node) {
    return node ? node->master_uri : "";
}
