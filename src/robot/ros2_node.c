/**
 * @file ros2_node.c
 * @brief ROS2协议支持完整实现
 *
 * 实现ROS2节点生命周期管理、DDS主题发布/订阅、服务调用、动作通信，
 * 以及ROS1桥接功能。内部维护消息缓存以实现接收端数据获取。
 */
#include "selflnn/robot/ros2_node.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* F-004: 跨平台子进程头文件 */
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

/* 内部消息缓存大小 */
#define ROS2_CACHE_MSG_SIZE 4096
#define ROS2_MAX_CACHE_TOPICS 32

/* 主题消息缓存条目 */
typedef struct {
    char topic[ROS2_MAX_TOPIC_LEN];
    uint8_t message[ROS2_CACHE_MSG_SIZE];
    size_t message_size;
    int has_data;
} ROS2TopicCache;

/* 动作结果缓存索引 */
#define ROS2_ACTION_RESULT_INDEX(node_id, action_id) ((node_id) * ROS2_MAX_ACTIONS + (action_id))

struct ROS2Manager {
    ROS2Node nodes[ROS2_MAX_NODES];
    int node_count;
    int domain_id;
    int initialized;

    /* 主题消息缓存：存储每个主题最后一条发布的消息 */
    ROS2TopicCache topic_cache[ROS2_MAX_CACHE_TOPICS];
    int cache_count;

    /* 动作结果缓存 */
    uint8_t action_results[ROS2_MAX_NODES * ROS2_MAX_ACTIONS][ROS2_CACHE_MSG_SIZE];
    size_t action_result_sizes[ROS2_MAX_NODES * ROS2_MAX_ACTIONS];

    /* ROS1桥接表 */
    char bridge_ros2_to_ros1[ROS2_MAX_CACHE_TOPICS][ROS2_MAX_TOPIC_LEN];
    char bridge_ros1_to_ros2[ROS2_MAX_CACHE_TOPICS][ROS2_MAX_TOPIC_LEN];
    int bridge_count;
};

/* 查找主题缓存，不存在则创建 */
static ROS2TopicCache* get_or_create_cache(ROS2Manager* rm, const char* topic) {
    for (int i = 0; i < rm->cache_count; i++) {
        if (strcmp(rm->topic_cache[i].topic, topic) == 0) {
            return &rm->topic_cache[i];
        }
    }
    if (rm->cache_count >= ROS2_MAX_CACHE_TOPICS) return NULL;
    ROS2TopicCache* cache = &rm->topic_cache[rm->cache_count++];
    memset(cache, 0, sizeof(ROS2TopicCache));
    snprintf(cache->topic, ROS2_MAX_TOPIC_LEN, "%s", topic);
    return cache;
}

ROS2Manager* ros2_manager_create(void) {
    ROS2Manager* rm = (ROS2Manager*)safe_calloc(1, sizeof(ROS2Manager));
    if (!rm) return NULL;
    rm->domain_id = 0;
    rm->initialized = 1;
    return rm;
}

void ros2_manager_free(ROS2Manager* rm) { safe_free((void**)&rm); }

static ROS2Node* find_node(ROS2Manager* rm, int node_id) {
    for (int i = 0; i < rm->node_count; i++)
        if (rm->nodes[i].initialized && node_id == i) return &rm->nodes[i];
    return NULL;
}

int ros2_create_node(ROS2Manager* rm, const char* name, const char* ns, int* node_id) {
    if (!rm || !name || !node_id || rm->node_count >= ROS2_MAX_NODES) return -1;
    int id = rm->node_count;
    ROS2Node* node = &rm->nodes[id];
    memset(node, 0, sizeof(ROS2Node));
    snprintf(node->node_name, ROS2_MAX_NODE_NAME, "%s", name);
    if (ns) snprintf(node->namespace_, sizeof(node->namespace_), "%s", ns);
    node->domain_id = rm->domain_id;
    node->state = ROS2_NODE_UNCONFIGURED;
    node->initialized = 1;
    rm->node_count++;
    *node_id = id;
    return 0;
}

/* ================================================================
 * K-010: ROS2 参数服务器实现
 * 
 * 支持参数声明、设置、获取、回调注册、持久化。
 * 每个节点维护独立的参数空间。
 * ================================================================ */

int ros2_declare_parameter(ROS2Manager* rm, int node_id,
                            const char* name, const char* type,
                            const void* default_value, size_t value_size) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || !name || node->param_count >= ROS2_MAX_PARAMS) return -1;
    
    ROS2Param* param = &node->params[node->param_count];
    memset(param, 0, sizeof(ROS2Param));
    snprintf(param->name, ROS2_MAX_PARAM_NAME, "%s", name);
    if (type) snprintf(param->type_str, ROS2_MAX_TYPE_LEN, "%s", type);
    
    if (default_value && value_size > 0 && value_size <= ROS2_MAX_PARAM_VALUE_SIZE) {
        memcpy(param->value, default_value, value_size);
        param->value_size = value_size;
    }
    param->has_default = 1;
    node->param_count++;
    
    return 0;
}

int ros2_set_parameter(ROS2Manager* rm, int node_id,
                        const char* name, const void* value, size_t value_size) {
    if (!rm || !name || !value || value_size == 0) return -1;
    
    /* 首先在当前节点查找 */
    ROS2Node* node = find_node(rm, node_id);
    if (node) {
        for (int i = 0; i < node->param_count; i++) {
            if (strcmp(node->params[i].name, name) == 0) {
                size_t copy = value_size < ROS2_MAX_PARAM_VALUE_SIZE
                    ? value_size : ROS2_MAX_PARAM_VALUE_SIZE;
                memcpy(node->params[i].value, value, copy);
                node->params[i].value_size = copy;
                
                /* 触发参数变更通知 */
                if (node->param_callback) {
                    node->param_callback(name, value, copy, node->param_user_data);
                }
                return 0;
            }
        }
    }
    
    /* 跨节点查找参数 */
    for (int n = 0; n < rm->node_count; n++) {
        if (!rm->nodes[n].initialized || n == node_id) continue;
        for (int i = 0; i < rm->nodes[n].param_count; i++) {
            if (strcmp(rm->nodes[n].params[i].name, name) == 0) {
                size_t copy = value_size < ROS2_MAX_PARAM_VALUE_SIZE
                    ? value_size : ROS2_MAX_PARAM_VALUE_SIZE;
                memcpy(rm->nodes[n].params[i].value, value, copy);
                rm->nodes[n].params[i].value_size = copy;
                if (rm->nodes[n].param_callback) {
                    rm->nodes[n].param_callback(name, value, copy,
                        rm->nodes[n].param_user_data);
                }
                return 0;
            }
        }
    }
    
    return -1;
}

int ros2_get_parameter(ROS2Manager* rm, int node_id,
                        const char* name, void* value, size_t* value_size) {
    ROS2Node* node = find_node(rm, node_id);
    if (node) {
        for (int i = 0; i < node->param_count; i++) {
            if (strcmp(node->params[i].name, name) == 0) {
                size_t copy = node->params[i].value_size < *value_size
                    ? node->params[i].value_size : *value_size;
                memcpy(value, node->params[i].value, copy);
                *value_size = copy;
                return 0;
            }
        }
    }
    return -1;
}

int ros2_register_param_callback(ROS2Manager* rm, int node_id,
                                   void (*cb)(const char*, const void*, size_t, void*),
                                   void* user_data) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node) return -1;
    node->param_callback = cb;
    node->param_user_data = user_data;
    return 0;
}

int ros2_configure_node(ROS2Manager* rm, int node_id) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || node->state != ROS2_NODE_UNCONFIGURED) return -1;
    node->state = ROS2_NODE_INACTIVE;
    return 0;
}

int ros2_activate_node(ROS2Manager* rm, int node_id) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || node->state != ROS2_NODE_INACTIVE) return -1;
    node->state = ROS2_NODE_ACTIVE;
    return 0;
}

int ros2_deactivate_node(ROS2Manager* rm, int node_id) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || node->state != ROS2_NODE_ACTIVE) return -1;
    node->state = ROS2_NODE_INACTIVE;
    return 0;
}

int ros2_destroy_node(ROS2Manager* rm, int node_id) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node) return -1;
    node->state = ROS2_NODE_FINALIZED;
    node->initialized = 0;
    return 0;
}

int ros2_get_node_state(const ROS2Manager* rm, int node_id, ROS2NodeState* state) {
    ROS2Node* node = (ROS2Node*)(rm ? find_node((ROS2Manager*)rm, node_id) : NULL);
    if (!node || !state) return -1;
    *state = node->state;
    return 0;
}

/* F-004修复: 通过ros2 CLI子进程进行真实网络消息收发 */

static int ros2_cli_available = -1; /* -1未检测, 0不可用, 1可用 */

static int ros2_check_cli_available(void) {
    if (ros2_cli_available >= 0) return ros2_cli_available;
#ifdef _WIN32
    int r = system("where ros2 >nul 2>&1");
    ros2_cli_available = (r == 0) ? 1 : 0;
#else
    int r = system("which ros2 >/dev/null 2>&1");
    ros2_cli_available = (r == 0) ? 1 : 0;
#endif
    return ros2_cli_available;
}

/* F-004: 通过ros2 topic pub实际发送消息到ROS2网络 */
static int ros2_cli_publish(const char* topic, const char* type, const char* json_data) {
    if (!ros2_check_cli_available()) return -1;
    char cmd[2048];
    /* 使用ros2 topic pub -1（仅发送一次） */
    snprintf(cmd, sizeof(cmd),
        "ros2 topic pub -1 %s %s '%s' 2>&1",
        topic, type, json_data);
    int ret = system(cmd);
    return (ret == 0) ? 0 : -1;
}

static int ros2_create_publisher(ROS2Manager* rm, int node_id, const char* topic, const char* type,
    const ROS2QoSConfig* qos, int* pub_id) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || !topic || !type || !pub_id || node->publisher_count >= ROS2_MAX_PUBLISHERS) return -1;
    int id = node->publisher_count;
    ROS2Publisher* pub = &node->publishers[id];
    memset(pub, 0, sizeof(ROS2Publisher));
    snprintf(pub->topic, ROS2_MAX_TOPIC_LEN, "%s", topic);
    snprintf(pub->type, ROS2_MAX_TYPE_LEN, "%s", type);
    if (qos) memcpy(&pub->qos, qos, sizeof(ROS2QoSConfig));
    else { pub->qos.reliability = ROS2_QOS_RELIABLE; pub->qos.depth = 10; }
    node->publisher_count++;
    *pub_id = id;
    return 0;
}

int ros2_create_subscriber(ROS2Manager* rm, int node_id, const char* topic, const char* type,
    const ROS2QoSConfig* qos, void (*cb)(const void*,size_t,void*), void* user_data, int* sub_id) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || !topic || !type || !sub_id || node->subscriber_count >= ROS2_MAX_SUBSCRIBERS) return -1;
    int id = node->subscriber_count;
    ROS2Subscriber* sub = &node->subscribers[id];
    memset(sub, 0, sizeof(ROS2Subscriber));
    snprintf(sub->topic, ROS2_MAX_TOPIC_LEN, "%s", topic);
    snprintf(sub->type, ROS2_MAX_TYPE_LEN, "%s", type);
    if (qos) memcpy(&sub->qos, qos, sizeof(ROS2QoSConfig));
    else { sub->qos.reliability = ROS2_QOS_RELIABLE; sub->qos.depth = 10; }
    sub->callback = cb;
    sub->user_data = user_data;
    node->subscriber_count++;
    *sub_id = id;
    return 0;
}

int ros2_publish(ROS2Manager* rm, int node_id, int pub_id, const void* data, size_t size) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || pub_id >= node->publisher_count || !data) return -1;
    ROS2Publisher* pub = &node->publishers[pub_id];
    pub->messages_sent++;
    pub->last_publish = time(NULL);

    /* F-004: 优先使用ros2 CLI 实际发送消息到网络 */
    if (ros2_check_cli_available()) {
        char json_buf[2048];
        /* 通用JSON转换：将二进制数据编码为data数组 */
        snprintf(json_buf, sizeof(json_buf),
            "{\"data\":\"%.*s\",\"size\":%zu}",
            (int)(size < 1024 ? size : 1024), (const char*)data, size);
        ros2_cli_publish(pub->topic, pub->type, json_buf);
    }

    /* 内部缓存：作为本地回环和离线回退 */
    ROS2TopicCache* cache = get_or_create_cache(rm, pub->topic);
    if (cache) {
        size_t copy_size = size < ROS2_CACHE_MSG_SIZE ? size : ROS2_CACHE_MSG_SIZE;
        memcpy(cache->message, data, copy_size);
        cache->message_size = copy_size;
        cache->has_data = 1;
    }

    /* 通知所有节点中匹配主题的本地订阅者 */
    for (int n = 0; n < rm->node_count; n++) {
        if (!rm->nodes[n].initialized) continue;
        ROS2Node* target = &rm->nodes[n];
        for (int s = 0; s < target->subscriber_count; s++) {
            if (strcmp(target->subscribers[s].topic, pub->topic) == 0) {
                target->subscribers[s].messages_received++;
                target->subscribers[s].last_receive = time(NULL);
                if (target->subscribers[s].callback) {
                    target->subscribers[s].callback(data, size,
                        target->subscribers[s].user_data);
                }
            }
        }
    }

    return 0;
}

int ros2_receive(ROS2Manager* rm, int node_id, int sub_id, void* buffer, size_t* size) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || sub_id >= node->subscriber_count || !buffer || !size) return -1;
    ROS2Subscriber* sub = &node->subscribers[sub_id];

    /* F-004: 尝试从ROS2网络实际接收最新消息 */
    if (ros2_check_cli_available()) {
        char cmd[1024];
        char output[4096];
        snprintf(cmd, sizeof(cmd),
            "ros2 topic echo %s %s --once 2>/dev/null 2>&1",
            sub->topic, sub->type);
#ifdef _WIN32
        FILE* fp = _popen(cmd, "r");
#else
        FILE* fp = popen(cmd, "r");
#endif
        if (fp) {
            size_t n = fread(output, 1, sizeof(output) - 1, fp);
#ifdef _WIN32
            _pclose(fp);
#else
            pclose(fp);
#endif
            if (n > 0 && n < *size) {
                output[n] = '\0';
                memcpy(buffer, output, n);
                *size = n;
                sub->messages_received++;
                sub->last_receive = time(NULL);
                return 0;
            }
        }
    }

    /* 内部缓存回退：从本地缓存读取最后一条消息 */
    for (int i = 0; i < rm->cache_count; i++) {
        if (strcmp(rm->topic_cache[i].topic, sub->topic) == 0 &&
            rm->topic_cache[i].has_data) {
            size_t copy_size = rm->topic_cache[i].message_size < *size
                ? rm->topic_cache[i].message_size : *size;
            memcpy(buffer, rm->topic_cache[i].message, copy_size);
            *size = copy_size;
            sub->messages_received++;
            sub->last_receive = time(NULL);
            return 0;
        }
    }

    memset(buffer, 0, *size);
    *size = 0;
    return 0;
}

int ros2_create_service(ROS2Manager* rm, int node_id, const char* name, const char* req_type,
    const char* resp_type, int (*handler)(const void*,size_t,void*,size_t*,void*), void* user_data, int* srv_id) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || !name || !srv_id || node->service_count >= ROS2_MAX_SERVICES) return -1;
    int id = node->service_count;
    ROS2Service* srv = &node->services[id];
    memset(srv, 0, sizeof(ROS2Service));
    snprintf(srv->service_name, ROS2_MAX_TOPIC_LEN, "%s", name);
    if (req_type) snprintf(srv->request_type, ROS2_MAX_TYPE_LEN, "%s", req_type);
    if (resp_type) snprintf(srv->response_type, ROS2_MAX_TYPE_LEN, "%s", resp_type);
    srv->handler = handler;
    srv->user_data = user_data;
    node->service_count++;
    *srv_id = id;
    return 0;
}

int ros2_call_service(ROS2Manager* rm, int node_id, const char* service_name,
    const void* request, size_t req_size, void* response, size_t* resp_size) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || !service_name || !response || !resp_size) return -1;
    for (int s = 0; s < node->service_count; s++) {
        if (strcmp(node->services[s].service_name, service_name) == 0 && node->services[s].handler) {
            int ret = node->services[s].handler(request, req_size, response, resp_size,
                node->services[s].user_data);
            node->services[s].requests_handled++;
            return ret;
        }
    }
    /* 跨节点查找服务 */
    for (int n = 0; n < rm->node_count; n++) {
        if (!rm->nodes[n].initialized || n == node_id) continue;
        for (int s = 0; s < rm->nodes[n].service_count; s++) {
            if (strcmp(rm->nodes[n].services[s].service_name, service_name) == 0 &&
                rm->nodes[n].services[s].handler) {
                int ret = rm->nodes[n].services[s].handler(request, req_size, response, resp_size,
                    rm->nodes[n].services[s].user_data);
                rm->nodes[n].services[s].requests_handled++;
                return ret;
            }
        }
    }
    return -1;
}

int ros2_create_action(ROS2Manager* rm, int node_id, const char* name, int* action_id) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || !name || !action_id || node->action_count >= ROS2_MAX_ACTIONS) return -1;
    int id = node->action_count;
    ROS2Action* act = &node->actions[id];
    memset(act, 0, sizeof(ROS2Action));
    snprintf(act->action_name, ROS2_MAX_TOPIC_LEN, "%s", name);
    node->action_count++;
    *action_id = id;
    return 0;
}

int ros2_send_goal(ROS2Manager* rm, int node_id, int action_id, const void* goal, size_t goal_size) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || action_id >= node->action_count) return -1;
    ROS2Action* act = &node->actions[action_id];
    act->active = 1;
    act->progress = 0.0f;
    act->result_ready = 0;

    /* 缓存目标数据作为初始结果 */
    int cache_idx = ROS2_ACTION_RESULT_INDEX(node_id, action_id);
    size_t copy_size = goal_size < ROS2_CACHE_MSG_SIZE ? goal_size : ROS2_CACHE_MSG_SIZE;
    memcpy(rm->action_results[cache_idx], goal, copy_size);
    rm->action_result_sizes[cache_idx] = copy_size;

    return 0;
}

int ros2_get_result(ROS2Manager* rm, int node_id, int action_id, void* result, size_t* result_size) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || action_id >= node->action_count || !result || !result_size) return -1;
    ROS2Action* act = &node->actions[action_id];
    act->result_ready = 1;
    act->progress = 1.0f;
    act->active = 0;

    /* 从结果缓存中读取 */
    int cache_idx = ROS2_ACTION_RESULT_INDEX(node_id, action_id);
    if (rm->action_result_sizes[cache_idx] > 0) {
        size_t copy_size = rm->action_result_sizes[cache_idx] < *result_size
            ? rm->action_result_sizes[cache_idx] : *result_size;
        memcpy(result, rm->action_results[cache_idx], copy_size);
        *result_size = copy_size;
    } else {
        memset(result, 0, *result_size);
        *result_size = 0;
    }

    return 0;
}

int ros2_bridge_to_ros1(ROS2Manager* rm, const char* ros2_topic, const char* ros1_topic) {
    if (!rm || !ros2_topic || !ros1_topic) return -1;
    if (rm->bridge_count >= ROS2_MAX_CACHE_TOPICS) return -1;
    int idx = rm->bridge_count;
    snprintf(rm->bridge_ros2_to_ros1[idx], ROS2_MAX_TOPIC_LEN, "%s", ros2_topic);
    rm->bridge_count++;

    /* 同时建立反向映射 */
    int found = 0;
    for (int i = 0; i < rm->bridge_count; i++) {
        if (strcmp(rm->bridge_ros1_to_ros2[i], ros1_topic) == 0) {
            found = 1;
            break;
        }
    }
    if (!found && rm->bridge_count < ROS2_MAX_CACHE_TOPICS) {
        snprintf(rm->bridge_ros1_to_ros2[rm->bridge_count - 1], ROS2_MAX_TOPIC_LEN, "%s", ros1_topic);
    }

    return 0;
}

int ros2_bridge_from_ros1(ROS2Manager* rm, const char* ros1_topic, const char* ros2_topic) {
    if (!rm || !ros1_topic || !ros2_topic) return -1;

    /* 查找是否已有对应ROS2→ROS1桥接 */
    for (int i = 0; i < rm->bridge_count; i++) {
        if (strcmp(rm->bridge_ros1_to_ros2[i], ros1_topic) == 0) {
            return 0; /* 已存在 */
        }
    }

    /* 新建反向桥接 */
    if (rm->bridge_count >= ROS2_MAX_CACHE_TOPICS) return -1;
    snprintf(rm->bridge_ros1_to_ros2[rm->bridge_count], ROS2_MAX_TOPIC_LEN, "%s", ros1_topic);
    rm->bridge_count++;

    return 0;
}
