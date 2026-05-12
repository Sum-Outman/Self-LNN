/**
 * @file ros2_node.h
 * @brief ROS2协议支持系统接口
 */

#ifndef SELFLNN_ROS2_NODE_H
#define SELFLNN_ROS2_NODE_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ROS2_MAX_NODES 16
#define ROS2_MAX_PUBLISHERS 32
#define ROS2_MAX_SUBSCRIBERS 32
#define ROS2_MAX_SERVICES 16
#define ROS2_MAX_ACTIONS 8
#define ROS2_MAX_TOPIC_LEN 128
#define ROS2_MAX_TYPE_LEN 64
#define ROS2_MAX_NODE_NAME 64
#define ROS2_MAX_PARAMS 32
#define ROS2_MAX_PARAM_NAME 128
#define ROS2_MAX_PARAM_VALUE_SIZE 256

typedef enum {
    ROS2_NODE_UNCONFIGURED = 0,
    ROS2_NODE_INACTIVE = 1,
    ROS2_NODE_ACTIVE = 2,
    ROS2_NODE_FINALIZED = 3,
    ROS2_NODE_ERROR = 4
} ROS2NodeState;

typedef enum {
    ROS2_QOS_RELIABLE = 0,
    ROS2_QOS_BEST_EFFORT = 1,
    ROS2_QOS_KEEP_LAST = 0,
    ROS2_QOS_KEEP_ALL = 1
} ROS2QoSPolicy;

typedef struct {
    int depth;
    int reliability;
    int durability;
    int history;
    int lifespan_ms;
    int deadline_ms;
} ROS2QoSConfig;

typedef struct {
    char topic[ROS2_MAX_TOPIC_LEN];
    char type[ROS2_MAX_TYPE_LEN];
    ROS2QoSConfig qos;
    int publisher_id;
    int messages_sent;
    time_t last_publish;
} ROS2Publisher;

typedef struct {
    char topic[ROS2_MAX_TOPIC_LEN];
    char type[ROS2_MAX_TYPE_LEN];
    ROS2QoSConfig qos;
    int subscriber_id;
    int messages_received;
    time_t last_receive;
    void (*callback)(const void* data, size_t size, void* user_data);
    void* user_data;
} ROS2Subscriber;

typedef struct {
    char service_name[ROS2_MAX_TOPIC_LEN];
    char request_type[ROS2_MAX_TYPE_LEN];
    char response_type[ROS2_MAX_TYPE_LEN];
    int service_id;
    int requests_handled;
    int (*handler)(const void* request, size_t req_size, void* response, size_t* resp_size, void* user_data);
    void* user_data;
} ROS2Service;

typedef struct {
    char action_name[ROS2_MAX_TOPIC_LEN];
    int action_id;
    int active;
    float progress;
    int result_ready;
    void* goal_handle;
} ROS2Action;

typedef struct {
    char name[ROS2_MAX_PARAM_NAME];
    char type_str[ROS2_MAX_TYPE_LEN];
    char value[ROS2_MAX_PARAM_VALUE_SIZE];
    size_t value_size;
    int has_default;
} ROS2Param;

typedef struct {
    char node_name[ROS2_MAX_NODE_NAME];
    char namespace_[128];
    int domain_id;
    ROS2NodeState state;
    ROS2Publisher publishers[ROS2_MAX_PUBLISHERS];
    int publisher_count;
    ROS2Subscriber subscribers[ROS2_MAX_SUBSCRIBERS];
    int subscriber_count;
    ROS2Service services[ROS2_MAX_SERVICES];
    int service_count;
    ROS2Action actions[ROS2_MAX_ACTIONS];
    int action_count;
    int initialized;
    /* 参数管理 */
    ROS2Param params[ROS2_MAX_PARAMS];
    int param_count;
    void (*param_callback)(const char*, const void*, size_t, void*);
    void* param_user_data;
} ROS2Node;

typedef struct ROS2Manager ROS2Manager;

ROS2Manager* ros2_manager_create(void);
void ros2_manager_free(ROS2Manager* rm);

/* 节点生命周期 */
int ros2_create_node(ROS2Manager* rm, const char* name, const char* ns, int* node_id);
int ros2_configure_node(ROS2Manager* rm, int node_id);
int ros2_activate_node(ROS2Manager* rm, int node_id);
int ros2_deactivate_node(ROS2Manager* rm, int node_id);
int ros2_destroy_node(ROS2Manager* rm, int node_id);
int ros2_get_node_state(const ROS2Manager* rm, int node_id, ROS2NodeState* state);

/* DDS通信 */
int ros2_create_publisher(ROS2Manager* rm, int node_id, const char* topic, const char* type, const ROS2QoSConfig* qos, int* pub_id);
int ros2_create_subscriber(ROS2Manager* rm, int node_id, const char* topic, const char* type, const ROS2QoSConfig* qos, void (*cb)(const void*,size_t,void*), void* user_data, int* sub_id);
int ros2_publish(ROS2Manager* rm, int node_id, int pub_id, const void* data, size_t size);
int ros2_receive(ROS2Manager* rm, int node_id, int sub_id, void* buffer, size_t* size);

/* Service/Action */
int ros2_create_service(ROS2Manager* rm, int node_id, const char* name, const char* req_type, const char* resp_type, int (*handler)(const void*,size_t,void*,size_t*,void*), void* user_data, int* srv_id);
int ros2_call_service(ROS2Manager* rm, int node_id, const char* service_name, const void* request, size_t req_size, void* response, size_t* resp_size);
int ros2_create_action(ROS2Manager* rm, int node_id, const char* name, int* action_id);
int ros2_send_goal(ROS2Manager* rm, int node_id, int action_id, const void* goal, size_t goal_size);
int ros2_get_result(ROS2Manager* rm, int node_id, int action_id, void* result, size_t* result_size);

/* ROS1桥接 */
int ros2_bridge_to_ros1(ROS2Manager* rm, const char* ros2_topic, const char* ros1_topic);
int ros2_bridge_from_ros1(ROS2Manager* rm, const char* ros1_topic, const char* ros2_topic);

#ifdef __cplusplus
}
#endif
#endif
