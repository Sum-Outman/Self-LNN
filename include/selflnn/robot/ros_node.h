#ifndef SELFLNN_ROS_NODE_H
#define SELFLNN_ROS_NODE_H

#include "selflnn/robot/ros_protocol.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ROS_NODE_MAX_PUBLISHERS   32
#define ROS_NODE_MAX_SUBSCRIBERS  32
#define ROS_NODE_MAX_SERVICES     16
#define ROS_DEFAULT_MASTER_URI    "http://localhost:11311"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4030)
#endif
typedef struct RosNode RosNode;

/* RosMessageCallback 定义在 ros_protocol.h 中（已通过第4行include导入），此处不再重复定义 */

typedef struct {
    char master_host[256];
    int master_port;
    char node_name[ROS_MAX_NODE_NAME];
    char node_host[256];
    int xmlrpc_port;
    int tcp_port;
    int tcp_timeout_ms;
    int heartbeat_interval_ms;
    int enable_auto_reconnect;
    int max_reconnect_attempts;
} RosNodeConfig;

typedef struct {
    char topic_name[ROS_MAX_TOPIC_NAME];
    char type_name[ROS_MAX_TYPE_NAME];
    int num_publishers;
    int num_subscribers;
    int is_service;
    int connection_count;
} RosTopicInfo;

typedef struct {
    char service_name[ROS_MAX_TOPIC_NAME];
    char type_name[ROS_MAX_TYPE_NAME];
    char uri[ROS_MAX_URI];
} RosServiceInfo;

typedef int (*RosServiceCallback)(const void* request, size_t req_size,
                                   void* response, size_t* resp_size,
                                   void* user_data);

RosNodeConfig ros_node_config_default(const char* node_name);
RosNode* ros_node_create(const RosNodeConfig* config);
void ros_node_destroy(RosNode* node);

int ros_node_connect_to_master(RosNode* node, const char* master_uri);
int ros_node_disconnect_from_master(RosNode* node);
int ros_node_is_connected(const RosNode* node);
int ros_node_get_state(const RosNode* node);

int ros_node_advertise(RosNode* node, const char* topic, const char* type,
                        const char* md5sum);
int ros_node_unadvertise(RosNode* node, const char* topic);

int ros_node_subscribe(RosNode* node, const char* topic, const char* type,
                        RosMessageCallback callback, void* user_data);
int ros_node_unsubscribe(RosNode* node, const char* topic);

int ros_node_publish(RosNode* node, const char* topic,
                      const void* msg, size_t msg_size);

int ros_node_advertise_service(RosNode* node, const char* service_name,
                                RosServiceCallback callback, void* user_data);
int ros_node_unadvertise_service(RosNode* node, const char* service_name);

int ros_node_spin(RosNode* node, int timeout_ms);
int ros_node_spin_once(RosNode* node);
void ros_node_stop_spin(RosNode* node);

int ros_node_get_published_topics(RosNode* node, RosTopicInfo* topics, int* count);
int ros_node_get_subscribed_topics(RosNode* node, RosTopicInfo* topics, int* count);
const char* ros_node_get_name(const RosNode* node);
const char* ros_node_get_master_uri(const RosNode* node);

#ifdef __cplusplus
}
#endif

#endif
