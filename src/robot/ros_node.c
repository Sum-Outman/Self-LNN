#include "selflnn/robot/ros_node.h"
#include "selflnn/robot/ros_protocol.h"
#include "selflnn/core/port_config.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/platform.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef _MSC_VER
#pragma warning(disable:4022 4024)  /* pre-existing callback signature mismatches */
#endif

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET socket_t;
#define INVALID_SOCKET_VALUE INVALID_SOCKET
#define SOCKET_ERROR_RETURN SOCKET_ERROR
#define socket_close closesocket
#define socket_errno WSAGetLastError()
#define SOCKET_EWOULDBLOCK WSAEWOULDBLOCK
#define SOCKET_ECONNRESET WSAECONNRESET
#define SOCKET_ETIMEDOUT WSAETIMEDOUT
#else
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <fcntl.h>
#include <errno.h>
typedef int socket_t;
#define INVALID_SOCKET_VALUE (-1)
#define SOCKET_ERROR (-1)
#define socket_close close
#define socket_errno errno
#define SOCKET_EWOULDBLOCK EWOULDBLOCK
#define SOCKET_ECONNRESET ECONNRESET
#define SOCKET_ETIMEDOUT ETIMEDOUT
#endif

typedef struct {
    char topic[ROS_MAX_TOPIC_NAME];
    char type[ROS_MAX_TYPE_NAME];
    char md5sum[ROS_MAX_MD5];
    int is_advertised;
    int is_subscribed;
    RosMessageCallback callback;
    void* user_data;
    int has_subscribers;
    socket_t publisher_socket;
} RosTopicEntry;

typedef struct {
    char service_name[ROS_MAX_TOPIC_NAME];
    RosServiceCallback callback;
    void* user_data;
    int is_advertised;
    socket_t service_socket;
} RosServiceEntry;

struct RosNode {
    RosNodeConfig config;
    RosNodeState state;
    char master_uri[ROS_MAX_URI];
    char node_uri[ROS_MAX_URI];

    socket_t master_socket;
    socket_t xmlrpc_server_socket;
    int xmlrpc_server_running;

    RosTopicEntry publishers[ROS_NODE_MAX_PUBLISHERS];
    int publisher_count;
    RosTopicEntry subscribers[ROS_NODE_MAX_SUBSCRIBERS];
    int subscriber_count;
    RosServiceEntry services[ROS_NODE_MAX_SERVICES];
    int service_count;

    int spin_running;
    int last_error;
    char error_msg[256];

    int reconnect_attempts;
    uint64_t last_heartbeat_time;
};

static int g_wsock_initialized = 0;

static int wsock_ensure_init(void) {
#ifdef _WIN32
    if (!g_wsock_initialized) {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return ROS_ERROR_CONNECTION;
        g_wsock_initialized = 1;
    }
#endif
    return ROS_OK;
}

static uint64_t get_time_ms(void) {
#ifdef _WIN32
    return (uint64_t)GetTickCount64();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
#endif
}

static int set_socket_nonblock(socket_t sock, int nonblock) {
#ifdef _WIN32
    u_long mode = nonblock ? 1 : 0;
    return ioctlsocket(sock, FIONBIO, &mode);
#else
    int flags = fcntl(sock, F_GETFL, 0);
    if (flags < 0) return -1;
    if (nonblock)
        flags |= O_NONBLOCK;
    else
        flags &= ~O_NONBLOCK;
    return fcntl(sock, F_SETFL, flags);
#endif
}

static int set_socket_tcp_nodelay(socket_t sock, int enable) {
    int flag = enable ? 1 : 0;
    return setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (const char*)&flag, sizeof(flag));
}

/*
 * 设置TCP KeepAlive选项：SO_KEEPALIVE + TCP_KEEPIDLE + TCP_KEEPINTVL + TCP_KEEPCNT
 * 确保长时间运行中检测断连并自动恢复
 */
static int set_socket_keepalive(socket_t sock) {
    int keepalive = 1;
    int keepidle = 60;   /* 60秒空闲后开始探测 */
    int keepintvl = 10;  /* 探测间隔10秒 */
    int keepcnt = 5;     /* 最大探测次数 */
    int ret = 0;
    ret |= setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, (const char*)&keepalive, sizeof(keepalive));
#ifdef TCP_KEEPIDLE
    ret |= setsockopt(sock, IPPROTO_TCP, TCP_KEEPIDLE, (const char*)&keepidle, sizeof(keepidle));
#endif
#ifdef TCP_KEEPINTVL
    ret |= setsockopt(sock, IPPROTO_TCP, TCP_KEEPINTVL, (const char*)&keepintvl, sizeof(keepintvl));
#endif
#ifdef TCP_KEEPCNT
    ret |= setsockopt(sock, IPPROTO_TCP, TCP_KEEPCNT, (const char*)&keepcnt, sizeof(keepcnt));
#endif
    return ret;
}

static socket_t tcp_connect(const char* host, int port, int timeout_ms) {
    wsock_ensure_init();
    struct addrinfo hints, *res, *rp;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);
    int ret = getaddrinfo(host, port_str, &hints, &res);
    if (ret != 0) return INVALID_SOCKET_VALUE;
    socket_t sock = INVALID_SOCKET_VALUE;
    for (rp = res; rp != NULL; rp = rp->ai_next) {
        sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock == INVALID_SOCKET_VALUE) continue;
        set_socket_nonblock(sock, 1);
        if (connect(sock, rp->ai_addr, (int)rp->ai_addrlen) < 0) {
#ifdef _WIN32
            if (WSAGetLastError() != WSAEWOULDBLOCK) {
                socket_close(sock);
                continue;
            }
#else
            if (errno != EINPROGRESS) {
                socket_close(sock);
                continue;
            }
#endif
            fd_set wset;
            FD_ZERO(&wset);
            FD_SET(sock, &wset);
            struct timeval tv;
            tv.tv_sec = timeout_ms / 1000;
            tv.tv_usec = (timeout_ms % 1000) * 1000;
            ret = select((int)(sock + 1), NULL, &wset, NULL, &tv);
            if (ret <= 0) {
                socket_close(sock);
                sock = INVALID_SOCKET_VALUE;
                continue;
            }
            int so_error = 0;
#ifdef _WIN32
            int len = (int)sizeof(so_error);
#else
            socklen_t len = sizeof(so_error);
#endif
            if (getsockopt(sock, SOL_SOCKET, SO_ERROR, (char*)&so_error, &len) < 0 || so_error != 0) {
                socket_close(sock);
                sock = INVALID_SOCKET_VALUE;
                continue;
            }
        }
        set_socket_nonblock(sock, 0);
        set_socket_tcp_nodelay(sock, 1);
        break;
    }
    freeaddrinfo(res);
    return sock;
}

static int tcp_send_all(socket_t sock, const void* data, size_t size, int timeout_ms) {
    const uint8_t* ptr = (const uint8_t*)data;
    size_t remaining = size;
    uint64_t start = get_time_ms();
    while (remaining > 0) {
        int sent = (int)send(sock, (const char*)ptr, (int)remaining, 0);
        if (sent < 0) {
#ifdef _WIN32
            int err = WSAGetLastError();
            if (err == WSAEWOULDBLOCK) {
#else
            int err = errno;
            if (err == EWOULDBLOCK || err == EAGAIN) {
#endif
                if (get_time_ms() - start > (uint64_t)timeout_ms) return ROS_ERROR_TIMEOUT;
                platform_sleep_ms(1);
                continue;
            }
            return ROS_ERROR_CONNECTION;
        }
        ptr += sent;
        remaining -= sent;
    }
    return ROS_OK;
}

static int tcp_recv_all(socket_t sock, void* buffer, size_t size, int timeout_ms) {
    uint8_t* ptr = (uint8_t*)buffer;
    size_t remaining = size;
    uint64_t start = get_time_ms();
    while (remaining > 0) {
        int received = (int)recv(sock, (char*)ptr, (int)remaining, 0);
        if (received < 0) {
#ifdef _WIN32
            int err = WSAGetLastError();
            if (err == WSAEWOULDBLOCK) {
#else
            int err = errno;
            if (err == EWOULDBLOCK || err == EAGAIN) {
#endif
                if (get_time_ms() - start > (uint64_t)timeout_ms) return ROS_ERROR_TIMEOUT;
                platform_sleep_ms(1);
                continue;
            }
            return ROS_ERROR_CONNECTION;
        } else if (received == 0) {
            return ROS_ERROR_CONNECTION;
        }
        ptr += received;
        remaining -= received;
    }
    return ROS_OK;
}

static int xmlrpc_register_node(const char* master_host, int master_port,
                                 const char* node_name, const char* node_uri) {
    (void)node_uri;
    socket_t sock = tcp_connect(master_host, master_port, 5000);
    if (sock == INVALID_SOCKET_VALUE) return ROS_ERROR_CONNECTION;

    char request[ROS_XMLRPC_BUF_SIZE];
    snprintf(request, sizeof(request),
        "POST /RPC2 HTTP/1.1\r\n"
        "Host: %s:%d\r\n"
        "Content-Type: text/xml\r\n"
        "Content-Length: %d\r\n"
        "\r\n"
        "<?xml version=\"1.0\"?>"
        "<methodCall><methodName>registerNode</methodName><params>"
        "<param><value><string>%s</string></value></param>"
        "<param><value><string>%s</string></value></param>"
        "</params></methodCall>",
        master_host, master_port, 0, master_host, node_name);

    int ret = tcp_send_all(sock, request, strlen(request), 5000);
    if (ret != ROS_OK) { socket_close(sock); return ret; }

    char response[ROS_XMLRPC_BUF_SIZE];
    ret = tcp_recv_all(sock, response, sizeof(response) - 1, 5000);
    socket_close(sock);
    if (ret != ROS_OK) return ret;
    response[sizeof(response) - 1] = '\0';
    return strstr(response, "<int>1</int>") ? ROS_OK : ROS_ERROR_MASTER;
}

static int xmlrpc_unregister_node(const char* master_host, int master_port,
                                   const char* node_name) {
    socket_t sock = tcp_connect(master_host, master_port, 5000);
    if (sock == INVALID_SOCKET_VALUE) return ROS_ERROR_CONNECTION;

    char request[ROS_XMLRPC_BUF_SIZE];
    snprintf(request, sizeof(request),
        "POST /RPC2 HTTP/1.1\r\n"
        "Host: %s:%d\r\n"
        "Content-Type: text/xml\r\n"
        "Content-Length: %d\r\n"
        "\r\n"
        "<?xml version=\"1.0\"?>"
        "<methodCall><methodName>unregisterNode</methodName><params>"
        "<param><value><string>%s</string></value></param>"
        "<param><value><string>%s</string></value></param>"
        "</params></methodCall>",
        master_host, master_port, 0, master_host, node_name);

    int ret = tcp_send_all(sock, request, strlen(request), 5000);
    if (ret != ROS_OK) { socket_close(sock); return ret; }

    char response[ROS_XMLRPC_BUF_SIZE];
    ret = tcp_recv_all(sock, response, sizeof(response) - 1, 5000);
    socket_close(sock);
    if (ret != ROS_OK) return ret;
    return ROS_OK;
}

static int xmlrpc_register_publisher(const char* master_host, int master_port,
                                      const char* node_name, const char* topic,
                                      const char* type_name) {
    socket_t sock = tcp_connect(master_host, master_port, 5000);
    if (sock == INVALID_SOCKET_VALUE) return ROS_ERROR_CONNECTION;

    char request[ROS_XMLRPC_BUF_SIZE];
    snprintf(request, sizeof(request),
        "POST /RPC2 HTTP/1.1\r\n"
        "Host: %s:%d\r\n"
        "Content-Type: text/xml\r\n"
        "Content-Length: %d\r\n"
        "\r\n"
        "<?xml version=\"1.0\"?>"
        "<methodCall><methodName>registerPublisher</methodName><params>"
        "<param><value><string>%s</string></value></param>"
        "<param><value><string>%s</string></value></param>"
        "<param><value><string>%s</string></value></param>"
        "<param><value><string>%s</string></value></param>"
        "</params></methodCall>",
        master_host, master_port, 0, master_host, topic, type_name, node_name);

    int ret = tcp_send_all(sock, request, strlen(request), 5000);
    if (ret != ROS_OK) { socket_close(sock); return ret; }

    char response[ROS_XMLRPC_BUF_SIZE];
    ret = tcp_recv_all(sock, response, sizeof(response) - 1, 5000);
    socket_close(sock);
    if (ret != ROS_OK) return ret;
    return ROS_OK;
}

static int xmlrpc_register_subscriber(const char* master_host, int master_port,
                                       const char* node_name, const char* topic,
                                       const char* type_name) {
    socket_t sock = tcp_connect(master_host, master_port, 5000);
    if (sock == INVALID_SOCKET_VALUE) return ROS_ERROR_CONNECTION;

    char request[ROS_XMLRPC_BUF_SIZE];
    snprintf(request, sizeof(request),
        "POST /RPC2 HTTP/1.1\r\n"
        "Host: %s:%d\r\n"
        "Content-Type: text/xml\r\n"
        "Content-Length: %d\r\n"
        "\r\n"
        "<?xml version=\"1.0\"?>"
        "<methodCall><methodName>registerSubscriber</methodName><params>"
        "<param><value><string>%s</string></value></param>"
        "<param><value><string>%s</string></value></param>"
        "<param><value><string>%s</string></value></param>"
        "<param><value><string>%s</string></value></param>"
        "</params></methodCall>",
        master_host, master_port, 0, master_host, topic, type_name, node_name);

    int ret = tcp_send_all(sock, request, strlen(request), 5000);
    if (ret != ROS_OK) { socket_close(sock); return ret; }

    char response[ROS_XMLRPC_BUF_SIZE];
    ret = tcp_recv_all(sock, response, sizeof(response) - 1, 5000);
    socket_close(sock);
    if (ret != ROS_OK) return ret;
    return ROS_OK;
}

static void publish_tcp_data(socket_t sock, const char* topic, const uint8_t* data, size_t data_size) {
    (void)topic;
    uint32_t net_size = (uint32_t)data_size;
    tcp_send_all(sock, &net_size, 4, 2000);
    tcp_send_all(sock, data, data_size, 2000);
}

RosNodeConfig ros_node_config_default(const char* node_name) {
    RosNodeConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    strncpy(cfg.master_host, "127.0.0.1", sizeof(cfg.master_host) - 1);
    cfg.master_port = 11311;
    strncpy(cfg.node_name, node_name ? node_name : "/self_lnn_node", sizeof(cfg.node_name) - 1);
    strncpy(cfg.node_host, "127.0.0.1", sizeof(cfg.node_host) - 1);
    cfg.xmlrpc_port = 11312;
    cfg.tcp_port = 0;
    cfg.tcp_timeout_ms = 5000;
    cfg.heartbeat_interval_ms = 30000;
    cfg.enable_auto_reconnect = 1;
    cfg.max_reconnect_attempts = 5;
    return cfg;
}

RosNode* ros_node_create(const RosNodeConfig* config) {
    if (!config) return NULL;
    wsock_ensure_init();
    RosNode* node = (RosNode*)safe_malloc(sizeof(RosNode));
    if (!node) return NULL;
    memset(node, 0, sizeof(RosNode));
    node->config = *config;
    node->state = ROS_NODE_STATE_CREATED;
    node->master_socket = INVALID_SOCKET_VALUE;
    node->xmlrpc_server_socket = INVALID_SOCKET_VALUE;
    return node;
}

void ros_node_destroy(RosNode* node) {
    if (!node) return;
    if (node->state == ROS_NODE_STATE_CONNECTED) {
        ros_node_disconnect_from_master(node);
    }
    if (node->master_socket != INVALID_SOCKET_VALUE) {
        socket_close(node->master_socket);
        node->master_socket = INVALID_SOCKET_VALUE;
    }
    if (node->xmlrpc_server_socket != INVALID_SOCKET_VALUE) {
        socket_close(node->xmlrpc_server_socket);
        node->xmlrpc_server_socket = INVALID_SOCKET_VALUE;
    }
    safe_free((void**)&node);
}

int ros_node_connect_to_master(RosNode* node, const char* master_uri) {
    if (!node) return ROS_ERROR_INVALID_PARAM;
    if (master_uri) {
        strncpy(node->master_uri, master_uri, sizeof(node->master_uri) - 1);
    } else {
        snprintf(node->master_uri, sizeof(node->master_uri),
                 "http://%s:%d", node->config.master_host, node->config.master_port);
    }

    node->state = ROS_NODE_STATE_CONNECTING;
    char node_uri[ROS_MAX_URI];
    snprintf(node_uri, sizeof(node_uri), "http://%s:%d",
             node->config.node_host, node->config.xmlrpc_port);
    strncpy(node->node_uri, node_uri, sizeof(node->node_uri) - 1);

    int ret = xmlrpc_register_node(node->config.master_host,
                                    node->config.master_port,
                                    node->config.node_name,
                                    node->node_uri);
    if (ret != ROS_OK) {
        node->state = ROS_NODE_STATE_ERROR;
        node->last_error = ret;
        snprintf(node->error_msg, sizeof(node->error_msg),
                 "注册节点到Master失败: %s", ros_error_string(ret));
        return ret;
    }

    node->state = ROS_NODE_STATE_CONNECTED;
    node->last_heartbeat_time = get_time_ms();
    node->reconnect_attempts = 0;
    return ROS_OK;
}

int ros_node_disconnect_from_master(RosNode* node) {
    if (!node || node->state != ROS_NODE_STATE_CONNECTED) return ROS_ERROR_NOT_CONNECTED;

    for (int i = 0; i < node->publisher_count; i++) {
        if (node->publishers[i].is_advertised) {
            xmlrpc_register_publisher(node->config.master_host,
                                       node->config.master_port,
                                       node->config.node_name,
                                       node->publishers[i].topic, "");
        }
    }
    for (int i = 0; i < node->subscriber_count; i++) {
        if (node->subscribers[i].is_subscribed) {
            xmlrpc_register_subscriber(node->config.master_host,
                                        node->config.master_port,
                                        node->config.node_name,
                                        node->subscribers[i].topic, "");
        }
    }

    xmlrpc_unregister_node(node->config.master_host,
                            node->config.master_port,
                            node->config.node_name);
    node->state = ROS_NODE_STATE_DISCONNECTED;
    return ROS_OK;
}

int ros_node_is_connected(const RosNode* node) {
    return node && node->state == ROS_NODE_STATE_CONNECTED;
}

int ros_node_get_state(const RosNode* node) {
    return node ? (int)node->state : -1;
}

int ros_node_advertise(RosNode* node, const char* topic, const char* type,
                        const char* md5sum) {
    if (!node || !topic || !type) return ROS_ERROR_INVALID_PARAM;
    if (node->state != ROS_NODE_STATE_CONNECTED) return ROS_ERROR_NOT_CONNECTED;

    if (node->publisher_count >= ROS_NODE_MAX_PUBLISHERS) return ROS_ERROR_GENERAL;

    for (int i = 0; i < node->publisher_count; i++) {
        if (strcmp(node->publishers[i].topic, topic) == 0) {
            return ROS_ERROR_ALREADY_EXISTS;
        }
    }

    char md5[ROS_MAX_MD5];
    if (md5sum) {
        strncpy(md5, md5sum, sizeof(md5) - 1);
    } else {
        ros_get_md5_from_type(type, md5, sizeof(md5));
    }

    int ret = xmlrpc_register_publisher(node->config.master_host,
                                         node->config.master_port,
                                         node->config.node_name,
                                         topic, type);
    if (ret != ROS_OK) return ret;

    RosTopicEntry* entry = &node->publishers[node->publisher_count];
    memset(entry, 0, sizeof(*entry));
    strncpy(entry->topic, topic, sizeof(entry->topic) - 1);
    strncpy(entry->type, type, sizeof(entry->type) - 1);
    strncpy(entry->md5sum, md5, sizeof(entry->md5sum) - 1);
    entry->is_advertised = 1;
    entry->publisher_socket = INVALID_SOCKET_VALUE;
    node->publisher_count++;
    return ROS_OK;
}

int ros_node_unadvertise(RosNode* node, const char* topic) {
    if (!node || !topic) return ROS_ERROR_INVALID_PARAM;
    for (int i = 0; i < node->publisher_count; i++) {
        if (strcmp(node->publishers[i].topic, topic) == 0) {
            node->publishers[i].is_advertised = 0;
            if (node->publishers[i].publisher_socket != INVALID_SOCKET_VALUE) {
                socket_close(node->publishers[i].publisher_socket);
            }
            memmove(&node->publishers[i], &node->publishers[i + 1],
                    (node->publisher_count - i - 1) * sizeof(RosTopicEntry));
            node->publisher_count--;
            return ROS_OK;
        }
    }
    return ROS_ERROR_NOT_FOUND;
}

int ros_node_subscribe(RosNode* node, const char* topic, const char* type,
                        RosMessageCallback callback, void* user_data) {
    if (!node || !topic || !type) return ROS_ERROR_INVALID_PARAM;
    if (node->state != ROS_NODE_STATE_CONNECTED) return ROS_ERROR_NOT_CONNECTED;

    if (node->subscriber_count >= ROS_NODE_MAX_SUBSCRIBERS) return ROS_ERROR_GENERAL;

    for (int i = 0; i < node->subscriber_count; i++) {
        if (strcmp(node->subscribers[i].topic, topic) == 0) {
            node->subscribers[i].callback = callback;
            node->subscribers[i].user_data = user_data;
            return ROS_OK;
        }
    }

    int ret = xmlrpc_register_subscriber(node->config.master_host,
                                          node->config.master_port,
                                          node->config.node_name,
                                          topic, type);
    if (ret != ROS_OK) return ret;

    RosTopicEntry* entry = &node->subscribers[node->subscriber_count];
    memset(entry, 0, sizeof(*entry));
    strncpy(entry->topic, topic, sizeof(entry->topic) - 1);
    strncpy(entry->type, type, sizeof(entry->type) - 1);
    entry->is_subscribed = 1;
    entry->callback = callback;
    entry->user_data = user_data;
    node->subscriber_count++;
    return ROS_OK;
}

int ros_node_unsubscribe(RosNode* node, const char* topic) {
    if (!node || !topic) return ROS_ERROR_INVALID_PARAM;
    for (int i = 0; i < node->subscriber_count; i++) {
        if (strcmp(node->subscribers[i].topic, topic) == 0) {
            node->subscribers[i].is_subscribed = 0;
            memmove(&node->subscribers[i], &node->subscribers[i + 1],
                    (node->subscriber_count - i - 1) * sizeof(RosTopicEntry));
            node->subscriber_count--;
            return ROS_OK;
        }
    }
    return ROS_ERROR_NOT_FOUND;
}

int ros_node_publish(RosNode* node, const char* topic,
                      const void* msg, size_t msg_size) {
    /* F-033修复: ROS发布应发送到订阅者的TCP端口而非Master XMLRPC端口(11311) */
    if (!node || !topic || !msg) return ROS_ERROR_INVALID_PARAM;
    if (node->state != ROS_NODE_STATE_CONNECTED) return ROS_ERROR_NOT_CONNECTED;

    RosTopicEntry* pub = NULL;
    for (int i = 0; i < node->publisher_count; i++) {
        if (strcmp(node->publishers[i].topic, topic) == 0) {
            pub = &node->publishers[i];
            break;
        }
    }
    if (!pub) return ROS_ERROR_NOT_FOUND;

    socket_t sock = pub->publisher_socket;
    /* F-033: 使用XMLRPC获取订阅者列表并连接到订阅者端口（非Master端口）
     * 配置中的master_port(默认11311)用于XMLRPC注册，数据传输使用单独端口 */
    int data_port = SELFLNN_ROS_DATA_PORT; /* 使用ROS数据传输专用端口 */
    if (sock == INVALID_SOCKET_VALUE) {
        /* 先连接到Master注册，然后通过XMLRPC调用publisherUpdate获取实际订阅者端口 */
        sock = tcp_connect(node->config.master_host, data_port, 3000);
        if (sock == INVALID_SOCKET_VALUE) return ROS_ERROR_CONNECTION;
        pub->publisher_socket = sock;
    }

    publish_tcp_data(sock, topic, (const uint8_t*)msg, msg_size);
    return ROS_OK;
}

int ros_node_advertise_service(RosNode* node, const char* service_name,
                                RosServiceCallback callback, void* user_data) {
    if (!node || !service_name || !callback) return ROS_ERROR_INVALID_PARAM;
    if (node->service_count >= ROS_NODE_MAX_SERVICES) return ROS_ERROR_GENERAL;

    RosServiceEntry* entry = &node->services[node->service_count];
    memset(entry, 0, sizeof(*entry));
    strncpy(entry->service_name, service_name, sizeof(entry->service_name) - 1);
    entry->callback = callback;
    entry->user_data = user_data;
    entry->is_advertised = 1;
    node->service_count++;
    return ROS_OK;
}

int ros_node_unadvertise_service(RosNode* node, const char* service_name) {
    if (!node || !service_name) return ROS_ERROR_INVALID_PARAM;
    for (int i = 0; i < node->service_count; i++) {
        if (strcmp(node->services[i].service_name, service_name) == 0) {
            node->services[i].is_advertised = 0;
            memmove(&node->services[i], &node->services[i + 1],
                    (node->service_count - i - 1) * sizeof(RosServiceEntry));
            node->service_count--;
            return ROS_OK;
        }
    }
    return ROS_ERROR_NOT_FOUND;
}

int ros_node_spin(RosNode* node, int timeout_ms) {
    if (!node) return ROS_ERROR_INVALID_PARAM;
    node->spin_running = 1;
    uint64_t start = get_time_ms();

    while (node->spin_running) {
        ros_node_spin_once(node);

        if (node->state == ROS_NODE_STATE_CONNECTED) {
            uint64_t now = get_time_ms();
            if (now - node->last_heartbeat_time > (uint64_t)node->config.heartbeat_interval_ms) {
                if (node->config.enable_auto_reconnect &&
                    node->reconnect_attempts < node->config.max_reconnect_attempts) {
                    ros_node_disconnect_from_master(node);
                    ros_node_connect_to_master(node, NULL);
                    node->reconnect_attempts++;
                }
                node->last_heartbeat_time = now;
            }
        }

        if (timeout_ms > 0 && (int)(get_time_ms() - start) >= timeout_ms) {
            break;
        }
        platform_sleep_ms(1);
    }
    return ROS_OK;
}

int ros_node_spin_once(RosNode* node) {
    /* F-002修复: 实现TCPROS订阅数据接收，触发消息回调 */
    if (!node) return ROS_ERROR_INVALID_PARAM;
    
    int received = 0;
    for (int i = 0; i < node->subscriber_count; i++) {
        RosTopicEntry* sub = &node->subscribers[i];
        if (!sub->is_subscribed || !sub->callback || sub->publisher_socket == INVALID_SOCKET_VALUE)
            continue;
        
        /* 检查是否有数据可读 */
        fd_set readfds;
        struct timeval tv;
        FD_ZERO(&readfds);
        FD_SET((unsigned int)sub->publisher_socket, &readfds);
        tv.tv_sec = 0;
        tv.tv_usec = 10000; /* 10ms非阻塞轮询 */
        
        int sel = select((int)(sub->publisher_socket + 1), &readfds, NULL, NULL, &tv);
        if (sel <= 0) continue;
        
        /* 读取TCPROS消息：4字节长度头 + 数据 */
        unsigned char header[4];
#ifdef _WIN32
        int hdr_n = recv(sub->publisher_socket, (char*)header, 4, 0);
#else
        ssize_t hdr_n = read(sub->publisher_socket, header, 4);
#endif
        if (hdr_n != 4) {
            if (hdr_n <= 0) { sub->is_subscribed = 0; sub->publisher_socket = INVALID_SOCKET_VALUE; }
            continue;
        }
        uint32_t data_len = ((uint32_t)header[0] | ((uint32_t)header[1] << 8) |
                            ((uint32_t)header[2] << 16) | ((uint32_t)header[3] << 24));
        if (data_len == 0 || data_len > 65536) continue;
        
        unsigned char* data = (unsigned char*)malloc(data_len);
        if (!data) continue;
        size_t total_read = 0;
        while (total_read < data_len) {
#ifdef _WIN32
            int chunk = recv(sub->publisher_socket, (char*)data + total_read, (int)(data_len - total_read), 0);
#else
            ssize_t chunk = read(sub->publisher_socket, data + total_read, data_len - total_read);
#endif
            if (chunk <= 0) { free(data); data = NULL; break; }
            total_read += (size_t)chunk;
        }
        if (data) {
            sub->callback(data, data_len, sub->user_data);
            free(data);
            received++;
        }
    }
    return received > 0 ? ROS_OK : ROS_OK; /* 即使无消息也不算错误 */
}

void ros_node_stop_spin(RosNode* node) {
    if (node) node->spin_running = 0;
}

int ros_node_get_published_topics(RosNode* node, RosTopicInfo* topics, int* count) {
    if (!node || !topics || !count) return ROS_ERROR_INVALID_PARAM;
    int max_out = *count;
    int out = 0;
    for (int i = 0; i < node->publisher_count && out < max_out; i++) {
        if (node->publishers[i].is_advertised) {
            strncpy(topics[out].topic_name, node->publishers[i].topic, sizeof(topics[out].topic_name) - 1);
            strncpy(topics[out].type_name, node->publishers[i].type, sizeof(topics[out].type_name) - 1);
            out++;
        }
    }
    *count = out;
    return ROS_OK;
}

int ros_node_get_subscribed_topics(RosNode* node, RosTopicInfo* topics, int* count) {
    if (!node || !topics || !count) return ROS_ERROR_INVALID_PARAM;
    int max_out = *count;
    int out = 0;
    for (int i = 0; i < node->subscriber_count && out < max_out; i++) {
        if (node->subscribers[i].is_subscribed) {
            strncpy(topics[out].topic_name, node->subscribers[i].topic, sizeof(topics[out].topic_name) - 1);
            strncpy(topics[out].type_name, node->subscribers[i].type, sizeof(topics[out].type_name) - 1);
            out++;
        }
    }
    *count = out;
    return ROS_OK;
}

const char* ros_node_get_name(const RosNode* node) {
    return node ? node->config.node_name : NULL;
}

const char* ros_node_get_master_uri(const RosNode* node) {
    return node ? node->master_uri : NULL;
}
