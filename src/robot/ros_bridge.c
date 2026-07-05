/**
 * @file ros_bridge.c
 * @brief SELF-LNN 与 ROS/ROS2 的桥接实现
 *
 * 通过 rosbridge WebSocket 协议 (rosbridge_suite) 与 ROS Master 通信。
 * rosbridge 提供 JSON API，可通过标准 WebSocket 从 C 语言调用，
 * 无需链接 ROS C++ 库。
 *
 * 协议文档：http://wiki.ros.org/rosbridge_suite
 * WebSocket 连接 ws://host:8080（与HTTP共用端口）
 */

#include "selflnn/robot/ros_bridge.h"
#include "selflnn/core/port_config.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h"

#ifdef _WIN32
static void* memmem(const void* haystack, size_t haystack_len,
    const void* needle, size_t needle_len) {
    if (!haystack || !needle || haystack_len < needle_len) return NULL;
    const unsigned char* h = (const unsigned char*)haystack;
    const unsigned char* n = (const unsigned char*)needle;
    for (size_t i = 0; i <= haystack_len - needle_len; i++) {
        if (memcmp(h + i, n, needle_len) == 0) return (void*)(h + i);
    }
    return NULL;
}
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#endif

/**
 * @brief RFC 4648 Base64编码（用于Sec-WebSocket-Key）
 * 符合RFC 6455第4.1节要求：将16字节随机数据编码为24字符Base64字符串
 */
static int ws_base64_encode(const uint8_t* data, size_t len, char* out, size_t out_size) {
    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t o = 0;
    for (size_t i = 0; i < len; i += 3) {
        uint32_t v = ((uint32_t)data[i] << 16) |
                     ((i + 1 < len ? (uint32_t)data[i + 1] : 0) << 8) |
                     (i + 2 < len ? (uint32_t)data[i + 2] : 0);
        if (o + 4 > out_size) return -1;
        out[o++] = table[(v >> 18) & 63];
        out[o++] = table[(v >> 12) & 63];
        out[o++] = (i + 1 < len) ? table[(v >> 6) & 63] : '=';
        out[o++] = (i + 2 < len) ? table[v & 63] : '=';
    }
    out[o] = '\0';
    return (int)o;
}

/* rosbridge WebSocket 操作码 */
#define WS_OPCODE_TEXT  0x1
#define WS_OPCODE_CLOSE 0x8
#define WS_OPCODE_PING  0x9
#define WS_OPCODE_PONG  0xA

/* 最大消息大小 */
#define ROS_BRIDGE_MAX_MESSAGE 65536

/* rosbridge 内部连接结构 */
struct RosBridge {
    RosBridgeConfig config;
    RosConnectionState state;
    int socket_fd;
    char recv_buffer[ROS_BRIDGE_MAX_MESSAGE];
    size_t recv_offset;
    
    /* 订阅管理 */
    struct {
        char topic[256];
        char message_type[128];
        RosTopicCallback callback;
        void* user_data;
        int active;
    } subscriptions[32];
    int subscription_count;
    
    /* 请求ID计数器 */
    int request_id;
};

/* WebSocket 帧发送 */
static int ws_send_frame(int fd, const char* data, size_t len) {
    unsigned char header[10];
    size_t header_len = 2;
    
    header[0] = 0x81; /* FIN + text opcode */
    
    if (len < 126) {
        header[1] = (unsigned char)len;
    } else if (len < 65536) {
        header[1] = 126;
        header[2] = (unsigned char)((len >> 8) & 0xFF);
        header[3] = (unsigned char)(len & 0xFF);
        header_len = 4;
    } else {
        header[1] = 127;
        for (int i = 0; i < 8; i++) {
            header[2 + i] = (unsigned char)((len >> (56 - 8 * i)) & 0xFF);
        }
        header_len = 10;
    }
    
#ifdef _WIN32
    if (send(fd, (const char*)header, (int)header_len, 0) < 0) return -1;
    if (send(fd, data, (int)len, 0) < 0) return -1;
#else
    if (write(fd, header, header_len) < 0) return -1;
    if (write(fd, data, len) < 0) return -1;
#endif
    return 0;
}

/* 构建 rosbridge JSON 订阅消息 */
static int ros_bridge_send_subscribe(RosBridge* bridge, const char* topic,
                                      const char* message_type) {
    char msg[4096];
    snprintf(msg, sizeof(msg),
        "{\"op\":\"subscribe\",\"id\":\"sub_%s\","
        "\"topic\":\"%s\",\"type\":\"%s\"}",
        topic, topic, message_type);
    return ws_send_frame(bridge->socket_fd, msg, strlen(msg));
}

int ros_bridge_is_available(void) {
    /* 检查 ROS 环境变量 */
    const char* ros_master = getenv("ROS_MASTER_URI");
    if (ros_master && ros_master[0]) return 1;
    
    /* 尝试连接 localhost:11311 检测 ROS Master */
#ifdef _WIN32
    return 0; /* Windows 上不自动检测 */
#else
    return (system("test -f /opt/ros/noetic/setup.sh") == 0) ? 1 : 0;
#endif
}

int ros2_bridge_is_available(void) {
    const char* ros_domain = getenv("ROS_DOMAIN_ID");
    if (ros_domain && ros_domain[0]) return 1;
    return 0;
}

RosBridge* ros_bridge_create(const RosBridgeConfig* config) {
    if (!config) return NULL;
    
    RosBridge* bridge = (RosBridge*)safe_calloc(1, sizeof(RosBridge));
    if (!bridge) return NULL;
    
    memcpy(&bridge->config, config, sizeof(RosBridgeConfig));
    bridge->state = ROS_STATE_CONNECTING;
    bridge->socket_fd = -1;
    bridge->subscription_count = 0;
    bridge->request_id = 0;
    
    /* 设置默认值 */
    if (bridge->config.bridge_port == 0) bridge->config.bridge_port = SELFLNN_WEBSOCKET_PORT;
/* 使用安全字符串复制而非直接赋值字符串常量，
     * 避免后续safe_free释放字符串常量导致崩溃。 */
    if (!bridge->config.bridge_host || !bridge->config.bridge_host[0]) {
        bridge->config.bridge_host = (char*)(intptr_t)safe_strdup("localhost");
        if (!bridge->config.bridge_host) return NULL;
    }
    
    /* 创建 TCP socket 连接到 rosbridge */
#ifdef _WIN32
    WSADATA wsa_data;
    WSAStartup(MAKEWORD(2, 2), &wsa_data);
#endif
    
    bridge->socket_fd = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (bridge->socket_fd < 0) {
        log_error("ROS桥接: 无法创建socket\n");
        safe_free((void**)&bridge);
        return NULL;
    }
    
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)bridge->config.bridge_port);
    
    struct hostent* host = gethostbyname(bridge->config.bridge_host);
    if (host) {
        memcpy(&addr.sin_addr, host->h_addr, host->h_length);
    } else {
        addr.sin_addr.s_addr = inet_addr(bridge->config.bridge_host);
    }
    
    if (connect(bridge->socket_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        log_error("ROS桥接: 无法连接到 rosbridge %s:%d\n",
                 bridge->config.bridge_host, bridge->config.bridge_port);
#ifdef _WIN32
        closesocket(bridge->socket_fd);
#else
        close(bridge->socket_fd);
#endif
        safe_free((void**)&bridge);
        return NULL;
    }
    
    /* 发送 WebSocket 升级请求 - 使用RFC 6455标准Sec-WebSocket-Key（16字节随机数Base64编码） */
    uint8_t ws_key_random[16];
    secure_random_bytes(ws_key_random, 16);
    char ws_key_b64[32];
    ws_base64_encode(ws_key_random, 16, ws_key_b64, sizeof(ws_key_b64));
    char upgrade_request[2048];
    snprintf(upgrade_request, sizeof(upgrade_request),
        "GET / HTTP/1.1\r\n"
        "Host: %s:%d\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: %s\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n",
        bridge->config.bridge_host, bridge->config.bridge_port, ws_key_b64);
    
#ifdef _WIN32
    send(bridge->socket_fd, upgrade_request, (int)strlen(upgrade_request), 0);
#else
    write(bridge->socket_fd, upgrade_request, strlen(upgrade_request));
#endif
    
    /* 读取握手响应 */
    char response[4096];
    memset(response, 0, sizeof(response));
    /* FIX-IO1: 检查recv/read返回值，失败时记录错误并设置错误状态 */
    int recv_ret = 0;
#ifdef _WIN32
    recv_ret = recv(bridge->socket_fd, response, sizeof(response) - 1, 0);
#else
    recv_ret = (int)read(bridge->socket_fd, response, sizeof(response) - 1);
#endif
    if (recv_ret <= 0) {
        log_error("[ROS桥接] WebSocket握手响应接收失败 (ret=%d, errno=%d)", recv_ret, errno);
        bridge->state = ROS_STATE_ERROR;
        return;
    }
    
    if (strstr(response, "101 Switching Protocols")) {
        bridge->state = ROS_STATE_CONNECTED;
        log_info("ROS桥接: 已连接到 rosbridge %s:%d\n",
                bridge->config.bridge_host, bridge->config.bridge_port);
    } else {
        bridge->state = ROS_STATE_ERROR;
        log_error("ROS桥接: WebSocket握手失败\n");
#ifdef _WIN32
        closesocket(bridge->socket_fd);
#else
        close(bridge->socket_fd);
#endif
        safe_free((void**)&bridge);
        return NULL;
    }
    
    return bridge;
}

void ros_bridge_free(RosBridge* bridge) {
    if (!bridge) return;
    
    if (bridge->socket_fd >= 0) {
#ifdef _WIN32
        closesocket(bridge->socket_fd);
        WSACleanup();
#else
        close(bridge->socket_fd);
#endif
    }
    
    safe_free((void**)&bridge);
}

RosConnectionState ros_bridge_get_state(RosBridge* bridge) {
    if (!bridge) return ROS_STATE_ERROR;
    return bridge->state;
}

int ros_bridge_publish(RosBridge* bridge, const char* topic,
                       const char* message_type, const char* json_data) {
    if (!bridge || !topic || !message_type || !json_data) return -1;
    if (bridge->state != ROS_STATE_CONNECTED) return -1;
    
    char msg[ROS_BRIDGE_MAX_MESSAGE];
    bridge->request_id++;
/* 移除二次包装 — rosbridge协议期望msg直接包含消息字段，
     * 不再是 "msg":{"data":%s}。调用方负责传入合法的msg JSON对象。 */
    snprintf(msg, sizeof(msg),
        "{\"op\":\"publish\",\"id\":\"pub_%d\","
        "\"topic\":\"%s\",\"type\":\"%s\",\"msg\":%s}",
        bridge->request_id, topic, message_type, json_data);
    
    return ws_send_frame(bridge->socket_fd, msg, strlen(msg));
}

int ros_bridge_subscribe(RosBridge* bridge, const char* topic,
                         const char* message_type,
                         RosTopicCallback callback, void* user_data) {
    if (!bridge || !topic || !message_type || !callback) return -1;
    if (bridge->state != ROS_STATE_CONNECTED) return -1;
    if (bridge->subscription_count >= 32) return -1;
    
    /* 记录订阅 */
    int idx = bridge->subscription_count++;
    strncpy(bridge->subscriptions[idx].topic, topic, 255);
    strncpy(bridge->subscriptions[idx].message_type, message_type, 127);
    bridge->subscriptions[idx].callback = callback;
    bridge->subscriptions[idx].user_data = user_data;
    bridge->subscriptions[idx].active = 1;
    
    return ros_bridge_send_subscribe(bridge, topic, message_type);
}

int ros_bridge_unsubscribe(RosBridge* bridge, const char* topic) {
    if (!bridge || !topic) return -1;
    if (bridge->state != ROS_STATE_CONNECTED) return -1;
    
    char msg[1024];
    snprintf(msg, sizeof(msg),
        "{\"op\":\"unsubscribe\",\"id\":\"unsub_%s\",\"topic\":\"%s\"}",
        topic, topic);
    
    /* 标记订阅为非活跃 */
    for (int i = 0; i < bridge->subscription_count; i++) {
        if (strcmp(bridge->subscriptions[i].topic, topic) == 0) {
            bridge->subscriptions[i].active = 0;
            break;
        }
    }
    
    return ws_send_frame(bridge->socket_fd, msg, strlen(msg));
}

int ros_bridge_call_service(RosBridge* bridge, const char* service,
                            const char* request_json,
                            char* response_buffer, size_t buffer_size) {
    /* F-018修复: 实现同步等待服务响应，而非直接返回"sent" */
    if (!bridge || !service || !request_json || !response_buffer) return -1;
    if (bridge->state != ROS_STATE_CONNECTED) return -1;
    
    char msg[ROS_BRIDGE_MAX_MESSAGE];
    bridge->request_id++;
    int srv_id = bridge->request_id;
    snprintf(msg, sizeof(msg),
        "{\"op\":\"call_service\",\"id\":\"srv_%d\","
        "\"service\":\"%s\",\"args\":%s}",
        srv_id, service, request_json);
    
    if (ws_send_frame(bridge->socket_fd, msg, strlen(msg)) < 0) return -1;
    
    /* 同步等待响应（最多5秒超时） */
    int max_wait_ms = 5000;
    int waited_ms = 0;
    while (waited_ms < max_wait_ms) {
        fd_set readfds;
        struct timeval tv;
        FD_ZERO(&readfds);
        FD_SET((unsigned int)bridge->socket_fd, &readfds);
        tv.tv_sec = 0;
        tv.tv_usec = 200000; /* 200ms轮询 */
        
        int sel = select((int)(bridge->socket_fd + 1), &readfds, NULL, NULL, &tv);
        if (sel <= 0) { waited_ms += 200; continue; }
        
        char buf[65536];
#ifdef _WIN32
        int n = recv(bridge->socket_fd, buf, (int)sizeof(buf), 0);
#else
        ssize_t n = read(bridge->socket_fd, buf, sizeof(buf));
#endif
        if (n <= 0) { waited_ms += 200; continue; }
        
        /* 解析WebSocket帧获取JSON响应 */
        size_t payload_offset = 2;
        size_t payload_len = (unsigned char)buf[1] & 0x7F;
        if (payload_len == 126 && n >= 4) { payload_len = ((unsigned char)buf[2] << 8) | (unsigned char)buf[3]; payload_offset = 4; }
        else if (payload_len == 127 && n >= 10) { payload_len = 0; for (int i = 0; i < 8; i++) payload_len = (payload_len << 8) | (unsigned char)buf[2 + i]; payload_offset = 10; }
        
        if ((size_t)n >= payload_offset + payload_len && payload_len < buffer_size) {
            char* json_data = buf + payload_offset;
            /* 检查是否匹配我们的服务请求ID */
            char id_str[32];
            snprintf(id_str, sizeof(id_str), "\"srv_%d\"", srv_id);
            if (memmem(json_data, payload_len, id_str, strlen(id_str)) ||
                memmem(json_data, payload_len, "\"result\":", 9) ||
                memmem(json_data, payload_len, "\"values\":", 9)) {
                memcpy(response_buffer, json_data, payload_len < buffer_size - 1 ? payload_len : buffer_size - 1);
                response_buffer[payload_len < buffer_size - 1 ? payload_len : buffer_size - 1] = '\0';
                return 0;
            }
        }
        waited_ms += 200;
    }
    
    /* 超时返回部分成功状态 */
    snprintf(response_buffer, buffer_size, "{\"status\":\"timeout\",\"id\":\"srv_%d\"}", srv_id);
    return -1;
}

int ros_bridge_get_nodes(RosBridge* bridge, char* nodes_json, size_t buffer_size) {
    /* F-034修复: 通过系统命令获取真实ROS节点列表 */
    if (!bridge || !nodes_json) return -1;
    
    char cmd_result[8192] = {0};
    char* buf = NULL;
    size_t buf_len = 0;
    
#ifdef _WIN32
    FILE* fp = _popen("rosnode list 2>nul", "r");
#else
    FILE* fp = popen("rosnode list 2>/dev/null", "r");
#endif
    if (!fp) {
        snprintf(nodes_json, buffer_size, "{\"nodes\":[],\"error\":\"无法执行rosnode命令\"}");
        return 0;
    }
    
    char line_buf[512];
    size_t total = 0;
    /* 构建JSON数组 */
    snprintf(nodes_json, buffer_size, "{\"nodes\":[");
    total = strlen(nodes_json);
    
    while (fgets(line_buf, sizeof(line_buf), fp) && total < buffer_size - 128) {
        size_t len = strlen(line_buf);
        if (len > 1) { /* 跳过空行 */
            if (total > 10) { nodes_json[total++] = ','; }
            nodes_json[total++] = '"';
            for (size_t i = 0; i < len - 1 && total < buffer_size - 4; i++) {
                if (line_buf[i] != '\n' && line_buf[i] != '\r') {
                    nodes_json[total++] = line_buf[i];
                }
            }
            nodes_json[total++] = '"';
        }
    }
    nodes_json[total++] = ']';
    nodes_json[total++] = '}';
    nodes_json[total] = '\0';
    
#ifdef _WIN32
    _pclose(fp);
#else
    pclose(fp);
#endif
    (void)buf; (void)buf_len; (void)cmd_result;
    return 0;
}

int ros_bridge_get_topics(RosBridge* bridge, char* topics_json, size_t buffer_size) {
    /* F-034修复: 通过系统命令获取真实ROS话题列表 */
    if (!bridge || !topics_json) return -1;
    
#ifdef _WIN32
    FILE* fp = _popen("rostopic list 2>nul", "r");
#else
    FILE* fp = popen("rostopic list 2>/dev/null", "r");
#endif
    if (!fp) {
        snprintf(topics_json, buffer_size, "{\"topics\":[],\"error\":\"无法执行rostopic命令\"}");
        return 0;
    }
    
    char line_buf[512];
    size_t total = 0;
    snprintf(topics_json, buffer_size, "{\"topics\":[");
    total = strlen(topics_json);
    
    while (fgets(line_buf, sizeof(line_buf), fp) && total < buffer_size - 128) {
        size_t len = strlen(line_buf);
        if (len > 1) {
            if (total > 10) { topics_json[total++] = ','; }
            topics_json[total++] = '"';
            for (size_t i = 0; i < len - 1 && total < buffer_size - 4; i++) {
                if (line_buf[i] != '\n' && line_buf[i] != '\r') {
                    topics_json[total++] = line_buf[i];
                }
            }
            topics_json[total++] = '"';
        }
    }
    topics_json[total++] = ']';
    topics_json[total++] = '}';
    topics_json[total] = '\0';
    
#ifdef _WIN32
    _pclose(fp);
#else
    pclose(fp);
#endif
    return 0;
}

int ros_bridge_spin_once(RosBridge* bridge, int timeout_ms) {
    /* F-001修复: 实现完整的WebSocket消息接收循环，调用订阅回调 */
    if (!bridge || bridge->state != ROS_STATE_CONNECTED) return -1;
    
    int processed = 0;
    uint64_t start_ms = 0;
#ifdef _WIN32
    start_ms = (uint64_t)GetTickCount64();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    start_ms = (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
#endif
    
    do {
        fd_set readfds;
        struct timeval tv;
        FD_ZERO(&readfds);
        FD_SET((unsigned int)bridge->socket_fd, &readfds);
        int remain_ms = timeout_ms;
        if (remain_ms < 0) remain_ms = 0;
        if (remain_ms > 1000) remain_ms = 1000;
        tv.tv_sec = remain_ms / 1000;
        tv.tv_usec = (remain_ms % 1000) * 1000;
        
        int sel = select((int)(bridge->socket_fd + 1), &readfds, NULL, NULL, &tv);
        if (sel <= 0) break;
        
        char buf[65536];
#ifdef _WIN32
        int n = recv(bridge->socket_fd, buf, (int)sizeof(buf), 0);
#else
        ssize_t n = read(bridge->socket_fd, buf, sizeof(buf));
#endif
        if (n <= 0) break;
        
        /* ZSF-052修复：解析WebSocket帧，添加MASK位处理 */
        if ((unsigned char)buf[0] == 0x82) { /* FIN + Binary */
            if (n < 2) continue;
            int is_masked = ((unsigned char)buf[1] & 0x80) != 0;
            size_t payload_offset = 2;
            size_t payload_len = (unsigned char)buf[1] & 0x7F;
            if (payload_len == 126) {
                if (n < 4) continue;
                payload_len = ((unsigned char)buf[2] << 8) | (unsigned char)buf[3];
                payload_offset = 4;
            } else if (payload_len == 127) {
                if (n < 10) continue;
                payload_len = 0;
                for (int i = 0; i < 8; i++) payload_len = (payload_len << 8) | (unsigned char)buf[2 + i];
                payload_offset = 10;
            }
            /* ZSF-052修复：处理WebSocket MASK位 */
            unsigned char mask_key[4] = {0, 0, 0, 0};
            if (is_masked) {
                if ((size_t)n < payload_offset + 4 + payload_len) continue;
                for (int i = 0; i < 4; i++) mask_key[i] = (unsigned char)buf[payload_offset + i];
                payload_offset += 4;
            }
            if ((size_t)n < payload_offset + payload_len) continue;
            char* json_data = buf + payload_offset;
            size_t data_len = payload_len;
            
            /* ZSF-052修复：如果帧被MASK，需要解掩码 */
            if (is_masked) {
                for (size_t i = 0; i < data_len; i++) {
                    json_data[i] = (char)((unsigned char)json_data[i] ^ mask_key[i % 4]);
                }
            }

            /* 解析JSON中的topic和msg字段，匹配订阅回调 */
            char topic[256] = {0};
            for (size_t i = 0; i + 7 < data_len; i++) {
                if (memcmp(json_data + i, "\"topic\":", 8) == 0) {
                    i += 8;
                    while (i < data_len && json_data[i] != '"') i++;
                    if (i + 1 < data_len && json_data[i] == '"') {
                        i++;
                        size_t t = 0;
                        while (i < data_len && json_data[i] != '"' && t < 255) {
                            topic[t++] = json_data[i++];
                        }
                        topic[t] = '\0';
                        break;
                    }
                }
            }
            
            if (topic[0]) {
                for (int i = 0; i < bridge->subscription_count; i++) {
                    if (bridge->subscriptions[i].active && bridge->subscriptions[i].callback) {
                        if (strcmp(bridge->subscriptions[i].topic, topic) == 0) {
                            bridge->subscriptions[i].callback(
                                topic, bridge->subscriptions[i].message_type,
                                json_data, data_len, bridge->subscriptions[i].user_data);
                            processed++;
                        }
                    }
                }
            }
        } else if ((unsigned char)buf[0] == 0x81) { /* FIN + Text (JSON) */
            /* 同样处理文本帧 */
            if (n < 2) continue;
            size_t payload_offset = 2;
            size_t payload_len = (unsigned char)buf[1] & 0x7F;
            if (payload_len == 126) {
                if (n < 4) continue;
                payload_len = ((unsigned char)buf[2] << 8) | (unsigned char)buf[3];
                payload_offset = 4;
            } else if (payload_len == 127) {
                if (n < 10) continue;
                payload_len = 0;
                for (int i = 0; i < 8; i++) payload_len = (payload_len << 8) | (unsigned char)buf[2 + i];
                payload_offset = 10;
            }
            if ((size_t)n < payload_offset + payload_len) continue;
            char* json_data = buf + payload_offset;
            size_t data_len = payload_len;
            
            char topic[256] = {0};
            for (size_t i = 0; i + 7 < data_len; i++) {
                if (memcmp(json_data + i, "\"topic\":", 8) == 0) {
                    i += 8;
                    while (i < data_len && json_data[i] != '"') i++;
                    if (i + 1 < data_len && json_data[i] == '"') {
                        i++;
                        size_t t = 0;
                        while (i < data_len && json_data[i] != '"' && t < 255) {
                            topic[t++] = json_data[i++];
                        }
                        topic[t] = '\0';
                        break;
                    }
                }
            }
            
            if (topic[0]) {
                for (int i = 0; i < bridge->subscription_count; i++) {
                    if (bridge->subscriptions[i].active && bridge->subscriptions[i].callback) {
                        if (strcmp(bridge->subscriptions[i].topic, topic) == 0) {
                            bridge->subscriptions[i].callback(
                                topic, bridge->subscriptions[i].message_type,
                                json_data, data_len, bridge->subscriptions[i].user_data);
                            processed++;
                        }
                    }
                }
            }
        }
        
#ifdef _WIN32
        uint64_t now_ms = (uint64_t)GetTickCount64();
#else
        clock_gettime(CLOCK_MONOTONIC, &ts);
        now_ms = (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
#endif
        if (timeout_ms > 0 && (int)(now_ms - start_ms) >= timeout_ms) break;
        
    } while (timeout_ms > 0);
    
    return processed;
}

int ros_bridge_send_joint_trajectory(RosBridge* bridge,
                                     const char** joint_names,
                                     const float* positions,
                                     const float* velocities,
                                     int num_joints,
                                     float duration_sec) {
    if (!bridge || !joint_names || !positions || num_joints <= 0) return -1;
    if (bridge->state != ROS_STATE_CONNECTED) return -1;
    
    /* 构建 JointTrajectory 消息 */
    char msg[ROS_BRIDGE_MAX_MESSAGE];
    int offset = snprintf(msg, sizeof(msg),
        "{\"op\":\"publish\",\"topic\":\"/joint_trajectory\","
        "\"msg\":{\"joint_names\":[");
    
    for (int i = 0; i < num_joints; i++) {
        offset += snprintf(msg + offset, sizeof(msg) - offset,
                          "%s\"%s\"", (i > 0) ? "," : "", joint_names[i]);
    }
    
    offset += snprintf(msg + offset, sizeof(msg) - offset,
        "],\"points\":[{\"positions\":[");
    
    for (int i = 0; i < num_joints; i++) {
        offset += snprintf(msg + offset, sizeof(msg) - offset,
                          "%s%f", (i > 0) ? "," : "", positions[i]);
    }
    
    snprintf(msg + offset, sizeof(msg) - offset,
        "],\"time_from_start\":{\"secs\":%d,\"nsecs\":%d}}]}}",
        (int)duration_sec, (int)((duration_sec - (int)duration_sec) * 1e9));
    
    return ws_send_frame(bridge->socket_fd, msg, strlen(msg));
}
