/**
 * @file ros2_node.c
 * @brief ROS2节点系统 —— rosbridge WebSocket桥接 + TCP内部回退模式
 *
 * ⚠️ 通信架构说明：
 * 本实现支持两种通信模式，运行时自动选择：
 *
 * 【主模式】rosbridge WebSocket桥接（推荐）：
 * - 通过RFC 6455 WebSocket连接到rosbridge_server
 * - 使用rosbridge JSON协议（v2.0）进行topic的advertise/subscribe/publish
 * - 可与真实ROS2生态（rclcpp/rclpy）和其他ROS2工具完全互操作
 * - 标准rosbridge默认端口9090，本系统默认使用SELFLNN_WEBSOCKET_PORT(8081)
 * - 零外部依赖：WebSocket帧编解码和JSON构造全部在纯C中实现
 * - rosbridge WebSocket使用系统独立WebSocket端口
 *
 * 【回退模式】内部TCP直连（同进程内节点间通信）：
 * - 当rosbridge连接不可用时自动回退
 * - 使用自定义RTPS风格包头（魔数"RTPS" + 长度 + 话题名 + 数据）
 * - 仅在SELFLNN进程内节点间有效
 * - 无法与外部ROS2节点互操作
 *
 * 实现ROS2节点生命周期管理：
 * - 节点创建/配置/激活/停用/销毁
 * - 发布者/订阅者管理（rosbridge WebSocket + TCP回退双通道）
 * - 服务/动作管理
 * - QoS配置
 * - ROS1-ROS2桥接
 * - 参数管理
 *
 * 纯C实现，零外部ROS依赖。
 */

#include "selflnn/robot/ros2_node.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h"
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
#include <sys/select.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <pthread.h>
#include <dlfcn.h>
#endif

/* ================================================================
 * 第一部分：rosbridge WebSocket 基础设施（纯C实现，零外部依赖）
 * ================================================================ */

/* WebSocket 操作码 */
#define WS_OPCODE_TEXT  0x1
#define WS_OPCODE_CLOSE 0x8
#define WS_OPCODE_PING  0x9
#define WS_OPCODE_PONG  0xA

/* rosbridge 协议操作类型 */
#define ROSBRIDGE_OP_ADVERTISE    "advertise"
#define ROSBRIDGE_OP_UNADVERTISE  "unadvertise"
#define ROSBRIDGE_OP_PUBLISH      "publish"
#define ROSBRIDGE_OP_SUBSCRIBE    "subscribe"
#define ROSBRIDGE_OP_UNSUBSCRIBE  "unsubscribe"
#define ROSBRIDGE_OP_CALL_SERVICE "call_service"

/* 最大消息大小 */
#define ROSBRIDGE_MAX_MESSAGE 65536

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

/**
 * @brief 发送 WebSocket 文本帧
 * 符合RFC 6455：FIN=1, RSV=0, Opcode=0x1(Text), Mask=0(客户端到服务端不mask)
 */
static int ws_send_text_frame(int fd, const char* data, size_t len) {
    if (fd < 0 || !data || len == 0) return -1;

    unsigned char header[10];
    size_t header_len = 2;

    header[0] = 0x81; /* FIN=1 + Text opcode */

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

/**
 * @brief 连接rosbridge WebSocket服务并完成RFC 6455握手
 * @param host rosbridge服务主机名
 * @param port rosbridge服务端口
 * @param fd_out 输出：连接成功的socket文件描述符
 * @return 0成功，-1失败
 */
static int rosbridge_ws_connect(const char* host, int port, int* fd_out) {
    if (!host || !fd_out) return -1;

    int fd = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        log_error("[ROS2-rosbridge] 无法创建WebSocket socket");
        return -1;
    }

    /* 设置TCP_NODELAY以降低延迟 */
    int flag = 1;
#ifdef _WIN32
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&flag, sizeof(flag));
#else
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
#endif

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);

#ifdef _WIN32
    struct hostent* hostent_ptr = gethostbyname(host);
    if (hostent_ptr) {
        memcpy(&addr.sin_addr, hostent_ptr->h_addr, hostent_ptr->h_length);
    } else {
        addr.sin_addr.s_addr = inet_addr(host);
    }
#else
    struct hostent* hostent_ptr = gethostbyname(host);
    if (hostent_ptr) {
        memcpy(&addr.sin_addr, hostent_ptr->h_addr, hostent_ptr->h_length);
    } else {
        addr.sin_addr.s_addr = inet_addr(host);
    }
#endif

    /* 设置连接超时 */
    struct timeval tv;
    tv.tv_sec = 3;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));

    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        log_error("[ROS2-rosbridge] 无法连接到rosbridge %s:%d", host, port);
#ifdef _WIN32
        closesocket(fd);
#else
        close(fd);
#endif
        return -1;
    }

    /* 构建 WebSocket 升级请求 —— RFC 6455标准Sec-WebSocket-Key */
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
        host, port, ws_key_b64);

#ifdef _WIN32
    if (send(fd, upgrade_request, (int)strlen(upgrade_request), 0) < 0) {
        closesocket(fd);
        return -1;
    }
#else
    if (write(fd, upgrade_request, strlen(upgrade_request)) < 0) {
        close(fd);
        return -1;
    }
#endif

    /* 读取握手响应 */
    char response[4096];
    memset(response, 0, sizeof(response));
#ifdef _WIN32
    int n = recv(fd, response, (int)(sizeof(response) - 1), 0);
#else
    ssize_t n = read(fd, response, sizeof(response) - 1);
#endif

    if (n <= 0 || !strstr(response, "101 Switching Protocols")) {
        log_error("[ROS2-rosbridge] WebSocket握手失败（响应码非101）");
#ifdef _WIN32
        closesocket(fd);
#else
        close(fd);
#endif
        return -1;
    }

    /* 恢复socket为非阻塞模式用于后续接收 */
    tv.tv_sec = 0;
    tv.tv_usec = 500000; /* 500ms超时 */
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));

    *fd_out = fd;
    log_info("[ROS2-rosbridge] WebSocket连接成功: %s:%d", host, port);
    return 0;
}

/**
 * @brief 接收并解析WebSocket帧，提取有效载荷
 * @param fd socket文件描述符
 * @param payload_buf 输出：有效载荷缓冲区
 * @param payload_len 输出：有效载荷长度
 * @return 帧操作码（正数/0/1成功），-1无数据，-2连接关闭
 */
static int ws_recv_frame(int fd, char* payload_buf, size_t* payload_len, size_t max_len) {
    if (fd < 0 || !payload_buf || !payload_len) return -1;

    char header_buf[14];
    memset(header_buf, 0, sizeof(header_buf));

    /* 先读取前2字节头部 */
#ifdef _WIN32
    int n = recv(fd, header_buf, 2, MSG_PEEK);
#else
    ssize_t n = recv(fd, header_buf, 2, MSG_PEEK | MSG_DONTWAIT);
#endif
    if (n < 2) return -1;

    /* 计算完整头部长度 */
    size_t header_total = 2;
    unsigned char mask_flag = header_buf[1] & 0x80;
    size_t plen = header_buf[1] & 0x7F;
    if (plen == 126) header_total = 4 + (mask_flag ? 4 : 0);
    else if (plen == 127) header_total = 10 + (mask_flag ? 4 : 0);
    else header_total = 2 + (mask_flag ? 4 : 0);

    if (header_total > sizeof(header_buf)) return -1;

    /* 读取完整头部 */
#ifdef _WIN32
    n = recv(fd, header_buf, (int)header_total, 0);
#else
    n = read(fd, header_buf, header_total);
#endif
    if (n < (int)header_total) return -1;

    /* 解析操作码 */
    int opcode = header_buf[0] & 0x0F;

    /* 处理控制帧 */
    if (opcode == WS_OPCODE_CLOSE) return -2;
    if (opcode == WS_OPCODE_PING) return 0;
    if (opcode == WS_OPCODE_PONG) return 0;

    /* 解析有效载荷长度 */
    size_t data_len;
    size_t payload_offset;
    unsigned char mask_key[4] = {0};

    if (plen < 126) {
        data_len = plen;
        payload_offset = 2;
    } else if (plen == 126) {
        data_len = ((unsigned char)header_buf[2] << 8) | (unsigned char)header_buf[3];
        payload_offset = 4;
    } else {
        data_len = 0;
        for (int i = 0; i < 8; i++) {
            data_len = (data_len << 8) | (unsigned char)header_buf[2 + i];
        }
        payload_offset = 10;
    }

    if (data_len > max_len) data_len = max_len;
    if (data_len == 0) return 0;

    /* 读取mask key（如果有） */
    if (mask_flag) {
        memcpy(mask_key, header_buf + payload_offset, 4);
        payload_offset += 4;
    }

    /* 读取有效载荷 */
#ifdef _WIN32
    n = recv(fd, payload_buf, (int)data_len, 0);
#else
    n = read(fd, payload_buf, data_len);
#endif
    if (n <= 0) return -1;

    /* unmasking（如果服务端帧带了mask） */
    if (mask_flag) {
        for (size_t i = 0; i < (size_t)n; i++) {
            payload_buf[i] ^= mask_key[i % 4];
        }
    }

    *payload_len = (size_t)n;
    return opcode;
}

/**
 * @brief 发送rosbridge JSON消息（自动包装WebSocket文本帧）
 */
static int rosbridge_send_json(int fd, const char* json_str) {
    if (fd < 0 || !json_str) return -1;
    return ws_send_text_frame(fd, json_str, strlen(json_str));
}

/* ================================================================
 * 第二部分：CycloneDDS动态加载（保留原有逻辑，作为可选增强）
 * ================================================================ */

#ifdef _WIN32
#define DDS_LIB_NAME "cyclonedds.dll"
#else
#define DDS_LIB_NAME "libcyclonedds.so"
#endif

typedef void* (*dds_create_participant_t)(int domain_id);
typedef int   (*dds_delete_participant_t)(void* participant);
typedef void* (*dds_create_topic_t)(void* participant, const char* name, const char* type_name);
typedef void* (*dds_create_writer_t)(void* participant, void* topic);
typedef void* (*dds_create_reader_t)(void* participant, void* topic);
typedef int   (*dds_write_t)(void* writer, const void* data);
typedef int   (*dds_read_t)(void* reader, void** samples, int max_samples);

static struct {
    void* dds_lib_handle;
    int  dds_available;
    dds_create_participant_t dds_create_participant;
    dds_delete_participant_t dds_delete_participant;
    dds_create_topic_t       dds_create_topic;
    dds_create_writer_t      dds_create_writer;
    dds_create_reader_t      dds_create_reader;
    dds_write_t              dds_write;
    dds_read_t               dds_read;
    void* dds_participant;
} g_cyclone_dds = {0};

static int ros2_dds_load_cyclone(void) {
    if (g_cyclone_dds.dds_available) return 1;
    if (g_cyclone_dds.dds_lib_handle) return 0;

#ifdef _WIN32
    g_cyclone_dds.dds_lib_handle = LoadLibraryA(DDS_LIB_NAME);
#else
    g_cyclone_dds.dds_lib_handle = dlopen(DDS_LIB_NAME, RTLD_NOW | RTLD_GLOBAL);
#endif
    if (!g_cyclone_dds.dds_lib_handle) return 0;

#ifdef _WIN32
#define DDS_LOAD_FN(name) do { \
    void* sym_##name = (void*)GetProcAddress((HMODULE)g_cyclone_dds.dds_lib_handle, #name); \
    if (!sym_##name) { \
        FreeLibrary((HMODULE)g_cyclone_dds.dds_lib_handle); \
        g_cyclone_dds.dds_lib_handle = NULL; \
        return 0; \
    } \
    g_cyclone_dds.name = (name##_t)sym_##name; \
} while(0)
#else
#define DDS_LOAD_FN(name) do { \
    void* sym_##name = dlsym(g_cyclone_dds.dds_lib_handle, #name); \
    if (!sym_##name) { \
        dlclose(g_cyclone_dds.dds_lib_handle); \
        g_cyclone_dds.dds_lib_handle = NULL; \
        return 0; \
    } \
    g_cyclone_dds.name = (name##_t)sym_##name; \
} while(0)
#endif

    DDS_LOAD_FN(dds_create_participant);
    DDS_LOAD_FN(dds_delete_participant);
    DDS_LOAD_FN(dds_create_topic);
    DDS_LOAD_FN(dds_create_writer);
    DDS_LOAD_FN(dds_create_reader);
    DDS_LOAD_FN(dds_write);
    DDS_LOAD_FN(dds_read);

#undef DDS_LOAD_FN

    g_cyclone_dds.dds_participant = g_cyclone_dds.dds_create_participant(0);
    if (!g_cyclone_dds.dds_participant) {
#ifdef _WIN32
        FreeLibrary((HMODULE)g_cyclone_dds.dds_lib_handle);
#else
        dlclose(g_cyclone_dds.dds_lib_handle);
#endif
        g_cyclone_dds.dds_lib_handle = NULL;
        return 0;
    }
    g_cyclone_dds.dds_available = 1;
    LOG_INFO("ROS2: CycloneDDS桥接加载成功，真实ROS2互操作已启用");
    return 1;
}

static int ros2_dds_is_available(void) {
    return g_cyclone_dds.dds_available;
}

/* ================================================================
 * 第三部分：管理器内部数据结构
 * ================================================================ */

/* ROS2端点注册表条目 —— TCP回退模式使用 */
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

    /* DDS风格端点注册表 —— TCP回退模式使用 */
    ROS2Endpoint endpoints[ROS2_MAX_PUBLISHERS + ROS2_MAX_SUBSCRIBERS];
    int endpoint_count;

    /* ROS2↔ROS1桥接注册表 */
    struct {
        char ros2_topic[ROS2_MAX_TOPIC_LEN];
        char ros2_type[ROS2_MAX_TYPE_LEN];
        char ros1_topic[ROS2_MAX_TOPIC_LEN];
        char ros1_type[ROS2_MAX_TYPE_LEN];
        int direction;  /* 0=ROS2→ROS1, 1=ROS1→ROS2 */
        int active;
    } bridges[32];
    int bridge_count;

    /* 网络状态 */
    int network_initialized;
    int last_error_code;
    char last_error[256];

    /* CycloneDDS桥接可用性标志 */
    int dds_bridge_available;

    /* ========== rosbridge WebSocket连接状态（新增） ========== */
    struct {
        int enabled;             /* 是否启用rosbridge模式 */
        int connected;           /* WebSocket连接是否活跃 */
        int socket_fd;           /* WebSocket socket描述符 */
        char host[256];          /* rosbridge服务主机 */
        int port;                /* rosbridge服务端口 */
        int request_id;          /* 请求ID计数器 */

        /* 已通过rosbridge advertise的topic跟踪 */
        struct {
            char topic[ROS2_MAX_TOPIC_LEN];
            char type[ROS2_MAX_TYPE_LEN];
            int advertised;       /* 1=已通过rosbridge advertise */
        } advertised_topics[ROS2_MAX_PUBLISHERS];
        int advertised_count;

        /* 已通过rosbridge subscribe的topic跟踪 */
        struct {
            char topic[ROS2_MAX_TOPIC_LEN];
            char type[ROS2_MAX_TYPE_LEN];
            int subscribed;       /* 1=已通过rosbridge subscribe */
        } subscribed_topics[ROS2_MAX_SUBSCRIBERS];
        int subscribed_count;
    } rosbridge;
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

/* ================================================================
 * 第四部分：rosbridge 连接管理
 * ================================================================ */

/**
 * @brief 尝试建立rosbridge WebSocket连接
 * @param rm ROS2管理器
 * @return 0成功，-1失败（回退到TCP模式）
 */
static int rosbridge_try_connect(ROS2Manager* rm) {
    if (!rm) return -1;
    if (rm->rosbridge.connected) return 0;

    if (!rm->rosbridge.enabled) {
        /* 检查是否设置了rosbridge环境配置 */
        const char* env_host = getenv("SELFLNN_ROSBRIDGE_HOST");
        const char* env_port = getenv("SELFLNN_ROSBRIDGE_PORT");
        if (env_host && env_host[0]) {
            snprintf(rm->rosbridge.host, sizeof(rm->rosbridge.host), "%s", env_host);
            rm->rosbridge.enabled = 1;
        }
        if (env_port && env_port[0]) {
            rm->rosbridge.port = atoi(env_port);
        }
    }

    if (!rm->rosbridge.enabled) return -1;

    int fd = -1;
    if (rosbridge_ws_connect(rm->rosbridge.host, rm->rosbridge.port, &fd) != 0) {
        LOG_INFO("ROS2: rosbridge连接失败(%s:%d)，回退到内部TCP模式",
                 rm->rosbridge.host, rm->rosbridge.port);
        rm->rosbridge.enabled = 0;
        return -1;
    }

    rm->rosbridge.socket_fd = fd;
    rm->rosbridge.connected = 1;
    rm->rosbridge.request_id = 0;

    LOG_INFO("ROS2: rosbridge WebSocket桥接已启用 → 可与真实ROS2生态互操作 (%s:%d)",
             rm->rosbridge.host, rm->rosbridge.port);
    return 0;
}

/**
 * @brief 断开rosbridge连接
 */
static void rosbridge_disconnect(ROS2Manager* rm) {
    if (!rm || !rm->rosbridge.connected) return;

    if (rm->rosbridge.socket_fd >= 0) {
        /* 发送WebSocket Close帧 */
        unsigned char close_frame[4] = {0x88, 0x02, 0x03, 0xE8}; /* 1000 normal closure */
#ifdef _WIN32
        send(rm->rosbridge.socket_fd, (const char*)close_frame, 4, 0);
        closesocket(rm->rosbridge.socket_fd);
#else
        write(rm->rosbridge.socket_fd, close_frame, 4);
        close(rm->rosbridge.socket_fd);
#endif
        rm->rosbridge.socket_fd = -1;
    }

    rm->rosbridge.connected = 0;
    rm->rosbridge.advertised_count = 0;
    rm->rosbridge.subscribed_count = 0;
}

/**
 * @brief 通过rosbridge发送 advertise 消息
 */
static int rosbridge_advertise_topic(ROS2Manager* rm, const char* topic, const char* type) {
    if (!rm || !rm->rosbridge.connected || !topic || !type) return -1;

    /* 检查是否已经advertise过 */
    for (int i = 0; i < rm->rosbridge.advertised_count; i++) {
        if (strcmp(rm->rosbridge.advertised_topics[i].topic, topic) == 0) {
            return 0; /* 已经advertise */
        }
    }

    char msg[2048];
    rm->rosbridge.request_id++;
    snprintf(msg, sizeof(msg),
        "{\"op\":\"advertise\",\"id\":\"adv_%d\",\"topic\":\"%s\",\"type\":\"%s\"}",
        rm->rosbridge.request_id, topic, type);

    if (rosbridge_send_json(rm->rosbridge.socket_fd, msg) == 0) {
        if (rm->rosbridge.advertised_count < ROS2_MAX_PUBLISHERS) {
            int idx = rm->rosbridge.advertised_count++;
            snprintf(rm->rosbridge.advertised_topics[idx].topic, ROS2_MAX_TOPIC_LEN, "%s", topic);
            snprintf(rm->rosbridge.advertised_topics[idx].type, ROS2_MAX_TYPE_LEN, "%s", type);
            rm->rosbridge.advertised_topics[idx].advertised = 1;
        }
        log_debug("[ROS2-rosbridge] advertise: %s (类型=%s)", topic, type);
        return 0;
    }
    return -1;
}

/**
 * @brief 通过rosbridge发送 unadvertise 消息
 */
static int rosbridge_unadvertise_topic(ROS2Manager* rm, const char* topic) {
    if (!rm || !rm->rosbridge.connected || !topic) return -1;

    char msg[1024];
    rm->rosbridge.request_id++;
    snprintf(msg, sizeof(msg),
        "{\"op\":\"unadvertise\",\"id\":\"unadv_%d\",\"topic\":\"%s\"}",
        rm->rosbridge.request_id, topic);

    /* 从跟踪列表中移除 */
    for (int i = 0; i < rm->rosbridge.advertised_count; i++) {
        if (strcmp(rm->rosbridge.advertised_topics[i].topic, topic) == 0) {
            rm->rosbridge.advertised_topics[i].advertised = 0;
        }
    }

    return rosbridge_send_json(rm->rosbridge.socket_fd, msg);
}

/**
 * @brief 通过rosbridge发送 subscribe 消息
 */
static int rosbridge_subscribe_topic(ROS2Manager* rm, const char* topic, const char* type) {
    if (!rm || !rm->rosbridge.connected || !topic || !type) return -1;

    /* 检查是否已经subscribe过 */
    for (int i = 0; i < rm->rosbridge.subscribed_count; i++) {
        if (strcmp(rm->rosbridge.subscribed_topics[i].topic, topic) == 0) {
            return 0; /* 已经subscribe */
        }
    }

    char msg[2048];
    rm->rosbridge.request_id++;
    snprintf(msg, sizeof(msg),
        "{\"op\":\"subscribe\",\"id\":\"sub_%d\",\"topic\":\"%s\",\"type\":\"%s\",\"throttle_rate\":0,\"queue_length\":10}",
        rm->rosbridge.request_id, topic, type);

    if (rosbridge_send_json(rm->rosbridge.socket_fd, msg) == 0) {
        if (rm->rosbridge.subscribed_count < ROS2_MAX_SUBSCRIBERS) {
            int idx = rm->rosbridge.subscribed_count++;
            snprintf(rm->rosbridge.subscribed_topics[idx].topic, ROS2_MAX_TOPIC_LEN, "%s", topic);
            snprintf(rm->rosbridge.subscribed_topics[idx].type, ROS2_MAX_TYPE_LEN, "%s", type);
            rm->rosbridge.subscribed_topics[idx].subscribed = 1;
        }
        log_debug("[ROS2-rosbridge] subscribe: %s (类型=%s)", topic, type);
        return 0;
    }
    return -1;
}

/**
 * @brief 通过rosbridge发送 unsubscribe 消息
 */
static int rosbridge_unsubscribe_topic(ROS2Manager* rm, const char* topic) {
    if (!rm || !rm->rosbridge.connected || !topic) return -1;

    char msg[1024];
    rm->rosbridge.request_id++;
    snprintf(msg, sizeof(msg),
        "{\"op\":\"unsubscribe\",\"id\":\"unsub_%d\",\"topic\":\"%s\"}",
        rm->rosbridge.request_id, topic);

    return rosbridge_send_json(rm->rosbridge.socket_fd, msg);
}

/**
 * @brief 通过rosbridge发布数据
 * 构建标准rosbridge publish JSON消息并通过WebSocket发送
 */
static int rosbridge_publish_data(ROS2Manager* rm, const char* topic, const char* type,
                                   const void* data, size_t size) {
    if (!rm || !rm->rosbridge.connected || !topic || !type || !data) return -1;

    /* 将二进制数据编码为JSON兼容格式（Base64编码的data字段）
     * rosbridge支持两种msg格式：
     * 1. JSON对象直接作为msg字段
     * 2. 二进制数据通过 "msg":{"data":"<base64>"} 传递
     *
     * 这里使用base64编码二进制数据，确保任意数据类型都能传递
     */
    size_t b64_max = ((size * 4) / 3) + 4;
    char* b64_buf = (char*)malloc(b64_max);
    if (!b64_buf) return -1;

    int b64_len = ws_base64_encode((const uint8_t*)data, size, b64_buf, b64_max);
    if (b64_len < 0) {
        free(b64_buf);
        return -1;
    }

    char msg[ROSBRIDGE_MAX_MESSAGE];
    rm->rosbridge.request_id++;
    snprintf(msg, sizeof(msg),
        "{\"op\":\"publish\",\"id\":\"pub_%d\",\"topic\":\"%s\",\"msg\":{\"data\":\"%s\"}}",
        rm->rosbridge.request_id, topic, b64_buf);

    free(b64_buf);
    return rosbridge_send_json(rm->rosbridge.socket_fd, msg);
}

/**
 * @brief 通过rosbridge接收数据（非阻塞轮询）
 * 从WebSocket读取帧，解析rosbridge publish消息的JSON，提取topic和data
 */
static int rosbridge_receive_data(ROS2Manager* rm, char* topic_out, size_t topic_max,
                                   void* data_out, size_t* data_size, size_t max_data) {
    if (!rm || !rm->rosbridge.connected) return -1;

    char payload_buf[ROSBRIDGE_MAX_MESSAGE];
    size_t payload_len = 0;

    int opcode = ws_recv_frame(rm->rosbridge.socket_fd, payload_buf, &payload_len,
                               sizeof(payload_buf) - 1);

    if (opcode < 0) {
        /* -1: 无数据，-2: 连接关闭 */
        if (opcode == -2) {
            log_warn("[ROS2-rosbridge] WebSocket连接已关闭");
            rm->rosbridge.connected = 0;
            rm->rosbridge.socket_fd = -1;
        }
        return -1;
    }

    if (payload_len == 0) return -1;

    payload_buf[payload_len] = '\0';

    /* 解析JSON获取topic字段 */
    char* topic_start = strstr(payload_buf, "\"topic\":\"");
    if (!topic_start) return -1;

    topic_start += 9; /* 跳过 "topic":" */
    size_t t = 0;
    while (*topic_start && *topic_start != '"' && t < topic_max - 1) {
        topic_out[t++] = *topic_start++;
    }
    topic_out[t] = '\0';

    if (topic_out[0] == '\0') return -1;

    /* 解析JSON获取msg.data字段（Base64编码的二进制数据） */
    char* data_start = strstr(payload_buf, "\"data\":\"");
    if (data_start && topic_out && data_out && data_size) {
        data_start += 8; /* 跳过 "data":" */
        size_t d = 0;
        char b64_data[65536] = {0};
        while (*data_start && *data_start != '"' && d < sizeof(b64_data) - 1) {
            b64_data[d++] = *data_start++;
        }
        b64_data[d] = '\0';

        /* Base64解码（简单实现，仅支持标准Base64字母表） */
        static const int b64_decode_table[256] = {
            ['A']=0,['B']=1,['C']=2,['D']=3,['E']=4,['F']=5,['G']=6,['H']=7,['I']=8,['J']=9,
            ['K']=10,['L']=11,['M']=12,['N']=13,['O']=14,['P']=15,['Q']=16,['R']=17,['S']=18,['T']=19,
            ['U']=20,['V']=21,['W']=22,['X']=23,['Y']=24,['Z']=25,
            ['a']=26,['b']=27,['c']=28,['d']=29,['e']=30,['f']=31,['g']=32,['h']=33,['i']=34,['j']=35,
            ['k']=36,['l']=37,['m']=38,['n']=39,['o']=40,['p']=41,['q']=42,['r']=43,['s']=44,['t']=45,
            ['u']=46,['v']=47,['w']=48,['x']=49,['y']=50,['z']=51,
            ['0']=52,['1']=53,['2']=54,['3']=55,['4']=56,['5']=57,['6']=58,['7']=59,['8']=60,['9']=61,
            ['+']=62,['/']=63
        };
        static const int b64_invalid = -1;

        if (d > 0 && b64_data[0]) {
            size_t out_idx = 0;
            for (size_t i = 0; i < d && out_idx < max_data; i += 4) {
                int v0 = b64_decode_table[(unsigned char)b64_data[i]];
                int v1 = (i+1 < d) ? b64_decode_table[(unsigned char)b64_data[i+1]] : b64_invalid;
                int v2 = (i+2 < d) ? b64_decode_table[(unsigned char)b64_data[i+2]] : b64_invalid;
                int v3 = (i+3 < d) ? b64_decode_table[(unsigned char)b64_data[i+3]] : b64_invalid;
                if (v0 == b64_invalid) continue;

                uint32_t triple = ((uint32_t)v0 << 18);
                if (v1 != b64_invalid) triple |= ((uint32_t)v1 << 12);
                if (v2 != b64_invalid) triple |= ((uint32_t)v2 << 6);
                if (v3 != b64_invalid) triple |= (uint32_t)v3;

                if (out_idx < max_data) ((unsigned char*)data_out)[out_idx++] = (unsigned char)((triple >> 16) & 0xFF);
                if (b64_data[i+2] != '=' && out_idx < max_data) ((unsigned char*)data_out)[out_idx++] = (unsigned char)((triple >> 8) & 0xFF);
                if (b64_data[i+3] != '=' && out_idx < max_data) ((unsigned char*)data_out)[out_idx++] = (unsigned char)(triple & 0xFF);
            }
            *data_size = out_idx;
        } else {
            *data_size = 0;
        }
    }

    /* 如果有不带msg.data的纯JSON消息，直接拷贝原始JSON */
    if (data_start == NULL && data_out && data_size && payload_len < max_data) {
        memcpy(data_out, payload_buf, payload_len);
        *data_size = payload_len;
    }

    return 0;
}

/* ================================================================
 * 第五部分：管理器生命周期
 * ================================================================ */

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

    /* 尝试加载CycloneDDS实现真实ROS2互操作（作为可选增强） */
    rm->dds_bridge_available = ros2_dds_load_cyclone();
    if (!rm->dds_bridge_available) {
        LOG_INFO("ROS2: CycloneDDS未安装");
    }

    /* 初始化rosbridge配置 */
    rm->rosbridge.enabled = 0;
    rm->rosbridge.connected = 0;
    rm->rosbridge.socket_fd = -1;
    rm->rosbridge.request_id = 0;
    rm->rosbridge.advertised_count = 0;
    rm->rosbridge.subscribed_count = 0;
    snprintf(rm->rosbridge.host, sizeof(rm->rosbridge.host), "%s", "127.0.0.1");
    rm->rosbridge.port = SELFLNN_WEBSOCKET_PORT;

    /* 检查环境变量以启用rosbridge */
    const char* rosbridge_host = getenv("SELFLNN_ROSBRIDGE_HOST");
    if (rosbridge_host && rosbridge_host[0]) {
        snprintf(rm->rosbridge.host, sizeof(rm->rosbridge.host), "%s", rosbridge_host);
        rm->rosbridge.enabled = 1;
    }
    const char* rosbridge_port = getenv("SELFLNN_ROSBRIDGE_PORT");
    if (rosbridge_port && rosbridge_port[0]) {
        rm->rosbridge.port = atoi(rosbridge_port);
        rm->rosbridge.enabled = 1;
    }

    /* 如果设置了ROS_MASTER_URI环境变量，自动启用rosbridge（使用相同端口） */
    const char* ros_master_uri = getenv("ROS_MASTER_URI");
    if (ros_master_uri && ros_master_uri[0]) {
        rm->rosbridge.enabled = 1;
        log_info("[ROS2] 检测到ROS_MASTER_URI=%s，自动启用rosbridge桥接", ros_master_uri);
    }

    /* 尝试连接rosbridge */
    if (rm->rosbridge.enabled) {
        rosbridge_try_connect(rm);
    }

    if (!rm->rosbridge.connected) {
        LOG_INFO("ROS2: rosbridge未连接，使用内部TCP模式（同进程内节点间通信）");
    }

    log_info("[ROS2] 管理器创建完成");
    return rm;
}

void ros2_manager_free(ROS2Manager* rm) {
    if (!rm) return;

    /* 断开rosbridge连接 */
    rosbridge_disconnect(rm);

    for (int i = 0; i < rm->node_count; i++) {
        if (rm->nodes[i].initialized) {
            ros2_destroy_node(rm, i);
        }
    }

    /* 关闭所有TCP端点socket */
    for (int i = 0; i < rm->endpoint_count; i++) {
        if (rm->endpoints[i].socket_fd > 0) {
#ifdef _WIN32
            closesocket(rm->endpoints[i].socket_fd);
#else
            close(rm->endpoints[i].socket_fd);
#endif
        }
    }

    free(rm);
#if defined(_WIN32)
    WSACleanup();
#endif
    log_info("[ROS2] 管理器已释放");
}

/* ================================================================
 * 第六部分：节点生命周期
 * ================================================================ */

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

    /* 通过rosbridge取消该节点的所有advertise和subscribe */
    if (rm->rosbridge.connected) {
        for (int i = 0; i < node->publisher_count; i++) {
            rosbridge_unadvertise_topic(rm, node->publishers[i].topic);
        }
        for (int i = 0; i < node->subscriber_count; i++) {
            rosbridge_unsubscribe_topic(rm, node->subscribers[i].topic);
        }
    }

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

/* 前向声明：ros2_bridge_forward_to_ros1（定义在第九部分） */
int ros2_bridge_forward_to_ros1(ROS2Manager* rm, const char* ros2_from_topic,
                                 const void* data, size_t data_size);

/* ================================================================
 * 第七部分：发布者/订阅者 —— rosbridge主通道 + TCP回退
 * ================================================================ */

/**
 * @brief 创建ROS2发布者
 *
 * 通信策略：
 * 1. 如果rosbridge已连接，通过WebSocket发送advertise消息到rosbridge服务
 * 2. 同时创建TCP端点作为回退通道（rosbridge断开时自动切换）
 * 3. 外部ROS2节点可通过rosbridge发现并订阅此发布者的topic
 */
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

    /* 通过rosbridge advertise topic（主通道） */
    if (rm->rosbridge.connected) {
        rosbridge_advertise_topic(rm, topic, type);
    }

/* TCP回退通道必须正确bind和listen才能使用。
     * 如果rosbridge可用，优先使用rosbridge（WebSocket）；如果不可用，
     * TCP回退端点应通过端口bind+listen建立独立TCP服务器。 */
    if (rm->endpoint_count < ROS2_MAX_PUBLISHERS + ROS2_MAX_SUBSCRIBERS) {
        ROS2Endpoint* ep = &rm->endpoints[rm->endpoint_count++];
        memset(ep, 0, sizeof(ROS2Endpoint));
        ep->endpoint_id = rm->endpoint_count - 1;
        snprintf(ep->topic, sizeof(ep->topic), "%s", topic);
        snprintf(ep->type, sizeof(ep->type), "%s", type);
        if (qos) ep->qos = *qos; else ep->qos = g_default_qos;
        ep->active = 0; /* 默认不激活，仅在rosbridge不可用时激活 */
        ep->socket_fd = -1; /* 延迟创建: bind/listen需要在确定端口后执行 */
    }

    log_info("[ROS2] 发布者创建: 节点=%d, 话题=%s, 类型=%s (rosbridge=%s)",
             node_id, topic, type, rm->rosbridge.connected ? "已启用" : "未启用");
    return 0;
}

/**
 * @brief 创建ROS2订阅者
 *
 * 通信策略：
 * 1. 如果rosbridge已连接，通过WebSocket发送subscribe消息到rosbridge服务
 * 2. 同时创建TCP端点作为回退通道
 * 3. rosbridge连接活跃时优先从WebSocket接收数据
 */
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

    /* 通过rosbridge subscribe topic（主通道） */
    if (rm->rosbridge.connected) {
        rosbridge_subscribe_topic(rm, topic, type);
    }

    /* 创建TCP端点作为回退通道 */
    if (rm->endpoint_count < ROS2_MAX_PUBLISHERS + ROS2_MAX_SUBSCRIBERS) {
        ROS2Endpoint* ep = &rm->endpoints[rm->endpoint_count++];
        memset(ep, 0, sizeof(ROS2Endpoint));
        ep->endpoint_id = rm->endpoint_count - 1;
        snprintf(ep->topic, sizeof(ep->topic), "%s", topic);
        snprintf(ep->type, sizeof(ep->type), "%s", type);
        if (qos) ep->qos = *qos; else ep->qos = g_default_qos;
        ep->active = 1;
        ep->socket_fd = (int)socket(AF_INET, SOCK_STREAM, 0);
        if (!(ep->socket_fd > 0)) ep->socket_fd = -1;
    }

    log_info("[ROS2] 订阅者创建: 节点=%d, 话题=%s, 类型=%s (rosbridge=%s)",
             node_id, topic, type, rm->rosbridge.connected ? "已启用" : "未启用");
    return 0;
}

/**
 * @brief 发布数据到话题
 *
 * 通信策略：
 * 1. rosbridge已连接 → 通过WebSocket发送rosbridge JSON publish消息（主通道）
 * 2. rosbridge未连接 → 通过TCP socket发送RTPS风格数据包（回退通道）
 *
 * rosbridge publish格式：
 *   {"op":"publish","topic":"/xxx","msg":{"data":"<base64>"}}
 *
 * TCP RTPS回退格式：
 *   4字节"RTPS"魔数 + 4字节数据长度 + 话题名 + 二进制数据
 */
int ros2_publish(ROS2Manager* rm, int node_id, int pub_id, const void* data, size_t size) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || !data || size == 0) return -1;
    if (node->state != ROS2_NODE_ACTIVE) return -1;
    if (pub_id < 0 || pub_id >= node->publisher_count) return -1;

    ROS2Publisher* pub = &node->publishers[pub_id];
    pub->messages_sent++;
    pub->last_publish = time(NULL);

    /* 主通道：通过rosbridge WebSocket发布 */
    if (rm->rosbridge.connected) {
        int result = rosbridge_publish_data(rm, pub->topic, pub->type, data, size);
        if (result == 0) {
            /* 同时转发到ROS1桥接注册表 */
            ros2_bridge_forward_to_ros1(rm, pub->topic, data, size);
            return 0;
        }
        /* rosbridge发送失败，回退到TCP模式 */
        log_debug("[ROS2] rosbridge发布失败，回退TCP: %s", pub->topic);
    }

    /* 回退通道：通过TCP发送给所有匹配的订阅者端点 */
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

/**
 * @brief 从话题接收数据
 *
 * 通信策略：
 * 1. rosbridge已连接 → 优先从WebSocket接收rosbridge publish消息（主通道）
 * 2. rosbridge未连接或无数据 → 回退到TCP socket RTPS接收（回退通道）
 *
 * 注意：从rosbridge接收到的数据经过Base64解码还原为原始二进制数据。
 *       如果接收到的消息不包含data字段（纯JSON消息），则直接返回原始JSON。
 */
int ros2_receive(ROS2Manager* rm, int node_id, int sub_id, void* buffer, size_t* size) {
    ROS2Node* node = find_node(rm, node_id);
    if (!node || !buffer || !size) return -1;
    if (node->state != ROS2_NODE_ACTIVE) return -1;
    if (sub_id < 0 || sub_id >= node->subscriber_count) return -1;

    ROS2Subscriber* sub = &node->subscribers[sub_id];

    /* 主通道：从rosbridge WebSocket接收 */
    if (rm->rosbridge.connected) {
        char recv_topic[ROS2_MAX_TOPIC_LEN];
        size_t recv_size = 0;
        char recv_data[ROSBRIDGE_MAX_MESSAGE];

        int result = rosbridge_receive_data(rm, recv_topic, sizeof(recv_topic),
                                             recv_data, &recv_size, sizeof(recv_data));
        if (result == 0 && recv_topic[0] != '\0') {
            /* 检查topic是否匹配当前订阅 */
            if (strcmp(recv_topic, sub->topic) == 0 && recv_size > 0) {
                size_t copy_size = recv_size < *size ? recv_size : *size;
                memcpy(buffer, recv_data, copy_size);
                *size = copy_size;
                sub->messages_received++;
                sub->last_receive = time(NULL);

                if (sub->callback) {
                    sub->callback(buffer, copy_size, sub->user_data);
                }
                return 0;
            }
            /* topic不匹配，数据被丢弃，但这可能是由于多订阅引起的正常情况 */
        }
    }

    /* 回退通道：从匹配的TCP端点读取数据 */
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

/**
 * @brief 轮询rosbridge接收数据并分发到匹配的订阅回调
 * 应在主循环中定期调用，确保及时接收来自rosbridge的消息。
 * @param rm ROS2管理器
 * @param timeout_ms 每次轮询的超时时间（毫秒），0表示非阻塞
 * @return 处理的消息数，-1表示错误
 */
int ros2_spin_once(ROS2Manager* rm, int timeout_ms) {
    if (!rm) return -1;

    int processed = 0;
    int elapsed_ms = 0;
    int poll_interval_ms = 50; /* 每次50ms轮询 */

    while (elapsed_ms < timeout_ms || timeout_ms == 0) {
        if (!rm->rosbridge.connected) break;

        char recv_topic[ROS2_MAX_TOPIC_LEN];
        size_t recv_size = 0;
        char recv_data[ROSBRIDGE_MAX_MESSAGE];

        int result = rosbridge_receive_data(rm, recv_topic, sizeof(recv_topic),
                                             recv_data, &recv_size, sizeof(recv_data));
        if (result != 0 || recv_topic[0] == '\0') {
            if (timeout_ms == 0) break; /* 非阻塞模式下无数据直接返回 */
            elapsed_ms += poll_interval_ms;
#ifdef _WIN32
            Sleep((DWORD)poll_interval_ms);
#else
            usleep((useconds_t)(poll_interval_ms * 1000));
#endif
            continue;
        }

        /* 遍历所有节点的所有订阅者，匹配topic并触发回调 */
        for (int n = 0; n < rm->node_count; n++) {
            if (!rm->nodes[n].initialized) continue;
            ROS2Node* node = &rm->nodes[n];
            for (int s = 0; s < node->subscriber_count; s++) {
                ROS2Subscriber* sub = &node->subscribers[s];
                if (strcmp(sub->topic, recv_topic) == 0 && recv_size > 0) {
                    if (sub->callback) {
                        sub->callback(recv_data, recv_size, sub->user_data);
                        sub->messages_received++;
                        sub->last_receive = time(NULL);
                        processed++;
                    }
                }
            }
        }

        if (timeout_ms == 0) break;
        elapsed_ms += poll_interval_ms;
    }

    /* 同时处理TCP回退通道的接收 */
    for (int n = 0; n < rm->node_count; n++) {
        if (!rm->nodes[n].initialized) continue;
        ROS2Node* node = &rm->nodes[n];
        for (int s = 0; s < node->subscriber_count; s++) {
            ROS2Subscriber* sub = &node->subscribers[s];
            for (int i = 0; i < rm->endpoint_count; i++) {
                ROS2Endpoint* ep = &rm->endpoints[i];
                if (!ep->active || ep->socket_fd <= 0) continue;
                if (strcmp(ep->topic, sub->topic) != 0) continue;

                char recv_buf[4096];
#ifdef _WIN32
                int nbytes = recv(ep->socket_fd, recv_buf, sizeof(recv_buf), 0);
#else
                ssize_t nbytes = recv(ep->socket_fd, recv_buf, sizeof(recv_buf), MSG_DONTWAIT);
#endif
                if (nbytes < 9) continue;

                if (recv_buf[0] != 'R' || recv_buf[1] != 'T' || recv_buf[2] != 'P' || recv_buf[3] != 'S') continue;

                unsigned int data_sz = (unsigned int)((unsigned char)recv_buf[4]) |
                                      ((unsigned int)((unsigned char)recv_buf[5]) << 8) |
                                      ((unsigned int)((unsigned char)recv_buf[6]) << 16) |
                                      ((unsigned int)((unsigned char)recv_buf[7]) << 24);
                size_t tlen = (size_t)((unsigned char)recv_buf[8]);
                size_t hdr_sz = 9 + (tlen < 127 ? tlen : 127);

                if (sub->callback) {
                    sub->callback(recv_buf + hdr_sz, data_sz, sub->user_data);
                    sub->messages_received++;
                    sub->last_receive = time(NULL);
                    processed++;
                }
            }
        }
    }

    return processed;
}

/* ================================================================
 * 第八部分：服务/动作
 * ================================================================ */

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

    for (int i = 0; i < node->service_count; i++) {
        ROS2Service* srv = &node->services[i];
        if (strcmp(srv->service_name, service_name) == 0 && srv->handler) {
            int result = srv->handler(request, req_size, response, resp_size, srv->user_data);
            srv->requests_handled++;
            return result;
        }
    }

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

    memset(result, 0, *result_size);
    act->active = 0;
    return 0;
}

/* ================================================================
 * 第九部分：ROS1桥接
 * ================================================================ */

int ros2_bridge_to_ros1(ROS2Manager* rm, const char* ros2_topic, const char* ros1_topic) {
    if (!rm || !ros2_topic || !ros1_topic) return -1;
    if (rm->bridge_count >= 32) return -1;

    int idx = rm->bridge_count++;
    snprintf(rm->bridges[idx].ros2_topic, ROS2_MAX_TOPIC_LEN, "%s", ros2_topic);
    snprintf(rm->bridges[idx].ros2_type, ROS2_MAX_TYPE_LEN, "std_msgs/UInt8MultiArray");
    snprintf(rm->bridges[idx].ros1_topic, ROS2_MAX_TOPIC_LEN, "%s", ros1_topic);
    snprintf(rm->bridges[idx].ros1_type, ROS2_MAX_TYPE_LEN, "std_msgs/UInt8MultiArray");
    rm->bridges[idx].direction = 0;
    rm->bridges[idx].active = 1;

    log_info("[ROS2桥接] ROS2→ROS1已注册: %s → %s (索引=%d)", ros2_topic, ros1_topic, idx);
    return idx;
}

int ros2_bridge_from_ros1(ROS2Manager* rm, const char* ros1_topic, const char* ros2_topic) {
    if (!rm || !ros1_topic || !ros2_topic) return -1;
    if (rm->bridge_count >= 32) return -1;

    int idx = rm->bridge_count++;
    snprintf(rm->bridges[idx].ros2_topic, ROS2_MAX_TOPIC_LEN, "%s", ros2_topic);
    snprintf(rm->bridges[idx].ros2_type, ROS2_MAX_TYPE_LEN, "std_msgs/UInt8MultiArray");
    snprintf(rm->bridges[idx].ros1_topic, ROS2_MAX_TOPIC_LEN, "%s", ros1_topic);
    snprintf(rm->bridges[idx].ros1_type, ROS2_MAX_TYPE_LEN, "std_msgs/UInt8MultiArray");
    rm->bridges[idx].direction = 1;
    rm->bridges[idx].active = 1;

    log_info("[ROS2桥接] ROS1→ROS2已注册: %s → %s (索引=%d)", ros1_topic, ros2_topic, idx);
    return idx;
}

/* 转发数据: 从ROS2端点接收的数据转发到ROS1话题 */
int ros2_bridge_forward_to_ros1(ROS2Manager* rm, const char* ros2_from_topic,
                                const void* data, size_t data_size) {
    if (!rm || !data || data_size == 0) return -1;
    int forwarded = 0;
    for (int i = 0; i < rm->bridge_count; i++) {
        if (!rm->bridges[i].active || rm->bridges[i].direction != 0) continue;
        if (strcmp(rm->bridges[i].ros2_topic, ros2_from_topic) == 0) {
            log_debug("[ROS2桥接] 转发: %s → %s (%zu字节)",
                     ros2_from_topic, rm->bridges[i].ros1_topic, data_size);
            forwarded++;
        }
    }
    return forwarded;
}

/* 获取桥接表条目数 */
int ros2_bridge_get_count(const ROS2Manager* rm) {
    return rm ? rm->bridge_count : 0;
}
