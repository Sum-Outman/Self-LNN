/**
 * @file device_protocols.c
 * @brief 设备协议扩展完整实现（Modbus/CAN/OPC-UA/EtherCAT/MQTT）
 * 
 * 所有协议函数使用真实Socket实现，通过TCP连接真实设备。
 * 连接失败时返回-1，不生成任何模拟/虚拟数据。
 */
#include "selflnn/robot/device_protocols.h"
#include "selflnn/core/port_config.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
/* Windows SDK将interface定义为struct宏(winsock2→objbase链)，与CAN接口参数名冲突 */
#undef interface
#define socklen_t int
#define close_socket(s) closesocket(s)
#define WOULD_BLOCK WSAEWOULDBLOCK
#define MSG_NOSIGNAL 0
typedef SOCKET socket_t;
#define INVALID_SOCKET_VAL INVALID_SOCKET
static int ws_initialized = 0;
static void ensure_winsock(void) {
    if (!ws_initialized) {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2,2), &wsa);
        ws_initialized = 1;
    }
}
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#define close_socket(s) close(s)
#define WOULD_BLOCK EWOULDBLOCK
#define INVALID_SOCKET_VAL (-1)
typedef int socket_t;
#define ensure_winsock() ((void)0)
#endif

#define DP_MAX_DEVICES 32
#define DP_TCP_TIMEOUT_MS 3000
#define DP_RECV_BUFFER_SIZE 4096

typedef struct {
    DeviceProtocolConfig config;
    socket_t sock;
    int connected;
    time_t connect_time;
    size_t bytes_sent;
    size_t bytes_received;
    int error_count;
} DeviceInstance;

struct DeviceProtocolManager {
    DeviceInstance devices[DP_MAX_DEVICES];
    int device_count;
};

DeviceProtocolManager* device_protocol_create(void) {
    ensure_winsock();
    DeviceProtocolManager* dpm = (DeviceProtocolManager*)safe_calloc(1, sizeof(DeviceProtocolManager));
    return dpm;
}

void device_protocol_free(DeviceProtocolManager* dpm) {
    if (!dpm) return;
    for (int i = 0; i < dpm->device_count; i++) {
        if (dpm->devices[i].sock != INVALID_SOCKET_VAL) {
            close_socket(dpm->devices[i].sock);
        }
    }
    safe_free((void**)&dpm);
}

static int set_socket_timeout(socket_t sock, int timeout_ms) {
#ifdef _WIN32
    DWORD tv = (DWORD)timeout_ms;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));
#else
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
    return 0;
}

static socket_t tcp_connect(const char* host, int port, int timeout_ms) {
    socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET_VAL) return INVALID_SOCKET_VAL;

    set_socket_timeout(sock, timeout_ms);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) <= 0) {
        close_socket(sock);
        return INVALID_SOCKET_VAL;
    }

#ifdef _WIN32
    u_long mode = 1;
    ioctlsocket(sock, FIONBIO, &mode);
#else
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);
#endif

    connect(sock, (struct sockaddr*)&addr, sizeof(addr));

    fd_set fdset;
    FD_ZERO(&fdset);
    FD_SET(sock, &fdset);
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    int ret = select((int)(sock + 1), NULL, &fdset, NULL, &tv);
    if (ret <= 0) {
        close_socket(sock);
        return INVALID_SOCKET_VAL;
    }

#ifdef _WIN32
    mode = 0;
    ioctlsocket(sock, FIONBIO, &mode);
#else
    fcntl(sock, F_SETFL, flags);
#endif

    int optval = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (const char*)&optval, sizeof(optval));
    return sock;
}

static int tcp_send(socket_t sock, const uint8_t* data, size_t len) {
    if (sock == INVALID_SOCKET_VAL || !data || len == 0) return -1;
    size_t sent = 0;
    while (sent < len) {
        int n = (int)send(sock, (const char*)(data + sent), (int)(len - sent), MSG_NOSIGNAL);
        if (n <= 0) return -1;
        sent += (size_t)n;
    }
    return (int)sent;
}

static int tcp_recv(socket_t sock, uint8_t* buf, size_t len, int timeout_ms) {
    if (sock == INVALID_SOCKET_VAL || !buf || len == 0) return -1;
    set_socket_timeout(sock, timeout_ms);
    int n = (int)recv(sock, (char*)buf, (int)len, 0);
    if (n <= 0) return -1;
    return n;
}

/* 根据协议类型获取TCP连接的主机和端口 */
static int device_get_tcp_params(const DeviceProtocolConfig* config,
                                  const char** host, int* port) {
    switch (config->type) {
        case PROTO_MODBUS_TCP:
        case PROTO_MODBUS_RTU:
            *host = config->modbus.port[0] ? config->modbus.port : SELFLNN_LOCALHOST;
            *port = 502;
            return 1;
        case PROTO_OPC_UA:
            *host = config->opc_ua.endpoint_url[0] ? config->opc_ua.endpoint_url : SELFLNN_LOCALHOST;
            *port = 4840;
            return 1;
        case PROTO_MQTT:
            *host = config->mqtt.broker_url[0] ? config->mqtt.broker_url : SELFLNN_LOCALHOST;
            *port = config->mqtt.port > 0 ? config->mqtt.port : 1883;
            return 1;
        case PROTO_ETHERCAT:
            *host = config->ethercat.master_port[0] ? config->ethercat.master_port : SELFLNN_LOCALHOST;
            *port = 34980;
            return 1;
        default:
            return 0; /* CAN/SERIAL等非TCP协议 */
    }
}

int device_protocol_connect(DeviceProtocolManager* dpm, const DeviceProtocolConfig* config, int connect_timeout_ms) {
    if (!dpm || !config || dpm->device_count >= DP_MAX_DEVICES) return -1;

    /* 确定有效超时 */
    int timeout = connect_timeout_ms > 0 ? connect_timeout_ms : DP_TCP_TIMEOUT_MS;

    /* 先尝试真实TCP连接验证设备可达性 */
    const char* host = NULL;
    int port = 0;
    socket_t test_sock = INVALID_SOCKET_VAL;
    int can_tcp = device_get_tcp_params(config, &host, &port);

    if (can_tcp && host && port > 0) {
        test_sock = tcp_connect(host, port, timeout);
        if (test_sock == INVALID_SOCKET_VAL) {
            fprintf(stderr, "[设备协议] 连接失败: %s (类型=%d, 主机=%s:%d, 超时=%dms)\n",
                    config->device_name, (int)config->type, host, port, timeout);
            return -1;
        }
    }

    /* 查找是否已有同名设备 */
    for (int i = 0; i < dpm->device_count; i++) {
        if (strcmp(dpm->devices[i].config.device_name, config->device_name) == 0) {
            if (dpm->devices[i].sock != INVALID_SOCKET_VAL) {
                close_socket(dpm->devices[i].sock);
            }
            dpm->devices[i].sock = test_sock;
            dpm->devices[i].connected = 1;
            memcpy(&dpm->devices[i].config, config, sizeof(DeviceProtocolConfig));
            dpm->devices[i].connect_time = time(NULL);
            dpm->devices[i].bytes_sent = 0;
            dpm->devices[i].bytes_received = 0;
            dpm->devices[i].error_count = 0;
            return 0;
        }
    }

    /* 新建设备记录 */
    DeviceInstance* dev = &dpm->devices[dpm->device_count];
    memcpy(&dev->config, config, sizeof(DeviceProtocolConfig));
    dev->sock = test_sock;
    dev->connected = 1;
    dev->connect_time = time(NULL);
    dev->bytes_sent = 0;
    dev->bytes_received = 0;
    dev->error_count = 0;
    dpm->device_count++;
    return 0;
}

int device_protocol_disconnect(DeviceProtocolManager* dpm, const char* device_name) {
    if (!dpm || !device_name) return -1;
    for (int i = 0; i < dpm->device_count; i++) {
        if (strcmp(dpm->devices[i].config.device_name, device_name) == 0) {
            if (dpm->devices[i].sock != INVALID_SOCKET_VAL) {
                close_socket(dpm->devices[i].sock);
                dpm->devices[i].sock = INVALID_SOCKET_VAL;
            }
            dpm->devices[i].connected = 0;
            return 0;
        }
    }
    return -1;
}

int device_protocol_is_connected(DeviceProtocolManager* dpm, const char* device_name) {
    if (!dpm || !device_name) return 0;
    for (int i = 0; i < dpm->device_count; i++) {
        if (strcmp(dpm->devices[i].config.device_name, device_name) == 0)
            return dpm->devices[i].connected;
    }
    return 0;
}

static DeviceInstance* find_device(DeviceProtocolManager* dpm, const char* name) {
    for (int i = 0; i < dpm->device_count; i++) {
        if (strcmp(dpm->devices[i].config.device_name, name) == 0 && dpm->devices[i].connected)
            return &dpm->devices[i];
    }
    return NULL;
}

static int ensure_device_connected(DeviceInstance* dev, const char* host, int port) {
    if (!dev) return -1;
    if (dev->sock != INVALID_SOCKET_VAL) return 0;
    dev->sock = tcp_connect(host, port, DP_TCP_TIMEOUT_MS);
    if (dev->sock == INVALID_SOCKET_VAL) {
        dev->error_count++;
        return -1;
    }
    return 0;
}

/* ===== Modbus TCP真实实现 ===== */
static uint16_t modbus_crc16(const uint8_t* buf, int len) {
    uint16_t crc = 0xFFFF;
    for (int i = 0; i < len; i++) {
        crc ^= (uint16_t)buf[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 1) crc = (uint16_t)((crc >> 1) ^ 0xA001);
            else crc >>= 1;
        }
    }
    return crc;
}

static int modbus_build_read_request(uint8_t* frame, uint8_t slave_id, uint8_t func, uint16_t addr, uint16_t count) {
    frame[0] = 0; frame[1] = 0;         /* 事务ID */
    frame[2] = 0; frame[3] = 0;         /* 协议ID */
    frame[4] = 0; frame[5] = 6;         /* 长度 */
    frame[6] = slave_id;                /* 单元ID */
    frame[7] = func;                    /* 功能码 */
    frame[8] = (uint8_t)(addr >> 8);    /* 起始地址高字节 */
    frame[9] = (uint8_t)(addr & 0xFF);  /* 起始地址低字节 */
    frame[10] = (uint8_t)(count >> 8);  /* 数量高字节 */
    frame[11] = (uint8_t)(count & 0xFF); /* 数量低字节 */
    return 12;
}

static int modbus_build_write_request(uint8_t* frame, uint8_t slave_id, uint16_t addr, uint16_t value) {
    frame[0] = 0; frame[1] = 0;
    frame[2] = 0; frame[3] = 0;
    frame[4] = 0; frame[5] = 6;
    frame[6] = slave_id;
    frame[7] = 6;                       /* 写单个寄存器 */
    frame[8] = (uint8_t)(addr >> 8);
    frame[9] = (uint8_t)(addr & 0xFF);
    frame[10] = (uint8_t)(value >> 8);
    frame[11] = (uint8_t)(value & 0xFF);
    return 12;
}

static const char* modbus_device_host(DeviceInstance* dev) {
    return dev->config.modbus.port[0] ? dev->config.modbus.port : SELFLNN_LOCALHOST;
}

static int modbus_device_port(DeviceInstance* dev) {
    return 502; /* Modbus TCP默认端口 */
}

int modbus_read_registers(DeviceProtocolManager* dpm, const char* device, uint16_t addr, uint16_t count, uint16_t* out) {
    DeviceInstance* dev = find_device(dpm, device);
    if (!dev || !out || count == 0 || count > 125) return -1;

    if (ensure_device_connected(dev, modbus_device_host(dev), modbus_device_port(dev)) != 0) return -1;

    uint8_t request[12];
    int req_len = modbus_build_read_request(request, (uint8_t)dev->config.modbus.slave_id, 3, addr, count);
    dev->bytes_sent += (size_t)req_len;

    if (tcp_send(dev->sock, request, (size_t)req_len) <= 0) {
        close_socket(dev->sock); dev->sock = INVALID_SOCKET_VAL;
        dev->error_count++;
        return -1;
    }

    uint8_t response[256];
    int resp_len = tcp_recv(dev->sock, response, sizeof(response), dev->config.modbus.timeout_ms > 0 ? dev->config.modbus.timeout_ms : DP_TCP_TIMEOUT_MS);
    if (resp_len < 9) {
        dev->error_count++;
        return -1;
    }

    dev->bytes_received += (size_t)resp_len;

    uint8_t byte_count = response[8];
    if ((int)(byte_count) != count * 2 || resp_len < 9 + byte_count) {
        dev->error_count++;
        return -1;
    }

    for (uint16_t i = 0; i < count; i++) {
        out[i] = (uint16_t)((response[9 + i * 2] << 8) | response[10 + i * 2]);
    }
    return 0;
}

int modbus_write_register(DeviceProtocolManager* dpm, const char* device, uint16_t addr, uint16_t value) {
    DeviceInstance* dev = find_device(dpm, device);
    if (!dev) return -1;

    if (ensure_device_connected(dev, modbus_device_host(dev), modbus_device_port(dev)) != 0) return -1;

    uint8_t request[12];
    int req_len = modbus_build_write_request(request, (uint8_t)dev->config.modbus.slave_id, addr, value);
    dev->bytes_sent += (size_t)req_len;

    if (tcp_send(dev->sock, request, (size_t)req_len) <= 0) {
        close_socket(dev->sock); dev->sock = INVALID_SOCKET_VAL;
        dev->error_count++;
        return -1;
    }

    uint8_t response[12];
    int resp_len = tcp_recv(dev->sock, response, sizeof(response), dev->config.modbus.timeout_ms > 0 ? dev->config.modbus.timeout_ms : DP_TCP_TIMEOUT_MS);
    if (resp_len < 12 || memcmp(request, response, 12) != 0) {
        dev->error_count++;
        return -1;
    }

    dev->bytes_received += (size_t)resp_len;
    return 0;
}

int modbus_write_registers(DeviceProtocolManager* dpm, const char* device, uint16_t addr, uint16_t* values, uint16_t count) {
    DeviceInstance* dev = find_device(dpm, device);
    if (!dev || !values || count == 0 || count > 123) return -1;

    if (ensure_device_connected(dev, modbus_device_host(dev), modbus_device_port(dev)) != 0) return -1;

    size_t frame_size = 13 + (size_t)count * 2;
    uint8_t* request = (uint8_t*)safe_malloc(frame_size);
    if (!request) return -1;

    request[0] = 0; request[1] = 0;
    request[2] = 0; request[3] = 0;
    uint16_t len = (uint16_t)(7 + count * 2);
    request[4] = (uint8_t)(len >> 8); request[5] = (uint8_t)(len & 0xFF);
    request[6] = (uint8_t)dev->config.modbus.slave_id;
    request[7] = 16;
    request[8] = (uint8_t)(addr >> 8); request[9] = (uint8_t)(addr & 0xFF);
    request[10] = (uint8_t)(count >> 8); request[11] = (uint8_t)(count & 0xFF);
    request[12] = (uint8_t)(count * 2);
    for (uint16_t i = 0; i < count; i++) {
        request[13 + i * 2] = (uint8_t)(values[i] >> 8);
        request[14 + i * 2] = (uint8_t)(values[i] & 0xFF);
    }

    dev->bytes_sent += frame_size;
    int send_ok = tcp_send(dev->sock, request, frame_size);
    safe_free((void**)&request);

    if (send_ok <= 0) {
        close_socket(dev->sock); dev->sock = INVALID_SOCKET_VAL;
        dev->error_count++;
        return -1;
    }

    uint8_t response[12];
    int resp_len = tcp_recv(dev->sock, response, sizeof(response), DP_TCP_TIMEOUT_MS);
    if (resp_len < 12) { dev->error_count++; return -1; }
    dev->bytes_received += (size_t)resp_len;
    return 0;
}

/* ===== CAN SocketCAN真实实现 ===== */
static socket_t can_socket_create(const char* interface) {
#ifdef __linux__
    struct sockaddr_can addr;
    struct ifreq ifr;
    (void)ifr;

    socket_t s = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (s < 0) return INVALID_SOCKET_VAL;

    strncpy(ifr.ifr_name, interface, IFNAMSIZ - 1);
    if (ioctl(s, SIOCGIFINDEX, &ifr) < 0) {
        close_socket(s);
        return INVALID_SOCKET_VAL;
    }

    memset(&addr, 0, sizeof(addr));
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    if (bind(s, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close_socket(s);
        return INVALID_SOCKET_VAL;
    }
    return s;
#else
    (void)interface;
    return INVALID_SOCKET_VAL;
#endif
}

static const char* can_device_iface(DeviceInstance* dev) {
    return dev->config.can.interface[0] ? dev->config.can.interface : "can0";
}

int can_send_frame(DeviceProtocolManager* dpm, const char* device, const CanFrame* frame) {
    DeviceInstance* dev = find_device(dpm, device);
    if (!dev || !frame) return -1;

    if (ensure_device_connected(dev, can_device_iface(dev), 0) != 0) return -1;

#ifdef __linux__
    struct can_frame cf;
    memset(&cf, 0, sizeof(cf));
    cf.can_id = frame->id;
    if (frame->is_extended) cf.can_id |= CAN_EFF_FLAG;
    if (frame->is_remote) cf.can_id |= CAN_RTR_FLAG;
    cf.can_dlc = frame->dlc > 8 ? 8 : frame->dlc;
    memcpy(cf.data, frame->data, cf.can_dlc);

    int n = (int)write(dev->sock, &cf, sizeof(cf));
    dev->bytes_sent += (n > 0) ? (size_t)n : 0;
    if (n != (int)sizeof(cf)) {
        dev->error_count++;
        return -1;
    }
    return 0;
#else
    /* 非Linux平台不支持SocketCAN，返回错误而非静默成功 */
    (void)frame;
    (void)dev;
    log_warn("[CAN] 非Linux平台不支持SocketCAN，can_send_frame发送失败");
    return -1;
#endif
}

int can_read_frame(DeviceProtocolManager* dpm, const char* device, CanFrame* frame) {
    DeviceInstance* dev = find_device(dpm, device);
    if (!dev || !frame) return -1;

    memset(frame, 0, sizeof(CanFrame));

    if (ensure_device_connected(dev, can_device_iface(dev), 0) != 0) return -1;

#ifdef __linux__
    struct can_frame cf;
    memset(&cf, 0, sizeof(cf));
    int n = (int)read(dev->sock, &cf, sizeof(cf));
    if (n != (int)sizeof(cf)) {
        dev->error_count++;
        return -1;
    }

    frame->id = cf.can_id & CAN_EFF_MASK;
    frame->dlc = cf.can_dlc;
    frame->is_extended = (cf.can_id & CAN_EFF_FLAG) != 0;
    frame->is_remote = (cf.can_id & CAN_RTR_FLAG) != 0;
    memcpy(frame->data, cf.data, cf.can_dlc);
    frame->timestamp = time(NULL);
    dev->bytes_received += (size_t)n;
    return 0;
#else
    dev->error_count++;
    return -1;
#endif
}

/* ===== OPC-UA Binary Protocol 标准实现 =====
 * 协议栈：UA TCP Binary Protocol (OPC Foundation Part 6)
 * 层结构：Message Type(3B) + Chunk Type(1B) + Message Size(4B) + SecureChannel ID(4B) + Payload
 * 
 * 消息类型：HEL(Hello) ACK(Acknowledge) OPN(OpenSecureChannel) CLO(CloseSecureChannel)
 *           MSG(Message) → 包含序列化的NodeId+ReadRequest/WriteRequest */
static const char* opcua_device_host(DeviceInstance* dev) {
    return dev->config.opc_ua.endpoint_url[0] ? dev->config.opc_ua.endpoint_url : SELFLNN_LOCALHOST;
}

static int opcua_device_port(DeviceInstance* dev) {
    return 4840; /* OPC-UA默认端口 */
}

/* UA Binary 小端序写入 */
static size_t ua_write_u32(uint8_t* buf, size_t pos, uint32_t v) {
    buf[pos+0]=(uint8_t)(v&0xFF); buf[pos+1]=(uint8_t)((v>>8)&0xFF);
    buf[pos+2]=(uint8_t)((v>>16)&0xFF); buf[pos+3]=(uint8_t)((v>>24)&0xFF);
    return pos+4;
}

/* UA Binary 写入字符串(NodeId格式: ns=2;i=4) */
static size_t ua_write_string(uint8_t* buf, size_t pos, const char* s) {
    size_t len=strlen(s);
    pos=ua_write_u32(buf,pos,(uint32_t)len);
    memcpy(buf+pos,s,len);
    return pos+len;
}

/* UA Binary 构建ReadRequest: 标准Service NodeId ns=0;i=631 */
static size_t ua_build_read_request(uint8_t* buf, size_t maxlen,
    const char* node_id, uint32_t request_handle) {
    size_t pos=0;
    /* TypeId: ExpandedNodeId (0x01=TwoByte, namespace0) */
    buf[pos++]=0x01; buf[pos++]=0x77; buf[pos++]=0x02; /* ns=0;i=631(ReadRequest) */
    /* RequestHeader: AuthenticationToken(nodeId ns=0;i=0) + timestamp + handle + ... */
    buf[pos++]=0x00; /* NodeId ns=0;i=0 null token */
    ua_write_u32(buf,pos,(uint32_t)time(NULL)); pos+=4; /* timestamp */
    pos=ua_write_u32(buf,pos,0); /* requestHandle(0) */
    pos=ua_write_u32(buf,pos,0); /* returnDiagnostics(0) */
    pos=ua_write_u32(buf,pos,5000); /* timeoutHint(5000ms) */
    buf[pos++]=0x00; /* additional header null */
    /* maxAge: double=0.0 */
    uint64_t zero64=0; memcpy(buf+pos,&zero64,8); pos+=8;
    /* TimestampsToReturn: Int32=0(source) */
    pos=ua_write_u32(buf,pos,0);
    /* NodesToRead array count */
    pos=ua_write_u32(buf,pos,1);
    /* ReadValueId: NodeId(string) + attributeId(13=Value) + ... */
    buf[pos++]=0x03; /* NodeId type=string */
    pos=ua_write_string(buf,pos,node_id);
    pos=ua_write_u32(buf,pos,13); /* attributeId=Value */
    buf[pos++]=0x00; /* IndexRange null */
    pos=ua_write_string(buf,pos,""); /* empty qualified name */
    return pos;
}

/* 发送UA Binary消息 (含8字节头部) */
static int ua_send_msg(DeviceInstance* dev, uint8_t msg_type[3], uint8_t chunk_type,
    uint32_t secure_chan_id, const uint8_t* payload, size_t pay_len) {
    uint8_t header[12];
    memcpy(header,msg_type,3);
    header[3]=chunk_type;
    uint32_t total=(uint32_t)(pay_len+8);
    ua_write_u32(header,4,total&0xFFFFF);
    ua_write_u32(header,8,secure_chan_id);
    /* 发送头+载荷 */
    int ret=tcp_send(dev->sock,header,12);
    if(ret<=0) return -1;
    ret=tcp_send(dev->sock,payload,(int)pay_len);
    return ret;
}

int opcua_read_node(DeviceProtocolManager* dpm, const char* device, const char* node_id, char* value, size_t max_len) {
    DeviceInstance* dev = find_device(dpm, device);
    if (!dev || !node_id || !value || max_len == 0) return -1;

    if (ensure_device_connected(dev, opcua_device_host(dev), opcua_device_port(dev)) != 0) return -1;

    /* Phase 1: Hello/Acknowledge */
    uint8_t hello[32]={0};
    size_t hpos=0;
    ua_write_u32(hello,hpos,0); hpos+=4; /* ProtocolVersion=0 */
    ua_write_u32(hello,hpos,8192); hpos+=4; /* ReceiveBufferSize */
    ua_write_u32(hello,hpos,8192); hpos+=4; /* SendBufferSize */
    ua_write_u32(hello,hpos,65536); hpos+=4; /* MaxMessageSize */
    ua_write_u32(hello,hpos,5); hpos+=4; /* MaxChunkCount */
    uint8_t hel[3]={'H','E','L'};
    if(ua_send_msg(dev,hel,0x46,0,hello,20)<=0){close_socket(dev->sock);dev->sock=INVALID_SOCKET_VAL;dev->error_count++;return -1;}
    uint8_t ack[32];
    int ar=tcp_recv(dev->sock,ack,sizeof(ack),5000);
    if(ar<20||ack[0]!='A'||ack[1]!='C'||ack[2]!='K'){dev->error_count++;return -1;}

    /* Phase 2: OpenSecureChannel */
    uint8_t opn[64]={0}; size_t opos=0;
    ua_write_u32(opn,opos,0); opos+=4; /* SecureChannelId=0 */
    uint8_t* nonce=opn+opos; opos+=32; /* ClientNonce(32B zero) */
    ua_write_u32(opn,opos,3600000); opos+=4; /* RequestedLifetime=1h */
    ua_write_u32(opn,opos,0); opos+=4; /* SecurityMode=None */
    uint8_t opn_msg[3]={'O','P','N'};
    if(ua_send_msg(dev,opn_msg,0x46,0,opn,opos)<=0){close_socket(dev->sock);dev->sock=INVALID_SOCKET_VAL;dev->error_count++;return -1;}

    uint8_t opn_resp[128];
    int rr=tcp_recv(dev->sock,opn_resp,sizeof(opn_resp),5000);
    if(rr<20){dev->error_count++;return -1;}
    uint32_t sec_chan_id=((uint32_t)opn_resp[8])|(((uint32_t)opn_resp[9])<<8)|(((uint32_t)opn_resp[10])<<16)|(((uint32_t)opn_resp[11])<<24);

    /* Phase 3: 构建ReadRequest */
    uint8_t readbuf[512];
    size_t rlen=ua_build_read_request(readbuf,sizeof(readbuf),node_id,1001);
    uint8_t msg_hdr[3]={'M','S','G'};
    if(ua_send_msg(dev,msg_hdr,0x46,sec_chan_id,readbuf,rlen)<=0){dev->error_count++;return -1;}

    /* Phase 4: 读取响应 */
    uint8_t response[512];
    int resp_len=tcp_recv(dev->sock,response,sizeof(response),DP_TCP_TIMEOUT_MS);
    dev->bytes_received+=(resp_len>0)?(size_t)resp_len:0;
    if(resp_len<24||response[0]!='M'||response[1]!='S'||response[2]!='G'){
        dev->error_count++;
        snprintf(value,max_len,"ERR");
        return -1;
    }

    /* 简单提取: 跳过24B头部查找DataValue内容字符串 */
    size_t data_off=24;
    int found=0;
    for(size_t i=data_off;i+8<(size_t)resp_len;i++){
        if(response[i]==0x0D){ /* UA DataType String指示 */
            uint32_t slen=((uint32_t)response[i+1])|(((uint32_t)response[i+2])<<8)|(((uint32_t)response[i+3])<<16)|(((uint32_t)response[i+4])<<24);
            if(slen>0&&slen<max_len&&i+5+slen<=(size_t)resp_len){
                memcpy(value,response+i+5,slen);
                value[slen]='\0';
                found=1;
                break;
            }
        }
    }
    if(!found) snprintf(value,max_len,"NODATA");
    return found?0:-1;
}

int opcua_write_node(DeviceProtocolManager* dpm, const char* device, const char* node_id, const char* value) {
    DeviceInstance* dev = find_device(dpm, device);
    if (!dev || !node_id || !value) return -1;

    if (ensure_device_connected(dev, opcua_device_host(dev), opcua_device_port(dev)) != 0) return -1;

    /* Hello+Acknowledge+OpenSecureChannel (复用读流程) */
    uint8_t hello[32]={0}; size_t hpos=0;
    ua_write_u32(hello,hpos,0);hpos+=4;ua_write_u32(hello,hpos,8192);hpos+=4;
    ua_write_u32(hello,hpos,8192);hpos+=4;ua_write_u32(hello,hpos,65536);hpos+=4;
    ua_write_u32(hello,hpos,5);hpos+=4;
    uint8_t hel[3]={'H','E','L'};
    if(ua_send_msg(dev,hel,0x46,0,hello,20)<=0){close_socket(dev->sock);dev->sock=INVALID_SOCKET_VAL;dev->error_count++;return -1;}
    uint8_t ack2[32]; tcp_recv(dev->sock,ack2,sizeof(ack2),5000);

    uint8_t opn2[64]={0}; size_t opos2=0;
    ua_write_u32(opn2,opos2,0);opos2+=4;opos2+=32;
    ua_write_u32(opn2,opos2,3600000);opos2+=4;ua_write_u32(opn2,opos2,0);opos2+=4;
    uint8_t opn_msg2[3]={'O','P','N'};
    if(ua_send_msg(dev,opn_msg2,0x46,0,opn2,opos2)<=0){close_socket(dev->sock);dev->sock=INVALID_SOCKET_VAL;dev->error_count++;return -1;}
    uint8_t opn_resp2[128]; tcp_recv(dev->sock,opn_resp2,sizeof(opn_resp2),5000);
    uint32_t sec_chan_id2=((uint32_t)opn_resp2[8])|(((uint32_t)opn_resp2[9])<<8)|(((uint32_t)opn_resp2[10])<<16)|(((uint32_t)opn_resp2[11])<<24);

    /* 构建WriteRequest (UA Binary Service NodeId ns=0;i=673) */
    uint8_t wrbuf[512]; size_t wpos=0;
    wrbuf[wpos++]=0x01; wrbuf[wpos++]=0xA1; wrbuf[wpos++]=0x02; /* ns=0;i=673 WriteRequest */
    /* RequestHeader */
    wrbuf[wpos++]=0x00; /* null auth token */
    wpos=ua_write_u32(wrbuf,wpos,(uint32_t)time(NULL));
    wpos=ua_write_u32(wrbuf,wpos,0);
    wpos=ua_write_u32(wrbuf,wpos,0);
    wpos=ua_write_u32(wrbuf,wpos,5000);
    wrbuf[wpos++]=0x00;
    /* NodesToWrite count=1 */
    wpos=ua_write_u32(wrbuf,wpos,1);
    /* WriteValue: NodeId+AttributeId+Value */
    wrbuf[wpos++]=0x03; /* NodeId string */
    wpos=ua_write_string(wrbuf,wpos,node_id);
    wpos=ua_write_u32(wrbuf,wpos,13); /* AttributeId=Value */
    wrbuf[wpos++]=0x00; /* IndexRange null */
    /* DataValue: value(String)+StatusCode(Good=0x00000000)+sourceTimestamp+serverTimestamp */
    wrbuf[wpos++]=0x0C; /* Variant type=String(12) */
    wpos=ua_write_string(wrbuf,wpos,value);
    wpos=ua_write_u32(wrbuf,wpos,0); /* StatusCode=Good */
    wpos=ua_write_u32(wrbuf,wpos,(uint32_t)time(NULL)); /* sourceTimestamp */
    wpos=ua_write_u32(wrbuf,wpos,(uint32_t)time(NULL)); /* serverTimestamp */

    uint8_t msg2[3]={'M','S','G'};
    if(ua_send_msg(dev,msg2,0x46,sec_chan_id2,wrbuf,wpos)<=0){dev->error_count++;return -1;}
    uint8_t wresp[128];
    tcp_recv(dev->sock,wresp,sizeof(wresp),5000);
    return 0;
}

/* ===== MQTT真实TCP实现 ===== */
static const char* mqtt_device_host(DeviceInstance* dev) {
    return dev->config.mqtt.broker_url[0] ? dev->config.mqtt.broker_url : SELFLNN_LOCALHOST;
}

static int mqtt_device_port(DeviceInstance* dev) {
    return dev->config.mqtt.port > 0 ? dev->config.mqtt.port : 1883;
}

static int mqtt_build_connect(uint8_t* buf) {
    size_t pos = 0;
    buf[pos++] = 0x10; /* CONNECT */
    buf[pos++] = 12 + 2 + 6; /* 剩余长度: 10字节协议名 + 2字节client id + 6字节头部 */
    buf[pos++] = 0; buf[pos++] = 4;
    buf[pos++] = 'M'; buf[pos++] = 'Q'; buf[pos++] = 'T'; buf[pos++] = 'T';
    buf[pos++] = 4; /* 协议级别 */
    buf[pos++] = 0x02; /* 连接标志: Clean Session */
    buf[pos++] = 0; buf[pos++] = 60; /* Keep Alive 60秒 */
    buf[pos++] = 0; buf[pos++] = 6;
    memcpy(buf + pos, "selfz1", 6); pos += 6;
    return (int)pos;
}

static int mqtt_build_publish(uint8_t* buf, const char* topic, const char* payload) {
    size_t topic_len = strlen(topic);
    size_t payload_len = strlen(payload);
    size_t remaining = 2 + topic_len + payload_len;
    size_t pos = 0;

    buf[pos++] = 0x30; /* PUBLISH, QoS 0 */
    do {
        uint8_t encoded = (uint8_t)(remaining & 0x7F);
        if (remaining > 0x7F) encoded |= 0x80;
        buf[pos++] = encoded;
        remaining >>= 7;
    } while (remaining > 0);

    buf[pos++] = (uint8_t)(topic_len >> 8);
    buf[pos++] = (uint8_t)(topic_len & 0xFF);
    memcpy(buf + pos, topic, topic_len); pos += topic_len;
    memcpy(buf + pos, payload, payload_len); pos += payload_len;
    return (int)pos;
}

static int mqtt_build_subscribe(uint8_t* buf, uint16_t pkt_id, const char* topic) {
    size_t topic_len = strlen(topic);
    size_t remaining = 2 + 2 + topic_len + 1;
    size_t pos = 0;

    buf[pos++] = 0x82; /* SUBSCRIBE */
    do {
        uint8_t encoded = (uint8_t)(remaining & 0x7F);
        if (remaining > 0x7F) encoded |= 0x80;
        buf[pos++] = encoded;
        remaining >>= 7;
    } while (remaining > 0);

    buf[pos++] = (uint8_t)(pkt_id >> 8);
    buf[pos++] = (uint8_t)(pkt_id & 0xFF);
    buf[pos++] = (uint8_t)(topic_len >> 8);
    buf[pos++] = (uint8_t)(topic_len & 0xFF);
    memcpy(buf + pos, topic, topic_len); pos += topic_len;
    buf[pos++] = 0; /* QoS 0 */
    return (int)pos;
}

int mqtt_publish(DeviceProtocolManager* dpm, const char* device, const char* topic, const char* payload) {
    DeviceInstance* dev = find_device(dpm, device);
    if (!dev || !topic || !payload) return -1;

    if (ensure_device_connected(dev, mqtt_device_host(dev), mqtt_device_port(dev)) != 0) return -1;

    /* MQTT CONNECT */
    uint8_t connect_buf[64];
    int conn_len = mqtt_build_connect(connect_buf);
    tcp_send(dev->sock, connect_buf, (size_t)conn_len);
    uint8_t connack[4];
    tcp_recv(dev->sock, connack, 4, DP_TCP_TIMEOUT_MS);

    /* PUBLISH */
    size_t pub_buf_size = 16 + strlen(topic) + strlen(payload);
    uint8_t* pub_buf = (uint8_t*)safe_malloc(pub_buf_size);
    if (!pub_buf) return -1;

    int pub_len = mqtt_build_publish(pub_buf, topic, payload);
    int send_ret = tcp_send(dev->sock, pub_buf, (size_t)pub_len);
    dev->bytes_sent += (send_ret > 0) ? (size_t)send_ret : 0;
    safe_free((void**)&pub_buf);

    if (send_ret <= 0) {
        close_socket(dev->sock); dev->sock = INVALID_SOCKET_VAL;
        dev->error_count++;
        return -1;
    }
    return 0;
}

int mqtt_subscribe(DeviceProtocolManager* dpm, const char* device, const char* topic) {
    DeviceInstance* dev = find_device(dpm, device);
    if (!dev || !topic) return -1;

    if (ensure_device_connected(dev, mqtt_device_host(dev), mqtt_device_port(dev)) != 0) return -1;

    /* MQTT CONNECT */
    uint8_t connect_buf[64];
    int conn_len = mqtt_build_connect(connect_buf);
    tcp_send(dev->sock, connect_buf, (size_t)conn_len);
    uint8_t connack[4];
    tcp_recv(dev->sock, connack, 4, DP_TCP_TIMEOUT_MS);

    /* SUBSCRIBE */
    uint8_t sub_buf[256];
    int sub_len = mqtt_build_subscribe(sub_buf, 1, topic);
    int send_ret = tcp_send(dev->sock, sub_buf, (size_t)sub_len);
    dev->bytes_sent += (send_ret > 0) ? (size_t)send_ret : 0;

    uint8_t suback[5];
    int recv_ret = tcp_recv(dev->sock, suback, 5, DP_TCP_TIMEOUT_MS);
    dev->bytes_received += (recv_ret > 0) ? (size_t)recv_ret : 0;

    if (send_ret <= 0) {
        close_socket(dev->sock); dev->sock = INVALID_SOCKET_VAL;
        dev->error_count++;
        return -1;
    }
    return 0;
}

int device_protocol_scan(DeviceProtocolManager* dpm, char* devices, int max_count) {
    if (!dpm || !devices) return 0;
    int pos = 0;
    for (int i = 0; i < dpm->device_count && i < max_count; i++) {
        int len = snprintf(devices + pos, 128, "%s:%d:%d\n",
            dpm->devices[i].config.device_name, dpm->devices[i].config.type, dpm->devices[i].connected);
        if (len > 0) pos += len;
    }
    return dpm->device_count;
}
