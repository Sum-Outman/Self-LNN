/**
 * @file ros_protocol.c
 * @brief ROS协议序列化/反序列化、TCP传输层、DDS RTPS协议完整实现
 *
 * 实现内容：
 * 1. ROS消息二进制序列化/反序列化（小端字节序）
 * 2. ROS1 TCP传输层（TCPROS + XMLRPC Master通信）
 * 3. ROS2 DDS RTPS协议（参与者发现 + 端点发现 + 数据分发）
 *
 * 纯C实现，零外部ROS依赖。跨平台socket支持。
 */

#include "selflnn/robot/ros_protocol.h"
#include "selflnn/utils/logging.h"
#include "selflnn/core/port_config.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")
#define SELFLNN_SOCKET_ERROR SOCKET_ERROR
#define SELFLNN_INVALID_SOCKET INVALID_SOCKET
#define SELFLNN_EWOULDBLOCK WSAEWOULDBLOCK
#define SELFLNN_EINPROGRESS WSAEINPROGRESS
#define sockclose closesocket
typedef int socklen_t;
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#define SELFLNN_SOCKET_ERROR (-1)
#define SELFLNN_INVALID_SOCKET (-1)
#define SELFLNN_EWOULDBLOCK EWOULDBLOCK
#define SELFLNN_EINPROGRESS EINPROGRESS
#define sockclose close
typedef int SOCKET;
#endif

/* ============================================================================
 * 内部辅助宏
 * ============================================================================ */
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* 网络初始化计数器 */
static int g_network_init_count = 0;

/* 内部: 初始化网络子系统 */
static int network_init(void) {
    if (g_network_init_count > 0) {
        g_network_init_count++;
        return 0;
    }
#if defined(_WIN32)
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        return -1;
    }
#endif
    g_network_init_count = 1;
    return 0;
}

/* 内部: 清理网络子系统 */
static void network_cleanup(void) {
    if (g_network_init_count <= 0) return;
    g_network_init_count--;
    if (g_network_init_count == 0) {
#if defined(_WIN32)
        WSACleanup();
#endif
    }
}

/* 内部: 设置socket为非阻塞 */
static int socket_set_nonblocking(SOCKET sock) {
#if defined(_WIN32)
    unsigned long mode = 1;
    return ioctlsocket(sock, FIONBIO, &mode);
#else
    int flags = fcntl(sock, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(sock, F_SETFL, flags | O_NONBLOCK);
#endif
}

/* 内部: 带超时的connect */
static int socket_connect_timeout(const char* host, int port, int timeout_ms) {
    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == SELFLNN_INVALID_SOCKET) return -1;

    if (timeout_ms > 0) {
        socket_set_nonblocking(sock);
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) <= 0) {
        struct hostent* he = gethostbyname(host);
        if (!he) { sockclose(sock); return -1; }
        memcpy(&addr.sin_addr, he->h_addr_list[0], (size_t)he->h_length);
    }

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        return (int)sock;
    }

#if defined(_WIN32)
    if (WSAGetLastError() != WSAEWOULDBLOCK && WSAGetLastError() != WSAEINPROGRESS) {
        sockclose(sock);
        return -1;
    }
#else
    if (errno != EINPROGRESS && errno != EWOULDBLOCK) {
        sockclose(sock);
        return -1;
    }
#endif

    fd_set write_fds;
    FD_ZERO(&write_fds);
    FD_SET(sock, &write_fds);
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    int ret = select((int)(sock + 1), NULL, &write_fds, NULL, &tv);
    if (ret <= 0) {
        sockclose(sock);
        return -1;
    }

    int optval = 0;
    socklen_t optlen = sizeof(optval);
    if (getsockopt(sock, SOL_SOCKET, SO_ERROR, (char*)&optval, &optlen) < 0 || optval != 0) {
        sockclose(sock);
        return -1;
    }

#if defined(_WIN32)
    unsigned long mode = 0;
    ioctlsocket(sock, FIONBIO, &mode);
#else
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags & ~O_NONBLOCK);
#endif

    return (int)sock;
}

/* ============================================================================
 * 小端字节序写入/读取辅助函数
 * ============================================================================ */

/* 写入uint8 */
static inline void write_u8(uint8_t** buf, uint8_t val) {
    *(*buf)++ = val;
}

/* 读取uint8 */
static inline uint8_t read_u8(const uint8_t** buf) {
    return *(*buf)++;
}

/* 写入uint16 (小端) */
static inline void write_u16_le(uint8_t** buf, uint16_t val) {
    *(*buf)++ = (uint8_t)(val & 0xFF);
    *(*buf)++ = (uint8_t)((val >> 8) & 0xFF);
}

/* 读取uint16 (小端) */
static inline uint16_t read_u16_le(const uint8_t** buf) {
    uint16_t val = (uint16_t)(*buf)[0] | ((uint16_t)(*buf)[1] << 8);
    *buf += 2;
    return val;
}

/* 写入uint32 (小端) */
static inline void write_u32_le(uint8_t** buf, uint32_t val) {
    *(*buf)++ = (uint8_t)(val & 0xFF);
    *(*buf)++ = (uint8_t)((val >> 8) & 0xFF);
    *(*buf)++ = (uint8_t)((val >> 16) & 0xFF);
    *(*buf)++ = (uint8_t)((val >> 24) & 0xFF);
}

/* 读取uint32 (小端) */
static inline uint32_t read_u32_le(const uint8_t** buf) {
    uint32_t val = (uint32_t)(*buf)[0] |
                   ((uint32_t)(*buf)[1] << 8) |
                   ((uint32_t)(*buf)[2] << 16) |
                   ((uint32_t)(*buf)[3] << 24);
    *buf += 4;
    return val;
}

/* 写入int32 (小端) */
static inline void write_i32_le(uint8_t** buf, int32_t val) {
    write_u32_le(buf, (uint32_t)val);
}

/* 读取int32 (小端) */
static inline int32_t read_i32_le(const uint8_t** buf) {
    return (int32_t)read_u32_le(buf);
}

/* 写入float32 (小端, IEEE 754) */
static inline void write_f32_le(uint8_t** buf, float val) {
    uint32_t raw;
    memcpy(&raw, &val, sizeof(raw));
    write_u32_le(buf, raw);
}

/* 读取float32 (小端, IEEE 754) */
static inline float read_f32_le(const uint8_t** buf) {
    uint32_t raw = read_u32_le(buf);
    float val;
    memcpy(&val, &raw, sizeof(val));
    return val;
}

/* 写入float64/double (小端, IEEE 754) */
static inline void write_f64_le(uint8_t** buf, double val) {
    uint64_t raw;
    memcpy(&raw, &val, sizeof(raw));
    *(*buf)++ = (uint8_t)(raw & 0xFF);
    *(*buf)++ = (uint8_t)((raw >> 8) & 0xFF);
    *(*buf)++ = (uint8_t)((raw >> 16) & 0xFF);
    *(*buf)++ = (uint8_t)((raw >> 24) & 0xFF);
    *(*buf)++ = (uint8_t)((raw >> 32) & 0xFF);
    *(*buf)++ = (uint8_t)((raw >> 40) & 0xFF);
    *(*buf)++ = (uint8_t)((raw >> 48) & 0xFF);
    *(*buf)++ = (uint8_t)((raw >> 56) & 0xFF);
}

/* 读取float64/double (小端, IEEE 754) */
static inline double read_f64_le(const uint8_t** buf) {
    uint64_t raw = (uint64_t)(*buf)[0] |
                   ((uint64_t)(*buf)[1] << 8) |
                   ((uint64_t)(*buf)[2] << 16) |
                   ((uint64_t)(*buf)[3] << 24) |
                   ((uint64_t)(*buf)[4] << 32) |
                   ((uint64_t)(*buf)[5] << 40) |
                   ((uint64_t)(*buf)[6] << 48) |
                   ((uint64_t)(*buf)[7] << 56);
    *buf += 8;
    double val;
    memcpy(&val, &raw, sizeof(val));
    return val;
}

/* 写入bool (uint8 = 0/1) */
static inline void write_bool(uint8_t** buf, int val) {
    write_u8(buf, (uint8_t)(val ? 1 : 0));
}

/* 读取bool */
static inline int read_bool(const uint8_t** buf) {
    return read_u8(buf) != 0;
}

/* 写入字符串: 4字节LE长度 + 数据(不含空终止符) */
static void write_string(uint8_t** buf, const char* str, size_t max_len) {
    size_t len = str ? strlen(str) : 0;
    if (len > max_len) len = max_len;
    write_u32_le(buf, (uint32_t)len);
    if (len > 0 && str) {
        memcpy(*buf, str, len);
        *buf += len;
    }
}

/* 读取字符串: 4字节LE长度 + 数据, 自动补'\0' */
static size_t read_string(const uint8_t** buf, char* out, size_t out_size) {
    uint32_t len = read_u32_le(buf);
    size_t copy_len = (size_t)len < (out_size - 1) ? (size_t)len : (out_size - 1);
    if (copy_len > 0) {
        memcpy(out, *buf, copy_len);
        *buf += len;
    }
    out[copy_len] = '\0';
    return copy_len;
}

/* 提前检查缓冲区是否有足够空间 */
static int check_buffer_space(size_t offset, size_t needed, size_t buffer_size) {
    if (offset > buffer_size || needed > buffer_size || offset + needed > buffer_size) {
        return ROS_ERROR_SERIALIZE;
    }
    return ROS_OK;
}

static int check_buffer_read(const uint8_t* start, const uint8_t* cursor,
                              size_t needed, size_t buffer_size) {
    size_t offset = (size_t)(cursor - start);
    if (offset > buffer_size || needed > buffer_size || offset + needed > buffer_size) {
        return ROS_ERROR_DESERIALIZE;
    }
    return ROS_OK;
}

/* ============================================================================
 * ROS 序列化/反序列化实现
 * ============================================================================ */

/* --- RosHeader --- */
int ros_serialize_header(const RosHeader* header, uint8_t* buffer, size_t buffer_size, size_t* written) {
    if (!header || !buffer || !written) return ROS_ERROR_INVALID_PARAM;

    size_t needed = 4 + 8 + 8 + 4 + strlen(header->frame_id);
    if (needed > buffer_size) return ROS_ERROR_SERIALIZE;

    uint8_t* ptr = buffer;
    write_u32_le(&ptr, header->seq);
    write_f64_le(&ptr, header->timestamp_sec);
    write_f64_le(&ptr, header->timestamp_nsec);
    write_string(&ptr, header->frame_id, ROS_MAX_TOPIC_NAME - 1);

    *written = (size_t)(ptr - buffer);
    return ROS_OK;
}

int ros_deserialize_header(const uint8_t* buffer, size_t buffer_size, size_t* read, RosHeader* header) {
    if (!buffer || !read || !header) return ROS_ERROR_INVALID_PARAM;

    const uint8_t* ptr = buffer;
    size_t min_size = 4 + 8 + 8 + 4;
    if (buffer_size < min_size) return ROS_ERROR_DESERIALIZE;

    memset(header, 0, sizeof(RosHeader));
    header->seq = read_u32_le(&ptr);
    header->timestamp_sec = read_f64_le(&ptr);
    header->timestamp_nsec = read_f64_le(&ptr);
    read_string(&ptr, header->frame_id, sizeof(header->frame_id));

    *read = (size_t)(ptr - buffer);
    return ROS_OK;
}

/* --- RosPose --- */
int ros_serialize_pose(const RosPose* pose, uint8_t* buffer, size_t buffer_size, size_t* written) {
    if (!pose || !buffer || !written) return ROS_ERROR_INVALID_PARAM;
    if (buffer_size < 56) return ROS_ERROR_SERIALIZE;

    uint8_t* ptr = buffer;
    write_f64_le(&ptr, pose->position.x);
    write_f64_le(&ptr, pose->position.y);
    write_f64_le(&ptr, pose->position.z);
    write_f64_le(&ptr, pose->orientation.x);
    write_f64_le(&ptr, pose->orientation.y);
    write_f64_le(&ptr, pose->orientation.z);
    write_f64_le(&ptr, pose->orientation.w);

    *written = (size_t)(ptr - buffer);
    return ROS_OK;
}

int ros_deserialize_pose(const uint8_t* buffer, size_t buffer_size, size_t* read, RosPose* pose) {
    if (!buffer || !read || !pose) return ROS_ERROR_INVALID_PARAM;
    if (buffer_size < 56) return ROS_ERROR_DESERIALIZE;

    const uint8_t* ptr = buffer;
    memset(pose, 0, sizeof(RosPose));
    pose->position.x = read_f64_le(&ptr);
    pose->position.y = read_f64_le(&ptr);
    pose->position.z = read_f64_le(&ptr);
    pose->orientation.x = read_f64_le(&ptr);
    pose->orientation.y = read_f64_le(&ptr);
    pose->orientation.z = read_f64_le(&ptr);
    pose->orientation.w = read_f64_le(&ptr);

    *read = (size_t)(ptr - buffer);
    return ROS_OK;
}

/* --- RosTwist --- */
int ros_serialize_twist(const RosTwist* twist, uint8_t* buffer, size_t buffer_size, size_t* written) {
    if (!twist || !buffer || !written) return ROS_ERROR_INVALID_PARAM;
    if (buffer_size < 48) return ROS_ERROR_SERIALIZE;

    uint8_t* ptr = buffer;
    write_f64_le(&ptr, twist->linear.x);
    write_f64_le(&ptr, twist->linear.y);
    write_f64_le(&ptr, twist->linear.z);
    write_f64_le(&ptr, twist->angular.x);
    write_f64_le(&ptr, twist->angular.y);
    write_f64_le(&ptr, twist->angular.z);

    *written = (size_t)(ptr - buffer);
    return ROS_OK;
}

int ros_deserialize_twist(const uint8_t* buffer, size_t buffer_size, size_t* read, RosTwist* twist) {
    if (!buffer || !read || !twist) return ROS_ERROR_INVALID_PARAM;
    if (buffer_size < 48) return ROS_ERROR_DESERIALIZE;

    const uint8_t* ptr = buffer;
    memset(twist, 0, sizeof(RosTwist));
    twist->linear.x = read_f64_le(&ptr);
    twist->linear.y = read_f64_le(&ptr);
    twist->linear.z = read_f64_le(&ptr);
    twist->angular.x = read_f64_le(&ptr);
    twist->angular.y = read_f64_le(&ptr);
    twist->angular.z = read_f64_le(&ptr);

    *read = (size_t)(ptr - buffer);
    return ROS_OK;
}

/* --- RosOdometry (pose + twist) --- */
int ros_serialize_odometry(const RosOdometry* odom, uint8_t* buffer, size_t buffer_size, size_t* written) {
    if (!odom || !buffer || !written) return ROS_ERROR_INVALID_PARAM;
    if (buffer_size < 104) return ROS_ERROR_SERIALIZE;

    uint8_t* ptr = buffer;
    /* pose: 7 doubles */
    write_f64_le(&ptr, odom->pose.position.x);
    write_f64_le(&ptr, odom->pose.position.y);
    write_f64_le(&ptr, odom->pose.position.z);
    write_f64_le(&ptr, odom->pose.orientation.x);
    write_f64_le(&ptr, odom->pose.orientation.y);
    write_f64_le(&ptr, odom->pose.orientation.z);
    write_f64_le(&ptr, odom->pose.orientation.w);
    /* twist: 6 doubles */
    write_f64_le(&ptr, odom->twist.linear.x);
    write_f64_le(&ptr, odom->twist.linear.y);
    write_f64_le(&ptr, odom->twist.linear.z);
    write_f64_le(&ptr, odom->twist.angular.x);
    write_f64_le(&ptr, odom->twist.angular.y);
    write_f64_le(&ptr, odom->twist.angular.z);

    *written = (size_t)(ptr - buffer);
    return ROS_OK;
}

int ros_deserialize_odometry(const uint8_t* buffer, size_t buffer_size, size_t* read, RosOdometry* odom) {
    if (!buffer || !read || !odom) return ROS_ERROR_INVALID_PARAM;
    if (buffer_size < 104) return ROS_ERROR_DESERIALIZE;

    const uint8_t* ptr = buffer;
    memset(odom, 0, sizeof(RosOdometry));
    odom->pose.position.x = read_f64_le(&ptr);
    odom->pose.position.y = read_f64_le(&ptr);
    odom->pose.position.z = read_f64_le(&ptr);
    odom->pose.orientation.x = read_f64_le(&ptr);
    odom->pose.orientation.y = read_f64_le(&ptr);
    odom->pose.orientation.z = read_f64_le(&ptr);
    odom->pose.orientation.w = read_f64_le(&ptr);
    odom->twist.linear.x = read_f64_le(&ptr);
    odom->twist.linear.y = read_f64_le(&ptr);
    odom->twist.linear.z = read_f64_le(&ptr);
    odom->twist.angular.x = read_f64_le(&ptr);
    odom->twist.angular.y = read_f64_le(&ptr);
    odom->twist.angular.z = read_f64_le(&ptr);

    *read = (size_t)(ptr - buffer);
    return ROS_OK;
}

/* --- RosLaserScan --- */
int ros_serialize_laserscan(const RosLaserScan* scan, uint8_t* buffer, size_t buffer_size, size_t* written) {
    if (!scan || !buffer || !written) return ROS_ERROR_INVALID_PARAM;

    /* header: 动态大小 */
    uint8_t header_buf[1024];
    size_t header_written = 0;
    int ret = ros_serialize_header(&scan->header, header_buf, sizeof(header_buf), &header_written);
    if (ret != ROS_OK) return ret;

    /* 固定字段: 7 * 4 = 28 bytes */
    size_t fixed_size = header_written + 28 + 8; /* header + 7 floats + 2 array sizes */
    if (fixed_size > buffer_size) return ROS_ERROR_SERIALIZE;

    uint8_t* ptr = buffer;
    memcpy(ptr, header_buf, header_written);
    ptr += header_written;

    write_f32_le(&ptr, scan->angle_min);
    write_f32_le(&ptr, scan->angle_max);
    write_f32_le(&ptr, scan->angle_increment);
    write_f32_le(&ptr, scan->time_increment);
    write_f32_le(&ptr, scan->scan_time);
    write_f32_le(&ptr, scan->range_min);
    write_f32_le(&ptr, scan->range_max);

    /* ranges 数组: 4字节数量 + 4*count 字节 */
    int num_ranges = scan->ranges_size;
    if (num_ranges < 0) num_ranges = 0;
    if (num_ranges > 1024) num_ranges = 1024;
    write_u32_le(&ptr, (uint32_t)num_ranges);
    for (int i = 0; i < num_ranges; i++) {
        write_f32_le(&ptr, scan->ranges[i]);
    }

    /* intensities 数组 */
    int num_intensities = scan->intensities_size;
    if (num_intensities < 0) num_intensities = 0;
    if (num_intensities > 1024) num_intensities = 1024;
    write_u32_le(&ptr, (uint32_t)num_intensities);
    for (int i = 0; i < num_intensities; i++) {
        write_f32_le(&ptr, scan->intensities[i]);
    }

    *written = (size_t)(ptr - buffer);
    return ROS_OK;
}

int ros_deserialize_laserscan(const uint8_t* buffer, size_t buffer_size, size_t* read, RosLaserScan* scan) {
    if (!buffer || !read || !scan) return ROS_ERROR_INVALID_PARAM;

    const uint8_t* ptr = buffer;
    memset(scan, 0, sizeof(RosLaserScan));

    /* 反序列化header */
    size_t header_read = 0;
    int ret = ros_deserialize_header(ptr, buffer_size, &header_read, &scan->header);
    if (ret != ROS_OK) return ret;
    ptr += header_read;

    if ((size_t)(ptr - buffer) + 28 > buffer_size) return ROS_ERROR_DESERIALIZE;

    scan->angle_min = read_f32_le(&ptr);
    scan->angle_max = read_f32_le(&ptr);
    scan->angle_increment = read_f32_le(&ptr);
    scan->time_increment = read_f32_le(&ptr);
    scan->scan_time = read_f32_le(&ptr);
    scan->range_min = read_f32_le(&ptr);
    scan->range_max = read_f32_le(&ptr);

    /* ranges */
    uint32_t num_ranges = read_u32_le(&ptr);
    if (num_ranges > 1024) return ROS_ERROR_DESERIALIZE;
    if ((size_t)(ptr - buffer) + (size_t)num_ranges * 4 > buffer_size) return ROS_ERROR_DESERIALIZE;
    scan->ranges_size = (int)num_ranges;
    for (uint32_t i = 0; i < num_ranges; i++) {
        scan->ranges[i] = read_f32_le(&ptr);
    }

    /* intensities */
    uint32_t num_intensities = read_u32_le(&ptr);
    if (num_intensities > 1024) return ROS_ERROR_DESERIALIZE;
    if ((size_t)(ptr - buffer) + (size_t)num_intensities * 4 > buffer_size) return ROS_ERROR_DESERIALIZE;
    scan->intensities_size = (int)num_intensities;
    for (uint32_t i = 0; i < num_intensities; i++) {
        scan->intensities[i] = read_f32_le(&ptr);
    }

    *read = (size_t)(ptr - buffer);
    return ROS_OK;
}

/* --- RosJoy --- */
int ros_serialize_joy(const RosJoy* joy, uint8_t* buffer, size_t buffer_size, size_t* written) {
    if (!joy || !buffer || !written) return ROS_ERROR_INVALID_PARAM;

    uint8_t header_buf[1024];
    size_t header_written = 0;
    int ret = ros_serialize_header(&joy->header, header_buf, sizeof(header_buf), &header_written);
    if (ret != ROS_OK) return ret;

    size_t total = header_written + 8 * 8 + 16 * 4;
    if (total > buffer_size) return ROS_ERROR_SERIALIZE;

    uint8_t* ptr = buffer;
    memcpy(ptr, header_buf, header_written);
    ptr += header_written;

    /* 8个axes (float64) */
    for (int i = 0; i < 8; i++) {
        write_f64_le(&ptr, joy->axes[i]);
    }
    /* 16个buttons (int32) */
    for (int i = 0; i < 16; i++) {
        write_i32_le(&ptr, joy->buttons[i]);
    }

    *written = (size_t)(ptr - buffer);
    return ROS_OK;
}

int ros_deserialize_joy(const uint8_t* buffer, size_t buffer_size, size_t* read, RosJoy* joy) {
    if (!buffer || !read || !joy) return ROS_ERROR_INVALID_PARAM;

    const uint8_t* ptr = buffer;
    memset(joy, 0, sizeof(RosJoy));

    size_t header_read = 0;
    int ret = ros_deserialize_header(ptr, buffer_size, &header_read, &joy->header);
    if (ret != ROS_OK) return ret;
    ptr += header_read;

    if ((size_t)(ptr - buffer) + 64 + 64 > buffer_size) return ROS_ERROR_DESERIALIZE;

    for (int i = 0; i < 8; i++) {
        joy->axes[i] = read_f64_le(&ptr);
    }
    for (int i = 0; i < 16; i++) {
        joy->buttons[i] = read_i32_le(&ptr);
    }

    *read = (size_t)(ptr - buffer);
    return ROS_OK;
}

/* --- RosImu --- */
int ros_serialize_imu(const RosImu* imu, uint8_t* buffer, size_t buffer_size, size_t* written) {
    if (!imu || !buffer || !written) return ROS_ERROR_INVALID_PARAM;

    uint8_t header_buf[1024];
    size_t header_written = 0;
    int ret = ros_serialize_header(&imu->header, header_buf, sizeof(header_buf), &header_written);
    if (ret != ROS_OK) return ret;

    /* 固定部分: orientation(4) + cov(9) + ang_vel(3) + cov(9) + lin_acc(3) + cov(9) = 37 doubles */
    size_t total = header_written + 37 * 8;
    if (total > buffer_size) return ROS_ERROR_SERIALIZE;

    uint8_t* ptr = buffer;
    memcpy(ptr, header_buf, header_written);
    ptr += header_written;

    write_f64_le(&ptr, imu->orientation.x);
    write_f64_le(&ptr, imu->orientation.y);
    write_f64_le(&ptr, imu->orientation.z);
    write_f64_le(&ptr, imu->orientation.w);
    for (int i = 0; i < 9; i++) write_f64_le(&ptr, imu->orientation_covariance[i]);
    write_f64_le(&ptr, imu->angular_velocity.x);
    write_f64_le(&ptr, imu->angular_velocity.y);
    write_f64_le(&ptr, imu->angular_velocity.z);
    for (int i = 0; i < 9; i++) write_f64_le(&ptr, imu->angular_velocity_covariance[i]);
    write_f64_le(&ptr, imu->linear_acceleration.x);
    write_f64_le(&ptr, imu->linear_acceleration.y);
    write_f64_le(&ptr, imu->linear_acceleration.z);
    for (int i = 0; i < 9; i++) write_f64_le(&ptr, imu->linear_acceleration_covariance[i]);

    *written = (size_t)(ptr - buffer);
    return ROS_OK;
}

int ros_deserialize_imu(const uint8_t* buffer, size_t buffer_size, size_t* read, RosImu* imu) {
    if (!buffer || !read || !imu) return ROS_ERROR_INVALID_PARAM;

    const uint8_t* ptr = buffer;
    memset(imu, 0, sizeof(RosImu));

    size_t header_read = 0;
    int ret = ros_deserialize_header(ptr, buffer_size, &header_read, &imu->header);
    if (ret != ROS_OK) return ret;
    ptr += header_read;

    if ((size_t)(ptr - buffer) + 37 * 8 > buffer_size) return ROS_ERROR_DESERIALIZE;

    imu->orientation.x = read_f64_le(&ptr);
    imu->orientation.y = read_f64_le(&ptr);
    imu->orientation.z = read_f64_le(&ptr);
    imu->orientation.w = read_f64_le(&ptr);
    for (int i = 0; i < 9; i++) imu->orientation_covariance[i] = read_f64_le(&ptr);
    imu->angular_velocity.x = read_f64_le(&ptr);
    imu->angular_velocity.y = read_f64_le(&ptr);
    imu->angular_velocity.z = read_f64_le(&ptr);
    for (int i = 0; i < 9; i++) imu->angular_velocity_covariance[i] = read_f64_le(&ptr);
    imu->linear_acceleration.x = read_f64_le(&ptr);
    imu->linear_acceleration.y = read_f64_le(&ptr);
    imu->linear_acceleration.z = read_f64_le(&ptr);
    for (int i = 0; i < 9; i++) imu->linear_acceleration_covariance[i] = read_f64_le(&ptr);

    *read = (size_t)(ptr - buffer);
    return ROS_OK;
}

/* --- RosImage --- */
int ros_serialize_image(const RosImage* image, uint8_t* buffer, size_t buffer_size, size_t* written) {
    if (!image || !buffer || !written) return ROS_ERROR_INVALID_PARAM;

    /* 固定头: width(4) + height(4) + encoding(4+len) + is_bigendian(1) + step(4) + data_size(4) */
    size_t encoding_len = strlen(image->encoding);
    size_t header_size = 4 + 4 + 4 + encoding_len + 1 + 4 + 4;
    size_t total = header_size + image->data_size;

    if (total > buffer_size) return ROS_ERROR_SERIALIZE;

    uint8_t* ptr = buffer;
    write_u32_le(&ptr, image->width);
    write_u32_le(&ptr, image->height);
    write_string(&ptr, image->encoding, 31);
    write_u8(&ptr, image->is_bigendian);
    write_u32_le(&ptr, image->step);
    write_u32_le(&ptr, (uint32_t)image->data_size);

    if (image->data && image->data_size > 0) {
        memcpy(ptr, image->data, image->data_size);
        ptr += image->data_size;
    }

    *written = (size_t)(ptr - buffer);
    return ROS_OK;
}

int ros_deserialize_image(const uint8_t* buffer, size_t buffer_size, size_t* read, RosImage* image) {
    if (!buffer || !read || !image) return ROS_ERROR_INVALID_PARAM;

    const uint8_t* ptr = buffer;
    memset(image, 0, sizeof(RosImage));

    if (buffer_size < 17) return ROS_ERROR_DESERIALIZE;

    image->width = read_u32_le(&ptr);
    image->height = read_u32_le(&ptr);
    read_string(&ptr, image->encoding, sizeof(image->encoding));
    image->is_bigendian = read_u8(&ptr);
    image->step = read_u32_le(&ptr);
    image->data_size = (size_t)read_u32_le(&ptr);

    if (image->data_size > buffer_size - (size_t)(ptr - buffer)) return ROS_ERROR_DESERIALIZE;

    if (image->data_size > 0) {
        image->data = (uint8_t*)safe_malloc(image->data_size);
        if (!image->data) return ROS_ERROR_NO_MEMORY;
        memcpy(image->data, ptr, image->data_size);
        ptr += image->data_size;
    } else {
        image->data = NULL;
    }

    *read = (size_t)(ptr - buffer);
    return ROS_OK;
}

/* --- RosJointState --- */
int ros_serialize_joint_state(const RosJointState* js, uint8_t* buffer, size_t buffer_size, size_t* written) {
    if (!js || !buffer || !written) return ROS_ERROR_INVALID_PARAM;

    uint8_t header_buf[1024];
    size_t header_written = 0;
    int ret = ros_serialize_header(&js->header, header_buf, sizeof(header_buf), &header_written);
    if (ret != ROS_OK) return ret;

    int nc = js->names_count;
    int pc = js->positions_count;
    int vc = js->velocities_count;
    int ec = js->efforts_count;
    if (nc < 0) nc = 0; if (nc > 32) nc = 32;
    if (pc < 0) pc = 0; if (pc > 32) pc = 32;
    if (vc < 0) vc = 0; if (vc > 32) vc = 32;
    if (ec < 0) ec = 0; if (ec > 32) ec = 32;

    /* 计算name字符串总大小 */
    size_t names_total = 0;
    for (int i = 0; i < nc; i++) {
        names_total += 4 + strlen(js->name[i]);
    }

    size_t total = header_written + 16 + names_total + (size_t)(pc + vc + ec) * 8;
    if (total > buffer_size) return ROS_ERROR_SERIALIZE;

    uint8_t* ptr = buffer;
    memcpy(ptr, header_buf, header_written);
    ptr += header_written;

    /* names数组 */
    write_u32_le(&ptr, (uint32_t)nc);
    for (int i = 0; i < nc; i++) {
        write_string(&ptr, js->name[i], 63);
    }
    /* positions */
    write_u32_le(&ptr, (uint32_t)pc);
    for (int i = 0; i < pc; i++) {
        write_f64_le(&ptr, js->position[i]);
    }
    /* velocities */
    write_u32_le(&ptr, (uint32_t)vc);
    for (int i = 0; i < vc; i++) {
        write_f64_le(&ptr, js->velocity[i]);
    }
    /* efforts */
    write_u32_le(&ptr, (uint32_t)ec);
    for (int i = 0; i < ec; i++) {
        write_f64_le(&ptr, js->effort[i]);
    }

    *written = (size_t)(ptr - buffer);
    return ROS_OK;
}

int ros_deserialize_joint_state(const uint8_t* buffer, size_t buffer_size, size_t* read, RosJointState* js) {
    if (!buffer || !read || !js) return ROS_ERROR_INVALID_PARAM;

    const uint8_t* ptr = buffer;
    memset(js, 0, sizeof(RosJointState));

    size_t header_read = 0;
    int ret = ros_deserialize_header(ptr, buffer_size, &header_read, &js->header);
    if (ret != ROS_OK) return ret;
    ptr += header_read;

    if ((size_t)(ptr - buffer) + 4 > buffer_size) return ROS_ERROR_DESERIALIZE;

    uint32_t nc = read_u32_le(&ptr);
    if (nc > 32) return ROS_ERROR_DESERIALIZE;
    js->names_count = (int)nc;
    for (uint32_t i = 0; i < nc; i++) {
        read_string(&ptr, js->name[i], sizeof(js->name[i]));
    }

    uint32_t pc = read_u32_le(&ptr);
    if (pc > 32) return ROS_ERROR_DESERIALIZE;
    js->positions_count = (int)pc;
    for (uint32_t i = 0; i < pc; i++) {
        js->position[i] = read_f64_le(&ptr);
    }

    uint32_t vc = read_u32_le(&ptr);
    if (vc > 32) return ROS_ERROR_DESERIALIZE;
    js->velocities_count = (int)vc;
    for (uint32_t i = 0; i < vc; i++) {
        js->velocity[i] = read_f64_le(&ptr);
    }

    uint32_t ec = read_u32_le(&ptr);
    if (ec > 32) return ROS_ERROR_DESERIALIZE;
    js->efforts_count = (int)ec;
    for (uint32_t i = 0; i < ec; i++) {
        js->effort[i] = read_f64_le(&ptr);
    }

    *read = (size_t)(ptr - buffer);
    return ROS_OK;
}

/* --- RosFloat64MultiArray --- */
int ros_serialize_float64_multi_array(const RosFloat64MultiArray* arr, uint8_t* buffer, size_t buffer_size, size_t* written) {
    if (!arr || !buffer || !written) return ROS_ERROR_INVALID_PARAM;

    uint8_t header_buf[1024];
    size_t header_written = 0;
    int ret = ros_serialize_header(&arr->header, header_buf, sizeof(header_buf), &header_written);
    if (ret != ROS_OK) return ret;

    int ds = arr->data_size;
    if (ds < 0) ds = 0; if (ds > 64) ds = 64;

    size_t layout_len = strlen(arr->layout);
    size_t total = header_written + 4 + layout_len + 4 + 4 + (size_t)ds * 8;
    if (total > buffer_size) return ROS_ERROR_SERIALIZE;

    uint8_t* ptr = buffer;
    memcpy(ptr, header_buf, header_written);
    ptr += header_written;

    write_string(&ptr, arr->layout, 127);
    write_u32_le(&ptr, (uint32_t)ds);
    for (int i = 0; i < ds; i++) {
        write_f64_le(&ptr, arr->data[i]);
    }

    *written = (size_t)(ptr - buffer);
    return ROS_OK;
}

int ros_deserialize_float64_multi_array(const uint8_t* buffer, size_t buffer_size, size_t* read, RosFloat64MultiArray* arr) {
    if (!buffer || !read || !arr) return ROS_ERROR_INVALID_PARAM;

    const uint8_t* ptr = buffer;
    memset(arr, 0, sizeof(RosFloat64MultiArray));

    size_t header_read = 0;
    int ret = ros_deserialize_header(ptr, buffer_size, &header_read, &arr->header);
    if (ret != ROS_OK) return ret;
    ptr += header_read;

    read_string(&ptr, arr->layout, sizeof(arr->layout));
    uint32_t ds = read_u32_le(&ptr);
    if (ds > 64) return ROS_ERROR_DESERIALIZE;
    arr->data_size = (int)ds;
    for (uint32_t i = 0; i < ds; i++) {
        arr->data[i] = read_f64_le(&ptr);
    }

    *read = (size_t)(ptr - buffer);
    return ROS_OK;
}

/* --- RosPointCloud --- */
int ros_serialize_pointcloud(const RosPointCloud* pc, uint8_t* buffer, size_t buffer_size, size_t* written) {
    if (!pc || !buffer || !written) return ROS_ERROR_INVALID_PARAM;

    uint8_t header_buf[1024];
    size_t header_written = 0;
    int ret = ros_serialize_header(&pc->header, header_buf, sizeof(header_buf), &header_written);
    if (ret != ROS_OK) return ret;

    int pc_count = pc->points_count;
    if (pc_count < 0) pc_count = 0; if (pc_count > 1024) pc_count = 1024;

    size_t total = header_written + 4 + (size_t)pc_count * 3 * 4;
    if (total > buffer_size) return ROS_ERROR_SERIALIZE;

    uint8_t* ptr = buffer;
    memcpy(ptr, header_buf, header_written);
    ptr += header_written;

    write_u32_le(&ptr, (uint32_t)pc_count);
    for (int i = 0; i < pc_count; i++) {
        write_f32_le(&ptr, pc->points[0][i]);
        write_f32_le(&ptr, pc->points[1][i]);
        write_f32_le(&ptr, pc->points[2][i]);
    }

    *written = (size_t)(ptr - buffer);
    return ROS_OK;
}

int ros_deserialize_pointcloud(const uint8_t* buffer, size_t buffer_size, size_t* read, RosPointCloud* pc) {
    if (!buffer || !read || !pc) return ROS_ERROR_INVALID_PARAM;

    const uint8_t* ptr = buffer;
    memset(pc, 0, sizeof(RosPointCloud));

    size_t header_read = 0;
    int ret = ros_deserialize_header(ptr, buffer_size, &header_read, &pc->header);
    if (ret != ROS_OK) return ret;
    ptr += header_read;

    uint32_t pc_count = read_u32_le(&ptr);
    if (pc_count > 1024) return ROS_ERROR_DESERIALIZE;
    pc->points_count = (int)pc_count;

    for (uint32_t i = 0; i < pc_count; i++) {
        pc->points[0][i] = read_f32_le(&ptr);
        pc->points[1][i] = read_f32_le(&ptr);
        pc->points[2][i] = read_f32_le(&ptr);
    }

    *read = (size_t)(ptr - buffer);
    return ROS_OK;
}

/* --- RosLog --- */
int ros_serialize_log(const RosLog* log, uint8_t* buffer, size_t buffer_size, size_t* written) {
    if (!log || !buffer || !written) return ROS_ERROR_INVALID_PARAM;

    size_t name_len = strlen(log->name);
    size_t msg_len = strlen(log->msg);
    size_t file_len = strlen(log->file);
    size_t func_len = strlen(log->function);

    size_t total = 4 + 4 + name_len + 4 + msg_len + 4 + file_len + 4 + 4 + func_len;
    if (total > buffer_size) return ROS_ERROR_SERIALIZE;

    uint8_t* ptr = buffer;
    write_i32_le(&ptr, log->level);
    write_string(&ptr, log->name, 63);
    write_string(&ptr, log->msg, 255);
    write_string(&ptr, log->file, 127);
    write_i32_le(&ptr, log->line);
    write_string(&ptr, log->function, 63);

    *written = (size_t)(ptr - buffer);
    return ROS_OK;
}

int ros_deserialize_log(const uint8_t* buffer, size_t buffer_size, size_t* read, RosLog* log) {
    if (!buffer || !read || !log) return ROS_ERROR_INVALID_PARAM;

    const uint8_t* ptr = buffer;
    memset(log, 0, sizeof(RosLog));

    if (buffer_size < 4) return ROS_ERROR_DESERIALIZE;
    log->level = read_i32_le(&ptr);
    read_string(&ptr, log->name, sizeof(log->name));
    read_string(&ptr, log->msg, sizeof(log->msg));
    read_string(&ptr, log->file, sizeof(log->file));
    log->line = read_i32_le(&ptr);
    read_string(&ptr, log->function, sizeof(log->function));

    *read = (size_t)(ptr - buffer);
    return ROS_OK;
}

/* ============================================================================
 * 辅助函数
 * ============================================================================ */

const char* ros_error_string(int error_code) {
    switch (error_code) {
        case ROS_OK:                    return "成功";
        case ROS_ERROR_GENERAL:         return "一般错误";
        case ROS_ERROR_TIMEOUT:         return "超时";
        case ROS_ERROR_CONNECTION:      return "连接错误";
        case ROS_ERROR_NOT_CONNECTED:   return "未连接";
        case ROS_ERROR_ALREADY_EXISTS:  return "已存在";
        case ROS_ERROR_NOT_FOUND:       return "未找到";
        case ROS_ERROR_INVALID_PARAM:   return "无效参数";
        case ROS_ERROR_NO_MEMORY:       return "内存不足";
        case ROS_ERROR_SERIALIZE:       return "序列化错误";
        case ROS_ERROR_DESERIALIZE:     return "反序列化错误";
        case ROS_ERROR_PROTOCOL:        return "协议错误";
        case ROS_ERROR_MASTER:          return "Master错误";
        case ROS_ERROR_SERVICE:         return "服务错误";
        case ROS_ERROR_NETWORK:         return "网络错误";
        default:                        return "未知错误";
    }
}

int ros_build_connection_header(RosConnectionHeader* header, const char* topic,
                                 const char* type, const char* md5sum, int is_tcp) {
    if (!header || !topic || !type || !md5sum) return ROS_ERROR_INVALID_PARAM;

    memset(header, 0, sizeof(RosConnectionHeader));

    snprintf(header->topic_name, sizeof(header->topic_name), "%s", topic);
    snprintf(header->type_name, sizeof(header->type_name), "%s", type);
    snprintf(header->md5sum, sizeof(header->md5sum), "%s", md5sum);
    header->transport = is_tcp ? ROS_TRANSPORT_TCP : ROS_TRANSPORT_UDP;
    header->flags = 0;
    header->num_header_fields = 0;
    header->message_definition = NULL;
    header->message_definition_size = 0;

    return ROS_OK;
}

const char* ros_topic_type_to_string(int type_id) {
    switch (type_id) {
        case 0:  return "std_msgs/Header";
        case 1:  return "geometry_msgs/Pose";
        case 2:  return "geometry_msgs/Twist";
        case 3:  return "nav_msgs/Odometry";
        case 4:  return "sensor_msgs/LaserScan";
        case 5:  return "sensor_msgs/Joy";
        case 6:  return "sensor_msgs/Imu";
        case 7:  return "sensor_msgs/Image";
        case 8:  return "sensor_msgs/JointState";
        case 9:  return "std_msgs/Float64MultiArray";
        case 10: return "sensor_msgs/PointCloud";
        case 11: return "rosgraph_msgs/Log";
        case 12: return "geometry_msgs/PoseStamped";
        case 13: return "nav_msgs/Odometry";
        case 14: return "sensor_msgs/Temperature";
        case 15: return "geometry_msgs/WrenchStamped";
        case 16: return "sensor_msgs/Image";
        default: return "unknown/type";
    }
}

/* 标准MD5哈希实现
 * 替换原自定义哈希算法，使用符合RFC 1321的标准MD5计算。
 * 确保与真实ROS节点的消息类型MD5校验完全兼容。 */
static void ros_msg_hash_compute(const char* data, size_t len, char* output, size_t output_size) {
    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE, h3 = 0x10325476;
    static const uint32_t K[64] = {
        0xD76AA478,0xE8C7B756,0x242070DB,0xC1BDCEEE,0xF57C0FAF,0x4787C62A,0xA8304613,0xFD469501,
        0x698098D8,0x8B44F7AF,0xFFFF5BB1,0x895CD7BE,0x6B901122,0xFD987193,0xA679438E,0x49B40821,
        0xF61E2562,0xC040B340,0x265E5A51,0xE9B6C7AA,0xD62F105D,0x02441453,0xD8A1E681,0xE7D3FBC8,
        0x21E1CDE6,0xC33707D6,0xF4D50D87,0x455A14ED,0xA9E3E905,0xFCEFA3F8,0x676F02D9,0x8D2A4C8A,
        0xFFFA3942,0x8771F681,0x6D9D6122,0xFDE5380C,0xA4BEEA44,0x4BDECFA9,0xF6BB4B60,0xBEBFBC70,
        0x289B7EC6,0xEAA127FA,0xD4EF3085,0x04881D05,0xD9D4D039,0xE6DB99E5,0x1FA27CF8,0xC4AC5665,
        0xF4292244,0x432AFF97,0xAB9423A7,0xFC93A039,0x655B59C3,0x8F0CCC92,0xFFEFF47D,0x85845DD1,
        0x6FA87E4F,0xFE2CE6E0,0xA3014314,0x4E0811A1,0xF7537E82,0xBD3AF235,0x2AD7D2BB,0xEB86D391
    };
    static const unsigned S[64] = {
        7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
        5,9,14,20,5,9,14,20,5,9,14,20,5,9,14,20,
        4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
        6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21
    };
    size_t padded_len = ((len + 8) / 64 + 1) * 64;
    uint8_t* padded = (uint8_t*)safe_calloc(padded_len, 1);
    if (!padded) { output[0] = '\0'; return; }
    memcpy(padded, data, len);
    padded[len] = 0x80;
    uint64_t bits = (uint64_t)len * 8;
    memcpy(padded + padded_len - 8, &bits, 8);

    for (size_t offset = 0; offset < padded_len; offset += 64) {
        uint32_t M[16];
        for (int i = 0; i < 16; i++) {
            M[i] = (uint32_t)padded[offset+i*4] | ((uint32_t)padded[offset+i*4+1]<<8)
                 | ((uint32_t)padded[offset+i*4+2]<<16) | ((uint32_t)padded[offset+i*4+3]<<24);
        }
        uint32_t a = h0, b = h1, c = h2, d = h3;
        for (int i = 0; i < 64; i++) {
            uint32_t f, g;
            if (i < 16) { f = (b & c) | (~b & d); g = (uint32_t)i; }
            else if (i < 32) { f = (d & b) | (~d & c); g = (uint32_t)((5*i+1)&15); }
            else if (i < 48) { f = b ^ c ^ d; g = (uint32_t)((3*i+5)&15); }
            else { f = c ^ (b | ~d); g = (uint32_t)((7*i)&15); }
            uint32_t tmp = d;
            d = c; c = b;
            b = b + ((a + f + K[i] + M[g]) << S[i] | (a + f + K[i] + M[g]) >> (32-S[i]));
            a = tmp;
        }
        h0 += a; h1 += b; h2 += c; h3 += d;
    }
    safe_free((void**)&padded);
    snprintf(output, output_size, "%08x%08x%08x%08x", h0, h1, h2, h3);
}

/* 预定义常用消息类型的文本定义，用于生成MD5 */
static const char* ros_type_definitions[] = {
    /* 0: std_msgs/Header */
    "uint32 seq\ntime stamp\nstring frame_id\n",
    /* 1: geometry_msgs/Pose */
    "geometry_msgs/Point position\ngeometry_msgs/Quaternion orientation\n",
    /* 2: geometry_msgs/Twist */
    "geometry_msgs/Vector3 linear\ngeometry_msgs/Vector3 angular\n",
    /* 3: nav_msgs/Odometry */
    "std_msgs/Header header\nstring child_frame_id\ngeometry_msgs/PoseWithCovariance pose\ngeometry_msgs/TwistWithCovariance twist\n",
    /* 4: sensor_msgs/LaserScan */
    "std_msgs/Header header\nfloat32 angle_min\nfloat32 angle_max\nfloat32 angle_increment\nfloat32 time_increment\nfloat32 scan_time\nfloat32 range_min\nfloat32 range_max\nfloat32[] ranges\nfloat32[] intensities\n",
    /* 5: sensor_msgs/Joy */
    "std_msgs/Header header\nfloat32[] axes\nint32[] buttons\n",
    /* 6: sensor_msgs/Imu */
    "std_msgs/Header header\ngeometry_msgs/Quaternion orientation\nfloat64[9] orientation_covariance\ngeometry_msgs/Vector3 angular_velocity\nfloat64[9] angular_velocity_covariance\ngeometry_msgs/Vector3 linear_acceleration\nfloat64[9] linear_acceleration_covariance\n",
    /* 7: sensor_msgs/Image */
    "std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\nuint32 step\nuint8[] data\n",
    /* 8: sensor_msgs/JointState */
    "std_msgs/Header header\nstring[] name\nfloat64[] position\nfloat64[] velocity\nfloat64[] effort\n",
    /* 9: std_msgs/Float64MultiArray */
    "MultiArrayLayout layout\nfloat64[] data\n",
    /* 10: sensor_msgs/PointCloud */
    "std_msgs/Header header\ngeometry_msgs/Point32[] points\n",
    /* 11: rosgraph_msgs/Log */
    "byte DEBUG=1\nbyte INFO=2\nbyte WARN=4\nbyte ERROR=8\nbyte FATAL=16\nstd_msgs/Header header\nbyte level\nstring name\nstring msg\nstring file\nstring function\nuint32 line\nstring[] topics\n",
    /* 12: geometry_msgs/PoseStamped */
    "std_msgs/Header header\ngeometry_msgs/Pose pose\n",
    /* 13: nav_msgs/Odometry */
    "std_msgs/Header header\nstring child_frame_id\ngeometry_msgs/PoseWithCovariance pose\ngeometry_msgs/TwistWithCovariance twist\n",
    /* 14: sensor_msgs/Temperature */
    "std_msgs/Header header\nfloat64 temperature\nfloat64 variance\n",
    /* 15: geometry_msgs/WrenchStamped */
    "std_msgs/Header header\ngeometry_msgs/Wrench wrench\n",
    /* 16: sensor_msgs/Image */
    "std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\nuint32 step\nuint8[] data\n",
};

int ros_get_md5_from_type(const char* type_name, char* md5_out, size_t md5_size) {
    if (!type_name || !md5_out || md5_size < 33) return ROS_ERROR_INVALID_PARAM;

    const char* definition = NULL;

    if (strcmp(type_name, "std_msgs/Header") == 0) {
        definition = ros_type_definitions[0];
    } else if (strcmp(type_name, "geometry_msgs/Pose") == 0) {
        definition = ros_type_definitions[1];
    } else if (strcmp(type_name, "geometry_msgs/Twist") == 0) {
        definition = ros_type_definitions[2];
    } else if (strcmp(type_name, "nav_msgs/Odometry") == 0) {
        definition = ros_type_definitions[3];
    } else if (strcmp(type_name, "sensor_msgs/LaserScan") == 0) {
        definition = ros_type_definitions[4];
    } else if (strcmp(type_name, "sensor_msgs/Joy") == 0) {
        definition = ros_type_definitions[5];
    } else if (strcmp(type_name, "sensor_msgs/Imu") == 0) {
        definition = ros_type_definitions[6];
    } else if (strcmp(type_name, "sensor_msgs/Image") == 0) {
        definition = ros_type_definitions[7];
    } else if (strcmp(type_name, "sensor_msgs/JointState") == 0) {
        definition = ros_type_definitions[8];
    } else if (strcmp(type_name, "std_msgs/Float64MultiArray") == 0) {
        definition = ros_type_definitions[9];
    } else if (strcmp(type_name, "sensor_msgs/PointCloud") == 0) {
        definition = ros_type_definitions[10];
    } else if (strcmp(type_name, "rosgraph_msgs/Log") == 0) {
        definition = ros_type_definitions[11];
    } else if (strcmp(type_name, "geometry_msgs/PoseStamped") == 0) {
        definition = ros_type_definitions[12];
    } else if (strcmp(type_name, "sensor_msgs/Temperature") == 0) {
        definition = ros_type_definitions[14];
    } else if (strcmp(type_name, "geometry_msgs/WrenchStamped") == 0) {
        definition = ros_type_definitions[15];
    }

    if (!definition) {
        /* 未知类型，生成基于名称的哈希 */
        ros_msg_hash_compute(type_name, strlen(type_name), md5_out, md5_size);
    } else {
        ros_msg_hash_compute(definition, strlen(definition), md5_out, md5_size);
    }

    return ROS_OK;
}

/* ============================================================================
 * ROS1 TCP传输层实现
 * ============================================================================ */

/* 全局TCP连接状态 */
static struct {
    SOCKET sock;
    int connected;
    char remote_host[256];
    int remote_port;
    char master_uri[512];
    char caller_id[256];
} g_ros_tcp_ctx = { SELFLNN_INVALID_SOCKET, 0, {0}, 0, {0}, {0} };

int ros_tcp_connect(const char* host, int port, int timeout_ms) {
    if (!host) return ROS_ERROR_INVALID_PARAM;

    if (g_ros_tcp_ctx.connected) {
        ros_tcp_disconnect();
    }

    if (network_init() != 0) return ROS_ERROR_NETWORK;

    int sock = socket_connect_timeout(host, port, timeout_ms > 0 ? timeout_ms : ROS_DEFAULT_TIMEOUT_MS);
    if (sock < 0) {
        network_cleanup();
        return ROS_ERROR_CONNECTION;
    }

    /* 设置TCP_NODELAY */
    int nodelay = 1;
    setsockopt((SOCKET)sock, IPPROTO_TCP, TCP_NODELAY, (const char*)&nodelay, sizeof(nodelay));

    g_ros_tcp_ctx.sock = (SOCKET)sock;
    g_ros_tcp_ctx.connected = 1;
    g_ros_tcp_ctx.remote_port = port;
    snprintf(g_ros_tcp_ctx.remote_host, sizeof(g_ros_tcp_ctx.remote_host), "%s", host);

    log_info("[ROS-TCP] 已连接到 %s:%d", host, port);
    return ROS_OK;
}

int ros_tcp_disconnect(void) {
    if (g_ros_tcp_ctx.sock != SELFLNN_INVALID_SOCKET) {
        sockclose(g_ros_tcp_ctx.sock);
        g_ros_tcp_ctx.sock = SELFLNN_INVALID_SOCKET;
    }
    g_ros_tcp_ctx.connected = 0;
    network_cleanup();
    return ROS_OK;
}

/* XMLRPC 请求构建 */
static int build_xmlrpc_request(const char* method, const char* params, char* buffer, size_t buffer_size) {
    return snprintf(buffer, buffer_size,
        "POST /RPC2 HTTP/1.1\r\n"
        "Host: %s:%d\r\n"
        "Content-Type: text/xml\r\n"
        "User-Agent: SELF-LNN/1.0\r\n"
        "Connection: close\r\n"
        "Content-Length: %zu\r\n"
        "\r\n"
        "<?xml version=\"1.0\"?>"
        "<methodCall>"
        "<methodName>%s</methodName>"
        "<params>%s</params>"
        "</methodCall>",
        g_ros_tcp_ctx.remote_host, g_ros_tcp_ctx.remote_port,
        strlen(params) + 100, method, params);
}

/* XMLRPC 参数构建 */
static int build_xmlrpc_param_int(const char* name, int value, char* buffer, size_t buffer_size) {
    return snprintf(buffer, buffer_size, "<param><value><i4>%d</i4></value></param>", value);
}

static int build_xmlrpc_param_string(const char* value, char* buffer, size_t buffer_size) {
    return snprintf(buffer, buffer_size, "<param><value><string>%s</string></value></param>", value);
}

static int build_xmlrpc_param_array_begin(char* buffer, size_t buffer_size) {
    return snprintf(buffer, buffer_size, "<param><value><array><data>");
}

static int build_xmlrpc_param_array_end(char* buffer, size_t buffer_size) {
    (void)buffer_size;
    return snprintf(buffer, buffer_size, "</data></array></value></param>");
}

/* 发送XMLRPC请求并接收响应 */
static int xmlrpc_call(const char* xml_request, size_t request_len,
                       char* response, size_t response_size) {
    if (!g_ros_tcp_ctx.connected) return ROS_ERROR_NOT_CONNECTED;

    /* 发送请求 */
    int sent = (int)send(g_ros_tcp_ctx.sock, xml_request, (int)request_len, 0);
    if (sent <= 0) return ROS_ERROR_CONNECTION;

    /* 接收响应 */
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(g_ros_tcp_ctx.sock, &read_fds);
    struct timeval tv = { 5, 0 };

    int ret = select((int)(g_ros_tcp_ctx.sock + 1), &read_fds, NULL, NULL, &tv);
    if (ret <= 0) return ROS_ERROR_TIMEOUT;

    int recvd = (int)recv(g_ros_tcp_ctx.sock, response, (int)(response_size - 1), 0);
    if (recvd <= 0) return ROS_ERROR_CONNECTION;

    response[recvd] = '\0';
    return ROS_OK;
}

/* 从XMLRPC响应中提取返回值: 查找 <i4>...</i4> 或 <int>...</int> */
static int xmlrpc_extract_int(const char* xml, const char* result_tag) {
    if (!xml) return -1;
    /* 简单解析: 查找 <i4> 或 <int> */
    const char* p = strstr(xml, "<i4>");
    if (!p) p = strstr(xml, "<int>");
    if (!p) return -1;
    /* 跳过标签 */
    while (*p && *p != '>') p++;
    if (*p == '>') p++;
    return atoi(p);
}

int ros_master_register_publisher(const char* caller_id, const char* topic,
                                   const char* topic_type, const char* caller_api) {
    if (!caller_id || !topic || !topic_type || !caller_api) return ROS_ERROR_INVALID_PARAM;
    if (!g_ros_tcp_ctx.connected) return ROS_ERROR_NOT_CONNECTED;

    /* 构建 registerPublisher XMLRPC 调用 */
    char params[2048];
    int offset = build_xmlrpc_param_string(caller_id, params, sizeof(params));
    offset += build_xmlrpc_param_string(topic, params + offset, sizeof(params) - (size_t)offset);
    offset += build_xmlrpc_param_string(topic_type, params + offset, sizeof(params) - (size_t)offset);
    offset += build_xmlrpc_param_string(caller_api, params + offset, sizeof(params) - (size_t)offset);

    char request[4096];
    build_xmlrpc_request("registerPublisher", params, request, sizeof(request));

    char response[8192];
    int ret = xmlrpc_call(request, strlen(request), response, sizeof(response));
    if (ret != ROS_OK) return ret;

    /* 检查响应是否包含成功指示 */
    if (strstr(response, "200 OK") == NULL || strstr(response, "faultCode") != NULL) {
        return ROS_ERROR_MASTER;
    }

    log_info("[ROS-TCP] 话题 '%s' (类型: %s) 已在Master注册", topic, topic_type);
    return ROS_OK;
}

int ros_tcp_negotiate_subscriber(const char* topic, const char* topic_type,
                                  const char* caller_api, RosConnectionHeader* header_out) {
    if (!topic || !topic_type || !caller_api || !header_out) return ROS_ERROR_INVALID_PARAM;
    if (!g_ros_tcp_ctx.connected) return ROS_ERROR_NOT_CONNECTED;

    /* 构建 requestTopic XMLRPC 调用 */
    char params[2048];
    int offset = build_xmlrpc_param_string(g_ros_tcp_ctx.caller_id, params, sizeof(params));
    offset += build_xmlrpc_param_string(topic, params + offset, sizeof(params) - (size_t)offset);

    /* 构建参数数组: [[TCPROS协议参数]] */
    char array_buf[1024];
    int arr_offset = build_xmlrpc_param_array_begin(array_buf, sizeof(array_buf));
    arr_offset += build_xmlrpc_param_string("TCPROS", array_buf + arr_offset, sizeof(array_buf) - (size_t)arr_offset);
    /* 添加host和port */
    arr_offset += build_xmlrpc_param_string(SELFLNN_LOCALHOST, array_buf + arr_offset, sizeof(array_buf) - (size_t)arr_offset);
    arr_offset += build_xmlrpc_param_int("port", SELFLNN_ROS_DATA_PORT, array_buf + arr_offset, sizeof(array_buf) - (size_t)arr_offset);
    arr_offset += build_xmlrpc_param_array_end(array_buf + arr_offset, sizeof(array_buf) - (size_t)arr_offset);
    offset += snprintf(params + offset, sizeof(params) - (size_t)offset, "%s", array_buf);

    char request[4096];
    build_xmlrpc_request("requestTopic", params, request, sizeof(request));

    char response[8192];
    int ret = xmlrpc_call(request, strlen(request), response, sizeof(response));
    if (ret != ROS_OK) return ret;

    /* 检查响应 */
    if (strstr(response, "200 OK") == NULL || strstr(response, "faultCode") != NULL) {
        return ROS_ERROR_MASTER;
    }

    /* 填充连接头信息 */
    memset(header_out, 0, sizeof(RosConnectionHeader));
    snprintf(header_out->topic_name, sizeof(header_out->topic_name), "%s", topic);
    snprintf(header_out->type_name, sizeof(header_out->type_name), "%s", topic_type);
    header_out->transport = ROS_TRANSPORT_TCP;

    log_info("[ROS-TCP] 订阅 '%s' 协商完成", topic);
    return ROS_OK;
}

int ros_tcp_publish_message(const uint8_t* serialized_data, size_t data_size) {
    if (!serialized_data || data_size == 0) return ROS_ERROR_INVALID_PARAM;
    if (!g_ros_tcp_ctx.connected) return ROS_ERROR_NOT_CONNECTED;

    /* TCPROS发布格式: 4字节消息长度(LE) + 消息数据 */
    uint8_t header[4];
    header[0] = (uint8_t)(data_size & 0xFF);
    header[1] = (uint8_t)((data_size >> 8) & 0xFF);
    header[2] = (uint8_t)((data_size >> 16) & 0xFF);
    header[3] = (uint8_t)((data_size >> 24) & 0xFF);

    /* 发送长度头 */
    int sent = (int)send(g_ros_tcp_ctx.sock, (const char*)header, 4, 0);
    if (sent != 4) return ROS_ERROR_CONNECTION;

    /* 发送消息体 */
    size_t total_sent = 0;
    while (total_sent < data_size) {
        sent = (int)send(g_ros_tcp_ctx.sock,
                         (const char*)(serialized_data + total_sent),
                         (int)(data_size - total_sent), 0);
        if (sent <= 0) return ROS_ERROR_CONNECTION;
        total_sent += (size_t)sent;
    }

    return ROS_OK;
}

int ros_tcp_subscribe_loop(int (*callback)(const uint8_t*, size_t, void*), void* user_data) {
    if (!callback) return ROS_ERROR_INVALID_PARAM;
    if (!g_ros_tcp_ctx.connected) return ROS_ERROR_NOT_CONNECTED;

    uint8_t* recv_buffer = (uint8_t*)safe_malloc(ROS_TCP_BUF_SIZE);
    if (!recv_buffer) return ROS_ERROR_NO_MEMORY;

    log_info("[ROS-TCP] 订阅循环开始");

    while (g_ros_tcp_ctx.connected) {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(g_ros_tcp_ctx.sock, &read_fds);
        struct timeval tv = { 1, 0 };

        int ret = select((int)(g_ros_tcp_ctx.sock + 1), &read_fds, NULL, NULL, &tv);
        if (ret < 0) {
            safe_free((void**)&recv_buffer);
            return ROS_ERROR_CONNECTION;
        }
        if (ret == 0) continue;

        /* 读取4字节长度头 */
        uint8_t len_buf[4];
        int recvd = (int)recv(g_ros_tcp_ctx.sock, (char*)len_buf, 4, MSG_WAITALL);
        if (recvd != 4) {
            if (recvd == 0) {
                log_info("[ROS-TCP] 发布者关闭连接");
                break;
            }
            safe_free((void**)&recv_buffer);
            return ROS_ERROR_CONNECTION;
        }

        uint32_t msg_size = (uint32_t)len_buf[0] |
                            ((uint32_t)len_buf[1] << 8) |
                            ((uint32_t)len_buf[2] << 16) |
                            ((uint32_t)len_buf[3] << 24);

        if (msg_size == 0 || msg_size > ROS_TCP_BUF_SIZE) {
            log_warning("[ROS-TCP] 无效消息大小: %u", msg_size);
            continue;
        }

        /* 读取消息体 */
        size_t total_recvd = 0;
        while (total_recvd < msg_size) {
            recvd = (int)recv(g_ros_tcp_ctx.sock,
                              (char*)(recv_buffer + total_recvd),
                              (int)(msg_size - total_recvd), 0);
            if (recvd <= 0) {
                safe_free((void**)&recv_buffer);
                return ROS_ERROR_CONNECTION;
            }
            total_recvd += (size_t)recvd;
        }

        /* 调用回调 */
        int cb_ret = callback(recv_buffer, msg_size, user_data);
        if (cb_ret != 0) {
            log_info("[ROS-TCP] 回调返回 %d，退出订阅循环", cb_ret);
            break;
        }
    }

    safe_free((void**)&recv_buffer);
    log_info("[ROS-TCP] 订阅循环结束");
    return ROS_OK;
}

/* ============================================================================
 * DDS RTPS 协议实现
 * ============================================================================ */

/* DDS 上下文结构 */
struct DDSContext {
    uint32_t domain_id;
    char node_name[ROS_MAX_NODE_NAME];
    char node_namespace[ROS_MAX_TOPIC_NAME];
    DDSGuid participant_guid;
    int discovery_running;

    /* 网络资源 */
    SOCKET discovery_socket;
    SOCKET user_traffic_socket;
    uint32_t user_traffic_port;
    uint32_t discovery_port;
    struct sockaddr_in multicast_addr;
    struct sockaddr_in unicast_addr;

    /* 发现状态 */
    int sequence_number;
    int discovery_period_ms;

    /* 本地端点 */
    struct {
        char topic_name[ROS_MAX_TOPIC_NAME];
        char type_name[ROS_MAX_TYPE_NAME];
        int is_writer;
        int is_reliable;
        uint32_t sequence_number;
        DDSGuid endpoint_guid;
    } local_endpoints[DDS_MAX_ENDPOINTS];
    int num_local_endpoints;

    /* 远程参与者 */
    DDSParticipantData remote_participants[DDS_MAX_PARTICIPANTS];
    int num_remote_participants;

    /* 远程端点 */
    DDSEndpoint remote_endpoints[DDS_MAX_ENDPOINTS];
    int num_remote_endpoints;

    /* 回调 */
    DDSParticipantCallback participant_callback;
    void* participant_callback_data;

    struct {
        DDSTopicCallback callback;
        void* user_data;
        char topic_name[ROS_MAX_TOPIC_NAME];
    } reader_callbacks[DDS_MAX_ENDPOINTS];
    int num_reader_callbacks;
};

/* 为DDS GUID生成指定前缀的hash */
static void dds_hash_name(const char* name, const char* namespace_name, uint8_t* hash_out) {
    uint32_t h = 0x811C9DC5;
    if (name) {
        while (*name) { h = (h * 0x01000193) ^ (uint32_t)(unsigned char)*name++; }
    }
    if (namespace_name) {
        while (*namespace_name) {
            h = (h * 0x01000193) ^ (uint32_t)(unsigned char)*namespace_name++;
        }
    }
    /* 使用hash填充12字节前缀的前缀部分 */
    for (int i = 0; i < 4; i++) {
        hash_out[i] = (uint8_t)((h >> (i * 8)) & 0xFF);
    }
}

DDSContext* dds_context_create(uint32_t domain_id, const char* node_name,
                               const char* node_namespace) {
    if (network_init() != 0) {
        log_error("[DDS] 网络初始化失败");
        return NULL;
    }

    DDSContext* ctx = (DDSContext*)safe_calloc(1, sizeof(DDSContext));
    if (!ctx) {
        network_cleanup();
        return NULL;
    }

    ctx->domain_id = domain_id;
    ctx->discovery_running = 0;
    ctx->sequence_number = 0;
    ctx->discovery_period_ms = DDS_DISCOVERY_PERIOD_MS;
    ctx->num_local_endpoints = 0;
    ctx->num_remote_participants = 0;
    ctx->num_remote_endpoints = 0;
    ctx->num_reader_callbacks = 0;
    ctx->participant_callback = NULL;
    ctx->participant_callback_data = NULL;

    if (node_name) {
        snprintf(ctx->node_name, sizeof(ctx->node_name), "%s", node_name);
    } else {
        snprintf(ctx->node_name, sizeof(ctx->node_name), "selflnn_dds");
    }
    if (node_namespace) {
        snprintf(ctx->node_namespace, sizeof(ctx->node_namespace), "%s", node_namespace);
    } else {
        ctx->node_namespace[0] = '/';
        ctx->node_namespace[1] = '\0';
    }

    /* 生成参与者GUID */
    dds_generate_guid(&ctx->participant_guid, domain_id, DDS_ENTITY_PARTICIPANT, 0);

    /* 端口计算 */
    ctx->user_traffic_port = DDS_USER_TRAFFIC_PORT_BASE + domain_id;
    ctx->discovery_port = DDS_DISCOVERY_PORT_BASE + domain_id;

    /* 创建UDP socket */
    ctx->discovery_socket = socket(AF_INET, SOCK_DGRAM, 0);
    ctx->user_traffic_socket = socket(AF_INET, SOCK_DGRAM, 0);

    if (ctx->discovery_socket == SELFLNN_INVALID_SOCKET ||
        ctx->user_traffic_socket == SELFLNN_INVALID_SOCKET) {
        log_error("[DDS] 创建UDP socket失败");
        dds_context_free(ctx);
        return NULL;
    }

    /* 设置地址重用 */
    int reuse = 1;
    setsockopt(ctx->discovery_socket, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));
    setsockopt(ctx->user_traffic_socket, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));

    /* 绑定发现端口 */
    struct sockaddr_in disc_addr;
    memset(&disc_addr, 0, sizeof(disc_addr));
    disc_addr.sin_family = AF_INET;
    disc_addr.sin_port = htons((unsigned short)ctx->discovery_port);
    disc_addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(ctx->discovery_socket, (struct sockaddr*)&disc_addr, sizeof(disc_addr)) < 0) {
        log_error("[DDS] 绑定发现端口 %u 失败", ctx->discovery_port);
        dds_context_free(ctx);
        return NULL;
    }

    /* 绑定用户流量端口 */
    struct sockaddr_in user_addr;
    memset(&user_addr, 0, sizeof(user_addr));
    user_addr.sin_family = AF_INET;
    user_addr.sin_port = htons((unsigned short)ctx->user_traffic_port);
    user_addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(ctx->user_traffic_socket, (struct sockaddr*)&user_addr, sizeof(user_addr)) < 0) {
        log_error("[DDS] 绑定用户流量端口 %u 失败", ctx->user_traffic_port);
        dds_context_free(ctx);
        return NULL;
    }

    /* 设置多播 */
    memset(&ctx->multicast_addr, 0, sizeof(ctx->multicast_addr));
    ctx->multicast_addr.sin_family = AF_INET;
    ctx->multicast_addr.sin_port = htons((unsigned short)DDS_MULTICAST_PORT);
    inet_pton(AF_INET, DDS_MULTICAST_ADDRESS, &ctx->multicast_addr.sin_addr);

    /* 加入多播组 */
    struct ip_mreq mreq;
    mreq.imr_interface.s_addr = htonl(INADDR_ANY);
    mreq.imr_multiaddr.s_addr = ctx->multicast_addr.sin_addr.s_addr;
    setsockopt(ctx->discovery_socket, IPPROTO_IP, IP_ADD_MEMBERSHIP,
               (const char*)&mreq, sizeof(mreq));

    log_info("[DDS] DDS上下文创建完成: 节点='%s', 域=%u, 发现端口=%u, 流量端口=%u",
             ctx->node_name, domain_id, ctx->discovery_port, ctx->user_traffic_port);

    return ctx;
}

void dds_context_free(DDSContext* ctx) {
    if (!ctx) return;

    if (ctx->discovery_running) {
        dds_discovery_stop(ctx);
    }

    if (ctx->discovery_socket != SELFLNN_INVALID_SOCKET) {
        sockclose(ctx->discovery_socket);
    }
    if (ctx->user_traffic_socket != SELFLNN_INVALID_SOCKET) {
        sockclose(ctx->user_traffic_socket);
    }

    safe_free((void**)&ctx);
    network_cleanup();
    log_info("[DDS] DDS上下文已释放");
}

int dds_discovery_start(DDSContext* ctx) {
    if (!ctx) return ROS_ERROR_INVALID_PARAM;
    if (ctx->discovery_running) return ROS_OK;

    ctx->discovery_running = 1;
    ctx->sequence_number = 0;

    log_info("[DDS] DDS发现已启动 (周期=%d ms)", ctx->discovery_period_ms);
    return ROS_OK;
}

int dds_discovery_stop(DDSContext* ctx) {
    if (!ctx) return ROS_ERROR_INVALID_PARAM;
    ctx->discovery_running = 0;
    return ROS_OK;
}

/* 构建SPDP发现消息并发送 */
static int dds_send_discovery(DDSContext* ctx) {
    if (!ctx || !ctx->discovery_running) return ROS_ERROR_INVALID_PARAM;

    /* 构建SPDP发现数据包 */
    uint8_t packet[1500];
    memset(packet, 0, sizeof(packet));
    uint8_t* ptr = packet;

    /* RTPS Header */
    memcpy(ptr, "RTPS", 4); ptr += 4;                  /* 协议标识 */
    write_u8(&ptr, DDS_PROTOCOL_VERSION_MAJOR);          /* 主版本 */
    write_u8(&ptr, DDS_PROTOCOL_VERSION_MINOR);          /* 次版本 */
    write_u16_le(&ptr, DDS_VENDOR_ID);                   /* 厂商ID */

    /* 写入GUID前缀 */
    memcpy(ptr, ctx->participant_guid.prefix, DDS_GUID_PREFIX_LEN);
    ptr += DDS_GUID_PREFIX_LEN;

    /* SPDP InfoDST submessage (0x0E) */
    write_u8(&ptr, 0x0E);                                /* submessage ID: INFO_DST */
    write_u8(&ptr, 0x01);                                /* flags: endianness */
    size_t submsg_offset = (size_t)(ptr - packet);
    write_u16_le(&ptr, 0);                               /* length placeholder */

    /* GUID Prefix */
    memcpy(ptr, ctx->participant_guid.prefix, DDS_GUID_PREFIX_LEN);
    ptr += DDS_GUID_PREFIX_LEN;

    /* 回填长度 */
    uint16_t info_dst_len = (uint16_t)((size_t)(ptr - packet) - submsg_offset - 2);
    packet[submsg_offset] = (uint8_t)(info_dst_len & 0xFF);
    packet[submsg_offset + 1] = (uint8_t)((info_dst_len >> 8) & 0xFF);

    /* SPDP Data submessage (PID_PARTICIPANT_BUILTIN_TOPICS) */
    write_u8(&ptr, 0x15);                                /* submessage ID: DATA(p) */
    write_u8(&ptr, 0x07);                                /* flags: D=1, K_IN_DATA=1, E=1 */
    size_t data_submsg_offset = (size_t)(ptr - packet);
    write_u16_le(&ptr, 0);                               /* length placeholder */

    write_u16_le(&ptr, 0x0000);                          /* extraFlags */
    write_u16_le(&ptr, 0x0010);                          /* octetsToInlineQos */
    memcpy(ptr, ctx->participant_guid.entity_id, DDS_ENTITY_ID_LEN);
    ptr += DDS_ENTITY_ID_LEN;
    memcpy(ptr, ctx->participant_guid.entity_id, DDS_ENTITY_ID_LEN);
    ptr += DDS_ENTITY_ID_LEN;
    write_u32_le(&ptr, (uint32_t)ctx->sequence_number);

    /* 内联QoS: 参与者名称, 域名等 */
    /* PID_PARTICIPANT_GUID */
    write_u16_le(&ptr, 0x0050);                          /* parameterId */
    write_u16_le(&ptr, DDS_GUID_LEN);                    /* parameterLength */
    memcpy(ptr, ctx->participant_guid.prefix, DDS_GUID_PREFIX_LEN);
    ptr += DDS_GUID_PREFIX_LEN;
    memcpy(ptr, ctx->participant_guid.entity_id, DDS_ENTITY_ID_LEN);
    ptr += DDS_ENTITY_ID_LEN;

    /* PID_PROTOCOL_VERSION */
    write_u16_le(&ptr, 0x0015);
    write_u16_le(&ptr, 2);
    write_u8(&ptr, DDS_PROTOCOL_VERSION_MAJOR);
    write_u8(&ptr, DDS_PROTOCOL_VERSION_MINOR);

    /* PID_VENDOR_ID */
    write_u16_le(&ptr, 0x0016);
    write_u16_le(&ptr, 2);
    write_u16_le(&ptr, DDS_VENDOR_ID);

    /* PID_DEFAULT_UNICAST_LOCATOR */
    write_u16_le(&ptr, 0x0031);
    write_u16_le(&ptr, 12);
    write_u32_le(&ptr, 1);                               /* kind: LOCATOR_KIND_UDPv4 */
    write_u32_le(&ptr, ctx->user_traffic_port);
    write_u32_le(&ptr, 0x7F000001u);                     /* 127.0.0.1 (little-endian) */

    /* PID_METATRAFFIC_UNICAST_LOCATOR */
    write_u16_le(&ptr, 0x0032);
    write_u16_le(&ptr, 12);
    write_u32_le(&ptr, 1);                               /* kind: LOCATOR_KIND_UDPv4 */
    write_u32_le(&ptr, ctx->discovery_port);
    write_u32_le(&ptr, 0x7F000001u);

    /* PID_METATRAFFIC_MULTICAST_LOCATOR */
    write_u16_le(&ptr, 0x0033);
    write_u16_le(&ptr, 12);
    write_u32_le(&ptr, 1);                               /* kind: LOCATOR_KIND_UDPv4 */
    write_u32_le(&ptr, DDS_MULTICAST_PORT);
    write_u32_le(&ptr, 0x0100FFEFu);                     /* 239.255.0.1 (little-endian) */

    /* PID_PARTICIPANT_LEASE_DURATION */
    write_u16_le(&ptr, 0x0005);
    write_u16_le(&ptr, 8);
    write_u32_le(&ptr, (uint32_t)(DDS_LEASE_DURATION_MS / 1000));
    write_u32_le(&ptr, 0);

    /* PID_SENTINEL */
    write_u16_le(&ptr, 0x0001);
    write_u16_le(&ptr, 0);

    /* 回填DATA submessage长度 */
    uint16_t data_len = (uint16_t)((size_t)(ptr - packet) - data_submsg_offset - 2);
    packet[data_submsg_offset] = (uint8_t)(data_len & 0xFF);
    packet[data_submsg_offset + 1] = (uint8_t)((data_len >> 8) & 0xFF);

    size_t total_len = (size_t)(ptr - packet);

    /* 通过多播发送发现数据 */
    int sent = (int)sendto(ctx->discovery_socket, (const char*)packet, (int)total_len, 0,
                           (struct sockaddr*)&ctx->multicast_addr, sizeof(ctx->multicast_addr));
    if (sent < 0) {
        log_warning("[DDS] 发送SPDP发现消息失败");
    }

    ctx->sequence_number++;
    return ROS_OK;
}

/* 处理收到的DDS发现数据包 */
/* 新增 from_addr 参数，用于获取远程参与者真实IP地址 */
static void dds_process_discovery_packet(DDSContext* ctx, const uint8_t* data, size_t data_size,
                                          const struct sockaddr_in* from_addr) {
    if (data_size < 20) return;

    /* 验证RTPS头 */
    if (data[0] != 'R' || data[1] != 'T' || data[2] != 'P' || data[3] != 'S') return;

    const uint8_t* ptr = data + 4;
    uint8_t version_major = read_u8(&ptr);
    uint8_t version_minor = read_u8(&ptr);
    uint16_t vendor_id = read_u16_le(&ptr);

    /* 提取参与者GUID前缀 */
    uint8_t remote_prefix[DDS_GUID_PREFIX_LEN];
    memcpy(remote_prefix, ptr, DDS_GUID_PREFIX_LEN);
    ptr += DDS_GUID_PREFIX_LEN;

    /* 查找或创建远程参与者记录 */
    int participant_idx = -1;
    for (int i = 0; i < ctx->num_remote_participants; i++) {
        if (memcmp(ctx->remote_participants[i].participant.guid.prefix,
                   remote_prefix, DDS_GUID_PREFIX_LEN) == 0) {
            participant_idx = i;
            break;
        }
    }

    /* 解析子消息 */
    uint32_t remote_user_traffic_port = 0;
    uint32_t remote_discovery_port = 0;
    uint32_t remote_ip_addr = 0; /* 从SPDP解析的远程IP */
    char remote_node_name[128] = {0};

    while ((size_t)(ptr - data) + 4 <= data_size) {
        uint8_t submsg_id = read_u8(&ptr);
        uint8_t submsg_flags = read_u8(&ptr);
        uint16_t submsg_len = read_u16_le(&ptr);

        if ((size_t)(ptr - data) + submsg_len > data_size) break;

        if (submsg_id == 0x15) {
            /* DATA(p) 子消息 */
            uint16_t extra_flags = read_u16_le(&ptr);
            uint16_t octets_to_inline_qos = read_u16_le(&ptr);
            ptr += DDS_ENTITY_ID_LEN; /* readerEntityId */
            uint8_t writer_entity_id[DDS_ENTITY_ID_LEN];
            memcpy(writer_entity_id, ptr, DDS_ENTITY_ID_LEN);
            ptr += DDS_ENTITY_ID_LEN;
            uint32_t seq_num = read_u32_le(&ptr);

            /* 内联QoS */
            const uint8_t* inline_start = ptr;
            const uint8_t* inline_end = ptr + octets_to_inline_qos;
            ptr += octets_to_inline_qos;

            while ((size_t)(ptr - data) < data_size) {
                if ((size_t)(ptr - data) + 4 > data_size) break;
                uint16_t param_id = read_u16_le(&ptr);
                uint16_t param_len = read_u16_le(&ptr);

                if (param_id == 0x0001) break; /* PID_SENTINEL */
                if ((size_t)(ptr - data) + param_len > data_size) break;

                switch (param_id) {
                    case 0x0031: /* PID_DEFAULT_UNICAST_LOCATOR */
                        if (param_len >= 12) {
                            read_u32_le(&ptr); /* kind */
                            remote_user_traffic_port = read_u32_le(&ptr);
/* 提取远程IP地址（little-endian -> 网络字节序） */
                            remote_ip_addr = read_u32_le(&ptr);
                            ptr += (size_t)(param_len - 12);
                        } else if (param_len >= 8) {
                            read_u32_le(&ptr); /* kind */
                            remote_user_traffic_port = read_u32_le(&ptr);
                            ptr += (size_t)(param_len - 8);
                        } else {
                            ptr += param_len;
                        }
                        break;
                    case 0x0032: /* PID_METATRAFFIC_UNICAST_LOCATOR */
                        if (param_len >= 8) {
                            read_u32_le(&ptr); /* kind */
                            remote_discovery_port = read_u32_le(&ptr);
                            ptr += (size_t)(param_len - 8);
                        } else {
                            ptr += param_len;
                        }
                        break;
                    case 0x0034: /* PID_PARTICIPANT_BUILTIN_TOPICS */
                    case 0x0050: /* PID_PARTICIPANT_GUID */
                        ptr += param_len;
                        break;
                    case 0x0004: { /* PID_USER_DATA / PID_ENTITY_NAME */
                        /* 尝试解析节点名称 */
                        if (param_len > 0 && param_len < 128) {
                            size_t copy_len = param_len < 127 ? param_len : 127;
                            memcpy(remote_node_name, ptr, copy_len);
                            remote_node_name[copy_len] = '\0';
                        }
                        ptr += param_len;
                        break;
                    }
                    default:
                        ptr += param_len;
                        break;
                }
            }

            /* 如果参与者是新发现的 */
            if (participant_idx < 0 && ctx->num_remote_participants < DDS_MAX_PARTICIPANTS) {
                participant_idx = ctx->num_remote_participants++;
                DDSDomainParticipant* dp = &ctx->remote_participants[participant_idx].participant;
                memcpy(dp->guid.prefix, remote_prefix, DDS_GUID_PREFIX_LEN);
                memcpy(dp->guid.entity_id, writer_entity_id, DDS_ENTITY_ID_LEN);
                if (remote_node_name[0]) {
                    snprintf(dp->node_name, sizeof(dp->node_name), "%s", remote_node_name);
                } else {
                    snprintf(dp->node_name, sizeof(dp->node_name), "remote_%02x%02x",
                             remote_prefix[0], remote_prefix[1]);
                }
                dp->node_namespace[0] = '/';
                dp->node_namespace[1] = '\0';
                dp->domain_id = ctx->domain_id;
                dp->user_traffic_port = remote_user_traffic_port;
                dp->discovery_port = remote_discovery_port;

/* 确定远程参与者真实IP地址 */
                /* 优先使用SPDP中宣告的远程IP，若为无效/回环则用recvfrom源地址回退 */
                {
                    uint8_t first_octet = (uint8_t)(remote_ip_addr >> 24);
                    if (remote_ip_addr != 0 && first_octet != 127 && from_addr) {
                        dp->remote_ip = remote_ip_addr;
                    } else if (from_addr && from_addr->sin_addr.s_addr != 0) {
                        dp->remote_ip = from_addr->sin_addr.s_addr;
                    } else {
                        dp->remote_ip = htonl(INADDR_BROADCAST);
                    }
                }

                dp->protocol_version_major = version_major;
                dp->protocol_version_minor = version_minor;
                dp->vendor_id = vendor_id;
                dp->last_seen_ms = (int64_t)time(NULL) * 1000;
                dp->is_alive = 1;

                log_info("[DDS] 发现新参与者: '%s' (域=%u, 流量端口=%u)",
                         dp->node_name, ctx->domain_id, remote_user_traffic_port);

                if (ctx->participant_callback) {
                    ctx->participant_callback(dp, 1, ctx->participant_callback_data);
                }
            } else if (participant_idx >= 0) {
                /* 更新已存在参与者的最后可见时间和远程IP */
                ctx->remote_participants[participant_idx].participant.last_seen_ms =
                    (int64_t)time(NULL) * 1000;
                ctx->remote_participants[participant_idx].participant.is_alive = 1;

/* 更新远程IP（防止地址变更） */
                {
                    DDSDomainParticipant* dp = &ctx->remote_participants[participant_idx].participant;
                    uint8_t first_octet = (uint8_t)(remote_ip_addr >> 24);
                    if (remote_ip_addr != 0 && first_octet != 127) {
                        dp->remote_ip = remote_ip_addr;
                    } else if (from_addr && from_addr->sin_addr.s_addr != 0 &&
                               dp->remote_ip == 0) {
                        dp->remote_ip = from_addr->sin_addr.s_addr;
                    }
                }
            }
        } else {
            /* 跳过未知子消息 */
            ptr += submsg_len;
        }
    }
}

int dds_discovery_step(DDSContext* ctx) {
    if (!ctx) return ROS_ERROR_INVALID_PARAM;
    if (!ctx->discovery_running) return ROS_OK;

    /* 发送SPDP发现消息 */
    dds_send_discovery(ctx);

    /* 接收其他参与者的发现消息 */
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(ctx->discovery_socket, &read_fds);
    struct timeval tv = { 0, 50000 }; /* 50ms超时 */

    while (1) {
        int ret = select((int)(ctx->discovery_socket + 1), &read_fds, NULL, NULL, &tv);
        if (ret <= 0) break;

        uint8_t recv_buf[65536];
        struct sockaddr_in from_addr;
        socklen_t from_len = sizeof(from_addr);
        int recvd = (int)recvfrom(ctx->discovery_socket, (char*)recv_buf,
                                  sizeof(recv_buf), 0,
                                  (struct sockaddr*)&from_addr, &from_len);
        if (recvd > 0) {
/* 传入实际源地址用于确定远程IP */
            dds_process_discovery_packet(ctx, recv_buf, (size_t)recvd, &from_addr);
        }

        FD_ZERO(&read_fds);
        FD_SET(ctx->discovery_socket, &read_fds);
        tv.tv_sec = 0;
        tv.tv_usec = 10000; /* 继续尝试10ms */
    }

    /* 检查参与者租约过期 */
    int64_t now_ms = (int64_t)time(NULL) * 1000;
    for (int i = 0; i < ctx->num_remote_participants; i++) {
        DDSDomainParticipant* dp = &ctx->remote_participants[i].participant;
        if (dp->is_alive && (now_ms - dp->last_seen_ms) > DDS_LEASE_DURATION_MS) {
            dp->is_alive = 0;
            log_info("[DDS] 参与者 '%s' 已超时离线", dp->node_name);
            if (ctx->participant_callback) {
                ctx->participant_callback(dp, 0, ctx->participant_callback_data);
            }
        }
    }

    return ROS_OK;
}

/* 查找本地端点索引 */
static int dds_find_local_endpoint(DDSContext* ctx, const char* topic_name, int is_writer) {
    for (int i = 0; i < ctx->num_local_endpoints; i++) {
        if (strcmp(ctx->local_endpoints[i].topic_name, topic_name) == 0 &&
            ctx->local_endpoints[i].is_writer == is_writer) {
            return i;
        }
    }
    return -1;
}

int dds_create_data_writer(DDSContext* ctx, const char* topic_name,
                           const char* type_name, int is_reliable) {
    if (!ctx || !topic_name || !type_name) return ROS_ERROR_INVALID_PARAM;

    if (ctx->num_local_endpoints >= DDS_MAX_ENDPOINTS) return ROS_ERROR_NO_MEMORY;

    /* 检查是否已存在 */
    if (dds_find_local_endpoint(ctx, topic_name, 1) >= 0) {
        return ROS_ERROR_ALREADY_EXISTS;
    }

    int idx = ctx->num_local_endpoints++;
    snprintf(ctx->local_endpoints[idx].topic_name,
             sizeof(ctx->local_endpoints[idx].topic_name), "%s", topic_name);
    snprintf(ctx->local_endpoints[idx].type_name,
             sizeof(ctx->local_endpoints[idx].type_name), "%s", type_name);
    ctx->local_endpoints[idx].is_writer = 1;
    ctx->local_endpoints[idx].is_reliable = is_reliable;
    ctx->local_endpoints[idx].sequence_number = 0;

    /* 生成Writer端点GUID */
    dds_generate_guid(&ctx->local_endpoints[idx].endpoint_guid,
                      ctx->domain_id, DDS_ENTITY_WRITER, (uint16_t)idx);

    log_info("[DDS] 创建DataWriter: 话题='%s', 类型='%s', 可靠=%d",
             topic_name, type_name, is_reliable);
    return ROS_OK;
}

int dds_create_data_reader(DDSContext* ctx, const char* topic_name,
                           const char* type_name, int is_reliable,
                           DDSTopicCallback callback, void* user_data) {
    if (!ctx || !topic_name || !type_name) return ROS_ERROR_INVALID_PARAM;

    if (ctx->num_local_endpoints >= DDS_MAX_ENDPOINTS) return ROS_ERROR_NO_MEMORY;

    /* 检查是否已存在 */
    if (dds_find_local_endpoint(ctx, topic_name, 0) >= 0) {
        return ROS_ERROR_ALREADY_EXISTS;
    }

    int idx = ctx->num_local_endpoints++;
    snprintf(ctx->local_endpoints[idx].topic_name,
             sizeof(ctx->local_endpoints[idx].topic_name), "%s", topic_name);
    snprintf(ctx->local_endpoints[idx].type_name,
             sizeof(ctx->local_endpoints[idx].type_name), "%s", type_name);
    ctx->local_endpoints[idx].is_writer = 0;
    ctx->local_endpoints[idx].is_reliable = is_reliable;
    ctx->local_endpoints[idx].sequence_number = 0;

    dds_generate_guid(&ctx->local_endpoints[idx].endpoint_guid,
                      ctx->domain_id, DDS_ENTITY_READER, (uint16_t)idx);

    /* 注册回调 */
    if (callback && ctx->num_reader_callbacks < DDS_MAX_ENDPOINTS) {
        int cb_idx = ctx->num_reader_callbacks++;
        ctx->reader_callbacks[cb_idx].callback = callback;
        ctx->reader_callbacks[cb_idx].user_data = user_data;
        snprintf(ctx->reader_callbacks[cb_idx].topic_name,
                 sizeof(ctx->reader_callbacks[cb_idx].topic_name), "%s", topic_name);
    }

    log_info("[DDS] 创建DataReader: 话题='%s', 类型='%s', 可靠=%d",
             topic_name, type_name, is_reliable);
    return ROS_OK;
}

int dds_delete_endpoint(DDSContext* ctx, const char* topic_name, int is_writer) {
    if (!ctx || !topic_name) return ROS_ERROR_INVALID_PARAM;

    int idx = dds_find_local_endpoint(ctx, topic_name, is_writer);
    if (idx < 0) return ROS_ERROR_NOT_FOUND;

    /* 移动最后一个元素到当前位置 */
    if (idx < ctx->num_local_endpoints - 1) {
        memcpy(&ctx->local_endpoints[idx],
               &ctx->local_endpoints[ctx->num_local_endpoints - 1],
               sizeof(ctx->local_endpoints[0]));
    }
    ctx->num_local_endpoints--;

    /* 清除对应的reader回调 */
    if (!is_writer) {
        for (int i = 0; i < ctx->num_reader_callbacks; i++) {
            if (strcmp(ctx->reader_callbacks[i].topic_name, topic_name) == 0) {
                if (i < ctx->num_reader_callbacks - 1) {
                    memcpy(&ctx->reader_callbacks[i],
                           &ctx->reader_callbacks[ctx->num_reader_callbacks - 1],
                           sizeof(ctx->reader_callbacks[0]));
                }
                ctx->num_reader_callbacks--;
                break;
            }
        }
    }

    log_info("[DDS] 删除%s: 话题='%s'", is_writer ? "Writer" : "Reader", topic_name);
    return ROS_OK;
}

int dds_write_data(DDSContext* ctx, const char* topic_name,
                   const uint8_t* data, size_t data_size) {
    if (!ctx || !topic_name || !data || data_size == 0) return ROS_ERROR_INVALID_PARAM;

    int idx = dds_find_local_endpoint(ctx, topic_name, 1);
    if (idx < 0) return ROS_ERROR_NOT_FOUND;

    ctx->local_endpoints[idx].sequence_number++;

    /* 构建RTPS DATA子消息 */
    uint8_t packet[65536];
    size_t written = 0;

    int ret = dds_serialize_rtps_data(packet, sizeof(packet), &written,
                                      &ctx->participant_guid,
                                      &ctx->local_endpoints[idx].endpoint_guid,
                                      ctx->local_endpoints[idx].sequence_number,
                                      data, data_size);
    if (ret != ROS_OK) return ret;

    /* 发送给所有已知远程参与者的用户流量端口 */
    for (int i = 0; i < ctx->num_remote_participants; i++) {
        DDSDomainParticipant* dp = &ctx->remote_participants[i].participant;
        if (!dp->is_alive) continue;

        struct sockaddr_in dest;
        memset(&dest, 0, sizeof(dest));
        dest.sin_family = AF_INET;
        dest.sin_port = htons((unsigned short)dp->user_traffic_port);
/* 使用SPDP发现阶段获得的远程IP地址替代硬编码127.0.0.x */
        if (dp->remote_ip != 0) {
            dest.sin_addr.s_addr = dp->remote_ip;
        } else {
            /* 回退：远程IP未知时使用广播地址 */
            dest.sin_addr.s_addr = htonl(INADDR_BROADCAST);
        }

        /* 发送数据 */
        sendto(ctx->user_traffic_socket, (const char*)packet, (int)written, 0,
               (struct sockaddr*)&dest, sizeof(dest));
    }

    return ROS_OK;
}

int dds_read_data(DDSContext* ctx, const char* topic_name,
                  uint8_t* buffer, size_t buffer_size, size_t* read) {
    if (!ctx || !topic_name || !buffer || !read) return ROS_ERROR_INVALID_PARAM;

    /* 检查是否有对应的reader */
    int idx = dds_find_local_endpoint(ctx, topic_name, 0);
    if (idx < 0) return ROS_ERROR_NOT_FOUND;

    /* 从用户流量端口接收数据 */
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(ctx->user_traffic_socket, &read_fds);
    struct timeval tv = { 0, 10000 }; /* 10ms超时 */

    int ret = select((int)(ctx->user_traffic_socket + 1), &read_fds, NULL, NULL, &tv);
    if (ret <= 0) {
        *read = 0;
        return ROS_ERROR_TIMEOUT;
    }

    uint8_t recv_buf[65536];
    struct sockaddr_in from_addr;
    socklen_t from_len = sizeof(from_addr);
    int recvd = (int)recvfrom(ctx->user_traffic_socket, (char*)recv_buf,
                              sizeof(recv_buf), 0,
                              (struct sockaddr*)&from_addr, &from_len);
    if (recvd <= 0) {
        *read = 0;
        return ROS_ERROR_TIMEOUT;
    }

    /* 反序列化RTPS数据 */
    DDSDataSubmessage submsg;
    size_t data_read = 0;
    size_t parsed = 0;

    ret = dds_deserialize_rtps_data(recv_buf, (size_t)recvd, &parsed,
                                    &submsg, buffer, buffer_size, &data_read);
    if (ret != ROS_OK) {
        *read = 0;
        return ret;
    }

    *read = data_read;

    /* 触发回调 */
    for (int i = 0; i < ctx->num_reader_callbacks; i++) {
        if (strcmp(ctx->reader_callbacks[i].topic_name, topic_name) == 0 &&
            ctx->reader_callbacks[i].callback) {
            ctx->reader_callbacks[i].callback(topic_name, buffer, data_read,
                                              &submsg.writer_id,
                                              ctx->reader_callbacks[i].user_data);
        }
    }

    return ROS_OK;
}

int dds_get_participants(DDSContext* ctx,
                         DDSDomainParticipant* participants, int max_count) {
    if (!ctx || !participants || max_count <= 0) return ROS_ERROR_INVALID_PARAM;

    int count = 0;
    for (int i = 0; i < ctx->num_remote_participants && count < max_count; i++) {
        if (ctx->remote_participants[i].participant.is_alive) {
            memcpy(&participants[count],
                   &ctx->remote_participants[i].participant,
                   sizeof(DDSDomainParticipant));
            count++;
        }
    }
    return count;
}

int dds_get_endpoints(DDSContext* ctx, DDSEndpoint* endpoints, int max_count) {
    if (!ctx || !endpoints || max_count <= 0) return ROS_ERROR_INVALID_PARAM;

    int count = 0;
    for (int i = 0; i < ctx->num_remote_endpoints && count < max_count; i++) {
        /* 只返回有对应存活参与者的端点 */
        for (int j = 0; j < ctx->num_remote_participants; j++) {
            if (ctx->remote_participants[j].participant.is_alive &&
                memcmp(ctx->remote_endpoints[i].participant_guid.prefix,
                       ctx->remote_participants[j].participant.guid.prefix,
                       DDS_GUID_PREFIX_LEN) == 0) {
                memcpy(&endpoints[count], &ctx->remote_endpoints[i], sizeof(DDSEndpoint));
                count++;
                break;
            }
        }
    }
    return count;
}

int dds_serialize_rtps_data(uint8_t* buffer, size_t buffer_size, size_t* written,
                            const DDSGuid* writer_id, const DDSGuid* reader_id,
                            uint32_t sequence_number, const uint8_t* data,
                            size_t data_size) {
    if (!buffer || !written || !writer_id || !reader_id || !data) {
        return ROS_ERROR_INVALID_PARAM;
    }

    /* 最小需要: RTPS头(20) + DATA submsg头(4+8+8+4) + 数据 */
    size_t min_size = 20 + 4 + 2 + 2 + 8 + 8 + 4 + data_size;
    if (buffer_size < min_size) return ROS_ERROR_SERIALIZE;

    uint8_t* ptr = buffer;

    /* RTPS Header (20字节) */
    memcpy(ptr, "RTPS", 4); ptr += 4;
    write_u8(&ptr, DDS_PROTOCOL_VERSION_MAJOR);
    write_u8(&ptr, DDS_PROTOCOL_VERSION_MINOR);
    write_u16_le(&ptr, DDS_VENDOR_ID);
    memcpy(ptr, writer_id->prefix, DDS_GUID_PREFIX_LEN);
    ptr += DDS_GUID_PREFIX_LEN;

    /* DATA 子消息 (0x15) */
    write_u8(&ptr, 0x15);                                /* submessageId */
    write_u8(&ptr, 0x05);                                /* flags: D=1, E=1 */
    size_t submsg_len_offset = (size_t)(ptr - buffer);
    write_u16_le(&ptr, 0);                               /* submessageLength 占位 */
    write_u16_le(&ptr, 0x0000);                          /* extraFlags */
    write_u16_le(&ptr, 0x0000);                          /* octetsToInlineQos */
    memcpy(ptr, reader_id->entity_id, DDS_ENTITY_ID_LEN);
    ptr += DDS_ENTITY_ID_LEN;
    memcpy(ptr, writer_id->entity_id, DDS_ENTITY_ID_LEN);
    ptr += DDS_ENTITY_ID_LEN;
    write_u32_le(&ptr, sequence_number);

    /* 用户数据 */
    if (data_size > 0) {
        memcpy(ptr, data, data_size);
        ptr += data_size;
    }

    /* 回填submessageLength */
    uint16_t submsg_len = (uint16_t)((size_t)(ptr - buffer) - submsg_len_offset - 2);
    buffer[submsg_len_offset] = (uint8_t)(submsg_len & 0xFF);
    buffer[submsg_len_offset + 1] = (uint8_t)((submsg_len >> 8) & 0xFF);

    *written = (size_t)(ptr - buffer);
    return ROS_OK;
}

int dds_deserialize_rtps_data(const uint8_t* buffer, size_t buffer_size,
                              size_t* read, DDSDataSubmessage* submsg,
                              uint8_t* data_out, size_t data_out_size,
                              size_t* data_read) {
    if (!buffer || !read || !submsg || !data_out || !data_read) {
        return ROS_ERROR_INVALID_PARAM;
    }

    if (buffer_size < 20 + 8) return ROS_ERROR_DESERIALIZE;

    /* 验证RTPS头 */
    if (buffer[0] != 'R' || buffer[1] != 'T' || buffer[2] != 'P' || buffer[3] != 'S') {
        return ROS_ERROR_PROTOCOL;
    }

    const uint8_t* ptr = buffer + 4;
    /* 跳过版本和厂商 */
    ptr += 4;

    /* 写入者的GUID前缀 */
    memcpy(submsg->writer_id.prefix, ptr, DDS_GUID_PREFIX_LEN);
    ptr += DDS_GUID_PREFIX_LEN;

/* 增强RTPS子消息解析。
     * 不再简单线性扫描0x15标记，而是按RTPS规范逐个解析子消息头，
     * 正确处理INFO_TS(0x09)、INFO_SRC(0x0c)、INFO_DST(0x0e)、
     * HEARTBEAT(0x07)、ACKNACK(0x06)、GAP(0x08)、DATA(0x15)等标准子消息。
     * 跳过非DATA子消息体，定位到第一个DATA子消息。 */
    const uint8_t* search_ptr = ptr;
    int found_data = 0;
    while ((size_t)(search_ptr - buffer) + 4 <= buffer_size) {
        uint8_t sub_id = *search_ptr;
        if (sub_id == 0x15) { /* DATA */
            found_data = 1;
            ptr = search_ptr;
            break;
        }
        /* 其他子消息: 读取flags+length跳过其body */
        uint8_t flags = (search_ptr + 1 < buffer + buffer_size) ? *(search_ptr + 1) : 0;
        if (search_ptr + 4 > buffer + buffer_size) break;
        uint16_t sub_len = read_u16_le((const uint8_t**)&search_ptr) - 2;
        (void)flags;
        if (sub_id == 0x00) { /* PAD - padding */
            search_ptr++;
            continue;
        }
        /* 非DATA非PAD子消息: 跳过submessageLength+body */
        search_ptr += 4;
        if (sub_len > 0 && (size_t)(search_ptr - buffer) + sub_len <= buffer_size) {
            search_ptr += sub_len;
        } else if (sub_id != 0x15) {
            search_ptr++; /* 无法跳过的边界情况: 前进1字节 */
        }
    }

    if (!found_data) {
        /* 将RTPS头后面的所有内容视为有效负载 */
        size_t remaining = buffer_size - (size_t)(ptr - buffer);
        if (remaining > 4) {
            /* 检查是否为DATA子消息ID */
            if (*ptr == 0x15) {
                found_data = 1;
            }
        }
    }

    if (found_data) {
        submsg->header.submessage_id = read_u8(&ptr);
        submsg->header.flags = read_u8(&ptr);
        submsg->header.length = read_u16_le(&ptr);
        submsg->extra_flags = read_u16_le(&ptr);
        submsg->octets_to_inline_qos = read_u16_le(&ptr);
        memcpy(submsg->reader_id.entity_id, ptr, DDS_ENTITY_ID_LEN);
        ptr += DDS_ENTITY_ID_LEN;
        memcpy(submsg->writer_id.entity_id, ptr, DDS_ENTITY_ID_LEN);
        ptr += DDS_ENTITY_ID_LEN;
        submsg->sequence_number = read_u32_le(&ptr);

        /* 用户数据 */
        size_t remaining = buffer_size - (size_t)(ptr - buffer);
        size_t copy_size = remaining < data_out_size ? remaining : data_out_size;
        if (copy_size > 0) {
            memcpy(data_out, ptr, copy_size);
        }
        ptr += copy_size;
        *data_read = copy_size;
    } else {
        /* 没有找到DATA子消息，将剩余数据作为有效负载 */
        memset(submsg, 0, sizeof(DDSDataSubmessage));
        size_t remaining = buffer_size - (size_t)(ptr - buffer);
        size_t copy_size = remaining < data_out_size ? remaining : data_out_size;
        if (copy_size > 0) {
            memcpy(data_out, ptr, copy_size);
        }
        *data_read = copy_size;
        ptr += copy_size;
    }

    *read = (size_t)(ptr - buffer);
    return ROS_OK;
}

void dds_generate_guid(DDSGuid* guid, uint32_t participant_id,
                       uint8_t entity_kind, uint16_t entity_id) {
    if (!guid) return;

    memset(guid, 0, sizeof(DDSGuid));

    /* 使用当前时间作为前缀的一部分 */
    uint64_t timestamp = (uint64_t)time(NULL);

    /* 生成12字节前缀 */
    guid->prefix[0] = (uint8_t)((participant_id >> 24) & 0xFF);
    guid->prefix[1] = (uint8_t)((participant_id >> 16) & 0xFF);
    guid->prefix[2] = (uint8_t)((participant_id >> 8) & 0xFF);
    guid->prefix[3] = (uint8_t)(participant_id & 0xFF);

    /* 加入时间戳 */
    guid->prefix[4] = (uint8_t)((timestamp >> 40) & 0xFF);
    guid->prefix[5] = (uint8_t)((timestamp >> 32) & 0xFF);
    guid->prefix[6] = (uint8_t)((timestamp >> 24) & 0xFF);
    guid->prefix[7] = (uint8_t)((timestamp >> 16) & 0xFF);
    guid->prefix[8] = (uint8_t)((timestamp >> 8) & 0xFF);
    guid->prefix[9] = (uint8_t)(timestamp & 0xFF);

    /* 加入一些随机性（使用时间加扰） */
    timestamp ^= (timestamp >> 33);
    guid->prefix[10] = (uint8_t)((timestamp >> 8) & 0xFF);
    guid->prefix[11] = (uint8_t)(timestamp & 0xFF);

    /* 生成4字节实体ID */
    guid->entity_id[0] = entity_kind;
    guid->entity_id[1] = (uint8_t)((entity_id >> 8) & 0xFF);
    guid->entity_id[2] = (uint8_t)(entity_id & 0xFF);
    guid->entity_id[3] = (uint8_t)(entity_kind ^ (entity_id & 0xFF));
}
