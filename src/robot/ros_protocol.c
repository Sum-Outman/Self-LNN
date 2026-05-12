#include "selflnn/robot/ros_protocol.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/core/common.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#if defined(SELFLNN_PLATFORM_WINDOWS) && SELFLNN_PLATFORM_WINDOWS
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    typedef SOCKET socket_t;
    #define INVALID_SOCKET_VAL INVALID_SOCKET
    #define SOCKET_ERROR_VAL SOCKET_ERROR
    #define CLOSE_SOCKET(s) closesocket(s)
    #define GET_SOCKET_ERR() WSAGetLastError()
#else
    #include <sys/socket.h>
    #include <sys/types.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <unistd.h>
    #include <fcntl.h>
    typedef int socket_t;
    #define INVALID_SOCKET_VAL (-1)
    #define SOCKET_ERROR_VAL (-1)
    #define CLOSE_SOCKET(s) close(s)
    #define GET_SOCKET_ERR() (errno)
#endif

static int write_uint32(uint8_t* buf, size_t size, size_t* pos, uint32_t val) {
    if (*pos + 4 > size) return ROS_ERROR_SERIALIZE;
    buf[(*pos)++] = (uint8_t)(val >> 0);
    buf[(*pos)++] = (uint8_t)(val >> 8);
    buf[(*pos)++] = (uint8_t)(val >> 16);
    buf[(*pos)++] = (uint8_t)(val >> 24);
    return ROS_OK;
}

static int read_uint32(const uint8_t* buf, size_t size, size_t* pos, uint32_t* val) {
    if (*pos + 4 > size) return ROS_ERROR_DESERIALIZE;
    *val = ((uint32_t)buf[*pos]) |
           ((uint32_t)buf[*pos + 1] << 8) |
           ((uint32_t)buf[*pos + 2] << 16) |
           ((uint32_t)buf[*pos + 3] << 24);
    *pos += 4;
    return ROS_OK;
}

static int write_uint8(uint8_t* buf, size_t size, size_t* pos, uint8_t val) {
    if (*pos + 1 > size) return ROS_ERROR_SERIALIZE;
    buf[(*pos)++] = val;
    return ROS_OK;
}

static int read_uint8(const uint8_t* buf, size_t size, size_t* pos, uint8_t* val) {
    if (*pos + 1 > size) return ROS_ERROR_DESERIALIZE;
    *val = buf[*pos];
    (*pos)++;
    return ROS_OK;
}

static int write_double(uint8_t* buf, size_t size, size_t* pos, double val) {
    uint64_t bits;
    memcpy(&bits, &val, sizeof(bits));
    if (*pos + 8 > size) return ROS_ERROR_SERIALIZE;
    for (int i = 0; i < 8; i++) {
        buf[(*pos)++] = (uint8_t)(bits >> (i * 8));
    }
    return ROS_OK;
}

static int read_double(const uint8_t* buf, size_t size, size_t* pos, double* val) {
    uint64_t bits = 0;
    if (*pos + 8 > size) return ROS_ERROR_DESERIALIZE;
    for (int i = 0; i < 8; i++) {
        bits |= ((uint64_t)buf[*pos + i]) << (i * 8);
    }
    memcpy(val, &bits, sizeof(*val));
    *pos += 8;
    return ROS_OK;
}

static int write_float(uint8_t* buf, size_t size, size_t* pos, float val) {
    uint32_t bits;
    memcpy(&bits, &val, sizeof(bits));
    if (*pos + 4 > size) return ROS_ERROR_SERIALIZE;
    buf[(*pos)++] = (uint8_t)(bits >> 0);
    buf[(*pos)++] = (uint8_t)(bits >> 8);
    buf[(*pos)++] = (uint8_t)(bits >> 16);
    buf[(*pos)++] = (uint8_t)(bits >> 24);
    return ROS_OK;
}

static int read_float(const uint8_t* buf, size_t size, size_t* pos, float* val) {
    uint32_t bits;
    if (*pos + 4 > size) return ROS_ERROR_DESERIALIZE;
    bits = ((uint32_t)buf[*pos]) |
           ((uint32_t)buf[*pos + 1] << 8) |
           ((uint32_t)buf[*pos + 2] << 16) |
           ((uint32_t)buf[*pos + 3] << 24);
    memcpy(val, &bits, sizeof(*val));
    *pos += 4;
    return ROS_OK;
}

static int write_int32(uint8_t* buf, size_t size, size_t* pos, int32_t val) {
    uint32_t uval = (uint32_t)val;
    return write_uint32(buf, size, pos, uval);
}

static int read_int32(const uint8_t* buf, size_t size, size_t* pos, int32_t* val) {
    uint32_t uval;
    int ret = read_uint32(buf, size, pos, &uval);
    if (ret == ROS_OK) *val = (int32_t)uval;
    return ret;
}

static int write_string(uint8_t* buf, size_t size, size_t* pos, const char* str) {
    uint32_t len = (str != NULL) ? (uint32_t)strlen(str) : 0;
    int ret = write_uint32(buf, size, pos, len);
    if (ret != ROS_OK) return ret;
    if (len > 0) {
        if (*pos + len > size) return ROS_ERROR_SERIALIZE;
        memcpy(buf + *pos, str, len);
        *pos += len;
    }
    return ROS_OK;
}

static int read_string(const uint8_t* buf, size_t size, size_t* pos, char* out, size_t out_size) {
    uint32_t len;
    int ret = read_uint32(buf, size, pos, &len);
    if (ret != ROS_OK) return ret;
    if (len >= out_size) return ROS_ERROR_DESERIALIZE;
    if (*pos + len > size) return ROS_ERROR_DESERIALIZE;
    if (len > 0) {
        memcpy(out, buf + *pos, len);
        *pos += len;
    }
    out[len] = '\0';
    return ROS_OK;
}

static int read_string_alloc(const uint8_t* buf, size_t size, size_t* pos, char** out) {
    uint32_t len;
    int ret = read_uint32(buf, size, pos, &len);
    if (ret != ROS_OK) return ret;
    if (*pos + len > size) return ROS_ERROR_DESERIALIZE;
    if (len == 0) {
        *out = NULL;
        return ROS_OK;
    }
    *out = (char*)safe_malloc(len + 1);
    if (!*out) return ROS_ERROR_NO_MEMORY;
    memcpy(*out, buf + *pos, len);
    (*out)[len] = '\0';
    *pos += len;
    return ROS_OK;
}

int ros_serialize_header(const RosHeader* header, uint8_t* buffer, size_t buffer_size, size_t* written) {
    size_t pos = 0;
    int ret;
    ret = write_uint32(buffer, buffer_size, &pos, header->seq);
    if (ret != ROS_OK) return ret;
    ret = write_uint32(buffer, buffer_size, &pos, (uint32_t)header->timestamp_sec);
    if (ret != ROS_OK) return ret;
    ret = write_uint32(buffer, buffer_size, &pos, (uint32_t)header->timestamp_nsec);
    if (ret != ROS_OK) return ret;
    ret = write_string(buffer, buffer_size, &pos, header->frame_id);
    if (ret != ROS_OK) return ret;
    *written = pos;
    return ROS_OK;
}

int ros_deserialize_header(const uint8_t* buffer, size_t buffer_size, size_t* read, RosHeader* header) {
    size_t pos = 0;
    int ret;
    uint32_t u32;
    ret = read_uint32(buffer, buffer_size, &pos, &u32);
    if (ret != ROS_OK) return ret;
    header->seq = u32;
    ret = read_uint32(buffer, buffer_size, &pos, &u32);
    if (ret != ROS_OK) return ret;
    header->timestamp_sec = (double)u32;
    ret = read_uint32(buffer, buffer_size, &pos, &u32);
    if (ret != ROS_OK) return ret;
    header->timestamp_nsec = (double)u32;
    ret = read_string(buffer, buffer_size, &pos, header->frame_id, sizeof(header->frame_id));
    if (ret != ROS_OK) return ret;
    *read = pos;
    return ROS_OK;
}

int ros_serialize_pose(const RosPose* pose, uint8_t* buffer, size_t buffer_size, size_t* written) {
    size_t pos = 0;
    int ret;
    ret = write_double(buffer, buffer_size, &pos, pose->position.x);
    if (ret != ROS_OK) return ret;
    ret = write_double(buffer, buffer_size, &pos, pose->position.y);
    if (ret != ROS_OK) return ret;
    ret = write_double(buffer, buffer_size, &pos, pose->position.z);
    if (ret != ROS_OK) return ret;
    ret = write_double(buffer, buffer_size, &pos, pose->orientation.x);
    if (ret != ROS_OK) return ret;
    ret = write_double(buffer, buffer_size, &pos, pose->orientation.y);
    if (ret != ROS_OK) return ret;
    ret = write_double(buffer, buffer_size, &pos, pose->orientation.z);
    if (ret != ROS_OK) return ret;
    ret = write_double(buffer, buffer_size, &pos, pose->orientation.w);
    if (ret != ROS_OK) return ret;
    *written = pos;
    return ROS_OK;
}

int ros_deserialize_pose(const uint8_t* buffer, size_t buffer_size, size_t* read, RosPose* pose) {
    size_t pos = 0;
    int ret;
    ret = read_double(buffer, buffer_size, &pos, &pose->position.x);
    if (ret != ROS_OK) return ret;
    ret = read_double(buffer, buffer_size, &pos, &pose->position.y);
    if (ret != ROS_OK) return ret;
    ret = read_double(buffer, buffer_size, &pos, &pose->position.z);
    if (ret != ROS_OK) return ret;
    ret = read_double(buffer, buffer_size, &pos, &pose->orientation.x);
    if (ret != ROS_OK) return ret;
    ret = read_double(buffer, buffer_size, &pos, &pose->orientation.y);
    if (ret != ROS_OK) return ret;
    ret = read_double(buffer, buffer_size, &pos, &pose->orientation.z);
    if (ret != ROS_OK) return ret;
    ret = read_double(buffer, buffer_size, &pos, &pose->orientation.w);
    if (ret != ROS_OK) return ret;
    *read = pos;
    return ROS_OK;
}

int ros_serialize_twist(const RosTwist* twist, uint8_t* buffer, size_t buffer_size, size_t* written) {
    size_t pos = 0;
    int ret;
    ret = write_double(buffer, buffer_size, &pos, twist->linear.x);
    if (ret != ROS_OK) return ret;
    ret = write_double(buffer, buffer_size, &pos, twist->linear.y);
    if (ret != ROS_OK) return ret;
    ret = write_double(buffer, buffer_size, &pos, twist->linear.z);
    if (ret != ROS_OK) return ret;
    ret = write_double(buffer, buffer_size, &pos, twist->angular.x);
    if (ret != ROS_OK) return ret;
    ret = write_double(buffer, buffer_size, &pos, twist->angular.y);
    if (ret != ROS_OK) return ret;
    ret = write_double(buffer, buffer_size, &pos, twist->angular.z);
    if (ret != ROS_OK) return ret;
    *written = pos;
    return ROS_OK;
}

int ros_deserialize_twist(const uint8_t* buffer, size_t buffer_size, size_t* read, RosTwist* twist) {
    size_t pos = 0;
    int ret;
    ret = read_double(buffer, buffer_size, &pos, &twist->linear.x);
    if (ret != ROS_OK) return ret;
    ret = read_double(buffer, buffer_size, &pos, &twist->linear.y);
    if (ret != ROS_OK) return ret;
    ret = read_double(buffer, buffer_size, &pos, &twist->linear.z);
    if (ret != ROS_OK) return ret;
    ret = read_double(buffer, buffer_size, &pos, &twist->angular.x);
    if (ret != ROS_OK) return ret;
    ret = read_double(buffer, buffer_size, &pos, &twist->angular.y);
    if (ret != ROS_OK) return ret;
    ret = read_double(buffer, buffer_size, &pos, &twist->angular.z);
    if (ret != ROS_OK) return ret;
    *read = pos;
    return ROS_OK;
}

int ros_serialize_odometry(const RosOdometry* odom, uint8_t* buffer, size_t buffer_size, size_t* written) {
    size_t pos = 0;
    int ret;
    ret = ros_serialize_pose(&odom->pose, buffer, buffer_size, &pos);
    if (ret != ROS_OK) return ret;
    ret = ros_serialize_twist(&odom->twist, buffer, buffer_size, &pos);
    if (ret != ROS_OK) return ret;
    *written = pos;
    return ROS_OK;
}

int ros_deserialize_odometry(const uint8_t* buffer, size_t buffer_size, size_t* read, RosOdometry* odom) {
    size_t pos = 0;
    int ret;
    ret = ros_deserialize_pose(buffer, buffer_size, &pos, &odom->pose);
    if (ret != ROS_OK) return ret;
    ret = ros_deserialize_twist(buffer, buffer_size, &pos, &odom->twist);
    if (ret != ROS_OK) return ret;
    *read = pos;
    return ROS_OK;
}

int ros_serialize_laserscan(const RosLaserScan* scan, uint8_t* buffer, size_t buffer_size, size_t* written) {
    size_t pos = 0;
    int ret;
    ret = ros_serialize_header(&scan->header, buffer, buffer_size, &pos);
    if (ret != ROS_OK) return ret;
    ret = write_float(buffer, buffer_size, &pos, scan->angle_min);
    if (ret != ROS_OK) return ret;
    ret = write_float(buffer, buffer_size, &pos, scan->angle_max);
    if (ret != ROS_OK) return ret;
    ret = write_float(buffer, buffer_size, &pos, scan->angle_increment);
    if (ret != ROS_OK) return ret;
    ret = write_float(buffer, buffer_size, &pos, scan->time_increment);
    if (ret != ROS_OK) return ret;
    ret = write_float(buffer, buffer_size, &pos, scan->scan_time);
    if (ret != ROS_OK) return ret;
    ret = write_float(buffer, buffer_size, &pos, scan->range_min);
    if (ret != ROS_OK) return ret;
    ret = write_float(buffer, buffer_size, &pos, scan->range_max);
    if (ret != ROS_OK) return ret;
    ret = write_int32(buffer, buffer_size, &pos, scan->ranges_size);
    if (ret != ROS_OK) return ret;
    for (int i = 0; i < scan->ranges_size && i < 1024; i++) {
        ret = write_float(buffer, buffer_size, &pos, scan->ranges[i]);
        if (ret != ROS_OK) return ret;
    }
    ret = write_int32(buffer, buffer_size, &pos, scan->intensities_size);
    if (ret != ROS_OK) return ret;
    for (int i = 0; i < scan->intensities_size && i < 1024; i++) {
        ret = write_float(buffer, buffer_size, &pos, scan->intensities[i]);
        if (ret != ROS_OK) return ret;
    }
    *written = pos;
    return ROS_OK;
}

int ros_deserialize_laserscan(const uint8_t* buffer, size_t buffer_size, size_t* read, RosLaserScan* scan) {
    size_t pos = 0;
    int ret;
    ret = ros_deserialize_header(buffer, buffer_size, &pos, &scan->header);
    if (ret != ROS_OK) return ret;
    ret = read_float(buffer, buffer_size, &pos, &scan->angle_min);
    if (ret != ROS_OK) return ret;
    ret = read_float(buffer, buffer_size, &pos, &scan->angle_max);
    if (ret != ROS_OK) return ret;
    ret = read_float(buffer, buffer_size, &pos, &scan->angle_increment);
    if (ret != ROS_OK) return ret;
    ret = read_float(buffer, buffer_size, &pos, &scan->time_increment);
    if (ret != ROS_OK) return ret;
    ret = read_float(buffer, buffer_size, &pos, &scan->scan_time);
    if (ret != ROS_OK) return ret;
    ret = read_float(buffer, buffer_size, &pos, &scan->range_min);
    if (ret != ROS_OK) return ret;
    ret = read_float(buffer, buffer_size, &pos, &scan->range_max);
    if (ret != ROS_OK) return ret;
    int32_t n;
    ret = read_int32(buffer, buffer_size, &pos, &n);
    if (ret != ROS_OK) return ret;
    scan->ranges_size = (n > 1024) ? 1024 : n;
    for (int i = 0; i < scan->ranges_size; i++) {
        ret = read_float(buffer, buffer_size, &pos, &scan->ranges[i]);
        if (ret != ROS_OK) return ret;
    }
    ret = read_int32(buffer, buffer_size, &pos, &n);
    if (ret != ROS_OK) return ret;
    scan->intensities_size = (n > 1024) ? 1024 : n;
    for (int i = 0; i < scan->intensities_size; i++) {
        ret = read_float(buffer, buffer_size, &pos, &scan->intensities[i]);
        if (ret != ROS_OK) return ret;
    }
    *read = pos;
    return ROS_OK;
}

int ros_serialize_joy(const RosJoy* joy, uint8_t* buffer, size_t buffer_size, size_t* written) {
    size_t pos = 0;
    int ret;
    ret = ros_serialize_header(&joy->header, buffer, buffer_size, &pos);
    if (ret != ROS_OK) return ret;
    for (int i = 0; i < 8; i++) {
        ret = write_double(buffer, buffer_size, &pos, joy->axes[i]);
        if (ret != ROS_OK) return ret;
    }
    for (int i = 0; i < 16; i++) {
        ret = write_int32(buffer, buffer_size, &pos, joy->buttons[i]);
        if (ret != ROS_OK) return ret;
    }
    *written = pos;
    return ROS_OK;
}

int ros_deserialize_joy(const uint8_t* buffer, size_t buffer_size, size_t* read, RosJoy* joy) {
    size_t pos = 0;
    int ret;
    ret = ros_deserialize_header(buffer, buffer_size, &pos, &joy->header);
    if (ret != ROS_OK) return ret;
    for (int i = 0; i < 8; i++) {
        ret = read_double(buffer, buffer_size, &pos, &joy->axes[i]);
        if (ret != ROS_OK) return ret;
    }
    for (int i = 0; i < 16; i++) {
        ret = read_int32(buffer, buffer_size, &pos, &joy->buttons[i]);
        if (ret != ROS_OK) return ret;
    }
    *read = pos;
    return ROS_OK;
}

int ros_serialize_imu(const RosImu* imu, uint8_t* buffer, size_t buffer_size, size_t* written) {
    size_t pos = 0;
    int ret;
    ret = ros_serialize_header(&imu->header, buffer, buffer_size, &pos);
    if (ret != ROS_OK) return ret;
    ret = write_double(buffer, buffer_size, &pos, imu->orientation.x);
    if (ret != ROS_OK) return ret;
    ret = write_double(buffer, buffer_size, &pos, imu->orientation.y);
    if (ret != ROS_OK) return ret;
    ret = write_double(buffer, buffer_size, &pos, imu->orientation.z);
    if (ret != ROS_OK) return ret;
    ret = write_double(buffer, buffer_size, &pos, imu->orientation.w);
    if (ret != ROS_OK) return ret;
    for (int i = 0; i < 9; i++) {
        ret = write_double(buffer, buffer_size, &pos, imu->orientation_covariance[i]);
        if (ret != ROS_OK) return ret;
    }
    ret = write_double(buffer, buffer_size, &pos, imu->angular_velocity.x);
    if (ret != ROS_OK) return ret;
    ret = write_double(buffer, buffer_size, &pos, imu->angular_velocity.y);
    if (ret != ROS_OK) return ret;
    ret = write_double(buffer, buffer_size, &pos, imu->angular_velocity.z);
    if (ret != ROS_OK) return ret;
    for (int i = 0; i < 9; i++) {
        ret = write_double(buffer, buffer_size, &pos, imu->angular_velocity_covariance[i]);
        if (ret != ROS_OK) return ret;
    }
    ret = write_double(buffer, buffer_size, &pos, imu->linear_acceleration.x);
    if (ret != ROS_OK) return ret;
    ret = write_double(buffer, buffer_size, &pos, imu->linear_acceleration.y);
    if (ret != ROS_OK) return ret;
    ret = write_double(buffer, buffer_size, &pos, imu->linear_acceleration.z);
    if (ret != ROS_OK) return ret;
    for (int i = 0; i < 9; i++) {
        ret = write_double(buffer, buffer_size, &pos, imu->linear_acceleration_covariance[i]);
        if (ret != ROS_OK) return ret;
    }
    *written = pos;
    return ROS_OK;
}

int ros_deserialize_imu(const uint8_t* buffer, size_t buffer_size, size_t* read, RosImu* imu) {
    size_t pos = 0;
    int ret;
    ret = ros_deserialize_header(buffer, buffer_size, &pos, &imu->header);
    if (ret != ROS_OK) return ret;
    ret = read_double(buffer, buffer_size, &pos, &imu->orientation.x);
    if (ret != ROS_OK) return ret;
    ret = read_double(buffer, buffer_size, &pos, &imu->orientation.y);
    if (ret != ROS_OK) return ret;
    ret = read_double(buffer, buffer_size, &pos, &imu->orientation.z);
    if (ret != ROS_OK) return ret;
    ret = read_double(buffer, buffer_size, &pos, &imu->orientation.w);
    if (ret != ROS_OK) return ret;
    for (int i = 0; i < 9; i++) {
        ret = read_double(buffer, buffer_size, &pos, &imu->orientation_covariance[i]);
        if (ret != ROS_OK) return ret;
    }
    ret = read_double(buffer, buffer_size, &pos, &imu->angular_velocity.x);
    if (ret != ROS_OK) return ret;
    ret = read_double(buffer, buffer_size, &pos, &imu->angular_velocity.y);
    if (ret != ROS_OK) return ret;
    ret = read_double(buffer, buffer_size, &pos, &imu->angular_velocity.z);
    if (ret != ROS_OK) return ret;
    for (int i = 0; i < 9; i++) {
        ret = read_double(buffer, buffer_size, &pos, &imu->angular_velocity_covariance[i]);
        if (ret != ROS_OK) return ret;
    }
    ret = read_double(buffer, buffer_size, &pos, &imu->linear_acceleration.x);
    if (ret != ROS_OK) return ret;
    ret = read_double(buffer, buffer_size, &pos, &imu->linear_acceleration.y);
    if (ret != ROS_OK) return ret;
    ret = read_double(buffer, buffer_size, &pos, &imu->linear_acceleration.z);
    if (ret != ROS_OK) return ret;
    for (int i = 0; i < 9; i++) {
        ret = read_double(buffer, buffer_size, &pos, &imu->linear_acceleration_covariance[i]);
        if (ret != ROS_OK) return ret;
    }
    *read = pos;
    return ROS_OK;
}

int ros_serialize_image(const RosImage* image, uint8_t* buffer, size_t buffer_size, size_t* written) {
    size_t pos = 0;
    int ret;
    ret = write_uint32(buffer, buffer_size, &pos, image->width);
    if (ret != ROS_OK) return ret;
    ret = write_uint32(buffer, buffer_size, &pos, image->height);
    if (ret != ROS_OK) return ret;
    ret = write_string(buffer, buffer_size, &pos, image->encoding);
    if (ret != ROS_OK) return ret;
    ret = write_uint8(buffer, buffer_size, &pos, image->is_bigendian);
    if (ret != ROS_OK) return ret;
    ret = write_uint32(buffer, buffer_size, &pos, image->step);
    if (ret != ROS_OK) return ret;
    ret = write_uint32(buffer, buffer_size, &pos, (uint32_t)image->data_size);
    if (ret != ROS_OK) return ret;
    if (image->data_size > 0 && image->data != NULL) {
        if (pos + image->data_size > buffer_size) return ROS_ERROR_SERIALIZE;
        memcpy(buffer + pos, image->data, image->data_size);
        pos += image->data_size;
    }
    *written = pos;
    return ROS_OK;
}

int ros_deserialize_image(const uint8_t* buffer, size_t buffer_size, size_t* read, RosImage* image) {
    size_t pos = 0;
    int ret;
    ret = read_uint32(buffer, buffer_size, &pos, &image->width);
    if (ret != ROS_OK) return ret;
    ret = read_uint32(buffer, buffer_size, &pos, &image->height);
    if (ret != ROS_OK) return ret;
    ret = read_string(buffer, buffer_size, &pos, image->encoding, sizeof(image->encoding));
    if (ret != ROS_OK) return ret;
    ret = read_uint8(buffer, buffer_size, &pos, &image->is_bigendian);
    if (ret != ROS_OK) return ret;
    ret = read_uint32(buffer, buffer_size, &pos, &image->step);
    if (ret != ROS_OK) return ret;
    uint32_t ds;
    ret = read_uint32(buffer, buffer_size, &pos, &ds);
    if (ret != ROS_OK) return ret;
    image->data_size = ds;
    if (image->data_size > 0) {
        if (pos + image->data_size > buffer_size) return ROS_ERROR_DESERIALIZE;
        image->data = (uint8_t*)safe_malloc(image->data_size);
        if (!image->data) return ROS_ERROR_NO_MEMORY;
        memcpy(image->data, buffer + pos, image->data_size);
        pos += image->data_size;
    } else {
        image->data = NULL;
    }
    *read = pos;
    return ROS_OK;
}

int ros_serialize_joint_state(const RosJointState* js, uint8_t* buffer, size_t buffer_size, size_t* written) {
    size_t pos = 0;
    int ret;
    ret = ros_serialize_header(&js->header, buffer, buffer_size, &pos);
    if (ret != ROS_OK) return ret;
    ret = write_int32(buffer, buffer_size, &pos, js->names_count);
    if (ret != ROS_OK) return ret;
    for (int i = 0; i < js->names_count && i < 32; i++) {
        ret = write_string(buffer, buffer_size, &pos, js->name[i]);
        if (ret != ROS_OK) return ret;
    }
    ret = write_int32(buffer, buffer_size, &pos, js->positions_count);
    if (ret != ROS_OK) return ret;
    for (int i = 0; i < js->positions_count && i < 32; i++) {
        ret = write_double(buffer, buffer_size, &pos, js->position[i]);
        if (ret != ROS_OK) return ret;
    }
    ret = write_int32(buffer, buffer_size, &pos, js->velocities_count);
    if (ret != ROS_OK) return ret;
    for (int i = 0; i < js->velocities_count && i < 32; i++) {
        ret = write_double(buffer, buffer_size, &pos, js->velocity[i]);
        if (ret != ROS_OK) return ret;
    }
    ret = write_int32(buffer, buffer_size, &pos, js->efforts_count);
    if (ret != ROS_OK) return ret;
    for (int i = 0; i < js->efforts_count && i < 32; i++) {
        ret = write_double(buffer, buffer_size, &pos, js->effort[i]);
        if (ret != ROS_OK) return ret;
    }
    *written = pos;
    return ROS_OK;
}

int ros_deserialize_joint_state(const uint8_t* buffer, size_t buffer_size, size_t* read, RosJointState* js) {
    size_t pos = 0;
    int ret;
    int32_t n;
    ret = ros_deserialize_header(buffer, buffer_size, &pos, &js->header);
    if (ret != ROS_OK) return ret;
    ret = read_int32(buffer, buffer_size, &pos, &n);
    if (ret != ROS_OK) return ret;
    js->names_count = (n > 32) ? 32 : n;
    for (int i = 0; i < js->names_count; i++) {
        ret = read_string(buffer, buffer_size, &pos, js->name[i], sizeof(js->name[i]));
        if (ret != ROS_OK) return ret;
    }
    ret = read_int32(buffer, buffer_size, &pos, &n);
    if (ret != ROS_OK) return ret;
    js->positions_count = (n > 32) ? 32 : n;
    for (int i = 0; i < js->positions_count; i++) {
        ret = read_double(buffer, buffer_size, &pos, &js->position[i]);
        if (ret != ROS_OK) return ret;
    }
    ret = read_int32(buffer, buffer_size, &pos, &n);
    if (ret != ROS_OK) return ret;
    js->velocities_count = (n > 32) ? 32 : n;
    for (int i = 0; i < js->velocities_count; i++) {
        ret = read_double(buffer, buffer_size, &pos, &js->velocity[i]);
        if (ret != ROS_OK) return ret;
    }
    ret = read_int32(buffer, buffer_size, &pos, &n);
    if (ret != ROS_OK) return ret;
    js->efforts_count = (n > 32) ? 32 : n;
    for (int i = 0; i < js->efforts_count; i++) {
        ret = read_double(buffer, buffer_size, &pos, &js->effort[i]);
        if (ret != ROS_OK) return ret;
    }
    *read = pos;
    return ROS_OK;
}

int ros_serialize_float64_multi_array(const RosFloat64MultiArray* arr, uint8_t* buffer, size_t buffer_size, size_t* written) {
    size_t pos = 0;
    int ret;
    ret = ros_serialize_header(&arr->header, buffer, buffer_size, &pos);
    if (ret != ROS_OK) return ret;
    ret = write_string(buffer, buffer_size, &pos, arr->layout);
    if (ret != ROS_OK) return ret;
    ret = write_int32(buffer, buffer_size, &pos, arr->data_size);
    if (ret != ROS_OK) return ret;
    for (int i = 0; i < arr->data_size && i < 64; i++) {
        ret = write_double(buffer, buffer_size, &pos, arr->data[i]);
        if (ret != ROS_OK) return ret;
    }
    *written = pos;
    return ROS_OK;
}

int ros_deserialize_float64_multi_array(const uint8_t* buffer, size_t buffer_size, size_t* read, RosFloat64MultiArray* arr) {
    size_t pos = 0;
    int ret;
    int32_t n;
    ret = ros_deserialize_header(buffer, buffer_size, &pos, &arr->header);
    if (ret != ROS_OK) return ret;
    ret = read_string(buffer, buffer_size, &pos, arr->layout, sizeof(arr->layout));
    if (ret != ROS_OK) return ret;
    ret = read_int32(buffer, buffer_size, &pos, &n);
    if (ret != ROS_OK) return ret;
    arr->data_size = (n > 64) ? 64 : n;
    for (int i = 0; i < arr->data_size; i++) {
        ret = read_double(buffer, buffer_size, &pos, &arr->data[i]);
        if (ret != ROS_OK) return ret;
    }
    *read = pos;
    return ROS_OK;
}

int ros_serialize_pointcloud(const RosPointCloud* pc, uint8_t* buffer, size_t buffer_size, size_t* written) {
    size_t pos = 0;
    int ret;
    ret = ros_serialize_header(&pc->header, buffer, buffer_size, &pos);
    if (ret != ROS_OK) return ret;
    ret = write_int32(buffer, buffer_size, &pos, pc->points_count);
    if (ret != ROS_OK) return ret;
    for (int i = 0; i < pc->points_count && i < 1024; i++) {
        ret = write_float(buffer, buffer_size, &pos, pc->points[0][i]);
        if (ret != ROS_OK) return ret;
        ret = write_float(buffer, buffer_size, &pos, pc->points[1][i]);
        if (ret != ROS_OK) return ret;
        ret = write_float(buffer, buffer_size, &pos, pc->points[2][i]);
        if (ret != ROS_OK) return ret;
    }
    *written = pos;
    return ROS_OK;
}

int ros_deserialize_pointcloud(const uint8_t* buffer, size_t buffer_size, size_t* read, RosPointCloud* pc) {
    size_t pos = 0;
    int ret;
    int32_t n;
    ret = ros_deserialize_header(buffer, buffer_size, &pos, &pc->header);
    if (ret != ROS_OK) return ret;
    ret = read_int32(buffer, buffer_size, &pos, &n);
    if (ret != ROS_OK) return ret;
    pc->points_count = (n > 1024) ? 1024 : n;
    for (int i = 0; i < pc->points_count; i++) {
        ret = read_float(buffer, buffer_size, &pos, &pc->points[0][i]);
        if (ret != ROS_OK) return ret;
        ret = read_float(buffer, buffer_size, &pos, &pc->points[1][i]);
        if (ret != ROS_OK) return ret;
        ret = read_float(buffer, buffer_size, &pos, &pc->points[2][i]);
        if (ret != ROS_OK) return ret;
    }
    *read = pos;
    return ROS_OK;
}

int ros_serialize_log(const RosLog* log, uint8_t* buffer, size_t buffer_size, size_t* written) {
    size_t pos = 0;
    int ret;
    ret = write_int32(buffer, buffer_size, &pos, log->level);
    if (ret != ROS_OK) return ret;
    ret = write_string(buffer, buffer_size, &pos, log->name);
    if (ret != ROS_OK) return ret;
    ret = write_string(buffer, buffer_size, &pos, log->msg);
    if (ret != ROS_OK) return ret;
    ret = write_string(buffer, buffer_size, &pos, log->file);
    if (ret != ROS_OK) return ret;
    ret = write_int32(buffer, buffer_size, &pos, log->line);
    if (ret != ROS_OK) return ret;
    ret = write_string(buffer, buffer_size, &pos, log->function);
    if (ret != ROS_OK) return ret;
    *written = pos;
    return ROS_OK;
}

int ros_deserialize_log(const uint8_t* buffer, size_t buffer_size, size_t* read, RosLog* log) {
    size_t pos = 0;
    int ret;
    ret = read_int32(buffer, buffer_size, &pos, &log->level);
    if (ret != ROS_OK) return ret;
    ret = read_string(buffer, buffer_size, &pos, log->name, sizeof(log->name));
    if (ret != ROS_OK) return ret;
    ret = read_string(buffer, buffer_size, &pos, log->msg, sizeof(log->msg));
    if (ret != ROS_OK) return ret;
    ret = read_string(buffer, buffer_size, &pos, log->file, sizeof(log->file));
    if (ret != ROS_OK) return ret;
    ret = read_int32(buffer, buffer_size, &pos, &log->line);
    if (ret != ROS_OK) return ret;
    ret = read_string(buffer, buffer_size, &pos, log->function, sizeof(log->function));
    if (ret != ROS_OK) return ret;
    *read = pos;
    return ROS_OK;
}

const char* ros_error_string(int error_code) {
    switch (error_code) {
        case 0: return "成功";
        case -1: return "一般错误";
        case -2: return "超时";
        case -3: return "连接错误";
        case -4: return "未连接";
        case -5: return "已存在";
        case -6: return "未找到";
        case -7: return "无效参数";
        case -8: return "内存不足";
        case -9: return "序列化错误";
        case -10: return "反序列化错误";
        case -11: return "协议错误";
        case -12: return "Master错误";
        case -13: return "服务错误";
        default: return "未知错误";
    }
}

int ros_build_connection_header(RosConnectionHeader* header, const char* topic,
                                 const char* type, const char* md5sum, int is_tcp) {
    if (!header || !topic || !type || !md5sum) return ROS_ERROR_INVALID_PARAM;
    memset(header, 0, sizeof(*header));
    strncpy(header->topic_name, topic, sizeof(header->topic_name) - 1);
    strncpy(header->type_name, type, sizeof(header->type_name) - 1);
    strncpy(header->md5sum, md5sum, sizeof(header->md5sum) - 1);
    header->transport = is_tcp ? ROS_TRANSPORT_TCP : ROS_TRANSPORT_UDP;
    header->flags = ROS_MSG_FLAG_TCP_NODELAY;
    return ROS_OK;
}

const char* ros_topic_type_to_string(int type_id) {
    switch (type_id) {
        case 0: return "geometry_msgs/Twist";
        case 1: return "geometry_msgs/Pose";
        case 2: return "nav_msgs/Odometry";
        case 3: return "sensor_msgs/LaserScan";
        case 4: return "sensor_msgs/Joy";
        case 5: return "sensor_msgs/Imu";
        case 6: return "sensor_msgs/Image";
        case 7: return "sensor_msgs/JointState";
        case 8: return "sensor_msgs/PointCloud";
        case 9: return "std_msgs/Float64MultiArray";
        case 10: return "rosgraph_msgs/Log";
        default: return "unknown";
    }
}

int ros_get_md5_from_type(const char* type_name, char* md5_out, size_t md5_size) {
    if (!type_name || !md5_out || md5_size < 64) return ROS_ERROR_INVALID_PARAM;
    if (strcmp(type_name, "geometry_msgs/Twist") == 0) {
        strncpy(md5_out, "9f195f881246fdfa2798d1d3eebca84a", md5_size - 1);
    } else if (strcmp(type_name, "geometry_msgs/Pose") == 0) {
        strncpy(md5_out, "e45d45a5a1ce597b249b23c4c4cc63a3", md5_size - 1);
    } else if (strcmp(type_name, "nav_msgs/Odometry") == 0) {
        strncpy(md5_out, "cd5e73d190d2a5f92f8de49c5b84c75f", md5_size - 1);
    } else if (strcmp(type_name, "sensor_msgs/LaserScan") == 0) {
        strncpy(md5_out, "90c7ef2dc6895d81024acba2ac42f369", md5_size - 1);
    } else if (strcmp(type_name, "sensor_msgs/Joy") == 0) {
        strncpy(md5_out, "5a9ea5f83505693b21e4235cd672e783", md5_size - 1);
    } else if (strcmp(type_name, "sensor_msgs/Imu") == 0) {
        strncpy(md5_out, "6a62c6daae103f4ff57a132d6f95cec2", md5_size - 1);
    } else if (strcmp(type_name, "sensor_msgs/Image") == 0) {
        strncpy(md5_out, "060021388200f6f0f447d0fcd9c64743", md5_size - 1);
    } else if (strcmp(type_name, "sensor_msgs/JointState") == 0) {
        strncpy(md5_out, "3066dcd76a6cfaef579bd0f34173eccd", md5_size - 1);
    } else if (strcmp(type_name, "sensor_msgs/PointCloud") == 0) {
        strncpy(md5_out, "c8c7a4395c4b3b99b7e4ba2b7c3c8f6a", md5_size - 1);
    } else if (strcmp(type_name, "std_msgs/Float64MultiArray") == 0) {
        strncpy(md5_out, "4b7d974086d4060e7db4613a4e6c9f2c", md5_size - 1);
    } else if (strcmp(type_name, "rosgraph_msgs/Log") == 0) {
        strncpy(md5_out, "acffd30cd6b6de30f120938c17c593fb", md5_size - 1);
    } else if (strcmp(type_name, "tf/tfMessage") == 0) {
        strncpy(md5_out, "94810edda583a504dfda3829e70d7eec", md5_size - 1);
    } else {
        strncpy(md5_out, "*", md5_size - 1);
    }
    md5_out[md5_size - 1] = '\0';
    return ROS_OK;
}

/* ================================================================
 * ROS TCP传输层实现 - 真实TCP+XMLRPC Master通信
 * ================================================================ */

typedef struct {
    socket_t fd;
    char remote_addr[64];
    int remote_port;
    int connected;
    int authenticated;
} RosTcpConnection;

static RosTcpConnection g_ros_tcp_conn = {0};
static uint8_t g_ros_tcp_rx_buf[65536];
static uint8_t g_ros_tcp_tx_buf[65536];

int ros_tcp_connect(const char* host, int port, int timeout_ms) {
    if (!host || port <= 0) return ROS_ERROR_INVALID_PARAM;
    if (g_ros_tcp_conn.connected) return ROS_OK;

    struct addrinfo hints, *res;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);
    if (getaddrinfo(host, port_str, &hints, &res) != 0) return ROS_ERROR_NETWORK;

    g_ros_tcp_conn.fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (g_ros_tcp_conn.fd == INVALID_SOCKET_VAL) {
        freeaddrinfo(res);
        return ROS_ERROR_NETWORK;
    }

    if (timeout_ms > 0) {
        DWORD tv = (DWORD)timeout_ms;
        setsockopt(g_ros_tcp_conn.fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
        setsockopt(g_ros_tcp_conn.fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));
    }

    int ret = connect(g_ros_tcp_conn.fd, res->ai_addr, (int)res->ai_addrlen);
    freeaddrinfo(res);
    if (ret != 0) { CLOSE_SOCKET(g_ros_tcp_conn.fd); return ROS_ERROR_NETWORK; }

    g_ros_tcp_conn.connected = 1;
    strncpy(g_ros_tcp_conn.remote_addr, host, sizeof(g_ros_tcp_conn.remote_addr) - 1);
    g_ros_tcp_conn.remote_port = port;
    return ROS_OK;
}

int ros_tcp_disconnect(void) {
    if (!g_ros_tcp_conn.connected) return ROS_OK;
    CLOSE_SOCKET(g_ros_tcp_conn.fd);
    memset(&g_ros_tcp_conn, 0, sizeof(g_ros_tcp_conn));
    return ROS_OK;
}

static int ros_tcp_send_raw(const void* data, size_t len) {
    if (!g_ros_tcp_conn.connected) return -1;
    uint32_t net_len = htonl((uint32_t)len);
    int sent = (int)send(g_ros_tcp_conn.fd, (const char*)&net_len, 4, 0);
    if (sent != 4) return -1;
    size_t total_sent = 0;
    while (total_sent < len) {
        sent = (int)send(g_ros_tcp_conn.fd, (const char*)data + total_sent, (int)(len - total_sent), 0);
        if (sent <= 0) return -1;
        total_sent += (size_t)sent;
    }
    return 0;
}

static int ros_tcp_recv_raw(void* data, size_t* len) {
    if (!g_ros_tcp_conn.connected || !len) return -1;
    uint32_t net_len = 0;
    int rcvd = (int)recv(g_ros_tcp_conn.fd, (char*)&net_len, 4, MSG_WAITALL);
    if (rcvd != 4) return -1;
    *len = ntohl(net_len);
    if (*len > 65536) return -1;
    size_t total_rcvd = 0;
    while (total_rcvd < *len) {
        rcvd = (int)recv(g_ros_tcp_conn.fd, (char*)data + total_rcvd, (int)(*len - total_rcvd), 0);
        if (rcvd <= 0) return -1;
        total_rcvd += (size_t)rcvd;
    }
    return 0;
}

int ros_master_register_publisher(const char* caller_id, const char* topic,
                                   const char* topic_type, const char* caller_api) {
    if (!caller_id || !topic || !topic_type) return ROS_ERROR_INVALID_PARAM;

    char xmlrpc_req[2048];
    snprintf(xmlrpc_req, sizeof(xmlrpc_req),
             "<?xml version=\"1.0\"?>"
             "<methodCall><methodName>registerPublisher</methodName>"
             "<params><param><value><string>%s</string></value></param>"
             "<param><value><string>%s</string></value></param>"
             "<param><value><string>%s</string></value></param>"
             "<param><value><string>%s</string></value></param>"
             "</params></methodCall>",
             caller_id, topic, topic_type, caller_api ? caller_api : "http://localhost:0/");

    size_t resp_len;
    memset(g_ros_tcp_rx_buf, 0, sizeof(g_ros_tcp_rx_buf));
    if (ros_tcp_send_raw(xmlrpc_req, strlen(xmlrpc_req)) != 0) return ROS_ERROR_NETWORK;
    if (ros_tcp_recv_raw(g_ros_tcp_rx_buf, &resp_len) != 0) return ROS_ERROR_NETWORK;

    return (strstr((char*)g_ros_tcp_rx_buf, "faultCode") != NULL) ? ROS_ERROR_PROTOCOL : ROS_OK;
}

int ros_tcp_negotiate_subscriber(const char* topic, const char* topic_type,
                                  const char* caller_api, RosConnectionHeader* header_out) {
    if (!topic || !topic_type || !header_out) return ROS_ERROR_INVALID_PARAM;

    char req[2048];
    snprintf(req, sizeof(req),
             "<?xml version=\"1.0\"?>"
             "<methodCall><methodName>requestTopic</methodName>"
             "<params><param><value><string>caller</string></value></param>"
             "<param><value><string>%s</string></value></param>"
             "<param><value><array><data>"
             "<value><array><data><value><string>TCPROS</string></value></data></array></value>"
             "</data></array></value></param></params></methodCall>", topic);

    size_t resp_len;
    memset(g_ros_tcp_rx_buf, 0, sizeof(g_ros_tcp_rx_buf));
    if (ros_tcp_send_raw(req, strlen(req)) != 0) return ROS_ERROR_NETWORK;
    if (ros_tcp_recv_raw(g_ros_tcp_rx_buf, &resp_len) != 0) return ROS_ERROR_NETWORK;

    ros_build_connection_header(header_out, topic, topic_type, caller_api, 1);
    header_out->md5sum[0] = '*';
    return ROS_OK;
}

int ros_tcp_publish_message(const uint8_t* serialized_data, size_t data_size) {
    if (!serialized_data || data_size == 0) return ROS_ERROR_INVALID_PARAM;
    if (!g_ros_tcp_conn.connected) return ROS_ERROR_NETWORK;
    return ros_tcp_send_raw(serialized_data, data_size);
}

int ros_tcp_subscribe_loop(int (*callback)(const uint8_t*, size_t, void*), void* user_data) {
    if (!callback) return ROS_ERROR_INVALID_PARAM;
    if (!g_ros_tcp_conn.connected) return ROS_ERROR_NETWORK;

    while (g_ros_tcp_conn.connected) {
        size_t data_len = 0;
        memset(g_ros_tcp_rx_buf, 0, sizeof(g_ros_tcp_rx_buf));
        if (ros_tcp_recv_raw(g_ros_tcp_rx_buf, &data_len) != 0) break;
        if (data_len > 0) {
            callback(g_ros_tcp_rx_buf, data_len, user_data);
        }
    }
    return ROS_OK;
}

// ============================================================================
// DDS GUID 生成
// ============================================================================

void dds_generate_guid(DDSGuid* guid, uint32_t participant_id,
                       uint8_t entity_kind, uint16_t entity_id) {
    if (!guid) return;
    // 前缀：使用时间戳 + 参与者ID 构建唯一标识
    uint64_t now = (uint64_t)time(NULL);
    guid->prefix[0] = (uint8_t)(now >> 40);
    guid->prefix[1] = (uint8_t)(now >> 32);
    guid->prefix[2] = (uint8_t)(now >> 24);
    guid->prefix[3] = (uint8_t)(now >> 16);
    guid->prefix[4] = (uint8_t)(now >> 8);
    guid->prefix[5] = (uint8_t)(now >> 0);
    // 后6字节使用参与者ID的哈希
    for (int i = 0; i < 6; i++) {
        guid->prefix[6 + i] = (uint8_t)(participant_id >> (i * 8));
    }
    // 实体ID
    guid->entity_id[0] = entity_kind;
    guid->entity_id[1] = (uint8_t)(entity_id >> 8);
    guid->entity_id[2] = (uint8_t)(entity_id >> 0);
    guid->entity_id[3] = 0;
}

// ============================================================================
// DDS RTPS 数据序列化/反序列化
// ============================================================================

#define RTPS_SUBMESSAGE_DATA      0x15
#define RTPS_SUBMESSAGE_HEARTBEAT 0x07
#define RTPS_SUBMESSAGE_ACKNACK   0x06
#define RTPS_SUBMESSAGE_GAP       0x08
#define RTPS_SUBMESSAGE_INFO_TS   0x09
#define RTPS_SUBMESSAGE_INFO_DST  0x0E
#define RTPS_SUBMESSAGE_INFO_SRC  0x0F

int dds_serialize_rtps_data(uint8_t* buffer, size_t buffer_size, size_t* written,
                            const DDSGuid* writer_id, const DDSGuid* reader_id,
                            uint32_t sequence_number, const uint8_t* data,
                            size_t data_size) {
    if (!buffer || !written || !writer_id || !reader_id) return -1;
    size_t pos = 0;

    // RTPS 消息头 (4字节)
    if (pos + 4 > buffer_size) return -1;
    buffer[pos++] = DDS_PROTOCOL_VERSION_MAJOR;  // 协议主版本
    buffer[pos++] = DDS_PROTOCOL_VERSION_MINOR;  // 协议次版本
    buffer[pos++] = (uint8_t)(DDS_VENDOR_ID >> 8);   // 厂商ID高字节
    buffer[pos++] = (uint8_t)(DDS_VENDOR_ID & 0xFF);   // 厂商ID低字节

    // DATA 子消息头 (4字节)
    size_t submsg_start = pos;
    if (pos + 4 > buffer_size) return -1;
    buffer[pos++] = RTPS_SUBMESSAGE_DATA;  // 子消息ID
    buffer[pos++] = 0x01;                  // flags: 小端序
    /* 子消息长度字段：预留2字节位置，数据写入完成后回填实际长度 */
    uint16_t length_pos = (uint16_t)pos;
    pos += 2;

    // extra_flags (2字节)
    if (pos + 2 > buffer_size) return -1;
    buffer[pos++] = 0;
    buffer[pos++] = 0;

    // octets_to_inline_qos (2字节)
    if (pos + 2 > buffer_size) return -1;
    buffer[pos++] = 0;
    buffer[pos++] = 0;

    // reader_id (16字节) - 全0表示任意reader
    if (pos + DDS_GUID_LEN > buffer_size) return -1;
    memset(buffer + pos, 0, DDS_GUID_LEN);
    pos += DDS_GUID_LEN;

    // writer_id (16字节)
    if (pos + DDS_GUID_LEN > buffer_size) return -1;
    memcpy(buffer + pos, writer_id, DDS_GUID_LEN);
    pos += DDS_GUID_LEN;

    // sequence_number (4字节)
    if (pos + 4 > buffer_size) return -1;
    buffer[pos++] = (uint8_t)(sequence_number >> 0);
    buffer[pos++] = (uint8_t)(sequence_number >> 8);
    buffer[pos++] = (uint8_t)(sequence_number >> 16);
    buffer[pos++] = (uint8_t)(sequence_number >> 24);

    // 内联 QOS (可选，跳过)
    // 序列化数据长度 (4字节)
    uint32_t data_len = (uint32_t)data_size;
    if (pos + 4 + data_len > buffer_size) return -1;
    buffer[pos++] = (uint8_t)(data_len >> 0);
    buffer[pos++] = (uint8_t)(data_len >> 8);
    buffer[pos++] = (uint8_t)(data_len >> 16);
    buffer[pos++] = (uint8_t)(data_len >> 24);

    // 序列化数据负载 (CDR 编码)
    if (data && data_size > 0) {
        memcpy(buffer + pos, data, data_size);
        pos += data_size;
    }

    // 回填子消息长度
    uint16_t submsg_len = (uint16_t)(pos - submsg_start - 4);
    buffer[length_pos] = (uint8_t)(submsg_len >> 0);
    buffer[length_pos + 1] = (uint8_t)(submsg_len >> 8);

    *written = pos;
    return 0;
}

int dds_deserialize_rtps_data(const uint8_t* buffer, size_t buffer_size,
                              size_t* read, DDSDataSubmessage* submsg,
                              uint8_t* data_out, size_t data_out_size,
                              size_t* data_read) {
    if (!buffer || !read || !submsg || !data_read) return -1;
    size_t pos = 0;

    // 跳过RTPS头 (4字节)
    if (pos + 4 > buffer_size) return -1;
    uint8_t ver_major = buffer[pos++];
    uint8_t ver_minor = buffer[pos++];
    (void)ver_major; (void)ver_minor;
    pos += 2; // vendor_id

    // 子消息头 (4字节)
    if (pos + 4 > buffer_size) return -1;
    submsg->header.submessage_id = buffer[pos++];
    submsg->header.flags = buffer[pos++];
    submsg->header.length = (uint16_t)(buffer[pos] | (buffer[pos + 1] << 8));
    pos += 2;

    if (submsg->header.submessage_id != RTPS_SUBMESSAGE_DATA) {
        return -1; // 不是DATA子消息
    }

    // extra_flags (2字节)
    if (pos + 2 > buffer_size) return -1;
    submsg->extra_flags = (uint16_t)(buffer[pos] | (buffer[pos + 1] << 8));
    pos += 2;

    // octets_to_inline_qos (2字节)
    if (pos + 2 > buffer_size) return -1;
    submsg->octets_to_inline_qos = (uint16_t)(buffer[pos] | (buffer[pos + 1] << 8));
    pos += 2;

    // reader_id (16字节)
    if (pos + DDS_GUID_LEN > buffer_size) return -1;
    memcpy(&submsg->reader_id, buffer + pos, DDS_GUID_LEN);
    pos += DDS_GUID_LEN;

    // writer_id (16字节)
    if (pos + DDS_GUID_LEN > buffer_size) return -1;
    memcpy(&submsg->writer_id, buffer + pos, DDS_GUID_LEN);
    pos += DDS_GUID_LEN;

    // sequence_number (4字节)
    if (pos + 4 > buffer_size) return -1;
    submsg->sequence_number = (uint32_t)(buffer[pos] |
                            (buffer[pos + 1] << 8) |
                            (buffer[pos + 2] << 16) |
                            (buffer[pos + 3] << 24));
    pos += 4;

    // 内联 QOS 跳过
    uint32_t inline_qos_len = submsg->octets_to_inline_qos;
    if (inline_qos_len > 0) {
        pos += inline_qos_len;
    }

    // 数据长度 (4字节)
    if (pos + 4 > buffer_size) return -1;
    uint32_t payload_len = (uint32_t)(buffer[pos] |
                           (buffer[pos + 1] << 8) |
                           (buffer[pos + 2] << 16) |
                           (buffer[pos + 3] << 24));
    pos += 4;

    // 数据负载
    if (payload_len > 0) {
        if (pos + payload_len > buffer_size) return -1;
        size_t copy_len = (payload_len < (uint32_t)data_out_size) ?
                          (size_t)payload_len : data_out_size;
        memcpy(data_out, buffer + pos, copy_len);
        *data_read = copy_len;
        pos += payload_len;
    } else {
        *data_read = 0;
    }

    *read = pos;
    return 0;
}

// ============================================================================
// DDS 上下文
// ============================================================================

struct DDSContext {
    uint32_t domain_id;
    char node_name[ROS_MAX_NODE_NAME];
    char node_namespace[ROS_MAX_TOPIC_NAME];
    DDSGuid participant_guid;

    // 本地端点
    int num_writers;
    int num_readers;
    struct {
        char topic_name[ROS_MAX_TOPIC_NAME];
        char type_name[ROS_MAX_TYPE_NAME];
        DDSGuid guid;
        int is_reliable;
        int is_active;
    } writers[DDS_MAX_ENDPOINTS];
    struct {
        char topic_name[ROS_MAX_TOPIC_NAME];
        char type_name[ROS_MAX_TYPE_NAME];
        DDSGuid guid;
        int is_reliable;
        int is_active;
        DDSTopicCallback callback;
        void* user_data;
    } readers[DDS_MAX_ENDPOINTS];

    // 远程参与者
    int num_participants;
    DDSParticipantData remote_participants[DDS_MAX_PARTICIPANTS];

    // Socket
    socket_t discovery_sock;
    socket_t user_sock;
    int is_running;
    uint32_t participant_serial;
    uint32_t discovery_port;
    uint32_t user_traffic_port;

    // 时间跟踪
    int64_t last_discovery_ms;
    int64_t current_time_ms;
};

static int64_t dds_get_time_ms(void) {
#if defined(SELFLNN_PLATFORM_WINDOWS) && SELFLNN_PLATFORM_WINDOWS
    return (int64_t)GetTickCount64();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + (int64_t)(ts.tv_nsec / 1000000);
#endif
}

DDSContext* dds_context_create(uint32_t domain_id, const char* node_name,
                               const char* node_namespace) {
    DDSContext* ctx = (DDSContext*)safe_calloc(1, sizeof(DDSContext));
    if (!ctx) return NULL;

    ctx->domain_id = domain_id;
    if (node_name) {
        strncpy(ctx->node_name, node_name, sizeof(ctx->node_name) - 1);
    } else {
        strncpy(ctx->node_name, "self_lnn_node", sizeof(ctx->node_name) - 1);
    }
    if (node_namespace) {
        strncpy(ctx->node_namespace, node_namespace, sizeof(ctx->node_namespace) - 1);
    } else {
        ctx->node_namespace[0] = '/';
        ctx->node_namespace[1] = '\0';
    }

    // 生成参与者 GUID
    ctx->participant_serial = (uint32_t)(time(NULL) & 0xFFFFFFFF);
    dds_generate_guid(&ctx->participant_guid, ctx->participant_serial,
                      (uint8_t)DDS_ENTITY_PARTICIPANT, 0);

    ctx->discovery_sock = INVALID_SOCKET_VAL;
    ctx->user_sock = INVALID_SOCKET_VAL;
    ctx->is_running = 0;
    ctx->current_time_ms = dds_get_time_ms();
    ctx->last_discovery_ms = ctx->current_time_ms;

    return ctx;
}

void dds_context_free(DDSContext* ctx) {
    if (!ctx) return;

    if (ctx->is_running) {
        dds_discovery_stop(ctx);
    }

    if (ctx->discovery_sock != INVALID_SOCKET_VAL) {
        CLOSE_SOCKET(ctx->discovery_sock);
    }
    if (ctx->user_sock != INVALID_SOCKET_VAL) {
        CLOSE_SOCKET(ctx->user_sock);
    }

    safe_free((void**)&ctx);
}

#if defined(SELFLNN_PLATFORM_WINDOWS) && SELFLNN_PLATFORM_WINDOWS
static int dds_init_winsock(void) {
    static int winsock_initialized = 0;
    if (!winsock_initialized) {
        WSADATA wsa_data;
        if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
            return -1;
        }
        winsock_initialized = 1;
    }
    return 0;
}
#endif

static int dds_create_udp_socket(socket_t* sock_out, uint16_t port,
                                 int enable_reuse, int join_multicast) {
    socket_t sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock == INVALID_SOCKET_VAL) {
        return -1;
    }

    if (enable_reuse) {
#if defined(SELFLNN_PLATFORM_WINDOWS) && SELFLNN_PLATFORM_WINDOWS
        BOOL reuse = TRUE;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));
#else
        int reuse = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
#endif
    }

    if (join_multicast) {
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(port);
        if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            CLOSE_SOCKET(sock);
            return -1;
        }

        struct ip_mreq mreq;
        mreq.imr_multiaddr.s_addr = inet_addr(DDS_MULTICAST_ADDRESS);
        mreq.imr_interface.s_addr = htonl(INADDR_ANY);
        if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP,
                       (const char*)&mreq, sizeof(mreq)) < 0) {
            CLOSE_SOCKET(sock);
            return -1;
        }
    } else {
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(port);
        if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            CLOSE_SOCKET(sock);
            return -1;
        }
    }

    // 设置非阻塞
#if defined(SELFLNN_PLATFORM_WINDOWS) && SELFLNN_PLATFORM_WINDOWS
    u_long nonblock = 1;
    ioctlsocket(sock, FIONBIO, &nonblock);
#else
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);
#endif

    *sock_out = sock;
    return 0;
}

int dds_discovery_start(DDSContext* ctx) {
    if (!ctx) return -1;

#if defined(SELFLNN_PLATFORM_WINDOWS) && SELFLNN_PLATFORM_WINDOWS
    if (dds_init_winsock() != 0) return -1;
#endif

    ctx->user_traffic_port = DDS_USER_TRAFFIC_PORT_BASE + ctx->domain_id;
    ctx->discovery_port = DDS_DISCOVERY_PORT_BASE + ctx->domain_id;

    // 创建发现 socket（多播）
    if (dds_create_udp_socket(&ctx->discovery_sock,
                              (uint16_t)ctx->discovery_port, 1, 1) != 0) {
        // 回退：只发送不接收
        ctx->discovery_sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (ctx->discovery_sock == INVALID_SOCKET_VAL) {
            return -1;
        }
    }

    // 创建用户数据 socket
    if (dds_create_udp_socket(&ctx->user_sock,
                              (uint16_t)ctx->user_traffic_port, 1, 0) != 0) {
        ctx->user_sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (ctx->user_sock == INVALID_SOCKET_VAL) {
            CLOSE_SOCKET(ctx->discovery_sock);
            ctx->discovery_sock = INVALID_SOCKET_VAL;
            return -1;
        }
    }

    ctx->is_running = 1;
    ctx->last_discovery_ms = ctx->current_time_ms;

    return 0;
}

int dds_discovery_stop(DDSContext* ctx) {
    if (!ctx) return -1;
    ctx->is_running = 0;
    if (ctx->discovery_sock != INVALID_SOCKET_VAL) {
        // 离开多播组
        struct ip_mreq mreq;
        mreq.imr_multiaddr.s_addr = inet_addr(DDS_MULTICAST_ADDRESS);
        mreq.imr_interface.s_addr = htonl(INADDR_ANY);
        setsockopt(ctx->discovery_sock, IPPROTO_IP, IP_DROP_MEMBERSHIP,
                   (const char*)&mreq, sizeof(mreq));
    }
    return 0;
}

// 发送 SPDP 发现数据包
static int dds_send_spdp(DDSContext* ctx) {
    if (!ctx || ctx->discovery_sock == INVALID_SOCKET_VAL) return -1;

    uint8_t buffer[1024];
    size_t pos = 0;

    // 构建 SPDP 数据
    // 参与者 GUID
    memcpy(buffer + pos, &ctx->participant_guid, DDS_GUID_LEN);
    pos += DDS_GUID_LEN;

    // 节点名称
    uint16_t name_len = (uint16_t)strlen(ctx->node_name);
    buffer[pos++] = (uint8_t)(name_len >> 0);
    buffer[pos++] = (uint8_t)(name_len >> 8);
    memcpy(buffer + pos, ctx->node_name, name_len);
    pos += name_len;

    // 域ID
    buffer[pos++] = (uint8_t)(ctx->domain_id >> 0);
    buffer[pos++] = (uint8_t)(ctx->domain_id >> 8);
    buffer[pos++] = (uint8_t)(ctx->domain_id >> 16);
    buffer[pos++] = (uint8_t)(ctx->domain_id >> 24);

    // 用户流量端口
    buffer[pos++] = (uint8_t)(ctx->user_traffic_port >> 0);
    buffer[pos++] = (uint8_t)(ctx->user_traffic_port >> 8);
    buffer[pos++] = (uint8_t)(ctx->user_traffic_port >> 16);
    buffer[pos++] = (uint8_t)(ctx->user_traffic_port >> 24);

    // Writer 数量
    buffer[pos++] = (uint8_t)(ctx->num_writers >> 0);
    buffer[pos++] = (uint8_t)(ctx->num_writers >> 8);

    // Reader 数量
    buffer[pos++] = (uint8_t)(ctx->num_readers >> 0);
    buffer[pos++] = (uint8_t)(ctx->num_readers >> 8);

    // 写入每个 Writer 信息
    for (int i = 0; i < ctx->num_writers; i++) {
        memcpy(buffer + pos, &ctx->writers[i].guid, DDS_GUID_LEN);
        pos += DDS_GUID_LEN;
        uint16_t tlen = (uint16_t)strlen(ctx->writers[i].topic_name);
        buffer[pos++] = (uint8_t)(tlen >> 0);
        buffer[pos++] = (uint8_t)(tlen >> 8);
        memcpy(buffer + pos, ctx->writers[i].topic_name, tlen);
        pos += tlen;
    }

    // 写入每个 Reader 信息
    for (int i = 0; i < ctx->num_readers; i++) {
        memcpy(buffer + pos, &ctx->readers[i].guid, DDS_GUID_LEN);
        pos += DDS_GUID_LEN;
        uint16_t tlen = (uint16_t)strlen(ctx->readers[i].topic_name);
        buffer[pos++] = (uint8_t)(tlen >> 0);
        buffer[pos++] = (uint8_t)(tlen >> 8);
        memcpy(buffer + pos, ctx->readers[i].topic_name, tlen);
        pos += tlen;
    }

    // 发送到多播地址
    struct sockaddr_in dest;
    memset(&dest, 0, sizeof(dest));
    dest.sin_family = AF_INET;
    dest.sin_addr.s_addr = inet_addr(DDS_MULTICAST_ADDRESS);
    dest.sin_port = htons((uint16_t)ctx->discovery_port);

    sendto(ctx->discovery_sock, (const char*)buffer, (int)pos, 0,
           (struct sockaddr*)&dest, sizeof(dest));

    return 0;
}

// 解析 SPDP 发现数据包
static int dds_recv_spdp(DDSContext* ctx) {
    if (!ctx || ctx->discovery_sock == INVALID_SOCKET_VAL) return -1;

    uint8_t buffer[2048];
    struct sockaddr_in src;
    socklen_t src_len = sizeof(src);

    int recv_len = recvfrom(ctx->discovery_sock, (char*)buffer, (int)sizeof(buffer),
                            0, (struct sockaddr*)&src, &src_len);
    if (recv_len <= 0) return 0;

    size_t pos = 0;

    // 解析参与者信息
    DDSDomainParticipant participant;
    memset(&participant, 0, sizeof(participant));

    // GUID
    if (pos + DDS_GUID_LEN > (size_t)recv_len) return -1;
    memcpy(&participant.guid, buffer + pos, DDS_GUID_LEN);
    pos += DDS_GUID_LEN;

    // 跳过自身
    if (memcmp(&participant.guid, &ctx->participant_guid, DDS_GUID_LEN) == 0) {
        return 0;
    }

    // 节点名称
    if (pos + 2 > (size_t)recv_len) return -1;
    uint16_t name_len = (uint16_t)(buffer[pos] | (buffer[pos + 1] << 8));
    pos += 2;
    if (pos + name_len > (size_t)recv_len) return -1;
    if (name_len > 0 && name_len < ROS_MAX_NODE_NAME) {
        memcpy(participant.node_name, buffer + pos, name_len);
        participant.node_name[name_len] = '\0';
    }
    pos += name_len;

    // 域ID
    if (pos + 4 > (size_t)recv_len) return -1;
    participant.domain_id = (uint32_t)(buffer[pos] | (buffer[pos + 1] << 8) |
                          (buffer[pos + 2] << 16) | (buffer[pos + 3] << 24));
    pos += 4;

    // 过滤不同域
    if (participant.domain_id != ctx->domain_id) return 0;

    // 用户流量端口
    if (pos + 4 > (size_t)recv_len) return -1;
    participant.user_traffic_port = (uint32_t)(buffer[pos] | (buffer[pos + 1] << 8) |
                                    (buffer[pos + 2] << 16) | (buffer[pos + 3] << 24));
    pos += 4;

    // Writer数量
    if (pos + 2 > (size_t)recv_len) return -1;
    uint16_t num_writers = (uint16_t)(buffer[pos] | (buffer[pos + 1] << 8));
    pos += 2;

    // Reader数量
    if (pos + 2 > (size_t)recv_len) return -1;
    uint16_t num_readers = (uint16_t)(buffer[pos] | (buffer[pos + 1] << 8));
    pos += 2;

    participant.last_seen_ms = ctx->current_time_ms;
    participant.is_alive = 1;
    participant.protocol_version_major = DDS_PROTOCOL_VERSION_MAJOR;
    participant.protocol_version_minor = DDS_PROTOCOL_VERSION_MINOR;
    participant.vendor_id = DDS_VENDOR_ID;

    // 查找或创建参与者
    int p_idx = -1;
    for (int i = 0; i < ctx->num_participants; i++) {
        if (memcmp(&ctx->remote_participants[i].participant.guid,
                   &participant.guid, DDS_GUID_LEN) == 0) {
            p_idx = i;
            break;
        }
    }
    if (p_idx < 0) {
        if (ctx->num_participants >= DDS_MAX_PARTICIPANTS) return 0;
        p_idx = ctx->num_participants;
        ctx->remote_participants[p_idx].num_publishers = 0;
        ctx->remote_participants[p_idx].num_subscribers = 0;
        ctx->num_participants++;
    }

    ctx->remote_participants[p_idx].participant = participant;
    ctx->remote_participants[p_idx].num_publishers = 0;
    ctx->remote_participants[p_idx].num_subscribers = 0;

    // 解析 Writers
    for (uint16_t i = 0; i < num_writers; i++) {
        if (ctx->remote_participants[p_idx].num_publishers >= DDS_MAX_ENDPOINTS) break;
        if (pos + DDS_GUID_LEN > (size_t)recv_len) break;
        int ep_idx = ctx->remote_participants[p_idx].num_publishers;
        memcpy(&ctx->remote_participants[p_idx].publishers[ep_idx].guid,
               buffer + pos, DDS_GUID_LEN);
        pos += DDS_GUID_LEN;
        if (pos + 2 > (size_t)recv_len) break;
        uint16_t tlen = (uint16_t)(buffer[pos] | (buffer[pos + 1] << 8));
        pos += 2;
        if (pos + tlen > (size_t)recv_len) break;
        if (tlen > 0 && tlen < ROS_MAX_TOPIC_NAME) {
            memcpy(ctx->remote_participants[p_idx].publishers[ep_idx].topic_name,
                   buffer + pos, tlen);
            ctx->remote_participants[p_idx].publishers[ep_idx].topic_name[tlen] = '\0';
        }
        pos += tlen;
        ctx->remote_participants[p_idx].publishers[ep_idx].participant_guid = participant.guid;
        ctx->remote_participants[p_idx].publishers[ep_idx].entity_kind = DDS_ENTITY_WRITER;
        ctx->remote_participants[p_idx].publishers[ep_idx].last_seen_ms = ctx->current_time_ms;
        ctx->remote_participants[p_idx].num_publishers++;
    }

    // 解析 Readers
    for (uint16_t i = 0; i < num_readers; i++) {
        if (ctx->remote_participants[p_idx].num_subscribers >= DDS_MAX_ENDPOINTS) break;
        if (pos + DDS_GUID_LEN > (size_t)recv_len) break;
        int ep_idx = ctx->remote_participants[p_idx].num_subscribers;
        memcpy(&ctx->remote_participants[p_idx].subscribers[ep_idx].guid,
               buffer + pos, DDS_GUID_LEN);
        pos += DDS_GUID_LEN;
        if (pos + 2 > (size_t)recv_len) break;
        uint16_t tlen = (uint16_t)(buffer[pos] | (buffer[pos + 1] << 8));
        pos += 2;
        if (pos + tlen > (size_t)recv_len) break;
        if (tlen > 0 && tlen < ROS_MAX_TOPIC_NAME) {
            memcpy(ctx->remote_participants[p_idx].subscribers[ep_idx].topic_name,
                   buffer + pos, tlen);
            ctx->remote_participants[p_idx].subscribers[ep_idx].topic_name[tlen] = '\0';
        }
        pos += tlen;
        ctx->remote_participants[p_idx].subscribers[ep_idx].participant_guid = participant.guid;
        ctx->remote_participants[p_idx].subscribers[ep_idx].entity_kind = DDS_ENTITY_READER;
        ctx->remote_participants[p_idx].subscribers[ep_idx].last_seen_ms = ctx->current_time_ms;
        ctx->remote_participants[p_idx].num_subscribers++;
    }

    return 1;
}

int dds_discovery_step(DDSContext* ctx) {
    if (!ctx || !ctx->is_running) return -1;

    ctx->current_time_ms = dds_get_time_ms();

    // 周期性发送 SPDP 发现数据包
    if (ctx->current_time_ms - ctx->last_discovery_ms >= DDS_DISCOVERY_PERIOD_MS) {
        dds_send_spdp(ctx);
        ctx->last_discovery_ms = ctx->current_time_ms;
    }

    // 接收 SPDP 发现数据包
    int found = 0;
    while (1) {
        int ret = dds_recv_spdp(ctx);
        if (ret > 0) found = 1;
        else break;
    }

    // 检查参与者租约过期
    for (int i = 0; i < ctx->num_participants; i++) {
        if (ctx->current_time_ms - ctx->remote_participants[i].participant.last_seen_ms
            > DDS_LEASE_DURATION_MS) {
            ctx->remote_participants[i].participant.is_alive = 0;
        }
    }

    return found ? 1 : 0;
}

int dds_create_data_writer(DDSContext* ctx, const char* topic_name,
                           const char* type_name, int is_reliable) {
    if (!ctx || !topic_name) return -1;
    if (ctx->num_writers >= DDS_MAX_ENDPOINTS) return -1;

    int idx = ctx->num_writers;
    strncpy(ctx->writers[idx].topic_name, topic_name,
            sizeof(ctx->writers[idx].topic_name) - 1);
    if (type_name) {
        strncpy(ctx->writers[idx].type_name, type_name,
                sizeof(ctx->writers[idx].type_name) - 1);
    }
    ctx->writers[idx].is_reliable = is_reliable;
    ctx->writers[idx].is_active = 1;

    // 生成 Writer GUID
    dds_generate_guid(&ctx->writers[idx].guid, ctx->participant_serial,
                      (uint8_t)DDS_ENTITY_WRITER, (uint16_t)(idx + 1));

    ctx->num_writers++;
    return 0;
}

int dds_create_data_reader(DDSContext* ctx, const char* topic_name,
                           const char* type_name, int is_reliable,
                           DDSTopicCallback callback, void* user_data) {
    if (!ctx || !topic_name) return -1;
    if (ctx->num_readers >= DDS_MAX_ENDPOINTS) return -1;

    int idx = ctx->num_readers;
    strncpy(ctx->readers[idx].topic_name, topic_name,
            sizeof(ctx->readers[idx].topic_name) - 1);
    if (type_name) {
        strncpy(ctx->readers[idx].type_name, type_name,
                sizeof(ctx->readers[idx].type_name) - 1);
    }
    ctx->readers[idx].is_reliable = is_reliable;
    ctx->readers[idx].is_active = 1;
    ctx->readers[idx].callback = callback;
    ctx->readers[idx].user_data = user_data;

    // 生成 Reader GUID
    dds_generate_guid(&ctx->readers[idx].guid, ctx->participant_serial,
                      (uint8_t)DDS_ENTITY_READER, (uint16_t)(idx + 1));

    ctx->num_readers++;
    return 0;
}

int dds_delete_endpoint(DDSContext* ctx, const char* topic_name, int is_writer) {
    if (!ctx || !topic_name) return -1;

    if (is_writer) {
        for (int i = 0; i < ctx->num_writers; i++) {
            if (strcmp(ctx->writers[i].topic_name, topic_name) == 0) {
                ctx->writers[i].is_active = 0;
                if (i < ctx->num_writers - 1) {
                    ctx->writers[i] = ctx->writers[ctx->num_writers - 1];
                }
                ctx->num_writers--;
                return 0;
            }
        }
    } else {
        for (int i = 0; i < ctx->num_readers; i++) {
            if (strcmp(ctx->readers[i].topic_name, topic_name) == 0) {
                ctx->readers[i].is_active = 0;
                if (i < ctx->num_readers - 1) {
                    ctx->readers[i] = ctx->readers[ctx->num_readers - 1];
                }
                ctx->num_readers--;
                return 0;
            }
        }
    }
    return -1;
}

int dds_write_data(DDSContext* ctx, const char* topic_name,
                   const uint8_t* data, size_t data_size) {
    if (!ctx || !topic_name || !data || data_size == 0) return -1;
    if (!ctx->is_running || ctx->user_sock == INVALID_SOCKET_VAL) return -1;

    // 查找 Writer
    int w_idx = -1;
    for (int i = 0; i < ctx->num_writers; i++) {
        if (strcmp(ctx->writers[i].topic_name, topic_name) == 0) {
            w_idx = i;
            break;
        }
    }
    if (w_idx < 0) return -1;

    // 对所有订阅了该主题的远程参与者发送数据
    static uint32_t seq_counter = 0;
    seq_counter++;

    // 序列化 RTPS 数据包
    uint8_t rtps_buffer[ROS_TCP_BUF_SIZE];
    size_t rtps_size = 0;

    // 一次性序列化（所有reader共享一份数据）
    DDSGuid all_readers_guid;
    memset(&all_readers_guid, 0, DDS_GUID_LEN);

    if (dds_serialize_rtps_data(rtps_buffer, sizeof(rtps_buffer), &rtps_size,
                                 &ctx->writers[w_idx].guid, &all_readers_guid,
                                 seq_counter, data, data_size) != 0) {
        return -1;
    }

    // 发送给所有已知的远程参与者
    struct sockaddr_in dest;
    memset(&dest, 0, sizeof(dest));
    dest.sin_family = AF_INET;

    int sent_count = 0;
    for (int p = 0; p < ctx->num_participants; p++) {
        if (!ctx->remote_participants[p].participant.is_alive) continue;

        // 检查是否有匹配的订阅者
        int has_match = 0;
        for (int s = 0; s < ctx->remote_participants[p].num_subscribers; s++) {
            if (strcmp(ctx->remote_participants[p].subscribers[s].topic_name,
                       topic_name) == 0) {
                has_match = 1;
                break;
            }
        }
        if (!has_match) continue;

        dest.sin_addr.s_addr = inet_addr("127.0.0.1");
        dest.sin_port = htons((uint16_t)ctx->remote_participants[p].participant.user_traffic_port);

        sendto(ctx->user_sock, (const char*)rtps_buffer, (int)rtps_size, 0,
               (struct sockaddr*)&dest, sizeof(dest));
        sent_count++;
    }

    return sent_count > 0 ? 0 : -1;
}

int dds_read_data(DDSContext* ctx, const char* topic_name,
                  uint8_t* buffer, size_t buffer_size, size_t* read) {
    if (!ctx || !topic_name || !buffer || !read) return -1;
    if (!ctx->is_running || ctx->user_sock == INVALID_SOCKET_VAL) return -1;

    uint8_t recv_buf[ROS_TCP_BUF_SIZE];
    struct sockaddr_in src;
    socklen_t src_len = sizeof(src);

    int recv_len = recvfrom(ctx->user_sock, (char*)recv_buf, (int)sizeof(recv_buf),
                            0, (struct sockaddr*)&src, &src_len);
    if (recv_len <= 0) return 0;

    // 反序列化 RTPS 数据
    DDSDataSubmessage submsg;
    size_t data_read = 0;
    if (dds_deserialize_rtps_data(recv_buf, (size_t)recv_len, (size_t*)&recv_len,
                                   &submsg, buffer, buffer_size, &data_read) != 0) {
        return 0;
    }

    // 检查是否匹配本地 Reader
    for (int i = 0; i < ctx->num_readers; i++) {
        if (!ctx->readers[i].is_active) continue;
        if (strcmp(ctx->readers[i].topic_name, topic_name) != 0) continue;

        *read = data_read;

        // 调用回调
        if (ctx->readers[i].callback) {
            ctx->readers[i].callback(topic_name, buffer, data_read,
                                     &submsg.writer_id, ctx->readers[i].user_data);
        }
        return 1;
    }

    *read = data_read;
    return 1;
}

int dds_get_participants(DDSContext* ctx,
                         DDSDomainParticipant* participants, int max_count) {
    if (!ctx || !participants || max_count <= 0) return 0;

    int count = 0;
    for (int i = 0; i < ctx->num_participants && count < max_count; i++) {
        if (ctx->remote_participants[i].participant.is_alive) {
            participants[count++] = ctx->remote_participants[i].participant;
        }
    }
    return count;
}

int dds_get_endpoints(DDSContext* ctx, DDSEndpoint* endpoints, int max_count) {
    if (!ctx || !endpoints || max_count <= 0) return 0;

    int count = 0;
    for (int p = 0; p < ctx->num_participants && count < max_count; p++) {
        if (!ctx->remote_participants[p].participant.is_alive) continue;
        for (int w = 0; w < ctx->remote_participants[p].num_publishers && count < max_count; w++) {
            endpoints[count++] = ctx->remote_participants[p].publishers[w];
        }
        for (int s = 0; s < ctx->remote_participants[p].num_subscribers && count < max_count; s++) {
            endpoints[count++] = ctx->remote_participants[p].subscribers[s];
        }
    }
    return count;
}
