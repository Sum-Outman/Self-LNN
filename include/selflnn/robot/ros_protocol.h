#ifndef SELFLNN_ROS_PROTOCOL_H
#define SELFLNN_ROS_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ROS_MAX_TOPIC_NAME      256
#define ROS_MAX_NODE_NAME       128
#define ROS_MAX_TYPE_NAME       128
#define ROS_MAX_MD5             64
#define ROS_MAX_URI             512
#define ROS_MAX_HEADER_FIELDS   32
#define ROS_MAX_CONNECTIONS     64
#define ROS_MAX_CALLBACKS       64
#define ROS_MAX_MESSAGE_SIZE    (16 * 1024 * 1024)
#define ROS_XMLRPC_BUF_SIZE     (64 * 1024)
#define ROS_TCP_BUF_SIZE        (16 * 1024 * 1024)
#define ROS_DEFAULT_TIMEOUT_MS  5000
#define ROS_HEARTBEAT_INTERVAL_MS 30000

#define ROS_MSG_FLAG_NONE       0x0000
#define ROS_MSG_FLAG_LATCHED    0x0001
#define ROS_MSG_FLAG_TCP_NODELAY 0x0002

typedef enum {
    ROS_OK = 0,
    ROS_ERROR_GENERAL = -1,
    ROS_ERROR_TIMEOUT = -2,
    ROS_ERROR_CONNECTION = -3,
    ROS_ERROR_NOT_CONNECTED = -4,
    ROS_ERROR_ALREADY_EXISTS = -5,
    ROS_ERROR_NOT_FOUND = -6,
    ROS_ERROR_INVALID_PARAM = -7,
    ROS_ERROR_NO_MEMORY = -8,
    ROS_ERROR_SERIALIZE = -9,
    ROS_ERROR_DESERIALIZE = -10,
    ROS_ERROR_PROTOCOL = -11,
    ROS_ERROR_MASTER = -12,
    ROS_ERROR_SERVICE = -13,
    ROS_ERROR_NETWORK = -14
} RosErrorCode;

typedef enum {
    ROS_NODE_STATE_CREATED = 0,
    ROS_NODE_STATE_CONNECTING = 1,
    ROS_NODE_STATE_CONNECTED = 2,
    ROS_NODE_STATE_DISCONNECTED = 3,
    ROS_NODE_STATE_ERROR = 4
} RosNodeState;

typedef enum {
    ROS_TRANSPORT_TCP = 0,
    ROS_TRANSPORT_UDP = 1
} RosTransportType;

typedef struct {
    char key[128];
    char value[512];
} RosHeaderField;

typedef struct {
    char topic_name[ROS_MAX_TOPIC_NAME];
    char type_name[ROS_MAX_TYPE_NAME];
    char md5sum[ROS_MAX_MD5];
    int message_definition_size;
    char* message_definition;
    RosTransportType transport;
    int flags;
    int num_header_fields;
    RosHeaderField header_fields[ROS_MAX_HEADER_FIELDS];
} RosConnectionHeader;

typedef struct {
    double x, y, z;
} RosVector3;

typedef struct {
    double x, y, z, w;
} RosQuaternion;

typedef struct {
    RosVector3 position;
    RosQuaternion orientation;
} RosPose;

typedef struct {
    RosVector3 linear;
    RosVector3 angular;
} RosTwist;

typedef struct {
    RosPose pose;
    RosTwist twist;
} RosOdometry;

typedef struct {
    uint32_t seq;
    double timestamp_sec;
    double timestamp_nsec;
    char frame_id[ROS_MAX_TOPIC_NAME];
} RosHeader;

typedef struct {
    RosHeader header;
    double axes[8];
    int32_t buttons[16];
} RosJoy;

typedef struct {
    RosHeader header;
    float angle_min;
    float angle_max;
    float angle_increment;
    float time_increment;
    float scan_time;
    float range_min;
    float range_max;
    float ranges[1024];
    int ranges_size;
    float intensities[1024];
    int intensities_size;
} RosLaserScan;

typedef struct {
    uint32_t width;
    uint32_t height;
    char encoding[32];
    uint8_t is_bigendian;
    uint32_t step;
    size_t data_size;
    uint8_t* data;
} RosImage;

typedef struct {
    RosHeader header;
    RosQuaternion orientation;
    double orientation_covariance[9];
    RosVector3 angular_velocity;
    double angular_velocity_covariance[9];
    RosVector3 linear_acceleration;
    double linear_acceleration_covariance[9];
} RosImu;

typedef struct {
    RosHeader header;
    char name[32][64];
    int names_count;
    double position[32];
    int positions_count;
    double velocity[32];
    int velocities_count;
    double effort[32];
    int efforts_count;
} RosJointState;

typedef struct {
    RosHeader header;
    RosPose pose;
    RosTwist twist;
} RosPoseStamped;

typedef struct {
    RosHeader header;
    RosVector3 linear_velocity;
    RosVector3 angular_velocity;
    RosPose pose;
    RosTwist twist;
    double pose_covariance[36];
    double twist_covariance[36];
} RosOdometryFull;

typedef struct {
    RosHeader header;
    float points[3][1024];
    int points_count;
} RosPointCloud;

typedef struct {
    RosHeader header;
    double temperature;
    double variance;
} RosTemperature;

typedef struct {
    RosHeader header;
    float force[3];
    float torque[3];
} RosWrenchStamped;

typedef struct {
    char name[64];
    float position;
    float velocity;
    float effort;
} RosJointStateItem;

typedef struct {
    unsigned char frame_id;
    unsigned char camera_id;
    int width;
    int height;
    unsigned short depth_data[640 * 480];
    int depth_size;
} RosDepthImage;

typedef struct {
    RosHeader header;
    double data[64];
    int data_size;
    char layout[128];
} RosFloat64MultiArray;

typedef struct {
    int32_t level;
    char name[64];
    char msg[256];
    char file[128];
    int32_t line;
    char function[64];
} RosLog;

typedef struct {
    char interface_name[64];
    float linear_x_min;
    float linear_x_max;
    float linear_y_min;
    float linear_y_max;
    float angular_z_min;
    float angular_z_max;
} RosTwistCmdVelLimits;

typedef struct {
    float kp, ki, kd;
    float i_clamp_min, i_clamp_max;
} RosPidConfig;

#define ROS_CMD_VEL_TOPIC       "/cmd_vel"
#define ROS_ODOM_TOPIC          "/odom"
#define ROS_JOINT_STATES_TOPIC  "/joint_states"
#define ROS_IMU_TOPIC           "/imu/data"
#define ROS_SCAN_TOPIC          "/scan"
#define ROS_CAMERA_TOPIC        "/camera/image_raw"
#define ROS_TF_TOPIC            "/tf"

int ros_serialize_header(const RosHeader* header, uint8_t* buffer, size_t buffer_size, size_t* written);
int ros_deserialize_header(const uint8_t* buffer, size_t buffer_size, size_t* read, RosHeader* header);

int ros_serialize_pose(const RosPose* pose, uint8_t* buffer, size_t buffer_size, size_t* written);
int ros_deserialize_pose(const uint8_t* buffer, size_t buffer_size, size_t* read, RosPose* pose);

int ros_serialize_twist(const RosTwist* twist, uint8_t* buffer, size_t buffer_size, size_t* written);
int ros_deserialize_twist(const uint8_t* buffer, size_t buffer_size, size_t* read, RosTwist* twist);

int ros_serialize_odometry(const RosOdometry* odom, uint8_t* buffer, size_t buffer_size, size_t* written);
int ros_deserialize_odometry(const uint8_t* buffer, size_t buffer_size, size_t* read, RosOdometry* odom);

int ros_serialize_laserscan(const RosLaserScan* scan, uint8_t* buffer, size_t buffer_size, size_t* written);
int ros_deserialize_laserscan(const uint8_t* buffer, size_t buffer_size, size_t* read, RosLaserScan* scan);

int ros_serialize_joy(const RosJoy* joy, uint8_t* buffer, size_t buffer_size, size_t* written);
int ros_deserialize_joy(const uint8_t* buffer, size_t buffer_size, size_t* read, RosJoy* joy);

int ros_serialize_imu(const RosImu* imu, uint8_t* buffer, size_t buffer_size, size_t* written);
int ros_deserialize_imu(const uint8_t* buffer, size_t buffer_size, size_t* read, RosImu* imu);

int ros_serialize_image(const RosImage* image, uint8_t* buffer, size_t buffer_size, size_t* written);
int ros_deserialize_image(const uint8_t* buffer, size_t buffer_size, size_t* read, RosImage* image);

int ros_serialize_joint_state(const RosJointState* js, uint8_t* buffer, size_t buffer_size, size_t* written);
int ros_deserialize_joint_state(const uint8_t* buffer, size_t buffer_size, size_t* read, RosJointState* js);

int ros_serialize_float64_multi_array(const RosFloat64MultiArray* arr, uint8_t* buffer, size_t buffer_size, size_t* written);
int ros_deserialize_float64_multi_array(const uint8_t* buffer, size_t buffer_size, size_t* read, RosFloat64MultiArray* arr);

int ros_serialize_pointcloud(const RosPointCloud* pc, uint8_t* buffer, size_t buffer_size, size_t* written);
int ros_deserialize_pointcloud(const uint8_t* buffer, size_t buffer_size, size_t* read, RosPointCloud* pc);

int ros_serialize_log(const RosLog* log, uint8_t* buffer, size_t buffer_size, size_t* written);
int ros_deserialize_log(const uint8_t* buffer, size_t buffer_size, size_t* read, RosLog* log);

const char* ros_error_string(int error_code);
int ros_build_connection_header(RosConnectionHeader* header, const char* topic,
                                 const char* type, const char* md5sum, int is_tcp);

typedef int (*RosSerializeFunc)(const void* msg, uint8_t* buffer, size_t buffer_size, size_t* written);
typedef int (*RosDeserializeFunc)(const uint8_t* buffer, size_t buffer_size, size_t* read, void* msg);
typedef void (*RosMessageCallback)(const char* topic, const void* msg, size_t msg_size, void* user_data);

typedef struct {
    char topic_name[ROS_MAX_TOPIC_NAME];
    char type_name[ROS_MAX_TYPE_NAME];
    int is_service;
    int is_subscriber;
    RosSerializeFunc serialize;
    RosDeserializeFunc deserialize;
    RosMessageCallback callback;
    void* user_data;
} RosTopicCallbackEntry;

const char* ros_topic_type_to_string(int type_id);
int ros_get_md5_from_type(const char* type_name, char* md5_out, size_t md5_size);

// ============================================================================
// ROS TCP传输层 (真实TCP+XMLRPC Master通信)
// ============================================================================
int ros_tcp_connect(const char* host, int port, int timeout_ms);
int ros_tcp_disconnect(void);
int ros_master_register_publisher(const char* caller_id, const char* topic,
                                   const char* topic_type, const char* caller_api);
int ros_tcp_negotiate_subscriber(const char* topic, const char* topic_type,
                                  const char* caller_api, RosConnectionHeader* header_out);
int ros_tcp_publish_message(const uint8_t* serialized_data, size_t data_size);
int ros_tcp_subscribe_loop(int (*callback)(const uint8_t*, size_t, void*), void* user_data);

// ============================================================================
// ROS2 DDS 协议支持
// ============================================================================
// DDS 使用 RTPS (Real-Time Publish-Subscribe) 协议
// 通过 UDP 多播进行自动发现，无需中心化的 ROS Master

#define DDS_DOMAIN_ID_DEFAULT       0
#define DDS_DISCOVERY_PORT_BASE     7400
#define DDS_USER_TRAFFIC_PORT_BASE  7410
#define DDS_MULTICAST_ADDRESS       "239.255.0.1"
#define DDS_MULTICAST_PORT          7400
#define DDS_MAX_PARTICIPANTS        32
#define DDS_MAX_ENDPOINTS           128
#define DDS_DISCOVERY_PERIOD_MS     2500
#define DDS_LEASE_DURATION_MS       10000
#define DDS_GUID_PREFIX_LEN         12
#define DDS_ENTITY_ID_LEN           4
#define DDS_GUID_LEN                16
#define DDS_PROTOCOL_VERSION_MAJOR  2
#define DDS_PROTOCOL_VERSION_MINOR  3
#define DDS_VENDOR_ID               0x0103

// DDS 实体类型枚举
typedef enum {
    DDS_ENTITY_PARTICIPANT = 0,
    DDS_ENTITY_WRITER = 1,
    DDS_ENTITY_READER = 2,
    DDS_ENTITY_TOPIC = 3,
    DDS_ENTITY_PUBLISHER = 4,
    DDS_ENTITY_SUBSCRIBER = 5
} DDSEntityKind;

// DDS GUID
typedef struct {
    uint8_t prefix[DDS_GUID_PREFIX_LEN];  // 全局唯一前缀
    uint8_t entity_id[DDS_ENTITY_ID_LEN]; // 实体ID
} DDSGuid;

// DDS 参与者信息（SPDP 发现数据）
typedef struct {
    DDSGuid guid;                           // 参与者 GUID
    char node_name[ROS_MAX_NODE_NAME];      // 节点名称
    char node_namespace[ROS_MAX_TOPIC_NAME];// 命名空间
    uint32_t domain_id;                     // 域ID
    uint32_t user_traffic_port;             // 用户流量端口
    uint32_t discovery_port;                // 发现端口
    uint32_t remote_ip;                     /* ZSFZS-F019: 远程参与者IP地址（网络字节序，从SPDP发现获取） */
    uint8_t protocol_version_major;         // RTPS 主版本
    uint8_t protocol_version_minor;         // RTPS 次版本
    uint16_t vendor_id;                     // 厂商ID
    int64_t last_seen_ms;                   // 最后可见时间
    int is_alive;                           // 是否存活
} DDSDomainParticipant;

// DDS 端点（Writer/Reader）信息（SEDP 发现数据）
typedef struct {
    DDSGuid guid;                           // 端点 GUID
    DDSGuid participant_guid;               // 所属参与者的 GUID
    char topic_name[ROS_MAX_TOPIC_NAME];    // 主题名称
    char type_name[ROS_MAX_TYPE_NAME];      // 类型名称
    uint8_t entity_kind;                    // Writer(1)/Reader(2)
    int is_reliable;                        // 是否可靠
    int is_transient_local;                 // 是否持久本地
    int64_t last_seen_ms;                   // 最后可见时间
} DDSEndpoint;

// DDS RTPS 子消息头
typedef struct {
    uint8_t submessage_id;                  // 子消息ID
    uint8_t flags;                          // 标记位
    uint16_t length;                        // 子消息长度
} DDSRTPSSubmessageHeader;

// DDS RTPS 数据子消息
typedef struct {
    DDSRTPSSubmessageHeader header;
    uint16_t extra_flags;
    uint16_t octets_to_inline_qos;
    DDSGuid reader_id;
    DDSGuid writer_id;
    uint32_t sequence_number;
} DDSDataSubmessage;

// DDS 参与者代理信息
typedef struct {
    DDSDomainParticipant participant;
    int num_publishers;
    int num_subscribers;
    DDSEndpoint publishers[DDS_MAX_ENDPOINTS];
    DDSEndpoint subscribers[DDS_MAX_ENDPOINTS];
} DDSParticipantData;

// DDS 主题数据回调
typedef void (*DDSTopicCallback)(const char* topic, const uint8_t* data,
                                 size_t data_size, const DDSGuid* writer_guid,
                                 void* user_data);

// DDS 参与者变化回调
typedef void (*DDSParticipantCallback)(const DDSDomainParticipant* participant,
                                       int is_join, void* user_data);

// DDS 上下文（顶层句柄）
typedef struct DDSContext DDSContext;

// --- ROS2 DDS API 函数 ---

// 创建/销毁 DDS 上下文
DDSContext* dds_context_create(uint32_t domain_id, const char* node_name,
                               const char* node_namespace);
void dds_context_free(DDSContext* ctx);

// 启动/停止 DDS 发现
int dds_discovery_start(DDSContext* ctx);
int dds_discovery_stop(DDSContext* ctx);
int dds_discovery_step(DDSContext* ctx);

// 创建/删除主题端点
int dds_create_data_writer(DDSContext* ctx, const char* topic_name,
                           const char* type_name, int is_reliable);
int dds_create_data_reader(DDSContext* ctx, const char* topic_name,
                           const char* type_name, int is_reliable,
                           DDSTopicCallback callback, void* user_data);
int dds_delete_endpoint(DDSContext* ctx, const char* topic_name, int is_writer);

// 发布/订阅数据
int dds_write_data(DDSContext* ctx, const char* topic_name,
                   const uint8_t* data, size_t data_size);
int dds_read_data(DDSContext* ctx, const char* topic_name,
                  uint8_t* buffer, size_t buffer_size, size_t* read);

// 获取远程参与者列表
int dds_get_participants(DDSContext* ctx,
                         DDSDomainParticipant* participants, int max_count);
int dds_get_endpoints(DDSContext* ctx, DDSEndpoint* endpoints, int max_count);

// RTPS 数据包序列化/反序列化
int dds_serialize_rtps_data(uint8_t* buffer, size_t buffer_size, size_t* written,
                            const DDSGuid* writer_id, const DDSGuid* reader_id,
                            uint32_t sequence_number, const uint8_t* data,
                            size_t data_size);
int dds_deserialize_rtps_data(const uint8_t* buffer, size_t buffer_size,
                              size_t* read, DDSDataSubmessage* submsg,
                              uint8_t* data_out, size_t data_out_size,
                              size_t* data_read);

// 生成唯一 GUID
void dds_generate_guid(DDSGuid* guid, uint32_t participant_id,
                       uint8_t entity_kind, uint16_t entity_id);

#ifdef __cplusplus
}
#endif

#endif
