#include "selflnn/robot/ros_gazebo_bridge.h"
#include "selflnn/robot/simulator.h"
#include "selflnn/core/port_config.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#define SOCKET_FD_TYPE SOCKET
#define INVALID_SOCKET_VAL INVALID_SOCKET
#define socket_close_pb closesocket
typedef int socklen_t;
#else
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <fcntl.h>
#include <errno.h>
#define SOCKET_FD_TYPE int
#define INVALID_SOCKET_VAL (-1)
#define socket_close_pb close
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ROS_GAZEBO_BRIDGE_PUBLISH_INTERVAL_DEFAULT 0.05f
#define ROS_GAZEBO_BRIDGE_SPIN_INTERVAL_MS 10
#define ROS_GAZEBO_BRIDGE_MAX_ERROR 256

#define ROS_GAZEBO_SERVICE_SPAWN_MODEL    "/gazebo/spawn_model"
#define ROS_GAZEBO_SERVICE_DELETE_MODEL   "/gazebo/delete_model"
#define ROS_GAZEBO_SERVICE_GET_STATE      "/gazebo/get_model_state"
#define ROS_GAZEBO_SERVICE_SET_STATE      "/gazebo/set_model_state"
#define ROS_GAZEBO_SERVICE_APPLY_WRENCH   "/gazebo/apply_body_wrench"
#define ROS_GAZEBO_SERVICE_PAUSE_PHYSICS  "/gazebo/pause_physics"
#define ROS_GAZEBO_SERVICE_UNPAUSE_PHYSICS "/gazebo/unpause_physics"
#define ROS_GAZEBO_SERVICE_RESET_WORLD    "/gazebo/reset_world"
#define ROS_GAZEBO_SERVICE_RESET_SIMULATION "/gazebo/reset_simulation"
#define ROS_GAZEBO_SERVICE_GET_WORLD      "/gazebo/get_world_properties"
#define ROS_GAZEBO_TOPIC_LINKS            "/gazebo/link_states"
#define ROS_GAZEBO_TOPIC_MODEL_STATES     "/gazebo/model_states"
#define ROS_GAZEBO_TOPIC_PARAM           "/gazebo/parameter_descriptions"
#define ROS_GAZEBO_SERVICE_SET_PARAM      "/gazebo/set_parameters"

typedef struct {
    int robot_id;
    Simulator* sim;
    RosNode* node;
    RosGazeboBridgeConfig config;
    RosGazeboBridgeState state;

    int advertise_joint_states;
    int advertise_odom;
    int advertise_scan;
    int advertise_imu;
    int advertise_camera;
    int advertise_tf;
    int advertise_link_states;
    int advertise_model_states;

    int sub_cmd_vel;
    int sub_joint_command;

    int srv_spawn;
    int srv_delete;
    int srv_get_state;
    int srv_set_state;
    int srv_apply_wrench;
    int srv_pause;
    int srv_unpause;
    int srv_reset_world;
    int srv_reset_sim;

    char latest_cmd_vel_data[ROS_TCP_BUF_SIZE];
    size_t latest_cmd_vel_size;
    volatile int has_cmd_vel;

    double last_publish_times[ROS_GAZEBO_BRIDGE_MAX_TOPICS];
    double current_sim_time;

    int running;
    int spin_thread_created;
    int bridge_spin_running;

    char last_error[ROS_GAZEBO_BRIDGE_MAX_ERROR];
    int error_code;

    int robot_count;
    RosGazeboRobotInfo robot_infos[ROS_GAZEBO_BRIDGE_MAX_ROBOTS];
    char model_names[ROS_GAZEBO_BRIDGE_MAX_ROBOTS][ROS_GAZEBO_BRIDGE_FRAME_ID_MAX];
    RosGazeboWorldState world_state;
} RosGazeboBridgeInternal;

static double bridge_get_time(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static void bridge_set_error(RosGazeboBridgeInternal* br, int code, const char* msg) {
    if (!br) return;
    strncpy(br->last_error, msg, sizeof(br->last_error) - 1);
    br->last_error[sizeof(br->last_error) - 1] = '\0';
    br->error_code = code;
}

static void cmd_vel_callback(const char* topic, const void* msg, size_t msg_size, void* user_data) {
    (void)topic;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)user_data;
    if (!br || !msg || msg_size == 0) return;
    size_t copy_size = msg_size < sizeof(br->latest_cmd_vel_data) ? msg_size : sizeof(br->latest_cmd_vel_data);
    memcpy(br->latest_cmd_vel_data, msg, copy_size);
    br->latest_cmd_vel_size = copy_size;
    br->has_cmd_vel = 1;
}

static int spawn_model_service_callback(const void* request, size_t req_size,
                                         void* response, size_t* resp_size, void* user_data) {
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)user_data;
    (void)req_size;
    const char* model_name = (const char*)request;
    float pos[3] = {0,0,0}, ori[4] = {0,0,0,1};
    if (ros_gazebo_bridge_spawn_model((RosGazeboBridge*)br, model_name, pos, ori) == 0) {
        const char* ok = "true";
        size_t ok_len = strlen(ok) + 1;
        if (*resp_size >= ok_len) {
            memcpy(response, ok, ok_len);
            *resp_size = ok_len;
        }
    } else {
        const char* fail = "false";
        size_t fail_len = strlen(fail) + 1;
        if (*resp_size >= fail_len) {
            memcpy(response, fail, fail_len);
            *resp_size = fail_len;
        }
    }
    return 0;
}

static int delete_model_service_callback(const void* request, size_t req_size,
                                          void* response, size_t* resp_size, void* user_data) {
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)user_data;
    (void)req_size;
    const char* model_name = (const char*)request;
    if (ros_gazebo_bridge_delete_model((RosGazeboBridge*)br, model_name) == 0) {
        const char* ok = "true";
        size_t ok_len = strlen(ok) + 1;
        if (*resp_size >= ok_len) {
            memcpy(response, ok, ok_len);
            *resp_size = ok_len;
        }
    } else {
        const char* fail = "false";
        size_t fail_len = strlen(fail) + 1;
        if (*resp_size >= fail_len) {
            memcpy(response, fail, fail_len);
            *resp_size = fail_len;
        }
    }
    return 0;
}

static int pause_physics_service_callback(const void* request, size_t req_size,
                                           void* response, size_t* resp_size, void* user_data) {
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)user_data;
    (void)request; (void)req_size;
    if (ros_gazebo_bridge_pause((RosGazeboBridge*)br) == 0) {
        const char* ok = "true";
        size_t ok_len = strlen(ok) + 1;
        if (*resp_size >= ok_len) {
            memcpy(response, ok, ok_len);
            *resp_size = ok_len;
        }
    } else {
        const char* fail = "false";
        size_t fail_len = strlen(fail) + 1;
        if (*resp_size >= fail_len) {
            memcpy(response, fail, fail_len);
            *resp_size = fail_len;
        }
    }
    return 0;
}

static int unpause_physics_service_callback(const void* request, size_t req_size,
                                             void* response, size_t* resp_size, void* user_data) {
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)user_data;
    (void)request; (void)req_size;
    if (ros_gazebo_bridge_resume((RosGazeboBridge*)br) == 0) {
        const char* ok = "true";
        size_t ok_len = strlen(ok) + 1;
        if (*resp_size >= ok_len) {
            memcpy(response, ok, ok_len);
            *resp_size = ok_len;
        }
    } else {
        const char* fail = "false";
        size_t fail_len = strlen(fail) + 1;
        if (*resp_size >= fail_len) {
            memcpy(response, fail, fail_len);
            *resp_size = fail_len;
        }
    }
    return 0;
}

static int reset_world_service_callback(const void* request, size_t req_size,
                                         void* response, size_t* resp_size, void* user_data) {
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)user_data;
    (void)request; (void)req_size;
    if (ros_gazebo_bridge_reset((RosGazeboBridge*)br) == 0) {
        const char* ok = "true";
        size_t ok_len = strlen(ok) + 1;
        if (*resp_size >= ok_len) {
            memcpy(response, ok, ok_len);
            *resp_size = ok_len;
        }
    } else {
        const char* fail = "false";
        size_t fail_len = strlen(fail) + 1;
        if (*resp_size >= fail_len) {
            memcpy(response, fail, fail_len);
            *resp_size = fail_len;
        }
    }
    return 0;
}

static void bridge_spin_loop(void* arg) {
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)arg;
    if (!br) return;

    br->bridge_spin_running = 1;
    while (br->running) {
        ros_node_spin(br->node, ROS_GAZEBO_BRIDGE_SPIN_INTERVAL_MS);

        double now = bridge_get_time();
        double pub_interval = (br->config.publish_rate > 0.001f) ? (1.0 / br->config.publish_rate) : ROS_GAZEBO_BRIDGE_PUBLISH_INTERVAL_DEFAULT;

        if (br->has_cmd_vel && br->robot_count > 0) {
            RosTwist cmd;
            if (ros_deserialize_twist((const uint8_t*)br->latest_cmd_vel_data, br->latest_cmd_vel_size, &(size_t){0}, &cmd) == ROS_OK) {
                RosTwist ros_cmd;
                ros_cmd.linear.x = cmd.linear.x;
                ros_cmd.linear.y = cmd.linear.y;
                ros_cmd.linear.z = cmd.linear.z;
                ros_cmd.angular.x = cmd.angular.x;
                ros_cmd.angular.y = cmd.angular.y;
                ros_cmd.angular.z = cmd.angular.z;

                RobotCommand robot_cmd;
                memset(&robot_cmd, 0, sizeof(robot_cmd));
                robot_cmd.mode = MOTION_MODE_VELOCITY;
                robot_cmd.target_linear_velocity[0] = (float)ros_cmd.linear.x;
                robot_cmd.target_linear_velocity[1] = (float)ros_cmd.linear.y;
                robot_cmd.target_linear_velocity[2] = (float)ros_cmd.linear.z;
                robot_cmd.target_angular_velocity[0] = (float)ros_cmd.angular.x;
                robot_cmd.target_angular_velocity[1] = (float)ros_cmd.angular.y;
                robot_cmd.target_angular_velocity[2] = (float)ros_cmd.angular.z;
                simulator_apply_robot_command(br->sim, 0, &robot_cmd);
            }
        }

        for (int i = 0; i < br->robot_count; i++) {
            if (br->config.enable_joint_state_pub && (now - br->last_publish_times[i * 4 + 0]) >= pub_interval) {
                ros_gazebo_bridge_publish_joint_states((RosGazeboBridge*)br, i);
                br->last_publish_times[i * 4 + 0] = now;
            }
            if (br->config.enable_odom_pub && (now - br->last_publish_times[i * 4 + 1]) >= pub_interval) {
                ros_gazebo_bridge_publish_odometry((RosGazeboBridge*)br, i);
                br->last_publish_times[i * 4 + 1] = now;
            }
        }

        if (br->config.enable_scan_pub && br->robot_count > 0) {
            if ((now - br->last_publish_times[1]) >= pub_interval) {
                ros_gazebo_bridge_publish_laserscan((RosGazeboBridge*)br, 0);
                br->last_publish_times[1] = now;
            }
        }
        if (br->config.enable_imu_pub && br->robot_count > 0) {
            if ((now - br->last_publish_times[2]) >= pub_interval) {
                ros_gazebo_bridge_publish_imu((RosGazeboBridge*)br, 0);
                br->last_publish_times[2] = now;
            }
        }

#ifdef _WIN32
        Sleep(1);
#else
        usleep(1000);
#endif
    }
    br->bridge_spin_running = 0;
}

RosGazeboBridgeConfig ros_gazebo_bridge_config_default(const char* node_name) {
    RosGazeboBridgeConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    strncpy(cfg.ros_master_host, "127.0.0.1", sizeof(cfg.ros_master_host) - 1);
    cfg.ros_master_port = SELFLNN_ROS_MASTER_PORT;
    strncpy(cfg.node_name, node_name ? node_name : "selflnn_gazebo_bridge", sizeof(cfg.node_name) - 1);
    strncpy(cfg.node_host, "127.0.0.1", sizeof(cfg.node_host) - 1);
    cfg.node_xmlrpc_port = SELFLNN_ROS_XMLRPC_PORT;
    cfg.node_tcp_port = SELFLNN_ROS_MASTER_PORT + 1;
    strncpy(cfg.world_name, "world", sizeof(cfg.world_name) - 1);
    cfg.publish_rate = 20.0f;
    cfg.enable_joint_state_pub = 1;
    cfg.enable_odom_pub = 1;
    cfg.enable_scan_pub = 1;
    cfg.enable_imu_pub = 1;
    cfg.enable_camera_pub = 0;
    cfg.enable_tf_pub = 1;
    cfg.enable_sim_control_srv = 1;
    cfg.use_external_gazebo = 0;
    cfg.internal = 1;
    return cfg;
}

RosGazeboBridge* ros_gazebo_bridge_create(const RosGazeboBridgeConfig* config) {
    if (!config) return NULL;

    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)safe_malloc(sizeof(RosGazeboBridgeInternal));
    if (!br) return NULL;
    memset(br, 0, sizeof(RosGazeboBridgeInternal));
    memcpy(&br->config, config, sizeof(RosGazeboBridgeConfig));

    br->state = ROS_GAZEBO_BRIDGE_STATE_DISCONNECTED;
    br->robot_count = 0;
    br->running = 0;
    br->bridge_spin_running = 0;
    br->has_cmd_vel = 0;
    br->latest_cmd_vel_size = 0;
    br->current_sim_time = 0.0;
    br->error_code = 0;
    br->last_error[0] = '\0';

    for (int i = 0; i < ROS_GAZEBO_BRIDGE_MAX_TOPICS; i++) {
        br->last_publish_times[i] = 0.0;
    }

    SimulatorConfig sim_cfg;
    memset(&sim_cfg, 0, sizeof(sim_cfg));
    sim_cfg.type = SIMULATOR_GAZEBO;
    strncpy(sim_cfg.name, config->world_name, sizeof(sim_cfg.name) - 1);
    strncpy(sim_cfg.hostname, "127.0.0.1", sizeof(sim_cfg.hostname) - 1);
    sim_cfg.port = SELFLNN_GAZEBO_PORT;
    sim_cfg.timeout_ms = 5000;
    sim_cfg.retry_count = 3;
    sim_cfg.retry_delay_ms = 1000;
    sim_cfg.timestep = 0.001f;
    sim_cfg.gravity = 9.81f;
    sim_cfg.enable_visualization = 0;
    sim_cfg.enable_sensors = 1;
    sim_cfg.sensor_update_rate = 100;

    br->sim = simulator_create(&sim_cfg);
    if (!br->sim) {
        bridge_set_error(br, -1, "创建Gazebo仿真器实例失败");
        safe_free((void**)&br);
        return NULL;
    }

    bool ros_ok = false;

    RosNodeConfig node_cfg;
    memset(&node_cfg, 0, sizeof(node_cfg));
    strncpy(node_cfg.master_host, config->ros_master_host, sizeof(node_cfg.master_host) - 1);
    node_cfg.master_port = config->ros_master_port;
    strncpy(node_cfg.node_name, config->node_name, sizeof(node_cfg.node_name) - 1);
    strncpy(node_cfg.node_host, config->node_host, sizeof(node_cfg.node_host) - 1);
    node_cfg.xmlrpc_port = config->node_xmlrpc_port;
    node_cfg.tcp_port = config->node_tcp_port;
    node_cfg.tcp_timeout_ms = 5000;
    node_cfg.heartbeat_interval_ms = ROS_HEARTBEAT_INTERVAL_MS;
    node_cfg.enable_auto_reconnect = 1;
    node_cfg.max_reconnect_attempts = 3;

    br->node = ros_node_create(&node_cfg);
    if (br->node) {
        char master_uri[ROS_MAX_URI];
        snprintf(master_uri, sizeof(master_uri), "http://%s:%d", config->ros_master_host, config->ros_master_port);
        if (ros_node_connect_to_master(br->node, master_uri) == 0) {
            ros_ok = true;
        } else {
            bridge_set_error(br, -1, "连接ROS Master失败，桥接将以离线模式运行");
        }
    } else {
        bridge_set_error(br, -1, "创建ROS节点失败，桥接将以离线模式运行");
    }

    if (ros_ok) {
        br->advertise_joint_states = ros_node_advertise(br->node, ROS_JOINT_STATES_TOPIC, "sensor_msgs/JointState", "3066dcd76a5cfaef2a0c6d4b5f5b5b5b");
        br->advertise_odom = ros_node_advertise(br->node, ROS_ODOM_TOPIC, "nav_msgs/Odometry", "3066dcd76a5cfaef2a0c6d4b5f5b5b5b");
        br->advertise_scan = ros_node_advertise(br->node, ROS_SCAN_TOPIC, "sensor_msgs/LaserScan", "3066dcd76a5cfaef2a0c6d4b5f5b5b5b");
        br->advertise_imu = ros_node_advertise(br->node, ROS_IMU_TOPIC, "sensor_msgs/Imu", "3066dcd76a5cfaef2a0c6d4b5f5b5b5b");
        br->advertise_tf = ros_node_advertise(br->node, ROS_TF_TOPIC, "tf2_msgs/TFMessage", "3066dcd76a5cfaef2a0c6d4b5f5b5b5b");
        br->advertise_link_states = ros_node_advertise(br->node, ROS_GAZEBO_TOPIC_LINKS, "gazebo_msgs/LinkStates", "3066dcd76a5cfaef2a0c6d4b5f5b5b5b");
        br->advertise_model_states = ros_node_advertise(br->node, ROS_GAZEBO_TOPIC_MODEL_STATES, "gazebo_msgs/ModelStates", "3066dcd76a5cfaef2a0c6d4b5f5b5b5b");

        br->sub_cmd_vel = ros_node_subscribe(br->node, ROS_CMD_VEL_TOPIC, "geometry_msgs/Twist", cmd_vel_callback, br);

        if (config->enable_sim_control_srv) {
            br->srv_spawn = ros_node_advertise_service(br->node, ROS_GAZEBO_SERVICE_SPAWN_MODEL, spawn_model_service_callback, br);
            br->srv_delete = ros_node_advertise_service(br->node, ROS_GAZEBO_SERVICE_DELETE_MODEL, delete_model_service_callback, br);
            br->srv_pause = ros_node_advertise_service(br->node, ROS_GAZEBO_SERVICE_PAUSE_PHYSICS, pause_physics_service_callback, br);
            br->srv_unpause = ros_node_advertise_service(br->node, ROS_GAZEBO_SERVICE_UNPAUSE_PHYSICS, unpause_physics_service_callback, br);
            br->srv_reset_world = ros_node_advertise_service(br->node, ROS_GAZEBO_SERVICE_RESET_WORLD, reset_world_service_callback, br);
            br->srv_reset_sim = ros_node_advertise_service(br->node, ROS_GAZEBO_SERVICE_RESET_SIMULATION, reset_world_service_callback, br);
        }
    }

    br->state = ROS_GAZEBO_BRIDGE_STATE_CONNECTED;
    return (RosGazeboBridge*)br;
}

void ros_gazebo_bridge_destroy(RosGazeboBridge* bridge) {
    if (!bridge) return;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;

    br->running = 0;
    int wait_count = 0;
    while (br->bridge_spin_running && wait_count < 50) {
#ifdef _WIN32
        Sleep(10);
#else
        usleep(10000);
#endif
        wait_count++;
    }

    if (br->node) {
        ros_node_destroy(br->node);
        br->node = NULL;
    }
    if (br->sim) {
        simulator_destroy(br->sim);
        br->sim = NULL;
    }

    br->state = ROS_GAZEBO_BRIDGE_STATE_DISCONNECTED;
    safe_free((void**)&br);
}

int ros_gazebo_bridge_connect(RosGazeboBridge* bridge) {
    if (!bridge) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;

    if (br->state >= ROS_GAZEBO_BRIDGE_STATE_CONNECTED) return 0;

    int sim_ok = 0;
    if (simulator_connect(br->sim) == 0) {
        sim_ok = 1;
    }

    if (br->node) {
        char master_uri[ROS_MAX_URI];
        snprintf(master_uri, sizeof(master_uri), "http://%s:%d",
                 br->config.ros_master_host, br->config.ros_master_port);
        if (ros_node_connect_to_master(br->node, master_uri) != 0) {
            bridge_set_error(br, -1, "连接ROS Master失败");
        }
    }

    br->state = sim_ok ? ROS_GAZEBO_BRIDGE_STATE_CONNECTED : ROS_GAZEBO_BRIDGE_STATE_ERROR;
    return (br->state == ROS_GAZEBO_BRIDGE_STATE_CONNECTED) ? 0 : -1;
}

int ros_gazebo_bridge_disconnect(RosGazeboBridge* bridge) {
    if (!bridge) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;

    br->running = 0;
    if (br->node) {
        ros_node_disconnect_from_master(br->node);
    }
    simulator_disconnect(br->sim);
    br->state = ROS_GAZEBO_BRIDGE_STATE_DISCONNECTED;
    return 0;
}

int ros_gazebo_bridge_is_connected(const RosGazeboBridge* bridge) {
    if (!bridge) return 0;
    const RosGazeboBridgeInternal* br = (const RosGazeboBridgeInternal*)bridge;
    return (br->state >= ROS_GAZEBO_BRIDGE_STATE_CONNECTED) ? 1 : 0;
}

int gazebo_is_environment_available(void) {
    /* 通过尝试TCP连接到默认ROS Master端口检测Gazebo/ROS环境是否可用 */
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return 0;
#endif
    int sock = (int)socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock < 0) return 0;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(11311);  /* ROS Master默认端口 */

#ifdef _WIN32
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
#else
    inet_aton("127.0.0.1", &addr.sin_addr);
#endif

    int result = 0;
    /* 设置非阻塞socket + 短超时（200ms），避免长时间等待 */
#ifdef _WIN32
    u_long mode = 1;
    ioctlsocket(sock, FIONBIO, &mode);
#else
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);
#endif

    fd_set fdset;
    struct timeval tv;
    FD_ZERO(&fdset);
    FD_SET((SOCKET_FD_TYPE)sock, &fdset);
    tv.tv_sec = 0;
    tv.tv_usec = 200000;  /* 200ms超时 */

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        result = 1;  /* 立即连接成功 */
    } else {
#ifdef _WIN32
        if (WSAGetLastError() == WSAEWOULDBLOCK) {
#else
        if (errno == EINPROGRESS) {
#endif
            if (select((int)sock + 1, NULL, &fdset, NULL, &tv) > 0) {
                int err = 0;
                socklen_t len = sizeof(err);
#ifdef _WIN32
                getsockopt(sock, SOL_SOCKET, SO_ERROR, (char*)&err, &len);
#else
                getsockopt(sock, SOL_SOCKET, SO_ERROR, &err, &len);
#endif
                if (err == 0) result = 1;
            }
        }
    }

    socket_close_pb(sock);
#ifdef _WIN32
    WSACleanup();
#endif
    return result;
}

RosGazeboBridgeState ros_gazebo_bridge_get_state(const RosGazeboBridge* bridge) {
    if (!bridge) return ROS_GAZEBO_BRIDGE_STATE_DISCONNECTED;
    return ((const RosGazeboBridgeInternal*)bridge)->state;
}

int ros_gazebo_bridge_start(RosGazeboBridge* bridge) {
    if (!bridge) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;

    if (simulator_start(br->sim) != 0) {
        bridge_set_error(br, -1, "启动Gazebo仿真失败");
        return -1;
    }

    br->running = 1;
    br->state = ROS_GAZEBO_BRIDGE_STATE_RUNNING;

    for (int i = 0; i < ROS_GAZEBO_BRIDGE_MAX_TOPICS; i++) {
        br->last_publish_times[i] = bridge_get_time();
    }

    if (!br->spin_thread_created) {
        br->spin_thread_created = 1;
        bridge_spin_loop(br);
    }

    return 0;
}

int ros_gazebo_bridge_stop(RosGazeboBridge* bridge) {
    if (!bridge) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;
    br->running = 0;
    int ret = simulator_stop(br->sim);
    br->state = ROS_GAZEBO_BRIDGE_STATE_CONNECTED;
    return ret;
}

int ros_gazebo_bridge_pause(RosGazeboBridge* bridge) {
    if (!bridge) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;
    return simulator_pause(br->sim);
}

int ros_gazebo_bridge_resume(RosGazeboBridge* bridge) {
    if (!bridge) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;
    return simulator_resume(br->sim);
}

int ros_gazebo_bridge_step(RosGazeboBridge* bridge, int num_steps) {
    if (!bridge) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;
    return simulator_step(br->sim, num_steps);
}

int ros_gazebo_bridge_reset(RosGazeboBridge* bridge) {
    if (!bridge) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;
    return simulator_reset(br->sim);
}

int ros_gazebo_bridge_spawn_model(RosGazeboBridge* bridge, const char* model_name,
                                   const float* position, const float* orientation) {
    if (!bridge || !model_name) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;

    RobotConfig robot_cfg;
    memset(&robot_cfg, 0, sizeof(robot_cfg));
    strncpy(robot_cfg.name, model_name, sizeof(robot_cfg.name) - 1);
    robot_cfg.type = ROBOT_TYPE_CUSTOM;

    float initial_pose[7];
    initial_pose[0] = position ? position[0] : 0.0f;
    initial_pose[1] = position ? position[1] : 0.0f;
    initial_pose[2] = position ? position[2] : 0.0f;
    initial_pose[3] = orientation ? orientation[0] : 0.0f;
    initial_pose[4] = orientation ? orientation[1] : 0.0f;
    initial_pose[5] = orientation ? orientation[2] : 0.0f;
    initial_pose[6] = orientation ? orientation[3] : 1.0f;

    int robot_id = simulator_load_robot(br->sim, &robot_cfg, initial_pose);
    if (robot_id < 0) {
        bridge_set_error(br, -1, "在Gazebo中加载模型失败");
        return -1;
    }

    if (robot_id < ROS_GAZEBO_BRIDGE_MAX_ROBOTS) {
        RosGazeboRobotInfo* info = &br->robot_infos[robot_id];
        memset(info, 0, sizeof(RosGazeboRobotInfo));
        strncpy(br->model_names[robot_id], model_name, ROS_GAZEBO_BRIDGE_FRAME_ID_MAX - 1);
        br->model_names[robot_id][ROS_GAZEBO_BRIDGE_FRAME_ID_MAX - 1] = '\0';
        info->num_links = 1;
        strncpy(info->links[0].name, "base_link", sizeof(info->links[0].name) - 1);
        info->links[0].type = ROS_GAZEBO_LINK_TYPE_BASE;
        info->links[0].pose_position[0] = initial_pose[0];
        info->links[0].pose_position[1] = initial_pose[1];
        info->links[0].pose_position[2] = initial_pose[2];
        info->links[0].pose_orientation[0] = initial_pose[3];
        info->links[0].pose_orientation[1] = initial_pose[4];
        info->links[0].pose_orientation[2] = initial_pose[5];
        info->links[0].pose_orientation[3] = initial_pose[6];
        br->robot_count = (robot_id + 1 > br->robot_count) ? (robot_id + 1) : br->robot_count;
    }

    return 0;
}

int ros_gazebo_bridge_delete_model(RosGazeboBridge* bridge, const char* model_name) {
    if (!bridge || !model_name) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;

    for (int i = 0; i < br->robot_count; i++) {
        if (strcmp(br->model_names[i], model_name) == 0) {
            int ret = simulator_remove_robot(br->sim, i);
            memset(&br->robot_infos[i], 0, sizeof(RosGazeboRobotInfo));
            memset(br->model_names[i], 0, ROS_GAZEBO_BRIDGE_FRAME_ID_MAX);
            if (i == br->robot_count - 1) {
                br->robot_count--;
            }
            return ret;
        }
    }
    return -1;
}

int ros_gazebo_bridge_get_model_state(RosGazeboBridge* bridge, const char* model_name,
                                       float* position, float* orientation,
                                       float* linear_vel, float* angular_vel) {
    if (!bridge || !model_name) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;

    for (int i = 0; i < br->robot_count; i++) {
        if (strcmp(br->model_names[i], model_name) == 0) {
            SimulatorRobotState state;
            memset(&state, 0, sizeof(state));
            if (simulator_get_robot_state(br->sim, i, &state) != 0) return -1;
            if (position) {
                position[0] = state.position[0];
                position[1] = state.position[1];
                position[2] = state.position[2];
            }
            if (orientation) {
                orientation[0] = state.orientation[0];
                orientation[1] = state.orientation[1];
                orientation[2] = state.orientation[2];
                orientation[3] = state.orientation[3];
            }
            if (linear_vel) {
                linear_vel[0] = state.velocity[0];
                linear_vel[1] = state.velocity[1];
                linear_vel[2] = state.velocity[2];
            }
            if (angular_vel) {
                angular_vel[0] = state.angular_velocity[0];
                angular_vel[1] = state.angular_velocity[1];
                angular_vel[2] = state.angular_velocity[2];
            }
            return 0;
        }
    }
    return -1;
}

int ros_gazebo_bridge_set_model_state(RosGazeboBridge* bridge, const char* model_name,
                                       const float* position, const float* orientation) {
    if (!bridge || !model_name) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;

    for (int i = 0; i < br->robot_count; i++) {
        if (strcmp(br->model_names[i], model_name) == 0) {
            float pose[7];
            if (position) {
                pose[0] = position[0]; pose[1] = position[1]; pose[2] = position[2];
            } else {
                pose[0] = 0; pose[1] = 0; pose[2] = 0;
            }
            if (orientation) {
                pose[3] = orientation[0]; pose[4] = orientation[1];
                pose[5] = orientation[2]; pose[6] = orientation[3];
            } else {
                pose[3] = 0; pose[4] = 0; pose[5] = 0; pose[6] = 1;
            }
            return simulator_set_joint_positions(br->sim, i, NULL, 0);
        }
    }
    return -1;
}

int ros_gazebo_bridge_apply_body_wrench(RosGazeboBridge* bridge, const char* model_name,
                                         const char* link_name,
                                         const float* force, const float* torque) {
    (void)link_name;
    if (!bridge || !model_name) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;

    for (int i = 0; i < br->robot_count; i++) {
        if (strcmp(br->robot_infos[i].joint_names[0], model_name) == 0) {
            RobotCommand cmd;
            memset(&cmd, 0, sizeof(cmd));
            cmd.mode = MOTION_MODE_TORQUE;
            if (force) {
                cmd.target_linear_velocity[0] = force[0];
                cmd.target_linear_velocity[1] = force[1];
                cmd.target_linear_velocity[2] = force[2];
            }
            if (torque) {
                cmd.target_angular_velocity[0] = torque[0];
                cmd.target_angular_velocity[1] = torque[1];
                cmd.target_angular_velocity[2] = torque[2];
            }
            return simulator_apply_robot_command(br->sim, i, &cmd);
        }
    }
    return -1;
}

int ros_gazebo_bridge_get_joint_state(RosGazeboBridge* bridge, int robot_id,
                                       float* positions, float* velocities,
                                       float* efforts, int max_joints) {
    if (!bridge || robot_id < 0) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;
    if (robot_id >= br->robot_count) return -1;

    int num_joints = 32;
    SimulatorRobotInfo rinfo;
    memset(&rinfo, 0, sizeof(rinfo));
    if (simulator_get_robot_info(br->sim, robot_id, &rinfo) == 0) {
        num_joints = rinfo.num_joints;
    }

    SimulatorRobotState state;
    memset(&state, 0, sizeof(state));
    if (simulator_get_robot_state(br->sim, robot_id, &state) != 0) return -1;

    int n = num_joints < max_joints ? num_joints : max_joints;
    if (positions) {
        memcpy(positions, state.joint_positions, n * sizeof(float));
    }
    if (velocities) {
        memcpy(velocities, state.joint_velocities, n * sizeof(float));
    }
    if (efforts) {
        memcpy(efforts, state.joint_torques, n * sizeof(float));
    }
    return n;
}

int ros_gazebo_bridge_set_joint_position(RosGazeboBridge* bridge, int robot_id,
                                          const char* joint_name, float position) {
    (void)joint_name;
    if (!bridge || robot_id < 0) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;
    return simulator_set_joint_positions(br->sim, robot_id, &position, 1);
}

int ros_gazebo_bridge_set_joint_velocity(RosGazeboBridge* bridge, int robot_id,
                                          const char* joint_name, float velocity) {
    (void)joint_name;
    if (!bridge || robot_id < 0) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;
    return simulator_set_joint_velocities(br->sim, robot_id, &velocity, 1);
}

int ros_gazebo_bridge_set_joint_torque(RosGazeboBridge* bridge, int robot_id,
                                        const char* joint_name, float torque) {
    (void)joint_name;
    if (!bridge || robot_id < 0) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;
    return simulator_set_joint_torques(br->sim, robot_id, &torque, 1);
}

int ros_gazebo_bridge_get_world_state(RosGazeboBridge* bridge, RosGazeboWorldState* state) {
    if (!bridge || !state) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;

    SimulatorStatus status;
    memset(&status, 0, sizeof(status));
    if (simulator_get_status(br->sim, &status) != 0) return -1;

    memset(state, 0, sizeof(RosGazeboWorldState));
    state->num_names = br->robot_count;
    for (int i = 0; i < br->robot_count && i < ROS_GAZEBO_BRIDGE_MAX_ROBOTS; i++) {
        char name_buf[ROS_GAZEBO_BRIDGE_FRAME_ID_MAX];
        snprintf(name_buf, sizeof(name_buf), "robot_%d", i);
        strncpy(state->names[i], name_buf, ROS_GAZEBO_BRIDGE_FRAME_ID_MAX - 1);
        state->names[i][ROS_GAZEBO_BRIDGE_FRAME_ID_MAX - 1] = '\0';
    }
    state->num_models = br->robot_count;
    state->num_lights = 1;
    state->num_physics_profiles = 1;
    state->simulation_time = (float)br->current_sim_time;
    state->paused = (br->state == ROS_GAZEBO_BRIDGE_STATE_CONNECTED) ? 1 : 0;
    state->step_count = status.step_count;
    state->real_time_factor = 1.0f;
    return 0;
}

int ros_gazebo_bridge_get_robot_info(RosGazeboBridge* bridge, int robot_id,
                                      RosGazeboRobotInfo* info) {
    if (!bridge || !info || robot_id < 0 || robot_id >= ROS_GAZEBO_BRIDGE_MAX_ROBOTS) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;
    if (robot_id >= br->robot_count) return -1;

    SimulatorRobotState state;
    memset(&state, 0, sizeof(state));
    if (simulator_get_robot_state(br->sim, robot_id, &state) != 0) return -1;

    SimulatorRobotInfo rinfo;
    int num_joints = 32;
    if (simulator_get_robot_info(br->sim, robot_id, &rinfo) == 0) {
        num_joints = rinfo.num_joints;
    }

    memcpy(info, &br->robot_infos[robot_id], sizeof(RosGazeboRobotInfo));
    info->num_joints = num_joints;
    for (int j = 0; j < num_joints && j < 32; j++) {
        info->joint_positions[j] = state.joint_positions[j];
        info->joint_velocities[j] = state.joint_velocities[j];
        info->joint_efforts[j] = state.joint_torques[j];
        snprintf(info->joint_names[j], ROS_GAZEBO_BRIDGE_FRAME_ID_MAX, "joint_%d", j);
    }
    if (info->num_links > 0) {
        info->links[0].pose_position[0] = state.position[0];
        info->links[0].pose_position[1] = state.position[1];
        info->links[0].pose_position[2] = state.position[2];
        info->links[0].pose_orientation[0] = state.orientation[0];
        info->links[0].pose_orientation[1] = state.orientation[1];
        info->links[0].pose_orientation[2] = state.orientation[2];
        info->links[0].pose_orientation[3] = state.orientation[3];
    }
    return 0;
}

int ros_gazebo_bridge_get_link_info(RosGazeboBridge* bridge, int robot_id,
                                     const char* link_name, RosGazeboLinkInfo* info) {
    if (!bridge || !link_name || !info || robot_id < 0) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;
    if (robot_id >= br->robot_count) return -1;

    RosGazeboRobotInfo* robot_info = &br->robot_infos[robot_id];
    for (int i = 0; i < robot_info->num_links; i++) {
        if (strcmp(robot_info->links[i].name, link_name) == 0) {
            memcpy(info, &robot_info->links[i], sizeof(RosGazeboLinkInfo));
            return 0;
        }
    }
    return -1;
}

int ros_gazebo_bridge_get_odometry(RosGazeboBridge* bridge, int robot_id,
                                    double* pos_x, double* pos_y, double* pos_z,
                                    double* ori_x, double* ori_y, double* ori_z, double* ori_w,
                                    double* lin_vel_x, double* lin_vel_y, double* lin_vel_z,
                                    double* ang_vel_x, double* ang_vel_y, double* ang_vel_z) {
    if (!bridge || robot_id < 0) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;
    if (robot_id >= br->robot_count) return -1;

    SimulatorRobotState state;
    memset(&state, 0, sizeof(state));
    if (simulator_get_robot_state(br->sim, robot_id, &state) != 0) return -1;

    if (pos_x) *pos_x = state.position[0];
    if (pos_y) *pos_y = state.position[1];
    if (pos_z) *pos_z = state.position[2];
    if (ori_x) *ori_x = state.orientation[0];
    if (ori_y) *ori_y = state.orientation[1];
    if (ori_z) *ori_z = state.orientation[2];
    if (ori_w) *ori_w = state.orientation[3];
    if (lin_vel_x) *lin_vel_x = state.velocity[0];
    if (lin_vel_y) *lin_vel_y = state.velocity[1];
    if (lin_vel_z) *lin_vel_z = state.velocity[2];
    if (ang_vel_x) *ang_vel_x = state.angular_velocity[0];
    if (ang_vel_y) *ang_vel_y = state.angular_velocity[1];
    if (ang_vel_z) *ang_vel_z = state.angular_velocity[2];
    return 0;
}

int ros_gazebo_bridge_get_sensor_data(RosGazeboBridge* bridge, int sensor_id,
                                       float* data, int max_size, int* size_out) {
    if (!bridge || !data || !size_out) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;

    SimulatorSensorData sensor_data;
    memset(&sensor_data, 0, sizeof(sensor_data));
    if (simulator_get_sensor_data(br->sim, sensor_id, &sensor_data) != 0) return -1;

    int n = sensor_data.data_size < max_size ? sensor_data.data_size : max_size;
    memcpy(data, sensor_data.data, n * sizeof(float));
    *size_out = n;
    return 0;
}

int ros_gazebo_bridge_publish_cmd_vel(RosGazeboBridge* bridge, const RosTwist* cmd) {
    if (!bridge || !cmd) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;

    if (br->robot_count > 0) {
        RobotCommand robot_cmd;
        memset(&robot_cmd, 0, sizeof(robot_cmd));
        robot_cmd.mode = MOTION_MODE_VELOCITY;
        robot_cmd.target_linear_velocity[0] = (float)cmd->linear.x;
        robot_cmd.target_linear_velocity[1] = (float)cmd->linear.y;
        robot_cmd.target_linear_velocity[2] = (float)cmd->linear.z;
        robot_cmd.target_angular_velocity[0] = (float)cmd->angular.x;
        robot_cmd.target_angular_velocity[1] = (float)cmd->angular.y;
        robot_cmd.target_angular_velocity[2] = (float)cmd->angular.z;
        return simulator_apply_robot_command(br->sim, 0, &robot_cmd);
    }
    return -1;
}

int ros_gazebo_bridge_publish_odometry(RosGazeboBridge* bridge, int robot_id) {
    if (!bridge || robot_id < 0 || !((RosGazeboBridgeInternal*)bridge)->node) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;
    if (!br->advertise_odom) return -1;

    SimulatorRobotState state;
    memset(&state, 0, sizeof(state));
    if (simulator_get_robot_state(br->sim, robot_id, &state) != 0) return -1;

    RosOdometry odom;
    memset(&odom, 0, sizeof(odom));
    odom.pose.position.x = state.position[0];
    odom.pose.position.y = state.position[1];
    odom.pose.position.z = state.position[2];
    odom.pose.orientation.x = state.orientation[0];
    odom.pose.orientation.y = state.orientation[1];
    odom.pose.orientation.z = state.orientation[2];
    odom.pose.orientation.w = state.orientation[3];
    odom.twist.linear.x = state.velocity[0];
    odom.twist.linear.y = state.velocity[1];
    odom.twist.linear.z = state.velocity[2];
    odom.twist.angular.x = state.angular_velocity[0];
    odom.twist.angular.y = state.angular_velocity[1];
    odom.twist.angular.z = state.angular_velocity[2];

    uint8_t buffer[ROS_TCP_BUF_SIZE];
    size_t written = 0;
    if (ros_serialize_odometry(&odom, buffer, sizeof(buffer), &written) == ROS_OK) {
        return ros_node_publish(br->node, ROS_ODOM_TOPIC, buffer, written);
    }
    return -1;
}

int ros_gazebo_bridge_publish_joint_states(RosGazeboBridge* bridge, int robot_id) {
    if (!bridge || robot_id < 0 || !((RosGazeboBridgeInternal*)bridge)->node) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;
    if (!br->advertise_joint_states) return -1;

    int num_joints = 32;
    SimulatorRobotInfo rinfo;
    memset(&rinfo, 0, sizeof(rinfo));
    if (simulator_get_robot_info(br->sim, robot_id, &rinfo) == 0) {
        num_joints = rinfo.num_joints;
    }

    SimulatorRobotState state;
    memset(&state, 0, sizeof(state));
    if (simulator_get_robot_state(br->sim, robot_id, &state) != 0) return -1;

    RosJointState js;
    memset(&js, 0, sizeof(js));
    js.names_count = num_joints;
    js.positions_count = num_joints;
    js.velocities_count = num_joints;
    js.efforts_count = num_joints;

    for (int j = 0; j < num_joints && j < 32; j++) {
        snprintf(js.name[j], sizeof(js.name[j]), "joint_%d", j);
        js.position[j] = state.joint_positions[j];
        js.velocity[j] = state.joint_velocities[j];
        js.effort[j] = state.joint_torques[j];
    }

    uint8_t buffer[ROS_TCP_BUF_SIZE];
    size_t written = 0;
    if (ros_serialize_joint_state(&js, buffer, sizeof(buffer), &written) == ROS_OK) {
        return ros_node_publish(br->node, ROS_JOINT_STATES_TOPIC, buffer, written);
    }
    return -1;
}

int ros_gazebo_bridge_publish_laserscan(RosGazeboBridge* bridge, int sensor_id) {
    if (!bridge || !((RosGazeboBridgeInternal*)bridge)->node) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;
    if (!br->advertise_scan) return -1;

    SimulatorSensorData sensor_data;
    memset(&sensor_data, 0, sizeof(sensor_data));
    if (simulator_get_sensor_data(br->sim, sensor_id, &sensor_data) != 0) {
        float ranges[1024];
        memset(ranges, 0, sizeof(ranges));
        for (int i = 0; i < 360 && i < 1024; i++) {
            ranges[i] = 5.0f;
        }
        RosLaserScan scan;
        memset(&scan, 0, sizeof(scan));
        scan.angle_min = -3.14159265f;
        scan.angle_max = 3.14159265f;
        scan.angle_increment = (float)(2.0 * 3.14159265 / 360.0);
        scan.range_min = 0.1f;
        scan.range_max = 10.0f;
        scan.ranges_size = 360;
        scan.intensities_size = 0;
        memcpy(scan.ranges, ranges, 360 * sizeof(float));

        uint8_t buffer[ROS_TCP_BUF_SIZE];
        size_t written = 0;
        if (ros_serialize_laserscan(&scan, buffer, sizeof(buffer), &written) == ROS_OK) {
            return ros_node_publish(br->node, ROS_SCAN_TOPIC, buffer, written);
        }
        return -1;
    }

    RosLaserScan scan;
    memset(&scan, 0, sizeof(scan));
    scan.angle_min = -3.14159265f;
    scan.angle_max = 3.14159265f;
    scan.angle_increment = (float)(2.0 * 3.14159265 / 360.0);
    scan.range_min = 0.1f;
    scan.range_max = 10.0f;

    int num_rays = sensor_data.data_size > 0 ? sensor_data.data_size : 360;
    if (num_rays > 1024) num_rays = 1024;
    scan.ranges_size = num_rays;
    scan.intensities_size = 0;

    for (int i = 0; i < num_rays; i++) {
        scan.ranges[i] = (i < sensor_data.data_size) ? sensor_data.data[i] : 5.0f;
    }

    uint8_t buffer[ROS_TCP_BUF_SIZE];
    size_t written = 0;
    if (ros_serialize_laserscan(&scan, buffer, sizeof(buffer), &written) == ROS_OK) {
        return ros_node_publish(br->node, ROS_SCAN_TOPIC, buffer, written);
    }
    return -1;
}

int ros_gazebo_bridge_publish_imu(RosGazeboBridge* bridge, int sensor_id) {
    if (!bridge || !((RosGazeboBridgeInternal*)bridge)->node) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;
    if (!br->advertise_imu) return -1;

    SimulatorSensorData sensor_data;
    memset(&sensor_data, 0, sizeof(sensor_data));
    SimulatorRobotState state;
    memset(&state, 0, sizeof(state));

    int has_state = (simulator_get_robot_state(br->sim, 0, &state) == 0);
    int has_sensor = (simulator_get_sensor_data(br->sim, sensor_id, &sensor_data) == 0);

    RosImu imu;
    memset(&imu, 0, sizeof(imu));

    if (has_sensor && sensor_data.data_size >= 6) {
        imu.linear_acceleration.x = sensor_data.data[0];
        imu.linear_acceleration.y = sensor_data.data[1];
        imu.linear_acceleration.z = sensor_data.data[2];
        imu.angular_velocity.x = sensor_data.data[3];
        imu.angular_velocity.y = sensor_data.data[4];
        imu.angular_velocity.z = sensor_data.data[5];
        if (sensor_data.data_size >= 10) {
            imu.orientation.x = sensor_data.data[6];
            imu.orientation.y = sensor_data.data[7];
            imu.orientation.z = sensor_data.data[8];
            imu.orientation.w = sensor_data.data[9];
        }
    }

    if (has_state && !has_sensor) {
        imu.linear_acceleration.x = state.acceleration[0];
        imu.linear_acceleration.y = state.acceleration[1];
        imu.linear_acceleration.z = state.acceleration[2];
        imu.angular_velocity.x = state.angular_velocity[0];
        imu.angular_velocity.y = state.angular_velocity[1];
        imu.angular_velocity.z = state.angular_velocity[2];
        imu.orientation.x = state.orientation[0];
        imu.orientation.y = state.orientation[1];
        imu.orientation.z = state.orientation[2];
        imu.orientation.w = state.orientation[3];
    }

    if (!has_state && !has_sensor) {
        imu.orientation.w = 1.0;
    }

    uint8_t buffer[ROS_TCP_BUF_SIZE];
    size_t written = 0;
    if (ros_serialize_imu(&imu, buffer, sizeof(buffer), &written) == ROS_OK) {
        return ros_node_publish(br->node, ROS_IMU_TOPIC, buffer, written);
    }
    return -1;
}

int ros_gazebo_bridge_publish_camera(RosGazeboBridge* bridge, int sensor_id) {
    (void)sensor_id;
    if (!bridge || !((RosGazeboBridgeInternal*)bridge)->node) return -1;
    RosGazeboBridgeInternal* br = (RosGazeboBridgeInternal*)bridge;
    if (!br->advertise_camera) return -1;

    RosImage image;
    memset(&image, 0, sizeof(image));
    image.width = 640;
    image.height = 480;
    strncpy(image.encoding, "rgb8", sizeof(image.encoding) - 1);
    image.is_bigendian = 0;
    image.step = 640 * 3;
    image.data_size = 640 * 480 * 3;

    uint8_t buffer[ROS_TCP_BUF_SIZE];
    size_t written = 0;
    if (ros_serialize_image(&image, buffer, sizeof(buffer), &written) == ROS_OK) {
        return ros_node_publish(br->node, ROS_CAMERA_TOPIC, buffer, written);
    }
    return -1;
}

int ros_gazebo_bridge_get_robot_count(RosGazeboBridge* bridge) {
    if (!bridge) return 0;
    return ((RosGazeboBridgeInternal*)bridge)->robot_count;
}

int ros_gazebo_bridge_get_simulator_handle(RosGazeboBridge* bridge, void** sim_out) {
    if (!bridge || !sim_out) return -1;
    *sim_out = ((RosGazeboBridgeInternal*)bridge)->sim;
    return 0;
}

const char* ros_gazebo_bridge_get_last_error(RosGazeboBridge* bridge) {
    if (!bridge) return "桥接未初始化";
    return ((RosGazeboBridgeInternal*)bridge)->last_error;
}

int ros_gazebo_bridge_get_error_code(RosGazeboBridge* bridge) {
    if (!bridge) return -1;
    return ((RosGazeboBridgeInternal*)bridge)->error_code;
}
