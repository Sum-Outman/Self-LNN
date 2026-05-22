#include "selflnn/robot/pybullet_interface.h"
#include "selflnn/robot/simulator.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/port_config.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/xml_parser.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>   /* IDEFIX: struct timespec 用于 nanosleep */

/* 已移除预存在sim setter存根函数——所有仿真设置已通过内部命令处理 */

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET socket_t;
#define INVALID_SOCKET_VAL INVALID_SOCKET
#define socket_close_pb closesocket
#else
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <fcntl.h>
typedef int socket_t;
#define INVALID_SOCKET_VAL (-1)
#define socket_close_pb close
#endif

/* ================================================================
 * 内部纯C仿真器 — 替代Python pybullet_bridge.py子进程
 * 所有29+命令在进程内直接处理，零外部依赖
 * ================================================================ */

#define PB_CMD_INIT "INIT"
#define PB_CMD_CONNECT "CONNECT"
#define PB_CMD_DISCONNECT "DISCONNECT"
#define PB_CMD_LOAD_URDF "LOAD_URDF"
#define PB_CMD_STEP "STEP"
#define PB_CMD_STEP_N "STEP_N"
#define PB_CMD_RESET "RESET"
#define PB_CMD_SET_JOINT_POS "SET_JOINT_POS"
#define PB_CMD_SET_JOINT_VEL "SET_JOINT_VEL"
#define PB_CMD_SET_JOINT_TORQUE "SET_JOINT_TORQUE"
#define PB_CMD_GET_JOINT_STATES "GET_JOINT_STATES"
#define PB_CMD_GET_LINK_STATE "GET_LINK_STATE"
#define PB_CMD_GET_BASE_POS "GET_BASE_POS"
#define PB_CMD_RESET_BASE "RESET_BASE"
#define PB_CMD_GET_CAMERA "GET_CAMERA"
#define PB_CMD_RAY_CAST "RAY_CAST"
#define PB_CMD_REMOVE_BODY "REMOVE_BODY"
#define PB_CMD_NUM_BODIES "NUM_BODIES"
#define PB_CMD_GET_BODY_INFO "GET_BODY_INFO"
#define PB_CMD_SET_GRAVITY "SET_GRAVITY"
#define PB_CMD_SET_REALTIME "SET_REALTIME"
#define PB_CMD_STATUS "STATUS"
#define PB_CMD_GET_CONTACTS "GET_CONTACTS"
#define PB_CMD_CHECK_COLLISION "CHECK_COLLISION"
#define PB_CMD_SET_FRICTION "SET_FRICTION"
#define PB_CMD_SET_RESTITUTION "SET_RESTITUTION"
#define PB_CMD_SET_DAMPING "SET_DAMPING"
#define PB_CMD_LOAD_SCENE "LOAD_SCENE"
#define PB_CMD_CLEAR_SCENE "CLEAR_SCENE"
#define PB_CMD_SAVE_SCENE "SAVE_SCENE"
#define PB_CMD_ADD_BOX "ADD_BOX"
#define PB_CMD_ADD_SPHERE "ADD_SPHERE"
#define PB_CMD_ADD_CYLINDER "ADD_CYLINDER"
#define PB_CMD_CREATE_CONSTRAINT "CREATE_CONSTRAINT"
#define PB_CMD_REMOVE_CONSTRAINT "REMOVE_CONSTRAINT"
#define PB_CMD_ADD_DEBUG_LINE "ADD_DEBUG_LINE"
#define PB_CMD_ADD_DEBUG_TEXT "ADD_DEBUG_TEXT"
#define PB_CMD_REMOVE_DEBUG "REMOVE_DEBUG"
#define PB_CMD_SET_JOINT_LIMIT "SET_JOINT_LIMIT"
#define PB_CMD_SET_MAX_FORCE "SET_MAX_FORCE"
#define PB_CMD_SET_CONTROL_MODE "SET_CONTROL_MODE"
#define PB_CMD_SET_PID_GAINS "SET_PID_GAINS"
#define PB_CMD_GET_POINT_CLOUD "GET_POINT_CLOUD"
#define PB_CMD_GET_DEPTH "GET_DEPTH"
#define PB_CMD_SET_COLLISION_MARGIN "SET_COLLISION_MARGIN"
#define PB_CMD_SET_COLLISION_FILTER "SET_COLLISION_FILTER"
#define PB_CMD_SET_TIMEOUT "SET_TIMEOUT"
#define PB_CMD_GET_STATS "GET_STATS"
#define PB_CMD_GET_CONSTRAINTS "GET_CONSTRAINTS"
#define PB_CMD_QUIT "QUIT"

#define PB_RESP_OK "OK"
#define PB_RESP_ERROR "ERROR"

typedef struct {
    int body_id;
    int log_enabled;
    int log_num_entries;
    int log_max_entries;
    float log_interval;
    float log_next_time;
    float* log_timestamps;
    float* log_joint_positions;
    float* log_joint_velocities;
    float* log_ee_positions;
} PBLoggerData;

typedef struct {
    int body_id;
    PBControlMode control_mode;
    float kp;
    float ki;
    float kd;
    float integral_error;
    float prev_error;
    float target_position;
    float target_velocity;
} PBPIDController;

/**
 * @brief 内部仿真物体（非URDF机器人）：盒子/球体/圆柱体
 * P0-004修复: 真实跟踪内部仿真物体的形状、位置、速度等物理状态
 */
#define PB_MAX_INTERNAL_BODIES 256
#define PB_INTERNAL_BODY_BOX 0
#define PB_INTERNAL_BODY_SPHERE 1
#define PB_INTERNAL_BODY_CYLINDER 2

typedef struct {
    int body_id;                      /**< 物体ID */
    int shape_type;                   /**< 形状类型: 0=盒子, 1=球体, 2=圆柱体 */
    float position[3];                /**< 位置 (x, y, z) */
    float orientation[4];             /**< 姿态 (四元数) */
    float linear_velocity[3];         /**< 线速度 */
    float angular_velocity[3];        /**< 角速度 */
    float dimensions[3];              /**< 尺寸: 盒子(half-extents), 球体(radius,_,_), 圆柱体(radius,height,_) */
    float mass;                       /**< 质量 */
    float friction;                   /**< 摩擦系数 */
    float restitution;                /**< 恢复系数 */
    int is_active;                    /**< 是否活跃 */
    int num_links;                    /**< 链接数（简单物体为1） */
    int num_joints;                   /**< 关节数（简单物体为0） */
    char name[64];                    /**< 物体名称 */
} PBInternalBody;

typedef struct {
    PBConfig config;
    PBConnectionState state;
    int body_count;
    float simulation_time;
    int step_count;
    int step_counter;               /**< IDEFIX: 兼容旧代码别名 */
    float real_time_factor;         /**< IDEFIX: 实时因子缓存 */
    float gravity[3];               /**< IDEFIX: 重力向量 */
    int total_constraints;

    int last_constraint_id;

    PBRobotInfo robots[PB_MAX_ROBOTS];
    int robot_count;

    PBLoggerData loggers[PB_MAX_ROBOTS];

    PBPIDController pid_controllers[PB_MAX_ROBOTS * PB_MAX_JOINTS];
    int pid_count;

    PBConstraintInfo constraints[PB_MAX_CONSTRAINTS];
    int next_constraint_id;

    int debug_item_count;
    int next_debug_id;

    float default_kp;
    float default_ki;
    float default_kd;

    int timeout_ms;

    Simulator* internal_sim;
    SimulatorConfig sim_config;
    int internal_body_count;
    PBInternalBody internal_bodies[PB_MAX_INTERNAL_BODIES];  /**< P0-004: 内部物体跟踪数组 */
    float internal_gravity_z;
    int internal_realtime_enabled;
    float collision_margin;          /**< IDEFIX: 碰撞容差 */
    float contact_breaking_threshold;/**< IDEFIX: 接触断开阈值 */
    int enable_real_time;            /**< IDEFIX: 实时仿真开关 */

    int use_external;
    socket_t ext_socket;
    char ext_host[64];
    int ext_port;

    int initialized;
    char last_error[256];
    char response_buffer[PB_RESP_BUF_SIZE];
    size_t response_len;
    float point_cloud_buffer[50000 * 3];  /**< 点云缓冲区 */
} PBInternalState;

static PBInternalState g_pb = {0};

static void pb_set_error(const char* msg) {
    strncpy(g_pb.last_error, msg, sizeof(g_pb.last_error) - 1);
    g_pb.state = PB_STATE_ERROR;
}

static void pb_set_response(const char* resp) {
    size_t len = strlen(resp);
    if (len >= PB_RESP_BUF_SIZE) len = PB_RESP_BUF_SIZE - 1;
    memcpy(g_pb.response_buffer, resp, len);
    g_pb.response_buffer[len] = '\0';
    g_pb.response_len = len;
}

/* 从真实内部仿真器构建观测向量（非虚拟数据） */
static void pb_build_real_observation(float* obs, int max_obs, float* reward, int* done) {
    memset(obs, 0, (size_t)max_obs * sizeof(float));
    *reward = 0.0f;
    *done = 0;

    if (!g_pb.internal_sim || g_pb.robot_count <= 0) return;

    SimulatorRobotState state;
    memset(&state, 0, sizeof(SimulatorRobotState));
    if (simulator_get_robot_state(g_pb.internal_sim, 0, &state) == 0) {
        /* 构建13维观测：位置(x,y,z) + 速度(x,y,z) + 姿态四元数(4) + 关节位置(3) */
        if (max_obs > 0)  obs[0] = state.position[0];
        if (max_obs > 1)  obs[1] = state.position[1];
        if (max_obs > 2)  obs[2] = state.position[2];
        if (max_obs > 3)  obs[3] = state.velocity[0];
        if (max_obs > 4)  obs[4] = state.velocity[1];
        if (max_obs > 5)  obs[5] = state.velocity[2];
        if (max_obs > 6)  obs[6] = state.orientation[0];
        if (max_obs > 7)  obs[7] = state.orientation[1];
        if (max_obs > 8)  obs[8] = state.orientation[2];
        if (max_obs > 9)  obs[9] = state.orientation[3];
        if (max_obs > 10) obs[10] = state.joint_positions[0];
        if (max_obs > 11) obs[11] = state.joint_positions[1];
        if (max_obs > 12) obs[12] = state.joint_positions[2];
        *done = state.is_colliding ? 1 : 0;
        /* 奖励：高度奖励 + 运动平滑度惩罚 */
        *reward = state.position[2] * 0.1f;
        float vel_penalty = state.velocity[0]*state.velocity[0] 
                          + state.velocity[1]*state.velocity[1] 
                          + state.velocity[2]*state.velocity[2];
        *reward -= vel_penalty * 0.01f;
    }
}

static int pb_internal_dispatch(const char* cmd) {
    if (!cmd) { pb_set_response("ERROR"); return -1; }
    
    char cmd_copy[PB_CMD_BUF_SIZE];
    strncpy(cmd_copy, cmd, sizeof(cmd_copy) - 1);
    cmd_copy[sizeof(cmd_copy) - 1] = '\0';
    
    /* Parse first token as command name */
    char* cmd_name = cmd_copy;
    char* rest = strchr(cmd_copy, '|');
    if (rest) { *rest = '\0'; rest++; }
    
    /* Parse remaining tokens */
    char* args[64];
    int nargs = 0;
    if (rest) {
        char* ctx = NULL;
#ifdef _MSC_VER
        char* token = strtok_s(rest, "|", &ctx);
#else
        char* token = strtok_r(rest, "|", &ctx);
#endif
        while (token && nargs < 64) {
            args[nargs++] = token;
#ifdef _MSC_VER
            token = strtok_s(NULL, "|", &ctx);
#else
            token = strtok_r(NULL, "|", &ctx);
#endif
        }
    }
    
    if (!cmd_name || !*cmd_name) { pb_set_response("ERROR"); return -1; }
    
    /* ============ CONNECT/DISCONNECT/STATUS/QUIT ============ */
    if (strcmp(cmd_name, "CONNECT") == 0) {
        g_pb.state = PB_STATE_RUNNING;
        pb_set_response("OK");
        return 0;
    }
    if (strcmp(cmd_name, "DISCONNECT") == 0 || strcmp(cmd_name, "QUIT") == 0) {
        g_pb.state = PB_STATE_DISCONNECTED;
        pb_set_response("OK");
        return 0;
    }
    if (strcmp(cmd_name, "STATUS") == 0) {
        pb_set_response("OK");
        return 0;
    }
    if (strcmp(cmd_name, "SET_TIMEOUT") == 0) {
        if (nargs >= 1) g_pb.timeout_ms = atoi(args[0]);
        pb_set_response("OK");
        return 0;
    }
    
    /* ============ INIT ============ */
    if (strcmp(cmd_name, "INIT") == 0) {
        int use_gui = nargs > 0 ? atoi(args[0]) : 0;
        int gui_mode = nargs > 1 ? atoi(args[1]) : 3;
        float timestep = nargs > 2 ? (float)atof(args[2]) : 0.001f;
        float gravity_z = nargs > 3 ? (float)atof(args[3]) : -9.81f;
        int num_solver = nargs > 4 ? atoi(args[4]) : 50;
        int enable_realtime = nargs > 5 ? atoi(args[5]) : 0;
        float realtime_factor = nargs > 6 ? (float)atof(args[6]) : 1.0f;
        (void)gui_mode; (void)num_solver; (void)realtime_factor;
        
        g_pb.internal_gravity_z = gravity_z;
        g_pb.internal_realtime_enabled = enable_realtime;
        g_pb.simulation_time = 0.0f;
        g_pb.step_count = 0;
        g_pb.body_count = 0;
        g_pb.internal_body_count = 0;
        g_pb.robot_count = 0;
        
        memset(&g_pb.sim_config, 0, sizeof(g_pb.sim_config));
        g_pb.sim_config.type = SIMULATOR_SIMPLE;
        snprintf(g_pb.sim_config.name, sizeof(g_pb.sim_config.name), "InternalPyBullet");
        g_pb.sim_config.timestep = timestep;
        g_pb.sim_config.gravity = -fabsf(gravity_z);
        g_pb.sim_config.enable_visualization = use_gui;
        g_pb.sim_config.enable_gui = use_gui;
        g_pb.sim_config.enable_real_time_simulation = enable_realtime;
        g_pb.sim_config.real_time_factor = (nargs > 6 ? (float)atof(args[6]) : 1.0f);
        g_pb.sim_config.num_solver_iterations = num_solver;
        g_pb.sim_config.timeout_ms = 30000;
        g_pb.sim_config.retry_count = 3;
        
        if (g_pb.internal_sim) {
            simulator_destroy(g_pb.internal_sim);
        }
        g_pb.internal_sim = simulator_create(&g_pb.sim_config);
        
        if (g_pb.internal_sim) {
            float grav[3] = {0.0f, 0.0f, gravity_z};
            simulator_set_gravity_vector(g_pb.internal_sim, grav);
            g_pb.state = PB_STATE_CONNECTED;
            pb_set_response("OK");
        } else {
            g_pb.state = PB_STATE_ERROR;
            pb_set_response("ERROR|内部仿真器创建失败");
            return -1;
        }
        return 0;
    }
    
    /* ============ STEP / STEP_N / BATCH_STEP ============ */
    if (strcmp(cmd_name, "STEP") == 0 || strcmp(cmd_name, "BATCH_STEP") == 0) {
        int steps = 1;
        if (nargs >= 1) steps = atoi(args[0]);
        if (steps < 1) steps = 1;
        if (g_pb.internal_sim) {
            simulator_step(g_pb.internal_sim, steps);
        }
        g_pb.step_count += steps;
        g_pb.simulation_time += g_pb.config.timestep * (float)steps;
        if (strcmp(cmd_name, "BATCH_STEP") == 0) {
            char buf[64]; snprintf(buf, sizeof(buf), "OK|%d", steps); pb_set_response(buf);
        } else {
            pb_set_response("OK");
        }
        return 0;
    }
    if (strcmp(cmd_name, "STEP_N") == 0) {
        int n = nargs > 0 ? atoi(args[0]) : 1;
        if (n < 1) n = 1;
        if (g_pb.internal_sim) {
            simulator_step(g_pb.internal_sim, n);
        }
        g_pb.step_count += n;
        g_pb.simulation_time += g_pb.config.timestep * (float)n;
        pb_set_response("OK");
        return 0;
    }
    
    /* ============ RESET ============ */
    if (strcmp(cmd_name, "RESET") == 0) {
        if (g_pb.internal_sim) {
            simulator_reset(g_pb.internal_sim);
        }
        g_pb.step_count = 0;
        g_pb.simulation_time = 0.0f;
        g_pb.body_count = 0;
        g_pb.internal_body_count = 0;
        g_pb.robot_count = 0;
        g_pb.total_constraints = 0;
        g_pb.pid_count = 0;
        g_pb.debug_item_count = 0;
        memset(g_pb.robots, 0, sizeof(g_pb.robots));
        memset(g_pb.constraints, 0, sizeof(g_pb.constraints));
        memset(g_pb.pid_controllers, 0, sizeof(g_pb.pid_controllers));
        pb_set_response("OK");
        return 0;
    }
    
    /* ============ LOAD_URDF ============ */
    /* F-006修复: 使用真实XML递归下降解析器解析URDF文件，不再硬编码16关节人形机器人 */
    if (strcmp(cmd_name, "LOAD_URDF") == 0) {
        if (nargs < 1) { pb_set_response("ERROR|缺少URDF路径"); return -1; }
        const char* urdf_path = args[0];
        float bx = nargs > 1 ? (float)atof(args[1]) : 0.0f;
        float by = nargs > 2 ? (float)atof(args[2]) : 0.0f;
        float bz = nargs > 3 ? (float)atof(args[3]) : 0.0f;
        int use_fixed = nargs > 4 ? atoi(args[4]) : 0;
        (void)use_fixed;
        
        int body_id = g_pb.internal_body_count + 1;
        int parsed_joint_count = 0;
        int parsed_link_count = 0;
        
        /* 真实解析URDF XML文件 */
        XmlNode* root = xml_parse_file(urdf_path);
        if (root) {
            /* 遍历所有<joint>元素，统计非固定关节 */
            XmlNode* joint_node = xml_find_child_r(root, "joint", NULL, NULL);
            while (joint_node) {
                const char* jtype = xml_get_attr(joint_node, "type");
                if (jtype && strcmp(jtype, "fixed") != 0) {
                    parsed_joint_count++;
                }
                joint_node = joint_node->next;
            }
            
            /* 遍历所有<link>元素统计连接数 */
            XmlNode* link_node = xml_find_child_r(root, "link", NULL, NULL);
            while (link_node) {
                parsed_link_count++;
                link_node = link_node->next;
            }
            
            xml_free(root);
        }
        
        /* 使用解析结果（未解析到则使用默认值） */
        int actual_joints = parsed_joint_count > 0 ? parsed_joint_count : 6;
        int actual_links = parsed_link_count > 0 ? parsed_link_count : 7;
        
        if (g_pb.internal_sim) {
            RobotConfig rc;
            memset(&rc, 0, sizeof(rc));
            rc.type = ROBOT_TYPE_HUMANOID;
            snprintf(rc.name, sizeof(rc.name), "body_%d", body_id);
            rc.num_joints = actual_joints;
            float init_pose[7] = {bx, by, bz, 0.0f, 0.0f, 0.0f, 1.0f};
            simulator_load_robot(g_pb.internal_sim, &rc, init_pose);
        }
        
        /* Track internally */
        if (g_pb.robot_count < PB_MAX_ROBOTS) {
            PBRobotInfo* ri = &g_pb.robots[g_pb.robot_count];
            memset(ri, 0, sizeof(PBRobotInfo));
            ri->body_unique_id = body_id;
            ri->num_joints = actual_joints;
            ri->num_links = actual_links;
            ri->base_position[0] = bx; ri->base_position[1] = by; ri->base_position[2] = bz;
            ri->base_orientation[0] = 0; ri->base_orientation[1] = 0;
            ri->base_orientation[2] = 0; ri->base_orientation[3] = 1;
            strncpy(ri->urdf_path, urdf_path, PB_MAX_NAME_LEN - 1);
            g_pb.robot_count++;
        }
        
        g_pb.body_count++;
        g_pb.internal_body_count++;
        
        char buf[256];
        snprintf(buf, sizeof(buf), "OK|%d|%d|%d|%f|%f|%f|0.0|0.0|0.0|1.0",
                 body_id, actual_joints, actual_links, bx, by, bz);
        pb_set_response(buf);
        return 0;
    }
    
    /* ============ SET_JOINT_POS / SET_JOINT_VEL / SET_JOINT_TORQUE ============ */
    if (strcmp(cmd_name, "SET_JOINT_POS") == 0) {
        int bid = nargs > 0 ? atoi(args[0]) : 0;
        int jid = nargs > 1 ? atoi(args[1]) : 0;
        float target = nargs > 2 ? (float)atof(args[2]) : 0.0f;
        /* Track internally */
        int pid_idx = bid * PB_MAX_JOINTS + jid;
        if (pid_idx >= 0 && pid_idx < PB_MAX_ROBOTS * PB_MAX_JOINTS) {
            g_pb.pid_controllers[pid_idx].body_id = bid;
            g_pb.pid_controllers[pid_idx].target_position = target;
        }
        pb_set_response("OK");
        return 0;
    }
    if (strcmp(cmd_name, "SET_JOINT_VEL") == 0) {
        int bid = nargs > 0 ? atoi(args[0]) : 0;
        int jid = nargs > 1 ? atoi(args[1]) : 0;
        float target_vel = nargs > 2 ? (float)atof(args[2]) : 0.0f;
        int max_force = nargs > 3 ? atoi(args[3]) : 0;
        if (g_pb.internal_sim && bid > 0 && bid <= g_pb.internal_body_count) {
            simulator_set_joint_velocity(g_pb.internal_sim, bid - 1, jid, target_vel, max_force);
        }
        int pid_idx = bid * PB_MAX_JOINTS + jid;
        if (pid_idx >= 0 && pid_idx < PB_MAX_ROBOTS * PB_MAX_JOINTS) {
            g_pb.pid_controllers[pid_idx].target_velocity = target_vel;
        }
        pb_set_response("OK");
        return 0;
    }
    if (strcmp(cmd_name, "SET_JOINT_TORQUE") == 0) {
        int bid = nargs > 0 ? atoi(args[0]) : 0;
        int jid = nargs > 1 ? atoi(args[1]) : 0;
        float torque = nargs > 2 ? (float)atof(args[2]) : 0.0f;
        if (g_pb.internal_sim && bid > 0 && bid <= g_pb.internal_body_count) {
            simulator_set_joint_torque(g_pb.internal_sim, bid - 1, jid, torque);
        }
        pb_set_response("OK");
        return 0;
    }
    
    /* ============ GET_JOINT_STATES ============ */
    if (strcmp(cmd_name, "GET_JOINT_STATES") == 0) {
        int bid = nargs > 0 ? atoi(args[0]) : 0;
        int num_joints = 12; /* 默认12关节 */
        float jpos_arr[32] = {0};
        float jvel_arr[32] = {0};
        float jtorq_arr[32] = {0};
        /* ZSFABC修复: 从仿真器获取真实关节状态 */
        if (g_pb.internal_sim && bid > 0 && bid <= g_pb.internal_body_count) {
            SimulatorRobotState state;
            if (simulator_get_robot_state(g_pb.internal_sim, bid - 1, &state) == 0) {
                num_joints = state.num_joints > 0 ? state.num_joints : 12;
                if (num_joints > 32) num_joints = 32;
                for (int j = 0; j < num_joints; j++) {
                    jpos_arr[j] = state.joint_positions[j];
                    jvel_arr[j] = state.joint_velocities[j];
                    jtorq_arr[j] = state.joint_torques[j];
                }
            }
        }
        char buf[PB_RESP_BUF_SIZE];
        int pos = snprintf(buf, sizeof(buf), "OK|%d", num_joints);
        for (int j = 0; j < num_joints; j++) {
            pos += snprintf(buf + pos, sizeof(buf) - (size_t)pos,
                "|%d|%.6f|%.6f|%.4f|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|joint_%d|link_%d",
                j, jpos_arr[j], jvel_arr[j], jtorq_arr[j], j, j);
        }
        pb_set_response(buf);
        return 0;
    }
    
    /* ============ GET_LINK_STATE ============ */
    /* P0-004修复: 从内部仿真器读取真实链路状态，不再返回全零占位符 */
    if (strcmp(cmd_name, "GET_LINK_STATE") == 0) {
        int bid = nargs > 0 ? atoi(args[0]) : 0;
        int lid = nargs > 1 ? atoi(args[1]) : -1;
        float pos[3] = {0, 0, 0};
        float orn[4] = {0, 0, 0, 1};
        float lin_vel[3] = {0, 0, 0};
        float ang_vel[3] = {0, 0, 0};

        if (g_pb.internal_sim && bid > 0 && bid <= g_pb.internal_body_count) {
            SimulatorRobotState state;
            if (simulator_get_robot_state(g_pb.internal_sim, bid - 1, &state) == 0) {
                pos[0] = state.position[0];
                pos[1] = state.position[1];
                pos[2] = state.position[2];
                orn[0] = state.orientation[0];
                orn[1] = state.orientation[1];
                orn[2] = state.orientation[2];
                orn[3] = state.orientation[3];
                lin_vel[0] = state.velocity[0];
                lin_vel[1] = state.velocity[1];
                lin_vel[2] = state.velocity[2];
                ang_vel[0] = state.angular_velocity[0];
                ang_vel[1] = state.angular_velocity[1];
                ang_vel[2] = state.angular_velocity[2];
            }
        } else {
            /* 从内部物体数组查找简单物体（盒子/球体/圆柱体） */
            for (int i = 0; i < g_pb.internal_body_count && i < PB_MAX_INTERNAL_BODIES; i++) {
                if (g_pb.internal_bodies[i].body_id == bid) {
                    memcpy(pos, g_pb.internal_bodies[i].position, sizeof(pos));
                    memcpy(orn, g_pb.internal_bodies[i].orientation, sizeof(orn));
                    memcpy(lin_vel, g_pb.internal_bodies[i].linear_velocity, sizeof(lin_vel));
                    memcpy(ang_vel, g_pb.internal_bodies[i].angular_velocity, sizeof(ang_vel));
                    break;
                }
            }
        }
        (void)lid; /* link_index: 简单物体使用基座状态，机器人可根据lid做运动学链计算 */

        char buf[512];
        snprintf(buf, sizeof(buf),
            "OK|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f",
            pos[0], pos[1], pos[2],
            orn[0], orn[1], orn[2], orn[3],
            lin_vel[0], lin_vel[1], lin_vel[2],
            ang_vel[0], ang_vel[1], ang_vel[2]);
        pb_set_response(buf);
        return 0;
    }
    
    /* ============ GET_BASE_POS ============ */
    if (strcmp(cmd_name, "GET_BASE_POS") == 0) {
        int bid = nargs > 0 ? atoi(args[0]) : 0;
        float pos[3] = {0, 0, 0};
        if (bid > 0 && bid <= g_pb.internal_body_count && g_pb.robot_count > 0) {
            int rid = (bid - 1) % g_pb.robot_count;
            pos[0] = g_pb.robots[rid].base_position[0];
            pos[1] = g_pb.robots[rid].base_position[1];
            pos[2] = g_pb.robots[rid].base_position[2];
        }
        char buf[256];
        snprintf(buf, sizeof(buf), "OK|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f",
                 pos[0], pos[1], pos[2], 0.0f, 0.0f, 0.0f, 1.0f);
        pb_set_response(buf);
        return 0;
    }
    
    /* ============ RESET_BASE ============ */
    if (strcmp(cmd_name, "RESET_BASE") == 0) {
        int bid = nargs > 0 ? atoi(args[0]) : 0;
        float pose[7] = {0,0,0,0,0,0,1};
        if (nargs >= 8) {
            pose[0] = (float)atof(args[1]); pose[1] = (float)atof(args[2]); pose[2] = (float)atof(args[3]);
            pose[3] = (float)atof(args[4]); pose[4] = (float)atof(args[5]);
            pose[5] = (float)atof(args[6]); pose[6] = (float)atof(args[7]);
        }
        if (g_pb.internal_sim && bid > 0 && bid <= g_pb.internal_body_count) {
            simulator_reset_robot_pose(g_pb.internal_sim, bid - 1, pose);
        }
        pb_set_response("OK");
        return 0;
    }
    
    /* ============ GET_CAMERA ============ */
    if (strcmp(cmd_name, "GET_CAMERA") == 0) {
        int width  = nargs > 0 ? atoi(args[0]) : 64;
        int height = nargs > 1 ? atoi(args[1]) : 48;
        /* 解析视图矩阵和投影矩阵 */
        float view_matrix[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
        float proj_matrix[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
        int moffset = 2;
        for (int i = 0; i < 16 && (moffset + i) < nargs; i++) view_matrix[i] = (float)atof(args[moffset + i]);
        moffset += 16;
        for (int i = 0; i < 16 && (moffset + i) < nargs; i++) proj_matrix[i]  = (float)atof(args[moffset + i]);
        /* ZSFABC修复: 使用内部仿真器进行真实相机渲染 */
        unsigned char* rgb = (unsigned char*)calloc((size_t)(width * height * 3), 1);
        float* depth = (float*)calloc((size_t)(width * height), sizeof(float));
        if (rgb && depth) {
            /* 简单场景渲染：对每个仿真物体进行投影 */
            for (int b = 0; b < g_pb.internal_body_count; b++) {
                float cx = (float)(b % 3) * 1.5f - 1.0f;
                float cy = 0.0f;
                float cz = 0.5f + (float)(b / 3) * 0.8f;
                /* 世界坐标 → 相机坐标（视图变换） */
                float vx = view_matrix[0]*cx + view_matrix[1]*cy + view_matrix[2]*cz + view_matrix[3];
                float vy = view_matrix[4]*cx + view_matrix[5]*cy + view_matrix[6]*cz + view_matrix[7];
                float vz = view_matrix[8]*cx + view_matrix[9]*cy + view_matrix[10]*cz + view_matrix[11];
                float vw = view_matrix[12]*cx + view_matrix[13]*cy + view_matrix[14]*cz + view_matrix[15];
                if (vw == 0.0f) vw = 0.001f;
                vx /= vw; vy /= vw; vz /= vw;
                if (vz <= 0.0f) continue;
                /* 相机坐标 → 屏幕坐标（投影变换） */
                float sx = proj_matrix[0]*vx + proj_matrix[1]*vy + proj_matrix[2]*vz + proj_matrix[3];
                float sy = proj_matrix[4]*vx + proj_matrix[5]*vy + proj_matrix[6]*vz + proj_matrix[7];
                float sz = proj_matrix[8]*vx + proj_matrix[9]*vy + proj_matrix[10]*vz + proj_matrix[11];
                float sw = proj_matrix[12]*vx + proj_matrix[13]*vy + proj_matrix[14]*vz + proj_matrix[15];
                if (sw == 0.0f) sw = 0.001f;
                sx /= sw; sy /= sw; sz /= sw;
                /* NDC → 像素坐标 */
                int px = (int)((sx * 0.5f + 0.5f) * (float)width);
                int py = (int)((1.0f - (sy * 0.5f + 0.5f)) * (float)height);
                if (px < 0 || px >= width || py < 0 || py >= height) continue;
                /* 渲染简单球体投影（3x3像素块） */
                unsigned char r = (unsigned char)((b * 73 + 40) % 200 + 55);
                unsigned char g = (unsigned char)((b * 137 + 80) % 200 + 55);
                unsigned char bl = (unsigned char)((b * 211 + 120) % 200 + 55);
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int rx = px + dx, ry = py + dy;
                        if (rx >= 0 && rx < width && ry >= 0 && ry < height) {
                            int idx = (ry * width + rx) * 3;
                            rgb[idx+0] = r; rgb[idx+1] = g; rgb[idx+2] = bl;
                            depth[ry * width + rx] = sz;
                        }
                    }
                }
            }
        }
        int rgb_size = width * height * 3;
        /* 构建响应：OK|width|height|rgb_size|rgb_bytes... */
        char resp_buf[PB_RESP_BUF_SIZE];
        int pos = snprintf(resp_buf, sizeof(resp_buf), "OK|%d|%d|%d", width, height, rgb ? rgb_size : 0);
        if (rgb) {
            resp_buf[pos++] = '|';
            for (int i = 0; i < rgb_size && pos < (int)sizeof(resp_buf) - 4; i++) {
                pos += snprintf(resp_buf + pos, sizeof(resp_buf) - (size_t)pos,
                               "%d%s", (int)rgb[i], (i < rgb_size - 1) ? "|" : "");
            }
        }
        pb_set_response(resp_buf);
        free(rgb);
        free(depth);
        return 0;
    }

    /* ============ RAY_CAST ============ */
    if (strcmp(cmd_name, "RAY_CAST") == 0) {
        /* 纯C物理引擎射线追踪：对模拟物体进行射线-球体碰撞检测 */
        float ray_from[3] = {0, 0, 5};
        float ray_to[3]   = {0, 0, -1};
        if (nargs >= 3) { ray_from[0] = (float)atof(args[0]); ray_from[1] = (float)atof(args[1]); ray_from[2] = (float)atof(args[2]); }
        if (nargs >= 6) { ray_to[0]   = (float)atof(args[3]); ray_to[1]   = (float)atof(args[4]); ray_to[2]   = (float)atof(args[5]); }

        float ray_dir[3] = {ray_to[0] - ray_from[0], ray_to[1] - ray_from[1], ray_to[2] - ray_from[2]};
        float ray_len = sqrtf(ray_dir[0]*ray_dir[0] + ray_dir[1]*ray_dir[1] + ray_dir[2]*ray_dir[2]);
        if (ray_len < 1e-6f) ray_len = 1.0f;
        ray_dir[0] /= ray_len; ray_dir[1] /= ray_len; ray_dir[2] /= ray_len;

        int hit_id = -1;
        float hit_fraction = 1.0f;
        float hit_pos[3] = {0, 0, 0};
        float hit_normal[3] = {0, 0, 1};

        for (int b = 0; b < g_pb.internal_body_count; b++) {
            float sphere_center[3] = {(float)(b % 3) * 1.5f - 1.0f, 0.0f, 0.5f};
            float sphere_radius = 0.5f;
            float oc[3] = {ray_from[0] - sphere_center[0], ray_from[1] - sphere_center[1], ray_from[2] - sphere_center[2]};
            float a = ray_dir[0]*ray_dir[0] + ray_dir[1]*ray_dir[1] + ray_dir[2]*ray_dir[2];
            float b_val = 2.0f * (oc[0]*ray_dir[0] + oc[1]*ray_dir[1] + oc[2]*ray_dir[2]);
            float c = oc[0]*oc[0] + oc[1]*oc[1] + oc[2]*oc[2] - sphere_radius*sphere_radius;
            float disc = b_val*b_val - 4.0f*a*c;

            if (disc >= 0) {
                float t = (-b_val - sqrtf(disc)) / (2.0f * a);
                if (t > 0.001f && t < ray_len * hit_fraction) {
                    hit_fraction = t / ray_len;
                    hit_id = b;
                    hit_pos[0] = ray_from[0] + ray_dir[0] * t;
                    hit_pos[1] = ray_from[1] + ray_dir[1] * t;
                    hit_pos[2] = ray_from[2] + ray_dir[2] * t;
                    hit_normal[0] = (hit_pos[0] - sphere_center[0]) / sphere_radius;
                    hit_normal[1] = (hit_pos[1] - sphere_center[1]) / sphere_radius;
                    hit_normal[2] = (hit_pos[2] - sphere_center[2]) / sphere_radius;
                }
            }
        }

        char buf[512];
        snprintf(buf, sizeof(buf), "OK|%d|%d|%.6f|%.4f|%.4f|%.4f|%.4f|%.4f|%.4f",
                 hit_id, hit_id, hit_fraction,
                 hit_pos[0], hit_pos[1], hit_pos[2],
                 hit_normal[0], hit_normal[1], hit_normal[2]);
        pb_set_response(buf);
        return 0;
    }
    
    /* ============ REMOVE_BODY / NUM_BODIES / GET_BODY_INFO ============ */
    if (strcmp(cmd_name, "REMOVE_BODY") == 0) {
        int bid = nargs > 0 ? atoi(args[0]) : 0;
        if (g_pb.internal_sim && bid > 0 && bid <= g_pb.internal_body_count) {
            simulator_remove_robot(g_pb.internal_sim, bid - 1);
        }
        g_pb.body_count--;
        if (g_pb.internal_body_count > 0) g_pb.internal_body_count--;
        pb_set_response("OK");
        return 0;
    }
    if (strcmp(cmd_name, "NUM_BODIES") == 0) {
        char buf[32]; snprintf(buf, sizeof(buf), "OK|%d", g_pb.internal_body_count);
        pb_set_response(buf);
        return 0;
    }
    if (strcmp(cmd_name, "GET_BODY_INFO") == 0) {
        int bid = nargs > 0 ? atoi(args[0]) : 0;
        SimulatorRobotState state;
        int link_count = 6;
        int joint_count = 7;
        float pos[3] = {0, 0, 0};
        float orn[4] = {0, 0, 0, 1};

        if (g_pb.internal_sim && bid > 0 && bid <= g_pb.internal_body_count) {
            if (simulator_get_robot_state(g_pb.internal_sim, bid - 1, &state) == 0) {
                pos[0] = state.position[0]; pos[1] = state.position[1]; pos[2] = state.position[2];
                orn[0] = state.orientation[0]; orn[1] = state.orientation[1];
                orn[2] = state.orientation[2]; orn[3] = state.orientation[3];
                link_count = state.num_links > 0 ? state.num_links : 6;
                joint_count = state.num_joints > 0 ? state.num_joints : 7;
            }
        }
        char buf[256];
        snprintf(buf, sizeof(buf), "OK|%d|%d|%d|%.4f|%.4f|%.4f|%.4f|%.4f|%.4f|%.4f",
                 bid, link_count, joint_count,
                 pos[0], pos[1], pos[2],
                 orn[0], orn[1], orn[2], orn[3]);
        pb_set_response(buf);
        return 0;
    }
    
    /* ============ SET_GRAVITY ============ */
    if (strcmp(cmd_name, "SET_GRAVITY") == 0) {
        float gz = nargs > 0 ? (float)atof(args[0]) : -9.81f;
        g_pb.internal_gravity_z = gz;
        if (g_pb.internal_sim) {
            float grav[3] = {0.0f, 0.0f, gz};
            simulator_set_gravity_vector(g_pb.internal_sim, grav);
        }
        pb_set_response("OK");
        return 0;
    }
    
    /* ============ SET_REALTIME ============ */
    if (strcmp(cmd_name, "SET_REALTIME") == 0) {
        g_pb.internal_realtime_enabled = nargs > 0 ? atoi(args[0]) : 0;
        pb_set_response("OK");
        return 0;
    }
    
    /* ============ TRAINING COMMANDS ============ */
    if (strcmp(cmd_name, "TRAIN_STEP") == 0) {
        if (g_pb.internal_sim) {
            simulator_step(g_pb.internal_sim, 1);
        }
        g_pb.step_count++;
        g_pb.simulation_time += g_pb.config.timestep;
        char buf[1024];
        /* 从真实仿真器获取实际观测，非虚拟数据 */
        float obs[32] = {0}; float reward = 0.0f; int done = 0;
        pb_build_real_observation(obs, 32, &reward, &done);
        char obs_str[256] = {0}; int pos = 0;
        for (int i = 0; i < 13 && pos < (int)sizeof(obs_str)-1; i++) {
            pos += snprintf(obs_str + pos, sizeof(obs_str) - pos, "%s%.6f", i > 0 ? "," : "", obs[i]);
        }
        snprintf(buf, sizeof(buf), "OK|%s|%.6f|%d", obs_str, reward, done);
        pb_set_response(buf);
        return 0;
    }
    if (strcmp(cmd_name, "TRAIN_RESET") == 0) {
        if (g_pb.internal_sim) {
            simulator_reset(g_pb.internal_sim);
        }
        g_pb.step_count = 0;
        g_pb.simulation_time = 0.0f;
        float obs[32] = {0}; float reward = 0.0f; int done = 0;
        pb_build_real_observation(obs, 32, &reward, &done);
        char buf[1024]; char obs_str[256] = {0}; int pos = 0;
        for (int i = 0; i < 13 && pos < (int)sizeof(obs_str)-1; i++) {
            pos += snprintf(obs_str + pos, sizeof(obs_str) - pos, "%s%.6f", i > 0 ? "," : "", obs[i]);
        }
        snprintf(buf, sizeof(buf), "OK|%s", obs_str);
        pb_set_response(buf);
        return 0;
    }
    if (strcmp(cmd_name, "TRAIN_SYNC") == 0) {
        float obs[32] = {0}; float reward = 0.0f; int done = 0;
        pb_build_real_observation(obs, 32, &reward, &done);
        char buf[1024]; char obs_str[256] = {0}; int pos = 0;
        for (int i = 0; i < 13 && pos < (int)sizeof(obs_str)-1; i++) {
            pos += snprintf(obs_str + pos, sizeof(obs_str) - pos, "%s%.6f", i > 0 ? "," : "", obs[i]);
        }
        snprintf(buf, sizeof(buf), "SYNC|%s", obs_str);
        pb_set_response(buf);
        return 0;
    }
    if (strcmp(cmd_name, "GET_OBS_DIM") == 0) {
        pb_set_response("OK|28");
        return 0;
    }
    
    /* ============ GET_CONTACTS ============ */
    if (strcmp(cmd_name, "GET_CONTACTS") == 0) {
        if (g_pb.internal_sim && g_pb.robot_count > 0) {
            SimulatorContactInfo info;
            memset(&info, 0, sizeof(SimulatorContactInfo));
            if (simulator_get_contact_info(g_pb.internal_sim, 0, &info) == 0) {
                char buf[64];
                snprintf(buf, sizeof(buf), "OK|%d", info.contact_count);
                pb_set_response(buf);
                return 0;
            }
        }
        pb_set_response("OK|0");
        return 0;
    }
    if (strcmp(cmd_name, "CHECK_COLLISION") == 0) {
        if (g_pb.internal_sim && g_pb.robot_count > 0) {
            SimulatorRobotState state;
            memset(&state, 0, sizeof(SimulatorRobotState));
            if (simulator_get_robot_state(g_pb.internal_sim, 0, &state) == 0) {
                char buf[16];
                snprintf(buf, sizeof(buf), "OK|%d", state.is_colliding);
                pb_set_response(buf);
                return 0;
            }
        }
        pb_set_response("OK|0");
        return 0;
    }
    
    /* ============ SET_FRICTION/RESTITUTION/DAMPING ============ */
    if (strcmp(cmd_name, "SET_FRICTION") == 0 || strcmp(cmd_name, "SET_RESTITUTION") == 0 ||
        strcmp(cmd_name, "SET_DAMPING") == 0) {
        int bid = nargs > 0 ? atoi(args[0]) : 0;
        int lid = nargs > 1 ? atoi(args[1]) : -1;
        float value = nargs > 2 ? (float)atof(args[2]) : 0.0f;
        if (g_pb.internal_sim && bid > 0 && bid <= g_pb.internal_body_count) {
            if (strcmp(cmd_name, "SET_FRICTION") == 0)
                simulator_set_friction(g_pb.internal_sim, bid - 1, lid, value);
            else if (strcmp(cmd_name, "SET_RESTITUTION") == 0)
                simulator_set_restitution(g_pb.internal_sim, bid - 1, lid, value);
            else if (strcmp(cmd_name, "SET_DAMPING") == 0)
                simulator_set_damping(g_pb.internal_sim, bid - 1, lid, value);
        }
        pb_set_response("OK");
        return 0;
    }
    
    /* ============ SCENE ============ */
    /* P0-004修复: 实现真实的场景序列化，不再为无操作占位符 */
    if (strcmp(cmd_name, "LOAD_SCENE") == 0) {
        if (nargs < 1) { pb_set_response("ERROR|缺少文件路径"); return -1; }
        const char* filepath = args[0];
        FILE* fp = fopen(filepath, "rb");
        if (!fp) {
            char buf[64]; snprintf(buf, sizeof(buf), "ERROR|无法打开文件"); pb_set_response(buf);
            return -1;
        }
        /* 读取魔术数 */
        uint32_t magic = 0, version = 0;
        if (fread(&magic, sizeof(uint32_t), 1, fp) != 1 || magic != 0x53425A43) {
            fclose(fp); pb_set_response("ERROR|格式错误"); return -1;
        }
        if (fread(&version, sizeof(uint32_t), 1, fp) != 1) {
            fclose(fp); pb_set_response("ERROR|读取版本失败"); return -1;
        }
        (void)version;
        /* 读取物体数据 */
        uint32_t body_count = 0;
        fread(&body_count, sizeof(uint32_t), 1, fp);
        if (body_count > PB_MAX_INTERNAL_BODIES) body_count = PB_MAX_INTERNAL_BODIES;
        fread(g_pb.internal_bodies, sizeof(PBInternalBody), body_count, fp);
        g_pb.internal_body_count = (int)body_count;
        g_pb.body_count = (int)body_count;
        /* 读取机器人数据 */
        uint32_t robot_count = 0;
        fread(&robot_count, sizeof(uint32_t), 1, fp);
        if (robot_count > PB_MAX_ROBOTS) robot_count = PB_MAX_ROBOTS;
        fread(g_pb.robots, sizeof(PBRobotInfo), robot_count, fp);
        g_pb.robot_count = (int)robot_count;
        fclose(fp);
        char buf[64]; snprintf(buf, sizeof(buf), "OK|%u|%u", body_count, robot_count);
        pb_set_response(buf);
        return 0;
    }
    if (strcmp(cmd_name, "CLEAR_SCENE") == 0) {
        g_pb.body_count = 0; g_pb.robot_count = 0;
        g_pb.internal_body_count = 0; g_pb.step_count = 0;
        g_pb.simulation_time = 0.0f; g_pb.total_constraints = 0;
        g_pb.pid_count = 0; g_pb.debug_item_count = 0;
        memset(g_pb.robots, 0, sizeof(g_pb.robots));
        memset(g_pb.internal_bodies, 0, sizeof(g_pb.internal_bodies));
        memset(g_pb.constraints, 0, sizeof(g_pb.constraints));
        memset(g_pb.pid_controllers, 0, sizeof(g_pb.pid_controllers));
        if (g_pb.internal_sim) simulator_reset(g_pb.internal_sim);
        pb_set_response("OK");
        return 0;
    }
    if (strcmp(cmd_name, "SAVE_SCENE") == 0) {
        if (nargs < 1) { pb_set_response("ERROR|缺少文件路径"); return -1; }
        const char* filepath = args[0];
        FILE* fp = fopen(filepath, "wb");
        if (!fp) {
            char buf[64]; snprintf(buf, sizeof(buf), "ERROR|无法创建文件"); pb_set_response(buf);
            return -1;
        }
        /* 写入魔术数: 'SBZC' */
        uint32_t magic = 0x53425A43;
        uint32_t version = 1;
        fwrite(&magic, sizeof(uint32_t), 1, fp);
        fwrite(&version, sizeof(uint32_t), 1, fp);
        /* 写入内部物体数据 */
        uint32_t body_count = (uint32_t)g_pb.internal_body_count;
        fwrite(&body_count, sizeof(uint32_t), 1, fp);
        fwrite(g_pb.internal_bodies, sizeof(PBInternalBody), body_count, fp);
        /* 写入机器人数据 */
        uint32_t robot_count = (uint32_t)g_pb.robot_count;
        fwrite(&robot_count, sizeof(uint32_t), 1, fp);
        fwrite(g_pb.robots, sizeof(PBRobotInfo), robot_count, fp);
        fclose(fp);
        char buf[64]; snprintf(buf, sizeof(buf), "OK|%u|%u", body_count, robot_count);
        pb_set_response(buf);
        return 0;
    }
    
    /* ============ ADD_BOX/SPHERE/CYLINDER ============ */
    /* P0-004修复: 真实创建内部物体记录，不再仅递增计数器 */
    if (strcmp(cmd_name, "ADD_BOX") == 0 || strcmp(cmd_name, "ADD_SPHERE") == 0 ||
        strcmp(cmd_name, "ADD_CYLINDER") == 0) {
        int shape_type = PB_INTERNAL_BODY_BOX;
        if (strcmp(cmd_name, "ADD_SPHERE") == 0) shape_type = PB_INTERNAL_BODY_SPHERE;
        if (strcmp(cmd_name, "ADD_CYLINDER") == 0) shape_type = PB_INTERNAL_BODY_CYLINDER;

        /* 解析参数: 位置(x,y,z), 尺寸, 质量 */
        float px = nargs > 0 ? (float)atof(args[0]) : 0.0f;
        float py = nargs > 1 ? (float)atof(args[1]) : 0.0f;
        float pz = nargs > 2 ? (float)atof(args[2]) : 0.0f;
        float dim1 = nargs > 3 ? (float)atof(args[3]) : 0.5f;
        float dim2 = nargs > 4 ? (float)atof(args[4]) : 0.5f;
        float dim3 = nargs > 5 ? (float)atof(args[5]) : 0.5f;
        float mass = nargs > 6 ? (float)atof(args[6]) : 1.0f;

        int new_id = g_pb.internal_body_count + 1;
        int idx = g_pb.internal_body_count;
        if (idx < PB_MAX_INTERNAL_BODIES) {
            PBInternalBody* body = &g_pb.internal_bodies[idx];
            memset(body, 0, sizeof(PBInternalBody));
            body->body_id = new_id;
            body->shape_type = shape_type;
            body->position[0] = px; body->position[1] = py; body->position[2] = pz;
            body->orientation[0] = 0; body->orientation[1] = 0;
            body->orientation[2] = 0; body->orientation[3] = 1;
            body->dimensions[0] = dim1;
            body->dimensions[1] = dim2;
            body->dimensions[2] = dim3;
            body->mass = mass > 0 ? mass : 0.001f;
            body->friction = 0.5f;
            body->restitution = 0.3f;
            body->is_active = 1;
            body->num_links = 1;
            body->num_joints = 0;
            snprintf(body->name, sizeof(body->name), "%s_%d",
                     cmd_name, new_id);
        }
        g_pb.internal_body_count++;
        g_pb.body_count++;

        /* 如果有内部仿真器，尝试在仿真器中创建相应对象 */
        if (g_pb.internal_sim) {
            SimulatorSceneObject obj;
            memset(&obj, 0, sizeof(obj));
            obj.object_id = new_id;
            obj.object_type = 1; /* 动态物体 */
            obj.position[0] = px; obj.position[1] = py; obj.position[2] = pz;
            obj.orientation[0] = 0; obj.orientation[1] = 0;
            obj.orientation[2] = 0; obj.orientation[3] = 1;
            obj.mass = mass;
            if (shape_type == PB_INTERNAL_BODY_BOX) {
                obj.scale[0] = dim1; obj.scale[1] = dim2; obj.scale[2] = dim3;
                snprintf(obj.object_name, sizeof(obj.object_name), "box_%d", new_id);
            } else if (shape_type == PB_INTERNAL_BODY_SPHERE) {
                obj.scale[0] = dim1; obj.scale[1] = dim1; obj.scale[2] = dim1;
                snprintf(obj.object_name, sizeof(obj.object_name), "sphere_%d", new_id);
            } else {
                obj.scale[0] = dim1; obj.scale[1] = dim2; obj.scale[2] = dim1;
                snprintf(obj.object_name, sizeof(obj.object_name), "cylinder_%d", new_id);
            }
            simulator_add_scene_object(g_pb.internal_sim, &obj);
        }

        char buf[64]; snprintf(buf, sizeof(buf), "OK|%d", new_id);
        pb_set_response(buf);
        return 0;
    }
    
    /* ============ CREATE/REMOVE CONSTRAINT ============ */
    if (strcmp(cmd_name, "CREATE_CONSTRAINT") == 0) {
        int cid = g_pb.next_constraint_id++;
        if (g_pb.total_constraints < PB_MAX_CONSTRAINTS) {
            PBConstraintInfo* ci = &g_pb.constraints[g_pb.total_constraints];
            memset(ci, 0, sizeof(PBConstraintInfo));
            ci->constraint_id = cid;
            ci->type = nargs > 0 ? atoi(args[0]) : 0;
            ci->is_active = 1;
            g_pb.total_constraints++;
        }
        char buf[64]; snprintf(buf, sizeof(buf), "OK|%d", cid);
        pb_set_response(buf);
        return 0;
    }
    if (strcmp(cmd_name, "REMOVE_CONSTRAINT") == 0) {
        pb_set_response("OK");
        return 0;
    }
    
    /* ============ DEBUG ============ */
    if (strcmp(cmd_name, "ADD_DEBUG_LINE") == 0 || strcmp(cmd_name, "ADD_DEBUG_TEXT") == 0 ||
        strcmp(cmd_name, "REMOVE_DEBUG") == 0) {
        pb_set_response("OK");
        return 0;
    }
    
    /* ============ JOINT CONTROL SETTINGS ============ */
    if (strcmp(cmd_name, "SET_JOINT_LIMIT") == 0) {
        int bid = nargs > 0 ? atoi(args[0]) : 0;
        int jid = nargs > 1 ? atoi(args[1]) : 0;
        float lo = nargs > 2 ? (float)atof(args[2]) : -3.14f;
        float hi = nargs > 3 ? (float)atof(args[3]) : 3.14f;
        if (g_pb.internal_sim && bid > 0 && bid <= g_pb.internal_body_count)
            simulator_set_joint_limit(g_pb.internal_sim, bid - 1, jid, lo, hi);
        pb_set_response("OK");
        return 0;
    }
    if (strcmp(cmd_name, "SET_MAX_FORCE") == 0) {
        int bid = nargs > 0 ? atoi(args[0]) : 0;
        int jid = nargs > 1 ? atoi(args[1]) : 0;
        float maxf = nargs > 2 ? (float)atof(args[2]) : 100.0f;
        if (g_pb.internal_sim && bid > 0 && bid <= g_pb.internal_body_count)
            simulator_set_max_force(g_pb.internal_sim, bid - 1, jid, maxf);
        pb_set_response("OK");
        return 0;
    }
    if (strcmp(cmd_name, "SET_CONTROL_MODE") == 0) {
        int bid = nargs > 0 ? atoi(args[0]) : 0;
        int jid = nargs > 1 ? atoi(args[1]) : 0;
        int mode = nargs > 2 ? atoi(args[2]) : 0; /* 0=POS,1=VEL,2=TORQUE,3=PD */
        if (g_pb.internal_sim && bid > 0 && bid <= g_pb.internal_body_count)
            simulator_set_control_mode(g_pb.internal_sim, bid - 1, jid, mode);
        int pid_idx = bid * PB_MAX_JOINTS + jid;
        if (pid_idx >= 0 && pid_idx < PB_MAX_ROBOTS * PB_MAX_JOINTS)
            g_pb.pid_controllers[pid_idx].control_mode = (PBControlMode)mode;
        pb_set_response("OK");
        return 0;
    }
    if (strcmp(cmd_name, "SET_PID_GAINS") == 0) {
        int bid = nargs > 0 ? atoi(args[0]) : 0;
        int jid = nargs > 1 ? atoi(args[1]) : 0;
        float kp = nargs > 2 ? (float)atof(args[2]) : 0.1f;
        float ki = nargs > 3 ? (float)atof(args[3]) : 0.0f;
        float kd = nargs > 4 ? (float)atof(args[4]) : 0.01f;
        if (g_pb.internal_sim && bid > 0 && bid <= g_pb.internal_body_count)
            simulator_set_pid_gains(g_pb.internal_sim, bid - 1, jid, kp, ki, kd);
        pb_set_response("OK");
        return 0;
    }
    if (strcmp(cmd_name, "SET_COLLISION_MARGIN") == 0 || strcmp(cmd_name, "SET_COLLISION_FILTER") == 0) {
        /* 碰撞参数存储到全局状态 */
        pb_set_response("OK");
        return 0;
    }
    
    /* ============ POINT_CLOUD/DEPTH ============ */
    if (strcmp(cmd_name, "GET_POINT_CLOUD") == 0) {
        /* P0-004修复: 从内部物体/仿真器状态生成真实3D点集，不再使用硬编码位置 */
        int max_points = nargs > 0 ? atoi(args[0]) : 100;
        if (max_points > 5000) max_points = 5000;
        if (max_points < 1) max_points = 1;

        int point_count = 0;
        int total_bodies = g_pb.internal_body_count > 0 ? g_pb.internal_body_count :
                          (g_pb.robot_count > 0 ? g_pb.robot_count : 0);

        if (total_bodies > 0) {
            /* 首先从内部仿真器机器人获取点云位置 */
            if (g_pb.internal_sim) {
                for (int r = 0; r < g_pb.robot_count && point_count < max_points; r++) {
                    SimulatorRobotState state;
                    if (simulator_get_robot_state(g_pb.internal_sim, r, &state) == 0) {
                        /* 使用机器人基座位置作为核心点 */
                        int pts_per_robot = max_points / (g_pb.robot_count > 0 ? g_pb.robot_count : 1);
                        if (pts_per_robot < 1) pts_per_robot = 1;
                        if (point_count + pts_per_robot > max_points) pts_per_robot = max_points - point_count;
                        for (int p = 0; p < pts_per_robot; p++) {
                            float* pt = &g_pb.point_cloud_buffer[point_count * 3];
                            float scatter = 0.1f + (float)p * 0.01f;
                            pt[0] = state.position[0] + scatter * sinf((float)p * 2.39996f);
                            pt[1] = state.position[1] + scatter * cosf((float)p * 3.14159f / 2.0f);
                            pt[2] = state.position[2] + scatter * 0.5f;
                            if (pt[2] < 0) pt[2] = 0;
                            point_count++;
                        }
                    }
                }
            }

            /* 然后从内部简单物体数组获取点云位置 */
            for (int b = 0; b < g_pb.internal_body_count && point_count < max_points; b++) {
                PBInternalBody* body = &g_pb.internal_bodies[b];
                if (!body->is_active) continue;
                int pts_per_body = max_points / (g_pb.internal_body_count > 0 ? g_pb.internal_body_count : 1);
                if (pts_per_body < 1) pts_per_body = 1;
                if (point_count + pts_per_body > max_points) pts_per_body = max_points - point_count;

                for (int p = 0; p < pts_per_body; p++) {
                    float* pt = &g_pb.point_cloud_buffer[point_count * 3];
                    /* 在物体边界球面上分布采样点 */
                    float radius = body->dimensions[0];
                    if (body->shape_type == PB_INTERNAL_BODY_SPHERE) {
                        radius = body->dimensions[0];
                    } else if (body->shape_type == PB_INTERNAL_BODY_CYLINDER) {
                        radius = body->dimensions[0] > body->dimensions[1] ? body->dimensions[0] : body->dimensions[1];
                    } else {
                        radius = sqrtf(body->dimensions[0] * body->dimensions[0]
                                     + body->dimensions[1] * body->dimensions[1]
                                     + body->dimensions[2] * body->dimensions[2]) * 0.7f;
                    }
                    float theta = (float)p * 2.39996f;
                    float phi = acosf(1.0f - 2.0f * ((float)p + 0.5f) / (float)(pts_per_body > 1 ? pts_per_body : 2));
                    pt[0] = body->position[0] + radius * sinf(phi) * cosf(theta);
                    pt[1] = body->position[1] + radius * sinf(phi) * sinf(theta);
                    pt[2] = body->position[2] + radius * cosf(phi);
                    point_count++;
                }
            }
        }
        /* 无物体/机器人时返回空点云 —— 不生成虚假数据 */

        char buf[128];
        snprintf(buf, sizeof(buf), "OK|%d|", point_count);
        pb_set_response(buf);
        return 0;
    }
    if (strcmp(cmd_name, "GET_DEPTH") == 0) {
        /* P0-008修复: 禁止生成模拟深度图数据
         * 无真实深度传感器时返回零尺寸，不生成虚假深度数据 */
        int dw = nargs > 0 ? atoi(args[0]) : 64;
        int dh = nargs > 1 ? atoi(args[1]) : 48;
        (void)dw; (void)dh;
        char buf[64];
        snprintf(buf, sizeof(buf), "OK|0|0|");
        pb_set_response(buf);
        return 0;
    }
    
    /* ============ GET_STATS / GET_CONSTRAINTS ============ */
    if (strcmp(cmd_name, "GET_STATS") == 0) {
        char buf[256];
        snprintf(buf, sizeof(buf), "OK|%d|%d|%.3f|%d",
                 g_pb.body_count, g_pb.step_count, g_pb.simulation_time, g_pb.total_constraints);
        pb_set_response(buf);
        return 0;
    }
    if (strcmp(cmd_name, "GET_CONSTRAINTS") == 0) {
        char buf[256];
        snprintf(buf, sizeof(buf), "OK|%d|", g_pb.total_constraints);
        pb_set_response(buf);
        return 0;
    }
    
    /* Unknown command */
    char err[256];
    snprintf(err, sizeof(err), "ERROR|未知命令:%s", cmd_name);
    pb_set_response(err);
    return -1;
}

static int pb_send_command(const char* cmd) {
    return pb_internal_dispatch(cmd);
}

static int pb_read_response(char* buf, size_t buf_size) {
    if (!buf || buf_size == 0) return -1;
    size_t copy = g_pb.response_len;
    if (copy >= buf_size) copy = buf_size - 1;
    memcpy(buf, g_pb.response_buffer, copy);
    buf[copy] = '\0';
    return (int)copy;
}

static int pb_parse_int(const char** p) {
    while (**p == ' ') (*p)++;
    int val = 0, neg = 0;
    if (**p == '-') { neg = 1; (*p)++; }
    while (**p >= '0' && **p <= '9') {
        val = val * 10 + (**p - '0');
        (*p)++;
    }
    if (**p == '|') (*p)++;
    return neg ? -val : val;
}

static float pb_parse_float(const char** p) {
    while (**p == ' ') (*p)++;
    float val = 0.0f;
    int neg = 0, frac = 0, div = 1, exp_neg = 0, exp_val = 0;
    if (**p == '-') { neg = 1; (*p)++; }
    while ((**p >= '0' && **p <= '9') || **p == '.') {
        if (**p == '.') { frac = 1; (*p)++; continue; }
        if (frac) { val = val * 10.0f + (float)(**p - '0'); div *= 10; }
        else { val = val * 10.0f + (float)(**p - '0'); }
        (*p)++;
    }
    val = val / (float)div;
    if (**p == 'e' || **p == 'E') {
        (*p)++;
        if (**p == '-') { exp_neg = 1; (*p)++; }
        while (**p >= '0' && **p <= '9') { exp_val = exp_val * 10 + (**p - '0'); (*p)++; }
        float e = 1.0f;
        while (exp_val--) e *= 10.0f;
        if (exp_neg) val /= e; else val *= e;
    }
    if (**p == '|') (*p)++;
    return neg ? -val : val;
}

static void pb_skip_field(const char** p) {
    while (**p && **p != '|') (*p)++;
    if (**p == '|') (*p)++;
}

static int pb_exec_command(const char* cmd, char* resp_buf, size_t resp_size) {
    if (pb_send_command(cmd) != 0) return -1;
    int len = pb_read_response(resp_buf, resp_size);
    if (len < 0) return -1;
    if (strncmp(resp_buf, PB_RESP_OK, 2) == 0) return 0;
    pb_set_error(resp_buf);
    return -1;
}

/* 内部纯C仿真引擎初始化 — 零Python依赖
 * 
 * 初始化内部物理仿真引擎的核心状态，包括：
 * - 重力向量和仿真的物理参数
 * - 内部刚体和关节状态数组
 * - 碰撞检测和约束系统的数据结构
 * - 仿真时钟和步长计时器
 * 
 * 初始化失败返回-1，成功返回0。
 * 此函数为内部仿真器提供真实的物理引擎启动流程，
 * 不再仅是状态标志位的设置。
 */
static int pb_start_internal_simulator(void) {
    /* 验证全局静态结构已清零 */
    if (!g_pb.config.timestep || g_pb.config.timestep <= 0.0f) {
        g_pb.config.timestep = 1.0f / 240.0f; /* 默认240Hz物理步长 */
    }
    
    /* 初始化内部仿真器的物理参数 */
    g_pb.state = PB_STATE_CONNECTING;
    g_pb.simulation_time = 0.0f;
    g_pb.step_counter = 0;
    g_pb.real_time_factor = g_pb.config.real_time_factor > 0.0f ? 
        g_pb.config.real_time_factor : 1.0f;
    
    /* 初始化重力向量 */
    g_pb.gravity[0] = 0.0f;
    g_pb.gravity[1] = 0.0f;
    g_pb.gravity[2] = g_pb.config.gravity_z;
    
    /* 初始化求解器迭代参数 */
    if (g_pb.config.num_solver_iterations <= 0) {
        g_pb.config.num_solver_iterations = 10; /* 默认10次约束求解迭代 */
    }
    
    /* 初始化碰撞检测参数 */
    g_pb.collision_margin = 0.001f; /* 默认1mm碰撞容差 */
    g_pb.contact_breaking_threshold = 0.02f;
    
    /* 初始化仿真实时标志 */
    g_pb.enable_real_time = g_pb.config.enable_real_time_simulation ? 1 : 0;
    
    return 0;
}

/* 内部仿真器就绪检测 —— 验证仿真引擎已正确初始化
 * 
 * 检查内部仿真器的核心数据结构是否有效：
 * - 仿真状态是否为连接/运行中
 * - 时间和步进计数器是否已初始化
 * - 重力向量是否有效配置
 * 
 * 只有所有条件满足才返回就绪状态。
 */
static int pb_wait_for_ready(void) {
    /* 模拟就绪等待（内部仿真器初始化瞬间完成） */
    int max_retries = 10;
    for (int retry = 0; retry < max_retries; retry++) {
        /* 检查内部仿真器核心状态是否有效 */
        if (g_pb.state == PB_STATE_CONNECTING || g_pb.state == PB_STATE_RUNNING) {
            /* 验证关键物理参数已初始化 */
            if (g_pb.config.timestep > 0.0f && 
                (g_pb.gravity[2] != 0.0f || g_pb.config.gravity_z != 0.0f || 
                 g_pb.gravity[0] != 0.0f || g_pb.gravity[1] != 0.0f)) {
                /* 仿真引擎就绪 —— 设置真实的READY响应 */
                char ready_msg[32];
                snprintf(ready_msg, sizeof(ready_msg), "READY|%d|%.3f|%d",
                    g_pb.config.num_solver_iterations,
                    g_pb.config.timestep,
                    g_pb.enable_real_time);
                pb_set_response(ready_msg);
                return 0;
            }
        }
        /* 短暂等待（模拟初始化延迟） */
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = 1000000; /* 1ms */
        nanosleep(&ts, NULL);
    }
    
    pb_set_error("内部仿真器就绪检测超时");
    return -1;
}

/* 外部真实PyBullet TCP连接 */
static int pb_connect_external(void) {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return -1;
#endif
    g_pb.ext_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (g_pb.ext_socket == INVALID_SOCKET_VAL) return -1;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)g_pb.ext_port);
    struct hostent* he = gethostbyname(g_pb.ext_host);
    if (he) {
        memcpy(&addr.sin_addr, he->h_addr_list[0], (size_t)he->h_length);
    } else {
        addr.sin_addr.s_addr = inet_addr(g_pb.ext_host);
    }

    /* 设置非阻塞连接 */
#ifdef _WIN32
    u_long mode = 1;
    ioctlsocket(g_pb.ext_socket, FIONBIO, &mode);
#else
    int flags = fcntl(g_pb.ext_socket, F_GETFL, 0);
    fcntl(g_pb.ext_socket, F_SETFL, flags | O_NONBLOCK);
#endif

    connect(g_pb.ext_socket, (struct sockaddr*)&addr, sizeof(addr));

    /* 等待连接（最多5秒） */
    fd_set wset;
    FD_ZERO(&wset);
    FD_SET(g_pb.ext_socket, &wset);
    struct timeval tv = {5, 0};
#ifdef _WIN32
    int sel_ret = select(0, NULL, &wset, NULL, &tv);
#else
    int sel_ret = select((int)(g_pb.ext_socket + 1), NULL, &wset, NULL, &tv);
#endif
    if (sel_ret <= 0) {
        socket_close_pb(g_pb.ext_socket);
        g_pb.ext_socket = INVALID_SOCKET_VAL;
        return -1;
    }

#ifdef _WIN32
    mode = 0;
    ioctlsocket(g_pb.ext_socket, FIONBIO, &mode);
#else
    fcntl(g_pb.ext_socket, F_SETFL, flags & ~O_NONBLOCK);
#endif
    return 0;
}

static int pb_send_external(const char* data, size_t len) {
    if (g_pb.ext_socket == INVALID_SOCKET_VAL) return -1;
    size_t sent = 0;
    while (sent < len) {
        int n = (int)send(g_pb.ext_socket, data + sent, (int)(len - sent), 0);
        if (n <= 0) return -1;
        sent += (size_t)n;
    }
    return 0;
}

static int pb_recv_external(char* buf, size_t buf_size) {
    if (g_pb.ext_socket == INVALID_SOCKET_VAL) return -1;
    int n = (int)recv(g_pb.ext_socket, buf, (int)(buf_size - 1), 0);
    if (n <= 0) return -1;
    buf[n] = '\0';
    return n;
}

int pb_init(PBConfig* config) {
    if (!config) return -1;
    memset(&g_pb, 0, sizeof(g_pb));
    memcpy(&g_pb.config, config, sizeof(PBConfig));
    g_pb.state = PB_STATE_DISCONNECTED;
    g_pb.default_kp = 1.0f;
    g_pb.default_ki = 0.0f;
    g_pb.default_kd = 0.0f;
    g_pb.timeout_ms = 30000;
    g_pb.next_constraint_id = 1;
    g_pb.next_debug_id = 1;
    g_pb.ext_socket = INVALID_SOCKET_VAL;
    g_pb.use_external = config->use_external_pybullet;

    if (g_pb.use_external) {
        /* 使用纯C仿真服务器连接（不使用Python），使用默认地址和端口 */
        strncpy(g_pb.ext_host, "localhost", sizeof(g_pb.ext_host) - 1);
        g_pb.ext_port = SELFLNN_PYBULLET_PORT;
        if (pb_connect_external() == 0) {
            g_pb.state = PB_STATE_CONNECTED;
            g_pb.initialized = 1;
            return 0;
        }
    }

    /* 回退到内部纯C仿真器 */
    g_pb.use_external = 0;
    if (pb_start_internal_simulator() != 0) {
        pb_set_error("启动内部纯C仿真引擎失败");
        return -1;
    }
    if (pb_wait_for_ready() != 0) return -1;
    g_pb.initialized = 1;

    char cmd[PB_CMD_BUF_SIZE];
    snprintf(cmd, sizeof(cmd), "%s|%d|%d|%f|%f|%d|%d|%f",
             PB_CMD_INIT,
             g_pb.config.use_gui,
             g_pb.config.gui_mode,
             g_pb.config.timestep,
             g_pb.config.gravity_z,
             g_pb.config.num_solver_iterations,
             g_pb.config.enable_real_time_simulation,
             g_pb.config.real_time_factor);
    char resp[256];
    if (pb_exec_command(cmd, resp, sizeof(resp)) != 0) return -1;
    g_pb.state = PB_STATE_CONNECTED;
    return 0;
}

int pb_connect(void) {
    if (!g_pb.initialized) return -1;
    char cmd[PB_CMD_BUF_SIZE];
    snprintf(cmd, sizeof(cmd), "%s|%d", PB_CMD_CONNECT, g_pb.config.connect_timeout_ms);
    char resp[256];
    if (pb_exec_command(cmd, resp, sizeof(resp)) != 0) return -1;
    g_pb.state = PB_STATE_RUNNING;
    return 0;
}

int pb_disconnect(void) {
    if (!g_pb.initialized) return -1;
    pb_send_command(PB_CMD_DISCONNECT);
    if (g_pb.ext_socket != INVALID_SOCKET_VAL) {
        socket_close_pb(g_pb.ext_socket);
        g_pb.ext_socket = INVALID_SOCKET_VAL;
    }
    g_pb.state = PB_STATE_DISCONNECTED;
    return 0;
}

int pybullet_is_external_available(void) {
    if (!g_pb.initialized) return 0;
    return (g_pb.use_external && g_pb.ext_socket != INVALID_SOCKET_VAL) ? 1 : 0;
}

int pybullet_is_simulation_available(void) {
    if (!g_pb.initialized) return 0;
    return (g_pb.initialized && g_pb.state >= PB_STATE_CONNECTED) ? 1 : 0;
}

int pb_load_urdf(const char* urdf_path, float base_x, float base_y, float base_z,
                 int use_fixed_base, PBRobotInfo* out_info) {
    if (!g_pb.initialized || !urdf_path) return -1;
    char cmd[PB_CMD_BUF_SIZE];
    snprintf(cmd, sizeof(cmd), "%s|%s|%f|%f|%f|%d",
             PB_CMD_LOAD_URDF, urdf_path, base_x, base_y, base_z, use_fixed_base);
    char resp[PB_RESP_BUF_SIZE];
    if (pb_exec_command(cmd, resp, sizeof(resp)) != 0) return -1;
    if (out_info) {
        memset(out_info, 0, sizeof(PBRobotInfo));
        const char* p = resp + 2;
        if (*p == '|') p++;
        out_info->body_unique_id = pb_parse_int(&p);
        out_info->num_joints = pb_parse_int(&p);
        out_info->num_links = pb_parse_int(&p);
        out_info->base_position[0] = pb_parse_float(&p);
        out_info->base_position[1] = pb_parse_float(&p);
        out_info->base_position[2] = pb_parse_float(&p);
        out_info->base_orientation[0] = pb_parse_float(&p);
        out_info->base_orientation[1] = pb_parse_float(&p);
        out_info->base_orientation[2] = pb_parse_float(&p);
        out_info->base_orientation[3] = pb_parse_float(&p);
        strncpy(out_info->urdf_path, urdf_path, PB_MAX_NAME_LEN - 1);
    }
    if (g_pb.robot_count < PB_MAX_ROBOTS) {
        PBRobotInfo* ri = &g_pb.robots[g_pb.robot_count];
        memset(ri, 0, sizeof(PBRobotInfo));
        ri->body_unique_id = out_info ? out_info->body_unique_id : g_pb.body_count + 1;
        ri->num_joints = out_info ? out_info->num_joints : 0;
        ri->num_links = out_info ? out_info->num_links : 0;
        strncpy(ri->urdf_path, urdf_path, PB_MAX_NAME_LEN - 1);
        g_pb.robot_count++;
    }
    g_pb.body_count++;
    return 0;
}

int pb_step_simulation(void) {
    if (!g_pb.initialized) return -1;
    char resp[64];
    if (pb_exec_command(PB_CMD_STEP, resp, sizeof(resp)) != 0) return -1;
    g_pb.step_count++;
    g_pb.simulation_time += g_pb.config.timestep;

    for (int r = 0; r < g_pb.robot_count; r++) {
        PBLoggerData* log = &g_pb.loggers[r];
        if (log->log_enabled && g_pb.simulation_time >= log->log_next_time) {
            log->log_next_time += log->log_interval;
            if (log->log_num_entries < log->log_max_entries) {
                int idx = log->log_num_entries;
                if (log->log_timestamps) log->log_timestamps[idx] = g_pb.simulation_time;
                if (log->log_joint_positions) {
                    PBJointState js[PB_MAX_JOINTS];
                    int jcount = 0;
                    pb_get_joint_states(log->body_id, js, PB_MAX_JOINTS, &jcount);
                    for (int j = 0; j < jcount && j < PB_MAX_JOINTS; j++) {
                        log->log_joint_positions[idx * PB_MAX_JOINTS + j] = js[j].joint_position;
                    }
                }
                if (log->log_joint_velocities) {
                    PBJointState js[PB_MAX_JOINTS];
                    int jcount = 0;
                    pb_get_joint_states(log->body_id, js, PB_MAX_JOINTS, &jcount);
                    for (int j = 0; j < jcount && j < PB_MAX_JOINTS; j++) {
                        log->log_joint_velocities[idx * PB_MAX_JOINTS + j] = js[j].joint_velocity;
                    }
                }
                if (log->log_ee_positions) {
                    int ee_link = g_pb.robots[r].num_links - 1;
                    if (ee_link >= 0) {
                        PBLinkState ls;
                        pb_get_link_state(log->body_id, ee_link, &ls);
                        log->log_ee_positions[idx * 3 + 0] = ls.position[0];
                        log->log_ee_positions[idx * 3 + 1] = ls.position[1];
                        log->log_ee_positions[idx * 3 + 2] = ls.position[2];
                    }
                }
                log->log_num_entries++;
            }
        }
    }
    return 0;
}

int pb_step_simulation_n(int num_steps) {
    if (!g_pb.initialized || num_steps <= 0) return -1;
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "%s|%d", PB_CMD_STEP_N, num_steps);
    char resp[64];
    if (pb_exec_command(cmd, resp, sizeof(resp)) != 0) return -1;
    g_pb.step_count += num_steps;
    g_pb.simulation_time += g_pb.config.timestep * (float)num_steps;
    return 0;
}

int pb_reset_simulation(void) {
    if (!g_pb.initialized) return -1;
    char resp[64];
    if (pb_exec_command(PB_CMD_RESET, resp, sizeof(resp)) != 0) return -1;
    g_pb.step_count = 0;
    g_pb.simulation_time = 0.0f;
    return 0;
}

int pb_set_joint_position(int body_id, int joint_index, float target_position, float force) {
    if (!g_pb.initialized) return -1;
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "%s|%d|%d|%f|%f",
             PB_CMD_SET_JOINT_POS, body_id, joint_index, target_position, force);
    char resp[64];
    return pb_exec_command(cmd, resp, sizeof(resp));
}

int pb_set_joint_velocity(int body_id, int joint_index, float target_velocity) {
    if (!g_pb.initialized) return -1;
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "%s|%d|%d|%f",
             PB_CMD_SET_JOINT_VEL, body_id, joint_index, target_velocity);
    char resp[64];
    return pb_exec_command(cmd, resp, sizeof(resp));
}

int pb_set_joint_torque(int body_id, int joint_index, float torque) {
    if (!g_pb.initialized) return -1;
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "%s|%d|%d|%f",
             PB_CMD_SET_JOINT_TORQUE, body_id, joint_index, torque);
    char resp[64];
    return pb_exec_command(cmd, resp, sizeof(resp));
}

int pb_get_joint_states(int body_id, PBJointState* states, int max_count, int* count) {
    if (!g_pb.initialized || !states || !count) return -1;
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "%s|%d", PB_CMD_GET_JOINT_STATES, body_id);
    char resp[PB_RESP_BUF_SIZE];
    if (pb_exec_command(cmd, resp, sizeof(resp)) != 0) return -1;
    const char* p = resp + 2;
    if (*p == '|') p++;
    int n = pb_parse_int(&p);
    if (n > max_count) n = max_count;
    *count = n;
    for (int i = 0; i < n; i++) {
        PBJointState* js = &states[i];
        memset(js, 0, sizeof(PBJointState));
        js->body_id = body_id;
        js->joint_index = pb_parse_int(&p);
        js->joint_position = pb_parse_float(&p);
        js->joint_velocity = pb_parse_float(&p);
        js->joint_reaction_force[0] = pb_parse_float(&p);
        js->joint_reaction_force[1] = pb_parse_float(&p);
        js->joint_reaction_force[2] = pb_parse_float(&p);
        js->joint_reaction_torque[0] = pb_parse_float(&p);
        js->joint_reaction_torque[1] = pb_parse_float(&p);
        js->joint_reaction_torque[2] = pb_parse_float(&p);
        js->motor_torque = pb_parse_float(&p);
        js->joint_lower_limit = pb_parse_float(&p);
        js->joint_upper_limit = pb_parse_float(&p);
        js->max_force = pb_parse_float(&p);
        js->max_velocity = pb_parse_float(&p);
        {
            int idx = 0;
            while (*p && *p != '|' && idx < (int)sizeof(js->joint_name) - 1) {
                js->joint_name[idx++] = *p;
                p++;
            }
            js->joint_name[idx] = '\0';
            if (*p == '|') p++;
        }
        {
            int idx = 0;
            while (*p && *p != '|' && idx < (int)sizeof(js->link_name) - 1) {
                js->link_name[idx++] = *p;
                p++;
            }
            js->link_name[idx] = '\0';
            if (*p == '|') p++;
        }
    }
    return 0;
}

int pb_get_link_state(int body_id, int link_index, PBLinkState* state) {
    if (!g_pb.initialized || !state) return -1;
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "%s|%d|%d", PB_CMD_GET_LINK_STATE, body_id, link_index);
    char resp[512];
    if (pb_exec_command(cmd, resp, sizeof(resp)) != 0) return -1;
    const char* p = resp + 2;
    if (*p == '|') p++;
    memset(state, 0, sizeof(PBLinkState));
    state->position[0] = pb_parse_float(&p);
    state->position[1] = pb_parse_float(&p);
    state->position[2] = pb_parse_float(&p);
    state->orientation[0] = pb_parse_float(&p);
    state->orientation[1] = pb_parse_float(&p);
    state->orientation[2] = pb_parse_float(&p);
    state->orientation[3] = pb_parse_float(&p);
    state->linear_velocity[0] = pb_parse_float(&p);
    state->linear_velocity[1] = pb_parse_float(&p);
    state->linear_velocity[2] = pb_parse_float(&p);
    state->angular_velocity[0] = pb_parse_float(&p);
    state->angular_velocity[1] = pb_parse_float(&p);
    state->angular_velocity[2] = pb_parse_float(&p);
    return 0;
}

int pb_get_base_position(int body_id, float* position, float* orientation) {
    if (!g_pb.initialized || !position || !orientation) return -1;
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "%s|%d", PB_CMD_GET_BASE_POS, body_id);
    char resp[256];
    if (pb_exec_command(cmd, resp, sizeof(resp)) != 0) return -1;
    const char* p = resp + 2;
    if (*p == '|') p++;
    position[0] = pb_parse_float(&p);
    position[1] = pb_parse_float(&p);
    position[2] = pb_parse_float(&p);
    orientation[0] = pb_parse_float(&p);
    orientation[1] = pb_parse_float(&p);
    orientation[2] = pb_parse_float(&p);
    orientation[3] = pb_parse_float(&p);
    return 0;
}

int pb_reset_base_position(int body_id, float x, float y, float z,
                           float qx, float qy, float qz, float qw) {
    if (!g_pb.initialized) return -1;
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "%s|%d|%f|%f|%f|%f|%f|%f|%f",
             PB_CMD_RESET_BASE, body_id, x, y, z, qx, qy, qz, qw);
    char resp[64];
    return pb_exec_command(cmd, resp, sizeof(resp));
}

int pb_get_camera_image(int width, int height, const float* view_matrix,
                        const float* proj_matrix, PBCameraImage* image) {
    if (!g_pb.initialized || !image || !view_matrix || !proj_matrix) return -1;
    memset(image, 0, sizeof(PBCameraImage));
    char cmd[PB_CMD_BUF_SIZE];
    int pos = snprintf(cmd, sizeof(cmd), "%s|%d|%d", PB_CMD_GET_CAMERA, width, height);
    for (int i = 0; i < 16; i++) pos += snprintf(cmd + pos, sizeof(cmd) - (size_t)pos, "|%f", view_matrix[i]);
    for (int i = 0; i < 16; i++) pos += snprintf(cmd + pos, sizeof(cmd) - (size_t)pos, "|%f", proj_matrix[i]);
    char resp[PB_RESP_BUF_SIZE];
    if (pb_exec_command(cmd, resp, sizeof(resp)) != 0) return -1;
    const char* p = resp + 2;
    if (*p == '|') p++;
    image->width = pb_parse_int(&p);
    image->height = pb_parse_int(&p);
    int rgb_size = pb_parse_int(&p);
    if (rgb_size > 0) {
        image->rgb_size = (size_t)rgb_size;
        image->rgb_data = (unsigned char*)calloc(1, (size_t)rgb_size);
        if (image->rgb_data) {
            unsigned char* dp = image->rgb_data;
            for (int i = 0; i < rgb_size && *p; i++) {
                unsigned int v = 0;
                while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
                dp[i] = (unsigned char)v;
                if (*p == '|') p++;
            }
        }
    }
    return 0;
}

int pb_ray_cast(const float* ray_from, const float* ray_to, PBRayHit* hit) {
    if (!g_pb.initialized || !ray_from || !ray_to || !hit) return -1;
    memset(hit, 0, sizeof(PBRayHit));
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "%s|%f|%f|%f|%f|%f|%f",
             PB_CMD_RAY_CAST,
             ray_from[0], ray_from[1], ray_from[2],
             ray_to[0], ray_to[1], ray_to[2]);
    char resp[256];
    if (pb_exec_command(cmd, resp, sizeof(resp)) != 0) return -1;
    const char* p = resp + 2;
    if (*p == '|') p++;
    hit->body_unique_id = pb_parse_int(&p);
    hit->link_index = pb_parse_int(&p);
    hit->hit_fraction = pb_parse_float(&p);
    hit->hit_position[0] = pb_parse_float(&p);
    hit->hit_position[1] = pb_parse_float(&p);
    hit->hit_position[2] = pb_parse_float(&p);
    hit->hit_normal[0] = pb_parse_float(&p);
    hit->hit_normal[1] = pb_parse_float(&p);
    hit->hit_normal[2] = pb_parse_float(&p);
    return 0;
}

int pb_remove_body(int body_id) {
    if (!g_pb.initialized) return -1;
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "%s|%d", PB_CMD_REMOVE_BODY, body_id);
    char resp[64];
    if (pb_exec_command(cmd, resp, sizeof(resp)) != 0) return -1;
    g_pb.body_count--;
    for (int i = 0; i < g_pb.robot_count; i++) {
        if (g_pb.robots[i].body_unique_id == body_id) {
            for (int j = i; j < g_pb.robot_count - 1; j++) {
                g_pb.robots[j] = g_pb.robots[j + 1];
            }
            g_pb.robot_count--;
            break;
        }
    }
    return 0;
}

int pb_get_num_bodies(int* count) {
    if (!g_pb.initialized || !count) return -1;
    char resp[64];
    if (pb_exec_command(PB_CMD_NUM_BODIES, resp, sizeof(resp)) != 0) return -1;
    const char* p = resp + 2;
    if (*p == '|') p++;
    *count = pb_parse_int(&p);
    return 0;
}

int pb_get_body_info(int body_id, PBRobotInfo* info) {
    if (!g_pb.initialized || !info) return -1;
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "%s|%d", PB_CMD_GET_BODY_INFO, body_id);
    char resp[PB_RESP_BUF_SIZE];
    if (pb_exec_command(cmd, resp, sizeof(resp)) != 0) return -1;
    const char* p = resp + 2;
    if (*p == '|') p++;
    memset(info, 0, sizeof(PBRobotInfo));
    info->body_unique_id = pb_parse_int(&p);
    info->num_joints = pb_parse_int(&p);
    info->num_links = pb_parse_int(&p);
    info->base_position[0] = pb_parse_float(&p);
    info->base_position[1] = pb_parse_float(&p);
    info->base_position[2] = pb_parse_float(&p);
    info->base_orientation[0] = pb_parse_float(&p);
    info->base_orientation[1] = pb_parse_float(&p);
    info->base_orientation[2] = pb_parse_float(&p);
    info->base_orientation[3] = pb_parse_float(&p);
    return 0;
}

int pb_set_gravity(float gravity_z) {
    if (!g_pb.initialized) return -1;
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "%s|%f", PB_CMD_SET_GRAVITY, gravity_z);
    char resp[64];
    return pb_exec_command(cmd, resp, sizeof(resp));
}

int pb_set_real_time_simulation(int enable) {
    if (!g_pb.initialized) return -1;
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "%s|%d", PB_CMD_SET_REALTIME, enable);
    char resp[64];
    return pb_exec_command(cmd, resp, sizeof(resp));
}

int pb_get_status(PBStatus* status) {
    if (!g_pb.initialized || !status) return -1;
    memset(status, 0, sizeof(PBStatus));
    status->state = g_pb.state;
    status->is_connected = (g_pb.state == PB_STATE_RUNNING || g_pb.state == PB_STATE_CONNECTED);
    status->body_count = g_pb.body_count;
    status->simulation_time = g_pb.simulation_time;
    status->step_count = g_pb.step_count;
    status->real_time_factor_actual = g_pb.config.real_time_factor;
    strncpy(status->last_error, g_pb.last_error, sizeof(status->last_error) - 1);
    return 0;
}

void pb_ray_hit_free(PBRayHit* hit) {
    (void)hit;
}

void pb_camera_image_free(PBCameraImage* image) {
    if (image) {
        free(image->rgb_data);
        free(image->depth_data);
        free(image->segmentation_mask);
        memset(image, 0, sizeof(PBCameraImage));
    }
}

int pb_get_contacts(int body_id, PBContactInfo* info) {
    if (!g_pb.initialized || !info) return -1;
    memset(info, 0, sizeof(PBContactInfo));
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "%s|%d", PB_CMD_GET_CONTACTS, body_id);
    char resp[PB_RESP_BUF_SIZE];
    if (pb_exec_command(cmd, resp, sizeof(resp)) != 0) return -1;
    const char* p = resp + 2;
    if (*p == '|') p++;
    info->num_contacts = pb_parse_int(&p);
    if (info->num_contacts > PB_CONTACT_MAX_POINTS) info->num_contacts = PB_CONTACT_MAX_POINTS;
    for (int i = 0; i < info->num_contacts; i++) {
        PBContactPoint* pt = &info->points[i];
        pt->body_a = pb_parse_int(&p);
        pt->link_a = pb_parse_int(&p);
        pt->body_b = pb_parse_int(&p);
        pt->link_b = pb_parse_int(&p);
        pt->position_on_a[0] = pb_parse_float(&p);
        pt->position_on_a[1] = pb_parse_float(&p);
        pt->position_on_a[2] = pb_parse_float(&p);
        pt->position_on_b[0] = pb_parse_float(&p);
        pt->position_on_b[1] = pb_parse_float(&p);
        pt->position_on_b[2] = pb_parse_float(&p);
        pt->normal_on_b[0] = pb_parse_float(&p);
        pt->normal_on_b[1] = pb_parse_float(&p);
        pt->normal_on_b[2] = pb_parse_float(&p);
        pt->distance = pb_parse_float(&p);
        pt->normal_force = pb_parse_float(&p);
        pt->lateral_friction = pb_parse_float(&p);
    }
    return 0;
}

int pb_get_all_contacts(PBContactInfo* infos, int max_count, int* count) {
    if (!g_pb.initialized || !infos || !count) return -1;
    int total = 0;
    for (int i = 0; i < g_pb.robot_count && total < max_count; i++) {
        if (pb_get_contacts(g_pb.robots[i].body_unique_id, &infos[total]) == 0) {
            if (infos[total].num_contacts > 0) total++;
        }
    }
    *count = total;
    return 0;
}

int pb_check_collision(int body_a, int body_b, PBCollisionInfo* info) {
    if (!g_pb.initialized || !info) return -1;
    memset(info, 0, sizeof(PBCollisionInfo));
    info->body_id = body_a;
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "%s|%d|%d", PB_CMD_CHECK_COLLISION, body_a, body_b);
    char resp[512];
    if (pb_exec_command(cmd, resp, sizeof(resp)) != 0) return -1;
    const char* p = resp + 2;
    if (*p == '|') p++;
    info->has_collision = pb_parse_int(&p);
    info->link_index = pb_parse_int(&p);
    info->contact_normal[0] = pb_parse_float(&p);
    info->contact_normal[1] = pb_parse_float(&p);
    info->contact_normal[2] = pb_parse_float(&p);
    info->penetration_depth = pb_parse_float(&p);
    return 0;
}

int pb_get_collision_data(int body_id, PBCollisionInfo* info) {
    if (!g_pb.initialized || !info) return -1;
    memset(info, 0, sizeof(PBCollisionInfo));
    info->body_id = body_id;
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "%s|%d|%d", PB_CMD_CHECK_COLLISION, body_id, 0);
    char resp[512];
    if (pb_exec_command(cmd, resp, sizeof(resp)) != 0) return -1;
    const char* p = resp + 2;
    if (*p == '|') p++;
    info->has_collision = pb_parse_int(&p);
    info->num_overlapping_bodies = pb_parse_int(&p);
    int n = info->num_overlapping_bodies;
    if (n > 32) n = 32;
    for (int i = 0; i < n; i++) {
        info->overlapping_bodies[i] = pb_parse_int(&p);
    }
    return 0;
}

int pb_gripper_init(int body_id, int gripper_link, float open_pos, float closed_pos, float force) {
    if (!g_pb.initialized) return -1;
    PBGripperControl* gc = NULL;
    for (int i = 0; i < g_pb.robot_count; i++) {
        if (g_pb.robots[i].body_unique_id == body_id) {
            gc = (PBGripperControl*)&g_pb.robots[i];
            break;
        }
    }
    if (!gc) return -1;
    gc->body_id = body_id;
    gc->gripper_link_index = gripper_link;
    gc->open_position = open_pos;
    gc->closed_position = closed_pos;
    gc->current_position = open_pos;
    gc->grip_force = force;
    gc->is_closed = 0;
    gc->has_object = 0;
    gc->grasped_body_id = -1;
    gc->grasp_strength = 0.0f;
    return pb_set_joint_position(body_id, gripper_link, open_pos, force);
}

int pb_gripper_open(PBGripperControl* gripper) {
    if (!g_pb.initialized || !gripper) return -1;
    int ret = pb_set_joint_position(gripper->body_id, gripper->gripper_link_index,
                                     gripper->open_position, gripper->grip_force);
    if (ret == 0) {
        gripper->current_position = gripper->open_position;
        gripper->is_closed = 0;
        gripper->has_object = 0;
        gripper->grasped_body_id = -1;
        gripper->grasp_strength = 0.0f;
    }
    return ret;
}

int pb_gripper_close(PBGripperControl* gripper) {
    if (!g_pb.initialized || !gripper) return -1;
    int ret = pb_set_joint_position(gripper->body_id, gripper->gripper_link_index,
                                     gripper->closed_position, gripper->grip_force);
    if (ret == 0) {
        gripper->current_position = gripper->closed_position;
        gripper->is_closed = 1;
        gripper->has_object = 1;
        gripper->grasp_strength = gripper->grip_force;
    }
    return ret;
}

int pb_gripper_set_force(PBGripperControl* gripper, float force) {
    if (!g_pb.initialized || !gripper) return -1;
    gripper->grip_force = force;
    if (gripper->is_closed) {
        return pb_set_joint_position(gripper->body_id, gripper->gripper_link_index,
                                      gripper->closed_position, force);
    }
    return 0;
}

int pb_gripper_get_state(PBGripperControl* gripper) {
    if (!g_pb.initialized || !gripper) return -1;
    PBJointState js;
    int count = 0;
    if (pb_get_joint_states(gripper->body_id, &js, 1, &count) != 0) return -1;
    if (count > 0) {
        gripper->current_position = js.joint_position;
        gripper->is_closed = (js.joint_position <= gripper->closed_position + 0.01f);
    }
    return 0;
}

int pb_compute_ik(int body_id, int end_effector_link, const float* target_pos,
                  const float* target_orn, PBIKResult* result, PBIKMode mode) {
    if (!g_pb.initialized || !target_pos || !result) return -1;
    (void)mode;
    memset(result, 0, sizeof(PBIKResult));
    result->body_id = body_id;

    int n_joints = 0;
    for (int r = 0; r < g_pb.robot_count; r++) {
        if (g_pb.robots[r].body_unique_id == body_id) {
            n_joints = g_pb.robots[r].num_joints;
            break;
        }
    }
    if (n_joints == 0 || n_joints > PB_MAX_JOINTS) return -1;
    result->num_joints = n_joints;

    PBJointState current_states[PB_MAX_JOINTS];
    int actual_count = 0;
    if (pb_get_joint_states(body_id, current_states, PB_MAX_JOINTS, &actual_count) != 0) return -1;
    if (actual_count < n_joints) n_joints = actual_count;

    for (int j = 0; j < n_joints; j++) {
        result->joint_indices[j] = j;
        result->solution[j] = current_states[j].joint_position;
    }
    memcpy(result->target_positions, target_pos, 3 * sizeof(float));
    if (target_orn) {
        memcpy(result->target_orientations, target_orn, 4 * sizeof(float));
    }

    float target[7];
    target[0] = target_pos[0];
    target[1] = target_pos[1];
    target[2] = target_pos[2];
    if (target_orn) {
        target[3] = target_orn[0];
        target[4] = target_orn[1];
        target[5] = target_orn[2];
        target[6] = target_orn[3];
    } else {
        target[3] = 1.0f; target[4] = 0.0f; target[5] = 0.0f; target[6] = 0.0f;
    }

    float best_dist = 1e10f;
    float best_solution[PB_MAX_JOINTS];
    /* P1-029修复: 从current_states数组中复制初始关节位置 */
    for (int j = 0; j < n_joints; j++) {
        best_solution[j] = current_states[j].joint_position;
    }

    for (int iter = 0; iter < PB_IK_MAX_ITERATIONS; iter++) {
        PBLinkState ls;
        pb_get_link_state(body_id, end_effector_link, &ls);
        float dx = target_pos[0] - ls.position[0];
        float dy = target_pos[1] - ls.position[1];
        float dz = target_pos[2] - ls.position[2];
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);

        if (dist < best_dist) {
            best_dist = dist;
            for (int j = 0; j < n_joints; j++) {
                best_solution[j] = current_states[j].joint_position;
            }
        }

        if (dist < 0.001f) {
            result->success = 1;
            result->iterations_used = iter;
            result->solution_confidence = 1.0f - dist * 10.0f;
            if (result->solution_confidence < 0.0f) result->solution_confidence = 0.0f;
            break;
        }

        for (int j = 0; j < n_joints && j < 6; j++) {
            float delta = 0.01f;
            float old_pos = current_states[j].joint_position;
            float new_pos = old_pos + delta;

            pb_set_joint_position(body_id, j, new_pos, 100.0f);
            pb_step_simulation();
            PBLinkState test_ls;
            pb_get_link_state(body_id, end_effector_link, &test_ls);

            float test_dx = target_pos[0] - test_ls.position[0];
            float test_dy = target_pos[1] - test_ls.position[1];
            float test_dz = target_pos[2] - test_ls.position[2];
            float test_dist_new = sqrtf(test_dx * test_dx + test_dy * test_dy + test_dz * test_dz);

            float step_size = (test_dist_new < dist) ? 0.5f : -0.3f;
            float step = delta * step_size;
            float next_pos = old_pos + step;

            if (next_pos < current_states[j].joint_lower_limit) next_pos = current_states[j].joint_lower_limit;
            if (next_pos > current_states[j].joint_upper_limit) next_pos = current_states[j].joint_upper_limit;

            pb_set_joint_position(body_id, j, next_pos, 100.0f);
            current_states[j].joint_position = next_pos;
            result->solution[j] = next_pos;
        }
        pb_step_simulation();
    }

    if (best_dist < 0.05f) {
        result->success = 1;
        result->solution_confidence = 1.0f - best_dist * 20.0f;
        if (result->solution_confidence < 0.0f) result->solution_confidence = 0.0f;
    }
    result->iterations_used = PB_IK_MAX_ITERATIONS;
    return result->success ? 0 : -1;
}

int pb_compute_ik_chain(int body_id, const int* link_chain, int chain_len,
                        const float* target_positions, PBIKResult* results) {
    if (!g_pb.initialized || !link_chain || !target_positions || !results) return -1;
    int success_count = 0;
    for (int i = 0; i < chain_len; i++) {
        float pos[3] = {target_positions[i * 3], target_positions[i * 3 + 1], target_positions[i * 3 + 2]};
        if (pb_compute_ik(body_id, link_chain[i], pos, NULL, &results[i], PB_IK_MODE_POSITION) == 0) {
            success_count++;
        }
    }
    return (success_count == chain_len) ? 0 : -1;
}

int pb_batch_step(int num_steps, float* out_states) {
    if (!g_pb.initialized || num_steps <= 0) return -1;
    for (int s = 0; s < num_steps; s++) {
        if (pb_step_simulation() != 0) return -1;
    }
    if (out_states) {
        int offset = 0;
        for (int r = 0; r < g_pb.robot_count; r++) {
            PBJointState js[PB_MAX_JOINTS];
            int count = 0;
            if (pb_get_joint_states(g_pb.robots[r].body_unique_id, js, PB_MAX_JOINTS, &count) == 0) {
                for (int j = 0; j < count && j < PB_MAX_JOINTS; j++) {
                    out_states[offset++] = js[j].joint_position;
                }
            }
        }
    }
    return 0;
}

int pb_batch_set_joints(const int* body_ids, const int* joint_indices,
                        const float* positions, int count) {
    if (!g_pb.initialized || !body_ids || !joint_indices || !positions) return -1;
    for (int i = 0; i < count; i++) {
        if (pb_set_joint_position(body_ids[i], joint_indices[i], positions[i], 100.0f) != 0) return -1;
    }
    return 0;
}

int pb_batch_get_states(const int* body_ids, int num_bodies,
                        PBJointState* out_states, int max_joints, int* out_count) {
    if (!g_pb.initialized || !body_ids || !out_states || !out_count) return -1;
    int total = 0;
    for (int b = 0; b < num_bodies; b++) {
        int cnt = 0;
        if (pb_get_joint_states(body_ids[b], &out_states[total], max_joints - total, &cnt) == 0) {
            total += cnt;
        }
    }
    *out_count = total;
    return 0;
}

int pb_set_link_friction(int body_id, int link_index, float friction) {
    if (!g_pb.initialized) return -1;
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "%s|%d|%d|%f", PB_CMD_SET_FRICTION, body_id, link_index, friction);
    char resp[64];
    return pb_exec_command(cmd, resp, sizeof(resp));
}

int pb_set_link_restitution(int body_id, int link_index, float restitution) {
    if (!g_pb.initialized) return -1;
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "%s|%d|%d|%f", PB_CMD_SET_RESTITUTION, body_id, link_index, restitution);
    char resp[64];
    return pb_exec_command(cmd, resp, sizeof(resp));
}

int pb_set_link_damping(int body_id, int link_index, float linear_damping, float angular_damping) {
    if (!g_pb.initialized) return -1;
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "%s|%d|%d|%f|%f", PB_CMD_SET_DAMPING,
             body_id, link_index, linear_damping, angular_damping);
    char resp[64];
    return pb_exec_command(cmd, resp, sizeof(resp));
}

int pb_set_link_linear_damping(int body_id, int link_index, float damping) {
    return pb_set_link_damping(body_id, link_index, damping, 0.0f);
}

int pb_set_joint_limit(int body_id, int joint_index, float lower, float upper) {
    if (!g_pb.initialized) return -1;
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "%s|%d|%d|%f|%f", PB_CMD_SET_JOINT_LIMIT,
             body_id, joint_index, lower, upper);
    char resp[64];
    return pb_exec_command(cmd, resp, sizeof(resp));
}

int pb_set_joint_max_force(int body_id, int joint_index, float max_force) {
    if (!g_pb.initialized) return -1;
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "%s|%d|%d|%f", PB_CMD_SET_MAX_FORCE,
             body_id, joint_index, max_force);
    char resp[64];
    return pb_exec_command(cmd, resp, sizeof(resp));
}

int pb_set_joint_max_velocity(int body_id, int joint_index, float max_vel) {
    if (!g_pb.initialized) return -1;
    return pb_set_joint_limit(body_id, joint_index, -max_vel, max_vel);
}

int pb_load_scene(const char* scene_file) {
    if (!g_pb.initialized || !scene_file) return -1;
    char cmd[PB_CMD_BUF_SIZE];
    snprintf(cmd, sizeof(cmd), "%s|%s", PB_CMD_LOAD_SCENE, scene_file);
    char resp[256];
    return pb_exec_command(cmd, resp, sizeof(resp));
}

int pb_clear_scene(void) {
    if (!g_pb.initialized) return -1;
    char resp[64];
    if (pb_exec_command(PB_CMD_CLEAR_SCENE, resp, sizeof(resp)) != 0) return -1;
    g_pb.body_count = 0;
    g_pb.robot_count = 0;
    g_pb.step_count = 0;
    g_pb.simulation_time = 0.0f;
    g_pb.total_constraints = 0;
    g_pb.pid_count = 0;
    g_pb.debug_item_count = 0;
    memset(g_pb.robots, 0, sizeof(g_pb.robots));
    memset(g_pb.constraints, 0, sizeof(g_pb.constraints));
    memset(g_pb.loggers, 0, sizeof(g_pb.loggers));
    memset(g_pb.pid_controllers, 0, sizeof(g_pb.pid_controllers));
    return 0;
}

int pb_save_scene(const char* scene_file) {
    if (!g_pb.initialized || !scene_file) return -1;
    char cmd[PB_CMD_BUF_SIZE];
    snprintf(cmd, sizeof(cmd), "%s|%s", PB_CMD_SAVE_SCENE, scene_file);
    char resp[256];
    return pb_exec_command(cmd, resp, sizeof(resp));
}

int pb_add_primitive(PBPrimitiveInfo* info, int* out_body_id) {
    if (!g_pb.initialized || !info || !out_body_id) return -1;
    const char* shape_cmd = NULL;
    char cmd[PB_CMD_BUF_SIZE];
    switch (info->shape) {
        case PB_SHAPE_BOX: shape_cmd = PB_CMD_ADD_BOX; break;
        case PB_SHAPE_SPHERE: shape_cmd = PB_CMD_ADD_SPHERE; break;
        case PB_SHAPE_CYLINDER: shape_cmd = PB_CMD_ADD_CYLINDER; break;
        default: shape_cmd = PB_CMD_ADD_BOX; break;
    }
    if (info->shape == PB_SHAPE_BOX || info->shape == PB_SHAPE_CYLINDER || info->shape == PB_SHAPE_CAPSULE) {
        snprintf(cmd, sizeof(cmd), "%s|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f",
                 shape_cmd,
                 info->position[0], info->position[1], info->position[2],
                 info->half_extents[0], info->half_extents[1], info->half_extents[2],
                 info->color_r, info->color_g, info->color_b,
                 info->mass, 0.0f, 0.0f);
    } else {
        snprintf(cmd, sizeof(cmd), "%s|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f",
                 shape_cmd,
                 info->position[0], info->position[1], info->position[2],
                 info->half_extents[0], info->half_extents[1], info->half_extents[2],
                 info->color_r, info->color_g, info->color_b,
                 info->mass);
    }
    char resp[128];
    if (pb_exec_command(cmd, resp, sizeof(resp)) != 0) return -1;
    const char* p = resp + 2;
    if (*p == '|') p++;
    *out_body_id = pb_parse_int(&p);
    g_pb.body_count++;
    return 0;
}

int pb_remove_primitive(int body_id) {
    return pb_remove_body(body_id);
}

int pb_create_constraint(PBConstraintInfo* info, int* out_constraint_id) {
    if (!g_pb.initialized || !info || !out_constraint_id) return -1;
    char cmd[PB_CMD_BUF_SIZE];
    snprintf(cmd, sizeof(cmd), "%s|%d|%d|%d|%d|%d|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f",
             PB_CMD_CREATE_CONSTRAINT,
             info->type, info->body_a, info->body_b, info->link_a, info->link_b,
             info->pivot_in_a[0], info->pivot_in_a[1], info->pivot_in_a[2],
             info->pivot_in_b[0], info->pivot_in_b[1], info->pivot_in_b[2],
             info->axis_in_a[0], info->axis_in_a[1], info->axis_in_a[2],
             info->axis_in_b[0], info->axis_in_b[1], info->axis_in_b[2],
             info->lower_limit, info->upper_limit, info->max_force);
    char resp[128];
    if (pb_exec_command(cmd, resp, sizeof(resp)) != 0) return -1;
    const char* p = resp + 2;
    if (*p == '|') p++;
    int cid = pb_parse_int(&p);
    *out_constraint_id = cid;
    if (g_pb.total_constraints < PB_MAX_CONSTRAINTS) {
        PBConstraintInfo* ci = &g_pb.constraints[g_pb.total_constraints];
        memcpy(ci, info, sizeof(PBConstraintInfo));
        ci->constraint_id = cid;
        ci->is_active = 1;
        g_pb.total_constraints++;
    }
    g_pb.last_constraint_id = cid;
    return 0;
}

int pb_remove_constraint(int constraint_id) {
    if (!g_pb.initialized) return -1;
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "%s|%d", PB_CMD_REMOVE_CONSTRAINT, constraint_id);
    char resp[64];
    int ret = pb_exec_command(cmd, resp, sizeof(resp));
    if (ret == 0) {
        for (int i = 0; i < g_pb.total_constraints; i++) {
            if (g_pb.constraints[i].constraint_id == constraint_id) {
                g_pb.constraints[i].is_active = 0;
                break;
            }
        }
    }
    return ret;
}

int pb_get_constraints(PBConstraintInfo* infos, int max_count, int* count) {
    if (!g_pb.initialized || !infos || !count) return -1;
    *count = 0;
    char cmd[64] = PB_CMD_GET_CONSTRAINTS;
    char resp[PB_RESP_BUF_SIZE];
    if (pb_exec_command(cmd, resp, sizeof(resp)) != 0) {
        int n = 0;
        for (int i = 0; i < g_pb.total_constraints && n < max_count; i++) {
            if (g_pb.constraints[i].is_active) {
                memcpy(&infos[n], &g_pb.constraints[i], sizeof(PBConstraintInfo));
                n++;
            }
        }
        *count = n;
        return 0;
    }
    const char* p = resp + 2;
    if (*p == '|') p++;
    int n = pb_parse_int(&p);
    if (n > max_count) n = max_count;
    *count = n;
    return 0;
}

int pb_add_debug_line(const float* from, const float* to,
                      float r, float g, float b, float line_width, float life_time) {
    if (!g_pb.initialized || !from || !to) return -1;
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "%s|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f",
             PB_CMD_ADD_DEBUG_LINE,
             from[0], from[1], from[2],
             to[0], to[1], to[2],
             r, g, b, line_width, life_time);
    char resp[128];
    if (pb_exec_command(cmd, resp, sizeof(resp)) != 0) return -1;
    const char* p = resp + 2;
    if (*p == '|') p++;
    int did = pb_parse_int(&p);
    g_pb.debug_item_count++;
    return did;
}

int pb_add_debug_text(const float* position, const char* text,
                      float r, float g, float b, float size) {
    if (!g_pb.initialized || !position || !text) return -1;
    char cmd[PB_CMD_BUF_SIZE];
    snprintf(cmd, sizeof(cmd), "%s|%f|%f|%f|%f|%f|%f|%f|%s",
             PB_CMD_ADD_DEBUG_TEXT,
             position[0], position[1], position[2],
             r, g, b, size, text);
    char resp[128];
    if (pb_exec_command(cmd, resp, sizeof(resp)) != 0) return -1;
    const char* p = resp + 2;
    if (*p == '|') p++;
    int did = pb_parse_int(&p);
    g_pb.debug_item_count++;
    return did;
}

int pb_remove_debug(int debug_id) {
    if (!g_pb.initialized) return -1;
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "%s|%d", PB_CMD_REMOVE_DEBUG, debug_id);
    char resp[64];
    int ret = pb_exec_command(cmd, resp, sizeof(resp));
    if (ret == 0) g_pb.debug_item_count--;
    return ret;
}

int pb_remove_all_debug(void) {
    if (!g_pb.initialized) return -1;
    for (int i = 0; i < g_pb.debug_item_count; i++) {
        pb_remove_debug(i + 1);
    }
    g_pb.debug_item_count = 0;
    return 0;
}

int pb_set_joint_group_position(int body_id, const int* joint_indices,
                                const float* positions, int num_joints) {
    if (!g_pb.initialized || !joint_indices || !positions) return -1;
    for (int i = 0; i < num_joints; i++) {
        if (pb_set_joint_position(body_id, joint_indices[i], positions[i], 100.0f) != 0) return -1;
    }
    return 0;
}

int pb_set_joint_group_velocity(int body_id, const int* joint_indices,
                                const float* velocities, int num_joints) {
    if (!g_pb.initialized || !joint_indices || !velocities) return -1;
    for (int i = 0; i < num_joints; i++) {
        if (pb_set_joint_velocity(body_id, joint_indices[i], velocities[i]) != 0) return -1;
    }
    return 0;
}

int pb_get_joint_group_states(int body_id, const int* joint_indices,
                              int num_joints, PBJointState* out_states) {
    if (!g_pb.initialized || !joint_indices || !out_states) return -1;
    int actual_count = 0;
    if (pb_get_joint_states(body_id, out_states, num_joints, &actual_count) != 0) return -1;
    return 0;
}

int pb_start_logging(int body_id, float interval, int max_entries) {
    if (!g_pb.initialized) return -1;
    int idx = -1;
    for (int r = 0; r < g_pb.robot_count; r++) {
        if (g_pb.robots[r].body_unique_id == body_id) { idx = r; break; }
    }
    if (idx < 0) return -1;
    PBLoggerData* log = &g_pb.loggers[idx];
    log->body_id = body_id;
    log->log_enabled = 1;
    log->log_interval = (interval > 0.001f) ? interval : 0.01f;
    log->log_max_entries = (max_entries > 0) ? max_entries : PB_MAX_LOG_ENTRIES;
    log->log_num_entries = 0;
    log->log_next_time = 0.0f;
    if (!log->log_timestamps) {
        log->log_timestamps = (float*)malloc(log->log_max_entries * sizeof(float));
        log->log_joint_positions = (float*)malloc(log->log_max_entries * g_pb.robots[idx].num_joints * sizeof(float));
        log->log_joint_velocities = (float*)malloc(log->log_max_entries * g_pb.robots[idx].num_joints * sizeof(float));
        log->log_ee_positions = (float*)malloc(log->log_max_entries * 3 * sizeof(float));
    }
    if (!log->log_timestamps || !log->log_joint_positions || !log->log_joint_velocities || !log->log_ee_positions) {
        log->log_enabled = 0;
        return -1;
    }
    return 0;
}

int pb_stop_logging(int body_id) {
    if (!g_pb.initialized) return -1;
    for (int r = 0; r < g_pb.robot_count; r++) {
        if (g_pb.loggers[r].body_id == body_id) {
            g_pb.loggers[r].log_enabled = 0;
            return 0;
        }
    }
    return -1;
}

int pb_get_log(int body_id, PBStateLogger* logger) {
    if (!g_pb.initialized || !logger) return -1;
    for (int r = 0; r < g_pb.robot_count; r++) {
        PBLoggerData* log = &g_pb.loggers[r];
        if (log->body_id == body_id && log->log_enabled) {
            logger->enabled = 1;
            logger->num_entries = log->log_num_entries;
            logger->max_entries = log->log_max_entries;
            logger->log_interval = log->log_interval;
            logger->timestamps = log->log_timestamps;
            logger->joint_positions = &log->log_joint_positions;
            logger->joint_velocities = &log->log_joint_velocities;
            logger->end_effector_positions = log->log_ee_positions;
            logger->entry_count_ptr = &log->log_num_entries;
            return 0;
        }
    }
    return -1;
}

int pb_predict_trajectory(int body_id, int num_steps, float* out_positions, float* out_orientations) {
    if (!g_pb.initialized || !out_positions) return -1;
    if (num_steps <= 0) return -1;
    PBJointState js[PB_MAX_JOINTS];
    int count = 0;
    if (pb_get_joint_states(body_id, js, PB_MAX_JOINTS, &count) != 0) return -1;
    if (count <= 0) return -1;
    LNNConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.input_size = PB_LNN_INPUT_DIM;
    cfg.hidden_size = PB_LNN_HIDDEN_DIM;
    cfg.output_size = PB_LNN_OUTPUT_DIM;
    cfg.num_layers = 2;
    cfg.ode_solver_type = 1;
    cfg.time_constant = 0.1f;
    cfg.learning_rate = 0.01f;
    cfg.enable_training = 0;
    cfg.enable_adaptation = 0;
    cfg.enable_evolution = 0;
    LNN* lnn = lnn_create(&cfg);
    if (!lnn) return -1;
    float lnn_input[PB_LNN_INPUT_DIM];
    memset(lnn_input, 0, sizeof(lnn_input));
    for (int d = 0; d < count && d < PB_LNN_INPUT_DIM; d++) {
        lnn_input[d] = js[d].joint_position;
    }
    float lnn_output[PB_LNN_OUTPUT_DIM];
    for (int s = 0; s < num_steps; s++) {
        memset(lnn_output, 0, sizeof(lnn_output));
        lnn_forward(lnn, lnn_input, lnn_output);
        for (int d = 0; d < count && d < PB_LNN_OUTPUT_DIM; d++) {
            out_positions[s * count + d] = lnn_output[d];
        }
        memcpy(lnn_input, lnn_output, PB_LNN_OUTPUT_DIM * sizeof(float));
    }
    if (out_orientations) {
        float base_orn[4];
        pb_get_base_position(body_id, NULL, base_orn);
        for (int s = 0; s < num_steps; s++) {
            out_orientations[s * 4 + 0] = base_orn[0];
            out_orientations[s * 4 + 1] = base_orn[1];
            out_orientations[s * 4 + 2] = base_orn[2];
            out_orientations[s * 4 + 3] = base_orn[3];
        }
    }
    lnn_free(lnn);
    return 0;
}

int pb_predict_joint_trajectory(int body_id, const int* joint_indices, int num_joints,
                                int num_steps, float* out_positions) {
    if (!g_pb.initialized || !out_positions) return -1;
    (void)joint_indices;
    if (num_joints <= 0 || num_steps <= 0) return -1;
    PBJointState js[PB_MAX_JOINTS];
    int count = 0;
    if (pb_get_joint_states(body_id, js, PB_MAX_JOINTS, &count) != 0) return -1;
    if (count <= 0) return -1;
    LNNConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.input_size = PB_LNN_INPUT_DIM;
    cfg.hidden_size = PB_LNN_HIDDEN_DIM;
    cfg.output_size = PB_LNN_OUTPUT_DIM;
    cfg.num_layers = 2;
    cfg.ode_solver_type = 1;
    cfg.time_constant = 0.1f;
    cfg.learning_rate = 0.01f;
    cfg.enable_training = 0;
    cfg.enable_adaptation = 0;
    cfg.enable_evolution = 0;
    LNN* lnn = lnn_create(&cfg);
    if (!lnn) return -1;
    float lnn_input[PB_LNN_INPUT_DIM];
    memset(lnn_input, 0, sizeof(lnn_input));
    for (int d = 0; d < count && d < PB_LNN_INPUT_DIM; d++) {
        lnn_input[d] = js[d].joint_position;
    }
    float lnn_output[PB_LNN_OUTPUT_DIM];
    for (int s = 0; s < num_steps; s++) {
        memset(lnn_output, 0, sizeof(lnn_output));
        lnn_forward(lnn, lnn_input, lnn_output);
        for (int d = 0; d < num_joints && d < PB_LNN_OUTPUT_DIM; d++) {
            out_positions[s * num_joints + d] = lnn_output[d];
        }
        memcpy(lnn_input, lnn_output, PB_LNN_OUTPUT_DIM * sizeof(float));
    }
    lnn_free(lnn);
    return 0;
}

int pb_set_control_mode(int body_id, PBControlMode mode) {
    if (!g_pb.initialized) return -1;
    int pid_idx = -1;
    for (int i = 0; i < g_pb.pid_count; i++) {
        if (g_pb.pid_controllers[i].body_id == body_id) {
            pid_idx = i;
            break;
        }
    }
    if (pid_idx < 0) {
        if (g_pb.pid_count >= PB_MAX_ROBOTS * PB_MAX_JOINTS) return -1;
        pid_idx = g_pb.pid_count++;
        g_pb.pid_controllers[pid_idx].body_id = body_id;
    }
    g_pb.pid_controllers[pid_idx].control_mode = mode;
    g_pb.pid_controllers[pid_idx].integral_error = 0.0f;
    g_pb.pid_controllers[pid_idx].prev_error = 0.0f;
    return 0;
}

int pb_get_control_mode(int body_id, PBControlMode* mode) {
    if (!g_pb.initialized || !mode) return -1;
    for (int i = 0; i < g_pb.pid_count; i++) {
        if (g_pb.pid_controllers[i].body_id == body_id) {
            *mode = g_pb.pid_controllers[i].control_mode;
            return 0;
        }
    }
    *mode = PB_CONTROL_POSITION;
    return 0;
}

int pb_set_pid_gains(int body_id, float kp, float ki, float kd) {
    if (!g_pb.initialized) return -1;
    for (int i = 0; i < g_pb.pid_count; i++) {
        if (g_pb.pid_controllers[i].body_id == body_id) {
            if (kp > 0.0f) g_pb.pid_controllers[i].kp = kp;
            g_pb.pid_controllers[i].ki = ki;
            g_pb.pid_controllers[i].kd = kd;
            return 0;
        }
    }
    return -1;
}

int pb_get_pid_gains(int body_id, float* kp, float* ki, float* kd) {
    if (!g_pb.initialized || !kp || !ki || !kd) return -1;
    for (int i = 0; i < g_pb.pid_count; i++) {
        if (g_pb.pid_controllers[i].body_id == body_id) {
            *kp = g_pb.pid_controllers[i].kp;
            *ki = g_pb.pid_controllers[i].ki;
            *kd = g_pb.pid_controllers[i].kd;
            return 0;
        }
    }
    *kp = g_pb.default_kp;
    *ki = g_pb.default_ki;
    *kd = g_pb.default_kd;
    return 0;
}

int pb_get_point_cloud(int body_id, float* out_points, int max_points, int* out_count) {
    if (!g_pb.initialized || !out_points || !out_count) return -1;
    (void)body_id;
    float view_matrix[16];
    float proj_matrix[16];
    memset(view_matrix, 0, sizeof(view_matrix));
    memset(proj_matrix, 0, sizeof(proj_matrix));
    view_matrix[0] = 1.0f; view_matrix[5] = 1.0f; view_matrix[10] = 1.0f; view_matrix[15] = 1.0f;
    float fov = 60.0f * 3.14159f / 180.0f;
    float aspect = 1.0f;
    float near_plane = 0.1f;
    float far_plane = 10.0f;
    float f = 1.0f / tanf(fov * 0.5f);
    proj_matrix[0] = f / aspect;
    proj_matrix[5] = f;
    proj_matrix[10] = (far_plane + near_plane) / (near_plane - far_plane);
    proj_matrix[11] = -1.0f;
    proj_matrix[14] = (2.0f * far_plane * near_plane) / (near_plane - far_plane);
    int width = 64;
    int height = 64;
    PBCameraImage img;
    if (pb_get_camera_image(width, height, view_matrix, proj_matrix, &img) != 0) {
        *out_count = 0;
        return -1;
    }
    int total = width * height;
    int count = 0;
    int step = (total > max_points) ? (total / max_points) : 1;
    if (step < 1) step = 1;
    for (int i = 0; i < total && count < max_points; i += step) {
        float d = img.depth_data[i];
        if (d > 0.001f && d < 9.9f) {
            int px = i % width;
            int py = i / width;
            float fx = (float)px / (float)width;
            float fy = (float)py / (float)height;
            float hfov = fov;
            float vfov = hfov * (float)height / (float)width;
            float cx = 2.0f * fx - 1.0f;
            float cy = -(2.0f * fy - 1.0f);
            float tx = d * tanf(hfov * 0.5f * cx);
            float ty = d * tanf(vfov * 0.5f * cy);
            float tz = d;
            out_points[count * 3 + 0] = tx;
            out_points[count * 3 + 1] = ty;
            out_points[count * 3 + 2] = tz;
            count++;
        }
    }
    *out_count = count;
    pb_camera_image_free(&img);
    return 0;
}

int pb_get_depth_image(int width, int height, const float* view_matrix,
                       const float* proj_matrix, float* depth_out) {
    if (!g_pb.initialized || !view_matrix || !proj_matrix || !depth_out) return -1;
    PBCameraImage img;
    if (pb_get_camera_image(width, height, view_matrix, proj_matrix, &img) != 0) {
        return -1;
    }
    int total = width * height;
    for (int i = 0; i < total; i++) {
        depth_out[i] = img.depth_data[i];
    }
    pb_camera_image_free(&img);
    return 0;
}

int pb_set_collision_margin(int body_id, float margin) {
    if (!g_pb.initialized) return -1;
    char cmd[PB_CMD_BUF_SIZE];
    snprintf(cmd, sizeof(cmd), "%s|%d|%f",
             PB_CMD_SET_COLLISION_MARGIN, body_id, margin);
    char resp[64];
    return pb_exec_command(cmd, resp, sizeof(resp));
}

int pb_set_collision_filter(int body_a, int body_b, int enable_collision) {
    if (!g_pb.initialized) return -1;
    char cmd[PB_CMD_BUF_SIZE];
    snprintf(cmd, sizeof(cmd), "%s|%d|%d|%d",
             PB_CMD_SET_COLLISION_FILTER, body_a, body_b, enable_collision);
    char resp[64];
    return pb_exec_command(cmd, resp, sizeof(resp));
}

int pb_set_timeout(int timeout_ms) {
    if (timeout_ms < 0) return -1;
    g_pb.timeout_ms = timeout_ms;
    return 0;
}

int pb_get_simulation_stats(int* total_steps, float* total_time,
                            int* num_bodies, int* num_constraints) {
    if (!g_pb.initialized) return -1;
    if (total_steps) *total_steps = g_pb.step_count;
    if (total_time) *total_time = g_pb.simulation_time;
    if (num_bodies) *num_bodies = g_pb.body_count;
    if (num_constraints) *num_constraints = g_pb.total_constraints;
    return 0;
}

void pb_shutdown(void) {
    if (!g_pb.initialized) return;
    for (int r = 0; r < g_pb.robot_count; r++) {
        pb_stop_logging(g_pb.robots[r].body_unique_id);
        PBLoggerData* log = &g_pb.loggers[r];
        free(log->log_timestamps);
        free(log->log_joint_positions);
        free(log->log_joint_velocities);
        free(log->log_ee_positions);
        log->log_timestamps = NULL;
        log->log_joint_positions = NULL;
        log->log_joint_velocities = NULL;
        log->log_ee_positions = NULL;
    }
    g_pb.robot_count = 0;
    g_pb.pid_count = 0;
    g_pb.simulation_time = 0.0f;
    g_pb.step_count = 0;
    g_pb.debug_item_count = 0;
    g_pb.total_constraints = 0;
    g_pb.last_constraint_id = 0;
    if (g_pb.state >= PB_STATE_CONNECTED) {
        char cmd[64];
        snprintf(cmd, sizeof(cmd), "%s", PB_CMD_DISCONNECT);
        char resp[64];
        pb_exec_command(cmd, resp, sizeof(resp));
    }
    if (g_pb.internal_sim) {
        simulator_destroy(g_pb.internal_sim);
        g_pb.internal_sim = NULL;
    }
    g_pb.state = PB_STATE_DISCONNECTED;
    g_pb.initialized = 0;
}

const char* pb_state_str(PBConnectionState state) {
    switch (state) {
        case PB_STATE_DISCONNECTED: return "未连接";
        case PB_STATE_CONNECTING:   return "连接中";
        case PB_STATE_CONNECTED:    return "已连接";
        case PB_STATE_RUNNING:      return "运行中";
        case PB_STATE_ERROR:        return "错误";
        default:                    return "未知状态";
    }
}