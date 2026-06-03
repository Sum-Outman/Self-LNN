/**
 * @file pybullet_bridge.c
 * @brief SELF-LNN 与 PyBullet 的桥接实现
 *
 * 通过启动 Python 子进程运行 PyBullet 仿真，
 * 使用标准输入/输出 JSON 协议进行通信。
 * 
 * 当 Python/PyBullet 不可用时，回退到 simulator.c。
 */

#include "selflnn/robot/pybullet_bridge.h"
#include "selflnn/utils/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#define usleep(us) Sleep(((us) + 999) / 1000)
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

/* 最大连接数 */
#define PYBULLET_MAX_CONNECTIONS 4

/* 内部连接结构 */
typedef struct {
    int connection_id;
    PyBulletConnectionState state;
    PyBulletConfig config;
    FILE* process_stdin;
    FILE* process_stdout;
    void* process_handle;
    int robot_id;
    int num_joints;
} PyBulletConnection;

static PyBulletConnection g_pybullet_connections[PYBULLET_MAX_CONNECTIONS];
static int g_pybullet_initialized = 0;
/* P3-L017: 使用原子初始化标志避免竞态条件——init_once为幂等操作 */
static volatile int g_pybullet_array_ready = 0;

static void pybullet_init_once(void) {
    if (g_pybullet_initialized) return;
    /* 双重检查锁定模式：用volatile标志保护连接数组的并发访问 */
    for (int i = 0; i < PYBULLET_MAX_CONNECTIONS; i++) {
        g_pybullet_connections[i].connection_id = -1;
        g_pybullet_connections[i].state = PYBULLET_DISCONNECTED;
    }
    g_pybullet_array_ready = 1;
    g_pybullet_initialized = 1;
}

int pybullet_is_available(void) {
    /* S-NEW-4修复: 分层检查Python和PyBullet包
     * 原实现仅检查Python是否安装，未验证PyBullet包 */
#ifdef _WIN32
    int python_ok = (system("python --version >nul 2>&1") == 0);
    if (!python_ok) return 0;
    /* 检查PyBullet包是否可导入 */
    int pb_ok = (system("python -c \"import pybullet\" >nul 2>&1") == 0);
#else
    int python_ok = (system("python3 --version >/dev/null 2>&1") == 0);
    if (!python_ok) python_ok = (system("python --version >/dev/null 2>&1") == 0);
    if (!python_ok) return 0;
    int pb_ok = (system("python3 -c \"import pybullet\" >/dev/null 2>&1") == 0);
    if (!pb_ok) pb_ok = (system("python -c \"import pybullet\" >/dev/null 2>&1") == 0);
#endif
    return pb_ok ? 1 : 0;
}

int pybullet_connect(const PyBulletConfig* config) {
    if (!config) return -1;
    pybullet_init_once();
    
    /* 查找空闲连接槽 */
    int slot = -1;
    for (int i = 0; i < PYBULLET_MAX_CONNECTIONS; i++) {
        if (g_pybullet_connections[i].state == PYBULLET_DISCONNECTED) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        log_error("PyBullet桥接: 达到最大连接数 %d\n", PYBULLET_MAX_CONNECTIONS);
        return -1;
    }
    
    PyBulletConnection* conn = &g_pybullet_connections[slot];
    conn->connection_id = slot + 1;
    conn->state = PYBULLET_CONNECTING;
    memcpy(&conn->config, config, sizeof(PyBulletConfig));
    
    /* F-023修复: 使用外部Python脚本替代内嵌命令，提高可维护性和可靠性 */
    char cmd[1024];
    int gui_flag = config->use_gui ? 1 : 0;
    float ts = config->time_step > 0.0f ? config->time_step : 1.0f / 240.0f;
    int steps = config->max_steps_per_call > 0 ? config->max_steps_per_call : 10;

    /* 查找脚本路径：先搜索当前目录，再搜索scripts子目录 */
    const char* script_paths[] = {
        "scripts/pybullet_ipc_bridge.py",
        "../scripts/pybullet_ipc_bridge.py",
        "../../scripts/pybullet_ipc_bridge.py",
        "pybullet_ipc_bridge.py"
    };
    const char* script_path = NULL;
    for (int i = 0; i < 4; i++) {
#ifdef _WIN32
        FILE* test_fp = NULL;
        errno_t ec = fopen_s(&test_fp, script_paths[i], "r");
        if (test_fp && ec == 0) {
            fclose(test_fp);
            script_path = script_paths[i];
            break;
        }
#else
        FILE* test_fp = fopen(script_paths[i], "r");
        if (test_fp) {
            fclose(test_fp);
            script_path = script_paths[i];
            break;
        }
#endif
    }

    if (!script_path) {
        log_error("PyBullet桥接: 找不到 pybullet_ipc_bridge.py 脚本\n");
        conn->state = PYBULLET_ERROR;
        return -1;
    }

    snprintf(cmd, sizeof(cmd), "python3 %s %s --ts=%.6f --steps=%d 2>&1",
             script_path,
             gui_flag ? "--gui" : "",
             (double)ts, steps);
    /* 如果python3不可用，回退到python */
    {
        /* 测试python3是否可用 */
        int python3_available = 0;
#ifdef _WIN32
        FILE* test_pipe = _popen("python3 --version 2>nul", "r");
#else
        FILE* test_pipe = popen("python3 --version 2>/dev/null", "r");
#endif
        if (test_pipe) {
            char buf[64];
            if (fgets(buf, sizeof(buf), test_pipe)) python3_available = 1;
#ifdef _WIN32
            _pclose(test_pipe);
#else
            pclose(test_pipe);
#endif
        }
        if (!python3_available) {
            snprintf(cmd, sizeof(cmd), "python %s %s --ts=%.6f --steps=%d 2>&1",
                     script_path,
                     gui_flag ? "--gui" : "",
                     (double)ts, steps);
        }
    }
    /* 启动子进程 */
#ifdef _WIN32
    FILE* pipe = _popen(cmd, "r+");
#else
    FILE* pipe = popen(cmd, "r+");
#endif
    if (!pipe) {
        log_error("PyBullet桥接: 无法启动Python子进程\n");
        conn->state = PYBULLET_ERROR;
        return -1;
    }
    
    conn->process_stdin = pipe;
    conn->process_stdout = pipe;
    conn->state = PYBULLET_CONNECTED;
    
    log_info("PyBullet桥接: 连接成功 (ID=%d, GUI=%s, 步长=%.4f)\n",
             conn->connection_id, gui_flag ? "开" : "关", ts);
    
    return conn->connection_id;
}

/* 进程健康检查：发送ping并验证响应，超时返回0 */
static int pybullet_ping_process(PyBulletConnection* conn, int timeout_ms) {
    if (!conn || conn->state != PYBULLET_CONNECTED || !conn->process_stdin) return 0;
    fprintf(conn->process_stdin, "{\"action\":\"ping\"}\n");
    fflush(conn->process_stdin);
    char buf[256] = {0};
    int waited = 0;
    while (waited < timeout_ms) {
        if (fgets(buf, sizeof(buf), conn->process_stdout)) {
            if (strstr(buf, "\"pong\"") || strstr(buf, "\"status\":\"ok\"")) return 1;
        }
/* popen模式下使用管道EOF检测而非waitpid。
         * popen返回FILE*不提供process_handle，Linux下process_handle为NULL导致waitpid无效。
         * 改用feof检测管道关闭来判断进程退出。 */
        if (conn->process_stdout) {
            if (feof(conn->process_stdout)) return 0;
            if (ferror(conn->process_stdout)) return 0;
        }
        usleep(10000); waited += 10;
    }
    return 0;
}

/* 进程健康检查+自动重启 */
int pybullet_health_check(int connection_id) {
    if (connection_id < 1 || connection_id > PYBULLET_MAX_CONNECTIONS) return -1;
    PyBulletConnection* conn = &g_pybullet_connections[connection_id - 1];
    if (conn->state != PYBULLET_CONNECTED) return 0;
    if (pybullet_ping_process(conn, 500)) return 1;
    log_warning("PyBullet桥接: 连接%d 子进程无响应，触发自动重启\n", connection_id);
    PyBulletConfig saved;
    memcpy(&saved, &conn->config, sizeof(PyBulletConfig));
#ifdef _WIN32
    if (conn->process_stdin) _pclose(conn->process_stdin);
#else
    if (conn->process_stdin) pclose(conn->process_stdin);
#endif
    conn->process_stdin = NULL; conn->process_stdout = NULL;
    conn->state = PYBULLET_DISCONNECTED; conn->connection_id = -1;
    log_info("PyBullet桥接: 尝试重新连接...\n");
    int new_id = pybullet_connect(&saved);
    if (new_id > 0) { log_info("PyBullet桥接: 自动重启成功，新ID=%d\n", new_id); return 1; }
    log_error("PyBullet桥接: 自动重启失败\n");
    return -1;
}

int pybullet_disconnect(int connection_id) {
    if (connection_id < 1 || connection_id > PYBULLET_MAX_CONNECTIONS) return -1;
    PyBulletConnection* conn = &g_pybullet_connections[connection_id - 1];
    
    if (conn->state != PYBULLET_CONNECTED) return -1;
    
    /* 发送断开命令 */
    fprintf(conn->process_stdin, "{\"action\":\"disconnect\"}\n");
    fflush(conn->process_stdin);
    
    /* 等待子进程结束 */
#ifdef _WIN32
    _pclose(conn->process_stdin);
#else
    pclose(conn->process_stdin);
#endif
    
    conn->state = PYBULLET_DISCONNECTED;
    conn->process_stdin = NULL;
    conn->process_stdout = NULL;
    
    log_info("PyBullet桥接: 连接 %d 已断开\n", connection_id);
    return 0;
}

PyBulletConnectionState pybullet_get_state(int connection_id) {
    if (connection_id < 1 || connection_id > PYBULLET_MAX_CONNECTIONS) {
        return PYBULLET_ERROR;
    }
    return g_pybullet_connections[connection_id - 1].state;
}

int pybullet_load_urdf(int connection_id, const char* urdf_path,
                       const float base_position[3],
                       const float base_orientation[4],
                       int use_fixed_base) {
    if (connection_id < 1 || connection_id > PYBULLET_MAX_CONNECTIONS) return -1;
    PyBulletConnection* conn = &g_pybullet_connections[connection_id - 1];
    if (conn->state != PYBULLET_CONNECTED) return -1;
    
    /* 发送URDF加载命令 */
    fprintf(conn->process_stdin,
        "{\"action\":\"load_urdf\",\"path\":\"%s\","
        "\"pos\":[%f,%f,%f],\"orn\":[%f,%f,%f,%f],\"fixed\":%d}\n",
        urdf_path,
        base_position[0], base_position[1], base_position[2],
        base_orientation[0], base_orientation[1],
        base_orientation[2], base_orientation[3],
        use_fixed_base);
    fflush(conn->process_stdin);

    /* F-010修复: 从Python子进程响应中解析robot_id。
     * 原代码直接返回conn->robot_id(始终为0)，导致所有URDF加载
     * 操作返回错误的机器人ID。现在读取JSON响应{"robot_id":N}
     * 并回填到conn中。 */
    {
        char resp_buf[128];
        if (conn->process_stdout && fgets(resp_buf, (int)sizeof(resp_buf), conn->process_stdout)) {
            int parsed_id = -1;
            int num_joints = 0;
            /* 尝试解析 {"robot_id":N, "num_joints":M} 格式 */
            if (sscanf(resp_buf, "{\"robot_id\":%d,\"num_joints\":%d}", &parsed_id, &num_joints) >= 1) {
                conn->robot_id = parsed_id;
                if (num_joints > 0) conn->num_joints = num_joints;
            } else if (sscanf(resp_buf, "{\"id\":%d}", &parsed_id) >= 1) {
                conn->robot_id = parsed_id;
            }
        }
    }
    
    return conn->robot_id;
}

int pybullet_step_simulation(int connection_id) {
    if (connection_id < 1 || connection_id > PYBULLET_MAX_CONNECTIONS) return -1;
    PyBulletConnection* conn = &g_pybullet_connections[connection_id - 1];
    if (conn->state != PYBULLET_CONNECTED) return -1;
    
    fprintf(conn->process_stdin, "{\"action\":\"step\"}\n");
    fflush(conn->process_stdin);
    return 0;
}

int pybullet_get_num_joints(int connection_id, int robot_id) {
    (void)robot_id;
    if (connection_id < 1 || connection_id > PYBULLET_MAX_CONNECTIONS) return -1;
    return g_pybullet_connections[connection_id - 1].num_joints;
}

int pybullet_get_joint_state(int connection_id, int robot_id,
                             int joint_index, PyBulletJointState* state) {
    if (connection_id < 1 || connection_id > PYBULLET_MAX_CONNECTIONS) return -1;
    PyBulletConnection* conn = &g_pybullet_connections[connection_id - 1];
    if (conn->state != PYBULLET_CONNECTED || !state) return -1;
    memset(state, 0, sizeof(PyBulletJointState));

    fprintf(conn->process_stdin,
        "{\"action\":\"get_joint\",\"robot\":%d,\"joint\":%d}\n",
        robot_id, joint_index);
    fflush(conn->process_stdin);

    char buf[512];
    if (conn->process_stdout && fgets(buf, (int)sizeof(buf), conn->process_stdout)) {
        float pos = 0.0f, vel = 0.0f, tor = 0.0f, tpos = 0.0f, tvel = 0.0f;
        if (sscanf(buf, "{\"pos\":%f,\"vel\":%f,\"torque\":%f,\"tpos\":%f,\"tvel\":%f}",
            &pos, &vel, &tor, &tpos, &tvel) >= 2) {
            state->position = pos;
            state->velocity = vel;
            state->torque = tor;
            state->target_position = tpos;
            state->target_velocity = tvel;
        }
    }
    return 0;
}

int pybullet_set_joint_control(int connection_id, int robot_id,
                               const int* joint_indices,
                               const float* target_positions,
                               int num_joints) {
    if (connection_id < 1 || connection_id > PYBULLET_MAX_CONNECTIONS) return -1;
    PyBulletConnection* conn = &g_pybullet_connections[connection_id - 1];
    if (conn->state != PYBULLET_CONNECTED) return -1;
    if (!joint_indices || !target_positions || num_joints <= 0) return -1;
    
    /* 构建关节控制命令 */
    fprintf(conn->process_stdin, "{\"action\":\"set_joints\",\"robot\":%d,\"joints\":[",
            robot_id);
    for (int i = 0; i < num_joints; i++) {
        fprintf(conn->process_stdin, "%s{\"idx\":%d,\"pos\":%f}",
                (i > 0) ? "," : "", joint_indices[i], target_positions[i]);
    }
    fprintf(conn->process_stdin, "]}\n");
    fflush(conn->process_stdin);
    return 0;
}

int pybullet_get_base_state(int connection_id, int robot_id,
                            PyBulletBaseState* state) {
    if (connection_id < 1 || connection_id > PYBULLET_MAX_CONNECTIONS) return -1;
    PyBulletConnection* conn = &g_pybullet_connections[connection_id - 1];
    if (conn->state != PYBULLET_CONNECTED || !state) return -1;
    memset(state, 0, sizeof(PyBulletBaseState));

    fprintf(conn->process_stdin,
        "{\"action\":\"get_base\",\"robot\":%d}\n", robot_id);
    fflush(conn->process_stdin);

    char buf[1024];
    if (conn->process_stdout && fgets(buf, (int)sizeof(buf), conn->process_stdout)) {
        sscanf(buf, "{\"pos\":[%f,%f,%f],\"orn\":[%f,%f,%f,%f],\"lv\":[%f,%f,%f],\"av\":[%f,%f,%f]}",
            &state->position[0], &state->position[1], &state->position[2],
            &state->orientation[0], &state->orientation[1],
            &state->orientation[2], &state->orientation[3],
            &state->linear_velocity[0], &state->linear_velocity[1], &state->linear_velocity[2],
            &state->angular_velocity[0], &state->angular_velocity[1], &state->angular_velocity[2]);
    }
    return 0;
}

int pybullet_get_contacts(int connection_id, int robot_id,
                          PyBulletContactPoint* contacts, int max_contacts) {
    if (connection_id < 1 || connection_id > PYBULLET_MAX_CONNECTIONS) return -1;
    PyBulletConnection* conn = &g_pybullet_connections[connection_id - 1];
    if (conn->state != PYBULLET_CONNECTED || !contacts || max_contacts <= 0) return -1;
    memset(contacts, 0, (size_t)max_contacts * sizeof(PyBulletContactPoint));

    fprintf(conn->process_stdin,
        "{\"action\":\"get_contacts\",\"robot\":%d,\"max\":%d}\n", robot_id, max_contacts);
    fflush(conn->process_stdin);

    char buf[4096];
    if (conn->process_stdout && fgets(buf, (int)sizeof(buf), conn->process_stdout)) {
        int count = 0;
        char* p = buf;
        while (*p && count < max_contacts) {
            if (*p == '{') {
                int ba, bb; float px, py, pz, nx, ny, nz, f, dist;
                if (sscanf(p, "{\"a\":%d,\"b\":%d,\"p\":[%f,%f,%f],\"n\":[%f,%f,%f],\"f\":%f,\"d\":%f}",
                    &ba, &bb, &px, &py, &pz, &nx, &ny, &nz, &f, &dist) >= 8) {
                    contacts[count].body_a = ba;
                    contacts[count].body_b = bb;
                    contacts[count].position[0] = px;
                    contacts[count].position[1] = py;
                    contacts[count].position[2] = pz;
                    contacts[count].normal[0] = nx;
                    contacts[count].normal[1] = ny;
                    contacts[count].normal[2] = nz;
                    contacts[count].force = f;
                    contacts[count].distance = dist;
                    count++;
                }
            }
            p++;
        }
        return count;
    }
    return 0;
}

int pybullet_get_camera_image(int connection_id, int width, int height,
                              const float camera_position[3],
                              const float camera_target[3],
                              const float camera_up[3],
                              PyBulletCameraImage* image) {
    if (connection_id < 1 || connection_id > PYBULLET_MAX_CONNECTIONS) return -1;
    PyBulletConnection* conn = &g_pybullet_connections[connection_id - 1];
    if (conn->state != PYBULLET_CONNECTED || !image) return -1;
    memset(image, 0, sizeof(PyBulletCameraImage));

    fprintf(conn->process_stdin,
        "{\"action\":\"camera\",\"w\":%d,\"h\":%d,"
        "\"eye\":[%f,%f,%f],\"target\":[%f,%f,%f],\"up\":[%f,%f,%f]}\n",
        width, height,
        camera_position[0], camera_position[1], camera_position[2],
        camera_target[0], camera_target[1], camera_target[2],
        camera_up[0], camera_up[1], camera_up[2]);
    fflush(conn->process_stdin);

    /* 响应: {"status":"ok","w":640,"h":480,"rgb_size":921600,...}
     * 实际RGB数据通过base64编码后的额外行传输 */
    char buf[1024];
    if (conn->process_stdout && fgets(buf, (int)sizeof(buf), conn->process_stdout)) {
        int w, h, rgb_size, depth_size;
        if (sscanf(buf, "{\"status\":\"ok\",\"w\":%d,\"h\":%d,\"rgb_size\":%d,\"depth_size\":%d}",
            &w, &h, &rgb_size, &depth_size) == 4 && w > 0 && h > 0) {
            image->width = w;
            image->height = h;
            if (rgb_size > 0) {
                image->rgb_data = (unsigned char*)malloc((size_t)rgb_size);
                if (image->rgb_data) {
                    size_t read = 0;
                    while (read < (size_t)rgb_size && !feof(conn->process_stdout)) {
                        read += fread(image->rgb_data + read, 1, (size_t)rgb_size - read, conn->process_stdout);
                    }
                }
            }
            if (depth_size > 0) {
                image->depth_data = (float*)malloc((size_t)depth_size * sizeof(float));
                if (image->depth_data) {
                    size_t depth_read = 0;
                    while (depth_read < (size_t)depth_size && !feof(conn->process_stdout)) {
                        depth_read += fread(image->depth_data + depth_read, sizeof(float), (size_t)depth_size - depth_read, conn->process_stdout);
                    }
                }
            }
        }
    }
    return (image->rgb_data || image->depth_data) ? 0 : -1;
}

void pybullet_free_camera_image(PyBulletCameraImage* image) {
    if (!image) return;
    free(image->rgb_data);
    free(image->depth_data);
    free(image->segmentation_data);
    memset(image, 0, sizeof(PyBulletCameraImage));
}

int pybullet_reset_simulation(int connection_id) {
    if (connection_id < 1 || connection_id > PYBULLET_MAX_CONNECTIONS) return -1;
    PyBulletConnection* conn = &g_pybullet_connections[connection_id - 1];
    if (conn->state != PYBULLET_CONNECTED) return -1;
    
    fprintf(conn->process_stdin, "{\"action\":\"reset\"}\n");
    fflush(conn->process_stdin);
    return 0;
}

/* P2-003修复: pybullet_load_robot — pybullet_load_urdf的别名 */
int pybullet_load_robot(int connection_id, const char* urdf_path,
                        const float base_position[3],
                        const float base_orientation[4],
                        int use_fixed_base) {
    return pybullet_load_urdf(connection_id, urdf_path,
                              base_position, base_orientation, use_fixed_base);
}

/* P2-003修复: pybullet_step — pybullet_step_simulation的别名 */
int pybullet_step(int connection_id) {
    return pybullet_step_simulation(connection_id);
}

/* P2-003修复: pybullet_get_joint_states — 获取所有关节状态 */
int pybullet_get_joint_states(int connection_id, int robot_id,
                              PyBulletJointState* states, int max_joints) {
    if (connection_id < 1 || connection_id > PYBULLET_MAX_CONNECTIONS) return -1;
    PyBulletConnection* conn = &g_pybullet_connections[connection_id - 1];
    if (conn->state != PYBULLET_CONNECTED || !states || max_joints <= 0) return -1;

    /* 发送批量获取所有关节状态的命令 */
    fprintf(conn->process_stdin,
        "{\"action\":\"get_all_joints\",\"robot\":%d,\"max\":%d}\n",
        robot_id, max_joints);
    fflush(conn->process_stdin);

    char buf[8192];
    if (!conn->process_stdout || !fgets(buf, (int)sizeof(buf), conn->process_stdout)) {
        return 0;
    }

    /* 解析JSON数组格式: {"joints":[{"pos":...,"vel":...,...}, ...]} */
    int count = 0;
    const char* p = buf;
    const char* arr_start = strstr(p, "\"joints\":[");
    if (!arr_start) {
        /* 回退: 尝试逐行读取每个关节 */
        for (int j = 0; j < max_joints; j++) {
            PyBulletJointState js;
            memset(&js, 0, sizeof(PyBulletJointState));
            int ret = pybullet_get_joint_state(connection_id, robot_id, j, &js);
            if (ret < 0) break;
            states[count++] = js;
        }
        return count;
    }

    p = arr_start + 10; /* 跳过"joints":[ */
    while (*p && count < max_joints) {
        while (*p && *p != '{') p++;
        if (!*p) break;
        
        float pos = 0, vel = 0, tor = 0, tpos = 0, tvel = 0;
        if (sscanf(p, "{\"pos\":%f,\"vel\":%f,\"torque\":%f,\"tpos\":%f,\"tvel\":%f}",
            &pos, &vel, &tor, &tpos, &tvel) >= 2) {
            states[count].position = pos;
            states[count].velocity = vel;
            states[count].torque = tor;
            states[count].target_position = tpos;
            states[count].target_velocity = tvel;
            count++;
        }
        while (*p && *p != '}') p++;
        if (*p == '}') p++;
        while (*p && (*p == ',' || *p == ' ')) p++;
    }
    return count;
}

/* P2-003修复: pybullet_apply_torque — 施加关节力矩 */
int pybullet_apply_torque(int connection_id, int robot_id,
                          int joint_index, float torque) {
    if (connection_id < 1 || connection_id > PYBULLET_MAX_CONNECTIONS) return -1;
    PyBulletConnection* conn = &g_pybullet_connections[connection_id - 1];
    if (conn->state != PYBULLET_CONNECTED) return -1;

    /* 发送施加力矩命令到Python子进程 */
    fprintf(conn->process_stdin,
        "{\"action\":\"apply_torque\",\"robot\":%d,\"joint\":%d,\"torque\":%f}\n",
        robot_id, joint_index, (double)torque);
    fflush(conn->process_stdin);

    /* 读取确认响应 */
    char buf[256];
    if (conn->process_stdout && fgets(buf, (int)sizeof(buf), conn->process_stdout)) {
        if (strstr(buf, "\"status\":\"ok\"") || strstr(buf, "\"ok\"")) {
            return 0;
        }
    }
    return 0;
}
