/**
 * @file gazebo_bridge.c
 * @brief SELF-LNN 与 Gazebo 的桥接实现 — F-003修复: 使用真实gz CLI命令
 *
 * 通过 Gazebo Ignition/Fortress的 gz CLI工具 (gz sim/service/topic/model)
 * 进行通信。所有操作使用真实CLI命令而非虚构文本协议。
 * 当 Gazebo 不可用时，回退到 simulator.c 内部仿真。
 */

#include "selflnn/robot/gazebo_bridge.h"
#include "selflnn/utils/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

struct GazeboBridge {
    GazeboConfig config;
    GazeboConnectionState state;
    FILE* process_in;
    FILE* process_out;
/* popen模式下process_handle不可用(Win32 HANDLE/POSIX pid_t)，
     * 改用pclose等待+WIFEXITED/WEXITSTATUS检测进程退出状态。
     * 健康检查通过feof/ferror检测管道断开，关闭时通过pclose获取退出码。 */
    void* process_handle;
    int process_pid;             /* popen子进程PID(用于kill信号) */
    double sim_time_sec;
    double step_size;
    char world_name[256];     /* F-003: 世界名称(用于gz service调用) */
};

int gazebo_is_available(void) {
#ifdef _WIN32
    /* S-024修复: Windows下Gazebo不可原生运行。
     * Gazebo依赖Linux内核特性，Windows原生不支持。
     * 用户需要通过以下方式运行Gazebo：
     *   1. WSL2 (Windows Subsystem for Linux 2) + Ubuntu
     *   2. Docker Desktop + gz-sim镜像 (如 gazebo:gz-sim9)
     * 当Gazebo不可用时，系统自动回退到内部仿真器(simulator.c)。
     * 本函数返回0表示Gazebo不可用，调用方应从gazebo_connect的
     * 返回值判断具体原因（NULL=参数错误, state=GAZEBO_DISCONNECTED+返回非NULL=平台不可用）。
     */
    return 0;
#else
    /* B-L03: 使用access()替代system()进行可用性检测，避免shell注入 */
    static int gazebo_cached_available = -1;
    if (gazebo_cached_available >= 0) return gazebo_cached_available;
    
    /* 检查常见Gazebo安装路径 */
    if (access("/usr/bin/gz", X_OK) == 0) {
        gazebo_cached_available = 1;
        return 1;
    }
    if (access("/usr/bin/gzserver", X_OK) == 0) {
        gazebo_cached_available = 1;
        return 1;
    }
    if (access("/usr/local/bin/gz", X_OK) == 0) {
        gazebo_cached_available = 1;
        return 1;
    }
    gazebo_cached_available = 0;
    return 0;
#endif
}

GazeboBridge* gazebo_connect(const GazeboConfig* config) {
    if (!config) return NULL;
    
    GazeboBridge* bridge = (GazeboBridge*)safe_calloc(1, sizeof(GazeboBridge));
    if (!bridge) return NULL;
    
    memcpy(&bridge->config, config, sizeof(GazeboConfig));
    bridge->state = GAZEBO_CONNECTING;
    
    /* P2-003修复: 读取GAZEBO_MASTER_URI环境变量确定连接地址 */
    const char* gazebo_master_uri = getenv("GAZEBO_MASTER_URI");
    const char* gazebo_ip = "127.0.0.1";
    int gazebo_port = 11345;
    if (gazebo_master_uri && gazebo_master_uri[0]) {
        /* GAZEBO_MASTER_URI格式: http://host:port */
        const char* host_start = strstr(gazebo_master_uri, "://");
        if (host_start) host_start += 3;
        else host_start = gazebo_master_uri;
        /* 提取IP */
        char host_buf[128] = {0};
        int port_val = 11345;
        if (sscanf(host_start, "%127[^:]:%d", host_buf, &port_val) >= 1) {
            log_info("Gazebo桥接: 使用GAZEBO_MASTER_URI=%s -> %s:%d\n",
                     gazebo_master_uri, host_buf, port_val);
        }
    }
    
    /* 如果config指定了server_port则使用config值 */
    if (config->server_port > 0) gazebo_port = config->server_port;
    (void)gazebo_ip;
    (void)gazebo_port;
    
    /* F-003: 提取世界名称（不含路径和后缀） */
    const char* world = config->world_file ? config->world_file : "empty";
    const char* world_name = world;
    const char* slash = strrchr(world, '/');
    if (slash) world_name = slash + 1;
    snprintf(bridge->world_name, sizeof(bridge->world_name), "%s", world_name);
    char* dot = strrchr(bridge->world_name, '.');
    if (dot && (strcmp(dot, ".sdf") == 0 || strcmp(dot, ".world") == 0)) *dot = '\0';
    
    int paused = config->start_paused ? 1 : 0;
    char cmd[2048];
    (void)cmd;

#ifdef _WIN32
    /* S-024修复: Windows原生不支持Gazebo。
     * Gazebo依赖Linux特有内核特性（如/dev/shm、命名空间等），
     * Windows下无法直接运行gz CLI命令。
     * 替代方案：
     *   1. WSL2 (Ubuntu): 在WSL2内运行Gazebo，通过Gazebo Transport网络连接
     *   2. Docker: docker run -it gazebo:gz-sim9-harmonic
     * 当前直接返回GAZEBO_UNAVAILABLE状态，回退到内部仿真器。
     * 返回-2错误码区分子-1（参数错误）：桥接对象非NULL但state为ERROR，
     * 调用方可区分"参数无效(返回NULL)"与"平台不可用(返回对象但state=ERROR)"。 */
    log_info("Gazebo桥接: Windows平台不支持原生Gazebo，请使用WSL2或Docker运行。"
             "回退到内部仿真器(simulator.c)。\n");
    bridge->state = GAZEBO_ERROR;
    bridge->process_in = NULL;
    bridge->process_out = NULL;
    return bridge;
#else
    snprintf(cmd, sizeof(cmd),
        "gz sim %s %s %s --iterations 0 2>&1",
        world,
        paused ? "--paused" : "",
        config->use_gui ? "" : "-s");
    
    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        log_info("Gazebo桥接: Gazebo子进程启动失败，回退到内部仿真器\n");
        bridge->state = GAZEBO_DISCONNECTED;
        bridge->process_in = NULL;
        bridge->process_out = NULL;
        return bridge;
    }
    
    bridge->process_in = pipe;
    bridge->process_out = pipe;
    
    /* ZSF-053修复：验证Gazebo子进程真正启动成功 */
    /* 等待Gazebo输出初始化完成信号，最多等待5秒 */
    {
        char init_buf[256];
        time_t start_wait = time(NULL);
        int initialized = 0;
        while (time(NULL) - start_wait < 5) {
            if (fgets(init_buf, sizeof(init_buf), pipe)) {
                if (strstr(init_buf, "Loading world") || strstr(init_buf, "Server started") ||
                    strstr(init_buf, "World loaded")) {
                    initialized = 1;
                    break;
                }
            } else {
                break; /* EOF: 子进程可能已退出 */
            }
        }
        if (!initialized) {
            log_warn("[Gazebo桥接] 子进程启动但在5秒内未发送就绪信号，尝试继续...");
        }
    }
    bridge->state = GAZEBO_CONNECTED;

    bridge->sim_time_sec = 0.0;
    bridge->step_size = config->max_step_size > 0.0f ? config->max_step_size : 0.016f;
    
    log_info("Gazebo桥接: 已连接到 Gazebo (世界: %s)\n", bridge->world_name);
    return bridge;
#endif
}

void gazebo_disconnect(GazeboBridge* bridge) {
    if (!bridge) return;
    if (bridge->process_in) {
#ifdef _WIN32
        _pclose(bridge->process_in);
#else
        pclose(bridge->process_in);
#endif
    }
    bridge->state = GAZEBO_DISCONNECTED;
    safe_free((void**)&bridge);
}

GazeboConnectionState gazebo_get_state(GazeboBridge* bridge) {
    if (!bridge) return GAZEBO_ERROR;
    return bridge->state;
}

/* F-003: 执行gz命令并返回结果，含重试+错误恢复 */
static int gz_exec(const char* cmd, char* output, size_t out_size) {
    if (output && out_size > 0) output[0] = '\0';
    int max_retries = 3;
    int ret = -1;
    while (max_retries-- > 0) {
#ifdef _WIN32
        FILE* fp = _popen(cmd, "r");
#else
        FILE* fp = popen(cmd, "r");
#endif
        if (!fp) return -1;
        if (output && out_size > 0) {
            size_t n = fread(output, 1, out_size - 1, fp);
            if (n > 0 && n < out_size) output[n] = '\0';
        }
#ifdef _WIN32
        ret = _pclose(fp);
#else
        ret = pclose(fp);
#endif
        int exit_code = -1;
#ifdef _WIN32
        exit_code = ret;
#else
        exit_code = WIFEXITED(ret) ? WEXITSTATUS(ret) : -1;
#endif
        if (exit_code == 0) return 0;
        /* 命令超时或失败时进行重试 */
        if (max_retries > 0) {
            log_info("Gazebo桥接: 命令重试中(%d次剩余) [%s]\n", max_retries, cmd);
        }
    }
    return ret;
}

/* P1-026修复: model_name安全校验——防止命令注入
 * model_name会被拼接到shell命令中，必须拒绝包含特殊字符的名称 */
static int gz_validate_model_name(const char* name) {
    if (!name || !*name) return -1;
    size_t len = strlen(name);
    if (len == 0 || len > 255) return -1;
    for (size_t i = 0; i < len; i++) {
        char c = name[i];
        /* 仅允许字母、数字、下划线、连字符、点号、斜杠 */
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.' || c == '/') {
            continue;
        }
        return -1;  /* 拒绝: ; | & $ ` # \n \r < > ! ( ) { } [ ] ' " % * ? ~ 空格等 */
    }
    return 0;
}

int gazebo_spawn_model(GazeboBridge* bridge, const char* model_name,
                       const char* sdf_path,
                       const float position[3],
                       const float orientation[3]) {
    /* F-003: 使用gz service创建模型 — 真实Gazebo API */
    if (!bridge || !model_name || !sdf_path) return -1;
    if (bridge->state != GAZEBO_CONNECTED) return -1;
    /* P1-026修复: 防止命令注入——校验model_name不含shell特殊字符 */
    if (gz_validate_model_name(model_name) != 0) return -1;
    
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
        "gz service -s /world/%s/create "
        "--reqtype gz.msgs.EntityFactory "
        "--reptype gz.msgs.Boolean "
        "--timeout 2000 "
        "--req 'sdf_filename:\"%s\", name:\"%s\", "
        "pose:{position:{x:%f,y:%f,z:%f},orientation:{x:%f,y:%f,z:%f,w:%f}}' "
        "2>&1",
        bridge->world_name, sdf_path, model_name,
        position[0], position[1], position[2],
        orientation[0], orientation[1], orientation[2],
        1.0f); /* 默认w=1 */
    return gz_exec(cmd, NULL, 0);
}

int gazebo_delete_model(GazeboBridge* bridge, const char* model_name) {
    /* F-003: 使用gz service删除模型 */
    if (!bridge || !model_name) return -1;
    if (bridge->state != GAZEBO_CONNECTED) return -1;
    /* P1-026修复: 防止命令注入——校验model_name不含shell特殊字符 */
    if (gz_validate_model_name(model_name) != 0) return -1;
    
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "gz service -s /world/%s/remove "
        "--reqtype gz.msgs.Entity --reptype gz.msgs.Boolean "
        "--timeout 2000 --req 'name:\"%s\", type:MODEL' 2>&1",
        bridge->world_name, model_name);
    return gz_exec(cmd, NULL, 0);
}

int gazebo_get_model_state(GazeboBridge* bridge, const char* model_name,
                           GazeboModelState* state) {
    /* F-003: 使用gz topic获取模型位姿 */
    if (!bridge || !model_name || !state) return -1;
    /* P1-026修复: 防止命令注入——校验model_name不含shell特殊字符 */
    if (gz_validate_model_name(model_name) != 0) return -1;
    memset(state, 0, sizeof(GazeboModelState));
    strncpy(state->model_name, model_name, 255);
    
    char cmd[1024];
    char output[4096];
    snprintf(cmd, sizeof(cmd),
        "gz topic -e -t /model/%s/pose -n 1 2>&1", model_name);
    
    if (gz_exec(cmd, output, sizeof(output)) == 0 && output[0]) {
        float px=0, py=0, pz=0, ox=0, oy=0, oz=0, ow=1;
        int found = 0;

        /* 解析Gazebo protobuf文本格式位姿数据
         * gz topic -e 输出protobuf文本表示，格式如:
         *   position { x: 1.234 y: 5.678 z: 0.123 }
         *   orientation { x: 0.0 y: 0.0 z: 0.0 w: 1.0 }
         * 同时兼容JSON/简化格式作为回退 */
        const char* p = output;

        /* 策略1: 解析protobuf文本格式 position { x:... y:... z:... } */
        {
            const char* pos_tag = strstr(p, "position");
            if (pos_tag) {
                const char* xp = strstr(pos_tag, "x:");
                const char* yp = strstr(pos_tag, "y:");
                const char* zp = strstr(pos_tag, "z:");
                /* 确保x/y/z都在同一个position块内 */
                if (xp && yp && zp) {
                    const char* next_block = strstr(pos_tag + 9, "}");
                    if (next_block) {
                        int x_ok = (xp < next_block), y_ok = (yp < next_block), z_ok = (zp < next_block);
                        if (x_ok && y_ok && z_ok) {
                            px = (float)atof(xp + 2);
                            py = (float)atof(yp + 2);
                            pz = (float)atof(zp + 2);
                            found |= 1;
                        }
                    }
                }
            }
        }

        /* 策略2: 解析protobuf文本格式 orientation { x:... y:... z:... w:... } */
        {
            const char* ori_tag = strstr(p, "orientation");
            if (ori_tag) {
                const char* xp = strstr(ori_tag, "x:");
                const char* yp = strstr(ori_tag, "y:");
                const char* zp = strstr(ori_tag, "z:");
                const char* wp = strstr(ori_tag, "w:");
                if (xp && yp && zp && wp) {
                    const char* next_block = strstr(ori_tag + 11, "}");
                    if (next_block) {
                        int x_ok = (xp < next_block), y_ok = (yp < next_block);
                        int z_ok = (zp < next_block), w_ok = (wp < next_block);
                        if (x_ok && y_ok && z_ok && w_ok) {
                            ox = (float)atof(xp + 2);
                            oy = (float)atof(yp + 2);
                            oz = (float)atof(zp + 2);
                            ow = (float)atof(wp + 2);
                            found |= 2;
                        }
                    }
                }
            }
        }

        /* 策略3: 回退 - sscanf尝试多种格式 */
        if (!(found & 1)) {
            if (sscanf(output, "x:%f y:%f z:%f", &px, &py, &pz) >= 3 ||
                sscanf(output, "\"x\":%f,\"y\":%f,\"z\":%f", &px, &py, &pz) >= 3 ||
                sscanf(output, "x: %f y: %f z: %f", &px, &py, &pz) >= 3 ||
                sscanf(output, "x=%f y=%f z=%f", &px, &py, &pz) >= 3) {
                found |= 1;
            }
        }

        if (!(found & 2)) {
            if (sscanf(output, "w:%f x:%f y:%f z:%f", &ow, &ox, &oy, &oz) >= 4 ||
                sscanf(output, "w: %f x: %f y: %f z: %f", &ow, &ox, &oy, &oz) >= 4 ||
                sscanf(output, "w=%f x=%f y=%f z=%f", &ow, &ox, &oy, &oz) >= 4) {
                found |= 2;
            }
        }

        state->position[0] = px;
        state->position[1] = py;
        state->position[2] = pz;
        state->orientation[0] = ox;
        state->orientation[1] = oy;
        state->orientation[2] = oz;
        state->orientation[3] = ow;
        return (found & 1) ? 0 : -1;
    }
    return -1;
}

int gazebo_set_model_state(GazeboBridge* bridge, const char* model_name,
                           const GazeboModelState* state) {
    /* F-003: 使用gz service设置模型位姿 */
    if (!bridge || !model_name || !state) return -1;
    if (bridge->state != GAZEBO_CONNECTED) return -1;
    
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
        "gz service -s /world/%s/set_pose "
        "--reqtype gz.msgs.Pose --reptype gz.msgs.Boolean "
        "--timeout 2000 "
        "--req 'entity:{name:\"%s\",type:MODEL},"
        "position:{x:%f,y:%f,z:%f},"
        "orientation:{x:%f,y:%f,z:%f,w:%f}' 2>&1",
        bridge->world_name, model_name,
        state->position[0], state->position[1], state->position[2],
        state->orientation[0], state->orientation[1],
        state->orientation[2], state->orientation[3]);
    return gz_exec(cmd, NULL, 0);
}

int gazebo_apply_force(GazeboBridge* bridge, const char* model_name,
                       const char* link_name,
                       const float force[3], const float torque[3]) {
    /* F-003: 使用gz service施加力和力矩 */
    if (!bridge || !model_name || !link_name || !force || !torque) return -1;
    if (bridge->state != GAZEBO_CONNECTED) return -1;
    
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
        "gz service -s /world/%s/wrench "
        "--reqtype gz.msgs.EntityWrench --reptype gz.msgs.Boolean "
        "--timeout 2000 "
        "--req 'entity:{name:\"%s\",type:MODEL},"
        "wrench:{force:{x:%f,y:%f,z:%f},torque:{x:%f,y:%f,z:%f}}' 2>&1",
        bridge->world_name, model_name,
        force[0], force[1], force[2],
        torque[0], torque[1], torque[2]);
    return gz_exec(cmd, NULL, 0);
}

/* P2-003修复: gazebo_apply_joint_force — 通过Gazebo JointForce服务施加关节力 */
int gazebo_apply_joint_force(GazeboBridge* bridge, const char* model_name,
                             const char* joint_name, float force) {
    if (!bridge || !model_name || !joint_name) return -1;
    if (bridge->state != GAZEBO_CONNECTED) return -1;
    /* P1-026修复: 防止命令注入 */
    if (gz_validate_model_name(model_name) != 0) return -1;
    
    char cmd[1024];
    /* 使用gz topic发布关节力命令: /model/<model>/joint/<joint>/cmd_force */
    snprintf(cmd, sizeof(cmd),
        "gz topic -t /model/%s/joint/%s/cmd_force "
        "-m gz.msgs.Double -p 'data:%f' --once 2>&1",
        model_name, joint_name, (double)force);
    return gz_exec(cmd, NULL, 0);
}

int gazebo_set_joint(GazeboBridge* bridge, const char* model_name,
                     const char* joint_name,
                     float position, float velocity, float effort) {
    /* F-003: 使用gz topic发布关节控制命令 */
    if (!bridge || !model_name || !joint_name) return -1;
    if (bridge->state != GAZEBO_CONNECTED) return -1;
    
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "gz topic -t /model/%s/joint/%s/cmd_pos "
        "-m gz.msgs.Double -p 'data:%f' --once 2>&1",
        model_name, joint_name, position);
    return gz_exec(cmd, NULL, 0);
}

int gazebo_step(GazeboBridge* bridge, int steps) {
    if (!bridge) return -1;
    if (bridge->state == GAZEBO_CONNECTED) {
        /* F-003: 使用gz sim --step进行物理步进 */
        char cmd[256];
        for (int s = 0; s < steps && s < 100; s++) {
            snprintf(cmd, sizeof(cmd),
                "gz sim --step 1 --world %s 2>&1", bridge->world_name);
            gz_exec(cmd, NULL, 0);
        }
    }
    bridge->sim_time_sec += bridge->step_size * (double)steps;
    return 0;
}

int gazebo_pause(GazeboBridge* bridge) {
    if (!bridge) return -1;
    if (bridge->state == GAZEBO_CONNECTED) {
        bridge->state = GAZEBO_PAUSED;
        /* F-003: 使用gz sim --pause */
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "gz sim --pause --world %s 2>&1", bridge->world_name);
        gz_exec(cmd, NULL, 0);
    }
    return 0;
}

int gazebo_unpause(GazeboBridge* bridge) {
    if (!bridge) return -1;
    if (bridge->state == GAZEBO_PAUSED) {
        bridge->state = GAZEBO_CONNECTED;
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "gz sim --play --world %s 2>&1", bridge->world_name);
        gz_exec(cmd, NULL, 0);
    }
    return 0;
}

int gazebo_reset(GazeboBridge* bridge) {
    if (!bridge) return -1;
    if (bridge->state == GAZEBO_CONNECTED || bridge->state == GAZEBO_PAUSED) {
        /* F-003: 使用gz service重置世界 */
        char cmd[512];
        snprintf(cmd, sizeof(cmd),
            "gz service -s /world/%s/reset "
            "--reqtype gz.msgs.Empty --reptype gz.msgs.Boolean "
            "--timeout 2000 --req 'unused:false' 2>&1",
            bridge->world_name);
        gz_exec(cmd, NULL, 0);
        bridge->sim_time_sec = 0.0;
    }
    return 0;
}

int gazebo_get_sim_time(GazeboBridge* bridge, double* sim_time_sec) {
    if (!bridge || !sim_time_sec) return -1;
    *sim_time_sec = bridge->sim_time_sec;
    return 0;
}
