/**
 * @file config_loader.c
 * @brief K-030: 系统配置文件加载与保存 (纯C, 零依赖)
 */

#include "selflnn/selflnn.h"
#include "selflnn/utils/json_parser.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/core/port_config.h"
#include "selflnn/backend/backend.h" /* SELFLNN_DEFAULT_PORT 别名 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
static SRWLOCK g_config_rwlock = SRWLOCK_INIT;
#define CONFIG_RDLOCK()  AcquireSRWLockShared(&g_config_rwlock)
#define CONFIG_RDUNLOCK() ReleaseSRWLockShared(&g_config_rwlock)
#define CONFIG_WRLOCK()  AcquireSRWLockExclusive(&g_config_rwlock)
#define CONFIG_WRUNLOCK() ReleaseSRWLockExclusive(&g_config_rwlock)
#else
#include <pthread.h>
static pthread_rwlock_t g_config_rwlock = PTHREAD_RWLOCK_INITIALIZER;
#define CONFIG_RDLOCK()  pthread_rwlock_rdlock(&g_config_rwlock)
#define CONFIG_RDUNLOCK() pthread_rwlock_unlock(&g_config_rwlock)
#define CONFIG_WRLOCK()  pthread_rwlock_wrlock(&g_config_rwlock)
#define CONFIG_WRUNLOCK() pthread_rwlock_unlock(&g_config_rwlock)
#endif

#define CONFIG_DEFAULT_PATH "config/system_config.json"

int selflnn_config_load_from_file(const char* filepath, SystemConfig* config) {
    if (!config) return -1;

    CONFIG_WRLOCK();

    const char* path = filepath ? filepath : CONFIG_DEFAULT_PATH;
    FILE* fp = fopen(path, "r");
    if (!fp) { log_warning("[配置] 文件不存在: %s", path); CONFIG_WRUNLOCK(); return -1; }

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    if (fsize <= 0 || fsize > 65536) { fclose(fp); CONFIG_WRUNLOCK(); return -1; }
    fseek(fp, 0, SEEK_SET);

    char* json_str = (char*)safe_malloc((size_t)fsize + 1);
    if (!json_str) { fclose(fp); CONFIG_WRUNLOCK(); return -1; }
    fread(json_str, 1, (size_t)fsize, fp);
    json_str[fsize] = '\0';
    fclose(fp);

    /* 使用项目内置JSON解析器进行结构化解析 */
    JsonValue* root = json_parse(json_str);
    if (!root) {
        log_error("[配置] JSON解析失败: %s", path);
        safe_free((void**)&json_str);
        CONFIG_WRUNLOCK();
        return -1;
    }

    const JsonValue* v;
    const char* str_val;

    v = json_get(root, "state_dimension");
    if (v && v->type == JSON_NUMBER) config->state_dimension = (int)v->data.number_val;
    if (config->state_dimension <= 0) config->state_dimension = 128;

    v = json_get(root, "multimodal_channels");
    if (v && v->type == JSON_NUMBER) config->multimodal_channels = (int)v->data.number_val;
    if (config->multimodal_channels <= 0) config->multimodal_channels = 9;

    v = json_get(root, "memory_capacity");
    if (v && v->type == JSON_NUMBER) config->memory_capacity = (int)v->data.number_val;
    if (config->memory_capacity <= 0) config->memory_capacity = 100000;

    v = json_get(root, "max_concurrent_tasks");
    if (v && v->type == JSON_NUMBER) config->max_concurrent_tasks = (int)v->data.number_val;
    if (config->max_concurrent_tasks <= 0) config->max_concurrent_tasks = 8;

    str_val = json_get_string(root, "power_mode");
    if (str_val) {
        if (strcmp(str_val, "performance") == 0)
            config->power_mode = POWER_MODE_PERFORMANCE;
        else if (strcmp(str_val, "power_saving") == 0)
            config->power_mode = POWER_MODE_POWER_SAVING;
        else config->power_mode = POWER_MODE_BALANCED;
    } else {
        config->power_mode = POWER_MODE_BALANCED;
    }

    str_val = json_get_string(root, "gpu_backend");
    if (str_val) {
        if (strcmp(str_val, "cuda") == 0) config->gpu_backend = GPU_BACKEND_CUDA;
        else if (strcmp(str_val, "opencl") == 0) config->gpu_backend = GPU_BACKEND_OPENCL;
        else if (strcmp(str_val, "vulkan") == 0) config->gpu_backend = GPU_BACKEND_VULKAN;
        else if (strcmp(str_val, "metal") == 0) config->gpu_backend = GPU_BACKEND_METAL;
        else if (strcmp(str_val, "rocm") == 0) config->gpu_backend = GPU_BACKEND_ROCM;
        else if (strcmp(str_val, "auto") == 0) config->gpu_backend = GPU_BACKEND_CPU;
        else config->gpu_backend = GPU_BACKEND_CPU;
    } else {
        config->gpu_backend = GPU_BACKEND_CPU;
    }

/* 加载与保存的字段集保持一致 */
    str_val = json_get_string(root, "model_path");
    if (str_val && config->model_path) {
        snprintf((char*)config->model_path, 1024, "%s", str_val);
    }

/* 加载端口字段（原只保存不加载） */
    v = json_get(root, "http_port");
    if (v && v->type == JSON_NUMBER) {
        int json_hp = (int)v->data.number_val;
        if (json_hp > 0 && json_hp <= 65535 && json_hp != SELFLNN_HTTP_PORT) {
            log_warning("[配置] system_config.json中http_port=%d与port_config.h中SELFLNN_HTTP_PORT=%d不一致，强制使用头文件定义值%d",
                        json_hp, SELFLNN_HTTP_PORT, SELFLNN_HTTP_PORT);
        }
    }
    config->http_port = SELFLNN_DEFAULT_PORT;

    v = json_get(root, "websocket_port");
    if (v && v->type == JSON_NUMBER) {
        int json_ws = (int)v->data.number_val;
        if (json_ws > 0 && json_ws <= 65535 && json_ws != SELFLNN_WEBSOCKET_PORT) {
            log_warning("[配置] system_config.json中websocket_port=%d与port_config.h中SELFLNN_WEBSOCKET_PORT=%d不一致，强制使用头文件定义值%d",
                        json_ws, SELFLNN_WEBSOCKET_PORT, SELFLNN_WEBSOCKET_PORT);
        }
    }
    config->websocket_port = SELFLNN_WEBSOCKET_PORT;

    v = json_get(root, "distributed_port");
    if (v && v->type == JSON_NUMBER) config->distributed_port = (int)v->data.number_val;
    if (config->distributed_port <= 0 || config->distributed_port > 65535) config->distributed_port = 8765;

    /* M-035修复: 从嵌套 training 节点读取混合精度配置
     * system_config.json: training.mixed_precision = "auto"/"fp16"/"bf16"/"off"
     * 映射到 SystemConfig.mixed_precision_mode: 0=关闭, 1=auto, 2=FP16, 3=BF16 */
    {
        const JsonValue* training_v = json_get(root, "training");
        if (training_v && training_v->type == JSON_OBJECT) {
            const char* mp_str = json_get_string(training_v, "mixed_precision");
            if (mp_str) {
                if (strcmp(mp_str, "auto") == 0)
                    config->mixed_precision_mode = 1;
                else if (strcmp(mp_str, "fp16") == 0 || strcmp(mp_str, "FP16") == 0)
                    config->mixed_precision_mode = 2;
                else if (strcmp(mp_str, "bf16") == 0 || strcmp(mp_str, "BF16") == 0)
                    config->mixed_precision_mode = 3;
                else if (strcmp(mp_str, "off") == 0 || strcmp(mp_str, "false") == 0)
                    config->mixed_precision_mode = 0;
                else
                    config->mixed_precision_mode = 1; /* 未知值默认auto */
                log_info("[M-035] JSON配置混合精度: %s → 模式=%d", mp_str, config->mixed_precision_mode);
            } else {
                config->mixed_precision_mode = 0;
            }
        } else {
            config->mixed_precision_mode = 0;
        }
    }

    json_free(root);
    safe_free((void**)&json_str);
    log_info("[配置] 加载成功: %s", path);

/* 配置Schema验证 */
    {
        char err[256];
        if (selflnn_config_validate(config, err, sizeof(err)) != 0) {
            log_warning("[配置] 验证警告: %s", err);
        }
    }

    CONFIG_WRUNLOCK();
    return 0;
}

/* ================================================================
 *: 配置Schema验证
 * ================================================================ */
int selflnn_config_validate(const SystemConfig* config, char* error_msg, size_t msg_size) {
    if (!config) {
        if (error_msg && msg_size > 0) snprintf(error_msg, msg_size, "config为NULL");
        return -1;
    }

    if (config->state_dimension < 1 || config->state_dimension > 4096) {
        if (error_msg && msg_size > 0)
            snprintf(error_msg, msg_size, "state_dimension=%d超出范围[1,4096]", config->state_dimension);
        return -1;
    }
    if (config->multimodal_channels < 1 || config->multimodal_channels > 256) {
        if (error_msg && msg_size > 0)
            snprintf(error_msg, msg_size, "multimodal_channels=%d超出范围[1,256]", config->multimodal_channels);
        return -1;
    }
    if (config->memory_capacity < 100 || config->memory_capacity > 100000000) {
        if (error_msg && msg_size > 0)
            snprintf(error_msg, msg_size, "memory_capacity=%d超出范围[100,100000000]", config->memory_capacity);
        return -1;
    }
    if (config->max_concurrent_tasks < 1 || config->max_concurrent_tasks > 1024) {
        if (error_msg && msg_size > 0)
            snprintf(error_msg, msg_size, "max_concurrent_tasks=%d超出范围[1,1024]", config->max_concurrent_tasks);
        return -1;
    }
    if (config->http_port > 0 && (config->http_port < 80 || config->http_port > 65535)) {
        if (error_msg && msg_size > 0)
            snprintf(error_msg, msg_size, "http_port=%d无效端口号", config->http_port);
        return -1;
    }
    if (config->websocket_port > 0 && (config->websocket_port < 80 || config->websocket_port > 65535)) {
        if (error_msg && msg_size > 0)
            snprintf(error_msg, msg_size, "websocket_port=%d无效端口号", config->websocket_port);
        return -1;
    }
    if (config->websocket_port > 0 && config->http_port > 0 &&
        config->websocket_port == config->http_port) {
        if (error_msg && msg_size > 0)
            snprintf(error_msg, msg_size, "http_port和websocket_port不能相同(%d)", config->http_port);
        return -1;
    }
    return 0;
}

int selflnn_config_save_to_file(const char* filepath, const SystemConfig* config) {
    if (!config) return -1;

    const char* path = filepath ? filepath : CONFIG_DEFAULT_PATH;
    FILE* fp = fopen(path, "w");
    if (!fp) { log_error("[配置] 无法创建: %s", path); return -1; }

    const char* pm = "balanced";
    if (config->power_mode == POWER_MODE_PERFORMANCE) pm = "performance";
    else if (config->power_mode == POWER_MODE_POWER_SAVING) pm = "power_saving";

/* GPU后端枚举补全 — 原只枚举cuda/opencl */
    const char* gb = "cpu";
    switch (config->gpu_backend) {
        case GPU_BACKEND_CUDA:     gb = "cuda"; break;
        case GPU_BACKEND_OPENCL:   gb = "opencl"; break;
        case GPU_BACKEND_VULKAN:   gb = "vulkan"; break;
        case GPU_BACKEND_METAL:    gb = "metal"; break;
        case GPU_BACKEND_ROCM:     gb = "rocm"; break;
        case GPU_BACKEND_ASCEND:   gb = "ascend"; break;
        case GPU_BACKEND_CAMBRICON:gb = "cambricon"; break;
        case GPU_BACKEND_TPU:      gb = "tpu"; break;
        case GPU_BACKEND_CPU:      gb = "cpu"; break;
        default:                   gb = "cpu"; break;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"state_dimension\": %d,\n", config->state_dimension);
    fprintf(fp, "  \"multimodal_channels\": %d,\n", config->multimodal_channels);
    fprintf(fp, "  \"memory_capacity\": %d,\n", config->memory_capacity);
    fprintf(fp, "  \"max_concurrent_tasks\": %d,\n", config->max_concurrent_tasks);
    fprintf(fp, "  \"power_mode\": \"%s\",\n", pm);
    fprintf(fp, "  \"gpu_backend\": \"%s\",\n", gb);
/* 端口号优先使用配置值，配置未设置时回退port_config.h */
    fprintf(fp, "  \"http_port\": %d,\n", config->http_port > 0 ? config->http_port : SELFLNN_DEFAULT_PORT);
    fprintf(fp, "  \"websocket_port\": %d,\n", config->websocket_port > 0 ? config->websocket_port : SELFLNN_WEBSOCKET_PORT);
    fprintf(fp, "  \"distributed_port\": %d,\n", config->distributed_port > 0 ? config->distributed_port : 8765);
    if (config->model_path && config->model_path[0]) {
        fprintf(fp, "  \"model_path\": \"%s\"\n", config->model_path);
    } else {
        fprintf(fp, "  \"model_path\": \"\"\n");
    }
    fprintf(fp, "}\n");

    fclose(fp);
    log_info("[配置] 保存成功: %s", path);
    return 0;
}
