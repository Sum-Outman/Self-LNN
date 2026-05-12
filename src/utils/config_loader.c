/**
 * @file config_loader.c
 * @brief K-030: 系统配置文件加载与保存 (纯C, 零依赖)
 */

#include "selflnn/selflnn.h"
#include "selflnn/utils/json_parser.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define CONFIG_DEFAULT_PATH "config/system_config.json"

int selflnn_config_load_from_file(const char* filepath, SystemConfig* config) {
    if (!config) return -1;

    const char* path = filepath ? filepath : CONFIG_DEFAULT_PATH;
    FILE* fp = fopen(path, "r");
    if (!fp) { log_warning("[配置] 文件不存在: %s", path); return -1; }

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    if (fsize <= 0 || fsize > 65536) { fclose(fp); return -1; }
    fseek(fp, 0, SEEK_SET);

    char* json_str = (char*)safe_malloc((size_t)fsize + 1);
    if (!json_str) { fclose(fp); return -1; }
    fread(json_str, 1, (size_t)fsize, fp);
    json_str[fsize] = '\0';
    fclose(fp);

    /* 使用项目内置JSON解析器进行结构化解析 */
    JsonValue* root = json_parse(json_str);
    if (!root) {
        log_error("[配置] JSON解析失败: %s", path);
        safe_free((void**)&json_str);
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
        else config->gpu_backend = GPU_BACKEND_CPU;
    } else {
        config->gpu_backend = GPU_BACKEND_CPU;
    }

    json_free(root);
    safe_free((void**)&json_str);
    log_info("[配置] 加载成功: %s", path);
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

    const char* gb = "cpu";
    if (config->gpu_backend == GPU_BACKEND_CUDA) gb = "cuda";
    if (config->gpu_backend == GPU_BACKEND_OPENCL) gb = "opencl";

    fprintf(fp, "{\n");
    fprintf(fp, "  \"state_dimension\": %d,\n", config->state_dimension);
    fprintf(fp, "  \"multimodal_channels\": %d,\n", config->multimodal_channels);
    fprintf(fp, "  \"memory_capacity\": %d,\n", config->memory_capacity);
    fprintf(fp, "  \"max_concurrent_tasks\": %d,\n", config->max_concurrent_tasks);
    fprintf(fp, "  \"power_mode\": \"%s\",\n", pm);
    fprintf(fp, "  \"gpu_backend\": \"%s\",\n", gb);
    fprintf(fp, "  \"http_port\": 8080,\n");
    fprintf(fp, "  \"websocket_port\": 9090,\n");
    fprintf(fp, "  \"distributed_port\": 8765\n");
    fprintf(fp, "}\n");

    fclose(fp);
    log_info("[配置] 保存成功: %s", path);
    return 0;
}
