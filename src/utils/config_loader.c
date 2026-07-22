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

/* P0修复: double→int安全转换，钳制到INT32范围防止转换溢出导致的未定义行为。
 * C标准规定浮点转整数若超出目标类型范围属未定义行为(UB)，恶意/异常JSON
 * 配置(如1e30)可能触发UB，故统一在此钳制。 */
static int json_to_int_clamped(double val) {
    if (val > 2147483647.0) val = 2147483647.0;
    if (val < -2147483648.0) val = -2147483648.0;
    return (int)val;
}

int selflnn_config_load_from_file(const char* filepath, SystemConfig* config) {
    if (!config) return -1;

    CONFIG_WRLOCK();

    const char* path = filepath ? filepath : CONFIG_DEFAULT_PATH;
    FILE* fp = fopen(path, "rb");
    if (!fp) { log_warning("[配置] 文件不存在: %s", path); CONFIG_WRUNLOCK(); return -1; }

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    if (fsize <= 0 || fsize > 65536) { fclose(fp); CONFIG_WRUNLOCK(); return -1; }
    fseek(fp, 0, SEEK_SET);

    char* json_str = (char*)safe_malloc((size_t)fsize + 1);
    if (!json_str) { fclose(fp); CONFIG_WRUNLOCK(); return -1; }
    /* P0修复: 检查fread返回值，确保配置文件数据完整读取。
     * 此前直接忽略返回值，若读取不完整(磁盘错误/管道截断)会导致JSON解析
     * 在残缺数据上产生误导性错误。 */
    size_t bytes_read = fread(json_str, 1, (size_t)fsize, fp);
    if (bytes_read != (size_t)fsize) {
        log_error("[配置] 文件读取不完整: 实际%zu/%ld字节", bytes_read, fsize);
        safe_free((void**)&json_str);
        fclose(fp);
        CONFIG_WRUNLOCK();
        return -1;
    }
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
    /* P0修复: 使用json_to_int_clamped钳制double→int范围，防止溢出UB */
    if (v && v->type == JSON_NUMBER) config->state_dimension = json_to_int_clamped(v->data.number_val);
    if (config->state_dimension <= 0) config->state_dimension = 256;

    v = json_get(root, "multimodal_channels");
    if (v && v->type == JSON_NUMBER) config->multimodal_channels = json_to_int_clamped(v->data.number_val);
    if (config->multimodal_channels <= 0) config->multimodal_channels = 64;

    v = json_get(root, "memory_capacity");
    if (v && v->type == JSON_NUMBER) config->memory_capacity = json_to_int_clamped(v->data.number_val);
    if (config->memory_capacity <= 0) config->memory_capacity = 10000;

    v = json_get(root, "max_concurrent_tasks");
    if (v && v->type == JSON_NUMBER) config->max_concurrent_tasks = json_to_int_clamped(v->data.number_val);
    if (config->max_concurrent_tasks <= 0) config->max_concurrent_tasks = 100;

    str_val = json_get_string(root, "power_mode");
    if (str_val) {
        if (strcmp(str_val, "performance") == 0 || strcmp(str_val, "high") == 0)
            config->power_mode = POWER_MODE_PERFORMANCE;
        else if (strcmp(str_val, "power_saving") == 0 || strcmp(str_val, "low") == 0)
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
        else if (strcmp(str_val, "auto") == 0) config->gpu_backend = GPU_BACKEND_AUTO;
        else if (strcmp(str_val, "ascend") == 0) config->gpu_backend = GPU_BACKEND_ASCEND;
        else if (strcmp(str_val, "cambricon") == 0) config->gpu_backend = GPU_BACKEND_CAMBRICON;
        else if (strcmp(str_val, "tpu") == 0) config->gpu_backend = GPU_BACKEND_TPU;
        else if (strcmp(str_val, "intel") == 0) config->gpu_backend = GPU_BACKEND_INTEL;
        else config->gpu_backend = GPU_BACKEND_CPU;
    } else {
        config->gpu_backend = GPU_BACKEND_AUTO;  /* P0-03修复: 默认自动检测GPU后端 */
    }

/* v9.25修复: model_path为const char*指针，原snprintf写入1024字节存在缓冲区溢出风险。
     * 改用safe_strdup安全分配，确保缓冲区大小精确匹配字符串长度。 */
    str_val = json_get_string(root, "model_path");
    if (str_val) {
        safe_free((void**)&config->model_path);
        config->model_path = safe_strdup(str_val);
    }

/* 加载端口字段（原只保存不加载） */
    v = json_get(root, "http_port");
    if (v && v->type == JSON_NUMBER) {
        int json_hp = json_to_int_clamped(v->data.number_val);   /* P0修复: 钳制防止溢出UB */
        if (json_hp > 0 && json_hp <= 65535) {
            /* 1.4修复: 使用JSON文件中的端口值，不再强制覆盖为硬编码默认值 */
            config->http_port = json_hp;
        } else {
            config->http_port = SELFLNN_DEFAULT_PORT;
        }
    } else {
        config->http_port = SELFLNN_DEFAULT_PORT;
    }

    v = json_get(root, "websocket_port");
    if (v && v->type == JSON_NUMBER) {
        int json_ws = json_to_int_clamped(v->data.number_val);   /* P0修复: 钳制防止溢出UB */
        if (json_ws > 0 && json_ws <= 65535 && json_ws != config->http_port) {  /* 不与HTTP端口冲突 */
            /* 1.4修复: 使用JSON文件中的WebSocket端口值 */
            config->websocket_port = json_ws;
        } else {
            config->websocket_port = SELFLNN_WEBSOCKET_PORT;
        }
    } else {
        config->websocket_port = SELFLNN_WEBSOCKET_PORT;
    }

    v = json_get(root, "distributed_port");
    if (v && v->type == JSON_NUMBER) config->distributed_port = json_to_int_clamped(v->data.number_val);   /* P0修复: 钳制防止溢出UB */
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

    /* P1-04修复: 解析 safety 嵌套段 */
    {
        const JsonValue* safety_v = json_get(root, "safety");
        if (safety_v && safety_v->type == JSON_OBJECT) {
            const JsonValue* v = json_get(safety_v, "content_filter_enabled");
            if (v && (v->type == JSON_NUMBER)) {
                config->safety_content_filter_enabled = (v->data.number_val != 0) ? 1 : 0;
            } else {
                config->safety_content_filter_enabled = 1; /* 默认启用 */
            }

            v = json_get(safety_v, "audit_logging_enabled");
            if (v && (v->type == JSON_NUMBER)) {
                config->safety_audit_logging_enabled = (v->data.number_val != 0) ? 1 : 0;
            } else {
                config->safety_audit_logging_enabled = 1; /* 默认启用 */
            }

            v = json_get(safety_v, "emergency_stop_enabled");
            if (v && (v->type == JSON_NUMBER)) {
                config->safety_emergency_stop_enabled = (v->data.number_val != 0) ? 1 : 0;
            } else {
                config->safety_emergency_stop_enabled = 1; /* 默认启用 */
            }

            v = json_get(safety_v, "behavior_constraints_strictness");
            if (v && v->type == JSON_NUMBER) {
                config->safety_behavior_constraints_strictness = (float)v->data.number_val;
            } else {
                config->safety_behavior_constraints_strictness = 1.0f; /* 默认严格度 */
            }

            v = json_get(safety_v, "max_violations_per_hour");
            if (v && v->type == JSON_NUMBER) {
                config->safety_max_violations_per_hour = json_to_int_clamped(v->data.number_val);
            } else {
                config->safety_max_violations_per_hour = 10; /* 默认值 */
            }

            v = json_get(safety_v, "circuit_breaker_threshold");
            if (v && v->type == JSON_NUMBER) {
                config->safety_circuit_breaker_threshold = json_to_int_clamped(v->data.number_val);
            } else {
                config->safety_circuit_breaker_threshold = 5; /* 默认值 */
            }

            v = json_get(safety_v, "circuit_breaker_cooldown_sec");
            if (v && v->type == JSON_NUMBER) {
                config->safety_circuit_breaker_cooldown_sec = json_to_int_clamped(v->data.number_val);
            } else {
                config->safety_circuit_breaker_cooldown_sec = 300; /* 默认5分钟冷却 */
            }

            log_info("[配置] safety段解析: content_filter=%d, audit_log=%d, emergency_stop=%d",
                     config->safety_content_filter_enabled, config->safety_audit_logging_enabled,
                     config->safety_emergency_stop_enabled);
        } else {
            /* 设置默认值 */
            config->safety_content_filter_enabled = 1;
            config->safety_audit_logging_enabled = 1;
            config->safety_emergency_stop_enabled = 1;
            config->safety_behavior_constraints_strictness = 1.0f;
            config->safety_max_violations_per_hour = 10;
            config->safety_circuit_breaker_threshold = 5;
            config->safety_circuit_breaker_cooldown_sec = 300;
        }
    }

    /* P1-04修复: 解析 evolution 嵌套段 */
    {
        const JsonValue* evol_v = json_get(root, "evolution");
        if (evol_v && evol_v->type == JSON_OBJECT) {
            const JsonValue* v = json_get(evol_v, "population_size");
            if (v && v->type == JSON_NUMBER) {
                config->evolution_population_size = json_to_int_clamped(v->data.number_val);
            } else {
                config->evolution_population_size = 100;
            }

            v = json_get(evol_v, "mutation_rate");
            if (v && v->type == JSON_NUMBER) {
                config->evolution_mutation_rate = (float)v->data.number_val;
            } else {
                config->evolution_mutation_rate = 0.1f;
            }

            v = json_get(evol_v, "crossover_rate");
            if (v && v->type == JSON_NUMBER) {
                config->evolution_crossover_rate = (float)v->data.number_val;
            } else {
                config->evolution_crossover_rate = 0.8f;
            }

            v = json_get(evol_v, "elite_count");
            if (v && v->type == JSON_NUMBER) {
                config->evolution_elite_count = json_to_int_clamped(v->data.number_val);
            } else {
                config->evolution_elite_count = 5;
            }

            v = json_get(evol_v, "generations");
            if (v && v->type == JSON_NUMBER) {
                config->evolution_generations = json_to_int_clamped(v->data.number_val);
            } else {
                config->evolution_generations = 50;
            }

            v = json_get(evol_v, "island_count");
            if (v && v->type == JSON_NUMBER) {
                config->evolution_island_count = json_to_int_clamped(v->data.number_val);
            } else {
                config->evolution_island_count = 1;
            }

            v = json_get(evol_v, "migration_interval");
            if (v && v->type == JSON_NUMBER) {
                config->evolution_migration_interval = json_to_int_clamped(v->data.number_val);
            } else {
                config->evolution_migration_interval = 10;
            }

            log_info("[配置] evolution段解析: population=%d, mutation_rate=%.3f",
                     config->evolution_population_size, config->evolution_mutation_rate);
        } else {
            /* 设置默认值 */
            config->evolution_population_size = 100;
            config->evolution_mutation_rate = 0.1f;
            config->evolution_crossover_rate = 0.8f;
            config->evolution_elite_count = 5;
            config->evolution_generations = 50;
            config->evolution_island_count = 1;
            config->evolution_migration_interval = 10;
        }
    }

    /* P1-04修复: 解析 laplace 嵌套段 */
    {
        const JsonValue* lap_v = json_get(root, "laplace");
        if (lap_v && lap_v->type == JSON_OBJECT) {
            const JsonValue* v = json_get(lap_v, "freq_bands");
            if (v && v->type == JSON_NUMBER) {
                config->laplace_freq_bands = json_to_int_clamped(v->data.number_val);
            } else {
                config->laplace_freq_bands = 32;
            }

            v = json_get(lap_v, "stability_threshold");
            if (v && v->type == JSON_NUMBER) {
                config->laplace_stability_threshold = (float)v->data.number_val;
            } else {
                config->laplace_stability_threshold = 0.85f;
            }

            v = json_get(lap_v, "spectral_efficiency_target");
            if (v && v->type == JSON_NUMBER) {
                config->laplace_spectral_efficiency_target = (float)v->data.number_val;
            } else {
                config->laplace_spectral_efficiency_target = 0.7f;
            }

            v = json_get(lap_v, "update_interval_sec");
            if (v && v->type == JSON_NUMBER) {
                config->laplace_update_interval_sec = json_to_int_clamped(v->data.number_val);
            } else {
                config->laplace_update_interval_sec = 1800; /* 30分钟 */
            }

            log_info("[配置] laplace段解析: freq_bands=%d, stability=%.3f",
                     config->laplace_freq_bands, config->laplace_stability_threshold);
        } else {
            /* 设置默认值 */
            config->laplace_freq_bands = 32;
            config->laplace_stability_threshold = 0.85f;
            config->laplace_spectral_efficiency_target = 0.7f;
            config->laplace_update_interval_sec = 1800;
        }
    }

    /* P1-04修复: 解析 agi 嵌套段 */
    {
        const JsonValue* agi_v = json_get(root, "agi");
        if (agi_v && agi_v->type == JSON_OBJECT) {
            const JsonValue* v = json_get(agi_v, "self_evolution_enabled");
            if (v && (v->type == JSON_NUMBER)) {
                config->agi_self_evolution_enabled = (v->data.number_val != 0) ? 1 : 0;
            } else {
                config->agi_self_evolution_enabled = 0; /* 默认关闭 */
            }

            v = json_get(agi_v, "self_learning_enabled");
            if (v && (v->type == JSON_NUMBER)) {
                config->agi_self_learning_enabled = (v->data.number_val != 0) ? 1 : 0;
            } else {
                config->agi_self_learning_enabled = 1; /* 默认启用 */
            }

            v = json_get(agi_v, "self_decision_enabled");
            if (v && (v->type == JSON_NUMBER)) {
                config->agi_self_decision_enabled = (v->data.number_val != 0) ? 1 : 0;
            } else {
                config->agi_self_decision_enabled = 1; /* 默认启用 */
            }

            v = json_get(agi_v, "self_execution_enabled");
            if (v && (v->type == JSON_NUMBER)) {
                config->agi_self_execution_enabled = (v->data.number_val != 0) ? 1 : 0;
            } else {
                config->agi_self_execution_enabled = 1; /* 默认启用 */
            }

            v = json_get(agi_v, "imitation_learning_enabled");
            if (v && (v->type == JSON_NUMBER)) {
                config->agi_imitation_learning_enabled = (v->data.number_val != 0) ? 1 : 0;
            } else {
                config->agi_imitation_learning_enabled = 0; /* 默认关闭 */
            }

            v = json_get(agi_v, "self_correction_enabled");
            if (v && (v->type == JSON_NUMBER)) {
                config->agi_self_correction_enabled = (v->data.number_val != 0) ? 1 : 0;
            } else {
                config->agi_self_correction_enabled = 1; /* 默认启用 */
            }

            v = json_get(agi_v, "reflection_enabled");
            if (v && (v->type == JSON_NUMBER)) {
                config->agi_reflection_enabled = (v->data.number_val != 0) ? 1 : 0;
            } else {
                config->agi_reflection_enabled = 1; /* 默认启用 */
            }

            v = json_get(agi_v, "planning_enabled");
            if (v && (v->type == JSON_NUMBER)) {
                config->agi_planning_enabled = (v->data.number_val != 0) ? 1 : 0;
            } else {
                config->agi_planning_enabled = 1; /* 默认启用 */
            }

            v = json_get(agi_v, "background_loop_interval_sec");
            if (v && v->type == JSON_NUMBER) {
                config->agi_background_loop_interval_sec = json_to_int_clamped(v->data.number_val);
            } else {
                config->agi_background_loop_interval_sec = 10; /* 默认10秒 */
            }

            log_info("[配置] agi段解析: self_evolution=%d, self_learning=%d, planning=%d",
                     config->agi_self_evolution_enabled, config->agi_self_learning_enabled,
                     config->agi_planning_enabled);
        } else {
            /* 设置默认值 */
            config->agi_self_evolution_enabled = 0;
            config->agi_self_learning_enabled = 1;
            config->agi_self_decision_enabled = 1;
            config->agi_self_execution_enabled = 1;
            config->agi_imitation_learning_enabled = 0;
            config->agi_self_correction_enabled = 1;
            config->agi_reflection_enabled = 1;
            config->agi_planning_enabled = 1;
            config->agi_background_loop_interval_sec = 10;
        }
    }

    /* P1-04修复: 解析 self_cognition 嵌套段 */
    {
        const JsonValue* sc_v = json_get(root, "self_cognition");
        if (sc_v && sc_v->type == JSON_OBJECT) {
            const JsonValue* v = json_get(sc_v, "enabled");
            if (v && (v->type == JSON_NUMBER)) {
                config->self_cognition_enabled = (v->data.number_val != 0) ? 1 : 0;
            } else {
                config->self_cognition_enabled = 1; /* 默认启用 */
            }

            v = json_get(sc_v, "metacognition_enabled");
            if (v && (v->type == JSON_NUMBER)) {
                config->self_cognition_metacognition_enabled = (v->data.number_val != 0) ? 1 : 0;
            } else {
                config->self_cognition_metacognition_enabled = 1; /* 默认启用 */
            }

            v = json_get(sc_v, "self_reflection_interval_minutes");
            if (v && v->type == JSON_NUMBER) {
                config->self_cognition_reflection_interval_min = json_to_int_clamped(v->data.number_val);
            } else {
                config->self_cognition_reflection_interval_min = 30; /* 默认30分钟 */
            }

            v = json_get(sc_v, "thought_chain_depth");
            if (v && v->type == JSON_NUMBER) {
                config->self_cognition_thought_chain_depth = json_to_int_clamped(v->data.number_val);
            } else {
                config->self_cognition_thought_chain_depth = 5; /* 默认深度5 */
            }

            v = json_get(sc_v, "correction_max_iterations");
            if (v && v->type == JSON_NUMBER) {
                config->self_cognition_correction_max_iterations = json_to_int_clamped(v->data.number_val);
            } else {
                config->self_cognition_correction_max_iterations = 3; /* 默认3次 */
            }

            log_info("[配置] self_cognition段解析: enabled=%d, reflection_interval=%d",
                     config->self_cognition_enabled, config->self_cognition_reflection_interval_min);
        } else {
            /* 设置默认值 */
            config->self_cognition_enabled = 1;
            config->self_cognition_metacognition_enabled = 1;
            config->self_cognition_reflection_interval_min = 30;
            config->self_cognition_thought_chain_depth = 5;
            config->self_cognition_correction_max_iterations = 3;
        }
    }

    /* P1-04修复: 解析 system 嵌套段 */
    {
        const JsonValue* sys_v = json_get(root, "system");
        if (sys_v && sys_v->type == JSON_OBJECT) {
            const JsonValue* v = json_get(sys_v, "auto_restart_on_failure");
            if (v && (v->type == JSON_NUMBER)) {
                config->system_auto_restart_on_failure = (v->data.number_val != 0) ? 1 : 0;
            } else {
                config->system_auto_restart_on_failure = 0; /* 默认关闭 */
            }

            v = json_get(sys_v, "auto_save_interval_minutes");
            if (v && v->type == JSON_NUMBER) {
                config->system_auto_save_interval_minutes = json_to_int_clamped(v->data.number_val);
            } else {
                config->system_auto_save_interval_minutes = 60; /* 默认60分钟 */
            }

            const char* ll_str = json_get_string(sys_v, "log_level");
            if (ll_str && ll_str[0]) {
                strncpy(config->system_log_level, ll_str, sizeof(config->system_log_level) - 1);
                config->system_log_level[sizeof(config->system_log_level) - 1] = '\0';
            } else {
                strncpy(config->system_log_level, "info", sizeof(config->system_log_level) - 1);
                config->system_log_level[sizeof(config->system_log_level) - 1] = '\0';
            }

            v = json_get(sys_v, "web_ui_enabled");
            if (v && (v->type == JSON_NUMBER)) {
                config->system_web_ui_enabled = (v->data.number_val != 0) ? 1 : 0;
            } else {
                config->system_web_ui_enabled = 1; /* 默认启用 */
            }

            log_info("[配置] system段解析: auto_restart=%d, web_ui=%d",
                     config->system_auto_restart_on_failure, config->system_web_ui_enabled);
        } else {
            /* 设置默认值 */
            config->system_auto_restart_on_failure = 0;
            config->system_auto_save_interval_minutes = 60;
            strncpy(config->system_log_level, "info", sizeof(config->system_log_level) - 1);
            config->system_log_level[sizeof(config->system_log_level) - 1] = '\0';
            config->system_web_ui_enabled = 1;
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
    FILE* fp = fopen(path, "wb");
    if (!fp) { log_error("[配置] 无法创建: %s", path); return -1; }

    const char* pm = "balanced";
    if (config->power_mode == POWER_MODE_PERFORMANCE) pm = "performance";
    else if (config->power_mode == POWER_MODE_POWER_SAVING) pm = "power_saving";

/* GPU后端枚举补全 — 原只枚举cuda/opencl */
    const char* gb = "cpu";
    switch (config->gpu_backend) {
        case GPU_BACKEND_AUTO:     gb = "auto"; break;
        case GPU_BACKEND_CUDA:     gb = "cuda"; break;
        case GPU_BACKEND_OPENCL:   gb = "opencl"; break;
        case GPU_BACKEND_VULKAN:   gb = "vulkan"; break;
        case GPU_BACKEND_METAL:    gb = "metal"; break;
        case GPU_BACKEND_ROCM:     gb = "rocm"; break;
        case GPU_BACKEND_ASCEND:   gb = "ascend"; break;
        case GPU_BACKEND_CAMBRICON:gb = "cambricon"; break;
        case GPU_BACKEND_TPU:      gb = "tpu"; break;
        case GPU_BACKEND_INTEL:    gb = "intel"; break;
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
        fprintf(fp, "  \"model_path\": \"%s\",\n", config->model_path);
    } else {
        fprintf(fp, "  \"model_path\": \"\",\n");
    }
    /* 【A-001修复】补全 mixed_precision_mode 字段保存 */
    fprintf(fp, "  \"mixed_precision_mode\": %d\n", config->mixed_precision_mode);
    fprintf(fp, "}\n");

    fclose(fp);
    log_info("[配置] 保存成功: %s", path);
    return 0;
}
