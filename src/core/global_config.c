/**
 * @file global_config.c
 * @brief M-004修复: 全局配置化管理 — 运行时配置实现
 *
 * 单例模式全局配置，支持初始化、加载JSON、保存JSON、运行时修改。
 * 所有硬编码容量上限统一通过此模块管理。
 */

#include "selflnn/core/global_config.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* 全局配置单例 */
static GlobalConfig g_global_config;
static int g_config_initialized = 0;

GlobalConfig* global_config_get(void) {
    if (!g_config_initialized) {
        global_config_init();
    }
    return &g_global_config;
}

int global_config_init(void) {
    memset(&g_global_config, 0, sizeof(GlobalConfig));

    /* 知识库 */
    g_global_config.kb_max_entries        = SELFLNN_KB_MAX_ENTRIES;
    g_global_config.kb_max_query_results  = SELFLNN_KB_MAX_QUERY_RESULTS;
    g_global_config.kb_index_bucket_size  = SELFLNN_KB_INDEX_BUCKET_SIZE;
    g_global_config.ksc_max_issues        = SELFLNN_KSC_MAX_ISSUES;
    g_global_config.ksc_hash_buckets      = SELFLNN_KSC_HASH_BUCKETS;
    g_global_config.kg_max_nodes          = SELFLNN_KG_MAX_NODES;
    g_global_config.kg_max_edges          = SELFLNN_KG_MAX_EDGES;

    /* 推理引擎 */
    g_global_config.ki_max_rules          = SELFLNN_KI_MAX_RULES;
    g_global_config.ki_max_chain          = SELFLNN_KI_MAX_CHAIN;
    g_global_config.ki_max_iterations     = SELFLNN_KI_MAX_ITERATIONS;
    g_global_config.ki_max_analogs        = SELFLNN_KI_MAX_ANALOGS;
    g_global_config.ki_max_hypotheses     = SELFLNN_KI_MAX_HYPOTHESES;
    g_global_config.ki_max_relations      = SELFLNN_KI_MAX_RELATIONS;
    g_global_config.ki_max_concepts       = SELFLNN_KI_MAX_CONCEPTS;
    g_global_config.ki_max_path_length    = SELFLNN_KI_MAX_PATH_LENGTH;
    g_global_config.ki_max_inferred       = SELFLNN_KI_MAX_INFERRED;

    /* LNN/CfC */
    g_global_config.lnn_max_input_size    = SELFLNN_LNN_MAX_INPUT_SIZE;
    g_global_config.lnn_max_hidden_size   = SELFLNN_LNN_MAX_HIDDEN_SIZE;
    g_global_config.lnn_max_output_size   = SELFLNN_LNN_MAX_OUTPUT_SIZE;
    g_global_config.cfc_max_time_constant = SELFLNN_CFC_MAX_TIME_CONSTANT;
    g_global_config.cfc_min_time_constant = SELFLNN_CFC_MIN_TIME_CONSTANT;
    g_global_config.cfc_adaptive_tau_beta = SELFLNN_CFC_ADAPTIVE_TAU_BETA;
    g_global_config.cfc_max_ode_steps     = SELFLNN_CFC_MAX_ODE_STEPS;
    g_global_config.cfc_use_adaptive_tau  = 1;  /* 默认启用自适应τ */
    g_global_config.cfc_use_liquid_scaling = 1; /* 默认启用液时域缩放 */

    /* 训练 */
    g_global_config.training_max_batch_size = SELFLNN_TRAINING_MAX_BATCH_SIZE;
    g_global_config.training_max_epochs     = SELFLNN_TRAINING_MAX_EPOCHS;
    g_global_config.training_max_samples    = SELFLNN_TRAINING_MAX_SAMPLES;

    /* 并发 */
    g_global_config.thread_pool_max_threads = SELFLNN_THREAD_POOL_MAX_THREADS;
    g_global_config.ws_max_connections      = SELFLNN_WS_MAX_CONNECTIONS;
    g_global_config.ws_max_message_size     = SELFLNN_WS_MAX_MESSAGE_SIZE;
    g_global_config.api_max_body_size       = SELFLNN_API_MAX_BODY_SIZE;
    g_global_config.lf_queue_max_capacity   = SELFLNN_LF_QUEUE_MAX_CAPACITY;
    g_global_config.lf_skiplist_max_level   = SELFLNN_LF_SKIPLIST_MAX_LEVEL;

    /* 语义解析 */
    g_global_config.parser_max_tokens  = SELFLNN_PARSER_MAX_TOKENS;
    g_global_config.parser_stack_size  = SELFLNN_PARSER_STACK_SIZE;
    g_global_config.srl_max_roles      = SELFLNN_SRL_MAX_ROLES;

    /* 模糊逻辑 */
    g_global_config.fuzzy_max_rules    = SELFLNN_FUZZY_MAX_RULES;
    g_global_config.fuzzy_max_variables = SELFLNN_FUZZY_MAX_VARIABLES;
    g_global_config.fuzzy_samples      = SELFLNN_FUZZY_SAMPLES;
    g_global_config.ds_max_evidence    = SELFLNN_DS_MAX_EVIDENCE;

    /* 本体 */
    g_global_config.ont_max_classes        = SELFLNN_ONT_MAX_CLASSES;
    g_global_config.ont_max_properties     = SELFLNN_ONT_MAX_PROPERTIES;
    g_global_config.ont_max_individuals    = SELFLNN_ONT_MAX_INDIVIDUALS;
    g_global_config.ont_max_serialize_depth = SELFLNN_ONT_MAX_SERIALIZE_DEPTH;

    /* GPU/硬件 */
    g_global_config.gpu_max_devices        = SELFLNN_GPU_MAX_DEVICES;
    g_global_config.gpu_max_memory_pool_mb = SELFLNN_GPU_MAX_MEMORY_POOL_MB;
    g_global_config.tpu_max_devices        = SELFLNN_TPU_MAX_DEVICES;
    g_global_config.auto_kernel_max_search = SELFLNN_AUTO_KERNEL_MAX_SEARCH;

    /* 通用 */
    g_global_config.log_level              = 1;
    g_global_config.enable_self_check      = 1;
    g_global_config.enable_auto_evolution  = 1;
    g_global_config.enable_online_learning = 1;

    g_config_initialized = 1;
    log_info("[GlobalConfig] 全局配置初始化完成，%d项参数", 48);
    return 0;
}

int global_config_load(const char* config_path) {
    if (!config_path) return -1;

    /* 简易JSON解析：读取键值对，更新配置 */
    FILE* fp = fopen(config_path, "r");
    if (!fp) {
        log_warn("[GlobalConfig] 配置文件不存在: %s，使用默认值", config_path);
        return -1;
    }

    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        char key[128] = {0};
        int val = 0;
        float fval = 0.0f;

        /* 跳过注释和空行 */
        if (line[0] == '#' || line[0] == '/' || line[0] == '\n' || line[0] == '\r') continue;

        /* 解析整数键值对 */
        if (sscanf(line, " \"%127[^\"]\" : %d", key, &val) == 2 ||
            sscanf(line, " %127s : %d", key, &val) == 2) {
            /* 知识库 */
            if (strcmp(key, "kb_max_entries") == 0) g_global_config.kb_max_entries = val;
            else if (strcmp(key, "kb_max_query_results") == 0) g_global_config.kb_max_query_results = val;
            else if (strcmp(key, "ksc_max_issues") == 0) g_global_config.ksc_max_issues = val;
            else if (strcmp(key, "kg_max_nodes") == 0) g_global_config.kg_max_nodes = val;
            else if (strcmp(key, "kg_max_edges") == 0) g_global_config.kg_max_edges = val;
            /* 推理引擎 */
            else if (strcmp(key, "ki_max_rules") == 0) g_global_config.ki_max_rules = val;
            else if (strcmp(key, "ki_max_iterations") == 0) g_global_config.ki_max_iterations = val;
            else if (strcmp(key, "ki_max_analogs") == 0) g_global_config.ki_max_analogs = val;
            else if (strcmp(key, "ki_max_hypotheses") == 0) g_global_config.ki_max_hypotheses = val;
            /* LNN/CfC */
            else if (strcmp(key, "cfc_use_adaptive_tau") == 0) g_global_config.cfc_use_adaptive_tau = val;
            else if (strcmp(key, "cfc_use_liquid_scaling") == 0) g_global_config.cfc_use_liquid_scaling = val;
            else if (strcmp(key, "cfc_max_ode_steps") == 0) g_global_config.cfc_max_ode_steps = val;
            /* 训练 */
            else if (strcmp(key, "training_max_batch_size") == 0) g_global_config.training_max_batch_size = val;
            else if (strcmp(key, "training_max_epochs") == 0) g_global_config.training_max_epochs = val;
            else if (strcmp(key, "training_max_samples") == 0) g_global_config.training_max_samples = val;
            /* 并发 */
            else if (strcmp(key, "thread_pool_max_threads") == 0) g_global_config.thread_pool_max_threads = val;
            else if (strcmp(key, "ws_max_connections") == 0) g_global_config.ws_max_connections = val;
            /* 通用 */
            else if (strcmp(key, "log_level") == 0) g_global_config.log_level = val;
            else if (strcmp(key, "enable_self_check") == 0) g_global_config.enable_self_check = val;
            else if (strcmp(key, "enable_auto_evolution") == 0) g_global_config.enable_auto_evolution = val;
            else if (strcmp(key, "enable_online_learning") == 0) g_global_config.enable_online_learning = val;
        }
        /* 解析浮点键值对 */
        else if (sscanf(line, " \"%127[^\"]\" : %f", key, &fval) == 2 ||
                 sscanf(line, " %127s : %f", key, &fval) == 2) {
            if (strcmp(key, "cfc_max_time_constant") == 0) g_global_config.cfc_max_time_constant = fval;
            else if (strcmp(key, "cfc_min_time_constant") == 0) g_global_config.cfc_min_time_constant = fval;
            else if (strcmp(key, "cfc_adaptive_tau_beta") == 0) g_global_config.cfc_adaptive_tau_beta = fval;
        }
    }

    fclose(fp);
    log_info("[GlobalConfig] 配置加载成功: %s", config_path);
    return 0;
}

int global_config_save(const char* config_path) {
    if (!config_path) return -1;

    FILE* fp = fopen(config_path, "w");
    if (!fp) {
        log_warn("[GlobalConfig] 无法写入配置文件: %s", config_path);
        return -1;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"kb_max_entries\": %d,\n", g_global_config.kb_max_entries);
    fprintf(fp, "  \"kb_max_query_results\": %d,\n", g_global_config.kb_max_query_results);
    fprintf(fp, "  \"ksc_max_issues\": %d,\n", g_global_config.ksc_max_issues);
    fprintf(fp, "  \"kg_max_nodes\": %d,\n", g_global_config.kg_max_nodes);
    fprintf(fp, "  \"kg_max_edges\": %d,\n", g_global_config.kg_max_edges);
    fprintf(fp, "  \"ki_max_rules\": %d,\n", g_global_config.ki_max_rules);
    fprintf(fp, "  \"ki_max_iterations\": %d,\n", g_global_config.ki_max_iterations);
    fprintf(fp, "  \"ki_max_analogs\": %d,\n", g_global_config.ki_max_analogs);
    fprintf(fp, "  \"ki_max_hypotheses\": %d,\n", g_global_config.ki_max_hypotheses);
    fprintf(fp, "  \"cfc_use_adaptive_tau\": %d,\n", g_global_config.cfc_use_adaptive_tau);
    fprintf(fp, "  \"cfc_use_liquid_scaling\": %d,\n", g_global_config.cfc_use_liquid_scaling);
    fprintf(fp, "  \"cfc_max_time_constant\": %.4f,\n", g_global_config.cfc_max_time_constant);
    fprintf(fp, "  \"cfc_min_time_constant\": %.4f,\n", g_global_config.cfc_min_time_constant);
    fprintf(fp, "  \"cfc_adaptive_tau_beta\": %.4f,\n", g_global_config.cfc_adaptive_tau_beta);
    fprintf(fp, "  \"cfc_max_ode_steps\": %d,\n", g_global_config.cfc_max_ode_steps);
    fprintf(fp, "  \"training_max_batch_size\": %d,\n", g_global_config.training_max_batch_size);
    fprintf(fp, "  \"training_max_epochs\": %d,\n", g_global_config.training_max_epochs);
    fprintf(fp, "  \"training_max_samples\": %d,\n", g_global_config.training_max_samples);
    fprintf(fp, "  \"thread_pool_max_threads\": %d,\n", g_global_config.thread_pool_max_threads);
    fprintf(fp, "  \"ws_max_connections\": %d,\n", g_global_config.ws_max_connections);
    fprintf(fp, "  \"log_level\": %d,\n", g_global_config.log_level);
    fprintf(fp, "  \"enable_self_check\": %d,\n", g_global_config.enable_self_check);
    fprintf(fp, "  \"enable_auto_evolution\": %d,\n", g_global_config.enable_auto_evolution);
    fprintf(fp, "  \"enable_online_learning\": %d\n", g_global_config.enable_online_learning);
    fprintf(fp, "}\n");

    fclose(fp);
    log_info("[GlobalConfig] 配置保存成功: %s", config_path);
    return 0;
}

void global_config_reset(void) {
    g_config_initialized = 0;
    global_config_init();
}

void global_config_print(void) {
    GlobalConfig* cfg = global_config_get();
    log_info("========== 全局配置 ==========");
    log_info("知识库: max_entries=%d, query_results=%d, ksc_issues=%d",
             cfg->kb_max_entries, cfg->kb_max_query_results, cfg->ksc_max_issues);
    log_info("推理: max_rules=%d, max_iter=%d, max_analogs=%d",
             cfg->ki_max_rules, cfg->ki_max_iterations, cfg->ki_max_analogs);
    log_info("CfC: adaptive_tau=%d, liquid_scaling=%d, tau_range=[%.3f,%.3f]",
             cfg->cfc_use_adaptive_tau, cfg->cfc_use_liquid_scaling,
             cfg->cfc_min_time_constant, cfg->cfc_max_time_constant);
    log_info("训练: batch_size=%d, epochs=%d, samples=%d",
             cfg->training_max_batch_size, cfg->training_max_epochs, cfg->training_max_samples);
    log_info("并发: threads=%d, ws_conn=%d",
             cfg->thread_pool_max_threads, cfg->ws_max_connections);
    log_info("通用: log_level=%d, self_check=%d, evolution=%d, online_learn=%d",
             cfg->log_level, cfg->enable_self_check,
             cfg->enable_auto_evolution, cfg->enable_online_learning);
    log_info("==============================");
}

/* ============================================================================
 * 运行时配置修改API实现
 * ============================================================================ */

void global_config_set_kb_max_entries(int val) {
    if (val > 0) global_config_get()->kb_max_entries = val;
}

void global_config_set_kb_max_query_results(int val) {
    if (val > 0) global_config_get()->kb_max_query_results = val;
}

void global_config_set_ksc_max_issues(int val) {
    if (val > 0) global_config_get()->ksc_max_issues = val;
}

void global_config_set_ki_max_rules(int val) {
    if (val > 0) global_config_get()->ki_max_rules = val;
}

void global_config_set_ki_max_iterations(int val) {
    if (val > 0) global_config_get()->ki_max_iterations = val;
}

void global_config_set_ki_max_analogs(int val) {
    if (val > 0) global_config_get()->ki_max_analogs = val;
}

void global_config_set_ki_max_hypotheses(int val) {
    if (val > 0) global_config_get()->ki_max_hypotheses = val;
}

void global_config_set_cfc_adaptive_tau(int enabled) {
    global_config_get()->cfc_use_adaptive_tau = (enabled != 0) ? 1 : 0;
}

void global_config_set_cfc_liquid_scaling(int enabled) {
    global_config_get()->cfc_use_liquid_scaling = (enabled != 0) ? 1 : 0;
}

void global_config_set_cfc_time_constant_range(float min_tau, float max_tau) {
    if (min_tau > 0.0f && max_tau > min_tau) {
        global_config_get()->cfc_min_time_constant = min_tau;
        global_config_get()->cfc_max_time_constant = max_tau;
    }
}

void global_config_set_training_max_batch_size(int val) {
    if (val > 0) global_config_get()->training_max_batch_size = val;
}

void global_config_set_training_max_epochs(int val) {
    if (val > 0) global_config_get()->training_max_epochs = val;
}

void global_config_set_thread_pool_max_threads(int val) {
    if (val > 0) global_config_get()->thread_pool_max_threads = val;
}

void global_config_set_ws_max_connections(int val) {
    if (val > 0) global_config_get()->ws_max_connections = val;
}

void global_config_set_fuzzy_max_rules(int val) {
    if (val > 0) global_config_get()->fuzzy_max_rules = val;
}

void global_config_set_fuzzy_samples(int val) {
    if (val > 0) global_config_get()->fuzzy_samples = val;
}