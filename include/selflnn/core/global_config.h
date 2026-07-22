/**
 * @file global_config.h
 * @brief M-004修复: 全局配置化管理 — 统一管理所有硬编码容量上限
 *
 * 将所有模块中分散的硬编码容量限制集中到单一配置头文件，
 * 支持运行时动态调整，避免代码中散布"魔法数字"。
 *
 * 使用方式：
 *   1. 编译时宏定义覆盖（如 -DSELFLNN_KB_MAX_ENTRIES=16384）
 *   2. 运行时通过 global_config_init() 加载配置文件
 *   3. 运行时通过 global_config_set_*() 系列函数动态修改
 */

#ifndef SELFLNN_GLOBAL_CONFIG_H
#define SELFLNN_GLOBAL_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 一、知识库容量配置
 * ============================================================================ */

/** 知识库最大条目数（默认4096，可通过宏覆盖） */
#ifndef SELFLNN_KB_MAX_ENTRIES
#define SELFLNN_KB_MAX_ENTRIES          8192
#endif

/** 知识库单次查询最大结果数 */
#ifndef SELFLNN_KB_MAX_QUERY_RESULTS
#define SELFLNN_KB_MAX_QUERY_RESULTS    512
#endif

/** 知识库倒排索引桶大小 */
#ifndef SELFLNN_KB_INDEX_BUCKET_SIZE
#define SELFLNN_KB_INDEX_BUCKET_SIZE    128
#endif

/** 知识自检最大问题数 */
#ifndef SELFLNN_KSC_MAX_ISSUES
#define SELFLNN_KSC_MAX_ISSUES          256
#endif

/** 知识自检哈希桶数 */
#ifndef SELFLNN_KSC_HASH_BUCKETS
#define SELFLNN_KSC_HASH_BUCKETS        128
#endif

/** 知识图谱最大节点数 */
#ifndef SELFLNN_KG_MAX_NODES
#define SELFLNN_KG_MAX_NODES            16384
#endif

/** 知识图谱最大边数 */
#ifndef SELFLNN_KG_MAX_EDGES
#define SELFLNN_KG_MAX_EDGES            65536
#endif

/* ============================================================================
 * 二、推理引擎容量配置
 * ============================================================================ */

/** 推理引擎最大规则数 */
#ifndef SELFLNN_KI_MAX_RULES
#define SELFLNN_KI_MAX_RULES            256
#endif

/** 推理引擎最大推理链长度 */
#ifndef SELFLNN_KI_MAX_CHAIN
#define SELFLNN_KI_MAX_CHAIN            64
#endif

/** 推理引擎最大迭代次数 */
#ifndef SELFLNN_KI_MAX_ITERATIONS
#define SELFLNN_KI_MAX_ITERATIONS       20
#endif

/** 推理引擎最大类比数 */
#ifndef SELFLNN_KI_MAX_ANALOGS
#define SELFLNN_KI_MAX_ANALOGS          16
#endif

/** 推理引擎最大假设数 */
#ifndef SELFLNN_KI_MAX_HYPOTHESES
#define SELFLNN_KI_MAX_HYPOTHESES       128
#endif

/** 推理引擎最大关系三元组 */
#ifndef SELFLNN_KI_MAX_RELATIONS
#define SELFLNN_KI_MAX_RELATIONS        512
#endif

/** 推理引擎最大概念数 */
#ifndef SELFLNN_KI_MAX_CONCEPTS
#define SELFLNN_KI_MAX_CONCEPTS         128
#endif

/** 推理引擎最大路径长度 */
#ifndef SELFLNN_KI_MAX_PATH_LENGTH
#define SELFLNN_KI_MAX_PATH_LENGTH      128
#endif

/** 推理引擎前向链接最大推断结果 */
#ifndef SELFLNN_KI_MAX_INFERRED
#define SELFLNN_KI_MAX_INFERRED         256
#endif

/* ============================================================================
 * 三、LNN/CfC神经网络配置
 * ============================================================================ */

/** LNN最大输入维度 */
#ifndef SELFLNN_LNN_MAX_INPUT_SIZE
#define SELFLNN_LNN_MAX_INPUT_SIZE      2048
#endif

/** LNN最大隐藏层维度 */
#ifndef SELFLNN_LNN_MAX_HIDDEN_SIZE
#define SELFLNN_LNN_MAX_HIDDEN_SIZE     4096
#endif

/** LNN最大输出维度 */
#ifndef SELFLNN_LNN_MAX_OUTPUT_SIZE
#define SELFLNN_LNN_MAX_OUTPUT_SIZE     2048
#endif

/** CfC最大时间常数 */
#ifndef SELFLNN_CFC_MAX_TIME_CONSTANT
#define SELFLNN_CFC_MAX_TIME_CONSTANT   10.0f
#endif

/** CfC最小时间常数 */
#ifndef SELFLNN_CFC_MIN_TIME_CONSTANT
#define SELFLNN_CFC_MIN_TIME_CONSTANT   0.01f
#endif

/** CfC自适应τ衰减系数 */
#ifndef SELFLNN_CFC_ADAPTIVE_TAU_BETA
#define SELFLNN_CFC_ADAPTIVE_TAU_BETA   2.0f
#endif

/** CfC ODE求解器最大步数 */
#ifndef SELFLNN_CFC_MAX_ODE_STEPS
#define SELFLNN_CFC_MAX_ODE_STEPS       1000
#endif

/** 训练批次最大大小 */
#ifndef SELFLNN_TRAINING_MAX_BATCH_SIZE
#define SELFLNN_TRAINING_MAX_BATCH_SIZE 1024
#endif

/** 训练最大轮数 */
#ifndef SELFLNN_TRAINING_MAX_EPOCHS
#define SELFLNN_TRAINING_MAX_EPOCHS     10000
#endif

/* ============================================================================
 * 四、并发与通信配置
 * ============================================================================ */

/** 线程池最大线程数 */
#ifndef SELFLNN_THREAD_POOL_MAX_THREADS
#define SELFLNN_THREAD_POOL_MAX_THREADS 256
#endif

/** WebSocket最大连接数 */
#ifndef SELFLNN_WS_MAX_CONNECTIONS
#define SELFLNN_WS_MAX_CONNECTIONS      1024
#endif

/** WebSocket最大消息大小 */
#ifndef SELFLNN_WS_MAX_MESSAGE_SIZE
#define SELFLNN_WS_MAX_MESSAGE_SIZE     65536
#endif

/** HTTP API最大请求体 */
#ifndef SELFLNN_API_MAX_BODY_SIZE
#define SELFLNN_API_MAX_BODY_SIZE       (16 * 1024 * 1024)  /* 16MB */
#endif

/** 无锁队列最大容量 */
#ifndef SELFLNN_LF_QUEUE_MAX_CAPACITY
#define SELFLNN_LF_QUEUE_MAX_CAPACITY   65536
#endif

/** 无锁跳表最大层级 */
#ifndef SELFLNN_LF_SKIPLIST_MAX_LEVEL
#define SELFLNN_LF_SKIPLIST_MAX_LEVEL   32
#endif

/* ============================================================================
 * 五、语义解析与NLP配置
 * ============================================================================ */

/** 依存分析器最大词元数 */
#ifndef SELFLNN_PARSER_MAX_TOKENS
#define SELFLNN_PARSER_MAX_TOKENS       2048
#endif

/** 依存分析器栈大小 */
#ifndef SELFLNN_PARSER_STACK_SIZE
#define SELFLNN_PARSER_STACK_SIZE       4096
#endif

/** 训练数据管线最大样本数 */
#ifndef SELFLNN_TRAINING_MAX_SAMPLES
#define SELFLNN_TRAINING_MAX_SAMPLES    100000
#endif

/** 语义角色标注最大角色数 */
#ifndef SELFLNN_SRL_MAX_ROLES
#define SELFLNN_SRL_MAX_ROLES           32
#endif

/* ============================================================================
 * 六、模糊逻辑与不确定性推理配置
 * ============================================================================ */

/** 模糊规则最大数量 */
#ifndef SELFLNN_FUZZY_MAX_RULES
#define SELFLNN_FUZZY_MAX_RULES         2048
#endif

/** 模糊变量最大数量 */
#ifndef SELFLNN_FUZZY_MAX_VARIABLES
#define SELFLNN_FUZZY_MAX_VARIABLES     64
#endif

/** 模糊隶属度采样点 */
#ifndef SELFLNN_FUZZY_SAMPLES
#define SELFLNN_FUZZY_SAMPLES           512
#endif

/** Dempster-Shafer最大证据集 */
#ifndef SELFLNN_DS_MAX_EVIDENCE
#define SELFLNN_DS_MAX_EVIDENCE         128
#endif

/* ============================================================================
 * 七、本体工程配置
 * ============================================================================ */

/** 本体最大类数 */
#ifndef SELFLNN_ONT_MAX_CLASSES
#define SELFLNN_ONT_MAX_CLASSES         1024
#endif

/** 本体最大属性数 */
#ifndef SELFLNN_ONT_MAX_PROPERTIES
#define SELFLNN_ONT_MAX_PROPERTIES      512
#endif

/** 本体最大实例数 */
#ifndef SELFLNN_ONT_MAX_INDIVIDUALS
#define SELFLNN_ONT_MAX_INDIVIDUALS     4096
#endif

/** OWL/RDF序列化最大深度 */
#ifndef SELFLNN_ONT_MAX_SERIALIZE_DEPTH
#define SELFLNN_ONT_MAX_SERIALIZE_DEPTH 32
#endif

/* ============================================================================
 * 八、硬件与GPU配置
 * ============================================================================ */

/** GPU最大设备数 */
#ifndef SELFLNN_GPU_MAX_DEVICES
#define SELFLNN_GPU_MAX_DEVICES         64
#endif

/** GPU内存池最大大小（MB） */
#ifndef SELFLNN_GPU_MAX_MEMORY_POOL_MB
#define SELFLNN_GPU_MAX_MEMORY_POOL_MB  8192
#endif

/** TPU最大设备数 */
#ifndef SELFLNN_TPU_MAX_DEVICES
#define SELFLNN_TPU_MAX_DEVICES         64
#endif

/** 自动内核优化最大搜索次数 */
#ifndef SELFLNN_AUTO_KERNEL_MAX_SEARCH
#define SELFLNN_AUTO_KERNEL_MAX_SEARCH  1000
#endif

/* ============================================================================
 * 九、运行时配置结构体
 * ============================================================================ */

/**
 * @brief 全局运行时配置
 *
 * 所有可动态调整的配置参数集中在此结构体中。
 * 初始化时使用上述宏定义的默认值，运行时可通过API修改。
 */
typedef struct {
    /* 知识库 */
    int kb_max_entries;
    int kb_max_query_results;
    int kb_index_bucket_size;
    int ksc_max_issues;
    int ksc_hash_buckets;
    int kg_max_nodes;
    int kg_max_edges;

    /* 推理引擎 */
    int ki_max_rules;
    int ki_max_chain;
    int ki_max_iterations;
    int ki_max_analogs;
    int ki_max_hypotheses;
    int ki_max_relations;
    int ki_max_concepts;
    int ki_max_path_length;
    int ki_max_inferred;

    /* LNN/CfC */
    int lnn_max_input_size;
    int lnn_max_hidden_size;
    int lnn_max_output_size;
    float cfc_max_time_constant;
    float cfc_min_time_constant;
    float cfc_adaptive_tau_beta;
    int cfc_max_ode_steps;
    int cfc_use_adaptive_tau;           /**< 全局自适应τ开关 */
    int cfc_use_liquid_scaling;         /**< 全局液时域缩放开关 */

    /* 训练 */
    int training_max_batch_size;
    int training_max_epochs;
    int training_max_samples;

    /* 并发 */
    int thread_pool_max_threads;
    int ws_max_connections;
    int ws_max_message_size;
    int api_max_body_size;
    int lf_queue_max_capacity;
    int lf_skiplist_max_level;

    /* 语义解析 */
    int parser_max_tokens;
    int parser_stack_size;
    int srl_max_roles;

    /* 模糊逻辑 */
    int fuzzy_max_rules;
    int fuzzy_max_variables;
    int fuzzy_samples;
    int ds_max_evidence;

    /* 本体 */
    int ont_max_classes;
    int ont_max_properties;
    int ont_max_individuals;
    int ont_max_serialize_depth;

    /* GPU/硬件 */
    int gpu_max_devices;
    int gpu_max_memory_pool_mb;
    int tpu_max_devices;
    int auto_kernel_max_search;

    /* 通用 */
    int log_level;                       /**< 日志级别 */
    int enable_self_check;               /**< 是否启用自检 */
    int enable_auto_evolution;           /**< 是否启用自动演化 */
    int enable_online_learning;          /**< 是否启用在线学习 */
} GlobalConfig;

/**
 * @brief 获取全局配置实例（单例模式）
 * @return GlobalConfig* 全局配置指针
 */
GlobalConfig* global_config_get(void);

/**
 * @brief 初始化全局配置为默认值
 * @return int 成功返回0
 */
int global_config_init(void);

/**
 * @brief 从JSON文件加载配置
 * @param config_path 配置文件路径
 * @return int 成功返回0
 */
int global_config_load(const char* config_path);

/**
 * @brief 保存当前配置到JSON文件
 * @param config_path 配置文件路径
 * @return int 成功返回0
 */
int global_config_save(const char* config_path);

/**
 * @brief 重置全局配置为默认值
 */
void global_config_reset(void);

/**
 * @brief 打印当前配置到日志
 */
void global_config_print(void);

/* ============================================================================
 * 十、运行时配置修改API
 * ============================================================================ */

/* 知识库配置 */
void global_config_set_kb_max_entries(int val);
void global_config_set_kb_max_query_results(int val);
void global_config_set_ksc_max_issues(int val);

/* 推理引擎配置 */
void global_config_set_ki_max_rules(int val);
void global_config_set_ki_max_iterations(int val);
void global_config_set_ki_max_analogs(int val);
void global_config_set_ki_max_hypotheses(int val);

/* LNN/CfC配置 */
void global_config_set_cfc_adaptive_tau(int enabled);
void global_config_set_cfc_liquid_scaling(int enabled);
void global_config_set_cfc_time_constant_range(float min_tau, float max_tau);

/* 训练配置 */
void global_config_set_training_max_batch_size(int val);
void global_config_set_training_max_epochs(int val);

/* 并发配置 */
void global_config_set_thread_pool_max_threads(int val);
void global_config_set_ws_max_connections(int val);

/* 模糊逻辑配置 */
void global_config_set_fuzzy_max_rules(int val);
void global_config_set_fuzzy_samples(int val);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_GLOBAL_CONFIG_H */